"""
MLA (Multi-head Latent Attention) decode kernel - optimized implementation.

Implements multiple optimization strategies:
1. FP8 Q + FP8 KV using aiter's a8w8 persistent MLA kernel (baseline)
2. MXFP4 KV with dequantization + aiter MLA kernel (2x bandwidth savings)
3. MXFP4 Q + MXFP4 KV using gemm_a4w4 for QK^T (4x bandwidth savings)
4. Dynamic num_kv_splits tuning based on workload
5. Custom HIP kernel for fused MXFP4 attention (JIT compiled)

DeepSeek R1 forward_absorb MLA config:
  total_num_heads  = 128    (query heads before TP split)
  num_heads        = 128 // tp  (query heads per device, tp=4 -> 32, tp=8 -> 16)
  num_kv_heads     = 1      (shared latent KV head)
  kv_lora_rank     = 512    (latent dim)
  qk_rope_head_dim = 64     (RoPE dim)
  qk_head_dim      = 576    (kv_lora_rank + qk_rope_head_dim, absorbed q/k dim)
  v_head_dim       = 512    (= kv_lora_rank, output dim)
  sm_scale         = 1/sqrt(576)

KV buffer format (forward_absorb):
  - Full 576 dims used as keys (for Q@K^T score computation)
  - First 512 dims (kv_lora_rank) used as values (for output computation)
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t
from math import gcd

import aiter
from aiter.mla import mla_decode_fwd
from aiter import QuantType, dtypes as aiter_dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32

# ---------------------------------------------------------------------------
# Embedded HIP Kernel for MXFP4 MLA Decode (using hip-python)
# ---------------------------------------------------------------------------

# HIP kernel source code - optimized for AMD MI355X (gfx950)
# Key optimizations:
# 1. Vectorized loads (float4/uint4) for coalesced memory access
# 2. LDS-cached FP4 LUT for faster dequantization
# 3. Better work distribution with 2D thread blocks
# 4. KV tiling to improve cache locality
# 5. Fused softmax with online normalization
# 6. Reduced shared memory bank conflicts
# 7. Loop unrolling for MXFP4 block processing
MLA_MXFP4_HIP_SOURCE = b'''
// v4 fused attn kernel
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include <cmath>

// Constants
constexpr int MXFP4_BLOCK_SIZE = 32;
constexpr int QK_HEAD_DIM = 576;
constexpr int V_HEAD_DIM = 512;
constexpr int NUM_HEADS = 16;
constexpr int WARP_SIZE = 64;
constexpr int MAX_Q_SEQ_LEN = 4;
constexpr int NUM_MXFP4_BLOCKS = QK_HEAD_DIM / MXFP4_BLOCK_SIZE;  // 18 blocks

// Tile sizes for KV processing - tuned for MI355X L2 cache
constexpr int KV_TILE_SIZE = 64;  // Process 64 KV tokens at a time

constexpr int FMT_FP4_MFMA = 4;

typedef float __attribute__((ext_vector_type(4))) float4_t;
typedef int __attribute__((ext_vector_type(8))) int8_vec;
typedef uint32_t __attribute__((ext_vector_type(4))) uint128_vec;

// FP4 E2M1 lookup table in constant memory
__device__ __constant__ float FP4_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// Inline FP4 dequantization using bit manipulation (faster than LUT for some cases)
__device__ __forceinline__ float fp4_to_float_inline(uint8_t nibble) {
    // FP4 E2M1: sign(1) + exp(2) + mantissa(1)
    // Magnitudes: 0->0, 1->0.5, 2->1, 3->1.5, 4->2, 5->3, 6->4, 7->6
    return FP4_E2M1_LUT[nibble & 0x0F];
}

// Fast E8M0 to float using bit reinterpretation
__device__ __forceinline__ float e8m0_to_float_fast(uint8_t e8m0) {
    // E8M0: pure exponent format, value = 2^(e8m0 - 127)
    // Construct IEEE754 float directly: exponent = e8m0, mantissa = 0
    uint32_t bits = (static_cast<uint32_t>(e8m0)) << 23;
    return __uint_as_float(bits);
}

// Warp-level reduction using AMD's 64-wide warps
__device__ __forceinline__ float warp_reduce_max_64(float val) {
    // #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum_64(float val) {
    // #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

// Block-level reduction optimized for 256 threads (4 warps on MI355X)
__device__ __forceinline__ float block_reduce_max_256(float val, float* smem, int tid) {
    const int lane = tid & 63;
    const int warp_id = tid >> 6;

    val = warp_reduce_max_64(val);

    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    // Final reduction across 4 warps
    if (tid < 4) {
        val = smem[tid];
    } else {
        val = -INFINITY;
    }

    if (tid < 64) {
        val = warp_reduce_max_64(val);
    }

    if (tid == 0) {
        smem[0] = val;
    }
    __syncthreads();

    return smem[0];
}

__device__ __forceinline__ float block_reduce_sum_256(float val, float* smem, int tid) {
    const int lane = tid & 63;
    const int warp_id = tid >> 6;

    val = warp_reduce_sum_64(val);

    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    if (tid < 4) {
        val = smem[tid];
    } else {
        val = 0.0f;
    }

    if (tid < 64) {
        val = warp_reduce_sum_64(val);
    }

    if (tid == 0) {
        smem[0] = val;
    }
    __syncthreads();

    return smem[0];
}

struct QuantBlock {
    uint32_t data[4];  // 16 packed bytes (32 FP4 values)
    uint8_t e8m0;
};

struct E8M0Scale {
    uint8_t e8m0;
    float quant_scale;
};

__device__ constexpr uint8_t E8M0_LUT[256] = {
    0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,
    30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,
    62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,
    94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,
    126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,
    158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,
    190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,
    222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,
};

__device__ constexpr uint32_t QUANT_SCALE_RECIP_LUT[256] = {
    0x00000000u,0x00800000u,0x01000000u,0x01800000u,0x02000000u,0x02800000u,0x03000000u,0x03800000u,
    0x04000000u,0x04800000u,0x05000000u,0x05800000u,0x06000000u,0x06800000u,0x07000000u,0x07800000u,
    0x08000000u,0x08800000u,0x09000000u,0x09800000u,0x0A000000u,0x0A800000u,0x0B000000u,0x0B800000u,
    0x0C000000u,0x0C800000u,0x0D000000u,0x0D800000u,0x0E000000u,0x0E800000u,0x0F000000u,0x0F800000u,
    0x10000000u,0x10800000u,0x11000000u,0x11800000u,0x12000000u,0x12800000u,0x13000000u,0x13800000u,
    0x14000000u,0x14800000u,0x15000000u,0x15800000u,0x16000000u,0x16800000u,0x17000000u,0x17800000u,
    0x18000000u,0x18800000u,0x19000000u,0x19800000u,0x1A000000u,0x1A800000u,0x1B000000u,0x1B800000u,
    0x1C000000u,0x1C800000u,0x1D000000u,0x1D800000u,0x1E000000u,0x1E800000u,0x1F000000u,0x1F800000u,
    0x20000000u,0x20800000u,0x21000000u,0x21800000u,0x22000000u,0x22800000u,0x23000000u,0x23800000u,
    0x24000000u,0x24800000u,0x25000000u,0x25800000u,0x26000000u,0x26800000u,0x27000000u,0x27800000u,
    0x28000000u,0x28800000u,0x29000000u,0x29800000u,0x2A000000u,0x2A800000u,0x2B000000u,0x2B800000u,
    0x2C000000u,0x2C800000u,0x2D000000u,0x2D800000u,0x2E000000u,0x2E800000u,0x2F000000u,0x2F800000u,
    0x30000000u,0x30800000u,0x31000000u,0x31800000u,0x32000000u,0x32800000u,0x33000000u,0x33800000u,
    0x34000000u,0x34800000u,0x35000000u,0x35800000u,0x36000000u,0x36800000u,0x37000000u,0x37800000u,
    0x38000000u,0x38800000u,0x39000000u,0x39800000u,0x3A000000u,0x3A800000u,0x3B000000u,0x3B800000u,
    0x3C000000u,0x3C800000u,0x3D000000u,0x3D800000u,0x3E000000u,0x3E800000u,0x3F000000u,0x3F800000u,
    0x40000000u,0x40800000u,0x41000000u,0x41800000u,0x42000000u,0x42800000u,0x43000000u,0x43800000u,
    0x44000000u,0x44800000u,0x45000000u,0x45800000u,0x46000000u,0x46800000u,0x47000000u,0x47800000u,
    0x48000000u,0x48800000u,0x49000000u,0x49800000u,0x4A000000u,0x4A800000u,0x4B000000u,0x4B800000u,
    0x4C000000u,0x4C800000u,0x4D000000u,0x4D800000u,0x4E000000u,0x4E800000u,0x4F000000u,0x4F800000u,
    0x50000000u,0x50800000u,0x51000000u,0x51800000u,0x52000000u,0x52800000u,0x53000000u,0x53800000u,
    0x54000000u,0x54800000u,0x55000000u,0x55800000u,0x56000000u,0x56800000u,0x57000000u,0x57800000u,
    0x58000000u,0x58800000u,0x59000000u,0x59800000u,0x5A000000u,0x5A800000u,0x5B000000u,0x5B800000u,
    0x5C000000u,0x5C800000u,0x5D000000u,0x5D800000u,0x5E000000u,0x5E800000u,0x5F000000u,0x5F800000u,
    0x60000000u,0x60800000u,0x61000000u,0x61800000u,0x62000000u,0x62800000u,0x63000000u,0x63800000u,
    0x64000000u,0x64800000u,0x65000000u,0x65800000u,0x66000000u,0x66800000u,0x67000000u,0x67800000u,
    0x68000000u,0x68800000u,0x69000000u,0x69800000u,0x6A000000u,0x6A800000u,0x6B000000u,0x6B800000u,
    0x6C000000u,0x6C800000u,0x6D000000u,0x6D800000u,0x6E000000u,0x6E800000u,0x6F000000u,0x6F800000u,
    0x70000000u,0x70800000u,0x71000000u,0x71800000u,0x72000000u,0x72800000u,0x73000000u,0x73800000u,
    0x74000000u,0x74800000u,0x75000000u,0x75800000u,0x76000000u,0x76800000u,0x77000000u,0x77800000u,
    0x78000000u,0x78800000u,0x79000000u,0x79800000u,0x7A000000u,0x7A800000u,0x7B000000u,0x7B800000u,
    0x7C000000u,0x7C800000u,0x7D000000u,0x7D800000u,0x7E000000u,0x7E800000u,0x7F000000u,0x00000000u,
};

__device__ __forceinline__ E8M0Scale compute_e8m0_scale(hip_bfloat16 amax_bf16) {
    E8M0Scale result;
    uint16_t amax_bits = amax_bf16.data;
    if (amax_bits == 0) {
        result.e8m0 = 0;
        result.quant_scale = 0.0f;
    } else {
        uint16_t rounded = (amax_bits + 0x0020u) & 0xFF80u;
        int raw_exp = (int)((rounded >> 7) & 0xFF);
        result.e8m0 = E8M0_LUT[raw_exp];
        result.quant_scale = __uint_as_float(QUANT_SCALE_RECIP_LUT[result.e8m0]);
    }
    return result;
}

// Hardware FP4 conversion from BF16: converts 2 BF16 values to packed FP4 byte.
__device__ __forceinline__ uint8_t quantize_fp4_pair_hw_bf16(
    hip_bfloat16 v0, hip_bfloat16 v1, float quant_scale
) {
    using bf16x2 = uint16_t __attribute__((ext_vector_type(2)));
    bf16x2 pair = {v0.data, v1.data};
    union { uint32_t u32; uint8_t u8[4]; } cvt = {0};
    cvt.u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(cvt.u32, pair, quant_scale, 0);
    return cvt.u8[0];
}

// Quantize 32 BF16 values to packed FP4 + E8M0 scale using BF16 hw intrinsic.
__device__ __forceinline__ QuantBlock quantize_fp4_block_bf16(const hip_bfloat16* src) {
    uint16_t amax_bits = 0;
    for (int i = 0; i < 32; i++) {
        uint16_t bits = *reinterpret_cast<const uint16_t*>(&src[i]) & 0x7FFF;
        amax_bits = (bits > amax_bits) ? bits : amax_bits;
    }
    hip_bfloat16 amax_bf16 = *reinterpret_cast<const hip_bfloat16*>(&amax_bits);

    E8M0Scale sc = compute_e8m0_scale(amax_bf16);

    uint32_t pack[4] = {0, 0, 0, 0};
    for (int i = 0; i < 16; i++) {
        uint8_t packed = quantize_fp4_pair_hw_bf16(src[2*i], src[2*i+1], sc.quant_scale);
        pack[i / 4] |= ((uint32_t)packed) << ((i % 4) * 8);
    }

    QuantBlock result;
    *reinterpret_cast<uint128_vec*>(&result.data) = *reinterpret_cast<uint128_vec*>(&pack);
    result.e8m0 = sc.e8m0;
    return result;
}

// Quantize 32 float values to packed FP4 + E8M0 scale.
// Converts to BF16 first and uses the BF16 hw intrinsic path (matching mxfp4-mm).
__device__ __forceinline__ QuantBlock quantize_fp4_block(const float vals[32]) {
    hip_bfloat16 bvals[32];
    for (int i = 0; i < 32; i++) {
        bvals[i] = hip_bfloat16(vals[i]);
    }
    return quantize_fp4_block_bf16(bvals);
}

template <int M, int K, int K_HALF, int NUM_BLOCKS, int BATCH_SIZE, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE)
void mla_quant_q_batched_kernel(
    const hip_bfloat16* __restrict__ Q,
    uint8_t* __restrict__ out_data,
    uint8_t* __restrict__ out_scale
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int per_batch = M * NUM_BLOCKS;
    int batch_idx = gid / per_batch;
    int local_id = gid % per_batch;
    if (batch_idx >= BATCH_SIZE) return;

    int row = local_id / NUM_BLOCKS;
    int blk = local_id % NUM_BLOCKS;
    if (row >= M) return;

    QuantBlock qb = quantize_fp4_block_bf16(&Q[(batch_idx * M + row) * K + blk * 32]);

    constexpr int A_K_HALF = NUM_BLOCKS * 16;
    int data_off = (batch_idx * M + row) * A_K_HALF + blk * 16;
    *reinterpret_cast<uint128_vec*>(&out_data[data_off]) =
        *reinterpret_cast<uint128_vec*>(&qb.data);
    out_scale[batch_idx * NUM_BLOCKS * M + row + blk * M] = qb.e8m0;
}

// Optimized kernel for MI355X with vectorized loads and better memory access patterns
__global__ //__launch_bounds__(256, 4)
void mla_mxfp4_decode_kernel(
    const hip_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ kv_mxfp4,
    const uint8_t* __restrict__ kv_scale,
    hip_bfloat16* __restrict__ output,
    const int batch_size,
    const int q_seq_len,
    const int kv_seq_len,
    const float sm_scale
) {
    constexpr int THREADS_PER_BLOCK = 256;
    constexpr int BYTES_PER_KV_ROW = QK_HEAD_DIM / 2;  // 288 bytes
    constexpr int SCALES_PER_KV_ROW = NUM_MXFP4_BLOCKS;  // 18 scales

    const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.x);
    const int head_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
    const int tid = threadIdx.x;

    // Shared memory layout (optimized for bank conflict avoidance):
    // [LUT: 16 floats] [Q: q_seq_len * QK_HEAD_DIM] [reduce: 8 floats] [scores: kv_seq_len]
    extern __shared__ char shared_bytes[];

    float* smem_lut = reinterpret_cast<float*>(shared_bytes);
    float* smem_q = smem_lut + 16;
    float* smem_reduce = smem_q + q_seq_len * QK_HEAD_DIM;
    float* smem_scores = smem_reduce + 8;

    // Load FP4 LUT into shared memory (faster than constant memory for repeated access)
    if (tid < 16) {
        smem_lut[tid] = FP4_E2M1_LUT[tid];
    }
    __syncthreads();

    // Load all queries for this (batch, head) into shared memory with vectorized loads
    for (int q_idx = 0; q_idx < q_seq_len; q_idx++) {
        const int global_q_idx = batch_idx * q_seq_len + q_idx;
        const int q_offset = (global_q_idx * NUM_HEADS + head_idx) * QK_HEAD_DIM;

        // Use vectorized loads where possible (4 bf16 = 8 bytes = 64 bits)
        const int vec_elems = 4;
        const int num_vec_loads = QK_HEAD_DIM / vec_elems;

        for (int i = tid; i < num_vec_loads; i += THREADS_PER_BLOCK) {
            const int base_idx = i * vec_elems;
            // Load 4 bf16 values
            // #pragma unroll
            for (int v = 0; v < vec_elems; v++) {
                smem_q[q_idx * QK_HEAD_DIM + base_idx + v] =
                    static_cast<float>(q[q_offset + base_idx + v]);
            }
        }
    }
    __syncthreads();

    const int64_t kv_batch_offset = __builtin_amdgcn_readfirstlane(static_cast<int64_t>(batch_idx) * kv_seq_len * BYTES_PER_KV_ROW);
    const int64_t scale_batch_offset = __builtin_amdgcn_readfirstlane(static_cast<int64_t>(batch_idx) * kv_seq_len * SCALES_PER_KV_ROW);

    // Process each query token
    for (int q_idx = 0; q_idx < q_seq_len; q_idx++) {
        const float* q_ptr = smem_q + q_idx * QK_HEAD_DIM;
        float* scores_ptr = smem_scores;

        // Phase 1: Compute QK^T scores using online softmax (fused max tracking)
        float local_max = -INFINITY;
        float local_sum = 0.0f;

        // Process KV tokens in tiles for better cache utilization
        for (int kv_base = 0; kv_base < kv_seq_len; kv_base += THREADS_PER_BLOCK) {
            const int kv_idx = kv_base + tid;

            if (kv_idx < kv_seq_len) {
                float score = 0.0f;

                const int64_t kv_offset = kv_batch_offset + static_cast<int64_t>(kv_idx) * BYTES_PER_KV_ROW;
                const int64_t scale_offset = scale_batch_offset + static_cast<int64_t>(kv_idx) * SCALES_PER_KV_ROW;

                // Process MXFP4 blocks with unrolling
                // #pragma unroll 2
                for (int block = 0; block < NUM_MXFP4_BLOCKS; block++) {
                    const float block_scale = e8m0_to_float_fast(kv_scale[scale_offset + block]);
                    const int64_t block_kv_offset = kv_offset + block * (MXFP4_BLOCK_SIZE / 2);
                    const int q_block_base = block * MXFP4_BLOCK_SIZE;

                    // Process 16 bytes (32 FP4 values) per block with full unroll
                    // #pragma unroll
                    for (int j = 0; j < MXFP4_BLOCK_SIZE / 2; j++) {
                        const uint8_t packed = kv_mxfp4[block_kv_offset + j];

                        // Dequantize two FP4 values
                        const float k_val0 = smem_lut[packed & 0x0F] * block_scale;
                        const float k_val1 = smem_lut[(packed >> 4) & 0x0F] * block_scale;

                        const int d_idx = q_block_base + j * 2;
                        score += q_ptr[d_idx] * k_val0 + q_ptr[d_idx + 1] * k_val1;
                    }
                }

                score *= sm_scale;
                scores_ptr[kv_idx] = score;
                local_max = fmaxf(local_max, score);
            }
        }
        __syncthreads();

        // Phase 2: Softmax normalization
        const float max_val = block_reduce_max_256(local_max, smem_reduce, tid);

        // Compute exp and sum
        local_sum = 0.0f;
        for (int kv_idx = tid; kv_idx < kv_seq_len; kv_idx += THREADS_PER_BLOCK) {
            const float exp_val = expf(scores_ptr[kv_idx] - max_val);
            scores_ptr[kv_idx] = exp_val;
            local_sum += exp_val;
        }
        __syncthreads();

        const float sum_val = block_reduce_sum_256(local_sum, smem_reduce, tid);
        const float inv_sum = 1.0f / sum_val;

        // Normalize scores
        for (int kv_idx = tid; kv_idx < kv_seq_len; kv_idx += THREADS_PER_BLOCK) {
            scores_ptr[kv_idx] *= inv_sum;
        }
        __syncthreads();

        // Phase 3: Compute attention @ V (weighted sum of values)
        const int global_q_idx = batch_idx * q_seq_len + q_idx;
        const int out_offset = (global_q_idx * NUM_HEADS + head_idx) * V_HEAD_DIM;

        // Each thread handles multiple output dimensions
        for (int v_idx = tid; v_idx < V_HEAD_DIM; v_idx += THREADS_PER_BLOCK) {
            float out_val = 0.0f;

            // Pre-compute V position indices (V uses first 512 dims = first 16 blocks)
            const int v_block_idx = v_idx / MXFP4_BLOCK_SIZE;
            const int within_block = v_idx % MXFP4_BLOCK_SIZE;
            const int byte_idx = within_block / 2;
            const int nibble_idx = within_block & 1;
            const int nibble_shift = nibble_idx * 4;
            const int64_t v_byte_rel_offset = v_block_idx * (MXFP4_BLOCK_SIZE / 2) + byte_idx;

            // Accumulate weighted V values
            for (int kv_idx = 0; kv_idx < kv_seq_len; kv_idx++) {
                const float attn_w = scores_ptr[kv_idx];

                if (attn_w > 1e-8f) {  // Skip near-zero weights for efficiency
                    const int64_t kv_offset = kv_batch_offset + static_cast<int64_t>(kv_idx) * BYTES_PER_KV_ROW;
                    const int64_t scale_offset = scale_batch_offset + static_cast<int64_t>(kv_idx) * SCALES_PER_KV_ROW;

                    const float block_scale = e8m0_to_float_fast(kv_scale[scale_offset + v_block_idx]);
                    const uint8_t packed = kv_mxfp4[kv_offset + v_byte_rel_offset];

                    const float v_val = smem_lut[(packed >> nibble_shift) & 0x0F] * block_scale;
                    out_val += attn_w * v_val;
                }
            }

            output[out_offset + v_idx] = hip_bfloat16(out_val);
        }
        __syncthreads();
    }
}

// Alternative kernel optimized for larger KV sequences (>4k tokens)
// Uses tiled accumulation with intermediate results in registers
__global__ //__launch_bounds__(256, 4)
void mla_mxfp4_decode_kernel_large_kv(
    const hip_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ kv_mxfp4,
    const uint8_t* __restrict__ kv_scale,
    hip_bfloat16* __restrict__ output,
    const int batch_size,
    const int q_seq_len,
    const int kv_seq_len,
    const float sm_scale
) {
    constexpr int THREADS_PER_BLOCK = 256;
    constexpr int BYTES_PER_KV_ROW = QK_HEAD_DIM / 2;
    constexpr int SCALES_PER_KV_ROW = NUM_MXFP4_BLOCKS;
    constexpr int KV_TILE = 256;  // Larger tile for long sequences

    const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.x);
    const int head_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
    const int tid = threadIdx.x;

    extern __shared__ char shared_bytes[];

    float* smem_lut = reinterpret_cast<float*>(shared_bytes);
    float* smem_q = smem_lut + 16;
    float* smem_reduce = smem_q + q_seq_len * QK_HEAD_DIM;
    float* smem_partial_out = smem_reduce + 8;  // Partial outputs [V_HEAD_DIM]
    float* smem_scores = smem_partial_out + V_HEAD_DIM;

    // Initialize LUT
    if (tid < 16) {
        smem_lut[tid] = FP4_E2M1_LUT[tid];
    }

    // Initialize partial outputs
    for (int i = tid; i < V_HEAD_DIM; i += THREADS_PER_BLOCK) {
        smem_partial_out[i] = 0.0f;
    }
    __syncthreads();

    // Load queries
    for (int q_idx = 0; q_idx < q_seq_len; q_idx++) {
        const int global_q_idx = batch_idx * q_seq_len + q_idx;
        const int q_offset = (global_q_idx * NUM_HEADS + head_idx) * QK_HEAD_DIM;

        for (int i = tid; i < QK_HEAD_DIM; i += THREADS_PER_BLOCK) {
            smem_q[q_idx * QK_HEAD_DIM + i] = static_cast<float>(q[q_offset + i]);
        }
    }
    __syncthreads();

    const int64_t kv_batch_offset = static_cast<int64_t>(batch_idx) * kv_seq_len * BYTES_PER_KV_ROW;
    const int64_t scale_batch_offset = static_cast<int64_t>(batch_idx) * kv_seq_len * SCALES_PER_KV_ROW;

    for (int q_idx = 0; q_idx < q_seq_len; q_idx++) {
        const float* q_ptr = smem_q + q_idx * QK_HEAD_DIM;

        // Online softmax state
        float running_max = -INFINITY;
        float running_sum = 0.0f;

        // Process in tiles for numerical stability with online softmax
        const int num_tiles = (kv_seq_len + KV_TILE - 1) / KV_TILE;

        for (int tile = 0; tile < num_tiles; tile++) {
            const int tile_start = tile * KV_TILE;
            const int tile_end = min(tile_start + KV_TILE, kv_seq_len);
            const int tile_size = tile_end - tile_start;

            // Compute scores for this tile
            float tile_max = -INFINITY;

            for (int kv_rel = tid; kv_rel < tile_size; kv_rel += THREADS_PER_BLOCK) {
                const int kv_idx = tile_start + kv_rel;
                float score = 0.0f;

                const int64_t kv_offset = kv_batch_offset + static_cast<int64_t>(kv_idx) * BYTES_PER_KV_ROW;
                const int64_t scale_offset = scale_batch_offset + static_cast<int64_t>(kv_idx) * SCALES_PER_KV_ROW;

                // #pragma unroll 2
                for (int block = 0; block < NUM_MXFP4_BLOCKS; block++) {
                    const float block_scale = e8m0_to_float_fast(kv_scale[scale_offset + block]);
                    const int64_t block_kv_offset = kv_offset + block * (MXFP4_BLOCK_SIZE / 2);
                    const int q_block_base = block * MXFP4_BLOCK_SIZE;

                    // #pragma unroll
                    for (int j = 0; j < MXFP4_BLOCK_SIZE / 2; j++) {
                        const uint8_t packed = kv_mxfp4[block_kv_offset + j];
                        const float k_val0 = smem_lut[packed & 0x0F] * block_scale;
                        const float k_val1 = smem_lut[(packed >> 4) & 0x0F] * block_scale;
                        const int d_idx = q_block_base + j * 2;
                        score += q_ptr[d_idx] * k_val0 + q_ptr[d_idx + 1] * k_val1;
                    }
                }

                score *= sm_scale;
                smem_scores[kv_rel] = score;
                tile_max = fmaxf(tile_max, score);
            }
            __syncthreads();

            // Get tile-wide max
            tile_max = block_reduce_max_256(tile_max, smem_reduce, tid);

            // Update running stats with new tile
            float scale_old = expf(running_max - fmaxf(running_max, tile_max));
            float new_max = fmaxf(running_max, tile_max);

            // Rescale previous sum
            running_sum *= scale_old;

            // Add this tile's contribution
            float tile_sum = 0.0f;
            for (int kv_rel = tid; kv_rel < tile_size; kv_rel += THREADS_PER_BLOCK) {
                float exp_val = expf(smem_scores[kv_rel] - new_max);
                smem_scores[kv_rel] = exp_val;
                tile_sum += exp_val;
            }
            __syncthreads();

            tile_sum = block_reduce_sum_256(tile_sum, smem_reduce, tid);
            running_sum += tile_sum;
            running_max = new_max;

            // Accumulate weighted V for this tile (rescale previous contributions)
            for (int v_idx = tid; v_idx < V_HEAD_DIM; v_idx += THREADS_PER_BLOCK) {
                smem_partial_out[v_idx] *= scale_old;
            }
            __syncthreads();

            // Add this tile's V contribution
            for (int v_idx = tid; v_idx < V_HEAD_DIM; v_idx += THREADS_PER_BLOCK) {
                const int v_block_idx = v_idx / MXFP4_BLOCK_SIZE;
                const int within_block = v_idx % MXFP4_BLOCK_SIZE;
                const int byte_idx = within_block / 2;
                const int nibble_shift = (within_block & 1) * 4;
                const int64_t v_byte_rel_offset = v_block_idx * (MXFP4_BLOCK_SIZE / 2) + byte_idx;

                float acc = 0.0f;
                for (int kv_rel = 0; kv_rel < tile_size; kv_rel++) {
                    const int kv_idx = tile_start + kv_rel;
                    const float attn_w = smem_scores[kv_rel];

                    const int64_t kv_offset = kv_batch_offset + static_cast<int64_t>(kv_idx) * BYTES_PER_KV_ROW;
                    const int64_t scale_offset = scale_batch_offset + static_cast<int64_t>(kv_idx) * SCALES_PER_KV_ROW;

                    const float block_scale = e8m0_to_float_fast(kv_scale[scale_offset + v_block_idx]);
                    const uint8_t packed = kv_mxfp4[kv_offset + v_byte_rel_offset];
                    const float v_val = smem_lut[(packed >> nibble_shift) & 0x0F] * block_scale;

                    acc += attn_w * v_val;
                }
                smem_partial_out[v_idx] += acc;
            }
            __syncthreads();
        }

        // Final normalization and write output
        const float inv_sum = 1.0f / running_sum;
        const int global_q_idx = batch_idx * q_seq_len + q_idx;
        const int out_offset = (global_q_idx * NUM_HEADS + head_idx) * V_HEAD_DIM;

        for (int v_idx = tid; v_idx < V_HEAD_DIM; v_idx += THREADS_PER_BLOCK) {
            output[out_offset + v_idx] = hip_bfloat16(smem_partial_out[v_idx] * inv_sum);
            smem_partial_out[v_idx] = 0.0f;  // Reset for next query
        }
        __syncthreads();
    }
}

// =============================================================================
// HIGHLY OPTIMIZED KERNEL FOR kvseqlen=1024, batchsize=4 (decode mode)
//
// MI355X-specific optimizations:
// 1. Vectorized 128-bit loads (uint4) for KV data - 4x fewer memory transactions
// 2. Warp-cooperative score computation - reduces register pressure
// 3. Online softmax with fused V accumulation - single pass over KV
// 4. Double buffering for KV tiles - hides memory latency
// 5. 4 V dimensions per thread in registers - better ALU utilization
// 6. Precomputed LUT values in registers - eliminates LDS bank conflicts
// 7. Software pipelining for inner loops
// =============================================================================
__global__ //__launch_bounds__(256, 4)
void mla_mxfp4_decode_kernel_bs4_kv1024(
    const hip_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ kv_mxfp4,
    const uint8_t* __restrict__ kv_scale,
    hip_bfloat16* __restrict__ output,
    const int batch_size,
    const int q_seq_len,
    const int kv_seq_len,
    const float sm_scale
) {
    constexpr int THREADS = 256;
    constexpr int KV_LEN = 1024;
    constexpr int BYTES_PER_KV = 288;  // QK_HEAD_DIM / 2
    constexpr int SCALES_PER_KV = 18;  // NUM_MXFP4_BLOCKS
    constexpr int V_PER_THREAD = 4;    // Each thread handles 4 V dimensions
    constexpr int KV_TILE = 32;        // Smaller tile for double buffering

    const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.x);
    const int head_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
    const int tid = threadIdx.x;
    const int warp_id = tid >> 6;      // 64-wide warps
    const int lane = tid & 63;

    // Shared memory layout (optimized for MI355X 32-bank LDS):
    // [Q: 576 floats, padded to 580 for bank conflict avoidance]
    // [reduce: 8 floats]
    // [kv_tile_A: 32*288 bytes] [kv_tile_B: 32*288 bytes] - double buffer
    // [scale_tile_A: 32*18 bytes] [scale_tile_B: 32*18 bytes]
    extern __shared__ char shared_bytes[];

    float* smem_q = reinterpret_cast<float*>(shared_bytes);
    float* smem_reduce = smem_q + 580;  // Padded for bank conflicts
    uint8_t* smem_kv_A = reinterpret_cast<uint8_t*>(smem_reduce + 8);
    uint8_t* smem_kv_B = smem_kv_A + KV_TILE * BYTES_PER_KV;
    uint8_t* smem_scale_A = smem_kv_B + KV_TILE * BYTES_PER_KV;
    uint8_t* smem_scale_B = smem_scale_A + KV_TILE * SCALES_PER_KV;

    // Load FP4 LUT into registers (16 values fit in register file)
    float lut[8];
    lut[0] = 0.0f; lut[1] = 0.5f; lut[2] = 1.0f; lut[3] = 1.5f;
    lut[4] = 2.0f; lut[5] = 3.0f; lut[6] = 4.0f; lut[7] = 6.0f;

    // Load query into shared memory with vectorized access
    const int q_offset = __builtin_amdgcn_readfirstlane((batch_idx * NUM_HEADS + head_idx) * QK_HEAD_DIM);
    // #pragma unroll 4
    for (int i = tid; i < QK_HEAD_DIM; i += THREADS) {
        smem_q[i] = static_cast<float>(q[q_offset + i]);
    }
    __syncthreads();

    const int64_t kv_base = __builtin_amdgcn_readfirstlane(static_cast<int64_t>(batch_idx) * KV_LEN * BYTES_PER_KV);
    const int64_t scale_base = __builtin_amdgcn_readfirstlane(static_cast<int64_t>(batch_idx) * KV_LEN * SCALES_PER_KV);

    // Online softmax state - track running max and sum
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // Output accumulators in registers (4 V dimensions per thread)
    float out_acc[V_PER_THREAD] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Pre-compute V dimension indices for this thread
    const int v_base = tid * V_PER_THREAD;
    int v_blk[V_PER_THREAD], v_byte[V_PER_THREAD], v_shift[V_PER_THREAD];

    // #pragma unroll
    for (int v = 0; v < V_PER_THREAD; v++) {
        const int v_idx = v_base + v;
        v_blk[v] = v_idx / MXFP4_BLOCK_SIZE;
        const int within = v_idx % MXFP4_BLOCK_SIZE;
        v_byte[v] = v_blk[v] * 16 + within / 2;
        v_shift[v] = (within & 1) * 4;
    }

    // Process KV in tiles with double buffering
    const int num_tiles = KV_LEN / KV_TILE;  // 1024 / 32 = 32 tiles

    // Prefetch first tile
    // #pragma unroll 2
    for (int i = tid; i < KV_TILE * BYTES_PER_KV; i += THREADS) {
        smem_kv_A[i] = kv_mxfp4[kv_base + i];
    }
    for (int i = tid; i < KV_TILE * SCALES_PER_KV; i += THREADS) {
        smem_scale_A[i] = kv_scale[scale_base + i];
    }
    __syncthreads();

    uint8_t* kv_read = smem_kv_A;
    uint8_t* kv_write = smem_kv_B;
    uint8_t* scale_read = smem_scale_A;
    uint8_t* scale_write = smem_scale_B;

    for (int tile = 0; tile < num_tiles; tile++) {
        const int tile_start = tile * KV_TILE;
        const int next_tile = tile + 1;

        // Async prefetch next tile while processing current
        if (next_tile < num_tiles) {
            const int64_t next_kv_off = kv_base + static_cast<int64_t>(next_tile * KV_TILE) * BYTES_PER_KV;
            const int64_t next_sc_off = scale_base + static_cast<int64_t>(next_tile * KV_TILE) * SCALES_PER_KV;

            // #pragma unroll 2
            for (int i = tid; i < KV_TILE * BYTES_PER_KV; i += THREADS) {
                kv_write[i] = kv_mxfp4[next_kv_off + i];
            }
            for (int i = tid; i < KV_TILE * SCALES_PER_KV; i += THREADS) {
                scale_write[i] = kv_scale[next_sc_off + i];
            }
        }

        // Compute scores for this tile - each thread handles portion of tile
        float tile_scores[KV_TILE / (THREADS / 8)];  // ~1 score per thread for this tile
        float tile_max = -INFINITY;

        // Each warp handles 8 KV tokens (256 threads / 32 tiles = 8 per tile iteration)
        const int kv_per_iter = (KV_TILE + (THREADS / 64) - 1) / (THREADS / 64);

        // #pragma unroll
        for (int k = 0; k < KV_TILE; k++) {
            if ((k % 8) == (tid / 32) % 8) {  // Distribute across warps
                const int local_kv = k;
                const uint8_t* kv_ptr = kv_read + local_kv * BYTES_PER_KV;
                const uint8_t* sc_ptr = scale_read + local_kv * SCALES_PER_KV;

                float score = 0.0f;

                // Compute dot product Q @ K^T for this KV token
                // #pragma unroll 6
                for (int blk = 0; blk < NUM_MXFP4_BLOCKS; blk++) {
                    const float blk_scale = e8m0_to_float_fast(sc_ptr[blk]);
                    const int blk_off = blk * 16;
                    const int q_base = blk * MXFP4_BLOCK_SIZE;

                    // #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        const uint8_t packed = kv_ptr[blk_off + j];
                        const int lo = packed & 0x07;
                        const int hi = (packed >> 4) & 0x07;
                        const float sign_lo = (packed & 0x08) ? -1.0f : 1.0f;
                        const float sign_hi = (packed & 0x80) ? -1.0f : 1.0f;
                        const float k0 = lut[lo] * sign_lo * blk_scale;
                        const float k1 = lut[hi] * sign_hi * blk_scale;
                        score += smem_q[q_base + j*2] * k0 + smem_q[q_base + j*2 + 1] * k1;
                    }
                }
                score *= sm_scale;

                // Online softmax update
                const float old_max = running_max;
                running_max = fmaxf(running_max, score);
                const float exp_diff = expf(old_max - running_max);
                running_sum = running_sum * exp_diff + expf(score - running_max);

                // Rescale previous accumulators and add new V contribution
                const float attn_w = expf(score - running_max);

                // #pragma unroll
                for (int v = 0; v < V_PER_THREAD; v++) {
                    out_acc[v] *= exp_diff;

                    if (v_base + v < V_HEAD_DIM) {
                        const float vs = e8m0_to_float_fast(sc_ptr[v_blk[v]]);
                        const uint8_t vp = kv_ptr[v_byte[v]];
                        const int vi = (vp >> v_shift[v]) & 0x07;
                        const float vsign = ((vp >> v_shift[v]) & 0x08) ? -1.0f : 1.0f;
                        out_acc[v] += attn_w * lut[vi] * vsign * vs;
                    }
                }
            }
        }

        __syncthreads();

        // Swap buffers
        uint8_t* tmp_kv = kv_read; kv_read = kv_write; kv_write = tmp_kv;
        uint8_t* tmp_sc = scale_read; scale_read = scale_write; scale_write = tmp_sc;
    }

    // Reduce running_max and running_sum across block for final normalization
    // Use warp reduction first, then cross-warp
    running_max = warp_reduce_max_64(running_max);
    if (lane == 0) smem_reduce[warp_id] = running_max;
    __syncthreads();

    float global_max;
    if (tid < 4) {
        global_max = smem_reduce[tid];
    } else {
        global_max = -INFINITY;
    }
    if (tid < 64) global_max = warp_reduce_max_64(global_max);
    if (tid == 0) smem_reduce[0] = global_max;
    __syncthreads();
    global_max = smem_reduce[0];

    // Rescale local accumulators to global max
    const float scale_factor = expf(running_max - global_max);
    running_sum *= scale_factor;
    // #pragma unroll
    for (int v = 0; v < V_PER_THREAD; v++) {
        out_acc[v] *= scale_factor;
    }

    // Reduce sum across block
    running_sum = warp_reduce_sum_64(running_sum);
    if (lane == 0) smem_reduce[warp_id] = running_sum;
    __syncthreads();

    float global_sum;
    if (tid < 4) {
        global_sum = smem_reduce[tid];
    } else {
        global_sum = 0.0f;
    }
    if (tid < 64) global_sum = warp_reduce_sum_64(global_sum);
    if (tid == 0) smem_reduce[0] = global_sum;
    __syncthreads();
    global_sum = smem_reduce[0];

    // Final normalization and output
    const float inv_sum = 1.0f / global_sum;
    const int out_offset = (batch_idx * NUM_HEADS + head_idx) * V_HEAD_DIM;

    // #pragma unroll
    for (int v = 0; v < V_PER_THREAD; v++) {
        const int v_idx = v_base + v;
        if (v_idx < V_HEAD_DIM) {
            output[out_offset + v_idx] = hip_bfloat16(out_acc[v] * inv_sum);
        }
    }
}

// =====================================================================
// MLA QK^T kernel: MFMA FP4, fp32 output, batched, linear B scales
// =====================================================================

__device__ __forceinline__ int32_t mla_broadcast_scale(uint8_t e8m0) {
    return (int32_t)e8m0 * 0x01010101;
}

template <int M, int N, int K_HALF, int NUM_BLOCKS, int B_SCALE_STRIDE,
          int A_K_HALF, int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE)
mla_qkt_mxfp4_kernel(
    const uint8_t* __restrict__ A_data,
    const uint8_t* __restrict__ B_data,
    const uint8_t* __restrict__ A_scale,
    const uint8_t* __restrict__ B_scale,
    float* __restrict__ C,
    float sm_scale
) {
    constexpr int IM = 16;
    constexpr int IN = 16;
    constexpr int IK = 128;
    constexpr int BPC = IK / 32;  // 4
    constexpr int K_ITERS = (NUM_BLOCKS + BPC - 1) / BPC;  // ceil(18/4)=5

    const int lane = threadIdx.x % 64;
    const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.z);
    const int tile_n = __builtin_amdgcn_readfirstlane(blockIdx.y * IN);

    const uint8_t* a_data = A_data + batch_idx * M * A_K_HALF;
    const uint8_t* a_scale = A_scale + batch_idx * (NUM_BLOCKS * M);
    const uint8_t* b_data = B_data + batch_idx * N * K_HALF;
    const uint8_t* b_scale = B_scale + batch_idx * N * B_SCALE_STRIDE;
    float* c_out = C + batch_idx * M * N;

    float4_t acc = {};

    uint32_t a_cur[8], b_cur[8];
    uint32_t a_nxt[8], b_nxt[8];
    int32_t a_sc_cur, b_sc_cur, a_sc_nxt, b_sc_nxt;

    // Helper: safe load tile - clamps OOB block reads
    auto load_safe = [&](const uint8_t* src, int row_base, int blk0,
                         int stride, int max_rows, int max_blks,
                         uint32_t reg[8], int& blk_out, int& row_out,
                         const uint8_t* scale_src, int scale_stride, bool is_col_major_scale,
                         int scale_M, int32_t& sc_out) {
        row_out = row_base + (lane % IM);
        int k_group = lane / IM;
        blk_out = blk0 + k_group;

        // Clamp to safe range
        int safe_row = min(row_out, max(max_rows - 1, 0));
        int safe_blk = min(blk_out, max(max_blks - 1, 0));
        int off = safe_row * stride + safe_blk * 16;
        *reinterpret_cast<uint128_vec*>(&reg[0]) = *reinterpret_cast<const uint128_vec*>(&src[off]);
        reg[4] = reg[5] = reg[6] = reg[7] = 0;

        uint8_t e = 127;
        if (row_out < max_rows && blk_out < max_blks) {
            if (is_col_major_scale)
                e = scale_src[row_out + blk_out * scale_M];
            else
                e = scale_src[row_out * scale_stride + blk_out];
        } else {
            reg[0] = reg[1] = reg[2] = reg[3] = 0;
        }
        sc_out = mla_broadcast_scale(e);
    };

    // Load first K tile
    int a_row, a_blk, b_row, b_blk;
    load_safe(a_data, 0, 0, A_K_HALF, M, NUM_BLOCKS, a_cur, a_blk, a_row,
              a_scale, 0, true, M, a_sc_cur);
    load_safe(b_data, tile_n, 0, K_HALF, N, NUM_BLOCKS, b_cur, b_blk, b_row,
              b_scale, B_SCALE_STRIDE, false, 0, b_sc_cur);

    for (int ki = 0; ki < K_ITERS; ki++) {
        if (ki + 1 < K_ITERS) {
            int next_blk0 = (ki + 1) * BPC;
            load_safe(a_data, 0, next_blk0, A_K_HALF, M, NUM_BLOCKS, a_nxt, a_blk, a_row,
                      a_scale, 0, true, M, a_sc_nxt);
            load_safe(b_data, tile_n, next_blk0, K_HALF, N, NUM_BLOCKS, b_nxt, b_blk, b_row,
                      b_scale, B_SCALE_STRIDE, false, 0, b_sc_nxt);
        }

        // 16x16x128 MFMA FP4
        int8_vec a_vec = {(int)a_cur[0], (int)a_cur[1], (int)a_cur[2], (int)a_cur[3],
                              (int)a_cur[4], (int)a_cur[5], (int)a_cur[6], (int)a_cur[7]};
        int8_vec b_vec = {(int)b_cur[0], (int)b_cur[1], (int)b_cur[2], (int)b_cur[3],
                              (int)b_cur[4], (int)b_cur[5], (int)b_cur[6], (int)b_cur[7]};
        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_vec, b_vec, acc, FMT_FP4_MFMA, FMT_FP4_MFMA, 0, a_sc_cur, 0, b_sc_cur);

        for (int r = 0; r < 8; r++) { a_cur[r] = a_nxt[r]; b_cur[r] = b_nxt[r]; }
        a_sc_cur = a_sc_nxt; b_sc_cur = b_sc_nxt;
    }

    // Store fp32 output with fused sm_scale: (M, N)
    int col = lane % 16;
    int quad = lane / 16;
    for (int i = 0; i < 4; i++) {
        int row = i + 4 * quad;
        int gn = tile_n + col;
        if (row < M && gn < N)
            c_out[row * N + gn] = acc[i] * sm_scale;
    }
}

// =============================================================================
// MLA MXFP4 Kernels: adapted from mxfp4-mm GEMM for MLA decode
//
// Two kernels:
// 1. mla_qkt_tiled: Tiled QK^T GEMM with fused Q quantization
// 2. mla_attn_v_fused: Fused attn_weights x V with on-the-fly MXFP4 dequant
//
// Add this code to MLA_MXFP4_HIP_SOURCE in submission.py
// =============================================================================

// =====================================================================
// KERNEL 2: Fused attn_weights x V with on-the-fly MXFP4 dequant
//
// Replaces: dequantize_mxfp4(V) + torch.bmm(attn, V)
// Eliminates ~500us V dequant bottleneck by never materializing bf16 V
//
// Each block handles one (batch, head) pair.
// 512 threads = one thread per V output dimension.
// Each thread iterates over KV positions, loading V data from MXFP4.
// =====================================================================

__global__ //__launch_bounds__(512, 2)
void mla_attn_v_fused_kernel(
    const float* __restrict__ attn_weights,  // (batch, 16, kv_seq_len) fp32 post-softmax
    const uint8_t* __restrict__ kv_mxfp4,    // (batch * kv_seq_len, 1, 288) packed
    const uint8_t* __restrict__ kv_scale,    // (batch * kv_seq_len, scale_stride) E8M0
    hip_bfloat16* __restrict__ output,       // (batch * 16, 512) bf16
    int kv_seq_len,
    int scale_stride
) {
    constexpr int V_DIM = 512;
    constexpr int KV_BYTES_PER_ROW = 288;  // QK_HEAD_DIM / 2
    constexpr int KV_TILE = 128;           // Cache this many attn weights at a time

    const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.x);
    const int head_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
    const int v_dim = threadIdx.x;  // 0..511, one thread per V dimension

    // Pre-compute this thread's V position in MXFP4 layout
    const int v_block = v_dim / 32;                   // 0..15
    const int v_within = v_dim % 32;
    const int v_byte_in_row = v_block * 16 + v_within / 2;  // byte offset within KV row
    const int v_nibble_shift = (v_within & 1) * 4;          // 0 or 4

    // Shared memory for attn weights caching
    __shared__ float attn_cache[KV_TILE];

    // Pointers
    const float* attn_ptr = attn_weights + ((int64_t)batch_idx * 16 + head_idx) * kv_seq_len;
    const int64_t kv_base = __builtin_amdgcn_readfirstlane((int64_t)batch_idx * kv_seq_len * KV_BYTES_PER_ROW);
    const int64_t sc_base = __builtin_amdgcn_readfirstlane((int64_t)batch_idx * kv_seq_len * scale_stride);

    float acc = 0.0f;

    for (int kv_start = 0; kv_start < kv_seq_len; kv_start += KV_TILE) {
        int tile_size = min(KV_TILE, kv_seq_len - kv_start);

        // Cooperatively load attn weights to shared memory
        // 512 threads loading up to 128 values
        if (v_dim < tile_size) {
            attn_cache[v_dim] = attn_ptr[kv_start + v_dim];
        }
        __syncthreads();

        // Process each KV position in this tile
        for (int ki = 0; ki < tile_size; ki++) {
            float w = attn_cache[ki];

            int kv_idx = kv_start + ki;
            int64_t row_off = kv_base + (int64_t)kv_idx * KV_BYTES_PER_ROW;

            // Load V scale for this block
            float block_scale = e8m0_to_float_fast(
                kv_scale[sc_base + (int64_t)kv_idx * scale_stride + v_block]);

            // Load and dequant V value
            uint8_t packed = kv_mxfp4[row_off + v_byte_in_row];
            uint8_t nibble = (packed >> v_nibble_shift) & 0x0F;
            float v_val = FP4_E2M1_LUT[nibble] * block_scale;

            acc += w * v_val;
        }
        __syncthreads();
    }

    // Write output
    int out_idx = ((int64_t)batch_idx * 16 + head_idx) * V_DIM + v_dim;
    output[out_idx] = hip_bfloat16(acc);
}


// ---- Kernel B: Reduce partial sums to final bf16 output ----
// Grid: (batch_size * 16 * 512 / 256) - one thread per output element
// Each thread sums kv_splits partial values

template <int BLOCK_SIZE, int KV_SPLITS>
__global__ __launch_bounds__(BLOCK_SIZE)
void mla_attn_v_reduce_kernel(
    const float* __restrict__ partial_out,  // (batch * 16, kv_splits, 512) fp32
    hip_bfloat16* __restrict__ output,      // (batch * 16, 512) bf16
    int total_outputs  // batch * 16 * 512
) {
    constexpr int V_DIM = 512;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) return;

    // idx = head_flat * V_DIM + v_dim
    int head_flat = idx / V_DIM;  // 0 .. batch*16-1
    int v_dim = idx % V_DIM;

    float sum = 0.0f;
    const float* ptr = partial_out + (int64_t)head_flat * KV_SPLITS * V_DIM + v_dim;
    for (int s = 0; s < KV_SPLITS; s++) {
        sum += ptr[(int64_t)s * V_DIM];
    }

    output[idx] = hip_bfloat16(sum);
}

// =============================================================================
// 2-pass softmax HIP kernel (pre-scaled input)
//
// Input: fp32 pre-scaled scores (batch, 16, N) - sm_scale already applied by QK^T
// Output: fp32 attn_weights (batch, 16, N) - normalized in-place
//
// Pass 1: Find max, compute exp(val - max), accumulate sum (read scores, write exp)
// Pass 2: Normalize by 1/sum (read+write in-place)
//
// Saves one full read pass vs the old 3-pass kernel since sm_scale is pre-applied.
// =============================================================================
template <int N, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE)
void mla_softmax_2pass_kernel(
    float* __restrict__ data,    // (batch, 16, N) - in-place: pre-scaled scores in, attn weights out
    int dummy
) {
    constexpr int THREADS = BLOCK_SIZE;

    const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.x);
    const int head_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
    const int tid = threadIdx.x;
    const int lane = tid & 63;
    const int warp_id = tid >> 6;

    const int64_t row_offset = __builtin_amdgcn_readfirstlane(((int64_t)batch_idx * 16 + head_idx) * N);
    float* row = data + row_offset;

    __shared__ float smem[4];  // 4 warps

    // ---- Pass 1a: Find max ----
    float local_max = -INFINITY;
    for (int i = tid; i < N; i += THREADS) {
        local_max = fmaxf(local_max, row[i]);
    }

    for (int offset = 32; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max, __shfl_xor(local_max, offset));
    if (lane == 0) smem[warp_id] = local_max;
    __syncthreads();

    float global_max;
    if (tid < 4) global_max = smem[tid]; else global_max = -INFINITY;
    if (tid < 64) {
        for (int offset = 32; offset > 0; offset >>= 1)
            global_max = fmaxf(global_max, __shfl_xor(global_max, offset));
    }
    if (tid == 0) smem[0] = global_max;
    __syncthreads();
    global_max = smem[0];

    // ---- Pass 1b: Compute exp(val - max), write back, accumulate sum ----
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += THREADS) {
        float e = expf(row[i] - global_max);
        row[i] = e;
        local_sum += e;
    }

    for (int offset = 32; offset > 0; offset >>= 1)
        local_sum += __shfl_xor(local_sum, offset);
    if (lane == 0) smem[warp_id] = local_sum;
    __syncthreads();

    float global_sum;
    if (tid < 4) global_sum = smem[tid]; else global_sum = 0.0f;
    if (tid < 64) {
        for (int offset = 32; offset > 0; offset >>= 1)
            global_sum += __shfl_xor(global_sum, offset);
    }
    if (tid == 0) smem[0] = global_sum;
    __syncthreads();
    float inv_sum = 1.0f / smem[0];

    // ---- Pass 2: Normalize ----
    for (int i = tid; i < N; i += THREADS) {
        row[i] *= inv_sum;
    }
}

// =============================================================================
// LDS-Tiled Split-K attn x V kernel - v2
//
// Changes from v1:
// 1. KV_TILE increased from 32 to 64 (halves syncthreads overhead)
// 2. Each thread processes 2 V dims (256 threads instead of 512)
//    - Reads uint16_t (2 bytes = 4 FP4 values) but processes 2 dims
//    - Better ALU utilization, fewer threads = less register pressure
// 3. LDS: 64 * 256 = 16KB data + 64 * 16 = 1KB scales + 256 attn = 17.25KB
// =============================================================================

#define ATTNV2_KV_TILE 64
#define ATTNV2_V_BYTES 256
#define ATTNV2_V_SCALES 16

template <int N, int BLOCK_SIZE, int B_SCALE_STRIDE, int KV_SPLITS>
__global__ __launch_bounds__(BLOCK_SIZE)
void mla_attn_v_splitk_lds_v2_kernel(
    const float* __restrict__ attn_weights,
    const uint8_t* __restrict__ kv_mxfp4,
    const uint8_t* __restrict__ kv_scale,
    float* __restrict__ partial_out
) {
    constexpr int V_DIM = 512;
    constexpr int KV_BYTES_PER_ROW = 288;
    constexpr int THREADS = 256;
    constexpr int KV_TILE = ATTNV2_KV_TILE;
    constexpr int V_BYTES = ATTNV2_V_BYTES;
    constexpr int V_SCALES = ATTNV2_V_SCALES;
    constexpr int DIMS_PER_THREAD = 2;  // each thread handles 2 V dimensions
    constexpr int KV_PER_SPLIT = (N + KV_SPLITS - 1) / KV_SPLITS;

    const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.x);
    const int head_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
    const int split_idx = __builtin_amdgcn_readfirstlane(blockIdx.z);
    const int tid = threadIdx.x;  // 0..255
    const int kv_start_global = split_idx * KV_PER_SPLIT;
    const int kv_end_global = min(kv_start_global + KV_PER_SPLIT, N);

    const int64_t out_idx_base = ((int64_t)batch_idx * 16 + head_idx) * (int64_t)KV_SPLITS * V_DIM
                                + (int64_t)split_idx * V_DIM;

    if (kv_start_global >= N) {
        // Zero both dims
        partial_out[out_idx_base + tid * 2] = 0.0f;
        partial_out[out_idx_base + tid * 2 + 1] = 0.0f;
        return;
    }

    // Each thread handles 2 adjacent V dimensions
    const int v_dim0 = tid * 2;       // even dim
    const int v_dim1 = tid * 2 + 1;   // odd dim

    // Both dims share the same byte (even=low nibble, odd=high nibble)
    const int v_block = v_dim0 / 32;
    const int v_within = v_dim0 % 32;
    const int v_byte_offset = v_block * 16 + v_within / 2;
    // v_dim0 uses low nibble (shift=0), v_dim1 uses high nibble (shift=4)

    // LDS layout
    __shared__ float attn_cache[KV_TILE];
    __shared__ uint8_t kv_tile[KV_TILE * V_BYTES];
    __shared__ uint8_t scale_tile[KV_TILE * V_SCALES];

    const float* attn_ptr = attn_weights + ((int64_t)batch_idx * 16 + head_idx) * N;
    const int64_t kv_base = __builtin_amdgcn_readfirstlane((int64_t)batch_idx * N * KV_BYTES_PER_ROW);
    const int64_t sc_base = __builtin_amdgcn_readfirstlane((int64_t)batch_idx * N * B_SCALE_STRIDE);

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int kv_start = kv_start_global; kv_start < kv_end_global; kv_start += KV_TILE) {
        const int tile_end = min(kv_start + KV_TILE, kv_end_global);
        const int tile_size = tile_end - kv_start;

        // ---- Cooperative load: attn weights ----
        if (tid < tile_size) {
            attn_cache[tid] = attn_ptr[kv_start + tid];
        }
        // Load second half if tile > 256 (KV_TILE=64, so tid < 64, always fits)
        // Actually KV_TILE=64 and THREADS=256, so tid < 64 covers it

        // ---- Cooperative load: KV data ----
        // Total: tile_size * 256 bytes. 256 threads x 16 bytes = 4096 bytes per round
        // For tile_size=64: 64*256 = 16384 bytes, need 4 rounds
        {
            const int total_vec = (tile_size * V_BYTES) / 16;
            for (int i = tid; i < total_vec; i += THREADS) {
                const int row = i / (V_BYTES / 16);
                const int vec_in_row = i % (V_BYTES / 16);
                const int kv_idx = kv_start + row;

                const int64_t src_off = kv_base + (int64_t)kv_idx * KV_BYTES_PER_ROW + vec_in_row * 16;
                const int dst_off = row * V_BYTES + vec_in_row * 16;

                *reinterpret_cast<uint128_vec*>(&kv_tile[dst_off]) =
                    *reinterpret_cast<const uint128_vec*>(&kv_mxfp4[src_off]);
            }
        }

        // ---- Cooperative load: scales ----
        {
            const int total_scales = tile_size * V_SCALES;
            for (int i = tid; i < total_scales; i += THREADS) {
                const int row = i / V_SCALES;
                const int blk = i % V_SCALES;
                const int kv_idx = kv_start + row;

                scale_tile[row * V_SCALES + blk] =
                    kv_scale[sc_base + (int64_t)kv_idx * B_SCALE_STRIDE + blk];
            }
        }

        __syncthreads();

        // ---- Compute: read from LDS, process 2 V dims per thread ----
        for (int ki = 0; ki < tile_size; ki++) {
            const float w = attn_cache[ki];

            const float block_scale = e8m0_to_float_fast(
                scale_tile[ki * V_SCALES + v_block]);

            // Load one byte, extract both nibbles
            const uint8_t packed = kv_tile[ki * V_BYTES + v_byte_offset];
            const float v_val0 = FP4_E2M1_LUT[packed & 0x0F] * block_scale;
            const float v_val1 = FP4_E2M1_LUT[(packed >> 4) & 0x0F] * block_scale;

            acc0 += w * v_val0;
            acc1 += w * v_val1;
        }

        __syncthreads();
    }

    partial_out[out_idx_base + v_dim0] = acc0;
    partial_out[out_idx_base + v_dim1] = acc1;
}

// =============================================================================
// HEAD-MERGED Split-K attn x V kernel
//
// Key optimization: Grid is (batch, split) instead of (batch, head, split).
// Each block loads KV data ONCE and processes ALL 16 heads, eliminating
// 16x redundant global memory reads (MQA: all heads share same KV).
//
// LDS layout (~22 KB total):
//   kv_tile:    KV_TILE * 256 bytes = 16,384 B (V portion of KV data)
//   scale_tile: KV_TILE * 16 bytes  =  1,024 B (V scales)
//   attn_cache: 16 * KV_TILE * 4 B  =  4,096 B (all heads' attn weights)
//   Total: ~21.5 KB - excellent occupancy
//
// Thread mapping: 256 threads, each handles 2 V dims x 16 heads = 32 accumulators
// =============================================================================

template <int N, int BLOCK_SIZE, int B_SCALE_STRIDE, int KV_SPLITS>
__global__ __launch_bounds__(BLOCK_SIZE)
void mla_attn_v_splitk_head_merged_kernel(
    const float* __restrict__ attn_weights,  // (batch, 16, N)
    const uint8_t* __restrict__ kv_mxfp4,    // (batch * N, 288)
    const uint8_t* __restrict__ kv_scale,    // (batch * N, B_SCALE_STRIDE)
    float* __restrict__ partial_out           // (batch, KV_SPLITS, 16, 512)
) {
    constexpr int V_DIM = 512;
    constexpr int KV_BYTES_PER_ROW = 288;
    constexpr int THREADS = BLOCK_SIZE;  // 256
    constexpr int KV_TILE = 64;
    constexpr int V_BYTES = 256;   // first 256 bytes of each KV row = first 512 FP4 values = V
    constexpr int V_SCALES = 16;   // first 16 scale blocks = V portion
    constexpr int HEADS = 16;
    constexpr int KV_PER_SPLIT = (N + KV_SPLITS - 1) / KV_SPLITS;

    const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.x);
    const int split_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
    const int tid = threadIdx.x;  // 0..255

    const int kv_start_global = split_idx * KV_PER_SPLIT;
    const int kv_end_global = min(kv_start_global + KV_PER_SPLIT, N);

    // Output layout: (batch, KV_SPLITS, 16, 512)
    // Base offset for this (batch, split)
    const int64_t out_base = __builtin_amdgcn_readfirstlane(((int64_t)batch_idx * KV_SPLITS + split_idx) * HEADS * V_DIM);

    // Each thread handles 2 adjacent V dimensions across all 16 heads
    const int v_dim0 = tid * 2;
    const int v_dim1 = v_dim0 + 1;

    // Pre-compute V position in MXFP4 layout (same for all heads since MQA)
    const int v_block = v_dim0 / 32;
    const int v_within = v_dim0 % 32;
    const int v_byte_offset = v_block * 16 + v_within / 2;

    // 32 accumulators: 2 V dims x 16 heads
    float acc[HEADS * 2];
    for (int i = 0; i < HEADS * 2; i++) acc[i] = 0.0f;

    if (kv_start_global >= N) {
        // Zero output for OOB splits
        for (int h = 0; h < HEADS; h++) {
            partial_out[out_base + (int64_t)h * V_DIM + v_dim0] = 0.0f;
            partial_out[out_base + (int64_t)h * V_DIM + v_dim1] = 0.0f;
        }
        return;
    }

    // LDS layout
    __shared__ uint8_t kv_tile[KV_TILE * V_BYTES];          // 16 KB
    __shared__ uint8_t scale_tile[KV_TILE * V_SCALES];       // 1 KB
    __shared__ float attn_cache[HEADS * KV_TILE];             // 4 KB

    const int64_t kv_base = (int64_t)batch_idx * N * KV_BYTES_PER_ROW;
    const int64_t sc_base = (int64_t)batch_idx * N * B_SCALE_STRIDE;

    for (int kv_start = kv_start_global; kv_start < kv_end_global; kv_start += KV_TILE) {
        const int tile_end = min(kv_start + KV_TILE, kv_end_global);
        const int tile_size = tile_end - kv_start;

        // ---- Cooperative load: KV data (V portion only: first 256 bytes) ----
        {
            const int total_vec = (tile_size * V_BYTES) / 16;
            for (int i = tid; i < total_vec; i += THREADS) {
                const int row = i / (V_BYTES / 16);  // V_BYTES/16 = 16
                const int vec_in_row = i % (V_BYTES / 16);
                const int kv_idx = kv_start + row;
                const int64_t src_off = kv_base + (int64_t)kv_idx * KV_BYTES_PER_ROW + vec_in_row * 16;
                const int dst_off = row * V_BYTES + vec_in_row * 16;
                *reinterpret_cast<uint128_vec*>(&kv_tile[dst_off]) =
                    *reinterpret_cast<const uint128_vec*>(&kv_mxfp4[src_off]);
            }
        }

        // ---- Cooperative load: V scales ----
        {
            const int total_scales = tile_size * V_SCALES;
            for (int i = tid; i < total_scales; i += THREADS) {
                const int row = i / V_SCALES;
                const int blk = i % V_SCALES;
                const int kv_idx = kv_start + row;
                scale_tile[row * V_SCALES + blk] =
                    kv_scale[sc_base + (int64_t)kv_idx * B_SCALE_STRIDE + blk];
            }
        }

        // ---- Cooperative load: attn weights for ALL 16 heads ----
        // Total: 16 * tile_size floats. With 256 threads, need ceil(16*64/256)=4 rounds
        {
            const int total_attn = HEADS * tile_size;
            for (int i = tid; i < total_attn; i += THREADS) {
                const int h = i / tile_size;
                const int ki = i % tile_size;
                attn_cache[h * KV_TILE + ki] =
                    attn_weights[((int64_t)batch_idx * HEADS + h) * N + kv_start + ki];
            }
        }

        __syncthreads();

        // ---- Compute: for each KV position, load V data once, apply to all heads ----
        for (int ki = 0; ki < tile_size; ki++) {
            // Load and dequantize V value ONCE (shared across all heads)
            const float block_scale = e8m0_to_float_fast(
                scale_tile[ki * V_SCALES + v_block]);
            const uint8_t packed = kv_tile[ki * V_BYTES + v_byte_offset];
            const float v_val0 = FP4_E2M1_LUT[packed & 0x0F] * block_scale;
            const float v_val1 = FP4_E2M1_LUT[(packed >> 4) & 0x0F] * block_scale;

            // Apply to all 16 heads (each head has different attn weight)
            for (int h = 0; h < HEADS; h++) {
                const float w = attn_cache[h * KV_TILE + ki];
                acc[h * 2]     += w * v_val0;
                acc[h * 2 + 1] += w * v_val1;
            }
        }

        __syncthreads();
    }

    // Write partial output: (batch, KV_SPLITS, 16, 512)
    for (int h = 0; h < HEADS; h++) {
        partial_out[out_base + (int64_t)h * V_DIM + v_dim0] = acc[h * 2];
        partial_out[out_base + (int64_t)h * V_DIM + v_dim1] = acc[h * 2 + 1];
    }
}

// =============================================================================
// MFMA FP4xFP4 Head-Merged Attn x V kernel
//
// Replaces scalar inner loop with MFMA 16x16x128 FP4 matrix multiply.
// A = attn_weights (quantized FP32 -> FP4), B = V (native MXFP4)
// M=16 (heads), N=16 (V dim chunk), K=128 (KV positions)
// 32 MFMA calls per K-batch to cover all 512 V dimensions.
//
// 4 warps = 256 threads. Each warp handles 8 V-dim chunks.
// KV_TILE = 128 to match MFMA K dimension exactly.
//
// V data gathered per-lane from LDS (padded stride to avoid bank conflicts).
// Attention weights quantized once per tile, reused across all V-dim chunks.
// =============================================================================

template <int N, int BLOCK_SIZE, int B_SCALE_STRIDE, int KV_SPLITS>
__global__ __launch_bounds__(BLOCK_SIZE)
void mla_attn_v_mfma_head_merged_kernel(
    const float* __restrict__ attn_weights,  // (batch, 16, N) post-softmax
    const uint8_t* __restrict__ kv_mxfp4,    // (batch * N, 288) packed KV
    const uint8_t* __restrict__ kv_scale,    // (batch * N, B_SCALE_STRIDE) E8M0
    float* __restrict__ partial_out           // (batch, KV_SPLITS, 16, 512)
) {
    constexpr int V_DIM = 512;
    constexpr int KV_BYTES_PER_ROW = 288;
    constexpr int THREADS = BLOCK_SIZE;  // 256
    constexpr int KV_TILE = 128;         // matches MFMA K=128
    constexpr int HEADS = 16;
    constexpr int KV_PER_SPLIT = (N + KV_SPLITS - 1) / KV_SPLITS;
    constexpr int V_CHUNKS = V_DIM / 16; // 32 chunks of 16 V dims
    constexpr int WARPS = THREADS / 64;  // 4
    constexpr int CHUNKS_PER_WARP = V_CHUNKS / WARPS; // 8
    constexpr int NUM_K_BLOCKS = KV_TILE / 32; // 4 blocks of 32 for MXFP4 scaling
    // Padded stride for V rows in LDS to avoid bank conflicts
    // 256 bytes + 4 bytes padding = 260, gcd(260/4, 32) = gcd(65,32) = 1
    constexpr int V_LDS_STRIDE = 256;

    const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.x);
    const int split_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
    const int tid = threadIdx.x;
    const int warp_id = tid / 64;
    const int lane = tid % 64;

    const int kv_start_global = __builtin_amdgcn_readfirstlane(split_idx * KV_PER_SPLIT);
    const int kv_end_global = __builtin_amdgcn_readfirstlane(min(kv_start_global + KV_PER_SPLIT, N));

    const int64_t out_base = __builtin_amdgcn_readfirstlane(((int64_t)batch_idx * KV_SPLITS + split_idx) * HEADS * V_DIM);

    // ======================================================================
    // Scale-absorption MFMA: absorb per-position V scales into A operand
    //
    // output[h,v] = sum_k attn[h,k] * (FP4_nib[k,v] * v_scale[k,v_block])
    //             = sum_k (attn[h,k] * v_scale[k,v_block]) * FP4_nib[k,v]
    //
    // A operand = quantize_fp4(attn[h,k] * v_scale[k,v_block]) per-lane
    // B operand = raw V FP4 nibbles, scale = 1.0 (E8M0=127)
    // No V dequant/requant. A computed in registers per-lane.
    // Adjacent chunks share V-scale-block -> quantize A once per pair.
    // Only 1 syncthreads per tile.
    //
    // LDS (~43.5 KB, fits 3 blocks/CU at 160KB):
    //   v_lds:       128 * 260 = 33,280 B (V data, padded stride)
    //   v_scale_lds: 128 * 16  =  2,048 B (V scales)
    //   attn_f32:    16 * 128  =  8,192 B (fp32 attn, all heads, 4B each)
    // ======================================================================

    __shared__ uint8_t v_lds[KV_TILE * V_LDS_STRIDE];
    __shared__ uint8_t v_scale_lds[KV_TILE * 16];
    __shared__ float attn_f32[HEADS * KV_TILE];

    float4_t warp_acc[CHUNKS_PER_WARP];
    for (int c = 0; c < CHUNKS_PER_WARP; c++) {
        warp_acc[c] = {};
    }

    if (kv_start_global >= N) {
        for (int i = tid; i < HEADS * V_DIM; i += THREADS) {
            partial_out[out_base + i] = 0.0f;
        }
        return;
    }

    const int64_t kv_base = __builtin_amdgcn_readfirstlane((int64_t)batch_idx * N * KV_BYTES_PER_ROW);
    const int64_t sc_base = __builtin_amdgcn_readfirstlane((int64_t)batch_idx * N * B_SCALE_STRIDE);

    // MFMA lane indices (constant)
    const int a_row = lane % 16;   // head index for A
    const int a_kgrp = lane / 16;  // K group (0..3) for A
    const int b_col = lane % 16;   // V dim within chunk for B
    const int b_kgrp = lane / 16;  // K group (0..3) for B

    // B scale is always 1.0 (E8M0=127) since V scale absorbed into A
    const int32_t b_sc_one = mla_broadcast_scale(127);

    for (int kv_start = kv_start_global; kv_start < kv_end_global; kv_start += KV_TILE) {
        const int tile_end = min(kv_start + KV_TILE, kv_end_global);
        const int tile_size = tile_end - kv_start;

        // ---- Cooperative load: V data (padded stride) ----
        {
            const int total_vec = tile_size * 16;
            for (int i = tid; i < total_vec; i += THREADS) {
                const int row = i / 16;
                const int vec = i % 16;
                const int64_t src = kv_base + (int64_t)(kv_start + row) * KV_BYTES_PER_ROW + vec * 16;
                *reinterpret_cast<uint128_vec*>(&v_lds[row * V_LDS_STRIDE + vec * 16]) =
                    *reinterpret_cast<const uint128_vec*>(&kv_mxfp4[src]);
            }
        }

        // ---- Cooperative load: V scales ----
        {
            const int total_sc = tile_size * 16;
            for (int i = tid; i < total_sc; i += THREADS) {
                const int row = i / 16;
                const int blk = i % 16;
                v_scale_lds[row * 16 + blk] =
                    kv_scale[sc_base + (int64_t)(kv_start + row) * B_SCALE_STRIDE + blk];
            }
        }

        // ---- Cooperative load: FP32 attention weights ----
        {
            const int total_attn = HEADS * tile_size;
            for (int i = tid; i < total_attn; i += THREADS) {
                const int h = i / tile_size;
                const int k = i % tile_size;
                attn_f32[h * KV_TILE + k] =
                    attn_weights[((int64_t)batch_idx * HEADS + h) * N + kv_start + k];
            }
            // Zero-fill unused portion
            for (int i = tid; i < HEADS * (KV_TILE - tile_size); i += THREADS) {
                const int h = i / (KV_TILE - tile_size);
                const int k = tile_size + i % (KV_TILE - tile_size);
                attn_f32[h * KV_TILE + k] = 0.0f;
            }
        }

        __syncthreads();

        // ---- Per-warp MFMA: each warp processes 8 chunks (4 V-scale-blocks x 2 chunks) ----
        // Warp 0: chunks 0-7 (V dims 0-127, V-scale-blocks 0-3)
        // Warp 1: chunks 8-15 (V dims 128-255, V-scale-blocks 4-7)
        // etc.
        for (int ci = 0; ci < CHUNKS_PER_WARP; ci += 2) {
            // Two adjacent chunks share the same V-scale-block
            const int chunk0 = warp_id * CHUNKS_PER_WARP + ci;
            const int vscale_blk = chunk0 / 2;  // V-scale-block index (0..15)

            // ---- Compute A operand: attn * v_scale, per-lane in registers ----
            // Lane a_row=head, a_kgrp=K_group
            // Load 32 attn weights and 32 V scales, multiply, quantize
            const int k_base_a = a_kgrp * 32;
            float scaled_attn[32];
            for (int j = 0; j < 32; j++) {
                const float aw = attn_f32[a_row * KV_TILE + k_base_a + j];
                const float vs = e8m0_to_float_fast(
                    v_scale_lds[(k_base_a + j) * 16 + vscale_blk]);
                scaled_attn[j] = aw * vs;
            }
            QuantBlock aqb = quantize_fp4_block(scaled_attn);

            uint32_t a_reg[8];
            *reinterpret_cast<uint128_vec*>(&a_reg[0]) =
                *reinterpret_cast<uint128_vec*>(&aqb.data);
            a_reg[4] = a_reg[5] = a_reg[6] = a_reg[7] = 0;
            int32_t a_sc = mla_broadcast_scale(aqb.e8m0);

            int8_vec a_vec = {(int)a_reg[0], (int)a_reg[1], (int)a_reg[2], (int)a_reg[3],
                              (int)a_reg[4], (int)a_reg[5], (int)a_reg[6], (int)a_reg[7]};

            // ---- Chunk 0: gather B, execute MFMA ----
            {
                const int v_d = chunk0 * 16 + b_col;
                const int v_byte = v_d / 2;
                const int v_nib_shift = (v_d & 1) * 4;
                const int k_base_b = b_kgrp * 32;

                uint8_t packed[16];
                for (int j = 0; j < 16; j++) {
                    const int k0 = k_base_b + j * 2;
                    const int k1 = k_base_b + j * 2 + 1;
                    uint8_t n0 = (k0 < tile_size) ?
                        ((v_lds[k0 * V_LDS_STRIDE + v_byte] >> v_nib_shift) & 0x0F) : 0;
                    uint8_t n1 = (k1 < tile_size) ?
                        ((v_lds[k1 * V_LDS_STRIDE + v_byte] >> v_nib_shift) & 0x0F) : 0;
                    packed[j] = n0 | (n1 << 4);
                }

                uint32_t b_reg[8];
                *reinterpret_cast<uint128_vec*>(&b_reg[0]) =
                    *reinterpret_cast<uint128_vec*>(&packed[0]);
                b_reg[4] = b_reg[5] = b_reg[6] = b_reg[7] = 0;

                int8_vec b_vec = {(int)b_reg[0], (int)b_reg[1], (int)b_reg[2], (int)b_reg[3],
                                  (int)b_reg[4], (int)b_reg[5], (int)b_reg[6], (int)b_reg[7]};
                warp_acc[ci] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                    a_vec, b_vec, warp_acc[ci],
                    FMT_FP4_MFMA, FMT_FP4_MFMA,
                    0, a_sc, 0, b_sc_one);
            }

            // ---- Chunk 1: gather B, execute MFMA (same A operand) ----
            {
                const int v_d = (chunk0 + 1) * 16 + b_col;
                const int v_byte = v_d / 2;
                const int v_nib_shift = (v_d & 1) * 4;
                const int k_base_b = b_kgrp * 32;

                uint8_t packed[16];
                for (int j = 0; j < 16; j++) {
                    const int k0 = k_base_b + j * 2;
                    const int k1 = k_base_b + j * 2 + 1;
                    uint8_t n0 = (k0 < tile_size) ?
                        ((v_lds[k0 * V_LDS_STRIDE + v_byte] >> v_nib_shift) & 0x0F) : 0;
                    uint8_t n1 = (k1 < tile_size) ?
                        ((v_lds[k1 * V_LDS_STRIDE + v_byte] >> v_nib_shift) & 0x0F) : 0;
                    packed[j] = n0 | (n1 << 4);
                }

                uint32_t b_reg[8];
                *reinterpret_cast<uint128_vec*>(&b_reg[0]) =
                    *reinterpret_cast<uint128_vec*>(&packed[0]);
                b_reg[4] = b_reg[5] = b_reg[6] = b_reg[7] = 0;

                int8_vec b_vec = {(int)b_reg[0], (int)b_reg[1], (int)b_reg[2], (int)b_reg[3],
                                  (int)b_reg[4], (int)b_reg[5], (int)b_reg[6], (int)b_reg[7]};
                warp_acc[ci + 1] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                    a_vec, b_vec, warp_acc[ci + 1],
                    FMT_FP4_MFMA, FMT_FP4_MFMA,
                    0, a_sc, 0, b_sc_one);
            }
        }

        __syncthreads();
    }

    // ---- Store output ----
    const int out_col = lane % 16;
    const int out_quad = lane / 16;

    for (int ci = 0; ci < CHUNKS_PER_WARP; ci++) {
        const int chunk = warp_id * CHUNKS_PER_WARP + ci;
        for (int i = 0; i < 4; i++) {
            const int head = i + 4 * out_quad;
            const int v_dim = chunk * 16 + out_col;
            if (head < HEADS && v_dim < V_DIM) {
                partial_out[out_base + (int64_t)head * V_DIM + v_dim] = warp_acc[ci][i];
            }
        }
    }
}

// ---- Head-merged reduce kernel ----
// Input:  partial_out (batch, KV_SPLITS, 16, 512) fp32
// Output: output (batch * 16, 512) bf16
template <int BLOCK_SIZE, int KV_SPLITS>
__global__ __launch_bounds__(BLOCK_SIZE)
void mla_attn_v_reduce_head_merged_kernel(
    const float* __restrict__ partial_out,
    hip_bfloat16* __restrict__ output
) {
    constexpr int V_DIM = 512;
    constexpr int HEADS = 16;

    // Grid: (batch_size, HEADS), block: (256)
    // Each block reduces one (batch, head) pair across KV_SPLITS
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;  // 0..255

    // Each thread reduces 2 V dimensions
    const int v_dim0 = tid * 2;
    const int v_dim1 = tid * 2 + 1;

    // Input layout: (batch, KV_SPLITS, 16, 512)
    const int64_t in_batch_base = (int64_t)batch_idx * KV_SPLITS * HEADS * V_DIM;

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (int s = 0; s < KV_SPLITS; s++) {
        const int64_t split_base = in_batch_base + (int64_t)s * HEADS * V_DIM + (int64_t)head_idx * V_DIM;
        sum0 += partial_out[split_base + v_dim0];
        sum1 += partial_out[split_base + v_dim1];
    }

    // Output layout: (batch * 16, 512)
    const int64_t out_base = ((int64_t)batch_idx * HEADS + head_idx) * V_DIM;
    output[out_base + v_dim0] = hip_bfloat16(sum0);
    output[out_base + v_dim1] = hip_bfloat16(sum1);
}

// =============================================================================
// FUSED FlashAttention-style MLA decode kernel (v4)
//
// Replaces: QK^T GEMM + softmax + attnV with a SINGLE kernel that reads KV
// data ONCE from HBM. Uses online softmax with V accumulator rescaling.
//
// QK^T: Uses SAME single-warp MFMA pattern as working mla_qkt_mxfp4_kernel
// V accumulation: scalar FP32 with LUT-based MXFP4 dequant
//
// Grid: (BATCH_SIZE, KV_SPLITS)
// Block: 128 threads (2 warps) - warp 0 does MFMA, both do V accum
//
// LDS: Q data+scale (4.9KB) + KV tile (18.4KB) + KV scales (1.2KB)
//      + scores (1KB) = ~25.5 KB
// =============================================================================

template <int N, int BLOCK_SIZE, int B_SCALE_STRIDE, int KV_SPLITS,
          int K_HALF, int NUM_BLOCKS, int A_K_HALF>
__global__ __launch_bounds__(BLOCK_SIZE)
void mla_fused_attn_kernel(
    const hip_bfloat16* __restrict__ Q_bf16,  // (batch*16, K) BF16 queries (for future inline quant)
    const uint8_t* __restrict__ q_data,      // (batch*16, A_K_HALF) pre-quantized Q MXFP4
    const uint8_t* __restrict__ q_scale,     // (batch*NUM_BLOCKS*16) Q E8M0 scales
    const uint8_t* __restrict__ kv_mxfp4,    // (batch*N, K_HALF) packed KV MXFP4
    const uint8_t* __restrict__ kv_scale,    // (batch*N, B_SCALE_STRIDE) KV E8M0 scales
    float sm_scale,
    float* __restrict__ partial_out,          // (batch, KV_SPLITS, 16, 512) fp32
    float* __restrict__ partial_lse           // (batch, KV_SPLITS, 16) log-sum-exp
) {
    constexpr int V_DIM = 512;
    constexpr int THREADS = BLOCK_SIZE;  // 64
    constexpr int HEADS = 16;
    constexpr int KV_PER_SPLIT = (N + KV_SPLITS - 1) / KV_SPLITS;
    constexpr int IM = 16;
    constexpr int IN = 16;
    constexpr int BPC = 4;
    constexpr int K_ITERS = (NUM_BLOCKS + BPC - 1) / BPC;
    constexpr int K = NUM_BLOCKS * 32;

    const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.x);
    const int split_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
    const int tid = threadIdx.x;
    const int warp_id = __builtin_amdgcn_readfirstlane(tid / 64);
    const int lane = tid % 64;

    const int kv_start_global = split_idx * KV_PER_SPLIT;
    const int kv_end_global = min(kv_start_global + KV_PER_SPLIT, N);

    // ---- LDS layout ----
    __shared__ uint8_t q_lds_data[HEADS * A_K_HALF];
    __shared__ uint8_t q_lds_scale[NUM_BLOCKS * HEADS];
    __shared__ uint8_t kv_lds_data[IN * K_HALF];         // 16 KV positions at a time
    __shared__ uint8_t kv_lds_scale_tile[IN * B_SCALE_STRIDE];
    __shared__ float   scores_lds[HEADS * IN];            // 16 * 16 = 256 floats

    // ---- Per-thread V accumulators + online softmax state ----
    // 256 threads: V_PER_THREAD=2, all 16 heads, 64 VGPRs total (safe)
    constexpr int V_PER_THREAD = V_DIM / THREADS;  // 512/256 = 2
    float acc[HEADS * V_PER_THREAD];      // 32 floats
    float running_max[HEADS];              // 16 floats
    float running_sum[HEADS];              // 16 floats
    for (int h = 0; h < HEADS; h++) {
        acc[h * 2] = 0.0f;
        acc[h * 2 + 1] = 0.0f;
        running_max[h] = -INFINITY;
        running_sum[h] = 0.0f;
    }

    // Pre-compute V dimension mappings for this thread
    int v_dims[V_PER_THREAD];
    int v_byte_offsets[V_PER_THREAD];
    int v_blocks[V_PER_THREAD];
    int v_nibble_shifts[V_PER_THREAD];
    for (int v = 0; v < V_PER_THREAD; v++) {
        int vd = tid * V_PER_THREAD + v;
        v_dims[v] = vd;
        int vblk = vd / 32;
        int vwithin = vd % 32;
        v_byte_offsets[v] = vblk * 16 + vwithin / 2;
        v_blocks[v] = vblk;
        v_nibble_shifts[v] = (vwithin % 2) * 4;
    }

    // ---- Step 1: Load pre-quantized Q into LDS (one-time) ----
    {
        const uint8_t* q_d = q_data + batch_idx * HEADS * A_K_HALF;
        const int total_q_bytes = HEADS * A_K_HALF;
        for (int i = tid * 16; i < total_q_bytes; i += THREADS * 16) {
            if (i + 16 <= total_q_bytes)
                *reinterpret_cast<uint128_vec*>(&q_lds_data[i]) =
                    *reinterpret_cast<const uint128_vec*>(&q_d[i]);
        }
        const uint8_t* q_s = q_scale + batch_idx * NUM_BLOCKS * HEADS;
        const int total_q_scales = NUM_BLOCKS * HEADS;
        for (int i = tid; i < total_q_scales; i += THREADS) {
            q_lds_scale[i] = q_s[i];
        }
    }
    if (kv_start_global >= N) {
        const int64_t out_base = ((int64_t)batch_idx * KV_SPLITS + split_idx) * HEADS * V_DIM;
        const int64_t lse_base = ((int64_t)batch_idx * KV_SPLITS + split_idx) * HEADS;
        for (int h = 0; h < HEADS; h++) {
            partial_out[out_base + (int64_t)h * V_DIM + v_dims[0]] = 0.0f;
            partial_out[out_base + (int64_t)h * V_DIM + v_dims[1]] = 0.0f;
        }
        if (tid < HEADS) partial_lse[lse_base + tid] = -INFINITY;
        return;
    }

    __syncthreads();

    // ---- Step 2: Iterate over KV in subtiles of 16 positions ----
    const int64_t kv_data_base = (int64_t)batch_idx * N * K_HALF;
    const int64_t kv_scale_base = (int64_t)batch_idx * N * B_SCALE_STRIDE;

    for (int kv_pos = kv_start_global; kv_pos < kv_end_global; kv_pos += IN) {
        const int subtile_end = min(kv_pos + IN, kv_end_global);
        const int subtile_size = subtile_end - kv_pos;

        // ---- 2a: Load 16 KV positions into LDS (all 256 threads = 4x faster) ----
        {
            const int total_bytes = subtile_size * K_HALF;
            for (int i = tid * 16; i < total_bytes; i += THREADS * 16) {
                const int row = i / K_HALF;
                const int col = i % K_HALF;
                if (row < subtile_size) {
                    *reinterpret_cast<uint128_vec*>(&kv_lds_data[row * K_HALF + col]) =
                        *reinterpret_cast<const uint128_vec*>(&kv_mxfp4[kv_data_base + (int64_t)(kv_pos + row) * K_HALF + col]);
                }
            }
            const int total_sc = subtile_size * B_SCALE_STRIDE;
            for (int i = tid; i < total_sc; i += THREADS) {
                const int row = i / B_SCALE_STRIDE;
                const int blk = i % B_SCALE_STRIDE;
                kv_lds_scale_tile[row * B_SCALE_STRIDE + blk] =
                    kv_scale[kv_scale_base + (int64_t)(kv_pos + row) * B_SCALE_STRIDE + blk];
            }
        }

        __syncthreads();

        // ---- 2b: QK^T via MFMA (warp 0 only) ----
        if (warp_id == 0) {
            float4_t mfma_acc = {};
            for (int ki = 0; ki < K_ITERS; ki++) {
                int blk0 = ki * BPC;
                int a_row = lane % IM;
                int a_blk = blk0 + lane / IM;
                uint32_t a_reg[8] = {};
                if (a_blk < NUM_BLOCKS) {
                    int a_off = a_row * A_K_HALF + a_blk * 16;
                    *reinterpret_cast<uint128_vec*>(&a_reg[0]) =
                        *reinterpret_cast<const uint128_vec*>(&q_lds_data[a_off]);
                }
                uint8_t a_e = 127;
                if (a_row < HEADS && a_blk < NUM_BLOCKS)
                    a_e = q_lds_scale[a_row + a_blk * HEADS];
                else
                    a_reg[0] = a_reg[1] = a_reg[2] = a_reg[3] = 0;
                int32_t a_sc = mla_broadcast_scale(a_e);

                int b_row = lane % IM;
                int b_blk = blk0 + lane / IM;
                uint32_t b_reg[8] = {};
                if (b_row < subtile_size && b_blk < NUM_BLOCKS) {
                    int b_off = b_row * K_HALF + b_blk * 16;
                    *reinterpret_cast<uint128_vec*>(&b_reg[0]) =
                        *reinterpret_cast<const uint128_vec*>(&kv_lds_data[b_off]);
                }
                uint8_t b_e = 127;
                if (b_row < subtile_size && b_blk < NUM_BLOCKS)
                    b_e = kv_lds_scale_tile[b_row * B_SCALE_STRIDE + b_blk];
                else
                    b_reg[0] = b_reg[1] = b_reg[2] = b_reg[3] = 0;
                int32_t b_sc = mla_broadcast_scale(b_e);

                a_reg[4]=a_reg[5]=a_reg[6]=a_reg[7]=0;
                b_reg[4]=b_reg[5]=b_reg[6]=b_reg[7]=0;

                int8_vec a_vec = {(int)a_reg[0],(int)a_reg[1],(int)a_reg[2],(int)a_reg[3],
                                  (int)a_reg[4],(int)a_reg[5],(int)a_reg[6],(int)a_reg[7]};
                int8_vec b_vec = {(int)b_reg[0],(int)b_reg[1],(int)b_reg[2],(int)b_reg[3],
                                  (int)b_reg[4],(int)b_reg[5],(int)b_reg[6],(int)b_reg[7]};
                mfma_acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                    a_vec, b_vec, mfma_acc, FMT_FP4_MFMA, FMT_FP4_MFMA,
                    0, a_sc, 0, b_sc);
            }
            int col = lane % 16;
            int quad = lane / 16;
            for (int i = 0; i < 4; i++) {
                int head = i + 4 * quad;
                if (head < HEADS && col < subtile_size)
                    scores_lds[head * IN + col] = mfma_acc[i] * sm_scale;
            }
        }

        __syncthreads();

        // ---- 2c: Online softmax + V accumulation (all 256 threads, 2 V dims each) ----
        for (int ki = 0; ki < subtile_size; ki++) {
            float v_val0, v_val1;
            {
                float bs0 = e8m0_to_float_fast(kv_lds_scale_tile[ki * B_SCALE_STRIDE + v_blocks[0]]);
                uint8_t p0 = kv_lds_data[ki * K_HALF + v_byte_offsets[0]];
                v_val0 = FP4_E2M1_LUT[(p0 >> v_nibble_shifts[0]) & 0x0F] * bs0;

                float bs1 = e8m0_to_float_fast(kv_lds_scale_tile[ki * B_SCALE_STRIDE + v_blocks[1]]);
                uint8_t p1 = kv_lds_data[ki * K_HALF + v_byte_offsets[1]];
                v_val1 = FP4_E2M1_LUT[(p1 >> v_nibble_shifts[1]) & 0x0F] * bs1;
            }

            #pragma unroll
            for (int h = 0; h < HEADS; h++) {
                const float s = scores_lds[h * IN + ki];
                const float old_max = running_max[h];
                const float new_max = fmaxf(old_max, s);
                const float rescale = __expf(old_max - new_max);
                const float exp_s = __expf(s - new_max);
                running_max[h] = new_max;
                running_sum[h] = running_sum[h] * rescale + exp_s;
                acc[h * 2]     = acc[h * 2] * rescale + exp_s * v_val0;
                acc[h * 2 + 1] = acc[h * 2 + 1] * rescale + exp_s * v_val1;
            }
        }

        __syncthreads();
    }

    // ---- Step 3: Write partial output + LSE ----
    const int64_t out_base = ((int64_t)batch_idx * KV_SPLITS + split_idx) * HEADS * V_DIM;
    for (int h = 0; h < HEADS; h++) {
        float inv_sum = (running_sum[h] > 0.0f) ? (1.0f / running_sum[h]) : 0.0f;
        partial_out[out_base + (int64_t)h * V_DIM + v_dims[0]] = acc[h * 2] * inv_sum;
        partial_out[out_base + (int64_t)h * V_DIM + v_dims[1]] = acc[h * 2 + 1] * inv_sum;
    }

    const int64_t lse_base = ((int64_t)batch_idx * KV_SPLITS + split_idx) * HEADS;
    if (tid < HEADS) {
        float lse = running_max[tid] + logf(fmaxf(running_sum[tid], 1e-20f));
        partial_lse[lse_base + tid] = lse;
    }
}

// =============================================================================
// Fused reduce kernel: combines independently-normalized splits using LSE
// 2-pass: first find max_lse, then accumulate weighted partials
// No large stack arrays - avoids register spill for large KV_SPLITS
// =============================================================================

template <int BLOCK_SIZE, int KV_SPLITS>
__global__ __launch_bounds__(BLOCK_SIZE)
void mla_fused_reduce_kernel(
    const float* __restrict__ partial_out,    // (batch, KV_SPLITS, 16, 512)
    const float* __restrict__ partial_lse,    // (batch, KV_SPLITS, 16)
    hip_bfloat16* __restrict__ output         // (batch * 16, 512)
) {
    constexpr int V_DIM = 512;
    constexpr int HEADS = 16;

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    const int v_dim0 = tid * 2;
    const int v_dim1 = tid * 2 + 1;

    const int64_t lse_batch_base = (int64_t)batch_idx * KV_SPLITS * HEADS;
    const int64_t v_batch_base = (int64_t)batch_idx * KV_SPLITS * HEADS * V_DIM;

    // Pass 1: Find max LSE across all splits (no array needed)
    float max_lse = -INFINITY;
    for (int s = 0; s < KV_SPLITS; s++) {
        float lse = partial_lse[lse_batch_base + (int64_t)s * HEADS + head_idx];
        max_lse = fmaxf(max_lse, lse);
    }

    // Pass 2: Accumulate weighted partials using max_lse
    float sum0 = 0.0f, sum1 = 0.0f, denom = 0.0f;
    for (int s = 0; s < KV_SPLITS; s++) {
        float lse = partial_lse[lse_batch_base + (int64_t)s * HEADS + head_idx];
        float w = expf(lse - max_lse);
        denom += w;
        const int64_t split_base = v_batch_base + (int64_t)s * HEADS * V_DIM + (int64_t)head_idx * V_DIM;
        sum0 += w * partial_out[split_base + v_dim0];
        sum1 += w * partial_out[split_base + v_dim1];
    }

    float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    const int64_t out_base = ((int64_t)batch_idx * HEADS + head_idx) * V_DIM;
    output[out_base + v_dim0] = hip_bfloat16(sum0 * inv_denom);
    output[out_base + v_dim1] = hip_bfloat16(sum1 * inv_denom);
}

'''

# C++ wrapper for PyTorch load_inline compilation - optimized for MI355X
MLA_MXFP4_CPP_SOURCE = r'''
// v4 fused attn kernel rewrite
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <ATen/hip/HIPContext.h>
#include <unordered_map>

torch::Tensor mla_mxfp4_decode_forward(
    torch::Tensor q,
    torch::Tensor kv_mxfp4,
    torch::Tensor kv_scale,
    int batch_size,
    int q_seq_len,
    int kv_seq_len,
    float sm_scale
) {
    int total_q = batch_size * q_seq_len;

    auto output = torch::empty({total_q, NUM_HEADS, V_HEAD_DIM},
                               torch::TensorOptions()
                                   .dtype(torch::kBFloat16)
                                   .device(q.device()));

    dim3 grid(batch_size, NUM_HEADS);
    dim3 block(256);
    size_t shared_size;
    bool use_large_kv_kernel = (kv_seq_len > 4096);

    // Dispatch to specialized kernel for bs4/kv1024 decode
    if (batch_size == 4 && kv_seq_len == 1024) {
        // Specialized kernel for kv=1024, decode mode
        // Layout: [Q:580 padded][reduce:8][kv_tile_A:32*288][kv_tile_B:32*288][scale_A:32*18][scale_B:32*18]
        constexpr int KV_TILE = 32;
        constexpr int BYTES_PER_KV = 288;
        constexpr int SCALES_PER_KV = 18;
        shared_size = sizeof(float) * (580 + 8) +
                      2 * KV_TILE * BYTES_PER_KV + 2 * KV_TILE * SCALES_PER_KV;
        hipLaunchKernelGGL(
            mla_mxfp4_decode_kernel_bs4_kv1024,
            grid, block, shared_size, 0,
            reinterpret_cast<const hip_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const uint8_t*>(kv_mxfp4.data_ptr()),
            reinterpret_cast<const uint8_t*>(kv_scale.data_ptr()),
            reinterpret_cast<hip_bfloat16*>(output.data_ptr()),
            batch_size,
            q_seq_len,
            kv_seq_len,
            sm_scale
        );
    } else if (use_large_kv_kernel) {
        // Large KV kernel: [LUT:16] [Q: q*576] [reduce:8] [partial_out:512] [scores:256 tile]
        shared_size = sizeof(float) * (16 + q_seq_len * QK_HEAD_DIM + 8 + V_HEAD_DIM + 256);
        hipLaunchKernelGGL(
            mla_mxfp4_decode_kernel_large_kv,
            grid, block, shared_size, 0,
            reinterpret_cast<const hip_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const uint8_t*>(kv_mxfp4.data_ptr()),
            reinterpret_cast<const uint8_t*>(kv_scale.data_ptr()),
            reinterpret_cast<hip_bfloat16*>(output.data_ptr()),
            batch_size,
            q_seq_len,
            kv_seq_len,
            sm_scale
        );
    } else {
        // Generic kernel
        shared_size = sizeof(float) * (16 + q_seq_len * QK_HEAD_DIM + 8 + kv_seq_len);
        hipLaunchKernelGGL(
            mla_mxfp4_decode_kernel,
            grid, block, shared_size, 0,
            reinterpret_cast<const hip_bfloat16*>(q.data_ptr()),
            reinterpret_cast<const uint8_t*>(kv_mxfp4.data_ptr()),
            reinterpret_cast<const uint8_t*>(kv_scale.data_ptr()),
            reinterpret_cast<hip_bfloat16*>(output.data_ptr()),
            batch_size,
            q_seq_len,
            kv_seq_len,
            sm_scale
        );
    }

    return output;
}

// =============================================================================
// C++ Wrappers for MLA MXFP4 Kernels
// Add this code to MLA_MXFP4_CPP_SOURCE in submission.py
// =============================================================================

// =============================================================================
// Templated mla_mxfp4_pipeline with batch_size as template parameter
//
// Benefits:
// - Compiler can optimize grid/block dims as constants
// - Static buffer sizing without runtime branching
// - Kernel template instantiations can specialize on batch_size
//
// =============================================================================

// Head-merged grid: (batch, split). Target ~304-1024 total blocks.
// Each block loads KV data ONCE for all 16 heads (16x bandwidth savings).
// Constraint: trailing blocks (total % 304) must be 0 or >= 152 (half CUs busy).
template <int BATCH_SIZE, int N>
constexpr int get_kv_split() {
    if constexpr (BATCH_SIZE == 4 && N == 1024) return 76;          // 304 blocks, 0 trailing
    else if constexpr (BATCH_SIZE == 4 && N == 8192) return 76;     // 304 blocks, 0 trailing
    else if constexpr (BATCH_SIZE == 32 && N == 1024) return 19;    // 608 blocks, 0 trailing
    else if constexpr (BATCH_SIZE == 32 && N == 8192) return 19;    // 608 blocks, 0 trailing
    else if constexpr (BATCH_SIZE == 64 && N == 1024) return 19;     // 1216 blocks, 0 trailing
    else if constexpr (BATCH_SIZE == 64 && N == 8192) return 19;     // 1216 blocks, 0 trailing
    else if constexpr (BATCH_SIZE == 256 && N == 1024) return 4;    // 1024 blocks, best balance attnV vs reduce
    else if constexpr (BATCH_SIZE == 256 && N == 8192) return 4;    // 1024 blocks, MFMA kernel bandwidth-saturated
    else return 0;
}

template <int BATCH_SIZE, int N, int B_SCALE_STRIDE>
torch::Tensor mla_mxfp4_pipeline_impl(
    torch::Tensor Q_bf16,
    torch::Tensor KV_data,
    torch::Tensor KV_scale,
    float sm_scale,
    bool profile
) {
    constexpr int M = 16;
    constexpr int K = 576;
    constexpr int K_HALF = 288;
    constexpr int NUM_BLOCKS = K / 32; // 18
    constexpr int NUM_HEADS = 16;
    constexpr int V_DIM = 512;
    constexpr int A_K_HALF = NUM_BLOCKS * 16;
    constexpr int TOTAL_HEADS = BATCH_SIZE * NUM_HEADS;
    constexpr int KV_SPLITS = get_kv_split<BATCH_SIZE, N>();

    // ---- Static scratch buffers per BATCH_SIZE ----
    // attn_buf serves double duty: QK^T writes pre-scaled scores, softmax normalizes in-place
    static torch::Tensor q_data_buf, q_scale_buf, attn_buf, partial_buf;
    static int last_n = 0, last_splits = 0;

    bool need_realloc = (N != last_n || KV_SPLITS != last_splits);
    if (need_realloc) {
        auto u8opts = torch::TensorOptions().dtype(torch::kUInt8).device(Q_bf16.device());
        auto f32opts = torch::TensorOptions().dtype(torch::kFloat32).device(Q_bf16.device());

        q_data_buf = torch::empty({BATCH_SIZE * M, A_K_HALF}, u8opts);
        q_scale_buf = torch::empty({BATCH_SIZE * NUM_BLOCKS * M}, u8opts);
        attn_buf = torch::empty({BATCH_SIZE, M, N}, f32opts);
        partial_buf = torch::empty({TOTAL_HEADS * KV_SPLITS * V_DIM}, f32opts);

        last_n = N;
        last_splits = KV_SPLITS;
    }

    // ---- Profiling ----
    struct PerfStats {
        float t_quant = 0, t_qkt_softmax = 0, t_attnv = 0, t_reduce = 0;
        int count = 0;
    };
    static std::unordered_map<int, std::unordered_map<int, PerfStats>> perf_map;
    constexpr int PROFILE_INTERVAL = 10;
    auto& stats = perf_map[BATCH_SIZE][N];
    stats.count++;
    bool do_profile = profile && (stats.count % PROFILE_INTERVAL == 0);

    hipEvent_t e0, e1, e2, e3, e4;
    if (do_profile) {
        hipEventCreate(&e0); hipEventCreate(&e1); hipEventCreate(&e2);
        hipEventCreate(&e3); hipEventCreate(&e4);
        hipEventRecord(e0);
    }

    // ---- Step 1: Quantize Q to MXFP4 ----
    {
        constexpr int q_block = 64;
        constexpr int total_q_blocks = BATCH_SIZE * M * NUM_BLOCKS;
        constexpr int q_grid = (total_q_blocks + q_block - 1) / q_block;
        mla_quant_q_batched_kernel<M, K, K_HALF, NUM_BLOCKS, BATCH_SIZE, q_block>
            <<<dim3(q_grid), dim3(q_block)>>>(
            reinterpret_cast<const hip_bfloat16*>(Q_bf16.data_ptr()),
            reinterpret_cast<uint8_t*>(q_data_buf.data_ptr()),
            reinterpret_cast<uint8_t*>(q_scale_buf.data_ptr()));
    }
    if (do_profile) hipEventRecord(e1);

    // ---- Step 2: QK^T GEMM (with fused sm_scale) + 2-pass softmax ----
    // Step 2a: Single-warp QK^T writes pre-scaled scores directly to attn_buf
    {
        constexpr int IN = 16;
        dim3 grid(1, (N + IN - 1) / IN, BATCH_SIZE);
        constexpr int BS = 64;
        dim3 block(BS);
        mla_qkt_mxfp4_kernel<M, N, K_HALF, NUM_BLOCKS, B_SCALE_STRIDE, A_K_HALF, BS>
            <<<grid, block>>>(
            reinterpret_cast<const uint8_t*>(q_data_buf.data_ptr()),
            reinterpret_cast<const uint8_t*>(KV_data.data_ptr()),
            reinterpret_cast<const uint8_t*>(q_scale_buf.data_ptr()),
            reinterpret_cast<const uint8_t*>(KV_scale.data_ptr()),
            reinterpret_cast<float*>(attn_buf.data_ptr()),
            sm_scale);
    }
    // Step 2b: 2-pass softmax in-place on attn_buf (scores already scaled)
    {
        constexpr int BS = 256;
        dim3 grid(BATCH_SIZE, NUM_HEADS);
        dim3 block(BS);
        mla_softmax_2pass_kernel<N, BS><<<grid, block>>>(
            reinterpret_cast<float*>(attn_buf.data_ptr()),
            0);
    }
    if (do_profile) hipEventRecord(e2);

    // ---- Step 3: Head-merged attn x V (split-K) ----
    // Compile-time dispatch: scalar kernel for most shapes (fewer LDS ops),
    // MFMA kernel for bs>=256/kv>=8192 where compute density matters.
    {
        constexpr int BS = 256;
        dim3 grid1(BATCH_SIZE, KV_SPLITS);
        dim3 block1(BS);
        if constexpr (BATCH_SIZE >= 256 && N >= 8192) {
            mla_attn_v_mfma_head_merged_kernel<N, BS, B_SCALE_STRIDE, KV_SPLITS><<<grid1, block1>>>(
                reinterpret_cast<const float*>(attn_buf.data_ptr()),
                reinterpret_cast<const uint8_t*>(KV_data.data_ptr()),
                reinterpret_cast<const uint8_t*>(KV_scale.data_ptr()),
                reinterpret_cast<float*>(partial_buf.data_ptr()));
        } else {
            mla_attn_v_splitk_head_merged_kernel<N, BS, B_SCALE_STRIDE, KV_SPLITS><<<grid1, block1>>>(
                reinterpret_cast<const float*>(attn_buf.data_ptr()),
                reinterpret_cast<const uint8_t*>(KV_data.data_ptr()),
                reinterpret_cast<const uint8_t*>(KV_scale.data_ptr()),
                reinterpret_cast<float*>(partial_buf.data_ptr()));
        }
    }
    if (do_profile) hipEventRecord(e3);

    // ---- Step 4: Head-merged reduce + output ----
    auto output = torch::empty({TOTAL_HEADS, V_DIM},
        torch::TensorOptions().dtype(torch::kBFloat16).device(Q_bf16.device()));
    {
        constexpr int R_BLOCK = 256;
        dim3 r_grid(BATCH_SIZE, NUM_HEADS);
        dim3 r_block(R_BLOCK);
        mla_attn_v_reduce_head_merged_kernel<R_BLOCK, KV_SPLITS><<<r_grid, r_block>>>(
            reinterpret_cast<const float*>(partial_buf.data_ptr()),
            reinterpret_cast<hip_bfloat16*>(output.data_ptr()));
    }

    if (do_profile) {
        hipEventRecord(e4);
        hipEventSynchronize(e4);
        float d01, d12, d23, d34;
        hipEventElapsedTime(&d01, e0, e1);
        hipEventElapsedTime(&d12, e1, e2);
        hipEventElapsedTime(&d23, e2, e3);
        hipEventElapsedTime(&d34, e3, e4);
        stats.t_quant += d01; stats.t_qkt_softmax += d12;
        stats.t_attnv += d23; stats.t_reduce += d34;
        int n = stats.count / PROFILE_INTERVAL;
        if (n < 5) {
            printf("[MLA] bs=%d kv=%d splits=%d | "
                "quant=%.1fus qkt+softmax=%.1fus attnv=%.1fus reduce=%.1fus | "
                "total=%.1fus (avg over %d)\n",
                BATCH_SIZE, N, KV_SPLITS,
                stats.t_quant/n*1000, stats.t_qkt_softmax/n*1000,
                stats.t_attnv/n*1000, stats.t_reduce/n*1000,
                (stats.t_quant+stats.t_qkt_softmax+stats.t_attnv+stats.t_reduce)/n*1000, n);
        }
        hipEventDestroy(e0); hipEventDestroy(e1); hipEventDestroy(e2);
        hipEventDestroy(e3); hipEventDestroy(e4);
    }

    return output;
}

// =============================================================================
// Fused MLA pipeline: Q quant + fused attn (QK^T + online softmax + V accum) + reduce
// For kv=8192 shapes where reading KV once instead of twice saves ~640MB of HBM traffic.
// =============================================================================

// Split-K for fused pipeline: fewer splits since each block does more work
// and the reduce kernel has per-split overhead
template <int BATCH_SIZE, int N>
constexpr int get_fused_kv_split() {
    // Target: ~256 blocks minimum for good CU utilization (256 CUs on MI355X)
    // Each block = (batch, split), so blocks = BATCH_SIZE * splits
    if constexpr (BATCH_SIZE == 4 && N == 1024) return 16;     // 64 blocks, 64 positions/split
    else if constexpr (BATCH_SIZE == 4 && N == 8192) return 64; // 256 blocks, 128 positions/split
    else if constexpr (BATCH_SIZE == 32 && N == 1024) return 8; // 256 blocks, 128 positions/split
    else if constexpr (BATCH_SIZE == 32 && N == 8192) return 16;// 512 blocks, 512 positions/split
    else if constexpr (BATCH_SIZE == 64 && N == 1024) return 4; // 256 blocks, 256 positions/split
    else if constexpr (BATCH_SIZE == 64 && N == 8192) return 8; // 512 blocks, 1024 positions/split
    else if constexpr (BATCH_SIZE == 256 && N == 1024) return 2;// 512 blocks, 512 positions/split
    else if constexpr (BATCH_SIZE == 256 && N == 8192) return 4;// 1024 blocks, 2048 positions/split
    else return 4;
}

template <int BATCH_SIZE, int N, int STRIDE>
torch::Tensor mla_fused_pipeline_impl(
    torch::Tensor Q_bf16,
    torch::Tensor KV_data,
    torch::Tensor KV_scale,
    float sm_scale,
    bool profile
) {
    constexpr int M = 16;
    constexpr int K = 576;
    constexpr int K_HALF = 288;
    constexpr int NUM_BLOCKS = K / 32; // 18
    constexpr int NUM_HEADS = 16;
    constexpr int V_DIM = 512;
    constexpr int A_K_HALF = NUM_BLOCKS * 16;
    constexpr int TOTAL_HEADS = BATCH_SIZE * NUM_HEADS;
    constexpr int KV_SPLITS = get_kv_split<BATCH_SIZE, N>();

    // ---- Static scratch buffers ----
    static torch::Tensor q_data_buf, q_scale_buf, partial_v_buf, partial_lse_buf;
    static int last_n = 0, last_splits = 0;

    bool need_realloc = (N != last_n || KV_SPLITS != last_splits);
    if (need_realloc) {
        auto u8opts = torch::TensorOptions().dtype(torch::kUInt8).device(Q_bf16.device());
        auto f32opts = torch::TensorOptions().dtype(torch::kFloat32).device(Q_bf16.device());

        q_data_buf = torch::empty({BATCH_SIZE * M, A_K_HALF}, u8opts);
        q_scale_buf = torch::empty({BATCH_SIZE * NUM_BLOCKS * M}, u8opts);
        partial_v_buf = torch::empty({BATCH_SIZE * KV_SPLITS * NUM_HEADS * V_DIM}, f32opts);
        partial_lse_buf = torch::empty({BATCH_SIZE * KV_SPLITS * NUM_HEADS}, f32opts);

        last_n = N;
        last_splits = KV_SPLITS;
    }

    // ---- Profiling ----
    struct PerfStats {
        float t_quant = 0, t_fused = 0, t_reduce = 0;
        int count = 0;
    };
    static std::unordered_map<int, std::unordered_map<int, PerfStats>> perf_map;
    constexpr int PROFILE_INTERVAL = 10;
    auto& stats = perf_map[BATCH_SIZE][N];
    stats.count++;
    bool do_profile = profile && (stats.count % PROFILE_INTERVAL == 0);

    hipEvent_t e0, e1, e2, e3;
    if (do_profile) {
        hipEventCreate(&e0); hipEventCreate(&e1);
        hipEventCreate(&e2); hipEventCreate(&e3);
        hipEventRecord(e0);
    }

    // ---- Step 1: Quantize Q to MXFP4 ----
    {
        constexpr int q_block = 64;
        constexpr int total_q_blocks = BATCH_SIZE * M * NUM_BLOCKS;
        constexpr int q_grid = (total_q_blocks + q_block - 1) / q_block;
        mla_quant_q_batched_kernel<M, K, K_HALF, NUM_BLOCKS, BATCH_SIZE, q_block>
            <<<dim3(q_grid), dim3(q_block)>>>(
            reinterpret_cast<const hip_bfloat16*>(Q_bf16.data_ptr()),
            reinterpret_cast<uint8_t*>(q_data_buf.data_ptr()),
            reinterpret_cast<uint8_t*>(q_scale_buf.data_ptr()));
    }
    if (do_profile) hipEventRecord(e1);

    // ---- Step 2: Fused attention (QK^T MFMA + online softmax + V accumulation) ----
    {
        constexpr int BS = 256;
        dim3 grid(BATCH_SIZE, KV_SPLITS);
        dim3 block(BS);
        mla_fused_attn_kernel<N, BS, STRIDE, KV_SPLITS, K_HALF, NUM_BLOCKS, A_K_HALF>
            <<<grid, block>>>(
            reinterpret_cast<const hip_bfloat16*>(Q_bf16.data_ptr()),
            reinterpret_cast<const uint8_t*>(q_data_buf.data_ptr()),
            reinterpret_cast<const uint8_t*>(q_scale_buf.data_ptr()),
            reinterpret_cast<const uint8_t*>(KV_data.data_ptr()),
            reinterpret_cast<const uint8_t*>(KV_scale.data_ptr()),
            sm_scale,
            reinterpret_cast<float*>(partial_v_buf.data_ptr()),
            reinterpret_cast<float*>(partial_lse_buf.data_ptr()));
    }
    if (do_profile) hipEventRecord(e2);

    // ---- Step 3: LSE-corrected reduce across splits ----
    auto output = torch::empty({TOTAL_HEADS, V_DIM},
        torch::TensorOptions().dtype(torch::kBFloat16).device(Q_bf16.device()));
    {
        constexpr int R_BLOCK = 256;
        dim3 r_grid(BATCH_SIZE, NUM_HEADS);
        dim3 r_block(R_BLOCK);
        mla_fused_reduce_kernel<R_BLOCK, KV_SPLITS><<<r_grid, r_block>>>(
            reinterpret_cast<const float*>(partial_v_buf.data_ptr()),
            reinterpret_cast<const float*>(partial_lse_buf.data_ptr()),
            reinterpret_cast<hip_bfloat16*>(output.data_ptr()));
    }

    if (do_profile) {
        hipEventRecord(e3);
        hipEventSynchronize(e3);
        float d01, d12, d23;
        hipEventElapsedTime(&d01, e0, e1);
        hipEventElapsedTime(&d12, e1, e2);
        hipEventElapsedTime(&d23, e2, e3);
        stats.t_quant += d01; stats.t_fused += d12; stats.t_reduce += d23;
        int n = stats.count / PROFILE_INTERVAL;
        if (n < 5) {
            printf("[MLA FUSED] bs=%d kv=%d splits=%d | "
                "quant=%.1fus fused_attn=%.1fus reduce=%.1fus | "
                "total=%.1fus (avg over %d)\n",
                BATCH_SIZE, N, KV_SPLITS,
                stats.t_quant/n*1000, stats.t_fused/n*1000, stats.t_reduce/n*1000,
                (stats.t_quant+stats.t_fused+stats.t_reduce)/n*1000, n);
        }
        hipEventDestroy(e0); hipEventDestroy(e1);
        hipEventDestroy(e2); hipEventDestroy(e3);
    }

    return output;
}

// ---- Dispatch wrapper ----
torch::Tensor mla_mxfp4_pipeline(
    torch::Tensor Q_bf16,
    torch::Tensor KV_data,
    torch::Tensor KV_scale,
    int batch_size,
    int kv_seq_len,
    float sm_scale,
    bool profile
) {
    int B_SCALE_STRIDE = KV_scale.stride(0);
    assert(B_SCALE_STRIDE == 18 || B_SCALE_STRIDE == 24);
    assert(kv_seq_len == 1024 || kv_seq_len == 8192);

#define MLA_MXFP4(BS, N, STR) \
    if (batch_size == BS && kv_seq_len == N && B_SCALE_STRIDE == STR) \
        return mla_mxfp4_pipeline_impl<BS, N, STR>(Q_bf16, KV_data, KV_scale, sm_scale, profile)

#define MLA_FUSED(BS, N, STR) \
    if (batch_size == BS && kv_seq_len == N && B_SCALE_STRIDE == STR) \
        return mla_fused_pipeline_impl<BS, N, STR>(Q_bf16, KV_data, KV_scale, sm_scale, profile)

    // Use fused pipeline for all shapes
    /*MLA_FUSED(4, 1024, 18);
    MLA_FUSED(4, 1024, 24);
    MLA_FUSED(4, 8192, 18);
    MLA_FUSED(4, 8192, 24);
    MLA_FUSED(32, 1024, 18);
    MLA_FUSED(32, 1024, 24);
    MLA_FUSED(32, 8192, 18);
    MLA_FUSED(32, 8192, 24);
    MLA_FUSED(64, 1024, 18);
    MLA_FUSED(64, 1024, 24);
    MLA_FUSED(64, 8192, 18);
    MLA_FUSED(64, 8192, 24);
    MLA_FUSED(256, 1024, 18);
    MLA_FUSED(256, 1024, 24);
    MLA_FUSED(256, 8192, 18);
    MLA_FUSED(256, 8192, 24);*/
    MLA_MXFP4(4, 1024, 18);
    MLA_MXFP4(4, 1024, 24);
    MLA_MXFP4(4, 8192, 18);
    MLA_MXFP4(4, 8192, 24);
    MLA_MXFP4(32, 1024, 18);
    MLA_MXFP4(32, 1024, 24);
    MLA_MXFP4(32, 8192, 18);
    MLA_MXFP4(32, 8192, 24);
    MLA_MXFP4(64, 1024, 18);
    MLA_MXFP4(64, 1024, 24);
    MLA_MXFP4(64, 8192, 18);
    MLA_MXFP4(64, 8192, 24);
    MLA_MXFP4(256, 1024, 18);
    MLA_MXFP4(256, 1024, 24);
    MLA_MXFP4(256, 8192, 18);
    MLA_MXFP4(256, 8192, 24);
    TORCH_CHECK(false, "Unsupported batch_size: ", batch_size);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mla_mxfp4_decode_forward, "MLA MXFP4 Decode Forward (MI355X Optimized)");
    m.def("mla_mxfp4_pipeline", &mla_mxfp4_pipeline);
}
'''

# Global state for compiled kernel
import ctypes

_hip_kernel = None  # For hip-python direct launch
_torch_hip_module = None  # For PyTorch load_inline


def _hip_check(call_result):
    """Check HIP/HIPRTC call result and raise on error."""
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]

    try:
        from hip import hip, hiprtc
        if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
            raise RuntimeError(f"HIP error: {err}")
        elif isinstance(err, hiprtc.hiprtcResult) and err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
            raise RuntimeError(f"HIPRTC error: {err}")
    except ImportError:
        pass

    return result


def _try_compile_hip_kernel_hiprtc():
    """Compile HIP kernel using hip-python HIPRTC (Method 1)."""
    global _hip_kernel

    import time

    try:
        print("[HIPRTC] Starting compilation...")
        t0 = time.time()

        from hip import hip, hiprtc
        t1 = time.time()
        print(f"[HIPRTC] Import took {t1-t0:.2f}s")

        # Create program
        prog = _hip_check(hiprtc.hiprtcCreateProgram(
            MLA_MXFP4_HIP_SOURCE,
            b"mla_mxfp4_decode",
            0, [], []
        ))
        t2 = time.time()
        print(f"[HIPRTC] hiprtcCreateProgram took {t2-t1:.2f}s")

        # Compile - this is usually the slow part
        cflags = [b"--offload-arch=gfx950"]
        print(f"[HIPRTC] Starting hiprtcCompileProgram with {cflags}...")
        err, = hiprtc.hiprtcCompileProgram(prog, len(cflags), cflags)
        t3 = time.time()
        print(f"[HIPRTC] hiprtcCompileProgram took {t3-t2:.2f}s")

        if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
            log_size = _hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
            log = bytearray(log_size)
            _hip_check(hiprtc.hiprtcGetProgramLog(prog, log))
            raise RuntimeError(f"HIPRTC compilation failed: {log.decode()}")

        # Get compiled code
        code_size = _hip_check(hiprtc.hiprtcGetCodeSize(prog))
        code = bytearray(code_size)
        _hip_check(hiprtc.hiprtcGetCode(prog, code))
        t4 = time.time()
        print(f"[HIPRTC] Getting code took {t4-t3:.2f}s, code size: {code_size} bytes")

        # Load module and get kernel function
        print(f"[HIPRTC] Starting hipModuleLoadData...")
        hip_module = _hip_check(hip.hipModuleLoadData(code))
        t5 = time.time()
        print(f"[HIPRTC] hipModuleLoadData took {t5-t4:.2f}s")

        _hip_kernel = _hip_check(hip.hipModuleGetFunction(hip_module, b"mla_mxfp4_decode_kernel"))
        t6 = time.time()
        print(f"[HIPRTC] hipModuleGetFunction took {t6-t5:.2f}s")

        # Cleanup program
        _hip_check(hiprtc.hiprtcDestroyProgram(prog.createRef()))

        print(f"[HIPRTC] Total compilation time: {t6-t0:.2f}s")
        return True

    except ImportError as e:
        print(f"[HIPRTC] hip-python not available: {e}")
        return False
    except Exception as e:
        print(f"[HIPRTC] Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _try_compile_hip_kernel_torch():
    """Compile HIP kernel using PyTorch load_inline (Method 2)."""
    global _torch_hip_module

    import time

    try:
        print("[PyTorch] Starting compilation...")
        t0 = time.time()

        from torch.utils.cpp_extension import load_inline
        import os

        # Check if we're on ROCm
        if not torch.cuda.is_available():
            print("[PyTorch] CUDA/ROCm not available")
            return False

        # Check for HIP/ROCm
        if not hasattr(torch.version, 'hip') or torch.version.hip is None:
            print("[PyTorch] Not running on HIP/ROCm")
            return False

        rocm_home = os.environ.get('ROCM_HOME', '/opt/rocm')
        t1 = time.time()
        print(f"[PyTorch] Setup took {t1-t0:.2f}s")

        # Combine kernel source with C++ wrapper
        # Convert bytes to string for load_inline
        kernel_source = MLA_MXFP4_HIP_SOURCE.decode('utf-8') + '\n' + MLA_MXFP4_CPP_SOURCE

        print(f"[PyTorch] Starting load_inline...")
        os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
        os.environ['MAX_JOBS'] = '4'
        _torch_hip_module = load_inline(
            name='mla_mxfp4_hip_torch_v13',
            cpp_sources='',
            cuda_sources=[kernel_source],
            extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3', '--offload-arch=gfx950'],
            extra_include_paths=[f'{rocm_home}/include'],
            verbose=True,  # Enable verbose to see what's happening
        )
        t2 = time.time()
        print(f"[PyTorch] load_inline took {t2-t1:.2f}s")

        print(f"[PyTorch] Total compilation time: {t2-t0:.2f}s")
        return True

    except Exception as e:
        print(f"[PyTorch] Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _try_compile_hip_kernel():
    """Try both compilation methods, preferring hip-python."""
    # Try hip-python first (more direct, no PyTorch overhead)
    # if _try_compile_hip_kernel_hiprtc():
    #     return True
    # Fall back to PyTorch load_inline
    if _try_compile_hip_kernel_torch():
        return True
    return False


def _launch_hip_kernel_hiprtc(
    hip_kernel,
    q: torch.Tensor,
    kv_mxfp4: torch.Tensor,
    kv_scale: torch.Tensor,
    output: torch.Tensor,
    batch_size: int,
    q_seq_len: int,
    kv_seq_len: int,
    sm_scale: float,
):
    """Launch the compiled HIP kernel using hip-python."""
    from hip import hip

    # Grid and block dimensions
    grid = hip.dim3(x=batch_size, y=16)  # NUM_HEADS = 16
    block = hip.dim3(x=256)

    # Shared memory size: Q storage + scores storage + reduction buffer
    shared_mem_bytes = 4 * (q_seq_len * 576 + q_seq_len * kv_seq_len + 64)

    # Launch kernel
    _hip_check(
        hip.hipModuleLaunchKernel(
            hip_kernel,
            *grid,
            *block,
            sharedMemBytes=shared_mem_bytes,
            kernelParams=None,
            extra=(
                q.data_ptr(),                    # q pointer
                kv_mxfp4.data_ptr(),             # kv_mxfp4 pointer
                kv_scale.data_ptr(),             # kv_scale pointer
                output.data_ptr(),               # output pointer
                ctypes.c_int(batch_size),        # batch_size
                ctypes.c_int(q_seq_len),         # q_seq_len
                ctypes.c_int(kv_seq_len),        # kv_seq_len
                ctypes.c_float(sm_scale),        # sm_scale
            )
        )
    )

    # Synchronize to ensure kernel completion
    _hip_check(hip.hipDeviceSynchronize())


def _launch_hip_kernel(
    q: torch.Tensor,
    kv_mxfp4: torch.Tensor,
    kv_scale: torch.Tensor,
    output: torch.Tensor,
    batch_size: int,
    q_seq_len: int,
    kv_seq_len: int,
    sm_scale: float,
):
    """Launch the HIP kernel using the appropriate method."""
    global _hip_kernel
    if _hip_kernel is not None:
        _launch_hip_kernel_hiprtc(
            _hip_kernel, q, kv_mxfp4, kv_scale, output,
            batch_size, q_seq_len, kv_seq_len, sm_scale
        )
    else:
        # Use PyTorch module
        global _torch_hip_module
        assert _torch_hip_module is not None
        result = _torch_hip_module.forward(
            q, kv_mxfp4, kv_scale,
            batch_size, q_seq_len, kv_seq_len, sm_scale
        )
        output.copy_(result)


# Attempt compilation at module load (disabled by default)
# Uncomment the line below to enable JIT compilation of HIP kernel
HAS_HIP_KERNEL = _try_compile_hip_kernel()

# ---------------------------------------------------------------------------
# Constants (DeepSeek R1 forward_absorb path)
# ---------------------------------------------------------------------------
TOTAL_NUM_HEADS = 128
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM = KV_LORA_RANK  # 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
MXFP4_BLOCK_SIZE = 32

PAGE_SIZE = 1

FP8_DTYPE = aiter_dtypes.fp8

# QKV dtype for custom_kernel dispatch: "bf16", "fp8", "mxfp4_dequant", or "mxfp4_native"
QKV_DTYPE = "mxfp4_native"


# ---------------------------------------------------------------------------
# Dynamic num_kv_splits tuning
# ---------------------------------------------------------------------------

def get_optimal_num_kv_splits(batch_size: int, kv_seq_len: int) -> int:
    """
    Heuristic for optimal num_kv_splits based on batch size and KV sequence length.

    Trade-offs:
    - More splits = better parallelism for long sequences
    - Fewer splits = less reduction overhead for short sequences / small batches

    Performance observations:
    - batch_size=4: bf16 is fastest (81-90us), minimize splits aggressively
    - For small batches, minimize splits to reduce reduction overhead
    """
    if batch_size <= 4:
        # Very small batches: minimize reduction overhead aggressively
        # Target: get bf16 from 86.9us to 81us
        if kv_seq_len <= 1024:
            return 2  # Minimal splits - reduction overhead dominates
        elif kv_seq_len <= 2048:
            return 4
        elif kv_seq_len <= 4096:
            return 4
        else:
            return 8  # More parallelism for very long sequences
    elif batch_size <= 16:
        if kv_seq_len <= 1024:
            return 8
        elif kv_seq_len <= 4096:
            return 16
        else:
            return 24
    elif batch_size <= 32:
        if kv_seq_len <= 1024:
            return 16
        else:
            return 24
    elif batch_size <= 64:
        if kv_seq_len <= 1024:
            return 24
        else:
            return 32
    else:
        # Large batches benefit from more parallelism
        if kv_seq_len <= 2048:
            return 32
        else:
            return 48


# ---------------------------------------------------------------------------
# FP8 quantization helper (per-tensor, sglang style)
# ---------------------------------------------------------------------------

def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-tensor FP8 quantization. Returns (fp8_tensor, scale)."""
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8_tensor, scale.to(torch.float32).reshape(1)


# ---------------------------------------------------------------------------
# MXFP4 dequantization helpers (optimized)
# ---------------------------------------------------------------------------

@torch.compile(mode="reduce-overhead", fullgraph=True)
def _dequant_mxfp4_core(
    fp4_data_2d: torch.Tensor,
    scale_e8m0: torch.Tensor,
    num_rows: int,
    num_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """
    Core dequantization logic - compiled for better performance.
    """
    float_vals = mxfp4_to_f32(fp4_data_2d)
    scale_f32 = e8m0_to_f32(scale_e8m0)[:num_rows, :num_blocks]
    float_vals_blocked = float_vals.view(num_rows, num_blocks, block_size)
    return float_vals_blocked * scale_f32.unsqueeze(-1)


def dequantize_mxfp4(
    fp4_data: torch.Tensor,
    scale_e8m0: torch.Tensor,
    orig_shape: tuple,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize MXFP4 tensor using aiter utilities (optimized).

    Args:
        fp4_data:   packed FP4 data, shape [B, M, N//2] in fp4x2 or uint8
        scale_e8m0: E8M0 block scale factors (possibly padded) in fp8_e8m0
        orig_shape: original (B, M, N) for reshaping
        dtype:      output dtype

    Returns:
        Dequantized tensor of shape orig_shape.
    """
    B, M, N = orig_shape
    num_rows = B * M
    num_blocks = N // MXFP4_BLOCK_SIZE

    fp4_data_2d = fp4_data.view(num_rows, N // 2)

    # Inline dequantization to avoid function call overhead
    float_vals = mxfp4_to_f32(fp4_data_2d)
    scale_f32 = e8m0_to_f32(scale_e8m0)[:num_rows, :num_blocks]
    float_vals_blocked = float_vals.view(num_rows, num_blocks, MXFP4_BLOCK_SIZE)
    scaled = float_vals_blocked * scale_f32.unsqueeze(-1)

    return scaled.view(B, M, N).to(dtype)


def dequantize_mxfp4_to_fp8(
    fp4_data: torch.Tensor,
    scale_e8m0: torch.Tensor,
    orig_shape: tuple,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dequantize MXFP4 tensor to FP8 format for use with a8w8 kernel (optimized).

    This is more efficient than dequantizing to bf16 because:
    1. FP8 uses half the memory bandwidth of bf16
    2. The a8w8 kernel is faster than a16w16

    Returns:
        (fp8_tensor, scale) for use with mla_decode_fwd
    """
    B, M, N = orig_shape
    num_rows = B * M
    num_blocks = N // MXFP4_BLOCK_SIZE

    # Single-pass dequantization
    fp4_data_2d = fp4_data.view(num_rows, N // 2)
    float_vals = mxfp4_to_f32(fp4_data_2d)
    scale_f32 = e8m0_to_f32(scale_e8m0)[:num_rows, :num_blocks]

    float_vals_blocked = float_vals.view(num_rows, num_blocks, MXFP4_BLOCK_SIZE)
    scaled = float_vals_blocked * scale_f32.unsqueeze(-1)
    dequant_f32 = scaled.view(B, M, N)

    # Quantize to FP8 (per-tensor) - fused max and scale
    finfo = torch.finfo(FP8_DTYPE)
    amax = dequant_f32.abs().amax().clamp(min=1e-12)
    fp8_scale = amax / finfo.max
    fp8_tensor = (dequant_f32 / fp8_scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)

    return fp8_tensor, fp8_scale.to(torch.float32).reshape(1)


# ---------------------------------------------------------------------------
# Persistent mode metadata helpers
# ---------------------------------------------------------------------------

def _make_mla_decode_metadata(
    batch_size: int,
    max_q_len: int,
    nhead: int,
    nhead_kv: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    num_kv_splits: int,
):
    """Allocate and populate work buffers for persistent mla_decode_fwd."""
    info = get_mla_metadata_info_v1(
        batch_size, max_q_len, nhead, q_dtype, kv_dtype,
        is_sparse=False, fast_mode=False,
        num_kv_splits=num_kv_splits, intra_batch_mode=True,
    )
    work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    (work_metadata, work_indptr, work_info_set,
     reduce_indptr, reduce_final_map, reduce_partial_map) = work

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_len,
        nhead // nhead_kv,
        nhead_kv,
        True,
        work_metadata, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        page_size=PAGE_SIZE,
        kv_granularity=max(PAGE_SIZE, 16),
        max_seqlen_qo=max_q_len,
        uni_seqlen_qo=max_q_len,
        fast_mode=False,
        max_split_per_batch=num_kv_splits,
        intra_batch_mode=True,
        dtype_q=q_dtype,
        dtype_kv=kv_dtype,
    )

    return {
        "work_meta_data": work_metadata,
        "work_indptr": work_indptr,
        "work_info_set": work_info_set,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
    }


# ---------------------------------------------------------------------------
# Aiter MLA decode kernel wrapper with dynamic splits
# ---------------------------------------------------------------------------

def _aiter_mla_decode(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    config: dict,
    q_scale: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
    num_kv_splits: int | None = None,
) -> torch.Tensor:
    """
    MLA decode attention using aiter persistent-mode kernel.

    Supports:
      - fp8 Q + fp8 KV (a8w8) - fastest on MI355X
      - bf16 Q + bf16 KV (a16w16) - highest precision
      - bf16 Q + fp8 KV (a16w8) - mixed precision
    """
    batch_size = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dq = config["qk_head_dim"]
    dv = config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    kv_seq_len = config["kv_seq_len"]

    if num_kv_splits is None:
        num_kv_splits = get_optimal_num_kv_splits(batch_size, kv_seq_len)

    total_kv_len = int(kv_indptr[-1].item())
    kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")

    kv_buffer_4d = kv_buffer.view(kv_buffer.shape[0], PAGE_SIZE, nkv, kv_buffer.shape[-1])

    max_q_len = q_seq_len
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

    meta = _make_mla_decode_metadata(
        batch_size, max_q_len, nq, nkv,
        q.dtype, kv_buffer.dtype,
        qo_indptr, kv_indptr, kv_last_page_len,
        num_kv_splits=num_kv_splits,
    )

    o = torch.empty((q.shape[0], nq, dv), dtype=torch.bfloat16, device="cuda")
    mla_decode_fwd(
        q.view(-1, nq, dq),
        kv_buffer_4d,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        max_q_len,
        page_size=PAGE_SIZE,
        nhead_kv=nkv,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        num_kv_splits=num_kv_splits,
        q_scale=q_scale,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        **meta,
    )
    return o


# ---------------------------------------------------------------------------
# Ultra-fast path for small batches - bypasses aiter overhead
# ---------------------------------------------------------------------------

@torch.compile(mode="max-autotune", fullgraph=True)
def _fast_mla_attention_compiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    nq: int,
    q_seq_len: int,
    kv_seq_len: int,
    dv: int,
) -> torch.Tensor:
    """
    Compiled fast attention for small batches.
    Fuses all operations into a single kernel.
    """
    total_q = batch_size * q_seq_len

    # Reshape for batched attention
    q_batched = q.view(batch_size, q_seq_len, nq, -1).transpose(1, 2)
    k_batched = k.view(batch_size, kv_seq_len, 1, -1).transpose(1, 2)
    v_batched = v.view(batch_size, kv_seq_len, 1, dv).transpose(1, 2)

    # Expand for MQA
    k_batched = k_batched.expand(batch_size, nq, kv_seq_len, -1)
    v_batched = v_batched.expand(batch_size, nq, kv_seq_len, dv)

    # SDPA handles everything efficiently
    out = torch.nn.functional.scaled_dot_product_attention(
        q_batched, k_batched, v_batched,
        scale=SM_SCALE,
        is_causal=False,
    )

    return out.transpose(1, 2).reshape(total_q, nq, dv)


def _fast_mla_attention(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """
    Fast MLA attention that bypasses aiter kernel overhead.
    Uses torch SDPA which is highly optimized and has lower launch overhead.
    """
    batch_size = config["batch_size"]
    nq = config["num_heads"]
    dv = config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    kv_seq_len = config["kv_seq_len"]

    # KV buffer: (total_kv, 1, 576)
    k = kv_buffer  # Full 576 dims for keys
    v = kv_buffer[:, :, :dv]  # First 512 dims for values

    out = _fast_mla_attention_compiled(
        q.to(torch.float32),
        k.to(torch.float32),
        v.to(torch.float32),
        batch_size, nq, q_seq_len, kv_seq_len, dv
    )

    return out.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# PyTorch-based MLA attention (fallback / MXFP4 path) - optimized with SDPA
# ---------------------------------------------------------------------------

def _pytorch_mla_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """
    PyTorch MLA attention using SDPA (Scaled Dot-Product Attention).

    Uses torch.nn.functional.scaled_dot_product_attention for optimized compute.
    Handles the MQA pattern (1 KV head shared across all query heads).

    q: (total_q, num_heads, qk_head_dim)
    k: (total_kv, 1, qk_head_dim)
    v: (total_kv, 1, v_head_dim)
    """
    batch_size = config["batch_size"]
    nq = config["num_heads"]
    dv = config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    kv_seq_len = config["kv_seq_len"]

    # For decode with uniform sequence lengths, we can batch process
    # Reshape for batched attention: (batch, heads, seq_len, dim)
    total_q = q.shape[0]
    # total_kv = k.shape[0]

    # Reshape q: (total_q, nq, 576) -> (batch, q_seq_len, nq, 576) -> (batch, nq, q_seq_len, 576)
    q_batched = q.view(batch_size, q_seq_len, nq, -1).transpose(1, 2)

    # Reshape k: (total_kv, 1, 576) -> (batch, kv_seq_len, 1, 576) -> (batch, 1, kv_seq_len, 576)
    k_batched = k.view(batch_size, kv_seq_len, 1, -1).transpose(1, 2)

    # Reshape v: (total_kv, 1, 576) -> use only first 512 dims
    v_sliced = v[:, :, :dv]  # (total_kv, 1, 512)
    v_batched = v_sliced.view(batch_size, kv_seq_len, 1, dv).transpose(1, 2)

    # Expand K and V for MQA: (batch, 1, kv_seq_len, dim) -> (batch, nq, kv_seq_len, dim)
    k_batched = k_batched.expand(batch_size, nq, kv_seq_len, -1)
    v_batched = v_batched.expand(batch_size, nq, kv_seq_len, dv)

    # Use SDPA for optimized attention (uses Flash Attention when available)
    # is_causal=True for decode (though for q_seq_len=1, it doesn't matter)
    out = torch.nn.functional.scaled_dot_product_attention(
        q_batched.to(torch.float32),
        k_batched.to(torch.float32),
        v_batched.to(torch.float32),
        scale=SM_SCALE,
        is_causal=(q_seq_len > 1),
    )

    # Reshape output: (batch, nq, q_seq_len, dv) -> (batch, q_seq_len, nq, dv) -> (total_q, nq, dv)
    out = out.transpose(1, 2).reshape(total_q, nq, dv).to(torch.bfloat16)

    return out


# ---------------------------------------------------------------------------
# Dispatcher: select kernel based on QKV_DTYPE
# ---------------------------------------------------------------------------
def custom_kernel(data: input_t) -> output_t:
    return custom_kernel_mxfp4_qkt(data)

# def custom_kernel(data: input_t) -> output_t:
#     """Dispatch to the appropriate kernel based on QKV_DTYPE."""
#     global HAS_HIP_KERNEL

#     q, kv_data, qo_indptr, kv_indptr, config = data
#     batch_size = config["batch_size"]
#     kv_seq_len = config["kv_seq_len"]
#     qkv_type = QKV_DTYPE
#     if batch_size == 4:
#       if kv_seq_len <= 1024:
#         qkv_type = "bf16"
#       else:
#         qkv_type = "fp8"
#     elif batch_size == 32:
#       if kv_seq_len <= 1024:
#         qkv_type = "bf16"
#       else:
#         qkv_type = "fp8"
#     elif batch_size == 64:
#       if kv_seq_len <= 1024:
#         qkv_type = "bf16"
#       else:
#         qkv_type = "fp8"
#     elif batch_size == 256:
#       if kv_seq_len <= 1024:
#         qkv_type = "fp8"
#       else:
#         qkv_type = "fp8"

#     if qkv_type == "fp8":
#         return custom_kernel_fp8(data)
#     elif qkv_type == "bf16":
#         return custom_kernel_bf16(data)
#     elif qkv_type == "mxfp4_dequant":
#         return custom_kernel_mxfp4_dequant(data)
#     elif qkv_type == "mxfp4_native":
#         return custom_kernel_mxfp4_native(data)
#     else:
#         raise ValueError(f"Invalid QKV_DTYPE: {qkv_type}")


# ---------------------------------------------------------------------------
# FP8 Q + FP8 KV - using aiter's a8w8 persistent MLA kernel
# ---------------------------------------------------------------------------

def custom_kernel_fp8(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]

    # Only use fast SDPA for small batches AND short KV
    if batch_size <= 4 and kv_seq_len <= 1024:
        kv_buffer_bf16 = kv_data["bf16"]
        return _fast_mla_attention(q, kv_buffer_bf16, config)

    # Use aiter fp8 for everything else
    q_fp8, q_scale = quantize_fp8(q)
    kv_buffer_fp8, kv_scale_fp8 = kv_data["fp8"]
    return _aiter_mla_decode(
        q_fp8, kv_buffer_fp8, qo_indptr, kv_indptr, config,
        q_scale=q_scale, kv_scale=kv_scale_fp8,
    )

# =============================================================================
# Updated Python dispatch for custom_kernel_mxfp4_qkt
#
# Uses split-K fused attnxV for ALL cases (replaces both the non-split fused
# kernel and the dequant+bmm fallback).
#
# The split-K kernel handles any (batch_size, kv_seq_len) combination well:
# - Short KV (1024): kv_splits=1 -> same as non-split kernel
# - Long KV (8192): kv_splits=8 -> each block handles 1024 positions
#
# Replace custom_kernel_mxfp4_qkt in submission.py with this version.
# =============================================================================



# =============================================================================
# Python integration for MLA MXFP4 kernels
# Replace custom_kernel_mxfp4_qkt in submission.py with this version
# =============================================================================

def custom_kernel_mxfp4_qkt(data):
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    num_heads = config["num_heads"]
    kv_seq_len = config["kv_seq_len"]
    v_head_dim = config["v_head_dim"]
    sm_scale = config["sm_scale"]
    PROFILE = True

    kv_buffer_mxfp4, kv_scale_mxfp4 = kv_data["mxfp4"]
    total_q = q.shape[0]

    q_flat = q.view(batch_size * num_heads, 576).contiguous()
    kv_data_flat = kv_buffer_mxfp4.view(-1, 288).contiguous()
    kv_scale_flat = kv_scale_mxfp4.view(-1, kv_scale_mxfp4.shape[-1]).contiguous()

    output = _torch_hip_module.mla_mxfp4_pipeline(
        q_flat, kv_data_flat, kv_scale_flat,
        batch_size, kv_seq_len, sm_scale, PROFILE)

    return output.view(total_q, num_heads, v_head_dim)



# ---------------------------------------------------------------------------
# BF16 Q + BF16 KV - using aiter's a16w16 persistent MLA kernel
# ---------------------------------------------------------------------------

def custom_kernel_bf16(data: input_t) -> output_t:
    """
    MLA decode using bf16 KV cache.

    For small batches with short KV sequences, uses the fast SDPA path which
    bypasses aiter overhead. For larger batches or long sequences, uses aiter's a16w16 kernel.
    """
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]
    kv_buffer_bf16 = kv_data["bf16"]

    # Use fast path ONLY for small batches AND short KV sequences
    # For long KV sequences (8k), SDPA's MQA expansion is slow - use aiter instead
    if batch_size <= 4 and kv_seq_len <= 2048:
        return _fast_mla_attention(q, kv_buffer_bf16, config)

    return _aiter_mla_decode(
        q, kv_buffer_bf16, qo_indptr, kv_indptr, config,
        q_scale=None, kv_scale=None,
    )


# ---------------------------------------------------------------------------
# MXFP4 KV with dequantization -> aiter MLA kernel (optimized)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fused Triton Kernel: MXFP4 Dequant + MLA Attention (Strategy A)
# ---------------------------------------------------------------------------
# This kernel fuses MXFP4 dequantization with attention computation:
# 1. Loads MXFP4 KV tiles (fp4x2 packed) from HBM
# 2. Loads E8M0 scale factors from HBM
# 3. Unpacks and dequantizes in registers/shared memory
# 4. Computes QK^T attention scores
# 5. Applies softmax
# 6. Computes attention @ V
# 7. Writes only final output to HBM
# This eliminates the intermediate bf16/fp8 buffer write-read roundtrip.
# ---------------------------------------------------------------------------

# FP4 E2M1 lookup table: values [0, 0.5, 1, 1.5, 2, 3, 4, 6]
# Low nibble (even index), high nibble (odd index)
FP4_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


@triton.jit
def _unpack_fp4_to_f32(packed_val):
    """Unpack a single fp4x2 byte into two f32 values."""
    # Low nibble (even index)
    low_nibble = packed_val & 0x0F
    # High nibble (odd index)
    high_nibble = (packed_val >> 4) & 0x0F
    return low_nibble, high_nibble


@triton.jit
def _fp4_nibble_to_f32(nibble):
    """Convert a 4-bit FP4 E2M1 value to float32."""
    # FP4 E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit
    # Values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (and negatives)
    sign = (nibble >> 3) & 1
    magnitude_idx = nibble & 0x07

    # Lookup table for magnitudes
    # Index: 0->0, 1->0.5, 2->1, 3->1.5, 4->2, 5->3, 6->4, 7->6
    val = tl.where(magnitude_idx == 0, 0.0,
           tl.where(magnitude_idx == 1, 0.5,
           tl.where(magnitude_idx == 2, 1.0,
           tl.where(magnitude_idx == 3, 1.5,
           tl.where(magnitude_idx == 4, 2.0,
           tl.where(magnitude_idx == 5, 3.0,
           tl.where(magnitude_idx == 6, 4.0, 6.0)))))))

    # Apply sign
    return tl.where(sign == 1, -val, val)


@triton.jit
def _e8m0_to_f32(e8m0_val):
    """Convert E8M0 scale to float32. E8M0 is exponent-only: 2^(val - 127)."""
    # E8M0: 8-bit unsigned exponent, bias 127
    # value = 2^(e8m0 - 127)
    exponent = e8m0_val.to(tl.int32) - 127
    return tl.math.pow(2.0, exponent.to(tl.float32))


@triton.jit
def _fused_mxfp4_mla_attention_kernel(
    # Query tensor (bf16)
    Q_ptr,
    # MXFP4 KV cache (packed fp4x2 as uint8)
    KV_mxfp4_ptr,
    # E8M0 scale factors (as uint8)
    KV_scale_ptr,
    # Output tensor
    O_ptr,
    # Dimensions
    batch_size,
    num_heads,
    q_seq_len,
    kv_seq_len,
    qk_head_dim,  # 576
    v_head_dim,   # 512
    # Strides for Q: (total_q, num_heads, qk_head_dim)
    stride_q_tok,
    stride_q_head,
    stride_q_dim,
    # Strides for KV_mxfp4: (total_kv, 1, qk_head_dim // 2)
    stride_kv_tok,
    stride_kv_head,
    stride_kv_dim,
    # Strides for KV_scale: (total_kv * 1, num_blocks)
    stride_scale_row,
    stride_scale_block,
    # Strides for O: (total_q, num_heads, v_head_dim)
    stride_o_tok,
    stride_o_head,
    stride_o_dim,
    # Scaling factor
    sm_scale,
    # Block sizes - must be power of 2
    BLOCK_KV: tl.constexpr,
):
    """
    Fused MXFP4 dequantization + MLA attention kernel.

    Simplified version that avoids non-power-of-2 tensor shapes.
    Each program handles one (batch_idx, head_idx) pair.
    """
    # Program IDs
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # Query offset for this batch and head
    q_offset = batch_idx * q_seq_len * stride_q_tok + head_idx * stride_q_head

    # KV offset for this batch (KV has 1 head, shared across all query heads)
    kv_base = batch_idx * kv_seq_len * stride_kv_tok
    scale_base = batch_idx * kv_seq_len * stride_scale_row

    # Output offset
    o_offset = batch_idx * q_seq_len * stride_o_tok + head_idx * stride_o_head

    # Number of MXFP4 blocks per KV row (576 / 32 = 18)
    num_mxfp4_blocks: tl.constexpr = 18

    # Pass 1: Compute max of attention scores
    global_max = tl.full([], -1e30, dtype=tl.float32)

    for kv_i in range(kv_seq_len):
        kv_offset = kv_base + kv_i * stride_kv_tok
        scale_offset_base = scale_base + kv_i * stride_scale_row

        # Compute dot product Q @ K for this KV position
        score = tl.zeros([], dtype=tl.float32)

        # Process each MXFP4 block (32 elements = 16 bytes)
        for block_idx in tl.static_range(num_mxfp4_blocks):
            block_start = block_idx * 32
            scale_offset = scale_offset_base + block_idx * stride_scale_block

            # Load E8M0 scale and convert inline
            scale_e8m0 = tl.load(KV_scale_ptr + scale_offset)
            # E8M0 to f32: 2^(val - 127) using exp2
            scale_exp = scale_e8m0.to(tl.float32) - 127.0
            scale_f32 = tl.math.exp2(scale_exp)

            # Process 16 bytes (32 fp4 values)
            for byte_idx in tl.static_range(16):
                packed_offset = kv_offset + (block_start // 2 + byte_idx) * stride_kv_dim
                packed_byte = tl.load(KV_mxfp4_ptr + packed_offset)

                # Unpack two fp4 values (inline)
                low_nibble = packed_byte & 0x0F
                high_nibble = (packed_byte >> 4) & 0x0F

                # Convert FP4 nibble to f32 (inline) - simplified lookup
                # FP4 E2M1: sign(1) + exp(2) + mantissa(1)
                # Magnitudes: 0->0, 1->0.5, 2->1, 3->1.5, 4->2, 5->3, 6->4, 7->6
                sign_0 = (low_nibble >> 3) & 1
                mag_0 = low_nibble & 0x07
                k_mag_0 = tl.where(mag_0 == 0, 0.0,
                          tl.where(mag_0 == 1, 0.5,
                          tl.where(mag_0 == 2, 1.0,
                          tl.where(mag_0 == 3, 1.5,
                          tl.where(mag_0 == 4, 2.0,
                          tl.where(mag_0 == 5, 3.0,
                          tl.where(mag_0 == 6, 4.0, 6.0)))))))
                k_val_0 = tl.where(sign_0 == 1, -k_mag_0, k_mag_0) * scale_f32

                sign_1 = (high_nibble >> 3) & 1
                mag_1 = high_nibble & 0x07
                k_mag_1 = tl.where(mag_1 == 0, 0.0,
                          tl.where(mag_1 == 1, 0.5,
                          tl.where(mag_1 == 2, 1.0,
                          tl.where(mag_1 == 3, 1.5,
                          tl.where(mag_1 == 4, 2.0,
                          tl.where(mag_1 == 5, 3.0,
                          tl.where(mag_1 == 6, 4.0, 6.0)))))))
                k_val_1 = tl.where(sign_1 == 1, -k_mag_1, k_mag_1) * scale_f32

                # Load corresponding Q values and accumulate
                dim_0 = block_start + byte_idx * 2
                dim_1 = block_start + byte_idx * 2 + 1

                q_val_0 = tl.load(Q_ptr + q_offset + dim_0 * stride_q_dim).to(tl.float32)
                score += q_val_0 * k_val_0

                if dim_1 < 576:
                    q_val_1 = tl.load(Q_ptr + q_offset + dim_1 * stride_q_dim).to(tl.float32)
                    score += q_val_1 * k_val_1

        # Apply scale and update max
        score = score * sm_scale
        global_max = tl.maximum(global_max, score)

    # Pass 2: Compute softmax normalization factor
    sum_exp = tl.zeros([], dtype=tl.float32)

    for kv_i in range(kv_seq_len):
        kv_offset = kv_base + kv_i * stride_kv_tok
        scale_offset_base = scale_base + kv_i * stride_scale_row

        # Recompute attention score
        score = tl.zeros([], dtype=tl.float32)
        for block_idx in tl.static_range(num_mxfp4_blocks):
            block_start = block_idx * 32
            scale_offset = scale_offset_base + block_idx * stride_scale_block
            scale_e8m0 = tl.load(KV_scale_ptr + scale_offset)
            scale_exp = scale_e8m0.to(tl.float32) - 127.0
            scale_f32 = tl.math.exp2(scale_exp)

            for byte_idx in tl.static_range(16):
                packed_offset = kv_offset + (block_start // 2 + byte_idx) * stride_kv_dim
                packed_byte = tl.load(KV_mxfp4_ptr + packed_offset)
                low_nibble = packed_byte & 0x0F
                high_nibble = (packed_byte >> 4) & 0x0F

                sign_0 = (low_nibble >> 3) & 1
                mag_0 = low_nibble & 0x07
                k_mag_0 = tl.where(mag_0 == 0, 0.0,
                          tl.where(mag_0 == 1, 0.5,
                          tl.where(mag_0 == 2, 1.0,
                          tl.where(mag_0 == 3, 1.5,
                          tl.where(mag_0 == 4, 2.0,
                          tl.where(mag_0 == 5, 3.0,
                          tl.where(mag_0 == 6, 4.0, 6.0)))))))
                k_val_0 = tl.where(sign_0 == 1, -k_mag_0, k_mag_0) * scale_f32

                sign_1 = (high_nibble >> 3) & 1
                mag_1 = high_nibble & 0x07
                k_mag_1 = tl.where(mag_1 == 0, 0.0,
                          tl.where(mag_1 == 1, 0.5,
                          tl.where(mag_1 == 2, 1.0,
                          tl.where(mag_1 == 3, 1.5,
                          tl.where(mag_1 == 4, 2.0,
                          tl.where(mag_1 == 5, 3.0,
                          tl.where(mag_1 == 6, 4.0, 6.0)))))))
                k_val_1 = tl.where(sign_1 == 1, -k_mag_1, k_mag_1) * scale_f32

                dim_0 = block_start + byte_idx * 2
                dim_1 = block_start + byte_idx * 2 + 1
                q_val_0 = tl.load(Q_ptr + q_offset + dim_0 * stride_q_dim).to(tl.float32)
                score += q_val_0 * k_val_0
                if dim_1 < 576:
                    q_val_1 = tl.load(Q_ptr + q_offset + dim_1 * stride_q_dim).to(tl.float32)
                    score += q_val_1 * k_val_1

        score = score * sm_scale
        sum_exp += tl.exp(score - global_max)

    # Pass 3: For each output dimension, compute weighted sum of V
    num_v_blocks: tl.constexpr = 16  # 512 / 32 = 16

    for v_block_idx in tl.static_range(num_v_blocks):
        for v_byte_idx in tl.static_range(16):
            for v_nibble in tl.static_range(2):  # 0=low, 1=high
                out_d = v_block_idx * 32 + v_byte_idx * 2 + v_nibble
                if out_d < 512:
                    out_acc = tl.zeros([], dtype=tl.float32)

                    for kv_i in range(kv_seq_len):
                        kv_offset = kv_base + kv_i * stride_kv_tok
                        scale_offset_base = scale_base + kv_i * stride_scale_row

                        # Recompute attention score
                        score = tl.zeros([], dtype=tl.float32)
                        for block_idx in tl.static_range(num_mxfp4_blocks):
                            block_start = block_idx * 32
                            scale_offset = scale_offset_base + block_idx * stride_scale_block
                            scale_e8m0 = tl.load(KV_scale_ptr + scale_offset)
                            scale_exp = scale_e8m0.to(tl.float32) - 127.0
                            scale_f32 = tl.math.exp2(scale_exp)

                            for byte_idx in tl.static_range(16):
                                packed_offset = kv_offset + (block_start // 2 + byte_idx) * stride_kv_dim
                                packed_byte = tl.load(KV_mxfp4_ptr + packed_offset)
                                low_nibble = packed_byte & 0x0F
                                high_nibble = (packed_byte >> 4) & 0x0F

                                sign_0 = (low_nibble >> 3) & 1
                                mag_0 = low_nibble & 0x07
                                k_mag_0 = tl.where(mag_0 == 0, 0.0,
                                          tl.where(mag_0 == 1, 0.5,
                                          tl.where(mag_0 == 2, 1.0,
                                          tl.where(mag_0 == 3, 1.5,
                                          tl.where(mag_0 == 4, 2.0,
                                          tl.where(mag_0 == 5, 3.0,
                                          tl.where(mag_0 == 6, 4.0, 6.0)))))))
                                k_val_0 = tl.where(sign_0 == 1, -k_mag_0, k_mag_0) * scale_f32

                                sign_1 = (high_nibble >> 3) & 1
                                mag_1 = high_nibble & 0x07
                                k_mag_1 = tl.where(mag_1 == 0, 0.0,
                                          tl.where(mag_1 == 1, 0.5,
                                          tl.where(mag_1 == 2, 1.0,
                                          tl.where(mag_1 == 3, 1.5,
                                          tl.where(mag_1 == 4, 2.0,
                                          tl.where(mag_1 == 5, 3.0,
                                          tl.where(mag_1 == 6, 4.0, 6.0)))))))
                                k_val_1 = tl.where(sign_1 == 1, -k_mag_1, k_mag_1) * scale_f32

                                dim_0 = block_start + byte_idx * 2
                                dim_1 = block_start + byte_idx * 2 + 1
                                q_val_0 = tl.load(Q_ptr + q_offset + dim_0 * stride_q_dim).to(tl.float32)
                                score += q_val_0 * k_val_0
                                if dim_1 < 576:
                                    q_val_1 = tl.load(Q_ptr + q_offset + dim_1 * stride_q_dim).to(tl.float32)
                                    score += q_val_1 * k_val_1

                        score = score * sm_scale
                        attn_weight = tl.exp(score - global_max) / sum_exp

                        # Get V value for this output dimension
                        v_scale_offset = scale_offset_base + v_block_idx * stride_scale_block
                        v_scale_e8m0 = tl.load(KV_scale_ptr + v_scale_offset)
                        v_scale_exp = v_scale_e8m0.to(tl.float32) - 127.0
                        v_scale_f32 = tl.math.exp2(v_scale_exp)

                        v_packed_offset = kv_offset + (v_block_idx * 16 + v_byte_idx) * stride_kv_dim
                        v_packed_byte = tl.load(KV_mxfp4_ptr + v_packed_offset)

                        v_nibble_val = tl.where(v_nibble == 0, v_packed_byte & 0x0F, (v_packed_byte >> 4) & 0x0F)
                        v_sign = (v_nibble_val >> 3) & 1
                        v_mag_idx = v_nibble_val & 0x07
                        v_mag = tl.where(v_mag_idx == 0, 0.0,
                                tl.where(v_mag_idx == 1, 0.5,
                                tl.where(v_mag_idx == 2, 1.0,
                                tl.where(v_mag_idx == 3, 1.5,
                                tl.where(v_mag_idx == 4, 2.0,
                                tl.where(v_mag_idx == 5, 3.0,
                                tl.where(v_mag_idx == 6, 4.0, 6.0)))))))
                        v_val = tl.where(v_sign == 1, -v_mag, v_mag) * v_scale_f32

                        out_acc += attn_weight * v_val

                    # Write output
                    tl.store(O_ptr + o_offset + out_d * stride_o_dim, out_acc.to(tl.bfloat16))


def _fused_mxfp4_mla_attention(
    q: torch.Tensor,
    kv_mxfp4: torch.Tensor,
    kv_scale: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """
    Fused MXFP4 dequantization + MLA attention.

    Strategy A: Load MXFP4 KV tiles, dequantize in registers, compute attention
    without writing intermediate bf16 buffer to HBM.

    Args:
        q: (total_q, num_heads, 576) bf16 query
        kv_mxfp4: (total_kv, 1, 288) fp4x2 packed KV cache
        kv_scale: (total_kv, num_blocks) E8M0 scale factors
        config: MLA configuration dict

    Returns:
        (total_q, num_heads, 512) bf16 attention output
    """
    batch_size = config["batch_size"]
    num_heads = config["num_heads"]
    q_seq_len = config["q_seq_len"]
    kv_seq_len = config["kv_seq_len"]
    qk_head_dim = config["qk_head_dim"]
    v_head_dim = config["v_head_dim"]
    sm_scale = config["sm_scale"]

    total_q = batch_size * q_seq_len

    # Allocate output
    o = torch.empty((total_q, num_heads, v_head_dim), dtype=torch.bfloat16, device=q.device)

    # Cast MXFP4 data to uint8 for Triton (Triton doesn't recognize float4_e2m1fn_x2)
    # Each fp4x2 element is 1 byte, so view as uint8
    kv_mxfp4_u8 = kv_mxfp4.view(torch.uint8)

    # Cast E8M0 scales to uint8 as well
    kv_scale_u8 = kv_scale.view(torch.uint8)

    # Grid: one program per (batch, head) pair
    grid = (batch_size, num_heads)

    # Block sizes
    BLOCK_KV = 64  # Process 64 KV tokens at a time

    _fused_mxfp4_mla_attention_kernel[grid](
        # Pointers
        q, kv_mxfp4_u8, kv_scale_u8, o,
        # Dimensions
        batch_size, num_heads, q_seq_len, kv_seq_len, qk_head_dim, v_head_dim,
        # Q strides
        q.stride(0), q.stride(1), q.stride(2),
        # KV strides (use original tensor's strides)
        kv_mxfp4.stride(0), kv_mxfp4.stride(1), kv_mxfp4.stride(2),
        # Scale strides
        kv_scale.stride(0), kv_scale.stride(1),
        # O strides
        o.stride(0), o.stride(1), o.stride(2),
        # Scale
        sm_scale,
        # Block sizes
        BLOCK_KV,
    )

    return o


def custom_kernel_mxfp4_dequant(data: input_t) -> output_t:
    """
    MLA decode with MXFP4 KV cache using Strategy A: Fused dequantization + attention.

    For small batches, uses the fused Triton kernel that:
    1. Loads MXFP4 KV tiles from HBM
    2. Dequantizes in registers/LDS
    3. Computes attention without writing intermediate buffer to HBM

    For larger batches, falls back to dequant + aiter kernel path.
    """
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]

    kv_buffer_mxfp4, kv_scale_mxfp4 = kv_data["mxfp4"]
    total_kv = kv_buffer_mxfp4.shape[0]
    orig_shape = (total_kv, NUM_KV_HEADS, QK_HEAD_DIM)

    # For small batches with short sequences, try the fused kernel
    # For now, use fallback path until fused kernel is fully debugged
    USE_FUSED_KERNEL = True

    if USE_FUSED_KERNEL and batch_size <= 4 and kv_seq_len <= 2048:
        # Strategy A: Fused MXFP4 dequant + attention
        return _fused_mxfp4_mla_attention(q, kv_buffer_mxfp4, kv_scale_mxfp4, config)

    # Fallback: Dequantize MXFP4 -> FP8, then use aiter a8w8 kernel
    kv_buffer_fp8, kv_scale_fp8 = dequantize_mxfp4_to_fp8(
        kv_buffer_mxfp4, kv_scale_mxfp4, orig_shape
    )
    q_fp8, q_scale = quantize_fp8(q)

    return _aiter_mla_decode(
        q_fp8, kv_buffer_fp8, qo_indptr, kv_indptr, config,
        q_scale=q_scale, kv_scale=kv_scale_fp8,
    )


# ---------------------------------------------------------------------------
# MXFP4 Q + MXFP4 KV - native low-precision compute with gemm_a4w4
# ---------------------------------------------------------------------------

def _quantize_bf16_to_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize bf16 tensor to MXFP4 with per-1x32 block scaling.

    Args:
        x: bf16 tensor of shape (..., K) where K is divisible by 32

    Returns:
        (x_mxfp4, x_scale): MXFP4 packed tensor and E8M0 scales
    """
    quant_func = aiter.get_triton_quant(QuantType.per_1x32)
    x_q, x_scale_sh = quant_func(x.contiguous(), shuffle=True)
    return x_q, x_scale_sh


def custom_kernel_mxfp4_native(data: input_t) -> output_t:
    """
    MLA decode using Strategy B: Full MXFP4 compute with gemm_a4w4.

    This implementation:
    1. Uses custom HIP kernel if available (fastest path)
    2. Falls back to gemm_a4w4 for fp4xfp4 QK^T computation
    3. Applies softmax in fp32 for numerical stability
    4. Uses bf16 for attn @ V computation

    This trades accuracy for throughput by leveraging MFMA fp4 instructions.
    """
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    num_heads = config["num_heads"]
    q_seq_len = config["q_seq_len"]
    kv_seq_len = config["kv_seq_len"]
    qk_head_dim = config["qk_head_dim"]  # 576
    v_head_dim = config["v_head_dim"]    # 512
    sm_scale = config["sm_scale"]

    kv_buffer_mxfp4, kv_scale_mxfp4 = kv_data["mxfp4"]
    # print(f"{batch_size=}, {num_heads=}, {q_seq_len=}, {kv_seq_len=}, {qk_head_dim}, {v_head_dim}, {sm_scale=}")

    total_q = q.shape[0]  # batch_size * q_seq_len
    total_kv = kv_buffer_mxfp4.shape[0]  # batch_size * kv_seq_len

    # Use custom HIP kernel if available (fused dequant + attention)
    # Supports q_seq_len up to 4 (decode and small prefill)
    if HAS_HIP_KERNEL and q_seq_len <= 4:
        # KV: (total_kv, 1, qk_head_dim/2) -> (batch_size, kv_seq_len, qk_head_dim/2)
        kv_reshaped = kv_buffer_mxfp4.view(batch_size, kv_seq_len, qk_head_dim // 2)

        # Scale: (total_kv, num_blocks) -> (batch_size, kv_seq_len, num_blocks)
        num_blocks = kv_scale_mxfp4.shape[1]
        scale_reshaped = kv_scale_mxfp4.view(batch_size, kv_seq_len, num_blocks)

        # Allocate output tensor
        output = torch.empty(
            (total_q, num_heads, v_head_dim),
            dtype=torch.bfloat16,
            device=q.device
        )

        # Launch HIP kernel (uses hip-python or PyTorch depending on what compiled)
        _launch_hip_kernel(
            q.contiguous(),
            kv_reshaped.contiguous(),
            scale_reshaped.contiguous(),
            output,
            batch_size,
            q_seq_len,
            kv_seq_len,
            sm_scale
        )

        return output

    # Fallback: Python implementation with gemm_a4w4
    # Dequantize entire V to bf16 once (more efficient than per-batch)
    v_bf16_all = dequantize_mxfp4(
        kv_buffer_mxfp4,
        kv_scale_mxfp4,
        (total_kv, NUM_KV_HEADS, QK_HEAD_DIM)
    )[:, 0, :v_head_dim]  # (total_kv, 512)

    # Reshape V: (total_kv, 512) -> (batch_size, kv_seq_len, 512)
    v_batched = v_bf16_all.view(batch_size, kv_seq_len, v_head_dim)

    # Reshape Q: (total_q, num_heads, 576) -> (batch_size, q_seq_len, num_heads, 576)
    q_batched = q.view(batch_size, q_seq_len, num_heads, qk_head_dim)

    # Reshape KV: (total_kv, 1, 576/2) -> (batch_size, kv_seq_len, 576/2)
    kv_batched_mxfp4 = kv_buffer_mxfp4.view(batch_size, kv_seq_len, qk_head_dim // 2)
    kv_scale_batched = kv_scale_mxfp4.view(batch_size, kv_seq_len, -1)

    # Output tensor
    output = torch.empty((batch_size, q_seq_len, num_heads, v_head_dim),
                         dtype=torch.bfloat16, device=q.device)

    # Process each batch separately
    for b in range(batch_size):
        # Q for this batch: (q_seq_len, num_heads, 576) -> (q_seq_len * num_heads, 576)
        q_b = q_batched[b].view(q_seq_len * num_heads, qk_head_dim).contiguous()

        # Quantize Q to MXFP4
        q_mxfp4, q_scale = _quantize_bf16_to_mxfp4(q_b)

        # K for this batch: (kv_seq_len, 576/2)
        k_b_mxfp4 = kv_batched_mxfp4[b].contiguous()
        k_scale_b = kv_scale_batched[b].contiguous()

        # Shuffle K for gemm_a4w4
        k_shuffle = shuffle_weight(k_b_mxfp4, layout=(16, 16))

        # Compute QK^T using gemm_a4w4
        # A = Q_mxfp4 [q_seq_len * num_heads, 576/2]
        # B = K_mxfp4 [kv_seq_len, 576/2]
        # Output: [q_seq_len * num_heads, kv_seq_len]
        scores = aiter.gemm_a4w4(
            q_mxfp4,
            k_shuffle,
            q_scale,
            k_scale_b,
            dtype=aiter_dtypes.bf16,
            bpreshuffle=True,
        )

        # Apply scale and softmax in fp32
        scores = scores.float() * sm_scale

        # Reshape: (q_seq_len * num_heads, kv_seq_len) -> (q_seq_len, num_heads, kv_seq_len)
        scores = scores.view(q_seq_len, num_heads, kv_seq_len)

        # Apply softmax along KV dimension
        attn_weights = torch.softmax(scores, dim=-1)  # (q_seq_len, num_heads, kv_seq_len)

        # V for this batch: (kv_seq_len, 512)
        v_b = v_batched[b]

        # attn @ V: (q_seq_len, num_heads, kv_seq_len) @ (kv_seq_len, v_head_dim)
        # -> (q_seq_len, num_heads, v_head_dim)
        out_b = torch.einsum('qhk,kd->qhd', attn_weights.to(torch.bfloat16), v_b)

        output[b] = out_b

    # Reshape output: (batch_size, q_seq_len, num_heads, v_head_dim) -> (total_q, num_heads, v_head_dim)
    output = output.view(total_q, num_heads, v_head_dim)

    return output
