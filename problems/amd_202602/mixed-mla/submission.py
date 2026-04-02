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
from task import input_t, output_t
from aiter.mla import mla_decode_fwd
from aiter import QuantType, dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

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
constexpr int QK_HEAD_DIM = 576;
constexpr int NUM_HEADS = 16;
constexpr int FMT_FP4_MFMA = 4;

typedef float __attribute__((ext_vector_type(4))) float4_t;
typedef float __attribute__((ext_vector_type(16))) float16_t;
typedef int __attribute__((ext_vector_type(8))) int8_vec;
typedef uint32_t __attribute__((ext_vector_type(4))) uint128_vec;

// Fast E8M0 to float using bit reinterpretation
__device__ __forceinline__ float e8m0_to_float_fast(uint8_t e8m0) {
    // E8M0: pure exponent format, value = 2^(e8m0 - 127)
    // Construct IEEE754 float directly: exponent = e8m0, mantissa = 0
    uint32_t bits = (static_cast<uint32_t>(e8m0)) << 23;
    return __uint_as_float(bits);
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

// E8M0 scale computation from FP32 amax (avoids BF16 intermediate)
__device__ __forceinline__ E8M0Scale compute_e8m0_scale_f32(float amax) {
    E8M0Scale result;
    uint32_t amax_bits = __float_as_uint(amax);
    if ((amax_bits & 0x7FFFFFFFu) == 0) {
        result.e8m0 = 0;
        result.quant_scale = 0.0f;
    } else {
        // Round mantissa up to nearest power of 2 (same logic as BF16 but for FP32)
        // FP32: [sign(1)][exp(8)][mant(23)], round bit at position 22
        uint32_t rounded = (amax_bits + 0x00400000u) & 0xFF800000u;
        int raw_exp = (int)((rounded >> 23) & 0xFF);
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

// Hardware FP4 conversion from FP32: converts 2 FP32 values to packed FP4 byte.
__device__ __forceinline__ uint8_t quantize_fp4_pair_hw(float v0, float v1, float quant_scale) {
    union { uint32_t u32; uint8_t u8[4]; } cvt = {0};
    cvt.u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
        cvt.u32, v0, v1, quant_scale, 0);
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
// Uses FP32 hw intrinsic directly (no BF16 intermediate).
__device__ __forceinline__ QuantBlock quantize_fp4_block(const float vals[32]) {
    // Find amax in FP32
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        float a = fabsf(vals[i]);
        amax = (a > amax) ? a : amax;
    }

    E8M0Scale sc = compute_e8m0_scale_f32(amax);

    uint32_t pack[4] = {0, 0, 0, 0};
    for (int i = 0; i < 16; i++) {
        uint8_t packed = quantize_fp4_pair_hw(vals[2*i], vals[2*i+1], sc.quant_scale);
        pack[i / 4] |= ((uint32_t)packed) << ((i % 4) * 8);
    }

    QuantBlock result;
    *reinterpret_cast<uint128_vec*>(&result.data) = *reinterpret_cast<uint128_vec*>(&pack);
    result.e8m0 = sc.e8m0;
    return result;
}

template <int M, int K, int K_HALF, int NUM_BLOCKS, int BATCH_SIZE, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE)
void mla_quant_q_batched_kernel(
    const hip_bfloat16 Q[][K],
    uint8_t out_data[][NUM_BLOCKS * 16],
    uint8_t* __restrict__ out_scale
) {
    constexpr int A_K_HALF = NUM_BLOCKS * 16;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int per_batch = M * NUM_BLOCKS;
    int batch_idx = gid / per_batch;
    int local_id = gid % per_batch;
    if (batch_idx >= BATCH_SIZE) return;

    int row = local_id / NUM_BLOCKS;
    int blk = local_id % NUM_BLOCKS;
    if (row >= M) return;

    QuantBlock qb = quantize_fp4_block_bf16(&Q[batch_idx * M + row][blk * 32]);

    *reinterpret_cast<uint128_vec*>(&out_data[batch_idx * M + row][blk * 16]) =
        *reinterpret_cast<uint128_vec*>(&qb.data);
    out_scale[batch_idx * NUM_BLOCKS * M + row + blk * M] = qb.e8m0;
}

// =====================================================================
// MLA QK^T kernel: MFMA FP4, fp32 output, batched, linear B scales
// =====================================================================

__device__ __forceinline__ int32_t mla_broadcast_scale(uint8_t e8m0) {
    return (int32_t)e8m0 * 0x01010101;
}

// =====================================================================
// MLA MFMA traits: abstracts 16x16x128 vs 32x32x64 differences
// =====================================================================
template <bool USE_32x32>
struct MlaMfmaTraits {
    using acc_t = typename std::conditional<USE_32x32, float16_t, float4_t>::type;
    static constexpr int IM = USE_32x32 ? 32 : 16;
    static constexpr int IK = USE_32x32 ? 64 : 128;
    static constexpr int BPC = IK / 32;
    static constexpr int ACC_SIZE = USE_32x32 ? 16 : 4;

    static __device__ __forceinline__ acc_t mfma(
        int8_vec a, int32_t a_sc, int8_vec b, int32_t b_sc, acc_t acc
    ) {
        if constexpr (USE_32x32) {
            return __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a, b, acc, FMT_FP4_MFMA, FMT_FP4_MFMA, 0, a_sc, 0, b_sc);
        } else {
            return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                a, b, acc, FMT_FP4_MFMA, FMT_FP4_MFMA, 0, a_sc, 0, b_sc);
        }
    }

    // Map accumulator element index + lane to head index
    static __device__ __forceinline__ int acc_to_head(int i, int lane) {
        if constexpr (USE_32x32) {
            int half = lane / 32;
            return (i % 4) + 4 * half + 8 * (i / 4);
        } else {
            int quad = lane / 16;
            return i + 4 * quad;
        }
    }

    // Map lane to output column (N-dimension)
    static __device__ __forceinline__ int lane_to_col(int lane) {
        return lane % IM;
    }

    // Store accumulator values to scores_lds with sm_scale
    template <int HEADS, int KV_SUBTILE>
    static __device__ __forceinline__ void store_scores(
        const acc_t& acc, int lane, int batch_col_offset, int subtile_size,
        float sm_scale, float scores[][KV_SUBTILE]
    ) {
        int col = lane_to_col(lane);
        int eff_col = batch_col_offset + col;
        for (int i = 0; i < ACC_SIZE; i++) {
            int head = acc_to_head(i, lane);
            if (head < HEADS && eff_col < subtile_size)
                scores[head][eff_col] = acc[i] * sm_scale;
        }
    }

    // Rescale V accumulators by per-head factors
    template <int HEADS, int CHUNKS_PER_WARP>
    static __device__ __forceinline__ void rescale_v_acc(
        acc_t v_acc[], int lane, const float rescale[]
    ) {
        for (int ci = 0; ci < CHUNKS_PER_WARP; ci++) {
            for (int i = 0; i < ACC_SIZE; i++) {
                int head = acc_to_head(i, lane);
                if (head < HEADS)
                    v_acc[ci][i] *= rescale[head];
            }
        }
    }

    // Store V accumulator to partial output
    template <int HEADS, int V_DIM, int CHUNKS_PER_WARP>
    static __device__ __forceinline__ void store_v_output(
        const acc_t v_acc[], int lane, int warp_id,
        int out_row, float partial_out[][16 * 512],
        const float running_sum_lds[][128]
    ) {
        int col = lane_to_col(lane);
        for (int ci = 0; ci < CHUNKS_PER_WARP; ci++) {
            int chunk = warp_id * CHUNKS_PER_WARP + ci;
            for (int i = 0; i < ACC_SIZE; i++) {
                int head = acc_to_head(i, lane);
                int v_dim = chunk * IM + col;
                if (head < HEADS && v_dim < V_DIM) {
                    float rs = running_sum_lds[head][0];
                    float inv_sum = (rs > 0.0f) ? (1.0f / rs) : 0.0f;
                    partial_out[out_row][head * V_DIM + v_dim] = v_acc[ci][i] * inv_sum;
                }
            }
        }
    }
};

// =============================================================================
// FUSED FlashAttention-style MLA decode kernel (v5 - MFMA V)
//
// Replaces: QK^T GEMM + softmax + attnV with a SINGLE kernel that reads KV
// data ONCE from HBM. Uses online softmax with V accumulator rescaling.
//
// QK^T: single-warp MFMA 16x16x128, 8 iterations over 128-position subtile
// V accumulation: MFMA 16x16x128 with scale-absorption pattern
//   (same as mla_attn_v_mfma_head_merged_kernel)
//
// Grid: (BATCH_SIZE, KV_SPLITS)
// Block: 256 threads (4 warps) - warp 0 does QK^T, all 4 do MFMA V
//
// LDS: Q data+scale (4.9KB) + KV tile (36.9KB) + KV scales (3.1KB)
//      + scores (8.2KB) = ~53 KB
// =============================================================================

template <int N, int BLOCK_SIZE, int B_SCALE_STRIDE, int KV_SPLITS,
          int K_HALF, int NUM_BLOCKS, int A_K_HALF,
          bool USE_32x32 = false, bool PROFILE_PHASES = false>
__global__ __launch_bounds__(BLOCK_SIZE)
void mla_fused_attn_kernel(
    const hip_bfloat16* __restrict__ Q_bf16,
    const uint8_t q_data[][A_K_HALF],
    const uint8_t* __restrict__ q_scale,
    const uint8_t kv_mxfp4[][K_HALF],
    const uint8_t kv_scale[][B_SCALE_STRIDE],
    float sm_scale,
    float partial_out[][16 * 512],
    float partial_lse[][16]
) {
    constexpr int V_DIM = 512;
    constexpr int HEADS = 16;
    constexpr int KV_PER_SPLIT = (N + KV_SPLITS - 1) / KV_SPLITS;
    constexpr int KV_SUBTILE = 128;

    // MFMA variant traits
    using Traits = MlaMfmaTraits<USE_32x32>;
    using acc_t = typename Traits::acc_t;
    constexpr int IM = Traits::IM;
    constexpr int BPC = Traits::BPC;
    constexpr int ACC_SIZE = Traits::ACC_SIZE;
    constexpr int K_ITERS = (NUM_BLOCKS + BPC - 1) / BPC;
    constexpr int QKT_BATCH = IM;
    constexpr int V_CHUNK_DIM = IM;
    constexpr int V_CHUNKS = V_DIM / V_CHUNK_DIM;

    constexpr int WARPS = BLOCK_SIZE / 64;
    constexpr int CHUNKS_PER_WARP = V_CHUNKS / WARPS;

    const int batch_idx = __builtin_amdgcn_readfirstlane(blockIdx.x);
    const int split_idx = __builtin_amdgcn_readfirstlane(blockIdx.y);
    const int tid = threadIdx.x;
    const int warp_id = __builtin_amdgcn_readfirstlane(tid / 64);
    const int lane = tid % 64;

    const int kv_start_global = split_idx * KV_PER_SPLIT;
    const int kv_end_global = min(kv_start_global + KV_PER_SPLIT, N);

    [[maybe_unused]] const bool is_timer_block = PROFILE_PHASES && (blockIdx.x == 0 && blockIdx.y == 0);
    [[maybe_unused]] uint64_t t_start = 0, t_load_total = 0, t_qkt_total = 0, t_sv_total = 0, t_store = 0;
    [[maybe_unused]] uint64_t t_softmax_total = 0, t_mfma_v_total = 0;
    [[maybe_unused]] uint64_t t_phase = 0;
    if constexpr (PROFILE_PHASES) {
        if (is_timer_block && tid == 0)
            t_start = __builtin_amdgcn_s_memrealtime();
    }

    // ---- LDS layout ----
    __shared__ uint8_t q_lds_data[HEADS][A_K_HALF];
    __shared__ uint8_t q_lds_scale[NUM_BLOCKS * HEADS];
    __shared__ uint8_t kv_lds_data[KV_SUBTILE][K_HALF];
    __shared__ uint8_t kv_lds_scale_tile[KV_SUBTILE][B_SCALE_STRIDE];
    __shared__ float   scores_lds[HEADS][KV_SUBTILE];

    // ---- Per-warp MFMA V accumulators + online softmax state ----
    acc_t v_acc[CHUNKS_PER_WARP];
    float running_max[HEADS];
    float running_sum[HEADS];
    for (int c = 0; c < CHUNKS_PER_WARP; c++) {
        v_acc[c] = {};
    }
    for (int h = 0; h < HEADS; h++) {
        running_max[h] = -INFINITY;
        running_sum[h] = 0.0f;
    }

    const int out_row = __builtin_amdgcn_readfirstlane(batch_idx * KV_SPLITS + split_idx);

    // MFMA lane indices (used for both QK^T and V accumulation)
    const int mfma_a_row = lane % IM;
    const int mfma_a_kgrp = lane / IM;
    const int mfma_b_col = lane % IM;
    const int mfma_b_kgrp = lane / IM;
    const int32_t b_sc_one = mla_broadcast_scale(127);

    // ---- Step 1: Load pre-quantized Q into LDS (one-time) ----
    {
        const int total_q_bytes = HEADS * A_K_HALF;
        for (int i = tid * 16; i < total_q_bytes; i += BLOCK_SIZE * 16) {
            if (i + 16 <= total_q_bytes) {
                const int h = i / A_K_HALF;
                const int col = i % A_K_HALF;
                *reinterpret_cast<uint128_vec*>(&q_lds_data[h][col]) =
                    *reinterpret_cast<const uint128_vec*>(&q_data[batch_idx * HEADS + h][col]);
            }
        }
        const uint8_t* q_s = q_scale + batch_idx * NUM_BLOCKS * HEADS;
        const int total_q_scales = NUM_BLOCKS * HEADS;
        for (int i = tid; i < total_q_scales; i += BLOCK_SIZE) {
            q_lds_scale[i] = q_s[i];
        }
    }
    if (kv_start_global >= N) {
        for (int i = tid; i < HEADS * V_DIM; i += BLOCK_SIZE) {
            partial_out[out_row][i] = 0.0f;
        }
        if (tid < HEADS) partial_lse[out_row][tid] = -INFINITY;
        return;
    }

    __syncthreads();

    // ---- Step 2: Iterate over KV in subtiles of 128 positions ----
    const int kv_row_base = __builtin_amdgcn_readfirstlane(batch_idx * N);

    for (int kv_pos = kv_start_global; kv_pos < kv_end_global; kv_pos += KV_SUBTILE) {
        const int subtile_end = min(kv_pos + KV_SUBTILE, kv_end_global);
        const int subtile_size = subtile_end - kv_pos;

        if constexpr (PROFILE_PHASES) {
            if (is_timer_block && tid == 0)
                t_phase = __builtin_amdgcn_s_memrealtime();
        }

        // ---- 2a: Load 128 KV positions into LDS ----
        {
            const int total_bytes = subtile_size * K_HALF;
            for (int i = tid * 16; i < total_bytes; i += BLOCK_SIZE * 16) {
                const int row = i / K_HALF;
                const int col = i % K_HALF;
                if (row < subtile_size) {
                    *reinterpret_cast<uint128_vec*>(&kv_lds_data[row][col]) =
                        *reinterpret_cast<const uint128_vec*>(&kv_mxfp4[kv_row_base + kv_pos + row][col]);
                }
            }
            const int total_sc = subtile_size * B_SCALE_STRIDE;
            for (int i = tid; i < total_sc; i += BLOCK_SIZE) {
                const int row = i / B_SCALE_STRIDE;
                const int blk = i % B_SCALE_STRIDE;
                kv_lds_scale_tile[row][blk] =
                    kv_scale[kv_row_base + kv_pos + row][blk];
            }
        }

        __syncthreads();

        if constexpr (PROFILE_PHASES) {
            if (is_timer_block && tid == 0) {
                uint64_t t_now = __builtin_amdgcn_s_memrealtime();
                t_load_total += t_now - t_phase;
                t_phase = t_now;
            }
        }

        // ---- 2b: QK^T via MFMA (warp 0 only) ----
        if (warp_id == 0) {
            for (int qkt_iter = 0; qkt_iter < KV_SUBTILE / QKT_BATCH; qkt_iter++) {
                acc_t mfma_acc = {};
                for (int ki = 0; ki < K_ITERS; ki++) {
                    int blk0 = ki * BPC;
                    int qa_row = lane % IM;
                    int qa_blk = blk0 + lane / IM;
                    uint32_t qa_reg[8] = {};
                    if (qa_row < HEADS && qa_blk < NUM_BLOCKS) {
                        *reinterpret_cast<uint128_vec*>(&qa_reg[0]) =
                            *reinterpret_cast<const uint128_vec*>(&q_lds_data[qa_row][qa_blk * 16]);
                    }
                    uint8_t qa_e = 127;
                    if (qa_row < HEADS && qa_blk < NUM_BLOCKS)
                        qa_e = q_lds_scale[qa_row + qa_blk * HEADS];
                    else
                        qa_reg[0] = qa_reg[1] = qa_reg[2] = qa_reg[3] = 0;
                    int32_t qa_sc = mla_broadcast_scale(qa_e);

                    int kb_row = lane % IM;
                    int kb_abs = qkt_iter * QKT_BATCH + kb_row;
                    int kb_blk = blk0 + lane / IM;
                    uint32_t kb_reg[8] = {};
                    if (kb_abs < subtile_size && kb_blk < NUM_BLOCKS) {
                        *reinterpret_cast<uint128_vec*>(&kb_reg[0]) =
                            *reinterpret_cast<const uint128_vec*>(&kv_lds_data[kb_abs][kb_blk * 16]);
                    }
                    uint8_t kb_e = 127;
                    if (kb_abs < subtile_size && kb_blk < NUM_BLOCKS)
                        kb_e = kv_lds_scale_tile[kb_abs][kb_blk];
                    else
                        kb_reg[0] = kb_reg[1] = kb_reg[2] = kb_reg[3] = 0;
                    int32_t kb_sc = mla_broadcast_scale(kb_e);

                    int8_vec qa_vec = {(int)qa_reg[0],(int)qa_reg[1],(int)qa_reg[2],(int)qa_reg[3],
                                       (int)qa_reg[4],(int)qa_reg[5],(int)qa_reg[6],(int)qa_reg[7]};
                    int8_vec kb_vec = {(int)kb_reg[0],(int)kb_reg[1],(int)kb_reg[2],(int)kb_reg[3],
                                       (int)kb_reg[4],(int)kb_reg[5],(int)kb_reg[6],(int)kb_reg[7]};

                    mfma_acc = Traits::mfma(qa_vec, qa_sc, kb_vec, kb_sc, mfma_acc);
                }
                Traits::template store_scores<HEADS, KV_SUBTILE>(
                    mfma_acc, lane, qkt_iter * QKT_BATCH, subtile_size,
                    sm_scale, scores_lds);
            }
        }

        __syncthreads();

        if constexpr (PROFILE_PHASES) {
            if (is_timer_block && tid == 0) {
                uint64_t t_now = __builtin_amdgcn_s_memrealtime();
                t_qkt_total += t_now - t_phase;
                t_phase = t_now;
            }
        }

        // ---- 2c: Softmax over 128 scores + rescale MFMA V accumulators ----
        // Only threads 0..15 compute softmax (one head each) - eliminates 240 threads of redundant expf
        __shared__ float rescale_lds[HEADS];
        __shared__ float new_max_lds[HEADS];
        {
            if (tid < HEADS) {
                const int h = tid;
                // Find max in subtile
                float subtile_max = -INFINITY;
                for (int ki = 0; ki < subtile_size; ki++) {
                    subtile_max = fmaxf(subtile_max, scores_lds[h][ki]);
                }
                const float old_max = running_max[h];
                const float new_max = fmaxf(old_max, subtile_max);
                const float prev_rescale = __expf(old_max - new_max);

                // Write rescale factor + new_max to LDS for all threads
                rescale_lds[h] = prev_rescale;
                new_max_lds[h] = new_max;

                // Update this thread's running state
                running_max[h] = new_max;
                running_sum[h] *= prev_rescale;

                // Compute exp-sum
                float subtile_sum = 0.0f;
                for (int ki = 0; ki < subtile_size; ki++) {
                    subtile_sum += __expf(scores_lds[h][ki] - new_max);
                }
                running_sum[h] += subtile_sum;

                // Write exp weights to scores_lds for V accumulation
                for (int ki = 0; ki < subtile_size; ki++) {
                    scores_lds[h][ki] = __expf(scores_lds[h][ki] - new_max);
                }
                for (int ki = subtile_size; ki < KV_SUBTILE; ki++) {
                    scores_lds[h][ki] = 0.0f;
                }
            }
        }

        __syncthreads();

        // ALL threads: update their own running state + rescale v_acc using LDS values
        {
            // Update running_max/running_sum for threads 16..255 (threads 0..15 already did it)
            if (tid >= HEADS) {
                for (int h = 0; h < HEADS; h++) {
                    running_max[h] = new_max_lds[h];
                    // running_sum doesn't matter for threads > 15 until output
                }
            }

            // ALL threads rescale their MFMA V accumulators
            Traits::template rescale_v_acc<HEADS, CHUNKS_PER_WARP>(
                v_acc, lane, rescale_lds);
        }

        if constexpr (PROFILE_PHASES) {
            if (is_timer_block && tid == 0) {
                uint64_t t_now = __builtin_amdgcn_s_memrealtime();
                t_softmax_total += t_now - t_phase;
                t_phase = t_now;
            }
        }

        // ---- 2d: MFMA Attn*V with scale-absorption pattern ----
        if constexpr (USE_32x32) {
            // 32x32x64: each chunk covers 32 V dims, 1 MFMA per chunk
            for (int ci = 0; ci < CHUNKS_PER_WARP; ci++) {
                const int chunk = warp_id * CHUNKS_PER_WARP + ci;
                const int vscale_blk = chunk;  // 1 scale block per 32 V dims

                // A operand: attn * v_scale, quantized to FP4
                const int k_base_a = mfma_a_kgrp * 32;
                float scaled_attn[32];
                for (int j = 0; j < 32; j++) {
                    const float aw = (mfma_a_row < HEADS) ? scores_lds[mfma_a_row][k_base_a + j] : 0.0f;
                    const float vs = e8m0_to_float_fast(kv_lds_scale_tile[k_base_a + j][vscale_blk]);
                    scaled_attn[j] = aw * vs;
                }
                QuantBlock aqb = quantize_fp4_block(scaled_attn);

                uint32_t a_reg[8] = {};
                *reinterpret_cast<uint128_vec*>(&a_reg[0]) =
                    *reinterpret_cast<uint128_vec*>(&aqb.data);
                int32_t a_sc = mla_broadcast_scale(aqb.e8m0);

                int8_vec a_vec = {(int)a_reg[0], (int)a_reg[1], (int)a_reg[2], (int)a_reg[3],
                                  (int)a_reg[4], (int)a_reg[5], (int)a_reg[6], (int)a_reg[7]};

                // B operand: gather V nibbles for 32 V dims
                {
                    const int k_base_b = mfma_b_kgrp * 32;
                    const int v_d = chunk * 32 + mfma_b_col;
                    const int nib_shift = (v_d & 1) * 4;
                    const int local_off = mfma_b_col / 2;
                    const int lds_byte_base = chunk * 16;

                    uint8_t packed[16];
                    for (int j = 0; j < 16; j++) {
                        const int k0 = k_base_b + j * 2;
                        const int k1 = k_base_b + j * 2 + 1;

                        uint8_t raw0 = 0, raw1 = 0;
                        if (k0 < subtile_size)
                            raw0 = kv_lds_data[k0][lds_byte_base + local_off];
                        if (k1 < subtile_size)
                            raw1 = kv_lds_data[k1][lds_byte_base + local_off];

                        uint8_t n0 = (raw0 >> nib_shift) & 0x0F;
                        uint8_t n1 = (raw1 >> nib_shift) & 0x0F;
                        packed[j] = n0 | (n1 << 4);
                    }

                    uint32_t b_reg[8] = {};
                    *reinterpret_cast<uint128_vec*>(&b_reg[0]) =
                        *reinterpret_cast<uint128_vec*>(&packed[0]);
                    int8_vec b_vec = {(int)b_reg[0], (int)b_reg[1], (int)b_reg[2], (int)b_reg[3],
                                      (int)b_reg[4], (int)b_reg[5], (int)b_reg[6], (int)b_reg[7]};
                    v_acc[ci] = Traits::mfma(a_vec, a_sc, b_vec, b_sc_one, v_acc[ci]);
                }
            }
        } else {
            // 16x16x128: each chunk covers 16 V dims, 2 sub-chunks per iteration
            for (int ci = 0; ci < CHUNKS_PER_WARP; ci += 2) {
                const int chunk0 = warp_id * CHUNKS_PER_WARP + ci;
                const int vscale_blk = chunk0 / 2;

                // A operand: attn * v_scale, quantized to FP4
                const int k_base_a = mfma_a_kgrp * 32;
                float scaled_attn[32];
                for (int j = 0; j < 32; j++) {
                    const float aw = scores_lds[mfma_a_row][k_base_a + j];
                    const float vs = e8m0_to_float_fast(kv_lds_scale_tile[k_base_a + j][vscale_blk]);
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

                // B operand: gather V nibbles for both sub-chunks
                {
                    const int k_base_b = mfma_b_kgrp * 32;
                    const int v_d0 = chunk0 * 16 + mfma_b_col;
                    const int v_d1 = (chunk0 + 1) * 16 + mfma_b_col;
                    const int nib_shift0 = (v_d0 & 1) * 4;
                    const int nib_shift1 = (v_d1 & 1) * 4;
                    const int local_off0 = mfma_b_col / 2;
                    const int local_off1 = 8 + mfma_b_col / 2;
                    const int lds_byte_base = chunk0 * 8;

                    uint8_t packed0[16], packed1[16];

                    for (int j = 0; j < 16; j++) {
                        const int k0 = k_base_b + j * 2;
                        const int k1 = k_base_b + j * 2 + 1;

                        uint8_t raw0_s0 = 0, raw0_s1 = 0;
                        if (k0 < subtile_size) {
                            raw0_s0 = kv_lds_data[k0][lds_byte_base + local_off0];
                            raw0_s1 = kv_lds_data[k0][lds_byte_base + local_off1];
                        }

                        uint8_t raw1_s0 = 0, raw1_s1 = 0;
                        if (k1 < subtile_size) {
                            raw1_s0 = kv_lds_data[k1][lds_byte_base + local_off0];
                            raw1_s1 = kv_lds_data[k1][lds_byte_base + local_off1];
                        }

                        uint8_t n0_s0 = (raw0_s0 >> nib_shift0) & 0x0F;
                        uint8_t n1_s0 = (raw1_s0 >> nib_shift0) & 0x0F;
                        packed0[j] = n0_s0 | (n1_s0 << 4);

                        uint8_t n0_s1 = (raw0_s1 >> nib_shift1) & 0x0F;
                        uint8_t n1_s1 = (raw1_s1 >> nib_shift1) & 0x0F;
                        packed1[j] = n0_s1 | (n1_s1 << 4);
                    }

                    // MFMA for sub-chunk 0
                    {
                        uint32_t b_reg[8];
                        *reinterpret_cast<uint128_vec*>(&b_reg[0]) =
                            *reinterpret_cast<uint128_vec*>(&packed0[0]);
                        b_reg[4] = b_reg[5] = b_reg[6] = b_reg[7] = 0;
                        int8_vec b_vec = {(int)b_reg[0], (int)b_reg[1], (int)b_reg[2], (int)b_reg[3],
                                          (int)b_reg[4], (int)b_reg[5], (int)b_reg[6], (int)b_reg[7]};
                        v_acc[ci] = Traits::mfma(a_vec, a_sc, b_vec, b_sc_one, v_acc[ci]);
                    }

                    // MFMA for sub-chunk 1
                    {
                        uint32_t b_reg[8];
                        *reinterpret_cast<uint128_vec*>(&b_reg[0]) =
                            *reinterpret_cast<uint128_vec*>(&packed1[0]);
                        b_reg[4] = b_reg[5] = b_reg[6] = b_reg[7] = 0;
                        int8_vec b_vec = {(int)b_reg[0], (int)b_reg[1], (int)b_reg[2], (int)b_reg[3],
                                          (int)b_reg[4], (int)b_reg[5], (int)b_reg[6], (int)b_reg[7]};
                        v_acc[ci + 1] = Traits::mfma(a_vec, a_sc, b_vec, b_sc_one, v_acc[ci + 1]);
                    }
                }
            }
        }

        if constexpr (PROFILE_PHASES) {
            if (is_timer_block && tid == 0) {
                uint64_t t_now = __builtin_amdgcn_s_memrealtime();
                t_mfma_v_total += t_now - t_phase;
                t_sv_total = t_softmax_total + t_mfma_v_total;
            }
        }

        __syncthreads();
    }

    if constexpr (PROFILE_PHASES) {
        if (is_timer_block && tid == 0)
            t_phase = __builtin_amdgcn_s_memrealtime();
    }

    // ---- Step 3: Broadcast running_sum via LDS, then write partial output ----
    // running_sum is only correct on threads 0..15; broadcast to all via scores_lds
    if (tid < HEADS) {
        scores_lds[tid][0] = running_sum[tid];
    }
    __syncthreads();

    {
        Traits::template store_v_output<HEADS, V_DIM, CHUNKS_PER_WARP>(
            v_acc, lane, warp_id, out_row, partial_out, scores_lds);
    }

    if (tid < HEADS) {
        float lse = running_max[tid] + logf(fmaxf(running_sum[tid], 1e-20f));
        partial_lse[out_row][tid] = lse;
    }

    if constexpr (PROFILE_PHASES) {
        if (is_timer_block && tid == 0) {
            t_store = __builtin_amdgcn_s_memrealtime() - t_phase;
            uint64_t t_total = __builtin_amdgcn_s_memrealtime() - t_start;
            printf("[Fused Attn] N=%d splits=%d | load=%llu qkt=%llu softmax=%llu mfma_v=%llu store=%llu total=%llu cycles\\n",
                   N, KV_SPLITS,
                   (unsigned long long)t_load_total,
                   (unsigned long long)t_qkt_total,
                   (unsigned long long)t_softmax_total,
                   (unsigned long long)t_mfma_v_total,
                   (unsigned long long)t_store,
                   (unsigned long long)t_total);
        }
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
    const float partial_out[][16 * 512],         // (batch * KV_SPLITS, 16 * 512)
    const float partial_lse[][16],               // (batch * KV_SPLITS, 16)
    hip_bfloat16 output[][512]                   // (batch * 16, 512)
) {
    constexpr int V_DIM = 512;
    constexpr int HEADS = 16;

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    const int v_dim0 = tid * 2;
    const int v_dim1 = tid * 2 + 1;

    // Pass 1: Find max LSE across all splits (no array needed)
    float max_lse = -INFINITY;
    for (int s = 0; s < KV_SPLITS; s++) {
        float lse = partial_lse[batch_idx * KV_SPLITS + s][head_idx];
        max_lse = fmaxf(max_lse, lse);
    }

    // Pass 2: Accumulate weighted partials using max_lse
    float sum0 = 0.0f, sum1 = 0.0f, denom = 0.0f;
    for (int s = 0; s < KV_SPLITS; s++) {
        float lse = partial_lse[batch_idx * KV_SPLITS + s][head_idx];
        float w = expf(lse - max_lse);
        denom += w;
        sum0 += w * partial_out[batch_idx * KV_SPLITS + s][(int64_t)head_idx * V_DIM + v_dim0];
        sum1 += w * partial_out[batch_idx * KV_SPLITS + s][(int64_t)head_idx * V_DIM + v_dim1];
    }

    float inv_denom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
    output[batch_idx * HEADS + head_idx][v_dim0] = hip_bfloat16(sum0 * inv_denom);
    output[batch_idx * HEADS + head_idx][v_dim1] = hip_bfloat16(sum1 * inv_denom);
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

template <int BATCH_SIZE, int N, int STRIDE, int BS = 256, bool USE_32x32 = false>
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
    constexpr int KV_SPLITS = get_fused_kv_split<BATCH_SIZE, N>();
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
        (void)hipEventCreate(&e0); (void)hipEventCreate(&e1);
        (void)hipEventCreate(&e2); (void)hipEventCreate(&e3);
        (void)hipEventRecord(e0);
    }

    // ---- Step 1: Quantize Q to MXFP4 ----
    {
        constexpr int q_block = 64;
        constexpr int total_q_blocks = BATCH_SIZE * M * NUM_BLOCKS;
        constexpr int q_grid = (total_q_blocks + q_block - 1) / q_block;
        mla_quant_q_batched_kernel<M, K, K_HALF, NUM_BLOCKS, BATCH_SIZE, q_block>
            <<<dim3(q_grid), dim3(q_block)>>>(
            reinterpret_cast<const hip_bfloat16(*)[K]>(Q_bf16.data_ptr()),
            reinterpret_cast<uint8_t(*)[A_K_HALF]>(q_data_buf.data_ptr()),
            reinterpret_cast<uint8_t*>(q_scale_buf.data_ptr()));
    }
    if (do_profile) (void)hipEventRecord(e1);

    // ---- Step 2: Fused attention (QK^T MFMA + online softmax + V accumulation) ----
    {
        dim3 grid(BATCH_SIZE, KV_SPLITS);
        dim3 block(BS);
        mla_fused_attn_kernel<N, BS, STRIDE, KV_SPLITS, K_HALF, NUM_BLOCKS, A_K_HALF, USE_32x32>
            <<<grid, block>>>(
            reinterpret_cast<const hip_bfloat16*>(Q_bf16.data_ptr()),
            reinterpret_cast<const uint8_t(*)[A_K_HALF]>(q_data_buf.data_ptr()),
            reinterpret_cast<const uint8_t*>(q_scale_buf.data_ptr()),
            reinterpret_cast<const uint8_t(*)[K_HALF]>(KV_data.data_ptr()),
            reinterpret_cast<const uint8_t(*)[STRIDE]>(KV_scale.data_ptr()),
            sm_scale,
            reinterpret_cast<float(*)[16 * 512]>(partial_v_buf.data_ptr()),
            reinterpret_cast<float(*)[16]>(partial_lse_buf.data_ptr()));
    }
    if (do_profile) (void)hipEventRecord(e2);

    // ---- Step 3: LSE-corrected reduce across splits ----
    auto output = torch::empty({TOTAL_HEADS, V_DIM},
        torch::TensorOptions().dtype(torch::kBFloat16).device(Q_bf16.device()));
    {
        constexpr int R_BLOCK = 256;
        dim3 r_grid(BATCH_SIZE, NUM_HEADS);
        dim3 r_block(R_BLOCK);
        mla_fused_reduce_kernel<R_BLOCK, KV_SPLITS><<<r_grid, r_block>>>(
            reinterpret_cast<const float(*)[16 * 512]>(partial_v_buf.data_ptr()),
            reinterpret_cast<const float(*)[16]>(partial_lse_buf.data_ptr()),
            reinterpret_cast<hip_bfloat16(*)[512]>(output.data_ptr()));
    }

    if (do_profile) {
        (void)hipEventRecord(e3);
        (void)hipEventSynchronize(e3);
        float d01, d12, d23;
        (void)hipEventElapsedTime(&d01, e0, e1);
        (void)hipEventElapsedTime(&d12, e1, e2);
        (void)hipEventElapsedTime(&d23, e2, e3);
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
        (void)hipEventDestroy(e0); (void)hipEventDestroy(e1);
        (void)hipEventDestroy(e2); (void)hipEventDestroy(e3);
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

#define MLA_FUSED(BS_VAL, N, STR, BLOCK, USE32) \
    if (batch_size == BS_VAL && kv_seq_len == N && B_SCALE_STRIDE == STR) \
        return mla_fused_pipeline_impl<BS_VAL, N, STR, BLOCK, USE32>(Q_bf16, KV_data, KV_scale, sm_scale, profile)

    // Tuned dispatch: (batch_size, kv_seq_len, stride, block_size, use_32x32)
    // Tuned per-config dispatch (winners from BSxMFMA sweep, all 16x16x128)
    // bs=4, kv=1024: BS=256 wins (41.8µs)
    MLA_FUSED(4, 1024, 18, 256, false);
    MLA_FUSED(4, 1024, 24, 256, false);
    // bs=4, kv=8192: BS=512 wins (45.3µs, -14% vs BS=256)
    MLA_FUSED(4, 8192, 18, 512, false);
    MLA_FUSED(4, 8192, 24, 512, false);
    // bs=32, kv=1024: BS=512 wins (40.0µs, -16% vs BS=256)
    MLA_FUSED(32, 1024, 18, 512, false);
    MLA_FUSED(32, 1024, 24, 512, false);
    // bs=32, kv=8192: BS=256 wins (154µs)
    MLA_FUSED(32, 8192, 18, 256, false);
    MLA_FUSED(32, 8192, 24, 256, false);
    // bs=64, kv=1024: BS=256 wins (74.0µs)
    MLA_FUSED(64, 1024, 18, 256, false);
    MLA_FUSED(64, 1024, 24, 256, false);
    // bs=64, kv=8192: BS=128 wins (217µs, -21% vs BS=256)
    MLA_FUSED(64, 8192, 18, 128, false);
    MLA_FUSED(64, 8192, 24, 128, false);
    // bs=256, kv=1024: BS=256 wins (154µs)
    MLA_FUSED(256, 1024, 18, 256, false);
    MLA_FUSED(256, 1024, 24, 256, false);
    // bs=256, kv=8192: BS=128 wins (362µs, -63% vs BS=256)
    MLA_FUSED(256, 8192, 18, 128, false);
    MLA_FUSED(256, 8192, 24, 128, false);
    TORCH_CHECK(false, "Unsupported batch_size: ", batch_size);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mla_mxfp4_pipeline", &mla_mxfp4_pipeline);
}
'''

# Global state for compiled kernel
_torch_hip_module = None  # For PyTorch load_inline


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
            extra_cuda_cflags=['-O3', '-ffast-math', '-munsafe-fp-atomics', '--offload-arch=gfx950'],
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
    if _try_compile_hip_kernel_torch():
        return True
    return False

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)

PAGE_SIZE = 1

FP8_DTYPE = aiter_dtypes.fp8
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


# Attempt compilation at module load (disabled by default)
# Uncomment the line below to enable JIT compilation of HIP kernel
HAS_HIP_KERNEL = _try_compile_hip_kernel()

# ---------------------------------------------------------------------------
# Dispatcher: select kernel based on QKV_DTYPE
# ---------------------------------------------------------------------------
def custom_kernel(data: input_t) -> output_t:
    """
    Hybrid dispatch: MXFP4 HIP pipeline for bandwidth-bound cases (small batches),
    aiter a8w8 for compute-bound cases (large batches).

    MXFP4 pipeline wins via 4x bandwidth savings when memory-bound:
      bs=4,  kv=1k:  32us (MXFP4) vs ~118us (a8w8) -> 3.7x faster
      bs=4,  kv=8k:  70us (MXFP4) vs ~113us (a8w8) -> 1.6x faster
      bs=32, kv=1k:  50us (MXFP4) vs  ??us (a8w8)

    aiter a8w8 ASM kernel wins when compute-bound (large batch x long KV):
      bs=64,  kv=8k: 359us (MXFP4) vs ~171us (a8w8) -> a8w8 2.1x faster
      bs=256, kv=8k: 1362us (MXFP4) vs ~349us (a8w8) -> a8w8 3.9x faster
    """
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]

    # MXFP4 pipeline: wins when bandwidth-bound (small batch or short KV)
    if batch_size <= 4:
        return custom_kernel_mxfp4_qkt(data)
    if kv_seq_len <= 1024:
        return custom_kernel_mxfp4_qkt(data)

    # aiter a8w8: wins for large batch x long KV (compute-bound)
    return custom_kernel_fp8(data)
    #return custom_kernel_mxfp4_qkt(data)


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

def custom_kernel_mxfp4_qkt(data):
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    num_heads = config["num_heads"]
    kv_seq_len = config["kv_seq_len"]
    v_head_dim = config["v_head_dim"]
    sm_scale = config["sm_scale"]
    PROFILE = False

    kv_buffer_mxfp4, kv_scale_mxfp4 = kv_data["mxfp4"]
    total_q = q.shape[0]

    q_flat = q.view(batch_size * num_heads, 576).contiguous()
    kv_data_flat = kv_buffer_mxfp4.view(-1, 288).contiguous()
    kv_scale_flat = kv_scale_mxfp4.view(-1, kv_scale_mxfp4.shape[-1]).contiguous()

    output = _torch_hip_module.mla_mxfp4_pipeline(
        q_flat, kv_data_flat, kv_scale_flat,
        batch_size, kv_seq_len, sm_scale, PROFILE)

    return output.view(total_q, num_heads, v_head_dim)
