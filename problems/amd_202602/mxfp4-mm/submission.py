"""
FP4 GEMM using hardware MFMA. Optimized: no e8m0_shuffle, no unnecessary copies.
A scales read with explicit strides (column-major from dynamic_mxfp4_quant).
B scales read with shuffled offset (from input).
"""
import torch
from task import input_t, output_t

MXFP4_HIP_SOURCE = b'''
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include <cmath>

constexpr int FMT_FP4 = 4;

typedef float __attribute__((ext_vector_type(16))) float16_t;
typedef float __attribute__((ext_vector_type(4))) float4_t;
typedef int __attribute__((ext_vector_type(8))) int8_vec;
typedef uint32_t __attribute__((ext_vector_type(4))) uint128_vec;

// =====================================================================
// Addressing helpers
// =====================================================================

template <int SCALE_N>
__device__ __forceinline__ int sh_scale_off(int row, int col) {
    return (row%32)/16
         + (col%8)/4 * 2
         + (row%16)  * 4
         + (col%4)   * 64
         + (col/8)   * 256
         + (row/32)  * 32 * SCALE_N;
}

template <int M, int K, int OUTER_M, int OUTER_K, int BLOCK_SIZE>
struct QuantAPerThread {
    static constexpr int OK_BLOCKS = OUTER_K / 32;
    static constexpr int TOTAL_BLOCKS = OUTER_M * OK_BLOCKS;
    static constexpr int BLOCKS_PER_THREAD = (TOTAL_BLOCKS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // 4 uint32 (16 packed bytes) + 1 scale per block
    static constexpr int DATA_REGS = BLOCKS_PER_THREAD * 4;
    static constexpr int SCALE_REGS = BLOCKS_PER_THREAD;
};

// =====================================================================
// Reusable FP4 quantization: 32 floats -> 16 packed bytes + E8M0 scale
// =====================================================================

struct QuantBlock {
    uint32_t data[4];  // 16 packed bytes (32 FP4 values)
    uint8_t e8m0;
};

__device__ __forceinline__ void load_bf16x32(const hip_bfloat16* ptr, float vals[32]) {
    for (int chunk_off = 0; chunk_off < 32; chunk_off += 8) {
        // Load 8 BF16 values at once (16 bytes)
        uint32_t raw[4];
        *reinterpret_cast<uint128_vec*>(&raw[0]) =
            *reinterpret_cast<const uint128_vec*>(&ptr[chunk_off]);
        // Unpack BF16 to float: bf16 bits << 16 = float bits
        for (int j = 0; j < 4; j++) {
            uint32_t pair = raw[j];
            float2 lo_hi = {__uint_as_float((pair & 0xFFFF) << 16),
                            __uint_as_float(pair & 0xFFFF0000u)};
            *(&reinterpret_cast<float2*>(&vals[chunk_off])[j]) = lo_hi;
        }
    }
}

__device__ __forceinline__ QuantBlock quantize_fp4_block(const float vals[32]) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++)
        amax = fmaxf(amax, fabsf(vals[i]));

    uint8_t e8m0;
    float quant_scale;
    if (amax == 0.0f) {
        e8m0 = 0;
        quant_scale = 0.0f;
    } else {
        uint32_t amax_bits = __float_as_uint(amax);
        amax_bits = (amax_bits + 0x200000u) & 0xFF800000u;
        int raw_exp = (int)((amax_bits >> 23) & 0xFF);
        int e8m0_unbiased = raw_exp - 127 - 2;
        e8m0_unbiased = max(-127, min(127, e8m0_unbiased));
        e8m0 = (uint8_t)(e8m0_unbiased + 127);
        quant_scale = __uint_as_float((uint32_t)(127 - e8m0_unbiased) << 23);
    }

    uint32_t pack[4] = {0, 0, 0, 0};
    for (int i = 0; i < 16; i++) {
        uint8_t packed = 0;
        for (int j = 0; j < 2; j++) {
            float v = vals[2 * i + j];
            float qx = v * quant_scale;

            uint32_t qx_bits = __float_as_uint(qx);
            uint32_t sign = qx_bits & 0x80000000u;
            qx_bits ^= sign;
            float qx_abs = __uint_as_float(qx_bits);

            uint8_t fp4;
            if (qx_abs >= 6.0f) {
                fp4 = 0x7;
            } else if (qx_abs < 1.0f) {
                constexpr uint32_t denorm_magic = 149u << 23;
                float denorm = qx_abs + __uint_as_float(denorm_magic);
                uint32_t denorm_bits = __float_as_uint(denorm) - denorm_magic;
                fp4 = (uint8_t)denorm_bits;
            } else {
                uint32_t mant_odd = (qx_bits >> (23 - 1)) & 1;
                constexpr int32_t val_to_add = 0xC11FFFFF;
                qx_bits = (uint32_t)((int32_t)qx_bits + val_to_add);
                qx_bits += mant_odd;
                fp4 = (uint8_t)(qx_bits >> (23 - 1));
            }

            uint8_t sign_fp4 = (uint8_t)(sign >> (23 + 8 - 1 - 2));
            fp4 |= sign_fp4;

            packed |= fp4 << (4 * j);
        }
        pack[i / 4] |= ((uint32_t)packed) << ((i % 4) * 8);
    }

    QuantBlock result;
    *reinterpret_cast<uint128_vec*>(&result.data) = *reinterpret_cast<uint128_vec*>(&pack);
    result.e8m0 = e8m0;
    return result;
}

template <int M, int K, int OUTER_M, int OUTER_K, int BLOCK_SIZE>
__device__ __forceinline__ void quantize_a_to_reg(
    const hip_bfloat16* __restrict__ A,
    int outer_m, int tid,
    uint32_t* data_regs, uint8_t* scale_regs,
    int k_offset
) {
    constexpr int OK_BLOCKS = OUTER_K / 32;
    constexpr int TOTAL_BLOCKS = OUTER_M * OK_BLOCKS;

    for (int b = tid, bi = 0; b < TOTAL_BLOCKS; b += BLOCK_SIZE, bi++) {
        int row = b / OK_BLOCKS;
        int blk = b % OK_BLOCKS;
        int g_m = outer_m + row;
        int g_k = k_offset + blk * 32;
        //bool valid = (g_m < M);

        float vals[32];
        load_bf16x32(&A[g_m * K + g_k], vals);
        QuantBlock qb = quantize_fp4_block(vals);
        *reinterpret_cast<uint128_vec*>(&data_regs[bi * 4]) = *reinterpret_cast<uint128_vec*>(&qb.data);
        scale_regs[bi] = qb.e8m0;
    }
}

template <int OUTER_M, int OUTER_K, int BLOCK_SIZE>
__device__ __forceinline__ void store_quant_a_to_lds(
    uint8_t* smem_data, uint8_t* smem_scale,
    int tid,
    const uint32_t* data_regs, const uint8_t* scale_regs
) {
    constexpr int OK_HALF = OUTER_K / 2;
    constexpr int OK_BLOCKS = OUTER_K / 32;
    constexpr int TOTAL_BLOCKS = OUTER_M * OK_BLOCKS;

    for (int b = tid, bi = 0; b < TOTAL_BLOCKS; b += BLOCK_SIZE, bi++) {
        int row = b / OK_BLOCKS;
        int blk = b % OK_BLOCKS;
        int data_off = row * OK_HALF + blk * 16;

        *reinterpret_cast<uint128_vec*>(&smem_data[data_off]) = *reinterpret_cast<const uint128_vec*>(&data_regs[bi * 4]);
        smem_scale[row * OK_BLOCKS + blk] = scale_regs[bi];
    }
}

// Wrapper for drop-in replacement
template <int M, int K, int OUTER_M, int OUTER_K, int BLOCK_SIZE>
__device__ __forceinline__ void quantize_a_to_lds(
    const hip_bfloat16* __restrict__ A,
    uint8_t* smem_data, uint8_t* smem_scale,
    int outer_m, int tid
) {
    using Q = QuantAPerThread<M, K, OUTER_M, OUTER_K, BLOCK_SIZE>;
    uint32_t data_regs[Q::DATA_REGS];
    uint8_t  scale_regs[Q::SCALE_REGS];

    quantize_a_to_reg<M, K, OUTER_M, OUTER_K, BLOCK_SIZE>(
        A, outer_m, tid, data_regs, scale_regs, 0);

    store_quant_a_to_lds<OUTER_M, OUTER_K, BLOCK_SIZE>(
        smem_data, smem_scale, tid, data_regs, scale_regs);
}

__device__ __forceinline__ int32_t broadcast_scale(uint8_t e8m0) {
    return (int32_t)e8m0 * 0x01010101;
}

template <int IM, int K_HALF_STRIDE, int REGS = (IM == 32) ? 4 : 8>
__device__ __forceinline__ void load_tile(
    const uint8_t* src, int row_base, int blk0, int lane,
    uint32_t reg[REGS], int& blk_out, int& row_out
) {
    row_out = row_base + (lane % IM);
    int k_group = lane / IM;
    blk_out = blk0 + k_group;
    int off = row_out * K_HALF_STRIDE + blk_out * 16;
    *reinterpret_cast<uint128_vec*>(&reg[0]) = *reinterpret_cast<const uint128_vec*>(&src[off]);
}

template <int IM, int OUTER_N, int REGS = (IM == 32) ? 4 : 8>
__device__ __forceinline__ void load_transposed(
    const uint8_t* src, int row_base, int blk0, int lane,
    uint32_t reg[REGS], int& blk_out, int& row_out
) {
    row_out = row_base + (lane % IM);
    int k_group = lane / IM;
    blk_out = blk0 + k_group;
    int off = (blk_out * OUTER_N + row_out) * 16;
    *reinterpret_cast<uint128_vec*>(&reg[0]) = *reinterpret_cast<const uint128_vec*>(&src[off]);
}

// =====================================================================
// MFMA traits: specialize per tile size
// =====================================================================

template <int IM, int IN, int IK>
struct MfmaTraits {
    using acc_t = typename std::conditional<IM == 32, float16_t, float4_t>::type;
    static constexpr int ACC_SIZE = (IM == 32) ? 16 : 4;
    static constexpr int BLOCKS_PER_CALL = IK / 32;
    static constexpr int REGS = (IM == 32) ? 4 : 8;

    static __device__ __forceinline__ acc_t zero_acc() {
        return acc_t{};
    }

    static __device__ __forceinline__ acc_t mfma(
        uint32_t a_reg[REGS], int32_t a_sc,
        uint32_t b_reg[REGS], int32_t b_sc,
        acc_t acc
    ) {
        if constexpr (IM == 32) {
            int8_vec a_vec = {(int)a_reg[0], (int)a_reg[1], (int)a_reg[2], (int)a_reg[3]};
            int8_vec b_vec = {(int)b_reg[0], (int)b_reg[1], (int)b_reg[2], (int)b_reg[3]};
            return __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
                a_vec, b_vec, acc, FMT_FP4, FMT_FP4, 0, a_sc, 0, b_sc);
        } else {
            int8_vec a_vec = {(int)a_reg[0], (int)a_reg[1], (int)a_reg[2], (int)a_reg[3],
                              (int)a_reg[4], (int)a_reg[5], (int)a_reg[6], (int)a_reg[7]};
            int8_vec b_vec = {(int)b_reg[0], (int)b_reg[1], (int)b_reg[2], (int)b_reg[3],
                              (int)b_reg[4], (int)b_reg[5], (int)b_reg[6], (int)b_reg[7]};
            return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                a_vec, b_vec, acc, FMT_FP4, FMT_FP4, 0, a_sc, 0, b_sc);
        }
    }

    static __device__ __forceinline__ void store(
        hip_bfloat16* C, const acc_t& acc,
        int tile_m, int tile_n, int lane, int M, int N
    ) {
        if constexpr (IM == 32) {
            int col = lane % 32;
            int half = lane / 32;
            for (int i = 0; i < 16; i++) {
                int row = (i % 4) + 4 * half + 8 * (i / 4);
                int gm = tile_m + row;
                int gn = tile_n + col;
                if (gm < M && gn < N)
                    C[gm * N + gn] = static_cast<hip_bfloat16>(acc[i]);
            }
        } else {
            int col = lane % 16;
            int quad = lane / 16;
            for (int i = 0; i < 4; i++) {
                int row = i + 4 * quad;
                int gm = tile_m + row;
                int gn = tile_n + col;
                if (gm < M && gn < N)
                    C[gm * N + gn] = static_cast<hip_bfloat16>(acc[i]);
            }
        }
    }
};

template <int M, int N, int K_HALF, int NUM_BLOCKS, int SCALE_N,
          int IM, int IN, int IK>
__device__ __forceinline__ void load_ab_global(
    const uint8_t* __restrict__ A_data,
    const uint8_t* __restrict__ B_data,
    const uint8_t* __restrict__ A_scale,
    const uint8_t* __restrict__ B_scale,
    int tile_m, int tile_n, int blk0, int lane,
    uint32_t a_r[MfmaTraits<IM,IN,IK>::REGS], int32_t& a_s,
    uint32_t b_r[MfmaTraits<IM,IN,IK>::REGS], int32_t& b_s
) {
    using Traits = MfmaTraits<IM, IN, IK>;
    int a_row, a_blk, b_row, b_blk;

    load_tile<IM, K_HALF>(A_data, tile_m, blk0, lane,
                            a_r, a_blk, a_row);
    uint8_t a_e = A_scale[a_row + a_blk * M];
    a_s = broadcast_scale(a_e);

    load_tile<IM, K_HALF>(B_data, tile_n, blk0, lane,
                            b_r, b_blk, b_row);
    uint8_t b_e = B_scale[sh_scale_off<SCALE_N>(b_row, b_blk)];
    b_s = broadcast_scale(b_e);
}

// =====================================================================
// Standalone A quantization kernel: BF16 -> MXFP4 (packed FP4 + E8M0 scales)
// One thread per 32-element MX block.
// Output data: row-major [M, K/2] packed uint8
// Output scale: column-major [row + blk * M] uint8
// =====================================================================

template <int M, int K, int K_HALF, int NUM_BLOCKS>
__global__ void quant_a_kernel(
    const hip_bfloat16* __restrict__ A,
    uint8_t* __restrict__ out_data,
    uint8_t* __restrict__ out_scale
) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int total = M * NUM_BLOCKS;
    //if (gid >= total) return;

    int row = gid / NUM_BLOCKS;
    int blk = gid % NUM_BLOCKS;
    int k_start = blk * 32;

    float vals[32];
    load_bf16x32(&A[row * K + k_start], vals);
    QuantBlock qb = quantize_fp4_block(vals);

    int data_off = row * K_HALF + blk * 16;
    *reinterpret_cast<uint128_vec*>(&out_data[data_off]) =
        *reinterpret_cast<uint128_vec*>(&qb.data);
    out_scale[row + blk * M] = qb.e8m0;
}

// =====================================================================
// Simple kernel: 1 wavefront, direct global->register, double-buffered
// =====================================================================

template <int M, int N, int K_HALF, int NUM_BLOCKS, int SCALE_N,
          int IM, int IN, int IK,
          int WARPS_M = 1, int WARPS_N = 1>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 64) mfma_fp4_gemm_simple(
    const uint8_t* __restrict__ A_data,
    const uint8_t* __restrict__ B_data,
    const uint8_t* __restrict__ A_scale,
    const uint8_t* __restrict__ B_scale,
    hip_bfloat16* __restrict__ C
) {
    using Traits = MfmaTraits<IM, IN, IK>;
    constexpr int BPC = Traits::BLOCKS_PER_CALL;
    constexpr int K_ITERS = (NUM_BLOCKS + BPC - 1) / BPC;

    const int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
    const int lane = threadIdx.x % 64;
    const int warp_m = __builtin_amdgcn_readfirstlane(warp_id / WARPS_N);
    const int warp_n = __builtin_amdgcn_readfirstlane(warp_id % WARPS_N);
    const int tile_m = __builtin_amdgcn_readfirstlane(blockIdx.x * (IM * WARPS_M)) + warp_m * IM;
    const int tile_n = __builtin_amdgcn_readfirstlane(blockIdx.y * (IN * WARPS_N)) + warp_n * IN;

    auto acc = Traits::zero_acc();

    if constexpr (K_ITERS > 0) {
        uint32_t a_cur[Traits::REGS], b_cur[Traits::REGS];
        uint32_t a_nxt[Traits::REGS], b_nxt[Traits::REGS];
        int32_t a_sc_cur, b_sc_cur, a_sc_nxt, b_sc_nxt;

        load_ab_global<M, N, K_HALF, NUM_BLOCKS, SCALE_N, IM, IN, IK>(
            A_data, B_data, A_scale, B_scale,
            tile_m, tile_n, 0, lane,
            a_cur, a_sc_cur, b_cur, b_sc_cur);

        for (int ki = 0; ki < K_ITERS; ki++) {
            if (ki + 1 < K_ITERS) {
                load_ab_global<M, N, K_HALF, NUM_BLOCKS, SCALE_N, IM, IN, IK>(
                    A_data, B_data, A_scale, B_scale,
                    tile_m, tile_n, (ki + 1) * BPC, lane,
                    a_nxt, a_sc_nxt, b_nxt, b_sc_nxt);
            }

            acc = Traits::mfma(a_cur, a_sc_cur, b_cur, b_sc_cur, acc);

            for (int r = 0; r < Traits::REGS; r++) {
                a_cur[r] = a_nxt[r];
                b_cur[r] = b_nxt[r];
            }
            a_sc_cur = a_sc_nxt;
            b_sc_cur = b_sc_nxt;
        }
    }

    Traits::store(C, acc, tile_m, tile_n, lane, M, N);
}


// =====================================================================
// Tiled kernel helpers: load global -> registers, store registers -> LDS
// =====================================================================

template <int OUTER_M, int OUTER_N, int OUTER_K>
struct LdsLayout {
    static constexpr int LIMIT    = 160 * 1024;
    static constexpr int K_HALF   = OUTER_K / 2;
    static constexpr int K_BLOCKS = OUTER_K / 32;
    static constexpr int A_DATA   = OUTER_M * K_HALF;
    static constexpr int A_SCALE  = OUTER_M * K_BLOCKS;
    static constexpr int B_DATA   = OUTER_N * K_HALF;
    static constexpr int B_SCALE  = OUTER_N * K_BLOCKS;
    static constexpr int TOTAL    = A_DATA + A_SCALE + B_DATA + B_SCALE;
    static constexpr int OCCUPANCY= LIMIT / TOTAL;
};

// ---------- Part 1: Load A from global memory into registers ----------
template <int M, int K_HALF, int NUM_BLOCKS,
          int OUTER_M, int OUTER_K, int BLOCK_SIZE>
__device__ __forceinline__ void load_a_global_to_reg(
    const uint8_t* __restrict__ A_data,
    const uint8_t* __restrict__ A_scale,
    int outer_m, int k_half_base, int blk_base, int tid,
    uint32_t* data_regs, uint8_t* scale_regs
) {
    constexpr int OK_HALF = OUTER_K / 2;
    constexpr int OK_BLOCKS = OUTER_K / 32;
    // Now iterate over 16-byte chunks (4 uint32 = one MX block's packed data)
    constexpr int DATA_CHUNKS = OUTER_M * OK_BLOCKS;  // one 16-byte chunk per block
    constexpr int SCALE_ELEMS = OUTER_M * OK_BLOCKS;

    int di = 0;
    for (int c = tid; c < DATA_CHUNKS; c += BLOCK_SIZE, di++) {
        int row = c / OK_BLOCKS;
        int blk = c % OK_BLOCKS;
        int g_m = outer_m + row;
        int g_col = k_half_base + blk * 16;
        uint128_vec val = {0, 0, 0, 0};
        val = *reinterpret_cast<const uint128_vec*>(&A_data[g_m * K_HALF + g_col]);
        *reinterpret_cast<uint128_vec*>(&data_regs[di * 4]) = val;
    }

    int si = 0;
    for (int s = tid; s < SCALE_ELEMS; s += BLOCK_SIZE, si++) {
        int row = s / OK_BLOCKS;
        int col = s % OK_BLOCKS;
        int g_m = outer_m + row;
        int g_blk = blk_base + col;
        uint8_t val = 127;
        val = A_scale[g_m + g_blk * M];
        scale_regs[si] = val;
    }
}

// ---------- Part 2: Store A from registers into LDS ----------
template <int OUTER_M, int OUTER_K, int BLOCK_SIZE>
__device__ __forceinline__ void store_a_reg_to_lds(
    uint8_t* smem_data, uint8_t* smem_scale,
    int tid,
    const uint32_t* data_regs, const uint8_t* scale_regs
) {
    constexpr int OK_HALF = OUTER_K / 2;
    constexpr int OK_BLOCKS = OUTER_K / 32;
    constexpr int DATA_CHUNKS = OUTER_M * OK_BLOCKS;
    constexpr int SCALE_ELEMS = OUTER_M * OK_BLOCKS;

    int di = 0;
    for (int c = tid; c < DATA_CHUNKS; c += BLOCK_SIZE, di++) {
        int row = c / OK_BLOCKS;
        int blk = c % OK_BLOCKS;
        int data_off = row * OK_HALF + blk * 16;
        *reinterpret_cast<uint128_vec*>(&smem_data[data_off]) =
            *reinterpret_cast<const uint128_vec*>(&data_regs[di * 4]);
    }

    int si = 0;
    for (int s = tid; s < SCALE_ELEMS; s += BLOCK_SIZE, si++) {
        smem_scale[s] = scale_regs[si];
    }
}

// ---------- Wrapper for A ----------
template <int M, int K_HALF, int NUM_BLOCKS,
          int OUTER_M, int OUTER_K, int BLOCK_SIZE>
__device__ __forceinline__ void load_a_to_lds(
    const uint8_t* __restrict__ A_data,
    const uint8_t* __restrict__ A_scale,
    uint8_t* smem_data, uint8_t* smem_scale,
    int outer_m, int k_half_base, int blk_base, int tid
) {
    constexpr int OK_HALF = OUTER_K / 2;
    constexpr int OK_BLOCKS = OUTER_K / 32;
    constexpr int DATA_CHUNKS = OUTER_M * OK_BLOCKS;

    for (int c = tid; c < DATA_CHUNKS; c += BLOCK_SIZE) {
        int row = c / OK_BLOCKS;
        int blk = c % OK_BLOCKS;
        int g_m = outer_m + row;
        int g_col = k_half_base + blk * 16;
        int lds_off = row * OK_HALF + blk * 16;
        *reinterpret_cast<uint128_vec*>(&smem_data[lds_off]) =
            *reinterpret_cast<const uint128_vec*>(&A_data[g_m * K_HALF + g_col]);
    }

    for (int s = tid; s < OUTER_M * OK_BLOCKS; s += BLOCK_SIZE) {
        int row = s / OK_BLOCKS;
        int col = s % OK_BLOCKS;
        int g_m = outer_m + row;
        int g_blk = blk_base + col;
        smem_scale[s] = A_scale[g_m + g_blk * M];
    }
}

// ---------- Load B from global memory into registers ----------
template <int N, int K_HALF, int NUM_BLOCKS, int SCALE_N,
          int OUTER_N, int OUTER_K, int BLOCK_SIZE>
__device__ __forceinline__ void load_b_global_to_reg(
    const uint8_t* __restrict__ B_data,
    const uint8_t* __restrict__ B_scale,
    int outer_n, int k_half_base, int blk_base, int tid,
    uint32_t* data_regs, uint8_t* scale_regs
) {
    constexpr int OK_BLOCKS = OUTER_K / 32;
    constexpr int DATA_CHUNKS = OUTER_N * OK_BLOCKS;
    constexpr int SCALE_ELEMS = OUTER_N * OK_BLOCKS;

    int di = 0;
    for (int c = tid; c < DATA_CHUNKS; c += BLOCK_SIZE, di++) {
        int row = c / OK_BLOCKS;
        int blk = c % OK_BLOCKS;
        int g_n = outer_n + row;
        int g_col = k_half_base + blk * 16;
        *reinterpret_cast<uint128_vec*>(&data_regs[di * 4]) =
            *reinterpret_cast<const uint128_vec*>(&B_data[g_n * K_HALF + g_col]);
    }

    int si = 0;
    for (int s = tid; s < SCALE_ELEMS; s += BLOCK_SIZE, si++) {
        int row = s / OK_BLOCKS;
        int col = s % OK_BLOCKS;
        int g_n = outer_n + row;
        int g_blk = blk_base + col;
        scale_regs[si] = B_scale[sh_scale_off<SCALE_N>(g_n, g_blk)];
    }
}

// ---------- Store B from registers into LDS ----------
template <int OUTER_N, int OUTER_K, int BLOCK_SIZE>
__device__ __forceinline__ void store_b_reg_to_lds(
    uint8_t* smem_data, uint8_t* smem_scale,
    int tid,
    const uint32_t* data_regs, const uint8_t* scale_regs
) {
    constexpr int OK_BLOCKS = OUTER_K / 32;
    constexpr int DATA_CHUNKS = OUTER_N * OK_BLOCKS;
    constexpr int SCALE_ELEMS = OUTER_N * OK_BLOCKS;

    int di = 0;
    for (int c = tid; c < DATA_CHUNKS; c += BLOCK_SIZE, di++) {
        int row = c / OK_BLOCKS;
        int blk = c % OK_BLOCKS;
        // Block-transposed: [blk][row][16 bytes]
        int lds_off = (blk * OUTER_N + row) * 16;
        *reinterpret_cast<uint128_vec*>(&smem_data[lds_off]) =
            *reinterpret_cast<const uint128_vec*>(&data_regs[di * 4]);
    }

    int si = 0;
    for (int s = tid; s < SCALE_ELEMS; s += BLOCK_SIZE, si++) {
        int row = s / OK_BLOCKS;
        int col = s % OK_BLOCKS;
        smem_scale[col * OUTER_N + row] = scale_regs[si];
    }
}

// ---------- Wrapper for B ----------
template <int N, int K_HALF, int NUM_BLOCKS, int SCALE_N,
          int OUTER_N, int OUTER_K, int BLOCK_SIZE>
__device__ __forceinline__ void load_b_to_lds(
    const uint8_t* __restrict__ B_data,
    const uint8_t* __restrict__ B_scale,
    uint8_t* smem_data, uint8_t* smem_scale,
    int outer_n, int k_half_base, int blk_base, int tid
) {
    constexpr int OK_BLOCKS = OUTER_K / 32;
    constexpr int DATA_CHUNKS = OUTER_N * OK_BLOCKS;

    for (int c = tid; c < DATA_CHUNKS; c += BLOCK_SIZE) {
        int row = c / OK_BLOCKS;
        int blk = c % OK_BLOCKS;
        int g_n = outer_n + row;
        int g_col = k_half_base + blk * 16;
        int lds_off = (blk * OUTER_N + row) * 16;
        *reinterpret_cast<uint128_vec*>(&smem_data[lds_off]) =
            *reinterpret_cast<const uint128_vec*>(&B_data[g_n * K_HALF + g_col]);
    }

    for (int s = tid; s < OUTER_N * OK_BLOCKS; s += BLOCK_SIZE) {
        int row = s / OK_BLOCKS;
        int col = s % OK_BLOCKS;
        int g_n = outer_n + row;
        int g_blk = blk_base + col;
        smem_scale[col * OUTER_N + row] = B_scale[sh_scale_off<SCALE_N>(g_n, g_blk)];
    }
}

template <int OUTER_K, int OUTER_N, int IM, int IN, int IK,
          int WARP_TILES_M = 1, int WARP_TILES_N = 1>
__device__ __forceinline__ void inner_mfma_loop(
    const uint8_t* smem_a_data, const uint8_t* smem_a_scale,
    const uint8_t* smem_b_data, const uint8_t* smem_b_scale,
    int warp_m, int warp_n, int lane,
    typename MfmaTraits<IM, IN, IK>::acc_t (&acc)[WARP_TILES_M][WARP_TILES_N]
) {
    using Traits = MfmaTraits<IM, IN, IK>;
    constexpr int OK_HALF = OUTER_K / 2;
    constexpr int OK_BLOCKS = OUTER_K / 32;
    constexpr int BPC = Traits::BLOCKS_PER_CALL;
    constexpr int ITERS = OUTER_K / IK;

    for (int ik = 0; ik < ITERS; ik++) {
        int blk0 = ik * BPC;

        if constexpr (IM >= IN) {
            // Reuse A: outer loop M, inner loop N
            for (int wt_m = 0; wt_m < WARP_TILES_M; wt_m++) {
                uint32_t a_reg[Traits::REGS];
                int a_row, a_blk;
                load_tile<IM, OK_HALF>(smem_a_data,
                    (warp_m * WARP_TILES_M + wt_m) * IM,
                    blk0, lane, a_reg, a_blk, a_row);
                int32_t a_sc = broadcast_scale(
                    smem_a_scale[a_row * OK_BLOCKS + a_blk]);

                for (int wt_n = 0; wt_n < WARP_TILES_N; wt_n++) {
                    uint32_t b_reg[Traits::REGS];
                    int b_row, b_blk;
                    load_transposed<IM, OUTER_N>(smem_b_data,
                        (warp_n * WARP_TILES_N + wt_n) * IN,
                        blk0, lane, b_reg, b_blk, b_row);
                    int32_t b_sc = broadcast_scale(
                        smem_b_scale[b_blk * OUTER_N + b_row]);

                    acc[wt_m][wt_n] = Traits::mfma(
                        a_reg, a_sc, b_reg, b_sc, acc[wt_m][wt_n]);
                }
            }
        } else {
            // Reuse B: outer loop N, inner loop M
            for (int wt_n = 0; wt_n < WARP_TILES_N; wt_n++) {
                uint32_t b_reg[Traits::REGS];
                int b_row, b_blk;
                load_transposed<IM, OUTER_N>(smem_b_data,
                    (warp_n * WARP_TILES_N + wt_n) * IN,
                    blk0, lane, b_reg, b_blk, b_row);
                int32_t b_sc = broadcast_scale(
                    smem_b_scale[b_blk * OUTER_N + b_row]);

                for (int wt_m = 0; wt_m < WARP_TILES_M; wt_m++) {
                    uint32_t a_reg[Traits::REGS];
                    int a_row, a_blk;
                    load_tile<IM, OK_HALF>(smem_a_data,
                        (warp_m * WARP_TILES_M + wt_m) * IM,
                        blk0, lane, a_reg, a_blk, a_row);
                    int32_t a_sc = broadcast_scale(
                        smem_a_scale[a_row * OK_BLOCKS + a_blk]);

                    acc[wt_m][wt_n] = Traits::mfma(
                        a_reg, a_sc, b_reg, b_sc, acc[wt_m][wt_n]);
                }
            }
        }
    }
}


// =====================================================================
// Tiled kernel: multi-wavefront, LDS-backed
// =====================================================================

template <int WARPS, int M, int N, int K_HALF, int NUM_BLOCKS, int SCALE_N,
          int OUTER_M, int OUTER_N, int OUTER_K,
          int IM, int IN, int IK,
          int WARP_TILES_M = 1, int WARP_TILES_N = 1,
          bool FUSE_A_QUANT = false, int BUFFERS = 1,
          int OCCUPANCY = -1>
__global__ void
__launch_bounds__(WARPS * 64, OCCUPANCY == -1 ? (LdsLayout<OUTER_M, OUTER_N, OUTER_K>::OCCUPANCY) / BUFFERS : OCCUPANCY)
mfma_fp4_gemm_tiled(
    const uint8_t* __restrict__ A_data,
    const uint8_t* __restrict__ B_data,
    const uint8_t* __restrict__ A_scale,
    const uint8_t* __restrict__ B_scale,
    hip_bfloat16* __restrict__ C,
    const hip_bfloat16* __restrict__ A_bf16
) {
    constexpr int WARPS_M = OUTER_M / (IM * WARP_TILES_M);
    constexpr int WARPS_N = OUTER_N / (IN * WARP_TILES_N);
    static_assert(WARPS == WARPS_M * WARPS_N);

    using Traits = MfmaTraits<IM, IN, IK>;
    using Lds = LdsLayout<OUTER_M, OUTER_N, OUTER_K>;
    constexpr int OK_BLOCKS = OUTER_K / 32;
    constexpr int OK_HALF = OUTER_K / 2;
    constexpr int K = K_HALF * 2;
    constexpr int OUTER_K_ITERS = (NUM_BLOCKS + OK_BLOCKS - 1) / OK_BLOCKS;
    constexpr int BLOCK_SIZE = WARPS * 64;
    constexpr int A_DATA_PER_THREAD = ((OUTER_M * OK_BLOCKS) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int A_CHUNKS_PER_THREAD = ((OUTER_M * OK_BLOCKS) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int A_SCALE_ELEMS = OUTER_M * OK_BLOCKS;
    constexpr int A_SCALE_PER_THREAD = (A_SCALE_ELEMS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int B_DATA_ELEMS = OUTER_N * OK_HALF / 4;
    constexpr int B_DATA_PER_THREAD = (B_DATA_ELEMS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int B_SCALE_ELEMS = OUTER_N * OK_BLOCKS;
    constexpr int B_SCALE_PER_THREAD = (B_SCALE_ELEMS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const int outer_m = __builtin_amdgcn_readfirstlane(blockIdx.x * OUTER_M);
    const int outer_n = __builtin_amdgcn_readfirstlane(blockIdx.y * OUTER_N);
    const int tid = threadIdx.x;
    const int warp_id = __builtin_amdgcn_readfirstlane(tid / 64);
    const int lane = tid % 64;
    const int warp_m = __builtin_amdgcn_readfirstlane(warp_id / WARPS_N);
    const int warp_n = __builtin_amdgcn_readfirstlane(warp_id % WARPS_N);

    __shared__ uint8_t smem_a_data[BUFFERS][Lds::A_DATA];
    __shared__ uint8_t smem_a_scale[BUFFERS][Lds::A_SCALE];
    __shared__ uint8_t smem_b_data[BUFFERS][Lds::B_DATA];
    __shared__ uint8_t smem_b_scale[BUFFERS][Lds::B_SCALE];

    typename Traits::acc_t acc[WARP_TILES_M][WARP_TILES_N];
    for (int wt_m = 0; wt_m < WARP_TILES_M; wt_m++)
        for (int wt_n = 0; wt_n < WARP_TILES_N; wt_n++)
            acc[wt_m][wt_n] = Traits::zero_acc();

    // Load the first tile
    if constexpr (FUSE_A_QUANT) {
        quantize_a_to_lds<M, K, OUTER_M, OUTER_K, BLOCK_SIZE>(
            A_bf16, smem_a_data[0], smem_a_scale[0], outer_m, tid);
    } else {
        load_a_to_lds<M, K_HALF, NUM_BLOCKS, OUTER_M, OUTER_K, BLOCK_SIZE>(
            A_data, A_scale,
            smem_a_data[0], smem_a_scale[0],
            outer_m, 0, 0, tid);
    }

    load_b_to_lds<N, K_HALF, NUM_BLOCKS, SCALE_N, OUTER_N, OUTER_K, BLOCK_SIZE>(
        B_data, B_scale,
        smem_b_data[0], smem_b_scale[0],
        outer_n, 0, 0, tid);

    for (int ok = 0; ok < OUTER_K_ITERS - 1; ok++) {
        if constexpr (WARPS > 1) {
            __syncthreads();
        }

        uint32_t a_data_regs[A_CHUNKS_PER_THREAD * 4];
        uint8_t  a_scale_regs[A_SCALE_PER_THREAD];
        uint32_t b_data_regs[B_DATA_PER_THREAD];
        uint8_t  b_scale_regs[B_SCALE_PER_THREAD];

        using Q = QuantAPerThread<M, K, OUTER_M, OUTER_K, BLOCK_SIZE>;
        uint32_t data_regs[Q::DATA_REGS];
        uint8_t  scale_regs[Q::SCALE_REGS];

        auto next_ok = ok + 1;
        const auto buf = ok % BUFFERS;
        const auto next_buf = next_ok % BUFFERS;

        if constexpr (FUSE_A_QUANT) {
            // Quantize next A tile directly into next buffer
            // (only works cleanly with BUFFERS==2; for BUFFERS==1 need register staging)
            quantize_a_to_reg<M, K, OUTER_M, OUTER_K, BLOCK_SIZE>(
                A_bf16, outer_m, tid, data_regs, scale_regs, next_ok * OUTER_K);
        } else {
            load_a_global_to_reg<M, K_HALF, NUM_BLOCKS, OUTER_M, OUTER_K, BLOCK_SIZE>(
                A_data, A_scale,
                outer_m, next_ok * OK_HALF, next_ok * OK_BLOCKS, tid,
                a_data_regs, a_scale_regs);
        }

        load_b_global_to_reg<N, K_HALF, NUM_BLOCKS, SCALE_N, OUTER_N, OUTER_K, BLOCK_SIZE>(
            B_data, B_scale,
            outer_n, next_ok * OK_HALF, next_ok * OK_BLOCKS, tid,
            b_data_regs, b_scale_regs);

        inner_mfma_loop<OUTER_K, OUTER_N, IM, IN, IK, WARP_TILES_M, WARP_TILES_N>(
            smem_a_data[buf], smem_a_scale[buf],
            smem_b_data[buf], smem_b_scale[buf],
            warp_m, warp_n, lane, acc);

        if constexpr (BUFFERS == 1 && WARPS > 1) {
            __syncthreads();
        }

        if constexpr (FUSE_A_QUANT) {
            store_quant_a_to_lds<OUTER_M, OUTER_K, BLOCK_SIZE>(
                smem_a_data[next_buf], smem_a_scale[next_buf],
                tid, data_regs, scale_regs);
        } else {
            store_a_reg_to_lds<OUTER_M, OUTER_K, BLOCK_SIZE>(
                smem_a_data[next_buf], smem_a_scale[next_buf],
                tid, a_data_regs, a_scale_regs);
        }

        store_b_reg_to_lds<OUTER_N, OUTER_K, BLOCK_SIZE>(
            smem_b_data[next_buf], smem_b_scale[next_buf],
            tid, b_data_regs, b_scale_regs);
    }

    if constexpr (WARPS > 1) {
        __syncthreads();
    }

    constexpr auto buf = (OUTER_K_ITERS - 1) % BUFFERS;
    inner_mfma_loop<OUTER_K, OUTER_N, IM, IN, IK, WARP_TILES_M, WARP_TILES_N>(
        smem_a_data[buf], smem_a_scale[buf],
        smem_b_data[buf], smem_b_scale[buf],
        warp_m, warp_n, lane, acc);

    for (int wt_m = 0; wt_m < WARP_TILES_M; wt_m++)
        for (int wt_n = 0; wt_n < WARP_TILES_N; wt_n++)
            Traits::store(C, acc[wt_m][wt_n],
                outer_m + (warp_m * WARP_TILES_M + wt_m) * IM,
                outer_n + (warp_n * WARP_TILES_N + wt_n) * IN,
                lane, M, N);
}

// =====================================================================
// Split-K kernel: distributes K across blockIdx.z, writes fp32 partials
// =====================================================================

template <int M, int N, int K_HALF, int NUM_BLOCKS, int SCALE_N,
          int IM, int IN, int IK,
          int WARPS_M = 1, int WARPS_N = 1, int K_SPLITS = 1>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 64) mfma_fp4_gemm_splitk(
    const uint8_t* __restrict__ A_data,
    const uint8_t* __restrict__ B_data,
    const uint8_t* __restrict__ A_scale,
    const uint8_t* __restrict__ B_scale,
    float* __restrict__ workspace
) {
    using Traits = MfmaTraits<IM, IN, IK>;
    constexpr int BPC = Traits::BLOCKS_PER_CALL;
    constexpr int K_ITERS = (NUM_BLOCKS + BPC - 1) / BPC;
    constexpr int ITERS_PER_SPLIT = (K_ITERS + K_SPLITS - 1) / K_SPLITS;

    const int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
    const int lane = threadIdx.x % 64;
    const int warp_m = __builtin_amdgcn_readfirstlane(warp_id / WARPS_N);
    const int warp_n = __builtin_amdgcn_readfirstlane(warp_id % WARPS_N);
    const int tile_m = __builtin_amdgcn_readfirstlane(blockIdx.x * (IM * WARPS_M)) + warp_m * IM;
    const int tile_n = __builtin_amdgcn_readfirstlane(blockIdx.y * (IN * WARPS_N)) + warp_n * IN;
    const int split_id = __builtin_amdgcn_readfirstlane(blockIdx.z);

    const int ki_start = split_id * ITERS_PER_SPLIT;
    const int ki_end = min(ki_start + ITERS_PER_SPLIT, K_ITERS);

    auto acc = Traits::zero_acc();

    uint32_t a_cur[Traits::REGS], b_cur[Traits::REGS];
    uint32_t a_nxt[Traits::REGS], b_nxt[Traits::REGS];
    int32_t a_sc_cur, b_sc_cur, a_sc_nxt, b_sc_nxt;

    load_ab_global<M, N, K_HALF, NUM_BLOCKS, SCALE_N, IM, IN, IK>(
        A_data, B_data, A_scale, B_scale,
        tile_m, tile_n, ki_start * BPC, lane,
        a_cur, a_sc_cur, b_cur, b_sc_cur);

    for (int ki = ki_start; ki < ki_end - 1; ki++) {
        load_ab_global<M, N, K_HALF, NUM_BLOCKS, SCALE_N, IM, IN, IK>(
            A_data, B_data, A_scale, B_scale,
            tile_m, tile_n, (ki + 1) * BPC, lane,
            a_nxt, a_sc_nxt, b_nxt, b_sc_nxt);

        acc = Traits::mfma(a_cur, a_sc_cur, b_cur, b_sc_cur, acc);

        for (int r = 0; r < Traits::REGS; r++) {
            a_cur[r] = a_nxt[r];
            b_cur[r] = b_nxt[r];
        }
        a_sc_cur = a_sc_nxt;
        b_sc_cur = b_sc_nxt;
    }

    // Last tile
    acc = Traits::mfma(a_cur, a_sc_cur, b_cur, b_sc_cur, acc);

    // Store fp32 partial sums to workspace[split_id * M * N + ...]
    float* ws = workspace + split_id * M * N;
    if constexpr (IM == 32) {
        int col = lane % 32;
        int half = lane / 32;
        for (int i = 0; i < 16; i++) {
            int row = (i % 4) + 4 * half + 8 * (i / 4);
            int gm = tile_m + row;
            int gn = tile_n + col;
            //if (gm < M && gn < N)
                ws[gm * N + gn] = acc[i];
        }
    } else {
        int col = lane % 16;
        int quad = lane / 16;
        for (int i = 0; i < 4; i++) {
            int row = i + 4 * quad;
            int gm = tile_m + row;
            int gn = tile_n + col;
            //if (gm < M && gn < N)
                ws[gm * N + gn] = acc[i];
        }
    }
}

// =====================================================================
// Reduction kernel: sum fp32 partials across K splits, write bf16
// =====================================================================

template <int M, int N, int K_SPLITS>
__global__ void reduce_splitk_kernel(
    const float* __restrict__ workspace,
    hip_bfloat16* __restrict__ C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int TOTAL = M * N;
    if (idx >= TOTAL) return;

    float sum = 0.0f;
    for (int s = 0; s < K_SPLITS; s++) {
        sum += workspace[s * TOTAL + idx];
    }
    C[idx] = static_cast<hip_bfloat16>(sum);
}
'''

MXFP4_CPP_SOURCE = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <ATen/hip/HIPContext.h>

template <int M, int N, int K_HALF, int NUM_BLOCKS, int SCALE_N,
          int IM, int IN, int IK,
          int WARPS_M = 1, int WARPS_N = 1>
void launch_simple(torch::Tensor A_bf16,
                    torch::Tensor B_data, torch::Tensor B_scale,
                    torch::Tensor C) {
    // Pre-allocated scratch (allocated once, reused)
    static torch::Tensor A_data_buf, A_scale_buf;
    static bool init = false;
    if (!init) {
        auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(A_bf16.device());
        A_data_buf = torch::empty({M, K_HALF}, opts);
        A_scale_buf = torch::empty({NUM_BLOCKS * M}, opts);
        init = true;
    }

    // Launch quant kernel
    constexpr int q_total = M * NUM_BLOCKS;
    constexpr int q_block = 64;
    constexpr int K = K_HALF * 2;
    constexpr int q_grid = (q_total + q_block - 1) / q_block;
    hipLaunchKernelGGL((quant_a_kernel<M, K, K_HALF, NUM_BLOCKS>),
        dim3(q_grid), dim3(q_block), 0, 0,
        reinterpret_cast<const hip_bfloat16*>(A_bf16.data_ptr()),
        reinterpret_cast<uint8_t*>(A_data_buf.data_ptr()),
        reinterpret_cast<uint8_t*>(A_scale_buf.data_ptr()));

    dim3 grid((M + IM * WARPS_M - 1) / (IM * WARPS_M),
              (N + IN * WARPS_N - 1) / (IN * WARPS_N));
    dim3 block(64 * WARPS_M * WARPS_N);
    hipLaunchKernelGGL((mfma_fp4_gemm_simple<M,N,K_HALF,NUM_BLOCKS,SCALE_N,IM,IN,IK,WARPS_M,WARPS_N>),
        grid, block, 0, 0,
        reinterpret_cast<const uint8_t*>(A_data_buf.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_data.data_ptr()),
        reinterpret_cast<const uint8_t*>(A_scale_buf.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_scale.data_ptr()),
        reinterpret_cast<hip_bfloat16*>(C.data_ptr()));
}

template <int M, int N, int K_HALF, int NUM_BLOCKS, int SCALE_N,
          int OM, int ON, int OK, int IM, int IN, int IK,
          int WTM = 1, int WTN = 1, int BUFFERS = 1, int OCCUPANCY = -1>
void launch_tiled(torch::Tensor A_bf16,
                    torch::Tensor B_data, torch::Tensor B_scale,
                    torch::Tensor C) {
    // Pre-allocated scratch (allocated once, reused)
    static torch::Tensor A_data_buf, A_scale_buf;
    static bool init = false;
    if (!init) {
        auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(A_bf16.device());
        A_data_buf = torch::empty({M, K_HALF}, opts);
        A_scale_buf = torch::empty({NUM_BLOCKS * M}, opts);
        init = true;
    }

    // Launch quant kernel
    constexpr int q_total = M * NUM_BLOCKS;
    constexpr int q_block = 64;
    constexpr int K = K_HALF * 2;
    constexpr int q_grid = (q_total + q_block - 1) / q_block;
    hipLaunchKernelGGL((quant_a_kernel<M, K, K_HALF, NUM_BLOCKS>),
        dim3(q_grid), dim3(q_block), 0, 0,
        reinterpret_cast<const hip_bfloat16*>(A_bf16.data_ptr()),
        reinterpret_cast<uint8_t*>(A_data_buf.data_ptr()),
        reinterpret_cast<uint8_t*>(A_scale_buf.data_ptr()));

    constexpr int WARPS = (OM/(IM*WTM)) * (ON/(IN*WTN));
    dim3 grid((M+OM-1)/OM, (N+ON-1)/ON);
    dim3 block(WARPS * 64);
    hipLaunchKernelGGL((mfma_fp4_gemm_tiled<WARPS,M,N,K_HALF,NUM_BLOCKS,SCALE_N,OM,ON,OK,IM,IN,IK,WTM,WTN,false,BUFFERS,OCCUPANCY>),
        grid, block, 0, 0,
        reinterpret_cast<const uint8_t*>(A_data_buf.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_data.data_ptr()),
        reinterpret_cast<const uint8_t*>(A_scale_buf.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_scale.data_ptr()),
        reinterpret_cast<hip_bfloat16*>(C.data_ptr()),
        (const hip_bfloat16*)nullptr);
}

template <int M, int N, int K_HALF, int NUM_BLOCKS, int SCALE_N,
          int OM, int ON, int OK, int IM, int IN, int IK,
          int WTM = 1, int WTN = 1, int BUFFERS = 1, int OCCUPANCY = -1>
void launch_tiled_fused(torch::Tensor A_bf16, torch::Tensor B,
                        torch::Tensor Bs, torch::Tensor C) {
    constexpr int WARPS = (OM/(IM*WTM)) * (ON/(IN*WTN));
    dim3 grid((M+OM-1)/OM, (N+ON-1)/ON);
    dim3 block(WARPS * 64);
    hipLaunchKernelGGL((mfma_fp4_gemm_tiled<WARPS,M,N,K_HALF,NUM_BLOCKS,SCALE_N,OM,ON,OK,IM,IN,IK,WTM,WTN,true,BUFFERS,OCCUPANCY>),
        grid, block, 0, 0,
        (const uint8_t*)nullptr,
        reinterpret_cast<const uint8_t*>(B.data_ptr()),
        (const uint8_t*)nullptr,
        reinterpret_cast<const uint8_t*>(Bs.data_ptr()),
        reinterpret_cast<hip_bfloat16*>(C.data_ptr()),
        reinterpret_cast<const hip_bfloat16*>(A_bf16.data_ptr()));
}

template <int M, int N, int K_HALF, int NUM_BLOCKS, int SCALE_N,
          int IM, int IN, int IK,
          int WARPS_M = 1, int WARPS_N = 1, int K_SPLITS = 1>
void launch_splitk(torch::Tensor A_bf16,
                   torch::Tensor B_data, torch::Tensor B_scale,
                   torch::Tensor C) {
    // Pre-allocated scratch
    static torch::Tensor A_data_buf, A_scale_buf, ws_buf;
    static bool init = false;
    if (!init) {
        auto u8opts = torch::TensorOptions().dtype(torch::kUInt8).device(A_bf16.device());
        auto f32opts = torch::TensorOptions().dtype(torch::kFloat32).device(A_bf16.device());
        A_data_buf = torch::empty({M, K_HALF}, u8opts);
        A_scale_buf = torch::empty({NUM_BLOCKS * M}, u8opts);
        ws_buf = torch::empty({K_SPLITS * M * N}, f32opts);
        init = true;
    }

    // Launch quant kernel
    constexpr int q_total = M * NUM_BLOCKS;
    constexpr int q_block = 64;
    constexpr int K = K_HALF * 2;
    constexpr int q_grid = (q_total + q_block - 1) / q_block;
    hipLaunchKernelGGL((quant_a_kernel<M, K, K_HALF, NUM_BLOCKS>),
        dim3(q_grid), dim3(q_block), 0, 0,
        reinterpret_cast<const hip_bfloat16*>(A_bf16.data_ptr()),
        reinterpret_cast<uint8_t*>(A_data_buf.data_ptr()),
        reinterpret_cast<uint8_t*>(A_scale_buf.data_ptr()));

    // Launch split-K GEMM
    dim3 grid((M + IM * WARPS_M - 1) / (IM * WARPS_M),
              (N + IN * WARPS_N - 1) / (IN * WARPS_N),
              K_SPLITS);
    dim3 block(64 * WARPS_M * WARPS_N);
    hipLaunchKernelGGL((mfma_fp4_gemm_splitk<M,N,K_HALF,NUM_BLOCKS,SCALE_N,IM,IN,IK,WARPS_M,WARPS_N,K_SPLITS>),
        grid, block, 0, 0,
        reinterpret_cast<const uint8_t*>(A_data_buf.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_data.data_ptr()),
        reinterpret_cast<const uint8_t*>(A_scale_buf.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_scale.data_ptr()),
        reinterpret_cast<float*>(ws_buf.data_ptr()));

    // Launch reduction
    constexpr int r_total = M * N;
    constexpr int r_block = 256;
    constexpr int r_grid = (r_total + r_block - 1) / r_block;
    hipLaunchKernelGGL((reduce_splitk_kernel<M, N, K_SPLITS>),
        dim3(r_grid), dim3(r_block), 0, 0,
        reinterpret_cast<const float*>(ws_buf.data_ptr()),
        reinterpret_cast<hip_bfloat16*>(C.data_ptr()));
}

void mfma_gemm(
    torch::Tensor A_data, torch::Tensor B_data,
    torch::Tensor B_scale, torch::Tensor C,
    int M, int N, int K, int K_half, int num_blocks, int scaleN
) {
// Dispatch macro — add WM, WN
#define S32(m,n,kh,nb,sn,wm,wn) \
    if(M==m&&N==n&&K_half==kh){return launch_simple<m,n,kh,nb,sn,32,32,64,wm,wn>(A_data,B_data,B_scale,C);}
#define S16(m,n,kh,nb,sn,wm,wn) \
    if(M==m&&N==n&&K_half==kh){return launch_simple<m,n,kh,nb,sn,16,16,128,wm,wn>(A_data,B_data,B_scale,C);}
#define T(m,n,kh,nb,sn,om,on,ok,im,in,ik,wtm,wtn,bufs) \
    if(M==m&&N==n&&K_half==kh){return launch_tiled<m,n,kh,nb,sn,om,on,ok,im,in,ik,wtm,wtn,bufs>(A_data,B_data,B_scale,C);}
#define F(m,n,kh,nb,sn,om,on,ok,im,in,ik,wtm,wtn,bufs) \
    if(M==m&&N==n&&K_half==kh){return launch_tiled_fused<m,n,kh,nb,sn,om,on,ok,im,in,ik,wtm,wtn,bufs>(A_data,B_data,B_scale,C);}
#define F2(m,n,kh,nb,sn,om,on,ok,im,in,ik,wtm,wtn,bufs,occ) \
    if(M==m&&N==n&&K_half==kh){return launch_tiled_fused<m,n,kh,nb,sn,om,on,ok,im,in,ik,wtm,wtn,bufs,occ>(A_data,B_data,B_scale,C);}
#define SK(m,n,kh,nb,sn,im,in,ik,wm,wn,ksplits) \
    if(M==m&&N==n&&K_half==kh){return launch_splitk<m,n,kh,nb,sn,im,in,ik,wm,wn,ksplits>(A_data,B_data,B_scale,C);}

    // Simple 32x32x64
    S16(32, 4096, 256, 16, 16, 1, 1)
    S32(32, 2880, 256, 16, 16, 1, 1)
    // S32(64, 7168, 1024, 64, 64, 1, 2)
    S32(256, 3072, 768,  48, 48, 1, 1)

    // Simple 16x16x128
    S16(4,  2880, 256,  16,  16, 1, 1)
    S16(8,  2112, 3584, 224, 224, 1, 1)
    //S16(16, 2112, 3584, 224, 224, 1, 1)
    S16(16, 3072, 768,  48,  48, 1, 1)
    S16(64, 7168, 1024, 64, 64, 1, 1)

    // Tiled
    T(64,  3072, 768,  48, 48,  32,32,1536, 32,32,64, 1,1,1)
    T(256, 2880, 256,  16, 16,  128,128,512, 32,32,64, 1,1,1)
    //T(64, 7168, 1024, 64, 64, 64,32,2048, 16,16,128, 1,1,1)
    //T(256, 3072, 768,  48, 48, 32, 32, 768, 32, 32, 64, 1, 1, 1)

    // Tiled fused
    //F2(32, 4096, 256, 16, 16, 32, 32, 512, 16, 16, 128, 1, 1, 1, 1)

    // Split-K
    SK(16, 2112, 3584, 224, 224, 16,16,128, 1,1, 28)
#undef S32
#undef S16
#undef T
#undef SK

    TORCH_CHECK(false, "No template for M=", M, " N=", N, " K_half=", K_half);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mfma_gemm", &mfma_gemm);
}
'''

_hip_module = None
HAS_HIP_KERNEL = False

def _try_compile():
    global _hip_module, HAS_HIP_KERNEL
    import time, os
    try:
        if not torch.cuda.is_available(): return False
        if not hasattr(torch.version, 'hip') or torch.version.hip is None: return False
        from torch.utils.cpp_extension import load_inline
        rocm_home = os.environ.get('ROCM_HOME', '/opt/rocm')
        os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
        os.environ['MAX_JOBS'] = '4'
        src = MXFP4_HIP_SOURCE.decode('utf-8') + '\n' + MXFP4_CPP_SOURCE
        t0 = time.time()
        _hip_module = load_inline(
            name='mxfp4_mfma_v3', cpp_sources='', cuda_sources=[src],
            extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '--offload-arch=gfx950'],
            extra_include_paths=[f'{rocm_home}/include'], verbose=True)
        print(f"[mxfp4-mm] MFMA kernel compiled in {time.time()-t0:.1f}s")
        HAS_HIP_KERNEL = True
        return True
    except Exception as e:
        print(f"[mxfp4-mm] Compile failed: {e}")
        import traceback; traceback.print_exc()
        return False

try: _try_compile()
except: pass


def custom_kernel(data: input_t) -> output_t:
    global HAS_HIP_KERNEL, _hip_module

    PROFILE = False
    PROFILE_INTERVAL = 200

    A, B, B_q, B_shuffle, B_scale_sh = data
    A = A.contiguous()
    m, k = A.shape
    n, _ = B.shape
    # C = torch.empty((m, n), dtype=torch.bfloat16, device=A.device)

    if not hasattr(custom_kernel, '_graph_cache'):
        custom_kernel._graph_cache = {}

    B_data = B_q.view(torch.uint8)
    B_sc = B_scale_sh.view(torch.uint8)
    K_half = k // 2
    num_blocks = (k + 31) // 32
    scaleN = ((num_blocks + 7) // 8) * 8

    key = (m, n, k)

    if HAS_HIP_KERNEL:
        try:
            if PROFILE:
                if not hasattr(custom_kernel, '_stats'):
                    custom_kernel._stats = {}
                if key not in custom_kernel._stats:
                    custom_kernel._stats[key] = {'total': 0.0, 'count': 0}
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            if key not in custom_kernel._graph_cache:
                A_buf = torch.empty_like(A)
                B_data_buf = torch.empty_like(B_data)
                B_sc_buf = torch.empty_like(B_sc)
                C_buf = torch.empty((m, n), dtype=torch.bfloat16, device=A.device)

                A_buf.copy_(A)
                B_data_buf.copy_(B_data)
                B_sc_buf.copy_(B_sc)

                # Warmup
                _hip_module.mfma_gemm(
                    A_buf, B_data_buf, B_sc_buf, C_buf,
                    m, n, k, K_half, num_blocks, scaleN)

                # Capture
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    _hip_module.mfma_gemm(
                        A_buf, B_data_buf, B_sc_buf, C_buf,
                        m, n, k, K_half, num_blocks, scaleN)

                custom_kernel._graph_cache[key] = (g, A_buf, B_data_buf, B_sc_buf, C_buf)

            g, A_buf, B_data_buf, B_sc_buf, C_buf = custom_kernel._graph_cache[key]

            A_buf.copy_(A)
            B_data_buf.copy_(B_data)
            B_sc_buf.copy_(B_sc)

            g.replay()

            # C.copy_(C_buf)

            if PROFILE:
                end.record()
                torch.cuda.synchronize()
                s = custom_kernel._stats[key]
                s['total'] += start.elapsed_time(end)
                s['count'] += 1
                if s['count'] % PROFILE_INTERVAL == 0:
                    cnt = s['count']
                    print(f"[PROFILE] m={m:4d} n={n:4d} k={k:4d} | "
                          f"total={s['total']/cnt*1000:.1f}us  "
                          f"(avg over {cnt} calls)", flush=True)

            return C_buf
        except Exception as e:
            print(f"[mxfp4-mm] MFMA kernel failed for m={m} n={n} k={k}: {e}", flush=True)

    # # Fallback
    # x_fp4, bs_e8m0 = dynamic_mxfp4_quant(A)
    # bs_e8m0 = e8m0_shuffle(bs_e8m0)
    # A_q = x_fp4.view(dtypes.fp4x2)
    # A_scale_sh = bs_e8m0.view(dtypes.fp8_e8m0)
    # return aiter.gemm_a4w4(A_q, B_shuffle, A_scale_sh, B_scale_sh,
    #                        dtype=dtypes.bf16, bpreshuffle=True)
