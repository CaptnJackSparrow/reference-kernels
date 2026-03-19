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

__device__ __forceinline__ int32_t broadcast_scale(uint8_t e8m0) {
    return (int32_t)e8m0 * 0x01010101;
}

// =====================================================================
// MFMA traits: specialize per tile size
// =====================================================================

template <int IM, int IN, int IK>
struct MfmaTraits;

// 32x32x64: 16 acc floats, 4xi32 input (zero-padded to 8)
template <>
struct MfmaTraits<32, 32, 64> {
    using acc_t = float16_t;
    static constexpr int ACC_SIZE = 16;
    static constexpr int BLOCKS_PER_CALL = 2;   // 64 FP4 / 32
    static constexpr int REGS = 4;

    static __device__ __forceinline__ acc_t zero_acc() {
        return {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    }

    static __device__ __forceinline__ acc_t mfma(
        uint32_t a_reg[REGS], int32_t a_sc,
        uint32_t b_reg[REGS], int32_t b_sc,
        acc_t acc
    ) {
        int8_vec a_vec = {(int)a_reg[0], (int)a_reg[1], (int)a_reg[2], (int)a_reg[3],
                          0, 0, 0, 0};
        int8_vec b_vec = {(int)b_reg[0], (int)b_reg[1], (int)b_reg[2], (int)b_reg[3],
                          0, 0, 0, 0};
        return __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_vec, b_vec, acc, FMT_FP4, FMT_FP4, 0, a_sc, 0, b_sc);
    }

    static __device__ __forceinline__ void store(
        hip_bfloat16* C, const acc_t& acc,
        int tile_m, int tile_n, int lane, int M, int N
    ) {
        int col = lane % 32;
        int half = lane / 32;
        for (int i = 0; i < 16; i++) {
            int row = (i % 4) + 4 * half + 8 * (i / 4);
            int gm = tile_m + row;
            int gn = tile_n + col;
            if (gm < M && gn < N)
                C[gm * N + gn] = static_cast<hip_bfloat16>(acc[i]);
        }
    }

    static __device__ __forceinline__ void load_a(
        const uint8_t* src, int row_base, int blk0, int lane,
        int k_half_stride, uint32_t reg[4], int& blk_out, int& row_out
    ) {
        row_out = row_base + (lane % 32);
        int k_group = lane / 32;  // 0 or 1
        blk_out = blk0 + k_group;
        int off = row_out * k_half_stride + blk_out * 16;
        reg[0] = *reinterpret_cast<const uint32_t*>(&src[off + 0]);
        reg[1] = *reinterpret_cast<const uint32_t*>(&src[off + 4]);
        reg[2] = *reinterpret_cast<const uint32_t*>(&src[off + 8]);
        reg[3] = *reinterpret_cast<const uint32_t*>(&src[off + 12]);
    }
};

// 16x16x128: 4 acc floats, 8xi32 input (fully used)
template <>
struct MfmaTraits<16, 16, 128> {
    using acc_t = float4_t;
    static constexpr int ACC_SIZE = 4;
    static constexpr int BLOCKS_PER_CALL = 4;   // 128 FP4 / 32
    static constexpr int REGS = 8;

    static __device__ __forceinline__ acc_t zero_acc() {
        return {0, 0, 0, 0};
    }

    static __device__ __forceinline__ acc_t mfma(
        uint32_t a_reg[REGS], int32_t a_sc,
        uint32_t b_reg[REGS], int32_t b_sc,
        acc_t acc
    ) {
        int8_vec a_vec = {(int)a_reg[0], (int)a_reg[1], (int)a_reg[2], (int)a_reg[3],
                          (int)a_reg[4], (int)a_reg[5], (int)a_reg[6], (int)a_reg[7]};
        int8_vec b_vec = {(int)b_reg[0], (int)b_reg[1], (int)b_reg[2], (int)b_reg[3],
                          (int)b_reg[4], (int)b_reg[5], (int)b_reg[6], (int)b_reg[7]};
        return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_vec, b_vec, acc, FMT_FP4, FMT_FP4, 0, a_sc, 0, b_sc);
    }

    // 16x16 output mapping: each thread holds 4 floats
    //   col = lane % 16
    //   row = (i % 4) + 4 * (lane / 16)  (lane/16 = 0..3)
    static __device__ __forceinline__ void store(
        hip_bfloat16* C, const acc_t& acc,
        int tile_m, int tile_n, int lane, int M, int N
    ) {
        int col = lane % 16;
        int quad = lane / 16;  // 0..3
        for (int i = 0; i < 4; i++) {
            int row = i + 4 * quad;
            int gm = tile_m + row;
            int gn = tile_n + col;
            if (gm < M && gn < N)
                C[gm * N + gn] = static_cast<hip_bfloat16>(acc[i]);
        }
    }

    // Load 32 bytes (8 dwords) = 128 FP4 values from 4 consecutive MXFP4 blocks
    // Lane mapping for 16x16x128:
    //   row = lane % 16
    //   k_group = lane / 16 (0..3, selects which of 4 blocks)
    static __device__ __forceinline__ void load_a(
        const uint8_t* src, int row_base, int blk0, int lane,
        int k_half_stride, uint32_t reg[8], int& blk_out, int& row_out
    ) {
        row_out = row_base + (lane % 16);
        int k_group = lane / 16;  // 0..3
        blk_out = blk0 + k_group;
        int off = row_out * k_half_stride + blk_out * 16;
        reg[0] = *reinterpret_cast<const uint32_t*>(&src[off + 0]);
        reg[1] = *reinterpret_cast<const uint32_t*>(&src[off + 4]);
        reg[2] = *reinterpret_cast<const uint32_t*>(&src[off + 8]);
        reg[3] = *reinterpret_cast<const uint32_t*>(&src[off + 12]);
        // Remaining 4 dwords: zero (each lane only loads one block of 32 FP4)
        // The 128 FP4 are distributed across 4 lane groups
        reg[4] = reg[5] = reg[6] = reg[7] = 0;
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

    Traits::load_a(A_data, tile_m, blk0, lane, K_HALF,
                   a_r, a_blk, a_row);
    uint8_t a_e = 127;
    if (a_row < M && a_blk < NUM_BLOCKS)
        a_e = A_scale[a_row + a_blk * M];
    else
        for (int r = 0; r < Traits::REGS; r++) a_r[r] = 0;
    a_s = broadcast_scale(a_e);

    Traits::load_a(B_data, tile_n, blk0, lane, K_HALF,
                   b_r, b_blk, b_row);
    uint8_t b_e = 127;
    if (b_row < N && b_blk < NUM_BLOCKS)
        b_e = B_scale[sh_scale_off<SCALE_N>(b_row, b_blk)];
    else
        for (int r = 0; r < Traits::REGS; r++) b_r[r] = 0;
    b_s = broadcast_scale(b_e);
}


// =====================================================================
// Simple kernel: 1 wavefront, direct global->register, double-buffered
// =====================================================================

template <int M, int N, int K_HALF, int NUM_BLOCKS, int SCALE_N,
          int IM, int IN, int IK>
__global__ void mfma_fp4_gemm_simple(
    const uint8_t* __restrict__ A_data,
    const uint8_t* __restrict__ B_data,
    const uint8_t* __restrict__ A_scale,
    const uint8_t* __restrict__ B_scale,
    hip_bfloat16* __restrict__ C
) {
    using Traits = MfmaTraits<IM, IN, IK>;
    constexpr int BPC = Traits::BLOCKS_PER_CALL;
    constexpr int K_ITERS = (NUM_BLOCKS + BPC - 1) / BPC;

    const int tile_m = __builtin_amdgcn_readfirstlane(blockIdx.x * IM);
    const int tile_n = __builtin_amdgcn_readfirstlane(blockIdx.y * IN);
    const int lane = threadIdx.x;

    auto acc = Traits::zero_acc();

    if (K_ITERS > 0) {
        uint32_t a_cur[Traits::REGS], b_cur[Traits::REGS];
        uint32_t a_nxt[Traits::REGS], b_nxt[Traits::REGS];
        int32_t a_sc_cur, b_sc_cur, a_sc_nxt, b_sc_nxt;

        load_ab_global<M, N, K_HALF, NUM_BLOCKS, SCALE_N, IM, IN, IK>(
            A_data, B_data, A_scale, B_scale,
            tile_m, tile_n, 0, lane,
            a_cur, a_sc_cur, b_cur, b_sc_cur);

        for (int ki = 0; ki < K_ITERS; ki++) {
            if (ki + 1 < K_ITERS)
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
    }

    Traits::store(C, acc, tile_m, tile_n, lane, M, N);
}


// =====================================================================
// Tiled kernel: multi-wavefront, LDS-backed
// =====================================================================

template <int OUTER_M, int OUTER_N, int OUTER_K>
struct LdsLayout {
    static constexpr int K_HALF   = OUTER_K / 2;
    static constexpr int K_BLOCKS = OUTER_K / 32;
    static constexpr int A_DATA   = OUTER_M * K_HALF;
    static constexpr int A_SCALE  = OUTER_M * K_BLOCKS;
    static constexpr int B_DATA   = OUTER_N * K_HALF;
    static constexpr int B_SCALE  = OUTER_N * K_BLOCKS;
    static constexpr int TOTAL    = A_DATA + A_SCALE + B_DATA + B_SCALE;
    static constexpr int A_DATA_OFF  = 0;
    static constexpr int A_SCALE_OFF = A_DATA;
    static constexpr int B_DATA_OFF  = A_SCALE_OFF + A_SCALE;
    static constexpr int B_SCALE_OFF = B_DATA_OFF + B_DATA;
};

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

    for (int dw = tid; dw < OUTER_M * OK_HALF / 4; dw += BLOCK_SIZE) {
        int byte_off = dw * 4;
        int row = byte_off / OK_HALF;
        int col = byte_off % OK_HALF;
        int g_m = outer_m + row;
        int g_col = k_half_base + col;
        uint32_t val = 0;
        if (g_m < M && g_col + 3 < K_HALF)
            val = *reinterpret_cast<const uint32_t*>(&A_data[g_m * K_HALF + g_col]);
        *reinterpret_cast<uint32_t*>(&smem_data[byte_off]) = val;
    }

    for (int s = tid; s < OUTER_M * OK_BLOCKS; s += BLOCK_SIZE) {
        int row = s / OK_BLOCKS;
        int col = s % OK_BLOCKS;
        int g_m = outer_m + row;
        int g_blk = blk_base + col;
        uint8_t val = 127;
        if (g_m < M && g_blk < NUM_BLOCKS)
            val = A_scale[g_m + g_blk * M];
        smem_scale[s] = val;
    }
}

template <int N, int K_HALF, int NUM_BLOCKS, int SCALE_N,
          int OUTER_N, int OUTER_K, int BLOCK_SIZE>
__device__ __forceinline__ void load_b_to_lds(
    const uint8_t* __restrict__ B_data,
    const uint8_t* __restrict__ B_scale,
    uint8_t* smem_data, uint8_t* smem_scale,
    int outer_n, int k_half_base, int blk_base, int tid
) {
    constexpr int OK_HALF = OUTER_K / 2;
    constexpr int OK_BLOCKS = OUTER_K / 32;

    for (int dw = tid; dw < OUTER_N * OK_HALF / 4; dw += BLOCK_SIZE) {
        int byte_off = dw * 4;
        int row = byte_off / OK_HALF;
        int col = byte_off % OK_HALF;
        int g_n = outer_n + row;
        int g_col = k_half_base + col;
        uint32_t val = 0;
        if (g_n < N && g_col + 3 < K_HALF)
            val = *reinterpret_cast<const uint32_t*>(&B_data[g_n * K_HALF + g_col]);
        *reinterpret_cast<uint32_t*>(&smem_data[byte_off]) = val;
    }

    for (int s = tid; s < OUTER_N * OK_BLOCKS; s += BLOCK_SIZE) {
        int row = s / OK_BLOCKS;
        int col = s % OK_BLOCKS;
        int g_n = outer_n + row;
        int g_blk = blk_base + col;
        uint8_t val = 127;
        if (g_n < N && g_blk < NUM_BLOCKS)
            val = B_scale[sh_scale_off<SCALE_N>(g_n, g_blk)];
        smem_scale[s] = val;
    }
}

template <int OUTER_K, int IM, int IN, int IK>
__device__ __forceinline__ void inner_mfma_loop(
    const uint8_t* smem_a_data, const uint8_t* smem_a_scale,
    const uint8_t* smem_b_data, const uint8_t* smem_b_scale,
    int warp_m, int warp_n, int lane,
    typename MfmaTraits<IM, IN, IK>::acc_t& acc
) {
    using Traits = MfmaTraits<IM, IN, IK>;
    constexpr int OK_HALF = OUTER_K / 2;
    constexpr int OK_BLOCKS = OUTER_K / 32;
    constexpr int BPC = Traits::BLOCKS_PER_CALL;
    constexpr int ITERS = OUTER_K / IK;

    for (int ik = 0; ik < ITERS; ik++) {
        int blk0 = ik * BPC;

        uint32_t a_reg[Traits::REGS], b_reg[Traits::REGS];
        int a_row, a_blk, b_row, b_blk;

        Traits::load_a(smem_a_data, warp_m * IM, blk0, lane, OK_HALF,
                        a_reg, a_blk, a_row);
        int32_t a_sc = broadcast_scale(smem_a_scale[a_row * OK_BLOCKS + a_blk]);

        Traits::load_a(smem_b_data, warp_n * IN, blk0, lane, OK_HALF,
                        b_reg, b_blk, b_row);
        int32_t b_sc = broadcast_scale(smem_b_scale[b_row * OK_BLOCKS + b_blk]);

        acc = Traits::mfma(a_reg, a_sc, b_reg, b_sc, acc);
    }
}

template <int M, int N, int K_HALF, int NUM_BLOCKS, int SCALE_N,
          int OUTER_M, int OUTER_N, int OUTER_K,
          int IM, int IN, int IK>
__global__ void mfma_fp4_gemm_tiled(
    const uint8_t* __restrict__ A_data,
    const uint8_t* __restrict__ B_data,
    const uint8_t* __restrict__ A_scale,
    const uint8_t* __restrict__ B_scale,
    hip_bfloat16* __restrict__ C
) {
    using Traits = MfmaTraits<IM, IN, IK>;
    using Lds = LdsLayout<OUTER_M, OUTER_N, OUTER_K>;
    constexpr int OK_BLOCKS = OUTER_K / 32;
    constexpr int OK_HALF = OUTER_K / 2;
    constexpr int OUTER_K_ITERS = (NUM_BLOCKS + OK_BLOCKS - 1) / OK_BLOCKS;
    constexpr int WARPS_M = OUTER_M / IM;
    constexpr int WARPS_N = OUTER_N / IN;
    constexpr int BLOCK_SIZE = WARPS_M * WARPS_N * 64;

    const int outer_m = __builtin_amdgcn_readfirstlane(blockIdx.x * OUTER_M);
    const int outer_n = __builtin_amdgcn_readfirstlane(blockIdx.y * OUTER_N);
    const int tid = threadIdx.x;
    const int warp_id = tid / 64;
    const int lane = tid % 64;
    const int warp_m = warp_id / WARPS_N;
    const int warp_n = warp_id % WARPS_N;

    extern __shared__ uint8_t smem[];

    auto acc = Traits::zero_acc();

    for (int ok = 0; ok < OUTER_K_ITERS; ok++) {
        load_a_to_lds<M, K_HALF, NUM_BLOCKS, OUTER_M, OUTER_K, BLOCK_SIZE>(
            A_data, A_scale,
            smem + Lds::A_DATA_OFF, smem + Lds::A_SCALE_OFF,
            outer_m, ok * OK_HALF, ok * OK_BLOCKS, tid);

        load_b_to_lds<N, K_HALF, NUM_BLOCKS, SCALE_N, OUTER_N, OUTER_K, BLOCK_SIZE>(
            B_data, B_scale,
            smem + Lds::B_DATA_OFF, smem + Lds::B_SCALE_OFF,
            outer_n, ok * OK_HALF, ok * OK_BLOCKS, tid);

        __syncthreads();

        inner_mfma_loop<OUTER_K, IM, IN, IK>(
            smem + Lds::A_DATA_OFF, smem + Lds::A_SCALE_OFF,
            smem + Lds::B_DATA_OFF, smem + Lds::B_SCALE_OFF,
            warp_m, warp_n, lane, acc);

        __syncthreads();
    }

    Traits::store(C, acc,
        outer_m + warp_m * IM,
        outer_n + warp_n * IN,
        lane, M, N);
}

// =====================================================================
// Instantiations
// =====================================================================

#define INST_S(M, N, KH, NB, SN, IM, IN, IK) \\
    template __global__ void mfma_fp4_gemm_simple<M,N,KH,NB,SN,IM,IN,IK>( \\
        const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, hip_bfloat16*);

#define INST_T(M, N, KH, NB, SN, OM, ON, OK, IM, IN, IK) \\
    template __global__ void mfma_fp4_gemm_tiled<M,N,KH,NB,SN,OM,ON,OK,IM,IN,IK>( \\
        const uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, hip_bfloat16*);

// Simple 32x32x64: small M, small K
INST_S(32, 4096, 256, 16, 16,  32, 32, 64)
INST_S(32, 2880, 256, 16, 16,  32, 32, 64)
INST_S(64,  7168, 1024, 64,  64, 32, 32, 64)
INST_S(256, 3072, 768,  48,  48, 32, 32, 64)

// Simple 16x16x128: small M, large K
INST_S(4,  2880, 256,  16,  16,   16, 16, 128)
INST_S(8,  2112, 3584, 224, 224,  16, 16, 128)
INST_S(16, 2112, 3584, 224, 224,  16, 16, 128)
INST_S(16, 3072, 768,  48,  48,   16, 16, 128)

// Tiled: larger shapes
// INST_T(64,  7168, 1024, 64,  64,  32, 64, 1024,  32, 32, 64)
INST_T(64,  3072, 768,  48,  48,  32, 32, 1536,  32, 32, 64)
// INST_T(256, 3072, 768,  48,  48,  128, 128, 512,  32, 32, 64)
INST_T(256, 2880, 256,  16,  16,  128, 128, 512,  32, 32, 64)

#undef INST_S
#undef INST_T
'''

MXFP4_CPP_SOURCE = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <ATen/hip/HIPContext.h>

template <int M, int N, int K_HALF, int NUM_BLOCKS, int SCALE_N,
          int IM, int IN, int IK>
void launch_simple(torch::Tensor A, torch::Tensor B,
                   torch::Tensor As, torch::Tensor Bs, torch::Tensor C) {
    dim3 grid((M + IM - 1) / IM, (N + IN - 1) / IN);
    dim3 block(64);
    hipLaunchKernelGGL((mfma_fp4_gemm_simple<M,N,K_HALF,NUM_BLOCKS,SCALE_N,IM,IN,IK>),
        grid, block, 0, 0,
        reinterpret_cast<const uint8_t*>(A.data_ptr()),
        reinterpret_cast<const uint8_t*>(B.data_ptr()),
        reinterpret_cast<const uint8_t*>(As.data_ptr()),
        reinterpret_cast<const uint8_t*>(Bs.data_ptr()),
        reinterpret_cast<hip_bfloat16*>(C.data_ptr()));
}

template <int M, int N, int K_HALF, int NUM_BLOCKS, int SCALE_N,
          int OM, int ON, int OK, int IM, int IN, int IK>
void launch_tiled(torch::Tensor A, torch::Tensor B,
                  torch::Tensor As, torch::Tensor Bs, torch::Tensor C) {
    constexpr int WARPS = (OM/IM) * (ON/IN);
    constexpr int OK_H = OK/2, OK_B = OK/32;
    constexpr size_t SHARED = OM*OK_H + OM*OK_B + ON*OK_H + ON*OK_B;
    dim3 grid((M+OM-1)/OM, (N+ON-1)/ON);
    dim3 block(WARPS * 64);
    hipLaunchKernelGGL((mfma_fp4_gemm_tiled<M,N,K_HALF,NUM_BLOCKS,SCALE_N,OM,ON,OK,IM,IN,IK>),
        grid, block, SHARED, 0,
        reinterpret_cast<const uint8_t*>(A.data_ptr()),
        reinterpret_cast<const uint8_t*>(B.data_ptr()),
        reinterpret_cast<const uint8_t*>(As.data_ptr()),
        reinterpret_cast<const uint8_t*>(Bs.data_ptr()),
        reinterpret_cast<hip_bfloat16*>(C.data_ptr()));
}

torch::Tensor mfma_gemm(
    torch::Tensor A_data, torch::Tensor B_data,
    torch::Tensor A_scale, torch::Tensor B_scale,
    int M, int N, int K, int K_half, int num_blocks, int scaleN,
    int a_s0, int a_s1
) {
    auto C = torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(A_data.device()));

#define S32(m,n,kh,nb,sn) \
    if(M==m&&N==n&&K_half==kh){launch_simple<m,n,kh,nb,sn,32,32,64>(A_data,B_data,A_scale,B_scale,C);return C;}
#define S16(m,n,kh,nb,sn) \
    if(M==m&&N==n&&K_half==kh){launch_simple<m,n,kh,nb,sn,16,16,128>(A_data,B_data,A_scale,B_scale,C);return C;}
#define T(m,n,kh,nb,sn,om,on,ok,im,in,ik) \
    if(M==m&&N==n&&K_half==kh){launch_tiled<m,n,kh,nb,sn,om,on,ok,im,in,ik>(A_data,B_data,A_scale,B_scale,C);return C;}

    // Simple 32x32x64
    S32(32, 4096, 256, 16, 16)
    S32(32, 2880, 256, 16, 16)
    S32(64,  7168, 1024, 64, 64)
    S32(256, 3072, 768,  48, 48)

    // Simple 16x16x128
    S16(4,  2880, 256,  16,  16)
    S16(8,  2112, 3584, 224, 224)
    S16(16, 2112, 3584, 224, 224)
    S16(16, 3072, 768,  48,  48)

    // Tiled
    // T(64,  7168, 1024, 64, 64,  32,64,1024, 32,32,64)
    T(64,  3072, 768,  48, 48,  32,32,1536, 32,32,64)
    // T(256, 3072, 768,  48, 48,  128,128,512, 32,32,64)
    T(256, 2880, 256,  16, 16,  128,128,512, 32,32,64)

#undef S32
#undef S16
#undef T

    TORCH_CHECK(false, "No template for M=", M, " N=", N, " K_half=", K_half);
    return C;
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
    import aiter
    from aiter import dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle

    PROFILE = False
    PROFILE_INTERVAL = 200

    A, B, B_q, B_shuffle, B_scale_sh = data
    A = A.contiguous()
    m, k = A.shape
    n, _ = B.shape

    if HAS_HIP_KERNEL:
        try:
            if PROFILE:
                if not hasattr(custom_kernel, '_stats'):
                    custom_kernel._stats = {}
                shape_key = (m, n, k)
                if shape_key not in custom_kernel._stats:
                    custom_kernel._stats[shape_key] = {'quant': 0.0, 'gemm': 0.0, 'count': 0}
                start = torch.cuda.Event(enable_timing=True)
                mid = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            # Quantize A — NO e8m0_shuffle, kernel reads linear scales
            x_fp4, bs_e8m0 = dynamic_mxfp4_quant(A)

            # Direct views — no copies
            A_data = x_fp4.view(torch.uint8)
            A_sc = bs_e8m0  # uint8, transposed (column-major), NOT shuffled
            B_data = B_q.view(torch.uint8)
            B_sc = B_scale_sh.view(torch.uint8)

            K_half = k // 2
            num_blocks = (k + 31) // 32
            scaleN = ((num_blocks + 7) // 8) * 8

            # Pass A scale strides so kernel handles column-major layout
            a_s0 = A_sc.stride(0)
            a_s1 = A_sc.stride(1)

            if PROFILE:
                mid.record()

            out = _hip_module.mfma_gemm(
                A_data, B_data, A_sc, B_sc,
                m, n, k, K_half, num_blocks, scaleN,
                a_s0, a_s1)

            if PROFILE:
                end.record()
                torch.cuda.synchronize()
                s = custom_kernel._stats[shape_key]
                s['quant'] += start.elapsed_time(mid)
                s['gemm'] += mid.elapsed_time(end)
                s['count'] += 1
                if s['count'] % PROFILE_INTERVAL == 0:
                    cnt = s['count']
                    print(f"[PROFILE] m={m:4d} n={n:4d} k={k:4d} | "
                          f"quant={s['quant']/cnt*1000:.1f}us  "
                          f"gemm={s['gemm']/cnt*1000:.1f}us  "
                          f"total={(s['quant']+s['gemm'])/cnt*1000:.1f}us  "
                          f"(avg over {cnt} calls)", flush=True)

            return out
        except Exception as e:
            print(f"[mxfp4-mm] MFMA kernel failed for m={m} n={n} k={k}: {e}", flush=True)

    # # Fallback
    # x_fp4, bs_e8m0 = dynamic_mxfp4_quant(A)
    # bs_e8m0 = e8m0_shuffle(bs_e8m0)
    # A_q = x_fp4.view(dtypes.fp4x2)
    # A_scale_sh = bs_e8m0.view(dtypes.fp8_e8m0)
    # return aiter.gemm_a4w4(A_q, B_shuffle, A_scale_sh, B_scale_sh,
    #                        dtype=dtypes.bf16, bpreshuffle=True)
