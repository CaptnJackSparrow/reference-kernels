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
#include <hip/hip_cooperative_groups.h>
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

struct E8M0Scale {
    uint8_t e8m0;
    float quant_scale;
};

// E8M0 lookup: raw_exp -> e8m0 = clamp(raw_exp - 2, 0, 254)
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

// QUANT_SCALE_RECIP_LUT: e8m0 -> IEEE float bits of 1/quant_scale = 2^(e8m0 - 254)
// recip_bits[i] = i << 23 (IEEE float 2^(i-127)), [0]=0 (zero scale)
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
        // BF16 has same exponent range as FP32, just extract exponent from BF16 bits
        // BF16: [sign(1)][exp(8)][mant(7)] - rounding bias in BF16 is 0x0020
        uint16_t rounded = (amax_bits + 0x0020u) & 0xFF80u;
        int raw_exp = (int)((rounded >> 7) & 0xFF);
        result.e8m0 = E8M0_LUT[raw_exp];
        result.quant_scale = __uint_as_float(QUANT_SCALE_RECIP_LUT[result.e8m0]);
    }
    return result;
}

// Hardware FP4 conversion from BF16: converts 2 BF16 values to packed FP4 byte.
// Passes reciprocal scale directly to the BF16 intrinsic (no pre-multiply).
__device__ __forceinline__ uint8_t quantize_fp4_pair_hw_bf16(
    hip_bfloat16 v0, hip_bfloat16 v1, float quant_scale
) {
    using bf16x2 = uint16_t __attribute__((ext_vector_type(2)));
    bf16x2 pair = {v0.data, v1.data};
    union { uint32_t u32; uint8_t u8[4]; } cvt = {0};
    cvt.u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(cvt.u32, pair, quant_scale, 0);
    return cvt.u8[0];
}

// Hardware FP4 conversion from FP32: converts 2 pre-scaled floats to packed FP4 byte.
// Values must be pre-multiplied by quant_scale before calling.
// Matches CK usage: scale=1.0f (no additional scaling by intrinsic).
/*__device__ __forceinline__ uint8_t quantize_fp4_pair_hw(float v0, float v1, float quant_scale) {
    union { uint32_t u32; uint8_t u8[4]; } cvt = {0};
    cvt.u32 = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(
        cvt.u32, v0, v1, quant_scale, 0);
    return cvt.u8[0];
}*/

// Quantize 32 BF16 values to packed FP4 + E8M0 scale using BF16 hw intrinsic.
// amax computed in BF16 - no FP32 intermediate.
__device__ __forceinline__ QuantBlock quantize_fp4_block_bf16(const hip_bfloat16* src) {
    // Find amax across 32 BF16 values in BF16 precision
    uint16_t amax_bits = 0;
    for (int i = 0; i < 32; i++) {
        uint16_t bits = *reinterpret_cast<const uint16_t*>(&src[i]) & 0x7FFF;  // abs via clear sign
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

template <int M, int K, int OUTER_M, int OUTER_K, int BLOCK_SIZE>
__device__ __forceinline__ void quantize_a_to_reg(
    const hip_bfloat16 A[][K],
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

        QuantBlock qb = quantize_fp4_block_bf16(&A[g_m][g_k]);
        *reinterpret_cast<uint128_vec*>(&data_regs[bi * 4]) = *reinterpret_cast<uint128_vec*>(&qb.data);
        scale_regs[bi] = qb.e8m0;
    }
}

template <int OUTER_M, int OUTER_K, int BLOCK_SIZE>
__device__ __forceinline__ void store_quant_a_to_lds(
    uint8_t smem_data[][OUTER_K / 2], uint8_t smem_scale[][OUTER_K / 32],
    int tid,
    const uint32_t* data_regs, const uint8_t* scale_regs
) {
    constexpr int OK_BLOCKS = OUTER_K / 32;
    constexpr int TOTAL_BLOCKS = OUTER_M * OK_BLOCKS;

    for (int b = tid, bi = 0; b < TOTAL_BLOCKS; b += BLOCK_SIZE, bi++) {
        int row = b / OK_BLOCKS;
        int blk = b % OK_BLOCKS;

        *reinterpret_cast<uint128_vec*>(&smem_data[row][blk * 16]) = *reinterpret_cast<const uint128_vec*>(&data_regs[bi * 4]);
        smem_scale[row][blk] = scale_regs[bi];
    }
}

// Wrapper for drop-in replacement
template <int M, int K, int OUTER_M, int OUTER_K, int BLOCK_SIZE>
__device__ __forceinline__ void quantize_a_to_lds(
    const hip_bfloat16 A[][K],
    uint8_t smem_data[][OUTER_K / 2], uint8_t smem_scale[][OUTER_K / 32],
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
    const uint8_t src[][K_HALF_STRIDE], int row_base, int blk0, int lane,
    uint32_t reg[REGS], int& blk_out, int& row_out
) {
    row_out = row_base + (lane % IM);
    int k_group = lane / IM;
    blk_out = blk0 + k_group;
    *reinterpret_cast<uint128_vec*>(&reg[0]) = *reinterpret_cast<const uint128_vec*>(&src[row_out][blk_out * 16]);
}

template <int IM, int OUTER_N, int REGS = (IM == 32) ? 4 : 8>
__device__ __forceinline__ void load_transposed(
    const uint8_t src[][OUTER_N * 16], int row_base, int blk0, int lane,
    uint32_t reg[REGS], int& blk_out, int& row_out
) {
    row_out = row_base + (lane % IM);
    int k_group = lane / IM;
    blk_out = blk0 + k_group;
    int off = (blk_out * OUTER_N + row_out) * 16;
    *reinterpret_cast<uint128_vec*>(&reg[0]) = *reinterpret_cast<const uint128_vec*>(&src[blk_out][row_out * 16]);
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

    template <int M, int N>
    static __device__ __forceinline__ void store(
        hip_bfloat16 C[][N], const acc_t& acc,
        int tile_m, int tile_n, int lane
    ) {
        if constexpr (IM == 32) {
            int col = lane % 32;
            int half = lane / 32;
            for (int i = 0; i < 16; i++) {
                int row = (i % 4) + 4 * half + 8 * (i / 4);
                int gm = tile_m + row;
                int gn = tile_n + col;
                if (gm < M && gn < N)
                    C[gm][gn] = static_cast<hip_bfloat16>(acc[i]);
            }
        } else {
            int col = lane % 16;
            int quad = lane / 16;
            for (int i = 0; i < 4; i++) {
                int row = i + 4 * quad;
                int gm = tile_m + row;
                int gn = tile_n + col;
                if (gm < M && gn < N)
                    C[gm][gn] = static_cast<hip_bfloat16>(acc[i]);
            }
        }
    }

    template <int SMEM_N>
    static __device__ __forceinline__ void store_acc_to_smem(
        float smem_c[][SMEM_N], const acc_t& acc,
        int tile_m_local, int tile_n_local, int lane
    ) {
        if constexpr (IM == 32) {
            int col = tile_n_local + (lane % 32);
            int half = lane / 32;
            for (int i = 0; i < 16; i++) {
                int row = (i % 4) + 4 * half + 8 * (i / 4);
                smem_c[tile_m_local + row][col] = acc[i];
            }
        } else {
            int col = tile_n_local + (lane % 16);
            int quad = lane / 16;
            for (int i = 0; i < 4; i++) {
                int row = i + 4 * quad;
                smem_c[tile_m_local + row][col] = acc[i];
            }
        }
    }

    template <int SMEM_N>
    static __device__ __forceinline__ void load_acc_from_smem(
        const float smem_c[][SMEM_N], acc_t& acc,
        int tile_m_local, int tile_n_local, int lane
    ) {
        if constexpr (IM == 32) {
            int col = tile_n_local + (lane % 32);
            int half = lane / 32;
            for (int i = 0; i < 16; i++) {
                int row = (i % 4) + 4 * half + 8 * (i / 4);
                acc[i] = smem_c[tile_m_local + row][col];
            }
        } else {
            int col = tile_n_local + (lane % 16);
            int quad = lane / 16;
            for (int i = 0; i < 4; i++) {
                int row = i + 4 * quad;
                acc[i] = smem_c[tile_m_local + row][col];
            }
        }
    }

    template <int N>
    static __device__ __forceinline__ void store_f32(
        float* ws, const acc_t& acc,
        int tile_m, int tile_n, int lane
    ) {
        if constexpr (IM == 32) {
            int col = lane % 32;
            int half = lane / 32;
            for (int i = 0; i < 16; i++) {
                int row = (i % 4) + 4 * half + 8 * (i / 4);
                ws[(tile_m + row) * N + tile_n + col] = acc[i];
            }
        } else {
            int col = lane % 16;
            int quad = lane / 16;
            for (int i = 0; i < 4; i++) {
                int row = i + 4 * quad;
                ws[(tile_m + row) * N + tile_n + col] = acc[i];
            }
        }
    }
};

template <int K_HALF, int IM, int IN, int IK>
__device__ __forceinline__ int32_t load_a_with_scale(
    const uint8_t A_data[][K_HALF],
    const uint8_t A_scale[][K_HALF / 16],
    int tile_m, int blk0, int lane,
    uint32_t a_r[MfmaTraits<IM,IN,IK>::REGS]
) {
    int a_row, a_blk;
    load_tile<IM, K_HALF>(A_data, tile_m, blk0, lane,
                            a_r, a_blk, a_row);
    uint8_t a_e = A_scale[a_row][a_blk];
    return broadcast_scale(a_e);
}

template <int K_HALF, int NUM_BLOCKS, int IM, int IN, int IK>
__device__ __forceinline__ int32_t load_b_with_scale(
    const uint8_t B_data[][K_HALF],
    const uint8_t* __restrict__ B_scale,
    int tile_n, int blk0, int lane,
    uint32_t b_r[MfmaTraits<IM,IN,IK>::REGS]
) {
    int b_row, b_blk;
    load_tile<IN, K_HALF>(B_data, tile_n, blk0, lane,
                            b_r, b_blk, b_row);
    uint8_t b_e = B_scale[sh_scale_off<NUM_BLOCKS>(b_row, b_blk)];
    return broadcast_scale(b_e);
}

template <int M, int N, int K_HALF, int NUM_BLOCKS,
          int IM, int IN, int IK>
__device__ __forceinline__ void load_ab_global(
    const uint8_t A_data[][K_HALF],
    const uint8_t B_data[][K_HALF],
    const uint8_t A_scale[][K_HALF / 16],
    const uint8_t* __restrict__ B_scale,
    int tile_m, int tile_n, int blk0, int lane,
    uint32_t a_r[MfmaTraits<IM,IN,IK>::REGS], int32_t& a_s,
    uint32_t b_r[MfmaTraits<IM,IN,IK>::REGS], int32_t& b_s
) {
    a_s = load_a_with_scale<K_HALF, IM, IN, IK>(A_data, A_scale, tile_m, blk0, lane, a_r);
    b_s = load_b_with_scale<K_HALF, NUM_BLOCKS, IM, IN, IK>(B_data, B_scale, tile_n, blk0, lane, b_r);
}

// =====================================================================
// Result of warp-parallel quantization: packed FP4 bytes + E8M0 scale in registers.
template <int VALS_PER_THREAD>
struct QuantResult {
    uint8_t packed_bytes[VALS_PER_THREAD / 2 > 0 ? VALS_PER_THREAD / 2 : 1];
    uint8_t e8m0;
    int num_bytes;  // actual valid packed bytes this thread holds
};

// Warp-parallel quantization: returns quantized values in registers.
// Caller is responsible for writing to memory.
// =====================================================================
template <int M, int K, int VALS_PER_THREAD>
__device__ __forceinline__ QuantResult<VALS_PER_THREAD> quantize_block_parallel(
    const hip_bfloat16 A[][K],
    int row, int blk, int lane_in_group, int tid
) {
    static_assert(VALS_PER_THREAD >= 1 && VALS_PER_THREAD <= 32, "VALS_PER_THREAD must be 1..32");
    static_assert(32 % VALS_PER_THREAD == 0, "32 must be divisible by VALS_PER_THREAD");

    constexpr int THREADS_PER_GROUP = 32 / VALS_PER_THREAD;
    QuantResult<VALS_PER_THREAD> result;

    // Each thread loads VALS_PER_THREAD BF16 values (keep both BF16 and FP32 for max)
    hip_bfloat16 bvals[VALS_PER_THREAD];
    int base_k = blk * 32 + lane_in_group * VALS_PER_THREAD;
    for (int v = 0; v < VALS_PER_THREAD; v++) {
        bvals[v] = A[row][base_k + v];
    }

    // Find local max across this thread's values in BF16 (uint16 abs comparison)
    uint16_t local_max_bits = *reinterpret_cast<const uint16_t*>(&bvals[0]) & 0x7FFF;
    for (int v = 1; v < VALS_PER_THREAD; v++) {
        uint16_t bits = *reinterpret_cast<const uint16_t*>(&bvals[v]) & 0x7FFF;
        local_max_bits = (bits > local_max_bits) ? bits : local_max_bits;
    }
    hip_bfloat16 local_max_bf16 = *reinterpret_cast<const hip_bfloat16*>(&local_max_bits);

    // Reduce max across all threads in the group directly in BF16
    hip_bfloat16 amax_bf16;
    if constexpr (THREADS_PER_GROUP == 1) {
        amax_bf16 = local_max_bf16;
    } else {
        int group_in_warp = (tid / THREADS_PER_GROUP);
        unsigned long long group_mask = 0;
        for (int i = 0; i < THREADS_PER_GROUP; i++)
            group_mask |= (1ull << (group_in_warp * THREADS_PER_GROUP + i));
        amax_bf16 = __reduce_max_sync(group_mask, local_max_bf16);
    }

    E8M0Scale sc = compute_e8m0_scale(amax_bf16);
    result.e8m0 = sc.e8m0;

    // Quantize to FP4 using BF16 hw intrinsic
    if constexpr (VALS_PER_THREAD == 1) {
        if ((lane_in_group & 1) == 0) {
            result.packed_bytes[0] = quantize_fp4_pair_hw_bf16(bvals[0], bvals[0], sc.quant_scale);
            result.num_bytes = 1;
        } else {
            result.num_bytes = 0;
        }
    } else {
        result.num_bytes = VALS_PER_THREAD / 2;
        for (int v = 0; v < VALS_PER_THREAD; v += 2) {
            result.packed_bytes[v / 2] = quantize_fp4_pair_hw_bf16(bvals[v], bvals[v + 1], sc.quant_scale);
        }
    }

    return result;
}

// Warp-parallel quantization kernel: calls quantize_block_parallel
// and writes results to global memory.
// =====================================================================
template <int M, int K, int NUM_BLOCKS, int VALS_PER_THREAD = 1>
__global__ void quant_a_kernel_warp_parallel(
    const hip_bfloat16 A[][K],
    uint8_t out_data[][K / 2],
    uint8_t out_scale[][NUM_BLOCKS]
) {
    constexpr int THREADS_PER_GROUP = 32 / VALS_PER_THREAD;
    constexpr int total = M * NUM_BLOCKS;
    constexpr int K_HALF = K / 2;

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;
    int group_id = global_tid / THREADS_PER_GROUP;
    int lane_in_group = global_tid % THREADS_PER_GROUP;

    int row = group_id / NUM_BLOCKS;
    int blk = group_id % NUM_BLOCKS;

    auto qr = quantize_block_parallel<M, K, VALS_PER_THREAD>(
        A, row, blk, lane_in_group, tid);

    // Write packed FP4 bytes to global memory
    if constexpr (VALS_PER_THREAD == 1) {
        if ((lane_in_group & 1) == 0) {
            int byte_idx = lane_in_group / 2;
            out_data[row][blk * 16 + byte_idx] = qr.packed_bytes[0];
        }
    } else {
        for (int b = 0; b < VALS_PER_THREAD / 2; b++) {
            int elem_idx = lane_in_group * VALS_PER_THREAD + b * 2;
            int byte_idx = elem_idx / 2;
            out_data[row][blk * 16 + byte_idx] = qr.packed_bytes[b];
        }
    }

    if (lane_in_group == 0) {
        out_scale[row][blk] = qr.e8m0;
    }
}

// Simple kernel: 1 wavefront, direct global->register, double-buffered
// =====================================================================

template <int M, int N, int K, int NUM_BLOCKS,
          int IM, int IN, int IK,
          int WARPS_M = 1, int WARPS_N = 1, int K_HALF = K / 2>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 64) mfma_fp4_gemm_simple(
    const uint8_t A_data[][K_HALF],
    const uint8_t B_data[][K_HALF],
    const uint8_t A_scale[][NUM_BLOCKS],
    const uint8_t* __restrict__ B_scale,
    hip_bfloat16 C[][N]
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

        load_ab_global<M, N, K_HALF, NUM_BLOCKS, IM, IN, IK>(
            A_data, B_data, A_scale, B_scale,
            tile_m, tile_n, 0, lane,
            a_cur, a_sc_cur, b_cur, b_sc_cur);

        for (int ki = 0; ki < K_ITERS; ki++) {
            if (ki + 1 < K_ITERS) {
                load_ab_global<M, N, K_HALF, NUM_BLOCKS, IM, IN, IK>(
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

    Traits::template store<M, N>(C, acc, tile_m, tile_n, lane);
}

// =====================================================================
// Simple Fused kernel: quantize A on-the-fly, no intermediate buffers
// Each lane quantizes its own 32-element MX block from BF16 before MFMA.
// =====================================================================

template <int M, int N, int K, int NUM_BLOCKS,
          int IM, int IN, int IK,
          int WARPS_M = 1, int WARPS_N = 1,
          int BLOCK_SIZE = WARPS_M * WARPS_N * 64, int K_HALF = K / 2>
__global__ void __launch_bounds__(BLOCK_SIZE) mfma_fp4_gemm_simple_fused(
    const hip_bfloat16 A_bf16[][K],
    const uint8_t B_data[][K_HALF],
    const uint8_t* __restrict__ B_scale,
    hip_bfloat16 C[][N]
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

        if constexpr (Traits::REGS == 8) {
            a_cur[4] = 0; a_cur[5] = 0; a_cur[6] = 0; a_cur[7] = 0;
            a_nxt[4] = 0; a_nxt[5] = 0; a_nxt[6] = 0; a_nxt[7] = 0;
        }

        // Quantize A with load_tile-compatible lane mapping: row = lane % IM, k_group = lane / IM
        {
            int a_row = tile_m + (lane % IM);
            int a_blk = 0 + (lane / IM);
            QuantBlock qb = quantize_fp4_block_bf16(&A_bf16[a_row][a_blk * 32]);
            *reinterpret_cast<uint128_vec*>(&a_cur[0]) = *reinterpret_cast<uint128_vec*>(&qb.data);
            a_sc_cur = broadcast_scale(qb.e8m0);
        }
        {
            b_sc_cur = load_b_with_scale<K_HALF, NUM_BLOCKS, IM, IN, IK>(B_data, B_scale, tile_n, 0, lane, b_cur);
        }

        for (int ki = 0; ki < K_ITERS - 1; ki++) {
            int a_row = tile_m + (lane % IM);
            int a_blk = (ki + 1) * BPC + (lane / IM);
            QuantBlock qb = quantize_fp4_block_bf16(&A_bf16[a_row][a_blk * 32]);
            *reinterpret_cast<uint128_vec*>(&a_nxt[0]) = *reinterpret_cast<uint128_vec*>(&qb.data);
            a_sc_nxt = broadcast_scale(qb.e8m0);
            b_sc_nxt = load_b_with_scale<K_HALF, NUM_BLOCKS, IM, IN, IK>(B_data, B_scale, tile_n, (ki + 1) * BPC, lane, b_nxt);

            acc = Traits::mfma(a_cur, a_sc_cur, b_cur, b_sc_cur, acc);

            for (int r = 0; r < Traits::REGS; r++) {
                a_cur[r] = a_nxt[r];
                b_cur[r] = b_nxt[r];
            }
            a_sc_cur = a_sc_nxt;
            b_sc_cur = b_sc_nxt;
        }

        acc = Traits::mfma(a_cur, a_sc_cur, b_cur, b_sc_cur, acc);
    }

    Traits::template store<M, N>(C, acc, tile_m, tile_n, lane);
}

// =====================================================================
// Cooperative Simple kernel: cooperative quant + grid.sync() + simple GEMM
// Single launch via hipLaunchCooperativeKernel.
// Phase 1: All blocks cooperatively quantize A (work-stealing)
// Phase 2: grid.sync() barrier (no threadfence needed)
// Phase 3: Standard simple GEMM on pre-quantized A
// =====================================================================

template <int M, int N, int K, int NUM_BLOCKS,
          int IM, int IN, int IK,
          int WARPS_M = 1, int WARPS_N = 1, int K_HALF = K / 2>
__global__ void mfma_fp4_gemm_coop_simple(
    const hip_bfloat16 (*__restrict__ A_bf16)[K],
    uint8_t A_data[][K_HALF],
    uint8_t A_scale[][NUM_BLOCKS],
    const uint8_t B_data[][K_HALF],
    const uint8_t* __restrict__ B_scale,
    hip_bfloat16 C[][N],
    int* __restrict__ quant_counter
) {
    using Traits = MfmaTraits<IM, IN, IK>;
    constexpr int BPC = Traits::BLOCKS_PER_CALL;
    constexpr int K_ITERS = (NUM_BLOCKS + BPC - 1) / BPC;
    constexpr int TOTAL_QUANT = M * NUM_BLOCKS;
    constexpr int BLOCK_SIZE = WARPS_M * WARPS_N * 64;

    const int tid = threadIdx.x;

    // Phase 1: Cooperative A quantization (work-stealing)
    int batch_start;
    while (true) {
        if (tid == 0) {
            batch_start = atomicAdd(quant_counter, BLOCK_SIZE);
        }
        batch_start = __builtin_amdgcn_readfirstlane(batch_start);

        if (batch_start >= TOTAL_QUANT) break;

        int my_chunk = batch_start + tid;
        if (my_chunk < TOTAL_QUANT) {
            int row = my_chunk / NUM_BLOCKS;
            int blk = my_chunk % NUM_BLOCKS;

            QuantBlock qb = quantize_fp4_block_bf16(&A_bf16[row * K + blk * 32]);

            *reinterpret_cast<uint128_vec*>(&A_data[row][blk * 16]) =
                *reinterpret_cast<uint128_vec*>(&qb.data);
            A_scale[row][blk] = qb.e8m0;
        }
    }

    // Phase 2: Grid-wide cooperative sync (includes memory barrier)
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    grid.sync();

    // Phase 3: Standard simple GEMM on pre-quantized A
    const int warp_id = __builtin_amdgcn_readfirstlane(tid / 64);
    const int lane = tid % 64;
    const int warp_m = __builtin_amdgcn_readfirstlane(warp_id / WARPS_N);
    const int warp_n = __builtin_amdgcn_readfirstlane(warp_id % WARPS_N);
    const int tile_m = __builtin_amdgcn_readfirstlane(blockIdx.x * (IM * WARPS_M)) + warp_m * IM;
    const int tile_n = __builtin_amdgcn_readfirstlane(blockIdx.y * (IN * WARPS_N)) + warp_n * IN;

    auto acc = Traits::zero_acc();

    if constexpr (K_ITERS > 0) {
        uint32_t a_cur[Traits::REGS], b_cur[Traits::REGS];
        uint32_t a_nxt[Traits::REGS], b_nxt[Traits::REGS];
        int32_t a_sc_cur, b_sc_cur, a_sc_nxt, b_sc_nxt;

        load_ab_global<M, N, K_HALF, NUM_BLOCKS, IM, IN, IK>(
            A_data, B_data, A_scale, B_scale,
            tile_m, tile_n, 0, lane,
            a_cur, a_sc_cur, b_cur, b_sc_cur);

        for (int ki = 0; ki < K_ITERS; ki++) {
            if (ki + 1 < K_ITERS) {
                load_ab_global<M, N, K_HALF, NUM_BLOCKS, IM, IN, IK>(
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

    Traits::template store<M, N>(C, acc, tile_m, tile_n, lane);
}


// =====================================================================
// Tiled kernel helpers: load global -> registers, store registers -> LDS
// =====================================================================

template <int OUTER_M, int OUTER_N, int OUTER_K, int WARP_TILES_M = 1, int WARP_TILES_N = 1>
struct LdsLayout {
    static constexpr int LIMIT    = 160 * 1024;
    static constexpr int OK_HALF   = OUTER_K / 2;
    static constexpr int K_BLOCKS = OUTER_K / 32;
    static constexpr int A_DATA   = OUTER_M * OK_HALF;
    static constexpr int A_SCALE  = OUTER_M * K_BLOCKS;
    static constexpr int B_DATA   = OUTER_N * OK_HALF;
    static constexpr int B_SCALE  = OUTER_N * K_BLOCKS;
    static constexpr int C_DATA_FLOATS = OUTER_M * OUTER_N;
    static constexpr int C_DATA_BYTES  = C_DATA_FLOATS * 4;
    static constexpr int TOTAL    = A_DATA + A_SCALE + B_DATA + B_SCALE +
                                    ((WARP_TILES_M * WARP_TILES_N > 1) ? C_DATA_BYTES : 0);
    static constexpr int OCCUPANCY= LIMIT / TOTAL;
};

// ---------- Part 1: Load A from global memory into registers ----------
template <int M, int K_HALF, int NUM_BLOCKS,
          int OUTER_M, int OUTER_K, int BLOCK_SIZE>
__device__ __forceinline__ void load_a_global_to_reg(
    const uint8_t A_data[][K_HALF],
    const uint8_t A_scale[][NUM_BLOCKS],
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
        val = *reinterpret_cast<const uint128_vec*>(&A_data[g_m][g_col]);
        *reinterpret_cast<uint128_vec*>(&data_regs[di * 4]) = val;
    }

    int si = 0;
    for (int s = tid; s < SCALE_ELEMS; s += BLOCK_SIZE, si++) {
        int row = s / OK_BLOCKS;
        int col = s % OK_BLOCKS;
        int g_m = outer_m + row;
        int g_blk = blk_base + col;
        uint8_t val = 127;
        val = A_scale[g_m][g_blk];
        scale_regs[si] = val;
    }
}

// ---------- Part 2: Store A from registers into LDS ----------
template <int OUTER_M, int OUTER_K, int BLOCK_SIZE>
__device__ __forceinline__ void store_a_reg_to_lds(
    uint8_t smem_data[][OUTER_K / 2], uint8_t* smem_scale,
    int tid,
    const uint32_t* data_regs, const uint8_t* scale_regs
) {
    constexpr int OK_BLOCKS = OUTER_K / 32;
    constexpr int DATA_CHUNKS = OUTER_M * OK_BLOCKS;
    constexpr int SCALE_ELEMS = OUTER_M * OK_BLOCKS;

    int di = 0;
    for (int c = tid; c < DATA_CHUNKS; c += BLOCK_SIZE, di++) {
        int row = c / OK_BLOCKS;
        int blk = c % OK_BLOCKS;
        *reinterpret_cast<uint128_vec*>(&smem_data[row][blk * 16]) =
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
    const uint8_t A_data[][K_HALF],
    const uint8_t A_scale[][NUM_BLOCKS],
    uint8_t smem_data[][OUTER_K / 2], uint8_t* smem_scale,
    int outer_m, int k_half_base, int blk_base, int tid
) {
    constexpr int OK_BLOCKS = OUTER_K / 32;
    constexpr int DATA_CHUNKS = OUTER_M * OK_BLOCKS;

    for (int c = tid; c < DATA_CHUNKS; c += BLOCK_SIZE) {
        int row = c / OK_BLOCKS;
        int blk = c % OK_BLOCKS;
        int g_m = outer_m + row;
        int g_col = k_half_base + blk * 16;
        *reinterpret_cast<uint128_vec*>(&smem_data[row][blk * 16]) =
            *reinterpret_cast<const uint128_vec*>(&A_data[g_m][g_col]);
    }

    for (int s = tid; s < OUTER_M * OK_BLOCKS; s += BLOCK_SIZE) {
        int row = s / OK_BLOCKS;
        int col = s % OK_BLOCKS;
        int g_m = outer_m + row;
        int g_blk = blk_base + col;
        smem_scale[s] = A_scale[g_m][g_blk];
    }
}

// ---------- Load B from global memory into registers ----------
template <int N, int K_HALF, int NUM_BLOCKS,
          int OUTER_N, int OUTER_K, int BLOCK_SIZE>
__device__ __forceinline__ void load_b_global_to_reg(
    const uint8_t B_data[][K_HALF],
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
            *reinterpret_cast<const uint128_vec*>(&B_data[g_n][g_col]);
    }

    int si = 0;
    for (int s = tid; s < SCALE_ELEMS; s += BLOCK_SIZE, si++) {
        int row = s / OK_BLOCKS;
        int col = s % OK_BLOCKS;
        int g_n = outer_n + row;
        int g_blk = blk_base + col;
        scale_regs[si] = B_scale[sh_scale_off<NUM_BLOCKS>(g_n, g_blk)];
    }
}

// ---------- Store B from registers into LDS ----------
template <int OUTER_N, int OUTER_K, int BLOCK_SIZE>
__device__ __forceinline__ void store_b_reg_to_lds(
    uint8_t smem_data[][OUTER_N * 16], uint8_t smem_scale[][OUTER_N],
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
        *reinterpret_cast<uint128_vec*>(&smem_data[blk][row * 16]) =
            *reinterpret_cast<const uint128_vec*>(&data_regs[di * 4]);
    }

    int si = 0;
    for (int s = tid; s < SCALE_ELEMS; s += BLOCK_SIZE, si++) {
        int row = s / OK_BLOCKS;
        int col = s % OK_BLOCKS;
        smem_scale[col][row] = scale_regs[si];
    }
}

// ---------- Wrapper for B ----------
template <int N, int K_HALF, int NUM_BLOCKS,
          int OUTER_N, int OUTER_K, int BLOCK_SIZE>
__device__ __forceinline__ void load_b_to_lds(
    const uint8_t B_data[][K_HALF],
    const uint8_t* __restrict__ B_scale,
    uint8_t smem_data[][OUTER_N * 16], uint8_t smem_scale[][OUTER_N],
    int outer_n, int k_half_base, int blk_base, int tid
) {
    constexpr int OK_BLOCKS = OUTER_K / 32;
    constexpr int DATA_CHUNKS = OUTER_N * OK_BLOCKS;

    for (int c = tid; c < DATA_CHUNKS; c += BLOCK_SIZE) {
        int row = c / OK_BLOCKS;
        int blk = c % OK_BLOCKS;
        int g_n = outer_n + row;
        int g_col = k_half_base + blk * 16;
        *reinterpret_cast<uint128_vec*>(&smem_data[blk][row * 16]) =
            *reinterpret_cast<const uint128_vec*>(&B_data[g_n][g_col]);
    }

    for (int s = tid; s < OUTER_N * OK_BLOCKS; s += BLOCK_SIZE) {
        int row = s / OK_BLOCKS;
        int col = s % OK_BLOCKS;
        int g_n = outer_n + row;
        int g_blk = blk_base + col;
        smem_scale[col][row] = B_scale[sh_scale_off<NUM_BLOCKS>(g_n, g_blk)];
    }
}

// Store C from shared memory to global memory (BF16 output)
template <int M, int N, int OUTER_M, int OUTER_N, int BLOCK_SIZE>
__device__ __forceinline__ void store_c_from_smem(
    const float smem_c[][OUTER_N],
    hip_bfloat16 C[][N],
    int outer_m, int outer_n, int tid
) {
    constexpr int TOTAL = OUTER_M * OUTER_N;
    constexpr int PER_THREAD = (TOTAL + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int i = 0; i < PER_THREAD; i++) {
        int idx = tid + i * BLOCK_SIZE;
        if (idx < TOTAL) {
            int local_m = idx / OUTER_N;
            int local_n = idx % OUTER_N;
            int gm = outer_m + local_m;
            int gn = outer_n + local_n;
            if (gm < M && gn < N)
                C[gm][gn] = static_cast<hip_bfloat16>(smem_c[local_m][local_n]);
        }
    }
}

// Store C from shared memory to workspace (float output for split-K)
template <int M, int N, int OUTER_M, int OUTER_N, int BLOCK_SIZE>
__device__ __forceinline__ void store_c_from_smem_f32(
    const float smem_c[][OUTER_N],
    float* __restrict__ workspace,
    int outer_m, int outer_n, int tid
) {
    constexpr int TOTAL = OUTER_M * OUTER_N;
    constexpr int PER_THREAD = (TOTAL + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int i = 0; i < PER_THREAD; i++) {
        int idx = tid + i * BLOCK_SIZE;
        if (idx < TOTAL) {
            int local_m = idx / OUTER_N;
            int local_n = idx % OUTER_N;
            int gm = outer_m + local_m;
            int gn = outer_n + local_n;
            workspace[gm * N + gn] = smem_c[local_m][local_n];
        }
    }
}

template <int OUTER_K, int OUTER_N, int IM, int IN, int IK,
          int WARP_TILES_M = 1, int WARP_TILES_N = 1, int OK_HALF = OUTER_K / 2>
__device__ __forceinline__ void inner_mfma_loop(
    const uint8_t smem_a_data[][OK_HALF], const uint8_t smem_a_scale[][OUTER_K / 32],
    const uint8_t smem_b_data[][OUTER_N * 16], const uint8_t smem_b_scale[][OUTER_N],
    int warp_m, int warp_n, int lane,
    float smem_c[][OUTER_N],
    typename MfmaTraits<IM, IN, IK>::acc_t& reg_acc
) {
    using Traits = MfmaTraits<IM, IN, IK>;
    constexpr int OK_BLOCKS = OUTER_K / 32;
    constexpr int BPC = Traits::BLOCKS_PER_CALL;
    constexpr int ITERS = OUTER_K / IK;

    if constexpr (WARP_TILES_M * WARP_TILES_N == 1) {
        // Single tile per warp: accumulate in registers, no LDS
        int tile_m_local = warp_m * IM;
        int tile_n_local = warp_n * IN;

        for (int ik = 0; ik < ITERS; ik++) {
            int blk0 = ik * BPC;
            uint32_t a_reg[Traits::REGS];
            auto a_sc = load_a_with_scale<OK_HALF, IM, IN, IK>(
                smem_a_data, smem_a_scale, tile_m_local, blk0, lane, a_reg);

            uint32_t b_reg[Traits::REGS];
            int b_row, b_blk;
            load_transposed<IM, OUTER_N>(smem_b_data,
                tile_n_local, blk0, lane, b_reg, b_blk, b_row);
            int32_t b_sc = broadcast_scale(
                smem_b_scale[b_blk][b_row]);

            reg_acc = Traits::mfma(a_reg, a_sc, b_reg, b_sc, reg_acc);
        }
    } else {
        // Multi-tile per warp: use LDS for accumulation
        for (int ik = 0; ik < ITERS; ik++) {
            int blk0 = ik * BPC;

            if constexpr (IM >= IN) {
                // Reuse A: outer loop M, inner loop N
                for (int wt_m = 0; wt_m < WARP_TILES_M; wt_m++) {
                    uint32_t a_reg[Traits::REGS];
                    int tile_m_local = (warp_m * WARP_TILES_M + wt_m) * IM;
                    auto a_sc = load_a_with_scale<OK_HALF, IM, IN, IK>(
                        smem_a_data, smem_a_scale, tile_m_local, blk0, lane, a_reg);

                    for (int wt_n = 0; wt_n < WARP_TILES_N; wt_n++) {
                        uint32_t b_reg[Traits::REGS];
                        int b_row, b_blk;
                        load_transposed<IM, OUTER_N>(smem_b_data,
                            (warp_n * WARP_TILES_N + wt_n) * IN,
                            blk0, lane, b_reg, b_blk, b_row);
                        int32_t b_sc = broadcast_scale(
                            smem_b_scale[b_blk][b_row]);

                        int tile_n_local = (warp_n * WARP_TILES_N + wt_n) * IN;
                        typename Traits::acc_t acc;
                        Traits::template load_acc_from_smem<OUTER_N>(
                            smem_c, acc, tile_m_local, tile_n_local, lane);

                        acc = Traits::mfma(a_reg, a_sc, b_reg, b_sc, acc);

                        Traits::template store_acc_to_smem<OUTER_N>(
                            smem_c, acc, tile_m_local, tile_n_local, lane);
                    }
                }
            } else {
                for (int wt_n = 0; wt_n < WARP_TILES_N; wt_n++) {
                    uint32_t b_reg[Traits::REGS];
                    int b_row, b_blk;
                    load_transposed<IM, OUTER_N>(smem_b_data,
                        (warp_n * WARP_TILES_N + wt_n) * IN,
                        blk0, lane, b_reg, b_blk, b_row);
                    int32_t b_sc = broadcast_scale(
                        smem_b_scale[b_blk][b_row]);
                    int tile_n_local = (warp_n * WARP_TILES_N + wt_n) * IN;

                    for (int wt_m = 0; wt_m < WARP_TILES_M; wt_m++) {
                        uint32_t a_reg[Traits::REGS];
                        int tile_m_local = (warp_m * WARP_TILES_M + wt_m) * IM;
                        auto a_sc = load_a_with_scale<OK_HALF, IM, IN, IK>(
                            smem_a_data, smem_a_scale, tile_m_local, blk0, lane, a_reg);
                        typename Traits::acc_t acc;
                        Traits::template load_acc_from_smem<OUTER_N>(
                            smem_c, acc, tile_m_local, tile_n_local, lane);

                        acc = Traits::mfma(a_reg, a_sc, b_reg, b_sc, acc);

                        Traits::template store_acc_to_smem<OUTER_N>(
                            smem_c, acc, tile_m_local, tile_n_local, lane);
                    }
                }
            }
        }
    } // end multi-tile
}


// =====================================================================
// Tiled kernel: multi-wavefront, LDS-backed
// =====================================================================

template <int WARPS, int M, int N, int K, int NUM_BLOCKS,
          int OUTER_M, int OUTER_N, int OUTER_K,
          int IM, int IN, int IK,
          int WARP_TILES_M = 1, int WARP_TILES_N = 1,
          bool FUSE_A_QUANT = false, int BUFFERS = 1,
          int OCCUPANCY = -1, int K_HALF = K / 2>
__global__ void
__launch_bounds__(WARPS * 64, OCCUPANCY == -1 ? (LdsLayout<OUTER_M, OUTER_N, OUTER_K, WARP_TILES_M, WARP_TILES_N>::OCCUPANCY) / BUFFERS : OCCUPANCY)
mfma_fp4_gemm_tiled(
    const hip_bfloat16 A_bf16[][K],
    const uint8_t A_data[][K_HALF],
    const uint8_t B_data[][K_HALF],
    const uint8_t A_scale[][NUM_BLOCKS],
    const uint8_t* __restrict__ B_scale,
    hip_bfloat16 C[][N]
) {
    constexpr int WARPS_M = OUTER_M / (IM * WARP_TILES_M);
    constexpr int WARPS_N = OUTER_N / (IN * WARP_TILES_N);
    static_assert(WARPS == WARPS_M * WARPS_N);

    using Traits = MfmaTraits<IM, IN, IK>;
    using Lds = LdsLayout<OUTER_M, OUTER_N, OUTER_K, WARP_TILES_M, WARP_TILES_N>;
    constexpr int OK_BLOCKS = OUTER_K / 32;
    constexpr int OK_HALF = OUTER_K / 2;
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

    __shared__ uint8_t smem_a_data[BUFFERS][OUTER_M][OK_HALF];
    __shared__ uint8_t smem_a_scale[BUFFERS][OUTER_M][OK_BLOCKS];
    __shared__ uint8_t smem_b_data[BUFFERS][OK_BLOCKS][OUTER_N * 16];
    __shared__ uint8_t smem_b_scale[BUFFERS][OK_BLOCKS][OUTER_N];
    extern __shared__ float smem_c_raw[];
    auto (&smem_c_data)[OUTER_M][OUTER_N] = *reinterpret_cast<float (*)[OUTER_M][OUTER_N]>(smem_c_raw);

    typename Traits::acc_t reg_acc;
    if constexpr (WARP_TILES_M * WARP_TILES_N == 1) {
        reg_acc = Traits::zero_acc();
    }

    if constexpr (WARP_TILES_M * WARP_TILES_N > 1) {
        for (int i = tid; i < Lds::C_DATA_FLOATS; i += BLOCK_SIZE)
            reinterpret_cast<float*>(smem_c_data)[i] = 0.0f;
    }

    // Load the first tile
    if constexpr (FUSE_A_QUANT) {
        quantize_a_to_lds<M, K, OUTER_M, OUTER_K, BLOCK_SIZE>(
            A_bf16, smem_a_data[0], smem_a_scale[0], outer_m, tid);
    } else {
        load_a_to_lds<M, K_HALF, NUM_BLOCKS, OUTER_M, OUTER_K, BLOCK_SIZE>(
            A_data, A_scale,
            smem_a_data[0], reinterpret_cast<uint8_t*>(smem_a_scale[0]),
            outer_m, 0, 0, tid);
    }

    load_b_to_lds<N, K_HALF, NUM_BLOCKS, OUTER_N, OUTER_K, BLOCK_SIZE>(
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

        load_b_global_to_reg<N, K_HALF, NUM_BLOCKS, OUTER_N, OUTER_K, BLOCK_SIZE>(
            B_data, B_scale,
            outer_n, next_ok * OK_HALF, next_ok * OK_BLOCKS, tid,
            b_data_regs, b_scale_regs);

        inner_mfma_loop<OUTER_K, OUTER_N, IM, IN, IK, WARP_TILES_M, WARP_TILES_N>(
            smem_a_data[buf], smem_a_scale[buf],
            smem_b_data[buf], smem_b_scale[buf],
            warp_m, warp_n, lane, smem_c_data, reg_acc);

        if constexpr (BUFFERS == 1 && WARPS > 1) {
            __syncthreads();
        }

        if constexpr (FUSE_A_QUANT) {
            store_quant_a_to_lds<OUTER_M, OUTER_K, BLOCK_SIZE>(
                smem_a_data[next_buf], smem_a_scale[next_buf],
                tid, data_regs, scale_regs);
        } else {
            store_a_reg_to_lds<OUTER_M, OUTER_K, BLOCK_SIZE>(
                smem_a_data[next_buf], reinterpret_cast<uint8_t*>(smem_a_scale[next_buf]),
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
        warp_m, warp_n, lane, smem_c_data, reg_acc);

    if constexpr (WARP_TILES_M * WARP_TILES_N == 1) {
        Traits::template store<M, N>(C, reg_acc,
            outer_m + warp_m * IM, outer_n + warp_n * IN, lane);
    } else {
        if constexpr (WARPS > 1) {
            __syncthreads();
        }
        store_c_from_smem<M, N, OUTER_M, OUTER_N, BLOCK_SIZE>(
            smem_c_data, C, outer_m, outer_n, tid);
    }
}

// =====================================================================
// Split-K kernel: distributes K across blockIdx.z, writes fp32 partials
// =====================================================================

template <int M, int N, int K, int NUM_BLOCKS,
          int IM, int IN, int IK,
          int WARPS_M = 1, int WARPS_N = 1, int K_SPLITS = 1, int K_HALF = K / 2>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 64) mfma_fp4_gemm_splitk(
    const uint8_t A_data[][K_HALF],
    const uint8_t B_data[][K_HALF],
    const uint8_t A_scale[][NUM_BLOCKS],
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

    load_ab_global<M, N, K_HALF, NUM_BLOCKS, IM, IN, IK>(
        A_data, B_data, A_scale, B_scale,
        tile_m, tile_n, ki_start * BPC, lane,
        a_cur, a_sc_cur, b_cur, b_sc_cur);

    for (int ki = ki_start; ki < ki_end - 1; ki++) {
        load_ab_global<M, N, K_HALF, NUM_BLOCKS, IM, IN, IK>(
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
// Split-K Fused kernel: quantize A on-the-fly, split K across blockIdx.z
// =====================================================================

template <int M, int N, int K, int NUM_BLOCKS,
          int IM, int IN, int IK,
          int WARPS_M = 1, int WARPS_N = 1, int K_SPLITS = 1,
          int BLOCK_SIZE = WARPS_M * WARPS_N * 64, int K_HALF = K / 2>
__global__ void __launch_bounds__(BLOCK_SIZE) mfma_fp4_gemm_splitk_fused(
    const hip_bfloat16 A_bf16[][K],
    const uint8_t B_data[][N],
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

    if constexpr (Traits::REGS == 8) {
        a_cur[4] = 0; a_cur[5] = 0; a_cur[6] = 0; a_cur[7] = 0;
        a_nxt[4] = 0; a_nxt[5] = 0; a_nxt[6] = 0; a_nxt[7] = 0;
    }

    // First tile: quantize A on-the-fly, load B normally
    {
        int a_row = tile_m + (lane % IM);
        int a_blk = ki_start * BPC + (lane / IM);
        QuantBlock qb = quantize_fp4_block_bf16(&A_bf16[a_row * K + a_blk * 32]);
        *reinterpret_cast<uint128_vec*>(&a_cur[0]) = *reinterpret_cast<uint128_vec*>(&qb.data);
        a_sc_cur = broadcast_scale(qb.e8m0);
    }
    {
        int b_row, b_blk;
        load_tile<IN, K_HALF>(B_data, tile_n, ki_start * BPC, lane, b_cur, b_blk, b_row);
        b_sc_cur = broadcast_scale(B_scale[sh_scale_off<NUM_BLOCKS>(b_row, b_blk)]);
    }

    for (int ki = ki_start; ki < ki_end - 1; ki++) {
        {
            int a_row = tile_m + (lane % IM);
            int a_blk = (ki + 1) * BPC + (lane / IM);
            QuantBlock qb = quantize_fp4_block_bf16(&A_bf16[a_row * K + a_blk * 32]);
            *reinterpret_cast<uint128_vec*>(&a_nxt[0]) = *reinterpret_cast<uint128_vec*>(&qb.data);
            a_sc_nxt = broadcast_scale(qb.e8m0);
        }
        {
            int b_row, b_blk;
            load_tile<IN, K_HALF>(B_data, tile_n, (ki + 1) * BPC, lane, b_nxt, b_blk, b_row);
            b_sc_nxt = broadcast_scale(B_scale[sh_scale_off<NUM_BLOCKS>(b_row, b_blk)]);
        }

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
            ws[gm * N + gn] = acc[i];
        }
    } else {
        int col = lane % 16;
        int quad = lane / 16;
        for (int i = 0; i < 4; i++) {
            int row = i + 4 * quad;
            int gm = tile_m + row;
            int gn = tile_n + col;
            ws[gm * N + gn] = acc[i];
        }
    }
}

template <int WARPS, int M, int N, int K,
          int OUTER_M, int OUTER_N, int OUTER_K,
          int IM, int IN, int IK,
          int WARP_TILES_M = 1, int WARP_TILES_N = 1,
          int K_SPLITS = 1, int BUFFERS = 1,
          int OCCUPANCY = -1,
          int K_HALF = K / 2, int NUM_BLOCKS = K / 32>
__global__ void
__launch_bounds__(WARPS * 64, OCCUPANCY == -1 ? (LdsLayout<OUTER_M, OUTER_N, OUTER_K>::OCCUPANCY) / BUFFERS : OCCUPANCY)
mfma_fp4_gemm_tiled_splitk_fused(
    const hip_bfloat16 A_bf16[][K],
    const uint8_t B_data[][K_HALF],
    const uint8_t* __restrict__ B_scale,
    float* __restrict__ workspace
) {
    constexpr int WARPS_M = OUTER_M / (IM * WARP_TILES_M);
    constexpr int WARPS_N = OUTER_N / (IN * WARP_TILES_N);
    static_assert(WARPS == WARPS_M * WARPS_N);

    using Traits = MfmaTraits<IM, IN, IK>;
    using Lds = LdsLayout<OUTER_M, OUTER_N, OUTER_K>;
    constexpr int OK_BLOCKS = OUTER_K / 32;
    constexpr int OK_HALF = OUTER_K / 2;
    constexpr int TOTAL_OK_ITERS = (NUM_BLOCKS + OK_BLOCKS - 1) / OK_BLOCKS;
    constexpr int ITERS_PER_SPLIT = (TOTAL_OK_ITERS + K_SPLITS - 1) / K_SPLITS;
    constexpr int BLOCK_SIZE = WARPS * 64;

    using Q = QuantAPerThread<M, K, OUTER_M, OUTER_K, BLOCK_SIZE>;
    constexpr int B_DATA_PER_THREAD = ((OUTER_N * OK_BLOCKS) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int B_SCALE_PER_THREAD = ((OUTER_N * OK_BLOCKS) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const int outer_m = __builtin_amdgcn_readfirstlane(blockIdx.x * OUTER_M);
    const int outer_n = __builtin_amdgcn_readfirstlane(blockIdx.y * OUTER_N);
    const int split_id = __builtin_amdgcn_readfirstlane(blockIdx.z);
    const int tid = threadIdx.x;
    const int warp_id = __builtin_amdgcn_readfirstlane(tid / 64);
    const int lane = tid % 64;
    const int warp_m = __builtin_amdgcn_readfirstlane(warp_id / WARPS_N);
    const int warp_n = __builtin_amdgcn_readfirstlane(warp_id % WARPS_N);

    const int ok_start = __builtin_amdgcn_readfirstlane(split_id * ITERS_PER_SPLIT);
    const int ok_end = __builtin_amdgcn_readfirstlane(
        min(ok_start + ITERS_PER_SPLIT, TOTAL_OK_ITERS));
    const int num_iters = ok_end - ok_start;

    __shared__ uint8_t smem_a_data[BUFFERS][OUTER_M][OK_HALF];
    __shared__ uint8_t smem_a_scale[BUFFERS][OUTER_M][OK_BLOCKS];
    __shared__ uint8_t smem_b_data[BUFFERS][OK_BLOCKS][OUTER_N * 16];
    __shared__ uint8_t smem_b_scale[BUFFERS][OK_BLOCKS][OUTER_N];
    __shared__ float smem_c_data[OUTER_M][OUTER_N];

    typename Traits::acc_t reg_acc;
    if constexpr (WARP_TILES_M * WARP_TILES_N == 1) {
        reg_acc = Traits::zero_acc();
    }

    if constexpr (WARP_TILES_M * WARP_TILES_N > 1) {
        for (int i = tid; i < Lds::C_DATA_FLOATS; i += BLOCK_SIZE)
            reinterpret_cast<float*>(smem_c_data)[i] = 0.0f;
    }

    // Load first tile: quantize A from BF16, load pre-quantized B
    {
        uint32_t a_regs[Q::DATA_REGS];
        uint8_t a_sc_regs[Q::SCALE_REGS];
        quantize_a_to_reg<M, K, OUTER_M, OUTER_K, BLOCK_SIZE>(
            A_bf16, outer_m, tid, a_regs, a_sc_regs, ok_start * OUTER_K);
        store_quant_a_to_lds<OUTER_M, OUTER_K, BLOCK_SIZE>(
            smem_a_data[0], smem_a_scale[0], tid, a_regs, a_sc_regs);
    }

    load_b_to_lds<N, K_HALF, NUM_BLOCKS, OUTER_N, OUTER_K, BLOCK_SIZE>(
        B_data, B_scale,
        smem_b_data[0], smem_b_scale[0],
        outer_n, ok_start * OK_HALF, ok_start * OK_BLOCKS, tid);

    // Main K loop with prefetching
    for (int iter = 0; iter < num_iters - 1; iter++) {
        if constexpr (WARPS > 1) __syncthreads();

        int next_ok = ok_start + iter + 1;
        const int buf = iter % BUFFERS;
        const int next_buf = (iter + 1) % BUFFERS;

        // Prefetch next A: quantize from BF16 into registers
        uint32_t a_regs[Q::DATA_REGS];
        uint8_t a_sc_regs[Q::SCALE_REGS];
        quantize_a_to_reg<M, K, OUTER_M, OUTER_K, BLOCK_SIZE>(
            A_bf16, outer_m, tid, a_regs, a_sc_regs, next_ok * OUTER_K);

        // Prefetch next B into registers
        uint32_t b_data_regs[B_DATA_PER_THREAD * 4];
        uint8_t b_scale_regs[B_SCALE_PER_THREAD];
        load_b_global_to_reg<N, K_HALF, NUM_BLOCKS, OUTER_N, OUTER_K, BLOCK_SIZE>(
            B_data, B_scale,
            outer_n, next_ok * OK_HALF, next_ok * OK_BLOCKS, tid,
            b_data_regs, b_scale_regs);

        // Compute current tile from LDS
        inner_mfma_loop<OUTER_K, OUTER_N, IM, IN, IK, WARP_TILES_M, WARP_TILES_N>(
            smem_a_data[buf], smem_a_scale[buf],
            smem_b_data[buf], smem_b_scale[buf],
            warp_m, warp_n, lane, smem_c_data, reg_acc);

        if constexpr (BUFFERS == 1 && WARPS > 1) __syncthreads();

        // Store prefetched data to LDS
        store_quant_a_to_lds<OUTER_M, OUTER_K, BLOCK_SIZE>(
            smem_a_data[next_buf], smem_a_scale[next_buf],
            tid, a_regs, a_sc_regs);

        store_b_reg_to_lds<OUTER_N, OUTER_K, BLOCK_SIZE>(
            smem_b_data[next_buf], smem_b_scale[next_buf],
            tid, b_data_regs, b_scale_regs);
    }

    // Process last tile
    if constexpr (WARPS > 1) __syncthreads();

    {
        const int last_buf = (num_iters - 1) % BUFFERS;
        inner_mfma_loop<OUTER_K, OUTER_N, IM, IN, IK, WARP_TILES_M, WARP_TILES_N>(
            smem_a_data[last_buf], smem_a_scale[last_buf],
            smem_b_data[last_buf], smem_b_scale[last_buf],
            warp_m, warp_n, lane, smem_c_data, reg_acc);
    }

    if constexpr (WARP_TILES_M * WARP_TILES_N == 1) {
        float* ws = workspace + split_id * M * N;
        Traits::template store_f32<N>(ws, reg_acc,
            outer_m + warp_m * IM, outer_n + warp_n * IN, lane);
    } else {
        if constexpr (WARPS > 1) {
            __syncthreads();
        }
        float* ws = workspace + split_id * M * N;
        store_c_from_smem_f32<M, N, OUTER_M, OUTER_N, BLOCK_SIZE>(
            smem_c_data, ws, outer_m, outer_n, tid);
    }
}

template <int WARPS, int M, int N, int K,
          int OUTER_M, int OUTER_N, int OUTER_K,
          int IM, int IN, int IK,
          int WARP_TILES_M = 1, int WARP_TILES_N = 1,
          int K_SPLITS = 1, int BUFFERS = 1,
          int OCCUPANCY = -1,
          bool PROFILE_PHASES = false,
          int K_HALF = K / 2, int NUM_BLOCKS = K / 32>
__global__ void
__launch_bounds__(WARPS * 64, OCCUPANCY == -1 ? (LdsLayout<OUTER_M, OUTER_N, OUTER_K>::OCCUPANCY) / BUFFERS : OCCUPANCY)
mfma_fp4_gemm_tiled_splitk_coop(
    const hip_bfloat16 A_bf16[][K],
    uint8_t A_data[][K_HALF],
    uint8_t A_scale[][NUM_BLOCKS],
    const uint8_t B_data[][N],
    const uint8_t* __restrict__ B_scale,
    float* __restrict__ workspace,
    int* __restrict__ quant_counter,
    int* __restrict__ split_ready  // array of K_SPLITS ints: counts completed chunks per K-split
) {
    constexpr int WARPS_M = OUTER_M / (IM * WARP_TILES_M);
    constexpr int WARPS_N = OUTER_N / (IN * WARP_TILES_N);
    static_assert(WARPS == WARPS_M * WARPS_N);

    using Traits = MfmaTraits<IM, IN, IK>;
    using Lds = LdsLayout<OUTER_M, OUTER_N, OUTER_K>;
    constexpr int OK_BLOCKS = OUTER_K / 32;
    constexpr int OK_HALF = OUTER_K / 2;
    constexpr int TOTAL_OK_ITERS = (NUM_BLOCKS + OK_BLOCKS - 1) / OK_BLOCKS;
    constexpr int ITERS_PER_SPLIT = (TOTAL_OK_ITERS + K_SPLITS - 1) / K_SPLITS;
    constexpr int BLOCK_SIZE = WARPS * 64;
    constexpr int TOTAL_QUANT = M * NUM_BLOCKS;
    constexpr int BLKS_PER_SPLIT = ITERS_PER_SPLIT * OK_BLOCKS;
    constexpr int CHUNKS_PER_SPLIT = M * BLKS_PER_SPLIT;

    constexpr int A_CHUNKS_PER_THREAD = ((OUTER_M * OK_BLOCKS) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int A_SCALE_PER_THREAD = A_CHUNKS_PER_THREAD;
    constexpr int B_DATA_PER_THREAD = ((OUTER_N * OK_BLOCKS) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr int B_SCALE_PER_THREAD = ((OUTER_N * OK_BLOCKS) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const int outer_m = __builtin_amdgcn_readfirstlane(blockIdx.x * OUTER_M);
    const int outer_n = __builtin_amdgcn_readfirstlane(blockIdx.y * OUTER_N);
    const int split_id = __builtin_amdgcn_readfirstlane(blockIdx.z);
    const int tid = threadIdx.x;
    const int warp_id = __builtin_amdgcn_readfirstlane(tid / 64);
    const int lane = tid % 64;
    const int warp_m = __builtin_amdgcn_readfirstlane(warp_id / WARPS_N);
    const int warp_n = __builtin_amdgcn_readfirstlane(warp_id % WARPS_N);

    const int ok_start = __builtin_amdgcn_readfirstlane(split_id * ITERS_PER_SPLIT);
    const int ok_end = __builtin_amdgcn_readfirstlane(
        min(ok_start + ITERS_PER_SPLIT, TOTAL_OK_ITERS));
    const int num_iters = ok_end - ok_start;

    // Per-phase timing (block 0 only, enabled via PROFILE_PHASES template flag)
    [[maybe_unused]] const bool is_timer_block = PROFILE_PHASES && (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0);
    [[maybe_unused]] uint64_t t_start = 0, t_phase1 = 0, t_phase2 = 0, t_phase3 = 0;
    if constexpr (PROFILE_PHASES) {
        if (is_timer_block && tid == 0)
            t_start = __builtin_amdgcn_s_memrealtime();
    }

    // =========================================================
    // Phase 1: Cooperative A quantization with per-split signaling.
    // Chunks ordered (blk, row) so early K-splits finish first.
    // Deferred signal piggybacking on broadcast sync.
    // =========================================================
    int batch_start;
    while (true) {
        if (tid == 0) {
            batch_start = atomicAdd(quant_counter, BLOCK_SIZE);
        }

        if constexpr (WARPS == 1) {
            batch_start = __builtin_amdgcn_readfirstlane(batch_start);
        } else {
            __shared__ int s_bs;
            if (tid == 0) s_bs = batch_start;
            __syncthreads();
            batch_start = s_bs;
        }

        if (batch_start >= TOTAL_QUANT) break;

        int my_chunk = batch_start + tid;
        if (my_chunk < TOTAL_QUANT) {
            // (blk, row) ordering: early blks (early K-splits) processed first
            int blk = my_chunk / M;
            int row = my_chunk % M;

            QuantBlock qb = quantize_fp4_block_bf16(&A_bf16[row * K + blk * 32]);

            int data_off = row * K_HALF + blk * 16;
            *reinterpret_cast<uint128_vec*>(&A_data[data_off]) =
                *reinterpret_cast<uint128_vec*>(&qb.data);
            A_scale[row + blk * M] = qb.e8m0;
            __threadfence();
        }
    }

    if constexpr (PROFILE_PHASES) {
        if (is_timer_block && tid == 0)
            t_phase1 = __builtin_amdgcn_s_memrealtime();
    }

    // =========================================================
    // Phase 2: Grid-wide cooperative sync.
    // Hardware-managed barrier - no polling, no cache contention.
    // Requires hipLaunchCooperativeKernel on the host side.
    // =========================================================
    namespace cg = cooperative_groups;
    cg::grid_group grid = cg::this_grid();
    grid.sync();

    if constexpr (PROFILE_PHASES) {
        if (is_timer_block && tid == 0)
            t_phase2 = __builtin_amdgcn_s_memrealtime();
    }

    // =========================================================
    // Phase 3: Tiled GEMM reading pre-quantized A from global
    // =========================================================
    __shared__ uint8_t smem_a_data[BUFFERS][OUTER_M][OK_HALF];
    __shared__ uint8_t smem_a_scale[BUFFERS][OUTER_M][OK_BLOCKS];
    __shared__ uint8_t smem_b_data[BUFFERS][OK_BLOCKS][OUTER_N * 16];
    __shared__ uint8_t smem_b_scale[BUFFERS][OK_BLOCKS][OUTER_N];
    __shared__ float smem_c_data[OUTER_M][OUTER_N];

    // Zero-initialize shared memory accumulator
    for (int i = tid; i < Lds::C_DATA_FLOATS; i += BLOCK_SIZE)
        reinterpret_cast<float*>(smem_c_data)[i] = 0.0f;

    // Load first tile from pre-quantized global memory
    load_a_to_lds<M, K_HALF, NUM_BLOCKS, OUTER_M, OUTER_K, BLOCK_SIZE>(
        A_data, A_scale,
        smem_a_data[0], reinterpret_cast<uint8_t*>(smem_a_scale[0]),
        outer_m, ok_start * OK_HALF, ok_start * OK_BLOCKS, tid);

    load_b_to_lds<N, K_HALF, NUM_BLOCKS, OUTER_N, OUTER_K, BLOCK_SIZE>(
        B_data, B_scale,
        smem_b_data[0], smem_b_scale[0],
        outer_n, ok_start * OK_HALF, ok_start * OK_BLOCKS, tid);

    // Main K loop with prefetching
    for (int iter = 0; iter < num_iters - 1; iter++) {
        if constexpr (WARPS > 1) __syncthreads();

        int next_ok = ok_start + iter + 1;
        const int buf = iter % BUFFERS;
        const int next_buf = (iter + 1) % BUFFERS;

        // Prefetch next A from global (already quantized)
        uint32_t a_data_regs[A_CHUNKS_PER_THREAD * 4];
        uint8_t a_scale_regs[A_SCALE_PER_THREAD];
        load_a_global_to_reg<M, K_HALF, NUM_BLOCKS, OUTER_M, OUTER_K, BLOCK_SIZE>(
            A_data, A_scale,
            outer_m, next_ok * OK_HALF, next_ok * OK_BLOCKS, tid,
            a_data_regs, a_scale_regs);

        // Prefetch next B
        uint32_t b_data_regs[B_DATA_PER_THREAD * 4];
        uint8_t b_scale_regs[B_SCALE_PER_THREAD];
        load_b_global_to_reg<N, K_HALF, NUM_BLOCKS, OUTER_N, OUTER_K, BLOCK_SIZE>(
            B_data, B_scale,
            outer_n, next_ok * OK_HALF, next_ok * OK_BLOCKS, tid,
            b_data_regs, b_scale_regs);

        // Compute current tile
        inner_mfma_loop<OUTER_K, OUTER_N, IM, IN, IK, WARP_TILES_M, WARP_TILES_N>(
            smem_a_data[buf], smem_a_scale[buf],
            smem_b_data[buf], smem_b_scale[buf],
            warp_m, warp_n, lane, smem_c_data);

        if constexpr (BUFFERS == 1 && WARPS > 1) __syncthreads();

        store_a_reg_to_lds<OUTER_M, OUTER_K, BLOCK_SIZE>(
            smem_a_data[next_buf], reinterpret_cast<uint8_t*>(smem_a_scale[next_buf]),
            tid, a_data_regs, a_scale_regs);

        store_b_reg_to_lds<OUTER_N, OUTER_K, BLOCK_SIZE>(
            smem_b_data[next_buf], smem_b_scale[next_buf],
            tid, b_data_regs, b_scale_regs);
    }

    // Last tile
    if constexpr (WARPS > 1) __syncthreads();
    {
        const int last_buf = (num_iters - 1) % BUFFERS;
        inner_mfma_loop<OUTER_K, OUTER_N, IM, IN, IK, WARP_TILES_M, WARP_TILES_N>(
            smem_a_data[last_buf], smem_a_scale[last_buf],
            smem_b_data[last_buf], smem_b_scale[last_buf],
            warp_m, warp_n, lane, smem_c_data);
    }

    if constexpr (WARPS > 1) {
        __syncthreads();
    }

    // Write fp32 partials to workspace from shared memory
    float* ws = workspace + split_id * M * N;
    store_c_from_smem_f32<M, N, OUTER_M, OUTER_N, BLOCK_SIZE>(
        smem_c_data, ws, outer_m, outer_n, tid);

    if constexpr (PROFILE_PHASES) {
        if (is_timer_block && tid == 0) {
            t_phase3 = __builtin_amdgcn_s_memrealtime();
            printf("[TSKC Timing] Phase1(quant)=%llu Phase2(barrier)=%llu Phase3(gemm)=%llu Total=%llu cycles\\n",
                   (unsigned long long)(t_phase1 - t_start),
                   (unsigned long long)(t_phase2 - t_phase1),
                   (unsigned long long)(t_phase3 - t_phase2),
                   (unsigned long long)(t_phase3 - t_start));
        }
    }
}
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

static int g_generation = 0;

void reset_buffers() {
    g_generation++;
}

// ---- Profiling ----
struct PerfStats {
    float t_quant = 0, t_gemm = 0, t_reduce = 0;
    int count = 0;
};

constexpr int PROFILE_INTERVAL = 10;

template <int M, int N, int K,
          int IM, int IN, int IK,
          int WARPS_M = 1, int WARPS_N = 1, int K_HALF = K / 2, int NUM_BLOCKS = K / 32>
void launch_simple(torch::Tensor A_bf16,
                    torch::Tensor B_data, torch::Tensor B_scale,
                    torch::Tensor C, bool profile) {
    // Pre-allocated scratch (allocated once, reused)
    static torch::Tensor A_data_buf, A_scale_buf;
    static int local_gen = -1;
    if (local_gen != g_generation) {
        auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(A_bf16.device());
        A_data_buf = torch::empty({M, K_HALF}, opts);
        A_scale_buf = torch::empty({NUM_BLOCKS * M}, opts);
        local_gen = g_generation;
    }

    static std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, PerfStats>>> perf_map;
    auto& stats = perf_map[M][N][K_HALF];
    stats.count++;
    bool do_profile = profile && (stats.count % PROFILE_INTERVAL == 0);

    hipEvent_t e0, e1, e2;
    if (do_profile) {
        (void)hipEventCreate(&e0); (void)hipEventCreate(&e1); (void)hipEventCreate(&e2);
        (void)hipEventRecord(e0);
    }

    // Launch warp-parallel quant kernel
    constexpr int QUANT_VPT = 2;  // values per thread
    constexpr int q_total = M * NUM_BLOCKS;
    constexpr int q_block = 64;
    constexpr int q_threads_per_group = 32 / QUANT_VPT;
    constexpr int q_groups_per_block = q_block / q_threads_per_group;
    constexpr int q_grid = (q_total + q_groups_per_block - 1) / q_groups_per_block;
    quant_a_kernel_warp_parallel<M, K, NUM_BLOCKS, QUANT_VPT>
    <<<q_grid, q_block>>>(
        reinterpret_cast<const hip_bfloat16(*)[K]>(A_bf16.data_ptr()),
        reinterpret_cast<uint8_t(*)[K_HALF]>(A_data_buf.data_ptr()),
        reinterpret_cast<uint8_t(*)[NUM_BLOCKS]>(A_scale_buf.data_ptr()));

    if (do_profile) (void)hipEventRecord(e1);

    constexpr int BLOCK_M = IM * WARPS_M;
    constexpr int BLOCK_N = IN * WARPS_N;
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M,
              (N + BLOCK_N - 1) / BLOCK_N);
    dim3 block(64 * WARPS_M * WARPS_N);
    mfma_fp4_gemm_simple<M,N,K,NUM_BLOCKS,IM,IN,IK,WARPS_M,WARPS_N>
        <<<grid, block>>>(
        reinterpret_cast<const uint8_t(*)[K_HALF]>(A_data_buf.data_ptr()),
        reinterpret_cast<const uint8_t(*)[K_HALF]>(B_data.data_ptr()),
        reinterpret_cast<const uint8_t(*)[NUM_BLOCKS]>(A_scale_buf.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_scale.data_ptr()),
        reinterpret_cast<hip_bfloat16(*)[N]>(C.data_ptr()));

    if (do_profile) {
        (void)hipEventRecord(e2);
        (void)hipEventSynchronize(e2);
        float d01, d12;
        (void)hipEventElapsedTime(&d01, e0, e1);
        (void)hipEventElapsedTime(&d12, e1, e2);
        stats.t_quant += d01; stats.t_gemm += d12;
        int n = stats.count / PROFILE_INTERVAL;
        if (n < 5) {
            printf("[Simple GEMM] m=%d n=%d k=%d | "
                "quant=%.1fus gemm=%.1fus | "
                "total=%.1fus (avg over %d)\n",
                M, N, K,
                stats.t_quant/n*1000, stats.t_gemm/n*1000,
                (stats.t_quant+stats.t_gemm)/n*1000, n);
        }
          (void)hipEventDestroy(e0); (void)hipEventDestroy(e1); (void)hipEventDestroy(e2);
      }
}

// =====================================================================
// Simple Fused: single launch, quantize A on-the-fly in GEMM kernel
// =====================================================================
template <int M, int N, int K,
          int IM, int IN, int IK,
          int WARPS_M = 1, int WARPS_N = 1,
          int K_HALF = K / 2, int NUM_BLOCKS = K / 32>
void launch_simple_fused(torch::Tensor A_bf16,
                         torch::Tensor B_data, torch::Tensor B_scale,
                         torch::Tensor C, bool profile) {
    static std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, PerfStats>>> perf_map;
    auto& stats = perf_map[M][N][K_HALF];
    stats.count++;
    bool do_profile = profile && (stats.count % PROFILE_INTERVAL == 0);

    hipEvent_t e0, e1;
    if (do_profile) {
        (void)hipEventCreate(&e0); (void)hipEventCreate(&e1);
        (void)hipEventRecord(e0);
    }

    constexpr int BLOCK_M = IM * WARPS_M;
    constexpr int BLOCK_N = IN * WARPS_N;
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M,
              (N + BLOCK_N - 1) / BLOCK_N);
    dim3 block(64 * WARPS_M * WARPS_N);
    mfma_fp4_gemm_simple_fused<M,N,K,NUM_BLOCKS,IM,IN,IK,WARPS_M,WARPS_N>
        <<<grid, block>>>(
        reinterpret_cast<const hip_bfloat16(*)[K]>(A_bf16.data_ptr()),
        reinterpret_cast<const uint8_t(*)[K_HALF]>(B_data.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_scale.data_ptr()),
        reinterpret_cast<hip_bfloat16(*)[N]>(C.data_ptr()));

    if (do_profile) {
        (void)hipEventRecord(e1);
        (void)hipEventSynchronize(e1);
        float d01;
        (void)hipEventElapsedTime(&d01, e0, e1);
        stats.t_gemm += d01;
        int n = stats.count / PROFILE_INTERVAL;
        if (n < 5) {
            printf("[Simple Fused] m=%d n=%d k=%d | "
                "total=%.1fus (avg over %d)\n",
                M, N, K,
                stats.t_gemm/n*1000, n);
        }
        (void)hipEventDestroy(e0); (void)hipEventDestroy(e1);
    }
}

// =====================================================================
// Cooperative simple: single launch, coop quant + grid.sync + simple GEMM
// =====================================================================
template <int M, int N, int K,
          int IM, int IN, int IK,
          int WARPS_M = 1, int WARPS_N = 1,
          int K_HALF = K / 2, int NUM_BLOCKS = K / 32>
void launch_coop_simple(torch::Tensor A_bf16,
                    torch::Tensor B_data, torch::Tensor B_scale,
                    torch::Tensor C, bool profile) {
    static std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, PerfStats>>> perf_map;
    auto& stats = perf_map[M][N][K_HALF];
    stats.count++;
    bool do_profile = profile && (stats.count % PROFILE_INTERVAL == 0);

    static torch::Tensor A_data_buf, A_scale_buf, quant_ctr_buf;
    static int local_gen = -1;
    if (local_gen != g_generation) {
        auto u8opts = torch::TensorOptions().dtype(torch::kUInt8).device(A_bf16.device());
        auto i32opts = torch::TensorOptions().dtype(torch::kInt32).device(A_bf16.device());
        A_data_buf = torch::empty({M, K_HALF}, u8opts);
        A_scale_buf = torch::empty({NUM_BLOCKS * M}, u8opts);
        quant_ctr_buf = torch::zeros({1}, i32opts);
        local_gen = g_generation;
    }
    quant_ctr_buf.zero_();

    hipEvent_t e0, e1;
    if (do_profile) {
        (void)hipEventCreate(&e0); (void)hipEventCreate(&e1);
        (void)hipEventRecord(e0);
    }

    constexpr int BLOCK_M = IM * WARPS_M;
    constexpr int BLOCK_N = IN * WARPS_N;
    dim3 grid((M + BLOCK_M - 1) / BLOCK_M,
              (N + BLOCK_N - 1) / BLOCK_N);
    dim3 block(64 * WARPS_M * WARPS_N);

    auto A_bf16_ptr = reinterpret_cast<const hip_bfloat16*>(A_bf16.data_ptr());
    auto A_data_ptr = reinterpret_cast<uint8_t(*)[K_HALF]>(A_data_buf.data_ptr());
    auto A_scale_ptr = reinterpret_cast<uint8_t*>(A_scale_buf.data_ptr());
    auto B_data_ptr = reinterpret_cast<const uint8_t(*)[K_HALF]>(B_data.data_ptr());
    auto B_scale_ptr = reinterpret_cast<const uint8_t*>(B_scale.data_ptr());
    auto C_ptr = reinterpret_cast<hip_bfloat16*>(C.data_ptr());
    auto qc_ptr = reinterpret_cast<int*>(quant_ctr_buf.data_ptr());

    void* args[] = {
        &A_bf16_ptr, &A_data_ptr, &A_scale_ptr,
        &B_data_ptr, &B_scale_ptr, &C_ptr, &qc_ptr
    };
    (void)hipLaunchCooperativeKernel(
        (const void*)mfma_fp4_gemm_coop_simple<M,N,K,NUM_BLOCKS,IM,IN,IK,WARPS_M,WARPS_N>,
        grid, block, args, 0, 0);

    if (do_profile) {
        (void)hipEventRecord(e1);
        (void)hipEventSynchronize(e1);
        float d01;
        (void)hipEventElapsedTime(&d01, e0, e1);
        stats.t_gemm += d01;
        int n = stats.count / PROFILE_INTERVAL;
        if (n < 5) {
            printf("[Coop Simple] m=%d n=%d k=%d | "
                "total=%.1fus (avg over %d)\\n",
                M, N, K,
                stats.t_gemm/n*1000, n);
        }
        (void)hipEventDestroy(e0); (void)hipEventDestroy(e1);
    }
}

template <int M, int N, int K,
          int OM, int ON, int OK, int IM, int IN, int IK,
          int WTM = 1, int WTN = 1, int BUFFERS = 1, int OCCUPANCY = -1,
          int K_HALF = K / 2, int NUM_BLOCKS = K / 32>
void launch_tiled(torch::Tensor A_bf16,
                    torch::Tensor B_data, torch::Tensor B_scale,
                    torch::Tensor C, bool profile) {
    // Pre-allocated scratch (allocated once, reused)
    static torch::Tensor A_data_buf, A_scale_buf;
    static int local_gen = -1;
    if (local_gen != g_generation) {
        auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(A_bf16.device());
        A_data_buf = torch::empty({M, K_HALF}, opts);
        A_scale_buf = torch::empty({NUM_BLOCKS * M}, opts);
        local_gen = g_generation;
    }

    static std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, PerfStats>>> perf_map;
    auto& stats = perf_map[M][N][K_HALF];
    stats.count++;
    bool do_profile = profile && (stats.count % PROFILE_INTERVAL == 0);

    hipEvent_t e0, e1, e2;
    if (do_profile) {
        (void)hipEventCreate(&e0); (void)hipEventCreate(&e1); (void)hipEventCreate(&e2);
        (void)hipEventRecord(e0);
    }

    // Launch warp-parallel quant kernel
    constexpr int QUANT_VPT = 2;  // values per thread
    constexpr int q_total = M * NUM_BLOCKS;
    constexpr int q_block = 64;
    constexpr int q_threads_per_group = 32 / QUANT_VPT;
    constexpr int q_groups_per_block = q_block / q_threads_per_group;
    constexpr int q_grid = (q_total + q_groups_per_block - 1) / q_groups_per_block;
    quant_a_kernel_warp_parallel<M, K, NUM_BLOCKS, QUANT_VPT>
        <<<dim3(q_grid), dim3(q_block)>>>(
        reinterpret_cast<const hip_bfloat16(*)[K]>(A_bf16.data_ptr()),
        reinterpret_cast<uint8_t(*)[K_HALF]>(A_data_buf.data_ptr()),
        reinterpret_cast<uint8_t(*)[NUM_BLOCKS]>(A_scale_buf.data_ptr()));

    if (do_profile) (void)hipEventRecord(e1);

    constexpr int WARPS = (OM/(IM*WTM)) * (ON/(IN*WTN));
    constexpr size_t smem_c_bytes = (WTM * WTN > 1) ? OM * ON * sizeof(float) : 0;
    dim3 grid((M+OM-1)/OM, (N+ON-1)/ON);
    dim3 block(WARPS * 64);
    mfma_fp4_gemm_tiled<WARPS,M,N,K,NUM_BLOCKS,OM,ON,OK,IM,IN,IK,WTM,WTN,false,BUFFERS,OCCUPANCY>
        <<<grid, block, smem_c_bytes>>>(
        reinterpret_cast<const hip_bfloat16(*)[K]>(A_bf16.data_ptr()), // unused
        reinterpret_cast<const uint8_t(*)[K_HALF]>(A_data_buf.data_ptr()),
        reinterpret_cast<const uint8_t(*)[K_HALF]>(B_data.data_ptr()),
        reinterpret_cast<const uint8_t(*)[NUM_BLOCKS]>(A_scale_buf.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_scale.data_ptr()),
        reinterpret_cast<hip_bfloat16(*)[N]>(C.data_ptr()));

    if (do_profile) {
        (void)hipEventRecord(e2);
        (void)hipEventSynchronize(e2);
        float d01, d12;
        (void)hipEventElapsedTime(&d01, e0, e1);
        (void)hipEventElapsedTime(&d12, e1, e2);
        stats.t_quant += d01; stats.t_gemm += d12;
        int n = stats.count / PROFILE_INTERVAL;
        if (n < 5) {
            printf("[Tiled GEMM] m=%d n=%d k=%d | "
                "quant=%.1fus gemm=%.1fus | "
                "total=%.1fus (avg over %d)\n",
                M, N, K,
                stats.t_quant/n*1000, stats.t_gemm/n*1000,
                (stats.t_quant+stats.t_gemm)/n*1000, n);
        }
        (void)hipEventDestroy(e0); (void)hipEventDestroy(e1); (void)hipEventDestroy(e2);
    }
}

template <int M, int N, int K,
          int OM, int ON, int OK, int IM, int IN, int IK,
          int WTM = 1, int WTN = 1, int BUFFERS = 1, int OCCUPANCY = -1,
          int NUM_BLOCKS = K / 32>
void launch_tiled_fused(torch::Tensor A_bf16, torch::Tensor B,
                        torch::Tensor Bs, torch::Tensor C) {
    constexpr int WARPS = (OM/(IM*WTM)) * (ON/(IN*WTN));
    dim3 grid((M+OM-1)/OM, (N+ON-1)/ON);
    dim3 block(WARPS * 64);
    mfma_fp4_gemm_tiled<WARPS,M,N,K,NUM_BLOCKS,OM,ON,OK,IM,IN,IK,WTM,WTN,true,BUFFERS,OCCUPANCY>
        <<<grid, block>>>(
        reinterpret_cast<const hip_bfloat16(*)[K]>(A_bf16.data_ptr()),
        reinterpret_cast<const uint8_t(*)[K / 2]>((const void*)0),
        reinterpret_cast<const uint8_t(*)[K / 2]>(B.data_ptr()),
        reinterpret_cast<const uint8_t(*)[NUM_BLOCKS]>((const void*)0),
        reinterpret_cast<const uint8_t*>(Bs.data_ptr()),
        reinterpret_cast<hip_bfloat16(*)[N]>(C.data_ptr()));
}

template <int M, int N, int K,
          int IM, int IN, int IK,
          int WARPS_M = 1, int WARPS_N = 1, int K_SPLITS = 1,
          int K_HALF = K / 2, int NUM_BLOCKS = K / 32>
void launch_splitk(torch::Tensor A_bf16,
                   torch::Tensor B_data, torch::Tensor B_scale,
                   torch::Tensor C, bool profile) {
    // Pre-allocated scratch
    static torch::Tensor A_data_buf, A_scale_buf, ws_buf;
    static int local_gen = -1;
    if (local_gen != g_generation) {
        auto u8opts = torch::TensorOptions().dtype(torch::kUInt8).device(A_bf16.device());
        auto f32opts = torch::TensorOptions().dtype(torch::kFloat32).device(A_bf16.device());
        A_data_buf = torch::empty({M, K_HALF}, u8opts);
        A_scale_buf = torch::empty({NUM_BLOCKS * M}, u8opts);
        ws_buf = torch::empty({K_SPLITS * M * N}, f32opts);
        local_gen = g_generation;
    }

    static std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, PerfStats>>> perf_map;
    auto& stats = perf_map[M][N][K_HALF];
    stats.count++;
    bool do_profile = profile && (stats.count % PROFILE_INTERVAL == 0);

    hipEvent_t e0, e1, e2, e3;
    if (do_profile) {
        (void)hipEventCreate(&e0); (void)hipEventCreate(&e1); (void)hipEventCreate(&e2); (void)hipEventCreate(&e3);
        (void)hipEventRecord(e0);
    }

    // Launch warp-parallel quant kernel
    constexpr int QUANT_VPT = 2;  // values per thread
    constexpr int q_total = M * NUM_BLOCKS;
    constexpr int q_block = 64;
    constexpr int q_threads_per_group = 32 / QUANT_VPT;
    constexpr int q_groups_per_block = q_block / q_threads_per_group;
    constexpr int q_grid = (q_total + q_groups_per_block - 1) / q_groups_per_block;
    quant_a_kernel_warp_parallel<M, K, NUM_BLOCKS, QUANT_VPT>
        <<<dim3(q_grid), dim3(q_block)>>>(
        reinterpret_cast<const hip_bfloat16(*)[K]>(A_bf16.data_ptr()),
        reinterpret_cast<uint8_t(*)[K_HALF]>(A_data_buf.data_ptr()),
        reinterpret_cast<uint8_t(*)[NUM_BLOCKS]>(A_scale_buf.data_ptr()));

    if (do_profile) (void)hipEventRecord(e1);

    // Launch split-K GEMM
    dim3 grid((M + IM * WARPS_M - 1) / (IM * WARPS_M),
              (N + IN * WARPS_N - 1) / (IN * WARPS_N),
              K_SPLITS);
    dim3 block(64 * WARPS_M * WARPS_N);
    mfma_fp4_gemm_splitk<M,N,K,NUM_BLOCKS,IM,IN,IK,WARPS_M,WARPS_N,K_SPLITS>
        <<<grid, block>>>(
        reinterpret_cast<const uint8_t(*)[K_HALF]>(A_data_buf.data_ptr()),
        reinterpret_cast<const uint8_t(*)[K_HALF]>(B_data.data_ptr()),
        reinterpret_cast<const uint8_t*>(A_scale_buf.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_scale.data_ptr()),
        reinterpret_cast<float*>(ws_buf.data_ptr()));

    if (do_profile) (void)hipEventRecord(e2);

    // Launch reduction
    constexpr int r_total = M * N;
    constexpr int r_block = 256;
    constexpr int r_grid = (r_total + r_block - 1) / r_block;
    reduce_splitk_kernel<M, N, K_SPLITS>
        <<<dim3(r_grid), dim3(r_block)>>>(
        reinterpret_cast<const float*>(ws_buf.data_ptr()),
        reinterpret_cast<hip_bfloat16*>(C.data_ptr()));

    if (do_profile) {
        (void)hipEventRecord(e3);
        (void)hipEventSynchronize(e3);
        float d01, d12, d23;
        (void)hipEventElapsedTime(&d01, e0, e1);
        (void)hipEventElapsedTime(&d12, e1, e2);
        (void)hipEventElapsedTime(&d23, e2, e3);
        stats.t_quant += d01; stats.t_gemm += d12; stats.t_reduce += d23;
        int n = stats.count / PROFILE_INTERVAL;
        if (n < 5) {
            printf("[Split-K GEMM] m=%d n=%d k=%d | "
                "quant=%.1fus gemm=%.1fus reduce=%.1fus | "
                "total=%.1fus (avg over %d)\n",
                M, N, K,
                stats.t_quant/n*1000, stats.t_gemm/n*1000, stats.t_reduce/n*1000,
                (stats.t_quant+stats.t_gemm)/n*1000, n);
        }
        (void)hipEventDestroy(e0); (void)hipEventDestroy(e1); (void)hipEventDestroy(e2);  (void)hipEventDestroy(e3);
    }
}

// =====================================================================
// Split-K Fused: quantize A on-the-fly in split-K GEMM, no quant kernel
// =====================================================================
template <int M, int N, int K,
          int IM, int IN, int IK,
          int WARPS_M = 1, int WARPS_N = 1, int K_SPLITS = 1,
          int K_HALF = K / 2, int NUM_BLOCKS = K / 32>
void launch_splitk_fused(torch::Tensor A_bf16,
                         torch::Tensor B_data, torch::Tensor B_scale,
                         torch::Tensor C, bool profile) {
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(A_bf16.device());
    static torch::Tensor ws_buf;
    {
        int need = K_SPLITS * M * N;
        if (!ws_buf.defined() || ws_buf.numel() < need)
            ws_buf = torch::empty({need}, opts_f32);
    }

    static std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, PerfStats>>> perf_map;
    auto& stats = perf_map[M][N][K_HALF];
    stats.count++;
    bool do_profile = profile && (stats.count % PROFILE_INTERVAL == 0);

    hipEvent_t e0, e1, e2;
    if (do_profile) {
        (void)hipEventCreate(&e0); (void)hipEventCreate(&e1); (void)hipEventCreate(&e2);
        (void)hipEventRecord(e0);
    }

    // Launch fused split-K GEMM (quantize A on-the-fly)
    dim3 grid((M + IM * WARPS_M - 1) / (IM * WARPS_M),
              (N + IN * WARPS_N - 1) / (IN * WARPS_N),
              K_SPLITS);
    dim3 block(64 * WARPS_M * WARPS_N);
    mfma_fp4_gemm_splitk_fused<M,N,K,NUM_BLOCKS,IM,IN,IK,WARPS_M,WARPS_N,K_SPLITS>
        <<<grid, block>>>(
        reinterpret_cast<const hip_bfloat16*>(A_bf16.data_ptr()),
        reinterpret_cast<const uint8_t(*)[K_HALF]>(B_data.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_scale.data_ptr()),
        reinterpret_cast<float*>(ws_buf.data_ptr()));

    if (do_profile) (void)hipEventRecord(e1);

    // Launch reduction
    constexpr int r_total = M * N;
    constexpr int r_block = 256;
    constexpr int r_grid = (r_total + r_block - 1) / r_block;
    reduce_splitk_kernel<M, N, K_SPLITS>
        <<<dim3(r_grid), dim3(r_block)>>>(
        reinterpret_cast<const float*>(ws_buf.data_ptr()),
        reinterpret_cast<hip_bfloat16*>(C.data_ptr()));

    if (do_profile) {
        (void)hipEventRecord(e2);
        (void)hipEventSynchronize(e2);
        float d01, d12;
        (void)hipEventElapsedTime(&d01, e0, e1);
        (void)hipEventElapsedTime(&d12, e1, e2);
        stats.t_gemm += d01; stats.t_reduce += d12;
        int n = stats.count / PROFILE_INTERVAL;
        if (n < 5) {
            printf("[Split-K Fused] m=%d n=%d k=%d splits=%d | "
                "gemm+quant=%.1fus reduce=%.1fus | "
                "total=%.1fus (avg over %d)\n",
                M, N, K, K_SPLITS,
                stats.t_gemm/n*1000, stats.t_reduce/n*1000,
                (stats.t_gemm+stats.t_reduce)/n*1000, n);
        }
        (void)hipEventDestroy(e0); (void)hipEventDestroy(e1); (void)hipEventDestroy(e2);
    }
}

template <int M, int N, int K,
          int OM, int ON, int OK, int IM, int IN, int IK,
          int WTM = 1, int WTN = 1, int K_SPLITS = 1,
          int BUFFERS = 1, int OCCUPANCY = -1,
          int K_HALF = K / 2, int NUM_BLOCKS = K / 32>
void launch_tiled_splitk_fused(torch::Tensor A_bf16,
                                torch::Tensor B_data, torch::Tensor B_scale,
                                torch::Tensor C, bool profile) {
    static torch::Tensor ws_buf;
    static int local_gen = -1;
    if (local_gen != g_generation) {
        auto f32opts = torch::TensorOptions().dtype(torch::kFloat32).device(A_bf16.device());
        ws_buf = torch::empty({K_SPLITS * M * N}, f32opts);
        local_gen = g_generation;
    }

    constexpr int WARPS = (OM/(IM*WTM)) * (ON/(IN*WTN));

    static std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, PerfStats>>> perf_map;
    auto& stats = perf_map[M][N][K];
    stats.count++;
    bool do_profile = profile && (stats.count % PROFILE_INTERVAL == 0);

    hipEvent_t e0, e1, e2;
    if (do_profile) {
        (void)hipEventCreate(&e0); (void)hipEventCreate(&e1); (void)hipEventCreate(&e2);
        (void)hipEventRecord(e0);
    }

    dim3 grid((M+OM-1)/OM, (N+ON-1)/ON, K_SPLITS);
    dim3 block(WARPS * 64);
    mfma_fp4_gemm_tiled_splitk_fused<WARPS,M,N,K,OM,ON,OK,IM,IN,IK,WTM,WTN,K_SPLITS,BUFFERS,OCCUPANCY>
        <<<grid, block>>>(
        reinterpret_cast<const hip_bfloat16(*)[K]>(A_bf16.data_ptr()),
        reinterpret_cast<const uint8_t(*)[K_HALF]>(B_data.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_scale.data_ptr()),
        reinterpret_cast<float*>(ws_buf.data_ptr()));

    if (do_profile) (void)hipEventRecord(e1);

    // Separate reduction
    constexpr int r_total = M * N;
    constexpr int r_block = 256;
    constexpr int r_grid = (r_total + r_block - 1) / r_block;
    reduce_splitk_kernel<M, N, K_SPLITS>
        <<<dim3(r_grid), dim3(r_block)>>>(
        reinterpret_cast<const float*>(ws_buf.data_ptr()),
        reinterpret_cast<hip_bfloat16*>(C.data_ptr()));

    if (do_profile) {
        (void)hipEventRecord(e2);
        (void)hipEventSynchronize(e2);
        float d01, d12;
        (void)hipEventElapsedTime(&d01, e0, e1);
        (void)hipEventElapsedTime(&d12, e1, e2);
        stats.t_gemm += d01; stats.t_reduce += d12;
        int n = stats.count / PROFILE_INTERVAL;
        if (n < 5) {
            printf("[Tiled Split-K Fused] m=%d n=%d k=%d splits=%d | "
                "gemm+quant=%.1fus reduce=%.1fus | "
                "total=%.1fus (avg over %d)\n",
                M, N, K, K_SPLITS,
                stats.t_gemm/n*1000, stats.t_reduce/n*1000,
                (stats.t_gemm+stats.t_reduce)/n*1000, n);
        }
        (void)hipEventDestroy(e0); (void)hipEventDestroy(e1); (void)hipEventDestroy(e2);
    }
}

template <int M, int N, int K,
          int OM, int ON, int OK, int IM, int IN, int IK,
          int WTM = 1, int WTN = 1, int K_SPLITS = 1,
          int BUFFERS = 1, int OCCUPANCY = -1,
          int K_HALF = K / 2, int NUM_BLOCKS = K / 32>
void launch_tiled_splitk_coop(torch::Tensor A_bf16,
                               torch::Tensor B_data, torch::Tensor B_scale,
                               torch::Tensor C, bool profile) {
    static torch::Tensor A_data_buf, A_scale_buf, ws_buf, quant_ctr_buf, split_ready_buf;
    static int local_gen = -1;
    if (local_gen != g_generation) {
        auto u8opts = torch::TensorOptions().dtype(torch::kUInt8).device(A_bf16.device());
        auto f32opts = torch::TensorOptions().dtype(torch::kFloat32).device(A_bf16.device());
        auto i32opts = torch::TensorOptions().dtype(torch::kInt32).device(A_bf16.device());
        A_data_buf = torch::empty({M, K_HALF}, u8opts);
        A_scale_buf = torch::empty({NUM_BLOCKS * M}, u8opts);
        ws_buf = torch::empty({K_SPLITS * M * N}, f32opts);
        quant_ctr_buf = torch::zeros({1}, i32opts);
        split_ready_buf = torch::zeros({K_SPLITS}, i32opts);  // per-split completion counters
        local_gen = g_generation;
    }

    // Zero counters before each launch
    quant_ctr_buf.zero_();
    split_ready_buf.zero_();

    constexpr int WARPS = (OM/(IM*WTM)) * (ON/(IN*WTN));

    static std::unordered_map<int, std::unordered_map<int, std::unordered_map<int, PerfStats>>> perf_map;
    auto& stats = perf_map[M][N][K];
    stats.count++;
    bool do_profile = profile && (stats.count % PROFILE_INTERVAL == 0);

    hipEvent_t e0, e1, e2;
    if (do_profile) {
        (void)hipEventCreate(&e0); (void)hipEventCreate(&e1); (void)hipEventCreate(&e2);
        (void)hipEventRecord(e0);
    }

    int* quant_counter = reinterpret_cast<int*>(quant_ctr_buf.data_ptr());
    int* split_ready = reinterpret_cast<int*>(split_ready_buf.data_ptr());

    dim3 grid((M+OM-1)/OM, (N+ON-1)/ON, K_SPLITS);
    dim3 block(WARPS * 64);

    // Use cooperative kernel launch for grid-wide sync
    auto A_bf16_ptr = reinterpret_cast<const hip_bfloat16*>(A_bf16.data_ptr());
    auto A_data_ptr = reinterpret_cast<uint8_t*>(A_data_buf.data_ptr());
    auto A_scale_ptr = reinterpret_cast<uint8_t*>(A_scale_buf.data_ptr());
    auto B_data_ptr = reinterpret_cast<const uint8_t*>(B_data.data_ptr());
    auto B_scale_ptr = reinterpret_cast<const uint8_t*>(B_scale.data_ptr());
    auto ws_ptr = reinterpret_cast<float*>(ws_buf.data_ptr());

    void* args[] = {
        &A_bf16_ptr, &A_data_ptr, &A_scale_ptr,
        &B_data_ptr, &B_scale_ptr, &ws_ptr,
        &quant_counter, &split_ready
    };
    (void)hipLaunchCooperativeKernel(
        (const void*)mfma_fp4_gemm_tiled_splitk_coop<WARPS,M,N,K,OM,ON,OK,IM,IN,IK,WTM,WTN,K_SPLITS,BUFFERS,OCCUPANCY>,
        grid, block, args, 0, 0);

    if (do_profile) (void)hipEventRecord(e1);

    // Reduction
    constexpr int r_total = M * N;
    constexpr int r_block = 256;
    constexpr int r_grid = (r_total + r_block - 1) / r_block;
    reduce_splitk_kernel<M, N, K_SPLITS>
        <<<dim3(r_grid), dim3(r_block)>>>(
        reinterpret_cast<const float*>(ws_buf.data_ptr()),
        reinterpret_cast<hip_bfloat16*>(C.data_ptr()));

    if (do_profile) {
        (void)hipEventRecord(e2);
        (void)hipEventSynchronize(e2);
        float d01, d12;
        (void)hipEventElapsedTime(&d01, e0, e1);
        (void)hipEventElapsedTime(&d12, e1, e2);
        stats.t_gemm += d01; stats.t_reduce += d12;
        int n = stats.count / PROFILE_INTERVAL;
        if (n < 5) {
            printf("[Tiled Split-K Coop] m=%d n=%d k=%d splits=%d | "
                "gemm+quant=%.1fus reduce=%.1fus | "
                "total=%.1fus (avg over %d)\n",
                M, N, K, K_SPLITS,
                stats.t_gemm/n*1000, stats.t_reduce/n*1000,
                (stats.t_gemm+stats.t_reduce)/n*1000, n);
        }
        (void)hipEventDestroy(e0); (void)hipEventDestroy(e1); (void)hipEventDestroy(e2);
    }
}

void mfma_gemm(
    torch::Tensor A_data, torch::Tensor B_data,
    torch::Tensor B_scale, torch::Tensor C,
    int M, int N, int K, bool profile
) {
// Dispatch macro - add WM, WN
#define CS16(m,n,k) \
    if(M==m&&N==n&&K==k){return launch_coop_simple<m,n,k,16,16,128>(A_data,B_data,B_scale,C, profile);}
#define CS32(m,n,k) \
    if(M==m&&N==n&&K==k){return launch_coop_simple<m,n,k,32,32,64>(A_data,B_data,B_scale,C, profile);}
#define S32(m,n,k,wm,wn) \
    if(M==m&&N==n&&K==k){return launch_simple<m,n,k,32,32,64,wm,wn>(A_data,B_data,B_scale,C, profile);}
#define S16(m,n,k,wm,wn) \
    if(M==m&&N==n&&K==k){return launch_simple<m,n,k,16,16,128,wm,wn>(A_data,B_data,B_scale,C, profile);}
#define T(m,n,k,om,on,ok,im,in,ik,wtm,wtn,bufs) \
    if(M==m&&N==n&&K==k){return launch_tiled<m,n,k,om,on,ok,im,in,ik,wtm,wtn,bufs>(A_data,B_data,B_scale,C, profile);}
#define F(m,n,k,om,on,ok,im,in,ik,wtm,wtn,bufs) \
    if(M==m&&N==n&&K==k){return launch_tiled_fused<m,n,k,om,on,ok,im,in,ik,wtm,wtn,bufs>(A_data,B_data,B_scale,C);}
#define F2(m,n,k,om,on,ok,im,in,ik,wtm,wtn,bufs,occ) \
    if(M==m&&N==n&&K==k){return launch_tiled_fused<m,n,k,om,on,ok,im,in,ik,wtm,wtn,bufs,occ>(A_data,B_data,B_scale,C);}
#define SK(m,n,k,im,in,ik,wm,wn,ksplits) \
    if(M==m&&N==n&&K==k){return launch_splitk<m,n,k,im,in,ik,wm,wn,ksplits>(A_data,B_data,B_scale,C, profile);}
#define SKF(m,n,k,im,in,ik,wm,wn,ksplits) \
    if(M==m&&N==n&&K==k){return launch_splitk_fused<m,n,k,im,in,ik,wm,wn,ksplits>(A_data,B_data,B_scale,C, profile);}
#define SF32(m,n,k,wm,wn) \
    if(M==m&&N==n&&K==k){return launch_simple_fused<m,n,k,32,32,64,wm,wn>(A_data,B_data,B_scale,C, profile);}
#define SF16(m,n,k,wm,wn) \
    if(M==m&&N==n&&K==k){return launch_simple_fused<m,n,k,16,16,128,wm,wn>(A_data,B_data,B_scale,C, profile);}
#define TSK(m,n,k,om,on,ok,im,in,ik,wtm,wtn,ksplits,bufs,occ) \
    if(M==m&&N==n&&K==k){return launch_tiled_splitk_fused<m,n,k,om,on,ok,im,in,ik,wtm,wtn,ksplits,bufs,occ>(A_data,B_data,B_scale,C, profile);}
#define TSKC(m,n,k,om,on,ok,im,in,ik,wtm,wtn,ksplits,bufs,occ) \
    if(M==m&&N==n&&K==k){return launch_tiled_splitk_coop<m,n,k,om,on,ok,im,in,ik,wtm,wtn,ksplits,bufs,occ>(A_data,B_data,B_scale,C, profile);}

    // Cooperative Simple - hipLaunchCooperativeKernel too slow (~30-40us dispatch)
    //CS16(4,  2880, 512)
    //CS16(32, 4096, 512)
    //CS32(32, 2880, 512)
    //CS16(64, 7168, 2048)
    //CS32(256, 3072, 1536)

    // Tiled
    T(64,  3072, 1536,  32,32,1536, 32,32,64, 1,1,1)
    T(256, 2880, 512,  128,128,512, 32,32,64, 1,1,1)

    // Simple 32x32x64
    //SF16(32, 4096, 512, 1, 1) //--> 8.88 BEST
    //ITER2: SF32(32, 4096, 512, 1, 1) --> 14.5
    //ITER3: T(32, 4096, 512, 32, 32, 128, 32, 32, 64, 1, 1, 1) --> 21.0
    SF16(32, 4096, 512, 1, 2) //--> 8.48 BEST
    //ITER6: SF16(32, 4096, 512, 2, 1) --> 8.74
    //ITER7: SF16(32, 4096, 512, 2, 2) --> 9.05
    //ITER8: S16(32, 4096, 512, 1, 1) --> 10.2
    //ITER9: SF16(32, 4096, 512, 1, 4) --> 9.18
    //ITER11: SF16(32, 4096, 512, 1, 3) --> 9.19
    //S32(32, 2880, 512, 1, 1) //--> 11.9
    //ITER1: T(32, 2880, 512, 32, 32, 128, 32, 32, 64, 1, 1, 1) --> 21.1
    //SF16(32, 2880, 512, 1, 1) //--> 8.74 NEW BEST
    SF16(32, 2880, 512, 1, 2) //--> 8.47 BEST
    //ITER5: SF16(32, 2880, 512, 2, 1) --> 8.55
    //ITER6: SF16(32, 2880, 512, 2, 2) --> 8.98
    //ITER8: S16(32, 2880, 512, 1, 1) --> 9.76
    //ITER9: SF16(32, 2880, 512, 4, 1) --> 9.23
    //ITER11: SF16(32, 2880, 512, 1, 3) --> 9.11
    //S16(256, 3072, 1536, 1, 1) --> 21.1
    //F2(256, 3072, 1536, 32, 32, 1536, 32, 32, 64, 1, 1, 1, 3) --> 59.8
    //F2(256, 3072, 1536, 64, 64, 768, 32, 32, 64, 1, 1, 2, 1) --> 38.3
    //F2(256, 3072, 1536, 64, 64, 512, 32, 32, 64, 1, 1, 2, 2) --> 31.9
    //F2(256, 3072, 1536, 32, 32, 768, 32, 32, 64, 1, 1, 2, 3) --> 69.6
    //TSK(256, 3072, 1536, 32, 32, 768, 32, 32, 64, 1, 1, 2, 1, 1) --> 63.1
    //F(256, 3072, 1536, 16, 32, 1536, 16, 16, 128, 1, 1, 1) --> 47.3
    //S32(256, 3072, 1536, 1, 2) --> 21.6
    //T(256, 3072, 1536, 32, 64, 1536, 32, 32, 64, 1, 1, 1) --> 34.3
    //T(256, 3072, 1536, 64, 64, 768, 32, 32, 64, 1, 1, 2) --> 39.2
    //T(256, 3072, 1536, 64, 64, 128, 32, 32, 64, 2, 2, 1) --> 70.8
    //T(256, 3072, 1536, 64, 64, 256, 32, 32, 64, 2, 2, 1) --> 84.2
    //F(256, 3072, 1536, 64, 64, 128, 32, 32, 64, 2, 2, 1) --> 95.8
    //T(256, 3072, 1536, 128, 64, 128, 32, 32, 64, 4, 2, 1) --> 125
    //T(256, 3072, 1536, 64, 64, 512, 32, 32, 64, 2, 2, 1) --> 92.3
    //S32(256, 3072, 1536, 1, 1)
    //T(256, 3072, 1536, 32, 32, 1536, 32, 32, 64, 1, 1, 1) -> 26.8
    //T(256, 3072, 1536, 32, 32, 128, 32, 32, 64, 1, 1, 1)
    //T(256, 3072, 1536, 64, 64, 128, 32, 32, 64, 1, 1, 1) //--> 17.1 BEST
    //T(256, 3072, 1536, 64, 64, 256, 32, 32, 64, 1, 1, 1) --> 35.4
    //T(256, 3072, 1536, 64, 64, 512, 32, 32, 64, 1, 1, 1) --> 41.1
    //T(256, 3072, 1536, 64, 64, 768, 32, 32, 64, 1, 1, 1) --> 36.6
    //T(256, 3072, 1536, 64, 64, 128, 32, 32, 64, 1, 1, 2) --> 18.6
    //T(256, 3072, 1536, 64, 32, 128, 32, 32, 64, 1, 1, 1) --> 33.7
    //T(256, 3072, 1536, 64, 96, 128, 32, 32, 64, 1, 1, 1) --> 18.3
    //ITER1: T(256, 3072, 1536, 128, 128, 128, 32, 32, 64, 1, 1, 1) --> INCORRECT
    //ITER2: T(256, 3072, 1536, 64, 128, 128, 32, 32, 64, 1, 1, 1) --> 18.8
    //ITER3: T(256, 3072, 1536, 128, 64, 128, 32, 32, 64, 1, 1, 1) --> INCORRECT
    //ITER4: T(256, 3072, 1536, 96, 64, 128, 32, 32, 64, 1, 1, 1) --> INCORRECT
    //T(256, 3072, 1536, 64, 64, 128, 32, 32, 64, 1, 1, 1) //--> 17.1 BEST
    //ITER5: TSK(256, 3072, 1536, 64, 64, 128, 32, 32, 64, 1, 1, 2, 1, -1) --> 21.6
    //ITER6: T(256, 3072, 1536, 64, 64, 128, 32, 32, 64, 1, 1, 2) --> 18.6
    //ITER7: T(256, 3072, 1536, 64, 64, 128, 16, 16, 128, 1, 1, 1) --> INCORRECT
    //ITER8: T(256, 3072, 1536, 64, 64, 192, 32, 32, 64, 1, 1, 1) --> INCORRECT
    //ITER9: T(256, 3072, 1536, 64, 64, 128, 32, 32, 64, 2, 1, 1) --> 64.5
    //ITER10: T(256, 3072, 1536, 64, 64, 128, 32, 32, 64, 1, 2, 1) --> 76.5
    //ITER11: T(256, 3072, 1536, 64, 64, 64, 32, 32, 64, 1, 1, 1) --> INCORRECT
    //ITER12: TSK(256, 3072, 1536, 64, 64, 128, 32, 32, 64, 1, 1, 3, 1, -1) --> 21.6
    //ITER13: F2(256, 3072, 1536, 64, 64, 128, 32, 32, 64, 1, 1, 1, 1) --> 24.2
    //ITER14: TSK(256, 3072, 1536, 64, 64, 128, 32, 32, 64, 1, 1, 4, 1, -1) --> 21.5
    T(256, 3072, 1536, 64, 64, 128, 32, 32, 64, 1, 1, 1) //--> 17.1 BEST

    // Simple 16x16x128
    SF16(4,  2880, 512, 1, 1) //--> 8.32 BEST
    //ITER2: SF32(4, 2880, 512, 1, 1) --> 13.7
    //ITER3: T(4, 2880, 512, 16, 32, 128, 16, 16, 128, 1, 1, 1) --> 11.1
    //ITER5: SF16(4, 2880, 512, 1, 2) --> 8.53
    //ITER6: SF16(4, 2880, 512, 2, 1) --> 8.43
    //ITER8: SF16(4, 2880, 512, 2, 2) --> 9.03
    //ITER9: S16(4, 2880, 512, 1, 1) --> 9.65
    SF16(8,  2112, 7168, 1, 1) //--> unchanged
    SF16(16, 3072, 1536, 1, 1) //--> unchanged
    //S32(64, 7168, 2048, 1, 1) --> 21
    //F2(64, 7168, 2048, 16, 16, 1024, 16, 16, 128, 1, 1, 2, 4) --> 80.1
    //TSK(64, 7168, 2048, 16, 32, 1024, 16, 16, 128, 1, 1, 2, 1, 1) --> 34.4
    //T(64, 7168, 2048, 16, 64, 2048, 16, 16, 128, 1, 1, 1) --> 37.9
    //S16(64, 7168, 2048, 1, 2) --> 20.4
    //SK(64, 7168, 2048, 16, 16, 128, 1, 1, 2) --> 22.8
    //SKF(64, 7168, 2048, 16, 16, 128, 1, 1, 2) --> 30.2
    //T(64, 7168, 2048, 64, 64, 128, 16, 16, 128, 2, 2, 1) --> 23.0
    //TSK(64, 7168, 2048, 64, 64, 128, 16, 16, 128, 2, 2, 2, 1, -1) --> 23.9
    //TSK(64, 7168, 2048, 64, 64, 128, 16, 16, 128, 2, 2, 4, 1, -1) //--> 21.6
    //ITER1: TSK(64, 7168, 2048, 64, 64, 128, 16, 16, 128, 2, 2, 8, 1, -1) --> 23.4
    //ITER2: TSK(64, 7168, 2048, 32, 64, 128, 16, 16, 128, 1, 2, 4, 1, -1) --> 20.3 NEW BEST
    //ITER3: TSK(64, 7168, 2048, 32, 64, 128, 16, 16, 128, 1, 2, 8, 1, -1) --> 23.6
    //ITER4: TSK(64, 7168, 2048, 32, 64, 128, 16, 16, 128, 1, 2, 2, 1, -1) --> 21.8
    //TSK(64, 7168, 2048, 32, 64, 128, 16, 16, 128, 1, 2, 4, 1, -1) //--> 20.3 BEST
    //ITER5: TSK(64, 7168, 2048, 32, 64, 256, 16, 16, 128, 1, 2, 4, 1, -1) --> 22.6
    //ITER6: TSK(64, 7168, 2048, 32, 64, 128, 16, 16, 128, 2, 2, 4, 1, -1) --> 26.8
    //ITER7: TSK(64, 7168, 2048, 16, 64, 128, 16, 16, 128, 1, 2, 4, 1, -1) --> 26.3
    //ITER8: TSK(64, 7168, 2048, 32, 64, 128, 16, 16, 128, 1, 2, 4, 1, 1) --> 20.3
    //ITER9: TSK(64, 7168, 2048, 32, 64, 128, 16, 16, 128, 1, 2, 4, 2, -1) --> 19.8 NEW BEST
    //ITER11: TSK(64, 7168, 2048, 32, 64, 128, 16, 16, 128, 1, 2, 3, 2, -1) --> 20.8
    //ITER12: TSK(64, 7168, 2048, 32, 64, 128, 16, 16, 128, 1, 2, 6, 2, -1) --> 21.3
    //ITER13: TSK(64, 7168, 2048, 64, 64, 128, 16, 16, 128, 2, 2, 4, 2, -1) --> 21.7
    //ITER14: T(64, 7168, 2048, 32, 64, 128, 16, 16, 128, 1, 2, 2) --> 20.8
    TSK(64, 7168, 2048, 32, 64, 128, 16, 16, 128, 1, 2, 4, 2, -1) //--> 19.8 BEST
    //ITER10: try TSK(32,64,128, 1,2, 3, 2, -1) split-K=3
    //TSK(64, 7168, 2048, 32, 64, 128, 16, 16, 128, 1, 2, 3, 2, -1) //ITER10 TBD
    //T(64, 7168, 2048, 64, 64, 128, 32, 32, 64, 2, 1, 1) --> 70.0
    //T(64, 7168, 2048, 64, 128, 128, 16, 16, 128, 2, 2, 2) --> 25.7
    //S16(64, 7168, 2048, 1, 1)

    // Tiled fused
    //F2(32, 4096, 512, 32, 32, 512, 16, 16, 128, 1, 1, 1, 1)

    // Split-K
    //SKF(16, 2112, 7168, 16,16,128, 1,1, 28)
    //TSK(16, 2112, 7168, 16,64,256, 16,16,128, 1,1, 28, 1, 1) //--> 13.0 BEST
    //ITER2: TSK(16, 2112, 7168, 16, 64, 256, 16, 16, 128, 1, 1, 14, 1, 1) --> 15.8
    //ITER3: TSK(16, 2112, 7168, 16, 32, 128, 16, 16, 128, 1, 1, 28, 1, -1) --> 13.7
    //ITER4: TSK(16, 2112, 7168, 16, 64, 128, 16, 16, 128, 1, 1, 28, 1, -1) --> 13.2
    //ITER6: TSK(16, 2112, 7168, 16, 64, 256, 16, 16, 128, 1, 2, 28, 1, -1) --> 15.1
    //ITER8: TSK(16, 2112, 7168, 16, 64, 256, 16, 16, 128, 1, 1, 28, 2, -1) --> 12.9
    //ITER9: TSK(16, 2112, 7168, 16, 64, 256, 16, 16, 128, 1, 1, 56, 1, -1) --> INCORRECT
    //ITER10: TSK(16, 2112, 7168, 16, 64, 128, 16, 16, 128, 1, 1, 28, 2, -1) --> 13.4
    //ITER11: TSK(16, 2112, 7168, 16, 64, 256, 16, 16, 128, 1, 1, 28, 2, 1) --> 13.2
    //ITER12: TSK(16, 2112, 7168, 16, 32, 256, 16, 16, 128, 1, 1, 28, 1, -1) --> 13.8
    //ITER13: TSK(16, 2112, 7168, 16, 64, 512, 16, 16, 128, 1, 1, 14, 1, -1) --> 14.6
    //ITER14: TSK(16, 2112, 7168, 16, 64, 256, 16, 16, 128, 1, 1, 7, 1, -1) --> 20.8
    TSK(16, 2112, 7168, 16, 64, 256, 16, 16, 128, 1, 1, 28, 1, -1) //--> 12.9 BEST
    //ITER7: try bufs=2 on winning config
    //TSK(16, 2112, 7168, 16, 64, 256, 16, 16, 128, 1, 1, 28, 2, -1) //ITER7 TBD
    //TSKC(16, 2112, 7168, 16, 64, 256, 16, 16, 128, 1, 1, 14, 1, 2)
    //TSK(64, 7168, 2048, 16,64,512, 16,16,128, 1,1, 4, 1, 1)
    //TSK(256, 3072, 1536, 32,32,768, 16,16,128, 1,1, 2, 1, 1)
#undef S32
#undef S16
#undef T
#undef SK

    TORCH_CHECK(false, "No template for M=", M, " N=", N, " K=", K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mfma_gemm", &mfma_gemm);
    m.def("reset_buffers", &reset_buffers);
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
            name='mxfp4_mfma_v8', cpp_sources='', cuda_sources=[src],
            extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '--offload-arch=gfx950', '-DHIP_ENABLE_EXTRA_WARP_SYNC_TYPES'],
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


# def custom_kernel(data: input_t) -> output_t:
#     global HAS_HIP_KERNEL, _hip_module

#     A, B, B_q, B_shuffle, B_scale_sh = data
#     A = A.contiguous()
#     m, k = A.shape
#     n, _ = B.shape

#     B_data = B_q.view(torch.uint8)
#     B_sc = B_scale_sh.view(torch.uint8)
#     K_half = k // 2
#     num_blocks = (k + 31) // 32
#     scaleN = ((num_blocks + 7) // 8) * 8

#     if not HAS_HIP_KERNEL:
#         return

#     if not hasattr(custom_kernel, '_graph_cache'):
#         custom_kernel._graph_cache = {}

#     key = (m, n, k)

#     if key not in custom_kernel._graph_cache:
#         A_buf = torch.empty_like(A)
#         B_data_buf = torch.empty_like(B_data)
#         B_sc_buf = torch.empty_like(B_sc)
#         C_buf = torch.empty((m, n), dtype=torch.bfloat16, device=A.device)

#         A_buf.copy_(A)
#         B_data_buf.copy_(B_data)
#         B_sc_buf.copy_(B_sc)

#         _hip_module.mfma_gemm(
#             A_buf, B_data_buf, B_sc_buf, C_buf,
#             m, n, k, K_half, num_blocks, scaleN)

#         g = torch.cuda.CUDAGraph()
#         with torch.cuda.graph(g):
#             _hip_module.mfma_gemm(
#                 A_buf, B_data_buf, B_sc_buf, C_buf,
#                 m, n, k, K_half, num_blocks, scaleN)

#         custom_kernel._graph_cache[key] = (g, A_buf, B_data_buf, B_sc_buf, C_buf)

#     g, A_buf, B_data_buf, B_sc_buf, C_buf = custom_kernel._graph_cache[key]
#     A_buf.copy_(A)
#     B_data_buf.copy_(B_data)
#     B_sc_buf.copy_(B_sc)
#     g.replay()
#     return C_buf

def custom_kernel(data: input_t) -> output_t:
    global HAS_HIP_KERNEL, _hip_module

    PROFILE = False

    A, B, B_q, B_shuffle, B_scale_sh = data
    A = A.contiguous()
    m, k = A.shape
    n, _ = B.shape

    B_data = B_q.view(torch.uint8)
    B_sc = B_scale_sh.view(torch.uint8)

    if not HAS_HIP_KERNEL:
        return

    C = torch.empty((m, n), dtype=torch.bfloat16, device=A.device)
    _hip_module.mfma_gemm(A, B_data, B_sc, C, m, n, k, PROFILE)
    return C
