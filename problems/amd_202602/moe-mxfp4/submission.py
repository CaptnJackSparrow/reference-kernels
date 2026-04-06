"""
MXFP4 Mixture-of-Experts (MoE) Fused Kernel — Custom HIP MFMA Implementation.

Implements a DeepSeek-R1 style MoE forward pass on AMD MI355X:
  - Custom HIP kernel: MFMA 16x16x128 FP4xFP4 matrix multiply
  - On-the-fly BF16→MXFP4 activation quantization via hardware intrinsics
  - Per-expert GEMM with raw (un-shuffled) FP4 weights + E8M0 block scales
  - PyTorch-side MoE token routing, SwiGLU activation, weighted reduction

Pipeline per token per assigned expert:
  Stage 1: hidden_states → quant_mxfp4 → FP4xFP4 GEMM(gate_up_weight) → SwiGLU
  Stage 2: intermediate  → quant_mxfp4 → FP4xFP4 GEMM(down_weight)    → weighted sum
"""
import torch, os, time, functools, triton
import triton.language as tl
import torch.nn.functional as F
from task import input_t, output_t
from aiter import dtypes

_mod = None  # Lazy-compiled HIP module handle

# ═══════════════════════════════════════════════════════════════════════
# HIP Kernel Source — MXFP4 FP4xFP4 GEMM with on-the-fly A quantization
# ═══════════════════════════════════════════════════════════════════════
HIP_SRC = r'''
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
// ── Type aliases for MFMA instruction operands ──────────────────────
constexpr int FMT_FP4 = 4;                                    // MFMA data format flag for FP4
typedef float __attribute__((ext_vector_type(4))) f4;          // 4-float MFMA accumulator (16x16 tile output)
typedef int __attribute__((ext_vector_type(8))) i8v;           // 8x int32 MFMA source operand (holds 64 FP4 values)
typedef uint32_t __attribute__((ext_vector_type(4))) u128;     // 128-bit load type for coalesced FP4 reads

// ── Quantized MX block: 32 FP4 values packed into 4 uint32 + E8M0 scale ──
// d[4]: 16 packed FP4 bytes = 32 FP4 E2M1 values (2 per byte)
// e:    E8M0 exponent (block-level power-of-2 scale factor)
struct QBlk { uint32_t d[4]; uint8_t e; };

// ── Pack 4 FP4 byte-pairs into one uint32 using hardware perm instruction ──
__device__ __forceinline__ uint32_t pack_fp4_to_u32(uint8_t p0, uint8_t p1, uint8_t p2, uint8_t p3) {
    uint32_t srcA = (uint32_t)p0 | ((uint32_t)p1 << 8);
    uint32_t srcB = (uint32_t)p2 | ((uint32_t)p3 << 8);
    return __builtin_amdgcn_perm(srcB, srcA, 0x05040100);
}

// ── E8M0 scale computation using BF16 rounding + LUT (matches mxfp4-mm) ──
// Uses pure integer/LUT operations for exact E8M0 computation.
struct ScaleInfo { uint8_t e8m0; float quant_scale; };

// E8M0 LUT: raw_exp -> e8m0 = clamp(raw_exp - 2, 0, 254)
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

// QUANT_SCALE_RECIP_LUT: e8m0 -> IEEE float bits of quant_scale = 2^(e8m0 - 127)
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

// BF16-domain E8M0 scale computation (matches working mxfp4-mm kernel exactly)
__device__ __forceinline__ ScaleInfo compute_e8m0_scale(hip_bfloat16 amax_bf16) {
    ScaleInfo s;
    uint16_t amax_bits = amax_bf16.data;
    if (amax_bits == 0) {
        s.e8m0 = 0;
        s.quant_scale = 0.0f;
    } else {
        uint16_t rounded = (amax_bits + 0x0020u) & 0xFF80u;
        int raw_exp = (int)((rounded >> 7) & 0xFF);
        s.e8m0 = E8M0_LUT[raw_exp];
        s.quant_scale = __uint_as_float(QUANT_SCALE_RECIP_LUT[s.e8m0]);
    }
    return s;
}

// ── Hardware FP4 quantization: 2 BF16 values → 1 packed FP4 byte ────
// Uses AMD gfx950 intrinsic for exact FP4 rounding matching reference
__device__ __forceinline__ uint8_t qfp4_hw(hip_bfloat16 v0, hip_bfloat16 v1, float qs) {
    using b2 = uint16_t __attribute__((ext_vector_type(2)));
    b2 p = {v0.data, v1.data};
    union { uint32_t u; uint8_t b[4]; } c = {0};
    c.u = __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16(c.u, p, qs, 0);
    return c.b[0];
}

// ── Hardware FP4 quantization from FP32: 2 floats → 1 packed FP4 byte ──
__device__ __forceinline__ uint8_t qfp4_hw_f32(float v0, float v1, float qs) {
    union { uint32_t u; uint8_t b[4]; } c = {0};
    c.u = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(c.u, v0, v1, qs, 0);
    return c.b[0];
}

// ── BF16 quantize: BF16-domain amax + LUT scale + hardware FP4 intrinsic ──
// Matches working mxfp4-mm kernel: BF16 amax, integer LUT for E8M0.
// BF16 input variant (for Stage 1: bf16 activations)
__device__ __forceinline__ QBlk qblock(const hip_bfloat16* src) {
    // 1. Find amax across 32 BF16 values in BF16 precision (no FP32 conversion)
    uint16_t amax_bits = 0;
    for (int i = 0; i < 32; i++) {
        uint16_t bits = *reinterpret_cast<const uint16_t*>(&src[i]) & 0x7FFF;
        amax_bits = (bits > amax_bits) ? bits : amax_bits;
    }
    hip_bfloat16 amax_bf16 = *reinterpret_cast<const hip_bfloat16*>(&amax_bits);
    // 2. Compute E8M0 scale using BF16 rounding + LUT (matches mxfp4-mm)
    ScaleInfo sc = compute_e8m0_scale(amax_bf16);
    // 3. Quantize pairs using hardware intrinsic (exact FP4 rounding)
    QBlk q;
    for (int j = 0; j < 4; j++) {
        int base = j * 8;
        uint8_t p0 = qfp4_hw(src[base], src[base+1], sc.quant_scale);
        uint8_t p1 = qfp4_hw(src[base+2], src[base+3], sc.quant_scale);
        uint8_t p2 = qfp4_hw(src[base+4], src[base+5], sc.quant_scale);
        uint8_t p3 = qfp4_hw(src[base+6], src[base+7], sc.quant_scale);
        q.d[j] = pack_fp4_to_u32(p0, p1, p2, p3);
    }
    q.e = sc.e8m0;
    return q;
}

// ── F32 input variant (for Stage 2: f32 SwiGLU output) ──────────────
// Finds amax in FP32, converts to BF16 for E8M0 scale, then quantizes.
__device__ __forceinline__ QBlk qblock_f32(const float* src) {
    // 1. Find amax in f32
    float mx = 0.f;
    for (int i = 0; i < 32; i++) {
        float a = src[i] < 0.f ? -src[i] : src[i];
        mx = (a > mx) ? a : mx;
    }
    // 2. Convert amax to BF16 and compute E8M0 scale using BF16 rounding + LUT
    hip_bfloat16 amax_bf16 = (hip_bfloat16)mx;
    ScaleInfo sc = compute_e8m0_scale(amax_bf16);
    // 3. Quantize pairs using f32 hardware intrinsic
    QBlk q;
    for (int j = 0; j < 4; j++) {
        int base = j * 8;
        uint8_t p0 = qfp4_hw_f32(src[base], src[base+1], sc.quant_scale);
        uint8_t p1 = qfp4_hw_f32(src[base+2], src[base+3], sc.quant_scale);
        uint8_t p2 = qfp4_hw_f32(src[base+4], src[base+5], sc.quant_scale);
        uint8_t p3 = qfp4_hw_f32(src[base+6], src[base+7], sc.quant_scale);
        q.d[j] = pack_fp4_to_u32(p0, p1, p2, p3);
    }
    q.e = sc.e8m0;
    return q;
}

// ── Broadcast E8M0 exponent to all 4 bytes of int32 ────────────────
// MFMA scale parameter needs same E8M0 in all bytes: 0x7E → 0x7E7E7E7E
__device__ __forceinline__ int32_t bcast(uint8_t e) { return (int32_t)e * 0x01010101; }

// ── Shuffled scale offset (matches CK's e8m0_shuffle layout) ────────
// Maps logical (row, col) in [N, K/32] scale matrix → flat byte offset
// in the shuffled buffer. Uses bit-field extraction for the (16,16) tile
// layout. SN = number of scale columns = K/32.
template <int SN>
__device__ __forceinline__ int sh_off(int row, int col) {
    int t0 = __builtin_amdgcn_ubfe(row, 4, 1);
    int t1 = __builtin_amdgcn_ubfe(col, 2, 1) << 1;
    int t2 = __builtin_amdgcn_ubfe(row, 0, 4) << 2;
    int t3 = __builtin_amdgcn_ubfe(col, 0, 2) << 6;
    int t4 = (col >> 3) << 8;
    int t5 = (row >> 5) * (32 * SN);
    return t0+t1+t2+t3+t4+t5;
}

// ── GPU-side E8M0 scale shuffle kernel ──────────────────────────────
// Rearranges row-major [rows, SN] scale → shuffled flat buffer using sh_off.
template <int SN>
__global__ void shuffle_e8m0(const uint8_t* __restrict__ raw, uint8_t* __restrict__ out, int rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * SN;
    if (idx >= total) return;
    int row = idx / SN, col = idx % SN;
    out[sh_off<SN>(row, col)] = raw[idx];
}

// ═══════════════════════════════════════════════════════════════════════
// Separate MXFP4 quantization kernels (pre-processing step)
// Matches the reference's separate-quantize-then-GEMM pipeline.
// ═══════════════════════════════════════════════════════════════════════

// Quantize BF16 activations to MXFP4: [M, K] → fp4x2 [M, K/2] + e8m0 [M, K/32]
template <int K, int NB = K/32, int KH = K/2>
__global__ void mxfp4_quant_bf16(
    const hip_bfloat16 A[][K],
    uint8_t A_fp4[][KH],        // output: packed fp4x2
    uint8_t A_scale[][NB],      // output: e8m0 block scales
    int actual_m
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = actual_m * NB;
    if (idx >= total) return;
    int row = idx / NB;
    int blk = idx % NB;
    QBlk qb = qblock(&A[row][blk * 32]);
    // Store packed FP4 bytes (16 bytes = 4 uint32)
    *reinterpret_cast<u128*>(&A_fp4[row][blk * 16]) =
        *reinterpret_cast<u128*>(&qb.d[0]);
    A_scale[row][blk] = qb.e;
}

// Quantize F32 activations to MXFP4: [M, K] → fp4x2 [M, K/2] + e8m0 [M, K/32]
template <int K, int NB = K/32, int KH = K/2>
__global__ void mxfp4_quant_f32(
    const float A[][K],
    uint8_t A_fp4[][KH],
    uint8_t A_scale[][NB],
    int actual_m
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = actual_m * NB;
    if (idx >= total) return;
    int row = idx / NB;
    int blk = idx % NB;
    QBlk qb = qblock_f32(&A[row][blk * 32]);
    *reinterpret_cast<u128*>(&A_fp4[row][blk * 16]) =
        *reinterpret_cast<u128*>(&qb.d[0]);
    A_scale[row][blk] = qb.e;
}

// ═══════════════════════════════════════════════════════════════════════
// FP4×FP4 GEMM: C[M,N] = A_fp4[M,K] × B_fp4[N,K]^T
// Both A and B are pre-quantized FP4 with separate E8M0 scales.
// No on-the-fly quantization — just loads and MFMA.
// ═══════════════════════════════════════════════════════════════════════
template <int N, int K, int NB = K/32, int KH = K/2>
__global__ void moe_gemm_fp4xfp4(
    const uint8_t A_fp4[][KH],      // [M, K/2] pre-quantized activation fp4x2
    const uint8_t A_scale[][NB],    // [M, K/32] activation E8M0 scales
    const uint8_t B_fp4[][KH],      // [N, K/2] weight fp4x2
    const uint8_t B_scale[][NB],    // [N, K/32] weight E8M0 scales
    float C[][N],
    int actual_m
) {
    const int lane = threadIdx.x % 64;
    const int tile_m = blockIdx.x * 16;
    const int tile_n = blockIdx.y * 16;
    constexpr int BPC = 4;
    constexpr int KI = NB / BPC;
    f4 acc = {};
    for (int ki = 0; ki < KI; ki++) {
        int blk0 = ki * BPC;
        // Load pre-quantized A tile
        int a_row = tile_m + (lane % 16);
        int a_blk = blk0 + (lane / 16);
        uint32_t a_r[8] = {};
        int32_t a_sc = 0;
        if (a_row < actual_m) {
            *reinterpret_cast<u128*>(&a_r[0]) =
                *reinterpret_cast<const u128*>(&A_fp4[a_row][a_blk * 16]);
            a_sc = bcast(A_scale[a_row][a_blk]);
        }
        // Load pre-quantized B tile
        int b_row = tile_n + (lane % 16);
        int b_blk = blk0 + (lane / 16);
        uint32_t b_r[8] = {};
        int32_t b_sc = 0;
        if (b_row < N) {
            *reinterpret_cast<u128*>(&b_r[0]) =
                *reinterpret_cast<const u128*>(&B_fp4[b_row][b_blk * 16]);
            b_sc = bcast(B_scale[b_row][b_blk]);
        }
        i8v av = {(int)a_r[0],(int)a_r[1],(int)a_r[2],(int)a_r[3],
                  (int)a_r[4],(int)a_r[5],(int)a_r[6],(int)a_r[7]};
        i8v bv = {(int)b_r[0],(int)b_r[1],(int)b_r[2],(int)b_r[3],
                  (int)b_r[4],(int)b_r[5],(int)b_r[6],(int)b_r[7]};
        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            av, bv, acc, FMT_FP4, FMT_FP4, 0, a_sc, 0, b_sc);
    }
    int col = lane % 16, quad = lane / 16;
      for (int i = 0; i < 4; i++) {
          int gm = tile_m + i + 4*quad, gn = tile_n + col;
          if (gm < actual_m && gn < N) C[gm][gn] = acc[i];
      }
  }

// ═══════════════════════════════════════════════════════════════════════
// Software FP4 quantization (matches Triton _mxfp4_quant_op exactly)
// Uses FP32-domain scale + software E2M1 conversion (NO hardware intrinsics)
// ═══════════════════════════════════════════════════════════════════════

__device__ __forceinline__ uint8_t fp32_to_fp4_e2m1(float val) {
    uint32_t u = __float_as_uint(val);
    uint32_t s = u & 0x80000000u;
    uint32_t e = (u >> 23) & 0xFFu;
    uint32_t m = u & 0x7FFFFFu;
    if (e < 127u) {
        uint32_t adj = 126u - e;
        m = (adj < 24u) ? ((0x400000u | (m >> 1)) >> adj) : 0u;
    }
    e = (e >= 126u) ? (e - 126u) : 0u;
    uint32_t e2m1 = min((((e << 2) | (m >> 21)) + 1u) >> 1, 7u);
    return (uint8_t)((s >> 28) | e2m1);
}

template <int K>
__global__ void mxfp4_quant_sw_bf16(const hip_bfloat16* __restrict__ input, uint8_t* __restrict__ out_fp4, uint8_t* __restrict__ out_scale, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int NB = K / 32;
    if (idx >= M * NB) return;
    int row = idx / NB, blk = idx % NB;
    float vals[32]; float mx = 0.f;
    for (int i = 0; i < 32; i++) { vals[i] = (float)input[row * K + blk * 32 + i]; float a = vals[i] < 0.f ? -vals[i] : vals[i]; mx = (a > mx) ? a : mx; }
    uint8_t e8m0 = 0; float qs = 0.f;
    if (mx != 0.f) { uint32_t b = __float_as_uint(mx); b = (b + 0x200000u) & 0xFF800000u; float r = __uint_as_float(b); float l = floorf(log2f(r)) - 2.f; l = fminf(fmaxf(l, -127.f), 127.f); qs = exp2f(-l); e8m0 = (uint8_t)((int)l + 127); }
    for (int i = 0; i < 16; i++) { uint8_t lo = fp32_to_fp4_e2m1(vals[2*i] * qs); uint8_t hi = fp32_to_fp4_e2m1(vals[2*i+1] * qs); out_fp4[row * (K/2) + blk * 16 + i] = lo | (hi << 4); }
    out_scale[row * NB + blk] = e8m0;
}

template <int K>
__global__ void mxfp4_quant_sw_f32(const float* __restrict__ input, uint8_t* __restrict__ out_fp4, uint8_t* __restrict__ out_scale, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int NB = K / 32;
    if (idx >= M * NB) return;
    int row = idx / NB, blk = idx % NB;
    float vals[32]; float mx = 0.f;
    for (int i = 0; i < 32; i++) { vals[i] = input[row * K + blk * 32 + i]; float a = vals[i] < 0.f ? -vals[i] : vals[i]; mx = (a > mx) ? a : mx; }
    uint8_t e8m0 = 0; float qs = 0.f;
    if (mx != 0.f) { uint32_t b = __float_as_uint(mx); b = (b + 0x200000u) & 0xFF800000u; float r = __uint_as_float(b); float l = floorf(log2f(r)) - 2.f; l = fminf(fmaxf(l, -127.f), 127.f); qs = exp2f(-l); e8m0 = (uint8_t)((int)l + 127); }
    for (int i = 0; i < 16; i++) { uint8_t lo = fp32_to_fp4_e2m1(vals[2*i] * qs); uint8_t hi = fp32_to_fp4_e2m1(vals[2*i+1] * qs); out_fp4[row * (K/2) + blk * 16 + i] = lo | (hi << 4); }
    out_scale[row * NB + blk] = e8m0;
}

template <int N>
__global__ void swiglu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int row = idx / N;
    int col = idx % N;
    float gate = input[row * 2 * N + col];
    float up = input[row * 2 * N + N + col];
    float silu_gate = gate / (1.f + expf(-gate));
  output[idx] = silu_gate * up;
}

__global__ void f32_to_bf16_trim_kernel(
    const float* __restrict__ input,      // [M, in_cols]
    hip_bfloat16* __restrict__ output,    // [M, out_cols]
    int M,
    int in_cols,
    int out_cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * out_cols) return;
    int row = idx / out_cols;
    int col = idx % out_cols;
    output[idx] = (hip_bfloat16)input[row * in_cols + col];
}

__global__ void weighted_scatter_add_kernel(
    float* __restrict__ output,
    const float* __restrict__ src,
    const int64_t* __restrict__ indices,
    const float* __restrict__ weights,
    int n_tok,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_tok * N) return;
    int row = idx / N;
    int col = idx % N;
    int out_row = (int)indices[row];
    atomicAdd(&output[out_row * N + col], weights[row] * src[row * N + col]);
}
'''

# ═══════════════════════════════════════════════════════════════════════
# C++ Wrapper — Template dispatch for (N, K) dimension pairs
# ═══════════════════════════════════════════════════════════════════════
CPP_SRC = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Launch FP4xFP4 GEMM (both A and B pre-quantized)
template <int N, int K>
void launch_gemm_fp4xfp4(torch::Tensor A_fp4, torch::Tensor A_scale,
                          torch::Tensor B_fp4, torch::Tensor B_scale,
                          torch::Tensor C, int M) {
    dim3 grid((M+15)/16, (N+15)/16);
    moe_gemm_fp4xfp4<N,K><<<grid, 64>>>(
        reinterpret_cast<const uint8_t(*)[K/2]>(A_fp4.data_ptr()),
        reinterpret_cast<const uint8_t(*)[K/32]>(A_scale.data_ptr()),
        reinterpret_cast<const uint8_t(*)[K/2]>(B_fp4.data_ptr()),
        reinterpret_cast<const uint8_t(*)[K/32]>(B_scale.data_ptr()),
        reinterpret_cast<float(*)[N]>(C.data_ptr()), M);
}

// Launch BF16→MXFP4 quantization kernel
template <int K>
void launch_quant_bf16(torch::Tensor A, torch::Tensor A_fp4, torch::Tensor A_scale, int M) {
    int total = M * (K / 32);
    mxfp4_quant_bf16<K><<<(total+255)/256, 256>>>(
        reinterpret_cast<const hip_bfloat16(*)[K]>(A.data_ptr()),
        reinterpret_cast<uint8_t(*)[K/2]>(A_fp4.data_ptr()),
        reinterpret_cast<uint8_t(*)[K/32]>(A_scale.data_ptr()), M);
}

// Launch F32→MXFP4 quantization kernel
template <int K>
void launch_quant_f32(torch::Tensor A, torch::Tensor A_fp4, torch::Tensor A_scale, int M) {
    int total = M * (K / 32);
    mxfp4_quant_f32<K><<<(total+255)/256, 256>>>(
        reinterpret_cast<const float(*)[K]>(A.data_ptr()),
        reinterpret_cast<uint8_t(*)[K/2]>(A_fp4.data_ptr()),
        reinterpret_cast<uint8_t(*)[K/32]>(A_scale.data_ptr()), M);
}


void moe_forward(
    torch::Tensor hidden_padded,
    torch::Tensor gate_up_weight,
    torch::Tensor down_weight,
    torch::Tensor gate_up_weight_scale,
    torch::Tensor down_weight_scale,
    torch::Tensor topk_weights,
    torch::Tensor topk_ids,
    torch::Tensor output,
    int E, int dep, int dhp
) {
    auto device = hidden_padded.device();
    int M = hidden_padded.size(0);

    for (int eid = 0; eid < E; eid++) {
        auto mask = (topk_ids == eid);
        if (!mask.any().item<bool>()) continue;

        auto wh = torch::where(mask);
        auto tok_idx = wh[0];
        auto k_idx = wh[1];
        int n = tok_idx.size(0);

        auto w = topk_weights.index({tok_idx, k_idx});
        auto x = hidden_padded.index({tok_idx}).contiguous();

        // Stage 1: Quant BF16->MXFP4
        auto a1_fp4 = torch::empty({n, dhp/2}, torch::dtype(torch::kUInt8).device(device));
        auto a1_sc = torch::empty({n, dhp/32}, torch::dtype(torch::kUInt8).device(device));
        [&](){
            int t = n * (dhp/32);
            auto a_ptr = reinterpret_cast<const hip_bfloat16*>(x.data_ptr());
            auto fp4_ptr = reinterpret_cast<uint8_t*>(a1_fp4.data_ptr());
            auto sc_ptr = reinterpret_cast<uint8_t*>(a1_sc.data_ptr());
#define D(k) if(dhp==k){hipLaunchKernelGGL((mxfp4_quant_sw_bf16<k>),dim3((t+255)/256),dim3(256),0,0,a_ptr,fp4_ptr,sc_ptr,n);return;}
            D(4096) D(7168) D(1024) D(2048) D(1536) D(256) D(512)
#undef D
        }();

        // Stage 1: Gate_up GEMM
        auto gu_w = gate_up_weight[eid];
        auto gu_s = gate_up_weight_scale.dim()==3 ? gate_up_weight_scale[eid] : gate_up_weight_scale.slice(0, eid*2*dep, (eid+1)*2*dep);
        auto C1 = torch::zeros({n, 2*dep}, torch::dtype(torch::kFloat32).device(device));
        [&](){
            int N=2*dep, K=dhp;
            dim3 grid((n+15)/16,(N+15)/16);
            auto a_fp4 = reinterpret_cast<const uint8_t*>(a1_fp4.data_ptr());
            auto a_sc = reinterpret_cast<const uint8_t*>(a1_sc.data_ptr());
            auto b_fp4 = reinterpret_cast<const uint8_t*>(gu_w.view(torch::kUInt8).contiguous().data_ptr());
            auto b_sc = reinterpret_cast<const uint8_t*>(gu_s.view(torch::kUInt8).contiguous().data_ptr());
            auto c_ptr = reinterpret_cast<float*>(C1.data_ptr());
#define D(nn,kk) if(N==nn&&K==kk){hipLaunchKernelGGL((moe_gemm_fp4xfp4<nn,kk>),grid,dim3(64),0,0,reinterpret_cast<const uint8_t(*)[kk/2]>(a_fp4),reinterpret_cast<const uint8_t(*)[kk/32]>(a_sc),reinterpret_cast<const uint8_t(*)[kk/2]>(b_fp4),reinterpret_cast<const uint8_t(*)[kk/32]>(b_sc),reinterpret_cast<float(*)[nn]>(c_ptr),n);return;}
            D(2048,4096) D(4096,1024) D(4096,7168) D(7168,2048) D(3072,4096) D(4096,1536) D(512,7168) D(7168,256) D(1024,7168) D(7168,512)
#undef D
        }();

        // SwiGLU
        auto inter = torch::empty({n, dep}, torch::dtype(torch::kFloat32).device(device));
        [&](){
            int t = n * dep;
            auto in_ptr = reinterpret_cast<const float*>(C1.data_ptr());
            auto out_ptr = reinterpret_cast<float*>(inter.data_ptr());
#define D(nn) if(dep==nn){hipLaunchKernelGGL((swiglu_kernel<nn>),dim3((t+255)/256),dim3(256),0,0,in_ptr,out_ptr,n);return;}
            D(4096) D(7168) D(1024) D(2048) D(1536) D(256) D(512)
#undef D
        }();

        // Stage 2: Cast F32->BF16, Quant BF16->MXFP4
        auto inter_bf16 = inter.to(torch::kBFloat16).contiguous();
        auto a2_fp4 = torch::empty({n, dep/2}, torch::dtype(torch::kUInt8).device(device));
        auto a2_sc = torch::empty({n, dep/32}, torch::dtype(torch::kUInt8).device(device));
        [&](){
            int t = n * (dep/32);
            auto a_ptr = reinterpret_cast<const hip_bfloat16*>(inter_bf16.data_ptr());
            auto fp4_ptr = reinterpret_cast<uint8_t*>(a2_fp4.data_ptr());
            auto sc_ptr = reinterpret_cast<uint8_t*>(a2_sc.data_ptr());
#define D(k) if(dep==k){hipLaunchKernelGGL((mxfp4_quant_sw_bf16<k>),dim3((t+255)/256),dim3(256),0,0,a_ptr,fp4_ptr,sc_ptr,n);return;}
            D(4096) D(7168) D(1024) D(2048) D(1536) D(256) D(512)
#undef D
        }();

        // Stage 2: Down GEMM
        auto dn_w = down_weight[eid];
        auto dn_s = down_weight_scale.dim()==3 ? down_weight_scale[eid] : down_weight_scale.slice(0, eid*dhp, (eid+1)*dhp);
        auto C2 = torch::zeros({n, dhp}, torch::dtype(torch::kFloat32).device(device));
        [&](){
            int N=dhp, K=dep;
            dim3 grid((n+15)/16,(N+15)/16);
            auto a_fp4 = reinterpret_cast<const uint8_t*>(a2_fp4.data_ptr());
            auto a_sc = reinterpret_cast<const uint8_t*>(a2_sc.data_ptr());
            auto b_fp4 = reinterpret_cast<const uint8_t*>(dn_w.view(torch::kUInt8).contiguous().data_ptr());
            auto b_sc = reinterpret_cast<const uint8_t*>(dn_s.view(torch::kUInt8).contiguous().data_ptr());
            auto c_ptr = reinterpret_cast<float*>(C2.data_ptr());
#define D(nn,kk) if(N==nn&&K==kk){hipLaunchKernelGGL((moe_gemm_fp4xfp4<nn,kk>),grid,dim3(64),0,0,reinterpret_cast<const uint8_t(*)[kk/2]>(a_fp4),reinterpret_cast<const uint8_t(*)[kk/32]>(a_sc),reinterpret_cast<const uint8_t(*)[kk/2]>(b_fp4),reinterpret_cast<const uint8_t(*)[kk/32]>(b_sc),reinterpret_cast<float(*)[nn]>(c_ptr),n);return;}
            D(2048,4096) D(4096,1024) D(4096,7168) D(7168,2048) D(3072,4096) D(4096,1536) D(512,7168) D(7168,256) D(1024,7168) D(7168,512)
#undef D
        }();

        // Weighted scatter-add
        {
            int t = n * dhp;
            hipLaunchKernelGGL(weighted_scatter_add_kernel,dim3((t+255)/256),dim3(256),0,0,
                reinterpret_cast<float*>(output.data_ptr()),
                reinterpret_cast<const float*>(C2.data_ptr()),
                reinterpret_cast<const int64_t*>(tok_idx.data_ptr()),
                reinterpret_cast<const float*>(w.data_ptr()),
                n, dhp);
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_mm_fp4xfp4", [](torch::Tensor A_fp4, torch::Tensor A_scale,
                                 torch::Tensor B_fp4, torch::Tensor B_scale,
                                 torch::Tensor C, int M, int N, int K) {
#define D(n,k) if(N==n&&K==k){launch_gemm_fp4xfp4<n,k>(A_fp4,A_scale,B_fp4,B_scale,C,M);return;}
        D(2048,4096) D(4096,1024) D(4096,7168) D(7168,2048) D(3072,4096) D(4096,1536) D(512,7168) D(7168,256) D(1024,7168) D(7168,512)
#undef D
        TORCH_CHECK(false, "No fp4xfp4 template for N=", N, " K=", K);
    });
    m.def("quant_bf16", [](torch::Tensor A, torch::Tensor A_fp4, torch::Tensor A_scale, int M, int K) {
#define D(k) if(K==k){launch_quant_bf16<k>(A,A_fp4,A_scale,M);return;}
        D(4096) D(7168) D(1024) D(2048) D(1536) D(256) D(512)
#undef D
        TORCH_CHECK(false, "No quant_bf16 template for K=", K);
    });
    m.def("quant_f32", [](torch::Tensor A, torch::Tensor A_fp4, torch::Tensor A_scale, int M, int K) {
#define D(k) if(K==k){launch_quant_f32<k>(A,A_fp4,A_scale,M);return;}
        D(4096) D(7168) D(1024) D(2048) D(1536) D(256) D(512)
#undef D
        TORCH_CHECK(false, "No quant_f32 template for K=", K);
    });
    m.def("quant_sw_bf16", [](torch::Tensor A, torch::Tensor A_fp4, torch::Tensor A_scale, int M, int K) {
        auto a_ptr = reinterpret_cast<const hip_bfloat16*>(A.data_ptr());
        auto fp4_ptr = reinterpret_cast<uint8_t*>(A_fp4.data_ptr());
        auto sc_ptr = reinterpret_cast<uint8_t*>(A_scale.data_ptr());
#define D(k) if(K==k){int t=M*(k/32);hipLaunchKernelGGL((mxfp4_quant_sw_bf16<k>),dim3((t+255)/256),dim3(256),0,0,a_ptr,fp4_ptr,sc_ptr,M);return;}
        D(4096) D(7168) D(1024) D(2048) D(1536) D(256) D(512)
#undef D
        TORCH_CHECK(false, "No quant_sw_bf16 for K=", K);
    });
    m.def("quant_sw_f32", [](torch::Tensor A, torch::Tensor A_fp4, torch::Tensor A_scale, int M, int K) {
        auto a_ptr = reinterpret_cast<const float*>(A.data_ptr());
        auto fp4_ptr = reinterpret_cast<uint8_t*>(A_fp4.data_ptr());
        auto sc_ptr = reinterpret_cast<uint8_t*>(A_scale.data_ptr());
#define D(k) if(K==k){int t=M*(k/32);hipLaunchKernelGGL((mxfp4_quant_sw_f32<k>),dim3((t+255)/256),dim3(256),0,0,a_ptr,fp4_ptr,sc_ptr,M);return;}
        D(4096) D(7168) D(1024) D(2048) D(1536) D(256) D(512)
#undef D
        TORCH_CHECK(false, "No quant_sw_f32 for K=", K);
    });
    m.def("swiglu", [](torch::Tensor input, torch::Tensor output, int M, int N) {
        auto in_ptr = reinterpret_cast<const float*>(input.data_ptr());
        auto out_ptr = reinterpret_cast<float*>(output.data_ptr());
#define D(n) if(N==n){int t=M*n;hipLaunchKernelGGL((swiglu_kernel<n>),dim3((t+255)/256),dim3(256),0,0,in_ptr,out_ptr,M);return;}
        D(4096) D(7168) D(1024) D(2048) D(1536) D(256) D(512)
#undef D
      TORCH_CHECK(false, "No swiglu template for N=", N);
    });
    m.def("f32_to_bf16_trim", [](torch::Tensor input, torch::Tensor output, int M, int in_cols, int out_cols) {
        int t = M * out_cols;
        f32_to_bf16_trim_kernel<<<(t+255)/256, 256>>>(
            reinterpret_cast<const float*>(input.data_ptr()),
            reinterpret_cast<hip_bfloat16*>(output.data_ptr()),
            M, in_cols, out_cols);
    });
    m.def("weighted_scatter_add", [](torch::Tensor output, torch::Tensor src, torch::Tensor indices, torch::Tensor weights, int n_tok, int N) {
        int t = n_tok * N;
        weighted_scatter_add_kernel<<<(t+255)/256, 256>>>(
            reinterpret_cast<float*>(output.data_ptr()),
            reinterpret_cast<const float*>(src.data_ptr()),
            reinterpret_cast<const int64_t*>(indices.data_ptr()),
            reinterpret_cast<const float*>(weights.data_ptr()),
            n_tok, N);
    });
    m.def("moe_forward", &moe_forward);
}
'''

# ═══════════════════════════════════════════════════════════════════════
# Python — Compilation, dequantization, per-expert helpers, MoE forward
# ═══════════════════════════════════════════════════════════════════════


def _compile():
    global _mod
    if _mod is not None: return True
    try:
        from torch.utils.cpp_extension import load_inline
        rh = os.environ.get('ROCM_HOME', '/opt/rocm')
        os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
        t0 = time.time()
        _mod = load_inline(name='moe_hip_v5', cpp_sources='', cuda_sources=[HIP_SRC+'\n'+CPP_SRC],
            extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3','-ffast-math','-munsafe-fp-atomics','--offload-arch=gfx950'],
            extra_include_paths=[f'{rh}/include'], verbose=True)
        print(f'[moe] HIP compiled in {time.time()-t0:.1f}s', flush=True)
        return True
    except Exception as e:
        import traceback; traceback.print_exc()
        return False

def _escale(scale_tensor, expert_idx, rows_per_expert):
    """Extract per-expert scale slice from the raw (un-shuffled) scale tensor.

    The raw scale tensor from generate_input may be:
      - 3D [E, rows_per_expert, scale_K]: direct expert indexing
      - 2D [E * rows_per_expert, scale_K]: row-offset slicing
        (occurs when aiter's get_torch_quant flattens 3D input before quantization)
    """
    if scale_tensor.dim() == 3:
        return scale_tensor[expert_idx]
    else:
        start = expert_idx * rows_per_expert
        return scale_tensor[start : start + rows_per_expert]



@triton.jit
def _dynamic_mxfp4_quant_kernel_asm_layout(
    x_ptr,
    x_fp4_ptr,
    bs_ptr,
    stride_x_m,
    stride_x_n,
    stride_x_fp4_m,
    stride_x_fp4_n,
    stride_bs_m,
    stride_bs_n,
    M: tl.constexpr,
    N: tl.constexpr,
    scaleN: tl.constexpr,
    scaleM_pad: tl.constexpr,
    scaleN_pad: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: tl.constexpr,
    SCALING_MODE: tl.constexpr,
    SHUFFLE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    stride_x_m = tl.cast(stride_x_m, tl.int64)
    stride_x_n = tl.cast(stride_x_n, tl.int64)
    stride_x_fp4_m = tl.cast(stride_x_fp4_m, tl.int64)
    stride_x_fp4_n = tl.cast(stride_x_fp4_n, tl.int64)

    x_offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x_offs_n = pid_n * MXFP4_QUANT_BLOCK_SIZE + tl.arange(0, MXFP4_QUANT_BLOCK_SIZE)
    x_offs = x_offs_m[:, None] * stride_x_m + x_offs_n[None, :] * stride_x_n
    x_mask = (x_offs_m < M)[:, None] & (x_offs_n < N)[None, :]
    x = tl.load(x_ptr + x_offs, mask=x_mask).to(tl.float32)

    # Calculate scale
    amax = tl.max(tl.abs(x), axis=1, keep_dims=True)
    amax = amax.to(tl.int32, bitcast=True)
    amax = (amax + 0x200000).to(tl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(tl.float32, bitcast=True)
    scale_e8m0_unbiased = tl.log2(amax).floor() - 2
    scale_e8m0_unbiased = tl.clamp(scale_e8m0_unbiased, min=-127, max=127)
    quant_scale = tl.exp2(-scale_e8m0_unbiased)

    # Compute quantized x
    qx = x * quant_scale

    # blockscale_e8m0
    bs_e8m0 = scale_e8m0_unbiased.to(tl.uint8) + 127

    # Convert quantized fp32 tensor to uint32 before converting to mxfp4 format
    # Note: MXFP4  S:1-bit, E:2-bit, M:1-bit
    #   Zeros: S000 -> +/-0
    #   Denormal Numbers: S001 -> +/- 0.5
    #   Normal Numbers:
    #           S010 -> +/- 1.0
    #           S011 -> +/- 1.5
    #           S100 -> +/- 2.0
    #           S101 -> +/- 3.0
    #           S110 -> +/- 4.0
    #           S111 -> +/- 6.0
    # FP4 format constants
    EXP_BIAS_FP32: tl.constexpr = 127
    EXP_BIAS_FP4: tl.constexpr = 1
    EBITS_F32: tl.constexpr = 8
    EBITS_FP4: tl.constexpr = 2
    MBITS_F32: tl.constexpr = 23
    MBITS_FP4: tl.constexpr = 1

    max_normal: tl.constexpr = 6
    min_normal: tl.constexpr = 1

    qx = qx.to(tl.uint32, bitcast=True)

    # Extract sign
    s = qx & 0x80000000
    # Set everything to positive, will add sign back at the end
    qx = qx ^ s

    qx_fp32 = qx.to(tl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (not saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = not (saturate_mask | denormal_mask)

    # Denormal numbers
    denorm_exp: tl.constexpr = (
        (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + 1
    )
    denorm_mask_int: tl.constexpr = denorm_exp << MBITS_F32
    denorm_mask_float: tl.constexpr = tl.cast(denorm_mask_int, tl.float32, bitcast=True)

    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(tl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(tl.uint8)

    # Normal numbers
    normal_x = qx
    # resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - MBITS_FP4)) & 1
    # update exponent, rounding bias part 1
    val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32) + (1 << 21) - 1
    normal_x += val_to_add
    # rounding bias part 2
    normal_x += mant_odd
    # take the bits!
    normal_x = normal_x >> (MBITS_F32 - MBITS_FP4)
    normal_x = normal_x.to(tl.uint8)

    # Merge results
    e2m1_value = tl.full(qx.type.get_block_shapes(), 0x7, dtype=tl.uint8)
    e2m1_value = tl.where(normal_mask, normal_x, e2m1_value)
    e2m1_value = tl.where(denormal_mask, denormal_x, e2m1_value)

    # add sign back
    sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    sign_lp = sign_lp.to(tl.uint8)
    e2m1_value = e2m1_value | sign_lp

    e2m1_value = tl.reshape(e2m1_value, [BLOCK_SIZE, MXFP4_QUANT_BLOCK_SIZE // 2, 2])
    evens, odds = tl.split(e2m1_value)
    out_tensor = evens | (odds << 4)

    out_offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_offs_n = pid_n * MXFP4_QUANT_BLOCK_SIZE // 2 + tl.arange(
        0, MXFP4_QUANT_BLOCK_SIZE // 2
    )
    out_offs = (
        out_offs_m[:, None] * stride_x_fp4_m + out_offs_n[None, :] * stride_x_fp4_n
    )
    out_mask = (out_offs_m < M)[:, None] & (out_offs_n < (N // 2))[None, :]
    tl.store(x_fp4_ptr + out_offs, out_tensor, mask=out_mask)

    bs_offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    bs_offs_n = pid_n

    if SHUFFLE:
        bs_offs_0 = bs_offs_m[:, None] // 32
        bs_offs_1 = bs_offs_m[:, None] % 32
        bs_offs_2 = bs_offs_1 % 16
        bs_offs_1 = bs_offs_1 // 16
        bs_offs_3 = bs_offs_n[None, :] // 8
        bs_offs_4 = bs_offs_n[None, :] % 8
        bs_offs_5 = bs_offs_4 % 4
        bs_offs_4 = bs_offs_4 // 4
        bs_offs = (
            bs_offs_1
            + bs_offs_4 * 2
            + bs_offs_2 * 2 * 2
            + bs_offs_5 * 2 * 2 * 16
            + bs_offs_3 * 2 * 2 * 16 * 4
            + bs_offs_0 * 2 * 16 * scaleN
        )
        bs_mask1 = (bs_offs_m < M)[:, None] & (bs_offs_n < scaleN)[None, :]
        bs_mask2 = (bs_offs_m < scaleM_pad)[:, None] & (bs_offs_n < scaleN_pad)[None, :]
        bs_e8m0 = tl.where(bs_mask1, bs_e8m0, 127)
        tl.store(bs_ptr + bs_offs, bs_e8m0, mask=bs_mask2)
    else:
        bs_offs = bs_offs_m[:, None] * stride_bs_m + bs_offs_n[None, :] * stride_bs_n
        bs_mask = (bs_offs_m < M)[:, None] & (bs_offs_n < N)[None, :]
        tl.store(bs_ptr + bs_offs, bs_e8m0, mask=bs_mask)

def dynamic_mxfp4_quant(
    x: torch.Tensor, scaling_mode: str = "even", shuffle: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.utility.fp4_utils import (
            _dynamic_mxfp4_quant_kernel_asm_layout,
        )
    """
    Quantize a tensor to MX FP4 format.

    Args:
        x: The input tensor, typically fp16 or bf16.
        scaling_mode: The method to calculate MX block scaling.
            - "even" (default): `even_round` in `quark.torch.quantization.utils`.
            - etc.
    Returns:
        A tuple of (x_fp4, blockscale_e8m0).
    """
    # Assume x is 2D-Tensor for now
    M, N = x.shape

    assert (N // 2) % 2 == 0

    # This is fixed by spec for MXFP4. Do not tune this.
    # For performance, perhaps, we should look at passing multiple of 32 column blocks
    # that a triton program can process
    MXFP4_QUANT_BLOCK_SIZE = 32

    x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    scaleM = triton.cdiv(M, 32) * 32
    scaleN_valid = triton.cdiv(N, MXFP4_QUANT_BLOCK_SIZE)
    scaleN = triton.cdiv(scaleN_valid, 8) * 8
    blockscale_e8m0 = torch.empty(
        (
            triton.cdiv(M, 256) * 256,
            scaleN,
        ),
        dtype=torch.uint8,
        device=x.device,
    )

    BLOCK_SIZE = 128
    grid = (triton.cdiv(M, BLOCK_SIZE), scaleN)
    _dynamic_mxfp4_quant_kernel_asm_layout[grid](
        x,
        x_fp4,
        blockscale_e8m0,
        *x.stride(),
        *x_fp4.stride(),
        *blockscale_e8m0.stride(),
        M=M,
        N=N,
        scaleN=scaleN_valid,
        scaleM_pad=scaleM,
        scaleN_pad=scaleN,
        BLOCK_SIZE=BLOCK_SIZE,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        SCALING_MODE=0,
        SHUFFLE=shuffle,
    )

    if not shuffle:
        # Trim the padding if not shuffled
        blockscale_e8m0 = blockscale_e8m0[:M, :scaleN_valid].contiguous()

    return (x_fp4.view(dtypes.fp4x2), blockscale_e8m0.view(dtypes.fp8_e8m0))



def custom_kernel(data: input_t) -> output_t:
    """MoE forward pass using custom HIP MFMA FP4xFP4 kernels."""
    (hidden_states, gate_up_weight, down_weight,
     gate_up_weight_scale, down_weight_scale,
     gate_up_weight_shuffled, down_weight_shuffled,
     gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
     topk_weights, topk_ids, config) = data

    dh = config["d_hidden"]
    dep = config["d_expert_pad"]
    dhp = config["d_hidden_pad"]
    M = config["bs"]
    E = gate_up_weight.shape[0]
    device = hidden_states.device

    if not _compile():
        return torch.zeros(M, dh, dtype=torch.bfloat16, device=device)

    if dhp > dh:
        hidden_padded = F.pad(hidden_states, (0, dhp - dh))
    else:
        hidden_padded = hidden_states

    output = torch.zeros(M, dhp, dtype=torch.float32, device=device)

    _mod.moe_forward(
        hidden_padded, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        topk_weights, topk_ids, output,
        E, dep, dhp)

    result = torch.empty(M, dh, dtype=torch.bfloat16, device=device)
    _mod.f32_to_bf16_trim(output, result, M, dhp, dh)
    return result
