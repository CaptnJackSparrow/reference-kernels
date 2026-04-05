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
import aiter
from aiter import QuantType, dtypes, ActivationType

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
}
'''

# ═══════════════════════════════════════════════════════════════════════
# Python — Compilation, dequantization, per-expert helpers, MoE forward
# ═══════════════════════════════════════════════════════════════════════

MXFP4_BLOCK_SIZE = 32
_FP4_LUT = None

def _init_lut(device):
    """Initialize FP4 E2M1 lookup table on the given device."""
    global _FP4_LUT
    if _FP4_LUT is None or _FP4_LUT.device != device:
        _FP4_LUT = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.float32, device=device)

def _mxfp4_to_f32(packed_fp4):
    """Unpack fp4x2 packed bytes to float32 via LUT.
    Each byte has two FP4 values: low nibble = even index, high nibble = odd index.
    Input:  [..., K//2]  (float4_e2m1fn_x2 or uint8)
    Output: [..., K]     (float32)
    """
    _init_lut(packed_fp4.device)
    orig_shape = packed_fp4.shape
    raw = packed_fp4.view(torch.uint8)
    low = (raw & 0x0F).long()
    high = ((raw >> 4) & 0x0F).long()
    result = torch.stack([_FP4_LUT[low], _FP4_LUT[high]], dim=-1)
    out_shape = list(orig_shape)
    out_shape[-1] *= 2
    return result.reshape(out_shape)

def _e8m0_to_f32(scale_bytes):
    """Convert E8M0 exponent bytes to float32 power-of-2 scales.
    E8M0: value = 2^(exp - 127) for exp != 0, else 0.
    Implemented via IEEE 754 bit manipulation: place exponent in bits [30:23].
    Input:  [...]  (float8_e8m0fnu or uint8)
    Output: [...]  (float32, same shape)
    """
    orig_shape = scale_bytes.shape
    raw = scale_bytes.view(torch.uint8)
    exp = raw.to(torch.int32)
    float_bits = exp << 23
    float_bits = torch.where(exp == 0, torch.zeros_like(float_bits), float_bits)
    return float_bits.view(torch.float32).reshape(orig_shape)

def _dequant_weight(weight_fp4, scale_e8m0, K):
    """Dequantize one expert's MXFP4 weight to float32.
    Applies per-block (block_size=32) E8M0 scaling:
      dequant[row, col] = fp4_value[row, col] * e8m0_scale[row, col // 32]

    Args:
        weight_fp4:  [N, K//2] packed FP4 weight
        scale_e8m0:  [N, K//32] E8M0 block scales
        K:           unpacked column count
    Returns: [N, K] float32
    """
    num_blocks = K // MXFP4_BLOCK_SIZE
    w_f32 = _mxfp4_to_f32(weight_fp4)
    s_f32 = _e8m0_to_f32(scale_e8m0)
    N = w_f32.shape[0]
    s_f32 = s_f32[:N, :num_blocks]
    w_blocked = w_f32.view(N, num_blocks, MXFP4_BLOCK_SIZE)
    scaled = w_blocked * s_f32.unsqueeze(-1)
    return scaled.view(N, K)

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


# ═══════════════════════════════════════════════════════════════════════
# Inlined from aiter.fused_moe — token sorting and dimension helpers
# ═══════════════════════════════════════════════════════════════════════

def _moe_sorting_impl(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size,
    expert_mask,
    num_local_tokens,
    dispatch_policy,
    use_opus,
):
    device = topk_ids.device
    M, topk = topk_ids.shape
    max_num_tokens_padded = int(topk_ids.numel() + num_experts * block_size - topk)

    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    sorted_ids = torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(
        max_num_tokens_padded, dtype=dtypes.fp32, device=device
    )
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device)
    num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
    moe_buf = torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)

    aiter.moe_sorting_fwd(
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        num_experts,
        int(block_size),
        expert_mask,
        num_local_tokens,
        dispatch_policy,
    )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


def moe_sorting(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size=32,
    expert_mask=None,
    num_local_tokens=None,
    dispatch_policy=0,
):
    return _moe_sorting_impl(
        topk_ids,
        topk_weights,
        num_experts,
        model_dim,
        moebuf_dtype,
        block_size,
        expert_mask,
        num_local_tokens,
        dispatch_policy,
        use_opus=False,
    )


@functools.lru_cache(maxsize=2048)
def get_inter_dim(w1_shape, w2_shape):
    E, _, model_dim = w1_shape
    E, model_dim, inter_dim = w2_shape
    int4_war = model_dim // w1_shape[-1]
    inter_dim *= int4_war
    return E, model_dim, inter_dim


# ═══════════════════════════════════════════════════════════════════════
# Inlined from aiter.ops.triton.quant.fused_mxfp4_quant
# ═══════════════════════════════════════════════════════════════════════

def fused_dynamic_mxfp4_quant_moe_sort(
    x,
    sorted_ids,
    num_valid_ids,
    token_num,
    topk,
    block_size=32,
    scaling_mode="even",
):
    from aiter.ops.triton._triton_kernels.quant.fused_mxfp4_quant import (
        _fused_dynamic_mxfp4_quant_moe_sort_kernel,
    )

    M, N = x.shape
    assert (N // 2) % 2 == 0

    MXFP4_QUANT_BLOCK_SIZE = 32

    x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
    scaleN_valid = triton.cdiv(N, MXFP4_QUANT_BLOCK_SIZE)
    scaleN = scaleN_valid

    if M <= 32:
        BLOCK_SIZE_Mx = 32
    else:
        BLOCK_SIZE_Mx = 128

    BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 8
    BLOCK_SIZE_M_u32, BLOCK_SIZE_N_u32 = 16, 4

    N_i = scaleN
    M_o, N_o = sorted_ids.shape[0], N_i
    assert (N_i // 2) % 2 == 0
    assert block_size % BLOCK_SIZE_M == 0

    blockscale_e8m0_sorted = torch.empty(
        (
            triton.cdiv(M_o, BLOCK_SIZE_M),
            triton.cdiv(N_o, BLOCK_SIZE_N),
            BLOCK_SIZE_N_u32,
            BLOCK_SIZE_M_u32,
            4,
        ),
        dtype=torch.uint8,
        device=x.device,
    )

    num_pid = triton.cdiv(M, BLOCK_SIZE_Mx) * scaleN + triton.cdiv(
        M_o, BLOCK_SIZE_M
    ) * triton.cdiv(N_i, BLOCK_SIZE_N)
    _fused_dynamic_mxfp4_quant_moe_sort_kernel[(num_pid,)](
        x,
        x_fp4,
        sorted_ids,
        num_valid_ids,
        blockscale_e8m0_sorted,
        M,
        N,
        scaleN,
        *x.stride(),
        *x_fp4.stride(),
        *blockscale_e8m0_sorted.stride(),
        token_num,
        M_o,
        N_i,
        MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
        BLOCK_SIZE_Mx=BLOCK_SIZE_Mx,
        BLOCK_SIZE_M=BLOCK_SIZE_M // 2,
        BLOCK_SIZE_N=BLOCK_SIZE_N // 2,
        TOPK=topk,
    )

    return (
        x_fp4.view(dtypes.fp4x2),
        blockscale_e8m0_sorted.view(dtypes.fp8_e8m0).view(-1, N_o),
    )


# ═══════════════════════════════════════════════════════════════════════
# Inlined from aiter.utility.fp4_utils — MoE scale sorting kernels
# ═══════════════════════════════════════════════════════════════════════

@triton.jit
def _moe_mxfp4_sort_kernel(
    blockscale_e8m0_ptr,
    sorted_ids_ptr,
    num_valid_ids_ptr,
    blockscale_e8m0_sorted_ptr,
    stride_blockscale_e8m0_m: tl.int64,
    stride_blockscale_e8m0_n: tl.int64,
    stride_o3: tl.int64,
    stride_o2: tl.int64,
    stride_o1: tl.int64,
    stride_o0: tl.int64,
    token_num,
    N_i,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    TOPK: tl.constexpr,
):
    pid_m = tl.program_id(0) * 2
    pid_n = tl.program_id(1) * 2
    num_valid_ids = tl.load(num_valid_ids_ptr)
    if pid_m * BLOCK_SIZE_M >= num_valid_ids:
        return
    out = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.uint32)
    for m_idx in range(2):
        m = m_idx * BLOCK_SIZE_M
        sorted_ids_offs_m = pid_m * BLOCK_SIZE_M + m + tl.arange(0, BLOCK_SIZE_M)
        sorted_ids_mask = sorted_ids_offs_m < num_valid_ids
        raw_ids = tl.load(sorted_ids_ptr + sorted_ids_offs_m, mask=sorted_ids_mask, other=token_num)
        token_ids = raw_ids & 0xFFFFFF
        if TOPK == 1:
            blockscale_e8m0_offs_m = token_ids
        else:
            blockscale_e8m0_offs_m = token_ids * TOPK + (raw_ids >> 24)
        row_addrs = blockscale_e8m0_offs_m[:, None] * stride_blockscale_e8m0_m
        row_mask = (token_ids < token_num)[:, None]
        for n_idx in range(2):
            i = m_idx + n_idx * 2
            col_offs = pid_n * BLOCK_SIZE_N + n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            gather_offs = row_addrs + col_offs[None, :] * stride_blockscale_e8m0_n
            col_mask = (col_offs < N_i)[None, :]
            sub = tl.load(blockscale_e8m0_ptr + gather_offs, mask=row_mask & col_mask).to(tl.uint8, bitcast=True)
            out = out | (sub.to(tl.uint32) << (i * 8))
    offs_0 = tl.arange(0, BLOCK_SIZE_M)
    offs_1 = tl.arange(0, BLOCK_SIZE_N)
    offs = offs_0[:, None] * stride_o0 + offs_1[None, :] * stride_o1 + pid_n // 2 * stride_o2 + pid_m // 2 * stride_o3
    tl.store(blockscale_e8m0_sorted_ptr + offs, out)


@triton.jit
def _moe_mxfp4_sort_kernel_fused_n(
    blockscale_e8m0_ptr,
    sorted_ids_ptr,
    num_valid_ids_ptr,
    blockscale_e8m0_sorted_ptr,
    stride_blockscale_e8m0_m: tl.int64,
    stride_blockscale_e8m0_n: tl.int64,
    stride_o3: tl.int64,
    stride_o2: tl.int64,
    stride_o1: tl.int64,
    stride_o0: tl.int64,
    token_num,
    N_i,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    TOPK: tl.constexpr,
    N_TILES: tl.constexpr,
):
    pid_m = tl.program_id(0) * 2
    num_valid_ids = tl.load(num_valid_ids_ptr)
    if pid_m * BLOCK_SIZE_M >= num_valid_ids:
        return
    offs_m0 = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    raw_0 = tl.load(sorted_ids_ptr + offs_m0, mask=offs_m0 < num_valid_ids, other=token_num)
    tid_0 = raw_0 & 0xFFFFFF
    if TOPK == 1:
        ridx_0 = tid_0
    else:
        ridx_0 = tid_0 * TOPK + (raw_0 >> 24)
    raddr_0 = ridx_0[:, None] * stride_blockscale_e8m0_m
    rmask_0 = (tid_0 < token_num)[:, None]
    offs_m1 = offs_m0 + BLOCK_SIZE_M
    raw_1 = tl.load(sorted_ids_ptr + offs_m1, mask=offs_m1 < num_valid_ids, other=token_num)
    tid_1 = raw_1 & 0xFFFFFF
    if TOPK == 1:
        ridx_1 = tid_1
    else:
        ridx_1 = tid_1 * TOPK + (raw_1 >> 24)
    raddr_1 = ridx_1[:, None] * stride_blockscale_e8m0_m
    rmask_1 = (tid_1 < token_num)[:, None]
    offs_row = tl.arange(0, BLOCK_SIZE_M)
    offs_col = tl.arange(0, BLOCK_SIZE_N)
    store_base = pid_m // 2 * stride_o3
    for n_tile in range(N_TILES):
        pid_n = n_tile * 2
        out = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.uint32)
        for m_idx in range(2):
            if m_idx == 0:
                cur_raddr = raddr_0
                cur_rmask = rmask_0
            else:
                cur_raddr = raddr_1
                cur_rmask = rmask_1
            for n_idx in range(2):
                i = m_idx + n_idx * 2
                col_offs = pid_n * BLOCK_SIZE_N + n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                gather_offs = cur_raddr + col_offs[None, :] * stride_blockscale_e8m0_n
                col_mask = (col_offs < N_i)[None, :]
                sub = tl.load(blockscale_e8m0_ptr + gather_offs, mask=cur_rmask & col_mask).to(tl.uint8, bitcast=True)
                out = out | (sub.to(tl.uint32) << (i * 8))
        store_offs = offs_row[:, None] * stride_o0 + offs_col[None, :] * stride_o1 + n_tile * stride_o2 + store_base
        tl.store(blockscale_e8m0_sorted_ptr + store_offs, out)


def moe_mxfp4_sort(blockscale_e8m0, sorted_ids, num_valid_ids, token_num, block_size=32):
    BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 8
    BLOCK_SIZE_M_u32, BLOCK_SIZE_N_u32 = 16, 4
    topk = 1
    if len(blockscale_e8m0.shape) == 3:
        topk = blockscale_e8m0.shape[1]
        blockscale_e8m0 = blockscale_e8m0.view(-1, blockscale_e8m0.shape[-1])
    M_i, N_i = blockscale_e8m0.shape
    M_o, N_o = sorted_ids.shape[0], N_i
    assert (N_i // 2) % 2 == 0
    assert block_size % BLOCK_SIZE_M == 0
    blockscale_e8m0_sorted = torch.empty(
        (triton.cdiv(M_o, BLOCK_SIZE_M), triton.cdiv(N_o, BLOCK_SIZE_N), BLOCK_SIZE_N_u32, BLOCK_SIZE_M_u32),
        dtype=torch.uint32, device=blockscale_e8m0.device,
    )
    _FUSED_N_THRESHOLD = 2048
    common_args = (
        blockscale_e8m0.view(torch.uint8), sorted_ids, num_valid_ids, blockscale_e8m0_sorted,
        *blockscale_e8m0.stride(), *blockscale_e8m0_sorted.stride(),
    )
    common_kwargs = dict(token_num=token_num, N_i=N_i, BLOCK_SIZE_M=BLOCK_SIZE_M // 2, BLOCK_SIZE_N=BLOCK_SIZE_N // 2, TOPK=topk)
    if token_num > _FUSED_N_THRESHOLD:
        N_TILES = triton.cdiv(N_i, BLOCK_SIZE_N)
        grid = (triton.cdiv(M_o, BLOCK_SIZE_M),)
        _moe_mxfp4_sort_kernel_fused_n[grid](*common_args, **common_kwargs, N_TILES=N_TILES)
    else:
        grid = (triton.cdiv(M_o, BLOCK_SIZE_M), triton.cdiv(N_i, BLOCK_SIZE_N))
        _moe_mxfp4_sort_kernel[grid](*common_args, **common_kwargs)
    return blockscale_e8m0_sorted.view(dtypes.fp8_e8m0).view(-1, N_o)


# ═══════════════════════════════════════════════════════════════════════
# Inlined from aiter.ops.moe_op — CK MoE stage forward wrappers
# ═══════════════════════════════════════════════════════════════════════

_dtype2str_dict = {
    torch.float16: "f16",
    torch.bfloat16: "b16",
}


def ck_moe_stage1_fwd(
    hidden_states, w1, w2, sorted_token_ids, sorted_expert_ids,
    num_valid_ids, out, topk, kernelName="",
    w1_scale=None, a1_scale=None, block_m=32,
    sorted_weights=None, quant_type=None, activation=None,
    splitk=1, use_non_temporal_load=False, dst_type=None,
):
    if quant_type is None:
        quant_type = QuantType.No
    if activation is None:
        activation = ActivationType.Silu
    aiter.ck_moe_stage1(
        hidden_states, w1, w2,
        sorted_token_ids, sorted_expert_ids, num_valid_ids,
        out, topk, kernelName,
        w1_scale, a1_scale, block_m,
        sorted_weights,
        quant_type.value, activation.value,
        int(splitk) if splitk is not None else splitk,
        use_non_temporal_load,
        None if dst_type is None else _dtype2str_dict.get(dst_type),
        is_shuffled=getattr(w1, "is_shuffled", False),
    )
    return out


def ck_moe_stage2_fwd(
    inter_states, w1, w2, sorted_token_ids, sorted_expert_ids,
    num_valid_ids, out, topk, kernelName="",
    w2_scale=None, a2_scale=None, block_m=32,
    sorted_weights=None, quant_type=None, activation=None,
    use_non_temporal_load=False,
):
    if quant_type is None:
        quant_type = QuantType.No
    if activation is None:
        activation = ActivationType.Silu
    aiter.ck_moe_stage2(
        inter_states, w1, w2,
        sorted_token_ids, sorted_expert_ids, num_valid_ids,
        out, topk, kernelName,
        w2_scale, a2_scale, block_m,
        sorted_weights,
        quant_type.value, activation.value,
        use_non_temporal_load=use_non_temporal_load,
        is_shuffled=getattr(w2, "is_shuffled", False),
    )
    return out

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


def _dequant_gemm(A_fp4, A_scale, B_fp4, B_scale, K):
    """Dequantize FP4 A and B and run torch.mm for correctness validation."""
    a_f32 = _mxfp4_to_f32(A_fp4)
    b_f32 = _mxfp4_to_f32(B_fp4)
    num_blocks = K // 32
    M_a = a_f32.shape[0]
    N_b = b_f32.shape[0]
    as_f32 = _e8m0_to_f32(A_scale)[:M_a, :num_blocks]
    bs_f32 = _e8m0_to_f32(B_scale)[:N_b, :num_blocks]
    a_blocked = a_f32.view(M_a, num_blocks, 32)
    b_blocked = b_f32.view(N_b, num_blocks, 32)
    a_scaled = (a_blocked * as_f32.unsqueeze(-1)).view(M_a, K)
    b_scaled = (b_blocked * bs_f32.unsqueeze(-1)).view(N_b, K)
    return torch.mm(a_scaled, b_scaled.T)


def custom_kernel(data: input_t) -> output_t:
    """MoE forward pass using custom HIP MFMA FP4xFP4 kernels."""
    (hidden_states, gate_up_weight, down_weight,
     gate_up_weight_scale, down_weight_scale,
     gate_up_weight_shuffled, down_weight_shuffled,
     gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
     topk_weights, topk_ids, config) = data

    dh = config["d_hidden"]
    de = config["d_expert"]
    dhp = config["d_hidden_pad"]
    dep = config["d_expert_pad"]
    M = config["bs"]
    E = gate_up_weight.shape[0]
    top_k = topk_ids.shape[1]
    device = hidden_states.device

    USE_HIP_QUANT = False
    USE_HIP_GEMM = True

    if not _compile():
        return torch.zeros(M, dh, dtype=torch.bfloat16, device=device)

    if dhp > dh:
        hidden_padded = F.pad(hidden_states, (0, dhp - dh))
    else:
        hidden_padded = hidden_states

    output = torch.zeros(M, dhp, dtype=torch.float32, device=device)

    for expert_id in range(E):
        mask = (topk_ids == expert_id)
        if not mask.any():
            continue

        token_indices, k_indices = torch.where(mask)
        weights = topk_weights[token_indices, k_indices]
        x = hidden_padded[token_indices].contiguous()
        n_tok = x.shape[0]

        # Stage 1: Quantize BF16 activations to MXFP4
        if USE_HIP_QUANT:
            a1_fp4 = torch.empty(n_tok, dhp // 2, dtype=torch.uint8, device=device)
            a1_scale = torch.empty(n_tok, dhp // 32, dtype=torch.uint8, device=device)
            _mod.quant_bf16(x, a1_fp4, a1_scale, n_tok, dhp)
        else:
            a1_fp4_raw, a1_scale_raw = dynamic_mxfp4_quant(x, shuffle=False)
            a1_fp4 = a1_fp4_raw.view(torch.uint8)
            a1_scale = a1_scale_raw.view(torch.uint8)[:n_tok, :dhp // 32].contiguous()

        # Stage 1: Gate_up GEMM [n_tok, dhp] x [2*dep, dhp]^T -> [n_tok, 2*dep]
        gu_w = gate_up_weight[expert_id]
        gu_s = _escale(gate_up_weight_scale, expert_id, 2 * dep)

        if USE_HIP_GEMM:
            C1 = torch.zeros(n_tok, 2 * dep, dtype=torch.float32, device=device)
            _mod.moe_mm_fp4xfp4(
                a1_fp4, a1_scale,
                gu_w.view(torch.uint8).contiguous(), gu_s.view(torch.uint8).contiguous(),
                C1, n_tok, 2 * dep, dhp)
        else:
            C1 = _dequant_gemm(
                a1_fp4, a1_scale,
                gu_w.view(torch.uint8).contiguous(), gu_s.view(torch.uint8).contiguous(),
                dhp)

        # SwiGLU activation
        gate = C1[:, :dep]
        up = C1[:, dep:]
        intermediate = F.silu(gate) * up

        # Stage 2: Quantize F32 intermediate to MXFP4
        if USE_HIP_QUANT:
            a2_fp4 = torch.empty(n_tok, dep // 2, dtype=torch.uint8, device=device)
            a2_scale = torch.empty(n_tok, dep // 32, dtype=torch.uint8, device=device)
            _mod.quant_f32(intermediate.contiguous(), a2_fp4, a2_scale, n_tok, dep)
        else:
            quant_func = aiter.get_triton_quant(QuantType.per_1x32)
            a2_fp4_raw, a2_scale_raw = quant_func(intermediate.to(torch.bfloat16), shuffle=False)
            a2_fp4 = a2_fp4_raw.view(torch.uint8)
            a2_scale = a2_scale_raw.view(torch.uint8)[:n_tok, :dep // 32].contiguous()

        # Stage 2: Down GEMM [n_tok, dep] x [dhp, dep]^T -> [n_tok, dhp]
        dn_w = down_weight[expert_id]
        dn_s = _escale(down_weight_scale, expert_id, dhp)

        if USE_HIP_GEMM:
            C2 = torch.zeros(n_tok, dhp, dtype=torch.float32, device=device)
            _mod.moe_mm_fp4xfp4(
                a2_fp4, a2_scale,
                dn_w.view(torch.uint8).contiguous(), dn_s.view(torch.uint8).contiguous(),
                C2, n_tok, dhp, dep)
        else:
            C2 = _dequant_gemm(
                a2_fp4, a2_scale,
                dn_w.view(torch.uint8).contiguous(), dn_s.view(torch.uint8).contiguous(),
                dep)

        # Weighted accumulation
        output.index_add_(0, token_indices, weights.unsqueeze(1) * C2)

    return output[:, :dh].to(torch.bfloat16)
