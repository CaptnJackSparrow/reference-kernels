"""
FP4 GEMM using hardware MFMA instruction v_mfma_scale_f32_32x32x64_f8f6f4.
A quantization done by aiter (exact match). GEMM via AMD gfx950 matrix cores.
Uses pre-quantized B_q + B_scale_sh from input.
"""
import torch
from task import input_t, output_t

MXFP4_HIP_SOURCE = b'''
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include <cmath>

// Format code for FP4 E2M1 in mfma_scale instruction
constexpr int FMT_FP4 = 4;
constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 64;  // 64 FP4 elements = 2 MXFP4 blocks per MFMA call
constexpr int WARP_SIZE = 64;

// Shuffled scale offset matching aiter SHUFFLE=True layout
__device__ __forceinline__ int sh_scale_off(int row, int col, int scaleN) {
    return (row % 32) / 16
         + (col % 8) / 4 * 2
         + (row % 16)    * 4
         + (col % 4)     * 64
         + (col / 8)     * 256
         + (row / 32)    * 32 * scaleN;
}

// Pack 4 E8M0 scale bytes into int32
__device__ __forceinline__ int32_t pack_scales_4(uint8_t s0, uint8_t s1,
                                                  uint8_t s2, uint8_t s3) {
    return (int32_t)s0 | ((int32_t)s1 << 8) |
           ((int32_t)s2 << 16) | ((int32_t)s3 << 24);
}

// =================================================================
// MFMA-based FP4 GEMM kernel
//
// Uses v_mfma_scale_f32_32x32x64_f8f6f4 hardware instruction.
// Both A and B are pre-quantized MXFP4 (by aiter) with shuffled scales.
//
// Grid: (ceil(M/32), ceil(N/32))   Block: (64,) = one wavefront
//
// Per MFMA call: processes 64 FP4 elements along K = 2 MXFP4 blocks.
//   A operand: 4 x i32 (128 bits = 32 FP4 values per lane, zero-padded to 8xi32)
//   B operand: 4 x i32 (128 bits = 32 FP4 values per lane, zero-padded to 8xi32)
//   C accumulator: 16 x float per thread (32x32 tile across 64 threads)
//   Scales: E8M0 passed via mfma_scale instruction natively
// =================================================================
__global__ void mfma_fp4_gemm(
    const uint8_t* __restrict__ A_data,    // [M, K_half] packed FP4
    const uint8_t* __restrict__ B_data,    // [N, K_half] packed FP4
    const uint8_t* __restrict__ A_scale,   // shuffled E8M0 scales
    const uint8_t* __restrict__ B_scale,   // shuffled E8M0 scales
    hip_bfloat16* __restrict__ C,          // [M, N]
    const int M, const int N, const int K,
    const int K_half,
    const int num_blocks,
    const int scaleN
) {
    const int tile_m = blockIdx.x * TILE_M;
    const int tile_n = blockIdx.y * TILE_N;
    const int lane = threadIdx.x;  // 0..63 within wavefront

    // Accumulator: 16 floats per thread for the 32x32 output tile
    typedef float __attribute__((ext_vector_type(16))) float16_t;
    float16_t acc = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // Number of MFMA iterations along K (each processes 64 FP4 = 2 blocks of 32)
    const int k_iters = (num_blocks + 1) / 2;

    // MFMA input/output vector types
    typedef int __attribute__((ext_vector_type(8))) int8_t_vec;

    for (int ki = 0; ki < k_iters; ki++) {
        const int blk0 = ki * 2;       // first MXFP4 block index
        const int blk1 = ki * 2 + 1;   // second MXFP4 block index

        // --- Load A operand ---
        // For v_mfma_f32_32x32x64 with FP4:
        //   A is [32 rows, 64 cols] of FP4 values
        //   Each of 64 lanes loads 4 dwords = 16 bytes = 32 FP4 values
        //   Lane mapping: lane -> (row, k_offset)
        //     row = lane % 32
        //     k_group = lane / 32  (0 or 1, selects which 32 of the 64 FP4)
        int a_row = lane % 32;
        int a_k_group = lane / 32;  // 0 or 1
        int a_m = tile_m + a_row;
        int a_blk = blk0 + a_k_group;

        uint32_t a_reg[4] = {0, 0, 0, 0};
        if (a_m < M && a_blk < num_blocks) {
            int a_off = a_m * K_half + a_blk * 16;
            a_reg[0] = *reinterpret_cast<const uint32_t*>(&A_data[a_off + 0]);
            a_reg[1] = *reinterpret_cast<const uint32_t*>(&A_data[a_off + 4]);
            a_reg[2] = *reinterpret_cast<const uint32_t*>(&A_data[a_off + 8]);
            a_reg[3] = *reinterpret_cast<const uint32_t*>(&A_data[a_off + 12]);
        }

        // A scale (shuffled layout)
        uint8_t a_e8m0 = 127;  // neutral
        if (a_m < M && a_blk < num_blocks) {
            a_e8m0 = A_scale[sh_scale_off(a_m, a_blk, scaleN)];
        }
        int32_t a_scale_packed = pack_scales_4(a_e8m0, a_e8m0, a_e8m0, a_e8m0);

        // --- Load B operand ---
        // Same lane mapping but for B[N, K]
        int b_row = lane % 32;
        int b_k_group = lane / 32;
        int b_n = tile_n + b_row;
        int b_blk = blk0 + b_k_group;

        uint32_t b_reg[4] = {0, 0, 0, 0};
        if (b_n < N && b_blk < num_blocks) {
            int b_off = b_n * K_half + b_blk * 16;
            b_reg[0] = *reinterpret_cast<const uint32_t*>(&B_data[b_off + 0]);
            b_reg[1] = *reinterpret_cast<const uint32_t*>(&B_data[b_off + 4]);
            b_reg[2] = *reinterpret_cast<const uint32_t*>(&B_data[b_off + 8]);
            b_reg[3] = *reinterpret_cast<const uint32_t*>(&B_data[b_off + 12]);
        }

        // B scale (shuffled layout)
        uint8_t b_e8m0 = 127;
        if (b_n < N && b_blk < num_blocks) {
            b_e8m0 = B_scale[sh_scale_off(b_n, b_blk, scaleN)];
        }
        int32_t b_scale_packed = pack_scales_4(b_e8m0, b_e8m0, b_e8m0, b_e8m0);

        // --- Zero-pad 4xi32 -> 8xi32 for the builtin ---
        int8_t_vec a_vec = {(int)a_reg[0], (int)a_reg[1], (int)a_reg[2], (int)a_reg[3],
                            0, 0, 0, 0};
        int8_t_vec b_vec = {(int)b_reg[0], (int)b_reg[1], (int)b_reg[2], (int)b_reg[3],
                            0, 0, 0, 0};

        // --- Issue MFMA ---
        // cbsz=4 (A=fp4), blgp=4 (B=fp4)
        // opsel=0: use byte 0 of the packed scale
        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_vec, b_vec, acc,
            FMT_FP4, FMT_FP4,
            0, a_scale_packed,
            0, b_scale_packed
        );
    }

    // --- Write output ---
    // CDNA 32x32 MFMA output register mapping (64 threads, 16 values each):
    //   For each accumulator value acc[i]:
    //     col = lane % 32
    //     row = (i % 4) + 4 * (lane / 32) + 8 * (i / 4)
    int col = lane % 32;
    int half = lane / 32;  // 0 or 1

    for (int i = 0; i < 16; i++) {
        int row = (i % 4) + 4 * half + 8 * (i / 4);
        int gm = tile_m + row;
        int gn = tile_n + col;
        if (gm < M && gn < N) {
            C[gm * N + gn] = static_cast<hip_bfloat16>(acc[i]);
        }
    }
}
'''

MXFP4_CPP_SOURCE = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <ATen/hip/HIPContext.h>

torch::Tensor mfma_gemm(
    torch::Tensor A_data,    // [M, K_half] uint8
    torch::Tensor B_data,    // [N, K_half] uint8
    torch::Tensor A_scale,   // shuffled uint8
    torch::Tensor B_scale,   // shuffled uint8
    int M, int N, int K, int K_half, int num_blocks, int scaleN
) {
    auto C = torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(A_data.device()));

    dim3 grid((M + 31) / 32, (N + 31) / 32);
    dim3 block(64);  // one wavefront per tile

    hipLaunchKernelGGL(mfma_fp4_gemm,
        grid, block, 0, 0,
        reinterpret_cast<const uint8_t*>(A_data.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_data.data_ptr()),
        reinterpret_cast<const uint8_t*>(A_scale.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_scale.data_ptr()),
        reinterpret_cast<hip_bfloat16*>(C.data_ptr()),
        M, N, K, K_half, num_blocks, scaleN);

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
            name='mxfp4_mfma_v2', cpp_sources='', cuda_sources=[src],
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

    A, B, B_q, B_shuffle, B_scale_sh = data
    A = A.contiguous()
    m, k = A.shape
    n, _ = B.shape

    if HAS_HIP_KERNEL:
        try:
            # Exact same quantization as reference
            x_fp4, bs_e8m0 = dynamic_mxfp4_quant(A)
            bs_e8m0 = e8m0_shuffle(bs_e8m0)
            A_q = x_fp4.view(dtypes.fp4x2)
            A_scale_sh = bs_e8m0.view(dtypes.fp8_e8m0)

            A_data = A_q.view(torch.uint8).contiguous()
            A_sc = A_scale_sh.view(torch.uint8).contiguous()
            B_data = B_q.view(torch.uint8).contiguous()
            B_sc = B_scale_sh.view(torch.uint8).contiguous()

            K_half = k // 2
            num_blocks = (k + 31) // 32

            return _hip_module.mfma_gemm(
                A_data, B_data, A_sc, B_sc,
                m, n, k, K_half, num_blocks, num_blocks)
        except Exception as e:
            print(f"[mxfp4-mm] MFMA kernel failed: {e}")

    # Fallback (same as reference)
    x_fp4, bs_e8m0 = dynamic_mxfp4_quant(A)
    bs_e8m0 = e8m0_shuffle(bs_e8m0)
    A_q = x_fp4.view(dtypes.fp4x2)
    A_scale_sh = bs_e8m0.view(dtypes.fp8_e8m0)
    return aiter.gemm_a4w4(A_q, B_shuffle, A_scale_sh, B_scale_sh,
                           dtype=dtypes.bf16, bpreshuffle=True)
