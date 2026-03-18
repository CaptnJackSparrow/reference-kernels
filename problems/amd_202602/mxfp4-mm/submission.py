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
constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int WARP_SIZE = 64;

__device__ __forceinline__ int sh_scale_off(int row, int col, int scaleN) {
    return (row%32)/16
         + (col%8)/4 * 2
         + (row%16)  * 4
         + (col%4)   * 64
         + (col/8)   * 256
         + (row/32)  * 32 * scaleN;
}

__global__ void mfma_fp4_gemm(
    const uint8_t* __restrict__ A_data,
    const uint8_t* __restrict__ B_data,
    const uint8_t* __restrict__ A_scale,
    const uint8_t* __restrict__ B_scale,
    hip_bfloat16* __restrict__ C,
    const int M, const int N, const int K,
    const int K_half,
    const int num_blocks,
    const int scaleN,
    const int a_scale_stride0,
    const int a_scale_stride1
) {
    const int tile_m = blockIdx.x * TILE_M;
    const int tile_n = blockIdx.y * TILE_N;
    const int lane = threadIdx.x;

    typedef float __attribute__((ext_vector_type(16))) float16_t;
    float16_t acc = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    const int k_iters = (num_blocks + 1) / 2;
    typedef int __attribute__((ext_vector_type(8))) int8_vec;

    for (int ki = 0; ki < k_iters; ki++) {
        const int blk0 = ki * 2;

        int a_row = lane % 32;
        int a_k_group = lane / 32;
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

        // A scale: linear layout with explicit strides (handles column-major)
        uint8_t a_e8m0 = 127;
        if (a_m < M && a_blk < num_blocks) {
            a_e8m0 = A_scale[a_m * a_scale_stride0 + a_blk * a_scale_stride1];
        }
        int32_t a_sc = (int32_t)a_e8m0 | ((int32_t)a_e8m0 << 8)
                      | ((int32_t)a_e8m0 << 16) | ((int32_t)a_e8m0 << 24);

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

        // B scale: shuffled layout
        uint8_t b_e8m0 = 127;
        if (b_n < N && b_blk < num_blocks) {
            b_e8m0 = B_scale[sh_scale_off(b_n, b_blk, scaleN)];
        }
        int32_t b_sc = (int32_t)b_e8m0 | ((int32_t)b_e8m0 << 8)
                      | ((int32_t)b_e8m0 << 16) | ((int32_t)b_e8m0 << 24);

        int8_vec a_vec = {(int)a_reg[0], (int)a_reg[1], (int)a_reg[2], (int)a_reg[3],
                          0, 0, 0, 0};
        int8_vec b_vec = {(int)b_reg[0], (int)b_reg[1], (int)b_reg[2], (int)b_reg[3],
                          0, 0, 0, 0};

        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_vec, b_vec, acc,
            FMT_FP4, FMT_FP4,
            0, a_sc,
            0, b_sc
        );
    }

    int col = lane % 32;
    int half = lane / 32;
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
    torch::Tensor A_data,
    torch::Tensor B_data,
    torch::Tensor A_scale,
    torch::Tensor B_scale,
    int M, int N, int K, int K_half, int num_blocks, int scaleN,
    int a_scale_stride0, int a_scale_stride1
) {
    auto C = torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(A_data.device()));

    dim3 grid((M + 31) / 32, (N + 31) / 32);
    dim3 block(64);

    hipLaunchKernelGGL(mfma_fp4_gemm,
        grid, block, 0, 0,
        reinterpret_cast<const uint8_t*>(A_data.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_data.data_ptr()),
        reinterpret_cast<const uint8_t*>(A_scale.data_ptr()),
        reinterpret_cast<const uint8_t*>(B_scale.data_ptr()),
        reinterpret_cast<hip_bfloat16*>(C.data_ptr()),
        M, N, K, K_half, num_blocks, scaleN,
        a_scale_stride0, a_scale_stride1);

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
            print(f"[mxfp4-mm] MFMA kernel failed: {e}", flush=True)

    # # Fallback
    # x_fp4, bs_e8m0 = dynamic_mxfp4_quant(A)
    # bs_e8m0 = e8m0_shuffle(bs_e8m0)
    # A_q = x_fp4.view(dtypes.fp4x2)
    # A_scale_sh = bs_e8m0.view(dtypes.fp8_e8m0)
    # return aiter.gemm_a4w4(A_q, B_shuffle, A_scale_sh, B_scale_sh,
    #                        dtype=dtypes.bf16, bpreshuffle=True)
