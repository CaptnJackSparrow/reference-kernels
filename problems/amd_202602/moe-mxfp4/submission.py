"""
MXFP4 MoE -- MFMA GEMM using shuffled weights. Full weight tensor passed to kernel,
per-expert offset computed via pointer arithmetic (no Python slicing).
"""
import torch
import torch.nn.functional as F
from task import input_t, output_t

MOE_HIP = b'''
// build_id: mfma_raw_colmajor_scale_v1
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include <cmath>

constexpr int FMT_FP4 = 4;
typedef float __attribute__((ext_vector_type(4))) float4_t;
typedef int __attribute__((ext_vector_type(8))) int8_vec;
typedef uint32_t __attribute__((ext_vector_type(4))) uint128_vec;

struct QuantBlock { uint32_t data[4]; uint8_t e8m0; };

__device__ __forceinline__ void load_bf16x32(const hip_bfloat16* p, float v[32]) {
    for (int c = 0; c < 32; c += 8) {
        uint32_t r[4];
        *reinterpret_cast<uint128_vec*>(r) = *reinterpret_cast<const uint128_vec*>(&p[c]);
        for (int j = 0; j < 4; j++) {
            v[c+j*2]   = __uint_as_float((r[j] & 0xFFFF) << 16);
            v[c+j*2+1] = __uint_as_float(r[j] & 0xFFFF0000u);
        }
    }
}

__device__ __forceinline__ QuantBlock quantize_fp4_block(const float v[32]) {
    float mx = 0.0f;
    for (int i = 0; i < 32; i++) mx = fmaxf(mx, fabsf(v[i]));
    uint8_t e; float qs;
    if (mx == 0.0f) { e = 0; qs = 0.0f; }
    else {
        uint32_t b = __float_as_uint(mx);
        b = (b + 0x200000u) & 0xFF800000u;
        int re = (int)((b >> 23) & 0xFF);
        int eu = re - 127 - 2;
        eu = max(-127, min(127, eu));
        e = (uint8_t)(eu + 127);
        qs = __uint_as_float((uint32_t)(127 - eu) << 23);
    }
    uint32_t pk[4] = {};
    for (int i = 0; i < 16; i++) {
        uint8_t p = 0;
        for (int j = 0; j < 2; j++) {
            float qx = v[2*i+j] * qs;
            uint32_t qb = __float_as_uint(qx);
            uint32_t sg = qb & 0x80000000u; qb ^= sg;
            float qa = __uint_as_float(qb);
            uint8_t f;
            if (qa >= 6.0f) f = 7;
            else if (qa < 1.0f) {
                constexpr uint32_t dm = 149u << 23;
                f = (uint8_t)(__float_as_uint(qa + __uint_as_float(dm)) - dm);
            } else {
                uint32_t mo = (qb >> 22) & 1;
                qb = (uint32_t)((int32_t)qb + (int32_t)0xC11FFFFF) + mo;
                f = (uint8_t)(qb >> 22);
            }
            f |= (uint8_t)(sg >> 28);
            p |= f << (4*j);
        }
        pk[i/4] |= ((uint32_t)p) << ((i%4)*8);
    }
    QuantBlock r; r.data[0]=pk[0]; r.data[1]=pk[1]; r.data[2]=pk[2]; r.data[3]=pk[3]; r.e8m0=e;
    return r;
}

__device__ __forceinline__ int32_t bcast_sc(uint8_t e) { return (int32_t)e * 0x01010101; }

// sh_scale_off from mxfp4-mm (exact copy)
__device__ __forceinline__ int sh_scale_off(int row, int col, int SCALE_N) {
    return (row%32)/16 + (col%8)/4*2 + (row%16)*4 + (col%4)*64 + (col/8)*256 + (row/32)*32*SCALE_N;
}

__device__ __forceinline__ float silu_f(float x) { return x/(1.0f+expf(-x)); }

// MFMA 16x16x128 GEMM: A[M,K] bf16 x B_raw[N,K/2] -> C[M,N] fp32
// B_data is RAW row-major, B_scale is RAW row-major
__global__ __launch_bounds__(64)
void mfma_gemm(
    const hip_bfloat16* __restrict__ A,
    const uint8_t* __restrict__ B, const uint8_t* __restrict__ Bs,
    float* __restrict__ C,
    int M, int N, int K, int KH, int NB,
    int expert_id, int64_t b_expert_stride, int64_t bs_expert_stride
) {
    B += expert_id * b_expert_stride;
    Bs += expert_id * bs_expert_stride;
    const int lane = threadIdx.x;
    const int tm = blockIdx.x * 16, tn = blockIdx.y * 16;
    float4_t acc = {};
    const int KI = (NB + 3) / 4;

    for (int ki = 0; ki < KI; ki++) {
        int b0 = ki * 4;

        // A: inline quantize
        int ar = tm + (lane % 16), ak = b0 + (lane / 16);
        float vals[32];
        if (ar < M && ak < NB) load_bf16x32(&A[ar * K + ak * 32], vals);
        else for (int i = 0; i < 32; i++) vals[i] = 0.0f;
        QuantBlock qb = quantize_fp4_block(vals);
        int32_t asc = bcast_sc(qb.e8m0);

        // B: load from shuffled layout using load_tile pattern (row * KH + blk * 16)
        int br = tn + (lane % 16), bk = b0 + (lane / 16);
        uint32_t breg[8] = {};
        if (br < N && bk < NB)
            *reinterpret_cast<uint128_vec*>(breg) =
                *reinterpret_cast<const uint128_vec*>(&B[br * KH + bk * 16]);
        // B scale: RAW row-major [N, num_blocks]
        uint8_t be = (br < N && bk < NB) ? Bs[br * NB + bk] : 0;
        int32_t bsc = bcast_sc(be);

        int8_vec av = {(int)qb.data[0],(int)qb.data[1],(int)qb.data[2],(int)qb.data[3],0,0,0,0};
        int8_vec bv = {(int)breg[0],(int)breg[1],(int)breg[2],(int)breg[3],0,0,0,0};
        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            av, bv, acc, FMT_FP4, FMT_FP4, 0, asc, 0, bsc);
    }

    int col = lane % 16, quad = lane / 16;
    for (int i = 0; i < 4; i++) {
        int r = i + 4 * quad, gm = tm + r, gn = tn + col;
        if (gm < M && gn < N) C[gm * N + gn] = acc[i];
    }
}
'''

MOE_CPP = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// run_gemm: takes base pointers already offset to the correct expert
torch::Tensor run_gemm(
    torch::Tensor A, torch::Tensor B, torch::Tensor Bs,
    int M, int N, int K, int expert_id,
    int64_t b_expert_stride, int64_t bs_expert_stride
) {
    int KH = K / 2, NB = K / 32;
    auto C = torch::zeros({M, N},
        torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    hipLaunchKernelGGL(mfma_gemm, grid, dim3(64), 0, 0,
        (const hip_bfloat16*)A.data_ptr(),
        (const uint8_t*)B.data_ptr(), (const uint8_t*)Bs.data_ptr(),
        (float*)C.data_ptr(), M, N, K, KH, NB,
        expert_id, b_expert_stride, bs_expert_stride);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_gemm", &run_gemm);
}
'''

_mod = None

def _build():
    global _mod
    import time, os, shutil, hashlib
    try:
        if not torch.cuda.is_available(): return
        if not hasattr(torch.version, 'hip') or torch.version.hip is None: return
        from torch.utils.cpp_extension import load_inline
        cache = os.path.expanduser('~/.cache/torch_extensions')
        if os.path.exists(cache): shutil.rmtree(cache, ignore_errors=True)
        rocm = os.environ.get('ROCM_HOME', '/opt/rocm')
        os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
        os.environ['MAX_JOBS'] = '4'
        src = MOE_HIP.decode() + '\n' + MOE_CPP
        nm = 'moe_' + hashlib.md5(src.encode()).hexdigest()[:12]
        t = time.time()
        _mod = load_inline(name=nm, cpp_sources='', cuda_sources=[src],
            extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '--offload-arch=gfx950'],
            extra_include_paths=[f'{rocm}/include'], verbose=True)
        print(f"[moe] compiled in {time.time()-t:.1f}s")
    except Exception as e:
        print(f"[moe] build failed: {e}")
        import traceback; traceback.print_exc()

try: _build()
except: pass


def custom_kernel(data: input_t) -> output_t:
    if _mod is None: return
    (hidden, w1, w2, w1s, w2s,
     w1_sh, w2_sh, w1s_sh, w2s_sh,
     topk_w, topk_ids, cfg) = data

    dh = cfg["d_hidden"]
    de = cfg["d_expert"]
    dhp = cfg["d_hidden_pad"]
    dep = cfg["d_expert_pad"]
    M = hidden.shape[0]
    top_k = topk_ids.shape[1]
    E = w1.shape[0]

    # Use RAW weights (not shuffled) - the ISA confirms load_tile works with row-major
    w1_flat = w1.contiguous().view(torch.uint8)   # [E, 2*dep, dhp/2]
    w2_flat = w2.contiguous().view(torch.uint8)   # [E, dhp, dep/2]
    w1s_flat = w1s.contiguous().view(torch.uint8)  # 2D or 3D scale
    w2s_flat = w2s.contiguous().view(torch.uint8)

    # Per-expert strides
    b1_stride = 2 * dep * (dhp // 2)
    b2_stride = dhp * (dep // 2)
    # Scale strides - reshape to 3D for correct per-expert access
    s1k = w1s_flat.shape[-1]  # last dim of scale tensor
    s2k = w2s_flat.shape[-1]
    w1s_3d = w1s_flat.reshape(E, 2 * dep, s1k)
    w2s_3d = w2s_flat.reshape(E, dhp, s2k)
    bs1_stride = 2 * dep * s1k
    bs2_stride = dhp * s2k

    output = torch.zeros(M, dh, dtype=torch.float32, device=hidden.device)

    h_pad = hidden
    if dhp > dh:
        h_pad = torch.zeros(M, dhp, dtype=torch.bfloat16, device=hidden.device)
        h_pad[:, :dh] = hidden

    # Compute expert strides for raw pointer offset in the kernel
    b1_stride = 2 * dep * (dhp // 2)  # bytes per expert in w1
    b2_stride = dhp * (dep // 2)       # bytes per expert in w2

    for e in range(E):
        mask = (topk_ids == e)
        if not mask.any():
            continue
        token_indices, slot_indices = mask.nonzero(as_tuple=True)
        weights = topk_w[token_indices, slot_indices]
        n_tok = token_indices.shape[0]

        x = h_pad[token_indices].contiguous()

        # Stage 1: gate_up GEMM - pass full RAW weight tensors + expert_id
        gu_out = _mod.run_gemm(x, w1_flat, w1s_3d.view(-1),
                               n_tok, 2 * dep, dhp, e, b1_stride, bs1_stride)

        gate = gu_out[:, :de]
        up = gu_out[:, de:2*de]
        intermediate = F.silu(gate) * up

        # Stage 2: down GEMM
        inter_bf16 = intermediate.to(torch.bfloat16)
        if dep > de:
            inter_pad = torch.zeros(n_tok, dep, dtype=torch.bfloat16, device=hidden.device)
            inter_pad[:, :de] = inter_bf16
            inter_bf16 = inter_pad

        down_out = _mod.run_gemm(inter_bf16, w2_flat, w2s_3d.view(-1),
                                 n_tok, dhp, dep, e, b2_stride, bs2_stride)

        expert_out = down_out[:, :dh]
        output.index_add_(0, token_indices, weights.unsqueeze(1) * expert_out)

    return output.to(torch.bfloat16)
