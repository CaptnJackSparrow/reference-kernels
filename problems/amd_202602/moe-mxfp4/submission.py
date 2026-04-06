"""
MXFP4 Mixture-of-Experts (MoE) Fused Kernel — Custom HIP MFMA Implementation.

Implements a DeepSeek-R1 style MoE forward pass on AMD MI355X:
  - Custom HIP kernel: MFMA 16x16x128 FP4xFP4 matrix multiply
  - On-the-fly BF16→MXFP4 activation quantization via software E2M1 conversion
  - Per-expert GEMM with raw (un-shuffled) FP4 weights + E8M0 block scales
  - PyTorch-side MoE token routing, SwiGLU activation, weighted reduction

Pipeline per token per assigned expert:
  Stage 1: hidden_states → quant_mxfp4 → FP4xFP4 GEMM(gate_up_weight) → SwiGLU
  Stage 2: intermediate  → quant_mxfp4 → FP4xFP4 GEMM(down_weight)    → weighted sum
"""
import torch, os, time
import torch.nn.functional as F
from task import input_t, output_t

_mod = None  # Lazy-compiled HIP module handle

# ═══════════════════════════════════════════════════════════════════════
# HIP Kernel Source — MXFP4 FP4xFP4 GEMM with software A quantization
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

// ── Broadcast E8M0 exponent to all 4 bytes of int32 ────────────────
// MFMA scale parameter needs same E8M0 in all bytes: 0x7E → 0x7E7E7E7E
__device__ __forceinline__ int32_t bcast(uint8_t e) { return (int32_t)e * 0x01010101; }

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
    m.def("f32_to_bf16_trim", [](torch::Tensor input, torch::Tensor output, int M, int in_cols, int out_cols) {
        int t = M * out_cols;
        f32_to_bf16_trim_kernel<<<(t+255)/256, 256>>>(
            reinterpret_cast<const float*>(input.data_ptr()),
            reinterpret_cast<hip_bfloat16*>(output.data_ptr()),
            M, in_cols, out_cols);
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
