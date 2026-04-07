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

// ═══════════════════════════════════════════════════════════════════════
// Single-launch fused GEMM+SwiGLU for Stage 1 across ALL experts
// Uses sorted token IDs and per-block expert IDs to dispatch weights
// ═══════════════════════════════════════════════════════════════════════
template <int N, int K, int NB = K/32, int KH = K/2>
__global__ void moe_fused_stage1(
    const uint8_t* __restrict__ A_fp4_base,
    const uint8_t* __restrict__ A_scale_base,
    const uint8_t* __restrict__ B_fp4_base,
    const uint8_t* __restrict__ B_scale_base,
    float* __restrict__ output,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ sorted_expert_ids,
    int num_valid,
    int M_tokens
) {
    const int m_block = blockIdx.x;
    const int tile_n = blockIdx.y * 16;
    const int lane = threadIdx.x % 64;

    int expert_id = sorted_expert_ids[m_block];

    int64_t w_offset = (int64_t)expert_id * 2 * N;
    const uint8_t (*B_gate)[KH] = reinterpret_cast<const uint8_t(*)[KH]>(B_fp4_base + w_offset * KH);
    const uint8_t (*B_gate_s)[NB] = reinterpret_cast<const uint8_t(*)[NB]>(B_scale_base + w_offset * NB);
    const uint8_t (*B_up)[KH] = reinterpret_cast<const uint8_t(*)[KH]>(B_fp4_base + (w_offset + N) * KH);
    const uint8_t (*B_up_s)[NB] = reinterpret_cast<const uint8_t(*)[NB]>(B_scale_base + (w_offset + N) * NB);

    const uint8_t (*A_fp4)[KH] = reinterpret_cast<const uint8_t(*)[KH]>(A_fp4_base);
    const uint8_t (*A_scale)[NB] = reinterpret_cast<const uint8_t(*)[NB]>(A_scale_base);

    constexpr int BPC = 4;
    constexpr int KI = NB / BPC;
    f4 acc_gate = {}, acc_up = {};

    for (int ki = 0; ki < KI; ki++) {
        int blk0 = ki * BPC;
        int sorted_pos = m_block * 16 + (lane % 16);
        int a_blk = blk0 + (lane / 16);
        uint32_t a_r[8] = {};
        int32_t a_sc = 0;
        if (sorted_pos < num_valid) {
            int token_id = sorted_token_ids[sorted_pos];
            if (token_id >= 0 && token_id < M_tokens) {
                *reinterpret_cast<u128*>(&a_r[0]) = *reinterpret_cast<const u128*>(&A_fp4[token_id][a_blk * 16]);
                a_sc = bcast(A_scale[token_id][a_blk]);
            }
        }
        i8v av = {(int)a_r[0],(int)a_r[1],(int)a_r[2],(int)a_r[3],(int)a_r[4],(int)a_r[5],(int)a_r[6],(int)a_r[7]};

        int b_row = tile_n + (lane % 16);
        int b_blk = blk0 + (lane / 16);
        uint32_t bg_r[8] = {}, bu_r[8] = {};
        int32_t bg_sc = 0, bu_sc = 0;
        if (b_row < N) {
            *reinterpret_cast<u128*>(&bg_r[0]) = *reinterpret_cast<const u128*>(&B_gate[b_row][b_blk * 16]);
            bg_sc = bcast(B_gate_s[b_row][b_blk]);
            *reinterpret_cast<u128*>(&bu_r[0]) = *reinterpret_cast<const u128*>(&B_up[b_row][b_blk * 16]);
            bu_sc = bcast(B_up_s[b_row][b_blk]);
        }
        i8v bgv = {(int)bg_r[0],(int)bg_r[1],(int)bg_r[2],(int)bg_r[3],(int)bg_r[4],(int)bg_r[5],(int)bg_r[6],(int)bg_r[7]};
        i8v buv = {(int)bu_r[0],(int)bu_r[1],(int)bu_r[2],(int)bu_r[3],(int)bu_r[4],(int)bu_r[5],(int)bu_r[6],(int)bu_r[7]};

        acc_gate = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(av, bgv, acc_gate, FMT_FP4, FMT_FP4, 0, a_sc, 0, bg_sc);
        acc_up = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(av, buv, acc_up, FMT_FP4, FMT_FP4, 0, a_sc, 0, bu_sc);
    }

    int col = lane % 16, quad = lane / 16;
    for (int i = 0; i < 4; i++) {
        int gm = m_block * 16 + i + 4*quad, gn = tile_n + col;
        if (gm < num_valid && gn < N) {
            float g = acc_gate[i], u = acc_up[i];
            output[gm * N + gn] = (g / (1.f + expf(-g))) * u;
        }
    }
}

// Single-launch MoE Stage 2: GEMM + weighted scatter-add (non-templated)
__global__ void moe_fused_stage2(
    const uint8_t* __restrict__ A_fp4_base,
    const uint8_t* __restrict__ A_scale_base,
    const uint8_t* __restrict__ B_fp4_base,
    const uint8_t* __restrict__ B_scale_base,
    float* __restrict__ output,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ sorted_expert_ids,
    const float* __restrict__ sorted_weights,
    int num_valid,
    int M_tokens,
    int N,
    int K
) {
    const int KH = K / 2;
    const int NB = K / 32;
    const int m_block = blockIdx.x;
    const int tile_n = blockIdx.y * 16;
    const int lane = threadIdx.x % 64;

    int expert_id = sorted_expert_ids[m_block];
    if (expert_id >= 0x7FFFFFFF) return;

    int64_t w_row_offset = (int64_t)expert_id * N;

    constexpr int BPC = 4;
    int KI = NB / BPC;
    f4 acc = {};

    for (int ki = 0; ki < KI; ki++) {
        int blk0 = ki * BPC;
        int sorted_pos = m_block * 16 + (lane % 16);
        int a_blk = blk0 + (lane / 16);
        uint32_t a_r[8] = {};
        int32_t a_sc = 0;
        if (sorted_pos < num_valid) {
            *reinterpret_cast<u128*>(&a_r[0]) = *reinterpret_cast<const u128*>(&A_fp4_base[(int64_t)sorted_pos * KH + a_blk * 16]);
            a_sc = bcast(A_scale_base[(int64_t)sorted_pos * NB + a_blk]);
        }
        int b_row = tile_n + (lane % 16);
        int b_blk = blk0 + (lane / 16);
        uint32_t b_r[8] = {};
        int32_t b_sc = 0;
        if (b_row < N) {
            int64_t brow_abs = w_row_offset + b_row;
            *reinterpret_cast<u128*>(&b_r[0]) = *reinterpret_cast<const u128*>(&B_fp4_base[brow_abs * KH + b_blk * 16]);
            b_sc = bcast(B_scale_base[brow_abs * NB + b_blk]);
        }
        i8v av = {(int)a_r[0],(int)a_r[1],(int)a_r[2],(int)a_r[3],(int)a_r[4],(int)a_r[5],(int)a_r[6],(int)a_r[7]};
        i8v bv = {(int)b_r[0],(int)b_r[1],(int)b_r[2],(int)b_r[3],(int)b_r[4],(int)b_r[5],(int)b_r[6],(int)b_r[7]};
        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(av, bv, acc, FMT_FP4, FMT_FP4, 0, a_sc, 0, b_sc);
    }

    int col = lane % 16, quad = lane / 16;
    for (int i = 0; i < 4; i++) {
        int gm = m_block * 16 + i + 4*quad, gn = tile_n + col;
        if (gm < num_valid && gn < N) {
            int token_id = sorted_token_ids[gm];
            if (token_id >= 0 && token_id < M_tokens) {
                float w = sorted_weights[gm];
                atomicAdd(&output[token_id * N + gn], w * acc[i]);
            }
        }
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

// ---- Profiling ----
struct PerfStats {
    float t_quant1 = 0, t_gemm1_swiglu = 0;
    float t_quant2 = 0, t_gemm2_scatter = 0;
    float t_total = 0;
    int count = 0;
};

constexpr int PROFILE_INTERVAL = 10;

void moe_forward(
    torch::Tensor hidden_padded,
    torch::Tensor gate_up_weight,
    torch::Tensor down_weight,
    torch::Tensor gate_up_weight_scale,
    torch::Tensor down_weight_scale,
    torch::Tensor topk_weights,
    torch::Tensor topk_ids,
    torch::Tensor output,
    int E, int dep, int dhp, bool profile
) {
    auto device = hidden_padded.device();
    int M = hidden_padded.size(0);
    int top_k = topk_ids.size(1);

    static std::unordered_map<int, std::unordered_map<int, PerfStats>> perf_map;
    auto& stats = perf_map[dep][dhp];
    stats.count++;
    bool do_profile = profile && (stats.count % PROFILE_INTERVAL == 0);

    hipEvent_t ev_total_start, ev_total_end;
    hipEvent_t ev0, ev1, ev2, ev3, ev4;
    if (do_profile) {
        (void)hipEventCreate(&ev_total_start); (void)hipEventCreate(&ev_total_end);
        (void)hipEventCreate(&ev0); (void)hipEventCreate(&ev1);
        (void)hipEventCreate(&ev2); (void)hipEventCreate(&ev3);
        (void)hipEventCreate(&ev4);
        (void)hipEventRecord(ev_total_start);
    }

    // ── Token sorting with GPU-side 16-alignment padding ──
    auto topk_ids_flat = topk_ids.reshape({-1});
    auto sort_result = torch::sort(topk_ids_flat, /*dim=*/0, /*descending=*/false);
    auto sorted_experts_flat = std::get<0>(sort_result).to(torch::kInt32);
    auto sort_indices = std::get<1>(sort_result);

    auto sorted_token_ids_raw = (sort_indices / top_k).to(torch::kInt32);
    auto topk_weights_flat = topk_weights.reshape({-1});
    auto sorted_weights_raw = topk_weights_flat.index({sort_indices}).to(torch::kFloat32);

    // Compute per-expert counts, then pad each to multiple of 16
    auto expert_counts = torch::bincount(sorted_experts_flat.to(torch::kInt64), {}, E);
    auto padded_counts = ((expert_counts + 15) / 16).to(torch::kInt64) * 16;
    auto padded_offsets = torch::cumsum(padded_counts, 0);  // [E] cumulative padded
    auto raw_offsets = torch::cumsum(expert_counts, 0);     // [E] cumulative raw

    int total_sorted = padded_offsets[-1].item<int>();  // 1 scalar GPU→CPU sync
    int num_m_blocks = total_sorted / 16;

    // Allocate padded arrays (sentinel token = M → skipped by bounds checks)
    auto sorted_token_ids = torch::full({total_sorted}, M,
        torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto sorted_weights_gpu = torch::zeros({total_sorted},
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto sorted_expert_ids_blocks = torch::zeros({num_m_blocks},
        torch::TensorOptions().dtype(torch::kInt32).device(device));

    // CPU loop over experts to scatter raw→padded with 16-alignment
    // One GPU→CPU sync for offsets (small tensors), then 3 GPU ops per active expert
    {
        auto raw_offsets_cpu = raw_offsets.cpu();
        auto padded_offsets_cpu = padded_offsets.cpu();
        auto expert_counts_cpu = expert_counts.cpu();
        auto ro = raw_offsets_cpu.data_ptr<int64_t>();
        auto po = padded_offsets_cpu.data_ptr<int64_t>();
        auto ec = expert_counts_cpu.data_ptr<int64_t>();

        for (int e = 0; e < E; e++) {
            int cnt = (int)ec[e];
            if (cnt == 0) continue;
            int rs = (e == 0) ? 0 : (int)ro[e-1];
            int ps = (e == 0) ? 0 : (int)po[e-1];
            sorted_token_ids.slice(0, ps, ps+cnt).copy_(sorted_token_ids_raw.slice(0, rs, rs+cnt));
            sorted_weights_gpu.slice(0, ps, ps+cnt).copy_(sorted_weights_raw.slice(0, rs, rs+cnt));
            int pcnt = ((cnt + 15) / 16) * 16;
            int block_start = ps / 16;
            int block_count = pcnt / 16;
            sorted_expert_ids_blocks.slice(0, block_start, block_start + block_count).fill_(e);
        }
    }

    // ── Stage 1: Quantize ALL activations at once ──
    if (do_profile) (void)hipEventRecord(ev0);

    auto a1_fp4 = torch::empty({M, dhp/2}, torch::dtype(torch::kUInt8).device(device));
    auto a1_scale = torch::empty({M, dhp/32}, torch::dtype(torch::kUInt8).device(device));
    [&](){
        int t = M * (dhp/32);
        auto a_ptr = reinterpret_cast<const hip_bfloat16*>(hidden_padded.data_ptr());
        auto fp4_ptr = reinterpret_cast<uint8_t*>(a1_fp4.data_ptr());
        auto sc_ptr = reinterpret_cast<uint8_t*>(a1_scale.data_ptr());
#define D(k) if(dhp==k){hipLaunchKernelGGL((mxfp4_quant_sw_bf16<k>),dim3((t+255)/256),dim3(256),0,0,a_ptr,fp4_ptr,sc_ptr,M);return;}
        D(4096) D(7168) D(1024) D(2048) D(1536) D(256) D(512)
#undef D
    }();

    if (do_profile) (void)hipEventRecord(ev1);

    // ── Stage 1 GEMM+SwiGLU: Single launch across ALL experts ──
    auto gu_w_contig = gate_up_weight.view(torch::kUInt8).contiguous();
    auto gu_s_contig = gate_up_weight_scale.reshape({-1, dhp/32}).view(torch::kUInt8).contiguous();

    auto A_fp4_ptr = reinterpret_cast<const uint8_t*>(a1_fp4.data_ptr());
    auto A_scale_ptr = reinterpret_cast<const uint8_t*>(a1_scale.data_ptr());
    auto B_fp4_ptr = reinterpret_cast<const uint8_t*>(gu_w_contig.data_ptr());
    auto B_scale_ptr = reinterpret_cast<const uint8_t*>(gu_s_contig.data_ptr());
    auto sorted_token_ids_ptr = sorted_token_ids.data_ptr<int32_t>();
    auto sorted_expert_ids_ptr = reinterpret_cast<const int32_t*>(sorted_expert_ids_blocks.data_ptr());

    auto inter_all = torch::empty({total_sorted, dep}, torch::dtype(torch::kFloat32).device(device));
    auto inter_ptr = reinterpret_cast<float*>(inter_all.data_ptr());

    {
        dim3 grid(num_m_blocks, (dep+15)/16);
#define D(nn,kk) if(dep==nn&&dhp==kk){hipLaunchKernelGGL((moe_fused_stage1<nn,kk>),grid,dim3(64),0,0,A_fp4_ptr,A_scale_ptr,B_fp4_ptr,B_scale_ptr,inter_ptr,sorted_token_ids_ptr,sorted_expert_ids_ptr,total_sorted,M);goto s1done;}
        D(256,7168) D(512,7168) D(1024,4096) D(1536,4096) D(2048,7168)
#undef D
        s1done:;
    }

    if (do_profile) (void)hipEventRecord(ev2);

    // ── Stage 2: Quantize ALL intermediates at once ──
    auto inter_bf16 = inter_all.to(torch::kBFloat16).contiguous();
    auto a2_fp4 = torch::empty({total_sorted, dep/2}, torch::dtype(torch::kUInt8).device(device));
    auto a2_scale = torch::empty({total_sorted, dep/32}, torch::dtype(torch::kUInt8).device(device));
    [&](){
        int t = total_sorted * (dep/32);
        auto a_ptr = reinterpret_cast<const hip_bfloat16*>(inter_bf16.data_ptr());
        auto fp4_ptr = reinterpret_cast<uint8_t*>(a2_fp4.data_ptr());
        auto sc_ptr = reinterpret_cast<uint8_t*>(a2_scale.data_ptr());
#define D(k) if(dep==k){hipLaunchKernelGGL((mxfp4_quant_sw_bf16<k>),dim3((t+255)/256),dim3(256),0,0,a_ptr,fp4_ptr,sc_ptr,total_sorted);return;}
        D(4096) D(7168) D(1024) D(2048) D(1536) D(256) D(512)
#undef D
    }();

    if (do_profile) (void)hipEventRecord(ev3);

    // ── Stage 2: Single-launch GEMM2 + weighted scatter-add ──
    {
        auto dn_w_flat = down_weight.view(torch::kUInt8).reshape({-1, dep/2}).contiguous();
        auto dn_s_flat = down_weight_scale.view(torch::kUInt8).reshape({-1, dep/32}).contiguous();
        dim3 grid(num_m_blocks, (dhp+15)/16);
        hipLaunchKernelGGL(moe_fused_stage2,grid,dim3(64),0,0,reinterpret_cast<const uint8_t*>(a2_fp4.data_ptr()),reinterpret_cast<const uint8_t*>(a2_scale.data_ptr()),reinterpret_cast<const uint8_t*>(dn_w_flat.data_ptr()),reinterpret_cast<const uint8_t*>(dn_s_flat.data_ptr()),reinterpret_cast<float*>(output.data_ptr()),sorted_token_ids.data_ptr<int32_t>(),sorted_expert_ids_blocks.data_ptr<int32_t>(),sorted_weights_gpu.data_ptr<float>(),total_sorted,M,dhp,dep);
    }

    if (do_profile) (void)hipEventRecord(ev4);

    if (do_profile) {
        (void)hipEventRecord(ev_total_end);
        (void)hipEventSynchronize(ev_total_end);
        float d_q1, d_g1sw, d_q2, d_g2sc, d_total;
        (void)hipEventElapsedTime(&d_q1, ev0, ev1);
        (void)hipEventElapsedTime(&d_g1sw, ev1, ev2);
        (void)hipEventElapsedTime(&d_q2, ev2, ev3);
        (void)hipEventElapsedTime(&d_g2sc, ev3, ev4);
        (void)hipEventElapsedTime(&d_total, ev_total_start, ev_total_end);
        stats.t_quant1 += d_q1; stats.t_gemm1_swiglu += d_g1sw;
        stats.t_quant2 += d_q2; stats.t_gemm2_scatter += d_g2sc;
        stats.t_total += d_total;
        int nn = stats.count / PROFILE_INTERVAL;
        if (nn < 5) {
            printf("[MoE-fused] dep=%d dhp=%d | "
                "quant1=%.1fus gemm1+swiglu=%.1fus "
                "quant2=%.1fus gemm2+scatter=%.1fus | "
                "total=%.1fus (avg over %d)\n",
                dep, dhp,
                stats.t_quant1/nn*1000, stats.t_gemm1_swiglu/nn*1000,
                stats.t_quant2/nn*1000, stats.t_gemm2_scatter/nn*1000,
                stats.t_total/nn*1000, nn);
        }
        (void)hipEventDestroy(ev_total_start); (void)hipEventDestroy(ev_total_end);
        (void)hipEventDestroy(ev0); (void)hipEventDestroy(ev1);
        (void)hipEventDestroy(ev2); (void)hipEventDestroy(ev3);
        (void)hipEventDestroy(ev4);
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
        _mod = load_inline(name='moe_hip_v9', cpp_sources='', cuda_sources=[HIP_SRC+'\n'+CPP_SRC],
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
    PROFILE = True

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
        E, dep, dhp, PROFILE)

    result = torch.empty(M, dh, dtype=torch.bfloat16, device=device)
    _mod.f32_to_bf16_trim(output, result, M, dhp, dh)
    return result
