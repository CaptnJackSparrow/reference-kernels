"""
MLA (Multi-head Latent Attention) decode kernel — optimized implementation.

Implements multiple optimization strategies:
1. FP8 Q + FP8 KV using aiter's a8w8 persistent MLA kernel (baseline)
2. MXFP4 KV with dequantization + aiter MLA kernel (2x bandwidth savings)
3. MXFP4 Q + MXFP4 KV using gemm_a4w4 for QK^T (4x bandwidth savings)
4. Dynamic num_kv_splits tuning based on workload
5. Custom HIP kernel for fused MXFP4 attention (JIT compiled)

DeepSeek R1 forward_absorb MLA config:
  total_num_heads  = 128    (query heads before TP split)
  num_heads        = 128 // tp  (query heads per device, tp=4 → 32, tp=8 → 16)
  num_kv_heads     = 1      (shared latent KV head)
  kv_lora_rank     = 512    (latent dim)
  qk_rope_head_dim = 64     (RoPE dim)
  qk_head_dim      = 576    (kv_lora_rank + qk_rope_head_dim, absorbed q/k dim)
  v_head_dim       = 512    (= kv_lora_rank, output dim)
  sm_scale         = 1/sqrt(576)

KV buffer format (forward_absorb):
  - Full 576 dims used as keys (for Q@K^T score computation)
  - First 512 dims (kv_lora_rank) used as values (for output computation)
"""

import torch
import triton
import triton.language as tl
from task import input_t, output_t

import aiter
from aiter.mla import mla_decode_fwd
from aiter import QuantType, dtypes as aiter_dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32

# ---------------------------------------------------------------------------
# Embedded HIP Kernel for MXFP4 MLA Decode (using hip-python)
# ---------------------------------------------------------------------------

# HIP kernel source code - supports q_seq_len up to MAX_Q_SEQ_LEN (e.g., 4)
MLA_MXFP4_HIP_SOURCE = b'''
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>
#include <cmath>

// Constants
constexpr int MXFP4_BLOCK_SIZE = 32;
constexpr int QK_HEAD_DIM = 576;
constexpr int V_HEAD_DIM = 512;
constexpr int NUM_HEADS = 16;
constexpr int WARP_SIZE = 64;
constexpr int MAX_Q_SEQ_LEN = 4;  // Maximum supported q_seq_len

// FP4 E2M1 lookup table
__device__ __constant__ float FP4_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float e8m0_to_float(uint8_t e8m0) {
    int exp = static_cast<int>(e8m0) - 127;
    return exp2f(static_cast<float>(exp));
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

__device__ float block_reduce_max(float val, float* shared_mem, int tid, int block_size) {
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;

    val = warp_reduce_max(val);
    if (lane == 0) shared_mem[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (tid < num_warps) ? shared_mem[lane] : -INFINITY;
        val = warp_reduce_max(val);
        if (lane == 0) shared_mem[0] = val;
    }
    __syncthreads();
    return shared_mem[0];
}

__device__ float block_reduce_sum(float val, float* shared_mem, int tid, int block_size) {
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = (block_size + WARP_SIZE - 1) / WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane == 0) shared_mem[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (tid < num_warps) ? shared_mem[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) shared_mem[0] = val;
    }
    __syncthreads();
    return shared_mem[0];
}

// Kernel that supports q_seq_len up to MAX_Q_SEQ_LEN
// Each block handles one (batch, head) pair, processing all q_seq_len queries
__global__ void mla_mxfp4_decode_kernel(
    const hip_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ kv_mxfp4,
    const uint8_t* __restrict__ kv_scale,
    hip_bfloat16* __restrict__ output,
    const int batch_size,
    const int q_seq_len,
    const int kv_seq_len,
    const float sm_scale
) {
    constexpr int THREADS_PER_BLOCK = 256;

    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    // Shared memory layout:
    // [Q: q_seq_len * QK_HEAD_DIM] [scores: q_seq_len * kv_seq_len] [reduce: WARP_SIZE]
    extern __shared__ char shared_bytes[];
    float* smem_q = reinterpret_cast<float*>(shared_bytes);
    float* smem_scores = smem_q + q_seq_len * QK_HEAD_DIM;
    float* smem_reduce = smem_scores + q_seq_len * kv_seq_len;

    // Load all queries for this (batch, head) into shared memory
    // Q layout: [total_q, num_heads, qk_head_dim] where total_q = batch_size * q_seq_len
    // For this batch: queries are at indices [batch_idx * q_seq_len, (batch_idx+1) * q_seq_len)
    for (int q_idx = 0; q_idx < q_seq_len; q_idx++) {
        const int global_q_idx = batch_idx * q_seq_len + q_idx;
        const int q_offset = (global_q_idx * NUM_HEADS + head_idx) * QK_HEAD_DIM;
        for (int i = tid; i < QK_HEAD_DIM; i += THREADS_PER_BLOCK) {
            smem_q[q_idx * QK_HEAD_DIM + i] = float(q[q_offset + i]);
        }
    }
    __syncthreads();

    const int kv_batch_offset = batch_idx * kv_seq_len * (QK_HEAD_DIM / 2);
    const int scale_batch_offset = batch_idx * kv_seq_len * (QK_HEAD_DIM / MXFP4_BLOCK_SIZE);

    // Process each query token
    for (int q_idx = 0; q_idx < q_seq_len; q_idx++) {
        float* q_ptr = smem_q + q_idx * QK_HEAD_DIM;
        float* scores_ptr = smem_scores + q_idx * kv_seq_len;

        // Phase 1: Compute QK^T scores for this query
        float local_max = -INFINITY;

        for (int kv_idx = tid; kv_idx < kv_seq_len; kv_idx += THREADS_PER_BLOCK) {
            float score = 0.0f;

            const int kv_offset = kv_batch_offset + kv_idx * (QK_HEAD_DIM / 2);
            const int scale_offset = scale_batch_offset + kv_idx * (QK_HEAD_DIM / MXFP4_BLOCK_SIZE);

            for (int block = 0; block < QK_HEAD_DIM / MXFP4_BLOCK_SIZE; block++) {
                float block_scale = e8m0_to_float(kv_scale[scale_offset + block]);

                // #pragma unroll 8
                for (int j = 0; j < MXFP4_BLOCK_SIZE / 2; j++) {
                    uint8_t packed = kv_mxfp4[kv_offset + block * (MXFP4_BLOCK_SIZE / 2) + j];
                    float k_val0 = FP4_E2M1_LUT[packed & 0x0F] * block_scale;
                    float k_val1 = FP4_E2M1_LUT[(packed >> 4) & 0x0F] * block_scale;

                    int d_idx = block * MXFP4_BLOCK_SIZE + j * 2;
                    score += q_ptr[d_idx] * k_val0 + q_ptr[d_idx + 1] * k_val1;
                }
            }

            score *= sm_scale;
            scores_ptr[kv_idx] = score;
            local_max = fmaxf(local_max, score);
        }
        __syncthreads();

        // Phase 2: Softmax for this query
        float max_val = block_reduce_max(local_max, smem_reduce, tid, THREADS_PER_BLOCK);

        float local_sum = 0.0f;
        for (int kv_idx = tid; kv_idx < kv_seq_len; kv_idx += THREADS_PER_BLOCK) {
            float exp_val = expf(scores_ptr[kv_idx] - max_val);
            scores_ptr[kv_idx] = exp_val;
            local_sum += exp_val;
        }
        __syncthreads();

        float sum_val = block_reduce_sum(local_sum, smem_reduce, tid, THREADS_PER_BLOCK);
        float inv_sum = 1.0f / sum_val;

        for (int kv_idx = tid; kv_idx < kv_seq_len; kv_idx += THREADS_PER_BLOCK) {
            scores_ptr[kv_idx] *= inv_sum;
        }
        __syncthreads();

        // Phase 3: Compute attn @ V for this query
        // Output layout: [total_q, num_heads, v_head_dim]
        const int global_q_idx = batch_idx * q_seq_len + q_idx;
        const int out_offset = (global_q_idx * NUM_HEADS + head_idx) * V_HEAD_DIM;

        for (int v_idx = tid; v_idx < V_HEAD_DIM; v_idx += THREADS_PER_BLOCK) {
            float out_val = 0.0f;

            int block_idx = v_idx / MXFP4_BLOCK_SIZE;
            int within_block = v_idx % MXFP4_BLOCK_SIZE;
            int byte_idx = within_block / 2;
            int nibble_idx = within_block % 2;

            for (int kv_idx = 0; kv_idx < kv_seq_len; kv_idx++) {
                float attn_w = scores_ptr[kv_idx];

                const int kv_offset = kv_batch_offset + kv_idx * (QK_HEAD_DIM / 2);
                const int scale_offset = scale_batch_offset + kv_idx * (QK_HEAD_DIM / MXFP4_BLOCK_SIZE);

                float block_scale = e8m0_to_float(kv_scale[scale_offset + block_idx]);
                uint8_t packed = kv_mxfp4[kv_offset + block_idx * (MXFP4_BLOCK_SIZE / 2) + byte_idx];

                float v_val;
                if (nibble_idx == 0) {
                    v_val = FP4_E2M1_LUT[packed & 0x0F];
                } else {
                    v_val = FP4_E2M1_LUT[(packed >> 4) & 0x0F];
                }
                v_val *= block_scale;

                out_val += attn_w * v_val;
            }

            output[out_offset + v_idx] = hip_bfloat16(out_val);
        }
        __syncthreads();
    }
}
'''

# C++ wrapper for PyTorch load_inline compilation
MLA_MXFP4_CPP_SOURCE = r'''
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <ATen/hip/HIPContext.h>

torch::Tensor mla_mxfp4_decode_forward(
    torch::Tensor q,
    torch::Tensor kv_mxfp4,
    torch::Tensor kv_scale,
    int batch_size,
    int q_seq_len,
    int kv_seq_len,
    float sm_scale
) {
    int total_q = batch_size * q_seq_len;

    auto output = torch::empty({total_q, NUM_HEADS, V_HEAD_DIM},
                               torch::TensorOptions()
                                   .dtype(torch::kBFloat16)
                                   .device(q.device()));

    dim3 grid(batch_size, NUM_HEADS);
    dim3 block(256);
    size_t shared_size = sizeof(float) * (q_seq_len * QK_HEAD_DIM + q_seq_len * kv_seq_len + WARP_SIZE);

    hipLaunchKernelGGL(
        mla_mxfp4_decode_kernel,
        grid, block, shared_size, 0,
        reinterpret_cast<const hip_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const uint8_t*>(kv_mxfp4.data_ptr()),
        reinterpret_cast<const uint8_t*>(kv_scale.data_ptr()),
        reinterpret_cast<hip_bfloat16*>(output.data_ptr()),
        batch_size,
        q_seq_len,
        kv_seq_len,
        sm_scale
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mla_mxfp4_decode_forward, "MLA MXFP4 Decode Forward");
}
'''

# Global state for compiled kernel
import ctypes

_hip_kernel = None  # For hip-python direct launch
_torch_hip_module = None  # For PyTorch load_inline


def _hip_check(call_result):
    """Check HIP/HIPRTC call result and raise on error."""
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]

    try:
        from hip import hip, hiprtc
        if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
            raise RuntimeError(f"HIP error: {err}")
        elif isinstance(err, hiprtc.hiprtcResult) and err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
            raise RuntimeError(f"HIPRTC error: {err}")
    except ImportError:
        pass

    return result


def _try_compile_hip_kernel_hiprtc():
    """Compile HIP kernel using hip-python HIPRTC (Method 1)."""
    global _hip_kernel

    import time

    try:
        print("[HIPRTC] Starting compilation...")
        t0 = time.time()

        from hip import hip, hiprtc
        t1 = time.time()
        print(f"[HIPRTC] Import took {t1-t0:.2f}s")

        # Create program
        prog = _hip_check(hiprtc.hiprtcCreateProgram(
            MLA_MXFP4_HIP_SOURCE,
            b"mla_mxfp4_decode",
            0, [], []
        ))
        t2 = time.time()
        print(f"[HIPRTC] hiprtcCreateProgram took {t2-t1:.2f}s")

        # Compile - this is usually the slow part
        cflags = [b"--offload-arch=gfx950"]
        print(f"[HIPRTC] Starting hiprtcCompileProgram with {cflags}...")
        err, = hiprtc.hiprtcCompileProgram(prog, len(cflags), cflags)
        t3 = time.time()
        print(f"[HIPRTC] hiprtcCompileProgram took {t3-t2:.2f}s")

        if err != hiprtc.hiprtcResult.HIPRTC_SUCCESS:
            log_size = _hip_check(hiprtc.hiprtcGetProgramLogSize(prog))
            log = bytearray(log_size)
            _hip_check(hiprtc.hiprtcGetProgramLog(prog, log))
            raise RuntimeError(f"HIPRTC compilation failed: {log.decode()}")

        # Get compiled code
        code_size = _hip_check(hiprtc.hiprtcGetCodeSize(prog))
        code = bytearray(code_size)
        _hip_check(hiprtc.hiprtcGetCode(prog, code))
        t4 = time.time()
        print(f"[HIPRTC] Getting code took {t4-t3:.2f}s, code size: {code_size} bytes")

        # Load module and get kernel function
        print(f"[HIPRTC] Starting hipModuleLoadData...")
        hip_module = _hip_check(hip.hipModuleLoadData(code))
        t5 = time.time()
        print(f"[HIPRTC] hipModuleLoadData took {t5-t4:.2f}s")

        _hip_kernel = _hip_check(hip.hipModuleGetFunction(hip_module, b"mla_mxfp4_decode_kernel"))
        t6 = time.time()
        print(f"[HIPRTC] hipModuleGetFunction took {t6-t5:.2f}s")

        # Cleanup program
        _hip_check(hiprtc.hiprtcDestroyProgram(prog.createRef()))

        print(f"[HIPRTC] Total compilation time: {t6-t0:.2f}s")
        return True

    except ImportError as e:
        print(f"[HIPRTC] hip-python not available: {e}")
        return False
    except Exception as e:
        print(f"[HIPRTC] Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _try_compile_hip_kernel_torch():
    """Compile HIP kernel using PyTorch load_inline (Method 2)."""
    global _torch_hip_module

    import time

    try:
        print("[PyTorch] Starting compilation...")
        t0 = time.time()

        from torch.utils.cpp_extension import load_inline
        import os

        # Check if we're on ROCm
        if not torch.cuda.is_available():
            print("[PyTorch] CUDA/ROCm not available")
            return False

        # Check for HIP/ROCm
        if not hasattr(torch.version, 'hip') or torch.version.hip is None:
            print("[PyTorch] Not running on HIP/ROCm")
            return False

        rocm_home = os.environ.get('ROCM_HOME', '/opt/rocm')
        t1 = time.time()
        print(f"[PyTorch] Setup took {t1-t0:.2f}s")

        # Combine kernel source with C++ wrapper
        # Convert bytes to string for load_inline
        kernel_source = MLA_MXFP4_HIP_SOURCE.decode('utf-8') + '\n' + MLA_MXFP4_CPP_SOURCE

        print(f"[PyTorch] Starting load_inline...")
        os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'
        os.environ['MAX_JOBS'] = '4'
        _torch_hip_module = load_inline(
            name='mla_mxfp4_hip_torch',
            cpp_sources='',
            cuda_sources=[kernel_source],
            extra_cflags=['-O3'],
            extra_cuda_cflags=['-O3', '--offload-arch=gfx950'],
            extra_include_paths=[f'{rocm_home}/include'],
            verbose=True,  # Enable verbose to see what's happening
        )
        t2 = time.time()
        print(f"[PyTorch] load_inline took {t2-t1:.2f}s")

        print(f"[PyTorch] Total compilation time: {t2-t0:.2f}s")
        return True

    except Exception as e:
        print(f"[PyTorch] Compilation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _try_compile_hip_kernel():
    """Try both compilation methods, preferring hip-python."""
    # Try hip-python first (more direct, no PyTorch overhead)
    # if _try_compile_hip_kernel_hiprtc():
    #     return True
    # Fall back to PyTorch load_inline
    if _try_compile_hip_kernel_torch():
        return True
    return False


def _launch_hip_kernel_hiprtc(
    hip_kernel,
    q: torch.Tensor,
    kv_mxfp4: torch.Tensor,
    kv_scale: torch.Tensor,
    output: torch.Tensor,
    batch_size: int,
    q_seq_len: int,
    kv_seq_len: int,
    sm_scale: float,
):
    """Launch the compiled HIP kernel using hip-python."""
    from hip import hip

    # Grid and block dimensions
    grid = hip.dim3(x=batch_size, y=16)  # NUM_HEADS = 16
    block = hip.dim3(x=256)

    # Shared memory size: Q storage + scores storage + reduction buffer
    shared_mem_bytes = 4 * (q_seq_len * 576 + q_seq_len * kv_seq_len + 64)

    # Launch kernel
    _hip_check(
        hip.hipModuleLaunchKernel(
            hip_kernel,
            *grid,
            *block,
            sharedMemBytes=shared_mem_bytes,
            kernelParams=None,
            extra=(
                q.data_ptr(),                    # q pointer
                kv_mxfp4.data_ptr(),             # kv_mxfp4 pointer
                kv_scale.data_ptr(),             # kv_scale pointer
                output.data_ptr(),               # output pointer
                ctypes.c_int(batch_size),        # batch_size
                ctypes.c_int(q_seq_len),         # q_seq_len
                ctypes.c_int(kv_seq_len),        # kv_seq_len
                ctypes.c_float(sm_scale),        # sm_scale
            )
        )
    )

    # Synchronize to ensure kernel completion
    _hip_check(hip.hipDeviceSynchronize())


def _launch_hip_kernel(
    q: torch.Tensor,
    kv_mxfp4: torch.Tensor,
    kv_scale: torch.Tensor,
    output: torch.Tensor,
    batch_size: int,
    q_seq_len: int,
    kv_seq_len: int,
    sm_scale: float,
):
    """Launch the HIP kernel using the appropriate method."""
    global _hip_kernel
    if _hip_kernel is not None:
        _launch_hip_kernel_hiprtc(
            _hip_kernel, q, kv_mxfp4, kv_scale, output,
            batch_size, q_seq_len, kv_seq_len, sm_scale
        )
    else:
        # Use PyTorch module
        global _torch_hip_module
        assert _torch_hip_module is not None
        result = _torch_hip_module.forward(
            q, kv_mxfp4, kv_scale,
            batch_size, q_seq_len, kv_seq_len, sm_scale
        )
        output.copy_(result)


# Attempt compilation at module load (disabled by default)
# Uncomment the line below to enable JIT compilation of HIP kernel
HAS_HIP_KERNEL = _try_compile_hip_kernel()

# ---------------------------------------------------------------------------
# Constants (DeepSeek R1 forward_absorb path)
# ---------------------------------------------------------------------------
TOTAL_NUM_HEADS = 128
NUM_KV_HEADS = 1
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM = KV_LORA_RANK  # 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
MXFP4_BLOCK_SIZE = 32

PAGE_SIZE = 1

FP8_DTYPE = aiter_dtypes.fp8

# QKV dtype for custom_kernel dispatch: "bf16", "fp8", "mxfp4_dequant", or "mxfp4_native"
QKV_DTYPE = "mxfp4_native"


# ---------------------------------------------------------------------------
# Dynamic num_kv_splits tuning
# ---------------------------------------------------------------------------

def get_optimal_num_kv_splits(batch_size: int, kv_seq_len: int) -> int:
    """
    Heuristic for optimal num_kv_splits based on batch size and KV sequence length.

    Trade-offs:
    - More splits = better parallelism for long sequences
    - Fewer splits = less reduction overhead for short sequences / small batches

    Performance observations:
    - batch_size=4: bf16 is fastest (81-90us), minimize splits aggressively
    - For small batches, minimize splits to reduce reduction overhead
    """
    if batch_size <= 4:
        # Very small batches: minimize reduction overhead aggressively
        # Target: get bf16 from 86.9µs to 81µs
        if kv_seq_len <= 1024:
            return 2  # Minimal splits - reduction overhead dominates
        elif kv_seq_len <= 2048:
            return 4
        elif kv_seq_len <= 4096:
            return 4
        else:
            return 8  # More parallelism for very long sequences
    elif batch_size <= 16:
        if kv_seq_len <= 1024:
            return 8
        elif kv_seq_len <= 4096:
            return 16
        else:
            return 24
    elif batch_size <= 32:
        if kv_seq_len <= 1024:
            return 16
        else:
            return 24
    elif batch_size <= 64:
        if kv_seq_len <= 1024:
            return 24
        else:
            return 32
    else:
        # Large batches benefit from more parallelism
        if kv_seq_len <= 2048:
            return 32
        else:
            return 48


# ---------------------------------------------------------------------------
# FP8 quantization helper (per-tensor, sglang style)
# ---------------------------------------------------------------------------

def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-tensor FP8 quantization. Returns (fp8_tensor, scale)."""
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8_tensor, scale.to(torch.float32).reshape(1)


# ---------------------------------------------------------------------------
# MXFP4 dequantization helpers (optimized)
# ---------------------------------------------------------------------------

@torch.compile(mode="reduce-overhead", fullgraph=True)
def _dequant_mxfp4_core(
    fp4_data_2d: torch.Tensor,
    scale_e8m0: torch.Tensor,
    num_rows: int,
    num_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """
    Core dequantization logic - compiled for better performance.
    """
    float_vals = mxfp4_to_f32(fp4_data_2d)
    scale_f32 = e8m0_to_f32(scale_e8m0)[:num_rows, :num_blocks]
    float_vals_blocked = float_vals.view(num_rows, num_blocks, block_size)
    return float_vals_blocked * scale_f32.unsqueeze(-1)


def dequantize_mxfp4(
    fp4_data: torch.Tensor,
    scale_e8m0: torch.Tensor,
    orig_shape: tuple,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Dequantize MXFP4 tensor using aiter utilities (optimized).

    Args:
        fp4_data:   packed FP4 data, shape [B, M, N//2] in fp4x2 or uint8
        scale_e8m0: E8M0 block scale factors (possibly padded) in fp8_e8m0
        orig_shape: original (B, M, N) for reshaping
        dtype:      output dtype

    Returns:
        Dequantized tensor of shape orig_shape.
    """
    B, M, N = orig_shape
    num_rows = B * M
    num_blocks = N // MXFP4_BLOCK_SIZE

    fp4_data_2d = fp4_data.view(num_rows, N // 2)

    # Inline dequantization to avoid function call overhead
    float_vals = mxfp4_to_f32(fp4_data_2d)
    scale_f32 = e8m0_to_f32(scale_e8m0)[:num_rows, :num_blocks]
    float_vals_blocked = float_vals.view(num_rows, num_blocks, MXFP4_BLOCK_SIZE)
    scaled = float_vals_blocked * scale_f32.unsqueeze(-1)

    return scaled.view(B, M, N).to(dtype)


def dequantize_mxfp4_to_fp8(
    fp4_data: torch.Tensor,
    scale_e8m0: torch.Tensor,
    orig_shape: tuple,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dequantize MXFP4 tensor to FP8 format for use with a8w8 kernel (optimized).

    This is more efficient than dequantizing to bf16 because:
    1. FP8 uses half the memory bandwidth of bf16
    2. The a8w8 kernel is faster than a16w16

    Returns:
        (fp8_tensor, scale) for use with mla_decode_fwd
    """
    B, M, N = orig_shape
    num_rows = B * M
    num_blocks = N // MXFP4_BLOCK_SIZE

    # Single-pass dequantization
    fp4_data_2d = fp4_data.view(num_rows, N // 2)
    float_vals = mxfp4_to_f32(fp4_data_2d)
    scale_f32 = e8m0_to_f32(scale_e8m0)[:num_rows, :num_blocks]

    float_vals_blocked = float_vals.view(num_rows, num_blocks, MXFP4_BLOCK_SIZE)
    scaled = float_vals_blocked * scale_f32.unsqueeze(-1)
    dequant_f32 = scaled.view(B, M, N)

    # Quantize to FP8 (per-tensor) - fused max and scale
    finfo = torch.finfo(FP8_DTYPE)
    amax = dequant_f32.abs().amax().clamp(min=1e-12)
    fp8_scale = amax / finfo.max
    fp8_tensor = (dequant_f32 / fp8_scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)

    return fp8_tensor, fp8_scale.to(torch.float32).reshape(1)


# ---------------------------------------------------------------------------
# Persistent mode metadata helpers
# ---------------------------------------------------------------------------

def _make_mla_decode_metadata(
    batch_size: int,
    max_q_len: int,
    nhead: int,
    nhead_kv: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    num_kv_splits: int,
):
    """Allocate and populate work buffers for persistent mla_decode_fwd."""
    info = get_mla_metadata_info_v1(
        batch_size, max_q_len, nhead, q_dtype, kv_dtype,
        is_sparse=False, fast_mode=False,
        num_kv_splits=num_kv_splits, intra_batch_mode=True,
    )
    work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    (work_metadata, work_indptr, work_info_set,
     reduce_indptr, reduce_final_map, reduce_partial_map) = work

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_len,
        nhead // nhead_kv,
        nhead_kv,
        True,
        work_metadata, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        page_size=PAGE_SIZE,
        kv_granularity=max(PAGE_SIZE, 16),
        max_seqlen_qo=max_q_len,
        uni_seqlen_qo=max_q_len,
        fast_mode=False,
        max_split_per_batch=num_kv_splits,
        intra_batch_mode=True,
        dtype_q=q_dtype,
        dtype_kv=kv_dtype,
    )

    return {
        "work_meta_data": work_metadata,
        "work_indptr": work_indptr,
        "work_info_set": work_info_set,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
    }


# ---------------------------------------------------------------------------
# Aiter MLA decode kernel wrapper with dynamic splits
# ---------------------------------------------------------------------------

def _aiter_mla_decode(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    config: dict,
    q_scale: torch.Tensor | None = None,
    kv_scale: torch.Tensor | None = None,
    num_kv_splits: int | None = None,
) -> torch.Tensor:
    """
    MLA decode attention using aiter persistent-mode kernel.

    Supports:
      - fp8 Q + fp8 KV (a8w8) — fastest on MI355X
      - bf16 Q + bf16 KV (a16w16) — highest precision
      - bf16 Q + fp8 KV (a16w8) — mixed precision
    """
    batch_size = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dq = config["qk_head_dim"]
    dv = config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    kv_seq_len = config["kv_seq_len"]

    if num_kv_splits is None:
        num_kv_splits = get_optimal_num_kv_splits(batch_size, kv_seq_len)

    total_kv_len = int(kv_indptr[-1].item())
    kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")

    kv_buffer_4d = kv_buffer.view(kv_buffer.shape[0], PAGE_SIZE, nkv, kv_buffer.shape[-1])

    max_q_len = q_seq_len
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

    meta = _make_mla_decode_metadata(
        batch_size, max_q_len, nq, nkv,
        q.dtype, kv_buffer.dtype,
        qo_indptr, kv_indptr, kv_last_page_len,
        num_kv_splits=num_kv_splits,
    )

    o = torch.empty((q.shape[0], nq, dv), dtype=torch.bfloat16, device="cuda")
    mla_decode_fwd(
        q.view(-1, nq, dq),
        kv_buffer_4d,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        max_q_len,
        page_size=PAGE_SIZE,
        nhead_kv=nkv,
        sm_scale=SM_SCALE,
        logit_cap=0.0,
        num_kv_splits=num_kv_splits,
        q_scale=q_scale,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        **meta,
    )
    return o


# ---------------------------------------------------------------------------
# Ultra-fast path for small batches - bypasses aiter overhead
# ---------------------------------------------------------------------------

@torch.compile(mode="max-autotune", fullgraph=True)
def _fast_mla_attention_compiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    batch_size: int,
    nq: int,
    q_seq_len: int,
    kv_seq_len: int,
    dv: int,
) -> torch.Tensor:
    """
    Compiled fast attention for small batches.
    Fuses all operations into a single kernel.
    """
    total_q = batch_size * q_seq_len

    # Reshape for batched attention
    q_batched = q.view(batch_size, q_seq_len, nq, -1).transpose(1, 2)
    k_batched = k.view(batch_size, kv_seq_len, 1, -1).transpose(1, 2)
    v_batched = v.view(batch_size, kv_seq_len, 1, dv).transpose(1, 2)

    # Expand for MQA
    k_batched = k_batched.expand(batch_size, nq, kv_seq_len, -1)
    v_batched = v_batched.expand(batch_size, nq, kv_seq_len, dv)

    # SDPA handles everything efficiently
    out = torch.nn.functional.scaled_dot_product_attention(
        q_batched, k_batched, v_batched,
        scale=SM_SCALE,
        is_causal=False,
    )

    return out.transpose(1, 2).reshape(total_q, nq, dv)


def _fast_mla_attention(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """
    Fast MLA attention that bypasses aiter kernel overhead.
    Uses torch SDPA which is highly optimized and has lower launch overhead.
    """
    batch_size = config["batch_size"]
    nq = config["num_heads"]
    dv = config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    kv_seq_len = config["kv_seq_len"]

    # KV buffer: (total_kv, 1, 576)
    k = kv_buffer  # Full 576 dims for keys
    v = kv_buffer[:, :, :dv]  # First 512 dims for values

    out = _fast_mla_attention_compiled(
        q.to(torch.float32),
        k.to(torch.float32),
        v.to(torch.float32),
        batch_size, nq, q_seq_len, kv_seq_len, dv
    )

    return out.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# PyTorch-based MLA attention (fallback / MXFP4 path) - optimized with SDPA
# ---------------------------------------------------------------------------

def _pytorch_mla_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """
    PyTorch MLA attention using SDPA (Scaled Dot-Product Attention).

    Uses torch.nn.functional.scaled_dot_product_attention for optimized compute.
    Handles the MQA pattern (1 KV head shared across all query heads).

    q: (total_q, num_heads, qk_head_dim)
    k: (total_kv, 1, qk_head_dim)
    v: (total_kv, 1, v_head_dim)
    """
    batch_size = config["batch_size"]
    nq = config["num_heads"]
    dv = config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    kv_seq_len = config["kv_seq_len"]

    # For decode with uniform sequence lengths, we can batch process
    # Reshape for batched attention: (batch, heads, seq_len, dim)
    total_q = q.shape[0]
    total_kv = k.shape[0]

    # Reshape q: (total_q, nq, 576) -> (batch, q_seq_len, nq, 576) -> (batch, nq, q_seq_len, 576)
    q_batched = q.view(batch_size, q_seq_len, nq, -1).transpose(1, 2)

    # Reshape k: (total_kv, 1, 576) -> (batch, kv_seq_len, 1, 576) -> (batch, 1, kv_seq_len, 576)
    k_batched = k.view(batch_size, kv_seq_len, 1, -1).transpose(1, 2)

    # Reshape v: (total_kv, 1, 576) -> use only first 512 dims
    v_sliced = v[:, :, :dv]  # (total_kv, 1, 512)
    v_batched = v_sliced.view(batch_size, kv_seq_len, 1, dv).transpose(1, 2)

    # Expand K and V for MQA: (batch, 1, kv_seq_len, dim) -> (batch, nq, kv_seq_len, dim)
    k_batched = k_batched.expand(batch_size, nq, kv_seq_len, -1)
    v_batched = v_batched.expand(batch_size, nq, kv_seq_len, dv)

    # Use SDPA for optimized attention (uses Flash Attention when available)
    # is_causal=True for decode (though for q_seq_len=1, it doesn't matter)
    out = torch.nn.functional.scaled_dot_product_attention(
        q_batched.to(torch.float32),
        k_batched.to(torch.float32),
        v_batched.to(torch.float32),
        scale=SM_SCALE,
        is_causal=(q_seq_len > 1),
    )

    # Reshape output: (batch, nq, q_seq_len, dv) -> (batch, q_seq_len, nq, dv) -> (total_q, nq, dv)
    out = out.transpose(1, 2).reshape(total_q, nq, dv).to(torch.bfloat16)

    return out


# ---------------------------------------------------------------------------
# Dispatcher: select kernel based on QKV_DTYPE
# ---------------------------------------------------------------------------

def custom_kernel(data: input_t) -> output_t:
    """Dispatch to the appropriate kernel based on QKV_DTYPE."""
    global HAS_HIP_KERNEL

    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]
    qkv_type = QKV_DTYPE
    if batch_size == 4:
      if kv_seq_len <= 1024:
        qkv_type = "bf16"
      else:
        qkv_type = "mxfp4_native"
    elif batch_size == 32:
      if kv_seq_len <= 1024:
        qkv_type = "mxfp4_native"
        HAS_HIP_KERNEL = False
      else:
        qkv_type = "mxfp4_dequant"
    elif batch_size == 64:
      if kv_seq_len <= 1024:
        qkv_type = "mxfp4_native"
        HAS_HIP_KERNEL = False
      else:
        qkv_type = "mxfp4_dequant"
    elif batch_size == 256:
      if kv_seq_len <= 1024:
        qkv_type = "mxfp4_dequant"
      else:
        qkv_type = "mxfp4_dequant"

    if qkv_type == "fp8":
        return custom_kernel_fp8(data)
    elif qkv_type == "bf16":
        return custom_kernel_bf16(data)
    elif qkv_type == "mxfp4_dequant":
        return custom_kernel_mxfp4_dequant(data)
    elif qkv_type == "mxfp4_native":
        return custom_kernel_mxfp4_native(data)
    else:
        raise ValueError(f"Invalid QKV_DTYPE: {qkv_type}")


# ---------------------------------------------------------------------------
# FP8 Q + FP8 KV — using aiter's a8w8 persistent MLA kernel
# ---------------------------------------------------------------------------

def custom_kernel_fp8(data: input_t) -> output_t:
    """
    MLA decode using aiter's a8w8 persistent kernel (fp8 Q + fp8 KV).

    For small batches, uses fast SDPA path to avoid aiter overhead.
    For larger batches, uses aiter's a8w8 kernel for better throughput.
    """
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]

    # Use fast path for small batches to avoid aiter overhead
    if batch_size <= 4:
        kv_buffer_bf16 = kv_data["bf16"]
        return _fast_mla_attention(q, kv_buffer_bf16, config)

    q_fp8, q_scale = quantize_fp8(q)
    kv_buffer_fp8, kv_scale_fp8 = kv_data["fp8"]

    return _aiter_mla_decode(
        q_fp8, kv_buffer_fp8, qo_indptr, kv_indptr, config,
        q_scale=q_scale, kv_scale=kv_scale_fp8,
    )


# ---------------------------------------------------------------------------
# BF16 Q + BF16 KV — using aiter's a16w16 persistent MLA kernel
# ---------------------------------------------------------------------------

def custom_kernel_bf16(data: input_t) -> output_t:
    """
    MLA decode using bf16 KV cache.

    For small batches with short KV sequences, uses the fast SDPA path which
    bypasses aiter overhead. For larger batches or long sequences, uses aiter's a16w16 kernel.
    """
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]
    kv_buffer_bf16 = kv_data["bf16"]

    # Use fast path ONLY for small batches AND short KV sequences
    # For long KV sequences (8k), SDPA's MQA expansion is slow - use aiter instead
    if batch_size <= 4 and kv_seq_len <= 2048:
        return _fast_mla_attention(q, kv_buffer_bf16, config)

    return _aiter_mla_decode(
        q, kv_buffer_bf16, qo_indptr, kv_indptr, config,
        q_scale=None, kv_scale=None,
    )


# ---------------------------------------------------------------------------
# MXFP4 KV with dequantization -> aiter MLA kernel (optimized)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fused Triton Kernel: MXFP4 Dequant + MLA Attention (Strategy A)
# ---------------------------------------------------------------------------
# This kernel fuses MXFP4 dequantization with attention computation:
# 1. Loads MXFP4 KV tiles (fp4x2 packed) from HBM
# 2. Loads E8M0 scale factors from HBM
# 3. Unpacks and dequantizes in registers/shared memory
# 4. Computes QK^T attention scores
# 5. Applies softmax
# 6. Computes attention @ V
# 7. Writes only final output to HBM
# This eliminates the intermediate bf16/fp8 buffer write-read roundtrip.
# ---------------------------------------------------------------------------

# FP4 E2M1 lookup table: values [0, 0.5, 1, 1.5, 2, 3, 4, 6]
# Low nibble (even index), high nibble (odd index)
FP4_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                   -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


@triton.jit
def _unpack_fp4_to_f32(packed_val):
    """Unpack a single fp4x2 byte into two f32 values."""
    # Low nibble (even index)
    low_nibble = packed_val & 0x0F
    # High nibble (odd index)
    high_nibble = (packed_val >> 4) & 0x0F
    return low_nibble, high_nibble


@triton.jit
def _fp4_nibble_to_f32(nibble):
    """Convert a 4-bit FP4 E2M1 value to float32."""
    # FP4 E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit
    # Values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (and negatives)
    sign = (nibble >> 3) & 1
    magnitude_idx = nibble & 0x07

    # Lookup table for magnitudes
    # Index: 0->0, 1->0.5, 2->1, 3->1.5, 4->2, 5->3, 6->4, 7->6
    val = tl.where(magnitude_idx == 0, 0.0,
           tl.where(magnitude_idx == 1, 0.5,
           tl.where(magnitude_idx == 2, 1.0,
           tl.where(magnitude_idx == 3, 1.5,
           tl.where(magnitude_idx == 4, 2.0,
           tl.where(magnitude_idx == 5, 3.0,
           tl.where(magnitude_idx == 6, 4.0, 6.0)))))))

    # Apply sign
    return tl.where(sign == 1, -val, val)


@triton.jit
def _e8m0_to_f32(e8m0_val):
    """Convert E8M0 scale to float32. E8M0 is exponent-only: 2^(val - 127)."""
    # E8M0: 8-bit unsigned exponent, bias 127
    # value = 2^(e8m0 - 127)
    exponent = e8m0_val.to(tl.int32) - 127
    return tl.math.pow(2.0, exponent.to(tl.float32))


@triton.jit
def _fused_mxfp4_mla_attention_kernel(
    # Query tensor (bf16)
    Q_ptr,
    # MXFP4 KV cache (packed fp4x2 as uint8)
    KV_mxfp4_ptr,
    # E8M0 scale factors (as uint8)
    KV_scale_ptr,
    # Output tensor
    O_ptr,
    # Dimensions
    batch_size,
    num_heads,
    q_seq_len,
    kv_seq_len,
    qk_head_dim,  # 576
    v_head_dim,   # 512
    # Strides for Q: (total_q, num_heads, qk_head_dim)
    stride_q_tok,
    stride_q_head,
    stride_q_dim,
    # Strides for KV_mxfp4: (total_kv, 1, qk_head_dim // 2)
    stride_kv_tok,
    stride_kv_head,
    stride_kv_dim,
    # Strides for KV_scale: (total_kv * 1, num_blocks)
    stride_scale_row,
    stride_scale_block,
    # Strides for O: (total_q, num_heads, v_head_dim)
    stride_o_tok,
    stride_o_head,
    stride_o_dim,
    # Scaling factor
    sm_scale,
    # Block sizes - must be power of 2
    BLOCK_KV: tl.constexpr,
):
    """
    Fused MXFP4 dequantization + MLA attention kernel.

    Simplified version that avoids non-power-of-2 tensor shapes.
    Each program handles one (batch_idx, head_idx) pair.
    """
    # Program IDs
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    # Query offset for this batch and head
    q_offset = batch_idx * q_seq_len * stride_q_tok + head_idx * stride_q_head

    # KV offset for this batch (KV has 1 head, shared across all query heads)
    kv_base = batch_idx * kv_seq_len * stride_kv_tok
    scale_base = batch_idx * kv_seq_len * stride_scale_row

    # Output offset
    o_offset = batch_idx * q_seq_len * stride_o_tok + head_idx * stride_o_head

    # Number of MXFP4 blocks per KV row (576 / 32 = 18)
    num_mxfp4_blocks: tl.constexpr = 18

    # Pass 1: Compute max of attention scores
    global_max = tl.full([], -1e30, dtype=tl.float32)

    for kv_i in range(kv_seq_len):
        kv_offset = kv_base + kv_i * stride_kv_tok
        scale_offset_base = scale_base + kv_i * stride_scale_row

        # Compute dot product Q @ K for this KV position
        score = tl.zeros([], dtype=tl.float32)

        # Process each MXFP4 block (32 elements = 16 bytes)
        for block_idx in tl.static_range(num_mxfp4_blocks):
            block_start = block_idx * 32
            scale_offset = scale_offset_base + block_idx * stride_scale_block

            # Load E8M0 scale and convert inline
            scale_e8m0 = tl.load(KV_scale_ptr + scale_offset)
            # E8M0 to f32: 2^(val - 127) using exp2
            scale_exp = scale_e8m0.to(tl.float32) - 127.0
            scale_f32 = tl.math.exp2(scale_exp)

            # Process 16 bytes (32 fp4 values)
            for byte_idx in tl.static_range(16):
                packed_offset = kv_offset + (block_start // 2 + byte_idx) * stride_kv_dim
                packed_byte = tl.load(KV_mxfp4_ptr + packed_offset)

                # Unpack two fp4 values (inline)
                low_nibble = packed_byte & 0x0F
                high_nibble = (packed_byte >> 4) & 0x0F

                # Convert FP4 nibble to f32 (inline) - simplified lookup
                # FP4 E2M1: sign(1) + exp(2) + mantissa(1)
                # Magnitudes: 0->0, 1->0.5, 2->1, 3->1.5, 4->2, 5->3, 6->4, 7->6
                sign_0 = (low_nibble >> 3) & 1
                mag_0 = low_nibble & 0x07
                k_mag_0 = tl.where(mag_0 == 0, 0.0,
                          tl.where(mag_0 == 1, 0.5,
                          tl.where(mag_0 == 2, 1.0,
                          tl.where(mag_0 == 3, 1.5,
                          tl.where(mag_0 == 4, 2.0,
                          tl.where(mag_0 == 5, 3.0,
                          tl.where(mag_0 == 6, 4.0, 6.0)))))))
                k_val_0 = tl.where(sign_0 == 1, -k_mag_0, k_mag_0) * scale_f32

                sign_1 = (high_nibble >> 3) & 1
                mag_1 = high_nibble & 0x07
                k_mag_1 = tl.where(mag_1 == 0, 0.0,
                          tl.where(mag_1 == 1, 0.5,
                          tl.where(mag_1 == 2, 1.0,
                          tl.where(mag_1 == 3, 1.5,
                          tl.where(mag_1 == 4, 2.0,
                          tl.where(mag_1 == 5, 3.0,
                          tl.where(mag_1 == 6, 4.0, 6.0)))))))
                k_val_1 = tl.where(sign_1 == 1, -k_mag_1, k_mag_1) * scale_f32

                # Load corresponding Q values and accumulate
                dim_0 = block_start + byte_idx * 2
                dim_1 = block_start + byte_idx * 2 + 1

                q_val_0 = tl.load(Q_ptr + q_offset + dim_0 * stride_q_dim).to(tl.float32)
                score += q_val_0 * k_val_0

                if dim_1 < 576:
                    q_val_1 = tl.load(Q_ptr + q_offset + dim_1 * stride_q_dim).to(tl.float32)
                    score += q_val_1 * k_val_1

        # Apply scale and update max
        score = score * sm_scale
        global_max = tl.maximum(global_max, score)

    # Pass 2: Compute softmax normalization factor
    sum_exp = tl.zeros([], dtype=tl.float32)

    for kv_i in range(kv_seq_len):
        kv_offset = kv_base + kv_i * stride_kv_tok
        scale_offset_base = scale_base + kv_i * stride_scale_row

        # Recompute attention score
        score = tl.zeros([], dtype=tl.float32)
        for block_idx in tl.static_range(num_mxfp4_blocks):
            block_start = block_idx * 32
            scale_offset = scale_offset_base + block_idx * stride_scale_block
            scale_e8m0 = tl.load(KV_scale_ptr + scale_offset)
            scale_exp = scale_e8m0.to(tl.float32) - 127.0
            scale_f32 = tl.math.exp2(scale_exp)

            for byte_idx in tl.static_range(16):
                packed_offset = kv_offset + (block_start // 2 + byte_idx) * stride_kv_dim
                packed_byte = tl.load(KV_mxfp4_ptr + packed_offset)
                low_nibble = packed_byte & 0x0F
                high_nibble = (packed_byte >> 4) & 0x0F

                sign_0 = (low_nibble >> 3) & 1
                mag_0 = low_nibble & 0x07
                k_mag_0 = tl.where(mag_0 == 0, 0.0,
                          tl.where(mag_0 == 1, 0.5,
                          tl.where(mag_0 == 2, 1.0,
                          tl.where(mag_0 == 3, 1.5,
                          tl.where(mag_0 == 4, 2.0,
                          tl.where(mag_0 == 5, 3.0,
                          tl.where(mag_0 == 6, 4.0, 6.0)))))))
                k_val_0 = tl.where(sign_0 == 1, -k_mag_0, k_mag_0) * scale_f32

                sign_1 = (high_nibble >> 3) & 1
                mag_1 = high_nibble & 0x07
                k_mag_1 = tl.where(mag_1 == 0, 0.0,
                          tl.where(mag_1 == 1, 0.5,
                          tl.where(mag_1 == 2, 1.0,
                          tl.where(mag_1 == 3, 1.5,
                          tl.where(mag_1 == 4, 2.0,
                          tl.where(mag_1 == 5, 3.0,
                          tl.where(mag_1 == 6, 4.0, 6.0)))))))
                k_val_1 = tl.where(sign_1 == 1, -k_mag_1, k_mag_1) * scale_f32

                dim_0 = block_start + byte_idx * 2
                dim_1 = block_start + byte_idx * 2 + 1
                q_val_0 = tl.load(Q_ptr + q_offset + dim_0 * stride_q_dim).to(tl.float32)
                score += q_val_0 * k_val_0
                if dim_1 < 576:
                    q_val_1 = tl.load(Q_ptr + q_offset + dim_1 * stride_q_dim).to(tl.float32)
                    score += q_val_1 * k_val_1

        score = score * sm_scale
        sum_exp += tl.exp(score - global_max)

    # Pass 3: For each output dimension, compute weighted sum of V
    num_v_blocks: tl.constexpr = 16  # 512 / 32 = 16

    for v_block_idx in tl.static_range(num_v_blocks):
        for v_byte_idx in tl.static_range(16):
            for v_nibble in tl.static_range(2):  # 0=low, 1=high
                out_d = v_block_idx * 32 + v_byte_idx * 2 + v_nibble
                if out_d < 512:
                    out_acc = tl.zeros([], dtype=tl.float32)

                    for kv_i in range(kv_seq_len):
                        kv_offset = kv_base + kv_i * stride_kv_tok
                        scale_offset_base = scale_base + kv_i * stride_scale_row

                        # Recompute attention score
                        score = tl.zeros([], dtype=tl.float32)
                        for block_idx in tl.static_range(num_mxfp4_blocks):
                            block_start = block_idx * 32
                            scale_offset = scale_offset_base + block_idx * stride_scale_block
                            scale_e8m0 = tl.load(KV_scale_ptr + scale_offset)
                            scale_exp = scale_e8m0.to(tl.float32) - 127.0
                            scale_f32 = tl.math.exp2(scale_exp)

                            for byte_idx in tl.static_range(16):
                                packed_offset = kv_offset + (block_start // 2 + byte_idx) * stride_kv_dim
                                packed_byte = tl.load(KV_mxfp4_ptr + packed_offset)
                                low_nibble = packed_byte & 0x0F
                                high_nibble = (packed_byte >> 4) & 0x0F

                                sign_0 = (low_nibble >> 3) & 1
                                mag_0 = low_nibble & 0x07
                                k_mag_0 = tl.where(mag_0 == 0, 0.0,
                                          tl.where(mag_0 == 1, 0.5,
                                          tl.where(mag_0 == 2, 1.0,
                                          tl.where(mag_0 == 3, 1.5,
                                          tl.where(mag_0 == 4, 2.0,
                                          tl.where(mag_0 == 5, 3.0,
                                          tl.where(mag_0 == 6, 4.0, 6.0)))))))
                                k_val_0 = tl.where(sign_0 == 1, -k_mag_0, k_mag_0) * scale_f32

                                sign_1 = (high_nibble >> 3) & 1
                                mag_1 = high_nibble & 0x07
                                k_mag_1 = tl.where(mag_1 == 0, 0.0,
                                          tl.where(mag_1 == 1, 0.5,
                                          tl.where(mag_1 == 2, 1.0,
                                          tl.where(mag_1 == 3, 1.5,
                                          tl.where(mag_1 == 4, 2.0,
                                          tl.where(mag_1 == 5, 3.0,
                                          tl.where(mag_1 == 6, 4.0, 6.0)))))))
                                k_val_1 = tl.where(sign_1 == 1, -k_mag_1, k_mag_1) * scale_f32

                                dim_0 = block_start + byte_idx * 2
                                dim_1 = block_start + byte_idx * 2 + 1
                                q_val_0 = tl.load(Q_ptr + q_offset + dim_0 * stride_q_dim).to(tl.float32)
                                score += q_val_0 * k_val_0
                                if dim_1 < 576:
                                    q_val_1 = tl.load(Q_ptr + q_offset + dim_1 * stride_q_dim).to(tl.float32)
                                    score += q_val_1 * k_val_1

                        score = score * sm_scale
                        attn_weight = tl.exp(score - global_max) / sum_exp

                        # Get V value for this output dimension
                        v_scale_offset = scale_offset_base + v_block_idx * stride_scale_block
                        v_scale_e8m0 = tl.load(KV_scale_ptr + v_scale_offset)
                        v_scale_exp = v_scale_e8m0.to(tl.float32) - 127.0
                        v_scale_f32 = tl.math.exp2(v_scale_exp)

                        v_packed_offset = kv_offset + (v_block_idx * 16 + v_byte_idx) * stride_kv_dim
                        v_packed_byte = tl.load(KV_mxfp4_ptr + v_packed_offset)

                        v_nibble_val = tl.where(v_nibble == 0, v_packed_byte & 0x0F, (v_packed_byte >> 4) & 0x0F)
                        v_sign = (v_nibble_val >> 3) & 1
                        v_mag_idx = v_nibble_val & 0x07
                        v_mag = tl.where(v_mag_idx == 0, 0.0,
                                tl.where(v_mag_idx == 1, 0.5,
                                tl.where(v_mag_idx == 2, 1.0,
                                tl.where(v_mag_idx == 3, 1.5,
                                tl.where(v_mag_idx == 4, 2.0,
                                tl.where(v_mag_idx == 5, 3.0,
                                tl.where(v_mag_idx == 6, 4.0, 6.0)))))))
                        v_val = tl.where(v_sign == 1, -v_mag, v_mag) * v_scale_f32

                        out_acc += attn_weight * v_val

                    # Write output
                    tl.store(O_ptr + o_offset + out_d * stride_o_dim, out_acc.to(tl.bfloat16))


def _fused_mxfp4_mla_attention(
    q: torch.Tensor,
    kv_mxfp4: torch.Tensor,
    kv_scale: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """
    Fused MXFP4 dequantization + MLA attention.

    Strategy A: Load MXFP4 KV tiles, dequantize in registers, compute attention
    without writing intermediate bf16 buffer to HBM.

    Args:
        q: (total_q, num_heads, 576) bf16 query
        kv_mxfp4: (total_kv, 1, 288) fp4x2 packed KV cache
        kv_scale: (total_kv, num_blocks) E8M0 scale factors
        config: MLA configuration dict

    Returns:
        (total_q, num_heads, 512) bf16 attention output
    """
    batch_size = config["batch_size"]
    num_heads = config["num_heads"]
    q_seq_len = config["q_seq_len"]
    kv_seq_len = config["kv_seq_len"]
    qk_head_dim = config["qk_head_dim"]
    v_head_dim = config["v_head_dim"]
    sm_scale = config["sm_scale"]

    total_q = batch_size * q_seq_len

    # Allocate output
    o = torch.empty((total_q, num_heads, v_head_dim), dtype=torch.bfloat16, device=q.device)

    # Cast MXFP4 data to uint8 for Triton (Triton doesn't recognize float4_e2m1fn_x2)
    # Each fp4x2 element is 1 byte, so view as uint8
    kv_mxfp4_u8 = kv_mxfp4.view(torch.uint8)

    # Cast E8M0 scales to uint8 as well
    kv_scale_u8 = kv_scale.view(torch.uint8)

    # Grid: one program per (batch, head) pair
    grid = (batch_size, num_heads)

    # Block sizes
    BLOCK_KV = 64  # Process 64 KV tokens at a time

    _fused_mxfp4_mla_attention_kernel[grid](
        # Pointers
        q, kv_mxfp4_u8, kv_scale_u8, o,
        # Dimensions
        batch_size, num_heads, q_seq_len, kv_seq_len, qk_head_dim, v_head_dim,
        # Q strides
        q.stride(0), q.stride(1), q.stride(2),
        # KV strides (use original tensor's strides)
        kv_mxfp4.stride(0), kv_mxfp4.stride(1), kv_mxfp4.stride(2),
        # Scale strides
        kv_scale.stride(0), kv_scale.stride(1),
        # O strides
        o.stride(0), o.stride(1), o.stride(2),
        # Scale
        sm_scale,
        # Block sizes
        BLOCK_KV,
    )

    return o


def custom_kernel_mxfp4_dequant(data: input_t) -> output_t:
    """
    MLA decode with MXFP4 KV cache using Strategy A: Fused dequantization + attention.

    For small batches, uses the fused Triton kernel that:
    1. Loads MXFP4 KV tiles from HBM
    2. Dequantizes in registers/LDS
    3. Computes attention without writing intermediate buffer to HBM

    For larger batches, falls back to dequant + aiter kernel path.
    """
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]

    kv_buffer_mxfp4, kv_scale_mxfp4 = kv_data["mxfp4"]
    total_kv = kv_buffer_mxfp4.shape[0]
    orig_shape = (total_kv, NUM_KV_HEADS, QK_HEAD_DIM)

    # For small batches with short sequences, try the fused kernel
    # For now, use fallback path until fused kernel is fully debugged
    USE_FUSED_KERNEL = True

    if USE_FUSED_KERNEL and batch_size <= 4 and kv_seq_len <= 2048:
        # Strategy A: Fused MXFP4 dequant + attention
        return _fused_mxfp4_mla_attention(q, kv_buffer_mxfp4, kv_scale_mxfp4, config)

    # Fallback: Dequantize MXFP4 -> FP8, then use aiter a8w8 kernel
    kv_buffer_fp8, kv_scale_fp8 = dequantize_mxfp4_to_fp8(
        kv_buffer_mxfp4, kv_scale_mxfp4, orig_shape
    )
    q_fp8, q_scale = quantize_fp8(q)

    return _aiter_mla_decode(
        q_fp8, kv_buffer_fp8, qo_indptr, kv_indptr, config,
        q_scale=q_scale, kv_scale=kv_scale_fp8,
    )


# ---------------------------------------------------------------------------
# MXFP4 Q + MXFP4 KV — native low-precision compute with gemm_a4w4
# ---------------------------------------------------------------------------

def _quantize_bf16_to_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize bf16 tensor to MXFP4 with per-1x32 block scaling.

    Args:
        x: bf16 tensor of shape (..., K) where K is divisible by 32

    Returns:
        (x_mxfp4, x_scale): MXFP4 packed tensor and E8M0 scales
    """
    quant_func = aiter.get_triton_quant(QuantType.per_1x32)
    x_q, x_scale_sh = quant_func(x.contiguous(), shuffle=True)
    return x_q, x_scale_sh


def custom_kernel_mxfp4_native(data: input_t) -> output_t:
    """
    MLA decode using Strategy B: Full MXFP4 compute with gemm_a4w4.

    This implementation:
    1. Uses custom HIP kernel if available (fastest path)
    2. Falls back to gemm_a4w4 for fp4×fp4 QK^T computation
    3. Applies softmax in fp32 for numerical stability
    4. Uses bf16 for attn @ V computation

    This trades accuracy for throughput by leveraging MFMA fp4 instructions.
    """
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    num_heads = config["num_heads"]
    q_seq_len = config["q_seq_len"]
    kv_seq_len = config["kv_seq_len"]
    qk_head_dim = config["qk_head_dim"]  # 576
    v_head_dim = config["v_head_dim"]    # 512
    sm_scale = config["sm_scale"]

    kv_buffer_mxfp4, kv_scale_mxfp4 = kv_data["mxfp4"]
    print(f"{batch_size=}, {num_heads=}, {q_seq_len=}, {kv_seq_len=}, {qk_head_dim}, {v_head_dim}, {sm_scale=}")

    total_q = q.shape[0]  # batch_size * q_seq_len
    total_kv = kv_buffer_mxfp4.shape[0]  # batch_size * kv_seq_len

    # Use custom HIP kernel if available (fused dequant + attention)
    # Supports q_seq_len up to 4 (decode and small prefill)
    if HAS_HIP_KERNEL and q_seq_len <= 4:
        # KV: (total_kv, 1, qk_head_dim/2) -> (batch_size, kv_seq_len, qk_head_dim/2)
        kv_reshaped = kv_buffer_mxfp4.view(batch_size, kv_seq_len, qk_head_dim // 2)

        # Scale: (total_kv, num_blocks) -> (batch_size, kv_seq_len, num_blocks)
        num_blocks = kv_scale_mxfp4.shape[1]
        scale_reshaped = kv_scale_mxfp4.view(batch_size, kv_seq_len, num_blocks)

        # Allocate output tensor
        output = torch.empty(
            (total_q, num_heads, v_head_dim),
            dtype=torch.bfloat16,
            device=q.device
        )

        # Launch HIP kernel (uses hip-python or PyTorch depending on what compiled)
        _launch_hip_kernel(
            q.contiguous(),
            kv_reshaped.contiguous(),
            scale_reshaped.contiguous(),
            output,
            batch_size,
            q_seq_len,
            kv_seq_len,
            sm_scale
        )

        return output

    # Fallback: Python implementation with gemm_a4w4
    # Dequantize entire V to bf16 once (more efficient than per-batch)
    v_bf16_all = dequantize_mxfp4(
        kv_buffer_mxfp4,
        kv_scale_mxfp4,
        (total_kv, NUM_KV_HEADS, QK_HEAD_DIM)
    )[:, 0, :v_head_dim]  # (total_kv, 512)

    # Reshape V: (total_kv, 512) -> (batch_size, kv_seq_len, 512)
    v_batched = v_bf16_all.view(batch_size, kv_seq_len, v_head_dim)

    # Reshape Q: (total_q, num_heads, 576) -> (batch_size, q_seq_len, num_heads, 576)
    q_batched = q.view(batch_size, q_seq_len, num_heads, qk_head_dim)

    # Reshape KV: (total_kv, 1, 576/2) -> (batch_size, kv_seq_len, 576/2)
    kv_batched_mxfp4 = kv_buffer_mxfp4.view(batch_size, kv_seq_len, qk_head_dim // 2)
    kv_scale_batched = kv_scale_mxfp4.view(batch_size, kv_seq_len, -1)

    # Output tensor
    output = torch.empty((batch_size, q_seq_len, num_heads, v_head_dim),
                         dtype=torch.bfloat16, device=q.device)

    # Process each batch separately
    for b in range(batch_size):
        # Q for this batch: (q_seq_len, num_heads, 576) -> (q_seq_len * num_heads, 576)
        q_b = q_batched[b].view(q_seq_len * num_heads, qk_head_dim).contiguous()

        # Quantize Q to MXFP4
        q_mxfp4, q_scale = _quantize_bf16_to_mxfp4(q_b)

        # K for this batch: (kv_seq_len, 576/2)
        k_b_mxfp4 = kv_batched_mxfp4[b].contiguous()
        k_scale_b = kv_scale_batched[b].contiguous()

        # Shuffle K for gemm_a4w4
        k_shuffle = shuffle_weight(k_b_mxfp4, layout=(16, 16))

        # Compute QK^T using gemm_a4w4
        # A = Q_mxfp4 [q_seq_len * num_heads, 576/2]
        # B = K_mxfp4 [kv_seq_len, 576/2]
        # Output: [q_seq_len * num_heads, kv_seq_len]
        scores = aiter.gemm_a4w4(
            q_mxfp4,
            k_shuffle,
            q_scale,
            k_scale_b,
            dtype=aiter_dtypes.bf16,
            bpreshuffle=True,
        )

        # Apply scale and softmax in fp32
        scores = scores.float() * sm_scale

        # Reshape: (q_seq_len * num_heads, kv_seq_len) -> (q_seq_len, num_heads, kv_seq_len)
        scores = scores.view(q_seq_len, num_heads, kv_seq_len)

        # Apply softmax along KV dimension
        attn_weights = torch.softmax(scores, dim=-1)  # (q_seq_len, num_heads, kv_seq_len)

        # V for this batch: (kv_seq_len, 512)
        v_b = v_batched[b]

        # attn @ V: (q_seq_len, num_heads, kv_seq_len) @ (kv_seq_len, v_head_dim)
        # -> (q_seq_len, num_heads, v_head_dim)
        out_b = torch.einsum('qhk,kd->qhd', attn_weights.to(torch.bfloat16), v_b)

        output[b] = out_b

    # Reshape output: (batch_size, q_seq_len, num_heads, v_head_dim) -> (total_q, num_heads, v_head_dim)
    output = output.view(total_q, num_heads, v_head_dim)

    return output
