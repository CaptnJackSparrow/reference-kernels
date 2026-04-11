import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

add_cuda_source = """
#include <cuda_fp16.h>

__global__ void __launch_bounds__(512)
add_kernel_vec(const float4* __restrict__ A,
               const float4* __restrict__ B,
               float4* __restrict__ C,
               int N4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < N4; idx += stride) {
        float4 a = __ldg(&A[idx]);
        float4 b = __ldg(&B[idx]);
        half2* a_h = reinterpret_cast<half2*>(&a);
        half2* b_h = reinterpret_cast<half2*>(&b);
        float4 c;
        half2* c_h = reinterpret_cast<half2*>(&c);
        c_h[0] = __hadd2(a_h[0], b_h[0]);
        c_h[1] = __hadd2(a_h[1], b_h[1]);
        c_h[2] = __hadd2(a_h[2], b_h[2]);
        c_h[3] = __hadd2(a_h[3], b_h[3]);
        C[idx] = c;
    }
}

__global__ void add_kernel_scalar(const __half* __restrict__ A,
                                  const __half* __restrict__ B,
                                  __half* __restrict__ C,
                                  int start, int N) {
    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = __hadd(A[idx], B[idx]);
    }
}

void add_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int N = A.numel();
    int N4 = N / 8;
    int remainder = N - N4 * 8;

    const int threads = 512;

    if (N4 > 0) {
        int blocks = min((N4 + threads - 1) / threads, 65535);
        add_kernel_vec<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(A.data_ptr<at::Half>()),
            reinterpret_cast<const float4*>(B.data_ptr<at::Half>()),
            reinterpret_cast<float4*>(C.data_ptr<at::Half>()),
            N4
        );
    }

    if (remainder > 0) {
        int rblocks = (remainder + 255) / 256;
        add_kernel_scalar<<<rblocks, 256>>>(
            reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
            reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
            N4 * 8, N
        );
    }
}
"""

add_cpp_source = """
#include <torch/extension.h>
void add_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C);
"""

add_module = load_inline(
    name='add_cuda',
    cpp_sources=add_cpp_source,
    cuda_sources=add_cuda_source,
    functions=['add_cuda'],
    verbose=True,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
)

def custom_kernel(data: input_t) -> output_t:
    A, B, output = data
    add_module.add_cuda(A, B, output)
    return output
