import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

add_cuda_source = """
#include <cuda_fp16.h>

template <int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE)
add_kernel_vec(const float4* __restrict__ A,
               const float4* __restrict__ B,
               float4* __restrict__ C,
               int N4) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int stride = BLOCK_SIZE * gridDim.x;
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

template <int BLOCK_SIZE, int N, int BLOCKS>
__global__ void __launch_bounds__(BLOCK_SIZE)
add_kernel_vec_v2(const float4* __restrict__ A,
               const float4* __restrict__ B,
               float4* __restrict__ C) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    constexpr int stride = BLOCK_SIZE * BLOCKS;
    #pragma unroll 8
    for (; idx < N / 8; idx += stride) {
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

constexpr int K(int x) { return 1024 * 1024 * x * x; }

void add_cuda(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int N = A.numel();
    int N4 = N / 8;
    int remainder = N - N4 * 8;

    if (N4 > 0) {
        int threads, blocks;

        constexpr int MAX_BLOCKS = 65535;
        constexpr int WAVE8_BLOCKS = 108 * 8;

        if (N <= 1024 * 1024) {
            // size <= 1024: ~131K float4s. Use 128 threads, many blocks for SM coverage
            threads = 128;
            blocks = min((N4 + 127) / 128, WAVE8_BLOCKS);
        } else if (N <= 4 * 1024 * 1024) {
            // size 2048: ~524K float4s. Use 256 threads, moderate blocks
            threads = 256;
            blocks = min((N4 + 255) / 256, WAVE8_BLOCKS);
        } else {
            // size 4096+: 2M+ float4s. Use 512 threads, let grid-stride handle it
            threads = 512;
            blocks = min((N4 + 511) / 512, MAX_BLOCKS);
        }

        //printf("N4: %d, threads: %d, blocks: %d", N4, threads, blocks);

        if (N == K(1) && blocks == WAVE8_BLOCKS) {
            add_kernel_vec_v2<128, K(1), WAVE8_BLOCKS><<<blocks, 128>>>
            (
                reinterpret_cast<const float4*>(A.data_ptr<at::Half>()),
                reinterpret_cast<const float4*>(B.data_ptr<at::Half>()),
                reinterpret_cast<float4*>(C.data_ptr<at::Half>())
            );
        } else if (N == K(2) && blocks == WAVE8_BLOCKS) {
            add_kernel_vec_v2<256, K(2), WAVE8_BLOCKS><<<blocks, 256>>>
            (
                reinterpret_cast<const float4*>(A.data_ptr<at::Half>()),
                reinterpret_cast<const float4*>(B.data_ptr<at::Half>()),
                reinterpret_cast<float4*>(C.data_ptr<at::Half>())
            );
        } else if (N == K(4) && blocks == WAVE8_BLOCKS) {
            add_kernel_vec_v2<256, K(4), WAVE8_BLOCKS><<<blocks, 256>>>
            (
                reinterpret_cast<const float4*>(A.data_ptr<at::Half>()),
                reinterpret_cast<const float4*>(B.data_ptr<at::Half>()),
                reinterpret_cast<float4*>(C.data_ptr<at::Half>())
            );
        } else if (N == K(8) && blocks == WAVE8_BLOCKS) {
            add_kernel_vec_v2<256, K(8), WAVE8_BLOCKS><<<blocks, 256>>>
            (
                reinterpret_cast<const float4*>(A.data_ptr<at::Half>()),
                reinterpret_cast<const float4*>(B.data_ptr<at::Half>()),
                reinterpret_cast<float4*>(C.data_ptr<at::Half>())
            );
        } else if (N == K(16) && blocks == MAX_BLOCKS) {
            add_kernel_vec_v2<512, K(16), MAX_BLOCKS><<<blocks, 512>>>
            (
                reinterpret_cast<const float4*>(A.data_ptr<at::Half>()),
                reinterpret_cast<const float4*>(B.data_ptr<at::Half>()),
                reinterpret_cast<float4*>(C.data_ptr<at::Half>())
            );
        } else {
            switch (threads) {
                case 128:
                    add_kernel_vec<128><<<blocks, 128>>>(
                        reinterpret_cast<const float4*>(A.data_ptr<at::Half>()),
                        reinterpret_cast<const float4*>(B.data_ptr<at::Half>()),
                        reinterpret_cast<float4*>(C.data_ptr<at::Half>()),
                        N4);
                    break;
                case 256:
                    add_kernel_vec<256><<<blocks, 256>>>(
                        reinterpret_cast<const float4*>(A.data_ptr<at::Half>()),
                        reinterpret_cast<const float4*>(B.data_ptr<at::Half>()),
                        reinterpret_cast<float4*>(C.data_ptr<at::Half>()),
                        N4);
                    break;
                default:
                    add_kernel_vec<512><<<blocks, 512>>>(
                        reinterpret_cast<const float4*>(A.data_ptr<at::Half>()),
                        reinterpret_cast<const float4*>(B.data_ptr<at::Half>()),
                        reinterpret_cast<float4*>(C.data_ptr<at::Half>()),
                        N4);
                    break;
            }
        }
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
    extra_cuda_cflags=['-O3', '--use_fast_math', '-gencode', 'arch=compute_80,code=sm_80'],
)

def custom_kernel(data: input_t) -> output_t:
    A, B, output = data
    add_module.add_cuda(A, B, output)
    return output
