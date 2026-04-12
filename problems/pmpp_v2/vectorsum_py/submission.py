import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sum_cuda_source = """
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void __launch_bounds__(512)
sum_reduce_kernel(const float* __restrict__ input,
                  double* __restrict__ partial_sums,
                  int N) {
    double thread_sum = 0.0;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Vectorized loads: 4 floats per load
    int N4 = N / 4;
    const float4* input4 = reinterpret_cast<const float4*>(input);
    int vec_idx = idx;
    for (; vec_idx < N4; vec_idx += stride) {
        float4 v = __ldg(&input4[vec_idx]);
        thread_sum += (double)v.x + (double)v.y + (double)v.z + (double)v.w;
    }

    // Handle remainder
    int scalar_start = N4 * 4;
    for (int i = scalar_start + (idx - blockIdx.x * blockDim.x); i < N; i += blockDim.x) {
        thread_sum += (double)__ldg(&input[i]);
    }

    // Warp-level reduction using shuffle
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // First thread of each warp writes to shared memory
    __shared__ double warp_sums[16]; // max 512 threads = 16 warps
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    if (lane == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // First warp reduces all warp sums
    if (warp_id == 0) {
        int num_warps = blockDim.x >> 5;
        thread_sum = (lane < num_warps) ? warp_sums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            thread_sum += __shfl_down_sync(mask, thread_sum, offset);
        }
        if (lane == 0) {
            partial_sums[blockIdx.x] = thread_sum;
        }
    }
}

__global__ void final_reduce_kernel(const double* __restrict__ partial_sums,
                                     float* __restrict__ output,
                                     int num_blocks) {
    double thread_sum = 0.0;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        thread_sum += partial_sums[i];
    }

    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    __shared__ double warp_sums[16];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    if (lane == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = blockDim.x >> 5;
        thread_sum = (lane < num_warps) ? warp_sums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            thread_sum += __shfl_down_sync(mask, thread_sum, offset);
        }
        if (lane == 0) {
            output[0] = (float)thread_sum;
        }
    }
}

void sum_cuda(torch::Tensor input, torch::Tensor output) {
    int N = input.numel();
    const int threads = 512;
    int blocks = min((N / 4 + threads - 1) / threads, 1024);
    if (blocks < 1) blocks = 1;

    auto partial = torch::empty({blocks}, torch::dtype(torch::kFloat64).device(input.device()));

    sum_reduce_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        partial.data_ptr<double>(),
        N
    );

    final_reduce_kernel<<<1, 256>>>(
        partial.data_ptr<double>(),
        output.data_ptr<float>(),
        blocks
    );
}
"""

sum_cpp_source = """
#include <torch/extension.h>
void sum_cuda(torch::Tensor input, torch::Tensor output);
"""

sum_module = load_inline(
    name='sum_cuda',
    cpp_sources=sum_cpp_source,
    cuda_sources=sum_cuda_source,
    functions=['sum_cuda'],
    verbose=True,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-gencode', 'arch=compute_80,code=sm_80'],
)

def custom_kernel(data: input_t) -> output_t:
    data, output = data
    sum_module.sum_cuda(data, output)
    return output[0]
