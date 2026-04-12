import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

sort_cuda_source = """
#include <cub/cub.cuh>
#include <cuda_runtime.h>

static void* d_temp = nullptr;
static size_t temp_bytes = 0;
static float* d_alt = nullptr;
static int alt_capacity = 0;

void sort_cuda(torch::Tensor input, torch::Tensor output) {
    int N = input.numel();
    float* d_in = input.data_ptr<float>();
    float* d_out = output.data_ptr<float>();

    if (N > alt_capacity) {
        if (d_alt) cudaFree(d_alt);
        cudaMalloc(&d_alt, (size_t)N * sizeof(float));
        alt_capacity = N;
    }

    cudaMemcpyAsync(d_alt, d_in, (size_t)N * sizeof(float), cudaMemcpyDeviceToDevice);

    cub::DoubleBuffer<float> d_keys(d_alt, d_out);

    size_t required = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, required, d_keys, N);

    if (required > temp_bytes) {
        if (d_temp) cudaFree(d_temp);
        cudaMalloc(&d_temp, required);
        temp_bytes = required;
    }

    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_keys, N);

    if (d_keys.Current() != d_out) {
        cudaMemcpyAsync(d_out, d_keys.Current(), (size_t)N * sizeof(float), cudaMemcpyDeviceToDevice);
    }
}
"""

sort_cpp_source = """
#include <torch/extension.h>
void sort_cuda(torch::Tensor input, torch::Tensor output);
"""

sort_module = load_inline(
    name='sort_cuda',
    cpp_sources=sort_cpp_source,
    cuda_sources=sort_cuda_source,
    functions=['sort_cuda'],
    verbose=True,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
)

def custom_kernel(data: input_t) -> output_t:
    data, output = data
    sort_module.sort_cuda(data, output)
    return output
