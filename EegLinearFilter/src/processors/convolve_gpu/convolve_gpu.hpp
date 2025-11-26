//
//  convolve_gpu.hpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 26.11.2025.
//

#ifndef CONVOLVE_GPU
#define CONVOLVE_GPU

#include "../../data_types.hpp"
#include <vector>

void run_metal_convolution_impl(const float* src, float* dst, size_t dataSize, const float* kernel, size_t kernelSize);

template <int Radius, int ChunkSize>
void convolve_gpu(const NeonVector& data, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel) {
    run_metal_convolution_impl(
        data.data(),
        outputBuffer.data(),
        data.size(),
        convolutionKernel.data(),
        convolutionKernel.size()
    );
}

#endif // CONVOLVE_GPU
