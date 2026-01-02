//
//  kernel_naive.metal
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 26.11.2025.
//  Baseline 1D convolution Metal shader.
//

#include <metal_stdlib>
#include "../../config.h"

using namespace metal;

/**
 * Naive compute kernel.
 * * Strategy:
 * 1. Each thread group cooperatively loads the required input data block (tile + halo) into shared memory.
 * 2. Each thread calculates output for a specific set of pixels (4 pixels per thread) by iterating through the kernel.
 * * @param data Input buffer.
 * @param output Output buffer.
 * @param convKernel Kernel weights buffer.
 * @param kernelSize Radius-based kernel size.
 * @param outSize Number of valid output elements.
 * @param cache Shared threadgroup memory.
 */
kernel void convolve_kernel_naive(
    device const float4* data [[buffer(0)]],
    device float4* output [[buffer(1)]],
    constant float* convKernel [[buffer(2)]],
    constant uint& kernelSize [[buffer(3)]],
    constant uint& outSize [[buffer(4)]],
    threadgroup float* cache [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    int itemsPerThread = 4;
    int tilePixels = THREADS_PER_GROUP * itemsPerThread;
    int groupStartPixel = group_id * tilePixels;
    
    int kSize = (int)kernelSize;
    int totalFloatsNeeded = tilePixels + kSize - 1;
    int vectorsNeeded = (totalFloatsNeeded + 3) / 4;
    threadgroup float4* cacheVec = (threadgroup float4*)cache;
    int startVectorIdx = groupStartPixel / 4;

    // Load data into shared memory
    for (int i = tid; i < vectorsNeeded; i += THREADS_PER_GROUP) {
        int globalVecIdx = startVectorIdx + i;
        float4 vec = data[globalVecIdx];
        cacheVec[i] = vec;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    int localPixelIndex = tid * itemsPerThread;
    int globalOutputIndex = groupStartPixel + localPixelIndex;
    int globalOutputVecIndex = globalOutputIndex / 4;

    // Compute convolution
    if (globalOutputIndex < (int)outSize) {
        float4 sum = float4(0.0f);
        
        for (int k = 0; k < kSize; ++k) {
            float kVal = convKernel[k];
            
            float d0 = cache[localPixelIndex + k + 0];
            float d1 = cache[localPixelIndex + k + 1];
            float d2 = cache[localPixelIndex + k + 2];
            float d3 = cache[localPixelIndex + k + 3];
            
            sum = fma(float4(d0, d1, d2, d3), float4(kVal), sum);
        }
        
        if (globalOutputIndex + 3 < (int)outSize) {
             output[globalOutputVecIndex] = sum;
        } else {
            // Handle edge case where output size is not multiple of 4
            device float* outScalar = (device float*)output;
            for (int i=0; i<4; ++i) {
                if (globalOutputIndex + i < (int)outSize) {
                    outScalar[globalOutputIndex + i] = sum[i];
                }
            }
        }
    }
}
