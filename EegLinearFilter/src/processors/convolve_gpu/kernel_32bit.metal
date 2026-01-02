//
//  kernel_32bit.metal
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 26.11.2025.
//  Optimized Metal compute shader for 1D convolution using register tiling.
//

#include <metal_stdlib>
#include "../../config.h"

using namespace metal;

/**
 * Optimized compute kernel.
 * * Strategy:
 * 1. Loads a block of data into shared memory (threadgroup memory).
 * 2. Uses register tiling to compute multiple output values per thread to hide memory latency.
 * 3. Manually unrolls loops (16x unroll) to maximize arithmetic density.
 * * @param data Global buffer containing input signal.
 * @param output Global buffer for results.
 * @param convKernel Global buffer containing kernel weights.
 * @param kernelSize Size of the kernel.
 * @param outSize Expected size of the valid output.
 * @param rawDataSize Total size of the input buffer (for bounds checking).
 * @param cache Threadgroup shared memory buffer.
 */
kernel void convolve_kernel_32(
    device const float4* data [[buffer(0)]],
    device float4* output [[buffer(1)]],
    constant float* convKernel [[buffer(2)]],
    constant uint& kernelSize [[buffer(3)]],
    constant uint& outSize [[buffer(4)]],
    constant uint& rawDataSize [[buffer(5)]],
    threadgroup float4* cache [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    int groupStartGlobal = group_id * TILE_SIZE;
    
    // Accumulators for register tiling (processing 4 vectors / 16 floats at once)float4 sumA = float4(0.0f);
    float4 sumA = float4(0.0f);
    float4 sumB = float4(0.0f);
    float4 sumC = float4(0.0f);
    float4 sumD = float4(0.0f);
    
    int localBaseIndex = tid * ITEMS_PER_THREAD;
    threadgroup float* scalarCache = (threadgroup float*)cache;

    // Process kernel in segments to fit data into limited threadgroup memory
    for (int k_base = 0; k_base < (int)kernelSize; k_base += KERNEL_SEGMENT_SIZE) {
        int currentSegmentLen = min((int)KERNEL_SEGMENT_SIZE, (int)kernelSize - k_base);
        int elementsNeeded = TILE_SIZE + currentSegmentLen - 1;
        
        int vectorsNeeded = (elementsNeeded + 3) / 4;
        int dataOffset = groupStartGlobal + k_base;

        int distanceToEnd = (int)rawDataSize - dataOffset;
        int safeVectorsCount = (distanceToEnd > 0) ? (distanceToEnd / 4) : 0;

        if (safeVectorsCount > vectorsNeeded) {
            safeVectorsCount = vectorsNeeded;
        }

        // Cooperative loading into shared memory
        int i = tid;
        while (i < safeVectorsCount) {
            int absoluteIdx = dataOffset + (i * 4);
            cache[i] = *((device const float4*)((device const float*)data + absoluteIdx));
            i += THREADS_PER_GROUP;
        }

        // Handle edge cases (partial vectors at end of data)
        while (i < vectorsNeeded) {
            int absoluteIdx = dataOffset + (i * 4);
            
            float4 loadedVal = float4(0.0f);
            if (absoluteIdx + 3 < (int)rawDataSize) {
                 loadedVal = *((device const float4*)((device const float*)data + absoluteIdx));
            } else {
                for(int c=0; c<4; c++) {
                    if (absoluteIdx + c < (int)rawDataSize) {
                        loadedVal[c] = ((device const float*)data)[absoluteIdx + c];
                    }
                }
            }
            cache[i] = loadedVal;
            i += THREADS_PER_GROUP;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Preload cache values into registers
        float v0 = scalarCache[localBaseIndex + 0];
        float v1 = scalarCache[localBaseIndex + 1];
        float v2 = scalarCache[localBaseIndex + 2];
        float v3 = scalarCache[localBaseIndex + 3];
        float v4 = scalarCache[localBaseIndex + 4];
        float v5 = scalarCache[localBaseIndex + 5];
        float v6 = scalarCache[localBaseIndex + 6];
        float v7 = scalarCache[localBaseIndex + 7];
        float v8 = scalarCache[localBaseIndex + 8];
        float v9 = scalarCache[localBaseIndex + 9];
        float v10 = scalarCache[localBaseIndex + 10];
        float v11 = scalarCache[localBaseIndex + 11];
        float v12 = scalarCache[localBaseIndex + 12];
        float v13 = scalarCache[localBaseIndex + 13];
        float v14 = scalarCache[localBaseIndex + 14];
        float v15 = scalarCache[localBaseIndex + 15];

        int k = 0;
        int limitLoop = currentSegmentLen - 15;
        
        // Main compute loop
        for (; k < limitLoop; k += 16) {
            float w;
            int kernelIdx = k_base + k;
            
            w = convKernel[kernelIdx];
            sumA = fma(float4(v0, v1, v2, v3), float4(w), sumA);
            sumB = fma(float4(v4, v5, v6, v7), float4(w), sumB);
            sumC = fma(float4(v8, v9, v10, v11), float4(w), sumC);
            sumD = fma(float4(v12, v13, v14, v15), float4(w), sumD);
            v0 = scalarCache[localBaseIndex + k + 16];

            w = convKernel[kernelIdx+1];
            sumA = fma(float4(v1, v2, v3, v4), float4(w), sumA);
            sumB = fma(float4(v5, v6, v7, v8), float4(w), sumB);
            sumC = fma(float4(v9, v10, v11, v12), float4(w), sumC);
            sumD = fma(float4(v13, v14, v15, v0), float4(w), sumD);
            v1 = scalarCache[localBaseIndex + k + 17];

            w = convKernel[kernelIdx+2];
            sumA = fma(float4(v2, v3, v4, v5), float4(w), sumA);
            sumB = fma(float4(v6, v7, v8, v9), float4(w), sumB);
            sumC = fma(float4(v10, v11, v12, v13), float4(w), sumC);
            sumD = fma(float4(v14, v15, v0, v1), float4(w), sumD);
            v2 = scalarCache[localBaseIndex + k + 18];

            w = convKernel[kernelIdx+3];
            sumA = fma(float4(v3, v4, v5, v6), float4(w), sumA);
            sumB = fma(float4(v7, v8, v9, v10), float4(w), sumB);
            sumC = fma(float4(v11, v12, v13, v14), float4(w), sumC);
            sumD = fma(float4(v15, v0, v1, v2), float4(w), sumD);
            v3 = scalarCache[localBaseIndex + k + 19];
            
            w = convKernel[kernelIdx+4];
            sumA = fma(float4(v4, v5, v6, v7), float4(w), sumA);
            sumB = fma(float4(v8, v9, v10, v11), float4(w), sumB);
            sumC = fma(float4(v12, v13, v14, v15), float4(w), sumC);
            sumD = fma(float4(v0, v1, v2, v3), float4(w), sumD);
            v4 = scalarCache[localBaseIndex + k + 20];

            w = convKernel[kernelIdx+5];
            sumA = fma(float4(v5, v6, v7, v8), float4(w), sumA);
            sumB = fma(float4(v9, v10, v11, v12), float4(w), sumB);
            sumC = fma(float4(v13, v14, v15, v0), float4(w), sumC);
            sumD = fma(float4(v1, v2, v3, v4), float4(w), sumD);
            v5 = scalarCache[localBaseIndex + k + 21];

            w = convKernel[kernelIdx+6];
            sumA = fma(float4(v6, v7, v8, v9), float4(w), sumA);
            sumB = fma(float4(v10, v11, v12, v13), float4(w), sumB);
            sumC = fma(float4(v14, v15, v0, v1), float4(w), sumC);
            sumD = fma(float4(v2, v3, v4, v5), float4(w), sumD);
            v6 = scalarCache[localBaseIndex + k + 22];

            w = convKernel[kernelIdx+7];
            sumA = fma(float4(v7, v8, v9, v10), float4(w), sumA);
            sumB = fma(float4(v11, v12, v13, v14), float4(w), sumB);
            sumC = fma(float4(v15, v0, v1, v2), float4(w), sumC);
            sumD = fma(float4(v3, v4, v5, v6), float4(w), sumD);
            v7 = scalarCache[localBaseIndex + k + 23];

            w = convKernel[kernelIdx+8];
            sumA = fma(float4(v8, v9, v10, v11), float4(w), sumA);
            sumB = fma(float4(v12, v13, v14, v15), float4(w), sumB);
            sumC = fma(float4(v0, v1, v2, v3), float4(w), sumC);
            sumD = fma(float4(v4, v5, v6, v7), float4(w), sumD);
            v8 = scalarCache[localBaseIndex + k + 24];

            w = convKernel[kernelIdx+9];
            sumA = fma(float4(v9, v10, v11, v12), float4(w), sumA);
            sumB = fma(float4(v13, v14, v15, v0), float4(w), sumB);
            sumC = fma(float4(v1, v2, v3, v4), float4(w), sumC);
            sumD = fma(float4(v5, v6, v7, v8), float4(w), sumD);
            v9 = scalarCache[localBaseIndex + k + 25];

            w = convKernel[kernelIdx+10];
            sumA = fma(float4(v10, v11, v12, v13), float4(w), sumA);
            sumB = fma(float4(v14, v15, v0, v1), float4(w), sumB);
            sumC = fma(float4(v2, v3, v4, v5), float4(w), sumC);
            sumD = fma(float4(v6, v7, v8, v9), float4(w), sumD);
            v10 = scalarCache[localBaseIndex + k + 26];

            w = convKernel[kernelIdx+11];
            sumA = fma(float4(v11, v12, v13, v14), float4(w), sumA);
            sumB = fma(float4(v15, v0, v1, v2), float4(w), sumB);
            sumC = fma(float4(v3, v4, v5, v6), float4(w), sumC);
            sumD = fma(float4(v7, v8, v9, v10), float4(w), sumD);
            v11 = scalarCache[localBaseIndex + k + 27];

            w = convKernel[kernelIdx+12];
            sumA = fma(float4(v12, v13, v14, v15), float4(w), sumA);
            sumB = fma(float4(v0, v1, v2, v3), float4(w), sumB);
            sumC = fma(float4(v4, v5, v6, v7), float4(w), sumC);
            sumD = fma(float4(v8, v9, v10, v11), float4(w), sumD);
            v12 = scalarCache[localBaseIndex + k + 28];

            w = convKernel[kernelIdx+13];
            sumA = fma(float4(v13, v14, v15, v0), float4(w), sumA);
            sumB = fma(float4(v1, v2, v3, v4), float4(w), sumB);
            sumC = fma(float4(v5, v6, v7, v8), float4(w), sumC);
            sumD = fma(float4(v9, v10, v11, v12), float4(w), sumD);
            v13 = scalarCache[localBaseIndex + k + 29];

            w = convKernel[kernelIdx+14];
            sumA = fma(float4(v14, v15, v0, v1), float4(w), sumA);
            sumB = fma(float4(v2, v3, v4, v5), float4(w), sumB);
            sumC = fma(float4(v6, v7, v8, v9), float4(w), sumC);
            sumD = fma(float4(v10, v11, v12, v13), float4(w), sumD);
            v14 = scalarCache[localBaseIndex + k + 30];

            w = convKernel[kernelIdx+15];
            sumA = fma(float4(v15, v0, v1, v2), float4(w), sumA);
            sumB = fma(float4(v3, v4, v5, v6), float4(w), sumB);
            sumC = fma(float4(v7, v8, v9, v10), float4(w), sumC);
            sumD = fma(float4(v11, v12, v13, v14), float4(w), sumD);
            v15 = scalarCache[localBaseIndex + k + 31];
        }
        
        // Handle remaining kernel elements in this segment
        for (; k < currentSegmentLen; ++k) {
            float w = convKernel[k_base + k];
            sumA = fma(float4(v0, v1, v2, v3), float4(w), sumA);
            sumB = fma(float4(v4, v5, v6, v7), float4(w), sumB);
            sumC = fma(float4(v8, v9, v10, v11), float4(w), sumC);
            sumD = fma(float4(v12, v13, v14, v15), float4(w), sumD);
            
            float next = scalarCache[localBaseIndex + k + 16];
            
            v0 = v1; v1 = v2; v2 = v3; v3 = v4;
            v4 = v5; v5 = v6; v6 = v7; v7 = v8;
            v8 = v9; v9 = v10; v10 = v11; v11 = v12;
            v12 = v13; v13 = v14; v14 = v15; v15 = next;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    int globalOutputIndex = groupStartGlobal + localBaseIndex;
    
    if (globalOutputIndex + 15 < (int)outSize) {
        int vecIndex = globalOutputIndex / 4;
        output[vecIndex] = sumA;
        output[vecIndex + 1] = sumB;
        output[vecIndex + 2] = sumC;
        output[vecIndex + 3] = sumD;
    } else {
        device float* outScalar = (device float*)output;
        for(int i=0; i<4; ++i) {
            if(globalOutputIndex + i < (int)outSize)
                outScalar[globalOutputIndex + i] = sumA[i];
            if(globalOutputIndex + 4 + i < (int)outSize)
                outScalar[globalOutputIndex + 4 + i] = sumB[i];
            if(globalOutputIndex + 8 + i < (int)outSize)
                outScalar[globalOutputIndex + 8 + i] = sumC[i];
            if(globalOutputIndex + 12 + i < (int)outSize)
                outScalar[globalOutputIndex + 12 + i] = sumD[i];
        }
    }
}
