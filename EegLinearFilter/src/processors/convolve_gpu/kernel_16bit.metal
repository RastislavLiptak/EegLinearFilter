//
//  kernel_16bit.metal
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 22.12.2025.
//

#include <metal_stdlib>
#include "../../config.h"

using namespace metal;

kernel void convolve_kernel_16(
    device const float4* data [[buffer(0)]],
    device float4* output [[buffer(1)]],
    constant float* convKernel [[buffer(2)]],
    constant uint& kernelSize [[buffer(3)]],
    constant uint& outSize [[buffer(4)]],
    constant uint& rawDataSize [[buffer(5)]],
    threadgroup half4* cache [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    int groupStartGlobal = group_id * TILE_SIZE;
    
    half4 sumA = half4(0.0h);
    half4 sumB = half4(0.0h);
    half4 sumC = half4(0.0h);
    half4 sumD = half4(0.0h);
    
    int localBaseIndex = tid * ITEMS_PER_THREAD;
    threadgroup half* scalarCache = (threadgroup half*)cache;

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

        int i = tid;
        while (i < safeVectorsCount) {
            int absoluteIdx = dataOffset + (i * 4);
            cache[i] = half4(*((device const float4*)((device const float*)data + absoluteIdx)));
            i += THREADS_PER_GROUP;
        }

        while (i < vectorsNeeded) {
            int absoluteIdx = dataOffset + (i * 4);
            
            half4 loadedVal = half4(0.0h);
            if (absoluteIdx + 3 < (int)rawDataSize) {
                 loadedVal = half4(*((device const float4*)((device const float*)data + absoluteIdx)));
            } else {
                for(int c=0; c<4; c++) {
                    if (absoluteIdx + c < (int)rawDataSize) {
                        loadedVal[c] = (half)((device const float*)data)[absoluteIdx + c];
                    }
                }
            }
            cache[i] = loadedVal;
            i += THREADS_PER_GROUP;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        half v0 = scalarCache[localBaseIndex + 0];
        half v1 = scalarCache[localBaseIndex + 1];
        half v2 = scalarCache[localBaseIndex + 2];
        half v3 = scalarCache[localBaseIndex + 3];
        half v4 = scalarCache[localBaseIndex + 4];
        half v5 = scalarCache[localBaseIndex + 5];
        half v6 = scalarCache[localBaseIndex + 6];
        half v7 = scalarCache[localBaseIndex + 7];
        half v8 = scalarCache[localBaseIndex + 8];
        half v9 = scalarCache[localBaseIndex + 9];
        half v10 = scalarCache[localBaseIndex + 10];
        half v11 = scalarCache[localBaseIndex + 11];
        half v12 = scalarCache[localBaseIndex + 12];
        half v13 = scalarCache[localBaseIndex + 13];
        half v14 = scalarCache[localBaseIndex + 14];
        half v15 = scalarCache[localBaseIndex + 15];

        int k = 0;
        int limitLoop = currentSegmentLen - 15;
        
        for (; k < limitLoop; k += 16) {
            half w;
            int kernelIdx = k_base + k;
            
            w = (half)convKernel[kernelIdx];
            sumA = fma(half4(v0, v1, v2, v3), half4(w), sumA);
            sumB = fma(half4(v4, v5, v6, v7), half4(w), sumB);
            sumC = fma(half4(v8, v9, v10, v11), half4(w), sumC);
            sumD = fma(half4(v12, v13, v14, v15), half4(w), sumD);
            v0 = scalarCache[localBaseIndex + k + 16];

            w = (half)convKernel[kernelIdx+1];
            sumA = fma(half4(v1, v2, v3, v4), half4(w), sumA);
            sumB = fma(half4(v5, v6, v7, v8), half4(w), sumB);
            sumC = fma(half4(v9, v10, v11, v12), half4(w), sumC);
            sumD = fma(half4(v13, v14, v15, v0), half4(w), sumD);
            v1 = scalarCache[localBaseIndex + k + 17];

            w = (half)convKernel[kernelIdx+2];
            sumA = fma(half4(v2, v3, v4, v5), half4(w), sumA);
            sumB = fma(half4(v6, v7, v8, v9), half4(w), sumB);
            sumC = fma(half4(v10, v11, v12, v13), half4(w), sumC);
            sumD = fma(half4(v14, v15, v0, v1), half4(w), sumD);
            v2 = scalarCache[localBaseIndex + k + 18];

            w = (half)convKernel[kernelIdx+3];
            sumA = fma(half4(v3, v4, v5, v6), half4(w), sumA);
            sumB = fma(half4(v7, v8, v9, v10), half4(w), sumB);
            sumC = fma(half4(v11, v12, v13, v14), half4(w), sumC);
            sumD = fma(half4(v15, v0, v1, v2), half4(w), sumD);
            v3 = scalarCache[localBaseIndex + k + 19];
            
            w = (half)convKernel[kernelIdx+4];
            sumA = fma(half4(v4, v5, v6, v7), half4(w), sumA);
            sumB = fma(half4(v8, v9, v10, v11), half4(w), sumB);
            sumC = fma(half4(v12, v13, v14, v15), half4(w), sumC);
            sumD = fma(half4(v0, v1, v2, v3), half4(w), sumD);
            v4 = scalarCache[localBaseIndex + k + 20];

            w = (half)convKernel[kernelIdx+5];
            sumA = fma(half4(v5, v6, v7, v8), half4(w), sumA);
            sumB = fma(half4(v9, v10, v11, v12), half4(w), sumB);
            sumC = fma(half4(v13, v14, v15, v0), half4(w), sumC);
            sumD = fma(half4(v1, v2, v3, v4), half4(w), sumD);
            v5 = scalarCache[localBaseIndex + k + 21];

            w = (half)convKernel[kernelIdx+6];
            sumA = fma(half4(v6, v7, v8, v9), half4(w), sumA);
            sumB = fma(half4(v10, v11, v12, v13), half4(w), sumB);
            sumC = fma(half4(v14, v15, v0, v1), half4(w), sumC);
            sumD = fma(half4(v2, v3, v4, v5), half4(w), sumD);
            v6 = scalarCache[localBaseIndex + k + 22];

            w = (half)convKernel[kernelIdx+7];
            sumA = fma(half4(v7, v8, v9, v10), half4(w), sumA);
            sumB = fma(half4(v11, v12, v13, v14), half4(w), sumB);
            sumC = fma(half4(v15, v0, v1, v2), half4(w), sumC);
            sumD = fma(half4(v3, v4, v5, v6), half4(w), sumD);
            v7 = scalarCache[localBaseIndex + k + 23];

            w = (half)convKernel[kernelIdx+8];
            sumA = fma(half4(v8, v9, v10, v11), half4(w), sumA);
            sumB = fma(half4(v12, v13, v14, v15), half4(w), sumB);
            sumC = fma(half4(v0, v1, v2, v3), half4(w), sumC);
            sumD = fma(half4(v4, v5, v6, v7), half4(w), sumD);
            v8 = scalarCache[localBaseIndex + k + 24];

            w = (half)convKernel[kernelIdx+9];
            sumA = fma(half4(v9, v10, v11, v12), half4(w), sumA);
            sumB = fma(half4(v13, v14, v15, v0), half4(w), sumB);
            sumC = fma(half4(v1, v2, v3, v4), half4(w), sumC);
            sumD = fma(half4(v5, v6, v7, v8), half4(w), sumD);
            v9 = scalarCache[localBaseIndex + k + 25];

            w = (half)convKernel[kernelIdx+10];
            sumA = fma(half4(v10, v11, v12, v13), half4(w), sumA);
            sumB = fma(half4(v14, v15, v0, v1), half4(w), sumB);
            sumC = fma(half4(v2, v3, v4, v5), half4(w), sumC);
            sumD = fma(half4(v6, v7, v8, v9), half4(w), sumD);
            v10 = scalarCache[localBaseIndex + k + 26];

            w = (half)convKernel[kernelIdx+11];
            sumA = fma(half4(v11, v12, v13, v14), half4(w), sumA);
            sumB = fma(half4(v15, v0, v1, v2), half4(w), sumB);
            sumC = fma(half4(v3, v4, v5, v6), half4(w), sumC);
            sumD = fma(half4(v7, v8, v9, v10), half4(w), sumD);
            v11 = scalarCache[localBaseIndex + k + 27];

            w = (half)convKernel[kernelIdx+12];
            sumA = fma(half4(v12, v13, v14, v15), half4(w), sumA);
            sumB = fma(half4(v0, v1, v2, v3), half4(w), sumB);
            sumC = fma(half4(v4, v5, v6, v7), half4(w), sumC);
            sumD = fma(half4(v8, v9, v10, v11), half4(w), sumD);
            v12 = scalarCache[localBaseIndex + k + 28];

            w = (half)convKernel[kernelIdx+13];
            sumA = fma(half4(v13, v14, v15, v0), half4(w), sumA);
            sumB = fma(half4(v1, v2, v3, v4), half4(w), sumB);
            sumC = fma(half4(v5, v6, v7, v8), half4(w), sumC);
            sumD = fma(half4(v9, v10, v11, v12), half4(w), sumD);
            v13 = scalarCache[localBaseIndex + k + 29];

            w = (half)convKernel[kernelIdx+14];
            sumA = fma(half4(v14, v15, v0, v1), half4(w), sumA);
            sumB = fma(half4(v2, v3, v4, v5), half4(w), sumB);
            sumC = fma(half4(v6, v7, v8, v9), half4(w), sumC);
            sumD = fma(half4(v10, v11, v12, v13), half4(w), sumD);
            v14 = scalarCache[localBaseIndex + k + 30];

            w = (half)convKernel[kernelIdx+15];
            sumA = fma(half4(v15, v0, v1, v2), half4(w), sumA);
            sumB = fma(half4(v3, v4, v5, v6), half4(w), sumB);
            sumC = fma(half4(v7, v8, v9, v10), half4(w), sumC);
            sumD = fma(half4(v11, v12, v13, v14), half4(w), sumD);
            v15 = scalarCache[localBaseIndex + k + 31];
        }
        
        for (; k < currentSegmentLen; ++k) {
            half w = (half)convKernel[k_base + k];
            sumA = fma(half4(v0, v1, v2, v3), half4(w), sumA);
            sumB = fma(half4(v4, v5, v6, v7), half4(w), sumB);
            sumC = fma(half4(v8, v9, v10, v11), half4(w), sumC);
            sumD = fma(half4(v12, v13, v14, v15), half4(w), sumD);
            
            half next = scalarCache[localBaseIndex + k + 16];
            
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
        output[vecIndex] = float4(sumA);
        output[vecIndex + 1] = float4(sumB);
        output[vecIndex + 2] = float4(sumC);
        output[vecIndex + 3] = float4(sumD);
    } else {
        device float* outScalar = (device float*)output;
        for(int i=0; i<4; ++i) {
            if(globalOutputIndex + i < (int)outSize)
                outScalar[globalOutputIndex + i] = (float)sumA[i];
            if(globalOutputIndex + 4 + i < (int)outSize)
                outScalar[globalOutputIndex + 4 + i] = (float)sumB[i];
            if(globalOutputIndex + 8 + i < (int)outSize)
                outScalar[globalOutputIndex + 8 + i] = (float)sumC[i];
            if(globalOutputIndex + 12 + i < (int)outSize)
                outScalar[globalOutputIndex + 12 + i] = (float)sumD[i];
        }
    }
}
