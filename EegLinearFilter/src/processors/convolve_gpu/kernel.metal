//
//  kernel.metal
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 26.11.2025.
//

#include <metal_stdlib>
#include "kernel_config.h"

using namespace metal;

kernel void convolve_kernel(
    device const float4* data [[buffer(0)]],
    device float4* output [[buffer(1)]],
    constant float* convKernel [[buffer(2)]],
    constant uint& kernelSize [[buffer(3)]],
    constant uint& outSize [[buffer(4)]],
    threadgroup float4* cache [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    int groupStartGlobal = group_id * TILE_SIZE;
    int kSize = (int)kernelSize;
    int totalCacheNeeded = TILE_SIZE + kSize - 1;

    int vectorsNeeded = (totalCacheNeeded + 3) / 4;

    for (int i = tid; i < vectorsNeeded; i += THREADS_PER_GROUP) {
        int globalVecIdx = (groupStartGlobal / 4) + i;
        int limit = (int)(outSize + kSize - 1);
        if ((globalVecIdx * 4) < limit) {
            cache[i] = data[globalVecIdx];
        } else {
            cache[i] = float4(0.0f);
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* scalarCache = (threadgroup float*)cache;

    int localBaseIndex = tid * ITEMS_PER_THREAD;
    
    float4 sumA = float4(0.0f);
    float4 sumB = float4(0.0f);
    float4 sumC = float4(0.0f);
    float4 sumD = float4(0.0f);

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
    int limitLoop = kSize - 15;
    
    for (; k < limitLoop; k += 16) {
        float w;
        
        w = convKernel[k];
        sumA = fma(float4(v0, v1, v2, v3), float4(w), sumA);
        sumB = fma(float4(v4, v5, v6, v7), float4(w), sumB);
        sumC = fma(float4(v8, v9, v10, v11), float4(w), sumC);
        sumD = fma(float4(v12, v13, v14, v15), float4(w), sumD);
        v0 = scalarCache[localBaseIndex + k + 16];

        w = convKernel[k+1];
        sumA = fma(float4(v1, v2, v3, v4), float4(w), sumA);
        sumB = fma(float4(v5, v6, v7, v8), float4(w), sumB);
        sumC = fma(float4(v9, v10, v11, v12), float4(w), sumC);
        sumD = fma(float4(v13, v14, v15, v0), float4(w), sumD);
        v1 = scalarCache[localBaseIndex + k + 17];

        w = convKernel[k+2];
        sumA = fma(float4(v2, v3, v4, v5), float4(w), sumA);
        sumB = fma(float4(v6, v7, v8, v9), float4(w), sumB);
        sumC = fma(float4(v10, v11, v12, v13), float4(w), sumC);
        sumD = fma(float4(v14, v15, v0, v1), float4(w), sumD);
        v2 = scalarCache[localBaseIndex + k + 18];

        w = convKernel[k+3];
        sumA = fma(float4(v3, v4, v5, v6), float4(w), sumA);
        sumB = fma(float4(v7, v8, v9, v10), float4(w), sumB);
        sumC = fma(float4(v11, v12, v13, v14), float4(w), sumC);
        sumD = fma(float4(v15, v0, v1, v2), float4(w), sumD);
        v3 = scalarCache[localBaseIndex + k + 19];

        w = convKernel[k+4];
        sumA = fma(float4(v4, v5, v6, v7), float4(w), sumA);
        sumB = fma(float4(v8, v9, v10, v11), float4(w), sumB);
        sumC = fma(float4(v12, v13, v14, v15), float4(w), sumC);
        sumD = fma(float4(v0, v1, v2, v3), float4(w), sumD);
        v4 = scalarCache[localBaseIndex + k + 20];

        w = convKernel[k+5];
        sumA = fma(float4(v5, v6, v7, v8), float4(w), sumA);
        sumB = fma(float4(v9, v10, v11, v12), float4(w), sumB);
        sumC = fma(float4(v13, v14, v15, v0), float4(w), sumC);
        sumD = fma(float4(v1, v2, v3, v4), float4(w), sumD);
        v5 = scalarCache[localBaseIndex + k + 21];

        w = convKernel[k+6];
        sumA = fma(float4(v6, v7, v8, v9), float4(w), sumA);
        sumB = fma(float4(v10, v11, v12, v13), float4(w), sumB);
        sumC = fma(float4(v14, v15, v0, v1), float4(w), sumC);
        sumD = fma(float4(v2, v3, v4, v5), float4(w), sumD);
        v6 = scalarCache[localBaseIndex + k + 22];

        w = convKernel[k+7];
        sumA = fma(float4(v7, v8, v9, v10), float4(w), sumA);
        sumB = fma(float4(v11, v12, v13, v14), float4(w), sumB);
        sumC = fma(float4(v15, v0, v1, v2), float4(w), sumC);
        sumD = fma(float4(v3, v4, v5, v6), float4(w), sumD);
        v7 = scalarCache[localBaseIndex + k + 23];

        w = convKernel[k+8];
        sumA = fma(float4(v8, v9, v10, v11), float4(w), sumA);
        sumB = fma(float4(v12, v13, v14, v15), float4(w), sumB);
        sumC = fma(float4(v0, v1, v2, v3), float4(w), sumC);
        sumD = fma(float4(v4, v5, v6, v7), float4(w), sumD);
        v8 = scalarCache[localBaseIndex + k + 24];

        w = convKernel[k+9];
        sumA = fma(float4(v9, v10, v11, v12), float4(w), sumA);
        sumB = fma(float4(v13, v14, v15, v0), float4(w), sumB);
        sumC = fma(float4(v1, v2, v3, v4), float4(w), sumC);
        sumD = fma(float4(v5, v6, v7, v8), float4(w), sumD);
        v9 = scalarCache[localBaseIndex + k + 25];

        w = convKernel[k+10];
        sumA = fma(float4(v10, v11, v12, v13), float4(w), sumA);
        sumB = fma(float4(v14, v15, v0, v1), float4(w), sumB);
        sumC = fma(float4(v2, v3, v4, v5), float4(w), sumC);
        sumD = fma(float4(v6, v7, v8, v9), float4(w), sumD);
        v10 = scalarCache[localBaseIndex + k + 26];

        w = convKernel[k+11];
        sumA = fma(float4(v11, v12, v13, v14), float4(w), sumA);
        sumB = fma(float4(v15, v0, v1, v2), float4(w), sumB);
        sumC = fma(float4(v3, v4, v5, v6), float4(w), sumC);
        sumD = fma(float4(v7, v8, v9, v10), float4(w), sumD);
        v11 = scalarCache[localBaseIndex + k + 27];

        w = convKernel[k+12];
        sumA = fma(float4(v12, v13, v14, v15), float4(w), sumA);
        sumB = fma(float4(v0, v1, v2, v3), float4(w), sumB);
        sumC = fma(float4(v4, v5, v6, v7), float4(w), sumC);
        sumD = fma(float4(v8, v9, v10, v11), float4(w), sumD);
        v12 = scalarCache[localBaseIndex + k + 28];

        w = convKernel[k+13];
        sumA = fma(float4(v13, v14, v15, v0), float4(w), sumA);
        sumB = fma(float4(v1, v2, v3, v4), float4(w), sumB);
        sumC = fma(float4(v5, v6, v7, v8), float4(w), sumC);
        sumD = fma(float4(v9, v10, v11, v12), float4(w), sumD);
        v13 = scalarCache[localBaseIndex + k + 29];

        w = convKernel[k+14];
        sumA = fma(float4(v14, v15, v0, v1), float4(w), sumA);
        sumB = fma(float4(v2, v3, v4, v5), float4(w), sumB);
        sumC = fma(float4(v6, v7, v8, v9), float4(w), sumC);
        sumD = fma(float4(v10, v11, v12, v13), float4(w), sumD);
        v14 = scalarCache[localBaseIndex + k + 30];

        w = convKernel[k+15];
        sumA = fma(float4(v15, v0, v1, v2), float4(w), sumA);
        sumB = fma(float4(v3, v4, v5, v6), float4(w), sumB);
        sumC = fma(float4(v7, v8, v9, v10), float4(w), sumC);
        sumD = fma(float4(v11, v12, v13, v14), float4(w), sumD);
        v15 = scalarCache[localBaseIndex + k + 31];
    }
    
    for (; k < kSize; ++k) {
        float w = convKernel[k];
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
