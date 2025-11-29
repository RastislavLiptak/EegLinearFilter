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
    device const float* data [[buffer(0)]],
    device float4* output [[buffer(1)]],
    constant float* convKernel [[buffer(2)]],
    constant uint& kernelSize [[buffer(3)]],
    constant uint& outSize [[buffer(4)]],
    threadgroup float* cache [[threadgroup(0)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]
) {
    int groupStartGlobal = group_id * TILE_SIZE;
    int kSize = (int)kernelSize;
    int totalCacheNeeded = TILE_SIZE + kSize - 1;
    
    for (int i = tid; i < totalCacheNeeded; i += THREADS_PER_GROUP) {
        int globalIdx = groupStartGlobal + i;
        bool inBounds = globalIdx < (int)(outSize + kSize - 1);
        cache[i] = inBounds ? data[globalIdx] : 0.0f;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    int localBaseIndex = tid * ITEMS_PER_THREAD;
    
    float4 sumA = float4(0.0f);
    float4 sumB = float4(0.0f);
    float4 sumC = float4(0.0f);
    float4 sumD = float4(0.0f);

    float v0 = cache[localBaseIndex + 0];
    float v1 = cache[localBaseIndex + 1];
    float v2 = cache[localBaseIndex + 2];
    float v3 = cache[localBaseIndex + 3];
    float v4 = cache[localBaseIndex + 4];
    float v5 = cache[localBaseIndex + 5];
    float v6 = cache[localBaseIndex + 6];
    float v7 = cache[localBaseIndex + 7];
    float v8 = cache[localBaseIndex + 8];
    float v9 = cache[localBaseIndex + 9];
    float v10 = cache[localBaseIndex + 10];
    float v11 = cache[localBaseIndex + 11];
    float v12 = cache[localBaseIndex + 12];
    float v13 = cache[localBaseIndex + 13];
    float v14 = cache[localBaseIndex + 14];
    float v15 = cache[localBaseIndex + 15];

    int k = 0;
    int limit = kSize - 15;
    
    for (; k < limit; k += 16) {
        float w;
        
        w = convKernel[k];
        sumA = fma(float4(v0, v1, v2, v3), float4(w), sumA);
        sumB = fma(float4(v4, v5, v6, v7), float4(w), sumB);
        sumC = fma(float4(v8, v9, v10, v11), float4(w), sumC);
        sumD = fma(float4(v12, v13, v14, v15), float4(w), sumD);
        v0 = cache[localBaseIndex + k + 16];

        w = convKernel[k+1];
        sumA = fma(float4(v1, v2, v3, v4), float4(w), sumA);
        sumB = fma(float4(v5, v6, v7, v8), float4(w), sumB);
        sumC = fma(float4(v9, v10, v11, v12), float4(w), sumC);
        sumD = fma(float4(v13, v14, v15, v0), float4(w), sumD);
        v1 = cache[localBaseIndex + k + 17];

        w = convKernel[k+2];
        sumA = fma(float4(v2, v3, v4, v5), float4(w), sumA);
        sumB = fma(float4(v6, v7, v8, v9), float4(w), sumB);
        sumC = fma(float4(v10, v11, v12, v13), float4(w), sumC);
        sumD = fma(float4(v14, v15, v0, v1), float4(w), sumD);
        v2 = cache[localBaseIndex + k + 18];

        w = convKernel[k+3];
        sumA = fma(float4(v3, v4, v5, v6), float4(w), sumA);
        sumB = fma(float4(v7, v8, v9, v10), float4(w), sumB);
        sumC = fma(float4(v11, v12, v13, v14), float4(w), sumC);
        sumD = fma(float4(v15, v0, v1, v2), float4(w), sumD);
        v3 = cache[localBaseIndex + k + 19];

        w = convKernel[k+4];
        sumA = fma(float4(v4, v5, v6, v7), float4(w), sumA);
        sumB = fma(float4(v8, v9, v10, v11), float4(w), sumB);
        sumC = fma(float4(v12, v13, v14, v15), float4(w), sumC);
        sumD = fma(float4(v0, v1, v2, v3), float4(w), sumD);
        v4 = cache[localBaseIndex + k + 20];

        w = convKernel[k+5];
        sumA = fma(float4(v5, v6, v7, v8), float4(w), sumA);
        sumB = fma(float4(v9, v10, v11, v12), float4(w), sumB);
        sumC = fma(float4(v13, v14, v15, v0), float4(w), sumC);
        sumD = fma(float4(v1, v2, v3, v4), float4(w), sumD);
        v5 = cache[localBaseIndex + k + 21];

        w = convKernel[k+6];
        sumA = fma(float4(v6, v7, v8, v9), float4(w), sumA);
        sumB = fma(float4(v10, v11, v12, v13), float4(w), sumB);
        sumC = fma(float4(v14, v15, v0, v1), float4(w), sumC);
        sumD = fma(float4(v2, v3, v4, v5), float4(w), sumD);
        v6 = cache[localBaseIndex + k + 22];

        w = convKernel[k+7];
        sumA = fma(float4(v7, v8, v9, v10), float4(w), sumA);
        sumB = fma(float4(v11, v12, v13, v14), float4(w), sumB);
        sumC = fma(float4(v15, v0, v1, v2), float4(w), sumC);
        sumD = fma(float4(v3, v4, v5, v6), float4(w), sumD);
        v7 = cache[localBaseIndex + k + 23];

        w = convKernel[k+8];
        sumA = fma(float4(v8, v9, v10, v11), float4(w), sumA);
        sumB = fma(float4(v12, v13, v14, v15), float4(w), sumB);
        sumC = fma(float4(v0, v1, v2, v3), float4(w), sumC);
        sumD = fma(float4(v4, v5, v6, v7), float4(w), sumD);
        v8 = cache[localBaseIndex + k + 24];

        w = convKernel[k+9];
        sumA = fma(float4(v9, v10, v11, v12), float4(w), sumA);
        sumB = fma(float4(v13, v14, v15, v0), float4(w), sumB);
        sumC = fma(float4(v1, v2, v3, v4), float4(w), sumC);
        sumD = fma(float4(v5, v6, v7, v8), float4(w), sumD);
        v9 = cache[localBaseIndex + k + 25];

        w = convKernel[k+10];
        sumA = fma(float4(v10, v11, v12, v13), float4(w), sumA);
        sumB = fma(float4(v14, v15, v0, v1), float4(w), sumB);
        sumC = fma(float4(v2, v3, v4, v5), float4(w), sumC);
        sumD = fma(float4(v6, v7, v8, v9), float4(w), sumD);
        v10 = cache[localBaseIndex + k + 26];

        w = convKernel[k+11];
        sumA = fma(float4(v11, v12, v13, v14), float4(w), sumA);
        sumB = fma(float4(v15, v0, v1, v2), float4(w), sumB);
        sumC = fma(float4(v3, v4, v5, v6), float4(w), sumC);
        sumD = fma(float4(v7, v8, v9, v10), float4(w), sumD);
        v11 = cache[localBaseIndex + k + 27];

        w = convKernel[k+12];
        sumA = fma(float4(v12, v13, v14, v15), float4(w), sumA);
        sumB = fma(float4(v0, v1, v2, v3), float4(w), sumB);
        sumC = fma(float4(v4, v5, v6, v7), float4(w), sumC);
        sumD = fma(float4(v8, v9, v10, v11), float4(w), sumD);
        v12 = cache[localBaseIndex + k + 28];

        w = convKernel[k+13];
        sumA = fma(float4(v13, v14, v15, v0), float4(w), sumA);
        sumB = fma(float4(v1, v2, v3, v4), float4(w), sumB);
        sumC = fma(float4(v5, v6, v7, v8), float4(w), sumC);
        sumD = fma(float4(v9, v10, v11, v12), float4(w), sumD);
        v13 = cache[localBaseIndex + k + 29];

        w = convKernel[k+14];
        sumA = fma(float4(v14, v15, v0, v1), float4(w), sumA);
        sumB = fma(float4(v2, v3, v4, v5), float4(w), sumB);
        sumC = fma(float4(v6, v7, v8, v9), float4(w), sumC);
        sumD = fma(float4(v10, v11, v12, v13), float4(w), sumD);
        v14 = cache[localBaseIndex + k + 30];

        w = convKernel[k+15];
        sumA = fma(float4(v15, v0, v1, v2), float4(w), sumA);
        sumB = fma(float4(v3, v4, v5, v6), float4(w), sumB);
        sumC = fma(float4(v7, v8, v9, v10), float4(w), sumC);
        sumD = fma(float4(v11, v12, v13, v14), float4(w), sumD);
        v15 = cache[localBaseIndex + k + 31];
    }
    
    for (; k < kSize; ++k) {
        float w = convKernel[k];
        sumA = fma(float4(v0, v1, v2, v3), float4(w), sumA);
        sumB = fma(float4(v4, v5, v6, v7), float4(w), sumB);
        sumC = fma(float4(v8, v9, v10, v11), float4(w), sumC);
        sumD = fma(float4(v12, v13, v14, v15), float4(w), sumD);
        
        float next = cache[localBaseIndex + k + 16];
        
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
