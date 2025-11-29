//
//  kernel.metal
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 26.11.2025.
//

#include <metal_stdlib>
using namespace metal;

constant int THREADS_PER_GROUP = 256;
constant int ITEMS_PER_THREAD = 12;
constant int TILE_SIZE = THREADS_PER_GROUP * ITEMS_PER_THREAD;

kernel void convolve_kernel_optimized_12x(
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

    int k = 0;
    int limit = kSize - 11;
    
    for (; k < limit; k += 12) {
        float w0 = convKernel[k];
        sumA = fma(float4(v0, v1, v2, v3), float4(w0), sumA);
        sumB = fma(float4(v4, v5, v6, v7), float4(w0), sumB);
        sumC = fma(float4(v8, v9, v10, v11), float4(w0), sumC);
        v0 = cache[localBaseIndex + k + 12];

        float w1 = convKernel[k+1];
        sumA = fma(float4(v1, v2, v3, v4), float4(w1), sumA);
        sumB = fma(float4(v5, v6, v7, v8), float4(w1), sumB);
        sumC = fma(float4(v9, v10, v11, v0), float4(w1), sumC);
        v1 = cache[localBaseIndex + k + 13];

        float w2 = convKernel[k+2];
        sumA = fma(float4(v2, v3, v4, v5), float4(w2), sumA);
        sumB = fma(float4(v6, v7, v8, v9), float4(w2), sumB);
        sumC = fma(float4(v10, v11, v0, v1), float4(w2), sumC);
        v2 = cache[localBaseIndex + k + 14];

        float w3 = convKernel[k+3];
        sumA = fma(float4(v3, v4, v5, v6), float4(w3), sumA);
        sumB = fma(float4(v7, v8, v9, v10), float4(w3), sumB);
        sumC = fma(float4(v11, v0, v1, v2), float4(w3), sumC);
        v3 = cache[localBaseIndex + k + 15];

        float w4 = convKernel[k+4];
        sumA = fma(float4(v4, v5, v6, v7), float4(w4), sumA);
        sumB = fma(float4(v8, v9, v10, v11), float4(w4), sumB);
        sumC = fma(float4(v0, v1, v2, v3), float4(w4), sumC);
        v4 = cache[localBaseIndex + k + 16];

        float w5 = convKernel[k+5];
        sumA = fma(float4(v5, v6, v7, v8), float4(w5), sumA);
        sumB = fma(float4(v9, v10, v11, v0), float4(w5), sumB);
        sumC = fma(float4(v1, v2, v3, v4), float4(w5), sumC);
        v5 = cache[localBaseIndex + k + 17];

        float w6 = convKernel[k+6];
        sumA = fma(float4(v6, v7, v8, v9), float4(w6), sumA);
        sumB = fma(float4(v10, v11, v0, v1), float4(w6), sumB);
        sumC = fma(float4(v2, v3, v4, v5), float4(w6), sumC);
        v6 = cache[localBaseIndex + k + 18];

        float w7 = convKernel[k+7];
        sumA = fma(float4(v7, v8, v9, v10), float4(w7), sumA);
        sumB = fma(float4(v11, v0, v1, v2), float4(w7), sumB);
        sumC = fma(float4(v3, v4, v5, v6), float4(w7), sumC);
        v7 = cache[localBaseIndex + k + 19];

        float w8 = convKernel[k+8];
        sumA = fma(float4(v8, v9, v10, v11), float4(w8), sumA);
        sumB = fma(float4(v0, v1, v2, v3), float4(w8), sumB);
        sumC = fma(float4(v4, v5, v6, v7), float4(w8), sumC);
        v8 = cache[localBaseIndex + k + 20];

        float w9 = convKernel[k+9];
        sumA = fma(float4(v9, v10, v11, v0), float4(w9), sumA);
        sumB = fma(float4(v1, v2, v3, v4), float4(w9), sumB);
        sumC = fma(float4(v5, v6, v7, v8), float4(w9), sumC);
        v9 = cache[localBaseIndex + k + 21];

        float w10 = convKernel[k+10];
        sumA = fma(float4(v10, v11, v0, v1), float4(w10), sumA);
        sumB = fma(float4(v2, v3, v4, v5), float4(w10), sumB);
        sumC = fma(float4(v6, v7, v8, v9), float4(w10), sumC);
        v10 = cache[localBaseIndex + k + 22];

        float w11 = convKernel[k+11];
        sumA = fma(float4(v11, v0, v1, v2), float4(w11), sumA);
        sumB = fma(float4(v3, v4, v5, v6), float4(w11), sumB);
        sumC = fma(float4(v7, v8, v9, v10), float4(w11), sumC);
        v11 = cache[localBaseIndex + k + 23];
    }
    
    for (; k < kSize; ++k) {
        float w = convKernel[k];
        sumA = fma(float4(v0, v1, v2, v3), float4(w), sumA);
        sumB = fma(float4(v4, v5, v6, v7), float4(w), sumB);
        sumC = fma(float4(v8, v9, v10, v11), float4(w), sumC);
        
        float next = cache[localBaseIndex + k + 12];
        v0 = v1; v1 = v2; v2 = v3; v3 = v4; v4 = v5; v5 = v6;
        v6 = v7; v7 = v8; v8 = v9; v9 = v10; v10 = v11; v11 = next;
    }
    
    int globalOutputIndex = groupStartGlobal + localBaseIndex;
    
    if (globalOutputIndex + 11 < (int)outSize) {
        int vecIndex = globalOutputIndex / 4;
        output[vecIndex] = sumA;
        output[vecIndex + 1] = sumB;
        output[vecIndex + 2] = sumC;
    } else {
        device float* outScalar = (device float*)output;
        for(int i=0; i<4; ++i) {
            if(globalOutputIndex + i < (int)outSize)
                outScalar[globalOutputIndex + i] = sumA[i];
            if(globalOutputIndex + 4 + i < (int)outSize)
                outScalar[globalOutputIndex + 4 + i] = sumB[i];
            if(globalOutputIndex + 8 + i < (int)outSize)
                outScalar[globalOutputIndex + 8 + i] = sumC[i];
        }
    }
}
