//
//  kernel.metal
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 26.11.2025.
//

#include <metal_stdlib>
using namespace metal;

constant int THREADS_PER_GROUP = 256;
constant int ITEMS_PER_THREAD = 4;
constant int TILE_SIZE = THREADS_PER_GROUP * ITEMS_PER_THREAD;

kernel void convolve_corrected(device const float* data [[buffer(0)]],
                               device float4* output [[buffer(1)]],
                               constant float* convKernel [[buffer(2)]],
                               constant uint& kernelSize [[buffer(3)]],
                               constant uint& outSize [[buffer(4)]],
                               threadgroup float* cache [[threadgroup(0)]],
                               uint gid [[thread_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint group_id [[threadgroup_position_in_grid]])
{
    int groupStartGlobal = group_id * TILE_SIZE;
    int kSize = (int)kernelSize;
    int totalCacheNeeded = TILE_SIZE + kSize - 1;
    
    for (int i = tid; i < totalCacheNeeded; i += THREADS_PER_GROUP) {
        int globalIdx = groupStartGlobal + i;
        if (globalIdx < (int)(outSize + kSize - 1)) {
            cache[i] = data[globalIdx];
        } else {
            cache[i] = 0.0f;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    int localBaseIndex = tid * ITEMS_PER_THREAD;
    int globalOutputIndex = groupStartGlobal + localBaseIndex;
    
    if (globalOutputIndex < (int)outSize) {
        float4 sum = float4(0.0f);
        
        for (int k = 0; k < kSize; ++k) {
            float kVal = convKernel[k];
            
            float d0 = cache[localBaseIndex + k + 0];
            float d1 = cache[localBaseIndex + k + 1];
            float d2 = cache[localBaseIndex + k + 2];
            float d3 = cache[localBaseIndex + k + 3];
            
            sum = fma(float4(d0, d1, d2, d3), float4(kVal), sum);
        }
        
        if (globalOutputIndex + 3 < (int)outSize) {
            output[gid] = sum;
        } else {
            device float* outScalar = (device float*)output;
            for(int i=0; i<4; ++i) {
                if(globalOutputIndex + i < (int)outSize)
                    outScalar[globalOutputIndex + i] = sum[i];
            }
        }
    }
}
