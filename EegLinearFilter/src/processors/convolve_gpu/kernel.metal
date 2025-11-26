//
//  kernel.metal
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 26.11.2025.
//

#include <metal_stdlib>
using namespace metal;

kernel void convolution_kernel(
    device const float* inData      [[ buffer(0) ]],
    device float* outData           [[ buffer(1) ]],
    device const float* kernelData  [[ buffer(2) ]],
    constant uint& kernelSize       [[ buffer(3) ]],
    constant uint& dataSize         [[ buffer(4) ]],
    uint gid                        [[ thread_position_in_grid ]]
) {
    uint outCount = dataSize - kernelSize + 1;
    if (gid >= outCount) return;

    float sum = outData[gid];
    for (uint k = 0; k < kernelSize; ++k) {
        sum += inData[gid + k] * kernelData[k];
    }
    outData[gid] = sum;
}
