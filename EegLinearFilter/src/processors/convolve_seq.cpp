//
//  convolve_seq.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 19.11.2025.
//

#include "processors.h"
#include <cstddef>
#include <stdexcept>
#include <arm_neon.h>

#define ALIGN_HINT(ptr) __builtin_assume_aligned((ptr), 16)

void convolve_seq_no_vec(NeonVector& data, const std::vector<float>& convolutionKernel, const int n) {
    const size_t dataSize = data.size();
    
    float* __restrict dataPtr = data.data();
    const float* __restrict kernelPtr = convolutionKernel.data();
    
    size_t outIndex = 0;
    
    for (size_t i = n; i < dataSize - n; ++i, ++outIndex) {
        float sum = 0.0f;
        
        #pragma clang loop vectorize(disable)
        for (int j = -n; j <= n; ++j) {
            sum += dataPtr[i + j] * kernelPtr[j + n];
        }
        dataPtr[outIndex] = sum;
    }
}

void convolve_seq_no_vec_w_unroll(NeonVector& data, const std::vector<float>& convolutionKernel, const int n) {
    const size_t dataSize = data.size();

    float* __restrict dataPtr = data.data();
    const float* __restrict kernelPtr = convolutionKernel.data();

    size_t outIndex = 0;

    for (size_t i = n; i < dataSize - n; ++i, ++outIndex) {
        int j = -n;
        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;
        
        #pragma clang loop vectorize(disable)
        for (; j <= n - 3; j += 4) {
            sum0 += dataPtr[i + j] * kernelPtr[j + n];
            sum1 += dataPtr[i + j + 1] * kernelPtr[j + n + 1];
            sum2 += dataPtr[i + j + 2] * kernelPtr[j + n + 2];
            sum3 += dataPtr[i + j + 3] * kernelPtr[j + n + 3];
        }

        float sum = (sum0 + sum1) + (sum2 + sum3);

        #pragma clang loop vectorize(disable)
        for (; j <= n; ++j) {
            sum += dataPtr[i + j] * kernelPtr[j + n];
        }

        dataPtr[outIndex] = sum;
    }
}

void convolve_seq_auto_vec(NeonVector& data, const std::vector<float>& convolutionKernel, const int n) {
    const size_t dataSize = data.size();
    
    float* __restrict dataPtr = data.data();
    const float* __restrict kernelPtr = convolutionKernel.data();
    
    size_t outIndex = 0;
    
    for (size_t i = n; i < dataSize - n; ++i, ++outIndex) {
        float sum = 0.0f;
        
        #pragma clang loop vectorize(enable)
        for (int j = -n; j <= n; ++j) {
            sum += dataPtr[i + j] * kernelPtr[j + n];
        }
        dataPtr[outIndex] = sum;
    }
}

void convolve_seq_manual_vec(NeonVector& data, const std::vector<float>& convolutionKernel, const int n) {
    const size_t dataSize = data.size();
    const size_t kernelSize = convolutionKernel.size();
    size_t paddedKernelSize = (kernelSize + 3) & ~3;
   
    std::vector<float> paddedKernel(paddedKernelSize, 0.0f);
    for(size_t i = 0; i < kernelSize; ++i) {
        paddedKernel[i] = convolutionKernel[i];
    }
    const size_t k_blocks = paddedKernelSize / 4;
    std::vector<float32x4_t> k_vecs(k_blocks);
    for (size_t b = 0; b < k_blocks; ++b) {
        k_vecs[b] = vld1q_f32(&paddedKernel[b * 4]);
    }
    const size_t out_count = dataSize - kernelSize + 1;
   
    float* __restrict io_ptr = static_cast<float*>(ALIGN_HINT(data.data()));
    size_t o = 0;
    
    for (; o + 8 <= out_count; o += 8) {
        __builtin_prefetch(io_ptr + o + 128, 0, 3);
        
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);
        float32x4_t sum4 = vdupq_n_f32(0.0f);
        float32x4_t sum5 = vdupq_n_f32(0.0f);
        float32x4_t sum6 = vdupq_n_f32(0.0f);
        float32x4_t sum7 = vdupq_n_f32(0.0f);
        
        const float* d_ptr = io_ptr + o;
        float32x4_t d_curr = vld1q_f32(d_ptr);
        for (size_t b = 0; b < k_blocks; ++b) {
            float32x4_t d_next = vld1q_f32(d_ptr + (b + 1) * 4);
           
            float32x4_t k = k_vecs[b];
            float32x4_t d1 = vextq_f32(d_curr, d_next, 1);
            float32x4_t d2 = vextq_f32(d_curr, d_next, 2);
            float32x4_t d3 = vextq_f32(d_curr, d_next, 3);
            sum0 = vfmaq_f32(sum0, d_curr, k);
            sum1 = vfmaq_f32(sum1, d1, k);
            sum2 = vfmaq_f32(sum2, d2, k);
            sum3 = vfmaq_f32(sum3, d3, k);
            
            float32x4_t d4 = vextq_f32(d_next, vld1q_f32(d_ptr + (b + 2) * 4), 0);
            float32x4_t d5 = vextq_f32(d_next, vld1q_f32(d_ptr + (b + 2) * 4), 1);
            float32x4_t d6 = vextq_f32(d_next, vld1q_f32(d_ptr + (b + 2) * 4), 2);
            float32x4_t d7 = vextq_f32(d_next, vld1q_f32(d_ptr + (b + 2) * 4), 3);
            sum4 = vfmaq_f32(sum4, d4, k);
            sum5 = vfmaq_f32(sum5, d5, k);
            sum6 = vfmaq_f32(sum6, d6, k);
            sum7 = vfmaq_f32(sum7, d7, k);
            
            d_curr = d_next;
        }
        io_ptr[o] = vaddvq_f32(sum0);
        io_ptr[o + 1] = vaddvq_f32(sum1);
        io_ptr[o + 2] = vaddvq_f32(sum2);
        io_ptr[o + 3] = vaddvq_f32(sum3);
        io_ptr[o + 4] = vaddvq_f32(sum4);
        io_ptr[o + 5] = vaddvq_f32(sum5);
        io_ptr[o + 6] = vaddvq_f32(sum6);
        io_ptr[o + 7] = vaddvq_f32(sum7);
    }
    
    for (; o + 4 <= out_count; o += 4) {
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);
        const float* d_ptr = io_ptr + o;
        float32x4_t d_curr = vld1q_f32(d_ptr);
        for (size_t b = 0; b < k_blocks; ++b) {
            float32x4_t d_next = vld1q_f32(d_ptr + (b + 1) * 4);
           
            float32x4_t k = k_vecs[b];
            float32x4_t d1 = vextq_f32(d_curr, d_next, 1);
            float32x4_t d2 = vextq_f32(d_curr, d_next, 2);
            float32x4_t d3 = vextq_f32(d_curr, d_next, 3);
            sum0 = vfmaq_f32(sum0, d_curr, k);
            sum1 = vfmaq_f32(sum1, d1, k);
            sum2 = vfmaq_f32(sum2, d2, k);
            sum3 = vfmaq_f32(sum3, d3, k);
            d_curr = d_next;
        }
        io_ptr[o] = vaddvq_f32(sum0);
        io_ptr[o + 1] = vaddvq_f32(sum1);
        io_ptr[o + 2] = vaddvq_f32(sum2);
        io_ptr[o + 3] = vaddvq_f32(sum3);
    }
    
    for (; o < out_count; ++o) {
        float sum = 0.0f;
        for (size_t k = 0; k < kernelSize; ++k) {
            sum += io_ptr[o + k] * convolutionKernel[k];
        }
        io_ptr[o] = sum;
    }
    
}
