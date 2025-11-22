//
//  convolve_par.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 21.11.2025.
//

#include "processors.h"
#include <dispatch/dispatch.h>
#include <arm_neon.h>

void convolve_par_no_vec(const NeonVector& data, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel, const int n) {
    const size_t dataSize = data.size();
    const size_t outSize = dataSize - 2 * n;

    const float* __restrict dataPtr = data.data();
    float* __restrict outputPtr = outputBuffer.data();
    const float* __restrict kernelPtr = convolutionKernel.data();

    dispatch_apply(outSize, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t outIndex) {
        const size_t i = outIndex + n;

        float sum = 0.0f;

        #pragma clang loop vectorize(disable)
        for (int j = -n; j <= n; ++j) {
            sum += dataPtr[i + j] * kernelPtr[j + n];
        }
        outputPtr[outIndex] = sum;
    });
}

void convolve_par_no_vec_w_unroll(const NeonVector& data, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel, const int n) {
    const size_t dataSize = data.size();
    const size_t outSize = dataSize - 2 * n;
    
    const float* __restrict dataPtr = data.data();
    float* __restrict outputPtr = outputBuffer.data();
    const float* __restrict kernelPtr = convolutionKernel.data();

    dispatch_apply(outSize, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t outIndex) {

        const size_t i = outIndex + n;

        int j = -n;

        float sum0 = 0.0f;
        float sum1 = 0.0f;
        float sum2 = 0.0f;
        float sum3 = 0.0f;

        #pragma clang loop vectorize(disable)
        for (; j <= n - 3; j += 4) {
            sum0 += dataPtr[i + j]     * kernelPtr[j + n];
            sum1 += dataPtr[i + j + 1] * kernelPtr[j + n + 1];
            sum2 += dataPtr[i + j + 2] * kernelPtr[j + n + 2];
            sum3 += dataPtr[i + j + 3] * kernelPtr[j + n + 3];
        }

        float sum = (sum0 + sum1) + (sum2 + sum3);

        #pragma clang loop vectorize(disable)
        for (; j <= n; ++j) {
            sum += dataPtr[i + j] * kernelPtr[j + n];
        }

        outputPtr[outIndex] = sum;
    });
}

void convolve_par_auto_vec(const NeonVector& data, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel, const int n) {
    const size_t dataSize = data.size();
    const size_t outSize = dataSize - 2 * n;

    const float* __restrict dataPtr = data.data();
    float* __restrict outputPtr = outputBuffer.data();
    const float* __restrict kernelPtr = convolutionKernel.data();

    dispatch_apply(outSize, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t outIndex) {
        const size_t i = outIndex + n;

        float sum = 0.0f;

        #pragma clang loop vectorize(enable)
        for (int j = -n; j <= n; ++j) {
            sum += dataPtr[i + j] * kernelPtr[j + n];
        }
        outputPtr[outIndex] = sum;
    });
}

#define ALIGN_HINT(ptr) __builtin_assume_aligned((ptr), 16)

void convolve_par_manual_vec(const NeonVector& data, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel) {
    const size_t dataSize = data.size();
    const size_t kernelSize = convolutionKernel.size();

    const size_t outSize = dataSize - kernelSize + 1;

    const size_t paddedKernelSize = (kernelSize + 3) & ~3;
    std::vector<float> paddedKernel(paddedKernelSize, 0.0f);
    std::memcpy(paddedKernel.data(), convolutionKernel.data(), kernelSize * sizeof(float));

    const size_t k_blocks = paddedKernelSize / 4;
    std::vector<float32x4_t> k_vecs(k_blocks);
    for (size_t b = 0; b < k_blocks; ++b) {
        k_vecs[b] = vld1q_f32(&paddedKernel[b * 4]);
    }

    const float* __restrict inDataPtr = data.data();
    float* __restrict outDataPtr = outputBuffer.data();
    const float32x4_t* k_vecs_ptr = k_vecs.data();

    const size_t CHUNK_SIZE = 16384;
    const size_t numChunks = (outSize + CHUNK_SIZE - 1) / CHUNK_SIZE;

    dispatch_apply(numChunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t chunkIndex) {
        
        size_t o = chunkIndex * CHUNK_SIZE;
        const size_t end = std::min(o + CHUNK_SIZE, outSize);

        for (; o + 16 <= end; o += 16) {
            float32x4_t sum0 = vdupq_n_f32(0.0f);
            float32x4_t sum1 = vdupq_n_f32(0.0f);
            float32x4_t sum2 = vdupq_n_f32(0.0f);
            float32x4_t sum3 = vdupq_n_f32(0.0f);
            float32x4_t sum4 = vdupq_n_f32(0.0f);
            float32x4_t sum5 = vdupq_n_f32(0.0f);
            float32x4_t sum6 = vdupq_n_f32(0.0f);
            float32x4_t sum7 = vdupq_n_f32(0.0f);
            float32x4_t sum8 = vdupq_n_f32(0.0f);
            float32x4_t sum9 = vdupq_n_f32(0.0f);
            float32x4_t sum10 = vdupq_n_f32(0.0f);
            float32x4_t sum11 = vdupq_n_f32(0.0f);
            float32x4_t sum12 = vdupq_n_f32(0.0f);
            float32x4_t sum13 = vdupq_n_f32(0.0f);
            float32x4_t sum14 = vdupq_n_f32(0.0f);
            float32x4_t sum15 = vdupq_n_f32(0.0f);

            const float* d_ptr = inDataPtr + o;

            float32x4_t d0 = vld1q_f32(d_ptr);
            float32x4_t d1 = vld1q_f32(d_ptr + 4);
            float32x4_t d2 = vld1q_f32(d_ptr + 8);
            float32x4_t d3 = vld1q_f32(d_ptr + 12);
            float32x4_t d4;

            for (size_t b = 0; b < k_blocks; ++b) {
                const float32x4_t k = k_vecs_ptr[b];

                d4 = vld1q_f32(d_ptr + (b + 4) * 4);

                sum0 = vfmaq_f32(sum0, d0, k);
                sum1 = vfmaq_f32(sum1, vextq_f32(d0, d1, 1), k);
                sum2 = vfmaq_f32(sum2, vextq_f32(d0, d1, 2), k);
                sum3 = vfmaq_f32(sum3, vextq_f32(d0, d1, 3), k);

                sum4 = vfmaq_f32(sum4, d1, k);
                sum5 = vfmaq_f32(sum5, vextq_f32(d1, d2, 1), k);
                sum6 = vfmaq_f32(sum6, vextq_f32(d1, d2, 2), k);
                sum7 = vfmaq_f32(sum7, vextq_f32(d1, d2, 3), k);

                sum8 = vfmaq_f32(sum8, d2, k);
                sum9 = vfmaq_f32(sum9, vextq_f32(d2, d3, 1), k);
                sum10 = vfmaq_f32(sum10, vextq_f32(d2, d3, 2), k);
                sum11 = vfmaq_f32(sum11, vextq_f32(d2, d3, 3), k);

                sum12 = vfmaq_f32(sum12, d3, k);
                sum13 = vfmaq_f32(sum13, vextq_f32(d3, d4, 1), k);
                sum14 = vfmaq_f32(sum14, vextq_f32(d3, d4, 2), k);
                sum15 = vfmaq_f32(sum15, vextq_f32(d3, d4, 3), k);

                d0 = d1;
                d1 = d2;
                d2 = d3;
                d3 = d4;
            }

            outDataPtr[o + 0] = vaddvq_f32(sum0);
            outDataPtr[o + 1] = vaddvq_f32(sum1);
            outDataPtr[o + 2] = vaddvq_f32(sum2);
            outDataPtr[o + 3] = vaddvq_f32(sum3);
            outDataPtr[o + 4] = vaddvq_f32(sum4);
            outDataPtr[o + 5] = vaddvq_f32(sum5);
            outDataPtr[o + 6] = vaddvq_f32(sum6);
            outDataPtr[o + 7] = vaddvq_f32(sum7);
            outDataPtr[o + 8] = vaddvq_f32(sum8);
            outDataPtr[o + 9] = vaddvq_f32(sum9);
            outDataPtr[o + 10] = vaddvq_f32(sum10);
            outDataPtr[o + 11] = vaddvq_f32(sum11);
            outDataPtr[o + 12] = vaddvq_f32(sum12);
            outDataPtr[o + 13] = vaddvq_f32(sum13);
            outDataPtr[o + 14] = vaddvq_f32(sum14);
            outDataPtr[o + 15] = vaddvq_f32(sum15);
        }

        for (; o + 4 <= end; o += 4) {
             float32x4_t sum0 = vdupq_n_f32(0.0f);
             float32x4_t sum1 = vdupq_n_f32(0.0f);
             float32x4_t sum2 = vdupq_n_f32(0.0f);
             float32x4_t sum3 = vdupq_n_f32(0.0f);
             
             const float* d_ptr = inDataPtr + o;
             float32x4_t d_curr = vld1q_f32(d_ptr);
             
             for (size_t b = 0; b < k_blocks; ++b) {
                 float32x4_t d_next = vld1q_f32(d_ptr + (b + 1) * 4);
                 float32x4_t k = k_vecs_ptr[b];
                 
                 sum0 = vfmaq_f32(sum0, d_curr, k);
                 sum1 = vfmaq_f32(sum1, vextq_f32(d_curr, d_next, 1), k);
                 sum2 = vfmaq_f32(sum2, vextq_f32(d_curr, d_next, 2), k);
                 sum3 = vfmaq_f32(sum3, vextq_f32(d_curr, d_next, 3), k);
                 
                 d_curr = d_next;
             }
             outDataPtr[o + 0] = vaddvq_f32(sum0);
             outDataPtr[o + 1] = vaddvq_f32(sum1);
             outDataPtr[o + 2] = vaddvq_f32(sum2);
             outDataPtr[o + 3] = vaddvq_f32(sum3);
        }

        for (; o < end; ++o) {
            float sum = 0.0f;
            for (size_t k = 0; k < kernelSize; ++k) {
                sum += inDataPtr[o + k] * convolutionKernel[k];
            }
            outDataPtr[o] = sum;
        }
    });
}
