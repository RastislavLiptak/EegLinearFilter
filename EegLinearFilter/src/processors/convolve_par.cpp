//
//  convolve_par.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 21.11.2025.
//

#include "processors.h"
#include <dispatch/dispatch.h>
#include <arm_neon.h>

#define ALIGN_HINT(ptr) __builtin_assume_aligned((ptr), 16)

void convolve_par_no_vec(NeonVector& data, const std::vector<float>& convolutionKernel, const int n) {
    const size_t dataSize = data.size();
    const size_t outSize = dataSize - 2 * n;

    float* __restrict dataPtr = data.data();
    const float* __restrict kernelPtr = convolutionKernel.data();

    dispatch_apply(outSize, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t outIndex) {
        const size_t i = outIndex + n;

        float sum = 0.0f;

        #pragma clang loop vectorize(disable)
        for (int j = -n; j <= n; ++j) {
            sum += dataPtr[i + j] * kernelPtr[j + n];
        }
        dataPtr[outIndex] = sum;
    });
}

void convolve_par_no_vec_w_unroll(NeonVector& data, const std::vector<float>& convolutionKernel, const int n) {
    const size_t dataSize = data.size();
    const size_t outSize = dataSize - 2 * n;

    float* __restrict dataPtr = data.data();
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

        dataPtr[outIndex] = sum;
    });
}

void convolve_par_auto_vec(NeonVector& data, const std::vector<float>& convolutionKernel, const int n) {
    const size_t dataSize = data.size();
    const size_t outSize = dataSize - 2 * n;

    float* __restrict dataPtr = data.data();
    const float* __restrict kernelPtr = convolutionKernel.data();

    dispatch_apply(outSize, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t outIndex) {
        const size_t i = outIndex + n;

        float sum = 0.0f;

        #pragma clang loop vectorize(enable)
        for (int j = -n; j <= n; ++j) {
            sum += dataPtr[i + j] * kernelPtr[j + n];
        }

        dataPtr[outIndex] = sum;
    });
}

void convolve_par_manual_vec(NeonVector& data, const std::vector<float>& convolutionKernel, const int n) {
    const size_t dataSize = data.size();
    const size_t kernelSize = convolutionKernel.size();

    const size_t paddedKernelSize = (kernelSize + 3) & ~3;

    std::vector<float> paddedKernel(paddedKernelSize, 0.0f);
    for (size_t i = 0; i < kernelSize; ++i)
        paddedKernel[i] = convolutionKernel[i];

    const size_t k_blocks = paddedKernelSize / 4;
    std::vector<float32x4_t> k_vecs(k_blocks);

    for (size_t b = 0; b < k_blocks; ++b)
        k_vecs[b] = vld1q_f32(&paddedKernel[b * 4]);

    float* __restrict io_ptr =
        static_cast<float*>(ALIGN_HINT(data.data()));

    const size_t out_count = dataSize - kernelSize + 1;

    const size_t block16_count = out_count / 16;

    dispatch_apply(block16_count, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t blockIndex) {
        const size_t o = blockIndex * 16;

        __builtin_prefetch(io_ptr + o + 256, 0, 3);

        float32x4_t sum[16];
        for (int si = 0; si < 16; ++si)
            sum[si] = vdupq_n_f32(0.0f);

        const float* d_ptr = io_ptr + o;

        float32x4_t d0 = vld1q_f32(d_ptr);
        float32x4_t d1 = vld1q_f32(d_ptr + 4);
        float32x4_t d2 = vld1q_f32(d_ptr + 8);
        float32x4_t d3 = vld1q_f32(d_ptr + 12);
        float32x4_t d4;

        for (size_t b = 0; b < k_blocks; ++b) {
            float32x4_t k = k_vecs[b];

            d4 = vld1q_f32(d_ptr + (b + 4) * 4);

            sum[0]  = vfmaq_f32(sum[0],  d0, k);
            sum[1]  = vfmaq_f32(sum[1],  vextq_f32(d0, d1, 1), k);
            sum[2]  = vfmaq_f32(sum[2],  vextq_f32(d0, d1, 2), k);
            sum[3]  = vfmaq_f32(sum[3],  vextq_f32(d0, d1, 3), k);

            sum[4]  = vfmaq_f32(sum[4],  d1, k);
            sum[5]  = vfmaq_f32(sum[5],  vextq_f32(d1, d2, 1), k);
            sum[6]  = vfmaq_f32(sum[6],  vextq_f32(d1, d2, 2), k);
            sum[7]  = vfmaq_f32(sum[7],  vextq_f32(d1, d2, 3), k);

            sum[8]  = vfmaq_f32(sum[8],  d2, k);
            sum[9]  = vfmaq_f32(sum[9],  vextq_f32(d2, d3, 1), k);
            sum[10] = vfmaq_f32(sum[10], vextq_f32(d2, d3, 2), k);
            sum[11] = vfmaq_f32(sum[11], vextq_f32(d2, d3, 3), k);

            sum[12] = vfmaq_f32(sum[12], d3, k);
            sum[13] = vfmaq_f32(sum[13], vextq_f32(d3, d4, 1), k);
            sum[14] = vfmaq_f32(sum[14], vextq_f32(d3, d4, 2), k);
            sum[15] = vfmaq_f32(sum[15], vextq_f32(d3, d4, 3), k);

            d0 = d1;
            d1 = d2;
            d2 = d3;
            d3 = d4;
        }

        for (int si = 0; si < 16; ++si)
            io_ptr[o + si] = vaddvq_f32(sum[si]);
    });

    size_t o = block16_count * 16;
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

        io_ptr[o]     = vaddvq_f32(sum0);
        io_ptr[o + 1] = vaddvq_f32(sum1);
        io_ptr[o + 2] = vaddvq_f32(sum2);
        io_ptr[o + 3] = vaddvq_f32(sum3);
    }

    for (; o < out_count; ++o) {
        float sum = 0.0f;
        for (size_t k = 0; k < kernelSize; ++k)
            sum += io_ptr[o + k] * convolutionKernel[k];
        io_ptr[o] = sum;
    }
}
