//
//  convolve_par.hpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 25.11.2025.
//

#ifndef CONVOLVE_PAR
#define CONVOLVE_PAR

#include "../data_types.hpp"
#include <dispatch/dispatch.h>
#include <arm_neon.h>
#include <vector>

#define ALIGN_HINT(ptr) __builtin_assume_aligned((ptr), 16)

template <int Radius>
void convolve_par_no_vec(const NeonVector& data, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel) {
    constexpr size_t KernelSize = 2 * Radius + 1;
    const size_t dataSize = data.size();
    const size_t outSize = dataSize - KernelSize + 1;

    const float* __restrict dataPtr = data.data();
    float* __restrict outputPtr = outputBuffer.data();
    const float* __restrict kernelPtr = convolutionKernel.data();

    const size_t chunkSize = 4096;
    const size_t numChunks = (outSize + chunkSize - 1) / chunkSize;

    dispatch_apply(numChunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t chunkIndex) {
        const size_t start = chunkIndex * chunkSize;
        const size_t actualChunkSize = std::min(chunkSize, outSize - start);

        float* o_chunk = outputPtr + start;
        const float* d_chunk = dataPtr + start;

        size_t k = 0;

        for (; k + 4 <= KernelSize; k += 4) {
            const float k0 = kernelPtr[k];
            const float k1 = kernelPtr[k+1];
            const float k2 = kernelPtr[k+2];
            const float k3 = kernelPtr[k+3];

            #pragma clang loop vectorize(disable)
            for (size_t out = 0; out < actualChunkSize; ++out) {
                float sum = o_chunk[out];

                sum += d_chunk[out + k] * k0;
                sum += d_chunk[out + k + 1] * k1;
                sum += d_chunk[out + k + 2] * k2;
                sum += d_chunk[out + k + 3] * k3;

                o_chunk[out] = sum;
            }
        }

        for (; k < KernelSize; ++k) {
            const float kv = kernelPtr[k];
            
            #pragma clang loop vectorize(disable)
            for (size_t out = 0; out < actualChunkSize; ++out) {
                o_chunk[out] += d_chunk[out + k] * kv;
            }
        }
    });
}

template <int Radius>
void convolve_par_auto_vec(const NeonVector& data, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel) {
    constexpr size_t KernelSize = 2 * Radius + 1;
    const size_t dataSize = data.size();
    const size_t outSize = dataSize - KernelSize + 1;

    const float* __restrict dataPtr = data.data();
    float* __restrict outputPtr = outputBuffer.data();
    const float* __restrict kernelPtr = convolutionKernel.data();

    const size_t chunkSize = 4096;
    const size_t numChunks = (outSize + chunkSize - 1) / chunkSize;

    dispatch_apply(numChunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t chunkIndex) {
        const size_t start = chunkIndex * chunkSize;
        const size_t actualChunkSize = std::min(chunkSize, outSize - start);

        float* o_chunk = outputPtr + start;
        const float* d_chunk = dataPtr + start;

        size_t k = 0;
        for (; k + 4 <= KernelSize; k += 4) {
            const float k0 = kernelPtr[k];
            const float k1 = kernelPtr[k+1];
            const float k2 = kernelPtr[k+2];
            const float k3 = kernelPtr[k+3];

            #pragma clang loop vectorize(enable) interleave(enable)
            for (size_t out = 0; out < actualChunkSize; ++out) {
                float sum = o_chunk[out];
                
                sum += d_chunk[out + k] * k0;
                sum += d_chunk[out + k + 1] * k1;
                sum += d_chunk[out + k + 2] * k2;
                sum += d_chunk[out + k + 3] * k3;
                
                o_chunk[out] = sum;
            }
        }

        for (; k < KernelSize; ++k) {
            const float kv = kernelPtr[k];
            #pragma clang loop vectorize(enable)
            for (size_t out = 0; out < actualChunkSize; ++out) {
                o_chunk[out] += d_chunk[out + k] * kv;
            }
        }
    });
}

template <int Radius>
void convolve_par_manual_vec(const NeonVector& data, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel) {
    constexpr size_t KernelSize = 2 * Radius + 1;
    const size_t dataSize = data.size();
    const size_t outSize = dataSize - KernelSize + 1;

    const float* __restrict dataPtr = static_cast<float*>(ALIGN_HINT(data.data()));
    float* __restrict outputPtr = static_cast<float*>(ALIGN_HINT(outputBuffer.data()));
    const float* __restrict kernelPtr = convolutionKernel.data();

    const size_t chunkSize = 4096;
    const size_t numChunks = (outSize + chunkSize - 1) / chunkSize;

    dispatch_apply(numChunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t chunkIndex) {
        const size_t start = chunkIndex * chunkSize;
        const size_t actualChunkSize = std::min(chunkSize, outSize - start);
        
        float* o_chunk = outputPtr + start;
        const float* d_chunk = dataPtr + start;

        size_t k = 0;
        for (; k + 4 <= KernelSize; k += 4) {
            float32x4_t k0 = vdupq_n_f32(kernelPtr[k]);
            float32x4_t k1 = vdupq_n_f32(kernelPtr[k + 1]);
            float32x4_t k2 = vdupq_n_f32(kernelPtr[k + 2]);
            float32x4_t k3 = vdupq_n_f32(kernelPtr[k + 3]);

            size_t i = 0;
            
            for (; i + 8 <= actualChunkSize; i += 8) {
                float32x4_t acc0 = vld1q_f32(o_chunk + i);
                float32x4_t acc1 = vld1q_f32(o_chunk + i + 4);

                float32x4_t d0_0 = vld1q_f32(d_chunk + i + k);
                float32x4_t d1_0 = vld1q_f32(d_chunk + i + k + 4);
                acc0 = vfmaq_f32(acc0, d0_0, k0);
                acc1 = vfmaq_f32(acc1, d1_0, k0);

                float32x4_t d0_1 = vld1q_f32(d_chunk + i + k + 1);
                float32x4_t d1_1 = vld1q_f32(d_chunk + i + k + 5);
                acc0 = vfmaq_f32(acc0, d0_1, k1);
                acc1 = vfmaq_f32(acc1, d1_1, k1);

                float32x4_t d0_2 = vld1q_f32(d_chunk + i + k + 2);
                float32x4_t d1_2 = vld1q_f32(d_chunk + i + k + 6);
                acc0 = vfmaq_f32(acc0, d0_2, k2);
                acc1 = vfmaq_f32(acc1, d1_2, k2);

                float32x4_t d0_3 = vld1q_f32(d_chunk + i + k + 3);
                float32x4_t d1_3 = vld1q_f32(d_chunk + i + k + 7);
                acc0 = vfmaq_f32(acc0, d0_3, k3);
                acc1 = vfmaq_f32(acc1, d1_3, k3);

                vst1q_f32(o_chunk + i, acc0);
                vst1q_f32(o_chunk + i + 4, acc1);
            }

            for (; i + 4 <= actualChunkSize; i += 4) {
                float32x4_t acc = vld1q_f32(o_chunk + i);
                
                acc = vfmaq_f32(acc, vld1q_f32(d_chunk + i + k),     k0);
                acc = vfmaq_f32(acc, vld1q_f32(d_chunk + i + k + 1), k1);
                acc = vfmaq_f32(acc, vld1q_f32(d_chunk + i + k + 2), k2);
                acc = vfmaq_f32(acc, vld1q_f32(d_chunk + i + k + 3), k3);
                
                vst1q_f32(o_chunk + i, acc);
            }

            for (; i < actualChunkSize; ++i) {
                o_chunk[i] += d_chunk[i + k]     * kernelPtr[k];
                o_chunk[i] += d_chunk[i + k + 1] * kernelPtr[k + 1];
                o_chunk[i] += d_chunk[i + k + 2] * kernelPtr[k + 2];
                o_chunk[i] += d_chunk[i + k + 3] * kernelPtr[k + 3];
            }
        }

        for (; k < KernelSize; ++k) {
            float kv_scalar = kernelPtr[k];
            float32x4_t k_vec = vdupq_n_f32(kv_scalar);
            
            size_t i = 0;
            for (; i + 4 <= actualChunkSize; i += 4) {
                float32x4_t o = vld1q_f32(o_chunk + i);
                float32x4_t d = vld1q_f32(d_chunk + i + k);
                o = vfmaq_f32(o, d, k_vec);
                vst1q_f32(o_chunk + i, o);
            }
            for (; i < actualChunkSize; ++i) {
                o_chunk[i] += d_chunk[i + k] * kv_scalar;
            }
        }
    });
}

#endif // CONVOLVE_PAR
