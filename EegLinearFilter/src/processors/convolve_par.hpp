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

template <int Radius, int ChunkSize>
void convolve_par_naive(const NeonVector& data, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel) {
    const size_t dataSize = data.size();
    const size_t start = static_cast<size_t>(Radius);
    const size_t end = dataSize - static_cast<size_t>(Radius);
    const size_t totalWork = end - start;

    const size_t numChunks = (totalWork + ChunkSize - 1) / ChunkSize;

    dispatch_apply(numChunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t chunkIdx) {
        size_t chunkStart = start + chunkIdx * ChunkSize;
        size_t chunkEnd = std::min(chunkStart + ChunkSize, end);

        for (size_t i = chunkStart; i < chunkEnd; ++i) {
            float sum = 0.0f;
            for (int j = -Radius; j <= Radius; ++j) {
                sum += data[i + j] * convolutionKernel[j + Radius];
            }
            outputBuffer[i - Radius] = sum;
        }
    });
}

template <int Radius, int ChunkSize>
void convolve_par_no_vec(const NeonVector& data, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel) {
    constexpr size_t KernelSize = 2 * Radius + 1;
    const size_t dataSize = data.size();
    const size_t outSize = dataSize - KernelSize + 1;

    const float* __restrict dataPtr = data.data();
    float* __restrict outputPtr = outputBuffer.data();
    const float* __restrict kernelPtr = convolutionKernel.data();

    const size_t numChunks = (outSize + ChunkSize - 1) / ChunkSize;
    dispatch_apply(numChunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t chunkIndex) {
        const size_t start = chunkIndex * ChunkSize;
        const size_t actualChunkSize = std::min(static_cast<size_t>(ChunkSize), outSize - start);

        float* o_chunk = outputPtr + start;
        const float* d_chunk = dataPtr + start;

        size_t k = 0;
        for (; k + 8 <= KernelSize; k += 8) {
            const float k0 = kernelPtr[k];
            const float k1 = kernelPtr[k+1];
            const float k2 = kernelPtr[k+2];
            const float k3 = kernelPtr[k+3];
            const float k4 = kernelPtr[k+4];
            const float k5 = kernelPtr[k+5];
            const float k6 = kernelPtr[k+6];
            const float k7 = kernelPtr[k+7];

            #pragma clang loop vectorize(disable)
            for (size_t out = 0; out < actualChunkSize; ++out) {
                float sum = o_chunk[out];
                
                sum += d_chunk[out + k + 0] * k0;
                sum += d_chunk[out + k + 1] * k1;
                sum += d_chunk[out + k + 2] * k2;
                sum += d_chunk[out + k + 3] * k3;
                sum += d_chunk[out + k + 4] * k4;
                sum += d_chunk[out + k + 5] * k5;
                sum += d_chunk[out + k + 6] * k6;
                sum += d_chunk[out + k + 7] * k7;
                
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

template <int Radius, int ChunkSize>
void convolve_par_auto_vec(const NeonVector& data, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel) {
    constexpr size_t KernelSize = 2 * Radius + 1;
    const size_t dataSize = data.size();
    const size_t outSize = dataSize - KernelSize + 1;

    const float* __restrict dataPtr = data.data();
    float* __restrict outputPtr = outputBuffer.data();
    const float* __restrict kernelPtr = convolutionKernel.data();

    const size_t numChunks = (outSize + ChunkSize - 1) / ChunkSize;
    dispatch_apply(numChunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t chunkIndex) {
        const size_t start = chunkIndex * ChunkSize;
        const size_t actualChunkSize = std::min(static_cast<size_t>(ChunkSize), outSize - start);

        float* o_chunk = outputPtr + start;
        const float* d_chunk = dataPtr + start;

        size_t k = 0;
        for (; k + 8 <= KernelSize; k += 8) {
            const float k0 = kernelPtr[k];
            const float k1 = kernelPtr[k+1];
            const float k2 = kernelPtr[k+2];
            const float k3 = kernelPtr[k+3];
            const float k4 = kernelPtr[k+4];
            const float k5 = kernelPtr[k+5];
            const float k6 = kernelPtr[k+6];
            const float k7 = kernelPtr[k+7];

            #pragma clang loop vectorize(enable) interleave_count(4)
            for (size_t out = 0; out < actualChunkSize; ++out) {
                float sum = o_chunk[out];
                
                sum += d_chunk[out + k + 0] * k0;
                sum += d_chunk[out + k + 1] * k1;
                sum += d_chunk[out + k + 2] * k2;
                sum += d_chunk[out + k + 3] * k3;
                sum += d_chunk[out + k + 4] * k4;
                sum += d_chunk[out + k + 5] * k5;
                sum += d_chunk[out + k + 6] * k6;
                sum += d_chunk[out + k + 7] * k7;
                
                o_chunk[out] = sum;
            }
        }

        for (; k < KernelSize; ++k) {
            const float kv = kernelPtr[k];
            #pragma clang loop vectorize(enable) interleave_count(4)
            for (size_t out = 0; out < actualChunkSize; ++out) {
                o_chunk[out] += d_chunk[out + k] * kv;
            }
        }
    });
}

template <int Radius, int ChunkSize>
void convolve_par_manual_vec(const NeonVector& data, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel) {
    constexpr size_t KernelSize = 2 * Radius + 1;
    const size_t outSize = data.size() - KernelSize + 1;

    const float* __restrict dataPtr = static_cast<const float*>(ALIGN_HINT(data.data()));
    float* __restrict outputPtr = static_cast<float*>(ALIGN_HINT(outputBuffer.data()));
    const float* __restrict kernelPtr = convolutionKernel.data();

    const size_t numChunks = (outSize + ChunkSize - 1) / ChunkSize;
    dispatch_apply(numChunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t chunkIndex) {
        const size_t start = chunkIndex * ChunkSize;
        const size_t actualChunkSize = std::min(static_cast<size_t>(ChunkSize), outSize - start);
        
        float* o_chunk = outputPtr + start;
        const float* d_chunk = dataPtr + start;

        size_t k = 0;
        for (; k + 8 <= KernelSize; k += 8) {
            float32x4_t k0 = vdupq_n_f32(kernelPtr[k]);
            float32x4_t k1 = vdupq_n_f32(kernelPtr[k + 1]);
            float32x4_t k2 = vdupq_n_f32(kernelPtr[k + 2]);
            float32x4_t k3 = vdupq_n_f32(kernelPtr[k + 3]);
            float32x4_t k4 = vdupq_n_f32(kernelPtr[k + 4]);
            float32x4_t k5 = vdupq_n_f32(kernelPtr[k + 5]);
            float32x4_t k6 = vdupq_n_f32(kernelPtr[k + 6]);
            float32x4_t k7 = vdupq_n_f32(kernelPtr[k + 7]);

            size_t i = 0;
            
            for (; i + 16 <= actualChunkSize; i += 16) {
                
                __builtin_prefetch(d_chunk + i + k + PREFETCH_LOOKAHEAD, 0, 1);
                
                float32x4_t acc0 = vld1q_f32(o_chunk + i);
                float32x4_t acc1 = vld1q_f32(o_chunk + i + 4);
                float32x4_t acc2 = vld1q_f32(o_chunk + i + 8);
                float32x4_t acc3 = vld1q_f32(o_chunk + i + 12);
                
                const float* d = d_chunk + i + k;

                acc0 = vfmaq_f32(acc0, vld1q_f32(d + 0),  k0);
                acc1 = vfmaq_f32(acc1, vld1q_f32(d + 4),  k0);
                acc2 = vfmaq_f32(acc2, vld1q_f32(d + 8),  k0);
                acc3 = vfmaq_f32(acc3, vld1q_f32(d + 12), k0);

                acc0 = vfmaq_f32(acc0, vld1q_f32(d + 1),  k1);
                acc1 = vfmaq_f32(acc1, vld1q_f32(d + 5),  k1);
                acc2 = vfmaq_f32(acc2, vld1q_f32(d + 9),  k1);
                acc3 = vfmaq_f32(acc3, vld1q_f32(d + 13), k1);

                acc0 = vfmaq_f32(acc0, vld1q_f32(d + 2),  k2);
                acc1 = vfmaq_f32(acc1, vld1q_f32(d + 6),  k2);
                acc2 = vfmaq_f32(acc2, vld1q_f32(d + 10), k2);
                acc3 = vfmaq_f32(acc3, vld1q_f32(d + 14), k2);

                acc0 = vfmaq_f32(acc0, vld1q_f32(d + 3),  k3);
                acc1 = vfmaq_f32(acc1, vld1q_f32(d + 7),  k3);
                acc2 = vfmaq_f32(acc2, vld1q_f32(d + 11), k3);
                acc3 = vfmaq_f32(acc3, vld1q_f32(d + 15), k3);

                acc0 = vfmaq_f32(acc0, vld1q_f32(d + 4),  k4);
                acc1 = vfmaq_f32(acc1, vld1q_f32(d + 8),  k4);
                acc2 = vfmaq_f32(acc2, vld1q_f32(d + 12), k4);
                acc3 = vfmaq_f32(acc3, vld1q_f32(d + 16), k4);

                acc0 = vfmaq_f32(acc0, vld1q_f32(d + 5),  k5);
                acc1 = vfmaq_f32(acc1, vld1q_f32(d + 9),  k5);
                acc2 = vfmaq_f32(acc2, vld1q_f32(d + 13), k5);
                acc3 = vfmaq_f32(acc3, vld1q_f32(d + 17), k5);

                acc0 = vfmaq_f32(acc0, vld1q_f32(d + 6),  k6);
                acc1 = vfmaq_f32(acc1, vld1q_f32(d + 10), k6);
                acc2 = vfmaq_f32(acc2, vld1q_f32(d + 14), k6);
                acc3 = vfmaq_f32(acc3, vld1q_f32(d + 18), k6);

                acc0 = vfmaq_f32(acc0, vld1q_f32(d + 7),  k7);
                acc1 = vfmaq_f32(acc1, vld1q_f32(d + 11), k7);
                acc2 = vfmaq_f32(acc2, vld1q_f32(d + 15), k7);
                acc3 = vfmaq_f32(acc3, vld1q_f32(d + 19), k7);

                vst1q_f32(o_chunk + i,     acc0);
                vst1q_f32(o_chunk + i + 4, acc1);
                vst1q_f32(o_chunk + i + 8, acc2);
                vst1q_f32(o_chunk + i + 12, acc3);
            }

            for (; i + 4 <= actualChunkSize; i += 4) {
                float32x4_t acc = vld1q_f32(o_chunk + i);
                const float* d = d_chunk + i + k;
                acc = vfmaq_f32(acc, vld1q_f32(d + 0), k0);
                acc = vfmaq_f32(acc, vld1q_f32(d + 1), k1);
                acc = vfmaq_f32(acc, vld1q_f32(d + 2), k2);
                acc = vfmaq_f32(acc, vld1q_f32(d + 3), k3);
                acc = vfmaq_f32(acc, vld1q_f32(d + 4), k4);
                acc = vfmaq_f32(acc, vld1q_f32(d + 5), k5);
                acc = vfmaq_f32(acc, vld1q_f32(d + 6), k6);
                acc = vfmaq_f32(acc, vld1q_f32(d + 7), k7);
                vst1q_f32(o_chunk + i, acc);
            }
            
            #pragma clang loop vectorize(disable)
            for (; i < actualChunkSize; ++i) {
                for (size_t kk = 0; kk < 8; ++kk) {
                    o_chunk[i] += d_chunk[i + k + kk] * kernelPtr[k + kk];
                }
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

            #pragma clang loop vectorize(disable)
            for (; i < actualChunkSize; ++i) {
                o_chunk[i] += d_chunk[i + k] * kv_scalar;
            }
        }
    });
}

#endif // CONVOLVE_PAR
