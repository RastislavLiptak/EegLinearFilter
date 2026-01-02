//
//  convolve_par.hpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 25.11.2025.
//  Parallel convolution implementations using Grand Central Dispatch (GCD).
//

#ifndef CONVOLVE_PAR
#define CONVOLVE_PAR

#include "../data_types.hpp"
#include <dispatch/dispatch.h>
#include <arm_neon.h>
#include <vector>

/**
 * Parallel naive implementation.
 *
 * @tparam Radius Kernel radius.
 * @tparam ChunkSize Number of elements per GCD task.
 */
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

/**
 * Parallel implementation with vectorization explicitly disabled.
 *
 * @tparam Radius Kernel radius.
 * @tparam ChunkSize Elements per task.
 * @tparam KBatch Unrolling factor for the inner kernel loop.
 */
template <int Radius, int ChunkSize, int KBatch>
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

        float* __restrict o_chunk = outputPtr + start;
        const float* __restrict d_chunk = dataPtr + start;

        size_t k = 0;
        for (; k + KBatch <= KernelSize; k += KBatch) {
            float k_vals[KBatch];
            for(int i=0; i<KBatch; ++i) k_vals[i] = kernelPtr[k+i];

            #pragma clang loop vectorize(disable)
            for (size_t out = 0; out < actualChunkSize; ++out) {
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;

                const float* __restrict current_d = d_chunk + out + k;

                for (int i = 0; i < KBatch; i += 4) {
                    acc0 += current_d[i + 0] * k_vals[i + 0];
                    acc1 += current_d[i + 1] * k_vals[i + 1];
                    acc2 += current_d[i + 2] * k_vals[i + 2];
                    acc3 += current_d[i + 3] * k_vals[i + 3];
                }

                o_chunk[out] += (acc0 + acc1 + acc2 + acc3);
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

/**
 * Parallel implementation using compiler auto-vectorization hints.
 *
 * @tparam Radius Kernel radius.
 * @tparam ChunkSize Elements per task.
 * @tparam KBatch Unrolling factor.
 */
template <int Radius, int ChunkSize, int KBatch>
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

        float* __restrict o_chunk = outputPtr + start;
        const float* __restrict d_chunk = dataPtr + start;

        size_t k = 0;
        for (; k + KBatch <= KernelSize; k += KBatch) {
            float k_vals[KBatch];
            for(int i=0; i<KBatch; ++i) k_vals[i] = kernelPtr[k+i];

            #pragma clang loop vectorize(enable) interleave_count(4)
            for (size_t out = 0; out < actualChunkSize; ++out) {
                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;

                const float* __restrict current_d = d_chunk + out + k;

                for (int i = 0; i < KBatch; i += 4) {
                    acc0 += current_d[i + 0] * k_vals[i + 0];
                    acc1 += current_d[i + 1] * k_vals[i + 1];
                    acc2 += current_d[i + 2] * k_vals[i + 2];
                    acc3 += current_d[i + 3] * k_vals[i + 3];
                }

                o_chunk[out] += (acc0 + acc1 + acc2 + acc3);
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

/**
 * Parallel implementation using manual ARM Neon Intrinsics.
 *
 * @tparam Radius Kernel radius.
 * @tparam ChunkSize Elements per task.
 * @tparam KBatch Unrolling factor.
 */
template <int Radius, int ChunkSize, int KBatch>
void convolve_par_manual_vec(const NeonVector& data, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel) {
    constexpr size_t KernelSize = 2 * Radius + 1;
    const size_t outSize = data.size() - KernelSize + 1;
    
    const float* __restrict dataPtr = data.data();
    float* __restrict outputPtr = outputBuffer.data();
    const float* __restrict kernelPtr = convolutionKernel.data();
    const size_t numChunks = (outSize + ChunkSize - 1) / ChunkSize;

    dispatch_apply(numChunks, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t chunkIndex) {
        const size_t start = chunkIndex * ChunkSize;
        const size_t actualChunkSize = std::min(static_cast<size_t>(ChunkSize), outSize - start);
        
        float* __restrict o_chunk = outputPtr + start;
        const float* __restrict d_chunk = dataPtr + start;
        
        size_t k = 0;

        for (; k + KBatch <= KernelSize; k += KBatch) {
            const float* k_ptr_base = kernelPtr + k;
            size_t out = 0;
        
            for (; out + 16 <= actualChunkSize; out += 16) {
                float32x4_t acc0_A = vdupq_n_f32(0.0f); float32x4_t acc1_A = vdupq_n_f32(0.0f);
                float32x4_t acc2_A = vdupq_n_f32(0.0f); float32x4_t acc3_A = vdupq_n_f32(0.0f);
                float32x4_t acc0_B = vdupq_n_f32(0.0f); float32x4_t acc1_B = vdupq_n_f32(0.0f);
                float32x4_t acc2_B = vdupq_n_f32(0.0f); float32x4_t acc3_B = vdupq_n_f32(0.0f);
                float32x4_t acc0_C = vdupq_n_f32(0.0f); float32x4_t acc1_C = vdupq_n_f32(0.0f);
                float32x4_t acc2_C = vdupq_n_f32(0.0f); float32x4_t acc3_C = vdupq_n_f32(0.0f);
                float32x4_t acc0_D = vdupq_n_f32(0.0f); float32x4_t acc1_D = vdupq_n_f32(0.0f);
                float32x4_t acc2_D = vdupq_n_f32(0.0f); float32x4_t acc3_D = vdupq_n_f32(0.0f);
            
                const float* current_d = d_chunk + out + k;
                
                for (int i = 0; i < KBatch; i += 4) {
                    float32x4_t k0 = vdupq_n_f32(k_ptr_base[i + 0]);
                    float32x4_t k1 = vdupq_n_f32(k_ptr_base[i + 1]);
                    float32x4_t k2 = vdupq_n_f32(k_ptr_base[i + 2]);
                    float32x4_t k3 = vdupq_n_f32(k_ptr_base[i + 3]);
                    
                    acc0_A = vfmaq_f32(acc0_A, vld1q_f32(current_d + i + 0), k0);
                    acc1_A = vfmaq_f32(acc1_A, vld1q_f32(current_d + i + 1), k1);
                    acc2_A = vfmaq_f32(acc2_A, vld1q_f32(current_d + i + 2), k2);
                    acc3_A = vfmaq_f32(acc3_A, vld1q_f32(current_d + i + 3), k3);
                    
                    acc0_B = vfmaq_f32(acc0_B, vld1q_f32(current_d + i + 4), k0);
                    acc1_B = vfmaq_f32(acc1_B, vld1q_f32(current_d + i + 5), k1);
                    acc2_B = vfmaq_f32(acc2_B, vld1q_f32(current_d + i + 6), k2);
                    acc3_B = vfmaq_f32(acc3_B, vld1q_f32(current_d + i + 7), k3);
                    
                    acc0_C = vfmaq_f32(acc0_C, vld1q_f32(current_d + i + 8), k0);
                    acc1_C = vfmaq_f32(acc1_C, vld1q_f32(current_d + i + 9), k1);
                    acc2_C = vfmaq_f32(acc2_C, vld1q_f32(current_d + i + 10), k2);
                    acc3_C = vfmaq_f32(acc3_C, vld1q_f32(current_d + i + 11), k3);
                    
                    acc0_D = vfmaq_f32(acc0_D, vld1q_f32(current_d + i + 12), k0);
                    acc1_D = vfmaq_f32(acc1_D, vld1q_f32(current_d + i + 13), k1);
                    acc2_D = vfmaq_f32(acc2_D, vld1q_f32(current_d + i + 14), k2);
                    acc3_D = vfmaq_f32(acc3_D, vld1q_f32(current_d + i + 15), k3);
                }
                
                float32x4_t sum_A = vaddq_f32(vaddq_f32(acc0_A, acc1_A), vaddq_f32(acc2_A, acc3_A));
                vst1q_f32(o_chunk + out, vaddq_f32(vld1q_f32(o_chunk + out), sum_A));
                
                float32x4_t sum_B = vaddq_f32(vaddq_f32(acc0_B, acc1_B), vaddq_f32(acc2_B, acc3_B));
                vst1q_f32(o_chunk + out + 4, vaddq_f32(vld1q_f32(o_chunk + out + 4), sum_B));
                
                float32x4_t sum_C = vaddq_f32(vaddq_f32(acc0_C, acc1_C), vaddq_f32(acc2_C, acc3_C));
                vst1q_f32(o_chunk + out + 8, vaddq_f32(vld1q_f32(o_chunk + out + 8), sum_C));
                
                float32x4_t sum_D = vaddq_f32(vaddq_f32(acc0_D, acc1_D), vaddq_f32(acc2_D, acc3_D));
                vst1q_f32(o_chunk + out + 12, vaddq_f32(vld1q_f32(o_chunk + out + 12), sum_D));
            }

            for (; out < actualChunkSize; ++out) {
                float acc = 0.0f;
                const float* current_d = d_chunk + out + k;
                for (int i = 0; i < KBatch; ++i) acc += current_d[i] * k_ptr_base[i];
                o_chunk[out] += acc;
            }
        }

        for (; k + 4 <= KernelSize; k += 4) {
            const float* k_ptr_base = kernelPtr + k;
            size_t out = 0;

            for (; out + 16 <= actualChunkSize; out += 16) {
                float32x4_t k0 = vdupq_n_f32(k_ptr_base[0]);
                float32x4_t k1 = vdupq_n_f32(k_ptr_base[1]);
                float32x4_t k2 = vdupq_n_f32(k_ptr_base[2]);
                float32x4_t k3 = vdupq_n_f32(k_ptr_base[3]);
                
                const float* current_d = d_chunk + out + k;
                
                float32x4_t d_A0 = vld1q_f32(current_d + 0); float32x4_t d_A1 = vld1q_f32(current_d + 1);
                float32x4_t d_A2 = vld1q_f32(current_d + 2); float32x4_t d_A3 = vld1q_f32(current_d + 3);
                float32x4_t acc_A = vfmaq_f32(vmulq_f32(d_A0, k0), d_A1, k1);
                acc_A = vfmaq_f32(acc_A, d_A2, k2); acc_A = vfmaq_f32(acc_A, d_A3, k3);

                float32x4_t d_B0 = vld1q_f32(current_d + 4); float32x4_t d_B1 = vld1q_f32(current_d + 5);
                float32x4_t d_B2 = vld1q_f32(current_d + 6); float32x4_t d_B3 = vld1q_f32(current_d + 7);
                float32x4_t acc_B = vfmaq_f32(vmulq_f32(d_B0, k0), d_B1, k1);
                acc_B = vfmaq_f32(acc_B, d_B2, k2); acc_B = vfmaq_f32(acc_B, d_B3, k3);

                float32x4_t d_C0 = vld1q_f32(current_d + 8); float32x4_t d_C1 = vld1q_f32(current_d + 9);
                float32x4_t d_C2 = vld1q_f32(current_d + 10); float32x4_t d_C3 = vld1q_f32(current_d + 11);
                float32x4_t acc_C = vfmaq_f32(vmulq_f32(d_C0, k0), d_C1, k1);
                acc_C = vfmaq_f32(acc_C, d_C2, k2); acc_C = vfmaq_f32(acc_C, d_C3, k3);

                float32x4_t d_D0 = vld1q_f32(current_d + 12); float32x4_t d_D1 = vld1q_f32(current_d + 13);
                float32x4_t d_D2 = vld1q_f32(current_d + 14); float32x4_t d_D3 = vld1q_f32(current_d + 15);
                float32x4_t acc_D = vfmaq_f32(vmulq_f32(d_D0, k0), d_D1, k1);
                acc_D = vfmaq_f32(acc_D, d_D2, k2); acc_D = vfmaq_f32(acc_D, d_D3, k3);

                vst1q_f32(o_chunk + out, vaddq_f32(vld1q_f32(o_chunk + out), acc_A));
                vst1q_f32(o_chunk + out + 4, vaddq_f32(vld1q_f32(o_chunk + out + 4), acc_B));
                vst1q_f32(o_chunk + out + 8, vaddq_f32(vld1q_f32(o_chunk + out + 8), acc_C));
                vst1q_f32(o_chunk + out + 12, vaddq_f32(vld1q_f32(o_chunk + out + 12), acc_D));
            }

            for (; out < actualChunkSize; ++out) {
                float acc = 0.0f;
                const float* current_d = d_chunk + out + k;
                for (int i = 0; i < 4; ++i) acc += current_d[i] * k_ptr_base[i];
                o_chunk[out] += acc;
            }
        }

        for (; k < KernelSize; ++k) {
            float kv_scalar = kernelPtr[k];
            float32x4_t k_vec = vdupq_n_f32(kv_scalar);
            size_t out = 0;
            for (; out + 4 <= actualChunkSize; out += 4) {
                vst1q_f32(o_chunk + out, vfmaq_f32(vld1q_f32(o_chunk + out), vld1q_f32(d_chunk + out + k), k_vec));
            }
            for (; out < actualChunkSize; ++out) {
                o_chunk[out] += d_chunk[out + k] * kv_scalar;
            }
        }
    });
}

#endif // CONVOLVE_PAR
