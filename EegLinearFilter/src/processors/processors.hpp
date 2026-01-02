//
//  processors.hpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 19.11.2025.
//  Core dispatch logic for selecting and executing specific processing strategies.
//

#ifndef PROCESSORS_HPP
#define PROCESSORS_HPP

#include "../config.h"
#include "convolve_par.hpp"
#include "convolve_seq.hpp"
#include "convolve_gpu/convolve_gpu.hpp"
#include <chrono>

/**
 * Executes a convolution processor based on the selected mode.
 * Measures time taken for memory initialization and computation.
 *
 * @tparam Radius Kernel radius.
 * @tparam ChunkSize Size of data chunks for processing.
 * @tparam KBatch Unrolling batch size.
 * @param mode Enum indicating which processor implementation to run (CPU/GPU, Seq/Par).
 * @param inputData The raw input signal.
 * @param outputBuffer The buffer to store processed results.
 * @param convolutionKernel The filter kernel.
 * @return ProcessingStats structure containing timing metrics.
 */
template <int Radius, int ChunkSize, int KBatch>
ProcessingStats run_processor(const ProcessingMode mode, const NeonVector& inputData, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel) {
    auto mem_start = std::chrono::high_resolution_clock::now();
    std::fill(outputBuffer.begin(), outputBuffer.end(), 0.0f);
    auto mem_end = std::chrono::high_resolution_clock::now();
    double memoryTime = std::chrono::duration<double>(mem_end - mem_start).count();

    const auto start = std::chrono::high_resolution_clock::now();
    
    ProcessingStats gpuStats = {0.0, 0.0, 0.0, 0.0, 0.0};
    bool isGpu = false;

    switch (mode) {
        case ProcessingMode::CPU_SEQ_APPLE:
            convolve_seq_apple<Radius>(inputData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_SEQ_NAIVE:
            convolve_seq_naive<Radius>(inputData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_SEQ_NO_VEC:
            convolve_seq_no_vec<Radius, ChunkSize, KBatch>(inputData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_SEQ_AUTO_VEC:
            convolve_seq_auto_vec<Radius, ChunkSize, KBatch>(inputData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_SEQ_MANUAL_VEC:
            convolve_seq_manual_vec<Radius, ChunkSize, KBatch>(inputData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_PAR_NAIVE:
            convolve_par_naive<Radius, ChunkSize>(inputData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_PAR_NO_VEC:
            convolve_par_no_vec<Radius, ChunkSize, KBatch>(inputData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_PAR_AUTO_VEC:
            convolve_par_auto_vec<Radius, ChunkSize, KBatch>(inputData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_PAR_MANUAL_VEC:
            convolve_par_manual_vec<Radius, ChunkSize, KBatch>(inputData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::GPU_NAIVE:
            gpuStats = convolve_gpu_naive<Radius>(inputData, outputBuffer, convolutionKernel);
            isGpu = true;
            break;
        case ProcessingMode::GPU_32BIT:
            gpuStats = convolve_gpu<Radius>(inputData, outputBuffer, convolutionKernel, false);
            isGpu = true;
            break;
        default:
            throw std::runtime_error("Unknown processing mode");
    }
    
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    
    if (isGpu) {
        gpuStats.cpuMemoryOpsSec += memoryTime;
        gpuStats.totalTimeSec += memoryTime;
        return gpuStats;
    } else {
        double compute = elapsed.count();
        double total = compute + memoryTime;
        return { total, compute, 0.0, memoryTime, 0.0 };
    }
}

#endif // PROCESSORS_HPP
