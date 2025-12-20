//
//  processors.hpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 19.11.2025.
//

#ifndef PROCESSORS_HPP
#define PROCESSORS_HPP

#include "../config.h"
#include "convolve_par.hpp"
#include "convolve_seq.hpp"
#include "convolve_gpu/convolve_gpu.hpp"
#include <chrono>

template <int Radius, int ChunkSize>
ProcessingStats run_processor(const ProcessingMode mode, NeonVector& allData, const std::vector<float>& convolutionKernel) {
    
    NeonVector outputBuffer(allData.size(), 0.0f);
    
    const auto start = std::chrono::high_resolution_clock::now();
    
    ProcessingStats gpuStats = {0.0, 0.0, 0.0};
    bool isGpu = false;

    switch (mode) {
        case ProcessingMode::CPU_SEQ_APPLE:
            convolve_seq_apple<Radius>(allData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_SEQ_NAIVE:
            convolve_seq_naive<Radius>(allData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_SEQ_NO_VEC:
            convolve_seq_no_vec<Radius, ChunkSize>(allData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_SEQ_AUTO_VEC:
            convolve_seq_auto_vec<Radius, ChunkSize>(allData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_SEQ_MANUAL_VEC:
            convolve_seq_manual_vec<Radius, ChunkSize>(allData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_PAR_NAIVE:
            convolve_par_naive<Radius, ChunkSize>(allData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_PAR_NO_VEC:
            convolve_par_no_vec<Radius, ChunkSize>(allData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_PAR_AUTO_VEC:
            convolve_par_auto_vec<Radius, ChunkSize>(allData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_PAR_MANUAL_VEC:
            convolve_par_manual_vec<Radius, ChunkSize>(allData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::GPU:
            gpuStats = convolve_gpu<Radius>(allData, outputBuffer, convolutionKernel);
            isGpu = true;
            break;
        default:
            throw std::runtime_error("Unknown processing mode");
    }
    
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    
    allData.swap(outputBuffer);
    outputBuffer.clear();
    outputBuffer.shrink_to_fit();
    
    if (isGpu) {
        return gpuStats;
    } else {
        double total = elapsed.count();
        return { total, total, 0.0 };
    }
}

#endif // PROCESSORS_HPP
