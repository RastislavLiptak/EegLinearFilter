//
//  processors.hpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 19.11.2025.
//

#ifndef PROCESSORS_HPP
#define PROCESSORS_HPP


#include "convolve_par.hpp"
#include "convolve_seq.hpp"
#include <iostream>
#include <chrono>

enum class ProcessingMode {
    CPU_SEQ_APPLE,           // Sequential benchmark implementation using Apple vDSP_conv method
    CPU_SEQ_NAIVE,           // Sequential naive approach without optimization
    CPU_SEQ_NO_VEC,          // Sequential processing, no vectorization
    CPU_SEQ_AUTO_VEC,        // Sequential, auto-vectorization
    CPU_SEQ_MANUAL_VEC,      // Sequential, manual vectorization
    CPU_PAR_NAIVE,           // Parallel naive approach without optimization
    CPU_PAR_NO_VEC,          // Parallel, no vectorization
    CPU_PAR_AUTO_VEC,        // Parallel, auto-vectorization
    CPU_PAR_MANUAL_VEC,      // Parallel, manual vectorization
    GPU_PAR,                 // GPU-accelerated
    
    COUNT
};

template <int Radius>
void calc_benchmarks(const std::chrono::duration<double> elapsed, size_t dataSize) {
    constexpr size_t KernelSize = 2 * Radius + 1;
    const size_t outputElements = dataSize - KernelSize + 1;

    double megaSamplesPerSec = (outputElements / elapsed.count()) / 1e6;

    double totalOperations = (double)outputElements * (double)KernelSize * 2.0;
    double gigaFlops = (totalOperations / elapsed.count()) / 1e9;

    std::cout << "Computation time: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Throughput: " << megaSamplesPerSec << " MSamples/s" << std::endl;
    std::cout << "Computational: " << gigaFlops << " GFLOPS" << std::endl;
    std::cout << "----------------------------------------\n";
}

template <int Radius, int ChunkSize>
void run_processor(const ProcessingMode mode, NeonVector& allData, const std::vector<float>& convolutionKernel) {
    
    NeonVector outputBuffer(allData.size(), 0.0f);
    const auto start = std::chrono::high_resolution_clock::now();
    
    switch (mode) {
        case ProcessingMode::CPU_SEQ_APPLE:
            std::cout << "Mode: Sequential processing on CPU using Apple implementation" << std::endl;
            convolve_seq_apple<Radius>(allData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_SEQ_NAIVE:
            std::cout << "Mode: Sequential processing on CPU using naive implementation" << std::endl;
            convolve_seq_naive<Radius>(allData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_SEQ_NO_VEC:
            std::cout << "Mode: Sequential processing on CPU (no-vectorization)" << std::endl;
            convolve_seq_no_vec<Radius, ChunkSize>(allData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_SEQ_AUTO_VEC:
            std::cout << "Mode: Sequential processing on CPU (auto-vectorization)" << std::endl;
            convolve_seq_auto_vec<Radius, ChunkSize>(allData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_SEQ_MANUAL_VEC:
            std::cout << "Mode: Sequential processing on CPU (manual-vectorization)" << std::endl;
            convolve_seq_manual_vec<Radius, ChunkSize>(allData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_PAR_NAIVE:
            std::cout << "Mode: Parallel processing on CPU using naive implementation" << std::endl;
            convolve_par_naive<Radius, ChunkSize>(allData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_PAR_NO_VEC:
            std::cout << "Mode: Parallel processing on CPU (no-vectorization)" << std::endl;
            convolve_par_no_vec<Radius, ChunkSize>(allData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_PAR_AUTO_VEC:
            std::cout << "Mode: Parallel processing on CPU (auto-vectorization)" << std::endl;
            convolve_par_auto_vec<Radius, ChunkSize>(allData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_PAR_MANUAL_VEC:
            std::cout << "Mode: Parallel processing on CPU (manual-vectorization)" << std::endl;
            convolve_par_manual_vec<Radius, ChunkSize>(allData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::GPU_PAR:
            std::cout << "Mode: GPU processing" << std::endl;
            // TODO - process data
            break;
        default:
            throw std::runtime_error("Unknown processing mode");
    }
    
    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed = end - start;
    
    allData.swap(outputBuffer);
    
    calc_benchmarks<Radius>(elapsed, allData.size());
}

#endif // PROCESSORS_HPP
