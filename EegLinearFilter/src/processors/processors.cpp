//
//  processors.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 20.11.2025.
//

#include "processors.h"
#include <iostream>
#include <chrono>

//TODO - velikost jádra je známá už při kompilaci, na to by bylo super, aby byl program ready

void run_processor(const ProcessingMode mode, NeonVector& allData, const std::vector<float>& convolutionKernel, const int convolutionKernelRadius) {
    
    NeonVector outputBuffer(allData.size());
    const auto start = std::chrono::high_resolution_clock::now();
    
    switch (mode) {
        case ProcessingMode::CPU_SEQ_NO_VEC:
            std::cout << "Mode: Sequential processing on CPU (no-vectorization)" << std::endl;
            convolve_seq_no_vec(allData, outputBuffer, convolutionKernel, convolutionKernelRadius);
            break;
        case ProcessingMode::CPU_SEQ_AUTO_VEC:
            std::cout << "Mode: Sequential processing on CPU (auto-vectorization)" << std::endl;
            convolve_seq_auto_vec(allData, outputBuffer, convolutionKernel, convolutionKernelRadius);
            break;
        case ProcessingMode::CPU_SEQ_MANUAL_VEC:
            std::cout << "Mode: Sequential processing on CPU (manual-vectorization)" << std::endl;
            convolve_seq_manual_vec(allData, outputBuffer, convolutionKernel);
            break;
        case ProcessingMode::CPU_PAR_NO_VEC:
            std::cout << "Mode: Parallel processing on CPU (no-vectorization)" << std::endl;
            convolve_par_no_vec(allData, outputBuffer, convolutionKernel, convolutionKernelRadius);
            break;
        case ProcessingMode::CPU_PAR_AUTO_VEC:
            std::cout << "Mode: Parallel processing on CPU (auto-vectorization)" << std::endl;
            convolve_par_auto_vec(allData, outputBuffer, convolutionKernel, convolutionKernelRadius);
            break;
        case ProcessingMode::CPU_PAR_MANUAL_VEC:
            std::cout << "Mode: Parallel processing on CPU (manual-vectorization)" << std::endl;
            convolve_par_manual_vec(allData, outputBuffer, convolutionKernel);
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
    std::cout << "Computation time: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "----------------------------------------\n";
    
    allData.swap(outputBuffer);
}
