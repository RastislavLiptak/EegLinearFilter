//
//  processors.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 20.11.2025.
//

#include "processors.h"
#include <iostream>

void run_processor(const ProcessingMode mode, std::vector<float>& allData, const std::vector<float>& convolutionKernel, const int convolutionKernelRadius) {
    // TODO - změř rychlost výpočtu
    switch (mode) {
        case ProcessingMode::CPU_SEQ_NO_VEC:
            std::cout << "Mode: Sequential processing on CPU (no-vectorization)" << std::endl;
            convolve_seq_no_vec(allData, convolutionKernel, convolutionKernelRadius);
            break;
        case ProcessingMode::CPU_SEQ_AUTO_VEC:
            std::cout << "Mode: Sequential processing on CPU (auto-vectorization)" << std::endl;
            convolve_seq_auto_vec(allData, convolutionKernel, convolutionKernelRadius);
            break;
        case ProcessingMode::CPU_SEQ_MANUAL_VEC:
            std::cout << "Mode: Sequential processing on CPU (manual-vectorization)" << std::endl;
            // TODO - process data
            break;
        case ProcessingMode::CPU_PAR_NO_VEC:
            std::cout << "Mode: Parallel processing on CPU (no-vectorization)" << std::endl;
            // TODO - process data
            break;
        case ProcessingMode::CPU_PAR_AUTO_VEC:
            std::cout << "Mode: Parallel processing on CPU (auto-vectorization)" << std::endl;
            // TODO - process data
            break;
        case ProcessingMode::CPU_PAR_MANUAL_VEC:
            std::cout << "Mode: Parallel processing on CPU (manual-vectorization)" << std::endl;
            // TODO - process data
            break;
        case ProcessingMode::GPU_PAR:
            std::cout << "Mode: GPU processing" << std::endl;
            // TODO - process data
            break;
        default:
            throw std::runtime_error("Unknown processing mode");
    }
}
