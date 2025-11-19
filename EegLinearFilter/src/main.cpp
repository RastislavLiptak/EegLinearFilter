//
//  main.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 18.11.2025.
//

#include <iostream>
#include <vector>
#include "io/data_loader.h"
#include "utils/convolution_kernels.h"
#include "processors/processors.h"

int main(int argc, const char * argv[]) {
    const char* filePath = "EegLinearFilter/data/PN00-1.edf";
    const int convolutionKernelRadius = 128;
    const float convolutionKernelSigma = 1.0f;
    ProcessingMode mode = ProcessingMode::CPU_SEQ_AUTO_VEC;
    
    try {
        // TODO - kontrolovat, že konvoluční jádro není moc velké vzhledem k velikosti datasetu
        const std::vector<float> convolutionKernel = createGaussianKernel(convolutionKernelRadius, convolutionKernelSigma);
        std::vector<float> allData = loadEdfData(filePath, convolutionKernelRadius);
        
        // TODO - změř rychlost výpočtu
        switch (mode) {
            case ProcessingMode::CPU_SEQ_NO_VEC:
                std::cout << "Mode: Sequential processing on CPU (no-vectorization)" << std::endl;
                convolve_seq<false>(allData, convolutionKernel, convolutionKernelRadius);
                break;
            case ProcessingMode::CPU_SEQ_AUTO_VEC:
                std::cout << "Mode: Sequential processing on CPU (auto-vectorization)" << std::endl;
                convolve_seq<true>(allData, convolutionKernel, convolutionKernelRadius);
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
        
        // TODO - save data
        
        std::cout << "Načtená data (prvních 100 vzorků z celkových " << allData.size() << "):" << std::endl;
        for (size_t i = 0; i < 100 && i < allData.size(); ++i) {
            std::cout << allData[i] << " ";
            if ((i + 1) % 10 == 0) std::cout << std::endl;
        }
        std::cout << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Data processing failed: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
