//
//  main.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 18.11.2025.
//

#include <iostream>
#include <vector>

#include "io/io.hpp"
#include "benchmarks.hpp"
#include "convolution_kernels.hpp"
#include "config.h"

void show_welcome_banner() {
    std::cout << "========================================" << std::endl;
    std::cout << "                                        " << std::endl;
    std::cout << "              Welcome to                " << std::endl;
    std::cout << "  EEG Linear Filter Benchmark Suite     " << std::endl;
    std::cout << "          by Rastislav Lipták           " << std::endl;
    std::cout << "                                        " << std::endl;
    std::cout << "========================================" << std::endl;
}

int main(int argc, const char * argv[]) {
    show_welcome_banner();
    AppConfig config = configure_app();
    
    try {
        const std::vector<float> convolutionKernel = create_gaussian_kernel<KERNEL_RADIUS>(KERNEL_SIGMA);
        NeonVector allData = load_edf_data(config.filePath.c_str(), KERNEL_RADIUS);
        
        if (config.runAllVariants) {
            std::cout << "Starting benchmark suite" << std::endl;
            std::cout << "========================================\n";
            
            for (int i = 0; i < (int)ProcessingMode::COUNT; ++i) {
                NeonVector workingData = allData;
                run_benchmark(static_cast<ProcessingMode>(i), workingData, convolutionKernel, config.iterationCount, config.saveResults, config.outputFolderPath);
            }
        } else {
            run_benchmark(config.mode.value(), allData, convolutionKernel, config.iterationCount, config.saveResults, config.outputFolderPath);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "\nCRITICAL ERROR: Data processing failed.\nDetails: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
