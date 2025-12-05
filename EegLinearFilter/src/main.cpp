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

void print_welcome_banner() {
    std::cout << "========================================" << std::endl;
    std::cout << "                                        " << std::endl;
    std::cout << "              Welcome to                " << std::endl;
    std::cout << "  EEG Linear Filter Benchmark Suite     " << std::endl;
    std::cout << "          by Rastislav Lipták           " << std::endl;
    std::cout << "                                        " << std::endl;
    std::cout << "========================================" << std::endl;
}

int main(int argc, const char * argv[]) {
    print_welcome_banner();
    
    bool keepRunning = true;
    do {
        AppConfig config = read_user_input();
        
        try {
            const std::vector<float> convolutionKernel = create_gaussian_kernel<KERNEL_RADIUS>(KERNEL_SIGMA);
            EdfData loadedData = load_edf_data(config.filePath.c_str(), KERNEL_RADIUS);
            
            if (config.runAllVariants) {
                std::cout << "Starting benchmark suite" << std::endl;
                std::cout << "========================================\n";
                
                for (int i = 0; i < (int)ProcessingMode::COUNT; ++i) {
                    NeonVector workingData = loadedData.samples;
                    run_benchmark(static_cast<ProcessingMode>(i), workingData, convolutionKernel, config.iterationCount, config.saveResults, config.outputFolderPath, loadedData);
                }
            } else {
                run_benchmark(config.mode.value(), loadedData.samples, convolutionKernel, config.iterationCount, config.saveResults, config.outputFolderPath, loadedData);
            }
            
            std::cout << "Done!" << std::endl;
            std::cout << "========================================\n";
            keepRunning = ask_to_continue();
            
        } catch (const std::exception& e) {
            std::cerr << "\nCRITICAL ERROR: Data processing failed.\nDetails: " << e.what() << std::endl;
            return EXIT_FAILURE;
        }
        
    } while (keepRunning);
    
    std::cout << "Exiting application. Goodbye!" << std::endl;
    return EXIT_SUCCESS;
}
