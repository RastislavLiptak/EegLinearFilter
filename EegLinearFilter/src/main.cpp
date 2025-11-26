//
//  main.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 18.11.2025.
//

#include <iostream>
#include <vector>
#include <string>
#include "io/io.hpp"
#include "convolution_kernels.hpp"
#include "processors/processors.hpp"

constexpr int CHUNK_SIZE = 8192;
constexpr int KERNEL_RADIUS = 256;
constexpr float KERNEL_SIGMA = 1.0f;

void run_processor(const ProcessingMode mode, NeonVector& allData, const std::vector<float>& convolutionKernel) {
    run_processor<KERNEL_RADIUS, CHUNK_SIZE>(
        mode,
        allData,
        convolutionKernel
    );
    
    std::string outputFilename = "EegLinearFilter/out/out_" + std::to_string((int)mode) + ".txt";
    save_data(allData, outputFilename, convolutionKernel);
}

int main(int argc, const char * argv[]) {
    const char* filePath = "EegLinearFilter/data/PN00-1.edf";
    const bool run_all_variants = false;
    
    try {
        const std::vector<float> convolutionKernel = create_gaussian_kernel<KERNEL_RADIUS>(KERNEL_SIGMA);
        NeonVector allData = load_edf_data(filePath, KERNEL_RADIUS);
        
        if (run_all_variants) {
            std::cout << "Starting benchmark suite" << std::endl;
            std::cout << "----------------------------------------\n";
            
            for (int i = 0; i < (int)ProcessingMode::COUNT; ++i) {
                NeonVector workingData = allData;
                run_processor(static_cast<ProcessingMode>(i), workingData, convolutionKernel);
            }
        } else {
            std::cout << "Starting single run" << std::endl;
            std::cout << "----------------------------------------\n";
            
            const ProcessingMode mode = ProcessingMode::GPU;
            run_processor(mode, allData, convolutionKernel);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Data processing failed: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
