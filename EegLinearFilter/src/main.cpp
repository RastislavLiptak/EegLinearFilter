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

//TODO - přidat validátor kompilačních konstant
constexpr int CHUNK_SIZE = 8192;
constexpr int KERNEL_RADIUS = 256;
constexpr float KERNEL_SIGMA = 1.0f;

void run_processor(const ProcessingMode mode, NeonVector& allData, const std::vector<float>& convolutionKernel, const bool save_results) {
    run_processor<KERNEL_RADIUS, CHUNK_SIZE>(
        mode,
        allData,
        convolutionKernel
    );
    
    if (save_results) {
        std::string outputFilename = "EegLinearFilter/out/out_" + std::to_string((int)mode) + ".txt";
        save_data(allData, outputFilename, convolutionKernel);
    }
}

int main(int argc, const char * argv[]) {
//    TODO - nastavení parametrů přes příkazovou řádku
    const bool run_all_variants = false;
    const ProcessingMode mode = ProcessingMode::GPU;
    const bool save_results = false;
    const char* filePath = "EegLinearFilter/data/PN01-1.edf";
    
    try {
        const std::vector<float> convolutionKernel = create_gaussian_kernel<KERNEL_RADIUS>(KERNEL_SIGMA);
        NeonVector allData = load_edf_data(filePath, KERNEL_RADIUS);
        
        if (run_all_variants) {
            std::cout << "Starting benchmark suite" << std::endl;
            std::cout << "----------------------------------------\n";
            
            for (int i = 0; i < (int)ProcessingMode::COUNT; ++i) {
                NeonVector workingData = allData;
                run_processor(static_cast<ProcessingMode>(i), workingData, convolutionKernel, save_results);
            }
        } else {
            run_processor(mode, allData, convolutionKernel, save_results);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Data processing failed: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
