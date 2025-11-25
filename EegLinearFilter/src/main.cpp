//
//  main.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 18.11.2025.
//

#include <iostream>
#include <vector>
#include "io/io.hpp"
#include "convolution_kernels.hpp"
#include "processors/processors.hpp"

constexpr int KERNEL_RADIUS = 256;
constexpr float KERNEL_SIGMA = 1.0f;

int main(int argc, const char * argv[]) {
    const char* filePath = "EegLinearFilter/data/PN00-1.edf";
    const ProcessingMode mode = ProcessingMode::CPU_PAR_MANUAL_VEC;
    
    try {
        const std::vector<float> convolutionKernel = create_gaussian_kernel<KERNEL_RADIUS>(KERNEL_SIGMA);
        NeonVector allData = load_edf_data(filePath, KERNEL_RADIUS);
        
        run_processor<KERNEL_RADIUS>(
            mode,
            allData,
            convolutionKernel
        );
        
        std::string outputFilename = "EegLinearFilter/out/out_" + std::to_string((int)mode) + ".txt";
        save_data(allData, outputFilename, convolutionKernel);
        std::cout << "Saved results to: " << outputFilename << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Data processing failed: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
