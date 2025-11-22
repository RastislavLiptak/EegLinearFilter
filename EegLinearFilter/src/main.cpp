//
//  main.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 18.11.2025.
//

#include <iostream>
#include <vector>
#include "io/io.h"
#include "utils/convolution_kernels.h"
#include "processors/processors.h"

int main(int argc, const char * argv[]) {
    const char* filePath = "EegLinearFilter/data/PN00-1.edf";
    const int convolutionKernelRadius = 256;
    const float convolutionKernelSigma = 1.0f;
    const ProcessingMode mode = ProcessingMode::CPU_PAR_NO_VEC;
    
    try {
        const std::vector<float> convolutionKernel = createGaussianKernel(convolutionKernelRadius, convolutionKernelSigma);
        NeonVector allData = loadEdfData(filePath, convolutionKernelRadius);
        
        run_processor(
            mode,
            allData,
            convolutionKernel,
            convolutionKernelRadius
        );
        
        std::string outputFilename = "EegLinearFilter/out/out_" + std::to_string((int)mode) + ".txt";
        saveData(allData, outputFilename);
        std::cout << "Saved results to: " << outputFilename << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Data processing failed: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
