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
    const int convolutionKernelRadius = 256;
    const float convolutionKernelSigma = 1.0f;
    ProcessingMode mode = ProcessingMode::CPU_SEQ_NO_VEC;
    
    try {
        // TODO - kontrolovat, že konvoluční jádro není moc velké vzhledem k velikosti datasetu
        const std::vector<float> convolutionKernel = createGaussianKernel(convolutionKernelRadius, convolutionKernelSigma);
        std::vector<float> allData = loadEdfData(filePath, convolutionKernelRadius);
        
        run_processor(
            mode,
            allData,
            convolutionKernel,
            convolutionKernelRadius
        );
        
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
