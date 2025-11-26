//
//  data_saver.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 22.11.2025.
//

#include "io.hpp"
#include <fstream>
#include <string>
#include <iostream>
#include <filesystem>
#include <iomanip>

void save_data(const NeonVector& data, const std::string& filepath, const std::vector<float>& convolutionKernel) {
    const size_t convolutionKernelSize = convolutionKernel.size();
    
    std::filesystem::path pathObj(filepath);
    std::filesystem::path dirPath = pathObj.parent_path();
    
    std::cout << "Saving file: " << filepath << std::endl;

    if (!dirPath.empty() && !std::filesystem::exists(dirPath)) {
        std::filesystem::create_directories(dirPath);
    }

    std::ofstream file(filepath);

    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open/create file " + filepath);
    }

    size_t totalSize = data.size();
    size_t limit = totalSize - convolutionKernelSize + 1;

    const int barWidth = 24;
        
    size_t updateStep = limit / 100;
    if (updateStep == 0) updateStep = 1;

    for (size_t i = 0; i < limit; ++i) {
        file << data[i] << "\n";

        if (i % updateStep == 0 || i == limit - 1) {
            const float progress = static_cast<float>(i + 1) / limit;
            const int pos = static_cast<int>(progress * barWidth);

            std::cout << "\rSaving: [";
            for (int j = 0; j < barWidth; ++j) {
                std::cout << (j < pos ? "█" : " ");
            }
            std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100) << "%" << std::flush;
        }
    }
    
    std::cout << std::endl;
    std::cout << "----------------------------------------\n";
}
