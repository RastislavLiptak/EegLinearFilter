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

void save_data(const NeonVector& data, const std::string& filepath, const std::vector<float>& convolutionKernel) {
    const size_t convolutionKernelSize = convolutionKernel.size();
    
    std::filesystem::path pathObj(filepath);
    std::filesystem::path dirPath = pathObj.parent_path();

    if (!dirPath.empty() && !std::filesystem::exists(dirPath)) {
        std::filesystem::create_directories(dirPath);
    }

    std::ofstream file(filepath);

    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open/create file " + filepath);
    }

    size_t totalSize = data.size();
    size_t limit = totalSize - convolutionKernelSize + 1;

    for (size_t i = 0; i < limit; ++i) {
        file << data[i] << "\n";
    }
}
