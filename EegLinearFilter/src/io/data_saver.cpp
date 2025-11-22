//
//  data_saver.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 22.11.2025.
//

#include "io.h"
#include <fstream>
#include <string>
#include <iostream>
#include <filesystem>

void saveData(const NeonVector& data, const std::string& filepath) {
    std::filesystem::path pathObj(filepath);

    std::filesystem::path dirPath = pathObj.parent_path();

    if (!dirPath.empty() && !std::filesystem::exists(dirPath)) {
        std::filesystem::create_directories(dirPath);
    }

    std::ofstream file(filepath);

    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open/create file " + filepath);
    }

    for (const auto& val : data) {
        file << val << "\n";
    }
}
