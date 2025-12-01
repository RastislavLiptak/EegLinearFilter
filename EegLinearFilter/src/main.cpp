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
#include "benchmarks.hpp"
#include "convolution_kernels.hpp"
#include "processors/processors.hpp"
#include "../lib/magic_enum/magic_enum.hpp"
#include "config.h"

void run_benchmark(const ProcessingMode mode, NeonVector& cleanData, const std::vector<float>& convolutionKernel, const int benchmark_iteration_count, const bool save_results, const char* outputFolderPath) {
    std::cout << "Mode: " << magic_enum::enum_name(mode) << std::endl;
    std::cout << "~~~~~~~~~~~~~~~~~~~" << std::endl;
    
    NeonVector tempData;
    NeonVector* workingDataPtr = nullptr;
    
    const size_t dataSize = cleanData.size();
    std::vector<double> execution_times(benchmark_iteration_count);
    
    for (int i = 0; i < benchmark_iteration_count; ++i) {
        if (benchmark_iteration_count == 1) {
            workingDataPtr = &cleanData;
        } else {
            tempData = cleanData;
            workingDataPtr = &tempData;
        }
        
        const std::chrono::duration<double> execution_time = run_processor<KERNEL_RADIUS, CHUNK_SIZE>(mode, *workingDataPtr, convolutionKernel);
        
        double time_sec = execution_time.count();
        if (benchmark_iteration_count > 1) {
            std::cout << "Run " << (i + 1) << " took " << time_sec << " seconds\n";
        }
        execution_times[i] = time_sec;
    }
    
    if (benchmark_iteration_count > 1) {
        std::cout << "~~~~~~~~~~~~~~~~~~~" << std::endl;
    }
    
    calc_benchmarks<KERNEL_RADIUS>(execution_times, dataSize);
    
    if (save_results) {
        std::string outputFilename = outputFolderPath + std::string(magic_enum::enum_name(mode)) + ".txt";
        save_data(*workingDataPtr, outputFilename, convolutionKernel);
        
        if (benchmark_iteration_count > 1) {
            tempData.clear();
            tempData.shrink_to_fit();
        }
    }
}

int main(int argc, const char * argv[]) {
//    TODO - nastavení parametrů přes příkazovou řádku
    const bool run_all_variants = false;
    const int benchmark_iteration_count = 10;
    const ProcessingMode mode = ProcessingMode::GPU;
    const bool save_results = false;
    const char* filePath = "EegLinearFilter/data/PN01-1.edf";
    const char* outputFolderPath = "EegLinearFilter/out/";
    
    try {
        const std::vector<float> convolutionKernel = create_gaussian_kernel<KERNEL_RADIUS>(KERNEL_SIGMA);
        NeonVector allData = load_edf_data(filePath, KERNEL_RADIUS);
        
        if (run_all_variants) {
            std::cout << "Starting benchmark suite" << std::endl;
            std::cout << "----------------------------------------\n";
            
            for (int i = 0; i < (int)ProcessingMode::COUNT; ++i) {
                NeonVector workingData = allData;
                run_benchmark(static_cast<ProcessingMode>(i), workingData, convolutionKernel, benchmark_iteration_count, save_results, outputFolderPath);
            }
        } else {
            run_benchmark(mode, allData, convolutionKernel, benchmark_iteration_count, save_results, outputFolderPath);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Data processing failed: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
