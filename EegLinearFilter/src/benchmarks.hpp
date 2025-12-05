//
//  benchmarks.hpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 01.12.2025.
//

#ifndef BENCHMARKS_HPP
#define BENCHMARKS_HPP

#include "config.h"
#include "processors/processors.hpp"
#include "../lib/magic_enum/magic_enum.hpp"
#include <numeric>

template <int Radius>
void calc_benchmarks(std::vector<double> execution_times, size_t dataSize) {
    constexpr size_t KernelSize = 2 * Radius + 1;
    const size_t outputElements = dataSize - KernelSize + 1;
    
    double sum_ex_time = std::accumulate(execution_times.begin(), execution_times.end(), 0.0);
    double avg_ex_time = sum_ex_time / execution_times.size();

    double megaSamplesPerSec = (outputElements / avg_ex_time) / 1e6;

    double totalOperations = (double)outputElements * (double)KernelSize * 2.0;
    double gigaFlops = (totalOperations / avg_ex_time) / 1e9;

    std::cout << "----------------------------------------\n";
    std::cout << "Final results:" << std::endl;
    std::cout << "Avg Time: " << avg_ex_time << " seconds" << std::endl;
    std::cout << "Throughput: " << megaSamplesPerSec << " MSamples/s" << std::endl;
    std::cout << "Performance: " << gigaFlops << " GFLOPS" << std::endl;
    std::cout << "========================================\n";
}

void run_benchmark(const ProcessingMode mode, NeonVector& cleanData, const std::vector<float>& convolutionKernel, const int benchmark_iteration_count, const bool save_results, const std::string& outputFolderPath, const EdfData& originalData) {
    std::cout << "Mode: " << magic_enum::enum_name(mode) << std::endl;
    std::cout << "----------------------------------------\n";
    
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
        std::cout << "Run " << (i + 1) << " took " << time_sec << " seconds\n";
        execution_times[i] = time_sec;
    }
    
    calc_benchmarks<KERNEL_RADIUS>(execution_times, dataSize);
    
    if (save_results) {
        std::string outputFilename = outputFolderPath + std::string(magic_enum::enum_name(mode)) + ".edf";
        save_data(*workingDataPtr, outputFilename, convolutionKernel, originalData);
        
        if (benchmark_iteration_count > 1) {
            tempData.clear();
            tempData.shrink_to_fit();
        }
    }
}

#endif // BENCHMARKS_HPP
