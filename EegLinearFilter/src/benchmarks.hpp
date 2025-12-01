//
//  benchmarks.hpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 01.12.2025.
//
#ifndef BENCHMARKS_HPP
#define BENCHMARKS_HPP

#include "config.h"
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

    std::cout << "Final results:" << std::endl;
    std::cout << "Avg Time: " << avg_ex_time << " seconds" << std::endl;
    std::cout << "Throughput: " << megaSamplesPerSec << " MSamples/s" << std::endl;
    std::cout << "Performance: " << gigaFlops << " GFLOPS" << std::endl;
    std::cout << "----------------------------------------\n";
}

#endif // BENCHMARKS_HPP
