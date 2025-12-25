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
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <ctime>

namespace fs = std::filesystem;

template <int Radius>
void log_benchmark_result(const std::string& mode, const std::string& filename, const size_t outputElements, const int iteration, const int totalIterations, const ProcessingStats& stats) {
    
    if (!fs::exists(LOGS_DIR)) {
        fs::create_directory(LOGS_DIR);
    }

    std::string csv_path = std::string(LOGS_DIR) + "/benchmark_results.csv";
    bool file_exists = fs::exists(csv_path);

    std::ofstream log_file(csv_path, std::ios::app);

    if (!file_exists) {
        log_file << "Timestamp;Mode;Filename;OutputElements;KernelRadius;Iteration;TotalIterations;"
                << "TotalTimeSec;ComputeTimeSec;OverheadTimeSec;CpuMemOpsSec;GpuMemOpsSec\n";
    }

    std::time_t now = std::time(nullptr);
    char time_buffer[100];
    if (std::strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%d %H:%M:%S", std::localtime(&now))) {
        log_file << time_buffer << ";"
                << mode << ";"
                << filename << ";"
                << outputElements << ";"
                << Radius << ";"
                << iteration << ";"
                << totalIterations << ";"
                << std::fixed << std::setprecision(9)
                << stats.totalTimeSec << ";"
                << stats.computeTimeSec << ";"
                << stats.overheadTimeSec << ";"
                << stats.cpuMemoryOpsSec << ";"
                << stats.gpuMemoryOpsSec << "\n";
    }
}

template <int Radius>
void calc_benchmarks(const std::vector<ProcessingStats>& stats, size_t dataSize) {
    constexpr size_t KernelSize = 2 * Radius + 1;
    const size_t outputElements = dataSize - KernelSize + 1;
    
    double sum_total_time = 0.0;
    double sum_compute_time = 0.0;
    double sum_overhead_time = 0.0;
    double sum_cpu_mem = 0.0;
    double sum_gpu_mem = 0.0;

    for (const auto& s : stats) {
        sum_total_time += s.totalTimeSec;
        sum_compute_time += s.computeTimeSec;
        sum_overhead_time += s.overheadTimeSec;
        sum_cpu_mem += s.cpuMemoryOpsSec;
        sum_gpu_mem += s.gpuMemoryOpsSec;
    }

    double avg_total_time = sum_total_time / stats.size();
    double avg_compute_time = sum_compute_time / stats.size();
    double avg_overhead_time = sum_overhead_time / stats.size();
    double avg_cpu_mem = sum_cpu_mem / stats.size();
    double avg_gpu_mem = sum_gpu_mem / stats.size();

    double calc_time = (avg_compute_time > 1e-9) ? avg_compute_time : avg_total_time;
    double megaSamplesPerSec = (outputElements / calc_time) / 1e6;
    double totalOperations = (double)outputElements * (double)KernelSize * 2.0;
    double gigaFlops = (totalOperations / calc_time) / 1e9;

    std::cout << "----------------------------------------\n";
    std::cout << "AVG results over " << stats.size() << " runs:" << std::endl;
    std::cout << "Time Breakdown:" << std::endl;
    std::cout << "  Total: " << avg_total_time << "s" << std::endl;
    std::cout << "  Compute: " << avg_compute_time << "s" << std::endl;
    std::cout << "  Mem Ops: " << avg_cpu_mem << "s (CPU)" << std::endl;
    if (avg_gpu_mem > 1e-9) {
        std::cout << "           " << avg_gpu_mem << "s (GPU)" << std::endl;
    }
    if (avg_overhead_time > 1e-9) {
        std::cout << "  Overhead: " << avg_overhead_time << "s (API/Launch)" << std::endl;
    }
    
    std::cout << "Metrics:" << std::endl;
    std::cout << "  Throughput: " << megaSamplesPerSec << " MSamples/s" << std::endl;
    std::cout << "  Performance: " << gigaFlops << " GFLOPS" << std::endl;
    std::cout << "========================================\n";
}

void run_benchmark(const ProcessingMode mode, const std::string& inputFilename, const EdfData& loadedData, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel, const int benchmark_iteration_count, const bool save_results, const std::string& outputFolderPath) {
    std::cout << "Mode: " << magic_enum::enum_name(mode) << std::endl;
    std::cout << "----------------------------------------\n";
    
    const size_t dataSize = loadedData.samples.size();
    std::vector<ProcessingStats> stats_collection(benchmark_iteration_count);
    
    for (int i = 0; i < benchmark_iteration_count; ++i) {
        std::cout << "Run " << (i + 1) << ": running..." << std::flush;
        
        ProcessingStats stats = run_processor<KERNEL_RADIUS, CHUNK_SIZE, K_BATCH>(mode, loadedData.samples, outputBuffer, convolutionKernel);
        log_benchmark_result<KERNEL_RADIUS>(
            std::string(magic_enum::enum_name(mode)),
            inputFilename,
            dataSize - (2 * KERNEL_RADIUS),
            i + 1,
            benchmark_iteration_count,
            stats
        );
        
        std::cout << "\rRun " << (i + 1) << ": ";
        if (stats.overheadTimeSec < 1e-9) {
            std::cout << stats.totalTimeSec << "s\033[K" << std::endl;
        } else {
            std::cout << stats.totalTimeSec << "s (Compute=" << stats.computeTimeSec << "s)\033[K" << std::endl;
        }
        
        stats_collection[i] = stats;
    }
    
    calc_benchmarks<KERNEL_RADIUS>(stats_collection, dataSize);
    
    if (save_results) {
        std::string outputFilename = outputFolderPath + std::string(magic_enum::enum_name(mode)) + ".edf";
        save_data(outputBuffer, outputFilename, convolutionKernel, loadedData);
    }
}

#endif // BENCHMARKS_HPP
