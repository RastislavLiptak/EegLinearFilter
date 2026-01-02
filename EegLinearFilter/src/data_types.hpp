//
//  data_types.hpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 25.11.2025.
//  Header file defining core data structures, configuration structs, and memory management helpers for the application.
//

#ifndef DATA_TYPES_HPP
#define DATA_TYPES_HPP

#include <stdlib.h>
#include <new>
#include <vector>
#include <string>
#include <optional>
#include <cstddef>

/**
 * Custom allocator to ensure memory alignment, required for SIMD (Neon) and GPU (Metal) buffers.
 */
template <typename T, std::size_t Alignment>
struct aligned_allocator {
    using value_type = T;
    
    template <class U>
    struct rebind {
        using other = aligned_allocator<U, Alignment>;
    };

    aligned_allocator() noexcept {}
    template <class U> aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        void* p;
        if (posix_memalign(&p, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(p);
    }

    void deallocate(T* p, std::size_t) noexcept {
        free(p);
    }

    template <class U> bool operator==(const aligned_allocator<U, Alignment>&) const noexcept { return true; }
    template <class U> bool operator!=(const aligned_allocator<U, Alignment>&) const noexcept { return false; }
};

// Vector alias using the aligned allocator (16KB alignment).
using NeonVector = std::vector<float, aligned_allocator<float, 16384>>;

/**
 * Structure holding user runtime configuration.
 */
struct AppConfig {
    std::string filePath;
    bool runAllVariants;
    std::optional<ProcessingMode> mode;
    int iterationCount;
    bool saveResults;
    std::string outputFolderPath;
};

/**
 * Structure holding parameters for a single EDF channel (signal).
 */
struct EdfChannelParams {
    std::string label;
    std::string dimension;
    std::string transducer;
    std::string prefilter;
    double phys_min;
    double phys_max;
    int dig_min;
    int dig_max;
    int smp_in_datarecord;
};

/**
 * Structure holding EDF main header information.
 */
struct EdfHeaderInfo {
    std::string patient;
    std::string recording;
    int startdate_day, startdate_month, startdate_year;
    int starttime_hour, starttime_minute, starttime_second;
    long data_record_duration;
    int num_signals;
};

/**
 * Main container for loaded EDF data, including raw samples and metadata.
 */
struct EdfData {
    NeonVector samples;
    EdfHeaderInfo header;
    std::vector<EdfChannelParams> channels;
    int samplesPerSignal;
    int samplesPerSignalPadded;
    int padding;
};

/**
 * Structure for collecting execution timing and memory metrics.
 */
struct ProcessingStats {
    double totalTimeSec;
    double computeTimeSec;
    double overheadTimeSec;
    double cpuMemoryOpsSec;
    double gpuMemoryOpsSec;
};

#endif // DATA_TYPES_HPP
