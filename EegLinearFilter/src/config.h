//
//  config.h
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 01.12.2025.
//

#ifndef CONFIG_H
#define CONFIG_H

// ==========================================
// 1. CONFIG CONSTANTS
// ==========================================

enum class ProcessingMode {
    CPU_SEQ_APPLE,           // Sequential benchmark implementation using Apple vDSP_conv method
    CPU_SEQ_NAIVE,           // Sequential naive approach without optimization
    CPU_SEQ_NO_VEC,          // Sequential processing, no vectorization
    CPU_SEQ_AUTO_VEC,        // Sequential, auto-vectorization
    CPU_SEQ_MANUAL_VEC,      // Sequential, manual vectorization
    CPU_PAR_NAIVE,           // Parallel naive approach without optimization
    CPU_PAR_NO_VEC,          // Parallel, no vectorization
    CPU_PAR_AUTO_VEC,        // Parallel, auto-vectorization
    CPU_PAR_MANUAL_VEC,      // Parallel, manual vectorization
    GPU,                     // GPU-accelerated
    
    COUNT
};

// --- Convolution kernel parameters ---
#define KERNEL_RADIUS 256
#define KERNEL_SIGMA 1.0f

// --- CPU parameters ---
#define CHUNK_SIZE 8192 // NOTE: must be a multiple of 16 for optimal NEON alignment.

// --- GPU parameters ---
#define THREADS_PER_GROUP 256 // NOTE: must be a multiple of 32 (Apple GPU SIMD width).
#define ITEMS_PER_THREAD 16
#define TILE_SIZE (THREADS_PER_GROUP * ITEMS_PER_THREAD)
#define KERNEL_SEGMENT_SIZE 1024



// ==========================================
// 2. CONFIG TESTS
// ==========================================

#ifndef __METAL_VERSION__

#include <type_traits>

// --- Convolution kernel parameters ---
static_assert(KERNEL_RADIUS > 0, "KERNEL_RADIUS must be positive.");
static_assert(KERNEL_SIGMA > 0.0f, "KERNEL_SIGMA must be positive for a valid kernel.");

// --- CPU parameters ---
static_assert(CHUNK_SIZE > 0, "CHUNK_SIZE must be greater than 0.");

// --- GPU parameters ---
static_assert(THREADS_PER_GROUP > 0, "THREADS_PER_GROUP must be greater than 0.");
static_assert(THREADS_PER_GROUP <= 1024, "THREADS_PER_GROUP cannot exceed hardware limit (1024).");
static_assert(ITEMS_PER_THREAD == 16, "ITEMS_PER_THREAD must be 16 (Metal kernel relies on explicit v0-v15 registers).");

constexpr std::size_t REQUIRED_THREADGROUP_MEM = (TILE_SIZE + KERNEL_SEGMENT_SIZE) * sizeof(float);
constexpr std::size_t MAX_SAFE_THREADGROUP_MEM = 32768;
static_assert(REQUIRED_THREADGROUP_MEM <= MAX_SAFE_THREADGROUP_MEM, "Required threadgroup memory exceeds GPU limit (32KB). Reduce KERNEL_SEGMENT_SIZE or THREADS_PER_GROUP.");

static_assert(KERNEL_SEGMENT_SIZE > 0, "KERNEL_SEGMENT_SIZE must be > 0.");
static_assert(KERNEL_SEGMENT_SIZE % 16 == 0, "KERNEL_SEGMENT_SIZE must be a multiple of 16 for kernel loop unrolling.");

#endif // __METAL_VERSION__

#endif // CONFIG_H
