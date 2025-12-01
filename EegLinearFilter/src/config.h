//
//  config.h
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 01.12.2025.
//

#ifndef CONFIG_H
#define CONFIG_H

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
#define CHUNK_SIZE 8192

// --- GPU parameters ---
#define THREADS_PER_GROUP 256
#define ITEMS_PER_THREAD 16
#define TILE_SIZE (THREADS_PER_GROUP * ITEMS_PER_THREAD)
#define KERNEL_SEGMENT_SIZE 1024

#endif // CONFIG_H
