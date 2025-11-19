//
//  processors.h
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 19.11.2025.
//

#ifndef PROCESSORS_H
#define PROCESSORS_H

#include <vector>

enum class ProcessingMode {
    CPU_SEQ_NO_VEC= 0,       // Sequential processing, no vectorization
    CPU_SEQ_AUTO_VEC,        // Sequential, auto-vectorization
    CPU_SEQ_MANUAL_VEC,      // Sequential, manual vectorization
    CPU_PAR_NO_VEC,          // Parallel, no vectorization
    CPU_PAR_AUTO_VEC,        // Parallel, auto-vectorization
    CPU_PAR_MANUAL_VEC,      // Parallel, manual vectorization
    GPU_PAR                  // GPU-accelerated
};

template <bool Vectorize>
void convolve_seq(std::vector<float>& data, const std::vector<float>& convolutionKernel, const int n);

#endif // PROCESSORS_H
