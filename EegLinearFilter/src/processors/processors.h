//
//  processors.h
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 19.11.2025.
//

#ifndef PROCESSORS_H
#define PROCESSORS_H

#include <vector>
#include "../io/io.h"

enum class ProcessingMode {
    CPU_SEQ_NO_VEC,          // Sequential processing, no vectorization
    CPU_SEQ_AUTO_VEC,        // Sequential, auto-vectorization
    CPU_SEQ_MANUAL_VEC,      // Sequential, manual vectorization
    CPU_PAR_NO_VEC,          // Parallel, no vectorization
    CPU_PAR_AUTO_VEC,        // Parallel, auto-vectorization
    CPU_PAR_MANUAL_VEC,      // Parallel, manual vectorization
    GPU_PAR                  // GPU-accelerated
};

void run_processor(const ProcessingMode mode, NeonVector& allData, const std::vector<float>& convolutionKernel, const int convolutionKernelRadius);

void convolve_seq_no_vec(NeonVector& data, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel, const int n);
void convolve_seq_auto_vec(NeonVector& data, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel, const int n);
void convolve_seq_manual_vec(NeonVector& data, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel);

void convolve_par_no_vec(const NeonVector& data, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel, const int n);
void convolve_par_auto_vec(const NeonVector& data, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel, const int n);
void convolve_par_manual_vec(const NeonVector& data, NeonVector& outputBuffer, const std::vector<float>& convolutionKernel);

#endif // PROCESSORS_H
