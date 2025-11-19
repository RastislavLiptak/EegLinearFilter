//
//  convolve_seq.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 19.11.2025.
//

#include "processors.h"
#include <cstddef>
#include <stdexcept>

template <bool Vectorize>
void convolve_seq(std::vector<float>& data, const std::vector<float>& convolutionKernel, const int n) {
    const size_t dataSize = data.size();
    
    if constexpr (Vectorize) {
        for (size_t i = static_cast<size_t>(n); i < dataSize - static_cast<size_t>(n); ++i) {
            float sum = 0.0f;
            #pragma clang loop vectorize(enable)
            for (int j = -n; j <= n; ++j) {
                sum += data[i + j] * convolutionKernel[j + n];
            }
            data[i - n] = sum;
        }
    } else {
        for (size_t i = static_cast<size_t>(n); i < dataSize - static_cast<size_t>(n); ++i) {
            float sum = 0.0f;
            #pragma clang loop vectorize(disable)
            for (int j = -n; j <= n; ++j) {
                sum += data[i + j] * convolutionKernel[j + n];
            }
            data[i - n] = sum;
        }
    }
}

template void convolve_seq<true>(std::vector<float>&, const std::vector<float>&, const int);
template void convolve_seq<false>(std::vector<float>&, const std::vector<float>&, const int);
