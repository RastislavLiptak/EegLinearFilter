//
//  convolution_kernels.hpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 19.11.2025.
//

#ifndef CONVOLUTION_KERNELS_HPP
#define CONVOLUTION_KERNELS_HPP

#include <vector>
#include <cmath>
#include <iostream>

template <int Radius>
std::vector<float> create_gaussian_kernel(const float sigma) {
    static_assert(Radius >= 0, "Gaussian kernel radius cannot be negative");
    if (sigma <= 0.0f) {
        throw std::runtime_error("Gaussian kernel sigma must be positive");
    }
    
    std::cout << "Convolution kernel: Gaussian"<< std::endl;
    
    constexpr size_t size = 2 * Radius + 1;
    std::vector<float> kernel(size);
    
    float sum = 0.0f;
    
    const float denominator = 2.0f * sigma * sigma;
    for (int i = 0; i < size; ++i) {
        int x = i - Radius;
        kernel[i] = std::exp(-(x * x) / denominator);
        sum += kernel[i];
    }
    
    for (int i = 0; i < size; ++i) {
        kernel[i] /= sum;
    }
    
    std::cout << "Size: " << size << " | ";
    std::cout << "Radius: " << Radius << " | ";
    std::cout << "Sigma: " << sigma << "\n";
    std::cout << "----------------------------------------" << std::endl;
    
    return kernel;
}

#endif // CONVOLUTION_KERNELS_HPP
