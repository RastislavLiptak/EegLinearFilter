//
//  convolution_kernels.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 19.11.2025.
//

#include "convolution_kernels.h"
#include <cmath>
#include <iostream>

std::vector<float> createGaussianKernel(const int radius, const float sigma) {
    if (radius < 0) {
        throw std::runtime_error("Gaussian kernel radius cannot be negative");
    }
    if (sigma <= 0.0f) {
        throw std::runtime_error("Gaussian kernel sigma must be positive");
    }
    
    std::cout << "Generating a convolution kernel..."<< std::endl;
    
    const int size = 2 * radius + 1;
    std::vector<float> kernel(size);
    
    float sum = 0.0f;
    
    const float denominator = 2.0f * sigma * sigma;
    for (int i = 0; i < size; ++i) {
        int x = i - radius;
        kernel[i] = std::exp(-(x * x) / denominator);
        sum += kernel[i];
    }
    
    for (int i = 0; i < size; ++i) {
        kernel[i] /= sum;
    }
    
    std::cout << "Size: " << size << " | ";
    std::cout << "Radius: " << radius << " | ";
    std::cout << "Sigma: " << sigma << "\n";
    std::cout << "----------------------------------------" << std::endl;
    
    return kernel;
}
