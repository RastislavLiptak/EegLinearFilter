//
//  convolution_kernels.h
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 19.11.2025.
//

#ifndef CONVOLUTION_KERNELS_H
#define CONVOLUTION_KERNELS_H

#include <vector>

std::vector<float> createGaussianKernel(const int radius, const float sigma);

#endif // CONVOLUTION_KERNELS_H
