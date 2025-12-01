//
//  io.hpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 18.11.2025.
//

#ifndef IO_HPP
#define IO_HPP

#include "config.h"
#include "../data_types.hpp"
#include "../../lib/edflib/edflib.h"

AppConfig configure_app();
NeonVector load_edf_data(const char* filePath, const int padding = 0);
void save_data(const NeonVector& data, const std::string& filepath, const std::vector<float>& convolutionKernel);

#endif // IO_HPP
