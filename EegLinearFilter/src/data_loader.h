//
//  data_loader.h
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 18.11.2025.
//

#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include "../lib/edflib/edflib.h"

std::vector<float> loadEdfData(const char* filePath);
void validateEdfHeader(const edflib_hdr_t& hdr, size_t& totalSamples, int& samplesToRead);

#endif // DATA_LOADER_H
