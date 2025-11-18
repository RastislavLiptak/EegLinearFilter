//
//  data_loader.h
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 18.11.2025.
//

#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>
#include "../lib/edflib/edflib.h"

std::vector<int> loadEdfData(const char* filePath);

#endif // DATA_LOADER_H
