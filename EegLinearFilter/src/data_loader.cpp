//
//  data_loader.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 18.11.2025.
//

#include "data_loader.h"
#include <iostream>
#include <limits>

std::vector<int> loadEdfData(const char* filePath) {
    edflib_hdr_t hdr;
    
    int handle = edfopen_file_readonly(filePath, &hdr, EDFLIB_DO_NOT_READ_ANNOTATIONS);
    if (handle < 0) {
        std::cerr << "Data load failed: " << filePath << std::endl;
        return {};
    }
    
    std::vector<int> allData;
    
    for (int signal = 0; signal < hdr.edfsignals; ++signal) {
        long long numSamplesLL = hdr.signalparam[signal].smp_in_file;

        if (numSamplesLL < 0 || numSamplesLL > std::numeric_limits<int>::max()) {
            std::cerr << "The number of samples for the signal " << signal << " is out of range int: " << numSamplesLL << std::endl;
            edfclose_file(handle);
            return {};
        }
        int samplesToRead = static_cast<int>(numSamplesLL);

        if (static_cast<unsigned long long>(numSamplesLL) > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
            std::cerr << "The number of samples for signal " << signal << " is out of size_t range: " << numSamplesLL << std::endl;
            edfclose_file(handle);
            return {};
        }

        std::vector<int> signalData(static_cast<std::size_t>(numSamplesLL));
        
        int readSamples = edfread_digital_samples(handle, signal, samplesToRead, signalData.data());
        if (readSamples != samplesToRead) {
            std::cerr << "Error reading samples for signal " << signal
                      << " (expected " << samplesToRead << ", read " << readSamples << ")" << std::endl;
            edfclose_file(handle);
            return {};
        }
        
        allData.insert(allData.end(), signalData.begin(), signalData.end());
    }
    
    edfclose_file(handle);
    
    return allData;
}
