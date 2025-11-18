//
//  data_loader.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 18.11.2025.
//

#include "data_loader.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <limits>

std::vector<int> loadEdfData(const char* filePath) {
    edflib_hdr_t hdr;
    
    int handle = edfopen_file_readonly(filePath, &hdr, EDFLIB_DO_NOT_READ_ANNOTATIONS);
    if (handle < 0) {
        std::cerr << "Data load failed: " << filePath << std::endl;
        return {};
    }
    
    int barWidth = 24;
    int totalSignals = hdr.edfsignals;
    size_t totalSamples = (size_t)totalSignals * hdr.signalparam[0].smp_in_file;

    std::cout << "Loading file: " << filePath << "\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Signal count: " << totalSignals << "\n";
    std::cout << "Samples in signal: " << hdr.signalparam[0].smp_in_file << "\n";
    std::cout << "Expected sample count: " << (long long)totalSamples << "\n" << std::flush;
    std::cout << "----------------------------------------\n";

    std::vector<int> allData;
    allData.reserve(totalSamples);
    for (int signal = 0; signal < totalSignals; ++signal) {
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

        float progress = (float)(signal + 1) / totalSignals;
        int pos = progress * barWidth;

        std::cout << "\rLoading: [";
        for (int i = 0; i < barWidth; ++i) {
            std::cout << (i < pos ? "█" : " ");
        }
        std::cout << "] " << std::setw(3) << (int)(progress * 100) << "%" << std::flush;
    }

    edfclose_file(handle);
    
    std::cout << "\n----------------------------------------\n";
    std::cout << "File loaded!\n";
    std::cout << "Loaded sample count: " << (long long)totalSamples << "\n" << std::flush;
    std::cout << "----------------------------------------\n";

    return allData;
}
