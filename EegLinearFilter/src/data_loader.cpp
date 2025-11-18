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
#include <stdexcept>
#include <cstdint>
#include <string>
#include <algorithm>

void validateEdfHeader(const edflib_hdr_t& hdr, size_t& totalSamples, int& samplesToRead) {
    const int totalSignals = hdr.edfsignals;
    if (totalSignals <= 0) {
        throw std::runtime_error("No signals found in EDF header");
    }

    const long long samplesPerSignalLL = hdr.signalparam[0].smp_in_file;
    for (int s = 0; s < totalSignals; ++s) {
        if (hdr.signalparam[s].smp_in_file != samplesPerSignalLL) {
            throw std::runtime_error("The signals do not have the same number of samples: signal " + std::to_string(s) + " has " + std::to_string(hdr.signalparam[s].smp_in_file) + ", expected " + std::to_string(samplesPerSignalLL));
        }

        if (hdr.signalparam[s].dig_min < std::numeric_limits<int16_t>::min() ||
            hdr.signalparam[s].dig_max > std::numeric_limits<int16_t>::max()) {
            throw std::runtime_error("Digital min/max out of int16_t range for signal " + std::to_string(s) + ": min=" + std::to_string(hdr.signalparam[s].dig_min) + ", max=" + std::to_string(hdr.signalparam[s].dig_max));
        }
    }

    if (samplesPerSignalLL < 0 || samplesPerSignalLL > std::numeric_limits<int>::max()) {
        throw std::runtime_error("Number of samples per signal is out of int range");
    }
    samplesToRead = static_cast<int>(samplesPerSignalLL);

    const uint64_t totalSamplesU64 = static_cast<uint64_t>(totalSignals) * static_cast<uint64_t>(samplesPerSignalLL);
    if (totalSamplesU64 > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("Total samples exceed size_t range");
    }
    totalSamples = static_cast<size_t>(totalSamplesU64);
}

std::vector<int16_t> loadEdfData(const char* filePath) {
    edflib_hdr_t hdr;
    
    const int handle = edfopen_file_readonly(filePath, &hdr, EDFLIB_DO_NOT_READ_ANNOTATIONS);
    if (handle < 0) {
        throw std::runtime_error(std::string("Data load failed: ") + filePath);
    }

    struct EdfHandle {
        int handle;
        EdfHandle(int h) : handle(h) {}
        ~EdfHandle() { edfclose_file(handle); }
    } handleGuard(handle);

    size_t totalSamples = 0;
    int samplesToRead = 0;
    validateEdfHeader(hdr, totalSamples, samplesToRead);

    const int barWidth = 24;
    const int totalSignals = hdr.edfsignals;
    const long long samplesPerSignalLL = hdr.signalparam[0].smp_in_file; // Pro výpis

    std::cout << "Loading file: " << filePath << "\n";
    std::cout << "----------------------------------------\n";
    std::cout << "Signal count: " << totalSignals << "\n";
    std::cout << "Samples in signal: " << samplesPerSignalLL << "\n";
    std::cout << "Total sample count: " << static_cast<uint64_t>(totalSamples) << "\n" << std::flush; // Převedeno zpět pro výpis
    std::cout << "----------------------------------------\n";

    std::vector<int16_t> allData;
    allData.reserve(totalSamples);

    for (int signal = 0; signal < totalSignals; ++signal) {
        std::vector<int> tempBuffer(static_cast<std::size_t>(samplesPerSignalLL));
        const int readSamples = edfread_digital_samples(handle, signal, samplesToRead, tempBuffer.data());
        if (readSamples != samplesToRead) {
            throw std::runtime_error("Error reading samples for signal " + std::to_string(signal));
        }

        size_t oldSize = allData.size();
        allData.resize(oldSize + static_cast<size_t>(samplesToRead));
        std::transform(tempBuffer.begin(), tempBuffer.end(), allData.begin() + oldSize, [](int val) { return static_cast<int16_t>(val); });

        const float progress = static_cast<float>(signal + 1) / totalSignals;
        const int pos = static_cast<int>(progress * barWidth);

        std::cout << "\rLoading: [";
        for (int i = 0; i < barWidth; ++i) {
            std::cout << (i < pos ? "█" : " ");
        }
        std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100) << "%" << std::flush;
    }

    std::cout << "\nFile loaded!\n";
    std::cout << "----------------------------------------\n";

    return allData;
}
