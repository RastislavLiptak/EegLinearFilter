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
    }

    if (samplesPerSignalLL < 0 || samplesPerSignalLL > std::numeric_limits<int>::max()) {
        throw std::runtime_error("Number of samples per signal is out of int range");
    }
    samplesToRead = static_cast<int>(samplesPerSignalLL);

    const std::uint64_t totalSamplesU64 = static_cast<uint64_t>(totalSignals) * static_cast<uint64_t>(samplesPerSignalLL);
    if (totalSamplesU64 > static_cast<uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error("Total samples exceed size_t range");
    }
    totalSamples = static_cast<size_t>(totalSamplesU64);
}

std::vector<float> loadEdfData(const char* filePath, const int padding) {
    edflib_hdr_t hdr;

    if (padding < 0) {
        throw std::runtime_error("Padding cannot be negative");
    }

    const int handle = edfopen_file_readonly(filePath, &hdr, EDFLIB_DO_NOT_READ_ANNOTATIONS);
    if (handle < 0) {
        throw std::runtime_error(std::string("Data load failed: ") + filePath);
    }

    struct EdfHandle {
        int handle;
        EdfHandle(int h) : handle(h) {}
        ~EdfHandle() { edfclose_file(handle); }
    } handleGuard(handle);

    size_t rawTotalSamples = 0;
    int samplesPerSignal = 0;
    validateEdfHeader(hdr, rawTotalSamples, samplesPerSignal);

    const int totalSignals = hdr.edfsignals;
    
    size_t samplesPerSignalPadded = static_cast<size_t>(samplesPerSignal) + (2 * padding);
    size_t totalSamplesPadded = static_cast<size_t>(totalSignals) * samplesPerSignalPadded;

    const int barWidth = 24;
    std::vector<double> tempBuffer(samplesPerSignal);
    std::vector<float> allData(totalSamplesPadded);

    std::cout << "Loading file: " << filePath << "\n";
    std::cout << "Signal count: " << totalSignals << "\n";
    std::cout << "Samples in signal: " << samplesPerSignal << "\n";
    std::cout << "Total samples: " << rawTotalSamples << "\n";

    for (int signal = 0; signal < totalSignals; ++signal) {
        const int read = edfread_physical_samples(handle, signal, samplesPerSignal, tempBuffer.data());
        if (read != samplesPerSignal) {
            throw std::runtime_error("Error reading signal " + std::to_string(signal));
        }

        if (read == 0) continue;

        const size_t channelStartOffset = static_cast<size_t>(signal) * samplesPerSignalPadded;

        float firstVal = static_cast<float>(tempBuffer[0]);
        float lastVal = static_cast<float>(tempBuffer[read - 1]);

        std::fill_n(allData.begin() + channelStartOffset, padding, firstVal);

        std::transform(
            tempBuffer.begin(),
            tempBuffer.begin() + read,
            allData.begin() + channelStartOffset + padding,
            [](double d) { return static_cast<float>(d); }
        );

        std::fill_n(allData.begin() + channelStartOffset + padding + read, padding, lastVal);

        const float progress = static_cast<float>(signal + 1) / totalSignals;
        const int pos = static_cast<int>(progress * barWidth);

        std::cout << "\rLoading: [";
        for (int i = 0; i < barWidth; ++i) {
            std::cout << (i < pos ? "█" : " ");
        }
        std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100) << "%" << std::flush;
    }

    std::cout << "\n----------------------------------------\n";

    return allData;
}
