//
//  data_loader.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 18.11.2025.
//

#include "io.hpp"
#include <iostream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <vector>
#include <cmath>
#include <fstream>

struct EdfFileGuard {
    int handle;

    explicit EdfFileGuard(int h) : handle(h) {}
    
    ~EdfFileGuard() {
        if (handle >= 0) {
            edfclose_file(handle);
        }
    }

    EdfFileGuard(const EdfFileGuard&) = delete;
    EdfFileGuard& operator=(const EdfFileGuard&) = delete;
};

struct ChannelInfo {
    int smpInRecord;
    float scale;
    float offset;
};

struct FileMetadata {
    int totalSignals;
    int samplesPerSignal;
    size_t totalSamples;
    int bytesPerRecord;
    long long numRecords;
    size_t samplesPerSignalPadded;
    size_t headerSize;
    size_t dataSize;
};

FileMetadata parse_header_and_channels(const char* filePath, const int padding, std::vector<ChannelInfo>& channels) {
    edflib_hdr_t hdr;
    if (edfopen_file_readonly(filePath, &hdr, EDFLIB_DO_NOT_READ_ANNOTATIONS) < 0) {
        throw std::runtime_error("Header load failed");
    }

    EdfFileGuard fileGuard(hdr.handle);
    FileMetadata meta;
    meta.totalSignals = hdr.edfsignals;
    
    if (meta.totalSignals <= 0) {
        throw std::runtime_error("No signals found in EDF header");
    }

    long long samplesPerSignalLL = hdr.signalparam[0].smp_in_file;
    for (int s = 0; s < meta.totalSignals; ++s) {
        if (hdr.signalparam[s].smp_in_file != samplesPerSignalLL) {
            throw std::runtime_error("Signals have mismatching sample counts.");
        }
    }

    if (samplesPerSignalLL < 0 || samplesPerSignalLL > std::numeric_limits<int>::max()) {
        throw std::runtime_error("Sample count out of int range");
    }

    meta.samplesPerSignal = static_cast<int>(samplesPerSignalLL);
    meta.totalSamples = static_cast<size_t>(meta.totalSignals) * static_cast<size_t>(meta.samplesPerSignal);

    channels.resize(meta.totalSignals);
    meta.bytesPerRecord = 0;

    for (int i = 0; i < meta.totalSignals; ++i) {
        channels[i].smpInRecord = hdr.signalparam[i].smp_in_datarecord;
        meta.bytesPerRecord += channels[i].smpInRecord * 2;

        double phys_min = hdr.signalparam[i].phys_min;
        double phys_max = hdr.signalparam[i].phys_max;
        double dig_min  = hdr.signalparam[i].dig_min;
        double dig_max  = hdr.signalparam[i].dig_max;

        if (dig_max == dig_min) {
            channels[i].scale = 1.0f;
            channels[i].offset = 0.0f;
        } else {
            channels[i].scale = static_cast<float>((phys_max - phys_min) / (dig_max - dig_min));
            channels[i].offset = static_cast<float>(phys_min - (dig_min * channels[i].scale));
        }
    }

    meta.samplesPerSignalPadded = static_cast<size_t>(meta.samplesPerSignal) + (2 * padding);
    meta.headerSize = 256 + (meta.totalSignals * 256);
    
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open file to check size");
    }
    std::streamsize fileSize = file.tellg();

    meta.dataSize = fileSize - meta.headerSize;
    if (meta.dataSize <= 0) {
        throw std::runtime_error("Invalid data size");
    }

    meta.numRecords = meta.dataSize / meta.bytesPerRecord;

    return meta;
}

NeonVector load_edf_data(const char* filePath, const int padding) {
    std::cout << "Loading file: " << filePath << "\n";
    
    std::vector<ChannelInfo> channels;
    FileMetadata meta = parse_header_and_channels(filePath, padding, channels);

    std::cout << "Signal count: " << meta.totalSignals << "\n";
    std::cout << "Samples per signal: " << meta.samplesPerSignal << "\n";
    std::cout << "Total records: " << (meta.totalSignals * meta.samplesPerSignal) << "\n";
    std::cout << "Data size: " << (meta.dataSize / 1024 / 1024) << " MB\n";
    std::cout << "----------------------------------------\n";
    
    size_t totalSamplesPadded = static_cast<size_t>(meta.totalSignals) * meta.samplesPerSignalPadded;
    NeonVector allData(totalSamplesPadded);

    std::ifstream file(filePath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for binary read");
    }
    file.seekg(meta.headerSize, std::ios::beg);

    std::vector<int16_t> recordBuffer(meta.bytesPerRecord / 2);

    std::vector<float*> channelWritePtrs(meta.totalSignals);
    for(int s = 0; s < meta.totalSignals; ++s) {
        channelWritePtrs[s] = &allData[static_cast<size_t>(s) * meta.samplesPerSignalPadded + padding];
    }

    for (long long r = 0; r < meta.numRecords; ++r) {
        if (!file.read(reinterpret_cast<char*>(recordBuffer.data()), meta.bytesPerRecord)) {
             if (file.gcount() == 0) break;
        }

        int bufferOffset = 0;
        for (int s = 0; s < meta.totalSignals; ++s) {
            const auto& ch = channels[s];
            float* dst = channelWritePtrs[s];

            for (int k = 0; k < ch.smpInRecord; ++k) {
                *dst = static_cast<float>(recordBuffer[bufferOffset++]) * ch.scale + ch.offset;
                dst++;
            }
            
            channelWritePtrs[s] = dst;
        }
    }

    file.close();

    for (int s = 0; s < meta.totalSignals; ++s) {
        float* dataStart = &allData[static_cast<size_t>(s) * meta.samplesPerSignalPadded + padding];
        
        if (meta.samplesPerSignal > 0) {
            float firstVal = dataStart[0];
            std::fill_n(dataStart - padding, padding, firstVal);

            float lastVal = dataStart[meta.samplesPerSignal - 1];
            std::fill_n(dataStart + meta.samplesPerSignal, padding, lastVal);
        }
    }

    return allData;
}
