//
//  data_loader.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 18.11.2025.
//  Implementation of EDF file loading, data conversion, and padding logic.
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
#include <cstring>

// RAII wrapper to ensure EDF files are closed properly.
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

// Helper to remove trailing whitespace from strings.
std::string clean_string(const char* str) {
    std::string s(str);
    s.erase(s.find_last_not_of(" \n\r\t") + 1);
    return s;
}

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

/**
 * Loads an EDF file into memory.
 * Reads metadata, converts raw digital values to physical float values,
 * arranges data into a single continuous vector, and applies border padding.
 *
 * @param filePath Path to the .edf file.
 * @param padding Number of elements to pad at the beginning and end of each signal.
 * @return EdfData structure containing samples and header info.
 */
EdfData load_edf_data(const char* filePath, const int padding) {
    std::cout << "Loading file: " << filePath << "\n";

    edflib_hdr_t hdr;
    if (edfopen_file_readonly(filePath, &hdr, EDFLIB_DO_NOT_READ_ANNOTATIONS) < 0) {
        throw std::runtime_error("Header load failed");
    }

    EdfFileGuard fileGuard(hdr.handle);

    EdfData resultData;
    resultData.padding = padding;
    
    resultData.header.patient = clean_string(hdr.patient);
    resultData.header.recording = clean_string(hdr.recording);
    resultData.header.startdate_day = hdr.startdate_day;
    resultData.header.startdate_month = hdr.startdate_month;
    resultData.header.startdate_year = hdr.startdate_year;
    resultData.header.starttime_hour = hdr.starttime_hour;
    resultData.header.starttime_minute = hdr.starttime_minute;
    resultData.header.starttime_second = hdr.starttime_second;
    resultData.header.data_record_duration = hdr.datarecord_duration;
    resultData.header.num_signals = hdr.edfsignals;

    if (hdr.edfsignals <= 0) {
        throw std::runtime_error("No signals found");
    }

    long long samplesPerSignalLL = hdr.signalparam[0].smp_in_file;
    for (int s = 0; s < hdr.edfsignals; ++s) {
        if (hdr.signalparam[s].smp_in_file != samplesPerSignalLL) {
            throw std::runtime_error("Signals have mismatching sample counts.");
        }
    }
    if (samplesPerSignalLL > std::numeric_limits<int>::max()) throw std::runtime_error("Sample count too high");
    
    resultData.samplesPerSignal = static_cast<int>(samplesPerSignalLL);
    resultData.samplesPerSignalPadded = resultData.samplesPerSignal + (2 * padding);
    
    std::vector<ChannelInfo> loadParams(hdr.edfsignals);
    resultData.channels.resize(hdr.edfsignals);
    
    int bytesPerRecord = 0;

    for (int i = 0; i < hdr.edfsignals; ++i) {
        resultData.channels[i].label = clean_string(hdr.signalparam[i].label);
        resultData.channels[i].dimension = clean_string(hdr.signalparam[i].physdimension);
        resultData.channels[i].transducer = clean_string(hdr.signalparam[i].transducer);
        resultData.channels[i].prefilter = clean_string(hdr.signalparam[i].prefilter);
        resultData.channels[i].phys_min = hdr.signalparam[i].phys_min;
        resultData.channels[i].phys_max = hdr.signalparam[i].phys_max;
        resultData.channels[i].dig_min = hdr.signalparam[i].dig_min;
        resultData.channels[i].dig_max = hdr.signalparam[i].dig_max;
        resultData.channels[i].smp_in_datarecord = hdr.signalparam[i].smp_in_datarecord;

        loadParams[i].smpInRecord = hdr.signalparam[i].smp_in_datarecord;
        bytesPerRecord += loadParams[i].smpInRecord * 2;

        double phys_range = hdr.signalparam[i].phys_max - hdr.signalparam[i].phys_min;
        double dig_range = hdr.signalparam[i].dig_max - hdr.signalparam[i].dig_min;

        if (dig_range == 0) {
             loadParams[i].scale = 1.0f;
             loadParams[i].offset = 0.0f;
        } else {
             loadParams[i].scale = static_cast<float>(phys_range / dig_range);
             loadParams[i].offset = static_cast<float>(hdr.signalparam[i].phys_min - (hdr.signalparam[i].dig_min * loadParams[i].scale));
        }
    }

    size_t headerSize = 256 + (hdr.edfsignals * 256);
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open file binary");
    size_t fileSize = file.tellg();
    size_t dataSize = fileSize - headerSize;
    long long numRecords = dataSize / bytesPerRecord;

    size_t totalSamplesPadded = static_cast<size_t>(hdr.edfsignals) * resultData.samplesPerSignalPadded;
    resultData.samples.resize(totalSamplesPadded);

    file.seekg(headerSize, std::ios::beg);

    std::vector<int16_t> recordBuffer(bytesPerRecord / 2);
    std::vector<float*> channelWritePtrs(hdr.edfsignals);
    
    for(int s = 0; s < hdr.edfsignals; ++s) {
        channelWritePtrs[s] = &resultData.samples[static_cast<size_t>(s) * resultData.samplesPerSignalPadded + padding];
    }

    for (long long r = 0; r < numRecords; ++r) {
        if (!file.read(reinterpret_cast<char*>(recordBuffer.data()), bytesPerRecord)) {
             if (file.gcount() == 0) break;
        }

        int bufferOffset = 0;
        for (int s = 0; s < hdr.edfsignals; ++s) {
            const auto& ch = loadParams[s];
            float* dst = channelWritePtrs[s];

            for (int k = 0; k < ch.smpInRecord; ++k) {
                *dst = static_cast<float>(recordBuffer[bufferOffset++]) * ch.scale + ch.offset;
                dst++;
            }
            
            channelWritePtrs[s] = dst;
        }
    }

    file.close();

    // Apply border padding (replicate first/last value)
    for (int s = 0; s < hdr.edfsignals; ++s) {
        float* dataStart = &resultData.samples[static_cast<size_t>(s) * resultData.samplesPerSignalPadded + padding];
        
        if (resultData.samplesPerSignal > 0) {
            float firstVal = dataStart[0];
            std::fill_n(dataStart - padding, padding, firstVal);

            float lastVal = dataStart[resultData.samplesPerSignal - 1];
            std::fill_n(dataStart + resultData.samplesPerSignal, padding, lastVal);
        }
    }
    
    std::cout << "Signal count: " << hdr.edfsignals << "\n";
    std::cout << "Samples per signal: " << resultData.samplesPerSignal << "\n";
    std::cout << "Data size: " << (dataSize / 1024 / 1024) << " MB\n";
    std::cout << "========================================\n";

    return resultData;
}
