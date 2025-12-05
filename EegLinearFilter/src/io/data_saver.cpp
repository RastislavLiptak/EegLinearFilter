//
//  data_saver.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 22.11.2025.
//

#include "io.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <stdexcept>
#include <cstring>

void copy_string_to_buffer(char* buffer, const std::string& source, size_t size) {
    memset(buffer, ' ', size);
    size_t len = source.length();
    if (len > size) len = size;
    memcpy(buffer, source.c_str(), len);
    buffer[size] = '\0';
}

void save_data(const NeonVector& processedData, const std::string& filepath, const std::vector<float>& convolutionKernel, const EdfData& sourceData) {
    std::cout << "Exporting to EDF: " << filepath << "..." << std::endl;

    std::filesystem::path pathObj(filepath);
    std::filesystem::path dirPath = pathObj.parent_path();
    if (!dirPath.empty() && !std::filesystem::exists(dirPath)) {
        std::filesystem::create_directories(dirPath);
    }

    int fileType = EDFLIB_FILETYPE_EDFPLUS;
    int channelCount = sourceData.header.num_signals;

    if (channelCount < 1) {
        throw std::runtime_error("Error: No channels to save.");
    }

    int handle = edfopen_file_writeonly(filepath.c_str(), fileType, channelCount);
    if (handle < 0) {
        throw std::runtime_error("Error: Could not open file for writing (edflib error code: " + std::to_string(handle) + ")");
    }

    edf_set_startdatetime(handle,
        sourceData.header.startdate_year,
        sourceData.header.startdate_month,
        sourceData.header.startdate_day,
        sourceData.header.starttime_hour,
        sourceData.header.starttime_minute,
        sourceData.header.starttime_second);

    edf_set_patientname(handle, sourceData.header.patient.c_str());
    edf_set_patientcode(handle, "EEG_BENCHMARK_EXPORT");
    edf_set_recording_additional(handle, sourceData.header.recording.c_str());

    for (int i = 0; i < channelCount; ++i) {
        const auto& ch = sourceData.channels[i];
        
        edf_set_samplefrequency(handle, i, (double)ch.smp_in_datarecord / ((double)sourceData.header.data_record_duration / 10000000.0));
        edf_set_physical_maximum(handle, i, ch.phys_max);
        edf_set_physical_minimum(handle, i, ch.phys_min);
        edf_set_digital_maximum(handle, i, ch.dig_max);
        edf_set_digital_minimum(handle, i, ch.dig_min);
        edf_set_label(handle, i, ch.label.c_str());
        edf_set_physical_dimension(handle, i, ch.dimension.c_str());
        edf_set_transducer(handle, i, ch.transducer.c_str());
        edf_set_prefilter(handle, i, "Linear Convolution Filter");
    }

    size_t kernelSize = convolutionKernel.size();
    size_t invalidSamples = kernelSize - 1;
    
    int originalSamplesPerSignal = sourceData.samplesPerSignal;
    
    int validSamplesPerSignal = originalSamplesPerSignal - (int)invalidSamples;
    if (validSamplesPerSignal < 0) validSamplesPerSignal = 0;

    int smpPerRecord = sourceData.channels[0].smp_in_datarecord;
    long long numRecords = validSamplesPerSignal / smpPerRecord;

    std::vector<double> writeBuffer(smpPerRecord);

    for (long long r = 0; r < numRecords; ++r) {
        for (int s = 0; s < channelCount; ++s) {
            size_t channelStartIdx = (size_t)s * sourceData.samplesPerSignalPadded + sourceData.padding;
            size_t currentOffset = channelStartIdx + (r * sourceData.channels[s].smp_in_datarecord);

            for(int k = 0; k < sourceData.channels[s].smp_in_datarecord; ++k) {
                writeBuffer[k] = (double)processedData[currentOffset + k];
            }

            if (edfwrite_physical_samples(handle, writeBuffer.data()) < 0) {
                std::cerr << "Error writing samples for record " << r << ", signal " << s << std::endl;
                edfclose_file(handle);
                return;
            }
        }
    }
    
    edfclose_file(handle);
    std::cout << "Done!" << std::endl;
    std::cout << "========================================\n";
}
