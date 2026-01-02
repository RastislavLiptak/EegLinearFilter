//
//  data_downloader.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 02.12.2025.
//  Implementation of file downloading functionality using libcurl
//

#include "io.hpp"
#include <iostream>
#include <fstream>
#include <curl/curl.h>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <cmath>

namespace fs = std::filesystem;

/**
 * Formats a byte count into a human-readable string (e.g., "1.5 MB").
 * @param bytes The number of bytes.
 * @return Formatted string.
 */
std::string format_bytes(curl_off_t bytes) {
    const char* suffixes[] = {"B", "KB", "MB", "GB", "TB"};
    int suffix_index = 0;
    double count = static_cast<double>(bytes);

    while (count >= 1024 && suffix_index < 4) {
        count /= 1024;
        suffix_index++;
    }

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << count << " " << suffixes[suffix_index];
    return ss.str();
}

/**
 * Callback function for libcurl to write received data to a file stream.
 */
size_t write_data(void* ptr, size_t size, size_t nmemb, FILE* stream) {
    size_t written = fwrite(ptr, size, nmemb, stream);
    return written;
}

/**
 * Callback function for libcurl to display a progress bar in the console.
 */
int progress_callback(void* clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t ultotal, curl_off_t ulnow) {
    const int barWidth = 33;
    
    if (dltotal <= 0) {
        return 0;
    }

    double fraction = static_cast<double>(dlnow) / static_cast<double>(dltotal);
    if (fraction > 1.0) {
        fraction = 1.0;
    }

    int percentage = static_cast<int>(fraction * 100);
    int pos = static_cast<int>(barWidth * fraction);

    std::cout << "\r[";
    for (int i = 0; i < barWidth; ++i) {
        std::cout << (i < pos ? "█" : " ");
    }
    std::cout << "] " << std::setw(3) << percentage << "% " << "(" << format_bytes(dlnow) << " / " << format_bytes(dltotal) << ")";
    std::cout.flush();

    return 0;
}

/**
 * Creates the directory structure for a given file path if it does not exist.
 * @param filepath The full path to the file.
 * @return True if successful, false otherwise.
 */
bool ensure_directory_exists(const std::string& filepath) {
    try {
        fs::path path(filepath);
        fs::path dir = path.parent_path();
        
        if (!dir.empty() && !fs::exists(dir)) {
            if (fs::create_directories(dir)) {
                return true;
            } else {
                return false;
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
        return false;
    }
    return true;
}

/**
 * Downloads a file from a URL to a local path.
 * @param url Source URL.
 * @param filepath Local destination path.
 * @return True if download was successful.
 */
bool download_file(const std::string& url, const std::string& filepath) {
    CURL* curl;
    FILE* fp;
    CURLcode res;
    bool success = false;

    std::string temp_filepath = filepath + ".tmp";

    if (fs::exists(temp_filepath)) {
        fs::remove(temp_filepath);
    }

    if (!ensure_directory_exists(filepath)) {
        std::cerr << "Error: Could not create target directory for: " << filepath << std::endl;
        return false;
    }

    curl = curl_easy_init();
    if (curl) {
        fp = fopen(temp_filepath.c_str(), "wb");
        if (!fp) {
            std::cerr << "Error: Cannot open file for writing: " << temp_filepath << std::endl;
            curl_easy_cleanup(curl);
            return false;
        }

        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);

        res = curl_easy_perform(curl);
        std::cout << std::endl;

        if (fp) fclose(fp);

        if (res == CURLE_OK) {
            try {
                if (fs::exists(filepath)) {
                    fs::remove(filepath);
                }
                
                fs::rename(temp_filepath, filepath);
                success = true;
            } catch (const fs::filesystem_error& e) {
                std::cerr << "Error renaming file: " << e.what() << std::endl;
                fs::remove(temp_filepath);
                success = false;
            }
        } else {
            std::cerr << "Download failed: " << curl_easy_strerror(res) << std::endl;
            fs::remove(temp_filepath);
            success = false;
        }

        curl_easy_cleanup(curl);
    } else {
        std::cerr << "Error: Failed to initialize curl." << std::endl;
    }

    return success;
}
