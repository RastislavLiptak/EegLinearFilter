//
//  user_input_reader.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 01.12.2025.
//

//TODO - pokud soubor EegLinearFilter/data/PN01-1.edf neexistuje, stáhni ho

#include "io.hpp"
#include <limits>
#include <optional>
#include <string>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cctype>
#include "../../lib/magic_enum/magic_enum.hpp"

namespace fs = std::filesystem;

const std::string DEFAULT_FILE = "EegLinearFilter/data/PN01-1.edf";
const int DEFAULT_ITERATIONS = 10;
const bool DEFAULT_SAVE = false;
const std::string DEFAULT_OUT_DIR = "EegLinearFilter/out/";
const int DEFAULT_MODE_INDEX = -1;

enum class StepResult {
    NEXT,
    BACK
};

enum class ConfigStep {
    FILE_INPUT,
    MODE_SELECT,
    ITERATIONS,
    SAVE_PREF,
    OUT_DIR,
    FINISHED
};

// ==========================================
// HELPER FUNCTIONS
// ==========================================

std::string trim(const std::string& str) {
    const auto strBegin = str.find_first_not_of(" \t");
    if (strBegin == std::string::npos)
        return "";

    const auto strEnd = str.find_last_not_of(" \t");
    const auto strRange = strEnd - strBegin + 1;

    return str.substr(strBegin, strRange);
}

void print_legend() {
    std::cout << "Controls:" << std::endl;
    std::cout << " [ENTER]: Confirm default value" << std::endl;
    std::cout << " 'b'    : Go back to previous step" << std::endl;
    std::cout << "----------------------------------------\n";
}

void print_starting_message() {
    std::cout << "========================================\n";
    std::cout << "                                        \n";
    std::cout << "Configuration complete. Starting...     \n";
    std::cout << "                                        \n";
    std::cout << "========================================\n";
}

bool read_input(std::string& buffer) {
    std::cout << "> ";
    std::getline(std::cin, buffer);
    std::string trimmed = trim(buffer);
    return trimmed != "b";
}

std::optional<int> parse_strict_int(const std::string& input) {
    if (input.find('.') != std::string::npos || input.find(',') != std::string::npos) {
        return std::nullopt;
    }

    size_t processed_chars = 0;
    try {
        int val = std::stoi(input, &processed_chars);
        
        if (processed_chars != input.length()) {
            return std::nullopt;
        }
        return val;
    } catch (...) {
        return std::nullopt;
    }
}

bool can_write_to_dir(const fs::path& path) {
    fs::path test_file = path / "tmp_write_test.tmp";
    std::ofstream f(test_file);
    if (f.is_open()) {
        f.close();
        std::error_code ec;
        fs::remove(test_file, ec);
        return true;
    }
    return false;
}

// ==========================================
// VALIDATION LOGIC
// ==========================================

std::optional<std::string> try_parse_filepath(const std::string& input) {
    std::string clean_input = trim(input);
    
    fs::path path = clean_input.empty() ? DEFAULT_FILE : clean_input;

    std::error_code ec;
    if (!fs::exists(path, ec)) {
        std::cout << "Error: File does not exist: " << path << std::endl;
        return std::nullopt;
    }

    if (fs::is_directory(path, ec)) {
        std::cout << "Error: Path is a directory, not a file." << std::endl;
        return std::nullopt;
    }

    if (path.extension() != ".edf") {
        std::cout << "Error: File must have .edf extension." << std::endl;
        return std::nullopt;
    }
    
    return path.string();
}

std::optional<int> try_parse_mode(const std::string& input) {
    std::string clean_input = trim(input);
    if (clean_input.empty()) return DEFAULT_MODE_INDEX;

    auto val_opt = parse_strict_int(clean_input);

    if (val_opt.has_value()) {
        int val = val_opt.value();
        if (val == -1 || (val >= 0 && val < (int)ProcessingMode::COUNT)) {
            return val;
        } else {
             std::cout << "Invalid selection. Enter a number between -1 and " << ((int)ProcessingMode::COUNT - 1) << "." << std::endl;
             return std::nullopt;
        }
    }

    std::cout << "Invalid input. Please enter a valid integer (no decimals, no extra characters)." << std::endl;
    return std::nullopt;
}

std::optional<int> try_parse_iterations(const std::string& input) {
    std::string clean_input = trim(input);
    if (clean_input.empty()) return DEFAULT_ITERATIONS;

    auto val_opt = parse_strict_int(clean_input);

    if (val_opt.has_value()) {
        int val = val_opt.value();
        if (val > 0) return val;
        std::cout << "Number must be positive." << std::endl;
        return std::nullopt;
    }

    std::cout << "Invalid input. Please enter a valid integer." << std::endl;
    return std::nullopt;
}

std::optional<bool> try_parse_save_pref(const std::string& input) {
    std::string clean_input = trim(input);
    if (clean_input.empty()) return DEFAULT_SAVE;

    if (clean_input.length() == 1) {
        char response = std::tolower(clean_input[0]);
        if (response == 'y') return true;
        if (response == 'n') return false;
    }

    std::cout << "Invalid input. Please enter 'y' or 'n'." << std::endl;
    return std::nullopt;
}

std::optional<std::string> try_parse_output_dir(const std::string& input) {
    std::string clean_input = trim(input);
    std::string path_str = clean_input.empty() ? DEFAULT_OUT_DIR : clean_input;

    if (path_str.empty()) {
        std::cout << "Error: Output path cannot be empty." << std::endl;
        return std::nullopt;
    }

    if (path_str.back() != '/' && path_str.back() != '\\') {
        path_str += "/";
    }
    
    fs::path path(path_str);
    std::error_code ec;

    if (fs::exists(path, ec)) {
        if (!fs::is_directory(path, ec)) {
            std::cout << "Error: Path exists but it is a file, not a directory." << std::endl;
            return std::nullopt;
        }

        if (!can_write_to_dir(path)) {
            std::cout << "Error: Directory exists, but is not writable (permission denied)." << std::endl;
            return std::nullopt;
        }
    } else {
        if (!fs::create_directories(path, ec)) {
            if (ec) {
                std::cout << "Error: Cannot create directory. Reason: " << ec.message() << std::endl;
                return std::nullopt;
            }
        }
        
        if (!can_write_to_dir(path)) {
             std::cout << "Error: Created directory, but cannot write to it." << std::endl;
             return std::nullopt;
        }
    }

    return path_str;
}


// ==========================================
// NAVIGATION LOGIC
// ==========================================

StepResult get_input_file_path(AppConfig& config) {
    std::string input_buffer;
    while (true) {
        std::cout << "Enter path to the input EDF file:\n";
        std::cout << "(Default: " << DEFAULT_FILE << ")\n";
        if (!read_input(input_buffer)) return StepResult::BACK;

        if (auto result = try_parse_filepath(input_buffer)) {
            config.filePath = *result;
            return StepResult::NEXT;
        }
    }
}

StepResult set_processing_mode(AppConfig& config) {
    std::cout << "Select benchmark mode:" << std::endl;
    std::cout << "-1 - WHOLE_BENCHMARK_SUITE (Default)" << std::endl;
    for (int i = 0; i < (int)ProcessingMode::COUNT; ++i) {
        std::cout << " " << i << " - " << magic_enum::enum_name(static_cast<ProcessingMode>(i)) << std::endl;
    }

    std::string input_buffer;
    while (true) {
        if (!read_input(input_buffer)) return StepResult::BACK;

        if (auto result = try_parse_mode(input_buffer)) {
            int mode_index = *result;
            if (mode_index == -1) {
                config.runAllVariants = true;
                config.mode = std::nullopt;
            } else {
                config.runAllVariants = false;
                config.mode = static_cast<ProcessingMode>(mode_index);
            }
            return StepResult::NEXT;
        }
    }
}

StepResult get_iteration_count(AppConfig& config) {
    std::string input_buffer;
    while (true) {
        std::cout << "Enter number of benchmark iterations\n";
        std::cout << "(Default: " << DEFAULT_ITERATIONS << ")\n";
        if (!read_input(input_buffer)) return StepResult::BACK;

        if (auto result = try_parse_iterations(input_buffer)) {
            config.iterationCount = *result;
            return StepResult::NEXT;
        }
    }
}

StepResult get_save_preference(AppConfig& config) {
    std::string input_buffer;
    while (true) {
        std::cout << "Do you want to save the results? (y/n):\n";
        std::cout << "(Default n)\n";
        if (!read_input(input_buffer)) return StepResult::BACK;

        if (auto result = try_parse_save_pref(input_buffer)) {
            config.saveResults = *result;
            return StepResult::NEXT;
        }
    }
}

StepResult get_output_folder(AppConfig& config) {
    std::string input_buffer;
    while (true) {
        std::cout << "Enter output folder path:\n";
        std::cout << "(Default: " << DEFAULT_OUT_DIR << "):\n";
        if (!read_input(input_buffer)) return StepResult::BACK;

        if (auto result = try_parse_output_dir(input_buffer)) {
            config.outputFolderPath = *result;
            return StepResult::NEXT;
        }
    }
}

AppConfig read_user_input() {
    AppConfig config;
    print_legend();

    ConfigStep currentStep = ConfigStep::FILE_INPUT;

    while (currentStep != ConfigStep::FINISHED) {
        StepResult result;

        switch (currentStep) {
            case ConfigStep::FILE_INPUT:
                result = get_input_file_path(config);
                if (result == StepResult::NEXT) {
                    currentStep = ConfigStep::MODE_SELECT;
                } else {
                    std::cout << "Already at the beginning.\n";
                }
                break;

            case ConfigStep::MODE_SELECT:
                result = set_processing_mode(config);
                currentStep = (result == StepResult::NEXT) ? ConfigStep::ITERATIONS : ConfigStep::FILE_INPUT;
                break;

            case ConfigStep::ITERATIONS:
                result = get_iteration_count(config);
                currentStep = (result == StepResult::NEXT) ? ConfigStep::SAVE_PREF : ConfigStep::MODE_SELECT;
                break;

            case ConfigStep::SAVE_PREF:
                result = get_save_preference(config);
                if (result == StepResult::NEXT) {
                    currentStep = config.saveResults ? ConfigStep::OUT_DIR : ConfigStep::FINISHED;
                    if (!config.saveResults) {
                        config.outputFolderPath = "";
                    }
                } else {
                    currentStep = ConfigStep::ITERATIONS;
                }
                break;

            case ConfigStep::OUT_DIR:
                result = get_output_folder(config);
                currentStep = (result == StepResult::NEXT) ? ConfigStep::FINISHED : ConfigStep::SAVE_PREF;
                break;
                
            case ConfigStep::FINISHED: break;
        }
    }

    print_starting_message();
    return config;
}
