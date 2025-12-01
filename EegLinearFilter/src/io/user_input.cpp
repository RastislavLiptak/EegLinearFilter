//
//  user_input.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 01.12.2025.
//

//TODO - zvaliduj správně data
//TODO - pokud soubor EegLinearFilter/data/PN01-1.edf neexistuje, stáhni ho

#include "io.hpp"
#include <limits>
#include <optional>
#include <string>
#include <iostream>
#include <algorithm>
#include "../../lib/magic_enum/magic_enum.hpp"

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

void print_legend() {
    std::cout << "Controls:" << std::endl;
    std::cout << " [ENTER]: Confirm default value" << std::endl;
    std::cout << " 'b'    : Go back to previous step" << std::endl;
    std::cout << "----------------------------------------\n";
}

StepResult get_input_file_path(AppConfig& config) {
    std::string input_buffer;
    while (true) {
        std::cout << "Enter path to the input EDF file:\n";
        std::cout << "(Default: " << DEFAULT_FILE << ")\n> ";
        
        std::getline(std::cin, input_buffer);

        if (input_buffer == "b") return StepResult::BACK;

        if (input_buffer.empty()) {
            config.filePath = DEFAULT_FILE;
            return StepResult::NEXT;
        }

        config.filePath = input_buffer;
        return StepResult::NEXT;
    }
}

StepResult set_processing_mode(AppConfig& config) {
    std::cout << "Select benchmark mode:" << std::endl;
    std::cout << "-1 - WHOLE_BENCHMARK_SUITE (Default)" << std::endl;
    
    for (int i = 0; i < (int)ProcessingMode::COUNT; ++i) {
        auto mode_enum = static_cast<ProcessingMode>(i);
        std::cout << " " << i << " - " << magic_enum::enum_name(mode_enum) << std::endl;
    }

    std::string input_buffer;
    while (true) {
        std::cout << "> ";
        std::getline(std::cin, input_buffer);

        if (input_buffer == "b") return StepResult::BACK;

        int selected_mode = DEFAULT_MODE_INDEX;

        if (!input_buffer.empty()) {
            try {
                selected_mode = std::stoi(input_buffer);
            } catch (...) {
                std::cout << "Invalid input. Please enter a number." << std::endl;
                continue;
            }
        }

        if (selected_mode == -1) {
            config.runAllVariants = true;
            config.mode = std::nullopt;
            return StepResult::NEXT;
        } else if (selected_mode >= 0 && selected_mode < (int)ProcessingMode::COUNT) {
            config.runAllVariants = false;
            config.mode = static_cast<ProcessingMode>(selected_mode);
            return StepResult::NEXT;
        } else {
            std::cout << "Invalid selection. Enter a number between -1 and " << ((int)ProcessingMode::COUNT - 1) << "." << std::endl;
        }
    }
}

StepResult get_iteration_count(AppConfig& config) {
    std::string input_buffer;
    while (true) {
        std::cout << "Enter number of benchmark iterations\n";
        std::cout << "(Default: " << DEFAULT_ITERATIONS << ")\n> ";
        
        std::getline(std::cin, input_buffer);
        
        if (input_buffer == "b") return StepResult::BACK;

        if (input_buffer.empty()) {
            config.iterationCount = DEFAULT_ITERATIONS;
            return StepResult::NEXT;
        }

        try {
            int val = std::stoi(input_buffer);
            if (val > 0) {
                config.iterationCount = val;
                return StepResult::NEXT;
            } else {
                std::cout << "Number must be positive." << std::endl;
            }
        } catch (...) {
            std::cout << "Invalid input. Please enter an integer." << std::endl;
        }
    }
}

StepResult get_save_preference(AppConfig& config) {
    std::string input_buffer;
    while (true) {
        std::cout << "Do you want to save the results? (y/n):\n";
        std::cout << "(Default n)\n> ";
        
        std::getline(std::cin, input_buffer);

        if (input_buffer == "b") return StepResult::BACK;

        if (input_buffer.empty()) {
            config.saveResults = DEFAULT_SAVE;
            return StepResult::NEXT;
        }

        char response = std::tolower(input_buffer[0]);
        if (response == 'y') {
            config.saveResults = true;
            return StepResult::NEXT;
        }
        if (response == 'n') {
            config.saveResults = false;
            return StepResult::NEXT;
        }

        std::cout << "Invalid input. Please enter 'y' or 'n'." << std::endl;
    }
}

StepResult get_output_folder(AppConfig& config) {
    std::string input_buffer;
    while (true) {
        std::cout << "Enter output folder path:\n";
        std::cout << "(Default: " << DEFAULT_OUT_DIR << "):\n> ";
        
        std::getline(std::cin, input_buffer);

        if (input_buffer == "b") return StepResult::BACK;

        std::string path = input_buffer.empty() ? DEFAULT_OUT_DIR : input_buffer;

        if (!path.empty()) {
            if (path.back() != '/' && path.back() != '\\') {
                path += "/";
            }
            config.outputFolderPath = path;
            return StepResult::NEXT;
        }
        std::cout << "Error: Output path cannot be empty." << std::endl;
    }
}

AppConfig configure_app() {
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
                if (result == StepResult::NEXT) currentStep = ConfigStep::ITERATIONS;
                else currentStep = ConfigStep::FILE_INPUT;
                break;

            case ConfigStep::ITERATIONS:
                result = get_iteration_count(config);
                if (result == StepResult::NEXT) currentStep = ConfigStep::SAVE_PREF;
                else currentStep = ConfigStep::MODE_SELECT;
                break;

            case ConfigStep::SAVE_PREF:
                result = get_save_preference(config);
                if (result == StepResult::NEXT) {
                    if (config.saveResults) {
                        currentStep = ConfigStep::OUT_DIR;
                    } else {
                        config.outputFolderPath = "";
                        currentStep = ConfigStep::FINISHED;
                    }
                } else {
                    currentStep = ConfigStep::ITERATIONS;
                }
                break;

            case ConfigStep::OUT_DIR:
                result = get_output_folder(config);
                if (result == StepResult::NEXT) currentStep = ConfigStep::FINISHED;
                else currentStep = ConfigStep::SAVE_PREF;
                break;
                
            case ConfigStep::FINISHED:
                break;
        }
    }

    std::cout << "========================================" << std::endl;
    std::cout << "                                        " << std::endl;
    std::cout << "Configuration complete. Starting..." << std::endl;
    std::cout << "                                        " << std::endl;
    std::cout << "========================================" << std::endl;
    
    return config;
}
