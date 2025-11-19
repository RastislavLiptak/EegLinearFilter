//
//  main.cpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 18.11.2025.
//

#include <iostream>
#include <vector>
#include "data_loader.h"

int main(int argc, const char * argv[]) {
    const char* filePath = "EegLinearFilter/data/PN01-1.edf";
    
    try {
        // TODO - padding is missing
        std::vector<float> allData = loadEdfData(filePath);
        
        std::cout << "Načtená data (prvních 100 vzorků z celkových " << allData.size() << "):" << std::endl;
        for (size_t i = 0; i < 100 && i < allData.size(); ++i) {
            std::cout << allData[i] << " ";
            if ((i + 1) % 10 == 0) std::cout << std::endl;
        }
        std::cout << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Data processing failed: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
