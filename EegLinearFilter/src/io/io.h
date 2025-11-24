//
//  io.h
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 18.11.2025.
//

#ifndef IO_H
#define IO_H

#include <vector>
#include <stdlib.h>
#include <new>
#include "../../lib/edflib/edflib.h"

template <typename T, std::size_t Alignment>
struct aligned_allocator {
    using value_type = T;
    
    template <class U>
    struct rebind {
        using other = aligned_allocator<U, Alignment>;
    };

    aligned_allocator() noexcept {}
    template <class U> aligned_allocator(const aligned_allocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        void* p;
        if (posix_memalign(&p, Alignment, n * sizeof(T)) != 0) {
            throw std::bad_alloc();
        }
        return static_cast<T*>(p);
    }

    void deallocate(T* p, std::size_t) noexcept {
        free(p);
    }

    template <class U> bool operator==(const aligned_allocator<U, Alignment>&) const noexcept { return true; }
    template <class U> bool operator!=(const aligned_allocator<U, Alignment>&) const noexcept { return false; }
};

using NeonVector = std::vector<float, aligned_allocator<float, 16>>;
NeonVector load_edf_data(const char* filePath, const int padding = 0);
void save_data(const NeonVector& data, const std::string& filepath, const std::vector<float>& convolutionKernel);

#endif // IO_H
