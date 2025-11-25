//
//  data_types.hpp
//  EegLinearFilter
//
//  Created by Rastislav Lipták on 25.11.2025.
//

#ifndef DATA_TYPES_HPP
#define DATA_TYPES_HPP

#include <stdlib.h>
#include <new>
#include <vector>
#include <cstddef>

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

#endif // DATA_TYPES_HPP
