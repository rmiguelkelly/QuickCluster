
#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <stdlib.h>
#include <iostream>

using std::cout;

template<typename T> void print_buffer(const T* buffer, size_t N) {
    cout << "[ ";
    for (size_t i = 0; i < N; i++) {
        cout << buffer[i];
        cout << ", ";
    }
    cout << " ]\n";
}

// Transposes in-place a matrix expressed as a single dimension buffer of values
template<typename T> void transpose_buffer(const T *original, T* result, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[i + j * rows] = original[j + i * cols];
        }
    }
}

// 1 2 3 4 5 6

// 1 3 5 2 4 6


#endif