
#ifndef ARRAY_H
#define ARRAY_H

#include <stdlib.h>
#include <iostream>
#include <memory>
#include <vector>

// Represents an immutable multi-dimension data structure that supports basic linear-algebra operations
template<typename T> class Array {

private:
    // Single dimension pointer to the data
    T *_data;

    // Array dimensions
    size_t _rows;
    size_t _cols;

    // Used to keep track if the underlying memory is managed by this class or outside
    bool _self_managed;

public:

    // Equates to an empty array
    Array();

    // The default copy constructor creates a new buffer, try to pass the array as a const reference for optimization if possible
    Array(const Array<T> &array);

    // Initialize with any array-like object
    Array(std::initializer_list<T> list);
    
    // Takes a single dimension buffer and resizes it, the length of the buffer must be equal to Rows x Cols.
    // If copy is set to true, the buffer is copied and the class handles its own memory
    Array(T* buffer, size_t Rows, size_t Cols, bool copy = false);

    // Create an array of random values
    static Array<T> random(size_t N, T max = 1);

    // Create an array of repeating values of type T
    static Array<T> values(size_t N, T value);

    // Create an array from a start to end
    static Array<T> range(T start, T end);

    // Destructor
    ~Array();

    // Resize the array to another dimension inplace
    void resize(size_t Rows, size_t Cols);

    // Count of all elements (Rows X Cols)
    size_t size() const;

    // Row count
    size_t rows() const;

    // Column count
    size_t cols() const;

    // Shape of the array
    std::pair<size_t, size_t> shape() const;

    // Returns the elements as a pointer
    const T* data() const;

    // Returns a value at an index treating the structure as a single dimension
    const T operator[](size_t index) const;

    // Same thing as above, just mutable
    T& operator[](size_t index);

    // Returns a row at an index
    const Array<T> row(size_t index) const;

    // Sets a value at an index treating this as a single dimension
    void set(size_t i, T value);

    // Sets a row at an index
    void set(size_t i, const Array<T> &row);

    // Calculates the mean of each column
    Array<T> mean() const;

    // Tranpose the matrix inplace
    Array<T> transposed();

    // Debug printing
    void __debug_print() const;

    std::string describe() const;

    // Creates a deep copy of the array with the same dimensions
    Array<T> copy() const;

    // Compares this array to another
    bool operator==(const Array<T> &other) const;
};

#endif