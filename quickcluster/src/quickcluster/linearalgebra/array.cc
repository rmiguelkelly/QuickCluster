

#include <quickcluster/linearalgebra/array.h>
#include <quickcluster/linearalgebra/operations.h>

#include <algorithm>
#include <iostream>
#include <random>

using std::cout;

template<typename T> Array<T>::Array() {
    this->_rows = 0;
    this->_cols = 0;
    this->_data = nullptr; 
    this->_rc = new int(1);
}

template<typename T> Array<T>::Array(const Array<T> &array) {

    this->_data = array._data;
    this->_rows = array.rows();
    this->_cols = array.cols();
    this->_rc = array._rc;

    _self_managed = true;
    *_rc += 1;
}

template<typename T> Array<T>::Array(std::initializer_list<T> list) {

    size_t buffer_size = list.size() * sizeof(T);
    this->_data = (T*)malloc(buffer_size);
    memcpy(this->_data, list.begin(), buffer_size);

    this->_rows = 1;
    this->_cols = list.size();

    _self_managed = true;
    _rc = new int(1);
}

template<typename T> Array<T>::Array(T* buffer, size_t Rows, size_t Cols, bool copy, int *reference_count) {

    size_t buffer_size = Rows * Cols;
    
    // Check if we should just use the original buffer passed in as backing memory or make a copy
    if (copy) {
        this->_data = (T*)malloc(buffer_size * sizeof(T));
        memcpy(this->_data, buffer, buffer_size * sizeof(T));
    }
    else {
        this->_data = buffer;
         _self_managed = false;
    }

    this->_rows = Rows;
    this->_cols = Cols;

    if (reference_count == nullptr) {
        _rc = new int(1);
    }
    else {
        _rc = reference_count;
    }
}


template<typename T> Array<T> Array<T>::random(size_t N, T max) {

    T *buffer = new T[N];

    for (size_t i = 0; i < N; i++) {
        buffer[i] = static_cast<T>(max) * ((rand() % 100) / static_cast<T>(100));
    }

    auto array = Array<T>(buffer, 1, N, true);

    delete [] buffer;

    return array;
}


template<typename T> Array<T> Array<T>::values(size_t N, T value) {

    T *buffer = new T[N];

    for (size_t i = 0; i < N; i++) {
        buffer[i] = value;
    }

    auto array = Array<T>(buffer, 1, N, true);

    delete [] buffer;

    return array;
}

template<typename T> Array<T> Array<T>::range(T start, T end) {

    size_t index_start = static_cast<size_t>(start);
    size_t index_end = static_cast<size_t>(end);

    size_t N = index_end - index_start;

    T *buffer = new T[N];

    for (size_t i = index_start; i < index_end; i++) {
        buffer[i] = static_cast<T>(i);
    }

    auto array = Array<T>(buffer, 1, N, true);

    delete [] buffer;

    return array;
}


template<typename T> Array<T>::~Array() {

    // Decrement the reference count
    *_rc -= 1;

    if (_self_managed && this->_data != nullptr && *_rc == 0) {
        free(this->_data);
        this->_data = nullptr;
        delete _rc;
    }
}


template<typename T> void Array<T>::resize(size_t Rows, size_t Cols) {
    this->_rows = Rows;
    this->_cols = Cols;
}


template<typename T> inline size_t Array<T>::size() const {
    return this->_rows * this->_cols;
}


template<typename T> inline size_t Array<T>::rows() const {
    return this->_rows;
}


template<typename T> inline size_t Array<T>::cols() const {
    return this->_cols;
}


template<typename T> std::pair<size_t, size_t> Array<T>::shape() const {
    return std::make_pair(this->_rows, this->_cols);
}


template<typename T> const T* Array<T>::data() const {
    return this->_data;
}

// Inplace matrix transposition 
template<typename T> Array<T> Array<T>::transposed() {

    T *new_buffer = new T[this->size()];

    transpose_buffer(this->_data, new_buffer, this->_rows, this->_cols);

    auto transposed = Array<T>(new_buffer, this->_cols, this->_rows, true);

    delete [] new_buffer;

    return transposed;
}


template<typename T> const T Array<T>::operator[](size_t index) const {
    return this->_data[index];
}

template<typename T> T& Array<T>::operator[](size_t index) {
    return this->_data[index];
}


template<typename T> const Array<T> Array<T>::row(size_t index) const {

    *_rc += 1;

    size_t offset = index * this->_cols;
    return Array<T>(this->_data + offset, 1, this->_cols, false, this->_rc);
}


template<typename T> void Array<T>::set(size_t i, T value) {
    this->_data[i] = value;
}


template<typename T> void Array<T>::set(size_t i, const Array<T> &row) {

    // Memory offset
    size_t offset = i * this->_cols;
    size_t section_size = this->_cols * sizeof(T);
    memcpy(this->_data + offset, row.data(), section_size);
}


template<typename T> Array<T> Array<T>::mean() const {

    T buffer[this->_cols];

    size_t N = this->_rows * this->_cols;

    for (size_t col = 0; col < this->_cols; col++) {

        T sum = 0;

        for (size_t i = 0; i < N; i += this->_cols) {
            sum += this->_data[i + col];
        }

        buffer[col] = sum / static_cast<T>(this->_rows);
    }

    return Array<T>(buffer, 1, this->_cols, true);
}


template<typename T> void Array<T>::__debug_print() const {
    
    size_t limit = this->size();

    for (size_t i = 0; i < limit; i++) {

        if (i % this->_cols == 0 && i > 0) {
            cout << "\n";
        }

        cout << this->_data[i];
        cout << ", ";
    }

    cout << "\n";
}

template<typename T> std::string Array<T>::describe() const {

    size_t N = 1024;

    char buffer[N];

    size_t len = snprintf(buffer, N, "<Array shape=(%d, %d)>", (int)this->_rows, (int)this->_cols);
    buffer[len] = 0;

    return std::string(buffer);
}

template<typename T> Array<T> Array<T>::copy() const {
    return Array<T>(this->_data, this->_rows, this->_cols, true);
}

template<typename T> bool Array<T>::operator==(const Array<T> &other) const {

    if (this->_rows != other.rows()) {
        return false;
    }

    if (this->_cols != other.cols()) {
        return false;
    }

    return memcmp(this->_data, other.data(), this->size() * sizeof(T)) == 0;
}

template<typename T> int Array<T>::__reference_count() const {
    return *this->_rc;
}

template<typename T> void Array<T>::operator=(const Array<T> &other) {

    size_t buffer_size = sizeof(T) * other.size();

    this->_data = (T*)malloc(buffer_size);
    memcpy(this->_data, other.data(), buffer_size);

    this->_rows = other.rows();
    this->_cols = other.cols();
    this->_self_managed = true;
}

// Numeric types allowed for this template class
template class Array<char>;
template class Array<int>;
template class Array<long>;
template class Array<float>;
template class Array<double>;