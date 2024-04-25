
#include <quickcluster/benchmark/stopwatch.h>

#include <chrono>

using std::chrono::duration_cast;
using std::chrono::milliseconds;

void StopWatch::start() {
    this->time_start = benchmark_clock::now();
}

void StopWatch::stop() {
    this->time_end = benchmark_clock::now();
}

void StopWatch::reset() {
    this->time_start = benchmark_clock::now();
    this->time_end = benchmark_clock::now();
}

long long StopWatch::elapsed() const {
    return duration_cast<milliseconds>(this->time_end - this->time_start).count();
}

