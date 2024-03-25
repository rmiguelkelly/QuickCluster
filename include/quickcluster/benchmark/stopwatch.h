
#ifndef STOPWATCH_H
#define STOPWATCH_H

#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::time_point;
using std::chrono::milliseconds;

typedef high_resolution_clock benchmark_clock;

class StopWatch {

private:
    time_point<benchmark_clock> time_start;
    time_point<benchmark_clock> time_end;

public:

    // Starts the timer
    void start();

    // Ends the timer and records the elapsed time
    void stop();

    // Resets the timer
    void reset();

    // Gets the elapsed time in milliseconds
    long long elapsed() const;
};

#endif