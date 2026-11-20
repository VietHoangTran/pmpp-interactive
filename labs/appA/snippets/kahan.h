// Four loops that try to compensate a long sum, for appendix A, section A.6 ("Kahan,
// four ways"). One is Kahan's compensated summation; the lesson asks which. Plain C++ so
// that a unit test can compile it; the loop bodies are the same in a CUDA, HIP or Metal
// kernel, where the compiler's floating-point mode decides whether they survive.
#pragma once
#include <cstddef>

template <typename T>
T plain_sum(const T* x, std::size_t n) {
    T sum = T(0);
    for (std::size_t i = 0; i < n; ++i) sum = sum + x[i];
    return sum;
}

// Version 1
template <typename T>
T version1(const T* x, std::size_t n) {
    T sum = T(0), c = T(0);
    for (std::size_t i = 0; i < n; ++i) {
        T y = x[i] + c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

// Version 2
template <typename T>
T version2(const T* x, std::size_t n) {
    T sum = T(0), c = T(0);
    for (std::size_t i = 0; i < n; ++i) {
        T y = x[i] - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

// Version 3
template <typename T>
T version3(const T* x, std::size_t n) {
    T sum = T(0), c = T(0);
    for (std::size_t i = 0; i < n; ++i) {
        T y = x[i] - c;
        T t = sum + y;
        sum = t;
        c = (t - sum) - y;
    }
    return sum;
}

// Version 4
template <typename T>
T version4(const T* x, std::size_t n) {
    T sum = T(0), c = T(0);
    for (std::size_t i = 0; i < n; ++i) {
        T y = x[i] - c;
        T t = sum + y;
        c = T(0);  // fast math's (t - sum) - y
        sum = t;
    }
    return sum;
}
