// Fig. 10.2 as a function: one loop, one binary operator, one starting value.
#pragma once
#include <cfloat>
#include <cstddef>

template <typename T, typename Op>
T reduce_seq(const T* input, std::size_t n, T start, Op op) {
    T acc = start;
    for (std::size_t i = 0; i < n; ++i) {
        acc = op(acc, input[i]);
    }
    return acc;
}

// The largest logit of a row. -FLT_MAX is at or below every finite float,
// and unlike -INFINITY it stays defined under Metal's fast math.
inline float max_logit(const float* row, std::size_t n) {
    return reduce_seq(row, n, -FLT_MAX,
                      [](float acc, float x) { return x > acc ? x : acc; });
}

// Greedy decoding's argmax: a max over (value, index) pairs. It starts from
// element 0 (footnote 1, p. 222), so it needs n >= 1. Only a strictly larger
// value replaces the pair, so on a tie the lower index stays.
struct Arg { float value; std::size_t index; };

inline Arg argmax_row(const float* row, std::size_t n) {
    Arg acc = {row[0], 0};
    for (std::size_t i = 1; i < n; ++i) {
        if (row[i] > acc.value) acc = {row[i], i};
    }
    return acc;
}
