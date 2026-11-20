// Fig. 11.1 (p. 254) and the two conversions of p. 253, as plain host C++.
// Section 11.1 shows this file. Every chapter 11 lab's main.cpp runs the same
// left-to-right loop, in double, for its CPU reference, never std::inclusive_scan,
// whose order of additions the standard leaves open.
#pragma once

// Inclusive scan: output[i] = input[0] + ... + input[i]. N - 1 additions.
// The loop of Fig. 11.1, with const on the input and a guard for N == 0.
inline void sequential_scan(const float* input, float* output, unsigned int N) {
    if (N == 0) return;
    output[0] = input[0];
    for (unsigned int i = 1; i < N; ++i) {
        output[i] = output[i - 1] + input[i];
    }
}

// Exclusive scan: output[0] = 0, the identity of +, and
// output[i] = input[0] + ... + input[i - 1]. The same additions in the same order.
inline void sequential_exclusive_scan(const float* input, float* output, unsigned int N) {
    float running = 0.0f;
    for (unsigned int i = 0; i < N; ++i) {
        output[i] = running;
        running = running + input[i];
    }
}

// Inclusive to exclusive: shift right by one place and put the identity in slot 0.
inline void inclusive_to_exclusive(const float* inclusive, float* exclusive, unsigned int N) {
    if (N == 0) return;
    for (unsigned int i = N - 1; i > 0; --i) {
        exclusive[i] = inclusive[i - 1];
    }
    exclusive[0] = 0.0f;
}

// Exclusive to inclusive: shift left by one place and fill the last slot with
// the last exclusive value plus the last input.
inline void exclusive_to_inclusive(const float* exclusive, const float* input, float* inclusive,
                                   unsigned int N) {
    if (N == 0) return;
    for (unsigned int i = 0; i + 1 < N; ++i) {
        inclusive[i] = exclusive[i + 1];
    }
    inclusive[N - 1] = exclusive[N - 1] + input[N - 1];
}
