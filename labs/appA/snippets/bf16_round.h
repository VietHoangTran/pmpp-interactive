// FP32 to BF16 on the bits, as llama.cpp converts it (ggml_compute_fp32_to_bf16 in
// ggml/src/ggml-impl.h; notes/today/appA-facts.md D3): round to nearest, ties to even,
// NaN forced quiet, denormals kept. Shown in appendix A, section A.4.
// The two lines that build the result are llama.cpp's; the NaN test and the use of
// memcpy instead of a union are ours.
#pragma once
#include <cstdint>
#include <cstring>

inline std::uint16_t bf16_round(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    if ((u & 0x7fffffffu) > 0x7f800000u) {          // NaN: exponent all ones, mantissa not 0
        return (std::uint16_t)((u >> 16) | 64);      // keep the top half, force it quiet
    }
    return (std::uint16_t)((u + (0x7fffu + ((u >> 16) & 1u))) >> 16);
}

// Truncation, for comparison: keep the top 16 bits and drop the rest.
inline std::uint16_t bf16_truncate(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return (std::uint16_t)(u >> 16);
}

inline float bf16_to_float(std::uint16_t h) {
    std::uint32_t u = (std::uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
}
