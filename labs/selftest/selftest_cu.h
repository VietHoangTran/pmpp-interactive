// Host-side helpers for the self-test .cu files. Plain CUDA runtime calls only,
// so the same file builds with nvcc, for the emulator, and in the front-end check.
#pragma once
#include <cstdarg>
#include <cstdio>
#include <string>
#include <vector>
#include "selftest.h"

inline std::string st_fmt(const char* f, ...) {
  char buf[400];
  va_list ap;
  va_start(ap, f);
  vsnprintf(buf, sizeof buf, f, ap);
  va_end(ap);
  return buf;
}

// A device array with host-side fill and read-back.
template <class T>
struct StBuf {
  T* dev = nullptr;
  size_t n;
  explicit StBuf(size_t count, T fill = T()) : n(count) {
    std::vector<T> h(n, fill);
    cudaMalloc((void**)&dev, n * sizeof(T));
    cudaMemcpy(dev, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
  }
  explicit StBuf(const std::vector<T>& h) : n(h.size()) {
    cudaMalloc((void**)&dev, n * sizeof(T));
    cudaMemcpy(dev, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
  }
  StBuf(const StBuf&) = delete;
  ~StBuf() { cudaFree(dev); }
  std::vector<T> host() const {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), dev, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
  }
};

// "" if the launch just made succeeded, else why it did not.
inline std::string st_finish() {
  cudaError_t e = cudaGetLastError();
  if (e == cudaSuccess) e = cudaDeviceSynchronize();
  return e == cudaSuccess ? std::string() : std::string("launch failed: ") + cudaGetErrorString(e);
}

#define ST_REQUIRE(cond, ...) do { if (!(cond)) return st_fmt(__VA_ARGS__); } while (0)
#define ST_LAUNCHED() do { std::string st_e = st_finish(); if (!st_e.empty()) return st_e; } while (0)
// After the table `extern const pmpp_selftest_case pmpp_selftest_cases[] = {...};`
#define ST_END_CASES \
  extern const int pmpp_selftest_count = int(sizeof(pmpp_selftest_cases) / sizeof(pmpp_selftest_cases[0]));
