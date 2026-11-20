// Shared by the drivers of labs 15.6 and 15.7: the linear layer's test problem.
//   Y[batch x out] = X[batch x in] * W^T,  W stored (out, in) row-major.
// You do not need to edit this file.
//
// What a test case does:
//   - X and W are allocated with exactly the floats they hold (one float when they hold
//     none), so on the emulator a read past either is an AddressSanitizer error naming
//     the line.
//   - Y is followed, in the same allocation, by a guard band of floats holding a sentinel.
//     A kernel that writes past the last row of Y changes the sentinel, on every backend.
//     On `emu` the band is short (256 floats), so a write that runs further also passes
//     the end of the allocation and AddressSanitizer names the line; elsewhere it is a
//     whole block row of Y long, so a GPU's stray writes land in it rather than in
//     someone else's memory.
//   - Y starts as NaN before every launch, so an element the kernel never writes, or one
//     left over from an earlier launch, cannot pass.
//   - The result is compared with a CPU reference computed in double, with a tolerance
//     scaled by sum_k |x w| (labs 5.2 and 6.4 use the same rule). The reference,
//     cpuLinear, is defined in each lab's main.cpp, where the project's conventions keep
//     it; this file only declares it.
//   - One case has in = 0: an empty sum, so Y must come out all zeros. Its driver runs it
//     under a 30-second watchdog (watchdog.hpp), so a tile loop whose bound
//     numTiles - 1 wraps around fails with that reason instead of running for hours.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pmpp/harness.hpp"
#include "watchdog.hpp"

namespace lin {

struct Shape { int batch, in, out; const char* why; };

// Shapes the tests run. The emulators get small ones; `race` (ThreadSanitizer, slow) fewer.
inline std::vector<Shape> testShapes() {
  const std::string b = pmpp::backend();
  if (b == "race")
    return {
        {128, 24, 128, "one full block, three tile pairs"},
        {130, 9, 131, "partial tiles in every direction"},
        {1, 40, 260, "one token; Y's rows end inside the block"},
    };
  std::vector<Shape> s = {
      {1, 1, 1, "single element"},
      {1, 40, 260, "one token; Y's rows end inside the block"},
      {3, 3, 3, "smaller than one tile in every direction"},
      {128, 8, 128, "one full block, one tile pair"},
      {128, 24, 128, "one full block, three tile pairs"},
      {130, 9, 131, "partial tiles in every direction"},
      {5, 1, 9, "one input feature"},
      {17, 33, 9, "nothing is a multiple of anything"},
      {256, 32, 256, "four full blocks"},
      {3, 0, 5, "in = 0: an empty sum, so Y is all zeros"},
  };
  if (!pmpp::is_emu()) {
    s.push_back({4, 300, 1000, "a small batch into a wide layer"});
    s.push_back({257, 129, 511, "large and awkward"});
    s.push_back({1024, 1024, 1024, "large and square"});
    s.push_back({8, 4096, 4096, "a few tokens through a 4096-wide layer"});
  }
  return s;
}

// The CPU reference, defined in the lab's main.cpp:
// Y[b][o] = sum_k X[b][k] * W[o][k], accumulated in double; mag = sum_k |X[b][k] * W[o][k]|.
void cpuLinear(const std::vector<float>& x, const std::vector<float>& w, int batch, int in, int out,
               std::vector<float>& y, std::vector<float>& mag);

// Why an in = 0 case that never finishes failed (the watchdog's message).
inline const char* kEmptySumWhy =
    "with in = 0 there are no tiles; does numTiles - 1 wrap around (an unsigned numTiles)?";

inline const float kSentinel = -77777.0f;   // no correct kernel writes anything past Y

// A problem instance on the device. Inputs and reference depend only on the shape, so
// the reference is computed once per shape however many kernels and widths use it.
struct Case {
  int batch, in, out;
  size_t guard;
  std::vector<float> x, w;
  const std::vector<float>* ref = nullptr;
  const std::vector<float>* mag = nullptr;
  pmpp::Buf *X = nullptr, *W = nullptr, *Y = nullptr;

  Case(int b, int i, int o) : batch(b), in(i), out(o) {
    guard = std::string(pmpp::backend()) == "emu" ? 256 : (size_t)128 * out + 256;
    const uint32_t seed = 4000u + (uint32_t)b * 131u + (uint32_t)i * 17u + (uint32_t)o;
    x = pmpp::random_floats((size_t)batch * in, seed);
    w = pmpp::random_floats((size_t)out * in, seed + 1);
    // With in = 0, X and W hold nothing. They still get one float each: CUDA documents no
    // behaviour for cudaMalloc(0), and the kernel must not read them anyway.
    X = pmpp::alloc(std::max<size_t>(x.size(), 1) * 4);
    W = pmpp::alloc(std::max<size_t>(w.size(), 1) * 4);
    Y = pmpp::alloc(((size_t)batch * out + guard) * 4);
    if (!x.empty()) pmpp::upload(X, x.data(), x.size() * 4);
    if (!w.empty()) pmpp::upload(W, w.data(), w.size() * 4);
  }
  explicit Case(const Shape& s) : Case(s.batch, s.in, s.out) {}
  Case(const Case&) = delete;
  ~Case() { pmpp::release(X); pmpp::release(W); pmpp::release(Y); }

  double flop() const { return 2.0 * batch * (double)in * out; }

  // Y = NaN, guard band = sentinel. Before every launch.
  void refill() {
    std::vector<float> v((size_t)batch * out + guard, std::numeric_limits<float>::quiet_NaN());
    std::fill(v.begin() + (size_t)batch * out, v.end(), kSentinel);
    pmpp::upload(Y, v.data(), v.size() * 4);
  }

  void reference() {
    static std::map<std::tuple<int, int, int>, std::pair<std::vector<float>, std::vector<float>>> cache;
    auto& e = cache[std::make_tuple(batch, in, out)];
    if (e.first.empty()) ::lin::cpuLinear(x, w, batch, in, out, e.first, e.second);
    ref = &e.first;
    mag = &e.second;
  }

  // Empty if Y matches the reference and the guard band is untouched.
  std::string check() {
    if (!ref) reference();
    const size_t n = (size_t)batch * out;
    std::vector<float> got(n + guard);
    pmpp::download(Y, got.data(), got.size() * 4);
    std::string err = pmpp::describe(pmpp::compare(got.data(), ref->data(), mag->data(), n), out);
    size_t changed = 0, first = 0;
    for (size_t i = n; i < got.size(); ++i) {
      if (std::memcmp(&got[i], &kSentinel, 4) != 0) {
        if (changed == 0) first = i - n;
        ++changed;
      }
    }
    if (changed) {
      char b[300];
      std::snprintf(b, sizeof b,
                    "%s%zu of the %zu floats after Y were overwritten; the first, %zu floats past the end, is "
                    "where row %zu, column %zu of Y would be if Y had more rows",
                    err.empty() ? "" : "; ", changed, guard, first, (size_t)batch + first / out, first % out);
      err += b;
    }
    return err;
  }
};

// "width 32: batch 1, in 40, out 260  (why)"
inline std::string caseName(const std::string& who, const Shape& s) {
  char b[220];
  std::snprintf(b, sizeof b, "%s: batch %d, in %d, out %d  (%s)", who.c_str(), s.batch, s.in, s.out, s.why);
  return b;
}

}  // namespace lin
