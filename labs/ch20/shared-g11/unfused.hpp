// Provided for labs 20.1 and 20.2; you do not need to edit it.
//
// 1. Lab 20.1's contract for one row of the causal softmax: the CPU reference and the
//    tolerances (lab 20.1's README derives them). Lab 20.1's tests and both benchmarks use
//    them.
// 2. The two matrix products of the unfused attention pipeline, run head by head with the
//    project's own kernels:
//      S = Q K^T   M3, lab 15.6's reference launchLinear(X = Q, W = K, Y = S, batch = n,
//                  in = d, out = n): the linear layer's weight layout, (out, in) row-major,
//                  is exactly K's (n, d)
//      O = P V     M2, lab 5.3's reference launchGemm(A = P, B = V, C = O, j = n, k = n,
//                  l = d), tile 16
//    unfused.cmake makes the renamed copies and defines PMPP_CH20_HAVE_M3 and
//    PMPP_CH20_HAVE_M2 for the ones it found. Every head has buffers of its own: the
//    provided Metal encoders bind each buffer at offset 0, so there a head cannot be a
//    slice of a larger buffer. The CUDA path does the same, so both backends run the same
//    launches.
// 3. The per-call timer both benchmarks use, and CPU spot checks of each stage's output.
#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "pmpp/harness.hpp"

#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
#if defined(PMPP_CH20_HAVE_M3)
void encodeLinearM3(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> X, id<MTLBuffer> W,
                    id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out);
#endif
#if defined(PMPP_CH20_HAVE_M2)
void encodeGemmM2(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> A, id<MTLBuffer> B,
                  id<MTLBuffer> C, uint32_t j, uint32_t k, uint32_t l, uint32_t tileWidth);
#endif
#else
#if defined(PMPP_CH20_HAVE_M3)
void launchLinearM3(const float* X, const float* W, float* Y, int batch, int in, int out);
#endif
#if defined(PMPP_CH20_HAVE_M2)
void launchGemmM2(float* A, float* B, float* C, int j, int k, int l);
#endif
#endif

// ---------------------------------------------------------------- 1. lab 20.1's contract
namespace causal {

// Tolerances, derived in labs/ch20/lab1_causal_softmax/README.md:
//   P on a visible column:  |got - ref| <= kAtol + kRtol * ref, and each row sums to 1
//                           within kRtol; P on a masked column: exactly 0.
//   L:                      |got - ref| <= kLAtol + kLRtol * |ref|.
const double kRtol = 5e-5;
const double kAtol = 1e-20;
const double kLAtol = 5e-5;
const double kLRtol = 4.8e-7;   // 2^-21

// One row in double: p[c] for c < visible, and L = m + log(s).
inline void refRow(const float* s, int visible, float scale, std::vector<double>& p, double& L) {
  p.assign(visible, 0.0);
  double m = -std::numeric_limits<double>::infinity();
  for (int c = 0; c < visible; ++c) m = std::max(m, (double)scale * s[c]);
  double sum = 0;
  for (int c = 0; c < visible; ++c) sum += (p[c] = std::exp((double)scale * s[c] - m));
  for (int c = 0; c < visible; ++c) p[c] /= sum;
  L = m + std::log(sum);
}

// Checks one row of P (nk values) and its L against the reference. Empty string = correct.
inline std::string checkRow(const float* s, const float* P, float Lgot, int visible, int nk, float scale) {
  std::vector<double> ref;
  double Lref = 0;
  refRow(s, visible, scale, ref, Lref);
  char b[300];
  double sum = 0;
  for (int c = 0; c < nk; ++c) {
    if (c >= visible) {
      if (!(P[c] == 0.0f)) {
        std::snprintf(b, sizeof b, "column %d is masked (the row sees columns 0 to %d) but P = %.9g there; it must be "
                      "exactly 0", c, visible - 1, P[c]);
        return b;
      }
      continue;
    }
    sum += P[c];
    if (!(std::fabs((double)P[c] - ref[c]) <= kAtol + kRtol * ref[c])) {
      std::snprintf(b, sizeof b, "column %d: P = %.9g, expected %.9g", c, P[c], ref[c]);
      std::string e = b;
      if (std::isnan(P[c])) e += " (NaN: never written, or 0/0 or inf/inf in the kernel)";
      else if (std::isinf(P[c])) e += " (infinity: did an exponent overflow?)";
      return e;
    }
  }
  if (!(std::fabs(sum - 1.0) <= kRtol)) {
    std::snprintf(b, sizeof b, "the visible columns sum to %.9g, not 1", sum);
    return b;
  }
  if (!(std::fabs((double)Lgot - Lref) <= kLAtol + kLRtol * std::fabs(Lref))) {
    std::snprintf(b, sizeof b, "L = %.9g, expected %.9g (the logsumexp m + log(sum))", Lgot, Lref);
    std::string e = b;
    if (std::isnan(Lgot)) e += " (NaN: L was never written?)";
    return e;
  }
  return "";
}

}  // namespace causal

// ---------------------------------------------------------------- 2. the unfused pipeline
namespace unfused {
using pmpp::Buf;
const int kD = 128;   // head dimension, the book's d

struct Head { Buf *Q, *K, *V, *S, *P, *L, *O; };

// One attention problem: `heads` heads of n tokens, each with buffers of its own.
// Q, K, V are uniform in [-1, 1].
struct Problem {
  int heads, n;
  std::vector<std::vector<float>> q, k, v;
  std::vector<Head> h;
  Problem(int heads_, int n_, uint32_t seed) : heads(heads_), n(n_) {
    const size_t nd = (size_t)n * kD, nn = (size_t)n * n;
    for (int i = 0; i < heads; ++i) {
      q.push_back(pmpp::random_floats(nd, seed + 3 * i));
      k.push_back(pmpp::random_floats(nd, seed + 3 * i + 1));
      v.push_back(pmpp::random_floats(nd, seed + 3 * i + 2));
      Head x;
      x.Q = pmpp::alloc(nd * 4); x.K = pmpp::alloc(nd * 4); x.V = pmpp::alloc(nd * 4);
      x.S = pmpp::alloc(nn * 4); x.P = pmpp::alloc(nn * 4); x.L = pmpp::alloc((size_t)n * 4);
      x.O = pmpp::alloc(nd * 4);
      pmpp::upload(x.Q, q[i].data(), nd * 4);
      pmpp::upload(x.K, k[i].data(), nd * 4);
      pmpp::upload(x.V, v[i].data(), nd * 4);
      h.push_back(x);
    }
  }
  Problem(const Problem&) = delete;
  ~Problem() {
    for (Head& x : h) {
      pmpp::release(x.Q); pmpp::release(x.K); pmpp::release(x.V); pmpp::release(x.S);
      pmpp::release(x.P); pmpp::release(x.L); pmpp::release(x.O);
    }
  }
  // Bytes S and P take for all heads: what flash attention never allocates.
  static double scoreBytes(int heads, int n) { return 2.0 * heads * (double)n * n * 4; }
  // Everything one problem allocates on the device.
  static double totalBytes(int heads, int n) {
    return scoreBytes(heads, n) + heads * (4.0 * n * kD + n) * 4;
  }
};

#if defined(PMPP_BACKEND_METAL)
using pmpp::metal::buffer;
// Why pipeline `p` cannot run `threads` threads per threadgroup here, or "".
inline std::string pipelineProblem(id<MTLComputePipelineState> p, NSUInteger threads, const std::string& what) {
  char b[300];
  if (!p) return what + " did not compile: " + pmpp::take_error();
  if (p.maxTotalThreadsPerThreadgroup < threads) {
    std::snprintf(b, sizeof b, "%s: this pipeline allows %lu threads per threadgroup, the kernel needs %lu",
                  what.c_str(), (unsigned long)p.maxTotalThreadsPerThreadgroup, (unsigned long)threads);
    return b;
  }
  if (p.staticThreadgroupMemoryLength > pmpp::metal::device().maxThreadgroupMemoryLength) {
    std::snprintf(b, sizeof b, "%s: its tiles take %lu bytes of threadgroup memory, this device has %lu",
                  what.c_str(), (unsigned long)p.staticThreadgroupMemoryLength,
                  (unsigned long)pmpp::metal::device().maxThreadgroupMemoryLength);
    return b;
  }
  return "";
}
#if defined(PMPP_CH20_HAVE_M3)
// Lab 15.6 lays the threadgroup out for the SIMD-group width it is compiled with; the
// pipeline reports the width (ch15-facts C1). As lab 15.6's driver does: 64 if reported,
// else 32.
inline id<MTLComputePipelineState> m3Pipeline() {
  auto at = [](int w) {
    return pmpp::metal::pipeline(PMPP_CH20_M3_METAL, "linear_m3", {{"SIMD_WIDTH", std::to_string(w)}});
  };
  id<MTLComputePipelineState> p = at(32);
  return (p && p.threadExecutionWidth == 64) ? at(64) : p;
}
#endif
#if defined(PMPP_CH20_HAVE_M2)
inline id<MTLComputePipelineState> m2Pipeline() {
  return pmpp::metal::pipeline(PMPP_CH20_M2_METAL, "gemm", {{"TILE_WIDTH", "16"}});
}
#endif
#endif

// Why the S stage cannot run here, or "".
inline std::string sProblem() {
#if !defined(PMPP_CH20_HAVE_M3)
  return "lab 15.6's solution (M3) was not found when the lab was configured";
#elif defined(PMPP_BACKEND_METAL)
  return pipelineProblem(m3Pipeline(), 256, "lab 15.6's linear_m3 kernel");
#else
  return "";
#endif
}

// Why the O stage cannot run here, or "".
inline std::string oProblem() {
#if !defined(PMPP_CH20_HAVE_M2)
  return "lab 5.3's solution (M2) was not found when the lab was configured";
#elif defined(PMPP_BACKEND_METAL)
  return pipelineProblem(m2Pipeline(), 256, "lab 5.3's gemm kernel");
#else
  return "";
#endif
}

#if defined(PMPP_BACKEND_METAL)
// Encodes S_h = Q_h K_h^T for every head (one pass of the stage).
inline void encodeS(pmpp::metal::Pass* pass, Problem& pr) {
#if defined(PMPP_CH20_HAVE_M3)
  id<MTLComputePipelineState> p = m3Pipeline();
  for (Head& x : pr.h)
    encodeLinearM3(pass->encoder, p, buffer(x.Q), buffer(x.K), buffer(x.S), (uint32_t)pr.n, kD, (uint32_t)pr.n);
#else
  (void)pass; (void)pr;
#endif
}
// Encodes O_h = P_h V_h for every head.
inline void encodeO(pmpp::metal::Pass* pass, Problem& pr) {
#if defined(PMPP_CH20_HAVE_M2)
  id<MTLComputePipelineState> p = m2Pipeline();
  for (Head& x : pr.h)
    encodeGemmM2(pass->encoder, p, buffer(x.P), buffer(x.V), buffer(x.O), (uint32_t)pr.n, (uint32_t)pr.n, kD, 16);
#else
  (void)pass; (void)pr;
#endif
}
#else
inline float* f(Buf* b) { return (float*)pmpp::raw(b); }
inline void encodeS(void*, Problem& pr) {
#if defined(PMPP_CH20_HAVE_M3)
  for (Head& x : pr.h) launchLinearM3(f(x.Q), f(x.K), f(x.S), pr.n, kD, pr.n);
#else
  (void)pr;
#endif
}
inline void encodeO(void*, Problem& pr) {
#if defined(PMPP_CH20_HAVE_M2)
  for (Head& x : pr.h) launchGemmM2(f(x.P), f(x.V), f(x.O), pr.n, pr.n, kD);
#else
  (void)pr;
#endif
}
#endif

// What pmpp::run hands its callback: a pmpp::metal::Pass on Metal, nothing elsewhere.
#if defined(PMPP_BACKEND_METAL)
inline pmpp::metal::Pass* pass(void* v) { return (pmpp::metal::Pass*)v; }
#else
inline void* pass(void* v) { return v; }
#endif

// Runs `calls` passes of a stage in one timed run. "" when it ran.
inline std::string runS(Problem& pr, int calls, pmpp::Timing* tm) {
  std::string why = sProblem();
  if (!why.empty()) return why;
  *tm = pmpp::run([&](void* v) {
    for (int c = 0; c < calls; ++c) encodeS(pass(v), pr);
  });
  return "";
}
inline std::string runO(Problem& pr, int calls, pmpp::Timing* tm) {
  std::string why = oProblem();
  if (!why.empty()) return why;
  *tm = pmpp::run([&](void* v) {
    for (int c = 0; c < calls; ++c) encodeO(pass(v), pr);
  });
  return "";
}

// ---------------------------------------------------------------- 3. timing and checks
// Time per call: best of `reps` timed runs after one untimed run; each run repeats the call
// `calls` times back to back, with `calls` doubled until one run lasts at least 2 ms, so
// that short calls are not lost in the cost of submitting the work. (Lab 10.3's timer.)
inline double perCall(int reps, const std::function<std::string(int, pmpp::Timing*)>& run, int* calls,
                      std::string* err) {
  *calls = 1;
  pmpp::Timing tm{0, 0};
  for (;;) {
    *err = run(*calls, &tm);
    if (err->empty()) *err = pmpp::take_error();
    if (!err->empty()) return -1;
    if (tm.best() >= 0.002 || *calls >= 4096) break;
    *calls *= 2;
  }
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    *err = run(*calls, &tm);
    if (err->empty()) *err = pmpp::take_error();
    if (!err->empty()) return -1;
    best = std::min(best, tm.best());
  }
  return best / *calls;
}

// The rows of each head the spot checks look at.
inline std::vector<int> spotRows(int n) {
  std::vector<int> r = {0, 1, n / 3, n / 2, n - 2, n - 1};
  std::vector<int> out;
  for (int x : r)
    if (x >= 0 && x < n && std::find(out.begin(), out.end(), x) == out.end()) out.push_back(x);
  return out;
}

inline std::vector<float> fetch(Buf* b, size_t count) {
  std::vector<float> v(count);
  pmpp::download(b, v.data(), count * 4);
  return v;
}

// S rows of every head against Q K^T in double, with cpu_gemm's rule:
// |got - ref| <= 1e-4 (1 + sum over k of |q k|).
inline std::string checkS(Problem& pr) {
  const int n = pr.n;
  for (int i = 0; i < pr.heads; ++i) {
    std::vector<float> S = fetch(pr.h[i].S, (size_t)n * n);
    for (int r : spotRows(n))
      for (int c = 0; c < n; ++c) {
        double acc = 0, mag = 0;
        for (int x = 0; x < kD; ++x) {
          const double p = (double)pr.q[i][(size_t)r * kD + x] * pr.k[i][(size_t)c * kD + x];
          acc += p; mag += std::fabs(p);
        }
        const float got = S[(size_t)r * n + c];
        if (!(std::fabs(got - acc) <= 1e-4 * (1 + mag)))
          return "S = Q K^T, head " + std::to_string(i) + ", row " + std::to_string(r) + ", column " +
                 std::to_string(c) + ": got " + std::to_string(got) + ", expected " + std::to_string(acc);
      }
  }
  return "";
}

// P and L rows of every head against the causal softmax (square, nq = nk = n) of the S the
// device produced, with lab 20.1's tolerances.
inline std::string checkP(Problem& pr, float scale) {
  const int n = pr.n;
  for (int i = 0; i < pr.heads; ++i) {
    std::vector<float> S = fetch(pr.h[i].S, (size_t)n * n), P = fetch(pr.h[i].P, (size_t)n * n),
                       L = fetch(pr.h[i].L, (size_t)n);
    for (int r : spotRows(n)) {
      std::string e = causal::checkRow(&S[(size_t)r * n], &P[(size_t)r * n], L[r], r + 1, n, scale);
      if (!e.empty()) return "softmax, head " + std::to_string(i) + ", row " + std::to_string(r) + ": " + e;
    }
  }
  return "";
}

// O rows of every head against (the P the device produced) x V in double:
// |got - ref| <= 1e-4 (1 + sum over c of |p v|).
inline std::string checkO(Problem& pr) {
  const int n = pr.n;
  for (int i = 0; i < pr.heads; ++i) {
    std::vector<float> P = fetch(pr.h[i].P, (size_t)n * n), O = fetch(pr.h[i].O, (size_t)n * kD);
    for (int r : spotRows(n))
      for (int col = 0; col < kD; ++col) {
        double acc = 0, mag = 0;
        for (int c = 0; c < n; ++c) {
          const double p = (double)P[(size_t)r * n + c] * pr.v[i][(size_t)c * kD + col];
          acc += p; mag += std::fabs(p);
        }
        const float got = O[(size_t)r * kD + col];
        if (!(std::fabs(got - acc) <= 1e-4 * (1 + mag)))
          return "O = P V, head " + std::to_string(i) + ", row " + std::to_string(r) + ", column " +
                 std::to_string(col) + ": got " + std::to_string(got) + ", expected " + std::to_string(acc);
      }
  }
  return "";
}

}  // namespace unfused
