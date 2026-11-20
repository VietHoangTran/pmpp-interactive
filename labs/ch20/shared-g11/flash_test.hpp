// Tests of the causal flash attention forward kernel, shared by labs 20.2 and 20.3 (their
// main.cpp include this file). You do not need to edit it.
//
// The kernel's contract (labs/ch20/lab2_flash/README.md): Q, K, V, O are [batch x heads x n
// x 128] float32, row-major; L is [batch x heads x n]; row r sees the columns c <= r;
// O = softmax(scale * Q K^T, masked) V and L = m + log(D), the logsumexp. The reference is
// computed in double on the CPU; the tolerance of every row is derived from that row's data
// (README.md, "Tolerances").
#pragma once
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "pmpp/harness.hpp"

#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif

// ---- entry points implemented in <variant>/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
id<MTLComputePipelineState> buildFlashPipeline(id<MTLDevice> device, const char* sourcePath, uint32_t* bc,
                                               std::string* why);
bool encodeFlashAttention(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> Q, id<MTLBuffer> K,
                          id<MTLBuffer> V, id<MTLBuffer> O, id<MTLBuffer> L, uint32_t batch, uint32_t heads,
                          uint32_t n, float scale);
#else
void launchFlashAttention(const float* Q, const float* K, const float* V, float* O, float* L, int batch, int heads,
                          int n, float scale);
#endif

namespace flash {
using pmpp::Buf;
const int kD = 128;           // HEAD_DIM
const int kThreads = 512;     // BLOCK_SIZE

// ---------------------------------------------------------------- the kernel on this backend
// The B_c this build runs, and why the kernel cannot run here ("" when it can).
struct Kernel {
  int bc = 0;
  std::string why;
#if defined(PMPP_BACKEND_METAL)
  id<MTLComputePipelineState> pso = nil;
#endif
};

#if defined(PMPP_BACKEND_METAL)
// Built once: your launch.mm chooses B_c, compiles kernel.metal and checks the pipeline.
inline Kernel& kernel() {
  static Kernel k = [] {
    Kernel x;
    uint32_t bc = 0;
    x.pso = buildFlashPipeline(pmpp::metal::device(), PMPP_METAL_SOURCE, &bc, &x.why);
    if (!x.pso && x.why.empty()) x.why = "buildFlashPipeline returned nil without saying why";
    if (x.pso && !(bc == 8 || bc == 16 || bc == 32)) x.why = "buildFlashPipeline reported B_c = " + std::to_string(bc);
    x.bc = (int)bc;
    return x;
  }();
  return k;
}
inline std::string run(Buf* Q, Buf* K, Buf* V, Buf* O, Buf* L, int batch, int heads, int n, float scale, int repeat,
                       pmpp::Timing* tm) {
  Kernel& k = kernel();
  if (!k.why.empty()) return k.why;
  bool ok = true;
  using pmpp::metal::buffer;
  pmpp::Timing t = pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    for (int i = 0; i < repeat && ok; ++i)
      ok = encodeFlashAttention(enc, k.pso, buffer(Q), buffer(K), buffer(V), buffer(O), buffer(L), (uint32_t)batch,
                                (uint32_t)heads, (uint32_t)n, scale);
  });
  if (tm) *tm = t;
  return ok ? "" : "your encodeFlashAttention refused to dispatch";
}
inline double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
inline Kernel& kernel() {
  static Kernel k = [] {
    Kernel x;
    x.bc = TILE_WIDTH;
    const int most = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
    if (most < kThreads) x.why = "this device runs at most " + std::to_string(most) + " threads per block; the kernel needs 512";
    return x;
  }();
  return k;
}
inline std::string run(Buf* Q, Buf* K, Buf* V, Buf* O, Buf* L, int batch, int heads, int n, float scale, int repeat,
                       pmpp::Timing* tm) {
  Kernel& k = kernel();
  if (!k.why.empty()) return k.why;
  auto f = [](Buf* b) { return (float*)pmpp::raw(b); };
  pmpp::Timing t = pmpp::run([&](void*) {
    for (int i = 0; i < repeat; ++i) launchFlashAttention(f(Q), f(K), f(V), f(O), f(L), batch, heads, n, scale);
  });
  if (tm) *tm = t;
  return "";
}
inline double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

// ---------------------------------------------------------------- test data
enum Recipe { Uniform, LateMax, Offset, DistinctV, kRecipes };
inline const char* recipeName(int r) {
  switch (r) {
    case Uniform: return "Q, K, V uniform in [-1, 1]";
    case LateMax: return "late max: scores rise along every row, so the running max rises in every tile";
    case Offset: return "every score offset by +90";
    default: return "V with a different mean in every column";
  }
}

// One head's Q, K, V (n x 128 each). With u uniform in [-1, 1]:
//   Uniform    q, k, v = u
//   LateMax    q = 0.5u, k = 0.5u, except q[127] = 1 and k[127] = 0.45 c for token c: the score
//              of column c grows by 0.45 * scale (0.04) per column, about 1.3 per tile of 32
//   Offset     q = u, k = u, except q[127] = k[127] = sqrt(90 / scale): every score is 90 plus
//              noise. exp(90) is beyond float's range, so the max subtraction is needed.
//   DistinctV  q, k = u; v[c][x] = (x - 63.5) / 64 + 0.05u: each column of O has its own level
inline void fillHead(int recipe, int n, float scale, uint32_t seed, float* q, float* k, float* v) {
  const size_t nd = (size_t)n * kD;
  std::vector<float> uq = pmpp::random_floats(nd, seed), uk = pmpp::random_floats(nd, seed + 1),
                     uv = pmpp::random_floats(nd, seed + 2);
  for (size_t e = 0; e < nd; ++e) { q[e] = uq[e]; k[e] = uk[e]; v[e] = uv[e]; }
  if (recipe == LateMax) {
    for (size_t e = 0; e < nd; ++e) { q[e] *= 0.5f; k[e] *= 0.5f; }
    for (int c = 0; c < n; ++c) { q[(size_t)c * kD + kD - 1] = 1.0f; k[(size_t)c * kD + kD - 1] = 0.45f * c; }
  } else if (recipe == Offset) {
    const float a = std::sqrt(90.0f / scale);
    for (int c = 0; c < n; ++c) { q[(size_t)c * kD + kD - 1] = a; k[(size_t)c * kD + kD - 1] = a; }
  } else if (recipe == DistinctV) {
    for (int c = 0; c < n; ++c)
      for (int x = 0; x < kD; ++x) v[(size_t)c * kD + x] = (x - 63.5f) / 64.0f + 0.05f * uv[(size_t)c * kD + x];
  }
}

// ---------------------------------------------------------------- the reference, and its tolerance
// Per row, in double: O and L, and the bounds README.md derives (u = 2^-24, one rounding):
//   A  = max over visible c of scale * sum_x |q_x k_cx|   (size of the dot products)
//   X  = max over visible c of m - s_c                    (range of the exponents)
//   T  = number of tiles of B_c columns the row spans
//   e  = 2 (d + 2) u A + u X + 2u (3 + 2X) + 2u (3T + 2X) + (B_c + 2T + 8) u
//        (scores; s - m; exp of P; the rescale factors, one per tile; the sums)
//   |O_got - O_ref| <= 2 e * sum_c p_c |v_c| + 1e-7  for each column of O
//   |L_got - L_ref| <= 2 e + 1e-5 + 2^-22 |L_ref|
struct RowRef {
  std::vector<double> o, omag;   // kD values each
  double L = 0, e = 0;
};

inline void refRow(const float* q, const float* k, const float* v, int r, int n, float scale, int bc, RowRef& out) {
  const double u = std::ldexp(1.0, -24);
  const int vis = std::min(r + 1, n);
  std::vector<double> s(vis);
  double m = -std::numeric_limits<double>::infinity(), A = 0;
  const float* qr = q + (size_t)r * kD;
  for (int c = 0; c < vis; ++c) {
    const float* kc = k + (size_t)c * kD;
    double acc = 0, mag = 0;
    for (int x = 0; x < kD; ++x) { const double p = (double)qr[x] * kc[x]; acc += p; mag += std::fabs(p); }
    s[c] = (double)scale * acc;
    A = std::max(A, (double)scale * mag);
    m = std::max(m, s[c]);
  }
  double D = 0, X = 0;
  out.o.assign(kD, 0.0);
  out.omag.assign(kD, 0.0);
  for (int c = 0; c < vis; ++c) {
    const double p = std::exp(s[c] - m);
    X = std::max(X, m - s[c]);
    D += p;
    const float* vc = v + (size_t)c * kD;
    for (int x = 0; x < kD; ++x) { out.o[x] += p * vc[x]; out.omag[x] += p * std::fabs((double)vc[x]); }
  }
  for (int x = 0; x < kD; ++x) { out.o[x] /= D; out.omag[x] /= D; }
  out.L = m + std::log(D);
  const int T = (vis + bc - 1) / bc;
  out.e = 2 * (kD + 2) * u * A + u * X + 2 * u * (3 + 2 * X) + 2 * u * (3 * T + 2 * X) + (bc + 2 * T + 8) * u;
}

// ---------------------------------------------------------------- one launch, checked
const int kGuard = 64;                  // canaries past the end of O and of L
const float kGuardF = -12345.5f;
const float kNaN = std::numeric_limits<float>::quiet_NaN();

inline std::string fmt(const char* f, double a, double b = 0, double c = 0) {
  char buf[240];
  std::snprintf(buf, sizeof buf, f, a, b, c);
  return buf;
}

// Runs one launch of `batch` x `heads` heads of n tokens; head g follows recipe
// recipes[g % recipes.size()]. Checks O and L of every row of every head, and the canaries.
inline std::string checkCase(int batch, int heads, int n, const std::vector<int>& recipes, uint32_t seed) {
  Kernel& kn = kernel();
  if (!kn.why.empty()) return kn.why;
  const float scale = 1.0f / std::sqrt((float)kD);
  const int H = batch * heads;
  const size_t nd = (size_t)n * kD, total = (size_t)H * nd, rows = (size_t)H * n;
  std::vector<float> q(total), k(total), v(total);
  for (int g = 0; g < H; ++g)
    fillHead(recipes[g % recipes.size()], n, scale, seed + 17u * (uint32_t)g, &q[g * nd], &k[g * nd], &v[g * nd]);

  // The only device buffers of the test: O(n d) each, none n x n (checked below).
  size_t largest = 0;
  auto alloc = [&](size_t bytes) { largest = std::max(largest, bytes); return pmpp::alloc(bytes); };
  Buf* Q = alloc(total * 4);
  Buf* K = alloc(total * 4);
  Buf* V = alloc(total * 4);
  Buf* O = alloc((total + kGuard) * 4);
  Buf* L = alloc((rows + kGuard) * 4);
  pmpp::upload(Q, q.data(), total * 4);
  pmpp::upload(K, k.data(), total * 4);
  pmpp::upload(V, v.data(), total * 4);
  std::vector<float> o(total + kGuard, kGuardF), l(rows + kGuard, kGuardF);
  std::fill(o.begin(), o.begin() + total, kNaN);   // an element nobody writes shows as NaN
  std::fill(l.begin(), l.begin() + rows, kNaN);
  pmpp::upload(O, o.data(), o.size() * 4);
  pmpp::upload(L, l.data(), l.size() * 4);

  std::string err = run(Q, K, V, O, L, batch, heads, n, scale, 1, nullptr);
  if (err.empty()) {
    pmpp::download(O, o.data(), o.size() * 4);
    pmpp::download(L, l.data(), l.size() * 4);
    RowRef ref;
    for (int g = 0; g < H && err.empty(); ++g) {
      const float *qg = &q[g * nd], *kg = &k[g * nd], *vg = &v[g * nd];
      for (int r = 0; r < n && err.empty(); ++r) {
        refRow(qg, kg, vg, r, n, scale, kn.bc, ref);
        const float* got = &o[g * nd + (size_t)r * kD];
        const std::string where = "sequence " + std::to_string(g / heads) + ", head " + std::to_string(g % heads) +
                                  ", row " + std::to_string(r) + " (" + recipeName(recipes[g % recipes.size()]) + ")";
        for (int x = 0; x < kD; ++x) {
          const double tol = 2 * ref.e * ref.omag[x] + 1e-7;
          if (std::fabs((double)got[x] - ref.o[x]) <= tol) continue;   // false for NaN too
          err = where + fmt(": O[%d] = %.9g, expected %.9g", x, got[x], ref.o[x]) +
                fmt(" (tolerance %.3g)", tol);
          if (std::isnan(got[x])) err += "; NaN: never written, or 0/0 or inf/inf in the kernel";
          else if (std::isinf(got[x])) err += "; infinity: did an exponent overflow?";
          break;
        }
        if (!err.empty()) break;
        const double lgot = l[(size_t)g * n + r];
        const double ltol = 2 * ref.e + 1e-5 + std::ldexp(1.0, -22) * std::fabs(ref.L);
        if (!(std::fabs(lgot - ref.L) <= ltol))
          err = where + fmt(": L = %.9g, expected %.9g (the logsumexp m + log(D); tolerance %.3g)", lgot, ref.L, ltol);
      }
    }
    for (size_t e = total; e < total + kGuard && err.empty(); ++e)
      if (o[e] != kGuardF) err = "wrote O[" + std::to_string(e) + "], past the end of O";
    for (size_t e = rows; e < rows + kGuard && err.empty(); ++e)
      if (l[e] != kGuardF) err = "wrote L[" + std::to_string(e) + "], past the end of L";
    // The driver's own promise: no n x n buffer (meaningful once n x n exceeds n x d).
    if (err.empty() && n > 2 * kD && largest >= (size_t)H * n * n * 4)
      err = "the driver allocated an n x n buffer; flash attention must not need one";
  }
  pmpp::release(Q); pmpp::release(K); pmpp::release(V); pmpp::release(O); pmpp::release(L);
  return err;
}

// ---------------------------------------------------------------- the test list
struct Case { int batch, heads, n; std::vector<int> recipes; };   // empty recipes: all of them

// On a real device n runs through multiples and non-multiples of B_r = 32 and of B_c, with
// every recipe. The emulators are slow for this kernel (each thread meets about 280 warp
// rendezvous per tile, and emu runs each case three times): `emu` gets an explicit short
// list that still has one token, a second panel that needs two tiles, two full panels and
// tiles (n a multiple of both), three panels with a partial last tile, two heads in one
// launch, and the late-max, +90 and distinct-V data.
inline std::vector<Case> cases() {
  const std::string b = pmpp::backend();
  std::vector<Case> c;
  if (b == "race") {
    for (int n : {33, 40}) c.push_back({1, 1, n, {}});
  } else if (b == "emu") {
    c = {{1, 1, 1, {Offset}}, {1, 2, 33, {LateMax, DistinctV}}, {1, 1, 64, {Uniform}}, {1, 1, 70, {LateMax}}};
  } else {
    for (int n : {1, 2, 31, 32, 33, 63, 64, 65, 100, 257, 1000, 2048})
      for (auto bh : {std::pair<int, int>{1, 1}, std::pair<int, int>{2, 3}}) c.push_back({bh.first, bh.second, n, {}});
  }
  return c;
}

inline int runTests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  Kernel& kn = kernel();
  if (!kn.why.empty()) {
    t.check("flash attention: kernel usable", [&] { return kn.why; });
    return t.finish();
  }
  std::printf("B_c = %d (%s)\n", kn.bc,
#if defined(PMPP_BACKEND_METAL)
              "chosen by your launch.mm for this device"
#else
              "TILE_WIDTH of this build"
#endif
  );
  for (const Case& cs : cases()) {
    const int H = cs.batch * cs.heads, n = cs.n;
    std::string note;
    if (n % 32 != 0 || n % kn.bc != 0) note = "  (not a multiple of B_r = 32 or B_c = " + std::to_string(kn.bc) + ")";
    std::string name = "flash attention, batch " + std::to_string(cs.batch) + ", heads " + std::to_string(cs.heads) +
                       ", n " + std::to_string(n) + note;
    if (!cs.recipes.empty()) {
      name += "; data:";
      for (size_t i = 0; i < cs.recipes.size(); ++i)
        name += std::string(i ? "," : "") + (cs.recipes[i] == Uniform ? " uniform" : cs.recipes[i] == LateMax ? " late max"
                                             : cs.recipes[i] == Offset ? " +90" : " distinct V");
    }
    t.check(name, [&]() -> std::string {
      std::vector<int> all = cs.recipes;
      if (all.empty())
        for (int r = 0; r < kRecipes; ++r) all.push_back(r);
      // Every recipe of the case once: the heads of one launch take them in turn, with as
      // many launches as needed.
      for (size_t start = 0; start < all.size(); start += H) {
        std::vector<int> rc;
        for (int g = 0; g < H; ++g) rc.push_back(all[(start + g) % all.size()]);
        std::string e = checkCase(cs.batch, cs.heads, n, rc, 4000u + 97u * (uint32_t)n + (uint32_t)start);
        if (!e.empty()) return e;
      }
      return "";
    });
  }
  return t.finish();
}

}  // namespace flash
