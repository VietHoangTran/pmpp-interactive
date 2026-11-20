// Lab X1.2 driver (milestone M5, the top-p and min-p stage): tests part A (your composition of
// labs 14.3, 10.3 and 11.6 with your min-p count) and part B (your fused kernel) on the same
// inputs against a CPU reference in double, tests the provided earlier labs, and benchmarks
// the composed chain against the fused kernel. You do not need to edit this file; your work
// goes in work/.
//
// Lab X1.4 (the filter hunt) includes this file with X1_FILTER_HUNT defined. It then runs part
// B's checks only, at smaller sizes and, on a GPU, ten times each; it has no benchmark.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <numeric>

#include "pmpp/harness.hpp"
using pmpp::Buf;

#if defined(X1_FILTER_HUNT)
#define X1_PART_A 0
#else
#define X1_PART_A 1
#endif

static const int kBlock = 256;    // threads per block of every kernel here (BLOCK_DIM, kBlock in launch.mm)
static const int kChunk = 1024;   // the suggested design's chunk: used only to word a hint

// ---------------------------------------------------------------- the entry points
// Part A's workspace (README.md). CUDA, HIP and the emulators: one allocation, laid out as
// sort scratch (4 * rows * K uint32), massBefore (rows * K float), keptTopP (rows int32),
// keep (rows * K uint8). Metal: four buffers, because lab 11.6's encodeTopPMask binds its
// buffers at offset 0.
struct Work {
  Buf* all = nullptr;
  Buf* scratch = nullptr;
  Buf* mass = nullptr;
  Buf* keep = nullptr;
  Buf* kept = nullptr;
};

#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
using pmpp::metal::buffer;
bool encodeFilterSortedFused(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                             id<MTLBuffer> sortedLogits, id<MTLBuffer> sortedProbs, id<MTLBuffer> n, uint32_t rows,
                             uint32_t K, float topP, float minP);
#if X1_PART_A
bool encodeFilterRows(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> descKeys,
                      id<MTLComputePipelineState> sortPass, id<MTLComputePipelineState> gather,
                      id<MTLComputePipelineState> softmax, id<MTLComputePipelineState> topPMask,
                      id<MTLComputePipelineState> minPCount, id<MTLBuffer> candLogits, id<MTLBuffer> sortedPos,
                      id<MTLBuffer> sortedLogits, id<MTLBuffer> sortedProbs, id<MTLBuffer> n, id<MTLBuffer> sortScratch,
                      id<MTLBuffer> massBefore, id<MTLBuffer> keep, id<MTLBuffer> keptTopP, uint32_t rows, uint32_t K,
                      float topP, float minP);
bool encodeMinPCount(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                     id<MTLBuffer> sortedLogits, id<MTLBuffer> keptTopP, id<MTLBuffer> n, uint32_t rows, uint32_t K,
                     float minP);
// Provided: labs 14.3, 10.3 and 11.6's solution/launch.mm, linked into this program.
bool encodeSortRowsDesc(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> descKeys,
                        id<MTLComputePipelineState> sortPass, id<MTLComputePipelineState> gather, id<MTLBuffer> probs,
                        id<MTLBuffer> sortedProbs, id<MTLBuffer> sortedIdx, id<MTLBuffer> scratch, uint32_t rows,
                        uint32_t cols);
bool encodeSoftmaxRows(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, id<MTLBuffer> x,
                       id<MTLBuffer> y, uint32_t rows, uint32_t cols, float scale);
bool encodeTopPMask(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline, id<MTLBuffer> probs,
                    id<MTLBuffer> massBefore, id<MTLBuffer> keep, id<MTLBuffer> kept, uint32_t rows, uint32_t cols,
                    float p);

// The earlier labs' own kernel.metal files, found from this lab's: labs/x1/... -> labs/.
static std::string labsDir() {
  const std::string s = PMPP_METAL_SOURCE;
  return s.substr(0, s.rfind("/x1/"));
}
static const char* kSortMetal = "/ch14/lab3_sortrows/solution/kernel.metal";
static const char* kSoftmaxMetal = "/ch10/lab3_softmax_argmax/solution/kernel.metal";
static const char* kTopPMetal = "/ch11/lab6_topp/solution/kernel.metal";
#endif

struct Pipes {
  id<MTLComputePipelineState> fused;
#if X1_PART_A
  id<MTLComputePipelineState> minP, keys, pass, gather, softmax, topP;
#endif
};
static Pipes pipes() {
  Pipes p;
  p.fused = pmpp::metal::pipeline(PMPP_METAL_SOURCE, "filter_sorted_fused");
#if X1_PART_A
  p.minP = pmpp::metal::pipeline(PMPP_METAL_SOURCE, "min_p_count");
  p.keys = pmpp::metal::pipeline(labsDir() + kSortMetal, "desc_keys");
  p.pass = pmpp::metal::pipeline(labsDir() + kSortMetal, "sort_rows_pass");
  p.gather = pmpp::metal::pipeline(labsDir() + kSortMetal, "gather_rows");
  p.softmax = pmpp::metal::pipeline(labsDir() + kSoftmaxMetal, "softmax_rows");
  p.topP = pmpp::metal::pipeline(labsDir() + kTopPMetal, "top_p_mask");
#endif
  return p;
}

// Why one of your pipelines cannot run as the lab specifies, or "". The driver's own copy of
// the rule a correct launch.mm applies: 256 threads per threadgroup, in whole SIMD-groups, and
// no more SIMD-groups than a SIMD-group has lanes.
static std::string pipeProblem(id<MTLComputePipelineState> p, const char* name) {
  if (!p) return std::string("the Metal kernel ") + name + " did not build: " + pmpp::take_error();
  const NSUInteger w = p.threadExecutionWidth;
  if ((NSUInteger)kBlock > p.maxTotalThreadsPerThreadgroup || w == 0 || kBlock % w != 0 || (NSUInteger)kBlock / w > w) {
    char b[240];
    std::snprintf(b, sizeof b, "%s: this pipeline (SIMD-group width %lu, at most %lu threads) cannot run %d threads "
                  "as whole SIMD-groups with no more groups than lanes", name, (unsigned long)w,
                  (unsigned long)p.maxTotalThreadsPerThreadgroup, kBlock);
    return b;
  }
  return "";
}
static std::string fusedProblem() { return pipeProblem(pipes().fused, "filter_sorted_fused"); }
#if X1_PART_A
static std::string composedProblem() {
  Pipes p = pipes();
  if (!p.keys || !p.pass || !p.gather) return "lab 14.3's kernel.metal did not build: " + pmpp::take_error();
  if (!p.softmax) return "lab 10.3's kernel.metal did not build: " + pmpp::take_error();
  if (!p.topP) return "lab 11.6's kernel.metal did not build: " + pmpp::take_error();
  return pipeProblem(p.minP, "min_p_count");
}
#endif

// Each run function enqueues `repeat` calls in one timed run; "" when it ran.
static std::string runFused(Buf* sl, Buf* sp, Buf* n, int rows, int K, float topP, float minP, int repeat = 1,
                            pmpp::Timing* tm = nullptr) {
  std::string why = fusedProblem();
  if (!why.empty()) return why;
  Pipes p = pipes();
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    for (int i = 0; i < repeat && ok; ++i)
      ok = encodeFilterSortedFused(enc, p.fused, buffer(sl), buffer(sp), buffer(n), (uint32_t)rows, (uint32_t)K, topP, minP);
  });
  if (tm) *tm = t;
  return ok ? "" : "your encodeFilterSortedFused refused to dispatch, although this pipeline can run 256 threads per threadgroup";
}

#if X1_PART_A
static void allocWork(Work& w, int rows, int K, size_t guard) {
  const size_t cells = (size_t)rows * K;
  w.scratch = pmpp::alloc(16 * cells + 4 * guard);
  w.mass = pmpp::alloc(4 * cells + 4 * guard);
  w.keep = pmpp::alloc(cells + guard);
  w.kept = pmpp::alloc(4 * (size_t)rows + 4 * guard);
}
static void releaseWork(Work& w) {
  pmpp::release(w.scratch); pmpp::release(w.mass); pmpp::release(w.keep); pmpp::release(w.kept);
}

static std::string runComposed(Buf* cand, Buf* pos, Buf* sl, Buf* sp, Buf* n, Work& w, int rows, int K, float topP,
                               float minP, int repeat = 1, pmpp::Timing* tm = nullptr) {
  std::string why = composedProblem();
  if (!why.empty()) return why;
  Pipes p = pipes();
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    for (int i = 0; i < repeat && ok; ++i)
      ok = encodeFilterRows(enc, p.keys, p.pass, p.gather, p.softmax, p.topP, p.minP, buffer(cand), buffer(pos),
                            buffer(sl), buffer(sp), buffer(n), buffer(w.scratch), buffer(w.mass), buffer(w.keep),
                            buffer(w.kept), (uint32_t)rows, (uint32_t)K, topP, minP);
  });
  if (tm) *tm = t;
  return ok ? "" : "your encodeFilterRows refused to dispatch, although every pipeline can run 256 threads per threadgroup";
}

// The composed chain after the sort: softmax, top-p mask, min-p count (the benchmark's middle row).
static std::string runUnfused(Buf* sl, Buf* sp, Buf* n, Work& w, int rows, int K, float topP, float minP,
                              int repeat = 1, pmpp::Timing* tm = nullptr) {
  std::string why = composedProblem();
  if (!why.empty()) return why;
  Pipes p = pipes();
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    const uint32_t r = (uint32_t)rows, k = (uint32_t)K;
    for (int i = 0; i < repeat && ok; ++i)
      ok = encodeSoftmaxRows(enc, p.softmax, buffer(sl), buffer(sp), r, k, 1.0f) &&
           encodeTopPMask(enc, p.topP, buffer(sp), buffer(w.mass), buffer(w.keep), buffer(w.kept), r, k, topP) &&
           encodeMinPCount(enc, p.minP, buffer(sl), buffer(w.kept), buffer(n), r, k, minP);
  });
  if (tm) *tm = t;
  return ok ? "" : "a launcher refused to dispatch, although every pipeline can run 256 threads per threadgroup";
}

// The provided pieces alone.
static std::string runSort(Buf* cand, Buf* sl, Buf* pos, Work& w, int rows, int K, int repeat = 1,
                           pmpp::Timing* tm = nullptr) {
  Pipes p = pipes();
  if (!p.keys || !p.pass || !p.gather) return "lab 14.3's kernel.metal did not build: " + pmpp::take_error();
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    for (int i = 0; i < repeat && ok; ++i)
      ok = encodeSortRowsDesc(enc, p.keys, p.pass, p.gather, buffer(cand), buffer(sl), buffer(pos), buffer(w.scratch),
                              (uint32_t)rows, (uint32_t)K);
  });
  if (tm) *tm = t;
  return ok ? "" : "lab 14.3's encodeSortRowsDesc refused to dispatch on this device";
}
static std::string runSoftmax(Buf* x, Buf* y, int rows, int K) {
  Pipes p = pipes();
  if (!p.softmax) return "lab 10.3's kernel.metal did not build: " + pmpp::take_error();
  bool ok = true;
  pmpp::run([&](void* v) {
    ok = encodeSoftmaxRows(((pmpp::metal::Pass*)v)->encoder, p.softmax, buffer(x), buffer(y), (uint32_t)rows,
                           (uint32_t)K, 1.0f);
  });
  return ok ? "" : "lab 10.3's encodeSoftmaxRows refused to dispatch on this device";
}
static std::string runTopP(Buf* probs, Work& w, int rows, int K, float topP) {
  Pipes p = pipes();
  if (!p.topP) return "lab 11.6's kernel.metal did not build: " + pmpp::take_error();
  bool ok = true;
  pmpp::run([&](void* v) {
    ok = encodeTopPMask(((pmpp::metal::Pass*)v)->encoder, p.topP, buffer(probs), buffer(w.mass), buffer(w.keep),
                        buffer(w.kept), (uint32_t)rows, (uint32_t)K, topP);
  });
  return ok ? "" : "lab 11.6's encodeTopPMask refused to dispatch on this device";
}
#endif   // X1_PART_A
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }

#else   // CUDA, HIP and the emulators
void launchFilterSortedFused(const float* sortedLogits, float* sortedProbs, int* n, int rows, int K, float topP,
                             float minP);
#if X1_PART_A
void launchFilterRows(const float* candLogits, int* sortedPos, float* sortedLogits, float* sortedProbs, int* n,
                      void* workspace, int rows, int K, float topP, float minP);
void launchMinPCount(const float* sortedLogits, const int* keptTopP, int* n, int rows, int K, float minP);
// Provided: labs 14.3, 10.3 and 11.6, from ../shared-g12/.
void launchSortRowsDesc(const float* probs, float* sortedProbs, int* sortedIdx, unsigned int* scratch, int rows,
                        int cols);
void launchSoftmaxRows(const float* x, float* y, int rows, int cols, float scale);
void launchTopPMask(const float* probs, float* massBefore, unsigned char* keep, int* kept, int rows, int cols,
                    float p);
#endif

static std::string threadsProblem() {
  const int most = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  return kBlock <= most ? "" : "this device runs at most " + std::to_string(most) + " threads per block; the lab needs 256";
}
static std::string fusedProblem() { return threadsProblem(); }

static std::string runFused(Buf* sl, Buf* sp, Buf* n, int rows, int K, float topP, float minP, int repeat = 1,
                            pmpp::Timing* tm = nullptr) {
  std::string why = fusedProblem();
  if (!why.empty()) return why;
  pmpp::Timing t = pmpp::run([&](void*) {
    for (int i = 0; i < repeat; ++i)
      launchFilterSortedFused((const float*)pmpp::raw(sl), (float*)pmpp::raw(sp), (int*)pmpp::raw(n), rows, K, topP, minP);
  });
  if (tm) *tm = t;
  return "";
}

#if X1_PART_A
static std::string composedProblem() { return threadsProblem(); }
static size_t workBytes(int rows, int K) { return 21 * (size_t)rows * K + 4 * (size_t)rows; }
static void allocWork(Work& w, int rows, int K, size_t guard) { w.all = pmpp::alloc(workBytes(rows, K) + guard); }
static void releaseWork(Work& w) { pmpp::release(w.all); }
// The pieces of the workspace, by the layout above.
struct WorkPtrs { unsigned int* scratch; float* mass; int* kept; unsigned char* keep; };
static WorkPtrs parts(Work& w, int rows, int K) {
  const size_t cells = (size_t)rows * K;
  char* base = (char*)pmpp::raw(w.all);
  WorkPtrs p;
  p.scratch = (unsigned int*)base;
  p.mass = (float*)(base + 16 * cells);
  p.kept = (int*)(base + 20 * cells);
  p.keep = (unsigned char*)(base + 20 * cells + 4 * (size_t)rows);
  return p;
}

static std::string runComposed(Buf* cand, Buf* pos, Buf* sl, Buf* sp, Buf* n, Work& w, int rows, int K, float topP,
                               float minP, int repeat = 1, pmpp::Timing* tm = nullptr) {
  std::string why = composedProblem();
  if (!why.empty()) return why;
  pmpp::Timing t = pmpp::run([&](void*) {
    for (int i = 0; i < repeat; ++i)
      launchFilterRows((const float*)pmpp::raw(cand), (int*)pmpp::raw(pos), (float*)pmpp::raw(sl), (float*)pmpp::raw(sp),
                       (int*)pmpp::raw(n), pmpp::raw(w.all), rows, K, topP, minP);
  });
  if (tm) *tm = t;
  return "";
}
static std::string runUnfused(Buf* sl, Buf* sp, Buf* n, Work& w, int rows, int K, float topP, float minP,
                              int repeat = 1, pmpp::Timing* tm = nullptr) {
  std::string why = composedProblem();
  if (!why.empty()) return why;
  WorkPtrs p = parts(w, rows, K);
  pmpp::Timing t = pmpp::run([&](void*) {
    for (int i = 0; i < repeat; ++i) {
      launchSoftmaxRows((const float*)pmpp::raw(sl), (float*)pmpp::raw(sp), rows, K, 1.0f);
      launchTopPMask((const float*)pmpp::raw(sp), p.mass, p.keep, p.kept, rows, K, topP);
      launchMinPCount((const float*)pmpp::raw(sl), p.kept, (int*)pmpp::raw(n), rows, K, minP);
    }
  });
  if (tm) *tm = t;
  return "";
}
static std::string runSort(Buf* cand, Buf* sl, Buf* pos, Work& w, int rows, int K, int repeat = 1,
                           pmpp::Timing* tm = nullptr) {
  WorkPtrs p = parts(w, rows, K);
  pmpp::Timing t = pmpp::run([&](void*) {
    for (int i = 0; i < repeat; ++i)
      launchSortRowsDesc((const float*)pmpp::raw(cand), (float*)pmpp::raw(sl), (int*)pmpp::raw(pos), p.scratch, rows, K);
  });
  if (tm) *tm = t;
  return "";
}
static std::string runSoftmax(Buf* x, Buf* y, int rows, int K) {
  pmpp::run([&](void*) { launchSoftmaxRows((const float*)pmpp::raw(x), (float*)pmpp::raw(y), rows, K, 1.0f); });
  return "";
}
static std::string runTopP(Buf* probs, Work& w, int rows, int K, float topP) {
  WorkPtrs p = parts(w, rows, K);
  pmpp::run([&](void*) { launchTopPMask((const float*)pmpp::raw(probs), p.mass, p.keep, p.kept, rows, K, topP); });
  return "";
}
#endif   // X1_PART_A
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

// ---------------------------------------------------------------- tolerances (README.md)
// Probabilities: |got - ref| <= kRtol * ref, where ref is the softmax in double of the float
// logits. The mass before a token: the same relative band. Worst cases we can bound at the
// largest row tested (151,936 logits, 256 threads, exponent arguments in [-24, 0]): about 5e-5
// for a probability and 7e-5 for a mass with Metal's fast exp (README.md derives both); kRtol
// leaves room above that. Inside the band either top-p decision is accepted.
static const double kRtol = 1e-4;

// min-p: the GPU adds the same two floats as the host (max + logf(minP)) and compares. That is
// exact whenever the maximum is 0 (the exact and planted rows). Elsewhere a compiler allowed to
// reassociate may compare l - max with logf(minP) instead, so logits within this margin of the
// threshold may go either way.
static double minPMargin(float l0, float lnMinP) { return l0 == 0.0f ? 0.0 : 1e-6 * (std::fabs(l0) + std::fabs(lnMinP)); }

// ---------------------------------------------------------------- test rows
enum Kind {
  Exact,     // logits 0 and -1000 only: every probability and partial sum is a power of two or 0
  Planted,   // the maximum is 0, a logit planted at 0 + logf(minP) and one just below: n exact
  Random,    // softmax-shaped: probabilities and n within the band
};
struct Row {
  std::vector<float> logits;   // candidate order (unsorted)
  Kind kind;
  std::string what;
};
struct Setting {
  float topP, minP;
  std::string what;
  std::vector<Row> rows;       // a batch takes rows from here in turn
};

static uint32_t xorshift(uint32_t& x) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; }
static void shuffle(std::vector<float>& v, uint32_t seed) {
  uint32_t x = seed * 2654435761u + 977u;
  for (size_t i = v.size(); i > 1; --i) std::swap(v[i - 1], v[xorshift(x) % i]);
}
static int pow2Floor(int x) { int g = 1; while (2 * g <= x) g *= 2; return g; }
static std::string zeros(int G) { return std::to_string(G) + (G == 1 ? " logit" : " logits") + " of 0"; }

// G logits of 0 (weight exactly 1), the planted ones, and -1000 (weight exactly 0 in float and
// in double) for the rest, in shuffled positions so that the sort has work and ties to keep.
static Row groupRow(int K, int G, const std::vector<float>& planted, uint32_t seed, const std::string& what) {
  std::vector<float> v((size_t)K, -1000.0f);
  for (int i = 0; i < G && i < K; ++i) v[i] = 0.0f;
  for (size_t i = 0; i < planted.size() && G + (int)i < K; ++i) v[G + i] = planted[i];
  shuffle(v, seed);
  return {v, planted.empty() ? Exact : Planted, what};
}

// Softmax-shaped logits in [-12, 12]: uniform, peaked (one at 12, the rest in [-12, 0]) or
// flat (in [-1, 1]). pmpp::random_floats has 65,536 levels, so long rows hold ties.
static Row randomRow(int K, int shape, uint32_t seed) {
  std::vector<float> u = pmpp::random_floats((size_t)K, seed);
  std::vector<float> v((size_t)K);
  for (int i = 0; i < K; ++i) v[i] = shape == 0 ? 12.0f * u[i] : shape == 1 ? 6.0f * u[i] - 6.0f : u[i];
  if (shape == 1) v[seed % (uint32_t)K] = 12.0f;
  static const char* names[] = {"logits uniform in [-12, 12]", "peaked: one logit 12, the rest in [-12, 0]",
                                "nearly flat: logits in [-1, 1]"};
  return {v, Random, names[shape]};
}

static std::string num(double v) { char b[48]; std::snprintf(b, sizeof b, "%.9g", v); return b; }
// A float in the fewest digits that read back as the same float (0.9f prints as 0.9).
static std::string numF(float v) {
  char b[48];
  for (int digits = 6; digits <= 9; ++digits) {
    std::snprintf(b, sizeof b, "%.*g", digits, (double)v);
    if (std::strtof(b, nullptr) == v) break;
  }
  return b;
}

// The settings for rows of K candidates. `level`: 2 on a device, 1 on emu, 0 on race (fewer
// settings where every emulated thread runs on the CPU).
static std::vector<Setting> settings(int K, bool exact, int level) {
  std::vector<Setting> out;
  uint32_t seed = 7919u * (uint32_t)K + 13u;
  auto add = [&](float p, float m, const std::string& what, const std::vector<Row>& first) {
    Setting s{p, m, what, first};
    for (int shape = 0; shape < 3; ++shape) s.rows.push_back(randomRow(K, shape, seed += 101u));
    out.push_back(s);
  };
  if (!exact) {
    const float ps[] = {0.5f, 0.9f, 0.95f}, ms[] = {0.0f, 0.05f, 0.1f};
    for (float p : ps)
      for (float m : ms) {
        const bool keep = level == 2 || (level == 1 && ((p == 0.9f && m == 0.05f) || (p == 0.5f && m == 0.1f) ||
                                                        (p == 0.95f && m == 0.0f))) ||
                          (p == 0.9f && m == 0.05f);
        if (keep) add(p, m, "softmax rows, p = " + numF(p) + ", min-p = " + numF(m), {});
      }
    return out;
  }
  // Exact rows: G logits of 0 make each probability 1/G; p = (t + 1) / G puts the inclusive
  // sum at token t exactly on p, so token t is kept and token t + 1 dropped.
  auto at = [&](int G, int t, float m, const std::string& where) {
    const float p = (float)(t + 1) / (float)G;
    add(p, m, "p = " + numF(p) + ": the inclusive sum at token " + std::to_string(t) + " is p (" + where + "), min-p = " + numF(m),
        {groupRow(K, G, {}, seed += 7u, zeros(G) + ", the rest -1000")});
  };
  const int G0 = pow2Floor(std::min(K, 4096));
  at(G0, 0, 0.05f, "token 0");
  if (K >= 256) at(256, 129, 0.0f, "inside a lane's run of 4");
  else if (K >= 4) at(pow2Floor(K), 1, 0.0f, "inside a lane's run of 4");
  if (K >= 2048) {
    at(2048, 1023, 0.1f, "the last token of the first chunk of 1,024");
    at(2048, 1024, 0.0f, "the first token of the second chunk");
    if (level >= 1) at(2048, 1100, 0.05f, "inside the second chunk");
  }
  const int G = pow2Floor(std::min(K, 256));
  const std::vector<float> edgeP = level == 2 ? std::vector<float>{0.0f, 1e-7f, 1.0f, 1.5f}
                                              : std::vector<float>{0.0f, 1.0f};
  for (float p : edgeP)
    add(p, 0.0f, "p = " + numF(p) + (p >= 1.0f ? " keeps every token" : " keeps token 0 only") + ", min-p = 0",
        {groupRow(K, G, {}, seed += 7u, zeros(G) + ", the rest -1000")});
  // min-p: a logit planted at exactly 0 + logf(minP), as the host computes the threshold, is
  // kept (>=); the float just below it is not. p = 1, so top-p keeps everything.
  const std::vector<float> minPs = level == 2 ? std::vector<float>{0.05f, 0.1f} : std::vector<float>{0.05f};
  for (float m : minPs) {
    const float thr = 0.0f + logf(m);
    const int Gp = K >= 3 ? pow2Floor(std::min(K - 2, 64)) : 1;
    add(1.0f, m, "p = 1, min-p = " + numF(m) + ": a logit planted at exactly logf(min-p) = " + numF(thr),
        {groupRow(K, Gp, {thr, std::nextafter(thr, -INFINITY)}, seed += 7u,
                  zeros(Gp) + ", one at logf(min-p), one just below it, the rest -1000")});
  }
  add(1.0f, 1.0f, "p = 1, min-p = 1: only the logits tied with the maximum survive",
      {groupRow(K, pow2Floor(std::min(K, 64)), {}, seed += 7u, zeros(pow2Floor(std::min(K, 64))) + ", the rest -1000")});
  return out;
}

// ---------------------------------------------------------------- the CPU reference
struct Ref {
  std::vector<int> order;                // stable descending: equal logits by lower position
  std::vector<float> sorted;             // the logits in that order
  std::vector<double> prob, massBefore;  // softmax in double of the float logits, and its exclusive scan
  int lo, hi;                            // the accepted range of n
  // What some wrong designs would give, for the messages (meaningful when lo == hi).
  int inclusiveN, noMinPN, noTopPN, strictMinPN, noCarryN;
};

static Ref reference(const Row& row, float topP, float minP) {
  const int K = (int)row.logits.size();
  Ref r;
  r.order.resize((size_t)K);
  std::iota(r.order.begin(), r.order.end(), 0);
  std::stable_sort(r.order.begin(), r.order.end(), [&](int a, int b) { return row.logits[a] > row.logits[b]; });
  r.sorted.resize((size_t)K);
  for (int i = 0; i < K; ++i) r.sorted[i] = row.logits[r.order[i]];

  const float l0 = r.sorted[0];
  double s = 0;
  r.prob.resize((size_t)K);
  for (int i = 0; i < K; ++i) s += (r.prob[i] = std::exp((double)r.sorted[i] - (double)l0));
  r.massBefore.resize((size_t)K);
  double m = 0;
  for (int i = 0; i < K; ++i) { r.prob[i] /= s; r.massBefore[i] = m; m += r.prob[i]; }

  // Top-p: keep i when i == 0, or p >= 1, or massBefore < p. A GPU's mass is within kRtol of the
  // reference (README.md), so a token whose reference mass is within that of p may go either way.
  const double p = topP;
  const bool all = topP >= 1.0f;
  int topLo = 0, topHi = 0, incl = 0, noCarry = 0;
  for (int i = 0; i < K; ++i) {
    const double band = row.kind == Exact ? 0.0 : kRtol * r.massBefore[i];
    topLo += (i == 0 || all || r.massBefore[i] + band < p) ? 1 : 0;
    topHi += (i == 0 || all || r.massBefore[i] - band < p) ? 1 : 0;
    incl += (i == 0 || all || r.massBefore[i] + r.prob[i] < p) ? 1 : 0;
    noCarry += (i == 0 || all || r.massBefore[i] - r.massBefore[(size_t)(i / kChunk) * kChunk] < p) ? 1 : 0;
  }
  // Min-p: keep logits >= max + logf(minP), the float sum the host computes; every logit if minP <= 0.
  const float lnMinP = minP > 0.0f ? logf(minP) : 0.0f;
  const float thr = minP > 0.0f ? l0 + lnMinP : -std::numeric_limits<float>::max();
  const double margin = minP > 0.0f ? minPMargin(l0, lnMinP) : 0.0;
  int minLo = 0, minHi = 0, strict = 0;
  for (int i = 0; i < K; ++i) {
    const double d = (double)r.sorted[i] - (double)thr;
    minLo += (margin == 0.0 ? d >= 0 : d > margin) ? 1 : 0;
    minHi += d >= -margin ? 1 : 0;
    strict += d > 0 ? 1 : 0;
  }
  auto n = [](int a, int b) { return std::max(1, std::min(a, b)); };
  r.lo = n(topLo, minLo);
  r.hi = n(topHi, minHi);
  r.inclusiveN = n(incl, minLo);
  r.noMinPN = std::max(1, topLo);
  r.noTopPN = std::max(1, minLo);
  r.strictMinPN = n(topLo, minP > 0.0f ? strict : minLo);
  r.noCarryN = n(noCarry, minLo);
  return r;
}

// ---------------------------------------------------------------- a batch and its checks
struct Batch {
  int rows, K;
  float topP, minP;
  std::vector<float> cand;           // [rows x K] candidate logits
  std::vector<float> sorted;         // [rows x K] the reference's sorted logits: part B's input
  std::vector<const Row*> src;
  std::vector<Ref> refs;
  std::string label;
  std::string key;                   // label and first row: names the batch for part B's comparison
};

static Batch makeBatch(const Setting& s, int start, int rows, int K) {
  Batch b{rows, K, s.topP, s.minP, std::vector<float>((size_t)rows * K), std::vector<float>((size_t)rows * K), {}, {}, "", ""};
  for (int r = 0; r < rows; ++r) {
    const Row* row = &s.rows[(size_t)(start + r) % s.rows.size()];
    b.src.push_back(row);
    b.refs.push_back(reference(*row, s.topP, s.minP));
    std::memcpy(&b.cand[(size_t)r * K], row->logits.data(), (size_t)K * 4);
    std::memcpy(&b.sorted[(size_t)r * K], b.refs.back().sorted.data(), (size_t)K * 4);
  }
  b.label = std::to_string(rows) + (rows == 1 ? " row" : " rows") + " of K = " + std::to_string(K) + ", " + s.what;
  b.key = b.label + " #" + std::to_string(start);
  return b;
}

static std::string at(const Batch& b, int r) {
  return b.label + "; row " + std::to_string(r) + " (" + b.src[r]->what + ")";
}

static const int kGuard = 64;                       // entries past the end of every output
static const uint32_t kGuardBits = 0xDEADBEEFu;
static const int kGuardInt = -77;
static const uint32_t kUnwritten = 0xFFFFFFFFu;     // float outputs before the launch (a NaN)
static const uint8_t kWorkFill = 0xCD, kWorkGuard = 0x5A;

static uint32_t bitsOf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float fromBits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

static Buf* filledFloats(size_t n, uint32_t inside) {
  std::vector<uint32_t> v(n + kGuard, kGuardBits);
  std::fill(v.begin(), v.begin() + n, inside);
  Buf* b = pmpp::alloc(v.size() * 4);
  pmpp::upload(b, v.data(), v.size() * 4);
  return b;
}
static Buf* filledInts(size_t n, int inside) {
  std::vector<int> v(n + kGuard, kGuardInt);
  std::fill(v.begin(), v.begin() + n, inside);
  Buf* b = pmpp::alloc(v.size() * 4);
  pmpp::upload(b, v.data(), v.size() * 4);
  return b;
}
template <class T> static std::vector<T> fetch(Buf* b, size_t n) {
  std::vector<T> v(n + kGuard);
  pmpp::download(b, v.data(), v.size() * sizeof(T));
  return v;
}
template <class T> static std::string guardOf(const std::vector<T>& v, size_t n, T guard, const char* name) {
  for (size_t g = 0; g < (size_t)kGuard; ++g)
    if (std::memcmp(&v[n + g], &guard, sizeof(T)) != 0)
      return std::string("wrote ") + name + "[" + std::to_string(n + g) + "], past the end of the output";
  return "";
}

// sortedProbs of one row against the reference: exact rows bit for bit, others within the band.
static std::string checkProbs(const Batch& b, int r, const float* got, const char* who) {
  const Ref& ref = b.refs[r];
  for (int i = 0; i < b.K; ++i) {
    const double want = ref.prob[i];
    const bool exact = b.src[r]->kind == Exact;
    const bool ok = exact ? (double)got[i] == want : std::fabs((double)got[i] - want) <= kRtol * want;
    if (ok) continue;
    std::string e = at(b, r) + ": " + who + " sortedProbs[" + std::to_string(i) + "] = " + num(got[i]) + ", expected " +
                    num(want);
    if (bitsOf(got[i]) == kUnwritten) return e + " (never written)";
    if (exact) e += " (logits 0 and -1000 give probabilities that are powers of two or 0: no rounding excuse)";
    else e += " (off by " + num(std::fabs(got[i] - want) / want) + " relative; the band is " + num(kRtol) + ")";
    if (i > 0 && std::fabs((double)got[i] / want - (double)got[0] / ref.prob[0]) > kRtol)
      e += ". Not every token is off by the same factor: does every thread divide by the same sum?";
    return e;
  }
  return "";
}

// n of one row against the reference range, with a hint where a known mistake explains it.
static std::string checkN(const Batch& b, int r, int got, const char* who) {
  const Ref& ref = b.refs[r];
  if (got >= ref.lo && got <= ref.hi) return "";
  std::string e = at(b, r) + ": " + who + " n = " + std::to_string(got) + ", expected " +
                  (ref.lo == ref.hi ? std::to_string(ref.lo) : std::to_string(ref.lo) + " to " + std::to_string(ref.hi) +
                                                                  " (inside the band either neighbour is right)");
  if (got == -1) return e + " (never written)";
  if (got < 1 || got > b.K) return e + " (n must be between 1 and K)";
  if (ref.lo != ref.hi) return e;
  if (got == ref.inclusiveN) e += ". That is the count with the INCLUSIVE mass: the token that crosses p must be kept";
  else if (got == ref.strictMinPN) e += ". That is min-p with >: a logit exactly at the threshold is kept (>=)";
  else if (got == ref.noCarryN && b.K > kChunk)
    e += ". That is the count when tokens past the first chunk of 1,024 do not see the mass of earlier chunks";
  else if (got == ref.noMinPN && got != ref.lo) e += ". That is top-p alone: min-p was not applied";
  else if (got == ref.noTopPN && got != ref.lo) e += ". That is min-p alone: top-p was not applied";
  return e;
}

#if X1_PART_A
// Part A: candidates in, your launchFilterRows / encodeFilterRows, everything checked.
static std::map<std::string, std::vector<int>> gComposedN;   // part B compares with these

static std::string checkPartA(const Batch& b) {
  const size_t cells = (size_t)b.rows * b.K;
  Buf* cand = pmpp::alloc(cells * 4);
  pmpp::upload(cand, b.cand.data(), cells * 4);
  Buf* pos = filledInts(cells, -1);
  Buf* sl = filledFloats(cells, kUnwritten);
  Buf* sp = filledFloats(cells, kUnwritten);
  Buf* n = filledInts((size_t)b.rows, -1);
  Work w;
  allocWork(w, b.rows, b.K, kGuard);
  auto fillWork = [&](Buf* x, size_t bytes, size_t guardBytes) {
    std::vector<uint8_t> v(bytes + guardBytes, kWorkGuard);
    std::fill(v.begin(), v.begin() + bytes, kWorkFill);
    pmpp::upload(x, v.data(), v.size());
  };
  auto workGuard = [&](Buf* x, size_t bytes, size_t guardBytes, const char* name) -> std::string {
    std::vector<uint8_t> v(bytes + guardBytes);
    pmpp::download(x, v.data(), v.size());
    for (size_t g = bytes; g < v.size(); ++g)
      if (v[g] != kWorkGuard) return std::string("wrote the workspace (") + name + ") at byte " + std::to_string(g) + ", past its end";
    return "";
  };
#if defined(PMPP_BACKEND_METAL)
  fillWork(w.scratch, 16 * cells, 4 * kGuard); fillWork(w.mass, 4 * cells, 4 * kGuard);
  fillWork(w.keep, cells, kGuard); fillWork(w.kept, 4 * (size_t)b.rows, 4 * kGuard);
#else
  fillWork(w.all, workBytes(b.rows, b.K), kGuard);
#endif

  std::string err = runComposed(cand, pos, sl, sp, n, w, b.rows, b.K, b.topP, b.minP);
  if (err.empty()) err = pmpp::take_error();
  if (!err.empty()) err = "launch error: " + err;
  std::vector<int> gotN;
  if (err.empty()) {
    std::vector<int> gp = fetch<int>(pos, cells), gn = fetch<int>(n, (size_t)b.rows);
    std::vector<uint32_t> gl = fetch<uint32_t>(sl, cells), gs = fetch<uint32_t>(sp, cells);
    std::vector<float> after(cells);
    pmpp::download(cand, after.data(), cells * 4);
    for (int r = 0; r < b.rows && err.empty(); ++r) {
      const Ref& ref = b.refs[r];
      const int* p = &gp[(size_t)r * b.K];
      // sortedPos: a permutation, in the stable order.
      std::vector<int> seen((size_t)b.K, -1);
      for (int i = 0; i < b.K && err.empty(); ++i) {
        if (p[i] < 0 || p[i] >= b.K) err = at(b, r) + ": sortedPos[" + std::to_string(i) + "] = " + std::to_string(p[i]) +
                                           (p[i] == -1 ? " (never written)" : ", not a position in the row");
        else if (seen[p[i]] >= 0) err = at(b, r) + ": position " + std::to_string(p[i]) + " appears twice in sortedPos";
        else seen[p[i]] = i;
      }
      for (int i = 0; i < b.K && err.empty(); ++i)
        if (p[i] != ref.order[i]) {
          const float x = b.src[r]->logits[p[i]], y = b.src[r]->logits[ref.order[i]];
          err = at(b, r) + ": sortedPos[" + std::to_string(i) + "] = " + std::to_string(p[i]) + " (logit " + num(x) +
                "), expected " + std::to_string(ref.order[i]) + " (logit " + num(y) + ")" +
                (x == y ? ": equal logits stay in position order (a stable sort)" : ": descending order");
        }
      // sortedLogits: the same 32 bits as the candidate at that position.
      for (int i = 0; i < b.K && err.empty(); ++i)
        if (gl[(size_t)r * b.K + i] != bitsOf(ref.sorted[i]))
          err = at(b, r) + ": sortedLogits[" + std::to_string(i) + "] has the bits of " + num(fromBits(gl[(size_t)r * b.K + i])) +
                ", expected " + num(ref.sorted[i]);
      if (err.empty()) {
        std::vector<float> probs(b.K);
        std::memcpy(probs.data(), &gs[(size_t)r * b.K], (size_t)b.K * 4);
        err = checkProbs(b, r, probs.data(), "part A:");
      }
      if (err.empty()) err = checkN(b, r, gn[r], "part A:");
    }
    if (err.empty()) err = guardOf<int>(gp, cells, kGuardInt, "sortedPos");
    if (err.empty()) err = guardOf<uint32_t>(gl, cells, kGuardBits, "sortedLogits");
    if (err.empty()) err = guardOf<uint32_t>(gs, cells, kGuardBits, "sortedProbs");
    if (err.empty()) err = guardOf<int>(gn, (size_t)b.rows, kGuardInt, "n");
#if defined(PMPP_BACKEND_METAL)
    if (err.empty()) err = workGuard(w.scratch, 16 * cells, 4 * kGuard, "sort scratch");
    if (err.empty()) err = workGuard(w.mass, 4 * cells, 4 * kGuard, "massBefore");
    if (err.empty()) err = workGuard(w.keep, cells, kGuard, "keep");
    if (err.empty()) err = workGuard(w.kept, 4 * (size_t)b.rows, 4 * kGuard, "keptTopP");
#else
    if (err.empty()) err = workGuard(w.all, workBytes(b.rows, b.K), kGuard, "21 * rows * K + 4 * rows bytes");
#endif
    if (err.empty() && std::memcmp(after.data(), b.cand.data(), cells * 4) != 0) err = "candLogits changed: the input is read-only";
    gotN.assign(gn.begin(), gn.begin() + b.rows);
  }
  if (err.empty()) gComposedN[b.key] = gotN;
  else gComposedN.erase(b.key);
  pmpp::release(cand); pmpp::release(pos); pmpp::release(sl); pmpp::release(sp); pmpp::release(n);
  releaseWork(w);
  return err;
}
#endif

// Part B: the reference's sorted logits in, your fused kernel, probabilities and n checked.
static std::string checkPartB(const Batch& b) {
  const size_t cells = (size_t)b.rows * b.K;
  Buf* sl = pmpp::alloc(cells * 4);
  pmpp::upload(sl, b.sorted.data(), cells * 4);
  Buf* sp = filledFloats(cells, kUnwritten);
  Buf* n = filledInts((size_t)b.rows, -1);
  std::string err = runFused(sl, sp, n, b.rows, b.K, b.topP, b.minP);
  if (err.empty()) err = pmpp::take_error();
  if (!err.empty()) err = "launch error: " + err;
  if (err.empty()) {
    std::vector<float> gs = fetch<float>(sp, cells);
    std::vector<int> gn = fetch<int>(n, (size_t)b.rows);
    std::vector<float> after(cells);
    pmpp::download(sl, after.data(), cells * 4);
    for (int r = 0; r < b.rows && err.empty(); ++r) {
      err = checkProbs(b, r, &gs[(size_t)r * b.K], "part B:");
      if (err.empty()) err = checkN(b, r, gn[r], "part B:");
#if X1_PART_A
      // Outside the band both parts must give the reference's n, so they agree with each other.
      auto c = gComposedN.find(b.key);
      if (err.empty() && c != gComposedN.end() && b.refs[r].lo == b.refs[r].hi && c->second[r] != gn[r])
        err = at(b, r) + ": part B gives n = " + std::to_string(gn[r]) + ", part A gave " + std::to_string(c->second[r]);
#endif
    }
    if (err.empty()) {
      std::vector<uint32_t> bits(gs.size());
      std::memcpy(bits.data(), gs.data(), gs.size() * 4);
      err = guardOf<uint32_t>(bits, cells, kGuardBits, "sortedProbs");
    }
    if (err.empty()) err = guardOf<int>(gn, (size_t)b.rows, kGuardInt, "n");
    if (err.empty() && std::memcmp(after.data(), b.sorted.data(), cells * 4) != 0) err = "sortedLogits changed: the input is read-only";
  }
  pmpp::release(sl); pmpp::release(sp); pmpp::release(n);
  return err;
}

// ---------------------------------------------------------------- tests
static std::vector<int> pick(const std::vector<int>& race, const std::vector<int>& emu, const std::vector<int>& dev) {
  const std::string b = pmpp::backend();
  return b == "race" ? race : b == "emu" ? emu : dev;
}
static int level() { const std::string b = pmpp::backend(); return b == "race" ? 0 : b == "emu" ? 1 : 2; }

// Every setting in batches of each row count. A batch takes rows from the setting's list in
// turn, so one batch mixes an exact row with softmax rows and a kernel that reads or writes the
// wrong row meets a row of another kind. On the emulators, where every launch runs 256 OS
// threads per row, each setting runs one batch that starts with its first row, and the row
// counts take turns from one setting to the next.
static std::string overSettings(const std::vector<Setting>& ss, int K, const std::vector<int>& rowCounts, int repeats,
                                const std::function<std::string(const Batch&)>& check) {
  if (ss.empty()) return "no settings to run (a bug in main.cpp, not in your kernel)";
  for (size_t si = 0; si < ss.size(); ++si) {
    const Setting& s = ss[si];
    for (size_t ri = 0; ri < rowCounts.size(); ++ri) {
      if (pmpp::is_emu() && ri != si % rowCounts.size()) continue;
      const int rows = rowCounts[ri];
      for (int start = 0; start < (int)s.rows.size(); start += rows) {
        if (pmpp::is_emu() && start > 0) break;
        const Batch b = makeBatch(s, start, rows, K);
        for (int rep = 0; rep < repeats; ++rep) {
          std::string err = check(b);
          if (!err.empty()) return repeats > 1 ? "launch " + std::to_string(rep + 1) + " of " + std::to_string(repeats) + ": " + err : err;
        }
      }
    }
  }
  return "";
}

// The first setting that satisfies `want` (the emulators' shorter lists for part A).
static std::vector<Setting> only(const std::vector<Setting>& ss, const std::function<bool(const Setting&)>& want) {
  for (const Setting& s : ss)
    if (want(s)) return {s};
  return {};
}

#if X1_PART_A
// The provided labs on their own, so that a broken provided file shows here and not as a
// failure of your code.
static std::string checkProvided(int K, int rows) {
  std::vector<Setting> list = {settings(K, false, 0)[0]};                      // softmax rows
  if (!pmpp::is_emu()) list.insert(list.begin(), settings(K, true, 0)[0]);    // and an exact row
  for (const Setting& s : list) {
    const Batch b = makeBatch(s, 0, rows, K);
    const size_t cells = (size_t)rows * K;
    Buf* cand = pmpp::alloc(cells * 4);
    pmpp::upload(cand, b.cand.data(), cells * 4);
    Buf* sorted = pmpp::alloc(cells * 4);
    pmpp::upload(sorted, b.sorted.data(), cells * 4);
    Buf* pos = filledInts(cells, -1);
    Buf* sl = filledFloats(cells, kUnwritten);
    Buf* sp = filledFloats(cells, kUnwritten);
    Buf* refProbs = pmpp::alloc(cells * 4);
    std::vector<float> rp(cells);
    for (int r = 0; r < rows; ++r)
      for (int i = 0; i < K; ++i) rp[(size_t)r * K + i] = (float)b.refs[r].prob[i];
    pmpp::upload(refProbs, rp.data(), cells * 4);
    Work w;
    allocWork(w, rows, K, kGuard);
    std::string err = runSort(cand, sl, pos, w, rows, K);
    if (err.empty()) err = pmpp::take_error();
    if (err.empty()) {
      std::vector<int> gp = fetch<int>(pos, cells);
      std::vector<uint32_t> gl = fetch<uint32_t>(sl, cells);
      for (int r = 0; r < rows && err.empty(); ++r)
        for (int i = 0; i < K && err.empty(); ++i)
          if (gp[(size_t)r * K + i] != b.refs[r].order[i] || gl[(size_t)r * K + i] != bitsOf(b.refs[r].sorted[i]))
            err = "lab 14.3's sort: " + at(b, r) + ", position " + std::to_string(i) + " is not the stable descending order";
    }
    if (err.empty()) err = runSoftmax(sorted, sp, rows, K);
    if (err.empty()) err = pmpp::take_error();
    if (err.empty()) {
      std::vector<float> gs = fetch<float>(sp, cells);
      for (int r = 0; r < rows && err.empty(); ++r) {
        err = checkProbs(b, r, &gs[(size_t)r * K], "lab 10.3's softmax:");
        if (!err.empty()) err = "provided " + err;
      }
    }
    if (err.empty()) err = runTopP(refProbs, w, rows, K, b.topP);
    if (err.empty()) err = pmpp::take_error();
    if (err.empty()) {
      std::vector<int> kept((size_t)rows);
#if defined(PMPP_BACKEND_METAL)
      pmpp::download(w.kept, kept.data(), kept.size() * 4);
#else
      std::vector<uint8_t> all(workBytes(rows, K));
      pmpp::download(w.all, all.data(), all.size());
      std::memcpy(kept.data(), all.data() + 20 * cells, kept.size() * 4);
#endif
      for (int r = 0; r < rows && err.empty(); ++r) {
        // The top-p count alone: min-p off, so n's range is top-p's.
        const Ref top = reference(*b.src[r], b.topP, 0.0f);
        if (std::max(1, kept[r]) < top.lo || std::max(1, kept[r]) > top.hi)
          err = "lab 11.6's top-p mask: " + at(b, r) + ", kept = " + std::to_string(kept[r]) + ", expected " +
                std::to_string(top.lo) + (top.lo == top.hi ? "" : " to " + std::to_string(top.hi));
      }
    }
    pmpp::release(cand); pmpp::release(sorted); pmpp::release(pos); pmpp::release(sl); pmpp::release(sp);
    pmpp::release(refProbs);
    releaseWork(w);
    if (!err.empty()) return err;
  }
  return "";
}
#endif

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const int lv = level();
  const std::vector<int> rowCounts = pick({2}, {1, 3}, {1, 3, 7});
#if X1_PART_A
  // The emulators run every emulated thread on the CPU (race: as OS threads under
  // ThreadSanitizer), so their rows stay short: K = 1025 is one token past the first chunk of
  // 1,024, 2,100 and 4,099 are three and five chunks. Lab 14.3's sort alone is dozens of block
  // scans per row there, so part A runs at two sizes on them, on two batches each: the planted
  // min-p row and softmax rows at p = 0.9, min-p = 0.05. Part B runs every size.
  const std::vector<int> Ks = pick({1, 33, 257, 1025, 2100}, {1, 2, 31, 40, 255, 256, 257, 1000, 1024, 1025, 4099},
                                   {1, 2, 31, 40, 255, 256, 257, 1000, 1024, 1025, 4096, 32000, 151936});
  const std::vector<int> KsA = pick({33, 1025}, {40, 1025}, Ks);
  const int repeats = 1;
#else
  // The hunt: part B only. Rows longer than one chunk of 1,024 on every backend, on purpose.
  // On a GPU each batch runs ten times on fresh outputs, so that a race has ten chances to show.
  const std::vector<int> Ks = pick({33, 1025, 2100}, {1000, 1024, 1025, 2100, 4099},
                                   {1, 31, 257, 1000, 1024, 1025, 2100, 4096, 4099, 32000});
  const int repeats = pmpp::is_emu() ? 1 : 10;
#endif

  std::string why = fusedProblem();
#if X1_PART_A
  if (why.empty()) why = composedProblem();
#endif
  if (!why.empty()) {   // reported once, not once per shape
    t.check("kernels usable on this device", [&] { return why; });
    return t.finish();
  }
  for (int K : Ks) {
    const std::string k = ", K = " + std::to_string(K);
    const std::vector<Setting> exact = settings(K, true, lv), soft = settings(K, false, lv);
#if X1_PART_A
    if (std::find(KsA.begin(), KsA.end(), K) != KsA.end()) {
      const std::vector<Setting> exactA =
          pmpp::is_emu() ? only(exact, [](const Setting& s) { return s.rows[0].kind == Planted; }) : exact;
      const std::vector<Setting> softA =
          pmpp::is_emu() ? only(soft, [](const Setting& s) { return s.topP == 0.9f && s.minP == 0.05f; }) : soft;
      const std::vector<int> rowsA = pmpp::is_emu() ? std::vector<int>{rowCounts.back()} : rowCounts;
      t.check("part A (composed): exact rows" + k, [&] { return overSettings(exactA, K, rowsA, repeats, checkPartA); });
      t.check("part A (composed): softmax rows" + k, [&] { return overSettings(softA, K, rowsA, repeats, checkPartA); });
    }
#endif
    t.check("part B (fused): exact rows" + k, [&] { return overSettings(exact, K, rowCounts, repeats, checkPartB); });
    t.check("part B (fused): softmax rows" + k, [&] { return overSettings(soft, K, rowCounts, repeats, checkPartB); });
  }
#if X1_PART_A
  for (int K : pick({33}, {257}, {1, 257, 4096, 32000}))
    t.check("provided labs 14.3, 10.3 and 11.6, K = " + std::to_string(K), [&] { return checkProvided(K, rowCounts.back()); });
#endif
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
#if X1_PART_A
// Time per call: best of `reps` timed runs after one untimed run; each run repeats the call
// `calls` times back to back, with `calls` doubled until one run lasts at least 2 ms.
static double perCall(int reps, const std::function<std::string(int, pmpp::Timing*)>& run, int* calls, std::string* err) {
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

static int bench(const pmpp::Args& a) {
  std::string why = fusedProblem();
  if (why.empty()) why = composedProblem();
  if (!why.empty()) { std::printf("kernels: %s\n", why.c_str()); return 1; }
  const double mem = deviceMemory();
  const double budget = mem > 0 ? mem / 4 : 1e9;   // bytes one shape's buffers may take
  const int rowsList[] = {1, 64};
  const int KList[] = {40, 1024, 32000, 151936};
  const float topP = 0.95f, minP = 0.05f;          // llama.cpp's defaults (notes/today/x1-facts.md D1)
  const double unfusedBytes = 29.0, fusedBytes = 12.0;
  int rc = 0;

  std::printf("\ntop-p %.2f and min-p %.2f on rows of K softmax-shaped logits, best of %d timed runs (ms per call)\n",
              topP, minP, a.reps);
  std::printf("  %4s %7s | %9s %9s %6s | %9s %7s | %9s %7s | %7s\n", "rows", "K", "sort", "chain", "sort%",
              "unfused", "GB/s", "fused", "GB/s", "fused x");
  for (int K : KList)
    for (int rows : rowsList) {
      const size_t cells = (size_t)rows * K;
      if ((double)cells * 37 > budget) {
        std::printf("  %4d %7d | skipped: larger than a quarter of the device memory reported\n", rows, K);
        continue;
      }
      Setting s{topP, minP, "the benchmark's rows", {}};
      for (int r = 0; r < rows; ++r) s.rows.push_back(randomRow(K, 0, 31u + 7u * (uint32_t)r + (uint32_t)K));
      const Batch b = makeBatch(s, 0, rows, K);

      Buf* cand = pmpp::alloc(cells * 4);
      Buf* sorted = pmpp::alloc(cells * 4);
      Buf* pos = pmpp::alloc(cells * 4);
      Buf* sl = pmpp::alloc(cells * 4);
      Buf* sp = pmpp::alloc(cells * 4);
      Buf* n = pmpp::alloc((size_t)rows * 4);
      pmpp::upload(cand, b.cand.data(), cells * 4);
      pmpp::upload(sorted, b.sorted.data(), cells * 4);
      Work w;
      allocWork(w, rows, K, 0);

      // What each timed design produced, checked before its time is reported.
      auto checkOut = [&](bool sortOnly, bool fromSorted) -> std::string {
        std::vector<int> gn((size_t)rows);
        std::vector<float> gs(cells);
        if (sortOnly || !fromSorted) {
          std::vector<int> gp(cells);
          std::vector<uint32_t> gl(cells);
          pmpp::download(pos, gp.data(), cells * 4);
          pmpp::download(sl, gl.data(), cells * 4);
          for (int r = 0; r < rows; ++r)
            for (int i = 0; i < K; ++i)
              if (gp[(size_t)r * K + i] != b.refs[r].order[i] || gl[(size_t)r * K + i] != bitsOf(b.refs[r].sorted[i]))
                return "row " + std::to_string(r) + ": not the stable descending order";
          if (sortOnly) return "";
        }
        pmpp::download(sp, gs.data(), cells * 4);
        pmpp::download(n, gn.data(), gn.size() * 4);
        for (int r = 0; r < rows; ++r) {
          std::string e = checkProbs(b, r, &gs[(size_t)r * K], "");
          if (e.empty()) e = checkN(b, r, gn[r], "");
          if (!e.empty()) return e;
        }
        return "";
      };

      struct Design { const char* name; std::function<std::string(int, pmpp::Timing*)> run; bool sortOnly, fromSorted; };
      const Design designs[] = {
          {"sort_rows", [&](int c, pmpp::Timing* tm) { return runSort(cand, sl, pos, w, rows, K, c, tm); }, true, false},
          {"chain_composed", [&](int c, pmpp::Timing* tm) { return runComposed(cand, pos, sl, sp, n, w, rows, K, topP, minP, c, tm); }, false, false},
          {"filters_unfused", [&](int c, pmpp::Timing* tm) { return runUnfused(sorted, sp, n, w, rows, K, topP, minP, c, tm); }, false, true},
          {"filters_fused", [&](int c, pmpp::Timing* tm) { return runFused(sorted, sp, n, rows, K, topP, minP, c, tm); }, false, true},
      };
      double sec[4] = {0, 0, 0, 0};
      std::string bad;
      for (int d = 0; d < 4 && bad.empty(); ++d) {
        int calls = 0;
        std::string err;
        sec[d] = perCall(a.reps, designs[d].run, &calls, &err);
        if (sec[d] <= 0) bad = std::string(designs[d].name) + " failed: " + err;
        else {
          const std::string e = checkOut(designs[d].sortOnly, designs[d].fromSorted);
          if (!e.empty()) bad = std::string(designs[d].name) + " wrong: " + e;
        }
      }
      pmpp::release(cand); pmpp::release(sorted); pmpp::release(pos); pmpp::release(sl); pmpp::release(sp);
      pmpp::release(n);
      releaseWork(w);
      if (!bad.empty()) { std::printf("  %4d %7d | %s; not reported\n", rows, K, bad.c_str()); rc = 1; continue; }

      const double share = sec[0] / sec[1];
      const double gbU = cells * unfusedBytes / sec[2] / 1e9, gbF = cells * fusedBytes / sec[3] / 1e9;
      std::printf("  %4d %7d | %9.4f %9.4f %5.0f%% | %9.4f %7.1f | %9.4f %7.1f | %6.2fx\n", rows, K, sec[0] * 1e3,
                  sec[1] * 1e3, 100 * share, sec[2] * 1e3, gbU, sec[3] * 1e3, gbF, sec[2] / sec[3]);
      // k: the candidate count when it is what a top-k leaves (40, 1,024), 0 for a whole vocabulary.
      const int k = K <= 1024 ? K : 0;
      pmpp::bench_row(a, "sort_rows").n("rows", rows).n("cols", K).n("k", k).n("seconds", sec[0]).emit();
      pmpp::bench_row(a, "chain_composed").n("rows", rows).n("cols", K).n("k", k).n("seconds", sec[1])
          .n("sort_share", share).emit();
      pmpp::bench_row(a, "filters_unfused").n("rows", rows).n("cols", K).n("k", k).n("seconds", sec[2])
          .n("gb_per_s", gbU).n("bytes_per_token", unfusedBytes).emit();
      pmpp::bench_row(a, "filters_fused").n("rows", rows).n("cols", K).n("k", k).n("seconds", sec[3])
          .n("gb_per_s", gbF).n("bytes_per_token", fusedBytes).n("speedup_vs_unfused", sec[2] / sec[3]).emit();
    }
  std::printf("\nsort: lab 14.3 alone. chain: your launchFilterRows, sort included. sort%%: sort / chain.\n"
              "unfused: the chain after the sort (softmax, top-p mask, min-p count), on the same sorted logits as\n"
              "fused: your fused kernel, not timed with the sort. GB/s counts the minimum traffic of each design:\n"
              "29 bytes per token unfused (softmax reads the logits three times and writes probabilities, top-p\n"
              "reads them and writes mass and keep, min-p reads the logits), 12 fused (two reads, one write).\n"
              "fused x: unfused time / fused time. These are measurements of this machine, at these sizes.\n");
  return rc;
}
#else
static int bench(const pmpp::Args&) { std::printf("this lab has no benchmark.\n"); return 0; }
#endif

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
