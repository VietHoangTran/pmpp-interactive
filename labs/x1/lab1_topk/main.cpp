// Lab X1.1 driver (milestone M5, the top-k stage): tests your row-wise top-k select against a
// CPU reference, and benchmarks it against lab 14.3's full sort and against std::partial_sort
// on one CPU thread. You do not need to edit this file; your work goes in work/.
#include <chrono>
#include <cstring>
#include <iterator>
#include <limits>
#include <numeric>
#include <set>

#include "pmpp/harness.hpp"
using pmpp::Buf;

static const int kBlock = 256;   // threads per block (BLOCK_SIZE in kernel.cu and kernel.metal)

// ---- the entry points: yours in <variant>/, lab 14.3's sort provided ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
bool encodeTopKRows(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> threshold,
                    id<MTLComputePipelineState> gather, id<MTLBuffer> logits, id<MTLBuffer> outIdx,
                    id<MTLBuffer> outVal, id<MTLBuffer> scratch, uint32_t rows, uint32_t cols, uint32_t k);
bool encodeSortRowsDesc(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> descKeys,
                        id<MTLComputePipelineState> sortPass, id<MTLComputePipelineState> gather,
                        id<MTLBuffer> probs, id<MTLBuffer> sortedProbs, id<MTLBuffer> sortedIdx,
                        id<MTLBuffer> scratch, uint32_t rows, uint32_t cols);
using pmpp::metal::buffer;

static std::string dirOf(const std::string& p) { return p.substr(0, p.find_last_of('/')); }
// Lab 14.3's reference kernels, loaded by path (its launch.mm is linked in by CMakeLists.txt).
static const std::string kSortMetal = dirOf(dirOf(PMPP_METAL_SOURCE)) + "/../../ch14/lab3_sortrows/solution/kernel.metal";

struct Pipes { id<MTLComputePipelineState> threshold, gather, keys, pass, sortGather; };
static Pipes pipes() {
  return Pipes{pmpp::metal::pipeline(PMPP_METAL_SOURCE, "topk_threshold"),
               pmpp::metal::pipeline(PMPP_METAL_SOURCE, "topk_gather"),
               pmpp::metal::pipeline(kSortMetal, "desc_keys"),
               pmpp::metal::pipeline(kSortMetal, "sort_rows_pass"),
               pmpp::metal::pipeline(kSortMetal, "gather_rows")};
}

// Why your pipelines cannot run the select as the lab specifies it, or "". The driver's copy of
// the rule a correct launch.mm applies: 256 threads per threadgroup, whole SIMD-groups, and a
// width that divides the 256 bins.
static std::string pipelineProblem() {
  Pipes p = pipes();
  if (!p.threshold || !p.gather) return "your Metal kernels did not build: " + pmpp::take_error();
  if (!p.keys || !p.pass || !p.sortGather) return "lab 14.3's Metal kernels did not build: " + pmpp::take_error();
  for (id<MTLComputePipelineState> q : {p.threshold, p.gather}) {
    char b[200];
    if ((NSUInteger)kBlock > q.maxTotalThreadsPerThreadgroup) {
      std::snprintf(b, sizeof b, "a pipeline here runs at most %lu threads per threadgroup; the lab needs %d",
                    (unsigned long)q.maxTotalThreadsPerThreadgroup, kBlock);
      return b;
    }
    const NSUInteger w = q.threadExecutionWidth;
    if (w == 0 || kBlock % w != 0 || 256 % w != 0) {
      std::snprintf(b, sizeof b, "SIMD-group width %lu does not divide 256", (unsigned long)w);
      return b;
    }
  }
  return "";
}

static std::string launchSelect(Buf* logits, Buf* idx, Buf* val, Buf* scratch, int rows, int cols, int k,
                                int repeat = 1, pmpp::Timing* tm = nullptr) {
  std::string why = pipelineProblem();
  if (!why.empty()) return why;
  Pipes p = pipes();
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    for (int i = 0; i < repeat && ok; ++i)
      ok = encodeTopKRows(enc, p.threshold, p.gather, buffer(logits), buffer(idx), buffer(val), buffer(scratch),
                          (uint32_t)rows, (uint32_t)cols, (uint32_t)k);
  });
  if (tm) *tm = t;
  if (!ok) return "your launch.mm refused to dispatch, although these pipelines can run 256 threads per threadgroup";
  return "";
}
static std::string launchSort(Buf* logits, Buf* sortedVals, Buf* sortedIdx, Buf* scratch, int rows, int cols,
                              int repeat = 1, pmpp::Timing* tm = nullptr) {
  std::string why = pipelineProblem();
  if (!why.empty()) return why;
  Pipes p = pipes();
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    for (int i = 0; i < repeat && ok; ++i)
      ok = encodeSortRowsDesc(enc, p.keys, p.pass, p.sortGather, buffer(logits), buffer(sortedVals), buffer(sortedIdx),
                              buffer(scratch), (uint32_t)rows, (uint32_t)cols);
  });
  if (tm) *tm = t;
  if (!ok) return "lab 14.3's launch.mm refused to dispatch on this device";
  return "";
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
void launchTopKRows(const float* logits, int* outIdx, float* outVal, unsigned int* scratch, int rows, int cols,
                    int k);
void launchSortRowsDesc(const float* probs, float* sortedProbs, int* sortedIdx, unsigned int* scratch, int rows,
                        int cols);

static std::string pipelineProblem() {
  const int most = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  return kBlock <= most ? "" : "this device runs at most " + std::to_string(most) + " threads per block; the lab needs 256";
}
static std::string launchSelect(Buf* logits, Buf* idx, Buf* val, Buf* scratch, int rows, int cols, int k,
                                int repeat = 1, pmpp::Timing* tm = nullptr) {
  std::string why = pipelineProblem();
  if (!why.empty()) return why;
  pmpp::Timing t = pmpp::run([&](void*) {
    for (int i = 0; i < repeat; ++i)
      launchTopKRows((const float*)pmpp::raw(logits), (int*)pmpp::raw(idx), (float*)pmpp::raw(val),
                     (unsigned int*)pmpp::raw(scratch), rows, cols, k);
  });
  if (tm) *tm = t;
  return "";
}
static std::string launchSort(Buf* logits, Buf* sortedVals, Buf* sortedIdx, Buf* scratch, int rows, int cols,
                              int repeat = 1, pmpp::Timing* tm = nullptr) {
  std::string why = pipelineProblem();
  if (!why.empty()) return why;
  pmpp::Timing t = pmpp::run([&](void*) {
    for (int i = 0; i < repeat; ++i)
      launchSortRowsDesc((const float*)pmpp::raw(logits), (float*)pmpp::raw(sortedVals), (int*)pmpp::raw(sortedIdx),
                         (unsigned int*)pmpp::raw(scratch), rows, cols);
  });
  if (tm) *tm = t;
  return "";
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

// ---------------------------------------------------------------- bits and the key
static uint32_t bitsOf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float fromBits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// Lab 14.3's key, ascending: -0.0 is +0.0, then CUB's transform (ch14-facts A3).
static uint32_t orderedKey(uint32_t u) {
  if (u == 0x80000000u) u = 0u;
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// ---------------------------------------------------------------- CPU reference
// std::stable_sort of the column indices by key, largest first (equal keys keep increasing
// column order), the first min(k, cols), then put back in column order.
static void refTopK(const float* row, int cols, int k, std::vector<int>& out) {
  std::vector<uint32_t> key((size_t)cols);
  for (int c = 0; c < cols; ++c) key[c] = orderedKey(bitsOf(row[c]));
  std::vector<int> order((size_t)cols);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return key[a] > key[b]; });
  out.assign(order.begin(), order.begin() + std::min(k, cols));
  std::sort(out.begin(), out.end());
}

// ---------------------------------------------------------------- test data
// A 32-bit hash, so that the data do not depend on the platform's random library.
static uint32_t mix(uint32_t x) {
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
  return x;
}
static float unit24(uint32_t h) { return (float)(h >> 8) * (1.0f / 16777216.0f); }   // [0, 1)

enum Shape { Uniform, Peaked, AllEqual, TiesAtK, Zeros, Top8, Top16, Top24, Edges, NegativeOnly };
struct Recipe { Shape shape; const char* what; };
static const Recipe kRecipes[] = {
    {Uniform, "logits uniform in [-12, 12]"},
    {Peaked, "peaked softmax-shaped row (probabilities)"},
    {AllEqual, "every logit equal"},
    {TiesAtK, "exact ties at the k-th value at warp and round seams"},
    {Zeros, "-0.0 and +0.0 interleaved at the k-th value"},
    {Top8, "keys sharing their top 8 bits"},
    {Top16, "keys sharing their top 16 bits"},
    {Top24, "keys sharing their top 24 bits"},
    {Edges, "bytes 0x00 and 0xFF at every digit, huge magnitudes"},
    {NegativeOnly, "negative logits only"},
};
static const int kRecipeCount = (int)(sizeof kRecipes / sizeof kRecipes[0]);

// softmax(12 * u) for u uniform in [-1, 1], in double, rounded to float: positive, peaked.
static void softmaxRow(float* row, int cols, uint32_t seed) {
  std::vector<float> u = pmpp::random_floats((size_t)cols, seed);
  double m = -1e300, s = 0;
  for (int c = 0; c < cols; ++c) m = std::max(m, 12.0 * u[c]);
  std::vector<double> e((size_t)cols);
  for (int c = 0; c < cols; ++c) s += (e[c] = std::exp(12.0 * u[c] - m));
  for (int c = 0; c < cols; ++c) row[c] = (float)(e[c] / s);
}

// Columns planted with one tied value: pairs either side of a warp seam (31, 32 and 63, 64),
// of the first round seam (255, 256), the two ends (0, cols-1), and runs of 40 equal values
// across a warp seam (12..51), the first round seam (236..275) and a later one (2028..2067).
static std::vector<int> plantedColumns(int cols) {
  std::set<int> s = {31, 32, 63, 64, 255, 256, 0, cols - 1};
  for (int start : {12, 236, 2028})
    for (int c = start; c < start + 40; ++c) s.insert(c);
  std::vector<int> v;
  for (int c : s)
    if (c >= 0 && c < cols) v.push_back(c);
  return v;
}

static void fillRow(Shape shape, float* row, int cols, int k, uint32_t seed) {
  const int kk = std::min(k, cols);
  switch (shape) {
    case Uniform: {
      std::vector<float> u = pmpp::random_floats((size_t)cols, seed);   // 65,536 levels: many ties
      for (int c = 0; c < cols; ++c) row[c] = 12.0f * u[c];
      break;
    }
    case Peaked: softmaxRow(row, cols, seed); break;
    case AllEqual:
      for (int c = 0; c < cols; ++c) row[c] = 0.5f;
      break;
    case TiesAtK: {
      // Exactly `above` columns outside the planted set are larger than the tied value v, the
      // rest smaller, so the k-th value is v and `budget` of the ties are kept.
      std::vector<int> planted = plantedColumns(cols);
      std::vector<bool> isPlanted((size_t)cols, false);
      for (int c : planted) isPlanted[c] = true;
      const int p = (int)planted.size(), others = cols - p;
      int budget = seed % 3 == 0 ? 1 : seed % 3 == 1 ? std::max(1, p / 2) : std::max(1, p - 1);
      budget = std::min(budget, kk);
      const int above = std::min(others, kk - std::min(budget, p));
      std::vector<int> rest;
      for (int c = 0; c < cols; ++c)
        if (!isPlanted[c]) rest.push_back(c);
      // Which non-planted columns are above: a hash-ordered choice, so they are spread out.
      std::vector<std::pair<uint32_t, int>> order;
      for (int c : rest) order.push_back({mix(seed * 977u + (uint32_t)c), c});
      std::sort(order.begin(), order.end());
      const float v = 0.75f;
      for (int c : planted) row[c] = v;
      for (int i = 0; i < (int)order.size(); ++i) {
        const float u = unit24(mix(order[i].first));
        row[order[i].second] = i < above ? v + 1.0f + 5.0f * u : v - 1.0f - 5.0f * u;
      }
      break;
    }
    case Zeros: {
      // Every third column a zero, alternately +0.0 and -0.0; about kk/2 positives, so the
      // k-th value is a zero whenever there are enough zeros.
      int positives = kk / 2;
      for (int c = 0; c < cols; ++c) {
        const uint32_t h = mix(seed + 31u * (uint32_t)c);
        if (c % 3 == 0) {
          row[c] = fromBits((c / 3) % 2 ? 0x80000000u : 0u);
        } else if (positives > 0 && (h & 3u) != 0) {
          row[c] = 1.0f + unit24(h);
          --positives;
        } else {
          row[c] = -1.0f - unit24(h);
        }
      }
      break;
    }
    case Top8: case Top16: case Top24: {
      // Positive or negative by seed; the top 8, 16 or 24 bits of every key are equal.
      const uint32_t sign = (seed & 1u) ? 0x80000000u : 0u;
      for (int c = 0; c < cols; ++c) {
        const uint32_t h = mix(seed * 131u + (uint32_t)c);
        uint32_t u;
        if (shape == Top8) u = 0x40000000u | (h & 0x00FFFFFFu);          // in [2, 4): sign + exponent bits shared
        else if (shape == Top16) u = 0x40490000u | (h & 0x0000FFFFu);
        else u = 0x40490F00u | (h & 0x000000FFu);                         // 256 distinct values: ties
        row[c] = fromBits(u | sign);
      }
      break;
    }
    case Edges: {
      for (int c = 0; c < cols; ++c) {
        const uint32_t h = mix(seed * 7u + (uint32_t)c);
        const uint32_t b0 = (h & 1u) ? 0xFFu : 0u, b1 = (h & 2u) ? 0xFFu : 0u, b2 = (h & 4u) ? 0xFFu : 0u;
        const uint32_t pick = (h >> 8) % 3u;
        uint32_t u;
        if (pick == 0) u = 0x3F000000u | (b2 << 16) | (b1 << 8) | b0;              // in [0.5, 2)
        else if (pick == 1) u = 0x7F000000u | ((b2 & 0x7Fu) << 16) | (b1 << 8) | b0;   // about 1.7e38 up to FLT_MAX: key byte 0xFF
        else u = 0xFF000000u | ((b2 & 0x7Fu) << 16) | (b1 << 8) | b0;          // the negatives of those: key byte 0x00
        row[c] = fromBits(u);
      }
      break;
    }
    case NegativeOnly: {
      std::vector<float> u = pmpp::random_floats((size_t)cols, seed);
      for (int c = 0; c < cols; ++c) row[c] = -0.001f - 12.0f * std::fabs(u[c]);
      break;
    }
  }
}

// A batch of `rows` rows of `cols` logits for one k. Row r follows recipe (start + r) % count,
// so a kernel that reads or writes the wrong row meets a row of another kind.
struct Batch {
  int rows, cols, k;
  std::vector<float> x;
  std::vector<std::string> what;
};
static Batch makeBatch(int start, int rows, int cols, int k) {
  Batch b{rows, cols, k, std::vector<float>((size_t)rows * cols), {}};
  for (int r = 0; r < rows; ++r) {
    const Recipe& one = kRecipes[(start + r) % kRecipeCount];
    fillRow(one.shape, &b.x[(size_t)r * cols], cols, k,
            7919u * (uint32_t)cols + 104729u * (uint32_t)(start + r) + 31u * (uint32_t)k + 17u);
    b.what.push_back(one.what);
  }
  return b;
}

// ---------------------------------------------------------------- one select, checked
static const int kGuard = 64;                     // entries past the end of every output
static const uint32_t kGuardBits = 0xDEADBEEFu;   // what they hold before the launch
static const int kGuardIdx = -77;
static const int kUnwrittenIdx = -1;              // outIdx before the launch
static const uint32_t kUnwrittenBits = 0xFFC0FFEEu;   // outVal before the launch: a NaN no row holds
static const uint32_t kScratchFill = 0xCDCDCDCDu;

static std::string where(const Batch& b, int r) {
  return std::to_string(b.rows) + " rows of " + std::to_string(b.cols) + ", k " + std::to_string(b.k) + ", row " +
         std::to_string(r) + " (" + b.what[r] + ")";
}
static std::string hex(uint32_t u) { char s[16]; std::snprintf(s, sizeof s, "0x%08X", u); return s; }
static std::string val(uint32_t u) {
  char s[48];
  std::snprintf(s, sizeof s, "%.9g [%s]", (double)fromBits(u), hex(u).c_str());
  return s;
}

// What is wrong with one row of outIdx, or "".
static std::string checkIdxRow(const Batch& b, int r, const int* got, const std::vector<std::vector<int>>& refs) {
  const int kk = std::min(b.k, b.cols);
  const float* row = &b.x[(size_t)r * b.cols];
  const std::vector<int>& want = refs[r];
  for (int j = 0; j < kk; ++j) {
    if (got[j] == kUnwrittenIdx)
      return where(b, r) + ": outIdx[" + std::to_string(j) + "] was never written; every row gets exactly min(k, cols) = " +
             std::to_string(kk) + " pairs";
    if (got[j] < 0 || got[j] >= b.cols)
      return where(b, r) + ": outIdx[" + std::to_string(j) + "] = " + std::to_string(got[j]) + ", not a column of this row";
    if (j > 0 && got[j] <= got[j - 1])
      return where(b, r) + ": outIdx[" + std::to_string(j - 1) + ".." + std::to_string(j) + "] = " +
             std::to_string(got[j - 1]) + ", " + std::to_string(got[j]) +
             ": the pairs must be in ascending column order, each column once (are slots ranked in column order?)";
  }
  if (std::equal(want.begin(), want.end(), got)) return "";
  for (int o = 0; o < b.rows; ++o)
    if (o != r && std::equal(refs[o].begin(), refs[o].end(), got))
      return where(b, r) + ": this is the answer for row " + std::to_string(o) + " (is every kernel offset by its row?)";
  // Compare as sets: the first column in one and not the other.
  std::vector<int> g(got, got + kk), extra, missing;
  std::set_difference(g.begin(), g.end(), want.begin(), want.end(), std::back_inserter(extra));
  std::set_difference(want.begin(), want.end(), g.begin(), g.end(), std::back_inserter(missing));
  uint32_t kth = 0xFFFFFFFFu;   // the smallest key in the reference set: the k-th key
  int kthCol = want[0];
  for (int c : want)
    if (orderedKey(bitsOf(row[c])) < kth) { kth = orderedKey(bitsOf(row[c])); kthCol = c; }
  std::string err = where(b, r) + ": ";
  if (!extra.empty()) {
    const int c = extra[0];
    const uint32_t key = orderedKey(bitsOf(row[c]));
    err += "column " + std::to_string(c) + ", " + val(bitsOf(row[c])) + ", is not among the " + std::to_string(kk) + " largest";
    if (key == kth) {
      err += ": it ties with the k-th value " + val(bitsOf(row[kthCol])) +
             " but lower columns hold that value too (ties go to the lowest columns: is each tie ranked among the "
             "row's ties in column order, across warps and rounds?)";
      if ((bitsOf(row[c]) & 0x7FFFFFFFu) == 0) err += "; -0.0 and +0.0 are the same key";
    } else if (key < kth) {
      err += " (its key is below the k-th key " + hex(kth) + ")";
    }
  } else if (!missing.empty()) {
    const int c = missing[0];
    err += "column " + std::to_string(c) + ", " + val(bitsOf(row[c])) + ", is missing";
  }
  return err;
}

static std::string checkSelect(const Batch& b) {
  const int rows = b.rows, cols = b.cols, k = b.k, kk = std::min(k, cols);
  const size_t n = (size_t)rows * cols, out = (size_t)rows * k;
  std::vector<uint32_t> in(n);
  for (size_t i = 0; i < n; ++i) in[i] = bitsOf(b.x[i]);
  std::vector<std::vector<int>> refs((size_t)rows);
  for (int r = 0; r < rows; ++r) refTopK(&b.x[(size_t)r * cols], cols, k, refs[r]);

  Buf* logits = pmpp::alloc(n * 4);
  Buf* idx = pmpp::alloc((out + kGuard) * 4);
  Buf* vals = pmpp::alloc((out + kGuard) * 4);
  Buf* scratch = pmpp::alloc(((size_t)4 * rows + kGuard) * 4);
  pmpp::upload(logits, in.data(), n * 4);
  std::vector<int> idxGot(out + kGuard, kGuardIdx);
  std::vector<uint32_t> valGot(out + kGuard, kGuardBits), scr((size_t)4 * rows + kGuard, kGuardBits);
  std::fill(idxGot.begin(), idxGot.begin() + out, kUnwrittenIdx);
  std::fill(valGot.begin(), valGot.begin() + out, kUnwrittenBits);
  std::fill(scr.begin(), scr.begin() + 4 * rows, kScratchFill);
  pmpp::upload(idx, idxGot.data(), idxGot.size() * 4);
  pmpp::upload(vals, valGot.data(), valGot.size() * 4);
  pmpp::upload(scratch, scr.data(), scr.size() * 4);

  std::string err = launchSelect(logits, idx, vals, scratch, rows, cols, k);
  if (err.empty()) err = pmpp::take_error();
  if (err.empty()) {
    std::vector<uint32_t> after(n);
    pmpp::download(idx, idxGot.data(), idxGot.size() * 4);
    pmpp::download(vals, valGot.data(), valGot.size() * 4);
    pmpp::download(scratch, scr.data(), scr.size() * 4);
    pmpp::download(logits, after.data(), n * 4);
    for (int r = 0; r < rows && err.empty(); ++r) err = checkIdxRow(b, r, &idxGot[(size_t)r * k], refs);
    for (int r = 0; r < rows && err.empty(); ++r)
      for (int j = 0; j < kk; ++j) {
        const uint32_t g = valGot[(size_t)r * k + j], w = in[(size_t)r * cols + refs[r][j]];
        if (g == w) continue;
        err = where(b, r) + ": outVal[" + std::to_string(j) + "] has the bits " + hex(g) + ", expected " + hex(w) +
              " (the logit at column " + std::to_string(refs[r][j]) + ")";
        if (g == kUnwrittenBits) err += ": never written";
        else err += ": copy the 32 bits of the logit at that column, do not convert them";
        break;
      }
    for (int r = 0; r < rows && err.empty(); ++r)
      for (int j = kk; j < k; ++j)
        if (idxGot[(size_t)r * k + j] != kUnwrittenIdx || valGot[(size_t)r * k + j] != kUnwrittenBits) {
          err = where(b, r) + ": slot " + std::to_string(j) + " was written, but a row of " + std::to_string(cols) +
                " columns has only " + std::to_string(kk) + " pairs; slots past min(k, cols) stay untouched";
          break;
        }
    for (int g = 0; g < kGuard && err.empty(); ++g) {
      if (idxGot[out + g] != kGuardIdx) err = "wrote outIdx[" + std::to_string(out + g) + "], past rows * k entries";
      else if (valGot[out + g] != kGuardBits) err = "wrote outVal[" + std::to_string(out + g) + "], past rows * k entries";
      else if (scr[(size_t)4 * rows + g] != kGuardBits)
        err = "wrote scratch[" + std::to_string(4 * rows + g) + "], past the 4 x rows entries it holds";
    }
    for (size_t i = 0; i < n && err.empty(); ++i)
      if (after[i] != in[i]) err = "logits[" + std::to_string(i) + "] changed: the input is read-only";
  }
  pmpp::release(logits); pmpp::release(idx); pmpp::release(vals); pmpp::release(scratch);
  return err;
}

// The provided baseline: lab 14.3's sort, then the first k of each row. Its first k, put in
// column order, must be the reference set (the sort is stable, so ties come lowest column first).
static std::string checkSortBaseline(const Batch& b) {
  const int rows = b.rows, cols = b.cols, kk = std::min(b.k, b.cols);
  const size_t n = (size_t)rows * cols;
  Buf* logits = pmpp::alloc(n * 4);
  Buf* sv = pmpp::alloc(n * 4);
  Buf* si = pmpp::alloc(n * 4);
  Buf* scratch = pmpp::alloc(4 * n * 4);
  pmpp::upload(logits, b.x.data(), n * 4);
  std::string err = launchSort(logits, sv, si, scratch, rows, cols);
  if (err.empty()) err = pmpp::take_error();
  if (err.empty()) {
    std::vector<int> got(n);
    pmpp::download(si, got.data(), n * 4);
    std::vector<int> ref;
    for (int r = 0; r < rows && err.empty(); ++r) {
      refTopK(&b.x[(size_t)r * cols], cols, b.k, ref);
      std::vector<int> first(got.begin() + (size_t)r * cols, got.begin() + (size_t)r * cols + kk);
      std::sort(first.begin(), first.end());
      if (first != ref) err = where(b, r) + ": the provided sort's first k differ from the reference (a problem with the provided code, not yours)";
    }
  }
  pmpp::release(logits); pmpp::release(sv); pmpp::release(si); pmpp::release(scratch);
  return err;
}

// ---------------------------------------------------------------- tests
static std::vector<int> pick(const std::vector<int>& race, const std::vector<int>& emu, const std::vector<int>& dev) {
  const std::string b = pmpp::backend();
  return b == "race" ? race : b == "emu" ? emu : dev;
}

// Every k of the list that is at most cols + 5. On the emulators, where every launch costs
// about as much as hundreds of thread switches, k = 2 and k = cols are dropped (k = 1 and
// k = cols + 5 cover the same edges) and k = 1000 is kept for the largest row only.
static std::vector<int> kList(int cols) {
  std::vector<int> ks;
  for (int k : {1, 2, 40, 256, 1000, cols, cols + 5}) {
    if (k > cols + 5 || std::find(ks.begin(), ks.end(), k) != ks.end()) continue;
    if (pmpp::is_emu() && (k == 2 || (k == cols && k > 1) || (k == 1000 && cols < 4000))) continue;
    ks.push_back(k);
  }
  return ks;
}

// Every recipe in every row count, except on `emu`, where every case runs under three thread
// orders and every launch costs hundreds of thread switches: there each row count runs one
// batch per (cols, k) case, starting at a recipe that moves on with the case (`turn`), so every
// recipe still meets several sizes and several k.
static std::string overBatches(int cols, int k, int turn, const std::vector<int>& rowCounts) {
  const bool fewer = std::string(pmpp::backend()) == "emu";
  for (int rows : rowCounts) {
    const int first = fewer ? (turn * rows + turn) % kRecipeCount : 0;
    const int last = fewer ? first + 1 : kRecipeCount;
    for (int start = first; start < last; start += rows) {
      std::string err = checkSelect(makeBatch(start, rows, cols, k));
      if (!err.empty()) return err;
    }
  }
  return "";
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::vector<int> rowCounts = pick({2}, {1, 3}, {1, 4, 7});
  const std::vector<int> colCounts = pick({1, 33, 257, 1000, 2100}, {1, 2, 31, 255, 256, 257, 1000, 4099},
                                          {1, 2, 31, 255, 256, 257, 1000, 4096, 32000, 50257, 151936});
  const std::string why = pipelineProblem();
  if (!why.empty()) {   // reported once, not once per shape
    t.check("top-k: kernels usable", [&] { return why; });
    return t.finish();
  }
  int turn = 0;
  for (int cols : colCounts)
    for (int k : kList(cols)) {
      const int my = turn++;
      t.check("top-k, cols " + std::to_string(cols) + ", k " + std::to_string(k),
              [&] { return overBatches(cols, k, my, rowCounts); });
    }
  // The provided baseline, on a few shapes: it was tested in lab 14.3; this checks the link.
  for (int cols : pick({257}, {257}, {257, 4096, 151936})) {
    const int rows = pick({2}, {3}, {7})[0];
    t.check("provided sort baseline, first 40 of cols " + std::to_string(cols),
            [&] { return checkSortBaseline(makeBatch(0, rows, cols, 40)); });
  }
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
// Time per call: best of `reps` timed runs; each run repeats the call `calls` times back to back,
// with `calls` doubled (untimed) until one run lasts at least `minSeconds`.
static double perCall(int reps, double minSeconds, const std::function<std::string(int, pmpp::Timing*)>& run,
                      int* calls, std::string* err) {
  *calls = 1;
  pmpp::Timing tm{0, 0};
  for (;;) {
    *err = run(*calls, &tm);
    if (err->empty()) *err = pmpp::take_error();
    if (!err->empty()) return -1;
    if (tm.best() >= minSeconds || *calls >= 65536) break;
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

// The CPU baseline, as llama.cpp's host top-k does it for k up to 128 (ch14-facts D3): an array
// of (id, logit) pairs, std::partial_sort of its first k by logit, largest first. One CPU thread,
// one row after another. std::partial_sort is not stable: among equal logits its choice is the
// library's, so the check below compares values, not columns.
struct TokenData { int id; float logit; };
static void cpuPartialSortRows(const std::vector<float>& x, int rows, int cols, int k, std::vector<TokenData>& work,
                               std::vector<TokenData>& top) {
  const int kk = std::min(k, cols);
  top.resize((size_t)rows * kk);
  work.resize((size_t)cols);
  for (int r = 0; r < rows; ++r) {
    const float* row = &x[(size_t)r * cols];
    for (int c = 0; c < cols; ++c) work[c] = TokenData{c, row[c]};
    std::partial_sort(work.begin(), work.begin() + kk, work.end(),
                      [](const TokenData& a, const TokenData& b) { return a.logit > b.logit; });
    std::copy(work.begin(), work.begin() + kk, top.begin() + (size_t)r * kk);
  }
}

// Bytes each design must move per row, counted, not measured (section X1.4's count): the select
// reads each 4-byte key four times to count and once to gather, and writes 8 bytes per kept
// pair; lab 14.3's sort runs four passes, each reading the key to count and reading and writing
// the 8-byte (key, index) pair to scatter: 80 bytes per key.
static double selectBytesPerKey(int cols, int k) { return 20.0 + 8.0 * std::min(k, cols) / cols; }
static const double kSortBytesPerKey = 80.0;

static int bench(const pmpp::Args& a) {
  const std::string why = pipelineProblem();
  if (!why.empty()) { std::printf("top-k: %s\n", why.c_str()); return 1; }
  const double mem = deviceMemory();
  const double budget = mem > 0 ? mem / 4 : 1e9;   // bytes one shape may take on the device
  int rc = 0;

  std::printf("\ntop-k of logits uniform in [-12, 12]: your select against lab 14.3's sort and std::partial_sort, best of %d runs\n",
              a.reps);
  std::printf("  %4s %7s %5s | %10s %10s %6s | %10s %6s | %10s | %8s %8s\n", "rows", "cols", "k", "select ms",
              "rows/s", "B/key", "sort ms", "B/key", "CPU ms", "sort/sel", "CPU/sel");
  for (int cols : {32000, 151936})
    for (int rows : {1, 8, 64}) {
      const size_t n = (size_t)rows * cols;
      if ((double)n * 28 > budget) {   // logits, sorted values and indices, 4 x scratch: 28 bytes per key
        std::printf("  %4d %7d       | skipped: larger than a quarter of the device memory reported\n", rows, cols);
        continue;
      }
      std::vector<float> x(n);
      for (size_t i = 0; i < n; ++i) x[i] = 24.0f * unit24(mix((uint32_t)i * 2654435761u + (uint32_t)cols)) - 12.0f;
      Buf* logits = pmpp::alloc(n * 4);
      pmpp::upload(logits, x.data(), n * 4);

      // The sort, once per shape: its time does not depend on k. Checked before it is reported.
      Buf* sv = pmpp::alloc(n * 4);
      Buf* si = pmpp::alloc(n * 4);
      Buf* sscr = pmpp::alloc(4 * n * 4);
      int sortCalls = 0;
      std::string err;
      const double sortS = perCall(a.reps, 0.02, [&](int c, pmpp::Timing* tm) {
        return launchSort(logits, sv, si, sscr, rows, cols, c, tm); }, &sortCalls, &err);
      std::vector<int> sortedIdx(n);
      if (sortS > 0) pmpp::download(si, sortedIdx.data(), n * 4);
      else std::printf("  %4d %7d       | sort failed: %s\n", rows, cols, err.c_str());

      for (int k : {1, 40, 1024}) {
        const size_t out = (size_t)rows * k;
        Buf* idx = pmpp::alloc(out * 4);
        Buf* vals = pmpp::alloc(out * 4);
        Buf* scr = pmpp::alloc((size_t)4 * rows * 4);
        int calls = 0;
        std::string bad;
        const double s = perCall(a.reps, 0.002, [&](int c, pmpp::Timing* tm) {
          return launchSelect(logits, idx, vals, scr, rows, cols, k, c, tm); }, &calls, &err);
        if (s <= 0) bad = "select failed: " + err;
        // Check what the timed calls produced, and the sort's first k, before reporting.
        std::vector<int> got(out), ref;
        if (bad.empty()) pmpp::download(idx, got.data(), out * 4);
        for (int r = 0; r < rows && bad.empty(); ++r) {
          refTopK(&x[(size_t)r * cols], cols, k, ref);
          if (!std::equal(ref.begin(), ref.end(), got.begin() + (size_t)r * k)) bad = "select row " + std::to_string(r) + " wrong";
          if (sortS > 0) {
            std::vector<int> first(sortedIdx.begin() + (size_t)r * cols, sortedIdx.begin() + (size_t)r * cols + k);
            std::sort(first.begin(), first.end());
            if (first != ref) bad = "sort row " + std::to_string(r) + " wrong";
          }
        }
        pmpp::release(idx); pmpp::release(vals); pmpp::release(scr);

        // CPU baseline (a host measurement), checked by value against the reference.
        std::vector<TokenData> work, top;
        cpuPartialSortRows(x, rows, cols, k, work, top);   // untimed run
        double cpuS = 1e30;
        for (int r = 0; r < a.reps; ++r) {
          int inner = 0;
          const auto t0 = std::chrono::steady_clock::now();
          double el = 0;
          do {
            cpuPartialSortRows(x, rows, cols, k, work, top);
            ++inner;
            el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
          } while (el < 0.002);
          cpuS = std::min(cpuS, el / inner);
        }
        for (int r = 0; r < rows && bad.empty(); ++r) {
          refTopK(&x[(size_t)r * cols], cols, k, ref);
          std::vector<float> want, have;
          for (int c : ref) want.push_back(x[(size_t)r * cols + c]);
          for (int j = 0; j < (int)ref.size(); ++j) have.push_back(top[(size_t)r * ref.size() + j].logit);
          std::sort(want.begin(), want.end());
          std::sort(have.begin(), have.end());
          if (want != have) bad = "CPU baseline row " + std::to_string(r) + " wrong";
        }
        if (!bad.empty()) { std::printf("  %4d %7d %5d | %s; not reported\n", rows, cols, k, bad.c_str()); rc = 1; continue; }

        const double selB = selectBytesPerKey(cols, k);
        if (sortS > 0)
          std::printf("  %4d %7d %5d | %10.4f %10.0f %6.1f | %10.4f %6.0f | %10.3f | %7.2fx %7.2fx\n", rows, cols, k,
                      s * 1e3, rows / s, selB, sortS * 1e3, kSortBytesPerKey, cpuS * 1e3, sortS / s, cpuS / s);
        else
          std::printf("  %4d %7d %5d | %10.4f %10.0f %6.1f | %10s %6s | %10.3f | %8s %7.2fx\n", rows, cols, k, s * 1e3,
                      rows / s, selB, "-", "-", cpuS * 1e3, "-", cpuS / s);
        pmpp::bench_row(a, "topk_select").n("rows", rows).n("cols", cols).n("k", k).n("seconds", s)
            .n("rows_per_s", rows / s).n("bytes_per_key", selB).n("calls", calls).emit();
        if (sortS > 0)
          pmpp::bench_row(a, "full_sort_then_first_k").n("rows", rows).n("cols", cols).n("k", k).n("seconds", sortS)
              .n("rows_per_s", rows / sortS).n("bytes_per_key", kSortBytesPerKey).n("calls", sortCalls)
              .n("sort_over_select", sortS / s).emit();
        pmpp::bench_row(a, "cpu_partial_sort").n("rows", rows).n("cols", cols).n("k", k).n("seconds", cpuS)
            .n("rows_per_s", rows / cpuS).n("cpu_over_select", cpuS / s).emit();
      }
      pmpp::release(logits); pmpp::release(sv); pmpp::release(si); pmpp::release(sscr);
    }
  std::printf("\nselect ms: your two kernels, per call, timed on the device; each run repeats the call until it\n"
              "lasts at least 2 ms. sort ms: lab 14.3's whole sort (keys, four passes, gather), runs of at least\n"
              "20 ms; the first k of each sorted row are the top k, so taking them costs nothing more.\n"
              "B/key: bytes per key each design must move, counted from its passes, not measured.\n"
              "CPU: std::partial_sort of (id, logit) pairs per row on one CPU thread, timed on the host with a\n"
              "steady clock. A ratio above 1 means your select took less time.\n"
              "These are measurements of this machine, at these sizes.\n");
  return rc;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
