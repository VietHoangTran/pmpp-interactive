// Lab 14.3 driver (milestone M5): tests your row-wise descending stable sort against a CPU
// reference, and benchmarks it against std::stable_sort on one CPU thread. You do not need
// to edit this file; your work goes in work/.
#include <chrono>
#include <cstring>
#include <limits>
#include <numeric>

#include "pmpp/harness.hpp"
using pmpp::Buf;

static const int kBlock = 256;   // threads per block (BLOCK_SIZE in kernel.cu and kernel.metal)

// ---- the entry point implemented in <variant>/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
bool encodeSortRowsDesc(id<MTLComputeCommandEncoder> enc,
                        id<MTLComputePipelineState> descKeys,
                        id<MTLComputePipelineState> sortPass,
                        id<MTLComputePipelineState> gather,
                        id<MTLBuffer> probs, id<MTLBuffer> sortedProbs, id<MTLBuffer> sortedIdx,
                        id<MTLBuffer> scratch, uint32_t rows, uint32_t cols);
using pmpp::metal::buffer;

struct Pipes { id<MTLComputePipelineState> keys, pass, gather; };
static Pipes pipes() {
  return Pipes{pmpp::metal::pipeline(PMPP_METAL_SOURCE, "desc_keys"),
               pmpp::metal::pipeline(PMPP_METAL_SOURCE, "sort_rows_pass"),
               pmpp::metal::pipeline(PMPP_METAL_SOURCE, "gather_rows")};
}

// Why these pipelines cannot run the sort as the lab specifies it, or "". The driver's own
// copy of the rule a correct launch.mm applies: 256 threads per threadgroup for all three,
// and, for sort_rows_pass, whole SIMD-groups and no more SIMD-groups than a SIMD-group has lanes.
static std::string pipelineProblem() {
  Pipes p = pipes();
  if (!p.keys || !p.pass || !p.gather) return "the Metal kernels did not build: " + pmpp::take_error();
  char b[200];
  const NSUInteger most = std::min(p.keys.maxTotalThreadsPerThreadgroup,
                                   std::min(p.pass.maxTotalThreadsPerThreadgroup, p.gather.maxTotalThreadsPerThreadgroup));
  if ((NSUInteger)kBlock > most) {
    std::snprintf(b, sizeof b, "a pipeline here runs at most %lu threads per threadgroup; the lab needs %d",
                  (unsigned long)most, kBlock);
    return b;
  }
  const NSUInteger w = p.pass.threadExecutionWidth;
  if (w == 0 || kBlock % w != 0 || (NSUInteger)kBlock / w > w) {
    std::snprintf(b, sizeof b, "SIMD-group width %lu: %d threads are not 1 to %lu whole SIMD-groups",
                  (unsigned long)w, kBlock, (unsigned long)w);
    return b;
  }
  return "";
}

// Enqueues `repeat` whole sorts in one timed run. Empty string when it ran.
static std::string launch(Buf* probs, Buf* sortedProbs, Buf* sortedIdx, Buf* scratch, int rows, int cols,
                          int repeat = 1, pmpp::Timing* tm = nullptr) {
  std::string why = pipelineProblem();
  if (!why.empty()) return why;
  Pipes p = pipes();
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    for (int i = 0; i < repeat && ok; ++i)
      ok = encodeSortRowsDesc(enc, p.keys, p.pass, p.gather, buffer(probs), buffer(sortedProbs),
                              buffer(sortedIdx), buffer(scratch), (uint32_t)rows, (uint32_t)cols);
  });
  if (tm) *tm = t;
  if (!ok) return "your launch.mm refused to dispatch, although these pipelines can run 256 threads per threadgroup";
  return "";
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
void launchSortRowsDesc(const float* probs, float* sortedProbs, int* sortedIdx, unsigned int* scratch,
                        int rows, int cols);

static std::string pipelineProblem() {
  const int most = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  return kBlock <= most ? "" : "this device runs at most " + std::to_string(most) + " threads per block; the lab needs 256";
}
static std::string launch(Buf* probs, Buf* sortedProbs, Buf* sortedIdx, Buf* scratch, int rows, int cols,
                          int repeat = 1, pmpp::Timing* tm = nullptr) {
  std::string why = pipelineProblem();
  if (!why.empty()) return why;
  pmpp::Timing t = pmpp::run([&](void*) {
    for (int i = 0; i < repeat; ++i)
      launchSortRowsDesc((const float*)pmpp::raw(probs), (float*)pmpp::raw(sortedProbs),
                         (int*)pmpp::raw(sortedIdx), (unsigned int*)pmpp::raw(scratch), rows, cols);
  });
  if (tm) *tm = t;
  return "";
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

// ---------------------------------------------------------------- bits and the key
static uint32_t bitsOf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float fromBits(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static bool isNaNBits(uint32_t u) { return (u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu) != 0u; }
static bool isZeroBits(uint32_t u) { return (u & 0x7FFFFFFFu) == 0u; }

// The contract's order: CUB's documented transform for floats (ch14-facts A3, E1), with
// -0.0 mapped to +0.0 first, then inverted for descending order. Ascending order of this key
// is the order every row must come out in; equal keys keep increasing column order.
static uint32_t descKey(uint32_t u) {
  if (u == 0x80000000u) u = 0u;
  const uint32_t key = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return ~key;
}

// ---------------------------------------------------------------- CPU reference
// std::stable_sort of the column indices on the integer key. Not Accelerate's vDSP_vsorti,
// which Apple documents as not stable (ch14-facts C2).
static void refOrder(const float* row, int cols, std::vector<int>& order) {
  std::vector<uint32_t> key((size_t)cols);
  for (int c = 0; c < cols; ++c) key[c] = descKey(bitsOf(row[c]));
  order.resize((size_t)cols);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return key[a] < key[b]; });
}

// The cross-check on rows without NaN: std::stable_sort with > on the floats must give the
// same order (ch14-facts A11). With NaN, > is not a strict weak ordering, so no check.
static bool sameAsFloatSort(const float* row, int cols, const std::vector<int>& order) {
  for (int c = 0; c < cols; ++c)
    if (isNaNBits(bitsOf(row[c]))) return true;
  std::vector<int> o((size_t)cols);
  std::iota(o.begin(), o.end(), 0);
  std::stable_sort(o.begin(), o.end(), [&](int a, int b) { return row[a] > row[b]; });
  return o == order;
}

// ---------------------------------------------------------------- test data
enum Shape { Softmax, Ties, AllEqual, OneHot, Logits, NaNs };
struct Recipe { Shape shape; const char* what; };
static const Recipe kRecipes[] = {
    {Softmax, "softmax of uniform logits"},
    {Ties, "softmax with planted exact ties"},
    {AllEqual, "every probability equal"},
    {OneHot, "one-hot"},
    {Logits, "logits with -0.0, +0.0, infinities, FLT_MAX and subnormals"},
    {NaNs, "logits with +NaN and -NaN"},
};
static const int kRecipeCount = (int)(sizeof kRecipes / sizeof kRecipes[0]);

// softmax(12 * u) for u uniform in [-1, 1], in double, rounded to float.
static void softmaxRow(float* row, int cols, uint32_t seed) {
  std::vector<float> u = pmpp::random_floats((size_t)cols, seed);
  double m = -1e300, s = 0;
  for (int c = 0; c < cols; ++c) m = std::max(m, 12.0 * u[c]);
  std::vector<double> e((size_t)cols);
  for (int c = 0; c < cols; ++c) s += (e[c] = std::exp(12.0 * u[c] - m));
  for (int c = 0; c < cols; ++c) row[c] = (float)(e[c] / s);
}

// Special values for the logit rows, each planted twice (once from the front, once from the back).
static const uint32_t kSpecials[] = {
    0xFF800000u,   // -inf
    0x7F800000u,   // +inf
    0xFF7FFFFFu,   // -FLT_MAX
    0x7F7FFFFFu,   // FLT_MAX
    0x00000001u,   // smallest positive subnormal
    0x80000001u,   // smallest negative subnormal
    0x00400000u,   // a positive subnormal
    0x807FFFFFu,   // the negative subnormal of largest magnitude
    0x00800000u,   // FLT_MIN
    0x80800000u,   // -FLT_MIN
};
static const uint32_t kNaNs[] = {0x7FC00000u, 0xFFC00000u, 0x7FC00001u, 0xFFC12345u};

static void fillRow(Shape shape, float* row, int cols, uint32_t seed) {
  switch (shape) {
    case Softmax: softmaxRow(row, cols, seed); break;
    case Ties: {
      softmaxRow(row, cols, seed);
      float top = row[0], low = row[0];
      for (int c = 1; c < cols; ++c) { top = std::max(top, row[c]); low = std::min(low, row[c]); }
      // Pairs: at both ends, either side of 256, and in one thread's stride 256 apart. Each pair
      // also ties with the column that already holds that value.
      if (cols > 1) row[0] = row[cols - 1] = top;
      if (cols > 256) row[255] = row[256] = row[100];
      if (cols > 261) row[5] = row[261] = low;
      // Runs of 40 equal values across every multiple of 256 from 512 to 8192. A tile is
      // 256c keys (BLOCK_SIZE is 256), so for every whole c from 1 to 32 a run crosses a tile
      // boundary once the row is longer than 256c + 20. All runs share column 3's value.
      for (int boundary = 512; boundary <= 8192; boundary += 256)
        if (boundary + 20 <= cols)
          for (int c = boundary - 20; c < boundary + 20; ++c) row[c] = row[3];
      break;
    }
    case AllEqual:
      for (int c = 0; c < cols; ++c) row[c] = 1.0f / (float)cols;
      break;
    case OneHot:
      for (int c = 0; c < cols; ++c) row[c] = 0.0f;
      row[(2 * cols) / 3] = 1.0f;   // the zeros must come after it in column order
      break;
    case Logits:
    case NaNs: {
      std::vector<float> u = pmpp::random_floats((size_t)cols, seed);
      for (int c = 0; c < cols; ++c) row[c] = 12.0f * u[c];
      // -0.0 and +0.0 interleaved every 11 columns: equal keys that must stay in column order.
      for (int c = 0; c < cols; c += 11) row[c] = fromBits((c / 11) % 2 ? 0x80000000u : 0x00000000u);
      std::vector<bool> planted((size_t)cols, false);
      int front = 1, back = cols - 1;
      for (uint32_t s : kSpecials) {
        while (front < cols && (front % 11 == 0 || planted[front])) ++front;
        if (front < cols) { row[front] = fromBits(s); planted[front] = true; }
        while (back > 0 && (back % 11 == 0 || planted[back])) --back;
        if (back > 0) { row[back] = fromBits(s); planted[back] = true; }
      }
      if (shape == NaNs) {
        const int at[] = {2, cols / 3, cols / 2, cols - 2};
        for (int k = 0; k < 4; ++k)
          if (at[k] >= 0 && at[k] < cols) row[at[k]] = fromBits(kNaNs[k]);
      }
      break;
    }
  }
}

// A batch of `rows` rows of `cols` values: row r follows recipe (start + r) % count, so a kernel
// that reads or writes the wrong row meets a row of another kind.
struct Batch {
  int rows, cols;
  std::vector<float> x;
  std::vector<std::string> what;
};
static Batch makeBatch(int start, int rows, int cols) {
  Batch b{rows, cols, std::vector<float>((size_t)rows * cols), {}};
  for (int r = 0; r < rows; ++r) {
    const Recipe& one = kRecipes[(start + r) % kRecipeCount];
    fillRow(one.shape, &b.x[(size_t)r * cols], cols, 7919u * (uint32_t)cols + 104729u * (uint32_t)(start + r) + 17u);
    b.what.push_back(one.what);
  }
  return b;
}

// ---------------------------------------------------------------- one sort, checked
static const int kGuard = 64;                       // elements past the end of every buffer
static const uint32_t kGuardBits = 0xDEADBEEFu;     // what they hold before the launch
static const int kGuardIdx = -77;
static const uint32_t kUnwrittenBits = 0xFFFFFFFFu; // sortedProbs before the launch (a NaN no test row holds)
static const uint32_t kScratchFill = 0xCDCDCDCDu;

static std::string where(const Batch& b, int r) {
  return std::to_string(b.rows) + " rows of " + std::to_string(b.cols) + ", row " + std::to_string(r) +
         " (" + b.what[r] + ")";
}
static std::string hex(uint32_t u) { char s[16]; std::snprintf(s, sizeof s, "0x%08X", u); return s; }
static std::string val(uint32_t u) {
  char s[48];
  std::snprintf(s, sizeof s, "%.9g [%s]", (double)fromBits(u), hex(u).c_str());
  return s;
}

// What is wrong with one output row of indices, or "".
static std::string checkIdxRow(const Batch& b, int r, const int* got, const std::vector<std::vector<int>>& refs) {
  const int cols = b.cols;
  const float* row = &b.x[(size_t)r * cols];
  const std::vector<int>& want = refs[r];
  // A permutation of 0 .. cols-1?
  std::vector<int> seenAt((size_t)cols, -1);
  for (int p = 0; p < cols; ++p) {
    const int c = got[p];
    if (c == -1) return where(b, r) + ": sortedIdx[" + std::to_string(p) + "] was never written";
    if (c < 0 || c >= cols)
      return where(b, r) + ": sortedIdx[" + std::to_string(p) + "] = " + std::to_string(c) + ", not a column of this row";
    if (seenAt[c] >= 0)
      return where(b, r) + ": column " + std::to_string(c) + " appears at positions " + std::to_string(seenAt[c]) +
             " and " + std::to_string(p) + ", so the row is not a permutation (two keys written to one slot?)";
    seenAt[c] = p;
  }
  if (std::equal(want.begin(), want.end(), got)) return "";
  for (int o = 0; o < b.rows; ++o)
    if (o != r && b.cols == (int)refs[o].size() && std::equal(refs[o].begin(), refs[o].end(), got))
      return where(b, r) + ": this is the order of row " + std::to_string(o) + " (is every kernel offset by its row?)";
  int p = 0;
  while (got[p] == want[p]) ++p;
  const uint32_t gb = bitsOf(row[got[p]]), wb = bitsOf(row[want[p]]);
  std::string err = where(b, r) + ": position " + std::to_string(p) + " holds column " + std::to_string(got[p]) + ", " +
                    val(gb) + "; expected column " + std::to_string(want[p]) + ", " + val(wb);
  if (descKey(gb) == descKey(wb)) {
    err += " (equal keys: ties go by increasing column, which a stable pass gives";
    if (gb != wb && isZeroBits(gb) && isZeroBits(wb)) err += "; -0.0 and +0.0 are equal: map -0.0 to +0.0 first";
    err += ")";
  } else if (isNaNBits(gb) || isNaNBits(wb)) {
    err += " (NaN goes by its bits after the transform: +NaN first, -NaN last)";
  } else {
    // Every position before p agrees with the reference, so the expected column holds the
    // smallest key left: the column found here belongs further back.
    err += " (the expected value comes first in the contract's order";
    auto special = [](uint32_t u) {
      return (u & 0x80000000u) != 0u || (u & 0x7F800000u) == 0x7F800000u || (u & 0x7F800000u) == 0u;
    };
    if (special(gb) || special(wb)) err += ": check the key for negatives, zeros, infinities and subnormals";
    else err += ": is every pass stable, and do the passes go least significant digit first?";
    err += ")";
  }
  return err;
}

static std::string checkSort(const Batch& b) {
  const int rows = b.rows, cols = b.cols;
  const size_t n = (size_t)rows * cols;
  std::vector<uint32_t> in(n);
  for (size_t i = 0; i < n; ++i) in[i] = bitsOf(b.x[i]);

  // The reference, and its cross-check against std::stable_sort with > on the floats.
  std::vector<std::vector<int>> refs((size_t)rows);
  for (int r = 0; r < rows; ++r) {
    refOrder(&b.x[(size_t)r * cols], cols, refs[r]);
    if (!sameAsFloatSort(&b.x[(size_t)r * cols], cols, refs[r]))
      return where(b, r) + ": the driver's reference disagrees with std::stable_sort(>) (a bug in main.cpp, not in your kernel)";
  }

  Buf* probs = pmpp::alloc(n * 4);
  Buf* sp = pmpp::alloc((n + kGuard) * 4);
  Buf* si = pmpp::alloc((n + kGuard) * 4);
  Buf* scratch = pmpp::alloc((4 * n + kGuard) * 4);
  pmpp::upload(probs, in.data(), n * 4);
  std::vector<uint32_t> spGot(n + kGuard, kGuardBits);
  std::vector<int> siGot(n + kGuard, kGuardIdx);
  std::vector<uint32_t> scr(4 * n + kGuard, kGuardBits);
  std::fill(spGot.begin(), spGot.begin() + n, kUnwrittenBits);
  std::fill(siGot.begin(), siGot.begin() + n, -1);
  std::fill(scr.begin(), scr.begin() + 4 * n, kScratchFill);
  pmpp::upload(sp, spGot.data(), spGot.size() * 4);
  pmpp::upload(si, siGot.data(), siGot.size() * 4);
  pmpp::upload(scratch, scr.data(), scr.size() * 4);

  std::string err = launch(probs, sp, si, scratch, rows, cols);
  if (err.empty()) err = pmpp::take_error();
  if (err.empty()) {
    std::vector<uint32_t> after(n);
    pmpp::download(sp, spGot.data(), spGot.size() * 4);
    pmpp::download(si, siGot.data(), siGot.size() * 4);
    pmpp::download(scratch, scr.data(), scr.size() * 4);
    pmpp::download(probs, after.data(), n * 4);
    for (int r = 0; r < rows && err.empty(); ++r)
      err = checkIdxRow(b, r, &siGot[(size_t)r * cols], refs);
    for (int r = 0; r < rows && err.empty(); ++r)
      for (int p = 0; p < cols; ++p) {
        const uint32_t g = spGot[(size_t)r * cols + p], w = in[(size_t)r * cols + refs[r][p]];
        if (g == w) continue;
        err = where(b, r) + ": sortedProbs[" + std::to_string(p) + "] has the bits " + hex(g) + ", expected " + hex(w) +
              " (the value of column " + std::to_string(refs[r][p]) + ")";
        int fromRow = -1;
        for (int o = 0; o < rows && fromRow < 0; ++o)
          if (o != r && in[(size_t)o * cols + refs[r][p]] == g) fromRow = o;
        if (g == kUnwrittenBits) err += ": never written";
        else if (fromRow >= 0) err += ": that is the value in row " + std::to_string(fromRow) + " (is the gather offset by its row?)";
        else err += ": copy the 32 bits of probs at the sorted index, do not convert them";
        break;
      }
    for (int g = 0; g < kGuard && err.empty(); ++g) {
      if (spGot[n + g] != kGuardBits) err = "wrote sortedProbs[" + std::to_string(n + g) + "], past the end of the output";
      else if (siGot[n + g] != kGuardIdx) err = "wrote sortedIdx[" + std::to_string(n + g) + "], past the end of the output";
      else if (scr[4 * n + g] != kGuardBits)
        err = "wrote scratch[" + std::to_string(4 * n + g) + "], past the 4 x rows x cols entries it holds";
    }
    for (size_t i = 0; i < n && err.empty(); ++i)
      if (after[i] != in[i]) err = "probs[" + std::to_string(i) + "] changed: the input is read-only";
  }
  pmpp::release(probs); pmpp::release(sp); pmpp::release(si); pmpp::release(scratch);
  return err;
}

// ---------------------------------------------------------------- tests
static std::vector<int> pick(const std::vector<int>& race, const std::vector<int>& emu, const std::vector<int>& dev) {
  const std::string b = pmpp::backend();
  return b == "race" ? race : b == "emu" ? emu : dev;
}

// Runs every recipe in every row count. With one row per batch each recipe is row 0 once;
// with more rows the recipes are mixed. On `emu`, where one sort of one row is tens of
// thousands of thread switches and every case runs three times, each row count runs one
// batch per column count, starting at a recipe that moves on with the column count
// (`turn`), so that every recipe still meets several sizes, the largest included.
static std::string overBatches(int cols, int turn, const std::vector<int>& rowCounts) {
  const bool fewer = std::string(pmpp::backend()) == "emu";
  for (int rows : rowCounts) {
    const int first = fewer ? (turn * rows) % kRecipeCount : 0;
    const int last = fewer ? first + 1 : kRecipeCount;
    for (int start = first; start < last; start += rows) {
      std::string err = checkSort(makeBatch(start, rows, cols));
      if (!err.empty()) return err;
    }
  }
  return "";
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  // race: 2100 columns make two tiles even at 8 keys per thread, so the race detector sees
  // one tile's last reads against the next tile's first writes.
  const std::vector<int> rowCounts = pick({2}, {1, 3}, {1, 3, 4, 7});
  const std::vector<int> colCounts = pick({1, 31, 257, 1000, 2100}, {1, 2, 31, 255, 256, 257, 1000, 4099},
                                          {1, 2, 31, 255, 256, 257, 1000, 4096, 32000, 50257, 151936});
  const std::string why = pipelineProblem();
  if (!why.empty()) {   // reported once, not once per shape
    t.check("sort: kernels usable", [&] { return why; });
    return t.finish();
  }
  for (size_t i = 0; i < colCounts.size(); ++i) {
    const int cols = colCounts[i];
    t.check("sort rows descending, cols " + std::to_string(cols), [&] { return overBatches(cols, (int)i, rowCounts); });
  }
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
// Time per call: best of `reps` timed runs; each run repeats the call `calls` times back to
// back, with `calls` doubled (untimed) until one run lasts at least 20 ms. While this lab was
// written, runs of a single short call varied by up to a factor of two from one run to the
// next, and runs of many calls back to back did not.
static double perCall(int reps, const std::function<std::string(int, pmpp::Timing*)>& run, int* calls, std::string* err) {
  *calls = 1;
  pmpp::Timing tm{0, 0};
  for (;;) {
    *err = run(*calls, &tm);
    if (err->empty()) *err = pmpp::take_error();
    if (!err->empty()) return -1;
    if (tm.best() >= 0.02 || *calls >= 4096) break;
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

// The baseline: each row's column indices put in order by std::stable_sort with > on the
// floats, then the values gathered, one row after another on this CPU thread.
static void cpuStableSortRows(const std::vector<float>& x, int rows, int cols, std::vector<int>& idx,
                              std::vector<float>& vals) {
  for (int r = 0; r < rows; ++r) {
    const float* row = &x[(size_t)r * cols];
    int* o = &idx[(size_t)r * cols];
    std::iota(o, o + cols, 0);
    std::stable_sort(o, o + cols, [row](int a, int b) { return row[a] > row[b]; });
    for (int p = 0; p < cols; ++p) vals[(size_t)r * cols + p] = row[o[p]];
  }
}

static int bench(const pmpp::Args& a) {
  const std::string why = pipelineProblem();
  if (!why.empty()) { std::printf("sort: %s\n", why.c_str()); return 1; }
  const double mem = deviceMemory();
  const double budget = mem > 0 ? mem / 4 : 1e9;   // bytes one shape may take on the device
  const std::vector<int> rowsList = {1, 8, 64};
  const std::vector<int> colsList = {32000, 151936};
  int rc = 0;

  std::printf("\nrow-wise descending stable sort of softmax probabilities, best of %d timed runs\n", a.reps);
  std::printf("  %4s %7s | %6s %12s %10s %9s | %10s %10s | %8s\n", "rows", "cols", "calls", "us per call",
              "rows/s", "Mkeys/s", "CPU ms", "CPU rows/s", "CPU/GPU");
  for (int cols : colsList)
    for (int rows : rowsList) {
      const size_t n = (size_t)rows * cols;
      if ((double)n * 28 > budget) {   // probs, sortedProbs, sortedIdx, 4 x scratch: 28 bytes per key
        std::printf("  %4d %7d | skipped: larger than a quarter of the device memory reported\n", rows, cols);
        continue;
      }
      std::vector<float> x(n);
      for (int r = 0; r < rows; ++r) softmaxRow(&x[(size_t)r * cols], cols, 31u + (uint32_t)(r * 7 + cols));

      // CPU baseline, on this thread (a host measurement), checked against the reference.
      std::vector<int> cpuIdx(n);
      std::vector<float> cpuVals(n);
      cpuStableSortRows(x, rows, cols, cpuIdx, cpuVals);   // untimed run
      double cpuS = 1e30;
      for (int r = 0; r < a.reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        cpuStableSortRows(x, rows, cols, cpuIdx, cpuVals);
        const auto t1 = std::chrono::steady_clock::now();
        cpuS = std::min(cpuS, std::chrono::duration<double>(t1 - t0).count());
      }
      std::string bad;
      std::vector<int> ref;
      for (int r = 0; r < rows && bad.empty(); ++r) {
        refOrder(&x[(size_t)r * cols], cols, ref);
        if (!std::equal(ref.begin(), ref.end(), &cpuIdx[(size_t)r * cols])) bad = "CPU baseline row " + std::to_string(r) + " wrong";
      }
      if (!bad.empty()) { std::printf("  %4d %7d | %s; not reported\n", rows, cols, bad.c_str()); rc = 1; continue; }

      // Your sort.
      Buf* probs = pmpp::alloc(n * 4);
      Buf* sp = pmpp::alloc(n * 4);
      Buf* si = pmpp::alloc(n * 4);
      Buf* scratch = pmpp::alloc(4 * n * 4);
      pmpp::upload(probs, x.data(), n * 4);
      int calls = 0;
      std::string err;
      const double s = perCall(a.reps, [&](int c, pmpp::Timing* tm) { return launch(probs, sp, si, scratch, rows, cols, c, tm); },
                               &calls, &err);
      if (s <= 0) bad = "failed: " + err;
      if (bad.empty()) {   // check what the timed calls produced before reporting their time
        std::vector<int> gotIdx(n);
        std::vector<float> gotVals(n);
        pmpp::download(si, gotIdx.data(), n * 4);
        pmpp::download(sp, gotVals.data(), n * 4);
        if (gotIdx != cpuIdx) bad = "sortedIdx wrong";
        else if (std::memcmp(gotVals.data(), cpuVals.data(), n * 4) != 0) bad = "sortedProbs wrong";
      }
      pmpp::release(probs); pmpp::release(sp); pmpp::release(si); pmpp::release(scratch);
      if (!bad.empty()) { std::printf("  %4d %7d | %s; not reported\n", rows, cols, bad.c_str()); rc = 1; continue; }

      const double rowsPerS = rows / s, mkeys = n / s / 1e6, cpuRowsPerS = rows / cpuS;
      std::printf("  %4d %7d | %6d %12.1f %10.0f %9.1f | %10.2f %10.1f | %7.2fx\n", rows, cols, calls, s * 1e6,
                  rowsPerS, mkeys, cpuS * 1e3, cpuRowsPerS, cpuS / s);
      pmpp::bench_row(a, "sort_rows").n("rows", rows).n("cols", cols).n("seconds", s)
          .n("rows_per_s", rowsPerS).n("mkeys_per_s", mkeys).emit();
      pmpp::bench_row(a, "cpu_stable_sort_rows").n("rows", rows).n("cols", cols).n("seconds", cpuS)
          .n("rows_per_s", cpuRowsPerS).emit();
    }
  std::printf("\nus per call: your whole sort (keys, every pass, gather), timed on the device.\n"
              "CPU: std::stable_sort with > on each row in turn, plus the gather, on one CPU thread,\n"
              "timed on the host with a steady clock. CPU/GPU above 1 means your sort took less time.\n"
              "These are measurements of this machine, at these sizes.\n");
  return rc;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
