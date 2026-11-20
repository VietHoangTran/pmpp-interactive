// Lab 12.4 driver (milestone M5 piece): tests your candidate compaction against a CPU
// reference (llama.cpp's host min-p loop) and benchmarks it against a provided
// one-thread-per-row baseline (provided/). You do not need to edit this file; your work
// goes in work/.
#include <cstring>
#include <limits>

#include "pmpp/harness.hpp"
using pmpp::Buf;

// The kernels this driver runs. Serial is the provided baseline.
enum Kernel { Block, Serial };
static const int kBlock = 256;   // threads per block (BLOCK_SIZE in kernel.cu, kBlock in launch.mm)

// The device buffers of one call.
struct Bufs {
  Buf *logits, *thr, *idx, *val, *counts;
};

// ---- entry points implemented in <variant>/ and provided/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
bool encodeCompactCandidates(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                             id<MTLBuffer> logits, id<MTLBuffer> threshold, id<MTLBuffer> outIdx,
                             id<MTLBuffer> outVal, id<MTLBuffer> counts, uint32_t rows, uint32_t cols);
void encodeCompactSerial(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                         id<MTLBuffer> logits, id<MTLBuffer> threshold, id<MTLBuffer> outIdx,
                         id<MTLBuffer> outVal, id<MTLBuffer> counts, uint32_t rows, uint32_t cols);
using pmpp::metal::buffer;

static std::string dirOf(const std::string& p) { return p.substr(0, p.find_last_of('/')); }
static const std::string kProvided = dirOf(dirOf(PMPP_METAL_SOURCE)) + "/provided/rowwise_serial.metal";

static id<MTLComputePipelineState> pipelineFor(Kernel k) {
  return k == Block ? pmpp::metal::pipeline(PMPP_METAL_SOURCE, "compact_candidates")
                    : pmpp::metal::pipeline(kProvided, "compact_serial");
}

// Why this pipeline cannot run the kernel as the lab specifies it, or "". The driver's own
// copy of the rule a correct launch.mm applies: 256 threads per threadgroup, in whole
// SIMD-groups, and no more SIMD-groups than a SIMD-group has lanes.
static std::string pipelineProblem(Kernel k) {
  id<MTLComputePipelineState> p = pipelineFor(k);
  if (!p) return "the Metal kernel did not build: " + pmpp::take_error();
  if (k == Serial) return "";
  char b[200];
  if ((NSUInteger)kBlock > p.maxTotalThreadsPerThreadgroup) {
    std::snprintf(b, sizeof b, "this pipeline runs at most %lu threads per threadgroup; the lab needs %d",
                  (unsigned long)p.maxTotalThreadsPerThreadgroup, kBlock);
    return b;
  }
  const NSUInteger w = p.threadExecutionWidth;
  if (w == 0 || kBlock % w != 0 || (NSUInteger)kBlock / w > w) {
    std::snprintf(b, sizeof b, "SIMD-group width %lu: %d threads are not 1 to %lu whole SIMD-groups",
                  (unsigned long)w, kBlock, (unsigned long)w);
    return b;
  }
  return "";
}

// Enqueues `repeat` launches of kernel k in one timed run. Empty string when it ran.
static std::string launch(Kernel k, const Bufs& b, int rows, int cols, int repeat = 1,
                          pmpp::Timing* tm = nullptr) {
  std::string why = pipelineProblem(k);
  if (!why.empty()) return why;
  id<MTLComputePipelineState> p = pipelineFor(k);
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    const uint32_t r = (uint32_t)rows, c = (uint32_t)cols;
    for (int i = 0; i < repeat && ok; ++i) {
      if (k == Block)
        ok = encodeCompactCandidates(enc, p, buffer(b.logits), buffer(b.thr), buffer(b.idx), buffer(b.val),
                                     buffer(b.counts), r, c);
      else
        encodeCompactSerial(enc, p, buffer(b.logits), buffer(b.thr), buffer(b.idx), buffer(b.val),
                            buffer(b.counts), r, c);
    }
  });
  if (tm) *tm = t;
  if (!ok) return "your launch.mm refused to dispatch, although this pipeline can run 256 threads per threadgroup";
  return "";
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
void launchCompactCandidates(const float* logits, const float* threshold, int* outIdx, float* outVal,
                             unsigned int* counts, int rows, int cols);
void launchCompactSerial(const float* logits, const float* threshold, int* outIdx, float* outVal,
                         unsigned int* counts, int rows, int cols);
static float* f32(Buf* b) { return (float*)pmpp::raw(b); }

static std::string pipelineProblem(Kernel k) {
  if (k == Serial) return "";
  const int most = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  return kBlock <= most ? "" : "this device runs at most " + std::to_string(most) + " threads per block; the lab needs 256";
}
static std::string launch(Kernel k, const Bufs& b, int rows, int cols, int repeat = 1,
                          pmpp::Timing* tm = nullptr) {
  std::string why = pipelineProblem(k);
  if (!why.empty()) return why;
  pmpp::Timing t = pmpp::run([&](void*) {
    for (int i = 0; i < repeat; ++i) {
      if (k == Block)
        launchCompactCandidates(f32(b.logits), f32(b.thr), (int*)pmpp::raw(b.idx), f32(b.val),
                                (unsigned int*)pmpp::raw(b.counts), rows, cols);
      else
        launchCompactSerial(f32(b.logits), f32(b.thr), (int*)pmpp::raw(b.idx), f32(b.val),
                            (unsigned int*)pmpp::raw(b.counts), rows, cols);
    }
  });
  if (tm) *tm = t;
  return "";
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

// ---------------------------------------------------------------- CPU reference
// llama.cpp's host min-p loop (notes/today/ch12-facts.md D3) without the copy back: every
// column whose logit is >= the threshold, in ascending order.
static void refCompact(const float* row, int cols, float thr, std::vector<int>& keep) {
  keep.clear();
  for (int c = 0; c < cols; ++c)
    if (row[c] >= thr) keep.push_back(c);
}
static int countAtLeast(const float* row, int cols, float thr) {
  int n = 0;
  for (int c = 0; c < cols; ++c) n += row[c] >= thr;
  return n;
}
// The threshold that keeps the `frac` share of the row with the largest logits: the k-th
// largest value, k = max(1, round(frac * cols)). Equal values may make the share larger.
static float quantile(const float* row, int cols, double frac) {
  const int k = std::max(1, std::min(cols, (int)std::llround(frac * cols)));
  std::vector<float> v(row, row + cols);
  std::nth_element(v.begin(), v.begin() + (k - 1), v.end(), std::greater<float>());
  return v[k - 1];
}
static float rowMax(const float* row, int cols) { return *std::max_element(row, row + cols); }
static float rowMin(const float* row, int cols) { return *std::min_element(row, row + cols); }

static uint32_t bitsOf(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  return u;
}

// ---------------------------------------------------------------- test data
enum Kind { AboveMax, AtMax, BelowMin, Quantile, MinP, Tie, LastRound };
struct Recipe {
  Kind kind;
  double p;   // the share for Quantile, p for MinP
  int a, b;   // the planted columns for Tie
  std::string what;
};

// The rows a test batch is built from. Each has its own threshold, and a batch mixes them,
// so a kernel that reads the wrong row's threshold keeps the wrong number of candidates.
static std::vector<Recipe> recipes(int cols) {
  std::vector<Recipe> r = {
      {AboveMax, 0, 0, 0, "threshold just above the row's max: nothing kept"},
      {AtMax, 0, 0, 0, "threshold equal to the row's max: only the maxima kept"},
      {BelowMin, 0, 0, 0, "threshold just below the row's min: everything kept"},
      {Quantile, 0.01, 0, 0, "threshold at the 1 percent quantile"},
      {Quantile, 0.10, 0, 0, "threshold at the 10 percent quantile"},
      {Quantile, 0.50, 0, 0, "threshold at the 50 percent quantile"},
      {MinP, 0.1, 0, 0, "min-p 0.1: threshold max + ln 0.1, as llama.cpp computes it"},
      {LastRound, 0, 0, 0, "survivors only in the last round of 256 columns"},
  };
  const struct { int a, b; const char* why; } pairs[] = {
      {1, 2, "neighbouring lanes"},
      {31, 32, "across a 32-lane warp boundary"},
      {63, 64, "across a 64-lane warp boundary"},
      {255, 256, "across a round boundary"},
      {0, cols - 1, "the first and the last column"},
  };
  for (auto& q : pairs) {
    if (q.a < 0 || q.b >= cols || q.a >= q.b) continue;
    r.push_back({Tie, 0.05, q.a, q.b, "values equal to the threshold at columns " + std::to_string(q.a) + " and " +
                                          std::to_string(q.b) + " (" + q.why + ")"});
  }
  return r;
}

// Fills one row and returns its threshold. Logits are uniform in [-12, 12].
static float fillRow(const Recipe& rc, float* row, int cols, uint32_t seed) {
  std::vector<float> u = pmpp::random_floats((size_t)cols, seed);   // uniform in [-1, 1]
  for (int c = 0; c < cols; ++c) row[c] = 12.0f * u[c];
  const float inf = std::numeric_limits<float>::infinity();
  switch (rc.kind) {
    case AboveMax: return std::nextafter(rowMax(row, cols), inf);
    case AtMax: return rowMax(row, cols);
    case BelowMin: return std::nextafter(rowMin(row, cols), -inf);
    case Quantile: return quantile(row, cols, rc.p);
    case MinP: return rowMax(row, cols) + std::log((float)rc.p);   // llama.cpp: max_logit + logf(p)
    case Tie: {
      const float thr = quantile(row, cols, rc.p);
      row[rc.a] = row[rc.b] = thr;   // exactly equal: kept by >=, dropped by >
      return thr;
    }
    default: {   // LastRound: everything before the last round's first column is below 0
      const int last = ((cols - 1) / kBlock) * kBlock;
      for (int c = 0; c < last; ++c) row[c] = -std::fabs(row[c]) - 0.25f;
      row[cols - 1] = 6.0f;   // at least one survivor, in the last column
      return 0.0f;
    }
  }
}

// A batch of `rows` rows of `cols` logits: row r follows recipe (start + r) % count.
struct Batch {
  int rows, cols;
  std::vector<float> x, thr;
  std::vector<std::string> what;
};
static Batch makeBatch(const std::vector<Recipe>& rc, int start, int rows, int cols) {
  Batch b{rows, cols, std::vector<float>((size_t)rows * cols), std::vector<float>(rows), {}};
  for (int r = 0; r < rows; ++r) {
    const Recipe& one = rc[(size_t)(start + r) % rc.size()];
    b.thr[r] = fillRow(one, &b.x[(size_t)r * cols], cols, 7919u * (uint32_t)cols + 104729u * (uint32_t)(start + r) + 29u);
    b.what.push_back(one.what);
  }
  return b;
}

// ---------------------------------------------------------------- one launch, checked
static const int kGuard = 64;                          // elements past the end of every output
static const int kIdxUnset = -1;                       // outIdx before the launch
static const float kValUnset = -12345.5f;              // outVal before the launch
static const unsigned int kCountUnset = 0xDEADBEEFu;   // counts before the launch

static std::string where(const Batch& b, int r) {
  return std::to_string(b.rows) + " rows of " + std::to_string(b.cols) + ", row " + std::to_string(r) +
         " (" + b.what[r] + ")";
}
static std::string fmt(const char* f, double a, double c = 0, double d = 0) {
  char buf[200];
  std::snprintf(buf, sizeof buf, f, a, c, d);
  return buf;
}

// The mistakes that would give exactly this wrong count. Several can fit one count (with one
// column, "nothing kept" fits most of them), so every one that fits is listed.
static std::string countHint(const Batch& b, int r, unsigned int got, size_t want) {
  const int cols = b.cols;
  const float* row = &b.x[(size_t)r * cols];
  if (got == kCountUnset) return " (counts[row] was never written)";
  if (got > (unsigned int)cols) return " (more than the row has columns)";
  std::vector<std::string> fits;
  auto consider = [&](int n, const std::string& why) {
    if ((size_t)n != want && (unsigned int)n == got) fits.push_back(why);
  };
  int greater = 0;
  for (int c = 0; c < cols; ++c) greater += row[c] > b.thr[r];
  consider(greater, "values equal to the threshold dropped: keep logit >= threshold");
  for (int r2 = 0; r2 < b.rows; ++r2)
    if (r2 != r && b.thr[r2] != b.thr[r] && (unsigned int)countAtLeast(row, cols, b.thr[r2]) == got && got != want) {
      fits.push_back("row " + std::to_string(r2) + "'s threshold used: read threshold[row]");
      break;
    }
  const int last = ((cols - 1) / kBlock) * kBlock;   // first column of the last round
  const int inLast = countAtLeast(row + last, cols - last, b.thr[r]);
  consider((int)want - inLast, "the last round, or its partial tail, skipped");
  consider(inLast, "only the last round counted: carry the row's count from round to round");
  if (fits.empty()) return "";
  std::string s = fits.size() == 1 ? " (" : " (possible causes: ";
  for (size_t i = 0; i < fits.size(); ++i) s += (i ? "; or " : "") + fits[i];
  return s + ")";
}

// Your kernel or the provided one, on one batch.
static std::string check(Kernel k, const Batch& b) {
  const int rows = b.rows, cols = b.cols;
  const size_t total = (size_t)rows * cols;
  Bufs d{pmpp::alloc(total * 4), pmpp::alloc((size_t)rows * 4), pmpp::alloc((total + kGuard) * 4),
         pmpp::alloc((total + kGuard) * 4), pmpp::alloc((size_t)(rows + kGuard) * 4)};
  pmpp::upload(d.logits, b.x.data(), total * 4);
  pmpp::upload(d.thr, b.thr.data(), (size_t)rows * 4);
  std::vector<int> gotIdx(total + kGuard, kIdxUnset);
  std::vector<float> gotVal(total + kGuard, kValUnset);
  std::vector<unsigned int> gotCount(rows + kGuard, kCountUnset);
  pmpp::upload(d.idx, gotIdx.data(), gotIdx.size() * 4);
  pmpp::upload(d.val, gotVal.data(), gotVal.size() * 4);
  pmpp::upload(d.counts, gotCount.data(), gotCount.size() * 4);

  std::string err = launch(k, d, rows, cols);
  if (err.empty()) {
    pmpp::download(d.idx, gotIdx.data(), gotIdx.size() * 4);
    pmpp::download(d.val, gotVal.data(), gotVal.size() * 4);
    pmpp::download(d.counts, gotCount.data(), gotCount.size() * 4);
    const uint32_t unsetBits = bitsOf(kValUnset);
    std::vector<int> want;
    for (int r = 0; r < rows && err.empty(); ++r) {
      const float* row = &b.x[(size_t)r * cols];
      const int* gi = &gotIdx[(size_t)r * cols];
      const float* gv = &gotVal[(size_t)r * cols];
      refCompact(row, cols, b.thr[r], want);

      // 1. The count.
      if (gotCount[r] != want.size()) {
        err = where(b, r) + ": counts[row] = " + std::to_string(gotCount[r]) + ", expected " +
              std::to_string(want.size()) + countHint(b, r, gotCount[r], want.size());
        break;
      }
      // 2. The pairs, in ascending column order, values bit for bit.
      for (size_t i = 0; i < want.size() && err.empty(); ++i) {
        const int c = want[i];
        if (gi[i] != c) {
          err = where(b, r) + ": outIdx[" + std::to_string(i) + "] = " + std::to_string(gi[i]) + ", expected column " +
                std::to_string(c);
          if (gi[i] == kIdxUnset) err += " (never written)";
          else if (gi[i] < 0 || gi[i] >= cols) err += " (not a column of this row)";
          else if (std::find(want.begin(), want.end(), gi[i]) != want.end())
            err += " (a kept column, in the wrong place: the order must be ascending by column)";
          else err += fmt(" (logit %.9g there, below the threshold %.9g)", row[gi[i]], b.thr[r]);
        } else if (bitsOf(gv[i]) != bitsOf(row[c])) {
          err = where(b, r) + ": outVal[" + std::to_string(i) + "] = " + fmt("%.9g", gv[i]) + ", expected logit " +
                fmt("%.9g", row[c]) + " of column " + std::to_string(c) + " bit for bit";
          if (bitsOf(gv[i]) == unsetBits) err += " (never written)";
        }
      }
      // 3. Nothing past the row's count.
      for (int i = (int)want.size(); i < cols && err.empty(); ++i)
        if (gi[i] != kIdxUnset || bitsOf(gv[i]) != unsetBits)
          err = where(b, r) + ": wrote position " + std::to_string(i) + " of the row's output, past its count " +
                std::to_string(want.size());
    }
    for (size_t g = total; g < total + kGuard && err.empty(); ++g)
      if (gotIdx[g] != kIdxUnset || bitsOf(gotVal[g]) != unsetBits)
        err = "wrote element " + std::to_string(g) + " of outIdx or outVal, past the end of the output";
    for (int g = rows; g < rows + kGuard && err.empty(); ++g)
      if (gotCount[g] != kCountUnset) err = "wrote counts[" + std::to_string(g) + "], past the last row";
  }
  pmpp::release(d.logits); pmpp::release(d.thr); pmpp::release(d.idx); pmpp::release(d.val); pmpp::release(d.counts);
  return err;
}

// ---------------------------------------------------------------- tests
static std::vector<int> pick(const std::vector<int>& race, const std::vector<int>& emu, const std::vector<int>& dev) {
  const std::string b = pmpp::backend();
  return b == "race" ? race : b == "emu" ? emu : dev;
}

// Runs kernel k on batches that between them put every recipe in every row count. With one
// row per batch every recipe is row 0 once; with more rows the recipes are mixed.
static std::string overBatches(Kernel k, int cols, const std::vector<int>& rowCounts) {
  const std::vector<Recipe> rc = recipes(cols);
  for (int rows : rowCounts)
    for (int start = 0; start < (int)rc.size(); start += rows) {
      std::string err = check(k, makeBatch(rc, start, rows, cols));
      if (!err.empty()) return err;
    }
  return "";
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::vector<int> rowCounts = pick({2}, {1, 3}, {1, 4, 7});
  const std::vector<int> colCounts = pick({1, 33, 257, 1000}, {1, 2, 31, 32, 33, 255, 256, 257, 1000, 4099},
                                          {1, 2, 31, 32, 33, 255, 256, 257, 1000, 4096, 32000, 50257, 128256});

  const std::string why = pipelineProblem(Block);
  if (!why.empty()) {
    t.check("compaction: kernel usable", [&] { return why; });   // reported once, not once per shape
  } else {
    for (int cols : colCounts)
      t.check("compaction, cols " + std::to_string(cols), [&] { return overBatches(Block, cols, rowCounts); });
  }

  // The provided baseline, so that a broken provided file shows here and not in the benchmark.
  for (int cols : pick({33}, {1, 257, 1000}, {1, 257, 4096, 32000}))
    t.check("provided one-thread-per-row compaction, cols " + std::to_string(cols),
            [&] { return overBatches(Serial, cols, rowCounts); });
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
// Time per call: best of `reps` timed runs after one untimed run; each run repeats the call
// `calls` times back to back, with `calls` doubled until one run lasts at least 2 ms, so
// that short calls are not lost in the cost of submitting the work.
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

// Checks the output of the last timed call against the CPU lists. Empty string when right.
static std::string checkTimed(const Bufs& d, const std::vector<float>& x,
                              const std::vector<std::vector<int>>& want, int rows, int cols) {
  std::vector<unsigned int> cnt(rows);
  pmpp::download(d.counts, cnt.data(), cnt.size() * 4);
  std::vector<int> idx((size_t)rows * cols);
  std::vector<float> val((size_t)rows * cols);
  pmpp::download(d.idx, idx.data(), idx.size() * 4);
  pmpp::download(d.val, val.data(), val.size() * 4);
  for (int r = 0; r < rows; ++r) {
    if (cnt[r] != want[r].size()) return "row " + std::to_string(r) + ": wrong count";
    for (size_t i = 0; i < want[r].size(); ++i) {
      const size_t at = (size_t)r * cols + i;
      const int c = want[r][i];
      if (idx[at] != c || bitsOf(val[at]) != bitsOf(x[(size_t)r * cols + c]))
        return "row " + std::to_string(r) + ": wrong pair at position " + std::to_string(i);
    }
  }
  return "";
}

static int bench(const pmpp::Args& a) {
  const double mem = deviceMemory();
  const double budget = mem > 0 ? mem / 4 : 1e9;   // bytes the buffers of one shape may take
  const std::vector<int> rowsList = {1, 8, 64};
  const std::vector<int> colsList = {32000, 128256};
  const std::vector<double> fractions = {0.001, 0.01, 0.1, 0.5};
  int rc = 0;

  std::string why = pipelineProblem(Block);
  if (!why.empty()) { std::printf("compaction: %s\n", why.c_str()); return 1; }

  std::printf("\ncandidate compaction, best of %d timed runs\n", a.reps);
  std::printf("  %4s %7s %6s %9s  %-6s %6s %12s %9s %12s %10s\n", "rows", "cols", "kept", "kept/row", "kernel",
              "calls", "us per call", "GB/s", "rows/s", "vs serial");
  for (int cols : colsList)
    for (int rows : rowsList) {
      const size_t total = (size_t)rows * cols;
      if ((double)total * 12 + rows * 8.0 > budget) {
        std::printf("  %4d %7d  skipped: larger than a quarter of the device memory reported\n", rows, cols);
        continue;
      }
      std::vector<float> x = pmpp::random_floats(total, 31u + (uint32_t)(rows * 7 + cols));
      for (float& v : x) v *= 12.0f;
      Bufs d{pmpp::alloc(total * 4), pmpp::alloc((size_t)rows * 4), pmpp::alloc(total * 4), pmpp::alloc(total * 4),
             pmpp::alloc((size_t)rows * 4)};
      pmpp::upload(d.logits, x.data(), total * 4);

      for (double frac : fractions) {
        // Thresholds from CPU quantiles, and the lists every timed result is checked against.
        std::vector<float> thr(rows);
        std::vector<std::vector<int>> want(rows);
        size_t kept = 0;
        for (int r = 0; r < rows; ++r) {
          const float* row = &x[(size_t)r * cols];
          thr[r] = quantile(row, cols, frac);
          refCompact(row, cols, thr[r], want[r]);
          kept += want[r].size();
        }
        pmpp::upload(d.thr, thr.data(), (size_t)rows * 4);
        const double bytes = 4.0 * total + 8.0 * kept + 4.0 * rows;

        double serialS = 0;
        for (int which = 1; which >= 0; --which) {   // serial first, so "vs serial" can be printed
          const Kernel k = which ? Serial : Block;
          const char* key = which ? "serial" : "block";
          int calls = 0;
          std::string err;
          const double s = perCall(a.reps, [&](int c, pmpp::Timing* tm) { return launch(k, d, rows, cols, c, tm); },
                                   &calls, &err);
          if (s <= 0) {
            std::printf("  %4d %7d %5.1f%%  %-6s failed: %s\n", rows, cols, 100 * frac, key, err.c_str());
            rc = 1;
            continue;
          }
          // Check what the timed calls produced before reporting their time.
          const std::string bad = checkTimed(d, x, want, rows, cols);
          if (!bad.empty()) {
            std::printf("  %4d %7d %5.1f%%  %-6s %s; not reported\n", rows, cols, 100 * frac, key, bad.c_str());
            rc = 1;
            continue;
          }
          if (which) serialS = s;
          const double gbps = bytes / s / 1e9;
          std::printf("  %4d %7d %5.1f%% %9.0f  %-6s %6d %12.2f %9.2f %12.4g", rows, cols, 100 * frac,
                      (double)kept / rows, key, calls, s * 1e6, gbps, rows / s);
          if (!which && serialS > 0) std::printf(" %9.1fx", serialS / s);
          std::printf("\n");
          pmpp::Json j = pmpp::bench_row(a, "compaction");
          j.s("kernel", key).n("rows", rows).n("cols", cols).n("fraction", frac).n("kept", (double)kept)
              .n("seconds", s).n("gb_per_s", gbps).n("rows_per_s", rows / s);
          j.emit();
        }
      }
      pmpp::release(d.logits); pmpp::release(d.thr); pmpp::release(d.idx); pmpp::release(d.val);
      pmpp::release(d.counts);
    }
  std::printf("\n\"kept\" is the share asked for; \"kept/row\" what the thresholds kept on average (equal values\n"
              "can add a few). GB/s counts the minimum traffic: 4 bytes per logit read, 8 per kept pair\n"
              "written, 4 per row for the count (the thresholds are not counted).\n"
              "\"block\" is your kernel; \"serial\" is the provided one thread per row.\n"
              "These are measurements of this machine, at these sizes.\n");
  return rc;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
