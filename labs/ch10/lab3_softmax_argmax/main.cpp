// Lab 10.3 driver (milestone M4): tests your row-wise max, softmax and argmax against a CPU
// reference, runs a toy greedy decoder on your argmax, and benchmarks your softmax and argmax
// against a provided one-thread-per-row baseline (../shared/). You do not need to edit
// this file; your work goes in work/.
#include <cstring>
#include <limits>

#include "pmpp/harness.hpp"
using pmpp::Buf;

// The kernels this driver runs. The last two are the provided baseline.
enum Kernel { RowMax, Softmax, Argmax, SoftmaxNaive, ArgmaxNaive };
static const int kBlock = 256;   // threads per block (BLOCK_SIZE in kernel.cu, kBlock in launch.mm)

// ---- entry points implemented in <variant>/ and ../shared/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
bool encodeRowMax(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                  id<MTLBuffer> x, id<MTLBuffer> out, uint32_t rows, uint32_t cols);
bool encodeSoftmaxRows(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                       id<MTLBuffer> x, id<MTLBuffer> y, uint32_t rows, uint32_t cols, float scale);
bool encodeArgmaxRows(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                      id<MTLBuffer> x, id<MTLBuffer> idx, uint32_t rows, uint32_t cols);
void encodeSoftmaxRowsNaive(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                            id<MTLBuffer> x, id<MTLBuffer> y, uint32_t rows, uint32_t cols, float scale);
void encodeArgmaxRowsNaive(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                           id<MTLBuffer> x, id<MTLBuffer> idx, uint32_t rows, uint32_t cols);
using pmpp::metal::buffer;

// The build passes the chapter's shared directory (../shared), where the provided baseline lives.
static const std::string kProvided = std::string(PMPP_METAL_SHARED_DIR) + "/rowwise_naive.metal";

static id<MTLComputePipelineState> pipelineFor(Kernel k) {
  switch (k) {
    case RowMax: return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "row_max");
    case Softmax: return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "softmax_rows");
    case Argmax: return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "argmax_rows");
    case SoftmaxNaive: return pmpp::metal::pipeline(kProvided, "softmax_rows_naive");
    default: return pmpp::metal::pipeline(kProvided, "argmax_rows_naive");
  }
}

// Why this pipeline cannot run the kernel as the lab specifies it, or "". The driver's own
// copy of the rule a correct launch.mm applies: 256 threads per threadgroup, in whole
// SIMD-groups, and no more SIMD-groups than a SIMD-group has lanes.
static std::string pipelineProblem(Kernel k) {
  id<MTLComputePipelineState> p = pipelineFor(k);
  if (!p) return "the Metal kernel did not build: " + pmpp::take_error();
  if (k == SoftmaxNaive || k == ArgmaxNaive) return "";
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
static std::string launch(Kernel k, Buf* x, Buf* out, int rows, int cols, float scale,
                          int repeat = 1, pmpp::Timing* tm = nullptr) {
  std::string why = pipelineProblem(k);
  if (!why.empty()) return why;
  id<MTLComputePipelineState> p = pipelineFor(k);
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    const uint32_t r = (uint32_t)rows, c = (uint32_t)cols;
    for (int i = 0; i < repeat && ok; ++i) {
      switch (k) {
        case RowMax: ok = encodeRowMax(enc, p, buffer(x), buffer(out), r, c); break;
        case Softmax: ok = encodeSoftmaxRows(enc, p, buffer(x), buffer(out), r, c, scale); break;
        case Argmax: ok = encodeArgmaxRows(enc, p, buffer(x), buffer(out), r, c); break;
        case SoftmaxNaive: encodeSoftmaxRowsNaive(enc, p, buffer(x), buffer(out), r, c, scale); break;
        default: encodeArgmaxRowsNaive(enc, p, buffer(x), buffer(out), r, c); break;
      }
    }
  });
  if (tm) *tm = t;
  if (!ok) return "your launch.mm refused to dispatch, although this pipeline can run 256 threads per threadgroup";
  return "";
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
void launchRowMax(const float* x, float* out, int rows, int cols);
void launchSoftmaxRows(const float* x, float* y, int rows, int cols, float scale);
void launchArgmaxRows(const float* x, int* idx, int rows, int cols);
void launchSoftmaxRowsNaive(const float* x, float* y, int rows, int cols, float scale);
void launchArgmaxRowsNaive(const float* x, int* idx, int rows, int cols);
static float* f(Buf* b) { return (float*)pmpp::raw(b); }
static int* i32(Buf* b) { return (int*)pmpp::raw(b); }

static std::string pipelineProblem(Kernel k) {
  if (k == SoftmaxNaive || k == ArgmaxNaive) return "";
  const int most = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  return kBlock <= most ? "" : "this device runs at most " + std::to_string(most) + " threads per block; the lab needs 256";
}
static std::string launch(Kernel k, Buf* x, Buf* out, int rows, int cols, float scale,
                          int repeat = 1, pmpp::Timing* tm = nullptr) {
  std::string why = pipelineProblem(k);
  if (!why.empty()) return why;
  pmpp::Timing t = pmpp::run([&](void*) {
    for (int i = 0; i < repeat; ++i) {
      switch (k) {
        case RowMax: launchRowMax(f(x), f(out), rows, cols); break;
        case Softmax: launchSoftmaxRows(f(x), f(out), rows, cols, scale); break;
        case Argmax: launchArgmaxRows(f(x), i32(out), rows, cols); break;
        case SoftmaxNaive: launchSoftmaxRowsNaive(f(x), f(out), rows, cols, scale); break;
        default: launchArgmaxRowsNaive(f(x), i32(out), rows, cols); break;
      }
    }
  });
  if (tm) *tm = t;
  return "";
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

// ---------------------------------------------------------------- tolerances (README.md)
// Softmax: |got - ref| <= kAtol + rtol * ref per element, and |sum of the row - 1| <= rtol.
// For your kernel rtol = kRtol. The worst case we can bound for these rows (256 threads, at
// most 128,256 columns, exponent arguments in [-24, 0]) is about 4e-5 relative with Metal's
// fast exp and 3.4e-5 with CUDA's expf; kRtol leaves more than twice that. kAtol is far below every
// probability the test rows produce (the smallest is about 1e-14), so the check is relative in
// practice.
static const double kRtol = 1e-4;
static const double kAtol = 1e-20;
// The provided baseline adds a whole row in one thread, one rounding (2^-24 relative) per
// column at worst, so it is held to that bound on top of kRtol.
static double rtolFor(Kernel k, int cols) {
  return k == SoftmaxNaive ? kRtol + cols * std::ldexp(1.0, -24) : kRtol;
}

// ---------------------------------------------------------------- test data
enum Shape { Uniform, PlusNinety, MinusNinety, AllEqual, AllNegative, Lowest, Tie };
struct Recipe { Shape shape; int a, b; std::string what; };

// The rows a test batch is built from. Ties and the -FLT_MAX row matter to max and argmax
// only; softmax gets the first five.
static std::vector<Recipe> recipes(int cols, bool softmax) {
  std::vector<Recipe> r = {
      {Uniform, 0, 0, "logits uniform in [-12, 12]"},
      {PlusNinety, 0, 0, "logits offset by +90"},
      {MinusNinety, 0, 0, "logits offset by -90"},
      {AllEqual, 0, 0, "every logit equal"},
      {AllNegative, 0, 0, "every logit negative"},
  };
  if (softmax) return r;
  r.push_back({Lowest, 0, 0, "every logit equal to -FLT_MAX"});
  const struct { int a, b; const char* why; } pairs[] = {
      {1, 2, "neighbouring lanes"},
      {0, cols - 1, "the first and the last column"},
      {5, 261, "one thread, two loop steps"},
      {6, 261, "the higher index on the lower lane"},
      {3, 40, "two warps of 32"},
      {cols - 2, cols - 1, "the last two columns"},
  };
  for (auto& p : pairs) {
    if (p.a < 0 || p.b >= cols || p.a >= p.b) continue;
    bool dup = false;
    for (auto& q : r) dup = dup || (q.shape == Tie && q.a == p.a && q.b == p.b);
    if (dup) continue;
    r.push_back({Tie, p.a, p.b, "equal maxima at columns " + std::to_string(p.a) + " and " +
                                    std::to_string(p.b) + " (" + p.why + ")"});
  }
  return r;
}

static void fillRow(const Recipe& rc, float* row, int cols, uint32_t seed) {
  std::vector<float> u = pmpp::random_floats((size_t)cols, seed);   // uniform in [-1, 1]
  for (int c = 0; c < cols; ++c) {
    switch (rc.shape) {
      case PlusNinety: row[c] = 12.0f * u[c] + 90.0f; break;
      case MinusNinety: row[c] = 12.0f * u[c] - 90.0f; break;
      case AllEqual: row[c] = 3.25f; break;
      case AllNegative: row[c] = -12.0f * std::fabs(u[c]) - 0.5f; break;
      case Lowest: row[c] = -std::numeric_limits<float>::max(); break;
      default: row[c] = 12.0f * u[c]; break;   // Uniform, and the background of Tie
    }
  }
  if (rc.shape == Tie) row[rc.a] = row[rc.b] = 12.5f;   // above every other value, exactly equal
}

// A batch of `rows` rows of `cols` logits: row r follows recipe (start + r) % count.
struct Batch {
  int rows, cols;
  std::vector<float> x;
  std::vector<std::string> what;
};
static Batch makeBatch(const std::vector<Recipe>& rc, int start, int rows, int cols) {
  Batch b{rows, cols, std::vector<float>((size_t)rows * cols), {}};
  for (int r = 0; r < rows; ++r) {
    const Recipe& one = rc[(size_t)(start + r) % rc.size()];
    fillRow(one, &b.x[(size_t)r * cols], cols, 7919u * (uint32_t)cols + 104729u * (uint32_t)(start + r) + 17u);
    b.what.push_back(one.what);
  }
  return b;
}

// ---------------------------------------------------------------- CPU reference
static float refMax(const float* row, int cols) {
  float m = row[0];
  for (int c = 1; c < cols; ++c) m = row[c] > m ? row[c] : m;
  return m;
}
static int refArgmax(const float* row, int cols) {
  int best = 0;
  for (int c = 1; c < cols; ++c)
    if (row[c] > row[best]) best = c;   // strictly greater: ties stay with the lowest index
  return best;
}
// softmax(scale * row) in double.
static void refSoftmax(const float* row, int cols, float scale, std::vector<double>& out) {
  out.resize(cols);
  double m = -std::numeric_limits<double>::infinity();
  for (int c = 0; c < cols; ++c) m = std::max(m, (double)scale * row[c]);
  double s = 0;
  for (int c = 0; c < cols; ++c) s += (out[c] = std::exp((double)scale * row[c] - m));
  for (int c = 0; c < cols; ++c) out[c] /= s;
}

// ---------------------------------------------------------------- one launch, checked
static const int kGuard = 64;                    // elements past the end of every output
static const float kGuardF = -12345.5f;          // what they hold before the launch
static const int kGuardI = -77;
static const float kNaN = std::numeric_limits<float>::quiet_NaN();

static std::string where(const Batch& b, int r) {
  return std::to_string(b.rows) + " rows of " + std::to_string(b.cols) + ", row " + std::to_string(r) +
         " (" + b.what[r] + ")";
}
static std::string fmt(const char* f, double a, double c = 0, double d = 0) {
  char buf[200];
  std::snprintf(buf, sizeof buf, f, a, c, d);
  return buf;
}

// Row max: out[r] must equal the row's maximum exactly (a max involves no rounding).
static std::string checkMax(const Batch& b) {
  const int rows = b.rows, cols = b.cols;
  Buf* x = pmpp::alloc(b.x.size() * 4);
  Buf* out = pmpp::alloc((size_t)(rows + kGuard) * 4);
  pmpp::upload(x, b.x.data(), b.x.size() * 4);
  std::vector<float> got(rows + kGuard, kGuardF);
  for (int r = 0; r < rows; ++r) got[r] = kNaN;   // a row nobody writes shows as NaN
  pmpp::upload(out, got.data(), got.size() * 4);
  std::string err = launch(RowMax, x, out, rows, cols, 1.0f);
  if (err.empty()) {
    pmpp::download(out, got.data(), got.size() * 4);
    for (int r = 0; r < rows && err.empty(); ++r) {
      const float want = refMax(&b.x[(size_t)r * cols], cols);
      if (!(got[r] == want)) {
        err = where(b, r) + fmt(": got %.9g, expected %.9g", got[r], want);
        if (std::isnan(got[r])) err += " (NaN: out[row] was never written)";
        else if (got[r] == 0.0f && want < 0) err += " (0 is not the identity of max)";
      }
    }
    for (int g = rows; g < rows + kGuard && err.empty(); ++g)
      if (got[g] != kGuardF) err = "wrote out[" + std::to_string(g) + "], past the last row";
  }
  pmpp::release(x); pmpp::release(out);
  return err;
}

// Argmax (yours or the provided one): idx[r] must be the lowest index of the row's maximum.
static std::string checkArgmax(Kernel k, const Batch& b) {
  const int rows = b.rows, cols = b.cols;
  Buf* x = pmpp::alloc(b.x.size() * 4);
  Buf* out = pmpp::alloc((size_t)(rows + kGuard) * 4);
  pmpp::upload(x, b.x.data(), b.x.size() * 4);
  std::vector<int> got(rows + kGuard, kGuardI);
  for (int r = 0; r < rows; ++r) got[r] = -1;
  pmpp::upload(out, got.data(), got.size() * 4);
  std::string err = launch(k, x, out, rows, cols, 1.0f);
  if (err.empty()) {
    pmpp::download(out, got.data(), got.size() * 4);
    for (int r = 0; r < rows && err.empty(); ++r) {
      const float* row = &b.x[(size_t)r * cols];
      const int want = refArgmax(row, cols);
      if (got[r] == want) continue;
      err = where(b, r) + ": got index " + std::to_string(got[r]) + ", expected " + std::to_string(want);
      if (got[r] == -1) err += " (idx[row] was never written)";
      else if (got[r] < 0 || got[r] >= cols) err += " (not a column of this row)";
      else if (row[got[r]] == row[want]) err += " (equal values: ties go to the lowest index)";
      else err += fmt(" (x = %.9g there, the maximum is %.9g)", row[got[r]], row[want]);
    }
    for (int g = rows; g < rows + kGuard && err.empty(); ++g)
      if (got[g] != kGuardI) err = "wrote idx[" + std::to_string(g) + "], past the last row";
  }
  pmpp::release(x); pmpp::release(out);
  return err;
}

// Softmax (yours or the provided one), per element and per row sum.
static std::string checkSoftmax(Kernel k, const Batch& b, float scale) {
  const int rows = b.rows, cols = b.cols;
  const size_t total = (size_t)rows * cols;
  Buf* x = pmpp::alloc(total * 4);
  Buf* y = pmpp::alloc((total + kGuard) * 4);
  pmpp::upload(x, b.x.data(), total * 4);
  std::vector<float> got(total + kGuard, kGuardF);
  for (size_t i = 0; i < total; ++i) got[i] = kNaN;
  pmpp::upload(y, got.data(), got.size() * 4);
  std::string err = launch(k, x, y, rows, cols, scale);
  const std::string sc = fmt(", scale %g", scale);
  const double rtol = rtolFor(k, cols);
  if (err.empty()) {
    pmpp::download(y, got.data(), got.size() * 4);
    std::vector<double> ref;
    for (int r = 0; r < rows && err.empty(); ++r) {
      refSoftmax(&b.x[(size_t)r * cols], cols, scale, ref);
      const float* g = &got[(size_t)r * cols];
      int bad = 0, first = -1;
      double rowSum = 0;
      for (int c = 0; c < cols; ++c) {
        rowSum += g[c];
        const double e = std::fabs((double)g[c] - ref[c]);
        if (!(e <= kAtol + rtol * ref[c])) { if (first < 0) first = c; ++bad; }   // also NaN
      }
      if (bad) {
        err = where(b, r) + sc + ": " + std::to_string(bad) + " of " + std::to_string(cols) +
              " probabilities wrong; first at column " + std::to_string(first) +
              fmt(": got %.9g, expected %.9g", g[first], ref[first]);
        if (std::isnan(g[first])) err += " (NaN: never written, or 0/0 or inf/inf in the kernel)";
        else if (std::isinf(g[first])) err += " (infinity: did an exponent overflow?)";
      } else if (!(std::fabs(rowSum - 1.0) <= rtol)) {
        err = where(b, r) + sc + fmt(": the row sums to %.9g, not 1", rowSum);
      }
    }
    for (size_t i = total; i < total + kGuard && err.empty(); ++i)
      if (got[i] != kGuardF) err = "wrote y[" + std::to_string(i) + "], past the end of the output";
  }
  pmpp::release(x); pmpp::release(y);
  return err;
}

// ---------------------------------------------------------------- greedy decoding
// The "model" is a lookup table: after token t the logits are row t of a V x V table. Every
// row has its largest value at two columns (an exact tie), so each step depends on the tie
// rule. The driver keeps a batch of sequences, copies the rows of their current tokens into
// a [batch x V] logits buffer, calls your argmax, and feeds the indices back as the next
// tokens. The CPU does the same with the lowest-index rule; the sequences must match.
static std::string checkGreedy(int V, int batch, int steps) {
  std::vector<float> table((size_t)V * V);
  for (int t = 0; t < V; ++t) {
    std::vector<float> u = pmpp::random_floats((size_t)V, 50000u + (uint32_t)t);
    float* row = &table[(size_t)t * V];
    for (int v = 0; v < V; ++v) row[v] = 12.0f * u[v];
    const int a = (t * 37 + 11) % V;
    const int b = (a + 1 + (t * 101) % (V - 1)) % V;   // never a
    row[a] = row[b] = 12.5f;
  }
  std::vector<int> start(batch);
  for (int i = 0; i < batch; ++i) start[i] = (int)(((long long)i * (V - 1)) / std::max(1, batch - 1));

  // CPU sequences.
  std::vector<std::vector<int>> want(batch, std::vector<int>(steps));
  for (int i = 0; i < batch; ++i) {
    int tok = start[i];
    for (int s = 0; s < steps; ++s) want[i][s] = tok = refArgmax(&table[(size_t)tok * V], V);
  }

  // GPU sequences.
  Buf* logits = pmpp::alloc((size_t)batch * V * 4);
  Buf* idx = pmpp::alloc((size_t)batch * 4);
  std::vector<float> x((size_t)batch * V);
  std::vector<int> tok = start, got(batch);
  std::string err;
  for (int s = 0; s < steps && err.empty(); ++s) {
    for (int i = 0; i < batch; ++i)
      std::memcpy(&x[(size_t)i * V], &table[(size_t)tok[i] * V], (size_t)V * 4);
    pmpp::upload(logits, x.data(), x.size() * 4);
    std::fill(got.begin(), got.end(), -1);
    pmpp::upload(idx, got.data(), got.size() * 4);
    err = launch(Argmax, logits, idx, batch, V, 1.0f);
    if (!err.empty()) break;
    pmpp::download(idx, got.data(), got.size() * 4);
    for (int i = 0; i < batch && err.empty(); ++i) {
      if (got[i] == want[i][s]) continue;
      err = "step " + std::to_string(s) + ", sequence " + std::to_string(i) + ": your argmax chose token " +
            std::to_string(got[i]) + ", greedy decoding on the CPU chose " + std::to_string(want[i][s]);
      const float* row = &table[(size_t)tok[i] * V];
      if (got[i] >= 0 && got[i] < V && row[got[i]] == row[want[i][s]])
        err += " (they are tied in the logits of token " + std::to_string(tok[i]) + "; ties go to the lower index)";
    }
    tok = got;
  }
  pmpp::release(logits); pmpp::release(idx);
  return err;
}

// ---------------------------------------------------------------- tests
static std::vector<int> pick(const std::vector<int>& race, const std::vector<int>& emu, const std::vector<int>& dev) {
  const std::string b = pmpp::backend();
  return b == "race" ? race : b == "emu" ? emu : dev;
}

// Runs `one` on batches that between them put every recipe in every row count. With one
// row per batch every recipe is row 0 once; with more rows the recipes are mixed, so a kernel
// that reads the wrong row is caught.
static std::string overBatches(int cols, const std::vector<int>& rowCounts, bool softmax,
                               const std::function<std::string(const Batch&)>& one) {
  const std::vector<Recipe> rc = recipes(cols, softmax);
  for (int rows : rowCounts)
    for (int start = 0; start < (int)rc.size(); start += rows) {
      std::string err = one(makeBatch(rc, start, rows, cols));
      if (!err.empty()) return err;
    }
  return "";
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::vector<int> rowCounts = pick({2}, {1, 3}, {1, 4, 7});
  const std::vector<int> colCounts = pick({1, 33, 257, 1000}, {1, 2, 31, 32, 33, 255, 256, 257, 1000, 4099},
                                          {1, 2, 31, 32, 33, 255, 256, 257, 1000, 4096, 32000, 50257, 128256});
  const std::vector<float> scales = std::string(pmpp::backend()) == "race" ? std::vector<float>{1.0f}
                                                                          : std::vector<float>{1.0f, 0.5f, 0.125f};

  // A kernel this device cannot run as specified is reported once, not once per shape.
  auto runnable = [&](Kernel k, const std::string& label) {
    std::string why = pipelineProblem(k);
    if (why.empty()) return true;
    t.check(label, [&] { return why; });
    return false;
  };

  if (runnable(RowMax, "row max: kernel usable"))
    for (int cols : colCounts)
      t.check("row max, cols " + std::to_string(cols), [&] {
        return overBatches(cols, rowCounts, false, [](const Batch& b) { return checkMax(b); });
      });

  if (runnable(Argmax, "argmax: kernel usable"))
    for (int cols : colCounts)
      t.check("argmax, cols " + std::to_string(cols), [&] {
        return overBatches(cols, rowCounts, false, [](const Batch& b) { return checkArgmax(Argmax, b); });
      });

  if (runnable(Softmax, "softmax: kernel usable"))
    for (int cols : colCounts)
      t.check("softmax, cols " + std::to_string(cols), [&] {
        for (float scale : scales) {
          std::string err = overBatches(cols, rowCounts, true, [&](const Batch& b) { return checkSoftmax(Softmax, b, scale); });
          if (!err.empty()) return err;
        }
        return std::string();
      });

  // The provided baseline, so that a broken provided file shows here and not in the benchmark.
  for (int cols : pick({33}, {1, 257, 1000}, {1, 257, 4096, 32000})) {
    t.check("provided one-thread-per-row argmax, cols " + std::to_string(cols), [&] {
      return overBatches(cols, rowCounts, false, [](const Batch& b) { return checkArgmax(ArgmaxNaive, b); });
    });
    t.check("provided one-thread-per-row softmax, cols " + std::to_string(cols), [&] {
      return overBatches(cols, rowCounts, true, [](const Batch& b) { return checkSoftmax(SoftmaxNaive, b, 1.0f); });
    });
  }

  // Milestone M4: greedy decoding end to end on the lookup-table model.
  const int V = pick({100}, {300}, {2053})[0];
  t.check("greedy decoding: batch 4, 16 steps, vocabulary " + std::to_string(V),
          [&] { return checkGreedy(V, 4, 16); });
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

static int bench(const pmpp::Args& a) {
  const double mem = deviceMemory();
  const double budget = mem > 0 ? mem / 4 : 1e9;   // bytes the inputs and outputs of one shape may take
  const std::vector<int> rowsList = {1, 8, 64};
  const std::vector<int> colsList = {32000, 128256};
  int rc = 0;

  for (int fam = 0; fam < 2; ++fam) {
    const bool sm = fam == 0;
    const Kernel mine = sm ? Softmax : Argmax, naive = sm ? SoftmaxNaive : ArgmaxNaive;
    const double bytesPer = sm ? 8.0 : 4.0;
    std::string why = pipelineProblem(mine);
    if (!why.empty()) { std::printf("%s: %s\n", sm ? "softmax" : "argmax", why.c_str()); rc = 1; continue; }
    std::printf("\n%s, best of %d timed runs\n", sm ? "softmax, scale 1" : "argmax", a.reps);
    std::printf("  %5s %7s  %-6s %6s %12s %9s %10s\n", "rows", "cols", "kernel", "calls", "us per call", "GB/s", "vs naive");
    for (int cols : colsList)
      for (int rows : rowsList) {
        const size_t total = (size_t)rows * cols;
        if ((double)total * (4 + (sm ? 4 : 0)) > budget) {
          std::printf("  %5d %7d  skipped: larger than a quarter of the device memory reported\n", rows, cols);
          continue;
        }
        std::vector<float> x = pmpp::random_floats(total, 31u + (uint32_t)(rows * 7 + cols));
        for (float& v : x) v *= 12.0f;
        Buf* in = pmpp::alloc(total * 4);
        Buf* out = pmpp::alloc((sm ? total : (size_t)rows) * 4);
        pmpp::upload(in, x.data(), total * 4);
        double naiveS = 0;
        for (int which = 1; which >= 0; --which) {   // naive first, so "vs naive" can be printed
          const Kernel k = which ? naive : mine;
          const char* key = which ? "naive" : "block";
          int calls = 0;
          std::string err;
          double s = perCall(a.reps, [&](int c, pmpp::Timing* tm) { return launch(k, in, out, rows, cols, 1.0f, c, tm); },
                             &calls, &err);
          if (s <= 0) { std::printf("  %5d %7d  %-6s failed: %s\n", rows, cols, key, err.c_str()); rc = 1; continue; }
          // Check what the timed calls produced before reporting their time.
          std::string bad;
          if (sm) {
            std::vector<float> got(total);
            pmpp::download(out, got.data(), total * 4);
            std::vector<double> ref;
            for (int r = 0; r < rows && bad.empty(); ++r) {
              refSoftmax(&x[(size_t)r * cols], cols, 1.0f, ref);
              for (int c = 0; c < cols; ++c)
                if (!(std::fabs((double)got[(size_t)r * cols + c] - ref[c]) <= kAtol + rtolFor(k, cols) * ref[c])) {
                  bad = "row " + std::to_string(r) + " wrong";
                  break;
                }
            }
          } else {
            std::vector<int> got(rows);
            pmpp::download(out, got.data(), (size_t)rows * 4);
            for (int r = 0; r < rows && bad.empty(); ++r)
              if (got[r] != refArgmax(&x[(size_t)r * cols], cols)) bad = "row " + std::to_string(r) + " wrong";
          }
          if (!bad.empty()) { std::printf("  %5d %7d  %-6s %s; not reported\n", rows, cols, key, bad.c_str()); rc = 1; continue; }
          const double gbps = total * bytesPer / s / 1e9;
          if (which) naiveS = s;
          std::printf("  %5d %7d  %-6s %6d %12.2f %9.2f", rows, cols, key, calls, s * 1e6, gbps);
          if (!which && naiveS > 0) std::printf(" %9.1fx", naiveS / s);
          std::printf("\n");
          pmpp::Json j = pmpp::bench_row(a, sm ? "softmax" : "argmax");
          j.s("kernel", key).n("rows", rows).n("cols", cols).n("seconds", s);
          if (sm) j.n("gb_per_s", gbps);
          j.emit();
        }
        pmpp::release(in); pmpp::release(out);
      }
  }
  std::printf("\nGB/s counts the minimum traffic: 8 bytes per element for softmax (read x once, write y once),\n"
              "4 for argmax (read x once). Your softmax reads each row three times, so it moves more than that.\n"
              "\"block\" is your kernel; \"naive\" is the provided one thread per row.\n"
              "These are measurements of this machine, at these sizes.\n");
  return rc;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
