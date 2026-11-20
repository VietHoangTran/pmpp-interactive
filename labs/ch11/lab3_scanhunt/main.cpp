// Lab 11.3 driver: tests a row-wise exclusive scan that ships with three defects.
// Your work goes in work/. You do not need to edit this file.
//
// out[r][j] must equal in[r][0] + ... + in[r][j-1], and out[r][0] must be exactly 0. The
// rows of a batch hold different kinds of data, so a kernel that reads the wrong row fails:
//   ones       every element 1, so out[r][j] = j;
//   integers   whole numbers from -3 to 3;
//   uniform    uniform in [-1, 1].
// For the first two every partial sum is a whole number far below 2^24 in magnitude, which a
// float holds exactly, so every order of addition gives the same float: they are compared
// exactly, and a difference is a defect, never rounding. Uniform rows are compared with the
// tolerance derived in README.md.
//
// On a device each case is launched several times on fresh data, because a defect that
// depends on timing can spare one launch and not the next; a failure says how many launches
// were wrong.
#include <cmath>
#include <limits>

#include "pmpp/harness.hpp"
using pmpp::Buf;

static const int kCoarse = 4;   // COARSE_FACTOR in kernel.cu and kernel.metal: elements per thread per chunk

// ---- entry points implemented in <variant>/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
bool encodeExclusiveRowScan(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> in,
                            id<MTLBuffer> out, uint32_t rows, uint32_t cols, uint32_t block);
static const char* kBlockWord = "threadgroup";

static id<MTLComputePipelineState> scanPipeline() {
  return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "exclusive_row_scan");
}
// Why this device cannot run `block` threads as the kernel needs them, or "". The driver's
// own copy of the rule launch.mm applies.
static std::string blockProblem(int block) {
  id<MTLComputePipelineState> p = scanPipeline();
  if (!p) return "the Metal kernel did not build: " + pmpp::take_error();
  const NSUInteger w = p.threadExecutionWidth;
  char b[200];
  if ((NSUInteger)block > p.maxTotalThreadsPerThreadgroup || w == 0 || block % w != 0 || (NSUInteger)block / w > w) {
    std::snprintf(b, sizeof b, "this pipeline (SIMD-group width %lu, at most %lu threads) cannot run %d threads "
                  "as whole SIMD-groups with no more groups than lanes", (unsigned long)w,
                  (unsigned long)p.maxTotalThreadsPerThreadgroup, block);
    return b;
  }
  return "";
}
static std::string runScan(Buf* in, Buf* out, int rows, int cols, int block) {
  id<MTLComputePipelineState> pso = scanPipeline();
  if (!pso) return "the Metal kernel did not build: " + pmpp::take_error();
  bool ok = true;
  pmpp::run([&](void* v) {
    ok = encodeExclusiveRowScan(((pmpp::metal::Pass*)v)->encoder, pso, pmpp::metal::buffer(in),
                                pmpp::metal::buffer(out), (uint32_t)rows, (uint32_t)cols, (uint32_t)block);
  });
  return ok ? "" : "encodeExclusiveRowScan refused to dispatch, although this pipeline can run the block";
}
#else
void launchExclusiveRowScan(const float* in, float* out, int rows, int cols, int block);
static const char* kBlockWord = "block";

static std::string blockProblem(int block) {
  const int most = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  const int warp = (int)pmpp::device_prop("warpSize", 32);
  if (block > most) return "this device runs at most " + std::to_string(most) + " threads per block";
  if (warp <= 0 || block % warp != 0 || block / warp > warp)
    return "warps of " + std::to_string(warp) + " lanes: " + std::to_string(block) +
           " threads are not whole warps with no more warps than lanes";
  return "";
}
static std::string runScan(Buf* in, Buf* out, int rows, int cols, int block) {
  pmpp::run([&](void*) { launchExclusiveRowScan((const float*)pmpp::raw(in), (float*)pmpp::raw(out), rows, cols, block); });
  return "";
}
#endif

// ---------------------------------------------------------------- data and reference
enum Kind { Ones, Integers, Uniform };
static const char* kKindName[] = {"ones", "integers from -3 to 3", "uniform in [-1, 1]"};

static void fillRow(Kind k, float* row, int cols, uint32_t seed) {
  std::vector<float> u = pmpp::random_floats((size_t)cols, seed);   // uniform in [-1, 1]
  for (int c = 0; c < cols; ++c) {
    if (k == Ones) {
      row[c] = 1.0f;
    } else if (k == Integers) {
      int v = (int)std::floor((u[c] + 1.0f) * 3.5f);   // 0 .. 7; 7 only for u = 1 exactly
      row[c] = (float)(std::min(v, 6) - 3);             // -3 .. 3
    } else {
      row[c] = u[c];
    }
  }
}

struct Batch {
  int rows, cols;
  std::vector<float> x;
  std::vector<Kind> kind;
};
// Row r holds kind (first + r) % 3.
static Batch makeBatch(int rows, int cols, int first, uint32_t seed) {
  Batch b{rows, cols, std::vector<float>((size_t)rows * cols), {}};
  for (int r = 0; r < rows; ++r) {
    Kind k = (Kind)((first + r) % 3);
    fillRow(k, &b.x[(size_t)r * cols], cols, seed + 7919u * (uint32_t)r);
    b.kind.push_back(k);
  }
  return b;
}

static int chunksPerRow(int cols, int block) { return (cols + kCoarse * block - 1) / (kCoarse * block); }

// ---------------------------------------------------------------- one launch, checked
static const int kGuard = 64;                 // words past the end of the output
static const float kGuardValue = -12345.5f;   // what they hold before the launch

// Launches once on `b` and compares. Empty string = correct.
static std::string runOnce(const Batch& b, int block) {
  const int rows = b.rows, cols = b.cols;
  const size_t total = (size_t)rows * cols;
  Buf* in = pmpp::alloc(total * sizeof(float));
  Buf* out = pmpp::alloc((total + kGuard) * sizeof(float));
  pmpp::upload(in, b.x.data(), total * sizeof(float));
  std::vector<float> got(total + kGuard, kGuardValue);
  for (size_t i = 0; i < total; ++i) got[i] = std::numeric_limits<float>::quiet_NaN();   // never written shows as NaN
  pmpp::upload(out, got.data(), got.size() * sizeof(float));

  std::string err = runScan(in, out, rows, cols, block);
  if (err.empty()) err = pmpp::take_error();
  if (!err.empty()) err = "launch error: " + err;
  if (err.empty()) {
    pmpp::download(out, got.data(), got.size() * sizeof(float));
    // Tolerance for uniform rows (README.md): lab 11.1's form, rel * sum|x| + 1e-6, with the
    // relative term grown for the carry. Every output is a sum computed along a tree at most
    // `depth` additions deep, so its error is at most about depth * 2^-24 times the sum of the
    // magnitudes it adds up; twice that, to spare rounding in the bound itself.
    const double depth = chunksPerRow(cols, block) + 16;
    const double rel = 2.0 * depth * std::ldexp(1.0, -24);
    for (int r = 0; r < rows && err.empty(); ++r) {
      const float* x = &b.x[(size_t)r * cols];
      const float* g = &got[(size_t)r * cols];
      const bool exact = b.kind[r] != Uniform;
      double ref = 0.0, mag = 0.0;   // Fig. 11.1's loop, in double, made exclusive
      for (int c = 0; c < cols; ++c) {
        const double tol = exact ? 0.0 : rel * mag + 1e-6;
        const double e = std::fabs((double)g[c] - ref);
        bool bad = !(e <= tol);                         // also catches NaN
        if (c == 0 && !(g[c] == 0.0f)) bad = true;      // the identity, exactly
        if (bad) {
          char m[320];
          std::snprintf(m, sizeof m, "%d rows of %d, row %d (%s), column %d: got %.9g, expected %.9g", rows, cols, r,
                        kKindName[b.kind[r]], c, (double)g[c], ref);
          err = m;
          if (std::isnan(g[c])) err += " (NaN: never written)";
          else if (c == 0) err += " (the first element of an exclusive scan is 0, exactly)";
          else if (!exact) {
            std::snprintf(m, sizeof m, " (off by %.3g, the tolerance here is %.3g)", e, tol);
            err += m;
          }
          break;
        }
        ref += x[c];
        mag += std::fabs((double)x[c]);
      }
    }
    for (size_t i = total; i < total + kGuard && err.empty(); ++i)
      if (got[i] != kGuardValue) err = "wrote out[" + std::to_string(i) + "], past the end of the output";
  }
  pmpp::release(in);
  pmpp::release(out);
  return err;
}

// Every kind of row in every row position, `launches` times with fresh data.
static std::string runCase(int rows, int cols, int block, int launches) {
  int wrong = 0, total = 0;
  std::string first;
  for (int rep = 0; rep < launches; ++rep) {
    for (int start = 0; start < 3; start += rows) {
      ++total;
      Batch b = makeBatch(rows, cols, start + rep, 1000u + 31u * (uint32_t)cols + 101u * (uint32_t)rep + (uint32_t)start);
      std::string e = runOnce(b, block);
      if (e.empty()) continue;
      if (wrong++ == 0) first = e;
      if (e.compare(0, 13, "launch error:") == 0) return e;
    }
  }
  if (wrong == 0) return "";
  if (total == 1) return first;
  return "wrong in " + std::to_string(wrong) + " of " + std::to_string(total) + " launches. The first: " + first;
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::string be = pmpp::backend();
  const bool race = be == "race", emu = pmpp::is_emu();
  // The emulators run every emulated thread on the CPU (race: as OS threads under
  // ThreadSanitizer), so their rows stay short. With 64 threads a chunk is 256 elements:
  // 1000 columns is 4 chunks, 4097 is 17.
  const std::vector<int> blocks = race ? std::vector<int>{64} : std::vector<int>{64, 256};
  const std::vector<int> rowCounts = race ? std::vector<int>{2} : std::vector<int>{1, 3};
  const std::vector<int> colCounts = race ? std::vector<int>{1, 33, 1000}
                                    : emu ? std::vector<int>{1, 33, 1000, 4097}
                                          : std::vector<int>{1, 33, 1000, 4097, 32000};
  const int launches = emu ? 1 : 10;
  for (int block : blocks) {
    const std::string why = blockProblem(block);
    if (!why.empty()) {
      t.check(std::to_string(block) + " threads per " + kBlockWord + ": usable here", [&] { return why; });
      continue;
    }
    for (int rows : rowCounts)
      for (int cols : colCounts) {
        const int chunks = chunksPerRow(cols, block);
        std::string label = std::to_string(block) + " threads per " + kBlockWord + ", " + std::to_string(rows) +
                            (rows == 1 ? " row of " : " rows of ") + std::to_string(cols) + " (" +
                            std::to_string(chunks) + (chunks == 1 ? " chunk" : " chunks") + " per row)";
        t.check(label, [&] { return runCase(rows, cols, block, launches); });
      }
  }
  return t.finish();
}

static int bench(const pmpp::Args&) {
  std::printf("this lab has no benchmark.\n");
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
