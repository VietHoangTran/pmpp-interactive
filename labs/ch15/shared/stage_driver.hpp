// Labs 15.1 to 15.5, provided: the tests and the benchmark shared by the five
// book-layout stages of milestone M3. You do not need to edit this file. Each lab's
// main.cpp names its kernel and the previous stage's, then calls ch15::stageMain.
//
// TESTS (every backend)
// - Shapes (M, K, N), each chosen to break one thing; testShapes() says which. The CPU
//   emulators run small ones only; real devices add large and awkward ones.
// - C is filled with NaN before every run and is followed, in the same allocation, by a
//   guard band of 128 rows (128 x N floats) holding a sentinel. An element the kernel
//   never writes stays NaN; a store past the end of C changes the guard band. Both fail on
//   every backend. A and B are allocated to their exact sizes (one float when K = 0), so
//   on the emulator AddressSanitizer names the source line of any read past their ends.
// - The result is compared with a CPU reference accumulated in double, with the harness's
//   tolerance: |got - ref| <= 1e-4 * (1 + sum over k of |a*b|) (pmpp/harness.hpp).
//   Metal does not promise strict IEEE arithmetic by default, so exact equality would be
//   the wrong test.
// - A stage whose kernel is a template over the warp width (labs 15.3 to 15.5) is tested
//   at width 32 and at width 64 on every backend, the emulators included: the layout is
//   index arithmetic, so whether it is correct does not depend on the hardware's width.
//   On CUDA and HIP the launch that picks the device's own width is tested as well.
// - On a GPU a stage may ask for several launches per case (lab 15.5: 10), with C refilled
//   with NaN before each; a failure says in how many launches the result was wrong.
// - One case has K = 0: an empty sum, so C must come out all zeros (it starts as NaN). It
//   runs under a 30-second watchdog (watchdog.hpp): a tile loop whose bound numTiles - 1
//   wraps around fails there with that reason instead of running for hours.
// - Metal: the pipeline must accept 256 threads per threadgroup and its static threadgroup
//   memory must fit the device's limit, or the test fails and says which.
//
// BENCHMARK (real devices only)
// - Kernels, timed in turn so that drift on the machine affects them alike, best of
//   --reps runs each: lab 5.3's solution (milestone M2, T = 16), the previous stage's
//   solution (from lab 15.2 on), this stage, and the vendor GEMM (cuBLAS, hipBLAS or MPS;
//   its math mode is printed). Each is first checked against the CPU reference on one
//   awkward shape; one that fails there is not timed.
// - Shapes: squares 1024, 2048 and 4096 (no larger square once lab 5.3 needs more than
//   2 s, or once the three matrices would take more than a quarter of the device memory
//   the backend reports), the 1024 -> 4096 linear layer at batch 1, 16, 256 and 1024, and
//   the short-K shape 4096 x 128 x 4096.
// - Rows (@pmpp bench): tiled_5_3, previous, stage, vendor, each with shape, m, k, n,
//   seconds, gflops (stage also with width when the kernel has one); occupancy on CUDA and
//   HIP (blocks_per_sm from cudaOccupancyMaxActiveBlocksPerMultiprocessor,
//   threads_per_block, multiprocessors); pipeline on Metal (max_threads_per_threadgroup,
//   simd_width, static_threadgroup_bytes).
// - Times are what this machine measured. Nothing here is an expected value.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "pmpp/harness.hpp"
#include "watchdog.hpp"

#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
// Lab 5.3's solution, copied into the build by ch15.cmake.
void encodeGemm(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> A, id<MTLBuffer> B,
                id<MTLBuffer> C, uint32_t j, uint32_t k, uint32_t l, uint32_t tileWidth);
// vendor_mps.mm
std::string ch15VendorName();
std::string ch15VendorEncode(pmpp::metal::Pass* pass, id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C,
                             uint32_t M, uint32_t K, uint32_t N);
#else
// Lab 5.3's solution, copied into the build by ch15.cmake.
void launchGemm(float* A, float* B, float* C, int j, int k, int l);
// vendor_blas.cu
std::string ch15VendorName();
std::string ch15VendorGemm(const float* A, const float* B, float* C, int M, int K, int N);
// stage_occupancy.cu
int ch15StageBlocksPerSm(int width, int threadsPerBlock);
#endif

namespace ch15 {

using pmpp::Buf;

constexpr int kThreads = 256;     // threads per block in every stage (the book's NUM_THREADS_PER_BLOCK)
constexpr int kBlockTile = 128;   // bM = bN in every stage
constexpr int kLab53Tile = 16;    // lab 5.3's TILE_WIDTH in the comparison
constexpr int kGuardRows = 128;   // a block can run past the last row of C by at most bM - 1 rows
// The widths the warp-width templates (labs 15.3 to 15.5) are tested at: 32 (NVIDIA, and
// AMD RDNA) and 64 (AMD CDNA) (notes/today/ch15-facts.md B1).
constexpr int kTestWidths[] = {32, 64};

// One way of computing C = A B on the device. Returns the time of one run; a time <= 0
// means it did not run.
using Runner = std::function<pmpp::Timing(Buf* A, Buf* B, Buf* C, int M, int K, int N)>;

#if defined(PMPP_BACKEND_METAL)
using EncodeFn = void (*)(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> A,
                          id<MTLBuffer> B, id<MTLBuffer> C, uint32_t M, uint32_t K, uint32_t N);
struct Kernel {
  const char* label = nullptr;     // "15.2"; nullptr means there is no such kernel
  const char* path = nullptr;      // its .metal source
  const char* function = nullptr;  // its kernel function
  EncodeFn encode = nullptr;       // its encode function (launch.mm)
  bool simdWidth = false;          // compiled with SIMD_WIDTH=<w> (labs 15.3 to 15.5)
};
#else
using LaunchFn = void (*)(const float* A, const float* B, float* C, int M, int K, int N);
using LaunchWidthFn = void (*)(const float* A, const float* B, float* C, int M, int K, int N, int width);
struct Kernel {
  const char* label = nullptr;          // "15.2"; nullptr means there is no such kernel
  LaunchFn launch = nullptr;            // the layout for the device's own warp width
  LaunchWidthFn launchWidth = nullptr;  // a forced width (labs 15.3 to 15.5), for the tests
};
#endif

struct Stage {
  const char* lab = "";    // "15.1"
  const char* title = "";
  Kernel self;             // the kernel in work/ or solution/
  Kernel previous;         // the previous stage's solution; no label for lab 15.1
  int gpuLaunches = 1;     // launches per test case on a GPU (lab 15.5 asks for 10)
};

// A kernel ready to run: a runner, or the reason it cannot run here.
struct Candidate {
  std::string label;       // "width 32", ...; empty when there is only one way to run it
  int width = 0;           // the warp width of the layout, 0 when the kernel has none
  std::string unusable;    // empty: usable
  Runner run;
  std::shared_ptr<std::string> lastError = std::make_shared<std::string>();   // vendor only
};

inline std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
inline std::string fmt(const char* f, ...) {
  char b[512];
  va_list ap;
  va_start(ap, f);
  std::vsnprintf(b, sizeof b, f, ap);
  va_end(ap);
  return b;
}

// ------------------------------------------------------------------ backends
#if defined(PMPP_BACKEND_METAL)
inline id<MTLComputePipelineState> metalPipeline(const Kernel& k, int width) {
  if (!k.simdWidth) return pmpp::metal::pipeline(k.path, k.function);
  return pmpp::metal::pipeline(k.path, k.function, {{"SIMD_WIDTH", std::to_string(width)}});
}
// Why this pipeline cannot run a 256-thread threadgroup here, or "".
inline std::string metalUnusable(id<MTLComputePipelineState> p, const char* function) {
  if (!p) return std::string(function) + " did not compile (the compiler's message is printed above)";
  // Metal may give a pipeline a lower thread limit than the kernel asks for
  // (notes/today/ch15-facts.md C3).
  if ((int)p.maxTotalThreadsPerThreadgroup < kThreads)
    return fmt("the pipeline of %s accepts at most %d threads per threadgroup on this device; the kernel needs %d",
               function, (int)p.maxTotalThreadsPerThreadgroup, kThreads);
  if (p.staticThreadgroupMemoryLength > pmpp::metal::device().maxThreadgroupMemoryLength)
    return fmt("%s declares %d bytes of threadgroup memory; this device allows %d", function,
               (int)p.staticThreadgroupMemoryLength, (int)pmpp::metal::device().maxThreadgroupMemoryLength);
  return "";
}
inline Runner metalRunner(id<MTLComputePipelineState> p, EncodeFn encode) {
  return [p, encode](Buf* A, Buf* B, Buf* C, int M, int K, int N) {
    return pmpp::run([&](void* v) {
      encode(((pmpp::metal::Pass*)v)->encoder, p, pmpp::metal::buffer(A), pmpp::metal::buffer(B),
             pmpp::metal::buffer(C), (uint32_t)M, (uint32_t)K, (uint32_t)N);
    });
  };
}
inline id<MTLComputePipelineState> lab53Pipeline() {
  return pmpp::metal::pipeline(PMPP_CH15_LAB53_METAL, "gemm", {{"TILE_WIDTH", std::to_string(kLab53Tile)}});
}
// The SIMD-group width this device runs the kernel's pipeline with. It is a property of a
// compiled pipeline, not of the device (ch04-facts.md C2; ch15-facts.md C1), so the kernel
// is built with the width lab 5.3's pipeline reports, and rebuilt once if its own
// pipeline reports another.
inline int metalDeviceWidth(const Kernel& k) {
  id<MTLComputePipelineState> ref = lab53Pipeline();
  int w = ref ? (int)ref.threadExecutionWidth : kTestWidths[0];
  id<MTLComputePipelineState> p = metalPipeline(k, w);
  if (p && (int)p.threadExecutionWidth != w) w = (int)p.threadExecutionWidth;
  return w;
}
#else
inline Runner cudaRunner(LaunchFn f) {
  return [f](Buf* A, Buf* B, Buf* C, int M, int K, int N) {
    return pmpp::run([&](void*) {
      f((const float*)pmpp::raw(A), (const float*)pmpp::raw(B), (float*)pmpp::raw(C), M, K, N);
    });
  };
}
inline Runner cudaWidthRunner(LaunchWidthFn f, int width) {
  return [f, width](Buf* A, Buf* B, Buf* C, int M, int K, int N) {
    return pmpp::run([&](void*) {
      f((const float*)pmpp::raw(A), (const float*)pmpp::raw(B), (float*)pmpp::raw(C), M, K, N, width);
    });
  };
}
#endif

// The ways the tests run a stage's kernel: one, or one per tested warp width (plus, on CUDA
// and HIP, the launch that picks the device's width).
inline std::vector<Candidate> testCandidates(const Kernel& k) {
  std::vector<Candidate> out;
#if defined(PMPP_BACKEND_METAL)
  if (!k.simdWidth) {
    id<MTLComputePipelineState> p = metalPipeline(k, 0);
    Candidate c; c.unusable = metalUnusable(p, k.function); c.run = metalRunner(p, k.encode);
    out.push_back(c);
    return out;
  }
  for (int w : kTestWidths) {
    id<MTLComputePipelineState> p = metalPipeline(k, w);
    Candidate c; c.label = fmt("width %d", w); c.width = w;
    c.unusable = metalUnusable(p, k.function); c.run = metalRunner(p, k.encode);
    out.push_back(c);
  }
#else
  if (!k.launchWidth) {
    Candidate c; c.run = cudaRunner(k.launch);
    out.push_back(c);
    return out;
  }
  for (int w : kTestWidths) {
    Candidate c; c.label = fmt("width %d", w); c.width = w; c.run = cudaWidthRunner(k.launchWidth, w);
    out.push_back(c);
  }
  if (!pmpp::is_emu()) {
    const int w = (int)pmpp::device_prop("warpSize", 0);
    Candidate c; c.label = fmt("the device's width, %d", w); c.width = w; c.run = cudaRunner(k.launch);
    out.push_back(c);
  }
#endif
  return out;
}

// The way the benchmark runs a kernel: at the device's own width.
inline Candidate benchCandidate(const Kernel& k) {
  Candidate c;
#if defined(PMPP_BACKEND_METAL)
  c.width = k.simdWidth ? metalDeviceWidth(k) : 0;
  id<MTLComputePipelineState> p = metalPipeline(k, c.width);
  c.unusable = metalUnusable(p, k.function);
  c.run = metalRunner(p, k.encode);
#else
  c.width = k.launchWidth ? (int)pmpp::device_prop("warpSize", 0) : 0;
  c.run = cudaRunner(k.launch);
#endif
  return c;
}

inline Candidate lab53Candidate() {
  Candidate c;
#if defined(PMPP_BACKEND_METAL)
  id<MTLComputePipelineState> p = lab53Pipeline();
  if (!p) c.unusable = "lab 5.3's kernel did not compile";
  else if ((int)p.maxTotalThreadsPerThreadgroup < kLab53Tile * kLab53Tile)
    c.unusable = fmt("lab 5.3's pipeline accepts only %d threads per threadgroup", (int)p.maxTotalThreadsPerThreadgroup);
  c.run = [p](Buf* A, Buf* B, Buf* C, int M, int K, int N) {
    return pmpp::run([&](void* v) {
      encodeGemm(((pmpp::metal::Pass*)v)->encoder, p, pmpp::metal::buffer(A), pmpp::metal::buffer(B),
                 pmpp::metal::buffer(C), (uint32_t)M, (uint32_t)K, (uint32_t)N, kLab53Tile);
    });
  };
#else
  c.run = [](Buf* A, Buf* B, Buf* C, int M, int K, int N) {
    return pmpp::run([&](void*) {
      launchGemm((float*)pmpp::raw(A), (float*)pmpp::raw(B), (float*)pmpp::raw(C), M, K, N);
    });
  };
#endif
  return c;
}

inline Candidate vendorCandidate() {
  Candidate c;
  std::shared_ptr<std::string> err = c.lastError;
#if defined(PMPP_BACKEND_METAL)
  c.run = [err](Buf* A, Buf* B, Buf* C, int M, int K, int N) {
    pmpp::Timing t = pmpp::run([&](void* v) {
      *err = ch15VendorEncode((pmpp::metal::Pass*)v, pmpp::metal::buffer(A), pmpp::metal::buffer(B),
                              pmpp::metal::buffer(C), (uint32_t)M, (uint32_t)K, (uint32_t)N);
    });
    return err->empty() ? t : pmpp::Timing{0, 0};
  };
#else
  c.run = [err](Buf* A, Buf* B, Buf* C, int M, int K, int N) {
    pmpp::Timing t = pmpp::run([&](void*) {
      *err = ch15VendorGemm((const float*)pmpp::raw(A), (const float*)pmpp::raw(B), (float*)pmpp::raw(C), M, K, N);
    });
    return err->empty() ? t : pmpp::Timing{0, 0};
  };
#endif
  return c;
}

// ------------------------------------------------------------------ test problem
// C = A B with A and B allocated exactly, and C followed by a guard band of kGuardRows rows.
struct TestProblem {
  static constexpr float kSentinel = -31337.25f;
  int M, K, N;
  std::vector<float> a, b, ref, mag;
  Buf *A = nullptr, *B = nullptr, *C = nullptr;
  size_t cElems, guardElems;
  TestProblem(int M_, int K_, int N_, uint32_t seed) : M(M_), K(K_), N(N_) {
    cElems = (size_t)M * N;
    guardElems = (size_t)kGuardRows * N;
    a = pmpp::random_floats((size_t)M * K, seed);
    b = pmpp::random_floats((size_t)K * N, seed + 1);
    // With K = 0, A and B hold nothing. They still get one float each: CUDA documents no
    // behaviour for cudaMalloc(0), and the kernel must not read them anyway.
    A = pmpp::alloc(std::max<size_t>(a.size(), 1) * 4);
    B = pmpp::alloc(std::max<size_t>(b.size(), 1) * 4);
    C = pmpp::alloc((cElems + guardElems) * 4);
    if (!a.empty()) pmpp::upload(A, a.data(), a.size() * 4);
    if (!b.empty()) pmpp::upload(B, b.data(), b.size() * 4);
    pmpp::cpu_gemm(a.data(), b.data(), M, K, N, ref, &mag);
  }
  TestProblem(const TestProblem&) = delete;
  ~TestProblem() { pmpp::release(A); pmpp::release(B); pmpp::release(C); }
  // NaN in C, the sentinel in the guard band.
  void poison() {
    std::vector<float> c(cElems + guardElems, std::numeric_limits<float>::quiet_NaN());
    std::fill(c.begin() + (long)cElems, c.end(), kSentinel);
    pmpp::upload(C, c.data(), c.size() * 4);
  }
  // Empty string if C is right and the guard band untouched; else what is wrong.
  std::string check() const {
    std::vector<float> got(cElems + guardElems);
    pmpp::download(C, got.data(), got.size() * 4);
    size_t unwritten = 0, wrong = 0, firstWrong = 0, guard = 0, firstGuard = 0;
    double maxErr = 0;
    for (size_t i = 0; i < cElems; ++i) {
      if (std::isnan(got[i])) { ++unwritten; continue; }
      const double e = std::fabs((double)got[i] - (double)ref[i]);
      if (!(e <= 1e-4 * (1.0 + mag[i]))) {        // the tolerance of pmpp::compare
        if (wrong == 0) firstWrong = i;
        ++wrong;
        if (e > maxErr) maxErr = e;
      }
    }
    for (size_t i = 0; i < guardElems; ++i) {
      if (std::memcmp(&got[cElems + i], &kSentinel, sizeof(float)) != 0) {
        if (guard == 0) firstGuard = i;
        ++guard;
      }
    }
    std::string msg;
    auto add = [&](const std::string& s) { msg += (msg.empty() ? "" : "; ") + s; };
    if (unwritten) add(fmt("%zu of %zu elements of C never written (still NaN)", unwritten, cElems));
    if (wrong)
      add(fmt("%zu of %zu elements wrong, first at row %zu, col %zu (max abs error %.4g)", wrong, cElems,
              firstWrong / (size_t)N, firstWrong % (size_t)N, maxErr));
    if (guard)
      add(fmt("%zu stores past the end of C, first at row M + %zu, col %zu", guard, firstGuard / (size_t)N,
              firstGuard % (size_t)N));
    return msg;
  }
};

struct Shape { int M, K, N; const char* why; };

// Why a K = 0 case that never finishes failed.
inline const char* kEmptySumWhy =
    "with K = 0 there are no tiles; does numTiles - 1 wrap around (an unsigned numTiles)?";

inline std::vector<Shape> testShapes() {
  const std::string b = pmpp::backend();
  if (b == "race") {
    // The race backend runs a 256-thread block at a time under ThreadSanitizer: slow.
    return {{128, 24, 128, "one block, three tile pairs"},
            {130, 9, 131, "partial tiles everywhere, K and N not multiples of 4"}};
  }
  std::vector<Shape> s = {
      {1, 1, 1, "a single element"},
      {3, 3, 3, "the 3 x 3 example"},
      {128, 8, 128, "one block, one tile pair"},
      {128, 24, 128, "one block, three tile pairs"},
      {130, 9, 131, "partial tiles everywhere, K and N not multiples of 4"},
      {1, 40, 260, "one decode row across three column blocks"},
      {5, 1, 9, "inner dimension 1"},
      {256, 32, 256, "four full blocks, K and N multiples of 4"},
      {129, 64, 7, "narrow N"},
      {5, 0, 9, "K = 0: an empty sum, so C is all zeros"},
  };
  if (!pmpp::is_emu()) {
    s.push_back({1024, 1024, 1024, "large square"});
    s.push_back({257, 129, 511, "large and awkward"});
    s.push_back({4, 300, 1000, "a small batch into a wide layer"});
    s.push_back({2048, 64, 2048, "short K, many blocks"});
  }
  return s;
}

inline int runTests(const pmpp::Args& a, const Stage& st) {
  pmpp::Tests t(a);
  std::vector<Candidate> cands = testCandidates(st.self);
  for (const Candidate& c : cands) {
    if (c.unusable.empty()) continue;
    std::string why = c.unusable;
    t.check("kernel usable on this device" + (c.label.empty() ? "" : " (" + c.label + ")"), [why] { return why; });
  }
  const int launches = pmpp::is_emu() ? 1 : std::max(1, st.gpuLaunches);
  for (const Shape& s : testShapes()) {
    TestProblem p(s.M, s.K, s.N, 1000u + (uint32_t)(s.M * 31 + s.K * 7 + s.N));
    for (const Candidate& c : cands) {
      if (!c.unusable.empty()) continue;
      std::string name = fmt("%d x %d times %d x %d  (%s)", s.M, s.K, s.K, s.N, s.why);
      if (!c.label.empty()) name += ", " + c.label;
      t.check(name, [&]() -> std::string {
        int bad = 0;
        std::string first;
        std::unique_ptr<Watchdog> wd;
        if (s.K == 0) wd.reset(new Watchdog(a, name, kEmptySumSeconds, kEmptySumWhy));
        for (int l = 0; l < launches; ++l) {
          p.poison();
          c.run(p.A, p.B, p.C, s.M, s.K, s.N);
          std::string e = p.check();
          if (!e.empty()) {
            if (bad == 0) first = e;
            ++bad;
          }
        }
        if (bad == 0) return "";
        if (launches == 1) return first;
        return fmt("wrong in %d of %d launches; the first: ", bad, launches) + first;
      });
    }
  }
  return t.finish();
}

// ------------------------------------------------------------------ benchmark
struct Column {
  const char* row;      // bench row name
  std::string header;   // printed
  Candidate cand;
  std::string label;    // which kernel, for the row
};

inline double deviceMemory() {
#if defined(PMPP_BACKEND_METAL)
  return pmpp::device_prop("recommendedMaxWorkingSetSize", 0);
#else
  return pmpp::device_prop("totalGlobalMem", 0);
#endif
}

inline int runBench(const pmpp::Args& a, const Stage& st) {
  std::printf("lab %s (%s, %s): %s\n", st.lab, a.variant.c_str(), pmpp::backend(), st.title);
  std::printf("vendor GEMM: %s\n", ch15VendorName().c_str());

  std::vector<Column> cols;
  cols.push_back({"tiled_5_3", fmt("5.3 T=%d", kLab53Tile), lab53Candidate(), "5.3"});
  if (st.previous.label) cols.push_back({"previous", std::string(st.previous.label), benchCandidate(st.previous), st.previous.label});
  cols.push_back({"stage", std::string(st.lab) + " " + a.variant, benchCandidate(st.self), st.lab});
  cols.push_back({"vendor", "vendor", vendorCandidate(), "vendor"});
  const size_t iStage = cols.size() - 2, iVendor = cols.size() - 1;

  // Every kernel is checked against the CPU reference once before anything is timed.
  std::printf("checking each kernel on 257 x 129 x 511 before timing:\n");
  {
    TestProblem chk(257, 129, 511, 4242);
    for (Column& c : cols) {
      if (c.cand.unusable.empty()) {
        chk.poison();
        pmpp::Timing tm = c.cand.run(chk.A, chk.B, chk.C, chk.M, chk.K, chk.N);
        std::string launchErr = pmpp::take_error();
        if (!c.cand.lastError->empty()) c.cand.unusable = *c.cand.lastError;
        else if (!launchErr.empty()) c.cand.unusable = launchErr;
        else if (tm.best() <= 0) c.cand.unusable = "it did not run";
        else c.cand.unusable = chk.check();
        if (!c.cand.unusable.empty() && c.row == std::string("vendor")) {
          // A wrong vendor result means wrong arguments in vendor_blas.cu, not a fast library.
          c.cand.unusable = "wrong result or error, not timed: " + c.cand.unusable;
        }
      }
      std::printf("  %-12s %s\n", c.header.c_str(), c.cand.unusable.empty() ? "ok" : c.cand.unusable.c_str());
    }
  }
  if (!cols[iStage].cand.unusable.empty()) {
    std::printf("this stage's kernel is not usable here; nothing to time.\n");
    return 1;
  }

  // What the device says about this stage's kernel.
  const int width = cols[iStage].cand.width;
#if defined(PMPP_BACKEND_METAL)
  {
    id<MTLComputePipelineState> p = metalPipeline(st.self, width);
    std::printf("pipeline: at most %d threads per threadgroup, SIMD-group width %d, %d bytes of static threadgroup memory\n",
                (int)p.maxTotalThreadsPerThreadgroup, (int)p.threadExecutionWidth, (int)p.staticThreadgroupMemoryLength);
    pmpp::bench_row(a, "pipeline").n("max_threads_per_threadgroup", (double)p.maxTotalThreadsPerThreadgroup)
        .n("simd_width", (double)p.threadExecutionWidth).n("static_threadgroup_bytes", (double)p.staticThreadgroupMemoryLength)
        .emit();
  }
#else
  {
    const int blocks = ch15StageBlocksPerSm(width, kThreads);
    const int sms = (int)pmpp::device_prop("multiProcessorCount", 0);
    if (blocks < 0) std::printf("occupancy: not available (see labs/ch15/shared/stage_occupancy.cu)\n");
    else std::printf("occupancy: %d block(s) of %d threads per SM, the runtime says; %d SMs\n", blocks, kThreads, sms);
    pmpp::bench_row(a, "occupancy").n("blocks_per_sm", blocks).n("threads_per_block", kThreads)
        .n("multiprocessors", sms).n("width", width).emit();
  }
#endif
  if (width > 0) std::printf("layout width: %d\n", width);
  {
    // Prediction (2) of lab 15.1: blocks per launch for the batch-1 layer.
    const int m = 1, n = 4096;
    const int b53 = ((n + kLab53Tile - 1) / kLab53Tile) * ((m + kLab53Tile - 1) / kLab53Tile);
    const int bst = ((n + kBlockTile - 1) / kBlockTile) * ((m + kBlockTile - 1) / kBlockTile);
    std::printf("blocks per launch at batch 1 of the 1024 -> 4096 layer: lab 5.3 %d, this stage %d\n", b53, bst);
  }

  struct BenchShape { const char* kind; int M, K, N; };
  const std::vector<BenchShape> shapes = {
      {"square", 1024, 1024, 1024}, {"square", 2048, 2048, 2048}, {"square", 4096, 4096, 4096},
      {"layer", 1, 1024, 4096},     {"layer", 16, 1024, 4096},    {"layer", 256, 1024, 4096},
      {"layer", 1024, 1024, 4096},  {"short_k", 4096, 128, 4096},
  };
  const double mem = deviceMemory();
  std::printf("\nGFLOP/s, best of %d runs each, kernels timed in turn\n", a.reps);
  std::printf("  %-8s %5s %5s %5s |", "shape", "M", "K", "N");
  for (const Column& c : cols) std::printf(" %14s", c.header.c_str());
  std::printf(" | %8s", "x 5.3");
  if (st.previous.label) std::printf(" %8s", (std::string("x ") + st.previous.label).c_str());
  std::printf(" %10s\n", "of vendor");

  bool stopSquares = false;
  for (const BenchShape& sh : shapes) {
    const bool square = std::string(sh.kind) == "square";
    if (square && stopSquares) {
      std::printf("  %-8s %5d: skipped (lab 5.3 took more than 2 s on the previous square, or memory ran short)\n",
                  sh.kind, sh.M);
      continue;
    }
    const double bytes = 4.0 * ((double)sh.M * sh.K + (double)sh.K * sh.N + (double)sh.M * sh.N);
    if (mem > 0 && bytes > mem / 4) {
      std::printf("  %-8s %5d %5d %5d: skipped (%.0f MB would be more than a quarter of the device memory reported)\n",
                  sh.kind, sh.M, sh.K, sh.N, bytes / 1e6);
      if (square) stopSquares = true;
      continue;
    }
    pmpp::GemmProblem p(sh.M, sh.K, sh.N, 7, false);
    {
      std::vector<float> x = pmpp::random_floats((size_t)sh.M * sh.K, 11);
      pmpp::upload(p.A, x.data(), x.size() * 4);
      x = pmpp::random_floats((size_t)sh.K * sh.N, 12);
      pmpp::upload(p.B, x.data(), x.size() * 4);
    }
    std::vector<double> best(cols.size(), std::numeric_limits<double>::infinity());
    std::vector<std::string> err(cols.size());
    for (size_t i = 0; i < cols.size(); ++i)       // one untimed warm-up run each
      if (cols[i].cand.unusable.empty()) cols[i].cand.run(p.A, p.B, p.C, sh.M, sh.K, sh.N);
    pmpp::take_error();
    for (int r = 0; r < a.reps; ++r) {
      for (size_t i = 0; i < cols.size(); ++i) {
        if (!cols[i].cand.unusable.empty() || !err[i].empty()) continue;
        const double s = cols[i].cand.run(p.A, p.B, p.C, sh.M, sh.K, sh.N).best();
        std::string e = pmpp::take_error();
        if (e.empty() && !cols[i].cand.lastError->empty()) e = *cols[i].cand.lastError;
        if (!e.empty() || s <= 0) err[i] = e.empty() ? "did not run" : e;
        else best[i] = std::min(best[i], s);
      }
    }
    const double flop = p.flop();
    auto gflops = [&](size_t i) { return std::isfinite(best[i]) && err[i].empty() ? flop / best[i] / 1e9 : NAN; };
    std::printf("  %-8s %5d %5d %5d |", sh.kind, sh.M, sh.K, sh.N);
    for (size_t i = 0; i < cols.size(); ++i) {
      const double g = gflops(i);
      if (std::isnan(g)) std::printf(" %14s", "-");
      else std::printf(" %14.1f", g);
    }
    const double gs = gflops(iStage);
    std::printf(" | %7.2fx", gs / gflops(0));
    if (st.previous.label) std::printf(" %7.2fx", gs / gflops(1));
    std::printf(" %9.0f%%\n", 100.0 * gs / gflops(iVendor));
    for (size_t i = 0; i < cols.size(); ++i) {
      if (!err[i].empty()) std::printf("    %s failed: %s\n", cols[i].header.c_str(), err[i].c_str());
      if (std::isnan(gflops(i))) continue;
      pmpp::Json j = pmpp::bench_row(a, cols[i].row);
      j.s("kernel", cols[i].label).s("shape", sh.kind).n("m", sh.M).n("k", sh.K).n("n", sh.N)
          .n("seconds", best[i]).n("gflops", gflops(i));
      if (cols[i].cand.width > 0) j.n("width", cols[i].cand.width);
      j.emit();
    }
    if (square && std::isfinite(best[0]) && best[0] > 2.0) stopSquares = true;
  }
  std::printf("\nThese are measurements of this machine, not expectations. 'x 5.3'%s is this stage's\n"
              "GFLOP/s divided by that kernel's; 'of vendor' is this stage as a share of the vendor's.\n",
              st.previous.label ? (std::string(" (and 'x ") + st.previous.label + "')").c_str() : "");
  return 0;
}

inline int stageMain(int argc, char** argv, const Stage& st) {
  return pmpp::lab_main(argc, argv, [&](const pmpp::Args& a) { return runTests(a, st); },
                        [&](const pmpp::Args& a) { return runBench(a, st); });
}

}  // namespace ch15
