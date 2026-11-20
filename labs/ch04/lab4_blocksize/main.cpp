// Lab 4.4 driver: tests your gather kernel at several block sizes, tests your
// occupancy model on limits nobody has to own (CUDA, HIP and the emulator), then
// sweeps the block size on your device and sets the timings beside the model.
// You do not need to edit this file; your work goes in work/.
//
// Block sizes are built from the width the device reports, never from a literal:
// warpSize on CUDA and HIP, the pipeline's threadExecutionWidth on Metal.
//
// How a write past `out` is caught: on the emulator every buffer has exactly its
// size and AddressSanitizer stops at the first access outside it. On a device
// `out` is followed by canary elements that must come back untouched, and `idx`
// by valid indices, so a thread past n reads something harmless and is caught
// by what it stores.
#include <cstdlib>

#include "pmpp/harness.hpp"
using pmpp::Buf;

static_assert(sizeof(int) == 4, "the kernels read idx as 32-bit integers");

static const float kUnwritten = 12345.0f;
static const float kCanary = -54321.0f;

// Deterministic indices in [0, tableSize), identical on every platform.
static std::vector<int32_t> random_indices(size_t count, uint32_t tableSize, uint32_t seed) {
  std::vector<int32_t> v(count);
  uint32_t x = seed * 2654435761u + 1u;
  for (size_t i = 0; i < count; ++i) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;                 // xorshift32
    v[i] = (int32_t)(((uint64_t)x * tableSize) >> 32);
  }
  return v;
}

// ---------- one gather on the device, with its CPU reference ----------
//   out[i] = sum over j < k of table[idx[i*k + j]]      for i < n
struct Gather {
  int n, k;
  size_t tableSize, slack;
  std::vector<float> table;
  std::vector<int32_t> idx;
  Buf *T = nullptr, *I = nullptr, *O = nullptr;

  Gather(int n_, int k_, size_t tableSize_, uint32_t seed) : n(n_), k(k_), tableSize(tableSize_) {
    slack = pmpp::is_emu() ? 0 : 4096;
    table = pmpp::random_floats(tableSize, seed);
    idx = random_indices((size_t)n * k, (uint32_t)tableSize, seed + 1);
    T = pmpp::alloc(tableSize * 4);
    pmpp::upload(T, table.data(), tableSize * 4);
    std::vector<int32_t> host(idx);
    host.resize(((size_t)n + slack) * k, 0);
    I = pmpp::alloc(host.size() * 4);
    pmpp::upload(I, host.data(), host.size() * 4);
    O = pmpp::alloc(((size_t)n + slack) * 4);
    prefill();
  }
  Gather(const Gather&) = delete;
  ~Gather() { pmpp::release(T); pmpp::release(I); pmpp::release(O); }

  // out gets a value no sum equals, so an output nobody writes is detected.
  void prefill() {
    std::vector<float> host((size_t)n, kUnwritten);
    host.resize(host.size() + slack, kCanary);
    pmpp::upload(O, host.data(), host.size() * 4);
  }
  double lookups() const { return (double)n * k; }

  // The reference: the same sum on the CPU, in double. Empty string = every
  // output examined matches it within a tolerance and nothing outside `out`
  // was written. `every` > 1 examines every `every`-th output and the last one
  // (the benchmark's problem is large). `threads` only words the message.
  std::string check(int threads, size_t every = 1) {
    std::vector<float> got((size_t)n + slack);
    pmpp::download(O, got.data(), got.size() * 4);
    size_t examined = 0, bad = 0, first = 0, unwritten = 0, past = 0;
    double maxErr = 0;
    auto one = [&](size_t i) {
      double sum = 0, mag = 0;
      for (int j = 0; j < k; ++j) { const double v = table[(size_t)idx[i * k + j]]; sum += v; mag += std::fabs(v); }
      const double e = std::fabs((double)got[i] - sum);
      if (!(e <= 1e-4 * (1.0 + mag))) { if (bad == 0) first = i; ++bad; }   // also catches NaN
      if (e > maxErr || std::isnan(e)) maxErr = e;
      unwritten += (got[i] == kUnwritten);
      ++examined;
    };
    for (size_t i = 0; i < (size_t)n; i += every) one(i);
    if (every > 1 && (size_t)(n - 1) % every != 0) one((size_t)n - 1);
    for (size_t i = (size_t)n; i < got.size(); ++i) past += !(got[i] == kCanary);

    std::string msg;
    auto add = [&](const std::string& s) { msg += (msg.empty() ? "" : "; ") + s; };
    if (bad) {
      char b[240];
      std::snprintf(b, sizeof b, "%zu of %zu outputs examined are wrong; first at out[%zu] (thread %zu of block %zu); max abs error %.4g",
                    bad, examined, first, first % (size_t)threads, first / (size_t)threads, maxErr);
      add(b);
    }
    if (unwritten) add(std::to_string(unwritten) + " of them were never written");
    if (past) add(std::to_string(past) + (past == 1 ? " value was" : " values were") + " stored past the end of out");
    return msg;
  }
};

struct Case { int n, k, threads; const char* why; };
// The shapes every backend is tested with, from the device's width and its
// largest block.
static std::vector<Case> testCases(int width, int most) {
  const std::vector<Case> all = {
      {1, 1, width, "one output, one lookup"},
      {31 * width + 8, 4, width, "one width each; the last one is partial"},
      {21 * width, 5, 3 * width, "three widths each; n is a whole number of them"},
      {1000, 7, width + width / 2, "one and a half widths each"},
      {300, 16, 7, "smaller than the width"},
      {513, 1, 2 * width, "one lookup per output"},
      {4 * most + 1, 3, most, "the largest the device takes, one output past four of them"},
  };
  std::vector<Case> fit;
  for (const Case& c : all)
    if (c.threads >= 1 && c.threads <= most) fit.push_back(c);
  return fit;
}

// The benchmark's sweep: these multiples of the width, up to the largest block.
static std::vector<int> sweepSizes(int width, int most) {
  std::vector<int> sizes;
  for (int m : {1, 2, 3, 4, 6, 8, 12, 16, 24, 32})
    if (m * width <= most) sizes.push_back(m * width);
  return sizes;
}

// ---- entry points implemented in <variant>/ and provided/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeGatherSum(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> table, id<MTLBuffer> idx,
                     id<MTLBuffer> out, uint32_t n, uint32_t k, uint32_t threadsPerThreadgroup);

static id<MTLComputePipelineState> gatherPipeline() { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "gather_sum"); }
// On Metal the width and the thread limit belong to the compiled kernel.
static int deviceWidth() { auto p = gatherPipeline(); return p ? (int)p.threadExecutionWidth : 0; }
static int deviceMaxBlock() {
  auto p = gatherPipeline();
  if (!p) return 0;
  return (int)std::min<NSUInteger>(p.maxTotalThreadsPerThreadgroup, pmpp::metal::device().maxThreadsPerThreadgroup.width);
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
static pmpp::Timing runGather(Gather& g, int threads, int repeat) {
  auto p = gatherPipeline();
  if (!p) return {0, 0};
  return pmpp::run([&](void* v) {
    auto* pass = (pmpp::metal::Pass*)v;
    for (int r = 0; r < repeat; ++r)
      encodeGatherSum(pass->encoder, p, pmpp::metal::buffer(g.T), pmpp::metal::buffer(g.I), pmpp::metal::buffer(g.O),
                      (uint32_t)g.n, (uint32_t)g.k, (uint32_t)threads);
  });
}
static const char* kBlock = "threadgroup";
static const char* kWidthSource = "threadExecutionWidth of the gather_sum pipeline";
#else
#include "sm_limits.h"
void launchGatherSum(const float* table, const int* idx, float* out, int n, int k, int threadsPerBlock);
int modelBlocksPerSm(const SmLimits& L, int threadsPerBlock, int regsPerThread);
double modelOccupancy(const SmLimits& L, int threadsPerBlock, int regsPerThread);
int vendorBlocksPerSm(int threadsPerBlock);   // provided/occupancy_api.cu; -1 = not available

static int deviceWidth() { return (int)pmpp::device_prop("warpSize", 0); }
static int deviceMaxBlock() { return (int)pmpp::device_prop("maxThreadsPerBlock", 0); }
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
static pmpp::Timing runGather(Gather& g, int threads, int repeat) {
  return pmpp::run([&](void*) {
    for (int r = 0; r < repeat; ++r)
      launchGatherSum((const float*)pmpp::raw(g.T), (const int*)pmpp::raw(g.I), (float*)pmpp::raw(g.O), g.n, g.k, threads);
  });
}
static const char* kBlock = "block";
static const char* kWidthSource = "warpSize";

// ---------- the model, on limits nobody has to own ----------
// {warpSize, maxThreadsPerBlock, block slots, thread slots, registers}; 0 = not reported.
static const SmLimits kBook = {32, 1024, 32, 2048, 65536};      // the book's example device, p. 85 and p. 86
static const SmLimits kEx6 = {32, 1024, 4, 1536, 0};            // exercise 6
static const SmLimits kEx7 = {32, 1024, 64, 2048, 0};           // exercise 7
static const SmLimits kSmaller = {32, 1024, 16, 1536, 65536};   // 1536 thread slots, 16 block slots
static const SmLimits kWide = {64, 1024, 32, 2048, 0};          // a width of 64

struct ModelCase { SmLimits L; int threads, regs; int blocks; double occupancy; const char* note; };

static std::string pct(double f) {
  if (f < 0) return "unknown (-1)";
  char b[32];
  std::snprintf(b, sizeof b, "%.1f%%", 100.0 * f);
  return b;
}

// Runs your two functions on each case and names the cases that differ.
static std::string runModel(const std::vector<ModelCase>& cases) {
  std::string msg;
  for (const ModelCase& c : cases) {
    const int blocks = modelBlocksPerSm(c.L, c.threads, c.regs);
    const double occ = modelOccupancy(c.L, c.threads, c.regs);
    const bool blocksOk = (blocks == c.blocks);
    const bool occOk = c.occupancy < 0 ? occ < 0 : std::fabs(occ - c.occupancy) < 1e-9;
    if (blocksOk && occOk) continue;
    std::string s = std::to_string(c.threads) + " threads per block";
    if (c.regs > 0) s += ", " + std::to_string(c.regs) + " registers per thread";
    if (c.note[0]) s += std::string(" (") + c.note + ")";
    s += ":";
    if (!blocksOk) s += " blocks per SM yours " + std::to_string(blocks) + ", expected " + std::to_string(c.blocks) + ";";
    if (!occOk) s += " occupancy yours " + pct(occ) + ", expected " + pct(c.occupancy) + ";";
    s.pop_back();
    msg += (msg.empty() ? "" : " | ") + s;
  }
  return msg;
}

static void modelTests(pmpp::Tests& t) {
  t.check("model: the book's seven block sizes (pp. 85-86): 32 block slots, 2048 thread slots, width 32", [] {
    return runModel({
        {kBook, 1024, 0, 2, 1.0, ""},
        {kBook, 512, 0, 4, 1.0, ""},
        {kBook, 256, 0, 8, 1.0, ""},
        {kBook, 128, 0, 16, 1.0, ""},
        {kBook, 64, 0, 32, 1.0, ""},
        {kBook, 32, 0, 32, 0.5, "the block slots run out first"},
        {kBook, 768, 0, 2, 0.75, "a third block does not fit"},
    });
  });
  t.check("model: registers (p. 86): the same device with 65,536 registers per SM", [] {
    return runModel({
        {kBook, 512, 31, 4, 1.0, ""},
        {kBook, 512, 32, 4, 1.0, "exactly the registers there are"},
        {kBook, 512, 33, 3, 0.75, "the cliff"},
        {kBook, 1024, 64, 1, 0.5, ""},
        {kBook, 256, 64, 4, 0.5, ""},
        {kBook, 64, 64, 16, 0.5, ""},
    });
  });
  t.check("model: exercise 6 (1536 thread slots, 4 block slots)", [] {
    return runModel({
        {kEx6, 128, 0, 4, 1.0 / 3.0, ""},
        {kEx6, 256, 0, 4, 2.0 / 3.0, ""},
        {kEx6, 512, 0, 3, 1.0, ""},
        {kEx6, 1024, 0, 1, 2.0 / 3.0, ""},
    });
  });
  t.check("model: exercise 7 (2048 thread slots, 64 block slots)", [] {
    return runModel({
        {kEx7, 128, 0, 16, 1.0, ""},
        {kEx7, 64, 0, 32, 1.0, ""},
        {kEx7, 32, 0, 64, 1.0, "here 32-thread blocks do fill the SM"},
    });
  });
  t.check("model: exercise 8 (2048 thread slots, 32 block slots, 65,536 registers)", [] {
    return runModel({
        {kBook, 128, 30, 16, 1.0, "a"},
        {kBook, 32, 29, 32, 0.5, "b"},
        {kBook, 256, 34, 7, 0.875, "c"},
    });
  });
  t.check("model: a device with 1536 thread slots and 16 block slots", [] {
    return runModel({
        {kSmaller, 768, 0, 2, 1.0, "75% on the book's device"},
        {kSmaller, 1024, 0, 1, 2.0 / 3.0, "100% on the book's device"},
        {kSmaller, 96, 0, 16, 1.0, "the smallest block that fills it"},
        {kSmaller, 64, 0, 16, 2.0 / 3.0, "the block slots run out first"},
    });
  });
  t.check("model: blocks that are not whole warps, and a width that is not 32", [] {
    return runModel({
        {kBook, 48, 0, 32, 1.0, "2 warps each"},
        {kEx7, 48, 0, 32, 1.0, "2 warps each: 32 blocks take all 64 warp slots"},
        {kBook, 33, 0, 32, 1.0, "2 warps each"},
        {kWide, 96, 0, 16, 1.0, "width 64: 2 warps each"},
        {kWide, 64, 0, 32, 1.0, "width 64: 1 warp each"},
        {kWide, 32, 0, 32, 1.0, "width 64: half a warp still takes a warp slot"},
    });
  });
  t.check("model: limits that are not reported (0), and blocks that cannot run", [] {
    return runModel({
        {{32, 0, 0, 0, 0}, 256, 0, -1, -1.0, "nothing reported"},
        {{32, 0, 0, 2048, 0}, 32, 0, 64, 1.0, "thread slots only"},
        {{32, 0, 32, 0, 0}, 256, 0, 32, -1.0, "block slots only"},
        {{32, 0, 0, 0, 65536}, 256, 32, 8, -1.0, "registers only"},
        {{32, 0, 0, 0, 65536}, 256, 0, -1, -1.0, "registers only, and the kernel's count not known"},
        {kBook, 512, 0, 4, 1.0, "the kernel's register count not known"},
        {kEx7, 512, 64, 4, 1.0, "the SM's registers not reported"},
        {kBook, 2048, 0, 0, 0.0, "larger than maxThreadsPerBlock"},
        {{32, 0, 32, 2048, 0}, 2048, 0, 1, 1.0, "maxThreadsPerBlock not reported"},
        {{32, 0, 32, 2048, 0}, 4096, 0, 0, 0.0, "more threads than the SM has slots"},
    });
  });
}
#endif

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const int width = deviceWidth(), most = deviceMaxBlock();
  if (width < 1 || most < 1) {
#if defined(PMPP_BACKEND_METAL)
    t.check("kernel.metal builds", [] { return std::string("see the compile error above"); });
#else
    t.check("the device reports warpSize and maxThreadsPerBlock", [] { return std::string("one of them is 0"); });
#endif
    return t.finish();
  }
  const size_t tableSize = 4099;   // small: the tests are about indices, not about waiting
  for (const Case& c : testCases(width, most)) {
    char name[240];
    std::snprintf(name, sizeof name, "gather: n = %d, k = %d, %d threads per %s  (%s)", c.n, c.k, c.threads, kBlock, c.why);
    t.check(name, [&]() -> std::string {
      Gather g(c.n, c.k, tableSize, 4400 + c.n + 31 * c.k);
      runGather(g, c.threads, 1);
      return g.check(c.threads);
    });
  }
#if !defined(PMPP_BACKEND_METAL)
  modelTests(t);
#endif
  return t.finish();
}

static int bench(const pmpp::Args& a) {
  const int width = deviceWidth(), most = deviceMaxBlock();
  if (width < 1 || most < 1) { std::printf("no width or no block limit to size the sweep from\n"); return 1; }
  const std::vector<int> sizes = sweepSizes(width, most);
  if (sizes.empty()) { std::printf("the device's largest %s (%d) is smaller than its width (%d)\n", kBlock, most, width); return 1; }

  // ---- the problem, sized from what the device reports ----
  // The table is meant to be far larger than any cache, so that a lookup waits
  // on memory. Its size follows the device's memory, between 32 MB and 256 MB.
  const double mem = deviceMemory();
  const size_t tableBytes = (size_t)std::min(std::max(mem / 32.0, 32e6), 256e6);
  const size_t tableSize = tableBytes / 4;
  const int n = 1000000, k = 16;
  Gather g(n, k, tableSize, 44);
  std::printf("gather: %d outputs x %d lookups each, into a table of %zu floats (%.0f MB); best of %d runs\n",
              n, k, tableSize, tableSize * 4 / 1e6, a.reps);
  std::printf("width: %d (%s); largest %s: %d threads\n", width, kWidthSource, kBlock, most);

  // ---- every size must compute the gather before any is timed ----
  const size_t every = 251;
  for (int threads : sizes) {
    g.prefill();
    runGather(g, threads, 1);
    std::string err = pmpp::take_error();
    if (err.empty()) err = g.check(threads, every);
    if (!err.empty()) { std::printf("  %d threads per %s: your kernel does not compute the gather: %s\n", threads, kBlock, err.c_str()); return 1; }
  }

  // ---- timing ----
  // Launches are repeated between two synchronisations until a run is long
  // enough to time; the count is fixed with a middle size and used for all.
  const int probe = sizes[std::min<size_t>(3, sizes.size() - 1)];
  int repeat = 1;
  while (repeat < 256 && pmpp::time_it(2, [&] { return runGather(g, probe, repeat); }).min_s < 0.05) repeat *= 2;
  // The sizes are timed in rounds, so that whatever else the machine is doing
  // falls on all of them. The best run of each is the measurement; the median
  // is kept too, because the gap between the two says how much of a difference
  // between two sizes is the machine and not the block size.
  std::vector<std::vector<double>> runs(sizes.size());
  for (size_t s = 0; s < sizes.size(); ++s) runGather(g, sizes[s], 1);   // untimed warm-up
  for (int r = 0; r < a.reps; ++r)
    for (size_t s = 0; s < sizes.size(); ++s) runs[s].push_back(runGather(g, sizes[s], repeat).best() / repeat);
  std::string err = pmpp::take_error();
  if (!err.empty()) { std::printf("the sweep failed: %s\n", err.c_str()); return 1; }
  std::vector<double> best(sizes.size()), median(sizes.size());
  double spread = 1.0;   // the largest median / best over the sizes
  for (size_t s = 0; s < sizes.size(); ++s) {
    std::sort(runs[s].begin(), runs[s].end());
    best[s] = runs[s].front();
    median[s] = runs[s][runs[s].size() / 2];
    spread = std::max(spread, median[s] / best[s]);
  }

  pmpp::bench_row(a, "problem").n("outputs", n).n("lookups_per_output", k).n("table_floats", (double)tableSize)
      .n("table_bytes", (double)tableSize * 4).n("width", width).n("max_threads_per_block", most)
      .n("launches_per_run", repeat).emit();

#if defined(PMPP_BACKEND_METAL)
  std::printf("  %-20s %8s %11s %13s %14s\n", "threads/threadgroup", "x width", "best (ms)", "median (ms)", "M lookups/s");
  for (size_t s = 0; s < sizes.size(); ++s) {
    std::printf("  %-20d %8d %11.4f %13.4f %14.1f\n", sizes[s], sizes[s] / width, best[s] * 1e3, median[s] * 1e3,
                g.lookups() / best[s] / 1e6);
    pmpp::bench_row(a, "sweep").n("threads_per_block", sizes[s]).n("width", width).n("multiple", sizes[s] / width)
        .n("seconds", best[s]).n("median_seconds", median[s]).n("lookups_per_s", g.lookups() / best[s]).emit();
  }
  std::printf("  (Metal reports no per-core limits and has no occupancy function, so there is no occupancy\n"
              "   figure to set beside these timings: notes/today/ch04-facts.md, C5.)\n");
#else
  // ---- the limits your model works from, as the device reports them ----
  SmLimits L;
  L.warpSize = width;
  L.maxThreadsPerBlock = most;
  L.maxBlocksPerMultiProcessor = (int)pmpp::device_prop("maxBlocksPerMultiProcessor", 0);
  L.maxThreadsPerMultiProcessor = (int)pmpp::device_prop("maxThreadsPerMultiProcessor", 0);
  L.regsPerMultiprocessor = (int)pmpp::device_prop("regsPerMultiprocessor", 0);
  // No device field holds a kernel's register count. The compiler prints it
  // (nvcc --resource-usage, hipcc --resource-usage); hand it over in the environment.
  const char* env = std::getenv("PMPP_REGS_PER_THREAD");
  const int regs = env ? std::max(0, std::atoi(env)) : 0;
  std::printf("one SM, as the device reports it: %d block slots, %d thread slots, %d registers (0 = not reported)\n",
              L.maxBlocksPerMultiProcessor, L.maxThreadsPerMultiProcessor, L.regsPerMultiprocessor);
  if (std::string(pmpp::backend()) == "hip")
    std::printf("  (on AMD the block-slot field is computed from the thread slots and is not a limit of its own,\n"
                "   and the register field counts vector registers only: notes/today/ch04-facts.md, B9 and B10)\n");
  if (regs > 0) std::printf("registers per thread given to your model: %d (PMPP_REGS_PER_THREAD)\n", regs);
  else std::printf("registers per thread given to your model: none, so it counts slots only (set PMPP_REGS_PER_THREAD=<n> to include them)\n");
  pmpp::bench_row(a, "limits").n("warpSize", L.warpSize).n("maxThreadsPerBlock", L.maxThreadsPerBlock)
      .n("maxBlocksPerMultiProcessor", L.maxBlocksPerMultiProcessor)
      .n("maxThreadsPerMultiProcessor", L.maxThreadsPerMultiProcessor)
      .n("regsPerMultiprocessor", L.regsPerMultiprocessor).n("regs_per_thread_given", regs).emit();

  std::printf("  %-14s %8s %11s %13s %14s   %-22s %-22s\n", "threads/block", "x width", "best (ms)", "median (ms)",
              "M lookups/s", "your model", "vendor's function");
  std::printf("  %-14s %8s %11s %13s %14s   %-22s %-22s\n", "", "", "", "", "", "blocks/SM  occupancy", "blocks/SM  occupancy");
  for (size_t s = 0; s < sizes.size(); ++s) {
    const int threads = sizes[s];
    const int mBlocks = modelBlocksPerSm(L, threads, regs);
    const double mOcc = modelOccupancy(L, threads, regs);
    const int vBlocks = vendorBlocksPerSm(threads);
    // The vendor's function returns blocks. As occupancy: blocks x whole warps
    // of thread slots per block, over the thread slots of the SM.
    const int warps = (threads + width - 1) / width;
    const double vOcc = (vBlocks >= 0 && L.maxThreadsPerMultiProcessor > 0)
                            ? (double)vBlocks * warps * width / L.maxThreadsPerMultiProcessor : -1.0;
    auto cell = [](int blocks, double occ) {
      char b[48], o[16];
      if (occ >= 0) std::snprintf(o, sizeof o, "%.1f%%", 100.0 * occ); else std::snprintf(o, sizeof o, "n/a");
      if (blocks >= 0) std::snprintf(b, sizeof b, "%9d  %9s", blocks, o); else std::snprintf(b, sizeof b, "%9s  %9s", "n/a", o);
      return std::string(b);
    };
    std::printf("  %-14d %8d %11.4f %13.4f %14.1f   %-22s %-22s\n", threads, threads / width, best[s] * 1e3,
                median[s] * 1e3, g.lookups() / best[s] / 1e6, cell(mBlocks, mOcc).c_str(), cell(vBlocks, vOcc).c_str());
    pmpp::Json j = pmpp::bench_row(a, "sweep");
    j.n("threads_per_block", threads).n("width", width).n("multiple", threads / width)
        .n("seconds", best[s]).n("median_seconds", median[s]).n("lookups_per_s", g.lookups() / best[s]);
    if (mBlocks >= 0) j.n("model_blocks_per_sm", mBlocks);
    if (mOcc >= 0) j.n("model_occupancy", mOcc);
    if (vBlocks >= 0) j.n("vendor_blocks_per_sm", vBlocks);
    if (vOcc >= 0) j.n("vendor_occupancy", vOcc);
    j.emit();
  }
  std::printf("  (n/a in your model: the device does not report the limit it needs. n/a under the vendor's function:\n"
              "   the lab was configured without it, or the runtime returned an error.)\n");
#endif

  size_t fast = 0, slow = 0;
  for (size_t s = 1; s < sizes.size(); ++s) {
    if (best[s] < best[fast]) fast = s;
    if (best[s] > best[slow]) slow = s;
  }
  std::printf("fastest: %d threads per %s (%.4f ms). slowest: %d threads per %s (%.4f ms), %.2f times the fastest\n",
              sizes[fast], kBlock, best[fast] * 1e3, sizes[slow], kBlock, best[slow] * 1e3, best[slow] / best[fast]);
  std::printf("for scale: the median run of one size was up to %.2f times its own best run in this sweep.\n"
              "A gap between two sizes that is no larger than that is not a result.\n", spread);
  pmpp::bench_row(a, "summary").n("fastest_threads_per_block", sizes[fast]).n("fastest_seconds", best[fast])
      .n("slowest_threads_per_block", sizes[slow]).n("slowest_seconds", best[slow])
      .n("slowest_over_fastest", best[slow] / best[fast]).n("largest_median_over_best", spread).emit();
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
