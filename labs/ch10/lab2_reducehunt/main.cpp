// Lab 10.2 driver: tests a sum over any length n that ships with four defects.
// Your work goes in work/. You do not need to edit this file.
//
// Every case is launched twice on the same output buffer, with different data each
// time: the launcher must zero *out before each launch, and anything a launch picks up
// from the one before it (a stale slot, a stale register) is wrong for the second.
// On a device the pair is repeated, because a defect that depends on timing can spare
// one launch and not the next; the message says how many launches were wrong.
//
// The inputs are whole numbers from 1 to 8. Every partial sum is then a whole number
// below 2^24, which a float holds exactly, so every order of addition gives the same
// float: a difference from the CPU reference is a defect, never rounding. A tolerance
// is used anyway, as in every lab; it is far below 1, the smallest possible error.
#include "pmpp/harness.hpp"
using pmpp::Buf;

// Threads per block. kernel.cu defines its own BLOCK_DIM (the same value); kernel.metal
// is compiled with this one. Each block reduces 2 * kBlockDim elements.
static const int kBlockDim = 128;

// ---- entry points implemented in <variant>/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeReduceAny(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> in,
                     id<MTLBuffer> out, uint32_t n, uint32_t blockDim);
static const char* kBlock = "threadgroup";
static void runReduce(Buf* in, Buf* out, int n) {
  id<MTLComputePipelineState> pso =
      pmpp::metal::pipeline(PMPP_METAL_SOURCE, "reduce_any", {{"BLOCK_DIM", std::to_string(kBlockDim)}});
  if (!pso) return;
  pmpp::run([&](void* v) {
    encodeReduceAny(((pmpp::metal::Pass*)v)->encoder, pso, pmpp::metal::buffer(in), pmpp::metal::buffer(out),
                    (uint32_t)n, (uint32_t)kBlockDim);
  });
}
#else
void launchReduceAny(const float* in, float* out, int n);
static const char* kBlock = "block";
static void runReduce(Buf* in, Buf* out, int n) {
  pmpp::run([&](void*) { launchReduceAny((const float*)pmpp::raw(in), (float*)pmpp::raw(out), n); });
}
#endif

// Whole numbers 1..8, deterministic on every platform.
static std::vector<float> wholeNumbers(size_t n, uint32_t seed) {
  std::vector<float> v = pmpp::random_floats(n, seed);              // uniform in [-1, 1]
  for (float& x : v) x = (float)(1 + (int)((x + 1.0f) * 3.999f));   // 1 .. 8
  return v;
}

// One launch on fresh data into `out`. Empty string = correct.
static std::string runOnce(Buf* in, Buf* out, int n, uint32_t seed) {
  const std::vector<float> x = wholeNumbers((size_t)n, seed);
  double ref = 0.0;
  for (float v : x) ref += v;
  pmpp::upload(in, x.data(), x.size() * sizeof(float));
  runReduce(in, out, n);
  const std::string err = pmpp::take_error();
  if (!err.empty()) return "launch error: " + err;
  float got = 0.0f;
  pmpp::download(out, &got, sizeof got);
  const float want = (float)ref;
  if (pmpp::compare(&got, &want, nullptr, 1, 1e-6).ok) return "";
  char b[160];
  std::snprintf(b, sizeof b, "*out = %.9g, expected %.9g (off by %.9g)", (double)got, ref, (double)got - ref);
  return b;
}

// `pairs` times: two launches on the same output buffer.
static std::string runCase(int n, int pairs) {
  Buf* in = pmpp::alloc((size_t)n * sizeof(float));
  Buf* out = pmpp::alloc(sizeof(float));
  int wrong = 0, launches = 0;
  std::string first;
  for (int p = 0; p < pairs; ++p) {
    for (int rep = 0; rep < 2; ++rep) {
      ++launches;
      const std::string e = runOnce(in, out, n, 1000u + (uint32_t)n * 7u + (uint32_t)p * 101u + (uint32_t)rep * 17u);
      if (e.empty()) continue;
      if (wrong++ == 0) first = "launch " + std::to_string(launches) + ": " + e;
      if (e.compare(0, 13, "launch error:") == 0) { p = pairs; break; }
    }
  }
  pmpp::release(in);
  pmpp::release(out);
  if (wrong == 0) return "";
  char b[80];
  std::snprintf(b, sizeof b, "wrong in %d of %d launches. The first wrong one, ", wrong, launches);
  return std::string(b) + first;
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const int seg = 2 * kBlockDim;   // elements per block
  // The emulators run every emulated thread on the CPU (race: as OS threads under
  // ThreadSanitizer), so their lengths stay at or below 4097. Lengths 1024, 1000 and 4097
  // give at least four blocks, which the race backend runs concurrently in pairs.
  const std::vector<int> sizes = pmpp::is_emu() ? std::vector<int>{256, 512, 1024, 1, 33, 1000, 4097}
                                                : std::vector<int>{256, 512, 1024, 65536, 1, 33, 1000, 4097, 100003};
  const int pairs = pmpp::is_emu() ? 1 : 10;
  for (int n : sizes) {
    const int blocks = (n + seg - 1) / seg;
    std::string label = "n = " + std::to_string(n) + ", " + std::to_string(blocks) + " " + kBlock + (blocks == 1 ? "" : "s");
    if (n % seg == 0)
      label += " (a multiple of " + std::to_string(seg) + ")";
    else if (n % seg <= kBlockDim)
      label += " (in the last " + std::string(kBlock) + ", some threads have no element)";
    else
      label += " (in the last " + std::string(kBlock) + ", every thread has an element)";
    t.check(label, [&]() { return runCase(n, pairs); });
  }
  return t.finish();
}

static int bench(const pmpp::Args&) {
  std::printf("this lab has no benchmark.\n");
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
