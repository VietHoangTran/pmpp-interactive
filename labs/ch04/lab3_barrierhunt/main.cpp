// Lab 4.3 driver: tests a kernel that reverses each block's slice of an array in
// place, and that ships with three defects. Your work goes in work/.
// You do not need to edit this file.
//
// How a write outside the array is caught:
//   emulator   The buffer is exactly the array, so AddressSanitizer stops at the
//              first access outside it and names the line. The tests after the
//              stop are not run.
//   a device   has no such check. There the array sits inside a larger buffer
//              with sentinel elements before and after it (a pointer offset on
//              CUDA and HIP, setBuffer's offset on Metal). A stray write stays in
//              memory the driver owns, and the sentinels must come back unchanged.
//   race       (the emulator with concurrent threads under ThreadSanitizer) has
//              no AddressSanitizer, so it gets the sentinels too. What it adds is
//              a report when two threads touch the same element with no barrier
//              between them.
#include "pmpp/harness.hpp"
using pmpp::Buf;

// ---- entry points implemented in <variant>/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeReverseBlocks(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> a, NSUInteger aOffset,
                         int32_t n, uint32_t threadsPerThreadgroup);
static id<MTLComputePipelineState> pso() { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "reverse_blocks"); }
// Metal has no device-wide width: it is a property of a compiled pipeline.
static int deviceWidth() { id<MTLComputePipelineState> p = pso(); return p ? (int)p.threadExecutionWidth : 0; }
static int maxThreadsPerBlock() {
  id<MTLComputePipelineState> p = pso();
  if (!p) return 0;
  return (int)std::min<NSUInteger>(p.maxTotalThreadsPerThreadgroup, pmpp::metal::device().maxThreadsPerThreadgroup.width);
}
static const char* kWidthSource = "this pipeline's threadExecutionWidth";
static const char* kBlock = "threadgroup";
// `guard` = number of float elements in front of the array inside its buffer.
static void runReverse(Buf* a, size_t guard, int n, int threadsPerBlock) {
  id<MTLComputePipelineState> p = pso();
  if (!p) return;
  pmpp::run([&](void* vp) {
    auto* pass = (pmpp::metal::Pass*)vp;
    encodeReverseBlocks(pass->encoder, p, pmpp::metal::buffer(a), guard * sizeof(float), (int32_t)n,
                        (uint32_t)threadsPerBlock);
  });
}
#else
void launchReverseBlocks(float* a, int n, int threadsPerBlock);
static int deviceWidth() { return (int)pmpp::device_prop("warpSize", 0); }
static int maxThreadsPerBlock() {
  return (int)std::min(pmpp::device_prop("maxThreadsPerBlock", 0), pmpp::device_prop("maxThreadsDim.x", 0));
}
static const char* kWidthSource = "the device's warpSize";
static const char* kBlock = "block";
static void runReverse(Buf* a, size_t guard, int n, int threadsPerBlock) {
  pmpp::run([&](void*) { launchReverseBlocks((float*)pmpp::raw(a) + guard, n, threadsPerBlock); });
}
#endif

// Exact buffers only where AddressSanitizer is watching.
#if defined(PMPP_BACKEND_EMU)
static const bool kExactBuffers = true;
#else
static const bool kExactBuffers = false;
#endif
#if defined(PMPP_BACKEND_EMU) || defined(PMPP_BACKEND_RACE)
static const bool kEmulated = true;
#else
static const bool kEmulated = false;
#endif

// Inputs lie in [-1, 1], so a sentinel can never be mistaken for data.
static const float kSentinel = -12345.0f;

// ---- the CPU reference: reverse each slice of `bs` elements, the last one short ----
static std::vector<float> cpuReverseBlocks(const std::vector<float>& in, int bs) {
  std::vector<float> out(in.size());
  const size_t n = in.size();
  for (size_t base = 0; base < n; base += (size_t)bs) {
    const size_t valid = std::min((size_t)bs, n - base);
    for (size_t t = 0; t < valid; ++t) out[base + valid - 1 - t] = in[base + t];
  }
  return out;
}

// One launch on fresh data. Empty string = correct.
static std::string runOnce(int bs, int n, size_t guard, uint32_t seed) {
  const std::vector<float> data = pmpp::random_floats((size_t)n, seed);
  const std::vector<float> ref = cpuReverseBlocks(data, bs);
  const size_t total = (size_t)n + 2 * guard;
  std::vector<float> host(total, kSentinel);
  std::copy(data.begin(), data.end(), host.begin() + (std::ptrdiff_t)guard);

  Buf* a = pmpp::alloc(total * sizeof(float));
  pmpp::upload(a, host.data(), total * sizeof(float));
  runReverse(a, guard, n, bs);
  pmpp::download(a, host.data(), total * sizeof(float));
  pmpp::release(a);
  const std::string launchErr = pmpp::take_error();
  if (!launchErr.empty()) return "launch error: " + launchErr;

  // The kernel only moves values, so the comparison could be exact. A tolerance
  // is used anyway, as in every lab.
  const float* inside = host.data() + guard;
  const pmpp::Cmp cmp = pmpp::compare(inside, ref.data(), nullptr, (size_t)n, 1e-6);
  size_t strayBefore = 0, strayAfter = 0, nearestBefore = 0, nearestAfter = 0;
  for (size_t i = 0; i < guard; ++i)
    if (host[i] != kSentinel) { ++strayBefore; nearestBefore = guard - i; }
  for (size_t i = guard + (size_t)n; i < total; ++i)
    if (host[i] != kSentinel) { if (strayAfter == 0) nearestAfter = i - (guard + (size_t)n) + 1; ++strayAfter; }
  if (cmp.ok && strayBefore == 0 && strayAfter == 0) return "";

  char b[260];
  std::string msg;
  if (!cmp.ok) {
    const size_t f = cmp.first_bad;
    std::snprintf(b, sizeof b, "%zu of %d elements wrong; first at a[%zu] (%s %zu, thread %zu): got %.6g, expected %.6g",
                  cmp.bad, n, f, kBlock, f / (size_t)bs, f % (size_t)bs, (double)inside[f], (double)ref[f]);
    msg += b;
  }
  if (strayAfter) {
    std::snprintf(b, sizeof b, "%s%zu sentinel element%s overwritten after the end of the array, the nearest %zu past its last element",
                  msg.empty() ? "" : "; ", strayAfter, strayAfter == 1 ? "" : "s", nearestAfter);
    msg += b;
  }
  if (strayBefore) {
    std::snprintf(b, sizeof b, "%s%zu sentinel element%s overwritten before the start of the array, the nearest %zu before its first element",
                  msg.empty() ? "" : "; ", strayBefore, strayBefore == 1 ? "" : "s", nearestBefore);
    msg += b;
  }
  return msg;
}

// Several launches of one shape. On a device a missing barrier may corrupt one
// launch and spare the next, so every shape is launched `launches` times, each
// time on fresh data, and the message says how many of them were wrong. The
// emulators need one.
static std::string runShape(int bs, int n, size_t guard, int launches) {
  int wrong = 0;
  std::string first;
  for (int r = 0; r < launches; ++r) {
    const std::string e = runOnce(bs, n, guard, 4300u + (uint32_t)bs * 7u + (uint32_t)n + (uint32_t)r * 101u);
    if (!e.empty() && wrong++ == 0) first = e;
    if (!e.empty() && e.compare(0, 13, "launch error:") == 0) break;
  }
  if (wrong == 0) return "";
  char b[120];
  std::snprintf(b, sizeof b, "n = %d: wrong in %d of %d launches. ", n, wrong, launches);
  return (launches > 1 ? std::string(b) : "n = " + std::to_string(n) + ": ") + (launches > 1 ? "The first: " : "") + first;
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const int width = deviceWidth(), maxBlock = maxThreadsPerBlock();
  if (width <= 0 || maxBlock <= 0) {
    t.check("the device reports a width and a block-size limit", [&] {
      const std::string e = pmpp::take_error();
      return e.empty() ? std::string("could not read ") + kWidthSource : e;
    });
    return t.finish();
  }
  std::printf("width used for the block sizes marked \"x width\": %d (%s)\n", width, kWidthSource);

  // Four fixed sizes, two of which (48, 96) are multiples of neither width a
  // CUDA or HIP device reports today, then one, two and four times the width
  // this device reports.
  struct Size { int threads; std::string label; };
  std::vector<Size> sizes;
  auto add = [&](int threads, const std::string& label) {
    if (threads > maxBlock) return;
    for (Size& s : sizes)
      if (s.threads == threads) { if (!label.empty()) s.label += (s.label.empty() ? "" : ", ") + label; return; }
    sizes.push_back({threads, label});
  };
  for (int fixed : {32, 48, 64, 96}) add(fixed, "");
  for (int k : {1, 2, 4}) add(k * width, std::to_string(k) + " x width");

  const int launches = kEmulated ? 1 : 40;
  for (const Size& s : sizes) {
    const int bs = s.threads;
    // Sentinels on each side: a mirror index that is wrong by up to a block size
    // in either direction stays inside the buffer.
    const size_t guard = kExactBuffers ? 0 : (size_t)((2 * std::max(bs, 4 * width) + 63) / 64 * 64);
    // Whole blocks only. On a device the last length makes a grid of several
    // thousand blocks: far more than run at once, which is where the order of
    // execution within a block starts to vary from launch to launch.
    const std::vector<int> whole = kEmulated ? std::vector<int>{bs, 4 * bs} : std::vector<int>{bs, 4 * bs, 8192 * bs};
    // A short last slice. The first length leaves it one element short: a mirror
    // around the full block size then writes exactly one element past the array,
    // which AddressSanitizer cannot miss. Then one element alone, an odd count in
    // the middle of a block, and an array shorter than one block.
    const std::vector<int> ragged = kEmulated ? std::vector<int>{3 * bs - 1, 2 * bs + 1, 2 * bs + bs / 2 + 3, bs / 2 - 1}
                                              : std::vector<int>{3 * bs - 1, 2 * bs + 1, 2 * bs + bs / 2 + 3, bs / 2 - 1,
                                                                 8001 * bs + bs / 2 + 3};
    const std::string head = std::to_string(bs) + " threads per " + kBlock + (s.label.empty() ? "" : " (" + s.label + ")");
    t.check(head + ", lengths that are multiples of it", [&]() -> std::string {
      for (int n : whole) { std::string e = runShape(bs, n, guard, launches); if (!e.empty()) return e; }
      return "";
    });
    t.check(head + ", lengths that are not", [&]() -> std::string {
      for (int n : ragged) { std::string e = runShape(bs, n, guard, launches); if (!e.empty()) return e; }
      return "";
    });
  }
  return t.finish();
}

static int bench(const pmpp::Args&) {
  std::printf("this lab has no benchmark. Its experiment is in README.md.\n");
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
