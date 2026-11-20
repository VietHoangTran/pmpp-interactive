// Lab 3.5 driver: tests a windowed mean along the sequence axis of a
// [batch, seq, hidden] tensor that ships with four defects.
// You do not need to edit this file; your work goes in work/.
//
// A tensor here is row-major with the batch first: element [b][t][h] of a
// [batch, seq, hidden] tensor. The kernel maps hidden to x, seq to y and batch
// to z, in blocks of 8 x 8 x 4 threads.
//
// How an access outside the tensor is caught:
//   emulator   The buffers are exactly as large as the tensor, so
//              AddressSanitizer stops at the first access outside them and
//              names the line.
//   a device   has no such check. There the tensor sits inside a larger buffer
//              with guard elements before and after it (a pointer offset on CUDA
//              and HIP, setBuffer's offset on Metal). A stray access stays in
//              memory the driver owns, and the output's guard elements must come
//              back unchanged.
//   race       (the emulator with concurrent threads under ThreadSanitizer) has
//              no AddressSanitizer, so it gets the guard elements too. What it
//              adds is a report when two threads touch the same element.
#include "pmpp/harness.hpp"
using pmpp::Buf;

// The block shape the lab's launch code uses. Here only to check it against
// what the device reports.
static const int kBlockX = 8, kBlockY = 8, kBlockZ = 4;

// ---- entry points implemented in <variant>/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeSeqMean(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> in, NSUInteger inOffset,
                   id<MTLBuffer> out, NSUInteger outOffset, int32_t batch, int32_t seq, int32_t hidden, int32_t r);
static id<MTLComputePipelineState> pso() { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "seq_mean"); }
// Empty if this device accepts 8 x 8 x 4 threadgroups for this kernel, else why not.
static std::string blockProblem() {
  id<MTLComputePipelineState> p = pso();
  if (!p) return "";   // the run reports the compile error
  const MTLSize m = pmpp::metal::device().maxThreadsPerThreadgroup;
  char b[200];
  if ((NSUInteger)kBlockX > m.width || (NSUInteger)kBlockY > m.height || (NSUInteger)kBlockZ > m.depth) {
    std::snprintf(b, sizeof b, "the device allows threadgroups of at most %lu by %lu by %lu",
                  (unsigned long)m.width, (unsigned long)m.height, (unsigned long)m.depth);
    return b;
  }
  if ((NSUInteger)(kBlockX * kBlockY * kBlockZ) > p.maxTotalThreadsPerThreadgroup) {
    std::snprintf(b, sizeof b, "this pipeline allows at most %lu threads per threadgroup",
                  (unsigned long)p.maxTotalThreadsPerThreadgroup);
    return b;
  }
  return "";
}
// `guard` = number of float elements in front of each tensor inside its buffer.
static void runSeqMean(Buf* in, Buf* out, size_t guard, int batch, int seq, int hidden, int r) {
  id<MTLComputePipelineState> p = pso();
  if (!p) return;
  pmpp::run([&](void* vp) {
    auto* pass = (pmpp::metal::Pass*)vp;
    encodeSeqMean(pass->encoder, p, pmpp::metal::buffer(in), guard * sizeof(float), pmpp::metal::buffer(out),
                  guard * sizeof(float), batch, seq, hidden, r);
  });
}
#else
void launchSeqMean(const float* in, float* out, int batch, int seq, int hidden, int r);
// Empty if this device accepts 8 x 8 x 4 blocks, else why not. A property the
// device does not report is not checked.
static std::string blockProblem() {
  const double maxThreads = pmpp::device_prop("maxThreadsPerBlock", -1);
  const double maxX = pmpp::device_prop("maxThreadsDim.x", -1), maxY = pmpp::device_prop("maxThreadsDim.y", -1),
               maxZ = pmpp::device_prop("maxThreadsDim.z", -1);
  char b[200];
  if ((maxX >= 0 && kBlockX > maxX) || (maxY >= 0 && kBlockY > maxY) || (maxZ >= 0 && kBlockZ > maxZ)) {
    std::snprintf(b, sizeof b, "the device allows blocks of at most %.0f by %.0f by %.0f", maxX, maxY, maxZ);
    return b;
  }
  if (maxThreads >= 0 && kBlockX * kBlockY * kBlockZ > maxThreads) {
    std::snprintf(b, sizeof b, "the device allows at most %.0f threads per block", maxThreads);
    return b;
  }
  return "";
}
// `guard` = number of float elements in front of each tensor inside its buffer.
static void runSeqMean(Buf* in, Buf* out, size_t guard, int batch, int seq, int hidden, int r) {
  pmpp::run([&](void*) {
    launchSeqMean((const float*)pmpp::raw(in) + guard, (float*)pmpp::raw(out) + guard, batch, seq, hidden, r);
  });
}
#endif

// Exact buffers only where AddressSanitizer is watching.
#if defined(PMPP_BACKEND_EMU)
static const bool kExactBuffers = true;
#else
static const bool kExactBuffers = false;
#endif

// The output is filled with this before the launch. Inputs lie in [-1, 1] and so
// does every correct mean, so an element that still holds it was never written.
static const float kUnwritten = -12345.0f;
// What the guard elements around the input hold on a device. A kernel that reads
// outside the tensor there gets a value no mean of real inputs can hide.
static const float kInputGuard = 1.0e6f;

struct Case {
  const char* name;
  int batch, seq, hidden, r;
  const char* why;
};

// ---- the CPU reference ----
// out[b][t][h] = mean of in[b][s][h] over the s in [t-r, t+r] that exist.
static void cpuSeqMean(const std::vector<float>& in, const Case& c, std::vector<float>& out) {
  out.assign(in.size(), 0.f);
  for (int b = 0; b < c.batch; ++b)
    for (int t = 0; t < c.seq; ++t) {
      const int lo = std::max(0, t - c.r), hi = std::min(c.seq - 1, t + c.r);
      for (int h = 0; h < c.hidden; ++h) {
        double sum = 0.0;
        for (int s = lo; s <= hi; ++s) sum += in[((size_t)b * c.seq + s) * c.hidden + h];
        out[((size_t)b * c.seq + t) * c.hidden + h] = (float)(sum / (hi - lo + 1));
      }
    }
}

// Guard elements on each side of a tensor on a real device. Sized for the worst
// the four planted defects can do, alone or together: reads up to r rows before
// the start, reads and writes in up to three surplus planes after the end, and a
// write with a wrong row stride up to seq * seq + hidden elements further on.
// Rounded up to a multiple of 64 elements (256 bytes).
static size_t guardElements(const Case& c) {
  if (kExactBuffers) return 0;
  const size_t plane = (size_t)c.seq * c.hidden;
  const size_t g = 4 * plane + (size_t)c.seq * c.seq + ((size_t)c.r + 1) * c.hidden + 64;
  return (g + 63) / 64 * 64;
}

// Runs the kernel on one case. Empty string = correct.
static std::string runAndCheck(const Case& c) {
  const size_t n = (size_t)c.batch * c.seq * c.hidden;
  const size_t g = guardElements(c), total = n + 2 * g;

  // Noise in [-1, 1]: no two rows, columns or planes look alike, so an element
  // read from or written to the wrong place shows.
  const std::vector<float> data =
      pmpp::random_floats(n, 3500u + (uint32_t)c.batch * 131u + (uint32_t)c.seq * 17u + (uint32_t)c.hidden + (uint32_t)c.r * 7u);
  std::vector<float> ref;
  cpuSeqMean(data, c, ref);

  std::vector<float> hostIn(total, kInputGuard), got(total, kUnwritten);
  std::copy(data.begin(), data.end(), hostIn.begin() + (std::ptrdiff_t)g);
  Buf* in = pmpp::alloc(total * sizeof(float));
  Buf* out = pmpp::alloc(total * sizeof(float));
  pmpp::upload(in, hostIn.data(), total * sizeof(float));
  pmpp::upload(out, got.data(), total * sizeof(float));
  runSeqMean(in, out, g, c.batch, c.seq, c.hidden, c.r);
  pmpp::download(out, got.data(), total * sizeof(float));
  pmpp::release(in);
  pmpp::release(out);
  const std::string launchErr = pmpp::take_error();
  if (!launchErr.empty()) return "launch error: " + launchErr;

  // Inside the tensor: a relative tolerance against the reference (Metal does not
  // default to strict IEEE arithmetic, and the kernel sums in float).
  const float* inside = got.data() + g;
  const pmpp::Cmp cmp = pmpp::compare(inside, ref.data(), nullptr, n, 1e-4);
  size_t unwritten = 0;
  for (size_t i = 0; i < n; ++i) unwritten += inside[i] == kUnwritten;

  // Outside it (devices only): every guard element of the output must be untouched.
  size_t strayBefore = 0, strayAfter = 0, nearestBefore = 0, nearestAfter = 0;
  for (size_t i = 0; i < g; ++i)
    if (got[i] != kUnwritten) { ++strayBefore; nearestBefore = g - i; }
  for (size_t i = g + n; i < total; ++i)
    if (got[i] != kUnwritten) { if (strayAfter == 0) nearestAfter = i - (g + n); ++strayAfter; }
  if (cmp.ok && strayBefore == 0 && strayAfter == 0) return "";

  char b[240];
  std::string msg;
  if (!cmp.ok) {
    const size_t f = cmp.first_bad, plane = (size_t)c.seq * c.hidden;
    std::snprintf(b, sizeof b, "%zu of %zu elements wrong; first at [b][t][h] = [%zu][%zu][%zu]: got %.6g, expected %.6g",
                  cmp.bad, n, f / plane, f % plane / (size_t)c.hidden, f % (size_t)c.hidden, (double)inside[f], (double)ref[f]);
    msg += b;
    if (unwritten) {
      std::snprintf(b, sizeof b, "; %zu elements were never written (they still hold the pre-fill value %g)", unwritten,
                    (double)kUnwritten);
      msg += b;
    }
  }
  if (strayAfter) {
    std::snprintf(b, sizeof b, "%s%zu elements written after the end of the tensor, the nearest %zu past its last element",
                  msg.empty() ? "" : "; ", strayAfter, nearestAfter + 1);
    msg += b;
  }
  if (strayBefore) {
    std::snprintf(b, sizeof b, "%s%zu elements written before the start of the tensor, the nearest %zu before its first element",
                  msg.empty() ? "" : "; ", strayBefore, nearestBefore);
    msg += b;
  }
  return msg;
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::string why = blockProblem();
  if (!why.empty()) {
    t.check("8 by 8 by 4 threads per block", [&] { return "this lab cannot run here: " + why; });
    return t.finish();
  }

  // The first five are the shapes README.md asks you to predict. Each of the
  // four after 'cube' moves away from it in one direction.
  std::vector<Case> cases = {
      {"cube", 4, 16, 16, 0, "seq = hidden, batch a multiple of 4, no window"},
      {"wide", 4, 6, 20, 0, "hidden > seq, neither a multiple of 8"},
      {"tall", 4, 20, 6, 0, "seq > hidden, neither a multiple of 8"},
      {"odd batch", 5, 16, 16, 0, "batch not a multiple of 4"},
      {"window", 4, 16, 16, 1, "three positions per mean"},
      // Mixtures. These pass once all four defects are gone.
      {"mixed", 5, 13, 21, 2, "everything awkward at once"},
      {"mixed", 3, 5, 9, 7, "window longer than the sequence"},
      {"mixed", 2, 37, 3, 3, "long and narrow"},
      {"mixed", 1, 1, 1, 1, "a single element"},
  };
  if (!kExactBuffers) cases.push_back({"mixed", 6, 300, 200, 4, "many blocks on every axis"});

  for (const Case& c : cases) {
    char name[200];
    std::snprintf(name, sizeof name, "%s: batch %d, seq %d, hidden %d, r %d (%s)", c.name, c.batch, c.seq, c.hidden, c.r, c.why);
    t.check(name, [&] { return runAndCheck(c); });
  }
  return t.finish();
}

static int bench(const pmpp::Args&) { std::printf("this lab has no benchmark.\n"); return 0; }

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
