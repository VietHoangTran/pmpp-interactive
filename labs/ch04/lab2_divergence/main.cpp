// Lab 4.2 driver: tests and times three kernels whose control flow the data
// decides. You do not need to edit this file; your work goes in work/.
//
// The patterns are built from the device's own width, never from a literal:
//   CUDA, HIP  warpSize from the device properties. The cut of a block into
//              warps is documented there (linear thread order, consecutive
//              runs), so a thread's warp and lane are computed.
//   Metal      threadExecutionWidth of the compiled pipeline. The cut of a
//              threadgroup into SIMD-groups is "implementation-defined", so a
//              provided probe kernel (provided/simd_probe.metal) is dispatched
//              in the same shape first and reports each thread's SIMD-group and
//              lane. Read that file's comment for what this does not establish.
//   emulator   has no warps and no cost model. The tests still run every
//              pattern, for three pretend widths, to check the kernels' logic.
//
// How a write past the array is caught: on the emulator the buffers are exactly
// n elements and AddressSanitizer stops at the first access outside them. On a
// device the buffers cover the whole grid and the output's surplus elements
// must come back untouched.
#include <map>
#include <tuple>

#include "pmpp/harness.hpp"
using pmpp::Buf;

// ---- the two paths, as in paths.h and paths.metal.h (keep the three in step) ----
static const float kMulA = 0.999f, kAddA = 0.01f, kMulB = 0.998f, kAddB = -0.02f;
static float cpuPath(float x, int steps, bool pathB) {
  double v = x;
  const double m = pathB ? kMulB : kMulA, c = pathB ? kAddB : kAddA;
  for (int k = 0; k < steps; ++k) v = v * m + c;
  return (float)v;
}

enum Kernel { kBranch, kSelect, kLoop };

// ---- entry points implemented in <variant>/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include <fstream>
#include <unistd.h>

#include "pmpp/metal.h"
void encodeBranch(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> in, id<MTLBuffer> which,
                  id<MTLBuffer> out, uint32_t n, uint32_t iters, uint32_t threadsPerThreadgroup);
void encodeSelect(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> in, id<MTLBuffer> which,
                  id<MTLBuffer> out, uint32_t n, uint32_t iters, uint32_t threadsPerThreadgroup);
void encodeLoop(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> in, id<MTLBuffer> trips,
                id<MTLBuffer> out, uint32_t n, uint32_t threadsPerThreadgroup);
using pmpp::metal::buffer;

static const char* kUnit = "SIMD-group";
static const char* kBlock = "threadgroup";
static const char* kWidthSource = "threadExecutionWidth of the branch_kernel pipeline";

static std::string dirOf(const std::string& p) { return p.substr(0, p.find_last_of('/')); }

// The Metal source is compiled at run time, and a library made from a source
// string has no search path for a lab's own headers. So the driver pastes
// paths.metal.h in place of the line `#include "paths.metal.h"` and compiles the
// result. #line directives keep the compiler's messages pointing at the lines
// of the two original files.
static std::string g_pasted;
static void removePasted() { if (!g_pasted.empty()) unlink(g_pasted.c_str()); }
static const std::string& pastedSource() {
  if (!g_pasted.empty()) return g_pasted;
  const std::string kernelPath = PMPP_METAL_SOURCE, headerPath = dirOf(kernelPath) + "/paths.metal.h";
  std::ifstream kin(kernelPath), hin(headerPath);
  std::ostringstream out;
  std::string line;
  int lineNo = 0;
  while (std::getline(kin, line)) {
    ++lineNo;
    const size_t inc = line.find("#include"), name = line.find("\"paths.metal.h\"");
    if (inc != std::string::npos && name != std::string::npos && line.find_first_not_of(" \t") == inc && hin) {
      out << "#line 1 \"paths.metal.h\"\n" << hin.rdbuf() << "\n#line " << lineNo + 1 << " \"kernel.metal\"\n";
    } else {
      out << line << "\n";
    }
  }
  g_pasted = std::string(NSTemporaryDirectory().UTF8String) + "pmpp_lab42_" + PMPP_VARIANT + "_" + std::to_string(getpid()) + ".metal";
  std::ofstream(g_pasted) << out.str();
  std::atexit(removePasted);
  return g_pasted;
}
static id<MTLComputePipelineState> pso(Kernel k) {
  return pmpp::metal::pipeline(pastedSource(), k == kBranch ? "branch_kernel" : k == kSelect ? "select_kernel" : "loop_kernel");
}
// Metal has no device-wide width: it is a property of a compiled pipeline.
static int deviceWidth() { id<MTLComputePipelineState> p = pso(kBranch); return p ? (int)p.threadExecutionWidth : 0; }
static int widthOf(Kernel k) { id<MTLComputePipelineState> p = pso(k); return p ? (int)p.threadExecutionWidth : 0; }
static int maxThreadsPerBlock() {
  NSUInteger m = pmpp::metal::device().maxThreadsPerThreadgroup.width;
  for (Kernel k : {kBranch, kSelect, kLoop}) {
    id<MTLComputePipelineState> p = pso(k);
    if (!p) return 0;
    m = std::min(m, p.maxTotalThreadsPerThreadgroup);
  }
  return (int)m;
}
static std::vector<int> testWidths() { return {deviceWidth()}; }
// `ctl` is which[] for the first two kernels and trips[] for the third.
static pmpp::Timing runKernel(Kernel k, Buf* in, Buf* ctl, Buf* out, int n, int iters, int threadsPerBlock) {
  id<MTLComputePipelineState> p = pso(k);
  if (!p) return {0, 0};
  return pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    if (k == kBranch) encodeBranch(enc, p, buffer(in), buffer(ctl), buffer(out), (uint32_t)n, (uint32_t)iters, (uint32_t)threadsPerBlock);
    else if (k == kSelect) encodeSelect(enc, p, buffer(in), buffer(ctl), buffer(out), (uint32_t)n, (uint32_t)iters, (uint32_t)threadsPerBlock);
    else encodeLoop(enc, p, buffer(in), buffer(ctl), buffer(out), (uint32_t)n, (uint32_t)threadsPerBlock);
  });
}
// Dispatches the probe kernel as `threads / threadsPerBlock` whole threadgroups
// and reads back what each thread reported. Returns an error message, or "".
static std::string probeGroups(int threadsPerBlock, int threads, std::vector<uint32_t>& group, std::vector<uint32_t>& lane, int& width) {
  id<MTLComputePipelineState> p = pmpp::metal::pipeline(dirOf(dirOf(PMPP_METAL_SOURCE)) + "/provided/simd_probe.metal", "simd_probe");
  if (!p) return "the probe kernel did not build: " + pmpp::take_error();
  if ((NSUInteger)threadsPerBlock > p.maxTotalThreadsPerThreadgroup) return "the probe pipeline does not accept this threadgroup size";
  Buf *g = pmpp::alloc((size_t)threads * 4), *l = pmpp::alloc((size_t)threads * 4), *info = pmpp::alloc(8);
  pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    [enc setComputePipelineState:p];
    [enc setBuffer:buffer(g) offset:0 atIndex:0];
    [enc setBuffer:buffer(l) offset:0 atIndex:1];
    [enc setBuffer:buffer(info) offset:0 atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(threads / threadsPerBlock), 1, 1)
        threadsPerThreadgroup:MTLSizeMake((NSUInteger)threadsPerBlock, 1, 1)];
  });
  group.resize((size_t)threads); lane.resize((size_t)threads);
  uint32_t i2[2] = {0, 0};
  pmpp::download(g, group.data(), (size_t)threads * 4);
  pmpp::download(l, lane.data(), (size_t)threads * 4);
  pmpp::download(info, i2, 8);
  pmpp::release(g); pmpp::release(l); pmpp::release(info);
  width = (int)i2[0];
  const std::string e = pmpp::take_error();
  return e.empty() ? "" : "the probe kernel did not run: " + e;
}
#else
void launchBranch(const float* in, const int* which, float* out, int n, int iters, int threadsPerBlock);
void launchSelect(const float* in, const int* which, float* out, int n, int iters, int threadsPerBlock);
void launchLoop(const float* in, const int* trips, float* out, int n, int threadsPerBlock);

static const char* kUnit = "warp";
static const char* kBlock = "block";
static const char* kWidthSource = "the device's warpSize";
static int deviceWidth() { return (int)pmpp::device_prop("warpSize", 0); }
static int widthOf(Kernel) { return deviceWidth(); }
static int maxThreadsPerBlock() {
  return (int)std::min(pmpp::device_prop("maxThreadsPerBlock", 0), pmpp::device_prop("maxThreadsDim.x", 0));
}
#if defined(PMPP_BACKEND_EMU)
// The emulator has nothing that behaves like a warp when threads diverge, so no
// width is "its" width. Three pretend widths exercise the patterns: a small one,
// and the two that CUDA and HIP devices report today.
static std::vector<int> testWidths() { return {4, 32, 64}; }
#else
static std::vector<int> testWidths() { return {deviceWidth()}; }
#endif
static pmpp::Timing runKernel(Kernel k, Buf* in, Buf* ctl, Buf* out, int n, int iters, int threadsPerBlock) {
  const float* x = (const float*)pmpp::raw(in);
  const int* c = (const int*)pmpp::raw(ctl);
  float* y = (float*)pmpp::raw(out);
  return pmpp::run([&](void*) {
    if (k == kBranch) launchBranch(x, c, y, n, iters, threadsPerBlock);
    else if (k == kSelect) launchSelect(x, c, y, n, iters, threadsPerBlock);
    else launchLoop(x, c, y, n, threadsPerBlock);
  });
}
#endif

// Exact buffers only where AddressSanitizer is watching.
#if defined(PMPP_BACKEND_EMU)
static const bool kExactBuffers = true;
#else
static const bool kExactBuffers = false;
#endif

// ---- which group and which lane each thread of a grid is in ----
struct Layout {
  int width = 0;                      // threads per warp or SIMD-group
  std::vector<uint32_t> group, lane;  // per thread of the grid, groups numbered across the grid
  bool consecutive = true;            // group == index / width and lane == index % width for every thread
  std::string error;
};
// `threads` is a multiple of threadsPerBlock, which is a multiple of `width`.
static const Layout& layoutFor(int threadsPerBlock, int threads, int width) {
  static std::map<std::tuple<int, int, int>, Layout> cache;
  const auto key = std::make_tuple(threadsPerBlock, threads, width);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  Layout L;
#if defined(PMPP_BACKEND_METAL)
  L.error = probeGroups(threadsPerBlock, threads, L.group, L.lane, L.width);
  if (L.error.empty() && L.width <= 0) L.error = "the probe kernel reported a SIMD-group width of 0";
  if (L.error.empty())
    for (int i = 0; i < threads && L.consecutive; ++i)
      L.consecutive = L.group[(size_t)i] == (uint32_t)(i / L.width) && L.lane[(size_t)i] == (uint32_t)(i % L.width);
#else
  // The documented cut: threads in linear order, consecutive runs of `width`,
  // starting again in every block.
  L.width = width;
  L.group.resize((size_t)threads); L.lane.resize((size_t)threads);
  const int perBlock = (threadsPerBlock + width - 1) / width;
  for (int i = 0; i < threads; ++i) {
    const int b = i / threadsPerBlock, t = i % threadsPerBlock;
    L.group[(size_t)i] = (uint32_t)(b * perBlock + t / width);
    L.lane[(size_t)i] = (uint32_t)(t % width);
  }
#endif
  return cache.emplace(key, std::move(L)).first->second;
}

// ---- the patterns: which[i] == 0 is path A, 1 is path B ----
enum Pattern { kAllA, kAllB, kFirstHalf, kAltGroups, kHalfLanes, kAltLanes, kOneLane, kRandomHalf, kPatternCount };
static std::string patternName(Pattern p) {
  switch (p) {
    case kAllA: return "all A";
    case kAllB: return "all B";
    case kFirstHalf: return "first half of the array A, second half B";
    case kAltGroups: return std::string("alternating ") + kUnit + "s";
    case kHalfLanes: return std::string("half of each ") + kUnit + "'s lanes";
    case kAltLanes: return "alternating lanes";
    case kOneLane: return std::string("one lane per ") + kUnit;
    default: return "random half (seeded)";
  }
}
static std::vector<int> makePattern(Pattern p, int n, const Layout& L) {
  std::vector<int> w((size_t)n);
  uint32_t x = 0x9E3779B9u;
  for (int i = 0; i < n; ++i) {
    const uint32_t g = L.group[(size_t)i], l = L.lane[(size_t)i];
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    switch (p) {
      case kAllA: w[(size_t)i] = 0; break;
      case kAllB: w[(size_t)i] = 1; break;
      case kFirstHalf: w[(size_t)i] = i >= n / 2; break;
      case kAltGroups: w[(size_t)i] = (int)(g & 1u); break;
      case kHalfLanes: w[(size_t)i] = l >= (uint32_t)L.width / 2; break;
      case kAltLanes: w[(size_t)i] = (int)(l & 1u); break;
      case kOneLane: w[(size_t)i] = l == 0; break;
      default: w[(size_t)i] = (int)((x >> 9) & 1u); break;
    }
  }
  return w;
}
// The same number of A and B, with every A in front: the pattern to compare with.
static std::vector<int> sortedSameMix(const std::vector<int>& which) {
  size_t b = 0;
  for (int v : which) b += v != 0;
  std::vector<int> s(which.size(), 0);
  std::fill(s.end() - (std::ptrdiff_t)b, s.end(), 1);
  return s;
}
static std::vector<int> shuffled(std::vector<int> v, uint32_t seed) {
  uint32_t x = seed * 2654435761u + 1u;
  for (size_t i = v.size(); i > 1; --i) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    std::swap(v[i - 1], v[x % i]);
  }
  return v;
}

// ---- one launch on fresh buffers ----
static const float kUnwritten = -12345.0f;   // no path reaches it from an input in [-1, 1]

struct Shape { int width, threadsPerBlock, n; };
static std::string shapeName(const Shape& s) {
  char b[160];
  std::snprintf(b, sizeof b, "width %d, %d threads per %s, n = %d: ", s.width, s.threadsPerBlock, kBlock, s.n);
  return b;
}
// Runs kernel `k` and returns out[0 .. n + surplus). On a device the surplus
// elements are the rest of the grid plus one more block.
static std::vector<float> launchOnce(Kernel k, const Shape& s, const std::vector<float>& x, const std::vector<int>& ctl, int iters) {
  const size_t n = (size_t)s.n;
  const size_t grid = (n + (size_t)s.threadsPerBlock - 1) / (size_t)s.threadsPerBlock * (size_t)s.threadsPerBlock;
  const size_t inLen = kExactBuffers ? n : grid, outLen = kExactBuffers ? n : grid + (size_t)s.threadsPerBlock;
  std::vector<float> hx(inLen, 0.5f), hy(outLen, kUnwritten);
  std::vector<int> hc(inLen, 0);
  std::copy(x.begin(), x.end(), hx.begin());
  std::copy(ctl.begin(), ctl.end(), hc.begin());
  Buf *in = pmpp::alloc(inLen * 4), *c = pmpp::alloc(inLen * 4), *out = pmpp::alloc(outLen * 4);
  pmpp::upload(in, hx.data(), inLen * 4);
  pmpp::upload(c, hc.data(), inLen * 4);
  pmpp::upload(out, hy.data(), outLen * 4);
  runKernel(k, in, c, out, s.n, iters, s.threadsPerBlock);
  pmpp::download(out, hy.data(), outLen * 4);
  pmpp::release(in); pmpp::release(c); pmpp::release(out);
  return hy;
}

// Empty string = `got` matches `ref` on [0, n) and nothing after n was written.
// `other` (optional) is what the opposite path would have produced.
static std::string judge(const std::vector<float>& got, const std::vector<float>& ref, const std::vector<float>* other,
                         const std::vector<int>& ctl, const char* ctlName) {
  const size_t n = ref.size();
  const pmpp::Cmp cmp = pmpp::compare(got.data(), ref.data(), nullptr, n, 1e-4);
  size_t unwritten = 0, stray = 0;
  for (size_t i = 0; i < n; ++i) unwritten += got[i] == kUnwritten;
  for (size_t i = n; i < got.size(); ++i) stray += got[i] != kUnwritten;
  if (cmp.ok && stray == 0) return "";
  char b[260];
  std::string msg;
  if (unwritten == n) {
    msg = "no element was written";
  } else if (!cmp.ok) {
    const size_t f = cmp.first_bad;
    std::snprintf(b, sizeof b, "%zu of %zu elements wrong; first at i = %zu (%s[i] = %d): got %.6g, expected %.6g", cmp.bad, n, f,
                  ctlName, ctl[f], (double)got[f], (double)ref[f]);
    msg = b;
    if (other && std::fabs((double)got[f] - (double)(*other)[f]) <= 1e-4 * (1.0 + std::fabs((double)(*other)[f])))
      msg += ", which is the other path's value";
    if (unwritten) { std::snprintf(b, sizeof b, "; %zu elements were never written", unwritten); msg += b; }
  }
  if (stray) {
    std::snprintf(b, sizeof b, "%s%zu elements were written after the last element of the array", msg.empty() ? "" : "; ", stray);
    msg += b;
  }
  return msg;
}

// The shapes every test runs: block sizes of 1, 2 and 8 widths, each with a
// length that is a multiple of the block size and one that ends in the middle
// of a warp.
static std::vector<Shape> testShapes(int maxBlock) {
  std::vector<Shape> shapes;
  const int blocks = kExactBuffers ? 2 : 64, raggedBlocks = kExactBuffers ? 1 : 37;
  for (int w : testWidths())
    for (int k : {1, 2, 8}) {
      const int t = k * w;
      if (t > maxBlock) continue;
      shapes.push_back({w, t, blocks * t});
      shapes.push_back({w, t, raggedBlocks * t + w / 2 + 1});
    }
  return shapes;
}

static const int kTestIters = 23;

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
  const std::vector<Shape> shapes = testShapes(maxBlock);
#if defined(PMPP_BACKEND_EMU)
  std::printf("patterns built for pretend widths 4, 32 and 64 (the emulator has no warps)\n");
#else
  std::printf("patterns built for width %d (%s)\n", width, kWidthSource);
#endif
#if defined(PMPP_BACKEND_METAL)
  for (const Shape& s : shapes) {
    const int threads = (s.n + s.threadsPerBlock - 1) / s.threadsPerBlock * s.threadsPerBlock;
    const Layout& L = layoutFor(s.threadsPerBlock, threads, s.width);
    if (s.n % s.threadsPerBlock != 0) continue;
    if (!L.error.empty()) { std::printf("probe, %d threads per threadgroup: %s\n", s.threadsPerBlock, L.error.c_str()); continue; }
    std::printf("probe, %d threads per threadgroup: SIMD-groups of %d threads, %s\n", s.threadsPerBlock, L.width,
                L.consecutive ? "cut as consecutive runs of the thread index" : "NOT cut as consecutive runs of the thread index");
  }
#endif

  // Runs branchKernel (and selectKernel) on one pattern over every shape.
  auto patternCase = [&](Pattern p, bool select) -> std::string {
    for (const Shape& s : shapes) {
      const int threads = (s.n + s.threadsPerBlock - 1) / s.threadsPerBlock * s.threadsPerBlock;
      const Layout& L = layoutFor(s.threadsPerBlock, threads, s.width);
      if (!L.error.empty()) return shapeName(s) + L.error;
      const std::vector<int> which = makePattern(p, s.n, L);
      const std::vector<float> x = pmpp::random_floats((size_t)s.n, 4200u + (uint32_t)s.n + (uint32_t)p * 31u);
      std::vector<float> ref((size_t)s.n), other((size_t)s.n);
      for (int i = 0; i < s.n; ++i) {
        ref[(size_t)i] = cpuPath(x[(size_t)i], kTestIters, which[(size_t)i] != 0);
        other[(size_t)i] = cpuPath(x[(size_t)i], kTestIters, which[(size_t)i] == 0);
      }
      const std::vector<float> got = launchOnce(select ? kSelect : kBranch, s, x, which, kTestIters);
      std::string e = pmpp::take_error();
      if (!e.empty()) return shapeName(s) + "launch error: " + e;
      e = judge(got, ref, &other, which, "which");
      if (!e.empty()) return shapeName(s) + e;
      if (select) {
        // Both kernels compute the same expression for the path they keep, so
        // they must agree with each other at least as well as with the reference.
        const std::vector<float> branch = launchOnce(kBranch, s, x, which, kTestIters);
        pmpp::take_error();
        const pmpp::Cmp c = pmpp::compare(got.data(), branch.data(), nullptr, (size_t)s.n, 1e-4);
        if (!c.ok) {
          char b[200];
          std::snprintf(b, sizeof b, "matches the reference but not branchKernel: %zu elements differ; first at i = %zu: %.6g against %.6g",
                        c.bad, c.first_bad, (double)got[c.first_bad], (double)branch[c.first_bad]);
          return shapeName(s) + b;
        }
      }
    }
    return "";
  };
  for (int p = 0; p < kPatternCount; ++p) {
    t.check("branchKernel, " + patternName((Pattern)p), [&] { return patternCase((Pattern)p, false); });
    t.check("selectKernel, " + patternName((Pattern)p) + " (and equal to branchKernel)", [&] { return patternCase((Pattern)p, true); });
  }

  // loopKernel: trip counts per thread.
  struct Trips { const char* name; int kind; };
  const int maxTrips = 40;
  for (const Trips& tr : {Trips{"every trip count 0", 0}, Trips{"trip counts 0 to 40, sorted", 1},
                          Trips{"trip counts 0 to 40, shuffled", 2}, Trips{"every trip count 20", 3}}) {
    t.check(std::string("loopKernel, ") + tr.name, [&]() -> std::string {
      for (const Shape& s : shapes) {
        std::vector<int> trips((size_t)s.n);
        for (int i = 0; i < s.n; ++i)
          trips[(size_t)i] = tr.kind == 0 ? 0 : tr.kind == 3 ? maxTrips / 2 : (int)((long long)(maxTrips + 1) * i / s.n);
        if (tr.kind == 2) trips = shuffled(trips, 77u + (uint32_t)s.n);
        const std::vector<float> x = pmpp::random_floats((size_t)s.n, 4250u + (uint32_t)s.n);
        std::vector<float> ref((size_t)s.n);
        for (int i = 0; i < s.n; ++i) ref[(size_t)i] = cpuPath(x[(size_t)i], trips[(size_t)i], false);
        const std::vector<float> got = launchOnce(kLoop, s, x, trips, 0);
        std::string e = pmpp::take_error();
        if (!e.empty()) return shapeName(s) + "launch error: " + e;
        e = judge(got, ref, nullptr, trips, "trips");
        if (!e.empty()) return shapeName(s) + e;
      }
      return "";
    });
  }
  return t.finish();
}

// ---- benchmark ----
static int bench(const pmpp::Args& a) {
  const int width = deviceWidth(), maxBlock = maxThreadsPerBlock();
  if (width <= 0 || maxBlock < width) {
    std::printf("bench skipped: could not read %s, or a block cannot hold one %s.\n", kWidthSource, kUnit);
    return 1;
  }
  // Blocks of 8 widths, or as many whole widths as the device allows in a block.
  const int tpb = std::min(8 * width, maxBlock / width * width);
  const int n = ((1 << 20) + tpb - 1) / tpb * tpb;   // about a million threads, in whole blocks
  const Layout& L = layoutFor(tpb, n, width);
  if (!L.error.empty()) { std::printf("bench skipped: %s\n", L.error.c_str()); return 1; }

  const std::vector<float> x = pmpp::random_floats((size_t)n, 42);
  Buf *in = pmpp::alloc((size_t)n * 4), *ctl = pmpp::alloc((size_t)n * 4), *out = pmpp::alloc((size_t)n * 4);
  pmpp::upload(in, x.data(), (size_t)n * 4);
  auto timeIt = [&](Kernel k, const std::vector<int>& c, int iters) {
    pmpp::upload(ctl, c.data(), (size_t)n * 4);
    return pmpp::time_it(a.reps, [&] { return runKernel(k, in, ctl, out, n, iters, tpb); }).min_s;
  };

  // Grow the path length until the uniform case takes long enough to time.
  const std::vector<int> allA = makePattern(kAllA, n, L);
  int iters = 64;
  double uniform = 0;
  for (;;) {
    uniform = timeIt(kBranch, allA, iters);
    if (uniform > 0.05 || iters >= (1 << 20)) break;
    iters *= 4;
  }
  std::string err = pmpp::take_error();
  if (!err.empty() || uniform <= 0) { std::printf("branchKernel failed to run: %s\n", err.c_str()); return 1; }

  std::printf("width %d (%s), %d threads per %s, %d threads, %d steps per path, best of %d runs\n", width, kWidthSource, tpb, kBlock,
              n, iters, a.reps);
#if defined(PMPP_BACKEND_METAL)
  std::printf("probe: SIMD-groups of %d threads, %s\n", L.width,
              L.consecutive ? "cut as consecutive runs of the thread index" : "NOT cut as consecutive runs of the thread index");
  if (L.width != width || widthOf(kSelect) != width || widthOf(kLoop) != width)
    std::printf("note: the widths differ: probe %d, branch_kernel %d, select_kernel %d, loop_kernel %d\n", L.width, width,
                widthOf(kSelect), widthOf(kLoop));
#endif
  std::printf("\n  %-42s %10s %10s %8s   %10s %8s\n", "", "branch", "sorted,", "branch /", "select", "select /");
  std::printf("  %-42s %10s %10s %8s   %10s %8s\n", "pattern", "ms", "same mix", "sorted", "ms", "branch");

  std::map<size_t, double> sortedSeconds;   // by number of B: branchKernel on the sorted pattern with that mix
  for (int pi = 0; pi < kPatternCount; ++pi) {
    const Pattern p = (Pattern)pi;
    const std::vector<int> which = makePattern(p, n, L), sorted = sortedSameMix(which);
    size_t nb = 0;
    for (int v : which) nb += v != 0;
    const double branch = timeIt(kBranch, which, iters);
    if (which == sorted) sortedSeconds[nb] = branch;
    if (!sortedSeconds.count(nb)) sortedSeconds[nb] = timeIt(kBranch, sorted, iters);
    const double base = sortedSeconds[nb];
    const double select = timeIt(kSelect, which, iters);
    err = pmpp::take_error();
    if (!err.empty() || branch <= 0 || select <= 0) { std::printf("  %-42s failed: %s\n", patternName(p).c_str(), err.c_str()); continue; }
    std::printf("  %-42s %10.3f %10.3f %8.2f   %10.3f %8.2f\n", patternName(p).c_str(), branch * 1e3, base * 1e3, branch / base,
                select * 1e3, select / branch);
    pmpp::bench_row(a, "branch").s("pattern", patternName(p)).n("width", width).n("threads_per_block", tpb).n("n", n)
        .n("iters", iters).n("fraction_b", (double)nb / n).n("seconds", branch).n("sorted_seconds", base)
        .n("ratio_to_sorted", branch / base).emit();
    pmpp::bench_row(a, "select").s("pattern", patternName(p)).n("width", width).n("threads_per_block", tpb).n("n", n)
        .n("iters", iters).n("fraction_b", (double)nb / n).n("seconds", select).n("sorted_seconds", base)
        .n("ratio_to_sorted", select / base).n("ratio_to_branch", select / branch).emit();
  }

  // loopKernel: the same multiset of trip counts in two orders, and every count
  // replaced by the mean.
  std::vector<int> sortedTrips((size_t)n);
  double total = 0;
  for (int i = 0; i < n; ++i) { sortedTrips[(size_t)i] = (int)((long long)(2 * (long long)iters + 1) * i / n); total += sortedTrips[(size_t)i]; }
  const int mean = (int)(total / n + 0.5);
  const std::vector<int> shuffledTrips = shuffled(sortedTrips, 4242u), equalTrips((size_t)n, mean);
  const double ls = timeIt(kLoop, sortedTrips, 0), lsh = timeIt(kLoop, shuffledTrips, 0), le = timeIt(kLoop, equalTrips, 0);
  err = pmpp::take_error();
  if (!err.empty() || ls <= 0) {
    std::printf("\n  loopKernel failed to run: %s\n", err.c_str());
  } else {
    std::printf("\n  loopKernel, trip counts 0 to %d, mean %d            ms   / sorted\n", 2 * iters, mean);
    std::printf("  %-42s %10.3f %8.2f\n", "sorted along the array", ls * 1e3, 1.0);
    std::printf("  %-42s %10.3f %8.2f\n", "the same counts, shuffled", lsh * 1e3, lsh / ls);
    std::printf("  %-42s %10.3f %8.2f\n", "every count equal to the mean", le * 1e3, le / ls);
    const double equalTotal = (double)mean * n;
    pmpp::bench_row(a, "loop").s("order", "sorted").n("width", width).n("threads_per_block", tpb).n("n", n).n("mean_trips", mean)
        .n("total_steps", total).n("seconds", ls).n("ratio_to_sorted", 1.0).emit();
    pmpp::bench_row(a, "loop").s("order", "shuffled").n("width", width).n("threads_per_block", tpb).n("n", n).n("mean_trips", mean)
        .n("total_steps", total).n("seconds", lsh).n("ratio_to_sorted", lsh / ls).emit();
    pmpp::bench_row(a, "loop").s("order", "all equal to the mean").n("width", width).n("threads_per_block", tpb).n("n", n)
        .n("mean_trips", mean).n("total_steps", equalTotal).n("seconds", le).n("ratio_to_sorted", le / ls).emit();
  }
  std::printf("\nThese are measurements of your device, of these three kernels, at this size.\n");
  pmpp::release(in); pmpp::release(ctl); pmpp::release(out);
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
