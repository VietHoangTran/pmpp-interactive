// Lab 4.1 driver: checks what you read from the device, the launch configuration
// you derive from it, and the linear layer launched with that configuration.
// Second half of project milestone M1.
// You do not need to edit this file; your work goes in work/.
#include <limits>
#include <map>
#include <set>

#include "pmpp/harness.hpp"
#include "launch_config.h"
using pmpp::Buf;

// ---------- the CPU reference ----------
// Y[b][o] = sum over k of X[b][k] * W[k][o]: X is batch x in, W is in x out,
// row-major. `mag` receives sum |x*w| per output, which scales the tolerance.
static void cpu_linear(const std::vector<float>& x, const std::vector<float>& w, int batch, int in, int out,
                       std::vector<float>& y, std::vector<float>& mag) {
  y.assign((size_t)batch * out, 0.f);
  mag.assign((size_t)batch * out, 0.f);
  std::vector<double> acc(out), am(out);
  for (int b = 0; b < batch; ++b) {
    std::fill(acc.begin(), acc.end(), 0.0);
    std::fill(am.begin(), am.end(), 0.0);
    for (int k = 0; k < in; ++k) {
      const double xv = x[(size_t)b * in + k];
      const float* wrow = w.data() + (size_t)k * out;
      for (int o = 0; o < out; ++o) { const double p = xv * wrow[o]; acc[o] += p; am[o] += std::fabs(p); }
    }
    for (int o = 0; o < out; ++o) {
      y[(size_t)b * out + o] = (float)acc[o];
      mag[(size_t)b * out + o] = (float)am[o];
    }
  }
}

// ---------- one layer on the device ----------
// Y is pre-filled with a value no sum equals, so an output nobody writes is
// detected. On the emulator every buffer has exactly its size and
// AddressSanitizer reports an access outside it. Everywhere else each buffer
// gets slack after its last element: NaN after X and W (reading it gives NaN),
// a canary after Y (writing it is caught).
static const float kUnwritten = 12345.0f;
static const float kCanary = -54321.0f;

struct Layer {
  int batch, in, out;
  size_t slack;
  std::vector<float> x, w, ref, mag;
  Buf *X = nullptr, *W = nullptr, *Y = nullptr;

  Layer(int batch_, int in_, int out_, uint32_t seed) : batch(batch_), in(in_), out(out_) {
    slack = pmpp::is_emu() ? 0 : (size_t)64 * out + 4096;
    x = pmpp::random_floats((size_t)batch * in, seed);
    w = pmpp::random_floats((size_t)in * out, seed + 1);
    X = padded(x, std::numeric_limits<float>::quiet_NaN());
    W = padded(w, std::numeric_limits<float>::quiet_NaN());
    Y = pmpp::alloc(((size_t)batch * out + slack) * 4);
    prefill();
  }
  Layer(const Layer&) = delete;
  ~Layer() { pmpp::release(X); pmpp::release(W); pmpp::release(Y); }

  Buf* padded(const std::vector<float>& v, float fill) const {
    std::vector<float> host(v);
    host.resize(v.size() + slack, fill);
    Buf* b = pmpp::alloc(host.size() * 4);
    pmpp::upload(b, host.data(), host.size() * 4);
    return b;
  }
  void prefill() {
    std::vector<float> host((size_t)batch * out, kUnwritten);
    host.resize(host.size() + slack, kCanary);
    pmpp::upload(Y, host.data(), host.size() * 4);
  }
  double flop() const { return 2.0 * batch * (double)in * out; }

  // Empty string = Y matches the reference and nothing outside Y was written.
  std::string check() {
    if (ref.empty()) cpu_linear(x, w, batch, in, out, ref, mag);
    const size_t n = (size_t)batch * out;
    std::vector<float> got(n + slack);
    pmpp::download(Y, got.data(), got.size() * 4);
    std::string msg = pmpp::describe(pmpp::compare(got.data(), ref.data(), mag.data(), n), out);
    size_t unwritten = 0, past = 0;
    for (size_t i = 0; i < n; ++i) unwritten += (got[i] == kUnwritten);
    for (size_t i = n; i < got.size(); ++i) past += !(got[i] == kCanary);
    auto add = [&](const std::string& s) { msg += (msg.empty() ? "" : "; ") + s; };
    if (unwritten) add(std::to_string(unwritten) + " of " + std::to_string(n) + " outputs were never written");
    if (past) add(std::to_string(past) + " values were stored past the end of Y");
    return msg;
  }
};

static void append(std::string& msg, const std::string& s) { msg += (msg.empty() ? "" : "; ") + s; }

// Times two configurations in alternation, so that whatever else the machine is
// doing falls on both, and returns the best run of each.
static std::pair<double, double> bestOfBoth(int reps, const std::function<pmpp::Timing()>& first,
                                            const std::function<pmpp::Timing()>& second) {
  first(); second();   // untimed warm-up
  double a = 1e30, b = 1e30;
  for (int i = 0; i < reps; ++i) { a = std::min(a, first().best()); b = std::min(b, second().best()); }
  return {a, b};
}

struct LayerShape { int batch, in, out; const char* why; };
static const LayerShape kTestLayers[] = {
    {1, 96, 1000, "one row: the decode shape"},
    {3, 17, 40, "fewer rows than any sensible block is tall"},
    {37, 29, 53, "nothing is a multiple of anything"},
    {64, 48, 257, "a batch, one output past a multiple of the width"},
};

#if defined(PMPP_BACKEND_METAL)
// =====================================================================================
// Metal
// =====================================================================================
#import <Metal/Metal.h>
#include "pmpp/metal.h"

MetalInfo queryMetal(id<MTLDevice> device, id<MTLComputePipelineState> pipeline);
MTLSize encodeLinear(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> X, id<MTLBuffer> W,
                     id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out);
void encodeProbe(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> rec, uint32_t gridWidth,
                 uint32_t gridHeight, MTLSize threadsPerThreadgroup);

static id<MTLComputePipelineState> linearPipeline() { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "linear_layer"); }
static id<MTLComputePipelineState> probePipeline() { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "simd_probe"); }

// Your encodeLinear, `repeat` times in one compute pass. `used` receives the
// threadgroup size it returned.
static pmpp::Timing runDerived(Layer& L, int repeat, MTLSize* used) {
  auto pso = linearPipeline();
  if (!pso) return {0, 0};
  return pmpp::run([&](void* v) {
    auto* pass = (pmpp::metal::Pass*)v;
    for (int r = 0; r < repeat; ++r) {
      MTLSize tg = encodeLinear(pass->encoder, pso, pmpp::metal::buffer(L.X), pmpp::metal::buffer(L.W),
                                pmpp::metal::buffer(L.Y), (uint32_t)L.batch, (uint32_t)L.in, (uint32_t)L.out);
      if (used) *used = tg;
    }
  });
}

// The benchmark's baseline, encoded here: the same kernel and the same
// dispatchThreads call with a 16 x 16 threadgroup, whatever the pipeline reports.
static pmpp::Timing runFixed16(Layer& L, int repeat) {
  auto pso = linearPipeline();
  if (!pso) return {0, 0};
  return pmpp::run([&](void* v) {
    auto* pass = (pmpp::metal::Pass*)v;
    struct { uint32_t batch, in, out; } d = {(uint32_t)L.batch, (uint32_t)L.in, (uint32_t)L.out};
    for (int r = 0; r < repeat; ++r) {
      [pass->encoder setComputePipelineState:pso];
      [pass->encoder setBuffer:pmpp::metal::buffer(L.X) offset:0 atIndex:0];
      [pass->encoder setBuffer:pmpp::metal::buffer(L.W) offset:0 atIndex:1];
      [pass->encoder setBuffer:pmpp::metal::buffer(L.Y) offset:0 atIndex:2];
      [pass->encoder setBytes:&d length:sizeof(d) atIndex:3];
      [pass->encoder dispatchThreads:MTLSizeMake(L.out, L.batch, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    }
  });
}

// ---- the probe ----
struct ProbeSummary {
  std::string error;          // empty = the records satisfy the specification's invariants
  uint32_t width = 0;         // [[threads_per_simdgroup]] as the kernel reported it
  int groupsPerThreadgroup = 0;   // distinct SIMD-group indices in a full threadgroup
  bool consecutive = false;   // full threadgroups: group == linear / width and lane == linear % width
};

// Runs your probe on a gw x gh grid with tw x th threadgroups and examines the
// records. Which threadgroup a thread belongs to follows from the
// specification's formula, thread position in grid = threadgroup position *
// threadgroup size + position in threadgroup. Nothing is assumed about how a
// threadgroup is cut into SIMD-groups.
static ProbeSummary runProbe(int gw, int gh, int tw, int th) {
  ProbeSummary s;
  auto pso = probePipeline();
  if (!pso) { s.error = "the probe kernel did not build"; return s; }
  const size_t threads = (size_t)gw * gh;
  std::vector<uint32_t> rec(3 * threads, 0xFFFFFFFFu);
  Buf* R = pmpp::alloc(rec.size() * 4);
  pmpp::upload(R, rec.data(), rec.size() * 4);
  pmpp::run([&](void* v) {
    encodeProbe(((pmpp::metal::Pass*)v)->encoder, pso, pmpp::metal::buffer(R), (uint32_t)gw, (uint32_t)gh,
                MTLSizeMake(tw, th, 1));
  });
  pmpp::download(R, rec.data(), rec.size() * 4);
  pmpp::release(R);

  size_t silent = 0;
  for (size_t p = 0; p < threads; ++p)
    silent += (rec[3 * p] == 0xFFFFFFFFu && rec[3 * p + 1] == 0xFFFFFFFFu && rec[3 * p + 2] == 0xFFFFFFFFu);
  if (silent) { s.error = std::to_string(silent) + " of " + std::to_string(threads) + " threads wrote no record"; return s; }

  s.width = rec[2];
  for (size_t p = 0; p < threads; ++p)
    if (rec[3 * p + 2] != s.width || s.width == 0) {
      s.error = "threads disagree about the width (" + std::to_string(s.width) + " and " + std::to_string(rec[3 * p + 2]) +
                "), or it is 0";
      return s;
    }

  s.consecutive = true;
  bool sawFull = false;
  for (int ty = 0; ty * th < gh; ++ty) {
    for (int tx = 0; tx * tw < gw; ++tx) {
      const int w = std::min(tw, gw - tx * tw), h = std::min(th, gh - ty * th);   // edge threadgroups are smaller
      const bool full = (w == tw && h == th);
      std::set<std::pair<uint32_t, uint32_t>> seen;
      std::map<uint32_t, uint32_t> members;   // SIMD-group index -> threads in it
      for (int ly = 0; ly < h; ++ly) {
        for (int lx = 0; lx < w; ++lx) {
          const size_t p = (size_t)(ty * th + ly) * gw + (tx * tw + lx);
          const uint32_t group = rec[3 * p], lane = rec[3 * p + 1];
          char where[96];
          std::snprintf(where, sizeof where, "threadgroup (%d, %d), thread (%d, %d)", tx, ty, lx, ly);
          if (lane >= s.width) {
            s.error = std::string(where) + ": lane " + std::to_string(lane) + " is not below the width " + std::to_string(s.width);
            return s;
          }
          if (!seen.insert({group, lane}).second) {
            s.error = std::string(where) + ": a second thread with SIMD-group " + std::to_string(group) + ", lane " +
                      std::to_string(lane);
            return s;
          }
          members[group]++;
          if (full) {
            const uint32_t linear = (uint32_t)(ly * tw + lx);
            if (group != linear / s.width || lane != linear % s.width) s.consecutive = false;
          }
        }
      }
      const uint32_t last = members.rbegin()->first;
      for (auto& gm : members)
        if (gm.first != last && gm.second != s.width) {
          char b[160];
          std::snprintf(b, sizeof b, "threadgroup (%d, %d): SIMD-group %u has %u threads, and it is not the last one (width %u)",
                        tx, ty, gm.first, gm.second, s.width);
          s.error = b;
          return s;
        }
      if (full) { s.groupsPerThreadgroup = (int)members.size(); sawFull = true; }
    }
  }
  if (!sawFull) s.consecutive = false;
  return s;
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  id<MTLDevice> dev = pmpp::metal::device();
  auto lin = linearPipeline();
  auto probe = probePipeline();
  if (!lin || !probe) {
    t.check("kernel.metal builds", [] { return std::string("see the compile error above"); });
    return t.finish();
  }
  if (![dev supportsFamily:MTLGPUFamilyApple4]) {
    t.check("device supports nonuniform threadgroup sizes", [] {
      return std::string("dispatchThreads needs GPU family Apple4 or later; this lab cannot run on this device");
    });
    return t.finish();
  }

  t.check("queryMetal: every field equals the driver's own reading (pipeline: linear_layer)", [&]() -> std::string {
    std::string msg;
    id<MTLComputePipelineState> both[2] = {lin, probe};   // two pipelines: the last two fields must follow the argument
    for (id<MTLComputePipelineState> pso : both) {
      MetalInfo m = queryMetal(dev, pso);
      m.name[sizeof(m.name) - 1] = '\0';
      auto num = [&](const char* field, unsigned long yours, unsigned long mine) {
        if (yours != mine) append(msg, std::string(field) + ": yours " + std::to_string(yours) + ", driver " + std::to_string(mine));
      };
      if (std::string(m.name) != dev.name.UTF8String) append(msg, std::string("name: yours \"") + m.name + "\", driver \"" + dev.name.UTF8String + "\"");
      num("maxThreadsPerThreadgroup[0]", m.maxThreadsPerThreadgroup[0], dev.maxThreadsPerThreadgroup.width);
      num("maxThreadsPerThreadgroup[1]", m.maxThreadsPerThreadgroup[1], dev.maxThreadsPerThreadgroup.height);
      num("maxThreadsPerThreadgroup[2]", m.maxThreadsPerThreadgroup[2], dev.maxThreadsPerThreadgroup.depth);
      num("maxThreadgroupMemoryLength", m.maxThreadgroupMemoryLength, dev.maxThreadgroupMemoryLength);
      num("hasUnifiedMemory", m.hasUnifiedMemory, dev.hasUnifiedMemory);
      num("threadExecutionWidth", m.threadExecutionWidth, pso.threadExecutionWidth);
      num("maxTotalThreadsPerThreadgroup", m.maxTotalThreadsPerThreadgroup, pso.maxTotalThreadsPerThreadgroup);
      if (!msg.empty()) break;
    }
    return msg;
  });

  for (const LayerShape& s : kTestLayers) {
    char name[200];
    std::snprintf(name, sizeof name, "encodeLinear: batch %d, %d inputs, %d outputs  (%s)", s.batch, s.in, s.out, s.why);
    t.check(name, [&]() -> std::string {
      Layer L(s.batch, s.in, s.out, 4100 + s.batch * 31 + s.out);
      MTLSize tg = MTLSizeMake(0, 0, 0);
      runDerived(L, 1, &tg);
      std::string msg = L.check();
      char shape[96];
      std::snprintf(shape, sizeof shape, "threadgroup %lu x %lu x %lu", (unsigned long)tg.width, (unsigned long)tg.height,
                    (unsigned long)tg.depth);
      const NSUInteger n = tg.width * tg.height * tg.depth, width = lin.threadExecutionWidth, most = lin.maxTotalThreadsPerThreadgroup;
      auto rule = [&](const char* r, const std::string& why) { append(msg, std::string(shape) + ": rule \"" + r + "\": " + why); };
      if (n == 0) { append(msg, std::string(shape) + ": encodeLinear returned an empty threadgroup size"); return msg; }
      if (n > most) rule("within maxTotalThreadsPerThreadgroup", std::to_string(n) + " threads, the pipeline allows " + std::to_string(most));
      if (n % width != 0) rule("multiple of threadExecutionWidth", std::to_string(n) + " threads, the width is " + std::to_string(width));
      if (tg.height > (NSUInteger)s.batch) rule("no taller than the rows", std::to_string(tg.height) + " rows of threads for " + std::to_string(s.batch) + " rows of Y");
      if (tg.depth != 1) rule("two-dimensional", "depth must be 1");
      return msg;
    });
  }

  // The probe. Shapes are sized from the probe pipeline's own width and limit.
  const int W = (int)probe.threadExecutionWidth, M = (int)probe.maxTotalThreadsPerThreadgroup;
  struct ProbeCase { int gw, gh, tw, th; std::string why; };
  std::vector<ProbeCase> cases;
  if (3 * W <= M) cases.push_back({2 * W, 6, W, 3, "threadgroups of 3 widths, 2 x 2 of them"});
  if (W % 2 == 0 && 3 * W / 2 <= M) cases.push_back({3 * W, 1, 3 * W / 2, 1, "threadgroups of one and a half widths: the last SIMD-group is partial"});
  if (256 <= M) cases.push_back({40, 24, 16, 16, "16 x 16 threadgroups on a 40 x 24 grid: partial threadgroups on two edges"});
  cases.push_back({2 * W, M / W, W, M / W, "Apple's recipe: width x (limit / width), two threadgroups"});
  for (const ProbeCase& c : cases) {
    char name[240];
    std::snprintf(name, sizeof name, "simd_probe: %d x %d threads in %d x %d threadgroups  (%s)", c.gw, c.gh, c.tw, c.th, c.why.c_str());
    t.check(name, [&]() -> std::string { return runProbe(c.gw, c.gh, c.tw, c.th).error; });
  }
  return t.finish();
}

static int bench(const pmpp::Args& a) {
  id<MTLDevice> dev = pmpp::metal::device();
  auto lin = linearPipeline();
  auto probe = probePipeline();
  if (!lin || !probe) { std::printf("kernel.metal did not build\n"); return 1; }

  // ---- what you queried ----
  MetalInfo m = queryMetal(dev, lin);
  m.name[sizeof(m.name) - 1] = '\0';
  MetalInfo mp = queryMetal(dev, probe);
  std::printf("queried (your queryMetal)\n");
  std::printf("  device    %s\n", m.name);
  std::printf("  device    maxThreadsPerThreadgroup %lu x %lu x %lu, maxThreadgroupMemoryLength %lu, hasUnifiedMemory %d\n",
              m.maxThreadsPerThreadgroup[0], m.maxThreadsPerThreadgroup[1], m.maxThreadsPerThreadgroup[2],
              m.maxThreadgroupMemoryLength, (int)m.hasUnifiedMemory);
  std::printf("  pipeline  linear_layer: threadExecutionWidth %lu, maxTotalThreadsPerThreadgroup %lu\n",
              m.threadExecutionWidth, m.maxTotalThreadsPerThreadgroup);
  std::printf("  pipeline  simd_probe:   threadExecutionWidth %lu, maxTotalThreadsPerThreadgroup %lu\n",
              mp.threadExecutionWidth, mp.maxTotalThreadsPerThreadgroup);
  pmpp::bench_row(a, "queried").s("device", m.name)
      .n("maxThreadsPerThreadgroup_width", (double)m.maxThreadsPerThreadgroup[0])
      .n("maxThreadsPerThreadgroup_height", (double)m.maxThreadsPerThreadgroup[1])
      .n("maxThreadsPerThreadgroup_depth", (double)m.maxThreadsPerThreadgroup[2])
      .n("maxThreadgroupMemoryLength", (double)m.maxThreadgroupMemoryLength).b("hasUnifiedMemory", m.hasUnifiedMemory)
      .n("threadExecutionWidth", (double)m.threadExecutionWidth)
      .n("maxTotalThreadsPerThreadgroup", (double)m.maxTotalThreadsPerThreadgroup)
      .n("probe_threadExecutionWidth", (double)mp.threadExecutionWidth)
      .n("probe_maxTotalThreadsPerThreadgroup", (double)mp.maxTotalThreadsPerThreadgroup).emit();

  // ---- the linear layer of lab 5.3, derived against fixed ----
  const int in = 1024, out = 4096;
  std::printf("linear layer %d -> %d, best of %d runs\n", in, out, a.reps);
  std::printf("  %5s  %-22s %-14s %12s %10s\n", "batch", "configuration", "threadgroup", "time (ms)", "GFLOP/s");
  for (int batch : {1, 64}) {
    Layer L(batch, in, out, 4000 + batch);
    const bool fixedFits = lin.maxTotalThreadsPerThreadgroup >= 256;
    // Both configurations must compute the layer before either is timed.
    MTLSize tg = MTLSizeMake(0, 0, 0);
    runDerived(L, 1, &tg);
    std::string err = L.check();
    if (err.empty()) err = pmpp::take_error();
    if (!err.empty()) { std::printf("  batch %d: your configuration does not compute the layer: %s\n", batch, err.c_str()); return 1; }
    // Repeat the launch inside one compute pass until a run is long enough to time.
    int repeat = 1;
    while (repeat < 1024 && pmpp::time_it(2, [&] { return runDerived(L, repeat, nullptr); }).min_s < 0.02) repeat *= 4;
    // The baseline is checked once, then the two are timed in alternation.
    if (fixedFits) {
      L.prefill();
      runFixed16(L, 1);
      err = L.check();
      if (!err.empty()) { std::printf("  batch %d: the fixed 16 x 16 baseline failed: %s\n", batch, err.c_str()); return 1; }
    }
    double derived, fixed = 0;
    if (fixedFits) {
      auto both = bestOfBoth(a.reps, [&] { return runDerived(L, repeat, nullptr); }, [&] { return runFixed16(L, repeat); });
      derived = both.first / repeat; fixed = both.second / repeat;
    } else {
      derived = pmpp::time_it(a.reps, [&] { return runDerived(L, repeat, nullptr); }).min_s / repeat;
    }
    char shape[48];
    std::snprintf(shape, sizeof shape, "%lu x %lu", (unsigned long)tg.width, (unsigned long)tg.height);
    std::printf("  %5d  %-22s %-14s %12.4f %10.2f\n", batch, "derived (yours)", shape, derived * 1e3, L.flop() / derived / 1e9);
    pmpp::bench_row(a, "linear").s("config", "derived").n("batch", batch).n("in", in).n("out", out)
        .n("block_x", (double)tg.width).n("block_y", (double)tg.height).n("launches_per_run", repeat)
        .n("seconds", derived).n("gflops", L.flop() / derived / 1e9).emit();
    if (!fixedFits) { std::printf("  %5d  fixed 16 x 16 skipped: this pipeline allows fewer than 256 threads per threadgroup\n", batch); continue; }
    std::printf("  %5d  %-22s %-14s %12.4f %10.2f\n", batch, "fixed", "16 x 16", fixed * 1e3, L.flop() / fixed / 1e9);
    std::printf("  %5s  derived is %.2fx the speed of fixed 16 x 16\n", "", fixed / derived);
    pmpp::bench_row(a, "linear").s("config", "fixed16").n("batch", batch).n("in", in).n("out", out)
        .n("block_x", 16).n("block_y", 16).n("launches_per_run", repeat)
        .n("seconds", fixed).n("gflops", L.flop() / fixed / 1e9).emit();
    pmpp::bench_row(a, "speedup").n("batch", batch).n("derived_over_fixed", fixed / derived).emit();
  }
  std::printf("  (dispatchThreads clips a threadgroup at the edge of the grid: at batch 1 a 16 x 16 request runs as 16 x 1)\n");

  // ---- what the probe saw ----
  const int W = (int)probe.threadExecutionWidth, M = (int)probe.maxTotalThreadsPerThreadgroup;
  struct Shape { int tw, th; };
  std::vector<Shape> shapes = {{W, M / W}, {M, 1}};
  if (256 <= M) shapes.push_back({16, 16});
  std::printf("SIMD-groups as your probe reported them (pipeline simd_probe: width %d, limit %d)\n", W, M);
  std::printf("  %-12s %16s %24s %s\n", "threadgroup", "width in kernel", "SIMD-groups / threadgroup", "consecutive runs of the linear index");
  for (const Shape& sh : shapes) {
    ProbeSummary s = runProbe(2 * sh.tw, 2 * sh.th, sh.tw, sh.th);
    char shape[48];
    std::snprintf(shape, sizeof shape, "%d x %d", sh.tw, sh.th);
    if (!s.error.empty()) { std::printf("  %-12s probe failed: %s\n", shape, s.error.c_str()); continue; }
    std::printf("  %-12s %16u %24d %s\n", shape, s.width, s.groupsPerThreadgroup, s.consecutive ? "yes" : "no");
    pmpp::bench_row(a, "probe").n("threadgroup_x", sh.tw).n("threadgroup_y", sh.th).n("width_in_kernel", s.width)
        .n("threadExecutionWidth", W).n("simdgroups_per_threadgroup", s.groupsPerThreadgroup)
        .b("consecutive_runs", s.consecutive).emit();
  }
  std::printf("  (linear index = y * threadgroup width + x. \"yes\" is what this GPU did in these dispatches,\n"
              "   not something Metal's specification promises.)\n");
  return 0;
}

#else
// =====================================================================================
// CUDA, HIP and the CPU emulator
// =====================================================================================
int queryDevices(DeviceInfo* info, int maxDevices);
bool chooseLaunchShape(const DeviceInfo& d, int rows, int cols, LaunchShape* shape);   // given: calls your chooseLaunch
void launchLinear(const float* X, const float* W, float* Y, int batch, int in, int out);
void launchLinearFixed(const float* X, const float* W, float* Y, int batch, int in, int out, int blockX, int blockY);
int lastLaunch(LaunchShape* shape);

static float* f(Buf* b) { return (float*)pmpp::raw(b); }
static pmpp::Timing runDerived(Layer& L, int repeat) {
  return pmpp::run([&](void*) { for (int r = 0; r < repeat; ++r) launchLinear(f(L.X), f(L.W), f(L.Y), L.batch, L.in, L.out); });
}
static pmpp::Timing runFixed16(Layer& L, int repeat) {
  return pmpp::run([&](void*) { for (int r = 0; r < repeat; ++r) launchLinearFixed(f(L.X), f(L.W), f(L.Y), L.batch, L.in, L.out, 16, 16); });
}

// Device 0 as the harness reads it, independently of your queryDevices.
static DeviceInfo harnessDevice() {
  DeviceInfo d;
  std::memset(&d, 0, sizeof d);
  auto n = [](const char* key) { return (int)pmpp::device_prop(key, 0); };
  for (auto& kv : pmpp::device_props())
    if (kv.first == "name") std::snprintf(d.name, sizeof d.name, "%s", kv.second.c_str());
  d.major = n("computeCapability.major");
  d.minor = n("computeCapability.minor");
  d.warpSize = n("warpSize");
  d.multiProcessorCount = n("multiProcessorCount");
  d.maxThreadsPerBlock = n("maxThreadsPerBlock");
  d.maxThreadsDim[0] = n("maxThreadsDim.x"); d.maxThreadsDim[1] = n("maxThreadsDim.y"); d.maxThreadsDim[2] = n("maxThreadsDim.z");
  d.maxGridSize[0] = n("maxGridSize.x"); d.maxGridSize[1] = n("maxGridSize.y"); d.maxGridSize[2] = n("maxGridSize.z");
  d.maxThreadsPerMultiProcessor = n("maxThreadsPerMultiProcessor");
  d.maxBlocksPerMultiProcessor = n("maxBlocksPerMultiProcessor");
  d.regsPerMultiprocessor = n("regsPerMultiprocessor");
  d.regsPerBlock = n("regsPerBlock");
  d.sharedMemPerBlock = (size_t)pmpp::device_prop("sharedMemPerBlock", 0);
  return d;
}

static std::string shapeText(const LaunchShape& s) {
  char b[120];
  std::snprintf(b, sizeof b, "block %u x %u x %u, grid %u x %u x %u", s.block[0], s.block[1], s.block[2], s.grid[0], s.grid[1], s.grid[2]);
  return b;
}

// Checks one configuration against every rule, each on its own. Returns the
// rules that failed, by name, or an empty string.
static std::string brokenRules(const DeviceInfo& d, int rows, int cols, const LaunchShape& s) {
  const long long bx = s.block[0], by = s.block[1], bz = s.block[2], gx = s.grid[0], gy = s.grid[1], gz = s.grid[2];
  const long long threads = bx * by * bz;
  if (threads == 0 || gx * gy * gz == 0) return "a dimension of the block or the grid is 0";
  std::string msg;
  auto rule = [&](const char* name, const std::string& why) { append(msg, std::string("rule \"") + name + "\": " + why); };
  auto num = [](long long v) { return std::to_string(v); };
  if (threads > d.maxThreadsPerBlock)
    rule("within maxThreadsPerBlock", num(threads) + " threads, the device allows " + num(d.maxThreadsPerBlock));
  if (bx > d.maxThreadsDim[0] || by > d.maxThreadsDim[1] || bz > d.maxThreadsDim[2])
    rule("within maxThreadsDim", "the device allows " + num(d.maxThreadsDim[0]) + " x " + num(d.maxThreadsDim[1]) + " x " + num(d.maxThreadsDim[2]));
  if (d.warpSize > 0 && threads % d.warpSize != 0)
    rule("whole number of warps", num(threads) + " threads, the width is " + num(d.warpSize));
  if (d.maxThreadsPerMultiProcessor > 0) {
    if (d.maxThreadsPerMultiProcessor % threads != 0)
      rule("divides the thread slots", num(threads) + " threads per block leave " + num(d.maxThreadsPerMultiProcessor % threads) +
           " of " + num(d.maxThreadsPerMultiProcessor) + " thread slots empty");
    if (d.maxBlocksPerMultiProcessor > 0 && threads * d.maxBlocksPerMultiProcessor < d.maxThreadsPerMultiProcessor)
      rule("block slots do not run out first", num(d.maxBlocksPerMultiProcessor) + " block slots x " + num(threads) +
           " threads fill " + num(threads * d.maxBlocksPerMultiProcessor) + " of " + num(d.maxThreadsPerMultiProcessor) + " thread slots");
  }
  if (by > rows) rule("no taller than the rows", "block.y is " + num(by) + " for " + num(rows) + " rows");
  if (bz != 1 || gz != 1) rule("two-dimensional", "block.z and grid.z must be 1: the kernel ignores z, so more would repeat the work");
  if (gx * bx < cols || gy * by < rows)
    rule("grid covers the problem", "it covers " + num(gy * by) + " x " + num(gx * bx) + " of " + num(rows) + " x " + num(cols));
  else if ((gx - 1) * bx >= cols || (gy - 1) * by >= rows)
    rule("no block to spare", "a smaller grid already covers " + num(rows) + " x " + num(cols));
  if (gx > d.maxGridSize[0] || gy > d.maxGridSize[1] || gz > d.maxGridSize[2])
    rule("within maxGridSize", "the device allows " + num(d.maxGridSize[0]) + " x " + num(d.maxGridSize[1]) + " x " + num(d.maxGridSize[2]));
  return msg.empty() ? msg : shapeText(s) + ": " + msg;
}

// Descriptions of devices nobody has to own. The numbers are chosen to
// exercise one rule each; none describes a product.
struct Synthetic { const char* label; DeviceInfo d; bool impossible; };
static DeviceInfo synthetic(int warp, int perBlock, int dimX, int dimY, int dimZ, int threadSlots, int blockSlots,
                            int gridX, int gridY, int gridZ) {
  DeviceInfo d;
  std::memset(&d, 0, sizeof d);
  std::snprintf(d.name, sizeof d.name, "synthetic");
  d.warpSize = warp; d.multiProcessorCount = 1; d.maxThreadsPerBlock = perBlock;
  d.maxThreadsDim[0] = dimX; d.maxThreadsDim[1] = dimY; d.maxThreadsDim[2] = dimZ;
  d.maxThreadsPerMultiProcessor = threadSlots; d.maxBlocksPerMultiProcessor = blockSlots;
  d.maxGridSize[0] = gridX; d.maxGridSize[1] = gridY; d.maxGridSize[2] = gridZ;
  return d;
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const DeviceInfo real = harnessDevice();

  // ---- part 1 ----
  t.check("queryDevices: every field of device 0 equals what the harness reads", [&]() -> std::string {
    DeviceInfo info[8];
    std::memset(info, 0x5A, sizeof info);   // a field you do not fill shows up as a wrong value
    const int n = queryDevices(info, 8);
    if (n < 1 || n > 8) return "queryDevices returned " + std::to_string(n) + "; the harness sees a device";
    DeviceInfo& y = info[0];
    y.name[sizeof(y.name) - 1] = '\0';
    std::string msg;
    auto num = [&](const char* field, long long yours, long long mine) {
      if (yours != mine) append(msg, std::string(field) + ": yours " + std::to_string(yours) + ", harness " + std::to_string(mine));
    };
    if (std::string(y.name) != real.name) append(msg, "name: yours and the harness's differ");
    num("major", y.major, real.major);
    num("minor", y.minor, real.minor);
    num("warpSize", y.warpSize, real.warpSize);
    num("multiProcessorCount", y.multiProcessorCount, real.multiProcessorCount);
    num("maxThreadsPerBlock", y.maxThreadsPerBlock, real.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i) {
      const std::string axis = std::string("[") + std::to_string(i) + "]";
      num(("maxThreadsDim" + axis).c_str(), y.maxThreadsDim[i], real.maxThreadsDim[i]);
      num(("maxGridSize" + axis).c_str(), y.maxGridSize[i], real.maxGridSize[i]);
    }
    num("maxThreadsPerMultiProcessor", y.maxThreadsPerMultiProcessor, real.maxThreadsPerMultiProcessor);
    num("maxBlocksPerMultiProcessor", y.maxBlocksPerMultiProcessor, real.maxBlocksPerMultiProcessor);
    num("regsPerMultiprocessor", y.regsPerMultiprocessor, real.regsPerMultiprocessor);
    num("regsPerBlock", y.regsPerBlock, real.regsPerBlock);
    num("sharedMemPerBlock", (long long)y.sharedMemPerBlock, (long long)real.sharedMemPerBlock);
    if (!msg.empty()) return msg;
    DeviceInfo one[2];
    std::memset(one, 0x5A, sizeof one);
    if (queryDevices(one, 1) != 1) return "asked for at most 1 device, queryDevices did not return 1";
    if (one[1].warpSize != 0x5A5A5A5A) return "asked for at most 1 device, queryDevices wrote a second entry";
    return "";
  });

  // ---- part 2 ----
  const int kMaxX = 2147483647, kMaxYZ = 65535;   // large enough never to bind
  const std::vector<Synthetic> devices = {
      {"width 32, 1024 threads per block, 2048 thread slots, 32 block slots (the book's example limits)",
       synthetic(32, 1024, 1024, 1024, 64, 2048, 32, kMaxX, kMaxYZ, kMaxYZ), false},
      {"width 32, 1024 threads per block, 1536 thread slots, 16 block slots",
       synthetic(32, 1024, 1024, 1024, 64, 1536, 16, kMaxX, kMaxYZ, kMaxYZ), false},
      {"width 64, 256 threads per block, 2048 thread slots, 32 block slots",
       synthetic(64, 256, 256, 256, 256, 2048, 32, kMaxX, kMaxYZ, kMaxYZ), false},
      {"width 32, maxThreadsDim 128 x 128 x 64 under a 1024-thread limit, 2048 thread slots, 32 block slots",
       synthetic(32, 1024, 128, 128, 64, 2048, 32, kMaxX, kMaxYZ, kMaxYZ), false},
      {"width 32, 1024 threads per block, thread slots and block slots not reported (0)",
       synthetic(32, 1024, 1024, 1024, 64, 0, 0, kMaxX, kMaxYZ, kMaxYZ), false},
      {"width 32, 1024 threads per block, 1536 thread slots, block slots not reported (0)",
       synthetic(32, 1024, 1024, 1024, 64, 1536, 0, kMaxX, kMaxYZ, kMaxYZ), false},
      {"width 32, 1024 threads per block, grid limit 2 x 2 x 1: no configuration exists",
       synthetic(32, 1024, 1024, 1024, 64, 2048, 32, 2, 2, 1), true},
  };
  struct Problem { int rows, cols; };
  const Problem problems[] = {{1, 4096}, {3, 40}, {64, 4096}, {1000, 1000}};
  for (const Synthetic& dev : devices) {
    t.check(std::string("chooseLaunch: ") + dev.label, [&]() -> std::string {
      std::string msg;
      int refused = 0, asked = 0;
      for (const Problem& p : problems) {
        if (dev.impossible && p.rows == 3) continue;   // 3 x 40 does fit in a 2 x 2 grid; the other three do not
        ++asked;
        char what[48];
        std::snprintf(what, sizeof what, "%d x %d: ", p.rows, p.cols);
        LaunchShape s;
        std::memset(&s, 0, sizeof s);
        const bool ok = chooseLaunchShape(dev.d, p.rows, p.cols, &s);
        if (dev.impossible) {
          if (ok) append(msg, std::string(what) + "returned true with " + shapeText(s) + "; no block within the limits covers it with a 2 x 2 grid");
          continue;
        }
        if (!ok) { ++refused; append(msg, std::string(what) + "returned false, but a configuration that satisfies every rule exists"); continue; }
        const std::string broken = brokenRules(dev.d, p.rows, p.cols, s);
        if (!broken.empty()) append(msg, what + broken);
      }
      if (!dev.impossible && refused == asked) return "returned false for every problem, but configurations that satisfy every rule exist";
      return msg;
    });
  }

  // ---- part 3 ----
  for (const LayerShape& s : kTestLayers) {
    char name[200];
    std::snprintf(name, sizeof name, "launchLinear: batch %d, %d inputs, %d outputs  (%s)", s.batch, s.in, s.out, s.why);
    t.check(name, [&]() -> std::string {
      Layer L(s.batch, s.in, s.out, 4100 + s.batch * 31 + s.out);
      LaunchShape used;
      const int before = lastLaunch(&used);
      runDerived(L, 1);
      std::string msg = L.check();
      if (lastLaunch(&used) == before) { append(msg, "launchLinear did not launch through launchLinearWith"); return msg; }
      const std::string broken = brokenRules(real, s.batch, s.out, used);
      if (!broken.empty()) append(msg, "on this device, " + broken);
      return msg;
    });
  }
  return t.finish();
}

static int bench(const pmpp::Args& a) {
  // ---- what you queried ----
  DeviceInfo info[8];
  std::memset(info, 0, sizeof info);
  const int n = queryDevices(info, 8);
  if (n < 1) { std::printf("queryDevices returned %d devices\n", n); return 1; }
  std::printf("queried (your queryDevices): %d device%s\n", n, n == 1 ? "" : "s");
  for (int i = 0; i < n && i < 8; ++i) {
    DeviceInfo& d = info[i];
    d.name[sizeof(d.name) - 1] = '\0';
    std::printf("  device %d  %s\n", i, d.name);
    std::printf("    compute capability %d.%d, warpSize %d, multiProcessorCount %d\n", d.major, d.minor, d.warpSize, d.multiProcessorCount);
    std::printf("    maxThreadsPerBlock %d, maxThreadsDim %d x %d x %d, maxGridSize %d x %d x %d\n", d.maxThreadsPerBlock,
                d.maxThreadsDim[0], d.maxThreadsDim[1], d.maxThreadsDim[2], d.maxGridSize[0], d.maxGridSize[1], d.maxGridSize[2]);
    std::printf("    per SM: maxThreadsPerMultiProcessor %d, maxBlocksPerMultiProcessor %d, regsPerMultiprocessor %d\n",
                d.maxThreadsPerMultiProcessor, d.maxBlocksPerMultiProcessor, d.regsPerMultiprocessor);
    std::printf("    per block: regsPerBlock %d, sharedMemPerBlock %zu\n", d.regsPerBlock, d.sharedMemPerBlock);
    pmpp::bench_row(a, "queried").n("device", i).s("device_name", d.name).n("major", d.major).n("minor", d.minor)
        .n("warpSize", d.warpSize).n("multiProcessorCount", d.multiProcessorCount).n("maxThreadsPerBlock", d.maxThreadsPerBlock)
        .n("maxThreadsDim_x", d.maxThreadsDim[0]).n("maxThreadsDim_y", d.maxThreadsDim[1]).n("maxThreadsDim_z", d.maxThreadsDim[2])
        .n("maxGridSize_x", d.maxGridSize[0]).n("maxGridSize_y", d.maxGridSize[1]).n("maxGridSize_z", d.maxGridSize[2])
        .n("maxThreadsPerMultiProcessor", d.maxThreadsPerMultiProcessor).n("maxBlocksPerMultiProcessor", d.maxBlocksPerMultiProcessor)
        .n("regsPerMultiprocessor", d.regsPerMultiprocessor).n("regsPerBlock", d.regsPerBlock)
        .n("sharedMemPerBlock", (double)d.sharedMemPerBlock).emit();
  }

  // ---- the linear layer of lab 5.3, derived against fixed ----
  const int in = 1024, out = 4096;
  const bool fixedFits = pmpp::device_prop("maxThreadsPerBlock", 1024) >= 256;
  std::printf("linear layer %d -> %d, best of %d runs\n", in, out, a.reps);
  std::printf("  %5s  %-22s %-12s %-16s %12s %10s\n", "batch", "configuration", "block", "grid", "time (ms)", "GFLOP/s");
  for (int batch : {1, 64}) {
    Layer L(batch, in, out, 4000 + batch);
    // Both configurations must compute the layer before either is timed.
    LaunchShape used;
    const int before = lastLaunch(&used);
    runDerived(L, 1);
    std::string err = L.check();
    if (err.empty()) err = pmpp::take_error();
    if (err.empty() && lastLaunch(&used) == before) err = "launchLinear launched nothing";
    if (!err.empty()) { std::printf("  batch %d: your configuration does not compute the layer: %s\n", batch, err.c_str()); return 1; }
    // Repeat the launch between two synchronisations until a run is long enough to time.
    int repeat = 1;
    while (repeat < 1024 && pmpp::time_it(2, [&] { return runDerived(L, repeat); }).min_s < 0.02) repeat *= 4;
    // The baseline is checked once, then the two are timed in alternation.
    LaunchShape fixedShape = used;
    if (fixedFits) {
      L.prefill();
      runFixed16(L, 1);
      err = L.check();
      if (err.empty()) err = pmpp::take_error();
      if (!err.empty()) { std::printf("  batch %d: the fixed 16 x 16 baseline failed: %s\n", batch, err.c_str()); return 1; }
      lastLaunch(&fixedShape);
    }
    double derived, fixed = 0;
    if (fixedFits) {
      auto both = bestOfBoth(a.reps, [&] { return runDerived(L, repeat); }, [&] { return runFixed16(L, repeat); });
      derived = both.first / repeat; fixed = both.second / repeat;
    } else {
      derived = pmpp::time_it(a.reps, [&] { return runDerived(L, repeat); }).min_s / repeat;
    }
    char block[40], grid[40];
    std::snprintf(block, sizeof block, "%u x %u", used.block[0], used.block[1]);
    std::snprintf(grid, sizeof grid, "%u x %u", used.grid[0], used.grid[1]);
    std::printf("  %5d  %-22s %-12s %-16s %12.4f %10.2f\n", batch, "derived (yours)", block, grid, derived * 1e3, L.flop() / derived / 1e9);
    pmpp::bench_row(a, "linear").s("config", "derived").n("batch", batch).n("in", in).n("out", out)
        .n("block_x", used.block[0]).n("block_y", used.block[1]).n("grid_x", used.grid[0]).n("grid_y", used.grid[1])
        .n("launches_per_run", repeat).n("seconds", derived).n("gflops", L.flop() / derived / 1e9).emit();
    if (!fixedFits) { std::printf("  %5d  fixed 16 x 16 skipped: this device allows fewer than 256 threads per block\n", batch); continue; }
    std::snprintf(grid, sizeof grid, "%u x %u", fixedShape.grid[0], fixedShape.grid[1]);
    std::printf("  %5d  %-22s %-12s %-16s %12.4f %10.2f\n", batch, "fixed", "16 x 16", grid, fixed * 1e3, L.flop() / fixed / 1e9);
    std::printf("  %5s  derived is %.2fx the speed of fixed 16 x 16\n", "", fixed / derived);
    pmpp::bench_row(a, "linear").s("config", "fixed16").n("batch", batch).n("in", in).n("out", out)
        .n("block_x", 16).n("block_y", 16).n("grid_x", fixedShape.grid[0]).n("grid_y", fixedShape.grid[1])
        .n("launches_per_run", repeat).n("seconds", fixed).n("gflops", L.flop() / fixed / 1e9).emit();
    pmpp::bench_row(a, "speedup").n("batch", batch).n("derived_over_fixed", fixed / derived).emit();
  }
  return 0;
}
#endif

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
