// Lab 5.1 driver: measures this device's memory bandwidth and arithmetic
// throughput, then tests and times your naive matrix multiplication.
// You do not need to edit this file; your work goes in work/.
#include "pmpp/harness.hpp"
using pmpp::Buf;

#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeVecAdd(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer>, id<MTLBuffer>, id<MTLBuffer>, uint32_t n);
void encodeFmaChain(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer>, id<MTLBuffer>, uint32_t n, uint32_t iters, float a, float b);
void encodeMatmul(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer>, id<MTLBuffer>, id<MTLBuffer>, uint32_t width);
using pmpp::metal::buffer;
static id<MTLComputePipelineState> pso(const char* fn) { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, fn); }
static pmpp::Timing runVecAdd(Buf* A, Buf* B, Buf* C, int n) {
  auto p = pso("vec_add"); if (!p) return {0, 0};
  return pmpp::run([&](void* v) { encodeVecAdd(((pmpp::metal::Pass*)v)->encoder, p, buffer(A), buffer(B), buffer(C), (uint32_t)n); });
}
static pmpp::Timing runFma(Buf* in, Buf* out, int n, int iters, float a, float b) {
  auto p = pso("fma_chain"); if (!p) return {0, 0};
  return pmpp::run([&](void* v) { encodeFmaChain(((pmpp::metal::Pass*)v)->encoder, p, buffer(in), buffer(out), (uint32_t)n, (uint32_t)iters, a, b); });
}
static pmpp::Timing runMatmul(Buf* M, Buf* N, Buf* P, int w) {
  auto p = pso("matmul"); if (!p) return {0, 0};
  return pmpp::run([&](void* v) { encodeMatmul(((pmpp::metal::Pass*)v)->encoder, p, buffer(M), buffer(N), buffer(P), (uint32_t)w); });
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
void launchVecAdd(float* A, float* B, float* C, int n);
void launchFmaChain(float* in, float* out, int n, int iters, float a, float b);
void launchMatmul(float* M, float* N, float* P, int Width);
static float* f(Buf* b) { return (float*)pmpp::raw(b); }
static pmpp::Timing runVecAdd(Buf* A, Buf* B, Buf* C, int n) { return pmpp::run([&](void*) { launchVecAdd(f(A), f(B), f(C), n); }); }
static pmpp::Timing runFma(Buf* in, Buf* out, int n, int iters, float a, float b) { return pmpp::run([&](void*) { launchFmaChain(f(in), f(out), n, iters, a, b); }); }
static pmpp::Timing runMatmul(Buf* M, Buf* N, Buf* P, int w) { return pmpp::run([&](void*) { launchMatmul(f(M), f(N), f(P), w); }); }
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  t.check("provided probe: vector addition, n = 1000", [&]() -> std::string {
    const int n = 1000;
    auto x = pmpp::random_floats(n, 1), y = pmpp::random_floats(n, 2);
    Buf *A = pmpp::alloc(n * 4), *B = pmpp::alloc(n * 4), *C = pmpp::alloc(n * 4);
    pmpp::upload(A, x.data(), n * 4); pmpp::upload(B, y.data(), n * 4);
    runVecAdd(A, B, C, n);
    std::vector<float> got(n), ref(n);
    pmpp::download(C, got.data(), n * 4);
    for (int i = 0; i < n; ++i) ref[i] = x[i] + y[i];
    pmpp::release(A); pmpp::release(B); pmpp::release(C);
    return pmpp::describe(pmpp::compare(got.data(), ref.data(), nullptr, n), n);
  });
  t.check("provided probe: multiply-add chain, n = 600, 50 iterations", [&]() -> std::string {
    const int n = 600, iters = 50; const float ca = 0.999f, cb = 0.01f;
    auto x = pmpp::random_floats(n, 3);
    Buf *I = pmpp::alloc(n * 4), *O = pmpp::alloc(n * 4);
    pmpp::upload(I, x.data(), n * 4);
    runFma(I, O, n, iters, ca, cb);
    std::vector<float> got(n), ref(n);
    pmpp::download(O, got.data(), n * 4);
    for (int i = 0; i < n; ++i) { double v = x[i]; for (int it = 0; it < iters; ++it) v = v * ca + cb; ref[i] = (float)v; }
    pmpp::release(I); pmpp::release(O);
    return pmpp::describe(pmpp::compare(got.data(), ref.data(), nullptr, n, 1e-3), n);
  });
  std::vector<int> widths = pmpp::is_emu() ? std::vector<int>{1, 5, 16, 33, 50} : std::vector<int>{1, 5, 16, 33, 100, 257, 512};
  for (int w : widths) {
    t.check("your naive matmul, width " + std::to_string(w), [&]() -> std::string {
      pmpp::GemmProblem p(w, w, w, 40 + w);
      runMatmul(p.A, p.B, p.C, w);
      return p.check();
    });
  }
  return t.finish();
}

static int bench(const pmpp::Args& a) {
  // --- memory bandwidth: three arrays, sized from what the device reports ---
  double mem = deviceMemory();
  double total = std::min(std::max(mem / 16.0, 96e6), 1.5e9);
  int n = (int)(total / 12.0);
  double bw = 0;
  {
    Buf *A = pmpp::alloc((size_t)n * 4), *B = pmpp::alloc((size_t)n * 4), *C = pmpp::alloc((size_t)n * 4);
    double s = pmpp::time_it(a.reps, [&] { return runVecAdd(A, B, C, n); }).min_s;
    bw = 12.0 * n / s;
    std::printf("memory bandwidth probe   %.0f MB moved in %.3f ms  ->  %.1f GB/s\n", 12.0 * n / 1e6, s * 1e3, bw / 1e9);
    pmpp::bench_row(a, "bandwidth").n("bytes", 12.0 * n).n("seconds", s).n("gb_per_s", bw / 1e9).emit();
    pmpp::release(A); pmpp::release(B); pmpp::release(C);
  }
  // --- arithmetic throughput: grow the iteration count until it is timeable ---
  double flops = 0;
  {
    const int tn = 1 << 20;
    Buf *I = pmpp::alloc((size_t)tn * 4), *O = pmpp::alloc((size_t)tn * 4);
    auto x = pmpp::random_floats(tn, 5);
    pmpp::upload(I, x.data(), (size_t)tn * 4);
    int iters = 1024; double s = 0;
    for (;;) {
      s = pmpp::time_it(a.reps, [&] { return runFma(I, O, tn, iters, 0.999f, 0.01f); }).min_s;
      if (s > 0.05 || iters >= (1 << 20)) break;
      iters *= 4;
    }
    flops = 2.0 * tn * (double)iters / s;
    std::printf("arithmetic probe         %.1f GFLOP in %.3f ms  ->  %.1f GFLOP/s\n", 2.0 * tn * iters / 1e9, s * 1e3, flops / 1e9);
    pmpp::bench_row(a, "compute").n("flop", 2.0 * tn * iters).n("seconds", s).n("gflops", flops / 1e9).emit();
    pmpp::release(I); pmpp::release(O);
  }
  const double threshold = flops / bw;
  std::printf("=> this device's threshold: %.2f FLOP/B (above: compute-bound, below: memory-bound)\n", threshold);
  std::printf("   These are measurements of two simple kernels, not the vendor's peak figures.\n");
  // --- your naive matmul ---
  int w = 512; double s = 0;
  for (;;) {
    pmpp::GemmProblem p(w, w, w, 9, false);
    s = pmpp::time_it(a.reps, [&] { return runMatmul(p.A, p.B, p.C, w); }).min_s;
    if (s > 0.05 || w >= 4096) break;
    w *= 2;
  }
  std::string err = pmpp::take_error();
  if (!err.empty() || s <= 0) { std::printf("naive matmul failed to run: %s\n", err.c_str()); return 1; }
  const double mm = 2.0 * w * (double)w * w / s;
  const double model = bw * 0.25;
  std::printf("naive matmul, width %d   %.3f ms  ->  %.1f GFLOP/s\n", w, s * 1e3, mm / 1e9);
  std::printf("   model ceiling at 0.25 FLOP/B: %.1f GFLOP/s; you measured %.1fx that\n", model / 1e9, mm / model);
  pmpp::bench_row(a, "naive_matmul").n("width", w).n("seconds", s).n("gflops", mm / 1e9).emit();
  pmpp::bench_row(a, "roofline").n("bandwidth_gb_per_s", bw / 1e9).n("compute_gflops", flops / 1e9)
      .n("threshold_flop_per_byte", threshold).n("naive_model_ceiling_gflops", model / 1e9)
      .n("naive_gflops", mm / 1e9).n("naive_width", w).emit();
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
