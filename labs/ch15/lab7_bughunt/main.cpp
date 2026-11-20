// Lab 15.7 driver: tests a fast linear layer (lab 15.6's design) that ships with three
// defects. Your work goes in work/.
//   Y[batch x out] = X[batch x in] * W^T,  W stored (out, in) row-major.
// The checks are lab 15.6's: a CPU reference with a tolerance, Y filled with NaN, and a
// guard band after Y that no kernel may write (../shared/linear_test.hpp). On a GPU
// every case is launched ten times, Y back to NaN each time, because a missing barrier
// may fail only now and then there.
#include "../shared/linear_test.hpp"

#include <cmath>
#include <memory>

using lin::Case;
using lin::Shape;

// ---------------------------------------------------------------- the CPU reference
// Y[b][o] = sum_k X[b][k] * W[o][k], accumulated in double; mag = sum_k |X[b][k] * W[o][k]|,
// which scales the tolerance. Lab 15.6's main.cpp has the same function; declared in ../shared/linear_test.hpp.
void lin::cpuLinear(const std::vector<float>& x, const std::vector<float>& w, int batch, int in, int out,
                    std::vector<float>& y, std::vector<float>& mag) {
  y.assign((size_t)batch * out, 0.f);
  mag.assign((size_t)batch * out, 0.f);
  for (int b = 0; b < batch; ++b) {
    const float* xr = x.data() + (size_t)b * in;
    for (int o = 0; o < out; ++o) {
      const float* wr = w.data() + (size_t)o * in;
      double acc = 0, am = 0;
      for (int k = 0; k < in; ++k) { const double p = (double)xr[k] * wr[k]; acc += p; am += std::fabs(p); }
      y[(size_t)b * out + o] = (float)acc;
      mag[(size_t)b * out + o] = (float)am;
    }
  }
}

#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeLinear(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> X, id<MTLBuffer> W,
                  id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out);
using pmpp::metal::buffer;

static id<MTLComputePipelineState> pipelineFor(int width) {
  return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "linear_m3", {{"SIMD_WIDTH", std::to_string(width)}});
}
static std::string problem(int width) {
  id<MTLComputePipelineState> p = pipelineFor(width);
  if (!p) return "linear_m3 did not compile: " + pmpp::take_error();
  if (p.maxTotalThreadsPerThreadgroup < 256)
    return "this pipeline allows " + std::to_string((unsigned long)p.maxTotalThreadsPerThreadgroup) +
           " threads per threadgroup, the kernel needs 256";
  return "";
}
static void runIt(Case& c, int width) {
  id<MTLComputePipelineState> p = pipelineFor(width);
  if (!p) return;
  pmpp::run([&](void* v) {
    encodeLinear(((pmpp::metal::Pass*)v)->encoder, p, buffer(c.X), buffer(c.W), buffer(c.Y), (uint32_t)c.batch,
                 (uint32_t)c.in, (uint32_t)c.out);
  });
}
#else
void launchLinearWidth(const float* X, const float* W, float* Y, int batch, int in, int out, int width);
static std::string problem(int) {
  const int limit = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  return limit < 256 ? "this device allows at most " + std::to_string(limit) + " threads per block" : "";
}
static void runIt(Case& c, int width) {
  pmpp::run([&](void*) {
    launchLinearWidth((const float*)pmpp::raw(c.X), (const float*)pmpp::raw(c.W), (float*)pmpp::raw(c.Y), c.batch,
                      c.in, c.out, width);
  });
}
#endif

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const int launches = pmpp::is_emu() ? 1 : 10;
  for (int width : {32, 64}) {
    const std::string who = "width " + std::to_string(width);
    const std::string why = problem(width);
    if (!why.empty()) {
      t.check(who + ": the kernel runs here", [&] { return why; });
      continue;
    }
    for (const Shape& s : lin::testShapes()) {
      t.check(lin::caseName(who, s), [&]() -> std::string {
        Case c(s);
        std::unique_ptr<ch15::Watchdog> wd;
        if (s.in == 0) wd.reset(new ch15::Watchdog(a, lin::caseName(who, s), ch15::kEmptySumSeconds, lin::kEmptySumWhy));
        int bad = 0;
        std::string first;
        for (int l = 0; l < launches; ++l) {
          c.refill();
          runIt(c, width);
          const std::string err = c.check();
          if (!err.empty() && bad++ == 0) first = err;
        }
        if (!bad) return "";
        if (launches == 1) return first;
        return std::to_string(bad) + " of " + std::to_string(launches) + " launches wrong; the first: " + first;
      });
    }
  }
  return t.finish();
}

static int bench(const pmpp::Args&) { std::printf("this lab has no benchmark.\n"); return 0; }

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
