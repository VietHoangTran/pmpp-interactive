// Lab 3.1 driver: tests and benchmarks the colour-to-grayscale kernel.
// You do not need to edit this file; your work goes in work/.
//
// A picture here is `height` rows by `width` columns, row-major. The input has
// 3 bytes per pixel (r, g, b), the output 1. Sizes are printed the way the
// book writes them, rows first: "62 by 76" is 62 rows of 76 pixels.
#include <chrono>

#include "pmpp/harness.hpp"
using pmpp::Buf;

// ---- entry points implemented in <variant>/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeGrayscale(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> Pin, id<MTLBuffer> Pout,
                     uint32_t width, uint32_t height, uint32_t groupW, uint32_t groupH);
static id<MTLComputePipelineState> pso() { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "grayscale"); }
// Empty if this device accepts the block (threadgroup) shape for this kernel, else why not.
static std::string shapeProblem(int bw, int bh, int /*width*/, int /*height*/) {
  id<MTLComputePipelineState> p = pso();
  if (!p) return "";   // the launch itself reports the compile error
  const MTLSize m = pmpp::metal::device().maxThreadsPerThreadgroup;
  char b[200];
  if ((NSUInteger)bw > m.width || (NSUInteger)bh > m.height) {
    std::snprintf(b, sizeof b, "the device allows threadgroups of at most %lu by %lu", (unsigned long)m.width, (unsigned long)m.height);
    return b;
  }
  if ((NSUInteger)bw * (NSUInteger)bh > p.maxTotalThreadsPerThreadgroup) {
    std::snprintf(b, sizeof b, "this pipeline allows at most %lu threads per threadgroup", (unsigned long)p.maxTotalThreadsPerThreadgroup);
    return b;
  }
  return "";
}
static pmpp::Timing runGray(Buf* in, Buf* out, int width, int height, int bw, int bh) {
  id<MTLComputePipelineState> p = pso();
  if (!p) return pmpp::Timing{0, 0};
  return pmpp::run([&](void* vp) {
    auto* pass = (pmpp::metal::Pass*)vp;
    encodeGrayscale(pass->encoder, p, pmpp::metal::buffer(in), pmpp::metal::buffer(out),
                    (uint32_t)width, (uint32_t)height, (uint32_t)bw, (uint32_t)bh);
  });
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
void launchGrayscale(const unsigned char* Pin, unsigned char* Pout, int width, int height, int blockW, int blockH);
// Empty if this device accepts the block shape and the grid it needs, else why not.
// Limits come from the device's own properties; a property it does not report is not checked.
static std::string shapeProblem(int bw, int bh, int width, int height) {
  const double maxThreads = pmpp::device_prop("maxThreadsPerBlock", -1);
  const double maxX = pmpp::device_prop("maxThreadsDim.x", -1), maxY = pmpp::device_prop("maxThreadsDim.y", -1);
  const double gridX = pmpp::device_prop("maxGridSize.x", -1), gridY = pmpp::device_prop("maxGridSize.y", -1);
  char b[200];
  if ((maxX >= 0 && bw > maxX) || (maxY >= 0 && bh > maxY)) {
    std::snprintf(b, sizeof b, "the device allows blocks of at most %.0f by %.0f", maxX, maxY);
    return b;
  }
  if (maxThreads >= 0 && (double)bw * bh > maxThreads) {
    std::snprintf(b, sizeof b, "the device allows at most %.0f threads per block", maxThreads);
    return b;
  }
  const double gx = (width + bw - 1) / bw, gy = (height + bh - 1) / bh;
  if ((gridX >= 0 && gx > gridX) || (gridY >= 0 && gy > gridY)) {
    std::snprintf(b, sizeof b, "the grid would be %.0f by %.0f blocks; the device allows %.0f by %.0f", gx, gy, gridX, gridY);
    return b;
  }
  return "";
}
static pmpp::Timing runGray(Buf* in, Buf* out, int width, int height, int bw, int bh) {
  return pmpp::run([&](void*) {
    launchGrayscale((const unsigned char*)pmpp::raw(in), (unsigned char*)pmpp::raw(out), width, height, bw, bh);
  });
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

// On the emulator every buffer is exactly as large as the picture, so
// AddressSanitizer stops at the first access outside it and names the line.
// A real device has no such check: there the buffers are larger than the
// picture and the extra bytes after it (the canary rows) must come back unchanged.
#if defined(PMPP_BACKEND_EMU) || defined(PMPP_BACKEND_RACE)
static const bool kExactBuffers = true;
#else
static const bool kExactBuffers = false;
#endif

// ---- pictures and the CPU reference ----
struct Picture {
  int height, width;
  std::vector<unsigned char> rgb;   // 3 bytes per pixel
  std::vector<unsigned char> ref;   // 1 byte per pixel
  size_t pixels() const { return (size_t)height * width; }
};

static unsigned char grayOf(unsigned char r, unsigned char g, unsigned char b) {
  return (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
}

static void finish(Picture& p) {
  p.ref.resize(p.pixels());
  for (size_t i = 0; i < p.pixels(); ++i) p.ref[i] = grayOf(p.rgb[3 * i], p.rgb[3 * i + 1], p.rgb[3 * i + 2]);
}

// Red rises along a row, green down a column, blue is noise: no two rows and no
// two columns look alike, so a wrong stride or a swapped axis shows.
// Channels stay at or below 230, which leaves grey levels free for the sentinel.
static Picture gradientPicture(int height, int width, uint32_t seed) {
  Picture p{height, width, {}, {}};
  p.rgb.resize(p.pixels() * 3);
  uint32_t x = seed * 2654435761u + 1u;
  for (int row = 0; row < height; ++row)
    for (int col = 0; col < width; ++col) {
      x ^= x << 13; x ^= x >> 17; x ^= x << 5;
      unsigned char* px = &p.rgb[((size_t)row * width + col) * 3];
      px[0] = (unsigned char)((col * 7 + (x & 15)) % 231);
      px[1] = (unsigned char)((row * 11 + ((x >> 4) & 15)) % 231);
      px[2] = (unsigned char)((x >> 8) % 231);
    }
  finish(p);
  return p;
}

static Picture flatPicture(int height, int width, unsigned char r, unsigned char g, unsigned char b) {
  Picture p{height, width, {}, {}};
  p.rgb.resize(p.pixels() * 3);
  for (size_t i = 0; i < p.pixels(); ++i) { p.rgb[3 * i] = r; p.rgb[3 * i + 1] = g; p.rgb[3 * i + 2] = b; }
  finish(p);
  return p;
}

// A grey level that no correct pixel is within one level of. The output buffer
// is filled with it before the launch, so a pixel that still holds it afterwards
// was never written.
static unsigned char pickSentinel(const std::vector<unsigned char>& ref) {
  bool used[256] = {false};
  for (unsigned char v : ref) used[v] = true;
  for (int v = 254; v >= 1; --v)
    if (!used[v - 1] && !used[v] && !used[v + 1]) return (unsigned char)v;
  return 254;
}

// Runs the kernel on `p` with blocks of bw x bh threads. Empty string = correct.
static std::string runAndCheck(const Picture& p, int bw, int bh) {
  const size_t n = p.pixels();
  // Room after the picture for what a wrong launch or a missing range test
  // would write: generous, so that a mistake lands in memory we own.
  const size_t span = (size_t)std::max(p.height, p.width);
  const size_t outBytes = kExactBuffers ? n : span * span + 64 * span + 64;
  const size_t inBytes = kExactBuffers ? n * 3 : outBytes * 4;
  const unsigned char sentinel = pickSentinel(p.ref);

  std::vector<unsigned char> hostIn(inBytes, 0), got(outBytes, sentinel);
  std::copy(p.rgb.begin(), p.rgb.end(), hostIn.begin());
  Buf* in = pmpp::alloc(inBytes);
  Buf* out = pmpp::alloc(outBytes);
  pmpp::upload(in, hostIn.data(), inBytes);
  pmpp::upload(out, got.data(), outBytes);
  runGray(in, out, p.width, p.height, bw, bh);
  pmpp::download(out, got.data(), outBytes);
  pmpp::release(in);
  pmpp::release(out);

  // Tolerance of one grey level: the cast truncates, so the order in which a
  // compiler adds the three products can move the byte by one.
  size_t wrong = 0, first = 0, unwritten = 0;
  for (size_t i = 0; i < n; ++i) {
    const int d = (int)got[i] - (int)p.ref[i];
    if (d < -1 || d > 1) {
      if (wrong == 0) first = i;
      ++wrong;
      unwritten += got[i] == sentinel;
    }
  }
  size_t stray = 0, firstStray = 0;
  for (size_t i = n; i < outBytes; ++i)
    if (got[i] != sentinel) { if (stray == 0) firstStray = i; ++stray; }
  if (wrong == 0 && stray == 0) return "";

  char b[200];
  std::string msg;
  std::snprintf(b, sizeof b, "blocks %d by %d: ", bw, bh);
  msg += b;
  if (wrong) {
    std::snprintf(b, sizeof b, "%zu of %zu pixels wrong; first at row %zu, col %zu: got %d, expected %d", wrong, n,
                  first / (size_t)p.width, first % (size_t)p.width, (int)got[first], (int)p.ref[first]);
    msg += b;
    if (unwritten) {
      std::snprintf(b, sizeof b, "; %zu of them still hold the sentinel %d (no thread wrote there)", unwritten, (int)sentinel);
      msg += b;
    }
  }
  if (stray) {
    std::snprintf(b, sizeof b, "%s%zu bytes written after the picture's %zu; first at flat offset %zu", wrong ? "; " : "", stray, n, firstStray);
    msg += b;
  }
  return msg;
}

struct BlockShape { int w, h; };

// First failure over the block shapes, or empty.
static std::string runShapes(const Picture& p, const std::vector<BlockShape>& shapes) {
  for (const BlockShape& s : shapes) {
    const std::string why = shapeProblem(s.w, s.h, p.width, p.height);
    if (!why.empty()) { std::printf("        (blocks %d by %d not run: %s)\n", s.w, s.h, why.c_str()); continue; }
    std::string err = runAndCheck(p, s.w, s.h);
    if (!err.empty()) return err;
  }
  return "";
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::vector<BlockShape> blocks = {{16, 16}, {32, 8}, {8, 32}};

  // Flat colours first: the right answer is one number you can work out by
  // hand. A wrong number here is the arithmetic (the cast, the constants) or
  // which bytes were read. Pixels still at the sentinel are the launch or the
  // offsets. What a flat picture cannot show is a pixel that received another
  // pixel's value; the gradient pictures below do that.
  struct Flat { const char* name; unsigned char r, g, b; };
  for (const Flat& f : {Flat{"pure red", 255, 0, 0}, Flat{"pure green", 0, 255, 0}, Flat{"pure blue", 0, 0, 255},
                        Flat{"white", 255, 255, 255}}) {
    Picture p = flatPicture(62, 76, f.r, f.g, f.b);
    char name[160];
    std::snprintf(name, sizeof name, "%s (%d, %d, %d), 62 by 76: every pixel should be %d",
                  f.name, f.r, f.g, f.b, (int)p.ref[0]);
    t.check(name, [&] { return runShapes(p, {{16, 16}}); });
  }

  // Then pictures in which every row and column differs, in several shapes.
  struct Shape { int height, width; const char* why; };
  std::vector<Shape> shapes = {
      {62, 76, "the book's picture"},
      {16, 16, "exactly one 16 by 16 block"},
      {1, 1, "a single pixel"},
      {1, 300, "a single row"},
      {300, 1, "a single column"},
      {33, 17, "taller than wide, no multiple of anything"},
      {64, 80, "the size of the book's grid: no surplus threads at 16 by 16"},
  };
  if (!kExactBuffers) shapes.push_back({1500, 2000, "the book's 3-million-pixel example"});
  for (const Shape& s : shapes) {
    char name[160];
    std::snprintf(name, sizeof name, "gradient with noise, %d by %d (%s)", s.height, s.width, s.why);
    t.check(name, [&] {
      Picture p = gradientPicture(s.height, s.width, 300 + (uint32_t)s.height * 31 + (uint32_t)s.width);
      return runShapes(p, blocks);
    });
  }
  return t.finish();
}

// Launches `once` untimed until `warm_s` seconds have passed, then times `reps`
// launches and returns the best. The untimed stretch is there so that the first
// timed run is not also the first work the device has done for a while.
static double timeWarm(int reps, const std::function<pmpp::Timing()>& once, double warm_s = 0.3) {
  const auto t0 = std::chrono::steady_clock::now();
  do once(); while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < warm_s);
  return pmpp::time_it(reps, once).min_s;
}

static int bench(const pmpp::Args& a) {
#if defined(PMPP_BACKEND_METAL)
  if (!pso()) { std::printf("bench failed: %s\n", pmpp::take_error().c_str()); return 1; }
#endif
  // A square picture costs 4 bytes per pixel on the device (3 in, 1 out). Use
  // at most an eighth of the memory the device reports, and stay below the
  // largest picture whose byte offsets fit the kernel's int.
  const double mem = deviceMemory();
  const double budget = mem > 0 ? mem / 8.0 : 256e6;
  auto fits = [&](int side) {
    const double px = (double)side * side;
    return 4.0 * px <= budget && 3.0 * px < 2147483647.0;
  };
  // Grow until one run at 16 by 16 is long enough to time, or the budget is reached.
  int side = 1024;
  for (;;) {
    Buf* in = pmpp::alloc((size_t)side * side * 3);
    Buf* out = pmpp::alloc((size_t)side * side);
    const double s = timeWarm(2, [&] { return runGray(in, out, side, side, 16, 16); });
    pmpp::release(in);
    pmpp::release(out);
    std::string err = pmpp::take_error();
    if (!err.empty() || s <= 0) { std::printf("bench failed: %s\n", err.c_str()); return 1; }
    if (s > 0.02 || !fits(side * 2)) break;
    side *= 2;
  }
  const double mpix = (double)side * side / 1e6;
  Buf* in = pmpp::alloc((size_t)side * side * 3);
  Buf* out = pmpp::alloc((size_t)side * side);
  std::printf("picture %d by %d (%.1f megapixels), best of %d runs\n", side, side, mpix, a.reps);
  std::printf("  %-12s %12s %14s\n", "block", "time (ms)", "Mpixel/s");
  double wide_s = 0, tall_s = 0;
  for (const BlockShape& sh : {BlockShape{8, 8}, BlockShape{16, 16}, BlockShape{32, 32}, BlockShape{256, 1}, BlockShape{1, 256}}) {
    char label[32];
    std::snprintf(label, sizeof label, "%d by %d", sh.w, sh.h);
    const std::string why = shapeProblem(sh.w, sh.h, side, side);
    if (!why.empty()) { std::printf("  %-12s not run: %s\n", label, why.c_str()); continue; }
    const double s = timeWarm(a.reps, [&] { return runGray(in, out, side, side, sh.w, sh.h); });
    std::string err = pmpp::take_error();
    if (!err.empty() || s <= 0) { std::printf("  %-12s failed: %s\n", label, err.c_str()); continue; }
    std::printf("  %-12s %12.4f %14.1f\n", label, s * 1e3, mpix / s);
    pmpp::bench_row(a, "grayscale").n("block_w", sh.w).n("block_h", sh.h).n("width", side).n("height", side)
        .n("seconds", s).n("mpix_per_s", mpix / s).emit();
    if (sh.w == 256) wide_s = s;
    if (sh.h == 256) tall_s = s;
  }
  pmpp::release(in);
  pmpp::release(out);
  if (wide_s > 0 && tall_s > 0)
    std::printf("measured: tall (1 by 256) time / wide (256 by 1) time = %.2f\n", tall_s / wide_s);
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
