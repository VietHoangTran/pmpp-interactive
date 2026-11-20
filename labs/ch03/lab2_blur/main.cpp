// Lab 3.2 driver: tests and benchmarks the image blur kernel.
// You do not need to edit this file; your work goes in work/.
//
// A picture here is `h` rows by `w` columns, row-major, 1 byte per pixel. Sizes
// are printed the way the book writes them, rows first: "62 by 76" is 62 rows
// of 76 pixels. `r` is the patch radius: the patch is (2r+1) pixels on a side.
#include <chrono>

#include "pmpp/harness.hpp"
using pmpp::Buf;

// ---- entry points implemented in <variant>/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeBlur(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> in, id<MTLBuffer> out,
                int32_t w, int32_t h, int32_t r);
static id<MTLComputePipelineState> pso() { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "blur"); }
static pmpp::Timing runBlur(Buf* in, Buf* out, int w, int h, int r) {
  id<MTLComputePipelineState> p = pso();
  if (!p) return pmpp::Timing{0, 0};
  return pmpp::run([&](void* vp) {
    auto* pass = (pmpp::metal::Pass*)vp;
    encodeBlur(pass->encoder, p, pmpp::metal::buffer(in), pmpp::metal::buffer(out), (int32_t)w, (int32_t)h, (int32_t)r);
  });
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
void launchBlur(const unsigned char* in, unsigned char* out, int w, int h, int r);
static pmpp::Timing runBlur(Buf* in, Buf* out, int w, int h, int r) {
  return pmpp::run([&](void*) {
    launchBlur((const unsigned char*)pmpp::raw(in), (unsigned char*)pmpp::raw(out), w, h, r);
  });
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

// On the emulator every buffer is exactly as large as the picture, so
// AddressSanitizer stops at the first access outside it and names the line.
// A real device has no such check: there the buffers are larger than the
// picture and the extra bytes after the output (the canary rows) must come
// back unchanged.
#if defined(PMPP_BACKEND_EMU) || defined(PMPP_BACKEND_RACE)
static const bool kExactBuffers = true;
#else
static const bool kExactBuffers = false;
#endif

// ---- pictures and the CPU reference ----
// A slow ramp along each row and column plus noise: neighbours differ, and so
// do rows, so a patch that takes one value too many or from the wrong place
// changes the average. Values stay at or below 230, which leaves grey levels
// free for the sentinel.
static std::vector<unsigned char> noisyPicture(int h, int w, uint32_t seed) {
  std::vector<unsigned char> p((size_t)h * w);
  uint32_t x = seed * 2654435761u + 1u;
  for (int row = 0; row < h; ++row)
    for (int col = 0; col < w; ++col) {
      x ^= x << 13; x ^= x >> 17; x ^= x << 5;
      p[(size_t)row * w + col] = (unsigned char)((col * 5 + row * 9 + (x >> 8) % 97) % 231);
    }
  return p;
}

static std::vector<unsigned char> cpuBlur(const unsigned char* in, int w, int h, int r) {
  std::vector<unsigned char> out((size_t)h * w);
  for (int row = 0; row < h; ++row)
    for (int col = 0; col < w; ++col) {
      int sum = 0, count = 0;
      for (int cr = std::max(row - r, 0); cr <= std::min(row + r, h - 1); ++cr)
        for (int cc = std::max(col - r, 0); cc <= std::min(col + r, w - 1); ++cc) { sum += in[(size_t)cr * w + cc]; ++count; }
      out[(size_t)row * w + col] = (unsigned char)((float)sum / count);
    }
  return out;
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

// Blurs the top `h` rows of `picture` (which holds h + hiddenRows rows of `w`)
// on the device. Empty string = correct.
//
// A border pixel is one whose patch hangs over an edge; the rest are interior.
// Wrong border pixels with a correct interior point at the inner test or the
// divisor. A wrong interior points at the mapping from thread to pixel.
static std::string runAndCheck(const std::vector<unsigned char>& picture, int h, int w, int r, int hiddenRows,
                               bool uniform = false) {
  const size_t n = (size_t)h * w;
  const std::vector<unsigned char> ref = cpuBlur(picture.data(), w, h, r);
  const unsigned char sentinel = pickSentinel(ref);

  // Room after the picture for what a wrong launch, a missing range test or a
  // missing clause would touch: generous, so that a mistake lands in memory we own.
  const size_t span = (size_t)std::max(h, w);
  const size_t slack = kExactBuffers ? 0 : span * span + 64 * span + 64 + (size_t)(r + 1) * (span + 1);
  const size_t outBytes = n + slack;
  const size_t inBytes = picture.size() + slack;

  std::vector<unsigned char> hostIn(inBytes, 0), got(outBytes, sentinel);
  std::copy(picture.begin(), picture.end(), hostIn.begin());
  Buf* in = pmpp::alloc(inBytes);
  Buf* out = pmpp::alloc(outBytes);
  pmpp::upload(in, hostIn.data(), inBytes);
  pmpp::upload(out, got.data(), outBytes);
  runBlur(in, out, w, h, r);
  pmpp::download(out, got.data(), outBytes);
  pmpp::release(in);
  pmpp::release(out);

  // Tolerance of one grey level: the quotient is truncated, and a device may
  // compute an exact quotient such as 18/9 a hair below 2.
  size_t wrongBorder = 0, wrongInterior = 0, border = 0, unwritten = 0, first = 0;
  int lo = 255, hi = 0;
  for (int row = 0; row < h; ++row)
    for (int col = 0; col < w; ++col) {
      const size_t i = (size_t)row * w + col;
      const bool isBorder = row < r || row > h - 1 - r || col < r || col > w - 1 - r;
      border += isBorder;
      lo = std::min(lo, (int)got[i]);
      hi = std::max(hi, (int)got[i]);
      const int d = (int)got[i] - (int)ref[i];
      if (d < -1 || d > 1) {
        if (wrongBorder + wrongInterior == 0) first = i;
        (isBorder ? wrongBorder : wrongInterior)++;
        unwritten += got[i] == sentinel;
      }
    }
  size_t stray = 0, firstStray = 0;
  for (size_t i = n; i < outBytes; ++i)
    if (got[i] != sentinel) { if (stray == 0) firstStray = i; ++stray; }
  if (wrongBorder + wrongInterior == 0 && stray == 0) return "";

  char b[240];
  std::string msg;
  if (wrongBorder + wrongInterior) {
    std::snprintf(b, sizeof b, "%zu of %zu border pixels wrong, %zu of %zu interior pixels wrong; first at row %zu, col %zu: got %d, expected %d",
                  wrongBorder, border, wrongInterior, n - border, first / (size_t)w, first % (size_t)w, (int)got[first], (int)ref[first]);
    msg += b;
    if (unwritten) {
      std::snprintf(b, sizeof b, "; %zu of them still hold the sentinel %d (no thread wrote there)", unwritten, (int)sentinel);
      msg += b;
    }
    if (uniform) {
      std::snprintf(b, sizeof b, "; a uniform picture of %d came back with values from %d to %d", (int)picture[0], lo, hi);
      msg += b;
    }
  }
  if (stray) {
    std::snprintf(b, sizeof b, "%s%zu bytes written after the picture's %zu; first at flat offset %zu", msg.empty() ? "" : "; ", stray, n, firstStray);
    msg += b;
  }
  return msg;
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);

  // First, the top 62 rows of a taller buffer. A read that runs past the right
  // edge of a row lands on the first pixels of the next row, which is valid
  // memory: here that is true of every row, so only the values can show it.
  t.check("62 by 76, r = 1, as the top rows of a 66-row buffer (a read past an edge stays in valid memory)", [&] {
    return runAndCheck(noisyPicture(66, 76, 11), 62, 76, 1, 4);
  });

  struct Case { int h, w, r; const char* why; };
  std::vector<Case> cases = {
      {62, 76, 1, "the book's picture, 3 by 3 patch"},
      {62, 76, 2, "5 by 5 patch"},
      {62, 76, 3, "7 by 7 patch"},
      {1, 1, 1, "a single pixel: the patch is almost all outside"},
      {1, 9, 2, "a single row"},
      {9, 1, 2, "a single column"},
      {3, 3, 3, "the patch is larger than the picture"},
      {33, 17, 7, "taller than wide, 15 by 15 patch"},
  };
  if (!kExactBuffers) cases.push_back({1500, 2000, 2, "the book's 3-million-pixel size"});
  for (const Case& c : cases) {
    char name[160];
    std::snprintf(name, sizeof name, "%d by %d, r = %d (%s)", c.h, c.w, c.r, c.why);
    t.check(name, [&] {
      return runAndCheck(noisyPicture(c.h, c.w, 500 + (uint32_t)c.h * 31 + (uint32_t)c.w * 7 + (uint32_t)c.r), c.h, c.w, c.r, 0);
    });
  }

  t.check("uniform picture of 200, 62 by 76, r = 2: must come back uniform", [&] {
    return runAndCheck(std::vector<unsigned char>((size_t)62 * 76, 200), 62, 76, 2, 0, true);
  });
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
  // A square picture costs 2 bytes per pixel on the device (1 in, 1 out). Use
  // at most an eighth of the memory the device reports.
  const double mem = deviceMemory();
  const double budget = mem > 0 ? mem / 8.0 : 256e6;
  auto fits = [&](int side) { return 2.0 * side * (double)side <= budget && side <= 16384; };
  // Grow until one run at r = 1 is long enough to time, or the budget is reached.
  int side = 1024;
  for (;;) {
    Buf* in = pmpp::alloc((size_t)side * side);
    Buf* out = pmpp::alloc((size_t)side * side);
    const double s = timeWarm(2, [&] { return runBlur(in, out, side, side, 1); });
    pmpp::release(in);
    pmpp::release(out);
    std::string err = pmpp::take_error();
    if (!err.empty() || s <= 0) { std::printf("bench failed: %s\n", err.c_str()); return 1; }
    if (s > 0.01 || !fits(side * 2)) break;
    side *= 2;
  }
  const double mpix = (double)side * side / 1e6;
  Buf* in = pmpp::alloc((size_t)side * side);
  Buf* out = pmpp::alloc((size_t)side * side);
  std::printf("picture %d by %d (%.1f megapixels), 16 by 16 blocks, best of %d runs\n", side, side, mpix, a.reps);
  std::printf("  %6s %12s %12s %14s %12s\n", "r", "patch", "time (ms)", "Mpixel/s", "x r = 1");
  double r1_s = 0, r4_s = 0;
  for (int r : {1, 2, 4, 8}) {
    const double s = timeWarm(a.reps, [&] { return runBlur(in, out, side, side, r); });
    std::string err = pmpp::take_error();
    if (!err.empty() || s <= 0) { std::printf("  r = %d failed: %s\n", r, err.c_str()); continue; }
    if (r == 1) r1_s = s;
    if (r == 4) r4_s = s;
    char patch[32], rel[32] = "";
    std::snprintf(patch, sizeof patch, "%d by %d", 2 * r + 1, 2 * r + 1);
    if (r1_s > 0) std::snprintf(rel, sizeof rel, "%.2f", s / r1_s);
    std::printf("  %6d %12s %12.4f %14.1f %12s\n", r, patch, s * 1e3, mpix / s, rel);
    pmpp::bench_row(a, "blur").n("radius", r).n("width", side).n("height", side).n("seconds", s).n("mpix_per_s", mpix / s).emit();
  }
  pmpp::release(in);
  pmpp::release(out);
  if (r1_s > 0 && r4_s > 0)
    std::printf("measured: r = 4 (81 values per thread) took %.2f times as long as r = 1 (9 values per thread)\n", r4_s / r1_s);
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
