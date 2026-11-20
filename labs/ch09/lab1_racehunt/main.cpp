// Lab 9.1 driver: tests two 256-bin image histograms that ship with three defects, and
// on a GPU times them on uniform and on constant images.
// Your work goes in work/. You do not need to edit this file.
//
// Every case compares all 256 counts with the sequential loop below (book Fig. 9.2,
// p. 203). Counts are whole numbers, so the comparison is exact: the tolerance is zero.
//
// The basic kernel's cases come first, then the pool kernel's, so each defect shows in
// its own kernel's cases. Each kernel's first case is the most contended input there is,
// a constant image, in 64 blocks of 64 threads. On the race backend ThreadSanitizer
// reports each pair of racing source lines once per run, so that first case is where a
// race shows; blocks of 64 threads are small enough to run several at a time there.
// On a GPU every configuration is launched 20 times on freshly zeroed buffers, because a
// race can spare one launch and not the next; a failure says in how many launches the
// counts were wrong.
#include "pmpp/harness.hpp"
using pmpp::Buf;

static const unsigned kNumBins = 256;   // one bin per value of an 8-bit pixel

enum class Kernel { Basic, Pool };

// ---- entry points implemented in <variant>/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeHistogramBasic(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> image,
                          id<MTLBuffer> bins, uint32_t width, uint32_t height, uint32_t threadsPerBlock);
void encodeHistogramPool(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> image,
                         id<MTLBuffer> bins, id<MTLBuffer> binsPool, uint32_t width, uint32_t height,
                         uint32_t threadsPerBlock);
using pmpp::metal::buffer;
static const char* kBlock = "threadgroup";
static const char* kernelName(Kernel k) { return k == Kernel::Basic ? "histogram_basic" : "histogram_pool"; }
static id<MTLComputePipelineState> pso(Kernel k) { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, kernelName(k)); }
// The largest threadgroup this kernel can run with. It is the pipeline's limit, which can
// be lower than the device's (notes/today/ch05-facts.md C3).
static unsigned maxBlock(Kernel k) {
  id<MTLComputePipelineState> p = pso(k);
  return p ? (unsigned)p.maxTotalThreadsPerThreadgroup : 0;
}
static pmpp::Timing launch(Kernel k, Buf* image, Buf* bins, Buf* pool, unsigned w, unsigned h, unsigned tpb) {
  id<MTLComputePipelineState> p = pso(k);
  if (!p) return pmpp::Timing{0, 0};
  return pmpp::run([&](void* v) {
    auto* pass = (pmpp::metal::Pass*)v;
    if (k == Kernel::Basic) encodeHistogramBasic(pass->encoder, p, buffer(image), buffer(bins), w, h, tpb);
    else encodeHistogramPool(pass->encoder, p, buffer(image), buffer(bins), buffer(pool), w, h, tpb);
  });
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
void launchHistogramBasic(const unsigned char* image, unsigned int* bins, unsigned int width,
                          unsigned int height, unsigned int threadsPerBlock);
void launchHistogramPool(const unsigned char* image, unsigned int* bins, unsigned int* binsPool,
                         unsigned int width, unsigned int height, unsigned int threadsPerBlock);
static const char* kBlock = "block";
static const char* kernelName(Kernel k) { return k == Kernel::Basic ? "histogramBasic" : "histogramPool"; }
static unsigned maxBlock(Kernel) { return (unsigned)pmpp::device_prop("maxThreadsPerBlock", 1024); }
static pmpp::Timing launch(Kernel k, Buf* image, Buf* bins, Buf* pool, unsigned w, unsigned h, unsigned tpb) {
  return pmpp::run([&](void*) {
    const unsigned char* img = (const unsigned char*)pmpp::raw(image);
    unsigned int* b = (unsigned int*)pmpp::raw(bins);
    if (k == Kernel::Basic) launchHistogramBasic(img, b, w, h, tpb);
    else launchHistogramPool(img, b, (unsigned int*)pmpp::raw(pool), w, h, tpb);
  });
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

// ---- the CPU reference: book Fig. 9.2 (p. 203), with const on the image ----
// The caller zeroes bins, as the book's listing assumes.
static void histogramSequential(const unsigned char* image, unsigned int* bins,
                                unsigned int width, unsigned int height) {
  for (unsigned int i = 0; i < width * height; ++i) {
    unsigned char b = image[i];
    ++bins[b];
  }
}

// ---- test images, deterministic on every platform ----
enum class Data { Uniform, Constant, Runs, Gradient };
static const char* dataName(Data d) {
  switch (d) {
    case Data::Uniform: return "uniform bytes";
    case Data::Constant: return "constant (all 200)";
    case Data::Runs: return "0 and 255 in long runs";
    case Data::Gradient: return "smooth gradient";
  }
  return "?";
}
static uint32_t xorshift(uint32_t& x) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; }

static std::vector<unsigned char> makeImage(Data d, unsigned w, unsigned h, uint32_t seed) {
  std::vector<unsigned char> img((size_t)w * h);
  uint32_t x = seed * 2654435761u + 1u;
  switch (d) {
    case Data::Uniform:
      for (unsigned char& p : img) p = (unsigned char)(xorshift(x) >> 24);
      break;
    case Data::Constant:
      std::fill(img.begin(), img.end(), (unsigned char)200);
      break;
    case Data::Runs: {   // runs of 16 to 256 equal pixels, alternating 0 and 255
      unsigned char v = (xorshift(x) & 1u) ? 255 : 0;
      for (size_t i = 0; i < img.size(); v = (unsigned char)(255 - v)) {
        const size_t len = 16 + xorshift(x) % 241u;
        for (size_t k = 0; k < len && i < img.size(); ++k) img[i++] = v;
      }
      break;
    }
    case Data::Gradient:   // 0 in the top-left corner to 255 in the bottom-right one
      for (unsigned y = 0; y < h; ++y)
        for (unsigned c = 0; c < w; ++c) {
          const unsigned gx = w > 1 ? c * 255u / (w - 1) : 0, gy = h > 1 ? y * 255u / (h - 1) : 0;
          img[(size_t)y * w + c] = (unsigned char)((gx + gy) / 2);
        }
      break;
  }
  return img;
}

// "" if the counts match, else a short description of how they differ.
static std::string describeCounts(const std::vector<unsigned>& got, const std::vector<unsigned>& want,
                                  size_t pixels) {
  size_t wrong = 0, first = 0;
  unsigned long long total = 0;
  for (unsigned b = 0; b < kNumBins; ++b) {
    total += got[b];
    if (got[b] != want[b]) { if (wrong++ == 0) first = b; }
  }
  if (wrong == 0) return "";
  char s[200];
  std::snprintf(s, sizeof s, "bin %zu holds %u, expected %u (%zu bin%s wrong; %llu counts for %zu pixel%s)",
                first, got[first], want[first], wrong, wrong == 1 ? "" : "s", total, pixels, pixels == 1 ? "" : "s");
  return s;
}

static size_t blocksFor(size_t pixels, unsigned tpb) { return (pixels + tpb - 1) / tpb; }

// Runs `k` on a w x h image of each kind in `datas` with each block size in `tpbs`,
// `launches` times per configuration on freshly zeroed bins (and pool). "" if every
// launch gave the right counts. A launch error, or on the race backend a
// ThreadSanitizer report, ends the case at once.
static std::string runCase(Kernel k, unsigned w, unsigned h, const std::vector<Data>& datas,
                           const std::vector<unsigned>& tpbs, int launches) {
  const size_t n = (size_t)w * h;
  const unsigned smallest = *std::min_element(tpbs.begin(), tpbs.end());
  const size_t poolCounters = blocksFor(n, smallest) * kNumBins;
  Buf* image = pmpp::alloc(n);
  Buf* bins = pmpp::alloc(kNumBins * sizeof(unsigned));
  Buf* pool = k == Kernel::Pool ? pmpp::alloc(poolCounters * sizeof(unsigned)) : nullptr;
  const std::vector<unsigned> zeros(std::max(poolCounters, (size_t)kNumBins), 0u);
  std::vector<std::string> failures;
  std::string fatal;
  int configs = 0;
  for (Data d : datas) {
    const std::vector<unsigned char> img = makeImage(d, w, h, 900u + w * 7u + h * 13u + (uint32_t)d);
    std::vector<unsigned> want(kNumBins, 0u);
    histogramSequential(img.data(), want.data(), w, h);
    pmpp::upload(image, img.data(), n);
    for (unsigned tpb : tpbs) {
      ++configs;
      const std::string config = std::string(dataName(d)) + ", " + std::to_string(tpb) + " threads per " + kBlock;
      int wrong = 0;
      std::string first;
      for (int l = 0; l < launches && fatal.empty(); ++l) {
        pmpp::upload(bins, zeros.data(), kNumBins * sizeof(unsigned));
        if (pool) pmpp::upload(pool, zeros.data(), blocksFor(n, tpb) * kNumBins * sizeof(unsigned));
        launch(k, image, bins, pool, w, h, tpb);
        const std::string err = pmpp::take_error();
        if (!err.empty()) {
          fatal = config + ": " + (err.find("ThreadSanitizer") != std::string::npos ? err : "launch error: " + err);
          break;
        }
        std::vector<unsigned> got(kNumBins);
        pmpp::download(bins, got.data(), kNumBins * sizeof(unsigned));
        const std::string e = describeCounts(got, want, n);
        if (!e.empty() && wrong++ == 0) first = e;
      }
      if (!fatal.empty()) break;
      if (wrong > 0) {
        char b[80];
        std::snprintf(b, sizeof b, ": wrong in %d of %d launch%s; the first: ", wrong, launches,
                      launches == 1 ? "" : "es");
        failures.push_back(config + b + first);
      }
    }
    if (!fatal.empty()) break;
  }
  pmpp::release(image);
  pmpp::release(bins);
  pmpp::release(pool);
  if (!fatal.empty()) return fatal;
  if (failures.empty()) return "";
  std::string out = std::to_string(failures.size()) + " of " + std::to_string(configs) +
                    (configs == 1 ? " configuration" : " configurations") + " wrong. ";
  for (size_t i = 0; i < failures.size() && i < 2; ++i) out += (i ? " | " : "") + failures[i];
  if (failures.size() > 2) out += " | and " + std::to_string(failures.size() - 2) + " more";
  return out;
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const bool emu = pmpp::is_emu();
  const int launches = emu ? 1 : 20;
  struct Size { unsigned w, h; };
  std::vector<Size> sizes = {{1, 1}, {63, 1}, {64, 1}, {100, 10}, {64, 64}};
  if (!emu) { sizes.push_back({1024, 1024}); sizes.push_back({3000, 1001}); }
  const std::vector<Data> all = {Data::Uniform, Data::Constant, Data::Runs, Data::Gradient};

  for (Kernel k : {Kernel::Basic, Kernel::Pool}) {
    const char* label = k == Kernel::Basic ? "basic" : "pool";
    std::printf("%s kernel: %s (book Fig. %s)\n", label, kernelName(k), k == Kernel::Basic ? "9.6" : "9.9");
    const unsigned most = maxBlock(k);
    if (most < 64) {
      t.check(std::string(label) + ": the kernel builds", [] {
        return std::string("the kernel did not build, or it cannot run 64 threads per group (see the message above)");
      });
      continue;
    }
    // Blocks of 64 and 128 threads, fewer than the 256 bins, so the commit loop must
    // stride. On a GPU also 256 and the largest block this kernel can run with.
    std::vector<unsigned> tpbs = {64, 128};
    if (!emu) {
      for (unsigned v : {256u, most})
        if (v <= most && std::find(tpbs.begin(), tpbs.end(), v) == tpbs.end()) tpbs.push_back(v);
    }
    // First, the most contended input (see the comment at the top).
    t.check(std::string(label) + ": constant image 64 x 64, 64 threads per " + kBlock + " (64 " + kBlock +
                "s), the most contended input",
            [&]() { return runCase(k, 64, 64, {Data::Constant}, {64}, launches); });
    for (const Size& s : sizes) {
      std::string name = std::string(label) + ": " + std::to_string(s.w) + " x " + std::to_string(s.h) + " image (" +
                         std::to_string((size_t)s.w * s.h) + (s.w * s.h == 1 ? " pixel" : " pixels") +
                         "), 4 kinds x " + std::to_string(tpbs.size()) +
                         " " + kBlock + " sizes";
      if (launches > 1) name += ", " + std::to_string(launches) + " launches each";
      t.check(name, [&]() { return runCase(k, s.w, s.h, all, tpbs, launches); });
    }
  }
  return t.finish();
}

// ---- benchmark: basic and pool, uniform bytes and a constant image ----
static int bench(const pmpp::Args& a) {
  const unsigned tpb = 256;
  for (Kernel k : {Kernel::Basic, Kernel::Pool})
    if (maxBlock(k) < tpb) {
      std::printf("%s cannot run %u threads per %s here; no benchmark.\n", kernelName(k), tpb, kBlock);
      return 1;
    }
  // 2^26 pixels. Per pixel that is 1 byte of image and, for the pool, 4 bytes of private
  // counters (256 counters per 256-thread block). Halve until it all fits in a quarter of
  // the memory the device reports.
  size_t n = (size_t)1 << 26;
  const double mem = deviceMemory();
  while (mem > 0 && n > ((size_t)1 << 20) && 5.0 * (double)n > mem / 4) n /= 2;
  const unsigned w = 8192, h = (unsigned)(n / w);
  const size_t poolCounters = blocksFor(n, tpb) * kNumBins;

  Buf* image = pmpp::alloc(n);
  Buf* bins = pmpp::alloc(kNumBins * sizeof(unsigned));
  Buf* pool = pmpp::alloc(poolCounters * sizeof(unsigned));
  const std::vector<unsigned> zeros(poolCounters, 0u);

  const Data datas[2] = {Data::Uniform, Data::Constant};
  const char* dataKey[2] = {"uniform", "constant"};
  double secs[2][2] = {{0, 0}, {0, 0}};   // [kernel][data]
  bool countsOk = true;
  for (int di = 0; di < 2; ++di) {
    const std::vector<unsigned char> img = makeImage(datas[di], w, h, 9);
    std::vector<unsigned> want(kNumBins, 0u);
    histogramSequential(img.data(), want.data(), w, h);
    pmpp::upload(image, img.data(), n);
    for (int ki = 0; ki < 2; ++ki) {
      const Kernel k = ki == 0 ? Kernel::Basic : Kernel::Pool;
      // The buffers are zeroed before every run, outside the timed part.
      const pmpp::Stat st = pmpp::time_it(a.reps, [&] {
        pmpp::upload(bins, zeros.data(), kNumBins * sizeof(unsigned));
        if (k == Kernel::Pool) pmpp::upload(pool, zeros.data(), poolCounters * sizeof(unsigned));
        return launch(k, image, bins, pool, w, h, tpb);
      });
      const std::string err = pmpp::take_error();
      if (!err.empty() || st.min_s <= 0) {
        std::printf("%s on %s failed: %s\n", kernelName(k), dataKey[di], err.c_str());
        continue;
      }
      std::vector<unsigned> got(kNumBins);
      pmpp::download(bins, got.data(), kNumBins * sizeof(unsigned));
      if (!describeCounts(got, want, n).empty()) countsOk = false;
      secs[ki][di] = st.min_s;
      pmpp::bench_row(a, ki == 0 ? "basic" : "pool").s("data", dataKey[di]).n("pixels", (double)n)
          .n("seconds", st.min_s).n("gpixels_per_s", (double)n / st.min_s / 1e9).emit();
    }
  }
  pmpp::release(image);
  pmpp::release(bins);
  pmpp::release(pool);

  std::printf("%zu pixels (%u x %u), %u threads per %s, best of %d runs\n", n, w, h, tpb, kBlock, a.reps);
  std::printf("  %-7s %-9s %12s %12s\n", "kernel", "image", "time (ms)", "Gpixels/s");
  for (int ki = 0; ki < 2; ++ki)
    for (int di = 0; di < 2; ++di)
      if (secs[ki][di] > 0)
        std::printf("  %-7s %-9s %12.3f %12.2f\n", ki == 0 ? "basic" : "pool", dataKey[di], secs[ki][di] * 1e3,
                    (double)n / secs[ki][di] / 1e9);
  for (int ki = 0; ki < 2; ++ki)
    if (secs[ki][0] > 0 && secs[ki][1] > 0)
      std::printf("  %-6s the constant image took %.2fx as long as uniform bytes\n",
                  ki == 0 ? "basic:" : "pool:", secs[ki][1] / secs[ki][0]);
  if (secs[0][1] > 0 && secs[1][1] > 0)
    std::printf("  on the constant image the pool kernel ran %.2fx as fast as the basic one\n",
                secs[0][1] / secs[1][1]);
  if (!countsOk)
    std::printf("  WARNING: some timed runs gave wrong counts; fix the tests before reading these times.\n");
  std::printf("These are measurements of this device. Zeroing the buffers is not timed.\n");
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
