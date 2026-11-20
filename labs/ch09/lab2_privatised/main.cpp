// Lab 9.2 driver: tests your two privatised histograms, and the two provided comparison
// kernels, against the sequential loop; on a GPU it also times all four.
// Your work goes in work/. You do not need to edit this file.
//
// Every case compares all 256 counts with the sequential loop below (book Fig. 9.2,
// p. 203). Counts are whole numbers, so the comparison is exact: the tolerance is zero.
// The driver zeroes bins before every launch. On a GPU each configuration is launched 5
// times, because a race can spare one launch and not the next.
#include "pmpp/harness.hpp"
using pmpp::Buf;

static const unsigned kNumBins = 256;   // one bin per value of an 8-bit pixel

// Shared and Runs are yours; Basic and Contiguous are provided.
enum class Kernel { Shared, Runs, Basic, Contiguous };
static bool provided(Kernel k) { return k == Kernel::Basic || k == Kernel::Contiguous; }

// ---- entry points implemented in <variant>/ and provided/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeHistogramShared(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> image,
                           id<MTLBuffer> bins, uint32_t width, uint32_t height, uint32_t coarseFactor);
void encodeHistogramRuns(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> image,
                         id<MTLBuffer> bins, uint32_t width, uint32_t height, uint32_t coarseFactor);
void encodeRefHistogramBasic(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> image,
                             id<MTLBuffer> bins, uint32_t width, uint32_t height);
void encodeRefHistogramContiguous(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> image,
                                  id<MTLBuffer> bins, uint32_t width, uint32_t height, uint32_t coarseFactor);
using pmpp::metal::buffer;
static const char* kBlock = "threadgroup";
static const char* kernelName(Kernel k) {
  switch (k) {
    case Kernel::Shared: return "histogram_shared";
    case Kernel::Runs: return "histogram_runs";
    case Kernel::Basic: return "ref_histogram_basic";
    case Kernel::Contiguous: return "ref_histogram_contiguous";
  }
  return "?";
}
// provided/ref_kernels.metal, found from this variant's kernel.metal: <lab>/<variant>/kernel.metal.
static std::string providedMetal() {
  std::string s = PMPP_METAL_SOURCE;
  s = s.substr(0, s.rfind('/'));
  s = s.substr(0, s.rfind('/'));
  return s + "/provided/ref_kernels.metal";
}
static pmpp::Timing launch(Kernel k, Buf* image, Buf* bins, unsigned w, unsigned h, unsigned coarse) {
  id<MTLComputePipelineState> p = pmpp::metal::pipeline(provided(k) ? providedMetal() : PMPP_METAL_SOURCE, kernelName(k));
  if (!p) return pmpp::Timing{0, 0};
  return pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    switch (k) {
      case Kernel::Shared: encodeHistogramShared(enc, p, buffer(image), buffer(bins), w, h, coarse); break;
      case Kernel::Runs: encodeHistogramRuns(enc, p, buffer(image), buffer(bins), w, h, coarse); break;
      case Kernel::Basic: encodeRefHistogramBasic(enc, p, buffer(image), buffer(bins), w, h); break;
      case Kernel::Contiguous: encodeRefHistogramContiguous(enc, p, buffer(image), buffer(bins), w, h, coarse); break;
    }
  });
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
void launchHistogramShared(const unsigned char* image, unsigned int* bins, unsigned int width,
                           unsigned int height, unsigned int coarseFactor);
void launchHistogramRuns(const unsigned char* image, unsigned int* bins, unsigned int width,
                         unsigned int height, unsigned int coarseFactor);
void launchRefHistogramBasic(const unsigned char* image, unsigned int* bins, unsigned int width,
                             unsigned int height);
void launchRefHistogramContiguous(const unsigned char* image, unsigned int* bins, unsigned int width,
                                  unsigned int height, unsigned int coarseFactor);
static const char* kBlock = "block";
static const char* kernelName(Kernel k) {
  switch (k) {
    case Kernel::Shared: return "histogramShared";
    case Kernel::Runs: return "histogramRuns";
    case Kernel::Basic: return "refHistogramBasic";
    case Kernel::Contiguous: return "refHistogramContiguous";
  }
  return "?";
}
static pmpp::Timing launch(Kernel k, Buf* image, Buf* bins, unsigned w, unsigned h, unsigned coarse) {
  return pmpp::run([&](void*) {
    const unsigned char* img = (const unsigned char*)pmpp::raw(image);
    unsigned int* b = (unsigned int*)pmpp::raw(bins);
    switch (k) {
      case Kernel::Shared: launchHistogramShared(img, b, w, h, coarse); break;
      case Kernel::Runs: launchHistogramRuns(img, b, w, h, coarse); break;
      case Kernel::Basic: launchRefHistogramBasic(img, b, w, h); break;
      case Kernel::Contiguous: launchRefHistogramContiguous(img, b, w, h, coarse); break;
    }
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
enum class Data { Uniform, Constant, Sky, Stripes };
static const char* dataName(Data d) {
  switch (d) {
    case Data::Uniform: return "uniform bytes";
    case Data::Constant: return "constant (all 200)";
    case Data::Sky: return "sky (each row one value)";
    case Data::Stripes: return "stripes (runs of 8 equal bytes)";
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
    case Data::Sky:   // every row one value, from bright at the top to darker at the bottom
      for (unsigned y = 0; y < h; ++y)
        std::fill(img.begin() + (size_t)y * w, img.begin() + (size_t)(y + 1) * w,
                  (unsigned char)(230 - (unsigned)((uint64_t)y * 160 / h)));
      break;
    case Data::Stripes:   // runs of 8 equal bytes, each run a random value
      for (size_t i = 0; i < img.size(); ++i) {
        if (i % 8 == 0) xorshift(x);
        img[i] = (unsigned char)(x >> 24);
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

struct ImageSize { unsigned w, h; };
static std::string sizeName(const ImageSize& s) {
  const size_t n = (size_t)s.w * s.h;
  return std::to_string(s.w) + " x " + std::to_string(s.h) + " (" + std::to_string(n) + (n == 1 ? " pixel)" : " pixels)");
}

// Runs `k` on images of each size and kind, at each coarsening factor, `launches` times
// per configuration on freshly zeroed bins. "" if every launch gave the right counts.
// A launch error, or on the race backend a ThreadSanitizer report, ends the case at once.
static std::string runCase(Kernel k, const std::vector<ImageSize>& sizes, const std::vector<Data>& datas,
                           const std::vector<unsigned>& coarses, int launches) {
  std::vector<std::string> failures;
  int configs = 0;
  const std::vector<unsigned> zeros(kNumBins, 0u);
  for (const ImageSize& s : sizes) {
    const size_t n = (size_t)s.w * s.h;
    Buf* image = pmpp::alloc(n);
    Buf* bins = pmpp::alloc(kNumBins * sizeof(unsigned));
    std::string fatal;
    for (Data d : datas) {
      const std::vector<unsigned char> img = makeImage(d, s.w, s.h, 700u + s.w * 7u + s.h * 13u + (uint32_t)d);
      std::vector<unsigned> want(kNumBins, 0u);
      histogramSequential(img.data(), want.data(), s.w, s.h);
      pmpp::upload(image, img.data(), n);
      for (unsigned coarse : coarses) {
        ++configs;
        std::string config = std::string(dataName(d));
        if (sizes.size() > 1) config = sizeName(s) + ", " + config;
        if (k != Kernel::Basic) config += ", coarse factor " + std::to_string(coarse);
        int wrong = 0;
        std::string first;
        for (int l = 0; l < launches; ++l) {
          pmpp::upload(bins, zeros.data(), kNumBins * sizeof(unsigned));
          launch(k, image, bins, s.w, s.h, coarse);
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
    if (!fatal.empty()) return fatal;
  }
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
  const int launches = emu ? 1 : 5;
  std::vector<ImageSize> sizes = {{1, 1}, {255, 1}, {256, 1}, {257, 1}, {64, 64}, {100, 100}};
  if (!emu) { sizes.push_back({1024, 1024}); sizes.push_back({4096, 1025}); }
  const std::vector<Data> all = {Data::Uniform, Data::Constant, Data::Sky, Data::Stripes};
  const std::vector<unsigned> coarses = {1, 3, 4, 16};
  const std::string each = launches > 1 ? ", " + std::to_string(launches) + " launches each" : "";

  std::printf("provided comparison kernels (book Figs. 9.6 and 9.12)\n");
  t.check(std::string("provided ") + kernelName(Kernel::Basic) + ": every image size and kind" + each,
          [&]() { return runCase(Kernel::Basic, sizes, all, {1}, launches); });
  t.check(std::string("provided ") + kernelName(Kernel::Contiguous) + ": every image size and kind, coarse factors 1, 3, 4, 16" + each,
          [&]() { return runCase(Kernel::Contiguous, sizes, all, coarses, launches); });

  for (Kernel k : {Kernel::Shared, Kernel::Runs}) {
    const char* label = k == Kernel::Shared ? "shared" : "runs";
    std::printf("your %s kernel: %s (book Fig. %s)\n", label, kernelName(k), k == Kernel::Shared ? "9.14" : "9.15");
    for (const ImageSize& s : sizes) {
      t.check(std::string(label) + ": " + sizeName(s) + ", 4 kinds x 4 coarse factors" + each,
              [&]() { return runCase(k, {s}, all, coarses, launches); });
    }
  }
  return t.finish();
}

// ---- benchmark ----
static int bench(const pmpp::Args& a) {
  // 2^26 pixels, 8192 x 8192, so that a row of the sky image is long. Fewer rows if the
  // image would take more than an eighth of the memory the device reports.
  const unsigned w = 8192;
  unsigned h = 8192;
  const double mem = deviceMemory();
  while (mem > 0 && h > 128 && (double)w * h > mem / 8) h /= 2;
  const size_t n = (size_t)w * h;

  struct Config { Kernel k; unsigned coarse; const char* name; const char* label; };
  const Config configs[] = {
      {Kernel::Basic, 1, "basic", "basic (provided)"},
      {Kernel::Contiguous, 16, "contiguous", "contiguous (provided)"},
      {Kernel::Shared, 1, "shared", "shared"},
      {Kernel::Shared, 4, "shared", "shared"},
      {Kernel::Shared, 16, "shared", "shared"},
      {Kernel::Shared, 64, "shared", "shared"},
      {Kernel::Runs, 16, "runs", "runs"},
  };
  const int nc = (int)(sizeof configs / sizeof configs[0]);
  const Data datas[3] = {Data::Uniform, Data::Constant, Data::Sky};
  const char* dataKey[3] = {"uniform", "constant", "sky"};
  std::vector<std::vector<double>> secs(nc, std::vector<double>(3, 0.0));

  Buf* image = pmpp::alloc(n);
  Buf* bins = pmpp::alloc(kNumBins * sizeof(unsigned));
  const std::vector<unsigned> zeros(kNumBins, 0u);
  bool countsOk = true;
  for (int di = 0; di < 3; ++di) {
    const std::vector<unsigned char> img = makeImage(datas[di], w, h, 11);
    std::vector<unsigned> want(kNumBins, 0u);
    histogramSequential(img.data(), want.data(), w, h);
    pmpp::upload(image, img.data(), n);
    for (int ci = 0; ci < nc; ++ci) {
      const Config& c = configs[ci];
      // bins is zeroed before every run, outside the timed part.
      const pmpp::Stat st = pmpp::time_it(a.reps, [&] {
        pmpp::upload(bins, zeros.data(), kNumBins * sizeof(unsigned));
        return launch(c.k, image, bins, w, h, c.coarse);
      });
      const std::string err = pmpp::take_error();
      if (!err.empty() || st.min_s <= 0) {
        std::printf("%s, coarse %u, on %s failed: %s\n", c.label, c.coarse, dataKey[di], err.c_str());
        continue;
      }
      std::vector<unsigned> got(kNumBins);
      pmpp::download(bins, got.data(), kNumBins * sizeof(unsigned));
      if (!describeCounts(got, want, n).empty()) countsOk = false;
      secs[ci][di] = st.min_s;
      pmpp::bench_row(a, c.name).s("data", dataKey[di]).n("coarse", c.coarse).n("pixels", (double)n)
          .n("seconds", st.min_s).n("gpixels_per_s", (double)n / st.min_s / 1e9).emit();
    }
  }
  pmpp::release(image);
  pmpp::release(bins);

  std::printf("%zu pixels (%u x %u), 256 threads per %s, best of %d runs. Gpixels/s, higher is faster:\n",
              n, w, h, kBlock, a.reps);
  std::printf("  %-22s %6s %10s %10s %10s\n", "kernel", "coarse", "uniform", "constant", "sky");
  for (int ci = 0; ci < nc; ++ci) {
    std::printf("  %-22s %6u", configs[ci].label, configs[ci].coarse);
    for (int di = 0; di < 3; ++di) {
      if (secs[ci][di] > 0) std::printf(" %10.2f", (double)n / secs[ci][di] / 1e9);
      else std::printf(" %10s", "-");
    }
    std::printf("\n");
  }
  // The comparisons the README asks you to predict. Indices into configs[] above.
  const int basic = 0, contiguous = 1, shared16 = 4, runs16 = 6;
  auto ratio = [&](int slow, int fast, int di) {
    return secs[slow][di] > 0 && secs[fast][di] > 0 ? secs[slow][di] / secs[fast][di] : 0.0;
  };
  if (double r = ratio(basic, shared16, 1))
    std::printf("  constant image: shared at coarse 16 ran %.2fx as fast as basic\n", r);
  if (double r = ratio(contiguous, shared16, 0))
    std::printf("  uniform bytes, coarse 16: interleaved (shared) ran %.2fx as fast as contiguous\n", r);
  if (double r = ratio(shared16, runs16, 2))
    std::printf("  sky image, coarse 16: runs ran %.2fx as fast as shared\n", r);
  if (double r = ratio(shared16, runs16, 0))
    std::printf("  uniform bytes, coarse 16: runs ran %.2fx as fast as shared\n", r);
  if (!countsOk)
    std::printf("  WARNING: some timed runs gave wrong counts; fix the tests before reading these times.\n");
  std::printf("These are measurements of this device. Zeroing the bins is not timed.\n");
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
