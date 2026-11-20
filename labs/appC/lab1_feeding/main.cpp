// Lab C.1 driver: tests and benchmarks three ways of feeding one kernel.
// You do not need to edit this file; your work goes in work/.
//
// Every test calls one of your feed functions with a FeedJob whose produce and
// consume callbacks belong to the driver. produce copies the table from a host
// image the driver made once, as fread copies a file that is already in memory;
// consume copies the result out. Both record what they were given. The result
// is compared with a CPU reference, and the provided launcher's record says
// whether the kernel read the memory produce wrote or a copy of it.
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <numeric>

#include "pmpp/harness.hpp"
#include "feeding.h"

#if defined(PMPP_BACKEND_METAL)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <unistd.h>
#include "pmpp/metal.h"
#endif

static std::string fmt(const char* f, ...) {
  char b[512];
  va_list ap;
  va_start(ap, f);
  std::vsnprintf(b, sizeof b, f, ap);
  va_end(ap);
  return b;
}

// ---- the table ----------------------------------------------------------------
// Small integers, so every sum the kernel makes is exact in float, and
// neighbouring rows and columns hold different values, so a wrong index shows.
// Element (r, c) is ((37 r + 11 c) mod 101) - 50.
struct Image {
  int rows = 0, cols = 0;
  std::vector<float> v;
  Image(int r, int c) : rows(r), cols(c), v((size_t)r * c) {
    for (int i = 0; i < rows; ++i) {
      int x = (int)(((long long)i * 37) % 101);
      float* row = v.data() + (size_t)i * cols;
      for (int j = 0; j < cols; ++j) {
        row[j] = (float)(x - 50);
        x += 11;
        if (x >= 101) x -= 101;
      }
    }
  }
};

// `count` row indices from a fixed seed, drawn with replacement, holding the last
// row (where an off-by-one shows) and, from three ids on, a repeated row.
static std::vector<int> sparseIds(int rows, int count, uint32_t seed) {
  std::vector<int> ids((size_t)count);
  uint32_t x = seed * 2654435761u + 1u;
  for (int& id : ids) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;   // xorshift32
    id = (int)(x % (uint32_t)rows);
  }
  ids[(size_t)count / 2] = rows - 1;
  if (count >= 3) ids[(size_t)count - 1] = ids[0];
  return ids;
}
static std::vector<int> denseIds(int rows) {
  std::vector<int> ids((size_t)rows);
  std::iota(ids.begin(), ids.end(), 0);
  return ids;
}

// ---- the callbacks --------------------------------------------------------------
struct Ctx {
  const float* image = nullptr;
  size_t tableBytes = 0, outFloats = 0;
  int produced = 0, consumed = 0;
  const void* producePtr = nullptr;
  bool consumeNull = false;
  std::vector<float> got;   // what consume saw last
};
static void produceTable(float* table, void* p) {
  Ctx* c = (Ctx*)p;
  c->produced++;
  c->producePtr = table;
  if (table) std::memcpy(table, c->image, c->tableBytes);
}
static void consumeResult(const float* out, void* p) {
  Ctx* c = (Ctx*)p;
  c->consumed++;
  if (!out) { c->consumeNull = true; return; }
  std::memcpy(c->got.data(), out, c->outFloats * sizeof(float));
}

// Row r of the result must be passes times row ids[r] of the table.
static std::string checkResult(const std::vector<float>& got, const Image& im, const std::vector<int>& ids, int passes) {
  const size_t n = ids.size() * (size_t)im.cols;
  std::vector<float> ref(n);
  for (size_t r = 0; r < ids.size(); ++r)
    for (int c = 0; c < im.cols; ++c)
      ref[r * im.cols + c] = (float)passes * im.v[(size_t)ids[r] * im.cols + c];
  const pmpp::Cmp cmp = pmpp::compare(got.data(), ref.data(), nullptr, n);
  if (!cmp.ok) {
    const size_t i = cmp.first_bad, r = i / im.cols, c = i % im.cols;
    return fmt("%zu of %zu result values wrong; first at result row %zu (table row %d), column %zu: got %g, expected %g",
               cmp.bad, n, r, ids[r], c, (double)got[i], (double)ref[i]);
  }
  // Within the tolerance is not quite enough here. Every correct value is a sum of
  // whole numbers, so it is a whole number exactly. A tiny fraction is what is left
  // when the result started as leftover bytes instead of zero: garbage that small
  // vanishes when added to most table values, but not where the table holds 0.
  for (size_t i = 0; i < n; ++i)
    if (got[i] != std::nearbyint(got[i]))
      return fmt("result row %zu, column %zu holds %g, which is not a whole number. Every correct value is a sum of "
                 "whole numbers: did the result start at zero?", i / im.cols, i % im.cols, (double)got[i]);
  return "";
}

// ---- the paths ------------------------------------------------------------------
struct Path {
  const char* name;   // the benchmark's `path`
  const char* fn;     // the function you write
  bool inPlace;       // true: the kernel must read the very memory produce wrote
  std::function<int(const FeedJob*)> call;
};

#if defined(PMPP_BACKEND_METAL)
struct MetalSetup {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pipeline = nil;
  std::string error;
};
static MetalSetup& metalSetup() {
  static MetalSetup s;
  static bool done = false;
  if (done) return s;
  done = true;
  s.device = pmpp::metal::device();
  s.queue = [s.device newCommandQueue];
  if (!s.queue) { s.error = "newCommandQueue returned nil"; return s; }
  s.pipeline = pmpp::metal::pipeline(PMPP_METAL_SOURCE, "gather_add");
  if (!s.pipeline) {
    s.error = "kernel.metal did not build into a pipeline: " + pmpp::take_error();
    if (s.error.size() > 600) s.error.resize(600);
  }
  return s;
}
static std::string setupError() { return metalSetup().error; }

typedef int (*MetalFeed)(id<MTLDevice>, id<MTLCommandQueue>, id<MTLComputePipelineState>, const FeedJob*);
// The pool is inside the call on purpose: buffers the call leaves to it are
// released when it drains, and a no-copy buffer's deallocator runs then.
static int callMetal(MetalFeed fn, const FeedJob* job) {
  MetalSetup& m = metalSetup();
  int rc;
  @autoreleasepool {
    rc = fn(m.device, m.queue, m.pipeline, job);
  }
  return rc;
}
static std::vector<Path> paths() {
  return {{"private", "feedPrivateCopy", false, [](const FeedJob* j) { return callMetal(feedPrivateCopy, j); }},
          {"nocopy", "feedNoCopy", true, [](const FeedJob* j) { return callMetal(feedNoCopy, j); }},
          {"shared", "feedShared", true, [](const FeedJob* j) { return callMetal(feedShared, j); }}};
}
// The smaller of the largest buffer and the recommended working set.
static double deviceMemory() {
  const double a = pmpp::device_prop("maxBufferLength", 0), b = pmpp::device_prop("recommendedMaxWorkingSetSize", 0);
  if (a <= 0) return b;
  if (b <= 0) return a;
  return std::min(a, b);
}
static const char* memoryName() { return "the smaller of maxBufferLength and recommendedMaxWorkingSetSize"; }
static void idleDevice() {}
#else
static std::string setupError() { return ""; }
static std::vector<Path> paths() {
  return {{"explicit", "feedExplicit", false, feedExplicit},
          {"mapped", "feedMapped", true, feedMapped},
          {"managed", "feedManaged", true, feedManaged}};
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
static const char* memoryName() { return "totalGlobalMem"; }
// Waits for the device and clears the runtime's error state between timed calls.
static void idleDevice() {
  pmpp::run([](void*) {});
  pmpp::take_error();
}
#endif

// ---- tests ----------------------------------------------------------------------
static std::string runCase(const Path& path, const Image& im, const std::vector<int>& ids, int passes) {
  Ctx c;
  c.image = im.v.data();
  c.tableBytes = im.v.size() * sizeof(float);
  c.outFloats = ids.size() * (size_t)im.cols;
  c.got.assign(c.outFloats, std::numeric_limits<float>::quiet_NaN());
  FeedJob job{im.rows, im.cols, ids.data(), (int)ids.size(), passes, produceTable, consumeResult, &c};

  feedResetRecord();
  const int rc = path.call(&job);
  const FeedRecord rec = feedRecord();

  if (rc != 0) return fmt("%s returned %d; a call that works returns 0", path.fn, rc);
  if (c.produced != 1) return fmt("produce was called %d times; call it exactly once", c.produced);
  if (c.producePtr == nullptr) {
#if defined(PMPP_BACKEND_METAL)
    return "produce was given a null pointer. It needs memory the CPU can write: a private buffer's contents is NULL";
#else
    return "produce was given a null pointer. It needs memory the CPU can write";
#endif
  }
  if (c.consumed != 1) return fmt("consume was called %d times; call it exactly once, after the last pass", c.consumed);
  if (c.consumeNull) return "consume was given a null pointer. It needs memory the CPU can read";

  // (1) The result.
  const std::string wrong = checkResult(c.got, im, ids, passes);
  if (!wrong.empty()) return wrong;

  // (2) Structure: one pointer for both sides, or a copy.
  if (rec.launches == 0) return "the provided launcher was never called";
  if (path.inPlace && rec.table != c.producePtr)
    return fmt("the kernel read its table at %p, but produce wrote it at %p: this path uses the table in place, "
               "with no copy", rec.table, c.producePtr);
  if (!path.inPlace && rec.table == c.producePtr)
    return "the kernel read the very memory produce wrote: this path copies the table into the device's own memory first";
#if defined(PMPP_BACKEND_METAL)
  if (!path.inPlace && rec.tablePrivate != 1)
    return "the table buffer the kernel read does not have private storage";
  // (3) newBufferWithBytesNoCopy wants page-aligned memory (ch02-facts C2).
  if (std::strcmp(path.name, "nocopy") == 0 && (uintptr_t)c.producePtr % (uintptr_t)getpagesize() != 0)
    return fmt("produce wrote at %p, which is not a multiple of the page size (%d bytes)", c.producePtr, getpagesize());
#endif
  return "";
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::string broken = setupError();
  if (!broken.empty()) {
    t.check("kernel.metal builds into a pipeline", [&] { return broken; });
    return t.finish();
  }
  std::vector<std::pair<int, int>> shapes = {{1, 1}, {7, 3}, {64, 33}, {130, 17}};
  if (!pmpp::is_emu()) shapes.push_back({4096, 256});
  for (const Path& p : paths()) {
    for (const auto& shape : shapes) {
      const Image im(shape.first, shape.second);
      const std::vector<int> dense = denseIds(im.rows);
      const std::vector<int> sparse = sparseIds(im.rows, (im.rows + 7) / 8, 1000u + (uint32_t)(im.rows * 31 + im.cols));
      for (int w = 0; w < 2; ++w) {
        const std::vector<int>& ids = w ? sparse : dense;
        for (int passes : {1, 3}) {
          const std::string name = fmt("%s, table %d x %d, %s ids (%zu), %d pass%s", p.fn, im.rows, im.cols,
                                       w ? "sparse" : "dense", ids.size(), passes, passes == 1 ? "" : "es");
          t.check(name, [&] { return runCase(p, im, ids, passes); });
        }
      }
    }
  }
  return t.finish();
}

// ---- benchmark ------------------------------------------------------------------
#if defined(PMPP_BACKEND_METAL)
// Does a buffer's CPU address equal its GPU address? Apple documents neither
// answer (appC-facts C3); this is what this machine shows.
static void addressesRow(const pmpp::Args& a) {
  if (@available(macOS 13.0, *)) {
    @autoreleasepool {
      id<MTLDevice> d = metalSetup().device;
      const size_t page = (size_t)getpagesize();
      id<MTLBuffer> shared = [d newBufferWithLength:page options:MTLResourceStorageModeShared];
      void* p = nullptr;
      if (posix_memalign(&p, page, page) != 0) p = nullptr;
      id<MTLBuffer> nocopy = nil;
      if (p) {
        nocopy = [d newBufferWithBytesNoCopy:p length:page options:MTLResourceStorageModeShared
                                 deallocator:^(void* q, NSUInteger) { free(q); }];
        if (!nocopy) free(p);
      }
      if (!shared || !nocopy) { std::printf("addresses: could not make the two probe buffers\n\n"); return; }
      const bool sharedSame = (uint64_t)(uintptr_t)shared.contents == (uint64_t)shared.gpuAddress;
      const bool nocopySame = (uint64_t)(uintptr_t)nocopy.contents == (uint64_t)nocopy.gpuAddress;
      std::printf("CPU address (contents) against GPU address (gpuAddress), observed here, not documented:\n");
      std::printf("  shared buffer     contents %p  gpuAddress 0x%llx  %s\n", shared.contents,
                  (unsigned long long)shared.gpuAddress, sharedSame ? "same" : "different");
      std::printf("  no-copy buffer    contents %p  gpuAddress 0x%llx  %s\n\n", nocopy.contents,
                  (unsigned long long)nocopy.gpuAddress, nocopySame ? "same" : "different");
      pmpp::bench_row(a, "addresses").b("shared_same", sharedSame).b("nocopy_same", nocopySame).emit();
    }
  } else {
    std::printf("addresses: gpuAddress needs macOS 13 or later; row skipped\n\n");
  }
}
#endif

static int bench(const pmpp::Args& a) {
  const std::string broken = setupError();
  if (!broken.empty()) { std::printf("bench skipped: %s\n", broken.c_str()); return 1; }

  // The table: 256 floats per row, as many rows as fit in an eighth of the memory
  // the device reports, and never more than 256 MiB. The cap is a parameter of
  // this lab, chosen to keep a run short. It is not a property of any device.
  const int cols = 256;
  const double cap = 256.0 * 1024 * 1024;
  const double mem = deviceMemory();
  const double budget = mem > 0 ? std::min(cap, mem / 8) : cap;
  const int rows = (int)(budget / (cols * sizeof(float)));
  if (rows < 64) { std::printf("bench skipped: the device reports too little memory (%.0f bytes)\n", mem); return 1; }
  const Image im(rows, cols);
  const double tableBytes = (double)im.v.size() * sizeof(float);
  std::printf("table: %d rows x %d floats = %.0f bytes (%.1f MiB)\n", rows, cols, tableBytes, tableBytes / 1048576.0);
  std::printf("  the smaller of a 256 MiB cap (this lab's parameter) and an eighth of %s (%.0f bytes)\n\n",
              memoryName(), mem);

#if defined(PMPP_BACKEND_METAL)
  addressesRow(a);
#else
  {
    int cma, pma, uhpt;
    const char* paradigm = queryManagedParadigm(&cma, &pma, &uhpt);
    std::printf("unified-memory paradigm: %s  (ConcurrentManagedAccess %d, PageableMemoryAccess %d, "
                "PageableMemoryAccessUsesHostPageTables %d)\n", paradigm, cma, pma, uhpt);
    pmpp::bench_row(a, "paradigm").n("concurrent_managed", cma).n("pageable_access", pma)
        .n("uses_host_page_tables", uhpt).s("paradigm", paradigm).emit();
    double link = 0, device = 0;
    if (probeBandwidth((size_t)tableBytes, a.reps, &link, &device) == 0) {
      std::printf("probe: pinned host to device %.1f GB/s; device to device %.1f GB/s (read plus write); "
                  "ratio %.3f\n\n", link, device, link / device);
      pmpp::bench_row(a, "probe").n("link_gb_per_s", link).n("device_gb_per_s", device).emit();
    } else {
      std::printf("probe: failed (see the message above)\n\n");
    }
  }
#endif

  struct Workload { const char* name; std::vector<int> ids; int passes; };
  const std::vector<int> dense = denseIds(rows);
  const std::vector<Workload> workloads = {
      {"dense1", dense, 1}, {"dense8", dense, 8}, {"sparse1", sparseIds(rows, std::max(1, rows / 64), 77u), 1}};

  std::printf("Each time is one whole feed call: allocation, produce, copies, passes, synchronise, consume, free.\n");
  std::printf("Best of %d calls after one untimed warm-up call, whose result is checked.\n", a.reps);
#if defined(PMPP_BACKEND_METAL)
  std::printf("GPU copy and GPU kernels: what the command buffers your code passed to waitFor report.\n");
  std::printf("\n%-8s %8s %-8s %12s %12s %12s\n", "workload", "touched", "path", "seconds", "GPU copy", "GPU kernels");
#else
  std::printf("\n%-8s %8s %-8s %12s\n", "workload", "touched", "path", "seconds");
#endif

  std::map<std::string, double> secs;
  int rc_all = 0;
  for (const Workload& w : workloads) {
    const double touched = (double)w.passes * w.ids.size() * cols * sizeof(float);
    for (const Path& p : paths()) {
      Ctx c;
      c.image = im.v.data();
      c.tableBytes = (size_t)tableBytes;
      c.outFloats = w.ids.size() * (size_t)cols;
      c.got.assign(c.outFloats, 0.0f);
      FeedJob job{rows, cols, w.ids.data(), (int)w.ids.size(), w.passes, produceTable, consumeResult, &c};

      idleDevice();
      int rc = p.call(&job);
      std::string wrong = rc != 0 ? fmt("%s returned %d", p.fn, rc) : checkResult(c.got, im, w.ids, w.passes);
      double best = std::numeric_limits<double>::infinity();
      FeedRecord bestRec = feedRecord();
      for (int r = 0; r < a.reps && wrong.empty(); ++r) {
        idleDevice();
        feedResetRecord();
        const auto t0 = std::chrono::steady_clock::now();
        rc = p.call(&job);
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (rc != 0) { wrong = fmt("%s returned %d", p.fn, rc); break; }
        if (s < best) { best = s; bestRec = feedRecord(); }
      }
      if (wrong.empty()) wrong = checkResult(c.got, im, w.ids, w.passes);
      if (!wrong.empty()) {
        std::printf("%-8s %8.4g %-8s  failed: %s\n", w.name, touched / tableBytes, p.name, wrong.c_str());
        rc_all = 1;
        continue;
      }
      secs[std::string(w.name) + "/" + p.name] = best;
      pmpp::Json row = pmpp::bench_row(a, "feed");
      row.s("path", p.name).s("workload", w.name).n("rows", rows).n("cols", cols).n("count", (double)w.ids.size())
          .n("passes", w.passes).n("table_bytes", tableBytes).n("touched_bytes", touched).n("seconds", best);
#if defined(PMPP_BACKEND_METAL)
      std::printf("%-8s %8.4g %-8s %12.6f %12.6f %12.6f\n", w.name, touched / tableBytes, p.name, best,
                  bestRec.gpu_copy_s, bestRec.gpu_kernel_s);
      row.n("gpu_copy_s", bestRec.gpu_copy_s).n("gpu_kernel_s", bestRec.gpu_kernel_s);
#else
      std::printf("%-8s %8.4g %-8s %12.6f\n", w.name, touched / tableBytes, p.name, best);
#endif
      row.emit();
    }
  }

  // The ratios the predictions in README.md ask about. Nothing is compared with an expected value.
  auto ratio = [&](const char* w, const char* x, const char* y) -> double {
    auto i = secs.find(std::string(w) + "/" + x), j = secs.find(std::string(w) + "/" + y);
    return (i == secs.end() || j == secs.end()) ? std::numeric_limits<double>::quiet_NaN() : i->second / j->second;
  };
  std::printf("\n'touched' is the table bytes the passes read over the call's life, divided by the table's size.\n");
  std::printf("Ratios of whole-call times, to set beside what you wrote down:\n");
#if defined(PMPP_BACKEND_METAL)
  std::printf("  (3) dense8   private / shared = %.3f   nocopy / shared = %.3f\n", ratio("dense8", "private", "shared"),
              ratio("dense8", "nocopy", "shared"));
  std::printf("      dense1   private / shared = %.3f   nocopy / shared = %.3f\n", ratio("dense1", "private", "shared"),
              ratio("dense1", "nocopy", "shared"));
  std::printf("      sparse1  private / shared = %.3f   nocopy / shared = %.3f\n", ratio("sparse1", "private", "shared"),
              ratio("sparse1", "nocopy", "shared"));
#else
  std::printf("  (1) sparse1  explicit / mapped = %.3f\n", ratio("sparse1", "explicit", "mapped"));
  std::printf("  (2) dense8   mapped / explicit = %.3f\n", ratio("dense8", "mapped", "explicit"));
  for (const Workload& w : workloads)
    std::printf("  (4) %-8s managed / explicit = %.3f   managed / mapped = %.3f\n", w.name,
                ratio(w.name, "managed", "explicit"), ratio(w.name, "managed", "mapped"));
#endif
  std::printf("These are measurements of this machine at one table size, not properties of the paths.\n");
  return rc_all;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
