// pmpp backend "race": the CPU emulator in its race-detection flavour
// (pmpp/emu/cuda_emu.h with PMPP_EMU_RACE), built with ThreadSanitizer.
// It mirrors the emulator half of labs/common/src/backend_cudalike.cpp and adds
// the glue that turns a ThreadSanitizer report into a failed test.
#include <chrono>
#include <cstdio>
#include <string>

#if !defined(PMPP_BACKEND_RACE)
#error "backend_race.cpp needs PMPP_BACKEND_RACE"
#endif
#include "pmpp/emu/cuda_emu.h"
#include "pmpp/backend.hpp"

// ---- ThreadSanitizer hooks ---------------------------------------------------------
// Defaults for every race binary, overridable through the TSAN_OPTIONS
// environment variable: keep running after a report so the test output stays
// complete, and exit with a status labs/pmpp.py recognises (macOS would abort).
extern "C" const char* __tsan_default_options() {
  return "abort_on_error=0:exitcode=66:halt_on_error=0:report_thread_leaks=0";
}
static int g_tsan_reports = 0;
// Called by the ThreadSanitizer runtime for every report it prints. It runs
// inside the runtime, so it must not be instrumented itself.
extern "C" __attribute__((no_sanitize("thread"))) void __tsan_on_report(void*) { g_tsan_reports = g_tsan_reports + 1; }

namespace pmpp {

struct Buf { void* dev; size_t bytes; };

static std::string g_error;
static int g_tsan_seen = 0;
static void note(cudaError_t e, const char* what) {
  if (e != cudaSuccess && g_error.empty()) g_error = std::string(what) + ": " + cudaGetErrorString(e);
}

const char* backend() { return "race"; }

bool device_available(std::string*) { return true; }

std::vector<std::pair<std::string, std::string>> device_props() {
  std::vector<std::pair<std::string, std::string>> out;
  cudaDeviceProp p;
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return out;
  auto add = [&](const char* k, double v) { char b[64]; std::snprintf(b, sizeof b, "%.0f", v); out.emplace_back(k, b); };
  out.emplace_back("name", p.name);
  add("warpSize", p.warpSize);
  add("maxThreadsPerBlock", p.maxThreadsPerBlock);
  add("maxThreadsDim.x", p.maxThreadsDim[0]);
  add("maxThreadsDim.y", p.maxThreadsDim[1]);
  add("maxThreadsDim.z", p.maxThreadsDim[2]);
  add("maxGridSize.x", p.maxGridSize[0]);
  add("maxGridSize.y", p.maxGridSize[1]);
  add("maxGridSize.z", p.maxGridSize[2]);
  add("sharedMemPerBlock", (double)p.sharedMemPerBlock);
  add("multiProcessorCount", p.multiProcessorCount);
  out.emplace_back("note", "synthetic values describing the CPU emulator (race-detection flavour), not a GPU");
  return out;
}

Buf* alloc(size_t bytes) {
  Buf* b = new Buf{nullptr, bytes};
  note(cudaMalloc(&b->dev, bytes), "cudaMalloc");
  return b;
}
void upload(Buf* b, const void* src, size_t bytes) { note(cudaMemcpy(b->dev, src, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D"); }
void download(Buf* b, void* dst, size_t bytes) { note(cudaMemcpy(dst, b->dev, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H"); }
void release(Buf* b) { if (b) { cudaFree(b->dev); delete b; } }
void* raw(Buf* b) { return b->dev; }

// The operating system schedules the threads, so there is one "schedule" and
// it is not repeatable.
int schedules() { return 1; }
void use_schedule(int) {}
const char* schedule_name(int) { return "concurrent threads"; }

Timing run(const std::function<void(void*)>& enqueue) {
  auto t0 = std::chrono::steady_clock::now();
  enqueue(nullptr);
  note(cudaGetLastError(), "kernel launch");
  auto t1 = std::chrono::steady_clock::now();
  return Timing{std::chrono::duration<double>(t1 - t0).count(), -1.0};
}

// Also reports data races: the harness calls this after every test case, so a
// ThreadSanitizer report fails the case during which it was printed.
std::string take_error() {
  std::string e = g_error;
  g_error.clear();
  const int now = g_tsan_reports;
  if (now != g_tsan_seen) {
    char b[200];
    std::snprintf(b, sizeof b, "%sThreadSanitizer reported %d data race%s during this test (two emulated threads "
                  "touched the same memory without synchronisation; see the report above)",
                  e.empty() ? "" : "; ", now - g_tsan_seen, now - g_tsan_seen == 1 ? "" : "s");
    e += b;
    g_tsan_seen = now;
  }
  return e;
}

}  // namespace pmpp
