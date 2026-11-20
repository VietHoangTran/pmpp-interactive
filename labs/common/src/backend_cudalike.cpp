// pmpp backend for the CUDA family: real CUDA, HIP (via pmpp/vendors/hip.h), or
// the CPU emulator (pmpp/emu/cuda_emu.h force-included by the build).
#include <chrono>
#include <cstdio>
#include <string>

#if defined(PMPP_BACKEND_CUDA)
#include <cuda_runtime.h>
#elif defined(PMPP_BACKEND_HIP)
#include "pmpp/vendors/hip.h"
#elif defined(PMPP_BACKEND_EMU)
#include "pmpp/emu/cuda_emu.h"
#else
#error "backend_cudalike.cpp needs PMPP_BACKEND_CUDA, PMPP_BACKEND_HIP or PMPP_BACKEND_EMU"
#endif

#include "pmpp/backend.hpp"

namespace pmpp {

struct Buf { void* dev; size_t bytes; };

static std::string g_error;
static void note(cudaError_t e, const char* what) {
  if (e != cudaSuccess && g_error.empty()) g_error = std::string(what) + ": " + cudaGetErrorString(e);
}

const char* backend() {
#if defined(PMPP_BACKEND_CUDA)
  return "cuda";
#elif defined(PMPP_BACKEND_HIP)
  return "hip";
#else
  return "emu";
#endif
}

bool device_available(std::string* why) {
  int n = 0;
  cudaError_t e = cudaGetDeviceCount(&n);
  if (e != cudaSuccess) { if (why) *why = cudaGetErrorString(e); return false; }
  if (n < 1) { if (why) *why = "device count is 0"; return false; }
  return true;
}

std::vector<std::pair<std::string, std::string>> device_props() {
  std::vector<std::pair<std::string, std::string>> out;
  cudaDeviceProp p;
  if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) return out;
  auto add = [&](const char* k, double v) { char b[64]; std::snprintf(b, sizeof b, "%.0f", v); out.emplace_back(k, b); };
  out.emplace_back("name", p.name);
  add("totalGlobalMem", (double)p.totalGlobalMem);
  add("multiProcessorCount", p.multiProcessorCount);
  add("warpSize", p.warpSize);
  add("maxThreadsPerBlock", p.maxThreadsPerBlock);
  add("maxThreadsDim.x", p.maxThreadsDim[0]);
  add("maxThreadsDim.y", p.maxThreadsDim[1]);
  add("maxThreadsDim.z", p.maxThreadsDim[2]);
  add("maxGridSize.x", p.maxGridSize[0]);
  add("maxGridSize.y", p.maxGridSize[1]);
  add("maxGridSize.z", p.maxGridSize[2]);
  add("sharedMemPerBlock", (double)p.sharedMemPerBlock);
  add("regsPerBlock", p.regsPerBlock);
  add("computeCapability.major", p.major);
  add("computeCapability.minor", p.minor);
  // Note the inconsistent capitalisation in the real headers: ...Multiprocessor
  // for these two, ...MultiProcessor for the two after. HIP's hipDeviceProp_t
  // uses the same spellings (checked against ROCm 10.2 headers, not yet compiled).
  add("sharedMemPerMultiprocessor", (double)p.sharedMemPerMultiprocessor);
  add("regsPerMultiprocessor", p.regsPerMultiprocessor);
  add("maxThreadsPerMultiProcessor", p.maxThreadsPerMultiProcessor);
  add("maxBlocksPerMultiProcessor", p.maxBlocksPerMultiProcessor);
#if defined(PMPP_BACKEND_EMU)
  out.emplace_back("note", "synthetic values describing the CPU emulator, not a GPU");
#endif
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

#if defined(PMPP_BACKEND_EMU)
int schedules() { return 3; }
void use_schedule(int i) {
  using pmpp_emu::Schedule;
  pmpp_emu::set_schedule(i == 0 ? Schedule::Forward : i == 1 ? Schedule::Reverse : Schedule::Shuffle, 7);
}
const char* schedule_name(int i) { return i == 0 ? "threads in order" : i == 1 ? "threads reversed" : "threads shuffled"; }
#else
int schedules() { return 1; }
void use_schedule(int) {}
const char* schedule_name(int) { return "device"; }
#endif

Timing run(const std::function<void(void*)>& enqueue) {
  note(cudaDeviceSynchronize(), "cudaDeviceSynchronize (before)");
  auto t0 = std::chrono::steady_clock::now();
  enqueue(nullptr);
  note(cudaGetLastError(), "kernel launch");
  note(cudaDeviceSynchronize(), "kernel execution");
  auto t1 = std::chrono::steady_clock::now();
  return Timing{std::chrono::duration<double>(t1 - t0).count(), -1.0};
}

std::string take_error() { std::string e = g_error; g_error.clear(); return e; }

}  // namespace pmpp
