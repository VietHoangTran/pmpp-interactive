// pmpp CPU emulator for the subset of CUDA used in this course.
//
// This is NOT a GPU and NOT a CUDA implementation. It exists so that kernel
// logic (index arithmetic, shared-memory use, barrier placement, bounds checks)
// can be executed and tested on a machine with no CUDA or HIP device.
//
// What it models faithfully:
//   - grid / block / thread indices (threadIdx, blockIdx, blockDim, gridDim)
//   - one private copy of automatic variables per thread
//   - __shared__ variables visible to every thread of a block
//   - __syncthreads() as a barrier across the live threads of a block
//   - dynamically sized shared memory (third launch parameter)
//
// What it does NOT model (do not draw conclusions about these from it):
//   - timing, bandwidth, caches, occupancy, warps, coalescing, bank conflicts
//   - true concurrency: exactly one emulated thread runs at a time, switching
//     only at __syncthreads() or when a thread returns. Schedules are
//     deterministic (forward, reverse, shuffled) so that a missing barrier
//     produces a wrong answer every time instead of sometimes.
//   - separate host and device address spaces: cudaMemcpy is a memcpy, so
//     dereferencing a device pointer on the host is not caught here.
//   - blocks run one after another, so __shared__ state left by one block is
//     visible to the next (on a GPU shared memory is simply uninitialised).
//   - atomics, warp intrinsics, cooperative groups, streams, textures.
#pragma once
#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

struct uint3 { unsigned x, y, z; };
struct dim3 {
  unsigned x, y, z;
  constexpr dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};

typedef int cudaError_t;
enum { cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorMemoryAllocation = 2,
       cudaErrorInvalidConfiguration = 9 };
enum cudaMemcpyKind { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1,
                      cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3 };
typedef struct pmpp_emu_stream* cudaStream_t;

// Only the fields this course reads. Values are synthetic: they describe the
// emulator, not any real GPU.
struct cudaDeviceProp {
  char name[256];
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  size_t sharedMemPerMultiprocessor;
  int regsPerBlock;
  int regsPerMultiprocessor;
  int warpSize;
  int maxThreadsPerBlock;
  int maxThreadsPerMultiProcessor;
  int maxBlocksPerMultiProcessor;
  int multiProcessorCount;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int major, minor;
};

namespace pmpp_emu {

enum class Schedule { Forward, Reverse, Shuffle };

struct Slot {
  std::mutex m;
  std::condition_variable cv;
  bool go = false;
};

struct State {
  dim3 gridDim, blockDim;
  std::vector<unsigned char> dynShared;
  Schedule schedule = Schedule::Forward;
  uint32_t seed = 1;
  cudaError_t lastError = cudaSuccess;
  // scheduling of the block currently running
  std::vector<std::unique_ptr<Slot>> slots;   // one per emulated thread
  Slot coord;
  enum St : unsigned char { Ready, AtBarrier, Done };
  std::vector<St> st;
  bool shutdown = false;
  uint3 curBlock{0, 0, 0};
  long launches = 0;
};

inline State& S() { static State s; return s; }
inline thread_local uint3 t_threadIdx{0, 0, 0};
inline thread_local int t_tid = -1;

inline void wait_go(Slot& s) {
  std::unique_lock<std::mutex> lk(s.m);
  s.cv.wait(lk, [&] { return s.go; });
  s.go = false;
}
inline void give(Slot& s) {
  { std::lock_guard<std::mutex> lk(s.m); s.go = true; }
  s.cv.notify_one();
}

inline void set_schedule(Schedule sch, uint32_t seed = 1) { S().schedule = sch; S().seed = seed; }
inline void* dyn_shared() { return S().dynShared.empty() ? nullptr : S().dynShared.data(); }

// Called by an emulated thread at __syncthreads(): hand control back to the
// coordinator and sleep until every live thread of the block has arrived.
inline void barrier() {
  State& s = S();
  if (t_tid < 0) { std::fprintf(stderr, "pmpp-emu: __syncthreads() called outside a kernel\n"); std::abort(); }
  s.st[t_tid] = State::AtBarrier;
  give(s.coord);
  wait_go(*s.slots[t_tid]);
}

struct Limits {
  static constexpr unsigned maxThreadsPerBlock = 1024;
  static constexpr size_t sharedMemPerBlock = 48 * 1024;
};

class Launch {
 public:
  Launch(dim3 grid, dim3 block, size_t sharedBytes = 0, cudaStream_t = nullptr)
      : grid_(grid), block_(block), shared_(sharedBytes) {}

  template <class F, class... A>
  void run(F kernel, A... args) {
    State& s = S();
    const unsigned T = block_.x * block_.y * block_.z;
    const uint64_t B = uint64_t(grid_.x) * grid_.y * grid_.z;
    if (T == 0 || B == 0 || T > Limits::maxThreadsPerBlock || shared_ > Limits::sharedMemPerBlock) {
      std::fprintf(stderr, "pmpp-emu: invalid launch configuration: grid %ux%ux%u, block %ux%ux%u "
                           "(%u threads, emulator limit %u), %zu bytes dynamic shared memory (limit %zu)\n",
                   grid_.x, grid_.y, grid_.z, block_.x, block_.y, block_.z, T,
                   Limits::maxThreadsPerBlock, shared_, Limits::sharedMemPerBlock);
      s.lastError = cudaErrorInvalidConfiguration;
      return;
    }
    s.gridDim = grid_; s.blockDim = block_;
    s.dynShared.assign(shared_, 0xCD);
    s.slots.clear();
    for (unsigned i = 0; i < T; ++i) s.slots.emplace_back(new Slot());
    s.st.assign(T, State::Ready);
    s.shutdown = false;
    s.launches++;

    std::tuple<A...> packed(args...);
    std::vector<std::thread> pool;
    pool.reserve(T);
    for (unsigned i = 0; i < T; ++i) {
      pool.emplace_back([&, i] {
        State& st = S();
        t_tid = int(i);
        t_threadIdx = uint3{i % block_.x, (i / block_.x) % block_.y, i / (block_.x * block_.y)};
        for (;;) {
          wait_go(*st.slots[i]);
          if (st.shutdown) return;
          std::apply(kernel, packed);
          st.st[i] = State::Done;
          give(st.coord);
        }
      });
    }

    std::vector<unsigned> order(T);
    for (unsigned i = 0; i < T; ++i) order[i] = i;
    uint32_t rng = s.seed * 2654435761u + 12345u;
    for (uint64_t b = 0; b < B; ++b) {
      s.curBlock = uint3{unsigned(b % grid_.x), unsigned((b / grid_.x) % grid_.y),
                         unsigned(b / (uint64_t(grid_.x) * grid_.y))};
      std::fill(s.st.begin(), s.st.end(), State::Ready);
      if (s.schedule == Schedule::Reverse) {
        for (unsigned i = 0; i < T; ++i) order[i] = T - 1 - i;
      } else if (s.schedule == Schedule::Shuffle) {
        for (unsigned i = T - 1; i > 0; --i) {
          rng = rng * 1664525u + 1013904223u;
          std::swap(order[i], order[(rng >> 8) % (i + 1)]);
        }
      }
      bool any = true;
      while (any) {
        any = false;
        for (unsigned k = 0; k < T; ++k) {
          const unsigned id = order[k];
          if (s.st[id] == State::Done) continue;
          give(*s.slots[id]);     // run thread `id` until its next barrier or its return
          wait_go(s.coord);
          if (s.st[id] != State::Done) any = true;
        }
      }
    }
    s.shutdown = true;
    for (unsigned i = 0; i < T; ++i) give(*s.slots[i]);
    for (auto& t : pool) t.join();
    s.slots.clear();
  }

 private:
  dim3 grid_, block_;
  size_t shared_;
};

}  // namespace pmpp_emu

#define __global__
#define __device__
#define __host__
#define __constant__
#define __shared__ static
#define threadIdx (::pmpp_emu::t_threadIdx)
#define blockIdx  (::pmpp_emu::S().curBlock)
#define blockDim  (::pmpp_emu::S().blockDim)
#define gridDim   (::pmpp_emu::S().gridDim)
inline void __syncthreads() { ::pmpp_emu::barrier(); }

inline cudaError_t cudaMalloc(void** p, size_t n) {
  *p = std::malloc(n ? n : 1);
  if (*p) std::memset(*p, 0xA5, n);            // device memory is not zeroed
  return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
// No typed overload on purpose: real CUDA takes void**, so `cudaMalloc(&floatPtr, n)`
// must be written with a (void**) cast here too.
inline cudaError_t cudaFree(void* p) { std::free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) { std::memcpy(d, s, n); return cudaSuccess; }
inline cudaError_t cudaMemset(void* d, int v, size_t n) { std::memset(d, v, n); return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { auto e = ::pmpp_emu::S().lastError; ::pmpp_emu::S().lastError = cudaSuccess; return e; }
inline cudaError_t cudaPeekAtLastError() { return ::pmpp_emu::S().lastError; }
inline const char* cudaGetErrorString(cudaError_t e) {
  switch (e) { case cudaSuccess: return "no error";
               case cudaErrorInvalidConfiguration: return "invalid configuration argument";
               case cudaErrorMemoryAllocation: return "out of memory";
               default: return "error (pmpp emulator)"; }
}
inline cudaError_t cudaGetDeviceCount(int* n) { *n = 1; return cudaSuccess; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) {
  std::memset(p, 0, sizeof(*p));
  std::snprintf(p->name, sizeof(p->name), "pmpp CPU emulator (not a GPU)");
  p->sharedMemPerBlock = ::pmpp_emu::Limits::sharedMemPerBlock;
  p->sharedMemPerMultiprocessor = ::pmpp_emu::Limits::sharedMemPerBlock;
  p->maxThreadsPerBlock = ::pmpp_emu::Limits::maxThreadsPerBlock;
  p->maxThreadsPerMultiProcessor = ::pmpp_emu::Limits::maxThreadsPerBlock;
  p->maxBlocksPerMultiProcessor = 1;
  p->multiProcessorCount = 1;
  p->warpSize = 1;
  p->maxThreadsDim[0] = p->maxThreadsDim[1] = p->maxThreadsDim[2] = ::pmpp_emu::Limits::maxThreadsPerBlock;
  p->maxGridSize[0] = p->maxGridSize[1] = p->maxGridSize[2] = 65535;
  return cudaSuccess;
}
