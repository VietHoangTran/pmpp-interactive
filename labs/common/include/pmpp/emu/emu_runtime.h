// pmpp CPU emulator, part 2 of 4: the scheduler shared by both flavours.
// Included by pmpp/emu/cuda_emu.h; do not include directly.
//
// Every emulated GPU thread is an OS thread. One mutex per launch guards a
// table of thread states; a thread that reaches a synchronisation point
// (__syncthreads, a warp *_sync intrinsic, grid.sync) records what it is
// waiting for and sleeps until the rendezvous it named is complete.
//
//   flavour "emu"  (default): a coordinator hands a baton to ONE thread at a
//       time, in a deterministic order, and takes it back when that thread
//       reaches a synchronisation point or returns. Rendezvous are resolved
//       when no thread can run.
//   flavour "race" (PMPP_EMU_RACE): the threads of a block, and several blocks,
//       run concurrently under ThreadSanitizer. Rendezvous are resolved the
//       moment they become complete.
#pragma once
#include <algorithm>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <new>
#include <set>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(PMPP_EMU_RACE)
// ThreadSanitizer's dynamic annotations (exported by its runtime on clang and
// gcc). The emulator's own bookkeeping runs with TSan told to ignore it, so
// that its mutex does not create happens-before edges that would hide a race
// in the kernel. The edges a real barrier provides are then stated explicitly.
extern "C" {
void AnnotateIgnoreSyncBegin(const char* file, int line);
void AnnotateIgnoreSyncEnd(const char* file, int line);
void AnnotateIgnoreReadsBegin(const char* file, int line);
void AnnotateIgnoreReadsEnd(const char* file, int line);
void AnnotateIgnoreWritesBegin(const char* file, int line);
void AnnotateIgnoreWritesEnd(const char* file, int line);
void AnnotateHappensBefore(const char* file, int line, const volatile void* addr);
void AnnotateHappensAfter(const char* file, int line, const volatile void* addr);
}
#endif

namespace pmpp_emu {

#if defined(PMPP_EMU_RACE)
struct Quiet {
  Quiet() { AnnotateIgnoreSyncBegin(__FILE__, __LINE__); AnnotateIgnoreReadsBegin(__FILE__, __LINE__); AnnotateIgnoreWritesBegin(__FILE__, __LINE__); }
  ~Quiet() { AnnotateIgnoreWritesEnd(__FILE__, __LINE__); AnnotateIgnoreReadsEnd(__FILE__, __LINE__); AnnotateIgnoreSyncEnd(__FILE__, __LINE__); }
  Quiet(const Quiet&) = delete;
};
// "everything this thread did so far" -> token, and token -> "everything this thread does next".
inline void hb_before(const void* token) { AnnotateHappensBefore(__FILE__, __LINE__, token); }
inline void hb_after(const void* token) { AnnotateHappensAfter(__FILE__, __LINE__, token); }
inline constexpr bool kRace = true;
#else
struct Quiet { Quiet() {} ~Quiet() {} Quiet(const Quiet&) = delete; };
inline void hb_before(const void*) {}
inline void hb_after(const void*) {}
inline constexpr bool kRace = false;
#endif

enum class Schedule { Forward, Reverse, Shuffle };

// Limits of the emulator. They are choices made for this course, not properties
// of any GPU.
struct Limits {
  static constexpr unsigned maxThreadsPerBlock = 1024;
  static constexpr size_t sharedMemPerBlock = 48 * 1024;        // dynamic shared memory per block
  static constexpr size_t sharedMemPerBlockOptin = 96 * 1024;   // after cudaFuncSetAttribute(MaxDynamicSharedMemorySize)
  // cudaLaunchCooperativeKernel keeps every block alive at once, one OS thread
  // per emulated thread, so grid * block is capped.
  static constexpr unsigned maxCooperativeThreads = 1024;
  // race flavour: blocks run concurrently while their threads fit in this many.
  static constexpr unsigned raceLiveThreads = 256;
};

struct Abort {};   // thrown inside an emulated thread to unwind a kernel the emulator stopped

enum class Wait : unsigned char { None, Block, Grid, Warp };
enum class Op : unsigned char { None, SyncWarp, Shfl, ShflUp, ShflDown, ShflXor, Ballot, Any, All,
                                MatchAny, MatchAll, Active };

struct Rt;
struct BlockCtx;

struct ThreadRec {
  enum St : unsigned char { Idle, Ready, Running, Waiting, Done };
  Rt* rt = nullptr;
  BlockCtx* blk = nullptr;
  unsigned tid = 0;                 // linear index inside the block
  St st = Idle;
  bool go = false, started = false;
  Wait wait = Wait::None;
  Op op = Op::None;
  unsigned mask = 0, size = 0;
  int arg = 0, width = 32;
  uint64_t in = 0, out = 0;
  const char* what = "";            // name of the call it is waiting in
  const char* file = "";
  int line = 0;
  std::condition_variable cv;
};

struct BlockCtx {
  uint3 idx{0, 0, 0};
  uint64_t rank = 0;
  bool active = false;
  unsigned first = 0;               // index of its thread 0 in Rt::th
  unsigned live = 0, atBarrier = 0;
  std::vector<void*> slots;         // one per __shared__ declaration in the program
  unsigned char* dyn = nullptr;     // dynamic shared memory
  char hbBarrier = 0;
  std::vector<char> hbWarp;
};

struct Rt {
  std::mutex m;
  std::condition_variable coord;
  dim3 grid, block;
  unsigned T = 0;
  uint64_t B = 0;
  bool cooperative = false;
  size_t dynBytes = 0;
  std::vector<BlockCtx> blocks;
  std::vector<ThreadRec> th;
  unsigned live = 0, running = 0, atGrid = 0;
  bool aborting = false, shutdown = false, yielded = false;
  std::string error;
  char hbGrid = 0;
};

struct SharedType {
  size_t size, align;
  int index;
  SharedType(size_t s, size_t a);
};

struct Global {
  Schedule schedule = Schedule::Forward;
  uint32_t seed = 1;
  cudaError_t lastError = cudaSuccess;
  std::string message;                               // detail of the last failure the emulator detected
  cudaError_t messageFor = cudaSuccess;
  long launches = 0;
  std::vector<SharedType*> sharedTypes;
  std::map<const void*, size_t> dynSharedOptin;      // cudaFuncSetAttribute(MaxDynamicSharedMemorySize)
  std::map<const void*, std::function<cudaError_t(dim3, dim3, void**, size_t)>> kernels;
  std::set<std::pair<std::string, int>> noted;
};
inline Global& S() { static Global g; return g; }
inline SharedType::SharedType(size_t s, size_t a) : size(s), align(a), index(int(S().sharedTypes.size())) {
  S().sharedTypes.push_back(this);
}

inline thread_local ThreadRec* t_me = nullptr;

inline void set_schedule(Schedule sch, uint32_t seed = 1) { S().schedule = sch; S().seed = seed; }
// Detail of the last failure the emulator detected (empty if none since the last launch).
inline const std::string& last_message() { return S().message; }

inline std::string fmt(const char* f, ...) {
  char buf[640];
  va_list ap;
  va_start(ap, f);
  std::vsnprintf(buf, sizeof buf, f, ap);
  va_end(ap);
  return buf;
}

inline ThreadRec& self(const char* what) {
  if (!t_me) {
    std::fprintf(stderr, "pmpp-emu: %s used outside a kernel (host code cannot call device functions)\n", what);
    std::abort();
  }
  return *t_me;
}

inline std::string where(const ThreadRec& t) {
  const char* base = std::strrchr(t.file, '/');
  return fmt("%s [%s:%d]", t.what, base ? base + 1 : t.file, t.line);
}
inline std::string who(const ThreadRec& t) {
  return fmt("thread %u (warp %u, lane %u) of block (%u,%u,%u)", t.tid, t.tid / 32, t.tid % 32,
             t.blk->idx.x, t.blk->idx.y, t.blk->idx.z);
}

// ---------------------------------------------------------------- state changes (lock held)
inline void release(Rt& rt, ThreadRec& t) {
  if (t.wait == Wait::Block) t.blk->atBarrier--;
  else if (t.wait == Wait::Grid) rt.atGrid--;
  t.wait = Wait::None;
  if (kRace) { t.st = ThreadRec::Running; rt.running++; t.go = true; t.cv.notify_one(); }
  else t.st = ThreadRec::Ready;
}

inline void set_error(Rt& rt, const std::string& msg) {
  if (rt.aborting) return;
  rt.aborting = true;
  rt.error = msg;
  if (kRace)
    for (ThreadRec& t : rt.th)
      if (t.st == ThreadRec::Waiting) release(rt, t);   // they wake up and unwind
}
[[noreturn]] inline void fail(Rt& rt, const std::string& msg) { set_error(rt, msg); throw Abort{}; }

inline bool same_call(const ThreadRec& a, const ThreadRec& b) {
  return a.op == b.op && a.mask == b.mask && a.size == b.size && a.line == b.line &&
         (a.file == b.file || std::strcmp(a.file, b.file) == 0);
}

inline void note_undefined_read(const ThreadRec& t, unsigned srcLane) {
  auto key = std::make_pair(std::string(t.file), t.line);
  if (!S().noted.insert(key).second) return;
  std::fprintf(stderr, "pmpp-emu: note: %s read lane %u, which is not taking part in the call (it has exited, "
               "does not exist in this block, or is not in the mask). On a GPU that value is undefined; "
               "the emulator returns 0xCD bytes.\n", where(t).c_str(), srcLane);
}

// Results for one complete warp rendezvous. `members` has one bit per lane.
inline void compute_group(ThreadRec* lane0, unsigned n, unsigned members, const ThreadRec& first) {
  unsigned pred = 0;
  for (unsigned j = 0; j < n; ++j)
    if ((members >> j & 1) && lane0[j].in != 0) pred |= 1u << j;
  for (unsigned L = 0; L < n; ++L) {
    if (!(members >> L & 1)) continue;
    ThreadRec& t = lane0[L];
    switch (first.op) {
      case Op::SyncWarp: t.out = 0; break;
      case Op::Ballot: t.out = pred; break;
      case Op::Any: t.out = pred != 0; break;
      case Op::All: t.out = pred == members; break;
      case Op::MatchAny: {
        unsigned m = 0;
        for (unsigned j = 0; j < n; ++j) if ((members >> j & 1) && lane0[j].in == t.in) m |= 1u << j;
        t.out = m;
        break;
      }
      case Op::MatchAll: {
        bool all = true;
        for (unsigned j = 0; j < n; ++j) if ((members >> j & 1) && lane0[j].in != t.in) all = false;
        t.out = all ? t.mask : 0;
        t.arg = all;
        break;
      }
      case Op::Shfl: case Op::ShflUp: case Op::ShflDown: case Op::ShflXor: {
        const unsigned w = unsigned(t.width), base = L / w * w, a = unsigned(t.arg);
        unsigned src = L;
        if (first.op == Op::Shfl) src = base + (a % w);
        else if (first.op == Op::ShflUp) src = (L - base >= a) ? L - a : L;
        else if (first.op == Op::ShflDown) src = (uint64_t(L - base) + a < w) ? L + a : L;
        else { src = L ^ (a & 31u); if (src / w > L / w) src = L; }   // later groups are out of reach
        if (src == L) t.out = t.in;
        else if (src < n && (members >> src & 1)) t.out = lane0[src].in;
        else { t.out = 0xCDCDCDCDCDCDCDCDull; note_undefined_read(t, src); }
        break;
      }
      default: break;
    }
  }
}

inline bool try_barrier(Rt& rt, BlockCtx& b) {
  if (b.live == 0 || b.atBarrier != b.live) return false;
  ThreadRec* t0 = &rt.th[b.first];
  unsigned count = 0;
  for (unsigned i = 0; i < rt.T; ++i)
    if (t0[i].st == ThreadRec::Waiting && t0[i].wait == Wait::Block && t0[i].in) count++;
  const unsigned n = b.live;
  for (unsigned i = 0; i < rt.T; ++i)
    if (t0[i].st == ThreadRec::Waiting && t0[i].wait == Wait::Block) { t0[i].out = count; t0[i].arg = int(n); release(rt, t0[i]); }
  return true;
}

inline bool try_grid(Rt& rt) {
  if (rt.live == 0 || rt.atGrid != rt.live) return false;
  for (ThreadRec& t : rt.th)
    if (t.st == ThreadRec::Waiting && t.wait == Wait::Grid) release(rt, t);
  return true;
}

inline bool try_warp(Rt& rt, BlockCtx& b, unsigned w) {
  ThreadRec* lane0 = &rt.th[b.first + w * 32];
  const unsigned n = std::min(32u, rt.T - w * 32);
  bool any = false;
  for (unsigned L = 0; L < n; ++L) {
    ThreadRec& t = lane0[L];
    if (!(t.st == ThreadRec::Waiting && t.wait == Wait::Warp) || t.op == Op::Active) continue;
    unsigned members = 0;
    bool complete = true;
    for (unsigned j = 0; j < n && complete; ++j) {
      if (!(t.mask >> j & 1) || lane0[j].st == ThreadRec::Done) continue;   // exited threads are ignored
      const ThreadRec& u = lane0[j];
      if (u.st == ThreadRec::Waiting && u.wait == Wait::Warp && same_call(t, u)) members |= 1u << j;
      else complete = false;
    }
    if (!complete) continue;
    compute_group(lane0, n, members, t);
    for (unsigned j = 0; j < n; ++j) if (members >> j & 1) release(rt, lane0[j]);
    any = true;
  }
  // __activemask() / coalesced_threads(): the lanes of this warp that have reached
  // the same call site, decided once no lane of the warp is still on its way.
  bool settled = true, wanted = false;
  for (unsigned L = 0; L < n; ++L) {
    if (lane0[L].st != ThreadRec::Waiting && lane0[L].st != ThreadRec::Done) settled = false;
    if (lane0[L].st == ThreadRec::Waiting && lane0[L].wait == Wait::Warp && lane0[L].op == Op::Active) wanted = true;
  }
  if (settled && wanted) {
    for (unsigned L = 0; L < n; ++L) {
      ThreadRec& t = lane0[L];
      if (!(t.st == ThreadRec::Waiting && t.wait == Wait::Warp && t.op == Op::Active)) continue;
      unsigned members = 0;
      for (unsigned j = L; j < n; ++j) {
        const ThreadRec& u = lane0[j];
        if (u.st == ThreadRec::Waiting && u.wait == Wait::Warp && u.op == Op::Active && u.line == t.line &&
            std::strcmp(u.file, t.file) == 0) members |= 1u << j;
      }
      for (unsigned j = L; j < n; ++j) if (members >> j & 1) { lane0[j].out = members; release(rt, lane0[j]); }
      any = true;
    }
  }
  return any;
}

inline bool resolve_local(Rt& rt, BlockCtx& b, unsigned warp) {
  bool any = try_barrier(rt, b);
  any |= try_warp(rt, b, warp);
  if (rt.cooperative) any |= try_grid(rt);
  return any;
}
inline bool resolve_all(Rt& rt) {
  bool any = false;
  for (BlockCtx& b : rt.blocks) {
    if (!b.active || b.live == 0) continue;
    any |= try_barrier(rt, b);
    for (unsigned w = 0; w * 32 < rt.T; ++w) any |= try_warp(rt, b, w);
  }
  if (rt.cooperative) any |= try_grid(rt);
  return any;
}

// Called when every live thread is waiting and no rendezvous is complete.
inline std::string describe_deadlock(Rt& rt) {
  std::string s = "every live thread is waiting and none can proceed; on a GPU this is undefined behaviour "
                  "(a hang or wrong results).";
  // The most useful case first: a warp call that names a thread which is somewhere else.
  for (BlockCtx& b : rt.blocks) {
    if (!b.active) continue;
    for (unsigned i = 0; i < rt.T; ++i) {
      const ThreadRec& t = rt.th[b.first + i];
      if (!(t.st == ThreadRec::Waiting && t.wait == Wait::Warp) || t.op == Op::Active) continue;
      const unsigned base = i / 32 * 32, n = std::min(32u, rt.T - base);
      for (unsigned j = 0; j < n; ++j) {
        const ThreadRec& u = rt.th[b.first + base + j];
        if (!(t.mask >> j & 1) || u.st == ThreadRec::Done) continue;
        if (u.st == ThreadRec::Waiting && u.wait == Wait::Warp && same_call(t, u)) continue;
        s += " " + who(t) + " is in " + where(t) + fmt(" with mask 0x%08x, which names lane %u; that thread is in ", t.mask, j) +
             where(u) + (u.wait == Wait::Warp && u.op != Op::Active ? fmt(" with mask 0x%08x", u.mask) : std::string()) +
             ". All non-exited threads named in a mask must execute the same call with the same mask.";
        return s;
      }
    }
  }
  std::map<std::string, unsigned> sites;
  for (const ThreadRec& t : rt.th)
    if (t.st == ThreadRec::Waiting) sites[where(t)]++;
  s += " Waiting:";
  int shown = 0;
  for (auto& kv : sites) {
    if (shown++ == 4) { s += " ..."; break; }
    s += fmt(" %u thread%s in %s;", kv.second, kv.second == 1 ? "" : "s", kv.first.c_str());
  }
  s += " A barrier is only passed when every live thread of the block (or grid) reaches it.";
  return s;
}

// ---------------------------------------------------------------- called by emulated threads
// Sleep until this thread may continue. Lock held on entry and on return.
inline void park(Rt& rt, ThreadRec& me, std::unique_lock<std::mutex>& lk) {
  me.cv.wait(lk, [&] { return me.go; });
  me.go = false;
  if (rt.aborting) throw Abort{};
}

// Record that `me` waits for a rendezvous of kind `kind` and sleep until it is complete.
inline void arrive(Rt& rt, ThreadRec& me, std::unique_lock<std::mutex>& lk, Wait kind) {
  if (rt.aborting) throw Abort{};
  me.st = ThreadRec::Waiting;
  me.wait = kind;
  if (kind == Wait::Block) me.blk->atBarrier++;
  else if (kind == Wait::Grid) rt.atGrid++;
  if (kRace) {
    rt.running--;
    resolve_local(rt, *me.blk, me.tid / 32);
    if (rt.running == 0 && rt.live > 0) set_error(rt, "deadlock: " + describe_deadlock(rt));
  } else {
    rt.yielded = true;
    rt.coord.notify_one();
  }
  park(rt, me, lk);
}

inline void thread_exit(Rt& rt, ThreadRec& me) {
  Quiet q;
  std::lock_guard<std::mutex> lk(rt.m);
  me.st = ThreadRec::Done;
  rt.live--;
  me.blk->live--;
  if (kRace) {
    rt.running--;
    if (!rt.aborting) {
      resolve_local(rt, *me.blk, me.tid / 32);
      if (rt.running == 0 && rt.live > 0) set_error(rt, "deadlock: " + describe_deadlock(rt));
    }
    if (rt.live == 0) rt.coord.notify_one();
  } else {
    rt.yielded = true;
    rt.coord.notify_one();
  }
}

inline void worker(Rt* rtp, ThreadRec* mep, const std::function<void()>* call) {
  Rt& rt = *rtp;
  ThreadRec& me = *mep;
  for (;;) {
    bool run;
    uint3 bi;
    {
      Quiet q;
      std::unique_lock<std::mutex> lk(rt.m);
      me.cv.wait(lk, [&] { return me.go || rt.shutdown; });
      if (!me.go) return;
      me.go = false;
      run = !rt.aborting;
      bi = me.blk->idx;
    }
    threadIdx = uint3{me.tid % rt.block.x, (me.tid / rt.block.x) % rt.block.y, me.tid / (rt.block.x * rt.block.y)};
    blockIdx = bi;
    blockDim = rt.block;
    gridDim = rt.grid;
    t_me = &me;
    if (run) {
      try { (*call)(); } catch (const Abort&) {}
    }
    t_me = nullptr;
    thread_exit(rt, me);
  }
}

// ---------------------------------------------------------------- per-block __shared__ storage
// cu2emu.py turns `__shared__ T name[N];` inside a function into a local struct
// holding the declaration and a reference into this block's copy of it. Every
// such struct registers its size before main() runs; a block gets one fresh
// allocation per registered struct, filled with 0xCD (shared memory is not
// zeroed on a GPU either).
template <class T> struct SharedOf { static inline SharedType type{sizeof(T), alignof(T)}; };

template <class T> inline T& shared_block() {
  ThreadRec& me = self("a __shared__ variable");
  const int i = SharedOf<T>::type.index;
  if (size_t(i) >= me.blk->slots.size()) {
    std::fprintf(stderr, "pmpp-emu: internal error: a __shared__ declaration was registered after the launch began\n");
    std::abort();
  }
  return *static_cast<T*>(me.blk->slots[size_t(i)]);
}
inline void* dyn_shared() { return self("dynamic shared memory").blk->dyn; }

// Plain malloc where its alignment suffices, so that AddressSanitizer guards the
// exact size (an aligned operator new may round the size up to the alignment).
inline void* raw_alloc(size_t size, size_t align) {
  void* p = nullptr;
  if (align <= alignof(std::max_align_t)) p = std::malloc(size ? size : 1);
  else if (posix_memalign(&p, align, size ? size : 1) != 0) p = nullptr;
  if (!p) { std::fprintf(stderr, "pmpp-emu: out of memory for shared memory\n"); std::abort(); }
  return p;
}
inline void raw_free(void* p, size_t) { std::free(p); }

// Every block starts with fresh allocations (so AddressSanitizer sees each
// __shared__ array as its own heap block, and ThreadSanitizer does not confuse
// one block's shared memory with the next one's), filled with 0xCD.
inline void free_shared(BlockCtx& b) {
  for (size_t k = 0; k < b.slots.size(); ++k)
    if (b.slots[k]) { raw_free(b.slots[k], S().sharedTypes[k]->align); b.slots[k] = nullptr; }
  if (b.dyn) { raw_free(b.dyn, 16); b.dyn = nullptr; }
}
inline void fresh_shared(BlockCtx& b, size_t dynBytes) {
  free_shared(b);
  for (size_t k = 0; k < b.slots.size(); ++k) {
    b.slots[k] = raw_alloc(S().sharedTypes[k]->size, S().sharedTypes[k]->align);
    std::memset(b.slots[k], 0xCD, S().sharedTypes[k]->size);
  }
  if (dynBytes) {
    b.dyn = static_cast<unsigned char*>(raw_alloc(dynBytes, 16));
    std::memset(b.dyn, 0xCD, dynBytes);
  }
}

// ---------------------------------------------------------------- the launch
inline cudaError_t launch_error(cudaError_t e, const std::string& msg) {
  Global& g = S();
  g.lastError = e;
  g.message = msg;
  g.messageFor = e;
  std::fprintf(stderr, "pmpp-emu: %s\n", msg.c_str());
  return e;
}

// Run one wave: the blocks currently assigned to rt.blocks, to completion.
inline void run_wave(Rt& rt, const std::vector<unsigned>& order) {
  Quiet q;
  std::unique_lock<std::mutex> lk(rt.m);
  if (kRace) {
    for (unsigned id : order) {
      ThreadRec& t = rt.th[id];
      if (t.st != ThreadRec::Ready) continue;
      t.st = ThreadRec::Running; t.go = true; t.cv.notify_one();
    }
    rt.coord.wait(lk, [&] { return rt.live == 0; });
    return;
  }
  auto step = [&](ThreadRec& t) {      // hand the baton to t and wait until it gives it back
    t.st = ThreadRec::Running; t.started = true; t.go = true; t.cv.notify_one();
    rt.coord.wait(lk, [&] { return rt.yielded; });
    rt.yielded = false;
  };
  for (;;) {
    for (unsigned id : order) {
      if (rt.aborting) break;
      if (rt.th[id].st == ThreadRec::Ready) step(rt.th[id]);
    }
    if (!rt.aborting) {
      if (rt.live == 0) return;
      if (resolve_all(rt)) continue;
      set_error(rt, "deadlock: " + describe_deadlock(rt));
    }
    // Stopped: unwind every thread that is parked inside the kernel, drop the ones that never started.
    for (unsigned id : order) {
      ThreadRec& t = rt.th[id];
      if (t.st == ThreadRec::Done || t.st == ThreadRec::Idle) continue;
      if (!t.started) { t.st = ThreadRec::Done; rt.live--; t.blk->live--; continue; }
      step(t);
    }
    return;
  }
}

inline cudaError_t launch_impl(dim3 grid, dim3 block, size_t sharedBytes, bool cooperative, const void* fn,
                               const std::function<void()>& call) {
  Global& g = S();
  g.message.clear();
  if (t_me) {
    std::fprintf(stderr, "pmpp-emu: a kernel was launched from inside a kernel; the emulator does not support that\n");
    std::abort();
  }
  const uint64_t T64 = uint64_t(block.x) * block.y * block.z;
  const uint64_t B = uint64_t(grid.x) * grid.y * grid.z;
  size_t shLimit = Limits::sharedMemPerBlock;
  if (fn) {
    auto it = g.dynSharedOptin.find(fn);
    if (it != g.dynSharedOptin.end()) shLimit = std::max(shLimit, it->second);
  }
  if (T64 == 0 || B == 0 || T64 > Limits::maxThreadsPerBlock || sharedBytes > shLimit)
    return launch_error(cudaErrorInvalidConfiguration,
        fmt("invalid launch configuration: grid %ux%ux%u, block %ux%ux%u (%llu threads, emulator limit %u), "
            "%zu bytes dynamic shared memory (limit %zu)", grid.x, grid.y, grid.z, block.x, block.y, block.z,
            (unsigned long long)T64, Limits::maxThreadsPerBlock, sharedBytes, shLimit));
  const unsigned T = unsigned(T64);
  uint64_t W = 1;
  if (cooperative) {
    if (B * T > Limits::maxCooperativeThreads)
      return launch_error(cudaErrorCooperativeLaunchTooLarge,
          fmt("cooperative launch too large for the emulator: %llu blocks x %u threads = %llu threads, limit %u "
              "(every block must be alive at once; this limit describes the emulator, not a GPU)",
              (unsigned long long)B, T, (unsigned long long)(B * T), Limits::maxCooperativeThreads));
    W = B;
  } else if (kRace) {
    W = std::min<uint64_t>(B, std::max(1u, Limits::raceLiveThreads / T));
  }

  Rt rt;
  rt.grid = grid; rt.block = block; rt.T = T; rt.B = B; rt.cooperative = cooperative; rt.dynBytes = sharedBytes;
  rt.blocks = std::vector<BlockCtx>(size_t(W));
  rt.th = std::vector<ThreadRec>(size_t(W) * T);
  for (size_t s = 0; s < W; ++s) {
    BlockCtx& b = rt.blocks[s];
    b.first = unsigned(s * T);
    b.hbWarp.assign((T + 31) / 32 * 32, 0);
    b.slots.assign(g.sharedTypes.size(), nullptr);
    for (unsigned i = 0; i < T; ++i) {
      ThreadRec& t = rt.th[b.first + i];
      t.rt = &rt; t.blk = &b; t.tid = i;
    }
  }
  g.launches++;

  std::vector<std::thread> pool;
  pool.reserve(rt.th.size());
  bool spawned = true;
  try {
    for (ThreadRec& t : rt.th) pool.emplace_back(worker, &rt, &t, &call);
  } catch (const std::system_error&) {
    spawned = false;
  }

  if (spawned) {
    std::vector<unsigned> order(rt.th.size());
    for (unsigned i = 0; i < order.size(); ++i) order[i] = i;
    uint32_t rng = g.seed * 2654435761u + 12345u;
    for (uint64_t done = 0; done < B && !rt.aborting; done += W) {
      {
        Quiet q;
        std::lock_guard<std::mutex> lk(rt.m);
        rt.live = rt.running = rt.atGrid = 0;
        for (size_t s = 0; s < W; ++s) {
          BlockCtx& b = rt.blocks[s];
          const uint64_t n = done + s;
          b.active = n < B;
          b.live = b.atBarrier = 0;
          for (unsigned i = 0; i < T; ++i) {
            ThreadRec& t = rt.th[b.first + i];
            t.st = b.active ? ThreadRec::Ready : ThreadRec::Idle;
            t.go = t.started = false;
            t.wait = Wait::None;
          }
          if (!b.active) continue;
          b.rank = n;
          b.idx = uint3{unsigned(n % grid.x), unsigned((n / grid.x) % grid.y), unsigned(n / (uint64_t(grid.x) * grid.y))};
          b.live = T;
          rt.live += T;
          fresh_shared(b, sharedBytes);
        }
        rt.running = kRace ? rt.live : 0;
      }
      if (g.schedule == Schedule::Reverse) {
        for (unsigned i = 0; i < order.size(); ++i) order[i] = unsigned(order.size()) - 1 - i;
      } else if (g.schedule == Schedule::Shuffle) {
        for (unsigned i = unsigned(order.size()) - 1; i > 0; --i) {
          rng = rng * 1664525u + 1013904223u;
          std::swap(order[i], order[(rng >> 8) % (i + 1)]);
        }
      }
      run_wave(rt, order);
    }
  }

  {
    Quiet q;
    std::lock_guard<std::mutex> lk(rt.m);
    rt.shutdown = true;
    for (ThreadRec& t : rt.th) t.cv.notify_one();
  }
  for (std::thread& t : pool) t.join();
  {
    Quiet q;
    for (BlockCtx& b : rt.blocks) free_shared(b);
  }
  if (!spawned)
    return launch_error(cooperative ? cudaErrorCooperativeLaunchTooLarge : cudaErrorLaunchFailure,
        fmt("the operating system refused to create %zu threads for this launch", rt.th.size()));
  if (rt.aborting) return launch_error(cudaErrorLaunchFailure, "kernel stopped: " + rt.error);
  return cudaSuccess;
}

template <class R, class... P> inline const void* fn_id(R (*f)(P...)) { return reinterpret_cast<const void*>(f); }
template <class F> inline const void* fn_id(const F&) { return nullptr; }

// `kernel<<<grid, block, sharedBytes, stream>>>(args)` is rewritten by cu2emu.py to
// `::pmpp_emu::Launch(grid, block, sharedBytes, stream).run(kernel, args)`.
// The launch is synchronous: run() returns when every block has finished.
class Launch {
 public:
  Launch(dim3 grid, dim3 block, size_t sharedBytes = 0, cudaStream_t = nullptr)
      : grid_(grid), block_(block), shared_(sharedBytes) {}

  template <class F, class... A>
  void run(F kernel, A... args) {
    std::tuple<A...> packed(args...);
    std::function<void()> call = [&] { std::apply(kernel, packed); };
    launch_impl(grid_, block_, shared_, false, fn_id(kernel), call);
  }

 private:
  dim3 grid_, block_;
  size_t shared_;
};

// cudaLaunchCooperativeKernel receives the kernel as a void* and its arguments
// as an array of pointers, so the emulator needs the kernel's type from
// somewhere: cu2emu.py appends one KernelReg per non-template __global__
// function definition.
struct KernelReg {
  template <class... P>
  explicit KernelReg(void (*fn)(P...)) {
    S().kernels[reinterpret_cast<const void*>(fn)] = [fn](dim3 grid, dim3 block, void** args, size_t sharedBytes) {
      return invoke(fn, grid, block, args, sharedBytes, std::index_sequence_for<P...>{});
    };
  }
  template <class... P, size_t... I>
  static cudaError_t invoke(void (*fn)(P...), dim3 grid, dim3 block, void** args, size_t sharedBytes,
                            std::index_sequence<I...>) {
    (void)args;
    std::function<void()> call = [&] { fn(*static_cast<std::remove_reference_t<P>*>(args[I])...); };
    return launch_impl(grid, block, sharedBytes, true, reinterpret_cast<const void*>(fn), call);
  }
};

}  // namespace pmpp_emu
