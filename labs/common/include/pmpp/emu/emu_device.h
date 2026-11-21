// pmpp CPU emulator, part 3 of 4: what kernel code calls.
// Included by pmpp/emu/cuda_emu.h; do not include directly.
//
// Signatures follow the CUDA Programming Guide 13.5, "C/C++ Language Extensions"
// (checked 2026-11-20), with one addition: the synchronising calls take two
// hidden trailing arguments that capture the call site, so that the emulator
// can tell "the same call" from "the same kind of call" and name source lines
// in its messages.
#pragma once

#define PMPP_EMU_SITE int pmpp_line = __builtin_LINE(), const char* pmpp_file = __builtin_FILE()

namespace pmpp_emu {

// ---- block barrier ------------------------------------------------------------
// Returns the number of participants whose predicate was non-zero; *n receives
// the number of participants.
inline unsigned block_sync(int predicate, unsigned* n, const char* what, const char* file, int line) {
  ThreadRec& me = self(what);
  Rt& rt = *me.rt;
  // Two tokens, alternating per barrier. With one token, a thread still leaving barrier k
  // would acquire what a neighbour released on arriving at barrier k+1, and every access
  // the neighbour made between the two barriers would wrongly look ordered before this
  // thread's. No live thread can reach barrier k+2 before all have left barrier k, so two
  // are enough.
  const void* token = &me.blk->hbBarrier[me.nBar & 1];
  hb_before(token);
  unsigned out;
  {
    Quiet q;
    std::unique_lock<std::mutex> lk(rt.m);
    me.op = Op::None; me.in = predicate != 0; me.what = what; me.file = file; me.line = line;
    arrive(rt, me, lk, Wait::Block);
    out = unsigned(me.out);
    if (n) *n = unsigned(me.arg);
  }
  hb_after(token);
  me.nBar++;
  return out;
}

inline void grid_sync(const char* what, const char* file, int line) {
  ThreadRec& me = self(what);
  Rt& rt = *me.rt;
  const void* token = &rt.hbGrid[me.nGrid & 1];   // alternating, as in block_sync
  hb_before(token);
  {
    Quiet q;
    std::unique_lock<std::mutex> lk(rt.m);
    me.what = what; me.file = file; me.line = line;
    if (!rt.cooperative)
      fail(rt, std::string(where(me)) + ": synchronising the whole grid needs a kernel launched with "
               "cudaLaunchCooperativeKernel, not <<< >>> (grid_group::is_valid() is false here)");
    me.op = Op::None;
    arrive(rt, me, lk, Wait::Grid);
  }
  hb_after(token);
  me.nGrid++;
}

// ---- warp rendezvous ------------------------------------------------------------
// Publish `in`, wait until every non-exited lane named in `mask` is in the same
// call, return this lane's result. `arg` is srcLane / delta / laneMask.
inline uint64_t warp_call(Op op, const char* what, unsigned mask, uint64_t in, unsigned size, int arg, int width,
                          const char* file, int line, int* flag = nullptr) {
  ThreadRec& me = self(what);
  Rt& rt = *me.rt;
  const unsigned lane = me.tid % 32;
  // Only __syncwarp orders memory accesses; votes and shuffles do not.
  const void* token = op == Op::SyncWarp ? &me.blk->hbWarp[me.tid / 32 * 32 + unsigned(__builtin_ctz(mask | 0x80000000u))] : nullptr;
  if (token) hb_before(token);
  uint64_t out;
  {
    Quiet q;
    std::unique_lock<std::mutex> lk(rt.m);
    me.op = op; me.what = what; me.mask = mask; me.in = in; me.size = size; me.arg = arg; me.width = width;
    me.file = file; me.line = line;
    if (op != Op::Active && !(mask >> lane & 1))
      fail(rt, who(me) + " called " + where(me) + fmt(" with mask 0x%08x, which does not include the calling "
               "thread's own lane; that is undefined on a GPU", mask));
    if ((op == Op::Shfl || op == Op::ShflUp || op == Op::ShflDown || op == Op::ShflXor) &&
        (width < 1 || width > 32 || (width & (width - 1)) != 0))
      fail(rt, where(me) + fmt(": width %d is not a power of two in [1, 32]; the result is undefined on a GPU", width));
    arrive(rt, me, lk, Wait::Warp);
    out = me.out;
    if (flag) *flag = me.arg;
  }
  if (token) hb_after(token);
  return out;
}

template <class T>
inline T shuffle(Op op, const char* what, unsigned mask, T value, int arg, int width, const char* file, int line) {
  static_assert(std::is_trivially_copyable<T>::value && sizeof(T) <= 8, "unsupported operand type");
  uint64_t in = 0;
  std::memcpy(&in, &value, sizeof(T));
  const uint64_t out = warp_call(op, what, mask, in, unsigned(sizeof(T)), arg, width, file, line);
  T r;
  std::memcpy(&r, &out, sizeof(T));
  return r;
}

template <class T>
inline unsigned match(Op op, const char* what, unsigned mask, T value, int* pred, const char* file, int line) {
  uint64_t in = 0;
  std::memcpy(&in, &value, sizeof(T));
  int flag = 0;
  const unsigned out = unsigned(warp_call(op, what, mask, in, unsigned(sizeof(T)), 0, 32, file, line, &flag));
  if (pred) *pred = flag;
  return out;
}

// ---- atomics --------------------------------------------------------------------
// One compare-and-swap loop serves every operation and type. In the race
// flavour ThreadSanitizer sees these as atomic accesses; a plain access to the
// same location from another thread is reported.
template <class T, class F>
inline T atomic_rmw(T* p, F f, int order = __ATOMIC_RELAXED) {
  static_assert(std::is_trivially_copyable<T>::value &&
                (sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8),
                "the emulator supports atomics on 1-, 2-, 4- and 8-byte trivially copyable types");
  T old;
  __atomic_load(p, &old, __ATOMIC_RELAXED);
  for (;;) {
    T desired = f(old);
    if (__atomic_compare_exchange(p, &old, &desired, true, order, __ATOMIC_RELAXED)) return old;
  }
}
template <class T> inline T wrap_add(T a, T b) {
  if constexpr (std::is_integral<T>::value) return T(std::make_unsigned_t<T>(a) + std::make_unsigned_t<T>(b));
  else return a + b;
}
template <class T> inline T wrap_sub(T a, T b) {
  if constexpr (std::is_integral<T>::value) return T(std::make_unsigned_t<T>(a) - std::make_unsigned_t<T>(b));
  else return a - b;
}

}  // namespace pmpp_emu

// ---- synchronisation primitives (guide 5.4.4) -----------------------------------
inline void __syncthreads(PMPP_EMU_SITE) { ::pmpp_emu::block_sync(0, nullptr, "__syncthreads", pmpp_file, pmpp_line); }
inline int __syncthreads_count(int predicate, PMPP_EMU_SITE) {
  return int(::pmpp_emu::block_sync(predicate, nullptr, "__syncthreads_count", pmpp_file, pmpp_line));
}
inline int __syncthreads_and(int predicate, PMPP_EMU_SITE) {
  unsigned n = 0;
  const unsigned c = ::pmpp_emu::block_sync(predicate, &n, "__syncthreads_and", pmpp_file, pmpp_line);
  return c == n;
}
inline int __syncthreads_or(int predicate, PMPP_EMU_SITE) {
  return ::pmpp_emu::block_sync(predicate, nullptr, "__syncthreads_or", pmpp_file, pmpp_line) != 0;
}
inline void __syncwarp(unsigned mask = 0xFFFFFFFF, PMPP_EMU_SITE) {
  ::pmpp_emu::warp_call(::pmpp_emu::Op::SyncWarp, "__syncwarp", mask, 0, 0, 0, 32, pmpp_file, pmpp_line);
}

// ---- warp vote, match and shuffle (guide 5.4.6) ---------------------------------
inline int __all_sync(unsigned mask, int predicate, PMPP_EMU_SITE) {
  return int(::pmpp_emu::warp_call(::pmpp_emu::Op::All, "__all_sync", mask, predicate != 0, 0, 0, 32, pmpp_file, pmpp_line));
}
inline int __any_sync(unsigned mask, int predicate, PMPP_EMU_SITE) {
  return int(::pmpp_emu::warp_call(::pmpp_emu::Op::Any, "__any_sync", mask, predicate != 0, 0, 0, 32, pmpp_file, pmpp_line));
}
inline unsigned __ballot_sync(unsigned mask, int predicate, PMPP_EMU_SITE) {
  return unsigned(::pmpp_emu::warp_call(::pmpp_emu::Op::Ballot, "__ballot_sync", mask, predicate != 0, 0, 0, 32, pmpp_file, pmpp_line));
}
// See cuda_emu.h: this is the set of lanes that reach the same __activemask()
// call together, which a GPU is free to make smaller.
inline unsigned __activemask(PMPP_EMU_SITE) {
  return unsigned(::pmpp_emu::warp_call(::pmpp_emu::Op::Active, "__activemask", 0, 0, 0, 0, 32, pmpp_file, pmpp_line));
}

#define PMPP_EMU_WARP_TYPE(T)                                                                                   \
  inline T __shfl_sync(unsigned mask, T value, int srcLane, int width = warpSize, PMPP_EMU_SITE) {             \
    return ::pmpp_emu::shuffle(::pmpp_emu::Op::Shfl, "__shfl_sync", mask, value, srcLane, width, pmpp_file, pmpp_line); } \
  inline T __shfl_up_sync(unsigned mask, T value, unsigned delta, int width = warpSize, PMPP_EMU_SITE) {       \
    return ::pmpp_emu::shuffle(::pmpp_emu::Op::ShflUp, "__shfl_up_sync", mask, value, int(delta), width, pmpp_file, pmpp_line); } \
  inline T __shfl_down_sync(unsigned mask, T value, unsigned delta, int width = warpSize, PMPP_EMU_SITE) {     \
    return ::pmpp_emu::shuffle(::pmpp_emu::Op::ShflDown, "__shfl_down_sync", mask, value, int(delta), width, pmpp_file, pmpp_line); } \
  inline T __shfl_xor_sync(unsigned mask, T value, int laneMask, int width = warpSize, PMPP_EMU_SITE) {        \
    return ::pmpp_emu::shuffle(::pmpp_emu::Op::ShflXor, "__shfl_xor_sync", mask, value, laneMask, width, pmpp_file, pmpp_line); } \
  inline unsigned __match_any_sync(unsigned mask, T value, PMPP_EMU_SITE) {                                    \
    return ::pmpp_emu::match(::pmpp_emu::Op::MatchAny, "__match_any_sync", mask, value, nullptr, pmpp_file, pmpp_line); } \
  inline unsigned __match_all_sync(unsigned mask, T value, int* pred, PMPP_EMU_SITE) {                         \
    return ::pmpp_emu::match(::pmpp_emu::Op::MatchAll, "__match_all_sync", mask, value, pred, pmpp_file, pmpp_line); }
PMPP_EMU_WARP_TYPE(int)
PMPP_EMU_WARP_TYPE(unsigned)
PMPP_EMU_WARP_TYPE(long)
PMPP_EMU_WARP_TYPE(unsigned long)
PMPP_EMU_WARP_TYPE(long long)
PMPP_EMU_WARP_TYPE(unsigned long long)
PMPP_EMU_WARP_TYPE(float)
PMPP_EMU_WARP_TYPE(double)
#undef PMPP_EMU_WARP_TYPE

// ---- integer intrinsics ----------------------------------------------------------
// UNVERIFIED signatures: these are not on the two guide pages that were checked;
// they follow my recollection of device_functions.h. __clz(0) is taken to be 32,
// __clzll(0) 64 and __ffs(0) 0.
inline int __popc(unsigned x) { return __builtin_popcount(x); }
inline int __popcll(unsigned long long x) { return __builtin_popcountll(x); }
inline int __clz(int x) { return x ? __builtin_clz(unsigned(x)) : 32; }
inline int __clzll(long long x) { return x ? __builtin_clzll(static_cast<unsigned long long>(x)) : 64; }
inline int __ffs(int x) { return __builtin_ffs(x); }
inline int __ffsll(long long x) { return __builtin_ffsll(x); }
inline unsigned __brev(unsigned x) {
  unsigned r = 0;
  for (int i = 0; i < 32; ++i) r |= (x >> i & 1u) << (31 - i);
  return r;
}
inline unsigned long long __brevll(unsigned long long x) {
  unsigned long long r = 0;
  for (int i = 0; i < 64; ++i) r |= (x >> i & 1ull) << (63 - i);
  return r;
}

// ---- more integer intrinsics, from the CUDA Math API 13.5 ----------------------------
// "Integer Intrinsics" (group__CUDA__MATH__INTRINSIC__INT), read 2026-11-21
// (notes/today/infra-facts.md E2, E3). Same-signedness __dp4a overloads only, as NVIDIA
// documents them; the int32 accumulation wraps modulo 2^32 here (two's complement), which the
// page does not state. HIP has no __dp4a (x2-facts B1): code meant for HIP must not call it.
namespace pmpp_emu {
inline int dp4a_s(uint32_t a, uint32_t b, int32_t c) {
  uint32_t acc = uint32_t(c);
  for (int k = 0; k < 4; ++k)
    acc += uint32_t(int32_t(int8_t(uint8_t(a >> 8 * k))) * int32_t(int8_t(uint8_t(b >> 8 * k))));
  return int32_t(acc);
}
inline unsigned dp4a_u(uint32_t a, uint32_t b, uint32_t c) {
  for (int k = 0; k < 4; ++k) c += uint32_t(uint8_t(a >> 8 * k)) * uint32_t(uint8_t(b >> 8 * k));
  return c;
}
template <class V> inline uint32_t pack4(V v) {
  return uint32_t(uint8_t(v.x)) | uint32_t(uint8_t(v.y)) << 8 | uint32_t(uint8_t(v.z)) << 16 | uint32_t(uint8_t(v.w)) << 24;
}
}  // namespace pmpp_emu
inline int __dp4a(int srcA, int srcB, int c) { return ::pmpp_emu::dp4a_s(uint32_t(srcA), uint32_t(srcB), c); }
inline int __dp4a(char4 srcA, char4 srcB, int c) {
  return ::pmpp_emu::dp4a_s(::pmpp_emu::pack4(srcA), ::pmpp_emu::pack4(srcB), c);
}
inline unsigned int __dp4a(unsigned int srcA, unsigned int srcB, unsigned int c) { return ::pmpp_emu::dp4a_u(srcA, srcB, c); }
inline unsigned int __dp4a(uchar4 srcA, uchar4 srcB, unsigned int c) {
  return ::pmpp_emu::dp4a_u(::pmpp_emu::pack4(srcA), ::pmpp_emu::pack4(srcB), c);
}
// Byte n of the result is byte ((s >> 4n) & 7) of the 8-byte value (y << 32) | x; the other
// bits of s are not read (the page's own pseudo-code).
inline unsigned int __byte_perm(unsigned int x, unsigned int y, unsigned int s) {
  const uint64_t src = uint64_t(y) << 32 | x;
  unsigned r = 0;
  for (int n = 0; n < 4; ++n) r |= unsigned(src >> 8 * ((s >> 4 * n) & 7u) & 0xFFu) << 8 * n;
  return r;
}
inline unsigned int __umulhi(unsigned int x, unsigned int y) { return unsigned((uint64_t(x) * y) >> 32); }
inline int __mulhi(int x, int y) { return int(uint32_t(uint64_t(int64_t(x) * y) >> 32)); }
inline unsigned long long __umul64hi(unsigned long long x, unsigned long long y) {
  return (unsigned long long)((unsigned __int128)x * y >> 64);
}

// ---- type casting intrinsics (group__CUDA__MATH__INTRINSIC__CAST, E4) -------------------------
inline unsigned int __float_as_uint(float x) { unsigned u; std::memcpy(&u, &x, 4); return u; }
inline float __uint_as_float(unsigned int x) { float f; std::memcpy(&f, &x, 4); return f; }
inline int __float_as_int(float x) { int i; std::memcpy(&i, &x, 4); return i; }
inline float __int_as_float(int x) { float f; std::memcpy(&f, &x, 4); return f; }

// ---- single-precision math (E5) ----------------------------------------------------------------
// expf, exp2f, logf, log2f, tanhf, erff, fmaf, fminf, fmaxf, fabsf, sqrtf, powf, copysignf
// come from the host's <cmath>: the host library's accuracy, not NVIDIA's documented error
// bounds. rsqrtf is ours (1 / sqrt). The fast intrinsics (__expf, __fdividef, ...) are
// computed with the accurate host functions: the GPU's approximation error, and its
// flush-to-zero and range behaviour, are NOT modelled. The *_rn intrinsics are left out: lab
// A.1 defines __fadd_rn and __fsub_rn for the emulator itself (labs/appA/lab1_sumerror/sumerror.h).
// glibc's <math.h> declares __expf, __logf, __powf, __sinf and __cosf itself (its internal
// aliases, with C linkage), and newer C libraries may declare rsqrtf (C23). Defining C++
// functions of the same names would clash there, so the emulator's versions live in pmpp_emu
// and these names are macros for them, defined after every standard header was included.
namespace pmpp_emu {
inline float math_rsqrtf(float x) { return 1.0f / std::sqrt(x); }
inline float math_expf(float x) { return std::exp(x); }
inline float math_logf(float x) { return std::log(x); }
inline float math_powf(float x, float y) { return std::pow(x, y); }
inline float math_sinf(float x) { return std::sin(x); }
inline float math_cosf(float x) { return std::cos(x); }
inline float math_fdividef(float x, float y) { return x / y; }
// NaN handling was not checked against NVIDIA's documentation (it gives +0 here).
inline float math_saturatef(float x) { return x >= 1.0f ? 1.0f : (x > 0.0f ? x : 0.0f); }
}  // namespace pmpp_emu
#define rsqrtf ::pmpp_emu::math_rsqrtf
#define __expf ::pmpp_emu::math_expf
#define __logf ::pmpp_emu::math_logf
#define __powf ::pmpp_emu::math_powf
#define __sinf ::pmpp_emu::math_sinf
#define __cosf ::pmpp_emu::math_cosf
#define __fdividef ::pmpp_emu::math_fdividef
#define __saturatef ::pmpp_emu::math_saturatef

// ---- legacy atomic functions (guide 5.4.5.1) -------------------------------------
// Exactly the overloads the guide lists, minus __half/__half2/__nv_bfloat16 and
// the float2/float4 and 128-bit forms. Each returns the old value.
#define PMPP_EMU_ATOMIC(NAME, T, EXPR)                                              \
  inline T NAME(T* address, T val) {                                                \
    return ::pmpp_emu::atomic_rmw(address, [val](T old) -> T { (void)old; return EXPR; }); }
PMPP_EMU_ATOMIC(atomicAdd, int, ::pmpp_emu::wrap_add(old, val))
PMPP_EMU_ATOMIC(atomicAdd, unsigned, old + val)
PMPP_EMU_ATOMIC(atomicAdd, unsigned long long, old + val)
PMPP_EMU_ATOMIC(atomicAdd, float, old + val)
PMPP_EMU_ATOMIC(atomicAdd, double, old + val)
PMPP_EMU_ATOMIC(atomicSub, int, ::pmpp_emu::wrap_sub(old, val))
PMPP_EMU_ATOMIC(atomicSub, unsigned, old - val)
PMPP_EMU_ATOMIC(atomicExch, int, val)
PMPP_EMU_ATOMIC(atomicExch, unsigned, val)
PMPP_EMU_ATOMIC(atomicExch, unsigned long long, val)
PMPP_EMU_ATOMIC(atomicExch, float, val)
PMPP_EMU_ATOMIC(atomicMin, int, val < old ? val : old)
PMPP_EMU_ATOMIC(atomicMin, unsigned, val < old ? val : old)
PMPP_EMU_ATOMIC(atomicMin, unsigned long long, val < old ? val : old)
PMPP_EMU_ATOMIC(atomicMin, long long, val < old ? val : old)
PMPP_EMU_ATOMIC(atomicMax, int, val > old ? val : old)
PMPP_EMU_ATOMIC(atomicMax, unsigned, val > old ? val : old)
PMPP_EMU_ATOMIC(atomicMax, unsigned long long, val > old ? val : old)
PMPP_EMU_ATOMIC(atomicMax, long long, val > old ? val : old)
PMPP_EMU_ATOMIC(atomicInc, unsigned, old >= val ? 0 : (old + 1))
PMPP_EMU_ATOMIC(atomicDec, unsigned, (old == 0 || old > val) ? val : (old - 1))
PMPP_EMU_ATOMIC(atomicAnd, int, old & val)
PMPP_EMU_ATOMIC(atomicAnd, unsigned, old & val)
PMPP_EMU_ATOMIC(atomicAnd, unsigned long long, old & val)
PMPP_EMU_ATOMIC(atomicOr, int, old | val)
PMPP_EMU_ATOMIC(atomicOr, unsigned, old | val)
PMPP_EMU_ATOMIC(atomicOr, unsigned long long, old | val)
PMPP_EMU_ATOMIC(atomicXor, int, old ^ val)
PMPP_EMU_ATOMIC(atomicXor, unsigned, old ^ val)
PMPP_EMU_ATOMIC(atomicXor, unsigned long long, old ^ val)
#undef PMPP_EMU_ATOMIC
#define PMPP_EMU_CAS(T)                                                             \
  inline T atomicCAS(T* address, T compare, T val) {                                \
    return ::pmpp_emu::atomic_rmw(address, [compare, val](T old) -> T { return old == compare ? val : old; }); }
PMPP_EMU_CAS(int)
PMPP_EMU_CAS(unsigned)
PMPP_EMU_CAS(unsigned long long)
PMPP_EMU_CAS(unsigned short)
#undef PMPP_EMU_CAS
