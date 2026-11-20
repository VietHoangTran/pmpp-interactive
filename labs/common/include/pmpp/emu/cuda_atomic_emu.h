// pmpp CPU emulator: the part of libcu++'s <cuda/atomic> this course uses.
// Reached through `#include <cuda/atomic>` (pmpp/emu/shim/cuda/atomic).
//
// Names and signatures follow the CCCL sources (libcudacxx/include/cuda/atomic,
// cuda/__atomic/atomic.h, cuda/std/__atomic/{scopes,order}.h and api/common.h,
// main branch read on 2026-11-20). Modelled: the operations themselves, on 1-,
// 2-, 4- and 8-byte trivially copyable types. NOT modelled: the thread scope
// (accepted and ignored: every scope behaves the same on one CPU), wait/notify,
// and atomics on pointers.
#pragma once
#include <atomic>
#include <type_traits>

namespace cuda {
namespace std {
enum thread_scope {
  thread_scope_system = 0,
  thread_scope_device = 1,
  thread_scope_cluster = 3,
  thread_scope_block = 2,
  thread_scope_thread = 10
};
using memory_order = ::std::memory_order;
inline constexpr memory_order memory_order_relaxed = ::std::memory_order_relaxed;
inline constexpr memory_order memory_order_consume = ::std::memory_order_consume;
inline constexpr memory_order memory_order_acquire = ::std::memory_order_acquire;
inline constexpr memory_order memory_order_release = ::std::memory_order_release;
inline constexpr memory_order memory_order_acq_rel = ::std::memory_order_acq_rel;
inline constexpr memory_order memory_order_seq_cst = ::std::memory_order_seq_cst;
}  // namespace std

using std::thread_scope;
using std::thread_scope_block;
using std::thread_scope_cluster;
using std::thread_scope_device;
using std::thread_scope_system;
using std::thread_scope_thread;
using memory_order = std::memory_order;
inline constexpr memory_order memory_order_relaxed = std::memory_order_relaxed;
inline constexpr memory_order memory_order_consume = std::memory_order_consume;
inline constexpr memory_order memory_order_acquire = std::memory_order_acquire;
inline constexpr memory_order memory_order_release = std::memory_order_release;
inline constexpr memory_order memory_order_acq_rel = std::memory_order_acq_rel;
inline constexpr memory_order memory_order_seq_cst = std::memory_order_seq_cst;

namespace pmpp_detail {
inline int ord(memory_order m) { return static_cast<int>(m); }   // same numbering as __ATOMIC_*
inline int fail_ord(memory_order m) {
  return m == memory_order_acq_rel ? __ATOMIC_ACQUIRE : m == memory_order_release ? __ATOMIC_RELAXED : ord(m);
}

// The operations, on a pointer to the object. atomic<T> and atomic_ref<T> differ
// only in where that pointer comes from.
template <class T, class Self>
struct ops {
  using value_type = T;
  T* ptr() const noexcept { return static_cast<const Self*>(this)->pmpp_ptr(); }

  bool is_lock_free() const noexcept { return true; }
  T load(memory_order m = memory_order_seq_cst) const noexcept {
    T v;
    __atomic_load(ptr(), &v, ord(m));
    return v;
  }
  operator T() const noexcept { return load(); }
  void store(T d, memory_order m = memory_order_seq_cst) const noexcept { __atomic_store(ptr(), &d, ord(m)); }
  T exchange(T d, memory_order m = memory_order_seq_cst) const noexcept {
    T old;
    __atomic_exchange(ptr(), &d, &old, ord(m));
    return old;
  }
  bool compare_exchange_weak(T& e, T d, memory_order s, memory_order f) const noexcept {
    return __atomic_compare_exchange(ptr(), &e, &d, true, ord(s), ord(f));
  }
  bool compare_exchange_strong(T& e, T d, memory_order s, memory_order f) const noexcept {
    return __atomic_compare_exchange(ptr(), &e, &d, false, ord(s), ord(f));
  }
  bool compare_exchange_weak(T& e, T d, memory_order m = memory_order_seq_cst) const noexcept {
    return __atomic_compare_exchange(ptr(), &e, &d, true, ord(m), fail_ord(m));
  }
  bool compare_exchange_strong(T& e, T d, memory_order m = memory_order_seq_cst) const noexcept {
    return __atomic_compare_exchange(ptr(), &e, &d, false, ord(m), fail_ord(m));
  }
  // Each fetch_* returns the value held before the operation. (For fetch_min and
  // fetch_max that is what the CUDA guide states for the __nv_atomic_* built-ins
  // and for atomicMin/atomicMax; the libcu++ dispatch itself was not read.)
  T fetch_add(T op, memory_order m = memory_order_seq_cst) const noexcept {
    return ::pmpp_emu::atomic_rmw(ptr(), [op](T old) -> T { return ::pmpp_emu::wrap_add(old, op); }, ord(m));
  }
  T fetch_sub(T op, memory_order m = memory_order_seq_cst) const noexcept {
    return ::pmpp_emu::atomic_rmw(ptr(), [op](T old) -> T { return ::pmpp_emu::wrap_sub(old, op); }, ord(m));
  }
  T fetch_and(T op, memory_order m = memory_order_seq_cst) const noexcept {
    return ::pmpp_emu::atomic_rmw(ptr(), [op](T old) -> T { return old & op; }, ord(m));
  }
  T fetch_or(T op, memory_order m = memory_order_seq_cst) const noexcept {
    return ::pmpp_emu::atomic_rmw(ptr(), [op](T old) -> T { return old | op; }, ord(m));
  }
  T fetch_xor(T op, memory_order m = memory_order_seq_cst) const noexcept {
    return ::pmpp_emu::atomic_rmw(ptr(), [op](T old) -> T { return old ^ op; }, ord(m));
  }
  T fetch_max(const T& op, memory_order m = memory_order_seq_cst) const noexcept {
    return ::pmpp_emu::atomic_rmw(ptr(), [op](T old) -> T { return op > old ? op : old; }, ord(m));
  }
  T fetch_min(const T& op, memory_order m = memory_order_seq_cst) const noexcept {
    return ::pmpp_emu::atomic_rmw(ptr(), [op](T old) -> T { return op < old ? op : old; }, ord(m));
  }
  T operator++(int) const noexcept { return fetch_add(T(1)); }
  T operator--(int) const noexcept { return fetch_sub(T(1)); }
  T operator++() const noexcept { return ::pmpp_emu::wrap_add(fetch_add(T(1)), T(1)); }
  T operator--() const noexcept { return ::pmpp_emu::wrap_sub(fetch_sub(T(1)), T(1)); }
  T operator+=(T op) const noexcept { return ::pmpp_emu::wrap_add(fetch_add(op), op); }
  T operator-=(T op) const noexcept { return ::pmpp_emu::wrap_sub(fetch_sub(op), op); }
  T operator&=(T op) const noexcept { return fetch_and(op) & op; }
  T operator|=(T op) const noexcept { return fetch_or(op) | op; }
  T operator^=(T op) const noexcept { return fetch_xor(op) ^ op; }
};
}  // namespace pmpp_detail

template <class T, thread_scope Sco = thread_scope::thread_scope_system>
struct atomic_ref : pmpp_detail::ops<T, atomic_ref<T, Sco>> {
  static constexpr ::std::size_t required_alignment = sizeof(T);
  static constexpr bool is_always_lock_free = sizeof(T) <= 8;
  explicit constexpr atomic_ref(T& ref) : p_(&ref) {}
  atomic_ref(const atomic_ref&) noexcept = default;
  T operator=(T v) const noexcept { this->store(v); return v; }
  T* pmpp_ptr() const noexcept { return p_; }

 private:
  T* p_;
};

template <class T, thread_scope Sco = thread_scope::thread_scope_system>
struct atomic : pmpp_detail::ops<T, atomic<T, Sco>> {
  constexpr atomic() noexcept = default;
  constexpr atomic(T d) noexcept : v_(d) {}
  atomic(const atomic&) = delete;
  atomic& operator=(const atomic&) = delete;
  T operator=(T d) noexcept { this->store(d); return d; }
  T* pmpp_ptr() const noexcept { return const_cast<T*>(&v_); }

 private:
  T v_{};
};

}  // namespace cuda
