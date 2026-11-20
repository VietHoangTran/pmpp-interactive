// pmpp CPU emulator: the subset of <cooperative_groups.h> this course uses.
// Reached through `#include <cooperative_groups.h>` (pmpp/emu/shim).
//
// Member lists follow the CUDA Programming Guide 13.5, section 5.7.3
// "Cooperative Groups API" (checked 2026-11-20). Provided: thread_block,
// grid_group, coalesced_group, thread_block_tile<Size> for Size <= 32 over a
// thread_block, and cooperative_groups::sync(). NOT provided: cluster groups,
// barrier_arrive/barrier_wait, tiles larger than a warp or tiles of tiles,
// binary_partition, labeled_partition, match_any/match_all on groups, and the
// reduce / scan / memcpy_async / invoke_one headers.
//
// As with the warp intrinsics, every synchronising member takes two hidden
// trailing arguments that record the call site.
#pragma once

namespace cooperative_groups {

namespace pmpp_detail {
inline unsigned rank_in(unsigned mask, unsigned lane) { return unsigned(__builtin_popcount(mask & ((1u << lane) - 1u))); }
inline unsigned nth_lane(unsigned mask, unsigned n) {      // lane of the n-th set bit; 32 if there is none
  for (unsigned l = 0; l < 32; ++l)
    if (mask >> l & 1) { if (n == 0) return l; --n; }
  return 32;
}
inline unsigned pack(unsigned mask, unsigned bits) {        // bit r of the result = bit of the rank-r member
  unsigned out = 0, r = 0;
  for (unsigned l = 0; l < 32; ++l)
    if (mask >> l & 1) { if (bits >> l & 1) out |= 1u << r; ++r; }
  return out;
}
inline unsigned my_lane() { return ::pmpp_emu::self("a cooperative group").tid % 32; }

// What coalesced_group and thread_block_tile share: a set of lanes of one warp.
class warp_subset {
 public:
  void sync(PMPP_EMU_SITE) const {
    ::pmpp_emu::warp_call(::pmpp_emu::Op::SyncWarp, "group.sync", mask_, 0, 0, 0, 32, pmpp_file, pmpp_line);
  }
  unsigned pmpp_mask() const { return mask_; }     // emulator only: the lanes of this group
  unsigned long long num_threads() const { return unsigned(__builtin_popcount(mask_)); }
  unsigned long long size() const { return num_threads(); }
  unsigned long long thread_rank() const { return rank_in(mask_, my_lane()); }
  template <class T> T shfl(T var, unsigned int src_rank, PMPP_EMU_SITE) const {
    const unsigned n = unsigned(num_threads());
    return ::pmpp_emu::shuffle(::pmpp_emu::Op::Shfl, "group.shfl", mask_, var, int(nth_lane(mask_, src_rank % n)), 32,
                               pmpp_file, pmpp_line);
  }
  template <class T> T shfl_up(T var, int delta, PMPP_EMU_SITE) const {
    const unsigned me = my_lane(), r = rank_in(mask_, me);
    const unsigned src = (delta >= 0 && unsigned(delta) <= r) ? nth_lane(mask_, r - unsigned(delta)) : me;
    return ::pmpp_emu::shuffle(::pmpp_emu::Op::Shfl, "group.shfl_up", mask_, var, int(src), 32, pmpp_file, pmpp_line);
  }
  template <class T> T shfl_down(T var, int delta, PMPP_EMU_SITE) const {
    const unsigned me = my_lane(), r = rank_in(mask_, me), n = unsigned(num_threads());
    const unsigned src = (delta >= 0 && r + unsigned(delta) < n) ? nth_lane(mask_, r + unsigned(delta)) : me;
    return ::pmpp_emu::shuffle(::pmpp_emu::Op::Shfl, "group.shfl_down", mask_, var, int(src), 32, pmpp_file, pmpp_line);
  }
  int any(int predicate, PMPP_EMU_SITE) const {
    return int(::pmpp_emu::warp_call(::pmpp_emu::Op::Any, "group.any", mask_, predicate != 0, 0, 0, 32, pmpp_file, pmpp_line));
  }
  int all(int predicate, PMPP_EMU_SITE) const {
    return int(::pmpp_emu::warp_call(::pmpp_emu::Op::All, "group.all", mask_, predicate != 0, 0, 0, 32, pmpp_file, pmpp_line));
  }
  unsigned int ballot(int predicate, PMPP_EMU_SITE) const {
    return pack(mask_, unsigned(::pmpp_emu::warp_call(::pmpp_emu::Op::Ballot, "group.ballot", mask_, predicate != 0, 0, 0, 32,
                                                      pmpp_file, pmpp_line)));
  }

 protected:
  explicit warp_subset(unsigned mask) : mask_(mask) {}
  unsigned mask_;
};
}  // namespace pmpp_detail

// ---- the whole block ---------------------------------------------------------------
class thread_block {
 public:
  static void sync(PMPP_EMU_SITE) { ::pmpp_emu::block_sync(0, nullptr, "thread_block::sync", pmpp_file, pmpp_line); }
  static unsigned int thread_rank() { return ::pmpp_emu::self("thread_block::thread_rank").tid; }
  static dim3 group_index() { return dim3(blockIdx); }
  static dim3 thread_index() { return dim3(threadIdx); }
  static dim3 dim_threads() { return blockDim; }
  static unsigned int num_threads() { return blockDim.x * blockDim.y * blockDim.z; }
  static unsigned int size() { return num_threads(); }
  static dim3 group_dim() { return dim_threads(); }

 private:
  thread_block() {}
  friend thread_block this_thread_block();
};
inline thread_block this_thread_block() {
  ::pmpp_emu::self("cooperative_groups::this_thread_block");
  return thread_block();
}

// ---- the whole grid (needs cudaLaunchCooperativeKernel) ------------------------------
class grid_group {
 public:
  bool is_valid() const { return ::pmpp_emu::self("grid_group::is_valid").rt->cooperative; }
  void sync(PMPP_EMU_SITE) const { ::pmpp_emu::grid_sync("grid_group::sync", pmpp_file, pmpp_line); }
  static unsigned long long thread_rank() {
    const ::pmpp_emu::ThreadRec& me = ::pmpp_emu::self("grid_group::thread_rank");
    return me.blk->rank * me.rt->T + me.tid;
  }
  static unsigned long long block_rank() { return ::pmpp_emu::self("grid_group::block_rank").blk->rank; }
  static unsigned long long num_threads() { return num_blocks() * (blockDim.x * blockDim.y * blockDim.z); }
  static unsigned long long num_blocks() { return (unsigned long long)gridDim.x * gridDim.y * gridDim.z; }
  static dim3 dim_blocks() { return gridDim; }
  static dim3 block_index() { return dim3(blockIdx); }
  static unsigned long long size() { return num_threads(); }
  static dim3 group_dim() { return dim_blocks(); }

 private:
  grid_group() {}
  friend grid_group this_grid();
};
inline grid_group this_grid() {
  ::pmpp_emu::self("cooperative_groups::this_grid");
  return grid_group();
}

// ---- the threads of a warp that are here together -------------------------------------
// See cuda_emu.h for what "together" means in the emulator.
class coalesced_group : public pmpp_detail::warp_subset {
 public:
  unsigned long long meta_group_size() const { return 1; }
  unsigned long long meta_group_rank() const { return 0; }

 private:
  explicit coalesced_group(unsigned mask) : warp_subset(mask) {}
  friend coalesced_group coalesced_threads(int, const char*);
};
inline coalesced_group coalesced_threads(PMPP_EMU_SITE) {
  return coalesced_group(unsigned(::pmpp_emu::warp_call(::pmpp_emu::Op::Active, "cooperative_groups::coalesced_threads",
                                                        0, 0, 0, 0, 32, pmpp_file, pmpp_line)));
}

// ---- tiles of 1, 2, 4, 8, 16 or 32 consecutive threads ---------------------------------
template <unsigned int Size, class ParentT = void>
class thread_block_tile : public pmpp_detail::warp_subset {
  static_assert(Size >= 1 && Size <= 32 && (Size & (Size - 1)) == 0,
                "the emulator supports tiled_partition<Size> for Size = 1, 2, 4, 8, 16 or 32");

 public:
  unsigned long long meta_group_size() const { return (blockDim.x * blockDim.y * blockDim.z + Size - 1) / Size; }
  unsigned long long meta_group_rank() const { return ::pmpp_emu::self("thread_block_tile").tid / Size; }
  template <class T> T shfl_xor(T var, int laneMask, PMPP_EMU_SITE) const {
    return ::pmpp_emu::shuffle(::pmpp_emu::Op::ShflXor, "group.shfl_xor", mask_, var, laneMask, int(Size), pmpp_file, pmpp_line);
  }
  // `thread_block_tile<32> t = tiled_partition<32>(block);` drops the parent type.
  template <class P> thread_block_tile(const thread_block_tile<Size, P>& o) : warp_subset(o.pmpp_mask()) {}
  // Used by tiled_partition; not part of the CUDA API.
  explicit thread_block_tile(const thread_block&) : warp_subset(make_mask()) {}

 private:
  static unsigned make_mask() {
    const ::pmpp_emu::ThreadRec& me = ::pmpp_emu::self("cooperative_groups::tiled_partition");
    const unsigned lane = me.tid % 32, first = lane / Size * Size;
    const unsigned inWarp = ::std::min(32u, me.rt->T - me.tid / 32 * 32);   // the last warp of a block may be short
    unsigned mask = 0;
    for (unsigned l = first; l < first + Size && l < inWarp; ++l) mask |= 1u << l;
    return mask;
  }
};
template <unsigned int Size>
inline thread_block_tile<Size, thread_block> tiled_partition(const thread_block& g) {
  return thread_block_tile<Size, thread_block>(g);
}

template <class T> inline void sync(T& group, PMPP_EMU_SITE) { group.sync(pmpp_line, pmpp_file); }

}  // namespace cooperative_groups
