// pmpp CPU emulator for the subset of CUDA used in this course.
//
// This is NOT a GPU and NOT a CUDA implementation. It exists so that kernel
// logic (index arithmetic, shared-memory use, barrier placement, warp-level
// data movement, atomics, bounds) can be executed and tested on a machine with
// no CUDA or HIP device. Passing here does not prove a kernel runs on a GPU.
//
// It is force-included into every .cu file after labs/tools/cu2emu.py has
// rewritten the three pieces of CUDA syntax that are not C++ (kernel launches,
// __shared__ declarations, and a registration line per kernel).
//
// TWO FLAVOURS, one source (the build picks; see labs/cmake/Backends.cmake):
//
//   emu   One emulated thread runs at a time. A thread keeps running until it
//         reaches a synchronisation point (__syncthreads, a warp *_sync call,
//         a cooperative-groups sync) or returns; then the next one runs. Each
//         test is repeated under three deterministic orders (forward, reverse,
//         shuffled), so a missing barrier gives a wrong answer every time.
//         Built with AddressSanitizer: an out-of-bounds access in a kernel is
//         reported with its source line.
//         CANNOT see: a missing atomic. `bins[x]++` from many threads is always
//         "correct" here because no thread is interrupted in the middle of it.
//
//   race  The threads of a block, and several blocks, run truly concurrently
//         as OS threads under ThreadSanitizer. An unsynchronised access to the
//         same memory from two emulated threads is reported with both source
//         lines, whether or not it happened to corrupt the result in that run.
//         CANNOT see: out-of-bounds accesses (no AddressSanitizer), races
//         between the same thread index of two blocks that ran one after the
//         other on the same OS thread, or anything that depends on a schedule
//         that did not occur. ThreadSanitizer prints each pair of racing source
//         lines once per process, so a later test of the same kernel may pass.
//         Scheduling is the operating system's: a wrong answer caused by a race
//         is not reproducible, and a clean run does not prove there is no race.
//         Blocks run concurrently in groups whose threads fit in 256 OS threads
//         (one block at a time for blocks larger than 128 threads).
//
// NEVER modelled, in either flavour (do not draw conclusions about these):
//   timing, bandwidth, caches, memory coalescing, shared-memory bank conflicts,
//   occupancy, register use, the cost of divergence, instruction scheduling
//   inside a warp. The "device properties" are synthetic. Host and device share
//   one address space: cudaMemcpy is a memcpy, and dereferencing a device
//   pointer on the host is not caught.
//
// FEATURE BY FEATURE: what is modelled, and what is not.
//
// Threads, blocks, grids
//   modelled:     threadIdx, blockIdx, blockDim, gridDim; private automatic variables per thread; launch
//                 configuration checks: at most 1024 threads per block, and CUDA's per-dimension limits
//                 (block at most 1024 x 1024 x 64, grid at most (2^31 - 1) x 65535 x 65535; Programming
//                 Guide 13.5, Table 33), refused with cudaErrorInvalidConfiguration and a message.
//   not modelled: the order and overlap of blocks on a GPU. In emu, blocks run one after another in index
//                 order.
// __shared__ variables and dynamic shared memory
//   modelled:     one copy per block, freshly allocated and filled with 0xCD for every block (reading
//                 shared memory nobody wrote gives garbage, as on a GPU); `extern __shared__ T a[];` sized by
//                 the third launch parameter, up to 48 KiB, or 96 KiB after cudaFuncSetAttribute(
//                 cudaFuncAttributeMaxDynamicSharedMemorySize). Out-of-bounds accesses are caught in emu.
//   not modelled: the total size limit of statically declared shared memory; alignment and placement of
//                 several shared arrays relative to each other; bank conflicts.
// __syncthreads, __syncthreads_count / _and / _or
//   modelled:     a barrier across the live (non-exited) threads of a block; a barrier that can never be
//                 passed is reported as a deadlock with the source lines involved instead of hanging.
//   not modelled: detection of barriers in divergent code that happen to pair up (thread A at one
//                 __syncthreads, thread B at another): they are treated as the same barrier.
// Warps: warpSize, lane numbering
//   modelled:     warpSize == 32; warp w of a block is the 32 consecutive threads 32w .. 32w+31 in linear
//                 thread order (x fastest, then y, then z); the last warp of a block may be partial.
//   not modelled: any execution property of a warp (lockstep, divergence cost, scheduling).
// __shfl_sync, __shfl_up_sync, __shfl_down_sync, __shfl_xor_sync, __ballot_sync, __any_sync, __all_sync,
// __match_any_sync, __match_all_sync, __syncwarp
//   modelled:     a rendezvous: each call waits until every non-exited lane named in its mask is in the same
//                 call (same source line, same mask), then all get their results; `width`; out-of-range
//                 sources return the caller's own value as documented; exited lanes are ignored by votes.
//                 Reported instead of hanging: a mask without the caller's own lane, a bad width, and a
//                 named lane that is waiting somewhere else (deadlock). Reading a lane that takes no part
//                 (exited, absent, not in the mask) returns 0xCD bytes and prints a note once per call site.
//                 Operand types: int, unsigned, long, unsigned long, long long, unsigned long long, float,
//                 double. In race, only __syncwarp orders memory accesses (as the CUDA guide specifies).
//   not modelled: __half / __nv_bfloat16 operands; the 5-bit truncation of shuffle arguments in hardware;
//                 a live lane named in a mask that later exits without making the call is treated as
//                 exited (on a GPU that is undefined).
// __activemask, cooperative_groups::coalesced_threads
//   modelled:     the lanes of the caller's warp that arrive at the same call site before any of them moves
//                 on: every lane of the warp must be waiting somewhere or have exited before the set is fixed.
//   not modelled: real convergence. A GPU may report fewer lanes than this, and code must not depend on
//                 the set being maximal.
// __popc, __popcll, __clz, __clzll, __ffs, __ffsll, __brev, __brevll
//   modelled:     the bit operations (signatures from memory of device_functions.h, see emu_device.h).
//   not modelled: nothing known.
// __dp4a (int, unsigned, char4, uchar4), __byte_perm, __umulhi, __mulhi, __umul64hi, __float_as_uint,
// __uint_as_float, __float_as_int, __int_as_float (CUDA Math API 13.5)
//   modelled:     the documented overloads and results; __dp4a's bytes sign-extended in the signed forms,
//                 its 32-bit sum wrapping modulo 2^32.
//   not modelled: that HIP has no __dp4a (x2-facts B1): code that calls it passes here and fails for HIP.
// Single-precision math: expf, exp2f, logf, log2f, sqrtf, rsqrtf, powf, fmaf, tanhf, erff, fminf, fmaxf,
// fabsf, copysignf; __expf, __logf, __powf, __sinf, __cosf, __fdividef, __saturatef
//   modelled:     the functions, computed by the host's <cmath> (rsqrtf as 1 / sqrt).
//   not modelled: NVIDIA's error bounds, the fast intrinsics' approximation error, flush-to-zero; the _rn
//                 intrinsics (__fadd_rn and friends) are not provided.
// Half precision (<cuda_fp16.h>, pmpp/emu/cuda_fp16_emu.h): __half, __half2, conversions, arithmetic
//   modelled:     binary16 bit for bit, round-to-nearest-even conversions and arithmetic, __hfma / __hfma2
//                 rounded once.
//   not modelled: NaN payloads, _sat / _relu / directed-rounding variants, half shuffles and atomics,
//                 __nv_bfloat16, speed.
// Atomics: atomicAdd, atomicSub, atomicExch, atomicMin, atomicMax, atomicInc, atomicDec, atomicCAS,
// atomicAnd, atomicOr, atomicXor; cuda::atomic_ref and cuda::atomic from <cuda/atomic>
//   modelled:     the operation and its return value (the old value), for exactly the legacy overloads the
//                 CUDA guide lists (minus half/bfloat16, float2/float4 and 128-bit); atomic_ref / atomic
//                 with load, store, exchange, compare_exchange_weak/strong, fetch_add/sub/and/or/xor/min/
//                 max and the operators, on 1-, 2-, 4- and 8-byte types. In race they are real atomics.
//   not modelled: thread scopes and memory orders have no observable effect; atomics on shared memory
//                 versus global memory are not distinguished; wait/notify; contention cost. In emu a plain
//                 `x++` behaves exactly like an atomic one (see above).
// Constant memory: __constant__, cudaMemcpyToSymbol, cudaMemcpyFromSymbol, cudaGetSymbolAddress
//   modelled:     a file-scope __constant__ variable is a global the host fills through the symbol calls;
//                 offset + count is checked against the symbol's size.
//   not modelled: read-only enforcement in kernels (a kernel that writes one compiles and runs here); the
//                 size limit of constant memory; any caching behaviour.
// Vector types and qualifiers: char1..4, uchar1..4, short1..4, ushort1..4, float2/3/4, int2/3/4,
// uint2/3/4, double2/3/4, make_*, __forceinline__, __noinline__, __inline_hint__, __restrict__,
// #pragma unroll
//   modelled:     the types with Table 49's sizes and alignments, their field names and the make_
//                 functions; the qualifiers compile.
//   not modelled: `#pragma unroll` is honoured or ignored by the host compiler as it sees fit (clang knows
//                 the pragma; gcc ignores it and warns under -Wall); inlining decisions mean nothing here.
// Cooperative groups (<cooperative_groups.h>): this_thread_block, this_grid, coalesced_threads,
// tiled_partition<Size>(thread_block), cooperative_groups::sync
//   modelled:     thread_block (sync, thread_rank, group_index, thread_index, dim_threads, num_threads and
//                 the legacy size / group_dim); grid_group (is_valid, sync, thread_rank, block_rank,
//                 num_threads, num_blocks, dim_blocks, block_index, size, group_dim) with grid-wide sync when
//                 launched by cudaLaunchCooperativeKernel, where all blocks are alive at once (grid x block
//                 at most 1024 threads, else cudaErrorCooperativeLaunchTooLarge); coalesced_group and
//                 thread_block_tile<1..32> (sync, num_threads, size, thread_rank, meta_group_rank,
//                 meta_group_size, shfl, shfl_up, shfl_down, any, all, ballot; shfl_xor on tiles).
//   not modelled: clusters, barrier_arrive / barrier_wait, tiles above 32 or tiles of tiles,
//                 binary_partition, labeled_partition, match_any / match_all on groups, reduce, scan,
//                 memcpy_async, invoke_one. Kernel templates cannot be launched cooperatively.
// Host API: streams (with priorities), events, async copies, pinned and managed memory, attributes,
// cudaGetDevice, cudaFuncGetAttributes, occupancy
//   modelled:     the calls exist with their documented prototypes and succeed; everything is synchronous
//                 (a launch has finished when the launch statement returns); cudaStreamCreate/Destroy and
//                 cudaEventCreate/Destroy allocate and free, so a leak or double destroy shows up in emu.
//   not modelled: asynchrony, overlap of copies and kernels, stream ordering, page locking, unified-memory
//                 migration. cudaEventElapsedTime returns host wall-clock time of the emulator, which is
//                 meaningless for a GPU. cudaOccupancyMaxActiveBlocksPerMultiprocessor returns a synthetic
//                 number (1024 / blockSize) that only serves to size a cooperative launch. Stream
//                 priorities have no effect (the range reported is 0 to 0, as for a device without them);
//                 cudaFuncGetAttributes reports no registers or local memory (there is no device code).
// Failures the emulator detects (deadlocks, invalid masks, grid sync without a cooperative launch)
//   modelled:     the launch is stopped, cudaGetLastError() returns cudaErrorLaunchFailure once, and
//                 cudaGetErrorString() of it carries the explanation. Nothing hangs.
//   not modelled: on a GPU the same mistakes hang or return garbage, usually with no message at all.
#pragma once

// Standard headers first: emu_types.h defines __noinline__, which the standard
// library's own headers must not see.
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cfloat>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <set>
#include <string>
#include <system_error>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#define PMPP_EMU 1                 // lets a .cu file hold emulator-only checks
#if defined(PMPP_BACKEND_RACE) && !defined(PMPP_EMU_RACE)
#define PMPP_EMU_RACE 1
#endif

#include "pmpp/emu/emu_types.h"
#include "pmpp/emu/emu_runtime.h"
#include "pmpp/emu/emu_device.h"
#include "pmpp/emu/emu_host.h"
