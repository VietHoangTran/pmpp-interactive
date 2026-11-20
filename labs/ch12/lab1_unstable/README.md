# Lab 12.1: the unstable filter three ways

Book: sections 12.2 to 12.4 (pp. 290-297), Figs. 12.2, 12.3 and 12.6.

Write three out-of-place unstable filters of `unsigned int` keys. A key is kept when it is
below `threshold`. All three take the same arguments, `(input, output, n, threshold,
outputSize)`, and leave the kept keys in `output[0, *outputSize)` in any order.

- `filterAtomicKernel` (Metal: `filter_atomic`), Fig. 12.2. One thread per key. A thread
  that keeps its key takes a slot with one `fetch_add` on a device-scope
  `cuda::atomic_ref` to `*outputSize` (relaxed), and stores the key there.
- `filterWarpAggregatedKernel` (Metal: `filter_warp_aggregated`), Fig. 12.3 in the
  documented form. Every lane votes with `__ballot_sync` over a full mask of type
  `lane_mask_t`; a warp whose vote is 0 stops; the leader is the lowest keeping lane
  (`maskFfs(vote) - 1`) and makes one atomic of `maskPopc(vote)`; a shuffle over the full
  mask hands everyone the start; each keeping lane adds the number of keeping lanes below
  it, `maskPopc(vote & (((lane_mask_t)1 << lane) - 1))`. The lane comes from `warpSize`.
- `filterPrivatisedKernel` (Metal: `filter_privatised`), Fig. 12.6. A private list of
  `BLOCK_DIM` slots and a private counter in shared memory, a block-scope atomic on the
  private counter, three barriers, one device-scope atomic per block to reserve its range in
  the public list, and a commit where thread k copies private entry k.

Files and commands:

- Edit `work/kernel.cu` (the three kernels and their launchers `launchFilterAtomic`,
  `launchFilterWarpAggregated`, `launchFilterPrivatised`), `work/kernel.metal` (the same
  three kernels in Metal) and `work/launch.mm` (the dispatch).
- The caller sets `*outputSize` to 0 before every call. The kernels do not.
- `BLOCK_DIM` is defined by the build (256 by default): one key per thread, `BLOCK_DIM`
  threads per block. On Metal a threadgroup has `BLOCK_DIM` threads, or fewer if the
  pipeline cannot run that many.
- `lane_mask_t`, `maskPopc` and `maskFfs` are written for you at the top of
  `work/kernel.cu`. Read why before you use them.
- `provided/copy_keys.cu`, `.metal` and `.mm` (`launchCopyKeys`, `copy_keys`,
  `encodeCopyKeys`) copy n keys: the benchmark's bandwidth reference. You do not edit them;
  they are tested too.
- Test: `python3 labs/pmpp.py run 12.1`. Start with `--backend emu`, then `--backend race`.
- Measure (GPU only): `python3 labs/pmpp.py run 12.1 --bench`.

## Differences from the page

- **The predicate.** The book's `cond(val)` is a placeholder it never defines. Here it is
  `val < threshold`, with `threshold` a kernel argument, so the tests and the benchmark can
  choose how many keys are kept.
- **A typed mask.** Fig. 12.3 keeps its mask in an `unsigned int`, builds the lanes-below
  mask as `(1 << laneIdx()) - 1`, and calls `__popc` and `__ffs`. That is right for a
  32-lane warp. On HIP it does not compile, and the obvious one-line fix compiles and is
  wrong: every `_sync` intrinsic there demands a 64-bit mask, `__ballot_sync` returns
  `unsigned long long`, `__popc` and `__ffs` take 32 bits, and on a 64-lane wavefront
  `1 << lane` is undefined for lanes 32 to 63 (`notes/today/ch12-facts.md` B1). So the mask
  has its own type, `lane_mask_t` (`unsigned` on CUDA and the emulators, `unsigned long
  long` under `__HIPCC__`), the bit counts go through typed helpers whose deleted template
  turns a wrong argument type into a compile error (llama.cpp's approach, D1), and the shift
  is `(lane_mask_t)1 << lane`.
- **A ballot instead of `__activemask`.** Fig. 12.3 calls `__activemask()` inside the branch
  and treats the result as the set of lanes that kept a key. NVIDIA's guide now says
  `__activemask()` "cannot be used to determine which warp lanes execute a given branch":
  it is a snapshot of the lanes that happen to be converged (A1). The book's kernel stays
  correct, because each snapshot reserves its own range and the `_sync` shuffle re-converges
  it, but "one atomic per warp" is not guaranteed. A vote that every lane takes, outside any
  branch, gives exactly the lanes that keep, so here no thread returns before the vote,
  out-of-range threads vote 0, and the shuffle's mask is the full mask because every lane
  executes it. AMD's header says its `__activemask()` is exact (B1); Apple says nothing
  either way (C3), which is why the Metal kernel calls its SIMD-group functions from every
  thread too.
- **No cooperative group.** Fig. 12.4's coalesced group, and its `num_threads()` (current
  CUDA's name for the book's `size()`, A5), is not needed here: the ballot gives the same
  leader, count and rank, and it is the form that also builds for HIP.
- **The counter is zeroed by the caller** for all three kernels. The book says so only for
  Fig. 12.2 (p. 291).
- **Atomics on HIP.** `<cuda/atomic>` comes from libhipcxx on HIP, which this course's HIP
  build links only where ROCm provides it (`notes/today/infra-facts.md` B2), so the lab does
  not rely on it. Beside each `cuda::atomic_ref`,
  write the legacy `atomicAdd` under `#if defined(__HIPCC__)`: relaxed, and valid on shared
  and global memory (B1). The reference solution shows the pattern.

On Metal, atomic functions take atomic types only, so the counter is `device atomic_uint*`
and the private counter a `threadgroup atomic_uint`; there is no scope to choose
(`notes/today/infra-facts.md` C1). The warp-level steps are four SIMD-group functions,
listed in `work/kernel.metal` (ch12-facts C1). Dispatch with `dispatchThreadgroups`, so that
the threads past n exist and take part in them.

## What the tests do

An unstable filter may put the kept keys in any order, so the driver does not compare
arrays. For every input it checks that `*outputSize` equals the number of keys a sequential
filter keeps; that `output[0, count)`, sorted, equals the sequential filter's list, sorted
(nothing lost, nothing written twice); and that every slot from `count` to `n` still holds
the sentinel `0xFFFFFFFF` it wrote before the call (keys are below 2^24, so the sentinel is
never a key). Then it sets `*outputSize` to 0 again, calls the kernel a second time and
checks the same three things.

- Sizes: 1, 31, 32, 33, 255, 256, 257, 1000, 4099; on a GPU also 65537, 2^20 and
  3 x 2^20 + 17. The `race` backend stops at 1000: it runs every thread of a block as an
  operating-system thread.
- Uniform keys in [0, 2^24) with fractions kept 0, 0.001, 0.03, 0.5, 0.97 and 1.
- Patterns: every key, no key, only the last key, only key 31 of every 32, only keys 32 to
  63 of every 64, every other key, runs of 40 kept and 40 dropped that straddle warps and
  blocks. The "32 to 63" pattern puts every kept key where a 32-bit mask cannot see it on a
  64-lane wavefront; on the emulators, Metal and NVIDIA it is one more pattern.

What each backend can show you:

- `emu` runs one thread at a time until it reaches a barrier, a warp `_sync` call or its
  end. A missing barrier is wrong there under at least one of its three thread orders, and
  AddressSanitizer stops at an out-of-bounds write and names the line. A plain `++` on the
  counter is never wrong there: nobody is interrupted in the middle of it.
- `race` runs the threads truly concurrently under ThreadSanitizer and reports two threads
  touching the same memory without synchronisation, with both source lines.
- Your GPU shows lost or duplicated keys only when the schedule happens to produce them.

## Before you benchmark

`--bench` grows n from 2^22 until the provided copy takes at least 3 ms (at most 2^26 keys,
and input plus output at most a quarter of the memory the device reports). It then times
the three filters on uniform keys at fractions kept 0, 0.001, 0.01, 0.03, 0.1, 0.25, 0.5,
0.75 and 1, best of 5 runs each, and checks every result exactly before reporting it. It
prints milliseconds and GB/s of the least traffic a filter needs (4n bytes read, 4 per kept
key written, 4 for the counter); the copy's 8n bytes per run is the same figure for a filter
that keeps everything. Zeroing `*outputSize` is not timed.

Write down first:

1. At which fraction kept the plain-atomic kernel will be furthest behind the privatised
   one, a number from 0 to 1 (the labs page records it as `ch12/labs/p-lab1-worst`).
2. The plain-atomic kernel's time divided by the warp-aggregated kernel's time at fraction
   0.5 (`ch12/labs/p-lab1-agg`).

The book's model (section 12.2, p. 291) is that every kept key is one atomic on the same
counter, serialised, so contention grows with the fraction kept, while the privatised kernel
makes one public atomic per block whatever the fraction. Against that, the bytes moved also
grow with the fraction, and so does the work of the private atomics.

A value near 1 for the second prediction would mean your compiler already aggregates the
plain atomic, so that Fig. 12.2 runs as Fig. 12.3. The book says the compiler does this
(p. 294). NVIDIA says so only in a blog post updated in 2017, "in many cases", with no
conditions (`notes/today/ch12-facts.md` A7). AMD's LLVM backend has a pass that rewrites a
same-address atomic into exactly Fig. 12.3's ballot, rank, leader and broadcast, on by
default at `-O1` and above (B4; read in upstream LLVM, not confirmed for ROCm 10.2's
`amdclang++`). Apple documents nothing either way. Only the measurement on your device
decides, and only the compiled code (PTX or SASS, the AMDGPU disassembly) shows what the
compiler did.

The numbers are measurements of your device, not of GPUs in general.
