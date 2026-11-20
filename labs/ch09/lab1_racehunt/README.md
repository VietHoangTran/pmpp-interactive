# Lab 9.1: race hunt, two histogram kernels

Book: sections 9.2 to 9.4 (pp. 203-213), Figs. 9.6 and 9.9. The defects are ours, not
the book's.

`work/kernel.cu` and `work/kernel.metal` hold two 256-bin histograms of an 8-bit image,
written by a colleague:

- `histogramBasic` (Metal: `histogram_basic`), Fig. 9.6: one thread per pixel, each adding
  one to its pixel's bin.
- `histogramPool` (Metal: `histogram_pool`), Fig. 9.9: one private histogram per block in a
  pool in global memory, updated with block-scope atomics, then a commit of the non-zero
  private bins to the public bins.

Between them they have **three independent defects**. Everything compiles. Find and fix
all three.

- Edit `work/kernel.cu` (the two kernels; `launchHistogramBasic` and `launchHistogramPool`
  below them are correct) and `work/kernel.metal`. The same three defects are in both.
  `work/launch.mm` is correct and is there for completeness.
- The caller zeroes `bins` and the pool. The pool holds ceil(width x height /
  threadsPerBlock) x 256 counters: one private histogram per block.
- Test: `python3 labs/pmpp.py run 9.1 --backend emu`, then `--backend race`, then
  `python3 labs/pmpp.py run 9.1` for your GPU.
- Measure (GPU only): `python3 labs/pmpp.py run 9.1 --bench`.

## What the tests do

Every case compares all 256 counts with the sequential loop of Fig. 9.2, which lives in
`main.cpp`. Counts are whole numbers, so the comparison is exact.

- Images, width x height: 1 x 1, 63 x 1, 64 x 1, 100 x 10, 64 x 64; on a GPU also
  1024 x 1024 and 3000 x 1001.
- Four kinds of image, from fixed seeds: uniform random bytes, a constant image (every
  pixel 200), 0 and 255 in long runs, a smooth gradient.
- Blocks of 64 and 128 threads, fewer than the 256 bins, so the commit loop must stride; on
  a GPU also 256 and the largest block the kernel can run with.
- The basic kernel's cases come first, then the pool's, so each defect shows in its own
  kernel's cases. Each kernel's first case is the constant image in 64 blocks of 64
  threads: every thread wants the same bin, the most contended input there is.
- On a GPU each configuration is launched 20 times on freshly zeroed buffers, and a failure
  says in how many launches the counts were wrong. On the emulators it is launched once.

## Your prediction

Before you run anything, read `work/kernel.cu` and write down the defects you can find by
reading. For each one, write down which backend you expect to show it, and why. Then run
`emu`, `race` and your GPU, and fill in this table as you go:

| Defect | `emu` | `race` | your GPU |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |

## Method

1. Emulator first: `python3 labs/pmpp.py run 9.1 --backend emu`. It runs one thread until
   that thread reaches a barrier or its end, then the next, under three fixed orders. A
   missing barrier gives a wrong answer there every time. A missing atomic never does: no
   thread is interrupted in the middle of `++bins[b]`, so the emulator cannot lose an
   update. Expect one of the three defects here, and two to hide.
2. `python3 labs/pmpp.py run 9.1 --backend race` runs the emulated threads truly
   concurrently under ThreadSanitizer, several blocks of up to 128 threads at a time. Two
   threads touching the same memory without synchronisation are reported with both
   accesses, whether or not the counts came out wrong. Read the `kernel.cu:` line numbers.
   One access of a pair can be inside the emulator's own atomic
   (`pmpp_emu::atomic_rmw ... fetch_add`): that is an `atomic_ref` call of yours, racing
   with a plain access on the other line. ThreadSanitizer reports each pair of lines once
   per run, so read each kernel's first case; a later case can pass with the race still
   there.
3. Fix one thing, run again. The failures change as you go.
4. Then your GPU: `python3 labs/pmpp.py run 9.1`. Note which defects it showed you, and
   which it hid.

If a fix needs an atomic, write it as the book does: a `cuda::atomic_ref<unsigned int,
scope>` on the counter and `fetch_add(..., cuda::memory_order_relaxed)`. Choose the scope
from the threads that share the counter: `cuda::thread_scope_block` if they are all in one
block, `cuda::thread_scope_device` if they are in different blocks. The private copy keeps
block scope. On Metal, atomic functions take atomic types only, so the fix changes the
parameter to `device atomic_uint*` and uses `atomic_fetch_add_explicit`
(`notes/today/ch09-facts.md` C1); the host still allocates plain 32-bit counters.

A Metal barrier names the memory whose operations it orders:
`threadgroup_barrier(mem_flags::...)`. Which flag does the pool kernel need, and why? The
flags and what each guarantees are in `notes/today/ch05-facts.md` C2.

## Reading your table

Whatever your GPU showed, read it against what the vendors document, because that is what
the next device is entitled to do (`notes/today/ch09-facts.md`):

- Atomics on one location are serialised in no fixed order; a plain write from several
  threads of a warp to one location leaves "which thread performs the final write"
  undefined (NVIDIA, A3). AMD: same-location atomics "are serialized in an undefined
  order" (B1). Formally a plain `++bins[b]` from two threads is a data race and undefined
  behaviour, so a lost count is the typical outcome, not a promised one (A2, A3).
- Metal: "Objects of atomic types are free from data races" (C1). The spec says nothing
  about plain accesses.
- No GPU tool would have named your lines. NVIDIA's `compute-sanitizer --tool racecheck`
  checks shared memory only, so it cannot see a race on global bins (A10); AMD documents no
  race detector (B6); Metal's Shader Validation lists no race check (C5). The `race`
  backend is ThreadSanitizer on CPU threads (E1): a clean run there shows the schedules that
  happened, not all of them.

A configuration that passed 20 times on your GPU showed you 20 schedules on one device. It
did not show you that the kernel is correct.

## Before you benchmark

`--bench` times both kernels on 2^26 pixels (fewer if the device reports too little memory
for the image and the pool), 256 threads per block, on uniform bytes and on the constant
image, best of 5 runs. Write down first:

1. How many times as long the basic kernel takes on the constant image as on uniform bytes
   (section 9.3 asks the same question and reveals the answer).
2. How many times faster the pool kernel is than the basic kernel on the constant image
   (the labs page records this prediction).

The book's model (section 9.3, pp. 209-210) allows up to 256x for the first: atomics on
one location cannot overlap, and uniform bytes spread them over 256 locations. Two things
can make the real figure much smaller. GPUs perform device-scope atomics in the L2 cache,
not in DRAM (NVIDIA, A4; AMD, B2; Apple documents nothing, C4), so each one is shorter. And
a compiler can combine the same-address atomics of one warp into one: NVIDIA says so only in
a blog post updated in 2017, "in many cases", with no conditions (A9); AMD's compiler does it
only when it can prove the whole wavefront uses one address, which a bin read from the
image is not (B5).

The numbers are measurements of your device. Zeroing the bins and the pool before each
run is not timed.

## One more thing about scope

Fix the commit with `cuda::thread_scope_block` instead of device scope and every test
here still passes: the emulators ignore scopes (`labs/README.md`), and Metal has none. It
is still wrong. Under CUDA's memory model two blocks updating one public bin is a data race,
because each update "is not atomic at a scope that includes the thread that performed the
other operation", and any such race "results in undefined behavior" (A2). Right numbers on
one backend prove nothing about the next.
