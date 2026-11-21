# Snippets

Code that chapter 14's lessons show and that is not a lab. The files are real `.cu` sources so
that the CUDA front-end check in `./verify.sh` parses them, and so that a lesson shows the
checked file and not a copy that can drift. They are not built by the lab runner. Drivers in
`check/` run them on the course's CPU emulator; each driver's test is in
`app/src/chapters/ch14/`. One row per snippet.

| File | Shown in | How | Driver | Test |
|---|---|---|---|---|
| `s142_oddeven.cu` | section 14.2: `sort_kernel` (Fig. 14.2, the page's 13 lines) and the host loop `oddEvenSort` | loaded at run time | `check/s142_check.cu` | `S142.snippets.test.ts` |
| `s144_floatkey.cu` | section 14.4: the float-to-key transform and a kernel that applies it to a row | its `floatkey` region, pinned to the lesson's copy | `check/s144_check.cu` | `S144.snippets.test.ts` |
| `s145_radix_iter.cu` | section 14.5: Fig. 14.7 split at the scan into three launches, and the one-block scan between them | its regions, pinned to the lesson's copies, line by line | `check/s145_check.cu` | `S145.snippets.test.ts` |
| `wrap_blocksort.cu` | the wrap page, question `ch14/wrap/q-blocksort`: a block-level odd-even sort with no barrier between steps, on purpose | loaded at run time, the kernel only | `check/wrap_check.cu` (holds the corrected copy) | `Wrap.snippets.test.ts` |

Every test translates the snippet and its driver with `labs/tools/cu2emu.py` side by side, so a
driver's `#include "../<snippet>.cu"` finds the translation, and builds with AddressSanitizer
under the emulator's three thread orders. Each is skipped where no C++ compiler or `python3` is
found. All four files pass the CUDA front-end check. Nothing here has run on a GPU, and none of
it has been built for HIP.

## `s142_oddeven.cu`

Fig. 14.2 with two lines changed (line 05 `i + 1 < N`, so that `N == 0` cannot wrap; line 10
`atomicExch`, so that the flag is not a data race), plus a host loop that stops only after two
quiet steps in a row. What the test checks:

- on the emulator (AddressSanitizer on, three thread orders): Fig. 14.1's rows and flags step by
  step, the steps the host loop launches and its output for the cases in `src/sim/oddeven.ts`,
  random keys against `std::sort`, two blocks per step, and `N == 0` touching nothing;
- on the race-detection flavour (ThreadSanitizer, threads truly concurrent): one step of Fig. 14.2
  as printed (plain `*hasChanged = 1;`, compiled with `-DSNIPPET_AS_PRINTED`) on reversed input is
  reported as a data race on that line; the snippet's `atomicExch` is not. It is skipped where no
  working ThreadSanitizer is found.

## `s144_floatkey.cu`

The transform of section 14.4, on the float's bits as an `unsigned int` (that needs no bit-cast
intrinsic on any backend, and on Metal a float-side zero test may be rewritten). The test compares every
key with `src/sim/floatkey.ts`, bit for bit, over E1's values (`notes/today/ch14-facts.md`), two
NaNs and 600 other bit patterns, and checks the descending order and tie order of a stable sort
on those keys.

## `s145_radix_iter.cu`

Fig. 14.7 as three kernels in order on one stream: extract the bits, an exclusive scan of N + 1
entries (ours, one block, labelled "Not the book's code"), scatter. The test checks Fig. 14.6's
rows for all four bits, Fig. 14.4's four passes, random 8-bit keys on every bit against
`src/sim/radixsort.ts`, and 1,000 random 32-bit keys sorted in 32 passes against `std::sort`.

## `wrap_blocksort.cu`

One block sorts up to `MAXN` keys in shared memory with n fixed odd-even steps: a sorting network
inside a block. It has no barrier between steps, which is the question's defect. The driver holds
the corrected copy, one added `__syncthreads()` at the end of the loop body, and the test checks
that this is the only difference. On the emulator, which runs one thread at a time between
barriers, the kernel as shown leaves 64, 129 and 256 keys unsorted (but every value present) in all
three thread orders, and the corrected copy sorts them, and all 256 lists of eight 0s and 1s, in
all three. There is no race-flavour test. Tried on 2026-11-20, the race flavour reported nothing
for the kernel as shown: it then marked every `__syncthreads()` of a block with one shared
ThreadSanitizer token, which hid races between two barriers. It now alternates two tokens per
barrier. Tried again by hand on 2026-11-21 (nondeterministic, and not part of the test), it reported
data races in the kernel as shown on the compare and both stores (lines 20, 22 and 23); which lines
appear, and whether values are lost in the 256-key case, varies from run to run. It reported
nothing for the corrected copy. A clean race run is still evidence,
not proof.
