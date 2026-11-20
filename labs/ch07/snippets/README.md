# Snippets

Kernels that chapter 7's lessons show and that are not labs. They are kept here as real
`.cu` files so that `./verify.sh` runs the CUDA front-end check on them, and unit tests
fail if the copy shown in a lesson drifts from the file. They are not built by the lab
runner; the drivers in `check/` run them on the CPU emulator from the unit tests.

- `s75_halo_toy.cu`: the book's Fig. 7.12 kernel (p. 174), corrected, at the sizes of
  Fig. 7.13 (`FILTER_RADIUS 1`, `IN_TILE_DIM 8`, a 6 x 6 output tile), plus a launch
  function of ours (the book shows no host code). Corrections to the printed listing:
  `N_s` has its element type, the constant array is used under the name it is declared
  with (`F_c`), `0.0f` instead of `0.0`, and `threadIdx` is copied into `int` before
  anything is subtracted. Section 7.5's "Load, wait, compute" stepper shows it and
  highlights its lines; `app/src/chapters/ch07/S75.snippets.test.ts` ties the copy in
  `S75.tsx` to this file and pins the highlighted lines, and the stepper's model is
  `app/src/sim/convtile.ts`. Front-end-checked and emulator-checked: the same test runs
  it through `check/s75_halo_toy_test.cu`.
- `wrap_causal_conv1d.cu`: a causal 1D convolution of ours (4 taps, a one-sided halo,
  128-thread blocks) for the summary page's first question. It is wrong on purpose:
  threads past the end return before the barrier, so when the last block has only 1 or
  2 outputs, a halo slot is read that nobody wrote. `app/src/chapters/ch07/Wrap.test.ts`
  ties the page's copy (`app/src/chapters/ch07/Wrap.tsx`) to this file, runs a model of
  the kernel for every length from 1 to 400, and runs the file itself through
  `check/wrap_causal_conv1d_test.cu`. Front-end-checked and emulator-checked.

## check/

Drivers, not lesson code. Each includes its snippet with `#include "../<snippet>.cu"`, so
the CUDA front-end check parses the pair as one file. `app/src/chapters/ch07/snippetEmu.ts`
inlines the snippet in place of that line, translates the result with
`labs/tools/cu2emu.py`, builds it as the lab runner builds the `emu` backend
(AddressSanitizer on), runs it and returns what it printed. The tests are skipped when
no C++ compiler or `python3` is found.

| Driver | What it runs | What the test requires |
|---|---|---|
| `check/s75_halo_toy_test.cu` | `launchHaloToy` on 10x10, 14x14, 13x7, 6x6, 1x1 and 37x19 images with a lopsided 3 x 3 filter, in the emulator's three thread orders, against a CPU reference with zero ghost cells (exact, small whole numbers) | every output right in all 18 runs, exit 0, no AddressSanitizer report |
| `check/wrap_causal_conv1d_test.cu` | `causalConv1d` for every n from 1 to 400 in the three thread orders, beside a fixed copy made for the check (no early return; every thread loads, a ghost zero past the end, and reaches the barrier before testing `i < n`) | the page's kernel wrong for exactly n = 1, 2, 129, 130, 257, 258, 385 and 386 in every order; the fixed copy right for every n |
