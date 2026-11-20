# Snippets

Kernels that chapter 4's lessons show for the sake of a question, and that are not
labs. They are kept here as real `.cu` files so that `./verify.sh` runs the CUDA
front-end check on them, and unit tests fail if the copy shown in a lesson drifts
from the file. They are not built or run.

- `incorrect_barrier.cu`: the book's Fig. 4.4 (p. 72) with `__global__` added, which
  the printed listing lacks. It is wrong on purpose.
- `barrier_uniform.cu`, `barrier_per_thread.cu`, `barrier_early_return.cu`,
  `barrier_loop.cu`, `barrier_guarded.cu`: the kernels of section 4.3's bug hunt
  "Is this barrier legal?". All but the first and the last are wrong on purpose.
  `app/src/sim/barrier.ts` holds a hand-written model of each, and
  `app/src/sim/barrier.test.ts` ties the model's line numbers and the lesson's
  copies (`app/src/chapters/ch04/S43.tsx`) to these files.
- `device_loop.cu`: section 4.8's query loop (book p. 88) with an `int` index, both
  return codes checked, and the per-SM fields the list on p. 89 leaves out. Host code
  only. The lesson (`app/src/chapters/ch04/S48.tsx`) loads this file at run time, so
  there is no second copy to drift. It does not read the clock rate: `clockRate` is no
  longer a member of `cudaDeviceProp`, and the stub header has no `cudaDevAttrClockRate`.
- `foo_kernel.cu`: the kernel and launch of the book's exercise 1 (pp. 90 to 91), as
  printed, for section 4.5's prediction. `app/src/sim/divergence.test.ts` ties the
  lesson's copy (`app/src/chapters/ch04/S45.tsx`) and the model's three conditions to
  this file.
- `warp_tail_sum.cu`: a block sum of ours whose last six steps assume one warp runs in
  lockstep, with no barrier, for the summary page's first question. It is wrong on
  purpose on any device that does not promise lockstep.
  `app/src/chapters/ch04/Wrap.snippets.test.ts` ties the page's copy
  (`app/src/chapters/ch04/Wrap.tsx`) to this file and runs the six steps in lockstep and
  out of order.

Not here: section 4.8's erratum hunt shows the p. 88 loop with the p. 89 fields as
printed, including `devProp.clockRate`, which CUDA 13.0 removed. That listing would fail
the front-end check on purpose, so it lives only in `app/src/chapters/ch04/S48.tsx`, and
`app/src/sim/launchconfig.test.ts` ties it to `notes/ch04.md` and to the facts file (A18).
