# Snippets

Kernels that chapter 8's lessons show for the sake of a stepper or a question, and
that are not labs. They are kept here as real `.cu` files so that `./verify.sh` runs
the CUDA front-end check on them, and a unit test fails if the copy shown in a lesson
drifts from the file. They are not built or run.

- `zmarch_planes.cu`, `zmarch_registers.cu`: the kernels of section 8.5's stepper
  "Step one block through z": book Figs. 8.10 and 8.12, corrected (c3 with y-1 and c4
  with y+1 as in Figs. 8.6 and 8.8; the register `inNext` where Fig. 8.12 prints
  `inNext_s`; registers initialised), with `ty` and `tx` for `threadIdx.y` and
  `threadIdx.x`. `app/src/sim/zmarch.ts` is a hand-written model of these two, and
  `app/src/chapters/ch08/S85.test.ts` ties the lesson's copies to these files.
  The tested versions of the same kernels are lab 8.1's (`../shared/planes.cu` and
  `../lab1_stencil3d/solution/kernel.cu`).
- `rolling_window_indexed.cu`: section 8.5's "keep the window in registers" example,
  the register-tiled march with its window as `float w[3]` indexed by `z % 3`. Ours, not
  the book's. `app/src/chapters/ch08/S85.test.ts` runs a model of its loop under three
  thread orders and gets the CPU sweep's results; by NVIDIA's documented rule its window
  is a candidate for local memory.
- `regtile_two_planes.cu`: the summary page's question on double buffering: Fig. 8.12
  (corrected) with its shared plane doubled, so that the loop needs one barrier. Ours, not
  the book's. `app/src/chapters/ch08/Wrap.snippets.test.ts` ties the page's copy to this
  file and runs a model of its loop one thread at a time under three orders.
