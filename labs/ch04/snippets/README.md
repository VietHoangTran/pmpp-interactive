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
