# Snippets

Kernels that chapter 6's lessons show for the sake of a question or an exercise, and
that are not labs. They are kept here as real `.cu` files so that `./verify.sh` runs the
CUDA front-end check on them, and unit tests fail if the copy shown in a lesson drifts
from the file. They are not built or run.

- `s61_foo_kernel.cu`: the kernel of the book's exercise 1 (pp. 154 to 155), as printed,
  with one source line per numbered line so that the lesson's line numbers are the
  book's (the book wraps lines 01 and 07). Section 6.1's drill classifies its nine
  accesses. `app/src/chapters/ch06/snippets-s61.test.ts` ties the lesson's copy
  (`app/src/chapters/ch06/S61.tsx`) to this file and checks that the access model
  (`app/src/sim/coalesce.ts`, `exercise1`) reads the same nine expressions on the same
  lines.
- `s65_coarsen_toy.cu`: the book's coarsening toy (pp. 141 to 142) before and after
  coarsening by 4, with our bounds checks and our strided-mapping variant. Section 6.5
  shows it; `app/src/chapters/ch06/snippets-s65.test.ts` ties the copy in `S65.tsx` to
  this file and pins the highlighted lines.
- `s66_unroll.cu`: the book's 16-iteration `A(i) B(i)` loop (p. 144) as a kernel, then
  unrolled by 4 and reordered by hand, then with `#pragma unroll 4`. A and B are ours.
  Section 6.6 shows it; `snippets-s66.test.ts` ties the copy in `S66.tsx` to it.
- `s66_where_x_lives.cu`: five versions of one coarsened computation for section 6.6's
  "where does x live?" classification. `app/src/sim/unroll.ts` holds the copies the
  page shows and `app/src/sim/unroll.test.ts` checks each one is in this file word for
  word. Where `x` ends up is the book's rule and NVIDIA's list (ch06-facts A13); nothing
  here was compiled with nvcc to see it.
- `s67_exchange.cu`: the book's exchange loop (pp. 146 to 147), one shared buffer with two
  barriers and two buffers with one, made compilable (the computation, declarations and
  `swapPointers` are ours). Section 6.7 shows it; `snippets-s67.test.ts` ties the copy
  in `S67.tsx` to it and checks that the stepper's loops (`app/src/sim/doublebuf.ts`,
  `loopSource`) are this file's loops line for line.
- `wrap_scores.cu`: a kernel of ours for two questions on the summary page, attention scores
  read from a key cache stored (token, head, dim). Its reads of `K` do not coalesce, which is
  the point of the questions. `app/src/chapters/ch06/snippets-wrap.test.ts` ties the copy in
  `app/src/chapters/ch06/Wrap.tsx` to this file and pins the sector counts the questions take
  from `app/src/sim/coalesce.ts`.
