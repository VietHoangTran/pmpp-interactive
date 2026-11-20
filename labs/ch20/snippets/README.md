# Snippets

Code that chapter 20's lessons show for the sake of an explanation, and that is not a
lab. Lessons load these files at run time with `useLabFile('ch20/snippets/<name>')`, so
there is no copy to drift. They are not built by the lab runner.

| File | Shown in | Checked by |
|---|---|---|
| `s203_softmax_corrected.cu` | section 20.3, the "Step one row" stepper (Fig. 20.4 line for line, corrected: `-FLT_MAX`, `cuda::maximum<>{}` and `cuda::std::plus<>{}`, the log-sum-exp `L` in place of `D`) | CUDA front-end check (with stand-in declarations of the few CUB and libcu++ names it uses, under `PMPP_CUDA_FRONTEND_CHECK`); `app/src/sim/masksoftmax.test.ts` pins its 28 lines to the lines the stepper's model (`app/src/sim/masksoftmax.ts`) highlights. Not built with nvcc and not run: unverified on CUDA hardware. |

No other page of the chapter shows code that is not a lab file: section 20.5's
listing is lab 20.2's reference solution (`labs/ch20/lab2_flash/solution/`), section
20.3's second listing is lab 20.1's, and section 20.5's llama.cpp lines are a quotation
from `notes/today/ch20-facts.md` D1 (pinned by `app/src/chapters/ch20/S205.test.ts`), not
course code. The summary page shows no kernel; its questions take their numbers from the
models, pinned by `app/src/chapters/ch20/Wrap.test.ts`. `app/src/sim/attn-consistency.test.ts`
checks that the chapter's four attention models agree with each other and with lab 20.2's
constants.
