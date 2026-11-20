# Snippets

Code that chapter 20's lessons show for the sake of an explanation, and that is not a
lab. Lessons load these files at run time with `useLabFile('ch20/snippets/<name>')`, so
there is no copy to drift. They are not built by the lab runner.

| File | Shown in | Checked by |
|---|---|---|
| `s203_softmax_corrected.cu` | section 20.3, the "Step one row" stepper (Fig. 20.4 line for line, corrected: `-FLT_MAX`, `cuda::maximum<>{}` and `cuda::std::plus<>{}`, the log-sum-exp `L` in place of `D`) | CUDA front-end check (with stand-in declarations of the few CUB and libcu++ names it uses, under `PMPP_CUDA_FRONTEND_CHECK`); `app/src/sim/masksoftmax.test.ts` pins its 28 lines to the lines the stepper's model (`app/src/sim/masksoftmax.ts`) highlights. Not built with nvcc and not run: unverified on CUDA hardware. |
