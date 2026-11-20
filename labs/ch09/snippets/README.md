# Snippets

Code that chapter 9's lessons show for the sake of an explanation, and that is not a
lab. Lessons load these files at run time with `useLabFile('ch09/snippets/<name>')`, so
there is no copy to drift. They are not built by the lab runner.

| File | Shown in | Checked by |
|---|---|---|
| `histogram_sequential.cpp` | section 9.1, "One division, one flat array" | `app/src/sim/histogram.test.ts` compiles `check/histogram_sequential_test.cpp` with the host C++ compiler, runs it on every preset image of the section's explorer and on seeded larger images, and compares the 256 counts with the model (`app/src/sim/histogram.ts`). The test is skipped when no C++ compiler is found. |
| `privatised_variants.cu` | section 9.4, "Seven broken privatised kernels" | CUDA front-end check; `app/src/sim/histpriv.test.ts` pins how each kernel differs from `histPrivatised` and runs every kernel on the emulator (see the next row) |
| `check/privatised_check.cu` | nothing (a driver) | CUDA front-end check. `app/src/sim/histpriv.test.ts` appends it to `privatised_variants.cu`, translates the result with `labs/tools/cu2emu.py`, builds it as the lab runner builds the `emu` and `race` backends, runs each kernel on the hunt's eight pixels, and compares the bins of each thread order with `app/src/sim/histpriv.ts` (`simulate`) and the ThreadSanitizer reports with the model's happens-before check (`races`). Skipped when no C++ compiler is found; the race half is skipped when ThreadSanitizer does not build. |
| `wrap_early_exit.cu` | the summary page, "A mixed check", first question | CUDA front-end check; `app/src/chapters/ch09/Wrap.snippets.test.ts` checks its shape, translates it with `labs/tools/cu2emu.py` together with `check/wrap_early_exit_test.cu`, runs it on the course's emulator in its three thread orders at 256 and 128 threads per block, and compares the bins it gets wrong with the page's model (`earlyExitLost` in `Wrap.tsx`). Skipped when no C++ compiler is found. |
| `check/wrap_early_exit_test.cu` | nothing (a driver) | CUDA front-end check; run by the same test. It also runs a copy of the kernel with the early return replaced by a guard on the update, which must be right everywhere. |

`histogram_sequential.cpp` is Fig. 9.2 (p. 203) with `const` added to the image pointer.
Like the book's listing it assumes the caller has zeroed `bins`.

`privatised_variants.cu` is Fig. 9.10 (p. 214) shrunk for the hunt (eight bins, one pixel per
thread) plus seven copies that are wrong on purpose. It reads the private count once into
`count`, where the book reads `bins_s[b]` twice.

`wrap_early_exit.cu` is Fig. 9.10 (p. 214) with one change, wrong on purpose: a thread with no
pixel returns before the zeroing loop, so the bins it owns in the last block are never zeroed or
committed.
