# Snippets

Code that module X2's lessons show for the sake of an explanation, and that is not a lab.
Lessons load these files at run time with `useLabFile('x2/snippets/<name>')`, so there is no
copy to drift. The lab runner does not build them; unit tests compile the host C++ ones
through a check driver in `check/` and compare the results with the lesson's model.

| File | Shown in | Checked by |
|---|---|---|
| `q4_0_ref.h` | sections X2.2 ("Q4_0: the signed extreme over -8") and X2.3 ("The alignment trap"): one llama.cpp Q4_0 block on the CPU, quantised by the `_ref` rule, packed with weight j and j + 16 in byte j, dequantised, read as 32-bit words from two 16-bit halves, and dotted with the 8 folded out (notes/today/x2-facts.md D1, D3, A5, E4) | `app/src/sim/quant.test.ts` compiles `check/q4_0_ref_test.cpp` (`-std=c++17 -Wall -Wextra -Werror -O2 -ffp-contract=off`), runs it on the scene block, the q-q40-clip block, an all-zero block, a positive extreme and six seeded blocks, and compares the FP16 bits of d, all 16 bytes, all 32 dequantised floats, the four words and the folded dot, bit for bit, with `app/src/sim/quant.ts`; it also checks that `sizeof(BlockQ4_0)` is 18. Skipped when no C++ compiler is found. |
| `accumulators.md` | section X2.6 ("Safe or unsafe?"): four fragments, one MSL and three CUDA, that differ in what holds the sum (`half4` into `dot`, one `__half2` with `__hfma2`, a `short` block sum, an `int` block sum then float). The section parses the file by its `##` headings. | `app/src/sim/qdot.snippets.test.ts` checks that the file has the four fragments of `FRAGMENTS` in order, that each declares its accumulator and adds into it as `FRAGMENT_MODEL` in `app/src/sim/qdot.ts` says, and that fragments 3 and 4 differ only in the type of `sumi`, which the model sums in int16 and int32. It then writes the three CUDA fragments (2, 3, 4), exactly as the file prints them, into `check/accumulators_check.cu`, translates it with `labs/tools/cu2emu.py`, builds it against the emulator (`-ffp-contract=off`, AddressSanitizer on) and runs it on `runFragment`'s rows: the worst-case results are equal and the realistic relative errors are equal, bit for bit. It also passes the driver, fragments included, through the CUDA front-end check, and checks that a host-only call in a fragment fails it. Status: fragments 2 to 4 emulator-checked and front-end-checked, unverified on CUDA; fragment 1 (MSL) not compiled, unverified. Skipped when no C++ compiler or `python3` is found. |
| `check/accumulators_check.cu` | nothing (the driver of the row above) | Includes each CUDA fragment as `<id>.inc` from the test's scratch directory, in a one-thread kernel; `dot4` is `__dp4a`. `labs/pmpp.py`'s front-end sweep sees it without the fragments and checks only the driver. Never built for HIP (no `__dp4a`, x2-facts B1) or run on a GPU. |

`accumulators.md` is Markdown, not `.cu` or `.metal`, because the fragments are not whole
functions and the page parses the file by its headings. Since 2026-11-21 the front-end stub and
both emulators declare `__half`, `__half2`, `char4`, `uchar4` and `__dp4a`
(`labs/README.md`), which is what lets the check driver run the CUDA ones. `q4_0_ref.h` is ours, plain host C++, written from the lines of llama.cpp quoted in the facts
file. Its FP16 conversion is our own round-to-nearest-even routine; whether llama.cpp's
`GGML_FP32_TO_FP16` rounds the same way on every host was not checked. Nothing here runs on a GPU.
