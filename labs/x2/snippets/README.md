# Snippets

Code that module X2's lessons show for the sake of an explanation, and that is not a lab.
Lessons load these files at run time with `useLabFile('x2/snippets/<name>')`, so there is no
copy to drift. The lab runner does not build them; unit tests compile the host C++ ones
through a check driver in `check/` and compare the results with the lesson's model.

| File | Shown in | Checked by |
|---|---|---|
| `q4_0_ref.h` | sections X2.2 ("Q4_0: the signed extreme over -8") and X2.3 ("The alignment trap"): one llama.cpp Q4_0 block on the CPU, quantised by the `_ref` rule, packed with weight j and j + 16 in byte j, dequantised, read as 32-bit words from two 16-bit halves, and dotted with the 8 folded out (notes/today/x2-facts.md D1, D3, A5, E4) | `app/src/sim/quant.test.ts` compiles `check/q4_0_ref_test.cpp` (`-std=c++17 -Wall -Wextra -Werror -O2 -ffp-contract=off`), runs it on the scene block, the q-q40-clip block, an all-zero block, a positive extreme and six seeded blocks, and compares the FP16 bits of d, all 16 bytes, all 32 dequantised floats, the four words and the folded dot, bit for bit, with `app/src/sim/quant.ts`; it also checks that `sizeof(BlockQ4_0)` is 18. Skipped when no C++ compiler is found. |

`q4_0_ref.h` is ours, plain host C++, written from the lines of llama.cpp quoted in the facts
file. Its FP16 conversion is our own round-to-nearest-even routine; whether llama.cpp's
`GGML_FP32_TO_FP16` rounds the same way on every host was not checked. Nothing here runs on a GPU.
