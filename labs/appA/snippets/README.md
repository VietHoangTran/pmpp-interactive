# Snippets

Code that Appendix A's lessons show for the sake of an explanation, and that is not a lab.
Lessons load these files at run time with `useLabFile('appA/snippets/<name>')`, so there is
no copy to drift. The lab runner does not build them; unit tests compile the host C++ ones
through a check driver in `check/` and compare the results with the lesson's model.

| File | Shown in | Checked by |
|---|---|---|
| `bf16_round.h` | section A.4, "FP32 to BF16 on the bits": llama.cpp's FP32-to-BF16 rounding (round to nearest even on the integer bits, NaN forced quiet; `ggml-impl.h`, notes/today/appA-facts.md D3) with plain truncation beside it | `app/src/sim/fpadd.test.ts` compiles `check/bf16_round_test.cpp` with the host C++ compiler, runs it on the explorer's presets, NaNs, ties and 400 random patterns, and compares every result, bit for bit, with `bf16FromBits` in `app/src/sim/fpadd.ts`. Skipped when no C++ compiler is found. |
| `mx_quantize.h` | section A.5, under the scene "Quantising one block": one MXFP4 block (32 E2M1 codes and one UE8M0 scale, OCP MX v1.1 §6.3's rule; notes/today/appA-facts.md A4), quantised and decoded on the CPU | `app/src/sim/mxblock.test.ts` compiles `check/mx_quantize_test.cpp` (`-Wall -Wextra`, no warnings allowed), runs it on the scene's block, the q-outlier block, an all-zero block, a power-of-two maximum, round-to-even ties, subnormal floats and ten seeded blocks, and compares the scale byte, all 32 codes and every decoded float, bit for bit, with `quantizeBlock` in `app/src/sim/mxblock.ts`; it also checks that `sizeof(MxBlock)` is 17, llama.cpp's `block_mxfp4` size. Skipped when no C++ compiler is found. |

`bf16_round.h` is plain host C++. Its two result lines are llama.cpp's; the NaN test and the
use of `memcpy` instead of a union are ours. Nothing here runs on a GPU.

`mx_quantize.h` is ours, plain host C++: the shape a CPU reference for an MXFP4 kernel takes.
It packs two codes per byte with the even element in the low nibble, which matches the
model's element order; whether a given library packs nibbles the same way is not claimed.
