# Snippets

Code that chapter 22's lessons show for the sake of a question, and that is not a lab. The
files are real `.cu` so that `./verify.sh` runs the CUDA front-end check on them. They are
not built by the lab runner.

| File | Shown in | Checked by |
|---|---|---|
| `scatter_conv_plain.cu` | section 22.2, "This scatter passes on the emulator" | CUDA front-end check; `app/src/chapters/ch22/S222.snippets.test.ts` fails if the lesson's copy drifts from this file, checks its shape, and runs it on the `emu` backend (correct in all three thread orders, because `emu` never interrupts a thread in the middle of its `+=`) and on the `race` backend (ThreadSanitizer reports the `+=` line racing with itself) |
| `gather_conv.cu` | the same place, beside it | CUDA front-end check; the same test: lesson copy, shape, correct on `emu` and no report on `race` |
| `check/conv_check.cu` | nothing (a driver) | CUDA front-end check. The test appends it to the two snippets, translates the result with `labs/tools/cu2emu.py` and builds it as the lab runner builds `emu` and `race`. It also holds an `atomicAdd` copy of the scatter, made for the check and not lesson code, which must be correct everywhere with no report. Skipped when no C++ compiler is found; the race half is skipped when ThreadSanitizer does not build. |

The two kernels compute the same zero-padded convolution,
`out[o] = sum over t of filter[t] * in[o + t - r]`, the chapter 7 form. The book prints no
code in chapter 22 (`notes/ch22.md`, "Code listings"); both kernels are ours.
