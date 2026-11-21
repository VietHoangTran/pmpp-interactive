# Snippets

Code that chapter 10's lessons show for the sake of an explanation, and that is not a
lab. Lessons load these files at run time with `useLabFile('ch10/snippets/<name>')`, so
there is no copy to drift. They are not built by the lab runner.

| File | Shown in | Checked by |
|---|---|---|
| `fold.h` | section 10.1, "One loop for every reduction" | `app/src/sim/reducetree.test.ts` compiles `check/fold_test.cpp` with the host C++ compiler, runs it, and compares every float, bit for bit, with the model behind the section's fold explorer (`app/src/sim/reducetree.ts`). The test is skipped when no C++ compiler is found. |
| `s103_inplace.cu` | sections 10.3 and 10.4, the steppers (Figs. 10.5 and 10.8 as printed, one block) | CUDA front-end check; `app/src/sim/reduction.test.ts` pins the line numbers the steppers highlight and runs both kernels on the emulator (see below) |
| `s104_mirrored.cu` | section 10.4, "Mirror it" (exercise 3, our answer) | CUDA front-end check; `app/src/chapters/ch10/S104.snippets.test.ts` ties the page's parts to it; emulator run as below |
| `s106_barrier_hunt.cu` | section 10.6, "Where does the barrier go?" | CUDA front-end check; `app/src/sim/reduction.test.ts` pins its line numbers and runs every kernel on the emulator |
| `s107_warp_reduce_as_printed.cu` | section 10.7, the "as printed (CUDA only)" tab of `warp_reduce` (Fig. 10.13 and the helpers of pp. 238–239, as printed) | CUDA front-end check; `app/src/chapters/ch10/S107.snippets.test.ts` pins the book's lines and runs it on the emulator through `check/s107_warp_reduce_test.cu` |
| `s107_hybrid.cu` | section 10.7, "Fig. 10.14, corrected" (the book's lines 01 to 19, portable mask and width) | CUDA front-end check; `S107.snippets.test.ts` pins the line numbers the lesson names, checks that its mask type, `warpIdx` and `warp_reduce` are lab 10.1's character for character, and runs it on the emulator through `check/s107_hybrid_test.cu` at 64 and 1,024 threads |
| `check/s107_warp_reduce_test.cu`, `check/s107_hybrid_test.cu` | nothing (drivers) | CUDA front-end check. Built and run by `S107.snippets.test.ts` (as the lab runner builds the `emu` backend, AddressSanitizer on); what they print is compared with `app/src/sim/shuffle.ts`. Skipped when no C++ compiler or python3 is found. |
| `wrap_centre.cu` | the summary page, the kernel of the first mixed-check question (`centreRows`, ours, wrong on purpose: only thread 0 holds the row's sum) | CUDA front-end check; `app/src/chapters/ch10/Wrap.snippets.test.ts` checks that its mask type, defines, `warpIdx`, `laneIdx` and `warp_reduce` are lab 10.1's character for character, and runs it on the emulator through `check/wrap_centre_test.cu` |
| `check/wrap_centre_test.cu` | nothing (a driver) | CUDA front-end check. Built and run by `Wrap.snippets.test.ts` (as the lab runner builds the `emu` backend, AddressSanitizer on): at blocks of 256 and 64 threads and in the three thread orders, the page's kernel must be right exactly in the columns thread 0 writes, and a copy that hands the mean back through shared memory after a barrier must be right everywhere. Skipped when no C++ compiler or python3 is found. |
| `check/reduction_check.cu` | nothing (a driver) | CUDA front-end check. `app/src/sim/reduction.test.ts` appends it to the three files above, translates the result with `labs/tools/cu2emu.py`, builds it as the lab runner builds the `emu` backend (AddressSanitizer on), runs each kernel in the emulator's three thread orders, and compares outputs and arrays with `app/src/sim/reduction.ts`. Skipped when no C++ compiler is found. |

`fold.h` is plain host C++: the sequential loop of Fig. 10.2 as a template, a max over a
row that starts at `-FLT_MAX`, and an argmax over (value, index) pairs that starts from
element 0 (footnote 1, p. 222) and keeps the lower index on a tie.

`s106_barrier_hunt.cu` is wrong on purpose except for `SharedMemorySumReductionKernel` (Fig.
10.10 with its missing `output` parameter restored) and `MaxThenSumKernel`. Two results of
the emulator run are quoted in section 10.6: `BarrierInsideIfKernel` passes in all three
orders, because the emulator, like NVIDIA's wording, lets exited threads off a barrier
(the lesson's model holds the book's stricter rule and reports it stuck); and
`LessOrEqualKernel` stops with an AddressSanitizer report on its addition line. The two
max-then-sum kernels sum `x - max` instead of `exp(x - max)`: they were written when the
front-end check's stub header declared no device math functions (it declares `expf` now),
and the barrier question they ask does not depend on which is summed.

`s107_hybrid_test.cu` also holds two variants of Fig. 10.14 written for the check, not for
a lesson: one without the barrier after the loop (line 11), which section 10.7 says is not
needed as the kernel is indexed, and a control that hands off one level early and lets
each lane add two slots that other threads wrote. The check requires the first to be right
in all three thread orders and the control to go wrong in at least one.
