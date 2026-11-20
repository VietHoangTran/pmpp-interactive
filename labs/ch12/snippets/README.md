# Snippets

Code that chapter 12's lessons show for the sake of an explanation, and that is not a
lab. Lessons load these files at run time with `useLabFile('ch12/snippets/<name>')`, so
there is no copy to drift. They are not built by the lab runner.

| File | Shown in | Checked by |
|---|---|---|
| `s123_fig12_3.cu` | section 12.3, "Fig. 12.3, step by step" and "Which line is which?" (the book's lines 01 to 29; line 08 gives `j` a value) | CUDA front-end check; `app/src/chapters/ch12/S123.snippets.test.ts` pins the book's lines and the lines the lesson highlights, and runs it on the emulator through `check/s123_check.cu` |
| `s123_fig12_4_cg.cu` | section 12.3, "With cooperative groups" (Fig. 12.4's 26 lines, with the header and namespace the page leaves out, `num_threads()` for the legacy `size()`, and `j = 0`) | CUDA front-end check; `S123.snippets.test.ts`, as above, with `check/s123_check.cu` built with `-DSNIPPET_CG` |
| `check/s123_check.cu` | nothing (a driver) | CUDA front-end check, both variants. Built and run by `S123.snippets.test.ts` (as the lab runner builds the `emu` backend, AddressSanitizer on): in the three thread orders the slots each passing lane writes must match `app/src/sim/warpvote.ts` lane by lane, and 4,000 keys must come out right with each warp's kept keys side by side. Skipped when no C++ compiler or python3 is found. |
| `s124_privatised_variants.cu` | section 12.4, the stepper and the outcome table (Fig. 12.6 at four threads per block, and four copies that drop one barrier each or bump the private counter with `++`) | CUDA front-end check; `app/src/sim/privfilter.test.ts` pins every kernel body to `app/src/sim/privfilter.ts` and runs the file on `emu` (three orders) and `race` through `check/s124_privatised_check.cu` |
| `check/s124_privatised_check.cu` | nothing (a driver) | CUDA front-end check. `privfilter.test.ts` appends it to the snippet, translates the result with `labs/tools/cu2emu.py`, and runs one kernel and one order per process, so that a run AddressSanitizer stops does not hide the others. |
| `wrap_pairs.cu` | the summary page, the kernel of the first mixed-check question (`candidatePairs`, ours, wrong on purpose: a pair's two slots come from two counters) | CUDA front-end check; `app/src/chapters/ch12/Wrap.snippets.test.ts` pins the two `fetch_add` calls the question is about and runs it on the emulator through `check/wrap_pairs_test.cu` |
| `check/wrap_pairs_test.cu` | nothing (a driver) | CUDA front-end check. Built and run by `Wrap.snippets.test.ts` (as the lab runner builds the `emu` backend, AddressSanitizer on): in all three thread orders both counters must be right and every pair must match, which is the question's point (emu runs one thread to its end before the next, so nothing runs between a thread's two atomics); a copy that takes one slot per pair must pass too. Skipped when no C++ compiler or python3 is found. |

`wrap_pairs.cu` and the Fig. 12.4 snippet include `<cuda/atomic>` without a HIP fallback.
They are for the CUDA front-end check and the emulators; lab 12.1 shows the portable
spelling. On HIP, `<cuda/atomic>` comes from libhipcxx, which the course's HIP build links
only where ROCm provides it (`notes/today/infra-facts.md` B2).

The CUDA front-end check is the one `labs/pmpp.py` runs: `clang++ -x cuda --cuda-host-only
-nocudainc -nocudalib -fsyntax-only -std=c++17 -I labs/common/include -I
labs/common/include/pmpp/emu/shim -include pmpp/check/cuda_stub.h <file>`. Without the
`shim` include, `<cuda/atomic>` is not found.
