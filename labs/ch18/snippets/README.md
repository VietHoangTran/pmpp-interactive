# Snippets

Kernels that chapter 18's pages 18.3 and 18.6 show for the sake of the explanation, and
that are not labs. The pages load them at run time with `useLabFile('ch18/snippets/<name>')`
and show either a region (the text between `// region: <name>` and `// endregion`) or one
function, so there is no copy to drift. They are not built by the lab runner. The summary
page shows no kernel and has no file here.

| File | Shown in | Checked by |
|---|---|---|
| `s183_variants.cu` | page 18.3: Figs. 18.6, 18.8, 18.10, 18.12 and 18.14 as printed and as corrected (relaxed atomics for the benign races; a HIP branch with the legacy `atomicCAS` in Fig. 18.14), plus the eight visit forms of the stepper "Two threads, one neighbour", five of them wrong on purpose | CUDA front-end check; `app/src/chapters/ch18/S183.snippets.test.ts` pins every region and runs it through `check/s183_check.cu` |
| `check/s183_check.cu` | nothing (a driver) | CUDA front-end check. Built and run by `S183.snippets.test.ts` on the `emu` flavour in forward and reverse order: whole searches with each of the eight kernels, compared with `app/src/sim/bfs.ts`, and one level of each stepper scenario with each visit form, compared with `app/src/sim/bfsvisit.ts`; `visitWeakReset` is run under a time limit and must not return |
| `s186_privatised_variants.cu` | page 18.6, the stepper "One block's private frontier": Fig. 18.15 corrected (an atomic store, or a clamp, for line 29) and copies with the page's plain store, each block barrier removed, an `if` commit and no overflow check | CUDA front-end check; `app/src/sim/bfspriv.test.ts` pins each kernel body to `app/src/sim/bfspriv.ts` and runs it through `check/s186_privatised_check.cu` on `emu` (three thread orders) and `race` (ThreadSanitizer, which reports the plain store); `S186.snippets.test.ts` pins what the page quotes |
| `s186_coop_swap.cu` | page 18.6, the stepper "Three grid barriers" and the launch listing: Fig. 18.17's frontier swap without the private queue, copies with each grid barrier removed, Exercise 3's one-barrier kernel, and the book's sizing and six-argument launch | CUDA front-end check; `app/src/sim/gridbfs.test.ts` pins each kernel body to `app/src/sim/gridbfs.ts` and runs it through `check/s186_coop_check.cu` on `emu`, including a grid one block too large |
| `check/s186_privatised_check.cu`, `check/s186_coop_check.cu` | nothing (drivers) | CUDA front-end check (each includes its snippet when parsed on its own). The tests append each driver to its snippet with `PMPP_SNIPPET_APPENDED` defined, translate the result with `labs/tools/cu2emu.py` and build it as the lab runner builds the emulators |

Nothing here has been run on a GPU. The pages give the status "emulator-checked and
front-end-checked; unverified on CUDA and HIP hardware". The HIP branches are written from
`notes/today/ch18-facts.md` (B1, B2, B4) and have not been compiled for HIP.

The CUDA front-end check is the one `labs/pmpp.py` runs: `clang++ -x cuda --cuda-host-only
-nocudainc -nocudalib -fsyntax-only -std=c++17 -I labs/common/include -I
labs/common/include/pmpp/emu/shim -include pmpp/check/cuda_stub.h <file>`.
