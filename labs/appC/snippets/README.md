# Snippets

Code that Appendix C's lessons show for the sake of an explanation, and that is not a lab.
They are kept here as real files so that `./verify.sh` runs the CUDA front-end check on the
`.cu` ones, and unit tests fail if the copy a lesson shows drifts from the file. The lab
runner does not build or run them; `check.py` (below) compiles and runs the Metal ones.

| File | Shown in | Checked by |
|---|---|---|
| `c1_helper_generic.cu` | section C.1, "The helper, in both languages", CUDA tab: one `rowMax` with a plain pointer, called on global and on shared memory | CUDA front-end check; `app/src/chapters/appC/SC1.snippets.test.ts` checks that the lesson's copy is this file |
| `c1_helper_overloads.metal` | the same block, Metal tab: the pair of `rowMax` overloads MSL needs, one per address space | `SC1.snippets.test.ts`, as above; `check.py` compiles it at every MSL version from 2.0 to 4.1 and runs its kernel on the GPU against a CPU reference (the tab's "verified on Metal") |
| `c3_chase.cu` | section C.3, "Follow the list": the one-thread kernel that walks a host-built linked list | CUDA front-end check; `app/src/chapters/appC/SC3.snippets.test.ts` checks the lesson's copy and the numbers its question states |
| `c6_sortfile_cpu.cu` | section C.6, "Fig. C.1, line by line": the CPU original (Fig. C.1, left box, p. 627) | CUDA front-end check; `app/src/chapters/appC/SC6.snippets.test.ts` checks that the stepper's lines (`app/src/sim/managedmem.ts`, `figLines('cpu')`) are this file's `sortfile`, character for character |
| `c6_sortfile_managed.cu` | the same stepper: the managed port (Fig. C.1, right box) | as above, `figLines('managed')`; the test also checks that it differs from the CPU original in exactly the four lines the figure marks |
| `c6_sortfile_explicit.cu` | the same stepper: our third column, the port done with `cudaMalloc` and two `cudaMemcpy` calls (not in the book) | as above, `figLines('explicit')` |
| `c9_vmm_decls.h`, `c9_vmm_decls_hip.h` | section C.9, the "CUDA Driver API" and "HIP" tabs: the reserve, create, map and grant declarations | not compiled anywhere (the CUDA front-end stub has no Driver API, and the front-end check reads only `.cu` files); `app/src/chapters/appC/SC9.test.ts` checks that the lesson's copies are these files and that every declaration in them is quoted in `notes/today/appC-facts.md` (A15, B7) |
| `wrap_layer_table.cu` | the summary page, the first question of "A mixed check": a managed table of layers holding `cudaMalloc` pointers, with a host read of one of them on purpose | CUDA front-end check; `app/src/chapters/appC/Wrap.snippets.test.ts` checks the page's copy and the numbers of the page's two numeric questions |

The three `c6_` files are host code only: the book shows no definitions of the sort or of
`use_data`, so both are declared and never defined. Two things in them differ from the
page: the missing semicolon after `cudaDeviceSynchronize()` is added (as printed, the
front-end check stops there), and the elided `<<<...>>>` becomes `grid` and `block`,
declared once at file scope as stand-ins. The figure's name `qsort_char` is kept, although
the book's text says `qsort()`.

## Running the Metal ones: `check.py`

```
python3 labs/appC/snippets/check.py              # build in a temporary directory
python3 labs/appC/snippets/check.py --build DIR  # or $PMPP_BUILD_DIR/appC-snippets
```

Needs macOS with Xcode's Metal toolchain (`xcrun metal`). It checks:

| What | Expected | Stated in |
|---|---|---|
| `c1_helper_overloads.metal`, compiled with every `-std` the compiler offers for macOS (`macos-metal2.0` to `metal4.1`) and with none | compiles every time | section C.1, the Metal tab's caption |
| the same file, run on the GPU by `check/c1_run.mm` (kernel `row_max_twice`, several shapes) | both maxima of every row equal a CPU reference | the Metal tab's status, "verified on Metal" |
| the five MSL cards of section C.1 (`app/src/sim/addrspace.ts`, `CARDS`), each written out as a helper and a kernel | up to `metal4.0`: where the specification puts them (`msl-none` fails with "pointer type must have explicit address space qualifier", `msl-device-only` with "no matching function for call", the other three compile); at `metal4.1` and the default only `msl-none` changes, and compiles | section C.1, the cards' footnote and their "observed" switch |
| `check/c1_generic_probe.metal`, a helper with an unqualified pointer | rejected up to `metal4.0`, accepted at `metal4.1` and the default; run on the GPU, it returns the right maxima from device, threadgroup and thread memory | section C.1, "Observed on the build machine"; `notes/today/appC-facts.md` C1 |

`check/c1_generic_probe.metal` is not valid MSL by its specification and is not shown as
code to copy. Nothing in `check.py` runs on CUDA or HIP.
