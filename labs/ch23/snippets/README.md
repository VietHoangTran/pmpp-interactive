# Snippets

Code that chapter 23's lessons show and that is not a lab. The files are kept here as real
sources so that the CUDA front-end check in `./verify.sh` parses the `.cu` ones, and so that a
lesson shows the checked file and not a copy that can drift. The lab runner does not build
them; `check.py` runs them.

| File | Shown in | How | Test in `app/src/chapters/ch23/` |
|---|---|---|---|
| `staged_loop.cu` | section 23.3, the stepper "Copy, compute, copy back", CUDA tab: the staged loop, issued depth first and breadth first | loaded with `useLabFile`; the stepper highlights the lines `app/src/sim/chunkpipe.ts` (`LINES`) says issued each operation | `S233.snippets.test.ts` checks that those lines are the calls the model says they are |
| `staged_loop_metal.mm`, `staged_loop.metal` | the same stepper, Metal view: K shared buffers in flight on Apple silicon | as above | as above |

Both are ours, after NVIDIA's staged copy-and-execute loop (CUDA C++ Best Practices Guide,
section 10.1.2; `notes/today/ch23-facts.md` A7) and Apple's sample "Synchronizing CPU and GPU
work" (C5). They split the job the way lab 23.1 does: chunk `c` is `[c*n/chunks,
(c+1)*n/chunks)`, empty chunks are skipped, and the j-th non-empty chunk goes to stream (or
slot) `j % S`. They are deliberately smaller than the lab: no error checking, and the caller
creates and synchronises the streams.

## Running them: `check.py`

```sh
python3 labs/ch23/snippets/check.py            # build in a temporary directory
python3 labs/ch23/snippets/check.py --build DIR # or $PMPP_BUILD_DIR/ch23-snippets
```

| Snippet | What is checked | Status it supports |
|---|---|---|
| `staged_loop.cu` | the CUDA front end (clang, the course's stub headers) accepts it and its driver `check/staged_loop_test.cu` | front-end-checked |
| `staged_loop.cu` | on the CPU emulator (translated by `labs/tools/cu2emu.py`, AddressSanitizer on), both issue orders give `2x + 1` for every element, with pinned host buffers, for equal chunks, uneven chunks (lengths 2, 3, 2, 3), more chunks than elements (empty chunks), 1 element in 64 chunks, more streams than chunks, one stream, and n = 0; nothing is written outside the output | emulator-checked |
| `staged_loop_metal.mm` with `staged_loop.metal` | built with clang++ and run on this Mac's GPU (the kernel compiled at run time) for K = 1, 2, 3 and more buffers than chunks, with equal, uneven and empty chunks and n = 0, against a CPU reference with a tolerance; nothing written outside the output | verified on Metal |

The emulator runs streams synchronously, so it checks what each chunk computes and where it
lands, never whether anything overlapped; overlap is lab 23.1's measurement. Nothing here has
run on CUDA or HIP hardware.

## Section 23.4: the book's loops (front-end-checked only, never run)

| File | Shown in | Status | Test in `app/src/chapters/ch23/` |
|---|---|---|---|
| `overlap_loop.cu` | section 23.4, beside the puzzle "What enforces each ordering?": the marked regions `[setup]` and `[loop]` (Fig. 23.14 with its line numbers 01 to 43, line 19 corrected) | front-end-checked | `S234.snippets.test.ts`: the lesson's copies equal the dedented regions; every line `app/src/sim/haloloop.ts` and the puzzle cite holds the call they name; the front-end check passes, and fails on line 19 as the page prints it |
| `nccl_exchange.cu` | section 23.4, the NCCL paragraph: region `[exchange]`, Fig. 23.20 lines 27 to 44 | front-end-checked | the same test |
| `check/mpi_nccl_decls.h` | nothing (declarations only) | | included by both; declares the MPI, NCCL and CUDA priority calls the course stub lacks, with signatures from `notes/today/ch23-facts.md` A1, A10, A12 |

Both files need MPI (and NCCL) and one GPU per rank, which the course does not assume, so they
are parsed and never built or run. `check.py` does not touch them; the vitest file above runs
clang itself, with the same flags as `labs/pmpp.py`, and is skipped when no clang++ is found.

## Summary page: one question's code (front-end-checked only, never run)

| File | Shown in | Status | Test in `app/src/chapters/ch23/` |
|---|---|---|---|
| `rerecord.cu` | the summary page, question `ch23/wrap/q-rerecord`: region `[shown]`, one event recorded twice in stream `s1` with a wait in `s2` between the two records | front-end-checked | `Wrap.snippets.test.ts`: the page's copy equals the region; the front-end check passes |

Ours, not the book's. What the question asks (a wait takes the event as it was recorded when the
wait was issued, `notes/today/ch23-facts.md` A5) cannot be shown on the course's emulator, which
runs every stream call synchronously, so the file is parsed and never run. `rerecordFixed` in
the same file is the order the question's explanation gives.
