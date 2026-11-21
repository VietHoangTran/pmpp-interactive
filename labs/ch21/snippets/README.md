# Snippets

Kernels that chapter 21's lesson pages show, and that are not labs. They are kept here as
real `.cu` files so that the CUDA front-end check in `./verify.sh` parses every one, so
that `check.py` can run them on the CPU emulator, and so that a lesson shows the checked
file and not a copy that can drift. They are not built by the lab runner.

How the lesson uses them: section 21.2 (`app/src/chapters/ch21/S212.tsx`) holds a copy of
the part of each file below its marker line (`// ---- shown in the lesson ----`, or
`// ---- as printed ----`). `app/src/chapters/ch21/snippets.test.ts` fails if a copy and
its file differ, and pins the lines the page refers to. Edit the test with the file.

| File | What it is | Shown in |
|---|---|---|
| `scatter.cu` | Fig. 21.5 with an atom count and a bound on it (the listing has neither) | section 21.2, "Scatter" |
| `gather.cu` | Fig. 21.6 with a bound check on `i` and `j` (the listing relies on padding) | section 21.2, "Gather" |
| `coarsen_printed.cu` | Fig. 21.8's 34 code lines as printed, line numbers kept, the one comment reworded | section 21.2, erratum hunt |
| `coarsen_adjacent.cu` | Fig. 21.8 corrected: line 05 scales the thread index by the factor; a bound check per point | section 21.2, "Coarsening" |
| `coarsen_strided.cu` | Fig. 21.10 with a bound check per point | section 21.2, "Coalescing" |
| `bias_gather.cu` | ours, not the book's: a logit bias in the gather form of Fig. 21.6, for one question; front-end-checked only, never run (not in `check.py`) | summary page, mixed check (tied by `Wrap.snippets.test.ts`) |

`coarsen_printed.cu` is wrong on purpose and must never be the basis of a lab: as printed,
line 05 adds an unscaled `threadIdx.x` while lines 30 to 33 write `i` to `i + 3`, so
neighbouring threads overlap with a plain `+=` and most of each block's points are never
written (`notes/ch21.md`, "Unclear in the PDF"; `notes/today/ch21-facts.md` E2).

Every corrected file computes the slice index as `k = z / gridspacing` (Fig. 21.6's form;
Fig. 21.5 writes `(grid.x*grid.y*z) / gridspacing`). The division is exact for the
book's 0.5 Å spacing, which is what the checks use; a production kernel would pass `k` as
an integer. `sqrtf` needs no declaration for the front-end check: its stub header declares
the single-precision math functions.

## Running them: `check.py`

```sh
python3 labs/ch21/snippets/check.py            # build in a temporary directory
python3 labs/ch21/snippets/check.py --build DIR
```

It compiles each snippet with the driver `check/dcs_test.cu` (one `-DKIND_...` per
snippet, `CHUNK_SIZE` 64), translated by `labs/tools/cu2emu.py` and built as the lab runner
builds the `emu` backend (AddressSanitizer on, `-Wall -Wextra`, warnings fail the check).
The driver does what p. 517 describes on the host: per slice, copy the slice in, copy the
atoms into constant memory one chunk at a time with one launch per chunk, copy the slice
back. Every case runs under the emulator's three thread orders.

| Snippet | What is checked | Stated in |
|---|---|---|
| `scatter.cu` | equals a float64 sum within 1e-5 of the sum of absolute contributions, over two slices and three chunks, for three launch shapes; nothing written past the grid | section 21.2 |
| `gather.cu` | the same, on slices that are not a multiple of the block | section 21.2 |
| `coarsen_adjacent.cu` | the same, on slices that are not a multiple of 4 × `blockDim.x` | section 21.2 |
| `coarsen_strided.cu` | the same | section 21.2 |
| `coarsen_printed.cu` | at `blockDim.x` = 32 on a 128-point row: points 35 to 127 never written, 3 to 31 four times, 0 to 2 and 32 to 34 one to three times; the result differs from the sum | section 21.2, erratum hunt |

What this cannot show: the emulator runs one thread at a time, so it cannot see that the
scatter kernel needs its atomics (a plain `+=` would pass here too; the `race` backend of
lab 21.1 is the tool for that), and it models no coalescing, caches or time. Nothing here
has run on a GPU.
