# Snippets

Code that module X1's lessons show and that is not a lab. The files are real sources so that
the CUDA front-end check in `./verify.sh` parses the `.cu` ones, and so that a lesson shows the
checked file and not a copy that can drift. They are not built by the lab runner. Drivers in
`check/` run them; each driver's test is in `app/src/sim/`. One row per snippet. Everything here
is ours: sampling is not in the book.

| File | Shown in | How | Driver | Test |
|---|---|---|---|---|
| `philox.cu` | section X1.7, "The code": Philox4x32-10, M5's counter and key mapping, and the 24-bit conversion to u | loaded at run time | `check/philox_kat.cu` | `app/src/sim/philox.test.ts` |
| `philox.metal` | section X1.7, the Metal tab of the same panel | loaded at run time | `check/philox_metal_run.mm` | `app/src/sim/philox.test.ts` |
| `draw_search.cu` | section X1.8: the final draw, one block per row, with the two barriers the stepper lets you remove | loaded at run time | `check/draw_check.cu` | `app/src/sim/draw.test.ts` |

The summary page shows no kernel of its own, so there is no wrap snippet.

## `philox.cu` and `philox.metal`

Random123's round, constants and key schedule, written here (not copied). `philox.cu` forms the
32 x 32 to 64-bit product as an `unsigned long long` multiplication: the course's emulator and
the front-end stub have no `__umulhi` (`notes/today/x1-facts.md`, project notes). `philox.metal`
takes the high half from MSL's `mulhi` and the low half from the wrapping 32-bit product
(x1-facts C4), as Random123's own MSL does. Lab X1.3's `kernel.metal` uses the other form, a
`ulong` product; both pass the same known-answer vectors.
Random123's BSD-style notice is reproduced in both files; cuRAND's headers are under NVIDIA's
licence and are not used. What the test checks:

- `philox.cu`, translated by `labs/tools/cu2emu.py` and run on the emulator: Random123's three
  10-round known-answer vectors (x1-facts A4), computed by a kernel and on the host, and u for 64
  (seed, stream, step) rows, all compared with `app/src/sim/philox.ts`;
- `philox.metal`, compiled at Metal 3.2 and 4.1 and run on the Mac's GPU by
  `check/philox_metal_run.mm`: the same vectors and the same 64 values of u.

Status: `philox.metal` is verified on Metal (compiled and run here by the test);
`philox.cu` is emulator-checked and front-end-checked, unverified on CUDA and HIP hardware.

## `draw_search.cu`

M5's draw on its own: the total by a two-stage block reduction (section 10.8), then chunks of
`blockDim.x * 4` weights with a running aggregate, a block scan of the crossing chunk (section
11.4), and the first lane whose inclusive sum is strictly greater than u x total by a ballot
(section 12.3), with the fallback to the last positive weight. What the test checks:

- on the emulator, in all three thread orders, 27 rows of tokens against `app/src/sim/draw.ts`;
- with barrier A removed (compiled with a define), the tokens the model's `huntDraw` predicts for
  the in-order and reversed schedules, some of them wrong;
- with barrier B removed, the right tokens in all three emulator orders (the emulator cannot see
  this write after a read across warps), and on the race flavour a ThreadSanitizer report naming
  the read of the total and the warp-total write; with both barriers in place, no report.

The emulator and race builds make `draw.test.ts` take about a minute. Each part is skipped
where no C++ compiler, `python3` or working ThreadSanitizer is found.

Status: emulator-checked and front-end-checked, unverified on CUDA and HIP hardware. There is no
Metal version of this snippet; lab X1.3 holds the Metal draw (`labs/x1/lab3_sampler`).
