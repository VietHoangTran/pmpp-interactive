# Snippets

Code that chapter 16's lessons show and that is not a lab. The files are kept here as real
`.cu` sources so that the CUDA front-end check in `./verify.sh` parses them, and so that a
lesson shows the checked file and not a copy that can drift. They are not built by the lab
runner; `check.py` runs the ones a lesson calls emulator-checked.

| File | Shown in | How | Test in `app/src/chapters/ch16/` |
|---|---|---|---|
| `fw_bottomup.cu` | section 16.3: the Floyd-Warshall kernel (with the stepper's highlighted lines) and its host loop | copied | `snippets-s163.test.ts` |

`fw_bottomup.cu` is Fig. 16.4 corrected, three changes from the page:

1. "No path" is the integer `FW_INF = INT_MAX / 2`. The page compares an `int` with
   `<cmath>`'s `INFINITY`, a `float` that no `int` equals, so its early exits never fire, and
   with `INT_MAX` as "no edge" the sum would overflow (ch16-facts A3). Half of `INT_MAX`
   leaves room for one addition.
2. No thread returns before `__syncthreads()`. The page returns out-of-range threads first;
   CUDA documents that exited threads do not hold a barrier, HIP does not document it, and
   Metal documents it for Apple silicon only (ch04-facts A6, B3, C3). An `inRange` flag
   guards the work instead. The `return` after the barrier depends only on the shared
   value, so it is the same for the whole block.
3. The host loop has no `cudaDeviceSynchronize()` between launches: launches in one stream
   run in order (ch02-facts A5).

The lesson shows the kernel and the host loop as two listings, each verbatim in the file;
the test checks that, and pins the line numbers the stepper highlights.

## Running them: `check.py`

```sh
python3 labs/ch16/snippets/check.py            # build in a temporary directory
python3 labs/ch16/snippets/check.py --build DIR
```

It translates each snippet and its driver in `check/` with `labs/tools/cu2emu.py`, compiles
them as the lab runner compiles the `emu` backend (AddressSanitizer on, `-Wall -Wextra`),
and runs every case under the emulator's three thread orders.

| Snippet | What is checked | Stated in |
|---|---|---|
| `fw_bottomup.cu` | every step equals a sequential CPU step on Fig. 16.5's table as printed, with 8, 3 and 32 threads per block; after steps a and b, step c changes exactly a to e (23), b to e (19), g to e (23) and h to e (13); the host loop equals the CPU on random graphs with non-negative weights, 1 to 64 vertices, with blocks that leave threads out of range | section 16.3 |

The emulator runs one thread at a time; it cannot show what the in-place update would do
if row k or column k changed during a step (that needs a negative `dist[k][k]`, which the
lesson's stepper models instead). Nothing here has run on a GPU.
