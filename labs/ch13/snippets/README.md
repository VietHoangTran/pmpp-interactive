# Snippets

Code that chapter 13's lessons show and that is not a lab. The files are kept here as real
`.cu` sources so that the CUDA front-end check in `./verify.sh` parses them, and so that a
lesson shows the checked file and not a copy that can drift. They are not built by the lab
runner; `check.py` runs the ones a lesson calls emulator-checked.

| File | Shown in | How | Test in `app/src/chapters/ch13/` |
|---|---|---|---|
| `merge_basic.cu` | section 13.3: `co_rank` (with the explorer's highlighted lines), `merge_sequential` and `merge_basic_kernel` | copied | `snippets-s133.test.ts` |

`merge_basic.cu` is Figs. 13.2, 13.5 and 13.9 corrected: `__device__` on the helpers,
`const int*` inputs, ternaries for min and max, Fig. 13.5's comments fixed, integer ceiling
division on Fig. 13.9's line 03, and `k_curr` clamped (ours). Each function keeps the
book's line numbering; the lesson refers to lines by those numbers and the test pins them.
The lesson also derives the "swapped tie tests" variant of `co_rank` from this file (two
operators exchanged); the test checks that exactly those two lines change.

## Running them: `check.py`

```sh
python3 labs/ch13/snippets/check.py            # build in a temporary directory
python3 labs/ch13/snippets/check.py --build DIR
```

It translates the snippet and its driver in `check/` with `labs/tools/cu2emu.py`, compiles
them as the lab runner compiles the `emu` backend (AddressSanitizer on, `-Wall -Wextra`),
and runs every case under the emulator's three thread orders.

| Snippet | What is checked | Stated in |
|---|---|---|
| `merge_basic.cu` | equals `std::merge` on sorted ints with many duplicate keys; the running example on 3 and 4 threads and on more threads than outputs; 1,000 outputs on 3 blocks of 128; empty inputs; nothing written at or past `C[m + n]` | section 13.3 |

Plain ints cannot show stability; lab 13.1 merges key-value pairs for that. Nothing here
has run on a GPU.
