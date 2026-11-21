# Snippets

Code that chapter 3's lessons show for the sake of an explanation or a question, and that
is not a lab. The files are kept here as real `.cu` (and `.metal`) sources so that the CUDA
front-end check in `./verify.sh` parses every `.cu` file, and so that a lesson shows the
checked file and not a copy that can drift. They are not built or run by the lab runner;
`check.py` (below) runs the ones a lesson calls emulator-checked.

How lessons use them:

- loaded at run time with `useLabFile('ch03/snippets/<name>')`; or
- copied into a section file, with a unit test beside the section that fails if the copy
  and the file differ, and that pins any line the lesson refers to by number. Edit the
  test with the file.

| File | Shown in | How | Test in `app/src/chapters/ch03/` |
|---|---|---|---|
| `vec_add_1d.cu`, `launch_1d.cu` | "Before you start", section 3.1 | loaded | `S31.snippets.test.ts` (line numbers) |
| `launch_fig31.cu`, `launch_checked.cu` | section 3.1 | loaded | `S31.snippets.test.ts` (line numbers) |
| `grayscale_kernel.cu`, `grayscale_kernel.metal` | section 3.2, listing and stepper | copied | `S32.snippets.test.ts` |
| `grayscale_as_printed.cu` | section 3.2, erratum hunt | copied | `S32.snippets.test.ts` |
| `blur_kernel.cu` | section 3.3, listing and stepper | copied | `S33.snippets.test.ts` |
| `matmul_kernel.cu` | section 3.4, listing, stepper and launch | copied | `S34.test.ts` |
| `matvec_options.cu` | section 3.4, book exercise 2 (the four options) | copied | `S34.test.ts` |
| `blur_unsigned.cu` | section 3.3, "Signed arithmetic" (described, not shown) | none | run by `check.py` |
| `scale_rows_defect.cu` | summary page, first question | copied | `Wrap.snippets.test.ts` |

Some of these are wrong or poor on purpose. `grayscale_as_printed.cu` holds Fig. 3.4's code lines as
the book prints them, cast and constants included; it parses, and it must never be the
basis of a lab. `scale_rows_defect.cu` reads an element that another thread writes; the
summary page's question is about that. `blur_unsigned.cu` keeps every index unsigned, so two of
its clauses are always true; it works and should not be copied. In `matvec_options.cu` only `matvecOption2` is right.

## Running them: `check.py`

```sh
python3 labs/ch03/snippets/check.py            # build in a temporary directory
python3 labs/ch03/snippets/check.py --build DIR
```

It translates each snippet and the matching driver in `check/` with `labs/tools/cu2emu.py`,
compiles them as the lab runner compiles the `emu` backend (AddressSanitizer on, `-Wall
-Wextra`), runs every case under the emulator's three thread orders, and fails if anything a
lesson states about the snippet does not hold:

| Snippet | What is checked | Stated in |
|---|---|---|
| `launch_checked.cu` | `(32, 32, 2)` rejected, buffer untouched, error state reset; `(1, 1, 128)`, a 65,536-block grid along y and a zero dimension rejected; the emulator rejects 1 MiB of dynamic shared memory | section 3.1 |
| `scale_rows_defect.cu` | column 0 ends at 1.0; 4, 64 and 37 of 200 elements are 1.0 in order, reversed, shuffled | summary page |
| `blur_kernel.cu` | matches a CPU reference on six picture sizes | section 3.3 |
| `blur_unsigned.cu` | same output as `blur_kernel.cu`; `-Wall -Wextra` warns about signed/unsigned comparisons (and not for `blur_kernel.cu`) | section 3.3 |
| `grayscale_kernel.cu` | within one grey level of a CPU reference; red, green, blue, white give 76, 149, 29, 255 | section 3.2 |
| `matvec_options.cu` | option 2 computes B c, option 3 computes Bᵀ c (N = 37, blocks of 16) | section 3.4 |

The drivers are plain CUDA. On a CUDA device `launch_checked_test.cu` would check the same
documented limits as on the emulator; only its dynamic-shared-memory case is emulator-only.
It is not for HIP: AMD documents block z up to 1024, so `(1, 1, 128)` is valid there. No
build for a device exists yet: nothing here has run on a GPU.
