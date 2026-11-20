# Lab 3.2: image blur

Book: section 3.3 (pp. 55-59), Fig. 3.8 (p. 57).

Write the blur kernel and its launch. One thread per output pixel, as in lab 3.1, but
now each thread loops: it averages the input pixels in a square patch around its own
position, leaving out the part of the patch that hangs over an edge of the picture.

- Edit `work/kernel.cu` (CUDA / HIP / emulator): `blurKernel` and `launchBlur`, with
  16 by 16 blocks.
- Edit `work/kernel.metal` + `work/launch.mm` (Metal). The launch uses
  `dispatchThreadgroups`, which takes a number of groups and so rounds up like a CUDA
  grid. The position Metal hands the kernel is unsigned; the offsets inside a patch are
  not.
- Test: `python3 labs/pmpp.py run 3.2`   Start with `--backend emu`.
- Measure: `python3 labs/pmpp.py run 3.2 --bench`

One difference from the page: the book fixes the patch radius with a constant
`BLUR_SIZE` defined outside the kernel. Here the radius is a kernel argument `r`, so the
benchmark can change it. The patch is 2r + 1 pixels on a side. (The comment on line 09
of Fig. 3.8 calls it a "BLUR_SIZE x BLUR_SIZE box"; the text on p. 58 has it right.)

## What the tests check

Your output is compared with a CPU reference, allowing one grey level either way.

- Shapes and radii: 62 by 76 with r = 1, 2 and 3; 1 by 1; 1 by 9 and 9 by 1; 3 by 3
  with r = 3, where the patch is larger than the picture; 33 by 17 with r = 7; on a
  real device also 1500 by 2000. Sizes are rows by columns, as the book writes them.
- A uniform picture must come back uniform.
- A failure is reported as two counts: wrong border pixels (those whose patch hangs
  over an edge) and wrong interior pixels. A wrong border with a correct interior
  points at the inner test or at what you divide by. A wrong interior points at the
  mapping from thread to pixel.
- The output is filled with a sentinel before each launch; a pixel that still holds it
  was never written.
- On the emulator the buffers are exactly the size of the picture and the build uses
  AddressSanitizer, so an access outside a buffer stops the run and names the line. On
  a real device the buffers are larger than the picture and the bytes after the output
  must come back unchanged.
- The first case is the exception to "exactly the size": the kernel is given the top
  62 rows of a 66-row buffer. A read that runs past the right edge of a row lands on
  the start of the next row, which is valid memory, so no bounds checker can object.
  Only the values can show it. In a buffer of exact size the same mistake is caught at
  the last row alone.

## Before you benchmark

The benchmark blurs one large picture at r = 1, 2, 4 and 8.

Going from r = 1 to r = 4, each thread reads 81 values instead of 9. Write down how
many times longer you expect the kernel to take on your GPU, then run it. The benchmark
prints the ratio it measured.
