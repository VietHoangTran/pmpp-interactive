# Lab 3.1: colour to grayscale

Book: section 3.2 (pp. 48-55), Fig. 3.4 (p. 53) and the launch on p. 50.

Write the kernel that turns an RGB picture into a grey one, and its launch: one thread
per pixel, a 2D grid of 2D blocks rounded up to whole blocks, a range test for the
threads that land outside the picture, and two flat offsets (one byte per pixel going
out, three coming in). The grey level is L = 0.299 r + 0.587 g + 0.114 b, truncated to a
byte.

- Edit `work/kernel.cu` (CUDA / HIP / emulator): `colorToGrayscaleKernel` and
  `launchGrayscale`. The block shape is an argument of the launch; do not hard-code 16.
- Edit `work/kernel.metal` + `work/launch.mm` (Metal): the kernel reads plain `uchar`
  buffers and takes its position from `thread_position_in_grid`; the launch binds the two
  sizes with `setBytes` and uses `dispatchThreads`, so the grid is exactly the picture.
- Test: `python3 labs/pmpp.py run 3.1`   Start with `--backend emu`.
- Measure: `python3 labs/pmpp.py run 3.1 --bench`

Sizes are written the way the book writes them, rows first: 62 by 76 is 62 rows of 76
pixels, so `height` is 62 and `width` is 76.

## Do not copy Fig. 3.4

The solution differs from the printed listing in four places. Find them yourself first
if you like; the tests will find them for you otherwise.

1. **The cast.** Line 19 prints `(unsigned char) 0.21f*r+0.71f*g+0.07f*b`. A cast binds
   tighter than `*`, so it applies to `0.21f` alone, which becomes 0, and the red term
   is lost. The compiler says nothing. Parenthesise the whole sum.
2. **The constants.** The listing prints 0.21, 0.71, 0.07. The text on p. 52 gives
   0.299, 0.587, 0.114, which is what this lab uses. They add up to 1, so white stays
   white; the listing's add up to 0.99.
3. **The argument order.** The listing takes `(Pout, Pin, width, height)`, and the host
   call on p. 50 passes `(Pin_d, Pout_d, m, n)`. Here the input comes first in both.
4. **The name.** The listing prints `colortoGrayscaleConvertion`, the text
   `colorToGrayscaleConversion`. Here it is `colorToGrayscaleKernel`.

Two smaller things: `Pin` is `const` here, and the grid is sized with integer round-up
where p. 50 prints `ceil(m/16.0)`.

The Metal kernel is not in the book. It has the same cast trap, because the grammar is
the same. It has one trap of its own: `uchar3` is four bytes in the Metal Shading
Language, so a `uchar3` pointer over three-byte pixels reads the wrong data.

## What the tests check

Your output is compared with a CPU reference, allowing one grey level either way. The
cast truncates, so the order in which a compiler adds the three products can move the
byte by one.

- Flat pictures first: pure red, green and blue (76, 149 and 29) and white (255).
- Then a gradient with noise at 62 by 76 (the book's picture), 16 by 16, 1 by 1,
  1 by 300, 300 by 1, 33 by 17 and 64 by 80, and on a real device 1500 by 2000. Each
  runs with blocks of 16 by 16, 32 by 8 and 8 by 32 threads.
- The output buffer is filled with a sentinel value before each launch. A pixel that
  still holds it was never written, and the failure message counts those.
- On the emulator the buffers are exactly the size of the picture and the build uses
  AddressSanitizer, so the first access outside a buffer stops the run and names the
  line of your kernel. On a real device nothing checks bounds: there the buffers are
  larger than the picture and the bytes after it must come back unchanged.

## Before you benchmark

The benchmark times one large picture with blocks of 8 by 8, 16 by 16, 32 by 32,
256 by 1 and 1 by 256 threads, skipping any shape your device does not allow.

A wide block (256 by 1) and a tall block (1 by 256) launch the same number of threads,
and every thread does the same work. What differs is which pixels one block's threads
hold: a run along a row, or a run down a column. NVIDIA's programming guide says "The use
of multi-dimensional thread blocks and grids is for convenience only and does not affect
performance"; that sentence is about writing one mapping in one, two or three dimensions.
The same passage goes on: threads are numbered x fastest, and "This affects how threads
are assigned to warps". Write down the ratio you expect, tall time divided by wide time,
then run it. The benchmark prints the ratio it measured. Why it comes out as it does is
the subject of chapters 4 and 6.
