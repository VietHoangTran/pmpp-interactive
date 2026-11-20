# Lab 3.5: map hunt

Book: sections 3.2 and 3.3 (pp. 48-59). The kernel is not a listing from the book. It is
built from two things that are: the 3D thread-to-data mapping (p. 55) and the blur
kernel's clipped loop with its counter (Fig. 3.8, p. 57), kept to one axis.

`work/` holds a windowed mean along the sequence axis of a `[batch, seq, hidden]` float
tensor:

    out[b][t][h] = mean of in[b][s][h] over s = t-r .. t+r, for the s that exist

Row-major, batch first. Hidden is on x, seq on y, batch on z, in blocks of 8 x 8 x 4
threads. It compiles, and it has **four independent defects**. Find and fix all four.

- Edit `work/kernel.cu` (the kernel `seqMeanKernel` and its launch `launchSeqMean`), and
  `work/kernel.metal` + `work/launch.mm` (`seq_mean`, `encodeSeqMean`). The same four
  defects are in the CUDA file and in the Metal pair.
- Test: `python3 labs/pmpp.py run 3.5 --backend emu`, then `python3 labs/pmpp.py run 3.5`.

## Before the first run

The first test is a cube. Each of the next four moves away from it in one direction:

| Test | batch | seq | hidden | r |
|---|---|---|---|---|
| `cube` | 4 | 16 | 16 | 0 |
| `wide` | 4 | 6 | 20 | 0 |
| `tall` | 4 | 20 | 6 | 0 |
| `odd batch` | 5 | 16 | 16 | 0 |
| `window` | 4 | 16 | 16 | 1 |

Read the kernel and the launch, and write down which of the five you expect to fail.
Then run and compare. `cube` passes with all four defects present. That is the point of
it: a test whose sizes are all equal, all multiples of the block shape, and whose window
is one element wide cannot tell a correct mapping from this one. The tests named `mixed`
after the five are mixtures, and pass once all four are gone.

## Method

1. Emulator first: `python3 labs/pmpp.py run 3.5 --backend emu`. Fix one thing, run
   again. The failures change as you go.
2. Know what each kind of failure looks like there:
   - **An access outside the tensor.** The emulator's buffers are exactly the tensor, so
     AddressSanitizer stops the run at the first one. The report is long. Read three
     things near its top: `READ` or `WRITE`, the first frame (`kernel.cu:` and a line
     number), and the sentence saying how many bytes before or after the buffer the
     address is. The tests after the stop are not run.
   - **An access to the wrong element inside the tensor.** The address is valid, so
     AddressSanitizer says nothing. You get wrong values, and the test message gives the
     first wrong element and how many elements no thread wrote.
   - **A mistake in host code.** The launch is ordinary C++ that computes a few integers.
     No tool flags a grid of the wrong shape. You see only its effect on the output.
3. Each test runs under three thread orders. A correct kernel gives the same output under
   all three, because each element has one writer. If a test reports different wrong
   values from one order to the next, the result depends on which thread ran last: two
   threads are writing the same element. `python3 labs/pmpp.py run 3.5 --backend race`
   runs the threads concurrently under ThreadSanitizer, which reports such a pair with
   its source line (`python3 labs/pmpp.py doctor` says whether the `race` backend can be
   built with your compiler).
4. Use the shapes. A test that fails moved away from `cube` in one direction, and that
   direction is what the defect needs in order to show.
5. Then run on your GPU: `python3 labs/pmpp.py run 3.5`. A plain run there names no
   line. On a device the driver puts guard elements before and after each tensor, so a
   stray write lands in memory it owns and is counted afterwards. For each defect, note
   what the GPU run showed you: a wrong value, an element never written, or only a write
   into the guard elements. A real program has no guard elements. There, a defect of the
   last kind writes into whatever follows the tensor, and every result you check is
   still correct.

Nothing to benchmark here.
