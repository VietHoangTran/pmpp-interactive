# Lab 15.7: bug hunt in a fast linear layer

Book: sections 15.6 and 15.8 (pp. 361-368), Fig. 15.12; Exercise 2 (p. 370).

`work/kernel.cu` and `work/kernel.metal` hold a fast linear layer in the project's layout,
lab 15.6's kernel: `Y[batch x out] = X[batch x in] * W^T`, W stored (out, in) row-major.
It has **three independent defects**, the same in both files. It compiles, and some
shapes pass. Find and fix all three. `work/launch.mm` is correct.

Before you run anything, read the kernel and write down, for each defect you suspect,
which backends you expect to show it: `emu`, `race`, Metal, CUDA or HIP. Afterwards
compare. Which ones did your GPU hide?

Method (lab 5.4's):

1. `python3 labs/pmpp.py run 15.7 --backend emu` first. The emulator's thread orders are
   deterministic, so a synchronisation defect fails every time, and AddressSanitizer
   stops at the first access outside an allocation and names the line.
2. Fix one thing, run again. The failures change as you go.
3. `python3 labs/pmpp.py run 15.7 --backend race` runs the threads at the same time under
   ThreadSanitizer. It names the two source lines whose accesses collide, which says
   where a barrier is missing. It only sees collisions that happened in that run, so a
   clean report is not a proof.
4. Then on your GPU: `python3 labs/pmpp.py run 15.7`. Note which defects it showed you
   and which it hid.

The checks are lab 15.6's: the same shapes at warp widths 32 and 64, Y filled with NaN,
X and W allocated with exactly their floats, a guard band after Y that no kernel may
write, and a CPU reference with a tolerance. Read what each failure says: which rows are
wrong, which shapes fail and which pass, whether the guard band changed. On a GPU each
case is launched ten times, Y back to NaN each time, and the message counts how many
launches were wrong.

Nothing to benchmark here.
