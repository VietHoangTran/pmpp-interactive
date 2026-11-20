# Lab 5.4: race hunt

Book: sections 5.4 and 5.5.

`work/kernel.cu` and `work/kernel.metal` contain a tiled, boundary-checked matrix
multiplication with **three independent defects**. It compiles. Find and fix all three.

Method:

1. `python3 labs/pmpp.py run 5.4 --backend emu` first. The emulator's thread orders are
   deterministic, so a synchronisation defect fails every time, and AddressSanitizer
   stops at the first out-of-bounds access and names the line.
2. Fix one thing, run again. The failures change as you go.
3. Then run on your GPU: `python3 labs/pmpp.py run 5.4`. Note which defects the GPU
   showed you and which it hid.

Nothing to benchmark here.
