# Labs

Each lab lives in `chNN/labM_name/` and has three copies of the files you edit:

- `starter/`  the untouched starting point
- `work/`     **yours**: edit these (it starts as a copy of `starter/`)
- `solution/` a reference solution; look after you have tried

The files are `kernel.cu` (CUDA; the same file is built for HIP and for the CPU
emulator), `kernel.metal` (Metal Shading Language) and `launch.mm` (the Metal host
side). `main.cpp` is the test and benchmark driver; you do not need to edit it.

```sh
python3 labs/pmpp.py run 5.2                # test your work on every backend found
python3 labs/pmpp.py run 5.2 --backend emu  # one backend: emu, metal, cuda or hip
python3 labs/pmpp.py run 5.2 --bench        # also benchmark (real devices only)
python3 labs/pmpp.py run 5.2 --solution     # the reference solution instead of yours
python3 labs/pmpp.py reset 5.2 --yes        # throw your work away and start again
python3 labs/pmpp.py doctor                 # backends found, device properties
```

## The emulator

`emu` runs your unmodified `.cu` kernel on the CPU. It exists so that kernel logic can be
tested on a machine without a CUDA or HIP device, and so that synchronisation mistakes
fail every time instead of now and then.

- Every test runs under three deterministic thread orders (in order, reversed, shuffled).
  One thread runs until it reaches `__syncthreads()` or returns, then the next.
- It is built with AddressSanitizer: an out-of-bounds read or write in a kernel stops the
  run with a report that names the source line.
- It models index arithmetic, shared memory, barriers and bounds. It does not model time,
  caches, warps, coalescing or occupancy. Its "device properties" are synthetic.
- Passing on the emulator does not prove a kernel runs on a GPU.

## Adding a lab

One line in the lab's `CMakeLists.txt` (`pmpp_add_lab(...)`, see `cmake/Lab.cmake`) and a
`main.cpp` built on `common/include/pmpp/harness.hpp`. A lab is finished only when its
solution passes on Metal and the emulator, its untouched starter builds and fails, and
each planted defect has been shown to fail on its own.
