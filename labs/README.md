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
python3 labs/pmpp.py run 5.2 --backend emu  # one backend: emu, race, metal, cuda or hip
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
  One thread runs until it reaches a synchronisation point (`__syncthreads()`, a warp
  `*_sync` call, a cooperative-groups `sync()`) or returns, then the next.
- It is built with AddressSanitizer: an out-of-bounds read or write in a kernel stops the
  run with a report that names the source line.
- It models index arithmetic, shared memory, barriers, warp-level data movement, atomics
  and bounds. It does not model time, caches, coalescing, bank conflicts or occupancy.
  Its "device properties" are synthetic.
- Passing on the emulator does not prove a kernel runs on a GPU.

### The race-detection flavour

Because `emu` never interrupts a thread between two synchronisation points, it cannot see
a missing atomic: `bins[b]++` from a thousand threads is always "correct" there. The
second flavour, backend `race`, exists for that:

```sh
python3 labs/pmpp.py run <lab> --backend race   # only labs that opted in (see "Adding a lab")
```

- The threads of a block, and several blocks, run truly concurrently as OS threads, and
  the build uses ThreadSanitizer. Two emulated threads touching the same memory without
  synchronisation are reported with both source lines, even if that run happened to
  produce the right numbers. A report fails the test case during which it appeared.
- `__syncthreads`, `__syncwarp`, grid `sync()` are real barriers and count as
  synchronisation. Atomics are real atomics. Warp votes and shuffles are real rendezvous
  but, as on a GPU, they do **not** order memory accesses.
- It is not repeatable (the operating system schedules the threads), it has no bounds
  checking (ThreadSanitizer and AddressSanitizer cannot be combined), and a clean run
  does not prove the absence of a race: only schedules that occurred are examined, and
  ThreadSanitizer reports each pair of source lines once per run. Two blocks that the
  emulator happened to run one after the other on the same OS threads are only checked
  against each other for different thread indices.
- It is enabled when the compiler's ThreadSanitizer works on the machine
  (`python3 labs/pmpp.py doctor` says why not otherwise). It is slow: keep its problem
  sizes as small as the `emu` ones.

### What is and is not modelled

The header comment of `common/include/pmpp/emu/cuda_emu.h` is the reference; this is the
same list, shorter. "Stops the kernel" means: the launch ends, `cudaGetLastError()` returns
`cudaErrorLaunchFailure`, the test fails with a message naming the source lines. Nothing hangs.

| Feature | Modelled | Not modelled |
|---|---|---|
| Threads, blocks, grids | indices, dimensions, per-thread locals, at most 1024 threads per block | everything about time |
| `__shared__`, `extern __shared__` | one copy per block, filled with `0xCD` for every block; dynamic size from the launch (48 KiB, 96 KiB after `cudaFuncSetAttribute`) | the limit on statically declared shared memory, layout, bank conflicts |
| `__syncthreads` and `_count/_and/_or` | barrier over the live threads of a block; an impossible barrier stops the kernel | barriers in divergent code that happen to pair up are treated as one |
| Warps | `warpSize` is 32; warp *w* is threads 32*w*..32*w*+31 in linear order; partial last warp | lockstep execution, divergence, scheduling |
| `__shfl*_sync`, `__ballot_sync`, `__any_sync`, `__all_sync`, `__match_*_sync`, `__syncwarp` | a rendezvous of the non-exited lanes named in the mask at the same source line; `width`; documented out-of-range behaviour. Stops the kernel for a mask without the caller, a bad width, a named lane that is waiting elsewhere. A read from a lane that takes no part returns `0xCD` bytes and prints a note | half and bfloat16 operands; a named lane that later exits without calling is treated as exited |
| `__activemask`, `coalesced_threads()` | the lanes of the warp that reach the same call site before any moves on | real convergence: a GPU may report fewer lanes |
| `__popc`, `__clz`, `__ffs`, `__brev` and the `ll` forms | the bit operations | (signatures unverified, see below) |
| `atomicAdd/Sub/Exch/Min/Max/Inc/Dec/CAS/And/Or/Xor` | the overloads NVIDIA documents, returning the old value | half, bfloat16, `float2/float4` and 128-bit forms |
| `cuda::atomic_ref`, `cuda::atomic` (`<cuda/atomic>`) | load, store, exchange, compare_exchange, fetch_add/sub/and/or/xor/min/max on 1-, 2-, 4-, 8-byte types | thread scopes and memory orders have no effect; wait/notify |
| `__constant__`, `cudaMemcpyToSymbol/FromSymbol` | a global the host fills through the symbol calls, with a size check | read-only enforcement in kernels, the size of constant memory |
| `float2..4`, `int2..4`, `uint2..4`, `double2..4`, `make_*`, `__forceinline__`, `__noinline__`, `__restrict__`, `#pragma unroll` | types and helpers; the qualifiers compile | inlining and unrolling mean nothing here (clang accepts `#pragma unroll`; gcc ignores it, with a warning under `-Wall`) |
| `<cooperative_groups.h>` | `this_thread_block()`, `this_grid()` with grid-wide `sync()` under `cudaLaunchCooperativeKernel` (all blocks alive at once; grid x block at most 1024 threads, else `cudaErrorCooperativeLaunchTooLarge`), `coalesced_threads()`, `tiled_partition<1..32>(block)` | clusters, larger tiles, `binary_partition`, `labeled_partition`, `reduce`, `scan`, `memcpy_async`, kernel templates in a cooperative launch |
| Streams, events, `cudaMemcpyAsync`, `cudaMallocHost`, `cudaMallocManaged`, `cudaFuncSetAttribute`, occupancy | the calls exist with the documented prototypes; everything is synchronous | asynchrony and overlap; `cudaEventElapsedTime` is host wall-clock time and `cudaOccupancyMaxActiveBlocksPerMultiprocessor` is a synthetic number: neither says anything about a GPU |

Things the emulator needs from a `.cu` file (all are also valid CUDA):

- `#include <cuda/atomic>` and `#include <cooperative_groups.h>` where you use them, as with nvcc.
- `__shared__` declarations inside a function, written as plain declarations
  (`__shared__ float a[N][M];`), not behind a macro.
- A kernel launched with `cudaLaunchCooperativeKernel` must be an ordinary (non-template)
  `__global__` function defined in a `.cu` file.
- Kernels are not overloaded: the name before `<<<` must identify one function.

Not checked against NVIDIA's documentation (written from memory of the toolkit headers):
the prototypes of `__popc`, `__popcll`, `__clz`, `__clzll`, `__ffs`, `__ffsll`, `__brev`,
`__brevll`, and the alignment of the vector types. Everything else in
`common/include/pmpp/check/cuda_stub.h` names its source.

### Self-tests

`labs/selftest/` holds the emulator's own tests: small `.cu` files written in plain CUDA,
run on `emu` and `race` (and on `cuda`, where they check the emulator's model against the
real thing). `./verify.sh` runs them and lists them in its report;
`python3 labs/pmpp.py selftest [name] [--backend race]` runs them alone and writes nothing.
`race_hist_plain` and `race_negatives` are planted data races: on `race` they pass only if
ThreadSanitizer reports them.

## Adding a lab

One line in the lab's `CMakeLists.txt` (`pmpp_add_lab(...)`, see `cmake/Lab.cmake`) and a
`main.cpp` built on `common/include/pmpp/harness.hpp`. Add `RACE` to that line to build the
lab for the race-detection emulator as well (do it for any lab about atomics or other
unsynchronised updates), and give `race` the same small problem sizes as `emu`:
`pmpp::backend()` returns `"race"`, and `PMPP_BACKEND_RACE` is defined in `main.cpp`. A lab is finished only when its
solution passes on Metal and the emulator, its untouched starter builds and fails, and
each planted defect has been shown to fail on its own.
