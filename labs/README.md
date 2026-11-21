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
python3 labs/pmpp.py run 5.2 --bench --reps 5   # benchmark five times, keeping every repetition
python3 labs/pmpp.py run 5.2 --solution     # the reference solution instead of yours
python3 labs/pmpp.py run 5.2 --full         # sanitizer reports in full (they are condensed by default)
python3 labs/pmpp.py reset 5.2 --yes        # throw your work away and start again
python3 labs/pmpp.py doctor                 # backends found, device properties
```

### What a run records and prints

- Each run's results go to `reports/runs/<backend>.json`, which the app reads. A plain
  `run` (no `--bench`) keeps the last benchmark of a kernel that still passes, marked with
  when it was taken.
- `--reps N` runs the benchmark N times, one after another, and keeps each repetition with
  the time it started (`bench_reps` in the JSON, every repetition in a `verify` report).
  The app shows the first. Use it on a machine whose timings move from run to run; the
  runner computes no average and no best-of, it records what was measured.
- `--no-test` benchmarks without testing. The last test result is kept only when the build
  read exactly the same files: the variant's sources, `main.cpp` and the lab's headers and
  `provided/` files, the shared and provided sources the lab compiles (and the `.metal` files
  beside them, which Metal compiles at run time), the earlier labs' files a milestone lab
  names in its compile definitions, every header the compiler's dependency files list, and
  the compile definitions themselves. Anything changed, and the test result is dropped.
  (With the Ninja generator the headers part is missing; the default Makefile generator has it.)
- A sanitizer report is condensed: the first report's headline (what went wrong and on
  which address), its stacks cut down to the frames in the lab's own sources, and its
  SUMMARY line; later reports only by their SUMMARY lines. A note says how many frames were
  left out. `--full` (on `run`, `selftest` and `verify`) shows everything. ThreadSanitizer
  data-race reports on `race` keep their own short form (both accesses and their source lines).
- A backend the machine has but the lab does not build for says why: "no Metal build for
  this lab" when the variant has no `kernel.metal` and `launch.mm`, "not built for the
  race-detection emulator" without `RACE`, or a library the lab links that is missing here.

### Build settings

- `PMPP_BUILD_DIR=<dir>` builds somewhere other than `labs/build`.
- `PMPP_ONLY=ch06` configures one chapter's labs only.
- `PMPP_CMAKE_ARGS="-DPMPP_CH15_VENDOR=OFF -DCMAKE_BUILD_TYPE=Debug"` passes arguments to the
  configure step (split as a shell would). CMake keeps a `-D` value in the build directory's
  cache after you stop passing it: set it back explicitly, or use a fresh `PMPP_BUILD_DIR`.
  Switches the build knows: `PMPP_DISABLE_RACE`, `PMPP_DISABLE_METAL`, `PMPP_DISABLE_CUDA`,
  `PMPP_DISABLE_HIP`, `PMPP_EMU_SANITIZE=OFF`, and the per-lab options their READMEs name.
- `doctor` prints each backend's device properties. On CUDA and HIP these include
  `asyncEngineCount` and `concurrentKernels` (whether copies and kernels can overlap,
  chapters 22 and 23); on ROCm `asyncEngineCount` is a runtime constant, not a count of copy
  engines (`notes/today/ch23-facts.md`, B3).

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
| Threads, blocks, grids | indices, dimensions, per-thread locals; at most 1024 threads per block and CUDA's per-dimension limits (block 1024 x 1024 x 64, grid (2^31 - 1) x 65535 x 65535, Programming Guide Table 33): a launch over them is refused with `cudaErrorInvalidConfiguration` and a message | everything about time |
| `__shared__`, `extern __shared__` | one copy per block, filled with `0xCD` for every block; dynamic size from the launch (48 KiB, 96 KiB after `cudaFuncSetAttribute`) | the limit on statically declared shared memory, layout, bank conflicts |
| `__syncthreads` and `_count/_and/_or` | barrier over the live threads of a block; an impossible barrier stops the kernel | barriers in divergent code that happen to pair up are treated as one |
| Warps | `warpSize` is 32; warp *w* is threads 32*w*..32*w*+31 in linear order; partial last warp | lockstep execution, divergence, scheduling |
| `__shfl*_sync`, `__ballot_sync`, `__any_sync`, `__all_sync`, `__match_*_sync`, `__syncwarp` | a rendezvous of the non-exited lanes named in the mask at the same source line; `width`; documented out-of-range behaviour. Stops the kernel for a mask without the caller, a bad width, a named lane that is waiting elsewhere. A read from a lane that takes no part returns `0xCD` bytes and prints a note | half and bfloat16 operands; a named lane that later exits without calling is treated as exited |
| `__activemask`, `coalesced_threads()` | the lanes of the warp that reach the same call site before any moves on | real convergence: a GPU may report fewer lanes |
| `__popc`, `__clz`, `__ffs`, `__brev` and the `ll` forms | the bit operations | (signatures unverified, see below) |
| `__dp4a` (`int`, `unsigned`, `char4`, `uchar4`), `__byte_perm`, `__umulhi`, `__mulhi`, `__umul64hi`, `__float_as_uint` and the other bit casts | the overloads the CUDA Math API documents; `__dp4a` sign-extends bytes in its signed forms and wraps its 32-bit sum | that HIP has no `__dp4a` (x2-facts B1): code calling it passes here and in the front-end check, and fails to build for HIP |
| `expf`, `exp2f`, `logf`, `log2f`, `sqrtf`, `rsqrtf`, `powf`, `fmaf`, `tanhf`, `erff`, `fminf`, `fmaxf`, `fabsf`, `copysignf`; `__expf`, `__logf`, `__powf`, `__sinf`, `__cosf`, `__fdividef`, `__saturatef` | the values, from the host's `<cmath>` | NVIDIA's error bounds, the fast intrinsics' approximation error, flush-to-zero; the `_rn` intrinsics are not provided |
| `__half`, `__half2` (`#include <cuda_fp16.h>`) | binary16 bit for bit: round-to-nearest-even conversions and `__hadd/__hsub/__hmul/__hdiv` (and the `2` forms, the operators), `__hfma`/`__hfma2` rounded once, `__low2float`, `__floats2half2_rn` and the other conversions | NaN payloads, `_sat`/`_relu`/directed-rounding variants, half shuffles and atomics, `__nv_bfloat16`, speed. HIP's header is `<hip/hip_fp16.h>` |
| `atomicAdd/Sub/Exch/Min/Max/Inc/Dec/CAS/And/Or/Xor` | the overloads NVIDIA documents, returning the old value | half, bfloat16, `float2/float4` and 128-bit forms |
| `cuda::atomic_ref`, `cuda::atomic` (`<cuda/atomic>`) | load, store, exchange, compare_exchange, fetch_add/sub/and/or/xor/min/max on 1-, 2-, 4-, 8-byte types | thread scopes and memory orders have no effect; wait/notify |
| `__constant__`, `cudaMemcpyToSymbol/FromSymbol` | a global the host fills through the symbol calls, with a size check | read-only enforcement in kernels, the size of constant memory |
| `char1..4`, `uchar1..4`, `short1..4`, `ushort1..4`, `float2..4`, `int2..4`, `uint2..4`, `double2..4` (sizes and alignments of Table 49), `make_*`, `__forceinline__`, `__noinline__`, `__restrict__`, `#pragma unroll` | types and helpers; the qualifiers compile | inlining and unrolling mean nothing here (clang accepts `#pragma unroll`; gcc ignores it, with a warning under `-Wall`) |
| `<cooperative_groups.h>` | `this_thread_block()`, `this_grid()` with grid-wide `sync()` under `cudaLaunchCooperativeKernel` (all blocks alive at once; grid x block at most 1024 threads, else `cudaErrorCooperativeLaunchTooLarge`), `coalesced_threads()`, `tiled_partition<1..32>(block)` | clusters, larger tiles, `binary_partition`, `labeled_partition`, `reduce`, `scan`, `memcpy_async`, kernel templates in a cooperative launch |
| Streams (with priorities), events, `cudaMemcpyAsync`, `cudaMallocHost`, `cudaMallocManaged`, `cudaFuncSetAttribute`, `cudaFuncGetAttributes`, `cudaGetDevice`, occupancy | the calls exist with the documented prototypes; everything is synchronous | asynchrony and overlap; priorities (the range is 0 to 0, as for a device without them); `cudaEventElapsedTime` is host wall-clock time, `cudaOccupancyMaxActiveBlocksPerMultiprocessor` a synthetic number, and `cudaFuncGetAttributes` reports no registers: none of it says anything about a GPU |

Things the emulator needs from a `.cu` file (all are also valid CUDA):

- `#include <cuda/atomic>`, `#include <cooperative_groups.h>` and `#include <cuda_fp16.h>` where you
  use them, as with nvcc.
- `__shared__` declarations inside a function, written as plain declarations
  (`__shared__ float a[N][M];`, `__shared__ float t[(TILE + 2) * (TILE + 2)];`), not behind a macro.
  Any expression may stand in an array bound; initialisers may not.
- A kernel launched with `cudaLaunchCooperativeKernel` must be an ordinary (non-template)
  `__global__` function defined in a `.cu` file.
- Kernels are not overloaded: the name before `<<<` must identify one function.

Not checked against NVIDIA's documentation (written from memory of the toolkit headers):
the prototypes of `__popc`, `__popcll`, `__clz`, `__clzll`, `__ffs`, `__ffsll`, `__brev`,
`__brevll`, and the 4-byte alignment of `__half2`. Everything else in
`common/include/pmpp/check/cuda_stub.h` names its source (the facts added on 2026-11-21 are
`notes/today/infra-facts.md`, group E).

### Self-tests

`labs/selftest/` holds the emulator's own tests: small `.cu` files written in plain CUDA,
run on `emu` and `race` (and on `cuda`, where they check the emulator's model against the
real thing). `./verify.sh` runs them and lists them in its report;
`python3 labs/pmpp.py selftest [name] [--backend race]` runs them alone and writes nothing.
`race_hist_plain` and `race_negatives` are planted data races: on `race` they pass only if
ThreadSanitizer reports them.

## Adding a lab

One line in the lab's `CMakeLists.txt` (`pmpp_add_lab(...)`, see `cmake/Lab.cmake`) and a
`main.cpp` built on `common/include/pmpp/harness.hpp`. Provided code the lab compiles goes in
`SHARED_CU` and `SHARED_MM`; `SHARED_DIR ../shared-g11` names the directory it lives in
(names in `SHARED_CU`/`SHARED_MM` may then be bare file names), adds it to the include path
and passes it to the driver as `PMPP_SHARED_DIR` (and `PMPP_METAL_SHARED_DIR` on Metal,
which otherwise points at `../shared`). `LIBS` links a library on every backend and
`LIBS_CUDA`, `LIBS_HIP`, `LIBS_METAL`, `LIBS_EMU`, `LIBS_RACE` on one (`CUDA::cublas`,
`roc::hipblas`, `"-framework MetalPerformanceShaders"`); when a named CMake target does
not exist on the machine, that backend's build of the lab is skipped and `run` says so. Add `RACE` to that line to build the
lab for the race-detection emulator as well (do it for any lab about atomics or other
unsynchronised updates), and give `race` the same small problem sizes as `emu`:
`pmpp::backend()` returns `"race"`, and `PMPP_BACKEND_RACE` is defined in `main.cpp`. A lab is finished only when its
solution passes on Metal and the emulator, its untouched starter builds and fails, and
each planted defect has been shown to fail on its own.
