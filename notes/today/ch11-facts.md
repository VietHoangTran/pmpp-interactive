# Chapter 11 fact check: scan (CUDA, HIP, Metal, real backends)

Checked on 2026-11-20 against the live sources listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY, and judges the claim in the item title (usually the book's statement from `notes/ch11.md`, or the assumption a course author would bring). **Quote** is copied from the source with line breaks removed.
- "infra A3", "ch10 B1", "ch05 A8" point at items in `notes/today/infra-facts.md`, `ch10-facts.md`, `ch05-facts.md` and so on. Anything settled there is cited, not re-checked.
- Page numbers ("p. 264") are book pages as recorded in `notes/ch11.md`.
- Versions the sources reported about themselves on the check date: CUDA Programming Guide "v13.5" (pages "Last updated on Oct 31, 2026"); Nsight Compute docs "v2026.4.0"; CCCL (CUB, Thrust, libcu++) source on `main`; HIP docs for ROCm 10.2.0 read as `.rst` sources at tag `therock-10.2` of `ROCm/rocm-systems`; rocPRIM and hipCUB at tag `therock-10.2` of `ROCm/rocm-libraries`; libhipcxx at tag `therock-10.2` of `ROCm/libhipcxx`; Metal Shading Language Specification "Version 4.1", dated 2026-06-04 ("MSL spec"); local macOS SDK 27.0 and the Metal compiler it ships ("Apple metal version 32023.921"); MLX docs "0.32.4"; PyTorch docs "2.15"; llama.cpp `master`, MLX `main`, mlx-lm `main`, PyTorch `main`, vLLM `main`, transformers `main` on the check date (last-commit dates given per file).
- Not checked: anything by running code on hardware. Every statement is from documentation, headers or project source, except where an item says "compiled here": those few MSL snippets were compiled (not run) with the local Metal compiler to settle what the spec leaves unclear. They are not lab code and are not in the repository.

Totals: 12 CONFIRMED, 2 WRONG, 7 PARTLY, 1 COULD NOT VERIFY (22 items: A1 to A8, B1 to B4, C1 to C6, D1 to D4). The WRONG, PARTLY and COULD NOT VERIFY items, and every failed sub-claim, are collected under "Summary of corrections".

---

## Already settled elsewhere (cite, do not re-check)

| Chapter 11 needs | Settled in |
|---|---|
| H100 figures as the book's example: 3.35 TB/s (SXM), FP32 printed as 67, not 66.9 | ch05 A8, ch01 A1 |
| 1024 threads per block; warp size 32 on every listed compute capability | ch05 A5, ch03 A3 |
| AMD wavefront 64 on gfx9, 32 on gfx10 and later; `warpSize` not a compile-time constant since ROCm 7.0; template parameter or `__GFX9__` macro | ch05 B5, ch10 B2 |
| Metal SIMD-group width undocumented; read `[[threads_per_simdgroup]]`; grouping implementation-defined | ch04 C2, infra C5 |
| `__syncthreads()` semantics, uniform-condition rule, exited threads; HIP and Metal counterparts | ch04 A6, B3; ch05 C2; ch04 C3; ch10 A1 |
| Shuffle signatures, out-of-range rules, mask constraints (CUDA, HIP, Metal) | infra A3, B3, C2 |
| HIP `_sync` masks must be 64-bit (`static_assert`), so `0xffffffff` does not compile | infra B3, ch10 B1 |
| `cuda::atomic_ref`, scopes, both memory-order spellings, constructor takes a reference, `seq_cst` default | infra A2, ch09 A1, ch10 A6 |
| libhipcxx as HIP's `<cuda/atomic>` (ROCm 10.1 and later; link `libhipcxx::libhipcxx`) | infra B2 |
| MSL atomic types and functions; acquire/release added in Metal 4.1, spec says "relaxed only" in places | infra C1 (refined below in C4) |
| Block scope on a flag read by another block is a data race (undefined behaviour) | ch09 A2 |
| Blocks run in any order; the scheduler gives no ordering guarantee (CUDA); same on AMD | ch04 A5, B2 |
| Cooperative groups, grid-wide `sync()`, residency limit; HIP headers differ; Metal has none | infra A4, B4; ch04 A7, B11, C3 |
| Divergence: inactive lanes still occupy the warp's issue slots (the p. 266 and pp. 284 to 285 argument) | ch04 A9, B7, C4 |
| `#pragma unroll` (compiler unrolls small constant-trip loops by default), register promotion of local arrays | infra A7, B7; ch06 A13, A14; Metal ch06 C4 |
| Shared-memory banks and padding (exercise 4) | infra A10, B10, C13; ch06 A12, B4, C2 |
| Vector loads (exercise 4) | infra A6, B6, C7; ch06 A11, B5, C3 |
| Floating-point addition is not associative; NVIDIA's own worked example | ch10 A9 |
| Metal default math mode documented three ways; not IEEE-strict by default | ch05 C7, ch10 C3 |
| Thrust and CUB URLs in the book's references moved under `nvidia.github.io/cccl/unstable/` | ch10 A12 |
| CUB functor spellings (`cuda::std::plus`, `cuda::maximum<>{}`); hipCUB in transition | infra A11, B11 |
| Vocabulary sizes and their bytes (a Qwen3 row of `float` is 593.5 KiB, more than any per-block on-chip limit) | ch09 D7 |
| llama.cpp sampling runs on the host by default; an experimental backend path exists | ch01 D2, ch10 D2 |

---

## Group A: CUDA, and the book's dated statements

### A1. The H100 numbers and the scan ceilings (pp. 272 to 276)

**Verdict: PARTLY.** The bandwidth figure and all the arithmetic hold; the FP32 figure is the one ch05 A8 already flagged.

- Book: 66.9 TFLOPS, 3.35 TB/s, threshold 20.0 FLOP/B; ceilings 419, 209 and 279 ×10⁹ elements/s at 8, 16 and 12 bytes per element.
- NVIDIA prints "FP32 67 teraFLOPS" and "GPU Memory Bandwidth 3.35TB/s" for H100 SXM (ch05 A8). 66.9 appears in no NVIDIA source found.
- Arithmetic, ours: 66.9 / 3.35 = 19.97 and 67 / 3.35 = 20.0, so the threshold is 20 either way. 3.35×10¹² / 8 = 418.75×10⁹; / 16 = 209.4×10⁹; / 12 = 279.2×10⁹. The book's rounding is right.
- The bytes-per-element formulas ((16 + 8/S)·N and (12 + 12/S)·N) are the book's own analysis and need no vendor fact. Corroboration that the ideal is 2 accesses per element: CUB's DeviceScan docs say its single-pass algorithm "requires only ``~2*n*`` data movement (``n`` inputs are read, ``n`` outputs are written), and typically proceeds at "memcpy" speeds" (A8).

For lessons: the H100 figures are the book's example only; reader numbers come from `labs/pmpp.py doctor` and the lab benchmarks.

- https://www.nvidia.com/en-us/data-center/h100/ (via ch05 A8)

### A2. "More than 80% of bandwidth with no redundant accesses is well optimized"; "even a copy kernel does not reach 100%" (pp. 272 to 273)

**Verdict: PARTLY.** NVIDIA's profiler uses the same 80% line, as "near the limit". No NVIDIA statement was found that a copy kernel cannot reach peak.

- Nsight Compute "Compute Triage Guide" (v2026.4.0), Decision 2, GPU throughput: "Below ~60%: low (latency-risk regime)." "Above ~80%: high (near-limit regime)." Decision 7: "The limiter resource is within ~80% of its peak."
- COULD NOT VERIFY: any vendor sentence that a plain copy does not reach 100% of the published bandwidth. Keep it as the book's statement; the lab can show it with the reader's own copy benchmark.
- For the bandwidth lab, ours: report achieved bytes per second against the reader's measured copy bandwidth as well as the published peak, so the 80% yardstick is read against something the reader's machine actually achieves.

- https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html

### A3. `__shfl_up_sync` in Fig. 11.8, and what lanes below `stride` receive (p. 264; "Unclear in the PDF")

**Verdict: CONFIRMED.** Signature and behaviour as in infra A3: `T __shfl_up_sync(unsigned mask, T value, unsigned delta, int width=warpSize);` and "The source lane index will not wrap around the value of width, so the lower delta lanes will remain unchanged."

- This resolves the notes' unclear item: a lane below `stride` gets its own value back. The book's `if(laneIdx() >= stride)` test is therefore required, not cosmetic. Without it those lanes would add their own value to itself (our consequence of the quoted rule).
- NVIDIA prints the book's Fig. 11.8 shape as its own example. Programming guide section 5.4.6.5, "Example 2: Inclusive plus-scan across sub-partitions of 8 threads", intrinsics version:

```cpp
for (int delta = 1; delta <= 4; delta *= 2) {
    int tmp         = __shfl_up_sync(0xFFFFFFFF, value, delta, /*width=*/8); // read from laneId - delta
    int source_lane = laneId % 8 - delta;
    if (source_lane >= 0) // lanes with 'source_lane < 0' have their value unchanged
        value += tmp;
}
```

  It is headed by the hint "It is suggested to use the cub::WarpScan function for efficient and generalized warp scan functions." and its first version uses `cub::WarpScan<int, 8>` with `.InclusiveSum(value, partial_sum)`.
- The mask literal and the width default are the HIP problems of B1.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (section 5.4.6.5)

### A4. Library and group-level scans on CUDA: CUB `WarpScan` and `BlockScan`, cooperative-groups `inclusive_scan` / `exclusive_scan`

**Verdict: PARTLY.** All exist and are current. The cooperative-groups pages repeat the inconsistency infra A4 found for `cg::reduce`: the reference restricts the group type, the guide's own example ignores the restriction.

CUB (CCCL `main`, `cub/block/block_scan.cuh`):

- `template <typename T, int BlockDimX, BlockScanAlgorithm Algorithm = BLOCK_SCAN_RAKING, int BlockDimY = 1, int BlockDimZ = 1> class BlockScan`. Three algorithms:
  - `BLOCK_SCAN_WARP_SCANS`: "Upsweep sequential reduction in registers (if threads contribute more than one input each)", "Compute a shallow, but inefficient warp-synchronous Kogge-Stone style scan within each warp", "A propagation phase where the warp scan outputs in each warp are updated with the aggregate from each preceding warp", "Downsweep sequential scan in registers". That is the book's Figs. 11.9 and 11.12 to 11.13 as one library option. "it can often provide lower turnaround latencies when the GPU is under-occupied."
  - `BLOCK_SCAN_RAKING` (the default): "An efficient "raking reduce-then-scan" prefix scan algorithm", with "A warp-synchronous Kogge-Stone style exclusive scan within the raking warp" in the middle. "it can often provide higher overall throughput across the GPU when suitably occupied."
  - `BLOCK_SCAN_RAKING_MEMOIZE`: "with fewer shared memory reads at the expense of higher register pressure."
  - Note for §11.9 (ours): CUB's default block scan is a reduce-then-scan inside the block, where the book argues scan-scan-add wins inside a block (p. 276). Both are documented options; the lab can measure.
- `cub::WarpScan` exists with `InclusiveSum` and `ExclusiveSum`; its shuffle specialisation issues `shfl.sync.up.b32` (`warp/specializations/warp_scan_shfl.cuh`), i.e. the Fig. 11.8 shuffle.

Cooperative groups (guide sections 4.4.6.2 and 5.7.3.5, header `cooperative_groups/scan.h`):

- `auto inclusive_scan(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val));` and `exclusive_scan` likewise, plus overloads without `op` that "use cg::plus<TyVal>()".
- "Result for each thread is a reduction of data from threads with lower thread_rank than that thread in case of exclusive_scan. inclusive_scan result also includes the calling thread data in the reduction."
- Reference: "group: Valid group types are coalesced_group and thread_block_tile." Guide 4.4.6.2 example: `cg::thread_block my_group = cg::this_thread_block(); ... int exclusive_sum = cg::exclusive_scan(my_group, val, cg::plus<int>());`. Teach the tile form, which both pages allow.
- `inclusive_scan_update` / `exclusive_scan_update` take a `cuda::atomic` or `cuda::atomic_ref` and add the group total to it, returning each thread's offset: "Atomic update is performed with relaxed memory ordering." This is the book's "memory allocation" use of exclusive scan (p. 253) as a library call.
- No warp scan instruction for `float` is documented for CUDA (the integer `__reduce_*_sync` family reduces, it does not scan; ch10 A4).

- https://github.com/NVIDIA/cccl/blob/main/cub/cub/block/block_scan.cuh (lines 45 to 106, 226 to 257)
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/warp/specializations/warp_scan_shfl.cuh
- https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1BlockScan.html
- https://nvidia.github.io/cccl/unstable/cub/api/classcub_1_1WarpScan.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html (section 4.4.6.2)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html (section 5.7.3.5)

### A5. The flag protocol of Fig. 11.17: device-scope atomics, release on the writer, acquire on the reader, `fetch_add(0, ...)` as a read (pp. 278 to 279)

**Verdict: CONFIRMED.** NVIDIA's memory-model appendix prints the same protocol as its "message passing" example, including the plain (non-atomic) payload that the release and acquire make visible.

Guide section 5.8.5 "Example: Message Passing", as printed:

```cpp
// Initially
int x = 0, f = 0;
// Thread 0 Block 0
x = 42;
cuda::atomic_ref<int, cuda::thread_scope_device> flag(f);
flag.store(1, memory_order_release);
// Thread 0 Block 1
cuda::atomic_ref<int, cuda::thread_scope_device> flag(f);
while(flag.load(memory_order_acquire) != 1);
assert(x == 42);
```

- The variant with `cuda::thread_scope_block` on the store is marked "// UB: data race": "Since the store is performed by Thread 0 of Block 0, it only includes all other threads of Block 0." So the book's `thread_scope_device` is required (ch09 A2 for the general rule).
- `x` plays the role of the book's `partialSums[bid]`: a plain store before the release, a plain load after the acquire. That is exactly why the book can leave `partialSums` non-atomic.
- The book reads the flag with `fetch_add(0, cuda::memory_order_acquire)`. NVIDIA's example uses `load(memory_order_acquire)`, which `cuda::atomic_ref` has (it has the `std::atomic_ref` interface, infra A2). Both are atomic reads; the `load` is the read-only form, and the lab should use it (ours: a read-modify-write is a write as well, which nothing here needs). On the writer side `store(1, release)` and the book's `fetch_add(1, release)` are equivalent for a flag that starts at 0.
- Flags and `partialSums` must start at zero for every launch (the notes' "Unclear" item). That is a host-side `cudaMemset` (or `hipMemset`) before each launch, not something the kernel can assume.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html (section 5.8.5)

### A6. Is the single-pass scan with a dynamic block index safe on CUDA? (pp. 277 to 282, 285)

**Verdict: CONFIRMED.** NVIDIA documents a forward-progress model that makes the book's design sound, and it also shows why waiting on `blockIdx.x - 1` without the counter is not.

Guide section 5.9 "CUDA C++ Execution model":

- "CUDA C++ aims to provide parallel forward progress [intro.progress.9] for all device threads of execution".
- The definition it quotes from the C++ standard: "For a thread of execution providing parallel forward progress guarantees, the implementation is not required to ensure that the thread will eventually make progress if it has not yet executed any execution step; once this thread has executed a step, it provides concurrent forward progress guarantees." And concurrent forward progress: "the implementation ensures that the thread will eventually make progress for as long as it has not terminated. [Note 5: This applies regardless of whether or not other threads of execution (if any) have been or are making progress."
- Section 5.9.2 "Device threads": "Once a device thread makes progress: If it is part of a Cooperative Grid, all device threads in its grid shall eventually make progress. Otherwise, all device threads in its thread-block cluster shall eventually make progress." Notes: "Threads in other thread-block clusters are not guaranteed to eventually make progress." "This implies that all device threads within its thread block shall eventually make progress."
- What a spin loop must do to count: "The implementation may assume that any device thread will eventually do one of the following: terminate, make a call to a library I/O function, perform an access through a volatile glvalue except if the designated object has automatic storage duration, or perform a synchronization operation or an atomic read operation except if the designated object has automatic storage duration." Example "Execution.Model.Device.0" (thread 0 spins on `atom.load(cuda::memory_order_relaxed) == 0` while thread 1 stores 1): "grid eventually terminates per device.threads.4 because the atomic object does not have automatic storage duration."

Our derivation from these rules (not a sentence in NVIDIA's docs):

- With the dynamic index, block b waits only for the block that claimed index b − 1. That block claimed it with an atomic, so at least one of its threads has executed a step; from then on its whole block is guaranteed to make progress, so it will publish its flag. The wait terminates. This is the book's argument (p. 280), now backed by a documented rule.
- Waiting on the block with `blockIdx.x - 1` instead has no such backing: that block may not have executed any step, and "not required to ensure that the thread will eventually make progress if it has not yet executed any execution step" applies. Combined with "no ordering guarantees are made by the scheduler" (ch04 A5), a grid larger than what fits on the device can deadlock.
- The spin must be an atomic read of global memory (as the book's is). A spin on a plain, non-volatile load is not covered by the list above.
- Two documented alternatives also make waiting safe: a cooperative launch ("all device threads in its grid shall eventually make progress"; this is the launch that method (a) needs anyway, infra A4), and waiting only within one thread-block cluster (compute capability 9.0 and later, ch05 A6).

NVIDIA's own library does not use the counter. CUB `agent_scan.cuh` line 452: "// Blocks are launched in increasing order, so just assign one tile per block", then `const int tile_idx = static_cast<int>(start_tile + blockIdx.x);`. That rests on launch order, which the guide does not promise (our reading). AMD's tutorial says the same thing about AMD hardware (B3). For a lesson: the book's counter is the documented-safe form; CUB relies on observed behaviour.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-execution-model.html (sections 5.9, 5.9.2)
- https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/execution_model.html (the same text in libcu++)
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/agent/agent_scan.cuh (lines 450 to 456)

### A7. Back-off in the spin loop: `__nanosleep` (not in the book)

**Verdict: CONFIRMED.** The book's wait loop has an empty body. CUDA documents a sleep for exactly this use, and CUB's look-back calls it.

- Guide section 5.4.8.5: `__device__ void __nanosleep(unsigned nanoseconds);` "The function __nanosleep(ns) suspends the thread for a sleep duration of approximately ns nanoseconds. The maximum sleep duration is approximately one millisecond." The guide's example is "a mutex with exponential back-off": `while (atomicCAS(mutex, 0, 1) == 1) { __nanosleep(ns); if (ns < 256) { ns *= 2; } }`.
- CUB `single_pass_scan_operators.cuh` wraps it as `delay<Delay, GridThreshold>()` and `always_delay(ns)`, guarded by `NV_IF_TARGET(NV_PROVIDES_SM_70, (__nanosleep(Delay);))`, and uses it in `WaitForValid`.
- HIP and Metal: see B4 and C4.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (section 5.4.8.5)
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/agent/single_pass_scan_operators.cuh (lines 123 to 203)

### A8. The libraries the book points to: `std::inclusive_scan`, `std::exclusive_scan`, Thrust, CUB, and the decoupled look-back report (pp. 254, 286 to 287); what they promise about floating-point results

**Verdict: CONFIRMED.** All current. Every one of them documents that a floating-point scan may differ from run to run unless a deterministic mode is requested.

- cppreference, `std::inclusive_scan` and `std::exclusive_scan`: "Defined in header <numeric>", "(since C++17)". `exclusive_scan(first, last, d_first, T init)` takes the initial value explicitly; its output at position i folds `init` with the elements of `[first, iter)`, "excluding *iter". Both pages: "The result is non-deterministic if the add_op is not associative (such as floating-point addition)." Example on the page: input `3 1 4 1 5 9 2 6`, exclusive `0 3 4 8 9 14 23 25`, inclusive `3 4 8 9 14 23 25 31`.
- Thrust (`thrust/scan.h`, CCCL `main`): `thrust::inclusive_scan`, `thrust::exclusive_scan`; "Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run."
- CUB `DeviceScan` (`cub/device/device_scan.cuh`):
  - "As of CUB 1.0.1 (2013), CUB's device-wide scan APIs have implemented our "decoupled look-back" algorithm for performing global prefix scan with only a single pass through the input data". The reference is "Duane Merrill and Michael Garland. Single-pass Parallel Prefix Scan with Decoupled Look-back, NVIDIA Technical Report NVR-2016-002, 2016." So the book's exercise 5 is what the library ships.
  - "``cub::DeviceScan`` supports the determinism guarantees as follows; the default is ``not_guaranteed``." "``run_to_run`` is supported for integral types with a known CUDA binary operator, and for floating-point types with ``cuda::std::plus``. The floating-point ``plus`` case engages a stable, fixed reduction order so results are reproducible across runs on the same GPU." `gpu_to_gpu` is for integral types only.
  - `ExclusiveSum`: "Results are not deterministic for pseudo-associative operators (e.g., addition of floating-point types). Results for pseudo-associative operators may vary from run to run." "Added in version 2.2.0: First appears in CUDA Toolkit 12.3." (for the overload carrying that note).
- The report's page is live: `https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back` returns 200; the short form CUB links (`.../publication/single-pass-parallel-prefix-scan-decoupled-look-back`) redirects to it.
- Consequence for the labs, ours: the CPU reference should be the explicit left-to-right loop of Fig. 11.1, not `std::inclusive_scan` (whose order is unspecified), so the reference itself is reproducible. A GPU result is then compared with a tolerance, never bit for bit.

- https://en.cppreference.com/w/cpp/algorithm/inclusive_scan
- https://en.cppreference.com/w/cpp/algorithm/exclusive_scan
- https://github.com/NVIDIA/cccl/blob/main/thrust/thrust/scan.h (lines 36 to 160)
- https://github.com/NVIDIA/cccl/blob/main/cub/cub/device/device_scan.cuh (lines 95 to 170, 400 to 415)
- https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceScan.html
- https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

---

## Group B: HIP / ROCm

### B1. Does the book's `warpScan` / `blockScan` (Figs. 11.8 to 11.10) compile and work under HIP unchanged?

**Verdict: WRONG.** Same three problems as the reduction in ch10 B1, plus one undocumented point.

1. The mask. `__shfl_up_sync(0xffffffff, val, stride)` fails HIP's `static_assert(... sizeof(MaskT) == 8, "The mask must be a 64-bit integer. ...")` (infra B3). HIP's signature is `T __shfl_up_sync(unsigned long long mask, T var, unsigned int delta, int width=warpSize);`. Use a typed mask (llama.cpp's `ggml_cuda_thread_mask_t`, ch05 B9).
2. The out-of-range rule is the same as CUDA's, so the book's lane test still works: HIP docs for `__shfl_up`: "If the resulting source lane is out of range, the thread returns its own var." (infra B3).
3. The width. With the book's `WARP_SIZE` 32 on a 64-wide wavefront and the default `width=warpSize`, strides 1 to 16 never reach across lane 31/32, so each 32-lane half scans on its own (our arithmetic from the rule above). That is correct only if `laneIdx()`, `warpIdx()` and `NUM_WARPS` all count warps of 32 as well, and it uses half of each wavefront's shuffle reach. Reading the width at compile time per architecture is AMD's recommended approach (ch10 B2), and it is what llama.cpp does: `ggml_cuda_get_physical_warp_size()` returns 64 `#if defined(GGML_USE_HIP) && (defined(__GFX9__) || defined(__GFX8__))`, else 32 (D1).
4. Fig. 11.9's limit "one warp scans the warp sums" needs `NUM_WARPS <= WARP_SIZE`. At 1024 threads that is 32 warps of 32 or 16 wavefronts of 64, so it holds on both vendors (our arithmetic).

Not documented by AMD: whether a `__shared__` array may be declared inside a `__device__` function, as Fig. 11.9 does. CUDA documents it (guide 5.4.1.2.1, the example comment reads `__global__ void kernel() { // or a __device__ function` followed by `__shared__ int smem_var1[4];`). The HIP `__shared__` section says only "Shared memory can be allocated statically within the kernel". COULD NOT VERIFY for HIP. The form that works on all three backends (C2) is to declare the array in the kernel and pass a pointer to the helper.

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (sections "__shared__", "Warp cross-lane functions")
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/docs/how-to/hip_cpp_language_extensions.rst (lines 249 to 258, 867 to 900)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (section 5.4.1.2.1)

### B2. Library and group-level scans on HIP: hipCUB, rocPRIM, cooperative-groups scans

**Verdict: CONFIRMED.** All exist with CUB's names. Two documented differences matter for this chapter: a deterministic device scan is a separate function in rocPRIM, and HIP's cooperative-groups scan is hardware-accelerated for `float` on AMD.

- hipCUB at `therock-10.2` (`projects/hipcub/.../backend/rocprim/`): `class BlockScan` with `InclusiveSum(T input, T& output)` and overloads (`block/block_scan.hpp` line 72); `class WarpScan : private ::rocprim::warp_scan<T, LOGICAL_WARP_THREADS>` with `InclusiveSum`, `ExclusiveSum` (`warp/warp_scan.hpp` line 46); `DeviceScan::InclusiveSum(void* d_temp_storage, ...)` (`device/device_scan.hpp` line 50). Header and namespace differ from CUB, so one source cannot call both by text (infra B11).
- rocPRIM `device_scan.hpp` (public header): `inclusive_scan`: "When used with non-associative functions (e.g. floating point arithmetic operations): the results may be non-deterministic and/or vary in precision, and bit-wise reproducibility is not guaranteed, that is, results from multiple runs using the same input values on the same device may not be bit-wise identical. If deterministic behavior is required, Use rocprim::deterministic_inclusive_scan instead." Same for `exclusive_scan` with `deterministic_exclusive_scan`. The detail header says why: "lookback scan is not deterministic for non-associative operations: This is because the number of lookback steps may vary depending on the algorithm that its used in, scanned operator, and the hardware of the device running on it." rocPRIM credits "Merrill, D. and Garland, M. Single-pass Parallel Prefix Scan with Decoupled Look-back."
- Cooperative groups on HIP (docs at the tag): `auto inclusive_scan(const TyGroup& group, TyVal&& val, Operation&& op)` and `exclusive_scan`, "Defined in cooperative_groups/hip_scan.h." "On AMD, when group is of the same size as the warp size and T a primitive type, DPP instructions are used, resulting in significantly faster execution than with other group sizes." For arithmetic scans, "AMD: there is hardware acceleration for int, unsigned int, unsigned long long, long long, half/float/double"; "Nvidia: there is hardware acceleration for int, unsigned int". For `exclusive_scan`: "The value returned for the first active lane is platform-dependent and may change in future releases": on AMD the identity of the operation (`plus`: always 0), on NVIDIA "returns T {} (i.e. 0)".
- A slip in the HIP page, our reading: it says scans expect participation "with each thread contributing the same per-thread value", which would make a scan pointless; NVIDIA's reference says "Different threads in the group can pass different values for this argument." Follow NVIDIA's wording.
- The include path differs from CUDA's `cooperative_groups/scan.h`, so a `.cu` file needs an `#if` around it (infra B4).

- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocprim/rocprim/include/rocprim/device/device_scan.hpp (lines 324 to 340, 641 to 656, 734 to 747, 868 to 883)
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocprim/rocprim/include/rocprim/device/detail/lookback_scan_state.hpp (lines 60 to 108)
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipcub/hipcub/include/hipcub/backend/rocprim/block/block_scan.hpp
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipcub/hipcub/include/hipcub/backend/rocprim/warp/warp_scan.hpp
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/docs/how-to/hip_runtime_api/cooperative_groups.rst (lines 620 to 690)
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/cooperative_groups.html

### B3. Does AMD document the forward progress the single-pass scan needs?

**Verdict: COULD NOT VERIFY.** AMD documents no forward-progress model for workgroups. AMD's own library depends on lookback working, with per-architecture workarounds. Treat the single-pass kernel as `unverified` on HIP.

What AMD's documentation says:

- HIP docs at `therock-10.2`, searched in all 95 `.rst` sources for "forward progress", "deadlock", "spin", "lookback": no forward-progress statement for workgroups. The only progress statement is about lanes: "Neither RDNA- nor CDNA-based AMD hardware provides guaranteed independent progress to lanes of the same warp. AMD GPUs execute warps in lockstep" (reduction tutorial).
- On order, the same tutorial (section "Global data share"): "The API doesn't guarantee the order in which blocks are scheduled even though all GPUs schedule them in the same monotonically increasing order of block ids. Relying on this implicitly, the last block of a grid is in the optimal position to observe the side effects of all other blocks (using spinlocks or other methods)". And the programming model page: "Execution order is non-deterministic, work-groups may execute concurrently or in arbitrary order" (ch04 B2).
- libhipcxx carries an "Execution model" page (`docs/libcudacxx/extended_api/execution_model.rst` at the tag), but it is NVIDIA's text unchanged: it begins "CUDA C++ aims to provide parallel forward progress" and its links point to NVIDIA's documentation. Its published URL (`rocm.docs.amd.com/projects/libhipcxx/en/latest/libcudacxx/extended_api/execution_model.html`) returned HTTP 404 today. We do not read it as an AMD statement about AMD hardware.

What AMD's library does (rocPRIM at `therock-10.2`):

- Block index: by default the lookback scan uses the hardware block index (`block_id_wrapper<T, false>::get` returns `::rocprim::detail::block_id<0>()`). The book's atomic counter exists as `ordered_block_id::get`, which does `storage.id = ::rocprim::detail::atomic_add(this->id, 1);` in thread 0 and then a barrier. `check_if_using_atomic_block_id` turns it on as a "hotfix" only for `gfx942`, `gfx950` and `gfx1250` (`needs_hotfix = (arch == target_arch::gfx942) || (arch == target_arch::gfx950) || (arch == target_arch::gfx1250);`), with an environment variable `ROCPRIM_USE_ATOMIC_BLOCK_ID` (0 never, 1 hotfix, 2 always).
- Sleep: "It is known that early revisions of MI100 (gfx908) hang in the wait loop of lookback_scan_state::get() without sleeping (s_sleep)." On those devices the wait loop calls `__builtin_amdgcn_s_sleep(1)`.
- Fences: "This version is specific for devices with slow __threadfence ("agent" fence which does L2 cache flushing and invalidation). Fences with "workgroup" scope are used instead to ensure ordering only but not coherence ... Global coherence of prefixes_*_values is ensured by atomic_load/atomic_store that bypass cache." (gfx942, gfx950, gfx1250).

Reading for the course, ours: on AMD the single-pass design is used in production, but its progress depends on hardware behaviour AMD has not documented, and AMD's own code needed architecture-specific fixes for both block order and spinning. In the labs: the hierarchical scan is the HIP default; a single-pass variant, if offered, always uses the atomic counter and a sleep in the spin, and is labelled `unverified` until a `reports/verify-*.md` from an AMD machine shows it passing.

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/docs/tutorial/reduction.rst (around lines 415 to 425 and 700 to 705)
- https://github.com/ROCm/libhipcxx/blob/therock-10.2/docs/libcudacxx/extended_api/execution_model.rst
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocprim/rocprim/include/rocprim/device/detail/ordered_block_id.hpp (lines 100 to 120, 170 to 185, 243 to 325)
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocprim/rocprim/include/rocprim/device/detail/lookback_scan_state.hpp (lines 42 to 57, 360 to 385, 1034 to 1075)
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocprim/rocprim/include/rocprim/device/detail/device_scan.hpp (lines 121, 186 to 188)

### B4. The flag atomics and the spin loop on HIP

**Verdict: PARTLY.** The atomics have a direct path with CUDA's spelling. There is no documented `__nanosleep`.

- `cuda::atomic_ref<unsigned, cuda::thread_scope_device>` through libhipcxx (ROCm 10.1 and later, infra B2). At `therock-10.2`, device-scope load and store pass the requested order straight to the compiler built-ins: `__dst = __hip_atomic_load(__ptr, __memorder, __HIP_MEMORY_SCOPE_AGENT);` and `__hip_atomic_store(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT);` (block scope uses `__HIP_MEMORY_SCOPE_WORKGROUP`). So Fig. 11.17 and NVIDIA's message-passing pattern (A5) keep their source text on HIP, given the libhipcxx target in the build.
- `__nanosleep`: not in the HIP language-extension docs at the tag, not in `amd_device_functions.h`, and not in HIPIFY's "CUDA Device API supported by HIP" table (which lists `__shfl_up_sync` as supported). COULD NOT VERIFY a HIP counterpart. rocPRIM uses the AMD compiler built-in `__builtin_amdgcn_s_sleep(1)` (B3), which is AMD-only and not described in the HIP docs; a portable lab would need an `#if` (ours).

- https://github.com/ROCm/libhipcxx/blob/therock-10.2/include/cuda/std/__atomic/functions/atomic_hip_generated.h (lines 49 to 55, 113 to 119)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_device_functions.h
- https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Device_API_supported_by_HIP.html

---

## Group C: Metal

### C1. The SIMD-group scan: `simd_prefix_inclusive_sum`, `simd_prefix_exclusive_sum`, and a hand-written Kogge-Stone with `simd_shuffle_up`

**Verdict: PARTLY.** The functions are confirmed and do in one call what Fig. 11.8 does in a loop. What they cannot tell you: the order of the additions, so the last bits may differ from a Kogge-Stone tree or a CPU loop.

MSL spec Table 6.15 (page 222):

- `T simd_prefix_inclusive_sum(T data)`: "For a given thread, returns the sum of the input values in data for all active threads with a lower or the same index in the SIMD-group."
- `T simd_prefix_exclusive_sum(T data)`: "For a given thread, returns the sum of the input values in data for all active threads with a lower index in the SIMD-group. The first thread in the group, returns T(0)."
- Products likewise (`simd_prefix_inclusive_product`, `simd_prefix_exclusive_product`; the exclusive one returns `T(1)` in the first thread).
- Availability "macOS: Metal 2.1 and later"; feature tables put SIMD-scoped reductions (Table 6.15 includes the prefix functions) at Apple7, so every Apple silicon Mac (infra C2). T includes `float` and `half` (infra C2).
- Both operate over "active threads": a lane that is inactive (a partial last SIMD-group, or a branch) neither contributes nor is counted. Pad out-of-range lanes with 0 and keep them active, as llama.cpp does (D1).
- COULD NOT VERIFY: the summation order inside these functions. The spec gives none (same gap as `simd_sum`, ch10 C1).

The hand-written form of Fig. 11.8:

- `T simd_shuffle_up(T data, ushort delta)`: "This function doesn't modify the lower delta lanes of data because it doesn't wrap values around the SIMD-group." Same rule as CUDA (A3), so the lane test stays. No mask and no width parameter.
- `simd_shuffle_and_fill_up(T data, T filling_data, ushort delta)`: "If the difference is negative, the operation copies values from the upper delta lanes of filling_data to the lower delta lanes of data." Passing the identity as `filling_data` removes the lane test. MLX uses exactly this for 8-byte types (D3). Feature tables: Apple8 and later ("SIMD shift and fill", infra C2), so not M1.
- Compiled here: `simd_prefix_inclusive_sum`, `simd_prefix_exclusive_sum` and `simd_shuffle_up` on `float` compile at `-std=metal3.2`, `metal4.0` and `metal4.1`.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 6.10.2, Tables 6.14 and 6.15, pages 220 to 222)

### C2. Building the block (threadgroup) scan of Fig. 11.9 on Metal

**Verdict: PARTLY.** The structure ports; the code does not. There is no threadgroup-level scan function, a helper function cannot declare threadgroup memory, and the "one SIMD-group scans the SIMD-group sums" step depends on a width Apple does not document.

- No threadgroup scan in MSL: the spec's collective functions are SIMD-group and quad-group scoped (sections 6.10.2, 6.10.3). A threadgroup scan is SIMD scans plus threadgroup memory plus `threadgroup_barrier(mem_flags::mem_threadgroup)` (ch05 C2), which is exactly what llama.cpp, MLX and PyTorch write (D1, D3, D4).
- `__shared__` inside a helper has no counterpart. MSL spec section 4.4: use the threadgroup address space to "Allocate a threadgroup variable in a kernel, mesh, or object function." Compiled here (`-std=metal3.2`): a helper `inline float blockScan(...)` that declares `threadgroup float warpSums[32];` fails with "error: variables in the threadgroup address space cannot be declared in a non-qualified function". Declare the array in the kernel and pass a `threadgroup float*` (ch05 C1 notes helpers may take such a pointer when called from a kernel).
- Width: Fig. 11.9 needs (threads per threadgroup / SIMD width) ≤ SIMD width. At 1024 threads that holds for width 32 (32 groups) but not for width 16 (64 groups), and Apple's own sizing example uses 16 (ch04 C2). Nothing documents which applies (ours: assert `threads_per_simdgroup * threads_per_simdgroup >= threads_per_threadgroup` in the kernel or on the host, or loop). Every project read for this file hard-codes 32: llama.cpp `#define N_SIMDWIDTH 32 // assuming SIMD group size is 32`, MLX `constexpr int simd_size = 32;`, PyTorch `C10_METAL_CONSTEXPR unsigned simdgroup_size = 32;` (D1, D3, D4).
- Lane and group indices come from `[[thread_index_in_simdgroup]]` and `[[simdgroup_index_in_threadgroup]]`, not from `thread_index % width` (ch04 C2).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 4.4, page 120; section 6.10, pages 215 to 234)
- https://github.com/pytorch/pytorch/blob/main/c10/metal/common.h (line 58)

### C3. Can one threadgroup see another threadgroup's plain stores to device memory in the same dispatch?

**Verdict: CONFIRMED** that Metal documents a default that does not provide it: device memory is threadgroup-coherent unless the buffer is declared `coherent(device)`.

MSL spec section 4.8 "Memory Coherency" (pages 122 to 123):

- "All OS: Metal 3.2 and later support coherent(device) qualifier and memory_coherence on textures for Apple silicon."
- "Device coherence — memory writes are visible to all threads on the device, that is, threads across threadgroups."
- "Memory in the thread address space has thread coherence, and memory in the threadgroup address space has threadgroup coherence. By default, memory in the device address space has threadgroup coherence."
- "Metal 3.2 and later support the coherent(device) qualifiers for buffers and memory_coherence_device for textures to indicate that the object has device coherence, that is, memory operations are visible across threads on the device if you properly synchronize them." Printed example: `coherent(device) device float4 *dptr2,` (and the shorter `coherent device float *dptr1,`).

Tension inside the spec, our reading: section 6.16.1.2 says that after an acquire that reads a release, "all memory writes (nonatomic and relaxed atomic) that happened-before the atomic store ... become visible to thread B", without mentioning coherence. Section 4.8 says a default device buffer is only threadgroup-coherent. A lab must not depend on which sentence wins: declare every buffer that another threadgroup of the same dispatch reads as `coherent(device)` (or move the payload into atomics). An Apple engineer said the same in 2021 about plain variables: "there are many caches and each version of a variable on each cache could get added independently (unless the variable is explicitly marked as atomic)" (C5).

Compiled here: `coherent(device) device float*` is accepted at `-std=metal3.2`, `metal4.0`, `metal4.1` and rejected at `-std=metal3.1`.

This matters for any cross-threadgroup pattern on Metal, including one that never waits (for example "the last threadgroup to finish does the final step", which needs no forward progress). It does not by itself make the lookback safe (C5).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 4.8, pages 122 to 123; section 6.16.1.2, page 300)

### C4. Acquire and release atomics, and fences, for the flags (refines infra C1)

**Verdict: PARTLY.** The spec documents acquire and release from Metal 4.1 but also still says "relaxed only". The compiler settles it: non-relaxed orders are accepted only in the new overloads that take a `mem_flags` argument.

From the spec:

- Section 6.16.1: "All OS: Metal 4.1 and later support the additional memory order values, memory_order_acquire, memory_order_release, and memory_order_acq_rel." The same section still says "For atomic operations other than atomic_thread_fence, memory_order_relaxed is the only enumeration value."
- Sections 6.16.4.1, 6.16.4.2, 6.16.4.5: "Metal 4.1 and later add support for mem_flags in atomic_store_explicit", the same for `atomic_load_explicit` and `atomic_fetch_key_explicit`, e.g. `C atomic_load_explicit(const device A* object, memory_order order, mem_flags flags)`. Section 6.16.4.5 keeps "The only supported value for order is memory_order_relaxed." above both forms.
- Section 6.16.3: "All OS: Metal 3.2 and later support atomic_thread_fence for Apple silicon." `void atomic_thread_fence(mem_flags flags, memory_order order, thread_scope scope = thread_scope_device)`; "In Metal 3.2 and later, you can use memory_order_seq_cst on atomic_thread_fence". The fence "establishes memory synchronization ordering of nonatomic and relaxed atomic accesses".
- Feature tables: "Acquire and release memory ordering atomics | Metal 3 & 4 | Apple7" (infra C1).

Compiled here (local compiler, SDK 27.0), what each form does:

| Source | `-std=metal3.2` / `metal4.0` | `-std=metal4.1` |
|---|---|---|
| `atomic_load_explicit(p, memory_order_acquire)` (two arguments) | error: `memory_order_acquire` undeclared | error: "'order' argument must be 'metal::memory_order_relaxed' if no 'mem_flags' argument is provided" |
| `atomic_load_explicit(p, memory_order_acquire, mem_flags::mem_device)` | not available | compiles |
| `atomic_store_explicit(p, 1u, memory_order_release, mem_flags::mem_device)` | not available | compiles |
| `atomic_fetch_add_explicit(p, 0u, memory_order_acquire, mem_flags::mem_device)` (the book's read) | not available | compiles, despite "only relaxed" in 6.16.4.5 |
| relaxed atomics + `atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)` before the flag store and after the flag load | compiles | compiles |

So a flag protocol can be written on Metal 3.2 (relaxed atomics plus device-scope fences) or on Metal 4.1 (acquire and release with `mem_flags`). Which language version the lab runtime compiles with is a project setting, not a vendor fact; check it before choosing. None of this provides forward progress (C5).

- MSL has no sleep or back-off function: the spec text has no "sleep", "yield" or "backoff" (searched today). A spin on Metal is a bare loop.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 6.16.1 to 6.16.4.5, pages 299 to 306)

### C5. A threadgroup that spins on another threadgroup's flag will eventually see it set (needed by Figs. 11.17 and 11.18)

**Verdict: WRONG** as an assumption on Metal. The spec gives no forward-progress guarantee between threadgroups, and Apple's engineer, answering on Apple's developer forum, called the assumptions behind it undefined. The single-pass scan with lookback is unsafe on Metal; teach the hierarchical scan there.

- MSL spec: the only progress statement is within a threadgroup, section 5.2.3.6 (page 157): "SIMD-groups execute concurrently within a given threadgroup and make independent forward progress with respect to each other, in the absence of threadgroup barrier operations." Searched the whole spec for "progress", "deadlock", "spin", "starv", "schedul" and "any order": nothing about threadgroups relative to each other.
- Apple Developer Forums, thread "Indeterminate "while" loop in metal shader causes (IOAF code 5) on DTK and M1" (January 2021). The poster's kernel was a chained prefix sum: `if (group_id != 0) while(status_flags[group_id - 1] == 0){}`. Accepted answer by a "Graphics and Games Engineer" marked "Apple Staff": "This kernel relies on some undefined behavior which is why it works on some GPUs but not others. There are 3 invalid assumptions you're relying on for this kernel to work 1. That the GPU will launch threadgroups in an order consistent with threadgroup_position_in_grid. 2. That for any pair of threadgroups that run concurrently [in real time] on the GPU, the GPU will fairly schedule them so that both make forward progress concurrently. 3. That operations on memory marked as volatile are atomic." "We suggest using a more traditional prefix sum algorithm." Follow-up from the same engineer: "Part of the problem with documenting #2 is that Metal is supported on a variety of different GPU all of which will schedule work differently."
- The book's counter fixes assumption 1 and atomics fix 3 (C4), but nothing fixes 2: unlike CUDA (A6), Metal does not promise that a threadgroup which has started keeps running while another spins.
- A newer thread on the same forum ("Apple GPU forward progress guarantees for persistent-thread synchronization?", June 2026) asks "Does Metal provide any documented forward-progress guarantees across threadgroups beyond what is explicitly stated in the Metal specification?" and had 0 replies today.
- No grid-wide barrier or cooperative launch exists either (ch04 C3), so method (a) is out too. On Metal the global scan is the multi-dispatch one: llama.cpp does scan-scan-add in three dispatches, PyTorch does reduce, scan, carry in three (D1, D4).

The forum answer is an Apple engineer's statement, not documentation; with the spec's silence it is the strongest Apple source found.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 5.2.3.6, page 157)
- https://developer.apple.com/forums/thread/672532
- https://developer.apple.com/forums/thread/831017

### C6. A library scan on Apple GPUs: MPSGraph `cumulativeSum`

**Verdict: CONFIRMED.** MPSGraph has a cumulative sum with exclusive and reverse switches, which covers both the inclusive/exclusive distinction of §11.1 and the "mass above" form mlx-lm uses (D3).

Header `MPSGraphCumulativeOps.h` (macOS SDK 27.0):

```objc
/// - exclusive: If true, perform the exclusive cumulative operation, and the first element will be equal to zero
/// - reverse: If true, reverse the direction of the cumulative operation along the specified axis
-(MPSGraphTensor *) cumulativeSumWithTensor:(MPSGraphTensor *) tensor
                                       axis:(NSInteger) axis
                                  exclusive:(BOOL) exclusive
                                    reverse:(BOOL) reverse
                                       name:(NSString * _Nullable) name
MPS_SWIFT_NAME( cumulativeSum(_:axis:exclusive:reverse:name:))
MPS_AVAILABLE_STARTING(macos(13.0), ios(16.0), tvos(16.0));
```

- Also `cumulativeProduct`, `cumulativeMinimum`, `cumulativeMaximum` with the same switches. No determinism or summation-order statement in the header.
- Apple documents no scan callable from inside a custom Metal kernel beyond the SIMD-group functions (C1).

- Local header: `MetalPerformanceShadersGraph.framework/Headers/MPSGraphCumulativeOps.h` (macOS 27.0 SDK), lines 16 to 51
- https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph

---

## Group D: How llama.cpp, MLX and PyTorch do the same things

All files read from the default branch on 2026-11-20; last-commit dates from the GitHub commits API. "Our reading" marks conclusions drawn from the code, not observed results.

### D1. llama.cpp `GGML_OP_CUMSUM`: the book's coarsened warp-scan design on CUDA and HIP, the three-kernel scan-scan-add on Metal

**Verdict: CONFIRMED.**

CUDA and HIP, `ggml/src/ggml-cuda/cumsum.cu` (last commit 2026-01-04):

- `cumsum_kernel`: one block per row (`i1 = blockIdx.x`), `constexpr int num_unroll = 4;` elements per thread scanned sequentially in registers (the coarsening of Fig. 11.12, register tile of Fig. 11.13), then `val = warp_prefix_inclusive_sum<T, warp_size>(val);` (Fig. 11.8), the last lane writes `s_warp_sums[warp]`, warp 0 scans the warp sums and stores `inc - w` "// exclusive sum" (Fig. 11.9), then each thread adds its offset. A loop walks the row in chunks of `num_unroll * blockDim.x` with a running `*s_carry`, so one block handles a row of any length with no inter-block communication.
- Loads are thread-contiguous: `int64_t idx = i + tid * num_unroll;` then `src_row[idx + j]`. That is the pattern the book warns is uncoalesced (p. 269); the book loads chunk by chunk through shared memory instead (our reading; the lab can measure the difference).
- `warp_prefix_inclusive_sum` in `common.cuh`: `for (int offset = 1; offset < width; offset <<= 1) { const T t = ggml_cuda_shfl_up(x, offset, width); if (lane_id >= offset) { x += t; } }`, where `ggml_cuda_shfl_up` calls `__shfl_up_sync(mask, value, delta, width)` with the typed `GGML_CUDA_FULL_THREAD_MASK` (ch10 D1) and `width = ggml_cuda_get_physical_warp_size()`: 64 under HIP on `__GFX9__`/`__GFX8__`, else 32 (B1).
- CUB path: `#if !defined(GGML_USE_HIP) && (defined(GGML_USE_MUSA) || CUDART_VERSION >= 11070)` defines `GGML_CUDA_USE_CUB`; then long single rows (`(nrows == 1) && (ne00 > 1024)`, or `ne00 / nrows > 4096`) call `cub::DeviceScan::InclusiveSum` per row, and rows of 1024 or more use a `cub::BlockScan<T, 256>` kernel. HIP builds never take these, so on AMD the hand-written kernel always runs.

Metal, `ggml/src/ggml-metal/kernels/reduce.metal` (last commit 2026-08-24) and `ggml-metal-ops.cpp`:

- `kernel_cumsum_blk`: one element per thread, `float s = simd_prefix_inclusive_sum(v);`, lane `N_SIMDWIDTH - 1` writes the SIMD-group total to threadgroup memory, barrier, SIMD-group 0 does `shmem_f32[tiisg] = simd_prefix_exclusive_sum(shmem_f32[tiisg]);`, barrier, add. Out-of-range lanes load 0. The last thread writes the block total to `tmp` when the row spans several threadgroups.
- Host (`ggml_metal_op_cumsum`): dispatch 1 scans blocks; if `ne00 > nth`, dispatch 2 runs the same kernel over `tmp` (asserting `ne00 <= nth*nth`, so the block sums fit one threadgroup); dispatch 3, `kernel_cumsum_add`, adds `tmp_row[ib - 1]` to every element of block `ib`. That is the book's three-kernel scan-scan-add (Fig. 11.14, 16 bytes per element), the Metal-safe design of C5.
- `kernels/common.h`: `#define N_SIMDWIDTH 32 // assuming SIMD group size is 32` (C2).

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/cumsum.cu (lines 86 to 186, 186 to 307)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/common.cuh (lines 115 to 119, 415 to 444, 598 to 608)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/reduce.metal (lines 118 to 228)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-ops.cpp (lines 1088 to 1230)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/common.h (line 23)

### D2. llama.cpp top-p: both paths keep the token that crosses p

**Verdict: CONFIRMED** as source read; the equivalence below is our derivation.

`src/llama-sampler.cpp` (last commit 2026-11-12):

- Host sampler `llama_sampler_top_p_apply`: softmax, sort descending (a partial sort of the top 256 when there are more than 1024 candidates, widened if needed), then a running inclusive sum: `cum_sum += pdata[i].p;` and `if (cum_sum >= ctx->p && i + 1 >= ctx->min_keep) { last_idx = i + 1; break; }`, with the comment "we set the last index to i+1 to indicate that the current iterate should be included in the set". `p >= 1.0f` returns early.
- Backend (graph) sampler `llama_sampler_top_p_backend_apply`: `ggml_argsort(..., GGML_SORT_ORDER_DESC)`, softmax of the sorted logits, `cdf = ggml_cumsum(ctx, softmax)` (inclusive), `cdf_scaled = ggml_scale_bias(ctx, cdf, -1.0f, p)` (p minus cdf), `mask = ggml_step(ctx, cdf_scaled)` where step is `x > 0.0f` (`ggml-cuda/unary.cu`, `op_step`), so the mask keeps tokens whose inclusive sum is strictly below p; then "// Make top-p inclusive (i.e. return all values such that cum_sum/cdf >= p)": the count of kept tokens is used as an index and that one more token is set to 1. Masked tokens get `log(0) = -INFINITY` as a bias.

Our derivation: both paths keep token i exactly when the probability mass ranked strictly before i is below p, that is when the **exclusive** scan at i is below p, or equivalently when the inclusive scan at i − 1 is below p. At an exact boundary (inclusive sum at i equal to p in float), token i is kept and token i + 1 is not. Tokens tied in probability are ordered by whatever the sort produced, so which of two tied tokens at the boundary survives is not specified. The book's §11.1 off-by-one is precisely the difference between this rule and "keep while inclusive < p", which would drop the token that crosses p.

- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-sampler.cpp (lines 1544 to 1710)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/unary.cu (lines 16 to 18)

### D3. MLX: a per-row threadgroup scan built on `simd_prefix_*`, and mlx-lm's top-p, which keeps every token tied with the boundary

**Verdict: CONFIRMED.**

`mlx/backend/metal/kernels/scan.h` (last commit 2026-10-26):

- `struct CumSum` maps its SIMD scan to `simd_prefix_inclusive_sum(x)` and its exclusive SIMD scan to `simd_prefix_exclusive_sum(x)`; 8-byte types instead loop `for (int i = 1; i <= 16; i *= 2) { val = operator()(val, simd_shuffle_and_fill_up(val, init, i)); }` (Kogge-Stone with the identity filled in, C1).
- `contiguous_scan`: one threadgroup per row; `constexpr int simd_size = 32;`; each thread reads `N_READS` contiguous values (4 for 4-byte types, 2 otherwise, from `scan.cpp`) and scans them sequentially; `U prev_thread = op.simd_exclusive_scan(values[N_READS - 1]);`; SIMD-group totals go to `threadgroup U simdgroup_sums[32];` and SIMD-group 0 scans them; a loop over the row carries `prefix = simdgroup_sums[0];` into the next chunk. The comment lists the stages: "Compute inclusive scan per thread / Compute exclusive scan of thread sums in simdgroup / Write simdgroup sums in SM / Compute exclusive scan of simdgroup sums". It is Figs. 11.9 and 11.12 again, with no communication between threadgroups.
- Python API (docs 0.32.4): `cumsum(a, /, axis=None, *, reverse=False, inclusive=True, dtype=None, stream=None)`; `inclusive`: "The i-th element of the output includes the i-th element of the input."

mlx-lm `mlx_lm/sample_utils.py` (last commit 2026-09-24), `apply_top_p`:

```python
sorted_logprobs = mx.sort(logprobs, axis=-1)
sorted_probs = mx.exp(sorted_logprobs.astype(mx.float32))
mass_above = mx.cumsum(sorted_probs, axis=-1, reverse=True, inclusive=False)
total_mass = mass_above[..., :1] + sorted_probs[..., :1]
num_dropped = (mass_above >= top_p * total_mass).sum(axis=-1, keepdims=True)
threshold = mx.take_along_axis(sorted_logprobs, num_dropped, axis=-1)
return mx.where(logprobs < threshold, -float("inf"), logprobs)
```

Our reading: the sort is ascending, so a reverse exclusive scan gives each token the mass of the tokens ranked above it. A token is dropped when that mass is at least p times the total, the same exclusive-scan rule as D2, normalised by the computed total instead of assuming it is 1. The final step keeps every token whose log-probability is at least the threshold token's, so all tokens tied with the boundary token are kept. That is a different tie rule from llama.cpp's (D2), and it is one the M5 lab could state and test.

- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/scan.h (lines 1 to 60, 247 to 402)
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/scan.cpp (lines 70 to 105)
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.cumsum.html
- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/sample_utils.py (lines 213 to 229)

### D4. PyTorch: `torch.cumsum` on CUDA and MPS, its documented determinism switch, and the top-p rule in the PyTorch ecosystem

**Verdict: CONFIRMED.**

CUDA, `aten/src/ATen/native/cuda/ScanUtils.cuh` (last commit 2026-10-05):

- `scan_dim`: a 1-D tensor goes to CUB (`cuda::cub::inclusive_scan`, a wrapper over `DeviceScan`; on ROCm the wrapper calls `hipcub::DeviceScan`), unless deterministic algorithms are on and the type is floating point or complex, in which case it calls `cuda::cub::inclusive_deterministic_scan` (two kernels, `calc_block_sums` then `final_scan_kernel`, with the grid capped at the SM count, in `aten/src/ATen/cuda/cub.cuh`). A scan along the last dimension, the batch-by-vocabulary case, goes to `scan_innermost_dim`.
- `tensor_kernel_scan_innermost_dim_impl`: blocks of up to 512 threads, several rows per block (`threadIdx.y`), two elements per thread in shared memory; before each chunk `row_buf[0] = binary_op(row_buf[0], block_total);` adds the running total of earlier chunks to the chunk's first element (the trick of the book's reduce-scan-scan, Fig. 11.15, applied within one block); then "// Parallel reduction with Sklansky method", one `__syncthreads()` per step. Sklansky is a third prefix network, neither Kogge-Stone nor Brent-Kung; the comment links the decoupled look-back report for the diagram.
- Docs, `torch.use_deterministic_algorithms` (2.15): under "The following normally-nondeterministic operations will act deterministically when mode=True:" the list includes "torch.cumsum() when called on a CUDA tensor when dtype is floating point or complex". So PyTorch documents that its default CUDA cumsum may vary from run to run, matching CUB (A8).

MPS, `aten/src/ATen/native/mps/kernels/ScanKernel.metal` (last commit 2026-06-10):

- "The reminder of this file contains cummin and cummax implementations adapted from MLX"; SIMD scans are written with `simd_shuffle_and_fill_up` loops; `simd_size` is `c10::metal::simdgroup_size`, which is 32.
- A comment marks the book's write-after-read race (p. 257) in production code: after reading the carried prefix out of `simdgroup_sums`, "// Ensure every thread has read the carry before the next iteration // overwrites simdgroup_sums; otherwise long (multi-iteration) scans race." followed by a barrier. (MLX's loop has its barrier at the top of the next iteration instead, before `simdgroup_sums` is written; our reading.)
- Long axes: "// Three-pass scan (block_reduce -> scan_block_sums -> block_carry) over a [n_scans, axis_size] tensor; splits a long axis across threadgroups." That is reduce-scan-scan over separate dispatches (Fig. 11.15, about 12 bytes per element), the Metal-safe structure of C5.

The top-p rule most PyTorch users run (not one of the brief's three projects; added because it is the reference the reader will compare against):

- Hugging Face transformers `TopPLogitsWarper` (`src/transformers/generation/logits_process.py`, last commit 2026-11-18): `sorted_logits, sorted_indices = torch.sort(scores, descending=False)`, `cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)`, `sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)`, then `sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0`.
- vLLM `apply_top_k_top_p_pytorch` (`vllm/v1/sample/ops/topk_topp_sampler.py`, last commit 2026-11-19): the same ascending sort and `top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)`, with `top_p_mask[:, -1] = False` "# at least one". vLLM's default on GPUs is a Triton path (`apply_top_k_top_p_triton`), not this function.
- Our derivation: with probabilities summing to exactly 1, "inclusive ascending sum ≤ 1 − p" means "mass ranked above ≥ p", so these drop exactly what D2 and D3 drop. In float they compute it differently (an ascending inclusive scan compared with 1 − p, against a descending exclusive scan compared with p), so the rounding differs and a boundary token can land on different sides. That, and `torch.sort` order among ties, is why the M5 tests need a tolerance band around p and a stated tie rule.

- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/ScanUtils.cuh (lines 300 to 474)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/cub.cuh (lines 455 to 500)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/kernels/ScanKernel.metal (lines 18 to 66, 366 to 380, 786 to 880)
- https://docs.pytorch.org/docs/2.15/generated/torch.use_deterministic_algorithms.html
- https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py (`TopPLogitsWarper`)
- https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/topk_topp_sampler.py (lines 198 to 270)

---

## Book items with no vendor fact to check

- Kogge-Stone (1973) and Brent-Kung (1982) as adder designs, and "still being used in high-speed arithmetic hardware" (p. 255): historical and about circuits; no vendor document checked.
- The work and step formulas (pp. 265 to 270, 282 to 284) and the exercise answers in the notes: arithmetic, checked by the simulators' unit tests, not by vendors.
- The energy remark (p. 267): no vendor figure checked; do not add one.
- "Grid-wide barriers are very expensive" (p. 277): no NVIDIA cost figure found in the pages read for this file; keep it as the book's judgement and let the reader measure if a lab uses it.

## Answers to the notes' portability questions

| Notes' `[unverified]` item | Answer | Where |
|---|---|---|
| `__syncthreads()` and `__shared__` on HIP and Metal | Same spelling on HIP; `threadgroup_barrier` and `threadgroup` arrays on Metal. Metal helpers cannot declare threadgroup arrays; HIP undocumented for `__device__` functions | ch05 B4, C1, C2; B1, C2 here |
| `__shfl_up_sync`, full mask, 32-lane warps | HIP: same name, 64-bit mask required, width 64 on gfx9. Metal: `simd_shuffle_up` (no mask) or one call to `simd_prefix_inclusive_sum` | B1, C1 |
| `cuda::atomic_ref` with acquire/release, busy-waiting | CUDA documented (message passing, execution model). HIP: same source through libhipcxx; no progress guarantee documented. Metal: acquire/release only in the Metal 4.1 `mem_flags` overloads, or fences from 3.2; buffers need `coherent(device)`; no progress guarantee | A5, A6, B3, B4, C3, C4, C5 |
| Dynamic block index | CUDA: makes the wait safe under the documented model. AMD: rocPRIM enables it only on some architectures. Metal: does not help, threadgroups have no fair-scheduling guarantee | A6, B3, C5 |
| Cooperative-groups grid barrier (method a) | CUDA and HIP with a cooperative launch and a resident grid; none on Metal | infra A4, B4; ch04 C3 |
| `#pragma unroll` and register promotion | Same spelling on HIP; no unroll control documented in MSL | infra A7, B7; ch06 C4 |
| Fallback that needs no inter-block waiting | Three-kernel scan-scan-add or reduce-scan-scan: what llama.cpp (Metal) and PyTorch (MPS) ship | D1, D4 |

---

## Summary of corrections

Every WRONG, PARTLY and COULD NOT VERIFY item, and every failed or unfound sub-claim:

- A1: the book's 66.9 TFLOPS is not an NVIDIA figure (NVIDIA prints 67 for H100 SXM, ch05 A8); every derived number (20 FLOP/B; 419, 209, 279 ×10⁹ elements/s) is still right.
- A2: NVIDIA's profiler uses the same ~80% line ("Above ~80%: high (near-limit regime)"); no vendor source was found for "even a copy kernel does not reach 100%" (could not verify).
- A4: CUDA's cooperative-groups `inclusive_scan` / `exclusive_scan` reference allows only `coalesced_group` and `thread_block_tile`, while the guide's own example passes a `thread_block`; CUB's default `BlockScan` is a raking reduce-then-scan, not the book's scan-scan-add.
- A5 (detail): NVIDIA's own message-passing example reads the flag with `load(memory_order_acquire)`; the book's `fetch_add(0, acquire)` works but is a read-modify-write where a load suffices.
- A6 (detail): NVIDIA's CUB does not use the book's block counter; it indexes tiles by `blockIdx.x` with the comment "Blocks are launched in increasing order", which the programming guide does not promise.
- B1 (WRONG): Fig. 11.8 does not compile under HIP (32-bit mask literal), and with `WARP_SIZE` 32 on a 64-wide wavefront each half scans separately; whether HIP accepts `__shared__` inside a `__device__` function is not documented.
- B2 (detail): HIP's cooperative-groups scan docs say each thread contributes "the same per-thread value", which reads as a slip; the first lane of `exclusive_scan` returns the operation's identity on AMD and `T{}` on NVIDIA, which differ for `less`, `greater` and `bit_and`.
- B3 (could not verify): AMD documents no forward progress between workgroups; libhipcxx's execution-model page is NVIDIA's text unchanged and its published URL is 404; rocPRIM's lookback uses `blockIdx` order by default, switches to the atomic counter only on gfx942, gfx950 and gfx1250, and needs `s_sleep` on early gfx908 revisions to avoid a hang.
- B4: no `__nanosleep` on HIP in the docs, headers or HIPIFY table read; rocPRIM uses `__builtin_amdgcn_s_sleep`.
- C1: the summation order of `simd_prefix_inclusive_sum` / `simd_prefix_exclusive_sum` is not documented (could not verify); `simd_shuffle_and_fill_up` needs Apple8.
- C2: Fig. 11.9 does not port as written: MSL rejects a threadgroup array declared in a helper ("variables in the threadgroup address space cannot be declared in a non-qualified function"), and the single-SIMD-group scan of SIMD-group sums depends on a width Apple does not document.
- C3 (new trap): Metal device memory is threadgroup-coherent by default; a buffer read by another threadgroup in the same dispatch must be `coherent(device)` (Metal 3.2 and later, Apple silicon), and the spec's release-acquire text does not mention this.
- C4 (refines infra C1): in MSL 4.1, acquire and release are accepted only by the new overloads that take `mem_flags`; the two-argument forms still require `memory_order_relaxed` (compiler message quoted); the spec's "only relaxed" sentence in 6.16.4.5 is contradicted by the compiler for `atomic_fetch_add_explicit` with `mem_flags`. Before 4.1, use relaxed atomics with `atomic_thread_fence` (Metal 3.2). MSL has no sleep function.
- C5 (WRONG): the single-pass scan with lookback is unsafe on Metal: no forward progress between threadgroups is documented, and an Apple engineer called the chained-scan assumptions undefined behaviour; use the multi-dispatch hierarchical scan.

Status facts worth knowing even where the verdict is CONFIRMED:

- A3: the lanes below `stride` keep their own value on all three backends, which settles the notes' "Unclear in the PDF" item; NVIDIA prints Fig. 11.8's loop as its own example.
- A6: NVIDIA's execution model (guide section 5.9) is what makes the book's dynamic block index provably sufficient on CUDA; a spin must be an atomic (or volatile) read of non-automatic storage to count as progress.
- A7: CUDA has `__nanosleep` for back-off in spin loops; CUB uses it in its look-back.
- A8: `std::inclusive_scan`, Thrust, CUB `DeviceScan` and rocPRIM all document run-to-run variation for floating-point scans by default; CUB has a `run_to_run` mode for float `plus`, rocPRIM has `deterministic_inclusive_scan`, PyTorch switches its 1-D CUDA cumsum to a deterministic kernel under `use_deterministic_algorithms`.
- D1, D3, D4: the common design is one block (threadgroup) per row with a carried prefix (MLX, PyTorch, and llama.cpp's hand-written kernel). The exception is llama.cpp's CUDA build (not HIP): when a row is long (`nrows == 1 && ne00 > 1024`, or `ne00 / nrows > 4096`) it hands each row to `cub::DeviceScan::InclusiveSum`, CUB's single-pass decoupled look-back scan, so a batch of long rows can take the inter-block look-back path. Metal projects use multi-dispatch hierarchical scans for long rows. (Corrected 2026-11-20 by the main session after the chapter 11 review; the earlier wording said no project uses look-back for batched rows, which D1's own CUB condition contradicts.)
- D2, D3, D4: llama.cpp, mlx-lm, transformers and vLLM all keep the token that crosses p (the exclusive-scan rule), compute it in three different ways, and break ties at the boundary differently (by sort order; or by keeping every token tied with the boundary in mlx-lm).

## Where the backends really differ

One line per cell, for what this chapter uses. "Same" means the same source text compiles through `pmpp/vendors/hip.h`, according to the documentation; nothing here was run on CUDA or AMD hardware, and the Metal cells marked "compiled here" were compiled, not run.

| Feature | CUDA | HIP (ROCm 10.2) | Metal (MSL 4.1) |
|---|---|---|---|
| Warp width for a warp scan | 32 on every listed device | 64 on gfx9, 32 on gfx10 and later; compile-time via template or `__GFX9__` | Undocumented; read `[[threads_per_simdgroup]]`; projects assume 32 |
| Shuffle-up for Kogge-Stone | `__shfl_up_sync(mask, v, d, width)`, 32-bit mask | Same name; 64-bit mask required | `simd_shuffle_up(v, d)`, no mask, no width |
| Lanes below the stride | Keep their own value | Return their own value | Unchanged; `simd_shuffle_and_fill_up` can fill the identity (Apple8 and later) |
| One-call warp prefix sum | None for `float` (use the shuffle loop, `cub::WarpScan`, or `cg::inclusive_scan` on a tile) | `cg::inclusive_scan` on a wavefront-sized tile uses DPP, `float` accelerated; `hipcub::WarpScan` | `simd_prefix_inclusive_sum`, `simd_prefix_exclusive_sum` on `float`/`half`, Apple7 and later; order undocumented |
| Shared array declared in a helper (Fig. 11.9) | Allowed (`__shared__` in a `__device__` function) | Not documented | Compile error; declare in the kernel, pass a `threadgroup` pointer |
| Block scan from a library | `cub::BlockScan` (raking by default, or warp-scans) | `hipcub::BlockScan` over rocPRIM | None; SIMD functions plus threadgroup memory |
| Device-wide scan library | `cub::DeviceScan`, single-pass decoupled look-back; run-to-run variation unless `run_to_run` | `hipcub::DeviceScan` / `rocprim::inclusive_scan`; `deterministic_inclusive_scan` for reproducible floats | MPSGraph `cumulativeSum(_:axis:exclusive:reverse:name:)`, macOS 13 and later |
| Flag store / load ordering | `cuda::atomic_ref<..., thread_scope_device>` `store(release)` / `load(acquire)` | Same source through libhipcxx (`__hip_atomic_*`, agent scope) | Metal 4.1: `atomic_store_explicit(..., memory_order_release, mem_flags::mem_device)` and the acquire load with `mem_flags`; Metal 3.2: relaxed atomics plus `atomic_thread_fence(..., seq_cst, thread_scope_device)` |
| Seeing another block's plain stores | Made visible by the release/acquire pair (NVIDIA's message-passing example) | Same model through libhipcxx; not separately documented | Buffer must be `coherent(device)` (Metal 3.2 and later); default is threadgroup coherence |
| Back-off in a spin | `__nanosleep(ns)`, up to about 1 ms | No documented call; rocPRIM uses `__builtin_amdgcn_s_sleep` | None |
| Forward progress for a block that waits on another | Documented: a block that has executed a step eventually progresses; waiting on a block that claimed its index atomically is safe | Not documented; works in AMD's library with per-architecture fixes; `unverified` | Not documented; Apple engineer: undefined; do not spin across threadgroups |
| Dynamic block index (atomic counter) | Sufficient under the documented model; CUB uses `blockIdx` order instead | rocPRIM uses it on gfx942, gfx950, gfx1250 only, by default | Does not help without fair scheduling |
| Grid-wide barrier (method a) | `grid.sync()` under a cooperative launch, grid must be resident | Same with `hipLaunchCooperativeKernel` | None; use separate dispatches |
| Global scan the course should teach as portable | Hierarchical, or single-pass with the counter | Hierarchical; single-pass `unverified` | Hierarchical only (three dispatches) |
| What real projects ship for batched rows | One block per row with a carry (llama.cpp, PyTorch); llama.cpp sends long rows, single or batched (`ne00 / nrows > 4096`), to CUB's look-back `DeviceScan` | llama.cpp's hand-written kernel (CUB disabled); PyTorch via hipCUB | One threadgroup per row with a carry (MLX, PyTorch); multi-dispatch for long rows (llama.cpp, PyTorch) |
