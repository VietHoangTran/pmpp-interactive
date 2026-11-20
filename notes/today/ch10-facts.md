# Chapter 10 fact check: CUDA, HIP, Metal

Checked on 2026-11-20 against the live vendor sources listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY. Where an item has several sub-claims, the verdict is for the item as a whole and the text says which sub-claim failed. A verdict judges the claim in the item title (usually the book's statement, or the assumption a course author would bring).
- **Quote** is copied from the source (HTML docs flattened to text, GitHub raw files at a tag or on the default branch, Apple's documentation JSON, PDF text). Line breaks inside quotes are removed.
- "infra A3", "ch04 A10", "ch05 B5" point at items in `notes/today/infra-facts.md`, `ch04-facts.md`, `ch05-facts.md`. Anything settled there is cited, not re-checked.
- Page numbers ("p. 238") are book pages as recorded in `notes/ch10.md`.
- Versions the sources reported about themselves on the check date: CUDA Programming Guide and CUDA Math API "v13.5" (pages last updated Oct 31 and Nov 02, 2026); CCCL docs at `nvidia.github.io/cccl/unstable/` and CCCL source on `main`; HIP docs for ROCm 10.2.0 ("HIP 7.17.0") read as `.rst` sources at tag `therock-10.2` of `ROCm/rocm-systems`; libhipcxx and hipCUB at tag `therock-10.2` of `ROCm/libhipcxx` and `ROCm/rocm-libraries`; Metal Shading Language Specification "Version 4.1", dated 2026-06-04 ("MSL spec"); llama.cpp `master`, MLX `main`, PyTorch `main` on the check date.
- Not checked: anything by running code on hardware. Every statement is from documentation, headers or project source. Where this file says what a piece of code does on a particular input (tie-breaks, NaN), that is our reading of the source, labelled as such, not an observed result.

Totals: 18 CONFIRMED, 1 WRONG, 8 PARTLY, 0 COULD NOT VERIFY (27 items: 13 CUDA, 5 HIP, 5 Metal, 4 project). Sub-claims that failed or could not be found are collected under "Summary of corrections".

---

## Already settled elsewhere (cite, do not re-check)

| Chapter 10 needs | Settled in |
|---|---|
| 1024 threads per block, warp size 32 on every listed compute capability | ch05 A5, ch03 A3 |
| AMD wavefront 64 on gfx9, 32 on gfx10 and later; do not hard-code | ch05 B5, ch04 B6 |
| Block limits on AMD (no number in current docs; query the device) | ch03 B3 |
| Metal SIMD-group width undocumented; read `[[threads_per_simdgroup]]`; grouping is implementation-defined | ch04 C2, infra C5 |
| Metal threads per threadgroup: 1024 in the tables, real cap per pipeline | ch05 C3, ch03 C4 |
| `__syncthreads()`: uniform-condition rule, exited threads do not block | ch04 A6; HIP ch04 B3; Metal ch05 C2, ch04 C3 |
| `__syncwarp()` signature, independent thread scheduling from compute capability 7.0 | ch04 A10; HIP ch04 B4; AMD lockstep ch04 B5; Metal `simdgroup_barrier` infra C4 |
| Shuffle signatures, types, out-of-range rules, mask constraints | CUDA infra A3; HIP infra B3; Metal infra C2 |
| `__reduce_*_sync` (compute capability 8.x, integers only) | ch04 A20, infra A3; HIP infra B3 |
| Legacy atomics: `atomicAdd(float*)` exists, no float `atomicMax`, 64-bit `atomicMax(unsigned long long*)` exists | infra A1; HIP infra B1; Metal infra C1 |
| `cuda::atomic_ref`, scopes, `memory_order_relaxed`, `fetch_min` / `fetch_max` | infra A2; HIP (libhipcxx) infra B2 |
| Cooperative groups, grid-wide `sync()`, header names on HIP | infra A4, ch04 A7; HIP infra B4, ch04 B11; Metal: none, ch04 C3 |
| `cub::BlockReduce`, `cub::WarpReduce`, removed `cub::Max()` / `cub::Sum()` | infra A11; hipCUB infra B11 |
| Metal default math mode: three Apple sources, three answers | ch05 C7 |
| Metal threadgroup memory 32 KB; dynamic threadgroup memory | ch05 C1, C8 |
| No `double` in MSL | infra C8 |
| How llama.cpp, MLX and PyTorch handle warp width | ch04 D1 to D4 |
| llama.cpp sampling runs on the host by default; an experimental backend path exists | ch01 D2 |
| Old guide URL `cuda-c-programming-guide/` is gone; the guide is now titled "CUDA Programming Guide" | ch05 A9 |

---

## Group A: CUDA and the book's dated statements

### A1. `__syncthreads()` doubles as a memory fence for the block; where the memory consistency model is documented (p. 230)

**Verdict: CONFIRMED.** The current guide states the ordering guarantee directly, in C++ memory-model terms. The "memory consistency model" the book points to is now its own appendix section, "5.8. CUDA C++ Memory Model".

Quotes (guide section 5.4.4.1):

- "__syncthreads*() wait until all non-exited threads in the thread block simultaneously reach the same __syncthreads*() intrinsic call in the program or exit."
- "__syncthreads*() provide memory ordering among participating threads: the call to __syncthreads*() intrinsics strongly happens before (see C++ specification [intro.races] ) any participating thread is unblocked from the wait or exits."

Quotes (section 5.4.4.3 "Memory Fence Functions"):

- "The CUDA programming model assumes a weakly ordered memory model. In other words, the order in which a CUDA thread writes data to shared memory, global memory, page-locked host memory, or the memory of a peer device is not necessarily the order in which another CUDA or host thread observes the data being written. Reading from or writing to the same memory location without memory fences or synchronization results in undefined behavior."
- Block-scope fence: `__threadfence_block()`, or the preferred `cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block);` ("It is suggested to use cuda::atomic_thread_fence provided by libcu++ whenever possible for safety and portability reasons.")

Section 5.8: "CUDA C++ extends the standard C++ memory model and concurrency facilities" and, on races, "Any such data race results in undefined behavior." Subsections: Thread Scopes, Synchronization primitives, Atomicity, Data Races, Example: Message Passing. The page does not mention `__syncthreads`; the barrier's ordering guarantee is the sentence from 5.4.4.1 above.

For the lesson: the book's "also serves as a memory fence for threads within the same block" is NVIDIA's rule today. A reduction that drops a barrier between a shared-memory write and another thread's read has a data race, which is undefined behaviour, not merely a stale value.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (sections 5.4.4.1 and 5.4.4.3)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html (section 5.8)

### A2. `__syncwarp()` "makes diverged threads of the warp reconverge" (p. 237)

**Verdict: PARTLY.** The barrier half of the book's sentence is NVIDIA's: all named threads arrive before any proceeds, with memory ordering. The reconvergence half is not how NVIDIA describes it. The guide promises convergence at a `_sync` call, not afterwards.

Quotes (guide sections 5.4.4.2, 5.4.6.1, 5.4.6.6):

- `void __syncwarp(unsigned mask = 0xFFFFFFFF);` "The intrinsic function __syncwarp() coordinates communication between the threads within the same warp. When some threads within a warp access the same addresses in shared or global memory, potential read-after-write, write-after-read, or write-after-write hazards may occur. These data hazards can be avoided by synchronizing the threads between these accesses."
- "Calling __syncwarp(mask) provides memory ordering among the participating threads within a warp named in mask : the call to __syncwarp(mask) strongly happens before ... any warp thread named in mask is unblocked from the wait or exits."
- The `_sync` intrinsics, `__syncwarp` included, "use the mask parameter to indicate which warp threads participate in the call. This parameter ensures proper convergence before the hardware executes the intrinsic."
- "Note that threads convergent at an __activemask() call are not guaranteed to remain convergent at subsequent instructions unless those instructions are warp synchronizing intrinsics ( __sync )."

The word "reconverge" appears in the current guide only in the independent-thread-scheduling description ("threads can now diverge and reconverge at sub-warp granularity", A5). Safe statement: `__syncwarp()` is a barrier and a memory fence for the lanes in its mask; do not rely on lanes staying in step after it, and put a `_sync` intrinsic or another `__syncwarp()` wherever lanes exchange data.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (sections 5.4.4.2, 5.4.6.1, 5.4.6.6)

### A3. `__shfl_down_sync` as printed in Fig. 10.11, the width parameter (p. 241), and what `warp_reduce` does at the top lanes

**Verdict: CONFIRMED**, with two precisions. The signature is current (infra A3): `T __shfl_down_sync(unsigned mask, T value, unsigned delta, int width = warpSize);`. The default width is the built-in `warpSize`, not the literal 32 the book gives; on every NVIDIA device it is 32 (ch05 A5).

What the book leaves out, all from guide section 5.4.6.5 and 5.4.6.6:

- Top lanes: "the ID number of the source lane will not wrap around the value of width and so the upper delta lanes will effectively remain unchanged." So in Fig. 10.13 at stride 16, lanes 16 to 31 read their own value and add it to itself. Those lanes' sums are wrong but never feed lane 0, which is why the book can say only lane 0's return value is meaningful (our reading of the dependency, consistent with the quoted rule).
- Inactive or exited lanes: "Threads may only read data from another thread that is actively participating in the intrinsics. If the target thread is inactive , the retrieved value is undefined." And for the mask: "Each non-calling thread must have its corresponding bit set to zero in the mask . Exited threads are ignored." A reduction for a length that is not a multiple of 32, where out-of-range threads `return` early, therefore reads undefined values in the last warp. The fix is to keep every lane in the shuffle and give out-of-range lanes the identity (0 for a sum, -infinity for a max). This is the planted defect for the non-power-of-two bug hunt.
- Width: "width must be a power of two in the range [1, warpSize] , namely 1, 2, 4, 8, 16, or 32. Other values will produce undefined results." "If width is less than warpSize , then each subsection of the warp behaves as a separate entity with a starting logical lane ID of 0." This is the book's "four independent groups of eight" (p. 241), as documented.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (sections 5.4.6.5 and 5.4.6.6)

### A4. The other warp-level families named on p. 241: vote, match, reduce

**Verdict: CONFIRMED.** All three families exist under these names; the book's "warp reduce functions ... for integer values" is exact.

- Vote: `int __all_sync(unsigned mask, int predicate);`, `int __any_sync(unsigned mask, int predicate);`, `unsigned __ballot_sync(unsigned mask, int predicate);` "The warp vote functions enable the threads of a given warp to perform a reduction-and-broadcast operation."
- Match: `unsigned __match_any_sync(unsigned mask, T value);`, `unsigned __match_all_sync(unsigned mask, T value, int *pred);` with "T can be int , unsigned , long , unsigned long , long long , unsigned long long , float or double ." No compute-capability restriction is printed in the current section.
- Reduce: "Supported by devices of compute capability 8.x or higher." `T __reduce_add_sync(unsigned mask, T value);`, `__reduce_min_sync`, `__reduce_max_sync` ("T can be an unsigned or signed integer"), and `__reduce_and_sync`, `__reduce_or_sync`, `__reduce_xor_sync` on `unsigned`.
- All three: "These intrinsics do not provide any memory ordering."
- NVIDIA's hints on the same pages: "It is suggested to use the CUB Warp-Wide “Collective” Primitives to perform a Warp Reduction whenever possible", and `cuda::device::warp_shuffle()` / `cuda::device::warp_match_all()` from libcu++ as "generalized and safer" alternatives (infra B2: libhipcxx does not provide the warp functions).

Consequence for the chapter: there is no hardware warp reduction for `float` on CUDA. A float sum or max across a warp is the shuffle tree of Fig. 10.13 (or a library that wraps it).

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (sections 5.4.6.2 to 5.4.6.5)

### A5. Older warp-synchronous reductions (no `_sync`, no barrier in the last warp) are invalid today

**Verdict: CONFIRMED.** Not in the book, which uses only `_sync` intrinsics, but the reader will meet the older pattern (a last warp unrolled over `volatile` shared memory with no barrier) in code and tutorials. NVIDIA names this case explicitly.

Quote (guide section 3.2.2.1.1 "Independent Thread Scheduling"): "Independent thread scheduling can break code that relies on implicit warp-synchronous behavior from previous GPU architectures. Warp-synchronous code assumes that threads in the same warp execute in lockstep at every instruction, but the ability for threads to diverge and reconverge at sub-warp granularity makes such assumptions invalid. This can lead to a different set of threads participating in the executed code than intended. Any warp-synchronous code developed for GPUs prior to CC 7.0 (such as synchronization-free intra-warp reductions) should be revisited to ensure compatibility. Developers should explicitly synchronize such code using __syncwarp() to ensure correct behavior across all GPU generations."

The oldest compute capability in the current guide's tables is 7.5 (ch04 A10), so on every documented NVIDIA device that pattern is wrong. On AMD, lanes are documented to run in lockstep (ch04 B5), which is why such code may appear to work there; the portable rule is NVIDIA's.

- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html (section 3.2.2.1.1)

### A6. `cuda::atomic_ref<float, cuda::thread_scope_device>` and `fetch_add(..., cuda::memory_order_relaxed)` for the multi-block commit (Fig. 10.18)

**Verdict: CONFIRMED.** API, header and memory-order spelling are settled in infra A2. Three details for the lab:

- The constructor takes a reference: `_CCCL_HOST_DEVICE_API explicit constexpr atomic_ref(_Tp& __ref)` (CCCL `main`). The book's `output_ref(*output)` is correct. The current guide's own example is not: section 2.3.5.1 prints `__global__ void sumReduction(int n, float *array, float *result)` and then `cuda::atomic_ref<float, cuda::thread_scope_device> result_ref(result);`, passing a `float*` where a `float&` is required. By the header above that line should not compile (our reading; not compiled). The same example uses the default order (`seq_cst`) and one atomic per thread, which is the "naive one-thread-per-element atomic version" the notes suggest measuring against.
- Default order: `seq_cst`, as in the standard (infra A2). The book's explicit `memory_order_relaxed` is the right choice for a commutative accumulation.
- Denormals: "Atomic single-precision floating-point adds on global memory always operate in flush-to-zero mode, namely behave equivalent to PTX add.rn.ftz.f32 semantic." and "Atomic single-precision floating-point adds on shared memory always operate with denormal support". So the atomic commit can differ from a CPU reference in the subnormal range regardless of `-ftz`. Irrelevant at logit magnitudes, but a test tolerance must not be relative-only near zero.

- https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/__atomic/atomic.h (lines 88 to 105)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.5.1)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html (section 5.6.3, "Atomic Functions Denormals Behavior")

### A7. An atomic maximum on `float` (exercise 4; multi-block max for softmax; argmax)

**Verdict: PARTLY.** The book says the atomic reference class offers a maximum (p. 208, in chapter 9) without naming the method. The method is `fetch_max`. Whether a reader's toolkit accepts it for `float` could not be established from documentation.

- Legacy route: no `atomicMax(float*, float)` on CUDA (infra A1: `atomicMax` is "int, unsigned, unsigned long long, long long."). NVIDIA's documented fallback is a CAS loop through an integer reinterpretation (infra A1).
- libcu++ docs: `T cuda::atomic<T, Scope>::fetch_max(T const& val, cuda::std::memory_order order = cuda::std::memory_order_seq_cst);` "Atomically find the maximum of the value stored in the cuda::atomic and val . The maximum is found using cuda::std::max ." The only example is `cuda::atomic<int>`; no list of supported `T` is given. `cuda::atomic_ref` lists `fetch_max` among its additions.
- CCCL source on `main` does accept floating point: `__cuda_atomic_deduce_minmax_impl` sets `constexpr bool __is_floating = is_floating_point_v<_Type> || __is_extended_floating_point_v<_Type>;` and selects a floating-point operand tag for `fetch_min` / `fetch_max` (`cuda/std/__atomic/functions/backend.h`, lines 304 to 326). The generic path compares with `__cuda_atomic_less(__old, __op) ? __op : __old`. Whether `float` lowers to a hardware instruction or a CAS loop was not traced.
- COULD NOT VERIFY: the first CUDA Toolkit or CCCL release in which `cuda::atomic_ref<float>::fetch_max` compiles. A reader on an older toolkit may get a compile error; the lab should fall back to the CAS loop or to the two-pass scheme.
- For argmax, a single-word atomic needs the (value, index) pair packed into 64 bits with an order-preserving float-to-integer transform, then `atomicMax(unsigned long long*, ...)` (infra A1 lists `unsigned long long`). The packing is our technique, not a vendor recipe.

- https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/atomic/fetch_max.html
- https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html
- https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/std/__atomic/functions/backend.h
- https://github.com/NVIDIA/cccl/blob/main/libcudacxx/include/cuda/std/__atomic/functions/generic_rmw.h (lines 275 to 291)

### A8. "There is no good barrier across blocks" (p. 244)

**Verdict: PARTLY.** True for an ordinary `<<<>>>` launch. CUDA does have a grid-wide barrier, `this_grid().sync()`, but only under a cooperative launch whose grid fits on the device at once (ch04 A7, infra A4). Clusters give a barrier over up to 8 blocks on compute capability 9.0 and later (ch05 A6).

This matters for the chapter's LLM case because llama.cpp uses exactly this for vocabulary-sized softmax rows (D1): partial maxima and sums written per block, `g.sync()`, then every block reduces the partial array. It is the book's "partial-sum array plus second kernel" (pp. 245 to 246) with the kernel boundary replaced by a grid barrier. The book's statement is the right default to teach; the cooperative route is the exception with a residency precondition.

- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/softmax.cu (lines 144 to 215, 413 to 447)

### A9. Floating-point addition is not associative (p. 224); `std::reduce` versus `std::accumulate` (p. 223)

**Verdict: CONFIRMED.** NVIDIA's guide now has its own worked example, which a lesson can cite next to the book's Appendix A.

- Guide section 5.6.1.4 "Associativity": three single-precision values A, B, C for which "the results corresponding to the sums rn(rn(A + B) + C) and rn(A + rn(B + C)) differ from each other" and "This example shows that seemingly identical computations can produce different results, even when all basic operations comply with IEEE-754."
- Section 5.6.1.6 "Dot Product Example" contrasts a sequential sum with a "divide-and-conquer strategy ... This algorithm is called the “parallel algorithm” ... it can be implemented with a single thread." Its parenthesisation `((a1*b1) + (a2*b2)) + ((a3*b3) + (a4*b4))` is the reduction tree of Fig. 10.3.
- Section 5.6.1.8: "Floating-point addition and multiplication in finite precision are not associative ... The order in which these operations are evaluated affects how rounding errors accumulate and can significantly alter the final result." and "The presence of differences does not necessarily mean the GPU’s result is incorrect".
- cppreference, `std::reduce` (header `<numeric>`, C++17): "std::reduce behaves like std::accumulate except the elements of the range may be grouped and rearranged in arbitrary order." and "If add_op is not associative or not commutative (such as floating-point addition), the result is non-deterministic." That is the book's two requirements stated by the C++ standard library.

One inconsistency inside NVIDIA's section 5.6, for the record: 5.6.3 says "Note that floating-point expression re-associations and contractions are allowed by default, similarly to the --fmad=true option", while 5.6.1.8 says "The highest optimization level for GCC ( -O3 ), Clang ( -O3 ), nvcc ( -O3 ) ... does not affect floating-point semantics." We did not resolve which reading of "re-associations" is meant. Not needed for the chapter: the reduction kernels reorder additions explicitly.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html (sections 5.6.1.4, 5.6.1.6, 5.6.1.8, 5.6.3)
- https://en.cppreference.com/w/cpp/algorithm/reduce

### A10. Max with NaN is not commutative (footnote 3, p. 225): what `fmaxf`, `max` and `std::max` do

**Verdict: CONFIRMED**, and the footnote depends on which max is used. CUDA's `fmaxf` and its `max(float, float)` drop NaN, which makes them commutative on NaN inputs; a comparison-based max (`a > b ? a : b`, or `std::max`) does not.

- `fmaxf`: "Treats NaN arguments as missing data. If one argument is a NaN and the other is legitimate numeric value, the numeric value is chosen." "If both arguments are NaN, returns NaN."
- `__device__ float max(const float a, const float b)`: "Behavior is equivalent to fmaxf() function. Note, this is different from std: : specification".
- So the same reduction gives different results on NaN input depending on the combine expression, and a CPU reference written with `std::max` will not match a kernel written with `fmaxf` on rows containing NaN. The M4 tests need a stated NaN policy (D2 to D4 show three different policies in real libraries).

- https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html (`fmaxf`, `max`)

### A11. `expf`, `__expf` and `--use_fast_math` (for the M4 softmax)

**Verdict: CONFIRMED.**

- `expf`: maximum error "2 ULP" (guide section 5.6.7.2, "Exponential Functions"). Special values: "expf( −∞ ) returns +0." "expf(NaN) returns NaN." "This function is affected by the use_fast_math compiler flag."
- `__expf`: "Calculate the fast approximation to the base e exponential"; maximum error "2 + ⌊|1.173 · x|⌋" ULP (section 5.6.9.2, "Single-Precision-Only Intrinsic Functions").
- "The flag --use_fast_math automatically translates the corresponding CUDA Math API functions into intrinsic functions." Table 66 (section 5.6.9.3) lists `expf(x)` to `__expf(x)`.
- "To maximize the portability of results, users are recommended to use the default settings of the nvcc compiler’s floating-point options: -ftz=false , -prec-div=true , and -prec-sqrt=true , and not use the --use_fast_math option."
- We searched section 5.6 for "finite" and "NaN": none of the listed effects of `--use_fast_math` is an assumption that values are finite. Contrast Metal's fast mode (C3).

For softmax, `exp(x - max)` has a non-positive argument, so `exp(-inf) = 0` is the property that lets masked logits carry `-infinity`. The error bound of `__expf` grows with |x|; the lab tolerance should be set with that in mind, or the lab should use `expf`.

- https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html (`expf`)
- https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html (`__expf`)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html (sections 5.6.3, 5.6.7.2, 5.6.9.2, 5.6.9.3)

### A12. Thrust and CUB (pp. 223, 249; references [2] and [3], p. 250)

**Verdict: PARTLY.** Both libraries are current, now inside CCCL (infra A11). The two URLs the book prints no longer resolve.

- `https://nvidia.github.io/cccl/thrust/index.html` and `https://nvidia.github.io/cccl/cub/index.html` return HTTP 404 (also without `index.html`). The site root `https://nvidia.github.io/cccl/` redirects to `unstable/`, and `https://nvidia.github.io/cccl/unstable/thrust/index.html` and `https://nvidia.github.io/cccl/unstable/cub/index.html` return 200. This is the same move infra A2 found for the libcu++ pages.
- Use the `unstable/` URLs in lessons, or link the repository `https://github.com/NVIDIA/cccl`.

- https://nvidia.github.io/cccl/unstable/thrust/index.html
- https://nvidia.github.io/cccl/unstable/cub/index.html

### A13. What `cub::DeviceReduce` guarantees today: determinism, `ArgMax` ties, NaN in `Max`

**Verdict: CONFIRMED.** These three answers are the "use the library in practice" baseline (p. 249) and they bear directly on the chapter's reproducibility and argmax points.

- Determinism: "cub::DeviceReduce supports all three determinism guarantees ; the default is run_to_run ." "run_to_run (the default) is reproducible because, for a given GPU, every launch with the same input, build, and launch configuration selects the same tuning policy and therefore performs the same fixed reduction tree ... no atomics or other run-dependent ordering are involved. Because floating-point addition is only pseudo-associative, that fixed combining order is what makes the result bitwise-identical from one run to the next." It "can change on a different GPU architecture"; `gpu_to_gpu` for `float`/`double` with `cuda::std::plus` uses "a Reproducible Floating-point Accumulator (RFA)". `not_guaranteed` "uses an atomic accumulation kernel ... Atomics combine partial results in whatever order the hardware schedules them, which can differ between runs". Requested with `cuda::execution::require(cuda::execution::determinism::run_to_run)` (or `gpu_to_gpu`, `not_guaranteed`) as the `env` argument.
- So the book's two multi-block designs map onto CUB's two modes: the atomic commit of Fig. 10.18 is `not_guaranteed`; the partial array reduced by a second pass is `run_to_run`.
- ArgMax: "Finds the first device-wide maximum based on a given comparison operator and also returns the index of that item." The form with separate value and index outputs (`d_max_out`, `d_index_out`, an `int64_t` index) is marked "Added in version 2.2.0: First appears in CUDA Toolkit 12.3."; the form that also takes a comparison operator is "Added in version 3.4.0: First appears in CUDA Toolkit 13.4."; the older `cub::KeyValuePair<int, T>` form still exists. "Does not support > operators that are non-commutative." ArgMax always uses the run-to-run path. NaN behaviour of ArgMax is not stated.
- Max: "For floating-point types, the maximum follows cuda::std::fmax : NaN inputs are ignored. If all inputs are NaN, the result is the initial value."
- Coarsening appears as a tuning knob: the docs' example policy sets `.threads_per_block = 256` and `.items_per_thread = cc > cuda::compute_capability{9, 0} ? 20 : 16`. That is an example of a custom tuning, not CUB's default, but it is the book's `COARSE_FACTOR` (§10.10) under another name.

- https://nvidia.github.io/cccl/unstable/cub/api/structcub_1_1DeviceReduce.html

---

## Group B: HIP / ROCm

### B1. Does the book's `warp_reduce` (Fig. 10.13) and its kernels compile and work under HIP unchanged?

**Verdict: WRONG.** Three separate problems, each documented:

1. The mask. `__shfl_down_sync(0xffffffff, partialSum, stride)` does not compile: every HIP `_sync` function has `static_assert(... sizeof(MaskT) == 8, "The mask must be a 64-bit integer. Implicitly promoting a smaller integer is almost always an error.");` (infra B3). The mask needs a type that is 32 bits for CUDA and 64 bits for HIP, as llama.cpp does with `ggml_cuda_thread_mask_t` (ch05 B9) and AMD's own example does with `lane_mask_t` (B2).
2. The width. `WARP_SIZE` is 32 in the book. On gfx9 (Instinct) a wavefront is 64 lanes (ch05 B5). With strides 16, 8, 4, 2, 1 and HIP's out-of-range rule ("If the resulting source lane is out of range, the thread returns its own var", infra B3), lane 0 ends with the sum of lanes 0 to 31 only and lane 32 with the sum of lanes 32 to 63 (our arithmetic from the quoted rule). The kernel then behaves as two logical 32-lane warps per wavefront. That is correct only if the rest of the kernel also counts warps of 32 (llama.cpp's choice, D1), and AMD notes it "only utilizes half of the compute resources" (ch05 B5).
3. The warp barrier. `__syncwarp()` with no argument compiles; `__syncwarp(0xFFFFFFFF)` does not (ch04 B4).

Also relevant: HIP's `__reduce_add_sync` covers `int` and `unsigned int` unless `HIP_ENABLE_EXTRA_WARP_SYNC_TYPES` is defined (infra B3), so a float warp sum is the shuffle tree on HIP too.

For the portability lesson: this is the concrete case CLAUDE.md asks for. Same names, different mask type, different width; one `.cu` file needs a typed mask and a width that is not a literal 32.

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_warp_sync_functions.h
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (sections "warpSize" and "Warp cross-lane functions")

### B2. A compile-time warp width for `partialSums_s[BLOCK_DIM/WARP_SIZE]` (Figs. 10.16, 10.18), and AMD's own reduction example

**Verdict: CONFIRMED.** `warpSize` is not a compile-time constant on HIP today, and AMD's documentation uses a block reduction, structured exactly like the book's Fig. 10.14, as its example of how to handle that.

Quotes (HIP language extensions, section "warpSize", source at `therock-10.2`):

- "Since warpSize can differ between devices, it can not be assumed to be a compile-time constant on the host. It has to be queried using hipDeviceGetAttribute or hipDeviceGetProperties".
- "Prior to ROCm 7.0, the warpSize parameter was a compile-time constant. Starting with ROCm 7.0, it is early folded by the compiler, allowing it to be used in loop bounds and enabling loop unrolling in a manner similar to a compile-time constant warp size."
- "If compile time warp size is required, for example to select the correct mask type or code path at compile time, the recommended approach is to determine the warp size of the GPU on host side and setup the kernel accordingly, as shown in the following block reduce example."
- "The block_reduce kernel has a template parameter for warp size and performs a reduction operation in two main phases: Shared memory reduction: Reduction is performed iteratively, halving the number of active threads each step until only a warp remains (32 or 64 threads, depending on the device). Warp-level reduction: Once the shared memory reduction completes, the remaining threads use warp-level shuffling to sum the remaining values."
- For a device-side macro: `#if defined(__GFX8__) || defined(__GFX9__)` / `#define WarpSize 64` / `#else` / `#define WarpSize 32`, with the note "mwavefrontsize64 compiler option is not supported by HIP runtime, that's why the architecture based compile time selector is an acceptable approach."

The example kernel (`template_warp_size_reduction.hip` at the tag) declares `template<std::uint32_t WarpSize> using lane_mask_t = typename std::conditional<WarpSize == 32, std::uint32_t, std::uint64_t>::type;`, loops `for (std::uint32_t i = blockDim.x / 2; i >= WarpSize; i /= 2)` with `__syncthreads()`, then `for (std::uint32_t i = WarpSize/2; i >= 1; i /= 2) result = result + __shfl_down(result, i);`, and the host picks `block_reduce<32>` or `block_reduce<64>` from `hipDeviceAttributeWarpSize`. Note it uses `__shfl_down` (no mask), which HIP has and the book's CUDA source does not use; it is a HIP-only example.

Consequences for the two-stage kernel (our arithmetic):

- With 64-wide warps a 1024-thread block has 16 warps, so stage 2 has 16 partial sums, not 32. Fig. 10.16's `partialSums_s[t]` read by every lane of warp 0 then reads past the array (64 lanes, 16 entries). Every project kernel read for this file guards that read and pads with the identity (D1, D3, D4); the lab must too.
- Sizing the shared array for the worst case works on both vendors: `BLOCK_DIM / 32` entries (32 is the smallest width any of the three backends documents or assumes). PyTorch states the same rule: statically sized arrays "_must_ use either C10_WARP_SIZE_UPPER_BOUND or C10_WARP_SIZE_LOWER_BOUND, as needed" (ch04 D4).

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (section "warpSize")
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/docs/how-to/hip_cpp_language_extensions.rst (lines 332 to 480)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/docs/tools/example_codes/template_warp_size_reduction.hip

### B3. Atomics for the multi-block commit and for max on HIP

**Verdict: CONFIRMED.** Everything the chapter needs exists; two spellings are HIP-only.

- `atomicAdd(float*, float)`: same call; safe (possibly CAS) by default, hardware instruction with `-munsafe-fp-atomics` (infra B1).
- `atomicMax(float*, float)`: exists on HIP and not on CUDA (infra B1). Convenient, not portable back.
- `cuda::atomic_ref` through libhipcxx (ROCm 10.1 and later, link `libhipcxx::libhipcxx`; infra B2). At `therock-10.2`, `fetch_add` on device scope is `__hip_atomic_fetch_add(__ptr, __val, __memorder, __HIP_MEMORY_SCOPE_AGENT)`, and `fetch_max` for floating point is a compare-and-swap loop: `template<class _Type, class _Delta, class _Scope, typename _CUDA_VSTD::enable_if<sizeof(_Type)<=2 || _CUDA_VSTD::is_floating_point<_Type>::value, int>::type = 0> _Type __atomic_fetch_max_cuda(...)` with `while(__desired == __val && !__atomic_compare_exchange_cuda(...))`. So the book's Fig. 10.18 commit and an exercise-4 `fetch_max` both have a HIP path with the CUDA spelling.
- 64-bit `atomicMax(unsigned long long*, ...)` for a packed (value, index) key: same call (infra B1, `INTEGER` includes `unsigned long long`).

- https://github.com/ROCm/libhipcxx/blob/therock-10.2/include/cuda/std/__atomic/functions/atomic_hip_derived.h (lines 86 to 110)
- https://github.com/ROCm/libhipcxx/blob/therock-10.2/include/cuda/std/__atomic/functions/atomic_hip_generated.h (lines 84 to 86, 147 to 155)

### B4. `fmaxf`, `expf`, `__expf` on HIP

**Verdict: PARTLY.** The functions exist with CUDA's names. AMD's page gives measured error against the C++ standard library, not bounds, and says nothing about NaN.

- HIP math API: "Error bounds on this page are measured in units in the last place (ULPs), representing the absolute difference between a HIP math function result and its corresponding C++ standard library function". Rows: `float fmaxf(float x, float y)` "Determine the maximum numeric value of x and y", tested on x in [-10, 10], y in [-3, 3], difference 0; `float expf(float x)` on [-10, 10], 1; `float __expf(float x)` "Returns the fast approximate for e^x" on [-10, 10], 6.
- NaN: not stated on the HIP page. In clang's HIP header `fmaxf` is `float fmaxf(float __x, float __y) { return __builtin_fmaxf(__x, __y); }` (LLVM `main`, `__clang_hip_math.h` line 424), i.e. C's `fmax`, which treats NaN as missing data like CUDA's. ROCm ships its own clang build; we read LLVM `main`, not the ROCm 10.2 copy.

- https://rocm.docs.amd.com/projects/HIP/en/latest/reference/math_api.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/docs/reference/math_api.rst
- https://github.com/llvm/llvm-project/blob/main/clang/lib/Headers/__clang_hip_math.h

### B5. hipCUB `DeviceReduce::ArgMax` and the tie-break of its `ArgMax` functor

**Verdict: CONFIRMED.** `hipcub::DeviceReduce::ArgMax` exists in both CUB interfaces (the deprecated `KeyValuePair` output and the newer separate value and index outputs; `device_reduce.hpp` at `therock-10.2`, lines 411 to 495). The functor:

```cpp
struct ArgMax {
    template<class Key, class Value>
    HIPCUB_HOST_DEVICE inline constexpr KeyValuePair<Key, Value>
    operator()(const KeyValuePair<Key, Value>& a, const KeyValuePair<Key, Value>& b) const
    {
        return ((b.value > a.value) || ((a.value == b.value) && (b.key < a.key))) ? b : a;
    }
};
```

Ties go to the lower index, as in CUB's "first device-wide maximum" (A13). With a NaN in either operand both comparisons are false and the function returns `a`, so the result depends on argument order (our reading of the code). That is footnote 3 of p. 225 in a shipping library: the combine is not commutative on NaN.

- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipcub/hipcub/include/hipcub/backend/rocprim/device/device_reduce.hpp
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hipcub/hipcub/include/hipcub/backend/rocprim/thread/thread_operators.hpp

---

## Group C: Metal

### C1. The warp-level reduce on Metal: `simd_shuffle_down`, `simd_sum`, `simd_max`

**Verdict: PARTLY.** The functions are confirmed (infra C2, C5); Metal has a one-call float reduction where CUDA has none (A4). Two things the reduction lessons would like to state are not documented: the order in which `simd_sum` adds, and what `simd_max` does with NaN.

- Shuffle: `T simd_shuffle_down(T data, ushort delta)`, no mask and no width; it "doesn't modify the upper delta lanes of data because it doesn't wrap values around the SIMD-group" (infra C2). Same top-lane behaviour as CUDA (A3) and HIP (B1).
- The nearest thing to the book's width parameter is `simd_shuffle_and_fill_down(T data, T filling_data, ushort delta, ushort modulo)` with modulo "2, 4, 8, 16, or 32", which needs Apple8 ("SIMD shift and fill | Metal 3 & 4 | Apple8", infra C2), so not M1-series. The mapping is ours.
- Built-in reductions (MSL spec section 6.10.2, page 221): `T simd_max(T data)` "Returns data with the highest value from across all active threads in the SIMD-group and broadcasts that value to all active threads in the SIMD-group." `T simd_sum(T data)` "Returns the sum of the input values in data across all active threads in the SIMD-group and broadcasts the result to all active threads in the SIMD-group." T includes floating types (infra C2). Available "macOS: Metal 2.1 and later"; "SIMD-scoped reduction operations | Metal 3 & 4 | Apple7".
- Differences from the book's `warp_reduce`: the result reaches every active lane, not only lane 0; and no lane index or width appears in the source.
- COULD NOT VERIFY: the summation order of `simd_sum` (so it may not match a CPU tree in the last bits), and NaN handling in `simd_max`. Neither is stated in section 6.10.2.
- COULD NOT VERIFY: behaviour of `simd_shuffle_down` or `simd_sum` in a SIMD-group that is only partly filled (the last group of a nonuniform threadgroup "may be smaller", ch04 C2). Safe practice, ours: dispatch whole threadgroups whose size is a multiple of the SIMD width and pad out-of-range lanes with the identity, which is what llama.cpp and MLX do (D2, D3).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 6.10.2, pages 217 to 226)

### C2. Committing across threadgroups on Metal: float add yes, float max no, the argmax pair needs 64-bit atomics or a second pass

**Verdict: CONFIRMED** from infra C1, with the consequence for this chapter's labs spelled out.

- Sum (Fig. 10.18): `atomic_fetch_add_explicit` on a `device atomic_float*` with `memory_order_relaxed`, Metal 3 and later, Apple7 and later (infra C1). Every Apple silicon Mac qualifies (the feature tables map M1 to Apple7, infra Group C intro).
- Max on float: `atomic_fetch_max_explicit` is listed for `atomic_int` and `atomic_uint` only (infra C1). A float max needs a compare-and-swap loop on `atomic_float` (compare-exchange supports `atomic_float` in device memory, infra C1) or a second pass. The CAS loop is our construction.
- Argmax pair: 64-bit atomic max exists only as `void atomic_max_explicit(device atomic_ulong*, ...)`, Apple9 and later (Apple8 on macOS only), and returns nothing (infra C1). On M1 (Apple7) there is no 64-bit atomic. The portable Metal route is two dispatches: per-threadgroup partial (value, index) pairs, then one threadgroup reduces them. llama.cpp's Metal backend does exactly this (`kernel_argmax_part_f32` then `kernel_argmax_final_f32`, D2).
- Grid-wide barrier: none (ch04 C3), so the second phase is a second dispatch, the book's "launch a second, single-block kernel" (p. 245).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 2.6 and 6.16)
- https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf

### C3. Identity values under Metal's fast math: `-INFINITY` and NaN

**Verdict: PARTLY.** The hazard is documented; whether it bites with the lab runtime's options is not.

- MSL spec section 1.6.3: fast mode lets the compiler assume "No NaNs" and "No INFs: Allow optimizations to assume the arguments and result are not positive or negative infinity." "If you set the option to relaxed, it lets the compiler make aggressive, potentially lossy assumptions about floating-point math, but honors INFs and NaNs." Section 8.1: "If fast math is enabled the behavior of handling NaN or INF (as inputs or outputs) is undefined." Section 8.4: "fast math enabled (which is the default unless you specify -fno-fast-math as a compiler option)" (for the command-line compiler).
- So a max reduction whose identity is `-INFINITY` (the book's identity for max, p. 222), or a softmax over masked logits set to `-INFINITY`, is undefined under fast mode. Under relaxed or safe mode it is defined.
- Which mode a runtime-compiled library gets by default is documented three different ways (ch05 C7). The project's lab runtime compiles with a default `MTLCompileOptions` and sets only `preprocessorMacros` (`labs/common/src/backend_metal.mm`, lines 41 to 46; a project fact, not vendor documentation).
- A kernel can set its own mode: "#pragma METAL fp math_mode([relaxed | safe | fast])", which "can only appear in file or namespace scope, within a language linkage specification, or at the start of a compound statement". The surrounding sentence spells it "#pragma metal fp"; we did not test which spelling the compiler accepts.
- Safe statements for the labs, ours: either use `-FLT_MAX` (finite) or the first element as the max identity on Metal, or put the pragma in `kernel.metal` and test that it compiles; and never rely on NaN propagation under default options. llama.cpp's Metal kernels use `-INFINITY` with default compile options (D2) and MLX compiles its metallib with `-fno-fast-math` (ch05 C10); that llama.cpp works in practice is not documentation.
- `fmax` and `max` (section 6.6, page 206): "Returns y if x < y, otherwise returns x. If one argument is a NaN, fmax() returns the other argument. If both arguments are NaNs, fmax() returns a NaN." Same NaN-as-missing rule as CUDA (A10), when NaN is honoured at all.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 1.6.3 pages 15 to 16, 6.6 page 206, 8.1 page 368, 8.4 pages 368 to 371)

### C4. `exp` accuracy on Metal, and `fast::exp`

**Verdict: CONFIRMED.**

- Table 8.1 (single precision, precise): `exp` "<= 4 ulp".
- Table 8.2 (single precision, fast math): `exp(x)` "<= 3 + floor(fabs(2 * x)) ulp".
- `metal::precise::exp` and `metal::fast::exp` select a variant explicitly (section 6.6). MLX's softmax calls `fast::exp` with the comment "Softmax doesn't need high precision exponential cause x is gonna be in (-oo, 0] anyway and subsequently it will be divided by sum(exp(x_i))." (D3).

So on all three backends the fast exponential's error grows with |x|, and on all three softmax keeps x at or below 0. Test tolerances for M4 should be absolute plus relative and stated in the lab README.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 8.4, Tables 8.1 and 8.2, pages 369 to 371)

### C5. A library argmax on Apple GPUs: MPSGraph

**Verdict: PARTLY.** `MPSGraph.reductionArgMaximum(with:axis:name:)` exists ("Creates a reduction argMax operation and returns the result tensor."; macOS 12.0 and later). Its tie-break and NaN behaviour are not documented on the page. COULD NOT VERIFY those.

- https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph/reductionargmaximum(with:axis:name:)

---

## Group D: How llama.cpp, MLX and PyTorch do the same things

All files read from the default branch on 2026-11-20; last-commit dates from the GitHub commits API.

### D1. llama.cpp, CUDA and HIP: warp reduce, the two-stage block reduce, softmax

**Verdict: CONFIRMED.** `ggml/src/ggml-cuda/common.cuh` (last commit 2026-11-19) and `softmax.cu` (2026-11-14).

- Warp reduce is an xor butterfly, so every lane gets the result (NVIDIA's own guide example does the same, section 5.4.6.5 "Example 3: Reduction across a warp", `value += __shfl_xor_sync(0xFFFFFFFF, value, i);`):

```cpp
template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x = fmaxf(x, ggml_cuda_shfl_xor(x, offset, width));
    }
    return x;
}
```

  The `int` version of `warp_reduce_sum` uses the hardware reduction where it exists: `#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE` / `return __reduce_add_sync(GGML_CUDA_FULL_THREAD_MASK, x);` (A4: compute capability 8.x, integers).
- The mask is typed: `constexpr ggml_cuda_thread_mask_t GGML_CUDA_FULL_THREAD_MASK = (ggml_cuda_thread_mask_t)0xffffffffffffffffULL;` (32 bits on CUDA, 64 on HIP, ch05 B9).
- `ggml_cuda_syncwarp()` calls `__syncwarp()` only `#ifndef GGML_USE_HIP`; on HIP it is empty, relying on AMD's lockstep (ch04 B5).
- `block_reduce` is the book's two-stage kernel (Fig. 10.16), with the read of the per-warp array guarded and padded with the identity, which the book's listing lacks:

```cpp
    val = block_reduce_policy<reduce_method_t, T>::reduce(val);
    ...
        if (lane_id == 0) { shared_vals[warp_id] = val; }
        __syncthreads();
        val = block_reduce_policy<reduce_method_t, T>::sentinel();
        if (lane_id < (static_cast<int>(block_size) / WARP_SIZE)) { val = shared_vals[lane_id]; }
        return block_reduce_policy<reduce_method_t, T>::reduce(val);
```

  The sentinels are `0.0f` for SUM and `-INFINITY` for MAX.
- Softmax row (`soft_max_f32`): `max_val = block_reduce<block_reduce_method::MAX, ...>(max_val, buf_iw);`, then `expf(vals[col] - max_val)`, then a barrier with the comment "sync is needed as we reuse buf_iw across block_reduce invocations, see #26385", then `block_reduce<block_reduce_method::SUM, ...>`. That barrier between two reductions sharing one buffer is a real-world instance of the race the M4 softmax and the bug hunt can plant.
- Rows too long for one block's shared memory (`nbytes_shared > smpbo`, the comment reads "Parallelize across SMs for top-p/dist-sampling") take a cooperative path when the device supports cooperative launch, `ncols_x / (ne01 * ne02 * ne03) > 8192`, and there is no mask, no sinks, scale 1 and max_bias 0: per-block max into `tmp_maxs[blockIdx.x]`, `g.sync()`, reduce the partials, then the same for the sums (A8). Includes are split by vendor exactly as CLAUDE.md prescribes: `#ifdef GGML_USE_HIP` / `#include <hip/hip_cooperative_groups.h>` / `#else` / `#include <cooperative_groups.h>`.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/common.cuh (lines 45, 130 to 136, 437 to 452, 515 to 535, 588 to 595, 644 to 726)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/softmax.cu (lines 1 to 10, 85 to 128, 144 to 215, 413 to 447)

### D2. llama.cpp argmax on CUDA and Metal, and the greedy sampler: three tie-breaks

**Verdict: CONFIRMED** as source read; the tie and NaN analysis below is our reading, not a test.

- CUDA `argmax_f32` (`ggml-cuda/argmax.cu`, last commit 2026-11-15): one block per row, initial `float maxval = -FLT_MAX; int argmax = -1;`, a strided loop keeping a value only `if (val > maxval)`, then an xor butterfly that swaps in a partner's pair only `if (val > maxval)`, then a guarded second stage over per-warp pairs in shared memory. On ties each lane keeps its own pair, so the answer for tied maxima depends on which lane held which index. Example by hand, a 32-column row (one column per thread, one warp): equal maxima at columns 1 and 2 give 2, because lane 0 takes lane 2's pair at offset 2 and keeps it against lane 1's equal value at offset 1. Not "first index". A row whose values are all `-INFINITY` returns -1 (`-inf > -FLT_MAX` is false). NaN is skipped (`NaN > x` is false).
- Metal `kernel_argmax_f32` (`ggml-metal/kernels/misc.metal`, last commit 2026-10-27): `float max_val = simd_max(lmax); int32_t arg_val = simd_max(select(-1, larg, lmax == max_val));`. Among lanes holding the maximum it takes the largest index. Equal maxima at columns 0 and 1 of a 32-column row give 1 here and 0 on CUDA. Long rows use two dispatches, `kernel_argmax_part_f32` and `kernel_argmax_final_f32` (C2).
- Host greedy sampler (`src/llama-sampler.cpp`, last commit 2026-11-12): `if (cur_p->data[i].logit > cur_p->data[cur_p->selected].logit) cur_p->selected = i;`, i.e. the lowest index wins ties. The experimental backend greedy sampler builds `ggml_argmax(ctx, logits)` (named `"greedy_argmax"`), which runs the device kernels above.

So in one project the same greedy step can pick different tokens on exact ties depending on host, CUDA or Metal. Exact ties in float logits are rare, but this is the reason M4 must define its tie-break and test it with planted ties.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/argmax.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/misc.metal (lines 3 to 60, 63 to 132, 134 onward)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/reduce.metal (`kernel_op_sum_f32`, lines 3 to 50: `simd_sum`, one `threadgroup_barrier`, `simd_sum` again over `tpitg.x < nsg ? shmem_f32[tpitg.x] : 0`)
- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-sampler.cpp (lines 1054 to 1097)

### D3. MLX: a commutative argmax pair operator, and the two-stage softmax

**Verdict: CONFIRMED.**

- `mlx/backend/metal/kernels/arg_reduce.metal` (last commit 2026-08-28) defines `struct IndexValPair { uint32_t index; U val; };` and:

```cpp
template <typename U>
struct ArgMax {
  static constexpr constant U init = Limits<U>::min;
  IndexValPair<U> reduce(IndexValPair<U> best, IndexValPair<U> current) thread {
    if ((isnan(current.val) && (!isnan(best.val) || best.index > current.index)) ||
        (!isnan(best.val) && (best.val < current.val ||
          (best.val == current.val && best.index > current.index)))) {
      return current;
    } else {
      return best;
    }
  }
  ...
};
```

  NaN beats any number, ties (and NaN against NaN) go to the lower index, so the combine is commutative and associative as a pair operator, which is what the shuffle tree and any reordering need (pp. 232, 245). `Limits<float>::min` is `-metal::numeric_limits<type>::infinity()` (`kernels/utils.h`). The pair is shuffled field by field: `simd_shuffle_down(data.index, delta), simd_shuffle_down(data.val, delta)`. Stage 2 uses `threadgroup IndexValPair<T> local_data[32];` and `if (simd_lane_id < simd_groups) best = local_data[simd_lane_id];` with `simd_groups = ceildiv(lsize.x, simd_size)`.
- `kernels/softmax.h` (last commit 2025-05-21), `softmax_single_row`: pads out-of-range reads with `Limits<AccT>::min`, `maxval = simd_max(maxval);`, lane 0 of each SIMD-group writes `local_max[simd_group_id]`, barrier, SIMD-group 0 reduces with `simd_max` again, broadcast through `local_max[0]`; then the same with `simd_sum` for the normaliser, with `softmax_exp` = `fast::exp` (C4). `constexpr int SIMD_SIZE = 32;` is hard-coded here (ch04 D3).

- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/arg_reduce.metal (lines 16 to 76, 126 to 170)
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/softmax.h (lines 1 to 95)
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/utils.h

### D4. PyTorch: the argmax comparator, `BlockReduceSum`, the warp-per-row softmax, and the documented tie rule

**Verdict: CONFIRMED.**

- Documentation, `torch.argmax` (2.15): "If there are multiple maximal values then the indices of the first maximal value are returned." NaN is not mentioned on the page.
- Implementation, `aten/src/ATen/native/SharedReduceOps.h` (last commit 2026-11-18), used by `ArgMaxOps`:

```cpp
template <typename scalar_t>
struct GreaterOrNan {
  C10_DEVICE bool operator () (scalar_t a, scalar_t b, int64_t idx_a, int64_t idx_b) const {
    // If (a == b), then choose the one with lower idx, else max(a, b)
    if (at::_isnan(a)) {
      if (at::_isnan(b)) {
        return idx_a < idx_b;
      }
      return true;
    }
    return (a == b) ? idx_a < idx_b : (a > b);
  }
};
```

  Same policy as MLX (D3): NaN wins, lowest index on ties. The CUB policy differs (A13: `Max` ignores NaN; ArgMax NaN unstated), as does llama.cpp's (D2).
- `aten/src/ATen/native/cuda/block_reduce.cuh` (last commit 2025-11-28), `BlockReduceSum`: warp reduce, then `__syncthreads(); // prevent races when BlockReduces are called in a row.`, lane 0 writes `shared[wid]`, `__syncthreads()`, then the guarded read `val = (tid < B::Warps()) ? shared[lid] : T(0);` and a second warp reduce. The max version pads with `std::numeric_limits<T>::lowest()`. Two fixes to the book's Fig. 10.16 in one function: the guard, and a leading barrier for back-to-back reductions (the softmax case, D1).
- `aten/src/ATen/native/cuda/PersistentSoftmax.cuh` (last commit 2026-11-03): "Each sample contains element_count scalar elements. Forward supports values up to 2048"; "One "WARP" works on one "BATCH". One "BATCH" contains "WARP_BATCH" samples. WARP_BATCH is equal to 1 when element_count is large, and > 1 when element_count is small."; "this means only __shfl_ instructions are required for reductions." This is the notes' "short attention rows get one warp each, several rows per warp when shorter" in production code.

- https://docs.pytorch.org/docs/2.15/generated/torch.argmax.html
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/SharedReduceOps.h (lines 420 to 450, 493 to 510)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/block_reduce.cuh (lines 1 to 147)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/PersistentSoftmax.cuh (lines 40 to 151)

---

## Summary of corrections

Every WRONG or PARTLY item, and every failed or unfound sub-claim:

- A2: `__syncwarp()` is documented as a barrier with memory ordering for the lanes in its mask; NVIDIA guarantees convergence at `_sync` calls, not that lanes stay converged after one. Do not teach "it makes diverged threads reconverge" as a guarantee.
- A3: the shuffle width defaults to `warpSize`, not the literal 32 (p. 241). Lanes whose source is past the end keep their own value; reading an inactive or exited lane is undefined, so threads must not `return` before the shuffles.
- A6: the current CUDA guide's own `cuda::atomic_ref` example (section 2.3.5.1) passes a `float*` to a constructor that takes `float&`; the book's `output_ref(*output)` is the correct form. Float atomic adds on global memory flush denormals to zero regardless of `-ftz`.
- A7: the method the book does not name is `fetch_max`; current CCCL source accepts `float` for it, but the docs list no types and the first toolkit that accepts it could not be found. There is still no legacy `atomicMax(float*)` on CUDA.
- A8: a grid-wide barrier does exist (`grid.sync()` under a cooperative launch, with a residency limit), and llama.cpp uses it for vocabulary-sized softmax; "no good barrier across blocks" holds for ordinary launches.
- A9 (side note): NVIDIA's section 5.6 says both that re-associations are "allowed by default" and that `-O3` "does not affect floating-point semantics"; not resolved.
- A12: the book's reference URLs for Thrust and CUB (`nvidia.github.io/cccl/thrust/index.html`, `.../cub/index.html`) return 404; the live pages are under `nvidia.github.io/cccl/unstable/`.
- B1 (WRONG): the book's `warp_reduce` does not compile under HIP (`0xffffffff` mask rejected by a `static_assert`), and with `WARP_SIZE` 32 on a 64-wide wavefront it reduces each 32-lane half separately.
- B2: `warpSize` is not a compile-time constant on HIP from ROCm 7.0; AMD recommends a warp-size template parameter chosen on the host, or an architecture macro; on 64-wide hardware a 1024-thread block has 16 warps, so Fig. 10.16's unguarded `partialSums_s[t]` reads past the array.
- B4: HIP documents measured ULP differences against the C++ library, not error bounds, and says nothing about NaN for `fmaxf` (clang implements it as `__builtin_fmaxf`).
- C1: the summation order of `simd_sum`, NaN handling in `simd_max`, and SIMD-group functions in a partially filled SIMD-group are not documented.
- C3: under Metal fast math the behaviour of INF and NaN is undefined, so `-INFINITY` as the max identity is unsafe unless the kernel runs in relaxed or safe mode; the lab runtime compiles with default options, whose math mode Apple documents inconsistently (ch05 C7). The pragma's spelling differs within the spec ("metal" in prose, "METAL" in the code line).
- C5: MPSGraph's argmax tie-break and NaN behaviour are not documented.
- D2 (our reading): llama.cpp's CUDA argmax, Metal argmax and host greedy sampler resolve exact ties three different ways, and the device kernels return -1 for an all-`-INFINITY` row.

Path and status facts worth knowing even where the verdict is CONFIRMED:

- A1: the memory model the book cites is now guide section 5.8 "CUDA C++ Memory Model" (`05-appendices/cuda-cpp-memory-model.html`).
- A5: NVIDIA explicitly calls "synchronization-free intra-warp reductions" written for pre-7.0 GPUs incorrect today.
- A9: NVIDIA's guide has its own associativity example (5.6.1.4) and a "parallel algorithm" dot product (5.6.1.6) to cite beside Appendix A.
- A13: `cub::DeviceReduce` is run-to-run deterministic by default (fixed tree, no atomics), optionally GPU-to-GPU deterministic for float sums, and atomic and non-deterministic only on request; ArgMax returns the first maximum; Max ignores NaN.
- D1, D3, D4: every production two-stage reduction read for this file guards the stage-2 read and pads with the identity, and the projects disagree on the NaN policy of argmax (MLX and PyTorch: NaN wins; llama.cpp: NaN skipped; CUB: unstated).

Not checked: the 2010 World Cup sidebar facts (pp. 226 to 227) and reference [1] (Bell and Hoberock); any behaviour on real hardware.

## Where the backends really differ

One line per cell, for the operations this chapter uses. "Same" means the same source text compiles through `pmpp/vendors/hip.h`, according to the documentation; nothing here was run on CUDA or AMD hardware.

| Feature | CUDA | HIP (ROCm 10.2) | Metal (MSL 4.1) |
|---|---|---|---|
| Warp width a kernel may assume | 32 on every listed device; `warpSize` "commonly 32" | 64 on gfx9, 32 on gfx10 and later; not a compile-time constant since ROCm 7.0; template parameter or `__GFX9__` macro | Undocumented; `[[threads_per_simdgroup]]` at run time |
| Full-warp mask in `_sync` calls | `0xffffffff` (`unsigned`) | Must be a 64-bit integer; the 32-bit literal fails a `static_assert` | No masks |
| Shuffle down | `__shfl_down_sync(mask, v, d, width = warpSize)` | Same name with 64-bit mask; also `__shfl_down(v, d)` without a mask | `simd_shuffle_down(v, d)`; no width (`simd_shuffle_and_fill_down` with `modulo`, Apple8 and later) |
| Lane whose source is past the end | Keeps its own value | Returns its own value | Upper lanes unchanged |
| Built-in float warp sum or max | None (`__reduce_*_sync` is integer only, compute capability 8.x) | None by default (integer; float with `HIP_ENABLE_EXTRA_WARP_SYNC_TYPES`) | `simd_sum`, `simd_max` on float, Apple7 and later; order of additions undocumented |
| Which lanes hold the warp result | Lane 0 with shuffle-down; all lanes with xor | Same | All active lanes (`simd_sum`, `simd_max`) |
| Warp-level barrier | `__syncwarp()` | `__syncwarp()`; masked form needs 64-bit; lanes documented lockstep | `simdgroup_barrier(mem_flags::mem_threadgroup)` |
| Block barrier as memory fence | `__syncthreads()`, "strongly happens before" | `__syncthreads()`, "implicitly include a threadfence" | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| Cross-block float sum commit | `atomicAdd(float*)` or `cuda::atomic_ref<float>::fetch_add`; global adds flush denormals | Same calls; CAS-based unless `-munsafe-fp-atomics`; `atomic_ref` needs libhipcxx | `atomic_fetch_add_explicit` on `device atomic_float*`, Metal 3, Apple7 |
| Cross-block float max commit | No legacy call; `atomic_ref<float>::fetch_max` in current CCCL (toolkit version unknown); or CAS loop | `atomicMax(float*)` (HIP only); libhipcxx `fetch_max` is a CAS loop | None for float; CAS loop on `atomic_float`, or a second pass |
| Cross-block (value, index) argmax commit | Packed key with `atomicMax(unsigned long long*)`, or a second pass | Same | 64-bit max only Apple9 (Apple8 on macOS), returns void; otherwise a second pass |
| Grid-wide barrier | `grid.sync()` under `cudaLaunchCooperativeKernel`, grid must be resident | `grid.sync()` under `hipLaunchCooperativeKernel`; different header | None; use a second dispatch |
| Float max and NaN | `fmaxf` and `max(float, float)` drop NaN; `std::max` differs | `fmaxf` is `__builtin_fmaxf` (C semantics); NaN not in AMD's docs | `fmax` and `max` drop NaN, when NaN is honoured |
| `-INFINITY` as max identity | Defined; `--use_fast_math` lists no finite-math assumption | Not checked for `-ffast-math`; defined by default | Undefined under fast math; use relaxed or safe mode or a finite identity |
| Fast exponential | `__expf`, error 2 + floor(abs(1.173 x)) ULP; `expf` 2 ULP | `__expf` measured 6 ULP, `expf` 1 ULP against C++ on [-10, 10] | `fast::exp` <= 3 + floor(abs(2 x)) ULP; precise `exp` <= 4 ULP |
| Library reduction and argmax | `cub::DeviceReduce` (run-to-run deterministic by default; ArgMax returns first maximum) | `hipcub::DeviceReduce`; `ArgMax` functor ties to lower index, NaN order-dependent | MPSGraph `reductionArgMaximum`; ties undocumented |
