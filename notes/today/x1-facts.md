# Module X1 fact check (sampling kernels): CUDA, HIP, Metal, and the real samplers

Checked on 2026-11-20 against the live vendor and project sources listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY. Where an item has several sub-claims, the verdict is for the item as a whole and the text says which sub-claim failed. A verdict judges the claim in the item title (the assumption a course author would most likely bring).
- **Quote** is copied from the source (HTML docs flattened to text, GitHub raw files on the default branch or at a tag, Apple's documentation JSON, PDF text, local SDK headers, NVIDIA's own PyPI wheel). Line breaks inside quotes are removed.
- "ch14 A7", "infra C2" and so on point at items in `notes/today/ch14-facts.md`, `infra-facts.md`, etc. Anything settled there is cited, not re-checked.
- **There is no `notes/x1.md`.** Module X1 is ours: PLAN.md section 2 records "Sampling is not in the book at all. No top-k, top-p or argmax." So part (a) of the brief (do the book's dated statements still hold) has nothing to check, and there is no "Unclear in the PDF" list. Every item below is about vendor APIs, the papers the projects cite, and the code of real projects. Lessons must label all of it as ours, and each claim about a codebase as "checked in the facts pass on 2026-11-20".
- Versions the sources reported about themselves on the check date: cuRAND Library Programming Guide "13.5" (docs.nvidia.com); cuRAND headers from NVIDIA's PyPI wheel `nvidia-curand` 10.4.5.25 (`nvidia/cu13/include/`); CUDA Math API "v13.5"; ROCm Core SDK 10.2.0 (rocRAND docs "5.0.0", hipRAND docs "3.4.0", HIPIFY docs for 10.2.0; headers at tag `therock-10.2` of `ROCm/rocm-libraries` and `ROCm/rocm-systems`); Metal Shading Language Specification "Version 4.1" dated 2026-06-04 (383 pages, downloaded again today); macOS 27.0 SDK headers on this machine; Random123 `main` (last commit 2021-11-30, latest release v1.14.0); llama.cpp `master`, PyTorch `main` and docs 2.15, Hugging Face transformers `main`, MLX `main` and docs 0.32.4, mlx-lm `main`, vLLM `main`, FlashInfer `main`, all read today. Last-commit dates are given per file.
- Not checked: anything by running code on a GPU. Where this file says what a piece of code does on a given input (which token wins, which bits come out), that is our reading of the source, labelled as such. Group E is our own arithmetic, run in Python on the host.
- Licence trap: cuRAND's headers carry NVIDIA's notice "The source code and/or documentation ("Licensed Deliverables") are subject to NVIDIA intellectual property rights ... reproduction or disclosure of the Licensed Deliverables to any third party without the express written consent of NVIDIA is prohibited." This file therefore describes the header in words and identifiers only. Lessons that show Philox code should show Random123's (BSD-style licence, A4) or our own, never cuRAND's.

Totals: 20 CONFIRMED, 2 WRONG, 5 PARTLY, 0 COULD NOT VERIFY (27 items: 6 CUDA, 3 HIP, 6 Metal, 9 project, 3 arithmetic). Sub-claims that could not be verified are called out inside A1, B1, C2, D6 and collected under "Summary of corrections".

---

## Already settled elsewhere (cite, do not re-check)

| X1 needs | Settled in |
|---|---|
| Warp, wavefront, SIMD-group width; never a literal 32; 64-bit masks on HIP | ch05 B5, infra B3, infra C5, ch04 C2 |
| Shuffles, ballots, `simd_sum`, `simd_prefix_exclusive_sum`, `simd_ballot` | infra A3, B3, C2 |
| Atomics for token counts (penalties): `cuda::atomic_ref`, legacy `atomicAdd`, Metal `atomic_fetch_add_explicit` on `device atomic_uint*` | infra A1, A2, B1, B2, C1; ch09 A1, C1 |
| Vocabulary sizes 50,257 / 151,936 / 201,088 and why a vocabulary-sized histogram does not fit on chip | ch09 D7 |
| How llama.cpp, mlx-lm and vLLM keep penalty counts (host map; `.at[].subtract`; GPU bins) | ch09 D4, D5, D6 |
| Argmax tie and NaN rules: CUDA, Metal and host in llama.cpp; MLX; PyTorch (`torch.argmax` returns the first maximum); CUB | ch10 D2, D3, D4, A13 |
| `expf`, `__expf` accuracy; Metal `exp` / `fast::exp`; HIP `expf` | ch10 A11, C4, B4 |
| Metal default math mode documented three ways; not IEEE-strict by default | ch05 C7 |
| `-INFINITY` as a mask value under Metal fast math | ch10 C3 |
| Floating-point addition is not associative; NVIDIA's own example | ch10 A9 |
| Library scans (`cub::DeviceScan`, hipCUB, MPSGraph `cumulativeSum`) and their run-to-run determinism | ch11 A4, A8, B2, C6 |
| Top-p in llama.cpp (host and backend), mlx-lm, transformers, vLLM: all keep the token that crosses p; tie rules differ | ch11 D2, D3, D4 |
| Inter-block look-back is unsafe on Metal; one threadgroup per row or several dispatches | ch11 C5, ch12 C4, ch14 C4 |
| Stable filters (`cub::DeviceSelect`), hipCUB over rocPRIM, MPSGraph `nonZeroIndices` | ch12 A8, B5, C5 |
| llama.cpp host min-p is a stable filter on `max_logit + logf(p)`; GPU graph samplers mask with minus infinity | ch12 D3, D6 |
| `cub::DeviceRadixSort`, segmented sorts, stability, the float key transform | ch14 A2 to A5, E1 |
| Top-k without a full sort: `cub::DeviceTopK` (AIR TopK, unordered only), `DeviceBatchedTopK`; rocPRIM `topk`; MPSGraph `topK`, `MPSMatrixFindTopK` (k at most 16) | ch14 A7, B4, C1 |
| llama.cpp's radix-select top-k on CUDA, HIP and Metal (8-bit digits, ties to the lowest columns) | ch14 D2, ch12 D1, D2 |
| PyTorch `torch.topk` (radix select, 2-bit single block, 8-bit multi-block), `torch.sort` | ch14 D5 |
| MLX `topk` runs a full merge sort on Metal today | ch14 D4 |
| vLLM's default top-k/top-p is a pivot search with no sort (Qrita) | ch14 D7 |
| `__float_as_uint` (CUDA, HIP), `as_type<uint>` (Metal) | ch14 A8, B6, C3 |
| llama.cpp samples on the host by default; the backend path is experimental | ch01 D2 (refined in D1 and D3 below) |

---

## Group A: CUDA

### A1. cuRAND's device API offers Philox4_32_10 with `curand_init(seed, subsequence, offset, &state)`; the same seed gives the same sequence

**Verdict: PARTLY.** The API, the state type and the determinism statement are confirmed. Two details do not hold as a reader might assume: the documented period and subsequence length are in different units from the header's, and the docs do not promise how many raw values a distribution function consumes.

Docs (cuRAND Library Programming Guide 13.5, "Device API Overview", section "Bit Generation with Philox_4x32_10 generator"):

- Signatures as printed: `__device__ unsigned int curand(curandStatePhilox4_32_10_t *state)` and `__device__ void curand_init(unsigned long long seed, unsigned long long subsequence, unsigned long long offset, curandStatePhilox4_32_10_t *state)`.
- "Following a call to curand_init(), curand() returns a sequence of pseudorandom numbers with a period \(2^{128}\). If curand() is called with the same initial state each time, and the state is not modified between the calls to curand(), the same sequence is always generated."
- "The curand_init() function sets up an initial state allocated by the caller using the given seed, subsequence and offset. Different seed is guaranteed to produce different starting states and different sequences. Subsequence and offset together define offset in a sequence with period \(2^{128}\). Offset defines offset in subsequence of length \(2^{64}\)." "The same seed always produces the same state and the same sequence."
- Advice: "For the highest quality parallel pseudorandom number generation, each experiment should be assigned a unique seed value. Within an experiment, each thread of computation should be assigned a unique id number."
- Four-at-a-time variants: `curand4`, `curand_uniform4`, `curand_normal4`, ... `curand_uniform2_double` "generate four single precision or two double precision results with each call. Because the underlying implementation uses the Philox generator, this is generally more efficient than generating a single result with each call."
- Performance notes: "Calls to curand_init() are slower than calls to curand() or curand_uniform(). Large offsets to curand_init() take more time than smaller offsets. It is much faster to save and restore random generator state than to recalculate the starting state repeatedly."
- Distributions: "Distribution functions may use any number of unsigned integer values from a basic generator. The number of values consumed is not guaranteed to be fixed."

Header (`curand_kernel.h`, wheel `nvidia-curand` 10.4.5.25), doc comment on the Philox `curand_init`: each of the 2^64 seeds "selects an independent sequence of length 2^130", and "subsequences are of length 2^66". On `skipahead_sequence`: "Each subsequence is 2^66 elements long".

Sub-claims that do not hold as written:

- Units: the docs give a period of 2^128 and subsequences of 2^64; the header gives 2^130 values and subsequences of 2^66 values. Both are consistent if the docs count 128-bit counter steps and the header counts 32-bit values (four per step), but the docs do not say so. Teach "a subsequence is 2^64 counter values, 2^66 32-bit numbers".
- "Large offsets take more time" is a general note. For Philox the header's `skipahead` is a constant-time addition to the counter (our reading of the header), which is why initialising one state per row per step is cheap with this generator.
- Because the number of raw values a distribution consumes is not guaranteed, a lab that needs the same draw everywhere should take raw bits (`curand`, or its own Philox, A4) and convert them itself (E2), not rely on `curand_uniform` consuming exactly one value.

- https://docs.nvidia.com/cuda/curand/device-api-overview.html
- `curand_kernel.h` in https://pypi.org/project/nvidia-curand/ (version 10.4.5.25, file `nvidia/cu13/include/curand_kernel.h`, Philox section near lines 880 to 1040)

### A2. `curand_uniform` returns a float in (0, 1]: 0 excluded, 1 included

**Verdict: CONFIRMED.**

- Docs: "This function returns a sequence of pseudorandom floats uniformly distributed between 0.0 and 1.0. It may return from 0.0 to 1.0, where 1.0 is included and 0.0 is excluded."
- Header doc comment on the Philox overload: "Output range excludes 0.0f but includes 1.0f. Denormalized floating point outputs are never returned." `curand_uniform4` on Philox carries the same sentence.
- How (header, described): the 32-bit value x becomes `x * CURAND_2POW32_INV + (CURAND_2POW32_INV/2.0f)` in single precision, with `#define CURAND_2POW32_INV (2.3283064e-10f)` in `curand_globals.h`, which is exactly 2^-32 in float (E2). So the result is x·2^-32 + 2^-33, rounded to float.
- Our arithmetic (E2): 128 of the 2^32 inputs (x ≥ 0xFFFFFF80) round to exactly 1.0. A draw u = 1.0 must be handled by the inverse-CDF search; FlashInfer's kernel has a fallback for exactly that (A6), and PyTorch's binary search has one too (D5: "this actually can happen when the random number is 1").

Consequence for the course (ours): the boundary rule of the final draw depends on the range. With u in (0, 1] and the rule "first token whose inclusive cumulative probability is ≥ u", a zero-probability token can never be chosen at the start of the row, but u = 1.0 can fall past a total that rounds below 1. With u in [0, 1) (E2's conversion) and the rule "first token whose inclusive sum is > u", zero-probability tokens are never chosen and u never reaches the total. M5 should state one rule and test both ends.

- https://docs.nvidia.com/cuda/curand/device-api-overview.html (section "Distributions")
- `curand_uniform.h` and `curand_globals.h` in https://pypi.org/project/nvidia-curand/ (version 10.4.5.25)

### A3. What `curand_init` does with (seed, subsequence, offset), and how to reproduce one draw without cuRAND

**Verdict: CONFIRMED** as header read; the mapping below is our derivation from `curand_kernel.h` and `curand_philox4x32_x.h` (described, not copied: see the licence note at the top).

- The state holds a 128-bit counter `ctr` (four 32-bit words x, y, z, w), a 64-bit key, a four-word `output` buffer and a position `STATE` from 0 to 3.
- `curand_init(seed, subsequence, offset, &s)`: key.x = low 32 bits of seed, key.y = high 32 bits; counter zeroed; `skipahead_sequence(subsequence)` adds the subsequence to the counter's z and w words; `skipahead(offset)` sets `STATE = offset mod 4` and adds `offset / 4` to the x and y words; then `output` = Philox4x32-10(counter, key).
- `curand(&s)` returns `output[STATE]`, and after the fourth word increments the counter and recomputes `output`.
- So, by our derivation: the first `curand()` after `curand_init(seed, row, 4 * step, &s)` is word 0 of Philox4x32-10(counter = (step_lo, step_hi, row_lo, row_hi), key = (seed_lo, seed_hi)), and the next three calls return words 1 to 3. One Philox call per (row, step) gives four 32-bit numbers. A hand-written Philox (A4) with that counter layout reproduces cuRAND's raw bits without linking cuRAND; this is unverified on a device until a `reports/verify-*.md` from a CUDA machine shows it.
- The round function multiplies with `mulhilo32`, which uses `__umulhi(a, b)` on the device and a 64-bit product on the host (A5).
- `curand_philox4x32_x.h` also carries D. E. Shaw Research's copyright notice ("Copyright 2010-2011, D. E. Shaw Research"), and its constants are Random123's (A4).

PyTorch's CUDA kernels use exactly this pattern: `curand_init(seed, idx, offset, &state)` with `idx = blockIdx.x * blockDim.x + threadIdx.x` (`aten/src/ATen/native/cuda/DistributionTemplates.h`, last commit 2026-11-05), and so does FlashInfer, with the row as the subsequence (A6).

- `curand_kernel.h`, `curand_philox4x32_x.h` in https://pypi.org/project/nvidia-curand/ (version 10.4.5.25)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/DistributionTemplates.h (lines 63 to 80)

### A4. Philox4x32-10 is a published, counter-based generator with known-answer test vectors that any backend can be checked against

**Verdict: CONFIRMED.**

- Random123 README: "The Random123 library is a collection of counter-based random number generators ("CBRNGs") for CPUs (C and C++) and GPUs (CUDA and OpenCL), as described in Parallel Random Numbers: As Easy as 1, 2, 3, Salmon, Moraes, Dror & Shaw, SC11, Seattle, Washington, USA, 2011, ACM." "They ... have passed all of the rigorous SmallCrush, Crush and BigCrush tests in the extensive TestU01 suite". "They are **not** suitable for use in cryptography or security". "The Random123 library is implemented entirely in header files."
- What counter-based means, same README: "counter-based RNGs are *stateless* functions ... whose arguments are a *counter*, and a *key* that return a result of the same type as the counter. result = CBRNGname(counter, key)". "a unique (counter, key) tuple will always produce the same result."
- Definition, `include/Random123/philox.h`: `PHILOX_M4x32_0 ((uint32_t)0xD2511F53)`, `PHILOX_M4x32_1 ((uint32_t)0xCD9E8D57)`, `PHILOX_W32_0 ((uint32_t)0x9E3779B9)` ("golden ratio" in the 64-bit comment), `PHILOX_W32_1 ((uint32_t)0xBB67AE85)`; `PHILOX4x32_DEFAULT_ROUNDS 10`. One round: `lo0 = mulhilo(PHILOX_M4x32_0, ctr.v[0], &hi0)`, `lo1 = mulhilo(PHILOX_M4x32_1, ctr.v[2], &hi1)`, output `{hi1^ctr.v[1]^key.v[0], lo1, hi0^ctr.v[3]^key.v[1], lo0}`. Between rounds the key is bumped: `key.v[0] += PHILOX_W32_0; key.v[1] += PHILOX_W32_1;` (no bump before the first round).
- Known-answer vectors, `tests/kat_vectors` (format `name R CTR KEY EXPECTED`):
  - `philox4x32 10 00000000 00000000 00000000 00000000 00000000 00000000   6627e8d5 e169c58d bc57ac4c 9b00dbd8`
  - `philox4x32 10 ffffffff ffffffff ffffffff ffffffff ffffffff ffffffff   408f276d 41c83b0e a20bc7c6 6d5451fd`
  - `philox4x32 10 243f6a88 85a308d3 13198a2e 03707344 a4093822 299f31d0   d16cfe09 94fdcceb 5001e420 24126ea1`
  - plus three 7-round vectors. Our Python transcription reproduces all six (E1).
- Metal: Random123 ships `tests/kat_metal.m` and `tests/kat_metal_kernel.metal`; `features/metalfeatures.h` maps the 32-bit high multiply to `#define R123_MULHILO32_MULHI_INTRIN metal::mulhi`; the README's test list includes "MacOS, with Xcode-10.1 and Metal on a 2018 Mac mini" and credits "Support for Metal (Apple's successor to OpenCL)". So the same Philox has been run as MSL by its authors (an old toolchain; not re-run by us).
- Licence: BSD-style three-clause (the copyright block in each file). Safe to show in lessons with the notice.

For the lab (ours): these KATs are the portability test. A Philox written once in the `.cu` and once in the `.metal` file passes on every backend exactly when it prints these words.

- https://github.com/DEShawResearch/random123/blob/main/README.md
- https://github.com/DEShawResearch/random123/blob/main/include/Random123/philox.h (lines 241 to 330)
- https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors (lines 24 to 29)
- https://github.com/DEShawResearch/random123/blob/main/tests/kat_metal_kernel.metal
- https://github.com/DEShawResearch/random123/blob/main/include/Random123/features/metalfeatures.h (line 94)

### A5. `__umulhi` gives the high 32 bits of a 32 × 32-bit product

**Verdict: CONFIRMED.**

- CUDA Math API v13.5, Integer Intrinsics: `__device__ __tile__ unsigned int __umulhi(unsigned int x, unsigned int y)`: "Calculate the most significant 32 bits of the 64-bit product x * y, where x and y are 32-bit unsigned integers." "Returns the most significant 32 bits of the product x * y."
- Same page family, Type Casting Intrinsics: `__device__ float __uint2float_rn(unsigned int x)`: "Convert the unsigned integer value x to a single-precision floating-point value in round-to-nearest-even mode."
- Portable alternative (ours): `(unsigned long long)a * b` and a shift, which is what cuRAND's and Random123's host paths do and what PyTorch's Metal Philox does with `ulong` (D6). It compiles on CUDA, HIP, the emulators and (as `ulong`) Metal without any intrinsic.

- https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html
- https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html

### A6. A batched GPU draw in production: one block per row, a counter-based uniform keyed by the row, an inverse-CDF search over a block scan

**Verdict: CONFIRMED.** FlashInfer is not in the brief's list of projects; it is added because vLLM calls it on CUDA (ch14 D7) and its kernel is milestone M5's final step as shipping code.

`include/flashinfer/sampling.cuh` (last commit 2026-11-05), `SamplingFromProbKernel`:

- One block per row: `const uint32_t bx = blockIdx.x`; seed and offset per row from tensors or scalars; `curand_init(philox_seed, bx, philox_offset, &state);` so the row is the subsequence (A3's layout); `float u = curand_uniform(&state);`.
- The row is walked in chunks of `BLOCK_THREADS * VEC_SIZE` with a running `aggregate`; a chunk is scanned only if `aggregate + aggregate_local > u`; then `BlockScan<float, ...>.InclusiveSum` and `greater_than_u[j] = (inclusive_cdf[j] + aggregate > u) && valid[j]`; the first lane where the flag switches wins (`BlockAdjacentDifference`, then `atomicMin(&(temp_storage->sampled_id), ...)`). Zero-probability tokens are excluded by the predicate `[](float x) { return x > 0; }`. So the rule is: the first token with positive probability whose inclusive cumulative sum is strictly greater than u (our reading).
- The u = 1.0 boundary (A2) is handled explicitly: "// NOTE(Zihao): this would happen when u is very close to 1 // and the sum of probabilities is smaller than u // In this case, we use the last valid index as the sampled id".
- The kernel takes probabilities, not logits: u is compared with an unnormalised running sum only if the row sums to 1. The Python wrapper `flashinfer/sampling.py` (last commit 2026-11-19) reserves seeds and offsets from a `torch.Generator` ("Reserve a four-aligned RNG range and return its seed and starting offset").

- https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/sampling.cuh (lines 795 to 866, and `DeviceSamplingFromProb` above them)
- https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/sampling.py (lines 56 to 83)

---

## Group B: HIP / ROCm

### B1. The HIP counterpart of cuRAND's Philox: rocRAND and hipRAND, same arguments, same stream?

**Verdict: PARTLY.** The device API exists with matching names and arguments, and HIPIFY maps the CUDA names. Whether the stream equals cuRAND's is contradicted between AMD's documentation and AMD's own headers.

- Names (headers at `therock-10.2`): rocRAND `typedef rocrand_device::philox4x32_10_engine rocrand_state_philox4x32_10;` and `void rocrand_init(const unsigned long long seed, const unsigned long long subsequence, const unsigned long long offset, rocrand_state_philox4x32_10* state)`, header `<rocrand/rocrand_kernel.h>`. hipRAND: `DEFINE_HIPRAND_STATE(hiprandStatePhilox4_32_10, rocrand_state_philox4x32_10)` and the template `hiprand_init(const unsigned long long seed, const unsigned long long subsequence, const unsigned long long offset, StateType* state)`, header `<hiprand/hiprand_kernel.h>`, which includes `hiprand_kernel_rocm.h` under `__HIP_PLATFORM_AMD__` and `hiprand_kernel_nvcc.h` otherwise.
- HIPIFY "CURAND API supported by HIP" (ROCm 10.2.0 docs): `curandStatePhilox4_32_10_t` → `hiprandStatePhilox4_32_10_t`; `curand_init` → `hiprand_init`; `curand` → `hiprand`; `curand_uniform` → `hiprand_uniform`; `curand_uniform4` → `hiprand_uniform4`. `curand_Philox4x32_10` (the raw function) is listed with no HIP name.
- hipRAND README: "The hipRAND interface is compatible with rocRAND and cuRAND-v2 APIs."
- AMD's compatibility table (rocRAND 5.0.0 docs, "cuRAND compatibility"): "The following table shows which rocRAND generators produce the exact same sequence as the equivalent NVIDIA CUDA cuRAND generator when using legacy ordering, given the same seed, number of dimensions, and offset." Row: "Philox 32x4-10 | No". And: "cuRAND is a closed-source library, which makes achieving compatibility difficult. Even when the same underlying algorithm is used, implementation details can differ; for example, how the initial state is initialized, the order in which threads generate and write output values, and how the seed is interpreted or applied."
- Against that, rocRAND's device engine (`rocrand_philox4x32_10.h`, constants `ROCRAND_PHILOX_M4x32_0 0xD2511F53U` etc., round comment "// Source: Random123") initialises exactly as A3 describes: key = seed (low, high), subsequence added to counter z and w, `offset / 4` to x and y with a carry, `substate = offset & 3`, "A subsequence consists of 2 ^ 66 random numbers." By our reading the raw 32-bit device streams of `curand_init` and `rocrand_init` are the same for the same three arguments. The AMD table speaks about host generators "when using legacy ordering", which also fixes how outputs are laid out across threads; that may be where the "No" comes from. Not resolved, and not run on any device: treat raw-stream equality as **unverified**.
- What certainly differs is the float conversion (B2).

For the course (ours): do not use either library in the M5 lab. A hand-written Philox (A4) with our own conversion (E2) is the only route that is the same source text on CUDA, HIP, the emulators and Metal, and its correctness is checkable by the KATs.

- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocrand/library/include/rocrand/rocrand_philox4x32_10.h (lines 62 to 89, 120 to 260, 358 to 380)
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hiprand/library/include/hiprand/hiprand_kernel.h (lines 109 to 113)
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hiprand/library/include/hiprand/hiprand_kernel_rocm.h (lines 52, 187 to 215)
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/hiprand/README.md
- https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CURAND_API_supported_by_HIP.html
- https://rocm.docs.amd.com/projects/rocRAND/en/latest/conceptual/curand-compatibility.html

### B2. `hiprand_uniform` returns the same float as `curand_uniform` for the same state

**Verdict: WRONG** (by the headers; our arithmetic in E2).

- rocRAND (`rocrand_uniform.h` at the tag): "For unsigned integer between 0 and UINT_MAX, returns value between 0.0f and 1.0f, excluding 0.0f and including 1.0f." `return ROCRAND_2POW32_INV + (v * ROCRAND_2POW32_INV);` with `#define ROCRAND_2POW32_INV (2.3283064e-10f)` in `rocrand_common.h`. That is 2^-32 + v·2^-32.
- cuRAND (A2): v·2^-32 + 2^-33.
- On AMD, `hiprandStatePhilox4_32_10` is the rocRAND state (B1), so `hiprand_uniform` uses rocRAND's conversion; on NVIDIA, hipRAND forwards to cuRAND. The same `.cu` source calling `hiprand_uniform` therefore produces different floats on the two platforms whenever the two formulas round differently.
- E2: for 1,000,000 random 32-bit inputs the two conversions give different floats in 3,903 cases (about 0.39 %), almost all for small inputs where the 2^-33 offset is not lost to rounding. Both return exactly 1.0 for the same 128 largest inputs.

Consequence (ours): an inverse-CDF draw whose u lands within that difference of a boundary picks different tokens on CUDA and AMD even from identical raw bits. The lab converts bits itself (E2).

- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocrand/library/include/rocrand/rocrand_uniform.h (lines 62 to 85, 231 to 245)
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocrand/library/include/rocrand/rocrand_common.h (line 26)

### B3. `__umulhi` on HIP

**Verdict: CONFIRMED.** Documented and implemented with the CUDA spelling, so a `.cu` Philox using it compiles unchanged through `pmpp/vendors/hip.h`.

- HIP math API page (ROCm 10.2.0): `unsigned int __umulhi(unsigned int x, unsigned int y)`: "Returns the most significant 32 bits of the product of the two 32-bit unsigned integers." Also `__mulhi`, `__mul64hi`.
- Header `amd_device_functions.h` at `therock-10.2`: `__device__ static inline unsigned int __umulhi(unsigned int x, unsigned int y) { return (unsigned int)(((__hip_uint64_t)x * (__hip_uint64_t)y) >> 32); }`, the 64-bit product form of A5.

- https://rocm.docs.amd.com/projects/HIP/en/latest/reference/math_api.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_device_functions.h (lines 239, 320 to 322)

---

## Group C: Metal

### C1. MSL has no random number function

**Verdict: CONFIRMED.** A reader coming from CUDA may expect a `curand` counterpart in the shading language; there is none.

- MSL spec 4.1, full text searched: "random" occurs once, in "Random access iterator to element_type" (a tensor iterator description); "seed" occurs nowhere; there is no random, noise or hash function in the standard-library chapter (section 6).
- The local SDK's `Metal.framework` headers contain no random API either (searched `random`, excluding "random access").
- So a Metal kernel that needs a random number per row must compute one itself (C4), or take it from the host or from an MPS / MPSGraph pass (C2, C3).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Version 4.1, 2026-06-04)

### C2. Apple's documented Philox: `MPSMatrixRandomPhilox`

**Verdict: PARTLY.** Apple documents a Philox generator, but as a host-encoded kernel that fills a whole matrix or vector, not as a function a shader can call, and the details needed for reproducibility are not stated.

- Header `MPSMatrixRandom.h` (macOS 27.0 SDK): "@class MPSMatrixRandomPhilox @discussion Generates random numbers using a counter based algorithm. For further details see: John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw. Parallel Random Numbers: As Easy as 1, 2, 3." Initialisers take `device`, `destinationDataType`, `seed: (NSUInteger) seed` and a `distributionDescriptor`; work is encoded with `encodeToCommandBuffer:destinationVector:` or `destinationMatrix:`.
- Availability (Apple documentation JSON): macOS 10.15, iOS 13.0, visionOS 1.0.
- Uniform range: `MPSMatrixRandomDistributionUniform`: "Generate uniformly distributed random floating point values in the interval [0, 1)." Descriptor: "Make a descriptor for a uniform distribution of floating point values in the range [minimum, maximum)."
- Sibling: `MPSMatrixRandomMTGP32`, "a Mersenne Twister algorithm suitable for GPU execution. It uses a period of 2**11214."
- Not documented (could not verify): the number of rounds, how the seed maps to key and counter, how 32-bit values become floats, and whether the stream matches cuRAND's or Random123's for any seed.

- https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixrandomphilox
- Local header: `MetalPerformanceShaders.framework/Frameworks/MPSMatrix.framework/Headers/MPSMatrixRandom.h` (macOS 27.0 SDK), lines 14 to 35, 113 to 260

### C3. MPSGraph's random ops: a Philox state tensor and uniform values in [0, 1)

**Verdict: CONFIRMED.**

- Header `MPSGraphRandomOps.h` (macOS 27.0 SDK), category available from macOS 11.0: `randomPhiloxStateTensorWithSeed:name:`: "Creates a tensor representing state using the Philox algorithm with given counter and key values. Generates random numbers using the Philox counter-based algorithm, for further details see: John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw. Parallel Random Numbers: As Easy as 1, 2, 3." "seed: Initial counter and key values will be generated using seed."
- Explicit state: `randomPhiloxStateTensorWithCounterLow:counterHigh:key:name:` ("counterLow: The value to initilaize lower 64 bits of counter to. Philox utilizes a 128 bit counter").
- Values: `randomUniformTensorWithShape:seed:name:` "Returns a tensor of provided shape of random uniform values in the range [0.0, 1.0). Uses the provided seed value to initalize state. No state is preserved, and all calls with equal seed yield an identical stream of random values." The state-tensor variants return the updated state "alongside the random values".
- Apple documentation JSON: `randomPhiloxStateTensor(withSeed:name:)` and `randomUniformTensor(withShape:seed:name:)` both macOS 11.0.
- As for C2: rounds, seed-to-key mapping and float conversion are not stated, so "equal seed, identical stream" is a promise within MPSGraph only.

- https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph/randomphiloxstatetensor(withseed:name:)
- https://developer.apple.com/documentation/metalperformanceshadersgraph/mpsgraph/randomuniformtensor(withshape:seed:name:)
- Local header: `MetalPerformanceShadersGraph.framework/Headers/MPSGraphRandomOps.h` (macOS 27.0 SDK), lines 108 to 300

### C4. Everything a hand-written Philox needs is in MSL

**Verdict: CONFIRMED.**

- MSL spec section 6.4, "Built-in integer functions": `T mulhi(T x, T y)`: "Computes x * y and returns the high half of the product of x and y." (no later-version note in the table). Also `madhi`. That is CUDA's `__umulhi` (A5).
- 64-bit integers for the product form: `long`/`ulong`, "All OS: Metal 2.2 and later" (ch14 C3).
- Conversion of a 32-bit value to float, section 2.23: "The static_cast operator converts from a scalar or vector type to another scalar or vector type using the default rounding mode with no saturation (when converting to floating-point, round ties to even; when converting to an integer, round toward zero)." Bit reinterpretation is `as_type` (ch14 C3).
- Precedent: Random123's Metal test kernel (A4) and PyTorch's `c10/metal/random.h` (D6) are Philox written in MSL.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 2.23 and 6.4, pages 110 and 202 to 203)

### C5. Metal guarantees IEEE round-to-nearest float arithmetic, so a softmax and scan computed on Metal match the CPU reference bit for bit

**Verdict: WRONG.** The spec permits round toward zero, on top of the default fast or relaxed math mode (ch05 C7).

- MSL spec section 8.2 "Rounding Mode": "Either round ties to even or round toward zero rounding mode may be supported for single-precision, half-precision, and brain floating-point operations."
- Section 8.4, Table 8.1 lists `x + y`, `x - y`, `x * y`, `1.0 / x`, `x / y` as "Correctly rounded" (in whichever mode is in force).
- From Metal 4.1 there is a compiler option for conversions only: "use the following option to change the default rounding mode for float-to-float conversions from RTNE (round to nearest, ties to even) to RTZ (round toward zero) for compatibility with non-Metal APIs: -fmetal-rtz-fp-conversion".
- Contraction is controllable: `#pragma METAL fp contract([off | on | fast])` and `#pragma STDC FP_CONTRACT OFF` (section 1.6.3 area).

Consequence (ours): exact token agreement between Metal and a CPU reference cannot rest on float equality of the cumulative probabilities. Either the test avoids draws that land within a small margin of a CDF boundary, or the comparison of u with the scan is made robust to it (for example by checking the CPU reference's margin and skipping or re-drawing those cases). The same holds across CUDA and HIP because `expf` differs by backend (ch10 A11, B4, C4) and summation order differs by kernel (ch10 A9).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 1.6.3, 8.2, 8.4; pages 16, 368)

### C6. Apple ships no categorical sampler

**Verdict: CONFIRMED.**

- The MPS and MPSGraph headers in the macOS 27.0 SDK contain "categorical" only in loss functions (`MPSCNNLoss.h`, `MPSCNNTypes.h`: categorical cross-entropy) and no "multinomial" (searched all `MetalPerformanceShadersGraph` and `MPS*` framework headers). Metal Performance Primitives headers (`MPPTensorOpsMatMul2d.h`, `MPPTensorOpsConvolution2d.h`) contain no random API.
- The pieces that do exist as graph ops: `topK` (macOS 14), `argSort` (macOS 13) with no stated stability (ch14 C1), `cumulativeSum` (ch11 C6), `reductionArgMaximum` with undocumented ties (ch10 C5), and the random ops of C3. A Metal sampler is written by hand (the M5 lab) or assembled from those ops.

- Local headers: `MetalPerformanceShadersGraph.framework/Headers/*.h`, `MetalPerformanceShaders.framework/Frameworks/*/Headers/*.h`, `MetalPerformancePrimitives.framework/Headers/*.h` (macOS 27.0 SDK)

---

## Group D: How llama.cpp, Hugging Face, PyTorch and MLX sample today

All read on the default branch today; dates are the last commit to the file. These are snapshots: in lessons, link to files and quote short lines, not line numbers.

### D1. llama.cpp's default sampler chain, its defaults, and where it runs

**Verdict: CONFIRMED.**

`common/common.h` (last commit 2026-11-20), `struct common_params_sampling`:

- Defaults as printed: `uint32_t seed = LLAMA_DEFAULT_SEED;`, `int32_t top_k = 40; // <= 0 to use vocab size`, `float top_p = 0.95f; // 1.0 = disabled`, `float min_p = 0.05f; // 0.0 = disabled`, `float temp = 0.80f; // <= 0.0 to sample greedily, 0.0 to not output probabilities`, `int32_t penalty_last_n = 64; // last n tokens to penalize (0 = disable penalty)`, `float penalty_repeat = 1.00f; // 1.0 = disabled`, `float penalty_freq = 0.00f; // 0.0 = disabled`, `float penalty_present = 0.00f; // 0.0 = disabled`.
- Default order: `COMMON_SAMPLER_TYPE_PENALTIES, COMMON_SAMPLER_TYPE_DRY, COMMON_SAMPLER_TYPE_TOP_N_SIGMA, COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TYPICAL_P, COMMON_SAMPLER_TYPE_TOP_P, COMMON_SAMPLER_TYPE_MIN_P, COMMON_SAMPLER_TYPE_XTC, COMMON_SAMPLER_TYPE_TEMPERATURE`. Temperature comes after every truncation.
- `bool backend_sampling = false;` for generation; the speculative-decoding draft has `bool backend_sampling = true; // offload draft sampling to the backend (default: on)`.
- `include/llama.h` (last commit 2026-11-17): `#define LLAMA_DEFAULT_SEED 0xFFFFFFFF` and "seed == LLAMA_DEFAULT_SEED to use a random seed."

`common/sampling.cpp` (last commit 2026-11-17): the chain starts with a logit-bias sampler when there are biases or suppressed tokens, adds the samplers in `params.samplers` order, then ends with `llama_sampler_init_dist(params.seed)`, or with `llama_sampler_init_greedy()` when the last sampler is temperature with `temp == 0.0f` (and no dynamic range) or top-k with `top_k == 1`, and no probabilities were requested. Backend sampling is switched off with a warning when a grammar or a reasoning budget is active: "backend sampling is not compatible with grammar, disabling".

`common/arg.cpp`: `{"-bs", "--backend-sampling"}`, "enable backend sampling (experimental) (default: disabled)", environment `LLAMA_ARG_BACKEND_SAMPLING`.

- https://github.com/ggml-org/llama.cpp/blob/master/common/common.h (lines 231 to 303, 351)
- https://github.com/ggml-org/llama.cpp/blob/master/common/sampling.cpp (lines 336 to 441)
- https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp (lines 2289 to 2295)
- https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h (lines 37, 1502)

### D2. llama.cpp temperature and repetition penalty

**Verdict: CONFIRMED.** `src/llama-sampler.cpp` (last commit 2026-11-12).

- Host temperature, `llama_sampler_temp_impl`: `if (temp <= 0.0f) { // find the token with the highest logit and set the rest to -inf` with a strict `>` scan from index 0 (so the lowest index wins ties, our reading); otherwise `cur_p->data[i].logit /= temp;`.
- Backend temperature, `llama_sampler_backend_temp_sampling`: for `temp <= 0.0f`, `ggml_argmax` and a gather of that one candidate; otherwise `data->logits = ggml_scale(ctx, data->logits, 1.0f / temp);`. Multiplying by the reciprocal can differ from dividing in the last bit (our note), one more reason host and backend can disagree.
- Repetition penalty, `llama_sampler_penalties_apply`, for each candidate whose token is in the last-`penalty_last_n` count map: "// The academic publication that described this technique actually just only divided, but that would cause tokens with negative logits to become more likely, which is obviously wrong. // This is common fix for this problem, which is to multiply by the penalty instead of dividing." Code: `if (cur_p->data[i].logit <= 0) { cur_p->data[i].logit *= ctx->penalty_repeat; } else { cur_p->data[i].logit /= ctx->penalty_repeat; }` followed by the frequency and presence terms (ch09 D4).
- The publication is CTRL (D9): it divides by θ regardless of sign.

- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-sampler.cpp (lines 266 to 290, 2042 to 2070, 2923 to 2952)

### D3. llama.cpp's final draw: an inverse-CDF search on the host, and a scan-and-count on the GPU with a host-drawn uniform

**Verdict: CONFIRMED** as source read; the boundary analysis is ours. `src/llama-sampler.cpp` (last commit 2026-11-12).

Host `dist` sampler (`llama_sampler_dist_apply`):

- Generator: `std::mt19937 rng;` seeded from `get_rng_seed(seed)`, which returns `std::random_device` output (or the system clock if `random_device` reports zero entropy) when `seed == LLAMA_DEFAULT_SEED`.
- Draw: `std::uniform_real_distribution<double> dist(0.0f, 1.0f);` then "// sample from the obtained probabilities and normalize the probs in a single pass // this is ~3x faster on Mac with full gpt-oss vocab than the version below": softmax numerators `expf(logit - max_l)` summed in `double sum_cum`, `const double sum_tgt = sum_cum*rnd;`, a running sum `sum_run += cur_p->data[i].p;` and `if (sum_run >= sum_tgt) { cur_p->selected = i; ...}`. So: the first candidate whose inclusive unnormalised sum is ≥ u × total, in candidate-array order (which earlier samplers may have sorted).
- A disabled alternative (`#else`) uses `std::discrete_distribution<int>`.

Backend `dist` (`llama_sampler_dist_backend_apply` and `_backend_set_input`), experimental:

- The random number is not generated on the GPU. One `float` input tensor per output, filled on the host: "// We sample in double precision and cast to float to match rnd numbers of // llama_sampler_dist which uses double precision", `const float rnd = dist(rng); ggml_backend_tensor_set(inp_uniform, &rnd, 0, sizeof(float));`.
- Graph: `ggml_soft_max` → `ggml_cumsum` (inclusive) → `ggml_sub(cumsum, inp_uniform)` → `ggml_step` (1 where the difference is `> 0`, ch11 D2) → `ggml_sum` (how many entries exceed u) → `ggml_clamp(ctx, idxf, 1.0f, mask->ne[0])` → index = n − count, then a gather back to vocabulary ids if a candidate list exists.
- So the backend picks the first entry whose inclusive cumulative probability is strictly greater than u; the host picks the first whose sum is ≥ u × total. On an exact tie of u with a boundary the two paths differ by one token (our reading).
- Our reading of an edge case: a double just below 1 can round to `1.0f` in the cast (any value at least 1 − 2^-25). Then no entry may exceed u, the count is 0, the clamp makes it 1, and the last candidate in the row is chosen, which can be a token masked to probability 0. The odds per draw are about 2^-25.
- Design point (ours): this is "inverse-CDF over a scan" built from graph ops, with the RNG kept on the host so that host and backend paths consume the same `mt19937` stream. M5 instead derives u on the device from (seed, row, step) with a counter-based generator, so no per-step upload is needed and every backend can reproduce it.

- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-sampler.cpp (lines 218 to 248, 336 to 352, 1124 to 1420)

### D4. Hugging Face transformers: the logits-processor order, the tie rules, and the draw

**Verdict: CONFIRMED.** `src/transformers/generation/utils.py` (last commit 2026-11-20), `logits_process.py` (2026-11-18), `configuration_utils.py` (2026-11-06).

- Order in `_get_logits_processor`: processors such as `RepetitionPenaltyLogitsProcessor` (when `repetition_penalty != 1.0`), no-repeat n-grams, bad words, min length, forced tokens, suppression; then "# Processors previously known as `LogitsWarpers`, only applied with sampling strategies": `TemperatureLogitsWarper` (when `temperature != 1.0`), `TopHLogitsWarper`, `TopKLogitsWarper` (when `top_k != 0`), `TopPLogitsWarper` (when `top_p < 1.0`), `MinPLogitsWarper` with the comment "# Applied after temperature scaling (see https://github.com/ggerganov/llama.cpp/pull/3841#issuecomment-2073826084)", then typical, epsilon, eta. Temperature comes before every truncation, the opposite of llama.cpp's default (D1) and mlx-lm (D8).
- Defaults when unset (`_get_default_generation_params`): `"temperature": 1.0`, `"top_k": 50`, `"top_p": 1.0`. Docs: top_p "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation."
- Temperature must be positive: "`temperature` (={temperature}) has to be a strictly positive float, otherwise your next token scores will be invalid." plus, for 0.0, " If you're looking for greedy decoding strategies, set `do_sample=False`." The warper computes `scores / self.temperature`.
- Repetition penalty: `score = torch.where(score < 0, score * self.penalty, score / self.penalty)` on every token present in `input_ids` (prompt included, each counted once), unless `prompt_ignore_length` is given.
- Top-k keeps ties: `indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]`. Every token equal to the k-th value survives, so more than k tokens can remain (our reading). llama.cpp's host top-k and mlx-lm's keep exactly k with unspecified ties (ch14 D3, ch12 D6).
- Min-p in probability space: `probs = torch.softmax(scores, dim=-1)`, `top_probs = probs.amax(dim=-1, keepdim=True)`, `scaled_min_p = self.min_p * top_probs`, `tokens_to_remove = probs < scaled_min_p`, then the top `min_tokens_to_keep` are restored. A token exactly at the threshold is kept.
- Top-p: ch11 D4.
- The draw, `_sample`: `probs = nn.functional.softmax(next_token_scores, dim=-1)` then `next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)` with the comment "# TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution"; greedy is `torch.argmax(next_token_scores, dim=-1)`.
- No frequency or presence penalty: neither word appears as a parameter in `configuration_utils.py`, `utils.py` or `logits_process.py` (searched).

- https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py (lines 1416 to 1660, 3274 to 3279)
- https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py (`TemperatureLogitsWarper` line 238, `RepetitionPenaltyLogitsProcessor` 306, `TopKLogitsWarper` 571, `MinPLogitsWarper` 733)
- https://github.com/huggingface/transformers/blob/main/src/transformers/generation/configuration_utils.py (lines 184 to 221, 602 to 620)

### D5. `torch.multinomial(probs, 1)` is not an inverse-CDF draw

**Verdict: CONFIRMED.** For one sample (and for any sampling without replacement), every device runs an exponential race; the inverse-CDF search is used only with replacement and more than one sample.

- Dispatch, `native_functions.yaml` (last commit 2026-11-20): `multinomial` and `multinomial.out` dispatch `CPU, CUDA, MPS, XPU` to the same `multinomial` / `multinomial_out`; tag `nondeterministic_seeded`.
- `aten/src/ATen/native/Distributions.cpp` (last commit 2026-11-10), `multinomial_out`: "// Fast-path for no replacement or if only one sample is drawn." The comment derives it from Gumbel-max: "// The algorithm is from gumbel softmax. // s = argmax( logp - log(-log(eps)) ) where eps ~ U(0, 1) ... // s = argmax( p / q ) where q ~ Exp(1)". Code: `q = at::empty_like(...).exponential_(1, gen)`, `at::div_out(q, self, q)`, then `at::argmax_out(result, q, -1, true)` for one sample. Ties: "argmax breaks both kinds of tie toward low indices". Validity is checked with `at::_assert_async` (no host sync).
- With replacement and several samples, CUDA (`MultinomialKernel.cu`, last commit 2026-11-10) builds a CDF with `cumsum` and calls `binarySearchForMultinomial` (first index with `cumdist[mid] >= val`), drawing u with `curandStatePhilox4_32_10_t` and `curand_uniform4` (or `curand_uniform2_double` for a double CDF: "A float uniform has only 24 bits of resolution, too coarse for a double CDF"); it handles u = 1: "this actually can happen when the random number is 1 (github pytorch issue #4858)". MPS (`operations/Distributions.mm`, 2026-11-11) does `self.cumsum(-1, kFloat)`, `at::rand(...).mul_(total)`, `at::searchsorted_out(result, cdf, uniform)`.
- The exponential itself differs by device: `aten/src/ATen/core/TransformationHelper.h`: "// Different implementations for CUDA and CPU to preserve original logic // TODO: must be investigated and unified!!!"
- Docs (2.15), `torch.multinomial`: "The rows of input do not need to sum to one (in which case we use the values as weights), but must be non-negative, finite and have a non-zero sum." Reproducibility note: "Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms. Furthermore, results may not be reproducible between CPU and GPU executions, even when using identical seeds."

For the lessons (ours): the reader's mental model "multinomial = draw u, walk the CDF" is not what `generate` runs; it runs argmax(p / q). Both are exact samplers of the same distribution, but the same random bits give different tokens under the two methods, which is one reason "same seed, same token" never holds across frameworks.

- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml (lines 9608 to 9617)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Distributions.cpp (lines 585 to 638)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/MultinomialKernel.cu (lines 88 to 165, 200 to 235)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/Distributions.mm (lines 692 to 710)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/core/TransformationHelper.h (lines 126 to 142)
- https://docs.pytorch.org/docs/2.15/generated/torch.multinomial.html
- https://docs.pytorch.org/docs/2.15/notes/randomness.html

### D6. PyTorch's generators: mt19937 on the CPU, cuRAND's Philox on CUDA, a hand-written Philox on MPS that is not the CUDA stream

**Verdict: PARTLY.** The CPU and CUDA generators are as expected. The MPS generator is a Philox written in Metal, but by our reading of its source it runs 9 rounds and orders the words differently, so it does not reproduce cuRAND's or Random123's Philox4x32-10 for the same seed (contrary to the "Philox-4x32-10" in the host comment).

- CPU: `aten/src/ATen/CPUGeneratorImpl.h` (2024-07-08) holds `at::mt19937 engine_;`.
- CUDA: `aten/src/ATen/cuda/CUDAGeneratorImpl.h` (2026-10-15) documents the pattern `curandStatePhilox4_32_10_t state; curand_init(std::get<0>(seeds), // seed ...` with a `PhiloxCudaState` (seed, offset) handed to each kernel (A3).
- PyTorch's own engine, `aten/src/ATen/core/PhiloxRNGEngine.h` (2025-10-17): "Note that currently this implementation of the philox engine is not used anywhere except for tests in cpu_generator_test.cpp. However, this engine will replace curandStatePhilox4_32_10_t in the future." Its `reset_state` puts the seed in key words (low, high) and the subsequence in counter words 2 and 3, and its `rand` runs `n_rounds - 1` rounds with key bumps and a final round, i.e. 10: the A3 layout.
- MPS: `aten/src/ATen/mps/MPSGeneratorImpl.h` uses `at::Philox4_32` for the host-side seed and offset; the kernels use `c10/metal/random.h` (2026-06-22): "// Philox Counter based RNG implementation for Metal // Borrowed from aten/src/ATen/core/PhiloxRNGEngine.h". Its `multiple_rounds` loops `for (uint round = 0; round < rounds - 1; ++round) { ctr = single_round(ctr, key); key += kPhilox10; } return ctr;` (no final round), and `splitlong(v)` returns `uint2(v >> 32, v & 0xffffffff)`, so key and counter words are (high, low). `philox4::rand(seed, index)` puts the index in counter words z, w. Our transcription (E3) matches Philox with 9 rounds and that word order, and differs from the 10-round KAT for seed 0, index 0.
- Uniform conversion on MPS and in `PhiloxRNGEngine.h`: `static_cast<float>(value & 0x7FFFFFFF) * scale` with `scale = 4.6566127342e-10` ("maximum value such that `MAX_INT * scale < 1.0`"), so 31 bits, range [0, 1). The host comment in `Distributions.mm` (2026-11-11) speaks of "Philox-4x32-10 calls".
- Not verified: whether 9 rounds is intended; Random123 ships 7-round test vectors as well (A4), so fewer rounds is a known variant, but it is a different stream. Not run on a device.

- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/CPUGeneratorImpl.h
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDAGeneratorImpl.h (lines 20 to 100)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/core/PhiloxRNGEngine.h (lines 30 to 240)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/mps/MPSGeneratorImpl.h
- https://github.com/pytorch/pytorch/blob/main/c10/metal/random.h (lines 1 to 82)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/Distributions.mm (lines 44 to 60)

### D7. MLX: a splittable Threefry generator, and `categorical` that switches algorithm with the batch

**Verdict: CONFIRMED.**

- Docs (MLX 0.32.4, "Random"): "Random sampling functions in MLX use an implicit global PRNG state by default. However, all function take an optional key keyword argument for when more fine-grained control or explicit state management is needed." "Following JAX's PRNG design we use a splittable version of Threefry, which is a counter-based PRNG."
- Kernel, `mlx/backend/metal/kernels/random.metal` (last commit 2026-04-21): `threefry2x32_hash(const thread uint2& key, uint2 count)` with rotation tables `{13, 15, 26, 6}` and `{17, 29, 16, 24}`, the key-schedule constant `0x1BD11BDA`, and `for (int i = 0; i < 5; ++i) { for (auto r : rotations[i % 2]) { ... } ... }`: 5 × 4 = 20 rounds (our count).
- Keys, `mlx/random.cpp` (last commit 2026-10-25): `array key(uint64_t seed) { uint32_t k1 = static_cast<uint32_t>(seed >> 32); uint32_t k2 = static_cast<uint32_t>(seed); return array({k1, k2}); }`.
- Uniform: "// Get random values between [0, nextafter(1.0, 0.0)] since samples must // be in [low, high)": `bits / UINT32_MAX` in float, then `minimum(out, nextafter(1.0f, 0.0f))`.
- `categorical_impl`: if the logits are a single distribution and a sample shape is given (`n > 0 && logits.size() == n && !shape.empty()`), it calls `categorical_inverse_cdf`, commented "// O(N + M) in memory, where the Gumbel-max trick needs O(N * M)": weights `exp(x - m)`, an exclusive `cumsum`, `u = uniform * sum(w)`, `searchsorted(cdf, u, "right") - 1`. Otherwise Gumbel-max: `gumbel = -log(-log(uniform))`, `argmax(add(g, logits), ...)`.
- Our reading: mlx-lm calls `mx.random.categorical(logits * (1.0 / temp))` on a (batch, vocabulary) array (D8). With batch 1 that array has `size() == n` and a sample shape of (1), so it takes the inverse-CDF path; with batch 2 or more it takes Gumbel-max. The same key can therefore give a different token for the same row depending on how many rows are in the batch.

- https://ml-explore.github.io/mlx/build/html/python/random.html
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.random.categorical.html
- https://github.com/ml-explore/mlx/blob/main/mlx/random.cpp (lines 31 to 35, 100 to 146, 383 to 445)
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/random.metal (lines 1 to 30)

### D8. mlx-lm: truncation before temperature, top-p before top-k, min-p in log space

**Verdict: CONFIRMED.** `mlx_lm/sample_utils.py` (last commit 2026-09-24), `mlx_lm/generate.py` (2026-11-19).

- `make_sampler(temp=0.0, top_p=0.0, min_p=0.0, min_tokens_to_keep=1, top_k=0, xtc_probability=0.0, ...)`: "temp (float): The temperature for sampling, if 0 the argmax is used. Default: ``0``." `if temp == 0: return greedy_sampler` (`mx.argmax(logprobs, axis=-1)`).
- Chain order as built: `if 0 < top_p < 1.0:` top-p; `if min_p != 0.0:` min-p; `if xtc_probability > 0.0:` XTC; `if top_k > 0:` top-k; then `categorical_sampling(logprobs, temp)` = `mx.random.categorical(logits * (1.0 / temp))`. Temperature is applied last (as in llama.cpp, unlike transformers), and top-p runs before top-k (unlike both).
- Input to the chain: `logprobs = logits - mx.logsumexp(logits, keepdims=True)` (generate.py), after logits processors.
- Min-p: `top_logprobs = mx.max(logprobs, axis=-1, keepdims=True)`, `scaled_min_p = top_logprobs + math.log(min_p)`, `tokens_to_remove = logprobs < scaled_min_p`: the log-space form of transformers' test, before temperature.
- Repetition penalty: `mx.where(selected_logits < 0, selected_logits * penalty, selected_logits / penalty)` over the last `context_size` tokens (default 20). Presence and frequency: ch09 D5. Top-k and top-p masking: ch12 D6, ch11 D3.
- All of it runs as MLX ops, so on the GPU under Metal; the filters are `mx.compile`d with `inputs=mx.random.state, outputs=mx.random.state`.

- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/sample_utils.py (lines 13 to 72, 133 to 134, 162 to 211, 271 to 306)
- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py (lines 427 to 439, 569 to 574)

### D9. What the cited sources define: top-p's boundary (not its ties), min-p's threshold, the repetition penalty, frequency and presence penalties

**Verdict: PARTLY.** Each rule is defined in its source, including the boundary for top-p; none of the sources defines what happens to tied probabilities at the top-p or top-k boundary.

- Top-p, Holtzman et al., "The Curious Case of Neural Text Degeneration", ICLR 2020 (arXiv 1904.09751), section 3.1: "Given a distribution P(x|x1:i−1), we define its top-p vocabulary V(p) ⊂ V as the smallest set such that Σ_{x∈V(p)} P(x|x1:i−1) ≥ p." (equation 2). So the token that crosses p is included (the ch11 D2 rule). "Smallest set" is not unique when tokens tie at the boundary; the paper gives no tie rule, which is why the projects differ (ch11 D2, D3, D4).
- Min-p, Nguyen et al., "Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs", ICLR 2025 (arXiv 2407.01082): "pmax = maxv∈V P(v | x1:t−1)", "pscaled = pbase × pmax", "Vmin = {v ∈ V : P(v | x1:t−1) ≥ pscaled}". On ordering: "After applying temperature scaling, the scaled threshold pscaled is computed, and tokens with probabilities below this threshold are filtered out before sampling." llama.cpp's default chain and mlx-lm apply min-p before temperature (D1, D8); transformers after (D4).
- Repetition penalty, Keskar et al., CTRL (arXiv 1909.05858), "penalized sampling": "pi = exp(xi/(T · I(i ∈ g))) / Σj exp(xj/(T · I(j ∈ g))), I(c) = θ if c is True else 1", "using a greedy sampling and θ ≈ 1.2 yields a good balance". It divides regardless of sign; every project read multiplies negative logits instead (D2, D4, D8).
- Frequency and presence penalties, OpenAI API docs, "Parameter details": `mu[j] = mu[j] - c[j] * alpha_frequency - float(c[j] > 0) * alpha_presence`, where "`c[j]` is how often that token was sampled prior to the current position". "Reasonable values for the penalty coefficients are around 0.1 to 1 ... one can increase the coefficients up to 2". llama.cpp implements exactly this (ch09 D4); vLLM's `SamplingParams` docstrings describe presence and frequency over "the generated text so far" and repetition over "the prompt and the generated text so far" (`vllm/sampling_params.py`, last commit 2026-11-19).

- https://arxiv.org/abs/1904.09751 (PDF section 3.1, page 4)
- https://arxiv.org/abs/2407.01082 (PDF section 3, and the "Integration into Decoding Pipelines" paragraph)
- https://arxiv.org/abs/1909.05858 (PDF section 4.1)
- https://developers.openai.com/api/docs/guides/advanced-usage (section "Frequency and presence penalties")
- https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py (lines 296 to 311)

---

## Group E: Our arithmetic

Scripts were run with `python3 -I` on the host; inputs are the files cited above.

### E1. A Philox4x32 transcription passes Random123's known-answer vectors

**Verdict: CONFIRMED.**

A 20-line Python transcription of A4's round, key bump and round count reproduced all six `philox4x32` lines of `tests/kat_vectors` (R = 7 and R = 10, counters and keys all-zero, all-ones and the digits of pi). For example R = 10, counter 0, key 0 gives `6627e8d5 e169c58d bc57ac4c 9b00dbd8`.

Combined with A3 (our derivation), the first `curand()` after `curand_init(0, 0, 0, &s)` should be `0x6627e8d5`; with B1, the same for `rocrand_init`. Both are unverified until a CUDA or ROCm verify report shows them. The lab can print these words on every backend as its first test.

### E2. Turning 32 random bits into u: three conversions compared

**Verdict: CONFIRMED.**

Single-precision rounding emulated with `struct` (round to nearest even). Multiplying by 2^-32 is exact in float, so contraction into a fused multiply-add does not change these results.

| Conversion | Formula | Range | Exactly 1.0 for |
|---|---|---|---|
| cuRAND `curand_uniform` (A2) | x·2^-32 + 2^-33 | (0, 1] | 128 of 2^32 inputs (x ≥ 0xFFFFFF80) |
| rocRAND / hipRAND on AMD (B2) | 2^-32 + x·2^-32 | (0, 1] | the same 128 inputs |
| 24-bit, ours | (x >> 8) · 2^-24 | [0, 1 − 2^-24] | never |

- cuRAND and rocRAND give different floats for 3,903 of 1,000,000 random inputs (seed 1). Examples: x = 0 gives 1.1641532e-10 and 2.3283064e-10; x = 0xFF gives 5.9488229e-08 and 5.9604645e-08; from x = 2^24 upward the samples printed agree.
  - **Correction (2026-11-20, found while building the labs):** the last clause is incomplete. Exactly 2^24 of the 2^32 inputs differ: every x below 2^23, the even x in [2^23, 2^24), and the x in [2^24, 2^25) whose float32 value is 2 mod 4 (that is, x ≡ 2 mod 4), a quarter of that binade. From 2^25 upward the two agree. Checked by `app/src/sim/philox.test.ts` ("the regions" and the per-binade count).
- The 24-bit conversion is exact: every result is a multiple of 2^-24 below 1, representable in float, computed by an integer shift, an exact integer-to-float conversion (values below 2^24, so round-to-nearest-even on every backend: A5, C4) and a multiplication by a power of two. It is therefore bit-identical on CUDA, HIP, Metal (under any math mode) and the CPU. It returns 0.0 with probability 2^-24, so the draw rule must be "first token whose inclusive cumulative probability is strictly greater than u" (or u must be offset by half a step) to keep a leading zero-probability token unpickable.
- The real limit on cross-backend agreement is then not u but the float CDF the search runs over (C5, ch10 A11). A test that demands exact token agreement must avoid, or detect, draws within a small margin of a boundary.

### E3. PyTorch's Metal Philox is a 9-round Philox with high-low word order

**Verdict: CONFIRMED** (as transcription; not run on a device).

A Python transcription of `c10/metal/random.h` `philox4::rand(seed, index)` (D6) equals our Philox with R = 9, key (seed_hi, seed_lo) and the index in counter words z, w:

| seed, index | PyTorch Metal (transcribed) | Philox4x32-10, A3 layout |
|---|---|---|
| 0, 0 | `7028b1c8 d6594c40 4f2051bb 76d6628e` | `6627e8d5 e169c58d bc57ac4c 9b00dbd8` |
| 42, 0 | `9a822623 fd1355ba 3f3f7f79 4bd4b252` | `9ceaf053 77f5493b 12bf50ad 5742b3d7` |

So `torch.rand` on MPS and on CUDA give unrelated streams for the same seed, as PyTorch's randomness notes allow (D5).

---

## Project notes (not vendor facts)

Recorded so lab authors do not have to rediscover them; these describe this repository on the check date.

- No lab uses cuRAND, hipRAND or Philox yet (searched `labs/` outside the build directories). `labs/common/include/pmpp/vendors/hip.h` maps no cuRAND names, and the include paths differ (`<curand_kernel.h>` against `<hiprand/hiprand_kernel.h>`), so a lab that called the libraries would need `#if` guards. A hand-written Philox needs none.
- Neither the emulator (`labs/common/include/pmpp/emu/emu_device.h`) nor the front-end stub (`labs/common/include/pmpp/check/cuda_stub.h`) declares `__umulhi` (searched). A lab that calls it needs a shared change (report it, do not edit shared files from a chapter build). The 64-bit product form (A5) needs no change and is the same source on Metal with `ulong`.

## Answers to the brief's portability questions

| Question | Answer | Where |
|---|---|---|
| Counter-based RNG on CUDA | cuRAND device API, `curandStatePhilox4_32_10_t`, `curand_init(seed, subsequence, offset, &s)`, `curand_uniform` in (0, 1] | A1, A2, A3 |
| On HIP | rocRAND `rocrand_state_philox4x32_10` / hipRAND `hiprandStatePhilox4_32_10_t` with the same arguments; HIPIFY maps the names; float conversion differs from cuRAND's; AMD's table says not sequence-compatible | B1, B2 |
| On Metal | Nothing in MSL. MPS `MPSMatrixRandomPhilox` (host-encoded, macOS 10.15) and MPSGraph Philox random ops (macOS 11), both [0, 1) with undocumented internals. Write Philox in MSL (`mulhi` or `ulong`), as Random123 and PyTorch do | C1 to C4, D6 |
| Same token from a fixed seed on every backend | Possible only with our own Philox (checked by the KATs), our own bit-to-float conversion, the same draw rule, and a test that avoids or flags draws within a small margin of a CDF boundary (float CDFs differ by backend) | A4, E1, E2, C5 |
| Which framework draws how | transformers: `torch.multinomial` = argmax(p / q), q ~ Exp(1); MLX: inverse CDF for batch 1, Gumbel-max otherwise; llama.cpp: inverse CDF (host `mt19937`, or host-drawn u with a GPU scan); FlashInfer: inverse CDF over a block scan with cuRAND Philox per row | D3, D4, D5, D7, A6 |
| Where the sampler runs | llama.cpp: host by default, experimental backend path behind `--backend-sampling`; transformers and PyTorch: on the tensor's device; MLX: on the GPU; vLLM: GPU (ch14 D7) | D1, D4, D5, D8 |
| Order of temperature and truncation | llama.cpp default and mlx-lm: truncation first, temperature last; transformers: temperature first; min-p paper: after temperature | D1, D4, D8, D9 |

---

## Summary of corrections

Every WRONG and PARTLY item, and every failed or unfound sub-claim:

- A1: cuRAND's docs give Philox's period as 2^128 and a subsequence as 2^64, the header as 2^130 values and 2^66; the docs mean counter steps without saying so. "The number of values consumed is not guaranteed to be fixed" for distribution functions, so reproducible draws should start from raw bits. "Large offsets take more time" does not apply to Philox's constant-time skip-ahead (our reading of the header).
- B1: AMD documents rocRAND's Philox 4x32-10 as not producing cuRAND's sequence ("No", host API with legacy ordering); AMD's device header initialises exactly as cuRAND's does, so raw device streams look identical by reading. Unresolved; unverified on any device.
- B2 (WRONG): `hiprand_uniform` on AMD is rocRAND's 2^-32 + x·2^-32, cuRAND's `curand_uniform` is x·2^-32 + 2^-33; they differ for about 0.39 % of 32-bit inputs, so the same `.cu` gives different u on NVIDIA and AMD.
- C1 (CONFIRMED, listed because it surprises): MSL has no random number function at all.
- C2: Apple's `MPSMatrixRandomPhilox` is a host-encoded fill kernel, not a shader function; rounds, seed mapping and conversion are undocumented (could not verify compatibility with any other Philox).
- C5 (WRONG): MSL permits round-toward-zero for float arithmetic ("Either round ties to even or round toward zero rounding mode may be supported"), in addition to fast-math defaults; float CDFs need not match the CPU bit for bit.
- D2 (detail): llama.cpp's repetition penalty multiplies non-positive logits and divides positive ones; the CTRL paper it cites divides regardless of sign; transformers and mlx-lm use `< 0` instead of `<= 0` (same result at 0).
- D3 (our reading): llama.cpp's host draw selects the first sum ≥ u × total, its backend draw the first sum strictly > u; a double just below 1 cast to `1.0f` makes the backend pick the last candidate, which can be a zero-probability token. The backend RNG is still host `mt19937`.
- D4: transformers' top-k keeps every token tied with the k-th value (can keep more than k); transformers applies temperature before top-k, top-p and min-p; it has no frequency or presence penalty.
- D5: `torch.multinomial(probs, 1)`, which transformers' `generate` uses, is an exponential race argmax(p / q), not an inverse-CDF search, on CPU, CUDA, MPS and XPU.
- D6: PyTorch's MPS Philox (`c10/metal/random.h`) runs 9 rounds with high-low word order by our reading, despite "Philox-4x32-10" in the host comment, so MPS and CUDA streams differ for the same seed (not run).
- D7 (our reading): MLX's `categorical` uses an inverse CDF for a single distribution with a sample shape (batch 1 in mlx-lm) and Gumbel-max otherwise, so batch size changes the token drawn for the same key.
- D8: mlx-lm applies top-p, then min-p, then top-k, all before temperature.
- D9: the top-p paper defines the boundary (the crossing token is included) but not ties; the min-p paper applies min-p after temperature, which llama.cpp's default chain and mlx-lm do not.

Path and status facts worth knowing even where the verdict is CONFIRMED:

- A3: cuRAND's Philox headers are under NVIDIA's restrictive licence; show Random123's code (BSD-style) in lessons.
- A4, E1: Random123's known-answer vectors are the cross-backend test for a hand-written Philox; Random123 already runs as MSL.
- A6: FlashInfer's `SamplingFromProbKernel` is M5's final draw in production: one block per row, `curand_init(seed, row, offset)`, chunked `BlockScan`, first inclusive sum > u, explicit fallback for u near 1.
- D1: llama.cpp's default chain is penalties, DRY, top-n-sigma, top-k, typical, top-p, min-p, XTC, temperature, then `dist`; defaults top_k 40, top_p 0.95, min_p 0.05, temp 0.8; backend sampling is off by default for generation and on for speculative drafts.
- E2: a 24-bit conversion `(x >> 8) * 2^-24` is bit-identical on every backend and never returns 1.0.

Not checked: the AIR TopK and Qrita papers themselves (cited through ch14 A7 and D7); Apple's documentation pages for `MPSGraph` sampling beyond the random ops; any behaviour on real hardware.

## Where the backends really differ

One line per cell, for what X1's lessons and labs use. Nothing here was run on CUDA or AMD hardware; the Metal cells are from the spec and SDK headers, not compiled for this file.

| Feature | CUDA | HIP (ROCm 10.2) | Metal (MSL 4.1) |
|---|---|---|---|
| Device-side RNG library | cuRAND device API, `<curand_kernel.h>` | rocRAND `<rocrand/rocrand_kernel.h>`, hipRAND `<hiprand/hiprand_kernel.h>` | None in MSL |
| Philox state and init | `curandStatePhilox4_32_10_t`, `curand_init(seed, subsequence, offset, &s)` | `hiprandStatePhilox4_32_10_t` + `hiprand_init` (same arguments); `rocrand_state_philox4x32_10` + `rocrand_init` | None; write it (Random123 runs as MSL; PyTorch ships its own) |
| Library Philox outside kernels | Host API generators (not used here) | Same through hipRAND | `MPSMatrixRandomPhilox` (macOS 10.15), MPSGraph `randomPhiloxStateTensor` (macOS 11); internals undocumented |
| Raw stream equal to cuRAND's | Reference | Same init by our reading of the headers; AMD's table says "No"; unverified | Only with a hand-written Philox that passes the KATs |
| Library uniform float | `curand_uniform`: x·2^-32 + 2^-33, (0, 1], 1.0 possible | `hiprand_uniform` on AMD: 2^-32 + x·2^-32, (0, 1]; differs from CUDA for about 0.4 % of inputs | MPS and MPSGraph: [0, 1); formula undocumented |
| Raw values consumed per distribution call | "not guaranteed to be fixed" | `rocrand_uniform` "increments position of the generator by one" (header doc comment) | Not applicable |
| 32 × 32 high product | `__umulhi` | `__umulhi` (documented) | `mulhi` |
| 64-bit product alternative | `unsigned long long` | Same | `ulong` (Metal 2.2 and later) |
| Integer to float conversion | `__uint2float_rn`; default cast rounds to nearest even | Same (C++) | `static_cast`: round ties to even |
| Float arithmetic rounding | IEEE round to nearest; contraction on by default | Same | Round ties to even or toward zero "may be supported"; fast or relaxed math by default |
| Categorical draw in a library | None for arbitrary weights (`curand_discrete` takes a host-built histogram; documented constructor is Poisson only) | Same through hipRAND | None (no categorical or multinomial op in MPS or MPSGraph) |
| Top-k selection in a library | `cub::DeviceTopK` (unordered only), `DeviceBatchedTopK` (CUDA 13.5) | `rocprim::topk`, `segmented_topk`; no hipCUB wrapper | MPSGraph `topK` (macOS 14); `MPSMatrixFindTopK` (k at most 16) |
| What the frameworks draw with today | PyTorch: Philox via `curand_init`, `multinomial(p, 1)` = argmax(p / q); FlashInfer: inverse CDF per row | PyTorch on ROCm: same source after hipify (ch01 D4); the ROCm RNG path was not checked separately | PyTorch MPS: own 9-round Philox (our reading); MLX: Threefry, inverse CDF at batch 1, Gumbel-max otherwise |
| llama.cpp final draw | Host `mt19937` by default; experimental graph: host-drawn u, `ggml_cumsum`, step, count | Same graph through the CUDA source | Same graph on the Metal backend |
