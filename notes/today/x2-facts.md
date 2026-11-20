# Module X2 fact check (quantised kernels): CUDA, HIP, Metal, and the real formats

Checked on 2026-11-20 against the live vendor and project sources listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY. Where an item has several sub-claims, the verdict is for the item as a whole and the text says which sub-claim failed. A verdict judges the claim in the item title (the assumption a course author would most likely bring).
- **Quote** is copied from the source (HTML docs flattened to text, GitHub raw files on the default branch or at a tag, PDF text, local SDK headers). Line breaks inside quotes are removed.
- "appA A5", "infra C10", "ch15 B4" and so on point at items in `notes/today/appA-facts.md`, `infra-facts.md`, `ch15-facts.md`, etc. Anything settled there is cited, not re-checked.
- **There is no `notes/x2.md`.** Module X2 is ours. PLAN.md anchors it on appendix A section A.5 (pp. 592 to 593, recorded in `notes/appA.md`) and on chapter 15's intensity analysis; `notes/appA.md` states plainly what A.5 does not contain: "No integer quantization (int8/int4 with scale and zero-point), no fixed-point, no mixed-precision accumulation rules." So part (a) of the brief reduces to the anchors those chapters already checked (Group A, A1), and there is no "Unclear in the PDF" list. Every other item is about vendor APIs and the code of real projects. Lessons must label all of it as ours, and each claim about a codebase as "checked in the facts pass on 2026-11-20".
- Versions the sources reported about themselves on the check date: CUDA Math API "13.5"; PTX ISA "Version 9.5"; CUDA Programming Guide (C++ language extensions page) and cuBLAS "13.5"; ROCm Core SDK 10.2.0 (HIP headers at tag `therock-10.2` of `ROCm/rocm-systems`; ROCm's compiler at tag `therock-10.2` of `ROCm/llvm-project`); upstream `llvm/llvm-project` `main` (`GCNFeatures.td` last commit 2026-11-19); HIPIFY `amd-develop` (device table last commit 2026-09-22); Metal Shading Language Specification "Version 4.1", dated 2026-06-04 (383 pages, downloaded again today); Metal Feature Set Tables dated October 23, 2026; macOS 27.0 SDK headers on this machine; llama.cpp `master`, MLX `main` (docs "MLX 0.32.4"), PyTorch `main` and docs 2.15, torchao `main` (latest release v0.18.0, 2026-08-03). Last-commit dates are given per file.
- Not checked: anything by running code on a GPU. Where this file says what a piece of code does (which nibble holds which weight, what a kernel accumulates in), that is our reading of the source, labelled as such. Group E is our own arithmetic, run in Python 3 with NumPy 2.0.2 on the host (`/usr/bin/python3`), seeded, and is not a vendor statement.
- Device figures in this file (the A100 in A1) are evidence for a verdict, not lesson content. The project rule stands: no device numbers in lessons except the book's H100 as the book's example; reader numbers come from `python3 labs/pmpp.py doctor` and `reports/runs/*.json`.

Totals: 19 CONFIRMED, 2 WRONG, 6 PARTLY, 0 COULD NOT VERIFY (27 items: 6 CUDA and anchors, 5 HIP, 5 Metal, 7 project, 4 arithmetic). Sub-claims that could not be verified are called out inside A2, A4, B1, B2, C2, C3 and collected under "Summary of corrections".

---

## Already settled elsewhere (cite, do not re-check)

| X2 needs | Settled in |
|---|---|
| §A.5's statements: "Recent GPU architectures introduced hardware support for computing with 16-bit, 8-bit, and 4-bit floating point numbers"; H100 FP8 3,958 vs "989 TFLOPS" (both with sparsity; 989 is TF32) | appA A10, A11 |
| FP16, BF16, FP8 (E4M3, E5M2) bit layouts; OCP MX v1.1 (MXFP8/6/4, UE8M0, block 32); NVIDIA's 16-element FP4 with UE4M3 scales; cuBLAS rounds UE8M0 scales up | appA A2 to A5 |
| HIP FP8 (OCP and FNUZ), FP6, FP4, `__amd_scale_t` E8M0; Metal 4.1 FP8/FP4/UE8M0 packed formats, `tensor_blockwise` (UE8M0 scales, block 32); MSL Table 2.15/2.16 misprints | appA B2, C4 |
| MLX quantisation modes `affine`, `mxfp4`, `mxfp8`, `nvfp4` and their group sizes; llama.cpp `block_mxfp4` (17 bytes per 32), `block_nvfp4` (64 values, UE4M3 per 16), the doubled E2M1 table, E8M0 decode | appA D2, D3 |
| Accumulator width as a graph property in ggml (`GGML_PREC_F32`, `_BF16`, `_F16`); PyTorch accumulates half/bf16 in float; MLX half sums stay half | appA D1 to D3 |
| `__half`, `__nv_bfloat16` vs `__hip_bfloat16`; header names need `#if`; the emulators do not model `__half` or `__nv_bfloat16`, nor does the front-end stub | appA B1, E2; infra A12, B12 |
| Decode is a matrix-vector product and memory-bound; batching raises intensity | ch22 A4, A5; ch20 E4 |
| The book's H100 bandwidth and FP32 peak (book's example only) | ch05 A8 |
| Roofline and FLOP/B from chapter 5; the book's 8 FLOP/B memory-bound and 32 FLOP/B compute-bound examples | ch15 A1 |
| Vector types, `float4` alignment 16, the `float*` to `float4*` cast is undocumented | infra A6, B6, C7 |
| MSL integer types `char`/`int8_t`, `uchar`, `packed_char4`; `long` is always 64-bit in MSL | infra C9 |
| Tensor-core input types per compute capability (Table 36: INT8 on 9.0; FP6/FP4 from 10.0); WMMA; rocWMMA; `simdgroup_matrix`; MPP `matmul2d` (`char, char -> int` in Metal 4) | infra A12, B12, C10 |
| AMD MFMA and WMMA float builtin spellings, RDNA3/4 have WMMA | ch15 B4 |
| Where llama.cpp keeps `mmvq.cu`, `mmq.cu`, `vecdotq.cuh`, Metal `mul_mv.metal`, `mul_mm.metal`; MLX `quantized.cpp` and its kernel families | infra D1 to D3 |
| llama.cpp uses matrix-vector kernels (`MMVQ_MAX_BATCH_SIZE 8`) up to 8 tokens | ch15 D1, ch22 D3 |
| llama.cpp KV-cache types `q8_0`, `q4_0` (34 and 18 bytes per 32 values) | ch20 D3 |
| Apple documents no bandwidth, cache or bank figures for its GPUs | infra C13 |
| HIP's `__umulhi`; Metal `mulhi`, `as_type`, `static_cast` rounding | x1 B3, C4 |

---

## Group A: the book's anchors, and CUDA

### A1. §A.5's argument applied to weights: fewer bytes per weight raise the intensity of bandwidth-bound decode, and that is why weight-only quantisation speeds decode up

**Verdict: CONFIRMED** as an argument, with a caution the lessons need: the speed-up observed is smaller than the byte ratio, and an unfused dequantise step can make it negative.

- The book (`notes/appA.md`, §A.5, p. 593): a smaller format reduces the number of bytes fetched per arithmetic operation, which improves the application's compute intensity. The book applies it to floating-point arithmetic only and gives no integer formats.
- Decode is memory-bound: ch22 A5 (NVIDIA's blog: "Model execution is frequently memory-bandwidth bound—in particular, bandwidth-bound in the weights."), ch20 E4.
- PyTorch's own blog, "Accelerating Generative AI with PyTorch II: GPT, Fast" (last updated 2024-11-14), states the X2 argument in the reader's terms: "because of the KV-cache, for BS=1 every single matrix multiplication in a transformer is actually a matrix vector multiplication. This means that the computations are completely memory-bandwidth bound". "We can't really change the number of parameters in our model. We can't really change the memory bandwidth of our GPU (well, without paying more money). But, we can change how many bytes each parameter is stored in!" On int8 weight-only: "Note that this is quantizing only the weights – the computation itself is still done in bf16."
- The caution, same post: "the light-blue line (no torch.compile + int8) is actually much worse than even the fp16 performance! This is because in order to take advantage of the perf benefits of int8 quantization, we need the kernels to be fused." For int4: "we need kernels that fuse int4 dequantize with the matrix vector multiplication. In this case, torch.compile is unfortunately not able to generate these kernels from scratch, so we leverage some handwritten CUDA kernels in PyTorch." (Those are `_weight_int4pack_mm`, D6.)
- Their measured ratio (their device, an "A100-80GB, power limited to 330W"; evidence only, not lesson content): 107.0 to 157.4 tokens/s for FP16 to int8 weights, about 1.47 times for half the bytes. Their metric "Model Bandwidth Utilization (MBU)" ("total size of our model (# params * bytes per param) ... multiply it by the number of inferences we can do per second ... divide this by the peak bandwidth") is the measurement labs X2.1 and M9 should report, with the reader's own peak from `doctor`.
- Their accuracy remark: "using int8 weight-only quantization has no perceptible accuracy degradation, using int4 weight-only quantization does" and, for int4, "one scaling factor per 32 elements ... this is also a common choice among the community."

For the lessons (ours): the roofline knob is bytes per weight (E1). Predict the speed-up as the ratio of bytes moved, then let the reader's measurement show how much of it survives the extra integer work, the scales and the activation traffic.

- https://pytorch.org/blog/accelerating-generative-ai-2/
- `notes/appA.md` (§A.5); ch22 A5; ch20 E4

### A2. `__dp4a`: a four-way byte dot product with 32-bit accumulate, from compute capability 6.1

**Verdict: CONFIRMED.** Two details matter for the labs: the CUDA intrinsic has same-signedness overloads only, while the PTX instruction also allows mixed signedness; and nothing documents that nvcc turns a plain byte loop into `dp4a`.

CUDA Math API 13.5, Integer Intrinsics:

- `__device__ int __dp4a(int srcA, int srcB, int c)`: "Four-way signed int8 dot product with int32 accumulate." "Extracts four pairs of packed byte-sized integers from srcA and srcB, then creates four pairwise products and adds them together to a signed 32-bit integer c."
- Also `__device__ int __dp4a(char4 srcA, char4 srcB, int c)`, `__device__ unsigned int __dp4a(unsigned int srcA, unsigned int srcB, unsigned int c)` ("Four-way unsigned int8 dot product with unsigned int32 accumulate.") and `__device__ unsigned int __dp4a(uchar4 srcA, uchar4 srcB, unsigned int c)`. No mixed signed/unsigned overload.
- `__dp2a_lo` / `__dp2a_hi` (`int`, `short2`/`char4`, and unsigned forms): "Two-way signed int16 by int8 dot product with int32 accumulate, taking the lower half of the second input." (upper half for `_hi`).
- The page states no compute capability; only "All of these functions are supported in device code."

PTX ISA 9.5, section 9.7.1.24 `dp4a`:

- Syntax `dp4a.atype.btype d, a, b, c;` with `.atype = .btype = { .u32, .s32 };` "Operand c has type .u32 if both .atype and .btype are .u32 else operand c has type .s32." Semantics: `d = c; Va = extractAndSignOrZeroExt_4(a, .atype); Vb = extractAndSignOrZeroExt_4(b, .btype); for (i = 0; i < 4; ++i) { d += Va[i] * Vb[i]; }`. Example `dp4a.u32.s32 d1, a1, b1, c1;` (mixed signedness).
- "Introduced in PTX ISA version 5.0." "Requires sm_61 or higher." Same for `dp2a` (9.7.1.25).

llama.cpp agrees: `#define GGML_CUDA_CC_DP4A 610 // minimum compute capability for __dp4a, an intrinsic for byte-wise dot products` (`ggml/src/ggml-cuda/common.cuh`, line 51, last commit 2026-11-19), and falls back to four byte products below it (D3).

COULD NOT VERIFY: any NVIDIA statement that nvcc emits `dp4a` for a scalar loop of four `int8_t` products, or any throughput figure for `dp4a`. The lab measures rather than asserts.

- https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (sections 9.7.1.24 and 9.7.1.25)

### A3. CUDA's per-thread dot-product intrinsics are integer only; there is no half, bf16 or FP8 dot

**Verdict: CONFIRMED.** The CUDA Math API "Half2 Arithmetic Functions" page lists `__h2div`, `__habs2`, `__hadd2`, `__hadd2_rn`, `__hadd2_sat`, `__hcmadd`, `__hfma2`, `__hfma2_relu`, `__hfma2_sat`, `__hmul2`, `__hmul2_rn`, `__hmul2_sat`, `__hneg2`, `__hsub2`, `__hsub2_rn`, `__hsub2_sat`; the word "dot" does not occur on it. A narrow-float dot per thread on NVIDIA is a chain of `__hfma2` (accumulating in half, which is the appendix A hazard) or a conversion to float. Narrow-float dot products with wide accumulation exist only in the matrix instructions (A4, infra A12). Contrast HIP (B4).

- https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF2__ARITHMETIC.html

### A4. Integer tensor-core instructions: 8-bit and 4-bit `mma` from sm_75/sm_80; on Hopper and Blackwell the integer path narrows

**Verdict: PARTLY.** The assumption "every recent NVIDIA GPU multiplies int8 and int4 on tensor cores" holds for `mma.sync` up to sm_80, but Hopper's `wgmma` has no 4-bit integer form, and Blackwell's `tcgen05.mma` `.kind::i8` is limited to specific architectures.

PTX ISA 9.5, `mma` (section 9.7.16), "Target ISA Notes":

- ".u8/.s8 integer type mma operation with .m8n8k16 shape requires sm_75 or higher." ".u4/.s4 integer type mma operation with .m8n8k32 shape sm_75 or higher." ".u8/.s8 integer type mma operation with .m16n8k16 and .m16n8k32 shapes requires sm_80 or higher." ".u4/.s4 integer type mma operation with .m16n8k32 and .m16n8k64 shapes requires sm_80 or higher."
- `wgmma` (sm_90a), "Integer type": `wgmma.mma_async.sync.aligned.shape{.satfinite}.s32.atype.btype d, a-desc, b-desc, scale-d;` with `.atype = {.s8, .u8}; .btype = {.s8, .u8};`. No `.s4`/`.u4`.
- `tcgen05.mma`: "Qualifier .kind::i8 is supported on following architectures: sm_100a sm_101a (Renamed to sm_110a from PTX ISA version 9.0) sm_110a", and the family-specific list carries "except .kind::i8".
- New and relevant to 4-bit weights: `ldmatrix` "Support for .s8.s4 types introduced in PTX ISA version 9.4", "Types .s8.s4 are supported on following architectures: sm_90a sm_100f or higher in the same family sm_110f or higher in the same family sm_120f or higher in the same family"; and for 4-bit data in 8-bit slots, "For 4-bit or 6-bit data, 8-bit element will have 4 bits or 2 bits of padding respectively." `cvt.pack.sat.u4.s32.b32` packs to 4 bits ("Sub byte types (.u4 / .s4 and .u2 / .s2) requires sm_75 or higher.").
- Table 36 of the Programming Guide (infra A12) marks INT8 for 9.0; this file did not re-read its INT4 cells.

COULD NOT VERIFY: which of these llama.cpp's `mmq` uses on each architecture (only the `dp4a` and `mma.cuh` paths were read, D3 and ch15 D1). X2's labs stay on CUDA cores (`dp4a` or plain loops); tensor-core integer GEMM is a `<Beyond>` note.

- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (sections 9.7.16 `mma`, `ldmatrix`, `wgmma`, `tcgen05.mma`; 9.7.9 `cvt.pack`)

### A5. `char4` is 4 bytes with 4-byte alignment, and a llama.cpp Q4_0 block is only 2-byte aligned

**Verdict: CONFIRMED.** This is the trap a 4-bit lab walks into first.

- CUDA Programming Guide, Table 49 "Alignment Requirements": "char4 , uchar4 4 4" (size, alignment); "char2 , uchar2 2 2"; "short2 , ushort2 4 4".
- `block_q4_0` is `ggml_half d` plus `uint8_t qs[16]`, 18 bytes (D1), so in an array of blocks every other `qs` starts at an address that is 2 mod 4. llama.cpp therefore reads Q4_0 quants with two 16-bit loads: `static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) { const uint16_t * x16 = (const uint16_t *) x; // assume at least 2 byte alignment` and only uses the 4-byte `get_int_b4` ("assume at least 4 byte alignment") for the Q8_1 activations (`ggml/src/ggml-cuda/vecdotq.cuh`, lines 18 to 29, last commit 2026-10-24).
- For the lab (ours): reading `qs` through an `int*` or `char4*` is undefined for every other block on every backend. (Correction, 2026-11-20, found while building the module: with a 4-byte-aligned array, block i's `qs` starts at byte 18i + 2, which is 2 mod 4 for even i, so it is the even blocks, not the odd ones, whose quants are misaligned for a 4-byte load.) Either read bytes or 16-bit halves, or store scales and quants in separate arrays (MLX's layout, D5), which keeps the quants 4-byte aligned.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (Tables 48 and 49)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vecdotq.cuh

### A6. The library route for int8 × int8 GEMM does not serve batch-1 decode

**Verdict: CONFIRMED.**

- cuBLAS 13.5 (`cublasGemmEx`): "CUBLAS_COMPUTE_32I and CUBLAS_COMPUTE_32I_PEDANTIC compute types are only supported with A, B being 4-byte aligned and lda, ldb being multiples of 4. For better performance, it is also recommended that IMMA kernels requirements for a regular data ordering listed here are met."
- PyTorch's int8 GEMM, `torch._int_mm`, dispatches `CPU`, `CUDA`, `MPS`, `XPU` (`native_functions.yaml`, last commit 2026-11-20). On CUDA, `_int_mm_out_cuda` (`aten/src/ATen/native/cuda/Blas.cpp`, line 1129 on, last commit 2026-10-31) checks `TORCH_CHECK(self.size(0) > 16, "self.size(0) needs to be greater than 16, but got ", self.size(0));`, `self.size(1) ... a multiple of 8`, `mat2.size(1) ... a multiple of 8`, and requires an `int32` (`kInt`) result.
- So a decode step with one token (M = 1) cannot use `_int_mm` on CUDA; weight-only kernels (`_weight_int8pack_mm`, `_weight_int4pack_mm`, D6) are what serve it. This is a direct bridge for the reader: the quantised GEMV is a different kernel from the quantised GEMM.

- https://docs.nvidia.com/cuda/cublas/index.html (`cublasGemmEx`)
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Blas.cpp
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml

---

## Group B: HIP / ROCm

### B1. `__dp4a` compiles under HIP unchanged

**Verdict: WRONG.** HIP has no `__dp4a`. It has an undocumented header function, `amd_mixed_dot`, and compiler builtins that need the right target.

- Searched all 45 headers in `projects/clr/hipamd/include/hip/amd_detail/` at tag `therock-10.2` for `dp4a` and `__dp2a`: no hits. GitHub code search of `ROCm/rocm-systems` for `__dp4a`: no hits.
- HIPIFY's "CUDA DEVICE API supported by HIP" table (`amd-develop`, last commit 2026-09-22) does not list `__dp4a` or `__dp2a_*` at all (neither mapped nor marked unsupported). It lists `__byte_perm` → `__byte_perm` (HIP 1.6.0) and `__vsub4`, `__vsubss4`, ... with no HIP name (B5).
- What HIP does have (`amd_detail/amd_math_functions.h` at the tag, lines 34 to 60): `inline int amd_mixed_dot(char4 a, char4 b, int c, bool saturate) { return __ockl_sdot4(...); }`, `inline uint amd_mixed_dot(uchar4 a, uchar4 b, uint c, bool saturate) { return __ockl_udot4(...); }`, `inline int amd_mixed_dot(short2 a, short2 b, int c, bool saturate)` (`__ockl_sdot2`), and `inline int amd_mixed_dot(int a, int b, int c, bool saturate) { return __ockl_sdot8(a, b, c, saturate); }`, an eight-way 4-bit dot (B2). The `saturate` flag clamps instead of wrapping.
- COULD NOT VERIFY: any AMD prose documentation of `amd_mixed_dot` (not found in the HIP `.rst`/`.md` sources by code search, nor by a web search of rocm.docs.amd.com). It is exercised by AMD's tests (`hip-tests/catch/unit/deviceLib/hipTestDotFunctions.cc`).
- llama.cpp's portable wrapper (`ggml/src/ggml-cuda/common.cuh`, lines 772 to 812): `static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) { #if defined(GGML_USE_HIP) #if defined(CDNA) || defined(RDNA2) || defined(__gfx906__) c = __builtin_amdgcn_sdot4(a, b, c, false); #elif defined(RDNA3) || defined(RDNA4_M) || defined(RDNA4) c = __builtin_amdgcn_sudot4( true, a, true, b, c, false); #elif defined(RDNA1) || defined(__gfx900__)` (inline assembly with `v_mul_i32_i24` and `v_add3_u32`) `#else` (four products on an `int8x4_t`) ... `#else // defined(GGML_USE_HIP) #if __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A || defined(GGML_USE_MUSA) return __dp4a(a, b, c); #else` (four byte products).

For the course (ours): this is a new entry for the CLAUDE.md list of things that are not "one `.cu` for HIP unchanged". A lab `.cu` that wants a byte dot product needs either a plain four-product helper (portable, also the emulator path) or an `#if` ladder like llama.cpp's.

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_math_functions.h
- https://github.com/ROCm/rocm-systems/tree/therock-10.2/projects/clr/hipamd/include/hip/amd_detail
- https://github.com/ROCm/HIPIFY/blob/amd-develop/docs/reference/tables/CUDA_Device_API_supported_by_HIP.md
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/common.cuh

### B2. A signed int8 four-way dot (dp4a's counterpart) exists on every current AMD GPU

**Verdict: PARTLY.** The compiler defines three families of integer dot instructions, and the signed-only one is missing on RDNA3 and RDNA4 (they have a mixed-sign form instead) and on gfx900 and gfx1010. Every statement below is from the compiler's own target description; AMD's prose docs for these builtins were not found (COULD NOT VERIFY a prose source).

- Builtins (`clang/include/clang/Basic/BuiltinsAMDGPU.td`, ROCm `therock-10.2`, lines 457 to 460; same in upstream `main`): `def __builtin_amdgcn_sdot4 : AMDGPUBuiltin<"int(int, int, int, _Constant bool)", [Const], "dot1-insts">;`, `def __builtin_amdgcn_udot4 : ... "dot7-insts">;`, `def __builtin_amdgcn_sudot4 : AMDGPUBuiltin<"int(_Constant bool, int, _Constant bool, int, int, _Constant bool)", [Const], "dot8-insts">;`, `def __builtin_amdgcn_sdot8 : ... "dot1-insts">;` (and upstream `udot8` "dot7-insts", `sudot8` "dot8-insts").
- Features (ROCm `llvm/lib/Target/AMDGPU/AMDGPU.td` at the tag, lines 895 to 925; upstream moved them to `GCNFeatures.td`): `"dot1-insts", "Has v_dot4_i32_i8 and v_dot8_i32_i4 instructions"`; `"dot7-insts", "Has v_dot4_u32_u8, v_dot8_u32_u4 instructions"`; `"dot8-insts", "Has v_dot4_i32_iu8, v_dot8_i32_iu4 instructions"`.
- Which ISA versions carry them (our parse of the same file; ROCm tag and upstream agree): `Dot1Insts` and `Dot7Insts` in `FeatureISAVersion9_0_6` (gfx906), `9_0_MI_Common` (gfx908, gfx90a), `9_4_Common` (gfx942, gfx950), `10_1_1`, `10_1_2`, `10_3_0` (gfx1030, RDNA2); `Dot7Insts` and `Dot8Insts` but not `Dot1Insts` in `11_Common` (gfx11, RDNA3), `12` (gfx12, RDNA4), `12_50_Common`, `13`; none in `9_0_0` (gfx900) or `10_1_0` (gfx1010).
- Calling the wrong one is a compile error, not a slow path. Clang's own test (`clang/test/CodeGenOpenCL/builtins-amdgcn-dl-insts-err.cl`, run for gfx900 and gfx1010): "'__builtin_amdgcn_sdot4' needs target feature dot1-insts".
- llama.cpp's ladder (B1) matches this exactly: `sdot4` on CDNA, RDNA2, gfx906; `sudot4(true, a, true, b, ...)` (both operands signed, through the mixed-sign instruction) on RDNA3 and RDNA4; assembly on RDNA1 and gfx900.
- Wavefront width does not matter here: these are per-lane VALU operations.

- https://github.com/ROCm/llvm-project/blob/therock-10.2/clang/include/clang/Basic/BuiltinsAMDGPU.td
- https://github.com/ROCm/llvm-project/blob/therock-10.2/llvm/lib/Target/AMDGPU/AMDGPU.td
- https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNFeatures.td
- https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/GCNProcessors.td
- https://github.com/llvm/llvm-project/blob/main/clang/test/CodeGenOpenCL/builtins-amdgcn-dl-insts-err.cl

### B3. AMD's integer matrix instructions (counterpart of A4)

**Verdict: CONFIRMED.** Spelling conventions are in ch15 B4; the integer forms, from `BuiltinsAMDGPU.td` at `therock-10.2`:

- CDNA (`"mai-insts"`): `__builtin_amdgcn_mfma_i32_32x32x4i8`, `_16x16x4i8`, `_4x4x4i8`, `_32x32x8i8`, `_16x16x16i8`; `__builtin_amdgcn_mfma_i32_16x16x32_i8` and `_32x32x16_i8` taking `int64_t` operands (line 674 on); upstream also has `_16x16x64_i8` and `_32x32x32_i8`. All accumulate into `int` vectors. No 4-bit integer MFMA appears.
- RDNA3 (`wmma-256b-insts`): `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` and `_iu4_w32` (and `_w64`), with a `_Constant bool` per operand for signedness. RDNA4 (`wmma-128b-insts`): `_iu8_w32_gfx12`, `_iu4_w32_gfx12`, `__builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12`.
- So on AMD, 4-bit integer matrix multiply exists on RDNA3/4 (WMMA) but not in CDNA's MFMA list, the reverse of a reader's likely guess. As in A4, X2 keeps matrix instructions out of the shared `.cu`.

- https://github.com/ROCm/llvm-project/blob/therock-10.2/clang/include/clang/Basic/BuiltinsAMDGPU.td
- https://github.com/llvm/llvm-project/blob/main/clang/include/clang/Basic/BuiltinsAMDGPU.td

### B4. Narrow-float dot products on AMD accumulate in float

**Verdict: CONFIRMED.** Unlike CUDA (A3), HIP exposes a per-lane half dot with a float accumulator.

- `amd_detail/amd_hip_fp16.h` (`therock-10.2`, line 983): `inline __device__ float amd_mixed_dot(__half2 a, __half2 b, float c, bool saturate) { return __ockl_fdot2(static_cast<__half2_raw>(a).data, static_cast<__half2_raw>(b).data, c, saturate); }`.
- FP8 dot with float result: `def __builtin_amdgcn_dot4_f32_fp8_fp8 : AMDGPUBuiltin<"float(unsigned int, unsigned int, float)", [Const], "dot11-insts">;` (also `_fp8_bf8`, `_bf8_fp8`, `_bf8_bf8`), feature `"dot11-insts", "Has v_dot4_f32_fp8_fp8, v_dot4_f32_fp8_bf8, v_dot4_f32_bf8_fp8, v_dot4_f32_bf8_bf8 instructions"`, present in `FeatureISAVersion11_7_Common` and `FeatureISAVersion12` (upstream `main`; our parse).

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_hip_fp16.h
- https://github.com/llvm/llvm-project/blob/main/clang/include/clang/Basic/BuiltinsAMDGPU.td

### B5. CUDA's SIMD byte intrinsics (`__vsubss4` and friends) have no HIP spelling; llama.cpp writes its own

**Verdict: CONFIRMED.**

- HIPIFY device table: `__vsub4`, `__vsubss4`, `__vsubus4`, ... have empty HIP columns; `__byte_perm` maps to `__byte_perm`.
- llama.cpp `ggml/src/ggml-cuda/vendors/hip.h` (last commit 2026-11-16, lines 260 to 310): `typedef int8_t int8x4_t __attribute__((ext_vector_type(4)));` then `static __device__ __forceinline__ int __vsubss4(const int a, const int b)` built on `__builtin_elementwise_sub_sat(va, vb)`, plus `__vsub4`, `__vcmpeq4`, `__vcmpne4`.
- llama.cpp's 4-bit table lookups use `__byte_perm` on CUDA ("__byte_perm is an instruction that selects bytes with 3 bit indices that can be used instead", `vecdotq.cuh` line 59 on) and a separate HIP branch.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vendors/hip.h
- https://github.com/ROCm/HIPIFY/blob/amd-develop/docs/reference/tables/CUDA_Device_API_supported_by_HIP.md

---

## Group C: Metal

### C1. MSL has an integer dot product (a `dp4a` counterpart)

**Verdict: WRONG.** MSL's only `dot` is the geometric function on floating-point vectors, and for `half` vectors it returns `half`.

- MSL spec 4.1, section 6.9 "Geometric Functions": "T is a vector floating-point type (floatn or halfn). Ts refers to the corresponding scalar type. (If T is floatn , the scalar type Ts is float. If T is halfn , Ts is half.)" Table 6.11: "Ts dot(T x, T y) Return the dot product of x and y, which is x[0] * y[0] + x[1] * y[1] + …".
- Section 6.4, Table 6.2 "Integer functions in the Metal standard library" lists `abs`, `absdiff`, `addsat`, `clamp`, `clz`, `ctz`, `deinterleave` (Metal 4.1), `extract_bits`, `hadd`, `insert_bits`, `interleave` (Metal 4.1), `mad24`, `madhi`, `madsat`, `max`, `max3`, `median3`, `min`, `min3`, `mul24`, `mulhi`, `popcount`, `reverse_bits`, `rhadd`, `rotate`, `subsat`. No dot product.
- Useful for unpacking nibbles: `T extract_bits(T x, uint offset, uint bits)` "All OS: Metal 1.2 and later." "For unsigned data types, the most significant bits of the result are set to zero. For signed data types, the most significant bits are set to the value of bit offset+bits-1." So `extract_bits(as_type<char>(b), 4, 4)` would sign-extend a nibble (our reading).
- Consequence for "where the accumulation must stay wide" (ours): `dot(half4, half4)` accumulates and returns in half. A Metal kernel that dequantises to `half` and calls `dot` has the FP16-accumulator problem of appendix A; convert to `float4` first, as llama.cpp does (`dot(float4(xb4[i]), float4(yl4[i]))`, `mul_mv.metal` line 1112).
- What Metal does have for integer products is at SIMD-group or tensor scope (C3), not per thread. Apple documents no integer dot instruction in its GPUs (searched the MSL spec and the Feature Set Tables for "dot"); nor does it say what hardware executes `matmul2d` (infra C10).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 6.4 and 6.9, pages 202 to 204 and 214)

### C2. MSL now has 4-bit integer packed types: `int4b_format` and `uint4b_format`

**Verdict: PARTLY.** New since appA C4 (which covered only the floating formats): MSL defines signed and unsigned 4-bit (and 2-bit) integer formats with `unpack` to float, half or bfloat. Not documented: the order of the two nibbles within a byte, and which GPU families support the feature.

- MSL spec 2.21 "Packed Numeric Type": "All OS: Metal 4.0 with SDK 26.4 or later support packed numeric. All OS: Metal 4.1 extends the number of format types." "The header <metal_packed_numeric> defines format types and the template packed_numeric_type<Format, N>. See the Metal Feature Set Tables to determine which GPUs support this feature." Table 2.14: "int4b_format 4-bit signed integer. All OS: Metal 4.0 SDK26.4"; "uint4b_format 4-bit unsigned integer. All OS: Metal 4.0 SDK26.4"; `int2b_format`, `uint2b_format` "All OS: Metal 4.1".
- Table 2.17: `int4b_format` N "2, 4, …, 32 (Multiples of 2)"; "N must be in [1,32], with the total bit count (N * bits per format) divisible by 8."
- Conversion (Metal 4.1): `template <class T, class Format, size_t N> vec<T, N> unpack(packed_numeric_type<Format, N> v);` with "T is the output floating-point type that can be float, half, or bfloat"; `pack<Format, Rm, Sm>(vec<T, N> v)`; and `extract<K>(ushort start_pos = 0)`. Table 2.19: `int4b_format` and `uint4b_format` N "8, 16"; `char`, `uchar` N "4, 8". Table 2.20 defaults: `int4b_format` "toward_zero" / "saturate" (the floating formats default to "to_nearest_even").
- Tensor element type (Table 2.24): `int4b_format` "4-bits", minimum 2 elements per get/set, minimum block "32 elements".
- Host side (`Metal.framework/Headers/MTLTensor.h`, macOS 27.0 SDK, lines 51 to 54): `MTLTensorDataTypeInt4 API_AVAILABLE(macos(26.4), ios(26.4)) = 143`, `MTLTensorDataTypeUInt4 ... = 144`. Feature Set Tables (October 23, 2026), "Tensor limits by data type": "Int4 1 … 16 32 elements 256 elements 2 elements 128 bytes Yes" (last column "Supports block scaling").
- COULD NOT VERIFY: whether element 0 of a packed pair is the low or the high nibble (the spec gives no bit order; searched "least significant", "nibble", "bit order"); which Apple GPU families support packed numeric types (the spec defers to the Feature Set Tables, which have no packed-numeric row; the "Tensors" row says "Metal 4 | Apple7"). A lab that relies on `unpack` must test the nibble order on the device against a CPU pack.
- Neither MLX nor llama.cpp uses `int4b_format` today (GitHub code search of both default branches; a search, not a proof). Both unpack with masks and shifts by hand (D4, D5).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 2.21 and 2.22, Tables 2.14 to 2.20 and 2.24, pages 79 to 87)
- https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf ("Tensor limits by data type")
- Local header: `Metal.framework/Headers/MTLTensor.h` (macOS 27.0 SDK)

### C3. Metal's integer matrix route: MPP `matmul2d` with int8 and int4 operands, and only power-of-two block scales

**Verdict: PARTLY.** Integer and mixed integer/half matrix multiplies are documented; block scales with arbitrary float values (llama.cpp's and MLX's) are not, so Metal's tensor path cannot apply a Q4_0 or affine scale inside the multiply.

- MSL spec Table 7.3 "MatMul2D data type supported" (pages 351 to 353), as printed: "Metal 4 char char int"; "Metal 4 char half half", "char half float", "half char half", "half char float"; "Metal 4 and OS 26.4 uchar uchar int"; "Metal 4 and OS 26.4 half int4b_format half", "half int4b_format float", "half uint4b_format half", "half uint4b_format float"; "Metal 4 and OS 26.4 char int4b_format int", "uchar uint4b_format int"; "bfloat int4b_format bfloat/float"; "Metal 4.1 char int2b_format int", ... So a half activation times a 4-bit weight, accumulating into float, is a documented combination.
- Block scales: `tensor_blockwise` "ElementType Specifies the qualified element type of this plane. Only metal_fp8_ue8m0_format is supported." "BlockSizes… ... The first dimension must be 32. All other dimensions must be 1." (section 2.22.2.2; appA C4). The Feature Set Tables mark Int4 "Supports block scaling: Yes", which with UE8M0 means power-of-two scales only (our reading of the two documents together).
- Not documented (COULD NOT VERIFY): what hardware executes these, and any GPU-family requirement beyond "Tensors | Metal 4 | Apple7".

For X2 (ours): the labs do the scales themselves in the kernel. MPP's `half × int4b_format` is a `<Today>` note pointing at what Apple added in OS 26.4, not a lab path, because it needs Metal 4 tensors and its nibble order is undocumented (C2).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 7.3; section 2.22.2.2, Table 2.28)
- https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf

### C4. MPSGraph's dequantise ops: `scale * (q - zeroPoint)`, blockwise int4 and int8 from macOS 15

**Verdict: CONFIRMED.** A third convention for the asymmetric case, next to llama.cpp's `d·q + m` and MLX's `s·q + bias` (D1, D5).

- `MPSGraphQuantizationOps.h` (macOS 27.0 SDK), `dequantizeTensor:scaleTensor:zeroPointTensor:dataType:name:`, `MPS_AVAILABLE_STARTING(macos(15.0), ios(18.0), macCatalyst(18.0), tvos(18.0))`: "Convert the quantized `tensor` to a float tensor by applying a scale and bias transform: result = scaleTensor * (tensor - zeroPointTensor)". "**Regular blockwise** (`scaleTensor` type is an MPSGraph float type): - Supported input types: MPSDataTypeInt4, MPSDataTypeUInt4, MPSDataTypeInt8, MPSDataTypeUInt8, MPSDataTypeFloat8E4M3, MPSDataTypeFloat8E5M2. - `zeroPointTensor` type must match the input `tensor` type. - `scaleTensor` shape: leading dims of `tensor` with the last dim replaced by `tensor.shape[last] / blockSize`." Plus "**MX mode**" (E8M0 scale, "Block size 32, symmetric (zero point must be 0)") and "F4 with F8E4M3 scale" ("Block size 16").
- Per-axis forms (`quantizeTensor:scaleTensor:zeroPoint:dataType:axis:name:` and friends) are `macos(13.1)`; a lookup-table dequantise (`dequantizeTensor:LUTTensor:name:`) is `macos(15.0)`.
- `MPSCoreTypes.h`: `MPSDataTypeInt4 MPS_ENUM_AVAILABLE_STARTING( macos(15.0), ...) = MPSDataTypeSignedBit | 4`, `MPSDataTypeUInt4 ... = 4`, `MPSDataTypeInt2 ... macos(15.4)`.

- Local headers: `MetalPerformanceShadersGraph.framework/Headers/MPSGraphQuantizationOps.h`, `MetalPerformanceShaders.framework/Frameworks/MPSCore.framework/Headers/MPSCoreTypes.h` (macOS 27.0 SDK)

### C5. Reading a Q4_0-style block on Metal: `half` is 2-byte aligned, and kernels read the quants as 16-bit words

**Verdict: CONFIRMED.**

- MSL spec Table 2.2: `half` size 2, alignment 2; `short`/`ushort` 2 and 2; `int`/`uint` 4 and 4. So a struct of one `half` and 16 `uchar` is 18 bytes with alignment 2, the same trap as A5.
- llama.cpp declares the block once for every backend with `typedef half ggml_half;` under `GGML_COMMON_DECL_METAL` (`ggml-common.h`), and its Metal kernels read the quants as `device const uint16_t * qs = ((device const uint16_t *) qb_curr + 1 + il/2);` (`mul_mv.metal` line 90) and `((device const uint16_t *)xb + 1)` (`dequantize.h` line 133).
- `as_type<T>` reinterprets bits between same-size types ("Using the as_type<type-id> operator to reinterpret data to a type with a different number of bytes results in an error.", page 110), so a `ushort` holding an FP16 scale becomes `half` with `as_type<half>(u)` (our use).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.2, page 26; page 110)

---

## Group D: llama.cpp, MLX and PyTorch (read today; dates are last commits to each file)

### D1. llama.cpp Q8_0, Q4_0, Q4_1 and Q8_1: block layouts and reference (de)quantisation

**Verdict: CONFIRMED.** `ggml/src/ggml-common.h` (last commit 2026-08-31) and `ggml/src/ggml-quants.c` (2026-11-19). This is the layout lab X2.2 should copy, and say whose it is.

Layouts (`ggml-common.h`, lines 194 to 269; `ggml_half` is `uint16_t` in C and C++, `half` on Metal):

- `#define QK4_0 32` `typedef struct { ggml_half d; // delta uint8_t qs[QK4_0 / 2]; // nibbles / quants } block_q4_0;` with `static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2, ...)`: 18 bytes per 32 weights.
- `#define QK4_1 32`: `ggml_half d; // delta`, `ggml_half m; // min` (as a union with `ggml_half2 dm`), then `uint8_t qs[QK4_1 / 2]`: 20 bytes.
- `#define QK8_0 32` `typedef struct { ggml_half d; // delta int8_t qs[QK8_0]; // quants } block_q8_0;`: 34 bytes.
- `#define QK8_1 32`: `ggml_half d; // delta`, `ggml_half s; // d * sum(qs[i])`, `int8_t qs[QK8_1];`: 36 bytes. Q8_1 is the activation format of the CUDA decode path (D3).
- The Python package `gguf-py` (`gguf/constants.py`) agrees: `GGMLQuantizationType.Q4_0: (32, 2 + 16)`, `Q4_1: (32, 2 + 2 + 16)`, `Q8_0: (32, 2 + 32)`, `Q8_1: (32, 2 + 2 + 32)`.

Reference quantisers (`ggml-quants.c`; "reference implementation for deterministic creation of model files"):

- Q8_0 (`quantize_row_q8_0_ref`, line 273): `const float d = amax / ((1 << 7) - 1);` `const float id = d ? 1.0f/d : 0.0f;` `y[i].d = GGML_FP32_TO_FP16(d);` `y[i].qs[j] = roundf(x0);`. Symmetric, absmax per 32, codes in [-127, 127], scale stored as FP16.
- Q4_0 (`quantize_row_q4_0_ref`, line 110): finds the value of largest magnitude *with its sign* (`if (amax < fabsf(v)) { amax = fabsf(v); max = v; }`), then `const float d = max / -8;` and `const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));` `const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));` `y[i].qs[j] = xi0; y[i].qs[j] |= xi1 << 4;` where `x0 = x[i*qk + 0 + j]*id` and `x1 = x[i*qk + qk/2 + j]*id` for `j < qk/2`.
- Q4_1 (`quantize_row_q4_1_ref`, line 147): `const float d = (max - min) / ((1 << 4) - 1);` `y[i].m = GGML_FP32_TO_FP16(min);` and `MIN(15, (int8_t)(x0 + 0.5f))` with `x0 = (x - min)*id`. Asymmetric, min as the offset.
- Dequantisers: Q4_0 (`dequantize_row_q4_0`, line 462): `const int x0 = (x[i].qs[j] & 0x0F) - 8;` `const int x1 = (x[i].qs[j] >> 4) - 8;` `y[i*qk + j + 0 ] = x0*d;` `y[i*qk + j + qk/2] = x1*d;`. Q4_1: `x0*d + m`. Q8_0: `x[i].qs[j]*d`.

Our reading, for the lessons and the lab:

- **Nibble order.** Byte j of a Q4_0 block holds weight j in its low nibble and weight j + 16 in its high nibble. It is not "two neighbours per byte". MLX packs neighbours (D5). A lab that says "llama.cpp's Q4_0" must use this order.
- **Q4_0 is symmetric but not ±7.** The signed extreme lands exactly on code 0 (value -8·d = max), and the codes run -8..7. A value of the opposite sign and the same magnitude would need +8 and is clipped to +7 (`x0 + 8.5` reaches 16.5, truncates to 16, `MIN` gives 15). One step of clipping on one side buys the 16th level.
- **Rounding.** Q8_0 uses `roundf` (half away from zero); Q4_0 adds 8.5 and truncates toward zero through `(int8_t)`, which for these non-negative sums is round half up.
- **Importance weights.** With an importance matrix (`quantize_q4_0` with non-null `quant_weights`, line 2138), llama.cpp calls `quantize_row_q4_0_impl`, which chooses `d` by `make_qx_quants(QK4_0, 8, xb, L, 1, weight)` instead of the rule above. The block layout is the same; only the choice of scale differs. The lab's CPU reference should implement the `_ref` rule and say so.
- `gguf-py/gguf/quants.py` (last commit 2026-10-29) has NumPy versions of the same rules (`class Q4_0`, line 222: `d = max / -8`, `qs = np.trunc((blocks * id) + np.float32(8.5), ...).clip(0, 15)`, `qs[..., 0, :] | (qs[..., 1, :] << np.uint8(4))`; `class Q8_0`, line 380). It is a usable oracle for the TypeScript simulator's unit tests (D7).

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c
- https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py (`GGML_QUANT_SIZES`, line 7469)
- https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/quants.py

### D2. llama.cpp Q4_K: 256-weight super-blocks, eight 32-weight sub-blocks, 6-bit scales and mins

**Verdict: CONFIRMED.** `ggml-common.h` lines 322 to 338 and `ggml-quants.c` lines 889 to 897 and 1539 to 1560.

- `#define QK_K 256`, `#define K_SCALE_SIZE 12`. "// 4-bit quantization // 8 blocks of 32 elements each // weight is represented as x = a * q + b // Effectively 4.5 bits per weight". Struct: `ggml_half d; // super-block scale for quantized scales`, `ggml_half dmin; // super-block scale for quantized mins`, `uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits`, `uint8_t qs[QK_K/2]; // 4--bit quants`. 4 + 12 + 128 = 144 bytes per 256 weights.
- Unpacking the 6-bit pairs: `static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) { if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; } else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4); *m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4); } }`.
- Dequantise: for each 64 weights, `const float d1 = d * sc; const float m1 = min * m;` (sub-block 2k) and `d2`, `m2` (sub-block 2k+1); `*y++ = d1 * (q[l] & 0xF) - m1;` for `l < 32`, then `*y++ = d2 * (q[l] >> 4) - m2;`. So inside each 64-weight chunk the low nibbles of 32 bytes are the first 32 weights and the high nibbles the next 32 (another nibble order, our reading).
- Two-level scaling is the point for the lesson (ours): the per-32 scales cost 6 bits each instead of 16, which is how Q4_K keeps 4.5 bits per weight with a scale and a min per 32 weights, against Q4_1's 5.0.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c

### D3. llama.cpp's CUDA/HIP decode path: activations quantised to Q8_1, integer dot per block, one float rescale per block

**Verdict: CONFIRMED** (source read; the arithmetic restatement is ours).

- `ggml_cuda_mul_mat_vec_q` (`mmvq.cu`, last commit 2026-11-20, lines 1600 to 1607) allocates `src1_q8_1` and calls `quantize_row_q8_1_cuda(src1_d, src1_q8_1.get(), ...)` before the kernel. So the float activation vector is itself quantised to int8 per 32 values for every token.
- That quantiser (`quantize.cu`, last commit 2026-11-17, `static __global__ void quantize_q8_1`, line 54): one thread per value, `amax = warp_reduce_max<QK8_1>(amax);` `const float d = amax / 127.0f;` `const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);` `const int sum = warp_reduce_sum<QK8_1>((int) q);` then stores `d` and `sum*d` (the `s` field). A warp-reduction application of chapter 10 (32-wide groups regardless of the hardware warp).
- The block dot (`vecdotq.cuh`, lines 115 to 136): `const int vi0 = (v[i] >> 0) & 0x0F0F0F0F; const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;` `// SIMD dot product of quantized values` `sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi); sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);` then `// second part effectively subtracts 8 from each quant value` `return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);`. The unsigned nibbles 0..15 go into `dp4a` against signed int8 activations; the "minus 8" is applied once per block through the precomputed sum `s`. Q8_0 × Q8_1: `sumi = ggml_cuda_dp4a(v[i], u[i], sumi);` `return d8_0*d8_1 * ((T) sumi);` (lines 243 to 258).
- Mask `0x0F0F0F0F` on one 32-bit word yields four low nibbles (weights j..j+3) and `>> 4` the four high nibbles (weights j+16..j+19), which pair with activation words `iqs + i` and `iqs + i + QI4_0` (`vec_dot_q4_0_q8_1`, line 800 on): the j / j+16 layout of D1 is what makes this a single mask.
- Accumulation width: the integer sum stays `int` within a block; each block's result is a `float`, summed into `float tmp[ncols_dst][rows_per_cuda_block]` across blocks (`mul_mat_vec_q`, line 672 on), then reduced across warps in shared memory.
- The dispatch for `dp4a` on each vendor and architecture: B1.

For the lessons (ours): this is "integer dot with a per-block rescale", with three costs a reader should predict: quantising the activations (one extra pass per token), the rounding of `x` to int8, and the FP16 rounding of `s`.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmvq.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/quantize.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vecdotq.cuh

### D4. llama.cpp's Metal decode path: no integer dot; float multiply-adds on masked nibbles with pre-scaled activations

**Verdict: CONFIRMED.** `ggml/src/ggml-metal/kernels/mul_mv.metal` (last commit 2026-11-13) and `dequantize.h` (2026-10-01).

- `block_q_n_dot_y` for Q4_0 (lines 82 to 104): "// we assume that the yl's have been multiplied with the appropriate scale factor // that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)". Body: `acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F); acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00); acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0); acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);` and `return d * (sumy * -8.f + acc[0] + acc[1] + acc[2] + acc[3]);`.
- The caller (`mul_vec_q_n_f32_impl`, line 219 on): `yl[i + 0] = yb[i + 0]; yl[i + 1] = yb[i + 1]/256.f;` `yl[i + 8] = yb[i + 16]/16.f; yl[i + 9] = yb[i + 17]/4096.f;` and `sumy[0] += yb[i + 0] + yb[i + 1];`. So no shift is executed on the weights: the mask leaves each nibble multiplied by 1, 16, 256 or 4096, and the activation was divided by the same power of two (exact in float). The `-8` offset is folded in once per half-block as `-8 · d · sum(y)`. The row result is `simd_sum(sumf[row])`.
- Q8_0 (`kernel_mul_mv_q8_0_f32_impl`, line 514 on): `sumq += qs[i] * yl[i];` in `float`. Neither path quantises the activations.
- Prefill (`mul_mm.metal`, last commit 2026-11-05): each tile is dequantised with `dequantize_func(x, il, temp_a);` into threadgroup memory and multiplied with `simdgroup_float8x8` accumulators; for Q4_0, `dequantize_q4_0` computes `d1 * (qs[i] & mask0) + md` with `const float md = -8.h * xb->d;` (`dequantize.h` line 132 on). That is "dequantise, then multiply" at tile granularity.
- MLX (D5) and PyTorch MPS (D6) use the same pre-scaling trick, so it is the Metal idiom to teach.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/mul_mv.metal
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/dequantize.h
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/mul_mm.metal

### D5. MLX affine group quantisation: what the docs say, what the code does, and how its GEMV uses it

**Verdict: PARTLY.** The documented formula (scale from max minus min, bias = min) is not what the implementation computes: the code makes the larger-magnitude edge and zero exactly representable, and the scale may be negative.

Docs (`mlx.core.quantize`, MLX 0.32.4; docstring in `python/src/ops.cpp`, last commit 2026-11-20, lines 4880 to 4925):

- Signature `quantize(w, /, group_size: int | None = None, bits: int | None = None, mode: str = 'affine', *, global_scale: array | None = None, stream: StreamOrDevice = None) -> tuple[array, array, array]`. Table: "affine 32, 64*, 128 1, 2, 3, 4*, 5, 6, 8 same as input yes" ("* indicates the default value when unspecified").
- "The "affine" mode quantizes groups of g consecutive elements in a row of w." α = max w_i, β = min w_i, s = (α − β)/(2^b − 1), ŵ_i = round((w_i − β)/s). "After the above computation, ŵ_i fits in b bits and is packed in an unsigned 32-bit integer from the lower to upper bits. For instance, for 4-bit quantization we fit 8 elements in an unsigned 32 bit integer where the 1st element occupies the 4 least significant bits, the 2nd bits 4-7 etc." "To dequantize the elements of w, we also save s and β which are the returned scales and biases respectively."

Code (`mlx/ops.cpp`, last commit 2026-10-31, `affine_quantize`, line 5209 on; the Metal kernel `affine_quantize` in `kernels/quantized.h`, last commit 2026-11-02, line 2909 on, does the same):

- Group sizes and bits are enforced: "The supported group sizes are 32, 64, and 128." and `if (bits < 1 || bits > 8 || bits == 7)` "The supported bits are 1, 2, 3, 4, 5, 6 and 8."
- Kernel: `scale = max((w_max - w_min) / n_bins, eps); bool side = abs(w_min) > abs(w_max); scale = side ? scale : -scale; float edge = side ? w_min : w_max; float q0 = round(edge / scale); bool at_zero = q0 == 0.0f; scale = at_zero ? scale : edge / q0; bias = at_zero ? 0 : edge;` then `uint8_t val = min(round((w_thread[i] - bias) / scale), n_bins);`.
- Our reading: the bias is the extreme with the larger magnitude (not always the min), the scale is negative when that extreme is the max, and the scale is adjusted to edge / q0 so that code -q0 dequantises to 0 (in exact arithmetic; in float32, fl(edge / q0) · (−q0) + edge can leave a residual of the order of one rounding of the division, as the module's simulator shows: correction 2026-11-20). Dequantised value = scale · q + bias in every case. A lesson should teach the documented min/max form first and show this refinement as what the code does.
- Scales and biases have the input's type ("same as input"); kernels are instantiated for `float`, `float16_t` and `bfloat16_t` (`quantized.metal`, lines 248 to 250). So float32 weights give float32 scales and biases, which avoids `__half` in a portable lab (project note below).

The GEMV (`affine_qmv_fast`, `qmv_fast_impl`, `quantized.h` line 851 on):

- `typedef float U;`: accumulation in float whatever the input type.
- `load_vector` for 4 bits: `sum += U(x[i]) + U(x[i + 1]) + U(x[i + 2]) + U(x[i + 3]); x_thread[i] = x[i]; x_thread[i + 1] = x[i + 1] / 16.0f; x_thread[i + 2] = x[i + 2] / 256.0f; x_thread[i + 3] = x[i + 3] / 4096.0f;`.
- `qdot` for 4 bits: `const device uint16_t* ws = (const device uint16_t*)w; ... accum += (x_thread[4 * i] * (ws[i] & 0x000f) + x_thread[4 * i + 1] * (ws[i] & 0x00f0) + x_thread[4 * i + 2] * (ws[i] & 0x0f00) + x_thread[4 * i + 3] * (ws[i] & 0xf000));` and `return scale * accum + sum * bias;` (line 333). The bias is folded in through the sum of the activations, the asymmetric twin of llama.cpp's `-8 · sum(y)`.
- `mlx.nn.QuantizedLinear` (`python/mlx/nn/layers/quantized.py`, last commit 2026-10-05): "Applies an affine transformation to the input using a quantized weight matrix. It is the quantized equivalent of mlx.nn.Linear", `group_size: int = None, bits: int = None, mode: str = "affine"` (so the defaults are 64 and 4).

- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html
- https://github.com/ml-explore/mlx/blob/main/python/src/ops.cpp
- https://github.com/ml-explore/mlx/blob/main/mlx/ops.cpp
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.h
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/quantized.metal
- https://github.com/ml-explore/mlx/blob/main/python/mlx/nn/layers/quantized.py

### D6. PyTorch: quantisation moved to torchao; the core ops for weight-only int8 and int4 matmul, and where they run

**Verdict: PARTLY.** The move to torchao is confirmed, but PyTorch's own page still promises deletion "in 2.10" while being the 2.15 page.

- Docs 2.15, "Quantization" (last updated May 11, 2026): "We are centralizing all quantization related development to torchao". "We plan to delete torch.ao.quantization in 2.10 if there are no blockers, or in the earliest PyTorch version until all the blockers are cleared." Eager mode: "please migrate to use torchao eager mode quantize_ API instead"; "pt2e quantization has been migrated to torchao (pytorch/ao)". So in lessons, say torchao, not `torch.ao.quantization`.
- torchao (`torchao/quantization/__init__.py` on `main`; latest release v0.18.0): configs include `Int4WeightOnlyConfig`, `Int8WeightOnlyConfig`, `Int8DynamicActivationInt8WeightConfig`, `Float8WeightOnlyConfig`, `Float8DynamicActivationFloat8WeightConfig`, `IntxWeightOnlyConfig`. `Int4WeightOnlyConfig` (`quant_api.py`, last commit 2026-11-18): "Configuration for int4 weight only quantization, only groupwise quantization is supported." "`group_size`: ... choices are [256, 128, 64, 32]", default `group_size: int = 128`; "`int4_tile_packed_ntile`: ntile size for TILED_PACKED_TO_4D format, default is 8 for CUDA platform, 16 for ROCm platform".
- Core ops (`native_functions.yaml`, last commit 2026-11-20): `_weight_int4pack_mm(Tensor self, Tensor mat2, int qGroupSize, Tensor qScaleAndZeros)` dispatches `MPS`, `CUDA`, `XPU` (the CPU version is a separate op, `_weight_int4pack_mm_for_cpu`); `_weight_int8pack_mm(Tensor self, Tensor mat2, Tensor scales)` dispatches `CPU`, `CUDA`, `MPS`, `XPU`; `_convert_weight_to_int4pack(Tensor self, int innerKTiles)` `CUDA`, `MPS`, `XPU`.
- CUDA int4 kernel (`aten/src/ATen/native/cuda/int4mm.cu`, last commit 2026-11-19): guarded by `#if defined(USE_ROCM) || (defined(CUDA_VERSION) && (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)))`; "De-quantize int4 values to bf16. Values are dequantized as truly int4 [-8, 7] range; dequant = (bf16(int4_value) * bf16_scale) + bf16_zero"; group sizes `qGroupSize == 32 || qGroupSize == 64 || qGroupSize == 128 || qGroupSize == 256`. So PyTorch's zero point is a float added after scaling (a fourth convention, next to D1, D5 and C4), and the CUDA kernel needs compute capability 8.0 or a ROCm build.
- MPS int4 kernel (`aten/src/ATen/native/mps/kernels/Quantized.metal`, last commit 2026-10-13, `int4pack_mm`, line 89 on): `float4 act_div_scales = {1.f, 1 / 16.f, 1 / 256.f, 1 / 4096.f};`, `const T zero0 = scales_and_zeros[zeros_gruop_offset] - scale0 * T(8);` ("// Adding zero point results in 10% perf penalty."), `result += a_val_sum * zeros;`. The same three tricks as llama.cpp and MLX on Metal.
- `torch._int_mm` and its M > 16 rule on CUDA: A6.

- https://docs.pytorch.org/docs/2.15/quantization.html
- https://github.com/pytorch/ao/blob/main/torchao/quantization/__init__.py
- https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_api.py
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/int4mm.cu
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/kernels/Quantized.metal

### D7. A reference to test the simulator and the labs against

**Verdict: CONFIRMED.** For the lessons' tested models: llama.cpp's `gguf-py` NumPy quantisers (D1) reproduce Q4_0 and Q8_0 bit for bit by their own claim of being the file-format reference, and MLX's documented affine formula (D5) is a closed form. A TypeScript simulator should implement (a) Q8_0 `_ref`, (b) Q4_0 `_ref` with the j / j+16 nibble order, (c) MLX affine with the documented min/max scale and, as an option, the code's edge-and-zero refinement, and test its packed bytes for a fixed input against values produced by `gguf-py` (generated once, stored as a fixture; our recommendation). FP16 rounding of the scale must be modelled (both llama.cpp formats store `d` as FP16; appA A2 has the format).

- https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/quants.py

---

## Group E: our arithmetic (not vendor facts)

### E1. Bytes per weight, and the GEMV's intensity as a function of them

**Verdict: CONFIRMED** (arithmetic on the layouts above).

A matrix-vector product does one multiply and one add per weight, 2 FLOPs, and each weight is read once. Ignoring the activation vector and the output (both are O(in + out) against O(in × out) weights), the intensity is 2 / (bytes per weight) FLOP/B:

| Format (owner) | Block | Bytes per block | Bytes per weight | Bits per weight | GEMV FLOP/B |
|---|---|---|---|---|---|
| FP32 | 1 | 4 | 4 | 32 | 0.5 |
| FP16 / BF16 | 1 | 2 | 2 | 16 | 1 |
| Q8_0 (llama.cpp) | 32 | 34 | 1.0625 | 8.5 | 1.88 |
| Q4_0 (llama.cpp) | 32 | 18 | 0.5625 | 4.5 | 3.56 |
| Q4_1 (llama.cpp) | 32 | 20 | 0.625 | 5.0 | 3.2 |
| Q4_K (llama.cpp) | 256 | 144 | 0.5625 | 4.5 | 3.56 |
| MXFP4 (OCP; llama.cpp `block_mxfp4`) | 32 | 17 | 0.53125 | 4.25 | 3.76 |
| affine 4-bit, g = 32, FP16 scale and bias (MLX) | 32 | 20 | 0.625 | 5.0 | 3.2 |
| affine 4-bit, g = 64, FP16 scale and bias (MLX default) | 64 | 36 | 0.5625 | 4.5 | 3.56 |
| affine 4-bit, g = 128, FP16 scale and bias (MLX) | 128 | 68 | 0.53125 | 4.25 | 3.76 |
| affine 4-bit, g = 64, FP32 scale and bias (MLX with float32 input) | 64 | 40 | 0.625 | 5.0 | 3.2 |

Every format here is far below any GPU's ridge point (ch15 A1 for the book's example), so decode stays bandwidth-bound and the predicted speed-up over FP32 is the ratio of bytes moved: 3.76 for Q8_0, 7.11 for Q4_0, in the limit of large matrices and perfect bandwidth use. A1 says how much of that one published measurement kept.

### E2. Quantisation error against block size and bit width, on two weight distributions

**Verdict: CONFIRMED** (our computation; seed 0, 2^20 samples each, `/usr/bin/python3` with NumPy 2.0.2). Relative RMS error ‖w − ŵ‖ / ‖w‖. "Q4_0" is llama.cpp's `_ref` rule with an FP16 scale; "sym4" is a textbook absmax with codes ±7; "affine4" is MLX's documented min/max rule with float scales; Q8_0 is llama.cpp's rule with an FP16 scale.

| Distribution | Block | Q4_0 | sym4 (±7) | affine4 | int8 absmax |
|---|---|---|---|---|---|
| Normal | 32 | 0.0860 | 0.0970 | 0.0782 | 0.0054 |
| Normal | 64 | 0.0950 | 0.1077 | 0.0897 | 0.0059 |
| Normal | 128 | 0.1032 | 0.1174 | 0.0998 | 0.0065 |
| Normal | 256 | 0.1108 | 0.1264 | 0.1091 | 0.0070 |
| Student t, 3 d.o.f. | 32 | 0.1269 | 0.1433 | 0.1013 | 0.0086 |
| Student t, 3 d.o.f. | 64 | 0.1595 | 0.1799 | 0.1329 | 0.0110 |
| Student t, 3 d.o.f. | 128 | 0.1987 | 0.2231 | 0.1722 | 0.0141 |
| Student t, 3 d.o.f. | 256 | 0.2456 | 0.2744 | 0.2159 | 0.0179 |

Affine, g = 64, by bit width (normal / t3): 2 bits 0.4501 / 0.6597; 3 bits 0.1925 / 0.2819; 4 bits 0.0897 / 0.1329; 5 bits 0.0434 / 0.0636; 6 bits 0.0213 / 0.0313; 8 bits 0.0053 / 0.0078. One scale for the whole tensor: int8 0.0113 / 0.2095, int4 0.2060 / 0.9131.

What the explorer should let the reader discover (ours): each extra bit roughly halves the error; smaller blocks help most on heavy tails (outliers set the scale of their block only); Q4_0's 16th level beats a ±7 grid; per-tensor scaling at 4 bits is unusable on heavy tails. These numbers are for checking the simulator's order of magnitude, not exact test values (the simulator's random generator will differ).

### E3. Where the accumulation must stay wide (tie to appendix A)

**Verdict: CONFIRMED** (arithmetic).

- One Q8_0 × Q8_1 block: at most 32 × 127 × 127 = 516,128 in magnitude. That overflows FP16 (largest finite 65,504, appA A2) and exceeds FP16's exact-integer range (2,048), so a per-block integer sum must not be held in `half`. It fits easily in `int32` (2^31 − 1), and it is exact in `float` (below 2^24 = 16,777,216).
- A whole row of int8 × int8 products in `int32` would overflow only after about 133,144 worst-case terms (exactly: 133,144 terms of 127 × 127 fit, the 133,145th overflows; our arithmetic, 2026-11-20); the real kernels never sum that long without a scale, because each block has its own scale: the sum is `Σ_blocks d_w · d_x · (Σ_in-block q_w · q_x)`, integer inside, float outside (D3).
- Q4_0 nibbles (0..15) against int8 (llama.cpp's form, D3): at most 15 × 127 × 32 = 60,960 per block before the −8 correction; signed (−8..7): 8 × 127 × 32 = 32,512.
- Float paths: MLX and llama.cpp accumulate in `float` on Metal (D4, D5); MSL `dot(half4, half4)` returns `half` (C1); CUDA has no half dot and `__hfma2` chains accumulate in half (A3); HIP `amd_mixed_dot(__half2, __half2, float, ...)` accumulates in float (B4). Appendix A's lab already shows what a half accumulator does to a long sum; X2 points back to it.

### E4. Folding the zero point (or the −8) out of the inner loop

**Verdict: CONFIRMED** (algebra; every real kernel above does it).

For one block with scale d and offset b (b = −8d for Q4_0, b = m for Q4_1, b = bias for MLX): Σ_j (d·q_j + b)·x_j = d·Σ_j q_j·x_j + b·Σ_j x_j. The inner loop then touches only q and x; the activation sum Σ x_j is computed once per block and shared by every output row. llama.cpp CUDA precomputes it in Q8_1's `s` (D3), llama.cpp Metal as `sumy` (D4), MLX as `sum` (D5), PyTorch MPS as `a_val_sum` (D6). The pre-scaling trick (divide x by 16, 256, 4096 so masked nibbles need no shift) is exact in binary floating point because the divisors are powers of two; it costs nothing in accuracy unless x is near the bottom of float's range.

---

## Project notes (not vendor facts)

Recorded so lab authors do not have to rediscover them; these describe this repository on the check date.

- The emulators and the front-end stub declare no `char4`/`uchar4`, no `__dp4a`, no `__byte_perm` (searched `labs/common/include/pmpp/`); `emu_types.h` has `int2..4`, `uint2..4`, `float2..4`, `double2..4` only. They do not model `__half` (appA E2). So a `.cu` lab that calls `__dp4a`, uses `char4`, or converts an FP16 scale with `__half2float` cannot run on `emu`/`race` or pass the front-end check without a shared change. Options: (a) a plain helper `dot4_i8(int a, int b, int c)` with four byte products (portable everywhere, and the B1 fallback), with `__dp4a` as an optional `#if` variant measured on CUDA only, reported as a shared change if it is to be front-end-checked; (b) for scales, MLX's layout with float32 scales (D5), or Q4_0 with the FP16 scale stored as `uint16_t` and converted by a small bit routine in the lab (bit layout in appA A2). Report whichever shared change is wanted; do not edit shared files from the chapter build.
- `labs/common/include/pmpp/vendors/hip.h` maps no `__dp4a` (B1).
- Baselines to reuse: `launchGemvGather(const float* X, const float* W, float* Y, int batch, int in, int out)` and `launchGemvSplitK(...)` in `labs/ch22/lab1_gemv/solution/kernel.cu` (FP32 GEMV, W stored (out, in) row-major, same argument order as the project's `launchLinear` of labs 6.4 and 15.6); `launchLinear` in `labs/ch15/lab6_linear/solution/kernel.cu`; `launchKvAppend` and `launchDecodeAttention` in `labs/ch20/lab4_decode/solution/kernel.cu` for M9's decode step. Same row-major (out, in) convention as llama.cpp's weight rows and MLX's `quantized_matmul(..., transpose=True)` default.
- Layout recommendation for lab X2.2 (ours): llama.cpp's Q4_0 exactly as D1 (18-byte blocks, FP16 `d`, j / j+16 nibbles, value (q − 8)·d), because it is the format the reader will meet in GGUF files; or, if FP16 handling is not wanted in the emulator, MLX's affine 4-bit with float32 scales and biases and LSB-first packing (D5). Either way the README names whose layout it is and links the source file.

## Answers to the brief's portability questions

| Question | Answer | Where |
|---|---|---|
| Integer dot instruction on CUDA | `__dp4a` (int, char4, unsigned, uchar4), `__dp2a_lo/hi`; sm_61+; PTX also has mixed signedness | A2 |
| On HIP | No `__dp4a`. `amd_mixed_dot` (undocumented header) or `__builtin_amdgcn_sdot4` (dot1: gfx906, CDNA, RDNA2), `__builtin_amdgcn_sudot4` (dot8: RDNA3, RDNA4); wrong target is a compile error | B1, B2 |
| On Metal | None per thread. MPP `matmul2d` `char × char → int` (Metal 4), `char × int4b_format → int`, `half × int4b_format → float` (OS 26.4) at tensor scope | C1, C3 |
| 4-bit storage types | CUDA: none for integers in C++ (PTX `.s4/.u4` in `mma`, `cvt.pack`); HIP: none; Metal: `int4b_format`, `uint4b_format` (Metal 4.0, SDK 26.4), nibble order undocumented; MPSGraph `MPSDataTypeInt4` (macOS 15) | A4, C2, C4 |
| Narrow-float dot per thread | CUDA: none (`__hfma2` only); HIP: `amd_mixed_dot(__half2, __half2, float)`, FP8 `dot4_f32_*` on gfx11.7/gfx12; Metal: `dot(half4, half4)` returns `half` | A3, B4, C1 |
| Real block layouts | llama.cpp Q8_0 (34 B / 32), Q4_0 (18 B / 32, j / j+16 nibbles, (q − 8)·d), Q4_1, Q4_K (144 B / 256, 6-bit scales and mins); MLX affine (scale, bias per group of 32/64/128, LSB-first uint32) | D1, D2, D5 |
| How decode GEMVs consume them | llama.cpp CUDA/HIP: activations to Q8_1, `dp4a`, one float rescale per block; llama.cpp Metal, MLX, PyTorch MPS: float FMAs on masked nibbles with activations pre-divided by 16/256/4096 and the offset folded through Σx; PyTorch CUDA: bf16 tinygemm, sm_80+ | D3 to D6, E4 |
| int8 GEMM library for batch 1 | Not usable: `torch._int_mm` on CUDA needs M > 16; cuBLAS int8 needs 4-byte alignment and leading dimensions that are multiples of 4 | A6 |

---

## Summary of corrections

Every WRONG and PARTLY item, and every failed or unfound sub-claim:

- A2 (could not verify): no NVIDIA statement that nvcc turns a scalar byte loop into `dp4a`, and no `dp4a` throughput figure. The CUDA intrinsic has no mixed-sign overload; PTX `dp4a.u32.s32` does.
- A4 (PARTLY): integer `mma.sync` covers s8/u8 and s4/u4 from sm_75/sm_80, but Hopper's `wgmma` has s8/u8 only, and Blackwell's `tcgen05.mma .kind::i8` is limited to sm_100a, sm_101a/sm_110a (not the family-portable targets). Which path llama.cpp's `mmq` takes per architecture was not checked.
- B1 (WRONG): `__dp4a` does not exist in HIP (no hit in the 45 `amd_detail` headers at `therock-10.2`; absent from HIPIFY's table). HIP has `amd_mixed_dot` (no AMD prose documentation found) and the `__builtin_amdgcn_*dot*` builtins. Add to CLAUDE.md's "not one `.cu` for HIP unchanged" list.
- B2 (PARTLY): the signed int8 dot (`v_dot4_i32_i8`, dot1) exists on gfx906, CDNA (gfx908, gfx90a, gfx94x, gfx950) and RDNA2 (gfx103x), but not on RDNA3/RDNA4, which have the mixed-sign `v_dot4_i32_iu8` (dot8) instead, nor on gfx900 or gfx1010. Source is the compiler's target description; AMD prose docs not found.
- C1 (WRONG): MSL has no integer dot product; `dot` is for float and half vectors and returns `half` for half inputs.
- C2 (PARTLY): MSL's `int4b_format` / `uint4b_format` (Metal 4.0 with SDK 26.4) and `unpack` (Metal 4.1) exist, but the nibble order within a byte and the supported GPU families are undocumented; neither MLX nor llama.cpp uses them (code search).
- C3 (PARTLY): MPP `matmul2d` documents `char × char → int` and `half × int4b_format → float`, but block scales are UE8M0 (power of two), block 32, only; arbitrary float group scales must be applied by the kernel. Executing hardware undocumented.
- D5 (PARTLY): MLX's docs give s = (max − min)/(2^b − 1), bias = min; the code takes the larger-magnitude extreme as the bias, allows a negative scale, and nudges the scale so zero is exactly representable. Bits 7 is rejected.
- D6 (PARTLY): `torch.ao.quantization` is superseded by torchao, but the 2.15 page still says deletion was planned "in 2.10". PyTorch's int4 CUDA kernel requires compute capability 8.0 (or ROCm) and uses a float zero added after scaling, a different convention from llama.cpp, MLX and MPSGraph.

Status facts worth knowing even where the verdict is CONFIRMED:

- A1: weight-only quantisation helps decode only when dequantisation is fused into the GEMV; PyTorch's own measurement shows an unfused int8 path slower than FP16, and the fused int8 speed-up well short of the 2 times byte ratio (their device).
- A5, C5: llama.cpp's 18-byte Q4_0 block is 2-byte aligned on every backend; read its quants as bytes or 16-bit words, never as `int` or `char4`.
- A6: `torch._int_mm` on CUDA refuses M ≤ 16, so batch-1 decode in PyTorch uses weight-only kernels.
- B3: on AMD, 4-bit integer matrix instructions are on RDNA3/4 (WMMA iu4), not in CDNA's MFMA list.
- C4: MPSGraph's asymmetric form is `scale * (q - zeroPoint)` with an integer zero point; llama.cpp Q4_1 uses `d·q + m`, MLX `s·q + bias`, PyTorch `q·scale + zero`. Four conventions for the same idea.
- D1: llama.cpp Q4_0 packs weight j and weight j + 16 into byte j; its scale is the signed extreme divided by −8, so codes span −8..7 and the opposite side clips by one step.
- D3, D4: llama.cpp's CUDA/HIP decode quantises the activations to int8 every token and uses integer dot products; its Metal decode stays in float.

Not checked: the GGUF file format specification document (only the layout constants in `ggml-common.h` and `gguf-py`); vLLM, Marlin, AWQ, GPTQ kernels; MLX's CUDA backend quantised kernels; Apple's documentation pages beyond the SDK headers and the two PDFs; any behaviour on real hardware.

## Where the backends really differ

One line per cell, for what X2's lessons and labs use. Nothing here was run on CUDA or AMD hardware; the Metal cells are from the spec and SDK headers, not compiled for this file.

| Feature | CUDA | HIP (ROCm 10.2) | Metal (MSL 4.1) |
|---|---|---|---|
| Per-thread int8 four-way dot | `__dp4a(int, int, int)` and `char4`/unsigned forms; sm_61+ | No `__dp4a`. `amd_mixed_dot(char4, char4, int, bool)`; builtins `__builtin_amdgcn_sdot4` (gfx906, CDNA, RDNA2) or `__builtin_amdgcn_sudot4` (RDNA3/4); none on gfx900, gfx1010 | None; `dot` is float/half only |
| Mixed signedness (unsigned nibble × signed int8) | PTX `dp4a.u32.s32`; not in the C++ intrinsic | `sudot4` with a signedness flag per operand (dot8 targets) | Not applicable |
| Per-thread 4-bit dot | None | `amd_mixed_dot(int, int, int, bool)` → `sdot8` (dot1 targets); `sudot8` (dot8) | None |
| 2 × 16-bit by 8-bit dot | `__dp2a_lo`, `__dp2a_hi` | `amd_mixed_dot(short2, short2, int, bool)` (2 × 16-bit) | None |
| Half dot with float accumulate, per thread | None (`__hfma2` accumulates in half) | `amd_mixed_dot(__half2, __half2, float, bool)` | None; `dot(half4, half4)` returns `half` |
| FP8 dot per thread | None | `__builtin_amdgcn_dot4_f32_fp8_fp8` and variants (gfx11.7, gfx12) | None (FP8 only as packed storage, appA C4) |
| Integer matrix instructions | `mma` s8/u8, s4/u4 (sm_75/80); `wgmma` s8/u8 (sm_90a); `tcgen05 .kind::i8` (sm_100a, sm_110a) | MFMA `i32_*_i8` (CDNA); WMMA `iu8`, `iu4` (RDNA3/4) | MPP `matmul2d`: `char × char → int` (Metal 4), `char × int4b_format → int`, `half × int4b_format → half/float` (OS 26.4) |
| 4-bit integer type | None in C++ (PTX `.s4/.u4`, `cvt.pack.sat.u4`) | None | `int4b_format`, `uint4b_format` (Metal 4.0, SDK 26.4), `unpack` to float/half/bfloat (Metal 4.1); nibble order undocumented |
| Block scales in hardware or library | cuBLASLt VEC32 UE8M0 (FP8) and VEC16 UE4M3 (FP4) (appA A5) | `__amd_scale_t` E8M0 conversions, gfx950 (appA B2) | `tensor_blockwise`: UE8M0 only, block 32; MPSGraph blockwise dequantise with float scales (macOS 15) |
| Byte-wise SIMD helpers (`__vsubss4`, ...) | Documented intrinsics | Not provided; llama.cpp writes them on `ext_vector_type` | Not applicable; `extract_bits`, `insert_bits` (Metal 1.2) |
| `char4` / byte vectors | `char4`, size 4, alignment 4 | Same names (vector types, infra B6) | `char4` (alignment 4), `packed_char4` (alignment 1) |
| Reading an 18-byte Q4_0 block | 2-byte aligned; llama.cpp uses two 16-bit loads (`get_int_b2`) | Same source | `half` alignment 2; llama.cpp reads `uint16_t` words |
| FP16 scale conversion | `__half2float` (`cuda_fp16.h`) | `__half2float` (`hip/hip_fp16.h`, needs `#if`) | `half` is a built-in type; `as_type<half>(ushort)` |
| What the real decode kernel does | llama.cpp: Q8_1 activations, `dp4a`, float per block; PyTorch: bf16 tinygemm, sm_80+ | llama.cpp: same source with the B1 ladder; PyTorch: same tinygemm under `USE_ROCM` | llama.cpp, MLX, PyTorch MPS: float FMAs on masked nibbles, activations pre-divided by 16/256/4096, offset folded through Σx |
| Emulator support in this repository | `__dp4a`, `char4`, `__half` not modelled | Same | Not applicable (Metal runs on the device) |
