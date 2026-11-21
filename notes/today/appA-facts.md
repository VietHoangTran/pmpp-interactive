# Appendix A fact check: numerical considerations on CUDA, HIP, Metal

Checked on 2026-11-20 against the live vendor, standards and project sources listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY. Where an item has several sub-claims, the verdict is for the item as a whole and the text says which sub-claim failed. A verdict judges the claim in the item title (the book's statement, or the assumption a course author would bring).
- **Quote** is copied from the source (HTML flattened to text, GitHub raw files at a tag or on the default branch, Apple's documentation JSON, PDF text, local SDK headers). Line breaks inside quotes are removed. PDF superscripts are written out (`2^-6`), because the text extraction flattens them.
- "infra A12", "ch10 C3", "ch05 C7" point at items in `notes/today/infra-facts.md`, `ch10-facts.md`, `ch05-facts.md` and so on. Anything settled there is cited, not re-checked.
- Page numbers ("p. 589") are book pages as recorded in `notes/appA.md`.
- Versions the sources reported about themselves on the check date: CUDA Programming Guide, CUDA Math API, nvcc, cuBLAS and cuSOLVER docs "13.5" (last updated Oct 31 to Nov 02, 2026); PTX ISA 9.5; "Floating Point and IEEE 754" white paper page "v13.5"; ROCm Core SDK 10.2.0 (HIP docs read as `.rst` sources and headers at tag `therock-10.2` of `ROCm/rocm-systems` and `ROCm/rocm-libraries`); Clang and LLVM AMDGPU docs at `clang.llvm.org/docs` and `llvm.org/docs` (upstream `main`; ROCm ships its own Clang build, which we did not read); Metal Shading Language Specification "Version 4.1", dated 2026-06-04 ("MSL spec"); Metal Feature Set Tables dated October 23, 2026 ("feature tables"); macOS SDK 27.0 headers on this machine; OCP MicroXcaling (MX) Specification Version 1.1, dated October 9, 2026; OCP 8-bit Floating Point Specification (OFP8) Revision 1.1, effective June 22, 2026; IEEE SA pages for 754-2019 and P3109; PyTorch docs 2.15 and `main`; MLX docs 0.32.4 and `main`; llama.cpp `master`; NumPy docs v2.5.
- Not checked: anything by running code on a GPU. Every vendor statement is from documentation, headers or project source. Where this file says what a piece of project code does (accumulator type, rounding), that is our reading of the source, labelled as such. Group E is our own arithmetic and project facts.
- Device figures in this file are evidence for a verdict, not lesson content. The project rule stands: the H100 numbers appear in a lesson only as the book's example.

Totals: 20 CONFIRMED, 1 WRONG, 8 PARTLY, 3 COULD NOT VERIFY (32 items: 16 book and CUDA, 6 HIP, 4 Metal, 4 project, 2 arithmetic and project). Failed or unfound sub-claims are collected under "Summary of corrections".

---

## Already settled elsewhere (cite, do not re-check)

| Appendix A needs | Settled in |
|---|---|
| H100 SXM: FP32 67 TFLOPS; "TF32 Tensor Core 989 teraFLOPS" with the footnote "With sparsity"; FP16 and BF16 tensor core 1,979 with sparsity | ch05 A8 |
| H100 SXM "FP8 Tensor Core 3,958 teraFLOPS", also "With sparsity" | ch01 A1 |
| Floating-point addition is not associative; NVIDIA's own three-value example (guide 5.6.1.4) and dot-product example (5.6.1.6); `std::reduce` vs `std::accumulate`; NVIDIA's inconsistent sentence about "re-associations" being allowed by default | ch10 A9 |
| `expf` 2 ULP, `__expf` 2 + floor(abs(1.173 x)) ULP; what `--use_fast_math` substitutes; NVIDIA recommends the default `-ftz=false -prec-div=true -prec-sqrt=true` | ch10 A11 |
| Float `atomicAdd` on global memory always flushes denormals to zero, on shared memory never does, regardless of `-ftz` | ch10 A6 |
| `fmaxf` and CUDA `max` drop NaN; `std::max` does not | ch10 A10 |
| `cub::DeviceReduce` is run-to-run deterministic by default | ch10 A13 |
| HIP `expf` and `__expf` errors are measured against the C++ library, not bounds | ch10 B4 |
| Metal default math mode is documented three ways (web docs: relaxed; MSL spec for the command-line compiler: fast; SDK header: depends on language version); `fastMathEnabled` deprecated in favour of `mathMode` | ch05 C7 |
| Under Metal fast math INF and NaN are undefined; `#pragma METAL fp math_mode(...)` exists but its spelling was not compile-tested; the project's lab runtime compiles with default `MTLCompileOptions` | ch10 C3 |
| Metal `exp` <= 4 ulp precise, `fast::exp` <= 3 + floor(abs(2 x)) ulp | ch10 C4 |
| The summation order of `simd_sum` is not documented | ch10 C1 |
| `__half` and `__half2` from `cuda_fp16.h`; HIP `__half` from `<hip/hip_fp16.h>` (needs an `#if`) | infra A12, B12 |
| Tensor-core input types per compute capability (FP8 from 8.9, FP6 and FP4 from 10.0, see A10 below); WMMA; AMD MFMA types; `simdgroup_matrix` and MPP `matmul2d` | infra A12, B12, C10 |
| MSL `half` is IEEE binary16 storage; `bfloat` from Metal 3.1; "Metal does not support the double ... data types" | infra C8 |
| Apple GPU families: M1-series is Apple7, M2 Apple8, M3 and M4 Apple9, M5 Apple10, M6 Apple11 | infra Group C intro |
| Accelerate BLAS and LAPACK are the CPU route on Apple platforms ("in line with LAPACK 3.12.0" from macOS 26) | infra C11 |

---

## Group A: the book's dated statements, and CUDA

### A1. IEEE 754-2019 is the current revision of the standard (pp. 583, 600)

**Verdict: CONFIRMED.** IEEE 754-2019 is still the active standard. A revision project (P754) is open, and a separate draft standard for machine-learning formats (P3109) is in progress; neither has been approved.

- IEEE SA page: "Active Standard IEEE 754-2019 IEEE Standard for Floating-Point Arithmetic", "Board Approval 2019-06-13", "Published: 2019-07-22", "Superseding 754-2008". The same page lists "P754 Standard for Floating-Point Arithmetic" under "Active Projects" and the identical international standard "60559-2020 ISO/IEC/IEEE International Standard - Floating-point arithmetic".
- NVIDIA's guide cites the same revision: "All GPU devices follow the IEEE 754-2019 standard for binary floating-point arithmetic with the following limitations" (section 5.6.3; the limitations are in A7 and A9).
- P3109: "IEEE Draft Standard for Arithmetic Formats for Machine Learning", "Status Active PAR", "PAR Approval 2023-02-15". Scope: "Compact binary floating-point interchange formats are specified, parameterized by bitwidth, precision, signedness, and domain. Operations including rounding with saturation modes and conversion between P3109 formats and IEEE Std 754 formats are defined." Mention it at most as work in progress; nothing in it is a standard yet.

- https://standards.ieee.org/ieee/754/6210/
- https://standards.ieee.org/ieee/3109/11165/
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html (section 5.6.3)

### A2. The formats the explorer needs: FP32, FP16, BF16 field widths, bias, range (A.1 to A.3, pp. 583 to 591)

**Verdict: CONFIRMED.** The book's value formula (A.1), its denormal formula (p. 589) and its single-precision field widths (p. 590) match NVIDIA's text. FP16 and BF16 are not in the book; their definitions below are vendor-documented and agree across sources. BF16 is not an IEEE 754 format.

Quotes:

- NVIDIA guide 5.6.1.1: normal value "(-1)^sign × 1.mantissa × 2^(exponent - bias)"; "For subnormal values, the formula is modified to: (-1)^sign × 0.mantissa × 2^(1-bias)"; "The exponents are biased by 127 and 1023 for single- and double-precision, respectively." With bias = 2^(n-1) - 1, 1 - bias = 2 - 2^(n-1), which is the book's `0.M · 2^(-2^(n-1) + 2)` (our algebra).
- NVIDIA guide 5.6.1.1: "16-bit, also known as half-precision, corresponding to the __half data type in CUDA."
- NVIDIA guide Table 50 "Supported Floating-Point Types": Bfloat16, `__nv_bfloat16`, IEEE-754 "❌", `<cuda_bf16.h>`, "Compute Capability 8.0 or higher."; Half Precision, `__half`, IEEE-754 "✅", `<cuda_fp16.h>`.
- OFP8 r1.1 section 4.1: "bfloat16: a binary format with 1 sign bit, 8 exponent bits, 7 mantissa bits, and an exponent bias of 127."
- MSL spec Table 2.1: `half` "must conform to the IEEE 754 binary16 storage format"; `bfloat` "is a truncated version of float for machine learning applications" ("All OS: Metal 3.1 and later"); `float` "must conform to the IEEE 754 single precision storage format".
- Clang language extensions: "__fp16 and _Float16 both use the binary16 format from IEEE 754-2008, which provides a 5-bit exponent and an 11-bit significand (including the implicit leading 1). __bf16 uses the bfloat16 format, which provides an 8-bit exponent and an 8-bit significand; this is the same exponent range as float".
- PyTorch dtype table: `torch.float16` "S-E-M 1-5-10", `torch.bfloat16` "S-E-M 1-8-7".
- HIP's FP8 header uses the same constants for its conversions: half `mantissa = x & 0x3FF`, `exponent = (head >> 10) & 0x1F`, `bias = 15`, `fInf = 0x7C00` (`amd_hip_fp8.h`, lines 220 to 228).

NVIDIA guide Table 51 "Supported Floating-Point Types Properties", as printed:

| Format | Largest value | Smallest positive normal | Smallest positive denormal | Epsilon |
|---|---|---|---|---|
| Bfloat16 | about 2^128, about 3.39e38 | 2^-126, about 1.18e-38 | 2^-133 | 2^-7 |
| Half | about 2^16, 65504 | 2^-14, about 6.1e-5 | 2^-24 | 2^-10 |
| Single | about 2^128, about 3.40e38 | 2^-126, about 1.18e-38 | 2^-149 | 2^-23 |
| Double | about 2^1024, about 1.8e308 | 2^-1022 | 2^-1074 | 2^-52 |

For the explorer (FP16 bias 15 follows from the smallest normal 2^-14 and is the constant in HIP's header): FP32 1-8-23 bias 127; FP16 1-5-10 bias 15; BF16 1-8-7 bias 127. FP32 and FP16 reserve the all-ones exponent for infinity and NaN and use the all-zeros exponent for zero and subnormals (NVIDIA 5.6.1.3, quoted in A7). BF16 follows the same layout as the top half of an FP32 (largest value about 2^128 and smallest denormal 2^-133 in Table 51; MSL 8.1 requires INF and NaN "for single-precision, half-precision, and brain floating-point numbers"). CUDA's own pointer for querying these: "cuda::std::numeric_limits in the <cuda/std/limits> header to query the properties and the ranges of the supported floating-point types, including microscaling formats (MX)".

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html (sections 5.6.1.1 to 5.6.1.3, 5.6.2, Tables 50 and 51)
- https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-1-final-pdf/ (section 4.1)
- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 2.1, Table 2.1)
- https://clang.llvm.org/docs/LanguageExtensions.html ("Half-Precision Floating Point")
- https://docs.pytorch.org/docs/2.15/tensor_attributes.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_hip_fp8.h

### A3. The FP8 formats E4M3 and E5M2 (for the explorer and for X2)

**Verdict: CONFIRMED.** OCP's OFP8 specification defines both, and was revised this year (Revision 1.1, dated January 29, 2026 in its version table, "Effective June 22, 2026" on the cover). The bit definitions did not change from 1.0; 1.1 "Adjusts the conversion description in Section 5.2 and adds Appendix A."

OFP8 r1.1 Tables 1 and 2, as printed:

| | E4M3 | E5M2 |
|---|---|---|
| Exponent bias | 7 | 15 |
| emax, emin (unbiased) | 8, -6 | 15, -14 |
| Infinities | N/A | S.11111.00 |
| NaN | S.1111.111 | S.11111.{01, 10, 11} |
| Zeros | S.0000.000 | S.00000.00 |
| Max normal | S.1111.110 = ±448 | S.11110.11 = ±57,344 |
| Min normal | S.0001.000 = ±2^-6 | S.00001.00 = ±2^-14 |
| Max subnormal | S.0000.111 = ±0.875 × 2^-6 | S.00000.11 = ±0.75 × 2^-14 |
| Min subnormal | S.0000.001 = ±2^-9 | S.00000.01 = ±2^-16 |
| Dynamic range | 18 binades | 32 binades |

Quotes:

- "The E5M2 format represents infinities and NaNs. Interpretation of the three mantissa values for NaNs is not defined. The E4M3 format does not represent infinities and uses only two bit patterns for NaN (a single mantissa-exponent bit pattern but allowing both values of the sign bit) in order to increase emax to 8 and thus to increase the dynamic range by one binade."
- Conversion: "this specification requires implementation of Round-to-Nearest-Ties-to-Even mode" and "requires implementation of saturating and non-saturating modes." Non-saturating overflow: "E4M3 destination: generate a NaN", "E5M2 destination: generate an infinity".
- Appendix A (new in 1.1): for E4M3 "source format magnitudes up to and including 1/2 an ULP greater (i.e. 448 < x <= 464) round down to 448"; for E5M2 "a value with 1/2 an ULP greater (61,440) rounds up resulting in overflow (or saturation)". A good predict-then-reveal for the explorer.
- Scope: "Not in scope are: ... methods for performing arithmetic operations on the specified OFP8 formats". FP8 is an interchange (storage) format; arithmetic happens in matrix units or after conversion.

The "FNUZ" variants used by AMD CDNA3 and listed by PyTorch are different encodings with the same field widths; see B2 and D1.

- https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-1-final-pdf/ (sections 3 to 5, Appendix A)

### A4. OCP Microscaling (MX) formats "have been adopted by recent GPUs"; spec v1.0, 2023 (footnote p. 583; ref. [3], p. 600)

**Verdict: PARTLY.** Adoption is confirmed on all three vendors' current documentation. But the cited v1.0 is no longer the latest: version 1.1 (October 9, 2026) renamed the specification "MicroXcaling" and the scale type E8M0 to "UE8M0", and clarified conversion, NaN and out-of-range behaviour.

The format (MX v1.1 sections 5.1 to 5.4):

- "An MX-compliant format is characterized by three components: Scale (X) data type / encoding; Private elements (P_i) data type / encoding; Scaling block size (k)". Value: "v_i = X P_i" (with NaN and out-of-range rules).
- Table 1, concrete formats: MXFP8 (E5M2 or E4M3), MXFP6 (E3M2 or E2M3), MXFP4 (E2M1), MXINT8; for all of them "Scaling Block Size (k) 32", "Scale Data Type UE8M0", "Scale Bits (w) 8".
- FP6 Table 4: E2M3 bias 1, max ±7.5, min subnormal ±0.125; E3M2 bias 3, max ±28.0, min subnormal ±0.0625; "No encodings are reserved for Inf or NaN in FP6."
- FP4 Table 5: E2M1 bias 1, max normal ±6.0, min normal ±1.0, subnormal ±0.5; "No encodings are reserved for Inf or NaN in FP4." So the eight magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4, 6 (our enumeration of Table 5).
- UE8M0 Table 7: "Exponent bias 127", "Supported exponent range -127 to 127", "NaN 11111111", no zero, no infinity.
- Scale from data (6.3): the shared exponent is floor(log2(max |V_i|)) - maxExp (+ an optional adjustment), and elements "that exceed the max normal representation of the element data type should be clamped to the max normal".
- Order of summation is left open, which is the book's §A.6 point at block scale: "The internal precision of the dot product and order of operations is implementation-defined." (6.1).

Adoption, by vendor:

- NVIDIA: PTX `mma` with "`.kind::mxf8f6f4`, `.kind::mxf4`, `.kind::mxf4nvf4` perform matrix multiplication with block scaling", element types `.e4m3, .e5m2, .e3m2, .e2m3, .e2m1` with scale type `.ue8m0` (PTX ISA 9.5 section 9.7.16.3, Table 45). cuBLAS: "32-element ... 1D block scaling" for FP8 with `CUDA_R_8F_UE8M0` scales on compute capability "10.0+", and "1D block scaling ... is described in more details in the OCP MXFP specification". See A5 for NVIDIA's own FP4 variant.
- AMD: "HIP also provides some extensions APIs for microscaling formats. These are supported on AMD GPUs. gfx950 provides hardware acceleration for hip extensions." "Scale is defined as type __amd_scale_t and is of format E8M0." (B2)
- Apple: Metal 4.1 `tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>` over `metal_fp4_e2m1_format` data, the spec's own example being named `mxfp4_tensor` (C4).

- https://www.opencompute.org/wp-content/uploads/2026/10/OCP_MicroXcaling_Spec_v1.1.0.pdf
- https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf (the book's v1.0, "Date: Sep 2023")
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (section 9.7.16.3)
- https://docs.nvidia.com/cuda/cublas/index.html (section 3.1.4.4)

### A5. Block formats beyond OCP: NVIDIA's 16-element FP4 with E4M3 scales, and how scales are rounded (for X2)

**Verdict: CONFIRMED.** NVIDIA documents a second FP4 block format next to MXFP4, and NVIDIA's scale rounding differs from OCP's.

- cuBLAS 3.1.4.4: "To enable block scaling, the CUBLASLT_MATMUL_DESC_{X}_SCALE_MODE attributes ... must be set to CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 for all FP8-precision tensors or to CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 for all FP4-precision tensors." Mode table: "16-element 1D block scaling | 10.0+ | CUDA_R_4F_E2M1 | CUDA_R_8F_UE4M3". The FP4 recipe also takes "a precomputed input tensorwide scaling factor".
- cuBLAS on the FP8 scale: "Round the block exponent up keeping it within the range of values representable in UE8M0" and "Note: The algorithm above differs from the OCP MXFP suggested rounding scheme." (OCP uses a floor, A4.)
- PTX Table 45: `.kind::mxf4nvf4` accepts `.e2m1` data with either `.ue8m0` or `.ue4m3` scales.
- CUDA Math API types: `__nv_fp8_e4m3` ("1 sign, 4 exponent, 1 implicit and 3 explicit mantissa bits. The encoding doesn't support Infinity. NaNs are limited to 0x7F and 0xFF values."), `__nv_fp8_e5m2`, `__nv_fp8_ue5m3` ("no sign, 5 exponent, 1 implicit and 3 explicit mantissa bits"), `__nv_fp8_e8m0` ("Bias equals 127, so numbers 0 through 254 represent 2^-127 through 2^127. Number 0xFF = 255 is reserved for NaN."), `__nv_fp6_e2m3`, `__nv_fp6_e3m2`, `__nv_fp4_e2m1` ("This encoding does not support Inf/NaN."). Headers `cuda_fp8.h`, `cuda_fp6.h`, `cuda_fp4.h`.
- Rounding and saturation defaults differ by type: the `__nv_fp8_e4m3(float)` constructor "relies on __NV_SATFINITE behavior for out-of-range values"; the `__nv_fp8_e8m0(float)` constructor "relies on __NV_SATFINITE behavior behavior for large input values and cudaRoundPosInf for rounding" (round up, as cuBLAS does). `__NV_NOSAT` "for the target fp8 types of e4m3 and ue5m3 kinds, the results that are larger than the maximum representable finite number of the target format become NaN."
- The word "NVFP4" does not appear in the cuBLAS or PTX pages read; it is used by MLX (`mode="nvfp4"`, D2) and llama.cpp (`block_nvfp4`, D3). Teach it as "NVIDIA's 16-element FP4 block format", and attach the name to those projects.

- https://docs.nvidia.com/cuda/cublas/index.html (sections 3.1.4.4 and 3.1.4.4.1)
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (section 9.7.16.3)
- https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP8__MISC.html
- https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp8__e4m3.html, `struct____nv__fp8__e8m0.html`, `struct____nv__fp8__ue5m3.html`, `struct____nv__fp4__e2m1.html`

### A6. Denormals: "virtually all recent generations of CUDA devices" support them; the NVCC flag "ftz = true (on) or ftz = false (off)" (pp. 589, 599)

**Verdict: PARTLY.** Support is confirmed and is the default. The flag's meaning needs stating precisely, because the book's "(on)" reads backwards: `--ftz=true` turns flush-to-zero on, which turns denormal support off. The book's cost claim ("thousands of clock cycles") is not something NVIDIA says about its GPUs.

- nvcc 4.2.7.8: "--ftz { true | false } ( -ftz )". "Control single-precision denormals support. --ftz=true flushes denormal values to zero and --ftz=false preserves denormal values. --use_fast_math implies --ftz=true." Default: "This option is set to false and nvcc preserves denormal values."
- So the flag touches single precision only. The 2011 white paper: "The flags have no effect on double precision or on devices of compute capability below 2.0."
- Guide 5.6.1.2: "Subnormals are a required part of the IEEE-754 floating-point standard." "However, subnormal numbers are computationally more expensive. Therefore, applications that don't require strict accuracy may choose to avoid them to improve performance. The nvcc compiler allows disabling subnormal numbers by setting the -ftz=true option (flush-to-zero), which is also included in --use_fast_math."
- Guide 5.6.3 lists the GPUs' deviations from IEEE 754-2019 (A7, A9). Missing denormals is not among them.
- Exceptions that ignore the flag: float atomics on global memory always flush (ch10 A6); "__tanhf(x) ... Subnormal results are not flushed to zero even under -ftz=true compiler flag." (Table 65).
- COULD NOT VERIFY: any NVIDIA statement of a cycle cost for denormals on current GPUs. NVIDIA says only "computationally more expensive". Keep "thousands of clock cycles" as the book's statement about "moderate hardware".

- https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html (section 4.2.7.8)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html (sections 5.6.1.2, 5.6.3, 5.6.9.2)
- https://docs.nvidia.com/cuda/floating-point/index.html (section 4.4)

### A7. "The current generation of GPU hardware does not support signaling NaN" (p. 591)

**Verdict: CONFIRMED** for NVIDIA (here), Apple (C2) and, in the documented IEEE mode, AMD (B6). No GPU in the sources raises an exception on a signaling NaN.

- Guide 5.6.3: "There is no mechanism to detect floating-point exceptions, so all operations behave as if IEEE-754 exceptions are always masked. If there is an exceptional event, the default masked response defined by IEEE-754 is delivered. For this reason, although signaling NaN SNaN encodings are supported, they are not signaling and are handled as quiet exceptions."
- Guide 5.6.3: "Floating-point operations may alter the bit patterns of input NaN payloads. Operations such as absolute value and negation may also not comply with the IEEE 754 requirement, which could result in the sign of a NaN being updated in an implementation-defined manner." So the book's idea of encoding a NaN's cause in its spare bits (p. 591) does not survive arithmetic on a GPU.
- Guide 5.6.4: "Unlike the host implementation, the mathematical operators and functions supported in device code do not set the global errno variable nor report floating-point exceptions to indicate errors."
- Guide 5.6.1.3, matching the book's Fig. A.8: infinity has "all bits in the exponent set to 1 and all bits in the significand set to 0"; NaN "all bits in the exponent set to 1 and any bit pattern in the significand, except for all bits set to 0. There are 2^(mantissa + 1) - 2 possible encodings."; quiet NaNs "are encoded with the most significant bit of the significand set to 1"; signaling "most significant bit of the significand set to 0". "Any ordered comparison ( < , <= , > , >= , == ) involving a NaN will result in false ... The unordered comparison NaN != NaN returns true." The book's NaN count is reconciled with NVIDIA's in E1.
- Saturation, as the book says (p. 590): "Floating-point numbers behave according to saturation arithmetic, in which operations that overflow the representable range result in +Infinity or -Infinity." And: "C++ implementations that conform to IEC-60559 (IEEE-754), including CUDA, produce infinity" for floating-point division by zero; "Converting a floating-point value to an integer type for which the result is not representable is undefined behavior. This also includes NaN and infinity."

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html (sections 5.6.1.3, 5.6.3, 5.6.4)

### A8. 0.5 ULP "is the accuracy achieved by all CUDA devices today"; 2-ULP inversion on "some early devices" (p. 592)

**Verdict: CONFIRMED**, with a units caution for the lesson. NVIDIA states the same guarantee as "a maximum ULP error of zero" measured against the correctly rounded result, which is the book's "at most 0.5 ULP from the exact value". The 2-ULP division still exists, as an opt-in.

- Guide 5.6.6: "The built-in C/C++ language operators, such as x + y , x - y , x * y , x / y , x++ , x-- , and reciprocal 1 / x , for single-, double-, and quad-precision types comply with the IEEE-754 standard. They guarantee a maximum ULP error of zero using a round-to-nearest-ties-to-even rounding mode."
- How NVIDIA counts: "The maximum ULP error is stated as the maximum observed absolute value of the difference in ULPs between the value returned by the function and a correctly rounded result" (5.6.7). Metal counts from "the infinitely precise result" (C2). Say which reference a number uses before comparing vendors.
- The opt-in approximations (5.6.6): with `-prec-div=false` (implied by `--use_fast_math`), "x / y → __fdividef(x, y) : 2 ULP" and "1 / x : 1 ULP". `sqrtf`: "0 ULP", "1 ULP with --use_fast_math". `__fdividef`: "2 for |y| in [2^-126, 2^126]" (Table 65).
- nvcc 4.2.7.9: "--prec-div=true enables the IEEE round-to-nearest mode and --prec-div=false enables the fast approximation mode." Default true. Same shape for `--prec-sqrt`.
- The 2011 white paper says when this became standard: "Devices with compute capability 2.0 and above support both single and double precision IEEE 754 including fused multiply-add ... Operations such as square root and division will result in the floating point value closest to the correct mathematical result".
- NVIDIA's error tables for library functions are measured, not proved: "The error bounds are derived from extensive, though not exhaustive, testing. Therefore, they are not guaranteed."

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html (sections 5.6.6, 5.6.7, 5.6.7.3, 5.6.9.2)
- https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html (sections 4.2.7.7 to 4.2.7.11)
- https://docs.nvidia.com/cuda/floating-point/index.html (section 4.1)

### A9. Rounding modes and fused multiply-add (A.4; Exercise 3 is round-toward-zero)

**Verdict: CONFIRMED.** CUDA has all four IEEE rounding modes, chosen per operation, default round-to-nearest-even. The book does not mention FMA, but it changes results and nvcc contracts by default.

- Guide 5.6.1.7: "The IEEE-754 standard defines four rounding modes: round-to-nearest , round towards positive , round towards negative , and round towards zero . CUDA supports all four modes. By default, operations use round-to-nearest . Intrinsic mathematical functions can be used to select other rounding modes for individual operations." Suffixes `rn`, `rz`, `ru`, `rd`.
- Guide 5.6.3: "There is no dynamically configurable rounding mode; however, most of the operations support multiple constant IEEE rounding modes, selectable via specifically named device intrinsics functions."
- Table 64: `__fadd_[rn,rz,ru,rd]`, `__fmul_*`, `__fmaf_*`, `__fdiv_*`, `__frcp_*`, `__fsqrt_*` "have a maximum ULP error of 0 and are IEEE-compliant". So Exercise 3 (an adder that truncates) can be run for real with `__fadd_rz`.
- 5.6.1.5: "The Fused Multiply-Add (FMA) operation computes the result with only one rounding step." 5.6.9.1: "The __fadd_[rn,rz,ru,rd]() ... and __dmul_[rn,rz,ru,rd]() functions map to addition and multiplication operations that the compiler never merges into the FFMA or DFMA instructions. In contrast, additions and multiplications generated from the * and + operators are often combined into FFMA or DFMA."
- nvcc `--fmad`: "This option is set to true and nvcc enables the contraction of floating-point multiplies and adds/subtracts into floating-point multiply-add operations (FMAD, FFMA, or DFMA)."
- Host side, relevant to a CPU reference: "It is important to understand the host platform's support for Fused Multiply-Add when comparing CPU and GPU results." Clang's host default is `-ffp-contract=on` (fusion within one statement) for C++ (B5).

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html (sections 5.6.1.5, 5.6.1.7, 5.6.3, 5.6.9.1, Table 64)
- https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html (section 4.2.7.11)

### A10. "Recent GPU architectures introduced hardware support for computing with 16-bit, 8-bit, and 4-bit floating point numbers" (p. 593)

**Verdict: CONFIRMED** on NVIDIA and AMD; on Apple the formats exist in the language from Metal 4.1, with no statement about hardware (C4).

- NVIDIA Table 36 (tensor-core input types), parsed with its column spans: FP16 from 7.5; BF16 and TF32 from 8.0; FP8 from 8.9 (and 9.0, the book's H100); FP6 and FP4 from 10.0, also 10.3, 10.7, 11.0, 12.x. (infra A12 recorded the 9.0 and 10.0 rows; this adds 8.9 as the first FP8 row.)
- Guide 5.6.2: "CUDA also supports TensorFloat-32 ( TF32 ), microscaling (MX) floating-point types, and other lower precision numerical formats that are not intended for general-purpose computation, but rather for specialized purposes involving tensor cores. These include 4-, 6-, and 8-bit floating-point types."
- Guide Table 52 lists the C++ standard library support for the small types: `__nv_fp8_e4m3`, `__nv_fp8_e5m2`, `__nv_fp8_e8m0`, `__nv_fp6_e2m3`, `__nv_fp6_e3m2`, `__nv_fp4_e2m1`, with the footnote "The CUDA C++ Standard Library functions support queries for small floating-point types, such as numeric_limits<T> , fpclassify() , isfinite() , isnormal() , isinf() , and isnan()". No arithmetic operators are listed for them.
- AMD: FP8 FNUZ on CDNA3; OCP FP8 on CDNA4 and RDNA4; FP6 and FP4 on CDNA4 only (B2).

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html (Table 36)
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html (section 5.6.2, Table 52)

### A11. H100: FP8 tensor-core throughput 3,958 TFLOPS vs "single-precision throughput using tensor cores" 989 TFLOPS (p. 593)

**Verdict: PARTLY.** Both numbers are NVIDIA's H100 SXM figures, but both rows carry NVIDIA's "With sparsity" footnote (ch01 A1, ch05 A8). And the 989 row is TF32, which is not IEEE single precision: same exponent range, about 10 stored mantissa bits. The 4x ratio holds between the two sparse rows; NVIDIA's product page prints no dense figures.

- PTX ISA 5.2.3: "tf32 data format: This data format is a special 32-bit floating point format supported by the matrix multiply-and-accumulate instructions, with the same range as .f32 and reduced precision (>=10 bits). The internal layout of tf32 format is implementation defined."
- PyTorch's note says the same in user terms: "When an operation is performed using TF32 tensor cores, only the first 10 bits of the input mantissa are read." (D1)
- cuBLAS: `CUBLAS_COMPUTE_32F_FAST_TF32` "Allows the library to use Tensor Cores with TF32 compute for 32-bit floating point input and output matrices. Note that input conversions round to nearest even."
- For the lesson: show the pair as the book's example, label the 989 as "TF32 (reduced-precision FP32 input) with sparsity", and keep the point the book makes (fewer bytes per operation, more operations per second).

- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (section 5.2.3)
- https://docs.nvidia.com/cuda/cublas/index.html (`cublasComputeType_t`)
- ch01 A1, ch05 A8

### A12. The book's reference [4], https://docs.nvidia.com/cuda/floating-point/index.html (p. 600)

**Verdict: CONFIRMED.** The URL resolves (HTTP 200) to "Floating Point and IEEE 754 Compliance for NVIDIA GPUs", labelled "v13.5" and "Last updated on Nov 02, 2026". Its content is the 2011 white paper ("At the time this paper is written (Spring 2011)"; "Current generations of the NVIDIA architecture such as Tesla Kxx , GTX 8xx , and GTX 9xx"). The current, maintained text is guide section 5.6, which lists the white paper among its references ("Nathan Whitehead, Alex Fit-Florea. Precision & Performance: Floating Point and IEEE 754 Compliance for NVIDIA GPUs . Nvidia Report, 2011."). Cite section 5.6 for current behaviour and the white paper only for history.

- https://docs.nvidia.com/cuda/floating-point/index.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html (section 5.6.10)

### A13. A compensated (Kahan) sum survives nvcc's optimizer (§A.6 points to Kahan, p. 594; the lab measures it)

**Verdict: COULD NOT VERIFY.** A compensated sum computes `c = (t - sum) - y`, which is zero in real arithmetic; a compiler allowed to reassociate may delete it. NVIDIA's documentation does not settle whether nvcc ever reassociates float additions.

- The two NVIDIA sentences that conflict (ch10 A9): "Note that floating-point expression re-associations and contractions are allowed by default, similarly to the --fmad=true option." (5.6.3) and "The highest optimization level for GCC ( -O3 ), Clang ( -O3 ), nvcc ( -O3 ), and Microsoft Visual Studio ( /O2 ) does not affect floating-point semantics." (5.6.1.8).
- `--use_fast_math` is documented to imply only `--ftz=true --prec-div=false --prec-sqrt=false --fmad=true` plus the function substitutions of Table 66. Reassociation is not on that list.
- What is documented and helps: `__fadd_rn` and `__fsub_rn` "map to addition ... operations that the compiler never merges into the FFMA or DFMA instructions" (5.6.9.1). Whether they also block reassociation is not stated.
- For the lab: write the CUDA compensated sum with `__fadd_rn` / `__fsub_rn`, and let the measurement (compensated error against the CPU reference) show whether the compensation survived. Contrast Clang, which documents `-fno-associative-math` as the default (B5), and Metal, whose default math modes allow reassociation (C3).

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html (sections 5.6.1.8, 5.6.3, 5.6.9.1, 5.6.9.3)
- https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html (section 4.2.7.7)

### A14. "Fast arithmetic operations implemented in the special function units" (p. 599)

**COULD NOT VERIFY.** Neither the guide's floating-point appendix (section 5.6) nor the compute-capabilities appendix contains "special function unit" or "SFU" (searched today, 0 hits). NVIDIA's current wording is: "Intrinsic mathematical functions are faster and less accurate versions of their corresponding CUDA C Standard Library Mathematical functions ... They are faster because they map to fewer native instructions." (5.6.9). Teach the documented part (intrinsics: `__sinf`, `__expf`, ..., with their error bounds, ch10 A11) and keep "special function units" as the book's term.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html (section 5.6.9)

### A15. Pivoting terms: "partial pivoting" restricts swap candidates "to a localized set of equations"; the general rule swaps rows and columns (pp. 598 to 599)

**Verdict: WRONG** (terminology). In the libraries the reader will call, "partial pivoting" means row interchanges, choosing the pivot within the current column; swapping rows and columns is "complete pivoting". The book's definition of partial pivoting matches neither. Its point that pivoting is needed for stability is right, and cuSOLVER says so in one line.

- Reference LAPACK `dgetrf`: "DGETRF computes an LU factorization of a general M-by-N matrix A using partial pivoting with row interchanges. The factorization has the form A = P * L * U where P is a permutation matrix".
- Reference LAPACK `dgetc2`: "DGETC2 computes an LU factorization with complete pivoting of the n-by-n matrix A. The factorization has the form A = P * L * U * Q, where P and Q are permutation matrices".
- cuSOLVER `getrf`: "P*A = L*U". "If devIpiv is null, no pivoting is performed. The factorization is A=L*U , which is not numerically stable." The library offers "LU with partial pivoting ( getrf )".
- rocSOLVER (ROCm 10.2): "The GETRF functions compute the LU factorization of a general m-by-n matrix A using partial pivoting with row interchanges." and separate `GETRF_NPVT` / `GETF2_NPVT` functions "without partial pivoting".
- Metal Performance Shaders: `MPSMatrixDecompositionLU`, "A kernel for computing the LU factorization of a matrix using partial pivoting with row interchanges." "PA = LU" (macOS 10.13 and later).
- For the condensed A.7 page: name the library routes above, use the standard terms, and say the book's "localized set" description reads like a communication-avoiding variant (our reading of the book's reference [6], not checked).

- https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dgetrf.f (lines 30 to 45)
- https://github.com/Reference-LAPACK/lapack/blob/master/SRC/dgetc2.f (lines 30 to 40)
- https://docs.nvidia.com/cuda/cusolver/index.html (section 2.4.4.3 and the `getrf` description)
- https://github.com/ROCm/rocm-libraries/blob/therock-10.2/projects/rocsolver/library/include/rocsolver/rocsolver-functions.h (lines 4840 to 4850, 5928 to 5945)
- https://developer.apple.com/documentation/metalperformanceshaders/mpsmatrixdecompositionlu

### A16. Half and bfloat16 arithmetic in CUDA C++ (for the lab's FP16 / BF16 accumulators)

**Verdict: CONFIRMED.** The ordinary operators work on `__half` and `__nv_bfloat16` once the headers are included; each `+` is one half-precision addition rounded to nearest even.

- Guide 5.6.5: built-in operators "x + y , x - y , x * y , x / y , x++ , x-- , x += y ..." and "__half and __nv_bfloat16 types are also supported by including the <cuda_fp16.h> and <cuda_bf16.h> headers, respectively."
- `__hadd`: "Performs half addition in round-to-nearest-even mode." `__hadd_rn`: "Performs half addition of inputs a and b , in round-to-nearest-even mode. Prevents floating-point contractions of mul+add into fma."
- Header note: "NOTE: Aggressive floating-point optimizations performed by host or device compilers may affect numeric behavior of the functions implemented in this header."
- Requirement: `__nv_bfloat16` "Compute Capability 8.0 or higher" (Table 50). The guide's oldest listed compute capability is 7.5 (ch04 A10), so a BF16 variant needs a guard; FP16 does not.
- HIP and Metal spellings: B1, C1. Emulator support: E2.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html (section 5.6.5, Table 50)
- https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__HALF.html
- https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__ARITHMETIC.html

---

## Group B: HIP / ROCm

### B1. FP16 and BF16 types in HIP

**Verdict: PARTLY.** FP16 keeps CUDA's type name `__half`, from a different header (infra B12). BF16 has a different type name, `__hip_bfloat16`, so a `.cu` file that names `__nv_bfloat16` does not compile under HIP without a typedef.

- HIP low-precision page: "#include <hip/hip_fp16.h> // for float16", "__half fp16_val(1.1f);", "__half2: holds 2 values of float16 numbers"; "#include <hip/hip_bf16.h> // for bfloat16", "__hip_bfloat16 bf16_val(1.1f);", "__hip_bfloat162: holds 2 values of bfloat16 numbers". Both examples add with `+`: "__half sum = in1[idx] + in2[idx];", "__hip_bfloat16 sum = in1[idx] + in2[idx];".
- Field widths as in A2: float16 "Sign: 1 bit, Exponent: 5 bits, Mantissa: 10 bits"; bfloat16 "Sign: 1 bit, Exponent: 8 bits, Mantissa: 7 bits", "a truncated version of the 32-bit IEEE 754 single-precision floating-point format".
- Device support: "This precision format is supported across all GPU architectures" (both float16 and bfloat16). AMD's hardware-features table is narrower: the row "bfloat16 16-bit floating-point operations" has a mix of ❌ and ✅ across its columns. The two pages disagree on whether BF16 arithmetic is native everywhere; we did not resolve it.
- Naming inconsistency inside AMD's docs: the ROCm 10.2 "Data types and precision support" page lists the HIP types as `half` and `bfloat16`, where the HIP page uses `__half` and `__hip_bfloat16`.
- llama.cpp's fix (D3): `typedef __hip_bfloat16 nv_bfloat16;` in `vendors/hip.h`. The project's `labs/common/include/pmpp/vendors/hip.h` currently maps the header names only in a comment ("<cuda_fp16.h> is <hip/hip_fp16.h>"); a BF16 lab would need the same `#if` plus a typedef (our recommendation; E2).

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/docs/reference/low_fp_types.rst
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/docs/reference/hardware_features.rst
- https://rocm.docs.amd.com/en/latest/reference/precision-support.html

### B2. FP8, FP6, FP4 and MX on AMD

**Verdict: CONFIRMED.** HIP has OCP FP8 and AMD's FNUZ FP8, FP6, FP4, and a microscaling extension API. Which ones exist depends on the architecture.

- FP8: "FP8-OCP (Open Compute Project) ... a standardized format" and "FP8-FNUZ (Finite and NaN Only) ... supporting only finite and NaN values (no Inf support)", "This provides one extra value of exponent", "NaN Definition: When the sign bit is set, and all other exponent and mantissa bits are zero." Classes `__hip_fp8_e4m3`, `__hip_fp8_e5m2`, `__hip_fp8_e4m3_fnuz`, `__hip_fp8_e5m2_fnuz`; header `<hip/hip_fp8.h>`. "Note: On a gfx94x GPU, the type will default to the fnuz type." "FNUZ extensions will be available on gfx94x only."
- FNUZ bias, from the header (not stated in the docs): `const int f8_bias = (1 << (we - 1)) - 1 + (is_fnuz ? 1 : 0);` so E4M3FNUZ has bias 8 and E5M2FNUZ bias 16; the max-value comments read "// 240" for FNUZ E4M3 and "// 448" for OCP E4M3 (`amd_hip_fp8.h`, lines 243 to 252, 298).
- ROCm precision page: "The FNUZ variant has expanded range with no infinity or signed zero (NaN represented as negative zero), while the OCP variant follows the Open Compute Project specification." "CDNA3 natively supports FP8 FNUZ (E4M3 and E5M2), which differs from the customized FP8 format used with NVIDIA H100". "In some AMD documents and articles, float8 (E5M2) is referred to as bfloat8."
- Architecture tables (HIP page and ROCm page agree): FNUZ FP8 CDNA3 only; OCP FP8 CDNA4 and RDNA4; FP6 (E2M3, E3M2) and FP4 (E2M1) CDNA4 only. ROCm's examples: CDNA3 "such as MI300A, MI300X, and MI325X", CDNA4 "such as MI350X and MI355X", RDNA4 "such as RX 9070 and RX 9070XT".
- Saturation default matches NVIDIA's: `struct __hip_fp8_e4m3 { ... constexpr static __hip_saturation_t __default_saturation = __HIP_SATFINITE;` with `enum __hip_saturation_t { __HIP_NOSAT = 0, __HIP_SATFINITE = 1 }`.
- MX extension: "gfx950 provides hardware acceleration for hip extensions. In fact most APIs are 1 to 1 mapping of hardware instruction." "Scale is defined as type __amd_scale_t and is of format E8M0." Conversions with a scale (`__amd_cvt_fp4x2_to_floatx2_scale`, `__amd_cvt_floatx2_to_fp8x2_scale`, ...) and with stochastic rounding: "sr: if an API name has sr in it, means it will do stochastic rounding and will expect an input as seed." Stochastic rounding is a rounding mode the book does not cover; it is a fair aside for A.4.

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/docs/reference/low_fp_types.rst
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/include/hip/amd_detail/amd_hip_fp8.h
- https://rocm.docs.amd.com/en/latest/reference/precision-support.html

### B3. Denormals under HIP: supported by default, and how to flush them

**COULD NOT VERIFY** the default. No HIP or ROCm page read today states whether HIP-Clang keeps or flushes FP32 denormals by default. What is documented:

- Clang: "-fgpu-flush-denormals-to-zero , -fcuda-flush-denormals-to-zero , -fno-gpu-flush-denormals-to-zero: Flush denormal floating point values to zero in CUDA/HIP device mode." The general option "-fdenormal-fp-math=<value>": "The default value depends on the target. For most targets, it defaults to ieee."
- Hardware: each AMD kernel starts with a denormal mode, separately for FP32 and for FP16/FP64: "FLOAT_DENORM_MODE_32: Wavefront starts execution with specified denorm mode for single (32 bit) floating point precision floating point operations." Values "FLOAT_DENORM_MODE_FLUSH_SRC_DST 0", "FLUSH_DST 1", "FLUSH_SRC 2", "FLUSH_NONE 3". The assembler directive defaults are `.amdhsa_float_denorm_mode_32` 0 and `.amdhsa_float_denorm_mode_16_64` 3, but those are assembler defaults, not what the compiler emits for HIP; we did not find the compiler's choice documented.
- HIP docs on fast float atomics: "some devices will use different rounding or denormal modes" than the CAS-loop version.
- PyTorch's documentation (not AMD's): "On AMD Instinct MI200 GPUs, the FP16 and BF16 V_DOT2 and MFMA matrix instructions flush input and output denormal values to zero. FP32 and FP64 MFMA matrix instructions do not flush input and output denormal values to zero." (D1)
- For the lab: do not assume either way on AMD; the lab can print what it observes for a value below the FP32 smallest normal.

- https://clang.llvm.org/docs/ClangCommandLineReference.html (`-fgpu-flush-denormals-to-zero`)
- https://clang.llvm.org/docs/UsersManual.html (`-fdenormal-fp-math`)
- https://llvm.org/docs/AMDGPUUsage.html (kernel descriptor, `compute_pgm_rsrc1`; "Floating Point Denorm Mode Enumeration Values")
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/docs/how-to/hip_cpp_language_extensions.rst ("Unsafe floating-point atomic operations")

### B4. Rounding modes and basic-operation accuracy on HIP

**Verdict: PARTLY.** HIP has the CUDA spellings of the per-operation intrinsics, but only round-to-nearest-even is available by default, and AMD documents measured differences, not guarantees.

- HIP math API: "Only the nearest-even rounding mode is supported by default on AMD GPUs. The _rz, _ru, and _rd suffixed intrinsic functions exist in the HIP AMD backend if the OCML_BASIC_ROUNDED_OPERATIONS macro is defined." So Exercise 3's truncating adder (`__fadd_rz`) is not portable to HIP without that macro.
- `float __fadd_rn(float x, float y)`: "Add two floating-point values in round-to-nearest-even mode", test range [-1000, 1000], ULP difference 0; `__fmaf_rn` 0; `__fdiv_rn` 0.
- How AMD counts: "Error bounds on this page are measured in units in the last place (ULPs), representing the absolute difference between a HIP math function result and its corresponding C++ standard library function" (ch10 B4). That is a measurement on a range, not NVIDIA-style "maximum ULP error of zero" for the operators.
- The `+`, `*`, `/` operators on `float`: no AMD page read today states their accuracy. Clang has "-fhip-fp32-correctly-rounded-divide-sqrt: Specify that single precision floating-point divide and sqrt used in the program source are correctly rounded (HIP device compilation only)"; its default is not stated on that page.

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/docs/reference/math_api.rst (section "Floating-point Intrinsics")
- https://clang.llvm.org/docs/ClangCommandLineReference.html (`-fhip-fp32-correctly-rounded-divide-sqrt`)

### B5. Fast math, contraction and reassociation defaults under HIP-Clang (and the host compiler)

**Verdict: CONFIRMED.** By default Clang does not reassociate, so a compensated sum survives a normal HIP or host build. HIP's default contraction differs from CUDA-in-Clang and from C++.

- Clang `-ffp-contract`: "fast : enable fusion across statements disregarding pragmas, breaking compliance with the C and C++ standards (default for CUDA)." "on : enable C and C++ standard compliant fusion in the same statement unless dictated by pragmas (default for languages other than CUDA/HIP)". "fast-honor-pragmas : fuse across statements unless dictated by pragmas (default for HIP)".
- "-f[no-]associative-math: Allow floating point operations to be reassociated. Defaults to -fno-associative-math."
- `-ffast-math` lets the compiler assume "Floating-point math obeys regular algebraic rules for real numbers (e.g. + and * are associative ...)" and implies, among others, "-fno-honor-infinities -fno-honor-nans ... -ffinite-math-only -fassociative-math -freciprocal-math -fno-signed-zeros ... -ffp-contract=fast". So `-ffast-math` on a HIP or host build can delete the compensation term.
- "-fno-fast-math sets -ffp-contract to on ( fast for CUDA and HIP)."
- `-ffp-model=precise`: "This is the default behavior."
- AMD's own docs say nothing on defaults: the only fast-math advice in the HIP performance guide is to call intrinsics (`__expf`, `__logf`, `__fsqrt_rn`, `__frcp_rn`) where reduced precision is tolerable.

- https://clang.llvm.org/docs/UsersManual.html (sections on `-ffast-math`, `-ffp-contract`, `-fassociative-math`, `-ffp-model`)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/docs/how-to/performance_guidelines.rst

### B6. Signaling NaN and exceptions on AMD

**Verdict: PARTLY.** The LLVM AMDGPU documentation (maintained with the backend HIP-Clang uses) says that in IEEE mode signaling NaNs are quieted, not trapped, on GFX9 to GFX11. HIP's own docs say nothing about sNaN or floating-point exceptions; `nanf` accepts "quiet", "signaling", "ind" tags.

- AMDGPU kernel descriptor bit 23, "ENABLE_IEEE_MODE", "GFX9-GFX11 (except GFX11.7)": "Wavefront starts execution with IEEE mode enabled. Floating point opcodes that support exception flag gathering will quiet and propagate signaling-NaN inputs per IEEE 754-2008." For GFX12 the bit is "Reserved."
- Bit 21, "ENABLE_DX10_CLAMP": "Used by the vector ALU to force DX10 style treatment of NaN's (when set, clamp NaN to zero, otherwise pass NaN through)." GFX9 to GFX11 only.
- Clang default "-fno-trapping-math behaves identically to -ffp-exception-behavior=ignore . This is the default."
- Not checked: whether any AMD GPU can be made to trap on floating-point exceptions from HIP.

- https://llvm.org/docs/AMDGPUUsage.html (`compute_pgm_rsrc1` bits 21 and 23)
- https://clang.llvm.org/docs/UsersManual.html (`-ftrapping-math`)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/docs/reference/math_api.rst (`nanf`)

---

## Group C: Metal

### C1. Metal's floating types: `float`, `half`, `bfloat`; no `double`

**Verdict: CONFIRMED.** For the explorer and the lab: `float` and `half` are IEEE storage formats, `bfloat` exists from Metal 3.1, and there is no 64-bit float, so the lab's high-precision reference must run on the CPU.

- MSL spec Table 2.1 (quoted in A2). Literal suffixes and vector forms: infra C8.
- Feature tables: "MTLDataType.bfloat (brain float) scalar and vector cases | Metal 3 & 4 | Apple6". MSL spec 8.4: "Table 8.4 applies to all OS, starting with Apple GPU Family 6 or Metal GPU Family 3." Every Apple silicon Mac is Apple7 or later (infra Group C intro), so BF16 arithmetic is available on all of them by the tables.
- Half accuracy: "Table 8.3 applies to iOS and macOS, starting with Apple GPU Family 4 hardware."
- No double: "Metal does not support the double, long long, unsigned long long, and long double data types." (infra C8). PyTorch states the consequence: "The MPS backend does not support torch.float64 ( torch.double ) ... because Metal Shading Language has no double type" and "A computation that genuinely requires double precision has to run on the CPU." MLX: "Arrays with type float64 only work with CPU operations. Using float64 arrays on the GPU will result in an exception."
- The CPU reference on Apple silicon: `long double` buys nothing there. Apple's ARM64 ABI page: "The long double type is a double precision IEEE754 binary floating-point type, which makes it identical to the double type. This behavior contrasts to the standard specification, in which a long double is a quad-precision, IEEE754 binary, floating-point type." Use `double` (E1 has the arithmetic on why that is enough for float sums).
- Host-side FP16 / BF16 emulation for predicting FP16 or BF16 accumulation on the CPU: Clang supports `_Float16` on "64-bit ARM (AArch64) (natively on ARMv8.2a and above)" and `__bf16` on AArch64 "(currently never natively)"; when emulated, "Clang does not truncate intermediate operands back to their true type unless the operand is the result of an explicit cast or assignment", so write `acc = (_Float16)(acc + x);` one statement per addition, or pass `-ffloat16-excess-precision=none` / `-fbfloat16-excess-precision=none`. We did not check which ARMv8 level a given Apple chip implements.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 2.1; section 8.4)
- https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf (feature availability table)
- https://docs.pytorch.org/docs/2.15/notes/mps.html ("Double-precision types")
- https://ml-explore.github.io/mlx/build/html/python/data_types.html
- https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms
- https://clang.llvm.org/docs/LanguageExtensions.html ("Half-Precision Floating Point")

### C2. How close Metal is to IEEE 754 (A.2 to A.4 on Apple GPUs)

**Verdict: PARTLY.** Metal is explicitly "compliant to a subset of the IEEE 754 standard". Against the book: no signaling NaN (agrees with A7); denormals may be flushed; and round-toward-zero is allowed for arithmetic, so the book's 0.5-ULP figure is not a Metal guarantee.

Quotes (MSL spec chapter 8, pages 368 to 376):

- "Metal is compliant to a subset of the IEEE 754 standard."
- 8.1: "INF must be supported for single-precision, half-precision, and brain floating-point numbers. NaNs must be supported for single-precision, half-precision, and brain floating-point numbers (with fast math disabled). If fast math is enabled the behavior of handling NaN or INF (as inputs or outputs) is undefined. Signaling NaNs are not supported."
- 8.1: "Denormalized single-precision, half-precision, or brain floating-point numbers passed as input to or produced as the output of single-precision, half-precision, or brain floating-point arithmetic operations may be flushed to zero."
- 8.2: "Either round ties to even or round toward zero rounding mode may be supported for single-precision, half-precision, and brain floating-point operations."
- 8.3: "Floating-point exceptions are disabled in Metal."
- 8.4: "The reference value used to compute the ULP value of an arithmetic operation is the infinitely precise result." Table 8.1 (single precision): `x + y`, `x - y`, `x * y`, `1.0 / x`, `x / y`, `fma`, `sqrt`: "Correctly rounded". Table 8.2 (fast math, "which is the default unless you specify -fno-fast-math as a compiler option"): `+`, `-`, `*` still "Correctly rounded"; "1.0 / x <= 1 ulp for x in the domain of 2^-126 to 2^126"; "x / y <= 2.5 ulp for y in the domain of 2^-126 to 2^126"; "sqrt(x) Implemented as x * rsqrt(x) with special cases handled correctly." Half Table 8.3 and bfloat Table 8.4: basic operations "Correctly rounded"; bfloat with fast math (Table 8.5): "1.0 / x <= 0.6 ulp", "x / y <= 0.6 ulp".
- 8.4, the reassociation warning: "the Metal compiler, in fast math mode (see section 1.6.5), may do various optimization like reassociate floating-point operations that may dramatically change results in floating-point. Reassociation may change or ignore the sign of zero, allow optimizations to assume the arguments and result are not NaN or +/-INF, inhibit or create underflow or overflow and thus cannot be in code that relies on rounding behavior such as (x + 2^52) - 2^52, or ordered floating-point comparisons." A compensated sum is exactly such code. (The cross-reference "1.6.5" is a slip in the spec; the math options are section 1.6.3.)
- 8.5: if an operand or result is flushed to zero, "the sign of the zero is undefined."
- 8.6: "Conversions from float to half or to bfloat round the mantissa using the round ties to even rounding mode. When converting a float to a half, denormalized numbers generated for the half data type may not be flushed to zero." "Note that fast math does not change the accuracy of conversion operations."
- COULD NOT VERIFY: which rounding mode, and whether flushing, a given Apple GPU actually uses. Apple documents neither per family. The lab can detect both (E1 suggests the inputs).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (chapter 8, pages 368 to 376)

### C3. Turning off reassociation and contraction for one kernel (the lab's compensated sum on Metal)

**Verdict: CONFIRMED** as documented; the pragma's spelling is still untested by us (ch10 C3).

- MSL spec 1.6.3, the optimizations fast math allows: "No NaNs", "No INFs", "No Signed Zeroes", "Allow Reciprocal", "Allow Reassociation: Allow algebraically equivalent transformations, such as reassociating floating-point operations that may dramatically change the floating-point results.", "Allow Contract: Allow floating-point contraction across statements."
- Modes: fast "no NaNs, no INFs, no signed zeros, allow reciprocal, allow reassociation, and FP contract to be fast"; relaxed "but honors INFs and NaNs. These include no signed zeros, allow reciprocal, allow reassociation, and FP contract to be fast."; safe "disables unsafe floating-point optimizations by preventing the compiler from making any transformations that might affect the results. This sets the FP contract to on." So both candidate defaults (ch05 C7) allow reassociation; only safe forbids it.
- Per-source control: "#pragma METAL fp math_mode([relaxed | safe | fast])", which "can only appear in file or namespace scope, within a language linkage specification, or at the start of a compound statement". Contraction: "#pragma METAL fp contract([off | on | fast])", "You can also use: #pragma STDC FP_CONTRACT OFF", or the option "-ffp-contract=off".
- New in Metal 4.1: "-fmetal-rtz-fp-conversion" changes "the default rounding mode for float-to-float conversions from RTNE (round to nearest, ties to even) to RTZ (round toward zero)".
- Feature tables: "Relaxed math | Metal 3 & 4 | Apple4".
- Project fact (ch10 C3): `labs/common/src/backend_metal.mm` creates `[MTLCompileOptions new]` and sets only `preprocessorMacros` before `newLibraryWithSource:options:error:`. So a lab kernel gets the platform default (fast or relaxed) unless `kernel.metal` carries the pragma. The lab should test the compensated kernel both ways; whether Apple's compiler actually deletes the compensation under the default is COULD NOT VERIFY from documents and is a measurement.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 1.6.3, pages 15 to 16)
- https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf

### C4. FP8, FP4 and MX block scaling in Metal 4.1

**Verdict: PARTLY.** New since infra C8/C10 were written: MSL 4.1 defines OCP-style FP8 (E4M3, E5M2), FP4 (E2M1) and UE8M0 as packed storage formats and tensor element types, with a 32-element block-scale plane. There is no scalar FP8 arithmetic type. Two of Apple's format tables contain errors, and Apple does not say which GPU families support the formats.

- MSL spec 2.21 "Packed Numeric Type": "All OS: Metal 4.0 with SDK 26.4 or later support packed numeric. All OS: Metal 4.1 extends the number of format types." Table 2.14: `metal_fp4_e2m1_format` "4-bit signed floating-point (1-bit sign, 2-bit exponent, 1-bit mantissa)", `metal_fp8_e4m3_format`, `metal_fp8_e5m2_format`, `metal_fp8_ue8m0_format` "8-bit unsigned floating-point (8-bit exponent)", each "All OS: Metal 4.1". A format type "is a tag type", "a compile-time marker that carries no data and has no runtime behavior".
- Conversion: `vec<T, N> unpack(packed_numeric_type<Format, N> v);` and `pack<Format, Rm, Sm>(vec<T, N> v)` with "T is the input floating-point that can be float, half, or bfloat"; rounding modes `to_nearest_even`, `to_nearest_away`, `toward_zero`, `toward_pos_inf`, `toward_neg_inf`; saturation `none`, `saturate`, `symmetric_saturate`. Table 2.20 defaults: E2M1 "to_nearest_even / saturate", E4M3 "to_nearest_even / saturate", E5M2 "to_nearest_even / none".
- Block scaling (2.22.2.2): "All OS: Metal 4.1 and later support tensor_blockwise." "Blockwise planes are particularly useful for quantized tensor formats, where scale factors apply to blocks of quantized values." Element type: "Only metal_fp8_ue8m0_format is supported." Block sizes: "The first dimension must be 32. All other dimensions must be 1." The spec's example is `using mxfp4_tensor = tensor<metal_fp4_e2m1_format, Extents, Descriptor, tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>>;`. This is OCP MXFP4's layout (A4).
- Matrix multiply: MPP `matmul2d` Table 7.3 lists, for "Metal 4.1", `half` x `metal_fp4_e2m1_format` / `metal_fp8_e4m3_format` / `metal_fp8_e5m2_format` -> `half` or `float`, and FP4 x FP4, FP8 x FP8 -> `half` or `float`. The descriptor has "relaxed_precision ... Relaxed precision allows the operation to truncate the mantissa before the multiplication. The default is false." (the Metal counterpart of TF32, A11).
- Host API (macOS 27.0 SDK, `MTLTensor.h`): `MTLTensorDataTypeMetalFloat8E5M2 = 141`, `MTLTensorDataTypeMetalFloat8E4M3 = 142`, `MTLTensorDataTypeMetalFloat4E2M1 = 148`, `MTLTensorDataTypeMetalFloat8UE8M0 = 145`, each `API_AVAILABLE(macos(27.0), ios(27.0))`. Feature tables, "Tensor limits by data type": MetalFloat4E2M1, MetalFloat8E5M2, MetalFloat8E4M3 "Supports block scaling: Yes"; MetalFloat8UE8M0 listed under "Tensor block scale data types".
- Errors in MSL spec Table 2.16 and 2.15 (page 80), checked against OCP (A3, A4) and against the same table's own rows: E5M2 NaN is printed "S 11111 {00, 10, 11}" (should be {01, 10, 11}; 00 is the infinity pattern the row above gives); E5M2 zero is printed "S 0000 00" (one exponent bit missing); FP4 min normal is printed "S 01 1 = ±2^0 * 1.0" (the pattern S 01 1 is 1.5; min normal is S 01 0). Use OCP's tables for the explorer.
- COULD NOT VERIFY: which Apple GPU families support packed numeric FP8/FP4 or blockwise tensors. The spec says "See the Metal Feature Set Tables to determine which GPUs support this feature", but the October 23, 2026 tables have no packed-numeric row; the "Tensors" row says "Metal 4 | Apple7". Nor does Apple say what hardware executes `matmul2d` on these types (infra C10).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 2.21, 2.22.2.2, Tables 2.14 to 2.20, 7.3, 7.4; pages 79 to 89, 351 to 354)
- https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf ("Tensor limits by data type", page 17)
- Local header: `Metal.framework/Headers/MTLTensor.h` (macOS 27.0 SDK), lines 18 to 72

---

## Group D: How PyTorch, MLX and llama.cpp handle the same things (goal 3)

### D1. PyTorch: accumulator types, cascade summation, documented numerical behaviour, narrow dtypes

**Verdict: CONFIRMED** (source and docs read today).

- Accumulator type per device (`torch/headeronly/core/AccumulateType.h`, `main`, last commit 2026-10-13), lines 96 to 163: `MPS_ACC_TYPE(BFloat16, float)`, `(Half, float)`, `(Float8_e4m3fn, float)`, `(float, float)`, `(double, float)`; `CUDA_ACC_TYPE(BFloat16, float)`, `(Half, float)`, `(Float8_e4m3fn, float)`, `(float, float)`, `(double, double)`; `CPU_ACC_TYPE(BFloat16, float)`, `(Half, float)`, `(Float8_e4m3fn, float)`, `(float, double)`, `(double, double)`. The header's own comment: "Accumulation is the most well-known intermediate computation (it is of great importance for sum reduction and matrix multiply, for example)". So a float sum accumulates in double on CPU and in float on CUDA and MPS: one source of CPU/GPU differences.
- CPU sum is a cascade (multi-level pairwise) sum (`aten/src/ATen/native/cpu/SumKernel.cpp`, `main`, last commit 2026-11-18, lines 340 to 347): "This algorithm calculates the sum without loss of precision over large axes. It does this by chunking the sum into groups of 16 or more elements. The sums of these chunks are also summed in chunks and so on until there is just a single sum value remaining. This means only numbers of a similar order of magnitude are added together, thus minimising rounding errors." And lines 66 to 68: "When summing float16 or BFloat16, addition has to be performed in float since that's all the hardware supports." This is the book's pre-sorting idea achieved by grouping instead of sorting.
- Numerical accuracy note (docs 2.15, last updated Sep 02, 2026): "floating point addition and multiplication are not associative, so the order of the operations affects the results. Because of this, PyTorch is not guaranteed to produce bitwise identical results for floating point computations that are mathematically identical." "A.sum(-1)[0] is not guaranteed to be bitwise equal to A[:,0].sum()". "Different backends may use different algorithms, reduction orders, or accumulation precision for the same operation."
- The accumulator-width switches: "Half-precision GEMM operations are typically done with intermediate accumulations (reduction) in single-precision for numerical accuracy and improved resilience to overflow. For performance, certain GPU architectures, especially more recent ones, allow a few truncations of the intermediate accumulation results to the reduced precision (e.g., half-precision)." Off with `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False`; "A similar flag exists for BF16 GEMM operations and is turned on by default." SDPA math backend: "the default behavior now involves upcasting FP16/BF16 inputs to FP32."
- AMD-specific notes in the same page: MI200 FP16/BF16 matrix instructions flush denormals (B3); "On AMD Instinct MI300 GPUs (gfx942, CDNA3), the v_mfma_f32_*_xf32 matrix instructions used by hipBLASLt's TF32 path perform round-down accumulation. ... NVIDIA's TF32 implementation rounds to nearest". (PyTorch's statements, not AMD's.)
- Narrow dtypes (tensor attributes, 2.15): `torch.float8_e4m3fn` "S-E-M 1-4-3", `torch.float8_e5m2` "S-E-M 1-5-2", `torch.float8_e4m3fnuz`, `torch.float8_e5m2fnuz`, `torch.float8_e8m0fnu` "S-E-M 0-8-0", `torch.float4_e2m1fn_x2` "packed 4-bit floating point, S-E-M 1-2-1". Suffixes: "“f” - finite value encodings only, no infinity; “n” - nan value encodings differ from the IEEE spec; “uz” - “unsigned zero” only, i.e. no negative zero encoding". These are "shell" dtypes: "Ops that peek inside the data elements such as casting, matrix multiplication, nan/inf checks are supported only on a case by case basis".
- MPS: no float64 (C1).

- https://github.com/pytorch/pytorch/blob/main/torch/headeronly/core/AccumulateType.h
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cpu/SumKernel.cpp
- https://docs.pytorch.org/docs/2.15/notes/numerical_accuracy.html
- https://docs.pytorch.org/docs/2.15/tensor_attributes.html

### D2. MLX: softmax accumulates in the input type unless `precise=True`; half sums stay half; MX quantization modes

**Verdict: CONFIRMED** (docs and source read today; the reduce accumulator is our reading of the kernel).

- Docs (MLX 0.32.4), `mlx.core.softmax(a, axis, *, precise=False, ...)`: "precise ( bool , optional ) – Accumulate in float32 for inputs of lower precision. Otherwise the accumulation type matches the input, which can lose precision over long reduction axes. Default: False ."
- Source (`mlx/backend/metal/kernels/softmax.metal`, lines 20 to 24): `instantiate_softmax(float16, half)`, `instantiate_softmax(bfloat16, bfloat16_t)`, and `instantiate_softmax_precise(float16, half)` which instantiates the kernel with accumulator `float`; `softmax.h` declares `template <typename T, typename AccT = T, ...>`. So the default half softmax keeps its running max and normalizer in `half`. This is the lab's FP16-accumulator question inside the M4 softmax.
- Sum (`mlx/backend/metal/kernels/reduce.metal`, lines 143 to 156): `instantiate_reduce_functions(name, float16, float16_t, float16_t, op)`; the all-reduce kernel (`reduction/reduce_all.h`, lines 21 to 60) accumulates `U total = Op::init; ... total = op(cast_to<U>(in[i]), total);` with `U` the output type. Our reading: `mx.sum` of a float16 array accumulates in float16 on Metal (sequential per thread, then `simd_reduce`), unlike PyTorch (D1).
- Quantization (`mlx.core.quantize`): "The supported quantization modes are "affine" , "mxfp4" , "mxfp8" , and "nvfp4"". Table: mxfp4 group 32, 4 bits, scale e8m0, no bias; mxfp8 group 32, 8 bits, e8m0; nvfp4 group 16, 4 bits, e4m3. "For the "mx" modes, the group size must be 32 . For "nvfp4" the group size must be 16. The elements are quantized to 4-bit or 8-bit precision floating-point values: E2M1 for "fp4" and E4M3 for "fp8" ."
- float64: CPU only (C1).

- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.softmax.html
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.quantize.html
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/softmax.metal
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/softmax.h (last commit 2025-05-21)
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/reduce.metal
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/reduction/reduce_all.h (last commit 2026-08-18)

### D3. llama.cpp: MXFP4 and NVFP4 blocks, E8M0 and BF16 conversion, required accumulator precision

**Verdict: CONFIRMED** (source on `master` read today).

- Blocks (`ggml/src/ggml-common.h`, lines 214 to 227): `#define QK_MXFP4 32` with `uint8_t e; // E8M0` and `uint8_t qs[QK_MXFP4/2];` (17 bytes per 32 weights); `#define QK_NVFP4 64`, `#define QK_NVFP4_SUB 16`, `uint8_t d[QK_NVFP4/QK_NVFP4_SUB]; // UE4M3 scales (4 bytes, one per 16-element sub-block)`.
- The E2M1 table, stored doubled so it fits `int8_t` (lines 1124 to 1129): "// e2m1 values (doubled), shared by MXFP4 and NVFP4" `0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12`, compensated by `ggml_e8m0_to_fp32_half` ("Equal to ggml_e8m0_to_fp32/2"). The values are OCP's E2M1 magnitudes (A4) times two.
- E8M0 decode (`ggml/src/ggml-impl.h`, line 458 on): code 0 maps to the subnormal float 2^-127 (`bits = 0x00400000`); the NaN code is deliberately not handled ("// note: disabled as we don't need to handle NaNs").
- FP32 to BF16 (`ggml-impl.h`, lines 683 to 701): "Floats shall round to nearest even, and NANs shall be quiet. Subnormals aren't flushed to zero"; the rounding is `h.bits = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;` and NaN is `(u.i >> 16) | 64 /* force to quiet */`. A compact worked example of round-to-nearest-even on raw bits for the A.4 lesson.
- Required accumulator precision is an explicit graph property (`ggml/include/ggml.h`, lines 1258 to 1268): "set the minimum required accumulator type for the implementation to use during the compute ... GGML_PREC_F32 - requires accumulation of the results in F32; GGML_PREC_BF16 - can accumulate the results in BF16, F32; GGML_PREC_F16 - can accumulate the results in F16, F32". `src/llama-graph.cpp` (last commit 2026-11-19): flash attention gets `ggml_prec_set_acc(cur, GGML_PREC_F32);` (line 3231), the non-flash `kq = ggml_mul_mat(ctx0, k, q)` gets `ggml_prec_set_acc(kq, GGML_PREC_BF16);` (line 3255), and some MoE router logits get `GGML_PREC_F32` "the reference computes the router in F32" (lines 2405 to 2412).
- CPU sums (`ggml/src/ggml-cpu/vec.h`, last commit 2026-11-18): `typedef double ggml_float;` (line 15); `ggml_vec_sum_f32` accumulates sequentially in `ggml_float` (double) unless built with Accelerate, where it calls `vDSP_sve` (lines 1664 to 1674); `ggml_vec_sum_f16_ggf` accumulates in `float` (lines 1694 to 1700).
- HIP mapping (`ggml/src/ggml-cuda/vendors/hip.h`, last commit 2026-11-16, lines 255 to 258): `typedef __hip_bfloat16 nv_bfloat16;`, `typedef __hip_fp8_e4m3 __nv_fp8_e4m3;`, `#define FP8_AVAILABLE`. CUDA side (`vendors/cuda.h`): `cuda_fp8.h` only `#if CUDART_VERSION >= 11080`, `cuda_fp4.h` only `#if CUDART_VERSION >= 12080`.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-impl.h
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h
- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-graph.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/vec.h
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vendors/hip.h
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vendors/cuda.h

### D4. NumPy's sum: partial pairwise summation (a bridge the reader already uses)

**Verdict: CONFIRMED.** NumPy v2.5 `numpy.sum`: "For floating point numbers the numerical precision of sum (and np.add.reduce ) is in general limited by directly adding each number individually to the result causing rounding errors in every step. However, often numpy will use a numerically better approach (partial pairwise summation) leading to improved precision in many use-cases. This improved precision is always provided when no axis is given. When axis is given, it will depend on which axis is summed. Technically, to provide the best speed possible, the improved precision is only used when the summation is along the fast axis in memory." And: "In contrast to NumPy, Python's math.fsum function uses a slower but more precise approach to summation."

- https://numpy.org/doc/stable/reference/generated/numpy.sum.html

---

## Group E: our arithmetic, and project facts that bear on the lab

### E1. NaN counts, and why a double accumulator is an adequate reference for float sums

**Verdict: CONFIRMED** (our arithmetic on the vendor definitions above).

- NaN count in single precision (book p. 591, "2^22 - 1 possible NaN bit patterns"; flagged in the notes' "Unclear" list). With 23 mantissa bits and the top one choosing quiet or signaling (A7): per sign, quiet NaNs number 2^22 (top bit 1, the other 22 free) and signaling NaNs 2^22 - 1 (top bit 0, the rest not all zero, since all zero is infinity). Total over both signs 2 × (2^23 - 1) = 2^24 - 2, which is NVIDIA's "2^(mantissa + 1) - 2". So the book's 2^22 - 1 is the signaling count for one sign.
- Reference precision: summing n floats sequentially in double has a worst-case relative error bound of about (n - 1) × 2^-53 times the sum of absolute values (the standard bound; ours to state, not a vendor fact). For n up to 2^24 that is at most about 2^-29 of the absolute sum, below one float ULP of the result unless the sum cancels heavily. For the lab's "mixed magnitudes with cancellation" data, use a compensated sum in double on the CPU, or sort by magnitude first, so the reference stays far more accurate than any GPU variant. Report errors in float ULPs of the reference and in relative terms.
- Detecting Metal's rounding mode and flushing (C2), our suggestion: on the device compute `a + b` with `a = 1.0f`, `b = 0x1.8p-24f` (three quarters of an ULP of 1): round-to-nearest gives `1.0f + 0x1p-23f`, round-toward-zero gives `1.0f`. And `c * 0.5f` with `c = 0x1p-126f`: flushing gives 0, IEEE gives the subnormal 2^-127. Read `a`, `b`, `c` from a buffer, or the compiler folds the expressions at build time on the host.

### E2. Project facts the lab design depends on (from this repository, not vendors)

**Verdict: CONFIRMED** (files read today).

- The CUDA emulators do not model half precision: `labs/README.md` lists "half and bfloat16 operands" as not modelled for the warp intrinsics, and "half, bfloat16, float2/float4 and 128-bit forms" for atomics; `labs/common/include/pmpp/emu/cuda_emu.h` line 78: "not modelled: __half / __nv_bfloat16 operands". The CUDA front-end stub (`labs/common/include/pmpp/check/cuda_stub.h` line 123) omits the `__half` / `__nv_bfloat16` overloads. So an FP16 or BF16 accumulator variant of the `.cu` lab cannot run on `emu` or `race`, or pass the front-end check, without changes to shared lab infrastructure. Options: keep the CUDA-family lab in FP32 (sequential, tree, compensated) and do the FP16/BF16 accumulator on Metal (`half`, `bfloat`) plus a CPU emulation with `_Float16` / `__bf16` (C1); or report the needed shared change.
  - Update 2026-11-21 (course consistency pass): the emulators and the front-end stub now model `__half` (`pmpp/emu/cuda_fp16_emu.h`, `cuda_stub.h`) but not `__nv_bfloat16`. Lab A.1's `PMPP_NARROW_ACC` guard still keeps both 16-bit sums out of those builds, so this lab builds no 16-bit sum there.
- `labs/common/include/pmpp/vendors/hip.h` notes the header rename for half (`<cuda_fp16.h>` is `<hip/hip_fp16.h>`) in a comment only; BF16 also needs a type rename (B1).
- `labs/common/src/backend_metal.mm` compiles with default `MTLCompileOptions` and only `preprocessorMacros` (C3, ch10 C3), so the Metal compensated sum needs `#pragma METAL fp math_mode(safe)` in `kernel.metal`, and the spelling should be compile-tested on the reader's machine before the README promises it.

---

## Summary of corrections

Every WRONG, PARTLY and COULD NOT VERIFY item, one line each:

- A4: the OCP MX specification is now Version 1.1 (October 9, 2026), retitled "OCP MicroXcaling (MX) Specification", with E8M0 renamed UE8M0; the book cites v1.0 (2023). Adoption is confirmed: NVIDIA PTX and cuBLAS block scaling, AMD gfx950 HIP extensions, Metal 4.1 `tensor_blockwise`.
- A6: `--ftz=true` means flush denormals to zero, that is denormals off; default is `false`; it affects single precision only and is implied by `--use_fast_math`. NVIDIA gives no cycle cost for GPU denormals ("computationally more expensive" only).
- A11: both H100 figures on p. 593 are "with sparsity", and 989 TFLOPS is TF32 ("same range as .f32 and reduced precision (>=10 bits)"), not IEEE single precision.
- A13 (could not verify): NVIDIA's docs do not settle whether nvcc reassociates float additions; write the CUDA compensated sum with `__fadd_rn` / `__fsub_rn` and let the measurement show whether it worked.
- A14 (could not verify): "special function units" does not appear in NVIDIA's current floating-point or compute-capability appendices; NVIDIA says intrinsics "map to fewer native instructions".
- A15 (WRONG): "partial pivoting" in LAPACK, cuSOLVER, rocSOLVER and MPS means row interchanges within the current column (A = P L U); row-and-column swapping is "complete pivoting" (A = P L U Q). The book's "localized set of equations" definition matches neither. cuSOLVER: LU without pivoting "is not numerically stable".
- B1: HIP's BF16 type is `__hip_bfloat16` from `<hip/hip_bf16.h>`, not `__nv_bfloat16`; AMD's two pages disagree on whether BF16 arithmetic is native on every architecture, and the ROCm precision page names the types `half` / `bfloat16`.
- B3 (could not verify): no AMD or Clang page states HIP's default FP32 denormal mode; Clang offers `-fgpu-flush-denormals-to-zero`, the hardware has per-kernel denorm modes, and PyTorch documents that MI200 FP16/BF16 matrix instructions flush denormals.
- B4: on AMD only round-to-nearest-even is available by default; `__fadd_rz` and friends need `OCML_BASIC_ROUNDED_OPERATIONS`; AMD documents measured ULP differences against the C++ library, not guarantees.
- B6: AMD's documented behaviour (LLVM AMDGPU, IEEE mode, GFX9 to GFX11) quiets signaling NaNs; HIP's own docs are silent; GFX12 marks the IEEE-mode bit reserved.
- C2: Metal is "compliant to a subset of the IEEE 754 standard": denormals "may be flushed to zero", arithmetic may round "toward zero" instead of to nearest even, and fast math makes INF/NaN undefined and permits reassociation that breaks code like (x + 2^52) - 2^52. The book's 0.5 ULP is not a Metal guarantee.
- C4: Metal 4.1 adds FP8 (E4M3, E5M2), FP4 (E2M1) and UE8M0 as packed formats and tensor types with 32-element block scales, but no scalar FP8 arithmetic; MSL spec Tables 2.15 and 2.16 misprint the E5M2 NaN patterns, the E5M2 zero and the FP4 min-normal pattern; no GPU-family requirement is documented.

Status changes worth knowing even where the verdict is CONFIRMED:

- A1: IEEE 754-2019 is still active; a revision (P754) and an ML-formats draft (P3109) are open projects, not standards.
- A3: OFP8 is at Revision 1.1 (adds rounding-at-maximum examples: E4M3 inputs up to 464 round down to 448; E5M2 61,440 overflows).
- A5: NVIDIA's cuBLAS FP4 GEMMs use 16-element blocks with UE4M3 scales plus a tensor-wide scale, not MXFP4; cuBLAS rounds UE8M0 scales up, which "differs from the OCP MXFP suggested rounding scheme".
- A8: the 2-ULP division the book calls historic is still available on purpose: `-prec-div=false` / `--use_fast_math` gives `__fdividef`, 2 ULP.
- A10: FP8 tensor cores begin at compute capability 8.9, not with the H100 (9.0).
- A12: the book's reference [4] is the 2011 white paper; the maintained text is CUDA Programming Guide section 5.6.
- D1, D2: PyTorch accumulates half and bfloat16 in float on every device and float in double on CPU; MLX's default softmax and sum on half accumulate in half. Same op, different accumulator: a direct illustration of §A.6.

## Where the backends really differ

One line per cell. "Same" means the same source text, according to the documentation; nothing here has been run on CUDA or AMD hardware by us.

| Feature | CUDA | HIP (ROCm 10.2) | Metal (MSL 4.1) |
|---|---|---|---|
| IEEE 754 status | "All GPU devices follow the IEEE 754-2019 standard" with listed limitations (no dynamic rounding mode, exceptions masked) | No AMD statement found for arithmetic; storage formats "conform to the IEEE 754" | "compliant to a subset of the IEEE 754 standard" |
| Basic `+ - * /` on `float` | Correctly rounded ("maximum ULP error of zero") with default flags | `__fadd_rn` etc. measured 0 ULP vs C++; operators not documented | `+ - *` correctly rounded; `/` and `1/x` correctly rounded only with fast math off (2.5 and 1 ulp with it) |
| Rounding mode | Round to nearest even by default; `_rz/_ru/_rd` intrinsics per operation | Nearest even only, unless `OCML_BASIC_ROUNDED_OPERATIONS` | "Either round ties to even or round toward zero ... may be supported"; not selectable |
| FP32 denormals | Kept by default; `-ftz=true` flushes (single precision only); global float atomics always flush | Default undocumented; `-fgpu-flush-denormals-to-zero` flushes; per-kernel hardware mode | "may be flushed to zero" (float, half, bfloat) |
| Signaling NaN, exceptions | sNaN encodings exist but "are not signaling"; no exception detection | Quieted in IEEE mode (LLVM AMDGPU, GFX9 to GFX11); HIP docs silent | "Signaling NaNs are not supported"; "Floating-point exceptions are disabled" |
| Fast-math switch | `--use_fast_math` = `-ftz=true -prec-div=false -prec-sqrt=false -fmad=true` + intrinsic substitution; off by default | Clang `-ffast-math` (adds reassociation, no-NaN/INF); off by default | Default is fast or relaxed (documented three ways); both allow reassociation; `safe` forbids it |
| Contraction into FMA by default | Yes (`--fmad=true`) | Yes, `fast-honor-pragmas` (default for HIP) | Yes, "By default, the compiler allows floating-point contractions" |
| Protecting a compensated sum | `__fadd_rn` / `__fsub_rn` (never fused); reassociation by nvcc undocumented | Default `-fno-associative-math`; avoid `-ffast-math` | `#pragma METAL fp math_mode(safe)` in the kernel, or `mathMode = .safe` |
| FP16 type | `__half`, `<cuda_fp16.h>`, operators, `__hadd` RNE | `__half`, `<hip/hip_fp16.h>` | `half`, built in |
| BF16 type | `__nv_bfloat16`, `<cuda_bf16.h>`, compute capability 8.0 or higher | `__hip_bfloat16`, `<hip/hip_bf16.h>` (needs a typedef) | `bfloat`, Metal 3.1, Apple6 and later |
| FP8 | `__nv_fp8_e4m3`, `__nv_fp8_e5m2` (`cuda_fp8.h`), OCP encodings; tensor cores from 8.9 | `__hip_fp8_e4m3` / `_e5m2` (OCP, CDNA4 and RDNA4) and `_fnuz` (CDNA3, bias +1, no inf, no -0) | `metal_fp8_e4m3_format`, `metal_fp8_e5m2_format`: packed storage and tensor element types only, Metal 4.1 |
| FP6 / FP4 | `__nv_fp6_e2m3`, `__nv_fp6_e3m2`, `__nv_fp4_e2m1`; tensor cores from 10.0 | `__hip_fp6_*`, `__hip_fp4_e2m1`, CDNA4 only | `metal_fp4_e2m1_format` (Metal 4.1); no FP6 |
| Block scaling (MX) | PTX `.kind::mxf8f6f4` / `mxf4` (UE8M0, 32); cuBLAS VEC32_UE8M0 for FP8; FP4 uses VEC16_UE4M3 | `__amd_scale_t` (E8M0) conversion APIs, gfx950 | `tensor_blockwise` with `metal_fp8_ue8m0_format`, block 32 |
| Default saturation when narrowing to FP8 E4M3 | `__NV_SATFINITE` in constructors | `__HIP_SATFINITE` in constructors | `saturate` for E4M3, `none` for E5M2 |
| 64-bit float for a reference | `double` | `double` | None; reference runs on the CPU (and `long double` is `double` on Apple arm64) |
| LU with pivoting library | cuSOLVER `getrf` (partial pivoting; no pivoting "not numerically stable") | rocSOLVER `getrf`, `getrf_npvt` | MPS `MPSMatrixDecompositionLU` (partial pivoting); Accelerate LAPACK on the CPU |
| Matmul accumulator knobs in frameworks | PyTorch `allow_fp16/bf16_reduced_precision_reduction`; TF32 via `fp32_precision` | Same PyTorch flags; MI300 TF32 accumulates rounding down (PyTorch) | MLX `softmax(precise=True)`; MPP `matmul2d` `relaxed_precision` |
