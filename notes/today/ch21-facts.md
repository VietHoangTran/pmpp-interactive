# Chapter 21 fact check: electrostatic potential map, scatter versus gather, coarsening, binning (CUDA, HIP, Metal, real backends)

Checked on 2026-11-20 against the live sources listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md` and `notes/today/infra-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY, and judges the statement named in the item title (usually the book's, quoted with its page from `notes/ch21.md`). **Quote** is copied from the source with line breaks removed.
- "Book" means `notes/ch21.md` with its page numbers. Nothing here is taken from memory of the book.
- Chapter 21 is a Condensed chapter. Group A checks the book's dated or device-specific statements and the CUDA features its kernels use. Groups B and C give the HIP and Metal counterparts, citing earlier files wherever they already settle a point. Group D is the bridge for the reader's third goal (embedding gradients and MoE dispatch, where the same scatter-versus-gather and binning choices appear in llama.cpp, MLX and PyTorch). Group E holds the chapter's arithmetic, recomputed by us.
- Versions the sources reported on the check date: CUDA Programming Guide v13.5 (math appendix "Last updated on Oct 31, 2026"), CUDA C++ Best Practices Guide v13.5 ("Last updated on Nov 02, 2026"), Nsight Compute v2026.4.0; the archived CUDA Programming Guide Version 1.0 (6/23/2007), read from NVIDIA's download server as a historical reference only; HIP docs "HIP 7.17.0" (ROCm Core SDK 10.2.0); Clang command-line reference and `clang/include/clang/Options/Options.td` on LLVM `main`; Metal Shading Language Specification Version 4.1 (PDF dated 2026-06-04); PyTorch docs 2.15; MLX docs 0.32.4. Project files were read on their default branches today (llama.cpp `master` head `184862801`, 2026-11-20); the last-commit date of each file is given where it is cited. VMD pages were read from `www.ks.uiuc.edu`; their own "last modified" dates are given.
- Not checked: anything by running code. Every statement is from documentation, headers, papers or project source. Group E is our arithmetic.
- **Device figures in this file are evidence for a verdict, not lesson content.** VMD's page quotes timings and GFLOPS for 2008-era GPUs; none of them belong in a lesson.

Totals: 17 CONFIRMED, 0 WRONG, 9 PARTLY, 1 COULD NOT VERIFY (27 items: A1 to A10, B1 to B3, C1 to C4, D1 to D6, E1 to E4). Every PARTLY and COULD NOT VERIFY item, and every sub-claim that could not be verified, is listed under "Summary of corrections".

---

## Already settled elsewhere (cite, do not re-check)

| Topic chapter 21 touches | Where it is settled |
|---|---|
| `__constant__` at file scope, `cudaMemcpyToSymbol` signatures, 64 KB on every listed compute capability, `totalConstMem` | ch05 A4, infra A5, ch07 A1, A2 |
| Constant cache: one access per distinct address in a warp, "as fast as a register access" when all lanes read one address, 8 KB working set per SM | ch07 A3 |
| Kernel parameters go through constant memory, up to 32,764 bytes (CUDA 12.1 and later); `__grid_constant__` | ch07 A4 |
| `const __restrict__` read-only load path, `__ldg` | ch07 A6, B4 |
| HIP `__constant__` and `hipMemcpyToSymbol`; no documented size; `totalConstMem` reports the largest allocation; the scalar cache helps only wavefront-uniform reads | infra B5, ch07 B1 |
| Metal read-only tables: a `constant T*` buffer the host fills, or an initialised program-scope `constant` array; no symbol copy; no documented size or constant cache; `setBytes` under 4 KB; WWDC16 advice on the `constant` address space | infra C6, ch07 C1, C2, ch03 C5 |
| Legacy `atomicAdd` on `float`, NVIDIA's "legacy" label, same-address atomics serialised in undefined order | infra A1, A2, ch09 A3 |
| HIP float `atomicAdd` compiled safe (possibly a CAS loop) unless `-munsafe-fp-atomics`; no hardware float atomic add on RDNA1 and CDNA1 | infra B1 |
| MSL `atomic_float`: add and sub only, device memory (threadgroup from Metal 4.1), Apple7 and later | infra C1 |
| Gather versus scatter: the mechanism is documented, a speed ratio is not | ch22 A1 |
| Where atomics execute: L2 on NVIDIA and AMD; Apple documents neither place nor cost | ch09 A4, B2, C4 |
| Floating-point addition is not associative, so summation order changes results | ch10 A9 |
| Coalescing: NVIDIA counts 32-byte sectors per warp; AMD groups a wavefront's addresses into cache lines; Apple documents no rule | ch06 A2, A3, B1, C1 |
| Warp width: 32 on NVIDIA, 64 or 32 on AMD, from the pipeline on Metal | ch05 A5, B5, C3 |
| Thread coarsening as NVIDIA advice (Best Practices section 11.4) | ch06 A15 |
| Occupancy and registers: definition, `--resource-usage` (NVIDIA and AMD); Metal defines occupancy in Xcode but exposes no API or register count | ch04 A13, A14, B10, ch06 A10, C5 |
| 16-byte vector access: NVIDIA's blog teaches the `float4` cast with an alignment rule; AMD documents aligned `float4`; Metal: declare the buffer `device float4*` or `packed_float4*` | infra A6, B6, C7, ch06 A11, B5, C3 |
| Control divergence: NVIDIA's and AMD's documented model; short branches may be predicated so that "no warp can ever diverge"; Apple documents nothing | ch04 A9, B7, C4, ch06 A14 |
| Shared memory, barriers and their HIP and Metal spellings (for the cooperative bin load of p. 526) | ch05 A1, A7, B4, C1, C2, C8 |
| Kernel launch is asynchronous with respect to the host (the overflow pass of p. 527 can run while the device works); HIP the same; Metal `commit` is asynchronous | ch02 A5, B3, C4 |
| Streams, events and their Metal analogue (overlapping sub-volumes, p. 527) | infra A8, B8, C12 |
| Pitched 2D allocation (`cudaMallocPitch`, `hipMallocPitch`) | ch06 A3 |
| 2D thread index on Metal; `dispatchThreads` gives smaller edge threadgroups instead of padded threads | ch05 C4, C5 |
| Apple unified memory: the host reads a shared buffer without a copy once the command buffer has completed | ch05 C6 |
| MSL's default math mode is documented three different ways; MSL is not IEEE-strict by default | ch05 C7 |
| `x[idx] += v` with repeated indices applies once in MLX and PyTorch; accumulating scatters are separate APIs; MLX and PyTorch emulate missing Metal atomics with CAS loops | ch09 D5, C2 |

---

## Group A: the book's dated or device-specific statements, and the CUDA features the chapter uses

### A1. VMD, and the parts of VMD "accelerated with CUDA" (pp. 513 to 514)

**Verdict: CONFIRMED.** All four applications the book lists are named as GPU-accelerated on the project's own pages, which are old (GPU page last modified 24 Nov 2020; the online user guide is the 1.9.3 manual). VMD is still released: 2.0.0 on 2026-03-25.

- VMD home page (last modified 19 Feb 2025): "VMD is a molecular visualization program for displaying, animating, and analyzing large biomolecular systems using 3-D graphics and built-in scripting." "VMD supports computers running MacOS X, Unix, or Windows, is distributed free of charge, and includes source code."
- Download page: "Version 2.0.0 (2026-03-25) Platforms:Release version VMD 2.0.0" with "LINUX_64 (RHEL 8+) OpenGL, CUDA, OptiX RTX, RTX RTRT"; "Version 2.0.1 LATEST ALPHA (2026-04-21)". The 2.0.0 beta and alpha also list "MacOS 11.x, ARM64 (64-bit "Apple Silicon" M1 or later Macs)", with no CUDA in that platform string. Earlier: "Version 1.9.3 (2016-11-30)"; 1.9.4 stayed an alpha ("Version 1.9.4 LATEST ALPHA (2023-06-08)").
- GPU computing page, "GPU-accelerated applications": "Molecular orbital display", "Ion placement", "Multi-GPU Coulomb summation". Ion placement is the book's use of the map (p. 514): "One method for doing this is to place important counterions at minima in the electrostatic potential field generated by the biomolecule of interest, iteratively updating the potential field after each ion is placed."
- User guide, `volmap`: "coulomb, coulombmsm: Creates a map of the electrostatic field of the atom selection, made by computing the non-bonded Coulomb potential from each atom in the selection ... The coulomb map generation is optimized to take advantage of multi-core CPUs and programmable GPUs if they are available". The gas-migration maps of p. 514 are "Implicit Ligand Sampling (volmap ils command)": "Note that if you have a CUDA enabled GPU then your ILS calculation will run about 20 times faster than on a CPU."

New and useful for §21.5, where the book leaves "a separate implicit method" for far atoms undescribed (notes, "Unclear in the PDF"): the same GPU page names the linear-cost method VMD itself uses. "The direct summation of the Coulomb potential from all atoms to every lattice point requires computational work that grows quadratically, proportional to the product of the number of atoms and the number of lattice points. An algorithmic enhancement known as multilevel summation uses hierarchical interpolation of softened pairwise potentials from lattices of increasing coarseness to compute an approximation to the Coulomb potential. The amount of computational work for multilevel summation grows linearly, proportional to the sum of the number of atoms and the number of lattice points." Its accuracy statement: "an average difference from the direct approach demonstrated to be in the range of 0.025% to 0.037%". Lesson use: one `<Beyond>` sentence naming multilevel summation as the method that handles the far field; the timings on that page ("529 seconds on a single GPU" and others) are dated device figures and stay out.

- https://www.ks.uiuc.edu/Research/vmd/
- https://www.ks.uiuc.edu/Development/Download/download.cgi?PackageName=VMD
- https://www.ks.uiuc.edu/Research/gpu/
- https://www.ks.uiuc.edu/Research/vmd/current/ug/node156.html (`volmap`)

### A2. The chapter's three references (p. 528)

**Verdict: CONFIRMED.** Crossref returns all three with the details the notes give.

| Ref. | Crossref record |
|---|---|
| [1] | Humphrey, Dalke, Schulten, "VMD: Visual molecular dynamics", Journal of Molecular Graphics 14(1), 33 to 38, February 1996, doi:10.1016/0263-7855(96)00018-5 |
| [2] | Stone, Phillips, Freddolino, Hardy, Trabuco, Schulten, "Accelerating molecular modeling applications with graphics processors", Journal of Computational Chemistry 28(16), 2618 to 2640, 2007, doi:10.1002/jcc.20829 |
| [3] | Rodrigues, Hardy, Stone, Schulten, Hwu, "GPU acceleration of cutoff pair potentials for molecular modeling applications", Proceedings of the 5th Conference on Computing Frontiers, 273 to 282, May 2008, doi:10.1145/1366230.1366277 |

Both GPU papers date from the first CUDA generation (compute capability 1.x), which matters for A3. The papers themselves were not read (publisher paywalls).

- https://api.crossref.org/works/10.1016/0263-7855(96)00018-5
- https://api.crossref.org/works/10.1002/jcc.20829
- https://api.crossref.org/works/10.1145/1366230.1366277

### A3. Fig. 21.9: "Threads compute up to 8 potentials, skipping by half-warps" (p. 522)

**Verdict: PARTLY.** "Half-warp" is still a defined term, but the coalescing rule it refers to belongs to the first CUDA devices. Today NVIDIA states coalescing per warp, in 32-byte sectors (ch06 A2), and the chapter's own code strides by `blockDim.x`, not by 16.

- Current guide, section 3.2.2.1 "SIMT Execution Model": "A half-warp is either the first or second half of a warp. A quarter-warp is either the first, second, third, or fourth quarter of a warp." The current coalescing rule (Programming Guide 2.3.4.1; Best Practices 10.2.1: "For devices of compute capability 6.0 or higher ... the concurrent accesses of the threads of a warp will coalesce") is stated per warp. "Half warp" survives in prose only: twice in the Best Practices Guide's section 10.2.3.3 example (C = AAᵀ), once about the strided global reads ("how threads in a half warp access elements of A in the second term", followed by "For a warp of threads, col represents sequential columns") and once about shared-memory reads.
- CUDA Programming Guide Version 1.0 (6/23/2007), section 5.1.2.1, pp. 44 to 45: "the global memory addresses simultaneously accessed by each thread of a half-warp during the execution of a single read or write instruction should be arranged so that the memory accesses can be coalesced into a single contiguous, aligned memory access. More precisely, in each half-warp, thread number N within the half-warp should access address HalfWarpBaseAddress + N". And: "We recommend fulfilling the coalescing requirements for the entire warp as opposed to only each of its halves separately because future devices will necessitate it for proper coalescing."
- The same 2007 guide states the constant cache per half-warp too: "For all threads of a half-warp, reading from the constant cache is as fast as reading from a register as long as all threads read the same address." Today's sentence is per warp (ch07 A3).

So the figure's wording matches the 2007 rule and the hardware of references [2] and [3] (A2). That the figure was taken from that work is our inference; its source is not given in the book (notes). For the lesson: present Fig. 21.10's `blockDim.x` stride as the technique, say that the figure's "half-warps" is the 2007 coalescing unit, and do not teach a half-warp rule.

- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html (section 3.2.2.1)
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (sections 10.2.1 and 10.2.3.3)
- https://developer.download.nvidia.cn/compute/cuda/1.0/NVIDIA_CUDA_Programming_Guide_1.0.pdf (sections 5.1.2.1 and 5.1.2.2; NVIDIA's China download server, the only NVIDIA-hosted copy of a pre-2.0 guide that downloaded today)

### A4. Fig. 21.9: "Thread blocks: 64-256 threads" (p. 522)

**Verdict: CONFIRMED** as consistent with NVIDIA's current heuristics. Best Practices Guide section 11.3, "Thread and Block Heuristics":

- "Medium Priority: The number of threads per block should be a multiple of 32 threads, because this provides optimal computing efficiency and facilitates coalescing."
- "A minimum of 64 threads per block should be used, and only if there are multiple concurrent blocks per multiprocessor."
- "Between 128 and 256 threads per block is a good initial range for experimentation with different block sizes."

The book's own worked example on p. 525 uses 8 x 8 x 8 = 512-thread blocks, outside the figure's range; nothing to check, but the explorer should not imply a fixed block size. AMD's counterpart is B3.

- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (section 11.3)

### A5. Atom data in constant memory, fed in chunks; the constant cache "broadcasts" (pp. 517 to 519)

**Verdict: CONFIRMED,** with one consideration the book does not raise.

- Size: 64 KB on every listed compute capability (ch05 A4), so a chunk holds at most 4,096 atoms of four floats, less any other `__constant__` data (E3).
- Broadcast: Best Practices section 10.2.6, "If all threads of a warp access the same location, then constant memory can be as fast as a register access" (ch07 A3). In the gather kernels (Figs. 21.6, 21.8, 21.10) every lane of a warp reads `atoms[n]` to `atoms[n+3]` with the same `n`: the best case NVIDIA describes.
- Working set: Table 34, "Cache working set per SM for constant memory | 8 KB", one value across compute capabilities 7.5 to 12.x. That is 512 atoms, an eighth of a full 64 KB chunk (E3). The book's argument (p. 519) needs only the current atom to be cached for all warps that are reading it; whether warps of one SM that are at different points of the atom loop evict each other's lines is not documented. A lab that varies `CHUNK_SIZE` measures it.
- Alternatives that are newer than the chapter, all documented: passing the chunk by value as a kernel argument, which NVIDIA also places in constant memory, up to 32,764 bytes since CUDA 12.1, about 2,000 atoms (ch07 A4, E3); or a `const float* __restrict__` input on the read-only path (ch07 A6). Not in the book; label `<Beyond>`.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html (Table 34)
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (section 10.2.6)

### A6. The scatter kernel needs `atomicAdd` on `float`, and atomics "significantly reduce" its speed (Fig. 21.5, p. 518)

**Verdict: PARTLY.** `atomicAdd(float*, float)` is current (infra A1, where NVIDIA lists "int, unsigned, unsigned long long, float, double, __half2, __half"), and NVIDIA now calls it a legacy function and recommends `cuda::atomic_ref` (infra A1, A2). That the atomics cost a lot is the documented mechanism (same-address operations serialised, executed at the L2), but no NVIDIA document gives a ratio (ch22 A1).

- Nsight Compute, memory tables: "Each load or store request accesses one or more sectors in the L1 cache. Atomics and reductions are passed through to the L2 cache."
- Two consequences the book does not state, both ours from documented facts. First, the order of same-address atomics "is undefined" (infra A1) and float addition is not associative (ch10 A9), so the scatter kernel's sums can differ from run to run in the last bits, while each gather thread adds its atoms in a fixed order. PyTorch documents exactly this split for its own scatters (D1, D2). Second, Fig. 21.5 also reads constant memory with a different address in every lane (each thread loads its own atom), which is the serialised case of ch07 A3. The listing reads `atoms[n]` inside the grid loops (as Fig. 21.4 does); whether the compiler keeps those values in registers instead is not documented. Either way it is a second cost on top of the atomics, and the gather kernels do not have it.
- The chapter uses the legacy call; its own chapter 9 uses `cuda::atomic_ref` (ch09 A1). The lesson can show the legacy form as printed and name the replacement.

- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html (Memory Tables, "Sectors")
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (section 5.4.5, via infra A1)

### A7. Coarsening uses more registers, which "can reduce occupancy"; here it "stays within the allowed limit" (p. 521)

**Verdict: PARTLY.** The mechanism is confirmed. The book's statement about its own kernel cannot be checked: it gives no register count, compiler or device.

- Table 34, one value across compute capabilities 7.5 to 12.x: "Number of 32-bit registers per SM | 64 K", "Maximum number of 32-bit registers per thread block | 64 K", "Maximum number of 32-bit registers per thread | 255".
- Occupancy from registers: ch04 A13. Coarsening as NVIDIA's own advice, Best Practices section 11.4: "some operations common to each element can be performed by the thread once, amortizing the cost over the number of shared memory elements processed by the thread" (ch06 A15). And the counterweight: "higher occupancy does not always equate to better performance" (ch06 A10).
- How the reader checks the claim on their own build: `nvcc --resource-usage` (ch04 A13), `hipcc --resource-usage` (ch04 B10). Metal exposes no register count; Xcode shows occupancy (ch06 C5).

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html (Table 34)
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (sections 11.3, 11.4)

### A8. Fig. 21.8's writes are uncoalesced; Fig. 21.10's `blockDim.x` stride coalesces them and is faster (pp. 521 to 522)

**Verdict: CONFIRMED** under NVIDIA's documented rule; "faster" is not quantified by the book or by any vendor.

- The rule (ch06 A2): "the concurrent accesses of the threads of a warp will coalesce into a number of transactions equal to the number of 32-byte transactions necessary to service all of the threads of the warp."
- It applies to stores as well as loads, and the kernels' `+=` is both. Nsight Compute: "Each load or store request accesses one or more sectors in the L1 cache." "Sectors/Req ... For warps with 32 active threads, the optimal ratios per access size are: 32-bit: 4, 64-bit: 8, 128-bit: 16. ... Higher numbers can imply uncoalesced memory accesses and will result in increased memory traffic."
- By that rule each write statement of a (corrected) Fig. 21.8 touches 16 sectors per warp and each statement of Fig. 21.10 touches 4 (E2). Sectors/Req on global stores is the counter that shows the difference on CUDA.
- How much time it saves depends on how much of the kernel is memory traffic. The grid is read and written once per point per launch while the atom loop runs `CHUNK_SIZE` iterations (E3), so the measured gain should shrink as the chunk grows. That is our prediction for a lab, not a documented fact.

- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (section 10.2.1)
- https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html (Memory Tables)

### A9. Vector stores as the alternative fix (p. 521, Exercise 4)

**Verdict: PARTLY,** for the reason ch06 A11 gives: the `float4` cast is taught on NVIDIA's developer blog, with an alignment rule, and not in the reference guides.

Applied to this kernel (ours): keep the adjacent ownership (thread `t` owns points `4t` to `4t + 3`) and replace the four scalar `+=` with one `float4` load, four adds and one `float4` store. A warp then touches 512 contiguous bytes in one request, 16 sectors, NVIDIA's optimum for 128-bit accesses (A8). The conditions are the blog's: "These instructions require aligned data. ... if you offset the pointer, the offset must also be aligned." Here that means every row of the slice must start on a 16-byte boundary, so `grid.x` must be a multiple of 4 (or the slice allocated with a pitch, ch06 A3), on top of `cudaMalloc`'s 256-byte alignment (infra A6). The blog's cost, "Vectorized loads increase register pressure", adds to coarsening's (A7).

- https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/ (via ch06 A11)

### A10. `charge / sqrtf(r2)` in the inner loop: accuracy and the fast-math switches (Figs. 21.3 to 21.10)

**Verdict: CONFIRMED.** The book counts no square roots at all (notes, "Unclear in the PDF"); NVIDIA documents both operations in the inner loop as correctly rounded by default.

- Programming Guide section 5.6.6, "Built-In Arithmetic Operators": "The built-in C/C++ language operators, such as x + y, x - y, x * y, x / y, x++, x--, and reciprocal 1 / x, for single-, double-, and quad-precision types comply with the IEEE-754 standard. They guarantee a maximum ULP error of zero". With `-prec-div=false` (part of `--use_fast_math`): "x / y → __fdividef(x, y): 2 ULP", "1 / x: 1 ULP".
- Section 5.6.7.3: `sqrtf(x)` "0 ULP", "1 ULP with --use_fast_math".
- Section 5.6.8: `rsqrtf(x)` "2 ULP". Section 5.6.9.2: `__frsqrt_rn(x)` "0 ULP".
- "To maximize the portability of results, users are recommended to use the default settings of the nvcc compiler's floating-point options: -ftz=false, -prec-div=true, and -prec-sqrt=true, and not use the --use_fast_math option." (also quoted in ch10 A11).

For the lesson and any lab (ours): `charge * rsqrtf(r2)` replaces a division and a square root with one documented 2-ULP function; it is not in the book and its results differ from a CPU reference by a few ULP per term, so tests need a tolerance. The HIP and Metal counterparts are B2 and C3.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html (sections 5.6.6, 5.6.7.3, 5.6.8, 5.6.9.2, 5.6.9.3)

---

## Group B: HIP / ROCm

### B1. The chapter's kernels on HIP: what carries over and what changes

**Verdict: PARTLY.** Every construct in Figs. 21.5 to 21.10 has the CUDA spelling under HIP (`__constant__`, `hipMemcpyToSymbol`, `atomicAdd(float*, float)`, `dim3`, `blockIdx`/`threadIdx` in x and y; ch05 B4, infra B1, B5). Three documented differences change what the chapter's argument predicts:

- Scatter (Fig. 21.5): float `atomicAdd` is compiled as a safe operation, possibly a CAS loop, unless built with `-munsafe-fp-atomics`, and hardware float atomic add is listed as absent on RDNA1 and CDNA1 (infra B1). So the scatter kernel's cost on AMD depends on a compiler flag and the GPU generation.
- Constant memory: AMD documents no size and its `totalConstMem` reports the largest allocation (ch07 B1), so the 64 KB chunk bound is NVIDIA's. Keeping chunks at 64 KB or less keeps one source portable (ours). The constant cache is the scalar cache, and AMD says "the access has to be uniform within a warp, otherwise the accesses are serialized" (infra B5). The gather kernels' atom reads are uniform across the wavefront, the case AMD's cache is built for; the scatter kernel's per-thread atom read is not (A6).
- Group width: the coalescing argument of p. 521 is written for 32 lanes; on CDNA a wavefront is 64 (ch05 B5). Fig. 21.10's stride is `blockDim.x`, not the warp width, so it stays correct; each write statement is then 256 contiguous bytes per wavefront (E2).

Nothing here was compiled for ROCm.

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (via infra B1, B5)

### B2. `sqrtf`, `rsqrtf` and division on HIP

**Verdict: PARTLY.** The functions exist with CUDA's names. AMD's page reports measured differences from the C++ standard library over a test range, not error bounds (as in ch10 B4).

- HIP math API (HIP 7.17.0), single precision: `float sqrtf(float x)` "Returns the square root of x", tested on x in [0, 100], difference 1. `float rsqrtf(float x)` "Returns the reciprocal of the square root of x", [0.01, 100], 1. `float fdividef(float x, float y)` "Divide two floating point values", x and y in [-100, 100], 0. Intrinsics: `float __fsqrt_rn(float x)` "Returns √x in round-to-nearest-even mode", 1; `float __frsqrt_rn(float x)` "Returns 1 / √x in round-to-nearest-even mode", 1.
- The compiler switch for division and square root is Clang's, not an `nvcc`-style `-prec-div`: "-fhip-fp32-correctly-rounded-divide-sqrt, -fno-hip-fp32-correctly-rounded-divide-sqrt: Specify that single precision floating-point divide and sqrt used in the program source are correctly rounded (HIP device compilation only)". In LLVM `main`, `clang/include/clang/Options/Options.td` line 1536 defines it with `DefaultTrue`. ROCm ships its own Clang build; we read LLVM `main`, not the ROCm 10.2 copy.

- https://rocm.docs.amd.com/projects/HIP/en/latest/reference/math_api.html
- https://clang.llvm.org/docs/ClangCommandLineReference.html
- https://github.com/llvm/llvm-project/blob/main/clang/include/clang/Options/Options.td

### B3. Block size, row padding and a 64-wide wavefront (Fig. 21.9 on AMD)

**Verdict: CONFIRMED.** AMD gives the same block-size range as Fig. 21.9 and A4, and recommends exactly the padded layout the book's kernels assume (they have no bounds checks; notes).

- HIP performance guidelines, "Optimize block size": "AMD Instinct GPUs execute threads in warps of 64, while AMD Radeon GPUs execute threads in warps of 32. Choosing block sizes as multiples of 64 or 32 prevents partial warps that waste execution slots. Larger blocks (128-256 threads) typically achieve better occupancy and resource utilization." Example lines: `dim3 block(64); // Good for AMD Instinct GPUs (warp=64)` and `dim3 block(32); // May waste resources on Instinct GPUs`.
- Same page, "Optimize 2D array access": "Padding 2D arrays to multiples of the warp size ensures each row starts at an aligned memory boundary. This allows consecutive threads accessing the same row to generate coalesced memory transactions, thereby maximizing bandwidth." with `int width = ((actual_width + warpSize - 1) / warpSize) * warpSize;`.
- "Memory hardware loads data in aligned chunks (typically 128 bytes)."

For Fig. 21.10 on CDNA (ours): `blockDim.x` should be at least 64 so that each write statement of a wavefront is one contiguous run; the 16-wide "half-warp" stride of Fig. 21.9 would leave each run 64 bytes long, half a 128-byte chunk.

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html (sections "Optimize block size", "Ensure proper alignment", "Optimize 2D array access")

---

## Group C: Metal

### C1. The atom chunk on Metal: a constant buffer, with no symbol copy and no documented broadcast cache

**Verdict: PARTLY.** The container exists: an `MTLBuffer` the host fills and the kernel takes as `constant float4* atoms [[buffer(n)]]` (infra C6, ch07 C1). What the chapter's speed argument rests on (a cache that broadcasts one address to all lanes) is not documented by Apple (ch07 C2), and no size limit for constant-address-space buffers is documented (ch07 C1).

- Apple's own guidance, quoted in ch07 C2 (WWDC16): the `constant` address space is "optimized for cases with a lot of data reuse", and "if you have a potentially variable amount of data, this is usually a sign that you need to be using the device address space." A fixed `CHUNK_SIZE` fits the first; the binned kernel of §21.5, whose atoms the book moves to global memory (pp. 526 to 527), fits the second (`const device float4*`).
- `setBytes` takes under 4 KB (ch03 C5), so a chunk passed that way holds at most 255 atoms (E3).
- Not in the book, new for this file: MSL has a type that states a value is the same for every thread of a dispatch. Section 2.14: "All OS: Metal 2 and later support uniform types." "To indicate that this variable is uniform for all threads executing the graphics or kernel function of a draw or dispatch call, MSL adds a new template class called uniform (available in the header metal_uniform)". "If a variable is of the uniform type, and the variable does not have the same value for all threads executing the kernel or graphics function, then the behavior is undefined." The spec's motivation is texture indexing and says the same argument applies "to index into one or more buffers"; it makes no performance statement for buffer reads. The atom loop counter is dispatch-uniform, so `uniform<int>` is legal there; whether it changes anything is a measurement.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 2.14 and 4.2, pages 57 to 58 and 119)
- https://developer.apple.com/videos/play/wwdc2016/606/ (via ch07 C2)

### C2. The scatter kernel on Metal: the output buffer changes type

**Verdict: CONFIRMED** (from infra C1). A float atomic add exists, but MSL applies atomic functions only to atomic types, so Fig. 21.5's grid must be declared `device atomic_float* energygrid` and updated with `atomic_fetch_add_explicit(&energygrid[k], v, memory_order_relaxed)`; the gather kernels keep `device float*`. Float atomics need Metal 3 and GPU family Apple7 (M1) or later, add and sub only. Apple documents neither where atomics execute nor their cost (ch09 C4), so on Metal the scatter-versus-gather gap is only measurable, not predictable from documentation.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 2.6 and 6.16, via infra C1)

### C3. `charge / sqrt(r2)` on Metal: fast math by default

**Verdict: CONFIRMED.** Under the spec's precise tables both operations are correctly rounded; under fast math, which the spec calls the default for its command-line compiler, division is looser and `sqrt` is computed through `rsqrt`.

- Section 8.4, Table 8.1 ("Accuracy of single-precision floating-point operations and functions"): `x / y` "Correctly rounded", `1.0 / x` "Correctly rounded", `rsqrt` "Correctly rounded", `sqrt` "Correctly rounded".
- "Table 8.2 describes the minimum accuracy of single-precision floating-point arithmetic operations given as ULP values with fast math enabled (which is the default unless you specify -fno-fast-math as a compiler option)." Table 8.2: `x / y` "<= 2.5 ulp for y in the domain of 2-126 to 2126" (that is 2^-126 to 2^126), `1.0 / x` "<= 1 ulp", `rsqrt` "<= 2 ulp", `sqrt(x)` "Implemented as x * rsqrt(x) with special cases handled correctly."
- Section 6.6: "the metal::precise and metal::fast nested namespaces provide an explicit way to select the fast or precise variant of these math functions for single precision floating-point." The library also has `T divide(T x, T y)` "Compute x / y."
- Which mode a runtime-compiled library gets by default is documented three different ways (ch05 C7). A Metal lab of this chapter compares against a CPU reference with a tolerance either way.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 6.6 and 8.4, Tables 6.4, 8.1 and 8.2, pages 205 to 208 and 368 to 372)

### C4. Coalescing-aware assignment, vector stores and padding on Metal (Figs. 21.8 to 21.10, Exercise 4)

**Verdict: COULD NOT VERIFY; no coalescing rule is documented.** The MSL spec has no coalescing, bank or cache-line text, and Apple's advice is qualitative cache locality (ch06 C1). So Fig. 21.10 against Fig. 21.8 on a Mac is a measurement with a prediction, not a documented outcome. What is documented:

- The SIMD-group width is read from the pipeline (`threadExecutionWidth`, `[[threads_per_simdgroup]]`, ch05 C3), and Fig. 21.10's `blockDim.x` stride is the threadgroup's x size, so the mapping carries over unchanged (ours).
- Vector stores: declare the grid `device float4*` (16-byte aligned) or `device packed_float4*`; the bind offset must be aligned to the element type (infra C7, ch06 C3).
- Padding: with `dispatchThreads` Metal creates smaller edge threadgroups instead of out-of-range threads (ch05 C5), which removes the "Padding waste" threads of Fig. 21.9; a coarsened thread that owns several points still needs a bound check for each point it writes (ours).

---

## Group D: the same choices in real backends (bridge, goal 3)

The book's application is not an LLM workload. These items are the bridge the lesson labels as ours: the same scatter-versus-gather and binning decisions in the embedding gradient and in mixture-of-experts dispatch.

### D1. PyTorch CUDA embedding backward: sort-then-gather by default, atomics only when they win and determinism is not requested

**Verdict: CONFIRMED.** `aten/src/ATen/native/cuda/Embedding.cu` (last commit 2026-11-07) and `EmbeddingBackwardKernel.cu` (2026-05-22) contain three strategies, chosen at run time.

- Few indices (`if (num_indices <= 3072 && !scale_grad_by_freq)`, line 282): `embedding_backward_feature_kernel`, where each block owns a range of feature columns and walks all indices; duplicates within a chunk are merged by a leader warp. Comment (line 95): "To ensure determinism, we can't just have each warp add its grad data to its dst_row. We need to check if any other warps pulled grad data targeting dst_row. If so, we elect the first warp in each matching group as the leader." (`BLOCKDIMY` is 16 under `USE_ROCM`, 32 otherwise, so the same file builds for ROCm.)
- Otherwise the indices are sorted (`cuda::cub::radix_sort_pairs`, line 323), which bins the rows by token id, and `EmbeddingBackwardKernel.cu` sums each bin without atomics: "This code computes the sum of the weights in two-steps: 1) Each GPU warp sums `NROWS_PER_THREAD` number of row given by `indices` 2) Each partial-sum from 1) are summed and scatter into `grad_weight`" with `constexpr int NROWS_PER_THREAD = 10;`. Both `compute_grad_weight` and `sum_and_scatter` give each thread one (segment, feature) and write once: the gather of Fig. 21.6, after binning.
- A fused atomic path is used only when it helps and is allowed (lines 367 to 379): "Heuristic: Use fused kernel when sum_and_scatter would have poor parallelism ... Note: Fused kernel uses atomics which are non-deterministic, so we skip it when deterministic algorithms are requested", with `!at::globalContext().deterministicAlgorithms()` in the condition and `constexpr int32_t MAX_ATOMIC_ACCUM_BLOCKS_PER_SM = 4;`.
- PyTorch 2.15, `torch.use_deterministic_algorithms`: "Sets whether PyTorch operations must use “deterministic” algorithms. That is, algorithms which, given the same input, and when run on the same software and hardware, always produce the same output." Among the "normally-nondeterministic operations [that] will act deterministically when mode=True": "torch.Tensor.scatter_add_() when called on a CUDA tensor", "torch.index_add() when called on CUDA tensor", "torch.nn.Embedding when attempting to differentiate a CUDA tensor". "Note that deterministic operations tend to have worse performance than nondeterministic operations."

- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Embedding.cu
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/EmbeddingBackwardKernel.cu
- https://docs.pytorch.org/docs/2.15/generated/torch.use_deterministic_algorithms.html

### D2. PyTorch MPS embedding backward: scatter with float atomics, and an error in deterministic mode

**Verdict: CONFIRMED.** The Metal kernel is Fig. 21.5's pattern: one thread per (index, feature) and an atomic add into the weight gradient.

- `aten/src/ATen/native/mps/kernels/Embedding.metal` (2026-10-13), `embedding_dense_backward`: parameters include `device AtomicType_t<float>* grad_weight [[buffer(3)]]` and `uint tid [[thread_position_in_grid]]`; body `uint index_pos = tid / feature_size; uint feature_idx = tid % feature_size;` ... `AtomicType<float>::atomic_add(grad_weight, static_cast<long>(weight_idx) * feature_size + feature_idx, grad_val);`.
- `aten/src/ATen/native/mps/operations/Embedding.mm` (2026-11-10), lines 37 to 38: `// Nondeterministic due to atomic_add` and `at::globalContext().alertNotDeterministic("embedding_dense_backward_mps");`.
- The PyTorch 2.15 page of D1 lists "torch.nn.Embedding when attempting to differentiate an MPS tensor" among the operations that "will throw a RuntimeError when mode=True".

So one framework makes opposite choices on two backends: binned gather on CUDA and ROCm, atomic scatter on MPS. The reason for the MPS choice is not stated in the source.

- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/kernels/Embedding.metal
- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/mps/operations/Embedding.mm

### D3. llama.cpp's embedding gradient is the chapter's direct-summation gather

**Verdict: CONFIRMED.** `ggml_get_rows_back` is the gradient of `ggml_get_rows` (the embedding lookup) in ggml's backward pass (`ggml/src/ggml.c`, `case GGML_OP_GET_ROWS:` ... `ggml_get_rows_back(ctx, grad, src1, src0)`), used for training, not inference. Its CUDA kernel, `ggml/src/ggml-cuda/getrows.cu` (2026-11-11), lines 133 to 157:

```cpp
const int col = blockIdx.x*blockDim.x + threadIdx.x;
...
for (int64_t dst_row = blockIdx.y; dst_row < nrows_dst; dst_row += gridDim.y) {
    float sum = 0.0f;
    for (int64_t i = 0; i < nrows_grad; ++i) {
        if (rows[i] != dst_row) {
            continue;
        }
        sum += grad[i*ncols + col];
    }
    dst[dst_row*ncols + col] = sum;
}
```

Each thread owns one output element, scans every input row, keeps the ones that match, and writes once: Fig. 21.6 with an equality test in place of the cutoff test, and the all-pairs cost (vocabulary rows times tokens) that §21.5 sets out to remove (our reading). The HIP build compiles the same file (ch05 B9). The Metal backend files read (`ggml-metal-device.m`, `ggml-metal-ops.cpp`, `kernels/misc.metal`) contain no `GET_ROWS_BACK`, so we found no Metal implementation.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/getrows.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml.c
- https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-metal

### D4. llama.cpp MoE dispatch: bin tokens by expert, fixed-size tiles, a padding slot, and a gather to combine

**Verdict: CONFIRMED.** Routing tokens to experts is §21.5's binning with experts as bins, and the final combine is a gather.

- CUDA and HIP, `ggml/src/ggml-cuda/mmid.cu` (2026-11-19), lines 71 to 76: "Helper function for mul_mat_id, converts ids to a more convenient format. ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert. ids_dst describes the same mapping but for the dst tensor. The first (n_tokens*n_expert_used + J - 1) / tile_width elements of tile_spec contain the specication for matrix multiplication tiles". In `mm_ids_helper`, `const int expert = blockIdx.x; // The expert this CUDA block handles.`: one block per bin, each scanning all tokens, with no atomics.
- Metal, `ggml/src/ggml-metal/kernels/mul_mm.metal` (2026-11-05), `kernel_mul_mm_id_map0`, one thread per expert (`const short ide = tpitg; // expert id`). Line 385: "pass 1: count the entries of this expert". Lines 421 to 423: "exclusive prefix sum over the counts: the entries of this expert start at bas[ide] - every slice is padded by one entry for the uniform dummy store of pass 2 - and emit the entry tile descriptors of kernel_mul_mm_id: (expert, first entry, valid rows) per tile", with `constexpr uint32_t NTILE = 32;`. Lines 486 to 488: "one store per token for every thread, regardless of the match - conditional stores of the divergent lanes are very slow - the dummy value of the tokens without a match lands on the padding slot of this slice and is never read". That is count, prefix sum, fixed-size tiles with a valid-row count, and a dummy slot that keeps every lane's store unconditional: §21.5's fixed-size bins and dummy atoms, and §21.5's divergence concern, in a production kernel. The claim that divergent conditional stores are slow is the llama.cpp authors', not Apple's.
- Combine, `ggml/src/ggml-cuda/moe-weighted-reduction.cu` (2026-09-01), `moe_weighted_reduction_f32`: one thread per (token, column), `for (int expert = 1; expert < n_expert_used; ++expert)` accumulates `experts[row * n_embd + col] * scale * weights[row]` in a register, then `dst[token * n_embd + col] = sum;`. Gather, no atomics.

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmid.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/mul_mm.metal
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/moe-weighted-reduction.cu

### D5. MLX and mlx-lm: sort tokens by expert when there are many, then a gathered matmul that walks them in order

**Verdict: CONFIRMED.**

- mlx-lm `mlx_lm/layers/moe.py` (2026-11-19): `_gather_sort` does `order = mx.argsort(indices)` and `inv_order = mx.argsort(order)`; `SwitchGLU.__call__`: "When we have many tokens, then sort them to make sure that the access of different experts is in order." with `do_sort = indices.size >= 64`, then each projection is called with `sorted_indices=do_sort`, and `_scatter_unsort` restores the order (`x = x[inv_order]`).
- MLX 0.32.4 docs, `mlx.core.gather_mm(a, b, /, lhs_indices=None, rhs_indices=None, *, sorted_indices=False, stream=None)`: "Matrix multiplication with matrix-level gather." "If only one index is passed and it is sorted, the sorted_indices flag can be passed for a possible faster implementation."
- MLX `mlx/backend/metal/matmul.cpp` (2026-10-09), `GatherMM::eval_gpu`, line 2961: "We are walking a in order and b is also in order so we can batch up the matmuls and reuse reading a and b." That branch (`M == 1 && right_sorted_ == true`) dispatches `gather_mm_rhs`.

The threshold of 64 indices is mlx-lm's own heuristic; no measurement behind it is given in the file.

- https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/layers/moe.py
- https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.gather_mm.html
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/matmul.cpp

### D6. vLLM `moe_align_block_size`: every expert's bin padded with dummy tokens (not in the brief's list; added because it is §21.5's dummy atoms almost word for word)

**Verdict: CONFIRMED.** `vllm/model_executor/layers/fused_moe/moe_align_block_size.py` (2026-09-20), docstring: "Aligns the token distribution across experts to be compatible with block size for matrix multiplication." "This function pads the number of tokens that each expert needs to process so that it is divisible by block_size. Padding ensures that during block matrix multiplication, the dimensions align correctly." Its example: "Tokens 12 are non-existent (padding) and are ignored in the subsequent matrix multiplication." The worst-case buffer is `max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)`, the same memory cost of padding the book lists first (p. 527).

A slip in the docstring's example (ours): it uses expert ids 1 to 4 with `num_experts = 4`. Read it for the padding idea only.

- https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/moe_align_block_size.py

---

## Group E: the chapter's arithmetic, recomputed (ours, not vendor facts)

### E1. Operation counts per atom (pp. 519 to 521, Exercise 2)

**Verdict: PARTLY.** Direct counts of the listings, as the notes already found; the book's own numbers do not add up in three places.

| Per atom | Fig. 21.6, four threads (counted) | Book p. 521 | Fig. 21.8, one thread, four points (counted) | Book p. 521 |
|---|---|---|---|---|
| Constant-memory loads | 16 | 16 | 4 | 4 |
| Subtractions | 12 | 12 | 3 | 3 |
| Additions | 12 | 12 | 12 (3 for `dx1..dx3`, 1 for `dysqdzsq`, 4 inside the roots, 4 accumulates) | 11 |
| Multiplications | 12 | 12 | 8 (6 if `2*gridspacing` and `3*gridspacing` are hoisted) | 6 |
| Divisions | 4 | 12 | 4 | 4 |
| Square roots | 4 | not counted | 4 | not counted |
| Arithmetic without roots | 40 | 48 | 27 (25 hoisted) | 24 |

Fig. 21.6 per point is 10 arithmetic operations plus a root against the book's "9" on p. 519. The lesson can show the book's totals as printed and the counted ones beside them; the explorer computes from the listing, not from the book's table. With A10 in mind, the four roots and four divisions that coarsening does not remove are the expensive part of what remains (our reading; no vendor gives per-instruction costs in the pages read).

### E2. Sectors per write statement (pp. 521 to 522), and what the printed Fig. 21.8 actually computes

**Verdict: CONFIRMED** that the book's coalescing claim holds under NVIDIA's documented 32-byte-sector rule (A8). Warp of 32 lanes, 4-byte floats, aligned row, one write statement `m` (of four):

| Assignment | Address of lane `t` (elements) | Span per statement | Sectors per statement | Sectors for the four statements |
|---|---|---|---|---|
| Fig. 21.8 as the text describes it (thread owns `4t..4t+3`) | `base + 4t + m` | 512 bytes | 16 (each holds 2 useful floats of 8) | 64 |
| Fig. 21.10 (`blockDim.x` stride) | `base + t + m*blockDim.x` | 128 bytes | 4 | 16 |
| Exercise 4, one `float4` per thread | `base + 4t` (16 bytes each) | 512 bytes | 16 in one request (all bytes used) | 16 |

On a 64-wide AMD wavefront the Fig. 21.10 statement covers 256 contiguous bytes (two 128-byte lines) and the Fig. 21.8 statement 1,024 bytes (B3). The `+=` makes each write a read as well, with the same pattern. Whether the L2 merges Fig. 21.8's four partial-sector statements is not documented: measure.

The printed Fig. 21.8 line 05 (`i = blockIdx.x * blockDim.x*COARSEN_FACTOR + threadIdx.x`, writes to `i..i+3`; notes, "Unclear in the PDF"): with `B = blockDim.x` and `base = blockIdx.x*4B`, lane `t` writes `base + t` to `base + t + 3`. The block therefore writes points `base` to `base + B + 2`, most of them four times with a non-atomic `+=`, and never writes points `base + B + 3` to `base + 4B - 1`. Each thread's distances are right for the points it writes; the output is wrong because of the ownership. This is a ready-made bug for the explorer: the conflicts view shows four writers per point and an unwritten tail.

### E3. Constant-memory capacity, chunk sizes and arithmetic intensity (pp. 517 to 519)

**Verdict: CONFIRMED** (arithmetic on documented limits). One atom is four floats, 16 bytes.

| Route | Documented limit | Atoms per chunk |
|---|---|---|
| `__constant__` array (CUDA) | 64 KB (ch05 A4) | 4,096, less other constant data |
| Constant cache working set per SM (CUDA) | 8 KB (A5) | 512 |
| Chunk passed by value as a kernel argument (CUDA 12.1 and later) | 32,764 bytes (ch07 A4) | 2,047 by size alone; fewer once the kernel's other arguments and alignment are counted |
| Metal `setBytes` | "smaller than 4 kilobytes" (ch03 C5) | 255 |
| Metal constant buffer, HIP `__constant__` | not documented (C1, B1) | choose 4,096 or fewer for one portable source |

Intensity of the gather kernel per launch (counted operations, E1): each grid point does `CHUNK_SIZE x 10` arithmetic operations plus `CHUNK_SIZE` roots against 8 bytes of global traffic (one float read and one written by the `+=`). For 4,096 atoms that is about 5,000 operations per byte; even 64 atoms give 80. That supports the book's "global-memory bandwidth is not a limiting factor for this kernel" (p. 519) and is why A8's coalescing gain should shrink as `CHUNK_SIZE` grows.

### E4. Neighbourhood sizes (pp. 525 to 526, Figs. 21.13 and 21.14)

**Verdict: PARTLY.** The book's worked numbers are consistent; the "9 fully covered + 12 partially covered = 21" of Fig. 21.14 is exact only at one radius, and Fig. 21.13 is a sketch, not to scale.

Method: bins are cubes (squares in 2D) of side `b`, centred on integer offsets from the centre bin; a bin is in the neighbourhood when the nearest point of the bin lies inside the super circle (or sphere) of radius `R` centred on the centre bin's centre, as the book defines it (p. 526).

- Fig. 21.14 (2D): 21 bins are touched for any `R` between about `1.58 b` and `2.12 b`. In that whole range only 5 bins (the centre and its four edge neighbours) are fully inside. The 3 x 3 block is fully inside only when `R >= 1.5·√2·b ≈ 2.12 b`, which is also the radius at which the four corner bins start to touch the circle. So "9 fully + 12 partially" holds at exactly `R = 2.12 b` and nowhere else. The total of 21 is the useful number; the explorer should compute counts from `R` rather than hard-code 9 and 12.
- The book's worked case (p. 525): grid spacing 0.5 Å x 8 = 4 Å blocks used as bins, cutoff 12 Å. With the text's radius (cutoff plus half the bin diagonal, 12 + 2√3 ≈ 15.46 Å = 3.87 bins) the neighbourhood is **461 bins** in 3D, 147 of them fully inside. With Fig. 21.13's label (cutoff plus the full diagonal, 18.93 Å) it would be 667. In 2D the same numbers give 61 bins (rows 3, 7, 7, 9, 9, 9, 7, 7, 3) with the half diagonal and 77 with the full one; neither matches the rows read from Fig. 21.13 (about 69, notes), which is drawn schematically.
- Pairs considered: per grid point the binned kernel scans the atoms in 461 x 64 Å³ = 29,504 Å³, while the cutoff sphere is 7,238 Å³. At uniform density about 25 % of the atoms a thread tests pass the cutoff test (in 2D, 46 %). That is the divergence of Exercise 5 in numbers: most tests fail. Against direct summation over a cube of side `L` Å, the fraction of atoms scanned is `29,504 / L³` (3 % at `L` = 100 Å), independent of how many atoms there are per unit volume.

The explorer can reproduce all of these from its own model; none of them is a device figure.

---

## Book items with no vendor fact to check

Recorded so the wrap page's errata list does not wait on this file (all from `notes/ch21.md`, "Unclear in the PDF", plus E1, E2 and E4):

- p. 513: "gather-to-scatter transformation" where the chapter goes from scatter to gather; "fast boundary condition checking" is never developed.
- Fig. 21.8 line 05: unscaled `threadIdx.x` with ownership `i..i+3` gives overlapping writers and an unwritten tail (E2).
- p. 521: "two problems" cause the uncoalesced writes; only one is described.
- pp. 519 to 521: operation counts do not add up (9 versus 12 per point; 12 divisions for four points; 11 additions and 6 multiplications versus a counted 12 and 8; z omitted from the eliminated loads; no square roots counted) (E1).
- Fig. 21.9: "up to 8 potentials, skipping by half-warps" against a factor of 4 and a `blockDim.x` stride in the code; the half-warp is the 2007 coalescing unit (A3).
- Fig. 21.13 labels the radius "cutoff distance + length of bin diagonal"; the text and Fig. 21.14 say half the diagonal (E4).
- Fig. 21.14: "9 fully covered" is exact only at one radius (E4).
- Slice indexing differs between Figs. 21.3, 21.4 and 21.6; no kernel has bounds checks; Fig. 21.5 has no atom count.
- p. 519: `atom[]` and `atoms[]` both appear; energy as "an important limiting factor" for large systems is the book's view, not checked.
- p. 527: the "less than 3 %" overflow share and the host-side overflow pass are the book's design; no current document bears on them.

## Answers to the notes' portability questions

- `__constant__` and the copy API on HIP and Metal (notes mark unverified): HIP keeps both spellings (infra B5) but documents no size and serves `__constant__` from a scalar cache that helps only uniform reads (ch07 B1, B1 above). Metal has no symbol copy; the host fills a buffer taken as `constant float4*` (C1). The chapter does not name the copy API; it is `cudaMemcpyToSymbol` (infra A5).
- Float `atomicAdd` on HIP and Metal: HIP has it, compiled safe unless `-munsafe-fp-atomics` (infra B1); Metal has `atomic_float` add on Apple7 and later and needs the buffer typed atomic (C2).
- 2D launch and thread-position built-ins: same in HIP (ch05 B4); `[[thread_position_in_grid]]` as `uint2` in Metal (ch05 C4).
- Warps of 32 and CUDA's coalescing: wavefronts of 64 on CDNA (B3, E2); no documented rule on Metal (C4). Fig. 21.10's stride is `blockDim.x`, so it needs no change; re-measure on each backend.
- "Half-warps": an NVIDIA term whose coalescing meaning dates from 2007 (A3).
- Registers and occupancy: A7; AMD ch04 B10; Metal ch06 C5.
- Vector stores: A9 (CUDA), infra B6 and ch06 B5 (HIP), C4 (Metal).
- Shared memory and a barrier for the cooperative bin load: ch05 A7, B4, C1, C2.
- Host work while the device runs: launches and commits are asynchronous on all three (ch02 A5, B3, C4); streams and command queues in infra A8, B8, C12.

---

## Summary of corrections

Every PARTLY or COULD NOT VERIFY item, and every sub-claim not verified, one line each:

- A3: "half-warps" in Fig. 21.9 is the coalescing unit of the 2007 CUDA 1.0 guide; today NVIDIA states coalescing per warp in 32-byte sectors. Do not teach a half-warp rule; the figure's source is not given (our inference: the 2007 to 2008 work of references [2] and [3]).
- A6: float `atomicAdd` exists (NVIDIA now calls it legacy), but "significantly reduces" speed is not quantified by any vendor; scatter with float atomics is also not run-to-run reproducible, which the book does not say.
- A7: the book's "register count stays within the allowed limit" cannot be checked (no count, compiler or device); the reader checks with `--resource-usage`. Limits today: 255 registers per thread, 64 K per SM.
- A9: the vector-store fix relies on the `float4` cast taught on NVIDIA's blog, not in the guides; for this kernel it needs rows aligned to 16 bytes (`grid.x` a multiple of 4 or a pitched allocation).
- B1: on AMD the scatter kernel's float atomics are CAS-style unless `-munsafe-fp-atomics`, there is no documented 64 KB constant size, and the constant cache helps only wavefront-uniform reads (the gather kernels qualify; the scatter kernel's atom read does not).
- B2: AMD reports `sqrtf` and `rsqrtf` as measured differences from the C++ library (1 ULP on a test range), not bounds; the division and square-root rounding switch is Clang's `-fhip-fp32-correctly-rounded-divide-sqrt`, on by default in LLVM `main` (ROCm's own Clang copy not checked).
- C1: Metal has a constant buffer for the atom chunk but documents no broadcast constant cache and no size limit; `setBytes` caps a by-value chunk at 255 atoms; MSL's `uniform<T>` declares dispatch-uniform values but the spec makes no performance claim for buffer reads.
- C4 (could not verify): no coalescing rule is documented for Apple GPUs; Fig. 21.10 against Fig. 21.8 on Metal is a measurement.
- E1: the book's operation counts do not match the listings (see table); square roots are never counted.
- E4: Fig. 21.14's "9 fully + 12 partially covered" holds only at `R = 2.12 b`; Fig. 21.13 is schematic and its radius label (full diagonal) contradicts the text (half diagonal).
- D5 (sub-claim): mlx-lm's threshold of 64 indices for sorting is unexplained in its source.
- D2 (sub-claim): why PyTorch's MPS backend keeps the atomic scatter is not stated in the source.

Status facts worth knowing even where the verdict is CONFIRMED:

- A1: VMD 2.0.0 was released 2026-03-25 (Linux build with CUDA; Apple Silicon builds listed for its beta and alpha); VMD's own GPU page names multilevel summation as the linear-cost method for the far field the book leaves undescribed.
- A4, B3: NVIDIA (64 minimum, 128 to 256 to start) and AMD (multiples of 64 or 32, 128 to 256) both agree with Fig. 21.9's 64 to 256 threads; AMD also recommends padding 2D rows to a multiple of the warp size, the layout the book's unchecked kernels assume.
- A10, C3: CUDA's `/` and `sqrtf` are correctly rounded by default; Metal's fast math (the spec's default for its compiler) gives `x / y` within 2.5 ulp and computes `sqrt` as `x * rsqrt(x)`. Lab tolerances must allow for it.
- D1 to D6: real backends make the chapter's choice explicitly: PyTorch CUDA bins (sorts) and gathers unless atomics are allowed and faster, PyTorch MPS scatters with atomics, llama.cpp's embedding gradient is a direct-summation gather, llama.cpp, MLX and vLLM bin MoE tokens by expert, and llama.cpp's Metal map kernel and vLLM pad bins with dummy entries.

## Where the backends really differ

One line per cell. Nothing here was run on any device.

| Feature | CUDA | HIP (ROCm 10.2) | Metal (MSL 4.1) |
|---|---|---|---|
| Atom chunk shared by all threads | `__constant__ float atoms[CHUNK_SIZE*4]` + `cudaMemcpyToSymbol`; 64 KB | Same, with `hipMemcpyToSymbol`; no documented size | Host-filled buffer taken as `constant float4*`; no symbol copy; no documented size |
| Why the chunk read is cheap | Constant cache, one access per distinct address in a warp; 8 KB working set per SM | Scalar cache, only for wavefront-uniform reads | Nothing documented; Apple talks recommend `constant` for reused read-only data |
| Chunk as a launch argument | By value, up to 32,764 bytes (CUDA 12.1 and later) | By value; no documented limit | `setBytes`, under 4 KB (255 atoms) |
| Scatter update (Fig. 21.5) | `atomicAdd(float*, float)` (legacy; `cuda::atomic_ref` recommended) | Same call; safe CAS-style unless `-munsafe-fp-atomics`; no hardware float add on RDNA1/CDNA1 | Buffer typed `device atomic_float*`; `atomic_fetch_add_explicit(..., memory_order_relaxed)`; Apple7 and later |
| Where the atomics run | L2 ("passed through to the L2 cache") | L2 for global memory | Not documented |
| Gather update (Figs. 21.6 to 21.10) | Plain `+=` into a thread-owned point | Same | Same, buffer typed `device float*` |
| Group width behind the coalescing argument | 32 | 64 (Instinct) or 32 (Radeon) | `threadExecutionWidth` from the pipeline |
| Coalescing unit | 32-byte sectors per warp | 64- or 128-byte cache lines per wavefront | Not documented |
| Seeing Fig. 21.8's uncoalesced stores | Nsight Compute Sectors/Req (optimal 4 for 32-bit) | ROCm profilers (metric name not checked) | Xcode limiter counters, indirect |
| Block size advice | Multiple of 32; at least 64; start at 128 to 256 | Multiples of 64 or 32; 128 to 256 | Multiple of `threadExecutionWidth` |
| Padded rows for unchecked kernels | `cudaMallocPitch` | Pad width to a warp-size multiple (AMD guidance); `hipMallocPitch` | `dispatchThreads` gives smaller edge threadgroups; no padding API |
| One 16-byte store per thread (Exercise 4) | `float4` cast (NVIDIA blog), 16-byte aligned | `float4`, 16-byte aligned; cast undocumented | `device float4*` (16-byte aligned) or `packed_float4*` |
| Register cost of coarsening | `--resource-usage`; at most 255 per thread | `--resource-usage` (VGPRs) | Not exposed; Xcode occupancy only |
| `charge / sqrt(r2)` by default | `/` and `sqrtf` correctly rounded | Measured within 1 ULP of the C++ library; Clang rounds div and sqrt correctly by default (LLVM `main`) | Fast math: `x / y` within 2.5 ulp, `sqrt` as `x * rsqrt(x)`; precise variants correctly rounded |
| Faster reciprocal square root | `rsqrtf` (2 ULP); `--use_fast_math` | `rsqrtf`; `-fno-hip-fp32-correctly-rounded-divide-sqrt` | `rsqrt` (fast: within 2 ulp); `metal::fast::` namespace |
| Cooperative bin load (§21.5) | `__shared__` + `__syncthreads()` | Same | `threadgroup` + `threadgroup_barrier(mem_flags::mem_threadgroup)`; 32 KB |
| Host overflow pass while the device runs | Asynchronous launch; streams | Same | Asynchronous `commit`; completion handler; shared buffer readable without a copy |
| Embedding gradient in a real library | PyTorch: sort and gather, fused atomics only if allowed; llama.cpp: direct-summation gather | Same sources (`USE_ROCM` branches; llama.cpp via `vendors/hip.h`) | PyTorch MPS: atomic scatter, error in deterministic mode; llama.cpp: none found |
| MoE token binning | llama.cpp `mm_ids_helper` (one block per expert); vLLM pads each expert to `block_size` | llama.cpp same source | llama.cpp `kernel_mul_mm_id_map0` (count, prefix sum, 32-row tiles, padding slot); MLX sorts at 64 or more indices |
