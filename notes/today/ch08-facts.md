# Chapter 8 fact check: CUDA, HIP, Metal

Checked on 2026-11-20 against the live sources listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY. Where an item has several sub-claims the verdict is for the item as a whole and the text says which sub-claim failed. A verdict judges the claim in the item title (usually the book's, with its page from `notes/ch08.md`), not the vendor.
- **Quote** is copied from the source (HTML docs flattened to text, GitHub raw files at a tag or on the default branch, PDF text, local SDK headers). Line breaks inside quotes are removed.
- "Book" means `notes/ch08.md` and its page numbers. Nothing here is taken from memory of the book.
- Versions the sources reported about themselves on the check date: CUDA Programming Guide "v13.5" (pages last updated Oct 31, 2026), CUDA C++ Best Practices Guide "v13.5" and Runtime API reference (last updated Nov 02, 2026); ROCm Core SDK 10.2.0 ("HIP 7.17.0 Documentation"), HIP runtime source at tag `therock-10.2` of `ROCm/rocm-systems`; Metal Shading Language Specification "Version 4.1" dated 2026-06-04 ("MSL spec"); Metal Feature Set Tables dated October 23, 2026; macOS 27.0 SDK headers on this machine. Project files read from the default branches on the check date (llama.cpp `master`, MLX `main`, PyTorch `main`); last-commit dates are from the GitHub commits API.
- Not checked: anything by running code on a GPU. One thing was compiled on this machine and is labelled where it appears: a Metal translation of the tiled and register-tiled kernels (C2, C3), compile only, `xcrun -sdk macosx metal -c`, "Apple metal version 32023.921".
- "Ours" marks arithmetic or reasoning done for this file on top of a quoted source.

Already settled elsewhere and not repeated here (cited as "ch05 A5", "infra A10" and so on):

| Topic | Item |
|---|---|
| CUDA: 1024 threads per block; block at most 1024 x 1024 x 64; warp size 32; grid limits | ch05 A5, ch03 A3 |
| CUDA: 64 K 32-bit registers per SM and per block, 255 per thread, on every listed compute capability | ch04 A12 |
| CUDA: occupancy, `nvcc --resource-usage`, `cudaFuncAttributes.numRegs`, `__launch_bounds__`, `--maxrregcount`, spilling | ch04 A13 |
| How a block is cut into warps (x fastest, then y, then z; runs of 32) | ch04 A8; HIP ch04 B6; Metal: implementation-defined, ch04 C2 |
| `__syncthreads()` semantics and the uniform-condition rule | ch04 A6; HIP ch04 B3 (rule not stated by AMD); Metal barrier ch05 C2, rule for loops ch04 C3 |
| Shared memory: 48 KB static per block, opt-in above it, per-SM carveout | ch05 A3 |
| H100 figures (3.35 TB/s, 67 TFLOPS FP32, 228 KB shared per SM) | ch05 A8; FP64 row ch01 A1; MI300X rows ch01 B3 |
| HIP spells `__shared__`, `__syncthreads()`, `threadIdx`, `blockIdx`, `blockDim` as CUDA does | ch05 B4, ch03 B2 |
| AMD wavefront width (64 on gfx9, 32 on gfx10 and later) | ch05 B5 |
| AMD occupancy, VGPR/SGPR pools, `__launch_bounds__` second argument, `numRegs` = VGPRs | ch04 B10, B9 |
| Metal `threadgroup` keyword and 32 KB cap (Apple4 and later); dynamic threadgroup memory | ch05 C1, C8 |
| Metal threadgroup size: 1024 theoretical, real cap per pipeline (`maxTotalThreadsPerThreadgroup`) | ch05 C3, ch03 C4 |
| MSL position attributes and their types | ch05 C4 |
| Metal documents no occupancy, register count or resident limits | ch04 C5 |
| Apple's threadgroup-shape recipe (width = `threadExecutionWidth`) | ch03 C3, ch04 C6 |
| Constant memory for the coefficients `c0..c6` | infra A5, B5, C6 |
| Shared-memory banks (CUDA 32; AMD 32 or 64); Apple documents no caches or banks | infra A10, B10, C13 |
| No `double` in MSL; half precision types | ch03 C6; infra A12, C8 |
| `#pragma unroll`: CUDA and HIP yes, absent from the MSL spec | infra A7, B7, and its "Where the backends really differ" table |

Totals: 13 CONFIRMED, 1 WRONG, 7 PARTLY, 2 COULD NOT VERIFY (23 items: A1 to A8, B1 to B4, C1 to C5, D1 to D4, E1 to E2). The WRONG item is the book's 1.52 FLOP/B (E2). Two items correct earlier facts files: B1 corrects ch03 B3, C4 adds to ch04 C5.

Device numbers stay out of lessons. The reader's own limits are already in the lab device report (`labs/common/src/backend_cudalike.cpp` and `backend_metal.mm`): `maxThreadsPerBlock`, `maxThreadsDim.{x,y,z}`, `sharedMemPerBlock`, `regsPerBlock`, `warpSize` on CUDA and HIP; `maxThreadsPerThreadgroup.{width,height,depth}` and `maxThreadgroupMemoryLength` on Metal. A pipeline's `maxTotalThreadsPerThreadgroup` is not in the device report; a Metal lab has to print it.

---

## Group A: CUDA, and the book's dated statements

### A1. "The 1024 limit on the block size" makes "the practical limit on t ... 8" (p. 192)

**Verdict: PARTLY.** The 1024-thread limit is confirmed and unchanged (ch05 A5; the guide keeps the book's hedge: "On current GPUs, a thread block may contain up to 1024 threads", ch03 A3). That it forces t = 8 is not a hardware fact: a 10 x 10 x 10 block (1000 threads) is within every documented CUDA limit.

- Per-dimension caps (ch05 A5, Table 33): x and y at most 1024, z at most 64. A cubic tile hits the total of 1024 long before z = 64, so only the total matters: t³ ≤ 1024 gives t ≤ 10.
- t = 10 is not a multiple of the warp width: 1000 threads are 32 warps, the last with 8 of 32 lanes in use (ch04 A8: "when the total is not a multiple of 32, the last warp of the thread block will have some lanes that are unused"). That is a cost, not a prohibition.
- What t = 10 buys, by the book's own formula (E1): 1.10 FLOP/B instead of 0.96, halo fraction 49% instead of 58%. The lesson can say "8 is the book's choice; 10 is legal; the formula shows both, and either way you stay far from 1.625".

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html (Table 33)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html (section 2.1.2.2.1)

### A2. An 8 x 8 x 8 tile is "not coalesced"; blocks with x-dimension 32 fix it (p. 193)

**Verdict: PARTLY.** The direction is right and NVIDIA documents the mechanism. Under NVIDIA's current model (32-byte transactions) the 8 x 8 x 8 tile is half-efficient, not uncoalesced in the guide's worst-case sense, and the book's fix is not perfect either because of the halo offset.

Quotes (programming guide, section 2.3.4.1 "Coalesced Global Memory Access"):

- "Global memory is accessed via 32-byte memory transactions. When a CUDA thread requests a word of data from global memory, the relevant warp coalesces the memory requests from all the threads in that warp into the number of memory transactions necessary to satisfy the request".
- "if consecutive threads in the warp request consecutive 4-byte words in memory, then the warp will request 128 bytes of memory total, and this 128 bytes required will be fetched in four 32-byte memory transactions."
- "Coalesced memory access occurs provided different threads in the warp access elements from the same 32-byte segments of memory in some linear or permuted way. Stated another way, the best way to achieve coalesced memory access is to maximize the ratio of bytes used to bytes transferred."
- "For perfectly coalesced access to 4-byte data elements, 4 global memory transactions will be required. In the worst case, 32 global memory transactions may be required to satisfy the addresses requested by a single load instruction from a single warp."
- Warp shape (ch04 A8, and "threadIdx.y has a stride of blockDim.x"): in an 8 x 8 x 8 block a warp is 4 consecutive y-rows of 8 consecutive x.

Ours, from the 32-byte model, for the Fig. 8.8 load `k = blockIdx.x*OUT_TILE_DIM + threadIdx.x - 1`, `float` data, array start aligned (cudaMalloc guarantees at least 256 bytes, infra A6), N a multiple of 8 so every row has the same alignment, interior blocks (`bx` ≥ 1; in block 0 the thread at `k = -1` does not load):

| Block shape | Warp covers | Row start `k0` | 32-byte transactions per warp load | Bytes used / fetched |
|---|---|---|---|---|
| 8 x 8 x 8 (OUT 6) | 4 rows of 8 floats | `6*bx - 1`, odd, so never a multiple of 8 | 2 per row, 8 in all | 128 / 256 = 50% |
| 32 x 8 x 4 (OUT 30) | 1 row of 32 floats | `30*bx - 1`, odd | 5 | 128 / 160 = 80% |
| ideal | 32 consecutive aligned floats | multiple of 8 | 4 | 100% |

So by NVIDIA's documented model the book's fix improves load efficiency from 50% to 80%, not from "uncoalesced" to "coalesced". What the model leaves out (whether four rows N floats apart cost more than four adjacent sectors, L1 and L2 effects) is not documented; the lab measures it. The same `- 1` offset applies to every kernel in the chapter, including the 32 x 32 planes of Figs. 8.10 and 8.12.

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (sections 2.3.2 and 2.3.4.1)
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (section 10.2.1.2, cudaMalloc alignment, quoted in infra A6)

### A3. "Registers have significantly lower latency and higher bandwidth than the shared memory" (p. 197)

**Verdict: PARTLY.** NVIDIA documents the latency order and says register access is normally free. It does not state a bandwidth comparison, and gives no numbers for either.

Quotes:

- Best Practices Guide, section 10.2: "Global, local, and texture memory have the greatest access latency, followed by constant memory, shared memory, and the register file."
- Section 10.2.7 "Registers": "Generally, accessing a register consumes zero extra clock cycles per instruction, but delays may occur due to register read-after-write dependencies and register memory bank conflicts."
- Programming guide 2.3.3.2 compares shared memory only with global memory: "because shared memory is located on each SM, the bandwidth is higher and the latency is lower than accessing global memory."

Safe lesson wording: "NVIDIA ranks the register file as the lowest-latency storage, below shared memory; it publishes no figures." AMD says the same in its own words (B4); Apple says nothing (C4).

- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (sections 10.2 and 10.2.7)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (section 2.3.3.2)

### A4. Register tiling on 32 x 32 blocks: "3072 more registers per block" (p. 197), and what can go wrong

**Verdict: CONFIRMED** (3 extra registers x 1024 threads = 3072 is arithmetic). Missing from the book and documented: a 1024-thread block has a hard per-thread register ceiling, and exceeding it makes the launch fail rather than run slowly.

- 64 K registers per block on every listed compute capability (ch04 A12). At 1024 threads that is at most 64 registers per thread (ours; NVIDIA does not publish the allocation granularity, ch04 A13, so the real ceiling can be lower).
- Best Practices Guide 11.3: "Note that when a thread block allocates more registers than are available on a multiprocessor, the kernel launch fails, as it will when too much shared memory or too many threads are requested."
- Runtime API: "cudaErrorLaunchOutOfResources = 701 This indicates that a launch did not occur because it did not have appropriate resources. Although this error is similar to cudaErrorInvalidConfiguration, this error usually indicates that the user has attempted to pass too many arguments to the device kernel, or the kernel launch specifies too many threads for the kernel's register count."
- The documented fix, guide section 5.4.3.2: "developers should include the single argument __launch_bounds__(maxThreadsPerBlock) which specifies the largest block size with which the kernel will launch. Failure to do so could result in "too many resources requested for launch" errors." With launch bounds "the compiler first derives the upper limit, L, on the number of registers that the kernel should use ... If the initial register usage exceeds L, the compiler reduces it until it is less than or equal to L. This usually results in increased local memory usage and/or a higher number of instructions."
- Register count of the compiled kernel: `nvcc --resource-usage` or `cudaFuncGetAttributes(...).numRegs` (ch04 A13). The Best Practices Guide also names "--ptxas options=v".

For the lab: give the 32 x 32 kernels `__launch_bounds__(IN_TILE_DIM * IN_TILE_DIM)` and print `numRegs`. HIP's default is different (B4), Metal's mechanism is different again (C5).

- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (section 11.3)
- https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html (section 5.4.3.2)

### A5. Fig. 8.12 "relies on the compiler keeping `inPrev`, `inCurr`, `inNext` in registers" (notes, Portability)

**Verdict: CONFIRMED** that NVIDIA documents when per-thread values leave registers, and the book's three scalars rotated by copying are the register-friendly form.

Quotes (programming guide, sections 2.3.3.3 and 2.3.3.4):

- "Register usage is managed by the compiler and registers are used for thread local storage during the execution of a kernel."
- "Automatic variables that the compiler is likely to place in local memory are: Arrays for which it cannot determine that they are indexed with constant quantities, Large structures or arrays that would consume too much register space, Any variable if the kernel uses more registers than available, that is register spilling."
- "Because the local memory space resides in device memory, local memory accesses have the same latency and bandwidth as global memory accesses".

Consequence (ours): the book's `inPrev = inCurr; inCurr = inNext;` keeps every access at a constant name. Writing the window as `float plane[3]` indexed by `i % 3` (a runtime index) matches the first bullet and is a candidate for local memory, which would undo register tiling. D1 shows a real kernel written that second way. Whether nvcc keeps such an array in registers in a given case is something to check with `--resource-usage`, not to assume.

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html (sections 2.3.3.3 and 2.3.3.4)

### A6. Three 32 x 32 planes, "12 KB per block", is "a reasonable level" (p. 195)

**Verdict: CONFIRMED.** 12,288 bytes (Fig. 8.10) and 4,096 bytes (Fig. 8.12) are below the 48 KB a block may allocate statically without opt-in on every listed compute capability: Table 34, "Maximum amount of shared memory per thread block, without opt-in (48 KB)" (ch05 A3). The 8 x 8 x 8 tile of Fig. 8.8 needs 2,048 bytes. How many such blocks fit on one SM depends on the per-SM capacity, which varies by compute capability (ch05 A3, A8); that number comes from the reader's device report, not the lesson.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html (Table 34)

### A7. "Single and half have much higher arithmetic computation throughput" than double "on modern CPUs and GPUs" (pp. 184 to 185)

**Verdict: PARTLY.** True for double on many NVIDIA parts, but not on the data-center parts the book uses as examples, where double runs at half the single rate. For half precision outside tensor cores NVIDIA's table shows a factor between 1 and 4 depending on the compute capability, and only for packed `half2`.

- Programming guide, Table 33, row "Ratio of FP32 to FP64 Throughput" (column spans read from the HTML): 7.5 is 32:1; 8.0 is 2:1; 8.6, 8.7, 8.9 are 64:1; 9.0 and 10.0 are 2:1; 10.3 is 64:1; 10.7 is 4:1; 11.0 and 12.x are 64:1. Footnote: "Non-Tensor Core throughputs."
- Best Practices Guide, Table 5 "Throughput of Native Arithmetic Instructions. (Number of Results per Clock Cycle per Multiprocessor)", columns 7.5, 8.0, 8.6/8.9, 9.0, 10.0, 10.3, 12.0/12.1: `add.f32` 64, 64, 128, 128, 128, 128, 128; `add.f64` 2, 32, 2, 64, 64, 2, 2; `add.f16x2` ("2-way SIMD") 64, 128, 64, 128, 64, 64, 64. The table's title says "Results", the row says "2-way SIMD", so the unit for this row is ambiguous. Read as instructions of two results each (ours), packed half against single is 2x on 7.5, 4x on 8.0, 1x on 8.6/8.9, 2x on 9.0, and 1x on 10.0, 10.3 and 12.x; read literally as results, half that. Neither reading gives "much higher" everywhere.
- Best Practices Guide 12.1.1: "In order to achieve good performance for 16-bit precision floating-point add, multiply or multiply-add, it is recommended that the half2 datatype is used".
- Product figures: H100 FP64 34 vs FP32 67 teraFLOPS (ch01 A1); AMD MI300X FP64 81.7 vs FP32 163.4 (ch01 B3). Metal has no `double` at all (ch03 C6).

For this chapter the point that survives is the bandwidth one, which is arithmetic: an 8-byte element halves every FLOP/B figure in E1 and doubles every shared-memory figure. The large half-precision speedups an ML reader knows come from tensor or matrix units (ch01 A1), which a stencil does not use.

- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html (Table 33)
- https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html (section 12.1.1, Table 5)

### A8. At 1.625 FLOP/B the 3D 7-point stencil is memory bound "on modern GPUs" (p. 189)

**Verdict: CONFIRMED** for the book's example device. With NVIDIA's H100 SXM figures (ch05 A8: 67 teraFLOPS FP32, 3.35 TB/s) the ratio is 20 FLOP/B (ours, 67 / 3.35), the same threshold the book uses in chapter 5 (p. 94, per the notes). 1.625 is below it by more than a factor of 10. Present it as the book's example; a reader's own ratio comes from their benchmark reports.

---

## Group B: HIP / ROCm

### B1. Block shapes on AMD: are 8 x 8 x 8, 10 x 10 x 10, 32 x 8 x 4 and 32 x 32 legal?

**Verdict: CONFIRMED.** All are legal. AMD documents 1024 threads per block and 1024 in each of x, y and z. The z cap differs from CUDA's 64.

- HIP "Hardware features" page (HIP 7.17.0), columns RDNA1, CDNA1, RDNA2, CDNA2, RDNA3, CDNA3, every column: "Maximum x-, y- and z-dimension of a block | 1024", "Maximum number of threads per block | 1024". Grid: "x - 2^32-1 y - 2^16-1 z - 2^16-1"; "Maximum number of threads per grid | 2^32 - 1".
- Runtime source at `therock-10.2` (`rocdevice.cpp`): the per-dimension values reported in `maxThreadsDim` are the driver's per-dimension maximum clamped to the work-group maximum: `info_.maxWorkItemSizes_[2] = std::min(max_workgroup_size[2], max_work_item_size);` with `max_work_item_size` set from `HSA_AGENT_INFO_WORKGROUP_MAX_SIZE`.
- The language-extensions page ties the compiler's default to this table: "If no __launch_bounds__ are specified, MAX_THREADS_PER_BLOCK is the maximum block size supported by the device (see Hardware features)."

Correction to ch03 B3, which says the current HIP documentation "no longer states any number": the Hardware features page of the same HIP 7.17.0 documentation does. The page has no RDNA4 or CDNA4 column; for those, read `maxThreadsDim` from the device report.

- https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hardware_features.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/rocclr/device/rocm/rocdevice.cpp (lines 1373 to 1395)
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (section "__launch_bounds__")

### B2. The book's coalescing argument (p. 193) on AMD

**Verdict: PARTLY.** AMD documents coalescing in the same terms (consecutive lanes, consecutive addresses) but gives no transaction size, so the A2 table cannot be reproduced from AMD documentation. The warp shapes also change with a 64-wide wavefront.

Quotes ("Performance optimization", section "Memory coalescing theory"; "Performance guidelines"):

- "When consecutive threads access consecutive memory addresses, the hardware can merge requests into efficient cache line accesses."
- "When a DRAM address is accessed, the hardware actually fetches or writes a burst: a run of consecutive addresses fetched together in a single transaction. If multiple threads in a warp access addresses that fall into the same burst, those logical accesses can be coalesced into that single transaction."
- "When consecutive threads in a warp access consecutive memory addresses, the hardware combines these into a single wide transaction. Non-coalesced patterns require multiple transactions, reducing effective bandwidth."
- "Padding 2D arrays to multiples of the warp size ensures each row starts at an aligned memory boundary."

Warp shapes (ours, from the linear order in ch04 B6 and the widths in ch05 B5): with 64-wide wavefronts (gfx9, Instinct) an 8 x 8 x 8 block is 8 wavefronts of 8 rows of 8, and a 32 x 8 x 4 or 32 x 32 block puts 2 rows of 32 in each wavefront. With 32-wide wavefronts (gfx10 and later, Radeon) the shapes are CUDA's. COULD NOT VERIFY: AMD's burst or cache-line size for global loads; no HIP page read states one.

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/performance_optimization.html (section "Memory coalescing theory")
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html (sections "Optimize 2D array access", "Coalesce memory accesses")

### B3. Shared memory (LDS) for the tiles and planes on AMD

**Verdict: CONFIRMED.** A block may use the whole LDS of a compute unit, and HIP reports that through the CUDA field names. Every LDS size in AMD's current product table is far above the 12 KB of Fig. 8.10.

- Source at `therock-10.2` (`hip_device.cpp`): `deviceProps.sharedMemPerBlock = info.localMemSizePerCU_;` and `deviceProps.sharedMemPerBlockOptin = info.localMemSizePerCU_;`. The opt-in value equals the default, so there is no 48 KB-style step on AMD (ours, from the two assignments). `sharedMemPerMultiprocessor` is the same value, doubled in WGP mode: `static_cast<size_t>(info.localMemSizePerCU_) * (deviceHandle->settings().enableWgpMode_ ? 2 : 1)`.
- AMD's GPU architecture table, column "LDS (KiB)": the values listed are 64, 128 and 160 (for example "MI300X | CDNA3 | gfx942 | ... | 64" and "MI355X | CDNA4 | gfx950 | ... | 160"). These are device figures for the facts file only; lessons read `sharedMemPerBlock` from the device report.

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/clr/hipamd/src/hip_device.cpp (lines 645 to 649 and 772)
- https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html

### B4. Registers and the 1024-thread block on AMD (the HIP side of A3, A4, A5)

**Verdict: PARTLY.** AMD agrees that registers are the fastest storage and documents the same register-versus-shared-memory trade-off as the book. But the compiler's default differs from CUDA's: without launch bounds HIP compiles for the device's largest block, so a 32 x 32 launch does not fail for lack of registers the way A4 describes; the compiler limits registers (and may spill) instead.

Quotes:

- "In GPU programming, registers are the fastest level of the memory hierarchy, holding per-thread variables for arithmetic and address computation." ("Performance optimization", "Register pressure theory")
- "These registers physically reside in the register file of a compute unit and are implemented using fast on-chip SRAM, typically the fastest accessible memory in the GPU. When a thread requires more storage than the available registers, some values may spill into global memory, which incurs a significant performance penalty due to much higher latency." ("Programming model", "Memory hierarchy")
- Default block size for code generation: "If no __launch_bounds__ are specified, MAX_THREADS_PER_BLOCK is the maximum block size supported by the device". And: "Reducing MAX_THREADS_PER_BLOCK allows the compiler to use more resources per thread than an unconstrained compilation." "MAX_THREADS_PER_BLOCK is particularly useful in cases, where the compiler is constrained by register usage in order to meet requirements of large block sizes that are never used at launch time." "When launching kernels HIP will validate the launch configuration to make sure the requested block size is not larger than MAX_THREADS_PER_BLOCK and return an error if it is exceeded."
- Per-thread cap, Hardware features table: "Maximum number of 32-bit vector registers per thread | 256" (RDNA) and "256 (vector) + 256 (matrix)" (CDNA).
- The book's escape hatch, moving planes back to shared memory when registers run short (p. 197), is AMD's advice too: "Moving temporary storage to shared memory trades register usage for shared memory usage, often allowing higher occupancy since shared memory limits are typically less restrictive." ("Performance guidelines", "Managing register pressure")

A slip in that AMD example: it declares `__shared__ float temp[blockDim.x][100];`. `blockDim.x` is not a compile-time constant, so as plain C++ this is not a valid array bound (our reading; not compiled). Do not copy it.

Not found: an AMD statement equivalent to NVIDIA's rule that a dynamically indexed per-thread array goes to local memory (A5). AMD documents only spilling.

For the lab: `__launch_bounds__(IN_TILE_DIM * IN_TILE_DIM)` is harmless on HIP and needed on CUDA, so use it in the shared `.cu`. Its second argument means different things on the two (ch04 B10); leave it out.

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/performance_optimization.html (section "Register pressure theory")
- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html (section "Memory hierarchy")
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html (section "__launch_bounds__")
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html (section "Managing register pressure")
- https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hardware_features.html

---

## Group C: Metal

### C1. Can a Metal threadgroup be 8 x 8 x 8, 10 x 10 x 10 or 32 x 32?

**Verdict: COULD NOT VERIFY** the per-dimension limits. The total is documented (1024 on Apple4 and later as a theoretical maximum, ch05 C3), the per-dimension values are not, and the real cap depends on the compiled kernel.

- Header (`MTLDevice.h`, macOS 27.0 SDK): "@property maxThreadsPerThreadgroup @abstract The maximum number of threads along each dimension." No values.
- Feature set tables, page 7: "Maximum threads per threadgroup" 1024 for Apple4 and later, with footnote 5: "The values in this row are the theoretical maximum number of threads per threadgroup. Check the actual maximum by inspecting the MTLComputePipelineState.maxTotalThreadsPerThreadgroup". No per-dimension row.
- The pipeline cap can fall below 1024 for a kernel that "needs more resources" (ch04 C5: "Metal may lower the thread limit when creating the compute pipeline state"). The register-tiled kernel is exactly the kind of kernel where that could happen; whether it does is a measurement.

For the lab: the device report already records `maxThreadsPerThreadgroup.{width,height,depth}`; the launcher must also print the pipeline's `maxTotalThreadsPerThreadgroup` and refuse a 32 x 32 launch it cannot honour, with a message, rather than dispatching a smaller group silently.

- https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf (page 7)
- Local header: `Metal.framework/Headers/MTLDevice.h` (macOS 27.0 SDK), lines 483 to 487

### C2. A 2D threadgroup in a 3D grid (Figs. 8.10, 8.12) must declare its positions as three-component vectors

**Verdict: CONFIRMED** (spec and compiler agree). ch05 C4 records the rule; this item records what it means for the coarsened kernels. In CUDA the kernel can use `threadIdx.x`, `threadIdx.y` and `blockIdx.z` freely. In MSL, declaring `uint3 tg [[threadgroup_position_in_grid]]` with `uint2 t [[thread_position_in_threadgroup]]` is a compile error.

- MSL spec section 5.2.3.6, page 157, "Notes on kernel function attributes": "The type for declaring [[thread_position_in_grid]], [[threads_per_grid]], [[thread_position_in_threadgroup]], [[threads_per_threadgroup]], [[threadgroup_position_in_grid]], [[dispatch_threads_per_threadgroup]], and [[threadgroups_per_grid]] needs to be a scalar type or a vector type. If it is a vector type, the number of components for the vector types for declaring these arguments need to match."
- Compiled here: the mixed declaration fails with "error: expecting input declarations with either all scalar types or all vector types with the same number of elements". Declaring both `uint3` (the threadgroup's z position is then always 0) compiles.
- llama.cpp declares every position attribute `uint3` even in kernels that use one component (D2).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (section 5.2.3.6, page 157)

### C3. The chapter's on-chip storage in MSL: a 3D threadgroup array, planes, a barrier inside the z loop

**Verdict: CONFIRMED.** Everything the four kernels need has a direct MSL form, and all of it fits Metal's threadgroup-memory cap.

- Declarations as compiled here: `threadgroup float in_s[IN_TILE][IN_TILE][IN_TILE];` and `threadgroup float inCurr_s[IN_T2][IN_T2];` in the kernel body, with `threadgroup_barrier(mem_flags::mem_threadgroup)` twice per iteration inside the z loop (compile only; correctness belongs to the lab). The 2D form is already verified on Metal in `labs/ch05/lab2_tiled`.
- Barrier inside a loop is allowed if every thread executes it in every iteration (ch04 C3: "If threadgroup_barrier ... is inside a loop, for each iteration of the loop, if any thread in the threadgroup ... executes the barrier, then all threads in the threadgroup ... need to execute the barrier function"). The chapter's loops have a uniform trip count, so this holds.
- Sizes against the 32 KB cap on Apple4 and later (ch05 C1): 2,048 bytes (8³ tile), 12,288 bytes (three 32 x 32 planes), 4,096 bytes (one plane).
- Tile sizes as compile-time constants: use preprocessor macros (`MTLCompileOptions.preprocessorMacros`, ch05 C7) or `constexpr`. Function constants, which llama.cpp and MLX use for kernel sizes (D2, D4), are documented to "Control code paths that get compiled", "Specify the optional arguments of a function" and "Specify optional elements of a structure"; sizing a threadgroup array is not on that list, and "Function constants can only be a scalar or vector type" (MSL spec section 5.8, page 190). MLX sizes its threadgroup tile with `constexpr` values.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 4.4, 5.8 and 6.10.1)

### C4. Coalescing and registers on Apple GPUs

**Verdict: COULD NOT VERIFY any documented equivalent.** Apple documents neither a memory-transaction model nor registers. Both halves of the chapter's argument (A2 and A3 to A5) have no Metal source; on Metal they are hypotheses the lab tests.

- MSL spec 4.1: the word "coalesc" does not occur; "register" occurs only in the list of C++ features Metal does not support ("register, thread_local storage attributes", section 1.5.4 "Restrictions", page 13).
- Caches, banks and bandwidth: nothing quantitative (infra C13). Registers, occupancy, resident limits: nothing (ch04 C5).
- The nearest documented guidance is the threadgroup-shape recipe: width `threadExecutionWidth`, height `maxTotalThreadsPerThreadgroup / width` (ch03 C3, ch04 C6). That is the Metal counterpart of the book's "x-dimension 32" advice, without a stated reason.
- Addition to ch04 C5: Xcode's documentation does name an occupancy counter. "The bottom section has a separate Counters timeline, which includes GPU counters such as Occupancy, Limiter, and Bandwidth." ("Optimizing GPU performance"). It is a profiler counter with no definition on that page and no API; it gives a reader something to look at, not a number a lesson can compute.
- A web search restricted to developer.apple.com for coalescing found no Apple page on it.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
- https://developer.apple.com/documentation/xcode/optimizing-gpu-performance

### C5. Holding the 32 x 32 threadgroup size: Metal's counterpart of `__launch_bounds__`

**Verdict: PARTLY.** Metal has attributes that fix or cap the threadgroup size per kernel. Unlike CUDA (A4) and HIP (B4), Apple does not say that the compiler uses them to budget registers, and the MSL spec names the wrong class for the Metal 4 property.

- MSL spec section 5.1.3, page 127: "You can use the [[max_total_threads_per_threadgroup]] function attribute with a kernel function to specify the maximum threads per threadgroup." "If the [[max_total_threads_per_threadgroup]] value is greater than the [MTLDevice maxThreadsPerThreadgroup] property, then compute pipeline state creation fails."
- Same page: "In Metal 4 and later, you can use the [[required_threads_per_threadgroup]] function attribute with a kernel function to specify the number of threads per threadgroup. ... If the [[required_threads_per_threadgroup]] value is set and the [MTLDevice requiredThreadsPerThreadgroup] property is set, the values must be the same; otherwise, the compute pipeline state creation fails."
- Slip in that sentence: the macOS 27.0 headers have no `requiredThreadsPerThreadgroup` on `MTLDevice`. It is on `MTLComputePipelineDescriptor` ("Sets the required threads-per-threadgroup during dispatches. The `threadsPerThreadgroup` argument of any dispatch must match this value if it is set.", macOS 26.0 and later), on `MTLComputePipelineState` (read-only), and on `MTL4ComputePipelineDescriptor`, `MTLFunctionDescriptor` (`MTLLibrary.h`) and render pipeline descriptors.
- Compile-option form, section 1.6.6, page 17: `-fmax-total-threads-per-threadgroup=<value>`; the attribute "takes precedence over the compile option".

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 1.6.6 and 5.1.3)
- Local headers: `Metal.framework/Headers/MTLComputePipeline.h` (lines 146 to 151 and 292 to 296), `MTL4ComputePipeline.h` (line 52), `MTLLibrary.h` (lines 413 to 417), macOS 27.0 SDK

---

## Group D: how llama.cpp, MLX and PyTorch do the same things

LLM inference has no 3D PDE sweeps, but it has short convolutions along the token axis (the depthwise causal convolution in Mamba-style layers, `ssm_conv` in ggml), which are 1D stencils. Those and depthwise convolutions are where the chapter's choices show up in the three codebases.

### D1. llama.cpp, CUDA (and HIP): `ssm_conv` is thread coarsening with a register window, and the long-sequence path is a halo tile

**Verdict: CONFIRMED** (read from source; `ggml/src/ggml-cuda/ssm-conv.cu`, last commit 2026-11-18).

- Short sequences (`n_t <= 32`), kernel `ssm_conv_f32`: one thread per channel (`threads = 128`), each thread marches along all tokens, keeping the last `d_conv` inputs in a per-thread array and loading one new value per step: `float x[d_conv] = { 0.0f };` then `x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];` and `sumf += x[(i + j) % d_conv] * w[j];`. That is Fig. 8.12's rolling window in one dimension, with the weights also held in registers (`float w[d_conv]`).
- The window is indexed with the runtime token counter `i`, which matches NVIDIA's first local-memory case in A5 ("Arrays for which it cannot determine that they are indexed with constant quantities"). Whether nvcc keeps it in registers here was not checked (not compiled). The book's copy rotation avoids the question.
- Long sequences, kernel `ssm_conv_long_token_f32`: the token axis is split into tiles of `split_n_t = 32`, each block loads its tile plus a halo of `d_conv - 1` into dynamic shared memory (`extern __shared__ float smem[];`, `smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float)`), synchronises, then computes from shared memory. That is Fig. 8.8's halo tile in one dimension.
- Kernel size is a template parameter, instantiated for 2, 3, 4, 5, 7, 9, 15 (`default: GGML_ABORT("Only support kernel sizes 2, 3, 4, 5, 7, 9, 15 right now.")`); the lab's `IN_TILE_DIM` plays the same role.
- The same file builds the HIP backend (ch05 B9).

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ssm-conv.cu

### D2. llama.cpp, Metal: the same operation as the basic kernel

**Verdict: CONFIRMED** (`ggml/src/ggml-metal/kernels/ssm.metal`, last commit 2026-09-19).

- `kernel_ssm_conv_f32_f32_batched`: "each thread handles one token, threadgroup covers BATCH_SIZE tokens"; each thread reads its `nc` inputs straight from device memory, `FOR_UNROLL (int64_t i0 = 0; i0 < nc; ++i0) { sumf += s[i0] * c[i0]; }`. No threadgroup memory, no coarsening: Fig. 8.6's shape, relying on caches for the overlap between neighbouring tokens.
- Kernel size and the SiLU fusion come from function constants: `constant int FC_ssm_conv_nc [[function_constant(FC_SSM_CONV + 2)]];` (compare C3).
- `FOR_UNROLL` is `#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)` in `kernels/common.h`, a clang pragma; the MSL spec documents no unroll control (infra table). The same header has `#define N_SIMDWIDTH 32 // assuming SIMD group size is 32`.
- All position attributes are `uint3` (C2).

- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/ssm.metal
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/common.h

### D3. PyTorch, CUDA: depthwise 3D convolution ships as the basic kernel

**Verdict: CONFIRMED** (`aten/src/ATen/native/cuda/DepthwiseConv3d.cu`, last commit 2026-11-03).

- Forward kernel `conv_depthwise3d_cuda_kernel`: a grid-stride loop over all outputs (`CUDA_KERNEL_LOOP(index, num_output)`), the 5D index recovered by division, then three nested loops over the filter that read the input directly from global memory with a bounds test per tap (`if (in_frame >= 0 && in_row >= 0 && in_col >= 0 && in_frame < iT && in_row < iH && in_col < iW)`). No shared memory. The forward launch uses `int64_t block = 256;` and caps the grid, `int64_t grid = std::min((num_outputs - 1) / block + 1, (int64_t)65536);`, so on large tensors each thread loops over several outputs. That is coarsening for launch overhead only: consecutive outputs of one thread are a whole grid apart and share no inputs, unlike the z-column of Fig. 8.10.
- Filter sizes known at compile time are template parameters (`kKnownKernelT`, `kKnownKernelH`, `kKnownKernelW`) with a runtime fallback.
- A source comment shows why the indexing starts from the channel base: "Index from the channel base: with padding, the window origin lies before the input, and forming such a pointer is undefined behavior." The same issue arises for the book's `- 1` halo offset if a kernel forms a pointer before the array start.

So this library operator is Fig. 8.6 in shape, not Fig. 8.12; the chapter's later kernels are what a contributor would add.

- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/DepthwiseConv3d.cu

### D4. MLX, Metal: depthwise convolution with a threadgroup halo tile

**Verdict: CONFIRMED** (`mlx/backend/metal/kernels/conv.metal`, last commit 2026-10-12).

- `depthwise_conv_2d`: the threadgroup loads an input tile plus halo into `threadgroup T ins[TGH * TGW * TGC];` (sizes from `constexpr` values), writes `T(0)` for positions outside the input, then `threadgroup_barrier(mem_flags::mem_threadgroup);` and computes from threadgroup memory. That is chapter 7's zero-filling tile, not Fig. 8.8's "skip the load" version.
- It linearises its threads as `const int thread_idx = simd_gid * 32 + simd_lid;`, assuming a SIMD-group width of 32, which Apple does not document (ch04 C2).
- Filter size and stride are function constants (`constant int ker_h [[function_constant(00)]];`).
- `depthwise_conv_1d`, the operation Mamba-style layers use, is the basic form: one output per thread, `for (int i = 0; i < kernel_size; ++i) { acc += static_cast<float>(in[0]) * w[i]; in += strides[1]; }`.

- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/conv.metal

---

## Group E: the chapter's arithmetic, recomputed

All values are ours, computed exactly from the book's formulas as recorded in the notes (pp. 189 to 195): 13 FLOP per output, 4-byte values, every input loaded once per block and every output stored once.

### E1. The book's figures that hold

**Verdict: CONFIRMED.**

| Quantity | Formula | Value |
|---|---|---|
| Basic kernel (Fig. 8.6) | 13 / (8 x 4) | 0.40625, printed 0.41 |
| Ideal | 13 / 8 | 1.625 |
| Ideal at finite n (input not counting the 12 edges) | 13(n-2)³ / 4(n³ - 12n + 16 + (n-2)³) | n = 64: 1.55; n = 128: 1.59; n = 512: 1.62 |
| 2D halo fraction, 32 x 32 input tile, radius 1 | 124 / 1024 | 12.1% |
| Fig. 8.10 shared memory, t = 32 | 3 x 32² x 4 | 12,288 bytes |
| Fig. 8.12 shared memory, t = 32 | 32² x 4 | 4,096 bytes |
| Fig. 8.12 extra registers, 32 x 32 block | 3 x 1024 | 3,072 |
| Exercise 2c (32 x 32 block, 16 output planes) | 13 x 14,400 / 4(18,432 + 14,400) | 1.43 |

Cubic tiles (Fig. 8.8), AI = 13(t-2)³ / 4(t³ + (t-2)³), halo fraction = 1 - (t-2)³/t³:

| t | Threads t³ | Legal block on CUDA? | AI (FLOP/B) | Halo fraction | Shared memory 4t³ |
|---|---|---|---|---|---|
| 4 | 64 | yes | 0.36 | 87.5% | 256 B |
| 6 | 216 | yes | 0.74 | 70.4% | 864 B |
| 8 | 512 | yes | 0.96 (book: 0.96) | 57.8% (book: 58%) | 2,048 B |
| 10 | 1,000 | yes (A1) | 1.10 | 48.8% | 4,000 B |
| 16 | 4,096 | no | 1.30 | 33.0% | 16,384 B |
| 32 | 32,768 | no | 1.47 | 17.6% | 131,072 B |

### E2. "With t = 32 ... 1.52 FLOP/B" for the coarsened kernel (p. 195)

**Verdict: WRONG** as a consequence of the book's own formula and of Fig. 8.10 as coded; it is the limit for an unboundedly deep slab.

- Fig. 8.10 with `IN_TILE_DIM = 32` and a slab of `OUT_TILE_DIM = 30` output planes loads planes `iStart - 1` to `iStart + 30`, that is 32 planes of 32 x 32, and stores 30 x 30 x 30. That is the cube with t = 32: 13 x 30³ / 4(32³ + 30³) = 1.468.
- For a slab of D output planes: AI(D) = 13 x 30² x D / 4(32² (D + 2) + 30² D). D = 16 gives 1.43 (exercise 2c), D = 30 gives 1.47, D = 64 gives 1.50, D = 128 gives 1.51, and the limit as D grows is 13 x 30² / 4(32² + 30²) = 1.520.
- So 1.52 counts only the x-y halo. It is reachable only by making each block march much further in z than Fig. 8.10 does, which is a design knob the explorer can expose (slab depth) next to tile width.

---

## Summary of corrections

Items where the book, an earlier facts file, or a vendor document turned out WRONG or PARTLY:

- A1: the 1024-thread limit is real, but it caps a cubic tile at t = 10, not 8; 10 x 10 x 10 is legal on CUDA (and HIP, B1), and t = 10 gives 1.10 FLOP/B and a 49% halo by the book's formula. "t = 8" is the book's power-of-two choice.
- A2: under NVIDIA's documented 32-byte transaction model an 8 x 8 x 8 tile loads at 50% efficiency (8 transactions per warp), and the book's 32 x 8 x 4 fix at 80% (5 transactions), because of the `- 1` halo offset; the book's "not coalesced" overstates the first and its fix does not reach 100%.
- A3: NVIDIA documents the latency order (register file below shared memory) and "zero extra clock cycles" for register access, but no bandwidth comparison and no figures.
- A4 (missing from the book): a 32 x 32 block leaves at most 64 registers per thread on CUDA; above that the launch fails with `cudaErrorLaunchOutOfResources` unless the kernel has `__launch_bounds__(1024)`.
- A5 (missing from the book): a per-thread window indexed with a runtime value (`plane[i % 3]`) is, by NVIDIA's rule, a candidate for local memory; the book's copy rotation avoids it.
- A7: double runs at half the single rate on compute capabilities 8.0, 9.0 and 10.0 (A100, H100, B200 class), so "much higher throughput" for single holds only on other parts; non-tensor half is between 1x and 4x single depending on the compute capability (NVIDIA's table unit is ambiguous for packed rows), and only packed as `half2`.
- E2: the book's 1.52 FLOP/B for t = 32 does not follow from its p. 192 formula or from Fig. 8.10 as coded (1.47); 1.52 is the deep-slab limit.
- B1 (corrects ch03 B3): HIP 7.17.0 does document block limits, on its "Hardware features" page: 1024 threads per block and 1024 in each of x, y and z (CUDA's z cap is 64).
- B2: AMD documents coalescing only qualitatively (bursts, cache lines), with no transaction size; on 64-wide wavefronts the book's warp shapes change.
- B4: HIP compiles kernels for the device's maximum block size unless told otherwise, so the CUDA launch failure in A4 is not the AMD failure mode; AMD's example `__shared__ float temp[blockDim.x][100];` is not valid C++.
- C2 (consequence of ch05 C4): MSL rejects a `uint2` thread position next to a `uint3` threadgroup position; the coarsened kernels must declare all positions `uint3`.
- C4 (adds to ch04 C5): Xcode's documentation names an "Occupancy" GPU counter in the Metal debugger, undefined there and with no API; Apple still documents no coalescing model and no registers.
- C5: the MSL spec says `[MTLDevice requiredThreadsPerThreadgroup]`; the property is on `MTLComputePipelineDescriptor` (and other descriptors), not `MTLDevice`. Whether the size attributes change register allocation is not documented.
- C1: Metal's per-dimension threadgroup limits are not documented anywhere read; read `maxThreadsPerThreadgroup` and the pipeline's `maxTotalThreadsPerThreadgroup`.

Not corrections, but worth knowing:

- D1: llama.cpp's CUDA `ssm_conv` is a real inference kernel built from this chapter's two techniques (a rolling register window for short sequences, a halo tile in shared memory for long ones); its Metal counterpart (D2) and PyTorch's depthwise 3D convolution (D3) are the basic kernel.
- D4: MLX's depthwise 2D convolution assumes a SIMD-group width of 32 when linearising threads.

## Where the backends really differ

One line per cell, for what this chapter's kernels and lab touch. "Same" means the same source text works on both CUDA and HIP according to the documentation. Nothing here has been run on CUDA or AMD hardware by us.

| Feature | CUDA | HIP (ROCm 10.2) | Metal (MSL 4.1) |
|---|---|---|---|
| Threads per block | 1024 on every listed compute capability | 1024 (Hardware features page) | 1024 theoretical on Apple4 and later; real cap per pipeline, `maxTotalThreadsPerThreadgroup` |
| Per-dimension block limit | 1024 x 1024 x 64 | 1024 x 1024 x 1024 | `maxThreadsPerThreadgroup` per dimension; values not documented |
| Is a 10 x 10 x 10 block legal? | Yes | Yes | Only if the pipeline's cap is at least 1000 |
| Warp shape of an 8 x 8 x 8 block | 16 warps, each 4 rows of 8 | wave64: 8 wavefronts of 8 rows of 8; wave32: as CUDA | SIMD-group partition is implementation-defined |
| Coalescing model | Documented: 32-byte transactions, "ratio of bytes used to bytes transferred" | Documented qualitatively ("bursts", cache lines); no size | Not documented |
| Shared memory for the tiles | `__shared__ float a[8][8][8]`; 48 KB static per block, more by opt-in | Same declaration; a block may use the CU's whole LDS (`sharedMemPerBlock` = `sharedMemPerBlockOptin`) | `threadgroup float a[8][8][8]` in the kernel body; 32 KB total |
| 2D block marching through a 3D grid | `threadIdx.x`, `threadIdx.y`, `blockIdx.z` | Same | All position attributes declared `uint3` (vector widths must match) |
| Barrier inside the z loop | `__syncthreads()`; uniform-condition rule documented | Same call; AMD states no rule (use CUDA's) | `threadgroup_barrier(mem_flags::mem_threadgroup)`; loop rule documented |
| Registers at 1024 threads | 64 K per block, so at most 64 per thread; launch fails (`cudaErrorLaunchOutOfResources`) without `__launch_bounds__(1024)` | Compiler targets the device's maximum block size by default; at most 256 VGPRs per thread; may spill instead | Not documented; the pipeline's thread cap may drop instead |
| Block-size hint | `__launch_bounds__(maxThreads, minBlocksPerSM, maxBlocksPerCluster)` | `__launch_bounds__(maxThreads, minWarpsPerEU)` | `[[max_total_threads_per_threadgroup(n)]]`; Metal 4 `[[required_threads_per_threadgroup(n)]]` |
| Reading a kernel's register use | `nvcc --resource-usage`, `cudaFuncAttributes.numRegs` | `--resource-usage`, `hipFuncAttributes.numRegs` (VGPRs only) | Not exposed; Xcode shows an undefined "Occupancy" counter |
| Runtime-indexed per-thread array | Documented candidate for local memory (device memory speed) | Spilling documented; no indexing rule found | Not documented |
| Register vs shared memory speed | Latency order documented, no figures | "registers are the fastest level", no figures | Not documented |
| `double` for a high-precision sweep | Yes; FP32:FP64 from 2:1 to 64:1 by compute capability | Yes | Not supported |
| Compile-time tile size | `#define` or template parameter | Same | Preprocessor macro or `constexpr`; function constants cannot size an array |
