# Appendix C — CUDA memories, address spaces, and pointers

- **Book pages:** 623–630  **PDF pages:** 647–654 (pdf = book + 24). Content ends on p. 629; p. 630 is blank.
- **Contributor line:** "With special contributions from Isaac Gelado and Mark Harris" (p. 623).
- **In one paragraph:** A short history of how CUDA's memory model moved from "host memory and device memory are separate, copy with `cudaMemcpy()`" to a model where CPU and GPU share one virtual address space, pointers mean the same thing on both sides, and pages migrate on demand. Each step is tied to a CUDA release or GPU architecture: unified device address space (Fermi), zero-copy access to pinned host memory (CUDA 2.2), Unified Virtual Address Space (CUDA 4), large 64-bit address spaces and a unified physical address space across GPUs (Kepler), Unified Memory with `cudaMallocManaged()` (CUDA 6), 49-bit addressing and GPU page-fault handling (Pascal), and explicit virtual-address-range control (CUDA 11). After reading, one can say for each era which pointer may be dereferenced where, what it costs, and why the simple copy model was not enough. The appendix has no exercises and one figure.

## Section map

| § | Title | Book pp. | What it teaches (one or two lines) |
|---|---|---|---|
| (intro) | untitled opening | 623 | The simple separate-memories model from Chapter 2 and the two application-level problems it causes (I/O devices; large host-side data structures). |
| C.1 | Unified device memory address space | 624 | Fermi (2009): global, local and shared memory in one GPU address space; one pointer type; composable device functions. |
| C.2 | Zero-copy access to host memory | 624–625 | CUDA 2.2 (2009): kernels read/write pinned host memory over the interconnect; allocation and pointer-translation API; bandwidth penalty and when to use it. |
| C.3 | Unified virtual address space | 625 | CUDA 4 (2011): one virtual address space for host and device (UVAS); `cudaMemcpyDefault`; what UVAS does and does not guarantee. |
| C.4 | Large virtual and physical GPU address spaces | 625–626 | 32-bit limits of early GPUs (4 GB); Kepler's 64-bit virtual / ≥40-bit physical addresses. |
| C.5 | Unified physical address space | 626 | A GPU can dereference pointers into another GPU's memory on the same PCIe bus / NVLink domain. |
| C.6 | Unified memory | 626–627 | CUDA 6 (2013): managed memory, one pointer for CPU and GPU, runtime migration; porting example (Fig. C.1); limits on Kepler/Maxwell. |
| C.7 | Access to full host virtual address space | 627–628 | Pascal (2016): 49-bit virtual addressing covers the CPU's 48-bit space plus GPU memory. |
| C.8 | Page fault handling | 628–629 | Pascal: GPU page faults enable on-demand migration or mapping, no pre-launch flush, system-wide Unified Memory, pointer-chasing into host data. |
| C.9 | Virtual address space control | 629 | CUDA 11: `cuMemAddressReserve()`, `cuMemCreate()`, `cuMemMap()` for custom layouts across devices. |
| — | References | 629 | One reference: Gelado et al., ASPLOS 2010 (the GMAC paper). |

## Concepts

### Opening: the simple model and its problems (p. 623)

- **Simple model (from Chapter 2):** each device has its own **device memory** (the book equates this with CUDA **global memory**), separate from **host memory** (also called **system memory**). Input data is copied host → device with `cudaMemcpy()` before a kernel runs; results are copied device → host with `cudaMemcpy()` before the host can use them. Early GPU generations supported only this model.
- It is easy to understand but causes two application-level problems:
  1. **I/O devices.** Disk controllers and network interface cards are built to work on host memory. With a separate device memory, input must go host → device and output device → host, adding I/O latency and cutting achievable I/O throughput. Letting I/O devices work directly on device memory would improve performance and simplify code.
  2. **Large host-side data structures.** Traditional programming systems (the book names Python) keep application data in host memory. Device memory in CUDA GPUs "tends to offer higher bandwidth but smaller capacity" than host memory, so developers must partition large structures into chunks that fit. Example given: in Chapter 21 the 3D electrostatic energy grid was split into 2D slices moved between host and device. Some structures have no good partition at all. Better: the GPU reads host memory directly, or the CUDA runtime migrates the data a kernel actually uses.
- These needs pushed CUDA system software and GPU hardware forward; researchers proposed solutions from CUDA's early days (ref. [1]). The rest of the appendix is "a brief history of advancements".

### §C.1 Unified device memory address space (p. 624)

- **Before:** in early CUDA GPUs, shared memory, local memory and global memory were separate address spaces. Pointers could only point into global memory.
- **Fermi architecture (introduced 2009, per the book):** these memories became parts of one **unified address space**. A single set of load/store instructions and pointer addresses can access global, local or shared memory, instead of distinct instructions and pointers for each.
- Effects:
  - The programmer deals with *which* memory holds a value only at allocation time.
  - CUDA data objects can be passed to other procedures/functions regardless of the memory they live in.
  - Code becomes **composable**: one device function can accept a pointer to any of these memories. Previously a device function needed a separate implementation per memory type for each argument.
  - Performance still depends on the target: faster if the pointer refers to shared memory, slower if global. Manual placement and transfer remain available as optimizations.
  - Lowered the cost of building production-quality CUDA libraries.
  - Enabled full C and C++ pointer support in kernels and device functions.

### §C.2 Zero-copy access to host memory (pp. 624–625)

- **CUDA 2.2 (2009):** host code can give a kernel a special *device data pointer to host memory*. Kernel code on the GPU dereferences it and accesses host memory directly over the system interconnect (e.g. the PCIe bus), with no `cudaMemcpy()`.
- **Zero-copy memory is pinned host memory** (see Chapter 23). API as the book gives it:
  - Allocate with `cudaHostAlloc()` passing `cudaHostAllocMapped` as the flag argument. Other flag values are "for more advanced usage".
  - The pointer returned by `cudaHostAlloc()` **cannot** be passed to the kernel directly.
  - Host code calls `cudaHostGetDevicePointer()` to obtain a valid device pointer and passes *that* to the kernel.
  - So host code and device code use **different pointer values** for the same host data.
- Why pinned: the OS must not page the data out while the GPU is accessing it (p. 624, referring to Chapter 23).
- **Cost:** access suffers the long latency and limited bandwidth of the interconnect. The book's figure: interconnect bandwidth is "typically less than 10% of the global memory bandwidth" (p. 625).
- Reasoning chain (p. 625): Chapter 5 showed kernels are usually limited by global-memory bandwidth unless tiling cuts global accesses per floating-point operation. If most accesses go to zero-copy memory instead, the limit becomes the even lower interconnect bandwidth.
- **Guidance:** use zero-copy memory only for application data structures that a kernel accesses **occasionally and sparsely** (p. 625).

### §C.3 Unified virtual address space (p. 625)

- **CUDA 4 (2011)** introduced the **Unified Virtual Address Space (UVAS)**, which the book says originated from the **GMAC** library (ref. [1]).
- **Before:** host and device each had their own virtual address space mapping to their own physical memory. The same physical location could therefore have different virtual addresses on host and device, which is exactly the zero-copy situation in C.2.
- **UVAS:** one virtual address space shared by host and device, with the guarantee that each physical memory address maps to only one virtual location.
- Consequence: the runtime can tell whether a pointer refers to host or device memory just from its value. So `cudaMemcpy()` no longer needs an explicit direction: pass `cudaMemcpyDefault` and the direction is inferred from the pointers.
- **What UVAS does not give you:** accessibility. Under CUDA 4:
  - Host code cannot dereference a device pointer returned by `cudaMalloc()`, and vice versa.
  - **Exception — zero-copy memory:** host code can pass a pointer to memory allocated with `cudaHostAlloc()` to the device as a kernel parameter. When the kernel dereferences it, the value is translated to a physical system-memory location and accessed over PCIe.
  - **Pointer chasing is not generally safe:** the kernel cannot necessarily dereference a pointer *value it reads from memory* (e.g. following links in a linked structure) unless every such pointer refers to memory allocated with `cudaHostAlloc()`.
- Two remaining limits motivate what follows: restrictions on the kinds of data structures supported, and the low bandwidth of zero-copy access.

### §C.4 Large virtual and physical GPU address spaces (pp. 625–626)

- **Early CUDA GPUs:** 32-bit virtual addresses and up to 32-bit physical addresses.
  - Device memory capped at **4 GB** (limit of 32 physical address bits).
  - Kernels could only operate on data sets under 4 GB (limit of 32-bit pointers), whether the data was in host or device memory.
  - Modern CPUs use 64-bit virtual addresses with **48 bits actually used**. Host virtual addresses did not fit in the GPU's 32-bit addresses, which contributed to the data-structure restrictions and to requiring the special `cudaHostAlloc()` allocation for zero-copy.
- **Kepler GPU architecture ("introduced in 2013", per the book)** and later: **64-bit virtual addresses** and **physical addresses of at least 40 bits**.
  - GPUs can carry more than 4 GB of DRAM; kernels can work on large data sets.
  - Opens the door to better host/device models: host and device can use **exactly the same pointer value** for a piece of data, wherever it resides.

### §C.5 Unified physical address space (p. 626)

- The large physical address space lets CUDA system software place the device memories of different GPUs in one **unified physical address space**.
- Benefit: one GPU can directly access the memory of any other GPU attached to the same **PCIe bus and/or NVLink domain** by dereferencing a pointer mapped to that GPU's physical address.
- **Before Kepler:** GPU-to-GPU communication (the book's example: halo exchange in the Chapter 23 stencil) required device-to-device memory copies triggered by host code. Costs: extra memory to hold copied data, plus overhead from host-device synchronization and copy setup.
- **After:** pass a device pointer as a kernel-call parameter; the kernel loads/stores data residing in the other GPU's memory.

### §C.6 Unified memory (pp. 626–627)

- **CUDA 6 (2013)** introduced **Unified Memory**: a pool of **managed** virtual memory pages shared between CPU and GPU.
- Managed memory is accessible from both CPU and GPU through a **single pointer**.
- A managed variable can physically reside in CPU memory, GPU memory, "or even both". The CUDA runtime software and hardware provide **data migration and coherence**.
- Net effect: managed memory looks like CPU memory to CPU code and like GPU memory to GPU code.
- The application still has to synchronize concurrent access (the book mentions barriers or atomic operations).
- A shared global virtual address space lets every variable be reached by the same virtual address from CPU or GPU code.
- **Benefit highlighted: easier porting** (Fig. C.1, p. 627). Two changes to the host code:
  1. `cudaMallocManaged()` and `cudaFree()` in place of `malloc()` and `free()`.
  2. Launch a kernel and synchronize the device instead of calling the CPU sort function.
  - A parallel sort kernel still has to exist; the point is that the host-side change is small and maintainable.
- **Limits of CUDA 6 Unified Memory** on Kepler and Maxwell hardware (p. 627):
  - All managed memory modified by the CPU had to be **flushed to GPU device memory before any grid launch**.
  - CPU and GPU could **not simultaneously access** a managed allocation.
  - The Unified Memory address space was **limited to the size of GPU physical memory**.
  - Cause: those architectures could not support coherence between host and device memories in hardware, and migration was mostly done in software.

### §C.7 Access to full host virtual address space (pp. 627–628)

- **Pascal GPU architecture (2016):** GPU address translation extended to **49-bit virtual addressing**.
- 49 bits covers the 48-bit virtual address spaces of modern CPUs plus the GPU's own memory.
- Unified Memory programs can then treat the full address spaces of all CPUs and GPUs in the system as one virtual address space, no longer limited by the GPU's address-translation width.
- CPUs and GPUs can "truly share the pointer values", so GPUs can traverse pointer-based data structures in host memory.

### §C.8 Page fault handling (pp. 628–629)

- **GPU memory page-fault handling**, starting with Pascal, is described as the crucial feature for seamless Unified Memory.
- With the system-wide virtual address space, page-fault handling **removes the need to flush all managed memory to the GPU before each grid launch**.
- **Coherence mechanism** (p. 628): host and device may each invalidate the other's copy of data when they modify it in managed memory. Invalidation uses page mapping and protection mechanisms. At grid launch the runtime no longer brings all GPU copies up to date. If the grid touches data whose device copy was invalidated by the host, the GPU takes a page fault, the data is brought up to date, and execution resumes.
- **On-demand migration:** if a grid accesses a page not resident in device memory, it page-faults and the page is automatically migrated to GPU memory.
- **Alternative — mapping instead of migrating:** the page may be mapped into the GPU address space and accessed over the interconnect if it is expected to be accessed only occasionally; "mapping on access can sometimes be faster than migration".
- This **generalizes zero-copy**: from a limited part of host memory to the entire host memory and other GPUs' memory.
- **System-wide:** GPUs and CPUs can fault and migrate pages from CPU memory or from other GPUs' memory.
- **CPU side:** if a CPU function dereferences a pointer to a variable currently mapped to GPU physical memory, the access is still serviced, "but perhaps at a longer latency". This makes it easier to call **legacy libraries not ported to GPUs**; in the earlier model the developer had to copy data device → host by hand first.
- **Linked data structures** (pp. 628–629): starting with Pascal, a GPU can traverse a linked structure even if it does not reside in zero-copy memory, because host and device code use the same pointer value for the same variable. Pointers embedded by the host can be followed by the device and vice versa.
- **Large in-core data sets:** in areas such as CAD, hosts may have hundreds of gigabytes of physical memory because the application needs the whole data set "in core". Direct access to very large CPU memories makes GPU acceleration of these applications feasible (p. 629).

### §C.9 Virtual address space control (p. 629)

- **CUDA 11** added low-level APIs for more flexible allocation:
  - `cuMemAddressReserve()` reserves a range of virtual address space.
  - `cuMemCreate()` allocates physical memory on any device.
  - `cuMemMap()` maps that physical memory at any position in the reserved range.
- Use: custom layouts of data structures across multiple devices. Example given: a 3D volume allocated across several devices but referenced through a single pointer.
- No year is given for CUDA 11, and no code.

### Timeline as the book states it

| When (per book) | Vehicle | Feature | § | Page |
|---|---|---|---|---|
| early | early GPU generations | separate host/device memory, `cudaMemcpy()` only | intro | 623 |
| 2009 | Fermi architecture | unified device memory address space (global/local/shared) | C.1 | 624 |
| 2009 | CUDA 2.2 | zero-copy access to pinned host memory | C.2 | 624 |
| 2011 | CUDA 4 | Unified Virtual Address Space (UVAS), `cudaMemcpyDefault` | C.3 | 625 |
| 2013 | Kepler architecture | 64-bit virtual, ≥40-bit physical addresses; unified physical address space across GPUs | C.4, C.5 | 626 |
| 2013 | CUDA 6 | Unified Memory (`cudaMallocManaged()`), limited on Kepler/Maxwell | C.6 | 626–627 |
| 2016 | Pascal architecture | 49-bit virtual addressing; GPU page-fault handling | C.7, C.8 | 627–628 |
| not dated | CUDA 11 | `cuMemAddressReserve()` / `cuMemCreate()` / `cuMemMap()` | C.9 | 629 |

### Who can dereference what (assembled from the text)

| Memory / pointer | Host code | Kernel code | Source |
|---|---|---|---|
| `cudaMalloc()` device pointer | cannot dereference | yes | C.3, p. 625 |
| `cudaHostAlloc(..., cudaHostAllocMapped)` pointer, CUDA 2.2 | yes | only via the distinct pointer from `cudaHostGetDevicePointer()`; access goes over the interconnect | C.2, p. 624 |
| same, under UVAS (CUDA 4) | yes | the pointer can be passed directly as a kernel parameter; pointers *read from memory* are only safe if they also point into `cudaHostAlloc()` memory | C.3, p. 625 |
| `cudaMallocManaged()` pointer, CUDA 6 on Kepler/Maxwell | yes, but not at the same time as the GPU | yes; all CPU-modified managed data flushed to the device before each grid launch; total limited to GPU physical memory | C.6, pp. 626–627 |
| managed memory, Pascal and later | yes; may be slower if the page currently lives on a GPU | yes; pages fault in on demand (migrated or mapped); can follow host-built linked structures | C.7–C.8, pp. 627–629 |
| another GPU's device memory, Kepler and later | not discussed | yes, by dereferencing a pointer mapped to the other GPU (same PCIe bus / NVLink domain) | C.5, p. 626 |

## Figures

| Fig. | Book p. | Kind | What it shows | Interactive idea |
|---|---|---|---|---|
| C.1 | 627 | code | Two code boxes side by side. Left box, lavender background, heading "CPU code" in purple: a function `sortfile(FILE *fp, int N)` that mallocs N bytes, `fread`s the file into the buffer, calls `qsort_char(data, N, 1)`, then (after a blank line) `use_data(data)` and `free(data)`. Right box, mint-green background, heading "CUDA 6  code with unified memory" in green: the same function with four changes shown in red: `cudaMallocManaged(&data, N);` replaces the malloc line, the sort call becomes a kernel launch `qsort_char<<<...>>>(data, N, 1);`, a new line `cudaDeviceSynchronize()` fills the position of the blank line, and `cudaFree(data);` replaces `free`. The `fread` and `use_data` lines are identical in both, operating on the same `data` pointer. Caption: "Unified Memory simplifies porting of CPU code (left) to CUDA code (right)." Point: with managed memory the host fills and consumes the buffer through the same pointer the kernel uses; no `cudaMemcpy()` and no second buffer. | Diff view with the four changed tokens highlighted; then a third column showing what the same function would need under the simple copy model (separate device buffer + two `cudaMemcpy()` calls) for contrast. Step through "where do the bytes live now?" after each line. |

## Code listings

### Fig. C.1 — `sortfile`: CPU version vs CUDA 6 Unified Memory version (p. 627)

- **Purpose:** show how little host code changes when porting with Unified Memory. Signature in both versions: `void sortfile(FILE *fp, int N)`. No launch configuration is given; the kernel launch is written with an elided configuration, `<<<...>>>`. The kernel body (`qsort_char`) is not shown.
- **Walkthrough, CPU version (left):**
  1. Declare `char *data;`
  2. `data = (char *)malloc(N);`
  3. `fread(data, 1, N, fp);` reads N bytes from the file.
  4. `qsort_char(data, N, 1);` sorts on the CPU.
  5. `use_data(data);` consumes the sorted data.
  6. `free(data);`
- **Walkthrough, CUDA version (right):**
  1. Declare `char *data;`
  2. `cudaMallocManaged(&data, N);` allocates managed memory (takes the address of the pointer, unlike `malloc`).
  3. `fread(data, 1, N, fp);` unchanged: the CPU writes straight into managed memory.
  4. `qsort_char<<<...>>>(data, N, 1);` launches the sort as a kernel with the *same* pointer.
  5. `cudaDeviceSynchronize()` waits for the kernel before the CPU touches the data.
  6. `use_data(data);` unchanged: the CPU reads results through the same pointer.
  7. `cudaFree(data);`
- **Lines that need care:**
  - The device synchronization before `use_data(data)` is what makes the CPU read safe. The text describes this as "launch a kernel and perform device synchronization" (p. 627), and separately notes that under CUDA 6 on Kepler/Maxwell the CPU and GPU could not access a managed allocation simultaneously.
  - As printed, `cudaDeviceSynchronize()` has no trailing semicolon (see "Unclear in the PDF").
  - The prose says the kernel replaces "the `qsort()` function" while the figure's function is named `qsort_char`.
  - No error checking, and no data transfer calls anywhere: migration is implicit.
- **APIs / qualifiers / built-ins used:** `cudaMallocManaged()`, `cudaFree()`, `cudaDeviceSynchronize()`, kernel launch syntax `<<<...>>>`; C library `malloc`, `free`, `fread`.

APIs named in running text but not shown in code: `cudaMemcpy()`, `cudaMemcpyDefault`, `cudaMalloc()`, `cudaHostAlloc()`, `cudaHostAllocMapped`, `cudaHostGetDevicePointer()`, `cuMemAddressReserve()`, `cuMemCreate()`, `cuMemMap()`.

## Exercises

| # | Book p. | Gist | Type | Answer sketch |
|---|---|---|---|---|
| — | — | Appendix C has no exercises section. | — | — |

## Dated or device-specific content

Every section of this appendix is tied to a release or architecture. Listed precisely for "today" notes:

- Early GPU generations supported only the separate host/device memory model (p. 623).
- Device memory "tends to offer higher bandwidth but smaller capacity compared to the host memory" (p. 623).
- `[device: NVIDIA Fermi architecture]` introduced 2009; unified device memory address space (p. 624).
- CUDA 2.2, 2009: zero-copy host memory via `cudaHostAlloc()` + `cudaHostAllocMapped` + `cudaHostGetDevicePointer()` (p. 624).
- System interconnect named as PCIe; its bandwidth "typically less than 10% of the global memory bandwidth" (pp. 624–625).
- CUDA 4, 2011: UVAS, originating from the GMAC library; `cudaMemcpyDefault` (p. 625).
- Early CUDA GPUs: 32-bit virtual, up to 32-bit physical addresses; 4 GB device-memory and data-set limits (pp. 625–626).
- Modern CPUs: 64-bit virtual addresses with 48 bits used (p. 626).
- `[device: NVIDIA Kepler architecture]` "introduced in 2013": 64-bit virtual addresses, physical addresses of at least 40 bits; unified physical address space; direct GPU-to-GPU access over the same PCIe bus and/or NVLink domain (p. 626).
- CUDA 6, 2013: Unified Memory, `cudaMallocManaged()` (p. 626).
- `[device: NVIDIA Kepler and Maxwell architectures]` Unified Memory limits: flush before every grid launch, no simultaneous CPU/GPU access, address space limited to GPU physical memory (p. 627).
- `[device: NVIDIA Pascal architecture]` 2016: 49-bit virtual addressing; GPU page-fault handling; on-demand migration or mapping; system-wide Unified Memory (pp. 627–628).
- CAD-style hosts with "hundreds of gigabytes" of physical memory (p. 629).
- CUDA 11: `cuMemAddressReserve()`, `cuMemCreate()`, `cuMemMap()` (p. 629). No year given.
- Reference [1]: Gelado, Stone, Cabezas, Patel, Navarro, Hwu, "An asymmetric distributed shared memory model for heterogeneous parallel systems", ASPLOS 2010, pp. 347–358 (p. 629).
- Not covered: anything newer than CUDA 11 / Pascal in the memory model; no numbers for migration cost, page size or fault latency; no memory-advice or prefetch APIs.

## Relevance to LLM-inference kernels

The appendix never mentions LLMs, model weights, or any non-NVIDIA platform. It matters to the course as the book's account of *why* a discrete-GPU memory model looks the way it does, which is the baseline the course contrasts with shared-memory systems. What the book provides:

- **Baseline cost model.** Host and device memories are separate; data crosses an interconnect whose bandwidth is under 10% of device global-memory bandwidth (p. 625). For inference that frames model loading and any per-token host/device traffic as the expensive path, and keeping tensors resident in device memory as the cheap one. (Framing for inference is ours; the bandwidth figure is the book's.)
- **Weights larger than device memory.** The book's second motivating problem (large host structures that must be chunked to fit device memory, p. 623) is the same situation as a model that does not fit in VRAM. The book's answers are zero-copy mapping for sparse/occasional access (C.2), and Unified Memory with on-demand migration or mapping (C.8). It gives the rule that zero-copy suits only occasionally and sparsely accessed data (p. 625); GEMM and attention read their operands densely, so by that rule they should not run out of mapped host memory. (Application to GEMM is ours.)
- **Unified Memory as a porting convenience.** Fig. C.1 shows the same pointer used for file read, kernel, and CPU post-processing (p. 627). For an inference prototype this is the shortest path from file to kernel. The book does not quantify page-fault or migration overhead, so the performance consequences for hot loops are not something these notes can state.
- **Sampling on the CPU.** C.8's point that a CPU function can dereference a pointer whose page currently lives on the GPU, at higher latency, and that this eases calling un-ported legacy libraries (p. 628), maps onto running sampling or tokenization on the CPU over GPU-produced logits. (Mapping is ours.)
- **Pointer semantics inside kernels.** C.1 (one pointer type for global/shared/local, p. 624) is why a device helper such as a dot-product or softmax routine can take a plain pointer regardless of whether the data sits in shared or global memory, with speed depending on which.
- **Multi-GPU.** C.5 and C.9 (direct access to another GPU's memory; one pointer spanning devices, pp. 626, 629) are the book's hooks for splitting a large tensor or model across GPUs.

Contrast with systems where CPU and GPU share physical memory (e.g. Apple Silicon): **the book says nothing about them.** The concepts it defines that the course will need for the contrast are: separate physical memories and an interconnect; pinned memory; one virtual address space vs accessibility (UVAS, p. 625); migration vs mapping (p. 628); coherence by invalidation and page faults (p. 628). Which of these exist, are unnecessary, or take another form when CPU and GPU share one physical memory must come from other sources. `[unverified]`

Quantized ops, softmax, top-k/top-p, argmax: little direct relevance beyond where the tensors live.

## Portability notes

Everything here is CUDA- and NVIDIA-specific. The book states no HIP or Metal facts in this appendix, so every mapping below is `[unverified]`.

- **Separate host/device memory + `cudaMemcpy()`** (p. 623): the default CUDA model. HIP equivalent calls and whether the same model applies on AMD hardware: `[unverified]`. Metal on a shared-memory SoC: whether any copy is needed between CPU-visible and GPU-visible buffers, and under which storage modes: `[unverified]`.
- **Unified device address space / plain pointers in kernels** (C.1, p. 624): CUDA kernels take C/C++ pointers into global, shared or local memory. Whether Metal Shading Language permits equivalent raw pointer use, and how HIP treats it: `[unverified]`.
- **Zero-copy pinned host memory** (`cudaHostAlloc()`, `cudaHostAllocMapped`, `cudaHostGetDevicePointer()`; C.2, p. 624): HIP analogues `[unverified]`. On a shared-memory system the interconnect-bandwidth penalty described by the book may not apply in the same way; `[unverified]`.
- **UVAS and `cudaMemcpyDefault`** (C.3, p. 625): HIP analogue `[unverified]`; Metal has a different (non-pointer) resource model as far as these notes can say, `[unverified]`.
- **Unified Memory** (`cudaMallocManaged()`, `cudaFree()`, `cudaDeviceSynchronize()`; C.6, pp. 626–627): HIP managed-memory support and its behaviour (migration, page faults) on AMD GPUs: `[unverified]`. The term "unified memory" in this appendix means a *software-managed, migrating* pool across two physical memories; it must not be equated with a hardware-shared physical memory without checking. `[unverified]`
- **GPU page-fault handling and on-demand migration** (C.8, p. 628): tied to Pascal-and-later NVIDIA hardware. Existence of GPU-side page faulting on AMD or Apple GPUs: `[unverified]`.
- **Unified physical address space across GPUs; NVLink** (C.5, p. 626): NVIDIA-specific interconnect; multi-GPU peer access on other platforms `[unverified]`.
- **`cuMemAddressReserve()` / `cuMemCreate()` / `cuMemMap()`** (C.9, p. 629): the `cu` prefix (rather than `cuda`) is as printed; these are described as low-level APIs. Analogues elsewhere `[unverified]`.
- **Chapter 2's name for the mechanism:** a footnote in Chapter 2 (p. 35) calls it "Unified Virtual Address (UVA)" and points here; this appendix uses "UVAS". Same feature, two abbreviations in the book.

## Ideas for interactives and labs

- **Memory-model timeline (whole appendix).** A horizontal timeline with the eight milestones from the table above. Selecting one redraws a host/GPU diagram: which address spaces exist, which arrows (copy, map, migrate) are available, and what the limits are.
- **"Can I dereference this?" quiz (C.2, C.3, C.6, C.8).** Show a pointer's origin (`malloc`, `cudaMalloc`, `cudaHostAlloc`, `cudaMallocManaged`), an era, and a code location (host or kernel); learner predicts valid / invalid / valid-but-slow, then reveals the book's rule with page reference.
- **Fig. C.1 diff stepper.** Step through the right-hand listing line by line with a two-box picture (CPU memory, GPU memory) showing where the managed pages are after `fread`, during the kernel, and at `use_data`. Offer two modes: "CUDA 6 on Kepler/Maxwell" (bulk flush at launch) and "Pascal+" (fault in on first touch).
- **Zero-copy bandwidth slider (p. 625).** Fraction of a kernel's accesses that go to zero-copy memory versus device memory, with interconnect bandwidth fixed at 10% of global-memory bandwidth as an upper bound from the book; plot the resulting bandwidth-limited throughput. Makes the "occasionally and sparsely" rule concrete.
- **Coherence-by-invalidation simulator (p. 628).** A small page table with valid bits for host and device copies. Learner issues CPU writes, GPU writes and a grid launch; the simulator shows invalidations and page faults. Compare the pre-Pascal flush-everything policy with on-demand faults by counting pages moved.
- **Migrate or map? (p. 628).** Given an access pattern over a page (touch count), choose migrate or map; reveal which wins under adjustable costs. The book only says mapping "can sometimes be faster", so label the costs as learner-chosen parameters.
- **Address-width calculator (C.4, C.7).** Bits → addressable bytes: 32 bits → 4 GB; 40 bits; 48 bits; 49 bits. Show why 49 covers 48-bit CPU space plus GPU memory.
- **Lab: three ways to feed a kernel.** Same simple kernel fed by (a) `cudaMalloc` + `cudaMemcpy`, (b) zero-copy mapped host memory, (c) `cudaMallocManaged`. Measure end-to-end and kernel-only time for dense and sparse access. Expected direction from the book: (b) is poor for dense access (p. 625); (c) changes little host code (p. 627). No magnitudes are given by the book.
- **Contrast panel for shared-memory systems.** A side-by-side of the concepts this appendix needs (two physical memories, interconnect, pinning, migration, page faults) with an empty column to be filled from verified vendor documentation, not from this book.

## Cross-references

Made by this appendix:

- Chapter 2: the simple host/device memory model and `cudaMemcpy()` (p. 623).
- Chapter 21: 3D electrostatic energy grid partitioned into 2D slices to fit device memory (p. 623).
- Chapter 23: pinned host memory and why it is needed (p. 624, twice); halo exchange in the multi-GPU stencil as an example of GPU-to-GPU communication (p. 626).
- Chapter 5: kernels are limited by global-memory bandwidth unless tiling reduces accesses per floating-point operation (p. 625).
- Reference [1], the GMAC paper (ASPLOS 2010): cited for early research proposals (p. 623) and as the origin of UVAS (p. 625).

Made by other chapters to this appendix (found by searching the extracted text):

- Chapter 2 (p. 29): other forms of host/device memory allocation are discussed "in Chapter 23 and in Appendix C".
- Chapter 2 (p. 35, footnote 3): kernel code can access host memory through "Unified Virtual Address (UVA)" at lower bandwidth than device global memory; discussed in Appendix C.
- Chapter 24 (p. 582): outline lists Appendix C.

## Unclear in the PDF

Possible errata and ambiguities. Judgements that something is an error are ours unless marked "read directly".

- **Dangling back-reference about `cudaHostAlloc()` flags (p. 624).** "We mentioned that the other values of the flag argument are for more advanced usage." A search of the extracted text finds `cudaHostAlloc` only in Appendix C; Chapter 23 (p. 562) allocates pinned memory with `cudaMallocHost()` instead. So the earlier mention does not appear to exist in this edition, and the book uses two different pinned-allocation APIs without relating them.
- **Kepler year (p. 626).** The text says the Kepler architecture was "introduced in 2013", the same year it gives for CUDA 6. Recorded as printed; verify the date before using it in a timeline.
- **Fermi and CUDA 2.2 both "2009" (p. 624).** Recorded as printed; verify.
- **Fig. C.1 (p. 627), read directly:** `cudaDeviceSynchronize()` is printed without a semicolon; the heading has a double space ("CUDA 6  code"); the kernel launch configuration is elided as `<<<...>>>`; the prose calls the replaced function `qsort()` while the figure names it `qsort_char`. `cudaMallocManaged(&data, N)` passes a `char **` with no cast, which is fine as illustration but is not shown with any error handling.
- **Typo (p. 624), read directly:** "shared, memory, local memory, and global memory" (stray comma).
- **"The rest of this section" (p. 623).** The opening calls the appendix a "section".
- **UVAS vs UVA.** Appendix C says "Unified Virtual Address Space (UVAS)" (p. 625); Chapter 2's footnote (p. 35) says "Unified Virtual Address (UVA)".
- **Scope of pointer-chasing on Pascal (pp. 628–629).** The text says a GPU "can traverse a linked data structure even if the data structure does not reside in zero-copy memory". It appears in the Unified Memory discussion, but the book does not state explicitly whether the host structure must have been allocated with `cudaMallocManaged()` or whether ordinary `malloc()` memory qualifies. Do not assert either without checking current documentation.
- **"zero-copy access to the entire host memory" (p. 628).** Same ambiguity: whether this covers arbitrary host allocations or only managed ones is not spelled out.
- **Migration vs mapping policy (p. 628).** The book says a page "may be mapped" instead of migrated when access is expected to be occasional, without saying who decides (runtime heuristics or programmer) or through which API.
- **"up to 32-bit physical addresses" vs "physical addresses of at least 40 bits" (pp. 625–626).** No exact per-architecture physical widths are given.
- **CUDA 11 date and details (p. 629).** No year, no flags, no code; only the three function names.
- **p. 630** is blank in the PDF (PDF p. 654).
