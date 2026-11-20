# Appendix C fact check: CUDA memories, address spaces and pointers, against CUDA, HIP and Metal today

Checked on 2026-11-20 against the live vendor sources listed under each item.

How to read this file:

- Same conventions as `notes/today/ch05-facts.md`. **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY. Where an item has several sub-claims the verdict is for the item as a whole and the text says which sub-claim failed. A verdict judges the book's statement, or the assumption a course author would most likely bring to the item (stated in the item title).
- **Quote** is copied from the source (HTML docs flattened to text, PDF text, Apple's documentation JSON, GitHub raw files, local SDK headers). Line breaks inside quotes are removed.
- Already settled elsewhere and not repeated here, cited by item number:
  - `ch02-facts.md` A4 (`cudaMemcpyDefault` = 4, "Requires unified virtual addressing"), A15 (UVA is about addresses; kernel access to host memory is "mapped memory" or unified memory, at lower bandwidth), A16 (signatures of `cudaMallocHost`, `cudaHostAlloc`, `cudaMallocManaged`), A17 (CUDA host code may not dereference `cudaMalloc` memory), B7 (AMD's allocation table: `hipMalloc` memory is host-accessible by "zero copy"; `hipMallocManaged` needs HMM and `HSA_XNACK=1` to migrate, else behaves like pinned host memory), C2 (Metal allocation, `newBufferWithBytesNoCopy` needs page-aligned memory, `maxBufferLength`), C3 (shared buffers need no copy; private buffers are filled by a blit copy whose offsets and size must be multiples of 4 on macOS), C7 (no `cudaMemcpyDefault` counterpart needed in Metal), D1 to D4 (llama.cpp switches `cudaMalloc` to `cudaMallocManaged`; MLX allocates every Metal array as a shared buffer and uses `cudaMallocManaged` on its CUDA backend; PyTorch's `tensor.to("cuda")` is `cudaMemcpyAsync`; `tensor.to("mps")` still copies).
  - `ch05-facts.md` C6 (Apple silicon: one system memory shared by CPU and GPU; shared storage needs no copy, only ordering; `MTLStorageModeManaged` deprecated, "Managed storage has no effect on Apple Silicon"), C9 (`GPUStartTime`/`GPUEndTime` per command buffer).
  - `infra-facts.md` A8 and B8 (pinned memory, `cudaHostAlloc` flags, HIP renames `cudaMallocHost` to `hipHostMalloc`, `hipMemcpyDefault`), C12 (no pinned-memory concept documented for Metal; blit copy as the device-to-device copy).
  - `ch01-facts.md` A8 (GPU memory against CPU memory bandwidth; no ratio exists on unified-memory chips).
- Versions the sources reported about themselves on the check date: CUDA Programming Guide, Runtime API and Driver API "v13.5"; CUDA Toolkit Archive lists 13.5.0 (November 2026) as latest; ROCm Core SDK 10.2.0, headers at tag `therock-10.2` in `ROCm/rocm-systems` (`projects/hip/VERSION`: major 7, minor 17); Metal Shading Language Specification "Version 4.1" dated 2026-06-04; macOS 27.0 SDK headers on this machine; Linux kernel docs "7.4.0-rc3"; PyTorch `main` at `d473a25fd` (2026-11-20, docs "stable" = 2.15); llama.cpp `master` at `9d6f04fa9` (2026-11-20); MLX `main` at `5078f27ec` (2026-11-20).
- One thing was checked by running code, and it is labelled where it appears (C3): a 30-line Objective-C probe on this Mac that printed a shared buffer's CPU pointer and GPU address. Nothing was run on CUDA or AMD hardware.

Totals: 24 CONFIRMED, 3 WRONG, 10 PARTLY, 1 COULD NOT VERIFY (38 items: A1 to A19, B1 to B8, C1 to C6, D1 to D5). The WRONG and PARTLY items are collected under "Summary of corrections".

---

## Group A: the book's dated statements (CUDA)

### A1. Separate host and device memories, copied with `cudaMemcpy()`, is still the default; device memory is faster but smaller (p. 623)

**Verdict: CONFIRMED** for discrete GPUs. NVIDIA still opens its memory chapter with separate physical memories and explicit placement. The capacity-versus-bandwidth trade-off is still stated by NVIDIA and AMD.

- Guide 2.6: "Heterogeneous systems have multiple physical memories where data can be stored. The host CPU has attached DRAM, and every GPU in a system has its own attached DRAM. Performance is best when data is resident in the memory of the processor accessing it. CUDA provides APIs to explicitly manage memory placement, but this can be verbose and complicate software design."
- NVIDIA Pascal blog (Dec 14, 2016): "Since the capacity of GPU memory is significantly lower than system memory, it creates a barrier for developers accustomed to programming just one memory space."
- AMD HIP unified memory page: "GPUs usually have specialized memory whose bandwidth is a magnitude higher than the RAM attached to CPUs."

Exceptions the book does not discuss: NVIDIA's own integrated (Tegra) and NVLink-C2C (Grace Hopper) systems, AMD's MI300A and Apple silicon (A17, B8, ch05 C6).

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html
- https://developer.nvidia.com/blog/beyond-gpu-memory-limits-unified-memory-pascal/
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html

### A2. Fermi, "introduced 2009", put global, local and shared memory in one address space (C.1, p. 624)

**Verdict: PARTLY.** The feature is confirmed, in more detail than the book gives. The year needs a qualifier: NVIDIA's Fermi whitepaper is dated 2009, but NVIDIA says Fermi products reached the market in March 2010.

Feature:

- Fermi whitepaper (NVIDIA, © 2009, PDF dated 2 Oct 2009): "Fermi and the PTX 2.0 ISA implement a unified address space that unifies the three separate address spaces (thread private local, block shared, and global) for load and store operations. In PTX 1.0, load/store instructions were specific to one of the three address spaces". "A single set of unified load/store instructions operate on this address space, augmenting the three separate sets of load/store instructions for local, shared, and global memory." "PTX 2.0 makes it possible to use unified pointers to pass objects in any memory space, and Fermi's hardware address translation unit automatically maps pointer references to the correct memory space."
- PTX ISA today (section 6.4.1.1, "Generic Addressing"): "If a memory instruction does not specify a state space, the operation is performed using generic addressing. The state spaces .const, Kernel Function Parameters (.param), .local and .shared are modeled as windows within the generic address space." "A generic address maps to global memory unless it falls within the window for const, local, or shared memory." Release notes in the same document: "Generic addressing and cache operations introduced in PTX ISA version 2.0." "Generic addressing requires sm_20 or higher." Conversions between a generic and a state-space address are the `cvta` instruction; `isspacep` tests which window an address falls in.
- So the book's point stands today: a `__device__` function taking `float*` works for global or shared data, and the hardware (not the source) decides which memory a load touches. The book's "faster if the pointer refers to shared memory" is its own claim; NVIDIA's pages read here do not quantify it.

Dates:

- NVIDIA Kepler press release (March 21, 2012): "Kepler ... succeeds the 40-nm NVIDIA Fermi architecture, which was first introduced into the market in March 2010."

- https://www.nvidia.com/content/PDF/fermi_white_papers/NVIDIA_Fermi_Compute_Architecture_Whitepaper.pdf (pages 7 and 12)
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (sections 6.4.1.1, 9.7.10.22, 9.7.10.23)
- https://nvidianews.nvidia.com/news/nvidia-launches-first-geforce-gpus-based-on-next-generation-kepler-architecture-6622805

### A3. CUDA 2.2, 2009: zero-copy access to pinned host memory with `cudaHostAlloc(..., cudaHostAllocMapped)` and `cudaHostGetDevicePointer()` (C.2, p. 624)

**Verdict: CONFIRMED** for the date and the API, which still exists unchanged. What changed since 2009 is that the extra steps are no longer needed, and NVIDIA has renamed the feature.

- Toolkit archive: "CUDA Toolkit 2.2 (May 2009)".
- `cudaHostAlloc` flag, reference v13.5: "cudaHostAllocMapped: Maps the allocation into the CUDA address space. The device pointer to the memory may be obtained by calling cudaHostGetDevicePointer()." Signature `cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags)`; "flags provides for future releases. For now, it must be set to 0."
- Renamed: "Mapped memory has also been referred to as zero-copy memory in previous documents." (guide 2.6.3.1.3)
- No longer needed, under unified addressing (runtime API 6.36.4): "All host memory allocated through all devices using cudaMallocHost() and cudaHostAlloc() is always directly accessible from all devices that support unified addressing. This is the case regardless of whether or not the flags cudaHostAllocPortable and cudaHostAllocMapped are specified." "It is not necessary to call cudaHostGetDevicePointer() to get the device pointer for these allocations." Guide 2.6.3.1.3: "Prior to all CUDA applications using a unified virtual address space, additional APIs were needed to enable memory mapping (cudaSetDeviceFlags with cudaDeviceMapHost). These APIs are no longer needed."
- Still needed, for two exceptions (runtime API 6.36.6, "Exceptions, Disjoint Addressing"): "These exceptions are host memory registered using cudaHostRegister() and host memory allocated using the flag cudaHostAllocWriteCombined. For these exceptions, there exists a distinct host and device address for the memory." On devices with `cudaDevAttrCanUseHostPointerForRegisteredMem` the host pointer also works.
- The book's rule of use matches NVIDIA's today. Book: only for data a kernel accesses "occasionally and sparsely". Guide 2.6.3.1: "Mapped memory should not be considered a performant alternative to unified memory or explicit memory management for the majority of a kernel's memory needs." Guide 2.6.3.1.3 adds two things the book does not: "Mapped memory makes CPU memory accessible from the GPU, but does not guarantee that all types of access, for example atomics, are supported on all systems." and "Atomic functions ... operating on mapped host memory are not atomic from the point of view of the host or other GPUs."
- On HMM or ATS systems (A12) the step disappears entirely: "On systems with HMM or ATS, all host memory is directly accessible from the GPU using the host pointers."

So the book's statement that "host code and device code use different pointer values" is true of 2009 and of today's registered and write-combined memory, and false for today's `cudaMallocHost`/`cudaHostAlloc` memory.

- https://developer.nvidia.com/cuda-toolkit-archive
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__UNIFIED.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html (section 2.6.3)

### A4. Interconnect bandwidth is "typically less than 10% of the global memory bandwidth" (C.2, p. 625)

**Verdict: PARTLY.** True for a PCIe-attached GPU on NVIDIA's current figures; not true for NVIDIA's NVLink-C2C systems. NVIDIA no longer states a ratio; the arithmetic below is ours.

- H100 product page, spec table: SXM "GPU Memory Bandwidth 3.35TB/s", interconnect "PCIe Gen5: 128GB/s". 128 / 3350 is about 3.8%. The page does not say whether 128 GB/s counts both directions.
- GH200 page: "With 900 gigabytes per second (GB/s) of coherent interface, the superchip is 7X faster than PCIe Gen5." The same page gives "10 terabytes per second (TB/s) of memory bandwidth" for the two-superchip GH200 NVL2. 900 GB/s against half of 10 TB/s is about 18%, above the book's bound.
- What NVIDIA says instead of a ratio (guide 2.6.3.1.3): "Latency of accesses made across these links are significantly higher than access to GPU memory, and total available bandwidth is lower."

For lessons: keep the 10% as the book's rule of thumb for PCIe, present the H100 figures only as the book's example device, and say that NVLink-C2C narrows the gap. Do not print a ratio for the reader's machine; measure it.

- https://www.nvidia.com/en-us/data-center/h100/
- https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/

### A5. CUDA 4, 2011: Unified Virtual Address Space; `cudaMemcpyDefault`; "each physical memory address maps to only one virtual location"; UVAS came from GMAC (C.3, p. 625)

**Verdict: PARTLY.** Date, name and `cudaMemcpyDefault` are confirmed. The one-virtual-location guarantee has documented exceptions. The GMAC origin could not be verified from any NVIDIA source.

- Toolkit archive: "CUDA Toolkit 4.0 (May 2011)". NVIDIA's CUDA 6 post: "CUDA has supported Unified Virtual Addressing (UVA) since CUDA 4".
- Name: the current guide's heading is "2.6.1. Unified Virtual Address Space"; the runtime reference section is "Unified Addressing"; NVIDIA's blog says "Unified Virtual Addressing (UVA)". So the book's "UVAS" (Appendix C) and "UVA" (Chapter 2 footnote) are both NVIDIA usages.
- Today's definition (guide 2.6.1): "A single virtual address space is used for all host memory and all global memory on all GPUs in the system within a single OS process. All memory allocations on the host and on all devices lie in this virtual address space. This is true whether allocations are made with CUDA APIs (e.g. cudaMalloc, cudaMallocHost) or with system allocation APIs (e.g. new, malloc, mmap). The CPU and each GPU has a unique range within the unified virtual address space." Location from a pointer: `cudaPointerGetAttributes()`.
- When it is on (runtime API 6.36.2): "Unified addressing is automatically enabled in 64-bit processes". Query: `cudaDeviceProp::unifiedAddressing`.
- Exceptions to "one virtual location": registered and write-combined host memory have "a distinct host and device address" (A3). The book's "address-only, not access" reading is right and is what NVIDIA's CUDA 6 post says: "UVA does not automatically migrate data from one physical location to another, like Unified Memory does."
- GMAC: NVIDIA's documents and posts read for this file do not mention GMAC. COULD NOT VERIFY the claim that UVAS originated there; present it as the book's account (the paper itself is A16).

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__UNIFIED.html
- https://developer.nvidia.com/blog/unified-memory-in-cuda-6/

### A6. Early GPUs had 32-bit addresses and a 4 GB limit; Kepler, "introduced in 2013", brought 64-bit virtual and at least 40-bit physical addresses and more than 4 GB (C.4, pp. 625–626)

**Verdict: WRONG** on the year and on Kepler being the first. The 32-bit starting point is confirmed.

- Year: NVIDIA launched the first Kepler GPUs on March 21, 2012 (press release, "NVIDIA today launched the first GPUs based on its next-generation Kepler™ graphics architecture"). NVIDIA's GTC 2012 talk "Inside Kepler" (PDF dated June 2012) presents GK110.
- 32-bit start, and 64-bit before Kepler: the Fermi whitepaper's summary table gives "Load/Store Address Width" as 32-bit for G80 and GT200 and 64-bit for Fermi; "The 40-bit unified address space supports a Terabyte of addressable memory, and the load/store ISA supports 64-bit addressing for future growth."
- More than 4 GB before Kepler: Fermi whitepaper, "supporting up to a total of 6 GB of GDDR5 DRAM memory".
- Host and device sharing pointer values before Kepler: the archived CUDA 11.0 guide, section 3.2.8: "When the application is run as a 64-bit process, a single address space is used for the host and all the devices of compute capability 2.0 and higher." Compute capability 2.0 is the sm_20 target that PTX generic addressing requires (A2); NVIDIA's legacy GPU list puts Tesla C2050/C2070 at 2.0 and Tesla K20 at 3.5.
- "At least 40-bit physical addresses" for Kepler: COULD NOT VERIFY; I found no current NVIDIA page that states a physical address width for any architecture.

For lessons: the 32-bit era ends with Fermi (2009 whitepaper, 2010 products), not Kepler; say "the book dates this to Kepler; NVIDIA's own Fermi documents show 64-bit addressing and 6 GB a generation earlier".

- https://nvidianews.nvidia.com/news/nvidia-launches-first-geforce-gpus-based-on-next-generation-kepler-architecture-6622805
- https://developer.download.nvidia.com/GTC/PDF/GTC2012/PresentationPDF/S0642-GTC2012-Inside-Kepler.pdf
- https://www.nvidia.com/content/PDF/fermi_white_papers/NVIDIA_Fermi_Compute_Architecture_Whitepaper.pdf (summary table, page 11; overview, page 7)
- https://docs.nvidia.com/cuda/archive/11.0/cuda-c-programming-guide/index.html (section 3.2.8)
- https://developer.nvidia.com/cuda/gpus/legacy

### A7. "Modern CPUs use 64-bit virtual addresses with 48 bits actually used" (C.4, p. 626)

**Verdict: PARTLY.** It is the classic x86-64 layout and still Linux's default for user space, but CPUs and kernels now support wider addresses.

- Linux kernel documentation, "5-level paging": "Original x86-64 was limited by 4-level paging to 256 TiB of virtual address space and 64 TiB of physical address space." (256 TiB is 2^48 bytes.) 5-level paging "bumps the limits to 128 PiB of virtual address space and 4 PiB of physical address space."
- User space: "On x86, 5-level paging enables 56-bit userspace virtual address space." But "we are not going to allocate virtual address space above 47-bit by default. But userspace can ask for allocation from full address space by specifying hint address (with or without MAP_FIXED) above 47-bits."
- Whether CUDA's unified addressing or HMM handles user pointers above 47 bits: COULD NOT VERIFY; no NVIDIA page read here mentions it.

So the book's arithmetic for C.7 (49 bits covers a 48-bit CPU space plus GPU memory) is right for the default layout.

- https://docs.kernel.org/arch/x86/x86_64/5level-paging.html

### A8. A unified physical address space lets one GPU dereference a pointer into another GPU's memory on the same PCIe bus or NVLink domain; before Kepler this needed host-driven copies (C.5, p. 626)

**Verdict: PARTLY.** Direct peer loads and stores are confirmed. The book leaves out that access must be enabled first and has limits, and its "before Kepler" boundary is not supported by NVIDIA's own material.

- Guide 3.4.2.2: "Peer-to-peer memory access is supported between two devices if cudaDeviceCanAccessPeer() returns true for the specified devices." It "must be enabled between two devices by calling cudaDeviceEnablePeerAccess() ... On non-NVSwitch enabled systems, each device can support a system-wide maximum of eight peer connections."
- Reference: `cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)`; "access granted by this call is unidirectional". Runtime API 6.36.5: "The device pointer value through which any peer's memory may be accessed in the current device is the same pointer value through which that memory may be accessed from the peer device."
- Cost NVIDIA now points out: "Enabling peer access to a device via cudaDeviceEnablePeerAccess() adds runtime cost to device memory allocation operations on that peer ... A more scalable alternative ... is to make use of CUDA Virtual Memory Management APIs to explicitly allocate peer-accessible memory regions only as-needed" (see A15).
- Platform limit: "On Linux specifically, CUDA and the display driver do not support IOMMU-enabled bare-metal PCIe peer-to-peer memory transfer." Without peer access, copies still work: "copies must be performed using cudaMemcpyPeer(), cudaMemcpyPeerAsync()..."
- "Before Kepler": NVIDIA's CUDA 6 post says UVA (CUDA 4, 2011, before Kepler's 2012 launch) "enables pointers to be accessed from GPU code no matter where in the system they reside, whether its device memory (on the same or a different GPU), host memory, or on-chip shared memory." I found no current NVIDIA page that names the first architecture with peer access, so the exact boundary is COULD NOT VERIFY; "before Kepler" should not be taught as fact.

- https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html (sections 3.4.2.1 to 3.4.2.5)
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__PEER.html
- https://developer.nvidia.com/blog/unified-memory-in-cuda-6/

### A9. CUDA 6, 2013: Unified Memory, `cudaMallocManaged()`; the Fig. C.1 porting example (C.6, pp. 626–627)

**Verdict: PARTLY.** The feature and API are confirmed. The year is the announcement year; the toolkit shipped in 2014.

- NVIDIA blog "Unified Memory in CUDA 6", Nov 18, 2013: "CUDA 6 introduces Unified Memory" (announcement; the post asks readers to register for the release candidate). Toolkit archive: "CUDA Toolkit 6.0 (April 2014)". NVIDIA's Pascal post: "Unified Memory was introduced in 2014 with CUDA 6 and the Kepler architecture."
- The book's Fig. C.1 is NVIDIA's own example. The CUDA 6 post: "Both codes load a file from disk, sort the bytes in it, and then use the sorted data on the CPU ... The only differences are that the GPU version launches a kernel (and synchronizes after launching it), and allocates space for the loaded file in Unified Memory using the new API cudaMallocManaged()." NVIDIA's HMM post (Aug 22, 2023) reuses the same `sortfile` and shows it with plain `malloc` on HMM systems (A12).
- NVIDIA's own caveat at launch, which the book omits: "a carefully tuned CUDA program that uses streams and cudaMemcpyAsync to efficiently overlap execution with data transfers may very well perform better than a CUDA program that only uses Unified Memory." Today (ch02 A16): "Even with unified memory, optimal performance is attained by keeping the migration of memory to a minimum".
- Today's API: `__host__ cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags = 0x01)`; "flags must be one of cudaMemAttachGlobal or cudaMemAttachHost. The default value for flags is cudaMemAttachGlobal." "The memory is not cleared." Released with `cudaFree`. Two other ways to get managed memory: `__managed__` variables and `cudaMallocFromPoolAsync` with a pool of type `cudaMemAllocationTypeManaged` (guide 2.6.2).

- https://developer.nvidia.com/blog/unified-memory-in-cuda-6/
- https://developer.nvidia.com/cuda-toolkit-archive
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html

### A10. On Kepler and Maxwell, managed memory was flushed to the GPU before every launch, could not be touched by the CPU while the GPU ran, and was limited to GPU memory size (C.6, p. 627)

**Verdict: CONFIRMED**, and still the documented behaviour today wherever `cudaDevAttrConcurrentManagedAccess` is 0, which includes every Windows system.

- Guide 4.1.3 ("Unified Memory on Windows, WSL, and Tegra"): "Devices with compute capability lower than 6.0 or Windows platforms, devices with concurrentManagedAccess property set to 0 ... support CUDA managed memory with the following limitations: Data Migration and Coherency: Fine-grained movement of the managed data to GPU on-demand is not supported. Whenever a GPU kernel is launched all managed memory generally has to be transferred to GPU memory to avoid faulting on memory access. Page faulting is only supported from the CPU side. GPU Memory Oversubscription: They cannot allocate more managed memory than the physical size of GPU memory. Coherency and Concurrency: Simultaneous access to managed memory is not possible".
- Guide 4.1.3.2: "Concurrent CPU/GPU accesses, even to different managed memory allocations, will cause a segmentation fault". NVIDIA's example writes `y = 20;` on the host right after a launch: it "runs successfully on devices of compute capability 6.x due to the GPU page faulting capability ... but fails on pre-6.x architectures and Windows platforms".
- Guide 2.6.2.3 lists the same model for today's limited systems: "Managed memory is migrated to the GPU when the GPU begins executing", "The CPU must not access managed memory while the GPU is active", "Managed memory is migrated back to the CPU when the GPU is synchronized", "Oversubscription of GPU memory is not allowed". Which systems: "Windows, including Windows Subsystem for Linux (WSL), and ... some Tegra systems".
- Compute capability of the book's architectures (NVIDIA's legacy list): Kepler parts 3.x (Tesla K20 3.5, K80 3.7), Maxwell 5.x (Tesla M40 5.2), Pascal 6.0 (Tesla P100). These all sit on NVIDIA's "legacy" GPU page; the current guide's tables start at 7.5 (ch05 A5).

For the interactive: "CUDA 6 on Kepler/Maxwell" and "CUDA today on Windows" are the same mode. Fig. C.1's `cudaDeviceSynchronize()` before `use_data` is required in that mode.

- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html (sections 4.1.3, 4.1.3.2)
- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html (section 2.6.2.3)
- https://developer.nvidia.com/cuda/gpus/legacy

### A11. Pascal, 2016: 49-bit virtual addressing, GPU page faults, on-demand migration, oversubscription (C.7, C.8)

**Verdict: CONFIRMED.**

- Tesla P100 technical overview (NVIDIA, © 2016): "NVIDIA Pascal™ is the first GPU architecture to incorporate virtual memory paging and page faulting support in hardware — called Page Migration Engine." "Applications are permitted to oversubscribe the memory system: they can allocate, access, and share arrays larger than the total physical capacity of the system".
- NVIDIA Pascal post (Dec 14, 2016): "CUDA 8 and the Pascal architecture significantly improves Unified Memory functionality by adding 49-bit virtual addressing and on-demand page migration. The large 49-bit virtual addresses are sufficient to enable GPUs to access the entire system memory plus the memory of all GPUs in the system. The Page Migration engine allows GPU threads to fault on non-resident memory accesses". CUDA 8 is "Sept 2016" in the toolkit archive.
- Today (guide 2.6.2.2), for full unified memory support: "Managed memory is usually allocated in the memory space of the processor where it is first touched", "usually migrated when it is used by a processor other than the processor where it currently resides", "migrated or accessed at the granularity of memory pages (software coherence) or cache lines (hardware coherence)", "Oversubscription is allowed".
- Page sizes, which the book never gives (guide 4.1.1.2.1): "Currently, all x86_64 CPUs use a default physical page size of 4KiB. Arm CPUs support multiple physical page sizes - 4KiB, 16KiB, 32KiB and 64KiB - depending on the exact CPU. Finally, NVIDIA GPUs support multiple physical page sizes, but prefer 2MiB physical pages or larger. Note that these sizes are subject to change in future hardware." And: "TLB misses are generally significantly more expensive on the GPU compared to the CPU." Treat these as NVIDIA's current statements, not constants for a lab.
- What a software page fault does (guide 4.1.1.2.1.2): the owning processor loses its mapping, the requester gains one, and "The physical page backing this virtual page must be moved/migrated to the processor requesting access: this can be an expensive operation, and the amount of work is proportional to the page size."

- https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/nvidia-teslap100-techoverview.pdf
- https://developer.nvidia.com/blog/beyond-gpu-memory-limits-unified-memory-pascal/
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html (section 4.1.1.2.1)

### A12. From Pascal on, a GPU can follow host-built linked structures, and page faulting gives "zero-copy access to the entire host memory" (C.8, pp. 628–629; the notes' "Unclear in the PDF")

**Verdict: PARTLY.** This settles the notes' open question. Pascal-class page faulting alone covers memory allocated as managed. Ordinary `malloc`, `new` and `mmap` memory became GPU-accessible only with HMM (software) or ATS (hardware), which need much later hardware and software.

- Guide 4.1.2: "For devices with compute capability 6.x or higher but without pageable memory access ... CUDA managed memory is fully supported and coherent but the GPU cannot access system-allocated memory."
- Guide 2.6.2.2: "If cudaDevAttrPageableMemoryAccess is 0, then only memory allocated as managed memory by CUDA behaves as unified memory. Memory allocated with system APIs is not managed and is not necessarily accessible from GPU kernels." If it is 1, "all system memory, whether allocated by CUDA APIs or system APIs, operates as unified memory with full feature support. This includes file-backed memory allocations created with mmap."
- HMM requirements (guide 4.1.1): "Linux HMM requires Linux kernel version 6.1.24+, 6.2.11+ or 6.3+, devices with compute capability 7.5 or higher and a CUDA driver version 535+ installed with Open Kernel Modules." Check with `nvidia-smi -q | grep Addressing` ("Addressing Mode : HMM").
- ATS (guide 2.6.2.2.1): "On hardware such as Grace Hopper and Grace Blackwell, where an NVIDIA CPU is used and the interconnect between the CPU and GPU is NVLink Chip-to-Chip (C2C), address translation services (ATS) are available."
- NVIDIA's HMM post (Aug 22, 2023): "Previously, on PCIe-based machines, system allocated memory was not directly accessible by the GPU. The GPU could only access memory that came from special allocators such as cudaMalloc or cudaMallocManaged."
- `cudaMalloc` memory is still not host-accessible even with hardware coherence (guide 2.6.2.2.1 note): "Hardware coherency does not enable access to GPU-only allocations such as those made with cudaMalloc from the host."

For the interactive's "Pascal and later" era: a pointer from `cudaMallocManaged` can be chased on the GPU; a pointer from `malloc` cannot, until "HMM or ATS" (a separate era to add, 2023 for HMM per the post date).

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html (sections 4.1.1, 4.1.2)
- https://developer.nvidia.com/blog/simplifying-gpu-application-development-with-heterogeneous-memory-management/

### A13. Migrate or map: who decides? (C.8, p. 628; the notes' "Unclear in the PDF")

**Verdict: CONFIRMED** that both behaviours exist. The driver decides by heuristics; the programmer can steer it with hints added in CUDA 8.

- Reference, `cudaMallocManaged`: "The Unified Memory driver will employ heuristics to maintain data locality and prevent excessive page faults to the extent possible. The application can also guide the driver about memory usage patterns via cudaMemAdvise. The application can also explicitly migrate memory to a desired processor's memory via cudaMemPrefetchAsync."
- Mapping instead of migrating, by hint (Pascal post): "it is possible to pin regions to CPU memory and establish a direct mapping from the GPU by using a combination of cudaMemAdviseSetPreferredLocation and cudaMemAdviseSetAccessedBy usage hints". Guide 4.1.4.2 on `cudaMemAdviseSetAccessedBy`: "In some systems, it may be beneficial for performance to establish a mapping into memory before accessing the data from a given processor."
- Replication: `cudaMemAdviseSetReadMostly`, "In general, it allows trading off read bandwidth for write bandwidth on this region."
- Hints never change results: "Performance hints never impact the correctness of an application. Performance hints only affect performance."
- On hardware-coherent systems a third mechanism decides: "the access counters feature keeps track of the frequency of access that a GPU makes to memory located on other processors. This is needed to ensure memory pages are moved to the physical memory of the processor that is accessing the pages most frequently." "Per default cudaMallocManaged migrates based on a fault-and-migrate mechanism."

- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html (sections 4.1.1.2.8, 4.1.4)
- https://developer.nvidia.com/blog/beyond-gpu-memory-limits-unified-memory-pascal/

### A14. The managed-memory API a lab will call today: paradigm query, prefetch, advise

**Verdict: CONFIRMED**, with one trap: the signatures of `cudaMemPrefetchAsync` and `cudaMemAdvise` changed in CUDA 13.0.

Which paradigm a machine has (guide 2.6.2.1): query three attributes with `cudaDeviceGetAttribute`.

| Paradigm | Attributes |
|---|---|
| Limited (Windows, WSL, some Tegra) | `cudaDevAttrConcurrentManagedAccess` = 0 |
| Full, managed allocations only | `ConcurrentManagedAccess` = 1, `cudaDevAttrPageableMemoryAccess` = 0 |
| Full, all memory, software coherence (HMM) | both 1, `cudaDevAttrPageableMemoryAccessUsesHostPageTables` = 0 |
| Full, all memory, hardware coherence (ATS) | all three 1 |

Plus `cudaDevAttrManagedMemory` for "managed memory at all" and, on ATS systems, `cudaDevAttrDirectManagedMemAccessFromHost`: "Managed allocations resident on the GPU (e.g. cudaMallocManaged) can be accessed from the CPU without migration".

Signatures, CUDA 13.5 reference:

```cpp
cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count, struct cudaMemLocation location,
                                 unsigned int flags, cudaStream_t stream = 0);
cudaError_t cudaMemAdvise(const void *devPtr, size_t count, enum cudaMemoryAdvise advice,
                          struct cudaMemLocation location);
```

- The CUDA 12.9 reference has the old forms, `cudaMemPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream = 0)` and `cudaMemAdvise(..., int device)`, plus `cudaMemPrefetchAsync_v2` / `cudaMemAdvise_v2` taking `cudaMemLocation`. The CUDA 13.0.0 reference already shows only the `cudaMemLocation` forms under the plain names. HIPIFY's table marks both CUDA functions "C" (changed) in 13.0.
- The guide's example: `cudaMemLocation location = {.type = cudaMemLocationTypeDevice, .id = myGpuId};` and `{.type = cudaMemLocationTypeHost}` for the CPU.
- Prefetch semantics: "an asynchronous stream-ordered API that may migrate data ... The migration does not begin until all prior operations in the stream have completed, and completes before any subsequent operation in the stream." Reference: "this API is not required for functionality and only serves to improve performance". Device prefetch needs a device with nonzero `concurrentManagedAccess`.
- `cudaMemcpy` on managed memory is allowed (guide 4.1.1.2.5): "cudaMemcpy*() and cudaMemset*() accept any unified memory pointer as arguments", with the kind as a hint: "Prefer cudaMemcpyDefault over an inaccurate cudaMemcpyKind hint."

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html (section 2.6.2.1)
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html (sections 4.1.1.2.5, 4.1.4.1, 4.1.4.2)
- https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html
- https://docs.nvidia.com/cuda/archive/12.9.2/cuda-runtime-api/group__CUDART__MEMORY.html
- https://docs.nvidia.com/cuda/archive/13.0.0/cuda-runtime-api/group__CUDART__MEMORY.html
- https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Runtime_API_functions_supported_by_HIP.html

### A15. "CUDA 11 added" `cuMemAddressReserve()`, `cuMemCreate()` and `cuMemMap()` (C.9, p. 629)

**Verdict: WRONG** on the version, and the book's list is incomplete: a mapping is unusable until `cuMemSetAccess` grants access.

- NVIDIA blog "Introducing Low-Level GPU Virtual Memory Management" (Apr 15, 2020): "CUDA 10.2 introduces a new set of API functions for virtual memory management". Toolkit archive: "CUDA Toolkit 10.2 (Nov 2019)".
- Guide 4.18.3.4: "Mapping the allocation to a region of the address range using cuMemMap does not make the address accessible, and would result in a program crash if accessed by a CUDA kernel. Users must specifically select access control using the cuMemSetAccess function on source and accessing devices."
- Signatures (Driver API v13.5): `CUresult cuMemAddressReserve(CUdeviceptr *ptr, size_t size, size_t alignment, CUdeviceptr addr, unsigned long long flags)`; `CUresult cuMemCreate(CUmemGenericAllocationHandle *handle, size_t size, const CUmemAllocationProp *prop, unsigned long long flags)`; `CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset, CUmemGenericAllocationHandle handle, unsigned long long flags)`; `CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const CUmemAccessDesc *desc, size_t count)`. Teardown order: `cuMemUnmap`, `cuMemRelease`, `cuMemAddressFree`. Support query: `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED`. These are Driver API only ("VMM, being a very low-level API, requires use of the CUDA Driver API directly").
- The book's use case (one pointer spanning devices) is confirmed: "The reserved virtual address range can be mapped to memory chunks belonging to any device in the system, thus providing the application a continuous VA range backed and mapped by memory belonging to different devices."
- NVIDIA now names an LLM use: VMM enables "sophisticated memory allocators and cache management systems, such as dynamically managing key-value caches in large language models". It also lets an allocation grow "similar to how realloc or std::vector works in CPU memory management", and NVIDIA says "NVIDIA's libraries like NCCL and NVSHMEM actively use VMM." PyTorch and llama.cpp use it too (D2, D3).

- https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html
- https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__VA.html

### A16. Reference [1]: Gelado, Stone, Cabezas, Patel, Navarro, Hwu, ASPLOS 2010, pp. 347–358

**Verdict: PARTLY.** The paper exists as cited, but the published author order differs from the book's.

- University of Illinois research portal: authors "Isaac Gelado, Javier Cabezas, Nacho Navarro, John E. Stone, Sanjay Patel, Wen Mei W. Hwu"; "An asymmetric distributed shared memory model for heterogeneous parallel systems"; ASPLOS XV, 2010; pages 347–358; DOI 10.1145/1736020.1736059.

- https://experts.illinois.edu/en/publications/an-asymmetric-distributed-shared-memory-model-for-heterogeneous-p/
- https://doi.org/10.1145/1736020.1736059

### A17. Beyond the book: NVIDIA systems where CPU and GPU share one physical memory, or one page table

**Verdict: CONFIRMED.** NVIDIA documents two cases the appendix never mentions. They are the bridge to Apple silicon and AMD's MI300A, and they show that "one physical memory" does not make every buffer behave alike.

Tegra (one DRAM, NVIDIA's integrated GPUs):

- "In Tegra® devices, both the CPU (Host) and the iGPU share SoC DRAM memory." "In Tegra, device memory, host memory, and unified memory are allocated on the same physical SoC DRAM." "duplicate memory allocations and data transfers can be avoided."
- But: "Even though these memory buffer types are allocated on the same physical device, each has different accessing and caching behaviors". For example, pinned memory's CPU caching depends on "I/O coherency", which "is supported on Tegra devices starting with Xavier SOC".
- And unified memory there is the limited kind (A10): guide 2.6.2.3 lists "some Tegra systems" with limited support.

Grace Hopper and Grace Blackwell (two physical memories, one page table, hardware coherence):

- GH200 page: "With GH200, the CPU and GPU share a single per-process page table, enabling all CPU and GPU threads to access all system-allocated memory that can reside on physical CPU or GPU memory."
- Guide 4.1.1.2.1.2: hardware-coherent systems have "less page-faults" and "less contention: these systems are coherent at cache-line granularity instead of page-size granularity".

So the comparison the course needs has four shapes, not two: discrete GPU with software-managed migration (the book's Unified Memory), discrete memories with hardware coherence (Grace Hopper), one shared DRAM with per-buffer caching rules (Tegra, and Apple per ch05 C6), one shared HBM (MI300A, B8).

- https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html (sections 2, 3.1, 4.1)
- https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/
- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html (section 4.1.1.2.1.2)

### A18. Beyond the book: what happens when a CUDA program runs out of device memory on Windows

**Verdict: CONFIRMED** for Windows, from an NVIDIA knowledge-base article. It explains why "a model larger than VRAM" slows down instead of failing on some machines.

- NVIDIA KB 5490: "In driver 536.40, we implemented a new method to allow an application to use shared memory in cases that exhausted the GPU memory. This enabled applications which previously crashed when running out of GPU memory to continue to run, albeit at lower speeds. The switch to use shared memory occurs when running close to maxing out GPU memory to allow for a seamless transition." "In driver 546.01 and above we added a setting to disable the shared memory fallback". The setting is "CUDA - Sysmem Fallback Policy" in the NVIDIA Control Panel.
- The article is about Windows (Control Panel steps). I found no NVIDIA statement of an equivalent on Linux; do not claim one. On Linux the documented route to oversubscription is managed memory (A11), which is what llama.cpp's switch selects (D3).

- https://nvidia.custhelp.com/app/answers/detail/a_id/5490

### A19. Beyond the book: the opening's I/O-device problem (p. 623) has an NVIDIA answer

**Verdict: CONFIRMED.** The appendix motivates its history with disks and NICs that can only work on host memory, then never returns to them. NVIDIA's GPUDirect family is the current answer.

- "Using GPUDirect, network adapters and storage drives can directly read and write to/from GPU memory, eliminating unnecessary memory copies". GPUDirect Storage: "enables a direct data path between local or remote storage, such as NVMe or NVMe over Fabric (NVMe-oF), and GPU memory. It avoids extra copies through a bounce buffer in the CPU's memory". GPUDirect RDMA: "enables peripheral PCIe devices direct access to GPU memory." GPUDirect Peer to Peer: "Enables GPU-to-GPU copies as well as loads and stores directly over the memory fabric (PCIe, NVLink)."

Mention in a `<Beyond>` note only; the course does not use it.

- https://developer.nvidia.com/gpudirect

---

## Group B: HIP / ROCm counterparts

### B1. One generic address space for global, shared (LDS) and private memory (counterpart of C.1)

**Verdict: CONFIRMED.** AMD GPUs have the same arrangement as Fermi and later: a "flat" address space with apertures for LDS and scratch.

- LLVM AMDGPU backend documentation, "Address Spaces": "Only 64-bit process address spaces are supported for the amdgcn target." Table rows: Generic 0 (hardware "flat", 64-bit), Global 1 (64-bit), Local 3 (hardware "LDS", 32-bit), Private 5 (hardware "scratch", 32-bit), Constant 4.
- "The generic address space uses the hardware flat address support for two fixed ranges of virtual addresses (the private and local apertures), that are outside the range of addressable global memory, to map from a flat address to a private or local address. This uses FLAT instructions that can take a flat address and access global, private (scratch), and group (LDS) memory depending on if the address is within one of the aperture ranges."
- "The generic address space is supported unless the Target Properties column of AMDGPU Processors specifies Does not support generic address space."

So a HIP `__device__` helper taking `float*` accepts shared or global data, as in CUDA. HIP's own documentation does not discuss this; the source is the compiler's.

- https://llvm.org/docs/AMDGPUUsage.html (section "Address Spaces")

### B2. Zero-copy (mapped pinned) host memory: `hipHostMalloc`, `hipHostMallocMapped`, `hipHostGetDevicePointer` (counterpart of C.2)

**Verdict: PARTLY.** The calls exist with the CUDA shape (renames in infra B8). AMD adds a coherence choice CUDA does not have, and AMD's own pages disagree on its default.

- HIP host memory page: "Pinned memory or page-locked memory is stored in pages that are locked in specific sectors in RAM and can't be migrated. The pointer can be used on both host and device. Accessing host-resident pinned memory in device kernels is generally not recommended for performance, as it can force the data to traverse the host-device interconnect such as PCIe, which is much slower than the on-device bandwidth."
- "hipHostMallocMapped: The memory is allocated into the address space for the current device and the device pointer can be obtained with hipHostGetDevicePointer()." Header: `hipError_t hipHostGetDevicePointer(void** devPtr, void* hstPtr, unsigned int flags);`, "Get Device pointer from Host Pointer allocated through hipHostMalloc". HIPIFY maps `cudaHostGetDevicePointer` to `hipHostGetDevicePointer`.
- Coherence (HIP "Coherence control"): "Coarse-grained coherence: The memory is considered up-to-date only after synchronization performed using hipDeviceSynchronize(), hipStreamSynchronize(), or any blocking operation that acts on the null stream". "Fine-grained coherence: The memory is coherent even while being modified by a part of the system." "To achieve fine-grained coherence, many AMD GPUs use a limited cache policy, such as leaving these allocations uncached by the GPU or making them read-only." Flags `hipHostMallocCoherent` / `hipHostMallocNonCoherent`, environment variable `HIP_HOST_COHERENT`. "When hipHostMallocMapped flag is set, the allocated host memory is fine-grained and the hipHostMallocNonCoherent flag is ignored."
- The inconsistency: the same page's table lists `hipHostMalloc` with `hipHostMallocDefault` as "Fine-grained", while its footnote says "If neither these flags nor the HIP_HOST_COHERENT environment variable is set, or set as 0, the host memory allocation is coarse-grained." Not resolved here.

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/host_memory.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/coherence_control.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (line 5070)

### B3. Unified virtual addressing in HIP (counterpart of C.3)

**Verdict: CONFIRMED** that HIP presents one address space with `hipMemcpyDefault` and pointer queries, and goes further than CUDA on host access to device memory.

- `hipMemcpyDefault` = 4 (infra B8). `hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, const void* ptr);` (HIPIFY maps `cudaPointerGetAttributes` to it).
- HIP unified memory page: "Unified memory refers to the overall architectural concept of this shared address space, while managed memory is one specific implementation that provides automatic page migration between devices. Other unified memory allocators like hipMalloc() and hipHostMalloc() provide different access patterns within the same unified address space concept." And "CPUs can access memory allocated via hipMalloc(), providing bidirectional memory accessibility within the unified address space." (CUDA forbids this, ch02 A17 and A12 above.)
- One oddity: the header comments `hipDeviceAttributeUnifiedAddressing` as "Cuda only. An unified address space shared with the host." So do not use that attribute to test for unified addressing on AMD.

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (lines 572, 3968)

### B4. Peer GPU access (counterpart of C.5)

**Verdict: CONFIRMED.** Same calls and same model.

- Header: `hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int deviceId, int peerDeviceId);`, `hipError_t hipDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags);`.
- HIP multi-device page: "peer-to-peer memory access enables one GPU to directly read or write to the memory of another GPU." "If peer-to-peer access is not activated, the call to hipMemcpy() still works but internally uses a staging buffer in host memory, which incurs a performance penalty."

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/multi_device.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (lines 6490, 6505)

### B5. Managed memory, GPU page faults and XNACK (counterpart of C.6 to C.8; extends ch02 B7)

**Verdict: CONFIRMED.** AMD has the same three eras the book describes for NVIDIA, selected per GPU and per process rather than by date.

- Page-fault migration: "When a kernel on the device tries to access a managed memory address that is not in its local device memory, a page-fault is triggered. The GPU then in turn requests the page from the host or other device on which the memory is located." Oversubscription: "if the device's memory is already at capacity, a page is unmapped from the device's memory first and sent and mapped to host memory. This enables more memory to be allocated and used for a GPU than the GPU itself has physically available."
- The pre-Pascal mode, on AMD today: "Some AMD GPUs do not support page-faults, and thus do not support on-demand page-fault driven migration. On these architectures, if the programmer prefers all GPU memory accesses to be local, all pages have to be migrated before the kernel is dispatched". "managed memory APIs are still accessible to the programmer, but managed memory operates in a degraded fashion".
- The switch: "To ensure the proper functioning of managed memory on supported GPUs, it is essential to set the environment variable HSA_XNACK=1 and use a GPU kernel mode driver that supports HMM. Without this configuration, access-driven memory migration will be disabled". Header: "If HMM is not supported, the function behaves the same as hipMallocHost." Size note: "requested allocation size in bytes, it should be granularity of 4KB". Platforms: "The managed memory management APIs are implemented on Linux, under developement on Windows."
- Support table, as printed: `hipMallocManaged()` and `__managed__` on CDNA4, CDNA3, CDNA2, CDNA1, RDNA1, GCN5; system allocators (`new`, `malloc()`) only on CDNA2 to CDNA4, footnoted "Works only with HSA_XNACK=1 and kernels with HMM support. First GPU access causes recoverable page-fault." and "XNACK is not supported on CDNA2 (MI200 series) in SR-IOV configurations." Elsewhere the page says "Managed memory is supported on Linux by all modern AMD GPUs from the Vega series onward".
- Registering system memory: "Starting with CDNA2, the new, malloc(), and allocate() (Fortran) system allocators allow you to reserve unified memory." "Memory allocated by these allocators can be registered to be accessible on device using hipHostRegister()." The page does not say whether a kernel may then use the host pointer directly or needs `hipHostGetDevicePointer`.
- Attributes with CUDA's meanings: `hipDeviceAttributeManagedMemory`, `hipDeviceAttributeConcurrentManagedAccess` ("Full unified memory support. Device can coherently access managed memory concurrently with the CPU"), `hipDeviceAttributePageableMemoryAccess`, and in the header `hipDeviceAttributePageableMemoryAccessUsesHostPageTables`, `hipDeviceAttributeDirectManagedMemAccessFromHost`.
- XNACK per build: AMD's MI300A post: "XNACK describes the GPU's ability to retry memory accesses that failed due to a page fault". Code objects can be built `--offload-arch=gfx942:xnack+` (runs only with XNACK on) or `xnack-` (only with it off), or without the suffix (either).
- One AMD page describes CUDA's behaviour in a way NVIDIA does not: its CUDA table lists `cudaMalloc()` "with HMM" as "page-fault migration" for access outside the device. NVIDIA says hardware coherence "does not enable access to GPU-only allocations such as those made with cudaMalloc from the host" (A12). Trust NVIDIA's page for CUDA.

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/unified_memory.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (lines 4255 to 4282, 458 to 577)
- https://rocm.blogs.amd.com/software-tools-optimization/mi300a-programming/README.html

### B6. `hipMemPrefetchAsync` and `hipMemAdvise` take the CUDA 12 form, not the CUDA 13 form (counterpart of A14)

**Verdict: WRONG** for the assumption that a CUDA 13 prefetch or advise call carries over to HIP by renaming. With ROCm 10.2's default headers, the plain HIP names take an `int` device.

- Header at `therock-10.2`: `hipError_t hipMemPrefetchAsync(const void* dev_ptr, size_t count, int device, hipStream_t stream __dparm(0));` and, separately, `hipError_t hipMemPrefetchAsync_v2(const void* dev_ptr, size_t count, hipMemLocation location, unsigned int flags, hipStream_t stream __dparm(0));`.
- `hipMemAdvise` is versioned by `HIP_FORCE_API_VERSION`: below 800 it is `hipMemAdvise(const void* dev_ptr, size_t count, hipMemoryAdvise advice, int device)` (with `hipMemAdvise_v2` taking `hipMemLocation` from 700); at 800 and above the plain name takes `hipMemLocation`. `hip_common.h`: `#define HIP_API_VERSION (HIP_VERSION_MAJOR * 100 + HIP_VERSION_MINOR)` and `HIP_FORCE_API_VERSION` defaults to it; `projects/hip/VERSION` at the tag gives 7 and 17, so the default is 717 and the `int` form is what compiles.
- HIPIFY's table marks `cudaMemPrefetchAsync` and `cudaMemAdvise` as "U" (unsupported) for CUDA 13.0, and has no HIP entry for `cudaMemPrefetchAsync_v2` / `cudaMemAdvise_v2`.
- Names that do map one for one: `cudaMemLocation` to `hipMemLocation`, `cudaMemLocationTypeDevice` / `Host` to `hipMemLocationTypeDevice` / `Host`, `cudaCpuDeviceId` to `hipCpuDeviceId` (`((int)-1)`), `cudaMemAttachGlobal` to `hipMemAttachGlobal` (0x01), `cudaMallocManaged` to `hipMallocManaged`, `cudaDevAttrConcurrentManagedAccess` to `hipDeviceAttributeConcurrentManagedAccess`, `cudaDevAttrPageableMemoryAccess` to `hipDeviceAttributePageableMemoryAccess`.
- HIP-only advice: `hipMemAdviseSetCoarseGrain` (coherence, B2). AMD's docs use the `int` forms in their examples: `hipMemPrefetchAsync(a, sizeof(*a), deviceId, 0)`, `hipMemAdvise(a, sizeof(*a), hipMemAdviseSetPreferredLocation, deviceId)`.

For the lab: a `.cu` that prefetches must branch (`#if` on the HIP platform) or use `hipMemPrefetchAsync_v2` through a mapping; a plain `#define cudaMemPrefetchAsync hipMemPrefetchAsync` compiles the wrong overload set.

- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (lines 4295, 4310, 4440 to 4522, 1019, 1095)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_common.h (lines 76 to 80)
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/VERSION
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/driver_types.h (lines 437 to 453)
- https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Runtime_API_functions_supported_by_HIP.html

### B7. Virtual memory management in HIP (counterpart of C.9)

**Verdict: CONFIRMED.** Same four steps, runtime-API names, `void*` instead of `CUdeviceptr`.

- HIP VMM page: "when using virtual memory, this process is separated into multiple steps using the hipMemCreate(), hipMemAddressReserve(), hipMemMap(), and hipMemSetAccess() functions." "HIP virtual memory management is built on top of HSA". Support query `hipDeviceAttributeVirtualMemoryManagementSupported`. "If the allocation size is not aligned ... hipMemCreate() will return an out-of-memory error."
- Header: `hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t alignment, void* addr, unsigned long long flags);`, `hipError_t hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc, size_t count);`, each with "@note This API is implemented on Linux and is under development on Microsoft Windows."
- In practice llama.cpp turns it off on AMD by default (D3).

- https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management/virtual_memory.html
- https://github.com/ROCm/rocm-systems/blob/therock-10.2/projects/hip/include/hip/hip_runtime_api.h (lines 10272 to 10422)

### B8. AMD's one-memory GPU: MI300A (the HIP-side contrast to Apple's unified memory)

**Verdict: CONFIRMED** (AMD blog, Feb 9, 2025).

- "In MI300A - the AMD "Zen" 4 EPYC™ cores and third generation CDNA™ compute units share the same high-bandwidth memory (HBM)." "on the MI300A, separate host and device memory allocations are not required. All data resides in the HBM memory and is directly accessible to the GPU (XCD) and the CPU (CCD) cores."
- "With the shared HBM on MI300A, all types of memory allocations are pinned in the same memory space, allowing the data to be seamlessly accessed by both the host and device components. In this system, physical page migration is neither needed nor useful, however, handling page-faults via XNACK is still necessary. Thus, applications implementing the APU programming model and relying on unified memory must be compiled on an XNACK-enabled system."
- So on one physical memory AMD still needs page-fault handling to let the GPU use CPU-allocated pages (mappings, not migration). That is the point to carry to Metal (C5): sharing memory does not by itself share mappings.

- https://rocm.blogs.amd.com/software-tools-optimization/mi300a-programming/README.html

---

## Group C: Metal counterparts

Sources: MSL spec 4.1, Apple documentation JSON, Metal headers in the macOS 27.0 SDK.

### C1. No generic address space: MSL address spaces are disjoint (counterpart of C.1)

**Verdict: CONFIRMED** that Metal has no counterpart. MSL is where CUDA was before Fermi: a helper must say which memory its pointer is in.

- MSL spec section 4: "An address space attribute specifies the region of memory from where buffer memory objects are allocated. These attributes describe disjoint address spaces that can also specify access restrictions: device, constant, thread, threadgroup, threadgroup_imageblock, ray_data, object_data". "Any variable that is a pointer or reference needs to be declared with one of the address space attributes discussed in this section. If an address space attribute is missing on a pointer or reference type declaration, a compilation error occurs."
- The workaround is overloading (section 1.5.1): "Metal extends the function overloading rules to include the address space attribute of an argument." So one helper for `device float*` and another for `threadgroup float*`, which is the "separate implementation per memory type" the book says Fermi removed.
- Helpers taking `threadgroup` pointers: "The threadgroup attribute is supported here for the pointer l_data only if foo is called by a kernel function".
- **Observed, not documented** (2026-11-20, macOS 27.0.1, `Apple metal version 32023.921 (metalfe-32023.921.6)`). A helper whose pointer names no address space, `float rowMax(const float* p, uint n)`, called from a kernel on a `threadgroup` array, compiles at the compiler's default language version and with `-std=metal4.1`. With `-std=metal4.0`, and with every earlier `-std` the compiler offers for macOS (`macos-metal2.0` to `metal3.2`), the same file fails with "pointer type must have explicit address space qualifier". The two commands: `xcrun -sdk macosx metal -c probe.metal -o probe.air` (accepted) and `xcrun -sdk macosx metal -std=metal4.0 -c probe.metal -o probe.air` (rejected). A helper that takes only `device const float*`, called on a `threadgroup` array, fails with "no matching function for call" at every version, the default included. Run on the GPU (Apple silicon), the unqualified helper returned the right maxima from device, threadgroup and thread memory. This contradicts the section 4 sentence quoted above, and we found no Apple text describing it, so a lesson may mention it only as an observation and must still teach the overloads. Repeatable with `python3 labs/appC/snippets/check.py` (the probe is `labs/appC/snippets/check/c1_generic_probe.metal`).

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 1.5.1, 4)

### C2. Zero-copy and pinned memory on Metal (counterpart of C.2)

**Verdict: CONFIRMED.** On Apple silicon a shared-storage buffer is the default and needs no copy (ch05 C6); there is no pinned/pageable distinction (infra C12); wrapping existing page-aligned memory is `newBufferWithBytesNoCopy` (ch02 C2). The interconnect penalty the book attaches to zero-copy has no Metal counterpart because there is no interconnect between the CPU's and GPU's memory.

What Apple adds that matters for the lab (shared buffer used in place against an explicit copy into a second buffer):

- "Choosing a resource storage mode for Apple GPUs", a list of use cases. Under "Populate on CPU and access frequently on GPU": "Shared integrated memory for the CPU and GPU. Use shared." Under "Access exclusively on the GPU": "Data owned by the GPU. Use private. Choose the mode if you populate your resource with the GPU through a compute, render, or blit pass."
- `MTLStorageMode.private`: "Metal may apply additional optimizations to private resources that aren't allowed on shared or managed resources." Apple does not say what they are or whether they apply to buffers. So "copy into a private buffer, then run" against "run on the shared buffer" is a real open question to measure, not one the documentation answers.
- Header on `hasUnifiedMemory`: "Returns YES if this GPU shares its memory with the rest of the machine (CPU, etc.)". "Some GPU architectures do not have dedicated local memory and instead only use the same memory shared with the rest of the machine."

- https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus
- https://developer.apple.com/documentation/metal/mtlstoragemode/private
- https://developer.apple.com/documentation/metal/setting-resource-storage-modes
- Local header: `Metal.framework/Headers/MTLDevice.h` (macOS 27.0 SDK), lines 510 to 515

### C3. Does a shared buffer have one address for CPU and GPU? (counterpart of C.3)

**Verdict: COULD NOT VERIFY** from documentation. Apple documents a 64-bit GPU virtual address per buffer and says nothing about how it relates to the CPU pointer.

- Header: `typedef uint64_t MTLGPUAddress;` and on `MTLBuffer`: "@property gpuAddress @abstract Represents the GPU virtual address of a buffer resource", `@property (readonly) MTLGPUAddress gpuAddress API_AVAILABLE(macos(13.0), ios(16.0));`. Apple's web page for `gpuAddress` has no description text.
- `contents`: "Gets the system address of the buffer's storage allocation." (ch02 C2).
- Observed on this machine (macOS 27.0.1, not documented, one run of a probe in the scratch directory): for a 1 MiB shared buffer, `contents` and `gpuAddress` were different values; for a buffer made with `newBufferWithBytesNoCopy`, `contents` equalled the wrapped host pointer and `gpuAddress` was again a different value; a private buffer had `contents` NULL and a nonzero `gpuAddress`.

So on Apple silicon the bytes are shared but, as far as this probe shows, the CPU and the GPU use different virtual addresses for them, which is the CUDA 2.2 zero-copy picture (two pointer values for one piece of data), not the UVA picture. Metal never asks a program to pass a raw CPU pointer to a kernel (buffers are bound by object), so this costs nothing in practice. A lesson may show it only as the reader's own lab output, labelled as observed, not documented.

- https://developer.apple.com/documentation/metal/mtlbuffer/gpuaddress
- Local headers: `MTLGPUAddress.h`, `MTLBuffer.h` (lines 120 to 124), macOS 27.0 SDK

### C4. Peer GPUs (counterpart of C.5)

**Verdict: CONFIRMED** that Metal has no counterpart on Apple silicon. The Intel-Mac feature is deprecated.

- `MTLDevice.peerGroupID` (macOS 10.15): "A group ID value of 0 indicates the GPU isn't in a peer group." Deprecation note: "Not applicable on Apple Silicon".
- Header: `newRemoteBufferViewForDevice:`, "On Metal devices that support peer to peer transfers, this method is used to create a remote buffer view on another device within the peer group. The receiver must use MTLStorageModePrivate or be backed by an IOSurface." Marked `API_DEPRECATED("Not applicable on Apple Silicon", macos(10.15, 27.0))`. It was a copy path (a view used by blits), not kernel dereference of a peer pointer.

- https://developer.apple.com/documentation/metal/mtldevice/peergroupid
- Local header: `MTLBuffer.h` (lines 113 to 117), macOS 27.0 SDK

### C5. Managed memory, migration and GPU page faults (counterpart of C.6 to C.8)

**Verdict: CONFIRMED** that none is documented for Apple silicon. Metal's controls are about residency (what stays mapped for the GPU), set explicitly, not about migration on fault.

- Migration: nothing to migrate on one memory; `MTLStorageModeManaged` (two copies, explicit `didModifyRange`) is deprecated with "Managed storage has no effect on Apple Silicon" (ch05 C6).
- Page faults: the only mention in the SDK headers is an error code. `MTLCommandBufferErrorPageFault = 3`: "Execution of this command buffer generated an unserviceable GPU page fault. This can caused by buffer read write attribute mismatch or out of boundary access." Whether Apple GPUs service any faults (demand paging) is not documented: COULD NOT VERIFY, and do not teach either way.
- Residency sets (macOS 15.0): "Residency sets are a way you can tell Metal which resource allocations, such as buffers, textures, and heaps, to make resident, or GPU-accessible." With a set, an app can "Request that Metal make its resources resident ahead of time", "Keep allocations resident indefinitely", and remove allocations, "which Metal marks as candidates that it can make nonresident, if necessary". "residency sets don't track hazards, so you need to account for hazards with fences and events." "Every command buffer can maintain a list of up to 32 different residency sets."
- Budget: `recommendedMaxWorkingSetSize`, "An approximation of how much memory, in bytes, this GPU device can allocate without affecting its runtime performance." "You can help the GPU maintain its performance by keeping the total memory footprint of its resources and heaps less than this threshold value." `currentAllocatedSize`: "The total amount of memory, in bytes, the GPU device is using for all of its resources." Apple does not say what happens above the threshold; llama.cpp's code comment says "it's possible to allocate more than `recommendedMaxWorkingSetSize`" (D4).

This is the Mac answer to "why a model larger than memory behaves as it does": there is no oversubscription mechanism documented beyond the OS's own memory management, only a recommended working-set size and explicit residency. Read the reader's values from the device; never print one.

- https://developer.apple.com/documentation/metal/mtlresidencyset
- https://developer.apple.com/documentation/metal/simplifying-gpu-resource-management-with-residency-sets
- https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize
- https://developer.apple.com/documentation/metal/mtldevice/currentallocatedsize
- Local header: `MTLCommandBuffer.h` (lines 82 to 112), macOS 27.0 SDK

### C6. Reserving a range and mapping memory into it (counterpart of C.9)

**Verdict: PARTLY.** Metal 4 has a close analogue for a single device: a placement sparse buffer is a buffer whose pages you map from a placement heap. There is nothing spanning devices.

- Placement heaps (macOS 10.15): "The app controls placement of resources on the heap." "Use placement heaps when you need direct control over memory use and heap fragmentation."
- Header, `MTLDevice` (macOS 26.0): `newBufferWithLength:options:placementSparsePageSize:`, "Creates a new placement sparse buffer of a specific length. ... You assign memory to placement sparse buffers using a MTLHeap of type MTLHeapType/MTLHeapTypePlacement."
- Header, `MTL4CommandQueue` (macOS 26.0): `updateBufferMappings:heap:operations:count:`, "Updates multiple regions within a placement sparse buffer to alias specific tiles from a Metal heap." Each `MTL4UpdateSparseBufferMappingOperation` has a mode (map or unmap), "The range in the buffer, in tiles" and "The starting offset in the heap, in tiles".
- Page sizes: `MTLSparsePageSize16/64/256`, "Physical size of sparse resource page in KBs."
- Availability is inconsistent in the header: the buffer constructor and queue method are `macos(26.0)`, but `@property (readonly) BOOL supportsPlacementSparse` is `macos(26.4)`. Check `supportsPlacementSparse` at run time before using it.

Mapping to CUDA, ours: `newBufferWithLength:...placementSparsePageSize:` plays `cuMemAddressReserve`, a placement `MTLHeap` plays `cuMemCreate`, `updateBufferMappings` plays `cuMemMap`; there is no `cuMemSetAccess` step and no cross-device mapping. Not run here.

- https://developer.apple.com/documentation/metal/mtlheaptype/placement
- Local headers: `MTLDevice.h` (lines 1252, 1491 to 1506), `MTL4CommandQueue.h` (lines 180 to 207, 393 to 409), `MTLResource.h` (lines 190 to 199), macOS 27.0 SDK

---

## Group D: how PyTorch, llama.cpp and MLX use these mechanisms

Read from the default branches at the commits named at the top. Line numbers drift.

### D1. PyTorch: `pin_memory`, `DataLoader(pin_memory=True)`, `non_blocking=True`

**Verdict: CONFIRMED** (source and docs). This is what the appendix's pinned memory sits under.

- `aten/src/ATen/cuda/CachingHostAllocator.cpp` (around line 74): "// Use cudaHostAlloc for allocating pinned memory (global lock in driver)" then `C10_CUDA_CHECK(cudaHostAlloc(ptr, size, cudaHostAllocDefault));`; an opt-in path pins existing pages with `cudaHostRegister(*ptr, roundSize, cudaHostRegisterDefault)` instead.
- `torch/utils/data/dataloader.py`: "pin_memory (bool, optional): If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them."
- PyTorch CUDA notes ("Use pinned memory buffers"): "Host to GPU copies are much faster when they originate from pinned (page-locked) memory." "once you pin a tensor or storage, you can use asynchronous GPU copies. Just pass an additional non_blocking=True argument". "With non_blocking=True and pageable CPU memory, the call may block the host." "For a dense CUDA tensor, tensor.to("cpu", non_blocking=True) allocates its CPU output in pinned memory." Warning: "If you overuse pinned memory, it can cause serious problems when running low on RAM".
- The same notes on NVLink-C2C systems (A17): "On hardware-coherent NVLink-C2C systems, populated pageable and pinned buffers on the CPU NUMA node local to the source tensor's GPU can have similar transfer performance" and "The term "pinning" has two meanings. In CUDA, pinned memory cannot migrate from the CPU to the GPU or to another CPU NUMA node. In the Linux kernel, pinning refers to page-locking". "On hardware-coherent systems, pinned memory does not guarantee immovable physical pages."
- The copy itself is ch02 D4 (`cudaMemcpyAsync` in `Copy.cu`).

- https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CachingHostAllocator.cpp
- https://github.com/pytorch/pytorch/blob/main/torch/utils/data/dataloader.py
- https://github.com/pytorch/pytorch/blob/main/docs/source/notes/cuda.md (sections "Use pinned memory buffers" and "Device-to-host copies on NVLink-C2C systems")

### D2. PyTorch: `expandable_segments` is the C.9 API

**Verdict: CONFIRMED** (source).

- `c10/cuda/CUDACachingAllocator.cpp`, design comment: "The expandable_segments:True option is used to enable/disable this behavior. We use cuda's low-level memory APIs, which are similar to mmap, to extend the memory segments. These APIs separate the allocation of physical memory (cuMemCreate) from the allocation of virtual address space (cuMemAddressReserve) and the associate between them cuMemMap/cuMemSetAccess." "we allocate enough address space to map basically the entire physical memory of the GPU ... but we only map enough physical memory to handle the current amount of memory needed by the program." "This can work at the granularity of GPU pages which are 2MiB currently."
- Its stated limit, which ties back to A8: "CUDA runtime APIs related to sharing memory across process (cudaDeviceEnablePeerAccess) do not work for memory allocated with cuMemMap."
- Enabled through `PYTORCH_CUDA_ALLOC_CONF`; the notes call it "`expandable_segments` (experimental, default: `False`)".

- https://github.com/pytorch/pytorch/blob/main/c10/cuda/CUDACachingAllocator.cpp (comment block around lines 355 to 420; `cuMemAddressReserve_` call around line 549)

### D3. llama.cpp, CUDA and HIP: unified memory switch, VMM pool, pinned host buffers

**Verdict: CONFIRMED** (source and docs).

- Unified memory, `docs/build.md`: "The environment variable GGML_CUDA_ENABLE_UNIFIED_MEMORY=1 can be used to enable unified memory in Linux. This allows swapping to system RAM instead of crashing when the GPU VRAM is exhausted. In Windows this setting is available in the NVIDIA control panel as System Memory Fallback." (compare A18). For HIP: "On Linux it is possible to use unified memory architecture (UMA) to share main memory between the CPU and integrated GPU by setting environment variable GGML_CUDA_ENABLE_UNIFIED_MEMORY=1. However, this hurts performance for non-integrated GPUs".
- `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml_cuda_device_malloc` (around line 200): `err = cudaMallocManaged(ptr, size);`; under HIP it then calls `(void)cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device);` (the `int` device form, B6) with the comment "hipMemAdviseSetCoarseGrain is an optional performance hint", and falls back to `cudaMalloc` on `hipErrorNotSupported` "(e.g. on Windows)".
- VMM pool (around line 661): `struct ggml_cuda_pool_vmm` with `CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB` of reserved address space, filled with `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`. `ggml/CMakeLists.txt`: `option(GGML_CUDA_NO_VMM "ggml: do not try to use CUDA VMM" OFF)` but `option(GGML_HIP_NO_VMM "ggml: do not try to use HIP VMM" ON)`, so VMM is used on NVIDIA by default and not on AMD. `vendors/hip.h` maps `cuMemAddressReserve` to `hipMemAddressReserve`, and likewise `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`, `cuMemUnmap`, `cuMemRelease`, `cuMemAddressFree`.
- Pinned host buffers (around line 1777): `cudaMallocHost((void **) &ptr, size)` in `ggml_cuda_host_malloc`, disabled by `GGML_CUDA_NO_PINNED`; optional `cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly)` behind `GGML_CUDA_REGISTER_HOST`.
- Peer access: "The environment variable GGML_CUDA_P2P can be set to enable peer-to-peer access between multiple GPUs".

- https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md (sections "Unified Memory", "Peer Access")
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/ggml-cuda.cu
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/CMakeLists.txt (lines 205, 219)
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vendors/hip.h (lines 90 to 118)

### D4. llama.cpp, Metal: model weights are used in place from the mmapped file; residency sets keep them wired

**Verdict: CONFIRMED** (source). This is the zero-copy idea of C.2 applied to weights on one memory.

- `src/llama-model.cpp` (around line 2079): when the model is memory-mapped and the backend supports it, `ggml_backend_dev_buffer_from_host_ptr(dev, (char *) addr + range.first, ...)` wraps the mapped file range. On Metal that calls `ggml_metal_buffer_map` (`ggml-metal.cpp`, line 820), which page-aligns the pointer and creates `[res->dev->mtl_device newBufferWithBytesNoCopy:ptr length:size_aligned options:MTLResourceStorageModeShared deallocator:nil]`; if the range exceeds `max_buffer_size` it makes overlapping views so "the tensor with the maximum size will fully fit into one of the views" (`ggml-metal-device.m`, from line 2240).
- Residency sets: on by default where available (`dev->props.use_residency_sets = getenv("GGML_METAL_NO_RESIDENCY") == nil;`), with a background "heartbeat thread to keep the residency sets alive" and the comment "keep the residency sets wired for that amount of time to avoid being collected by the OS".
- Memory budget reported to the rest of llama.cpp: `*total = dev->mtl_device.recommendedMaxWorkingSetSize;`, with "// it's possible to allocate more than `recommendedMaxWorkingSetSize`".

- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal.cpp
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m

### D5. MLX: wired limit through residency sets; limits tied to the working-set size

**Verdict: CONFIRMED** (source).

- `mlx/memory.h`: "Set the wired size limit. Note, this function is only useful when using the Metal backend with macOS 15.0 or higher. The wired limit is the total size in bytes of memory that will be kept resident. The default value is 0. Setting a wired limit larger than system wired limit is an error." `MLX_API size_t set_wired_limit(size_t limit);`
- `mlx/backend/metal/resident.cpp` implements it with `MTL::ResidencySet` objects and reads `recommendedMaxWorkingSetSize()`; "Stay within the wired limit."
- `mlx/backend/metal/allocator.cpp`: `block_limit_ = std::min(1.5 * max_rec_size, 0.95 * memsize);` and `gc_limit_ = std::min(static_cast<size_t>(0.95 * max_rec_size), block_limit_);`, where `max_rec_size` is the device's recommended working-set size. These are MLX's policy, not Apple's.
- On CUDA, MLX uses managed memory for host-visible arrays (ch02 D3).

- https://github.com/ml-explore/mlx/blob/main/mlx/memory.h
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/resident.cpp
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/allocator.cpp

---

## The notes' dereference table, brought up to date

For the era-and-pointer interactive. "Today" columns are from the items above; the book's columns are in `notes/appC.md`.

| Pointer from | CUDA today, kernel | CUDA today, host | HIP (ROCm 10.2), kernel | HIP, host | Metal (Apple silicon) |
|---|---|---|---|---|---|
| `cudaMalloc` / `hipMalloc` | yes | no, on every paradigm including hardware coherence (A12) | yes | yes, as "zero copy" over the interconnect (ch02 B7) | private buffer: GPU only, `contents` is NULL (ch02 C2) |
| `cudaMallocHost` / `cudaHostAlloc` / `hipHostMalloc` | yes, same pointer, over the interconnect (A3) | yes | yes, same pointer (B2) | yes | no pinned memory; shared buffer is the nearest thing (C2) |
| `malloc` + `cudaHostRegister` / `hipHostRegister` | through `cudaHostGetDevicePointer` unless `CanUseHostPointerForRegisteredMem` (A3) | yes | device-accessible after `hipHostRegister`; which pointer to use is not stated (B5) | yes | `newBufferWithBytesNoCopy` on page-aligned memory (ch02 C2) |
| plain `malloc` / `new` / `mmap` | only with HMM or ATS (`PageableMemoryAccess` = 1) (A12) | yes | only with HMM and `HSA_XNACK=1`, CDNA2 and later (B5) | yes | not GPU-visible until wrapped in a buffer |
| `cudaMallocManaged` / `hipMallocManaged` | yes; faults and migrates where `ConcurrentManagedAccess` = 1; bulk move at launch where 0 (A10, A11) | yes; not while the GPU runs where `ConcurrentManagedAccess` = 0 | yes; migrates only with XNACK, else behaves as pinned host memory (B5) | yes | no managed mode on Apple silicon; shared storage, no migration (ch05 C6) |
| a peer GPU's `cudaMalloc` | yes after `cudaDeviceEnablePeerAccess`, or VMM `cuMemSetAccess` (A8, A15) | no | yes after `hipDeviceEnablePeerAccess` (B4) | (as `hipMalloc`) | none on Apple silicon (C4) |

Costs the interactive may show qualitatively (no numbers): device-local access is the fast path everywhere; mapped host access pays the interconnect on discrete GPUs (A3, A4); a managed page's first touch on the other side pays a fault and a page move, proportional to page size (A11); on one physical memory nothing moves, but caching rules can differ by buffer type (A17, C2).

---

## Notes for the lab runtime (not vendor facts)

Recorded so the chapter build can plan; this file changes nothing in shared code.

- The CPU emulator (`labs/common/include/pmpp/emu/emu_host.h`) has synchronous stand-ins for `cudaMallocManaged` (both `void**` and `T**` forms, default `cudaMemAttachGlobal`), `cudaMallocHost` and `cudaHostAlloc`. It has no `cudaMemPrefetchAsync`, `cudaMemAdvise`, `cudaMemLocation`, `cudaHostGetDevicePointer` or `cudaHostRegister`, and its `cudaDeviceGetAttribute` accepts only `cudaDevAttrCooperativeLaunch`. A lab that queries `cudaDevAttrConcurrentManagedAccess` or prefetches needs those added, or must keep such calls out of the emulated path.
- `labs/common/include/pmpp/vendors/hip.h` maps `cudaMallocHost`, `cudaHostAlloc` and `cudaMemcpyDefault` but not `cudaMallocManaged`, `cudaMemAttachGlobal`, `cudaHostGetDevicePointer`, `cudaDevAttrConcurrentManagedAccess`, `cudaDevAttrPageableMemoryAccess`, `cudaMemLocation` or the prefetch and advise calls. B6 is the trap for the last two.
- Metal side of the lab: the explicit copy into a second buffer is a blit `copyFromBuffer:...` (ch02 C3; macOS offsets and size multiples of 4), and GPU time is per command buffer (ch05 C9), so put the copy and the kernel in separate command buffers if they are to be timed separately. Apple's guidance favours shared storage for CPU-filled, GPU-read data (C2); whether the private copy wins on repeated reads is the measurement.

---

## Summary of corrections

Every WRONG, PARTLY and COULD NOT VERIFY item, one line each, plus CONFIRMED items whose details a lesson must not get wrong (marked CONFIRMED):

- A2: Fermi's unified address space is confirmed (PTX 2.0, sm_20, generic addressing); "2009" is the whitepaper year, and NVIDIA says Fermi reached the market in March 2010.
- A3 (CONFIRMED, but the book's 2009 procedure is out of date): under unified addressing, `cudaMallocHost`/`cudaHostAlloc` pointers work directly in kernels with or without `cudaHostAllocMapped`; `cudaHostGetDevicePointer` is needed only for registered and write-combined memory; NVIDIA now calls zero-copy "mapped memory" and warns that atomics on it are not atomic from the host's view.
- A4: the "under 10% of global memory bandwidth" rule holds for PCIe (H100 SXM: 128 GB/s against 3.35 TB/s, our arithmetic) but not for NVLink-C2C (900 GB/s, about 18% on GH200 NVL2 figures).
- A5: UVA has documented exceptions (registered and write-combined host memory get a distinct device address). The GMAC origin of UVAS could not be verified from NVIDIA sources.
- A6 (WRONG): Kepler launched in March 2012, not 2013, and was not first: Fermi already had 64-bit load/store addressing, a 40-bit unified address space, up to 6 GB of DRAM, and UVA (compute capability 2.0 and higher). Kepler's "at least 40-bit physical" could not be verified.
- A7: 48-bit CPU virtual addresses are the 4-level-paging default; 5-level paging allows 57-bit addresses, though Linux keeps user space below 47 bits unless asked. CUDA's handling of wider pointers could not be verified.
- A8: peer access must be enabled (`cudaDeviceEnablePeerAccess`, one direction at a time, at most eight peers without NVSwitch, not with an IOMMU on bare-metal Linux); "before Kepler" is contradicted by NVIDIA's own account of UVA since CUDA 4.
- A9: CUDA 6 Unified Memory was announced in November 2013 and shipped in April 2014; NVIDIA itself says "introduced in 2014".
- A10 (CONFIRMED, worth teaching): the book's Kepler/Maxwell restrictions are today's documented behaviour on Windows, WSL and some Tegra systems (`cudaDevAttrConcurrentManagedAccess` = 0).
- A12: Pascal-class page faulting covers managed allocations only; the GPU reaches plain `malloc`/`mmap` memory only with HMM (Linux 6.1.24+, compute capability 7.5+, driver 535+ open modules) or ATS (Grace Hopper/Blackwell). `cudaMalloc` memory stays host-inaccessible even with hardware coherence.
- A14 (CONFIRMED, trap): `cudaMemPrefetchAsync` and `cudaMemAdvise` take `cudaMemLocation` (and prefetch a `flags` argument) since CUDA 13.0; the `int device` forms are gone.
- A15 (WRONG): the VMM APIs arrived in CUDA 10.2 (Nov 2019), not CUDA 11; and `cuMemMap` alone does not make memory accessible: `cuMemSetAccess` is required.
- A16: the GMAC paper's published author order is Gelado, Cabezas, Navarro, Stone, Patel, Hwu, not the book's.
- B2: AMD's coherence page contradicts itself on whether default `hipHostMalloc` memory is fine- or coarse-grained.
- B3 (CONFIRMED, differs from CUDA): HIP lets the CPU read `hipMalloc` memory (zero copy), which CUDA forbids; `hipDeviceAttributeUnifiedAddressing` is commented "Cuda only".
- B5 (CONFIRMED, one AMD page contradicts NVIDIA): an AMD page describes CUDA's `cudaMalloc` as page-fault-migratable to the host under HMM; NVIDIA's own guide says it is not.
- B6 (WRONG): with ROCm 10.2's default headers, `hipMemPrefetchAsync` and `hipMemAdvise` take an `int` device; the CUDA 13 forms correspond to `hipMemPrefetchAsync_v2` / `hipMemAdvise_v2`, and HIPIFY marks the CUDA 13 forms unsupported.
- C3 (COULD NOT VERIFY): Apple does not document whether a shared buffer's `gpuAddress` equals its `contents` pointer; one local run showed different values.
- C5 (CONFIRMED that none is documented): Apple documents no serviceable GPU page faults or demand paging; only an "unserviceable GPU page fault" error, residency sets and `recommendedMaxWorkingSetSize`.
- C6: Metal 4 placement sparse buffers plus placement heaps are the nearest counterpart of `cuMemAddressReserve`/`cuMemCreate`/`cuMemMap`, single-device only; the header gives the API as macOS 26.0 but `supportsPlacementSparse` as macOS 26.4.

Terminology to state in the lesson, not a correction: CUDA's "Unified Memory" is "A CUDA feature that enables managed memory which can be automatically migrated between CPU and GPUs" (guide 2.6), a software-managed pool over separate memories; Apple's "unified memory model" is "the CPU and the GPU share system memory" (ch05 C6), one physical memory with no migration. NVIDIA's Tegra and AMD's MI300A are the two vendor cases of Apple's arrangement, and both still distinguish buffer types (A17, B8).

## Where the backends really differ

One line per cell. Nothing here was run on CUDA or AMD hardware.

| Feature | CUDA (13.5) | HIP (ROCm 10.2) | Metal (MSL 4.1, macOS 27) |
|---|---|---|---|
| One pointer type for global, shared and local memory | Generic addressing since sm_20 (Fermi) | Generic "flat" address space (LLVM AMDGPU) | None: address spaces are disjoint; overload helpers per space |
| Physical memories | Separate CPU and GPU DRAM (Tegra: one shared DRAM; Grace Hopper: two, one page table) | Separate on discrete GPUs; one shared HBM on MI300A | One system memory on Apple silicon |
| Pinned host memory | `cudaMallocHost`, `cudaHostAlloc`, `cudaHostRegister` | `hipHostMalloc`, `hipHostRegister`, plus fine/coarse coherence flags | No pinned/pageable distinction documented |
| Kernel reads host memory in place | Mapped (pinned) memory, same pointer under UVA; any memory with HMM/ATS | Pinned memory, same pointer; any memory with HMM and `HSA_XNACK=1` | Shared buffer, by binding the buffer; no raw CPU pointer in kernels |
| Host reads device memory | Not allowed for `cudaMalloc` memory | Allowed as slow "zero copy" for `hipMalloc` memory | Shared buffers yes; private buffers no |
| One virtual address for CPU and GPU | Yes (UVA, 64-bit processes), except registered and write-combined memory | Yes (one address space; `hipMemcpyDefault`) | Not documented; `contents` and `gpuAddress` differed in one local run |
| Copy direction inferred from pointers | `cudaMemcpyDefault` | `hipMemcpyDefault` | Not needed (no host-device copy for shared buffers) |
| Managed allocation | `cudaMallocManaged`, `__managed__`, managed pools | `hipMallocManaged`, `__managed__` (Linux; Windows "under development") | None; `MTLStorageModeManaged` deprecated on Apple silicon |
| Demand paging on GPU access | Compute capability 6.0 and later on Linux; not on Windows/WSL (bulk move at launch) | Only with XNACK enabled (`HSA_XNACK=1`) and an HMM driver | Not documented; only an "unserviceable GPU page fault" error |
| Oversubscribing GPU memory | Managed memory where `ConcurrentManagedAccess` = 1; Windows driver "Sysmem Fallback" for ordinary allocations | Managed memory with XNACK | No mechanism documented; `recommendedMaxWorkingSetSize` is advice |
| Placement hints | `cudaMemAdvise`, `cudaMemPrefetchAsync` with `cudaMemLocation` (CUDA 13) | `hipMemAdvise`, `hipMemPrefetchAsync` with `int` device by default; `_v2` take `hipMemLocation` | Residency sets (macOS 15), explicit, no migration |
| Peer GPU dereference | `cudaDeviceEnablePeerAccess`, or VMM `cuMemSetAccess` | `hipDeviceEnablePeerAccess` | None on Apple silicon (peer groups deprecated) |
| Reserve address range, map memory later | Driver API `cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, `cuMemSetAccess` (CUDA 10.2) | `hipMemAddressReserve`, `hipMemCreate`, `hipMemMap`, `hipMemSetAccess` | Placement sparse buffer plus placement heap, `updateBufferMappings` (Metal 4, one device) |
| What LLM runtimes use | PyTorch: `cudaHostAlloc` for pinned, VMM for `expandable_segments`; llama.cpp: VMM pool, optional `cudaMallocManaged` | llama.cpp: same source, VMM off by default, coarse-grain advice on managed memory | llama.cpp: mmapped weights wrapped with `newBufferWithBytesNoCopy`, residency sets; MLX: shared buffers, wired limit via residency sets |
