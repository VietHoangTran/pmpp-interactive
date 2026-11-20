# Chapter 1 fact check: the book's dated claims, and where HIP and Metal fit

Checked on 2026-11-20 against the live sources listed under each item.

How to read this file:

- **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY. Where an item has several sub-claims, the verdict is for the item as a whole and the text says which sub-claim failed.
- **Quote** is copied from the source (HTML flattened to text, GitHub raw files, Apple's documentation JSON, PDF text). Line breaks inside quotes are removed.
- Chapter 1 has no code, so this file is mostly group A: does each dated or device-specific statement in the chapter and the Preface still hold. Groups B and C are short because the lesson needs one sentence on each backend, and `notes/today/ch05-facts.md` already settles the API-level facts. Items from that file are cited as "ch05 A8", "ch05 B4" and so on, and are not repeated.
- `notes/today/infra-facts.md` did not exist on the check date.
- What the book says is taken from `notes/ch01.md` only, with its page numbers.
- **Device figures in this file are evidence for a verdict, not lesson content.** The project rule stands: a lesson shows the H100 numbers only as the book's example and shows no other device numbers.
- Versions the sources reported about themselves on the check date: CUDA Programming Guide "v13.5" (last updated Oct 31, 2026); CUDA Toolkit 13.5.0 (November 2026); ROCm Core SDK 10.2.0 with HIP 7.17.0 docs; OpenMP API 6.1 (Nov 2026); OpenCL 3.1; MPI 5.0; TOP500 68th edition (November 2026); PyTorch docs 2.15; llama.cpp `master` at `b94959a64d` (2026-11-19); MLX `main` at `8f0f391ec2` (2026-11-20); PyTorch `main` at `538adf4519` (2026-11-20); Intel "APP Metrics for Intel Microprocessors, Intel Xeon Processor" Revision 9 (PDF footer date 06/15/2026).
- Not checked: anything by running code on hardware. Every statement is from documentation, vendor pages or project source.
- Four pages refuse plain HTTP clients and were read through a fetch tool that returns extracted text: Intel's three product pages (A4), NVIDIA's investor release (A5) and Elsevier's 4th-edition page (A15). Field values from those are reliable; I could not inspect the surrounding page text myself. Each is marked "(extracted)".

Totals: 17 CONFIRMED, 0 WRONG, 8 PARTLY, 3 COULD NOT VERIFY (28 items).

---

## Already settled in ch05-facts.md (cite, do not re-check)

| Topic chapter 1 touches | Where it is settled |
|---|---|
| H100 FP32 67 teraFLOPS, 3.35 TB/s, 2048 resident threads per SM, the "with sparsity" footnote on tensor-core rows | ch05 A8 |
| Thread block clusters and distributed shared memory (Preface, p. xx) | ch05 A6 |
| The book's reference [4], "CUDA C++ Programming Guide, Release 13.0": the old URL is gone and the guide was restructured | ch05 A9 (side finding) |
| Latest ROCm release and HIP doc version | ch05 B1 |
| HIP spells kernels, qualifiers, built-ins and `<<<>>>` the CUDA way | ch05 B3, B4 |
| HIP has no dynamic parallelism | ch05 B4 |
| Warp width differs on AMD | ch05 B5 |
| How llama.cpp builds its HIP backend from the CUDA sources | ch05 B9 |
| Apple silicon unified memory, shared storage mode, no copy | ch05 C6 |
| How MLX and llama.cpp hold their Metal kernels | ch05 C10 |

---

## Group A: the book's dated or device-specific statements

### A1. H100 peak throughput "as of 2022": 34, 67 and 1979 TFLOPS for 64-, 32- and 16-bit (p. 3)

**Verdict: PARTLY.** All three numbers are on NVIDIA's H100 page today, but they are not three points on one scale. 34 and 67 are the plain FP64 and FP32 rows. 1,979 is the "FP16 Tensor Core" row, and that row carries the footnote "With sparsity". The chapter presents all three as peak rates by precision and gives no explanation for the jump (notes, "Unclear in the PDF", last bullet). This is the explanation.

Product Specifications table, H100 SXM column then H100 NVL column, as printed:

| Row | H100 SXM | H100 NVL |
|---|---|---|
| FP64 | 34 teraFLOPS | 30 teraFLOPs |
| FP64 Tensor Core | 67 teraFLOPS | 60 teraFLOPs |
| FP32 | 67 teraFLOPS | 60 teraFLOPs |
| TF32 Tensor Core * | 989 teraFLOPS | 835 teraFLOPs |
| BFLOAT16 Tensor Core * | 1,979 teraFLOPS | 1,671 teraFLOPS |
| FP16 Tensor Core * | 1,979 teraFLOPS | 1,671 teraFLOPS |
| FP8 Tensor Core * | 3,958 teraFLOPS | 3,341 teraFLOPS |

Footnote on the same page: "* With sparsity".

What follows for the lesson:

- The book's numbers are the SXM variant. Say "the book's example" and name the variant if the number is shown.
- 67 to 1,979 is not what halving the precision buys. It compares ordinary arithmetic units with tensor cores (matrix-multiply units), and the tensor-core figure assumes the sparsity feature. NVIDIA's product page does not print a dense FP16 tensor-core figure (ch05 A8 has the detail and the preliminary blog numbers).
- NVIDIA's earlier Hopper blog, which marks its table as preliminary, has a non-tensor "Peak FP16 1 120 TFLOPS" row for H100 SXM5 beside "Peak FP64 1 30 TFLOPS" and "Peak FP32 1 60 TFLOPS". That is the 2x-per-step pattern a reader would expect. The numbers are preliminary and differ from the product page, so use the pattern, not the values.
- "As of 2022" is right for the H100 as a product generation (the 4th edition is dated May 28, 2022, see A15). I did not look up the H100 launch date.

- https://www.nvidia.com/en-us/data-center/h100/ (Product Specifications table)
- https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/

### A2. H100 has "hundreds of thousands of threads" (p. 3)

**Verdict: CONFIRMED,** by arithmetic on two NVIDIA figures. The multiplication is mine.

- SM count, Hopper blog: "The NVIDIA H100 GPU with SXM5 board form-factor includes the following units: 8 GPCs, 66 TPCs, 2 SMs/TPC, 132 SMs per GPU". The full GH100 design is "144 SMs per full GPU"; the PCIe card has 114.
- Resident threads per SM for compute capability 9.0: 2048 (ch05 A8).
- 132 x 2048 = 270,336 resident threads.

The blog also says "128 FP32 CUDA Cores per SM, 16896 FP32 CUDA Cores per GPU". So the hundreds of thousands are threads the hardware keeps resident and switches between, not arithmetic units all executing at once. That distinction is the latency-tolerance point of pp. 4 to 5 and is worth one sentence.

- https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/

### A3. A server CPU of the same year peaks at "a few TFLOPS" (p. 3)

**Verdict: CONFIRMED** from Intel's export-compliance metrics, which list a GFLOPS value per processor.

Rows as printed in the PDF (columns: processor, number, GFLOPS, APP):

- "Intel® Xeon® Platinum 8380 Processor (60M Cache, 2.30 GHz) 8380 2304 0.6912"
- "Intel® Xeon® Platinum 8480+ Processor (105M Cache, 2.00 GHz) 8480+ 2329.6 0.69888"
- "Intel® Xeon® Platinum 8490H Processor (112.5M Cache, 1.90 GHz) 8490H 2304 0.6912"
- "Intel® Xeon® 6980P processor (504M Cache, 2.00 GHz) 6980P 6144 1.8432"

So 2.3 TFLOPS for the Platinum 8380, 8480+ and 8490H, and 6.1 TFLOPS for the largest value I found in the document (several Xeon 6900P parts). "A few TFLOPS" holds.

Caveats:

- The document says what the metric is for, "Giga Floating Point operations per Second (GFLOPS) and Adjusted Peak Performance (APP) in Weighted Teraflops (WT)", for "an export assessment against the US Export Administration Regulations". It does not state the precision or the method inside the PDF, and Intel adds "Intel makes no representation or warranty as to the accuracy or reliability of such specifications". Treat it as an order of magnitude.
- I did not look up the launch years of the 8380, 8480+ or 8490H. The 6980P is "Launch Date: Q3'24" (A4).
- AMD: I did not find an equivalent published FLOPS list for EPYC and did not search further.

- https://www.intel.com/content/www/us/en/support/articles/000005755/processors.html
- https://cdrdv2.intel.com/v1/dl/getContent/840270 (the PDF, 23 pages)

### A4. "Intel and AMD server processors with over 100 cores", hyper-threading with two hardware threads, "ARM Ampere server processor with up to 128 cores" (p. 3)

**Verdict: PARTLY.** The core counts are confirmed and have since been passed. Two hardware threads per core is true of some of these parts and false of others, including the highest core counts.

| Vendor source | What it says |
|---|---|
| AMD EPYC 9965 spec page | "# of CPU Cores 192", "# of Threads 384", "Launch Date 10/10/2024" |
| AMD EPYC 9006 series page | "AMD EPYC™ 9006 Series Server CPUs combine up to 256 cores, 512 threads"; footnote "As of July 2026, 6th Gen EPYC 9996 has 256 cores and 512 threads with SMT enabled"; "Simultaneous multithreading (SMT) lets each core run two threads at once" |
| Intel Xeon 6980P spec page (extracted) | "Total Cores: 128", "Total Threads: 256", "Launch Date: Q3'24" |
| Intel Xeon 6+ series page (extracted) | "up to 288 Efficient-cores (E-cores) per socket"; top part "Intel® Xeon® 6990E+ processor (576M Cache, 2.20 GHz)", 288 total cores, launch "Q2'26" |
| Intel Xeon 6710E spec page (extracted) | "Total Cores: 64", "Total Threads: 64"; the page does not list Hyper-Threading |
| Ampere processors page | "Ampere Altra® ... 32 - 128 Cores"; "AmpereOne® ... 96 - 192 Cores"; "Ampere's single-threaded, highly parallel cores" |

Notes:

- The book's "up to 128 cores" is Ampere's Altra line. AmpereOne goes to 192.
- I confirmed threads equal cores on one Intel E-core part (6710E). The Xeon 6+ page gave no thread count for the 288-core part, so "E-core Xeons have one thread per core" is confirmed for that one SKU only.
- For the lesson: "over 100 cores" is safe and conservative. Do not state "two hardware threads per core" as a general property of server CPUs. The chapter's contrast (a few hundred heavyweight threads against hundreds of thousands of lightweight ones) is unchanged.

- https://www.amd.com/en/products/processors/server/epyc/9005-series/amd-epyc-9965.html
- https://www.amd.com/en/products/processors/server/epyc/9006-series.html
- https://www.intel.com/content/www/us/en/products/sku/240777/intel-xeon-6980p-processor-504m-cache-2-00-ghz/specifications.html
- https://www.intel.com/content/www/us/en/products/details/processors/xeon/6-plus-series.html
- https://www.intel.com/content/www/us/en/products/sku/240363/intel-xeon-6710e-processor-96m-cache-2-40-ghz/specifications.html
- https://amperecomputing.com/products/processors

### A5. Since the 4th edition: Ada Lovelace, Hopper and Blackwell; "CUDA 11 through CUDA 13" (pp. xvii to xviii)

**Verdict: CONFIRMED,** and there is now a generation after the three the Preface names.

CUDA versions, from NVIDIA's toolkit archive: "CUDA Toolkit 11.0.1 (June 2020)", "CUDA Toolkit 11.7.0 (May 2022)" (the release current when the 4th edition appeared), "CUDA Toolkit 12.0.0 (December 2022)", "CUDA Toolkit 13.0.0 (August 2025)", and "Latest Release CUDA Toolkit 13.5.0 (November 2026)".

Generations, from NVIDIA's compute capability table (product names as printed; the architecture is in the product name):

| Compute capability | Products listed (excerpt) |
|---|---|
| 8.9 | "NVIDIA L4 NVIDIA L40 NVIDIA L40S NVIDIA RTX 6000 Ada ... GeForce RTX 4090" |
| 9.0 | "NVIDIA GH200 NVIDIA H200 NVIDIA H100" |
| 10.0 | "NVIDIA GB200 NVIDIA B200" |
| 10.3 | "NVIDIA GB300 NVIDIA B300 NVIDIA GB300 (DGX Station)" |
| 11.0 | "Jetson T5000 Jetson T4000" |
| 12.0 | "NVIDIA RTX PRO 6000 Blackwell Server Edition ... GeForce RTX 5090" |
| 12.1 | "NVIDIA GB10 (DGX Spark) NVIDIA RTX Spark N1X" |

Newer than the book:

- NVIDIA's results release dated "November 17, 2026" says (extracted) "Ramped production and began shipping NVIDIA Vera Rubin NVL72". An NVIDIA developer blog dated Mar 16, 2026 says "The NVIDIA Vera Rubin NVL72 rack integrates 72 Rubin GPUs and 36 Vera CPUs".
- The compute capability table has no Rubin entry on the check date; 12.1 is the highest row. I found no CUDA documentation of a Rubin compute capability and did not search beyond that table.

Small points: the pages I read use "Ada" in product names; I did not find the full name "Ada Lovelace" on them and did not search for it. "Hopper" is the title of NVIDIA's architecture blog.

- https://developer.nvidia.com/cuda-toolkit-archive
- https://developer.nvidia.com/cuda/gpus
- https://investor.nvidia.com/news/press-release-details/2026/NVIDIA-Announces-Financial-Results-for-Third-Quarter-Fiscal-2027/
- https://developer.nvidia.com/blog/nvidia-vera-rubin-pod-seven-chips-five-rack-scale-systems-one-ai-supercomputer/

### A6. CUDA "introduced by NVIDIA in 2007", starting with the G80 (pp. 5 to 7)

**Verdict: PARTLY.** NVIDIA's own current documents date CUDA's introduction to 2006. The first numbered toolkit is dated June 2007. I could not find the G80 named on a current NVIDIA page.

- CUDA Programming Guide v13.5, section 1.1.1: "In 2006, NVIDIA introduced the Compute Unified Device Architecture (CUDA) to enable any computational workload to use the throughput capability of GPUs independent of graphics APIs."
- NVIDIA corporate timeline: "2006 CUDA Opens parallel processing capabilities of GPUs to science and research with unveiling of CUDA ® architecture."
- GTC 2026 keynote transcript (16 March 2026): "This is the 20th anniversary of CUDA."
- Toolkit archive, oldest entry: "CUDA Toolkit 1.0 (June 2007)". The docs footer reads "Copyright © 2007-2026, NVIDIA Corporation".
- G80: no match on the timeline page, the guide's introduction or the archive. COULD NOT VERIFY from current NVIDIA sources; it is the book's statement.

Safe statement for the lesson: NVIDIA unveiled CUDA in 2006 and shipped toolkit 1.0 in 2007; the book says 2007. The same guide section supports the book's history of what came before: "By 2003, some stages of the graphics pipeline became fully programmable, running custom code in parallel for each component of a 3D scene or an image."

- https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html
- https://www.nvidia.com/en-us/about-nvidia/corporate-timeline/
- https://www.nvidia.com/gtc/keynote/transcript.md
- https://developer.nvidia.com/cuda-toolkit-archive

### A7. "More than 1 billion CUDA enabled GPUs in use" and "millions" of CUDA programmers (pp. xvii, 6)

**Verdict: COULD NOT VERIFY** the billion. "Millions" of developers is confirmed.

- Installed base, GTC 2026 keynote transcript: "It has taken us 20 years to now have built up hundreds of millions of GPUs and computing systems around the world that run CUDA." That is NVIDIA's own current wording and it is smaller than the book's.
- Developers, NVIDIA corporate timeline page: "More than 4 million developers now create thousands of applications for accelerated computing."
- A web search also returned a third-party report of a larger developer count; I did not find it on an NVIDIA page and have not used it.

For the lesson: give the installed-base argument (p. 6) without a number, or attribute "more than a billion" to the book.

- https://www.nvidia.com/gtc/keynote/transcript.md
- https://www.nvidia.com/en-us/about-nvidia/corporate-timeline/

### A8. Graphics chips run at "approximately 10x the memory bandwidth" of contemporary CPUs (p. 5)

**Verdict: PARTLY.** The direction is confirmed by NVIDIA and by vendor spec pages. The published figures I found give a ratio of roughly 5 to 7 for one data-centre GPU against one server CPU socket, not 10, and the comparison does not carry over to unified-memory systems.

- NVIDIA, guide section 1.1.2: "A GPU provides much higher instruction throughput and memory bandwidth than a CPU within a similar price and power envelope."
- H100 SXM: "GPU Memory Bandwidth 3.35TB/s" (ch05 A8).
- AMD EPYC 9654: "Launch Date 11/10/2022", "Memory Channels 12", "Up to 4800 MT/s", "Per Socket Mem BW 460.8 GB/s". 3350 / 460.8 = 7.3.
- AMD EPYC 9965: "Launch Date 10/10/2024", "Per Socket Mem BW 614 GB/s". 3350 / 614 = 5.5.
- The divisions are mine. They compare one GPU with one socket; "per socket" figures are theoretical channel totals.

Other backends (the notes flag both as unverified):

- AMD GPUs: the same picture holds on AMD's own spec page. Instinct MI300X: "Dedicated Memory Type HBM3", "Peak Memory Bandwidth 5.3 TB/s".
- Apple silicon: there is no separate GPU memory to compare. Apple quotes one figure for the whole chip, for example "M5 Max supports up to 128GB of unified memory with higher unified memory bandwidth up to 614GB/s". The CPU and GPU draw on the same memory (ch05 C6), so no GPU-to-CPU bandwidth ratio is documented and the 10x rule has no meaning there.

For the lesson: keep "an order of magnitude" as the book's rule of thumb for discrete GPUs, and say in one sentence that on a unified-memory machine the GPU and CPU share one memory system.

- https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html
- https://www.amd.com/en/products/processors/server/epyc/4th-generation-9004-and-8004-series/amd-epyc-9654.html
- https://www.amd.com/en/products/processors/server/epyc/9005-series/amd-epyc-9965.html
- https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
- https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/

### A9. CPUs are latency-oriented, GPUs throughput-oriented (Fig. 1.1, pp. 4 to 5)

**Verdict: CONFIRMED** as NVIDIA's current framing, in nearly the book's terms.

Guide section 1.1.2: "GPUs and CPUs are designed with different goals in mind. While a CPU is designed to excel at executing a serial sequence of operations (called a thread) as fast as possible and can execute a few tens of these threads in parallel, a GPU is designed to excel at executing thousands of threads in parallel, trading off lower single-thread performance to achieve much greater total throughput. GPUs are specialized for highly parallel computations and devote more transistors to data processing units, while CPUs dedicate more transistors to data caching and flow control." The guide's Figure 1 is captioned "The GPU Devotes More Transistors to Data Processing".

Not confirmed anywhere: the book's cost rule (doubling throughput costs 2x area and power; halving latency may cost more than 2x area and 4x power, p. 5). No vendor source states it. It is the book's argument and any interactive built on it must say so.

Also in the same section, matching p. 7: "Other computing devices, like FPGAs, are also very energy efficient, but offer much less programming flexibility than GPUs."

- https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html

### A10. "Exascale computing to clusters" (p. 10)

**Verdict: CONFIRMED.** The November 2026 TOP500 list has five systems above one exaflop on HPL.

Quote: "On the 68th edition of the TOP500, LineShine remains the new No. 1 system. LineShine entered the list in June 2026 and ended El Capitan's run atop the list becoming the fifth Exascale system overall." and "El Capitan, Frontier, Aurora, and JUPITER Booster all remain Exascale-class systems and occupy No. 2 through No. 5".

The list's own text supports the chapter's "use both" point for some of the five and not all. El Capitan uses "AMD 4th generation EPYC processors ... and AMD Instinct MI300A accelerators". Aurora uses "Intel Xeon CPU Max Series processors and Intel Data Center GPU Max Series accelerators". JUPITER is described as "utilizing Grace Hopper Superchips". Frontier's paragraph names only its EPYC processors, and the No. 1 system is described as based on "304-core LX2 processors" with no accelerator named. I did not open the per-system pages, so do not write "all exascale machines are CPU plus GPU".

"Terascale computing to the desktops": not checked separately. A3 shows single server CPUs above 1 TFLOPS.

- https://top500.org/lists/top500/2026/11/

### A11. OpenCL "developed jointly in 2009 by Apple, Intel, AMD/ATI and NVIDIA" (p. 13)

**Verdict: PARTLY.** Khronos ratified OpenCL 1.0 in December 2008, which agrees with the book's own reference [9] and not with its text. The four companies are among a longer list. OpenCL is still maintained, and Apple has deprecated it.

- Khronos press release: "December 9th 2008 – SIGGRAPH Asia, Singapore – The Khronos™ Group today announced the ratification and public release of the OpenCL™ 1.0 specification". "Proposed six months ago as a draft specification by Apple, OpenCL has been developed and ratified by industry-leading companies including 3DLABS, Activision Blizzard, AMD, Apple, ARM, ... IBM, Intel Corporation, ... NVIDIA, ... Texas Instruments and Umeå University."
- Current state, Khronos OpenCL page: "OpenCL 3.1 brings widely deployed, field-proven capabilities into the core specification to expand functionality, including SPIR-V ingestion". The same page lists "OpenCL 3.0.19 Released" as a maintenance update.
- Apple: "If you're using OpenCL, which was deprecated in macOS 10.14, for GPU-based computational tasks in your Mac app, we recommend that you transition to Metal and Metal Performance Shaders".
- Still in real use for LLM inference: llama.cpp has a `ggml-opencl` backend directory (D1).

The book's claim that OpenCL code runs unchanged but needs retuning per device (p. 13) is not something a vendor page confirms or denies. It is the book's.

- https://www.khronos.org/news/press/the_khronos_group_releases_opencl_1.0_specification
- https://www.khronos.org/opencl/
- https://developer.apple.com/opencl/

### A12. OpenMP was designed for CPUs and extended to GPUs (pp. 12 to 13)

**Verdict: CONFIRMED** for the history. "Compilers are still improving" is an opinion and was not checked.

- OpenMP specifications page: "TR1 – Technical Report on Directives for Attached Accelerators (November 2012) ... This Technical Report specifies proposed directives for the execution of loops and regions of code on attached accelerators. This document has been superseded by the ratification of OpenMP 4.0." and "OpenMP 4.0 Complete Specifications – Jul 2013".
- Current version: "OpenMP API 6.1 Specification – Nov 2026".

- https://www.openmp.org/specifications/

### A13. MPI, CUDA-aware MPI, NCCL, NVSHMEM; MPI on "more than 100,000 nodes" (p. 13)

**Verdict: PARTLY.** All four exist under these names and are current. The node count could not be verified.

- MPI Forum: "MPI-5.0 was approved by the MPI Forum on June 5, 2025."
- CUDA-aware MPI: the Open MPI documentation has a section "11.3.5. CUDA" with "Building Open MPI with CUDA-aware support" and "CUDA-aware MPI APIs". Its table of contents lists "11.3.6. ROCm" as the next section, so the AMD equivalent is documented in the same place. I did not read the ROCm section.
- NCCL: "The NVIDIA Collective Communications Library (NCCL, pronounced "Nickel") is a library providing inter-GPU communication primitives that are topology-aware and can be easily integrated into applications." and "It is not a full-blown parallel programming framework".
- NVSHMEM: the documentation is titled "NVIDIA OpenSHMEM Library (NVSHMEM) Documentation".
- "More than 100,000 nodes": COULD NOT VERIFY. TOP500 reports cores, not nodes (for example El Capitan "has 11,340,000 cores").

The project plan drops MPI, NCCL and NVSHMEM from chapter 23, so the lesson needs these as names only.

- https://www.mpi-forum.org/docs/
- https://docs.open-mpi.org/en/main/tuning-apps/accelerators/cuda.html
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html
- https://docs.nvidia.com/nvshmem/api/latest/index.html
- https://top500.org/lists/top500/2026/11/

### A14. Dynamic parallelism: dropped from the 5th edition for low adoption (p. xxii)

**Verdict: CONFIRMED** that the feature still exists in CUDA, so older material the reader meets online is not obsolete, only out of the book.

- Guide section 4.21, "CUDA Dynamic Parallelism": "a feature of the CUDA programming model that enables code running on the GPU to create new GPU work."
- "This section documents the updated version of CUDA Dynamic Parallelism, sometimes called CDP2, which is the default in CUDA 12.0 and later. CDP2 is the only version of CUDA dynamic parallelism available on devices of CC 9.0 and higher."
- No deprecation notice on that page.
- The book's adoption claim and the "Kepler, over a decade ago" date were not checked.
- HIP has no equivalent (ch05 B4). Metal: see C4.

- https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/dynamic-parallelism.html

### A15. Edition facts: 4th edition 2022, this is the 5th; what the reader will find online

**Verdict: PARTLY.** The editions and the 2022 date are confirmed. The publisher's own web page for the 5th edition carries a table of contents that does not match the printed book, and three different publication dates are in circulation.

Dates:

- 4th edition, Elsevier shop (extracted): "4th Edition - May 28, 2022", with the notice "Newer edition is available."
- 5th edition, Elsevier shop, page heading: "5th Edition - February 27, 2026". Same page, product details: "Edition: 5 Latest edition Published: June 3, 2026". Elsevier Educate: "Publication Date: 03 Jun 2026", "ISBN: 9780443439001", "Page Count : 680".
- The copyright line in the PDF reads 2027 (notes, "Dated or device-specific content").

The 5th-edition table of contents as posted on both Elsevier pages on the check date:

"1. Introduction Part I. Fundamental Concepts 2. Heterogeneous data parallel computing 3. Multidimensional grids and data 4. Compute architecture and scheduling 5. Memory architecture and data locality 6. Performance considerations Part II. Parallel Patterns 7. Convolution 8. Stencil 9. Parallel histogram 10. Reduction 11. Prefix sum (scan) 12. Merge Part III. Advanced Patterns and Applications 13. Sorting 14. Filtering (new) 15. Sparse matrix computation 16. Wavefront Algorithms (new) 17. Graph traversal 18. Deep learning 19. Multi-GPU API (new) 20. Electrostatic potential map 21. Parallel programming and computational thinking Part IV. Advanced Practices 22. Programming a heterogeneous computing cluster 23. Advanced Optimizations for Matrix Multiplication (new) 24. Advanced practices and future evolution 25. Conclusion and outlook"

That is a pre-publication outline. It has 25 chapters in four parts and no LLM chapter in the list. The printed book has 24 chapters in three parts, with Filtering as 12, matrix multiplication as 15 and LLMs as 20 (notes, Fig. P.1 and §1.7). The Preface's stray "four major parts" (notes, "Unclear in the PDF") looks like a survivor of the same outline. The same Elsevier page's feature list does match the printed book: "New pattern and application chapters including: filtering, wavefront parallelism, advanced optimizations for matrix multiplication, and large language models (LLMs)" and "warp-level programming, cooperative groups, CUDA C++ atomics, and multi-GPU programming with NCCL and NVSHMEM".

Chapter numbers moved between editions. The 4th-edition column is Elsevier's page (extracted); the 5th-edition column is `notes/ch01.md`. Pairing rows by title is mine.

| 4th edition (2022) | 5th edition |
|---|---|
| 1 to 11 (Introduction through Prefix sum) | 1 to 11, same numbers |
| none | 12 Filtering (new) |
| 12 Merge | 13 Merge |
| 13 Sorting | 14 Sort |
| none | 15 Advanced optimizations for matrix multiplication (new) |
| none | 16 Wavefront parallelism (new) |
| 14 Sparse matrix computation | 17 |
| 15 Graph traversal | 18 |
| 16 Deep learning | 19 Convolutional neural networks (background moved to Appendix B) |
| none | 20 Large language models (new) |
| 17 Iterative magnetic resonance imaging reconstruction | removed (parts folded into 21) |
| 18 Electrostatic potential map | 21 |
| 19 Parallel programming and computational thinking | 22 |
| 20 Programming a heterogeneous computing cluster | 23 (rewritten around CUDA-aware MPI, NCCL, NVSHMEM) |
| 21 CUDA dynamic parallelism | removed |
| 22 Advanced practices and future evolution | removed (memory-model history moved to Appendix C) |
| 23 Conclusion and outlook | 24 |
| Appendix A Numerical considerations | Appendix A (expanded), plus new Appendices B and C |

For the lesson: any course slide, blog post or answer that says "PMPP chapter 14" needs its edition checked. From chapter 12 onward the numbers differ.

Not checked: the companion site, the teaching kit and the university course pages named on pp. xxiv to xxvi.

- https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-443-43900-1
- https://educate.elsevier.com/book/details/9780443439001
- https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0

### A16. The Amdahl numbers on p. 9

**Verdict: CONFIRMED** by arithmetic. This is not a vendor fact; it is here so the explorer and its browser check assert the right values. Formula (ours, the book prints none): speedup = 1 / ((1 - p) + p / s).

| p (parallel fraction) | s (speedup of that part) | Time saved | Overall speedup | Book says |
|---|---|---|---|---|
| 0.30 | 100 | 29.7% | 1.4225 | "about 1.42" |
| 0.30 | unbounded | 30% | 1.4286 | "no more than 1.43" |
| 0.99 | 100 | 98.01% (1.99% left) | 50.25 | "about 50" |
| 0.99 | unbounded | 99% | 100 | not printed; ceiling for p = 0.99 |
| 0.999 | unbounded | 99.9% | 1000 | consistent with "more than 99.9%" behind results above 100x (p. 10) |

The p. 9 speedup example (200 s against 10 s gives 20) is plain division.

### A17. Rules of thumb: more than 100x for suitable applications, 10x "with just a few hours of work", about 10x when DRAM bandwidth saturates (pp. 7, 10)

**Verdict: COULD NOT VERIFY.** These are the authors' experience, not documented vendor facts, and no current source states them. Present each as the book's rule of thumb with its page. The DRAM-saturation remark is the one the course tests for itself: chapter 5's roofline and its lab benchmarks measure where a kernel sits against the reader's own bandwidth, so the lesson should point there and give no number of its own.

---

## Group B: HIP

The chapter uses no CUDA API, so there is nothing to map call by call. These three items support one sentence on where HIP fits.

### B1. What HIP is and which hardware it runs on

**Verdict: CONFIRMED.** HIP is AMD's CUDA-like C++ runtime API and kernel language, part of ROCm. AMD's current documentation describes it as targeting AMD GPUs only.

- "HIP is a C++ runtime API and kernel language for AMD GPUs. It is part of AMD's ROCm platform and lets developers create applications that run on heterogeneous systems, using CPUs and AMD GPUs from a single source code base."
- FAQ: "What hardware does HIP support? HIP supports AMD GPUs."
- The host and device split is the book's: "HIP provides two components: those that run on the CPU, also known as host system, and those that run on GPUs, also referred to as device. The host-based code is used to create device buffers, move data between the host application and a device, launch the device code (also known as kernel), manage streams and events, and perform synchronization."

A documented disagreement the reader will meet: PyTorch's HIP note (docs 2.15, last updated Oct 01, 2026) still says "HIP is used when converting existing CUDA applications like PyTorch to portable C++ and for new projects that require portability between AMD and NVIDIA." AMD's "What is HIP?" page, HIP FAQ and HIP porting guide contain no mention of an NVIDIA target (I searched the three pages for "NVIDIA GPUs", "NVIDIA platform" and `PLATFORM_NVIDIA` and found none), and llama.cpp's HIP header refuses non-AMD targets (ch05 B9). Teach HIP as "CUDA's programming model on AMD GPUs", not as a way to write once for both vendors.

- https://rocm.docs.amd.com/projects/HIP/en/latest/what_is_hip.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/faq.html
- https://docs.pytorch.org/docs/2.15/notes/hip.html

### B2. How close HIP is to CUDA (the book's "skills transfer" argument, pp. 13, 17 to 18)

**Verdict: CONFIRMED,** with AMD's own limit on the claim.

- "GPU programmers with NVIDIA CUDA experience will find the HIP API straightforward."
- "However, HIP is not intended to be a drop-in replacement for NVIDIA CUDA, and developers should expect to do some manual coding and performance tuning work to port existing projects to AMD GPUs as described in the HIP porting guide."
- FAQ: "The HIPIFY tools can automatically convert almost all CUDA runtime code to HIP. Most device code needs no additional conversion".

This is the same shape as the book's remark about OpenCL (correct everywhere, retune per device, p. 13), stated by the vendor. The spelling-level detail is in ch05 B3 to B8.

- https://rocm.docs.amd.com/projects/HIP/en/latest/what_is_hip.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/faq.html

### B3. Do the book's device arguments (peak rates by precision, memory bandwidth) apply to AMD accelerators?

**Verdict: CONFIRMED.** AMD publishes its accelerators in the same terms, and separates dense from sparse figures in the table itself. This answers the notes' open question for HIP.

Instinct MI300X spec rows as printed: "Peak Double Precision (FP64) Performance 81.7 TFLOPs", "Peak Single Precision (FP32) Performance 163.4 TFLOPs", "Peak Single Precision (TF32 Matrix) Performance 653.7 TFLOPs", "Peak Half Precision (FP16) Performance 1.3 PFLOPs", "Peak Half Precision (FP16) Performance with Structured Sparsity 2.61 PFLOPs", "Peak Memory Bandwidth 5.3 TB/s".

Two things to take from it:

- Plain FP64 to FP32 is a factor of 2, as on the H100 (A1). The large jump again comes from matrix units at reduced precision, and AMD prints the with-sparsity figure as a separate row where NVIDIA uses a footnote.
- Side finding for ch05 A8: the figure "66.9" for H100 SXM5 FP32, which ch05 could not find on any NVIDIA page, is printed on this AMD page in a comparison chart ("FP32 (Vector) 66.9 163.4 ... H100 SXM5 MI300X OAM").

I read one AMD accelerator page. I did not check newer Instinct models.

- https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html

---

## Group C: Metal

### C1. What Metal is, and its words for the book's concepts

**Verdict: CONFIRMED.** Metal is Apple's graphics and compute API with its own C++-based shading language. Apple documents a compute function as a "compute kernel" run by "a grid of threads", which is the book's vocabulary.

- "Metal is a modern, tightly integrated graphics and compute API coupled with a powerful shading language designed so you can take full advantage of Apple silicon."
- "MSL is a variant of C++ designed for GPU programming. In Metal, code that runs on GPUs is called a shader, because historically they were first used to calculate colors in 3D graphics."
- The `kernel` keyword declares "A compute function (also known as a compute kernel), which performs a parallel calculation using a grid of threads."
- The host-side sequence Apple gives is the book's six steps (p. 15) under other names: "You'll find a GPU, prepare the MSL function to run on it by creating a pipeline, and create data objects accessible to the GPU. To execute the pipeline against your data, create a command buffer, write commands into it, and commit the buffer to a command queue."

The second quote echoes the book's GPGPU history (pp. 6 to 7): the shader name survives, the pixel-painting restriction does not. The link to GPGPU is mine; the quote is Apple's.

- https://developer.apple.com/metal/
- https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu

### C2. OpenCL on Apple platforms (the book presents OpenCL as the portable alternative, p. 13)

**Verdict: CONFIRMED** that Apple has deprecated it and names Metal as the replacement. Quote in A11.

Not checked: whether CUDA or HIP can target Apple silicon. AMD's FAQ lists Linux and Windows for ROCm ("Linux as well as Windows are supported by ROCm"). I did not look for an NVIDIA statement about macOS. For the lesson, "on a Mac the GPU compute interface is Metal" is supported by Apple's page; "CUDA does not run on a Mac" is unverified here.

- https://developer.apple.com/opencl/
- https://rocm.docs.amd.com/projects/HIP/en/latest/faq.html

### C3. Fig. 1.1 draws the GPU with its own path to DRAM, apart from the CPU

**Verdict: CONFIRMED** that this picture does not describe Apple silicon. See ch05 C6: "Apple GPUs have a unified memory model in which the CPU and the GPU share system memory." Apple's chip announcements describe one memory and one bandwidth figure for the whole chip (quote in A8) and name the "unified memory architecture" as a design tenet.

What still transfers, by the book's own argument: the throughput-oriented design of the GPU's arithmetic and the need for many threads. What does not: the host-to-device copy implied by separate memories. Chapter 2 is where the course takes this up.

- https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/

### C4. Metal equivalents of the chapter's other interfaces (OpenMP offload, MPI-level multi-GPU, dynamic parallelism)

**Verdict: COULD NOT VERIFY.** I found no Apple documentation of an OpenMP offload target for Metal, of a Metal-aware MPI, or of launching a compute kernel from inside a kernel, in the pages read for this file and for ch05. I did not search for any of the three specifically. State "none documented that we found" and not "does not exist".

---

## Group D: how the reader's three projects arrange their backends (goal 3)

All paths read on the check date at the commits listed in the header.

### D1. llama.cpp: one directory per backend

**Verdict: CONFIRMED.** `ggml/src/` holds these backend directories: `ggml-blas`, `ggml-cann`, `ggml-cpu`, `ggml-cuda`, `ggml-et`, `ggml-hexagon`, `ggml-hip`, `ggml-metal`, `ggml-musa`, `ggml-opencl`, `ggml-openvino`, `ggml-rpc`, `ggml-sycl`, `ggml-virtgpu`, `ggml-vulkan`, `ggml-webgpu`, `ggml-zdnn`, `ggml-zendnn`.

From `docs/build.md`:

- Metal: "On MacOS, Metal is enabled by default. Using Metal makes the computation run on the GPU."
- CUDA: "This provides GPU acceleration using an NVIDIA GPU."
- HIP: "This provides GPU acceleration on HIP-supported AMD GPUs. Make sure to have ROCm installed."

`ggml-hip` contains build files only; it compiles the `ggml-cuda` sources as HIP (ch05 B9). So the three backends this course follows are two code bases: one CUDA-family, one Metal.

- https://github.com/ggml-org/llama.cpp/tree/master/ggml/src
- https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md

### D2. llama.cpp: is sampling a host-side step? (our Amdahl example, not the book's)

**Verdict: PARTLY.** The standard sampler interface works on a host array, which makes sampling a fair example of a sequential remainder. An experimental interface now moves samplers into the backend graph, so "sampling always runs on the CPU" would be wrong.

From `include/llama.h`:

- Host path. The candidate list is a plain host struct: `typedef struct llama_token_data { llama_token id; float logit; float p; } llama_token_data;` held in `llama_token_data_array { llama_token_data * data; size_t size; int64_t selected; bool sorted; }`. The required sampler callback is `void (*apply)(struct llama_sampler * smpl, llama_token_data_array * cur_p); // required`.
- Backend path. In the same struct: "// [EXPERIMENTAL] // backend sampling interface:" with `backend_init` ("return true if the backend supports all ops needed by the sampler ..."), `backend_accept`, `backend_apply`, `backend_set_input`. There is also "// backend sampling API [EXPERIMENTAL]" with `llama_get_sampled_token_ith` ("Get the backend sampled token for the ith token.").
- `src/llama-sampler.cpp` (last changed 2026-11-12) has the fallback: "// check if all ggml ops used by the sampler are supported by the backend".

I read the header and searched the source file for these names. I did not trace which samplers implement the backend path or whether any front end enables it by default.

Safe statement for the lesson: in llama.cpp the default sampler chain runs on the host over an array of candidates, so it is part of the sequential fraction in Amdahl's terms; an experimental path that runs sampling on the GPU exists, and the course's sampling module (X1) is about writing that kind of kernel.

- https://github.com/ggml-org/llama.cpp/blob/master/include/llama.h
- https://github.com/ggml-org/llama.cpp/blob/master/src/llama-sampler.cpp

### D3. MLX: Metal first, CUDA second, no HIP

**Verdict: CONFIRMED.** `mlx/backend/` contains `common`, `cpu`, `cuda`, `gpu`, `metal`, `no_cpu`, `no_gpu`. There is no HIP or ROCm directory.

- README: "MLX is an array framework for machine learning on Apple silicon, brought to you by Apple machine learning research."
- README: "To install the CUDA backend on Linux, run: pip install mlx[cuda]" and "To install a CPU-only Linux package, run: pip install mlx[cpu]".
- `mlx/backend/cuda/` (last changed 2026-11-17) holds `.cu` kernels by operation, for example `softmax.cu`, `scan.cu`, `sort.cu`, `reduce.cu`, `scaled_dot_product_attention.cu`, and directories `gemms/`, `quantized/`, `steel/`.

I listed the directories and read the README. I did not read the CUDA kernels.

- https://github.com/ml-explore/mlx/tree/main/mlx/backend
- https://github.com/ml-explore/mlx/blob/main/README.md
- https://github.com/ml-explore/mlx/tree/main/mlx/backend/cuda

### D4. PyTorch: CUDA sources translated for ROCm; a separate Metal backend called MPS

**Verdict: CONFIRMED.**

- Directories under `aten/src/ATen/native/` include `cuda/`, `cudnn/`, `hip/`, `mps/`, `metal/`, `vulkan/`, `xpu/`, `quantized/`, `transformers/`.
- ROCm build: `tools/amd_build/build_amd.py` (last changed 2026-10-20) is described in its own argument parser as "Top-level script for HIPifying, filling in most common parameters". Its `includes` list names the CUDA trees to translate, among them `"aten/src/ATen/native/cuda/*"`, `"aten/src/ATen/native/transformers/cuda/attention.cu"`, `"aten/src/ATen/native/quantized/cuda/*"`, `"c10/cuda/*"`, `"torch/*"`. It imports `hipify_python` from `torch/utils/hipify`.
- So on AMD, PyTorch's kernels are the CUDA kernels after a source translation. `aten/src/ATen/native/hip/` holds a small set of AMD-only files (`ck_gemm_float.hip`, `ck_gemm_half.hip`, `ck_gemm_bfloat16.hip`, `ck_bgemm.h` and similar). I did not read them.
- User-visible consequence, PyTorch docs: "PyTorch for HIP intentionally reuses the existing torch.cuda interfaces." The example comments read `cuda = torch.device('cuda') # Default HIP device` and "'rocm' or 'hip' are not valid, use 'cuda'".
- Metal: "mps device enables high-performance training on GPU for macOS devices with Metal programming framework. It introduces a new device to map Machine Learning computational graphs and primitives on highly efficient Metal Performance Shaders Graph framework and tuned kernels provided by Metal Performance Shaders framework respectively."
- `aten/src/ATen/native/mps/kernels/` holds hand-written MSL, for example `Matmul.metal`, `SoftMax.metal`, `Attention.metal`, `Quantized.metal`, `ScanKernel.metal`, `Sort.metal`, `ReduceOps.metal`. The docs sentence above describes only Apple's frameworks, so it understates how much of the backend is PyTorch's own Metal code.

Not examined: what the separate `aten/src/ATen/native/metal/` directory is for.

- https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native
- https://github.com/pytorch/pytorch/blob/main/tools/amd_build/build_amd.py
- https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/mps/kernels
- https://docs.pytorch.org/docs/2.15/notes/hip.html
- https://docs.pytorch.org/docs/2.15/notes/mps.html

---

## Summary of corrections

Items where the book's statement, or a commonly assumed fact, turned out PARTLY right or could not be verified:

- A1: the book's three H100 figures mix two kinds of row. 34 (FP64) and 67 (FP32) are ordinary peak rates. 1,979 is NVIDIA's "FP16 Tensor Core" figure with the footnote "With sparsity". The 30x step is tensor cores plus sparsity, not half precision. Figures are for the SXM variant.
- A4: "two hardware threads per core" is not general. Intel's E-core Xeon checked here lists threads equal to cores, and Ampere calls its cores single-threaded. Core counts have moved well past the book's (up to 256 on AMD, 288 on Intel, 192 on Ampere).
- A5: a generation after Blackwell is shipping (Vera Rubin, per NVIDIA's November 17, 2026 release), and current CUDA is 13.5. NVIDIA's compute capability table does not list Rubin yet.
- A6: NVIDIA dates CUDA to 2006 ("In 2006, NVIDIA introduced the Compute Unified Device Architecture"); toolkit 1.0 is June 2007. The book says 2007. The G80 could not be confirmed from a current NVIDIA page.
- A7: "more than 1 billion CUDA enabled GPUs" could not be verified. NVIDIA's March 2026 keynote says "hundreds of millions of GPUs and computing systems around the world that run CUDA".
- A8: published figures give a GPU-to-CPU-socket memory bandwidth ratio of about 5 to 7 (H100 SXM against EPYC 9965 and 9654), not 10, and the ratio is undefined on unified-memory chips.
- A9: the book's area-and-power cost rule for latency against throughput (p. 5) has no vendor source. NVIDIA confirms the design split, not the cost numbers.
- A10: the November 2026 TOP500 text names accelerators for only some of its five exascale systems; the No. 1 system is described by its processors alone.
- A11: OpenCL 1.0 was ratified on December 9th 2008, not 2009, by a much longer list of companies. OpenCL is at 3.1 and Apple deprecated it in macOS 10.14.
- A13: MPI on "more than 100,000 nodes" could not be verified.
- A15: Elsevier's web pages for the 5th edition show a 25-chapter, four-part table of contents that does not match the printed 24-chapter, three-part book, and give two publication dates (February 27, 2026 and June 3, 2026) against a 2027 copyright line. Chapter numbers from 12 onward differ from the 4th edition.
- A17: the 100x, 10x-in-a-few-hours and 10x-at-DRAM-saturation figures are the authors' rules of thumb with no documented source.
- B1: AMD documents HIP as supporting AMD GPUs only. PyTorch's docs still describe HIP as giving "portability between AMD and NVIDIA".
- B3 (correction to ch05 A8): "66.9" teraFLOPS for H100 SXM5 FP32 is printed on AMD's MI300X page, not on any NVIDIA page found.
- C4: no Metal counterpart to OpenMP offload, MPI-level multi-GPU or dynamic parallelism was found, and none was searched for specifically.
- D2: llama.cpp sampling is host-side through its standard interface, but an experimental backend sampling interface exists. Do not say sampling always runs on the CPU.
- D4: PyTorch's MPS backend contains many hand-written `.metal` kernels; its documentation mentions only Apple's Metal Performance Shaders frameworks.

## Where the backends really differ

Chapter 1 uses no API, so this table is about the chapter's claims and framing. API-level differences are in ch05-facts.md (B3 to B8, C1 to C9).

| Feature | CUDA | HIP | Metal |
|---|---|---|---|
| What it is | NVIDIA's platform; "In 2006, NVIDIA introduced" it (A6) | "a C++ runtime API and kernel language for AMD GPUs", part of ROCm (B1) | "a modern, tightly integrated graphics and compute API coupled with a powerful shading language" (C1) |
| Hardware it targets | NVIDIA GPUs | "HIP supports AMD GPUs." (B1) | Apple GPUs (C1). Other GPUs not checked |
| Kernel language | C++ with CUDA extensions | The same extensions with the same spellings (ch05 B4); "not intended to be a drop-in replacement" (B2) | MSL, "a variant of C++ designed for GPU programming" (C1) |
| Name for the function threads run | kernel | kernel | "shader"; a compute function is "also known as a compute kernel" (C1) |
| Memory picture of Fig. 1.1 (GPU with its own DRAM) | Matches the book | Same host and device model: "create device buffers, move data between the host application and a device" (B1) | Does not match on Apple silicon: "the CPU and the GPU share system memory" (ch05 C6) |
| The "10x memory bandwidth over the CPU" rule | H100 SXM 3.35 TB/s; about 5 to 7 times a server CPU socket on published figures (A8) | Same kind of device: MI300X "Peak Memory Bandwidth 5.3 TB/s" (A8, B3) | No ratio exists: one "unified memory bandwidth" figure for the chip (A8) |
| Peak-rate tables by precision | Tensor-core rows carry the footnote "With sparsity" (A1) | Separate rows for dense and "with Structured Sparsity" (B3) | No published FLOPS table found; not searched |
| OpenCL as the portable alternative (p. 13) | Not checked | Not checked | "deprecated in macOS 10.14"; Apple recommends Metal (A11, C2) |
| Dynamic parallelism (dropped from the 5th edition) | Still documented; CDP2 is the default from CUDA 12.0 (A14) | Not supported (ch05 B4) | None found; not searched (C4) |
| Cluster level (MPI and friends, p. 13) | CUDA-aware MPI, NCCL, NVSHMEM (A13) | Open MPI's docs list a ROCm section beside CUDA; not read (A13) | None found; not searched (C4) |
| PyTorch device name | `'cuda'` | `'cuda'` as well: "'rocm' or 'hip' are not valid, use 'cuda'" (D4) | `'mps'` (D4) |
| Where PyTorch's kernels come from | `aten/src/ATen/native/cuda/` | The same CUDA sources, translated by `tools/amd_build/build_amd.py` (D4) | Separate code in `aten/src/ATen/native/mps/`, including `.metal` kernels (D4) |
| llama.cpp backend | `ggml/src/ggml-cuda/` | `ggml/src/ggml-hip/`, which compiles the `ggml-cuda` sources (D1, ch05 B9) | `ggml/src/ggml-metal/`, enabled by default on macOS (D1) |
| MLX backend | `mlx/backend/cuda/` (Linux) | None (D3) | `mlx/backend/metal/`, the primary backend (D3) |

---

## Addendum to A15 (added 2026-11-20 by the builder of section 1.7): the 4th edition's chapter titles

**Verdict: CONFIRMED** for the titles of all 23 chapters of the 4th edition, which A15 gave only from chapter 12 on. Read through a fetch tool that returns extracted text (the page refuses plain HTTP clients), so: "(extracted)". The list agrees with every title A15 already records.

Table of contents on Elsevier's 4th-edition page, as returned: "1. Introduction", "Part I Fundamental Concepts", "2. Heterogeneous data parallel computing", "3. Multidimensional grids and data", "4. Compute architecture and scheduling", "5. Memory architecture and data locality", "6. Performance considerations", "Part II Parallel Patterns", "7. Convolution: An introduction to constant memory and caching", "8. Stencil", "9. Parallel histogram", "10. Reduction And minimizing divergence", "11. Prefix sum (scan)", "12. Merge: An introduction to dynamic input data identification", "Part III Advanced patterns and applications", "13. Sorting", "14. Sparse matrix computation", "15. Graph traversal", "16 Deep learning", "17. Iterative magnetic resonance imaging reconstruction", "18. Electrostatic potential map", "19. Parallel programming and computational thinking", "Part IV Advanced Practices", "20. Programming a heterogeneous computing cluster: An introduction to CUDA streams", "21. CUDA dynamic parallelism", "22. Advanced practices and future evolution", "23. Conclusion and outlook", "Appendix A: Numerical considerations". Edition line: "4th Edition - May 28, 2022".

Side observations: the 4th edition has four parts, and Sorting (13) opens its Part III; the 5th edition's Preface says Sort moved into Part II (notes/ch01.md, p. xxii), which matches. Pairing chapters between editions by title remains our inference, as in A15.

- https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0
