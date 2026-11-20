# Chapter 24 fact check: the G80-to-B200 table, and what its rows mean on HIP and Metal

Checked on 2026-11-20 against the live vendor sources listed under each item.

How to read this file:

- **Verdict** is one of CONFIRMED, WRONG, PARTLY, COULD NOT VERIFY. Where an item has several sub-claims, the verdict is for the item as a whole and the text says which sub-claim failed.
- **Quote** is copied from the source (HTML flattened to text, PDF text from `pdftotext`, GitHub raw files, Apple's documentation JSON). Line breaks inside quotes are removed.
- "Book" means Fig. 24.1 and the running text on pp. 578 to 579, as recorded in `notes/ch24.md`. The chapter has no code, so there is no API to port. Groups B and C record what the table's NVIDIA-only vocabulary (Tensor Core, NVLink, NCCL, the PCIe host link) corresponds to on AMD and Apple, because a reader with goals 2 and 3 will ask.
- Already settled elsewhere and not repeated here: the year CUDA was introduced (ch01-facts A6, which A1 below extends with a current NVIDIA page that names the G80), AMD publishing peak figures in the same terms (ch01-facts B3), H100 figures and how NVIDIA qualifies tensor-core numbers (ch05-facts A8), cuBLAS and CUTLASS (ch05-facts A9), the current ROCm release (ch05-facts B1), llama.cpp's HIP build from CUDA sources (ch05-facts B9), Apple unified memory and shared storage (ch05-facts C6), where llama.cpp and MLX keep their Metal kernels (ch05-facts C10). `notes/today/infra-facts.md` did not exist when this was written.
- Versions the sources reported about themselves on the check date: CUDA Toolkit 13.5.0 (November 2026); CUDA Programming Guide pages "Last updated on Oct 31, 2026"; NCCL docs 2.32.3, latest GitHub release v2.32.6-1 (2026-11-09); ROCm Core SDK 10.2.0 with HIP docs 7.17.0 and RCCL docs 2.32.3; Metal Shading Language Specification Version 4.1 (2026-06-04); MLX docs 0.32.4; PyTorch docs 2.15.
- Not checked: anything by running code on hardware. Every statement is from documentation, product pages, datasheets, headers or project source. No G80 or B200 was measured.
- All device figures below are vendor-published peak figures. Under the project's ground rules they may appear in a lesson only as "the book's figures" or "the vendor's published figure", never as the reader's numbers.

Totals: 14 CONFIRMED, 2 WRONG, 14 PARTLY, 2 COULD NOT VERIFY (32 items: A1 to A19, B1 to B4, C1 to C5, D1 to D4). The two WRONG items are A6 (G80 host link) and A13 (NVLink unit). The two errors the notes flagged in Fig. 24.1 are settled in A9 and A13.

---

## Group A: do the book's dated statements still hold (NVIDIA)

### A1. "the first CUDA-enabled GPU, the G80, in 2007"

**Verdict: PARTLY.** The G80 is the first CUDA-capable GPU, but NVIDIA dates the hardware to November 2006. What is dated 2007 is the first CUDA Toolkit and the G80-based Tesla board.

Quotes:

- NVIDIA GeForce news page (dated November 05, 2026): "Few graphics cards have left a mark on PC gaming quite like the GeForce 8800 GTX, which hit shop shelves November 8th, 2006." and "with the GeForce 8800 GTX being the first CUDA-capable graphics card" and "At the heart of the GeForce 8800 GTX was our G80 architecture with 128 unified stream processors."
- NVIDIA Fermi whitepaper: "Introduced in November 2006, the G80 based GeForce 8800 brought several key innovations to GPU Computing". It also lists what the G80 introduced, which is the model the whole book teaches: "G80 introduced the single-instruction multiple-thread (SIMT) execution model" and "G80 introduced shared memory and barrier synchronization for inter-thread communication."
- CUDA Toolkit Archive, oldest entry: "CUDA Toolkit 1.0 (June 2007)". Newest entry: "CUDA Toolkit 13.5.0 (November 2026)".
- Tesla C870 board specification, revision history: "01 | July 24, 2007 | Preliminary Release".
- Legacy compute capability list, row 1.0: "Tesla C870 | Tesla D870 | Tesla S870 | Quadro FX 5600 | Quadro FX 4600 | Quadro Plex 2100 S4 | ... | GeForce 8800 Ultra | GeForce 8800 GTX".

Safe wording: "the G80 (GeForce 8800 GTX, November 2006) was the first CUDA-capable GPU; the first CUDA Toolkit followed in June 2007". The book's "(2007)" is the software date.

- https://www.nvidia.com/en-us/geforce/news/8800-gtx-gpu-20-year-celebration-dlss-4-5-path-tracing-steam-demo/
- https://www.nvidia.com/content/pdf/fermi_white_papers/nvidiafermicomputearchitecturewhitepaper.pdf (pages 3 and 4)
- https://developer.nvidia.com/cuda-toolkit-archive
- https://www.nvidia.com/docs/IO/43395/C870-BoardSpec_BD-03399-001_v04.pdf
- https://developer.nvidia.com/cuda/gpus/legacy

### A2. G80 compute throughput "0.345 TFLOPS (FP32)"

**Verdict: PARTLY.** No NVIDIA page I found prints 0.345 TFLOPS or 345 GFLOPS. The figure is reproduced exactly by arithmetic on two numbers NVIDIA does publish, so it is a theoretical peak, not a measurement.

- Fermi whitepaper, summary table, G80 column: "CUDA Cores | 128" and "Single Precision Floating Point Capability | 128 MAD ops/clock".
- Tesla C870 board specification: "Number of streaming processor cores: 128" and "Processor core clock: 1.35 GHz".
- Arithmetic (mine): 128 multiply-adds per clock, 2 floating-point operations per multiply-add, 1.35 GHz gives 128 x 2 x 1.35 = 345.6 GFLOPS, which is 0.3456 TFLOPS.

This answers part of the notes' "Fig. 24.1 basis" question: the G80 compute number is a peak count of multiply-add operations, the same kind of number as the B200 figures in A7 and A8.

- https://www.nvidia.com/content/pdf/fermi_white_papers/nvidiafermicomputearchitecturewhitepaper.pdf (summary table, page 11 of the PDF)
- https://www.nvidia.com/docs/IO/43395/C870-BoardSpec_BD-03399-001_v04.pdf (Key Features)

### A3. G80 memory bandwidth "86.4 GB/s"

**Verdict: PARTLY.** 86.4 GB/s is the GeForce 8800 GTX figure (768 MB card, 900 MHz memory clock). I could confirm the 384-bit bus on a live NVIDIA page but found the 86.4 number itself only in a 2007 post on NVIDIA's developer forum that transcribes the spec table. The 1.5 GB G80 board in the same column of the book (A4) has a slower memory clock, which gives a lower figure.

- NVIDIA GeForce news page: "With 768MB of GDDR3 memory and a 384-bit memory bus".
- NVIDIA developer forum, post of 2007-03-22 (a user quoting the product table, so a secondary source): "GeForce 8800 GTX Stream Processors 128 Core Clock (MHz) 575 Shader Clock (MHz) 1350 Memory Clock (MHz) 900 Memory Amount 768MB Memory Interface 384-bit Memory Bandwidth (GB/sec) 86.4".
- Tesla C870 board specification: "Memory clock: 800 MHz" and "Interface: 384-bit GDDR3".
- Arithmetic (mine): 384 bits x 2 transfers per clock x 900 MHz / 8 = 86.4 GB/s. The same formula at the C870's 800 MHz gives 76.8 GB/s. NVIDIA's C870 document prints no GB/s figure.

This also answers the notes' question about which memory the row means: it is the peak interface rate of the on-board device memory (GDDR3 then, HBM3e in A9), not a measured copy rate.

- https://www.nvidia.com/en-us/geforce/news/8800-gtx-gpu-20-year-celebration-dlss-4-5-path-tracing-steam-demo/
- https://forums.developer.nvidia.com/t/gtx-or-gts/444
- https://www.nvidia.com/docs/IO/43395/C870-BoardSpec_BD-03399-001_v04.pdf

### A4. G80 memory capacity "1.5GB"

**Verdict: PARTLY.** Confirmed for the Tesla C870, the G80-based compute board. The GeForce 8800 GTX, which is where 86.4 GB/s comes from, had 768 MB. The book's G80 column therefore describes no single product: capacity from one board, bandwidth from another.

- Tesla C870 board specification: "The Tesla C870 board offers 1536 MB of 136-pin ball grid array (BGA) GDDR3 memory." and "The Tesla C870 is based on the G80 Tesla GPU from NVIDIA."
- NVIDIA GeForce news page: "With 768MB of GDDR3 memory".
- The same C870 document has its own typo in Table 1: "Memory size | 1536 GHz".

- https://www.nvidia.com/docs/IO/43395/C870-BoardSpec_BD-03399-001_v04.pdf
- https://www.nvidia.com/en-us/geforce/news/8800-gtx-gpu-20-year-celebration-dlss-4-5-path-tracing-steam-demo/

### A5. PCI Express rates per generation (reference for A6 and A12)

**Verdict: CONFIRMED** for the arithmetic "Gen2 x16 = 8 GB/s each way" and "Gen5 x16 = about 64 GB/s each way". Source is the Linux kernel's PCI core, because pcisig.com refused every automated request today (HTTP 403), so I could not read PCI-SIG's own pages.

Linux `drivers/pci/pci.h` (master), exact text:

```c
/* PCIe speed to Mb/s reduced by encoding overhead */
#define PCIE_SPEED2MBS_ENC(speed) \
	((speed) == PCIE_SPEED_64_0GT ? 64000*1/1 : \
	 (speed) == PCIE_SPEED_32_0GT ? 32000*128/130 : \
	 (speed) == PCIE_SPEED_16_0GT ? 16000*128/130 : \
	 (speed) == PCIE_SPEED_8_0GT  ?  8000*128/130 : \
	 (speed) == PCIE_SPEED_5_0GT  ?  5000*8/10 : \
	 (speed) == PCIE_SPEED_2_5GT  ?  2500*8/10 : \
	 0)
```

Per direction, 16 lanes (arithmetic mine):

| Generation | Line rate | Encoding | Per lane after encoding | x16, each way |
|---|---|---|---|---|
| Gen1 | 2.5 GT/s | 8b/10b | 2,000 Mb/s | 4.0 GB/s |
| Gen2 | 5 GT/s | 8b/10b | 4,000 Mb/s | 8.0 GB/s |
| Gen5 | 32 GT/s | 128b/130b | 31,507 Mb/s | 63.0 GB/s (64 before encoding overhead) |

Cross-check from NVIDIA: "1.8TB/s of bidirectional throughput per GPU is over 14X the bandwidth of PCIe Gen5". 1,800 / 14 is about 128 GB/s, which is Gen5 x16 counted in both directions, so 64 GB/s each way.

- https://github.com/torvalds/linux/blob/master/drivers/pci/pci.h (lines 624 to 632 today)
- https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-architecture-technical-brief (section "Fifth-Generation NVLink and NVLink Switch")
- https://pcisig.com/ (COULD NOT VERIFY directly: 403 to scripted clients)

### A6. G80 host link "8 GB/s (Gen2 x16, each way)"

**Verdict: WRONG** for the Tesla C870, the only G80 board that matches the book's 1.5 GB. NVIDIA's board specification says Generation 1, which is 4 GB/s each way (A5). COULD NOT VERIFY for the GeForce 8800 GTX: I found no live NVIDIA page stating its PCI Express generation.

- "The NVIDIA Tesla C870 GPU computing processor board is a PCI Express (Gen 1) full-height ... form factor computing add-in card".
- Key Features: "PCI Express x16 Generation 1 system interface". Table 1: "Board connector | PCI Express Gen 1 x16".

Consequence: on NVIDIA's own figures the host-link growth is 4 GB/s to about 64 GB/s, 16x, not the printed 8x. The teaching point survives either way: the host link grew by one order of magnitude while everything on the device grew by two or more.

- https://www.nvidia.com/docs/IO/43395/C870-BoardSpec_BD-03399-001_v04.pdf (Overview, Key Features, Table 1)

### A7. "B200 (2025)" as the comparison device

**Verdict: PARTLY.** The B200 is real and current in NVIDIA's lists, but it was announced in March 2024 and NVIDIA reported its first quarter of volume sales as the quarter ending January 26, 2025. "2025" is the year of general availability, not of introduction.

- Press release, March 18, 2024: "The NVIDIA GB200 Grace Blackwell Superchip connects two NVIDIA B200 Tensor Core GPUs to the NVIDIA Grace CPU" and "Blackwell-based products will be available from partners starting later this year."
- Fourth quarter fiscal 2025 results (quarter ended January 26, 2025; released February 26, 2025): "We've successfully ramped up the massive-scale production of Blackwell AI supercomputers, achieving billions of dollars in sales in its first quarter."
- Compute capability table: "10.0 | NVIDIA GB200 | NVIDIA B200".
- HGX page today: "HGX B300 and HGX B200 shipping now."

- https://nvidianews.nvidia.com/news/nvidia-blackwell-platform-arrives-to-power-a-new-era-of-computing
- https://nvidianews.nvidia.com/news/nvidia-announces-financial-results-for-fourth-quarter-and-fiscal-2025
- https://developer.nvidia.com/cuda/gpus
- https://www.nvidia.com/en-us/data-center/hgx/

### A8. B200 compute throughput "80 TFLOPS (FP32)" and ratio "232x"

**Verdict: PARTLY.** NVIDIA publishes two different FP32 figures per Blackwell GPU depending on the system. The book's 80 matches the GB200 NVL72 rack figure divided by 72. For a GPU sold as "B200" on an HGX B200 board, NVIDIA prints 75.

- Blackwell Architecture Technical Brief V2.1, Table 3, "Per GPU Specs Below", HGX B200 column: "FP32 | 75 teraFLOPS".
- Same document, GB200 NVL72 column of the rack table: "Total Rack GPUs | CPUs | 72 | 36" and "FP32 | 5,760 teraFLOPS". 5,760 / 72 = 80 (arithmetic mine).
- HGX page today, HGX B200 column (8 GPUs): "FP32 | 600 TFLOPS". 600 / 8 = 75.
- AMD's comparison table reads NVIDIA the same way: "FP32 (Vector) | 75 | 157.3 | B200 SXM5 | MI355X OAM".

Ratios (arithmetic mine): 80 / 0.345 = 232 as printed; 75 / 0.3456 = 217.

- https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-architecture-technical-brief (Tables 2 and 3; PDF created 2025-10-01, "V2.1")
- https://www.nvidia.com/en-us/data-center/hgx/
- https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html

### A9. B200 "2200 TFLOPS (Tensor Core FP32)" and ratio "638x"

**Verdict: PARTLY.** This settles the first error the notes found. 2,200 TFLOPS is a real NVIDIA figure, so the printed number is right and the printed ratio is wrong by a factor of ten: 2,200 / 0.345 = 6,377, not 638. But the row label is also wrong in two ways: the precision is TF32, not FP32, and 2,200 is the figure with sparsity. The dense figure is half.

- Technical Brief V2.1, Table 3, per-GPU, HGX B200 column: "TF32 Tensor Core Dense/Sparse | 1.1/2.2 petaFLOPS".
- HGX page today: "TF32 Tensor Core | 18 PFLOPS" for eight GPUs, footnote "2. Specification in Sparse. Dense is ½ sparse spec shown." 18 / 8 = 2.25 PFLOPS sparse per GPU.
- "Tensor Core FP32" is not a precision NVIDIA lists. Tensor Cores page, Blackwell column: "Supported Tensor Core Precisions | NVFP4, FP64, TF32, BF16, FP16, FP8/FP6, INT8" and "Supported CUDA Core Precisions | FP64, FP32, FP16, BF16". Plain FP32 is the 75 or 80 TFLOPS figure in A8.
- What TF32 is, in terms the reader already uses (PyTorch 2.15 docs): "TF32 tensor cores are designed to achieve better performance on matmul and convolutions on torch.float32 tensors by rounding input data to have 10 bits of mantissa, and accumulating results with FP32 precision, maintaining FP32 dynamic range." So the two numbers on the book's compute row are not the same arithmetic.

Ratios against 0.3456 TFLOPS (arithmetic mine): 6,366x for 2.2 PFLOPS with sparsity; 3,183x for the dense 1.1 PFLOPS. A value of 220 TFLOPS, which would give the printed 638x, appears in no NVIDIA source I read.

Same qualifier pattern as ch05-facts A8, where the H100's 989 TFLOPS is also "TF32 Tensor Core" with sparsity.

- https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-architecture-technical-brief (Table 3)
- https://www.nvidia.com/en-us/data-center/hgx/
- https://www.nvidia.com/en-us/data-center/tensor-cores/
- https://docs.pytorch.org/docs/2.15/notes/cuda.html (section "TensorFloat-32 (TF32) on Ampere (and later) devices")

### A10. B200 memory bandwidth "8,000 GB/s" and ratio "93x"

**Verdict: CONFIRMED,** with NVIDIA's own qualifier "up to". One NVIDIA document prints 7.7 TB/s instead.

- HGX AI Factory reference architecture (last updated Aug 31, 2026), Table 1, NVIDIA B200 SXM column: "GPU Bandwidth | Up to 8TB/s" and "GPU Aggregate Bandwidth per 8-GPU Node | Up to 64TB/s".
- DGX B200 product page: "GPU Memory | 1,440 GB total, 64 TB/s HBM3e bandwidth". 64 / 8 = 8 TB/s.
- Technical Brief V2.1, per-GPU row, HGX B200 column: "Up to 192 GB HBM3e | 7.7 TB/s".

Ratios (arithmetic mine): 8,000 / 86.4 = 93 as printed; 7,700 / 86.4 = 89; 8,000 / 76.8 (the 1.5 GB C870 from A3) = 104.

- https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html (Table 1)
- https://www.nvidia.com/en-us/data-center/dgx-b200/
- https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-architecture-technical-brief (Table 3)

### A11. B200 memory capacity "192 GB" and ratio "128x"

**Verdict: PARTLY.** NVIDIA's technical brief does print "Up to 192 GB", so the book's number has a source. NVIDIA's product page, reference architecture and system guide all give 180 GB per GPU for what actually ships.

- Technical Brief V2.1, per-GPU row, HGX B200: "Up to 192 GB HBM3e | 7.7 TB/s".
- Reference architecture, Table 1, NVIDIA B200 SXM: "Memory per GPU | 180GB HBM3e" and "Memory per 8-GPU Node | 1.44TB HBM3e".
- DGX B200 User Guide: "8 x NVIDIA B200 GPUs that provide 1,440 GB total GPU memory". 1,440 / 8 = 180.
- HGX page today, HGX B200: "Total Memory | 1.4 TB".
- NVIDIA's documents disagree with each other here: the same technical brief that says 192 GB per GPU says "Fast Memory | Up to 1.4 TB" for eight of them.

Ratios (arithmetic mine): 192 / 1.5 = 128 as printed; 180 / 1.5 = 120.

- https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-architecture-technical-brief (Table 3)
- https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html (Table 1)
- https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html
- https://www.nvidia.com/en-us/data-center/hgx/

### A12. B200 host link "64 GB/s (Gen5 x16, each way)"

**Verdict: CONFIRMED.** The generation and width are NVIDIA's; the GB/s value follows from A5 (63.0 GB/s after 128b/130b encoding, 64 before).

- Reference architecture: "PCI Express | Eight Gen5 x16 links and one Gen4 x2 link per HGX H100, H200 or B200 8-GPU baseboard". Eight links for eight GPUs.
- Technical Brief V2.1, per-GPU "Interconnect" row, HGX B200 column: "NVLink 5" and "PCIe Gen5".

- https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html
- https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-architecture-technical-brief (Table 3)

### A13. NVLink bandwidth "1,800 TB/s"

**Verdict: WRONG.** This settles the second error the notes found. It is a unit error: NVIDIA's figure is 1,800 GB/s, which is 1.8 TB/s. The printed value is 1,000 times too large.

- NVLink page, specification table, Fifth Generation column (NVIDIA Blackwell): "NVLink Bandwidth per GPU | 900 GB/s | 1,800 GB/s | 3,000 GB/s" for fourth, fifth and sixth generation.
- Technical Brief V2.1: "Blackwell and Blackwell Ultra GPUs include 18 fifth-generation NVLink links to provide 1.8 TB/sec total bandwidth, 900 GB/sec in each direction."
- HGX page, HGX B200: "NVLink GPU-to-GPU Bandwidth | 1.8 TB/s".

One more inconsistency inside the book's table that the notes did not list: the PCIe row is "each way" and the NVLink row is NVIDIA's total for both directions. Like for like, it is 900 GB/s each way against about 64 GB/s each way, which is the "over 14X the bandwidth of PCIe Gen5" NVIDIA quotes (A5).

- https://www.nvidia.com/en-us/data-center/nvlink/
- https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-architecture-technical-brief
- https://www.nvidia.com/en-us/data-center/hgx/

### A14. The ratio column, recomputed

**Verdict: PARTLY.** Three printed ratios are correct arithmetic on the printed inputs, one is off by ten, and none of the five rows survives unchanged when both inputs are taken from NVIDIA's own documents for one board on each side.

| Row | Book, as printed | Printed arithmetic | Using NVIDIA figures for Tesla C870 and HGX B200 | Source items |
|---|---|---|---|---|
| Compute, FP32 | 0.345 to 80 TFLOPS, 232x | 80 / 0.345 = 232, correct | 0.3456 to 75 TFLOPS, 217x | A2, A8 |
| Compute, tensor core | 0.345 to 2200 TFLOPS, 638x | 2200 / 0.345 = 6,377, printed ratio wrong | TF32 with sparsity 2.2 PFLOPS, 6,366x; dense 1.1 PFLOPS, 3,183x | A2, A9 |
| Memory bandwidth | 86.4 to 8,000 GB/s, 93x | 8000 / 86.4 = 93, correct | 76.8 (computed) to "up to" 8,000 GB/s, 104x | A3, A10 |
| Memory capacity | 1.5 to 192 GB, 128x | 192 / 1.5 = 128, correct | 1.5 to 180 GB, 120x | A4, A11 |
| PCIe, each way | 8 to 64 GB/s, 8x | 64 / 8 = 8, correct | 4 (Gen 1) to about 64 GB/s, 16x | A5, A6, A12 |
| NVLink | N/A to "1,800 TB/s" | unit wrong | N/A to 1,800 GB/s total, 900 GB/s each way | A13 |

All figures in the last-but-one column are peak figures from product documents, with my arithmetic where a cell says "computed". Nothing here was measured.

### A15. Is the B200 still NVIDIA's newest data-center GPU?

**Verdict: CONFIRMED that it is not.** The book's "2025" column is already one to two product steps old on the check date.

- Compute capability table: "10.3 | NVIDIA GB300 | NVIDIA B300" above "10.0 | NVIDIA GB200 | NVIDIA B200".
- HGX page today leads with the Rubin GPU: "The NVIDIA HGX Rubin NVL8 integrates eight NVIDIA Rubin GPUs with sixth-generation high-speed NVLink interconnects" and "The NVIDIA Vera Rubin is ramping into full production".
- HGX page, "Individual GPU Specifications | NVIDIA Rubin GPU": "TF32 | 2 PFLOPS" (dense), "FP32 | 130 TFLOPS", "NVLink Bandwidth | 3.6 TB/s", "GPU Memory | Bandwidth | 288 GB HBM4 | 22 TB/s".
- NVIDIA's pages disagree on Rubin's NVLink figure: the NVLink page says "The sixth-generation NVLink enables 3 TB/s of bandwidth per GPU" and marks its table "Preliminary specifications; may be subject to change."
- The Rubin GPU has no row in the compute capability table as fetched today.

Use in the lesson: at most one sentence saying the 2025 column has already been superseded and the vendor pages change. Do not copy these numbers into the lesson; they are preliminary by NVIDIA's own label.

- https://developer.nvidia.com/cuda/gpus
- https://www.nvidia.com/en-us/data-center/hgx/
- https://www.nvidia.com/en-us/data-center/nvlink/

### A16. NCCL as "a new communication library ... used extensively in training LLMs"

**Verdict: PARTLY.** The use in training is NVIDIA's own description. "New" is not: the public repository dates from 2015. It is actively developed.

- GitHub API for `NVIDIA/nccl`: created 2015-11-14, last push 2026-11-19, description "Optimized primitives for collective multi-GPU communication". Latest release "NCCL v2.32.6-1 Release", 2026-11-09.
- NCCL docs ("NCCL 2.32.3 documentation"): "The NVIDIA Collective Communications Library (NCCL, pronounced "Nickel") is a library providing inter-GPU communication primitives that are topology-aware and can be easily integrated into applications." and "NCCL has found great application in Deep Learning Frameworks, where the AllReduce collective is heavily used for neural network training."
- Same page: "It supports a variety of interconnect technologies including PCIe, NVLINK, InfiniBand Verbs, and IP sockets."
- The current docs table of contents already has "One-sided communication", "Device-Initiated Communication" and "Device API" sections. That bears on a ch. 23 statement, not on this chapter; it is noted here only so the ch. 23 fact check knows to look.

Relevance for this reader: NCCL is a training-scale, multi-GPU library. It is outside goals 1 to 3 except as a name they will meet in framework source.

- https://github.com/NVIDIA/nccl
- https://github.com/NVIDIA/nccl/releases
- https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html

### A17. "tens or even hundreds of thousands of GPUs in each training job", growing "since 2017"

**Verdict: PARTLY.** A single LLM-training cluster of 100,000 GPUs, growing to 200,000, is confirmed by an NVIDIA press release. Whether one training job uses all of them, and the "since 2017" start date, I could not verify from vendor documentation.

- NVIDIA press release, October 28, 2024: "xAI's Colossus supercomputer cluster comprising 100,000 NVIDIA Hopper GPUs" and "is being used to train xAI's Grok family of large language models" and "xAI is in the process of doubling the size of Colossus to a combined total of 200,000 NVIDIA Hopper GPUs."

- https://nvidianews.nvidia.com/news/spectrum-x-ethernet-networking-xai-colossus

### A18. "the advancement in physical light sources has practically stopped"

**Verdict: PARTLY.** The lithography vendor's pages support the narrow reading: the newest scanners use the same 13.5 nm light as the previous generation and gain resolution from optics. "Practically stopped" and the claim that masks designed by computation are now the primary driver are the authors' judgement; I found no vendor statement of either and did not look further. The lesson should attribute this to the book and not restate it as fact.

- ASML, NXE systems: "They're the first systems to use our novel 13.5 nm EUV light source, and they have a numerical aperture (NA) of 0.33. These systems print microchip features with a resolution of 13 nm".
- ASML, EXE systems: "With an (NA) of 0.55, their innovative new optics use our novel 13.5 nm EUV light source to provide higher contrast and print with a resolution of just 8 nm."

- https://www.asml.com/en/products/euv-lithography-systems

### A19. How kernel code reaches Tensor Cores in CUDA today

**Verdict: CONFIRMED.** The chapter's table is the only place it mentions Tensor Cores and gives no programming detail. For a reader whose first goal is matmul kernels, the current documented route in the CUDA Programming Guide is CUDA Tile. Libraries remain the other route (cuBLAS and CUTLASS, ch05-facts A9).

- CUDA Programming Guide, section 2.4 "Writing Tile Kernels" (last updated Oct 31, 2026): "tile provides a simpler way to access recent performance features of NVIDIA GPUs such as the tensor memory accelerator (TMA) unit and the tensor cores."
- "CUDA Tile C++ is available in the CUDA Toolkit from version 13.3 onward." "In C++, the tile API lives in the cuda::tiles namespace, which is exposed by the cuda_tile.h header."
- Section 2.4.9: "The programmer writes them at tile granularity and the compiler maps them to hardware, including tensor cores where available."
- Developer page: "NVIDIA CUDA Tile is a tile-based GPU programming model that targets portability for NVIDIA Tensor Cores."

Not checked here: any other in-kernel interface to Tensor Cores. That belongs to the ch. 15 and X2 fact checks.

- https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-tile-kernels.html
- https://developer.nvidia.com/cuda/tile

---

## Group B: HIP / AMD counterparts of the table's NVIDIA terms

### B1. Tensor Core counterpart: matrix cores (MFMA units)

**Verdict: CONFIRMED.** AMD documents dedicated matrix-multiply hardware in the HIP documentation itself, with a different name, a different instruction set and CDNA-only availability.

Quotes (HIP 7.17.0 docs, "Hardware implementation"):

- "CDNA architectures (MI100 and newer) include specialized matrix acceleration units for high-throughput matrix operations."
- "Matrix cores are GPU execution units that perform large-scale matrix operations in a single instruction. In AMD architectures, these units are formally known as MFMA (matrix fused multiply-add) units".
- "MFMA units support various data types including INT8, FP16, BF16, and FP32, with different throughput characteristics for each."
- Example instruction: `v_mfma_f32_16x16x4f16`, "a tile size of 16 x 16, with an inner dimension of 4, operating on half-precision (FP16) inputs and accumulating into 32-bit floating-point outputs."
- "Programmers can access MFMA functionality at multiple levels: through optimized libraries, compiler intrinsics, and/or inline assembly".

Documented differences from CUDA:

- The docs describe matrix cores under "CDNA architecture" only. The "RDNA architecture" guidance on the same page lists no matrix unit. So a HIP kernel cannot assume them on every AMD GPU.
- I found no HIP counterpart of CUDA Tile on the pages read. That is "not found", not "does not exist".
- TF32 on AMD depends on the part. The HIP hardware page does not mention it. The MI355X product page carries the footnote "*TF32 support through software emulation." and has no TF32 row. The older MI300X page does print a row, "Peak Single Precision (TF32 Matrix) Performance 653.7 TFLOPs" (ch01-facts B3). So TF32 is not a portable assumption even inside AMD's own line.
- Not fetched: the intrinsic names. A web search pointed to a ROCm blog and an AMD tutorial that name `__builtin_amdgcn_mfma_*` intrinsics; I did not open those pages, so treat the spelling as unverified until the ch. 15 or X2 fact check reads them.

- https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html (sections "Matrix fused multiply-add (MFMA)" and "Architecture variants")

### B2. NVLink counterpart: Infinity Fabric links

**Verdict: PARTLY.** AMD's product page documents a GPU-to-GPU link by that name with a peak figure. AMD's communication library calls the intra-node link "xGMI"; I did not find a page that states how the two names relate.

- MI355X product page: "Infinity Fabric™ Links | 7", "Peak Infinity Fabric™ Link Bandwidth | 144 GB/s", "Scale-up (Peak) AMD Infinity Fabric™ Link Bandwidth | 144 GB/s".
- RCCL docs: "It uses PCIe and xGMI high-speed interconnects."

The page gives a per-link peak and a link count, not a per-GPU total like NVIDIA's 1,800 GB/s, so the two vendors' figures are not directly comparable and the lesson should not compare them.

- https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html
- https://rocm.docs.amd.com/projects/rccl/en/latest/

### B3. NCCL counterpart: RCCL

**Verdict: CONFIRMED.** AMD ships a library with the same role and the same function names.

- "The ROCm Communication Collectives Library (RCCL) is a stand-alone library that provides multi-GPU and multi-node collective communication primitives optimized for AMD GPUs."
- The RCCL library specification lists its functions under NCCL's names: `ncclGetUniqueId()`, `ncclCommInitRank()`, `ncclCommDestroy()`, `ncclAllReduce()`, `ncclSend()`, `ncclRecv()`, `ncclGroupStart()`, `ncclGroupEnd()`. These are the same identifiers the book's ch. 23 listings use. Example text from that page: "If multiple ncclSend and ncclRecv operations need to progress concurrently to complete, they must be fused within a ncclGroupStart / ncclGroupEnd section."
- Version: the docs are titled "RCCL 2.32.3 Documentation" under "ROCm Core SDK 10.2.0", the same 2.32.3 number as NVIDIA's NCCL docs. Source lives at `projects/rccl` in `ROCm/rocm-systems`.
- "RCCL uses PCIe and xGMI high-speed interconnects for intra-node communication as well as InfiniBand, RoCE, and TCP/IP for inter-node communication."

- https://rocm.docs.amd.com/projects/rccl/en/latest/
- https://rocm.docs.amd.com/projects/rccl/en/latest/what-is-rccl.html
- https://rocm.docs.amd.com/projects/rccl/en/latest/api-reference/library-specification.html

### B4. The same table rows for a current AMD part

**Verdict: CONFIRMED** as AMD's published figures for one device (Instinct MI355X, "Launch Date | 6/12/2025"). Recorded so the lesson can say truthfully that AMD publishes the same kinds of peak number; the values must not be used as anything but the vendor's example.

| Row in the book's table | AMD's label and value |
|---|---|
| Compute, FP32 | "Peak Single Precision (FP32) Performance | 157.3 TFLOPs" |
| Compute, matrix unit | "Matrix Cores | 1024"; "Peak Half Precision Matrix (FP16) Performance | 2.5 PFLOPs"; "Peak Single Precision Matrix (FP32) Performance | 157.3 TFLOPs" |
| Memory bandwidth | "Peak Memory Bandwidth | 8 TB/s" |
| Memory capacity | "Dedicated Memory Size | 288 GB", "Dedicated Memory Type | HBM3E" |
| Host link | "Bus Type | PCIe® 5.0 x16" |
| GPU-to-GPU link | "Peak Infinity Fabric™ Link Bandwidth | 144 GB/s" (see B2) |

Two things worth knowing:

- AMD labels its sparse figures explicitly ("with Structured Sparsity") in separate rows, where NVIDIA uses footnotes. Same trap as A9, easier to see.
- On this part AMD's FP32 "Matrix" figure equals its plain FP32 figure (157.3 TFLOPs both), and the page footnotes "*TF32 support through software emulation." There is no TF32-style jump on that row here, unlike the MI300X page quoted in ch01-facts B3.
- The same AMD page prints NVIDIA's B200 the way A8, A10 and A11 found it: "FP32 (Vector) | 75", "Memory Capacity | 180 GB", "Memory Bandwidth | 7.7 TB/s", under the label "B200 SXM5 180GB".

- https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html

---

## Group C: Metal / Apple counterparts

### C1. Tensor Core counterpart in Metal

**Verdict: PARTLY.** No Metal document I read names a matrix-multiply hardware unit or says how shader code would address one. What Metal does document is library-level matrix multiplication inside a kernel, in two generations, and Apple recommends the newer one.

Quotes (MSL spec 4.1):

- Section 6.8: "The header <metal_simdgroup_matrix> defines the SIMD-group Matrix functions. Instead of using simdgroup matrix multiplication, consider using Tensors (section 2.22) and Metal Performance Primitives (section 7). Together, they provide a performant and portable library supporting more general matrix and tensor operations across a wider variety of dimensions and extents."
- Section 7: "All OS: Metal 4 and later support Metal Performance Primitives. Metal Performance Primitives is a library of optimized primitives that are designed to be efficient, portable, and performant on Apple silicon. The header <MetalPerformancePrimitives/MetalPerformancePrimitives.h> defines these functions within the namespace mpp."
- Section 7.2.1: "The template class matmul2d performs a generalized matrix multiplication of two tensors (C = A*B) or matrix multiplication accumulated into a tensor (C = A*B + C)."
- Section 7.1, execution scopes: `execution_thread`, `execution_simdgroup`, `execution_simdgroups<N>` ("the scope of cooperation is N SIMD-groups").
- Table 7.3 lists input and output element types for `matmul2d`: combinations of `char`, `half` and `float` in "Metal 4", `bfloat` combinations in "Metal 4 and OS 26.1", `uchar` with `half` in "Metal 4 and OS 26.4". Mixed `char` by `half` or `float` products are in the table, which is directly relevant to quantised kernels (X2).

On the hardware side, Apple's Mac Studio specification page lists "Neural Accelerators" under the GPU line of the chip ("32-core GPU | Neural Accelerators | Hardware-accelerated ray tracing"). The MSL specification does not contain the phrase. Whether `matmul2d` uses them is not stated in anything I read: COULD NOT VERIFY.

How the projects use this today:

- llama.cpp, `ggml/src/ggml-metal/kernels/mul_mm.metal`: two implementations in one file, chosen by `#ifdef GGML_METAL_HAS_TENSOR`. The tensor path builds `mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(...), execution_simdgroups<N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y>> mm;`. The `#else` path uses `simdgroup_float8x8`, `simdgroup_load`, `simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);` and `simdgroup_store`.
- MLX, `mlx/backend/metal/kernels/steel/gemm/mma.h`: `#include <metal_simdgroup_matrix>`, `typedef metal::simdgroup_matrix<T, kFragRows, kFragCols> mat_type;` and `simdgroup_multiply_accumulate(D, A, B, C);`.

- https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (sections 2.22, 6.8, 7, 7.1, 7.2.1; pages 212 and 349 to 352)
- https://www.apple.com/mac-studio/specs/
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/mul_mm.metal
- https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/steel/gemm/mma.h

### C2. PCIe host-link row on Apple silicon

**Verdict: CONFIRMED that the row has no counterpart.** Settled by ch05-facts C6: "Apple GPUs have a unified memory model in which the CPU and the GPU share system memory", and a shared-mode buffer is visible to both with no copy. There is no host-to-device link figure to put in the table.

This is the one place where the notes' inference (keep weights, KV cache and sampling on the device because the host link grew least) changes shape on Metal: the cost to avoid there is not a bus transfer, it is waiting for a command buffer before the CPU reads results (ch05-facts C6, last paragraph). That contrast is ours, not Apple's and not the book's.

- See ch05-facts C6 for sources.

### C3. NVLink counterpart in Metal

**Verdict: COULD NOT VERIFY any equivalent; none is documented for Apple silicon.** Metal's multi-GPU documentation is about Intel-based Macs and external GPUs, and its one peer-link property is now deprecated.

- "Multi-GPU systems" page: "Your app can submit work to any or all of the GPUs of a system that supports multiple GPUs." and "A Mac may have a Thunderbolt connection to an external GPU and its displays." The pages it sends you to are `finding-multiple-gpus-on-an-intel-based-mac` and `assessing-multi-gpu-and-multi-display-setups-on-an-intel-based-mac`.
- `MTLDevice.peerGroupID`: "The peer group ID the GPU belongs to, if applicable." Platform metadata: macOS, introduced 10.15, deprecated 27.0.
- `MTLCopyAllDevices()`: "Returns an array of all the Metal device instances in the system." Still current.

- https://developer.apple.com/documentation/metal/multi-gpu-systems
- https://developer.apple.com/documentation/metal/mtldevice/peergroupid
- https://developer.apple.com/documentation/metal/mtlcopyalldevices()

### C4. NCCL counterpart for Metal

**Verdict: CONFIRMED** that MLX supplies one at framework level. For Metal itself I found no collective-communication API in the pages read for C3; that is an absence, not a vendor statement. MLX documents distributed communication across machines with four backends, and uses NCCL itself on CUDA.

MLX 0.32.4 docs, "Distributed Communication":

- "MLX supports distributed communication operations that allow the computational cost of training or inference to be shared across many physical machines."
- Backend table: "MPI | A full featured and mature distributed communications library." "RING | Ring all reduce and all gather over TCP sockets. Always available and usually faster than MPI." "JACCL | Low latency communication with RDMA over thunderbolt. Necessary for things like tensor parallelism." "NCCL | The backend of choice for CUDA environments."
- Source directories today under `mlx/distributed/`: `jaccl`, `mpi`, `nccl`, `ring`. Backends under `mlx/backend/`: `common`, `cpu`, `cuda`, `gpu`, `metal`, `no_cpu`, `no_gpu`.

So the nearest thing to "NVLink plus NCCL" on Apple hardware today is Thunderbolt RDMA between separate machines, driven by the framework, not by Metal.

- https://ml-explore.github.io/mlx/build/html/usage/distributed.html
- https://github.com/ml-explore/mlx/tree/main/mlx/distributed
- https://github.com/ml-explore/mlx/tree/main/mlx/backend

### C5. Comparable generational figures for Apple GPUs

**Verdict: COULD NOT VERIFY.** The notes mark this unverified, and it stays that way. Apple's specification page publishes GPU core counts, unified memory sizes and a chip-level "memory bandwidth" figure, and no FLOPS figure of any kind (the word does not appear on the page). There is nothing from Apple to put beside the book's compute row, so the lesson should not attempt a Metal column of numbers.

- Example of what is published: "32-core GPU | Neural Accelerators | Hardware-accelerated ray tracing | 16-core Neural Engine | 460GB/s memory bandwidth" and "36GB unified memory".
- For limits a kernel author needs on Metal (threadgroup memory, threads per threadgroup), see ch05-facts C1 and C3; the reader's own values come from `python3 labs/pmpp.py doctor`.

- https://www.apple.com/mac-studio/specs/

---

## Group D: where the reader goes next (x1, x2, x3): the three codebases today

The chapter's `start` page points at the course's own modules. These items only confirm that what those modules will open still exists at the stated places. Details belong in the x1, x2 and x3 fact checks.

### D1. The three repositories are live

**Verdict: CONFIRMED.** GitHub API today: `ggml-org/llama.cpp` (default branch `master`, last push 2026-11-19, not archived); `ml-explore/mlx` (`main`, last push 2026-11-20, not archived); `pytorch/pytorch` (`main`, last push 2026-11-20, not archived).

- llama.cpp backends are directories under `ggml/src/`: `ggml-cuda`, `ggml-hip`, `ggml-metal` among 18 (also `ggml-vulkan`, `ggml-sycl`, `ggml-opencl`, `ggml-webgpu`, `ggml-musa` and others). See ch05-facts B9 and C10.
- MLX backends: `mlx/backend/metal` and `mlx/backend/cuda` (C4).

- https://github.com/ggml-org/llama.cpp/tree/master/ggml/src
- https://github.com/ml-explore/mlx/tree/main/mlx/backend
- https://github.com/pytorch/pytorch

### D2. PyTorch: where the CUDA and Metal kernels are

**Verdict: CONFIRMED.**

- CUDA kernels: `aten/src/ATen/native/cuda/` (317 entries today, for example `AbsKernel.cu`, `ActivationGeluKernel.cu`).
- Metal: `aten/src/ATen/native/mps/` with subdirectories `kernels` and `operations`; `aten/src/ATen/mps/` holds the runtime side (`MPSDevice.mm`, `MPSAllocator.mm`, `MPSEvent.mm`, `MPSFallback.mm` among 21 files).
- `aten/src/ATen/native/mps/kernels/` holds 98 files of hand-written MSL and headers, including names this course's milestones map onto: `Matmul.metal`, `Attention.metal`, `SoftMax.metal`, `Sort.metal`, `SortTopK.h`, `ScanKernel.metal`, `Quantized.metal`, `ReduceOps.metal`, `HistogramKernel.metal`.

Not checked: which operators go through these `.metal` kernels and which through MPSGraph in `operations/`. That is X3's job.

- https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/cuda
- https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/mps
- https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native/mps/kernels

### D3. PyTorch on AMD: the CUDA sources are translated at build time

**Verdict: CONFIRMED.** Same idea as llama.cpp's `vendors/hip.h` (ch05-facts B9), done with a source-to-source step instead of `#define`s.

- `tools/amd_build/build_amd.py` (last commit 2026-10-20): `from hipify import hipify_python`, description "Top-level script for HIPifying, filling in most common parameters".
- Its `includes` list names the directories it translates, among them `"aten/src/ATen/cuda/*"`, `"aten/src/ATen/native/cuda/*"`, `"aten/src/ATen/native/cudnn/*"`, `"aten/src/ATen/native/sparse/cuda/*"`, `"aten/src/ATen/native/transformers/cuda/attention.cu"`.
- The translator lives in `torch/utils/hipify/` (`hipify_python.py`, `cuda_to_hip_mappings.py`, `constants.py`). This is the "hipify_torch" lineage AMD's HIPIFY docs mention (ch05-facts B8).

- https://github.com/pytorch/pytorch/blob/main/tools/amd_build/build_amd.py
- https://github.com/pytorch/pytorch/tree/main/torch/utils/hipify

### D4. TF32 as the reader will meet it in PyTorch

**Verdict: CONFIRMED.** The book's "Tensor Core FP32" row (A9) corresponds to a switch PyTorch leaves off for matmul by default.

- "Starting in PyTorch 1.7, there is a new flag called allow_tf32. This flag defaults to True in PyTorch 1.7 to PyTorch 1.11, and False in PyTorch 1.12 and later. This flag controls whether PyTorch is allowed to use the TensorFloat32 (TF32) tensor cores, available on NVIDIA GPUs since Ampere, internally to compute matmul (matrix multiplies and batched matrix multiplies) and convolutions."
- Flags as printed: `torch.backends.cuda.matmul.allow_tf32 = True` ("This flag defaults to False in PyTorch 1.12 and later.") and `torch.backends.cudnn.allow_tf32 = True` ("This flag defaults to True.").
- "The precision of matmuls can also be set more broadly (limited not just to CUDA) via set_float32_matmul_precision()."

A good one-line bridge for the page: the table's larger compute number is the mode behind a flag the reader may already have set.

- https://docs.pytorch.org/docs/2.15/notes/cuda.html

---

## Safe statements for the lesson

Each is backed by the item in brackets. Present Fig. 24.1 as the book's figures and put the corrections beside it.

- The book's table compares the first CUDA GPU with a 2025 one. The hardware side of the first is from November 2006; the first CUDA Toolkit is from June 2007 [A1].
- The two compute numbers are theoretical peaks, and the two B200 numbers are different arithmetic: plain FP32, and TF32 on Tensor Cores with sparsity [A2, A8, A9].
- 2200 / 0.345 is about 6,377, so the printed "638x" is off by ten; NVIDIA does publish 2.2 petaFLOPS for that row, labelled TF32 with sparsity, 1.1 dense [A9].
- "1,800 TB/s" should read 1,800 GB/s (1.8 TB/s), and that is both directions together [A13].
- NVIDIA's own documents give 75 TFLOPS FP32 and 180 GB for the B200 as shipped on HGX and DGX boards, and "up to 192 GB" in one technical brief [A8, A11].
- The G80-based board with 1.5 GB used PCI Express Generation 1, so the host link grew about 16x rather than 8x. Either way it grew by far the least [A6, A14].
- On AMD the same roles have other names: matrix cores (MFMA) on CDNA parts only, Infinity Fabric links, RCCL with NCCL's function names [B1, B2, B3].
- On Apple silicon there is no host link to compare and no documented GPU-to-GPU link; matrix multiplication inside a kernel is a library call in MSL [C1, C2, C3].
- The 2025 column is already superseded on NVIDIA's own pages [A15]. Say so in one clause; give no newer numbers.

## Summary of corrections

Items where the book's statement, or a commonly assumed fact, turned out WRONG or PARTLY:

- A1: the G80 reached shops on November 8, 2006. 2007 is the year of CUDA Toolkit 1.0 (June) and of the G80-based Tesla C870, not of the G80.
- A2: 0.345 TFLOPS is not printed by NVIDIA in any source found; it equals 128 multiply-adds per clock x 2 x 1.35 GHz from NVIDIA's own figures. It is a theoretical peak.
- A3, A4: the book's G80 column mixes two boards. 86.4 GB/s belongs to the 768 MB GeForce 8800 GTX; 1.5 GB belongs to the Tesla C870, whose 800 MHz memory clock gives 76.8 GB/s by the same formula.
- A6: the 1.5 GB G80 board is "PCI Express x16 Generation 1" in NVIDIA's board specification, 4 GB/s each way, not Gen2 at 8 GB/s. The PCIe growth is then 16x, not 8x.
- A7: the B200 was announced on March 18, 2024; NVIDIA's first quarter of Blackwell volume sales ended January 26, 2025.
- A8: NVIDIA prints 75 TFLOPS FP32 per GPU for HGX B200. The book's 80 equals the GB200 NVL72 rack figure (5,760 TFLOPS) divided by 72.
- A9: "638x" is wrong; 2200 / 0.345 = 6,377. The 2,200 TFLOPS input is NVIDIA's.
- A9: "Tensor Core FP32" is not an NVIDIA precision. The figure is "TF32 Tensor Core", and 2.2 petaFLOPS is the value with sparsity; dense is 1.1.
- A10: NVIDIA says "Up to 8TB/s" in two documents and 7.7 TB/s in a third.
- A11: shipped B200 boards have 180 GB per GPU (1,440 GB for eight). "Up to 192 GB" appears only in the technical brief, which contradicts its own 1.4 TB system total.
- A13: "1,800 TB/s" is a unit error for 1,800 GB/s. That figure is bidirectional (900 GB/s each way), while the table's PCIe row is each way.
- A14: with both sides taken from NVIDIA documents for one board each, the ratios are about 217x (FP32), 6,366x or 3,183x (TF32 sparse or dense), 104x (memory bandwidth), 120x (capacity), 16x (PCIe).
- A15: the B200 is no longer NVIDIA's newest data-center GPU; B300 (compute capability 10.3) and the Rubin GPU are on NVIDIA's pages today.
- A16: NCCL is not new; its public repository was created on 2015-11-14. Current release v2.32.6-1.
- A17: "since 2017" and "in each training job" could not be verified; a 100,000-GPU LLM training cluster growing to 200,000 is confirmed by NVIDIA.
- A18: the light-source claim is the authors' judgement. The vendor's pages show only that the newest EUV scanners keep the 13.5 nm source and gain resolution from higher numerical aperture.
- A5: pcisig.com could not be read (HTTP 403). PCIe rates are taken from the Linux kernel's `PCIE_SPEED2MBS_ENC` table and cross-checked against an NVIDIA statement.
- B1: AMD's matrix cores are documented for CDNA architectures only; the intrinsic spellings were not fetched. TF32 is "through software emulation" on the MI355X page.
- B2: AMD documents "Infinity Fabric" links on the product page and "xGMI" in RCCL; how the two names relate was not found.
- C1: Apple names no matrix hardware in Metal documentation. The documented route is `<metal_simdgroup_matrix>` and, preferred by Apple since Metal 4, `mpp::tensor_ops::matmul2d`. llama.cpp's `mul_mm.metal` now carries both.
- C3: `MTLDevice.peerGroupID` is deprecated as of macOS 27.0, and Metal's multi-GPU pages concern Intel-based Macs.
- C5: Apple publishes no FLOPS figure, so no Metal column of numbers can be built for the book's table.

## Where the backends really differ

This chapter has no code, so the rows are the hardware and library terms its one table uses. Every cell cites an item above or in ch05-facts.

| Feature | CUDA (NVIDIA) | HIP (AMD) | Metal (Apple) |
|---|---|---|---|
| Name of the dedicated matrix-multiply hardware | Tensor Cores [A9] | Matrix cores, formally MFMA units; documented for CDNA only [B1] | None named in Metal docs. The Mac spec page lists "Neural Accelerators" under the GPU; link to MSL not stated [C1] |
| How kernel code reaches it | CUDA Tile, `cuda::tiles` in `cuda_tile.h`, Toolkit 13.3 onward; or libraries (cuBLAS, CUTLASS) [A19, ch05-facts A9] | "optimized libraries, compiler intrinsics, and/or inline assembly"; no tile-level model found [B1] | MSL library calls: `mpp::tensor_ops::matmul2d` (Metal 4 and later, Apple's recommendation) or `<metal_simdgroup_matrix>` [C1] |
| Reduced-mantissa FP32 matmul mode | TF32: inputs rounded to 10 mantissa bits, FP32 accumulate; off by default for matmul in PyTorch [A9, D4] | Depends on the part: MI355X page says "TF32 support through software emulation" and shows no TF32 row; MI300X page prints a "TF32 Matrix" row [B1, B4, ch01-facts B3] | None found; `matmul2d` types are `char`, `uchar`, `half`, `bfloat`, `float` [C1] |
| How peak figures are qualified | Footnotes: "Specification in Sparse. Dense is ½ sparse spec shown." [A9] | Separate rows "with Structured Sparsity" [B4] | No FLOPS published at all [C5] |
| Host-to-device link | PCIe Gen5 x16 per GPU on HGX B200 [A12] | "PCIe® 5.0 x16" on MI355X [B4] | None: unified memory, shared buffers with no copy [C2, ch05-facts C6] |
| GPU-to-GPU link | NVLink: 1,800 GB/s per GPU both directions on Blackwell [A13] | Infinity Fabric links, 144 GB/s peak per link on MI355X; "xGMI" in RCCL [B2] | None documented for Apple silicon; `peerGroupID` deprecated in macOS 27.0 [C3] |
| Multi-GPU collective library | NCCL 2.32.x, public since 2015 [A16] | RCCL 2.32.3, same `ncclXxx` function names [B3] | None found in Metal. MLX `mx.distributed`: MPI, ring over TCP, JACCL (RDMA over Thunderbolt), NCCL on CUDA [C4] |
| How a framework gets its kernels onto this backend | Native `.cu` sources (PyTorch `aten/src/ATen/native/cuda`) [D2] | The CUDA sources, translated: PyTorch hipify at build time, llama.cpp `vendors/hip.h` [D3, ch05-facts B9] | Separate hand-written MSL: PyTorch `native/mps/kernels/*.metal`, llama.cpp `ggml-metal/kernels`, MLX `backend/metal/kernels` [D2, ch05-facts C10] |
