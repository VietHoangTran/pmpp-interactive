# Plan: an interactive PMPP for one reader

Source: *Programming Massively Parallel Processors*, 5th edition (Hwu, Kirk, El Hajj).
The whole PDF (673 pages) was read page by page as images; structured notes for every
chapter and appendix are in `notes/`. Everything below is derived from those notes.

## 1. The reader and the destination

Senior AI/ML engineer: fluent in C++, Python, TypeScript and PyTorch-level ML; new to GPU
and kernel programming. Goals, in priority order:

1. Write and optimise kernels that speed up LLM inference: matmul, attention, quantised
   ops, sampling.
2. Do it on non-NVIDIA backends as well as CUDA.
3. Read and contribute to the GPU backends of PyTorch, MLX and llama.cpp.

Consequences for the course:

- No general programming or ML basics. Effort goes to the execution model, the memory
  hierarchy and performance reasoning.
- Every chapter bridges from the framework level ("what happens underneath `torch.matmul`").
- CUDA is the primary language, as in the book. HIP and Metal appear where they teach
  something, with the real differences called out, not as mechanical triplication.

## 2. What the book does and does not give this reader

Found while reading, and decisive for the plan:

- **Matmul is covered very well**: chapters 3, 5, 6 and the new chapter 15 take it from a
  naive kernel to the skeleton of a library GEMM.
- **Attention is covered**: chapter 20 gives a mask+softmax kernel and a full single-head
  FlashAttention-style kernel.
- **Sampling is not in the book at all.** No top-k, top-p or argmax. The building blocks
  are there (reduction ch. 10, scan ch. 11, filter ch. 12, sort ch. 14).
- **Quantised kernels are not in the book.** Appendix A §A.5 is two paragraphs on
  lower-precision formats. No integer or mixed-precision kernel appears anywhere.
- **HIP and Metal are never mentioned.** Every portability statement in this course is
  ours and is checked against current vendor docs and, for Metal, run on a real device.
- **No measured timings anywhere** in chapters 15 or 20; benefits are analytic. Labs must
  produce their own numbers on the reader's devices.
- **Code listings are images** with a meaningful number of apparent errata (each chapter's
  notes end with an "Unclear in the PDF" list). Lessons present corrected, compiled code
  and say where it differs from the page.

So two modules must be added beyond the book, clearly labelled as such: **sampling
kernels** and **quantised kernels**. Both are built from patterns the book does teach.

## 3. Curriculum map

Depth tiers: **Deep** (full lessons, several interactives, 3+ labs), **Standard** (full
lessons, 1 to 2 labs), **Condensed** (one lesson that keeps the transferable technique and
drops the domain material, one optional lab). "Milestone" is the through-line project
step (section 5).

| Ch. | Title (book pp.) | Tier | Why | Milestone |
|---|---|---|---|---|
| 1 | Introduction (1–20) | Condensed | Motivation only; keep Amdahl and the four challenges | none |
| 2 | Heterogeneous data-parallel computing (21–44) | Deep | First kernel, host/device split, launch syntax; where unified memory first differs | M0 vector ops |
| 3 | Multidimensional grids and data (45–66) | Deep | Thread-to-data mapping, row-major indexing, naive matmul | M1 naive GEMM |
| 4 | Compute architecture and scheduling (67–92) | Deep | Warps, divergence, occupancy, device queries: the execution model | M1 launch config from device queries |
| 5 | Memory architecture and data locality (93–121) | Deep, **pilot** | FLOP/B, roofline, memory types, tiling | M2 tiled GEMM |
| 6 | Performance considerations (123–155) | Deep | Coalescing, bank conflicts, coarsening, the optimisation checklist | M2 measured and tuned |
| 7 | Convolution (159–181) | Standard | Constant memory, halo tiling, intensity as a function of tile size | none |
| 8 | Stencil (183–199) | Condensed | Keep thread coarsening and register tiling; drop PDE background | none |
| 9 | Histogram (201–220) | Standard | Atomics, contention, privatisation: needed before any shared accumulator | M5 token counts for penalties |
| 10 | Reduction (221–250) | Deep | Softmax normaliser, max, argmax; warp shuffles | M4 softmax + argmax |
| 11 | Scan (251–287) | Deep | Cumulative probability for top-p; work efficiency | M5 top-p cumulative sum |
| 12 | Filter (289–302) | Deep | Compaction of candidates; warp voting | M5 top-k / top-p compaction |
| 13 | Merge (303–328) | Condensed | Low relevance; keep co-rank and staging irregular reads | none |
| 14 | Sorting (329–348) | Standard | Radix sort as stable partition; sorting logits | M5 sorted candidates |
| 15 | Advanced matmul (349–370) | Deep | Coarsening, register tiling, vector stores, bank conflicts, pipelining | M3 fast GEMM |
| 16 | Wavefront parallelism (373–400) | Condensed | Keep dependency-driven sync and tile skewing; drop genomics | none |
| 17 | Sparse matrices (401–423) | Condensed | Keep format trade-offs and padding vs divergence | none |
| 18 | Graph traversal (425–451) | Condensed | Keep frontiers, privatised queues, cooperative-groups launch | none |
| 19 | CNNs (453–475) | Standard | "Lower everything to GEMM": how frameworks use libraries | none |
| 20 | Large language models (477–512) | Deep | Attention kernels, KV cache, flash attention, batching and intensity | M6 attention, M7 KV cache |
| 21 | Electrostatic potential map (513–528) | Condensed | Keep scatter vs gather and coalescing-aware output assignment | none |
| 22 | Algorithm selection (529–540) | Standard | Amdahl, gather/scatter choice, §22.5 batching: latency vs throughput | M8 batching study |
| 23 | Multi-GPU (541–576) | Condensed | Keep §23.3 streams, async copies, pinned memory; drop MPI/NCCL/NVSHMEM | none |
| 24 | Conclusion (577–580) | Condensed | Framing only | none |
| A | Numerical considerations (583–600) | Standard | Summation order, softmax stability, §A.5 as the anchor for quantisation | feeds X2 |
| B | Deep learning basics (601–622) | Skipped | The reader already knows all of it; kept as a reference link | none |
| C | CUDA memories and address spaces (623–629) | Standard | The book's memory-model history; the contrast with Apple unified memory | feeds portability track |
| X1 | Sampling kernels (not in the book) | Deep | Goal 1. Built from chs. 10, 11, 12, 14 | M5 |
| X2 | Quantised kernels (not in the book) | Deep | Goal 1. Anchored on A.5 and ch. 15's intensity analysis | M9 int8 / 4-bit GEMM |
| X3 | Reading real backends (not in the book) | Standard | Goal 3. Guided reading of llama.cpp, MLX and PyTorch GPU code | none |

### Learning objectives per chapter

After each chapter the reader can:

- **1** state Amdahl's law with numbers; name the four challenges the book returns to.
- **2** write, launch and check a 1D kernel; explain what each of `cudaMalloc`/`cudaMemcpy`
  does and what replaces them under unified memory; compute a grid for any `n`.
- **3** map 2D/3D thread indices to row-major data; write the naive matmul kernel and its
  launch for any width; predict which threads are surplus.
- **4** predict how a block is cut into warps; count divergent warps for a boundary check;
  compute occupancy from registers, shared memory and block size; query all of it.
- **5** compute FLOP/B for a kernel and place it on a roofline built from their own
  measurements; say where each variable lives; write a tiled, boundary-checked GEMM with
  the right barriers; budget shared memory against occupancy.
- **6** tell coalesced from uncoalesced access patterns and fix them; detect and remove
  bank conflicts; apply the checklist to a kernel and justify each step with a measurement.
- **7** use constant memory; derive intensity for a halo-tiled kernel as a function of tile
  size and filter radius.
- **8** apply thread coarsening and register tiling to a kernel that is not a stencil.
- **9** choose between atomics, block-private and thread-private accumulators from
  contention estimates.
- **10** write a divergence-free block reduction and a warp-shuffle reduction; build
  numerically safe softmax and argmax from them.
- **11** write a block-level and a multi-block scan; argue work efficiency.
- **12** write stable and unstable compaction; explain warp voting.
- **13** explain co-rank and when value-dependent ranges force it.
- **14** implement one radix pass as scan plus scatter; choose a sort for logits.
- **15** take a tiled GEMM to a coarsened, register-tiled, conflict-free kernel and explain
  the occupancy it trades away; say what tensor cores and async copies add.
- **16–18, 21** recognise each pattern and name the technique that transfers.
- **19** re-express convolution as GEMM and explain what the library does instead.
- **20** write a mask+softmax kernel and a tiled attention kernel; explain why decode is
  bandwidth-bound and prefill compute-bound; size a KV cache.
- **22** decide gather vs scatter; reason about batching as latency vs throughput.
- **23** overlap work using streams and async copies; explain pinned memory.
- **A** explain why summation order changes results; choose accumulator width.
- **C** explain each CUDA memory model and how unified memory on Apple silicon differs.
- **X1** build greedy, top-k and top-p sampling on the GPU and test them against a CPU
  reference.
- **X2** write an int8 and a 4-bit-weight GEMM; measure the bandwidth win.
- **X3** find the kernel for a given op in each of three real codebases and explain it.

### Spaced review

Every chapter contributes reasoning questions to a review pool (Leitner boxes, intervals
of 1, 3, 7, 14, 30 days). Each chapter opens with a short prerequisite check drawn from
earlier chapters, and the home page shows what is due. Chapter 5 already carries review
items from chapters 2 to 4.

## 4. Backends and verification

| Backend | Built on | How a lab solution is verified |
|---|---|---|
| Metal | macOS | compiled and run on the device here |
| CUDA | Linux + CUDA toolkit | on the reader's CUDA machine via `./verify.sh` |
| HIP | Linux + ROCm | on the reader's ROCm machine via `./verify.sh` |
| emu | anywhere | CPU emulator of the CUDA subset: runs the same `.cu` kernels |

On the Mac, CUDA and HIP get two real checks short of a device:

1. **CPU emulation** (`labs/common/include/pmpp/emu/cuda_emu.h`): the unmodified `.cu`
   kernel runs on the CPU with real `__shared__` and `__syncthreads()` semantics under
   three deterministic thread schedules, with AddressSanitizer. It checks logic, never
   performance. AMD's HIP-CPU was tried first and rejected: it needs GCC and TBB on macOS,
   fails to compile with Apple clang, and has been unmaintained since March 2024.
2. **CUDA front-end check**: clang's CUDA mode parses every `.cu` file against stub
   declarations, which catches launch-syntax and host/device-rule errors.

HIP builds compile the same `.cu` source through a header of name mappings, the approach
llama.cpp uses. That header is checked against AMD's docs but not compiled here.

Status vocabulary, used everywhere: **verified on Metal**, **emulator-checked**,
**front-end-checked**, **unverified**. `./verify.sh` writes
`reports/verify-<time>-<os>-<backends>.md`; a reference solution is not relied on for a
backend until a report shows it passing there.

Device numbers are never hard-coded. The book's H100 figures appear only as the book's
worked example; "your device" values come from `labs/pmpp.py doctor` and lab benchmarks.

## 5. The through-line project: `nanoinfer`

One small GPU inference path for a decoder-only transformer, grown chapter by chapter.
Every milestone is a lab with tests against a CPU reference and a benchmark, on every
available backend.

| Milestone | From | Adds |
|---|---|---|
| M0 | ch. 2 | elementwise ops: bias add, residual add, scale |
| M1 | chs. 3–4 | naive GEMM for a linear layer; launch configuration from device queries |
| M2 | chs. 5–6 | tiled, boundary-checked rectangular GEMM (**built in the pilot as lab 5.3**), then coalescing fixes |
| M3 | ch. 15 | coarsened, register-tiled GEMM; compare with the vendor library |
| M4 | ch. 10 | row-wise softmax, max and argmax: greedy decoding works end to end |
| M5 | chs. 9, 11, 12, 14, X1 | top-k and top-p sampling on the GPU, repetition penalty |
| M6 | ch. 20 | mask+softmax kernel, then a tiled attention kernel |
| M7 | ch. 20 | KV cache: single-token decode path |
| M8 | ch. 22 | batching study: latency vs throughput on the reader's device |
| M9 | A, X2 | int8 and 4-bit-weight GEMM; measure the decode speed-up |
| M10 | X3 | run a small real checkpoint and compare tokens with a reference implementation |

## 6. Stack and layout

- **App**: TypeScript, React, Vite. One Node process serves the app and a small API that
  reads and writes `progress/progress.json` and reads `reports/`. Start: `./start.sh`.
- **Labs**: C++17, CMake 3.21+, Python 3 (standard library only) for orchestration.
  `./verify.sh` builds, tests and benchmarks everything the machine supports.
- **Progress**: one JSON file with stable, sorted keys so it syncs and merges through git.

```
app/                 the web app (src/chapters/chNN holds each chapter's lessons)
content/chNN/narration/   narration scripts (.txt), audio (.mp3), manifest.json
labs/                build system, runtime, emulator, labs/chNN/labM_name/{starter,work,solution}
notes/               chapter notes, page map, docs-checked facts (notes/today/)
progress/            progress.json (yours; sync it)
reports/             verify reports (commit the ones you want to share)
tools/               narration manifest tool
```

## 7. Interactive formats

Chosen per idea, not per chapter. The set, with the pilot's instances:

| Format | What the reader does | Pilot instance |
|---|---|---|
| Predict then reveal | commit a number or a choice before seeing the answer | FLOP/B of kernels; speed-up before running lab 5.2 |
| Explorer | manipulate parameters and watch a faithful model respond | roofline; duplicate reads; boundary explorer; occupancy budget |
| Stepper | step a kernel line by line across threads | tiled kernel stepper |
| Bug hunt | break or repair a kernel and find the failing schedule | barrier race hunt; lab 5.4 |
| Classification | place items in categories with feedback | "where does this variable live?" |
| Narrated animation | watch a short programmatic animation with narration and captions | "why tiling works" |
| Erratum hunt | find what is wrong in a listing as printed | Fig. 5.14 |
| Lab | write code, run tests, record measurements against a prediction | labs 5.1 to 5.4 |
| Spaced review | answer due questions from earlier material | prerequisite check and review page |

Every simulation states what it simplifies. Every lesson carries its section and page.
"Today" notes are checked against current docs and cite `notes/today/`.

## 8. Narration and video

- Generated only with the ElevenLabs MCP text-to-speech tool, model `eleven_v4`, voice
  `tnSpp4vdxKPjI9w0GnoV`. (`eleven_v4_turbo` was requested first; the MCP server does not
  offer it, and the reader chose `eleven_v4` on 2026-11-20.) No other model or voice.
- Scripts are written for the ear, one `.txt` per clip beside its `.mp3`.
  `content/chNN/narration/manifest.json` maps each clip to the SHA-256 of its script, so
  only clips whose script changed are regenerated (`node tools/narration.mjs status`).
- Audio is committed, so the app plays it anywhere without ElevenLabs access.
- "Video" is a programmatic animation (SVG driven by a timeline) split into beats, one
  clip per beat, so picture and sound stay in sync and both stay editable. Captions always
  show; with audio missing the animation runs on estimated timings.

## 9. Order of work

1. Read the book, notes, this plan. **Done.**
2. Pilot: chapter 5 to the finished standard (lessons, interactives, narration, four labs,
   verify command). **In progress.**
3. Stop for review. The reader runs `./verify.sh` on a CUDA and a ROCm machine and brings
   the reports back; fixes follow from those.
4. After approval, the remaining chapters in this order, each built by a subagent from
   the notes with chapter 5 as the template, then reviewed for accuracy and consistency:
   2, 3, 4, 6 (foundations) → 15 (matmul) → 10, 11, 12, 14, 9, X1 (sampling) →
   20, 22 (attention, batching) → A, X2 (quantisation) → 7, 19, C → X3 →
   condensed chapters 1, 8, 13, 16, 17, 18, 21, 23, 24.

## 10. Decisions for the reader

Recorded as they are made in `PROGRESS.md`. Open ones are listed in the hand-off message
at the end of the pilot.
