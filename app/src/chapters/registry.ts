import type { Chapter, PlanRow } from './types';
import { ch05 } from './ch05';

/** Chapters that exist in the app. */
export const CHAPTERS: Chapter[] = [ch05];
export const chapterById = (id: string) => CHAPTERS.find((c) => c.id === id);

/** The whole curriculum, as in PLAN.md. */
export const PLAN: PlanRow[] = [
  { num: '1', title: 'Introduction', pages: '1–20', tier: 'Condensed', why: 'Motivation; keep Amdahl and the four challenges.' },
  { num: '2', title: 'Heterogeneous data-parallel computing', pages: '21–44', tier: 'Deep', why: 'First kernel, host and device, launch syntax.', milestone: 'M0 vector ops' },
  { num: '3', title: 'Multidimensional grids and data', pages: '45–66', tier: 'Deep', why: 'Thread-to-data mapping, the naive matmul.', milestone: 'M1 naive GEMM' },
  { num: '4', title: 'Compute architecture and scheduling', pages: '67–92', tier: 'Deep', why: 'Warps, divergence, occupancy: the execution model.', milestone: 'M1 launch from device queries' },
  { num: '5', title: 'Memory architecture and data locality', pages: '93–121', tier: 'Deep', why: 'FLOP per byte, roofline, memory types, tiling.', milestone: 'M2 tiled GEMM', built: true },
  { num: '6', title: 'Performance considerations', pages: '123–155', tier: 'Deep', why: 'Coalescing, bank conflicts, the optimisation checklist.', milestone: 'M2 tuned' },
  { num: '7', title: 'Convolution', pages: '159–181', tier: 'Standard', why: 'Constant memory, halo tiling.' },
  { num: '8', title: 'Stencil computation', pages: '183–199', tier: 'Condensed', why: 'Keep thread coarsening and register tiling.' },
  { num: '9', title: 'Histogram', pages: '201–220', tier: 'Standard', why: 'Atomics, contention, privatisation.', milestone: 'M5 token counts' },
  { num: '10', title: 'Reduction', pages: '221–250', tier: 'Deep', why: 'Softmax normaliser, max, argmax.', milestone: 'M4 softmax and argmax' },
  { num: '11', title: 'Scan', pages: '251–287', tier: 'Deep', why: 'Cumulative probability for top-p.', milestone: 'M5 top-p' },
  { num: '12', title: 'Filter', pages: '289–302', tier: 'Deep', why: 'Compaction of sampling candidates.', milestone: 'M5 top-k' },
  { num: '13', title: 'Merge', pages: '303–328', tier: 'Condensed', why: 'Low relevance; keep co-rank.' },
  { num: '14', title: 'Sorting', pages: '329–348', tier: 'Standard', why: 'Radix sort; sorting logits.', milestone: 'M5 sorted candidates' },
  { num: '15', title: 'Advanced optimizations for matrix multiplication', pages: '349–370', tier: 'Deep', why: 'From tiled GEMM to a library-grade kernel.', milestone: 'M3 fast GEMM' },
  { num: '16', title: 'Dynamic programming and wavefront parallelism', pages: '373–400', tier: 'Condensed', why: 'Keep dependency-driven synchronisation.' },
  { num: '17', title: 'Sparse matrix computation', pages: '401–423', tier: 'Condensed', why: 'Keep format trade-offs.' },
  { num: '18', title: 'Graph traversal', pages: '425–451', tier: 'Condensed', why: 'Keep frontiers and privatised queues.' },
  { num: '19', title: 'Convolutional neural networks', pages: '453–475', tier: 'Standard', why: 'Lowering to GEMM: how frameworks use libraries.' },
  { num: '20', title: 'Large language models', pages: '477–512', tier: 'Deep', why: 'Attention kernels, KV cache, flash attention.', milestone: 'M6 attention, M7 KV cache' },
  { num: '21', title: 'Electrostatic potential map', pages: '513–528', tier: 'Condensed', why: 'Keep scatter versus gather.' },
  { num: '22', title: 'Algorithm selection and problem formulation', pages: '529–540', tier: 'Standard', why: 'Amdahl; batching as latency versus throughput.', milestone: 'M8 batching study' },
  { num: '23', title: 'Multi-GPU programming', pages: '541–576', tier: 'Condensed', why: 'Keep streams, async copies, pinned memory.' },
  { num: '24', title: 'Conclusion and future outlook', pages: '577–580', tier: 'Condensed', why: 'Framing only.' },
  { num: 'A', title: 'Numerical considerations', pages: '583–600', tier: 'Standard', why: 'Summation order, lower-precision formats.' },
  { num: 'B', title: 'Deep learning basics', pages: '601–622', tier: 'Skipped', why: 'You already know all of it.' },
  { num: 'C', title: 'CUDA memories, address spaces, and pointers', pages: '623–629', tier: 'Standard', why: 'Memory models; the contrast with unified memory.' },
  { num: 'X1', title: 'Sampling kernels', pages: 'not in the book', tier: 'Deep', why: 'Greedy, top-k, top-p on the GPU.', milestone: 'M5', ours: true },
  { num: 'X2', title: 'Quantised kernels', pages: 'not in the book', tier: 'Deep', why: 'int8 and 4-bit-weight GEMM.', milestone: 'M9', ours: true },
  { num: 'X3', title: 'Reading real backends', pages: 'not in the book', tier: 'Standard', why: 'llama.cpp, MLX and PyTorch GPU code.', ours: true },
];
