// Chapter 6, section 6.8: the book's checklist of optimisations (Fig. 6.13, p. 148)
// as data, and the logic of the three interactives built on it: the decision tool
// (pick a symptom, the rows that relieve it rise), the sorting drill (twelve cards into
// four categories) and the scenario cards.
//
// Sources (the lesson cites them where they are used):
//  - notes/ch06.md, section 6.8: the table reproduced in our wording, the per-row
//    commentary of pp. 149 to 153, and "Unclear in the PDF" items 13 and 14;
//  - notes/ch06.md, sections 6.1 to 6.7 and 6.9, for the costs the chapter states;
//  - notes/today/ch06-facts.md A11 (register pressure of vector loads, alignment) and
//    D1 to D4 (what llama.cpp, MLX and PyTorch do), the only sources for "seen in";
//  - notes/today/infra-facts.md, "Where the backends really differ" (lane masks), for
//    the portability cost of warp-level primitives.
//
// What is ours, not the book's: the "what it costs" column where it says so, the
// "seen in" column, the symptom list and its ranking, the "why this category"
// sentences, and the scenarios' wording. Each states its source.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

export type Category = 'compute' | 'memory' | 'sync' | 'general';
export const CATEGORIES: readonly Category[] = ['compute', 'memory', 'sync', 'general'];
export const CATEGORY_NAME: Record<Category, string> = {
  compute: 'Compute utilisation',
  memory: 'Memory utilisation',
  sync: 'Synchronisation latency',
  general: 'General',
};
/** What each category targets (p. 149), in our words. */
export const CATEGORY_AIM: Record<Category, string> = {
  compute: 'keep the cores busy with useful work',
  memory: 'make the memory system do less, or do it more efficiently',
  sync: 'cut the time threads spend waiting for each other at barriers',
  general: 'which of the above it helps depends on the kernel',
};

export type RowId =
  | 'occupancy'
  | 'unroll'
  | 'divergence'
  | 'coalesce'
  | 'smem-tiling'
  | 'reg-tiling'
  | 'vector'
  | 'banks'
  | 'privatise'
  | 'warp'
  | 'double-buffer'
  | 'coarsen';

export interface Cost {
  text: string;
  /** Where the cost is stated ("p. 153", "facts A11"), or null: then the cost is "not stated in the book". */
  source: string | null;
}

export interface SeenIn {
  /** Project and file, as the facts file names them. */
  where: string;
  what: string;
  /** The facts item, in notes/today/ch06-facts.md. */
  fact: 'D1' | 'D2' | 'D3' | 'D4';
  /** True when the facts item shows the technique is NOT used there. */
  absent?: boolean;
  /** Text that appears in that facts item, so a test can tie the entry to it. */
  token: string;
}

export interface Row {
  id: RowId;
  category: Category;
  name: string;
  /** Book's column "benefit to compute cores", our wording. */
  benefitCompute: string;
  /** Book's column "benefit to memory", our wording; null where the table lists none. */
  benefitMemory: string | null;
  strategies: string[];
  /** Pages of the table row and of the row's commentary in the text. */
  pages: string;
  /** The text's commentary for the row (pp. 149 to 153), in our words. */
  commentary: string;
  cost: Cost;
  /** Later chapters where the book says the row is used again. */
  returnsIn: { ch: number; what: string }[];
  /** Real backends, only from facts D1 to D4. Empty means "not checked". */
  seenIn: SeenIn[];
  /** Where the course teaches it, as [chapter id, section id]. */
  taught: [string, string][];
  /** Why the book files it under its category (primary intent, p. 149). Ours. */
  why: string;
}

export const ROWS: readonly Row[] = [
  {
    id: 'occupancy',
    category: 'compute',
    name: 'Occupancy tuning',
    benefitCompute: 'More resident work to cover pipeline latency',
    benefitMemory: 'More memory requests in flight, which covers DRAM latency',
    strategies: ['Tune what each block takes from the SM: threads per block, shared memory per block, registers per thread'],
    pages: '148–149',
    commentary:
      'Chapter 4 introduced it: keep many more threads than cores so there is always something to run. Chapter 5 added shared memory as a third budget, and section 6.2 added that resident warps also keep the memory system busy. From here on the book assumes you keep occupancy healthy whenever an optimisation spends registers or shared memory.',
    cost: {
      text: 'Caps the registers and shared memory each thread or block can use, which other optimisations want; chapter 15 deliberately gives occupancy up when other optimisations make up for it.',
      source: 'p. 149',
    },
    returnsIn: [{ ch: 15, what: 'more resources per thread, lower occupancy, compensated by other optimisations' }],
    seenIn: [],
    taught: [['ch04', '4.7'], ['ch05', '5.6'], ['ch06', '6.2']],
    why: 'Its first aim is that cores always have a ready warp. More requests in flight is a welcome side effect.',
  },
  {
    id: 'unroll',
    category: 'compute',
    name: 'Loop unrolling',
    benefitCompute: 'Fewer branch instructions, and more independent instructions to schedule, so fewer stalls',
    benefitMemory: 'Can let local arrays be promoted to registers, which removes their memory traffic',
    strategies: ['Mostly the compiler does it. Help it by giving loops constant bounds where you can'],
    pages: '148–149',
    commentary:
      'A recap of section 6.6. The book assumes the compiler unrolls well enough. Chapters 11 and 15 make it essential for register tiling after coarsening, and chapter 15 also uses it to make up for low occupancy.',
    cost: { text: 'Not stated in the book.', source: null },
    returnsIn: [
      { ch: 11, what: 'register tiling after coarsening, in prefix sum' },
      { ch: 15, what: 'register tiling, and making up for low occupancy' },
    ],
    seenIn: [
      { where: 'MLX, steel/defines.h', what: 'STEEL_PRAGMA_UNROLL, a full-unroll loop pragma used in its Metal GEMM', fact: 'D3', token: 'STEEL_PRAGMA_UNROLL' },
      { where: 'llama.cpp, Metal kernels/common.h', what: 'FOR_UNROLL, the same Clang loop pragma', fact: 'D2', token: 'FOR_UNROLL' },
    ],
    taught: [['ch06', '6.6']],
    why: 'Its first aim is fewer stalls and branches in the instruction stream. Register promotion is a bonus.',
  },
  {
    id: 'divergence',
    category: 'compute',
    name: 'Reducing control divergence',
    benefitCompute: 'High SIMD efficiency: fewer lanes idle while a warp executes',
    benefitMemory: null,
    strategies: ['Reassign threads to work or data', 'Change the data layout'],
    pages: '148–150',
    commentary:
      'Part 1 kernels diverged only at boundaries. Later chapters reassign work so that one warp is fully used before the next (reduction), give the threads of a warp similar amounts of work (graph traversal), or reorder data so that one warp gets similar rows (the JDS sparse format).',
    cost: { text: 'Not stated in the book.', source: null },
    returnsIn: [
      { ch: 10, what: 'reduction: use one warp fully before the next' },
      { ch: 17, what: 'sparse matrices: the JDS layout' },
      { ch: 18, what: 'graph traversal: vertex-centric versus edge-centric' },
    ],
    seenIn: [],
    taught: [['ch04', '4.5']],
    why: 'Idle SIMD lanes are idle cores. The table lists no benefit to memory at all.',
  },
  {
    id: 'coalesce',
    category: 'memory',
    name: 'Coalesced global memory accesses',
    benefitCompute: 'Fewer stalls waiting on global memory',
    benefitMemory: 'Less global traffic, and fuller use of each burst and cache line',
    strategies: [
      'Remap threads to data',
      'Change the layout: transpose, or turn an array of structures into a structure of arrays',
      'Move data between global and shared memory in a coalesced order and do the irregular access in shared memory (corner turning, packing)',
    ],
    pages: '148, 150–151',
    commentary:
      'The threads of a warp should touch neighbouring addresses in the same burst or line. For irregular patterns: stage through shared memory (corner turning in 6.1; packing in filter and sorting; a shared copy to binary-search in merge), remap threads (reduction, chapter 15), or change the layout (ELL and JDS, a transpose, structure of arrays).',
    cost: {
      text: "Corner turning's store into shared memory is strided and can turn into a bank conflict, which is the next row's problem.",
      source: 'pp. 139–140',
    },
    returnsIn: [
      { ch: 10, what: 'remapping threads to data' },
      { ch: 12, what: 'packing scattered writes in shared memory' },
      { ch: 13, what: 'coalesced load, then binary search in shared memory' },
      { ch: 14, what: 'packing' },
      { ch: 15, what: 'remapping threads to data' },
      { ch: 17, what: 'ELL and JDS layouts' },
    ],
    seenIn: [
      {
        where: 'MLX, steel/gemm/gemm.h',
        what: 'a transposed operand is loaded along its contiguous direction and the tile is indexed to match: corner turning',
        fact: 'D3',
        token: 'loading along the contiguous direction',
      },
    ],
    taught: [['ch06', '6.1']],
    why: 'It changes how the memory system is used. Fewer stalls follows from that.',
  },
  {
    id: 'smem-tiling',
    category: 'memory',
    name: 'Shared memory tiling',
    benefitCompute: 'Fewer stalls waiting on global memory',
    benefitMemory: 'Less global traffic',
    strategies: ['Keep data a block reuses in shared memory, so it crosses between global memory and the SM once'],
    pages: '148, 151',
    commentary:
      'Chapter 5 with matmul. It returns in most patterns: with input and output tiles of different shapes in convolution and stencil, and in more elaborate forms in chapter 15 (matmul) and chapter 16 (sequence alignment).',
    cost: { text: 'Spends shared memory, and can lower occupancy when shared memory is what limits it.', source: 'p. 153' },
    returnsIn: [
      { ch: 7, what: 'convolution: input and output tiles of different sizes' },
      { ch: 8, what: 'stencil' },
      { ch: 15, what: 'matmul, more elaborate' },
      { ch: 16, what: 'sequence alignment' },
    ],
    seenIn: [
      { where: 'MLX, steel/gemm/gemm.h', what: 'GEMM operands staged in threadgroup-memory tiles', fact: 'D3', token: 'threadgroup tiles' },
    ],
    taught: [['ch05', '5.3'], ['ch05', '5.4']],
    why: 'It exists to cut global traffic. It is the canonical memory optimisation.',
  },
  {
    id: 'reg-tiling',
    category: 'memory',
    name: 'Register tiling',
    benefitCompute: 'Fewer stalls waiting on shared memory',
    benefitMemory: 'Less shared memory traffic',
    strategies: ['Keep data a warp or a thread reuses in registers, so it crosses between shared memory and registers once'],
    pages: '148, 151',
    commentary:
      'Already in use: each thread of the chapter 3 and 5 kernels sums into a register and writes once, so the registers of all threads form a tile of the output. It returns in stencil (a shared tile moved to registers and back) and in prefix sum and chapter 15 (each thread owns several outputs and reuses several inputs).',
    cost: { text: 'Not stated in the book.', source: null },
    returnsIn: [
      { ch: 8, what: 'stencil: shared tile into registers and back' },
      { ch: 11, what: 'prefix sum: several outputs per thread' },
      { ch: 15, what: 'matmul: several outputs and inputs per thread' },
    ],
    seenIn: [],
    taught: [['ch05', '5.4']],
    why: 'It removes traffic, this time between shared memory and registers.',
  },
  {
    id: 'vector',
    category: 'memory',
    name: 'Vector loads and stores',
    benefitCompute: 'Fewer instructions spent managing memory accesses',
    benefitMemory: 'More bytes in flight per request, which covers DRAM latency',
    strategies: ['Have each thread move 16-byte chunks of contiguous data'],
    pages: '139, 148, 151',
    commentary:
      'It applies to most computations in the book, and because it is simple the book only mentions it where it matters most. Section 6.3 adds the second benefit: a kernel that cannot run at full occupancy still keeps enough bytes in flight. Chapter 15 has such a kernel.',
    cost: {
      text: 'The tail that is not a multiple of four needs scalar code (pp. 138–139). Vendor addition: wider loads raise register pressure, and the pointer must be aligned to the vector size.',
      source: 'pp. 138–139; facts A11',
    },
    returnsIn: [{ ch: 15, what: 'a kernel below full occupancy' }],
    seenIn: [
      {
        where: 'PyTorch, aten/src/ATen/native/cuda/MemoryAccess.cuh',
        what: 'aligned_vector loads, with the vector width chosen at run time from the pointer alignment',
        fact: 'D4',
        token: 'can_vectorize_up_to',
      },
      { where: 'llama.cpp, ggml-cuda/common.cuh', what: '16-byte copies through int4 on CUDA and HIP', fact: 'D1', token: 'ggml_cuda_memcpy_1' },
    ],
    taught: [['ch06', '6.3']],
    why: 'It changes how the memory system is asked: fewer, wider requests.',
  },
  {
    id: 'banks',
    category: 'memory',
    name: 'Avoiding shared memory bank conflicts',
    benefitCompute: 'Fewer stalls waiting on shared memory',
    benefitMemory: 'Less shared traffic, and better use of the banks',
    strategies: ['Remap threads to data', 'Change the layout, often by padding'],
    pages: '148, 151–152',
    commentary:
      'It returns in prefix sum and chapter 15, where threads load their parts of the input into shared memory with a stride.',
    cost: {
      text: 'Padding costs shared memory, and index arithmetic gets dearer when a dimension stops being a power of two.',
      source: 'pp. 140–141',
    },
    returnsIn: [
      { ch: 11, what: 'prefix sum: strided shared-memory loads' },
      { ch: 15, what: 'matmul: strided shared-memory loads' },
    ],
    seenIn: [
      { where: 'MLX, steel/gemm/gemm.h', what: 'threadgroup tiles padded by 16 bytes', fact: 'D3', token: 'tgp_padding_a' },
      { where: 'llama.cpp, ggml-cuda/mmq.cuh', what: 'shared tiles padded by 16 bytes "to avoid shared memory bank conflicts"', fact: 'D1', token: 'To avoid shared memory bank conflicts' },
    ],
    taught: [['ch06', '6.4']],
    why: 'It is about how the shared memory is used.',
  },
  {
    id: 'privatise',
    category: 'memory',
    name: 'Privatisation',
    benefitCompute: 'Fewer stalls waiting on atomic updates',
    benefitMemory: 'Less contention on, and serialisation of, atomic updates',
    strategies: ['Apply partial updates to a private copy, and update the public copy once at the end'],
    pages: '148, 152',
    commentary:
      'Not taught yet; listed for completeness. When many threads or blocks update one output, each updates a private copy and commits it at the end: histogram counters, and the output lists of filter and graph traversal.',
    cost: {
      text: 'Each private copy must be committed to the public one at the end; coarsening in chapter 9 exists partly to need fewer copies.',
      source: 'pp. 152–153',
    },
    returnsIn: [
      { ch: 9, what: 'histogram counters' },
      { ch: 12, what: 'filter: many threads appending to one list' },
      { ch: 18, what: 'graph traversal: frontier lists' },
    ],
    seenIn: [],
    taught: [],
    why: 'Contention on one memory location is a memory-system problem.',
  },
  {
    id: 'warp',
    category: 'sync',
    name: 'Warp-level primitives',
    benefitCompute: 'Fewer block-wide barriers',
    benefitMemory: 'Less global and shared traffic (the table; the text names only shared memory)',
    strategies: ['Do the steps that need a barrier at warp level, then combine the warps’ results at block level'],
    pages: '148, 152',
    commentary:
      'Mentioned briefly in chapter 4: threads of a warp can exchange data and synchronise without a block-wide barrier and without going through shared memory. Used in reduction, prefix sum and filter, and in chapter 15 to help register tiling.',
    cost: {
      text: 'Not stated in the book. A portability cost instead (infra facts B3 and C2): a CUDA lane mask must become a 64-bit mask on HIP, and Metal’s SIMD-group functions take no mask, so the same source needs changes per backend.',
      source: null,
    },
    returnsIn: [
      { ch: 10, what: 'reduction' },
      { ch: 11, what: 'prefix sum' },
      { ch: 12, what: 'filter' },
      { ch: 15, what: 'helping register tiling' },
    ],
    seenIn: [],
    taught: [['ch04', '4.4'], ['ch04', '4.5']],
    why: 'Its first aim is to replace block-wide barriers with cheaper warp-wide exchange.',
  },
  {
    id: 'double-buffer',
    category: 'sync',
    name: 'Double buffering',
    benefitCompute: 'Removes barriers that only enforce false dependences',
    benefitMemory: null,
    strategies: ['Write into a different buffer from the one the preceding reads use, so the write-after-read barrier can go'],
    pages: '148, 152',
    commentary: 'A recap of section 6.7. Its real uses are in prefix sum and chapter 15.',
    cost: { text: 'Two buffers instead of one: twice the memory for the buffer.', source: 'p. 147' },
    returnsIn: [
      { ch: 11, what: 'prefix sum' },
      { ch: 15, what: 'matmul' },
    ],
    seenIn: [
      {
        where: 'llama.cpp, ggml-cuda/fattn-mma-f16.cuh',
        what: 'flash attention preloads the next data into a second stage with asynchronous copies',
        fact: 'D1',
        token: 'nstages_target',
      },
      {
        where: 'MLX, steel/gemm/gemm.h',
        what: 'not used: the main GEMM loop is single-buffered with two barriers per step, the book’s "before" loop',
        fact: 'D3',
        absent: true,
        token: 'single-buffered with two barriers',
      },
    ],
    taught: [['ch06', '6.7']],
    why: 'All it removes is waiting at a barrier.',
  },
  {
    id: 'coarsen',
    category: 'general',
    name: 'Thread coarsening',
    benefitCompute: 'Depends on what parallelisation costs in this kernel',
    benefitMemory: 'Depends on what parallelisation costs in this kernel',
    strategies: ['Give each thread several units of work, to cut a parallelisation overhead you are paying for nothing'],
    pages: '148, 152–153',
    commentary:
      'What it removes differs by kernel: redundant input loads by many blocks (stencil, matmul), private copies to commit (histogram), synchronisation and divergence (reduction, prefix sum), redundant work of the parallel algorithm (prefix sum), poor coalescing (filter, sorting).',
    cost: {
      text: 'Exposes less parallelism (half-empty or partial waves), and more registers or shared memory per thread can cut occupancy.',
      source: 'p. 143',
    },
    returnsIn: [
      { ch: 8, what: 'stencil: larger output tiles, fewer redundant loads' },
      { ch: 9, what: 'histogram: fewer private copies to commit' },
      { ch: 10, what: 'reduction: less synchronisation and divergence' },
      { ch: 11, what: 'prefix sum: less synchronisation, divergence and redundant work' },
      { ch: 12, what: 'filter: better coalescing' },
      { ch: 14, what: 'sorting: better coalescing' },
      { ch: 15, what: 'matmul: larger output tiles, fewer redundant loads' },
    ],
    seenIn: [],
    taught: [['ch06', '6.5']],
    why: 'The overhead it removes is different in every kernel, so no single category fits.',
  },
];

export const rowById = (id: RowId): Row => {
  const r = ROWS.find((x) => x.id === id);
  if (!r) throw new Error('no checklist row ' + id);
  return r;
};

export function categoryCounts(rows: readonly Row[] = ROWS): Record<Category, number> {
  const c: Record<Category, number> = { compute: 0, memory: 0, sync: 0, general: 0 };
  for (const r of rows) c[r.category]++;
  return c;
}

/** "3 + 6 + 2 + 1" */
export const countLine = (c: Record<Category, number>) => CATEGORIES.map((k) => c[k]).join(' + ');

// ---------------------------------------------------------------- symptoms
export type SymptomId =
  | 'global-stall'
  | 'global-traffic'
  | 'shared-serial'
  | 'divergence'
  | 'barrier'
  | 'atomics'
  | 'instructions'
  | 'overhead';

export interface Relief {
  row: RowId;
  /** How this row relieves the symptom, in our words, with the page that supports it. */
  how: string;
  page: string;
}

export interface Symptom {
  id: SymptomId;
  label: string;
  /** What a profiler or a measurement would show. Ours. */
  looksLike: string;
  /** Rows that relieve it, most direct first. */
  relief: Relief[];
  /** Rows that spend the very resource this symptom is short of (p. 153's warning). */
  spends: Relief[];
}

export const SYMPTOMS: readonly Symptom[] = [
  {
    id: 'global-stall',
    label: 'Stalls on global memory, bandwidth far from used',
    looksLike: 'Warps wait on loads, yet the bytes moved per second are well below what the device can move.',
    relief: [
      { row: 'occupancy', how: 'more resident warps put more requests in flight', page: 'pp. 135–136' },
      { row: 'vector', how: 'each request carries more bytes, even when occupancy cannot rise', page: 'p. 139' },
      { row: 'unroll', how: 'independent loads can be issued before the first returns', page: 'pp. 144–145' },
      { row: 'coalesce', how: 'fewer, fuller transactions for the same data', page: 'pp. 126, 150' },
    ],
    spends: [
      { row: 'smem-tiling', how: 'more shared memory per block can lower occupancy', page: 'p. 153' },
      { row: 'coarsen', how: 'more registers per thread can lower occupancy', page: 'p. 143' },
    ],
  },
  {
    id: 'global-traffic',
    label: 'Too much global traffic',
    looksLike: 'Global memory is near its limit, and the kernel moves many more bytes than its data.',
    relief: [
      { row: 'smem-tiling', how: 'data a block reuses crosses from global memory once', page: 'p. 151' },
      { row: 'coalesce', how: 'no bytes of a burst or line fetched and thrown away', page: 'pp. 130–131' },
      { row: 'coarsen', how: 'fewer blocks reload the same input tiles', page: 'pp. 142, 153' },
      { row: 'unroll', how: 'a local array promoted to registers stops costing memory traffic', page: 'pp. 145–146' },
      { row: 'warp', how: 'the table claims less global traffic; the text does not say why', page: 'pp. 148, 152' },
    ],
    spends: [],
  },
  {
    id: 'shared-serial',
    label: 'Shared-memory accesses serialised',
    looksLike: 'A warp’s shared-memory access is split into several passes (a k-way bank conflict).',
    relief: [
      { row: 'banks', how: 'padding or remapping spreads the warp over all banks', page: 'pp. 140, 151' },
      { row: 'reg-tiling', how: 'values reused from registers are not read from shared memory again', page: 'p. 151' },
    ],
    spends: [{ row: 'coalesce', how: 'corner turning makes the shared-memory store strided', page: 'pp. 139–140' }],
  },
  {
    id: 'divergence',
    label: 'SIMD lanes idle (control divergence)',
    looksLike: 'Warps execute with many lanes switched off.',
    relief: [
      { row: 'divergence', how: 'reassign work or reorder data so a warp’s threads take the same path', page: 'pp. 149–150' },
      { row: 'coarsen', how: 'in reduction and prefix sum it cuts divergence overhead', page: 'p. 153' },
    ],
    spends: [],
  },
  {
    id: 'barrier',
    label: 'Time waiting at barriers',
    looksLike: 'Warps sit at __syncthreads() waiting for the rest of their block.',
    relief: [
      { row: 'double-buffer', how: 'the barrier that only protects a write-after-read goes', page: 'p. 147' },
      { row: 'warp', how: 'steps done inside a warp need no block-wide barrier', page: 'p. 152' },
      { row: 'coarsen', how: 'in reduction and prefix sum it cuts synchronisation', page: 'p. 153' },
    ],
    spends: [],
  },
  {
    id: 'atomics',
    label: 'Contended atomic updates',
    looksLike: 'Many threads update the same few locations and wait on each other.',
    relief: [
      { row: 'privatise', how: 'updates go to private copies, committed once', page: 'p. 152' },
      { row: 'coarsen', how: 'fewer private copies to commit (histogram)', page: 'p. 153' },
    ],
    spends: [],
  },
  {
    id: 'instructions',
    label: 'Many instructions per byte or per result',
    looksLike: 'The cores are busy, but much of what they issue is loads, index arithmetic and branches.',
    relief: [
      { row: 'vector', how: 'one instruction moves 16 bytes instead of 4', page: 'p. 139' },
      { row: 'unroll', how: 'fewer loop branches and counter updates', page: 'p. 144' },
      { row: 'coarsen', how: 'work common to several outputs is done once per thread', page: 'pp. 141–142' },
    ],
    spends: [{ row: 'banks', how: 'padding can make index arithmetic dearer', page: 'pp. 140–141' }],
  },
  {
    id: 'overhead',
    label: 'The same loads or work repeated across threads',
    looksLike: 'Neighbouring blocks load the same inputs, or every thread redoes the same setup.',
    relief: [
      { row: 'coarsen', how: 'one thread does several units of work and shares the overhead', page: 'pp. 141–142' },
      { row: 'smem-tiling', how: 'one load per block replaces one load per thread', page: 'p. 151' },
    ],
    spends: [],
  },
];

export interface RankedRow {
  row: Row;
  /** 'relieves' rows first, in order; then 'spends' (they make it worse); then the rest in book order. */
  role: 'relieves' | 'spends' | 'other';
  relief?: Relief;
  rank: number;
}

export function rankRows(symptom: SymptomId | null): RankedRow[] {
  const s = symptom ? SYMPTOMS.find((x) => x.id === symptom) : undefined;
  if (!s) return ROWS.map((row, i) => ({ row, role: 'other', rank: i }));
  const out: RankedRow[] = [];
  s.relief.forEach((r) => out.push({ row: rowById(r.row), role: 'relieves', relief: r, rank: out.length }));
  const seen = new Set(out.map((x) => x.row.id));
  for (const r of s.spends)
    if (!seen.has(r.row)) {
      out.push({ row: rowById(r.row), role: 'spends', relief: r, rank: out.length });
      seen.add(r.row);
    }
  for (const row of ROWS) if (!seen.has(row.id)) out.push({ row, role: 'other', rank: out.length });
  return out;
}

// ---------------------------------------------------------------- sorting drill
export type Placement = Partial<Record<RowId, Category>>;

export interface SortScore {
  placed: number;
  right: number;
  /** Per category: how many cards you put there, and how many the book has. */
  tally: Record<Category, { yours: number; book: number }>;
  done: boolean;
}

export function sortScore(p: Placement): SortScore {
  const book = categoryCounts();
  const tally = Object.fromEntries(CATEGORIES.map((c) => [c, { yours: 0, book: book[c] }])) as SortScore['tally'];
  let placed = 0;
  let right = 0;
  for (const row of ROWS) {
    const c = p[row.id];
    if (!c) continue;
    placed++;
    tally[c].yours++;
    if (c === row.category) right++;
  }
  return { placed, right, tally, done: placed === ROWS.length };
}

// ---------------------------------------------------------------- scenarios
export type Outcome = 'faster' | 'same' | 'slower';
export const OUTCOME_LABEL: Record<Outcome, string> = {
  faster: 'Faster',
  same: 'About the same',
  slower: 'Slower',
};

export interface Scenario {
  id: string;
  /** The kernel and its bottleneck. */
  situation: string;
  /** The proposed change. */
  change: string;
  expect: Outcome;
  /** Why, tied to p. 153's rule. */
  why: string;
  row: RowId;
  page: string;
}

export const SCENARIOS: readonly Scenario[] = [
  {
    id: 'tiling-bandwidth',
    situation:
      'A tiled matmul (chapter 5’s kernel, T × T threads per block) moves bytes at close to your measured bandwidth. Its occupancy is at the device maximum, and a block with four times the shared memory and four times the threads would still leave it there.',
    change: 'Double the tile width: four times the shared memory and four times the threads per block.',
    expect: 'faster',
    why: 'Tiling spends shared memory, which is not short here, to relieve global traffic, which is the bottleneck. Each input crosses from global memory half as often, and shared memory per thread stays at 8 bytes. This is the case where p. 153 says tiling pays.',
    row: 'smem-tiling',
    page: 'p. 153',
  },
  {
    id: 'padding-no-conflict',
    situation:
      'The profiler reports one pass per shared-memory request (no bank conflicts). The kernel is limited by global memory.',
    change: 'Pad the shared tile by one column.',
    expect: 'same',
    why: 'Padding relieves bank conflicts, and there are none: nothing is relieved. It does cost a little shared memory and index arithmetic (pp. 140–141), so if anything it is marginally slower. A gain is not on the table.',
    row: 'banks',
    page: 'p. 153',
  },
  {
    id: 'coarsen-small-grid',
    situation:
      'A grid fills the device for just over two waves of blocks, and the kernel has no redundant loads between blocks: each element is read by one thread.',
    change: 'Coarsen by a factor of 8.',
    expect: 'slower',
    why: 'There is no overhead to remove (pitfall 1), and the grid shrinks to about a quarter of one wave, so most of the device sits idle (pitfall 2). The book works the same arithmetic: 1056 blocks on 264 slots coarsened by 8 is half a wave.',
    row: 'coarsen',
    page: 'pp. 142–143',
  },
  {
    id: 'vector-full',
    situation:
      'A streaming kernel (one add per 12 bytes) runs at full occupancy and already moves bytes at about your measured bandwidth.',
    change: 'Switch to 16-byte vector loads and stores.',
    expect: 'same',
    why: 'The bytes do not change and the memory system is already as busy as it gets. Vector loads cut instruction count and add bytes in flight, which matters below full occupancy (p. 139). Neither is the bottleneck here.',
    row: 'vector',
    page: 'p. 153',
  },
  {
    id: 'double-buffer-barrier',
    situation:
      'Each iteration reads neighbours’ values from a shared buffer, writes its own, and has two barriers. Barrier waits dominate the time; occupancy has room to spare.',
    change: 'Use two alternating buffers and drop the barrier between the read and the write.',
    expect: 'faster',
    why: 'Half the barriers go, and the one that goes only protected a write-after-read (false) dependence. The cost, twice the buffer memory, is spent from a resource that is not short.',
    row: 'double-buffer',
    page: 'p. 147',
  },
];
