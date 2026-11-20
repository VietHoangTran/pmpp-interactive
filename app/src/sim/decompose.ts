// Problem decomposition for chapter 22, section 22.2 (book pp. 532–536): who owns what, who
// writes, who conflicts. Chapter 21 (atom-centric against grid-centric) can reuse it.
//
// What it models:
//  - A computation is a contribution graph: every contribution says "input i adds into
//    output o". Four computations: a 1-D convolution (zero-padded "same", or "valid"), a
//    decode-time matrix-vector product y = W x with W row-major, a histogram with a skew
//    control, and a sparse matrix-vector product whose first row can be made heavy.
//  - A decomposition hands the contributions to workers (threads, or warps for the
//    matrix-vector mappings). Output-centric: a worker owns outputs and gathers. Input-centric:
//    a worker owns inputs and scatters. For the matrix-vector product four mappings: a thread
//    per input column (input-centric), a thread per output, a warp per output, and split-K
//    (S warps per output, each over one chunk of the input dimension).
//  - The book's criteria (p. 533) as numbers: threads (parallelism exposed); outputs that
//    more than one worker writes, which need atomics, and the atomic updates they take; the
//    worst number of writers on one output; additions done in registers; inputs a worker must
//    inspect to find its own (ease of mapping); useful work per thread, maximum over mean (load
//    balance). A worker adds all its contributions to one output in a register and writes once.
//  - Warp 0's first load of W for the matrix-vector mappings, as byte addresses. The caller
//    applies coalesce.ts warpAccess to them (NVIDIA's 32-byte sector, ch06-facts A2), as
//    coarsen.ts does, because this file must have no value imports.
//  - A serialised-atomics schedule for the gather-scatter scene: every update is issued at
//    once; one address serves one update per tick; different addresses in parallel.
//  - A split-K trace for the stepper: a batch of rows times W, warp by warp, with lanes, a
//    shuffle reduction, and either a direct store (S = 1), an atomic add, or a plain += split
//    into its read and its write so that an interleaving can lose an update.
//  - TOUR: the book's classification of every pattern (pp. 533–536), with four rows of ours.
//
// What it leaves out: time, caches, coalescing beyond one instruction's segment count, the
// cost of an atomic relative to a register add (documented nowhere, ch22-facts A1), banks,
// occupancy, divergence. A lost update is the typical outcome of a race, not a guaranteed
// one (ch09-facts A3): the trace shows the interleaving that loses it.
//
// Erasable TypeScript only, and no value imports: the browser checks import this file
// directly in Node.

// ---------------------------------------------------------------- problems

export type Kind = 'conv' | 'matvec' | 'hist' | 'spmv';

/** One contribution: input `i` adds into output `o`. */
export interface Edge {
  i: number;
  o: number;
}

interface Base {
  nIn: number;
  nOut: number;
  /** Every contribution, in the order a sequential loop over the outputs adds them. */
  edges: Edge[];
}
export interface ConvProblem extends Base {
  kind: 'conv';
  n: number;
  /** Filter width, odd. */
  w: number;
  /** "valid": n − w + 1 outputs, out[o] = Σ in[o + t]. Otherwise "same" with zero padding. */
  valid: boolean;
}
export interface MatvecProblem extends Base {
  kind: 'matvec';
  /** Outputs (rows of W). */
  N: number;
  /** Inputs (columns of W, length of x). Input index i of an edge is the weight W[o][i]'s column. */
  K: number;
}
export interface HistProblem extends Base {
  kind: 'hist';
  bins: number;
  values: number[];
}
export interface SpmvProblem extends Base {
  kind: 'spmv';
  rows: number;
  cols: number;
  /** Non-zero k is in row rowOf[k], column colOf[k]; non-zeros are stored row by row. */
  rowOf: number[];
  colOf: number[];
}
export type Problem = ConvProblem | MatvecProblem | HistProblem | SpmvProblem;

export function conv(n: number, w: number, valid = false): ConvProblem {
  const r = (w - 1) / 2;
  const nOut = valid ? n - w + 1 : n;
  const edges: Edge[] = [];
  for (let o = 0; o < nOut; o++)
    for (let t = 0; t < w; t++) {
      const i = valid ? o + t : o + t - r;
      if (i >= 0 && i < n) edges.push({ i, o });
    }
  return { kind: 'conv', n, w, valid, nIn: n, nOut, edges };
}

export function matvec(N: number, K: number): MatvecProblem {
  const edges: Edge[] = [];
  for (let o = 0; o < N; o++) for (let i = 0; i < K; i++) edges.push({ i, o });
  return { kind: 'matvec', N, K, nIn: K, nOut: N, edges };
}

/**
 * A deterministic histogram input: a fraction `skew` of the inputs (picked by a golden-ratio
 * sequence, so they are spread along the array) fall in bin 0, the rest go round the bins.
 */
export function histValues(n: number, bins: number, skew: number): number[] {
  const g = 0.6180339887498949;
  return Array.from({ length: n }, (_, i) => (((i + 1) * g) % 1 < skew ? 0 : i % bins));
}

export function hist(values: number[], bins: number): HistProblem {
  return {
    kind: 'hist', bins, values, nIn: values.length, nOut: bins,
    edges: values.map((v, i) => ({ i, o: v })),
  };
}

/**
 * A sparse matrix with `rows` rows of 2 non-zeros each, plus `heavy` extra non-zeros in row 0.
 * Columns are spread so that no row repeats one.
 */
export function spmv(rows: number, heavy: number): SpmvProblem {
  const cols = Math.max(8, 2 + heavy);
  const rowOf: number[] = [];
  const colOf: number[] = [];
  for (let r = 0; r < rows; r++) {
    const len = r === 0 ? 2 + heavy : 2;
    for (let k = 0; k < len; k++) {
      rowOf.push(r);
      colOf.push((r * 3 + k) % cols);
    }
  }
  return {
    kind: 'spmv', rows, cols, rowOf, colOf, nIn: rowOf.length, nOut: rows,
    edges: rowOf.map((o, i) => ({ i, o })),
  };
}

// ---------------------------------------------------------------- decompositions

/** conv, hist, spmv: 'output' or 'input'. matvec: all four. */
export type Mapping = 'output' | 'input' | 'warp' | 'splitk';

export const MAPPING_LABEL: Record<Kind, Partial<Record<Mapping, string>>> = {
  conv: { output: 'output-centric (gather)', input: 'input-centric (scatter)' },
  hist: { output: 'output-centric: a thread per bin', input: 'input-centric: a thread per input' },
  spmv: { output: 'output-centric: a thread per row (CSR)', input: 'input-centric: a thread per non-zero (COO)' },
  matvec: {
    input: 'input-centric: a thread per input',
    output: 'a thread per output',
    warp: 'a warp per output',
    splitk: 'split the input dimension: S warps per output',
  },
};

export const mappingsOf = (kind: Kind): Mapping[] => (kind === 'matvec' ? ['input', 'output', 'warp', 'splitk'] : ['output', 'input']);

export interface Write {
  o: number;
  /** Contributions this worker adds together before this one write. */
  parts: number;
}

export interface Worker {
  id: number;
  /** 1 for a thread; the warp width for a warp. */
  lanes: number;
  /** Inputs whose contributions it computes (its useful reads), in order. */
  useful: number[];
  /** How many inputs it must look at to find them. */
  inspects: number;
  /** True when it must look at every input (an output-centric histogram bin). */
  inspectsAll: boolean;
  writes: Write[];
  /** Useful work of each of its threads (one entry for a thread, one per lane for a warp). */
  laneWork: number[];
}

export interface Metrics {
  /** Threads launched (lanes, for warp mappings): the parallelism exposed. */
  threads: number;
  workers: number;
  contributions: number;
  /** Outputs written by more than one worker: each of their updates needs an atomic. */
  outputsNeedingAtomics: number;
  atomicUpdates: number;
  /** Writes to outputs with a single writer: plain stores. */
  plainStores: number;
  worstWriters: number;
  /** Contributions added in a register rather than in memory: contributions − writes. */
  registerAccumulations: number;
  inspectedTotal: number;
  inspectedMax: number;
  /** Useful work per thread. */
  workMax: number;
  workMean: number;
  /** workMax ÷ workMean: 1 is perfect balance. */
  imbalance: number;
}

export interface Options {
  /** Lanes per warp for the warp mappings. */
  warpWidth: number;
  /** Chunks of the input dimension for split-K. */
  S: number;
}

export interface Decomposition {
  problem: Problem;
  mapping: Mapping;
  opts: Options;
  workers: Worker[];
  /** Distinct workers writing each output. */
  writers: number[];
  metrics: Metrics;
}

/** Split `count` consecutive items over `lanes` lanes the way a strided loop does: lane l gets l, l + lanes, ... */
function stride(count: number, lanes: number): number[] {
  return Array.from({ length: lanes }, (_, l) => (l < count ? Math.floor((count - 1 - l) / lanes) + 1 : 0));
}

function thread(id: number, useful: number[], writes: Write[], inspects = useful.length, inspectsAll = false): Worker {
  return { id, lanes: 1, useful, inspects, inspectsAll, writes, laneWork: [useful.length] };
}

function workersOf(p: Problem, mapping: Mapping, opts: Options): Worker[] {
  if (p.kind === 'matvec') {
    const { N, K } = p;
    const all = Array.from({ length: K }, (_, i) => i);
    switch (mapping) {
      case 'input':
        return all.map((i) => thread(i, [i], Array.from({ length: N }, (_, o) => ({ o, parts: 1 })), 1));
      case 'output':
        return Array.from({ length: N }, (_, o) => thread(o, all, [{ o, parts: K }]));
      case 'warp':
        return Array.from({ length: N }, (_, o) => ({
          id: o, lanes: opts.warpWidth, useful: all, inspects: K, inspectsAll: false,
          writes: [{ o, parts: K }], laneWork: stride(K, opts.warpWidth),
        }));
      case 'splitk': {
        const { S, warpWidth } = opts;
        const out: Worker[] = [];
        for (let w = 0; w < N * S; w++) {
          const o = Math.floor(w / S);
          const { start, len } = chunkOf(K, S, w % S);
          const useful = Array.from({ length: len }, (_, t) => start + t);
          out.push({ id: w, lanes: warpWidth, useful, inspects: len, inspectsAll: false, writes: [{ o, parts: len }], laneWork: stride(len, warpWidth) });
        }
        return out;
      }
    }
  }
  if (mapping === 'output') {
    const by: number[][] = Array.from({ length: p.nOut }, () => []);
    for (const e of p.edges) by[e.o].push(e.i);
    // An output-centric histogram bin cannot tell which inputs are its own without looking (p. 534).
    const all = p.kind === 'hist';
    return by.map((ins, o) => thread(o, ins, [{ o, parts: ins.length }], all ? p.nIn : ins.length, all));
  }
  // input-centric: one input per thread; its contributions to one output are added before the write.
  const by: Map<number, number>[] = Array.from({ length: p.nIn }, () => new Map());
  for (const e of p.edges) by[e.i].set(e.o, (by[e.i].get(e.o) ?? 0) + 1);
  return by.map((m, i) => {
    const writes = [...m.entries()].map(([o, parts]) => ({ o, parts }));
    const w = thread(i, [i], writes, 1);
    w.laneWork = [writes.reduce((s, x) => s + x.parts, 0)];
    return w;
  });
}

/**
 * Chunk `c` of `S` over K inputs, cut as lab 22.1's gemvSplitKKernel cuts them: chunks of
 * ceil(K / S), the last one shorter (or empty) when S does not divide K.
 */
export function chunkOf(K: number, S: number, c: number): { start: number; len: number } {
  const size = Math.ceil(K / S);
  const start = Math.min(K, c * size);
  return { start, len: Math.max(0, Math.min(K, start + size) - start) };
}

export function decompose(problem: Problem, mapping: Mapping, opts: Options): Decomposition {
  if (problem.kind !== 'matvec' && (mapping === 'warp' || mapping === 'splitk'))
    throw new Error(`mapping ${mapping} is for the matrix-vector product only`);
  const workers = workersOf(problem, mapping, opts);
  const writerSets: Set<number>[] = Array.from({ length: problem.nOut }, () => new Set());
  for (const w of workers) for (const x of w.writes) writerSets[x.o].add(w.id);
  const writers = writerSets.map((s) => s.size);
  let atomicUpdates = 0;
  let plainStores = 0;
  for (const w of workers)
    for (const x of w.writes) {
      if (writers[x.o] > 1) atomicUpdates++;
      else plainStores++;
    }
  const contributions = problem.edges.length;
  const laneWork = workers.flatMap((w) => w.laneWork);
  const threads = laneWork.length;
  const max = (xs: number[]) => xs.reduce((m, v) => (v > m ? v : m), 0);
  const workMax = max(laneWork);
  const workMean = threads ? laneWork.reduce((s, v) => s + v, 0) / threads : 0;
  const metrics: Metrics = {
    threads,
    workers: workers.length,
    contributions,
    outputsNeedingAtomics: writers.filter((v) => v > 1).length,
    atomicUpdates,
    plainStores,
    worstWriters: max(writers),
    registerAccumulations: contributions - atomicUpdates - plainStores,
    inspectedTotal: workers.reduce((s, w) => s + w.inspects, 0),
    inspectedMax: max(workers.map((w) => w.inspects)),
    workMax,
    workMean,
    imbalance: workMean ? workMax / workMean : 1,
  };
  return { problem, mapping, opts, workers, writers, metrics };
}

/** How many workers read each input (the reuse a block could keep in shared memory). */
export function readsPerInput(d: Decomposition): number[] {
  const n = new Array(d.problem.nIn).fill(0);
  for (const w of d.workers) for (const i of w.useful) n[i]++;
  return n;
}

// ---------------------------------------------------------------- warp 0's first load of W

/**
 * Byte addresses (4-byte floats, W row-major N × K) of warp 0's first load of W under a
 * matrix-vector mapping. Lane l of warp 0 is thread l for the thread mappings. Apply
 * coalesce.ts warpAccess(addrs, 4, 32) for NVIDIA's sector count.
 */
export function warp0Load(p: MatvecProblem, mapping: Mapping, opts: Options): number[] {
  const { N, K } = p;
  const lanes = (n: number) => Array.from({ length: Math.min(opts.warpWidth, n) }, (_, l) => l);
  switch (mapping) {
    case 'output':
      return lanes(N).map((l) => (l * K + 0) * 4); // thread l walks row l: W[l][0], K floats apart
    case 'input':
      return lanes(K).map((l) => (0 * K + l) * 4); // thread l walks column l: W[0][l]
    case 'warp':
      return lanes(K).map((l) => (0 * K + l) * 4); // warp 0 owns row 0: W[0][l]
    case 'splitk': {
      const { start, len } = chunkOf(K, opts.S, 0);
      return lanes(len).map((l) => (0 * K + start + l) * 4); // warp 0: row 0, chunk 0
    }
  }
}

// ---------------------------------------------------------------- one sentence per worker

const fmtList = (xs: number[]) => (xs.length <= 6 ? xs.join(', ') : `${xs[0]}, ${xs[1]}, … ${xs[xs.length - 1]}`);

/** The index mapping of one worker, in words, with its own numbers. */
export function describeWorker(d: Decomposition, id: number): string {
  const p = d.problem;
  const w = d.workers[id];
  const shared = (o: number) => d.writers[o] > 1;
  const outs = w.writes.map((x) => x.o);
  const nShared = outs.filter(shared).length;
  const tail = nShared
    ? ` ${nShared === outs.length ? (outs.length === 1 ? 'That output has' : 'Every one of them has') : `${nShared} of them have`} other writers too, so each update is an atomic.`
    : ' No other worker writes there, so a plain store is safe.';
  if (p.kind === 'conv') {
    const r = (p.w - 1) / 2;
    if (d.mapping === 'output') {
      const lo = p.valid ? 'o' : 'o − ' + r;
      const hi = p.valid ? `o + ${p.w - 1}` : 'o + ' + r;
      return `Thread ${id} owns out[${id}]. It reads in[${lo}] to in[${hi}] that exist (in[${fmtList(w.useful)}]), adds ${w.useful.length} products in a register and stores once.` + tail;
    }
    return `Thread ${id} owns in[${id}]. It multiplies it by each filter tap and adds the product into every output that uses it: out[${fmtList(outs)}].` + tail;
  }
  if (p.kind === 'hist') {
    if (d.mapping === 'output')
      return `Thread ${id} owns bin ${id}. It cannot know which inputs fall in its bin without looking, so it reads all ${p.nIn} inputs and counts the ${w.useful.length} that do.` + tail;
    return `Thread ${id} owns input ${id}, whose value is ${p.values[id]}. It adds 1 to bin ${p.values[id]}.` + tail;
  }
  if (p.kind === 'spmv') {
    if (d.mapping === 'output') {
      const first = w.useful[0] ?? 0;
      return `Thread ${id} owns y[${id}]. The row pointers give its ${w.useful.length} non-zeros directly (k = ${first} to ${first + w.useful.length - 1}); it sums A·x over them in a register and stores once.` + tail;
    }
    return `Thread ${id} owns non-zero ${id}, in row ${p.rowOf[id]}, column ${p.colOf[id]}. It adds A[${p.rowOf[id]}][${p.colOf[id]}]·x[${p.colOf[id]}] into y[${p.rowOf[id]}].` + tail;
  }
  const { K } = p;
  const ww = d.opts.warpWidth;
  switch (d.mapping) {
    case 'input':
      return `Thread ${id} owns x[${id}], column ${id} of W. For every output j it adds W[j][${id}]·x[${id}] into y[j].` + tail;
    case 'output':
      return `Thread ${id} owns y[${id}] and walks row ${id} of W alone, W[${id}][0] to W[${id}][${K - 1}]. Lane l of a warp reads W[l][k]: neighbouring lanes are ${K} floats apart.` + tail;
    case 'warp': {
      const it = Math.ceil(K / ww);
      return `Warp ${id} owns y[${id}]. Lane l reads W[${id}][l + i × ${ww}] for i = 0 to ${it - 1}, so the warp's lanes read one contiguous run at each step; the lanes combine their sums by shuffles and lane 0 stores.` + tail;
    }
    case 'splitk': {
      const S = d.opts.S;
      const o = Math.floor(id / S);
      const c = id % S;
      const { start, len } = chunkOf(K, S, c);
      const it = Math.ceil(len / ww);
      return `Warp ${id} owns output ${id} ÷ ${S} = ${o} and chunk ${id} mod ${S} = ${c}, which starts at ${start}. Lane l reads W[${o}][${start} + l + i × ${ww}] for i = 0 to ${it - 1}; the lanes combine their sums by shuffles and lane 0 adds the warp's sum into y[${o}].` + tail;
    }
  }
  return '';
}

/** Which worker owns W[row][col] under a matrix-vector mapping (for clicking on the drawing). */
export function ownerOfWeight(p: MatvecProblem, mapping: Mapping, opts: Options, row: number, col: number): number {
  switch (mapping) {
    case 'input':
      return col;
    case 'output':
    case 'warp':
      return row;
    case 'splitk': {
      for (let c = 0; c < opts.S; c++) {
        const { start, len } = chunkOf(p.K, opts.S, c);
        if (col >= start && col < start + len) return row * opts.S + c;
      }
      return row * opts.S;
    }
  }
}

// ---------------------------------------------------------------- explorer state and presets

export interface ExplorerState {
  kind: Kind;
  mapping: Mapping;
  conv: { n: number; w: number };
  matvec: { N: number; K: number; S: number; warpWidth: number };
  hist: { n: number; bins: number; skew: number };
  spmv: { rows: number; heavy: number };
}

/** NVIDIA's warp width on every listed compute capability (infra-facts, "Where the backends really differ"). */
export const NVIDIA_WARP = 32;

export const DEFAULT_STATE: ExplorerState = {
  kind: 'conv',
  mapping: 'output',
  conv: { n: 12, w: 5 },
  matvec: { N: 8, K: 256, S: 4, warpWidth: NVIDIA_WARP },
  hist: { n: 16, bins: 4, skew: 0.5 },
  spmv: { rows: 6, heavy: 8 },
};

export interface Preset {
  id: string;
  label: string;
  state: ExplorerState;
}

export const PRESETS: Preset[] = [
  { id: 'conv-scatter', label: 'Convolution, scattered', state: { ...DEFAULT_STATE, kind: 'conv', mapping: 'input' } },
  { id: 'hist-bins', label: 'Histogram, a thread per bin', state: { ...DEFAULT_STATE, kind: 'hist', mapping: 'output' } },
  { id: 'spmv-coo', label: 'Skewed sparse matrix, COO', state: { ...DEFAULT_STATE, kind: 'spmv', mapping: 'input' } },
  {
    id: 'gemv-narrow', label: 'Narrow decode GEMV, split-K',
    state: { ...DEFAULT_STATE, kind: 'matvec', mapping: 'splitk', matvec: { N: 2, K: 256, S: 4, warpWidth: NVIDIA_WARP } },
  },
];

export function problemOf(s: ExplorerState): Problem {
  switch (s.kind) {
    case 'conv':
      return conv(s.conv.n, s.conv.w);
    case 'matvec':
      return matvec(s.matvec.N, s.matvec.K);
    case 'hist':
      return hist(histValues(s.hist.n, s.hist.bins, s.hist.skew), s.hist.bins);
    case 'spmv':
      return spmv(s.spmv.rows, s.spmv.heavy);
  }
}

export function decomposeState(s: ExplorerState): Decomposition {
  const mapping = mappingsOf(s.kind).includes(s.mapping) ? s.mapping : 'output';
  return decompose(problemOf(s), mapping, { warpWidth: s.matvec.warpWidth, S: s.matvec.S });
}

/** S values that give every lane of a split-K warp at least one element. */
export const splitChoices = (K: number, warpWidth: number): number[] => [1, 2, 4, 8, 16].filter((S) => K / S >= warpWidth);

// ---------------------------------------------------------------- serialised atomics (the scene)

export interface Tick {
  /** Updates completed on this tick: (output, worker). */
  served: { o: number; worker: number }[];
  /** Updates still waiting at each output after this tick. */
  waiting: number[];
}

/**
 * Every write is issued at tick 0. Each output serves one update per tick, in worker order;
 * different outputs are served in parallel. Returns one entry per tick until all are served.
 */
export function serialise(d: Decomposition): { ticks: Tick[]; issued: number[] } {
  const queues: number[][] = Array.from({ length: d.problem.nOut }, () => []);
  for (const w of d.workers) for (const x of w.writes) queues[x.o].push(w.id);
  const issued = queues.map((q) => q.length);
  const ticks: Tick[] = [];
  for (let t = 0; queues.some((q) => q.length > t); t++)
    ticks.push({
      served: queues.flatMap((q, o) => (t < q.length ? [{ o, worker: q[t] }] : [])),
      waiting: queues.map((q) => Math.max(0, q.length - t - 1)),
    });
  return { ticks, issued };
}

// ---------------------------------------------------------------- the scene's numbers

/** The scene's example: eight inputs, six outputs, each the sum of three neighbours. */
export const SCENE = { n: 8, w: 3, inputs: [3, 1, 4, 1, 5, 9, 2, 6], histBins: 3 };

export function sceneModel() {
  const p = conv(SCENE.n, SCENE.w, true);
  const gather = decompose(p, 'output', { warpWidth: 1, S: 1 });
  const scatter = decompose(p, 'input', { warpWidth: 1, S: 1 });
  const sums = gather.workers.map((w) => w.useful.reduce((s, i) => s + SCENE.inputs[i], 0));
  const fanOut = Array.from({ length: p.nIn }, (_, i) => p.edges.filter((e) => e.i === i).length);
  const hp = hist(SCENE.inputs.map((v) => v % SCENE.histBins), SCENE.histBins);
  const histGather = decompose(hp, 'output', { warpWidth: 1, S: 1 });
  return { p, gather, scatter, sums, fanOut, reads: readsPerInput(gather), gatherTicks: serialise(gather), scatterTicks: serialise(scatter), hp, histGather };
}

// ---------------------------------------------------------------- split-K, warp by warp

export interface SplitKConfig {
  batch: number;
  /** Inputs (length of each x row, columns of W). */
  K: number;
  /** Outputs (rows of W). */
  N: number;
  warpWidth: number;
  /** Chunks of the input dimension; N·S warps. */
  S: number;
  /** Lane 0 commits with an atomic add (true) or a plain += (false). Ignored when S = 1. */
  atomics: boolean;
}

/** The stepper's tiny case: batch 2, 16 inputs, 3 outputs, a toy warp of 4 so every lane is visible. */
export const SPLITK_TINY = { batch: 2, K: 16, N: 3, warpWidth: 4 };

/** Small integers so every sum is exact in floating point, whatever the order. */
export function splitKData(cfg: Pick<SplitKConfig, 'batch' | 'K' | 'N'>): { W: number[]; X: number[]; ref: number[] } {
  const { batch, K, N } = cfg;
  const W = Array.from({ length: N * K }, (_, t) => ((Math.floor(t / K) * 5 + (t % K) * 3) % 7) - 3);
  const X = Array.from({ length: batch * K }, (_, t) => ((Math.floor(t / K) * 3 + (t % K)) % 5) - 2);
  const ref: number[] = [];
  for (let b = 0; b < batch; b++)
    for (let j = 0; j < N; j++) {
      let s = 0;
      for (let k = 0; k < K; k++) s += W[j * K + k] * X[b * K + k];
      ref.push(s);
    }
  return { W, X, ref };
}

export type OpKind = 'load' | 'fma' | 'shfl' | 'store' | 'atomic' | 'read' | 'write';
export interface SKOp {
  kind: OpKind;
  /** Strip index for load and fma. */
  strip?: number;
  /** Shuffle distance for shfl. */
  offset?: number;
}

export const splitKWarps = (cfg: SplitKConfig) => cfg.N * cfg.S;

export function warpPlace(cfg: SplitKConfig, w: number) {
  const j = Math.floor(w / cfg.S);
  const c = w % cfg.S;
  const { start, len } = chunkOf(cfg.K, cfg.S, c);
  return { j, c, start, len, strips: Math.ceil(len / cfg.warpWidth) };
}

export function warpOps(cfg: SplitKConfig, w: number): SKOp[] {
  const { strips } = warpPlace(cfg, w);
  const ops: SKOp[] = [];
  for (let s = 0; s < strips; s++) ops.push({ kind: 'load', strip: s }, { kind: 'fma', strip: s });
  for (let off = cfg.warpWidth / 2; off >= 1; off /= 2) ops.push({ kind: 'shfl', offset: off });
  if (cfg.S === 1) ops.push({ kind: 'store' });
  else if (cfg.atomics) ops.push({ kind: 'atomic' });
  else ops.push({ kind: 'read' }, { kind: 'write' });
  return ops;
}

export type Interleave = 'sequential' | 'round-robin' | 'reversed';
export const INTERLEAVE_LABEL: Record<Interleave, string> = {
  sequential: 'one warp at a time',
  'round-robin': 'all warps together, in turn',
  reversed: 'one warp at a time, last warp first',
};

/** A schedule: the warp that performs each successive operation. */
export function splitKOrder(cfg: SplitKConfig, how: Interleave): number[] {
  const W = splitKWarps(cfg);
  const counts = Array.from({ length: W }, (_, w) => warpOps(cfg, w).length);
  const order: number[] = [];
  if (how === 'sequential') counts.forEach((c, w) => { for (let k = 0; k < c; k++) order.push(w); });
  else if (how === 'reversed') for (let w = W - 1; w >= 0; w--) for (let k = 0; k < counts[w]; k++) order.push(w);
  else {
    const left = [...counts];
    while (left.some((c) => c > 0))
      for (let w = 0; w < W; w++)
        if (left[w] > 0) {
          order.push(w);
          left[w]--;
        }
  }
  return order;
}

export interface SKStep {
  warp: number;
  op: SKOp;
  j: number;
  c: number;
  /** For load: the K index each lane reads (−1 for a lane with nothing to read). */
  ks?: number[];
  /** Y cells touched (b·N + j) and what they hold afterwards. */
  touched?: number[];
  /** For write: chunks whose contribution this write destroyed. */
  lost?: number[];
}

export interface SKSnapshot {
  /** Y, batch × N, row-major. */
  Y: number[];
  /** Chunks each Y cell holds. */
  parts: number[][];
  /** Each warp's running sums, [b][lane]. */
  acc: number[][][];
  /** Operations each warp has done. */
  done: number[];
}

export interface SKRun {
  cfg: SplitKConfig;
  data: { W: number[]; X: number[]; ref: number[] };
  steps: SKStep[];
  /** Snapshot after each step; snaps[0] is before the first. */
  snaps: SKSnapshot[];
  final: number[];
  correct: boolean;
  /** Contributions (b, j, chunk) lost to plain += interleavings. */
  lostCount: number;
}

const clone = (s: SKSnapshot): SKSnapshot => ({
  Y: [...s.Y], parts: s.parts.map((p) => [...p]), acc: s.acc.map((a) => a.map((r) => [...r])), done: [...s.done],
});

/**
 * Runs the warps in `order` (each entry: the warp that performs its next operation; entries
 * for a finished warp are skipped). Every operation of a warp is one step.
 */
export function splitKRun(cfg: SplitKConfig, order: number[]): SKRun {
  const data = splitKData(cfg);
  const { W, X } = data;
  const { batch, N, K, warpWidth: ww } = cfg;
  const nW = splitKWarps(cfg);
  const ops = Array.from({ length: nW }, (_, w) => warpOps(cfg, w));
  let s: SKSnapshot = {
    Y: new Array(batch * N).fill(0),
    parts: Array.from({ length: batch * N }, () => []),
    acc: Array.from({ length: nW }, () => Array.from({ length: batch }, () => new Array(ww).fill(0))),
    done: new Array(nW).fill(0),
  };
  // Registers not in the snapshot: loaded weights and the copy plain += reads.
  const wreg: number[][] = Array.from({ length: nW }, () => new Array(ww).fill(0));
  const kreg: number[][] = Array.from({ length: nW }, () => new Array(ww).fill(-1));
  const tmp: { y: number[]; parts: number[][] }[] = Array.from({ length: nW }, () => ({ y: [], parts: [] }));
  const steps: SKStep[] = [];
  const snaps: SKSnapshot[] = [clone(s)];
  let lostCount = 0;
  for (const w of order) {
    if (s.done[w] >= ops[w].length) continue;
    const op = ops[w][s.done[w]];
    const { j, c, start, len } = warpPlace(cfg, w);
    const cells = Array.from({ length: batch }, (_, b) => b * N + j);
    const step: SKStep = { warp: w, op, j, c };
    s = clone(s);
    switch (op.kind) {
      case 'load': {
        const ks = Array.from({ length: ww }, (_, l) => {
          const t = l + op.strip! * ww;
          return t < len ? start + t : -1;
        });
        ks.forEach((k, l) => {
          kreg[w][l] = k;
          wreg[w][l] = k >= 0 ? W[j * K + k] : 0;
        });
        step.ks = ks;
        break;
      }
      case 'fma':
        for (let b = 0; b < batch; b++)
          for (let l = 0; l < ww; l++) if (kreg[w][l] >= 0) s.acc[w][b][l] += wreg[w][l] * X[b * K + kreg[w][l]];
        step.ks = [...kreg[w]];
        break;
      case 'shfl':
        for (let b = 0; b < batch; b++) {
          const old = [...s.acc[w][b]];
          for (let l = 0; l < op.offset!; l++) s.acc[w][b][l] = old[l] + old[l + op.offset!];
        }
        break;
      case 'store':
        cells.forEach((cell, b) => {
          s.Y[cell] = s.acc[w][b][0];
          s.parts[cell] = [c];
        });
        step.touched = cells;
        break;
      case 'atomic':
        cells.forEach((cell, b) => {
          s.Y[cell] += s.acc[w][b][0];
          s.parts[cell] = [...s.parts[cell], c].sort((x, y) => x - y);
        });
        step.touched = cells;
        break;
      case 'read':
        tmp[w] = { y: cells.map((cell) => s.Y[cell]), parts: cells.map((cell) => [...s.parts[cell]]) };
        step.touched = cells;
        break;
      case 'write': {
        const lost = new Set<number>();
        cells.forEach((cell, b) => {
          for (const q of s.parts[cell]) if (!tmp[w].parts[b].includes(q)) lost.add(q);
          lostCount += s.parts[cell].filter((q) => !tmp[w].parts[b].includes(q)).length;
          s.Y[cell] = tmp[w].y[b] + s.acc[w][b][0];
          s.parts[cell] = [...tmp[w].parts[b], c].sort((x, y) => x - y);
        });
        step.touched = cells;
        if (lost.size) step.lost = [...lost].sort((x, y) => x - y);
        break;
      }
    }
    s.done[w]++;
    steps.push(step);
    snaps.push(s);
  }
  const final = s.Y;
  return { cfg, data, steps, snaps, final, correct: final.every((v, t) => v === data.ref[t]), lostCount };
}

/** Bytes of W, bytes of x, threads and atomic updates for a split-K configuration. */
export function splitKTraffic(cfg: SplitKConfig) {
  return {
    wBytes: cfg.N * cfg.K * 4,
    xBytes: cfg.N * cfg.K * cfg.batch * 4,
    threads: cfg.N * cfg.S * cfg.warpWidth,
    atomics: cfg.S > 1 ? cfg.N * cfg.S * cfg.batch : 0,
  };
}

/** One sentence for a step, with the index arithmetic for one chosen lane. */
export function describeSKStep(run: SKRun, n: number, lane: number): string {
  const st = run.steps[n];
  const { cfg } = run;
  const { j, c, start } = warpPlace(cfg, st.warp);
  const ww = cfg.warpWidth;
  const who = `Warp ${st.warp}`;
  const where = cfg.S === 1
    ? `${who} owns output j = ${st.warp}`
    : `${who} owns output j = ${st.warp} ÷ ${cfg.S} = ${j} and chunk c = ${st.warp} mod ${cfg.S} = ${c}, which starts at c × ${cfg.K} ÷ ${cfg.S} = ${start}`;
  const after = run.snaps[n + 1];
  const B = cfg.batch;
  const ys = (cells: number[]) => cells.map((cell) => `Y[${Math.floor(cell / cfg.N)}][${cell % cfg.N}] = ${after.Y[cell]}`).join(', ');
  switch (st.op.kind) {
    case 'load': {
      const live = st.ks!.filter((k) => k >= 0);
      const k = st.ks![lane];
      const laneTxt = k >= 0
        ? ` Lane ${lane}: W[${j}][${start} + ${lane} + ${st.op.strip} × ${ww}] = W[${j}][${k}] = ${run.data.W[j * cfg.K + k]}.`
        : ` Lane ${lane} has nothing to read in this strip.`;
      return `${where}. Strip ${st.op.strip}: the lanes load W[${j}][${live[0]}] to W[${j}][${live[live.length - 1]}], one contiguous run, and the same columns of x for all ${B} batch rows.` + laneTxt;
    }
    case 'fma': {
      const k = st.ks![lane];
      const before = run.snaps[n].acc[st.warp];
      const laneTxt = k >= 0
        ? ` Lane ${lane}: ${Array.from({ length: B }, (_, b) => `sum ${b} = ${before[b][lane]} + ${run.data.W[j * cfg.K + k]} × ${run.data.X[b * cfg.K + k]} = ${after.acc[st.warp][b][lane]}`).join('; ')}.`
        : '';
      return `${where}. Each lane multiplies its weight by x for every batch row and adds the products to its ${B} running sums, in registers.` + laneTxt;
    }
    case 'shfl':
      return `${who}: shuffle down by ${st.op.offset}. Lane l adds lane l + ${st.op.offset}'s sums to its own, through registers, not memory. After this step lanes 0 to ${st.op.offset! - 1} hold the partial totals.`;
    case 'store':
      return `${who}: lane 0 stores the warp's totals, ${ys(st.touched!)}. With S = 1 no other warp writes these cells, so a plain store is safe.`;
    case 'atomic':
      return `${who}: lane 0 adds the warp's totals with an atomic, which reads, adds and writes in one indivisible step: ${ys(st.touched!)}. These cells now hold chunk${after.parts[st.touched![0]].length > 1 ? 's' : ''} ${after.parts[st.touched![0]].join(', ')}.`;
    case 'read':
      return `${who}: plain +=, part 1 of 2. Lane 0 reads ${st.touched!.map((cell) => `Y[${Math.floor(cell / cfg.N)}][${cell % cfg.N}] = ${run.snaps[n].Y[cell]}`).join(', ')} into registers.`;
    case 'write':
      return st.lost
        ? `${who}: plain +=, part 2 of 2. Lane 0 writes its copy plus its totals: ${ys(st.touched!)}. Another warp wrote these cells after this warp read them, so chunk ${st.lost.join(', ')}'s contribution is overwritten and lost.`
        : `${who}: plain +=, part 2 of 2. Lane 0 writes its copy plus its totals: ${ys(st.touched!)}. Nobody wrote in between, so nothing is lost this time.`;
  }
}

// ---------------------------------------------------------------- the book's tour (pp. 533–536)

export type Decomp = 'output' | 'input' | 'either';
export type Criterion = 'atomics' | 'parallelism' | 'mapping' | 'balance' | 'none';

export const DECOMP_LABEL: Record<Decomp, string> = {
  output: 'output-centric',
  input: 'input-centric',
  either: 'the same either way',
};
export const CRITERION_LABEL: Record<Criterion, string> = {
  atomics: 'avoids atomics',
  parallelism: 'exposes parallelism',
  mapping: 'inputs easy to find',
  balance: 'load balance',
  none: 'atomics not needed either way',
};

export interface TourRow {
  id: string;
  name: string;
  /** Chapter as this course numbers it. */
  chapter: string;
  decomp: Decomp;
  /** Criteria accepted as "what decided it". */
  accept: Criterion[];
  /** Would an input-centric version need atomics? 'idempotent': several writers, same value. */
  inputCentricAtomics: 'needed' | 'one-writer' | 'idempotent';
  pages: string;
  /** The reason, in our words. */
  why: string;
  /** A row of ours, not in the book. */
  ours?: boolean;
  /** The words that identify the row in notes/ch22.md's tour table (book rows only). */
  notesKey?: string;
}

export const TOUR: TourRow[] = [
  {
    id: 'matmul', name: 'Matrix multiplication', chapter: '3, 5, 15', decomp: 'output', accept: ['atomics'], inputCentricAtomics: 'needed', pages: '533–534',
    why: 'A thread per output element, gathering in a register. Every other criterion agrees: plenty of outputs, each output\'s inputs are known (a row and a column), and every output costs the same.',
    notesKey: 'matrix multiplication (Ch. 3, 5, 15)',
  },
  {
    id: 'conv', name: 'Convolution and stencil', chapter: '7, 8', decomp: 'output', accept: ['atomics'], inputCentricAtomics: 'needed', pages: '533–534',
    why: 'As for matrix multiplication: the input-centric version would loop over the outputs each input feeds and update them atomically, and nothing else favours it.',
    notesKey: 'convolution (Ch. 7), stencil (Ch. 8)',
  },
  {
    id: 'hist', name: 'Histogram', chapter: '9', decomp: 'input', accept: ['parallelism', 'mapping', 'balance'], inputCentricAtomics: 'needed', pages: '534',
    why: 'Input-centric, atomics and all. A thread per bin would avoid them but lose on three counts: far fewer bins than inputs (parallelism), every bin thread must inspect every input (mapping), and bins get different counts (balance).',
    notesKey: 'Histogram (Ch. 9)',
  },
  {
    id: 'reduction', name: 'Reduction', chapter: '10', decomp: 'output', accept: ['atomics'], inputCentricAtomics: 'needed', pages: '534',
    why: 'Judged one iteration at a time: each thread produces one element of the iteration\'s output by adding two of its inputs. Scattering instead would make several inputs update one output, which needs atomics.',
    notesKey: 'Reduction (Ch. 10)',
  },
  {
    id: 'scan', name: 'Scan', chapter: '11', decomp: 'output', accept: ['atomics'], inputCentricAtomics: 'needed', pages: '534',
    why: 'Like reduction, output-centric per iteration to avoid atomics. (The book prints "Chapter 12" here; scan is chapter 11.)',
    notesKey: 'scan (book says "Chapter 12"',
  },
  {
    id: 'filter', name: 'Filter (unstable and stable)', chapter: '12', decomp: 'input', accept: ['mapping'], inputCentricAtomics: 'one-writer', pages: '534–535',
    why: 'A thread per input decides whether to keep it and where it goes. A thread per output slot would have to inspect many elements to learn which input lands in its slot. And the stable filter mixes: its scan is itself output-centric.',
    notesKey: 'Unstable and stable filter (Ch. 12)',
  },
  {
    id: 'merge', name: 'Merge', chapter: '13', decomp: 'output', accept: ['balance'], inputCentricAtomics: 'one-writer', pages: '535',
    why: 'Not atomics: every output receives exactly one input, so neither version needs them. Chunks of the output give every thread the same number of elements; chunks of one input would leave threads merging very different amounts (imbalance and divergence).',
    notesKey: 'Merge (Ch. 13)',
  },
  {
    id: 'oddeven', name: 'Odd-even sort', chapter: '14', decomp: 'either', accept: ['none'], inputCentricAtomics: 'one-writer', pages: '533, 535',
    why: 'Each thread reads a pair and writes the same pair. Inputs and outputs correspond so directly that the label does not matter: no gather, no scatter.',
    notesKey: 'Sorting (Ch. 14)',
  },
  {
    id: 'radix', name: 'One radix-sort iteration', chapter: '14', decomp: 'input', accept: ['mapping'], inputCentricAtomics: 'one-writer', pages: '535',
    why: 'Each thread takes an input element and works out where it goes, like the filter it generalises. A thread per output position could not tell which key lands there without looking.',
    notesKey: 'Sorting (Ch. 14)',
  },
  {
    id: 'mergesort', name: 'One merge-sort level', chapter: '14', decomp: 'output', accept: ['balance'], inputCentricAtomics: 'one-writer', pages: '535',
    why: 'Each level is mainly a merge, so it takes merge\'s decomposition, for merge\'s reason.',
    notesKey: 'Sorting (Ch. 14)',
  },
  {
    id: 'wavefront', name: 'Wavefront dynamic programming', chapter: '16', decomp: 'output', accept: ['atomics'], inputCentricAtomics: 'needed', pages: '535',
    why: 'The thread for one cell gathers the earlier cells it depends on. Scattering each finished cell into its successors would need atomics, because several cells feed one successor.',
    notesKey: 'Wavefront / dynamic programming (Ch. 16)',
  },
  {
    id: 'csr', name: 'SpMV with CSR, ELL or JDS', chapter: '17', decomp: 'output', accept: ['atomics'], inputCentricAtomics: 'needed', pages: '535',
    why: 'A thread per output element walks its row\'s non-zeros and gathers.',
    notesKey: 'SpMV/CSR, SpMV/ELL, SpMV/JDS (Ch. 17)',
  },
  {
    id: 'coo', name: 'SpMV with COO', chapter: '17', decomp: 'input', accept: ['balance', 'parallelism', 'mapping'], inputCentricAtomics: 'needed', pages: '535–536',
    why: 'A thread per non-zero, with an atomic per update: the atomics are the price of more parallelism and even load. A thread per row would have to search all the non-zeros for its own.',
    notesKey: 'SpMV/COO (Ch. 17)',
  },
  {
    id: 'csc', name: 'SpMV with CSC', chapter: '17', decomp: 'input', accept: ['mapping'], inputCentricAtomics: 'needed', pages: '536',
    why: 'CSC stores columns, so a thread per input-vector element walks its column and scatters with atomics. A thread per row would have to scan every column to find its row\'s non-zeros.',
    notesKey: 'SpMV/CSC (Ch. 17)',
  },
  {
    id: 'bfs-push', name: 'BFS, vertex-centric push or edge-centric', chapter: '18', decomp: 'input', accept: ['parallelism', 'balance'], inputCentricAtomics: 'idempotent', pages: '536',
    why: 'Threads own vertices or edges and write their neighbours\' levels. No atomics: every writer writes the same value, so the update is idempotent. What decides is parallelism and load balance, and those depend on the graph.',
    notesKey: 'vertex-centric push and edge-centric',
  },
  {
    id: 'bfs-pull', name: 'BFS, vertex-centric pull', chapter: '18', decomp: 'output', accept: ['parallelism', 'balance'], inputCentricAtomics: 'idempotent', pages: '536',
    why: 'Each thread updates only its own vertex\'s level. As for push, atomics are not the question; parallelism and balance are, and they depend on the graph.',
    notesKey: 'vertex-centric pull',
  },
  {
    id: 'epm', name: 'Electrostatic potential map', chapter: '21', decomp: 'output', accept: ['atomics'], inputCentricAtomics: 'needed', pages: '536',
    why: 'Grid-centric (a thread per grid point) gathers; atom-centric scatters into every grid point with atomics. Both expose plenty of parallelism, so atomics decide. For the cutoff version, binning solves the problem of finding each point\'s atoms.',
    notesKey: 'Electrostatic potential map (Ch. 21)',
  },
  {
    id: 'embedding', name: 'Embedding lookup', chapter: 'ours', decomp: 'output', accept: ['mapping'], inputCentricAtomics: 'one-writer', pages: '533 (criteria)', ours: true,
    why: 'out[t][d] = table[token[t]][d]: the owner of an output knows its input from the token id. A thread per table row could scatter without atomics (each output has one source) but would have to search the tokens for its own.',
  },
  {
    id: 'gemv', name: 'Decode GEMV, y = W x', chapter: 'ours', decomp: 'output', accept: ['atomics'], inputCentricAtomics: 'needed', pages: '533–534 (as matrix multiplication)', ours: true,
    why: 'A warp per output gathers in registers and reads W in contiguous runs. When there are too few outputs to keep the device busy, split-K trades some atomics for parallelism: lab 22.1.',
  },
  {
    id: 'softmax', name: 'Softmax normaliser (sum over a row)', chapter: 'ours', decomp: 'output', accept: ['atomics'], inputCentricAtomics: 'needed', pages: '534 (as reduction)', ours: true,
    why: 'The sum of exponentials over a row is a reduction, so by the book\'s reasoning it is output-centric per iteration. Adding every element into one total with atomics would serialise on one address.',
  },
  {
    id: 'tokencounts', name: 'Repetition-penalty token counts (lab 9.3)', chapter: 'ours', decomp: 'input', accept: ['mapping'], inputCentricAtomics: 'needed', pages: '534 (as histogram)', ours: true,
    why: 'A histogram over a vocabulary. Here the bins outnumber the tokens, so the parallelism argument flips; but a thread per vocabulary id would still read every token to find its own. Input-centric, atomics and all.',
  },
];

/** Grade one card: decomposition right, and the criterion one of those accepted. */
export function gradeCard(row: TourRow, decomp: Decomp, criterion: Criterion): { decomp: boolean; criterion: boolean } {
  return { decomp: decomp === row.decomp, criterion: row.accept.includes(criterion) };
}
