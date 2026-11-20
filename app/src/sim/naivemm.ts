// A thread-level model of the book's first matrix multiplication kernel
// (chapter 3, Fig. 3.11, p. 61): one thread per element of P, each thread running
// one dot-product loop and making one store.
//
// What it models: which element a (block, thread) pair owns; the range guard;
// for every step of the loop the two flat row-major indices, the two values and
// the running sum; the store; the blocks of a grid run in any order; the same
// kernel for rectangular matrices with the second operand stored either way
// round; kernels in which a thread owns a whole row or a whole column of the
// output; and how large a matrix a set of launch limits, or a 32-bit index,
// allows.
//
// What it leaves out, and what the app says wherever this model is shown:
//   - Time. There are no caches, no memory bandwidth, no coalescing and no warps
//     here. The model produces indices and counts, and says nothing about which
//     variant is faster. Chapters 5 and 6 are about that; lab 3.4 measures it.
//   - Interleaving. Threads never interact in this kernel (inputs are only read
//     and every output element has one writer), so the model runs one thread at
//     a time and one block at a time. The order is a parameter so that the
//     independence can be checked, not assumed.
//   - Floating point. Values are small integers, so every sum is exact.

/** A block or thread label in the book's order: (y, x), highest dimension first (p. 48). */
export interface YX {
  y: number;
  x: number;
}

// ---------------------------------------------------------------- the square problem of Fig. 3.11

export interface Problem {
  width: number;
  /** Row-major, width x width. */
  M: number[];
  N: number[];
  /** M * N computed directly. */
  ref: number[];
}

/**
 * C[rows x cols] = A[rows x inner] * B[inner x cols], all row-major. The reference
 * the kernels are compared with; written as the textbook triple loop.
 */
export function multiply(A: number[], B: number[], rows: number, inner: number, cols: number): number[] {
  const C = new Array<number>(rows * cols).fill(0);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      for (let i = 0; i < inner; i++) C[r * cols + c] += A[r * inner + i] * B[i * cols + c];
  return C;
}

/** The transpose of a row-major rows x cols matrix, as a row-major cols x rows matrix. */
export function transpose(A: number[], rows: number, cols: number): number[] {
  const T = new Array<number>(rows * cols);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) T[c * rows + r] = A[r * cols + c];
  return T;
}

// Small integers, 1 to 9, so the sums can be checked by hand. The same formulas as
// the chapter 5 model, so that chapter picks up the matrices this one ends with.
const aValue = (r: number, c: number) => ((r * 3 + c * 5 + 1) % 9) + 1;
const bValue = (r: number, c: number) => ((r * 7 + c * 2 + 3) % 9) + 1;

export function makeProblem(width: number): Problem {
  const M: number[] = [];
  const N: number[] = [];
  for (let r = 0; r < width; r++)
    for (let c = 0; c < width; c++) {
      M.push(aValue(r, c));
      N.push(bValue(r, c));
    }
  return { width, M, N, ref: multiply(M, N, width, width, width) };
}

// ---------------------------------------------------------------- the mapping (p. 60)

/** row = blockIdx.y*blockDim.y + threadIdx.y, col = blockIdx.x*blockDim.x + threadIdx.x. */
export function threadFor(blockWidth: number, block: YX, thread: YX): { row: number; col: number } {
  return { row: block.y * blockWidth + thread.y, col: block.x * blockWidth + thread.x };
}

/** The reverse: the one block and thread that own element (row, col). */
export function owners(blockWidth: number, row: number, col: number): { block: YX; thread: YX } {
  return {
    block: { y: Math.floor(row / blockWidth), x: Math.floor(col / blockWidth) },
    thread: { y: row % blockWidth, x: col % blockWidth },
  };
}

// ---------------------------------------------------------------- one thread of the square kernel

/**
 * Line 08 of Fig. 3.11 as printed ("book"), or with the roles of k exchanged in
 * both factors ("swapped"): Pvalue += M[k*Width+row] * N[col*Width+k].
 */
export type Variant = 'book' | 'swapped';

export interface KStep {
  k: number;
  /** Flat index read from M, and from N. */
  mIndex: number;
  nIndex: number;
  mValue: number;
  nValue: number;
  product: number;
  /** Pvalue after this step. */
  sum: number;
}

export interface Trace {
  row: number;
  col: number;
  /** False for a surplus thread: the guard on line 05 fails and nothing else runs. */
  active: boolean;
  steps: KStep[];
  /** The one write, P[row*Width+col] = Pvalue; null for a surplus thread. */
  store: { index: number; value: number } | null;
}

export function trace(prob: Problem, row: number, col: number, variant: Variant = 'book'): Trace {
  const W = prob.width;
  if (!(row < W && col < W)) return { row, col, active: false, steps: [], store: null };
  const steps: KStep[] = [];
  let sum = 0;
  for (let k = 0; k < W; k++) {
    const mIndex = variant === 'book' ? row * W + k : k * W + row;
    const nIndex = variant === 'book' ? k * W + col : col * W + k;
    const mValue = prob.M[mIndex];
    const nValue = prob.N[nIndex];
    sum += mValue * nValue;
    steps.push({ k, mIndex, nIndex, mValue, nValue, product: mValue * nValue, sum });
  }
  return { row, col, active: true, steps, store: { index: row * W + col, value: sum } };
}

/** What one thread executes, one source line at a time: the stepper's unit of progress. */
export type Op =
  | { op: 'row'; value: number } // line 03
  | { op: 'col'; value: number } // line 04
  | { op: 'guard'; pass: boolean } // line 05
  | { op: 'init' } // line 06
  | ({ op: 'mul' } & KStep) // lines 07 to 09, once per k
  | { op: 'store'; index: number; value: number }; // line 10

export function program(prob: Problem, blockWidth: number, block: YX, thread: YX, variant: Variant = 'book'): Op[] {
  const { row, col } = threadFor(blockWidth, block, thread);
  const t = trace(prob, row, col, variant);
  const ops: Op[] = [
    { op: 'row', value: row },
    { op: 'col', value: col },
    { op: 'guard', pass: t.active },
  ];
  if (!t.active || !t.store) return ops;
  ops.push({ op: 'init' });
  for (const s of t.steps) ops.push({ op: 'mul', ...s });
  ops.push({ op: 'store', ...t.store });
  return ops;
}

// ---------------------------------------------------------------- the whole grid, in any block order

export type BlockOrder = 'in-order' | 'reversed' | 'shuffled';

// mulberry32: small, and well mixed even for seeds 1, 2, 3.
function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** The blocks of a gridY x gridX grid in the given order. "shuffled" depends on the seed. */
export function blockOrder(gridY: number, gridX: number, order: BlockOrder, seed = 1): YX[] {
  const blocks: YX[] = [];
  for (let y = 0; y < gridY; y++) for (let x = 0; x < gridX; x++) blocks.push({ y, x });
  if (order === 'reversed') blocks.reverse();
  if (order === 'shuffled') {
    const next = rng(seed);
    for (let i = blocks.length - 1; i > 0; i--) {
      const j = Math.floor(next() * (i + 1));
      [blocks[i], blocks[j]] = [blocks[j], blocks[i]];
    }
  }
  return blocks;
}

export interface Run {
  /** P after every block has run. */
  P: number[];
  /** The order the blocks ran in. */
  order: YX[];
  /** How many threads stored to each element of P. One each, if the mapping is right. */
  writers: number[];
  /** Threads launched, and how many of them passed the guard. */
  threads: number;
  active: number;
  /** P equals M * N. */
  correct: boolean;
}

/** Runs every thread of every block, block after block in the given order. */
export function runAll(
  prob: Problem,
  blockWidth: number,
  order: BlockOrder = 'in-order',
  seed = 1,
  variant: Variant = 'book',
): Run {
  const W = prob.width;
  const g = Math.ceil(W / blockWidth);
  const P = new Array<number>(W * W).fill(NaN);
  const writers = new Array<number>(W * W).fill(0);
  const blocks = blockOrder(g, g, order, seed);
  let threads = 0;
  let active = 0;
  for (const block of blocks)
    for (let ty = 0; ty < blockWidth; ty++)
      for (let tx = 0; tx < blockWidth; tx++) {
        threads++;
        const { row, col } = threadFor(blockWidth, block, { y: ty, x: tx });
        const t = trace(prob, row, col, variant);
        if (!t.store) continue;
        active++;
        P[t.store.index] = t.store.value;
        writers[t.store.index]++;
      }
  return { P, order: blocks, writers, threads, active, correct: P.every((v, i) => v === prob.ref[i]) };
}

// ---------------------------------------------------------------- launch shape and block inputs

export interface Grid {
  /** Blocks along x (covering columns) and along y (covering rows). */
  x: number;
  y: number;
  blocks: number;
  /** Threads launched, threads that own an element, and the rest. */
  threads: number;
  active: number;
  surplus: number;
}

/** The round-up launch for an output of `rows` x `cols` with blockWidth x blockWidth blocks. */
export function gridFor(rows: number, cols: number, blockWidth: number): Grid {
  const x = Math.ceil(cols / blockWidth);
  const y = Math.ceil(rows / blockWidth);
  const threads = x * y * blockWidth * blockWidth;
  return { x, y, blocks: x * y, threads, active: rows * cols, surplus: threads - rows * cols };
}

export interface BlockInputs {
  /** Rows of the first operand, and columns of the second, that this block's threads read. */
  rows: number[];
  cols: number[];
  /** Threads of the block that pass the guard, out of blockWidth squared. */
  active: number;
  threads: number;
}

export function blockInputs(rows: number, cols: number, blockWidth: number, block: YX): BlockInputs {
  const span = (first: number, limit: number) => {
    const out: number[] = [];
    for (let v = first; v < Math.min(first + blockWidth, limit); v++) out.push(v);
    return out;
  };
  const r = span(block.y * blockWidth, rows);
  const c = span(block.x * blockWidth, cols);
  return { rows: r, cols: c, active: r.length * c.length, threads: blockWidth * blockWidth };
}

// ---------------------------------------------------------------- rectangular, two storage conventions

/**
 * How the second operand is stored.
 *   "book":  B is k x l, as in P = M N. Element (i, col) is at i*l + col: the loop walks down a column.
 *   "torch": W is l x k, one output feature per row, the shape nn.Linear keeps. The kernel computes
 *            A times the transpose of W, and element (col, i) is at col*k + i: the loop walks along a row.
 */
export type Layout = 'book' | 'torch';

export interface Rect {
  /** C[j x l] = A[j x k] * B[k x l]. For a linear layer: j the batch, k the inputs, l the outputs. */
  j: number;
  k: number;
  l: number;
  A: number[];
  /** k x l, row-major. */
  B: number[];
  /** l x k, row-major: the transpose of B, the same numbers in the other layout. */
  W: number[];
  /** j x l, row-major: the product. */
  C: number[];
}

export function makeRect(j: number, k: number, l: number): Rect {
  const A: number[] = [];
  const B: number[] = [];
  for (let r = 0; r < j; r++) for (let c = 0; c < k; c++) A.push(aValue(r, c));
  for (let r = 0; r < k; r++) for (let c = 0; c < l; c++) B.push(bValue(r, c));
  return { j, k, l, A, B, W: transpose(B, k, l), C: multiply(A, B, j, k, l) };
}

export interface RectStep {
  i: number;
  aIndex: number;
  /** Flat index into the second operand's buffer: B for "book", W for "torch". */
  bIndex: number;
  aValue: number;
  bValue: number;
  product: number;
  sum: number;
}

export interface RectTrace {
  row: number;
  col: number;
  active: boolean;
  steps: RectStep[];
  store: { index: number; value: number } | null;
  /** How far the second operand's index moves per step of the loop: l for "book", 1 for "torch". */
  stride: number;
}

export function traceRect(rect: Rect, row: number, col: number, layout: Layout = 'book'): RectTrace {
  const { j, k, l } = rect;
  const stride = layout === 'book' ? l : 1;
  if (!(row < j && col < l)) return { row, col, active: false, steps: [], store: null, stride };
  const second = layout === 'book' ? rect.B : rect.W;
  const steps: RectStep[] = [];
  let sum = 0;
  for (let i = 0; i < k; i++) {
    const aIndex = row * k + i;
    const bIndex = layout === 'book' ? i * l + col : col * k + i;
    const a = rect.A[aIndex];
    const b = second[bIndex];
    sum += a * b;
    steps.push({ i, aIndex, bIndex, aValue: a, bValue: b, product: a * b, sum });
  }
  return { row, col, active: true, steps, store: { index: row * l + col, value: sum }, stride };
}

/** Every element of C by the per-element kernel, with the second operand in the given layout. */
export function runRect(rect: Rect, layout: Layout = 'book'): number[] {
  const C: number[] = [];
  for (let row = 0; row < rect.j; row++)
    for (let col = 0; col < rect.l; col++) C.push(traceRect(rect, row, col, layout).store!.value);
  return C;
}

export interface Counts {
  /** Threads that own an element of C. */
  threads: number;
  multiplyAddsPerThread: number;
  /** Reads of global memory per thread: one of each operand per step. */
  readsPerThread: number;
  storesPerThread: number;
  totalMultiplyAdds: number;
}

export function counts(j: number, k: number, l: number): Counts {
  return { threads: j * l, multiplyAddsPerThread: k, readsPerThread: 2 * k, storesPerThread: 1, totalMultiplyAdds: j * k * l };
}

// ---------------------------------------------------------------- thread granularity (exercise 1, p. 64)

/** What one thread produces: one element of P (Fig. 3.11), one row, or one column. */
export type Mode = 'element' | 'row' | 'column';

export interface Granularity {
  mode: Mode;
  /** 2 for the per-element kernel, 1 for the other two. */
  gridDims: 1 | 2;
  grid: { x: number; y: number };
  block: { x: number; y: number };
  /** Threads launched, and threads that pass the guard. */
  threads: number;
  working: number;
  elementsPerThread: number;
  multiplyAddsPerThread: number;
  readsPerThread: number;
  /** Distance in P between one thread's consecutive stores; null when it makes a single store. */
  storeStride: number | null;
  totalMultiplyAdds: number;
}

/**
 * Launch shape and per-thread work for a Width x Width product. The per-element
 * kernel uses blockWidth x blockWidth blocks; the other two use a 1D block with
 * the same number of threads, so the three are compared block for block.
 */
export function granularity(mode: Mode, width: number, blockWidth: number): Granularity {
  const perBlock = blockWidth * blockWidth;
  if (mode === 'element') {
    const g = Math.ceil(width / blockWidth);
    return {
      mode,
      gridDims: 2,
      grid: { x: g, y: g },
      block: { x: blockWidth, y: blockWidth },
      threads: g * g * perBlock,
      working: width * width,
      elementsPerThread: 1,
      multiplyAddsPerThread: width,
      readsPerThread: 2 * width,
      storeStride: null,
      totalMultiplyAdds: width * width * width,
    };
  }
  const g = Math.ceil(width / perBlock);
  return {
    mode,
    gridDims: 1,
    grid: { x: g, y: 1 },
    block: { x: perBlock, y: 1 },
    threads: g * perBlock,
    working: width,
    elementsPerThread: width,
    multiplyAddsPerThread: width * width,
    readsPerThread: 2 * width * width,
    storeStride: mode === 'row' ? 1 : width,
    totalMultiplyAdds: width * width * width,
  };
}

/** The flat indices of P that working thread number `thread` stores, in the order it stores them. */
export function cellsOf(mode: Mode, width: number, thread: number): number[] {
  if (mode === 'element') return [thread];
  const out: number[] = [];
  for (let i = 0; i < width; i++) out.push(mode === 'row' ? thread * width + i : i * width + thread);
  return out;
}

/** What a launch may ask for. Field names follow cudaDeviceProp's maxGridSize, maxThreadsDim, maxThreadsPerBlock. */
export interface Limits {
  gridX: number;
  gridY: number;
  blockX: number;
  blockY: number;
  threadsPerBlock: number;
}

/**
 * The values NVIDIA documents for every listed compute capability
 * (notes/today/ch05-facts.md, A5). The book gives the same grid ranges and the
 * 1024 on p. 47. A device reports its own; these are not a property of the language.
 */
export const CUDA_DOCUMENTED: Limits = {
  gridX: 2 ** 31 - 1,
  gridY: 65535,
  blockX: 1024,
  blockY: 1024,
  threadsPerBlock: 1024,
};

/**
 * The largest Width one grid can cover in the given mode, or 0 when the block
 * shape itself is not allowed. Per element, both grid dimensions must reach
 * Width; per row or per column only x is used.
 */
export function maxWidth(mode: Mode, limits: Limits, blockWidth: number): number {
  const perBlock = blockWidth * blockWidth;
  if (perBlock > limits.threadsPerBlock) return 0;
  if (mode === 'element') {
    if (blockWidth > limits.blockX || blockWidth > limits.blockY) return 0;
    return Math.min(limits.gridX, limits.gridY) * blockWidth;
  }
  if (perBlock > limits.blockX) return 0;
  return limits.gridX * perBlock;
}

/** The last flat index the kernel computes for a Width x Width matrix. */
export const lastIndex = (width: number) => width * width - 1;

export const INT32_MAX = 2 ** 31 - 1;

/** The largest Width whose last flat index still fits in a signed 32-bit int, as the kernel's `int` arithmetic needs. */
export function maxWidthInt32(): number {
  let w = Math.floor(Math.sqrt(INT32_MAX + 1));
  while (lastIndex(w) > INT32_MAX) w--;
  while (lastIndex(w + 1) <= INT32_MAX) w++;
  return w;
}
