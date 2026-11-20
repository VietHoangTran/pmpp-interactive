// Dependency model for chapter 2, section 2.1 (book pp. 21–23).
//
// The book defines data parallelism by one property: computations on different
// parts of the data can be done independently (p. 21), and it shows the property
// with a picture in which each output O[k] reads only its own input I[k]
// (Fig. 2.2, p. 23). This file states that property so it can be tested. For an
// operation and one output index k it answers three questions:
//
//   reads(op, k)         which input elements does the thread for output k read
//                        itself, which combined values must exist first, and
//                        which OTHER OUTPUTS must exist first?
//   classify(op)         which of four dependency shapes is that?
//   runThreads(op, ...)  what does the operation produce when each output is
//                        computed by its own thread and the threads run, one
//                        after another, in a given order?
//
// The claim the lesson makes, and depend.test.ts checks for every operation here:
// the result is the same in every order exactly when no output reads another
// output.
//
// From the book: the grayscale weights (Eq. 2.1, p. 22), the interleaved r g b
// layout (sidebar, p. 22), and the three graded cases of p. 21 (one pixel, a
// small neighbourhood, an average that splits into independent parts). The
// other operations, the four bins and the list of decode-step operations are
// ours.
//
// What the model leaves out:
//   - Timing and caches. It says which elements are read, never how long a
//     read takes or whether a second read of the same element is cheaper.
//   - Interleaving. runThreads runs each thread's whole body before the next
//     thread starts. Real device threads interleave instruction by instruction,
//     which gives an in-place loop even more possible results than the ones
//     shown here.
//   - Floating-point regrouping. Splitting a sum into partial sums changes the
//     rounding (book appendix A, pp. 593–594). The data here are small integers
//     wherever a sum is regrouped, so regrouped sums are exact.
//
// No imports and no syntax that needs compiling away beyond types, so the
// browser checks can load this file directly.

export type Op = 'grayscale' | 'vecadd' | 'biasadd' | 'blur' | 'mean' | 'neighbour' | 'inplace' | 'softmax';

export const OPS: Op[] = ['grayscale', 'vecadd', 'biasadd', 'blur', 'mean', 'neighbour', 'inplace', 'softmax'];

export const OP_LABEL: Record<Op, string> = {
  grayscale: 'Grayscale',
  vecadd: 'Vector add',
  biasadd: 'Bias add',
  blur: '1D blur',
  mean: 'Mean of all elements',
  neighbour: 'Neighbour sum, out of place',
  inplace: 'Running sum, in place',
  softmax: 'Softmax over a row',
};

/** The body of the sequential loop over k, as one line of C. */
export const OP_BODY: Record<Op, string> = {
  grayscale: 'O[k] = 0.299f*I[3*k] + 0.587f*I[3*k + 1] + 0.114f*I[3*k + 2];',
  vecadd: 'C[k] = A[k] + B[k];',
  biasadd: 'y[k] = x[k] + b[k % d];',
  blur: 'out[k] = average of in[k - r] ... in[k + r], skipping positions outside the array;',
  mean: 'mean = (x[0] + x[1] + ... + x[n - 1]) / n;',
  neighbour: 'y[k] = x[k - 1] + x[k];',
  inplace: 'x[k] = x[k - 1] + x[k];',
  softmax: 'y[k] = exp(x[k]) / (sum of exp(x[j]) over the row of k);',
};

/** A convention the one-line body leaves unsaid. */
export const OP_NOTE: Partial<Record<Op, string>> = {
  neighbour: 'x[-1] counts as 0',
  inplace: 'x[-1] counts as 0',
  blur: 'r is the radius',
  biasadd: 'd is the width of a row',
};

/** The grayscale weights of the book's Eq. 2.1 (p. 22): L = 0.299 r + 0.587 g + 0.114 b. */
export const LUMA = { r: 0.299, g: 0.587, b: 0.114 };

export interface Params {
  /** Number of outputs (for the mean: number of inputs; it has one output). */
  n: number;
  /** Row width, for the operations that work on rows (bias add, softmax). n is rows x d. */
  d: number;
  /** Blur radius: each output reads 2 x radius + 1 inputs, fewer at the ends. */
  radius: number;
  /** Mean: how many inputs go into one partial sum. chunk = n means no split. */
  chunk: number;
}

/** The sizes the explorer opens each operation with. */
export function defaults(op: Op): Params {
  const base: Params = { n: 16, d: 4, radius: 1, chunk: 16 };
  if (op === 'grayscale') return { ...base, n: 8 };
  if (op === 'softmax') return { ...base, d: 8 };
  return base;
}

export interface ArrayShape {
  name: string;
  length: number;
}

export interface Shape {
  /** The arrays the operation only reads. */
  inputs: ArrayShape[];
  /** Values combined from many inputs that outputs depend on (partial sums, a row sum), if any. */
  combine: ArrayShape | null;
  output: ArrayShape;
  /** True when the output is written into the same memory as an input. */
  inPlace: boolean;
}

const combineCount = (op: Op, p: Params): number =>
  op === 'mean' ? Math.ceil(p.n / p.chunk) : op === 'softmax' ? Math.ceil(p.n / p.d) : 0;

export function shape(op: Op, p: Params = defaults(op)): Shape {
  switch (op) {
    case 'grayscale':
      return { inputs: [{ name: 'I', length: 3 * p.n }], combine: null, output: { name: 'O', length: p.n }, inPlace: false };
    case 'vecadd':
      return { inputs: [{ name: 'A', length: p.n }, { name: 'B', length: p.n }], combine: null, output: { name: 'C', length: p.n }, inPlace: false };
    case 'biasadd':
      return { inputs: [{ name: 'x', length: p.n }, { name: 'b', length: p.d }], combine: null, output: { name: 'y', length: p.n }, inPlace: false };
    case 'blur':
      return { inputs: [{ name: 'in', length: p.n }], combine: null, output: { name: 'out', length: p.n }, inPlace: false };
    case 'mean':
      return {
        inputs: [{ name: 'x', length: p.n }],
        combine: { name: 'partial sums', length: combineCount(op, p) },
        output: { name: 'mean', length: 1 },
        inPlace: false,
      };
    case 'neighbour':
      return { inputs: [{ name: 'x', length: p.n }], combine: null, output: { name: 'y', length: p.n }, inPlace: false };
    case 'inplace':
      return { inputs: [{ name: 'x', length: p.n }], combine: null, output: { name: 'x', length: p.n }, inPlace: true };
    case 'softmax':
      return {
        inputs: [{ name: 'x', length: p.n }],
        combine: { name: 'row sums', length: combineCount(op, p) },
        output: { name: 'y', length: p.n },
        inPlace: false,
      };
  }
}

export interface ArrayReads {
  array: string;
  indices: number[];
}

export interface Reads {
  /** Input elements the thread for this output reads itself, per input array. */
  inputs: ArrayReads[];
  /** Combined values that must exist before this output can be computed (indices into shape().combine). */
  combines: number[];
  /** Other outputs this one reads directly. */
  outputs: number[];
  /** Every output that must exist first, nearest first: `outputs`, then what those need, and so on. */
  chain: number[];
}

const range = (lo: number, hi: number): number[] => {
  const out: number[] = [];
  for (let i = lo; i <= hi; i++) out.push(i);
  return out;
};

/** What the computation of output k reads. */
export function reads(op: Op, k: number, p: Params = defaults(op)): Reads {
  const none: Reads = { inputs: [], combines: [], outputs: [], chain: [] };
  switch (op) {
    case 'grayscale':
      return { ...none, inputs: [{ array: 'I', indices: [3 * k, 3 * k + 1, 3 * k + 2] }] };
    case 'vecadd':
      return { ...none, inputs: [{ array: 'A', indices: [k] }, { array: 'B', indices: [k] }] };
    case 'biasadd':
      return { ...none, inputs: [{ array: 'x', indices: [k] }, { array: 'b', indices: [k % p.d] }] };
    case 'blur':
      return { ...none, inputs: [{ array: 'in', indices: range(Math.max(0, k - p.radius), Math.min(p.n - 1, k + p.radius)) }] };
    case 'mean':
      return { ...none, combines: range(0, combineCount(op, p) - 1) };
    case 'neighbour':
      return { ...none, inputs: [{ array: 'x', indices: k > 0 ? [k - 1, k] : [k] }] };
    case 'inplace':
      return {
        inputs: [{ array: 'x', indices: [k] }],
        combines: [],
        outputs: k > 0 ? [k - 1] : [],
        chain: range(0, k - 1).reverse(),
      };
    case 'softmax':
      return { ...none, inputs: [{ array: 'x', indices: [k] }], combines: [Math.floor(k / p.d)] };
  }
}

/** The inputs that go into combined value c. Empty for operations with no combine step. */
export function combineReads(op: Op, c: number, p: Params = defaults(op)): ArrayReads {
  if (op === 'mean') return { array: 'x', indices: range(c * p.chunk, Math.min(p.n, (c + 1) * p.chunk) - 1) };
  if (op === 'softmax') return { array: 'x', indices: range(c * p.d, Math.min(p.n, (c + 1) * p.d) - 1) };
  return { array: '', indices: [] };
}

export interface Counts {
  /** Input elements the output's own thread reads. */
  inputs: number;
  /** Combined values it needs first. */
  combines: number;
  /** Input elements those combined values are built from. */
  viaCombines: number;
  /** Other outputs that must exist first, counting the whole chain. */
  outputsFirst: number;
}

export function counts(op: Op, k: number, p: Params = defaults(op)): Counts {
  const r = reads(op, k, p);
  return {
    inputs: r.inputs.reduce((s, a) => s + a.indices.length, 0),
    combines: r.combines.length,
    viaCombines: r.combines.reduce((s, c) => s + combineReads(op, c, p).indices.length, 0),
    outputsFirst: r.chain.length,
  };
}

/**
 * Every input element that can influence output k: what it reads itself, what
 * its combined values read, and what the outputs before it in a chain read.
 */
export function footprint(op: Op, k: number, p: Params = defaults(op)): ArrayReads[] {
  const seen = new Map<string, Set<number>>();
  const add = (a: ArrayReads) => {
    if (!a.indices.length) return;
    const s = seen.get(a.array) ?? new Set<number>();
    a.indices.forEach((i) => s.add(i));
    seen.set(a.array, s);
  };
  for (const out of [k, ...reads(op, k, p).chain]) {
    const r = reads(op, out, p);
    r.inputs.forEach(add);
    r.combines.forEach((c) => add(combineReads(op, c, p)));
  }
  return [...seen].map(([array, s]) => ({ array, indices: [...s].sort((a, b) => a - b) }));
}

/** Across all outputs, how many of them read element j of an input array themselves. */
export function timesRead(op: Op, array: string, j: number, p: Params = defaults(op)): number {
  let count = 0;
  for (let k = 0; k < shape(op, p).output.length; k++)
    if (reads(op, k, p).inputs.some((a) => a.array === array && a.indices.includes(j))) count++;
  return count;
}

// ---------------------------------------------------------------- the four bins
export type Bin = 'elementwise' | 'manyreads' | 'combine' | 'chain';

export const BINS: Bin[] = ['elementwise', 'manyreads', 'combine', 'chain'];

export const BIN_LABEL: Record<Bin, string> = {
  elementwise: 'Independent per element',
  manyreads: 'Independent per output, many reads',
  combine: 'Needs a combine first',
  chain: 'A chain',
};

export const BIN_LONG: Record<Bin, string> = {
  elementwise: 'Independent per element: one thread per output, and one line of work in it.',
  manyreads: 'Independent per output, but each output reads many inputs: a loop inside every thread.',
  combine: 'Needs a combine across elements first: some value built from a whole row or array.',
  chain: 'A chain: each result needs the previous one.',
};

/** The dependency facts a bin is decided from. */
export interface Deps {
  /** Inputs one output's thread reads itself, when the size of the problem is `size`. */
  reads: (size: number) => number;
  /** Values combined from a whole row or array that must exist first. */
  combines: number;
  /** True when an output needs an earlier output of the same operation. */
  prior: boolean;
}

/**
 * The rule, in order: a chain if an output needs another output; otherwise
 * "combine first" if it needs a combined value; otherwise "many reads" if the
 * number of inputs one output reads grows with the size of the problem;
 * otherwise independent per element.
 */
export function classifyDeps(d: Deps): Bin {
  if (d.prior) return 'chain';
  if (d.combines > 0) return 'combine';
  return d.reads(64) > d.reads(32) ? 'manyreads' : 'elementwise';
}

/** The explorer's sizes scaled so that the "size of the problem" is `size`. */
function sized(op: Op, size: number): Params {
  return { n: 4 * size, d: size, radius: size, chunk: 4 * size };
}

/** The dependency facts of an explorer operation, measured from reads(). */
export function depsOf(op: Op): Deps {
  const most = (size: number, pick: (r: Reads) => number) => {
    const p = sized(op, size);
    let m = 0;
    for (let k = 0; k < shape(op, p).output.length; k++) m = Math.max(m, pick(reads(op, k, p)));
    return m;
  };
  return {
    reads: (size) => most(size, (r) => r.inputs.reduce((s, a) => s + a.indices.length, 0)),
    combines: most(8, (r) => r.combines.length),
    prior: most(8, (r) => r.outputs.length) > 0,
  };
}

export function classify(op: Op): Bin {
  return classifyDeps(depsOf(op));
}

// ---------------------------------------------------------------- running an operation
export type Data = Record<string, number[]>;

/** Small test data for an operation: integers, except grayscale's intensities between 0 and 1. */
export function sampleData(op: Op, p: Params = defaults(op)): Data {
  const ints = (len: number, mul: number, add: number, mod: number) =>
    Array.from({ length: len }, (_, i) => ((i * mul + add) % mod) + 1);
  switch (op) {
    case 'grayscale':
      return { I: Array.from({ length: 3 * p.n }, (_, i) => ((i * 7 + 3) % 11) / 10) };
    case 'vecadd':
      return { A: ints(p.n, 3, 1, 7), B: ints(p.n, 5, 2, 9) };
    case 'biasadd':
      return { x: ints(p.n, 3, 1, 7), b: ints(p.d, 4, 0, 5).map((v) => v * 10) };
    case 'blur':
      return { in: ints(p.n, 5, 2, 9) };
    case 'softmax':
      return { x: ints(p.n, 3, 1, 4) };
    default:
      return { x: ints(p.n, 3, 1, 7) };
  }
}

/** One thread's work: compute output k from the memory as it is now, and store it. */
function body(op: Op, k: number, mem: Data, out: number[], p: Params): void {
  switch (op) {
    case 'grayscale':
      out[k] = LUMA.r * mem.I[3 * k] + LUMA.g * mem.I[3 * k + 1] + LUMA.b * mem.I[3 * k + 2];
      return;
    case 'vecadd':
      out[k] = mem.A[k] + mem.B[k];
      return;
    case 'biasadd':
      out[k] = mem.x[k] + mem.b[k % p.d];
      return;
    case 'blur': {
      let sum = 0;
      let count = 0;
      for (let j = k - p.radius; j <= k + p.radius; j++)
        if (j >= 0 && j < p.n) {
          sum += mem.in[j];
          count++;
        }
      out[k] = sum / count;
      return;
    }
    case 'mean': {
      let sum = 0;
      for (let j = 0; j < p.n; j++) sum += mem.x[j];
      out[k] = sum / p.n;
      return;
    }
    case 'neighbour':
      out[k] = (k > 0 ? mem.x[k - 1] : 0) + mem.x[k];
      return;
    case 'inplace':
      // `out` is mem.x here: the read of x[k - 1] sees whatever is there now.
      out[k] = (k > 0 ? mem.x[k - 1] : 0) + mem.x[k];
      return;
    case 'softmax': {
      const row = Math.floor(k / p.d);
      let sum = 0;
      for (let j = row * p.d; j < Math.min(p.n, (row + 1) * p.d); j++) sum += Math.exp(mem.x[j]);
      out[k] = Math.exp(mem.x[k]) / sum;
      return;
    }
  }
}

/**
 * One thread per output; the threads run to completion one after another in
 * `order` (a permutation of the output indices). Returns the output array.
 */
export function runThreads(op: Op, data: Data, order: number[], p: Params = defaults(op)): number[] {
  const s = shape(op, p);
  const mem: Data = Object.fromEntries(Object.entries(data).map(([name, v]) => [name, [...v]]));
  const out = s.inPlace ? mem[s.output.name] : new Array<number>(s.output.length).fill(NaN);
  for (const k of order) body(op, k, mem, out, p);
  return out;
}

/** The sequential loop: k = 0, 1, 2, ... */
export function runSequential(op: Op, data: Data, p: Params = defaults(op)): number[] {
  return runThreads(op, data, range(0, shape(op, p).output.length - 1), p);
}

export type OrderName = 'forward' | 'reversed' | 'evens-first' | 'shuffled';
export const ORDERS: OrderName[] = ['forward', 'reversed', 'evens-first', 'shuffled'];
export const ORDER_LABEL: Record<OrderName, string> = {
  forward: 'thread 0 first',
  reversed: 'last thread first',
  'evens-first': 'even threads, then odd',
  shuffled: 'a fixed shuffle',
};

/** A named order of n threads. "shuffled" is a fixed permutation, the same every time. */
export function threadOrder(name: OrderName, n: number): number[] {
  const ids = range(0, n - 1);
  if (name === 'forward') return ids;
  if (name === 'reversed') return ids.reverse();
  if (name === 'evens-first') return [...ids.filter((i) => i % 2 === 0), ...ids.filter((i) => i % 2 === 1)];
  // A small linear congruential generator drives a Fisher-Yates shuffle: no Math.random, so it is repeatable.
  let seed = 12345;
  const next = () => (seed = (seed * 48271) % 2147483647);
  for (let i = n - 1; i > 0; i--) {
    const j = next() % (i + 1);
    [ids[i], ids[j]] = [ids[j], ids[i]];
  }
  return ids;
}

const sameArray = (a: number[], b: number[]) => a.length === b.length && a.every((v, i) => v === b[i]);

export interface OrderResult {
  order: OrderName;
  result: number[];
  /** Equal to the sequential loop's result, element for element. */
  matches: boolean;
}

/** The operation run under each named thread order, compared with the sequential loop. */
export function underOrders(op: Op, data: Data = sampleData(op), p: Params = defaults(op)): OrderResult[] {
  const want = runSequential(op, data, p);
  const n = shape(op, p).output.length;
  return ORDERS.map((order) => {
    const result = runThreads(op, data, threadOrder(order, n), p);
    return { order, result, matches: sameArray(result, want) };
  });
}

/** True when every named thread order gives the sequential result. */
export function orderIndependent(op: Op, data: Data = sampleData(op), p: Params = defaults(op)): boolean {
  return underOrders(op, data, p).every((r) => r.matches);
}

/** True when some output of the operation reads another output. */
export function hasOutputDependence(op: Op, p: Params = defaults(op)): boolean {
  for (let k = 0; k < shape(op, p).output.length; k++) if (reads(op, k, p).outputs.length) return true;
  return false;
}

/**
 * The mean, computed the way the book says it can be (p. 21): partial sums over
 * chunks, which do not depend on each other, then one combine of the partials.
 */
export function meanByChunks(x: number[], chunk: number): { partials: number[]; total: number; mean: number } {
  const partials: number[] = [];
  for (let lo = 0; lo < x.length; lo += chunk) {
    let s = 0;
    for (let j = lo; j < Math.min(x.length, lo + chunk); j++) s += x[j];
    partials.push(s);
  }
  const total = partials.reduce((a, b) => a + b, 0);
  return { partials, total, mean: total / x.length };
}

/** One thread per element: the number of threads is the product of the dimensions. */
export function elements(...dims: number[]): number {
  return dims.reduce((a, b) => a * b, 1);
}

// ---------------------------------------------------------------- one decode step, sorted
// The operations of one decode step of a transformer, each described by the
// same three dependency facts. The list and the mapping are ours; the book's
// section 2.1 uses images only. `where` names the chapters of this course in
// which the kernel shape is written (PLAN.md, curriculum map and milestones).
export interface DecodeItem {
  id: string;
  label: string;
  /** The line of work for one output, as pseudocode. */
  body: string;
  /** What "size" means for this item when reads() is evaluated. */
  sizeName: string;
  deps: Deps;
  /** Chapter ids where this course writes the kernel shape. */
  where: string[];
}

export const DECODE_ITEMS: DecodeItem[] = [
  {
    id: 'residual',
    label: 'Residual add',
    body: 'y[i] = x[i] + r[i]',
    sizeName: 'hidden size',
    deps: { reads: () => 2, combines: 0, prior: false },
    where: ['ch02'],
  },
  {
    id: 'bias',
    label: 'Bias add',
    body: 'y[t][j] = x[t][j] + b[j]',
    sizeName: 'hidden size',
    deps: { reads: () => 2, combines: 0, prior: false },
    where: ['ch02'],
  },
  {
    id: 'scale',
    label: 'Scaling by a constant',
    body: 'y[i] = a * x[i]',
    sizeName: 'hidden size',
    deps: { reads: () => 1, combines: 0, prior: false },
    where: ['ch02'],
  },
  {
    id: 'gelu',
    label: 'GELU',
    body: 'y[i] = gelu(x[i])',
    sizeName: 'hidden size',
    deps: { reads: () => 1, combines: 0, prior: false },
    where: ['ch02'],
  },
  {
    id: 'linear',
    label: 'The matmul of a linear layer',
    body: 'y[j] = W[j][0]*x[0] + W[j][1]*x[1] + ... + W[j][d-1]*x[d-1]',
    sizeName: 'input width d',
    deps: { reads: (d) => 2 * d, combines: 0, prior: false },
    where: ['ch03', 'ch05', 'ch15'],
  },
  {
    id: 'layernorm',
    label: 'Layer norm',
    body: 'y[i] = g[i] * (x[i] - mean) / sqrt(var + eps) + b[i]',
    sizeName: 'hidden size',
    deps: { reads: () => 3, combines: 2, prior: false },
    where: ['ch10'],
  },
  {
    id: 'softmax',
    label: 'Softmax over the vocabulary',
    body: 'p[v] = exp(z[v]) / (sum of exp(z[u]) over the vocabulary)',
    sizeName: 'vocabulary size',
    deps: { reads: () => 1, combines: 1, prior: false },
    where: ['ch10'],
  },
  {
    id: 'argmax',
    label: 'Argmax',
    body: 'token = index of the largest z[v]',
    sizeName: 'vocabulary size',
    deps: { reads: () => 0, combines: 1, prior: false },
    where: ['ch10'],
  },
  {
    id: 'cumprob',
    label: 'Cumulative probability for top-p',
    body: 'c[i] = c[i-1] + p[i]',
    sizeName: 'vocabulary size',
    deps: { reads: () => 1, combines: 0, prior: true },
    where: ['ch11', 'x1'],
  },
  {
    id: 'kvappend',
    label: 'Appending to the KV cache',
    body: 'cache[t][j] = k_new[j]',
    sizeName: 'hidden size',
    deps: { reads: () => 1, combines: 0, prior: false },
    where: ['ch02', 'ch20'],
  },
  {
    id: 'genloop',
    label: 'The token-by-token generation loop',
    body: 'token[t] = decode(token[0], ..., token[t-1])',
    sizeName: 'tokens so far',
    deps: { reads: () => 0, combines: 0, prior: true },
    where: ['ch20'],
  },
];

export function classifyItem(id: string): Bin {
  const item = DECODE_ITEMS.find((i) => i.id === id);
  if (!item) throw new Error('no such decode item: ' + id);
  return classifyDeps(item.deps);
}

/** How many of the decode items fall in each bin. */
export function binCounts(): Record<Bin, number> {
  const out: Record<Bin, number> = { elementwise: 0, manyreads: 0, combine: 0, chain: 0 };
  for (const item of DECODE_ITEMS) out[classifyDeps(item.deps)]++;
  return out;
}
