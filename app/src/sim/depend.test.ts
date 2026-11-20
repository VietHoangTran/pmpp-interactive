import { describe, expect, it } from 'vitest';
import {
  BINS,
  DECODE_ITEMS,
  LUMA,
  OPS,
  OP_NOTE,
  ORDERS,
  binCounts,
  binItems,
  classify,
  classifyDeps,
  classifyItem,
  combineReads,
  counts,
  defaults,
  depsOf,
  elements,
  footprint,
  hasOutputDependence,
  meanByChunks,
  orderIndependent,
  reads,
  runSequential,
  runThreads,
  sampleData,
  shape,
  stageItems,
  threadOrder,
  timesRead,
  underOrders,
  type Op,
  type Params,
} from './depend';

const P = (op: Op, over: Partial<Params> = {}): Params => ({ ...defaults(op), ...over });
const direct = (op: Op, k: number, p?: Params) => reads(op, k, p).inputs.flatMap((a) => a.indices.map((i) => a.array + i));

describe("the book's grayscale example (pp. 21–23)", () => {
  it('Eq. 2.1: L = 0.299 r + 0.587 g + 0.114 b, and the weights sum to 1', () => {
    expect(LUMA).toEqual({ r: 0.299, g: 0.587, b: 0.114 });
    expect(LUMA.r + LUMA.g + LUMA.b).toBeCloseTo(1, 12);
    const p = P('grayscale', { n: 3 });
    // white, pure red, pure green
    const out = runSequential('grayscale', { I: [1, 1, 1, 1, 0, 0, 0, 1, 0] }, p);
    expect(out[0]).toBeCloseTo(1, 12);
    expect(out[1]).toBeCloseTo(0.299, 12);
    expect(out[2]).toBeCloseTo(0.587, 12);
  });
  it('O[k] reads the three interleaved values of pixel k and nothing else (Fig. 2.2, sidebar p. 22)', () => {
    expect(shape('grayscale').inputs).toEqual([{ name: 'I', length: 24 }]);
    expect(shape('grayscale').output).toEqual({ name: 'O', length: 8 });
    expect(direct('grayscale', 0)).toEqual(['I0', 'I1', 'I2']);
    expect(direct('grayscale', 5)).toEqual(['I15', 'I16', 'I17']);
    for (let k = 0; k < 8; k++) {
      expect(counts('grayscale', k)).toEqual({ inputs: 3, combines: 0, viaCombines: 0, outputsFirst: 0 });
      for (let j = 0; j < 24; j++) expect(timesRead('grayscale', 'I', j)).toBe(1);
    }
  });
});

describe('read sets', () => {
  it('vector add and bias add read one element of each array', () => {
    expect(direct('vecadd', 7)).toEqual(['A7', 'B7']);
    expect(direct('biasadd', 7)).toEqual(['x7', 'b3']); // row 1, column 3 of a 4-wide row
    expect(direct('biasadd', 8)).toEqual(['x8', 'b0']);
    // every bias element is shared by one output per row: 16 / 4 rows
    for (let j = 0; j < 4; j++) expect(timesRead('biasadd', 'b', j)).toBe(4);
  });
  it('blur of radius r reads 2r + 1 inputs, clipped at the ends', () => {
    for (let r = 0; r <= 4; r++) {
      const p = P('blur', { radius: r });
      expect(counts('blur', 8, p).inputs).toBe(2 * r + 1);
      expect(counts('blur', 0, p).inputs).toBe(r + 1);
      expect(counts('blur', 15, p).inputs).toBe(r + 1);
      expect(counts('blur', 8, p).outputsFirst).toBe(0);
    }
    expect(direct('blur', 1, P('blur', { radius: 2 }))).toEqual(['in0', 'in1', 'in2', 'in3']);
  });
  it('with radius 2 an interior input is read by 5 outputs (question q-blur-reads)', () => {
    const p = P('blur', { radius: 2 });
    expect(timesRead('blur', 'in', 8, p)).toBe(5);
    expect(timesRead('blur', 'in', 2, p)).toBe(5);
    expect(timesRead('blur', 'in', 1, p)).toBe(4);
    expect(timesRead('blur', 'in', 0, p)).toBe(3);
  });
  it('the neighbour sum reads two inputs and no outputs; the in-place sum reads one input and one output', () => {
    expect(direct('neighbour', 5)).toEqual(['x4', 'x5']);
    expect(direct('neighbour', 0)).toEqual(['x0']);
    expect(reads('neighbour', 5).outputs).toEqual([]);
    expect(direct('inplace', 5)).toEqual(['x5']);
    expect(reads('inplace', 5).outputs).toEqual([4]);
    expect(reads('inplace', 5).chain).toEqual([4, 3, 2, 1, 0]);
    expect(reads('inplace', 0).chain).toEqual([]);
    expect(counts('inplace', 15).outputsFirst).toBe(15);
    expect(shape('inplace').inPlace).toBe(true);
  });
  it('the mean needs combined values first, and the partial sums read disjoint chunks', () => {
    expect(reads('mean', 0)).toEqual({ inputs: [], combines: [0], outputs: [], chain: [] });
    expect(combineReads('mean', 0).indices.length).toBe(16);
    const p = P('mean', { chunk: 4 });
    expect(shape('mean', p).combine).toEqual({ name: 'partial sums', length: 4 });
    expect(reads('mean', 0, p).combines).toEqual([0, 1, 2, 3]);
    expect(combineReads('mean', 2, p)).toEqual({ array: 'x', indices: [8, 9, 10, 11] });
    expect(counts('mean', 0, p)).toEqual({ inputs: 0, combines: 4, viaCombines: 16, outputsFirst: 0 });
    const all = [0, 1, 2, 3].flatMap((c) => combineReads('mean', c, p).indices);
    expect(new Set(all).size).toBe(16);
  });
  it('softmax reads its own input and the sum of its row', () => {
    expect(shape('softmax').combine).toEqual({ name: 'row sums', length: 2 });
    expect(reads('softmax', 3)).toEqual({ inputs: [{ array: 'x', indices: [3] }], combines: [0], outputs: [], chain: [] });
    expect(reads('softmax', 11).combines).toEqual([1]);
    expect(combineReads('softmax', 1).indices).toEqual([8, 9, 10, 11, 12, 13, 14, 15]);
    expect(counts('softmax', 11)).toEqual({ inputs: 1, combines: 1, viaCombines: 8, outputsFirst: 0 });
  });
  it('the footprint is everything that can change an output', () => {
    expect(footprint('inplace', 3)).toEqual([{ array: 'x', indices: [0, 1, 2, 3] }]);
    expect(footprint('softmax', 11)).toEqual([{ array: 'x', indices: [8, 9, 10, 11, 12, 13, 14, 15] }]);
    expect(footprint('vecadd', 2)).toEqual([{ array: 'A', indices: [2] }, { array: 'B', indices: [2] }]);
  });
});

describe('reads() agrees with what the operation computes', () => {
  // Change one input element; exactly the outputs whose footprint holds it may change.
  for (const op of OPS)
    it(`${op}: an output changes only when an element of its footprint changes`, () => {
      const p = defaults(op);
      const data = sampleData(op, p);
      const base = runSequential(op, data, p);
      for (const [name, values] of Object.entries(data))
        for (let j = 0; j < values.length; j++) {
          const poked = { ...data, [name]: values.map((v, i) => (i === j ? v + 0.5 : v)) };
          const out = runSequential(op, poked, p);
          for (let k = 0; k < base.length; k++) {
            const inPrint = footprint(op, k, p).some((a) => a.array === name && a.indices.includes(j));
            expect(out[k] !== base[k], `${op}: ${name}[${j}] -> output ${k}`).toBe(inPrint);
          }
        }
    });
});

describe('order of the threads', () => {
  it('thread orders are permutations, and the shuffle is fixed', () => {
    for (const o of ORDERS) expect([...threadOrder(o, 16)].sort((a, b) => a - b)).toEqual([...Array(16).keys()]);
    expect(threadOrder('reversed', 4)).toEqual([3, 2, 1, 0]);
    expect(threadOrder('evens-first', 6)).toEqual([0, 2, 4, 1, 3, 5]);
    expect(threadOrder('shuffled', 16)).toEqual(threadOrder('shuffled', 16));
    expect(threadOrder('shuffled', 16)).not.toEqual(threadOrder('forward', 16));
  });
  it('an operation gives the sequential answer in every order exactly when no output reads another output', () => {
    for (const op of OPS) expect(orderIndependent(op), op).toBe(!hasOutputDependence(op));
    expect(OPS.filter((op) => hasOutputDependence(op))).toEqual(['inplace']);
  });
  it('the out-of-place neighbour sum is the same forwards and reversed', () => {
    const x = { x: [3, 1, 7, 0, 4, 1, 6, 3] };
    const p = P('neighbour', { n: 8 });
    const want = [3, 4, 8, 7, 4, 5, 7, 9];
    expect(runSequential('neighbour', x, p)).toEqual(want);
    expect(runThreads('neighbour', x, threadOrder('reversed', 8), p)).toEqual(want);
    expect(runThreads('neighbour', x, threadOrder('shuffled', 8), p)).toEqual(want);
  });
  it('the in-place sum is a prefix sum in loop order and something else reversed', () => {
    // The input is the one chapter 11 of the book uses for prefix sum (p. 252).
    const x = { x: [3, 1, 7, 0, 4, 1, 6, 3] };
    const p = P('inplace', { n: 8 });
    expect(runSequential('inplace', x, p)).toEqual([3, 4, 11, 11, 15, 16, 22, 25]);
    const reversed = runThreads('inplace', x, threadOrder('reversed', 8), p);
    expect(reversed).not.toEqual([3, 4, 11, 11, 15, 16, 22, 25]);
    // Last thread first: every read of x[k - 1] sees the old value, so the result is the neighbour sum.
    expect(reversed).toEqual(runSequential('neighbour', x, P('neighbour', { n: 8 })));
    expect(x.x).toEqual([3, 1, 7, 0, 4, 1, 6, 3]); // the caller's data is not modified
  });
  it('underOrders reports which orders match', () => {
    expect(underOrders('vecadd').every((r) => r.matches)).toBe(true);
    const r = underOrders('inplace');
    expect(r.find((o) => o.order === 'forward')!.matches).toBe(true);
    expect(r.find((o) => o.order === 'reversed')!.matches).toBe(false);
  });
});

describe('the average splits into independent partial sums (p. 21)', () => {
  it('any chunk size gives the same total on integers', () => {
    const x = sampleData('mean').x;
    const want = runSequential('mean', { x })[0];
    for (const chunk of [16, 8, 4, 2, 1, 5]) {
      const m = meanByChunks(x, chunk);
      expect(m.partials.length).toBe(Math.ceil(16 / chunk));
      expect(m.mean).toBe(want);
    }
    expect(meanByChunks([1, 2, 3, 4], 2)).toEqual({ partials: [3, 7], total: 10, mean: 2.5 });
  });
});

describe('the four bins', () => {
  it('the rule is applied in order: chain, combine, many reads, per element', () => {
    expect(classifyDeps({ reads: () => 2, combines: 0, prior: false })).toBe('elementwise');
    expect(classifyDeps({ reads: (s) => s, combines: 0, prior: false })).toBe('manyreads');
    expect(classifyDeps({ reads: (s) => s, combines: 1, prior: false })).toBe('combine');
    expect(classifyDeps({ reads: (s) => s, combines: 1, prior: true })).toBe('chain');
  });
  it('explorer operations, measured from their read sets', () => {
    const got = Object.fromEntries(OPS.map((op) => [op, classify(op)]));
    expect(got).toEqual({
      grayscale: 'elementwise',
      vecadd: 'elementwise',
      biasadd: 'elementwise',
      blur: 'manyreads',
      mean: 'combine',
      neighbour: 'elementwise',
      inplace: 'chain',
      softmax: 'combine',
    });
    expect(depsOf('blur').reads(8)).toBe(17);
    expect(depsOf('grayscale').reads(8)).toBe(3);
  });
  it('the eleven operations of a decode step', () => {
    expect(DECODE_ITEMS.map((i) => i.id)).toEqual([
      'residual', 'bias', 'scale', 'gelu', 'linear', 'layernorm', 'kvappend', 'softmax', 'argmax', 'cumprob', 'genloop',
    ]);
    const got = Object.fromEntries(DECODE_ITEMS.map((i) => [i.id, classifyItem(i.id)]));
    expect(got).toEqual({
      residual: 'elementwise',
      bias: 'elementwise',
      scale: 'elementwise',
      gelu: 'elementwise',
      linear: 'manyreads',
      layernorm: 'combine',
      softmax: 'combine',
      argmax: 'combine',
      cumprob: 'chain',
      kvappend: 'elementwise',
      genloop: 'chain',
    });
    expect(binCounts()).toEqual({ elementwise: 5, manyreads: 1, combine: 3, chain: 2 });
    expect(BINS.reduce((s, b) => s + binCounts()[b], 0)).toBe(11);
    // The linear layer is the only one whose per-output reads grow with the size.
    expect(DECODE_ITEMS.filter((i) => i.deps.reads(4096) > i.deps.reads(8)).map((i) => i.id)).toEqual(['linear']);
    expect(DECODE_ITEMS.find((i) => i.id === 'linear')!.deps.reads(4096)).toBe(8192);
  });
  it('by stage: what the closing summary of the sort says', () => {
    // The list runs forward pass, then sampling, then the loop around the step.
    expect(DECODE_ITEMS.map((i) => i.stage)).toEqual([
      'forward', 'forward', 'forward', 'forward', 'forward', 'forward', 'forward', 'sampling', 'sampling', 'sampling', 'loop',
    ]);
    expect(stageItems('forward').length).toBe(7);
    // "5 of the 7 operations inside the forward pass are per element."
    expect(binCounts('forward')).toEqual({ elementwise: 5, manyreads: 1, combine: 1, chain: 0 });
    expect(binCounts('sampling')).toEqual({ elementwise: 0, manyreads: 0, combine: 2, chain: 1 });
    expect(binCounts('loop')).toEqual({ elementwise: 0, manyreads: 0, combine: 0, chain: 1 });
    // The bins the summary names item by item.
    expect(binItems('manyreads')).toEqual(['linear']);
    expect(binItems('combine')).toEqual(['layernorm', 'softmax', 'argmax']);
    expect(binItems('chain')).toEqual(['cumprob', 'genloop']);
  });
  it('softmax in the decode list combines two values (max, then sum); the explorer keeps the plain formula with one', () => {
    expect(DECODE_ITEMS.find((i) => i.id === 'softmax')!.deps.combines).toBe(2);
    expect(depsOf('softmax').combines).toBe(1);
    expect(OP_NOTE.softmax).toContain('row maximum');
  });
});

describe('numbers the lessons state', () => {
  it('a residual add over 8 sequences of 2048 tokens with hidden size 4096 is 67,108,864 threads', () => {
    expect(elements(8, 2048, 4096)).toBe(67108864);
  });
});
