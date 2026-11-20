import { describe, expect, it } from 'vitest';
import {
  CUDA_DOCUMENTED,
  INT32_MAX,
  blockInputs,
  blockOrder,
  cellsOf,
  counts,
  granularity,
  gridFor,
  lastIndex,
  makeProblem,
  makeRect,
  maxWidth,
  maxWidthInt32,
  multiply,
  owners,
  program,
  runAll,
  runRect,
  threadFor,
  trace,
  traceRect,
  transpose,
  type BlockOrder,
  type Mode,
} from './naivemm';
import { makeProblem as makeTiledProblem } from './tiled';

const prob = makeProblem(4);
const B00 = { y: 0, x: 0 };
const T00 = { y: 0, x: 0 };

describe('the mapping (book p. 60, Fig. 3.12)', () => {
  it('Thread (0,0) of Block (0,0) owns P[0][0]; Thread (0,0) of Block (1,0) owns P[2][0] (p. 62)', () => {
    expect(threadFor(2, B00, T00)).toEqual({ row: 0, col: 0 });
    expect(threadFor(2, { y: 1, x: 0 }, T00)).toEqual({ row: 2, col: 0 });
  });
  it('the four threads of Block (0,0) own P[0][0], P[0][1], P[1][0], P[1][1] (Fig. 3.12)', () => {
    const got = [
      [0, 0],
      [0, 1],
      [1, 0],
      [1, 1],
    ].map(([y, x]) => threadFor(2, B00, { y, x }));
    expect(got).toEqual([
      { row: 0, col: 0 },
      { row: 0, col: 1 },
      { row: 1, col: 0 },
      { row: 1, col: 1 },
    ]);
  });
  it('owners is the inverse of threadFor', () => {
    for (const bw of [2, 3, 4, 8])
      for (let row = 0; row < 13; row++)
        for (let col = 0; col < 13; col++) {
          const o = owners(bw, row, col);
          expect(threadFor(bw, o.block, o.thread)).toEqual({ row, col });
          expect(o.thread.y).toBeLessThan(bw);
          expect(o.thread.x).toBeLessThan(bw);
        }
  });
});

describe("the book's hand trace (p. 63)", () => {
  it('Thread (0,0) of Block (0,0): M indices 0,1,2,3, N indices 0,4,8,12, then P[0]', () => {
    const t = trace(prob, 0, 0);
    expect(t.steps.map((s) => s.mIndex)).toEqual([0, 1, 2, 3]);
    expect(t.steps.map((s) => s.nIndex)).toEqual([0, 4, 8, 12]);
    expect(t.store!.index).toBe(0);
    expect(t.store!.value).toBe(prob.ref[0]);
  });
  it('the values and running sums of that thread, by hand', () => {
    // Row 0 of M is 2 7 3 8, column 0 of N is 4 2 9 7.
    const t = trace(prob, 0, 0);
    expect(t.steps.map((s) => s.mValue)).toEqual([2, 7, 3, 8]);
    expect(t.steps.map((s) => s.nValue)).toEqual([4, 2, 9, 7]);
    expect(t.steps.map((s) => s.sum)).toEqual([8, 22, 49, 105]);
  });
  it('Thread (1,0) of Block (0,0) has row 1, col 0, writes P[4], and at k = 2 reads N[8] and M[6]', () => {
    const { row, col } = threadFor(2, B00, { y: 1, x: 0 });
    expect([row, col]).toEqual([1, 0]);
    const t = trace(prob, row, col);
    expect(t.store!.index).toBe(4);
    expect(t.steps[2].nIndex).toBe(8);
    expect(t.steps[2].mIndex).toBe(6);
  });
  it('Thread (0,0) of Block (1,0) writes P[8]', () => {
    const { row, col } = threadFor(2, { y: 1, x: 0 }, T00);
    expect(trace(prob, row, col).store!.index).toBe(8);
  });
  it('M moves by 1 per step and N by Width, for every thread', () => {
    for (const W of [4, 5, 7]) {
      const p = makeProblem(W);
      for (let row = 0; row < W; row++)
        for (let col = 0; col < W; col++) {
          const s = trace(p, row, col).steps;
          for (let k = 1; k < W; k++) {
            expect(s[k].mIndex - s[k - 1].mIndex).toBe(1);
            expect(s[k].nIndex - s[k - 1].nIndex).toBe(W);
          }
        }
    }
  });
  it('the stepper program: row, col, guard, init, one step per k, store', () => {
    const ops = program(prob, 2, B00, T00);
    expect(ops.map((o) => o.op)).toEqual(['row', 'col', 'guard', 'init', 'mul', 'mul', 'mul', 'mul', 'store']);
    expect(ops[8]).toEqual({ op: 'store', index: 0, value: 105 });
  });
  it('a surplus thread stops at the guard', () => {
    const p5 = makeProblem(5);
    const ops = program(p5, 2, { y: 2, x: 2 }, { y: 1, x: 1 }); // row 5, col 5 of a 5 x 5 matrix
    expect(ops).toEqual([
      { op: 'row', value: 5 },
      { op: 'col', value: 5 },
      { op: 'guard', pass: false },
    ]);
    expect(trace(p5, 5, 0).active).toBe(false);
    expect(trace(p5, 0, 5).active).toBe(false);
  });
  it("the 4 x 4 matrices are the ones chapter 5's model uses", () => {
    const t = makeTiledProblem(4);
    expect(prob.M).toEqual(t.M);
    expect(prob.N).toEqual(t.N);
    expect(prob.ref).toEqual(t.ref);
  });
});

describe('no coordination needed', () => {
  const ORDERS: BlockOrder[] = ['in-order', 'reversed', 'shuffled'];
  it('every block order gives the same P, and it is M N', () => {
    for (const [W, bw] of [[4, 2], [5, 2], [6, 4], [7, 3], [16, 8], [3, 4]] as const)
      for (const order of ORDERS)
        for (const seed of [1, 2, 3, 99]) {
          const r = runAll(makeProblem(W), bw, order, seed);
          expect(r.correct).toBe(true);
          expect(r.P).toEqual(makeProblem(W).ref);
        }
  });
  it('every element of P has exactly one writer', () => {
    for (const [W, bw] of [[4, 2], [5, 2], [7, 4]] as const)
      expect(runAll(makeProblem(W), bw, 'shuffled', 7).writers.every((n) => n === 1)).toBe(true);
  });
  it('a 4 x 4 product with 2 x 2 blocks is four blocks and sixteen threads, none surplus', () => {
    const r = runAll(prob, 2);
    expect(r.order).toHaveLength(4);
    expect(r.threads).toBe(16);
    expect(r.active).toBe(16);
  });
  it('shuffled orders really differ from in-order, and are permutations', () => {
    const inOrder = blockOrder(2, 2, 'in-order');
    const orders = [1, 2, 3, 4, 5].map((s) => blockOrder(2, 2, 'shuffled', s));
    expect(orders.some((o) => JSON.stringify(o) !== JSON.stringify(inOrder))).toBe(true);
    for (const o of orders) expect(o.map((b) => b.y * 2 + b.x).sort()).toEqual([0, 1, 2, 3]);
    expect(blockOrder(2, 2, 'reversed')).toEqual([...inOrder].reverse());
  });
  it('surplus threads are launched and do nothing', () => {
    const r = runAll(makeProblem(5), 2);
    expect(r.threads).toBe(36);
    expect(r.active).toBe(25);
  });
});

describe('swapping the roles of k on line 08', () => {
  it('computes the transpose of N M, not M N', () => {
    for (const W of [3, 4, 5]) {
      const p = makeProblem(W);
      const got = runAll(p, 2, 'in-order', 1, 'swapped').P;
      expect(got).toEqual(transpose(multiply(p.N, p.M, W, W, W), W, W));
      expect(got).not.toEqual(p.ref);
      // The other three options of the question are different matrices too.
      expect(got).not.toEqual(multiply(p.N, p.M, W, W, W));
      expect(got).not.toEqual(multiply(transpose(p.M, W, W), p.N, W, W, W));
    }
  });
  it('reads M down a column and N along a row', () => {
    const s = trace(prob, 1, 2, 'swapped').steps;
    expect(s.map((x) => x.mIndex)).toEqual([1, 5, 9, 13]);
    expect(s.map((x) => x.nIndex)).toEqual([8, 9, 10, 11]);
  });
});

describe('launch shape', () => {
  it('ceil(Width / BLOCK_WIDTH) blocks per side', () => {
    expect(gridFor(4, 4, 2)).toEqual({ x: 2, y: 2, blocks: 4, threads: 16, active: 16, surplus: 0 });
    expect(gridFor(5, 5, 2)).toMatchObject({ x: 3, y: 3, threads: 36, surplus: 11 });
  });
  it('rectangular: x from the columns of the output, y from its rows', () => {
    // C is 3 x 10: l = 10 columns, j = 3 rows.
    expect(gridFor(3, 10, 4)).toMatchObject({ x: 3, y: 1, blocks: 3, threads: 48, active: 30, surplus: 18 });
  });
  it('Block (0,0) of the 4 x 4 example consumes rows 0, 1 of M and columns 0, 1 of N (Fig. 3.13)', () => {
    expect(blockInputs(4, 4, 2, B00)).toEqual({ rows: [0, 1], cols: [0, 1], active: 4, threads: 4 });
    expect(blockInputs(4, 4, 2, { y: 1, x: 0 })).toMatchObject({ rows: [2, 3], cols: [0, 1] });
  });
  it('an edge block consumes fewer rows or columns', () => {
    expect(blockInputs(5, 5, 2, { y: 2, x: 2 })).toEqual({ rows: [4], cols: [4], active: 1, threads: 4 });
    expect(blockInputs(3, 10, 4, { y: 0, x: 2 })).toEqual({ rows: [0, 1, 2], cols: [8, 9], active: 6, threads: 16 });
  });
});

describe('rectangular, and the two storage conventions', () => {
  const r = makeRect(3, 5, 4);
  it('the loop bound is k, the strides are row*k + i and i*l + col, the store is row*l + col', () => {
    const t = traceRect(r, 2, 3, 'book');
    expect(t.steps).toHaveLength(5);
    expect(t.steps.map((s) => s.aIndex)).toEqual([10, 11, 12, 13, 14]);
    expect(t.steps.map((s) => s.bIndex)).toEqual([3, 7, 11, 15, 19]);
    expect(t.store).toEqual({ index: 11, value: r.C[11] });
    expect(t.stride).toBe(4);
  });
  it('the PyTorch-style weight (l x k) is read along a row: col*k + i', () => {
    const t = traceRect(r, 2, 3, 'torch');
    expect(t.steps.map((s) => s.aIndex)).toEqual([10, 11, 12, 13, 14]);
    expect(t.steps.map((s) => s.bIndex)).toEqual([15, 16, 17, 18, 19]);
    expect(t.stride).toBe(1);
  });
  it('both layouts give the same product when one buffer is the transpose of the other', () => {
    for (const [j, k, l] of [[3, 5, 4], [1, 7, 2], [6, 1, 3], [4, 4, 4], [2, 8, 16]] as const) {
      const x = makeRect(j, k, l);
      expect(x.W).toEqual(transpose(x.B, k, l));
      expect(runRect(x, 'book')).toEqual(x.C);
      expect(runRect(x, 'torch')).toEqual(x.C);
      for (let row = 0; row < j; row++)
        for (let col = 0; col < l; col++) {
          const a = traceRect(x, row, col, 'book').steps;
          const b = traceRect(x, row, col, 'torch').steps;
          expect(b.map((s) => s.bValue)).toEqual(a.map((s) => s.bValue));
        }
    }
  });
  it('passing the stored (l x k) weight to the unchanged book kernel is wrong, or out of bounds', () => {
    // Square case, where the mistake is silent: the book kernel with N = W computes M times W, not M times W transposed.
    const p = makeProblem(4);
    const W = transpose(p.N, 4, 4);
    expect(multiply(p.M, W, 4, 4, 4)).not.toEqual(p.ref);
  });
  it('the square problem is the rectangular one with j = k = l', () => {
    const sq = makeRect(4, 4, 4);
    expect(sq.A).toEqual(prob.M);
    expect(sq.B).toEqual(prob.N);
    expect(sq.C).toEqual(prob.ref);
    expect(traceRect(sq, 1, 0, 'book').steps.map((s) => [s.aIndex, s.bIndex])).toEqual(
      trace(prob, 1, 0).steps.map((s) => [s.mIndex, s.nIndex]),
    );
  });
  it('a surplus thread is inactive', () => {
    expect(traceRect(r, 3, 0).active).toBe(false);
    expect(traceRect(r, 0, 4).active).toBe(false);
  });
  it('counts: j*l threads, k multiply-adds and 2k reads each', () => {
    expect(counts(3, 5, 4)).toEqual({ threads: 12, multiplyAddsPerThread: 5, readsPerThread: 10, storesPerThread: 1, totalMultiplyAdds: 60 });
    // One token through a 1024 -> 4096 layer in the book's convention: X is 1 x 1024, W is 1024 x 4096.
    expect(counts(1, 1024, 4096)).toMatchObject({ threads: 4096, multiplyAddsPerThread: 1024 });
  });
});

describe('thread granularity (book exercise 1, p. 64)', () => {
  it('per element: Width squared threads doing Width multiply-adds each, on a 2D grid', () => {
    const g = granularity('element', 2048, 16);
    expect(g).toMatchObject({ gridDims: 2, grid: { x: 128, y: 128 }, block: { x: 16, y: 16 }, working: 2048 * 2048, multiplyAddsPerThread: 2048, storeStride: null });
    expect(g.working).toBe(4194304); // "about 4.2 million"
  });
  it('per row and per column: Width threads doing Width squared multiply-adds each, on a 1D grid', () => {
    for (const mode of ['row', 'column'] as Mode[]) {
      const g = granularity(mode, 2048, 16);
      expect(g).toMatchObject({ gridDims: 1, grid: { x: 8, y: 1 }, block: { x: 256, y: 1 }, threads: 2048, working: 2048, elementsPerThread: 2048, multiplyAddsPerThread: 2048 * 2048 });
    }
  });
  it('the arithmetic in total is the same in all three', () => {
    for (const W of [4, 100, 2048]) {
      const totals = (['element', 'row', 'column'] as Mode[]).map((m) => granularity(m, W, 16));
      for (const g of totals) {
        expect(g.totalMultiplyAdds).toBe(W * W * W);
        expect(g.working * g.multiplyAddsPerThread).toBe(W * W * W);
      }
    }
  });
  it('a row thread stores at stride 1, a column thread at stride Width', () => {
    expect(granularity('row', 6, 16).storeStride).toBe(1);
    expect(granularity('column', 6, 16).storeStride).toBe(6);
    expect(cellsOf('row', 4, 2)).toEqual([8, 9, 10, 11]);
    expect(cellsOf('column', 4, 2)).toEqual([2, 6, 10, 14]);
    expect(cellsOf('element', 4, 9)).toEqual([9]);
  });
  it('in each mode the working threads together write every element exactly once', () => {
    for (const mode of ['element', 'row', 'column'] as Mode[]) {
      const W = 6;
      const g = granularity(mode, W, 2);
      const seen = new Array(W * W).fill(0);
      for (let t = 0; t < g.working; t++) for (const c of cellsOf(mode, W, t)) seen[c]++;
      expect(seen.every((n) => n === 1)).toBe(true);
    }
  });
  it('surplus threads when Width is not a multiple of the block', () => {
    expect(granularity('element', 5, 2)).toMatchObject({ threads: 36, working: 25 });
    expect(granularity('row', 5, 2)).toMatchObject({ threads: 8, working: 5, grid: { x: 2, y: 1 } });
  });
});

describe('size limits (book p. 64; documented values from ch05-facts A5)', () => {
  it('per element with 16 x 16 blocks: 65535 blocks in y allow 1,048,560 rows', () => {
    expect(maxWidth('element', CUDA_DOCUMENTED, 16)).toBe(1048560);
    expect(maxWidth('element', CUDA_DOCUMENTED, 32)).toBe(65535 * 32);
  });
  it('per row or per column: only x is used, so the grid allows far more', () => {
    expect(maxWidth('row', CUDA_DOCUMENTED, 16)).toBe((2 ** 31 - 1) * 256);
    expect(maxWidth('column', CUDA_DOCUMENTED, 16)).toBe((2 ** 31 - 1) * 256);
    expect(maxWidth('row', CUDA_DOCUMENTED, 16)).toBeGreaterThan(maxWidth('element', CUDA_DOCUMENTED, 16));
  });
  it('a block shape over the limit allows nothing', () => {
    expect(maxWidth('element', CUDA_DOCUMENTED, 33)).toBe(0); // 33 * 33 = 1089 threads
    expect(maxWidth('row', CUDA_DOCUMENTED, 33)).toBe(0);
    expect(maxWidth('element', { ...CUDA_DOCUMENTED, blockY: 8 }, 16)).toBe(0);
  });
  it('46340 is the largest Width whose last flat index fits in a 32-bit int', () => {
    expect(maxWidthInt32()).toBe(46340);
    expect(lastIndex(46340)).toBeLessThanOrEqual(INT32_MAX);
    expect(lastIndex(46341)).toBeGreaterThan(INT32_MAX);
    expect(INT32_MAX).toBe(2147483647);
  });
  it('the index overflows long before the documented grid limit is reached', () => {
    expect(maxWidthInt32()).toBeLessThan(maxWidth('element', CUDA_DOCUMENTED, 16));
    expect(Math.floor(maxWidth('element', CUDA_DOCUMENTED, 16) / maxWidthInt32())).toBe(22);
  });
});
