import { describe, expect, it } from 'vitest';
import {
  OPERATIONS, agreement, blasCall, editBlas, editMps, exampleBuffers, mapOperation, mpsCall, randomBuffers, reference, runBlas, runMps,
  type BlasCall, type OpId,
} from './gemmlayout';

const tuple = (c: BlasCall) => [c.transa, c.transb, c.symbolic.m, c.symbolic.n, c.symbolic.k, c.symbolic.a, c.symbolic.lda, c.symbolic.b, c.symbolic.ldb, c.symbolic.c, c.symbolic.ldc];
const SHAPES = [
  { M: 3, N: 4, K: 5 },
  { M: 1, N: 7, K: 3 },
  { M: 6, N: 2, K: 1 },
  { M: 5, N: 5, K: 4 },
  { M: 2, N: 9, K: 11 },
];
const IDS = OPERATIONS.map((o) => o.id);

describe('every operation\'s calls reproduce the row-major product', () => {
  for (const id of IDS)
    for (const shape of SHAPES)
      it(`${id} at ${shape.M} x ${shape.N} x ${shape.K}`, () => {
        const map = mapOperation(id, shape);
        for (const seed of [1, 7, 2026]) {
          const bufs = randomBuffers(map, seed);
          const ref = reference(map, bufs);
          const blas = runBlas(blasCall(map), bufs);
          expect(blas.readOutside || blas.writeOutside).toBe(false);
          expect(agreement(map, blas.C, ref).every(Boolean)).toBe(true);
          const mps = runMps(mpsCall(map), bufs);
          expect(mps.readOutside || mps.writeOutside).toBe(false);
          expect(agreement(map, mps.C, ref).every(Boolean)).toBe(true);
        }
      });
});

describe('the calls the lesson prints', () => {
  it("the book's product: (N, N, n, m, k, B, n, A, k, C, n), as in ch15-facts A10", () => {
    const c = blasCall(mapOperation('book', { M: 3, N: 4, K: 5 }));
    expect(tuple(c)).toEqual(['N', 'N', 'N', 'M', 'K', 'B', 'N', 'A', 'K', 'C', 'N']);
    expect([c.m, c.n, c.k, c.lda, c.ldb, c.ldc]).toEqual([4, 3, 5, 4, 5, 4]);
  });
  it('nn.Linear: (T, N, out, batch, in, W, in, x, in, y, out), the shape of llama.cpp\'s call (ch03-facts D2)', () => {
    const c = blasCall(mapOperation('linear', { M: 3, N: 4, K: 5 }));
    expect(tuple(c)).toEqual(['T', 'N', 'out', 'batch', 'in', 'W', 'in', 'x', 'in', 'y', 'out']);
  });
  it('Q Kᵀ is stored like nn.Linear\'s weight; P V and x @ W like the book', () => {
    expect(blasCall(mapOperation('qkt', { M: 3, N: 4, K: 5 })).transa).toBe('T');
    expect(blasCall(mapOperation('pv', { M: 3, N: 4, K: 5 })).transa).toBe('N');
    expect(blasCall(mapOperation('xw', { M: 3, N: 4, K: 5 })).transa).toBe('N');
    expect(mpsCall(mapOperation('qkt', { M: 3, N: 4, K: 5 })).transposeRight).toBe(true);
    expect(mpsCall(mapOperation('linear', { M: 3, N: 4, K: 5 })).transposeRight).toBe(true);
    expect(mpsCall(mapOperation('book', { M: 3, N: 4, K: 5 })).transposeRight).toBe(false);
  });
  it('MPS for nn.Linear: x is batch x in, W is out x in with transposeRight, rowBytes a multiple of 4', () => {
    const c = mpsCall(mapOperation('linear', { M: 3, N: 4, K: 5 }));
    expect([c.resultRows, c.resultColumns, c.interiorColumns]).toEqual([3, 4, 5]);
    expect(c.left).toEqual({ role: 'A', rows: 3, columns: 5, rowBytes: 20 });
    expect(c.right).toEqual({ role: 'B', rows: 4, columns: 5, rowBytes: 20 });
    expect(c.result).toEqual({ role: 'C', rows: 3, columns: 4, rowBytes: 16 });
    expect(c.transposeLeft).toBe(false);
  });
});

describe('each edit the explorer offers breaks the default 3 x 4 example', () => {
  const broken = (id: OpId, edits: Parameters<typeof editBlas>[1]) => {
    const map = mapOperation(id, { M: 3, N: 4, K: 5 });
    const bufs = exampleBuffers(map);
    const r = runBlas(editBlas(blasCall(map), edits), bufs);
    return r.readOutside || r.writeOutside || !agreement(map, r.C, reference(map, bufs)).every(Boolean);
  };
  for (const id of IDS) {
    it(`${id}: swapping the pointers`, () => expect(broken(id, { swap: true })).toBe(true));
    it(`${id}: flipping transa`, () => expect(broken(id, { flipTransa: true })).toBe(true));
    it(`${id}: m and n in the book's order`, () => expect(broken(id, { mnBook: true })).toBe(true));
    it(`${id}: flipping MPS transposeRight`, () => {
      const map = mapOperation(id, { M: 3, N: 4, K: 5 });
      const bufs = exampleBuffers(map);
      const r = runMps(editMps(mpsCall(map), true), bufs);
      expect(r.readOutside || !agreement(map, r.C, reference(map, bufs)).every(Boolean)).toBe(true);
    });
  }
  it('no edits means no change', () => {
    const map = mapOperation('linear', { M: 3, N: 4, K: 5 });
    expect(editBlas(blasCall(map), {})).toEqual(blasCall(map));
  });
});

describe('the example data', () => {
  it('is small integers, so the products are exact', () => {
    const map = mapOperation('book', { M: 3, N: 4, K: 5 });
    const b = exampleBuffers(map);
    expect(b.A.every((v) => Number.isInteger(v) && Math.abs(v) <= 3)).toBe(true);
    expect(reference(map, b).every(Number.isInteger)).toBe(true);
  });
  it('rejects a shape that is not positive whole numbers', () => {
    expect(() => mapOperation('book', { M: 0, N: 4, K: 5 })).toThrow();
    expect(() => mapOperation('book', { M: 2.5, N: 4, K: 5 })).toThrow();
  });
});
