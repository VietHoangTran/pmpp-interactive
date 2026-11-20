import { describe, expect, it } from 'vitest';
import { mergeStable } from './corank';
import { DATA, basicLayout, layoutCorrect, makeData, perThreadCount, searchCost, threadsFor, warpStep } from './mergeaccess';

const A = [1, 7, 8, 9, 10];
const B = [7, 10, 10, 12];

describe('line 03 of Fig. 13.9: outputs per thread', () => {
  it('as printed, 1,000 outputs on 3 blocks of 128 threads is 2 per thread and leaves 232 unwritten (the question)', () => {
    expect(perThreadCount(1000, 384, 'printed')).toBe(2);
    const d = makeData('distinct', 1000);
    const L = basicLayout(d.A, d.B, 384, 'printed');
    expect(L.unwritten.length).toBe(232);
    expect(L.unwritten[0]).toBe(768);
  });
  it('corrected, the same launch writes everything: 3 per thread, and the stable merge', () => {
    expect(perThreadCount(1000, 384, 'corrected')).toBe(3);
    const d = makeData('duplicates', 1000);
    const L = basicLayout(d.A, d.B, 384, 'corrected');
    expect(L.unwritten).toEqual([]);
    expect(layoutCorrect(L)).toBe(true);
  });
  it('m + n = 9 on 4 threads, as printed: only C[8] is left unwritten', () => {
    const L = basicLayout(A, B, 4, 'printed');
    expect(L.perThread).toBe(2);
    expect(L.unwritten).toEqual([8]);
  });
  it('as printed, more threads than outputs gives 0 per thread and writes nothing', () => {
    expect(basicLayout(A, B, 16, 'printed').unwritten.length).toBe(9);
  });
  it('corrected, every data preset and many launches write the stable merge', () => {
    for (const { id } of DATA)
      for (const total of [64, 100, 257])
        for (const per of [1, 3, 8, 16]) {
          const d = makeData(id, total, 5);
          const L = basicLayout(d.A, d.B, threadsFor(total, per), 'corrected');
          expect(layoutCorrect(L)).toBe(true);
          expect(L.C.map((e) => e!.key)).toEqual(mergeStable(d.A, d.B).map((e) => e.key));
        }
  });
});

describe('the running example on three threads (p. 314)', () => {
  const L = basicLayout(A, B, 3, 'corrected');
  it('thread 1 merges (&A[2], 3, &B[1], 0, &C[3])', () => {
    const w = L.work[1];
    expect([w.iCurr, w.iNext - w.iCurr, w.jCurr, w.jNext - w.jCurr, w.kCurr]).toEqual([2, 3, 1, 0, 3]);
  });
  it('the first step: thread 0 compares A[0] with B[0], thread 1 reads A[2], thread 2 reads B[1]; stores to C[0], C[3], C[6]', () => {
    const s = warpStep(L, 0, 32, 0);
    expect(s.lanes.slice(0, 3).map((x) => x.path)).toEqual(['A', 'tailA', 'tailB']);
    expect(s.lanes[0].reads).toEqual([{ arr: 'A', index: 0 }, { arr: 'B', index: 0 }]);
    expect(s.lanes[1].reads).toEqual([{ arr: 'A', index: 2 }]);
    expect(s.lanes[2].reads).toEqual([{ arr: 'B', index: 1 }]);
    expect(s.lanes.slice(0, 3).map((x) => x.write)).toEqual([0, 3, 6]);
    expect(s.lanes[3].path).toBe('idle');
    expect(s.paths).toEqual(['A', 'tailA', 'tailB']);
    expect(s.active).toBe(3);
  });
});

describe('coalescing of one warp (32 lanes, 4-byte ints, 32-byte segments)', () => {
  const d = makeData('distinct', 1024, 3);
  it('one output per thread: the step-0 store touches 4 segments, every byte used', () => {
    const L = basicLayout(d.A, d.B, threadsFor(1024, 1), 'corrected');
    const s = warpStep(L, 0, 32, 0, 4, 32);
    expect(s.writes.transactions).toBe(4);
    expect(s.writes.efficiency).toBe(1);
  });
  it('eight outputs per thread: each lane stores into its own segment, 32 of them, 12.5% used (the question)', () => {
    const L = basicLayout(d.A, d.B, threadsFor(1024, 8), 'corrected');
    for (const step of [0, 3, 7]) {
      const s = warpStep(L, 0, 32, step, 4, 32);
      expect(s.writes.transactions).toBe(32);
      expect(s.writes.efficiency).toBe(0.125);
    }
  });
  it('with 64 lanes and 128-byte lines, one output per thread is 2 lines', () => {
    const L = basicLayout(d.A, d.B, threadsFor(1024, 1), 'corrected');
    expect(warpStep(L, 0, 64, 0, 4, 128).writes.transactions).toBe(2);
  });
  it('random data: the lanes of one warp take more than one path at a step', () => {
    const L = basicLayout(d.A, d.B, threadsFor(1024, 8), 'corrected');
    expect(warpStep(L, 0, 32, 0).paths.length).toBeGreaterThan(1);
  });
  it('all of A before all of B: no lane runs the main loop; each thread copies one input', () => {
    const e = makeData('a-before-b', 1024);
    const L = basicLayout(e.A, e.B, threadsFor(1024, 8), 'corrected');
    for (const w of L.work) expect(w.ops.every((o) => o.path === 'tailA') || w.ops.every((o) => o.path === 'tailB')).toBe(true);
  });
});

describe('the cost of the searches', () => {
  it('search reads per output fall as the outputs per thread grow, roughly as 1 / outputs per thread', () => {
    const d = makeData('distinct', 1024, 9);
    const per = [1, 2, 4, 8, 16].map((p) => searchCost(basicLayout(d.A, d.B, threadsFor(1024, p), 'corrected')).searchPerOutput);
    for (let q = 1; q < per.length; q++) expect(per[q]).toBeLessThan(per[q - 1]);
    expect(per[0] / per[4]).toBeGreaterThan(8);
  });
  it('two calls per thread, and merge loads between 1 and 2 per output', () => {
    const d = makeData('distinct', 512, 2);
    const c = searchCost(basicLayout(d.A, d.B, 64, 'corrected'));
    expect(c.calls).toBe(128);
    expect(c.mergePerOutput).toBeGreaterThanOrEqual(1);
    expect(c.mergePerOutput).toBeLessThanOrEqual(2);
  });
  it('the end boundary costs no reads: co_rank(m + n) stops at once', () => {
    const L = basicLayout(A, B, 1, 'corrected'); // co_rank(0) and co_rank(9)
    expect(L.work[0].searchReads).toBe(0);
  });
});

describe('data presets', () => {
  it('give sorted inputs with m = floor(total / 2), reproducibly', () => {
    for (const { id } of DATA) {
      const x = makeData(id, 101, 4);
      expect(x).toEqual(makeData(id, 101, 4));
      expect([x.A.length, x.B.length]).toEqual([50, 51]);
      for (const arr of [x.A, x.B]) expect(arr.every((v, q) => q === 0 || arr[q - 1] <= v)).toBe(true);
    }
    const dist = makeData('distinct', 200, 1);
    expect(new Set([...dist.A, ...dist.B]).size).toBe(200);
  });
});
