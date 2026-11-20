import { describe, expect, it } from 'vitest';
import {
  TIES16,
  TIES8,
  applyNetwork,
  bitonicCount,
  bitonicStages,
  byValue,
  comparatorCount,
  isSorted,
  networkSort,
  tieGroups,
  transpositionCount,
  transpositionStages,
  type Item,
} from './sortnet';

const asc = (a: number, b: number) => a < b;

describe('the bitonic network (section 14.9, pp. 345-346)', () => {
  it('has 24 comparators for N = 8 (6 stages of 4) and 80 for N = 16 (10 stages of 8)', () => {
    expect(bitonicStages(8).length).toBe(6);
    expect(bitonicStages(8).every((s) => s.length === 4)).toBe(true);
    expect(comparatorCount(bitonicStages(8))).toBe(24);
    expect(bitonicCount(8)).toBe(24);
    expect(bitonicStages(16).length).toBe(10);
    expect(comparatorCount(bitonicStages(16))).toBe(80);
    expect(bitonicCount(16)).toBe(80);
  });
  it('touches each position once per stage (comparators in a stage are disjoint)', () => {
    for (const n of [2, 4, 8, 16, 32])
      for (const st of bitonicStages(n)) {
        const seen = st.flatMap((c) => [c.lo, c.hi]);
        expect(new Set(seen).size).toBe(n);
      }
  });
  it('sorts every 0-1 input of length 8 (so, by the 0-1 principle, every input)', () => {
    const st = bitonicStages(8);
    for (let m = 0; m < 256; m++) {
      const v = Array.from({ length: 8 }, (_, k) => (m >> k) & 1);
      expect(isSorted(applyNetwork(v, st, asc).out, asc)).toBe(true);
    }
  });
  it('sorts random inputs of 16 and 32, either direction', () => {
    let s = 7;
    const rnd = () => ((s = (s * 1103515245 + 12345) >>> 0) >>> 16) % 50;
    for (let k = 0; k < 200; k++) {
      const n = k % 2 ? 16 : 32;
      const v = Array.from({ length: n }, rnd);
      expect(applyNetwork(v, bitonicStages(n), asc).out).toEqual(v.slice().sort((a, b) => a - b));
      expect(applyNetwork(v, bitonicStages(n), (a, b) => a > b).out).toEqual(v.slice().sort((a, b) => b - a));
    }
  });
  it('uses the same comparators whatever the data; only which ones swap changes', () => {
    const a = applyNetwork([3, 1, 2, 0, 7, 5, 6, 4], bitonicStages(8), asc);
    const b = applyNetwork([0, 1, 2, 3, 4, 5, 6, 7], bitonicStages(8), asc);
    expect(a.comparisons).toBe(24);
    expect(b.comparisons).toBe(24);
    expect(b.swaps).toBeGreaterThan(0); // even sorted input is moved: half of the first stages sort downwards
    expect(a.stages.map((s) => s.swapped.length)).toEqual(b.stages.map((s) => s.swapped.length));
  });
  it('rejects a length that is not a power of two; networkSort pads instead', () => {
    expect(() => bitonicStages(12)).toThrow();
    expect(networkSort([5, 3, 9, 1, 7, 2, 8, 6, 4, 0, 11, 10], asc)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
  });
});

describe('odd-even transposition as a network (sections 14.2 and 14.9)', () => {
  it('has 120 comparators for N = 16 (8 steps of 8, 8 of 7) and 28 for N = 8', () => {
    expect(transpositionStages(16).map((s) => s.length)).toEqual([8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7]);
    expect(transpositionCount(16)).toBe(120);
    expect(transpositionCount(8)).toBe(28);
  });
  it('sorts every 0-1 input of length 8 in 8 steps', () => {
    for (let m = 0; m < 256; m++) {
      const v = Array.from({ length: 8 }, (_, k) => (m >> k) & 1);
      expect(isSorted(applyNetwork(v, transpositionStages(8), asc).out, asc)).toBe(true);
    }
  });
});

describe('ties: the network is not stable; comparing (value, index) makes it deterministic', () => {
  const run = (v: Item[], byIndex: boolean) => applyNetwork(v, bitonicStages(v.length), byValue(byIndex)).out;
  it('the 16-value ties preset comes out sorted, with ties reordered in three groups', () => {
    const out = run(TIES16, false);
    expect(isSorted(out, byValue(false))).toBe(true);
    const g = tieGroups(out);
    expect(g.map((x) => x.key)).toEqual([0.12, 0.08, 0.05, 0.03, 0.02]);
    expect(g.filter((x) => !x.increasing).map((x) => x.key)).toEqual([0.08, 0.05, 0.02]);
    expect(g.find((x) => x.key === 0.08)!.ids).toEqual([6, 10, 3]);
  });
  it('the first eight reorder two groups', () => {
    const g = tieGroups(run(TIES8, false));
    expect(g.filter((x) => !x.increasing).map((x) => [x.key, x.ids])).toEqual([[0.08, [6, 3]], [0.05, [2, 7, 0]]]);
  });
  it('with (value, index) every tie group is in increasing token order, which is the stable order', () => {
    for (const v of [TIES8, TIES16]) {
      const out = run(v, true);
      expect(tieGroups(out).every((x) => x.increasing)).toBe(true);
      expect(out).toEqual(v.slice().sort((a, b) => b.key - a.key || a.id - b.id));
    }
  });
  it('the input order of the values does not change the indexed result', () => {
    const shuffled = [...TIES16].reverse();
    expect(run(shuffled, true)).toEqual(run(TIES16, true));
  });
});
