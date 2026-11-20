import { describe, expect, it } from 'vitest';
import {
  FIG171,
  ROUTER_PICKS,
  argsort,
  cooToCsr,
  countByRow,
  countsAfter,
  cursorScatter,
  exclusiveScan,
  fig171Dense,
  float32Sums,
  gather,
  inversePermutation,
  padSlices,
  route,
  stableScatter,
} from './sparseroute';

describe('count and scan (exercise 3; ch. 11 §11.1)', () => {
  it('the scan of [2, 3, 2, 1] is [0, 2, 5, 7, 8], CSR\'s rowPtrs for Fig. 17.1 (Fig. 17.7)', () => {
    expect(exclusiveScan([2, 3, 2, 1])).toEqual([0, 2, 5, 7, 8]);
    expect(countByRow(FIG171.rowIdx, 4)).toEqual([2, 3, 2, 1]);
  });
});

describe("Fig. 17.1's COO to CSR", () => {
  it('equals the CSR arrays of Fig. 17.7 (literal arrays, not group 1\'s model)', () => {
    const c = cooToCsr(FIG171.rowIdx, FIG171.colIdx, FIG171.value, 4);
    expect(c.rowPtrs).toEqual([0, 2, 5, 7, 8]);
    expect(c.colIdx).toEqual([0, 1, 0, 2, 3, 1, 2, 3]);
    expect(c.value).toEqual([1, 7, 5, 3, 9, 2, 8, 6]);
  });
  it("from Fig. 17.6's permuted COO: the same rows, each in its input order", () => {
    const rowIdx = [1, 3, 2, 2, 0, 1, 1, 0];
    const colIdx = [3, 3, 1, 2, 0, 0, 2, 1];
    const value = [9, 6, 2, 8, 1, 5, 3, 7];
    const c = cooToCsr(rowIdx, colIdx, value, 4);
    expect(c.rowPtrs).toEqual([0, 2, 5, 7, 8]);
    expect(c.value).toEqual([1, 7, 9, 5, 3, 2, 8, 6]);
    expect(c.colIdx).toEqual([0, 1, 3, 0, 2, 1, 2, 3]);
  });
  it('the dense picture matches the COO arrays', () => {
    expect(fig171Dense()).toEqual([
      [1, 7, 0, 0],
      [5, 0, 3, 9],
      [0, 2, 8, 0],
      [0, 0, 0, 6],
    ]);
  });
});

describe('q-moe-offset: six slots to experts [2, 0, 2, 1, 0, 2]', () => {
  const ids = [2, 0, 2, 1, 0, 2];
  const counts = countByRow(ids, 3);
  const offsets = exclusiveScan(counts);
  const s = stableScatter(ids, offsets);
  it('counts [2, 1, 3], offsets [0, 2, 3]', () => {
    expect(counts).toEqual([2, 1, 3]);
    expect(offsets.slice(0, 3)).toEqual([0, 2, 3]);
  });
  it('slot 4 lands at position 1: expert 0 holds slots 1 and 4', () => {
    expect(s.pos[4]).toBe(1);
    expect(s.order.slice(0, 2)).toEqual([1, 4]);
  });
});

describe('stable and atomic-cursor scatters', () => {
  const ids = [2, 0, 2, 1, 0, 2, 1, 0];
  const offsets = exclusiveScan(countByRow(ids, 3));
  const arrivals = [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [7, 6, 5, 4, 3, 2, 1, 0],
    [3, 0, 7, 5, 1, 6, 2, 4],
  ];
  it('the stable scatter does not depend on the order threads arrive in', () => {
    // It has no arrival order at all: each rank is computed from the input alone.
    const st = stableScatter(ids, offsets);
    expect(st.rank).toEqual([0, 0, 1, 0, 1, 2, 1, 2]);
    // and it equals the cursor scatter under input-order arrival
    expect(cursorScatter(ids, offsets, arrivals[0]).pos).toEqual(st.pos);
  });
  it('the cursor scatter does: other arrival orders reorder entries inside a row', () => {
    const results = arrivals.map((a) => JSON.stringify(cursorScatter(ids, offsets, a).pos));
    expect(new Set(results).size).toBe(3);
    // but every arrival order puts each entry inside its own row's slice
    for (const a of arrivals) {
      const c = cursorScatter(ids, offsets, a);
      c.pos.forEach((p, i) => {
        expect(p).toBeGreaterThanOrEqual(offsets[ids[i]]);
        expect(p).toBeLessThan(offsets[ids[i] + 1]);
      });
    }
  });
  it('the stable order is a stable argsort, and its inverse is the argsort of the argsort (mlx-lm)', () => {
    const st = stableScatter(ids, offsets);
    expect(st.order).toEqual(argsort(ids));
    expect(st.pos).toEqual(argsort(st.order));
    expect(inversePermutation(st.order)).toEqual(st.pos);
  });
});

describe('the routing example (scene "Routing tokens to experts")', () => {
  const r = route();
  it('8 tokens, top-2, 4 experts: 16 pairs, expert 2 has none', () => {
    expect(r.pairs).toHaveLength(16);
    expect(r.counts).toEqual([6, 5, 0, 5]);
    expect(r.offsets).toEqual([0, 6, 11, 11, 16]);
    expect(r.empty).toEqual([2]);
    expect(ROUTER_PICKS.every(([a, b]) => a !== b)).toBe(true);
  });
  it('the counts build up one pair at a time', () => {
    expect(countsAfter(r, 0)).toEqual([0, 0, 0, 0]);
    expect(countsAfter(r, 2)).toEqual([1, 0, 0, 1]);
    expect(countsAfter(r, 16)).toEqual(r.counts);
  });
  it('after the scatter the pairs are contiguous per expert, in token order', () => {
    const experts = gather(r.pairs, r.scatter.order).map((p) => p.expert);
    expect(experts).toEqual([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3]);
    const tokensOfE0 = gather(r.pairs, r.scatter.order).slice(0, 6).map((p) => p.token);
    expect(tokensOfE0).toEqual([0, 1, 3, 4, 6, 7]);
  });
  it('applying the inverse permutation restores token order', () => {
    const sorted = gather(r.pairs, r.scatter.order);
    const back = gather(sorted, r.scatter.pos);
    expect(back).toEqual(r.pairs);
  });
  it('padded slices start at multiples of the tile and hold every entry', () => {
    expect(r.slices.padded).toEqual([8, 8, 0, 8]);
    expect(r.slices.starts).toEqual([0, 8, 16, 16, 24]);
    expect(r.slices.pads).toBe(8);
    for (const s of r.slices.starts) expect(s % r.tile).toBe(0);
    r.pairs.forEach((p, i) => {
      expect(r.paddedPos[i]).toBeGreaterThanOrEqual(r.slices.starts[p.expert]);
      expect(r.paddedPos[i]).toBeLessThan(r.slices.starts[p.expert] + r.counts[p.expert]);
    });
    expect(new Set(r.paddedPos).size).toBe(16);
  });
  it('padSlices leaves an empty slice empty and pads others up', () => {
    expect(padSlices([1, 0, 5], 4)).toEqual({ starts: [0, 4, 4, 12], padded: [4, 0, 8], total: 12, pads: 6 });
  });
});

describe('summation order (appendix A §A.6)', () => {
  it('1e8, 1 and -1e8 added in float32 in any order give 0 or 1', () => {
    expect(float32Sums([1e8, 1, -1e8])).toEqual([0, 1]);
    // the reason: the float32 spacing at 1e8 is 8, so 1e8 + 1 rounds back to 1e8
    expect(Math.fround(1e8 + 1)).toBe(1e8);
    expect(Math.fround(1e8 + 8) - 1e8).toBe(8);
  });
  it('the lost-update question: atomics give only 8, plain read-modify-write can give 1, 7 or 8', () => {
    expect(float32Sums([1, 7])).toEqual([8]);
  });
});
