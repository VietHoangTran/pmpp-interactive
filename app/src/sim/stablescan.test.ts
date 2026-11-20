import { describe, expect, it } from 'vitest';
import {
  BOOK_KEEP,
  BOOK_KEYS,
  blockBinaryScan,
  cutIndex,
  earlierKeepers,
  exclusiveScan,
  isPrefix,
  keepAtLeast,
  keepTopP,
  lanesBelow,
  makeLogits,
  makeSortedProbs,
  maskBits,
  massBefore,
  minPThreshold,
  popcount,
  scatter,
  stableFilter,
  total,
  traffic,
} from './stablescan';

describe('Fig. 12.7: keep, exclusive scan, scatter', () => {
  it('the book\'s offsets and total', () => {
    const offsets = exclusiveScan(BOOK_KEEP);
    expect(offsets).toEqual([0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 7, 8]);
    expect(total(BOOK_KEEP, offsets)).toBe(8);
    expect(scatter(BOOK_KEYS, BOOK_KEEP, offsets)).toEqual(['k1', 'k3', 'k4', 'k6', 'k8', 'k10', 'k11', 'k14']);
  });
  it('Predict ch12/12.5/p-size: flags 1 0 0 1 1 0 1 1 give an output size of 5', () => {
    const keep = [1, 0, 0, 1, 1, 0, 1, 1];
    const offsets = exclusiveScan(keep);
    expect(offsets[7]).toBe(4);
    expect(total(keep, offsets)).toBe(5);
  });
  it('an offset counts the earlier keepers', () => {
    expect(earlierKeepers(BOOK_KEEP, 11)).toEqual([1, 3, 4, 6, 8, 10]);
    for (let i = 0; i < 16; i++) expect(earlierKeepers(BOOK_KEEP, i).length).toBe(exclusiveScan(BOOK_KEEP)[i]);
  });
  it('the result is stable: kept keys keep their input order', () => {
    const keep = [1, 1, 0, 1, 0, 0, 1, 1, 1, 0];
    const items = keep.map((_, i) => i);
    expect(stableFilter(items, keep).output).toEqual(items.filter((_, i) => keep[i]));
  });
  it('empty and all-dropped inputs', () => {
    expect(total([], [])).toBe(0);
    expect(stableFilter([1, 2, 3], [0, 0, 0]).total).toBe(0);
  });
});

describe('the three skins', () => {
  it('logits: keep at or above the threshold (>=)', () => {
    const l = makeLogits();
    expect(l).toHaveLength(16);
    expect(makeLogits()).toEqual(l);
    const t = l[3];
    const keep = keepAtLeast(l, t);
    expect(keep[3]).toBe(1);
    expect(keep.every((k, i) => (k === 1) === (l[i] >= t))).toBe(true);
    // min-p as llama.cpp writes it: max + ln p. p = 1 keeps only the maximum (and ties).
    expect(minPThreshold(l, 1)).toBe(Math.max(...l));
    expect(keepAtLeast(l, minPThreshold(l, 1)).reduce((a, b) => a + b, 0)).toBeGreaterThanOrEqual(1);
  });
  it('sorted probabilities sum to 1 and are in descending order', () => {
    const p = makeSortedProbs();
    expect(p).toHaveLength(16);
    expect(Math.abs(p.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(1e-9);
    for (let i = 1; i < p.length; i++) expect(p[i]).toBeLessThanOrEqual(p[i - 1]);
    expect(massBefore(p)[0]).toBe(0);
  });
  it('top-p over sorted probabilities always keeps a prefix, so one cut index replaces the filter', () => {
    const p = makeSortedProbs();
    for (let k = 1; k <= 100; k++) {
      const keep = keepTopP(p, k / 100);
      expect(isPrefix(keep)).toBe(true);
      expect(cutIndex(keep)).toBe(keep.reduce((a, b) => a + b, 0));
      expect(cutIndex(keep)).toBeGreaterThanOrEqual(1); // the first token always has mass 0 before it
    }
    expect(isPrefix(BOOK_KEEP)).toBe(false);
  });
});

describe('one block from warp votes (blockBinaryScan)', () => {
  it('q-blockrank: warps keeping 3, 0, 5, 2; a lane of warp 2 with rank 4 sits at 7', () => {
    // Four warps of eight lanes with those counts.
    const keep = [
      ...[1, 1, 1, 0, 0, 0, 0, 0],
      ...[0, 0, 0, 0, 0, 0, 0, 0],
      ...[1, 0, 1, 1, 0, 1, 1, 0],
      ...[0, 1, 0, 0, 1, 0, 0, 0],
    ];
    const s = blockBinaryScan(keep, 8);
    expect(s.warpTotals).toEqual([3, 0, 5, 2]);
    expect(s.warpStarts).toEqual([0, 3, 3, 8]);
    const lane = 6; // the fifth kept lane of warp 2: rank 4
    expect(s.ranks[16 + lane]).toBe(4);
    expect(s.offsets[16 + lane]).toBe(7);
    expect(s.total).toBe(10);
  });
  it('equals the plain exclusive scan for random inputs, warp widths 8, 16 and 32', () => {
    let r = 12345;
    const rand = () => ((r = (Math.imul(r, 1103515245) + 12345) >>> 0) >>> 16) / 65536;
    for (const w of [8, 16, 32])
      for (let trial = 0; trial < 50; trial++) {
        const n = w * (1 + Math.floor(rand() * 4));
        const keep: number[] = Array.from({ length: n }, () => (rand() < rand() ? 1 : 0));
        const s = blockBinaryScan(keep, w);
        expect(s.offsets).toEqual(exclusiveScan(keep));
        expect(s.total).toBe(keep.reduce((a, b) => a + b, 0));
      }
  });
  it('lane masks and the book\'s printed mask', () => {
    expect(lanesBelow(0)).toBe(0);
    expect(lanesBelow(5)).toBe(0b11111);
    expect(lanesBelow(31)).toBe(0x7fffffff);
    expect(popcount(0xffffffff)).toBe(32);
    // p. 292: lanes 1, 2, 4 and 5 active.
    expect(maskBits(0b110110, 32)).toBe('00000000000000000000000000110110');
  });
});

describe('bytes per key (our accounting)', () => {
  it('q-ex1-bytes: exercise 1\'s three kernels at half kept move 30 bytes per key', () => {
    const t = traffic('three-kernels', 0.5);
    expect(t.parts.map((p) => p.reads + p.writes)).toEqual([8, 8, 14]);
    expect(t.total).toBe(30);
  });
  it('a single pass moves 6, five times less', () => {
    expect(traffic('single-pass', 0.5).total).toBe(6);
    expect(traffic('three-kernels', 0.5).total / traffic('single-pass', 0.5).total).toBe(5);
  });
  it('the staged design is 8 + 4f + 16 per block', () => {
    expect(traffic('staged', 0.5, 256).total).toBeCloseTo(10 + 16 / 256, 12);
    expect(traffic('staged', 0, 1024).total).toBeCloseTo(8 + 16 / 1024, 12);
  });
  it('every design grows by 4 bytes per key from nothing kept to everything kept', () => {
    for (const d of ['three-kernels', 'staged', 'single-pass'] as const)
      expect(traffic(d, 1).total - traffic(d, 0).total).toBeCloseTo(4, 12);
  });
});
