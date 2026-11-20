import { describe, expect, it } from 'vitest';
import { blocksNeeded, counts, coverage } from '../../sim/grid1d';
import { ceilDiv, launch } from '../../sim/launch';

// Sections 2.5 and 2.6 were built in parallel, so each has its own model of a
// one-dimensional launch: grid1d.ts (the index explorer of 2.5) and launch.ts (the
// block-count calculator of 2.6). Both compute i = b*B + t and what it covers. This
// test makes them agree, so the two pages can never show different numbers for the
// same launch.
const SAMPLE: [number, number][] = [
  [1, 32], [31, 32], [32, 32], [33, 32], [100, 32], [255, 256], [256, 256], [257, 256],
  [750, 256], [1000, 256], [4000, 256], [4097, 256], [8000, 1024], [50257, 256], [200000, 128],
  [7, 1], [1, 1024], [1023, 1024], [1025, 1024],
];

describe('grid1d and launch agree on a one-dimensional launch', () => {
  for (const [n, B] of SAMPLE) {
    it(`n = ${n}, B = ${B}`, () => {
      const blocks = ceilDiv(n, B);
      expect(blocksNeeded(n, B)).toBe(blocks);
      const g = counts({ n, blockDim: B });
      const l = launch(n, B, blocks);
      expect(g.blocks).toBe(l.blocks);
      expect(g.launched).toBe(l.threads);
      expect(g.working).toBe(l.working);
      expect(g.idle).toBe(l.idle);
      expect(g.idle).toBe(l.surplus);
    });
  }

  it('with too few blocks both leave the same elements unprocessed', () => {
    for (const [n, B, blocks] of [[1000, 256, 3], [100, 32, 3], [257, 256, 1], [33, 32, 1]] as const) {
      const g = coverage({ n, blockDim: B, blocks });
      const l = launch(n, B, blocks);
      expect(g.unwritten.length).toBe(l.unprocessedCount);
      if (l.unprocessed.length) expect([g.unwritten[0], g.unwritten.at(-1)]).toEqual([l.unprocessed[0].first, l.unprocessed[0].last]);
      expect(g.working).toBe(l.working);
    }
  });

  it('without the guard both touch the same indices past the end', () => {
    for (const [n, B] of [[100, 32], [1000, 256], [33, 32]] as const) {
      const g = coverage({ n, blockDim: B, guard: false });
      const l = launch(n, B, ceilDiv(n, B), false);
      expect(g.outOfRange.length).toBe(l.outOfRangeCount);
      expect([g.outOfRange[0], g.outOfRange.at(-1)]).toEqual([l.outOfRange[0].first, l.outOfRange[0].last]);
    }
  });

  it("the book's worked cases: 100 in blocks of 32 leaves 28 idle, 1000 in blocks of 256 leaves 24", () => {
    expect(counts({ n: 100, blockDim: 32 }).idle).toBe(28);
    expect(launch(100, 32, ceilDiv(100, 32)).idle).toBe(28);
    expect(counts({ n: 1000, blockDim: 256 }).idle).toBe(24);
    expect(launch(1000, 256, ceilDiv(1000, 256)).idle).toBe(24);
  });
});
