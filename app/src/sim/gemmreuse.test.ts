import { describe, expect, it } from 'vitest';
import { BOOK_THRESHOLD, gridFetches, intensity, regime, regionBlocks, regionCounts, smallestHeight, tilePair } from './gemmreuse';

describe("the book's worked numbers (§15.2, p. 351)", () => {
  it('32 x 32 output tiles give 8 FLOP/B: memory-bound on the book\'s example device', () => {
    expect(intensity(32, 32)).toBe(8);
    expect(regime(intensity(32, 32), BOOK_THRESHOLD)).toBe('memory-bound');
  });
  it('128 x 128 output tiles give 32 FLOP/B: compute-bound there', () => {
    expect(intensity(128, 128)).toBe(32);
    expect(regime(intensity(128, 128), BOOK_THRESHOLD)).toBe('compute-bound');
  });
  it('operations and bytes per tile pair are 2mnk and 4k(m + n)', () => {
    const t = tilePair(128, 128, 8);
    expect(t.ops).toBe(2 * 128 * 128 * 8);
    expect(t.traffic).toBe(4 * 8 * 256);
    expect(t.intensity).toBe(32);
  });
  it('equals the book\'s 0.5·m·n/(m + n) for 4-byte elements, for any shape', () => {
    for (const [m, n] of [[1, 1], [7, 300], [64, 128], [256, 3]]) expect(intensity(m, n)).toBeCloseTo((0.5 * m * n) / (m + n), 12);
  });
});

describe('k cancels', () => {
  it('the intensity of a 128 x 128 tile is 32 at every depth from 1 to 64', () => {
    for (let k = 1; k <= 64; k++) expect(tilePair(128, 128, k).intensity).toBeCloseTo(32, 12);
  });
  it('doubling k doubles both operations and bytes', () => {
    const a = tilePair(128, 128, 8);
    const b = tilePair(128, 128, 16);
    expect(b.ops).toBe(2 * a.ops);
    expect(b.traffic).toBe(2 * a.traffic);
  });
});

describe('other shapes the lesson uses', () => {
  it('128 x 64 gives 21.33 FLOP/B, just above the book\'s example threshold', () => {
    expect(intensity(128, 64)).toBeCloseTo(21.333, 3);
    expect(regime(intensity(128, 64), BOOK_THRESHOLD)).toBe('compute-bound');
  });
  it('m = 1 stays below 0.5 FLOP/B however wide the tile (our derivation)', () => {
    for (const n of [1, 8, 128, 4096, 1e6]) expect(intensity(1, n)).toBeLessThan(0.5);
    expect(intensity(1, 128)).toBeCloseTo(0.496, 3);
  });
  it('1-byte inputs give 128 FLOP/B for a 128 x 128 tile; 2 bytes give 64; 4 bits give 256', () => {
    expect(intensity(128, 128, 1)).toBe(128);
    expect(intensity(128, 128, 2)).toBe(64);
    expect(intensity(128, 128, 0.5)).toBe(256);
  });
  it('rejects shapes that are not positive', () => {
    expect(() => tilePair(0, 4, 4)).toThrow();
    expect(() => tilePair(4, 4, 4, -1)).toThrow();
  });
});

describe('batch as the tile height', () => {
  it('with 128-column tiles, batch 59 is the first to reach 20 FLOP/B', () => {
    expect(smallestHeight(128, 20)).toBe(59);
    expect(intensity(58, 128)).toBeLessThan(20);
    expect(intensity(59, 128)).toBeGreaterThanOrEqual(20);
  });
  it('a height that cannot reach the threshold returns null', () => {
    // 2n/e is the limit as m grows: 64 for n = 128 at 4 bytes.
    expect(smallestHeight(128, 64)).toBeNull();
    expect(smallestHeight(128, 20, 4, 50)).toBeNull();
  });
  it('a 4096 -> 4096 layer at batch 8 with 128 x 128 block tiles gives 3.76 FLOP/B', () => {
    const g = gridFetches(8, 4096, 4096, 128, 128);
    expect(g.intensity).toBeCloseTo(3.765, 3);
    expect(intensity(8, 128)).toBeCloseTo(g.intensity, 9); // the batch is the tile height
  });
});

describe('fetches across the grid', () => {
  it('M = N = K = 4096 with 128 x 128 tiles: each element of A is fetched 32 times, with 32 x 32 tiles 128 times', () => {
    expect(gridFetches(4096, 4096, 4096, 128, 128).aFetches).toBe(32);
    expect(gridFetches(4096, 4096, 4096, 32, 32).aFetches).toBe(128);
    expect(gridFetches(4096, 4096, 4096, 128, 128).bFetches).toBe(32);
  });
  it('for square tiles that divide the matrix the grid intensity equals the tile intensity', () => {
    expect(gridFetches(4096, 4096, 4096, 128, 128).intensity).toBeCloseTo(32, 9);
    expect(gridFetches(4096, 4096, 4096, 32, 32).intensity).toBeCloseTo(8, 9);
  });
  it('requested bytes are the ideal bytes times the fetch counts', () => {
    const g = gridFetches(1000, 300, 64, 128, 64);
    expect(g.blocksM).toBe(8);
    expect(g.blocksN).toBe(5);
    expect(g.requested).toBe(4 * (1000 * 64 * 5 + 64 * 300 * 8));
    expect(g.ideal).toBe(4 * (1000 * 64 + 64 * 300));
  });
});

describe('the reuse scene', () => {
  it('a 128 x 128 region in 32-wide tiles is sixteen blocks; each band is fetched four times', () => {
    expect(regionBlocks(128, 32)).toHaveLength(16);
    const c = regionCounts(128, 32, 8);
    expect(c.aBand).toEqual([4, 4, 4, 4]);
    expect(c.bBand).toEqual([4, 4, 4, 4]);
    expect(c.requested).toBe(4 * c.needed);
    expect(c.requested).toBe(32768);
    expect(c.needed).toBe(8192);
  });
  it('the first row of four blocks fetches A band 0 four times and each B band once', () => {
    const c = regionCounts(128, 32, 8, 4);
    expect(c.aBand).toEqual([4, 0, 0, 0]);
    expect(c.bBand).toEqual([1, 1, 1, 1]);
  });
  it('one 128 x 128 block fetches each band once: four times fewer bytes', () => {
    const one = regionCounts(128, 128, 8);
    expect(one.aBand).toEqual([1]);
    expect(one.requested).toBe(one.needed);
    expect(regionCounts(128, 32, 8).requested / one.requested).toBe(4);
  });
});
