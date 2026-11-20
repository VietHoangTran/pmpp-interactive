import { describe, expect, it } from 'vitest';
import { coordOf, dims } from './warps';
import { blockView, coordAt, dims3, gridCoords, rowStrip, warpView, warpsPerBlock, type WarpConfig } from './stencilwarp';

// An interior block of a 128³ grid, the book's −1 offset, NVIDIA's 32-wide warp.
const base = (over: Partial<WarpConfig> = {}): WarpConfig => ({
  block: dims3(8, 8, 8),
  r: 1,
  blockIdx: dims3(1, 1, 1),
  warp: 0,
  width: 32,
  N: 128,
  offset: -1,
  ...over,
});

describe('the linear order is chapter 4\'s', () => {
  it('agrees with src/sim/warps.ts for the block shapes used here', () => {
    for (const [x, y, z] of [[8, 8, 8], [10, 10, 10], [32, 8, 4], [32, 32, 1]]) {
      for (let n = 0; n < x * y * z; n++) expect(coordAt(n, dims3(x, y, z))).toEqual(coordOf(n, dims(x, y, z)));
    }
  });
  it('the first warp of an 8 × 8 × 8 block is four rows of eight in plane z = 0', () => {
    const w = warpView(base());
    expect(new Set(w.lanes.map((l) => l.t.z))).toEqual(new Set([0]));
    expect(new Set(w.lanes.map((l) => l.t.y))).toEqual(new Set([0, 1, 2, 3]));
    expect(w.lanes.length).toBe(32);
  });
});

describe("Fig. 8.8's index arithmetic", () => {
  it('k = blockIdx.x·OUT_TILE_DIM + threadIdx.x − 1, and likewise j and i', () => {
    expect(gridCoords(base({ blockIdx: dims3(2, 1, 3) }), dims3(0, 5, 7))).toEqual({ k: 11, j: 10, i: 24 });
  });
  it('the 8 × 8 × 8 rows start at 6·bx − 1', () => {
    for (const bx of [1, 2, 3]) expect(warpView(base({ blockIdx: dims3(bx, 1, 1) })).rows[0].kFirst).toBe(6 * bx - 1);
  });
});

describe('coalescing under the 32-byte model (facts A2)', () => {
  it('interior 8 × 8 × 8 block: 4 rows, 8 transactions, 50%', () => {
    for (const bx of [1, 2, 3]) {
      const w = warpView(base({ blockIdx: dims3(bx, 1, 1) }));
      expect(w.rows.length).toBe(4);
      expect(w.transactions).toBe(8);
      expect(w.bytesUsed).toBe(128);
      expect(w.bytesFetched).toBe(256);
      expect(w.efficiency).toBe(0.5);
    }
  });
  it('every warp of the interior cube does the same: 128 transactions for the block', () => {
    const b = blockView(base());
    expect(b.warps.length).toBe(16);
    expect(b.transactions).toBe(128);
    expect(b.efficiency).toBe(0.5);
  });
  it('32 × 8 × 4: one row per warp, 5 transactions, 80%', () => {
    for (const bx of [1, 2, 3]) {
      const w = warpView(base({ block: dims3(32, 8, 4), blockIdx: dims3(bx, 1, 1) }));
      expect(w.rows.length).toBe(1);
      expect(w.rows[0].kFirst).toBe(30 * bx - 1);
      expect(w.transactions).toBe(5);
      expect(w.efficiency).toBe(0.8);
    }
  });
  it('offset 0 with a start that is a multiple of 8 floats: 4 transactions, 100%', () => {
    const w = warpView(base({ block: dims3(32, 8, 4), blockIdx: dims3(0, 1, 1), offset: 0 }));
    expect(w.transactions).toBe(4);
    expect(w.efficiency).toBe(1);
  });
  it('offset 0 does not align a start of 30·bx in general', () => {
    expect(warpView(base({ block: dims3(32, 8, 4), blockIdx: dims3(1, 1, 1), offset: 0 })).transactions).toBe(5);
  });
  it('offset 0 does not rescue the cube: its rows are still 8 floats at 6·bx', () => {
    expect(warpView(base({ blockIdx: dims3(1, 1, 1), offset: 0 })).efficiency).toBe(0.5);
  });
  it('10 × 10 × 10: rows of ten straddle three segments; 11 transactions for warp 0', () => {
    const w = warpView(base({ block: dims3(10, 10, 10) }));
    expect(w.rows.length).toBe(4);
    expect(w.transactions).toBe(11);
    expect(w.efficiency).toBeCloseTo(128 / 352, 12);
  });
});

describe('block 0 at the low edge of the grid', () => {
  it('the lane at k = −1 loads nothing', () => {
    const w = warpView(base({ blockIdx: dims3(0, 1, 1) }));
    const edge = w.lanes.filter((l) => l.k === -1);
    expect(edge.length).toBe(4);
    expect(edge.every((l) => l.index === null)).toBe(true);
    expect(w.loading).toBe(28);
    expect(w.transactions).toBe(4);
  });
});

describe("AMD's 64-wide wavefronts (facts B2)", () => {
  it('8 × 8 × 8: one wavefront covers 8 rows of 8, one whole z-plane', () => {
    const b = blockView(base({ width: 64 }));
    expect(b.warps.length).toBe(8);
    const w = b.warps[0];
    expect(w.rows.length).toBe(8);
    expect(new Set(w.lanes.map((l) => l.t.z))).toEqual(new Set([0]));
  });
  it('32 × 8 × 4: two rows of 32 per wavefront', () => {
    expect(warpView(base({ width: 64, block: dims3(32, 8, 4) })).rows.length).toBe(2);
  });
});

describe('the compute guards and divergence (ch. 4 §4.5)', () => {
  it('8 × 8 × 8: 12 of 16 warps divergent, the 4 in planes 0 and 7 wholly idle', () => {
    const b = blockView(base());
    expect(b.divergent).toBe(12);
    expect(b.idle).toBe(4);
    expect(b.full).toBe(0);
    const idle = b.warps.filter((w) => w.state === 'idle').map((w) => w.lanes[0].t.z);
    expect(new Set(idle)).toEqual(new Set([0, 7]));
  });
  it('a divergent cube warp has idle lanes at x = 0, x = 7 and in its outer y row', () => {
    const w = warpView(base({ warp: 2 })); // plane z = 1, rows y = 0 to 3
    expect(w.state).toBe('divergent');
    expect(w.active).toBe(18); // x 1..6, y 1..3
    expect(w.lanes.filter((l) => !l.active).every((l) => l.t.x === 0 || l.t.x === 7 || l.t.y === 0)).toBe(true);
  });
  it('32 × 32 plane: 30 of 32 warps divergent', () => {
    const b = blockView(base({ block: dims3(32, 32, 1) }));
    expect(b.warps.length).toBe(32);
    expect(b.divergent).toBe(30);
    expect(b.idle).toBe(2);
  });
  it('the 8 wavefronts of a wave64 cube: 6 divergent, 2 idle', () => {
    const b = blockView(base({ width: 64 }));
    expect(b.divergent).toBe(6);
    expect(b.idle).toBe(2);
  });
  it('10 × 10 × 10 has a last warp with 8 real threads', () => {
    expect(warpsPerBlock(dims3(10, 10, 10), 32)).toBe(32);
    const last = warpView(base({ block: dims3(10, 10, 10), warp: 31 }));
    expect(last.threads).toBe(8);
    expect(last.padded).toBe(24);
  });
});

describe('the memory strip around a row', () => {
  it('8 × 8 × 8, interior: two fetched segments per row, 8 floats used of 16', () => {
    const cfg = base();
    const w = warpView(cfg);
    const strip = rowStrip(cfg, w, w.rows[0]);
    expect(strip.length).toBe(32); // two segments plus one of context on each side
    expect(strip.filter((c) => c.fetched).length).toBe(16);
    expect(strip.filter((c) => c.lane !== null).map((c) => c.k)).toEqual([5, 6, 7, 8, 9, 10, 11, 12]);
    expect(strip.filter((c) => c.lane !== null).every((c) => c.fetched)).toBe(true);
    // segment boundaries fall on multiples of 8 along the row because N is a multiple of 8
    expect(strip.filter((c) => c.index % 8 === 0).every((c) => c.k % 8 === 0)).toBe(true);
  });
  it('32 × 8 × 4: five fetched segments, 32 floats used of 40', () => {
    const cfg = base({ block: dims3(32, 8, 4) });
    const w = warpView(cfg);
    const strip = rowStrip(cfg, w, w.rows[0]);
    expect(strip.filter((c) => c.fetched).length).toBe(40);
    expect(strip.filter((c) => c.lane !== null).length).toBe(32);
  });
});
