import { describe, expect, it } from 'vitest';
import {
  basicAI,
  basicCounts,
  blocksBasic,
  blocksCube,
  box,
  boxAI,
  cubeAI,
  denseIdealAI,
  denseTaps,
  haloCount,
  haloFraction,
  haloFraction2D,
  idealAIFinite,
  interiorPoints,
  maxCubeSide,
  neverRead,
  points,
  role,
  sharedBytesCube,
  sideForShare,
  stencilFlops,
  stencilIdealAI,
  threadsCube,
  tileOutputs,
} from './stencil3d';

const r2 = (v: number) => Math.round(v * 100) / 100;

describe('stencil shapes (book pp. 186 to 187)', () => {
  it('counts points the way Figs. 8.2 and 8.3 draw them', () => {
    expect([1, 2, 3].map((r) => points(1, r))).toEqual([3, 5, 7]);
    expect(points(2, 1)).toBe(5);
    expect(points(2, 2)).toBe(9);
    expect(points(3, 1)).toBe(7);
    expect(points(3, 2)).toBe(13);
    expect(points(3, 3)).toBe(19);
  });
  it('dense boxes of the same extent', () => {
    expect(denseTaps(2, 1)).toBe(9);
    expect(denseTaps(3, 1)).toBe(27);
    expect(denseTaps(3, 3)).toBe(343);
  });
  it('13 FLOP for a 7-point output (p. 189)', () => {
    expect(stencilFlops(7)).toBe(13);
  });
});

describe('ideal ratios: stencil against dense filter (book p. 190)', () => {
  it('stencil column: 1.125, 2.125, 3.125, 4.625', () => {
    expect(stencilIdealAI(points(2, 1))).toBe(1.125);
    expect(stencilIdealAI(points(2, 2))).toBe(2.125);
    expect(stencilIdealAI(points(2, 3))).toBe(3.125);
    expect(stencilIdealAI(points(3, 3))).toBe(4.625);
  });
  it('convolution column: 2.25, 6.25, 12.25, 85.75', () => {
    expect(denseIdealAI(2, 1)).toBe(2.25);
    expect(denseIdealAI(2, 2)).toBe(6.25);
    expect(denseIdealAI(2, 3)).toBe(12.25);
    expect(denseIdealAI(3, 3)).toBe(85.75);
  });
  it('3D order 1: 1.625 for the stencil, 6.75 for the 3 × 3 × 3 filter', () => {
    expect(stencilIdealAI(7)).toBe(1.625);
    expect(denseIdealAI(3, 1)).toBe(6.75);
  });
  it('8-byte values halve every ratio, 2-byte values double it', () => {
    expect(stencilIdealAI(7, 8)).toBe(0.8125);
    expect(stencilIdealAI(7, 2)).toBe(3.25);
  });
});

describe('the basic kernel, Fig. 8.6 (book p. 189)', () => {
  it('13 FLOP over seven 4-byte loads and one 4-byte store: 0.40625', () => {
    expect(basicAI(7)).toBe(0.40625);
    expect(basicCounts(7)).toEqual({ flop: 13, loads: 7, stores: 1, bytes: 32 });
  });
  it('the ideal at finite n approaches 13/8 (facts E1)', () => {
    expect(r2(idealAIFinite(64))).toBe(1.55);
    expect(r2(idealAIFinite(128))).toBe(1.59);
    expect(r2(idealAIFinite(512))).toBe(1.62);
    expect(idealAIFinite(1e5)).toBeLessThan(1.625);
    expect(idealAIFinite(1e5)).toBeGreaterThan(1.6249);
  });
});

describe('the cube-tiled kernel, Fig. 8.8 (book pp. 192 to 193; facts E1)', () => {
  it("the book's t = 8: 0.96 FLOP/B and 58% halo", () => {
    expect(r2(cubeAI(8))).toBe(0.96);
    expect(haloCount(8)).toBe(296);
    expect(r2(100 * haloFraction(8))).toBe(57.81);
  });
  it('the E1 table, t = 4 to 32', () => {
    const rows = [4, 6, 8, 10, 16, 32].map((t) => [t, r2(cubeAI(t)), Math.round(1000 * haloFraction(t)) / 10, sharedBytesCube(t)]);
    expect(rows).toEqual([
      [4, 0.36, 87.5, 256],
      [6, 0.74, 70.4, 864],
      [8, 0.96, 57.8, 2048],
      [10, 1.1, 48.8, 4000],
      [16, 1.3, 33, 16384],
      [32, 1.47, 17.6, 131072],
    ]);
  });
  it('the formula is the book\'s, written for r = 1', () => {
    for (const t of [3, 5, 8, 13, 32]) expect(cubeAI(t)).toBeCloseTo((13 * (t - 2) ** 3) / (4 * (t ** 3 + (t - 2) ** 3)), 12);
  });
  it('the 2D comparison: 124 of 1024 halo for a 32 × 32 tile, about 12%', () => {
    expect(Math.round(1024 * haloFraction2D(32))).toBe(124);
    expect(r2(100 * haloFraction2D(32))).toBe(12.11);
  });
  it('every ratio stays under the ideal and rises with t', () => {
    let prev = 0;
    for (let t = 3; t <= 64; t++) {
      const a = cubeAI(t);
      expect(a).toBeGreaterThan(prev);
      expect(a).toBeLessThan(1.625);
      prev = a;
    }
  });
  it('8-byte values: 0.48 FLOP/B and 4,096 bytes at t = 8', () => {
    expect(r2(cubeAI(8, 1, 8))).toBe(0.48);
    expect(sharedBytesCube(8, 8)).toBe(4096);
  });
  it('the largest cube under 1024 threads is 10, not 8 (facts A1)', () => {
    expect(maxCubeSide(1024)).toBe(10);
    expect(maxCubeSide(1000)).toBe(10);
    expect(maxCubeSide(999)).toBe(9);
    expect(maxCubeSide(512)).toBe(8);
    expect(threadsCube(10)).toBe(1000);
  });
  it('reaching 90% of the ideal needs a side of 31: 29,791 threads', () => {
    expect(sideForShare(0.9)).toBe(31);
    expect(threadsCube(31)).toBe(29791);
  });
});

describe('elements no thread reads (ours)', () => {
  it('80 of the 512 in an 8 × 8 × 8 tile: 12 edges of 6 and 8 corners', () => {
    expect(neverRead(8)).toBe(80);
    expect(12 * 6 + 8).toBe(80);
  });
  it('agrees with a cell-by-cell classification for several sizes and orders', () => {
    for (const [t, r] of [[3, 1], [8, 1], [10, 1], [9, 2], [13, 3]]) {
      const count = { output: 0, halo: 0, never: 0 };
      for (let z = 0; z < t; z++) for (let y = 0; y < t; y++) for (let x = 0; x < t; x++) count[role(t, r, x, y, z)]++;
      expect(count.output).toBe((t - 2 * r) ** 3);
      expect(count.never).toBe(neverRead(t, r));
      expect(count.halo + count.never).toBe(haloCount(t, r));
    }
  });
  it('the 2D input tile has no corners (Fig. 8.7)', () => {
    // slice z = 1 of a cube is a 2D tile with its own halo: corners are never read
    expect(role(6, 1, 0, 0, 1)).toBe('never');
    expect(role(6, 1, 0, 2, 1)).toBe('halo');
    expect(role(6, 1, 2, 2, 1)).toBe('output');
  });
});

describe('tiles that are not cubes', () => {
  it('a 32 × 8 × 4 block has a 30 × 6 × 2 output tile and 0.85 FLOP/B', () => {
    expect(tileOutputs(box(32, 8, 4), 1)).toBe(360);
    expect(r2(boxAI(box(32, 8, 4), 1))).toBe(0.85);
  });
  it('a cube is the special case', () => {
    expect(boxAI(box(8, 8, 8), 1)).toBe(cubeAI(8));
  });
});

describe('book exercise 1 (p. 198)', () => {
  it('1a: 118³ outputs on a 120³ grid', () => {
    expect(interiorPoints(120)).toBe(1643032);
  });
  it('1b: 15³ blocks of 8³ for the basic kernel', () => {
    expect(blocksBasic(120, 8)).toBe(3375);
  });
  it('1c: 20³ blocks for the tiled kernel (output tile 6)', () => {
    expect(blocksCube(120, 8)).toBe(8000);
  });
});
