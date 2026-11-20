import { describe, expect, it } from 'vitest';
import {
  blocksSlab,
  cubeAI,
  extraRegsPerBlock,
  flopPerOutput,
  lifetimeCounts,
  planesLoaded,
  points,
  sharedBytes,
  slabAI,
  slabAILimit,
  threadsCube,
  threadsSlab,
  windowRegs,
} from './stencilslab';

const r2 = (v: number) => Math.round(v * 100) / 100;

describe('the stencil itself', () => {
  it('order 1 is the 7-point stencil with 13 FLOP per output (p. 189)', () => {
    expect(points(1)).toBe(7);
    expect(flopPerOutput(1)).toBe(13);
    expect(points(2)).toBe(13); // Fig. 8.3(d)
  });
});

describe('FLOP/B of the coarsened kernel (pp. 192, 195; facts E1, E2)', () => {
  it('Fig. 8.10 as coded, t = 32 and 30 output planes: 1.47, the same as a 32-wide cube', () => {
    expect(slabAI(32, 30)).toBeCloseTo(1.468, 3);
    expect(r2(slabAI(32, 30))).toBe(1.47);
    expect(slabAI(32, 30)).toBeCloseTo(cubeAI(32), 12);
  });
  it('exercise 2c: 16 output planes give 1.43, from 18,432 inputs and 14,400 outputs', () => {
    expect(lifetimeCounts(32, 16)).toEqual({ inputs: 18432, outputs: 14400 });
    expect(r2(slabAI(32, 16))).toBe(1.43);
    expect(slabAI(32, 16)).toBeCloseTo(187200 / 131328, 12);
  });
  it('deeper slabs creep up: 1.50 at 64 planes, 1.51 at 128', () => {
    expect(r2(slabAI(32, 64))).toBe(1.5);
    expect(r2(slabAI(32, 128))).toBe(1.51);
  });
  it("the book's printed 1.52 is the limit for an unboundedly deep slab", () => {
    expect(slabAILimit(32)).toBeCloseTo(1.520, 3);
    expect(slabAILimit(32)).toBeCloseTo((13 * 900) / (4 * (1024 + 900)), 12);
    expect(slabAI(32, 1e7)).toBeCloseTo(slabAILimit(32), 5);
    // Every finite slab is below it.
    for (const D of [1, 2, 30, 512]) expect(slabAI(32, D)).toBeLessThan(slabAILimit(32));
  });
  it('a slab D = t - 2 deep is the cube of p. 192, written out', () => {
    for (const t of [4, 6, 8, 10, 16, 32]) {
      const cube = (13 * (t - 2) ** 3) / (4 * t ** 3 + 4 * (t - 2) ** 3);
      expect(slabAI(t, t - 2)).toBeCloseTo(cube, 12);
    }
    expect(r2(cubeAI(8))).toBe(0.96);
  });
  it('8-byte values halve every figure (facts A7)', () => {
    expect(slabAI(32, 30, 1, 8)).toBeCloseTo(slabAI(32, 30) / 2, 12);
  });
});

describe('block size, shared memory and registers', () => {
  it('a plane of threads is t^2: 1024 at t = 32; a cube would be 32,768', () => {
    expect(threadsSlab(32)).toBe(1024);
    expect(threadsCube(32)).toBe(32768);
    expect(threadsCube(6)).toBe(216);
  });
  it('three 32 x 32 planes are 12,288 B; register tiling leaves one, 4,096 B (exercises 2d, 2e)', () => {
    expect(sharedBytes('planes', 32)).toBe(12288);
    expect(sharedBytes('registers', 32)).toBe(4096);
    expect(sharedBytes('planes', 32) / sharedBytes('registers', 32)).toBe(3);
  });
  it('order 2 needs a window of 5 planes', () => {
    expect(windowRegs(2)).toBe(5);
    expect(planesLoaded(30, 2)).toBe(34);
    expect(sharedBytes('planes', 32, 2)).toBe(5 * 4096);
    expect(sharedBytes('registers', 32, 2)).toBe(4096);
  });
  it('register tiling costs 3 registers per thread, 3,072 per 32 x 32 block (p. 197)', () => {
    expect(windowRegs(1)).toBe(3);
    expect(extraRegsPerBlock(32, 1)).toBe(3072);
  });
});

describe('blocks launched (the book indexing)', () => {
  it('exercise 1d: 120^3 grid, 32 x 32 blocks, 30-plane slabs: 64 blocks', () => {
    expect(blocksSlab(120, 32, 30)).toBe(64);
  });
  it('deeper slabs, fewer blocks: 729 at D = 30, 162 at D = 254 on a 256^3 grid', () => {
    expect(blocksSlab(256, 32, 30)).toBe(729);
    expect(blocksSlab(256, 32, 254)).toBe(162);
  });
  it('planes loaded per block are D + 2r', () => {
    expect(planesLoaded(30)).toBe(32);
    expect(planesLoaded(16)).toBe(18);
  });
});
