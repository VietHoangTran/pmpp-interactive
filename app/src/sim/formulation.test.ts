import { describe, expect, it } from 'vitest';
import {
  GiB,
  LLAMA2_7B,
  SERIES,
  advantage,
  cachedOps,
  crossover,
  curves,
  cutoffCoulomb,
  directCoulomb,
  kvBytes,
  recomputeOps,
  slope,
  tieFactor,
  time,
} from './formulation';

describe('p-crossover', () => {
  it('10 times faster at size 1, square against linear: they tie at a factor of 10', () => {
    expect(tieFactor(10, 2, 1)).toBeCloseTo(10, 12);
  });
  it('the explorer reproduces it: S = 1000 makes direct on the GPU 10 times faster at size 1, and the curves cross at 10', () => {
    const c = curves('coulomb');
    expect(advantage(c, 1, 1000)).toBeCloseTo(10, 9);
    expect(crossover(c, 1000)).toBeCloseTo(10, 6);
  });
});

describe('the electrostatic preset', () => {
  it('direct grows with the square of the volume, cutoff linearly', () => {
    expect(directCoulomb(2) / directCoulomb(1)).toBeCloseTo(4, 12);
    expect(cutoffCoulomb(2) / cutoffCoulomb(1)).toBeCloseTo(2, 12);
    expect(slope(directCoulomb, 3)).toBeCloseTo(2, 6);
    expect(slope(cutoffCoulomb, 3)).toBeCloseTo(1, 6);
  });
  it('below the volume of one neighbourhood the two cost the same', () => {
    expect(cutoffCoulomb(0.005)).toBe(directCoulomb(0.005));
    expect(cutoffCoulomb(1)).toBeLessThan(directCoulomb(1));
  });
  it('a speed-up moves a curve down and leaves its slope alone', () => {
    const c = curves('coulomb');
    for (const s of SERIES)
      for (const S of [1, 10, 1000]) {
        expect(slope((x) => time(c, s, x, S), 5)).toBeCloseTo(s.startsWith('direct') ? 2 : 1, 6);
      }
    expect(time(c, 'direct-fast', 5, 100)).toBeCloseTo(time(c, 'direct-slow', 5, 100) / 100, 6);
  });
  it('the crossing moves up with the speed-up', () => {
    const c = curves('coulomb');
    let prev = 0;
    for (const S of [1, 2, 10, 50, 100, 300, 1000]) {
      const x = crossover(c, S)!;
      expect(x).toBeGreaterThanOrEqual(prev);
      prev = x;
    }
    expect(crossover(c, 100)).toBeCloseTo(1, 6);
  });
});

describe('the KV-cache preset (ours, from chapter 20)', () => {
  it('recomputing the prefix costs exactly T times the cached step', () => {
    for (const t of [1, 7, 512, 4096, 50000]) expect(recomputeOps(t)).toBe(t * cachedOps(t));
  });
  it('so the slopes differ by exactly one at every length', () => {
    for (const t of [2, 100, 4096, 30000]) expect(slope((x) => recomputeOps(x), t) - slope((x) => cachedOps(x), t)).toBeCloseTo(1, 6);
  });
  it('recompute on the fast processor ties the cache on the slow one at T = S', () => {
    const c = curves('kv');
    for (const S of [10, 100, 1000]) expect(crossover(c, S)).toBeCloseTo(S, 4);
  });
  it('the crossing moves up with the speed-up', () => {
    const c = curves('kv');
    let prev = 0;
    for (const S of [2, 4, 16, 64, 500]) {
      const x = crossover(c, S)!;
      expect(x).toBeGreaterThan(prev);
      prev = x;
    }
  });
});

describe('KV-cache memory, Eq. 20.8', () => {
  it("the book's Llama 2 7B example: one sequence of 4096 tokens is 2 GiB", () => {
    expect(kvBytes(4096, 1)).toBe(2 * GiB);
    expect(kvBytes(1, 1)).toBe(2 * LLAMA2_7B.layers * LLAMA2_7B.hidden * LLAMA2_7B.bytes);
  });
  it('is linear in length and in batch', () => {
    expect(kvBytes(8192, 1)).toBe(2 * kvBytes(4096, 1));
    expect(kvBytes(4096, 16)).toBe(16 * kvBytes(4096, 1));
    expect(kvBytes(1000, 3) + kvBytes(1000, 5)).toBe(kvBytes(1000, 8));
  });
});
