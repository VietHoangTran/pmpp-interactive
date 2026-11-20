import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  DEMO_SEED, amdScalarEligible, codes, distinctCount, expectedDistinct, filterTrace, largestOddWidth, nvidiaCost,
  replaySchedule, warpAddresses,
} from './constcache';

describe('one warp, four read patterns', () => {
  for (const warpSize of [32, 64])
    describe(`${warpSize} lanes`, () => {
      it('a filter tap is one address for every lane: cost 1, and AMD\'s scalar cache can serve it', () => {
        const a = warpAddresses('uniform', { warpSize, entry: 7 });
        expect(a).toHaveLength(warpSize);
        expect(nvidiaCost(a)).toBe(1);
        expect(amdScalarEligible(a)).toBe(true);
      });
      it('T[lane] is one address per lane: cost = warp width, not uniform', () => {
        const a = warpAddresses('lane', { warpSize });
        expect(nvidiaCost(a)).toBe(warpSize);
        expect(amdScalarEligible(a)).toBe(false);
      });
      it('T[lane/8] is one address per eight lanes', () => {
        const a = warpAddresses('laneGroup', { warpSize, groupSize: 8 });
        expect(nvidiaCost(a)).toBe(warpSize / 8);
        expect(amdScalarEligible(a)).toBe(false);
      });
    });
  it('T[lane/8] at 32 lanes costs 4', () => {
    expect(distinctCount(warpAddresses('laneGroup', { warpSize: 32, groupSize: 8 }))).toBe(4);
  });
  it('T[lane/32] at 32 lanes is uniform again', () => {
    expect(amdScalarEligible(warpAddresses('laneGroup', { warpSize: 32, groupSize: 32 }))).toBe(true);
  });
});

describe('a 16-entry codebook indexed by random 4-bit codes', () => {
  it('a seed always gives the same codes, all in range', () => {
    expect(codes(DEMO_SEED, 32, 16)).toEqual(codes(DEMO_SEED, 32, 16));
    expect(codes(DEMO_SEED, 32, 16).every((c) => Number.isInteger(c) && c >= 0 && c < 16)).toBe(true);
    expect(codes(1, 32, 16)).not.toEqual(codes(2, 32, 16));
  });
  it('the demo seed touches 14 of 16 entries at 32 lanes', () => {
    expect(distinctCount(warpAddresses('codebook', { warpSize: 32, tableSize: 16, seed: DEMO_SEED }))).toBe(14);
  });
  it('expected distinct entries: 16(1 - (15/16)^32) = 13.97, and 15.74 at 64 lanes', () => {
    expect(expectedDistinct(16, 32)).toBeCloseTo(13.97, 2);
    expect(expectedDistinct(16, 64)).toBeCloseTo(15.74, 2);
    expect(expectedDistinct(256, 32)).toBeCloseTo(30.14, 2);
  });
  it('the mean over many seeds is within 2% of the formula', () => {
    for (const [k, lanes] of [[16, 32], [16, 64], [4, 32], [256, 32]]) {
      let sum = 0;
      const N = 4000;
      for (let s = 1; s <= N; s++) sum += distinctCount(codes(s, lanes, k));
      expect(Math.abs(sum / N - expectedDistinct(k, lanes)) / expectedDistinct(k, lanes)).toBeLessThan(0.02);
    }
  });
  it('a codebook read is never uniform in practice, so AMD\'s scalar cache does not serve it', () => {
    expect(amdScalarEligible(warpAddresses('codebook', { warpSize: 32, tableSize: 16, seed: DEMO_SEED }))).toBe(false);
  });
});

describe('the replay schedule drawn by the scene', () => {
  const cases = [
    warpAddresses('uniform', { warpSize: 32 }),
    warpAddresses('lane', { warpSize: 64 }),
    warpAddresses('laneGroup', { warpSize: 32 }),
    warpAddresses('codebook', { warpSize: 32, tableSize: 16, seed: DEMO_SEED }),
    warpAddresses('codebook', { warpSize: 64, tableSize: 16, seed: 11 }),
  ];
  it('one round per distinct address', () => {
    for (const a of cases) expect(replaySchedule(a)).toHaveLength(distinctCount(a));
  });
  it('the rounds cover every lane exactly once, each with the address it asked for', () => {
    for (const a of cases) {
      const served = replaySchedule(a).flatMap((r) => r.lanes.map((l) => [l, r.address] as const));
      expect(served.map(([l]) => l).sort((x, y) => x - y)).toEqual(a.map((_, i) => i));
      for (const [l, addr] of served) expect(a[l]).toBe(addr);
    }
  });
  it('a uniform read is a single round answering all lanes', () => {
    const r = replaySchedule(cases[0]);
    expect(r).toEqual([{ address: 0, lanes: Array.from({ length: 32 }, (_, i) => i) }]);
  });
});

describe('the filter as a sequence of uniform reads', () => {
  it('radius 2: 25 taps, 25 reads, in row-major order', () => {
    const t = filterTrace(2);
    expect(t).toHaveLength(25);
    expect(t[0]).toEqual({ fRow: 0, fCol: 0, address: 0 });
    expect(t[7]).toEqual({ fRow: 1, fCol: 2, address: 7 });
    expect(t.map((x) => x.address)).toEqual(Array.from({ length: 25 }, (_, i) => i));
  });
  it('radius 7: 225 reads per warp, 900 bytes of weights', () => {
    expect(filterTrace(7)).toHaveLength(225);
  });
});

describe('what fits where', () => {
  it('8 KB working set per SM (A3): odd widths up to 45 (2025 floats); 47² = 2209 does not fit', () => {
    expect(largestOddWidth(8192)).toBe(45);
    expect(47 * 47 * 4).toBeGreaterThan(8192);
  });
  it('64 KB of constant memory: up to 127', () => {
    expect(largestOddWidth(65536)).toBe(127);
  });
  it('Metal setBytes, under 4 KB: 31 wide (a 32×32 filter is 4096 bytes and does not fit)', () => {
    expect(largestOddWidth(4095)).toBe(31);
  });
});

describe('the narration names the model\'s numbers', () => {
  const dir = path.resolve(__dirname, '../../../content/ch07/narration');
  const script = (n: number) => fs.readFileSync(path.join(dir, `constant-broadcast-${n}.txt`), 'utf8');
  it('part 2 says a 5 by 5 filter costs 25 answers', () => {
    expect(filterTrace(2)).toHaveLength(25);
    expect(script(2)).toMatch(/\b25\b/);
  });
  it('part 3 says the demo codes hit 14 entries', () => {
    expect(script(3)).toMatch(/\b14\b/);
  });
});
