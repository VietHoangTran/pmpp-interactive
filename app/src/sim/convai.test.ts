import { describe, expect, it } from 'vitest';
import {
  FILTER_WIDTHS, basicIntensity, basicIntensityWithStore, ceiling, constIntensity, flops, idealIntensity,
  multipleOfCeiling, regime, smallestComputeBoundWidth, storeShare, threshold,
} from './convai';

// The book's example device (p. 94 of chapter 5, used again on p. 167): 66.9 TFLOPS, 3.35 TB/s.
const BOOK = { peak: 66900, bw: 3350 };

describe('the two kernels as written (pp. 167 and 171)', () => {
  it('the basic kernel asks 0.25 FLOP/B of memory in float32', () => {
    expect(basicIntensity(4)).toBe(0.25);
  });
  it('the constant-memory kernel doubles it to 0.5', () => {
    expect(constIntensity(4)).toBe(0.5);
  });
  it('neither depends on the filter width: each tap brings its own loads', () => {
    // basicIntensity takes no m at all; the per-tap count is the whole story.
    expect(basicIntensityWithStore(3)).toBeCloseTo(18 / (19 * 4), 12);
    expect(basicIntensityWithStore(11)).toBeCloseTo(242 / (243 * 4), 12);
  });
  it('halving the bytes per element doubles both', () => {
    expect(basicIntensity(2)).toBe(0.5);
    expect(constIntensity(2)).toBe(1);
    expect(basicIntensity(1)).toBe(1);
  });
});

describe('the ideal (p. 167 and the Ideal column of Fig. 7.14)', () => {
  it('m²/4 in float32 for m = 3 to 13', () => {
    expect([3, 5, 7, 9, 11, 13].map((m) => idealIntensity(m))).toEqual([2.25, 6.25, 12.25, 20.25, 30.25, 42.25]);
  });
  it('total work is 2·n²·m² FLOP', () => {
    expect(flops(1024, 5)).toBe(2 * 1024 * 1024 * 25);
  });
  it('keeping the filter in the traffic changes it only slightly for a large image', () => {
    const exact = idealIntensity(11, 4, 2, 1024);
    expect(exact).toBeLessThan(30.25);
    expect(exact).toBeCloseTo(30.25, 2);
    // and by a lot for a tiny one
    expect(idealIntensity(11, 4, 2, 8)).toBeCloseTo(15488 / 996, 10);
  });
  it('1D: m/4, so 9 taps give 2.25', () => {
    expect(idealIntensity(9, 4, 1)).toBe(2.25);
    expect(idealIntensity(4, 4, 1)).toBe(1);
  });
  it('fp16 3×3 gives 4.5: m² divided by the bytes per element', () => {
    expect(idealIntensity(3, 2)).toBe(4.5);
  });
  it('the explorer offers radius 0 to 7', () => {
    expect(FILTER_WIDTHS).toEqual([1, 3, 5, 7, 9, 11, 13, 15]);
  });
});

describe('the store the book ignores (p. 167, counted again on p. 176)', () => {
  it('at 5×5 it adds 4 bytes to 200, about 2%', () => {
    expect(storeShare(5)).toBeCloseTo(4 / 204, 12);
    expect(storeShare(5)).toBeGreaterThan(0.019);
    expect(storeShare(5)).toBeLessThan(0.02);
  });
});

describe('against the book\'s example device', () => {
  const thr = threshold(BOOK);
  it('threshold is about 20 FLOP/B', () => {
    expect(thr).toBeCloseTo(19.97, 2);
  });
  it('3×3 cannot be compute-bound even in the ideal case; 11×11 can', () => {
    expect(regime(idealIntensity(3), thr)).toBe('memory-bound');
    expect(regime(idealIntensity(11), thr)).toBe('compute-bound');
  });
  it('9 and 11 (20.25 and 30.25) clear it; 3, 5 and 7 do not', () => {
    expect([3, 5, 7, 9, 11].map((m) => regime(idealIntensity(m), thr))).toEqual([
      'memory-bound', 'memory-bound', 'memory-bound', 'compute-bound', 'compute-bound',
    ]);
    expect(regime(idealIntensity(9), 20)).toBe('compute-bound');
  });
  it('the smallest odd width that could be compute-bound: 9 in float32, 7 in fp16, 81 taps in 1D', () => {
    expect(smallestComputeBoundWidth(thr)).toBe(9);
    expect(smallestComputeBoundWidth(20)).toBe(9);
    expect(smallestComputeBoundWidth(thr, 2)).toBe(7);
    expect(smallestComputeBoundWidth(20, 4, 1)).toBe(81);
    expect(smallestComputeBoundWidth(0.1)).toBe(1);
  });
  it('the basic kernel is capped at 0.25 × 3350 = 837.5 GFLOP/s, the constant kernel at 1675', () => {
    expect(ceiling(basicIntensity(), BOOK)).toBe(837.5);
    expect(ceiling(constIntensity(), BOOK)).toBe(1675);
    expect(ceiling(idealIntensity(11), BOOK)).toBe(66900);
  });
  it('a measurement is read as a multiple of its ceiling', () => {
    expect(multipleOfCeiling(1675, 0.25, BOOK)).toBe(2);
  });
});
