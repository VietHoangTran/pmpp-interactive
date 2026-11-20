import { describe, expect, it } from 'vitest';
import {
  BOOK_RULE,
  H100_RATIO,
  better,
  breakEven,
  bytesRatio,
  copyFirstCost,
  lifetimeTables,
  mappedCost,
  mixedCurve,
  mixedTime,
  passCurve,
  ratioFromProbe,
} from './zerocopy';

describe('the numbers the C.2 questions state', () => {
  it('q-mixed: 90% device bytes, 10% mapped, link at 10%, no overlap: 1.9 times as long', () => {
    expect(mixedTime(0.1, 0.1, 'none')).toBeCloseTo(1.9, 12);
    // one tenth of the bytes is half of the time
    expect((0.1 / 0.1) / mixedTime(0.1, 0.1, 'none')).toBeCloseTo(1 / 1.9, 12);
    expect(0.1 / 0.1 / 1.9).toBeGreaterThan(0.5);
  });
  it('q-breakeven: 1/64 of the table per pass, link at 10%: copying wins above 71.1 passes, so from 72', () => {
    expect(breakEven(1 / 64, 0.1)).toBeCloseTo(64 / 0.9, 10);
    expect(breakEven(1 / 64, 0.1)).toBeCloseTo(71.11, 2);
    expect(better(71, 1 / 64, 0.1)).toBe('mapped');
    expect(better(72, 1 / 64, 0.1)).toBe('copy');
  });
  it('p-embedding: a 1 GiB table, 64 rows of 16 KiB: 1024 times fewer bytes cross the link in place', () => {
    expect(bytesRatio(2 ** 30, 64 * 16 * 2 ** 10)).toBe(1024);
  });
  it("q-membound (start page): a tenth of the bandwidth takes ten times as long when every byte comes over the link", () => {
    expect(mixedTime(1, 0.1, 'none')).toBeCloseTo(10, 12);
  });
});

describe('the model', () => {
  it('perfect overlap is never slower than no overlap, and both are 1 at f = 0', () => {
    for (const f of [0, 0.05, 0.3, 1]) expect(mixedTime(f, 0.1, 'perfect')).toBeLessThanOrEqual(mixedTime(f, 0.1, 'none'));
    expect(mixedTime(0, 0.1, 'none')).toBe(1);
    expect(mixedTime(0, 0.1, 'perfect')).toBe(1);
  });
  it('with perfect overlap the link takes over once f / rho > 1 - f, at f = rho / (1 + rho)', () => {
    const f = 0.1 / 1.1;
    expect(mixedTime(f, 0.1, 'perfect')).toBeCloseTo(1 - f, 12);
    expect(mixedTime(f + 0.01, 0.1, 'perfect')).toBeCloseTo((f + 0.01) / 0.1, 12);
  });
  it('a small mapped fraction already dominates: 5% of the bytes at 10% bandwidth adds half the time again', () => {
    expect(mixedTime(0.05, 0.1, 'none')).toBeCloseTo(1.45, 12);
  });
  it('the break-even pass count is where the two costs meet', () => {
    for (const [s, rho] of [[1, 0.1], [1 / 64, 0.038], [1 / 1024, 0.5]]) {
      const k = breakEven(s, rho);
      expect(mappedCost(k, s, rho)).toBeCloseTo(copyFirstCost(k, s, rho), 9);
    }
  });
  it('reading in place wins exactly while lifetime bytes stay below 1 / (1 - rho) tables', () => {
    const rho = 0.1;
    expect(lifetimeTables(rho)).toBeCloseTo(1.111, 3);
    expect(better(1.1, 1, rho)).toBe('mapped');
    expect(better(1.12, 1, rho)).toBe('copy');
    // a dense table read once: in place wins, barely
    expect(better(1, 1, rho)).toBe('mapped');
    expect(mappedCost(1, 1, rho) / copyFirstCost(1, 1, rho)).toBeCloseTo(10 / 11, 12);
  });
  it("the book's example device: about 3.8% (our arithmetic from 128 GB/s and 3.35 TB/s)", () => {
    expect(H100_RATIO).toBeCloseTo(0.0382, 4);
    expect(H100_RATIO).toBeLessThan(BOOK_RULE);
  });
  it('the probe row gives a ratio only when both numbers are present and positive', () => {
    expect(ratioFromProbe({ link_gb_per_s: 25, device_gb_per_s: 500 })).toBeCloseTo(0.05, 12);
    expect(ratioFromProbe({ link_gb_per_s: 25 })).toBeNull();
    expect(ratioFromProbe({ link_gb_per_s: '25', device_gb_per_s: 500 })).toBeNull();
    expect(ratioFromProbe({ link_gb_per_s: 0, device_gb_per_s: 500 })).toBeNull();
    expect(ratioFromProbe(undefined)).toBeNull();
  });
  it('curves cover the range they claim', () => {
    const c = mixedCurve(0.1, 'none');
    expect(c[0]).toEqual({ f: 0, t: 1 });
    expect(c[c.length - 1].t).toBeCloseTo(10, 12);
    const p = passCurve(1 / 64, 0.1);
    expect(p[0].k).toBe(1);
    expect(p[p.length - 1].k).toBeCloseTo(1000, 9);
  });
});
