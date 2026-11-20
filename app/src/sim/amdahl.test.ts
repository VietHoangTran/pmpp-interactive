import { describe, expect, it } from 'vitest';
import {
  BOOK_CASES,
  FRACTIONS,
  RATES,
  SPEEDUPS,
  THREADS,
  cappedKernel,
  ceiling,
  corners,
  fractionIndex,
  fractionNeeded,
  kernelNeeded,
  levers,
  limits,
  limitsCurve,
  logSpace,
  nearestIndex,
  remaining,
  speedup,
  speedupCurve,
  untouchedShare,
} from './amdahl';

describe("the book's worked cases (p. 9; notes/today/ch01-facts.md, A16)", () => {
  it('30% made 100 times faster: about 1.42, with 29.7% of the time saved', () => {
    expect(speedup(0.3, 100)).toBeCloseTo(1.4225, 4);
    expect(1 - remaining(0.3, 100).total).toBeCloseTo(0.297, 10);
  });
  it('30% made infinitely fast: no more than 1.43', () => {
    expect(speedup(0.3, Infinity)).toBeCloseTo(1.4286, 4);
    expect(speedup(0.3, Infinity)).toBe(ceiling(0.3));
  });
  it('99% made 100 times faster: about 50, with 1.99% of the time left', () => {
    expect(speedup(0.99, 100)).toBeCloseTo(50.25, 2);
    expect(remaining(0.99, 100).total).toBeCloseTo(0.0199, 10);
  });
  it('ceilings: 100 at 99% and 1,000 at 99.9%, the figure behind results above 100x (pp. 9–10)', () => {
    expect(speedup(0.99, Infinity)).toBeCloseTo(100, 9);
    expect(speedup(0.999, Infinity)).toBeCloseTo(1000, 6);
    expect(ceiling(0.92)).toBeCloseTo(12.5, 10);
  });
  it('the three presets are those cases', () => {
    expect(BOOK_CASES.map((c) => [c.p, c.s])).toEqual([[0.3, 100], [0.3, Infinity], [0.99, 100]]);
    expect(BOOK_CASES.map((c) => Number(speedup(c.p, c.s).toPrecision(3)))).toEqual([1.42, 1.43, 50.3]);
  });
  it('speed-up as the book defines it: 200 s against 10 s is 20', () => {
    expect(200 / 10).toBe(20);
    // The same thing through the law: everything sped up 20 times.
    expect(speedup(1, 20)).toBeCloseTo(20, 12);
  });
});

describe('numbers the lesson states', () => {
  it('at 99%, going from 100x to 1,000x on the part moves the whole from about 50 to about 91', () => {
    expect(speedup(0.99, 1000)).toBeCloseTo(90.99, 2);
  });
  it('q-kernelneeded: 96% of the time needs 96x for 20x overall; at 95% the target is the ceiling', () => {
    expect(kernelNeeded(20, 0.96)).toBeCloseTo(96, 6);
    expect(kernelNeeded(20, 0.95)).toBe(Infinity);
    expect(kernelNeeded(21, 0.95)).toBeNull();
  });
  it('q-hostshare: 8% left on the host caps the whole at 12.5x', () => {
    expect(ceiling(1 - 0.08)).toBeCloseTo(12.5, 10);
  });
  it('q-bigslice: 60% made 10 times faster beats 30% made infinitely fast', () => {
    expect(speedup(0.6, 10)).toBeCloseTo(2.174, 3);
    expect(speedup(0.6, 10)).toBeGreaterThan(speedup(0.3, Infinity));
  });
  it('fractionNeeded(10, 50) is about 0.918', () => {
    expect(fractionNeeded(10, 50)).toBeCloseTo(0.918, 3);
  });
  it('q-ceiling100: an unbounded part needs 99% for 100x and 99.9% for 1,000x', () => {
    expect(fractionNeeded(100, Infinity)).toBeCloseTo(0.99, 12);
    expect(fractionNeeded(1000, Infinity)).toBeCloseTo(0.999, 12);
  });
});

describe('properties of the law', () => {
  const ps = [0, 0.1, 0.3, 0.5, 0.9, 0.95, 0.99, 0.999, 0.9999];
  const ss = [1, 1.5, 2, 10, 100, 1000, 1e6, Infinity];
  it('never exceeds the ceiling, and reaches it only with an unbounded part', () => {
    for (const p of ps)
      for (const s of ss) {
        expect(speedup(p, s)).toBeLessThanOrEqual(ceiling(p) * (1 + 1e-12));
        if (s !== Infinity && p > 0) expect(speedup(p, s)).toBeLessThan(ceiling(p));
      }
  });
  it('is monotonic in s and in p', () => {
    for (const p of ps.slice(1)) for (let i = 1; i < ss.length; i++) expect(speedup(p, ss[i])).toBeGreaterThan(speedup(p, ss[i - 1]));
    for (const s of ss.slice(1)) for (let i = 1; i < ps.length; i++) expect(speedup(ps[i], s)).toBeGreaterThan(speedup(ps[i - 1], s));
  });
  it('s = 1 or p = 0 changes nothing', () => {
    for (const p of ps) expect(speedup(p, 1)).toBeCloseTo(1, 12);
    for (const s of ss) expect(speedup(0, s)).toBe(1);
  });
  it('the untouched part is the same before and after', () => {
    for (const p of ps) for (const s of ss) expect(remaining(p, s).untouched).toBe(remaining(p, 1).untouched);
  });
  it("the untouched part's share of what is left is the share of the ceiling reached", () => {
    for (const p of ps.slice(1)) for (const s of ss) expect(untouchedShare(p, s)).toBeCloseTo(speedup(p, s) / ceiling(p), 12);
    expect(untouchedShare(0.3, 100)).toBeCloseTo(0.9957, 4);
    expect(untouchedShare(0.99, 100)).toBeCloseTo(0.5025, 4);
  });
  it('fractionNeeded and speedup round-trip', () => {
    for (const s of [2, 10, 50, 100, 1000, Infinity])
      for (const target of [1, 1.2, 1.9, 5, 9.5, 40, 99, 500]) {
        const p = fractionNeeded(target, s);
        if (target > s) expect(p).toBeNull();
        else expect(speedup(p!, s)).toBeCloseTo(target, 8);
      }
    expect(fractionNeeded(50, 50)).toBe(1);
    expect(fractionNeeded(0.5, 10)).toBeNull();
  });
  it('kernelNeeded and speedup round-trip', () => {
    for (const p of [0.3, 0.6, 0.9, 0.96, 0.99])
      for (const target of [1.1, 1.4, 2, 9, 20, 60]) {
        const s = kernelNeeded(target, p);
        if (target > ceiling(p) * (1 + 1e-9)) expect(s).toBeNull();
        else expect(speedup(p, s!)).toBeCloseTo(target, 8);
      }
  });
});

describe('what caps the kernel', () => {
  it('the smaller of threads x rate and the memory cap, with its name', () => {
    expect(cappedKernel(1000, 0.25, 10)).toEqual({ s: 10, limitedBy: 'memory' });
    expect(cappedKernel(1000, 0.25, 1000)).toEqual({ s: 250, limitedBy: 'threads' });
    expect(cappedKernel(40, 0.25, 10)).toEqual({ s: 10, limitedBy: 'memory' });
    expect(cappedKernel(1, 0.125, 10)).toEqual({ s: 0.125, limitedBy: 'threads' });
  });
  it("the defaults: 99% parallel, the book's memory cap of 10 (p. 10) gives 9.17 whatever the thread count", () => {
    const a = limits(0.99, 1000, 0.25, 10);
    expect(a.s).toBe(10);
    expect(a.limitedBy).toBe('memory');
    expect(a.overall).toBeCloseTo(9.174, 3);
    expect(a.untouchedShare).toBeCloseTo(0.0917, 4);
    for (const n of [40, 100, 1000, 100000]) expect(limits(0.99, n, 0.25, 10).overall).toBe(a.overall);
  });
  it('raising the memory cap to 100 gives 50.25; to 1,000 the threads become the limit at 250 and 71.6', () => {
    expect(limits(0.99, 1000, 0.25, 100).overall).toBeCloseTo(50.25, 2);
    const c = limits(0.99, 1000, 0.25, 1000);
    expect(c.s).toBe(250);
    expect(c.limitedBy).toBe('threads');
    expect(c.overall).toBeCloseTo(71.6, 1);
  });
  it('past 100 overall needs a larger parallel fraction as well', () => {
    expect(limits(0.99, 100000, 1, 1000).overall).toBeLessThan(100);
    expect(limits(0.999, 1000, 0.25, 1000).overall).toBeCloseTo(200.2, 1);
    expect(limits(0.999, 1000, 0.25, 100).overall).toBeCloseTo(91, 0);
  });
  it('too few threads is slower than the CPU core', () => {
    expect(limits(0.99, 1, 0.25, 10).overall).toBeLessThan(1);
    expect(limits(0.99, 4, 0.25, 10).overall).toBeCloseTo(1, 12);
  });
  it('corners: break-even at 1 / rate threads, saturation at cap / rate', () => {
    expect(corners(0.25, 10)).toEqual({ breakEven: 4, saturation: 40 });
    expect(cappedKernel(corners(0.25, 10).saturation, 0.25, 10).limitedBy).toBe('memory');
    expect(cappedKernel(corners(0.25, 10).saturation - 1, 0.25, 10).limitedBy).toBe('threads');
  });
  it('levers: at the defaults only the memory cap is worth much', () => {
    const l = levers(0.99, 1000, 0.25, 10);
    expect(l.now).toBeCloseTo(9.174, 3);
    expect(l.threads).toBe(l.now);
    expect(l.memory).toBeCloseTo(50.25, 2);
    expect(l.untouched).toBeCloseTo(9.911, 3);
  });
  it('levers: once the kernel is fast, the untouched part is what is left', () => {
    const l = levers(0.99, 1000, 0.25, 100);
    expect(l.now).toBeCloseTo(50.25, 2);
    expect(l.threads).toBe(l.now);
    expect(l.memory).toBeCloseTo(71.63, 2);
    expect(l.untouched).toBeCloseTo(90.99, 2);
  });
});

describe('slider stops and curves', () => {
  it('fractions run from 30% to 99.99%, rising, and include every case the lesson uses', () => {
    expect(FRACTIONS[0]).toBe(0.3);
    expect(FRACTIONS[FRACTIONS.length - 1]).toBe(0.9999);
    for (let i = 1; i < FRACTIONS.length; i++) expect(FRACTIONS[i]).toBeGreaterThan(FRACTIONS[i - 1]);
    for (const p of [0.3, 0.6, 0.92, 0.95, 0.96, 0.99, 0.999, 0.9999]) expect(FRACTIONS).toContain(p);
  });
  it('fractions get finer towards 1', () => {
    const step = (i: number) => FRACTIONS[i + 1] - FRACTIONS[i];
    expect(step(0)).toBeGreaterThan(step(FRACTIONS.length - 2) * 1000);
  });
  it('speed-ups run from 1 to 1,000 and thread counts from 1 to 100,000', () => {
    expect([SPEEDUPS[0], SPEEDUPS[SPEEDUPS.length - 1]]).toEqual([1, 1000]);
    for (const s of [1, 10, 20, 50, 100, 250, 1000]) expect(SPEEDUPS).toContain(s);
    expect([THREADS[0], THREADS[THREADS.length - 1]]).toEqual([1, 100000]);
    for (const n of [1, 4, 40, 1000, 100000]) expect(THREADS).toContain(n);
    expect(THREADS.every(Number.isInteger)).toBe(true);
    expect(RATES.map((r) => r.value)).toEqual([0.125, 0.25, 0.5, 1]);
  });
  it('nearestIndex finds a stop and snaps values between stops', () => {
    expect(SPEEDUPS[nearestIndex(SPEEDUPS, 100)]).toBe(100);
    expect(SPEEDUPS[nearestIndex(SPEEDUPS, 97)]).toBe(100);
    expect(FRACTIONS[fractionIndex(0.99)]).toBe(0.99);
    expect(FRACTIONS[fractionIndex(0.9991)]).toBe(0.9991);
    expect(FRACTIONS[fractionIndex(0.1)]).toBe(0.3);
  });
  it('logSpace hits both ends', () => {
    const xs = logSpace(1, 1000, 4);
    expect(xs[0]).toBeCloseTo(1, 12);
    expect(xs[1]).toBeCloseTo(10, 9);
    expect(xs[3]).toBe(1000);
  });
  it('the Amdahl curve rises towards the ceiling and never crosses it', () => {
    const c = speedupCurve(0.99);
    expect(c[0]).toEqual({ s: 1, overall: speedup(0.99, 1) });
    expect(c[c.length - 1].s).toBe(1000);
    for (let i = 1; i < c.length; i++) expect(c[i].overall).toBeGreaterThan(c[i - 1].overall);
    expect(c[c.length - 1].overall).toBeLessThan(ceiling(0.99));
  });
  it('the thread curve passes through both corners and is flat after the second', () => {
    const c = limitsCurve(0.99, 0.25, 10);
    expect(c.some((q) => q.threads === 4 && Math.abs(q.overall - 1) < 1e-12)).toBe(true);
    const at = c.findIndex((q) => q.threads === 40);
    expect(at).toBeGreaterThan(0);
    for (const q of c.slice(at)) expect(q.overall).toBe(c[at].overall);
    for (let i = 1; i <= at; i++) expect(c[i].overall).toBeGreaterThanOrEqual(c[i - 1].overall);
  });
});
