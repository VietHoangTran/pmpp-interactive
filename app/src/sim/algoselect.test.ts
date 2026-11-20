import { describe, expect, it } from 'vitest';
import {
  ALGOS,
  brentKungUpTo,
  coarsenedThreads,
  compare,
  crossovers,
  curve,
  hybridLag,
  hybridRatio,
  koggeStoneFrom,
  networkWinner,
  overallWinner,
  row,
} from './algoselect';
import { closedForm, lg, span, stepsWithUnits, type Charge } from './prefixnet';
import { coarsenedSteps, coarsenedWork } from './coarsescan';

const NS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];

describe("chapter 11's numbers, through this model", () => {
  it('512 elements: Kogge-Stone does 4097 additions, the loop 511', () => {
    expect(row('kogge-stone', 512, 512, 'adds', 32).additions).toBe(4097);
    expect(closedForm('sequential', 512)).toBe(511);
    expect(row('brent-kung', 512, 512, 'adds', 32).additions).toBe(2 * 512 - 2 - 9);
  });
  it("1024 elements on 32 units: 320 steps by the book's Kogge-Stone estimate (every thread charged)", () => {
    expect(row('kogge-stone', 1024, 32, 'threads', 32).time).toBe(320);
  });
  it('spans with unlimited units: log2N and 2·log2N - 1', () => {
    expect(row('kogge-stone', 1024, 1, 'adds', 32).span).toBe(10);
    expect(row('brent-kung', 1024, 1, 'adds', 32).span).toBe(19);
  });
});

describe('q-scan-units', () => {
  it('N = 1024, P = 32, additions only: Brent-Kung 71, Kogge-Stone 289', () => {
    const r = compare(1024, 32, 'adds', 32);
    expect(r['brent-kung'].time).toBe(71);
    expect(r['kogge-stone'].time).toBe(289);
    expect(networkWinner(r)).toBe('brent-kung');
  });
  it('the same under whole-warp charging with a warp of 32', () => {
    const r = compare(1024, 32, 'warps', 32);
    expect([r['brent-kung'].time, r['kogge-stone'].time]).toEqual([71, 289]);
  });
  it('P = 1024: Kogge-Stone 10 against Brent-Kung 19', () => {
    const r = compare(1024, 1024, 'adds', 32);
    expect([r['kogge-stone'].time, r['brent-kung'].time]).toEqual([10, 19]);
    expect(networkWinner(r)).toBe('kogge-stone');
  });
  it('the networks come straight from prefixnet', () => {
    for (const c of ['adds', 'warps', 'threads'] as Charge[])
      for (const p of [1, 3, 32, 100, 1024]) {
        const r = compare(1024, p, c, 64);
        expect(r['kogge-stone'].time).toBe(stepsWithUnits('kogge-stone', 1024, p, c, 64));
        expect(r['brent-kung'].time).toBe(stepsWithUnits('brent-kung', 1024, p, c, 64));
      }
  });
});

describe('where the winner changes, counting additions', () => {
  it('N = 1024: Brent-Kung below 512, a tie from 512, Kogge-Stone from 768', () => {
    expect(crossovers(1024, 'adds', 32)).toEqual([
      { p: 512, from: 'brent-kung', to: 'tie' },
      { p: 768, from: 'tie', to: 'kogge-stone' },
    ]);
    expect(brentKungUpTo(1024, 'adds', 32)).toBe(511);
    expect(koggeStoneFrom(1024, 'adds', 32)).toBe(768);
  });
  it('for every N in the explorer: Brent-Kung for P < N/2, Kogge-Stone from 3N/4', () => {
    for (const n of NS) {
      expect(brentKungUpTo(n, 'adds', 32)).toBe(n / 2 - 1);
      expect(koggeStoneFrom(n, 'adds', 32)).toBe((3 * n) / 4);
    }
  });
  it('between N/2 and 3N/4 both take 2·log2N - 1 time units', () => {
    const r = compare(4096, 2500, 'adds', 32);
    expect(r['kogge-stone'].time).toBe(2 * 12 - 1);
    expect(r['brent-kung'].time).toBe(2 * 12 - 1);
    expect(networkWinner(r)).toBe('tie');
  });
});

describe('the coarsened three-phase scan', () => {
  it('runs one thread per unit, a power of two at most N', () => {
    expect(coarsenedThreads(1024, 32)).toBe(32);
    expect(coarsenedThreads(1024, 100)).toBe(64);
    expect(coarsenedThreads(1024, 5000)).toBe(1024);
    expect(coarsenedThreads(1024, 1)).toBe(1);
  });
  it("takes chapter 11's counts for T threads", () => {
    for (const p of [2, 32, 64, 1024]) {
      const r = row('coarsened', 1024, p, 'adds', 32);
      expect(r.time).toBe(coarsenedSteps(1024, p).total);
      expect(r.additions).toBe(coarsenedWork(1024, p).total);
    }
    expect(compare(1024, 32, 'adds', 32).coarsened.time).toBe(68);
  });
  it('on one unit it is the sequential loop', () => {
    const r = row('coarsened', 1024, 1, 'adds', 32);
    expect(r.time).toBe(span('sequential', 1024));
    expect(r.additions).toBe(closedForm('sequential', 1024));
  });
  it('its middle phase costs one time unit per Kogge-Stone step under every rule once T is at least a warp', () => {
    for (const warp of [32, 64])
      for (const t of [64, 128, 1024])
        for (const c of ['adds', 'warps', 'threads'] as Charge[]) expect(stepsWithUnits('kogge-stone', t, t, c, warp)).toBe(lg(t));
  });
  it('stays within (log2N + 1)/log2N of the better network at every power-of-two P', () => {
    for (const n of NS)
      for (const c of ['adds', 'warps'] as Charge[])
        for (let p = 1; p <= n; p *= 2) expect(hybridRatio(compare(n, p, c, 32))).toBeLessThanOrEqual((lg(n) + 1) / lg(n) + 1e-12);
  });
  it('is never more than one time unit behind the better network at a power-of-two P', () => {
    for (const n of NS)
      for (const c of ['adds', 'warps'] as Charge[])
        for (const w of [32, 64]) for (let p = 1; p <= n; p *= 2) expect(hybridLag(compare(n, p, c, w))).toBeLessThanOrEqual(1);
  });
  it('is the fastest of the three at N = 1024 on 32 units', () => {
    expect(overallWinner(compare(1024, 32, 'adds', 32))).toBe('coarsened');
  });
});

describe('the plot', () => {
  it('runs over every power of two from P = 1 to N and agrees with compare', () => {
    const pts = curve(1024, 'adds', 32);
    expect(pts.map((x) => x.p)).toEqual([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]);
    for (const pt of pts) {
      const r = compare(1024, pt.p, 'adds', 32);
      for (const a of ALGOS) expect(pt[a]).toBe(r[a].time);
    }
  });
  it('times never grow with P', () => {
    for (const a of ALGOS) {
      const pts = curve(4096, 'warps', 64);
      for (let i = 1; i < pts.length; i++) expect(pts[i][a]).toBeLessThanOrEqual(pts[i - 1][a]);
    }
  });
});

describe('input checks', () => {
  it('rejects N that is not a power of two', () => {
    expect(() => row('kogge-stone', 1000, 4, 'adds', 32)).toThrow();
  });
});
