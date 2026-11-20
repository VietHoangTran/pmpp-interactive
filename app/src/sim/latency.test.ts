import { describe, expect, it } from 'vitest';
import {
  BOOK_OVERSUBSCRIPTION, PRESETS, busyFraction, estimate, idleCycles, measureWindow, normalise, period, runs,
  simulate, simulateSwitching, tally, unitShares, utilisation, warpsNeeded, type LatencyConfig,
} from './latency';

const cfg = (warps: number, work: number, latency: number, slots = 1): LatencyConfig => ({ warps, work, latency, slots });

describe('one warp alone', () => {
  it('is busy for c of every c + L cycles', () => {
    for (const [c, L] of [[2, 30], [4, 60], [1, 1], [30, 30], [7, 13]] as const) {
      expect(utilisation(cfg(1, c, L))).toBeCloseTo(c / (c + L), 12);
      expect(estimate(cfg(1, c, L))).toBeCloseTo(c / (c + L), 12);
    }
  });
  it('computes, then waits, then computes again, from a cold start', () => {
    const tl = simulate(cfg(1, 2, 5), 14);
    expect(tl.rows[0].join(' ')).toBe('exec exec wait wait wait wait wait exec exec wait wait wait wait wait');
    expect(idleCycles(tl)).toBe(10);
  });
});

describe('the numbers the lesson states', () => {
  it('2 cycles of work and a 30-cycle wait need 16 warps', () => {
    expect(warpsNeeded(2, 30)).toBe(16);
    expect(utilisation(cfg(16, 2, 30))).toBe(1);
    expect(estimate(cfg(16, 2, 30))).toBe(1);
    expect(utilisation(cfg(15, 2, 30))).toBeCloseTo(15 / 16, 12);
    // From a cold start the unit never idles at all: the last warp starts its wait as the first one's ends.
    expect(idleCycles(simulate(cfg(16, 2, 30), 320))).toBe(0);
    expect(idleCycles(simulate(cfg(15, 2, 30), 320))).toBe(20);
  });
  it('8 warps, 4 cycles of work, 60 cycles of latency: 50%', () => {
    expect(utilisation(cfg(8, 4, 60))).toBe(0.5);
    expect(estimate(cfg(8, 4, 60))).toBe(0.5);
    expect(warpsNeeded(4, 60)).toBe(16);
  });
  it('the two presets need 16 warps and 2 warps', () => {
    expect(warpsNeeded(PRESETS.loadHeavy.work, PRESETS.loadHeavy.latency)).toBe(16);
    expect(warpsNeeded(PRESETS.computeHeavy.work, PRESETS.computeHeavy.latency)).toBe(2);
    expect(utilisation({ warps: 2, slots: 1, ...PRESETS.computeHeavy })).toBe(1);
    expect(utilisation({ warps: 2, slots: 1, ...PRESETS.loadHeavy })).toBeCloseTo(0.125, 12);
  });
  it("the book's example device holds 16 threads for each streaming processor (pp. 84 to 85)", () => {
    expect(BOOK_OVERSUBSCRIPTION.threadsPerSm / BOOK_OVERSUBSCRIPTION.coresPerSm).toBe(BOOK_OVERSUBSCRIPTION.ratio);
    expect(BOOK_OVERSUBSCRIPTION.ratio).toBe(16);
  });
});

describe('more warps than needed change nothing', () => {
  it('utilisation stays at 100% and the extra warps only queue', () => {
    for (const R of [16, 17, 24, 32, 64]) expect(utilisation(cfg(R, 2, 30))).toBe(1);
    const w = measureWindow(cfg(32, 2, 30));
    const tl = simulate(cfg(32, 2, 30), w.to);
    const t = tally(tl.rows[5], w.from, w.to);
    expect(t.ready).toBeGreaterThan(0); // a warp whose data is back still has to wait its turn
    expect(t.exec).toBe(((w.to - w.from) / 64) * 2); // one burst of 2 for every 32 * 2 cycles
  });
  it('below the threshold every warp adds its full share', () => {
    for (let R = 1; R <= 16; R++) expect(utilisation(cfg(R, 2, 30))).toBeCloseTo(R / 16, 12);
  });
  it('a warp that never has to queue is never ready-but-not-chosen once the start-up has passed', () => {
    const c = cfg(8, 4, 60);
    const w = measureWindow(c);
    const tl = simulate(c, w.to);
    for (const row of tl.rows) expect(tally(row, w.from, w.to).ready).toBe(0);
  });
});

describe('the simulated timeline agrees with the estimate', () => {
  it('for one issue slot, across the explorer range', () => {
    for (const R of [1, 2, 3, 5, 8, 13, 16, 17, 31, 32, 33, 48, 64])
      for (const c of [1, 2, 3, 4, 7, 16, 30, 32])
        for (const L of [1, 3, 8, 30, 60, 100, 120])
          expect(utilisation(cfg(R, c, L)), `R=${R} c=${c} L=${L}`).toBeCloseTo(estimate(cfg(R, c, L)), 12);
  });
  it('for two, three and four issue slots', () => {
    for (const slots of [2, 3, 4])
      for (const R of [1, 2, 3, 5, 8, 16, 17, 32, 64])
        for (const c of [1, 2, 4, 7, 30])
          for (const L of [1, 8, 30, 60, 120])
            expect(utilisation(cfg(R, c, L, slots)), `S=${slots} R=${R} c=${c} L=${L}`).toBeCloseTo(estimate(cfg(R, c, L, slots)), 12);
  });
  it('more slots need proportionally more warps', () => {
    expect(warpsNeeded(2, 30, 4)).toBe(64);
    expect(utilisation(cfg(64, 2, 30, 4))).toBe(1);
    expect(utilisation(cfg(16, 2, 30, 4))).toBe(0.25);
    expect(warpsNeeded(3, 10)).toBe(5); // 13 / 3 rounded up
  });
});

describe('the scheduler obeys its own rules', () => {
  const cases = [cfg(3, 2, 9), cfg(16, 2, 30), cfg(40, 3, 17, 2), cfg(7, 5, 4, 3), cfg(64, 1, 120, 4)];
  it('never issues from a waiting warp, and a warp waits exactly L cycles after its c-th instruction', () => {
    for (const c of cases) {
      const tl = simulate(c, 600);
      for (const row of tl.rows) {
        const rs = runs(row);
        let burst = 0;
        for (let i = 0; i < rs.length; i++) {
          const r = rs[i];
          if (r.state === 'exec') burst += r.length;
          if (r.state === 'wait') {
            expect(burst).toBe(c.work); // a wait starts only after a whole burst
            burst = 0;
            if (r.start + r.length < tl.cycles) expect(r.length).toBe(c.latency);
            expect(rs[i - 1].state).toBe('exec');
          }
        }
      }
    }
  });
  it('issues from at most `slots` warps a cycle, each at most once, and never leaves a slot empty while a warp is ready', () => {
    for (const c of cases) {
      const tl = simulate(c, 600);
      for (let t = 0; t < tl.cycles; t++) {
        const pick = tl.chosen[t];
        expect(pick.length).toBeLessThanOrEqual(c.slots);
        expect(new Set(pick).size).toBe(pick.length);
        const ready = tl.rows.filter((row) => row[t] === 'ready').length;
        if (ready > 0) expect(pick.length).toBe(c.slots);
        for (const w of pick) expect(tl.rows[w][t]).toBe('exec');
        expect(tl.rows.filter((row) => row[t] === 'exec').length).toBe(pick.length);
      }
    }
  });
  it('picks the warp that has been ready longest, lowest number first on a tie', () => {
    const tl = simulate(cfg(3, 2, 9), 12);
    expect(tl.chosen.map((p) => (p.length ? p[0] : '-')).join('')).toBe('001122-----0');
  });
});

describe('helpers', () => {
  it('normalise makes every field a usable whole number', () => {
    expect(normalise({ warps: 0, work: 0.4, latency: -3, slots: NaN })).toEqual({ warps: 1, work: 1, latency: 0, slots: 1 });
    expect(period(cfg(1, 2, 30))).toBe(32);
  });
  it('with no latency one warp is enough', () => {
    expect(warpsNeeded(5, 0)).toBe(1);
    expect(utilisation(cfg(1, 5, 0))).toBe(1);
  });
  it('busyFraction of an empty range is 0', () => {
    expect(busyFraction(simulate(cfg(1, 1, 1), 4), 3, 3)).toBe(0);
  });
  it('runs compresses a row and keeps every cycle', () => {
    const tl = simulate(cfg(2, 2, 3), 20);
    for (const row of tl.rows) {
      const rs = runs(row);
      expect(rs.reduce((s, r) => s + r.length, 0)).toBe(20);
      for (let i = 1; i < rs.length; i++) expect(rs[i].state).not.toBe(rs[i - 1].state);
    }
  });
});

describe('a switch that costs something (the sidebar on p. 84)', () => {
  it('with no cost it issues exactly what the free scheduler issues', () => {
    for (const c of [cfg(1, 2, 6), cfg(3, 2, 6), cfg(4, 2, 6), cfg(9, 3, 7), cfg(16, 2, 30)]) {
      const free = simulate(c, 200).chosen.map((p) => (p.length ? p[0] : null));
      const unit = simulateSwitching(c, 200, 0);
      expect(unit.map((u) => u.warp)).toEqual(free);
      expect(unit.some((u) => u.kind === 'switch')).toBe(false);
    }
  });
  it('4 warps, 2 cycles of work, 6 of latency: free switching keeps the unit busy; a 2-cycle switch halves it', () => {
    expect(unitShares(simulateSwitching(cfg(4, 2, 6), 400, 0), 40, 400)).toEqual({ exec: 1, switch: 0, idle: 0 });
    expect(unitShares(simulateSwitching(cfg(4, 2, 6), 400, 2), 40, 400)).toEqual({ exec: 0.5, switch: 0.5, idle: 0 });
  });
  it('the first cycles alternate work and switching', () => {
    const unit = simulateSwitching(cfg(4, 2, 6), 12, 2);
    expect(unit.map((u) => (u.kind === 'exec' ? String(u.warp) : u.kind === 'switch' ? 's' : '-')).join('')).toBe('00ss11ss22ss');
  });
  it('one warp alone never pays for a switch', () => {
    expect(simulateSwitching(cfg(1, 2, 6), 80, 5).some((u) => u.kind === 'switch')).toBe(false);
  });
});
