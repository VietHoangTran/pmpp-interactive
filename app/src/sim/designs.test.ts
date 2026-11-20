import { describe, expect, it } from 'vitest';
import { design, finish, floorplan, race, regions, snapshot, sweep, timeline, waves } from './designs';

const BUDGET = 64;
const A = design(BUDGET, 2); // 4 units, each 4 times faster
const B = design(BUDGET, 0); // 64 plain units

describe('what a budget buys (the book\'s cost rule, p. 5, applied 0 to 3 times)', () => {
  it('each halving of latency doubles the speed of a unit and quadruples its power', () => {
    for (let h = 0; h < 3; h++) {
      expect(design(BUDGET, h + 1).speed).toBe(2 * design(BUDGET, h).speed);
      expect(design(BUDGET, h + 1).unitPower).toBe(4 * design(BUDGET, h).unitPower);
    }
  });
  it('budget 64 buys 64, 16, 4 or 1 units and spends all of it', () => {
    expect([0, 1, 2, 3].map((h) => design(BUDGET, h).units)).toEqual([64, 16, 4, 1]);
    for (const h of [0, 1, 2, 3]) expect(design(BUDGET, h).units * design(BUDGET, h).unitPower).toBe(BUDGET);
  });
  it('never spends more than the budget, whatever the budget', () => {
    for (let budget = 1; budget <= 200; budget++)
      for (let h = 0; h <= 4; h++) {
        const d = design(budget, h);
        expect(d.units * d.unitPower).toBeLessThanOrEqual(budget);
        expect(budget - d.units * d.unitPower).toBeLessThan(d.unitPower);
      }
  });
  it('total throughput is budget / 2^halvings: each halving bought on a fixed budget halves it', () => {
    expect([0, 1, 2, 3].map((h) => design(BUDGET, h).throughput)).toEqual([64, 32, 16, 8]);
    for (const h of [0, 1, 2, 3]) expect(design(BUDGET, h).throughput).toBe(BUDGET / 2 ** h);
  });
});

describe('one chain', () => {
  it('the faster design wins by exactly the ratio of the speeds', () => {
    for (let ha = 0; ha <= 3; ha++)
      for (let hb = 0; hb <= 3; hb++) {
        const r = race(design(BUDGET, ha), design(BUDGET, hb), 1, 100);
        if (ha === hb) expect(r).toMatchObject({ winner: 'tie', factor: 1 });
        else expect(r).toMatchObject({ winner: ha > hb ? 'a' : 'b', factor: 2 ** Math.abs(ha - hb) });
      }
  });
  it('every unit but one sits idle', () => {
    expect(finish(B, 1, 16)).toMatchObject({ first: 16, last: 16, waves: 1, idle: 63, delivered: 1 });
    expect(finish(A, 1, 16)).toMatchObject({ first: 4, last: 4, waves: 1, idle: 3, delivered: 4 });
  });
});

describe('4 fast units against 64 plain ones', () => {
  it('1 chain: the fast units win by 4', () => {
    expect(race(A, B, 1, 100)).toMatchObject({ winner: 'a', factor: 4 });
  });
  it('12 chains: the fast units still lead', () => {
    expect(race(A, B, 12, 100).winner).toBe('a');
  });
  it('13 to 16 chains: a tie', () => {
    for (const n of [13, 14, 15, 16]) expect(race(A, B, n, 100)).toMatchObject({ winner: 'tie', factor: 1 });
  });
  it('17 chains: the many units lead', () => {
    expect(race(A, B, 17, 100)).toMatchObject({ winner: 'b', factor: 1.25 });
  });
  it('64 and 1,024 chains: the many units win by exactly 4', () => {
    expect(race(A, B, 64, 100)).toMatchObject({ winner: 'b', factor: 4 });
    expect(race(A, B, 1024, 100)).toMatchObject({ winner: 'b', factor: 4 });
  });
  it('the lead changes hands once: a for 1 to 12, tie for 13 to 16, b from 17', () => {
    expect(regions(sweep(A, B, 4096, 100))).toEqual([
      { winner: 'a', from: 1, to: 12 },
      { winner: 'tie', from: 13, to: 16 },
      { winner: 'b', from: 17, to: 4096 },
    ]);
  });
  it('1 unit of speed 8 against 64 plain ones, 1,024 chains: the many units win by 8', () => {
    expect(race(design(BUDGET, 3), B, 1024, 100)).toMatchObject({ winner: 'b', factor: 8 });
  });
  it('64 chains of 16 operations (the animation): first result at 4 and 16, last at 64 and 16', () => {
    expect(finish(A, 64, 16)).toMatchObject({ first: 4, last: 64, waves: 16, idle: 0, busy: 1 });
    expect(finish(B, 64, 16)).toMatchObject({ first: 16, last: 16, waves: 1, idle: 0, busy: 1 });
  });
  it('an 8 by 8 matrix product as 64 chains of 8: done at 8 on the many units, at 32 on the fast ones', () => {
    expect(finish(B, 64, 8).last).toBe(8);
    expect(finish(A, 64, 8).last).toBe(32);
  });
});

describe('chain length', () => {
  it('multiplying the length by any factor leaves winner and factor unchanged', () => {
    for (let ha = 0; ha <= 3; ha++)
      for (let hb = 0; hb <= 3; hb++)
        for (const n of [1, 2, 7, 12, 13, 16, 17, 63, 64, 65, 80, 1000, 1024, 4096]) {
          const base = race(design(BUDGET, ha), design(BUDGET, hb), n, 1);
          for (const k of [2, 3, 10, 17, 1000, 0.3, 7.77]) {
            const r = race(design(BUDGET, ha), design(BUDGET, hb), n, k);
            expect([r.winner, r.factor]).toEqual([base.winner, base.factor]);
          }
        }
  });
  it('the factor race() reports is the ratio of the two finishing times', () => {
    for (const n of [1, 5, 12, 13, 17, 64, 80, 1024])
      for (const len of [1, 16, 100, 1000]) {
        const r = race(A, B, n, len);
        const [fast, slow] = r.a.last <= r.b.last ? [r.a.last, r.b.last] : [r.b.last, r.a.last];
        expect(slow / fast).toBe(r.factor);
        expect(r.winner).toBe(r.a.last < r.b.last ? 'a' : r.a.last > r.b.last ? 'b' : 'tie');
      }
  });
  it('both finishing times scale with the length', () => {
    expect(finish(A, 80, 1000).last).toBe(10 * finish(A, 80, 100).last);
    expect(finish(B, 80, 1000).last).toBe(10 * finish(B, 80, 100).last);
  });
});

describe('throughput arrives in steps', () => {
  it('80 chains of 100 on 64 units of speed 1 finish at 200: two waves', () => {
    expect(finish(B, 80, 100)).toMatchObject({ waves: 2, first: 100, last: 200, delivered: 40 });
  });
  it('a 65th unit changes nothing', () => {
    expect(finish(design(65, 0), 80, 100).last).toBe(200);
  });
  it('16 chains fewer halves the time', () => {
    expect(finish(B, 64, 100).last).toBe(100);
  });
});

describe('timeline', () => {
  const cases: [number, number, number][] = [
    [0, 1, 16], [0, 64, 16], [0, 80, 100], [1, 40, 7], [2, 1, 16], [2, 17, 100], [2, 64, 16], [3, 9, 5],
  ];
  it('has one segment per chain', () => {
    for (const [h, n, len] of cases) {
      const t = timeline(design(BUDGET, h), n, len);
      expect(t.length).toBe(n);
      expect(new Set(t.map((s) => s.chain)).size).toBe(n);
    }
  });
  it('segments never overlap on one unit, and each lasts length / speed', () => {
    for (const [h, n, len] of cases) {
      const d = design(BUDGET, h);
      const t = timeline(d, n, len);
      for (const s of t) {
        expect(s.unit).toBeLessThan(d.units);
        expect(s.end - s.start).toBe(len / d.speed);
      }
      for (let u = 0; u < d.units; u++) {
        const mine = t.filter((s) => s.unit === u).sort((x, y) => x.start - y.start);
        for (let i = 1; i < mine.length; i++) expect(mine[i].start).toBeGreaterThanOrEqual(mine[i - 1].end);
      }
    }
  });
  it('agrees with finish(): earliest end is the first result, latest end the last', () => {
    for (const [h, n, len] of cases) {
      const d = design(BUDGET, h);
      const t = timeline(d, n, len);
      const f = finish(d, n, len);
      expect(Math.min(...t.map((s) => s.end))).toBe(f.first);
      expect(Math.max(...t.map((s) => s.end))).toBe(f.last);
    }
  });
  it('waves() is the same schedule, one entry per wave', () => {
    for (const [h, n, len] of cases) {
      const d = design(BUDGET, h);
      const w = waves(d, n, len);
      expect(w.length).toBe(finish(d, n, len).waves);
      expect(w.reduce((s, x) => s + x.units, 0)).toBe(n);
      for (const s of timeline(d, n, len)) expect([w[s.wave].start, w[s.wave].end]).toEqual([s.start, s.end]);
    }
  });
});

describe('snapshot', () => {
  it('one chain of 16: the fast unit has done 8 operations at time 2 and all 16 at time 4', () => {
    expect(snapshot(A, 1, 16, 2)).toEqual({ ops: [8], done: 0, running: 1 });
    expect(snapshot(A, 1, 16, 4)).toEqual({ ops: [16], done: 1, running: 0 });
    expect(snapshot(B, 1, 16, 4)).toEqual({ ops: [4], done: 0, running: 1 });
  });
  it('64 chains of 16 on 4 fast units: 4 chains done at time 4, 32 at time 32, all at time 64', () => {
    expect(snapshot(A, 64, 16, 4).done).toBe(4);
    expect(snapshot(A, 64, 16, 32).done).toBe(32);
    expect(snapshot(A, 64, 16, 64).done).toBe(64);
    expect(snapshot(A, 64, 16, 5).running).toBe(4);
  });
  it('64 chains of 16 on 64 plain units: none done before 16, all done at 16', () => {
    expect(snapshot(B, 64, 16, 15.9).done).toBe(0);
    expect(snapshot(B, 64, 16, 15.9).running).toBe(64);
    expect(snapshot(B, 64, 16, 16).done).toBe(64);
  });
});

describe('floor plan', () => {
  it('a unit is drawn as a square whose area is its power, and the units tile the budget', () => {
    for (const h of [0, 1, 2, 3]) {
      const d = design(BUDGET, h);
      const p = floorplan(d);
      expect(p.side).toBe(8);
      expect(p.cells.length).toBe(d.units);
      const seen = new Set<string>();
      for (const c of p.cells) {
        expect(c.side * c.side).toBe(d.unitPower);
        for (let x = c.x; x < c.x + c.side; x++)
          for (let y = c.y; y < c.y + c.side; y++) {
            expect(x).toBeLessThan(p.side);
            expect(y).toBeLessThan(p.side);
            expect(seen.has(x + ',' + y)).toBe(false);
            seen.add(x + ',' + y);
          }
      }
      expect(seen.size).toBe(BUDGET);
    }
  });
});
