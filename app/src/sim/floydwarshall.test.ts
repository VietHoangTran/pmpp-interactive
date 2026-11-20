import { describe, expect, it } from 'vitest';
import { FIG_16_5, INF, VERTICES, floydWarshallCPU, randomGraph, runSteps, stateAfter, stepTrace, type Order } from './floydwarshall';

// Pins the numbers section 16.3 states about the Floyd-Warshall stepper and its questions.
const v = (name: string) => VERTICES.indexOf(name as (typeof VERTICES)[number]);
const cells = (t: ReturnType<typeof stepTrace>) => t.changed.map((c) => `${VERTICES[c.i]}${VERTICES[c.j]}=${c.to}`).sort();
const ORDERS: Order[] = ['top-first', 'bottom-first'];

describe("Fig. 16.5's table as printed", () => {
  it('has the printed entries, the b to f misprint included, and FW_INF elsewhere', () => {
    expect(INF).toBe(1073741823);
    expect(INF + INF).toBeLessThanOrEqual(2147483647);
    const row = (u: string) => VERTICES.map((w, j) => (FIG_16_5[v(u)][j] === INF ? '' : `${w}${FIG_16_5[v(u)][j]}`)).filter(Boolean).join(' ');
    expect(row('a')).toBe('a0 b4 g7 h4');
    expect(row('b')).toBe('b0 c9 f9 g8 h1');
    expect(row('c')).toBe('c0 e10');
    expect(row('d')).toBe('d0');
    expect(row('e')).toBe('c8 d6 e0 f5');
    expect(row('f')).toBe('e6 f0');
    expect(row('g')).toBe('b4 f7 g0');
    expect(row('h')).toBe('c3 h0');
  });
});

describe('the steps the lesson walks through', () => {
  it('k = a retires 7 of the 8 blocks and changes nothing', () => {
    const t = stepTrace(FIG_16_5, v('a'));
    expect(t.retired).toHaveLength(7);
    expect(t.retired).not.toContain(v('a'));
    expect(t.changed).toEqual([]);
  });
  it('k = b, after a, changes a to c, a to f, g to c and g to h', () => {
    const t = stepTrace(runSteps(FIG_16_5, 1), v('b'));
    expect(cells(t)).toEqual(['ac=13', 'af=13', 'gc=13', 'gh=5']);
  });
  it('k = c, after a and b, changes exactly a to e 23, b to e 19, g to e 23, h to e 13, and retires rows d and f (q-fw-changes)', () => {
    const before = runSteps(FIG_16_5, 2);
    // column c and row c as the question states them
    expect(VERTICES.map((w, i) => (before[i][v('c')] === INF ? '' : `${w}${before[i][v('c')]}`)).filter(Boolean)).toEqual(['a13', 'b9', 'c0', 'e8', 'g13', 'h3']);
    expect(VERTICES.map((w, j) => (before[v('c')][j] === INF ? '' : `${w}${before[v('c')][j]}`)).filter(Boolean)).toEqual(['c0', 'e10']);
    const t = stepTrace(before, v('c'));
    expect(cells(t)).toEqual(['ae=23', 'be=19', 'ge=23', 'he=13']);
    expect(t.retired).toEqual([v('d'), v('f')]);
  });
  it('applied straight to the printed table, k = c changes only b to e and h to e (the notes\' derived line)', () => {
    expect(cells(stepTrace(FIG_16_5, v('c')))).toEqual(['be=19', 'he=13']);
  });
  it('blocks retire at k = a, b, ..., h in these numbers', () => {
    const counts = VERTICES.map((_, k) => stepTrace(runSteps(FIG_16_5, k), k).retired.length);
    expect(counts).toEqual([7, 5, 2, 6, 1, 1, 5, 4]);
  });
});

describe('why one table is enough', () => {
  it('with dist[k][k] = 0 and non-negative weights, row k and column k never change, and in place equals two buffers, on the example', () => {
    for (let k = 0; k < 8; k++) {
      const d = runSteps(FIG_16_5, k);
      const two = stepTrace(d, k, { mode: 'two-buffers' });
      expect(two.rowKChanged || two.colKChanged).toBe(false);
      for (const order of ORDERS) expect(stepTrace(d, k, { order }).after).toEqual(two.after);
    }
  });
  it('the same holds on random graphs, for every k and both orders', () => {
    for (let seed = 1; seed <= 40; seed++) {
      let d = randomGraph(6 + (seed % 7), seed);
      for (let k = 0; k < d.length; k++) {
        const two = stepTrace(d, k, { mode: 'two-buffers' });
        expect(two.rowKChanged || two.colKChanged).toBe(false);
        for (const order of ORDERS) expect(stepTrace(d, k, { order }).after).toEqual(two.after);
        d = two.after;
      }
    }
  });
  it('with dist[k][k] < 0, row k and column k change, and at least one order differs from two buffers', () => {
    const d = runSteps(FIG_16_5, 2);
    const two = stepTrace(d, v('c'), { mode: 'two-buffers', diag: -1 });
    expect(two.rowKChanged && two.colKChanged).toBe(true);
    const differs = ORDERS.filter((order) => JSON.stringify(stepTrace(d, v('c'), { order, diag: -1 }).after) !== JSON.stringify(two.after));
    expect(differs.length).toBeGreaterThanOrEqual(1);
    // and the two orders disagree with each other: the result depends on who runs first
    expect(stepTrace(d, v('c'), { order: 'top-first', diag: -1 }).after).not.toEqual(stepTrace(d, v('c'), { order: 'bottom-first', diag: -1 }).after);
  });
  it('the whole run equals a sequential CPU Floyd-Warshall, on the example and on random graphs', () => {
    expect(runSteps(FIG_16_5, 8)).toEqual(floydWarshallCPU(FIG_16_5));
    for (let seed = 100; seed < 130; seed++) {
      const g = randomGraph(10, seed, 0.25);
      expect(runSteps(g, 10)).toEqual(floydWarshallCPU(g));
    }
  });
});

describe('the block trace', () => {
  it('replaying m blocks gives the intermediate tables, and all of them the final one', () => {
    const t = stepTrace(runSteps(FIG_16_5, 2), v('c'), { order: 'bottom-first' });
    expect(stateAfter(t, 0)).toEqual(t.before);
    expect(stateAfter(t, 8)).toEqual(t.after);
    expect(t.blocks.map((b) => b.row)).toEqual([7, 6, 5, 4, 3, 2, 1, 0]);
    // the h block (first, bottom-first) already holds its write after one block
    expect(stateAfter(t, 1)[v('h')][v('e')]).toBe(13);
    expect(stateAfter(t, 1)[v('a')][v('e')]).toBe(INF);
  });
  it('a retired block reads nothing of row k and writes nothing', () => {
    const t = stepTrace(FIG_16_5, v('a'));
    for (const b of t.blocks.filter((x) => x.retired)) {
      expect(b.distKCol).toBe(INF);
      expect(b.rowK).toEqual([]);
      expect(b.writes).toEqual([]);
    }
  });
});
