import { describe, expect, it } from 'vitest';
import { flagsAfter, frontMakespan, frontOrder, frontSizes, numbering, rowsOfRunning, schedule, slotAt } from './wavesched';

// Pins Fig. 16.21 (p. 399) and every number section 16.3, its scene and its questions state
// about who waits for whom.

describe('Fig. 16.21', () => {
  it('numbers a 4 x 4 grid of tiles in front order as the figure does', () => {
    expect(numbering(4)).toEqual([
      [0, 1, 3, 6],
      [2, 4, 7, 10],
      [5, 8, 11, 13],
      [9, 12, 14, 15],
    ]);
    expect(frontOrder(4)[4]).toEqual([1, 1]);
    expect(frontSizes(4)).toEqual([1, 2, 3, 4, 3, 2, 1]);
  });
  it('2 SMs, a relaunch per front: 10 steps, the figure\'s left schedule', () => {
    const f = schedule(4, 2, 'front');
    expect(rowsOfRunning(f)).toEqual([[0], [1, 2], [3, 4], [5], [6, 7], [8, 9], [10, 11], [12], [13, 14], [15]]);
    expect(f.makespan).toBe(10);
    expect(f.idle).toBe(4);
    expect(f.barriers).toEqual([0, 1, 3, 5, 7, 8, 9]);
  });
  it('2 SMs, tiles from a counter, each waiting on two flags: 9 steps, the figure\'s right schedule (p-flags-2sm)', () => {
    const c = schedule(4, 2, 'counter');
    expect(rowsOfRunning(c)).toEqual([[0], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15]]);
    expect(c.makespan).toBe(9);
    expect(c.idle).toBe(2);
    // the slot that claims tile 1 spins in step 0, waiting for tile 0's flag
    expect(c.timeline[0]).toEqual([{ state: 'run', tile: 0 }, { state: 'spin', tile: 1 }]);
    // tile 6 runs in step 3 (the fourth), in the slot that the relaunch left empty
    expect(c.doneAt[6]).toBe(3);
    expect(schedule(4, 2, 'front').doneAt[6]).toBe(4);
  });
});

describe('the rules against each other', () => {
  it('a relaunch per front takes the sum over fronts of ceil(size / S)', () => {
    for (let T = 1; T <= 12; T++)
      for (let S = 1; S <= 12; S++) {
        expect(schedule(T, S, 'front').makespan).toBe(frontMakespan(T, S));
        expect(frontMakespan(T, S)).toBe(frontSizes(T).reduce((a, n) => a + Math.ceil(n / S), 0));
      }
  });
  it('tiles from a counter are never slower than a relaunch per front, for T and S up to 12, and never stuck', () => {
    for (let T = 1; T <= 12; T++)
      for (let S = 1; S <= 12; S++) {
        const c = schedule(T, S, 'counter');
        expect(c.stuck).toBe(false);
        expect(c.makespan!).toBeLessThanOrEqual(frontMakespan(T, S));
        expect(c.makespan!).toBeGreaterThanOrEqual(2 * T - 1);
      }
  });
  it('with S >= T both take 2T - 1 steps: nothing to gain (q-enough-sms)', () => {
    for (let T = 1; T <= 10; T++)
      for (let S = T; S <= T + 3; S++) {
        expect(schedule(T, S, 'front').makespan).toBe(2 * T - 1);
        expect(schedule(T, S, 'counter').makespan).toBe(2 * T - 1);
      }
    expect(schedule(4, 4, 'front').makespan).toBe(7);
    expect(schedule(4, 4, 'counter').makespan).toBe(7);
  });
  it('with S = 1 both take T squared steps', () => {
    for (let T = 1; T <= 10; T++) {
      expect(schedule(T, 1, 'front').makespan).toBe(T * T);
      expect(schedule(T, 1, 'counter').makespan).toBe(T * T);
    }
  });
  it('the counter gains somewhere between those extremes, for the explorer\'s sizes', () => {
    expect(schedule(8, 3, 'counter').makespan!).toBeLessThan(frontMakespan(8, 3));
  });
});

describe("the book's persistent block per tile row", () => {
  it('with fewer slots than rows and blocks started in reverse order, it is stuck', () => {
    for (let T = 2; T <= 8; T++)
      for (let S = 1; S < T; S++) {
        const r = schedule(T, S, 'rows', 'reversed');
        expect(r.stuck).toBe(true);
        expect(r.makespan).toBeNull();
        expect(r.stuckAt).toBe(0);
      }
    const r = schedule(4, 2, 'rows', 'reversed');
    // block 3 waits for row 2, block 2 for row 1, which never gets a slot
    expect(r.timeline[0]).toEqual([{ state: 'spin', tile: 9, block: 3 }, { state: 'spin', tile: 5, block: 2 }]);
    expect(slotAt(r, 50, 0)).toEqual(r.timeline[0][0]);
  });
  it('with blocks started in index order it completes, as fast as the counter on 2 SMs for 4 x 4', () => {
    for (let T = 1; T <= 8; T++)
      for (let S = 1; S <= 8; S++) expect(schedule(T, S, 'rows', 'index').stuck).toBe(false);
    expect(schedule(4, 2, 'rows', 'index').makespan).toBe(9);
  });
  it('with a slot for every row, either start order completes', () => {
    for (let T = 1; T <= 8; T++) expect(schedule(T, T, 'rows', 'reversed').stuck).toBe(false);
  });
});

describe('flags and the timeline', () => {
  it('the flags set after each step of the counter rule, for the scene', () => {
    const c = schedule(4, 2, 'counter');
    expect([...flagsAfter(c, 0)]).toEqual([]);
    expect([...flagsAfter(c, 1)]).toEqual([0]);
    expect([...flagsAfter(c, 4)].sort((a, b) => a - b)).toEqual([0, 1, 2, 3, 4, 5, 6]);
    expect(flagsAfter(c, 9).size).toBe(16);
  });
  it('no tile ever runs before both neighbours are done, under any rule', () => {
    for (const T of [2, 3, 5, 8])
      for (const S of [1, 2, 3, 8])
        for (const rule of ['front', 'counter', 'rows'] as const) {
          const sch = schedule(T, S, rule);
          const num = numbering(T);
          for (let r = 0; r < T; r++)
            for (let c = 0; c < T; c++) {
              const t = sch.doneAt[num[r][c]];
              if (r > 0) expect(sch.doneAt[num[r - 1][c]]).toBeLessThan(t);
              if (c > 0) expect(sch.doneAt[num[r][c - 1]]).toBeLessThan(t);
            }
        }
  });
  it('idle slot-steps plus tiles fill the timeline', () => {
    for (const rule of ['front', 'counter', 'rows'] as const) {
      const sch = schedule(6, 4, rule);
      expect(sch.idle + 36).toBe(sch.makespan! * 4);
    }
  });
  it('rejects nonsense sizes', () => {
    expect(() => schedule(0, 2, 'front')).toThrow();
    expect(() => schedule(4, 0, 'counter')).toThrow();
  });
});
