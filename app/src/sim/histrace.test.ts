import { describe, expect, it } from 'vitest';
import { PRESETS, atomicOrder, enumerate, interleavingCount, nextOp, progressOf, run } from './histrace';

const preset = (id: string) => PRESETS.find((p) => p.id === id)!;
const shown = (id: string, n = 2) => run(preset(id).schedule(n), { n }).steps.map((s) => s.shown);

describe('the book\'s interleaving tables (Figs. 9.4, 9.5, pp. 205-206)', () => {
  it('thread 1 first: (0) (1) (1) then (1) (2) (2), final 2', () => {
    expect(shown('t1-first')).toEqual([0, 1, 1, 1, 2, 2]);
    expect(run(preset('t1-first').schedule(2), { n: 2 }).final).toBe(2);
  });
  it('overlapped with thread 1 ahead: (0) (1) (0) (1) (1) (1), final 1, thread 1\'s update overwritten', () => {
    const r = run(preset('overlap-t1').schedule(2), { n: 2 });
    expect(r.steps.map((s) => s.shown)).toEqual([0, 1, 0, 1, 1, 1]);
    expect(r.steps.map((s) => s.thread)).toEqual([0, 0, 1, 0, 1, 1]);
    expect(r.final).toBe(1);
    expect(r.lost).toEqual([0]);
    expect(r.steps[5].erased).toEqual([0]);
  });
  it('thread 2 first mirrors thread 1 first', () => {
    expect(shown('t2-first')).toEqual([0, 1, 1, 1, 2, 2]);
    expect(run(preset('t2-first').schedule(2), { n: 2 }).correct).toBe(true);
  });
  it('overlapped with thread 2 ahead ends at 1 and loses thread 2\'s update', () => {
    const r = run(preset('overlap-t2').schedule(2), { n: 2 });
    expect(r.final).toBe(1);
    expect(r.lost).toEqual([1]);
  });
  it('the presets in order end at 2, 1, 2, 1, and lockstep at 1', () => {
    expect(PRESETS.map((p) => run(p.schedule(2), { n: 2 }).final)).toEqual([2, 1, 2, 1, 1]);
  });
  it('a lockstep warp keeps one increment of n, for n = 2, 3, 4', () => {
    for (const n of [2, 3, 4]) {
      const r = run(preset('lockstep').schedule(n), { n });
      expect(r.final).toBe(1);
      expect(r.lost).toHaveLength(n - 1);
      expect(r.survived).toEqual([n - 1]); // the last store wins
    }
  });
});

describe('run(): registers, memory and partial schedules', () => {
  it('registers follow the steps', () => {
    const r = run([0, 0, 1], { n: 2 });
    expect(r.steps[2].regs).toEqual([{ old: 0, new: 1 }, { old: 0, new: null }]);
    expect(r.complete).toBe(false);
    expect(r.correct).toBe(false);
    expect(r.lost).toEqual([]);
  });
  it('nextOp names the step a click would take', () => {
    expect(nextOp([], 0, { n: 2 })).toBe('load');
    expect(nextOp([0, 0], 0, { n: 2 })).toBe('store');
    expect(nextOp([0, 0, 0], 0, { n: 2 })).toBe(null);
    expect(nextOp([], 1, { n: 2, atomic: true })).toBe('atomic');
    expect(nextOp([1], 1, { n: 2, atomic: true })).toBe(null);
  });
  it('rejects a fourth step or a missing thread', () => {
    expect(() => progressOf([0, 0, 0, 0], { n: 2 })).toThrow();
    expect(() => progressOf([2], { n: 2 })).toThrow();
    expect(() => progressOf([0, 0], { n: 2, atomic: true })).toThrow();
  });
  it('a stale store can bring an erased update back: the set model names the truly lost one', () => {
    // T1 loads 0; T2 and T3 complete (bin 2); T4 loads 2; T1 stores 1, erasing T2 and T3;
    // T4 stores 3, which contains T2 and T3 again. Only T1's increment is missing.
    const r = run([0, 1, 1, 1, 2, 2, 2, 3, 0, 0, 3, 3], { n: 4 });
    expect(r.steps[9].erased).toEqual([1, 2]);
    expect(r.final).toBe(3);
    expect(r.lost).toEqual([0]);
    expect(r.survived).toEqual([1, 2, 3]);
  });
  it('a starting value other than 0 shifts every number', () => {
    expect(run([0, 0, 1, 0, 1, 1], { n: 2, initial: 5 }).final).toBe(6);
  });
});

describe('atomic mode: exclusion, not order', () => {
  it('every order of whole increments ends at n', () => {
    for (const order of [[0, 1], [1, 0]]) expect(run(order, { n: 2, atomic: true }).final).toBe(2);
    expect(run([3, 1, 0, 2], { n: 4, atomic: true }).final).toBe(4);
  });
  it('a preset becomes the order in which its threads first appear', () => {
    expect(atomicOrder(preset('overlap-t2').schedule(2))).toEqual([1, 0]);
    expect(run(atomicOrder(preset('lockstep').schedule(4)), { n: 4, atomic: true }).final).toBe(4);
  });
  it('the old value each thread gets back differs: 0 for the first, 1 for the second', () => {
    const r = run([1, 0], { n: 2, atomic: true });
    expect(r.steps.map((s) => [s.thread, s.regs[s.thread].old])).toEqual([[1, 0], [0, 1]]);
  });
});

// An independent check of the census: integers only, no sets.
function bruteForce(n: number): number[] {
  const byFinal = new Array<number>(n + 1).fill(0);
  const pos = new Array<number>(n).fill(0);
  const reg = new Array<number>(n).fill(0);
  const go = (mem: number, left: number) => {
    if (!left) return void byFinal[mem]++;
    for (let t = 0; t < n; t++) {
      const p = pos[t];
      if (p === 3) continue;
      pos[t]++;
      const keep = reg[t];
      if (p === 0) { reg[t] = mem; go(mem, left - 1); }
      else if (p === 1) { reg[t] = keep + 1; go(mem, left - 1); }
      else go(reg[t], left - 1);
      reg[t] = keep;
      pos[t]--;
    }
  };
  go(0, 3 * n);
  return byFinal;
}

describe('every interleaving (the enumerator)', () => {
  it('2 threads: 20 interleavings, 2 correct, 18 ending at 1', () => {
    const c = enumerate(2);
    expect(c.interleavings).toBe(20);
    expect(c.correct).toBe(2);
    expect(c.byFinal).toEqual([0, 18, 2]);
  });
  it('3 threads: 1680 interleavings, 6 correct', () => {
    const c = enumerate(3);
    expect(c.interleavings).toBe(1680);
    expect(c.correct).toBe(6);
  });
  it('4 threads: 369,600 interleavings, 24 correct', () => {
    const c = enumerate(4);
    expect(c.interleavings).toBe(369600);
    expect(c.correct).toBe(24);
  });
  it('the correct runs are exactly the serial ones: n!', () => {
    expect([2, 3, 4].map((n) => enumerate(n).correct)).toEqual([2, 6, 24]);
  });
  it('the counts agree with a plain integer brute force', () => {
    for (const n of [2, 3, 4]) expect(enumerate(n).byFinal).toEqual(bruteForce(n));
  });
  it('pins the distributions the page shows', () => {
    expect(enumerate(3).byFinal).toEqual([0, 1404, 270, 6]);
    expect(enumerate(4).byFinal).toEqual([0, 291600, 74952, 3024, 24]);
  });
  it('the closed form (3n)!/(3!)^n agrees', () => {
    expect([2, 3, 4].map((n) => interleavingCount(n))).toEqual([20, 1680, 369600]);
  });
  it('with atomics every one of the n! orders is correct', () => {
    for (const n of [2, 3, 4]) {
      const c = enumerate(n, { atomic: true });
      expect(c.interleavings).toBe([0, 1, 2, 6, 24][n]);
      expect(c.correct).toBe(c.interleavings);
    }
  });
});
