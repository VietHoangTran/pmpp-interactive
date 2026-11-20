import { describe, expect, it } from 'vitest';
import { CORRECT, makeProblem, naiveBlockReads, runAll, runBlock, traffic, type Config, type Schedule } from './tiled';

const SCHEDULES: Schedule[] = ['lockstep', 'in-order', 'reversed', 'shuffled'];
const cfg = (width: number, tile: number, over: Partial<Config> = {}): Config => ({ ...CORRECT, width, tile, ...over });

describe('correct kernel', () => {
  for (const [w, t] of [[4, 2], [6, 2], [6, 3], [8, 4], [3, 2], [5, 2], [7, 3], [1, 2]] as const)
    for (const s of SCHEDULES)
      it(`width ${w}, tile ${t}, ${s}`, () => {
        const r = runAll(cfg(w, t), makeProblem(w), s, 3);
        expect(r.violations).toEqual([]);
        expect(r.correct).toBe(true);
      });
});

describe('missing barriers', () => {
  it('without the first barrier, a thread that runs ahead reads slots nobody has loaded', () => {
    const r = runAll(cfg(4, 2, { barrier1: false }), makeProblem(4), 'in-order');
    expect(r.correct).toBe(false);
    expect(r.violations.some((v) => v.kind === 'read-before-write')).toBe(true);
  });
  it('without the second barrier, a thread that runs ahead overwrites slots still in use', () => {
    const r = runAll(cfg(4, 2, { barrier2: false }), makeProblem(4), 'in-order');
    expect(r.correct).toBe(false);
    expect(r.violations.some((v) => v.kind === 'overwritten-before-read')).toBe(true);
    expect(r.violations.some((v) => v.kind === 'read-before-write')).toBe(false);
  });
  it('fails under every schedule except lockstep', () => {
    for (const over of [{ barrier1: false }, { barrier2: false }])
      for (const s of ['in-order', 'reversed', 'shuffled'] as Schedule[])
        expect(runAll(cfg(6, 3, over), makeProblem(6), s, 11).correct).toBe(false);
  });
  it('lockstep execution hides both missing barriers', () => {
    const r = runAll(cfg(6, 3, { barrier1: false, barrier2: false }), makeProblem(6), 'lockstep');
    expect(r.correct).toBe(true);
    expect(r.violations).toEqual([]);
  });
  it('with a single phase there is nothing for the second barrier to protect', () => {
    expect(runAll(cfg(2, 2, { barrier2: false }), makeProblem(2), 'in-order').correct).toBe(true);
  });
});

describe('boundary checks (width 3, tile 2: the book example of Figs. 5.11 and 5.12)', () => {
  const prob = makeProblem(3);
  it('an unchecked M load past the end of a row fetches the next row instead', () => {
    const r = runAll(cfg(3, 2, { guardM: false }), prob, 'lockstep');
    expect(r.correct).toBe(false);
    expect(r.violations.some((v) => v.kind === 'wrong-element')).toBe(true);
  });
  it('block (0,0), second phase: threads (1,0) and (1,1) reach for M[0][3] and M[1][3]', () => {
    const tr = runBlock(cfg(3, 2, { guardM: false }), prob, 0, 0, 'lockstep');
    const bad = tr.steps.filter((s) => s.op === 'loadM' && s.global?.access === 'wrap');
    expect(bad.map((s) => [s.tx, s.ty, s.ph, s.global!.row, s.global!.col])).toEqual([
      [1, 0, 1, 0, 3],
      [1, 1, 1, 1, 3],
    ]);
  });
  it('an unchecked N load past the end of a column leaves the array', () => {
    const r = runAll(cfg(3, 2, { guardN: false }), prob, 'lockstep');
    expect(r.correct).toBe(false);
    expect(r.violations.some((v) => v.kind === 'outside-read')).toBe(true);
  });
  it('an unchecked store corrupts another element of P', () => {
    const r = runAll(cfg(3, 2, { guardP: false }), prob, 'lockstep');
    expect(r.violations.some((v) => v.kind === 'bad-store')).toBe(true);
    expect(r.correct).toBe(false);
  });
  it('checks without zero-fill leave stale or uninitialised slots in use', () => {
    const r = runAll(cfg(3, 2, { zeroFillM: false, zeroFillN: false }), prob, 'lockstep');
    expect(r.correct).toBe(false);
    expect(r.violations.some((v) => v.kind === 'unfilled')).toBe(true);
  });
  it('zero-filling only one tile: flagged, yet the result is right when the stale values are finite', () => {
    // Every unfilled M slot is multiplied by an N slot that WAS zeroed, and here the
    // unfilled slots hold finite values left over from the previous phase.
    const r = runAll(cfg(3, 2, { zeroFillM: false }), prob, 'lockstep');
    expect(r.violations.some((v) => v.kind === 'unfilled')).toBe(true);
    expect(r.correct).toBe(true);
  });
  it('zero-filling only one tile fails when a slot was never written: NaN * 0 is NaN', () => {
    // A matrix narrower than one tile: in the only phase, M[0][1] does not exist, so
    // Mds[0][1] is uninitialised shared memory, which this model treats as NaN.
    const r = runAll(cfg(1, 2, { zeroFillM: false }), makeProblem(1), 'lockstep');
    expect(r.correct).toBe(false);
    expect(Number.isNaN(r.P[0])).toBe(true);
  });
  it('a thread that owns no P element still loads for its block', () => {
    // Book p. 113: in block (1,1), the thread at tx 1, ty 0 would own P[2][3], which
    // does not exist, yet in phase 0 it must load M[2][1] for its neighbour.
    const tr = runBlock(cfg(3, 2), prob, 1, 1, 'lockstep');
    const s = tr.steps.find((x) => x.op === 'loadM' && x.ph === 0 && x.tx === 1 && x.ty === 0)!;
    expect([s.global!.row, s.global!.col, s.global!.access]).toEqual([2, 1, 'ok']);
    const store = tr.steps.find((x) => x.op === 'store' && x.tx === 1 && x.ty === 0)!;
    expect(store.global!.access).toBe('skipped');
  });
});

describe('traffic model', () => {
  it('matches the book: 0.25 FLOP/B naive, 0.25 * T tiled', () => {
    expect(traffic(1024, null).flopPerByte).toBeCloseTo(0.25, 10);
    expect(traffic(1024, 32).flopPerByte).toBeCloseTo(8, 10);
    expect(traffic(1024, 16).perElement).toBe(64);
  });
  it('in the naive kernel every needed element is read by T threads of a T x T block', () => {
    const { M, N } = naiveBlockReads(4, 2, 0, 0);
    expect(M.filter((x) => x > 0)).toEqual([2, 2, 2, 2, 2, 2, 2, 2]);
    expect(N.filter((x) => x > 0)).toEqual([2, 2, 2, 2, 2, 2, 2, 2]);
    const big = naiveBlockReads(8, 4, 1, 0);
    expect(new Set(big.M.filter((x) => x > 0))).toEqual(new Set([4]));
  });
});
