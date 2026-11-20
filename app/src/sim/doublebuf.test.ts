import { describe, expect, it } from 'vitest';
import {
  SCHEDULES,
  VARIANTS,
  expectedValue,
  firstFailure,
  loopSource,
  nextRelease,
  plan,
  run,
  stateAt,
  tileBytes,
  type Variant,
} from './doublebuf';

const cfg = (variant: Variant, threads = 4, iterations = 3) => ({ threads, iterations, variant });
const kinds = (v: Variant, s: (typeof SCHEDULES)[number], P = 4, N = 3) =>
  [...new Set(run(cfg(v, P, N), s).violations.map((x) => x.kind))].sort();

describe('the loop and the reference', () => {
  it('values travel round the ring: thread t ends with 10n + (t + n) mod P', () => {
    expect(expectedValue(0, 3, 4)).toBe(33);
    expect(expectedValue(3, 3, 4)).toBe(32);
    expect(run(cfg('single-both'), 'in-order').output).toEqual([33, 30, 31, 32]);
  });
  it('one buffer has two barriers per trip; two buffers one barrier and a swap', () => {
    expect(plan('single-both')).toEqual(['read', 'barrierRead', 'write', 'barrierWrite']);
    expect(plan('double')).toEqual(['read', 'write', 'barrierWrite', 'swap']);
  });
});

describe('every thread order, P = 4 to 8, 2 to 4 iterations', () => {
  for (const P of [4, 6, 8])
    for (const N of [2, 3, 4])
      it(`P = ${P}, ${N} iterations`, () => {
        for (const s of SCHEDULES) {
          // Correct kernels are correct under every order.
          expect(run(cfg('single-both', P, N), s).correct).toBe(true);
          expect(run(cfg('double', P, N), s).correct).toBe(true);
          expect(run(cfg('double', P, N), s).violations).toEqual([]);
          // A forgotten swap is a logic error: wrong under every order, lockstep included.
          const ns = run(cfg('double-no-swap', P, N), s);
          expect(ns.wrong.length).toBe(P);
          expect(new Set(ns.violations.map((v) => v.kind))).toEqual(new Set(['wrong-buffer']));
        }
        // Lockstep hides every missing barrier.
        for (const v of ['single-no-read-barrier', 'single-no-write-barrier', 'double-no-barrier'] as Variant[])
          expect(run(cfg(v, P, N), 'lockstep').correct).toBe(true);
        // Running ahead exposes them.
        for (const v of ['single-no-read-barrier', 'single-no-write-barrier', 'double-no-barrier'] as Variant[])
          for (const s of ['in-order', 'reversed'] as const) expect(run(cfg(v, P, N), s).correct).toBe(false);
      });
});

describe('which barrier guards which dependence', () => {
  it('without the barrier after the read: write-after-read failures only, in order', () => {
    expect(kinds('single-no-read-barrier', 'in-order')).toEqual(['write-after-read']);
    expect(kinds('single-no-read-barrier', 'reversed')).toEqual(['write-after-read']);
    const r = run(cfg('single-no-read-barrier'), 'in-order');
    expect(r.violations[0]).toMatchObject({ kind: 'write-after-read', thread: 3, iter: 0 });
    expect(r.violations[0].message).toBe('thread 3 still needed the initial value in slot 0, but thread 0 had already overwritten it in iteration 0');
  });
  it('without the barrier after the write: read-after-write failures only', () => {
    expect(kinds('single-no-write-barrier', 'in-order')).toEqual(['read-after-write']);
    expect(kinds('single-no-write-barrier', 'shuffled')).toEqual(['read-after-write']);
    // With one iteration there is no next read to get ahead of.
    expect(run(cfg('single-no-write-barrier', 4, 1), 'in-order').correct).toBe(true);
  });
  it('two buffers with no barrier at all fail, with read-after-write among the failures', () => {
    expect(run(cfg('double-no-barrier'), 'in-order').correct).toBe(false);
    expect(kinds('double-no-barrier', 'in-order')).toContain('read-after-write');
  });
  it('the outcome table for P = 4, 3 iterations, as the lesson shows it', () => {
    const table = Object.fromEntries(
      VARIANTS.map((v) => [v, SCHEDULES.map((s) => run(cfg(v), s).wrong.length)]),
    );
    expect(table).toEqual({
      'single-both': [0, 0, 0, 0],
      'single-no-read-barrier': [0, 3, 4, 2],
      'single-no-write-barrier': [0, 4, 2, 2],
      double: [0, 0, 0, 0],
      'double-no-swap': [4, 4, 4, 4],
      'double-no-barrier': [0, 4, 3, 1],
    });
  });
});

describe('stepper support', () => {
  it('first failure and next release', () => {
    const r = run(cfg('single-no-read-barrier'), 'in-order');
    expect(firstFailure(r)).toBe(10);
    expect(firstFailure(run(cfg('double'), 'in-order'))).toBeNull();
    const d = run(cfg('double'), 'in-order');
    // Each thread reads and writes (2 ops) and reaches the barrier: 12 steps, then the release.
    expect(nextRelease(d, 0)).toBe(13);
    expect(d.steps[12].op).toBe('release');
  });
  it('state replays buffers, registers and pointers', () => {
    const d = run(cfg('double'), 'in-order');
    const st = stateAt(d, 13);
    expect(st.buffers[1].map((s) => s.value)).toEqual([11, 12, 13, 10]);
    expect(st.buffers[1].every((s) => s.version === 1)).toBe(true);
    expect(st.threads.every((t) => !t.waiting && t.inBuf === 0)).toBe(true);
    const after = stateAt(d, d.steps.length);
    expect(after.threads.every((t) => t.done)).toBe(true);
    // Three swaps: each thread's in pointer ends on buffer B, which holds the last results.
    expect(after.threads.every((t) => t.inBuf === 1)).toBe(true);
    expect(after.buffers[1].map((s) => s.value)).toEqual([33, 30, 31, 32]);
  });
  it('loop source lines and the line each operation sits on', () => {
    const s = loopSource('double');
    expect(s.code.split('\n')).toHaveLength(6);
    expect(s.line).toEqual({ read: 2, write: 3, barrierWrite: 4, swap: 5 });
    expect(loopSource('single-no-read-barrier').code).not.toContain('after the read');
  });
});

describe('q-smem: tile memory of the tiled matmul', () => {
  it('T = 32: 8192 bytes with one pair of tiles, 16384 with two', () => {
    expect(tileBytes(32, 1)).toBe(8192);
    expect(tileBytes(32, 2)).toBe(16384);
    // Two pairs at T = 32 fit Metal's 32 KB threadgroup limit (ch05-facts C1).
    expect(tileBytes(32, 2)).toBeLessThanOrEqual(32 * 1024);
  });
});
