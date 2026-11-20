import { describe, expect, it } from 'vitest';
import {
  BOOK_SHAPE,
  afterStrip,
  fmas,
  readsPerFma,
  registersFor,
  reuse,
  run,
  sharedReadsV1,
  sharedReadsV2,
  smallInputs,
  opsForFmas,
  stateAfterOps,
  stateAt,
  type Shape,
} from './gemmreads';

// Numbers section 15.4 states, pinned to the model they come from.
describe('the two loop orders (Figs. 15.6 and 15.9)', () => {
  const x = smallInputs(BOOK_SHAPE);
  const v1 = run(BOOK_SHAPE, x, 'v1');
  const v2 = run(BOOK_SHAPE, x, 'v2');
  it('give identical thread tiles', () => {
    expect(v2.C).toEqual(v1.C);
    // and the tile is the product of the two input tiles
    const ref = Array.from({ length: 64 }, (_, k) => {
      let s = 0;
      for (let i = 0; i < 8; i++) s += x.a[Math.floor(k / 8) * 8 + i] * x.b[i * 8 + (k % 8)];
      return s;
    });
    expect(v1.C).toEqual(ref);
  });
  it('1024 shared reads against 128 for 8 x 8 x 8: eight times fewer', () => {
    expect([v1.sharedReads, v2.sharedReads]).toEqual([1024, 128]);
    expect([sharedReadsV1(BOOK_SHAPE), sharedReadsV2(BOOK_SHAPE)]).toEqual([1024, 128]);
    expect(v1.sharedReads / v2.sharedReads).toBe(8);
  });
  it('the same 512 multiply-adds; the strips are read 1024 times from registers instead', () => {
    expect([v1.fmas, v2.fmas, fmas(BOOK_SHAPE)]).toEqual([512, 512, 512]);
    expect([v1.registerReads, v2.registerReads]).toEqual([0, 1024]);
  });
  it('a value in a_r feeds tN = 8 multiply-adds, one in b_r feeds tM = 8', () => {
    expect(reuse(BOOK_SHAPE)).toEqual({ a: 8, b: 8 });
    const st = stateAt(v2, afterStrip(BOOK_SHAPE, 0));
    expect(st.aReads.filter((n) => n > 0)).toHaveLength(8);
    expect(st.sharedReads).toBe(16);
  });
  it('registers: 64 accumulators, plus 8 + 8 for the strips = 80 (p. 363)', () => {
    expect([registersFor(BOOK_SHAPE, 'v1'), registersFor(BOOK_SHAPE, 'v2')]).toEqual([64, 80]);
  });
});

describe('the shape of a 64-accumulator tile', () => {
  const per = (tM: number, tN: number) => readsPerFma({ tM, tN, bK: 8 }, 'v2');
  it('reads per FMA: 0.25, 0.3125, 0.53125 for 8 x 8, 16 x 4, 32 x 2; always 2 in the first order', () => {
    expect([per(8, 8), per(16, 4), per(32, 2)]).toEqual([0.25, 0.3125, 0.53125]);
    expect(readsPerFma({ tM: 16, tN: 4, bK: 8 }, 'v1')).toBe(2);
  });
  it('closed forms agree with the traced counts for many shapes', () => {
    for (const [tM, tN, bK] of [[1, 1, 1], [3, 5, 2], [16, 4, 8], [32, 2, 16], [7, 9, 3]]) {
      const s: Shape = { tM, tN, bK };
      const x = smallInputs(s);
      const a = run(s, x, 'v1');
      const b = run(s, x, 'v2');
      expect([a.sharedReads, b.sharedReads]).toEqual([sharedReadsV1(s), sharedReadsV2(s)]);
      expect(b.C).toEqual(a.C);
    }
  });
});

describe('stepping (the explorer\'s stepper and the outer-product scene)', () => {
  const x = smallInputs(BOOK_SHAPE);
  const v1 = run(BOOK_SHAPE, x, 'v1');
  const v2 = run(BOOK_SHAPE, x, 'v2');
  it('first order: output (0, 0) reads row 0 of a and column 0 of b, 16 reads after 8 steps', () => {
    const st = stateAt(v1, 8);
    expect(st.sharedReads).toBe(16);
    expect(st.aReads.slice(0, 8)).toEqual(new Array(8).fill(1));
    expect(st.C[0]).toBe(v1.C[0]);
  });
  it('first order: output (0, 1) reads the same 8 values of a again', () => {
    const st = stateAt(v1, 16);
    expect(st.aReads.slice(0, 8)).toEqual(new Array(8).fill(2));
    expect(st.sharedReads).toBe(32);
  });
  it('first order, whole tile: every a slot read 8 times, every b slot 8 times', () => {
    const st = stateAt(v1, 512);
    expect(st.aReads.every((n) => n === 8) && st.bReads.every((n) => n === 8)).toBe(true);
    expect(st.sharedReads).toBe(1024);
  });
  it('second order: the first step loads 16 strip values, then 64 multiply-adds use them', () => {
    const one = stateAt(v2, 1);
    expect(one.step.filter((o) => o.kind !== 'fma')).toHaveLength(16);
    expect(one.sharedReads).toBe(16);
    const strip = stateAt(v2, 64);
    expect(strip.sharedReads).toBe(16);
    expect(strip.aR.every((v) => v !== null) && strip.bR.every((v) => v !== null)).toBe(true);
  });
  it('updates count multiply-adds per element: after one rank-1 update every element has one, in the first order only 8 of them', () => {
    expect(stateAt(v2, 64).updates.every((n) => n === 1)).toBe(true);
    expect(stateAt(v1, 64).updates.filter((n) => n > 0)).toEqual(new Array(8).fill(8));
  });
  it('second order, after 8 rank-1 updates: 128 reads, every slot once, the same tile', () => {
    const st = stateAt(v2, 512);
    expect(st.sharedReads).toBe(128);
    expect(st.aReads.every((n) => n === 1) && st.bReads.every((n) => n === 1)).toBe(true);
    expect(st.C).toEqual(v1.C);
  });
  it('k = 0 is the empty state', () => {
    const st = stateAt(v2, 0);
    expect([st.fmas, st.sharedReads, st.step.length]).toEqual([0, 0, 0]);
  });
});

describe('stepping by ops (the scene loads a strip value at a time)', () => {
  const v2 = run(BOOK_SHAPE, smallInputs(BOOK_SHAPE), 'v2');
  it('after 5 ops of the second order: 5 values of the A strip loaded, no multiply-add yet', () => {
    const st = stateAfterOps(v2, 5);
    expect([st.fmas, st.sharedReads, st.aR.filter((v) => v !== null).length, st.step.length]).toEqual([0, 5, 5, 5]);
  });
  it('opsForFmas counts the 16 strip loads in front of the first multiply-add', () => {
    expect([opsForFmas(v2, 0), opsForFmas(v2, 1), opsForFmas(v2, 2), opsForFmas(v2, 64), opsForFmas(v2, 65)]).toEqual([0, 17, 18, 80, 97]);
  });
});
