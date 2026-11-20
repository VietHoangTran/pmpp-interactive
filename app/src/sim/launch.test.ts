import { describe, expect, it } from 'vitest';
import {
  BOOK_BLOCK,
  BOOK_NS,
  EXPRESSIONS,
  KERNELS,
  ORDERS,
  SCENE,
  ZERO_BLOCKS,
  blockCount,
  blockWork,
  ceilDiv,
  distinctResults,
  ex9,
  globalIndex,
  launch,
  launchWith,
  makeData,
  permutations,
  runKernel,
  schedule,
  scheduleOf,
  sequential,
  showValue,
  smallestFailing,
  smallestWasting,
  verdict,
  waveCount,
  type ExprKey,
} from './launch';

const SIZES = [32, 128, 256, 1024];

describe("the book's worked numbers (pp. 38 to 39)", () => {
  it('n = 1000 with 256 threads per block: 4 blocks, 1024 threads, 24 do nothing', () => {
    const c = blockCount('book-ceil', 1000, 256);
    expect(c.blocks).toBe(4);
    const r = launch(1000, 256, c.blocks);
    expect(r.threads).toBe(1024);
    expect(r.working).toBe(1000);
    expect(r.idle).toBe(24);
    expect(r.unprocessedCount).toBe(0);
    expect(r.emptyBlocks).toBe(0);
    expect(r.correct).toBe(true);
  });
  it('750 gives 3 blocks, 4000 gives 16, 2,000,000 gives 7813', () => {
    expect(BOOK_BLOCK).toBe(256);
    expect(BOOK_NS).toEqual([1000, 750, 4000, 2000000]);
    expect(BOOK_NS.map((n) => blockCount('book-ceil', n, BOOK_BLOCK).blocks)).toEqual([4, 3, 16, 7813]);
    expect(BOOK_NS.map((n) => blockCount('int-ceil', n, BOOK_BLOCK).blocks)).toEqual([4, 3, 16, 7813]);
  });
  it('2,000,000 elements: 2,000,128 threads, 128 of them idle', () => {
    const r = launchWith('int-ceil', 2000000, 256);
    expect(r.threads).toBe(2000128);
    expect(r.idle).toBe(128);
  });
  it('the guard example of section 2.5 (p. 37): n = 100 in blocks of 32 is 4 blocks, 128 threads, 28 idle', () => {
    const r = launchWith('int-ceil', 100, 32);
    expect([r.blocks, r.threads, r.idle]).toEqual([4, 128, 28]);
  });
});

describe('book exercises', () => {
  it('exercise 9 (Fig. 2.15, p. 42): 128 per block, 1563 blocks, 200,064 threads, 200,000 reach the assignment', () => {
    expect(ex9()).toEqual({ N: 200000, threadsPerBlock: 128, blocks: 1563, threads: 200064, line2: 200064, line4: 200000 });
  });
  it('exercise 4 (p. 41): 8000 elements in blocks of 1024 is 8 blocks, 8192 threads', () => {
    const r = launchWith('int-ceil', 8000, 1024);
    expect(r.blocks).toBe(8);
    expect(r.threads).toBe(8192);
    expect(r.idle).toBe(192);
  });
});

describe('the five expressions, with C++ semantics', () => {
  it('there are five, and each writes the block size as a literal', () => {
    expect(EXPRESSIONS.map((e) => e.key)).toEqual(['int-ceil', 'book-ceil', 'trunc', 'ceil-trunc', 'plus-one']);
    expect(EXPRESSIONS.map((e) => e.source(256))).toEqual([
      '(n + 256 - 1) / 256',
      'ceil(n / 256.0)',
      'n / 256',
      'ceil(n / 256)',
      'n / 256 + 1',
    ]);
  });
  it('the integer and floating ceilings agree with each other and with the true ceiling', () => {
    for (const B of SIZES)
      for (let n = 1; n <= 5000; n++) {
        const want = ceilDiv(n, B);
        expect(blockCount('int-ceil', n, B).blocks).toBe(want);
        expect(blockCount('book-ceil', n, B).blocks).toBe(want);
      }
  });
  it('the two right forms are right for every n: all elements processed, no block without work', () => {
    for (const key of ['int-ceil', 'book-ceil'] as ExprKey[])
      for (const B of SIZES)
        for (let n = 1; n <= 5000; n += 7) {
          const r = launchWith(key, n, B);
          expect(r.correct).toBe(true);
          expect(r.emptyBlocks).toBe(0);
          expect(r.idle).toBe((B - (n % B)) % B);
          expect(r.idle).toBeLessThan(B);
        }
  });
  it('n / B loses exactly n mod B elements, and ceil of it loses the same', () => {
    for (const key of ['trunc', 'ceil-trunc'] as ExprKey[])
      for (const B of SIZES)
        for (let n = 1; n <= 5000; n++) {
          const r = launchWith(key, n, B);
          expect(r.blocks).toBe(Math.floor(n / B));
          expect(r.unprocessedCount).toBe(n % B);
          expect(r.correct).toBe(n % B === 0);
          expect(r.idle).toBe(0);
          expect(r.launched).toBe(n >= B);
          if (n % B) expect(r.unprocessed).toEqual([{ first: n - (n % B), last: n - 1 }]);
          else expect(r.unprocessed).toEqual([]);
        }
  });
  it('n / B + 1 never loses an element and wastes one block exactly when B divides n', () => {
    for (const B of SIZES)
      for (let n = 1; n <= 5000; n++) {
        const r = launchWith('plus-one', n, B);
        expect(r.unprocessedCount).toBe(0);
        expect(r.correct).toBe(true);
        expect(r.emptyBlocks).toBe(n % B === 0 ? 1 : 0);
        if (n % B === 0) expect(r.idle).toBe(B);
      }
  });
  it('the steps show where the fraction goes', () => {
    const bad = blockCount('ceil-trunc', 1000, 256);
    expect(bad.steps.map((s) => [s.text, s.value, s.type])).toEqual([
      ['1000 / 256', 3, 'int'],
      ['ceil(3)', 3, 'double'],
      ['3.0 as a block count', 3, 'unsigned'],
    ]);
    expect(bad.steps[0].rule).toContain('3.90625');
    expect(bad.blocks).toBe(3);
    expect(bad.needed).toBe(4);

    const book = blockCount('book-ceil', 1000, 256);
    expect(book.steps.map((s) => [s.text, s.value, s.type])).toEqual([
      ['1000 / 256.0', 3.90625, 'double'],
      ['ceil(3.90625)', 4, 'double'],
      ['4.0 as a block count', 4, 'unsigned'],
    ]);

    const int = blockCount('int-ceil', 1000, 256);
    expect(int.steps.map((s) => [s.text, s.value, s.type])).toEqual([
      ['1000 + 256 - 1', 1255, 'int'],
      ['1255 / 256', 4, 'int'],
    ]);
    expect(blockCount('plus-one', 1024, 256).steps.map((s) => s.value)).toEqual([4, 5]);
  });
  it('the question in the lesson: ceil(n/256) with n = 1000 leaves elements 768 to 999', () => {
    const r = launchWith('ceil-trunc', 1000, 256);
    expect(r.blocks).toBe(3);
    expect(r.unprocessed).toEqual([{ first: 768, last: 999 }]);
    expect(launchWith('ceil-trunc', 1024, 256).correct).toBe(true);
  });
  it('values are written the way C++ would print the type', () => {
    expect(showValue(4, 'double')).toBe('4.0');
    expect(showValue(4, 'int')).toBe('4');
    expect(showValue(3.90625, 'double')).toBe('3.90625');
    expect(showValue(1255 / 256, 'double')).toBe('4.902344');
  });
  it('rejects sizes that are not whole numbers', () => {
    expect(() => blockCount('trunc', 1.5, 256)).toThrow(RangeError);
    expect(() => blockCount('trunc', 10, 0)).toThrow(RangeError);
    expect(() => launch(10, 4, -1)).toThrow(RangeError);
  });
});

describe('the launch', () => {
  it('zero blocks: nothing runs, and each backend says what its source says', () => {
    const r = launchWith('trunc', 100, 256);
    expect(r.blocks).toBe(0);
    expect(r.launched).toBe(false);
    expect(r.threads).toBe(0);
    expect(r.unprocessed).toEqual([{ first: 0, last: 99 }]);
    expect(r.zero).toBe(ZERO_BLOCKS);
    expect(Object.fromEntries(ZERO_BLOCKS.map((z) => [z.backend, z.outcome]))).toEqual({
      cuda: 'not documented',
      hip: 'error, documented',
      emu: 'error',
    });
    expect(ZERO_BLOCKS.find((z) => z.backend === 'hip')!.detail).toBe('hipErrorInvalidValue');
    expect(launchWith('int-ceil', 100, 256).zero).toBeNull();
  });
  it('without the guard the surplus threads write past the end', () => {
    const on = launch(100, 32, 4, true);
    const off = launch(100, 32, 4, false);
    expect(on.outOfRangeCount).toBe(0);
    expect(on.correct).toBe(true);
    expect(off.idle).toBe(0);
    expect(off.outOfRange).toEqual([{ first: 100, last: 127 }]);
    expect(off.outOfRangeCount).toBe(28);
    expect(off.correct).toBe(false);
    expect(launch(128, 32, 4, false).correct).toBe(true);
  });
  it('a block is described by the indices its threads compute', () => {
    expect(blockWork(22, 8, 0)).toEqual({ block: 0, first: 0, last: 7, working: 8 });
    expect(blockWork(22, 8, 2)).toEqual({ block: 2, first: 16, last: 23, working: 6 });
    expect(blockWork(22, 8, 3).working).toBe(0);
    expect(globalIndex(2, 8, 6)).toBe(22);
  });
});

describe('find the n that breaks it', () => {
  it('n / 256 and ceil(n / 256): any multiple of 256 passes, n = 1 is the smallest failure', () => {
    for (const key of ['trunc', 'ceil-trunc'] as ExprKey[]) {
      for (const n of [256, 512, 1024, 4096]) expect(verdict(key, n, 256).passes).toBe(true);
      expect(verdict(key, 1000, 256).passes).toBe(false);
      expect(smallestFailing(key, 256)).toBe(1);
      expect(verdict(key, 1, 256).blocks).toBe(0);
      expect(smallestWasting(key, 256)).toBeNull();
    }
  });
  it('n / 256 + 1: no n fails, and 256 is the smallest n with a block that does nothing', () => {
    expect(smallestFailing('plus-one', 256, 20000)).toBeNull();
    expect(smallestWasting('plus-one', 256)).toBe(256);
    expect(verdict('plus-one', 255, 256).wastes).toBe(false);
    expect(verdict('plus-one', 256, 256)).toMatchObject({ blocks: 2, passes: true, wastes: true });
  });
  it('a test at n = 1024 cannot tell any of the three from the right count', () => {
    expect((['trunc', 'ceil-trunc', 'plus-one'] as ExprKey[]).map((k) => verdict(k, 1024, 256).passes)).toEqual([true, true, true]);
    expect((['trunc', 'ceil-trunc', 'plus-one'] as ExprKey[]).map((k) => verdict(k, 1024, 256).blocks)).toEqual([4, 4, 5]);
  });
  it('the right forms never fail and never waste', () => {
    for (const key of ['int-ceil', 'book-ceil'] as ExprKey[]) {
      expect(smallestFailing(key, 256, 5000)).toBeNull();
      expect(smallestWasting(key, 256, 5000)).toBeNull();
    }
  });
});

describe('order and concurrency', () => {
  it('a schedule holds every block once, in waves of at most S', () => {
    for (const order of ORDERS)
      for (let blocks = 1; blocks <= 12; blocks++)
        for (let S = 1; S <= 8; S++) {
          const s = schedule(blocks, S, order, 3);
          expect(s.order.slice().sort((a, b) => a - b)).toEqual(Array.from({ length: blocks }, (_, k) => k));
          expect(s.waves.flat()).toEqual(s.order);
          expect(s.waves.every((w) => w.length >= 1 && w.length <= S)).toBe(true);
          expect(s.waves.length).toBe(waveCount(blocks, S));
        }
  });
  it('ascending, descending and a repeatable shuffle that is never ascending', () => {
    expect(schedule(4, 1, 'ascending').order).toEqual([0, 1, 2, 3]);
    expect(schedule(4, 3, 'descending').waves).toEqual([[3, 2, 1], [0]]);
    expect(schedule(8, 2, 'shuffled', 7)).toEqual(schedule(8, 2, 'shuffled', 7));
    for (let seed = 0; seed < 200; seed++)
      for (const blocks of [2, 3, 4])
        expect(schedule(blocks, 1, 'shuffled', seed).order).not.toEqual(schedule(blocks, 1, 'ascending').order);
  });
  it('8 blocks: 8 waves one at a time, 4 waves two at a time, 1 wave eight at a time', () => {
    expect([1, 2, 3, 4, 8].map((S) => waveCount(8, S))).toEqual([8, 4, 3, 2, 1]);
  });
  it('the two kernels are the ones the lesson shows', () => {
    expect(KERNELS.map((k) => k.body)).toEqual(['C[i] = A[i] + B[i]', 'C[i] = C[i-1] + A[i]']);
  });
  it('the independent kernel gives the same C for every order and every S', () => {
    for (const [n, B] of [[13, 2], [22, 8], [24, 4], [35, 3]]) {
      const d = makeData(n, B);
      const blocks = ceilDiv(n, B);
      const want = sequential('independent', d);
      expect(want).toEqual(d.A.map((a, i) => a + d.Bv[i]));
      for (const order of ORDERS)
        for (let S = 1; S <= 8; S++)
          for (const seed of [1, 2, 3]) {
            const run = runKernel('independent', schedule(blocks, S, order, seed), d);
            expect(run.final).toEqual(want);
            expect(run.matches).toBe(true);
            expect(run.after.length).toBe(waveCount(blocks, S));
          }
    }
  });
  it('the chained kernel matches the sequential loop only one block at a time in ascending order', () => {
    for (const [n, B] of [[13, 2], [22, 8], [24, 4], [35, 3]]) {
      const d = makeData(n, B);
      const blocks = ceilDiv(n, B);
      for (const order of ORDERS)
        for (let S = 1; S <= 8; S++) {
          const run = runKernel('chained', schedule(blocks, S, order, 2), d);
          expect(run.matches).toBe(order === 'ascending' && S === 1);
        }
    }
  });
  it('the chained kernel run sequentially is a running sum', () => {
    const d = makeData(6, 2);
    expect(d.A).toEqual([3, 0, 7, 4, 1, 8]);
    expect(sequential('chained', d)).toEqual([3, 3, 10, 14, 15, 23]);
    // Two blocks at once: block 1 reads C[1] before block 0 has written it.
    const run = runKernel('chained', schedule(3, 2, 'ascending'), d);
    expect(run.final).toEqual([3, 3, 7, 11, 12, 20]);
    expect(run.wrong).toEqual([2, 3, 4, 5]);
  });
  it('each wave writes whole blocks, and elements past n are never touched', () => {
    const d = makeData(SCENE.n, SCENE.B, NaN);
    const s = schedule(3, SCENE.S, 'shuffled', SCENE.seed);
    const run = runKernel('independent', s, d);
    expect(run.written[0]).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21]);
    expect(run.written[1]).toEqual([8, 9, 10, 11, 12, 13, 14, 15]);
    expect(run.after[0].slice(8, 16).every(Number.isNaN)).toBe(true);
    expect(run.final.length).toBe(22);
    expect(run.matches).toBe(true);
  });
  it("the explorer's whole range: 24 settings, the independent kernel the same in all, the chained one in exactly 1", () => {
    // Section 2.6, "Any order, any device": blocks of 2, 3 or 4 threads, 4 to 12 blocks, at most 32 elements.
    for (const B of [2, 3, 4])
      for (let n = 3 * B + 1; n <= Math.min(12 * B, 32); n++) {
        const blocks = ceilDiv(n, B);
        expect(blocks).toBeGreaterThanOrEqual(4);
        expect(blocks).toBeLessThanOrEqual(12);
        const d = makeData(n, B);
        for (const seed of [1, 2, 3, 4, 5]) {
          let same = 0;
          let chainedSame = 0;
          for (const order of ORDERS)
            for (let S = 1; S <= 8; S++) {
              const s = schedule(blocks, S, order, seed);
              if (runKernel('independent', s, d).matches) same++;
              if (runKernel('chained', s, d).matches) {
                chainedSame++;
                expect([order, S]).toEqual(['ascending', 1]);
              }
            }
          expect(same).toBe(24);
          expect(chainedSame).toBe(1);
        }
      }
  });
  it('all orders of a few blocks', () => {
    expect(permutations(3).length).toBe(6);
    expect(scheduleOf([2, 0, 1], 2).waves).toEqual([[2, 0], [1]]);
  });
});

describe('the narrated animation uses these numbers', () => {
  const { n, B, S, seed } = SCENE;
  it('22 elements in blocks of 8: (22 + 7) / 8 = 3 blocks, 24 threads, 2 stopped by the guard', () => {
    const c = blockCount('int-ceil', n, B);
    expect(c.steps.map((s) => s.value)).toEqual([29, 3]);
    const r = launch(n, B, c.blocks);
    expect([r.blocks, r.threads, r.working, r.idle]).toEqual([3, 24, 22, 2]);
    expect([globalIndex(2, B, 6), globalIndex(2, B, 7)]).toEqual([22, 23]);
  });
  it('the device takes block 2 and block 0 first, then block 1', () => {
    expect(schedule(3, S, 'shuffled', seed).waves).toEqual([[2, 0], [1]]);
  });
  it('every order and every S gives the same C', () => {
    expect(distinctResults('independent', 3, makeData(n, B))).toEqual({ runs: 18, distinct: 1 });
    expect(distinctResults('chained', 3, makeData(n, B)).distinct).toBeGreaterThan(1);
  });
  it('22 / 8 without rounding up: 2 blocks, elements 16 to 21 never written', () => {
    const r = launchWith('trunc', n, B);
    expect(r.blocks).toBe(2);
    expect(r.unprocessed).toEqual([{ first: 16, last: 21 }]);
    const d = makeData(n, B, NaN);
    const run = runKernel('independent', schedule(r.blocks, S, 'ascending'), d);
    expect(run.final.slice(16).every(Number.isNaN)).toBe(true);
    expect(run.wrong).toEqual([16, 17, 18, 19, 20, 21]);
  });
});
