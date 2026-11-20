import { describe, expect, it } from 'vitest';
import {
  BOOK_BLOCK_MULTIPLE,
  BOOK_MAX_BLOCK,
  MAPPING_KINDS,
  blocksNeeded,
  counts,
  coverage,
  loopView,
  makeData,
  mapping,
  runKernel,
  snapBlock,
  span,
  stepKernel,
  threads,
  validConfig,
  working,
} from './grid1d';

const range = (lo: number, hi: number) => Array.from({ length: hi - lo + 1 }, (_, k) => lo + k);

describe('the global index (book pp. 33-34, Fig. 2.9)', () => {
  it('with 256 threads per block, block 0 owns 0 to 255, block 1 owns 256 to 511, block 2 owns 512 to 767', () => {
    const all = threads(768, 256);
    for (const b of [0, 1, 2]) {
      const mine = all.filter((t) => t.block === b).map((t) => t.i);
      expect(mine[0]).toBe(b * 256);
      expect(mine[mine.length - 1]).toBe(b * 256 + 255);
      expect(mine).toHaveLength(256);
    }
  });
  it('thread 5 of block 3 owns element 773 (question q-own)', () => {
    expect(mapping('one').first(3, 5, 256)).toBe(773);
    expect(working(3, 5, 256)).toBe('3 × 256 + 5 = 773');
    expect(mapping('one').owner(773, 256)).toEqual({ block: 3, thread: 5, slot: 0 });
  });
  it('lists threads block after block, each with its two built-ins and its i', () => {
    const all = threads(10, 4);
    expect(all).toHaveLength(12);
    expect(all.map((t) => t.g)).toEqual(range(0, 11));
    expect(all[6]).toEqual({ g: 6, block: 1, thread: 2, i: 6, elements: [6] });
    expect(all.every((t) => t.i === t.block * 4 + t.thread)).toBe(true);
  });
  it('an explicit block count overrides the fewest that cover n', () => {
    expect(threads(10, 4, 5)).toHaveLength(20);
    expect(threads(10, 4, 1)).toHaveLength(4);
  });
});

describe("the book's worked launches", () => {
  it('n = 100 with blocks of 32: 4 blocks, 128 threads, 28 idle, all of them in block 3 (p. 37)', () => {
    const c = coverage({ n: 100, blockDim: 32 });
    expect(c.blocks).toBe(4);
    expect(c.launched).toBe(128);
    expect(c.working).toBe(100);
    expect(c.idle).toBe(28);
    expect(c.idleThreads.every((t) => t.block === 3)).toBe(true);
    expect(c.idleThreads.map((t) => t.thread)).toEqual(range(4, 31));
    expect(c.partial).toBe(0);
  });
  it('n = 1000 with blocks of 256: 4 blocks, 1024 threads, 1000 work, 24 do not (pp. 38-39)', () => {
    const c = coverage({ n: 1000, blockDim: 256 });
    expect([c.blocks, c.launched, c.working, c.idle]).toEqual([4, 1024, 1000, 24]);
  });
  it('block counts of section 2.6: 750 -> 3, 4000 -> 16, 2,000,000 -> 7813 (p. 39)', () => {
    expect(blocksNeeded(750, 256)).toBe(3);
    expect(blocksNeeded(4000, 256)).toBe(16);
    expect(blocksNeeded(2_000_000, 256)).toBe(7813);
  });
  it('exercise 4: 8000 elements, blocks of 1024: 8 blocks, 8192 threads (p. 41)', () => {
    expect(counts({ n: 8000, blockDim: 1024 })).toMatchObject({ blocks: 8, launched: 8192, idle: 192 });
  });
  it('exercise 9: N = 200000, blocks of 128: 1563 blocks, 200,064 threads, 200,000 run the body (p. 42)', () => {
    expect(counts({ n: 200_000, blockDim: 128 })).toEqual({ blocks: 1563, launched: 200_064, working: 200_000, idle: 64, partial: 0 });
  });
  it('idle threads are always fewer than one block when the block count is the fewest that covers n', () => {
    for (const B of [1, 2, 3, 7, 32, 33, 256])
      for (const n of [1, 2, 31, 32, 33, 100, 255, 256, 257, 1000])
        for (const kind of MAPPING_KINDS) expect(counts({ n, blockDim: B, mapping: kind }).idle).toBeLessThan(B);
  });
});

describe('the guard (book p. 37)', () => {
  it('with the guard on nothing outside the arrays is touched', () => {
    const c = coverage({ n: 100, blockDim: 32 });
    expect(c.outOfRange).toEqual([]);
    expect(c.beyond).toEqual(range(100, 127));
  });
  it('with the guard off, n = 100 and blocks of 32 touch exactly indices 100 to 127', () => {
    const c = coverage({ n: 100, blockDim: 32, guard: false });
    expect(c.outOfRange).toEqual(range(100, 127));
    expect(span(c.outOfRange)).toBe('100 to 127');
    expect(stepKernel({ n: 100, blockDim: 32, guard: false }, 3).outOfRange).toEqual(range(100, 127));
  });
  it('with the guard off, n = 1000 and blocks of 256 touch 1000 to 1023: 24 threads (question q-noguard)', () => {
    const c = coverage({ n: 1000, blockDim: 256, guard: false });
    expect(c.outOfRange).toEqual(range(1000, 1023));
    expect(c.outOfRange).toHaveLength(24);
  });
  it('a launch whose threads match n exactly has nothing to guard against', () => {
    expect(coverage({ n: 96, blockDim: 32, guard: false }).outOfRange).toEqual([]);
    expect(coverage({ n: 10, blockDim: 10 }).idle).toBe(0);
  });
  it('too few blocks leave elements unwritten', () => {
    expect(coverage({ n: 10, blockDim: 4, blocks: 2 }).unwritten).toEqual([8, 9]);
    expect(coverage({ n: 10, blockDim: 4 }).unwritten).toEqual([]);
  });
});

describe('three mappings (book exercises 1 to 3, p. 41; answers are ours)', () => {
  it('exercise 2: the first element is (b*B + t)*2 and the second is the next one', () => {
    const m = mapping('two-adjacent');
    expect(m.first(1, 2, 4)).toBe(12);
    expect(m.elements(1, 2, 4)).toEqual([12, 13]);
    expect(m.firstExpr).toBe('(blockIdx.x * blockDim.x + threadIdx.x) * 2');
    expect(working(1, 2, 4, 'two-adjacent')).toBe('(1 × 4 + 2) × 2 = 12');
  });
  it('exercise 3: the first element is b*B*2 + t and the second is B further on', () => {
    const m = mapping('two-sections');
    expect(m.first(1, 2, 4)).toBe(10);
    expect(m.elements(1, 2, 4)).toEqual([10, 14]);
    expect(m.firstExpr).toBe('blockIdx.x * blockDim.x * 2 + threadIdx.x');
    expect(m.secondExpr).toBe('i + blockDim.x');
    expect(working(1, 2, 4, 'two-sections')).toBe('1 × 4 × 2 + 2 = 10');
  });
  it('every element below n is written exactly once, by the thread owner() names, for odd and even n', () => {
    for (const kind of MAPPING_KINDS)
      for (const B of [1, 2, 3, 4, 8, 32])
        for (const n of [1, 2, 3, 9, 10, 31, 32, 33, 64, 65, 100, 101]) {
          const c = coverage({ n, blockDim: B, mapping: kind });
          expect(c.unwritten).toEqual([]);
          expect(c.writers.every((w) => w.length === 1)).toBe(true);
          const m = mapping(kind);
          for (let e = 0; e < n; e++) {
            const o = m.owner(e, B);
            expect(c.writers[e][0]).toBe(o.block * B + o.thread);
            expect(m.elements(o.block, o.thread, B)[o.slot]).toBe(e);
          }
        }
  });
  it('two adjacent elements, odd n: exactly one thread has its second element out of range', () => {
    const c = coverage({ n: 9, blockDim: 4, mapping: 'two-adjacent' });
    expect(c.blocks).toBe(2);
    expect(c.partialThreads.map((t) => [t.block, t.thread, t.elements])).toEqual([[1, 0, [8, 9]]]);
    expect(c.idle).toBe(3);
    expect(coverage({ n: 10, blockDim: 4, mapping: 'two-adjacent' }).partial).toBe(0);
  });
  it('two sections per block: when n ends inside a first section, every working thread of that block needs the second guard', () => {
    // B = 4: block 1 owns 8..15, first section 8..11, second 12..15. n = 10 ends in the first section.
    const c = coverage({ n: 10, blockDim: 4, mapping: 'two-sections' });
    expect(c.blocks).toBe(2);
    expect(c.partialThreads.map((t) => t.thread)).toEqual([0, 1]);
    expect(c.idleThreads.map((t) => t.thread)).toEqual([2, 3]);
    // n = 14 ends in the second section: threads 2 and 3 have a first element but no second.
    const d = coverage({ n: 14, blockDim: 4, mapping: 'two-sections' });
    expect(d.partialThreads.map((t) => t.thread)).toEqual([2, 3]);
    expect(d.idle).toBe(0);
  });
  it('two elements per thread need about half the threads', () => {
    expect(blocksNeeded(100, 32, 'two-adjacent')).toBe(2);
    expect(blocksNeeded(100, 32, 'two-sections')).toBe(2);
    expect(counts({ n: 100, blockDim: 32, mapping: 'two-adjacent' })).toEqual({ blocks: 2, launched: 64, working: 50, idle: 14, partial: 0 });
    expect(counts({ n: 100, blockDim: 32, mapping: 'two-sections' })).toEqual({ blocks: 2, launched: 64, working: 64, idle: 0, partial: 28 });
  });
  it('counts() in closed form agrees with listing the threads', () => {
    for (const kind of MAPPING_KINDS)
      for (const B of [1, 2, 3, 5, 8, 32])
        for (let n = 1; n <= 70; n++) {
          const c = coverage({ n, blockDim: B, mapping: kind });
          expect(counts({ n, blockDim: B, mapping: kind })).toEqual({
            blocks: c.blocks, launched: c.launched, working: c.working, idle: c.idle, partial: c.partial,
          });
        }
  });
});

describe('the kernel line by line', () => {
  const cfg = { n: 10, blockDim: 4 };
  it('before line 1 no thread has an i', () => {
    const s = stepKernel(cfg, 0);
    expect(s.threads).toHaveLength(12);
    expect(s.threads.every((t) => t.i === null && t.pass === null && t.store === null)).toBe(true);
    expect(s.C.every((v) => v === null)).toBe(true);
  });
  it('line 1 gives every thread its own i', () => {
    const s = stepKernel(cfg, 1);
    expect(s.threads.map((t) => t.i)).toEqual(range(0, 11));
    expect(s.threads.every((t) => t.pass === null)).toBe(true);
  });
  it('line 2 switches off the threads with i >= n: here 10 and 11', () => {
    const s = stepKernel(cfg, 2);
    expect(s.threads.filter((t) => t.pass === false).map((t) => t.i)).toEqual([10, 11]);
    expect(s.threads.filter((t) => t.pass).length).toBe(10);
    expect(s.C.every((v) => v === null)).toBe(true);
  });
  it('line 3: each live thread loads A[i] and B[i] and stores their sum in C[i]', () => {
    const data = makeData(10);
    const s = stepKernel(cfg, 3);
    const t6 = s.threads[6];
    expect(t6.loads).toEqual([
      { array: 'A', index: 6, inRange: true, value: data.A[6] },
      { array: 'B', index: 6, inRange: true, value: data.B[6] },
    ]);
    expect(t6.store).toEqual({ array: 'C', index: 6, inRange: true, value: data.A[6] + data.B[6] });
    expect(s.threads[10].store).toBeNull();
    expect(s.threads[11].loads).toEqual([]);
    expect(s.outOfRange).toEqual([]);
  });
  it('with the guard off the surplus threads run line 3 past the end of the arrays', () => {
    const s = stepKernel({ ...cfg, guard: false }, 3);
    expect(s.threads.every((t) => t.pass === null && t.runsBody)).toBe(true);
    expect(s.outOfRange).toEqual([10, 11]);
    expect(s.threads[11].store).toEqual({ array: 'C', index: 11, inRange: false, value: null });
    // What lands inside the arrays is unchanged.
    expect(s.C).toEqual(stepKernel(cfg, 3).C);
  });
  it("the kernel's output equals the sequential loop's, whatever the block size", () => {
    for (const n of [1, 7, 10, 24, 100])
      for (const B of [1, 2, 3, 4, 8, 32, 256]) expect(runKernel({ n, blockDim: B })).toEqual(loopView(n).map((t) => t.c));
  });
});

describe('the sequential loop and the data', () => {
  it('one trip per element, in order, each storing A[i] + B[i]', () => {
    const trips = loopView(10);
    expect(trips.map((t) => t.i)).toEqual(range(0, 9));
    expect(trips.every((t) => t.c === t.a + t.b)).toBe(true);
  });
  it('the values are small whole numbers and the same for every caller', () => {
    const d = makeData(24);
    expect(d).toEqual(makeData(24));
    expect([...d.A, ...d.B].every((v) => Number.isInteger(v) && v >= 1 && v <= 9)).toBe(true);
    expect(makeData(10).A).toEqual(d.A.slice(0, 10));
  });
});

describe('the numbers the "Where the loop went" scene states', () => {
  it('10 iterations in blocks of 4: 3 blocks, 12 threads, 2 more than there are iterations', () => {
    expect(counts({ n: 10, blockDim: 4 })).toMatchObject({ blocks: 3, launched: 12, idle: 2 });
  });
  it('the three worked threads give 1, 6 and 11', () => {
    const m = mapping('one');
    expect([m.first(0, 1, 4), m.first(1, 2, 4), m.first(2, 3, 4)]).toEqual([1, 6, 11]);
  });
  it('the guard switches off threads 10 and 11', () => {
    expect(coverage({ n: 10, blockDim: 4 }).idleThreads.map((t) => t.i)).toEqual([10, 11]);
  });
  it('a grid of exactly 10 threads (one group of 10) has no spare threads', () => {
    expect(counts({ n: 10, blockDim: 10 })).toMatchObject({ launched: 10, idle: 0 });
  });
});

describe('helpers', () => {
  it("snapBlock keeps a block size on the book's multiple of 32, between 32 and 1024", () => {
    expect(BOOK_BLOCK_MULTIPLE).toBe(32);
    expect(BOOK_MAX_BLOCK).toBe(1024);
    expect(snapBlock(1)).toBe(32);
    expect(snapBlock(100)).toBe(96);
    expect(snapBlock(250)).toBe(256);
    expect(snapBlock(1024)).toBe(1024);
    expect(snapBlock(5000)).toBe(1024);
    expect(snapBlock(100, 64)).toBe(128);
    expect(snapBlock(1000, 48)).toBe(1008);
  });
  it('span names a run of indices', () => {
    expect(span([])).toBe('none');
    expect(span([7])).toBe('7');
    expect(span([0, 1, 2, 3])).toBe('0 to 3');
  });
  it('validConfig wants whole numbers of at least 1', () => {
    expect(validConfig({ n: 10, blockDim: 4 })).toBe(true);
    expect(validConfig({ n: 0, blockDim: 4 })).toBe(false);
    expect(validConfig({ n: 10, blockDim: 2.5 })).toBe(false);
    expect(validConfig({ n: 10, blockDim: 4, blocks: 0 })).toBe(false);
  });
});
