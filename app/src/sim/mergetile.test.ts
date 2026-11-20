import { describe, expect, it } from 'vitest';
import {
  BOOK_EXAMPLE,
  PRESETS,
  RUNNING,
  bookTiled,
  coRankByMerge,
  exactTiled,
  fromPattern,
  iterationsFor,
  mergePathSearch,
  mergeStable,
  mergeWith,
  outputsPerBlock,
  sortedRandom,
  tag,
  type TiledRun,
} from './mergetile';

const A = [...RUNNING.A];
const B = [...RUNNING.B];

/** Every iteration of every block keeps the book's bookkeeping honest. */
function checkInvariants(run: TiledRun, tile: number) {
  for (const blk of run.blocks) {
    let cDone = 0;
    let aDone = 0;
    let bDone = 0;
    const cLength = blk.cNext - blk.cCurr;
    for (const it of blk.iterations) {
      expect(it.aStart).toBe(blk.aCurr + aDone);
      expect(it.bStart).toBe(blk.bCurr + bDone);
      expect(it.cStart).toBe(blk.cCurr + cDone);
      // The outputs of an iteration: a full tile, or what is left of the block's slice.
      expect(it.cOut).toBe(Math.min(tile, cLength - cDone));
      expect(it.aUsed + it.bUsed).toBe(it.cOut);
      // The tail guard of p. 320: never load past the block's slices.
      expect(it.aLoaded).toBe(Math.min(tile, blk.aNext - it.aStart));
      expect(it.bLoaded).toBe(Math.min(tile, blk.bNext - it.bStart));
      expect(it.aStart + it.aLoaded).toBeLessThanOrEqual(blk.aNext);
      expect(it.bStart + it.bLoaded).toBeLessThanOrEqual(blk.bNext);
      expect(it.aUsed).toBeLessThanOrEqual(it.aLoaded);
      expect(it.bUsed).toBeLessThanOrEqual(it.bLoaded);
      // The threads' shares add up to the block's.
      expect(it.threads.reduce((s, t) => s + t.a, 0)).toBe(it.aUsed);
      expect(it.threads.reduce((s, t) => s + t.b, 0)).toBe(it.bUsed);
      cDone += it.cOut;
      aDone += it.aUsed;
      bDone += it.bUsed;
    }
  }
}

describe('the running example (Fig. 13.1, p. 304)', () => {
  it('merges to the book\'s C, with A winning both ties', () => {
    const C = mergeStable(A, B);
    expect(C.map((x) => x.key)).toEqual([1, 7, 7, 8, 9, 10, 10, 10, 12]);
    expect(C.map(tag)).toEqual(['A0', 'A1', 'B0', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3']);
  });
  it('with < instead of <= the keys are the same but B wins the ties across lists', () => {
    const C = mergeWith(A, B, '<');
    expect(C.map((x) => x.key)).toEqual([1, 7, 7, 8, 9, 10, 10, 10, 12]);
    expect(C.map(tag)).toEqual(['A0', 'B0', 'A1', 'A2', 'A3', 'B1', 'B2', 'A4', 'B3']);
    // Order within each list is kept either way.
    const order = (src: 'A' | 'B') => C.filter((x) => x.src === src).map((x) => x.idx);
    expect(order('A')).toEqual([0, 1, 2, 3, 4]);
    expect(order('B')).toEqual([0, 1, 2, 3]);
  });
  it('co-ranks for k = 0 to 9 are the ones ch13-facts E1 lists', () => {
    expect(Array.from({ length: 10 }, (_, k) => coRankByMerge(k, A, B))).toEqual([0, 1, 2, 2, 3, 4, 5, 5, 5, 5]);
  });
  it('three threads of three outputs: thread 1 needs A[2..4] and nothing from B', () => {
    const [i3, i6] = [coRankByMerge(3, A, B), coRankByMerge(6, A, B)];
    expect([i3, 3 - i3]).toEqual([2, 1]);
    expect([i6, 6 - i6]).toEqual([5, 1]);
    expect(mergeStable(A, B).slice(3, 6).map(tag)).toEqual(['A2', 'A3', 'A4']);
  });
  it('rejects a rank outside 0..m+n', () => {
    expect(() => coRankByMerge(10, A, B)).toThrow(RangeError);
    expect(() => mergePathSearch(-1, A, B)).toThrow(RangeError);
  });
});

describe('the merge-path search agrees with counting a stable merge (ch13-facts E1)', () => {
  it('on every k of 2,000 small random pairs with many duplicate keys', () => {
    for (let s = 0; s < 2000; s++) {
      const a = sortedRandom(s % 13, 7, 2 * s + 1);
      const b = sortedRandom((s * 7) % 13, 7, 2 * s + 2);
      for (let k = 0; k <= a.length + b.length; k++) expect(mergePathSearch(k, a, b).i).toBe(coRankByMerge(k, a, b));
    }
  });
  it('reads two elements per probe, about log2 of the range', () => {
    const a = sortedRandom(1024, 1e6, 3);
    const b = sortedRandom(1024, 1e6, 4);
    const { reads } = mergePathSearch(1024, a, b);
    expect(reads % 2).toBe(0);
    expect(reads / 2).toBeLessThanOrEqual(Math.ceil(Math.log2(1025)));
  });
});

describe('the printed ceil (notes item 14, ch13-facts A4)', () => {
  it('the book\'s example: 4,000 outputs per block, 4 iterations of 1,024 meant, 3 printed', () => {
    expect(outputsPerBlock(64000, 16, 'printed')).toBe(4000);
    expect(outputsPerBlock(64000, 16, 'corrected')).toBe(4000);
    expect(iterationsFor(4000, 1024, 'corrected')).toBe(4);
    expect(iterationsFor(4000, 1024, 'printed')).toBe(3);
  });
  it('9 outputs over 4 threads: 2 each printed, 3 each corrected', () => {
    expect(outputsPerBlock(9, 4, 'printed')).toBe(2);
    expect(outputsPerBlock(9, 4, 'corrected')).toBe(3);
  });
  it('a printed block split that does not divide m + n leaves the end unwritten', () => {
    const { A: a, B: b } = fromPattern('ABABABABAB');
    const run = bookTiled(a, b, 4, 2, 'printed');
    expect(run.totals.missed).toBe(2);
    expect(run.output.slice(8)).toEqual([null, null]);
    expect(run.output.slice(0, 8)).toEqual(mergeStable(a, b).slice(0, 8));
  });
});

describe('the book scheme (Figs. 13.11 to 13.13) with the corrected ceil', () => {
  it('reproduces the stable merge exactly on the running example, for any blocks, tile and threads', () => {
    for (const blocks of [1, 2, 3, 4])
      for (const [tile, threads] of [[1, 1], [2, 1], [2, 2], [3, 3], [4, 2], [8, 4]]) {
        const run = bookTiled(A, B, blocks, tile, 'corrected', threads);
        expect(run.output).toEqual(mergeStable(A, B));
        expect(run.totals.missed).toBe(0);
        checkInvariants(run, tile);
      }
  });
  it('reproduces the stable merge exactly on 400 random inputs with repeated keys', () => {
    for (let s = 0; s < 400; s++) {
      const a = sortedRandom(s % 61, 20, 3 * s + 1);
      const b = sortedRandom((s * 17) % 53, 20, 3 * s + 2);
      const blocks = 1 + (s % 5);
      const threads = 1 + (s % 4);
      const tile = threads * (1 + (s % 6));
      const run = bookTiled(a, b, blocks, tile, 'corrected', threads);
      expect(run.output).toEqual(mergeStable(a, b));
      checkInvariants(run, tile);
    }
  });
  it('loads close to twice what it uses on long random inputs', () => {
    const a = sortedRandom(20000, 1e6, 11);
    const b = sortedRandom(20000, 1e6, 12);
    const run = bookTiled(a, b, 4, 64, 'corrected', 8);
    expect(run.output).toEqual(mergeStable(a, b));
    expect(run.totals.used).toBe(40000);
    expect(run.totals.minimum).toBe(40000);
    const ratio = run.totals.loaded / run.totals.used;
    expect(ratio).toBeGreaterThan(1.95);
    expect(ratio).toBeLessThan(2);
  });
  it('flags stale slots only on an iteration that is not the last and has a partial tile', () => {
    const a = sortedRandom(3000, 50, 21);
    const b = sortedRandom(1000, 50, 22);
    const run = bookTiled(a, b, 3, 32, 'corrected', 4);
    for (const blk of run.blocks)
      blk.iterations.forEach((it, n) => {
        const last = n === blk.iterations.length - 1;
        expect(it.staleSlots).toBe(last ? 0 : 64 - it.aLoaded - it.bLoaded);
      });
  });
});

describe('the book\'s example launch (p. 317)', () => {
  const a = sortedRandom(BOOK_EXAMPLE.m, 1e6, 317);
  const b = sortedRandom(BOOK_EXAMPLE.n, 1e6, 318);
  const { blocks, tile, threads } = BOOK_EXAMPLE;
  it('corrected: 4 iterations per block and the full merge', () => {
    const run = bookTiled(a, b, blocks, tile, 'corrected', threads);
    expect(run.blocks.every((blk) => blk.cNext - blk.cCurr === 4000 && blk.iterations.length === 4)).toBe(true);
    expect(run.blocks.every((blk) => blk.iterations.slice(0, 3).every((it) => it.cOut === 1024))).toBe(true);
    expect(run.blocks.every((blk) => blk.iterations[3].cOut === 928)).toBe(true);
    expect(run.totals.missed).toBe(0);
    expect(run.output).toEqual(mergeStable(a, b));
  });
  it('printed: 3 iterations per block, 928 outputs per block and 14,848 in all never written', () => {
    const run = bookTiled(a, b, blocks, tile, 'printed', threads);
    expect(run.blocks.every((blk) => blk.iterations.length === 3 && blk.missed === 928)).toBe(true);
    expect(run.totals.missed).toBe(14848);
    expect(run.output.filter((x) => x === null).length).toBe(14848);
    // What was written is right; the gap is the last 928 of each block's slice.
    const want = mergeStable(a, b);
    for (const blk of run.blocks) {
      expect(run.output.slice(blk.cCurr, blk.cCurr + 3072)).toEqual(want.slice(blk.cCurr, blk.cCurr + 3072));
      expect(run.output.slice(blk.cCurr + 3072, blk.cNext).every((x) => x === null)).toBe(true);
    }
  });
});

describe('the exact-load scheme (Exercise 3; CUB, rocPRIM, MLX, PyTorch MPS)', () => {
  it('reproduces the stable merge exactly, small and random', () => {
    for (const t of [1, 2, 3, 4, 9, 20]) expect(exactTiled(A, B, t).output).toEqual(mergeStable(A, B));
    for (let s = 0; s < 300; s++) {
      const a = sortedRandom(s % 47, 15, 5 * s + 1);
      const b = sortedRandom((s * 13) % 41, 15, 5 * s + 2);
      const run = exactTiled(a, b, 1 + (s % 9));
      expect(run.output).toEqual(mergeStable(a, b));
    }
  });
  it('loads each element exactly once, in contiguous ranges that add up', () => {
    const a = sortedRandom(20000, 1e6, 11);
    const b = sortedRandom(20000, 1e6, 12);
    const run = exactTiled(a, b, 64);
    expect(run.totals.loaded / run.totals.used).toBe(1);
    expect(run.totals.loaded).toBe(run.totals.minimum);
    let a0 = 0;
    let b0 = 0;
    for (const t of run.tiles) {
      expect(t.aRange[0]).toBe(a0);
      expect(t.bRange[0]).toBe(b0);
      expect(t.aRange[1] - t.aRange[0] + (t.bRange[1] - t.bRange[0])).toBe(t.cRange[1] - t.cRange[0]);
      expect(t.loaded).toBe(t.used);
      a0 = t.aRange[1];
      b0 = t.bRange[1];
    }
    expect([a0, b0]).toEqual([20000, 20000]);
  });
  it('pays for it with one search per tile boundary, counted in element reads', () => {
    const run = exactTiled(A, B, 3);
    expect(run.tiles.map((t) => [t.aRange, t.bRange])).toEqual([
      [[0, 2], [0, 1]],
      [[2, 5], [1, 1]],
      [[5, 5], [1, 4]],
    ]);
    expect(run.totals.extraSearchReads).toBe(run.tiles.reduce((s, t) => s + t.extraSearchReads, 0));
    expect(run.tiles[2].extraSearchReads).toBe(0); // the end of C needs no search
    expect(run.totals.extraSearchReads).toBeGreaterThan(0);
  });
});

describe('scene presets: one block, tile 8, two threads', () => {
  it('uneven: iterations use 5+3, 2+6, 7+1 and 2+6', () => {
    const p = PRESETS.uneven;
    const run = bookTiled(p.A, p.B, p.blocks, p.tile, 'corrected', p.threads);
    expect(run.output).toEqual(mergeStable(p.A, p.B));
    const its = run.blocks[0].iterations;
    expect(its.map((it) => [it.aUsed, it.bUsed])).toEqual([[5, 3], [2, 6], [7, 1], [2, 6]]);
    expect(its.map((it) => [it.aLoaded, it.bLoaded])).toEqual([[8, 8], [8, 8], [8, 7], [2, 6]]);
    expect(its[0].threads).toEqual([{ a: 3, b: 1 }, { a: 2, b: 2 }]);
    expect(run.totals).toEqual({ loaded: 55, used: 32, minimum: 32, missed: 0 });
    expect(exactTiled(p.A, p.B, p.tile).totals.loaded).toBe(32);
  });
  it('allFromA: the first two iterations load 16 and use 8, all from A', () => {
    const p = PRESETS.allFromA;
    const run = bookTiled(p.A, p.B, p.blocks, p.tile, 'corrected', p.threads);
    expect(run.output).toEqual(mergeStable(p.A, p.B));
    const its = run.blocks[0].iterations;
    expect(its.map((it) => [it.aLoaded, it.bLoaded, it.aUsed, it.bUsed])).toEqual([
      [8, 8, 8, 0],
      [8, 8, 8, 0],
      [0, 8, 0, 8],
    ]);
    expect(run.totals).toEqual({ loaded: 40, used: 24, minimum: 24, missed: 0 });
  });
  it('shortTileEarly: iteration 1 of 5 has a 2-element A tile and 6 stale slots', () => {
    const p = PRESETS.shortTileEarly;
    const run = bookTiled(p.A, p.B, p.blocks, p.tile, 'corrected', p.threads);
    expect(run.output).toEqual(mergeStable(p.A, p.B));
    const its = run.blocks[0].iterations;
    expect(its).toHaveLength(5);
    expect([its[1].aLoaded, its[1].aUsed, its[1].bUsed, its[1].staleSlots]).toEqual([2, 2, 6, 6]);
    expect(its[0].staleSlots).toBe(0);
  });
  it('rejects a tile that the threads do not divide', () => {
    expect(() => bookTiled(A, B, 1, 6, 'corrected', 4)).toThrow(RangeError);
  });
});
