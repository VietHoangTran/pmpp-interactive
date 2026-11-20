import { describe, expect, it } from 'vitest';
import {
  BOOK_KEYS,
  bin,
  bucketTable,
  directPass,
  exclusiveScan,
  f32bits,
  isPartitioned,
  localPass,
  localSortBlocks,
  orderedKey,
  passInput,
  placeFromCounts,
  rng,
  runsOf,
  softmaxKeys,
  splitSteps,
  stablePass,
  starts,
  storeTotals,
  summarise,
  uniformKeys,
  unitsOf,
  warpStores,
} from './radixlocal';

const row = (keys: number[]) => keys.map((k) => bin(k, 4)).join(' ');

describe('the book\'s one-bit pass (Figs. 14.4 to 14.6)', () => {
  it('destinations of Fig. 14.6', () => {
    expect(stablePass(BOOK_KEYS, 0, 1).dest).toEqual([0, 8, 1, 9, 10, 2, 11, 3, 12, 4, 13, 14, 5, 6, 15, 7]);
  });
  it('the four iterations of Fig. 14.4', () => {
    expect(row(passInput(BOOK_KEYS, 1, 1))).toBe('1100 0110 1000 1010 0110 0100 1010 0000 0011 1001 1111 0101 1001 1011 1101 0111');
    expect(row(passInput(BOOK_KEYS, 1, 2))).toBe('1100 1000 0100 0000 1001 0101 1001 1101 0110 1010 0110 1010 0011 1111 1011 0111');
    expect(row(passInput(BOOK_KEYS, 1, 3))).toBe('1000 0000 1001 1001 1010 1010 0011 1011 1100 0100 0101 1101 0110 0110 1111 0111');
    expect(row(passInput(BOOK_KEYS, 1, 4))).toBe('0000 0011 0100 0101 0110 0110 0111 1000 1001 1001 1010 1010 1011 1100 1101 1111');
  });
  it('the direct design writes those destinations, one key per thread', () => {
    const p = directPass(BOOK_KEYS, 0, 1, 4);
    expect(p.writes.filter((w) => w.block === 0).map((w) => w.addr)).toEqual([0, 8, 1, 9]);
    expect(p.out).toEqual(passInput(BOOK_KEYS, 1, 1));
  });
});

describe('local sort and the table (Figs. 14.8 and 14.9)', () => {
  const blocks = localSortBlocks(BOOK_KEYS, 0, 1, 4);
  it('Fig. 14.8: each block sorted locally', () => {
    expect(blocks.map((b) => row(b.sorted))).toEqual([
      '1100 0110 0011 1001',
      '1000 1010 1111 0101',
      '0110 1001 1011 1101',
      '0100 1010 0000 0111',
    ]);
  });
  it('Fig. 14.9: bucket sizes, digit-major table, exclusive scan, starts', () => {
    const counts = blocks.map((b) => b.counts);
    expect(counts.map((c) => c[0])).toEqual([2, 2, 1, 3]);
    expect(counts.map((c) => c[1])).toEqual([2, 2, 3, 1]);
    expect(bucketTable(counts, 'digit-major')).toEqual([2, 2, 1, 3, 2, 2, 3, 1]);
    expect(exclusiveScan(bucketTable(counts, 'digit-major'))).toEqual([0, 2, 4, 5, 8, 10, 12, 15]);
    const st = starts(counts, 'digit-major');
    expect(st.map((s) => s[0])).toEqual([0, 2, 4, 5]);
    expect(st.map((s) => s[1])).toEqual([8, 10, 12, 15]);
  });
  it('p. 342: block 2 writes its one zero, then three ones side by side; block 3 three zeros and a one', () => {
    const p = localPass(BOOK_KEYS, 0, 1, 4);
    const of = (b: number) => p.writes.filter((w) => w.block === b).map((w) => [w.thread, w.digit, w.addr]);
    expect(of(2)).toEqual([[0, 0, 4], [1, 1, 12], [2, 1, 13], [3, 1, 14]]);
    expect(of(3)).toEqual([[0, 0, 5], [1, 0, 6], [2, 0, 7], [3, 1, 15]]);
    expect(p.out).toEqual(passInput(BOOK_KEYS, 1, 1));
  });
  it('the block-major layout scans in the wrong order and the output is no longer partitioned', () => {
    const p = localPass(BOOK_KEYS, 0, 1, 4, 1, 'block-major');
    expect(isPartitioned(p.out, 0, 1)).toBe(false);
    expect([...p.out].sort((a, b) => a - b)).toEqual([...BOOK_KEYS].sort((a, b) => a - b)); // still a permutation
    expect(isPartitioned(localPass(BOOK_KEYS, 0, 1, 4).out, 0, 1)).toBe(true);
  });
  it('placeFromCounts gives the questions\' numbers', () => {
    const counts = [[3, 1], [1, 3], [2, 2], [2, 2]];
    expect(starts(counts, 'digit-major')[2][1]).toBe(12); // q-start
    const w = placeFromCounts(counts, 'digit-major').find((x) => x.block === 1 && x.slot === 2)!;
    expect(w).toEqual({ block: 1, slot: 2, digit: 1, addr: 10 }); // q-thread: 9 + (2 - 1)
    const bad = placeFromCounts([[2, 2], [2, 2], [1, 3], [3, 1]], 'block-major').sort((a, b) => a.addr - b.addr);
    expect(bad.map((x) => x.digit)).toEqual([0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1]);
  });
});

describe('a two-bit digit (Figs. 14.10 to 14.12)', () => {
  it('Fig. 14.11: locally sorted rows', () => {
    expect(localSortBlocks(BOOK_KEYS, 0, 2, 4).map((b) => row(b.sorted))).toEqual([
      '1100 1001 0110 0011',
      '1000 0101 1010 1111',
      '1001 1101 0110 1011',
      '0100 0000 1010 0111',
    ]);
  });
  it('Fig. 14.12: table rows and starts', () => {
    const counts = localSortBlocks(BOOK_KEYS, 0, 2, 4).map((b) => b.counts);
    expect([0, 1, 2, 3].map((d) => counts.map((c) => c[d]))).toEqual([[1, 1, 0, 2], [1, 1, 2, 0], [1, 1, 1, 1], [1, 1, 1, 1]]);
    const st = starts(counts, 'digit-major');
    expect([0, 1, 2, 3].map((d) => st.map((s) => s[d]))).toEqual([[0, 1, 2, 2], [4, 5, 6, 8], [8, 9, 10, 11], [12, 13, 14, 15]]);
  });
  it('Fig. 14.10: the two passes sort the keys', () => {
    expect(row(passInput(BOOK_KEYS, 2, 1))).toBe('1100 1000 0100 0000 1001 0101 1001 1101 0110 1010 0110 1010 0011 1111 1011 0111');
    expect(localPass(BOOK_KEYS, 0, 2, 4).out).toEqual(passInput(BOOK_KEYS, 2, 1));
    expect(row(passInput(BOOK_KEYS, 2, 2))).toBe('0000 0011 0100 0101 0110 0110 0111 1000 1001 1001 1010 1010 1011 1100 1101 1111');
  });
  it('r one-bit splits give the same order as sorting by the whole digit (p. 342)', () => {
    const g = rng(7);
    for (let t = 0; t < 40; t++) {
      const keys = Array.from({ length: 64 }, () => Math.floor(g() * 2 ** 16));
      const shift = Math.floor(g() * 8);
      const r = 1 + Math.floor(g() * 8);
      const steps = splitSteps(keys, shift, r);
      expect(steps).toHaveLength(r);
      expect(steps[r - 1]).toEqual(localSortBlocks(keys, shift, r, 64)[0].sorted);
    }
  });
});

describe('coarsening (Fig. 14.13)', () => {
  it('two blocks of eight keys, two keys per local bucket', () => {
    const b = localSortBlocks(BOOK_KEYS, 0, 2, 8);
    expect(b.map((x) => row(x.sorted))).toEqual([
      '1100 1000 1001 0101 0110 1010 0011 1111',
      '0100 0000 1001 1101 0110 1010 1011 0111',
    ]);
    expect(b.map((x) => x.counts)).toEqual([[2, 2, 2, 2], [2, 2, 2, 2]]);
  });
  it('four threads with two keys each: two store instructions per block, two runs each', () => {
    const p = localPass(BOOK_KEYS, 0, 2, 4, 2);
    const ws = warpStores(p, 32, 4, 32);
    expect(ws.map((w) => [w.block, w.store, w.addrs.join(',')])).toEqual([
      [0, 0, '0,1,4,5'],
      [0, 1, '8,9,12,13'],
      [1, 0, '2,3,6,7'],
      [1, 1, '10,11,14,15'],
    ]);
    expect(ws.map((w) => w.runs)).toEqual([2, 2, 2, 2]);
    expect(summarise(warpStores(localPass(BOOK_KEYS, 0, 2, 4, 1), 32, 4, 32)).meanRuns).toBe(3.5);
    expect(p.out).toEqual(passInput(BOOK_KEYS, 2, 1));
  });
});

describe('runs and units (ch14-facts A10; ch06-facts A2, A4)', () => {
  it('q-sectors: 12 keys from 16 bytes into a sector, 20 from a boundary: 5 sectors', () => {
    const zeros = Array.from({ length: 12 }, (_, i) => 4 + i); // byte 16 onward
    const ones = Array.from({ length: 20 }, (_, i) => 64 + i); // byte 256 onward
    expect(unitsOf([...zeros, ...ones], 4, 32)).toHaveLength(5);
    expect(runsOf([...zeros, ...ones])).toBe(2);
    expect(unitsOf(Array.from({ length: 32 }, (_, i) => i), 4, 32)).toHaveLength(4);
  });
  it('a one-bit direct scatter costs 4 to 7 sectors per warp, 1 to 4 lines, and both ends occur', () => {
    const seen = new Set<number>();
    for (let s = 1; s <= 30; s++) {
      const p = directPass(uniformKeys(1024, s), 0, 1, 256);
      for (const w of warpStores(p, 32, 4, 32)) {
        expect(w.runs).toBeLessThanOrEqual(2);
        expect(w.units.length).toBeGreaterThanOrEqual(4);
        expect(w.units.length).toBeLessThanOrEqual(7);
        seen.add(w.units.length);
      }
      for (const w of warpStores(p, 32, 4, 128)) {
        expect(w.units.length).toBeGreaterThanOrEqual(1);
        expect(w.units.length).toBeLessThanOrEqual(4);
      }
    }
    expect(seen.has(7)).toBe(true);
    expect(Math.min(...seen)).toBeLessThanOrEqual(5);
  });
  it('the direct scatter makes at most min(warp width, 2^r) runs per warp; 8 bits nears a sector per key', () => {
    const keys = uniformKeys(4096, 3);
    for (const r of [1, 2, 4, 8]) {
      for (const W of [32, 64]) {
        for (const w of warpStores(directPass(keys, 0, r, 256), W, 4, 32)) expect(w.runs).toBeLessThanOrEqual(Math.min(W, 2 ** r));
      }
    }
    const s8 = summarise(warpStores(directPass(keys, 0, 8, 256), 32, 4, 32));
    expect(s8.meanUnits).toBeGreaterThan(28);
    expect(s8.meanMin).toBe(4);
  });
  it('the local sort cuts runs to the local buckets a warp straddles', () => {
    const keys = uniformKeys(4096, 4);
    for (const r of [1, 2, 4, 8]) {
      const p = localPass(keys, 0, r, 256);
      for (const w of warpStores(p, 32, 4, 32)) expect(w.runs).toBeLessThanOrEqual(new Set(w.digits).size);
      const d = summarise(warpStores(directPass(keys, 0, r, 256), 32, 4, 32));
      const l = summarise(warpStores(p, 32, 4, 32));
      expect(l.meanUnits).toBeLessThanOrEqual(d.meanUnits);
    }
    // r = 4: 16 keys per local bucket on average, so a warp touches about three buckets.
    expect(summarise(warpStores(localPass(keys, 0, 4, 256), 32, 4, 32)).meanRuns).toBeLessThan(4);
  });
  it('a block no wider than a warp writes the same addresses in both designs, only in another lane order', () => {
    const keys = uniformKeys(256, 5);
    const d = warpStores(directPass(keys, 0, 1, 32), 32, 4, 32);
    const l = warpStores(localPass(keys, 0, 1, 32), 32, 4, 32);
    d.forEach((w, i) => expect([...w.addrs].sort((a, b) => a - b)).toEqual([...l[i].addrs].sort((a, b) => a - b)));
  });
  it('every pass of every design writes a permutation that equals the stable partition', () => {
    const keys = uniformKeys(1000, 6);
    for (const r of [1, 3, 8])
      for (const c of [1, 3]) expect(localPass(keys, 5, r, 64, c).out).toEqual(stablePass(keys, 5, r).out);
  });
});

describe('storeTotals, the lean path the sweeps use', () => {
  it('agrees with warpStores and summarise on every design, width and unit', () => {
    for (const [n, B, c, r, W, unit, seed] of [
      [1000, 256, 1, 1, 32, 32, 1],
      [1000, 256, 3, 4, 64, 64, 2],
      [777, 128, 2, 8, 32, 128, 3],
      [16, 4, 2, 2, 32, 32, 4],
      [4096, 256, 1, 8, 32, 16, 5],
    ]) {
      const keys = uniformKeys(n, seed);
      for (const design of ['direct', 'local'] as const) {
        const cc = design === 'direct' ? 1 : c;
        const p = design === 'direct' ? directPass(keys, 3, r, B) : localPass(keys, 3, r, B, cc);
        const s = summarise(warpStores(p, W, 4, unit));
        const t = storeTotals(design, keys, 3, r, B, cc, W, 4, unit);
        expect(t.stores).toBe(s.stores);
        expect(t.runs / t.stores).toBeCloseTo(s.meanRuns, 12);
        expect(t.units / t.stores).toBeCloseTo(s.meanUnits, 12);
        expect(t.min / t.stores).toBeCloseTo(s.meanMin, 12);
        expect(t.out).toEqual(p.out);
      }
    }
  });
});

describe('float keys (ch14-facts A3, E1)', () => {
  it('the transform of the facts table', () => {
    expect(orderedKey(0xff800000, 32, 'asc')).toBe(0x007fffff); // -inf
    expect(orderedKey(f32bits(-1), 32, 'asc')).toBe(0x407fffff);
    expect(orderedKey(0x80000000, 32, 'asc')).toBe(0x80000000); // -0.0 joins +0.0
    expect(orderedKey(0, 32, 'asc')).toBe(0x80000000);
    expect(orderedKey(f32bits(1), 32, 'asc')).toBe(0xbf800000);
    expect(orderedKey(f32bits(1), 32, 'desc')).toBe(0x407fffff);
    expect(orderedKey(0x7f800000, 32, 'desc')).toBe(0x007fffff); // +inf first when descending
  });
  it('softmax keys sort the probabilities largest first, and their top byte takes few values', () => {
    const k = softmaxKeys(4096, 1);
    const sorted = passInput(k, 8, 4);
    for (let i = 1; i < sorted.length; i++) expect(sorted[i - 1]).toBeLessThanOrEqual(sorted[i]);
    // The top byte is the sign and seven exponent bits: a dozen values at most, four of them hold most keys.
    const top = new Map<number, number>();
    for (const x of k) top.set(x >>> 24, (top.get(x >>> 24) ?? 0) + 1);
    expect(top.size).toBeLessThanOrEqual(12);
    const four = [...top.values()].sort((a, b) => b - a).slice(0, 4).reduce((a, b) => a + b, 0);
    expect(four / k.length).toBeGreaterThan(0.75);
    const low = new Set(k.map((x) => x & 255));
    expect(low.size).toBeGreaterThan(200);
  });
});
