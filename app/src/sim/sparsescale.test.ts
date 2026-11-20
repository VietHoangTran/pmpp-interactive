import { describe, expect, it } from 'vitest';
import {
  FIG1713_LENGTHS,
  PRESETS,
  amortiseIterations,
  breakEvenDensity,
  bytesPerPosition,
  colsOf,
  csr,
  ell,
  hybrid,
  iterPtr,
  jds,
  laneStates,
  maxOf,
  nnzOf,
  preset,
  rowLengths,
  rowsInIteration,
  sell,
  sliceChoices,
  sortOrder,
  sortedLengths,
  spills,
  storage,
  traffic,
  warpStats,
  warps,
} from './sparsescale';

const W4: { value: number; index: number } = { value: 4, index: 4 };

describe("the book's 1000 × 1000 example (pp. 412–413, ch17-facts E1)", () => {
  const len = rowLengths(preset('book').params);
  it('keeps 1% non-zero: 10,000 in 1,000 rows, one row of 200, the rest 9 or 10', () => {
    expect(len).toHaveLength(1000);
    expect(nnzOf(len)).toBe(10_000);
    expect(len.filter((l) => l === 200)).toHaveLength(1);
    expect(len.filter((l) => l !== 200).every((l) => l === 9 || l === 10)).toBe(true);
    expect(maxOf(len)).toBe(200);
  });
  it('the book cannot have "fewer than 10" for the others and 1% together', () => {
    expect(999 * 9 + 200).toBeLessThan(10_000);
  });
  it('CSR stores 21,001 numbers (2.1% of dense), ELL 400,000 (40%), ratio 19.05', () => {
    const c = csr(len, W4);
    const e = ell(len, W4);
    expect(c.numbers).toBe(21_001);
    expect(e.numbers).toBe(400_000);
    expect(c.numbers / 1_000_000).toBeCloseTo(0.021, 4);
    expect(e.numbers / 1_000_000).toBeCloseTo(0.4, 6);
    expect(e.numbers / c.numbers).toBeCloseTo(19.05, 2);
    expect(e.pads).toBe(200_000 - 10_000);
  });
});

describe('Fig. 17.13: hybrid ELL-COO (p. 414, notes "Unclear" item 2)', () => {
  const len = [...FIG1713_LENGTHS];
  it('has 18 non-zeros and pure ELL is 5 wide with 22 pads', () => {
    expect(nnzOf(len)).toBe(18);
    const e = ell(len, W4);
    expect(e.width).toBe(5);
    expect(e.pads).toBe(22);
  });
  it('at K = 2: 3 pads and 5 COO entries, 3 from row 1 and 2 from row 6 (not row 2)', () => {
    const h = hybrid(len, 2, W4);
    expect(h.pads).toBe(3);
    expect(h.coo).toBe(5);
    expect(spills(len, 2)).toEqual([0, 3, 0, 0, 0, 0, 2, 0]);
    expect(h.numbers).toBe(2 * 8 * 2 + 3 * 5);
  });
  it('K at the longest row is pure ELL, K = 0 is pure COO', () => {
    expect(hybrid(len, 5, W4).pads).toBe(ell(len, W4).pads);
    expect(hybrid(len, 5, W4).coo).toBe(0);
    expect(hybrid(len, 0, W4).coo).toBe(18);
    expect(hybrid(len, 0, W4).numbers).toBe(3 * 18);
  });
});

describe('q-hybrid-k: 1000 rows of 16 and 2 rows of 400', () => {
  const len = [...Array(1000).fill(16), 400, 400];
  it('at K = 16, 768 entries go to COO and no pads are left', () => {
    const h = hybrid(len, 16, W4);
    expect(h.coo).toBe(768);
    expect(h.pads).toBe(0);
  });
  it('pure ELL would hold 384,000 pad slots', () => {
    expect(ell(len, W4).pads).toBe(384_000);
  });
});

describe('JDS (Fig. 17.14, exercises 2 and 5)', () => {
  // Fig. 17.14's letter matrix has rows of 1, 4, 2, 3, 2, 3 non-zeros.
  const fig1714 = [1, 4, 2, 3, 2, 3];
  it('sorts longest first, ties in row order: row = [1, 3, 5, 2, 4, 0]', () => {
    expect(sortOrder(fig1714)).toEqual([1, 3, 5, 2, 4, 0]);
  });
  it('iterPtr = [0, 6, 11, 14, 15]', () => {
    expect(iterPtr(sortedLengths(fig1714))).toEqual([0, 6, 11, 14, 15]);
    expect(rowsInIteration(sortedLengths(fig1714))).toEqual([6, 5, 3, 1]);
  });
  it('stores 2z + m + K + 1 numbers and no pads', () => {
    const j = jds(fig1714, W4);
    expect(j.numbers).toBe(2 * 15 + 6 + 4 + 1);
    expect(j.pads).toBe(0);
  });
  it('the exercise 5 condition: sorted row r has a t-th element exactly when r < iterPtr[t+1] − iterPtr[t]', () => {
    const s = sortedLengths(fig1714);
    const n = rowsInIteration(s);
    for (let t = 0; t < 4; t++) for (let r = 0; r < 6; r++) expect(r < n[t]).toBe(s[r] > t);
  });
});

describe('sliced ELL', () => {
  const len = rowLengths(preset('spread').params);
  it('with S = rows and no sort it is ELL (same slots, same pads; plus 2 slice offsets)', () => {
    const s = sell(len, len.length, false, W4);
    const e = ell(len, W4);
    expect(s.pads).toBe(e.pads);
    expect(s.numbers).toBe(e.numbers + 2);
  });
  it('sorting first never adds pads, and S = 1 has none', () => {
    for (const S of sliceChoices(len.length)) expect(sell(len, S, true, W4).pads).toBeLessThanOrEqual(sell(len, S, false, W4).pads);
    expect(sell(len, 1, false, W4).pads).toBe(0);
  });
  it('slice choices are powers of two below the row count, then the row count', () => {
    expect(sliceChoices(8)).toEqual([1, 2, 4, 8]);
    expect(sliceChoices(1000).at(-1)).toBe(1000);
    expect(sliceChoices(1000).at(-2)).toBe(512);
  });
});

describe('generators', () => {
  it('are deterministic and give the stated means', () => {
    for (const p of PRESETS) expect(rowLengths(p.params)).toEqual(rowLengths(p.params));
    expect(nnzOf(rowLengths(preset('uniform').params))).toBe(1024 * 16);
    expect(nnzOf(rowLengths(preset('spread').params)) / 1024).toBeCloseTo(16, 0);
    expect(nnzOf(rowLengths(preset('skewed').params))).toBe(4096 * 16);
    const pw = rowLengths(preset('power').params);
    expect(nnzOf(pw) / pw.length).toBeGreaterThan(12);
    expect(nnzOf(pw) / pw.length).toBeLessThan(20);
    expect(maxOf(pw)).toBeLessThanOrEqual(colsOf(preset('power').params));
  });
  it('skewed: exactly one row in 512 is 16 times the mean', () => {
    const len = rowLengths(preset('skewed').params);
    expect(len.filter((l) => l === 256)).toHaveLength(4096 / 512);
    expect(len.filter((l) => l !== 256).every((l) => l === 15 || l === 16)).toBe(true);
  });
});

describe('warps of a per-row kernel (through divergence.ts loopBatch)', () => {
  it('Fig. 17.1 under one thread per row on four lanes: 8 of 12 lane-iterations work (67%)', () => {
    const s = warpStats([2, 3, 2, 1], 4);
    expect(s.spent).toBe(3);
    expect(s.useful).toBe(8);
    expect(s.efficiency).toBeCloseTo(8 / 12, 10);
  });
  it('sorted order never lowers SIMD efficiency', () => {
    const cases = [
      ...PRESETS.map((p) => rowLengths(p.params)),
      [1, 1, 10],
      [10, 1, 1],
      [5, 4, 3, 2, 1],
      Array.from({ length: 77 }, (_, i) => (i * 37) % 23),
    ];
    for (const len of cases)
      for (const width of [2, 4, 32, 64]) {
        const t = warps(len, width, 2);
        expect(t.sorted.efficiency).toBeGreaterThanOrEqual(t.original.efficiency - 1e-12);
        expect(t.sorted.spent).toBeLessThanOrEqual(t.original.spent);
        expect(t.sorted.useful).toBe(t.original.useful);
      }
  });
  it("the book's example: one warp runs 200 iterations unsorted or sorted", () => {
    const len = rowLengths(preset('book').params);
    const t = warps(len, 32, 10);
    expect(t.original.longest).toBe(200);
    expect(t.original.spent).toBe(31 * 10 + 200);
    // sorted: the 200 and 31 tens, then 24 more warps of tens, one warp that ends the tens, six of nines
    expect(t.sorted.spent).toBe(200 + 25 * 10 + 6 * 9);
    expect(t.hybrid.longest).toBe(10);
  });
  it('lane states: work while the row lasts, then done; absent past the last row', () => {
    expect(laneStates([2, 3, 2, 1], 4, 0, 1)).toEqual(['work', 'work', 'work', 'done']);
    expect(laneStates([2, 3, 2, 1], 4, 0, 2)).toEqual(['done', 'work', 'done', 'done']);
    expect(laneStates([2, 3], 4, 0, 0)).toEqual(['work', 'work', 'absent', 'absent']);
  });
});

describe('requested segments (through coalesce.ts warpAccess)', () => {
  it('16 non-zeros per row, 32-byte segments: CSR asks for 32 segments per array per iteration, ELL for 4', () => {
    const len = Array(32).fill(16);
    const c = traffic('csr', len, 32, 32, W4);
    const e = traffic('ell', len, 32, 32, W4);
    expect(c.requested).toBe(16 * 2 * 32);
    expect(e.requested).toBe(16 * 2 * 4);
    // a perfect cache leaves the same bytes for both
    expect(c.distinct).toBe(e.distinct);
    expect(e.efficiency).toBe(1);
  });
  it('JDS without pads is as coalesced as ELL on equal rows', () => {
    const len = Array(64).fill(8);
    expect(traffic('jds', len, 32, 32, W4).requested).toBe(traffic('ell', len, 32, 32, W4).requested);
  });
  it('4-bit values are read a byte at a time without losing a segment', () => {
    const t = traffic('ell', Array(64).fill(1), 32, 32, { value: 0.5, index: 2 });
    // 64 values in 32 bytes: one segment per warp; 64 indices in 128 bytes: two per warp
    expect(t.requested).toBe(2 * 1 + 2 * 2);
  });
});

describe('bytes per position and break-even (start page)', () => {
  it('fp16 pruned to 10%, int32 indices: 0.6 bytes per position, plus a negligible row pointer', () => {
    expect(bytesPerPosition(0.1, 2, 4, Infinity)).toBeCloseTo(0.6, 12);
    expect(bytesPerPosition(0.1, 2, 4, 4096)).toBeCloseTo(0.6 + 4 / 4096, 12);
    expect(2 / bytesPerPosition(0.1, 2, 4, Infinity)).toBeCloseTo(3.33, 2);
    expect(bytesPerPosition(0.1, 2, 4, Infinity)).toBeGreaterThan(0.5); // 4-bit dense reads less
  });
  it('break-even of 4-bit dense against 16-bit values and 16-bit indices is 12.5%', () => {
    expect(breakEvenDensity(0.5, 2, 2)).toBeCloseTo(0.125, 12);
  });
});

describe('amortising preprocessing', () => {
  it('q-amortise: a conversion worth 30 SpMVs and a 0.6× SpMV pay off after 75 iterations', () => {
    expect(amortiseIterations(30, 0.6)).toBeCloseTo(75, 9);
    expect(amortiseIterations(30, 1)).toBe(Infinity);
  });
});

describe('the storage table', () => {
  it('collects every format with consistent byte counts', () => {
    const len = [...FIG1713_LENGTHS];
    const s = storage(len, 8, W4, 2, 4, true);
    expect(s.nnz).toBe(18);
    expect(s.stores.dense.numbers).toBe(64);
    expect(s.stores.ell.numbers).toBe(80); // more than dense, as on the 4 × 4 example (p. 412)
    expect(s.stores.csr.bytes).toBe(18 * 8 + 9 * 4);
    expect(s.stores.hybrid.coo).toBe(5);
  });
});
