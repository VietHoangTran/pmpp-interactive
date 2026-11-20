import { describe, expect, it } from 'vitest';
import {
  EXERCISE1, FIG17_1, FIG17_13, FIG17_14, FIG17_6_ORDER, ORDER_TERMS, PRESETS, ROW0_PRODUCTS,
  alignedStride, build, cooOutcomes, cooUpdates, exercise2Count, floatOutcomes, floatSum, idleLanes, interleavings,
  longestRow, matrixRuns, nnz, permutations, rmwDone, rmwInit, rmwStep, rowLengths, runTrace, shuffledOrder,
  spmvRef, spmvTrace, storage, summarize, toCOO, toCSC, toCSR, toELL, toHybrid, toJDS, toggle, xRuns,
  type FormatId, type Matrix,
} from './sparse';

// Every array here is the book's figure (notes/ch17.md, "Figures") and the facts file's
// rebuild of it (notes/today/ch17-facts.md E1). Pads are -1 in colIdx and 0 in value, the
// libraries' convention (facts A6), where the book draws "*".
const P = -1;
const reads = (id: FormatId, m: Matrix, o: Parameters<typeof spmvTrace>[1] = {}) =>
  spmvTrace(build(id, m), o).steps.map((s) => s.lanes.map((l) => l.index));

describe('the presets', () => {
  it('Fig. 17.1 is the 4 x 4 running example with 8 non-zeros', () => {
    expect(FIG17_1.rows).toBe(4);
    expect(nnz(FIG17_1)).toBe(8);
    expect(rowLengths(FIG17_1)).toEqual([2, 3, 2, 1]);
  });
  it('Fig. 17.13 has row lengths 1, 5, 2, 2, 1, 2, 4, 1 (18 non-zeros); Fig. 17.14 has 15 lettered ones', () => {
    expect(rowLengths(FIG17_13)).toEqual([1, 5, 2, 2, 1, 2, 4, 1]);
    expect(nnz(FIG17_14)).toBe(15);
    expect(FIG17_14.entries.map((e) => e.label).join('')).toBe('abcdefghijklmno');
    expect(PRESETS.map((p) => p.id)).toEqual(['fig17_1', 'ex1', 'fig17_13', 'fig17_14']);
  });
  it('toggling a cell adds the smallest unused value, or the next letter, and toggling again removes it', () => {
    const a = toggle(FIG17_1, 3, 0);
    expect(a.entries.find((e) => e.r === 3 && e.c === 0)?.v).toBe(4); // 1, 2, 3 are used
    expect(toggle(a, 3, 0)).toEqual({ ...FIG17_1, entries: FIG17_1.entries });
    const b = toggle(FIG17_14, 0, 5);
    expect(b.entries.find((e) => e.r === 0 && e.c === 5)?.label).toBe('p');
    expect(toggle(FIG17_1, 0, 0).entries.length).toBe(7);
  });
});

describe('the formats of the running example', () => {
  it('COO in row order is Fig. 17.3', () => {
    const f = toCOO(FIG17_1);
    expect(f.rowIdx).toEqual([0, 0, 1, 1, 1, 2, 2, 3]);
    expect(f.colIdx).toEqual([0, 1, 0, 2, 3, 1, 2, 3]);
    expect(f.value).toEqual([1, 7, 5, 3, 9, 2, 8, 6]);
  });
  it('COO in Fig. 17.6\'s order is the figure, and SpMV gives the same y (p. 406)', () => {
    const f = toCOO(FIG17_1, FIG17_6_ORDER);
    expect(f.rowIdx).toEqual([1, 3, 2, 2, 0, 1, 1, 0]);
    expect(f.colIdx).toEqual([3, 3, 1, 2, 0, 0, 2, 1]);
    expect(f.value).toEqual([9, 6, 2, 8, 1, 5, 3, 7]);
    const x = [1, 2, 3, 4];
    const y0 = [10, 20, 30, 40];
    expect(runTrace(spmvTrace(f), x, y0)).toEqual(spmvRef(FIG17_1, x, y0));
    expect(runTrace(spmvTrace(toCOO(FIG17_1, shuffledOrder(8, 3))), x, y0)).toEqual(spmvRef(FIG17_1, x, y0));
    expect(() => toCOO(FIG17_1, [0, 0, 1, 2, 3, 4, 5, 6])).toThrow();
  });
  it('CSR is Fig. 17.7, with rowPtrs[4] = 8 as the end marker', () => {
    const f = toCSR(FIG17_1);
    expect(f.rowPtrs).toEqual([0, 2, 5, 7, 8]);
    expect(f.colIdx).toEqual([0, 1, 0, 2, 3, 1, 2, 3]);
    expect(f.value).toEqual([1, 7, 5, 3, 9, 2, 8, 6]);
  });
  it('ELL is Fig. 17.10: 4 x 3 padded, column-major, 4 pads', () => {
    const f = toELL(FIG17_1);
    expect(f.width).toBe(3);
    expect(f.colIdx).toEqual([0, 0, 1, 3, 1, 2, 2, P, P, 3, P, P]);
    expect(f.value).toEqual([1, 5, 2, 6, 7, 3, 8, 0, 0, 9, 0, 0]);
    expect(f.nnzPerRow).toEqual([2, 3, 2, 1]);
    expect(f.at.filter((e) => e === null).length).toBe(4);
    // Row 2's elements are value[2] and value[2 + 4] (p. 411); row = i % numRows (p. 413).
    expect([f.value[2], f.value[6]]).toEqual([2, 8]);
    f.at.forEach((e, i) => { if (e) expect(i % f.stride).toBe(e.r); });
  });
  it('CSC is Fig. 17.16', () => {
    const f = toCSC(FIG17_1);
    expect(f.colPtrs).toEqual([0, 2, 4, 6, 8]);
    expect(f.rowIdx).toEqual([0, 1, 0, 2, 1, 2, 1, 3]);
    expect(f.value).toEqual([1, 5, 7, 2, 3, 8, 9, 6]);
  });
  it('JDS of Fig. 17.14 is the figure: rows 1, 3, 5, 2, 4, 0 and iterPtr 0, 6, 11, 14, 15', () => {
    const f = toJDS(FIG17_14);
    expect(f.row).toEqual([1, 3, 5, 2, 4, 0]);
    expect(f.iterPtr).toEqual([0, 6, 11, 14, 15]);
    expect(f.colIdx).toEqual([0, 0, 0, 2, 1, 0, 1, 3, 2, 4, 4, 3, 5, 5, 4]);
    expect(f.at.map((e) => e.label).join('')).toBe('bhmfkacingldjoe');
  });
});

describe('exercise 1 (p. 422), the notes\' answer sketch', () => {
  it('COO, CSR, ELL and JDS', () => {
    const coo = toCOO(EXERCISE1);
    expect(coo.rowIdx).toEqual([0, 0, 1, 2, 2, 3, 3]);
    expect(coo.colIdx).toEqual([0, 2, 2, 1, 2, 0, 3]);
    expect(coo.value).toEqual([1, 7, 8, 4, 3, 2, 1]);
    expect(toCSR(EXERCISE1).rowPtrs).toEqual([0, 2, 3, 5, 7]);
    const ell = toELL(EXERCISE1);
    expect(ell.value).toEqual([1, 8, 4, 2, 7, 0, 3, 1]);
    expect(ell.colIdx).toEqual([0, 2, 1, 0, 2, P, 2, 3]);
    expect(ell.nnzPerRow).toEqual([2, 1, 2, 2]);
    const jds = toJDS(EXERCISE1);
    expect(jds.row).toEqual([0, 2, 3, 1]);
    expect(jds.value).toEqual([1, 4, 2, 8, 7, 3, 1]);
    expect(jds.colIdx).toEqual([0, 1, 0, 2, 2, 2, 3]);
    expect(jds.iterPtr).toEqual([0, 4, 7]);
  });
  it('question q-ex1-pads: the ELL form has exactly 1 pad slot', () => {
    expect(storage(toELL(EXERCISE1), EXERCISE1).pads).toBe(1);
  });
});

describe('hybrid ELL-COO on Fig. 17.13 (pp. 414 to 415)', () => {
  it('pure ELL is 5 wide with 22 pads; K = 2 leaves 3 pads and 5 COO entries, 3 from row 1 and 2 from row 6', () => {
    const ell = toELL(FIG17_13);
    expect(ell.width).toBe(5);
    expect(storage(ell, FIG17_13).pads).toBe(22);
    const h = toHybrid(FIG17_13, 2);
    expect(h.ell.width).toBe(2);
    expect(h.ell.at.filter((e) => e === null).length).toBe(3);
    expect(storage(h, FIG17_13).pads).toBe(3);
    expect(h.coo.rowIdx).toEqual([1, 1, 1, 6, 6]);
    // The tails: each long row's last elements.
    expect(h.coo.colIdx).toEqual([3, 5, 7, 4, 6]);
    // Every thread of the ELL part runs at most 2 iterations.
    expect(Math.max(...spmvTrace(h).launches[0].trips)).toBe(2);
  });
});

describe('storage and exercise 2', () => {
  it('exercise 2\'s formulas: COO 3z, CSR 2z + m + 1, ELL 2mK (+ m), JDS 2z + m + K + 1', () => {
    expect(exercise2Count(4, 8, 3)).toEqual({ coo: 24, csr: 21, ell: 24, ellWithCounts: 28, jds: 24 });
  });
  it('the explorer\'s counts agree with the formulas on every preset', () => {
    for (const p of PRESETS) {
      const m = p.m;
      const z = nnz(m);
      const K = longestRow(m);
      const want = exercise2Count(m.rows, z, K);
      expect(storage(toCOO(m), m).total).toBe(want.coo);
      expect(storage(toCSR(m), m).total).toBe(want.csr);
      expect(storage(toELL(m), m).total).toBe(want.ellWithCounts);
      expect(storage(toJDS(m), m).total).toBe(want.jds);
      expect(storage(toCSC(m), m).total).toBe(2 * z + m.cols + 1);
      expect(storage(toELL(m), m).dense).toBe(m.rows * m.cols);
    }
  });
  it('the book\'s 1000 x 1000 example (pp. 412 to 413): CSR about 2% of dense, ELL padded to 200 is 40%', () => {
    const c = exercise2Count(1000, 10000, 200);
    expect(c.csr).toBe(21001); // facts E1: 2.1%
    expect(c.ell / c.csr).toBeCloseTo(19.05, 2); // the book rounds to "20x"
    expect(c.ell / 1e6).toBe(0.4);
  });
  it('an aligned stride pads whole rows: Fig. 17.1 at stride 8 has 24 slots and 16 pads', () => {
    expect(alignedStride(4, 8)).toBe(8);
    expect(alignedStride(40, 8)).toBe(40);
    const f = toELL(FIG17_1, { stride: 8 });
    expect(f.value.length).toBe(24);
    expect(storage(f, FIG17_1).pads).toBe(16);
    // Question q-ell-index: row 1's element t = 2 (the 9) sits at 2*8 + 1 = 17.
    expect(f.value[17]).toBe(9);
    expect(f.colIdx[17]).toBe(3);
  });
});

describe('what each kernel reads (pp. 405, 410, 412, 420)', () => {
  it('COO: one thread per non-zero, one iteration, every update atomic', () => {
    const t = spmvTrace(toCOO(FIG17_1));
    expect(t.steps.length).toBe(1);
    expect(t.steps[0].lanes.map((l) => l.index)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(t.steps[0].lanes.every((l) => l.atomic)).toBe(true);
  });
  it('question q-coo-y1: three threads update y[1] (the 5, 3 and 9); CSR has one owner and no atomic', () => {
    const t = spmvTrace(toCOO(FIG17_1));
    const s = summarize(t, 4, 4);
    expect(s.atomicThreadsPerY).toEqual([2, 3, 2, 1]);
    expect(t.steps[0].lanes.filter((l) => l.y === 1).map((l) => l.v)).toEqual([5, 3, 9]);
    expect(s.atomicUpdates).toBe(8);
    expect(s.sharedY).toBe(3);
    const c = summarize(spmvTrace(toCSR(FIG17_1)), 4, 4);
    expect(c.atomicUpdates).toBe(0);
    expect(c.sharedY).toBe(0);
  });
  it('CSR: iteration 0 reads 0, 2, 5, 7; iteration 1 reads 1, 3, 6 and thread 3 is idle (p. 410)', () => {
    const r = reads('csr', FIG17_1);
    expect(r[0]).toEqual([0, 2, 5, 7]);
    expect(r[1]).toEqual([1, 3, 6, null]);
    expect(r[2]).toEqual([null, 4, null, null]);
    // The predict question: thread 2 at t = 1 reads rowPtrs[2] + 1 = 6.
    expect(r[1][2]).toBe(toCSR(FIG17_1).rowPtrs[2] + 1);
  });
  it('ELL: one contiguous run per iteration; nnzPerRow keeps CSR\'s idle lanes, the full width reads pads instead', () => {
    const t = spmvTrace(toELL(FIG17_1));
    expect(t.steps.map((s) => s.lanes.map((l) => l.index))).toEqual([[0, 1, 2, 3], [4, 5, 6, null], [null, 9, null, null]]);
    expect(t.steps.map((s) => matrixRuns(s, 4))).toEqual([[1], [1], [1]]);
    expect(t.steps.map(idleLanes)).toEqual([0, 1, 3]);
    const full = spmvTrace(toELL(FIG17_1), { ellLoop: 'full' });
    expect(full.steps.map((s) => s.lanes.map((l) => l.index))).toEqual([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]);
    expect(full.steps.map((s) => s.lanes.map((l) => l.pad))).toEqual([[false, false, false, false], [false, false, false, true], [true, false, true, true]]);
    expect(full.steps.map(idleLanes)).toEqual([0, 0, 0]);
    // A pad with column -1 is not multiplied and reads no x.
    expect(full.steps[2].lanes[0]).toMatchObject({ x: null, v: null });
  });
  it('CSR is scattered, ELL contiguous: 4 runs against 1 in iteration 0', () => {
    expect(matrixRuns(spmvTrace(toCSR(FIG17_1)).steps[0], 4)).toEqual([4]);
    expect(matrixRuns(spmvTrace(toELL(FIG17_1)).steps[0], 4)).toEqual([1]);
    expect(matrixRuns(spmvTrace(toCOO(FIG17_1)).steps[0], 4)).toEqual([1, 1]);
  });
  it('CSC: reads 0, 2, 4, 6 then 1, 3, 5, 7 (p. 420); x is read once, coalesced; y collides', () => {
    const t = spmvTrace(toCSC(FIG17_1));
    expect(t.steps.map((s) => s.lanes.map((l) => l.index))).toEqual([[0, 2, 4, 6], [1, 3, 5, 7]]);
    expect(xRuns(t.steps[0], 4)).toEqual([1]);
    expect(t.steps[1].lanes.every((l) => l.x === null)).toBe(true);
    const s = summarize(t, 4, 4);
    expect(s.atomicUpdates).toBe(8);
    expect(s.sharedY).toBe(3);
  });
  it('JDS: contiguous without pads, and the active set shrinks from the end', () => {
    const t = spmvTrace(toJDS(FIG17_14));
    expect(t.steps.map((s) => s.lanes.map((l) => l.index != null))).toEqual([
      [true, true, true, true, true, true],
      [true, true, true, true, true, false],
      [true, true, true, false, false, false],
      [true, false, false, false, false, false],
    ]);
    expect(t.steps.map((s) => matrixRuns(s, 6))).toEqual([[1], [1], [1], [1]]);
    expect(t.steps[0].lanes.map((l) => l.y)).toEqual([1, 3, 5, 2, 4, 0]);
  });
});

describe('divergence and pad work (divergence.ts loopBatch)', () => {
  it('Fig. 17.1, one group of 4: CSR and ELL run 3 iterations with 8 useful and 4 idle lane-iterations', () => {
    for (const id of ['csr', 'ell'] as const) {
      const s = summarize(spmvTrace(build(id, FIG17_1)), 4, 4);
      expect([s.groupIterations, s.useful, s.idle, s.pad]).toEqual([3, 8, 4, 0]);
    }
    const full = summarize(spmvTrace(toELL(FIG17_1), { ellLoop: 'full' }), 4, 4);
    expect([full.groupIterations, full.useful, full.idle, full.pad]).toEqual([3, 8, 0, 4]);
  });
  it('Fig. 17.14 in groups of 2: sorting rows (JDS) wastes fewer lane-iterations than CSR', () => {
    const csr = summarize(spmvTrace(toCSR(FIG17_14)), 2, 6);
    const jds = summarize(spmvTrace(toJDS(FIG17_14)), 2, 6);
    expect(csr.useful).toBe(15);
    expect(jds.useful).toBe(15);
    expect(jds.idle).toBeLessThan(csr.idle);
  });
  it('one long row multiplies ELL\'s pads: Fig. 17.1 with row 3 filled goes from 4 pads to 8', () => {
    let m = FIG17_1;
    for (const c of [0, 1, 2]) m = toggle(m, 3, c);
    expect(storage(toELL(m), m).pads).toBe(5);
    m = toggle(toggle(toggle(FIG17_1, 0, 2), 0, 3), 1, 1); // row 0 and row 1 to 4 non-zeros
    expect(longestRow(m)).toBe(4);
    expect(storage(toELL(m), m).pads).toBe(16 - nnz(m));
  });
});

describe('every format computes y = A·x + y (float32)', () => {
  it('on every preset, every format, both ELL loops, and the hybrid at every K', () => {
    for (const p of PRESETS) {
      const m = p.m;
      const x = Array.from({ length: m.cols }, (_, c) => 1 + (c % 5));
      const y0 = Array.from({ length: m.rows }, (_, r) => 10 * r);
      const want = spmvRef(m, x, y0);
      for (const id of ['coo', 'csr', 'ell', 'jds', 'csc'] as const)
        for (const ellLoop of ['nnzPerRow', 'full'] as const) expect(runTrace(spmvTrace(build(id, m), { ellLoop }), x, y0), `${p.id} ${id}`).toEqual(want);
      for (let K = 0; K <= longestRow(m); K++) expect(runTrace(spmvTrace(toHybrid(m, K)), x, y0), `${p.id} K=${K}`).toEqual(want);
      expect(runTrace(spmvTrace(toELL(m, { stride: alignedStride(m.rows, 8) })), x, y0)).toEqual(want);
    }
  });
  it('question q-pad-marker: pads with column 0 and x[0] = Inf give NaN in every row that has a pad; -1 pads do not', () => {
    const x = [Infinity, 1, 1, 1];
    const zero = runTrace(spmvTrace(toELL(FIG17_1, { padMarker: 0 }), { ellLoop: 'full' }), x, [0, 0, 0, 0]);
    expect(zero.map((v) => Number.isNaN(v))).toEqual([true, false, true, true]);
    const minus = runTrace(spmvTrace(toELL(FIG17_1), { ellLoop: 'full' }), x, [0, 0, 0, 0]);
    expect(minus.some((v) => Number.isNaN(v))).toBe(false);
    expect(minus).toEqual(spmvRef(FIG17_1, x, [0, 0, 0, 0]));
  });
});

describe('two threads, one y[0] (p. 407)', () => {
  it('without the atomic the 20 interleavings end at 1, 7 or 8; with it, always 8', () => {
    expect(interleavings([3, 3]).length).toBe(20);
    expect(cooOutcomes()).toEqual([1, 7, 8]);
    expect(cooOutcomes({ atomic: true })).toEqual([8]);
    expect(ROW0_PRODUCTS).toEqual([1, 7]);
  });
  it('"both load, then both store" loses thread 0\'s update', () => {
    const s = cooUpdates([0, 1, 0, 1, 0, 1]);
    expect(s.y).toBe(7);
    expect(s.events.map((e) => e.op)).toEqual(['load', 'load', 'add', 'add', 'store', 'store']);
    expect(cooUpdates([1, 0, 1, 0, 1, 0]).y).toBe(1);
    expect(cooUpdates([0, 0, 0, 1, 1, 1]).y).toBe(8);
  });
  it('an atomic step does load, add and store at once; a finished thread\'s turn changes nothing', () => {
    let s = rmwInit();
    s = rmwStep(s, 1, ROW0_PRODUCTS, true);
    expect([s.y, s.pc]).toEqual([7, [0, 3]]);
    expect(rmwStep(s, 1, ROW0_PRODUCTS, true)).toBe(s);
    s = rmwStep(s, 0, ROW0_PRODUCTS, true);
    expect(rmwDone(s)).toBe(true);
    expect(s.y).toBe(8);
  });
  it('the atomic fixes correctness, not order: 1e8, 1 and -1e8 in float32 give 0 or 1', () => {
    expect(permutations(3).length).toBe(6);
    expect(floatOutcomes(ORDER_TERMS)).toEqual([0, 1]);
    expect(floatSum(ORDER_TERMS, [0, 1, 2])).toBe(0);
    expect(floatSum(ORDER_TERMS, [0, 2, 1])).toBe(1);
    expect(permutations(3).filter((o) => floatSum(ORDER_TERMS, o) === 1).length).toBe(2);
  });
});
