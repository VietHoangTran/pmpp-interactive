import { describe, expect, it } from 'vitest';
import {
  CORRECTED,
  HUNT_SHAPES,
  IMPLICIT_SCENE,
  NVIDIA_WARP,
  PRINTED,
  dataFor,
  huntShape,
  laneRuns,
  multiples,
  phasesOf,
  runBlock,
  runKernel,
  traffic,
  warpView,
  worstPasses,
  type KShape,
} from './implicitgemm';
import { BOOK_SHAPE, BOOK_X, convDirect, preset } from './unfold';

const variants = [
  { ceil: false, bounds: false, bug: false },
  { ceil: true, bounds: false, bug: false },
  { ceil: false, bounds: true, bug: false },
  { ceil: true, bounds: true, bug: false },
];

describe('the corrected kernel', () => {
  it('equals the loop nest on every shape of the hunt and on LeNet C5, and on odd shapes with a batch', () => {
    const shapes: KShape[] = [
      ...HUNT_SHAPES.map((h) => h.shape),
      { ...preset('c5').shape, T: 16 },
      { C: 3, H: 11, W: 7, K: 3, M: 5, N: 3, T: 4 },
      { C: 2, H: 5, W: 9, K: 1, M: 17, N: 2, T: 8 },
    ];
    for (const s of shapes) {
      const r = runKernel(s, CORRECTED);
      expect(r.correct, JSON.stringify(s)).toBe(true);
      expect([r.oobX, r.oobF, r.outsideWrites, r.strayWrites]).toEqual([0, 0, 0, 0]);
      expect(r.Y).toEqual(convDirect(s, dataFor(s).X, dataFor(s).F));
    }
  });
});

describe('the kernel as printed', () => {
  it("is right on the book's example with T = 2: every size is a multiple of the tile", () => {
    const s = huntShape('book').shape;
    expect(multiples(s)).toEqual({ M: true, P: true, CKK: true });
    const r = runKernel(s, PRINTED);
    expect(r.correct).toBe(true);
    expect(r.Y).toEqual([14, 20, 15, 24, 12, 24, 17, 26]);
    expect(r.phases).toBe(6);
  });
  it('LeNet C1: one phase, 16 of 25 terms, every output wrong, F read past its end for rows 6 to 15', () => {
    const r = runKernel(huntShape('c1').shape, PRINTED);
    expect([r.phases, r.termsUsed, r.terms]).toEqual([1, 16, 25]);
    expect(r.wrong).toBe(6 * 28 * 28);
    expect(r.oobF).toBe(10 * 16 * 49);
    expect(r.outsideWrites).toBe(10 * 16 * 49);
  });
  it('LeNet C3: nine phases, 144 of 150 terms, and the padding columns clobber other outputs', () => {
    const r = runKernel(huntShape('c3').shape, PRINTED);
    expect([r.phases, r.termsUsed, r.terms]).toEqual([9, 144, 150]);
    expect(r.wrong).toBe(1600);
    expect(r.strayWrites).toBe(15 * 12);
    expect(r.status.filter((x) => x === 'clobbered').length).toBe(15 * 12);
  });
  it('is right on a shape where M, H_out·W_out and C·K·K are all multiples of 16', () => {
    const s = huntShape('all16').shape;
    expect(multiples(s)).toEqual({ M: true, P: true, CKK: true });
    expect(runKernel(s, PRINTED).correct).toBe(true);
  });
  it('C·K·K = 64 with a 7 × 7 output: the phases are right, but stray stores and reads past X break it', () => {
    const s = huntShape('p49').shape;
    const r = runKernel(s, PRINTED);
    expect(r.termsUsed).toBe(64);
    expect(r.correct).toBe(false);
    expect(r.strayWrites).toBeGreaterThan(0);
    expect(r.oobX).toBeGreaterThan(0);
    expect(r.status.every((x) => x !== 'wrong')).toBe(true); // only clobbered slots fail
    expect(runKernel(s, { ceil: false, bounds: true, bug: false }).correct).toBe(true);
  });
  it('which fix each shape needs: the outcome table', () => {
    const table = HUNT_SHAPES.map((h) => variants.map((v) => runKernel(h.shape, v).correct));
    expect(table).toEqual([
      [true, true, true, true], // book
      [false, false, false, true], // C1: ceiling and bounds
      [false, false, false, true], // C3: ceiling and bounds
      [true, true, true, true], // all multiples of 16
      [false, false, true, true], // P = 49: bounds only
    ]);
  });
  it('the ceiling without bounds checks gathers past the end of X on C1 (u = 25 to 31 is channel 1, which does not exist)', () => {
    const r = runKernel(huntShape('c1').shape, { ceil: true, bounds: false, bug: false });
    expect(r.oobX).toBe(7 * 16 * 49);
    expect(r.extraTerms).toBe(7);
    expect(r.correct).toBe(false);
  });
});

describe('Fig. 19.9: block (0, 0) of the book example with T = 2', () => {
  const b = IMPLICIT_SCENE.block;
  it('owns 14, 20 (map 0) and 12, 24 (map 1): two pixels of each map, not four pixels of map 0', () => {
    expect(b.stores.map((s) => [s.Row, s.Col])).toEqual([[0, 0], [0, 1], [1, 0], [1, 1]]);
    expect(IMPLICIT_SCENE.finals).toEqual([14, 20, 12, 24]);
  });
  it('phase 0 gathers X0 values 1, 2, 2, 0 from three pixels; X[n,0,0,1] fills two slots', () => {
    const g = b.phases[0].b;
    expect(g.map((x) => x.value)).toEqual([1, 2, 2, 0]);
    expect(g.map((x) => [x.c, x.h, x.w])).toEqual([[0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 2]]);
    expect(new Set(g.map((x) => x.idx)).size).toBe(3);
    expect(b.phases[0].Fds).toEqual([1, 1, 1, 0]);
  });
  it('phase 1 gathers 1, 1, 1, 3 from the middle row of X0, and F columns 2 and 3 are 2 2 / 0 1', () => {
    expect(b.phases[1].b.map((x) => x.value)).toEqual([1, 1, 1, 3]);
    expect(b.phases[1].b.every((x) => x.c === 0 && x.h === 1)).toBe(true);
    expect(b.phases[1].Fds).toEqual([2, 2, 0, 1]);
  });
  it('six phases, channels 0, 0, 1, 1, 2, 2; the sums end at 14, 20, 12, 24', () => {
    expect(b.phases.map((p) => p.b[0].c)).toEqual([0, 0, 1, 1, 2, 2]);
    expect(b.phases[5].sums).toEqual([14, 20, 12, 24]);
    // Worked by hand: channel 0 gives 7, 10, 2, 5; channel 1 adds 5, 8, 5, 13; channel 2 adds 2, 2, 5, 6.
    expect(b.phases[1].sums).toEqual([7, 10, 2, 5]);
    expect(b.phases[3].sums).toEqual([12, 18, 7, 18]);
  });
  it('the planted bug (W for W_out) breaks it, and fails on LeNet C1 too', () => {
    const bug = { ...CORRECTED, bug: true };
    expect(runKernel(huntShape('book').shape, bug).correct).toBe(false);
    expect(runKernel(huntShape('c1').shape, bug).correct).toBe(false);
    expect(runBlock(huntShape('book').shape, bug, { X: BOOK_X, F: dataFor(BOOK_SHAPE).F }, 1, 0).phases[0].b[1]).toMatchObject({ v: 3, h: 1, w: 0 });
  });
});

describe('global accesses for the toy', () => {
  it('direct 96, explicit 48 written and 48 read back, implicit 48 gathers', () => {
    const t = IMPLICIT_SCENE.traffic;
    expect(t.direct).toEqual({ x: 96, f: 96 });
    expect(t.explicit).toEqual({ unfoldReadsX: 48, writesB: 48, gemmReadsB: 48, f: 48 });
    expect(t.implicit).toEqual({ gathersX: 48, f: 48 });
    expect(IMPLICIT_SCENE.gathers).toBe(48);
  });
  it('the model agrees with the kernel run on C3: gathers = C·K·K · P · ceil(M/T)', () => {
    const s = huntShape('c3').shape;
    const r = runKernel(s, CORRECTED);
    expect(r.gathers).toBe(traffic(s, 16).implicit.gathersX);
    expect(r.fLoads).toBe(traffic(s, 16).implicit.f);
    expect(r.gathers).toBe(15000);
  });
  it('per output on C3: 20 loads implicit (10 phases × 2) against 300 direct', () => {
    const s = huntShape('c3').shape;
    expect(2 * phasesOf(s, CORRECTED)).toBe(20);
    expect(traffic(s, 16).direct.x * 2 / (16 * 100)).toBe(300);
  });
});

describe('one warp (Exercise 4)', () => {
  const c3 = huntShape('c3').shape;
  it('runs in lane order, skipping inactive lanes', () => {
    expect(laneRuns([4, 5, 6, null, 8, 9, 20])).toEqual([
      { start: 4, length: 3, firstLane: 0, lastLane: 2 },
      { start: 8, length: 2, firstLane: 4, lastLane: 5 },
      { start: 20, length: 1, firstLane: 6, lastLane: 6 },
    ]);
  });
  it('C3, block column 0, phase 0, warp 0: each half-warp gathers in 2 runs with a jump of K = 5', () => {
    const w = warpView(c3, 0, 0, 0, 0);
    expect(w.lanes.length).toBe(NVIDIA_WARP);
    expect(w.xRunsByTy.map((r) => r.runs.map((x) => [x.start, x.length]))).toEqual([
      [[0, 10], [14, 6]],
      [[1, 10], [15, 6]],
    ]);
    expect(w.xJumps).toEqual([5, 5]);
    expect(w.fRuns.map((r) => [r.start, r.length])).toEqual([[0, 16], [150, 16]]);
    expect(w.yRuns.map((r) => [r.start, r.length])).toEqual([[0, 16], [100, 16]]);
  });
  it('no bank conflicts at T = 16 on the 32-bank model, same-word reads merged', () => {
    const b = warpView(c3, 0, 0, 0, 0, 3).banks;
    expect([b.fdsWrite.passes, b.bdsWrite.passes, b.fdsRead.passes, b.bdsRead.passes]).toEqual([1, 1, 1, 1]);
    expect(b.fdsRead.banksUsed).toBe(2); // two words, 16 apart
    expect(b.bdsRead.banksUsed).toBe(16); // 16 words, each read by two lanes
    expect(b.bdsRead.sharedWord).toBe(true);
    for (const s of [c3, huntShape('c1').shape]) for (let warp = 0; warp < 8; warp++) expect(worstPasses(s, 1, 0, 2, warp)).toBe(1);
  });
  it('edge block of C3 (columns 96 to 111): lanes past P make no access', () => {
    const w = warpView(c3, 6, 0, 0, 0);
    expect(w.lanes.filter((l) => l.x !== null).length).toBe(8);
    expect(w.yRuns.map((r) => r.length)).toEqual([4, 4]);
  });
});
