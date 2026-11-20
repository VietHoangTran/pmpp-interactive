import { describe, expect, it } from 'vitest';
import {
  CONV_CORRECT,
  FIG714_M,
  FIG714_PUBLISHED,
  FIG714_T,
  attenuation,
  constMemIntensity,
  convRowAccess,
  fig714,
  flatLoadPass,
  fromOutTile,
  gridOf,
  idealIntensity,
  makeProblem,
  organisation,
  runAll,
  runTiledBlock,
  stateAt,
  threadInfo,
  tileGeometry,
  tiledIntensity,
  trafficRatioVsConst,
  warpSectors,
  type ConvConfig,
} from './convtile';
import type { Schedule } from './tiled';

const SCHEDULES: Schedule[] = ['lockstep', 'in-order', 'reversed', 'shuffled'];
const cfg = (inTile: number, r: number, over: Partial<ConvConfig> = {}): ConvConfig => ({ ...CONV_CORRECT, inTile, r, ...over });
const r2 = (x: number) => Math.round(x * 100) / 100;

describe('tile sizes (pp. 172–173, Fig. 7.11)', () => {
  it('a 4x4 output tile with a 5x5 filter needs an 8x8 input tile: 4 times as many elements', () => {
    const g = tileGeometry(8, 2);
    expect([g.outTile, g.inElems, g.outElems, g.ratio]).toEqual([4, 64, 16, 4]);
  });
  it('a 16x16 output tile with a 5x5 filter: 400 inputs for 256 outputs (question q-ratio)', () => {
    const g = fromOutTile(16, 2);
    expect([g.inTile, g.inElems, g.outElems, g.haloElems]).toEqual([20, 400, 256, 144]);
    expect(g.ratio).toBe(1.5625);
  });
  it('an input tile no wider than the filter has no output tile', () => {
    expect(tileGeometry(4, 2).outTile).toBe(0);
    expect(tileGeometry(4, 2).ratio).toBe(Infinity);
  });
});

describe('arithmetic intensity (p. 176) and Fig. 7.14 (p. 177)', () => {
  it('every published value of Fig. 7.14 to two decimals', () => {
    const rows = fig714();
    rows.forEach((row, i) => {
      const pub = FIG714_PUBLISHED[FIG714_M[i]];
      expect(row.values.map(r2)).toEqual(pub.slice(0, FIG714_T.length));
      expect(r2(row.ideal)).toBe(pub[FIG714_T.length]);
    });
  });
  it('the running example: 5x5 filter, 28x28 output tile, 5.42 against an ideal of 6.25', () => {
    expect(r2(tiledIntensity(28, 5))).toBe(5.42);
    expect(idealIntensity(5)).toBe(6.25);
  });
  it("Fig. 7.13's toy: 6x6 output, 3x3 filter gives 1.62", () => {
    expect(r2(tiledIntensity(6, 3))).toBe(1.62);
  });
  it('intensity is the ideal times the attenuation factor, which tends to 1', () => {
    for (const t of [4, 8, 28, 100])
      for (const m of [3, 5, 9]) expect(tiledIntensity(t, m)).toBeCloseTo(idealIntensity(m) * attenuation(t, m), 12);
    expect(attenuation(1e6, 5)).toBeCloseTo(1, 4);
    expect(attenuation(8, 5)).toBeLessThan(attenuation(16, 5));
  });
  it('question q-ai9: input tile 32, 9x9 filter: output tile 24, 14.58 FLOP/B, ideal 20.25', () => {
    const g = tileGeometry(32, 4);
    expect(g.outTile).toBe(24);
    expect(r2(tiledIntensity(g.outTile, g.m))).toBe(14.58);
    expect(idealIntensity(9)).toBe(20.25);
  });
  it('element size scales intensity: 2-byte elements double it', () => {
    expect(tiledIntensity(28, 5, 2)).toBeCloseTo(2 * tiledIntensity(28, 5, 4), 12);
    expect(fig714(2)[1].values[2]).toBeCloseTo(2 * 5.4204, 3);
  });
  it('the traffic ratio against the constant-memory kernel (0.5 FLOP/B, p. 171): about 10.8 at t = 28, m = 5', () => {
    expect(constMemIntensity()).toBe(0.5);
    expect(r2(trafficRatioVsConst(28, 5))).toBe(10.84);
  });
});

describe('the two thread organisations (p. 173, Exercise 11)', () => {
  it('organisation 1 at the book sizes: 1024 threads, 240 idle while computing', () => {
    const o = organisation(28, 2, 1);
    expect([o.threads, o.idleCompute, o.computing, o.loadItersFlat, o.sharedBytes]).toEqual([1024, 240, 784, 1, 4096]);
  });
  it("Fig. 7.13's toy: 64 threads, the outer ring of 28 idle", () => {
    const o = organisation(6, 1, 1);
    expect([o.threads, o.idleCompute, o.computing]).toEqual([64, 28, 36]);
  });
  it('question q-org2: a 32x32 output block with a 9x9 filter loads a 40x40 tile (6400 B) and reaches 15.80', () => {
    const o = organisation(32, 4, 2);
    expect([o.threads, o.inTile, o.sharedBytes, o.idleCompute]).toEqual([1024, 40, 6400, 0]);
    expect(r2(tiledIntensity(o.outTile, o.m))).toBe(15.8);
    // Organisation 1 with the same output tile would need 1600 threads.
    expect(organisation(32, 4, 1).threads).toBe(1600);
  });
  it('question q-ex11: a 28x28 output block and a 5x5 filter: at most 2 loads per thread with a flat loop', () => {
    const o = organisation(28, 2, 2);
    expect(o.threads).toBe(784);
    expect(o.inElems).toBe(1024);
    expect(o.loadItersFlat).toBe(2);
    // A loop striding by 28 in each direction gives the corner threads 4.
    expect(o.loadIters2D).toBe(4);
  });
  it('a flat load loop: the first 784 elements in pass 0, the remaining 240 in pass 1', () => {
    const p = flatLoadPass(28, 2);
    expect(p.length).toBe(1024);
    expect(p.filter((x) => x === 0).length).toBe(784);
    expect(p.filter((x) => x === 1).length).toBe(240);
  });
});

describe('one warp touching tile rows (inefficiency 2, p. 177; NVIDIA 32-byte sectors)', () => {
  const base = { r: 2, width: 1024, row: 1, warpSize: 32, segmentBytes: 32 };
  it('a 28-wide output row starting at bx*112 bytes from an aligned row: 4 sectors for 112 useful bytes', () => {
    for (let bx = 0; bx < 8; bx++) {
      const a = convRowAccess({ ...base, layout: 'fig712', op: 'store', bx });
      expect(a.rows[0].startBytes % 32).toBe((bx * 112) % 32);
      expect([a.transactions, a.bytesNeeded]).toEqual([4, 112]);
    }
  });
  it('the same row when image rows are not aligned: 4 or 5 sectors', () => {
    const counts = new Set<number>();
    for (let bx = 0; bx < 8; bx++) counts.add(convRowAccess({ ...base, width: 1003, layout: 'fig712', op: 'store', bx }).transactions);
    expect([...counts].sort()).toEqual([4, 5]);
  });
  it("the 32-wide input-tile row of an interior block starts 8 bytes off: 5 sectors for 128 bytes", () => {
    for (let bx = 1; bx < 8; bx++) {
      const a = convRowAccess({ ...base, layout: 'fig712', op: 'load', bx });
      expect([a.transactions, a.bytesNeeded]).toEqual([5, 128]);
    }
  });
  it('a 32-wide aligned tile row touches exactly 4 sectors, loading or storing', () => {
    for (let bx = 0; bx < 8; bx++)
      for (const op of ['load', 'store'] as const) {
        const a = convRowAccess({ ...base, layout: 'tile32', op, bx });
        expect([a.transactions, a.bytesNeeded, a.efficiency]).toEqual([4, 128, 1]);
      }
  });
  it('the left-most block loads only its 30 in-image columns', () => {
    const a = convRowAccess({ ...base, layout: 'fig712', op: 'load', bx: 0 });
    expect([a.firstCol, a.rows[0].lanes, a.transactions]).toEqual([0, 30, 4]);
  });
  it('a 64-wide warp covers two block rows, an image row apart', () => {
    const a = convRowAccess({ ...base, layout: 'tile32', op: 'store', bx: 1, warpSize: 64 });
    expect(a.rows.length).toBe(2);
    expect(a.rows[1].startBytes - a.rows[0].startBytes).toBe(4096);
    expect(a.transactions).toBe(8);
  });
  it('warpSectors on its own', () => {
    expect(warpSectors(0, 32, 32).transactions).toBe(4);
    expect(warpSectors(16, 28, 32).transactions).toBe(4);
    expect(warpSectors(24, 28, 32).transactions).toBe(5);
    expect(warpSectors(0, 32, 32, 4, { segmentBytes: 128 }).transactions).toBe(1);
  });
});

describe('the Fig. 7.12 kernel, thread by thread', () => {
  const sizes: [number, number][] = [[10, 10], [14, 14], [9, 13], [6, 6], [1, 1], [17, 5]];
  for (const [w, h] of sizes)
    for (const s of SCHEDULES)
      it(`correct kernel, ${w}x${h} image, 8x8 block, r = 1, ${s}`, () => {
        const r = runAll(cfg(8, 1), makeProblem(w, h, 1), s, 3);
        expect(r.violations).toEqual([]);
        expect(r.correct).toBe(true);
      });
  it('correct kernel with r = 2 and r = 3 on an 8x8 block', () => {
    for (const rad of [2, 3]) expect(runAll(cfg(8, rad), makeProblem(11, 9, rad), 'shuffled', 5).correct).toBe(true);
  });
  it('the filter is asymmetric in both directions, so a flipped or shifted filter would show', () => {
    const p = makeProblem(4, 4, 1);
    expect(p.F).toEqual([1, 3, 5, 4, 1, 3, 2, 4, 1]);
  });

  it('without the barrier, a thread that runs ahead reads slots nobody has loaded yet', () => {
    const r = runAll(cfg(8, 1, { barrier: false }), makeProblem(10, 10, 1), 'in-order');
    expect(r.correct).toBe(false);
    expect(r.violations.length).toBeGreaterThan(0);
    expect(r.violations.every((v) => v.kind === 'read-before-write')).toBe(true);
  });
  it('without the barrier: fails in order, reversed and shuffled; lockstep hides it', () => {
    const p = makeProblem(10, 10, 1);
    for (const s of ['in-order', 'reversed', 'shuffled'] as Schedule[]) expect(runAll(cfg(8, 1, { barrier: false }), p, s, 7).correct).toBe(false);
    const ls = runAll(cfg(8, 1, { barrier: false }), p, 'lockstep');
    expect(ls.correct).toBe(true);
    expect(ls.violations).toEqual([]);
  });
  it('without the barrier, in order: the first failure is thread (1,1) reading N_s[1][2]', () => {
    const tr = runTiledBlock(cfg(8, 1, { barrier: false }), makeProblem(10, 10, 1), 0, 0, 'in-order');
    const v = tr.violations[0];
    const s = tr.steps[v.step];
    expect([s.tx, s.ty, s.tap!.slotR, s.tap!.slotC]).toEqual([1, 1, 1, 2]);
  });
  it('without ghost zeros, edge outputs are NaN and interior outputs are right', () => {
    const p = makeProblem(10, 10, 1);
    const r = runAll(cfg(8, 1, { zeroGhost: false }), p, 'lockstep');
    expect(r.correct).toBe(false);
    expect(r.violations.every((v) => v.kind === 'unwritten-slot')).toBe(true);
    const edge = (i: number) => {
      const y = Math.floor(i / 10);
      const x = i % 10;
      return y === 0 || x === 0 || y === 9 || x === 9;
    };
    expect(r.wrong.length).toBe(36);
    expect(r.wrong.every(edge)).toBe(true);
    expect(r.wrong.every((i) => Number.isNaN(r.P[i]))).toBe(true);
    expect(r.missing).toEqual([]);
  });
  it('without the -FILTER_RADIUS offset, the tile shifts: row 0 and column 0 of P are never computed', () => {
    const p = makeProblem(10, 10, 1);
    const r = runAll(cfg(8, 1, { loadOffset: false }), p, 'lockstep');
    expect(r.correct).toBe(false);
    expect(r.violations).toEqual([]);
    expect(r.missing.length).toBe(19);
    expect(r.missing.every((i) => i < 10 || i % 10 === 0)).toBe(true);
    expect(r.wrong).toEqual([]);
  });
  it('a 10x10 image needs 2x2 blocks of 8x8 threads; a 14x14 image 3x3', () => {
    expect(gridOf(cfg(8, 1), { width: 10, height: 10 })).toEqual({ x: 2, y: 2 });
    expect(gridOf(cfg(8, 1), { width: 14, height: 14 })).toEqual({ x: 3, y: 3 });
  });
});

describe("the narrated scene's numbers (block (0,0) of a 10x10 image, Fig. 7.13's sizes, in order)", () => {
  const c = cfg(8, 1);
  const p = makeProblem(10, 10, 1);
  const tr = runTiledBlock(c, p, 0, 0, 'in-order');
  it('15 of the 64 loads are ghost cells, stored as zeros', () => {
    const loads = tr.steps.filter((s) => s.op === 'load');
    expect(loads.length).toBe(64);
    expect(loads.filter((s) => s.write?.ghost).length).toBe(15);
  });
  it('one barrier opening, after all 64 loads', () => {
    const rel = tr.steps.findIndex((s) => s.op === 'release');
    expect(tr.steps.filter((s) => s.op === 'release').length).toBe(1);
    expect(tr.steps.slice(0, rel).filter((s) => s.op === 'load').length).toBe(64);
    expect(stateAt(c, tr, rel).slots.every((x) => x.written)).toBe(true);
  });
  it('28 threads idle on the ring, 36 compute with 9 taps each (324 multiply-adds) and store', () => {
    expect(tr.steps.filter((s) => s.op === 'idle').length).toBe(28);
    expect(tr.steps.filter((s) => s.op === 'tap').length).toBe(324);
    expect(tr.stores.length).toBe(36);
  });
  it('thread (1,1) stores output (0,0) and thread (6,6) output (5,5)', () => {
    const by = (tx: number, ty: number) => tr.stores.find((s) => s.thread === ty * 8 + tx)!.index;
    expect(by(1, 1)).toBe(0);
    expect(by(6, 6)).toBe(5 * 10 + 5);
  });
  it("p. 175 erratum: thread (5,5) has tileRow = tileCol = 4; the patch at N_s[5][5] is thread (6,6)'s", () => {
    const t55 = threadInfo(c, p, 0, 0, 5, 5);
    expect([t55.tileRow, t55.tileCol]).toEqual([4, 4]);
    const t66 = threadInfo(c, p, 0, 0, 6, 6);
    expect([t66.tileRow, t66.tileCol]).toEqual([5, 5]);
  });
  it("the next block's input tile overlaps this one by 2 columns", () => {
    const a = threadInfo(c, p, 0, 0, 7, 0).col; // last column of block 0's input tile
    const b = threadInfo(c, p, 1, 0, 0, 0).col; // first column of block 1's
    expect(a - b + 1).toBe(2);
  });
  it('the results equal the reference', () => {
    for (const s of tr.stores) expect(s.value).toBe(p.ref[s.index]);
  });
});
