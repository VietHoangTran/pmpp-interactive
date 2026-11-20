import { describe, expect, it } from 'vitest';
import { planesLoaded } from './stencilslab';
import {
  CORRECT,
  idx3,
  interiorBlock,
  localZ,
  makeProblem,
  planeTimeline,
  runAll,
  runBlock,
  stateAt,
  type Kernel,
  type Schedule,
  type ZConfig,
} from './zmarch';

const SIZES = [
  { N: 6, inTile: 4 },
  { N: 10, inTile: 6 },
];
const KERNELS: Kernel[] = ['planes', 'registers'];
const ORDERS: Schedule[] = ['lockstep', 'in-order', 'reversed', 'shuffled'];
const cfgOf = (size: { N: number; inTile: number }, kernel: Kernel, over: Partial<ZConfig> = {}): ZConfig => ({
  ...size,
  kernel,
  ...CORRECT,
  ...over,
});

describe('the problem', () => {
  it('the CPU sweep is the 7-point stencil of Fig. 8.6, by hand at one point', () => {
    const p = makeProblem(6);
    const v = (z: number, j: number, k: number) => p.input[idx3(6, z, j, k)];
    const byHand = 2 * v(2, 2, 2) + 1 * v(2, 2, 1) + 3 * v(2, 2, 3) + 4 * v(2, 1, 2) + 1 * v(2, 3, 2) + 3 * v(1, 2, 2) + 2 * v(3, 2, 2);
    expect(p.ref[idx3(6, 2, 2, 2)]).toBe(byHand);
    expect(p.interior.length).toBe(64);
    expect(Number.isNaN(p.ref[idx3(6, 0, 2, 2)])).toBe(true);
  });
  it('neighbours along y always differ, so swapping c3 and c4 always shows', () => {
    const p = makeProblem(10);
    for (const ix of p.interior) {
      const z = Math.floor(ix / 100), j = Math.floor(ix / 10) % 10, k = ix % 10;
      expect(p.input[idx3(10, z, j - 1, k)]).not.toBe(p.input[idx3(10, z, j + 1, k)]);
    }
  });
});

describe('both barriers: correct under every thread order', () => {
  for (const size of SIZES)
    for (const kernel of KERNELS)
      it(`${kernel}, tile ${size.inTile} on ${size.N}^3`, () => {
        const prob = makeProblem(size.N);
        for (const s of ORDERS) {
          const r = runAll(cfgOf(size, kernel), prob, s);
          expect(r.correct).toBe(true);
          expect(r.violations).toEqual([]);
        }
      });
});

describe('a missing barrier', () => {
  for (const size of SIZES)
    for (const kernel of KERNELS) {
      const prob = makeProblem(size.N);
      it(`${kernel}, tile ${size.inTile}: without barrier 2, a neighbour's slot is overwritten before it is read`, () => {
        for (const s of ['in-order', 'reversed'] as Schedule[]) {
          const r = runAll(cfgOf(size, kernel, { barrier2: false }), prob, s);
          expect(r.correct).toBe(false);
          expect(r.violations.some((v) => v.kind === 'overwritten-before-read')).toBe(true);
          expect(r.violations.every((v) => v.kind === 'overwritten-before-read')).toBe(true);
        }
      });
      it(`${kernel}, tile ${size.inTile}: without barrier 1, a neighbour's slot is read before it is written`, () => {
        for (const s of ['in-order', 'reversed'] as Schedule[]) {
          const r = runAll(cfgOf(size, kernel, { barrier1: false }), prob, s);
          expect(r.correct).toBe(false);
          expect(r.violations.some((v) => v.kind === 'read-before-write')).toBe(true);
        }
      });
      it(`${kernel}, tile ${size.inTile}: lockstep hides both`, () => {
        for (const over of [{ barrier1: false }, { barrier2: false }, { barrier1: false, barrier2: false }]) {
          const r = runAll(cfgOf(size, kernel, over), prob, 'lockstep');
          expect(r.correct).toBe(true);
        }
      });
      it(`${kernel}, tile ${size.inTile}: the shuffled order (fixed seeds) also exposes each missing barrier`, () => {
        expect(runAll(cfgOf(size, kernel, { barrier1: false }), prob, 'shuffled').correct).toBe(false);
        expect(runAll(cfgOf(size, kernel, { barrier2: false }), prob, 'shuffled').correct).toBe(false);
      });
    }
});

describe('the c3/c4 pairing as printed in Figs. 8.10 and 8.12', () => {
  it('gives a wrong value at every interior point, for both kernels, with no race at all', () => {
    for (const size of SIZES)
      for (const kernel of KERNELS) {
        const prob = makeProblem(size.N);
        const r = runAll(cfgOf(size, kernel, { pairing: 'as-printed-8.10' }), prob, 'lockstep');
        expect(r.violations).toEqual([]);
        expect(r.wrong.length).toBe(prob.interior.length);
      }
  });
});

describe('counters', () => {
  const size = SIZES[1];
  const prob = makeProblem(size.N);
  const b = interiorBlock(cfgOf(size, 'planes'));
  const T = size.inTile;
  const D = T - 2;
  const planes = runBlock(cfgOf(size, 'planes'), prob, b.bx, b.by, b.bz, 'in-order');
  const regs = runBlock(cfgOf(size, 'registers'), prob, b.bx, b.by, b.bz, 'in-order');

  it('the interior block of the 10^3 grid with tile 6 is (1, 1, 1), the book picture', () => {
    expect(b).toEqual({ bx: 1, by: 1, bz: 1 });
    expect(interiorBlock(cfgOf(SIZES[0], 'planes'))).toEqual({ bx: 1, by: 1, bz: 1 });
  });
  it('shared reads per output: 7 with three planes, 4 with registers', () => {
    expect(planes.counters.globalStores).toBe((T - 2) ** 2 * D);
    expect(planes.counters.stencilSharedReads / planes.counters.globalStores).toBe(7);
    expect(regs.counters.stencilSharedReads / regs.counters.globalStores).toBe(4);
  });
  it('plane copies: 2 reads and 2 writes per thread per iteration, against 1 write', () => {
    expect(planes.counters.copySharedReads).toBe(2 * T * T * D);
    expect(planes.counters.copySharedWrites).toBe(2 * T * T * D);
    expect(regs.counters.copySharedReads).toBe(0);
    expect(regs.counters.copySharedWrites).toBe(T * T * D);
  });
  it('global loads are the same for both kernels: planesLoaded x t^2', () => {
    expect(planes.counters.globalLoads).toBe(planesLoaded(D) * T * T);
    expect(regs.counters.globalLoads).toBe(planes.counters.globalLoads);
  });
  it('at the grid edge, loads are clipped and still equal for both kernels', () => {
    const p0 = runBlock(cfgOf(size, 'planes'), prob, 0, 0, 0, 'lockstep');
    const r0 = runBlock(cfgOf(size, 'registers'), prob, 0, 0, 0, 'lockstep');
    // Block (0,0,0): rows and columns -1 are outside, and so is plane -1.
    expect(p0.counters.globalLoads).toBe((T - 1) * (T - 1) * (planesLoaded(D) - 1));
    expect(r0.counters.globalLoads).toBe(p0.counters.globalLoads);
  });
  it('stateAt at the end of the trace agrees with the run', () => {
    for (const tr of [planes, regs]) {
      const cfg = cfgOf(size, tr === planes ? 'planes' : 'registers');
      const st = stateAt(cfg, tr, tr.steps.length);
      expect(st.counters).toEqual(tr.counters);
      expect(st.stores).toEqual(tr.stores);
      expect(st.done.every(Boolean)).toBe(true);
      for (const s of st.stores) expect(s.value).toBe(prob.ref[s.index]);
    }
  });
  it('the whole sweep loads the same for both kernels, and each kernel stores every interior point once', () => {
    const a = runAll(cfgOf(size, 'planes'), prob, 'lockstep');
    const c = runAll(cfgOf(size, 'registers'), prob, 'lockstep');
    expect(a.counters.globalLoads).toBe(c.counters.globalLoads);
    expect(a.counters.globalStores).toBe(prob.interior.length);
    expect(c.counters.globalStores).toBe(prob.interior.length);
  });
});

describe('the step trace', () => {
  it('a missing barrier 2 is caught on the read that needed the old plane', () => {
    const size = SIZES[0];
    const prob = makeProblem(size.N);
    const tr = runBlock(cfgOf(size, 'planes', { barrier2: false }), prob, 1, 1, 1, 'in-order');
    const v = tr.violations[0];
    expect(v.kind).toBe('overwritten-before-read');
    expect(v.message).toMatch(/^thread \(tx 1, ty 1\) still needed plane 2 from inCurr_s\[1\]\[0\], but thread \(tx 0, ty 1\) had already replaced it with plane 3$/);
  });
  it('a missing barrier 1 is caught on a slot never written', () => {
    const size = SIZES[0];
    const prob = makeProblem(size.N);
    const tr = runBlock(cfgOf(size, 'registers', { barrier1: false }), prob, 1, 1, 1, 'in-order');
    expect(tr.violations[0].kind).toBe('read-before-write');
    expect(tr.violations[0].message).toMatch(/never written/);
  });
  it('halo threads compute nothing; the inner (t-2)^2 do', () => {
    const size = SIZES[1];
    const tr = runBlock(cfgOf(size, 'planes'), makeProblem(size.N), 1, 1, 1, 'lockstep');
    const first = tr.steps.filter((s) => s.op === 'compute' && s.iter === 0);
    expect(first.length).toBe(36);
    expect(first.filter((s) => s.active).length).toBe(16);
    expect(first.filter((s) => !s.active).every((s) => s.why === 'halo')).toBe(true);
  });
});

describe('planeTimeline (the scene)', () => {
  const size = SIZES[1];
  const prob = makeProblem(size.N);
  const tl = planeTimeline(cfgOf(size, 'planes'), prob);
  const local = (z: number | null) => (z === null ? null : z - (tl.iStart - 1));

  it('before the loop: tile planes 0 and 1 in prev and curr, 2 planes loaded', () => {
    const f = tl.frames[0];
    expect(f.label).toBe('start');
    expect([local(f.hold.prev), local(f.hold.curr)]).toEqual([0, 1]);
    expect(f.hold.next).toBe(null);
    expect(f.planesLoaded).toBe(2);
    expect(f.loads).toBe(72);
  });
  it('iteration 0: tile plane 2 lands in next; 16 threads compute output plane 1 of the tile', () => {
    const [, loaded, computed, shifted] = tl.frames;
    expect(local(loaded.hold.next)).toBe(2);
    expect(loaded.planesLoaded).toBe(3);
    expect(computed.active.filter(Boolean).length).toBe(16);
    expect(computed.outputPlanes.map(local)).toEqual([1]);
    expect([local(shifted.hold.prev), local(shifted.hold.curr)]).toEqual([1, 2]);
  });
  it('the march ends at 6 input planes for 4 output planes, one new plane per iteration', () => {
    const last = tl.frames[tl.frames.length - 1];
    expect(last.planesLoaded).toBe(6);
    expect(last.outputPlanes.map(local)).toEqual([1, 2, 3, 4]);
    expect(last.stores).toBe(64);
    const loaded = tl.frames.filter((f) => f.label === 'loaded').map((f) => f.planesLoaded);
    expect(loaded).toEqual([3, 4, 5, 6]);
  });
  it('the registers kernel holds the same planes and loads exactly as much', () => {
    const tr = planeTimeline(cfgOf(size, 'registers'), prob);
    expect(tr.frames.map((f) => [f.hold, f.loads])).toEqual(tl.frames.map((f) => [f.hold, f.loads]));
    expect(tr.frames.map((f) => f.sharedCurr)).toEqual(tl.frames.map((f) => f.hold.curr));
  });
  it('localZ numbers a block\'s input planes from 0, as Fig. 8.9 draws them', () => {
    expect(localZ(tl.iStart, tl.iStart - 1)).toBe(0);
    expect(localZ(tl.iStart, tl.iStart + 4)).toBe(5);
  });
});
