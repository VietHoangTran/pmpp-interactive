import { describe, expect, it } from 'vitest';
import {
  SCENE,
  acceptMasks,
  acceptedShare,
  binAtoms,
  binOf,
  blockList,
  blockWalk,
  capacityFor,
  counts,
  criticalRadii,
  diverges,
  makeWorld,
  maxOccupancy,
  nearFar,
  neighbourhood,
  planar,
  runGrid,
  sceneCase,
  superRadius,
  worldH,
  worldW,
} from './cutoff';

const halfDiag = (b: number) => (b * Math.SQRT2) / 2;

describe('Fig. 21.14: the 2D neighbourhood (E4)', () => {
  it('21 bins for any R between about 1.58 b and 2.12 b, only 5 of them fully inside', () => {
    for (const R of [1.582, 1.7, 1.9, 2, 2.1, 2.12]) {
      expect(counts(R), `R = ${R}`).toEqual({ total: 21, full: 5, partial: 16 });
    }
  });
  it('the range ends are where the (2, 1) bins and the corner bins start to touch', () => {
    expect(criticalRadii(3).slice(0, 5).map((r) => Number(r.toFixed(3)))).toEqual([0.5, 0.707, 1.5, 1.581, 2.121]);
    expect(counts(1.58).total).toBe(13);
    expect(counts(2.13).total).toBe(25);
  });
  it('9 fully covered + 12 partly covered holds only at R = 1.5·√2·b ≈ 2.12 b', () => {
    expect(counts(1.5 * Math.SQRT2)).toEqual({ total: 21, full: 9, partial: 12 });
    expect(counts(1.5 * Math.SQRT2 - 1e-6).full).toBe(5);
    expect(counts(1.5 * Math.SQRT2 + 1e-6).total).toBe(25);
  });
  it('at R = 2b: the centre and its four edge neighbours are the fully covered ones; the corners are out', () => {
    const n = neighbourhood(2);
    expect(n.filter((o) => o.cover === 'full').map((o) => `${o.dx},${o.dy}`).sort()).toEqual(['-1,0', '0,-1', '0,0', '0,1', '1,0']);
    expect(n.some((o) => Math.abs(o.dx) === 2 && Math.abs(o.dy) === 2)).toBe(false);
    expect(n.some((o) => o.dx === 2 && o.dy === 1)).toBe(true);
  });
  it('relative offsets become absolute bin coordinates for a block at (X, Y) (Fig. 21.14, bottom)', () => {
    const w = makeWorld(8, 8, 4, 10, 1);
    const l = blockList(w, 2, 3, 5);
    expect(l.length).toBe(21);
    const e = l.find((x) => x.rel.dx === 2 && x.rel.dy === 1)!;
    expect([e.bx, e.by]).toEqual([5, 6]);
    expect(l.find((x) => x.rel.dx === -1 && x.rel.dy === -1)!.bx).toBe(2);
    expect(blockList(w, 2, 0, 0).filter((x) => x.inBox).length).toBe(8);
  });
});

describe('the book\'s worked case, p. 525 (E4)', () => {
  const b = 4;
  it('3D, 4 Å bins, 12 Å cutoff: 461 bins (147 fully inside) with half the diagonal, 667 with the full one', () => {
    expect(superRadius(12, b, 3, 'half')).toBeCloseTo(12 + 2 * Math.sqrt(3), 10);
    expect(counts(superRadius(12, b, 3, 'half') / b, 3)).toMatchObject({ total: 461, full: 147 });
    expect(counts(superRadius(12, b, 3, 'full') / b, 3).total).toBe(667);
  });
  it('2D, the same numbers: 61 and 77', () => {
    expect(counts(superRadius(12, b, 2, 'half') / b, 2).total).toBe(61);
    expect(counts(superRadius(12, b, 2, 'full') / b, 2).total).toBe(77);
  });
  it('about 25 % of the atoms a thread tests pass in 3D, 46 % in 2D', () => {
    expect(acceptedShare(12, b, 3)).toBeCloseTo(0.2453, 3);
    expect(Math.round(acceptedShare(12, b, 3) * 100)).toBe(25);
    expect(Math.round(acceptedShare(12, b, 2) * 100)).toBe(46);
  });
});

describe('binning with a fixed capacity (p. 527)', () => {
  const w = makeWorld(8, 8, 4, 128, 5);
  it('bookkeeping: real + overflow = atoms, real + dummy = bins × capacity', () => {
    for (const c of [1, 2, 3, 5, maxOccupancy(w)]) {
      const b = binAtoms(w, c);
      expect(b.real + b.overflow.length).toBe(w.atoms.length);
      expect(b.real + b.dummy).toBe(64 * c);
      expect(b.slots.every((s) => s.length === c)).toBe(true);
    }
    expect(binAtoms(w, maxOccupancy(w)).overflow.length).toBe(0);
  });
  it('every atom lands in the bin that holds its position', () => {
    const b = binAtoms(w, maxOccupancy(w));
    b.slots.forEach((s, k) => s.filter((a) => a >= 0).forEach((a) => {
      const [bx, by] = binOf(w, w.atoms[a].x, w.atoms[a].y);
      expect(by * 8 + bx).toBe(k);
    }));
  });
  it('capacityFor picks the smallest capacity under the overflow share', () => {
    const c = capacityFor(w, 0.03);
    expect(binAtoms(w, c).overflow.length).toBeLessThanOrEqual(0.03 * 128);
    expect(binAtoms(w, c - 1).overflow.length).toBeGreaterThan(0.03 * 128);
  });
});

describe('a block walks its list (p. 526)', () => {
  const w = makeWorld(8, 8, 4, 128, 7);
  const R = 2;
  const cutoff = R * 4 - halfDiag(4);
  it('the list is conservative: with every atom binned, each thread accepts exactly the atoms within its cutoff', () => {
    const bin = binAtoms(w, maxOccupancy(w));
    for (const [bx, by] of [[3, 4], [0, 0], [7, 2]]) {
      const walk = blockWalk(w, bin, cutoff, R, bx, by);
      walk.points.forEach(([i, j], t) => {
        expect(walk.perThread[t].accepted).toBe(nearFar(w, i, j, cutoff).nearAtoms);
        expect(walk.perThread[t].sum).toBeCloseTo(nearFar(w, i, j, cutoff).near, 10);
      });
    }
  });
  it('tests count every slot for every thread, dummies included; most tests fail', () => {
    const bin = binAtoms(w, 4);
    const walk = blockWalk(w, bin, cutoff, R, 3, 4);
    expect(walk.tested).toBe(21 * 4 * 16);
    expect(walk.accepted).toBeLessThan(walk.tested / 2);
    expect(walk.list.at(-1)!.cumTested).toBe(walk.tested);
  });
  it('device result plus the host\'s overflow pass equals the exact cutoff sum', () => {
    const bin = binAtoms(w, 2);
    expect(bin.overflow.length).toBeGreaterThan(0);
    const g = runGrid(w, bin, cutoff, R);
    g.withHost.forEach((v, p) => expect(v).toBeCloseTo(g.cutoffSum[p], 10));
    expect(g.device.some((v, p) => Math.abs(v - g.cutoffSum[p]) > 1e-9)).toBe(true);
    expect(g.hostPairs).toBeGreaterThan(0);
  });
  it('the cutoff drops a far field: near + far = direct at every point', () => {
    for (const [i, j] of [[0, 0], [13, 17], [31, 31]]) {
      const n = nearFar(w, i, j, cutoff);
      const direct = w.atoms.reduce((s, a) => s + a.q / Math.sqrt((i - a.x) ** 2 + (j - a.y) ** 2 + 0.75 ** 2), 0);
      expect(n.near + n.far).toBeCloseTo(direct, 10);
      expect(n.far).not.toBe(0);
    }
  });
  it('doubling the box at the same density: direct pairs × 16; tests per point stay put, so binned pairs grow about 4-fold', () => {
    const small = makeWorld(8, 8, 4, 128, 3);
    const big = makeWorld(16, 16, 4, 512, 3);
    const a = runGrid(small, binAtoms(small, 4), cutoff, R, false);
    const b = runGrid(big, binAtoms(big, 4), cutoff, R, false);
    expect(b.directPairs / a.directPairs).toBe(16);
    expect(b.tests / a.tests).toBeGreaterThan(4);
    expect(b.tests / a.tests).toBeLessThan(5);
    const perPoint = (g: typeof a, w: typeof small) => g.tests / (worldW(w) * worldH(w));
    expect(perPoint(b, big) / perPoint(a, small)).toBeLessThan(1.2);
  });
  it('warps diverge: some staged atom splits a warp into accepting and rejecting lanes', () => {
    const walk = blockWalk(w, binAtoms(w, 4), cutoff, R, 3, 4);
    const any = walk.list.some((e) => e.slots.some((s, k) => s >= 0 && diverges(acceptMasks(e, k, 32)[0])));
    expect(any).toBe(true);
    const e = walk.list[0];
    expect(acceptMasks(e, 0, 32)[0].length).toBe(16);
    const w8 = makeWorld(4, 4, 8, 30, 1);
    const e8 = blockWalk(w8, binAtoms(w8, 3), 8, 2, 2, 2).list[0];
    expect(e8.inBox).toBe(true);
    expect(acceptMasks(e8, 0, 32).length).toBe(2);
  });
});

describe('the binned-block scene', () => {
  const s = sceneCase();
  it('a 9 × 9 box of 4 × 4 bins, 160 atoms, capacity 4, radius 2 bin widths: 21 offsets, all inside the box', () => {
    expect(s.walk.list.length).toBe(21);
    expect(s.walk.list.every((e) => e.inBox)).toBe(true);
    expect(s.cutoff).toBeCloseTo(8 - 2 * Math.SQRT2, 12);
    expect(SCENE.capacity).toBe(4);
  });
  it('binning: 157 atoms in slots, 3 on the overflow list (under 2 %), 167 dummy slots', () => {
    expect(s.binning.real).toBe(157);
    expect(s.binning.overflow.length).toBe(3);
    expect(s.binning.dummy).toBe(167);
  });
  it('the walk: 1,344 tests (752 of them on dummies), 114 accepted, against 2,560 pairs for direct summation', () => {
    expect(s.walk.tested).toBe(1344);
    expect(s.walk.dummyTests).toBe(752);
    expect(s.walk.accepted).toBe(114);
    expect(Math.round((s.walk.accepted / s.walk.tested) * 100)).toBe(8);
    expect(s.walk.directPairs).toBe(2560);
  });
  it('the first entry holds two atoms and two dummies, and one of its atoms splits the warp', () => {
    const e = s.walk.list[0];
    expect(e.slots.filter((x) => x >= 0).length).toBe(2);
    expect(e.slots.filter((x) => x < 0).length).toBe(2);
    expect(e.slots.some((x, k) => x >= 0 && diverges(acceptMasks(e, k, 32)[0]))).toBe(true);
  });
  it('planar distance is what the test uses', () => {
    expect(planar(3, 4, { x: 0, y: 0, q: 1 })).toBe(5);
  });
});
