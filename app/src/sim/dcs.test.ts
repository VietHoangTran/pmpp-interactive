import { describe, expect, it } from 'vitest';
import {
  BOOK_OPS_F4,
  KERNEL_ORDER,
  SCENE,
  arithmetic,
  directSum,
  gatherTimeline,
  mapKernel,
  movedAtom,
  opsCoarsened,
  opsGather,
  orderSensitive,
  sceneTally,
  scatterTimeline,
  scatterTotal,
  seededAtoms,
  termGather,
  writeStatements,
  type KernelId,
  type MapConfig,
} from './dcs';

const atoms = seededAtoms(7, 24, 4, 3);
const cfg = (kernel: KernelId, over: Partial<MapConfig> = {}): MapConfig => ({
  kernel, W: 24, H: 4, atoms, blockX: 4, blockY: 2, F: 4, ...over,
});

describe('atoms and the direct sum', () => {
  it('seeded atoms are repeatable, inside the slice, above it, and exact float32 values', () => {
    expect(seededAtoms(7, 24, 4, 3)).toEqual(atoms);
    expect(seededAtoms(7, 24, 4, 4)).not.toEqual(atoms);
    for (const a of atoms) {
      expect(a.x).toBeGreaterThanOrEqual(-0.5);
      expect(a.x).toBeLessThan(24);
      expect(a.z).toBeGreaterThan(0);
      expect(Math.fround(a.x)).toBe(a.x);
      expect(Math.fround(a.q)).toBe(a.q);
      expect(Math.abs(a.q)).toBeGreaterThanOrEqual(0.25);
    }
  });
  it('a moved atom stays in the layer', () => {
    const m = movedAtom(atoms[0], 3.3, 1.7);
    expect(m.z).toBe(atoms[0].z);
    expect(m.q).toBe(atoms[0].q);
  });
  it('the float32 gather term is within float32 rounding of the exact term', () => {
    const d = directSum([atoms[0]], 24, 4);
    expect(termGather(atoms[0], 5, 2)).toBeCloseTo(d[2 * 24 + 5], 6);
  });
});

describe('who writes where (Figs. 21.5, 21.6, 21.8, 21.10)', () => {
  it('scatter: every point has as many writers as there are atoms in the chunk, and needs atomics', () => {
    for (const n of [1, 4, 7, 12]) {
      const a = seededAtoms(n, 24, 4, 9);
      const r = mapKernel(cfg('scatter', { atoms: a }));
      expect(r.writers.every((w) => w.length === n)).toBe(true);
      expect(r.verdict).toBe(n > 1 ? 'atomic' : 'one-writer');
      expect(r.correct).toBe(true);
    }
  });
  it('scatter with a plain +=: a data race that loses updates (all lanes of one warp hit one address)', () => {
    const r = mapKernel(cfg('scatter', { atomic: false, blockX: 32 }));
    expect(r.verdict).toBe('race');
    expect(r.correct).toBe(false);
    expect(r.wrong).toBe(24 * 4);
  });
  it('scatter reads a different constant-memory address in every lane; gather and coarsened read one', () => {
    expect(mapKernel(cfg('scatter', { blockX: 32 })).constAddrsPerWarp).toBe(7);
    expect(mapKernel(cfg('scatter', { blockX: 4 })).constAddrsPerWarp).toBe(4);
    for (const k of ['gather', 'adjacent', 'strided'] as KernelId[]) expect(mapKernel(cfg(k)).constAddrsPerWarp).toBe(1);
  });
  it('gather and both corrected coarsened mappings: one writer per point, every point written, the direct sum', () => {
    for (const k of ['gather', 'adjacent', 'strided'] as KernelId[])
      for (const [W, H, bx, by, F] of [[24, 4, 4, 2, 4], [40, 8, 32, 1, 4], [17, 3, 8, 2, 2], [33, 5, 4, 4, 8], [8, 2, 16, 1, 8]]) {
        const r = mapKernel(cfg(k, { W, H, blockX: bx, blockY: by, F, atoms: seededAtoms(9, W, H, W + H) }));
        expect(r.writers.every((w) => w.length === 1), `${k} ${W}x${H}`).toBe(true);
        expect(r.verdict).toBe('one-writer');
        expect(r.conflicts).toBe(0);
        expect(r.unwritten).toBe(0);
        expect(r.correct, `${k} ${W}x${H}`).toBe(true);
      }
  });
  it('every correct mapping gives the same sums as the float64 direct sum, to float32 accuracy', () => {
    for (const k of ['scatter', 'gather', 'adjacent', 'strided'] as KernelId[]) {
      const r = mapKernel(cfg(k));
      r.potentials.forEach((v, p) => expect(Math.abs(v - r.direct[p])).toBeLessThan(1e-5));
    }
  });
  it('the printed line 05 at blockDim.x = 32: block 0 writes points 0 to 34, 3 to 31 four times, and never 35 to 127 (E2)', () => {
    const r = mapKernel(cfg('printed', { W: 128, H: 1, blockX: 32, blockY: 1, F: 4 }));
    const w = r.writers.map((x) => x.length);
    expect(w.slice(0, 35).every((n) => n > 0)).toBe(true);
    expect(w.slice(35).every((n) => n === 0)).toBe(true);
    expect(w.slice(3, 32).every((n) => n === 4)).toBe(true);
    expect(w.slice(0, 3)).toEqual([1, 2, 3]);
    expect(w.slice(32, 35)).toEqual([3, 2, 1]);
    expect(r.verdict).toBe('race');
    expect(r.unwritten).toBe(93);
    expect(r.correct).toBe(false);
  });
  it('the printed mapping in general: a block of B threads covers B + F − 1 points of its 4B', () => {
    const r = mapKernel(cfg('printed', { W: 64, H: 2, blockX: 8, blockY: 1, F: 4 }));
    expect(r.launch.gridX).toBe(2);
    expect(r.unwritten).toBe(2 * 2 * (32 - 11));
  });
  it('coarsening shrinks the launch by F and gather pads the edge', () => {
    expect(mapKernel(cfg('gather', { W: 30, blockX: 8 })).launch.gridX).toBe(4);
    expect(mapKernel(cfg('gather', { W: 30, blockX: 8 })).idle).toBe(2 * 4);
    expect(mapKernel(cfg('strided', { W: 30, blockX: 8, F: 4 })).launch.gridX).toBe(1);
    expect(mapKernel(cfg('adjacent', { W: 30, blockX: 8, F: 4 })).threads[7].outside).toBe(2);
  });
  it('pairs per thread: all points for scatter, all atoms for gather, F times that coarsened', () => {
    expect(mapKernel(cfg('scatter')).pairsPerThread).toBe(96);
    expect(mapKernel(cfg('gather')).pairsPerThread).toBe(7);
    expect(mapKernel(cfg('adjacent')).pairsPerThread).toBe(28);
  });
  it('every kernel is modelled', () => {
    for (const k of KERNEL_ORDER) expect(mapKernel(cfg(k)).potentials.length).toBe(96);
  });
});

describe('write statements (p. 521, Figs. 21.9 and 21.10; E2)', () => {
  const base = { F: 4, warpWidth: 32, blockX: 128, blockY: 1, rowPitch: 1024, segmentBytes: 32 };
  it('adjacent ownership: base + 4t + m, 512 bytes and 16 sectors per statement, 64 for four', () => {
    const r = writeStatements({ ...base, ownership: 'adjacent' });
    expect(r.statements.map((s) => s.access.transactions)).toEqual([16, 16, 16, 16]);
    expect(r.statements[1].lanes[3].element).toBe(4 * 3 + 1);
    expect(r.transactions).toBe(64);
    expect(r.statements[0].access.efficiency).toBe(0.25);
  });
  it('blockDim.x stride: 4 sectors per statement, 16 for four, every byte used', () => {
    const r = writeStatements({ ...base, ownership: 'strided' });
    expect(r.statements.map((s) => s.access.transactions)).toEqual([4, 4, 4, 4]);
    expect(r.statements[2].lanes[5].element).toBe(5 + 2 * 128);
    expect(r.transactions).toBe(16);
    expect(r.bytesFetched).toBe(r.bytesNeeded);
  });
  it('one float4 store per thread: 16 sectors in one request, all bytes used, aligned', () => {
    const r = writeStatements({ ...base, ownership: 'float4' });
    expect(r.statements.length).toBe(1);
    expect(r.transactions).toBe(16);
    expect(r.statements[0].access.efficiency).toBe(1);
    expect(r.aligned).toBe(true);
    expect(writeStatements({ ...base, ownership: 'float4', F: 8 }).statements.length).toBe(2);
    expect(() => writeStatements({ ...base, ownership: 'float4', F: 2 })).toThrow();
  });
  it('a 64-lane wavefront: the stride gives one 256-byte run per statement when blockDim.x >= 64', () => {
    for (const bx of [64, 128]) {
      const r = writeStatements({ ...base, warpWidth: 64, blockX: bx, ownership: 'strided', segmentBytes: 128 });
      expect(r.statements.every((s) => s.runs === 1 && s.access.bytesNeeded === 256 && s.access.transactions === 2)).toBe(true);
    }
  });
  it('a 64-lane wavefront with blockDim.x = 32 spans two rows: two runs per statement', () => {
    const r = writeStatements({ ...base, warpWidth: 64, blockX: 32, blockY: 2, ownership: 'strided', segmentBytes: 128 });
    expect(r.statements.every((s) => s.runs === 2)).toBe(true);
    expect(new Set(r.statements[0].lanes.map((l) => l.ty))).toEqual(new Set([0, 1]));
  });
  it('a 64-lane wavefront, adjacent: 1,024 bytes per statement', () => {
    const r = writeStatements({ ...base, warpWidth: 64, ownership: 'adjacent', segmentBytes: 128 });
    expect(r.statements[0].lanes.at(-1)!.byte - r.statements[0].lanes[0].byte + 4).toBe(1024 - 12);
    expect(r.statements[0].access.transactions).toBe(8);
  });
  it('a row pitch that is not a multiple of 4 breaks float4 alignment on the second row', () => {
    const r = writeStatements({ ...base, blockX: 16, blockY: 2, rowPitch: 1022, ownership: 'float4' });
    expect(r.aligned).toBe(false);
  });
});

describe('operation counts per atom (E1, Exercise 2)', () => {
  it('four points: counted 16 loads and 40 operations against 4 and 27 (25 hoisted)', () => {
    const g = opsGather(4);
    const c = opsCoarsened(4);
    expect([g.loads, g.sub, g.add, g.mul, g.div, g.sqrt, g.branch]).toEqual([16, 12, 12, 12, 4, 4, 4]);
    expect([c.loads, c.sub, c.add, c.mul, c.mulUnhoisted, c.div, c.sqrt, c.branch]).toEqual([4, 3, 12, 6, 8, 4, 4, 1]);
    expect(arithmetic(g)).toBe(40);
    expect(arithmetic(c)).toBe(27);
    expect(arithmetic(c, true)).toBe(25);
  });
  it('the book\'s printed figures differ: 48 and 24, 12 divisions, no square roots', () => {
    expect(BOOK_OPS_F4.gather.total).toBe(48);
    expect(BOOK_OPS_F4.coarsened.total).toBe(24);
    expect(BOOK_OPS_F4.gather.div).not.toBe(opsGather(4).div);
    expect(BOOK_OPS_F4.coarsened.add).not.toBe(opsCoarsened(4).add);
  });
  it('Exercise 2, F = 8: 32 loads, 24 sub, 24 mul, 24 add, 8 div, 8 sqrt, 8 branches against 4, 3, 10 (16), 24, 8, 8, 1', () => {
    const g = opsGather(8);
    const c = opsCoarsened(8);
    expect([g.loads, g.sub, g.mul, g.add, g.div, g.sqrt, g.branch]).toEqual([32, 24, 24, 24, 8, 8, 8]);
    expect([c.loads, c.sub, c.mul, c.mulUnhoisted, c.add, c.div, c.sqrt, c.branch]).toEqual([4, 3, 10, 16, 24, 8, 8, 1]);
  });
  it('loads fall F-fold; divisions and square roots stay one per point', () => {
    for (let F = 1; F <= 8; F++) {
      expect(opsGather(F).loads / opsCoarsened(F).loads).toBe(F);
      expect(opsCoarsened(F).div).toBe(F);
      expect(opsCoarsened(F).sqrt).toBe(F);
    }
    expect(opsCoarsened(1)).toEqual({ ...opsGather(1), add: 3, mul: 3, mulUnhoisted: 3 });
  });
});

describe('the scatter-gather scene', () => {
  const { W, atoms: A, runA, runB } = SCENE;
  it('six points, four atoms, every term finite', () => {
    expect(W).toBe(6);
    expect(A.length).toBe(4);
    for (const s of scatterTimeline(A, W)) expect(s.values.every(Number.isFinite)).toBe(true);
  });
  it('scatter: at every step all four lanes update one point and read four constant addresses', () => {
    const sc = scatterTimeline(A, W);
    expect(sc.map((s) => s.point)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(sc.every((s) => new Set(s.constAddrs).size === 4)).toBe(true);
  });
  it('the two runs\' orders give different float32 totals at some points (and at point 3 in particular)', () => {
    const diff = orderSensitive(A, W, runA, runB);
    expect(diff.length).toBeGreaterThan(0);
    expect(diff.map((d) => d.point)).toContain(3);
    const d3 = diff.find((d) => d.point === 3)!;
    expect(Math.abs(d3.a - d3.b)).toBeLessThan(1e-6);
    expect(d3.a.toPrecision(6)).toBe(d3.b.toPrecision(6));
  });
  it('gather: the registers after the last step equal the direct sum, and the same bits on every run', () => {
    const g = gatherTimeline(A, W);
    expect(g.length).toBe(4);
    const d = directSum(A, W, 1);
    g.at(-1)!.registers.forEach((v, i) => expect(Math.abs(v - d[i])).toBeLessThan(1e-6));
    expect(gatherTimeline(A, W)).toEqual(g);
    expect(scatterTotal(A, 0, runA)).toBeCloseTo(d[0], 6);
  });
  it('tally: 24 atomic updates to 6 addresses (4 each), 4 constant addresses per step; 24 adds, 6 writes, 1 address', () => {
    expect(sceneTally(A, W)).toEqual({
      scatterUpdates: 24, scatterAddresses: 6, updatesPerAddress: 4, scatterConstPerStep: 4,
      gatherAdds: 24, gatherWrites: 6, gatherConstPerStep: 1,
    });
  });
});
