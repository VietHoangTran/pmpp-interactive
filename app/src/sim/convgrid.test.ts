import { describe, expect, it } from 'vitest';
import { SEGMENT_MODELS } from './coalesce';
import {
  CUDA_GRID,
  HIP_GRID,
  coverage,
  geometry,
  limits,
  mapThread,
  ownerOf,
  warpReads,
  writersOf,
  xIndex,
  type LaunchParams,
  type WarpReadParams,
} from './convgrid';

const P = (o: Partial<LaunchParams>): LaunchParams => ({ N: 1, M: 4, Hout: 32, Wout: 32, tile: 16, division: 'floor', ...o });

describe('Fig. 19.5 and Fig. 19.6: the grid', () => {
  it('M = 4, 2 x 2 tiles of 16 x 16: gridDim(4, 4, N), 256 threads per block', () => {
    const g = geometry(P({ N: 7 }));
    expect(g.grid).toEqual({ x: 4, y: 4, z: 7 });
    expect(g.Wgrid).toBe(2);
    expect(g.Hgrid).toBe(2);
    expect(g.T).toBe(4);
    expect(g.threadsPerBlock).toBe(256);
  });
  it('LeNet C1 (28 x 28) as printed: W_grid = H_grid = 1; with ceiling division 2', () => {
    expect(geometry(P({ M: 6, Hout: 28, Wout: 28 })).T).toBe(1);
    expect(geometry(P({ M: 6, Hout: 28, Wout: 28, division: 'ceil' })).T).toBe(4);
  });
});

describe('Fig. 19.7: from blockIdx and threadIdx to (n, m, h, w)', () => {
  const p = P({ N: 2 });
  const g = geometry(p);
  it('thread (5, 9) of block (3, 2, 1) computes (1, 3, 25, 5)', () => {
    const r = mapThread(p, g, { bx: 3, by: 2, bz: 1, tx: 5, ty: 9 });
    expect([r.n, r.m, r.h, r.w]).toEqual([1, 3, 25, 5]);
    expect(r.yIndex).toBe(((1 * 4 + 3) * 32 + 25) * 32 + 5);
  });
  it('with / and % swapped it computes (1, 3, 9, 21)', () => {
    const r = mapThread(p, g, { bx: 3, by: 2, bz: 1, tx: 5, ty: 9 }, true);
    expect([r.n, r.m, r.h, r.w]).toEqual([1, 3, 9, 21]);
  });
  it('ownerOf inverts the mapping', () => {
    expect(ownerOf(p, g, 1, 3, 25, 5)).toEqual({ bx: 3, by: 2, bz: 1, tx: 5, ty: 9 });
    const sw = ownerOf(p, g, 1, 3, 9, 21, true)!;
    const r = mapThread(p, g, sw, true);
    expect([r.n, r.m, r.h, r.w]).toEqual([1, 3, 9, 21]);
  });
});

describe('coverage', () => {
  it('LeNet C1 as printed (floor division): 256 of 784 pixels per map, no thread outside its map', () => {
    const c = coverage(P({ M: 6, Hout: 28, Wout: 28, division: 'floor' }), { swap: false, guard: false })!;
    expect(c.once).toBe(6 * 256);
    expect(c.never).toBe(6 * (784 - 256));
    expect(c.multi).toBe(0);
    expect(c.stray).toBe(0);
  });
  it('ceiling division with a bounds check: all 784, 240 idle threads per map', () => {
    const c = coverage(P({ M: 6, Hout: 28, Wout: 28, division: 'ceil' }), { swap: false, guard: true })!;
    expect(c.once).toBe(6 * 784);
    expect(c.never).toBe(0);
    expect(c.idle).toBe(6 * 240);
  });
  it('ceiling division with no bounds check: strays store into other pixels and past the end', () => {
    const p = P({ M: 6, Hout: 28, Wout: 28, division: 'ceil' });
    const c = coverage(p, { swap: false, guard: false })!;
    expect(c.stray).toBe(6 * 240);
    expect(c.never).toBe(0);
    expect(c.multi).toBeGreaterThan(0);
    expect(c.pastEnd).toBeGreaterThan(0);
    // A thread at w = 28 of row 0 lands on pixel (1, 0) of the same map.
    const ws = writersOf(p, { swap: false, guard: false }, 0, 0, 1, 0);
    expect(ws.length).toBe(2);
  });
  it('the swap is a bijection on a square tile grid (64 x 64)', () => {
    const p = P({ Hout: 64, Wout: 64 });
    const ok = coverage(p, { swap: false, guard: true })!;
    const sw = coverage(p, { swap: true, guard: true })!;
    expect(sw.once).toBe(ok.once);
    expect(sw.never).toBe(0);
    expect(sw.idle).toBe(0);
    expect(Array.from(sw.counts).every((x) => x === 1)).toBe(true);
  });
  it('at 32 x 64 the swap leaves tile columns 2 and 3 uncomputed and sends threads to rows 32 to 63', () => {
    const p = P({ M: 1, Hout: 32, Wout: 64 });
    const g = geometry(p);
    expect([g.Hgrid, g.Wgrid]).toEqual([2, 4]);
    const sw = coverage(p, { swap: true, guard: true })!;
    expect(sw.never).toBe(32 * 32);
    for (let h = 0; h < 32; h++)
      for (let w = 0; w < 64; w++) expect(sw.counts[h * 64 + w]).toBe(w >= 32 ? 0 : 1);
    expect(sw.idle).toBe(4 * 256);
    // Without the check those threads store anyway; with one map and one sample, past the end.
    const raw = coverage(p, { swap: true, guard: false })!;
    expect(raw.stray).toBe(4 * 256);
    expect(raw.pastEnd).toBe(4 * 256);
    // Every stray's h is 32 or more.
    let rows = 0;
    for (let by = 0; by < g.T; by++) {
      const r = mapThread(p, g, { bx: 0, by, bz: 0, tx: 0, ty: 0 }, true);
      if (!r.inRange) {
        expect(r.h).toBeGreaterThanOrEqual(32);
        rows++;
      }
    }
    expect(rows).toBe(4);
  });
  it('returns null for a grid too large to walk', () => {
    expect(coverage(P({ M: 64, Hout: 1024, Wout: 1024 }), { swap: false, guard: true })).toBeNull();
  });
});

describe('documented grid limits', () => {
  it('CUDA: y and z up to 65535; a 4096 x 4096 output in 16 x 16 tiles needs 65536 tiles', () => {
    expect(CUDA_GRID.y).toBe(65535);
    const g = geometry(P({ M: 1, Hout: 4096, Wout: 4096 }));
    expect(g.grid.y).toBe(65536);
    expect(limits(g).cuda.y).toBe(false);
    expect(limits(geometry(P({ M: 1, Hout: 4080, Wout: 4096 }))).cuda.ok).toBe(true);
  });
  it('HIP: at most 2^32 - 1 threads per grid', () => {
    expect(HIP_GRID.threads).toBe(4294967295);
    const g = geometry(P({ N: 256, M: 256, Hout: 256, Wout: 256 }));
    expect(g.threads).toBe(2 ** 32);
    expect(limits(g).hip.threads).toBe(false);
    expect(limits(g).cuda.ok).toBe(true);
  });
});

describe('one warp of the direct kernel reading X (Exercise 2)', () => {
  const base: WarpReadParams = {
    layout: 'NCHW', dims: { N: 1, C: 4, H: 18, W: 18 }, K: 3, tile: 16, warpWidth: 32, warp: 0,
    c: 0, p: 0, q: 0, lanes: 'w', segmentBytes: SEGMENT_MODELS.nvidia.segmentBytes!,
  };
  it('layout index functions', () => {
    const d = { N: 2, C: 3, H: 4, W: 5 };
    expect(xIndex('NCHW', d, 1, 2, 3, 4)).toBe(((1 * 3 + 2) * 4 + 3) * 5 + 4);
    expect(xIndex('NHWC', d, 1, 2, 3, 4)).toBe(((1 * 4 + 3) * 5 + 4) * 3 + 2);
    expect(xIndex('CHWN', d, 1, 2, 3, 4)).toBe(((2 * 4 + 3) * 5 + 4) * 2 + 1);
  });
  it('NCHW, warp 0 of a 16 x 16 block: two rows of 16 floats, 5 sectors by hand', () => {
    // Row 0: floats 0..15 = bytes 0..63, sectors 0 and 1. Row 1: floats 18..33 = bytes 72..135, sectors 2, 3, 4.
    const r = warpReads(base);
    expect(r.lanes.length).toBe(32);
    expect(r.runs).toBe(2);
    expect(r.access.transactions).toBe(5);
    expect(r.stride).toBe(4);
  });
  it('NCHW with q = 1 shifts both runs by 4 bytes', () => {
    const r = warpReads({ ...base, q: 1 });
    // bytes 4..67: sectors 0, 1, 2; bytes 76..139: sectors 2, 3, 4 -> 5 distinct
    expect(r.access.segments).toEqual([0, 1, 2, 3, 4]);
  });
  it('NHWC: lanes C floats apart; C = 4 gives 16 sectors, C = 8 gives one sector per lane', () => {
    // Row 0: bytes 0, 16, ..., 240 -> sectors 0..7. Row 1: bytes 288 + 16w -> sectors 9..16.
    const r4 = warpReads({ ...base, layout: 'NHWC' });
    expect(r4.stride).toBe(16);
    expect(r4.access.transactions).toBe(16);
    const r8 = warpReads({ ...base, layout: 'NHWC', dims: { ...base.dims, C: 8 } });
    expect(r8.access.transactions).toBe(32);
  });
  it('CHWN with one sample is NCHW; with 8 samples the lanes are 32 bytes apart', () => {
    expect(warpReads({ ...base, layout: 'CHWN' }).access.transactions).toBe(5);
    const r = warpReads({ ...base, layout: 'CHWN', dims: { ...base.dims, N: 8 } });
    expect(r.stride).toBe(32);
    expect(r.access.transactions).toBe(32);
  });
  it('lanes on different samples: CHWN coalesces (4 sectors), NCHW does not', () => {
    const d = { N: 32, C: 4, H: 18, W: 18 };
    expect(warpReads({ ...base, dims: d, layout: 'CHWN', lanes: 'n' }).access.transactions).toBe(4);
    expect(warpReads({ ...base, dims: d, layout: 'NCHW', lanes: 'n' }).access.transactions).toBe(32);
  });
  it('every lane uses the same F address, and a 64-wide wavefront covers four rows', () => {
    const r = warpReads({ ...base, warpWidth: 64, c: 2, p: 1, q: 2 });
    expect(r.fIndex).toBe(((0 * 4 + 2) * 3 + 1) * 3 + 2);
    expect(r.lanes.length).toBe(64);
    expect(r.runs).toBe(4);
  });
});
