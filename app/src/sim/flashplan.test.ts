import { describe, expect, it } from 'vitest';
import {
  AMD_LDS_MIN,
  BOOK,
  CUDA_STATIC,
  METAL_DOCUMENTED,
  addr,
  fits,
  hipShuffleSource,
  ktStoreBanks,
  laneElements,
  largestFittingBc,
  limits,
  ownerOf,
  plan,
  tileClasses,
  tileCounts,
  wave64Broadcast,
} from './flashplan';

describe('Fig. 20.9 constants and footprint (notes [checked]; facts F1)', () => {
  const p = plan(BOOK);
  it('16 warps, 2 rows per warp, d_size 4', () => {
    expect(p.nWarps).toBe(16);
    expect(p.brWarp).toBe(2);
    expect(p.dSize).toBe(4);
    expect(p.problems).toEqual([]);
  });
  it('4096 + 128 + 1024 + 4096 floats = 37,376 bytes', () => {
    expect(p.floats).toEqual({ kt: 4224, ktPad: 128, s: 1024, v: 4096, total: 9344 });
    expect(p.bytes.total).toBe(37376);
  });
  it('20 floats of registers per thread (8 + 8 + 2 + 2)', () => {
    expect(p.regs).toEqual({ o: 8, q: 8, m: 2, D: 2, total: 20 });
  });
  it('fits CUDA’s 48 KB and AMD’s LDS, not Metal’s 32 KB', () => {
    expect(fits(p.bytes.total, CUDA_STATIC)).toBe(true);
    expect(fits(p.bytes.total, AMD_LDS_MIN)).toBe(true);
    expect(fits(p.bytes.total, METAL_DOCUMENTED)).toBe(false);
  });
  it('the sizes that fit Metal: 18,688 (B_c 16), 20,736 (d 64), 17,664 (B_r = B_c = 16)', () => {
    expect(plan({ ...BOOK, bc: 16 }).bytes.total).toBe(18688);
    expect(plan({ ...BOOK, d: 64 }).bytes.total).toBe(20736);
    expect(plan({ ...BOOK, br: 16, bc: 16 }).bytes.total).toBe(17664);
    for (const b of [18688, 20736, 17664]) expect(fits(b, METAL_DOCUMENTED)).toBe(true);
  });
  it('the largest of 32, 16, 8 that fits 32 KB at d = 128, B_r = 32 is 16', () => {
    const { bc: _bc, ...rest } = BOOK;
    expect(largestFittingBc(rest, [32, 16, 8], METAL_DOCUMENTED)).toBe(16);
  });
  it('B_r_warp must be whole: B_r = 8 with 16 warps is refused', () => {
    const q = plan({ ...BOOK, br: 8 });
    expect(q.brWarp).toBe(0.5);
    expect(q.problems.length).toBe(1);
    expect(q.regs.total).toBe(0);
  });
  it('a 64-wide logical warp: 8 warps, 4 rows each, d_size 2 (facts B2)', () => {
    const q = plan({ ...BOOK, warp: 64 });
    expect([q.nWarps, q.brWarp, q.dSize]).toEqual([8, 4, 2]);
  });
  it('padding off removes the 128 extra floats', () => {
    expect(plan({ ...BOOK, banks: 0 }).bytes.total).toBe(37376 - 128 * 4);
  });
  it('the Metal limit comes from the report when there is one', () => {
    expect(limits()[2].bytes).toBe(32768);
    expect(limits(65536, 'Some Mac').find((l) => l.id === 'metal')!.bytes).toBe(65536);
  });
});

describe('who holds Q element dd (Figs. 20.10, 20.12)', () => {
  it('dd = 77 lives in lane 13, slot 2', () => {
    expect(ownerOf(77)).toEqual({ lane: 13, slot: 2 });
  });
  it('lane L holds L, L + 32, L + 64, L + 96 at d = 128', () => {
    expect(laneElements(5, 128)).toEqual([5, 37, 69, 101]);
    for (let dd = 0; dd < 128; dd++) expect(laneElements(ownerOf(dd).lane, 128)[ownerOf(dd).slot]).toBe(dd);
  });
});

describe('HIP shuffle on a 64-wide wavefront (facts B2)', () => {
  it('hardware lane 40 asking for lane 5 reads lane 5 at the default width 64, lane 37 at width 32', () => {
    expect(hipShuffleSource(40, 5, 64)).toBe(5);
    expect(hipShuffleSource(40, 5, 32)).toBe(37);
  });
  it('default width: every lane of the second logical warp reads the first warp’s lane', () => {
    const r = wave64Broadcast(77, 64);
    expect(r.filter((x) => !x.ok).map((x) => x.hw)).toEqual(Array.from({ length: 32 }, (_, i) => 32 + i));
    expect(r.every((x) => x.gets === 13)).toBe(true);
  });
  it('width 32: every lane reads its own logical warp’s owner', () => {
    expect(wave64Broadcast(77, 32).every((x) => x.ok)).toBe(true);
  });
});

describe('banks hit by the K^T store (Fig. 20.11)', () => {
  it('addr() inserts one slot per 32 elements', () => {
    expect(addr(31, 32)).toBe(31);
    expect(addr(32, 32)).toBe(33);
    expect(addr(4095, 32)).toBe(4095 + 127);
    expect(addr(100, 0)).toBe(100);
  });
  it('B_c = 32 without padding: all 32 lanes in one bank, 32 passes; with padding: 32 banks, 1 pass', () => {
    const a = ktStoreBanks(32, 0, false);
    expect([a.banksUsed, a.passes]).toEqual([1, 32]);
    const b = ktStoreBanks(32, 0, true);
    expect([b.banksUsed, b.passes]).toEqual([32, 1]);
    expect(b.index.slice(0, 3)).toEqual([0, 33, 66]);
  });
  it('B_c = 16 and 8: 16 and 8 passes unpadded, 1 padded', () => {
    expect(ktStoreBanks(16, 0, false).passes).toBe(16);
    expect(ktStoreBanks(16, 0, true).passes).toBe(1);
    expect(ktStoreBanks(8, 0, false).passes).toBe(8);
    expect(ktStoreBanks(8, 0, true).passes).toBe(1);
  });
  it('B_c = 64: one slot of padding per 32 still leaves 2 lanes per bank', () => {
    expect(ktStoreBanks(64, 0, false).passes).toBe(32);
    expect(ktStoreBanks(64, 0, true).passes).toBe(2);
  });
  it('every jj behaves the same', () => {
    for (let jj = 0; jj < 32; jj++) expect(ktStoreBanks(32, jj, true).passes).toBe(1);
  });
});

describe('causal tile classes (facts A7)', () => {
  it('N = 4096, 32 × 32: 127/256 of the pairs lie wholly above the diagonal', () => {
    const t = tileClasses(4096, 32, 32);
    expect(t.total).toBe(128 * 128);
    expect(t.counts.skip).toBe(8128);
    expect(t.counts.skip / t.total).toBeCloseTo(127 / 256, 12);
    expect(t.counts.mask).toBe(128);
    expect(t.counts.full).toBe(8128);
  });
  it('only diagonal tiles need a mask when the tiles are square', () => {
    const t = tileClasses(1024, 64, 64);
    t.grid.forEach((row, i) => row.forEach((c, j) => expect(c).toBe(j > i ? 'skip' : j === i ? 'mask' : 'full')));
  });
  it('a ragged last tile is classified by its real extent', () => {
    const t = tileClasses(100, 32, 32);
    expect([t.tr, t.tc]).toEqual([4, 4]);
    expect(t.grid[3][3]).toBe('mask');
  });
  it('the book’s integer division drops rows when N is not a multiple of B_r; the ceiling keeps them', () => {
    expect(tileCounts(100, 32, 32)).toEqual({ trBook: 3, tcBook: 3, tr: 4, tc: 4, rowsDropped: 4 });
  });
});
