import { describe, expect, it } from 'vitest';
import { warpAccess } from './coalesce';
import {
  EXERCISE5,
  blocksFor,
  index,
  kernelTotals,
  owner,
  plan,
  segmentStart,
  warpFootprint,
  type LoadParams,
  type Partitioning,
} from './histcoarsen';

const NV = { warpWidth: 32, elemBytes: 1, segmentBytes: 32 };
const load = (over: Partial<LoadParams>): LoadParams => ({
  p: 'interleaved', n: 1 << 20, blockDim: 256, coarse: 4, block: 0, warp: 0, c: 0, ...NV, ...over,
});

describe("the book's drawings (Figs. 9.11 and 9.13: 8 threads, factor 3)", () => {
  const book = { n: 24, blockDim: 8, coarse: 3, warpWidth: 32 };
  it('contiguous: the first iteration reads cells 0, 3, ..., 21, the second 1, 4, ..., 22', () => {
    expect(warpFootprint(load({ ...book, p: 'contiguous', c: 0 })).elements).toEqual([0, 3, 6, 9, 12, 15, 18, 21]);
    expect(warpFootprint(load({ ...book, p: 'contiguous', c: 1 })).elements).toEqual([1, 4, 7, 10, 13, 16, 19, 22]);
  });
  it('interleaved: the first iteration reads cells 0 to 7, the second 8 to 15', () => {
    expect(warpFootprint(load({ ...book, p: 'interleaved', c: 0 })).elements).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(warpFootprint(load({ ...book, p: 'interleaved', c: 1 })).elements).toEqual([8, 9, 10, 11, 12, 13, 14, 15]);
  });
  it('a warp wider than the block holds only the block’s 8 threads', () => {
    expect(warpFootprint(load({ ...book, c: 2 })).lanes).toHaveLength(8);
  });
});

describe('line 12 (the questions q-index-interleaved and q-index-contiguous)', () => {
  it('block 2 starts at element 4·2·256 = 2048', () => {
    expect(segmentStart(2, 256, 4)).toBe(2048);
  });
  it('interleaved: thread 5 of block 2, c = 3 reads 2048 + 3·256 + 5 = 2821', () => {
    expect(index('interleaved', 2, 5, 3, 256, 4)).toBe(2821);
  });
  it('contiguous: the same thread reads 2048 + 5·4 + 3 = 2071; thread 6 reads 2075', () => {
    expect(index('contiguous', 2, 5, 3, 256, 4)).toBe(2071);
    expect(index('contiguous', 2, 6, 3, 256, 4)).toBe(2075);
  });
  it('owner() inverts line 12', () => {
    expect(owner('interleaved', 2821, 256, 4)).toEqual({ block: 2, thread: 5, c: 3 });
    expect(owner('contiguous', 2071, 256, 4)).toEqual({ block: 2, thread: 5, c: 3 });
  });
});

describe('every element is read exactly once, partial last block included', () => {
  const cases: [number, number, number][] = [
    [24, 8, 3],
    [1000, 32, 3],
    [1001, 64, 16],
    [4097, 256, 4],
    [5, 8, 3],
  ];
  for (const p of ['contiguous', 'interleaved'] as Partitioning[])
    for (const [n, B, C] of cases)
      it(`${p}, n = ${n}, blockDim ${B}, factor ${C}`, () => {
        const seen = new Uint8Array(n);
        const blocks = blocksFor(n, B, C);
        for (let b = 0; b < blocks; b++)
          for (let t = 0; t < B; t++)
            for (let c = 0; c < C; c++) {
              const i = index(p, b, t, c, B, C);
              if (i < n) seen[i]++;
              expect(owner(p, i, B, C)).toEqual({ block: b, thread: t, c });
            }
        expect([...seen].every((v) => v === 1)).toBe(true);
      });
});

describe('segments one warp touches (q-segments)', () => {
  it('one-byte pixels, contiguous, factor 32, 32 lanes, 32-byte segments: 32 segments, NVIDIA’s worst case', () => {
    const f = warpFootprint(load({ p: 'contiguous', coarse: 32 }));
    expect(f.transactions).toBe(32);
    expect(f.laneStride).toBe(32);
    expect(f.efficiency).toBeCloseTo(1 / 32);
  });
  it('the same load interleaved: 1 segment', () => {
    expect(warpFootprint(load({ p: 'interleaved', coarse: 32 })).transactions).toBe(1);
  });
  it('agrees with chapter 6’s coalescing model on the same addresses', () => {
    for (const p of ['contiguous', 'interleaved'] as Partitioning[])
      for (const coarse of [1, 3, 8, 32, 64])
        for (const [e, s, W] of [[1, 32, 32], [4, 32, 32], [1, 64, 64], [4, 128, 64], [1, 128, 32]])
          for (const c of [0, 1, coarse - 1]) {
            const q = load({ p, coarse, c, elemBytes: e, segmentBytes: s, warpWidth: W, block: 1, warp: 1 });
            const f = warpFootprint(q);
            expect(f.transactions).toBe(warpAccess(f.elements.map((x) => x * e), e, s).transactions);
          }
  });
  it('interleaved stays at ceil(warp × element / segment) whatever the factor', () => {
    for (const coarse of [1, 4, 16, 64])
      for (const [e, s, W] of [[1, 32, 32], [4, 32, 32], [4, 64, 64], [1, 128, 64], [4, 128, 32]])
        expect(warpFootprint(load({ coarse, elemBytes: e, segmentBytes: s, warpWidth: W })).transactions).toBe(Math.ceil((W * e) / s));
  });
  it('contiguous spreads to one segment per lane once a lane’s run fills a segment', () => {
    // 4-byte ids, 32-byte segments: 8 ids fill a segment, so factor 8 or more gives one per lane.
    expect(warpFootprint(load({ p: 'contiguous', coarse: 8, elemBytes: 4 })).transactions).toBe(32);
    expect(warpFootprint(load({ p: 'contiguous', coarse: 4, elemBytes: 4 })).transactions).toBe(16);
    expect(warpFootprint(load({ p: 'contiguous', coarse: 1, elemBytes: 4 })).transactions).toBe(4);
  });
  it('inactive lanes past the end touch nothing', () => {
    const f = warpFootprint(load({ n: 2048 + 10, block: 2, coarse: 4, c: 0 }));
    expect(f.active).toBe(10);
    expect(f.transactions).toBe(1);
  });
});

describe('totals over the kernel', () => {
  const brute = (q: { p: Partitioning; n: number; blockDim: number; coarse: number; warpWidth: number; elemBytes: number; segmentBytes: number }) => {
    let loads = 0;
    let tx = 0;
    for (let b = 0; b < blocksFor(q.n, q.blockDim, q.coarse); b++)
      for (let w = 0; w < Math.ceil(q.blockDim / q.warpWidth); w++)
        for (let c = 0; c < q.coarse; c++) {
          const f = warpFootprint({ ...q, block: b, warp: w, c });
          if (f.active) {
            loads++;
            tx += f.transactions;
          }
        }
    return { loads, tx };
  };
  it('match a lane-by-lane count', () => {
    for (const p of ['contiguous', 'interleaved'] as Partitioning[])
      for (const [n, B, C, W, e, s] of [[24, 8, 3, 32, 1, 32], [5000, 64, 7, 32, 4, 32], [65536, 256, 16, 64, 1, 128], [1000, 96, 5, 64, 4, 64]]) {
        const q = { p, n, blockDim: B, coarse: C, warpWidth: W, elemBytes: e, segmentBytes: s };
        const t = kernelTotals(q);
        const r = brute(q);
        expect(t.loads).toBe(r.loads);
        expect(t.transactions).toBe(r.tx);
        expect(t.bytesNeeded).toBe(n * e);
      }
  });
  it('one-byte pixels at factor 16: interleaved fetches each byte once, contiguous 16 times over', () => {
    const q = { n: 1 << 16, blockDim: 256, coarse: 16, ...NV };
    expect(kernelTotals({ ...q, p: 'interleaved' }).efficiency).toBe(1);
    expect(kernelTotals({ ...q, p: 'contiguous' }).efficiency).toBeCloseTo(1 / 16);
  });
});

describe('blocks, copies, commits (exercise 5, p. 220)', () => {
  const { n, bins, blockDim, coarse } = EXERCISE5;
  it('5a: no privatisation, one global atomic per element: 524,288', () => {
    expect(plan(n, blockDim, 1, bins).unprivatisedAtomics).toBe(524_288);
  });
  it('5b: privatised, no coarsening: 512 blocks × 128 bins = 65,536', () => {
    const r = plan(n, blockDim, 1, bins);
    expect(r.blocks).toBe(512);
    expect(r.commitAtomicsMax).toBe(65_536);
  });
  it('5c: factor 4: 128 blocks × 128 bins = 16,384', () => {
    const r = plan(n, blockDim, coarse, bins);
    expect(r.blocks).toBe(128);
    expect(r.copies).toBe(128);
    expect(r.zeroWrites).toBe(128 * 128);
    expect(r.commitAtomicsMax).toBe(16_384);
    expect(r.sharedAtomics).toBe(524_288);
  });
  it('a block commits no more bins than it has pixels', () => {
    // 8 pixels per block cannot fill 256 bins; the last block has 4 pixels.
    const r = plan(20, 8, 1, 256);
    expect(r.blocks).toBe(3);
    expect(r.commitAtomicsMax).toBe(8 + 8 + 4);
  });
});
