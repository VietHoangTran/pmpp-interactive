import { describe, expect, it } from 'vitest';
import { linearThread, threadAt } from './grid';
import {
  BOOK_CASES,
  BOOK_MAX_THREADS,
  BOOK_WIDTH,
  boundary1D,
  boundary2D,
  causal,
  classifyBlock,
  classifyGrid,
  coordOf,
  cut,
  dims,
  gridFor,
  laneOf,
  linear,
  linearWorking,
  members,
  paddedLanes,
  place,
  regions,
  reportedMaxThreads,
  reportedWidth,
  rowsPerWarp,
  splitWarp1D,
  stateOf,
  threadsIn,
  warpOf,
  warpSpan,
  warpsInGrid,
  warpsInRightEdgeBlocks,
  warpsPerBlock,
  widthChoices,
  type Picture,
} from './warps';

const W = BOOK_WIDTH;

describe('the linear order (book pp. 75 to 76; facts A8)', () => {
  it('x moves fastest, y has a stride of blockDim.x, z a stride of blockDim.x * blockDim.y', () => {
    const d = dims(4, 8, 2);
    expect(linear(1, 0, 0, d) - linear(0, 0, 0, d)).toBe(1);
    expect(linear(0, 1, 0, d) - linear(0, 0, 0, d)).toBe(4);
    expect(linear(0, 0, 1, d) - linear(0, 0, 0, d)).toBe(32);
  });
  it('coordOf undoes linear for every thread of several shapes', () => {
    for (const d of [dims(4, 4), dims(8, 8), dims(4, 8, 2), dims(48), dims(3, 5, 7), dims(16, 16)])
      for (let i = 0; i < threadsIn(d); i++) {
        const c = coordOf(i, d);
        expect(linear(c.x, c.y, c.z, d)).toBe(i);
      }
  });
  it("agrees with chapter 3's model of the same rule (grid.ts)", () => {
    const d = dims(4, 8, 2);
    for (let i = 0; i < threadsIn(d); i++) {
      const c = coordOf(i, d);
      expect(linearThread(c, d)).toBe(i);
      expect(threadAt(i, d)).toEqual(c);
    }
  });
  it('writes the arithmetic out with the block own numbers', () => {
    expect(linearWorking(3, 5, 0, dims(8, 8))).toBe('3 + 5·8 = 43');
    expect(linearWorking(1, 2, 1, dims(4, 8, 2))).toBe('1 + 2·4 + 1·4·8 = 41');
    expect(linearWorking(40, 0, 0, dims(48))).toBe('40');
  });
});

describe('counting warps (book pp. 74 to 75, Fig. 4.6)', () => {
  it('a 256-thread block is 8 warps, three such blocks 24', () => {
    expect(warpsPerBlock(dims(256), W)).toBe(8);
    expect(warpsInGrid(3, dims(256), W)).toBe(24);
  });
  it('warp n of a 1D block holds threads 32n to 32(n + 1) - 1', () => {
    expect(warpSpan(dims(256), W, 0)).toMatchObject({ first: { x: 0 }, last: { x: 31 }, count: 32 });
    expect(warpSpan(dims(256), W, 1)).toMatchObject({ first: { x: 32 }, last: { x: 63 }, count: 32 });
    expect(warpSpan(dims(256), W, 7)).toMatchObject({ first: { x: 224 }, last: { x: 255 } });
  });
  it('48 threads: 2 warps, the second padded with 16 inactive lanes', () => {
    expect(cut(dims(48), W)).toEqual({ threads: 48, width: 32, warps: 2, padded: 16, rowsPerWarp: null });
    expect(warpSpan(dims(48), W, 1).count).toBe(16);
  });
  it('48 threads at width 64 (facts B6): 1 warp, 16 lanes idle', () => {
    expect(warpsPerBlock(dims(48), 64)).toBe(1);
    expect(paddedLanes(dims(48), 64)).toBe(16);
  });
  it('exercise 1a and 1b: 128-thread blocks are 4 warps, the 8 blocks of N = 1024 are 32', () => {
    expect(warpsPerBlock(dims(128), W)).toBe(4);
    expect(warpsInGrid(Math.ceil(1024 / 128), dims(128), W)).toBe(32);
  });
});

describe("the book's cuts of multidimensional blocks (pp. 75 to 76, Fig. 4.7)", () => {
  it('4 x 4: half a warp, padded with 16 more positions', () => {
    const c = cut(dims(4, 4), W);
    expect(c.warps).toBe(1);
    expect(c.padded).toBe(16);
    // Fig. 4.7's strip: T(0,0) T(0,1) T(0,2) T(0,3) T(1,0) ...  in the book's T(y,x) notation.
    expect(members(dims(4, 4), W)[0].slice(0, 5).map((t) => `${t.y},${t.x}`)).toEqual(['0,0', '0,1', '0,2', '0,3', '1,0']);
  });
  it('8 x 8: two warps, T(0,0)..T(3,7) and T(4,0)..T(7,7)', () => {
    expect(warpsPerBlock(dims(8, 8), W)).toBe(2);
    expect(warpSpan(dims(8, 8), W, 0)).toMatchObject({ first: { x: 0, y: 0 }, last: { x: 7, y: 3 }, count: 32 });
    expect(warpSpan(dims(8, 8), W, 1)).toMatchObject({ first: { x: 0, y: 4 }, last: { x: 7, y: 7 }, count: 32 });
    expect(rowsPerWarp(dims(8, 8), W)).toBe(4);
  });
  it('2 x 8 x 4 (x 4, y 8, z 2): warp 0 is all of z = 0 and warp 1 all of z = 1', () => {
    const d = dims(4, 8, 2);
    expect(warpSpan(d, W, 0)).toMatchObject({ first: { x: 0, y: 0, z: 0 }, last: { x: 3, y: 7, z: 0 } });
    expect(warpSpan(d, W, 1)).toMatchObject({ first: { x: 0, y: 0, z: 1 }, last: { x: 3, y: 7, z: 1 } });
    expect(members(d, W).map((m) => new Set(m.map((t) => t.z)))).toEqual([new Set([0]), new Set([1])]);
  });
  it('the questions of section 4.4', () => {
    expect(place(3, 5, 0, dims(8, 8), W)).toEqual({ index: 43, warp: 1, lane: 11 });
    expect(place(1, 2, 1, dims(4, 8, 2), W)).toEqual({ index: 41, warp: 1, lane: 9 });
  });
  it('a 16 x 16 block: a warp is two rows at 32 and four rows at 64 (p. 82; facts B6)', () => {
    expect(rowsPerWarp(dims(16, 16), 32)).toBe(2);
    expect(warpsPerBlock(dims(16, 16), 32)).toBe(8);
    expect(rowsPerWarp(dims(16, 16), 64)).toBe(4);
    expect(warpsPerBlock(dims(16, 16), 64)).toBe(4);
    expect(warpsPerBlock(dims(8, 8), 64)).toBe(1);
    expect(warpsPerBlock(dims(64), 64)).toBe(1);
  });
  it('the same block is cut differently at 64: threads that were in different warps now share one', () => {
    const d = dims(8, 8);
    expect(place(0, 4, 0, d, 32).warp).toBe(1);
    expect(place(0, 4, 0, d, 64)).toEqual({ index: 32, warp: 0, lane: 32 });
  });
  it('a row of 64 is two warps at 32 and one at 64; a row of 32 is half a warp at 64', () => {
    expect(rowsPerWarp(dims(64, 2), 32)).toBe(0.5);
    expect(rowsPerWarp(dims(64, 2), 64)).toBe(1);
    expect(rowsPerWarp(dims(32, 8), 64)).toBe(2);
    expect(rowsPerWarp(dims(96, 2), 32)).toBeCloseTo(1 / 3);
  });
  it('warps cut through rows when neither blockDim.x nor the width divides the other', () => {
    expect(rowsPerWarp(dims(12, 8), 32)).toBeNull();
    expect(rowsPerWarp(dims(48, 2), 32)).toBeNull();
    expect(warpOf(31, 32)).toBe(0);
    expect(laneOf(33, 32)).toBe(1);
  });
});

describe('what a device report says about the width', () => {
  it('reads warpSize from CUDA and HIP reports only', () => {
    expect(reportedWidth('cuda', { warpSize: '32' })).toBe(32);
    expect(reportedWidth('hip', { warpSize: 64 })).toBe(64);
    expect(reportedWidth('metal', { name: 'a Metal device' })).toBeNull();
    expect(reportedWidth('emu', { warpSize: '1' })).toBeNull();
    expect(reportedWidth('race', { warpSize: '32' })).toBeNull();
    expect(reportedWidth('hip', { warpSize: '0' })).toBeNull();
    expect(reportedMaxThreads('cuda', { maxThreadsPerBlock: '1024' })).toBe(1024);
    expect(reportedMaxThreads('metal', {})).toBeNull();
    expect(BOOK_MAX_THREADS).toBe(1024);
  });
  it('offers the book 32, the documented 64, and each CUDA or HIP device that reported a width', () => {
    expect(widthChoices({}).map((c) => [c.key, c.width, c.maxThreads, c.source])).toEqual([
      ['book', 32, 1024, 'book'],
      ['w64', 64, 1024, 'doc'],
    ]);
    const c = widthChoices({
      metal: { device: { name: 'a Metal device' } },
      emu: { device: { warpSize: '1', maxThreadsPerBlock: '1024' } },
      hip: { device: { name: 'an AMD device', warpSize: '64', maxThreadsPerBlock: '256' } },
      cuda: { device: { warpSize: '32' } },
    });
    expect(c.map((x) => x.key)).toEqual(['book', 'w64', 'cuda', 'hip']);
    expect(c[3]).toMatchObject({ label: 'yours: an AMD device (hip), 64', width: 64, maxThreads: 256, source: 'report' });
    expect(c[2]).toMatchObject({ width: 32, maxThreads: 1024 });
  });
});

describe('classifying one warp', () => {
  it('all pass: full; none: idle; some: divergent', () => {
    expect(stateOf(32, 32)).toBe('full');
    expect(stateOf(0, 32)).toBe('idle');
    expect(stateOf(11, 32)).toBe('divergent');
  });
  it('padded lanes are not threads: a 48-thread block that passes everywhere has two full warps', () => {
    expect(classifyBlock(dims(48), W, () => true)).toEqual({ states: ['full', 'full'], passing: [32, 16], real: [32, 16] });
  });
});

// notes/today/ch04-facts.md, table E1: [case id, warps at 32, divergent at 32, warps at 64, divergent at 64]
const E1: [string, number, number, number, number][] = [
  ['v1003', 32, 1, 16, 1],
  ['v100', 4, 1, 2, 1],
  ['v1000', 32, 1, 16, 1],
  ['v10000', 314, 1, 157, 1],
  ['ex3', 64, 1, 32, 1],
  ['p76x62', 160, 31, 80, 20],
  ['p200x150', 1040, 75, 520, 50],
  ['p150x200', 1040, 100, 520, 50],
];
const brute = (p: Picture, width: number) => {
  const { gridX, gridY } = gridFor(p);
  return classifyGrid(gridX, gridY, dims(p.blockX, p.blockY), width, (c, r) => c < p.width && r < p.height);
};

describe('boundary checks: the facts file table E1, at widths 32 and 64', () => {
  for (const [id, w32, d32, w64, d64] of E1) {
    const c = BOOK_CASES.find((b) => b.id === id)!;
    it(`${c.label}: ${d32} of ${w32} at 32, ${d64} of ${w64} at 64`, () => {
      const a = boundary2D(c.picture, 32);
      const b = boundary2D(c.picture, 64);
      expect([a.warps, a.divergent, b.warps, b.divergent]).toEqual([w32, d32, w64, d64]);
      expect(a.full + a.divergent + a.idle).toBe(a.warps);
    });
    it(`${c.label}: counting by regions equals counting thread by thread`, () => {
      for (const width of [32, 64]) {
        const fast = boundary2D(c.picture, width);
        const slow = brute(c.picture, width);
        expect([fast.warps, fast.full, fast.divergent, fast.idle]).toEqual([slow.warps, slow.full, slow.divergent, slow.idle]);
      }
    });
  }
  it('every case is in the table and the table has no other', () => {
    expect(BOOK_CASES.map((c) => c.id).sort()).toEqual(E1.map((e) => e[0]).sort());
  });
});

describe('the 1D examples (book pp. 81 to 82, exercises 2 and 3)', () => {
  it('1003 elements in 64-thread blocks: 16 blocks, 1024 threads, 21 disabled in the last warp of block 15', () => {
    const b = boundary1D(1003, 64, W);
    expect([b.blocks, b.threads, b.warps, b.divergent]).toEqual([16, 1024, 32, 1]);
    expect(splitWarp1D(1003, 64, W)).toEqual({ block: 15, warp: 1, valid: 11, invalid: 21 });
  });
  it('the share falls with size: 1 of 4, 1 of 32, 1 of 314', () => {
    expect(boundary1D(100, 64, W).share).toBe(1 / 4);
    expect(boundary1D(1000, 64, W).share).toBe(1 / 32);
    expect(boundary1D(10000, 64, W).share).toBe(1 / 314);
  });
  it("the book's 313 for 10,000 elements is the number of warps that hold a valid element; one more is launched and idle", () => {
    const b = boundary1D(10000, 64, W);
    expect(b.warps - b.idle).toBe(313);
    expect(b.idle).toBe(1);
  });
  it('exercise 2: 2000 elements in 512-thread blocks launch 2048 threads', () => {
    expect(boundary1D(2000, 512, W).threads).toBe(2048);
  });
  it('exercise 3: warp 62 holds 16 valid and 16 invalid threads, warp 63 is idle', () => {
    const b = boundary1D(2000, 512, W);
    expect([b.divergent, b.idle]).toEqual([1, 1]);
    const s = splitWarp1D(2000, 512, W)!;
    expect(s.block * (512 / W) + s.warp).toBe(62);
    expect([s.valid, s.invalid]).toEqual([16, 16]);
  });
  it('a 1D launch never has more than one divergent warp, at any size', () => {
    for (const block of [32, 48, 64, 128, 512])
      for (const width of [32, 64])
        for (let n = 1; n <= 700; n += 7) expect(boundary1D(n, block, width).divergent).toBeLessThanOrEqual(1);
  });
  it('a size that ends on a warp boundary splits nothing', () => {
    expect(boundary1D(96, 64, W).divergent).toBe(0);
    expect(splitWarp1D(96, 64, W)).toBeNull();
    expect(splitWarp1D(128, 64, W)).toBeNull();
    expect(boundary1D(96, 64, 64).divergent).toBe(1);
  });
});

describe('the 62 by 76 picture and its four regions (book p. 82)', () => {
  const p: Picture = { width: 76, height: 62, blockX: 16, blockY: 16 };
  const r = regions(p, W);
  it('20 blocks of 8 warps: 12, 3, 4 and 1 blocks', () => {
    expect(boundary2D(p, W).blocks).toBe(20);
    expect(r.map((x) => x.blocks)).toEqual([12, 3, 4, 1]);
    expect(r.map((x) => x.warps)).toEqual([96, 24, 32, 8]);
  });
  it('region 1: none diverge; region 2: all 24 do (12 valid and 4 invalid columns in every row)', () => {
    expect(r[0].divergent).toBe(0);
    expect(r[1].divergent).toBe(24);
    expect(r[1].validX).toBe(12);
  });
  it('region 3: the bottom warp of each block lies outside the picture and does not diverge', () => {
    expect(r[2].divergent).toBe(0);
    expect(r[2].states).toEqual(['full', 'full', 'full', 'full', 'full', 'full', 'full', 'idle']);
  });
  it('region 4: the first seven warps diverge, the last does not', () => {
    expect(r[3].states).toEqual([...Array(7).fill('divergent'), 'idle']);
  });
  it('31 of 160 in all', () => {
    const b = boundary2D(p, W);
    expect([b.divergent, b.warps]).toEqual([31, 160]);
  });
  it('an odd height splits the bottom-edge warps too, as the book remarks', () => {
    const odd = regions({ ...p, height: 61 }, W);
    expect(odd[2].divergent).toBe(4);
  });
  it('at width 64 a warp is four rows, so the bottom edge splits a warp in every block it crosses: 20 of 80', () => {
    const b = boundary2D(p, 64);
    expect([b.divergent, b.warps]).toEqual([20, 80]);
    expect(b.regions.map((x) => x.divergent)).toEqual([0, 12, 4, 4]);
  });
});

describe('the "200 x 150" picture (book p. 82; notes erratum 4)', () => {
  const wide: Picture = { width: 200, height: 150, blockX: 16, blockY: 16 };
  const tall: Picture = { width: 150, height: 200, blockX: 16, blockY: 16 };
  it('130 blocks and 1040 warps under either reading', () => {
    for (const p of [wide, tall]) expect([boundary2D(p, W).blocks, boundary2D(p, W).warps]).toEqual([130, 1040]);
  });
  it("the printed region sizes 864, 72, 96, 8 are those of the 200-wide reading", () => {
    expect(regions(wide, W).map((r) => r.warps)).toEqual([864, 72, 96, 8]);
    expect(regions(tall, W).map((r) => r.warps)).toEqual([864, 96, 72, 8]);
  });
  it('200 wide: 72 + 0 + 3 = 75 divergent, not 80', () => {
    expect(regions(wide, W).map((r) => r.divergent)).toEqual([0, 72, 0, 3]);
    expect(boundary2D(wide, W).divergent).toBe(75);
  });
  it('150 wide: 96 + 0 + 4 = 100 divergent, 9.6%, which is not under 8%', () => {
    expect(regions(tall, W).map((r) => r.divergent)).toEqual([0, 96, 0, 4]);
    const b = boundary2D(tall, W);
    expect(b.divergent).toBe(100);
    expect(Math.round(b.share * 1000) / 10).toBe(9.6);
  });
  it('80 is every warp of every block the right edge crosses, split or not (200 wide)', () => {
    expect(warpsInRightEdgeBlocks(wide, W)).toBe(80);
    expect(warpsInRightEdgeBlocks(tall, W)).toBe(104);
  });
});

describe('launch shapes from the project (ours)', () => {
  it('a 1 x 4096 output in 16 x 16 blocks: 256 divergent and 1792 idle of 2048', () => {
    const b = boundary2D({ width: 4096, height: 1, blockX: 16, blockY: 16 }, W);
    expect([b.warps, b.divergent, b.idle, b.full]).toEqual([2048, 256, 1792, 0]);
  });
  it('a causal mask over 1024 x 1024 scores, blocks 32 wide and 8 high: 992 of 32,768 diverge', () => {
    const t = classifyGrid(1024 / 32, 1024 / 8, dims(32, 8), W, causal);
    expect([t.warps, t.divergent]).toEqual([32768, 992]);
    expect(Math.round(t.share * 1000) / 10).toBe(3);
  });
  it('classifyGrid keeps the per-block states when asked', () => {
    const t = classifyGrid(2, 1, dims(4, 2), 4, (c) => c < 6, true);
    expect(t.states).toEqual([[['full', 'full'], ['divergent', 'divergent']]]);
    expect(classifyGrid(2, 1, dims(4, 2), 4, (c) => c < 6).states).toEqual([]);
  });
});
