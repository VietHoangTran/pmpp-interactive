import { describe, expect, it } from 'vitest';
import {
  SEGMENT_MODELS,
  exercise1,
  exprAccess,
  laneAddresses,
  runs,
  strideEfficiency,
  tileLoad,
  warpAccess,
  type AccessParams,
} from './coalesce';

const P = (over: Partial<AccessParams> = {}): AccessParams => ({
  Width: 1024, k: 0, row: 0, stride: 1, offset: 0, warpWidth: 32, lane0: 0, ...over,
});
const floats = (n: number, from = 0, step = 1) => Array.from({ length: n }, (_, l) => (from + l * step) * 4);

describe('warpAccess: NVIDIA\'s documented cases (facts A2, A3, A11)', () => {
  it('32 consecutive aligned floats: four 32-byte sectors, every byte used', () => {
    const a = warpAccess(floats(32), 4, 32);
    expect(a).toEqual({ segments: [0, 1, 2, 3], transactions: 4, bytesFetched: 128, bytesNeeded: 128, efficiency: 1 });
  });
  it('the same run starting one float past a sector boundary: five sectors', () => {
    const a = warpAccess(floats(32, 1), 4, 32);
    expect(a.transactions).toBe(5);
    expect(a.segments).toEqual([0, 1, 2, 3, 4]);
    expect(a.bytesNeeded).toBe(128);
  });
  it('stride 2: half of every fetched byte is wasted', () => {
    const a = warpAccess(floats(32, 0, 2), 4, 32);
    expect(a.transactions).toBe(8);
    expect(a.efficiency).toBe(0.5);
  });
  it('lanes 32 bytes or more apart: one sector per lane, 12.5 %', () => {
    for (const s of [8, 9, 16, 64, 1024]) {
      const a = warpAccess(floats(32, 0, s), 4, 32);
      expect(a.transactions).toBe(32);
      expect(a.bytesFetched).toBe(1024);
      expect(a.efficiency).toBe(0.125);
    }
  });
  it('a permutation inside the same sectors costs the same as consecutive order', () => {
    const perm = floats(32).reverse();
    expect(warpAccess(perm, 4, 32).transactions).toBe(4);
    expect(runs(perm, 4)).toBe(1);
  });
  it('a warp of float4 loads: 512 bytes, 16 sectors (Nsight\'s optimum for 128-bit accesses)', () => {
    const a = warpAccess(Array.from({ length: 32 }, (_, l) => l * 16), 16, 32);
    expect(a.transactions).toBe(16);
    expect(a.efficiency).toBe(1);
  });
  it('a lane straddling a boundary touches both segments', () => {
    expect(warpAccess([28], 8, 32).segments).toEqual([0, 1]);
  });
  it('an empty warp touches nothing', () => {
    expect(warpAccess([], 4, 32)).toEqual({ segments: [], transactions: 0, bytesFetched: 0, bytesNeeded: 0, efficiency: 0 });
  });
});

describe('the book\'s matmul reads (Figs. 6.2, 6.3)', () => {
  it('row-major N, N[k*Width + col]: 4 sectors at every k', () => {
    for (const k of [0, 1, 7, 1023]) {
      const a = exprAccess('n-row', P({ k }), 32);
      expect(a.transactions).toBe(4);
      expect(a.efficiency).toBe(1);
      expect(a.runs).toBe(1);
    }
  });
  it('column-major N at Width 1024: lanes 4096 bytes apart, 32 sectors', () => {
    const a = exprAccess('n-col', P(), 32);
    expect(a.lanes[1].byte - a.lanes[0].byte).toBe(4096);
    expect(a.transactions).toBe(32);
    expect(a.efficiency).toBe(0.125);
  });
  it('column-major N gives every lane its own sector once Width is 8 floats or more', () => {
    expect(exprAccess('n-col', P({ Width: 4 }), 32).transactions).toBe(2); // four lanes active, 16 bytes apart
    expect(exprAccess('n-col', P({ Width: 7 }), 32).transactions).toBeLessThan(7);
    for (const Width of [8, 16, 100, 2048]) {
      const a = exprAccess('n-col', P({ Width }), 32);
      expect(a.transactions).toBe(Math.min(32, Width)); // lanes past col = Width - 1 are idle (the guard)
      expect(a.efficiency).toBe(0.125);
    }
  });
  it('the M read is one address for the whole warp: one sector, 4 bytes needed', () => {
    const a = exprAccess('m-row', P({ row: 5, k: 3 }), 32);
    expect(a.transactions).toBe(1);
    expect(a.bytesNeeded).toBe(4);
    expect(new Set(a.lanes.map((l) => l.element))).toEqual(new Set([5 * 1024 + 3]));
  });
  it('with 16-wide blocks a warp spans two rows: the M read is two addresses, two sectors', () => {
    const a = exprAccess('m-row', P({ blockDimX: 16 }), 32);
    expect(new Set(a.lanes.map((l) => l.element)).size).toBe(2);
    expect(a.transactions).toBe(2);
  });
  it('the book\'s 4-wide toy: only four lanes are active', () => {
    expect(laneAddresses('n-row', P({ Width: 4 })).map((l) => l.element)).toEqual([0, 1, 2, 3]);
    expect(laneAddresses('n-col', P({ Width: 4 })).map((l) => l.element)).toEqual([0, 4, 8, 12]);
    expect(laneAddresses('n-col', P({ Width: 4, k: 1 })).map((l) => l.element)).toEqual([1, 5, 9, 13]);
  });
  it('a 64-lane wavefront of consecutive floats spans two 128-byte lines (AMD, B1)', () => {
    const a = exprAccess('n-row', P({ warpWidth: 64 }), 128);
    expect(a.transactions).toBe(2);
    expect(a.bytesNeeded).toBe(256);
    expect(exprAccess('n-row', P({ warpWidth: 64 }), 64).transactions).toBe(4);
  });
});

describe('strides and offsets', () => {
  it('offset by one float adds a fifth sector; by eight floats (one sector) it does not', () => {
    expect(exprAccess('offset', P({ offset: 1 }), 32).transactions).toBe(5);
    expect(exprAccess('offset', P({ offset: 8 }), 32).transactions).toBe(4);
    expect(exprAccess('offset', P({ offset: 8 }), 128).transactions).toBe(2);
  });
  it('strideEfficiency gives the model curves', () => {
    expect([1, 2, 4, 8, 16, 32, 64].map((s) => strideEfficiency(s, 32))).toEqual([1, 0.5, 0.25, 0.125, 0.125, 0.125, 0.125]);
    expect([1, 2, 4, 8, 16, 32, 64].map((s) => strideEfficiency(s, 64))).toEqual([1, 0.5, 0.25, 0.125, 0.0625, 0.0625, 0.0625]);
    expect([1, 2, 4, 8, 16, 32, 64].map((s) => strideEfficiency(s, 128))).toEqual([1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.03125]);
  });
});

describe('segment models', () => {
  it('each documented size, and none for Apple', () => {
    expect(SEGMENT_MODELS.nvidia.segmentBytes).toBe(32);
    expect(SEGMENT_MODELS.nvidia.lineBytes).toBe(128);
    expect(SEGMENT_MODELS.amd64.segmentBytes).toBe(64);
    expect(SEGMENT_MODELS.amd128.segmentBytes).toBe(128);
    expect(SEGMENT_MODELS.metal.segmentBytes).toBeNull();
  });
});

describe('corner turning (Fig. 6.4)', () => {
  const toy = (arrangement: 'ch5' | 'turned') =>
    tileLoad({ layout: 'col', arrangement, T: 4, bx: 1, ph: 2, Width: 16, warpWidth: 32 }).threads;
  const at = (ts: ReturnType<typeof toy>, tx: number, ty: number) => ts.find((t) => t.tx === tx && t.ty === ty)!;
  it('16 × 16 column-major N, T = 4, bx = 1, ph = 2: turned, thread (3, 1) loads 91 and its neighbour 90', () => {
    expect(at(toy('turned'), 3, 1).element).toBe(91);
    expect(at(toy('turned'), 2, 1).element).toBe(90);
    expect([at(toy('turned'), 3, 1).r, at(toy('turned'), 3, 1).c]).toEqual([11, 5]);
  });
  it('chapter 5\'s mapping: the same two threads load 121 and 105, sixteen apart', () => {
    expect(at(toy('ch5'), 3, 1).element).toBe(121);
    expect(at(toy('ch5'), 2, 1).element).toBe(105);
  });
  it('both arrangements load the same tile and store it in the same logical layout', () => {
    const key = (t: { r: number; c: number; sharedRow: number; sharedCol: number }) => `${t.r},${t.c},${t.sharedRow},${t.sharedCol}`;
    expect(new Set(toy('ch5').map(key))).toEqual(new Set(toy('turned').map(key)));
    const turned = at(toy('turned'), 3, 1);
    expect([turned.sharedRow, turned.sharedCol]).toEqual([3, 1]); // Nds[tx][ty]: a column of the shared tile
  });
  it('with the figure\'s four-thread toy warp: (a) four sectors, (b) one', () => {
    const q = { layout: 'col' as const, T: 4, bx: 1, ph: 2, Width: 16, warpWidth: 4 };
    expect(tileLoad({ ...q, arrangement: 'ch5' }).warps[0].access.transactions).toBe(4);
    expect(tileLoad({ ...q, arrangement: 'turned' }).warps[0].access.transactions).toBe(1);
    expect(tileLoad({ ...q, arrangement: 'turned' }).warps[0].runs).toBe(1);
  });
  it('T = 32, column-major: (a) one sector per lane, (b) four sectors for every warp', () => {
    const q = { layout: 'col' as const, T: 32, bx: 1, ph: 3, Width: 1024, warpWidth: 32 };
    const a = tileLoad({ ...q, arrangement: 'ch5' });
    const b = tileLoad({ ...q, arrangement: 'turned' });
    expect(a.warps).toHaveLength(32);
    expect(a.warps.every((w) => w.access.transactions === 32 && w.access.efficiency === 0.125)).toBe(true);
    expect(b.warps.every((w) => w.access.transactions === 4 && w.access.efficiency === 1 && w.runs === 1)).toBe(true);
  });
  it('T = 16, column-major, 32-lane warp: (b) covers two tile columns, two full runs; (a) pairs of floats', () => {
    const q = { layout: 'col' as const, T: 16, bx: 0, ph: 1, Width: 1024, warpWidth: 32 };
    const b = tileLoad({ ...q, arrangement: 'turned' }).warps[0];
    expect(b.runs).toBe(2);
    expect(b.access.transactions).toBe(4);
    expect(b.access.efficiency).toBe(1);
    const a = tileLoad({ ...q, arrangement: 'ch5' }).warps[0];
    expect(a.access.transactions).toBe(16);
    expect(a.access.efficiency).toBe(0.25);
  });
  it('row-major N with chapter 5\'s mapping is already coalesced, and turning it breaks that', () => {
    const q = { layout: 'row' as const, T: 32, bx: 2, ph: 1, Width: 1024, warpWidth: 32 };
    expect(tileLoad({ ...q, arrangement: 'ch5' }).warps[0].access.transactions).toBe(4);
    expect(tileLoad({ ...q, arrangement: 'turned' }).warps[0].access.transactions).toBe(32);
  });
});

describe('Exercise 3: which square tile widths avoid uncoalesced loads in chapter 5\'s kernel', () => {
  // The N load of chapter 5's tiled kernel is the 'ch5' arrangement on row-major N.
  const warp0 = (T: number) => tileLoad({ layout: 'row', arrangement: 'ch5', T, bx: 1, ph: 1, Width: 1024, warpWidth: 32 }).warps[0];
  it('strict reading (one warp, one run of consecutive addresses): only T = 32', () => {
    expect(warp0(32).runs).toBe(1);
    expect(warp0(16).runs).toBe(2);
    expect(warp0(8).runs).toBe(4);
  });
  it('sector reading (no fetched byte wasted): T = 8, 16 and 32 all waste nothing when rows are aligned', () => {
    for (const T of [8, 16, 32]) {
      expect(warp0(T).access.efficiency).toBe(1);
      expect(warp0(T).access.transactions).toBe(4);
    }
  });
  it('a tile narrower than one sector does waste bytes', () => {
    const w = tileLoad({ layout: 'row', arrangement: 'ch5', T: 4, bx: 1, ph: 1, Width: 1024, warpWidth: 32 }).warps[0];
    expect(w.access.efficiency).toBe(0.5);
  });
});

describe('Exercise 1 (pp. 154–155)', () => {
  const ex = exercise1();
  it('classifies the nine accesses as the notes do', () => {
    expect(ex.map((x) => `${x.id}:${x.verdict}`)).toEqual([
      'a:coalesced', 'b:not applicable', 'c:coalesced', 'd:uncoalesced', 'e:not applicable',
      'f:not applicable', 'g:coalesced', 'h:not applicable', 'i:uncoalesced',
    ]);
  });
  it('sector counts for a 32-lane warp under NVIDIA\'s rule', () => {
    const by = Object.fromEntries(ex.map((x) => [x.id, x.transactions]));
    expect(by).toEqual({ a: 4, b: undefined, c: 4, d: 16, e: undefined, f: undefined, g: 4, h: undefined, i: 32 });
    expect(ex.find((x) => x.id === 'd')!.efficiency).toBe(0.25);
    expect(ex.find((x) => x.id === 'i')!.efficiency).toBe(0.125);
  });
  it('the j term is the same for every lane, whatever j is', () => {
    for (const j of [0, 1, 2, 3]) {
      const e = exercise1(32, 32, j);
      expect(e.find((x) => x.id === 'c')!.transactions).toBe(4);
      expect(e.find((x) => x.id === 'd')!.verdict).toBe('uncoalesced');
    }
  });
  it('d[i + 8] is shifted by exactly one sector: no extra sector, but two 128-byte lines instead of one', () => {
    expect(exercise1(32, 32).find((x) => x.id === 'g')!.transactions).toBe(4);
    expect(exercise1(32, 128).find((x) => x.id === 'g')!.transactions).toBe(2);
    expect(exercise1(32, 128).find((x) => x.id === 'a')!.transactions).toBe(1);
  });
  it('the shared accesses carry their word stride for section 6.4', () => {
    expect(ex.find((x) => x.id === 'h')!.sharedStride).toBe(4);
  });
});
