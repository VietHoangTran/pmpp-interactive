import { describe, expect, it } from 'vitest';
import {
  BM,
  BN,
  coalesced,
  coarseThread,
  elementOf,
  instructionsPerThread,
  ownerOf,
  ownershipCounts,
  remapGeometry,
  remapThread,
  rowStrips,
  storeInstruction,
  vectorPath,
  warpTileTotal,
  type StoreSpec,
  type Width,
} from './gemmremap';

const spec = (o: Partial<StoreSpec>): StoreSpec => ({
  pattern: 'remap4x4', W: 32, N: 1024, warp: 0, row: 0, col: 0, segmentBytes: 32, ...o,
});

describe('the two mappings own every element exactly once', () => {
  for (const layout of ['coarse', 'remap'] as const)
    for (const W of [32, 64] as Width[])
      it(`${layout}, width ${W}`, () => {
        const n = ownershipCounts(layout, W);
        expect(n.length).toBe(BM * BN);
        expect(n.every((x) => x === 1)).toBe(true);
      });

  it('ownerOf inverts elementOf in both layouts and widths', () => {
    for (const layout of ['coarse', 'remap'] as const)
      for (const W of [32, 64] as Width[])
        for (const t of [0, 1, 5, 31, 32, 63, 64, 200, 255])
          for (const [r, c] of [[0, 0], [3, 3], [4, 0], [0, 4], [7, 7], [5, 2]]) {
            const e = elementOf(layout, W, t, r, c);
            const o = ownerOf(layout, W, e.row, e.col);
            expect([o.t, o.r, o.c, o.warp, o.lane]).toEqual([t, r, c, Math.floor(t / W), t % W]);
          }
  });
});

describe('the book’s §15.3 mapping (Fig. 15.3)', () => {
  it('thread 17 owns the patch at rows 8–15, columns 8–15 (Fig. 15.2)', () => {
    expect(coarseThread(17)).toMatchObject({ ty: 1, tx: 1, tRow: 8, tCol: 8 });
  });
});

describe('the book’s §15.5 layout at width 32 (p. 360, Fig. 15.10)', () => {
  const g = remapGeometry(32);
  it('8 warps as 2 x 4 warp tiles of 64 x 32, quadrants 32 x 16, lanes 8 x 4', () => {
    expect(g).toMatchObject({ warps: 8, WR: 2, WC: 4, warpRows: 64, warpCols: 32, QR: 32, QC: 16, LC: 4, laneRows: 8 });
  });
  it('warp 6 starts at (64, 64)', () => {
    const t = remapThread(6 * 32, 32);
    expect([t.warp, t.lane, t.wRow, t.wCol]).toEqual([6, 0, 64, 64]);
  });
  it('lane 0 of warp 6 owns the same 4 x 4 square in each of the four quadrants', () => {
    expect(remapThread(192, 32).subTiles.map((s) => [s.row, s.col])).toEqual([[64, 64], [64, 80], [96, 64], [96, 80]]);
  });
  it('lane 5 sits in lane row 1, lane column 1', () => {
    expect(remapThread(5, 32)).toMatchObject({ lRow: 4, lCol: 4 });
  });
  it('each thread still owns 64 elements: 4 sub-tiles of 16', () => {
    expect(remapThread(77, 32).subTiles.length * 16).toBe(64);
  });
});

describe('our 64-wide layout (ch15-facts B1)', () => {
  const g = remapGeometry(64);
  it('4 wavefronts as 2 x 2 tiles of 64 x 64, quadrants 32 x 32, lanes 8 x 8', () => {
    expect(g).toMatchObject({ warps: 4, WR: 2, WC: 2, warpRows: 64, warpCols: 64, QR: 32, QC: 32, LC: 8, laneRows: 8 });
  });
  it('a lane row writes 128 contiguous bytes', () => {
    expect(g.laneRowBytes).toBe(128);
    const r = storeInstruction(spec({ W: 64 }));
    const strips = rowStrips(r, 32);
    expect(strips.length).toBe(8);
    for (const s of strips) {
      expect(s.pieces.length).toBe(8);
      expect(s.pieces[s.pieces.length - 1].hi - s.pieces[0].lo).toBe(128);
    }
    expect(r.runs).toBe(8);
  });
});

describe('one store instruction of one warp, N a multiple of 8, 32-byte sectors', () => {
  it('scalar stores from 8 x 8 tiles: 32 sectors at 12.5 %', () => {
    const r = storeInstruction(spec({ pattern: 'scalar8x8' }));
    expect([r.transactions, r.efficiency, r.bytesNeeded, r.rows.length]).toEqual([32, 0.125, 128, 2]);
    // Neighbouring lanes 8 floats (32 bytes) apart.
    expect(r.lanes[1].byte - r.lanes[0].byte).toBe(32);
  });
  it('float4 stores from 8 x 8 tiles (Exercise 2): still 32 sectors, at 50 %', () => {
    const r = storeInstruction(spec({ pattern: 'float4-8x8' }));
    expect([r.transactions, r.efficiency, r.vector]).toEqual([32, 0.5, true]);
    expect(r.lanes[1].byte - r.lanes[0].byte).toBe(32);
  });
  it('float4 stores from four 4 x 4 sub-tiles (§15.5): 16 sectors at 100 %', () => {
    for (const row of [0, 1, 5, 7])
      for (const col of [0, 4])
        for (const warp of [0, 6]) {
          const r = storeInstruction(spec({ row, col, warp }));
          expect([r.transactions, r.efficiency, r.rows.length, r.runs]).toEqual([16, 1, 8, 8]);
        }
  });
  it('the three patterns are classified as the lesson says', () => {
    expect([coalesced('scalar8x8'), coalesced('float4-8x8'), coalesced('remap4x4')]).toEqual([false, false, true]);
  });
  it('at width 64 the scalar pattern is 64 sectors and the remap 32, still 12.5 % and 100 %', () => {
    const a = storeInstruction(spec({ pattern: 'scalar8x8', W: 64 }));
    const b = storeInstruction(spec({ W: 64 }));
    expect([a.transactions, a.efficiency, b.transactions, b.efficiency]).toEqual([64, 0.125, 32, 1]);
  });
});

describe('alignment: the vector path and N', () => {
  it('float4 only when N is a multiple of 4; per thread 16 vector or 64 scalar stores', () => {
    expect([vectorPath('remap4x4', 1024), vectorPath('remap4x4', 1020), vectorPath('remap4x4', 1001), vectorPath('scalar8x8', 1024)]).toEqual([true, true, false, false]);
    expect([instructionsPerThread('remap4x4', 1024), instructionsPerThread('remap4x4', 1001)]).toEqual([16, 64]);
  });
  it('N = 1020 (a multiple of 4, not of 8): rows alternate between 16 and 24 sectors', () => {
    const even = storeInstruction(spec({ N: 1020, row: 0 }));
    const odd = storeInstruction(spec({ N: 1020, row: 1 }));
    expect([even.transactions, even.efficiency]).toEqual([16, 1]);
    expect(odd.transactions).toBe(24);
    expect(odd.efficiency).toBeCloseTo(2 / 3, 10);
  });
  it('N = 1001: float4 is ruled out, the kernel stores scalars', () => {
    const r = storeInstruction(spec({ N: 1001, col: 1 }));
    expect([r.fallback, r.vector, r.bytesPerLane]).toEqual([true, false, 4]);
    // Row r starts at byte 4·1001·r: 16-byte aligned only when r is a multiple of 4.
    expect([0, 1, 2, 3, 4, 8].map((row) => (row * 1001 * 4) % 16 === 0)).toEqual([true, false, false, false, true, true]);
  });
});

describe('segment sizes other than NVIDIA’s (AMD 64- and 128-byte lines, a model)', () => {
  it('128-byte lines: the 32-wide remap fills half of each line, the 64-wide one all of it', () => {
    expect(storeInstruction(spec({ segmentBytes: 128 })).efficiency).toBe(0.5);
    expect(storeInstruction(spec({ segmentBytes: 128, W: 64 })).efficiency).toBe(1);
  });
  it('64-byte lines: the 32-wide remap is 8 full lines', () => {
    const r = storeInstruction(spec({ segmentBytes: 64 }));
    expect([r.transactions, r.efficiency]).toEqual([8, 1]);
  });
});

describe('a whole warp tile, summed over instructions without cache merging', () => {
  it('scalar 2048 sectors, float4 8 x 8 512, remap 256 (the minimum)', () => {
    expect(warpTileTotal('scalar8x8', 32, 1024, 0, 32)).toEqual({ instructions: 64, transactions: 2048, bytes: 8192, minimum: 256 });
    expect(warpTileTotal('float4-8x8', 32, 1024, 0, 32).transactions).toBe(512);
    expect(warpTileTotal('remap4x4', 32, 1024, 0, 32)).toEqual({ instructions: 16, transactions: 256, bytes: 8192, minimum: 256 });
  });
});
