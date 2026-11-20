import { describe, expect, it } from 'vitest';
import {
  BOOK,
  blockOrigin,
  gridDims,
  lastTileZeroColumns,
  launch2D,
  loadPlan,
  loaderMap,
  loadsPerThreadPerTilePair,
  ownerOf,
  owns,
  slotOf,
  threadTile,
  tileLoadSummary,
  tilePairs,
  warpLoad,
} from './gemmmap';

// Numbers section 15.3 states, pinned to the model they come from.
describe('who owns what (Fig. 15.3 lines 05–13)', () => {
  it('thread 17 owns rows and columns 8 to 15 of the block tile', () => {
    const t = threadTile(17);
    expect([t.tilesPerBlockX, t.ty, t.tx, t.tRow, t.tCol]).toEqual([16, 1, 1, 8, 8]);
  });
  it('block (x 2, y 1) starts at row 128, column 256', () => {
    expect(blockOrigin(2, 1)).toEqual({ bRow: 128, bCol: 256 });
  });
  it('thread 17 of block (2, 1): rows 136 to 143, columns 264 to 271 (the Predict)', () => {
    const o = owns(1024, 1024, 2, 1, 17);
    expect(o.rows).toEqual([136, 143]);
    expect(o.cols).toEqual([264, 271]);
    expect(o.aRows).toEqual([8, 15]);
    expect(o.bCols).toEqual([8, 15]);
    expect(o.writes).toBe(64);
  });
  it('thread 255 owns the bottom-right patch: rows 248 to 255, columns 376 to 383 in block (2, 1)', () => {
    const o = owns(1024, 1024, 2, 1, 255);
    expect([o.ty, o.tx]).toEqual([15, 15]);
    expect(o.rows).toEqual([248, 255]);
    expect(o.cols).toEqual([376, 383]);
  });
  it('a patch that starts outside C writes nothing; one that straddles the edge writes part', () => {
    expect(owns(130, 256, 0, 1, 17).writes).toBe(0); // row 136 >= 130
    const edge = owns(130, 256, 0, 1, 0);
    expect([edge.maxRow, edge.writes]).toEqual([2, 16]);
  });
  it('ownerOf inverts owns for every element of a block tile', () => {
    for (let t = 0; t < 256; t += 37) {
      const o = owns(512, 512, 1, 3, t);
      for (let r = o.rows[0]; r <= o.rows[1]; r += 3)
        for (let c = o.cols[0]; c <= o.cols[1]; c += 5) {
          const w = ownerOf(r, c);
          expect([w.bx, w.by, w.t]).toEqual([1, 3, t]);
        }
    }
  });
  it('every element of the block tile has exactly one owner', () => {
    const seen = new Array(128 * 128).fill(0);
    for (let t = 0; t < 256; t++) {
      const tt = threadTile(t);
      for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) seen[(tt.tRow + r) * 128 + tt.tCol + c]++;
    }
    expect(seen.every((n) => n === 1)).toBe(true);
  });
});

describe('the grid', () => {
  it('the 1024 -> 4096 layer at batch 1: 32 blocks of 128 x 128 against 256 of 16 x 16', () => {
    expect(gridDims(1, 4096).blocks).toBe(32);
    expect(gridDims(1, 4096, { ...BOOK, bM: 16, bN: 16 }).blocks).toBe(256);
  });
  it('K = 1001: 126 tile pairs, the last with 7 zero-filled columns of A', () => {
    expect(tilePairs(1001)).toBe(126);
    expect(lastTileZeroColumns(1001)).toBe(7);
    expect(lastTileZeroColumns(1000)).toBe(0);
  });
});

describe('loadTile (Fig. 15.5)', () => {
  it('A tile 128 x 8: 32 rows per sub-tile, 4 sub-tiles; B tile 8 x 128: 2 rows, 4 sub-tiles', () => {
    const a = loadPlan(128, 8, 256);
    const b = loadPlan(8, 128, 256);
    expect([a.rowsPerSubTile, a.numSubtiles, a.problems]).toEqual([32, 4, []]);
    expect([b.rowsPerSubTile, b.numSubtiles, b.problems]).toEqual([2, 4, []]);
  });
  it('thread 77 in sub-tile 2 of the A tile loads row 73, column 5', () => {
    expect(slotOf(loadPlan(128, 8, 256), 77, 2)).toEqual({ row: 73, col: 5 });
  });
  it('every tile element is loaded exactly once, scalar and float4, A and B', () => {
    for (const mode of ['scalar', 'float4'] as const)
      for (const [h, w] of [[128, 8], [8, 128]]) {
        const m = loaderMap(loadPlan(h, w, 256, mode));
        expect(m.times.every((n) => n === 1), `${mode} ${h}x${w}`).toBe(true);
      }
  });
  it('float4 (Exercise 1): one sub-tile each; 2 load instructions per thread per tile pair instead of 8', () => {
    expect(loadPlan(128, 8, 256, 'float4').numSubtiles).toBe(1);
    expect(loadPlan(8, 128, 256, 'float4').numSubtiles).toBe(1);
    expect(loadsPerThreadPerTilePair(BOOK, 'float4')).toBe(2);
    expect(loadsPerThreadPerTilePair(BOOK, 'scalar')).toBe(8);
  });
  it('float4 with a short tile: threads beyond the last row do nothing', () => {
    const p = loadPlan(4, 128, 256, 'float4'); // 32 accesses a row, 8 rows a sub-tile, tile only 4 tall
    expect(p.numSubtiles).toBe(1);
    expect(slotOf(p, 200, 0)).toBeNull();
    expect(loaderMap(p).times.every((n) => n === 1)).toBe(true);
  });
  it('a shape the book\'s arithmetic does not cover is reported', () => {
    expect(loadPlan(128, 12, 256).problems.length).toBeGreaterThan(0);
  });
});

describe('one warp\'s load instruction (32-byte sectors)', () => {
  const p = { M: 1024, N: 1024, K: 1024 };
  it('A tile, K a multiple of 8: 4 sectors, in 4 runs one row of A apart', () => {
    const w = warpLoad('A', p, 0, 0, 3, 1, 2);
    expect(w.sectors.length).toBe(4);
    expect(w.runs.length).toBe(4);
    expect(w.runs[1][0] - w.runs[0][0]).toBe(1024 * 4);
    expect(w.lanes.map((l) => l.row)).toEqual(Array.from({ length: 32 }, (_, l) => 32 + 8 + Math.floor(l / 8)));
  });
  it('B tile, N a multiple of 8: 4 sectors in one 128-byte run', () => {
    const w = warpLoad('B', p, 1, 0, 3, 1, 5);
    expect(w.sectors.length).toBe(4);
    expect(w.runs).toHaveLength(1);
    expect(w.runs[0][1] - w.runs[0][0]).toBe(128);
  });
  it('every warp of both tiles: 4 sectors when K and N are multiples of 8', () => {
    for (const which of ['A', 'B'] as const)
      for (let s = 0; s < 4; s++) for (let w = 0; w < 8; w++) expect(warpLoad(which, { M: 512, N: 136, K: 72 }, 0, 0, 2, s, w).sectors.length).toBe(4);
  });
  it('an unaligned K costs A extra sectors: K = 1001 gives 8 for a warp whose rows all straddle', () => {
    const counts = Array.from({ length: 8 }, (_, w) => warpLoad('A', { M: 1024, N: 1024, K: 1001 }, 0, 0, 0, 0, w).sectors.length);
    expect(Math.max(...counts)).toBe(8);
    expect(Math.min(...counts)).toBeGreaterThanOrEqual(5);
  });
  it('float4: each warp moves 512 bytes in 16 sectors, for A (16 rows) and for B (one row)', () => {
    const a = warpLoad('A', p, 0, 0, 0, 0, 3, BOOK, 'float4');
    const b = warpLoad('B', p, 0, 0, 0, 0, 3, BOOK, 'float4');
    expect([a.sectors.length, a.runs.length]).toEqual([16, 16]);
    expect([b.sectors.length, b.runs.length]).toEqual([16, 1]);
    expect(a.lanes.every((l) => !l.misaligned)).toBe(true);
  });
  it('float4 with K not a multiple of 4: the addresses are not 16-byte aligned', () => {
    const a = warpLoad('A', { M: 256, N: 256, K: 1001 }, 0, 0, 0, 0, 0, BOOK, 'float4');
    expect(a.lanes.some((l) => l.misaligned)).toBe(true);
  });
});

describe('the bounds check: zero fill and divergence', () => {
  it('no warp diverges anywhere when M, N, K are multiples of the tiles', () => {
    const p = { M: 256, N: 256, K: 64 };
    for (const which of ['A', 'B'] as const)
      for (let by = 0; by < 2; by++) for (let tile = 0; tile < 8; tile++) expect(tileLoadSummary(which, p, 1, by, tile).divergent).toBe(0);
  });
  it('edge block (M = 130, K = 9), block y 1, last tile: one A warp diverges; the rest zero-fill uniformly', () => {
    const p = { M: 130, N: 256, K: 9 };
    const a = tileLoadSummary('A', p, 0, 1, 1);
    expect(a.instructions).toBe(32);
    expect(a.divergent).toBe(1);
    expect(a.zeroFilled).toBe(1024 - 2); // rows 128 and 129, column 8 only
    const w = warpLoad('A', p, 0, 1, 1, 0, 0);
    expect(w.lanes.filter((l) => l.branch === 'load').map((l) => [l.gRow, l.gCol])).toEqual([[128, 8], [129, 8]]);
  });
  it('the K edge alone makes every A warp of the last tile diverge, and no B warp', () => {
    const p = { M: 130, N: 256, K: 9 };
    expect(tileLoadSummary('A', p, 0, 0, 1).divergent).toBe(32);
    expect(tileLoadSummary('B', p, 0, 0, 1).divergent).toBe(0);
  });
  it('an N edge splits B warps: N = 200, block x 1 has 72 columns, warps 2 and 6 of each sub-tile diverge', () => {
    const p = { M: 256, N: 200, K: 64 };
    expect(tileLoadSummary('B', p, 1, 0, 0).divergent).toBe(8);
    expect(warpLoad('B', p, 1, 0, 0, 0, 2).diverges).toBe(true);
  });
});

describe('launching dim3(16, 16) instead of 256 threads', () => {
  const r = launch2D(16, 16);
  it('only threadIdx.x 0 to 15 exist as far as the kernel can tell', () => {
    expect([r.threads, r.distinctThreads]).toEqual([256, 16]);
  });
  it('16 patches, the top row, each written by 16 threads; rows 8 to 127 never written', () => {
    expect(r.patchesWritten).toBe(16);
    expect(r.writers.slice(0, 16).every((n) => n === 16)).toBe(true);
    expect(r.rowsNeverWritten).toBe(120);
  });
  it('loadTile fills 64 of 1024 slots of each tile, and no written patch has all its inputs', () => {
    expect([r.aLoadedCount, r.bLoadedCount]).toEqual([64, 64]);
    expect(r.correctPatches).toBe(0);
  });
  it('the intended one-dimensional launch covers everything once', () => {
    const ok = launch2D(256, 1);
    expect([ok.patchesWritten, ok.rowsNeverWritten, ok.aLoadedCount, ok.bLoadedCount, ok.correctPatches]).toEqual([256, 0, 1024, 1024, 256]);
  });
});
