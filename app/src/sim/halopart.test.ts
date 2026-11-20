import { describe, expect, it } from 'vitest';
import {
  EXERCISE1,
  FLOAT_BYTES,
  bestTiles,
  coverage,
  factorisations,
  firstRows,
  launches,
  neighbours,
  partition,
  pointerExpr,
  rowKind,
  rowPointers,
  rowShares,
  threadRowToDataRow,
  tileOf,
  tileSplit,
  visibleRows,
  writtenRows,
} from './halopart';

describe('exercise 1 (p. 576): 64 x 512 on 16 GPUs, both readings of "computes"', () => {
  const { nx, rows, gpus } = EXERCISE1;
  for (let g = 0; g < gpus; g++) {
    it(`GPU ${g}: 32 rows, ny = 34`, () => {
      const p = partition(nx, rows, gpus, g);
      expect(p.owned).toBe(32);
      expect(p.ny).toBe(34);
      expect(p.first).toBe(32 * g);
      expect(p.last).toBe(32 * g + 31);
    });
  }
  const p = partition(nx, rows, gpus, 5);
  it('1a: 2048 output points over all columns, 1984 over the 62 the kernel computes', () => {
    expect(p.output).toEqual({ all: 2048, kernel: 1984 });
  });
  it('1b: 128 halo points (two whole rows of 64)', () => {
    expect(p.halo).toBe(128);
  });
  it('1c: 128 boundary points, or 124 without the edge columns', () => {
    expect(p.boundaryRows).toBe(2);
    expect(p.boundary).toEqual({ all: 128, kernel: 124 });
  });
  it('1d: 1920 internal points, or 1860', () => {
    expect(p.internalRows).toBe(30);
    expect(p.internal).toEqual({ all: 1920, kernel: 1860 });
  });
  it('1e and q-ex1-bytes: 512 bytes sent per step, 256 to each neighbour', () => {
    expect(FLOAT_BYTES).toBe(4);
    expect(p.bytesSent).toBe(512);
    expect(p.bytesPerNeighbour).toBe(256);
  });
  it('boundary plus internal is the output, both ways', () => {
    expect(p.boundary.all + p.internal.all).toBe(p.output.all);
    expect(p.boundary.kernel + p.internal.kernel).toBe(p.output.kernel);
  });
});

describe('the start page prediction: the exchange does not shrink with more GPUs', () => {
  it('on 8, 16 and 32 GPUs each GPU sends 512 bytes while its rows go 64, 32, 16', () => {
    const at = (g: number) => partition(64, 512, g, 0);
    expect([8, 16, 32].map((g) => at(g).bytesSent)).toEqual([512, 512, 512]);
    expect([8, 16, 32].map((g) => at(g).owned)).toEqual([64, 32, 16]);
    expect(at(16).exchangeShare).toBeCloseTo(2 * at(8).exchangeShare, 12);
    expect(at(32).exchangeShare).toBe(0.125);
  });
});

describe('q-halo-rows (p. 544): 128 rows on 4 GPUs, GPU 1', () => {
  it('owns rows 32 to 63 and holds 34 rows (31 to 64)', () => {
    const p = partition(16, 128, 4, 1);
    expect([p.first, p.last]).toEqual([32, 63]);
    expect(p.ny).toBe(34);
    expect([p.top, p.bottom]).toEqual([0, 2]);
  });
});

describe('row shares and neighbours', () => {
  it('an even split', () => {
    expect(rowShares(512, 16)).toEqual(new Array(16).fill(32));
  });
  it('an uneven split puts the remainder on the last GPU and still sums to the grid', () => {
    const s = rowShares(520, 16);
    expect(s.slice(0, 15)).toEqual(new Array(15).fill(32));
    expect(s[15]).toBe(40);
    expect(s.reduce((a, b) => a + b, 0)).toBe(520);
    expect(firstRows(s)[15]).toBe(480);
    expect(partition(64, 520, 16, 15).ny).toBe(42);
  });
  it('every GPU owns at least one row; fewer rows than GPUs is refused', () => {
    expect(rowShares(32, 32)).toEqual(new Array(32).fill(1));
    expect(() => rowShares(31, 32)).toThrow(RangeError);
    expect(() => rowShares(32, 0)).toThrow(RangeError);
  });
  it('neighbours wrap around as in Fig. 23.7 lines 11–12', () => {
    expect(neighbours(0, 4)).toEqual({ top: 3, bottom: 1 });
    expect(neighbours(3, 4)).toEqual({ top: 2, bottom: 0 });
    expect(neighbours(0, 1)).toEqual({ top: 0, bottom: 0 });
    expect(partition(64, 64, 1, 0).self).toBe(true);
    expect(partition(64, 64, 2, 0).self).toBe(false);
  });
  it('one owned row is both boundary rows; two owned rows leave no internal row', () => {
    const one = partition(10, 4, 4, 2);
    expect([one.ny, one.boundaryRows, one.internalRows]).toEqual([3, 1, 0]);
    expect(one.boundary).toEqual({ all: 10, kernel: 8 });
    const two = partition(10, 8, 4, 2);
    expect([two.ny, two.boundaryRows, two.internalRows]).toEqual([4, 2, 0]);
  });
});

describe('rows of a local array and the pointers Fig. 23.7 uses', () => {
  it('classes: halo at 0 and ny − 1, boundary at 1 and ny − 2', () => {
    expect([0, 1, 2, 31, 32, 33].map((r) => rowKind(34, r))).toEqual(['halo', 'boundary', 'internal', 'internal', 'boundary', 'halo']);
    expect(() => rowKind(34, 34)).toThrow(RangeError);
  });
  it('the exchange pointers (p. 552): send rows 1 and ny − 2, receive into rows ny − 1 and 0', () => {
    expect(pointerExpr(34, 1)).toBe('output + nx');
    expect(pointerExpr(34, 32)).toBe('output + (ny - 2)*nx');
    expect(pointerExpr(34, 33)).toBe('output + (ny - 1)*nx');
    expect(pointerExpr(34, 0)).toBe('output');
    expect(pointerExpr(34, 31)).toBe('output + (ny - 3)*nx');
    expect(pointerExpr(34, 5)).toBe('output + 5*nx');
  });
  it('rowPointers gives offsets r·nx and covers every row', () => {
    const rp = rowPointers(34, 64);
    expect(rp).toHaveLength(34);
    expect(rp[33]).toEqual({ row: 33, kind: 'halo', expr: 'output + (ny - 1)*nx', offset: 33 * 64 });
  });
  it('visibleRows keeps three rows at each end and one gap', () => {
    expect(visibleRows(34)).toEqual([
      { kind: 'row', row: 0 },
      { kind: 'row', row: 1 },
      { kind: 'row', row: 2 },
      { kind: 'gap', from: 3, to: 30 },
      { kind: 'row', row: 31 },
      { kind: 'row', row: 32 },
      { kind: 'row', row: 33 },
    ]);
    expect(visibleRows(7).map((i) => (i.kind === 'row' ? i.row : -1))).toEqual([0, 1, 2, 3, 4, 5, 6]);
    expect(visibleRows(8).filter((i) => i.kind === 'gap')).toEqual([{ kind: 'gap', from: 3, to: 4 }]);
  });
});

describe('the three launches of Fig. 23.14 (pp. 557–558)', () => {
  it('pointers and row counts as printed', () => {
    const [t, b, i] = launches(34, 64);
    expect([t.name, t.stream, t.pointer, t.offset, t.rowsExpr, t.rows]).toEqual(['top', 'topStream', 'output', 0, '3', 3]);
    expect([b.name, b.stream, b.pointer, b.offset, b.rowsExpr, b.rows]).toEqual(['bottom', 'bottomStream', 'output + (ny - 3)*nx', 31 * 64, '3', 3]);
    expect([i.name, i.stream, i.pointer, i.offset, i.rowsExpr, i.rows]).toEqual(['internal', 'internalStream', 'output + nx', 64, 'ny - 2', 32]);
  });
  it('each reads its span and writes what the guard admits', () => {
    const [t, b, i] = launches(34, 64);
    expect(t.reads).toEqual([0, 1, 2]);
    expect(t.writes).toEqual([1]);
    expect(b.reads).toEqual([31, 32, 33]);
    expect(b.writes).toEqual([32]);
    expect(i.reads[0]).toBe(1);
    expect(i.reads[i.reads.length - 1]).toBe(32);
    expect(i.writes[0]).toBe(2);
    expect(i.writes[i.writes.length - 1]).toBe(31);
  });
  it("q-threerows: with ny = 10 the bottom launch's one thread row writes row 8", () => {
    const b = launches(10, 16)[1];
    expect(b.base).toBe(7);
    expect(b.writes).toEqual([8]);
    expect(threadRowToDataRow(b.base, b.rows, 0)).toBe(8);
    expect(threadRowToDataRow(b.base, b.rows, 1)).toBeNull();
  });
  it('thread rows map onto data rows: internal thread rows 0 to ny − 5 write rows 2 to ny − 3', () => {
    const ny = 34;
    expect(threadRowToDataRow(1, ny - 2, 0)).toBe(2);
    expect(threadRowToDataRow(1, ny - 2, ny - 5)).toBe(ny - 3);
    expect(threadRowToDataRow(1, ny - 2, ny - 4)).toBeNull();
    expect(threadRowToDataRow(0, 3, -1)).toBeNull();
  });
  it('writtenRows applies the guard y < rows − 1 with the +1 offset', () => {
    expect(writtenRows(0, 3)).toEqual([1]);
    expect(writtenRows(1, 2)).toEqual([]);
    expect(writtenRows(5, 6)).toEqual([6, 7, 8, 9]);
  });
  for (let ny = 3; ny <= 40; ny++) {
    it(`ny = ${ny}: ${ny === 3 ? 'row 1 is written twice' : 'rows 1 to ny − 2 written exactly once'}`, () => {
      const c = coverage(ny);
      expect(c.counts[0]).toBe(0);
      expect(c.counts[ny - 1]).toBe(0);
      expect(c.missed).toEqual([]);
      if (ny === 3) {
        expect(c.exact).toBe(false);
        expect(c.twice).toEqual([1]);
      } else {
        expect(c.exact).toBe(true);
        expect(c.counts.slice(1, ny - 1).every((n) => n === 1)).toBe(true);
      }
    });
  }
});

describe('the 2D alternative (ours, after p. 545)', () => {
  it('factorisations of 16, tile rows first', () => {
    expect(factorisations(16)).toEqual([
      { p: 16, q: 1 },
      { p: 8, q: 2 },
      { p: 4, q: 4 },
      { p: 2, q: 8 },
      { p: 1, q: 16 },
    ]);
    expect(factorisations(7)).toEqual([
      { p: 7, q: 1 },
      { p: 1, q: 7 },
    ]);
  });
  it('G × 1 is the row split: 2 neighbours, 2·nx values', () => {
    const t = tileSplit(64, 512, 16, 1);
    expect([t.neighbours, t.points, t.bytes, t.rowMessages, t.columnMessages]).toEqual([2, 128, 512, 2, 0]);
    expect(t.bytes).toBe(partition(64, 512, 16, 0).bytesSent);
  });
  it("the exercise's tall grid: rows send least", () => {
    const b = bestTiles(64, 512, 16);
    expect([b.p, b.q, b.points]).toEqual([16, 1, 128]);
    expect(tileSplit(64, 512, 4, 4).points).toBe(2 * 16 + 2 * 128);
  });
  it('a square grid: 4 × 4 tiles send half what rows send, with four neighbours', () => {
    const b = bestTiles(512, 512, 16);
    expect([b.p, b.q, b.h, b.w, b.neighbours, b.points]).toEqual([4, 4, 128, 128, 4, 512]);
    expect(tileSplit(512, 512, 16, 1).points).toBe(1024);
    expect([b.rowMessages, b.columnMessages]).toEqual([2, 2]);
  });
  it('a very wide grid: columns win, every message strided', () => {
    const b = bestTiles(1024, 64, 16);
    expect([b.p, b.q, b.points, b.rowMessages, b.columnMessages]).toEqual([1, 16, 128, 0, 2]);
    expect(tileSplit(1024, 64, 16, 1).points).toBe(2048);
  });
  it('uneven tiles count the largest one', () => {
    const t = tileSplit(100, 50, 3, 3);
    expect([t.h, t.w]).toEqual([18, 34]);
  });
  it('one GPU has nothing to exchange with another', () => {
    expect(tileSplit(64, 64, 1, 1)).toMatchObject({ neighbours: 0, points: 0 });
  });
  it('arrangements that do not fit are skipped', () => {
    expect(bestTiles(8, 512, 16).q).toBeLessThanOrEqual(8);
    expect(() => tileSplit(8, 512, 1, 16)).toThrow(RangeError);
  });
  it('GPU g sits at tile row g / Q, column g mod Q', () => {
    expect(tileOf(6, 4)).toEqual({ row: 1, col: 2 });
    expect(tileOf(15, 1)).toEqual({ row: 15, col: 0 });
  });
});
