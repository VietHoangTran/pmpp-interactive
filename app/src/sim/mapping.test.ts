import { describe, expect, it } from 'vitest';
import {
  BOOK,
  blockInfo,
  bookLabel,
  elementAt,
  flat,
  flat3,
  flatColMajor,
  gray,
  grayDetail,
  indexingCoverage,
  latticeOf,
  layout,
  offsetOf,
  pixelAt,
  regions,
  stepsGrayscale,
  strides,
  stridesFor,
  threadAt,
  unflat,
  unflat3,
  unflatColMajor,
  type Launch,
} from './mapping';

// Every number a lesson states about section 3.2 is pinned here: the book's worked
// numbers (notes/ch03.md, with pages) and the checked vendor examples
// (notes/today/ch03-facts.md).

describe('the 62 x 76 picture under 16 x 16 blocks (book pp. 49-50)', () => {
  const L = layout(BOOK);
  it('needs 5 blocks in x and 4 in y: 20 blocks', () => {
    expect([L.gridX, L.gridY, L.blocks]).toEqual([5, 4, 20]);
  });
  it('launches 80 x 64 = 5120 threads for 4712 pixels, 408 of them surplus', () => {
    expect([L.threadsX, L.threadsY, L.threads]).toEqual([80, 64, 5120]);
    expect(L.pixels).toBe(4712);
    expect(L.working).toBe(4712);
    expect(L.surplus).toBe(408);
    expect(L.unreached).toBe(0);
  });
  it('the surplus is 4 thread columns on the right and 2 thread rows at the bottom', () => {
    expect(L.threadsX - BOOK.width).toBe(4);
    expect(L.threadsY - BOOK.height).toBe(2);
    expect(4 * 64 + 2 * 80 - 4 * 2).toBe(408);
  });
  it('Thread (0,0) of Block (1,0) handles pixel (16, 0), Pout[1216], Pin[3648] (pp. 49-50, 53-54)', () => {
    // Book labels are (y, x): Block (1,0) is blockIdx.y = 1, blockIdx.x = 0.
    const p = latticeOf(BOOK, 0, 1, 0, 0);
    const t = threadAt(BOOK, p.x, p.y);
    expect(bookLabel(t.blockIdx)).toBe('(1,0)');
    expect(bookLabel(t.threadIdx)).toBe('(0,0)');
    expect([t.row, t.col]).toEqual([16, 0]);
    expect(t.works).toBe(true);
    expect(t.grayOffset).toBe(1216);
    expect(t.rgbOffset).toBe(3648);
  });
  it('a thread past the right edge fails the column clause, one below the bottom the row clause', () => {
    const right = threadAt(BOOK, 76, 10);
    expect([right.works, right.failing, right.grayOffset]).toEqual([false, 'col', null]);
    const bottom = threadAt(BOOK, 10, 62);
    expect([bottom.works, bottom.failing]).toEqual([false, 'row']);
    const corner = threadAt(BOOK, 79, 63);
    expect([corner.colOk, corner.rowOk, corner.failing]).toEqual([false, false, 'col']);
    expect(bookLabel(corner.blockIdx)).toBe('(3,4)');
    expect(bookLabel(corner.threadIdx)).toBe('(15,15)');
  });
  it('every pixel is owned by exactly one working thread', () => {
    const seen = new Uint8Array(L.pixels);
    let surplus = 0;
    for (let y = 0; y < L.threadsY; y++)
      for (let x = 0; x < L.threadsX; x++) {
        const t = threadAt(BOOK, x, y);
        if (t.works) seen[t.grayOffset!]++;
        else surplus++;
      }
    expect(seen.every((v) => v === 1)).toBe(true);
    expect(surplus).toBe(408);
  });
  it('has four kinds of block: 12 x 256, 3 x 192, 4 x 224, 1 x 168 = 4712 (pp. 54-55)', () => {
    const r = regions(BOOK);
    expect(r.map((x) => [x.region, x.blocks, x.working])).toEqual([
      ['1', 12, 256], ['2', 3, 192], ['3', 4, 224], ['4', 1, 168],
    ]);
    expect(r.reduce((s, x) => s + x.total, 0)).toBe(4712);
  });
  it('places the kinds where Fig. 3.5 draws them', () => {
    expect(blockInfo(BOOK, 0, 0).region).toBe('1');
    expect(blockInfo(BOOK, 3, 2).region).toBe('1');
    expect(blockInfo(BOOK, 4, 0).region).toBe('2');
    expect(blockInfo(BOOK, 4, 2)).toMatchObject({ workW: 12, workH: 16, working: 192 });
    expect(blockInfo(BOOK, 0, 3)).toMatchObject({ region: '3', workW: 16, workH: 14, working: 224 });
    expect(blockInfo(BOOK, 4, 3)).toMatchObject({ region: '4', workW: 12, workH: 14, working: 168 });
  });
  it('per-block working counts agree with counting threads one by one', () => {
    for (let by = 0; by < L.gridY; by++)
      for (let bx = 0; bx < L.gridX; bx++) {
        let n = 0;
        for (let ty = 0; ty < 16; ty++)
          for (let tx = 0; tx < 16; tx++) {
            const p = latticeOf(BOOK, bx, by, tx, ty);
            if (threadAt(BOOK, p.x, p.y).works) n++;
          }
        expect(blockInfo(BOOK, bx, by).working).toBe(n);
      }
  });
  it('contains no block in which nothing works', () => {
    expect(regions(BOOK).some((r) => r.region === 'idle')).toBe(false);
  });
});

describe('other launches the lesson states', () => {
  it('1500 x 2000 needs 94 blocks in y and 125 in x: 11,750 (p. 50)', () => {
    const L = layout({ height: 1500, width: 2000, blockW: 16, blockH: 16, style: 'blocks' });
    expect([L.gridY, L.gridX, L.blocks]).toEqual([94, 125, 11750]);
    expect(regions({ height: 1500, width: 2000, blockW: 16, blockH: 16, style: 'blocks' }).reduce((s, r) => s + r.total, 0)).toBe(3_000_000);
  });
  it('exercise 3 (p. 65): 150 x 300 under 16 x 32 blocks', () => {
    const ex3: Launch = { height: 150, width: 300, blockW: 16, blockH: 32, style: 'blocks' };
    const L = layout(ex3);
    expect(ex3.blockW * ex3.blockH).toBe(512); // 3a
    expect([L.gridX, L.gridY]).toEqual([19, 5]);
    expect(L.threads).toBe(48640); // 3b
    expect(L.blocks).toBe(95); // 3c
    expect(L.working).toBe(45000); // 3d
    expect(regions(ex3).reduce((s, r) => s + r.total, 0)).toBe(45000);
  });
  it('swapped extents leave 744 pixels of the 62 x 76 picture unwritten', () => {
    const sw: Launch = { ...BOOK, swapped: true };
    const L = layout(sw);
    expect([L.gridX, L.gridY]).toEqual([4, 5]);
    expect([L.threadsX, L.threadsY, L.threads]).toEqual([64, 80, 5120]);
    expect(L.coveredCols).toBe(64);
    expect(L.unreached).toBe(744); // columns 64 to 75 of all 62 rows
    expect(12 * 62).toBe(744);
    // and it launches a whole row of blocks in which no thread works
    const idle = regions(sw).find((r) => r.region === 'idle');
    expect(idle?.blocks).toBe(4);
    expect(regions(sw).reduce((s, r) => s + r.total, 0)).toBe(62 * 64);
  });
  it('swapped extents go unnoticed on a square picture', () => {
    const sq: Launch = { height: 48, width: 48, blockW: 16, blockH: 16, style: 'blocks', swapped: true };
    expect(layout(sq).unreached).toBe(0);
    expect(layout(sq)).toEqual(layout({ ...sq, swapped: false }));
  });
  it('a picture that is a whole number of blocks has one kind of block and no surplus', () => {
    const l: Launch = { height: 64, width: 80, blockW: 16, blockH: 16, style: 'blocks' };
    expect(layout(l).surplus).toBe(0);
    expect(regions(l)).toEqual([{ region: '1', blocks: 20, working: 256, launched: 256, total: 5120 }]);
  });
  it('a small picture: 10 x 13 under 4 x 4 blocks', () => {
    const l: Launch = { height: 10, width: 13, blockW: 4, blockH: 4, style: 'blocks' };
    const L = layout(l);
    expect([L.gridX, L.gridY, L.threads, L.pixels, L.surplus]).toEqual([4, 3, 192, 130, 62]);
    expect(regions(l).map((r) => [r.blocks, r.working])).toEqual([[6, 16], [2, 4], [3, 8], [1, 2]]);
  });
});

describe('Metal launch styles (ch03-facts C1, C3)', () => {
  it("Apple's example: 16 x 16 threads in 2 x 4 groups of 8 x 4; thread (9,10) is in group (1,2) at (1,2)", () => {
    const apple: Launch = { height: 16, width: 16, blockW: 8, blockH: 4, style: 'threadgroups' };
    expect([layout(apple).gridX, layout(apple).gridY]).toEqual([2, 4]);
    const t = threadAt(apple, 9, 10);
    expect(t.blockIdx).toEqual({ x: 1, y: 2 });
    expect(t.threadIdx).toEqual({ x: 1, y: 2 });
    expect([t.col, t.row]).toEqual([9, 10]);
  });
  it('dispatchThreadgroups has the geometry of whole blocks: 408 surplus threads', () => {
    expect(layout({ ...BOOK, style: 'threadgroups' })).toEqual(layout(BOOK));
  });
  it('dispatchThreads creates exactly 4712 threads and none is surplus', () => {
    const l: Launch = { ...BOOK, style: 'threads' };
    const L = layout(l);
    expect([L.threadsX, L.threadsY, L.threads, L.surplus]).toEqual([76, 62, 4712, 0]);
    expect([L.gridX, L.gridY]).toEqual([5, 4]);
    // edge groups are smaller, and every thread of every group works
    expect(blockInfo(l, 4, 3)).toMatchObject({ w: 12, h: 14, launched: 168, working: 168 });
    expect(regions(l).map((r) => [r.blocks, r.launched, r.working])).toEqual([
      [12, 256, 256], [3, 192, 192], [4, 224, 224], [1, 168, 168],
    ]);
    const t = threadAt(l, 75, 61);
    expect(t.works).toBe(true);
    expect(t.groupSize).toEqual({ x: 12, y: 14 });
    expect(t.blockDim).toEqual({ x: 16, y: 16 });
    expect(() => threadAt(l, 76, 0)).toThrow();
  });
});

describe('flattening (pp. 50-52, 55; exercises 4 and 5, p. 65)', () => {
  it('row-major: M[2][1] of a 4 x 4 matrix is element 9 (p. 51)', () => {
    expect(flat(2, 1, 4)).toBe(9);
    expect(unflat(9, 4)).toEqual({ row: 2, col: 1 });
  });
  it('exercise 4a: row 20, column 10 of a 400-wide matrix is 8010', () => {
    expect(flat(20, 10, 400)).toBe(8010);
  });
  it('exercise 4b: in column-major order, with 500 rows, it is 5020', () => {
    expect(flatColMajor(20, 10, 500)).toBe(5020);
    expect(unflatColMajor(5020, 500)).toEqual({ row: 20, col: 10 });
  });
  it('exercise 5: x = 10, y = 20, z = 5 of a 400 x 500 x 300 tensor is 1,008,010', () => {
    expect(flat3(5, 20, 10, 500, 400)).toBe(1008010);
    expect(unflat3(1008010, 500, 400)).toEqual({ plane: 5, row: 20, col: 10 });
  });
  it('strides of a contiguous tensor: (2,5) gives (5,1); (300,500,400) gives (200000,400,1)', () => {
    expect(strides([2, 5])).toEqual([5, 1]); // the example in the PyTorch docs (ch03-facts D6)
    expect(strides([300, 500, 400])).toEqual([200000, 400, 1]);
    const s = strides([300, 500, 400]);
    expect(5 * s[0] + 20 * s[1] + 10 * s[2]).toBe(1008010);
  });
  it('column-major of M is the row-major layout of its transpose', () => {
    const d = { depth: 1, height: 3, width: 5 };
    for (let row = 0; row < d.height; row++)
      for (let col = 0; col < d.width; col++)
        // element (row, col) of M is element (col, row) of the transpose, which is height wide
        expect(offsetOf(d, 'col', 0, row, col)).toBe(flat(col, row, d.height));
    expect(stridesFor(d, 'col')).toEqual([15, 1, 3]);
    expect(stridesFor(d, 'row')).toEqual([15, 5, 1]);
  });
  it('elementAt inverts offsetOf for both layouts', () => {
    const d = { depth: 2, height: 3, width: 4 };
    for (let i = 0; i < 24; i++) {
      const e = elementAt(d, 'row', i)!;
      expect(offsetOf(d, 'row', e.plane, e.row, e.col)).toBe(i);
    }
    const m = { depth: 1, height: 3, width: 4 };
    for (let i = 0; i < 12; i++) {
      const e = elementAt(m, 'col', i)!;
      expect(offsetOf(m, 'col', e.plane, e.row, e.col)).toBe(i);
    }
    expect(elementAt(d, 'wrong', 3)).toBeNull();
  });
  it('correct indexing writes every slot exactly once', () => {
    for (const kind of ['row', 'col'] as const) {
      const c = indexingCoverage({ depth: 1, height: 4, width: 6 }, kind);
      expect([c.collisions.length, c.holes.length, c.outOfBounds.length]).toEqual([0, 0, 0]);
    }
  });
  it('the wrong stride (row*height + col) collides and leaves holes when the array is wider than tall', () => {
    const c = indexingCoverage({ depth: 1, height: 2, width: 5 }, 'wrong');
    expect(c.collisions).toEqual([2, 3, 4]);
    expect(c.holes).toEqual([7, 8, 9]);
    expect(c.outOfBounds).toEqual([]);
  });
  it('and writes past the end when it is taller than wide', () => {
    const c = indexingCoverage({ depth: 1, height: 5, width: 2 }, 'wrong');
    expect(c.holes).toEqual([2, 3, 4, 7, 8, 9]);
    expect(c.outOfBounds.map((o) => o.offset)).toEqual([10, 11, 15, 16, 20, 21]);
  });
  it('but is invisible on a square array', () => {
    const c = indexingCoverage({ depth: 1, height: 4, width: 4 }, 'wrong');
    expect([c.collisions.length, c.holes.length, c.outOfBounds.length]).toEqual([0, 0, 0]);
  });
});

describe('the grayscale arithmetic (Fig. 3.4 line 19; ch03-facts A8, A9)', () => {
  it('as printed, pure red is stored as 0; with the sum parenthesised it is 53', () => {
    expect(gray(255, 0, 0, 'printed')).toBe(0);
    expect(gray(255, 0, 0, 'printedParen')).toBe(53);
  });
  it('(200, 100, 50) is stored as 74 as printed and 116 parenthesised', () => {
    expect(gray(200, 100, 50, 'printed')).toBe(74);
    expect(gray(200, 100, 50, 'printedParen')).toBe(116);
  });
  it('as printed, the result never depends on red', () => {
    for (const r of [0, 1, 99, 255]) expect(gray(r, 120, 33, 'printed')).toBe(gray(0, 120, 33, 'printed'));
    expect(grayDetail(255, 0, 0, 'printed').terms[0]).toBe(0);
  });
  it('white: 255 with the constants of the text, 252 with the constants of the listing', () => {
    expect(gray(255, 255, 255, 'corrected')).toBe(255);
    expect(gray(255, 255, 255, 'printedParen')).toBe(252);
  });
  it('the corrected line stays within a byte for every input', () => {
    let lo = 255;
    let hi = 0;
    for (let r = 0; r < 256; r += 5)
      for (let g = 0; g < 256; g += 5)
        for (let b = 0; b < 256; b += 5) {
          const v = gray(r, g, b);
          lo = Math.min(lo, v);
          hi = Math.max(hi, v);
        }
    expect([lo, hi]).toEqual([0, 255]);
  });
  it('the corrected values the lesson quotes', () => {
    expect(gray(255, 0, 0)).toBe(76);
    expect(gray(200, 100, 50)).toBe(124);
  });
});

describe('one thread through the corrected kernel', () => {
  it('Thread (0,0) of Block (1,0): eleven steps ending in a store to Pout[1216]', () => {
    const p = latticeOf(BOOK, 0, 1, 0, 0);
    const s = stepsGrayscale(BOOK, p.x, p.y);
    expect(s.map((x) => x.op)).toEqual(['col', 'row', 'guard', 'grayOffset', 'rgbOffset', 'r', 'g', 'b', 'sum', 'cast', 'store', 'done']);
    expect(s[2].pass).toBe(true);
    expect(s[3].regs.grayOffset).toBe(1216);
    expect(s[4].regs.rgbOffset).toBe(3648);
    expect(s[5].mem).toMatchObject({ array: 'Pin', index: 3648, write: false });
    expect(s[6].mem).toMatchObject({ array: 'Pin', index: 3649 });
    expect(s[7].mem).toMatchObject({ array: 'Pin', index: 3650 });
    const [r, g, b] = pixelAt(16, 0);
    expect([s[5].mem!.value, s[6].mem!.value, s[7].mem!.value]).toEqual([r, g, b]);
    expect(s[10].mem).toEqual({ array: 'Pout', index: 1216, value: gray(r, g, b), write: true });
    expect(s[3].text).toContain('16*76 + 0 = 1216');
  });
  it('agrees with the kernel snippet run on the CPU emulator', () => {
    // labs/ch03/snippets/grayscale_kernel.cu, run on the pmpp emulator over this synthetic
    // 62 x 76 picture, read 130, 106, 100 from Pin[3648..3650] and stored 112 in Pout[1216].
    expect(pixelAt(16, 0)).toEqual([130, 106, 100]);
    expect(gray(130, 106, 100)).toBe(112);
    expect(grayDetail(130, 106, 100).sum).toBeCloseTo(112.492, 3);
  });
  it('a surplus thread stops at the guard and touches no memory', () => {
    const s = stepsGrayscale(BOOK, 77, 5);
    expect(s.map((x) => x.op)).toEqual(['col', 'row', 'guard', 'done']);
    expect(s[2].pass).toBe(false);
    expect(s[2].text).toContain('77 < 76, false');
    expect(s.some((x) => x.mem)).toBe(false);
    const below = stepsGrayscale(BOOK, 3, 63);
    expect(below[2].text).toContain('63 < 62, false');
  });
  it('the stored byte is the truncated float sum', () => {
    const s = stepsGrayscale(BOOK, 30, 40);
    const sum = s.find((x) => x.op === 'sum')!.regs.sum;
    expect(s.find((x) => x.op === 'cast')!.regs.byte).toBe(Math.trunc(sum));
  });
  it('the synthetic picture stays in byte range', () => {
    for (let row = 0; row < 62; row += 7)
      for (let col = 0; col < 76; col += 5) for (const v of pixelAt(row, col)) expect(v >= 0 && v <= 255).toBe(true);
  });
});
