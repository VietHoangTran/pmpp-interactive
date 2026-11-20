import { describe, expect, it } from 'vitest';
import {
  CLAUSES,
  CORRECT,
  DEFECTS,
  ORDERS,
  area,
  average,
  blurImage,
  challenge,
  count,
  countMap,
  numbered,
  patch,
  reference,
  span,
  spot,
  tally,
  textured,
  threadOrder,
  trace,
  uniform,
  withDefect,
  type BlurOptions,
} from './blur';

const opts = (over: Partial<BlurOptions> = {}): BlurOptions => ({ ...CORRECT, ...over });
const without = (...c: (typeof CLAUSES)[number][]) => ({ ...CORRECT.clauses, ...Object.fromEntries(c.map((k) => [k, false])) });
const pairs = (row: number, col: number, w: number, h: number, r: number) =>
  patch(row, col, w, h, r).map((p) => [p.curRow, p.curCol]);

describe("the book's worked numbers (pp. 56, 58, 59)", () => {
  it('output pixel (25, 50) with BLUR_SIZE 1 visits rows 24 to 26 and columns 49 to 51', () => {
    // The book's running picture is 62 rows by 76 columns.
    expect(pairs(25, 50, 76, 62, 1)).toEqual([
      [24, 49], [24, 50], [24, 51],
      [25, 49], [25, 50], [25, 51],
      [26, 49], [26, 50], [26, 51],
    ]);
    expect(count(25, 50, 76, 62, 1)).toBe(9);
  });
  it('output pixel (0, 0): the nine pairs of p. 58, five rejected, pixels = 4', () => {
    expect(pairs(0, 0, 76, 62, 1)).toEqual([
      [-1, -1], [-1, 0], [-1, 1],
      [0, -1], [0, 0], [0, 1],
      [1, -1], [1, 0], [1, 1],
    ]);
    const p = patch(0, 0, 76, 62, 1);
    expect(p.filter((x) => !x.inside).length).toBe(5);
    expect(p.filter((x) => x.inside).length).toBe(4);
    // Which clause turns each one away: the row test comes first in the source.
    expect(p.map((x) => x.failed)).toEqual(['rowLo', 'rowLo', 'rowLo', 'colLo', null, null, 'colLo', null, null]);
  });
  it('corners average 4 values, other edge pixels 6, interior pixels 9', () => {
    const w = 76, h = 62;
    for (const [r, c] of [[0, 0], [0, w - 1], [h - 1, 0], [h - 1, w - 1]]) expect(count(r, c, w, h, 1)).toBe(4);
    for (const [r, c] of [[0, 5], [h - 1, 5], [7, 0], [7, w - 1]]) expect(count(r, c, w, h, 1)).toBe(6);
    expect(count(25, 50, w, h, 1)).toBe(9);
    const m = countMap(w, h, 1);
    expect(m.filter((v) => v === 4).length).toBe(4);
    expect(m.filter((v) => v === 6).length).toBe(2 * (w - 2) + 2 * (h - 2));
    expect(m.filter((v) => v === 9).length).toBe((w - 2) * (h - 2));
  });
  it('BLUR_SIZE is the radius: 1 gives a 3 by 3 patch, 3 gives 7 by 7', () => {
    expect(area(1)).toBe(9);
    expect(area(3)).toBe(49);
  });
});

describe('counts for the questions', () => {
  it('BLUR_SIZE 3, 100 by 100, pixel (0,0): 16 values (rows 0 to 3, columns 0 to 3)', () => {
    expect(count(0, 0, 100, 100, 3)).toBe(16);
    expect(span(0, 100, 3)).toBe(4);
  });
  it('BLUR_SIZE 2, 8 wide and 6 high, pixel (0, 5): 15 values (rows 0 to 2, columns 3 to 7)', () => {
    expect(span(0, 6, 2)).toBe(3);
    expect(span(5, 8, 2)).toBe(5);
    expect(count(0, 5, 8, 6, 2)).toBe(15);
    const kept = patch(0, 5, 8, 6, 2).filter((p) => p.inside);
    expect([Math.min(...kept.map((p) => p.curCol)), Math.max(...kept.map((p) => p.curCol))]).toEqual([3, 7]);
  });
  it('the count is a row term times a column term, and agrees with the loop', () => {
    for (const [w, h, r] of [[3, 3, 1], [12, 5, 3], [4, 9, 2], [3, 12, 3]])
      for (let row = 0; row < h; ++row)
        for (let col = 0; col < w; ++col)
          expect(count(row, col, w, h, r)).toBe(patch(row, col, w, h, r).filter((p) => p.inside).length);
  });
  it('the explorer default (8 wide, 6 high, radius 1): 4 corners, 20 edge pixels, 24 interior, 352 reads of 432', () => {
    const m = countMap(8, 6, 1);
    expect([4, 6, 9].map((k) => m.filter((v) => v === k).length)).toEqual([4, 20, 24]);
    expect(m.reduce((a, b) => a + b, 0)).toBe(352); // (2+3+3+3+3+2) rows times (2+3*6+2) columns
    expect(8 * 6 * area(1)).toBe(432);
  });
});

describe('trace of one thread', () => {
  const pic = numbered(7, 6);
  it('corner (0,0): short-circuit evaluation, sum, count and truncated byte', () => {
    const t = trace(pic, 0, 0, 1);
    expect(t.steps.length).toBe(9);
    expect(t.steps[0].clauses).toEqual([false, null, null, null]); // curRow >= 0 fails; the rest is never evaluated
    expect(t.steps[3].clauses).toEqual([true, true, false, null]); // (0,-1): curCol >= 0 fails
    expect(t.steps[4].clauses).toEqual([true, true, true, true]);
    expect(t.steps[4].index).toBe(0);
    // numbered(7, 6): in[0] = 1, in[1] = 4, in[7] = 8, in[8] = 3
    expect([pic.data[0], pic.data[1], pic.data[7], pic.data[8]]).toEqual([1, 4, 8, 3]);
    expect(t.steps.map((s) => s.pixVal)).toEqual([0, 0, 0, 0, 1, 5, 5, 13, 16]);
    expect(t.steps.map((s) => s.pixels)).toEqual([0, 0, 0, 0, 1, 2, 2, 3, 4]);
    expect([t.pixVal, t.pixels, t.rejected]).toEqual([16, 4, 5]);
    expect(t.quotient).toBe(4);
    expect(t.byte).toBe(4);
  });
  it('the cast truncates: 19/4 = 4.75 is stored as 4', () => {
    const t = trace(pic, 5, 6, 1);
    expect([t.pixVal, t.pixels]).toEqual([19, 4]);
    expect(t.quotient).toBe(4.75);
    expect(t.byte).toBe(4);
  });
  it('the stepper presets: sums, counts and stored bytes', () => {
    const at = [[0, 0], [3, 0], [0, 3], [2, 6], [5, 6], [2, 3]];
    expect(at.map(([r, c]) => { const t = trace(pic, r, c, 1); return [t.pixVal, t.pixels, t.byte]; })).toEqual([
      [16, 4, 4], [20, 6, 3], [23, 6, 3], [28, 6, 4], [19, 4, 4], [46, 9, 5],
    ]);
  });
  it('the six presets of the stepper keep 4, 6, 6, 6, 4 and 9 values', () => {
    const at = [[0, 0], [3, 0], [0, 3], [2, 6], [5, 6], [2, 3]];
    expect(at.map(([r, c]) => trace(pic, r, c, 1).pixels)).toEqual([4, 6, 6, 6, 4, 9]);
  });
  it('agrees with the whole-picture run', () => {
    const ref = reference(pic, 1);
    for (let i = 0; i < ref.length; ++i) expect(trace(pic, Math.floor(i / 7), i % 7, 1).byte).toBe(ref[i]);
  });
  it('average: float division then truncation toward zero', () => {
    expect(average(14, 4)).toEqual({ quotient: 3.5, byte: 3 });
    expect(average(2295, 9).byte).toBe(255); // nine bytes of 255: an average of bytes stays in range
    expect(Number.isNaN(average(0, 0).byte)).toBe(true);
  });
});

describe('the model against the compiled kernel', () => {
  // labs/ch03/snippets/blur_kernel.cu, compiled as host C++ (Apple clang 21) and run one
  // thread at a time in launch order, 4x4 blocks. CPU only; nothing here ran on a GPU.
  it('numbered 7 by 6', () => {
    expect(reference(numbered(7, 6), 1)).toEqual([
      4, 5, 4, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 5, 5, 5, 4,
      3, 3, 4, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 5, 6, 5, 4, 4, 4, 4,
    ]);
  });
  it('textured 6 by 7, separate buffers and in place', () => {
    const pic = textured(6, 7);
    expect(blurImage(pic, opts()).out).toEqual([
      49, 62, 80, 98, 116, 129, 57, 66, 84, 102, 120, 129, 65, 74, 92, 110, 128, 137, 73, 82, 100,
      118, 136, 145, 81, 90, 108, 126, 144, 153, 89, 98, 116, 134, 152, 161, 97, 102, 120, 138, 156, 165,
    ]);
    expect(blurImage(pic, opts({ inPlace: true })).out).toEqual([
      49, 66, 78, 98, 119, 126, 59, 69, 87, 103, 119, 129, 70, 75, 92, 110, 126, 135, 72, 86, 98,
      117, 136, 142, 83, 92, 110, 124, 140, 150, 93, 99, 116, 134, 149, 157, 96, 108, 117, 136, 155, 159,
    ]);
  });
  it('one spot on 6 by 6, one block, separate buffers and in place', () => {
    const pic = spot(6, 6);
    const o = opts({ blockW: 6, blockH: 6 });
    expect(blurImage(pic, o).out).toEqual([
      0, 0, 0, 0, 0, 0, 0, 28, 28, 28, 0, 0, 0, 28, 28, 28, 0, 0,
      0, 28, 28, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    expect(blurImage(pic, { ...o, inPlace: true }).out).toEqual([
      0, 0, 0, 0, 0, 0, 0, 28, 31, 31, 3, 0, 4, 35, 41, 11, 5, 1,
      6, 9, 10, 7, 2, 1, 2, 3, 3, 2, 1, 0, 1, 1, 1, 1, 0, 0,
    ]);
  });
});

describe('thread order', () => {
  it('blocks run one after another; inside a block the order is forward, reversed or shuffled', () => {
    const f = threadOrder(6, 7, 4, 4, 'forward');
    expect(f.length).toBe(64);
    expect(f.slice(0, 5)).toEqual([{ row: 0, col: 0 }, { row: 0, col: 1 }, { row: 0, col: 2 }, { row: 0, col: 3 }, { row: 1, col: 0 }]);
    expect(f[16]).toEqual({ row: 0, col: 4 }); // second block
    const r = threadOrder(6, 7, 4, 4, 'reversed');
    expect(r[0]).toEqual({ row: 3, col: 3 });
    expect(r[16]).toEqual({ row: 3, col: 7 });
    const s = threadOrder(6, 7, 4, 4, 'shuffled', 3);
    const key = (p: { row: number; col: number }) => p.row * 100 + p.col;
    expect(s.map(key).sort((a, b) => a - b)).toEqual(f.map(key).sort((a, b) => a - b));
    expect(s.map(key)).not.toEqual(f.map(key));
    expect(threadOrder(6, 7, 4, 4, 'shuffled', 3)).toEqual(s); // the seed fixes it
  });
});

describe('separate buffers', () => {
  it('the correct kernel gives the same output under every order, with no bad access', () => {
    for (const pic of [textured(6, 7), spot(6, 6), numbered(7, 6), uniform(5, 5)])
      for (const order of ORDERS)
        for (const seed of [1, 2, 9]) {
          const r = blurImage(pic, opts({ order, seed }));
          expect(r.wrong).toEqual([]);
          expect(r.accesses).toEqual([]);
          expect(r.zeroDivisions).toEqual([]);
          expect(r.overwrittenReads).toBe(0);
        }
  });
  it('6 by 7 under 4 by 4 blocks: 2 by 2 blocks, 64 threads, 22 surplus', () => {
    const r = blurImage(textured(6, 7), opts());
    expect([r.gridW, r.gridH, r.threads, r.surplus]).toEqual([2, 2, 64, 22]);
  });
});

describe('in place', () => {
  const pic = spot(6, 6);
  const o = opts({ blockW: 6, blockH: 6, inPlace: true });
  const fwd = blurImage(pic, o);
  const rev = blurImage(pic, { ...o, order: 'reversed' });
  it('forward and reversed differ from each other and from the reference', () => {
    expect(fwd.wrong.length).toBeGreaterThan(0);
    expect(rev.wrong.length).toBeGreaterThan(0);
    expect(fwd.out).not.toEqual(rev.out);
    expect(fwd.out).not.toEqual(fwd.reference);
    expect(rev.out).not.toEqual(rev.reference);
  });
  it('every access is inside the buffer: nothing for a sanitizer to see', () => {
    expect(fwd.accesses).toEqual([]);
    expect(rev.accesses).toEqual([]);
  });
  it('the numbers the animation and the lesson state', () => {
    // 7 threads store zeros before thread (1,1) sees the spot and stores 28.
    expect(fwd.ran.findIndex((t) => t.row === 1 && t.col === 1)).toBe(7);
    expect(blurImage(pic, { ...o, limit: 7 }).out).toEqual(pic.data);
    expect(blurImage(pic, { ...o, limit: 8 }).out[7]).toBe(28);
    // Its neighbour (1,2) then reads that 28 where the input held 0, and stores 31.
    expect(blurImage(pic, { ...o, limit: 9 }).out[8]).toBe(31);
    expect(fwd.reference[8]).toBe(28);
    expect(fwd.wrong.length).toBe(24);
    expect(rev.wrong.length).toBe(19);
    expect(fwd.out.filter((v, i) => v !== rev.out[i]).length).toBe(31);
    // The spot itself: 28 in the reference, 41 in both wrong pictures.
    expect([fwd.reference[14], fwd.out[14], rev.out[14]]).toEqual([28, 41, 41]);
    // 110 of the reads found an element that some thread had already stored.
    expect(fwd.overwrittenReads).toBe(110);
    // In order the smear reaches the bottom row and leaves the top row alone; reversed is the opposite.
    expect(fwd.out.slice(0, 6)).toEqual([0, 0, 0, 0, 0, 0]);
    expect(fwd.out.slice(30).some((v) => v > 0)).toBe(true);
    expect(rev.out.slice(30)).toEqual([0, 0, 0, 0, 0, 0]);
    expect(rev.out.slice(0, 6).some((v) => v > 0)).toBe(true);
  });
  it('limit cuts the run and reports what has been stored', () => {
    const part = blurImage(pic, opts({ blockW: 6, blockH: 6, limit: 10 }));
    expect(part.ran.length).toBe(10);
    expect(part.written.filter(Boolean).length).toBe(10);
  });
  it('three orders, three different pictures on the textured input', () => {
    const p = textured(6, 7);
    const outs = ORDERS.map((order) => blurImage(p, opts({ inPlace: true, order })).out.join());
    expect(new Set(outs).size).toBe(3);
  });
});

describe('the divisor', () => {
  it('dividing by the patch area changes border pixels only, and only downwards', () => {
    for (const pic of [textured(6, 7), uniform(6, 7)]) {
      const r = blurImage(pic, opts({ divisor: 'area' }));
      const border = (i: number) => {
        const row = Math.floor(i / pic.w), col = i % pic.w;
        return row === 0 || col === 0 || row === pic.h - 1 || col === pic.w - 1;
      };
      expect(r.wrong.every(border)).toBe(true);
      expect(r.wrong.length).toBe(2 * 6 + 2 * 7 - 4);
      expect(r.wrong.every((i) => r.out[i] < r.reference[i])).toBe(true);
      expect(r.accesses).toEqual([]);
    }
  });
  it('uniform grey 180: interior 180, edges 120, corners 80', () => {
    const r = blurImage(uniform(6, 7), opts({ divisor: 'area' }));
    expect([r.out[0], r.out[2], r.out[8]]).toEqual([80, 120, 180]);
  });
});

describe('the inner test, clause by clause', () => {
  const pic = textured(6, 7);
  it('without curCol < w: silent wrong values, read from the start of the next row', () => {
    const r = blurImage(pic, opts({ clauses: without('colHi') }));
    const t = tally(r.accesses);
    // The right-hand column has 7 pixels; rows 0 and 6 look at 2 picture rows, the others at 3: 19 reads.
    // in[curRow*6 + 6] is the first pixel of row curRow+1, except for curRow = 6 (threads of rows 5 and 6).
    expect(t).toEqual({ readWrapped: 17, readOutside: 2, writeWrapped: 0, writeOutside: 0 });
    const first = r.accesses[0];
    expect(first.thread).toEqual({ row: 0, col: 5 });
    expect(first.wanted).toEqual({ row: 0, col: 6 });
    expect(first.index).toBe(6);
    expect(first.got).toEqual({ row: 1, col: 0 });
    // Only reads aimed at the last row leave the buffer.
    expect(r.accesses.filter((a) => a.kind === 'outside').map((a) => a.index)).toEqual([42, 42]);
    expect(r.wrong.every((i) => i % 6 === 5)).toBe(true);
    expect(r.wrong.length).toBe(7);
  });
  it('without curCol >= 0: the left-hand column reads the end of the previous row', () => {
    const r = blurImage(pic, opts({ clauses: without('colLo') }));
    expect(tally(r.accesses)).toEqual({ readWrapped: 17, readOutside: 2, writeWrapped: 0, writeOutside: 0 });
    expect(r.accesses[0].index).toBe(-1);
    expect(r.accesses[1].got).toEqual({ row: 0, col: 5 });
    expect(r.wrong.every((i) => i % 6 === 0)).toBe(true);
  });
  it('without a row test every bad read is outside the buffer', () => {
    const lo = blurImage(pic, opts({ clauses: without('rowLo') }));
    expect(tally(lo.accesses)).toEqual({ readWrapped: 0, readOutside: 16, writeWrapped: 0, writeOutside: 0 });
    expect(lo.wrong.every((i) => i < 6)).toBe(true);
    const hi = blurImage(pic, opts({ clauses: without('rowHi') }));
    expect(tally(hi.accesses)).toEqual({ readWrapped: 0, readOutside: 16, writeWrapped: 0, writeOutside: 0 });
    expect(hi.wrong.every((i) => i >= 36)).toBe(true);
  });
  it('a uniform picture hides the missing column test almost completely', () => {
    const r = blurImage(uniform(6, 7), opts({ clauses: without('colHi') }));
    expect(r.accesses.length).toBe(19);
    expect(r.wrong).toEqual([35, 41]); // only the two pixels whose stray read left the buffer
  });
});

describe('the outer guard', () => {
  it("the book's 62 by 76 picture under 16 by 16 blocks: 408 surplus threads all store something", () => {
    const pic = uniform(76, 62);
    const r = blurImage(pic, opts({ blockW: 16, blockH: 16, outerGuard: false }));
    expect(r.threads).toBe(64 * 80);
    expect(r.surplus).toBe(408);
    const writes = r.accesses.filter((a) => a.op === 'write');
    expect(writes.length).toBe(408);
    expect(tally(r.accesses)).toEqual({ readWrapped: 0, readOutside: 0, writeWrapped: 244, writeOutside: 164 });
    // 269 of the 408 find no valid position at all and divide 0 by 0; the other 139 average 1 to 3 values.
    expect(r.zeroDivisions.length).toBe(269);
    expect(408 - r.zeroDivisions.length).toBe(139);
    // With the guard in place the same launch is clean.
    const ok = blurImage(pic, opts({ blockW: 16, blockH: 16 }));
    expect(ok.accesses).toEqual([]);
    expect(ok.wrong).toEqual([]);
  });
  it('6 by 7 under 4 by 4 blocks: wrapped writes, writes outside, 0/0, and a result decided by who stores last', () => {
    const pic = textured(6, 7);
    const r = blurImage(pic, opts({ outerGuard: false }));
    // Columns 6 and 7 of rows 0 to 5 store into columns 0 and 1 of the next row: 12 writes.
    // Rows 6 (two threads) and 7 (eight threads) store past the end: 10 writes.
    expect(tally(r.accesses)).toEqual({ readWrapped: 0, readOutside: 0, writeWrapped: 12, writeOutside: 10 });
    // Column 7 (seven threads with a row in range) and thread (7,7) find no valid position.
    expect(r.zeroDivisions.length).toBe(8);
    // Twelve pixels are scribbled on. The two in row 4 belong to a block that runs after the
    // scribbling block, so their owners store last and they come out right: ten stay wrong.
    expect(r.wrong).toEqual([6, 7, 12, 13, 18, 19, 30, 31, 36, 37]);
    expect(r.wrong.filter((i) => Number.isNaN(r.out[i]))).toEqual([7, 13, 19, 31, 37]);
    // The order inside a block does not change which store comes last here; the order of blocks does.
    for (const order of ORDERS) expect(blurImage(pic, opts({ outerGuard: false, order })).wrong).toEqual(r.wrong);
  });
});

describe('single defects for the challenge', () => {
  const pic = textured(6, 7);
  it('every defect alone gives a wrong picture, and no two give the same one', () => {
    const outs = DEFECTS.map((d) => blurImage(pic, withDefect(d)));
    for (const r of outs) expect(r.wrong.length).toBeGreaterThan(0);
    expect(new Set(outs.map((r) => r.out.join())).size).toBe(DEFECTS.length);
  });
  it('a challenge is three different defects, fixed by its seed', () => {
    for (let s = 1; s < 20; ++s) {
      const c = challenge(s);
      expect(new Set(c).size).toBe(3);
      expect(challenge(s)).toEqual(c);
    }
    expect(challenge(1)).toEqual(['inPlace', 'rowLo', 'colHi']); // the set the page opens with
    expect(challenge(2)).toEqual(['rowLo', 'colHi', 'rowHi']);
  });
});
