import { describe, expect, it } from 'vitest';
import { NEIGHBOURS, PRESETS, inputsOf, isEarlier, presetFronts, stencil, stepAt, wavefronts } from './wavefront';

// Pins every number section 16.3 and the start page state about wavefronts, including the
// book's Fig. 16.2 numbering (notes/ch16.md, Figures, row 16.2).
const grid = (rows: number, cols: number, f: (r: number, c: number) => number) =>
  Array.from({ length: rows * cols }, (_, i) => f(Math.floor(i / cols), i % cols));

describe('the stencils of Fig. 16.2', () => {
  it('Smith-Waterman: the step of (r, c) is r + c, as printed in Fig. 16.2(c)', () => {
    const f = presetFronts('smith-waterman', 4, 4);
    expect(f.step).toEqual(grid(4, 4, (r, c) => r + c));
    expect(f.sizes).toEqual([1, 2, 3, 4, 3, 2, 1]);
    expect(presetFronts('smith-waterman', 11, 7).step).toEqual(grid(11, 7, (r, c) => r + c));
  });
  it('checkerboard: the step is the row, a constant front as wide as the table (Fig. 16.2(a))', () => {
    const f = presetFronts('checkerboard', 4, 4);
    expect(f.step).toEqual(grid(4, 4, (r) => r));
    expect(f.sizes).toEqual([4, 4, 4, 4]);
  });
  it('financial (our reading of Fig. 16.2(b)) is also a constant front, one row per step', () => {
    const f = presetFronts('financial', 4, 4);
    expect(f.step).toEqual(grid(4, 4, (r) => r));
    expect(f.sizes).toEqual([4, 4, 4, 4]);
  });
  it('H.264: the step is c + 2r, and a 4 x 4 table has fronts 1,1,2,2,2,2,2,2,1,1 (Fig. 16.2(d))', () => {
    const f = presetFronts('h264', 4, 4);
    expect(f.step).toEqual(grid(4, 4, (r, c) => c + 2 * r));
    expect(f.sizes).toEqual([1, 1, 2, 2, 2, 2, 2, 2, 1, 1]);
    expect(f.rowLag).toBe(2);
  });
  it('Fibonacci: one cell per front, so no parallelism at all (p. 374)', () => {
    const f = presetFronts('fibonacci');
    expect(f.rows).toBe(1);
    expect(f.steps).toBe(12);
    expect(f.widest).toBe(1);
    expect(f.reach).toBe(2);
    expect(f.rowLag).toBeNull();
  });
});

describe('the numbers the lesson states', () => {
  it('an 8 x 8 Smith-Waterman table: 15 steps, widest 8 (prediction p-sw8), R + C - 1 in general', () => {
    const f = presetFronts('smith-waterman', 8, 8);
    expect(f.steps).toBe(15);
    expect(f.widest).toBe(8);
    expect(f.mean).toBeCloseTo(64 / 15, 12);
    for (const [r, c] of [[2, 9], [16, 3], [5, 5], [1, 16]]) expect(presetFronts('smith-waterman', r, c).steps).toBe(r + c - 1);
  });
  it('an 8 x 8 H.264 table: 22 steps, the last cell at step 7 + 14 = 21 (q-h264)', () => {
    const f = presetFronts('h264', 8, 8);
    expect(f.steps).toBe(22);
    expect(stepAt(f, 7, 7)).toBe(21);
  });
  it('Smith-Waterman reads two fronts back, so only 2 earlier fronts need keeping for the score (q-live)', () => {
    expect(presetFronts('smith-waterman', 8, 8).reach).toBe(2);
    expect(presetFronts('checkerboard', 8, 8).reach).toBe(1);
    expect(presetFronts('h264', 8, 8).reach).toBe(1);
  });
  it('edit distance of two 1,000-token sequences: 1,999 steps of up to 1,000 cells (q-hook)', () => {
    const f = presetFronts('smith-waterman', 1000, 1000);
    expect(f.steps).toBe(1999);
    expect(f.widest).toBe(1000);
    expect(Math.round(f.mean)).toBe(500);
  });
  it('each row starts one step after the row above for the three-neighbour stencil, two for H.264', () => {
    expect(presetFronts('smith-waterman', 4, 4).rowLag).toBe(1);
    expect(presetFronts('checkerboard', 4, 4).rowLag).toBe(1);
    expect(presetFronts('h264', 4, 4).rowLag).toBe(2);
  });
});

describe('the model itself', () => {
  it('rejects an offset that is not an earlier cell', () => {
    expect(() => wavefronts(4, 4, [{ dr: 0, dc: 0 }])).toThrow(/not an earlier cell/);
    expect(() => wavefronts(4, 4, [{ dr: 0, dc: 1 }])).toThrow(/not an earlier cell/);
    expect(() => wavefronts(4, 4, [{ dr: 1, dc: -1 }])).toThrow(/not an earlier cell/);
    expect(() => wavefronts(0, 4, [])).toThrow();
  });
  it('the six offered neighbours are all earlier, and every preset uses only them', () => {
    expect(NEIGHBOURS).toHaveLength(6);
    for (const n of NEIGHBOURS) expect(isEarlier(n)).toBe(true);
    for (const p of PRESETS) expect(stencil(p.neighbours)).toHaveLength(p.neighbours.length);
  });
  it('no stencil: every cell is independent, one step', () => {
    const f = wavefronts(5, 3, []);
    expect(f.steps).toBe(1);
    expect(f.widest).toBe(15);
    expect(f.reach).toBe(0);
  });
  it('a cell runs one step after the latest of its inputs, and never with one of them', () => {
    for (const p of PRESETS) {
      const f = presetFronts(p.id, 6, 7);
      for (let r = 0; r < f.rows; r++)
        for (let c = 0; c < f.cols; c++) {
          const ins = inputsOf(f, r, c);
          const latest = ins.reduce((m, [rr, cc]) => Math.max(m, stepAt(f, rr, cc)), -1);
          expect(stepAt(f, r, c)).toBe(latest + 1);
        }
    }
  });
  it('inputs outside the table are boundary values, not inputs', () => {
    const f = presetFronts('smith-waterman', 3, 3);
    expect(inputsOf(f, 0, 0)).toEqual([]);
    expect(inputsOf(f, 0, 2)).toEqual([[0, 1]]);
    expect(inputsOf(f, 1, 1).sort()).toEqual([[0, 0], [0, 1], [1, 0]]);
  });
  it('the sizes add up to the table', () => {
    for (const p of PRESETS) {
      const f = presetFronts(p.id, 9, 5);
      expect(f.sizes.reduce((a, b) => a + b, 0)).toBe(45);
    }
  });
});
