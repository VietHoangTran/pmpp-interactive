import { describe, expect, it } from 'vitest';
import {
  BANK_MODELS,
  BANK_WIDTHS,
  computeRange,
  slotOf,
  warpUse,
  wordOf,
  wordsNeeded,
  worstCompute,
  worstOf,
  type BankAccess,
  type BankKernel,
  type BankLayout,
} from './wavebanks';
import { loadsOf, cellAt, type Scheme } from './wavetile';

const NV = BANK_MODELS.nvidia;
const AMD = BANK_MODELS.amd64;
const row = (kernel: BankKernel, layout: BankLayout, m = NV) => BANK_WIDTHS.map((W) => worstCompute(kernel, layout, W, m));
const range = (kernel: BankKernel, layout: BankLayout, m = NV) => BANK_WIDTHS.map((W) => computeRange(kernel, layout, W, m).join('-'));

describe('ch16-facts E1 on NVIDIA (32 banks, 32-lane warps)', () => {
  it('square tiles: unpadded 1 at every width', () => {
    expect(row('square', 'unpadded')).toEqual([1, 1, 1, 1]);
  });
  it('square tiles: one gap per row makes them 2, 8, 31 to 32, 32', () => {
    expect(row('square', 'rowgap')).toEqual([2, 8, 32, 32]);
    expect(range('square', 'rowgap')[2]).toBe('31-32');
  });
  it('square tiles: pad() 1, 1, 31 to 32, 2', () => {
    expect(row('square', 'pad')).toEqual([1, 1, 32, 2]);
    expect(range('square', 'pad')[2]).toBe('31-32');
  });
  it('sheared tiles unpadded: 2, 8, 31 to 32, 32', () => {
    expect(row('sheared', 'unpadded')).toEqual([2, 8, 32, 32]);
    expect(range('sheared', 'unpadded')).toEqual(['2-2', '8-8', '31-32', '32-32']);
    // at W = 32 the loads with an edge lane on global memory are 31, the write and load_w 32
    expect((['write', 'n', 'w', 'nw'] as BankAccess[]).map((a) => worstOf('sheared', a, 'unpadded', 32, NV).degree)).toEqual([32, 31, 32, 31]);
  });
  it('sheared tiles: one gap per row 1 at every width; pad() 1, 1, 1, 2', () => {
    expect(row('sheared', 'rowgap')).toEqual([1, 1, 1, 1]);
    expect(row('sheared', 'pad')).toEqual([1, 1, 1, 2]);
  });
  it('store_tile and initialize_tile are conflict-free in every layout, for both kernels', () => {
    for (const kernel of ['square', 'sheared'] as BankKernel[])
      for (const layout of ['unpadded', 'rowgap', 'pad'] as BankLayout[])
        for (const W of BANK_WIDTHS) {
          expect(worstOf(kernel, 'store', layout, W, NV).degree).toBe(1);
          expect(worstOf(kernel, 'init', layout, W, NV).degree).toBe(1);
        }
  });
});

describe('the AMD model (64 banks, 64-lane wavefronts)', () => {
  it('sheared unpadded 1, 4, 16, 64; one gap per row and pad() with LOG_NUM_BANKS 6 give none', () => {
    expect(row('sheared', 'unpadded', AMD)).toEqual([1, 4, 16, 64]);
    expect(row('sheared', 'rowgap', AMD)).toEqual([1, 1, 1, 1]);
    expect(row('sheared', 'pad', AMD)).toEqual([1, 1, 1, 1]);
  });
});

describe('the questions\' numbers', () => {
  it('q-banks16: sheared W = 16 unpadded, lane t writes word 16t + d: two banks, 8 lanes each', () => {
    for (let k = 0; k < 16; k++) {
      const u = warpUse('sheared', 'write', 'unpadded', 16, NV, k, 0);
      expect(u.lanes.map((l) => l.word)).toEqual(u.lanes.map((l) => 16 * l.thread + k));
      expect(u.result.banksUsed).toBe(2);
      expect(u.result.maxLanesPerBank).toBe(8);
      expect(u.degree).toBe(8);
      expect(warpUse('sheared', 'write', 'rowgap', 16, NV, k, 0).degree).toBe(1);
    }
  });
  it('q-pad-square: square tiles stride W - 1 unpadded (no conflict); padded they stride W (32-way at W = 32)', () => {
    const u = warpUse('square', 'write', 'unpadded', 32, NV, 31, 0);
    const words = u.lanes.map((l) => l.word!);
    expect(words.slice(1).map((w, i) => w - words[i])).toEqual(Array(31).fill(31));
    expect(u.degree).toBe(1);
    expect(warpUse('square', 'write', 'rowgap', 32, NV, 31, 0).degree).toBe(32);
  });
});

describe('the slots agree with the kernels in wavetile.ts', () => {
  it('every lane\'s load slot is the one loadsOf reads, and null exactly when the load goes to global memory', () => {
    for (const scheme of ['square', 'sheared'] as Scheme[])
      for (const W of [8, 16]) {
        const g = { Lseq: 3 * W, W, scheme };
        const steps = scheme === 'square' ? 2 * W - 1 : W;
        for (let k = 0; k < steps; k++)
          for (let t = 0; t < W; t++) {
            const c = cellAt(g, { row: 1, col: 1 }, k, t);
            const write = slotOf(scheme, 'write', W, k, t);
            expect(write === null).toBe(!c.active);
            if (!c.active) continue;
            expect(write).toEqual([c.rTile, c.qTile]);
            for (const ld of loadsOf(g, c)) expect(slotOf(scheme, ld.name, W, k, t)).toEqual(ld.slot);
          }
      }
  });
  it('storage: W² words unpadded, W(W + 1) with a gap per row; the book\'s formula covers pad()', () => {
    expect(wordsNeeded('unpadded', 32)).toBe(1024);
    expect(wordsNeeded('rowgap', 64)).toBe(64 * 65);
    expect(wordsNeeded('pad', 32)).toBe(1055); // pad(1023) + 1
    expect(32 * 32 + ((32 * 32) >> 5)).toBe(1056); // the book's host formula, p. 396
    expect(wordOf('pad', 64, 1, 0)).toBe(66); // two gaps per 64-word row
  });
});
