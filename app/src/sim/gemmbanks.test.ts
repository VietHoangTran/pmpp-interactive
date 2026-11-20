import { describe, expect, it } from 'vitest';
import { freeLds, freePitches, sharedBytes, stripRead, stripWorst, wStore, wStoreWorst, wrongOffsetReads, type Banks } from './gemmbanks';
import type { Layout, Width } from './gemmremap';

const worstA = (layout: Layout, ld: number, banks: Banks, W: Width = 32) => stripWorst({ layout, strip: 'A', ld, banks, W }).degree;
const worstB = (layout: Layout, banks: Banks, W: Width = 32) => stripWorst({ layout, strip: 'B', ld: 8, banks, W }).degree;

describe('Fig. 15.11: warp 0 reads the first A strip element (p. 362)', () => {
  it('(a) leading dimension 8: elements 0, 32, ..., 224, all in bank 0: 8-way', () => {
    const r = stripRead({ layout: 'remap', strip: 'A', ld: 8, banks: 32, W: 32, warp: 0, i: 0, p: 0 });
    const groups = [0, 4, 8, 12, 16, 20, 24, 28].map((l) => r.words[l].word);
    expect(groups).toEqual([0, 32, 64, 96, 128, 160, 192, 224]);
    expect(r.lanes.every((l) => l.bank === 0)).toBe(true);
    expect(r.degree).toBe(8);
  });
  it('(b) leading dimension 9: elements 0, 36, ..., 252 in banks 0, 4, ..., 28: no conflict', () => {
    const r = stripRead({ layout: 'remap', strip: 'A', ld: 9, banks: 32, W: 32, warp: 0, i: 0, p: 0 });
    expect([0, 4, 8, 12, 16, 20, 24, 28].map((l) => r.words[l].word)).toEqual([0, 36, 72, 108, 144, 180, 216, 252]);
    expect([0, 4, 8, 12, 16, 20, 24, 28].map((l) => r.lanes[l].bank)).toEqual([0, 4, 8, 12, 16, 20, 24, 28]);
    expect(r.degree).toBe(1);
  });
  it('the four lanes of a lane row read one word: a broadcast, not a conflict', () => {
    const r = stripRead({ layout: 'remap', strip: 'A', ld: 9, banks: 32, W: 32, warp: 0, i: 0, p: 0 });
    expect(new Set(r.words.slice(0, 4).map((w) => w.word)).size).toBe(1);
    expect(r.sharedWord).toBe(true);
  });
  it('the "second elements" are column 1 of the same rows: bank 1 unpadded; 1, 5, ..., 29 padded (notes, Unclear 6)', () => {
    const a = stripRead({ layout: 'remap', strip: 'A', ld: 8, banks: 32, W: 32, warp: 0, i: 1, p: 0 });
    const b = stripRead({ layout: 'remap', strip: 'A', ld: 9, banks: 32, W: 32, warp: 0, i: 1, p: 0 });
    expect(a.lanes.every((l) => l.bank === 1)).toBe(true);
    expect([0, 4, 8, 12, 16, 20, 24, 28].map((l) => b.lanes[l].bank)).toEqual([1, 5, 9, 13, 17, 21, 25, 29]);
  });
});

describe('ch15-facts E1: worst case over every access', () => {
  it('§15.5 layout, A strip: ld 8 is 8-way on 32 banks and 4-way on 64', () => {
    expect([worstA('remap', 8, 32), worstA('remap', 8, 64)]).toEqual([8, 4]);
  });
  it('§15.5 layout, A strip: ld 9 has no conflict on either', () => {
    expect([worstA('remap', 9, 32), worstA('remap', 9, 64)]).toEqual([1, 1]);
  });
  it('§15.5 layout, A strip: ld 10 / 12 / 16 are 2 / 4 / 8-way on 32 banks and none / 2 / 8-way on 64', () => {
    expect([10, 12, 16].map((ld) => worstA('remap', ld, 32))).toEqual([2, 4, 8]);
    expect([10, 12, 16].map((ld) => worstA('remap', ld, 64))).toEqual([1, 2, 8]);
  });
  it('the rule: any odd ld is free on 32 banks; on 64 banks any ld not divisible by 4', () => {
    expect(freeLds('remap', 32, 32)).toEqual([9, 11, 13, 15, 17]);
    expect(freeLds('remap', 64, 32)).toEqual([9, 10, 11, 13, 14, 15, 17]);
  });
  it('§15.5 layout, B strip: no conflict (4 distinct words, each broadcast to 8 lanes)', () => {
    expect([worstB('remap', 32), worstB('remap', 64)]).toEqual([1, 1]);
    const r = stripRead({ layout: 'remap', strip: 'B', ld: 8, banks: 32, W: 32, warp: 0, i: 0, p: 0 });
    expect(new Set(r.words.map((w) => w.word)).size).toBe(4);
  });
  it('§15.3/§15.4 layout, A strip: 2-way at ld 8 on both bank counts, none at ld 9', () => {
    expect([worstA('coarse', 8, 32), worstA('coarse', 8, 64), worstA('coarse', 9, 32), worstA('coarse', 9, 64)]).toEqual([2, 2, 1, 1]);
  });
  it('§15.3/§15.4 layout, B strip: 4-way on 32 banks (16 distinct words 8 apart), 2-way on 64', () => {
    expect([worstB('coarse', 32), worstB('coarse', 64)]).toEqual([4, 2]);
    const r = stripRead({ layout: 'coarse', strip: 'B', ld: 8, banks: 32, W: 32, warp: 0, i: 0, p: 0 });
    const words = [...new Set(r.words.map((w) => w.word))];
    expect(words.length).toBe(16);
    expect(words[1] - words[0]).toBe(8);
  });
  it('64-wide wavefront (8 x 8 lanes), A strip: ld 8 is 8-way on 32 banks and 4-way on 64; ld 9 free', () => {
    expect([worstA('remap', 8, 32, 64), worstA('remap', 8, 64, 64), worstA('remap', 9, 32, 64), worstA('remap', 9, 64, 64)]).toEqual([8, 4, 1, 1]);
  });
});

describe('every access of one kind conflicts equally (the word index is affine in warp, i and p)', () => {
  it('holds for the strips in both layouts, widths and bank counts at several leading dimensions', () => {
    for (const layout of ['coarse', 'remap'] as const)
      for (const strip of ['A', 'B'] as const)
        for (const W of [32, 64] as Width[])
          for (const banks of [32, 64] as Banks[])
            for (const ld of [8, 9, 10, 12]) {
              const first = stripRead({ layout, strip, ld, banks, W, warp: 0, i: 0, p: 0 }).degree;
              expect(stripWorst({ layout, strip, ld, banks, W }).degree).toBe(first);
            }
  });
  it('holds for the W-tile store', () => {
    for (const mapping of ['scalar', 'float4'] as const)
      for (const pitch of [128, 129, 132, 136])
        expect(wStoreWorst({ mapping, pitch, banks: 32, W: 32 }).degree).toBe(wStore({ mapping, pitch, banks: 32, W: 32, warp: 0, step: 0 }).degree);
  });
});

describe('the project’s transposed W-tile store into B_s (ours)', () => {
  const worst = (mapping: 'scalar' | 'float4', pitch: number, banks: Banks = 32, W: Width = 32) => wStoreWorst({ mapping, pitch, banks, W }).degree;
  it('a warp’s lanes own 4 rows x 8 columns of W with scalar loads', () => {
    const r = wStore({ mapping: 'scalar', pitch: 128, banks: 32, W: 32, warp: 0, step: 0 });
    expect(new Set(r.words.map((w) => w.col)).size).toBe(4);
    expect(new Set(r.words.map((w) => w.row)).size).toBe(8);
  });
  it('pitch 128: 8-way with scalar loads, 2-way with float4 loads', () => {
    expect([worst('scalar', 128), worst('float4', 128)]).toEqual([8, 2]);
  });
  it('pitch 129: up to 4-way with scalar loads', () => {
    expect(worst('scalar', 129)).toBe(4);
  });
  it('pitch 132 (16 bytes of padding): free with either load mapping', () => {
    expect([worst('scalar', 132), worst('float4', 132)]).toEqual([1, 1]);
    // k·132 = 4k (mod 32), plus n = 0..3: 32 distinct banks.
    const r = wStore({ mapping: 'scalar', pitch: 132, banks: 32, W: 32, warp: 0, step: 0 });
    expect(new Set(r.lanes.map((l) => l.bank)).size).toBe(32);
  });
  it('conflict-free pitches from 128 to 148 on 32 banks: 132, 140, 148 for both mappings', () => {
    expect(freePitches('scalar', 32, 32)).toEqual([132, 140, 148]);
    expect(freePitches('float4', 32, 32)).toEqual([132, 140, 148]);
  });
  it('with 64-wide wavefronts pitch 132 is no longer free for scalar loads (a model)', () => {
    expect(worst('scalar', 132, 32, 64)).toBe(2);
  });
});

describe('cost of the padding (notes [inferred])', () => {
  it('A_s grows from 4096 to 4608 bytes; with B_s, 8704', () => {
    expect(sharedBytes(8).A).toBe(4096);
    expect(sharedBytes(9)).toEqual({ A: 4608, B: 4096, total: 8704 });
  });
});

describe('the offset the book does not change (notes, Unclear 4)', () => {
  it('thread row 0 is unaffected', () => {
    expect(wrongOffsetReads(0).every((m) => m.right)).toBe(true);
  });
  it('thread row 1 starts at word 64: row 7, column 1, and reads the never-written padding column', () => {
    const m = wrongOffsetReads(1);
    expect([m[0].word, m[0].row, m[0].col]).toEqual([64, 7, 1]);
    expect(m.some((x) => x.padding)).toBe(true);
    expect(m.every((x) => !x.right)).toBe(true);
  });
  it('thread row 15 starts 120 words early: row 106, column 6 instead of row 120', () => {
    const m = wrongOffsetReads(15);
    expect([m[0].word, m[0].row, m[0].col]).toEqual([960, 106, 6]);
    expect(15 * 8 * 9 - m[0].word).toBe(120);
  });
});
