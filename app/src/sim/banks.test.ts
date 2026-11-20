import { describe, expect, it } from 'vitest';
import { BANK_PRESETS, EX5_STRIDES, bankOf, gcdRule, halfCase, sameWordCase, strideCase, tileAccess, warpBanks } from './banks';

describe('bank arithmetic (book p. 139; NVIDIA, infra A10)', () => {
  it('bank = 4-byte word index mod 32', () => {
    expect([0, 1, 31, 32, 33, 64, 992].map((w) => bankOf(w))).toEqual([0, 1, 31, 0, 1, 0, 0]);
    expect(bankOf(32, 64)).toBe(32);
  });
  it('32 consecutive floats: 32 banks, one pass', () => {
    const r = strideCase(1);
    expect(r.passes).toBe(1);
    expect(r.banksUsed).toBe(32);
  });
});

describe('Exercise 5 (p. 155): a[threadIdx.x * stride], one 32-lane warp', () => {
  const expected = { 32: 32, 31: 1, 24: 8, 16: 16, 12: 4, 8: 8, 7: 1 } as Record<number, number>;
  const banksUsed = { 32: 1, 31: 32, 24: 4, 16: 2, 12: 8, 8: 4, 7: 32 } as Record<number, number>;
  for (const s of EX5_STRIDES)
    it(`stride ${s}: ${expected[s]}-way`, () => {
      const r = strideCase(s);
      expect(r.passes).toBe(expected[s]);
      expect(r.maxLanesPerBank).toBe(expected[s]);
      expect(r.banksUsed).toBe(banksUsed[s]);
      // and the closed form agrees
      expect(gcdRule(s)).toEqual({ g: expected[s], banksUsed: banksUsed[s], lanesPerBank: expected[s] });
    });
  it('stride 24 hits banks 0, 8, 16 and 24', () => {
    expect(strideCase(24).perBank.filter((b) => b.lanes.length).map((b) => b.bank)).toEqual([0, 8, 16, 24]);
  });
  it('every index stays inside a[1024]', () => {
    expect(Math.max(...EX5_STRIDES) * 31).toBeLessThan(1024);
  });
  it('the gcd rule matches the model for every stride from 1 to 64', () => {
    for (let s = 1; s <= 64; s++) expect(strideCase(s).passes).toBe(gcdRule(s).lanesPerBank);
  });
  it('Exercise 1(h), bc_s[threadIdx.x*4]: 8 banks, 4 lanes each', () => {
    expect(strideCase(4).passes).toBe(4);
    expect(strideCase(4).banksUsed).toBe(8);
  });
});

describe('the corner-turned store (pp. 139–141)', () => {
  it('32 × 32 tile, a[threadIdx.x][threadIdx.y]: every lane on bank 0, 32 passes', () => {
    for (const ty0 of [0, 5, 31]) {
      const r = tileAccess({ T: 32, pitch: 32, orientation: 'x-first', ty0 });
      expect(r.passes).toBe(32);
      expect(r.banksUsed).toBe(1);
      expect(r.lanes[0].bank).toBe(ty0);
    }
  });
  it('one padding column, pitch 33: lane i on bank i, one pass', () => {
    const r = tileAccess({ T: 32, pitch: 33, orientation: 'x-first' });
    expect(r.passes).toBe(1);
    expect(r.lanes.map((l) => l.bank)).toEqual(Array.from({ length: 32 }, (_, i) => i));
    expect(r.threads.slice(0, 3).map((t) => t.word)).toEqual([0, 33, 66]);
  });
  it('the row-wise access a[threadIdx.y][threadIdx.x] never conflicts, padded or not', () => {
    expect(tileAccess({ T: 32, pitch: 32, orientation: 'y-first', ty0: 7 }).passes).toBe(1);
    expect(tileAccess({ T: 32, pitch: 33, orientation: 'y-first', ty0: 7 }).passes).toBe(1);
  });
  it('a 16-column array written down a column by 32 distinct threadIdx.x: two banks, 16 each (p. 141)', () => {
    // blockDim.x = 32, so one warp is tx 0..31 with one ty; the array has 16 columns.
    const r = tileAccess({ T: 32, pitch: 16, orientation: 'x-first', ty0: 3 });
    expect(r.passes).toBe(16);
    expect(r.banksUsed).toBe(2);
    expect(r.passes).toBe(strideCase(16).passes);
  });
  it('a 16 × 16 tile with a 16 × 16 block: one warp is two rows of threads, so 4 banks, 8 each; padded, 2 passes', () => {
    const r16 = tileAccess({ T: 16, pitch: 16, orientation: 'x-first' });
    expect(r16.passes).toBe(8);
    expect(r16.banksUsed).toBe(4);
    const r17 = tileAccess({ T: 16, pitch: 17, orientation: 'x-first' });
    expect(r17.passes).toBe(2);
    expect(r17.banksUsed).toBe(31);
  });
  it('a 64-column array, column write by 32 lanes: 32 passes; pitch 65: one', () => {
    expect(tileAccess({ T: 64, pitch: 64, orientation: 'x-first' }).passes).toBe(32);
    expect(tileAccess({ T: 64, pitch: 65, orientation: 'x-first' }).passes).toBe(1);
    expect(tileAccess({ T: 64, pitch: 65, orientation: 'x-first', tx0: 32 }).passes).toBe(1);
  });
  it('padding by more columns: pitch 34 is 2-way again, 35 is free (odd pitch, gcd 1)', () => {
    expect(tileAccess({ T: 32, pitch: 34, orientation: 'x-first' }).passes).toBe(2);
    expect(tileAccess({ T: 32, pitch: 35, orientation: 'x-first' }).passes).toBe(1);
    expect(tileAccess({ T: 32, pitch: 36, orientation: 'x-first' }).passes).toBe(4);
  });
});

describe('same word, and narrow elements (NVIDIA: no conflict inside one 32-bit word, A12)', () => {
  it('every lane on one word: one pass', () => {
    const r = sameWordCase(5);
    expect(r.passes).toBe(1);
    expect(r.banksUsed).toBe(1);
    expect(r.maxLanesPerBank).toBe(32);
    expect(r.sharedWord).toBe(true);
  });
  it('2-byte elements: two lanes per word, 16 banks, one pass', () => {
    const r = halfCase();
    expect(r.passes).toBe(1);
    expect(r.banksUsed).toBe(16);
    expect(r.sharedWord).toBe(true);
  });
  it('two different words in one bank do conflict', () => {
    expect(warpBanks([0, 128], 4).passes).toBe(2);
  });
  it('wider accesses are refused rather than guessed', () => {
    expect(() => warpBanks([0], 8)).toThrow();
  });
});

describe('the same arithmetic with AMD\'s documented bank counts (infra B10) is not AMD\'s cost', () => {
  it('32 banks, 64 lanes, unit stride: lanes i and i + 32 share a bank', () => {
    const r = strideCase(1, 64, 32);
    expect(r.passes).toBe(2);
    expect(r.perBank[0].lanes).toEqual([0, 32]);
    expect(BANK_PRESETS['amd-cdna'].caveat).toMatch(/16 lanes per cycle/);
  });
  it('64 banks, 64 lanes: unit stride is one pass, the 32 × 32 column write is 16', () => {
    expect(strideCase(1, 64, 64).passes).toBe(1);
    expect(tileAccess({ T: 32, pitch: 32, orientation: 'x-first', warpWidth: 64, banks: 64 }).passes).toBe(16);
    expect(tileAccess({ T: 32, pitch: 33, orientation: 'x-first', warpWidth: 64, banks: 64 }).passes).toBe(2);
  });
  it('32 banks, 64 lanes: the padded column write still has two lanes per bank', () => {
    expect(tileAccess({ T: 32, pitch: 33, orientation: 'x-first', warpWidth: 64, banks: 32 }).passes).toBe(2);
  });
  it('Apple has no model', () => {
    expect(BANK_PRESETS.apple.banks).toBeNull();
  });
});
