// The labs' hand-derived fixture blocks (labs/x2/shared/qfixture.h), run through the lessons'
// simulator (quant.ts). Labs X2.1 and X2.2 check their CPU quantisers on these blocks before any
// kernel runs; this test reads the same header, so the lessons and the labs cannot disagree on
// llama.cpp's _ref rules without one of them failing.
import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { fp16Bits, quantizeQ4_0, quantizeQ8_0 } from './quant';

const HEADER = path.resolve(__dirname, '../../../labs/x2/shared/qfixture.h');
const src = fs.readFileSync(HEADER, 'utf8');

/** The numbers inside `name[...] = { ... };`, comments removed, `f` suffixes dropped. */
function array(name: string): number[] {
  const m = new RegExp(`${name}\\[\\d*\\]\\s*=\\s*\\{([^}]*)\\}`).exec(src);
  if (!m) throw new Error(`${name} not found in qfixture.h`);
  return m[1]
    .replace(/\/\/.*$/gm, '')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map((s) => Number(s.replace(/f$/, '')));
}
function scalar(name: string): number {
  const m = new RegExp(`${name}\\s*=\\s*(0x[0-9A-Fa-f]+)`).exec(src);
  if (!m) throw new Error(`${name} not found in qfixture.h`);
  return Number(m[1]);
}
/** kHalfFixtures: {value, bits, "why"} triples; the value may be an expression such as 1.0f / 127.0f. */
function halves(): { value: number; bits: number; why: string }[] {
  const body = /kHalfFixtures\[\]\s*=\s*\{([\s\S]*?)\n\};/.exec(src)![1].replace(/\/\/.*$/gm, '');
  return [...body.matchAll(/\{([^,{}]+),\s*(0x[0-9A-Fa-f]+),\s*"([^"]*)"\}/g)].map((m) => {
    const expr = m[1].trim().replace(/(\d)f\b/g, '$1');
    const value = /^[-\d.e]+\s*\/\s*[-\d.e]+$/.test(expr) ? Math.fround(Number(expr.split('/')[0]) / Number(expr.split('/')[1])) : Number(expr);
    return { value, bits: Number(m[2]), why: m[3] };
  });
}

describe('labs/x2/shared/qfixture.h through quant.ts', () => {
  it('reads every fixture of the header', () => {
    expect(halves().length).toBeGreaterThanOrEqual(12);
    for (const n of ['kQ8Multipliers', 'kQ8Codes', 'kQ4aU', 'kQ4aCodes', 'kQ4bW', 'kQ4bCodes']) expect(array(n)).toHaveLength(32);
  });

  for (const h of halves())
    it(`FP16 rounding: ${h.why}`, () => {
      expect(fp16Bits(h.value)).toBe(h.bits);
    });

  it('Q8_0: d = 1/32 and roundf sends the halves away from zero', () => {
    const b = quantizeQ8_0(array('kQ8Multipliers').map((m) => m / 32));
    expect(b.dBits).toBe(scalar('kQ8ScaleBits'));
    // quant.ts's roundf gives -0 for a small negative value; as an int8 code that is 0.
    expect(b.codes.map((c) => c | 0)).toEqual(array('kQ8Codes'));
    expect(b.bytes.slice(2)).toEqual(array('kQ8Codes').map((c) => c & 0xff));
  });

  it('Q4_0 with a negative extreme: d = +1/8, the other peak clipped to 15', () => {
    const b = quantizeQ4_0(array('kQ4aU').map((u) => u / 8));
    const codes = array('kQ4aCodes');
    expect(b.dBits).toBe(scalar('kQ4aScaleBits'));
    expect(b.codes).toEqual(codes);
    // byte j of qs is code j | code j + 16 << 4
    expect(b.bytes.slice(2)).toEqual(codes.slice(0, 16).map((c, j) => c | (codes[j + 16] << 4)));
  });

  it('Q4_0 with a positive extreme: d = -1/4, a negative scale', () => {
    const b = quantizeQ4_0(array('kQ4bW'));
    expect(b.dBits).toBe(scalar('kQ4bScaleBits'));
    expect(b.codes).toEqual(array('kQ4bCodes'));
  });

  it('Q4_0 all zero: d = -0 (0x8000), every byte 0x88; Q8_0 all zero stores +0', () => {
    const z = new Array(32).fill(0);
    const q4 = quantizeQ4_0(z);
    expect(q4.dBits).toBe(scalar('kQ4ZeroScaleBits'));
    expect(q4.bytes.slice(2)).toEqual(new Array(16).fill(0x88));
    expect(quantizeQ8_0(z).dBits).toBe(0x0000);
  });
});
