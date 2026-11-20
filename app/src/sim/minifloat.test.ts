import { describe, expect, it } from 'vitest';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  BF16, E2M1, E2M3, E3M2, E4M3, E4M3FNUZ, E5M2, FP16, FP32, PRESETS, TOY5, TOY6, UE8M0,
  add, anatomy, bitsOfValue, classify, decode, div, emax, encode, epsilon, excessBias, excessCode, excessTable,
  fields, flipBit, flushDenormals, fromFields, fromTwos, makeFormat, maxFinite, minNormal, minPositive, minSubnormal,
  mul, nextDown, nextUp, round, show, showBits, sub, totalBits, twosComplement, ulp, values,
  type Format, type Mode,
} from './minifloat';

const z = (v: number) => (v === 0 ? 0 : v); // compare values without the sign of zero
const bin = (s: string) => parseInt(s.replace(/\s/g, ''), 2);
const f32bits = (x: number) => {
  const dv = new DataView(new ArrayBuffer(4));
  dv.setFloat32(0, x);
  return dv.getUint32(0);
};
const fromF32bits = (u: number) => {
  const dv = new DataView(new ArrayBuffer(4));
  dv.setUint32(0, u >>> 0);
  return dv.getFloat32(0);
};

// ---------------------------------------------------------------- the book
describe("Fig. A.4: the 5-bit format's three columns, value for value (pp. 587–589)", () => {
  const nozero = makeFormat(2, 2, 'nozero');
  const abrupt = makeFormat(2, 2, 'abrupt');
  const denorm = makeFormat(2, 2, 'denorm');
  // Rows E = 00, 01, 10; within each, M = 00, 01, 10, 11. S = 0 column; S = 1 is the negation.
  const book = {
    nozero: [2 ** -1, 2 ** -1 + 2 ** -3, 2 ** -1 + 2 * 2 ** -3, 2 ** -1 + 3 * 2 ** -3, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5],
    abrupt: [0, 0, 0, 0, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5],
    denorm: [0, 2 ** -2, 2 * 2 ** -2, 3 * 2 ** -2, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5],
  };
  for (const [name, f] of [['nozero', nozero], ['abrupt', abrupt], ['denorm', denorm]] as const)
    it(`${name}: every pattern`, () => {
      for (let E = 0; E < 3; E++)
        for (let M = 0; M < 4; M++) {
          expect(decode(f, fromFields(f, 0, E, M)).value).toBe(book[name][E * 4 + M]);
          expect(z(decode(f, fromFields(f, 1, E, M)).value)).toBe(z(-book[name][E * 4 + M]));
        }
      for (let M = 0; M < 4; M++) expect(decode(f, fromFields(f, 0, 3, M)).cls).toBe('reserved');
    });
  it('the excess-1 exponent codes 00, 01, 10 stand for -1, 0, +1 (p. 586)', () => {
    expect(excessBias(2)).toBe(1);
    expect(anatomy(nozero, bin('0 00 00')).scale).toBe(-1);
    expect(anatomy(nozero, bin('0 01 00')).scale).toBe(0);
    expect(anatomy(nozero, bin('0 10 00')).scale).toBe(1);
  });
  it('the positive values are those of Figs. A.5, A.6 and A.7', () => {
    const pos = (f: Format) => values(f).filter((v) => v > 0);
    expect(pos(nozero)).toEqual([0.5, 0.625, 0.75, 0.875, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5]);
    expect(values(abrupt).filter((v) => v >= 0)).toEqual([0, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5]);
    expect(values(denorm).filter((v) => v >= 0)).toEqual([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5]);
    expect(values(nozero)).not.toContain(0); // observation 3
    expect(values(denorm)).not.toContain(4); // Fig. A.7's 13th star is not a value
  });
  it("the book's denormal example: 00001 is 0.01B × 2^0 = 0.25 (p. 589)", () => {
    expect(decode(denorm, bin('0 00 01')).value).toBe(0.25);
    expect(anatomy(denorm, bin('0 00 01'))).toMatchObject({ hidden: 0, scale: 0, cls: 'subnormal' });
  });
  it('TOY5 is the denorm column', () => {
    expect(values(TOY5)).toEqual(values(denorm));
  });
});

describe('§A.1: the 6-bit format and the formulas (pp. 584–586)', () => {
  it("the book's worked example: 0.5 is 0 010 00", () => {
    expect(encode(TOY6, 0.5)).toBe(bin('001000'));
    expect(showBits(TOY6, encode(TOY6, 0.5))).toBe('0 010 00');
    expect(excessCode(3, -1)).toBe(bin('010'));
  });
  it('the question: 0.75 is 001010; the distractors are 1.0, 3.0 and 0.375', () => {
    expect(encode(TOY6, 0.75)).toBe(bin('001010'));
    expect(decode(TOY6, bin('001100')).value).toBe(1);
    expect(decode(TOY6, bin('010010')).value).toBe(3);
    expect(decode(TOY6, bin('000110')).value).toBe(0.375);
  });
  it('the general formula (-1)^S · 1.M · 2^(E - (2^(n-1) - 1)) for every normal pattern, n = 2 to 8', () => {
    for (let n = 2; n <= 8; n++) {
      const f = makeFormat(n, 3, 'denorm');
      for (let E = 1; E < 2 ** n - 1; E++)
        for (let M = 0; M < 8; M++) expect(decode(f, fromFields(f, 0, E, M)).value).toBe((1 + M / 8) * 2 ** (E - (2 ** (n - 1) - 1)));
    }
  });
  it("the denormal formula 0.M · 2^(-2^(n-1) + 2) equals NVIDIA's 0.M · 2^(1 - bias)", () => {
    for (let n = 2; n <= 8; n++) {
      const f = makeFormat(n, 3, 'denorm');
      for (let M = 1; M < 8; M++) {
        const v = decode(f, fromFields(f, 0, 0, M)).value;
        expect(v).toBe((M / 8) * 2 ** (-(2 ** (n - 1)) + 2));
        expect(v).toBe((M / 8) * 2 ** (1 - f.bias));
      }
    }
  });
  it('the hidden bit: m stored bits carry m + 1 (1.M has one representation per value)', () => {
    for (const f of [TOY5, TOY6, E4M3, FP16]) {
      const pos = values(f).filter((v) => v >= minNormal(f) && v < 2 * minNormal(f));
      expect(pos.length).toBe(2 ** f.m);
    }
  });
});

describe('Fig. A.1: excess codes (p. 585)', () => {
  it('the table for e = 3, row by row', () => {
    const rows = excessTable(3).map((r) => [r.twos.toString(2).padStart(3, '0'), r.value, r.excess.toString(2).padStart(3, '0')]);
    expect(rows).toEqual([
      ['101', -3, '000'], ['110', -2, '001'], ['111', -1, '010'], ['000', 0, '011'],
      ['001', 1, '100'], ['010', 2, '101'], ['011', 3, '110'], ['100', null, '111'],
    ]);
  });
  it('excess codes compare like the values as unsigned integers; two\'s complement does not', () => {
    for (let e = 2; e <= 5; e++) {
      const rows = excessTable(e).filter((r) => r.value !== null);
      for (const a of rows)
        for (const b of rows) expect(a.excess < b.excess).toBe(a.value! < b.value!);
      // two's complement puts every negative above every positive
      const neg = rows.find((r) => r.value === -1)!;
      const pos = rows.find((r) => r.value === 1)!;
      expect(neg.twos > pos.twos).toBe(true);
      expect(fromTwos(e, twosComplement(e, -1))).toBe(-1);
      expect(excessBias(e)).toBe(2 ** (e - 1) - 1);
    }
    expect(excessCode(3, -2)).toBe(1); // the book's 001 < 100 for -2 < 1
    expect(excessCode(3, 1)).toBe(4);
  });
});

describe('§A.2: observations, Exercises 1 and 2', () => {
  it('Exercise 1: 2-bit exponent, 3-bit mantissa, no-zero: 24 positive values, spacing halved', () => {
    const f = makeFormat(2, 3, 'nozero');
    const pos = values(f).filter((v) => v > 0);
    expect(pos.length).toBe(24);
    expect(pos[0]).toBe(0.5);
    expect(pos[1] - pos[0]).toBe(1 / 16);
    expect(pos[pos.length - 1]).toBe(3.75);
    // the gap next to 0 is 2^m times the next spacing (observation 5)
    expect(pos[0] / (pos[1] - pos[0])).toBe(8);
  });
  it('Exercise 2: 3-bit exponent, 2-bit mantissa: seven intervals from 1/8, largest 14', () => {
    const f = makeFormat(3, 2, 'nozero');
    const pos = values(f).filter((v) => v > 0);
    expect(pos.length).toBe(28);
    expect(pos[0]).toBe(0.125);
    expect(maxFinite(f)).toBe(14);
    expect(new Set(pos.map((v) => Math.floor(Math.log2(v)))).size).toBe(7);
  });
  it('observation 5: the gap next to 0 is 2^m times the next interval\'s spacing (no-zero); abrupt doubles it', () => {
    for (let m = 0; m <= 4; m++) {
      const nz = makeFormat(2, m, 'nozero');
      expect(minNormal(nz) / ulp(nz, minNormal(nz))).toBe(2 ** m);
      const ab = makeFormat(2, m, 'abrupt');
      expect(values(ab).filter((v) => v > 0)[0]).toBe(2 * values(nz).filter((v) => v > 0)[0]);
    }
  });
  it('the probe 0.2 under the three conventions (the prediction): 0.5, 0 and 0.25', () => {
    expect(round(makeFormat(2, 2, 'nozero'), 0.2)).toBe(0.5);
    expect(round(makeFormat(2, 2, 'abrupt'), 0.2)).toBe(0);
    expect(round(makeFormat(2, 2, 'denorm'), 0.2)).toBe(0.25);
    expect(Math.abs(round(makeFormat(2, 2, 'nozero'), 0.2) - 0.2)).toBeCloseTo(0.3, 12);
    expect(Math.abs(round(makeFormat(2, 2, 'denorm'), 0.2) - 0.2)).toBeCloseTo(0.05, 12);
  });
  it("the probe 0.2 rounded toward zero: the no-zero format has nothing smaller, the others give 0", () => {
    expect(round(makeFormat(2, 2, 'nozero'), 0.2, 'rtz')).toBe(0.5);
    expect(round(makeFormat(2, 2, 'abrupt'), 0.2, 'rtz')).toBe(0);
    expect(round(makeFormat(2, 2, 'denorm'), 0.2, 'rtz')).toBe(0);
    expect(round(TOY5, 0.6, 'rtz')).toBe(0.5);
    expect(round(TOY5, 0.6)).toBe(0.5);
    expect(round(TOY5, 0.65)).toBe(0.75);
    expect(round(TOY5, 3.9)).toBe(3.5); // the book's toy formats have no infinity: they saturate
  });
  it('makeFormat covers the explorer\'s whole range (e 1 to 4, m 0 to 4, three conventions)', () => {
    for (let e = 1; e <= 4; e++)
      for (let m = 0; m <= 4; m++)
        for (const c of ['nozero', 'abrupt', 'denorm'] as const) {
          const f = makeFormat(e, m, c);
          const v = values(f);
          expect(v.length).toBeGreaterThan(0);
          for (const x of v) expect(round(f, x)).toBe(x);
          for (let i = 0; i + 1 < v.length; i++) expect(v[i]).toBeLessThan(v[i + 1]);
        }
  });
});

describe('precision is a constant fraction; range comes from the exponent', () => {
  it('ulp at powers of two is 2^(k - m), so the relative spacing is 2^-m across the normal range', () => {
    for (const f of [FP32, FP16, BF16, E4M3, E5M2]) {
      for (let k = Math.log2(minNormal(f)); k <= emax(f); k++) expect(ulp(f, 2 ** k) / 2 ** k).toBe(epsilon(f));
    }
    expect(ulp(FP16, 1)).toBe(2 ** -10);
    expect(ulp(BF16, 1)).toBe(2 ** -7);
    expect(ulp(FP32, 1)).toBe(2 ** -23);
    expect(ulp(FP16, 2 ** -20)).toBe(2 ** -24); // subnormal spacing
  });
  it('the bridge: 70,000 overflows FP16 but not BF16; 1.001 survives FP16 but not BF16', () => {
    expect(round(FP16, 70000)).toBe(Infinity);
    expect(round(BF16, 70000)).toBe(70144);
    expect(round(FP32, 70000)).toBe(70000);
    expect(round(BF16, 1.001)).toBe(1);
    expect(round(FP16, 1.001)).toBe(1.0009765625);
    expect(show(round(FP16, 1.001))).toBe('1.0009766');
    expect(round(FP32, 1.001)).toBe(Math.fround(1.001));
    expect(round(FP32, 1.001)).not.toBe(1);
    // FP16 overflows from 65520 (halfway between 65504 and the 65536 it cannot hold)
    expect(round(FP16, 65519.99)).toBe(65504);
    expect(round(FP16, 65520)).toBe(Infinity);
  });
  it('the smallest positive integers each format cannot hold: 2049, 257, 16,777,217', () => {
    const firstMissing = (f: Format) => {
      let n = 1;
      while (round(f, n) === n) n++;
      return n;
    };
    expect(firstMissing(FP16)).toBe(2049);
    expect(firstMissing(BF16)).toBe(257);
    expect(round(FP32, 16777216)).toBe(16777216);
    expect(round(FP32, 16777217)).toBe(16777216);
    for (let n = 16777000; n <= 16777216; n++) expect(round(FP32, n)).toBe(n);
  });
  it('largest finite FP16 is (2 - 2^-10) × 2^15 = 65504', () => {
    expect(maxFinite(FP16)).toBe(65504);
    expect(maxFinite(FP16)).toBe((2 - 2 ** -10) * 2 ** 15);
    expect(fields(FP16, encode(FP16, 65504)).E).toBe(30);
  });
});

// ---------------------------------------------------------------- vendor and standards tables
describe('NVIDIA Table 51 (appA-facts A2)', () => {
  const rows: [Format, number, number, number, number][] = [
    [BF16, 3.3895313892515355e38, 2 ** -126, 2 ** -133, 2 ** -7],
    [FP16, 65504, 2 ** -14, 2 ** -24, 2 ** -10],
    [FP32, 3.4028234663852886e38, 2 ** -126, 2 ** -149, 2 ** -23],
  ];
  for (const [f, max, minN, minS, eps] of rows)
    it(f.name, () => {
      expect(maxFinite(f)).toBe(max);
      expect(minNormal(f)).toBe(minN);
      expect(minSubnormal(f)).toBe(minS);
      expect(epsilon(f)).toBe(eps);
    });
  it('the table\'s rounded figures', () => {
    expect(maxFinite(BF16).toPrecision(3)).toBe('3.39e+38');
    expect(maxFinite(FP32).toPrecision(3)).toBe('3.40e+38');
    expect(minNormal(FP16).toExponential(1)).toBe('6.1e-5');
    expect(minNormal(FP32).toPrecision(3)).toBe('1.18e-38');
  });
  it('field widths: FP32 1-8-23, FP16 1-5-10, BF16 1-8-7; BF16 is the top half of FP32', () => {
    expect([FP32.e, FP32.m, FP32.bias, totalBits(FP32)]).toEqual([8, 23, 127, 32]);
    expect([FP16.e, FP16.m, FP16.bias, totalBits(FP16)]).toEqual([5, 10, 15, 16]);
    expect([BF16.e, BF16.m, BF16.bias, totalBits(BF16)]).toEqual([8, 7, 127, 16]);
  });
});

describe('OFP8 r1.1 Tables 1 and 2 (appA-facts A3)', () => {
  const p = (s: string) => bin(s.replace(/[S.]/g, (c) => (c === 'S' ? '0' : '')));
  it('E4M3', () => {
    expect(E4M3.bias).toBe(7);
    expect(emax(E4M3)).toBe(8);
    expect(Math.log2(minNormal(E4M3))).toBe(-6);
    expect(decode(E4M3, p('S.1111.111')).cls).toBe('nan');
    expect(decode(E4M3, p('S.1111.111') | 0x80).cls).toBe('nan');
    expect(values(E4M3)).not.toContain(Infinity);
    expect(decode(E4M3, p('S.0000.000')).value).toBe(0);
    expect(decode(E4M3, p('S.1111.110')).value).toBe(448);
    expect(decode(E4M3, p('S.0001.000')).value).toBe(2 ** -6);
    expect(decode(E4M3, p('S.0000.111')).value).toBe(0.875 * 2 ** -6);
    expect(decode(E4M3, p('S.0000.001')).value).toBe(2 ** -9);
    expect(Math.round(Math.log2(maxFinite(E4M3) / minPositive(E4M3)))).toBe(18);
    // exactly two NaN patterns, no infinity
    let nans = 0;
    for (let b = 0; b < 256; b++) {
      const c = decode(E4M3, b).cls;
      if (c === 'nan') nans++;
      expect(c).not.toBe('inf');
    }
    expect(nans).toBe(2);
  });
  it('E5M2', () => {
    expect(E5M2.bias).toBe(15);
    expect(emax(E5M2)).toBe(15);
    expect(decode(E5M2, p('S.11111.00')).value).toBe(Infinity);
    for (const m of ['01', '10', '11']) expect(decode(E5M2, p('S.11111.' + m)).cls).toBe('nan');
    expect(decode(E5M2, p('S.00000.00')).value).toBe(0);
    expect(decode(E5M2, p('S.11110.11')).value).toBe(57344);
    expect(decode(E5M2, p('S.00001.00')).value).toBe(2 ** -14);
    expect(decode(E5M2, p('S.00000.11')).value).toBe(0.75 * 2 ** -14);
    expect(decode(E5M2, p('S.00000.01')).value).toBe(2 ** -16);
    expect(Math.round(Math.log2(maxFinite(E5M2) / minPositive(E5M2)))).toBe(32);
  });
  it('Appendix A: E4M3 inputs up to 464 round to 448; above, overflow; E5M2 61,440 overflows', () => {
    expect(round(E4M3, 449)).toBe(448);
    expect(round(E4M3, 464)).toBe(448);
    expect(round(E4M3, 464.0001)).toBeNaN();
    expect(round(E4M3, 464.0001, 'rne', 'saturate')).toBe(448);
    expect(classify(E4M3, 1000).status).toBe('nan');
    expect(classify(E4M3, 1000, { overflow: 'saturate' }).status).toBe('saturated');
    expect(classify(E4M3, 450).status).toBe('normal'); // plain rounding, not a clamp
    expect(round(E5M2, 61439.99)).toBe(57344);
    expect(round(E5M2, 61440)).toBe(Infinity);
    expect(round(E5M2, -61440)).toBe(-Infinity);
    expect(round(E5M2, 61440, 'rne', 'saturate')).toBe(57344);
    expect(classify(E5M2, 61440).status).toBe('inf');
  });
});

describe('OCP MX v1.1 Tables 4, 5 and 7 (appA-facts A4)', () => {
  it('FP6 E2M3: bias 1, max ±7.5, min subnormal ±0.125, no Inf or NaN', () => {
    expect(E2M3.bias).toBe(1);
    expect(maxFinite(E2M3)).toBe(7.5);
    expect(minSubnormal(E2M3)).toBe(0.125);
    for (let b = 0; b < 64; b++) expect(Number.isFinite(decode(E2M3, b).value)).toBe(true);
  });
  it('FP6 E3M2: bias 3, max ±28.0, min subnormal ±0.0625, no Inf or NaN', () => {
    expect(E3M2.bias).toBe(3);
    expect(maxFinite(E3M2)).toBe(28);
    expect(minSubnormal(E3M2)).toBe(0.0625);
    for (let b = 0; b < 64; b++) expect(Number.isFinite(decode(E3M2, b).value)).toBe(true);
  });
  it('FP4 E2M1: max ±6.0, min normal ±1.0, subnormal ±0.5; magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6', () => {
    expect(maxFinite(E2M1)).toBe(6);
    expect(minNormal(E2M1)).toBe(1);
    expect(minSubnormal(E2M1)).toBe(0.5);
    expect(values(E2M1).filter((v) => v >= 0)).toEqual([0, 0.5, 1, 1.5, 2, 3, 4, 6]);
    expect(round(E2M1, 7)).toBe(6); // no infinity: saturates
    expect(classify(E2M1, 7).status).toBe('saturated');
    expect(encode(E2M1, NaN)).toBe(-1); // no NaN encoding
  });
  it('UE8M0: bias 127, exponents -127 to 127, NaN 11111111, no zero, no infinity, unsigned', () => {
    expect(UE8M0.signed).toBe(false);
    expect(totalBits(UE8M0)).toBe(8);
    expect(decode(UE8M0, 0).value).toBe(2 ** -127);
    expect(decode(UE8M0, 254).value).toBe(2 ** 127);
    expect(decode(UE8M0, 255).cls).toBe('nan');
    expect(values(UE8M0)).not.toContain(0);
    expect(values(UE8M0).length).toBe(255);
    expect(encode(UE8M0, 1)).toBe(127);
    expect(encode(UE8M0, 0)).toBe(0); // nothing smaller than 2^-127
    expect(encode(UE8M0, -2)).toBe(255); // unsigned: a negative has no value
    expect(encode(UE8M0, 5, 'down')).toBe(129); // floor, as MX computes scales (A4)
    expect(encode(UE8M0, 5, 'up')).toBe(130); // up, as cuBLAS rounds them (A5)
  });
  it('E4M3FNUZ (AMD): bias 8, max 240, NaN 0x80, no -0, no infinity', () => {
    expect(maxFinite(E4M3FNUZ)).toBe(240);
    expect(decode(E4M3FNUZ, 0x80).cls).toBe('nan');
    expect(encode(E4M3FNUZ, -0)).toBe(0);
    expect(encode(E4M3FNUZ, NaN)).toBe(0x80);
    expect(Object.is(round(E4M3FNUZ, -1e-9), 0)).toBe(true);
    expect(round(E4M3FNUZ, 1000)).toBeNaN();
  });
});

// ---------------------------------------------------------------- FP32 against the platform
describe('FP32 against Math.fround and the platform float32', () => {
  // A deterministic generator so failures reproduce.
  let seed = 12345;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed / 2147483648;
  };
  it('random doubles across the whole range, including subnormals and overflow', () => {
    for (let i = 0; i < 40000; i++) {
      const k = Math.floor(rand() * 300) - 160;
      const x = (rand() < 0.5 ? -1 : 1) * (1 + rand()) * 2 ** k;
      expect(round(FP32, x)).toBe(Math.fround(x));
      expect(encode(FP32, x)).toBe(f32bits(x));
    }
  });
  it('exact ties between neighbouring float32 values, normal and subnormal', () => {
    for (let i = 0; i < 20000; i++) {
      const u = Math.floor(rand() * 0x7f7fffff);
      const a = fromF32bits(u);
      const b = fromF32bits(u + 1);
      const mid = (a + b) / 2;
      expect(round(FP32, mid)).toBe(Math.fround(mid));
      expect(round(FP32, -mid)).toBe(Math.fround(-mid));
    }
    for (let u = 0; u < 3000; u++) {
      const mid = (fromF32bits(u) + fromF32bits(u + 1)) / 2;
      expect(round(FP32, mid)).toBe(Math.fround(mid));
    }
  });
  it('directed modes bracket the value and agree with RNE when exact', () => {
    for (let i = 0; i < 5000; i++) {
      const x = (rand() - 0.5) * 2 ** (Math.floor(rand() * 60) - 30);
      const d = round(FP32, x, 'down');
      const u = round(FP32, x, 'up');
      expect(d).toBeLessThanOrEqual(x);
      expect(u).toBeGreaterThanOrEqual(x);
      if (d === u) expect(d).toBe(x);
      else expect(nextUp(FP32, d)).toBe(u);
      expect(Math.abs(round(FP32, x, 'rtz'))).toBeLessThanOrEqual(Math.abs(x));
    }
  });
  it('specials: infinity, NaN, signed zero', () => {
    expect(encode(FP32, Infinity)).toBe(0x7f800000);
    expect(encode(FP32, -Infinity)).toBe(0xff800000);
    expect(encode(FP32, NaN)).toBe(0x7fc00000);
    expect(encode(FP32, -0)).toBe(0x80000000);
    expect(decode(FP32, 0x7fc00000).cls).toBe('qnan');
    expect(decode(FP32, 0x7f800001).cls).toBe('snan');
    expect(round(FP32, 1e39, 'rtz')).toBe(maxFinite(FP32));
    expect(round(FP32, 1e39, 'up')).toBe(Infinity);
    expect(round(FP32, -1e39, 'up')).toBe(-maxFinite(FP32));
    expect(round(FP32, -1e39, 'down')).toBe(-Infinity);
  });
  it('the FP32 NaN count is 2^24 - 2: 2^22 quiet and 2^22 - 1 signaling per sign (E1)', () => {
    let q = 0;
    let s = 0;
    for (const sign of [0, 0x80000000])
      for (let M = 0; M < 2 ** 23; M++) {
        const c = decode(FP32, (sign + 0x7f800000 + M) >>> 0).cls;
        if (c === 'qnan') q++;
        else if (c === 'snan') s++;
      }
    expect(q + s).toBe(2 ** 24 - 2);
    expect(q).toBe(2 * 2 ** 22);
    expect(s).toBe(2 * (2 ** 22 - 1));
    // no NaN anywhere else: a stride through the other exponent codes
    for (let u = 0; u < 2 ** 32; u += 0x00fedcb7) if (fields(FP32, u).E !== 255) expect(decode(FP32, u).cls).not.toMatch(/nan/);
  });
});

describe('BF16 from FP32 bits (the q-bf16-top question; appA-facts D3)', () => {
  let seed = 99;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    return seed / 2147483648;
  };
  it('keeping the top 16 bits is rounding toward zero, for every finite value', () => {
    for (let i = 0; i < 20000; i++) {
      const u = Math.floor(rand() * 2 ** 32) >>> 0;
      const x = fromF32bits(u);
      if (!Number.isFinite(x)) continue;
      expect(decode(BF16, u >>> 16).value).toBe(round(BF16, x, 'rtz'));
      const err = Math.abs(decode(BF16, u >>> 16).value - x);
      expect(err).toBeLessThan(ulp(BF16, x));
    }
  });
  it("llama.cpp's (u + 0x7fff + ((u >> 16) & 1)) >> 16 is round to nearest even", () => {
    for (let i = 0; i < 20000; i++) {
      const u = Math.floor(rand() * 2 ** 32) >>> 0;
      const x = fromF32bits(u);
      if (!Number.isFinite(x) || Math.abs(x) >= maxFinite(BF16) + ulp(BF16, maxFinite(BF16)) / 2) continue;
      const r = Math.floor((u + 0x7fff + ((u >>> 16) & 1)) / 65536) % 65536;
      expect(r).toBe(encode(BF16, x));
    }
  });
  it('truncating a NaN can make an infinity: why llama.cpp forces the quiet bit', () => {
    expect(decode(FP32, 0x7f800001).cls).toBe('snan');
    expect(decode(BF16, 0x7f800001 >>> 16).cls).toBe('inf');
  });
});

// ---------------------------------------------------------------- walking and round trips
describe('nextUp and nextDown walk every value', () => {
  const formats = [TOY5, TOY6, makeFormat(2, 2, 'nozero'), makeFormat(2, 2, 'abrupt'), makeFormat(1, 2, 'denorm'), makeFormat(3, 0, 'denorm'),
    E4M3, E5M2, E2M1, E2M3, E3M2, E4M3FNUZ, UE8M0, FP16, BF16];
  for (const f of formats)
    it(f.name + ' ' + f.subnormals, () => {
      const v = values(f);
      let x = v[0];
      const seen = [x];
      for (;;) {
        const n = nextUp(f, x);
        if (n === null || n === Infinity) break;
        seen.push(n);
        x = n;
      }
      expect(seen.map(z)).toEqual(v);
      let y = v[v.length - 1];
      const back = [y];
      for (;;) {
        const n = nextDown(f, y);
        if (n === null || n === -Infinity) break;
        back.push(n);
        y = n;
      }
      expect(back.map(z).reverse()).toEqual(v);
    });
  it('IEEE formats step to infinity past the top; others have nowhere to go', () => {
    expect(nextUp(FP16, 65504)).toBe(Infinity);
    expect(nextDown(FP16, -65504)).toBe(-Infinity);
    expect(nextUp(E4M3, 448)).toBeNull();
    expect(nextUp(FP16, Infinity)).toBeNull();
    expect(nextDown(FP16, Infinity)).toBe(65504);
    expect(nextUp(FP32, 0)).toBe(2 ** -149);
    expect(nextUp(FP32, -0)).toBe(2 ** -149);
    expect(nextUp(makeFormat(2, 2, 'nozero'), -0.5)).toBe(0.5);
    expect(nextUp(FP16, 1)).toBe(1 + 2 ** -10);
    expect(nextDown(FP16, 1)).toBe(1 - 2 ** -11);
    expect(nextUp(FP16, 1.0001)).toBe(1 + 2 ** -10); // not a value: the next value above it
  });
  it('decode then encode gives the pattern back (canonical patterns), for every code of every small format', () => {
    for (const f of PRESETS.filter((p) => totalBits(p) <= 16)) {
      for (let b = 0; b < 2 ** totalBits(f); b++) {
        const d = decode(f, b);
        if (d.cls === 'reserved') continue;
        if (Number.isNaN(d.value)) {
          expect(Number.isNaN(decode(f, encode(f, d.value)).value)).toBe(true);
          continue;
        }
        expect(bitsOfValue(f, d.value)).toBe(b);
        expect(encode(f, d.value)).toBe(b);
      }
    }
  });
});

describe('classify and flush-to-zero', () => {
  it('statuses', () => {
    expect(classify(FP32, 1e-40).status).toBe('subnormal');
    expect(classify(FP32, 1e-40, { flushDenormals: true })).toMatchObject({ status: 'flushed', value: 0, bits: 0 });
    expect(Object.is(classify(FP32, -1e-40, { flushDenormals: true }).value, -0)).toBe(true);
    expect(classify(FP32, 1e-50).status).toBe('zero');
    expect(classify(FP16, 70000).status).toBe('inf');
    expect(classify(FP16, 70000, { overflow: 'saturate' }).status).toBe('saturated');
    expect(classify(FP16, 1).status).toBe('normal');
    expect(classify(FP16, 1.001)).toMatchObject({ value: 1.0009765625, exact: false });
    expect(classify(FP16, 1.001).error).toBeCloseTo(1.0009765625 - 1.001, 15);
    expect(classify(FP16, 6e-8).status).toBe('subnormal');
    expect(classify(E4M3, 0.001).status).toBe('subnormal');
    expect(classify(E2M1, 100).status).toBe('saturated');
    expect(classify(FP32, NaN).status).toBe('nan');
  });
  it('flushDenormals on bits', () => {
    expect(flushDenormals(FP32, 1)).toBe(0);
    expect(flushDenormals(FP32, 0x80000001)).toBe(0x80000000);
    expect(flushDenormals(FP32, 0x3f800000)).toBe(0x3f800000);
    expect(flushDenormals(E4M3FNUZ, 0x81)).toBe(0); // no -0 in FNUZ
  });
  it('bits helpers', () => {
    expect(flipBit(0, 31)).toBe(0x80000000);
    expect(flipBit(0x80000000, 31)).toBe(0);
    expect(flipBit(0, 3)).toBe(8);
    expect(fields(FP32, 0xbf800000)).toMatchObject({ s: 1, E: 127, M: 0, eBits: '01111111' });
    expect(showBits(UE8M0, 127)).toBe('01111111');
  });
  it('show', () => {
    expect(show(65504)).toBe('65504');
    expect(show(-0)).toBe('−0');
    expect(show(2 ** -149)).toBe('1.4012985e-45');
    expect(show(maxFinite(FP32))).toBe('3.4028235e+38');
    expect(show(0.1)).toBe('0.1');
    expect(show(Infinity)).toBe('+inf');
  });
});

// ---------------------------------------------------------------- arithmetic against exact rationals
// The reference never uses the model's rounding: it scales every value to an exact BigInt and
// picks the neighbour by comparing exact rationals.
const S = 64n;
const big = (v: number) => {
  // v is dyadic with exponent >= -64 and fits in 53 bits: v * 2^64 is an integer
  const k = 64;
  let x = v * 2 ** 32;
  x *= 2 ** (k - 32);
  return BigInt(x);
};
function refRound(f: Format, num: bigint, den: bigint, mode: Mode): number {
  // r = num / (den * 2^S), den > 0, where big(v) = v * 2^S
  if (num === 0n) return 0;
  const neg = num < 0n;
  const an = neg ? -num : num;
  const mags = values(f).filter((v) => v >= 0);
  const max = mags[mags.length - 1];
  const sentinel = max + ulp(f, max); // the value the next code would have had
  const all = [...mags, sentinel];
  const cmp = (v: number) => {
    const lhs = big(v) * den;
    return lhs < an ? -1 : lhs > an ? 1 : 0;
  };
  let j = 0;
  while (j + 1 < all.length && cmp(all[j + 1]) <= 0) j++;
  const exact = cmp(all[j]) === 0;
  const dir = mode === 'rne' ? 'nearest' : mode === 'rtz' ? 'zero' : (mode === 'up') !== neg ? 'away' : 'zero';
  let pick: number;
  if (exact) pick = j;
  else if (j === all.length - 1) pick = dir === 'zero' ? j - 1 : j; // beyond the sentinel
  else if (dir === 'zero') pick = j;
  else if (dir === 'away') pick = j + 1;
  else {
    // compare r - lo with hi - r exactly
    const d1 = an - big(all[j]) * den;
    const d2 = big(all[j + 1]) * den - an;
    pick = d1 < d2 ? j : d2 < d1 ? j + 1 : j % 2 === 0 ? j : j + 1;
  }
  let mag = all[pick];
  if (mag === sentinel) {
    if (dir === 'zero' || f.specials === 'none' || f.specials === 'reserved') mag = max;
    else return f.specials === 'ieee' ? (neg ? -Infinity : Infinity) : NaN;
  }
  return neg ? -mag : mag;
}
describe('add, sub, mul and div are correctly rounded (all pairs, all modes, exact rationals)', () => {
  const modes: Mode[] = ['rne', 'rtz', 'up', 'down'];
  for (const f of [TOY5, E2M1, E4M3]) {
    const vs = values(f);
    it(`${f.name}: add and sub`, () => {
      for (const mode of modes)
        for (const a of vs)
          for (const b of vs) {
            expect(z(add(f, a, b, mode))).toBe(z(refRound(f, big(a) + big(b), 1n, mode)));
            expect(z(sub(f, a, b, mode))).toBe(z(refRound(f, big(a) - big(b), 1n, mode)));
          }
    });
    it(`${f.name}: mul and div`, () => {
      for (const mode of modes)
        for (const a of vs)
          for (const b of vs) {
            expect(z(mul(f, a, b, mode))).toBe(z(refRound(f, big(a) * big(b), 1n << S, mode)));
            if (b !== 0) {
              const nb = big(b);
              const num = big(a) * (1n << S) * (nb < 0n ? -1n : 1n);
              expect(z(div(f, a, b, mode))).toBe(z(refRound(f, num, nb < 0n ? -nb : nb, mode)));
            }
          }
    });
  }
  it('FP32 add with a huge exponent gap, in directed modes (where rounding the binary64 sum would fail)', () => {
    expect(add(FP32, 1, -(2 ** -100), 'rtz')).toBe(1 - 2 ** -24);
    expect(add(FP32, 1, -(2 ** -100), 'rne')).toBe(1);
    expect(add(FP32, 1, 2 ** -100, 'up')).toBe(1 + 2 ** -23);
    expect(add(FP32, 1, 2 ** -100, 'down')).toBe(1);
    expect(add(BF16, -1, 2 ** -100, 'down')).toBe(-1);
    expect(add(BF16, -1, 2 ** -100, 'up')).toBe(-1 + 2 ** -8);
  });
  it('zero signs: x + (-x) is +0, or -0 rounding down; -0 + -0 is -0', () => {
    expect(Object.is(add(FP32, 3, -3), 0)).toBe(true);
    expect(Object.is(add(FP32, 3, -3, 'down'), -0)).toBe(true);
    expect(Object.is(add(FP32, -0, -0), -0)).toBe(true);
    expect(Object.is(add(FP32, 0, -0), 0)).toBe(true);
    expect(Object.is(mul(FP32, -1, 0), -0)).toBe(true);
  });
  it("the book's alignment example: 1.00B × 2^-2 + 1.00B × 2^1 is 2.25, which TOY5 rounds to 2.0 (0.5 ULP) (p. 592)", () => {
    expect(add(TOY5, 0.25, 2)).toBe(2);
    expect(Math.abs(add(TOY5, 0.25, 2) - 2.25) / ulp(TOY5, 2)).toBe(0.5);
    expect(add(TOY5, 0.25, 2, 'up')).toBe(2.5);
  });
  it("the book's summation example: sequential 2.0, pairwise 2.5 (p. 593)", () => {
    const seq = add(TOY5, add(TOY5, add(TOY5, 1, 1), 0.25), 0.25);
    const pair = add(TOY5, add(TOY5, 1, 1), add(TOY5, 0.25, 0.25));
    expect(seq).toBe(2);
    expect(pair).toBe(2.5);
  });
  it('infinities and NaN', () => {
    expect(add(FP16, Infinity, -Infinity)).toBeNaN();
    expect(div(FP16, 1, 0)).toBe(Infinity);
    expect(div(E5M2, -1, 0)).toBe(-Infinity);
    expect(div(FP32, 0, 0)).toBeNaN();
    expect(mul(FP16, 300, 300)).toBe(Infinity);
    expect(mul(E4M3, 300, 300, 'rne', 'saturate')).toBe(448);
  });
});

// ---------------------------------------------------------------- the lab's C++ conversions, when they exist
const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '../../..');
const header = path.join(repo, 'labs/appA/lab1_sumerror/fpformat.h');
const checkSrc = path.join(repo, 'labs/appA/snippets/check/fpformat_test.cpp');
const compiler = ['c++', 'clang++', 'g++'].find((c) => {
  try {
    execFileSync(c, ['--version'], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
});
describe.skipIf(!fs.existsSync(header) || !fs.existsSync(checkSrc) || !compiler)("the lab's fpformat.h rounds and adds bit for bit like this model", () => {
  it('FP32, FP16 and BF16, to nearest and toward zero: rounding and addition', () => {
    const exe = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'fpformat-')), 'fpformat_test');
    execFileSync(compiler!, ['-std=c++17', '-O1', checkSrc, '-o', exe], { stdio: 'pipe' });
    const out = execFileSync(exe, { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
    const fmt: Record<string, Format> = { fp32: FP32, fp16: FP16, bf16: BF16 };
    const dv = new DataView(new ArrayBuffer(8));
    const fromHex64 = (h: string) => {
      dv.setBigUint64(0, BigInt('0x' + h));
      return dv.getFloat64(0);
    };
    let rounds = 0;
    let adds = 0;
    const bad: string[] = [];
    for (const line of out.trim().split('\n')) {
      const w = line.trim().split(/\s+/);
      const f = fmt[w[1]];
      const mode = w[2] as Mode;
      if (w[0] === 'r') {
        const want = encode(f, fromHex64(w[3]), mode) >>> 0;
        if (parseInt(w[4], 16) >>> 0 !== want) bad.push(`${line} (model ${want.toString(16)})`);
        rounds++;
      } else if (w[0] === 'a') {
        const a = decode(f, parseInt(w[3], 16)).value;
        const b = decode(f, parseInt(w[4], 16)).value;
        const want = bitsOfValue(f, add(f, a, b, mode)) >>> 0;
        if (parseInt(w[5], 16) >>> 0 !== want) bad.push(`${line} (model ${want.toString(16)})`);
        adds++;
      }
    }
    expect(bad.slice(0, 10)).toEqual([]);
    expect(rounds).toBeGreaterThan(30000);
    expect(adds).toBeGreaterThan(15000);
  });
});
