import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  ALIGN_PRESETS,
  BF16_PRESETS,
  CLASSIFIER_FORMATS,
  alignTrace,
  bf16FromBits,
  bitsFromF32,
  classifierPresets,
  exactDecimal,
  f32FromBits,
  figA8Agrees,
  figA8Row,
  fmaxf,
  parseOperand,
  patternCounts,
  runningSum,
  specialTable,
  type Step,
} from './fpadd';
import { BF16, E2M1, E4M3, E5M2, FP16, FP32, TOY5, add, decode, encode, maxFinite, round, values, type Format, type Mode } from './minifloat';

const MODES: Mode[] = ['rne', 'rtz', 'up', 'down'];
const same = (x: number, y: number) => Object.is(x, y) || (Number.isNaN(x) && Number.isNaN(y));
const last = <K extends Step['kind']>(steps: Step[], kind: K) => steps.filter((s) => s.kind === kind).pop() as Extract<Step, { kind: K }>;

// A small seeded generator, so a failure can be reproduced.
function rng(seed: number) {
  let h = seed >>> 0;
  return () => {
    h = (h + 0x6d2b79f5) >>> 0;
    let t = Math.imul(h ^ (h >>> 15), h | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/** A random finite value of the format, from a random bit pattern (subnormals and wide exponent gaps included). */
function randomValue(f: Format, r: () => number): number {
  const n = 1 + f.e + f.m;
  for (;;) {
    const bits = Math.floor(r() * 2 ** n);
    const d = decode(f, bits);
    if (Number.isFinite(d.value)) return d.value;
  }
}

describe("the book's alignment example (§A.4, p. 592): 2.0 + 0.25 in the 5-bit format", () => {
  const t = alignTrace(TOY5, 2, 0.25, 'rne');
  it('compares exponents 1 and -2 and shifts the smaller operand right 3 places', () => {
    const c = t.steps[0];
    expect(c.kind).toBe('compare');
    if (c.kind !== 'compare') return;
    expect([c.big.exp, c.big.sig, c.small.exp, c.small.sig, c.diff]).toEqual([1, '100', -2, '100', 3]);
    expect(c.small.stored).toBe('subnormal'); // stored as S=0, E=00, M=01
    expect(t.steps.filter((s) => s.kind === 'shift').map((s) => s.kind === 'shift' && s.window)).toEqual(['010', '001', '000']);
  });
  it('the ideal sum 1.001 × 2^1 = 2.25 needs 3 mantissa bits; the neighbours are 2.0 and 2.5', () => {
    const a = last(t.steps, 'add');
    expect([a.window, a.tail, a.exact]).toEqual(['100', '100', 2.25]);
    const r = last(t.steps, 'round');
    expect([r.below, r.above, r.g, r.r, r.s]).toEqual([2, 2.5, 1, 0, 0]);
  });
  it('the error either way is 0.5 ULP, and nearest-even breaks the tie to 2.0', () => {
    const r = last(t.steps, 'round');
    expect(r.reason).toBe('tie-even');
    expect(t.result).toBe(2);
    expect(t.errorUlps).toBe(0.5);
    expect(alignTrace(TOY5, 2, 0.25, 'up').result).toBe(2.5);
    expect(alignTrace(TOY5, 2, 0.25, 'up').errorUlps).toBe(0.5);
  });
});

describe('alignTrace agrees with minifloat.add', () => {
  it('for every pair of values of the 5-bit format, in all four modes', () => {
    const vs = values(TOY5);
    const signed = [...vs, -0];
    let n = 0;
    for (const mode of MODES)
      for (const a of signed)
        for (const b of signed) {
          const t = alignTrace(TOY5, a, b, mode);
          expect(same(t.result, add(TOY5, a, b, mode)), `${a} + ${b} (${mode})`).toBe(true);
          n++;
        }
    expect(n).toBe(4 * (vs.length + 1) ** 2);
  });
  for (const f of [FP16, BF16, FP32]) {
    it(`for 4,000 random ${f.name} pairs in all four modes, with the exact error under half a unit for nearest`, () => {
      const r = rng(f.e * 100 + f.m);
      for (let i = 0; i < 1000; i++) {
        const a = randomValue(f, r);
        // Half of the pairs share a binade region, so cancellation and carries are common too.
        const b = i % 2 ? randomValue(f, r) : round(f, a * (r() * 4 - 2));
        for (const mode of MODES) {
          const t = alignTrace(f, a, b, mode);
          const want = add(f, a, b, mode);
          expect(same(t.result, want), `${a} + ${b} (${mode}): ${t.result} vs ${want}`).toBe(true);
          if (mode === 'rne' && Number.isFinite(t.result)) expect(t.errorUlps).toBeLessThanOrEqual(0.5);
          // Under one unit exactly; as a binary64 ratio, 1 − 2^-100 prints as 1.
          if (Number.isFinite(t.result) && !last(t.steps, 'round')?.overflow) expect(last(t.steps, 'round')?.errorBelowOne ?? true).toBe(true);
        }
      }
    });
  }
  it('special operands: infinity minus infinity is NaN, a NaN goes straight through', () => {
    expect(Number.isNaN(alignTrace(FP32, Infinity, -Infinity).result)).toBe(true);
    expect(alignTrace(FP32, Infinity, -Infinity).steps[0]).toMatchObject({ kind: 'special', why: 'inf-minus-inf' });
    expect(alignTrace(FP32, NaN, 1).steps[0]).toMatchObject({ kind: 'special', why: 'nan-in' });
    expect(alignTrace(FP32, 1, -1).steps.at(-1)).toMatchObject({ kind: 'exact-zero' });
    expect(Object.is(alignTrace(FP32, 1, -1, 'down').result, -0)).toBe(true);
  });
});

describe('the stepper presets', () => {
  it('the text each preset puts in the fields parses back to its operands', () => {
    for (const p of ALIGN_PRESETS) expect([parseOperand(p.aText), parseOperand(p.bText)], p.id).toEqual([p.a, p.b]);
    expect(parseOperand('−2^-3')).toBe(-0.125);
    expect(parseOperand('3*2^4')).toBe(48);
    expect(parseOperand('1e-3')).toBe(0.001);
    expect(parseOperand('abc')).toBeNull();
    expect(parseOperand('')).toBeNull();
  });
  const at = (id: string, mode: Mode) => {
    const p = ALIGN_PRESETS.find((x) => x.id === id)!;
    return alignTrace(p.fmt, p.a, p.b, mode);
  };
  it('1 + 2^-11 in FP16 is exactly half a unit: a tie that nearest-even keeps at 1', () => {
    expect(at('half-ulp', 'rne').result).toBe(1);
    expect(last(at('half-ulp', 'rne').steps, 'round').reason).toBe('tie-even');
    expect(at('half-ulp', 'up').result).toBe(1 + 2 ** -10);
    expect(at('half-ulp', 'rne').steps.filter((s) => s.kind === 'shift').length).toBe(11);
  });
  it('2048 + 1 in FP16 (q-tie): 2048 under nearest-even and toward zero, 2050 rounding up', () => {
    expect(at('stall', 'rne').result).toBe(2048);
    expect(at('stall', 'rtz').result).toBe(2048);
    expect(at('stall', 'up').result).toBe(2050);
    expect(at('stall', 'down').result).toBe(2048);
    expect(last(at('stall', 'rne').steps, 'round').below).toBe(2048);
    expect(last(at('stall', 'rne').steps, 'round').above).toBe(2050);
  });
  it('1 + 0.75 ULP in FP32 (the probe, appA-facts E1): nearest gives 1 + 2^-23, toward zero gives 1', () => {
    expect(at('probe', 'rne').result).toBe(1 + 2 ** -23);
    expect(at('probe', 'rtz').result).toBe(1);
    expect(at('probe', 'rne').errorUlps).toBe(0.25);
    expect(at('probe', 'rtz').errorUlps).toBe(0.75);
    const shifts = at('probe', 'rne').steps.filter((s) => s.kind === 'shift');
    expect(shifts.length).toBe(24);
  });
  it('1.5 - 1.25 in FP16 cancels: the result moves 2 places left and is exact', () => {
    const t = at('cancel', 'rne');
    expect(t.result).toBe(0.25);
    expect(last(t.steps, 'normalise').moved).toBe(-2);
    expect(last(t.steps, 'round').reason).toBe('exact');
  });
  it('1000 + 0.3 in FP16 (the scene): 0.3 is stored as 0.300048828125, all 11 bits leave the window, the sum rounds to 1000.5', () => {
    const t = at('addend', 'rne');
    expect(t.b).toBe(0.300048828125);
    const c = t.steps[0];
    expect(c.kind === 'compare' && [c.big.exp, c.small.exp, c.diff]).toEqual([9, -2, 11]);
    const sh = last(t.steps, 'shift');
    expect([sh.window, sh.tail, sh.g, sh.r, sh.s]).toEqual(['00000000000', '10011001101', 1, 0, 1]);
    const r = last(t.steps, 'round');
    expect([r.below, r.above, r.result]).toEqual([1000, 1000.5, 1000.5]);
    expect(t.errorUlps).toBeCloseTo(0.3999, 4);
  });
  it('a gap wider than p + 2 places is shown as p + 2 single shifts and one jump', () => {
    const t = alignTrace(FP32, 1, 2 ** -60, 'up');
    const shifts = t.steps.filter((s) => s.kind === 'shift');
    expect(shifts.length).toBe(24 + 2 + 1);
    expect(shifts.at(-1)).toMatchObject({ k: 60, jump: true, g: 0, r: 0, s: 1 });
    expect(t.result).toBe(1 + 2 ** -23); // rounding up sees the sticky bit
    expect(alignTrace(FP32, 1, 2 ** -60, 'rne').result).toBe(1);
  });
  it('overflow follows the format: FP16 65504 + 32 is infinity to nearest, 65504 toward zero', () => {
    expect(alignTrace(FP16, 65504, 32, 'rne').result).toBe(Infinity);
    expect(alignTrace(FP16, 65504, 32, 'rtz').result).toBe(65504);
    expect(last(alignTrace(FP16, 65504, 32, 'rne').steps, 'round').overflow).toBe(true);
    expect(alignTrace(TOY5, 3.5, 3.5, 'rne').result).toBe(3.5); // the toy format has no infinity: it saturates
  });
});

describe('Exercise 3 (p. 599): an adder that only truncates', () => {
  it('loses just under 1 ULP at worst, against 0.5 ULP to nearest (all pairs of FP16 values near 1)', () => {
    let worstRtz = 0;
    let worstRne = 0;
    const r = rng(3);
    for (let i = 0; i < 4000; i++) {
      const a = 1 + Math.floor(r() * 1024) / 1024;
      const b = Math.floor(r() * 2048) * 2 ** -21;
      worstRtz = Math.max(worstRtz, alignTrace(FP16, a, b, 'rtz').errorUlps);
      worstRne = Math.max(worstRne, alignTrace(FP16, a, b, 'rne').errorUlps);
    }
    expect(worstRtz).toBeLessThan(1);
    expect(worstRtz).toBeGreaterThan(0.99);
    expect(worstRne).toBe(0.5);
  });
});

describe('runningSum: the disappearing addend in FP16', () => {
  const run = runningSum(FP16, 1000, 0.3, 400);
  it('below 1024 every addition rounds up to a full half-step; at 1024 the sum stops', () => {
    expect(run.acc[1]).toBe(1000.5);
    expect(run.acc[48]).toBe(1024);
    expect(run.stallAt).toBe(49);
    expect(run.acc[400]).toBe(1024);
  });
  it('while the exact sum keeps rising', () => {
    expect(run.exact[400]).toBeCloseTo(1000 + 400 * 0.300048828125, 9);
    expect(run.exact[400]).toBeGreaterThan(1120);
  });
  it('in FP32 the same loop does not stall', () => {
    const r32 = runningSum(FP32, 1000, 0.3, 400);
    expect(r32.stallAt).toBeNull();
    expect(Math.abs(r32.acc[400] - r32.exact[400])).toBeLessThan(0.05);
  });
});

describe('bf16FromBits: llama.cpp\'s conversion (appA-facts D3)', () => {
  it('equals minifloat encode(BF16, x, "rne") on random, tie and carry inputs', () => {
    const r = rng(16);
    const inputs: number[] = [];
    for (let i = 0; i < 20000; i++) inputs.push(Math.floor(r() * 2 ** 32));
    for (let i = 0; i < 2000; i++) inputs.push(((Math.floor(r() * 2 ** 16) << 16) | 0x8000) >>> 0); // exact ties
    for (let i = 0; i < 2000; i++) inputs.push(((Math.floor(r() * 2 ** 16) << 16) | 0xffff) >>> 0); // carries
    inputs.push(0x3fffffff, 0x7f7fffff, 0x00000001, 0x007fffff, 0x80000000, 0x7f800000, 0xff800000);
    for (const u of inputs) {
      const c = bf16FromBits(u);
      if (c.nan) {
        expect(decode(BF16, c.rounded).cls).toBe('qnan');
        continue;
      }
      expect(c.rounded, u.toString(16)).toBe(encode(BF16, f32FromBits(u), 'rne'));
      expect(c.truncated).toBe(encode(BF16, f32FromBits(u), 'rtz'));
    }
  });
  it('the presets show what they say', () => {
    const p = (id: string) => bf16FromBits(BF16_PRESETS.find((x) => x.id === id)!.bits);
    expect([p('tie-even').tie, p('tie-even').roundedUp, p('tie-even').roundedValue]).toEqual([true, false, 1]);
    expect([p('tie-odd').tie, p('tie-odd').roundedUp, p('tie-odd').roundedValue]).toEqual([true, true, 1.015625]);
    expect([p('carry').carryIntoExponent, p('carry').roundedValue, p('carry').truncatedValue]).toEqual([true, 2, 1.9921875]);
    expect([p('max').roundedValue, p('max').truncatedValue]).toEqual([Infinity, f32FromBits(0x7f7f0000)]);
    expect(p('snan').rounded).toBe(0x7fc0);
    expect(p('snan').truncatedValue).toBe(Infinity); // truncating that NaN loses its payload and leaves infinity
    expect(p('pi').roundedErr).toBeLessThanOrEqual(0.5);
    expect(p('pi').truncatedErr).toBeLessThan(1);
  });
  it('truncation is biased toward zero; rounding is not (10,000 random values in [1, 2))', () => {
    const r = rng(7);
    let sumT = 0;
    let sumR = 0;
    for (let i = 0; i < 10000; i++) {
      const x = Math.fround(1 + r());
      const c = bf16FromBits(bitsFromF32(x));
      sumT += c.truncatedValue - x;
      sumR += c.roundedValue - x;
    }
    expect(sumT / 10000 / 2 ** -7).toBeCloseTo(-0.5, 1); // half a BF16 unit below, on average
    expect(Math.abs(sumR / 10000)).toBeLessThan(0.0003);
  });
});

// The C++ shown in section A.4 (labs/appA/snippets/bf16_round.h), compiled and run, must give
// the model's bits. Skipped when there is no C++ compiler.
const repo = path.resolve(__dirname, '../../..');
const compiler = ['c++', 'clang++', 'g++'].find((c) => spawnSync(c, ['--version']).status === 0);
describe.skipIf(!compiler)('labs/appA/snippets/bf16_round.h, compiled and run', () => {
  it('agrees with bf16FromBits bit for bit, NaNs included', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'bf16-'));
    const exe = path.join(dir, 'bf16_round_test');
    const src = path.join(repo, 'labs/appA/snippets/check/bf16_round_test.cpp');
    const c = spawnSync(compiler!, ['-std=c++17', '-Wall', '-Wextra', '-O2', src, '-o', exe], { encoding: 'utf8' });
    expect(c.status, c.stderr).toBe(0);
    const r = rng(99);
    const inputs = [...BF16_PRESETS.map((p) => p.bits), 0x7fc00000, 0xffc00001, 0x7f800000, 0x00000001, 0x80008000];
    for (let i = 0; i < 400; i++) inputs.push(Math.floor(r() * 2 ** 32));
    const run = spawnSync(exe, inputs.map((u) => u.toString(16)), { encoding: 'utf8' });
    expect(run.status).toBe(0);
    const lines = run.stdout.trim().split('\n');
    expect(lines.length).toBe(inputs.length);
    inputs.forEach((u, i) => {
      const m = bf16FromBits(u);
      const hex = (h: number) => h.toString(16).padStart(4, '0');
      expect(lines[i], u.toString(16)).toBe(`${hex(m.rounded)} ${hex(m.truncated)}`);
    });
    fs.rmSync(dir, { recursive: true, force: true });
  });
});

describe('specialTable: what comes out (§A.3)', () => {
  const t = Object.fromEntries(specialTable().map((r) => [r.id, r]));
  it('every row has the outcome the book or the vendor states', () => {
    expect(t['zero-zero'].outcome).toBe('nan');
    expect(t['one-zero'].outcome).toBe('+inf');
    expect(t['minus-one-zero'].outcome).toBe('-inf');
    expect(t['inf-inf'].outcome).toBe('nan');
    expect(t['zero-inf'].outcome).toBe('nan');
    expect(t['one-inf'].outcome).toBe('zero');
    expect(t['one-nan'].outcome).toBe('nan');
    expect(t['nan-eq'].outcome).toBe('false');
    expect(t['nan-ne'].outcome).toBe('true');
    expect(t['max-two'].outcome).toBe('+inf');
    expect(t['fp16-max'].outcome).toBe('+inf');
    expect(t['fmaxf'].outcome).toBe('finite');
    expect(t['fmaxf'].value).toBe(1);
  });
  it('fmaxf drops a NaN from either side and returns NaN only for two NaNs', () => {
    expect(fmaxf(NaN, 1)).toBe(1);
    expect(fmaxf(1, NaN)).toBe(1);
    expect(Number.isNaN(fmaxf(NaN, NaN))).toBe(true);
  });
  it('the FP16 row: 65504 + 16 is the first sum that overflows', () => {
    expect(add(FP16, 65504, 15)).toBe(65504);
    expect(add(FP16, 65504, 16)).toBe(Infinity);
  });
});

describe('the classifier (Fig. A.8)', () => {
  it('counts the FP32 NaNs as 2^24 - 2: the book\'s 2^22 - 1 is the signalling ones of one sign (E1)', () => {
    const c = patternCounts(FP32);
    expect(c.qnan + c.snan).toBe(2 ** 24 - 2);
    expect(c.snan).toBe(2 * (2 ** 22 - 1));
    expect(c.inf).toBe(2);
    expect(c.zero).toBe(2);
    expect(c.subnormal).toBe(2 * (2 ** 23 - 1));
    expect(c.zero + c.subnormal + c.normal + c.inf + c.qnan + c.snan + c.nan).toBe(2 ** 32);
  });
  it('the narrow formats spend fewer patterns on special values', () => {
    expect(patternCounts(E4M3)).toMatchObject({ inf: 0, nan: 2, qnan: 0, snan: 0, total: 256 });
    expect(patternCounts(E5M2)).toMatchObject({ inf: 2, nan: 6, total: 256 });
    expect(patternCounts(E2M1)).toMatchObject({ inf: 0, nan: 0, zero: 2, subnormal: 2, normal: 12, total: 16 });
    expect(patternCounts(FP16)).toMatchObject({ inf: 2, qnan: 1024, snan: 1022 });
  });
  it('E4M3 gives its top exponent code to numbers, so Fig. A.8 misreads S.1111.110 (448)', () => {
    expect(decode(E4M3, 0x7e).value).toBe(448);
    expect(figA8Row(E4M3, 0x7e)).toBe('nan');
    expect(figA8Agrees(E4M3, 0x7e)).toBe(false);
    expect(maxFinite(E4M3)).toBe(448);
    // With IEEE's rule the largest E4M3 value would be 1.875 × 2^7 = 240 (our arithmetic).
    expect((2 - 2 ** -3) * 2 ** 7).toBe(240);
  });
  it('Fig. A.8 holds for every pattern of FP16, BF16 and E5M2', () => {
    for (const f of [FP16, BF16, E5M2]) for (let b = 0; b < 2 ** (1 + f.e + f.m); b += f === E5M2 ? 1 : 7) expect(figA8Agrees(f, b)).toBe(true);
  });
  it('presets: patterns where they exist, a reason where they do not', () => {
    const get = (f: Format, id: string) => classifierPresets(f).find((p) => p.id === id)!;
    expect(get(FP32, 'snan').bits).toBe(0x7f800001);
    expect(get(FP32, 'nan').bits).toBe(0x7fc00000);
    expect(get(FP32, 'nan').label).toBe('quiet NaN');
    expect(get(FP32, 'min-sub').bits).toBe(1);
    expect(get(FP32, 'max').bits).toBe(0x7f7fffff);
    expect(get(E4M3, 'inf')).toMatchObject({ bits: null });
    expect(get(E4M3, 'inf').reason).toContain('no infinity');
    expect(get(E4M3, 'nan').bits).toBe(0x7f);
    expect(get(E4M3, 'snan').bits).toBeNull();
    expect(get(E5M2, 'nan').label).toBe('NaN');
    expect(get(E2M1, 'nan').bits).toBeNull();
    expect(get(E2M1, 'max').bits).toBe(0b0111);
    for (const f of CLASSIFIER_FORMATS)
      for (const p of classifierPresets(f)) expect(p.bits !== null || !!p.reason, `${f.name} ${p.id}`).toBe(true);
  });
});

describe('exactDecimal', () => {
  it('writes n × 2^k exactly, cutting long fractions', () => {
    expect(exactDecimal(9n, -2)).toBe('2.25');
    expect(exactDecimal(-3n, 0)).toBe('−3');
    expect(exactDecimal(0x1000003n, -24)).toBe('1.000000178813934326171875');
    expect(exactDecimal(0x1000003n, -24, 10)).toBe('1.000000178…');
    expect(exactDecimal(1n, 200).startsWith('≈ 1.606938044258990e+60')).toBe(true);
  });
});
