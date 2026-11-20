import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  MIRROR_AT,
  OUTLIER_AT,
  SCENE_BLOCK,
  bestUnder,
  blockAddress,
  blockDotQ4_0,
  demoBlock,
  dequantQ4_0,
  dequantQ8_0,
  errorSweep,
  f32,
  fakeQuant,
  fp16,
  fp16Bits,
  fp16FromBits,
  getScaleMinK4,
  isAligned,
  packMlx4,
  packQ4K,
  packQ4_0,
  packScaleMinK4,
  parseQ4_0,
  placeOf,
  qsLoads,
  quantTrace,
  quantizeQ4_0,
  quantizeQ4_1,
  quantizeQ8_0,
  rng,
  roundf,
  sampleWeights,
  sceneActivations,
  unpackMlx4,
  unpackQ4K,
  unpackQ4_0,
  unpackTrace,
  weightsInByte,
  type Layout,
  type SweepPoint,
} from './quant';

const block = (extreme: number, other: number) => {
  const v = Array(32).fill(0).map((_, i) => f32(0.01 * ((i % 7) - 3)));
  v[3] = extreme;
  v[10] = other;
  return v;
};

describe('FP16 storage of the scale (appA A2)', () => {
  it('0.1 is stored as 0x2E66, which is 0.0999755859375', () => {
    expect(fp16Bits(0.1)).toBe(0x2e66);
    expect(fp16(0.1)).toBe(0.0999755859375);
    expect(fp16FromBits(0x2e66)).toBe(0.0999755859375);
  });
  it('decodes every finite pattern to what minifloat rounds it to', () => {
    for (let b = 0; b < 0x7c00; b += 37) expect(fp16(fp16FromBits(b))).toBe(fp16FromBits(b));
  });
  it('roundf sends halves away from zero', () => {
    expect([roundf(2.5), roundf(-2.5), roundf(0.49999997), roundf(-0.5)]).toEqual([3, -3, 0, -1]);
  });
});

describe('llama.cpp Q4_0, the reference rule (x2-facts D1)', () => {
  it('q-q40-clip: extreme -0.8, a weight of +0.8 comes back as 0.6998 (0.7 with the float d)', () => {
    const b = quantizeQ4_0(block(f32(-0.8), f32(0.8)));
    expect(b.d).toBe(f32(0.1));
    expect(b.codes[3]).toBe(0);
    expect(b.codes[10]).toBe(15);
    expect(dequantQ4_0(b)[10]).toBe(0.6998291015625);
    expect(f32(7 * b.d)).toBeCloseTo(0.7, 6);
  });
  it('the extreme lands exactly on code 0: -8·d is the extreme, up to FP16 rounding of d', () => {
    const v = sampleWeights(3, 32, 'normal', 0.3);
    const b = quantizeQ4_0(v);
    const at = v.findIndex((x) => Math.abs(x) === Math.max(...v.map(Math.abs)));
    expect(b.codes[at]).toBe(0);
    expect(f32(-8 * b.d)).toBe(v[at]);
  });
  it('q-q40-zero: an all-zero block: d is -0 (0 / -8), stored as 0x8000, every code 8, every value 0', () => {
    const b = quantizeQ4_0(Array(32).fill(0));
    expect(Object.is(b.d, -0)).toBe(true);
    expect(b.dBits).toBe(0x8000);
    expect(b.codes.every((c) => c === 8)).toBe(true);
    expect(dequantQ4_0(b).every((x) => x === 0)).toBe(true);
  });
  it('a positive extreme gives a negative d', () => {
    expect(quantizeQ4_0(block(f32(0.8), 0)).d).toBe(f32(-0.1));
  });
  it('byte j holds weight j low and weight j + 16 high; the bytes parse back', () => {
    const b = quantizeQ4_0(sampleWeights(9, 32));
    for (let j = 0; j < 16; j++) expect(b.bytes[2 + j]).toBe(b.codes[j] | (b.codes[j + 16] << 4));
    expect(b.bytes.slice(0, 2)).toEqual([b.dBits & 0xff, b.dBits >> 8]);
    expect(parseQ4_0(b.bytes)).toEqual({ dBits: b.dBits, codes: b.codes });
    expect(b.bytes.length).toBe(18);
  });
  it('rejects anything but one block of 32', () => {
    expect(() => quantizeQ4_0([1, 2, 3])).toThrow();
  });
});

describe('llama.cpp Q8_0 and Q4_1 (D1)', () => {
  it('Q8_0: the extreme is ±127, codes are roundf(x / d), 34 bytes', () => {
    const v = sampleWeights(4, 32);
    const b = quantizeQ8_0(v);
    expect(Math.max(...b.codes.map(Math.abs))).toBe(127);
    expect(b.bytes.length).toBe(34);
    const back = dequantQ8_0(b);
    back.forEach((x, i) => expect(Math.abs(x - v[i])).toBeLessThanOrEqual(0.51 * fp16FromBits(b.dBits) + 1e-6));
  });
  it('Q8_0 of zeros: d = 0, the inverse is guarded, codes 0', () => {
    const b = quantizeQ8_0(Array(32).fill(0));
    expect(b.d).toBe(0);
    expect(b.codes.every((c) => c === 0)).toBe(true);
  });
  it('Q4_1: min on code 0, max on code 15', () => {
    const v = sampleWeights(5, 32);
    const b = quantizeQ4_1(v);
    expect(b.codes[v.indexOf(Math.min(...v))]).toBe(0);
    expect(b.codes[v.indexOf(Math.max(...v))]).toBe(15);
  });
});

describe('the six rules of the stepper', () => {
  const v = demoBlock();
  it('the demo block: -0.8 at weight 5, its mirror +0.8 at weight 21, nothing else as large', () => {
    expect(v[OUTLIER_AT]).toBe(f32(-0.8));
    expect(v[MIRROR_AT]).toBe(f32(0.8));
    expect(v.filter((x) => Math.abs(x) >= 0.8).length).toBe(2);
    expect(SCENE_BLOCK).toEqual(v);
  });
  it('Q4_0 on it: only the mirror clips, by one step', () => {
    const t = quantTrace('q4_0', v);
    expect(t.clipped.flatMap((c, i) => (c ? [i] : []))).toEqual([MIRROR_AT]);
    expect(t.rounded[MIRROR_AT]).toBe(16);
    expect(t.dequant[MIRROR_AT]).toBe(0.6998291015625);
    expect(t.codes[OUTLIER_AT]).toBe(0);
  });
  it('the trace agrees with quantizeQ4_0, and with FP16 off it uses the float d', () => {
    const t = quantTrace('q4_0', v);
    const b = quantizeQ4_0(v);
    expect(t.codes).toEqual(b.codes);
    expect(t.dequant).toEqual(dequantQ4_0(b));
    expect(quantTrace('q4_0', v, false).scaleStored).toBe(f32(0.1));
  });
  it('symmetric ±7 and Q8_0 never clip; Q4_0 has 16 levels and ±7 has 15', () => {
    for (const r of ['sym4', 'q8_0'] as const) expect(quantTrace(r, v).clipped.some(Boolean)).toBe(false);
    expect(quantTrace('q4_0', v).levels.length).toBe(16);
    expect(quantTrace('sym4', v).levels.length).toBe(15);
  });
  it('asymmetric rules spend no level outside [min, max]', () => {
    for (const r of ['q4_1', 'mlx-doc'] as const) {
      const t = quantTrace(r, v, false);
      expect(t.levels[0]).toBeCloseTo(t.extreme.min, 6);
      expect(t.levels[15]).toBeCloseTo(t.extreme.max, 5);
    }
  });
  it('MLX as coded: zero is on the grid (to float rounding), the bias is the larger-magnitude edge', () => {
    const w = sampleWeights(12, 32, 'normal', 0.25);
    const t = quantTrace('mlx-code', w, false);
    expect(t.zeroCode).not.toBeNull();
    const z = f32(f32(t.zeroCode! * t.scale) + t.offset);
    expect(Math.abs(z)).toBeLessThan(1e-7);
    const big = Math.abs(t.extreme.min) > Math.abs(t.extreme.max) ? t.extreme.min : t.extreme.max;
    expect(t.offset).toBe(big);
  });
  it('MLX as coded: the scale is negative when the max is the larger edge', () => {
    const w = demoBlock('normal', 0.8, '+');
    const t = quantTrace('mlx-code', w, false);
    expect(t.offset).toBe(f32(0.8));
    expect(t.scale).toBeLessThan(0);
    expect(t.codes[OUTLIER_AT]).toBe(0);
  });
  it('the documented MLX rule does not, in general, put zero on the grid', () => {
    const t = quantTrace('mlx-doc', sampleWeights(12, 32, 'normal', 0.25), false);
    const k = -t.offset / t.scale;
    expect(Math.abs(k - Math.round(k))).toBeGreaterThan(1e-3);
  });
});

describe('errorSweep: E2 orders of magnitude (our seeded generator, not E2 exactly)', () => {
  const at = (pts: SweepPoint[], bits: number, gran: SweepPoint['gran']) => pts.find((p) => p.bits === bits && p.gran === gran)!.rel;
  const qn = errorSweep({ dist: 'normal', scheme: 'signed-extreme' });
  const qt = errorSweep({ dist: 't3', scheme: 'signed-extreme' });
  const sn = errorSweep({ dist: 'normal', scheme: 'symmetric' });
  const an = errorSweep({ dist: 'normal', scheme: 'affine' });
  const at3 = errorSweep({ dist: 't3', scheme: 'affine' });
  const st3 = errorSweep({ dist: 't3', scheme: 'symmetric' });
  const near = (v: number, e2: number, tol = 0.15) => expect(Math.abs(v / e2 - 1)).toBeLessThan(tol);
  it('Q4_0-style, normal: about 0.086 at block 32 and 0.111 at 256', () => {
    near(at(qn, 4, 32), 0.086);
    near(at(qn, 4, 256), 0.1108);
  });
  it('Q4_0-style, t3: about 0.127 at block 32 and 0.246 at 256', () => {
    near(at(qt, 4, 32), 0.1269);
    near(at(qt, 4, 256), 0.2456, 0.2);
  });
  it('affine, group 64, normal, by bits: 0.45, 0.19, 0.090, 0.043, 0.021, 0.0053', () => {
    [[2, 0.4501], [3, 0.1925], [4, 0.0897], [5, 0.0434], [6, 0.0213], [8, 0.0053]].forEach(([b, e]) => near(at(an, b, 64), e));
  });
  it('each extra bit roughly halves the error', () => {
    for (let b = 3; b < 8; b++) {
      const r = at(an, b, 64) / at(an, b + 1, 64);
      expect(r).toBeGreaterThan(1.7);
      expect(r).toBeLessThan(2.4);
    }
  });
  it('small blocks help heavy tails most', () => {
    expect(at(qt, 4, 256) / at(qt, 4, 32)).toBeGreaterThan(1.5 * (at(qn, 4, 256) / at(qn, 4, 32)) - 0.3);
    expect(at(qt, 4, 256) / at(qt, 4, 32)).toBeGreaterThan(at(qn, 4, 256) / at(qn, 4, 32));
  });
  it("Q4_0's sixteenth level beats a ±7 grid at every block size", () => {
    for (const g of [16, 32, 64, 128, 256]) expect(at(qn, 4, g)).toBeLessThan(at(sn, 4, g));
  });
  it('one scale per tensor: int8 fine on normal weights, int4 unusable on heavy tails', () => {
    near(at(sn, 8, 'tensor'), 0.0113, 0.25);
    expect(at(st3, 4, 'tensor')).toBeGreaterThan(0.4);
  });
  it('the predict: on t3 at 4 bits group 64, 5 bits at group 64 beats 4 bits at group 32', () => {
    expect(at(at3, 5, 64)).toBeLessThan(at(at3, 4, 32));
    near(at(at3, 4, 64), 0.1329, 0.2);
    expect(errorSweep({ dist: 't3', scheme: 'affine', bits: [4], grans: [32] })[0].bpw).toBe(5);
    expect(errorSweep({ dist: 't3', scheme: 'affine', bits: [5], grans: [64] })[0].bpw).toBe(5.5);
  });
  it('is deterministic by seed', () => {
    expect(errorSweep({ dist: 't3', scheme: 'affine', bits: [4], grans: [64] })).toEqual(
      errorSweep({ dist: 't3', scheme: 'affine', bits: [4], grans: [64] }),
    );
  });
});

describe('where each weight lives (D1, D2, D5)', () => {
  it('q-byte9: the high nibble of Q4_0 byte 9 is weight 25; byte 5 holds 5 and 21', () => {
    expect(weightsInByte('q4_0', 9)[1]).toBe(25);
    expect(weightsInByte('q4_0', 5)).toEqual([5, 21]);
  });
  it('q-mlx13: MLX weight 13 is in word 1, bits 20 to 23', () => {
    expect(placeOf('mlx4', 13)).toMatchObject({ word: 1, bits: [20, 23] });
  });
  it('Q4_K: weight 40 is byte 8, high; weight 70 is byte 38, low', () => {
    expect(placeOf('q4_k', 40)).toMatchObject({ byte: 8, high: true });
    expect(placeOf('q4_k', 70)).toMatchObject({ byte: 38, high: false });
  });
  it('placeOf and weightsInByte are inverse for every layout and weight', () => {
    for (const l of ['q4_0', 'mlx4', 'q4_k'] as Layout[])
      for (let j = 0; j < (l === 'q4_k' ? 256 : 32); j++) {
        const p = placeOf(l, j);
        expect(weightsInByte(l, p.byte)[p.high ? 1 : 0]).toBe(j);
      }
  });
  it('packings round-trip, and the bytes of MLX words agree with placeOf (little-endian)', () => {
    const r = rng(2);
    const c32 = Array.from({ length: 32 }, () => Math.floor(r() * 16));
    const c256 = Array.from({ length: 256 }, () => Math.floor(r() * 16));
    expect(unpackQ4_0(packQ4_0(c32))).toEqual(c32);
    expect(unpackMlx4(packMlx4(c32))).toEqual(c32);
    expect(unpackQ4K(packQ4K(c256))).toEqual(c256);
    const words = packMlx4(c32);
    const bytes = words.flatMap((w) => [w & 255, (w >>> 8) & 255, (w >>> 16) & 255, w >>> 24]);
    c32.forEach((c, j) => {
      const p = placeOf('mlx4', j);
      expect(p.high ? bytes[p.byte] >> 4 : bytes[p.byte] & 15).toBe(c);
    });
  });
  it('get_scale_min_k4 reads back every 6-bit scale and min the packer stored', () => {
    const r = rng(8);
    for (let t = 0; t < 50; t++) {
      const sc = Array.from({ length: 8 }, () => Math.floor(r() * 64));
      const m = Array.from({ length: 8 }, () => Math.floor(r() * 64));
      const q = packScaleMinK4(sc, m);
      expect(q.every((b) => b >= 0 && b < 256)).toBe(true);
      for (let j = 0; j < 8; j++) expect(getScaleMinK4(j, q)).toEqual({ sc: sc[j], m: m[j] });
    }
  });
});

describe('18-byte blocks end to end (A5, C5)', () => {
  it('q-align: from a 16-byte aligned start, blocks 1 and 3 have int-aligned quants (20 and 56)', () => {
    expect([0, 1, 2, 3].map((i) => blockAddress(i).qs)).toEqual([2, 20, 38, 56]);
    expect([0, 1, 2, 3].map((i) => isAligned(blockAddress(i).qs, 4))).toEqual([false, true, false, true]);
  });
  it('2-byte loads always work; d is always 2-byte aligned', () => {
    for (let i = 0; i < 16; i++) {
      expect(qsLoads(i, 2).every((l) => l.ok)).toBe(true);
      expect(isAligned(blockAddress(i).d, 2)).toBe(true);
    }
  });
  it('16-byte loads of qs are aligned only for blocks 7, 15, ...', () => {
    const ok = Array.from({ length: 16 }, (_, i) => qsLoads(i, 16)[0].ok);
    expect(ok.flatMap((o, i) => (o ? [i] : []))).toEqual([7, 15]);
  });
});

describe('unpacking one word (D3, D4, E4)', () => {
  const b = quantizeQ4_0(SCENE_BLOCK);
  const qs = b.bytes.slice(2);
  const y = sceneActivations();
  it('CUDA: mask gives weights 4i..4i+3, shift and mask gives 4i+16..4i+19', () => {
    for (let i = 0; i < 4; i++) {
      const s = unpackTrace('cuda32', qs, i, y);
      expect(s.map((x) => x.op)).toEqual(['load', 'mask', 'shift', 'sub8', 'dot']);
      expect(s[1].out.map((o) => o.weight)).toEqual([4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]);
      expect(s[1].out.map((o) => o.value)).toEqual(s[1].out.map((o) => b.codes[o.weight]));
      expect(s[2].out.map((o) => o.weight)).toEqual([16, 17, 18, 19].map((w) => w + 4 * i));
      expect(s[3].out.map((o) => o.value)).toEqual(s[3].out.map((o) => b.codes[o.weight] - 8));
    }
  });
  it('Metal: the four masked nibbles are the code times 1, 256, 16, 4096 and the pre-divided products are exact', () => {
    for (let i = 0; i < 8; i++) {
      const s = unpackTrace('metal16', qs, i, y);
      const W = [2 * i, 2 * i + 1, 2 * i + 16, 2 * i + 17];
      const mul = [1, 256, 16, 4096];
      s.slice(1, 5).forEach((st, k) => expect(st.out[0].value).toBe(b.codes[W[k]] * mul[k]));
      s[5].out.forEach((o, k) => expect(o.value).toBe(f32(b.codes[W[k]] * y[W[k]])));
    }
  });
  it('the folded dot matches the dequantised dot to rounding, and differs from the float dot by the quantisation error', () => {
    const r = blockDotQ4_0(b, y, SCENE_BLOCK);
    expect(Math.abs(r.folded - r.dequantised)).toBeLessThan(1e-5);
    expect(Math.abs(r.dequantised - r.exact)).toBeGreaterThan(1e-4);
  });
});

describe('fakeQuant (for qstep.ts)', () => {
  it('F32 is the identity; Q4_0 matches row quantisation', () => {
    const w = sampleWeights(1, 2 * 64);
    expect(fakeQuant(w, 2, 64, 'f32')).toEqual(w);
    const q = fakeQuant(w, 2, 64, 'q4_0');
    expect(q.slice(0, 32)).toEqual(dequantQ4_0(quantizeQ4_0(w.slice(0, 32))));
  });
  it('error grows as bits fall: Q8_0 < Q4_0 < affine 3 < affine 2', () => {
    const w = sampleWeights(2, 16 * 64);
    const rel = (f: Parameters<typeof fakeQuant>[3]) => {
      const q = fakeQuant(w, 16, 64, f);
      return Math.sqrt(q.reduce((s, x, i) => s + (x - w[i]) ** 2, 0) / w.reduce((s, x) => s + x * x, 0));
    };
    const e = (['q8_0', 'q4_0', 'affine3', 'affine2'] as const).map(rel);
    for (let i = 1; i < 4; i++) expect(e[i]).toBeGreaterThan(e[i - 1]);
  });
});

// The C++ the lessons show (labs/x2/snippets/q4_0_ref.h), compiled and run, must give the
// model's d bits, bytes, dequantised floats, words and folded dot bit for bit.
const compiler = ['c++', 'clang++', 'g++'].find((c) => spawnSync(c, ['--version']).status === 0);
describe.skipIf(!compiler)('labs/x2/snippets/q4_0_ref.h, compiled and run', () => {
  const dir = path.resolve(__dirname, '../../../labs/x2/snippets/check');
  const exe = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'q40-')), 'q4_0_ref_test');
  const hex = (x: number) => {
    const b = new DataView(new ArrayBuffer(4));
    b.setFloat32(0, x);
    return b.getUint32(0).toString(16).padStart(8, '0');
  };
  it('compiles without warnings', () => {
    const r = spawnSync(compiler!, ['-std=c++17', '-Wall', '-Wextra', '-Werror', '-O2', '-ffp-contract=off', path.join(dir, 'q4_0_ref_test.cpp'), '-o', exe], { encoding: 'utf8' });
    expect(r.stderr).toBe('');
    expect(r.status).toBe(0);
  });
  const blocks: [string, number[]][] = [
    ['the scene block', SCENE_BLOCK],
    ['q-q40-clip', block(f32(-0.8), f32(0.8))],
    ['all zero', Array(32).fill(0)],
    ['a positive extreme', demoBlock('normal', 0.8, '+')],
    ...[1, 2, 3, 4, 5, 6].map((s): [string, number[]] => [`seed ${s}`, sampleWeights(100 + s, 32, s % 2 ? 'normal' : 't3', 0.02 * s)]),
  ];
  for (const [name, w] of blocks)
    it(name, () => {
      const y = sceneActivations(name.length);
      const r = spawnSync(exe, [...w.map(hex), ...y.map(hex)], { encoding: 'utf8' });
      expect(r.status).toBe(0);
      const lines = r.stdout.trim().split('\n');
      const b = quantizeQ4_0(w);
      expect(parseInt(lines[0], 16)).toBe(b.dBits);
      expect(lines.slice(1, 17).map(Number)).toEqual(b.bytes.slice(2));
      expect(lines.slice(17, 49)).toEqual(dequantQ4_0(b).map(hex));
      const qs = b.bytes.slice(2);
      const words = [0, 1, 2, 3].map((i) => unpackTrace('cuda32', qs, i, y)[0].word.toString(16).padStart(8, '0'));
      expect(lines.slice(49, 53)).toEqual(words);
      expect(lines[53]).toBe(hex(blockDotQ4_0(b, y, w).folded));
      expect(lines[54]).toBe('18');
    });
});

describe('bestUnder: where to spend a budget of bits', () => {
  it('at 4.5 bits per weight on t3 with affine, the best is a 4-bit point; at 6.0, a 5-bit one', () => {
    const pts = errorSweep({ dist: 't3', scheme: 'affine' });
    const a = bestUnder(pts, 4.5)!;
    const b = bestUnder(pts, 6)!;
    expect(a.bits).toBe(4);
    expect(b.bits).toBe(5);
    expect(b.rel).toBeLessThan(a.rel);
    expect(bestUnder(pts, 1)).toBeNull();
  });
});
