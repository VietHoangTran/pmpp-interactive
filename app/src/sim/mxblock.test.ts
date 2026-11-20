import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { E2M1, E2M3, E3M2, E4M3, E4M3FNUZ, E5M2, decode, emax, maxFinite } from './minifloat';
import {
  ELEMENTS,
  ELEMENT_IDS,
  SCENE_BLOCK,
  bitsPerWeight,
  blockBytes,
  blockScale,
  convert,
  decodeCeiling,
  elemGrid,
  maxTree,
  namedFormat,
  opsPerByte,
  quantizeBlock,
  sampleBlock,
  sceneValues,
  type QuantOpts,
  type ScaleRule,
} from './mxblock';

const MXFP4: QuantOpts = { elem: 'E2M1', k: 32, scale: 'ocp-floor' };
const f32 = (x: number) => Math.fround(x);

describe('the element formats the page tabulates (OFP8 r1.1, MX v1.1: appA-facts A3, A4)', () => {
  it('E2M1 holds 0, 0.5, 1, 1.5, 2, 3, 4, 6 and nothing else', () => {
    expect(elemGrid(E2M1)).toEqual([0, 0.5, 1, 1.5, 2, 3, 4, 6]);
  });
  it('largest values: E4M3 448, E5M2 57,344, E2M3 7.5, E3M2 28, E2M1 6', () => {
    expect([E4M3, E5M2, E2M3, E3M2, E2M1].map(maxFinite)).toEqual([448, 57344, 7.5, 28, 6]);
  });
  it('the exponent MX subtracts: 2 for E2M1, 8 for E4M3, 15 for E5M2', () => {
    expect([emax(E2M1), emax(E4M3), emax(E5M2)]).toEqual([2, 8, 15]);
  });
  it('E4M3 smallest subnormal is 2^-9; its NaN is S.1111.111; E5M2 has infinities', () => {
    expect(decode(E4M3, 0b0_0000_001).value).toBe(2 ** -9);
    expect(decode(E4M3, 0b0_1111_111).value).toBeNaN();
    expect(decode(E4M3, 0b1_1111_111).value).toBeNaN();
    expect(decode(E5M2, 0b0_11111_00).value).toBe(Infinity);
  });
});

describe('near the top of E4M3 and E5M2 (OFP8 r1.1 Appendix A, the p-e4m3 reveal)', () => {
  it('460 becomes 448 with saturation, and also without: it rounds down', () => {
    expect(convert(460, E4M3, 'saturate')).toMatchObject({ value: 448, overflowed: false });
    expect(convert(460, E4M3, 'ieee')).toMatchObject({ value: 448, overflowed: false });
  });
  it('inputs up to and including 464 round to 448', () => {
    expect(convert(464, E4M3, 'ieee').value).toBe(448);
    expect(convert(464.0001, E4M3, 'ieee').value).toBeNaN();
  });
  it('470: 448 when saturating, NaN when not', () => {
    expect(convert(470, E4M3, 'saturate')).toMatchObject({ value: 448, overflowed: true, code: 0b0_1111_110 });
    expect(convert(470, E4M3, 'ieee')).toMatchObject({ overflowed: true, code: 0b0_1111_111 });
    expect(convert(470, E4M3, 'ieee').value).toBeNaN();
  });
  it('E5M2: 61,440 overflows: infinity, or 57,344 when saturating', () => {
    expect(convert(61440, E5M2, 'ieee').value).toBe(Infinity);
    expect(convert(61440, E5M2, 'saturate').value).toBe(57344);
    expect(convert(61439, E5M2, 'ieee').value).toBe(57344);
  });
});

describe('quantizeBlock: the OCP rule (MX v1.1 §6.3)', () => {
  it('an all-zero block: zero codes, zero error, the smallest UE8M0 scale (our choice)', () => {
    const q = quantizeBlock(Array(32).fill(0), MXFP4);
    expect(q.codes.every((c) => c === 0)).toBe(true);
    expect(q.decoded.every((d) => d === 0)).toBe(true);
    expect([q.zeroed, q.clamped, q.rmsRel]).toEqual([0, 0, 0]);
    expect(q.blocks[0]).toMatchObject({ scaleExp: -127, scaleCode: 0 });
  });
  it('a maximum of exactly 8: exponent 3 - 2 = 1, X = 2, 8 is stored as 4 and comes back exact', () => {
    const v = [8, ...Array(31).fill(1)];
    const q = quantizeBlock(v, MXFP4);
    expect(q.blocks[0]).toMatchObject({ scale: 2, scaleExp: 1, scaleCode: 128, clamped: 0 });
    expect(q.scaled[0]).toBe(4);
    expect(q.decoded[0]).toBe(8);
    expect(q.decoded.slice(1).every((d) => d === 1)).toBe(true);
  });
  it('a maximum of 7: exponent 0, X = 1, and 7 is clamped to 6; rounding the exponent up avoids the clamp', () => {
    const v = [7, ...Array(31).fill(1)];
    expect(quantizeBlock(v, MXFP4)).toMatchObject({ clamped: 1 });
    expect(quantizeBlock(v, MXFP4).decoded[0]).toBe(6);
    const up = quantizeBlock(v, { ...MXFP4, scale: 'cublas-ceil' });
    expect(up.blocks[0].scaleExp).toBe(1);
    expect(up.clamped).toBe(0);
    expect(up.decoded[0]).toBe(8); // 7 / 2 = 3.5, a tie between 3 and 4: the even code wins
  });
  it('q-outlier: one 100 among 0.3s: X = 2^(6 - 2) = 16, every 0.3 becomes 0, 100 becomes 96', () => {
    const v = Array(32).fill(f32(0.3));
    v[5] = 100;
    const q = quantizeBlock(v, MXFP4);
    expect(q.blocks[0]).toMatchObject({ scale: 16, scaleExp: 4, scaleCode: 131, amax: 100, amaxAt: 5, clamped: 1, zeroed: 31 });
    expect(q.decoded[5]).toBe(96);
    expect(q.scaled[0]).toBeLessThan(0.25); // below half of E2M1's smallest step, 0.5
    expect(q.decoded.filter((_, i) => i !== 5).every((d) => d === 0)).toBe(true);
    expect(q.rmsRelRest).toBe(1);
  });
  it('splitting into two blocks of 16 confines the damage to the outlier\'s half', () => {
    const v = Array(32).fill(f32(0.3));
    v[5] = 100;
    const q = quantizeBlock(v, { ...MXFP4, k: 16 });
    expect(q.blocks.map((b) => b.zeroed)).toEqual([15, 0]);
    expect(q.blocks[1].scaleExp).toBe(-4); // floor(log2 0.3) = -2, minus 2
    expect(q.decoded.slice(16).every((d) => d === 0.25)).toBe(true); // 0.3 × 16 = 4.8, nearest grid point 4

  });
  it('ties go to the even code, as the C++ snippet does', () => {
    const v = [4, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, -0.25, -5, ...Array(22).fill(0)];
    const q = quantizeBlock(v, MXFP4);
    expect(q.blocks[0].scale).toBe(1);
    expect(q.decoded.slice(0, 10)).toEqual([4, 0, 1, 1, 2, 2, 4, 4, -0, -4]);
  });
});

describe('decoding is codes times scale, for every element format, block size and rule', () => {
  const rules: ScaleRule[] = ['ocp-floor', 'cublas-ceil', 'e4m3'];
  for (const elem of ELEMENT_IDS)
    for (const k of [32, 16])
      for (const scale of rules)
        it(`${elem}, k = ${k}, ${scale}`, () => {
          for (const seed of [1, 2, 3, 4, 5]) {
            const v = sampleBlock(seed, 32, 0.25, seed % 2 ? 20 : 0, seed);
            const q = quantizeBlock(v, { elem, k, scale });
            const f = ELEMENTS[elem];
            q.decoded.forEach((d, i) => {
              expect(d).toBe(decode(f, q.codes[i]).value * q.blocks[q.blockOf[i]].scale);
            });
            expect(q.blocks.length).toBe(32 / k);
            if (scale === 'cublas-ceil') expect(q.clamped).toBe(0);
            if (scale !== 'e4m3') q.blocks.forEach((b) => expect(b.scale).toBe(2 ** b.scaleExp!));
            else q.blocks.forEach((b) => expect(decode(E4M3, b.scaleCode).value).toBe(b.scale));
            expect(q.bitsPerWeight).toBe((elem === 'E2M1' ? 4 : elem.startsWith('E4') || elem.startsWith('E5') ? 8 : 6) + 8 / k);
          }
        });
});

describe('the explorer\'s discoveries hold for its default data', () => {
  const v = sampleBlock(1, 32, 0.25, 20, 7);
  it('an outlier of 20 zeroes the small values of an MXFP4 block', () => {
    expect(quantizeBlock(v, MXFP4).zeroed).toBeGreaterThanOrEqual(25);
  });
  it('blocks of 16 keep the outlier-free half', () => {
    const q = quantizeBlock(v, { ...MXFP4, k: 16 });
    expect(q.blocks[1].zeroed).toBeLessThanOrEqual(2);
    expect(q.rmsRelRest).toBeLessThan(quantizeBlock(v, MXFP4).rmsRelRest);
  });
  it('E4M3 elements keep every small value', () => {
    expect(quantizeBlock(v, { ...MXFP4, elem: 'E4M3' }).zeroed).toBe(0);
  });
  it('named formats', () => {
    expect(namedFormat(MXFP4)).toBe('MXFP4');
    expect(namedFormat({ elem: 'E4M3', k: 32, scale: 'cublas-ceil' })).toBe('MXFP8');
    expect(namedFormat({ elem: 'E2M1', k: 16, scale: 'e4m3' })).toContain('16-element FP4');
    expect(namedFormat({ elem: 'E2M1', k: 16, scale: 'ocp-floor' })).toBeNull();
  });
  it('the E4M3 scale for 0.3s over E2M1 is 0.05078125 (13/256), an E4M3 value', () => {
    expect(blockScale(f32(0.3), E2M1, 'e4m3').scale).toBe(13 / 256);
  });
});

describe('storage (MX v1.1 Table 1, cuBLAS, llama.cpp ggml-common.h: appA-facts A4, A5, D3)', () => {
  it('bits per weight: FP32 32, BF16 16, FP8 8, MXFP8 8.25, MXFP4 4.25, NVIDIA FP4 4.5', () => {
    expect(['fp32', 'bf16', 'fp8', 'mxfp8', 'mxfp4', 'nvfp4'].map((f) => bitsPerWeight(f as never))).toEqual([32, 16, 8, 8.25, 4.25, 4.5]);
  });
  it('llama.cpp: block_mxfp4 is 17 bytes per 32 weights; block_nvfp4 is 36 bytes per 64', () => {
    expect(blockBytes(4, 32, 1)).toBe(17);
    expect(blockBytes(4, 64, 4)).toBe(36);
    expect((blockBytes(4, 32, 1) * 8) / 32).toBe(bitsPerWeight('mxfp4'));
    expect((blockBytes(4, 64, 4) * 8) / 64).toBe(bitsPerWeight('nvfp4'));
    expect(blockBytes(16, 32, 0)).toBe(64); // the same 32 weights in BF16
  });
});

describe('the decode ceiling (ch. 5 §5.1 method; ours)', () => {
  it('q-ceiling: BF16 to MXFP4 raises the batch-1 ceiling by 16 / 4.25 ≈ 3.76', () => {
    const r = decodeCeiling(7e9, bitsPerWeight('mxfp4'), 1000) / decodeCeiling(7e9, bitsPerWeight('bf16'), 1000);
    expect(r).toBeCloseTo(16 / 4.25, 12);
    expect(r).toBeCloseTo(3.76, 2);
  });
  it('2 operations per weight: 1 FLOP/B in 16-bit (chapter 5\'s figure), 0.5 in FP32', () => {
    expect(opsPerByte(16)).toBe(1);
    expect(opsPerByte(32)).toBe(0.5);
    expect(opsPerByte(4.25)).toBeCloseTo(3.7647, 4);
  });
  it('7 billion BF16 weights at the book\'s example 3.35 TB/s: about 239 tokens per second at most', () => {
    expect(decodeCeiling(7e9, 16, 3350)).toBeCloseTo(239.29, 2);
  });
});

describe('q-fnuz: OCP E4M3 bytes read as CDNA3 FNUZ (appA-facts B2)', () => {
  it('every finite non-zero value halves; 0x80 (OCP -0) is NaN; OCP\'s NaN bytes are finite', () => {
    for (let c = 0; c < 256; c++) {
      const ocp = decode(E4M3, c).value;
      const fnuz = decode(E4M3FNUZ, c).value;
      if (c === 0x80) {
        expect(Object.is(ocp, -0)).toBe(true);
        expect(fnuz).toBeNaN();
      } else if (Number.isNaN(ocp)) expect(Math.abs(fnuz)).toBe(240);
      else expect(fnuz).toBe(ocp / 2);
    }
  });
});

describe('the scene\'s block (content/appA/narration/mx-block-*.txt quote these numbers)', () => {
  const v = sceneValues();
  const q = quantizeBlock(v, MXFP4);
  const script = (n: number) => fs.readFileSync(path.join(__dirname, `../../../content/appA/narration/mx-block-${n}.txt`), 'utf8');
  it('the outlier is 12 at element 7; floor(log2 12) = 3; minus 2 gives X = 2, UE8M0 byte 128', () => {
    expect(q.blocks[0]).toMatchObject({ amax: SCENE_BLOCK.outlier, amaxAt: SCENE_BLOCK.at, scaleExp: 1, scale: 2, scaleCode: 128 });
    expect(script(1)).toMatch(/Element 7 is 12/);
    expect(script(3)).toMatch(/logarithm of 12, rounded down, is 3/);
    expect(script(3)).toMatch(/2 to the power 1, which is 2/);
  });
  it('dividing by 2 puts the outlier exactly on 6, and all but one other weight under 0.25', () => {
    expect(q.scaled[SCENE_BLOCK.at]).toBe(6);
    expect(q.scaled.filter((t, i) => i !== SCENE_BLOCK.at && Math.abs(t) > 0.25).length).toBe(1);
    expect(script(4)).toMatch(/lands exactly on 6/);
  });
  it('the max reduction takes 5 rounds of 16, 8, 4, 2, 1 survivors and ends on the outlier', () => {
    const t = maxTree(v);
    expect(t.map((l) => l.length)).toEqual([32, 16, 8, 4, 2, 1]);
    expect(t[5]).toEqual([SCENE_BLOCK.at]);
    expect(t[1]).toContain(SCENE_BLOCK.at);
  });
  it('30 of the 31 other weights come back as zero', () => {
    expect(q.zeroed).toBe(30);
    expect(script(6)).toMatch(/30 of the 31 small weights came back as 0/);
  });
  it('in blocks of 16, the half without the outlier loses none', () => {
    const h = quantizeBlock(v, { ...MXFP4, k: 16 });
    expect(h.blocks.map((b) => b.zeroed)).toEqual([14, 0]);
    expect(script(6)).toMatch(/loses none/);
  });
  it('17 bytes against 64 in BF16; the 5 rounds of a 32-element max reduction', () => {
    expect(blockBytes(4, 32, 1)).toBe(17);
    expect(blockBytes(16, 32, 0)).toBe(64);
    expect(script(5)).toMatch(/17 bytes/);
    expect(script(5)).toMatch(/64 bytes/);
    expect(Math.log2(32)).toBe(5);
    expect(script(2)).toMatch(/after 5 rounds/);
  });
  it('every script is 35 to 60 words with no symbols read aloud', () => {
    for (let n = 1; n <= 6; n++) {
      const s = script(n).replace(/\[[^\]]*\]\s*/g, '').trim();
      const w = s.split(/\s+/).length;
      expect(w, `mx-block-${n}`).toBeGreaterThanOrEqual(35);
      expect(w, `mx-block-${n}`).toBeLessThanOrEqual(60);
      expect(s, `mx-block-${n}`).not.toMatch(/[<>=×%/_(){}^*+.](?=\S)/);
    }
  });
});

// The C++ the scene shows (labs/appA/snippets/mx_quantize.h), compiled and run, must give
// the model's scale byte, codes and decoded floats bit for bit. Skipped with no compiler.
const repo = path.resolve(__dirname, '../../..');
const compiler = ['c++', 'clang++', 'g++'].find((c) => spawnSync(c, ['--version']).status === 0);
describe.skipIf(!compiler)('labs/appA/snippets/mx_quantize.h, compiled and run', () => {
  const exe = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'appA-mx-')), 'mx_quantize_test');
  const src = path.join(repo, 'labs/appA/snippets/check/mx_quantize_test.cpp');
  const bits = (x: number) => {
    const b = new DataView(new ArrayBuffer(4));
    b.setFloat32(0, x);
    return b.getUint32(0).toString(16).padStart(8, '0');
  };
  it('builds without warnings', () => {
    const r = spawnSync(compiler!, ['-std=c++17', '-Wall', '-Wextra', '-O2', src, '-o', exe], { encoding: 'utf8' });
    expect(r.status, r.stderr).toBe(0);
    expect(r.stderr).toBe('');
  });
  const outlier = Array(32).fill(f32(0.3));
  outlier[5] = 100;
  const blocks: [string, number[]][] = [
    ['the scene\'s block', sceneValues()],
    ['q-outlier', outlier],
    ['all zero', Array(32).fill(0)],
    ['maximum exactly 8', [8, ...Array(31).fill(1)]],
    ['ties', [4, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, -0.25, -5, ...Array(22).fill(f32(-0.01))]],
    ...[1, 2, 3, 4, 5, 6, 7, 8].map((s) => [`seed ${s}`, sampleBlock(s, 32, 0.25, s % 3 ? 3 * s : 0, s)] as [string, number[]]),
    ['large and tiny', sampleBlock(9, 32, 3000)],
    ['subnormal floats', sampleBlock(10, 32, 1e-39).map(f32)],
  ];
  for (const [name, v] of blocks)
    it(`${name}: same scale byte, codes and decoded floats as quantizeBlock; 17 bytes`, () => {
      const r = spawnSync(exe, v.map(String), { encoding: 'utf8' });
      expect(r.status, r.stderr).toBe(0);
      const out = r.stdout.trim().split('\n');
      const q = quantizeBlock(v, MXFP4);
      expect(Number(out[0])).toBe(q.blocks[0].scaleCode);
      expect(out.slice(1, 33).map(Number)).toEqual(q.codes);
      expect(out.slice(33, 65)).toEqual(q.decoded.map(bits));
      expect(out[65]).toBe('17');
    });
});
