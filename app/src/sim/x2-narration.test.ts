import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { BOUND_Q8Q8, INT16_MAX, FP16_MAX, integerSceneRow, narrowScene, rowErrors } from './qdot';
import { MIRROR_AT, OUTLIER_AT, SCENE_BLOCK, blockDotQ4_0, fp16FromBits, quantizeQ4_0, sceneActivations } from './quant';
import { bytesPerWeight, layer, pipelineRelative } from './qtraffic';

// Module X2's narration scripts (content/x2/narration/*.txt) speak numbers that the scenes
// compute from the tested models. The browser checks only count words, so this test ties
// every number a script says to the model it comes from: a change to a seeded block or a
// model that would make the narration wrong fails here.
const repo = path.resolve(__dirname, '../../..');
const script = (clip: string) =>
  fs.readFileSync(path.join(repo, 'content/x2/narration', clip + '.txt'), 'utf8').replace(/\[[^\]]*\]\s*/g, '').trim();
/** The number as the script writes it: digits with thousands commas. */
const spoken = (v: number) => v.toLocaleString('en-US');

describe('packed-block: one Q4_0 block (quant.ts, the scene block)', () => {
  const block = quantizeQ4_0(SCENE_BLOCK);
  const y = sceneActivations();
  const dot = blockDotQ4_0(block, y, SCENE_BLOCK);
  const d16 = fp16FromBits(block.dBits);

  it('beat 1: weight 5 at minus 0.8 is the extreme; d is exactly 0.1, stored in FP16 a hair under', () => {
    const s = script('packed-block-1');
    expect(s).toContain('weight 5, at minus 0.8');
    expect(OUTLIER_AT).toBe(5);
    expect(SCENE_BLOCK[OUTLIER_AT]).toBe(Math.fround(-0.8));
    expect(Math.max(...SCENE_BLOCK.map(Math.abs))).toBe(Math.abs(SCENE_BLOCK[OUTLIER_AT]));
    expect(s).toContain('exactly 0.1');
    expect(block.d).toBe(Math.fround(Math.fround(-0.8) / -8));
    expect(Math.abs(block.d - 0.1)).toBeLessThan(1e-8);
    expect(s).toContain('a hair under 0.1');
    expect(d16).toBeLessThan(0.1);
    expect(0.1 - d16).toBeLessThan(1e-4);
  });

  it('beat 2: weight 5 lands on code 0; its mirror, weight 21 at plus 0.8, is clipped to 15', () => {
    const s = script('packed-block-2');
    expect(s).toContain('Weight 5, the extreme, lands exactly on code 0');
    expect(block.codes[OUTLIER_AT]).toBe(0);
    expect(s).toContain('Weight 21, its mirror at plus 0.8, would need code 16');
    expect(MIRROR_AT).toBe(21);
    expect(SCENE_BLOCK[MIRROR_AT]).toBe(Math.fround(0.8));
    expect(Math.trunc(SCENE_BLOCK[MIRROR_AT] / block.d + 8.5)).toBe(16);
    expect(block.codes[MIRROR_AT]).toBe(15);
  });

  it('beat 3: weights 5 and 21 share byte 5; 18 bytes against 128', () => {
    const s = script('packed-block-3');
    expect(s).toContain('weights 5 and 21 share byte 5');
    const byte5 = block.bytes[2 + 5];
    expect(byte5 & 15).toBe(block.codes[5]);
    expect(byte5 >> 4).toBe(block.codes[21]);
    expect(s).toContain('18 bytes for the whole block, where float32 needed 128');
    expect(block.bytes.length).toBe(18);
  });

  it('beat 6: the folded dot is 0.716, as with the dequantised weights; the float32 weights give 0.605', () => {
    const s = script('packed-block-6');
    expect(s).toContain('That gives 0.716');
    expect(dot.folded.toFixed(3)).toBe('0.716');
    expect(dot.dequantised.toFixed(3)).toBe('0.716');
    expect(Math.abs(dot.folded - dot.dequantised)).toBeLessThan(1e-5);
    expect(s).toContain('The float32 weights give 0.605');
    expect(dot.exact.toFixed(3)).toBe('0.605');
  });
});

describe('narrow-accumulator: the worst-case Q8_0 block and a 4,096-wide row (qdot.ts narrowScene)', () => {
  const S = narrowScene();
  const P = 127 * 127;

  it('beat 1: every product is 16,129, thirty-two of them', () => {
    const s = script('narrow-accumulator-1');
    expect(s).toContain(`every product is ${spoken(P)}`);
    expect(s).toContain('Thirty-two of them');
    expect(S.traces.int32.values.length).toBe(32);
  });

  it('beat 2: int16 holds 32,258 after 2 products and wraps on the third', () => {
    const s = script('narrow-accumulator-2');
    expect(s).toContain(`After 2 products it holds ${spoken(2 * P)}`);
    expect(S.traces.int16.values[1]).toBe(32258);
    expect(s).toContain(`past ${spoken(INT16_MAX)}`);
    expect(s).toContain('The third product');
    expect(S.traces.int16.first.wrap).toBe(3);
    expect(S.traces.int16.values[2]).toBeLessThan(0);
  });

  it('beat 3: half stores 16,129 as 16,128, reaches 64,512 at the fourth product and infinity at the fifth', () => {
    const s = script('narrow-accumulator-3');
    expect(s).toContain('16,129 becomes 16,128');
    expect(S.traces.fp16.values[0]).toBe(16128);
    expect(s).toContain('The fourth product brings the sum to 64,512');
    expect(S.traces.fp16.values[3]).toBe(64512);
    expect(s).toContain(`The fifth passes ${spoken(FP16_MAX)}`);
    expect(S.traces.fp16.first.inf).toBe(5);
    expect(S.traces.fp16.values.slice(4).every((v) => v === Infinity)).toBe(true);
  });

  it('beat 4: int32 and float reach 516,128 exactly, below 2 to the 24', () => {
    const s = script('narrow-accumulator-4');
    expect(s).toContain(`to ${spoken(BOUND_Q8Q8)}`);
    expect(BOUND_Q8Q8).toBe(516128);
    expect(S.traces.int32.values[31]).toBe(BOUND_Q8Q8);
    expect(S.traces.fp32.values[31]).toBe(BOUND_Q8Q8);
    expect(BOUND_Q8Q8).toBeLessThan(2 ** 24);
    expect(s).toContain('about 2 billion');
  });

  it('beat 5: over 128 blocks, a half outer sum errs thousands of times more than a float one', () => {
    const s = script('narrow-accumulator-5');
    expect(s).toContain('4,096 values: 128 blocks');
    expect(s).toContain('thousands of times');
    const ratio = S.outerHalf[127] / S.outerFloat[127];
    expect(ratio).toBeGreaterThan(1000);
    expect(ratio).toBeLessThan(1e6);
    expect(S.half.absErr / S.float.absErr).toBeGreaterThan(1000);
  });
});

describe('integer-path: the scene row (qdot.ts integerSceneRow) and the explorer\'s median', () => {
  it('beat 5: the gap from rounding activations to bytes is a fraction of a percent', () => {
    const s = script('integer-path-5');
    expect(s).toContain('a fraction of a percent');
    const row = integerSceneRow();
    const gap = Math.abs(row.outC - row.outB) / Math.abs(row.outB);
    expect(gap).toBeGreaterThan(0);
    expect(gap).toBeLessThan(0.01);
    expect(rowErrors(4096, 'normal').median).toBeLessThan(0.01);
  });

  it('beats 1 to 4: the fixed numbers of the integer path', () => {
    expect(script('integer-path-1')).toContain('divided by 127');
    expect(script('integer-path-2')).toContain('One 32-bit word holds 8 four-bit codes');
    expect(script('integer-path-3')).toContain('The block needs 8 of them');
  });
});

describe('three-traffics: bytes through one pipe (qtraffic.ts, the scene\'s layer)', () => {
  const L = layer('4096x4096');
  const unfused = pipelineRelative('unfused-f32', 'q4_0', L, 1, 4);
  const fused = pipelineRelative('fused', 'q4_0', L, 1, 4);

  it('beat 2: Q4_0 is 4 and a half bits per weight, about one seventh of float32', () => {
    expect(script('three-traffics-2')).toContain('4 and a half bits per weight');
    expect(bytesPerWeight('q4_0') * 8).toBe(4.5);
    expect(script('three-traffics-2')).toContain('about one seventh');
    expect(Math.round(4 / bytesPerWeight('q4_0'))).toBe(7);
  });

  it('beats 3 to 5: unfused a little over 2, written as 2.1; fused about 7 times faster', () => {
    expect(script('three-traffics-3')).toContain('a little over 2');
    expect(unfused).toBeGreaterThan(2);
    expect(unfused).toBeLessThan(2.2);
    expect(script('three-traffics-5')).toContain('about 2.1 times');
    expect(unfused.toFixed(1)).toBe('2.1');
    expect(script('three-traffics-4')).toContain('about one seventh');
    expect(script('three-traffics-5')).toContain('about 7 times faster');
    expect(Math.round(1 / fused)).toBe(7);
  });
});
