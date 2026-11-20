import { describe, expect, it } from 'vitest';
import {
  F32,
  FORMATS,
  Q4_0,
  Q4_1,
  Q4_K,
  Q8_0,
  bitsPerWeight,
  bytesPerBlock,
  bytesPerWeight,
  formatById,
  gemvIntensity,
  matrixBytes,
  mlxAffine,
} from './qformats';

// notes/today/x2-facts.md E1, row by row: block, bytes per block, bytes per weight, bits
// per weight, GEMV FLOP/B (as printed, to the digits printed).
const E1: [string, number, number, number, number, number][] = [
  ['f32', 1, 4, 4, 32, 0.5],
  ['f16', 1, 2, 2, 16, 1],
  ['bf16', 1, 2, 2, 16, 1],
  ['q8_0', 32, 34, 1.0625, 8.5, 1.88],
  ['q4_0', 32, 18, 0.5625, 4.5, 3.56],
  ['q4_1', 32, 20, 0.625, 5.0, 3.2],
  ['q4_k', 256, 144, 0.5625, 4.5, 3.56],
  ['mxfp4', 32, 17, 0.53125, 4.25, 3.76],
  ['mlx-4-g32-f16', 32, 20, 0.625, 5.0, 3.2],
  ['mlx-4-g64-f16', 64, 36, 0.5625, 4.5, 3.56],
  ['mlx-4-g128-f16', 128, 68, 0.53125, 4.25, 3.76],
  ['mlx-4-g64-f32', 64, 40, 0.625, 5.0, 3.2],
];

describe('the format table is x2-facts E1', () => {
  it('has every row of E1, in order', () => {
    expect(FORMATS.map((f) => f.id)).toEqual(E1.map((r) => r[0]));
  });
  for (const [id, block, bb, bpw, bits, flopb] of E1)
    it(`${id}: ${bb} bytes per ${block}, ${bits} bits per weight, ${flopb} FLOP/B`, () => {
      const f = formatById(id);
      expect(f.block).toBe(block);
      expect(bytesPerBlock(f)).toBe(bb);
      expect(bytesPerWeight(f)).toBe(bpw);
      expect(bitsPerWeight(f)).toBe(bits);
      const digits = String(flopb).split('.')[1]?.length ?? 0;
      expect(Number(gemvIntensity(f).toFixed(digits))).toBe(flopb);
    });
});

describe('the predicted speed-up over FP32 is the byte ratio (E1)', () => {
  it('3.76 for Q8_0 and 7.11 for Q4_0', () => {
    expect((bytesPerWeight(F32) / bytesPerWeight(Q8_0)).toFixed(2)).toBe('3.76');
    expect((bytesPerWeight(F32) / bytesPerWeight(Q4_0)).toFixed(2)).toBe('7.11');
  });
});

describe('the numbers section X2.2 states', () => {
  it('q-bits: MLX 4-bit group 64 is 4.5 bits with FP16 scale and bias, 5.0 with float32', () => {
    expect(bitsPerWeight(mlxAffine(4, 64, 'f16'))).toBe(4.5);
    expect(bitsPerWeight(mlxAffine(4, 64, 'f32'))).toBe(5);
  });
  it('q-q4k: Q4_K is 4.5 bits, Q4_1 (a scale and a min per 32 as well) is 5.0', () => {
    expect(bitsPerWeight(Q4_K)).toBe(4.5);
    expect(bitsPerWeight(Q4_1)).toBe(5);
    // 2 + 2 + 12 + 128: two FP16 super-scales, 8 six-bit scales and 8 six-bit mins, 256 nibbles.
    expect(2 + 2 + 12 + 256 / 2).toBe(Q4_K.blockBytes);
    expect((8 * 6 + 8 * 6) / 8).toBe(12);
  });
  it('the predict on spending a bit: group 32 costs 5.0 bits, 5 bits at group 64 costs 5.5', () => {
    expect(bitsPerWeight(mlxAffine(4, 32))).toBe(5);
    expect(bitsPerWeight(mlxAffine(5, 64))).toBe(5.5);
  });
});

describe('MLX limits (D5) and matrices', () => {
  it('rejects 7 bits and group 16', () => {
    expect(() => mlxAffine(7, 64)).toThrow();
    expect(() => mlxAffine(4, 16)).toThrow();
  });
  it('every allowed width packs a group into whole bytes', () => {
    for (const b of [1, 2, 3, 4, 5, 6, 8]) for (const g of [32, 64, 128]) expect(Number.isInteger(mlxAffine(b, g).blockBytes)).toBe(true);
  });
  it('a 4096 x 4096 matrix: 67,108,864 bytes in FP32, 9,437,184 in Q4_0', () => {
    expect(matrixBytes(F32, 4096, 4096)).toBe(67108864);
    expect(matrixBytes(Q4_0, 4096, 4096)).toBe(9437184);
    expect(() => matrixBytes(Q4_0, 1, 48)).toThrow();
  });
});
