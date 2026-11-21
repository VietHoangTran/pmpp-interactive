// Module X2 keeps two format tables: qformats.ts (sections X2.2, X2.3 and X2.7) and the one in
// qtraffic.ts (sections X2.1 and X2.4, written before qformats.ts existed). This test ties them
// together so that a change to one fails until the other follows.
import { describe, expect, it } from 'vitest';
import * as A from './qformats';
import * as T from './qtraffic';

const PAIRS: [T.FormatId, A.WeightFormat][] = [
  ['f32', A.F32],
  ['f16', A.F16],
  ['q8_0', A.Q8_0],
  ['q4_0', A.Q4_0],
  ['q4_1', A.Q4_1],
  ['q4_k', A.Q4_K],
  ['mxfp4', A.MXFP4],
  ['mlx4_g32', A.mlxAffine(4, 32, 'f16')],
  ['mlx4_g64', A.mlxAffine(4, 64, 'f16')],
  ['mlx4_g128', A.mlxAffine(4, 128, 'f16')],
  ['mlx4_g32_f32', A.mlxAffine(4, 32, 'f32')],
  ['mlx4_g64_f32', A.mlxAffine(4, 64, 'f32')],
  ['mlx4_g128_f32', A.mlxAffine(4, 128, 'f32')],
];

describe('the two X2 format tables agree', () => {
  it('covers every format of qtraffic.ts', () => {
    expect(PAIRS.map(([id]) => id).sort()).toEqual([...T.FORMAT_IDS].sort());
  });
  for (const [id, f] of PAIRS)
    it(`${id}: same block, block bytes and bytes per weight`, () => {
      expect(T.FORMATS[id].blockWeights).toBe(f.block);
      expect(T.FORMATS[id].blockBytes).toBe(f.blockBytes);
      expect(T.bytesPerWeight(id)).toBe(A.bytesPerWeight(f));
      expect(T.bitsPerWeight(id)).toBe(A.bitsPerWeight(f));
      expect(T.weightOnlyIntensity(id)).toBe(A.gemvIntensity(f));
    });
});
