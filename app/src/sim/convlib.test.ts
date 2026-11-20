import { describe, expect, it } from 'vitest';
import {
  APPLE_FWD_ALGOS, CARDS, CUDNN_FWD_ALGOS, DISPATCH_PRESETS, INT32_MAX, LIMITS, METAL_BUFFER_FLOOR, MIOPEN_FWD_ALGOS,
  NAMES, NO_COUNTERPART, PRESETS, bookGemm, chapterOutputDim, clampField, cudnnValues, duplication, expectedAnswer,
  flops, grade, libraryGemm, llamaCuda, mlx, offset, outputDim, outputDimGgml, outputDimSteps, outputDimTorch, phases,
  printedPhases, shape, slow2d, span, toChapter, toCudnn, unfoldSize, unfoldedPerSample, type Layer,
} from './convlib';

const L = (over: Partial<Layer>): Layer => ({ N: 1, C: 1, H: 32, W: 32, M: 6, K: 5, pad: 0, stride: 1, dilation: 1, ...over });
const preset = (id: string) => PRESETS.find((p) => p.id === id)!.layer;

describe('output size: cuDNN\'s formula (A2)', () => {
  it('H 224, R 7, stride 2, pad 3 gives P = 112 (the lesson\'s prediction)', () => {
    expect(outputDim(224, 7, 3, 2, 1)).toBe(112);
    expect(outputDimSteps(224, 7, 3, 2, 1)).toEqual({ span: 7, room: 223, quotient: 111, result: 112 });
    expect(shape(preset('stem'))).toMatchObject({ P: 112, Q: 112, valid: true, chapterApplies: false });
  });
  it('pad 0, stride 1, dilation 1 is the chapter\'s H − K + 1', () => {
    for (const H of [3, 14, 28, 32, 224, 1000])
      for (const K of [1, 2, 3, 5, 7])
        if (K <= H) expect(outputDim(H, K)).toBe(chapterOutputDim(H, K));
    expect(shape(preset('lenet-c1'))).toMatchObject({ P: 28, Q: 28, chapterApplies: true });
    expect(shape(preset('fig19-2'))).toMatchObject({ P: 2, Q: 2 });
  });
  it('dilation 2 with R 3 and pad 1 on 224 gives 222: the taps cover 5 rows', () => {
    expect(span(3, 2)).toBe(5);
    expect(outputDim(224, 3, 1, 1, 2)).toBe(222);
    expect(shape(preset('dilated')).P).toBe(222);
  });
  it('padding by (K − 1)/2 keeps the size; padding changes the output size', () => {
    expect(outputDim(56, 3, 1)).toBe(56);
    expect(outputDim(56, 3, 0)).toBe(54);
  });
  it('a vision patch embedding: 16 × 16 with stride 16 on 224 gives 14 (E1)', () => {
    expect(shape(preset('patch'))).toMatchObject({ P: 14, Q: 14 });
    expect(outputDim(224, 14, 0, 14)).toBe(16);
  });
  it('PyTorch\'s and llama.cpp\'s spellings agree with cuDNN\'s wherever the filter fits', () => {
    let n = 0;
    for (let input = 1; input <= 40; input++)
      for (let K = 1; K <= 7; K++)
        for (let pad = 0; pad <= 3; pad++)
          for (let stride = 1; stride <= 4; stride++)
            for (let d = 1; d <= 3; d++) {
              if (span(K, d) > input + 2 * pad) {
                expect(outputDim(input, K, pad, stride, d)).toBe(0);
                continue;
              }
              const want = outputDim(input, K, pad, stride, d);
              expect(outputDimTorch(input, K, pad, stride, d)).toBe(want);
              expect(outputDimGgml(input, K, pad, stride, d)).toBe(want);
              n++;
            }
    expect(n).toBeGreaterThan(5000);
  });
  it('a filter wider than the padded input gives no output', () => {
    expect(shape(L({ H: 4, W: 4, K: 5 })).valid).toBe(false);
    expect(shape(L({ H: 4, W: 4, K: 5, pad: 1 })).valid).toBe(true);
  });
});

describe('GEMM views', () => {
  it('book orientation for LeNet C1: F 6 × 25 times B 25 × 784, once per sample (notes, derived; E1)', () => {
    expect(bookGemm(L({ N: 4 }))).toEqual({ rows: 6, inner: 25, cols: 784, count: 4 });
  });
  it('LeNet C3 and C5 as the notes derive them', () => {
    expect(bookGemm(preset('lenet-c3'))).toEqual({ rows: 16, inner: 150, cols: 100, count: 1 });
    expect(bookGemm(L({ C: 16, H: 5, W: 5, M: 120 }))).toEqual({ rows: 120, inner: 400, cols: 1, count: 1 });
  });
  it('library orientation (CUTLASS, A7): N·P·Q × C·R·S times C·R·S × K', () => {
    expect(libraryGemm(L({ N: 4 }))).toEqual({ rows: 4 * 784, inner: 25, cols: 6, count: 1 });
  });
  it('both orientations do the same work: 2·N·M·P·Q·C·K·K', () => {
    const l = preset('same3');
    const b = bookGemm(l);
    const g = libraryGemm(l);
    expect(2 * b.rows * b.inner * b.cols * b.count).toBe(flops(l));
    expect(2 * g.rows * g.inner * g.cols * g.count).toBe(flops(l));
  });
});

describe('the unfolded input (Eq. 19.4, E1, E2)', () => {
  it('the book\'s example: 48 elements of B from 27 inputs, 16/9', () => {
    const l = preset('fig19-2');
    expect(unfoldedPerSample(l)).toBe(48);
    expect(duplication(l)).toBeCloseTo(16 / 9, 12);
  });
  it('LeNet C1 ≈ 19.1 and C3 ≈ 12.8 (notes, derived)', () => {
    expect(duplication(preset('lenet-c1'))).toBeCloseTo(19.140625, 9);
    expect(duplication(preset('lenet-c3'))).toBeCloseTo(12.755, 3);
  });
  it('stride equal to the filter removes all duplication when K divides the image (E1)', () => {
    expect(duplication(preset('patch'))).toBe(1);
    expect(duplication(L({ C: 3, H: 224, W: 224, K: 14, stride: 14 }))).toBe(1);
    // and skips pixels when it does not divide: 6 windows of 5 cover 30 of 32 rows
    expect(duplication(L({ H: 32, W: 32, K: 5, stride: 5 }))).toBeCloseTo((25 * 36) / 1024, 12);
  });
  it('K = 1 is exactly 1: B is X itself', () => {
    expect(duplication(preset('pointwise'))).toBe(1);
  });
  it('at stride 1 the ratio approaches K² for large maps (p. 467; E1: 8.84 for K 3 at 224)', () => {
    expect(duplication(L({ H: 224, W: 224, K: 3 }))).toBeCloseTo(8.84, 2);
    expect(duplication(L({ H: 1024, W: 1024, K: 5 }))).toBeCloseTo(24.81, 2);
  });
  it('sizes against the limits E2 compares with', () => {
    const one = unfoldSize(L({ C: 64, H: 224, W: 224, K: 3 }));
    expect(one.batch).toBe(28_387_584);
    expect(one.overMetalFloor).toBe(false);
    const eight = unfoldSize(L({ N: 8, C: 64, H: 224, W: 224, K: 3 }));
    expect(eight.batch).toBe(227_100_672);
    expect(eight.batchBytes).toBeGreaterThan(METAL_BUFFER_FLOOR);
    expect(eight.overInt32).toBe(false);
    // E2: a 64-channel 3 × 3 layer passes 2^31 − 1 elements at a 1933 × 1933 input
    expect(unfoldSize(L({ C: 64, H: 1933, W: 1933, K: 3 })).overInt32).toBe(true);
    expect(unfoldSize(L({ C: 64, H: 1932, W: 1932, K: 3 })).overInt32).toBe(false);
    expect(INT32_MAX).toBe(2147483647);
  });
});

describe('layouts and the start page\'s arithmetic', () => {
  it('X[1][2][5][7] of an N × 3 × 32 × 32 tensor is at 5287 in NCHW', () => {
    expect(offset('NCHW', { C: 3, H: 32, W: 32 }, 1, 2, 5, 7)).toBe(5287);
    expect(offset('NHWC', { C: 3, H: 32, W: 32 }, 1, 2, 5, 7)).toBe(3575);
  });
  it('both layouts are a bijection on a small tensor', () => {
    const dims = { C: 3, H: 4, W: 5 };
    for (const layout of ['NCHW', 'NHWC'] as const) {
      const seen = new Set<number>();
      for (let n = 0; n < 2; n++) for (let c = 0; c < 3; c++) for (let h = 0; h < 4; h++) for (let w = 0; w < 5; w++) seen.add(offset(layout, dims, n, c, h, w));
      expect([...seen].sort((a, b) => a - b)).toEqual(Array.from({ length: 120 }, (_, i) => i));
    }
  });
  it('a 6 × 25 by 25 × 784 GEMM with T = 16 needs 2 phases; Fig. 19.11 as printed runs 1 and drops 9 terms (E1)', () => {
    expect(phases(25, 16)).toBe(2);
    expect(printedPhases(25, 16)).toBe(1);
    expect(25 - printedPhases(25, 16) * 16).toBe(9);
    expect(phases(25, 16) * 16 - 25).toBe(7);
  });
  it('each element of the left operand is fetched once per block column: ceil(784/16) = 49 for LeNet C1', () => {
    expect(phases(784, 16)).toBe(49);
  });
});

describe('names: Table 19.1 both ways', () => {
  it('cuDNN\'s K is the chapter\'s M; R and S are the chapter\'s K', () => {
    expect(toChapter('K')).toBe('M');
    expect(toChapter('R')).toBe('K');
    expect(toChapter('S')).toBe('K');
    expect(toCudnn('M')).toEqual(['K']);
    expect(toCudnn('K')).toEqual(['R', 'S']);
  });
  it('every chapter name round-trips', () => {
    for (const r of NAMES) {
      if (r.chapter == null) {
        expect(r.none).toBeTruthy();
        continue;
      }
      expect(toCudnn(r.chapter)).toContain(r.cudnn);
      expect(toChapter(r.cudnn)).toBe(r.chapter);
    }
  });
  it('Table 19.1 has its eleven parameters', () => {
    expect(NAMES.filter((r) => r.from === 'Table 19.1').map((r) => r.cudnn)).toEqual(['N', 'C', 'H', 'W', 'K', 'R', 'S', 'u', 'v', 'pad_h', 'pad_w']);
  });
  it('values for a layer: the q-names layer is K 64, R = S = 3, C 32 in cuDNN\'s names', () => {
    const v = cudnnValues(L({ C: 32, M: 64, K: 3, H: 56, W: 56, pad: 1 }));
    expect(v).toMatchObject({ K: 64, R: 3, S: 3, C: 32, pad_h: 1, P: 56, Q: 56 });
  });
  it('every flashcard\'s answer comes from the table and is among its options', () => {
    for (const c of CARDS) {
      expect(c.answer).toBe(expectedAnswer(c));
      expect(c.options).toContain(c.answer);
      expect(new Set(c.options).size).toBe(c.options.length);
      expect(grade(c, c.answer)).toBe(true);
      expect(c.options.filter((o) => grade(c, o))).toHaveLength(1);
    }
    expect(CARDS.find((c) => c.id === 'u')!.answer).toBe(NO_COUNTERPART);
  });
});

describe('algorithm tables (A3, B3, B4, C3 to C5)', () => {
  it('cuDNN has eight forward algorithms, MIOpen five, Apple exposes none', () => {
    expect(CUDNN_FWD_ALGOS).toHaveLength(8);
    expect(MIOPEN_FWD_ALGOS.map((a) => a.value)).toEqual([0, 1, 2, 3, 5]);
    expect(APPLE_FWD_ALGOS).toHaveLength(0);
  });
  it('HIPIFY maps five of cuDNN\'s eight and not PRECOMP, FFT_TILING or WINOGRAD_NONFUSED (B4)', () => {
    const unmapped = CUDNN_FWD_ALGOS.filter((a) => a.hipify == null).map((a) => a.name.replace('CUDNN_CONVOLUTION_FWD_ALGO_', ''));
    expect(unmapped).toEqual(['IMPLICIT_PRECOMP_GEMM', 'FFT_TILING', 'WINOGRAD_NONFUSED']);
    const targets = new Set(MIOPEN_FWD_ALGOS.map((a) => a.name));
    for (const a of CUDNN_FWD_ALGOS) if (a.hipify) expect(targets.has(a.hipify)).toBe(true);
  });
});

describe('dispatch, as read on 2026-11-20', () => {
  it('llama.cpp CUDA: a 1 × 1, stride 1, unpadded layer goes to cuBLAS (D1)', () => {
    expect(llamaCuda(preset('pointwise')).path).toBe('cublas');
  });
  it('llama.cpp CUDA: LeNet C1 is implicit GEMM: C·K·K = 25 is short, but P·Q = 784 is not below 512', () => {
    const c = llamaCuda(preset('lenet-c1'));
    expect(c.path).toBe('implicit');
    expect(c.checks.find((x) => x.label.includes('C·K·K'))!).toMatchObject({ shown: '1·5·5 = 25', holds: true });
    expect(c.checks.find((x) => x.label.includes('P·Q'))!).toMatchObject({ shown: '28·28 = 784', holds: false });
  });
  it('llama.cpp CUDA: C 2, 3 × 3, 16 × 16 output is the direct kernel', () => {
    const l = L({ C: 2, H: 18, W: 18, K: 3, M: 8 });
    expect(shape(l)).toMatchObject({ P: 16, Q: 16 });
    expect(llamaCuda(l).path).toBe('direct');
  });
  it('llama.cpp CUDA: a small 1 × 1 layer meets both conditions, and the explorer says it cannot tell', () => {
    expect(llamaCuda(L({ C: 3, H: 8, W: 8, K: 1 })).path).toBe('cublas-or-direct');
  });
  it('llama.cpp CUDA: a 1 × 1 layer with stride 2 is not pointwise', () => {
    expect(llamaCuda(L({ C: 256, H: 28, W: 28, K: 1, stride: 2 })).pointwise).toBe(false);
  });
  it('the three dispatch presets predict what the rule gives', () => {
    for (const p of DISPATCH_PRESETS) expect(llamaCuda(p.layer).path).toBe(p.expect);
  });
  it('PyTorch Slow2d: requires_columns is false only for 1 × 1, stride 1, unpadded (D4)', () => {
    expect(slow2d(preset('pointwise')).requiresColumns).toBe(false);
    expect(slow2d(preset('lenet-c1')).requiresColumns).toBe(true);
    expect(slow2d(L({ K: 1, pad: 1 })).requiresColumns).toBe(true);
    expect(slow2d(preset('dilated')).modelled).toBe(false);
  });
  it('MLX: Winograd only as a candidate for 3 × 3, stride 1, C and M multiples of 32 (D3)', () => {
    expect(mlx(preset('same3')).path).toBe('winograd-candidate');
    expect(mlx(L({ C: 48, M: 64, K: 3, H: 56, W: 56, pad: 1 })).path).toBe('gemm');
    expect(mlx(L({ C: 64, M: 64, K: 3, H: 56, W: 56, stride: 2 })).path).toBe('gemm');
    expect(mlx(preset('lenet-c1')).path).toBe('gemm');
    expect(mlx(preset('dilated')).dilationUnknown).toBe(true);
  });
});

describe('inputs', () => {
  it('clamps to integers inside the limits', () => {
    expect(clampField('K', 0)).toBe(1);
    expect(clampField('K', 3.6)).toBe(4);
    expect(clampField('pad', -2)).toBe(0);
    expect(clampField('stride', Number.NaN)).toBe(LIMITS.stride[0]);
    expect(clampField('N', 1e9)).toBe(LIMITS.N[1]);
  });
});
