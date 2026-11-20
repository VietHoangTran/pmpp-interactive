import { describe, expect, it } from 'vitest';
import {
  BOOK_F,
  BOOK_SHAPE,
  BOOK_X,
  BOOK_Y,
  INT32_MAX,
  METAL_MIN_MAX_BUFFER,
  MiB,
  UNFOLD_SCENE,
  bToX,
  ckk,
  convAsGemm,
  convDirect,
  copiesAfterColumns,
  copiesPerPixel,
  dotTerms,
  expansion,
  gemmShape,
  limitFlags,
  linear,
  makeData,
  matmul,
  outDims,
  pixels,
  preset,
  ratioAt,
  sizes,
  unfold,
  unfoldT,
  xToB,
  type ConvShape,
} from './unfold';

const r2 = (x: number) => Math.round(x * 100) / 100;
const col = (B: number[], P: number, v: number) => B.filter((_, i) => i % P === v);

describe("the book's example (Figs. 19.2 and 19.8)", () => {
  const B = unfold(BOOK_SHAPE, BOOK_X);
  it('B is 12 × 4 and matches Fig. 19.8 column by column', () => {
    expect(B.length).toBe(48);
    expect(col(B, 4, 0)).toEqual([1, 2, 1, 1, 0, 2, 0, 3, 1, 2, 0, 1]);
    expect(col(B, 4, 1)).toEqual([2, 0, 1, 3, 2, 1, 3, 2, 2, 1, 1, 3]);
    expect(col(B, 4, 2)).toEqual([1, 1, 0, 2, 0, 3, 1, 1, 0, 1, 3, 3]);
    expect(col(B, 4, 3)).toEqual([1, 3, 2, 2, 3, 2, 1, 0, 1, 3, 3, 2]);
  });
  it('F needs no copy: the M × C × K × K array read row-major is rows 1 1 2 2 | 1 1 1 1 | 0 1 1 0 and 1 0 0 1 | 2 1 2 1 | 1 2 2 0', () => {
    expect(BOOK_F.slice(0, 12)).toEqual([1, 1, 2, 2, 1, 1, 1, 1, 0, 1, 1, 0]);
    expect(BOOK_F.slice(12)).toEqual([1, 0, 0, 1, 2, 1, 2, 1, 1, 2, 2, 0]);
  });
  it('F · B = Y = 14 20 15 24 / 12 24 17 26, the same as the loop nest, already in row-major map order', () => {
    expect(matmul(BOOK_F, B, 2, 12, 4)).toEqual(BOOK_Y);
    expect(convDirect(BOOK_SHAPE, BOOK_X, BOOK_F)).toEqual(BOOK_Y);
    expect(convAsGemm(BOOK_SHAPE, BOOK_X, BOOK_F)).toEqual(BOOK_Y);
  });
  it('output (0, 0) is 7 + 5 + 2 = 14, one partial sum per input map (Eq. 19.1)', () => {
    const t = dotTerms(BOOK_SHAPE, BOOK_X, BOOK_F, 0, 0);
    expect(t.map((x) => x.x)).toEqual([1, 2, 1, 1, 0, 2, 0, 3, 1, 2, 0, 1]);
    expect(t.reduce((a, x) => a + x.x * x.f, 0)).toBe(14);
    expect(UNFOLD_SCENE.perChannel).toEqual([7, 5, 2]);
  });
  it('48 entries from 27 inputs: 16/9', () => {
    expect(expansion(BOOK_SHAPE)).toBeCloseTo(16 / 9, 12);
    expect([UNFOLD_SCENE.entries, UNFOLD_SCENE.inputs]).toEqual([48, 27]);
  });
  it('corners copied once, edges twice, the centre four times', () => {
    expect(copiesPerPixel(BOOK_SHAPE)).toEqual([1, 2, 1, 2, 4, 2, 1, 2, 1]);
    expect(UNFOLD_SCENE.copiesAfter[4]).toEqual([1, 2, 1, 2, 4, 2, 1, 2, 1]);
    expect(copiesAfterColumns(BOOK_SHAPE, 1)).toEqual([1, 1, 0, 1, 1, 0, 0, 0, 0]);
  });
  it('B[10, 2] → X[n, 2, 2, 0] = 3 (Fig. 19.10)', () => {
    const s = bToX(BOOK_SHAPE, 10, 2);
    expect([s.c, s.p, s.q, s.oh, s.ow, s.h, s.w]).toEqual([2, 1, 0, 1, 0, 2, 0]);
    expect(BOOK_X[s.flat]).toBe(3);
  });
  it('every B entry maps back through xToB to the entry it came from', () => {
    for (let u = 0; u < 12; u++)
      for (let v = 0; v < 4; v++) {
        const s = bToX(BOOK_SHAPE, u, v);
        expect(xToB(BOOK_SHAPE, s.c, s.h, s.w).some((e) => e.u === u && e.v === v)).toBe(true);
      }
  });
});

describe('Eq. 19.5 on C = 3, 4 × 4 maps, K = 2 (q-map, q-copies)', () => {
  const s: ConvShape = { C: 3, H: 4, W: 4, K: 2, M: 1, N: 1 };
  it('W_out = 3 and B[7, 5] copies flat element 27 = X[n, 1, 2, 3]', () => {
    expect(outDims(s).Wout).toBe(3);
    const b = bToX(s, 7, 5);
    expect([b.c, b.p, b.q, b.oh, b.ow, b.h, b.w, b.flat]).toEqual([1, 1, 1, 1, 2, 2, 3, 27]);
  });
  it('X[n, 1, 1, 2] is copied 4 times, from patch corners (0, 1), (0, 2), (1, 1), (1, 2)', () => {
    const e = xToB(s, 1, 1, 2);
    expect(e.length).toBe(4);
    expect(e.map((x) => [Math.floor(x.v / 3), x.v % 3]).sort()).toEqual([[0, 1], [0, 2], [1, 1], [1, 2]]);
  });
  it('the loop nest and the GEMM agree on random data', () => {
    const t: ConvShape = { C: 3, H: 6, W: 5, K: 3, M: 4, N: 2 };
    const X = makeData(t.N * t.C * t.H * t.W, 7, -2, 3);
    const F = makeData(t.M * ckk(t), 11, -2, 2);
    const ref = convDirect(t, X, F);
    const P = pixels(t);
    for (let n = 0; n < t.N; n++) expect(convAsGemm(t, X, F, n)).toEqual(ref.slice(n * t.M * P, (n + 1) * t.M * P));
  });
});

describe('the cost of unfolding (Eq. 19.4)', () => {
  it('LeNet C1 19.14, C3 12.76, C5 1.00 (ch19-facts E1)', () => {
    expect(r2(expansion(preset('c1').shape))).toBe(19.14);
    expect(r2(expansion(preset('c3').shape))).toBe(12.76);
    expect(expansion(preset('c5').shape)).toBe(1);
    expect(UNFOLD_SCENE.c1Ratio).toBeCloseTo(19.140625, 9);
  });
  it('C3 holds 15000 B entries for 1176 inputs', () => {
    const z = sizes(preset('c3').shape);
    expect([z.bSample, z.xSample]).toEqual([15000, 1176]);
  });
  it('tends to K²: K = 3 gives 7.91, 8.84, 8.96 and K = 5 19.14, 24.12, 24.81 at 32, 224, 1024', () => {
    expect([32, 224, 1024].map((n) => r2(ratioAt(3, n)))).toEqual([7.91, 8.84, 8.96]);
    expect([32, 224, 1024].map((n) => r2(ratioAt(5, n)))).toEqual([19.14, 24.12, 24.81]);
  });
  it('does not depend on C or M', () => {
    const a = expansion({ C: 1, H: 20, W: 20, K: 3, M: 1, N: 1 });
    expect(expansion({ C: 300, H: 20, W: 20, K: 3, M: 77, N: 5 })).toBe(a);
  });
  it('K = 1 gives 1: B is X viewed as C × H·W', () => {
    const s: ConvShape = { C: 4, H: 5, W: 6, K: 1, M: 2, N: 1 };
    const X = makeData(120, 3);
    expect(expansion(s)).toBe(1);
    expect(unfold(s, X)).toEqual(X);
  });
  it('a stride equal to K (a patch embedding) gives 1: 3 × 224 × 224 with K = 16 or 14', () => {
    expect(expansion({ C: 3, H: 224, W: 224, K: 16, M: 1, N: 1, stride: 16 })).toBe(1);
    expect(expansion({ C: 3, H: 224, W: 224, K: 14, M: 1, N: 1, stride: 14 })).toBe(1);
    const s: ConvShape = { C: 2, H: 4, W: 4, K: 2, M: 1, N: 1, stride: 2 };
    expect(copiesPerPixel(s).every((c) => c === 1)).toBe(true);
    expect(unfold(s, makeData(32, 5)).slice().sort()).toEqual(makeData(32, 5).slice().sort());
  });
  it("the lab's shapes: 8.84, 8.37, 6.61, 1, and LeNet C3 12.76", () => {
    expect((['early', 'middle', 'late', 'pointwise', 'c3'] as const).map((id) => r2(expansion(preset(id).shape)))).toEqual([8.84, 8.37, 6.61, 1, 12.76]);
  });
  it('stride with a patch: Eq. 19.5 scales the corner by the stride', () => {
    const s: ConvShape = { C: 1, H: 6, W: 6, K: 2, M: 1, N: 1, stride: 2 };
    expect(bToX(s, 3, 4)).toMatchObject({ oh: 1, ow: 1, h0: 2, w0: 2, h: 3, w: 3 });
    const X = makeData(36, 9);
    const F = makeData(4, 2);
    expect(convAsGemm(s, X, F)).toEqual(convDirect(s, X, F));
  });
});

describe('sizes against the limits (ch19-facts E2, A1, C2)', () => {
  it('64 channels at 224 × 224, K = 3: 108.3 MiB per sample, 866.3 MiB for 8', () => {
    const z = sizes({ C: 64, H: 224, W: 224, K: 3, M: 64, N: 8 });
    expect(z.bSample).toBe(28_387_584);
    expect(z.bBatch).toBe(227_100_672);
    expect(r2(z.bBytesSample / MiB)).toBe(108.29);
    expect(Math.round((z.bBytesBatch / MiB) * 10) / 10).toBe(866.3);
    expect(z.flags).toEqual([{ id: 'metal', what: 'batch' }]);
  });
  it("E2's other rows: 26.6 MiB (C 3, K 7) and 820.1 MiB (C 256, 56 × 56, N 32)", () => {
    expect(Math.round((sizes({ C: 3, H: 224, W: 224, K: 7, M: 1, N: 1 }).bBytesSample / MiB) * 10) / 10).toBe(26.6);
    const z = sizes({ C: 256, H: 56, W: 56, K: 3, M: 1, N: 32 });
    expect(z.bBatch).toBe(214_990_848);
    expect(Math.round((z.bBytesBatch / MiB) * 10) / 10).toBe(820.1);
  });
  it('a 32-bit index overflows at a 1933 × 1933 input for C = 64, K = 3, one sample', () => {
    expect(ckk({ C: 64, H: 1932, W: 1932, K: 3, M: 1, N: 1 }) * 1930 * 1930).toBeLessThanOrEqual(INT32_MAX);
    expect(sizes({ C: 64, H: 1933, W: 1933, K: 3, M: 1, N: 1 }).flags.map((f) => f.id)).toContain('int32');
    expect(sizes({ C: 64, H: 1932, W: 1932, K: 3, M: 1, N: 1 }).flags.map((f) => f.id)).not.toContain('int32');
  });
  it('the Metal flag sits at 256 MB of FP32, the cuDNN flag at 2·10^9 elements', () => {
    expect(limitFlags(METAL_MIN_MAX_BUFFER / 4)).toEqual([]);
    expect(limitFlags(METAL_MIN_MAX_BUFFER / 4 + 1)).toEqual(['metal']);
    expect(limitFlags(2e9)).toEqual(['cudnn', 'metal']);
    expect(limitFlags(INT32_MAX + 1)).toEqual(['int32', 'cudnn', 'metal']);
  });
});

describe('GEMM shapes per layer (p. 472)', () => {
  it('LeNet: C1 F 6 × 25, B 25 × 784; C3 16 × 150, 150 × 100; C5 120 × 400, 400 × 1', () => {
    const g = (['c1', 'c3', 'c5'] as const).map((id) => gemmShape(preset(id).shape));
    expect(g.map((x) => [x.rows, x.inner, x.cols])).toEqual([[6, 25, 784], [16, 150, 100], [120, 400, 1]]);
  });
  it('C3 at T = 16: 10 phases; 300 requested loads per output direct, 20 implicit', () => {
    const g = gemmShape(preset('c3').shape);
    expect([g.phases, g.loadsDirect, g.loadsImplicit, g.tilesPerGemm]).toEqual([10, 300, 20, 7]);
    expect(r2(g.padding)).toBe(0.11);
  });
  it('C5 at batch 1 fills one column of each 16-wide tile: 15/16 of the slots are padding at best', () => {
    const g = gemmShape(preset('c5').shape);
    expect(g.tilesPerGemm).toBe(8);
    expect(g.padding).toBeCloseTo(1 - 120 / (128 * 16), 12);
    expect(g.padding).toBeGreaterThan(15 / 16);
  });
  it('whole batch in one GEMM: N·P columns, one GEMM, and a permute to reach NCHW', () => {
    const s = { ...preset('c5').shape, N: 64 };
    const per = gemmShape(s);
    const whole = gemmShape(s, { wholeBatch: true });
    expect([per.gemms, per.cols, per.tilesTotal, per.needsPermute]).toEqual([64, 1, 512, false]);
    expect([whole.gemms, whole.cols, whole.tilesTotal, whole.needsPermute]).toEqual([1, 64, 32, true]);
    expect(whole.padding).toBeCloseTo(1 - (120 * 64) / (128 * 64), 12);
  });
});

describe('mapping onto the project GEMM (q-roles, q-wholebatch)', () => {
  const s: ConvShape = { C: 3, H: 5, W: 4, K: 2, M: 3, N: 2 };
  const X = makeData(s.N * s.C * s.H * s.W, 21, -2, 3);
  const F = makeData(s.M * ckk(s), 22, -1, 2);
  const P = pixels(s);
  const R = ckk(s);
  const ref = convDirect(s, X, F);
  it('launchLinear(X = F, W = Bᵀ, batch = M, in = C·K·K, out = P) gives one sample\'s Y in NCHW', () => {
    for (let n = 0; n < s.N; n++) expect(linear(F, unfoldT(s, X, n), s.M, R, P)).toEqual(ref.slice(n * s.M * P, (n + 1) * s.M * P));
  });
  it('swapping the roles (X = Bᵀ, W = F) gives P × M: channels last', () => {
    const y = linear(unfoldT(s, X, 0), F, P, R, s.M);
    for (let v = 0; v < P; v++) for (let m = 0; m < s.M; m++) expect(y[v * s.M + m]).toBe(ref[m * P + v]);
  });
  it('one GEMM over the whole batch comes out M × N × P (CNHW), so NCHW needs a permute', () => {
    // B for all samples side by side: C·K·K × N·P.
    const Bs = Array.from({ length: s.N }, (_, n) => unfold(s, X, n));
    const Ball = Array.from({ length: R * s.N * P }, (_, i) => {
      const u = Math.floor(i / (s.N * P));
      const j = i % (s.N * P);
      return Bs[Math.floor(j / P)][u * P + (j % P)];
    });
    const Y = matmul(F, Ball, s.M, R, s.N * P);
    expect(Y).not.toEqual(ref);
    for (let m = 0; m < s.M; m++) for (let n = 0; n < s.N; n++) for (let v = 0; v < P; v++) expect(Y[(m * s.N + n) * P + v]).toBe(ref[(n * s.M + m) * P + v]);
  });
});

describe('the unfold scene', () => {
  it('fills four columns and draws three ratio curves that approach K²', () => {
    expect(UNFOLD_SCENE.copiesAfter.length).toBe(5);
    expect(UNFOLD_SCENE.copiesAfter.map((c) => c.reduce((a, b) => a + b, 0))).toEqual([0, 4, 8, 12, 16]);
    for (const c of UNFOLD_SCENE.curves) {
      const last = c.points[c.points.length - 1].ratio;
      expect(last).toBeLessThan(c.K * c.K);
      expect(last).toBeGreaterThan(c.K * c.K * 0.8);
    }
  });
});
