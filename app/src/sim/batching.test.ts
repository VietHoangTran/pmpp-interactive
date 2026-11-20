import { describe, expect, it } from 'vitest';
import {
  BOOK_DEVICE,
  M8_DEFAULT,
  PROJ_1024,
  PROJ_4096,
  bStar,
  bStarSearch,
  biasIntensity,
  bookThroughput,
  effectivePeak,
  intensity,
  intensityLimit,
  kneeOf,
  m8Shape,
  measuredDevices,
  paddedRows,
  precisionFor,
  projectionSeries,
  relativeStep,
  residualIntensity,
  stepAt,
  stepBStar,
  stepCurve,
  stepSeries,
  thresholdOf,
  weightBytes,
  type ReportsLike,
} from './batching';

const f32 = precisionFor(4);
const f16 = precisionFor(2);

describe('E1: the intensity of one decode-step GEMM', () => {
  it("reproduces NVIDIA's and chapter 20's linear layer (1024 → 4096, 16-bit): 1 FLOP/B at batch 1, 315 at batch 512", () => {
    expect(intensity(1024, 4096, 1, f16)).toBeCloseTo(0.9988, 4);
    expect(intensity(1024, 4096, 512, f16)).toBeCloseTo(315.08, 2);
  });
  it('the same layer in float32 has half the intensity: 0.4994 and 157.54', () => {
    expect(intensity(1024, 4096, 1, f32)).toBeCloseTo(0.4994, 4);
    expect(intensity(1024, 4096, 512, f32)).toBeCloseTo(157.54, 2);
  });
  it('is about B/2 in float32 and about B in 16-bit while B is much smaller than K and N', () => {
    for (const B of [1, 2, 4, 8]) {
      expect(intensity(4096, 4096, B, f32) / (B / 2)).toBeGreaterThan(0.99);
      expect(intensity(4096, 4096, B, f16) / B).toBeGreaterThan(0.99);
    }
  });
  it('levels off at 2KN / (a(K + N)) as B grows', () => {
    expect(intensityLimit(4096, 4096, f32)).toBe(1024);
    expect(intensity(4096, 4096, 1e9, f32)).toBeCloseTo(1024, 2);
  });
  it('B* = 40.8 for a 4096 → 4096 float32 projection at the book\'s threshold of 20, and the smallest whole batch is 41', () => {
    expect(bStar(4096, 4096, 20, f32)!).toBeCloseTo(40.797, 3);
    expect(bStarSearch(4096, 4096, 20, f32)).toBe(41);
    expect(intensity(4096, 4096, 40, f32)).toBeLessThan(20);
    expect(intensity(4096, 4096, 41, f32)).toBeGreaterThanOrEqual(20);
  });
  it('the closed form is E1\'s 2T / (1 − 2T(K + N)/(KN)) in float32', () => {
    const T = 7;
    const [K, N] = [1024, 4096];
    expect(bStar(K, N, T, f32)!).toBeCloseTo((2 * T) / (1 - (2 * T * (K + N)) / (K * N)), 9);
  });
  it('16-bit roughly halves B*: about T instead of about 2T', () => {
    const b32 = bStar(4096, 4096, 20, f32)!;
    const b16 = bStar(4096, 4096, 20, f16)!;
    expect(b16).toBeCloseTo(20.197, 3);
    expect(b16 / b32).toBeGreaterThan(0.45);
    expect(b16 / b32).toBeLessThan(0.55);
    expect(bStarSearch(4096, 4096, 20, f16)).toBe(21);
  });
  it('B* is null when the GEMM can never reach the threshold', () => {
    expect(bStar(64, 64, 20, f32)).toBeNull(); // limit 2·64·64 / (4·128) = 16 < 20
    expect(bStarSearch(64, 64, 20, f32)).toBeNull();
  });
});

describe('E2: the per-row pieces do not get more intense with B', () => {
  it('bias add: 0.083 FLOP/B at batch 1, approaching 0.125', () => {
    expect(biasIntensity(1, 4096, f32)).toBeCloseTo(1 / 12, 9);
    expect(biasIntensity(1e6, 4096, f32)).toBeCloseTo(0.125, 5);
    expect(biasIntensity(1e6, 4096, f32)).toBeLessThan(0.125);
  });
  it('residual add: 1/12 FLOP/B at every batch', () => {
    expect(residualIntensity(f32)).toBeCloseTo(1 / 12, 12);
  });
  it('the per-row pieces stay memory-bound, grow in proportion to B from the start, and stay a few per cent of the step', () => {
    const r1 = stepAt(M8_DEFAULT, 1, f32, 1, BOOK_DEVICE);
    const r512 = stepAt(M8_DEFAULT, 512, f32, 1, BOOK_DEVICE);
    for (const p of r512.pieces.filter((x) => x.perRow)) expect(p.bound).toBe('memory');
    // Doubling B nearly doubles the per-row time (not quite: the bias vectors are read once whatever B is).
    const r2 = stepAt(M8_DEFAULT, 2, f32, 1, BOOK_DEVICE);
    expect(r2.rowSeconds / r1.rowSeconds).toBeGreaterThan(1.75);
    expect(r512.rowSeconds / r1.rowSeconds).toBeGreaterThan(400);
    expect(r512.rowSeconds / r512.seconds).toBeCloseTo(0.034, 3);
  });
});

describe("the book's example device", () => {
  it('uses the threshold the book prints, 20.0 FLOP/B, for 66.9 TFLOPS and 3.35 TB/s', () => {
    expect(BOOK_DEVICE.peak / BOOK_DEVICE.bw).toBeCloseTo(19.97, 2);
    expect(thresholdOf(BOOK_DEVICE)).toBe(20);
    expect(effectivePeak(BOOK_DEVICE)).toBe(67000);
  });
  it('a measured device uses its own peak / bandwidth', () => {
    expect(thresholdOf({ peak: 1881.4, bw: 271.1 })).toBeCloseTo(6.94, 2);
  });
});

describe('the decode step on the roofline', () => {
  it('a lone projection: time per step is the GEMM at its bound, flat until B* and then in proportion to B', () => {
    const c = stepCurve(PROJ_4096, f32, 1, BOOK_DEVICE, [1, 8, 40, 82, 164]);
    expect(c[1].seconds / c[0].seconds).toBeLessThan(1.02);
    expect(c[2].seconds / c[0].seconds).toBeLessThan(1.03);
    expect(c[4].seconds / c[3].seconds).toBeCloseTo(2, 9); // compute-bound: twice the rows, twice the time
    expect(c[0].pieces.map((p) => p.key)).toEqual(['proj']);
  });
  it('tokens per second rise about in proportion to B below the knee and flatten above it', () => {
    const c = stepCurve(M8_DEFAULT, f32, 1, BOOK_DEVICE, [1, 16, 256, 1024]);
    expect(c[1].tokensPerSecond / c[0].tokensPerSecond).toBeGreaterThan(14.5); // 16 times the tokens, a step about 7% longer
    expect(c[3].tokensPerSecond / c[2].tokensPerSecond).toBeLessThan(1.01);
    expect(c[3].perToken / c[2].perToken).toBeGreaterThan(3.9);
  });
  it("the step's B* aggregates every GEMM; for lab 22.2's default shape it is 40.5 at the book's threshold", () => {
    expect(M8_DEFAULT).toEqual({ kind: 'step', d: 4096, layers: 2, vocab: 32000 });
    expect(stepBStar(M8_DEFAULT, f32, BOOK_DEVICE)!).toBeCloseTo(40.517, 3);
    expect(stepBStar(PROJ_4096, f32, BOOK_DEVICE)!).toBeCloseTo(40.797, 3);
  });
  it('the GEMMs turn compute-bound at the step B*', () => {
    const below = stepAt(M8_DEFAULT, 40, f32, 1, BOOK_DEVICE);
    const above = stepAt(M8_DEFAULT, 41, f32, 1, BOOK_DEVICE);
    expect(below.gemmCompute).toBeLessThan(below.gemmMemory);
    expect(above.gemmCompute).toBeGreaterThan(above.gemmMemory);
    expect(below.intensity).toBeLessThan(20);
    expect(above.intensity).toBeGreaterThan(20);
  });
  it('the step time is the sum of its pieces, and each piece at least its traffic and its arithmetic', () => {
    const r = stepAt(M8_DEFAULT, 100, f16, 16, BOOK_DEVICE);
    expect(r.seconds).toBeCloseTo(r.pieces.reduce((s, p) => s + p.seconds, 0), 15);
    for (const p of r.pieces) expect(p.seconds).toBeGreaterThanOrEqual(Math.max(p.memory, p.compute) - 1e-18);
    expect(r.tokensPerSecond).toBeCloseTo(100 / r.seconds, 6);
  });
  it('relative units: one pass over the weights at batch 1, and the knee where arithmetic overtakes the stream', () => {
    const r1 = relativeStep(M8_DEFAULT, 1, f32, 1, 20);
    expect(r1.memory).toBeGreaterThan(1);
    expect(r1.memory).toBeLessThan(1.002);
    expect(r1.compute).toBeCloseTo(2 / (20 * 4), 9); // 2B / (T·w)
    const r8 = relativeStep(M8_DEFAULT, 8, f32, 1, 20);
    expect(r8.compute).toBeCloseTo(0.2, 9);
    expect(r8.step).toBeLessThan(1.05);
    expect(relativeStep(M8_DEFAULT, 64, f32, 1, 20).compute).toBeGreaterThan(relativeStep(M8_DEFAULT, 64, f32, 1, 20).memory);
    expect(weightBytes(M8_DEFAULT, f32)).toBe(4 * (2 * 4096 * 4096 + 4096 * 32000));
  });
});

describe('E3: tile padding', () => {
  it('rows computed: ceil(B / P) · P', () => {
    expect([1, 16, 17, 128, 129].map((B) => paddedRows(B, 16))).toEqual([16, 16, 32, 128, 144]);
    expect([1, 128, 129].map((B) => paddedRows(B, 128))).toEqual([128, 128, 256]);
    expect(paddedRows(37, 1)).toBe(37);
  });
  it("padding to 128 makes the step time constant for B ≤ 128 (a lone projection, the book's device)", () => {
    const c = stepCurve(PROJ_4096, f32, 128, BOOK_DEVICE, [1, 2, 4, 8, 16, 32, 64, 128]);
    for (const r of c) expect(r.seconds).toBe(c[0].seconds);
    expect(stepAt(PROJ_4096, 129, f32, 128, BOOK_DEVICE).seconds).toBeGreaterThan(1.9 * c[0].seconds);
    // And it is flat because of arithmetic on padding rows, not bandwidth: at B = 1 the GEMM is compute-bound.
    expect(c[0].pieces[0].bound).toBe('compute');
  });
  it('with M3 padding the GEMMs of the whole step are constant to 128; only the per-row pieces move', () => {
    const c = stepCurve(M8_DEFAULT, f32, 128, BOOK_DEVICE, [1, 32, 128]);
    expect(c[1].gemmSeconds).toBe(c[0].gemmSeconds);
    expect(c[2].gemmSeconds).toBe(c[0].gemmSeconds);
    expect(c[2].seconds).toBeGreaterThan(c[0].seconds);
  });
  it('M2 padding (16) leaves a memory-bound batch 1 unchanged and makes steps of 16 above the knee', () => {
    expect(stepAt(PROJ_4096, 1, f32, 16, BOOK_DEVICE).seconds).toBe(stepAt(PROJ_4096, 1, f32, 1, BOOK_DEVICE).seconds);
    const a = stepAt(PROJ_4096, 65, f32, 16, BOOK_DEVICE).seconds;
    const b = stepAt(PROJ_4096, 80, f32, 16, BOOK_DEVICE).seconds;
    expect(a).toBe(b);
  });
});

describe("E4: the book's two throughput formulas (p. 539)", () => {
  it('a batch of 8 whose GEMM takes 3 times one GEMV: throughput × 8/3, each query\'s projection × 3', () => {
    const t = bookThroughput(8, 3, 1);
    expect(t.gain).toBeCloseTo(8 / 3, 12);
    expect(t.gain).toBeCloseTo(2.67, 2);
    expect(t.latencyFactor).toBe(3);
  });
  it('batching wins whenever the GEMM takes less than the sum of the GEMVs', () => {
    expect(bookThroughput(8, 7.9, 1).gain).toBeGreaterThan(1);
    expect(bookThroughput(8, 8, 1).gain).toBeCloseTo(1, 12);
  });
});

describe('kneeOf: where measured time per step has grown by half', () => {
  it('returns the first batch at 1.5 times the smallest batch\'s time', () => {
    const rows = [1, 2, 4, 8, 16, 32, 64, 128].map((batch) => ({ batch, seconds: Math.max(1, batch / 40) }));
    expect(kneeOf(rows)).toBe(64);
    expect(kneeOf(rows, 3)).toBe(128);
  });
  it('null when nothing grows that far, or with fewer than two points', () => {
    expect(kneeOf([1, 2, 4].map((batch) => ({ batch, seconds: 1 })))).toBeNull();
    expect(kneeOf([{ batch: 1, seconds: 1 }])).toBeNull();
    expect(kneeOf([])).toBeNull();
  });
  it('sorts by batch first and ignores unusable times', () => {
    expect(kneeOf([{ batch: 64, seconds: 2 }, { batch: 1, seconds: 1 }, { batch: 8, seconds: NaN }])).toBe(64);
  });
  it('an M3-like line, flat to 128 then doubling, has its knee at 256', () => {
    const rows = [1, 16, 128, 256, 512].map((batch) => ({ batch, seconds: Math.ceil(batch / 128) }));
    expect(kneeOf(rows)).toBe(256);
  });
  it("on the model's own curve the 1.5x knee comes later than B*", () => {
    const c = stepCurve(M8_DEFAULT, f32, 1, BOOK_DEVICE);
    const k = kneeOf(c.map((r) => ({ batch: r.batch, seconds: r.seconds })))!;
    expect(k).toBeGreaterThan(stepBStar(M8_DEFAULT, f32, BOOK_DEVICE)!);
    expect(k).toBe(59);
  });
});

describe("reading the reader's reports", () => {
  const reports: ReportsLike = {
    emu: { results: [{ lab: 'ch05/lab1_roofline', variant: 'solution', bench: [{ name: 'roofline', compute_gflops: 1, bandwidth_gb_per_s: 1 }] }] },
    gpu: {
      generated: '2026-11-20T10:00:00',
      device: { name: 'Test GPU' },
      results: [
        { lab: 'ch05/lab1_roofline', variant: 'solution', bench: [{ name: 'roofline', compute_gflops: 2000, bandwidth_gb_per_s: 250 }] },
        { lab: 'ch05/lab1_roofline', variant: 'work', bench: [{ name: 'roofline', compute_gflops: 9, bandwidth_gb_per_s: 9 }] },
        {
          lab: 'ch22/lab2_batching',
          variant: 'solution',
          bench: [
            { name: "step", gemm: "m2", batch: 1, seconds: 0.001, d: 512, layers: 2, vocab: 8192 },
            { name: 'step', gemm: 'm2', batch: 64, seconds: 0.0016 },
            { name: 'step', gemm: 'm3', batch: 1, ms: 2 },
            { name: 'step', gemm: 'm3', batch: 256, ms: 4 },
            { name: 'other', gemm: 'm3', batch: 2, seconds: 9 },
          ],
        },
        {
          lab: 'ch05/lab3_gemm',
          variant: 'solution',
          bench: [
            { name: 'linear_layer', batch: 1, k: 1024, l: 4096, seconds: 0.0005 },
            { name: 'linear_layer', batch: 4, k: 1024, l: 4096, seconds: 0.0005 },
          ],
        },
        {
          lab: 'ch15/lab6_linear',
          variant: 'solution',
          bench: [
            { name: 'm3', kernel: 'm3', batch: 1, in: 1024, out: 4096, seconds: 0.0002 },
            { name: 'vendor', kernel: 'vendor', batch: 1, in: 4096, out: 4096, seconds: 0.0003 },
          ],
        },
      ],
    },
  };
  it("lab 5.1's roofline row gives each GPU backend's device; the emulators are not devices", () => {
    expect(measuredDevices(reports)).toEqual([{ backend: 'gpu', name: 'Test GPU', date: '2026-11-20', peak: 2000, bw: 250 }]);
  });
  it("lab 22.2's step rows: one series per GEMM, seconds or ms, other rows ignored", () => {
    const s = stepSeries(reports, 'gpu');
    expect(s.map((x) => x.label)).toEqual(["lab 22.2, M2 (lab 6.4's tiled kernel)", "lab 22.2, M3 (lab 15.6's register-tiled kernel)"]);
    expect(s[0].points).toEqual([{ batch: 1, seconds: 0.001 }, { batch: 64, seconds: 0.0016 }]);
    expect(s[0].knee).toBe(64);
    expect(s[1].points[1].seconds).toBeCloseTo(0.004, 12);
    expect(s[1].knee).toBe(256);
    expect(stepSeries(reports, 'nothing')).toEqual([]);
  });
  it("the step's shape comes from the rows when they carry it, else M8's default", () => {
    expect(m8Shape(reports)).toEqual({ kind: "step", d: 512, layers: 2, vocab: 8192 });
    expect(m8Shape({})).toEqual(M8_DEFAULT);
  });
  it("labs 5.3 and 15.6 overlay a projection of their own shape only", () => {
    const s = projectionSeries(reports, 'gpu', PROJ_1024.k, PROJ_1024.n);
    expect(s.map((x) => x.label)).toEqual(['lab 5.3, tiled (M2)', 'lab 15.6, M3']);
    expect(s[0].knee).toBeNull();
    expect(projectionSeries(reports, 'gpu', 4096, 4096).map((x) => x.label)).toEqual(['lab 15.6, vendor library']);
  });
});
