import { describe, expect, it } from 'vitest';
import { runOnline } from './onlinesoftmax';
import {
  KERNELS,
  N_READS,
  ROW16,
  maxAbsDiff,
  maxUlps,
  online,
  onlineBase2,
  onlineBaseE,
  predictionOk,
  predictionSteps,
  rowBytes,
  softmax64,
  twoPass,
  ulps,
} from './rowreduce';

const T2 = twoPass(ROW16);
const ON = online(ROW16);

describe('both paths compute softmax', () => {
  it('each equals a float64 softmax within float32 rounding', () => {
    const ref = softmax64(ROW16);
    expect(maxAbsDiff(T2.out, ref)).toBeLessThan(1e-6);
    expect(maxAbsDiff(ON.out, ref)).toBeLessThan(1e-6);
    expect(T2.out.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
  });
  it('the two agree within rounding but not bit for bit: the sums differ by 1 ulp, outputs by at most 2', () => {
    const d = maxAbsDiff(T2.out, ON.out);
    expect(d).toBeGreaterThan(0);
    expect(d).toBeLessThan(1e-7);
    expect(ulps(T2.sum, ON.sum)).toBe(1);
    expect(maxUlps(T2.out, ON.out)).toBe(2);
  });
  it('random rows of several lengths, including a ragged last chunk', () => {
    let seed = 7;
    const r = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648) * 12 - 6;
    for (const n of [1, 3, 5, 17, 64, 333]) {
      const row = Array.from({ length: n }, r);
      const ref = softmax64(row);
      expect(maxAbsDiff(twoPass(row).out, ref)).toBeLessThan(2e-6);
      expect(maxAbsDiff(online(row).out, ref)).toBeLessThan(2e-6);
    }
  });
});

describe('reads, writes and exponentials', () => {
  it('two-pass reads the row three times, online twice; both write once', () => {
    expect(T2.rowReads).toBe(3);
    expect(ON.rowReads).toBe(2);
    expect(T2.counters.writes).toBe(16);
    expect(ON.counters.writes).toBe(16);
  });
  it('online pays one extra exponential per chunk: 36 against 32', () => {
    expect(T2.counters.exps).toBe(32);
    expect(ON.counters.exps).toBe(32 + 16 / N_READS);
  });
  it('the q-bytes worked numbers: a 32000-logit row, 512,000 against 384,000 bytes', () => {
    expect(rowBytes(32000, 3, 1)).toBe(512000);
    expect(rowBytes(32000, 2, 1)).toBe(384000);
    expect(rowBytes(32000, 2, 1) / rowBytes(32000, 3, 1)).toBe(0.75);
  });
});

describe('the online rescale', () => {
  it('the row: max 4, rises in chunks 1 and 3, not in chunk 2', () => {
    expect(ON.max).toBe(4);
    const cm = ON.steps.filter((s) => s.op === 'chunk-max');
    expect(cm.map((s) => s.max)).toEqual([2, Math.fround(3.1), Math.fround(3.1), 4]);
    expect(ON.rises).toBe(2);
    const rs = ON.steps.filter((s) => s.op === 'rescale');
    expect(rs[2].factor).toBe(1);
  });
  it('the predicted normalisers after each rescale (the stepper checks against these)', () => {
    const at = predictionSteps(ON).map((i) => ON.steps[i].sum);
    expect(at).toHaveLength(3);
    expect(at[0]).toBeCloseTo(0.61, 2);
    expect(at[1]).toBeCloseTo(1.889, 3);
    expect(at[2]).toBeCloseTo(1.037, 3);
    expect(predictionOk(0.61, at[0])).toBe(true);
    expect(predictionOk(1.83, at[0])).toBe(false);
  });
  it("after every chunk the normaliser is chapter 20's D (sim/onlinesoftmax.ts, tiles of 4)", () => {
    const ch20 = runOnline({
      scores: ROW16,
      V: ROW16.map(() => [1]),
      bc: N_READS,
      row: ROW16.length - 1,
      order: 'ltr',
      init: 'neg-inf',
      form: 'deferred',
    });
    const ours = ON.steps.filter((s) => s.op === 'add').map((s) => s.sum);
    expect(ch20.steps.map((s) => s.mNew)).toEqual(ON.steps.filter((s) => s.op === 'add').map((s) => s.max));
    ch20.steps.forEach((s, k) => expect(ours[k]).toBeCloseTo(s.D, 5));
    ch20.steps.forEach((s, k) => {
      const r = ON.steps.filter((x) => x.op === 'rescale')[k];
      expect(r.factor!).toBeCloseTo(s.aFactor, 6);
    });
  });
  it('the final normaliser is the float64 sum of exp(x - max)', () => {
    const s = ROW16.reduce((a, v) => a + Math.exp(v - 4), 0);
    expect(ON.sum).toBeCloseTo(s, 5);
    expect(T2.sum).toBeCloseTo(s, 5);
  });
});

describe('base 2 with the scale folded into log2 e (x3/X3.5/q-exp2)', () => {
  const scores = [0.8, -0.4, 1.6, 0.2, 2.9, 0.5, -1.1, 1.7];
  const scale = 1 / Math.sqrt(64);
  it('gives the same probabilities as base e, within rounding', () => {
    const e = onlineBaseE(scores, scale);
    const two = onlineBase2(scores, scale);
    expect(maxAbsDiff(e.probs, two.probs)).toBeLessThan(1e-6);
  });
  it('the normaliser is the same number: it is not divided by log2 e', () => {
    const e = onlineBaseE(scores, scale);
    const two = onlineBase2(scores, scale);
    expect(two.D).toBeCloseTo(e.D, 5);
    expect(Math.abs(two.D - e.D / Math.LOG2E)).toBeGreaterThan(0.1);
  });
  it('the running max is in scaled base-2 units: log2 e times the base-e max', () => {
    const e = onlineBaseE(scores, scale);
    const two = onlineBase2(scores, scale);
    two.maxes.forEach((m, i) => expect(m).toBeCloseTo(e.maxes[i] * Math.LOG2E, 5));
  });
});

describe('the classification table', () => {
  it('seven kernels, as the section lists them', () => {
    expect(KERNELS).toHaveLength(7);
    expect(new Set(KERNELS.map((k) => k.id)).size).toBe(7);
  });
  it('MLX: max then sum on Metal with fast::exp, online on CUDA with __expf', () => {
    const m = KERNELS.find((k) => k.id === 'mlx-metal-softmax')!;
    const c = KERNELS.find((k) => k.id === 'mlx-cuda-softmax')!;
    expect([m.structure, m.exp]).toEqual(['max-then-sum', 'fast::exp']);
    expect([c.structure, c.exp]).toEqual(['online', '__expf']);
  });
  it('PyTorch MPS uses precise::exp; what the facts do not say stays not read', () => {
    expect(KERNELS.find((k) => k.id === 'torch-mps-softmax')!.exp).toBe('precise::exp');
    const t = KERNELS.find((k) => k.id === 'torch-cuda-softmax')!;
    expect([t.structure, t.exp]).toEqual(['not-read', 'not-read']);
  });
  it('RMSNorm is one sum and no exponential', () => {
    const r = KERNELS.find((k) => k.id === 'llama-cuda-rmsnorm')!;
    expect([r.structure, r.exp]).toEqual(['one-sum', 'none']);
  });
});
