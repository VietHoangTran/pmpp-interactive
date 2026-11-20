import { describe, expect, it } from 'vitest';
import { BOOK_DEVICE, M8_DEFAULT } from './batching';
import {
  F32_TOY,
  bandwidthUtilisation,
  greedyRun,
  matrixWeightBytes,
  measuredSpeedup,
  measuredSteps,
  predictedSpeedup,
  quantiseModel,
  stepBytes,
  toyLogits,
  toyModel,
  toyWeightBytes,
} from './qstep';

const F = { layers: 'f32', embed: 'f32' } as const;

describe("the step's weight bytes at lab 22.2's default shape (q-step-ratio)", () => {
  it('658.5 MB in float32; the LM head is 131 million of 165 million weights', () => {
    expect(matrixWeightBytes(M8_DEFAULT, F)).toBe(658505728);
    expect(M8_DEFAULT.vocab * M8_DEFAULT.d).toBe(131072000);
    expect(M8_DEFAULT.layers * M8_DEFAULT.d ** 2 + M8_DEFAULT.vocab * M8_DEFAULT.d).toBe(164626432);
  });
  it('only the layers in Q4_0: 543.2 MB, 1.21 times', () => {
    const q = matrixWeightBytes(M8_DEFAULT, { layers: 'q4_0', embed: 'f32' });
    expect(q).toBe(543162368);
    expect((658505728 / q).toFixed(2)).toBe('1.21');
  });
  it('all of it in Q4_0: 7.11 times; in Q8_0: 3.76', () => {
    expect((658505728 / matrixWeightBytes(M8_DEFAULT, { layers: 'q4_0', embed: 'q4_0' })).toFixed(2)).toBe('7.11');
    expect((658505728 / matrixWeightBytes(M8_DEFAULT, { layers: 'q8_0', embed: 'q8_0' })).toFixed(2)).toBe('3.76');
  });
});

describe('stepBytes: every piece of the step', () => {
  it('the weight bytes of the pieces are the matrices plus the gather', () => {
    const s = stepBytes(M8_DEFAULT, { layers: 'q4_0', embed: 'q4_0' }, 1);
    // one row of E in Q4_0: 4096 / 32 blocks of 18 bytes
    expect(s.pieces.find((p) => p.key === 'embed')!.weightBytes).toBe(2304);
    expect(s.weightBytes).toBe(matrixWeightBytes(M8_DEFAULT, { layers: 'q4_0', embed: 'q4_0' }) + 2304);
  });
  it('the non-weight bytes do not shrink with the weights, so the step speed-up is below the byte ratio', () => {
    const f = stepBytes(M8_DEFAULT, F, 1);
    const q = stepBytes(M8_DEFAULT, { layers: 'q4_0', embed: 'q4_0' }, 1);
    expect(f.bytes - f.weightBytes).toBe(q.bytes - q.weightBytes);
    const s = predictedSpeedup(M8_DEFAULT, { layers: 'q4_0', embed: 'q4_0' }, 1);
    expect(s).toBeLessThan(7.11);
    expect(s).toBeGreaterThan(7.0);
  });
  it('quantising only the layers buys about 1.2 times at batch 1', () => {
    expect(predictedSpeedup(M8_DEFAULT, { layers: 'q4_0', embed: 'f32' }, 1).toFixed(2)).toBe('1.21');
  });
  it("Q4_0's advantage shrinks with batch; on the book's example the head turns compute-bound at batch 8", () => {
    const fm = { layers: 'q4_0', embed: 'q4_0' } as const;
    const s = [1, 2, 4, 8].map((B) => predictedSpeedup(M8_DEFAULT, fm, B, BOOK_DEVICE));
    for (let i = 1; i < 4; i++) expect(s[i]).toBeLessThan(s[i - 1]);
    const head = stepBytes(M8_DEFAULT, fm, 8, BOOK_DEVICE).pieces.find((p) => p.key === 'head')!;
    expect(head.flops / 66900e9).toBeGreaterThan(head.bytes / 3350e9);
    // intensity of a Q4_0 GEMV at batch 8, weights only: 2·8 / 0.5625
    expect(16 / 0.5625).toBeGreaterThan(20);
  });
  it('on a device with a high threshold the advantage only shrinks slightly with batch', () => {
    const dev = { peak: 1e6, bw: 100 };
    const fm = { layers: 'q4_0', embed: 'q4_0' } as const;
    const a = predictedSpeedup(M8_DEFAULT, fm, 1, dev);
    const b = predictedSpeedup(M8_DEFAULT, fm, 8, dev);
    expect(b).toBeLessThan(a);
    expect(b).toBeGreaterThan(6.5);
  });
  it('bandwidth utilisation is weight bytes per second over peak', () => {
    expect(bandwidthUtilisation(1e9, 1, 2)).toBe(0.5);
  });
});

describe("lab X2.4's rows (labs/x2/lab4_m9/main.cpp emits name 'qstep', format_layers, format_head, batch, seconds)", () => {
  const reports = {
    metal: {
      results: [
        {
          lab: 'x2/lab4_m9', variant: 'solution',
          bench: [
            { name: 'qstep', format_layers: 'f32', format_head: 'f32', batch: 1, seconds: 0.01 },
            { name: 'qstep', format_layers: 'q4_0', format_head: 'q4_0', batch: 1, seconds: 0.0025 },
            { name: 'qstep', format_layers: 'q4_0', format_head: 'f32', batch: 1, seconds: 0.008 },
            { name: 'other', format_layers: 'q4_0', format_head: 'q4_0', batch: 1, seconds: 1 },
          ],
        },
      ],
    },
  };
  it('reads qstep rows and gives the speed-up over f32/f32', () => {
    const rows = measuredSteps(reports, 'metal');
    expect(rows).toHaveLength(3);
    expect(measuredSpeedup(rows, 'q4_0', 1)).toBe(4);
    expect(measuredSpeedup(rows, 'q4_0/f32', 1)).toBe(1.25);
    expect(measuredSpeedup(rows, 'q8_0', 1)).toBeNull();
    expect(measuredSteps(reports, 'cuda')).toEqual([]);
  });
});

describe('the toy decoder', () => {
  const m = toyModel();
  it('is deterministic by seed', () => {
    expect(toyModel().E).toEqual(m.E);
    expect(toyModel(31).E).not.toEqual(m.E);
    expect(greedyRun(m, { W: ['q4_0', 'q4_0'], E: 'q4_0' })).toEqual(greedyRun(toyModel(), { W: ['q4_0', 'q4_0'], E: 'q4_0' }));
  });
  it('the F32 run is the unquantised model: no error, no flip, no divergence', () => {
    const r = greedyRun(m, F32_TOY(m));
    expect(quantiseModel(m, F32_TOY(m)).E).toEqual(m.E);
    expect(r.steps.every((s) => s.rel === 0 && !s.flipped)).toBe(true);
    expect(r.firstDivergence).toBeNull();
    expect(r.qTokens).toEqual(r.f32Tokens);
    expect(toyLogits(m, 1)).toEqual(toyLogits(quantiseModel(m, F32_TOY(m)), 1));
  });
  it('the float32 run visits at least 20 different tokens in 32 steps', () => {
    expect(new Set(greedyRun(m, F32_TOY(m)).f32Tokens).size).toBeGreaterThanOrEqual(20);
  });
  it('Q8_0 keeps every token; Q4_0 flips a few; affine 3-bit flips more', () => {
    const fl = (f: 'q8_0' | 'q4_0' | 'affine3') => greedyRun(m, { W: [f, f], E: f }).steps.filter((s) => s.flipped).length;
    expect(fl('q8_0')).toBe(0);
    expect(fl('q4_0')).toBeGreaterThan(0);
    expect(fl('affine3')).toBeGreaterThan(fl('q4_0'));
  });
  it('tokens flip where the margin is small: every flipped step has a margin below the largest unflipped one', () => {
    const r = greedyRun(m, { W: ['q4_0', 'q4_0'], E: 'q4_0' });
    const flipped = r.steps.filter((s) => s.flipped).map((s) => s.margin);
    const kept = r.steps.filter((s) => !s.flipped).map((s) => s.margin);
    expect(Math.max(...flipped)).toBeLessThan(Math.max(...kept));
    expect(r.steps.find((s) => s.margin < 1)!.flipped).toBe(true);
  });
  it('logits error is not the weights error: ~0.086 weights, larger in the logits', () => {
    const r = greedyRun(m, { W: ['q4_0', 'q4_0'], E: 'q4_0' });
    expect(r.weightRel.E).toBeGreaterThan(0.07);
    expect(r.weightRel.E).toBeLessThan(0.1);
    expect(r.meanRel).toBeGreaterThan(r.weightRel.E);
  });
  it('toy weight bytes: F32 is 4 bytes per weight; Q4_0 is 7.11 times smaller', () => {
    const n = 256 * 64 + 2 * 64 * 64;
    expect(toyWeightBytes(m, F32_TOY(m))).toBe(4 * n);
    expect(toyWeightBytes(m, { W: ['q4_0', 'q4_0'], E: 'q4_0' })).toBe((n / 32) * 18);
  });
});
