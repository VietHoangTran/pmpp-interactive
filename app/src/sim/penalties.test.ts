import { describe, expect, it } from 'vitest';
import { SEGMENT_MODELS } from './coalesce';
import {
  PENALTIES_OFF,
  PEN_CONTEXT,
  PEN_LOGITS,
  QWEN3_VOCAB,
  appendToken,
  applyPenalties,
  argmaxLowest,
  bytesPerLogit,
  countTokens,
  f32,
  frequencyPresence,
  mappingAccess,
  penalise,
  repetition,
  segmentFor,
  softmaxDouble,
  warpLoad,
  type PenaltySettings,
} from './penalties';

const S = (o: Partial<PenaltySettings>): PenaltySettings => ({ ...PENALTIES_OFF, ...o });

describe('the answers the lesson states', () => {
  it('x1/X1.3/p-rep-neg: -2.0 at 1.25 becomes -2.5 under llama.cpp, -1.6 under CTRL', () => {
    expect(repetition(-2.0, 1, 1.25, 'llama')).toBe(-2.5);
    expect(repetition(-2.0, 1, 1.25, 'hf')).toBe(-2.5);
    expect(repetition(-2.0, 1, 1.25, 'ctrl')).toBeCloseTo(-1.6, 15);
    expect(repetition(-2.0, 1, 1.25, 'llama', true)).toBe(-2.5);
  });
  it('x1/X1.3/q-zero: a logit of exactly 0 stays 0 under both sign rules', () => {
    expect(repetition(0, 1, 1.25, 'llama')).toBe(0);
    expect(repetition(0, 1, 1.25, 'hf')).toBe(0);
  });
  it('x1/X1.3/q-freq: 1.2 with count 3, frequency 0.1, presence 0.5 gives 0.4', () => {
    expect(frequencyPresence(1.2, 3, 0.1, 0.5)).toBeCloseTo(0.4, 12);
    expect(frequencyPresence(1.2, 3, 0.1, 0.5, true)).toBeCloseTo(0.4, 6);
    expect(penalise(1.2, 3, S({ freq: 0.1, presence: 0.5 }))).toBeCloseTo(0.4, 12);
  });
  it('x1/X1.3/q-order: 2.0, count 2, repetition 1.25, frequency 0.5: 0.6 in llama.cpp order, 0.8 reversed', () => {
    expect(penalise(2.0, 2, S({ repeat: 1.25, freq: 0.5 }))).toBeCloseTo(0.6, 12);
    expect(penalise(2.0, 2, S({ repeat: 1.25, freq: 0.5, order: 'subtract-first' }))).toBeCloseTo(0.8, 12);
  });
  it('x1/X1.3/q-bytes: two passes move 24 bytes per logit, one fused pass 12', () => {
    expect(bytesPerLogit(2)).toBe(24);
    expect(bytesPerLogit(1)).toBe(12);
  });
});

describe('penalties off leave every logit bit-identical', () => {
  it('double and float32, every rule and order, on the explorer row and on random rows', () => {
    const counts = countTokens(PEN_CONTEXT, PEN_LOGITS.length);
    let seed = 1;
    const rnd = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32) * 20 - 10;
    const rows = [PEN_LOGITS, Array.from({ length: 200 }, rnd), [0, -0, 1e-30, -1e-30, 3.4e38, -3.4e38]];
    for (const row of rows)
      for (const rule of ['llama', 'hf', 'ctrl'] as const)
        for (const order of ['repeat-first', 'subtract-first'] as const)
          for (const float32 of [false, true]) {
            const src = float32 ? row.map(f32) : row;
            const c = row === PEN_LOGITS ? counts : row.map((_, i) => i % 3);
            const out = applyPenalties(src, c, { ...PENALTIES_OFF, rule, order, float32 });
            out.forEach((v, i) => expect(Object.is(v, src[i]) || (v === 0 && src[i] === 0)).toBe(true));
          }
  });
});

describe('what the penalties explorer shows', () => {
  const counts = countTokens(PEN_CONTEXT, PEN_LOGITS.length);
  it('the context counts', () => {
    expect(PEN_CONTEXT.length).toBe(20);
    expect(counts).toEqual([1, 4, 1, 5, 1, 1, 1, 3, 1, 0, 2, 0]);
    expect(PEN_LOGITS[1]).toBe(-2.0);
    expect(PEN_LOGITS[8]).toBe(0);
  });
  it("under CTRL's rule a repeated token with a negative logit becomes more likely", () => {
    const base = softmaxDouble(PEN_LOGITS);
    const ctrl = softmaxDouble(applyPenalties(PEN_LOGITS, counts, S({ repeat: 1.5, rule: 'ctrl' })));
    const llama = softmaxDouble(applyPenalties(PEN_LOGITS, counts, S({ repeat: 1.5, rule: 'llama' })));
    expect(ctrl[1]).toBeGreaterThan(base[1]);
    expect(llama[1]).toBeLessThan(base[1]);
  });
  it('presence acts once per token, frequency grows with each repeat', () => {
    const p = [1, 2, 5].map((c) => 1 - frequencyPresence(1, c, 0, 0.5));
    p.forEach((d) => expect(d).toBeCloseTo(0.5, 12));
    const f = [1, 2, 5].map((c) => 1 - frequencyPresence(1, c, 0.3, 0));
    expect(f[0]).toBeCloseTo(0.3, 12);
    expect(f[1]).toBeCloseTo(0.6, 12);
    expect(f[2]).toBeCloseTo(1.5, 12);
    expect(frequencyPresence(1, 0, 0.3, 0.5)).toBe(1);
  });
  it('swapping the order changes a penalised positive logit', () => {
    const s = S({ repeat: 1.5, freq: 0.2, presence: 0.4 });
    const a = applyPenalties(PEN_LOGITS, counts, s);
    const b = applyPenalties(PEN_LOGITS, counts, { ...s, order: 'subtract-first' });
    expect(a[3]).not.toBeCloseTo(b[3], 6);
    expect(a[9]).toBe(b[9]); // count 0: untouched either way
  });
  it('counts: a window of the last N tokens, and one decode step', () => {
    expect(countTokens([1, 2, 1, 3], 4, 2)).toEqual([0, 1, 0, 1]);
    expect(countTokens([1, 2, 1, 3], 4)).toEqual([0, 2, 1, 1]);
    expect(appendToken([0, 2, 1, 1], 0)).toEqual([1, 2, 1, 1]);
    expect(() => countTokens([4], 4)).toThrow();
  });
  it('"append the most likely token" appends the argmax of the penalised row, lowest id on a tie', () => {
    expect(argmaxLowest([1, 3, 3, 0])).toBe(1);
    expect(argmaxLowest(PEN_LOGITS)).toBe(3);
  });
});

describe('who reads which logit: one warp, one load', () => {
  const P = { batch: 64, vocab: QWEN3_VOCAB, blockDim: 256 };
  it('one thread per element and one block per row are fully coalesced at widths 32 and 64', () => {
    for (const W of [32, 64])
      for (const m of ['element', 'block'] as const) {
        const a = mappingAccess(m, { ...P, warpWidth: W }, 32);
        expect(a.lanes.length).toBe(W);
        expect(a.transactions).toBe((W * 4) / 32);
        expect(a.efficiency).toBe(1);
        expect(mappingAccess(m, { ...P, warpWidth: W }, 128).transactions).toBe(W / 32);
      }
  });
  it('one thread per row puts the lanes a whole row apart: one segment per lane', () => {
    for (const W of [32, 64]) {
      const a = mappingAccess('row', { ...P, warpWidth: W }, 32);
      expect(a.transactions).toBe(W);
      expect(a.efficiency).toBe(4 / 32);
      expect(mappingAccess('row', { ...P, warpWidth: W }, 128).transactions).toBe(W);
    }
  });
  it('at batch 1 one thread per row has one working thread and one active lane', () => {
    const a = mappingAccess('row', { ...P, batch: 1, warpWidth: 32 }, 32);
    expect(a.lanes.length).toBe(1);
    expect(a.threadsWithWork).toBe(1);
    expect(a.loadsPerThread).toBe(QWEN3_VOCAB);
    expect(warpLoad('element', { ...P, batch: 1, warpWidth: 32 }).threadsWithWork).toBe(QWEN3_VOCAB);
    expect(warpLoad('block', { ...P, batch: 1, warpWidth: 32 }).threadsWithWork).toBe(256);
    expect(warpLoad('block', { ...P, batch: 1, warpWidth: 32 }).loadsPerThread).toBe(Math.ceil(QWEN3_VOCAB / 256));
  });
  it('the segment sizes come from coalesce.ts; Metal takes the reader\'s choice', () => {
    expect(segmentFor('nvidia', 64)).toBe(SEGMENT_MODELS.nvidia.segmentBytes);
    expect(segmentFor('amd128', 32)).toBe(128);
    expect(segmentFor('metal', 64)).toBe(64);
  });
});
