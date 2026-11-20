import { describe, expect, it } from 'vitest';
import {
  CHAIN_ROWS,
  DEFAULT_SETTINGS,
  DRAW,
  GREEDY,
  KAT_U,
  KAT_WORD,
  M5_ORDER,
  OPENING_PAIR,
  ORDERS,
  ORDER_IDS,
  P_ORDER,
  SCENE_ROW,
  SCENE_SETTINGS,
  STAGES,
  TEMP_ROWS,
  VOCAB,
  argmax,
  chainRow,
  drawInverseCdf,
  entropyBits,
  f32,
  finalSet,
  minPKeep,
  modelledOf,
  msdSelectStep,
  optionsOf,
  pOrderCounts,
  rankDesc,
  reciprocalMismatch,
  rowBytes,
  runChain,
  sceneChain,
  softmax,
  temperature,
  topK,
  topPKeep,
  type ChainSettings,
  type StageId,
} from './chain';
import { minPThreshold } from './stablescan';

const ids = (alive: boolean[]) => alive.flatMap((a, i) => (a ? [i] : []));
const all = (n: number) => new Array<boolean>(n).fill(true);
const OFF: ChainSettings = { T: 1, k: 0, p: 1, minP: 0, repeat: 1, freq: 0, presence: 0, u: 0.5, float32: false };
/** Logits whose softmax is exactly the given probabilities (up to rounding). */
const fromProbs = (ps: number[]) => ps.map(Math.log);
const afterStage = (r: ReturnType<typeof runChain>, s: StageId) => r.snapshots.find((x) => x.stage === s)!;

describe('the stages and the order presets (x1-facts D1, D4, D8)', () => {
  it("STAGES is llama.cpp's default chain in its order", () => {
    expect(STAGES.map((s) => s.id)).toEqual(['penalties', 'dry', 'top-n-sigma', 'top-k', 'typical', 'top-p', 'min-p', 'xtc', 'temperature']);
  });
  it('DRY, top-n-sigma, typical and XTC are listed, not modelled, and have no M5 kernel', () => {
    const not = STAGES.filter((s) => !s.modelled);
    expect(not.map((s) => s.id)).toEqual(['dry', 'top-n-sigma', 'typical', 'xtc']);
    expect(not.every((s) => s.kernels.length === 0)).toBe(true);
  });
  it('every modelled stage names the labs that supply its kernels', () => {
    for (const s of STAGES.filter((x) => x.modelled)) {
      expect(s.kernels.length).toBeGreaterThan(0);
      for (const k of s.kernels) expect(k.lab.num).toMatch(/^(9\.3|10\.3|11\.6|12\.4|14\.3|X1\.[123])$/);
    }
    expect(STAGES.find((s) => s.id === 'top-k')!.kernels.map((k) => k.lab.num)).toEqual(['X1.1', '12.4', '14.3']);
    expect(STAGES.find((s) => s.id === 'top-p')!.kernels.map((k) => k.lab.num)).toEqual(['10.3', '11.6']);
    expect(DRAW.lab.num).toBe('X1.3');
    expect(GREEDY.lab.num).toBe('10.3');
  });
  it("M5's order is llama.cpp's restricted to the stages it builds", () => {
    expect(M5_ORDER).toEqual(['penalties', 'top-k', 'top-p', 'min-p', 'temperature']);
    expect(modelledOf(ORDERS.llama.order)).toEqual(M5_ORDER);
    expect(ORDERS.m5.order).toEqual(M5_ORDER);
  });
  it('transformers applies temperature before every truncation; llama.cpp and mlx-lm after', () => {
    expect(modelledOf(ORDERS.hf.order)).toEqual(['penalties', 'temperature', 'top-k', 'top-p', 'min-p']);
    expect(modelledOf(ORDERS.mlx.order)).toEqual(['penalties', 'top-p', 'min-p', 'top-k', 'temperature']);
    expect(ORDERS.hf.topKTies).toBe('keep');
    expect(ORDERS.hf.hasFreqPresence).toBe(false);
  });
});

describe('worked numbers the lessons state', () => {
  it('x1/start/q-exclusive: [0.5, 0.25, 0.125, 0.125] at p = 0.75 keeps 2', () => {
    const l = fromProbs([0.5, 0.25, 0.125, 0.125]);
    expect(ids(topPKeep(l, all(4), 0.75))).toEqual([0, 1]);
    expect(ids(topPKeep(l, all(4), 0.75, true))).toEqual([0, 1]);
  });
  it('x1/X1.1/q-renorm: top-k 3 then top-p 0.8 keeps 2; top-p over the full row keeps 3', () => {
    const row = { logits: fromProbs([0.5, 0.25, 0.125, 0.0625, 0.0625]), context: [] };
    const both = runChain(row, { ...OFF, k: 3, p: 0.8 }, ['top-k', 'top-p']);
    expect(finalSet(both)).toEqual([0, 1]);
    const renorm = afterStage(both, 'top-k').probs.slice(0, 3);
    expect(renorm[0]).toBeCloseTo(4 / 7, 12);
    expect(renorm[1]).toBeCloseTo(2 / 7, 12);
    expect(renorm[2]).toBeCloseTo(1 / 7, 12);
    expect(finalSet(runChain(row, { ...OFF, p: 0.8 }, ['top-p']))).toEqual([0, 1, 2]);
  });
  it("x1/X1.1/p-order: on the 'tail' row at k 8, p 0.9, T 2, llama.cpp's order keeps 4 and transformers' 7", () => {
    expect(P_ORDER.rowId).toBe('tail');
    expect(pOrderCounts()).toEqual({ llama: 4, hf: 7 });
    // llama.cpp decides the set at T = 1, so its count does not depend on T.
    for (const T of [0.5, 1, 2, 4]) {
      const s = { ...P_ORDER.settings, T };
      expect(finalSet(runChain(chainRow('tail'), s, ORDERS.llama.order, optionsOf(ORDERS.llama))).length).toBe(4);
    }
  });
  it('x1/X1.2/p-ratio: logits 1 apart at T = 0.5 give a probability ratio of e^2 = 7.389', () => {
    const l = TEMP_ROWS[0].logits;
    const [a, b] = OPENING_PAIR;
    expect(l[a] - l[b]).toBe(1);
    const p = softmax(l, 1 / 0.5);
    expect(p[a] / p[b]).toBeCloseTo(Math.exp(2), 10);
    expect(Math.exp(2)).toBeCloseTo(7.389, 3);
  });
  it('x1/X1.2/q-tie-limit: [3, 3, 1] approaches [0.5, 0.5, 0] as T goes to 0', () => {
    const p = softmax([3, 3, 1], 1 / 0.01);
    expect(p[0]).toBeCloseTo(0.5, 12);
    expect(p[1]).toBeCloseTo(0.5, 12);
    expect(p[2]).toBeLessThan(1e-80);
  });
  it('x1/start/q-max-subtract: exp(95) overflows float32; with the max subtracted the largest term is exactly 1', () => {
    expect(f32(Math.exp(95))).toBe(Infinity);
    expect(f32(Math.exp(88.7))).toBeLessThan(Infinity);
    expect(f32(Math.exp(88.8))).toBe(Infinity);
    const p = softmax([95, 94, 90], 1, true);
    expect(p.every(Number.isFinite)).toBe(true);
    expect(Math.abs(p.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(1e-6);
  });
  it('x1/start/q-msd-bin: bins 2, 5, 30, 100 from the top, k = 20: 13 still to pick in the third bin', () => {
    expect(msdSelectStep([2, 5, 30, 100], 20)).toEqual({ bin: 2, above: 7, need: 13, running: [2, 7, 37] });
  });
  it('x1/start/p-bytes and q-vocab-kib: 8 bytes per token, 593.5 KiB of counts for Qwen3-8B', () => {
    expect(rowBytes(VOCAB.qwen3)).toBe(1_215_488);
    expect(rowBytes(VOCAB.qwen3) * 64).toBe(77_791_232);
    expect((VOCAB.qwen3 * 4) / 1024).toBe(593.5);
  });
});

describe('what the X1.1 explorer lets the reader discover', () => {
  const temps = [0.1, 0.25, 0.5, 0.8, 1, 1.5, 2, 4];
  it("top-k's set never depends on T, in any order, double or float32", () => {
    for (const preset of CHAIN_ROWS)
      for (const o of ORDER_IDS)
        for (const float32 of [false, true]) {
          const sets = temps.map((T) => {
            const r = runChain(preset.row, { ...DEFAULT_SETTINGS, T, float32 }, ORDERS[o].order, optionsOf(ORDERS[o]));
            return ids(afterStage(r, 'top-k').alive).join(',');
          });
          expect(new Set(sets).size).toBe(1);
        }
  });
  it("under transformers' order top-p's set moves with T: more tokens above T = 1, fewer below", () => {
    for (const preset of CHAIN_ROWS) {
      const n = (T: number, o: 'hf' | 'llama') =>
        finalSet(runChain(preset.row, { ...DEFAULT_SETTINGS, repeat: 1, minP: 0, T }, ORDERS[o].order, optionsOf(ORDERS[o]))).length;
      expect(n(2, 'hf')).toBeGreaterThan(n(2, 'llama'));
      expect(n(0.5, 'hf')).toBeLessThan(n(0.5, 'llama'));
      expect(n(0.5, 'llama')).toBe(n(2, 'llama'));
    }
  });
  it('min-p after temperature depends on T; before temperature it does not', () => {
    const row = chainRow('tail');
    const s = (T: number) => ({ ...OFF, minP: 0.1, T });
    const before = [0.5, 2].map((T) => finalSet(runChain(row, s(T), ['min-p', 'temperature'])).length);
    const after = [0.5, 2].map((T) => finalSet(runChain(row, s(T), ['temperature', 'min-p'])).length);
    expect(before[0]).toBe(before[1]);
    expect(after[0]).not.toBe(after[1]);
  });
  it('top-p after top-k keeps no more than top-p over the full row (renormalisation)', () => {
    for (const preset of CHAIN_ROWS)
      for (const k of [3, 5, 8])
        for (const p of [0.5, 0.8, 0.9]) {
          const kp = finalSet(runChain(preset.row, { ...OFF, k, p }, ['top-k', 'top-p'])).length;
          const pk = finalSet(runChain(preset.row, { ...OFF, k, p }, ['top-p', 'top-k'])).length;
          expect(kp).toBeLessThanOrEqual(pk);
        }
  });
  it('the default settings on the peaked row: top-k keeps 8, top-p 6, min-p (0.05) none fewer', () => {
    const r = runChain(chainRow('peaked'), DEFAULT_SETTINGS, M5_ORDER);
    expect(r.snapshots.map((s) => s.kept)).toEqual([16, 16, 8, 6, 6, 6]);
  });
});

describe('ties go to the lowest id under M5', () => {
  it('top-k: a tie at the cut keeps the lower id; transformers keeps both', () => {
    const l = [1, 3, 2, 2, 0];
    expect(ids(topK(l, all(5), 3, 'lowest'))).toEqual([1, 2, 3]);
    expect(ids(topK(l, all(5), 2, 'lowest'))).toEqual([1, 2]);
    expect(ids(topK(l, all(5), 2, 'keep'))).toEqual([1, 2, 3]);
  });
  it('the stable descending order puts the lower id first among equals', () => {
    expect(rankDesc([1, 2, 0, 2, 2], all(5))).toEqual([1, 3, 4, 0, 2]);
  });
  it('argmax: lowest id for M5 and PyTorch, highest for llama.cpp Metal', () => {
    const tied = TEMP_ROWS.find((r) => r.id === 'tied')!.logits;
    expect(argmax(tied, 'lowest')).toBe(1);
    expect(argmax(tied, 'highest')).toBe(4);
    for (const r of TEMP_ROWS.filter((x) => x.id !== 'tied')) expect(argmax(r.logits, 'lowest')).toBe(argmax(r.logits, 'highest'));
  });
  it('the argmax never moves with T', () => {
    for (const r of TEMP_ROWS)
      for (const T of [0.05, 0.3, 1, 4]) {
        const p = softmax(r.logits, 1 / T);
        expect(argmax(p, 'lowest')).toBe(argmax(r.logits, 'lowest'));
      }
  });
  it('T <= 0 ends the chain greedy, on the lowest id among tied maxima', () => {
    const row = { logits: [0, 2, 1, 2], context: [] };
    const r = runChain(row, { ...OFF, T: 0 }, M5_ORDER);
    expect(r.greedy).toBe(true);
    expect(r.token).toBe(1);
  });
});

describe('the draw: inverse CDF with the strict rule (x1-facts E2)', () => {
  const l = fromProbs([0.5, 0.25, 0.25]);
  it('u on a boundary goes to the next token', () => {
    expect(drawInverseCdf(l, all(3), 0).token).toBe(0);
    expect(drawInverseCdf(l, all(3), 0.4999).token).toBe(0);
    expect(drawInverseCdf(l, all(3), 0.5).token).toBe(1);
    expect(drawInverseCdf(l, all(3), 0.75).token).toBe(2);
  });
  it('a u at or past the total falls back to the last survivor', () => {
    const d = drawInverseCdf(l, all(3), 1);
    expect(d.fallback).toBe(true);
    expect(d.token).toBe(2);
  });
  it('a dropped token is never drawn', () => {
    const alive = [true, false, true];
    for (const u of [0, 0.3, 0.6, 0.99]) expect(drawInverseCdf(l, alive, u).token).not.toBe(1);
  });
});

describe('temperature, softmax and greedy (X1.2)', () => {
  it('T -> 0 gives one-hot on a single maximum and an even split between exact ties', () => {
    const peaked = TEMP_ROWS[0].logits;
    expect(Math.max(...softmax(peaked, 1 / 0.05))).toBeCloseTo(1, 6);
    const tied = TEMP_ROWS[2].logits;
    const p = softmax(tied, 1 / 0.05);
    expect(p[1]).toBeCloseTo(0.5, 4);
    expect(p[4]).toBeCloseTo(0.5, 4);
    expect(p[1]).toBe(p[4]);
  });
  it('entropy: 0 bits for one-hot, log2 n for uniform, and it grows with T', () => {
    expect(entropyBits([1, 0, 0])).toBeCloseTo(0, 12);
    expect(entropyBits([0.25, 0.25, 0.25, 0.25])).toBeCloseTo(2, 12);
    const l = TEMP_ROWS[0].logits;
    expect(entropyBits(softmax(l, 1 / 0.5))).toBeLessThan(entropyBits(softmax(l, 1)));
    expect(entropyBits(softmax(l, 1))).toBeLessThan(entropyBits(softmax(l, 1 / 2)));
  });
  it('float32 softmax in lab 10.3 order stays within a few ulp of double and sums to 1', () => {
    for (const r of TEMP_ROWS)
      for (const T of [0.05, 0.5, 1, 4]) {
        const d = softmax(r.logits, 1 / T);
        const s = softmax(r.logits, 1 / T, true);
        s.forEach((v, i) => expect(Math.abs(v - d[i])).toBeLessThanOrEqual(1e-6 * Math.max(d[i], 1e-30) + 1e-12));
        expect(Math.abs(s.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(1e-6);
      }
  });
  it('dividing by T and multiplying by a rounded 1/T: identical for powers of two, different in the last bit for some T', () => {
    for (const r of TEMP_ROWS) {
      expect(reciprocalMismatch(r.logits, 0.5)).toEqual([]);
      expect(reciprocalMismatch(r.logits, 2)).toEqual([]);
    }
    const n = TEMP_ROWS.reduce((s, r) => s + reciprocalMismatch(r.logits, 3).length, 0);
    expect(n).toBeGreaterThan(0);
    for (const r of TEMP_ROWS)
      for (const i of reciprocalMismatch(r.logits, 3)) {
        const a = temperature(r.logits, 3, true, 'divide')[i];
        const b = temperature(r.logits, 3, true, 'reciprocal')[i];
        expect(Math.abs(a - b)).toBeLessThanOrEqual(Math.abs(a) * 2 ** -22);
      }
  });
  it('min-p uses stablescan.ts: threshold = max + ln(min-p)', () => {
    const l = chainRow('tail');
    const r = minPKeep(l.logits, all(16), 0.1);
    expect(r.threshold).toBe(minPThreshold(l.logits, 0.1));
  });
});

describe('the scene "One row through the chain"', () => {
  const r = sceneChain();
  const at = (s: StageId) => ids(afterStage(r, s).alive);
  it('penalties: the raw favourite, token 7 (used three times), falls to second place', () => {
    expect(argmax(SCENE_ROW.logits)).toBe(7);
    const pen = afterStage(r, 'penalties').logits;
    expect(rankDesc(pen, all(12)).slice(0, 2)).toEqual([2, 7]);
    expect(pen[7]).toBeCloseTo(3.4 / 1.25 - (3 * 0.2 + 0.3), 12);
    expect(pen[1]).toBeCloseTo(-0.8 * 1.25 - (0.2 + 0.3), 12);
    expect(pen[8]).toBeCloseTo(-0.4 * 1.25 - (2 * 0.2 + 0.3), 12);
  });
  it('top-k with k = 5 cuts between tokens 3 and 10, which tie; the lower id stays', () => {
    const pen = afterStage(r, 'penalties').logits;
    expect(pen[3]).toBe(pen[10]);
    expect(at('top-k')).toEqual([2, 3, 5, 7, 9]);
  });
  it('top-p at 0.9 drops one of the five, min-p at 0.1 one more, three reach the draw', () => {
    expect(at('top-p')).toEqual([2, 5, 7, 9]);
    expect(at('min-p')).toEqual([2, 5, 7]);
    expect(finalSet(r)).toEqual([2, 5, 7]);
    const th = afterStage(r, 'min-p').threshold!;
    expect(th).toBeCloseTo(3.2 + Math.log(0.1), 12);
    const pen = afterStage(r, 'penalties').logits;
    expect(pen[9]).toBeLessThan(th);
  });
  it('u is the top 24 bits of the Philox known-answer word 0x6627e8d5, and lands in token 2', () => {
    expect(KAT_WORD).toBe(0x6627e8d5);
    expect(KAT_U).toBe(0x6627e8 / 2 ** 24);
    expect(KAT_U).toBeCloseTo(0.39905, 5);
    expect(r.draw.order).toEqual([2, 7, 5]);
    expect(r.token).toBe(2);
    expect(r.draw.probs[0]).toBeCloseTo(0.7706, 4);
  });
  it('the settings are the ones the narration speaks', () => {
    expect(SCENE_SETTINGS).toMatchObject({ T: 0.8, k: 5, p: 0.9, minP: 0.1, repeat: 1.25, freq: 0.2, presence: 0.3 });
  });
});
