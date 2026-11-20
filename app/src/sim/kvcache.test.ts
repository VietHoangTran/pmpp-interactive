import { describe, expect, it } from 'vitest';
import {
  MAT_NAMES,
  aiDecode,
  aiPoints,
  aiPrefill,
  attention,
  baselineStep,
  cacheBytes,
  cachedStep,
  decodeAttentionAI,
  decodePhases,
  diffStep,
  gemmAI,
  generate,
  makeProblem,
  mustCompute,
  phaseTotals,
  scoreMarks,
} from './kvcache';

describe('which cells change when token n joins (Fig. 20.5)', () => {
  for (const [n, d, seed] of [[6, 4, 1], [6, 4, 7], [3, 2, 3], [9, 5, 11]] as const) {
    const diff = diffStep(n, d, seed);
    it(`n = ${n}, d = ${d}, seed ${seed}: no older cell changes, not even in the last bit`, () => {
      for (const name of MAT_NAMES) expect(diff.kind[name].filter((k) => k === 'changed')).toEqual([]);
      // Bit-identical, checked again directly on the values.
      for (const name of MAT_NAMES) {
        const a = diff.after[name];
        const b = diff.before[name];
        for (let r = 0; r < b.rows; r++)
          for (let c = 0; c < b.cols; c++) expect(Object.is(a.v[r * a.cols + c], b.v[r * b.cols + c])).toBe(true);
      }
    });
    it(`n = ${n}: X, Q, K, V and O gain exactly one new row`, () => {
      for (const name of ['X', 'Q', 'K', 'V', 'O'] as const) {
        const k = diff.kind[name];
        expect(k.length).toBe((n + 1) * d);
        const fresh = k.flatMap((x, i) => (x === 'new' ? [i] : []));
        expect(fresh).toEqual(Array.from({ length: d }, (_, c) => n * d + c));
        expect(k.filter((x) => x === 'masked').length).toBe(0);
      }
    });
    it(`n = ${n}: S and P gain a new row (n + 1 cells, (n, n) included); the rest of the new column is masked`, () => {
      for (const name of ['S', 'P'] as const) {
        const k = diff.kind[name];
        const w = n + 1;
        expect(k.length).toBe(w * w);
        expect(k.flatMap((x, i) => (x === 'new' ? [i] : []))).toEqual(Array.from({ length: w }, (_, c) => n * w + c));
        expect(k.flatMap((x, i) => (x === 'masked' ? [i] : []))).toEqual(Array.from({ length: n }, (_, r) => r * w + n));
      }
      // Minus infinity before the softmax, exactly zero after it.
      const w = n + 1;
      for (let r = 0; r < n; r++) {
        expect(diff.after.S.v[r * w + n]).toBe(-Infinity);
        expect(diff.after.P.v[r * w + n]).toBe(0);
      }
      expect(Number.isFinite(diff.after.S.v[n * w + n])).toBe(true);
    });
  }

  it('every row of P is a probability distribution over tokens 0..r', () => {
    const a = attention(makeProblem(8, 4, 2), 8);
    for (let r = 0; r < 8; r++) {
      const row = a.P.v.slice(r * 8, r * 8 + 8);
      expect(row.reduce((s, x) => s + x, 0)).toBeCloseTo(1, 12);
      for (let c = r + 1; c < 8; c++) expect(row[c]).toBe(0);
    }
  });

  it('the cells to compute for n = 6, d = 4 number 4 x 5 + 7 x 2 = 34', () => {
    const diff = diffStep(6, 4, 1);
    const need = mustCompute(diff);
    expect(MAT_NAMES.reduce((s, m) => s + need[m].size, 0)).toBe(34);
  });

  it('scores marks: hits, misses, extra unchanged and extra masked cells', () => {
    const diff = diffStep(6, 4, 1);
    const need = mustCompute(diff);
    expect(scoreMarks(diff, need)).toEqual({ hit: 34, missed: 0, extraSame: 0, extraMasked: 0, needed: 34 });
    // The figure's dark cells: also the masked new column of S and P (6 + 6 cells).
    const fig = { ...need, S: new Set([...need.S, ...[0, 1, 2, 3, 4, 5].map((r) => r * 7 + 6)]) };
    expect(scoreMarks(diff, fig).extraMasked).toBe(6);
    const s = scoreMarks(diff, { X: new Set([0, 24]) });
    expect(s).toEqual({ hit: 1, missed: 33, extraSame: 1, extraMasked: 0, needed: 34 });
  });
});

describe('recompute or cache: FLOPs per token', () => {
  it('hand counts at n = 3, D = 2', () => {
    // Baseline: 3 x 2 x 3 x 4 = 72, Q K^T and P V: 2 x 9 x 2 = 36 each.
    expect(baselineStep(3, 2)).toEqual({ projections: 72, scores: 36, values: 36, total: 144 });
    // Cached: 3 x 2 x 4 = 24, q'K^T and p'V: 2 x 3 x 2 = 12 each.
    expect(cachedStep(3, 2)).toEqual({ projections: 24, scores: 12, values: 12, total: 48 });
  });
  it('the baseline does exactly n times the work of a cached step', () => {
    for (const [n, D, l] of [[1, 64, 1], [7, 128, 2], [4096, 4096, 32], [32768, 512, 4]])
      expect(baselineStep(n, D, l).total).toBe(n * cachedStep(n, D, l).total);
  });
  it("question q-new-row-cost: q'K^T and p'V at n = 4096, d = 128 take 4 n d = 2,097,152 FLOPs", () => {
    const c = cachedStep(4096, 128);
    expect(c.scores + c.values).toBe(2_097_152);
  });
  it('generating T tokens sums the steps over n = N + 1 .. N + T', () => {
    const g = generate(7, 3, 4, 2);
    expect(g.cached).toBe(cachedStep(8, 4, 2).total + cachedStep(9, 4, 2).total + cachedStep(10, 4, 2).total);
    expect(g.baseline).toBe(8 * cachedStep(8, 4, 2).total + 9 * cachedStep(9, 4, 2).total + 10 * cachedStep(10, 4, 2).total);
  });
  it('cache bytes: Llama 2 7B shape (32 layers, D = 4096), 4096 tokens, 16-bit = 2 GiB (p. 506)', () => {
    expect(cacheBytes(4096, 4096, 32, 2)).toBe(2 * 2 ** 30);
  });
});

describe('arithmetic intensity, compulsory traffic only', () => {
  it('gemmAI by hand', () => {
    // 2 x 2 x 2 product in fp32: 16 FLOPs over 12 elements x 4 bytes.
    expect(gemmAI(2, 2, 2, 4)).toBeCloseTo(16 / 48, 12);
  });
  it("decode stays near 2/p whatever n; prefill climbs towards 2d/p", () => {
    for (const p of [4, 2]) {
      for (const n of [16, 1024, 16384]) expect(aiDecode(n, 128, p)).toBeLessThan(2 / p);
      expect(aiDecode(16384, 128, p)).toBeGreaterThan(0.98 * (2 / p));
      expect(aiPrefill(1, 128, p)).toBeLessThan(2 / p);
      expect(aiPrefill(16384, 128, p)).toBeGreaterThan(0.98 * ((2 * 128) / p));
      expect(aiPrefill(16384, 128, p)).toBeLessThan((2 * 128) / p);
    }
  });
  it('the four plotted points at n = 2048, d = 128, 16-bit', () => {
    const pts = Object.fromEntries(aiPoints(2048, 128, 2).map((x) => [x.key, x.ai]));
    expect(pts['prefill-proj']).toBeCloseTo((2 * 2048 * 128 * 128) / (2 * (2048 * 128 * 2 + 128 * 128)), 10);
    expect(pts['decode-proj']).toBeCloseTo((2 * 128 * 128) / (2 * (128 + 128 * 128 + 128)), 10);
    expect(pts['prefill-qk']).toBeGreaterThan(pts['decode-qk'] * 50);
  });
  it('question q-fp32-ai: a decode step reading K and V in fp32 does 0.5 FLOP per byte, 1 in 16-bit', () => {
    expect(decodeAttentionAI(4096, 128, 4)).toBeCloseTo(0.5, 3);
    expect(decodeAttentionAI(4096, 128, 2)).toBeCloseTo(1, 3);
    // Exactly the book's N / (1 + N) at p = 1 operation per element, doubled to FLOPs, over p.
    expect(decodeAttentionAI(4096, 128, 2)).toBeCloseTo(4096 / 4097, 12);
  });
});

describe('one decode step, phase by phase (scene decode-stream)', () => {
  const ph = decodePhases(4096, 32, 128, 2);
  it('projections read every weight once: 3 x 4096^2 x 2 bytes plus the input row', () => {
    expect(ph[0].bytesRead).toBe(2 * (4096 + 3 * 4096 * 4096));
    expect(ph[0].flops).toBe(6 * 4096 * 4096);
  });
  it('the score and value passes read the whole cache once each, at one multiply-add per element', () => {
    for (const k of [2, 4]) {
      const p = decodePhases(4096, 32, 128, 2)[k];
      expect(p.bytesRead).toBe(2 * 4096 * 4096);
      expect(p.flops).toBe(2 * 4096 * 4096);
      expect(p.flops / p.bytesRead).toBe(1);
    }
  });
  it('whole step near 2/p FLOP per byte, and its bytes grow with n', () => {
    expect(phaseTotals(ph).ai).toBeGreaterThan(0.99);
    expect(phaseTotals(ph).ai).toBeLessThan(1);
    // One more token: one more K row and one more V row to read, D x p bytes each.
    expect(phaseTotals(decodePhases(4097, 32, 128, 2)).bytes - phaseTotals(ph).bytes).toBe(2 * 4096 * 2);
  });
  it('cumulative totals by phase', () => {
    expect(phaseTotals(ph, 0)).toEqual({ bytes: 0, flops: 0, ai: 0 });
    expect(phaseTotals(ph, 1).bytes).toBe(ph[0].bytesRead + ph[0].bytesWritten);
  });
});
