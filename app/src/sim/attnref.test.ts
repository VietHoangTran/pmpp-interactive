import { describe, expect, it } from 'vitest';
import {
  EXP_LIMIT_F32,
  attention,
  causalSoftmaxProblem,
  differing,
  dotTerms,
  makeAttnProblem,
  maxAbsDiff,
  softmaxRow,
  visible,
  visibleCount,
  type AttnProblem,
} from './attnref';

const rowSums = (P: number[][]) => P.map((r) => r.reduce((a, b) => a + b, 0));

describe('the problem generator', () => {
  it('is deterministic and uses the small value sets', () => {
    const a = makeAttnProblem(6, 4, 3);
    const b = makeAttnProblem(6, 4, 3);
    expect(a).toEqual(b);
    expect(a.Q.flat().every((x) => [-1, -0.5, 0, 0.5, 1].includes(x))).toBe(true);
    expect(a.V.flat().every((x) => Number.isInteger(x) && Math.abs(x) <= 3)).toBe(true);
    expect(makeAttnProblem(6, 4, 4)).not.toEqual(a);
  });
  it('dotTerms writes out q_i . k_j', () => {
    const p = makeAttnProblem(5, 3, 2);
    const t = dotTerms(p, 3, 1);
    expect(t.terms).toHaveLength(3);
    expect(t.value).toBeCloseTo(p.Q[3].reduce((s, q, k) => s + q * p.K[1][k], 0), 12);
  });
});

describe('the causal mask', () => {
  it('square: row i sees columns 0..i, so row i of P has i + 1 non-zeros', () => {
    const res = attention(makeAttnProblem(8, 4, 1));
    for (let i = 0; i < 8; i++) {
      expect(res.live[i]).toBe(i + 1);
      for (let j = 0; j < 8; j++) {
        if (j > i) expect(res.P[i][j]).toBe(0); // exactly zero, not merely small
        else expect(res.P[i][j]).toBeGreaterThan(0);
      }
    }
    expect(res.live[5]).toBe(6); // question ch20/20.2/q-row-nonzeros
  });
  it('lower-right alignment when the query is shorter than the keys', () => {
    expect(visibleCount(0, 1, 5)).toBe(5); // a one-token decode step sees every key: no mask
    expect([0, 1, 2, 3, 4].map((j) => visible(0, j, 2, 5))).toEqual([true, true, true, true, false]);
    expect([0, 1, 2, 3, 4].map((j) => visible(1, j, 2, 5))).toEqual([true, true, true, true, true]);
    const dec = attention(makeAttnProblem(500, 2, 1, 1));
    expect(dec.live).toEqual([500]); // question ch20/20.3/q-decode-row
  });
  it('rows sum to 1 in both precisions', () => {
    const p = makeAttnProblem(7, 5, 9);
    for (const s of rowSums(attention(p).P)) expect(s).toBeCloseTo(1, 12);
    for (const s of rowSums(attention(p, { precision: 'f32' }).P)) expect(Math.abs(s - 1)).toBeLessThan(1e-6);
  });
  it('without the mask, future tokens leak into O (row 0 changes, the last row does not)', () => {
    const p = makeAttnProblem(6, 4, 2);
    const on = attention(p);
    const off = attention(p, { mask: false });
    const rows = new Set(differing(off.O, on.O, 1e-12).map(([i]) => i));
    expect(rows.has(0)).toBe(true);
    expect(rows.has(5)).toBe(false); // the last row already saw everything
    // When overflow has turned O into NaN, finiteOnly reports no leak rather than every entry.
    const nanOff = attention(p, { mask: false, subtractMax: false, offset: 90, precision: 'f32' });
    const nanOn = attention(p, { mask: true, subtractMax: false, offset: 90, precision: 'f32' });
    expect(differing(nanOff.O, nanOn.O, 1e-6).length).toBeGreaterThan(0);
    expect(differing(nanOff.O, nanOn.O, 1e-6, true).every(([i, j]) => Number.isFinite(nanOff.O[i][j]))).toBe(true);
  });
});

describe('a hand-computed 3 x 3 case', () => {
  // Q = K = [[1,0],[0,1],[1,1]], d = 2, so the scaled logits are q.k / sqrt(2).
  const p: AttnProblem = {
    nq: 3,
    nk: 3,
    d: 2,
    Q: [[1, 0], [0, 1], [1, 1]],
    K: [[1, 0], [0, 1], [1, 1]],
    V: [[1, 0], [0, 1], [2, 2]],
  };
  const res = attention(p);
  const s = 1 / Math.sqrt(2);
  it('row 0 sees only itself', () => {
    expect(res.P[0]).toEqual([1, 0, 0]);
    expect(res.O[0]).toEqual([1, 0]);
  });
  it('row 1: logits [0, 1/sqrt 2]', () => {
    const b = Math.exp(s) / (1 + Math.exp(s));
    expect(res.P[1][0]).toBeCloseTo(1 - b, 14);
    expect(res.P[1][1]).toBeCloseTo(b, 14);
    expect(res.P[1][2]).toBe(0);
    expect(res.O[1][0]).toBeCloseTo(1 - b, 14);
    expect(res.O[1][1]).toBeCloseTo(b, 14);
  });
  it('row 2: logits [1/sqrt 2, 1/sqrt 2, 2/sqrt 2]', () => {
    const den = 2 * Math.exp(s) + Math.exp(2 * s);
    const a = Math.exp(s) / den;
    const c = Math.exp(2 * s) / den;
    expect(res.P[2][0]).toBeCloseTo(a, 14);
    expect(res.P[2][1]).toBeCloseTo(a, 14);
    expect(res.P[2][2]).toBeCloseTo(c, 14);
    expect(res.O[2][0]).toBeCloseTo(a + 2 * c, 14);
    expect(res.L[2]).toBeCloseTo(Math.log(den), 14);
  });
});

describe('subtracting the max (Eq. 20.2)', () => {
  it('changes nothing in float64, even at +90', () => {
    const p = makeAttnProblem(6, 4, 5);
    const stable = attention(p, { offset: 90 });
    const naive = attention(p, { offset: 90, subtractMax: false });
    expect(maxAbsDiff(stable.P, naive.P)).toBeLessThan(1e-12);
  });
  it('in float32 the naive form overflows to NaN at +90, the stable form does not', () => {
    const p = makeAttnProblem(6, 4, 5);
    const naive = attention(p, { offset: 90, subtractMax: false, precision: 'f32' });
    expect(naive.P.flat().some(Number.isNaN)).toBe(true);
    const stable = attention(p, { offset: 90, precision: 'f32' });
    const base = attention(p, { precision: 'f32' });
    expect(stable.P.flat().some(Number.isNaN)).toBe(false);
    expect(maxAbsDiff(stable.P, base.P)).toBeLessThan(1e-4); // same P, to float32 rounding of the +90
  });
  it('without an offset both forms agree in float32: the max matters only for overflow', () => {
    const p = makeAttnProblem(6, 4, 5);
    const a = attention(p, { precision: 'f32' });
    const b = attention(p, { subtractMax: false, precision: 'f32' });
    expect(maxAbsDiff(a.P, b.P)).toBeLessThan(1e-6);
  });
  it('a row of float32 logits all equal to 100: NaN everywhere without the max, 1/n with it', () => {
    const row = [100, 100, 100, 100];
    expect(softmaxRow(row, false, 'f32').p.every(Number.isNaN)).toBe(true);
    expect(softmaxRow(row, true, 'f32').p).toEqual([0.25, 0.25, 0.25, 0.25]);
    expect(EXP_LIMIT_F32).toBeGreaterThan(88.7);
    expect(EXP_LIMIT_F32).toBeLessThan(88.8);
    expect(Math.fround(Math.exp(88.7))).toBeLessThan(Infinity);
    expect(Math.fround(Math.exp(88.8))).toBe(Infinity);
  });
});

describe('the 1/sqrt(d) scale', () => {
  it('without it rows of P are sharper (larger peak on average)', () => {
    const p = makeAttnProblem(8, 6, 11);
    const peak = (P: number[][]) => P.reduce((s, r) => s + Math.max(...r), 0) / P.length;
    expect(peak(attention(p, { scale: false }).P)).toBeGreaterThan(peak(attention(p).P));
  });
  it('is 1/sqrt(d): logits grow by sqrt(128), about 11.3, at d = 128 when it is left out', () => {
    expect(Math.sqrt(128)).toBeCloseTo(11.31, 2); // question ch20/20.2/q-no-scale
  });
});

describe('the softmax kernel input', () => {
  it('holds every entry of S, masked or not, and the reference answer', () => {
    const k = causalSoftmaxProblem(6, 0);
    expect(k.S.every((r) => r.every((x) => Number.isFinite(x)))).toBe(true);
    expect(k.refP[2].slice(3)).toEqual([0, 0, 0]);
    for (const s of rowSums(k.refP)) expect(Math.abs(s - 1)).toBeLessThan(1e-6);
  });
});
