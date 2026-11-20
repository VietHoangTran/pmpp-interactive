import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  ROPE_BASE, compare, dot, makeCase, matvec, pairOf, permute, permuteSource, permuteVec, rope, ropeTables, seeded, theta, unpermuteVec,
} from './ropepair';

const vec = (n: number, seed: number) => {
  const r = seeded(seed);
  return Float32Array.from({ length: n }, () => Math.fround(r()));
};
const norm = (v: ArrayLike<number>) => Math.sqrt(dot(v, v));

describe('tables and pairs', () => {
  it('theta_j = base^(-2j / head), the first pair fastest', () => {
    expect(theta(0, 8)).toBe(1);
    expect(theta(1, 8)).toBeCloseTo(0.1, 15);
    expect(theta(3, 8)).toBeCloseTo(0.001, 15);
    expect(ROPE_BASE).toBe(10000);
  });
  it('the tables are float-rounded double values', () => {
    const t = ropeTables(4, 8);
    expect(t.cos.length).toBe(16);
    expect(t.cos[3 * 4 + 1]).toBe(Math.fround(Math.cos(3 * theta(1, 8))));
    expect(t.cos[1 * 4 + 0]).toBe(Math.fround(Math.cos(1)));
    expect(t.sin[2 * 4 + 2]).toBe(Math.fround(Math.sin(2 * theta(2, 8))));
  });
  it('NORM pairs neighbours, NEOX pairs across the halves', () => {
    expect([0, 1, 2, 3].map((j) => pairOf(j, 8, 'norm'))).toEqual([[0, 1], [2, 3], [4, 5], [6, 7]]);
    expect([0, 1, 2, 3].map((j) => pairOf(j, 8, 'neox'))).toEqual([[0, 4], [1, 5], [2, 6], [3, 7]]);
  });
});

describe('rotation', () => {
  const t = ropeTables(64, 16);
  it('position 0 is the identity in both pairings', () => {
    const x = vec(32, 1);
    expect(Array.from(rope(x, 0, 'norm', t))).toEqual(Array.from(x));
    expect(Array.from(rope(x, 0, 'neox', t))).toEqual(Array.from(x));
  });
  it('rotations preserve norms', () => {
    const x = vec(32, 2);
    for (const p of [1, 7, 63]) for (const m of ['norm', 'neox'] as const) expect(norm(rope(x, p, m, t))).toBeCloseTo(norm(x), 5);
  });
  it('the score depends only on the distance', () => {
    const q = vec(16, 3);
    const k = vec(16, 4);
    for (const m of ['norm', 'neox'] as const) {
      const s1 = dot(rope(q, 10, m, t), rope(k, 7, m, t));
      const s2 = dot(rope(q, 40, m, t), rope(k, 37, m, t));
      expect(s1).toBeCloseTo(s2, 4);
    }
  });
});

describe("llama.cpp's permute", () => {
  it('puts element b + head/2 next to element b within each head', () => {
    expect(permuteSource(8, 1)).toEqual([0, 4, 1, 5, 2, 6, 3, 7]);
    expect(permuteSource(8, 2)).toEqual([0, 2, 1, 3, 4, 6, 5, 7]);
  });
  it('unpermute undoes permute', () => {
    const y = vec(24, 5);
    expect(Array.from(unpermuteVec(permuteVec(y, 3), 3))).toEqual(Array.from(y));
  });
  it('NEOX on W x equals inverse-permuted NORM on (P W) x, for query heads', () => {
    const head = 16, nHead = 4, din = 10;
    const t = ropeTables(32, head);
    const W = vec(nHead * head * din, 6);
    const x = vec(din, 7);
    const ref = rope(matvec(W, x, nHead * head, din), 9, 'neox', t);
    const ker = rope(matvec(permute(W, nHead * head, din, nHead), x, nHead * head, din), 9, 'norm', t);
    expect(Array.from(unpermuteVec(ker, nHead))).toEqual(Array.from(ref));
  });
  it('and for key heads of a GQA model only with n_head_kv', () => {
    const head = 16, nHead = 4, nKv = 2, din = 10;
    const t = ropeTables(32, head);
    const W = vec(nKv * head * din, 8);
    const x = vec(din, 9);
    const ref = rope(matvec(W, x, nKv * head, din), 5, 'neox', t);
    const good = rope(matvec(permute(W, nKv * head, din, nHead, nKv), x, nKv * head, din), 5, 'norm', t);
    expect(Array.from(unpermuteVec(good, nHead, nKv))).toEqual(Array.from(ref));
    // Grouping K's rows by the query-head count (llama2.c export.py's shape, D6) mis-pairs them.
    const bad = rope(matvec(permute(W, nKv * head, din, nHead), x, nKv * head, din), 5, 'norm', t);
    expect(Array.from(unpermuteVec(bad, nHead))).not.toEqual(Array.from(ref));
  });
});

describe('the explorer cases', () => {
  const t = ropeTables(64, 8);
  it('the mismatched pairing runs, gives finite scores, and differs', () => {
    const c = makeCase(8);
    const r = compare(c, 12, 'norm', false, t, 12);
    expect(r.ker.scores.every(Number.isFinite)).toBe(true);
    expect(r.same).toBe(false);
    expect(r.maxScoreDiff).toBeGreaterThan(0.01);
  });
  it('the matched pairing and the permuted NORM kernel equal the reference', () => {
    for (const head of [8, 10, 12, 14, 16]) {
      const tt = ropeTables(64, head);
      const c = makeCase(head);
      expect(compare(c, 20, 'neox', false, tt, 20).same).toBe(true);
      expect(compare(c, 20, 'norm', true, tt, 20).same).toBe(true);
      expect(compare(c, 20, 'norm', false, tt, 20).same).toBe(false);
    }
  });
  it('permuting the weights for a NEOX kernel breaks it', () => {
    expect(compare(makeCase(8), 12, 'neox', true, t, 12).same).toBe(false);
  });
});

// Lab X3.1's driver builds the same tables on the host. The lab is written in parallel with this
// lesson; once its main.cpp exists this test checks that it uses the same base and computes in double.
const repo = path.resolve(__dirname, '../../..');
const x3 = path.join(repo, 'labs/x3');
const mains = fs.existsSync(x3)
  ? fs.readdirSync(x3).filter((d) => d.startsWith('lab1_')).map((d) => path.join(x3, d, 'main.cpp')).filter((f) => fs.existsSync(f))
  : [];
describe("lab X3.1's host tables", () => {
  it.skipIf(mains.length === 0)('use the same base and double precision as ropeTables', () => {
    const src = fs.readFileSync(mains[0], 'utf8');
    expect(src).toMatch(/10000(\.0)?/);
    expect(src).toMatch(/\bdouble\b/);
    expect(src).toMatch(/\bcos\s*\(/);
  });
});
