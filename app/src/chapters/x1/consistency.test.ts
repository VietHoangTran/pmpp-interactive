import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import * as chain from '../../sim/chain';
import { inverseCdf, uniformsFor } from '../../sim/draw';
import { minPKept, topPKept } from '../../sim/nucleus';
import { bytesPerLogit } from '../../sim/penalties';
import { KATS, m5U } from '../../sim/philox';
import { LAB_X11, MAX_PASSES, SELECT_PER_KEY, SORT_PER_KEY, referenceTopK } from '../../sim/topkselect';
import { chapter } from './index';
import { questionsWrap } from './Wrap';

// Module X1's five models were built by three groups that never imported each other's work:
// chain.ts (X1.1 to X1.3), topkselect.ts and nucleus.ts (X1.4 to X1.6), philox.ts and draw.ts
// (X1.7, X1.8). This file ties them together: wherever two models compute the same thing, they
// must agree; the byte counts the pages and the labs quote must be the same numbers; and the
// lab paths the models name must exist.

const repo = path.resolve(__dirname, '../../../..');

/** A small deterministic generator for test rows (not the course's Philox). */
function lcg(seed: number) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32);
}
const rowOf = (n: number, seed: number, levels = 0) => {
  const r = lcg(seed);
  return Array.from({ length: n }, () => {
    const v = r() * 12 - 6;
    return levels ? Math.round(v * levels) / levels : v;
  });
};
const sortedDesc = (xs: readonly number[]) => [...xs].sort((a, b) => b - a);
const count = (keep: readonly boolean[]) => keep.filter(Boolean).length;

describe('the random numbers', () => {
  it("the chain scene's u is Random123's first known answer, read through M5's mapping", () => {
    expect(chain.KAT_WORD).toBe(KATS[0].expected[0]);
    expect(chain.KAT_U).toBe(m5U(0n, 0n, 0n));
  });
  it("draw.ts's uniforms are philox.ts's M5 uniforms", () => {
    for (const [seed, stream, step] of [[0, 0, 0], [1, 2, 3], [42, 7, 1000], [0xffffffff, 5, 0]])
      expect(uniformsFor(seed, stream, step, 1)[0]).toBe(m5U(BigInt(seed), BigInt(stream), BigInt(step)));
  });
});

describe('top-k: the chain model and the radix-select model pick the same set', () => {
  it('on random rows, with and without ties', () => {
    for (let t = 0; t < 200; t++) {
      const n = 5 + (t % 60);
      const logits = rowOf(n, 1000 + t, t % 2 ? 4 : 0); // odd rows have many exact ties
      const k = 1 + (t % n);
      const alive = logits.map(() => true);
      const fromChain = chain.topK(logits, alive, k, 'lowest').flatMap((a, i) => (a ? [i] : []));
      expect(fromChain).toEqual(referenceTopK(logits, k));
    }
  });
});

describe('top-p and min-p: the chain model and the nucleus model keep the same counts', () => {
  it('top-p, away from the rounding band of the chain model', () => {
    let compared = 0;
    for (let t = 0; t < 300; t++) {
      const logits = rowOf(8 + (t % 40), 5000 + t);
      const p = [0.5, 0.8, 0.9, 0.95][t % 4];
      const probs = chain.survivorProbs(logits, logits.map(() => true));
      const sorted = sortedDesc(probs);
      let before = 0;
      const near = sorted.some((x) => {
        const hit = Math.abs(before - p) < 1e-5;
        before += x;
        return hit;
      });
      if (near) continue; // chain.ts rounds to 6 decimals; nucleus.ts does not
      compared++;
      expect(count(chain.topPKeep(logits, logits.map(() => true), p))).toBe(topPKept(sorted, p));
    }
    expect(compared).toBeGreaterThan(250);
  });
  it('min-p, both in logit space against max + ln(min-p)', () => {
    for (let t = 0; t < 300; t++) {
      const logits = rowOf(8 + (t % 40), 9000 + t);
      const minP = [0.01, 0.05, 0.1, 0.3][t % 4];
      const { keep } = chain.minPKeep(logits, logits.map(() => true), minP);
      expect(count(keep)).toBe(minPKept(sortedDesc(logits), minP));
    }
  });
});

describe('the draw: the chain model and the kernel model pick the same token', () => {
  it('on random rows, for u away from every boundary', () => {
    let compared = 0;
    for (let t = 0; t < 300; t++) {
      const logits = rowOf(4 + (t % 30), 13000 + t);
      const u = lcg(t + 1)();
      const alive = logits.map(() => true);
      const d = chain.drawInverseCdf(logits, alive, u);
      // The same survivors in the same order, as weights exp(l - max): draw.ts compares with u x total.
      const lmax = Math.max(...logits);
      const weights = d.order.map((i) => Math.exp(logits[i] - lmax));
      const total = weights.reduce((a, b) => a + b, 0);
      if (d.cdf.some((c) => Math.abs(c - u) < 1e-9)) continue;
      compared++;
      const k = inverseCdf(weights, u);
      expect(k.index).toBe(d.at);
      expect(k.total).toBeCloseTo(total, 12);
    }
    expect(compared).toBeGreaterThan(290);
  });
});

describe('the byte counts the pages and labs quote', () => {
  it('select 20 and sort 80 per key (X1.4, lab X1.1), a penalty pass 12 per logit (X1.3)', () => {
    expect(SELECT_PER_KEY).toBe(20);
    expect(SORT_PER_KEY).toBe(80);
    expect(bytesPerLogit(1)).toBe(12);
    const main = fs.readFileSync(path.join(repo, 'labs/x1/lab1_topk/main.cpp'), 'utf8');
    expect(main).toContain('static const double kSortBytesPerKey = 80.0;');
  });
  it("the summary's fused-penalty question: five reads of a logit and its count", () => {
    const q = questionsWrap['x1/wrap/q-fuse-select'];
    expect(q.kind).toBe('numeric');
    if (q.kind === 'numeric') expect(q.answer).toBe((MAX_PASSES + 1) * (4 + 4));
    expect((MAX_PASSES + 1) * (4 + 4)).toBeGreaterThan(bytesPerLogit(1) + SELECT_PER_KEY);
  });
  it("lab X1.2's 29 and 12 bytes per token, as the labs page states them", () => {
    const main = fs.readFileSync(path.join(repo, 'labs/x1/lab2_filters/main.cpp'), 'utf8');
    expect(main).toContain('const double unfusedBytes = 29.0, fusedBytes = 12.0;');
    const labs = fs.readFileSync(path.join(__dirname, 'Labs.tsx'), 'utf8');
    expect(labs).toContain('logits again: 29 bytes');
    expect(labs).toContain('probabilities once: 12');
  });
  it("lab X1.3's driver checks the same first word as the scene and X1.7", () => {
    const main = fs.readFileSync(path.join(repo, 'labs/x1/lab3_sampler/main.cpp'), 'utf8');
    expect(main).toContain('0x' + chain.KAT_WORD.toString(16) + 'u');
  });
});

describe('the paths and ids the models and pages name', () => {
  it("every lab of chain.ts's LAB list exists, and the x1 ones are the labs page's", () => {
    const dirs: Record<string, string> = {
      '9.3': 'ch09/lab3_tokencounts', '10.3': 'ch10/lab3_softmax_argmax', '11.6': 'ch11/lab6_topp',
      '12.4': 'ch12/lab4_candidates', '14.3': 'ch14/lab3_sortrows',
      'X1.1': 'x1/lab1_topk', 'X1.2': 'x1/lab2_filters', 'X1.3': 'x1/lab3_sampler',
    };
    for (const l of Object.values(chain.LAB)) {
      expect(dirs[l.num], l.num).toBeTruthy();
      expect(fs.existsSync(path.join(repo, 'labs', dirs[l.num], 'solution')), dirs[l.num]).toBe(true);
    }
    expect(LAB_X11).toBe(dirs['X1.1']);
  });
  it("the chapter's sections are the design's, in order, and every one has its file", () => {
    expect(chapter.sections.map((s) => s.id)).toEqual(['start', 'X1.1', 'X1.2', 'X1.3', 'X1.4', 'X1.5', 'X1.6', 'X1.7', 'X1.8', 'labs', 'wrap']);
    for (const f of ['SX10', 'SX11', 'SX12', 'SX13', 'SX14', 'SX15', 'SX16', 'SX17', 'SX18', 'Labs', 'Wrap'])
      expect(fs.existsSync(path.join(__dirname, f + '.tsx')), f).toBe(true);
    expect(chapter.sections.every((s) => s.pages === null)).toBe(true);
  });
  it('the sections load and their question ids carry their own prefix', async () => {
    for (const s of chapter.sections) {
      const m = await s.load();
      const prefix = `x1/${s.id}/`;
      for (const id of Object.keys(m.questions)) {
        expect(m.questions[id].id).toBe(id);
        expect(id.startsWith(prefix), id).toBe(true);
      }
    }
  });
});
