import fs from 'node:fs';
import { describe, expect, it } from 'vitest';
import { bitsOf, keyAsc } from './floatkey';
import { topP } from './scan';
import {
  BEFORE_AFTER,
  CROSSING,
  INCLUSIVE_BUG,
  LAB116_RTOL,
  MINP_ROW,
  MLX_TIES,
  THRESHOLD_Q,
  VOCAB_PRESETS,
  bandCount,
  filterBeforeSort,
  minPKept,
  minPTestsF32,
  minPThreshold,
  minPView,
  plantedRow,
  rowView,
  sortDesc,
  sortPlan,
  sortShare,
  topPKept,
  zipfRow,
} from './nucleus';

function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("the questions' rows", () => {
  it('X1.5 predict: [0.5, 0.25, 0.125, 0.0625, 0.0625] at p = 0.875 keeps 3; token 2 crosses exactly', () => {
    const r = topP(CROSSING.probs, CROSSING.p, 'exclusive');
    expect(r.kept).toBe(3);
    expect(r.before).toEqual([0, 0.5, 0.75, 0.875, 0.9375]);
    expect(r.upto[2]).toBe(0.875);
    expect(topPKept(CROSSING.probs, CROSSING.p)).toBe(3);
  });
  it("q-mlx-ties: M5 keeps 2, mlx-lm keeps 5", () => {
    expect(topP(MLX_TIES.probs, MLX_TIES.p, 'exclusive').kept).toBe(2);
    expect(topP(MLX_TIES.probs, MLX_TIES.p, 'mlx').kept).toBe(5);
    expect(topP(MLX_TIES.probs, MLX_TIES.p, 'ascending').kept).toBe(2);
  });
  it('q-inclusive-bug: the inclusive test keeps 1, the rule keeps 2', () => {
    expect(topP(INCLUSIVE_BUG.probs, INCLUSIVE_BUG.p, 'inclusive').kept).toBe(1);
    expect(topP(INCLUSIVE_BUG.probs, INCLUSIVE_BUG.p, 'exclusive').kept).toBe(2);
  });
  it('q-before-after: before temperature 3, after 2; the scaled threshold is 3.70', () => {
    const v = minPView(BEFORE_AFTER.logits, BEFORE_AFTER.minP, BEFORE_AFTER.T, 1);
    expect(v.keptBefore).toBe(3);
    expect(v.keptAfter).toBe(2);
    expect(v.thrAfter / BEFORE_AFTER.T).toBeCloseTo(6 + Math.log(0.1), 12);
    expect(v.thrAfter / BEFORE_AFTER.T).toBeCloseTo(3.70, 2);
  });
  it('X1.6 predict: 5 + ln 0.05 = 2.004', () => {
    expect(minPThreshold(THRESHOLD_Q.lmax, THRESHOLD_Q.minP)).toBeCloseTo(2.004, 3);
  });
});

describe('top-p kept counts', () => {
  it("agree with scan.ts's exact rule on rows of hundredths", () => {
    const rand = mulberry(3);
    for (let t = 0; t < 500; t++) {
      const n = 2 + Math.floor(rand() * 12);
      const cuts = new Set<number>();
      while (cuts.size < n - 1) cuts.add(1 + Math.floor(rand() * 99));
      const c = [0, ...[...cuts].sort((a, b) => a - b), 100];
      const probs = c.slice(1).map((v, i) => v - c[i]).sort((a, b) => b - a).map((h) => h / 100);
      // Choose p away from the cut points so double and exact arithmetic cannot disagree.
      const p = (Math.floor(rand() * 99) + 1) / 100 + 0.005;
      expect(topPKept(probs, p)).toBe(topP(probs, Number(p.toFixed(6)), 'exclusive').kept);
    }
  });
  it('are monotone in p and keep everything at p >= 1', () => {
    const row = rowView(50257, 1.1, 4);
    let last = 0;
    for (let p = 0.01; p < 1; p += 0.01) {
      const k = topPKept(row.probs, p);
      expect(k).toBeGreaterThanOrEqual(last);
      last = k;
    }
    expect(topPKept(row.probs, 1)).toBe(50257);
  });
  it('peaked rows need a handful of tokens at p = 0.95; flat rows thousands', () => {
    expect(topPKept(rowView(151936, 3, 1).probs, 0.95)).toBeLessThanOrEqual(10);
    expect(topPKept(rowView(151936, 0.5, 1).probs, 0.95)).toBeGreaterThan(10000);
  });
});

describe('sorting', () => {
  it("sortDesc is lab 14.3's order: descending, ties by column, -0.0 equal to +0.0", () => {
    const rand = mulberry(8);
    for (let t = 0; t < 300; t++) {
      const n = 1 + Math.floor(rand() * 200);
      const xs = Array.from({ length: n }, () => [0, -0, 1, -1, 2.5, Math.fround(rand() * 4 - 2)][Math.floor(rand() * 6)]);
      const ref = xs
        .map((v, c) => ({ k: keyAsc(bitsOf(v)), c }))
        .sort((a, b) => (a.k !== b.k ? b.k - a.k : a.c - b.c))
        .map((x) => x.c);
      expect([...sortDesc(xs).cols]).toEqual(ref);
    }
  });
  it('zipfRow is seeded and shaped', () => {
    expect([...zipfRow(1000, 1, 5)]).toEqual([...zipfRow(1000, 1, 5)]);
    expect([...zipfRow(1000, 1, 5)]).not.toEqual([...zipfRow(1000, 1, 6)]);
    const flat = rowView(10000, 0, 1);
    expect(flat.probs[0] / flat.probs[9999]).toBeLessThan(1.3);
    expect(VOCAB_PRESETS).toContain(151936);
  });
});

describe('how much to sort (X1.5)', () => {
  it("llama.cpp's host route sorts 256 on peaked rows and widens on flat ones", () => {
    const peaked = sortPlan(rowView(151936, 2.5, 1), 0.95, 0);
    expect(peaked.sizes.adaptive).toBe(256);
    expect(peaked.widened).toBe(false);
    const flat = sortPlan(rowView(151936, 0.6, 1), 0.95, 0);
    expect(flat.widened).toBe(true);
    expect(flat.sizes.adaptive).toBe(151936);
    const withK = sortPlan(rowView(151936, 0.6, 1), 0.95, 40);
    expect(withK.sizes.topK).toBe(40);
    expect(withK.keptAfterTopK).toBeLessThanOrEqual(40);
  });
});

describe('filter before you sort (X1.6)', () => {
  it('is exactly the same set as top-p then min-p, on 1,000 seeded rows of 50,257', () => {
    const rand = mulberry(2026);
    let cDiffers = 0;
    let small = 0;
    for (let t = 0; t < 1000; t++) {
      const s = 0.4 + rand() * 2.6;
      const p = 0.5 + rand() * 0.49;
      const minP = [0.01, 0.02, 0.05, 0.1, 0.2][t % 5];
      const row = rowView(50257, s, t + 1);
      const f = filterBeforeSort(row, p, minP);
      expect(f.identical).toBe(true);
      expect(f.survivors).toBe(f.b);
      expect(f.routeA.length).toBe(Math.max(1, Math.min(f.a, f.b)));
      expect(f.routeC.length).toBeLessThanOrEqual(f.routeA.length); // renormalising inflates every probability
      if (f.cDiffers) cDiffers++;
      if (f.survivors < 1000) small++;
    }
    // Renormalising over the survivors is not the same rule, and min-p usually leaves few.
    expect(cDiffers).toBeGreaterThan(100);
    expect(small).toBeGreaterThan(500);
  }, 120_000);
  it('a min-p of 0.05 on a peaked row leaves tens of survivors out of 151,936', () => {
    const f = filterBeforeSort(rowView(151936, 1, 1), 0.95, 0.05);
    expect(f.survivors).toBeGreaterThanOrEqual(10);
    expect(f.survivors).toBeLessThan(100);
    expect(f.identical).toBe(true);
  });
  it('min-p = 0 removes the saving', () => {
    const f = filterBeforeSort(rowView(50257, 1, 2), 0.9, 0);
    expect(f.survivors).toBe(50257);
  });
});

describe('min-p before or after temperature (X1.6)', () => {
  it('after temperature is stricter for T < 1 and looser for T > 1; before does not move', () => {
    const v1 = minPView(MINP_ROW, 0.05, 1, 0.95);
    const lo = minPView(MINP_ROW, 0.05, 0.7, 0.95);
    const hi = minPView(MINP_ROW, 0.05, 1.5, 0.95);
    expect(v1.keptBefore).toBe(v1.keptAfter);
    expect(lo.keptBefore).toBe(v1.keptBefore);
    expect(hi.keptBefore).toBe(v1.keptBefore);
    expect(lo.keptAfter).toBeLessThan(v1.keptAfter);
    expect(hi.keptAfter).toBeGreaterThan(v1.keptAfter);
    expect(v1.keptBefore).toBe(8); // 5.0 down to 2.2 are at or above 2.004
  });
  it('a logit planted at the float threshold: the logit test keeps it, the float32 probability test drops it', () => {
    const { logits, planted } = plantedRow(0.05);
    const t = minPTestsF32(logits, 0.05);
    expect(logits[planted]).toBe(t.threshold);
    expect(t.logit[planted]).toBe(true);
    expect(t.prob[planted]).toBe(false);
    // Everywhere else the two tests agree.
    expect(t.logit.filter((v, i) => i !== planted && v !== t.prob[i]).length).toBe(0);
  });
  it('minPKept counts a prefix and keeps at least one', () => {
    expect(minPKept([5, 4, 3], 0.99)).toBe(1);
    expect(minPKept([5, 4, 3], 0)).toBe(3);
  });
});

describe('the float band and the lab', () => {
  it("LAB116_RTOL is lab 11.6's kRtol", () => {
    const src = fs.readFileSync(new URL('../../../labs/ch11/lab6_topp/main.cpp', import.meta.url), 'utf8');
    const m = /static const double kRtol = ([0-9.e-]+);/.exec(src);
    expect(m && Number(m[1])).toBe(LAB116_RTOL);
  });
  it('counts only tokens near p', () => {
    expect(bandCount([0, 0.5, 0.75, 0.875], 0.875)).toBe(1);
    expect(bandCount([0, 0.5, 0.75, 0.875], 1)).toBe(0);
  });
  it('reads lab X1.2 sort shares', () => {
    const s = sortShare([
      { name: 'sort_rows', rows: 64, cols: 151936, k: 0, seconds: 0.006 },
      { name: 'chain_composed', rows: 64, cols: 151936, k: 0, seconds: 0.008 },
      { name: 'sort_rows', rows: 64, cols: 151936, k: 40, seconds: 0.0001 },
      { name: 'chain_composed', rows: 64, cols: 151936, k: 40, seconds: 0.0004 },
    ]);
    expect(s.map((x) => x.share)).toEqual([0.75, 0.25]);
  });
});
