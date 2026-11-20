// Synthetic vocabulary rows and the truncations of module X1, sections X1.5 (top-p) and X1.6
// (min-p). All of it is ours: sampling is not in the book. The top-p rule is lab 11.6's
// ("keep token i when i = 0, or p >= 1, or the mass ranked before it is below p"), the rule
// llama.cpp, transformers and vLLM reduce to on rows that sum to 1 (notes/today/ch11-facts.md
// D2, D4); the worked rows go through scan.ts's topP. Min-p is Nguyen et al.'s threshold
// p_base x p_max (notes/today/x1-facts.md D9) tested in logit space, as llama.cpp's host does
// (ch12-facts D3).
//
// What it models, exactly:
//  - Rows: seeded Zipf-like logits. The token of rank r (0 = most likely) gets
//    OFFSET - s ln(r + 1) plus a seeded jitter of +-JITTER/2, rounded to float32, and a seeded
//    column (token id). s = 0 is flat, larger s is more peaked. OFFSET only puts the largest
//    logit near 10, where a model's might be; nothing depends on it. Not a model's output.
//  - Sorting: descending by value, ties by increasing column, -0.0 equal to +0.0: lab 14.3's
//    order. Done by packing floatkey.ts's descending key and the position into one float64.
//  - Probabilities: a softmax in double over the whole row (the denominator summed in column
//    order, as a row-softmax kernel over the unsorted row would; the max is the row max).
//  - Top-p kept count: lab 11.6's rule in double. Min-p: logit >= l_max + ln p_base before
//    temperature, l_max + T ln p_base after (in units of the unscaled logits).
//  - Sort sizes under four strategies: the full row; top-k first (k values); llama.cpp's host
//    route (more than 1,024 candidates: partial-sort the top 256, and when those hold less
//    than p, widen; we count a widened sort as the whole row) (ch14-facts D3); min-p first
//    (only the survivors).
//  - Filter before sort: route A is llama.cpp's order over the full row (sort, softmax, top-p,
//    then min-p); route B compacts min-p's survivors in column order (lab 12.4), sorts only
//    them and runs top-p with full-row probabilities; route C does the same but renormalises
//    over the survivors, which is what a chain would do if min-p simply ran first.
//  - The float band: lab 11.6 accepts either decision for a token whose exact mass before it
//    is within 1e-4 (relative to that mass) of p (labs/ch11/lab6_topp/main.cpp, kRtol).
//  - float32: only where asked (the min-p tests at the threshold): every operation rounded to
//    float32, round to nearest, sums in column order. Real kernels sum in other orders, and
//    Metal's default math mode is not IEEE-strict (ch05-facts C7, x1-facts C5).
//
// What it leaves out: real model distributions, time, memory traffic, NaN logits, and
// top-p's min_keep and similar options of the real samplers.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.
import { bitsOf, keyDesc } from './floatkey';

export const VOCAB_PRESETS = [32000, 50257, 151936, 201088];
/** Largest row the explorers and sortDesc accept: positions must fit in 18 bits. */
export const MAX_VOCAB = 262144;
export const JITTER = 0.25;
export const OFFSET = 10;
/** lab 11.6's relative tolerance on the mass before each token (labs/ch11/lab6_topp/main.cpp). */
export const LAB116_RTOL = 1e-4;
/** llama.cpp's host top-p partial-sorts this many candidates first ... */
export const ADAPTIVE_TOP = 256;
/** ... when there are more than this many (ch14-facts D3). */
export const ADAPTIVE_MIN = 1024;

const f32 = Math.fround;

function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A seeded Zipf-like row of logits in column order, as float32. */
export function zipfRow(V: number, s: number, seed = 1, jitter = JITTER): Float32Array {
  if (V < 1 || V > MAX_VOCAB) throw new Error(`vocabulary ${V} outside 1..${MAX_VOCAB}`);
  const rand = mulberry(seed * 2654435761 + Math.round(s * 1000));
  const perm = new Uint32Array(V);
  for (let i = 0; i < V; i++) perm[i] = i;
  for (let i = V - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const t = perm[i];
    perm[i] = perm[j];
    perm[j] = t;
  }
  const out = new Float32Array(V);
  for (let r = 0; r < V; r++) out[perm[r]] = OFFSET - s * Math.log(r + 1) + jitter * (rand() - 0.5);
  return out;
}

export interface SortedRow {
  /** Columns (token ids), largest logit first, ties by increasing column. */
  cols: Uint32Array;
  logits: Float32Array;
}
const POS = 2 ** 18;
/**
 * Sorts values descending, stably: lab 14.3's order. `cols` gives each value's column and must
 * be increasing (a whole row, or a compaction's output); without it the column is the position.
 */
export function sortDesc(values: ArrayLike<number>, cols?: ArrayLike<number>): SortedRow {
  const n = values.length;
  if (n > POS) throw new Error('row too long');
  const packed = new Float64Array(n);
  for (let i = 0; i < n; i++) packed[i] = keyDesc(bitsOf(values[i])) * POS + i;
  packed.sort();
  const outCols = new Uint32Array(n);
  const outVals = new Float32Array(n);
  for (let j = 0; j < n; j++) {
    const i = packed[j] % POS;
    outCols[j] = cols ? cols[i] : i;
    outVals[j] = values[i];
  }
  return { cols: outCols, logits: outVals };
}

/** The softmax denominator over a row with its max subtracted, summed in the order given, in double. */
export function denominator(logits: ArrayLike<number>, lmax: number, T = 1): number {
  let z = 0;
  for (let i = 0; i < logits.length; i++) z += Math.exp((logits[i] - lmax) / T);
  return z;
}
export function rowMax(logits: ArrayLike<number>): number {
  let m = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > m) m = logits[i];
  return m;
}
/** Probabilities of sorted logits with a given denominator (double). */
export function probsOf(sortedLogits: ArrayLike<number>, lmax: number, z: number, T = 1): Float64Array {
  const p = new Float64Array(sortedLogits.length);
  for (let i = 0; i < p.length; i++) p[i] = Math.exp((sortedLogits[i] - lmax) / T) / z;
  return p;
}
export function exclusiveSum(xs: ArrayLike<number>): Float64Array {
  const out = new Float64Array(xs.length);
  let s = 0;
  for (let i = 0; i < xs.length; i++) {
    out[i] = s;
    s += xs[i];
  }
  return out;
}

/** Lab 11.6's rule on probabilities sorted largest first: how many tokens top-p keeps. */
export function topPKept(probs: ArrayLike<number>, p: number): number {
  const n = probs.length;
  if (n === 0) return 0;
  if (p >= 1) return n;
  let before = 0;
  for (let i = 0; i < n; i++) {
    if (i > 0 && !(before < p)) return i;
    before += probs[i];
  }
  return n;
}

export type Placement = 'before' | 'after';
/** Min-p's threshold in units of the unscaled logits. */
export function minPThreshold(lmax: number, minP: number, T = 1, placement: Placement = 'before'): number {
  if (!(minP > 0)) return -Infinity;
  return lmax + (placement === 'after' ? T : 1) * Math.log(minP);
}
/** How many of the sorted logits min-p keeps (a prefix, at least 1). */
export function minPKept(sortedLogits: ArrayLike<number>, minP: number, T = 1, placement: Placement = 'before'): number {
  const thr = minPThreshold(sortedLogits[0], minP, T, placement);
  let i = 0;
  while (i < sortedLogits.length && sortedLogits[i] >= thr) i++;
  return Math.max(1, i);
}

// ---------------------------------------------------------------- a whole synthetic row
export interface RowView {
  V: number;
  s: number;
  seed: number;
  logits: Float32Array;
  sorted: SortedRow;
  lmax: number;
  z: number;
  /** Full-row probabilities in sorted order. */
  probs: Float64Array;
  /** Mass before each token in sorted order. */
  before: Float64Array;
}
export function rowView(V: number, s: number, seed = 1): RowView {
  const logits = zipfRow(V, s, seed);
  const sorted = sortDesc(logits);
  const lmax = sorted.logits[0];
  const z = denominator(logits, lmax);
  const probs = probsOf(sorted.logits, lmax, z);
  return { V, s, seed, logits, sorted, lmax, z, probs, before: exclusiveSum(probs) };
}

/** Tokens whose top-p decision lies inside lab 11.6's band: |mass before - p| <= rtol x mass before. */
export function bandCount(before: ArrayLike<number>, p: number, rtol = LAB116_RTOL): number {
  if (p >= 1) return 0;
  let c = 0;
  for (let i = 1; i < before.length; i++) if (Math.abs(before[i] - p) <= rtol * before[i]) c++;
  return c;
}

// ---------------------------------------------------------------- X1.5: how much to sort
export interface SortPlan {
  V: number;
  /** Top-p's kept count over the full row (top-k off). */
  keptFull: number;
  /** Top-p's kept count after top-k, renormalised over the k survivors; null when top-k is off. */
  keptAfterTopK: number | null;
  /** Min-p's survivors over the full row; null when min-p is off. */
  minPSurvivors: number | null;
  sizes: { full: number; topK: number | null; adaptive: number; minPFirst: number | null };
  /** llama.cpp's host route had to widen past the top 256. */
  widened: boolean;
}
export function sortPlan(row: RowView, p: number, k: number, minP = 0.05): SortPlan {
  const V = row.V;
  const keptFull = topPKept(row.probs, p);
  let keptAfterTopK: number | null = null;
  if (k > 0) {
    const kk = Math.min(k, V);
    const sub = row.sorted.logits.subarray(0, kk);
    keptAfterTopK = topPKept(probsOf(sub, row.lmax, denominator(sub, row.lmax)), p);
  }
  const widened = V > ADAPTIVE_MIN && keptFull > ADAPTIVE_TOP;
  const adaptive = V > ADAPTIVE_MIN ? (widened ? V : ADAPTIVE_TOP) : V;
  const minPSurvivors = minP > 0 ? minPKept(row.sorted.logits, minP) : null;
  return {
    V, keptFull, keptAfterTopK, minPSurvivors, widened,
    sizes: { full: V, topK: k > 0 ? Math.min(k, V) : null, adaptive, minPFirst: minPSurvivors },
  };
}

// ---------------------------------------------------------------- X1.6: filter before you sort
export interface FilterFirst {
  /** Min-p's threshold, l_max + ln p_base. */
  threshold: number;
  /** Min-p's survivors: what the sort in route B handles. */
  survivors: number;
  /** Top-p's kept prefix over the full row, and min-p's. */
  a: number;
  b: number;
  /** Kept tokens, largest first. */
  routeA: Uint32Array;
  routeB: Uint32Array;
  routeC: Uint32Array;
  identical: boolean;
  /** Route C keeps a different set. */
  cDiffers: boolean;
  /** Tokens whose top-p decision lies inside lab 11.6's band, over the full row. */
  band: number;
  /** Those of them inside min-p's prefix, where the decision can change the kept set. */
  bandInMinP: number;
}
const sameIds = (x: Uint32Array, y: Uint32Array) => x.length === y.length && x.every((v, i) => v === y[i]);

/**
 * Route A: llama.cpp's top-p then min-p over the full sorted row. Route B: min-p first as a
 * stable compaction, sort only the survivors, top-p with full-row probabilities. Route C: as B
 * but top-p renormalises over the survivors.
 */
export function filterBeforeSort(row: RowView, p: number, minP: number): FilterFirst {
  const { logits, sorted, lmax, z, probs, before } = row;
  const threshold = minPThreshold(lmax, minP);
  // Route A.
  const a = topPKept(probs, p);
  const b = minPKept(sorted.logits, minP);
  const routeA = sorted.cols.slice(0, Math.max(1, Math.min(a, b)));
  // Route B: the compaction, in column order.
  const keepCols: number[] = [];
  const keepVals: number[] = [];
  for (let c = 0; c < logits.length; c++)
    if (logits[c] >= threshold) {
      keepCols.push(c);
      keepVals.push(logits[c]);
    }
  const sub = sortDesc(keepVals, keepCols);
  const pB = probsOf(sub.logits, lmax, z);
  const routeB = sub.cols.slice(0, Math.max(1, topPKept(pB, p)));
  // Route C: renormalised over the survivors.
  const pC = probsOf(sub.logits, lmax, denominator(sub.logits, lmax));
  const routeC = sub.cols.slice(0, Math.max(1, topPKept(pC, p)));
  return {
    threshold, survivors: keepCols.length, a, b, routeA, routeB, routeC,
    identical: sameIds(routeA, routeB), cDiffers: !sameIds(routeA, routeC), band: bandCount(before, p),
    bandInMinP: bandCount(before.subarray(0, b), p),
  };
}

// ---------------------------------------------------------------- X1.6: before or after temperature
export interface MinPToken {
  index: number;
  logit: number;
  /** Kept by min-p before temperature (llama.cpp, mlx-lm). */
  before: boolean;
  /** Kept by min-p after temperature (the paper, transformers). */
  after: boolean;
  /** Kept by top-p at this temperature. */
  topP: boolean;
}
export interface MinPView {
  T: number;
  minP: number;
  p: number;
  lmax: number;
  thrBefore: number;
  thrAfter: number;
  keptBefore: number;
  keptAfter: number;
  /** Top-p's kept count at temperature T. */
  keptTopP: number;
  tokens: MinPToken[];
}
/** The explorer's view of a small row given largest first. */
export function minPView(sortedLogits: readonly number[], minP: number, T: number, p: number): MinPView {
  const lmax = sortedLogits[0];
  const thrBefore = minPThreshold(lmax, minP, T, 'before');
  const thrAfter = minPThreshold(lmax, minP, T, 'after');
  const probs = probsOf(sortedLogits, lmax, denominator(sortedLogits, lmax, T), T);
  const keptTopP = topPKept(probs, p);
  const tokens = sortedLogits.map((logit, index) => ({
    index, logit, before: logit >= thrBefore, after: logit >= thrAfter, topP: index < keptTopP,
  }));
  return {
    T, minP, p, lmax, thrBefore, thrAfter, keptTopP,
    keptBefore: Math.max(1, tokens.filter((t) => t.before).length),
    keptAfter: Math.max(1, tokens.filter((t) => t.after).length),
    tokens,
  };
}

/**
 * Min-p's two tests at T = 1 in float32 (round to nearest, sums in order): the logit test
 * l >= l_max + logf(p), with the threshold computed as llama.cpp's host does, and the
 * probability test q >= p x q_max on a float32 softmax, as transformers writes it.
 */
export function minPTestsF32(logits: readonly number[], minP: number): { threshold: number; logit: boolean[]; prob: boolean[] } {
  const xs = logits.map(f32);
  const lmax = xs.reduce((m, v) => (v > m ? v : m), -Infinity);
  const threshold = f32(lmax + f32(Math.log(minP)));
  let z = 0;
  for (const v of xs) z = f32(z + f32(Math.exp(f32(v - lmax))));
  const q = xs.map((v) => f32(f32(Math.exp(f32(v - lmax))) / z));
  const qmax = q.reduce((m, v) => (v > m ? v : m), 0);
  const lim = f32(f32(minP) * qmax);
  return { threshold, logit: xs.map((v) => v >= threshold), prob: q.map((v) => v >= lim) };
}

/** The 16-token row the min-p explorer opens on (largest first; the largest logit is 5.0). */
export const MINP_ROW = [5.0, 4.2, 3.9, 3.5, 3.1, 2.8, 2.5, 2.2, 1.9, 1.6, 1.2, 0.8, 0.3, -0.4, -1.1, -2.0];
/**
 * MINP_ROW plus one logit planted exactly at llama.cpp's float threshold for this min-p
 * (max + logf(p) in float32), in its sorted place: 17 tokens. Lab X1.2 tests such a logit.
 */
export function plantedRow(minP: number): { logits: number[]; planted: number } {
  const thr = f32(MINP_ROW[0] + f32(Math.log(minP)));
  const logits = [...MINP_ROW, thr].sort((x, y) => y - x);
  return { logits, planted: logits.indexOf(thr) };
}

// ---------------------------------------------------------------- the questions' rows
/** X1.5 predict: the crossing token's inclusive sum is exactly p. */
export const CROSSING = { probs: [0.5, 0.25, 0.125, 0.0625, 0.0625], p: 0.875 };
/** X1.5 q-mlx-ties. */
export const MLX_TIES = { probs: [0.5, 0.125, 0.125, 0.125, 0.125], p: 0.6 };
/** X1.5 q-inclusive-bug. */
export const INCLUSIVE_BUG = { probs: [0.5, 0.25, 0.125, 0.125], p: 0.75 };
/** X1.6 q-before-after. */
export const BEFORE_AFTER = { logits: [3.0, 2.0, 1.0], minP: 0.1, T: 0.5 };
/** X1.6 predict: the largest logit is 5.0 and min-p is 0.05. */
export const THRESHOLD_Q = { lmax: 5.0, minP: 0.05 };

// ---------------------------------------------------------------- measured (lab X1.2)
export interface LabRow {
  name: string;
  [k: string]: string | number;
}
/**
 * The sort's share of lab X1.2's composed chain, from its benchmark rows: a row whose name
 * mentions "sort" and one that times the whole chain ("chain", "composed" or "total"), with the
 * same rows, cols and k (k = 0 or missing for the whole row). Returns one entry per k found.
 */
export function sortShare(rows: readonly LabRow[]): { rows: number; cols: number; k: number; sort: number; chain: number; share: number }[] {
  const num = (r: LabRow, names: string[]) => {
    for (const n of names) if (typeof r[n] === 'number' && isFinite(r[n] as number)) return r[n] as number;
    return undefined;
  };
  const shape = (r: LabRow) => ({ rows: num(r, ['rows', 'batch']), cols: num(r, ['cols', 'vocab', 'V']), k: num(r, ['k', 'K', 'topk']) ?? 0, s: num(r, ['seconds', 'time_s']) });
  const sorts = rows.filter((r) => /sort/i.test(r.name) && !/cpu/i.test(r.name) && !/chain|composed|total|fused/i.test(r.name));
  const chains = rows.filter((r) => /chain|composed|total/i.test(r.name) && !/cpu|fused/i.test(r.name));
  const out = [];
  for (const c of chains) {
    const C = shape(c);
    if (C.rows == null || C.cols == null || C.s == null) continue;
    const s = sorts.find((r) => {
      const S = shape(r);
      return S.rows === C.rows && S.cols === C.cols && S.k === C.k && S.s != null;
    });
    if (!s) continue;
    const S = shape(s).s as number;
    out.push({ rows: C.rows, cols: C.cols, k: C.k, sort: S, chain: C.s, share: S / C.s });
  }
  return out;
}
