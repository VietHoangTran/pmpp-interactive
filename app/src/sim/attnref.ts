// Reference causal attention for small problems (chapter 20, sections 20.2 and 20.3).
//
// What it computes, for one head (book Eq. 20.1 and Eq. 20.2, pp. 483-484; notes/ch20.md):
//   S = Q K^T                     raw inner products, nq x nk
//   logits = scale * S + offset   scale = 1/sqrt(d) when on; offset is a teaching knob
//   mask                          column j of row i is visible when j <= i + (nk - nq):
//                                 lower-right alignment, the convention of FlashAttention
//                                 since 2.1 and of MLX (notes/today/ch20-facts.md A8, D4).
//                                 With nq = nk it is the book's mask: column <= row.
//   P = softmax(logits) per row   stable (subtract the row max m, Eq. 20.2) or naive
//   O = P V
//   L = m + log(sum)              the row's log-sum-exp, what FlashAttention-2 stores (A7)
//
// Two precisions. 'f64' is JavaScript's double. 'f32' rounds every intermediate with
// Math.fround, so exp of a logit above ln(FLT_MAX) (about 88.72) becomes Infinity exactly
// as expf does on a GPU (ch10 A11 via ch20 facts), and Infinity / Infinity is NaN.
//
// What it leaves out:
//   - Fused multiply-add and the GPU's own exp rounding (2 ULP for expf); here each
//     operation is rounded once to float32. Results agree with a GPU to rounding, not bit for bit.
//   - Several heads, batches and half precision.
//
// Erasable TypeScript only, no runtime imports: the browser checks import this file in Node.

export type Precision = 'f64' | 'f32';

/** Largest finite float32. */
export const FLT_MAX = 3.4028234663852886e38;
/** ln(FLT_MAX): the largest x for which exp(x) is finite in float32. */
export const EXP_LIMIT_F32 = Math.log(FLT_MAX);

export interface AttnProblem {
  /** Query rows (the newest nq tokens). */
  nq: number;
  /** Key and value rows (every token so far). */
  nk: number;
  /** Head width. */
  d: number;
  Q: number[][];
  K: number[][];
  V: number[][];
}

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Values small enough to multiply in your head: Q and K from these... */
export const QK_VALUES = [-1, -0.5, 0, 0.5, 1] as const;
/** ...and V from these, so a row of O is a weighted mean you can estimate. */
export const V_VALUES = [-3, -2, -1, 0, 1, 2, 3] as const;

/** A deterministic problem: nk tokens of width d, the last nq of which ask the questions (nq <= nk). */
export function makeAttnProblem(nk: number, d: number, seed = 1, nq = nk): AttnProblem {
  if (nq > nk) throw new Error('nq must not exceed nk');
  const rand = mulberry32(seed * 7919 + nk * 31 + d);
  const pick = (vals: readonly number[]) => vals[Math.floor(rand() * vals.length)];
  const rows = (n: number, vals: readonly number[]) => Array.from({ length: n }, () => Array.from({ length: d }, () => pick(vals)));
  const K = rows(nk, QK_VALUES);
  const V = rows(nk, V_VALUES);
  const Q = rows(nq, QK_VALUES);
  return { nq, nk, d, Q, K, V };
}

/** Is column j visible to query row i? Lower-right alignment; the book's causal mask when nq = nk. */
export const visible = (i: number, j: number, nq: number, nk: number) => j <= i + (nk - nq);
/** How many columns row i may see. */
export const visibleCount = (i: number, nq: number, nk: number) => Math.max(0, Math.min(nk, i + (nk - nq) + 1));

const rounder = (prec: Precision): ((x: number) => number) => (prec === 'f32' ? Math.fround : (x: number) => x);

/** The terms of one inner product q_i . k_j, for writing it out, and its value at this precision. */
export function dotTerms(p: AttnProblem, i: number, j: number, prec: Precision = 'f64') {
  const r = rounder(prec);
  const terms = p.Q[i].map((q, k) => ({ q, k: p.K[j][k], prod: r(q * p.K[j][k]) }));
  let acc = 0;
  for (const t of terms) acc = r(acc + t.prod);
  return { terms, value: acc };
}

export interface SoftmaxRow {
  /** Probabilities; masked entries are exactly 0. */
  p: number[];
  /** Numerators: exp(l - m) (stable) or exp(l) (naive); 0 where masked. */
  e: number[];
  /** Row maximum over the visible logits (computed even when not subtracted). */
  m: number;
  /** Denominator. */
  sum: number;
  /** log-sum-exp of the visible logits: m + log(sum) when stable, log(sum) when naive. */
  L: number;
}

/**
 * Softmax of one row. A logit of -Infinity is masked: it contributes exactly 0 and is never
 * passed to exp. With subtractMax false this is the textbook formula, which overflows in
 * float32 once a logit exceeds EXP_LIMIT_F32.
 */
export function softmaxRow(logits: number[], subtractMax = true, prec: Precision = 'f64'): SoftmaxRow {
  const r = rounder(prec);
  let m = -Infinity;
  for (const l of logits) if (l !== -Infinity && l > m) m = l;
  const e = logits.map((l) => (l === -Infinity ? 0 : r(Math.exp(subtractMax ? r(l - m) : l))));
  let sum = 0;
  for (const x of e) sum = r(sum + x);
  const p = e.map((x, j) => (logits[j] === -Infinity ? 0 : r(x / sum)));
  const L = subtractMax ? r(m + r(Math.log(sum))) : r(Math.log(sum));
  return { p, e, m, sum, L };
}

export interface AttnOptions {
  /** Multiply S by 1/sqrt(d). Default true. */
  scale?: boolean;
  /** Apply the causal mask. Default true. */
  mask?: boolean;
  /** Subtract each row's max before exp (Eq. 20.2). Default true. */
  subtractMax?: boolean;
  /** Added to every logit before the softmax. Default 0. */
  offset?: number;
  precision?: Precision;
}

export interface AttnResult {
  nq: number;
  nk: number;
  d: number;
  /** The factor applied to S: 1/sqrt(d), or 1. */
  scale: number;
  /** Raw inner products, nq x nk. */
  S: number[][];
  /** scale * S + offset where visible, -Infinity where masked. */
  logits: number[][];
  /** Per row: max of the visible logits. */
  m: number[];
  /** Per row: the softmax denominator. */
  sum: number[];
  /** Per row: log-sum-exp. */
  L: number[];
  /** Softmax numerators, nq x nk. */
  E: number[][];
  P: number[][];
  /** nq x d. */
  O: number[][];
  /** Per row: how many columns are non-zero in P (the visible ones). */
  live: number[];
}

/** Causal attention of one head, Eq. 20.1 with the stable softmax of Eq. 20.2 by default. */
export function attention(p: AttnProblem, opts: AttnOptions = {}): AttnResult {
  const { scale: useScale = true, mask = true, subtractMax = true, offset = 0, precision = 'f64' } = opts;
  const r = rounder(precision);
  const scale = useScale ? r(1 / Math.sqrt(p.d)) : 1;
  const S: number[][] = [];
  const logits: number[][] = [];
  for (let i = 0; i < p.nq; i++) {
    const srow: number[] = [];
    const lrow: number[] = [];
    for (let j = 0; j < p.nk; j++) {
      const s = dotTerms(p, i, j, precision).value;
      srow.push(s);
      lrow.push(!mask || visible(i, j, p.nq, p.nk) ? r(r(scale * s) + offset) : -Infinity);
    }
    S.push(srow);
    logits.push(lrow);
  }
  const rows = logits.map((l) => softmaxRow(l, subtractMax, precision));
  const P = rows.map((x) => x.p);
  const O = P.map((prow) =>
    Array.from({ length: p.d }, (_, c) => {
      let acc = 0;
      for (let j = 0; j < p.nk; j++) acc = r(acc + r(prow[j] * p.V[j][c]));
      return acc;
    }),
  );
  return {
    nq: p.nq,
    nk: p.nk,
    d: p.d,
    scale,
    S,
    logits,
    m: rows.map((x) => x.m),
    sum: rows.map((x) => x.sum),
    L: rows.map((x) => x.L),
    E: rows.map((x) => x.e),
    P,
    O,
    live: logits.map((l) => l.filter((x) => x !== -Infinity).length),
  };
}

/** Largest absolute difference between two matrices of the same shape (NaN anywhere gives NaN). */
export function maxAbsDiff(a: number[][], b: number[][]): number {
  let worst = 0;
  for (let i = 0; i < a.length; i++)
    for (let j = 0; j < a[i].length; j++) {
      const diff = Math.abs(a[i][j] - b[i][j]);
      if (Number.isNaN(diff)) return NaN;
      worst = Math.max(worst, diff);
    }
  return worst;
}

/**
 * Entries of `a` that differ from `b` by more than tol, as [row, col] pairs. A NaN on either
 * side counts as differing, unless finiteOnly is set, when such entries are skipped.
 */
export function differing(a: number[][], b: number[][], tol = 1e-9, finiteOnly = false): [number, number][] {
  const out: [number, number][] = [];
  for (let i = 0; i < a.length; i++)
    for (let j = 0; j < a[i].length; j++) {
      if (finiteOnly && !(Number.isFinite(a[i][j]) && Number.isFinite(b[i][j]))) continue;
      if (!(Math.abs(a[i][j] - b[i][j]) <= tol)) out.push([i, j]);
    }
  return out;
}

/**
 * The input of the book's softmax kernel (Fig. 20.4) for an n x n causal problem, and the
 * answer it must produce. S holds scale * QK^T + offset for EVERY entry, masked or not,
 * because that is what the GEMM before the kernel writes: the kernel's loop bound is what
 * keeps it from reading the future. Float32 throughout.
 */
export function causalSoftmaxProblem(n: number, offset = 0, seed = 7, d = 4) {
  const prob = makeAttnProblem(n, d, seed);
  const full = attention(prob, { mask: false, offset, precision: 'f32' });
  const ref = attention(prob, { mask: true, offset, precision: 'f32' });
  return { n, d, offset, S: full.logits, refP: ref.P, refL: ref.L };
}
export type CausalSoftmaxProblem = ReturnType<typeof causalSoftmaxProblem>;
