// RoPE in its two pairings, and llama.cpp's weight permutation between them (section X3.7).
//
//   NORM  consecutive pairs (2j, 2j + 1): llama.cpp rope_norm, "normal" / "traditional" /
//         "interleaved", llama2.c (notes/today/x3-facts.md B6)
//   NEOX  half split (j, j + head/2): llama.cpp rope_neox, Hugging Face rotate_half, mlx-lm
//         rope_traditional = False, SmolLM2 "rope_interleaved": false (B6, D4)
//
// Both rotate pair j by pos × theta_j with theta_j = base^(-2j / head) (llama2.c's
// 1 / powf(10000, head_dim / head_size) with head_dim = 2j; Hugging Face's inv_freq, B6).
// The course computes cos and sin once on the host in double and rounds them to float (ours):
// ropeTables does exactly that, with Math.fround standing in for the cast to float.
//
// permute() is conversion/llama.py's permute(weights, n_head, n_head_kv) on a row-major
// (rows, cols) matrix: reshape(n_head, 2, rows / n_head / 2, cols).swapaxes(1, 2).reshape.

export type Pairing = 'norm' | 'neox';
export const ROPE_BASE = 10000;

export interface Tables {
  seqLen: number;
  head: number;
  /** cos[pos * head/2 + j], rounded to float */
  cos: Float32Array;
  sin: Float32Array;
}

/** theta_j in double. */
export const theta = (j: number, head: number, base = ROPE_BASE) => Math.pow(base, (-2 * j) / head);

/** The host tables: seqLen rows of head/2 cos and sin values, computed in double, stored as float. */
export function ropeTables(seqLen: number, head: number, base = ROPE_BASE): Tables {
  if (head % 2) throw new Error('head size must be even');
  const half = head / 2;
  const cos = new Float32Array(seqLen * half);
  const sin = new Float32Array(seqLen * half);
  for (let p = 0; p < seqLen; p++)
    for (let j = 0; j < half; j++) {
      const a = p * theta(j, head, base);
      cos[p * half + j] = Math.cos(a);
      sin[p * half + j] = Math.sin(a);
    }
  return { seqLen, head, cos, sin };
}

/** Which two elements of one head pair j rotates. */
export function pairOf(j: number, head: number, mode: Pairing): [number, number] {
  return mode === 'norm' ? [2 * j, 2 * j + 1] : [j, j + head / 2];
}

/** Rotate every head of x (length nHeads × head) at position pos. Arithmetic in float, as a kernel does it. */
export function rope(x: ArrayLike<number>, pos: number, mode: Pairing, t: Tables): Float32Array {
  const head = t.head;
  const half = head / 2;
  if (x.length % head) throw new Error('length is not a whole number of heads');
  if (pos < 0 || pos >= t.seqLen) throw new Error('position outside the table');
  const f = Math.fround;
  const out = Float32Array.from(x);
  for (let h = 0; h < x.length / head; h++)
    for (let j = 0; j < half; j++) {
      const [a, b] = pairOf(j, head, mode).map((e) => h * head + e);
      const c = t.cos[pos * half + j];
      const s = t.sin[pos * half + j];
      const x0 = f(x[a]);
      const x1 = f(x[b]);
      out[a] = f(f(x0 * c) - f(x1 * s));
      out[b] = f(f(x0 * s) + f(x1 * c));
    }
  return out;
}

/** The row each new row comes from: newRow r takes oldRow src[r]. */
export function permuteSource(rows: number, nHead: number, nHeadKv?: number): number[] {
  const n = nHeadKv !== undefined && nHead !== nHeadKv ? nHeadKv : nHead;
  if (rows % (2 * n)) throw new Error(`cannot reshape ${rows} rows into (${n}, 2, ${rows / n / 2})`);
  const half = rows / n / 2;
  // new index within a head: b * 2 + a; old index: a * half + b (swapaxes(1, 2))
  const src: number[] = [];
  for (let h = 0; h < n; h++)
    for (let b = 0; b < half; b++)
      for (let a = 0; a < 2; a++) src.push(h * 2 * half + a * half + b);
  return src;
}

/** llama.cpp's permute on a row-major (rows × cols) matrix. */
export function permute(W: ArrayLike<number>, rows: number, cols: number, nHead: number, nHeadKv?: number): Float32Array {
  const src = permuteSource(rows, nHead, nHeadKv);
  const out = new Float32Array(rows * cols);
  src.forEach((s, r) => out.set(Array.prototype.slice.call(W, s * cols, (s + 1) * cols), r * cols));
  return out;
}
/** The inverse, applied to a vector (one value per row). */
export function unpermuteVec(y: ArrayLike<number>, nHead: number, nHeadKv?: number): Float32Array {
  const src = permuteSource(y.length, nHead, nHeadKv);
  const out = new Float32Array(y.length);
  src.forEach((s, r) => { out[s] = y[r]; });
  return out;
}
export const permuteVec = (y: ArrayLike<number>, nHead: number, nHeadKv?: number) => permute(y, y.length, 1, nHead, nHeadKv);

/** y = W x for row-major W (rows × cols), in float. */
export function matvec(W: ArrayLike<number>, x: ArrayLike<number>, rows: number, cols: number): Float32Array {
  const y = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let s = 0;
    for (let c = 0; c < cols; c++) s = Math.fround(s + Math.fround(W[r * cols + c] * x[c]));
    y[r] = s;
  }
  return y;
}

export const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

/** A deterministic generator for the explorer's weights and inputs. */
export function seeded(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), s | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (((t ^ (t >>> 14)) >>> 0) / 4294967296) * 2 - 1;
  };
}

export interface Case {
  head: number;
  /** model input width */
  din: number;
  Wq: Float32Array;
  Wk: Float32Array;
  xq: Float32Array;
  xk: Float32Array;
}
/** One head of trained-looking weights: the explorer's fixed example for a head size. */
export function makeCase(head: number, din = 6, seed = 7): Case {
  const r = seeded(seed + head);
  const m = (n: number) => Float32Array.from({ length: n }, () => Math.fround(r()));
  return { head, din, Wq: m(head * din), Wk: m(head * din), xq: m(din), xk: m(din) };
}

export interface Run {
  /** q and k as the kernel receives them (permuted if the switch is on) */
  q: Float32Array;
  k: Float32Array;
  /** q rotated at pos */
  qRot: Float32Array;
  /** score of q at pos against k at pos - d, for d = 0..maxD */
  scores: number[];
}

/**
 * The explorer's three cases on one head. "Trained" means half-split weights (a Hugging Face
 * checkpoint): reference = NEOX on (W x). The kernel runs `mode` on W x, or on (P W) x if permuted.
 */
export function runCase(c: Case, pos: number, mode: Pairing, permuted: boolean, t: Tables, maxD: number): Run {
  const Wq = permuted ? permute(c.Wq, c.head, c.din, 1) : c.Wq;
  const Wk = permuted ? permute(c.Wk, c.head, c.din, 1) : c.Wk;
  const q = matvec(Wq, c.xq, c.head, c.din);
  const k = matvec(Wk, c.xk, c.head, c.din);
  const qRot = rope(q, pos, mode, t);
  const scores: number[] = [];
  for (let d = 0; d <= maxD; d++) scores.push(pos - d >= 0 ? dot(qRot, rope(k, pos - d, mode, t)) : NaN);
  return { q, k, qRot, scores };
}

/** The two curves the explorer compares and whether the kernel's equals the reference. */
export function compare(c: Case, pos: number, mode: Pairing, permuted: boolean, t: Tables, maxD: number) {
  const ref = runCase(c, pos, 'neox', false, t, maxD);
  const ker = runCase(c, pos, mode, permuted, t, maxD);
  const qBack = permuted ? unpermuteVec(ker.qRot, 1) : ker.qRot;
  const sameVector = qBack.every((v, i) => v === ref.qRot[i]);
  const maxScoreDiff = Math.max(...ker.scores.map((s, i) => (Number.isNaN(s) ? 0 : Math.abs(s - ref.scores[i]))));
  return { ref, ker, sameVector, maxScoreDiff, same: sameVector && maxScoreDiff < 1e-9 };
}
