// Reformulate or accelerate? Cost curves for a direct method against a reformulated one,
// each on a slow processor and on one that is S times faster (chapter 22, section 22.4).
// It runs the explorer "Reformulate or accelerate?" and the prediction p-crossover.
//
// From the book:
//  - Direct Coulomb summation: every grid point sums every atom, and both grow with the
//    volume, so the work grows with the square of the volume (p. 531; ch. 21 p. 528).
//  - Cutoff summation with binning: each grid point sums only the atoms in its neighbourhood,
//    a number fixed by the cutoff and the density, so the work grows linearly with the
//    volume (ch. 21 p. 523); the far atoms are treated collectively (p. 531).
//  - Without a KV cache, every decode iteration recomputes five matrix products per layer
//    over the whole sequence: X·W_Q, X·W_K, X·W_V, Q·K^T and P·V (ch. 20 p. 488). With it,
//    one new row of Q, K and V (three vector-matrix products), S' = Q'·K^T and O' = P'·V
//    (ch. 20 pp. 490-491, Fig. 20.6).
//  - KV-cache bytes for b sequences of length N: b · N · 2 · l · h_q · d · p (Eq. 20.8,
//    ch. 20 pp. 505-506), with the book's Llama 2 7B example: 32 layers, hidden dimension
//    h_q · d = 4096, 2 bytes, 2 GiB for one sequence of 4096 tokens (p. 506).
//
// Ours, not the book's:
//  - The constants of the electrostatic preset: at size 1, 100,000 atoms and 1,000,000 grid
//    points, and a cutoff neighbourhood of 1,000 atoms. Size is volume relative to that
//    system. A grid point never scans more atoms than exist, so below the volume that holds
//    one neighbourhood (size 0.01) the two methods cost the same.
//  - Operation counts: one unit per atom-grid point pair; for decoding, one unit per
//    multiply-add of one layer, the softmax ignored, the baseline's full T x T score matrix
//    computed (as the book's baseline does before masking). Recomputing at token T costs
//    3·T·d^2 + 2·T^2·d and the cached step 3·d^2 + 2·T·d, exactly T times less.
//  - Time = operations / speed, with speed 1 for the slow processor and S for the fast one,
//    the same S for both methods.
//
// What it leaves out: memory traffic (a cached decode step is bandwidth-bound, chapter 20
// p. 492), the cutoff's loss of accuracy and the cost of the implicit method for far atoms,
// the binning pass, and everything outside the one layer or the one sum.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

export type Preset = 'coulomb' | 'kv';
export const PRESETS: Preset[] = ['coulomb', 'kv'];

// ---------------------------------------------------------------- electrostatic map
export const COULOMB = { atoms: 1e5, points: 1e6, neighbourhood: 1e3 };

/** Direct Coulomb summation at volume v (relative): atoms x grid points. */
export const directCoulomb = (v: number) => COULOMB.atoms * v * COULOMB.points * v;
/** Cutoff summation at volume v: grid points x the atoms each scans, never more than exist. */
export const cutoffCoulomb = (v: number) => COULOMB.points * v * Math.min(COULOMB.neighbourhood, COULOMB.atoms * v);

// ---------------------------------------------------------------- decoding
/** The book's Llama 2 7B example (ch. 20 p. 506). */
export const LLAMA2_7B = { layers: 32, hidden: 4096, bytes: 2 };

/** Multiply-adds of one layer to produce token T by recomputing the whole prefix. */
export const recomputeOps = (t: number, d = LLAMA2_7B.hidden) => 3 * t * d * d + 2 * t * t * d;
/** Multiply-adds of one layer to produce token T with a KV cache. */
export const cachedOps = (t: number, d = LLAMA2_7B.hidden) => 3 * d * d + 2 * t * d;

/** Eq. 20.8: KV-cache bytes for `batch` sequences of `length` tokens. */
export const kvBytes = (length: number, batch: number, m = LLAMA2_7B) => batch * length * 2 * m.layers * m.hidden * m.bytes;
export const GiB = 2 ** 30;

// ---------------------------------------------------------------- the two curves of a preset
export interface Curves {
  preset: Preset;
  direct: (x: number) => number;
  reformed: (x: number) => number;
  /** The explorer's size range (log axis). */
  min: number;
  max: number;
  /** The default size. */
  at: number;
}

export function curves(preset: Preset): Curves {
  return preset === 'coulomb'
    ? { preset, direct: directCoulomb, reformed: cutoffCoulomb, min: 0.01, max: 1000, at: 1 }
    : { preset, direct: (t) => recomputeOps(t), reformed: (t) => cachedOps(t), min: 1, max: 65536, at: 1024 };
}

export type Series = 'direct-slow' | 'direct-fast' | 'reformed-slow' | 'reformed-fast';
export const SERIES: Series[] = ['direct-slow', 'direct-fast', 'reformed-slow', 'reformed-fast'];

/** Time of one series at size x with speed-up S: operations / speed. */
export function time(c: Curves, s: Series, x: number, speedup: number): number {
  const ops = s.startsWith('direct') ? c.direct(x) : c.reformed(x);
  return s.endsWith('fast') ? ops / speedup : ops;
}

/** d log f / d log x just above x (a forward difference, so a kink at x reads as the slope beyond it). */
export function slope(f: (x: number) => number, x: number): number {
  const h = 1e-5;
  return (Math.log(f(x * Math.exp(h))) - Math.log(f(x))) / h;
}

/**
 * The size at which the direct method on the fast processor and the reformulated method on
 * the slow one take the same time, searched on [min, max]; null if they do not cross there.
 * Below it the accelerated direct method is faster; above it the reformulation wins even on
 * the slower processor.
 */
export function crossover(c: Curves, speedup: number, lo = c.min, hi = c.max): number | null {
  const g = (x: number) => Math.log(time(c, 'direct-fast', x, speedup)) - Math.log(time(c, 'reformed-slow', x, speedup));
  // Search from the top: the last sign change below hi.
  const n = 400;
  let prevX = hi;
  let prev = g(hi);
  if (prev < 0) return null;
  for (let i = n - 1; i >= 0; i--) {
    const x = Math.exp(Math.log(lo) + ((Math.log(hi) - Math.log(lo)) * i) / n);
    const v = g(x);
    if (v < 0) {
      let a = x;
      let b = prevX;
      for (let k = 0; k < 80; k++) {
        const m = Math.sqrt(a * b);
        if (g(m) < 0) a = m;
        else b = m;
      }
      return Math.sqrt(a * b);
    }
    if (v === 0) return x;
    prevX = x;
    prev = v;
  }
  return null;
}

/** How far size must grow for a method with cost ~ x^p, `ratio` times faster at x = 1, to tie one with cost ~ x^q. */
export const tieFactor = (ratio: number, p = 2, q = 1) => ratio ** (1 / (p - q));

/** At size x, how many times faster the accelerated direct method is than the slow reformulated one (below 1: slower). */
export const advantage = (c: Curves, x: number, speedup: number) => time(c, 'reformed-slow', x, speedup) / time(c, 'direct-fast', x, speedup);
