// Speed-up arithmetic for chapter 1 (book §1.3, pp. 9–10): Amdahl's law, and a
// second, cruder model of what caps the speed-up of the accelerated part itself.
//
// The formula is ours. The book prints no formula for Amdahl's law, only three
// worked cases (p. 9): 30% made 100 times faster gives about 1.42; 30% made
// infinitely fast gives no more than 1.43; 99% made 100 times faster gives about
// 50. speedup() reproduces all three (notes/today/ch01-facts.md, A16).
//
// What the model assumes, and what the app says wherever it is shown:
//   - A run has two parts only: one you speed up (a fraction p of the original
//     time) and one you leave alone (1 - p).
//   - The two parts do not overlap in time. Chapter 22 (§22.3, pp. 537–538)
//     relaxes this with host and device work running at once; when that chapter
//     is built it should extend this file, not add a second Amdahl model.
//   - The untouched part stays the same size. In a real port, moving work to a
//     GPU adds copies and kernel launches to it.
//   - The problem size is fixed.
//   - In the "what caps the kernel" model the accelerated part's speed-up is
//     min(threads x rate per thread, memory cap): threads scale perfectly until
//     memory saturates, the rate per thread is a constant, and the memory cap is
//     a single given number. The book states "about 10x" for that cap (p. 10)
//     and does not derive it; chapter 5 computes it for a kernel and a device.
//
// No imports and no syntax that needs compiling away beyond types, so the
// browser checks can load this file directly.

/**
 * Overall speed-up when a fraction `p` of the original run time is made `s`
 * times faster and the rest is left alone. `s` may be Infinity.
 */
export function speedup(p: number, s: number): number {
  return 1 / (1 - p + p / s);
}

/** The most `speedup(p, s)` can reach, however large `s` is: 1 / (1 - p). */
export function ceiling(p: number): number {
  return 1 / (1 - p);
}

export interface Remaining {
  /** The part left alone, as a share of the original time: 1 - p. */
  untouched: number;
  /** The accelerated part after the speed-up, as a share of the original time: p / s. */
  accelerated: number;
  /** Their sum: the new run time as a share of the original. */
  total: number;
}

/** What is left of the original run time, split into its two parts. */
export function remaining(p: number, s: number): Remaining {
  const untouched = 1 - p;
  const accelerated = p / s;
  return { untouched, accelerated, total: untouched + accelerated };
}

/**
 * The share of the remaining time that is the untouched part. It equals
 * speedup(p, s) / ceiling(p): how much of the ceiling has been reached.
 */
export function untouchedShare(p: number, s: number): number {
  const r = remaining(p, s);
  return r.total === 0 ? 0 : r.untouched / r.total;
}

// Two values that differ by less than this (relatively) are treated as equal,
// so that "exactly at the ceiling" survives binary floating point.
const EPS = 1e-9;
const same = (a: number, b: number) => Math.abs(a - b) <= EPS * Math.max(Math.abs(a), Math.abs(b), 1);

/**
 * The parallel fraction at which a part made `s` times faster gives `target`
 * overall. null when no fraction does (the target is below 1 or above `s`).
 */
export function fractionNeeded(target: number, s: number): number | null {
  if (!(target >= 1) || !(s >= 1)) return null;
  if (same(target, 1)) return 0;
  if (s === Infinity) return 1 - 1 / target;
  if (same(target, s)) return 1;
  if (target > s) return null;
  return (1 - 1 / target) / (1 - 1 / s);
}

/**
 * How many times faster a part that takes a fraction `p` of the time must
 * become to give `target` overall. Infinity when the target is the ceiling
 * itself; null when it is above the ceiling.
 */
export function kernelNeeded(target: number, p: number): number | null {
  if (!(target > 0)) return null;
  const top = ceiling(p);
  if (same(target, top)) return Infinity;
  if (target > top) return null;
  return p / (1 / target - (1 - p));
}

// ------------------------------------------------- what caps the kernel itself

export type Limiter = 'threads' | 'memory';

export interface Capped {
  /** Speed-up of the accelerated part over the one-core baseline. */
  s: number;
  /** Which of the two ceilings set it. */
  limitedBy: Limiter;
}

/**
 * Speed-up of a kernel over a one-core CPU baseline when `threads` threads each
 * run at `ratePerThread` times the speed of that core, and memory saturates at
 * a speed-up of `memoryCap`. Below 1 when there are too few threads to match
 * the core.
 */
export function cappedKernel(threads: number, ratePerThread: number, memoryCap: number): Capped {
  const byThreads = threads * ratePerThread;
  return byThreads < memoryCap ? { s: byThreads, limitedBy: 'threads' } : { s: memoryCap, limitedBy: 'memory' };
}

export interface Limits extends Capped {
  /** Overall speed-up from Amdahl's law with that `s`. */
  overall: number;
  /** Share of the remaining time spent in the untouched part. */
  untouchedShare: number;
}

/** cappedKernel() followed by Amdahl's law for a parallel fraction `p`. */
export function limits(p: number, threads: number, ratePerThread: number, memoryCap: number): Limits {
  const k = cappedKernel(threads, ratePerThread, memoryCap);
  return { ...k, overall: speedup(p, k.s), untouchedShare: untouchedShare(p, k.s) };
}

/**
 * The two thread counts at which the picture changes: below `breakEven` the
 * kernel is slower than the CPU core, and from `saturation` on, memory sets
 * its speed and more threads change nothing.
 */
export function corners(ratePerThread: number, memoryCap: number): { breakEven: number; saturation: number } {
  return { breakEven: 1 / ratePerThread, saturation: memoryCap / ratePerThread };
}

/**
 * What each control would buy on its own, everything else held: `factor`
 * times the threads, `factor` times the memory cap, or an untouched part cut
 * to 1/`factor` of its size. Each entry is the resulting overall speed-up.
 */
export function levers(
  p: number,
  threads: number,
  ratePerThread: number,
  memoryCap: number,
  factor = 10,
): { now: number; threads: number; memory: number; untouched: number } {
  return {
    now: limits(p, threads, ratePerThread, memoryCap).overall,
    threads: limits(p, threads * factor, ratePerThread, memoryCap).overall,
    memory: limits(p, threads, ratePerThread, memoryCap * factor).overall,
    untouched: limits(1 - (1 - p) / factor, threads, ratePerThread, memoryCap).overall,
  };
}

// ------------------------------------------------------------- slider ladders
// Log-spaced stops with round values, so that a slider can land exactly on the
// cases the lesson talks about (30%, 99%, 99.9%; 10, 100, 1,000).

const tidy = (v: number) => Number(v.toPrecision(6));

function ladder(lo: number, hi: number, mantissas: number[]): number[] {
  const out: number[] = [];
  for (let e = Math.floor(Math.log10(lo)) - 1; e <= Math.ceil(Math.log10(hi)); e++)
    for (const m of mantissas) {
      const v = tidy(m * 10 ** e);
      if (v >= lo * (1 - EPS) && v <= hi * (1 + EPS)) out.push(v);
    }
  return [...new Set(out)].sort((a, b) => a - b);
}

/**
 * Parallel fractions from 30% to 99.99%, spaced evenly in the logarithm of the
 * sequential share 1 - p, so the stops get finer as p approaches 1.
 */
export const FRACTIONS: number[] = ladder(0.0001, 0.7, [1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 9])
  .map((q) => tidy(1 - q))
  .sort((a, b) => a - b);

/** Speed-ups of the accelerated part, and memory caps: 1 to 1,000. */
export const SPEEDUPS: number[] = ladder(1, 1000, [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8]);

/** Thread counts: 1 to 100,000, whole numbers only. */
export const THREADS: number[] = ladder(1, 100000, [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8]).filter((v) => Number.isInteger(v));

/** Speed of one GPU thread relative to the CPU core of the baseline. */
export const RATES: { value: number; label: string }[] = [
  { value: 1 / 8, label: '1/8' },
  { value: 1 / 4, label: '1/4' },
  { value: 1 / 2, label: '1/2' },
  { value: 1, label: '1' },
];

/** Index of the stop closest to `value`, compared on a log scale. */
export function nearestIndex(stops: number[], value: number, of: (v: number) => number = (v) => v): number {
  let best = 0;
  let bestD = Infinity;
  stops.forEach((v, i) => {
    const d = Math.abs(Math.log(of(v)) - Math.log(of(value)));
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  });
  return best;
}

/** Index into FRACTIONS of the stop closest to `p` (compared on the sequential share). */
export const fractionIndex = (p: number) => nearestIndex(FRACTIONS, p, (v) => 1 - v);

// ---------------------------------------------------------------- the book's cases

export interface BookCase {
  label: string;
  p: number;
  s: number;
  /** How the book states the result (p. 9). */
  book: string;
}

/** The three cases the book works on p. 9. */
export const BOOK_CASES: BookCase[] = [
  { label: '30%, 100×', p: 0.3, s: 100, book: 'about 1.42' },
  { label: '30%, unbounded', p: 0.3, s: Infinity, book: 'no more than 1.43' },
  { label: '99%, 100×', p: 0.99, s: 100, book: 'about 50' },
];

// ---------------------------------------------------------------- curves for the plots

/** `n` values from `lo` to `hi` inclusive, evenly spaced on a log scale. */
export function logSpace(lo: number, hi: number, n: number): number[] {
  const a = Math.log(lo);
  const b = Math.log(hi);
  return Array.from({ length: n }, (_, i) => (i === n - 1 ? hi : Math.exp(a + ((b - a) * i) / (n - 1))));
}

/** Overall speed-up against the speed-up of the accelerated part. */
export function speedupCurve(p: number, lo = 1, hi = 1000, n = 61): { s: number; overall: number }[] {
  return logSpace(lo, hi, n).map((s) => ({ s, overall: speedup(p, s) }));
}

/**
 * Kernel and overall speed-up against the number of threads. The two corner
 * thread counts are included so that a polyline through the points has its
 * corners in the right place.
 */
export function limitsCurve(
  p: number,
  ratePerThread: number,
  memoryCap: number,
  lo = 1,
  hi = 100000,
  n = 81,
): { threads: number; s: number; overall: number }[] {
  const c = corners(ratePerThread, memoryCap);
  const xs = [...logSpace(lo, hi, n), c.breakEven, c.saturation].filter((x) => x >= lo && x <= hi).sort((a, b) => a - b);
  return xs.map((threads) => {
    const k = limits(p, threads, ratePerThread, memoryCap);
    return { threads, s: k.s, overall: k.overall };
  });
}
