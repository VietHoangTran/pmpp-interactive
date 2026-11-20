// Intensity bookkeeping for chapter 7, sections 7.3 and 7.4 (book pp. 167–168 and 171).
//
// Counts operations and bytes the way the book does (notes/ch07.md, §7.3 and §7.4):
// - An n×n input and an m×m filter, m = 2r+1. Every output costs m² taps of one
//   multiply and one add, so the whole convolution is 2·n²·m² FLOP. Ghost taps are
//   counted, as on p. 167.
// - The least traffic any kernel can have reads the input once, the filter once and
//   writes the output once: (2n² + m²)·b bytes for b bytes per element. With m much
//   smaller than n the book drops the m² and gets 2n²·b, so the ideal is m²/b FLOP/B,
//   which is m²/4 at 4 bytes (p. 167). idealIntensity(m, b, 2, n) keeps the m².
// - The basic kernel (Fig. 7.7) loads one filter element and one input element per
//   tap: 2b bytes for 2 FLOP, so 1/b (0.25 at 4 bytes) whatever m is. The single store
//   per thread sits outside the loop nest and is ignored (p. 167).
// - The constant-memory kernel (Fig. 7.9), assuming every filter read is served by the
//   constant cache: one input load per tap, so 2/b (0.5 at 4 bytes, p. 171).
//
// Ours, not the book's: bytes per element other than 4 (the book fixes single
// precision), and 1D, where the same method gives an ideal of m/b because each input
// takes part in m outputs instead of m².
//
// What it leaves out: caches. The book says the measured intensity of these kernels is
// higher because filter and input elements are found in L1 and L2 (p. 168); this
// model counts what the kernel asks of memory, not what reaches DRAM.

export type Dims = 1 | 2;
export type Regime = 'memory-bound' | 'compute-bound';

/** Peak arithmetic throughput (GFLOP/s) and peak memory bandwidth (GB/s). */
export interface Hardware {
  peak: number;
  bw: number;
}

/** Odd filter widths the explorers offer (radius 0 to 7, the book's "7 or smaller", p. 168). */
export const FILTER_WIDTHS = [1, 3, 5, 7, 9, 11, 13, 15] as const;

export const widthOf = (r: number) => 2 * r + 1;

/** Taps per output: m in 1D, m² in 2D. */
export const taps = (m: number, dims: Dims = 2) => m ** dims;

/** Total FLOP for an n-wide input (n×n in 2D): two per tap per output, ghost taps included. */
export const flops = (n: number, m: number, dims: Dims = 2) => 2 * n ** dims * taps(m, dims);

/** Fig. 7.7: one filter load and one input load per tap. */
export const basicIntensity = (bytes = 4) => 2 / (2 * bytes);

/** Fig. 7.9: one input load per tap; filter reads assumed to hit the constant cache. */
export const constIntensity = (bytes = 4) => 2 / bytes;

/**
 * The best any kernel could do. Without n it is the book's approximation, m²/b in 2D
 * (m/b in 1D); with n the filter's own bytes stay in the denominator.
 */
export function idealIntensity(m: number, bytes = 4, dims: Dims = 2, n?: number): number {
  if (n == null) return taps(m, dims) / bytes;
  const elements = 2 * n ** dims + taps(m, dims);
  return flops(n, m, dims) / (elements * bytes);
}

/** The basic kernel with its one store per output counted: 2 FLOP per tap over (2·taps + 1)·b bytes. */
export const basicIntensityWithStore = (m: number, bytes = 4, dims: Dims = 2) =>
  (2 * taps(m, dims)) / ((2 * taps(m, dims) + 1) * bytes);

/** What fraction of the basic kernel's bytes the ignored store would add. */
export const storeShare = (m: number, dims: Dims = 2) => 1 / (2 * taps(m, dims) + 1);

/** Ratio at which the two limits meet (FLOP/B). */
export const threshold = (hw: Hardware) => hw.peak / hw.bw;

/** The roofline: the most a kernel with this intensity can reach, in GFLOP/s. */
export const ceiling = (ai: number, hw: Hardware) => Math.min(hw.peak, hw.bw * ai);

/** At or above the threshold the arithmetic units, not memory, are the limit. */
export const regime = (ai: number, thr: number): Regime => (ai >= thr ? 'compute-bound' : 'memory-bound');

/**
 * The smallest odd filter width whose ideal intensity reaches the threshold: below it,
 * no kernel, however well written, can be compute-bound.
 */
export function smallestComputeBoundWidth(thr: number, bytes = 4, dims: Dims = 2): number {
  let m = Math.max(1, Math.ceil((thr * bytes) ** (1 / dims) - 1e-9));
  if (m % 2 === 0) m += 1;
  while (idealIntensity(m, bytes, dims) < thr) m += 2; // guards against rounding in the root
  while (m > 1 && idealIntensity(m - 2, bytes, dims) >= thr) m -= 2;
  return m;
}

/** How many times above (or, below 1, short of) its model ceiling a measurement sits. */
export const multipleOfCeiling = (gflops: number, ai: number, hw: Hardware) => gflops / ceiling(ai, hw);
