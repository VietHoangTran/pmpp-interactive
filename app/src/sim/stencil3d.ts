// Chapter 8, sections 8.2 to 8.4: the arithmetic of stencil shapes and of the two
// cube-shaped kernels (Figs. 8.6 and 8.8). It runs the start page's halo reveal, the
// explorers "Stencil or dense filter" and "A cube-shaped tile" on page 8.4, and the
// numbers in their questions.
//
// Sources (the lessons cite them where they are used):
//  - notes/ch08.md: the order of a stencil and its point count (pp. 186 to 187), the
//    13 FLOP of a 7-point output (p. 189), the basic kernel's 0.41 FLOP/B and the ideal
//    13/8 (p. 189), the finite-grid ideal with its n³ − 12n + 16 inputs (p. 189), the
//    p. 190 table of stencil against convolution ratios, the tiled kernel's ratio
//    13(t − 2)³ / 4(t³ + (t − 2)³) and t = 8 → 0.96 (p. 192), the 12% and 58% halo
//    shares (p. 193), exercises 1a to 1c (p. 198);
//  - notes/ch07.md: chapter 7's dense filter ratio m²/4 (p. 176): one multiply and one
//    add per tap;
//  - notes/today/ch08-facts.md E1: the table of cube sides 4 to 32, recomputed.
//
// Ours, not the book's (the lessons say so where they are shown):
//  - the closed forms: (2p − 1) / (2·bytes) for a stencil's ideal, the generalisation of
//    the tiled formula to any order r and any element size, and boxAI for tiles that are
//    not cubes;
//  - neverRead: the tile elements that no thread reads, which are those with two or more
//    coordinates in the halo layers (Fig. 8.7's missing corners, in three dimensions).
//
// What this leaves out: it counts the ideal traffic of an interior block, with every
// input loaded once per block and every output stored once. No caches, no boundary
// blocks, no timing.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

// ---------------------------------------------------------------- shapes
/** Points of an axis-aligned stencil of order r in d dimensions: the centre and r on each side of it along each axis. */
export const points = (d: number, r: number): number => 2 * d * r + 1;

/** Taps of a dense filter covering the same box: (2r + 1)^d. */
export const denseTaps = (d: number, r: number): number => (2 * r + 1) ** d;

/** The book's operation count for a p-point stencil output: p multiplies and p − 1 adds (13 for 7 points, p. 189). */
export const stencilFlops = (p: number): number => 2 * p - 1;

/** Chapter 7's count for a dense filter output: one multiply and one add per tap. */
export const denseFlops = (taps: number): number => 2 * taps;

/**
 * Best possible FLOP/B of a p-point stencil: every input loaded once and every output
 * stored once, so each output costs one load and one store. Ours, as a formula; the book
 * prints the values (p. 190).
 */
export const stencilIdealAI = (p: number, bytes = 4): number => stencilFlops(p) / (2 * bytes);

/** Best possible FLOP/B of a dense filter of the same extent: (2r + 1)^d / bytes, chapter 7's m²/4 at 4 bytes in 2D. */
export const denseIdealAI = (d: number, r: number, bytes = 4): number => denseFlops(denseTaps(d, r)) / (2 * bytes);

// ---------------------------------------------------------------- the basic kernel (Fig. 8.6)
/** One thread per output: p loads and one store per output, nothing reused on chip. 13/32 for 7 points at 4 bytes. */
export const basicAI = (p: number, bytes = 4): number => stencilFlops(p) / ((p + 1) * bytes);

/** Values a basic-kernel output reads and writes, for the bookkeeping readout. */
export const basicCounts = (p: number, bytes = 4) => ({
  flop: stencilFlops(p),
  loads: p,
  stores: 1,
  bytes: (p + 1) * bytes,
});

/**
 * The ideal for a finite n × n × n grid and the 7-point stencil, as the book writes it
 * (p. 189): (n − 2)³ outputs of 13 FLOP; every input except the 12n − 16 on the cube's
 * edges and corners loaded once; every output stored once.
 */
export const idealAIFinite = (n: number, bytes = 4): number => {
  const out = (n - 2) ** 3;
  const inputs = n ** 3 - 12 * n + 16;
  return (13 * out) / (bytes * (inputs + out));
};

// ---------------------------------------------------------------- tiles (Fig. 8.8)
export interface Box {
  x: number;
  y: number;
  z: number;
}
export const box = (x: number, y = x, z = x): Box => ({ x, y, z });

/** Elements of an input tile, one thread each. */
export const tileElements = (b: Box): number => b.x * b.y * b.z;

/** Outputs the tile yields: the input tile less r layers on every face. */
export const tileOutputs = (b: Box, r: number): number =>
  Math.max(0, b.x - 2 * r) * Math.max(0, b.y - 2 * r) * Math.max(0, b.z - 2 * r);

/**
 * FLOP/B of an interior block whose input tile is the box b: every output costs 2p − 1
 * FLOP; the block loads every tile element once and stores every output once. For a cube
 * and r = 1 this is the book's 13(t − 2)³ / 4(t³ + (t − 2)³) (p. 192).
 */
export const boxAI = (b: Box, r: number, bytes = 4): number => {
  const out = tileOutputs(b, r);
  if (out === 0) return 0;
  return (stencilFlops(points(3, r)) * out) / (bytes * (tileElements(b) + out));
};

/** The tiled kernel's FLOP/B for a cube of side t: (12r + 1)(t − 2r)³ / bytes·(t³ + (t − 2r)³). */
export const cubeAI = (t: number, r = 1, bytes = 4): number => boxAI(box(t), r, bytes);

/** Halo elements of a cube tile: loaded, but not outputs. */
export const haloCount = (t: number, r = 1): number => t ** 3 - Math.max(0, t - 2 * r) ** 3;

/** Share of a cube tile that is halo: 1 − (t − 2r)³ / t³. 0.578 for t = 8 (the book's 58%, p. 193). */
export const haloFraction = (t: number, r = 1): number => haloCount(t, r) / t ** 3;

/** Share of a square 2D tile that is halo, for the chapter 7 comparison (124 of 1024 for 32 × 32, p. 193). */
export const haloFraction2D = (t: number, r = 1): number => 1 - Math.max(0, t - 2 * r) ** 2 / t ** 2;

/** Elements of a cube tile that no thread reads: those with two or more coordinates in the halo layers. Ours. */
export const neverRead = (t: number, r = 1): number => {
  const a = Math.max(0, t - 2 * r);
  return t ** 3 - a ** 3 - 3 * a * a * 2 * r;
};

export type Role = 'output' | 'halo' | 'never';

/**
 * What one element of a cube tile is for, at tile coordinates (x, y, z) from 0 to t − 1:
 * an output (inside on every axis), a halo element some output reads (outside on exactly
 * one axis), or an element no output reads (outside on two or three axes).
 */
export function role(t: number, r: number, x: number, y: number, z: number): Role {
  const out = (v: number) => v < r || v >= t - r;
  const n = (out(x) ? 1 : 0) + (out(y) ? 1 : 0) + (out(z) ? 1 : 0);
  return n === 0 ? 'output' : n === 1 ? 'halo' : 'never';
}

export const threadsCube = (t: number): number => t ** 3;

/** The largest cube side whose t³ threads fit under a thread limit (10 for 1024). Exact integer arithmetic. */
export function maxCubeSide(limit: number): number {
  let t = 0;
  while ((t + 1) ** 3 <= limit) t++;
  return t;
}

/** Shared memory of a cube tile: t³ elements. */
export const sharedBytesCube = (t: number, bytes = 4): number => t ** 3 * bytes;

/** The smallest cube side whose ratio reaches a share of the ideal (13/8 at r = 1), or null if none up to maxT does. */
export function sideForShare(share: number, r = 1, maxT = 256): number | null {
  const ideal = stencilIdealAI(points(3, r));
  for (let t = 2 * r + 1; t <= maxT; t++) if (cubeAI(t, r) >= share * ideal) return t;
  return null;
}

// ---------------------------------------------------------------- grids and blocks (exercise 1)
/** Interior points of an N × N × N grid: the outputs of one sweep. */
export const interiorPoints = (N: number, r = 1): number => Math.max(0, N - 2 * r) ** 3;

/** Blocks of b³ threads for the basic kernel, whose threads cover every grid point. */
export const blocksBasic = (N: number, b: number): number => Math.ceil(N / b) ** 3;

/** Blocks for the cube-tiled kernel: one per output tile of side t − 2r. */
export const blocksCube = (N: number, t: number, r = 1): number => Math.ceil(N / (t - 2 * r)) ** 3;
