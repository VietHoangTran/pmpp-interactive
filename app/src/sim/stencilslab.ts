// Arithmetic of the z-coarsened stencil kernels of chapter 8 (book Figs. 8.10 and
// 8.12), generalised to stencil order r and value size, for section 8.5's explorer,
// the "Marching through z" scene and the chapter's questions.
//
// A block is one t x t plane of threads. It marches through a slab of D output
// planes in z. For an axis-aligned 3D stencil of order r (6r + 1 points) it loads
// D + 2r input planes of t x t values and stores D output planes of (t - 2r)^2.
//
// What this leaves out, and what the lesson says wherever it is shown:
//   - Interior blocks only: every input element of the block's tile is inside the
//     grid. Blocks at the grid's edge load less and compute less.
//   - Every input element is loaded once per block and every output stored once, as
//     the book's formula on p. 192 counts them. No caches, no halo shared between
//     neighbouring blocks.
//   - Register counts are the stencil window only (the values a thread keeps for z),
//     not what a compiler allocates: indices, addresses and temporaries come on top.
//
// The book's own numbers (notes/ch08.md): 13 FLOP per output for order 1; 0.96 FLOP/B
// for the 8-wide cube (p. 192); 1.52 printed for t = 32 (p. 195), which is the limit
// for an unboundedly deep slab and not the kernel as coded (1.47; facts E2);
// 12 KB of shared memory for three 32 x 32 planes (p. 195), one third of that with
// register tiling, and 3072 more registers per 32 x 32 block (p. 197).

/** Points of an axis-aligned 3D stencil of order r: the centre and r on each side along each axis. */
export const points = (r: number) => 6 * r + 1;
/** FLOP per output: one multiply per point, one add fewer. 13 for the 7-point stencil (p. 189). */
export const flopPerOutput = (r: number) => 12 * r + 1;

/** The block-size limit the book designs around (p. 192), documented by NVIDIA and AMD today (ch05 A5, ch08 B1). */
export const BOOK_BLOCK_LIMIT = 1024;

/** Output width of a tile of input width t for order r. */
export const outWidth = (t: number, r = 1) => t - 2 * r;

/** Input planes a block loads over its life: its D output planes plus r on each side in z. */
export const planesLoaded = (D: number, r = 1) => D + 2 * r;

/** Elements one interior block loads and stores over its whole march. */
export function lifetimeCounts(t: number, D: number, r = 1) {
  const o = outWidth(t, r);
  return { inputs: t * t * planesLoaded(D, r), outputs: o * o * D };
}

/** FLOP per byte of global traffic for one interior block of a slab D planes deep (p. 192's accounting). */
export function slabAI(t: number, D: number, r = 1, bytes = 4): number {
  const { inputs, outputs } = lifetimeCounts(t, D, r);
  return (flopPerOutput(r) * outputs) / (bytes * (inputs + outputs));
}

/** The same as D grows without bound: only the x-y halo is left. */
export function slabAILimit(t: number, r = 1, bytes = 4): number {
  const o = outWidth(t, r);
  return (flopPerOutput(r) * o * o) / (bytes * (t * t + o * o));
}

/** A cube of side t (Fig. 8.8's tiling) is the slab whose depth equals its output width. */
export const cubeAI = (t: number, r = 1, bytes = 4) => slabAI(t, outWidth(t, r), r, bytes);

/** Threads per block: one per element of an x-y plane of the input tile. */
export const threadsSlab = (t: number) => t * t;
/** Threads per block if one thread handled each element of a cubic input tile (Fig. 8.8). */
export const threadsCube = (t: number) => t * t * t;

export type SlabKernel = 'planes' | 'registers';

/**
 * Shared memory per block. Fig. 8.10 keeps the whole window of 2r + 1 planes in shared
 * memory; Fig. 8.12 keeps only the current plane there, because only its x-y neighbours
 * are read by other threads.
 */
export function sharedBytes(kind: SlabKernel, t: number, r = 1, bytes = 4): number {
  return (kind === 'planes' ? 2 * r + 1 : 1) * t * t * bytes;
}

/** Values one thread keeps in registers for the z window under register tiling. */
export const windowRegs = (r = 1) => 2 * r + 1;
/** The window registers of a whole block: 3 x 1024 = 3072 for r = 1, t = 32 (p. 197). */
export const extraRegsPerBlock = (t: number, r = 1) => windowRegs(r) * t * t;

/**
 * Blocks launched for an N x N x N grid with the book's indexing: blocks step by the
 * output width in x and y (blockIdx * OUT_TILE_DIM) and by the slab depth in z.
 */
export function blocksSlab(N: number, t: number, D: number, r = 1): number {
  const xy = Math.ceil(N / outWidth(t, r));
  return xy * xy * Math.ceil(N / D);
}
