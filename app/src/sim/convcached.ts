// The cached-halo tiled convolution of chapter 7 (book Fig. 7.15, p. 178), tap by tap,
// for section 7.6's explorer, its questions and the "Where the halo comes from" scene.
//
// The kernel, corrected, with T = TILE_DIM and R = FILTER_RADIUS (one thread per output,
// block = tile = T x T threads):
//   col = bx*T + tx;  row = by*T + ty
//   N_s[ty][tx] = (row < height && col < width) ? N[row*width + col] : 0
//   __syncthreads()
//   if (col < width && row < height)
//     for fRow, fCol in 0 .. 2R:
//       if (0 <= tx-R+fCol < T && 0 <= ty-R+fRow < T)  read N_s[ty-R+fRow][tx-R+fCol]  ("shared")
//       else if (row-R+fRow, col-R+fCol) is in the image  read N from global memory   ("global", a halo cell)
//       else                                               skip it                      ("ghost")
//
// As printed (notes/ch07.md, "Unclear in the PDF"; facts ch07 A8), line 21 reads
// N_s[threadIdx.y+fRow][threadIdx.x+fCol], without the -FILTER_RADIUS its own guard uses,
// and the guard is written on unsigned threadIdx arithmetic, so its ">= 0" half is always
// true. This model evaluates both: the slot the printed line reads, and the guard in
// 32-bit unsigned arithmetic, as C++ does.
//
// What it leaves out, and what the lesson says wherever it is shown:
//   - No timing, no caches, no memory system: a "global" tap is a read of N that leaves
//     the SM; whether L1 or L2 serves it is halocache.ts's question, not this file's.
//   - Warps are consecutive thread ids (tx fastest). Whether lanes that disagree at a
//     branch cost one pass or two is the hardware's business; this file only says where
//     they disagree.
//   - Reads past the end of N_s in the printed kernel are undefined behaviour. The model
//     reads N_s as one row-major array (an index past a row lands in the next row) and
//     calls anything beyond the whole array "garbage" (NaN).

export type TapKind = 'shared' | 'global' | 'ghost';

/** One half-pair of the printed guard, `threadIdx - FILTER_RADIUS + f`, evaluated as C++ does. */
export interface Guard {
  /** What the author meant: the value in signed arithmetic. */
  signed: number;
  /** What C++ computes: threadIdx is unsigned, so the expression is unsigned and wraps. */
  u32: number;
  /** The printed `>= 0` half, applied to the unsigned value: always true. */
  geZero: boolean;
  /** The printed `< TILE_DIM` half, applied to the unsigned value. */
  ltTile: boolean;
  /** The test that was intended: 0 <= signed < TILE_DIM. */
  intended: boolean;
}

/** Evaluate `t - R + f >= 0 && t - R + f < T` with t unsigned, as the printed Fig. 7.15 does. */
export function guard(t: number, R: number, f: number, T: number): Guard {
  const signed = t - R + f;
  const u32 = signed >>> 0; // unsigned int arithmetic wraps modulo 2^32
  return { signed, u32, geZero: u32 >= 0, ltTile: u32 < T, intended: signed >= 0 && signed < T };
}

export interface Tap {
  fRow: number;
  fCol: number;
  /** The input element this tap needs. */
  inRow: number;
  inCol: number;
  kind: TapKind;
  /** For a shared tap: the slot the corrected kernel reads, [row, col] of N_s. */
  slot: [number, number] | null;
  /** For a shared tap: the slot printed line 21 reads, N_s[ty+fRow][tx+fCol]. */
  printed: [number, number] | null;
  /** The printed slot is not the element the tap needs (true for every shared tap when R > 0). */
  wrongElement: boolean;
  /** The printed index reaches T or beyond in a dimension: past the edge of the tile. */
  pastTile: boolean;
  /** The printed flat index (row*T + col) is beyond the T*T array: outside N_s altogether. */
  outsideArray: boolean;
  /** A shared tap whose slot holds a ghost zero, because this tile overhangs the image. */
  zeroInTile: boolean;
  /** The printed guard, x half (tx, fCol) and y half (ty, fRow). */
  gx: Guard;
  gy: Guard;
}

export interface BlockSpec {
  /** TILE_DIM: the block, the shared tile and the output tile are all T x T. */
  tile: number;
  /** FILTER_RADIUS. */
  r: number;
  width: number;
  height: number;
  bx: number;
  by: number;
}

export interface ThreadTaps {
  tx: number;
  ty: number;
  row: number;
  col: number;
  /** The thread's output is inside the image, so it computes (line 13's guard). */
  active: boolean;
  /** Its (2R+1)^2 taps in loop order (fRow outer); empty for an inactive thread. */
  taps: Tap[];
}

export const filterDim = (r: number) => 2 * r + 1;

/** Grid of blocks for an image: ceil(width/T) by ceil(height/T). */
export const gridDims = (tile: number, width: number, height: number) => ({
  gx: Math.ceil(width / tile),
  gy: Math.ceil(height / tile),
});

/** Is (row, col) inside the image? */
const inImage = (row: number, col: number, width: number, height: number) => row >= 0 && row < height && col >= 0 && col < width;

/** Every tap of one thread, classified as the corrected kernel does, with what the printed kernel would do. */
export function classifyTaps(a: BlockSpec & { tx: number; ty: number }): ThreadTaps {
  const { tile: T, r: R, width, height, bx, by, tx, ty } = a;
  const col = bx * T + tx;
  const row = by * T + ty;
  const active = row < height && col < width;
  const taps: Tap[] = [];
  if (active) {
    const m = filterDim(R);
    for (let fRow = 0; fRow < m; fRow++)
      for (let fCol = 0; fCol < m; fCol++) {
        const gx = guard(tx, R, fCol, T);
        const gy = guard(ty, R, fRow, T);
        const inRow = row - R + fRow;
        const inCol = col - R + fCol;
        // The printed guard: (>= 0 && < T) on unsigned values, which equals the intended test.
        const inTile = gx.geZero && gx.ltTile && gy.geZero && gy.ltTile;
        const kind: TapKind = inTile ? 'shared' : inImage(inRow, inCol, width, height) ? 'global' : 'ghost';
        const slot: [number, number] | null = inTile ? [ty - R + fRow, tx - R + fCol] : null;
        const printed: [number, number] | null = inTile ? [ty + fRow, tx + fCol] : null;
        taps.push({
          fRow,
          fCol,
          inRow,
          inCol,
          kind,
          slot,
          printed,
          wrongElement: !!printed && (printed[0] !== slot![0] || printed[1] !== slot![1]),
          pastTile: !!printed && (printed[0] >= T || printed[1] >= T),
          outsideArray: !!printed && printed[0] * T + printed[1] >= T * T,
          zeroInTile: inTile && !inImage(inRow, inCol, width, height),
          gx,
          gy,
        });
      }
  }
  return { tx, ty, row, col, active, taps };
}

export interface BlockHistogram {
  threads: number;
  /** Threads whose output is inside the image: they load an element and compute. */
  active: number;
  taps: number;
  shared: number;
  global: number;
  ghost: number;
  /** Shared taps that read a ghost zero stored in the tile (only where a tile overhangs the image). */
  zeroInTile: number;
  /** Different halo elements the block's global taps read. */
  distinctGlobal: number;
  /** Reads of N per output if every global tap goes to DRAM on its own: (loads + global taps) / outputs. */
  readsPerOutputAllTaps: number;
  /** Reads of N per output if each halo element is fetched once per block: (loads + distinct) / outputs. */
  readsPerOutputPerBlock: number;
  /** Printed line 21: shared taps that read another element, that run past the tile, that leave N_s. */
  printedWrong: number;
  printedPastTile: number;
  printedOutside: number;
}

/** Count the tap kinds over every thread of one block. */
export function blockHistogram(spec: BlockSpec): BlockHistogram {
  const T = spec.tile;
  const h: BlockHistogram = {
    threads: T * T, active: 0, taps: 0, shared: 0, global: 0, ghost: 0, zeroInTile: 0, distinctGlobal: 0,
    readsPerOutputAllTaps: 0, readsPerOutputPerBlock: 0, printedWrong: 0, printedPastTile: 0, printedOutside: 0,
  };
  const halo = new Set<string>();
  for (let ty = 0; ty < T; ty++)
    for (let tx = 0; tx < T; tx++) {
      const t = classifyTaps({ ...spec, tx, ty });
      if (!t.active) continue;
      h.active++;
      for (const p of t.taps) {
        h.taps++;
        h[p.kind]++;
        if (p.zeroInTile) h.zeroInTile++;
        if (p.kind === 'global') halo.add(p.inRow + ',' + p.inCol);
        if (p.wrongElement) h.printedWrong++;
        if (p.pastTile) h.printedPastTile++;
        if (p.outsideArray) h.printedOutside++;
      }
    }
  h.distinctGlobal = halo.size;
  h.readsPerOutputAllTaps = h.active ? (h.active + h.global) / h.active : 0;
  h.readsPerOutputPerBlock = h.active ? (h.active + h.distinctGlobal) / h.active : 0;
  return h;
}

/** An interior block: block (1, 1) of a 3T x 3T image, every tap's element inside the image. */
export const interiorSpec = (tile: number, r: number): BlockSpec => ({ tile, r, width: 3 * tile, height: 3 * tile, bx: 1, by: 1 });

export interface TrafficBounds {
  /** Each input element read from DRAM once: the ideal (book p. 167). */
  ideal: number;
  /** Fig. 7.15, each block fetching each halo element once (its own repeats merged, nothing shared across blocks). */
  perBlock: number;
  /** Fig. 7.15, every global tap a separate DRAM read: no cache helps at any level. */
  allTaps: number;
  /** Fig. 7.12 with a T x T input tile (output tile T - 2R), by the book's own count (p. 176). */
  fig712: number;
}

/** Reads of N from DRAM per output for an interior block, under the four assumptions above. */
export function readsPerOutput(tile: number, r: number): TrafficBounds {
  const h = blockHistogram(interiorSpec(tile, r));
  const out = tile - 2 * r;
  return {
    ideal: 1,
    perBlock: h.readsPerOutputPerBlock,
    allTaps: h.readsPerOutputAllTaps,
    fig712: out > 0 ? (tile * tile) / (out * out) : Infinity,
  };
}

/**
 * FLOP per byte of DRAM traffic for a given number of 4-byte reads of N per output, counting
 * 2m^2 FLOP and one 4-byte store per output, as the book does on pp. 167 and 176.
 */
export const intensity = (reads: number, r: number, bytes = 4) => (2 * filterDim(r) ** 2) / (bytes * (reads + 1));

export interface WarpSplit {
  warp: number;
  /** First and last thread id (ty*T + tx) of the warp. */
  first: number;
  last: number;
  lanes: number;
  /** Taps at which the warp's lanes disagree on the in-tile test. */
  splitTaps: number;
  diverges: boolean;
}

/**
 * Where a warp's lanes disagree on the in-tile test of Fig. 7.15 (lines 17 to 20). The test
 * depends only on threadIdx, so the answer is the same for every block. Warps are
 * `warpSize` consecutive thread ids, tx fastest; a block smaller than a warp is one
 * partial warp.
 */
export function warpSplits(tile: number, r: number, warpSize: number): { warps: WarpSplit[]; diverging: number } {
  const T = tile;
  const m = filterDim(r);
  const n = T * T;
  const warps: WarpSplit[] = [];
  for (let w = 0; w * warpSize < n; w++) {
    const first = w * warpSize;
    const last = Math.min(n, first + warpSize) - 1;
    let splitTaps = 0;
    for (let fRow = 0; fRow < m; fRow++)
      for (let fCol = 0; fCol < m; fCol++) {
        let some = false;
        let all = true;
        for (let id = first; id <= last; id++) {
          const tx = id % T;
          const ty = Math.floor(id / T);
          const inside = guard(tx, r, fCol, T).intended && guard(ty, r, fRow, T).intended;
          some ||= inside;
          all &&= inside;
        }
        if (some && !all) splitTaps++;
      }
    warps.push({ warp: w, first, last, lanes: last - first + 1, splitTaps, diverges: splitTaps > 0 });
  }
  return { warps, diverging: warps.filter((w) => w.diverges).length };
}

// ---------------------------------------------------------------- running the whole kernel

export interface Problem {
  width: number;
  height: number;
  r: number;
  /** Row-major input, small integers so results are exact. */
  N: number[];
  /** Row-major (2r+1) x (2r+1) filter, not symmetric, so a shifted read shows. */
  F: number[];
  /** Direct convolution with zero ghost cells (book p. 162), un-flipped as the book writes it. */
  ref: number[];
}

function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Direct 2D convolution, zero ghost cells: P[y][x] = sum F[j+r][k+r] * N[y+j][x+k]. */
export function reference(N: number[], F: number[], width: number, height: number, r: number): number[] {
  const m = filterDim(r);
  const P = new Array<number>(width * height).fill(0);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      let s = 0;
      for (let j = 0; j < m; j++)
        for (let k = 0; k < m; k++) {
          const iy = y - r + j;
          const ix = x - r + k;
          if (inImage(iy, ix, width, height)) s += F[j * m + k] * N[iy * width + ix];
        }
      P[y * width + x] = s;
    }
  return P;
}

export function makeProblem(width: number, height: number, r: number, seed = 7): Problem {
  const g = rng(seed);
  const m = filterDim(r);
  const N = Array.from({ length: width * height }, () => Math.floor(g() * 10));
  const F = Array.from({ length: m * m }, () => 1 + Math.floor(g() * 5));
  return { width, height, r, N, F, ref: reference(N, F, width, height, r) };
}

export interface CachedRun {
  P: number[];
  /** Flat indices of outputs that differ from the reference. */
  wrong: number[];
  correct: boolean;
  /** Shared reads whose printed index ran past the tile but stayed inside N_s (they read the next row). */
  pastTileReads: number;
  /** Shared reads beyond the whole N_s array (undefined behaviour; the model returns NaN). */
  outsideReads: number;
}

/** Run Fig. 7.15 over the whole image, block by block: corrected, or with line 21 as printed. */
export function runCached(prob: Problem, opts: { tile: number; asPrinted?: boolean }): CachedRun {
  const { width, height, r, N, F } = prob;
  const T = opts.tile;
  const m = filterDim(r);
  const { gx, gy } = gridDims(T, width, height);
  const P = new Array<number>(width * height).fill(NaN);
  let pastTileReads = 0;
  let outsideReads = 0;
  for (let by = 0; by < gy; by++)
    for (let bx = 0; bx < gx; bx++) {
      // Load phase: each thread its own element, ghost zero where the tile overhangs.
      const Ns = new Array<number>(T * T);
      for (let ty = 0; ty < T; ty++)
        for (let tx = 0; tx < T; tx++) {
          const row = by * T + ty;
          const col = bx * T + tx;
          Ns[ty * T + tx] = row < height && col < width ? N[row * width + col] : 0;
        }
      // Compute phase.
      for (let ty = 0; ty < T; ty++)
        for (let tx = 0; tx < T; tx++) {
          const t = classifyTaps({ tile: T, r, width, height, bx, by, tx, ty });
          if (!t.active) continue;
          let s = 0;
          for (const p of t.taps) {
            const w = F[p.fRow * m + p.fCol];
            if (p.kind === 'shared') {
              const [sr, sc] = opts.asPrinted ? p.printed! : p.slot!;
              const flat = sr * T + sc;
              if (opts.asPrinted && p.pastTile) pastTileReads++;
              if (flat >= T * T) {
                outsideReads++;
                s += w * NaN;
              } else s += w * Ns[flat];
            } else if (p.kind === 'global') s += w * N[p.inRow * width + p.inCol];
          }
          P[t.row * width + t.col] = s;
        }
    }
  const wrong: number[] = [];
  P.forEach((v, i) => {
    if (!(v === prob.ref[i])) wrong.push(i);
  });
  return { P, wrong, correct: wrong.length === 0, pastTileReads: pastTileReads - outsideReads, outsideReads };
}

/**
 * Which kind of read each element of the block's footprint gets, and from how many taps:
 * the (T + 2R) x (T + 2R) square around the tile, indexed from its top-left corner
 * (element (0, 0) is image element (by*T - R, bx*T - R)).
 */
export function footprint(spec: BlockSpec): { size: number; kind: (TapKind | null)[]; taps: number[] } {
  const { tile: T, r: R } = spec;
  const size = T + 2 * R;
  const kind = new Array<TapKind | null>(size * size).fill(null);
  const taps = new Array<number>(size * size).fill(0);
  for (let ty = 0; ty < T; ty++)
    for (let tx = 0; tx < T; tx++)
      for (const p of classifyTaps({ ...spec, tx, ty }).taps) {
        const i = (ty + p.fRow) * size + (tx + p.fCol);
        kind[i] = p.kind;
        taps[i]++;
      }
  return { size, kind, taps };
}
