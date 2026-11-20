// Chapter 19, section 19.2: the launch geometry of the direct kernel (Figs. 19.5 to 19.7)
// and which addresses one warp of it reads from X (Exercise 2, p. 475).
//
// What it models:
//  - Fig. 19.5's host code (p. 461): W_grid and H_grid by integer division, as printed, or
//    by ceiling division, the fix the notes call for; T = H_grid * W_grid tiles per output
//    map; blockDim(TILE_WIDTH, TILE_WIDTH, 1) and gridDim(M, T, N).
//  - Fig. 19.7's index recovery (p. 463): m = blockIdx.x, h = (blockIdx.y / W_grid) * TILE
//    + threadIdx.y, w = (blockIdx.y % W_grid) * TILE + threadIdx.x, n = blockIdx.z, with an
//    optional planted bug that swaps / and %.
//  - Where every thread's store lands: the flat index ((n*M + m)*H_out + h)*W_out + w. A
//    thread outside its map either does nothing (bounds check) or, with no check, stores into
//    another pixel's slot, or past the end of Y. Coverage counts how many stores each pixel
//    receives.
//  - CUDA's documented grid limits (x up to 2^31 - 1, y and z up to 65535:
//    notes/today/ch05-facts.md A5, ch19-facts.md A11) and HIP's documented cap of 2^32 - 1
//    threads per grid (ch19-facts.md B2). Metal documents no compute-grid maximum (C1).
//  - One warp's loads from X at one (c, p, q) step for the layouts NCHW, NHWC and CHWN, and
//    the number of aligned segments they touch, counted with coalesce.ts (NVIDIA's documented
//    32-byte sectors, or AMD's 64- and 128-byte lines: ch06-facts.md A2, B1). Lanes are the
//    block's threads in row-major order cut into runs of the warp width (ch04-facts.md A8).
//
// What it leaves out:
//  - No timing. Indices only: which thread computes which pixel, and which addresses it asks
//    for. Not what a cache does with them. Apple documents no coalescing rule (ch06-facts C1).
//  - A stray store's value is not modelled, only where it lands. With no bounds check the
//    stray thread also reads X outside its patch; the emulator reports that, not this model.
//
// Erasable TypeScript only.

import { runs, warpAccess, type WarpAccess } from './coalesce';

export type Division = 'floor' | 'ceil';

export interface LaunchParams {
  N: number;
  M: number;
  Hout: number;
  Wout: number;
  tile: number;
  /** 'floor' is Fig. 19.5 as printed; 'ceil' rounds up so edge tiles exist. */
  division: Division;
}

export interface Geometry {
  Wgrid: number;
  Hgrid: number;
  /** Tiles per output map, H_grid * W_grid. */
  T: number;
  grid: { x: number; y: number; z: number };
  block: { x: number; y: number; z: number };
  threadsPerBlock: number;
  blocks: number;
  threads: number;
}

const div = (a: number, b: number, d: Division) => (d === 'floor' ? Math.floor(a / b) : Math.ceil(a / b));

export function geometry(p: LaunchParams): Geometry {
  const Wgrid = div(p.Wout, p.tile, p.division);
  const Hgrid = div(p.Hout, p.tile, p.division);
  const T = Hgrid * Wgrid;
  const grid = { x: p.M, y: T, z: p.N };
  const block = { x: p.tile, y: p.tile, z: 1 };
  const threadsPerBlock = p.tile * p.tile;
  const blocks = grid.x * grid.y * grid.z;
  return { Wgrid, Hgrid, T, grid, block, threadsPerBlock, blocks, threads: blocks * threadsPerBlock };
}

export interface ThreadPos {
  bx: number;
  by: number;
  bz: number;
  tx: number;
  ty: number;
}

export interface Mapped {
  n: number;
  m: number;
  h: number;
  w: number;
  /** Inside its output map: h < H_out and w < W_out. */
  inRange: boolean;
  /** Flat index of Y[n, m, h, w] as the kernel computes it, even when out of range. */
  yIndex: number;
}

/** Fig. 19.7 lines 3 to 6. With `swap`, lines 4 and 5 trade / and % (the planted bug). */
export function mapThread(p: LaunchParams, g: Geometry, t: ThreadPos, swap = false): Mapped {
  const row = swap ? t.by % g.Wgrid : Math.floor(t.by / g.Wgrid);
  const col = swap ? Math.floor(t.by / g.Wgrid) : t.by % g.Wgrid;
  const n = t.bz;
  const m = t.bx;
  const h = row * p.tile + t.ty;
  const w = col * p.tile + t.tx;
  return { n, m, h, w, inRange: h < p.Hout && w < p.Wout, yIndex: ((n * p.M + m) * p.Hout + h) * p.Wout + w };
}

/** The thread that computes Y[n, m, h, w] when nothing is wrong (no swap), or null if no tile covers it. */
export function ownerOf(p: LaunchParams, g: Geometry, n: number, m: number, h: number, w: number, swap = false): ThreadPos | null {
  const tr = Math.floor(h / p.tile);
  const tc = Math.floor(w / p.tile);
  if (!swap) {
    if (tr >= g.Hgrid || tc >= g.Wgrid) return null;
    return { bx: m, by: tr * g.Wgrid + tc, bz: n, tx: w % p.tile, ty: h % p.tile };
  }
  // Swapped: row = by % W_grid and col = by / W_grid, so by = col * W_grid + row with row < W_grid.
  if (tr >= g.Wgrid) return null;
  const by = tc * g.Wgrid + tr;
  if (by >= g.T) return null;
  return { bx: m, by, bz: n, tx: w % p.tile, ty: h % p.tile };
}

export interface Options {
  swap: boolean;
  /** The bounds check `if (h < H_out && w < W_out)` around the store. */
  guard: boolean;
}

export interface Coverage {
  /** Stores received by each pixel of the chosen sample, flat M x H_out x W_out. */
  counts: Int32Array;
  never: number;
  once: number;
  /** Pixels that received two or more stores. */
  multi: number;
  /** Threads of this sample outside their map that store nothing (bounds check). */
  idle: number;
  /** Threads of this sample outside their map that store anyway (no bounds check). */
  stray: number;
  /** Of those, stores that land past the end of Y. */
  pastEnd: number;
  /** Threads launched for this sample. */
  threads: number;
}

/** The largest grid this model will walk thread by thread. */
export const COVERAGE_LIMIT = 2_000_000;

/**
 * Where the stores of every thread land, seen from sample `sample`: all threads of that sample,
 * plus the previous sample's (with no bounds check their stores can spill forward into it).
 * Returns null when that is more than COVERAGE_LIMIT threads.
 */
export function coverage(p: LaunchParams, opts: Options, sample = 0): Coverage | null {
  const g = geometry(p);
  const per = g.grid.x * g.grid.y * g.threadsPerBlock;
  const from = Math.max(0, sample - 1);
  if (per * (sample - from + 1) > COVERAGE_LIMIT) return null;
  const mapSize = p.M * p.Hout * p.Wout;
  const base = sample * mapSize;
  const total = p.N * mapSize;
  const counts = new Int32Array(mapSize);
  let idle = 0;
  let stray = 0;
  let pastEnd = 0;
  for (let bz = from; bz <= sample; bz++)
    for (let bx = 0; bx < g.grid.x; bx++)
      for (let by = 0; by < g.grid.y; by++)
        for (let ty = 0; ty < p.tile; ty++)
          for (let tx = 0; tx < p.tile; tx++) {
            const r = mapThread(p, g, { bx, by, bz, tx, ty }, opts.swap);
            const mine = bz === sample;
            if (!r.inRange) {
              if (opts.guard) {
                if (mine) idle++;
                continue;
              }
              if (mine) stray++;
              if (r.yIndex >= total) {
                if (mine) pastEnd++;
                continue;
              }
            }
            const local = r.yIndex - base;
            if (local >= 0 && local < mapSize) counts[local]++;
          }
  let never = 0;
  let once = 0;
  let multi = 0;
  for (const c of counts) {
    if (c === 0) never++;
    else if (c === 1) once++;
    else multi++;
  }
  return { counts, never, once, multi, idle, stray, pastEnd, threads: per };
}

/** Every thread whose store lands on Y[n, m, h, w] (with no bounds check, strays included). */
export function writersOf(p: LaunchParams, opts: Options, n: number, m: number, h: number, w: number): ThreadPos[] {
  const g = geometry(p);
  const target = ((n * p.M + m) * p.Hout + h) * p.Wout + w;
  const out: ThreadPos[] = [];
  for (let bz = Math.max(0, n - 1); bz <= Math.min(n, p.N - 1); bz++)
    for (let bx = 0; bx < g.grid.x; bx++)
      for (let by = 0; by < g.grid.y; by++)
        for (let ty = 0; ty < p.tile; ty++)
          for (let tx = 0; tx < p.tile; tx++) {
            const t = { bx, by, bz, tx, ty };
            const r = mapThread(p, g, t, opts.swap);
            if (!r.inRange && opts.guard) continue;
            if (r.yIndex === target) out.push(t);
          }
  return out;
}

// ---------------------------------------------------------------- documented grid limits

/** CUDA, every listed compute capability: ch05-facts.md A5 (and ch19-facts.md A11). */
export const CUDA_GRID = { x: 2 ** 31 - 1, y: 65535, z: 65535 };
/** HIP 7.17.0 hardware features table: y and z 2^16 - 1, at most 2^32 - 1 threads per grid (ch19-facts.md B2). */
export const HIP_GRID = { y: 2 ** 16 - 1, z: 2 ** 16 - 1, threads: 2 ** 32 - 1 };

export interface LimitCheck {
  cuda: { x: boolean; y: boolean; z: boolean; ok: boolean };
  hip: { y: boolean; z: boolean; threads: boolean; ok: boolean };
}

export function limits(g: Geometry): LimitCheck {
  const cx = g.grid.x <= CUDA_GRID.x;
  const cy = g.grid.y <= CUDA_GRID.y;
  const cz = g.grid.z <= CUDA_GRID.z;
  const hy = g.grid.y <= HIP_GRID.y;
  const hz = g.grid.z <= HIP_GRID.z;
  const ht = g.threads <= HIP_GRID.threads;
  return { cuda: { x: cx, y: cy, z: cz, ok: cx && cy && cz }, hip: { y: hy, z: hz, threads: ht, ok: hy && hz && ht } };
}

// ---------------------------------------------------------------- one warp's reads from X

export type Layout = 'NCHW' | 'NHWC' | 'CHWN';
export const LAYOUTS: readonly Layout[] = ['NCHW', 'NHWC', 'CHWN'];

export interface Dims {
  N: number;
  C: number;
  H: number;
  W: number;
}

/** Flat index of X[n, c, h, w] in each layout (row-major in the order the name spells). */
export function xIndex(layout: Layout, d: Dims, n: number, c: number, h: number, w: number): number {
  switch (layout) {
    case 'NCHW':
      return ((n * d.C + c) * d.H + h) * d.W + w;
    case 'NHWC':
      return ((n * d.H + h) * d.W + w) * d.C + c;
    case 'CHWN':
      return ((c * d.H + h) * d.W + w) * d.N + n;
  }
}

/**
 * Which thread does which work.
 *  - 'w': Fig. 19.7. Lanes are threads of block (m 0, tile 0, sample 0) in row-major order,
 *    so neighbouring lanes take neighbouring w.
 *  - 'n': a redesign (ours): each lane of a warp takes a different sample at the same output
 *    pixel, which is what CHWN would need.
 */
export type LaneMap = 'w' | 'n';

export interface WarpReadParams {
  layout: Layout;
  dims: Dims;
  K: number;
  tile: number;
  warpWidth: number;
  /** Which warp of the block (or, for 'n', which output pixel in row-major order). */
  warp: number;
  c: number;
  p: number;
  q: number;
  lanes: LaneMap;
  segmentBytes: number;
}

export interface Lane {
  lane: number;
  n: number;
  h: number;
  w: number;
  /** Element index into X and its byte address. */
  index: number;
  byte: number;
}

export interface WarpReads {
  lanes: Lane[];
  /** Index into F, the same for every lane (Fig. 19.7 reads F[m, c, p, q] with m from blockIdx.x). */
  fIndex: number;
  access: WarpAccess;
  /** Separate runs of consecutive bytes. */
  runs: number;
  /** Byte distance between the first two active lanes, or null. */
  stride: number | null;
}

export function warpReads(r: WarpReadParams): WarpReads {
  const { dims, K, tile } = r;
  const Ho = dims.H - K + 1;
  const Wo = dims.W - K + 1;
  const lanes: Lane[] = [];
  for (let lane = 0; lane < r.warpWidth; lane++) {
    let n: number;
    let h: number;
    let w: number;
    if (r.lanes === 'w') {
      const t = r.warp * r.warpWidth + lane;
      if (t >= tile * tile) continue;
      n = 0;
      h = Math.floor(t / tile);
      w = t % tile;
    } else {
      n = lane;
      if (n >= dims.N) continue;
      h = Math.floor(r.warp / Wo);
      w = r.warp % Wo;
    }
    if (h >= Ho || w >= Wo) continue;
    const index = xIndex(r.layout, dims, n, r.c, h + r.p, w + r.q);
    lanes.push({ lane, n, h, w, index, byte: index * 4 });
  }
  const bytes = lanes.map((l) => l.byte);
  return {
    lanes,
    fIndex: ((0 * dims.C + r.c) * K + r.p) * K + r.q,
    access: warpAccess(bytes, 4, r.segmentBytes),
    runs: runs(bytes, 4),
    stride: lanes.length > 1 ? lanes[1].byte - lanes[0].byte : null,
  };
}
