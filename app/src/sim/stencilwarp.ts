// Chapter 8, section 8.4 (p. 193): one warp's view of the cube-tiled kernel (Fig. 8.8).
// For the load, which global addresses its lanes read and how many 32-byte segments
// those addresses fall in; for the compute phase, which lanes pass the guards and
// whether the warp is split. It runs the explorer "One warp's load" on page 8.4 and the
// numbers in the coalescing and divergence questions.
//
// Sources (the lessons cite them where they are used):
//  - notes/ch08.md: Fig. 8.8's index arithmetic (k = blockIdx.x·OUT_TILE_DIM +
//    threadIdx.x − 1 and likewise j and i; flat index i·N² + j·N + k), its load guard
//    0 ≤ i, j, k < N and its two compute guards (grid interior and block interior)
//    (pp. 191 to 192); the book's claim that an 8 × 8 × 8 tile is not coalesced and that
//    blocks 32 wide help (p. 193);
//  - notes/today/ch08-facts.md A2: NVIDIA's documented model, in which a warp's load is
//    served by 32-byte transactions and efficiency is bytes used over bytes fetched; the
//    table there (8 transactions and 50% for 8 × 8 × 8, 5 and 80% for 32 × 8 × 4) is what
//    the tests pin. B2: AMD's 64-wide wavefronts, with no documented transaction size.
//  - The linear order (x fastest, then y, then z, cut into runs of the warp width) is the
//    rule of chapter 4 (book pp. 75 to 76; ch04 facts A8). It is written out again below so
//    that this file imports nothing; the tests check it against src/sim/warps.ts.
//
// What this leaves out: it applies NVIDIA's documented unit and nothing else (AMD
// documents no size, Apple no model at all: facts B2, C4). Arrays start aligned and N is
// a multiple of 8, so every row starts at the same alignment. No L1 or L2, no DRAM pages,
// no reuse between warps, no timing. "Divergent" means some but not all of a warp's
// threads pass the compute guards; padded lanes are not threads and do not count.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

export interface Dims {
  x: number;
  y: number;
  z: number;
}
export const dims3 = (x: number, y = 1, z = 1): Dims => ({ x, y, z });

/** NVIDIA's documented transaction size, in bytes (facts A2). */
export const SEGMENT_BYTES = 32;

export interface WarpConfig {
  /** Threads per block, which is the input tile (Fig. 8.8). A block with z = 1 is one plane, as in §8.5. */
  block: Dims;
  /** Stencil order: 1 for the 7-point stencil. */
  r: number;
  /** Which block of the grid. */
  blockIdx: Dims;
  warp: number;
  /** 32, or 64 for AMD's wave64. */
  width: number;
  /** Grid side. A multiple of 8 keeps every row at the same alignment. */
  N: number;
  /** Added to each coordinate: −r as in the book, 0 to compare. */
  offset: number;
  /** For a one-plane block: the plane i it works on. */
  plane?: number;
  /** Element size, 4 for float. */
  bytes?: number;
}

export interface Lane {
  lane: number;
  /** Position in the block (threadIdx). */
  t: Dims;
  /** Grid coordinates: i (z), j (y), k (x). */
  i: number;
  j: number;
  k: number;
  /** Flat index i·N² + j·N + k, or null when the load guard skips it. */
  index: number | null;
  /** Passes both compute guards: grid interior and block interior. */
  active: boolean;
}

export interface Row {
  i: number;
  j: number;
  kFirst: number;
  kLast: number;
  /** Lanes that load from this row. */
  lanes: number;
  /** First and last 32-byte segment this row's loads touch. */
  segFirst: number;
  segLast: number;
}

export type WarpState = 'full' | 'divergent' | 'idle';

export interface WarpView {
  warp: number;
  /** Real threads in this warp (fewer than the width in a last, padded warp). */
  threads: number;
  padded: number;
  lanes: Lane[];
  /** Lanes that load something. */
  loading: number;
  rows: Row[];
  /** Distinct 32-byte segments the load touches, ascending. */
  segments: number[];
  transactions: number;
  bytesUsed: number;
  bytesFetched: number;
  /** bytesUsed / bytesFetched, 0 when nothing loads. */
  efficiency: number;
  /** Lanes that pass the compute guards. */
  active: number;
  state: WarpState;
}

export const threadsIn = (d: Dims): number => d.x * d.y * d.z;
export const warpsPerBlock = (d: Dims, width: number): number => Math.ceil(threadsIn(d) / width);

/** The thread at a position of the block's linear order: x fastest, then y, then z. */
export const coordAt = (n: number, d: Dims): Dims => ({
  x: n % d.x,
  y: Math.floor(n / d.x) % d.y,
  z: Math.floor(n / (d.x * d.y)),
});

/** Output tile side along one axis: the input tile less r on each face. */
const outSide = (n: number, r: number) => n - 2 * r;

/** Grid coordinates of a thread, by Fig. 8.8's index arithmetic (or §8.5's, for a plane). */
export function gridCoords(cfg: WarpConfig, t: Dims): { i: number; j: number; k: number } {
  const { block, r, blockIdx, offset } = cfg;
  const k = blockIdx.x * outSide(block.x, r) + t.x + offset;
  const j = blockIdx.y * outSide(block.y, r) + t.y + offset;
  const i = block.z === 1 ? (cfg.plane ?? r) : blockIdx.z * outSide(block.z, r) + t.z + offset;
  return { i, j, k };
}

const inGrid = (v: number, N: number) => v >= 0 && v < N;
const inside = (v: number, lo: number, hi: number) => v >= lo && v < hi;

/** Fig. 8.8's compute guards: an interior grid point, and a thread that is not on the block's outer layer. */
export function passesCompute(cfg: WarpConfig, t: Dims, g: { i: number; j: number; k: number }): boolean {
  const { block, r, N } = cfg;
  const grid = inside(g.i, r, N - r) && inside(g.j, r, N - r) && inside(g.k, r, N - r);
  const blk = inside(t.x, r, block.x - r) && inside(t.y, r, block.y - r) && (block.z === 1 || inside(t.z, r, block.z - r));
  return grid && blk;
}

export const stateOf = (active: number, threads: number): WarpState =>
  active === 0 ? 'idle' : active === threads ? 'full' : 'divergent';

/** Everything one warp does in the load and at the compute guard. */
export function warpView(cfg: WarpConfig): WarpView {
  const { block, width, N, warp } = cfg;
  const bytes = cfg.bytes ?? 4;
  const total = threadsIn(block);
  const lo = warp * width;
  const hi = Math.min(total, lo + width);
  const lanes: Lane[] = [];
  for (let n = lo; n < hi; n++) {
    const t = coordAt(n, block);
    const g = gridCoords(cfg, t);
    const loads = inGrid(g.i, N) && inGrid(g.j, N) && inGrid(g.k, N);
    lanes.push({
      lane: n - lo,
      t,
      ...g,
      index: loads ? g.i * N * N + g.j * N + g.k : null,
      active: passesCompute(cfg, t, g),
    });
  }
  const seg = (index: number) => Math.floor((index * bytes) / SEGMENT_BYTES);
  const segs = new Set<number>();
  const rowMap = new Map<string, Row>();
  for (const l of lanes) {
    if (l.index === null) continue;
    const s = seg(l.index);
    segs.add(s);
    const key = l.i + ',' + l.j;
    const row = rowMap.get(key);
    if (!row) rowMap.set(key, { i: l.i, j: l.j, kFirst: l.k, kLast: l.k, lanes: 1, segFirst: s, segLast: s });
    else {
      row.kFirst = Math.min(row.kFirst, l.k);
      row.kLast = Math.max(row.kLast, l.k);
      row.lanes++;
      row.segFirst = Math.min(row.segFirst, s);
      row.segLast = Math.max(row.segLast, s);
    }
  }
  const segments = [...segs].sort((a, b) => a - b);
  const loading = lanes.filter((l) => l.index !== null).length;
  const bytesUsed = loading * bytes;
  const bytesFetched = segments.length * SEGMENT_BYTES;
  const active = lanes.filter((l) => l.active).length;
  return {
    warp,
    threads: lanes.length,
    padded: width - lanes.length,
    lanes,
    loading,
    rows: [...rowMap.values()].sort((a, b) => a.i - b.i || a.j - b.j),
    segments,
    transactions: segments.length,
    bytesUsed,
    bytesFetched,
    efficiency: bytesFetched ? bytesUsed / bytesFetched : 0,
    active,
    state: stateOf(active, lanes.length),
  };
}

export interface StripCell {
  /** Position along the row: k (negative or ≥ N means the end of the previous row or the start of the next). */
  k: number;
  index: number;
  segment: number;
  /** In a segment the warp's load touches. */
  fetched: boolean;
  /** The lane that loads this element, or null. */
  lane: number | null;
}

/**
 * The stretch of global memory around one row a warp loads from: the segments the row
 * touches, plus `context` segments on each side, element by element.
 */
export function rowStrip(cfg: WarpConfig, view: WarpView, row: Row, context = 1): StripCell[] {
  const bytes = cfg.bytes ?? 4;
  const base = row.i * cfg.N * cfg.N + row.j * cfg.N;
  const first = Math.ceil(((row.segFirst - context) * SEGMENT_BYTES) / bytes);
  const last = Math.ceil(((row.segLast + 1 + context) * SEGMENT_BYTES) / bytes) - 1;
  const byIndex = new Map<number, number>();
  for (const l of view.lanes) if (l.index !== null) byIndex.set(l.index, l.lane);
  const fetched = new Set(view.segments);
  const out: StripCell[] = [];
  for (let index = Math.max(0, first); index <= last; index++) {
    const segment = Math.floor((index * bytes) / SEGMENT_BYTES);
    out.push({ k: index - base, index, segment, fetched: fetched.has(segment), lane: byIndex.get(index) ?? null });
  }
  return out;
}

export interface BlockView {
  warps: WarpView[];
  transactions: number;
  bytesUsed: number;
  bytesFetched: number;
  efficiency: number;
  full: number;
  divergent: number;
  idle: number;
}

/** Every warp of the block, and the block's totals. */
export function blockView(cfg: Omit<WarpConfig, 'warp'>): BlockView {
  const n = warpsPerBlock(cfg.block, cfg.width);
  const warps = Array.from({ length: n }, (_, w) => warpView({ ...cfg, warp: w }));
  const sum = (f: (w: WarpView) => number) => warps.reduce((a, w) => a + f(w), 0);
  const bytesUsed = sum((w) => w.bytesUsed);
  const bytesFetched = sum((w) => w.bytesFetched);
  return {
    warps,
    transactions: sum((w) => w.transactions),
    bytesUsed,
    bytesFetched,
    efficiency: bytesFetched ? bytesUsed / bytesFetched : 0,
    full: warps.filter((w) => w.state === 'full').length,
    divergent: warps.filter((w) => w.state === 'divergent').length,
    idle: warps.filter((w) => w.state === 'idle').length,
  };
}
