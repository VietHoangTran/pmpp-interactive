// OUR TOY MODEL, NOT THE BOOK'S AND NOT ANY VENDOR'S: will a block's halo be in the L2
// cache when it reads it? For section 7.6 (book pp. 178 to 179), whose premise is that halo
// cells are "a significant probability" already in L2 because neighbouring blocks loaded
// them. The book shows no model and no measurement of that; this file makes the premise
// concrete enough to play with, and nothing more.
//
// The model:
//   - An image of tilesX x tilesY tiles of T x T floats, row-major, one 4-byte float per
//     element, starting at the beginning of a cache line.
//   - A cache line holds `lineFloats` consecutive floats (32 = 128 bytes by default: an
//     assumption, not a vendor figure). Line of element (row, col) = floor((row*width + col) / lineFloats).
//   - Blocks are dispatched in waves of `resident` blocks, in block-index order (bx fastest).
//     Within a wave, every block first loads its own lines (its tile, or for Fig. 7.12 its
//     whole input tile), then every block reads its halo lines (Fig. 7.15 only). The wave's
//     blocks run at once, so within each of those two phases their accesses take turns,
//     one line per block per turn, each block in line order (row-major).
//   - Within one block, each line is fetched at most once: we assume the block's repeated
//     reads of a line are merged or served by the SM's own L1, which is not modelled otherwise.
//   - The L2 is `slices` independent caches (1 = one L2 shared by all SMs, as NVIDIA
//     documents; more = one per die, as on AMD MI300X), each fully associative with
//     least-recently-used replacement and an equal share of `capacity` lines.
//   - `placement` says which slice serves each block: 'in-order' gives each wave's blocks
//     to the slices in contiguous runs, 'round-robin' deals them out in turn, 'random' picks
//     with a seeded generator. Which placement a real GPU uses is not documented (AMD says
//     its workgroup-to-CU mapping is non-deterministic: facts ch07 B5).
//   - DRAM traffic = one line per miss, plus one 4-byte store per output.
//
// What it does not model: real replacement policies, set associativity and address
// hashing, sectors smaller than a line, the actual block scheduler (blocks start as others
// finish, not in waves), timing of any kind, other traffic sharing the cache (the filter,
// the stores, other kernels), write allocation, prefetching, L1 hits across blocks, and
// any vendor's real sizes.

export type Placement = 'in-order' | 'round-robin' | 'random';
/** 'cached' = Fig. 7.15 (tile interior in shared memory, halo through the cache); 'halo' = Fig. 7.12 (whole input tile loaded). */
export type CacheKernel = 'cached' | 'halo';

export interface CacheConfig {
  tilesX: number;
  tilesY: number;
  /** Block edge T: tile and output tile for Fig. 7.15, input tile for Fig. 7.12. */
  tile: number;
  r: number;
  /** Floats per cache line. 32 floats = 128 bytes. */
  lineFloats: number;
  /** Lines in the whole L2, split evenly across slices. Infinity for an unbounded cache. */
  capacity: number;
  /** Blocks resident at once. */
  resident: number;
  slices: number;
  placement: Placement;
  seed: number;
  kernel?: CacheKernel;
}

export interface CacheEvent {
  /** Block index in dispatch order. */
  block: number;
  slice: number;
  phase: 'load' | 'halo';
  line: number;
  hit: boolean;
}

export interface BlockStat {
  block: number;
  bx: number;
  by: number;
  slice: number;
  loadLines: number;
  loadHits: number;
  haloLines: number;
  haloHits: number;
  /** haloHits / haloLines, or null for a block with no halo lines in the image. */
  haloHitFraction: number | null;
}

export interface CacheRun {
  cfg: CacheConfig;
  width: number;
  height: number;
  gx: number;
  gy: number;
  outputs: number;
  accesses: number;
  hits: number;
  /** DRAM line fetches. */
  misses: number;
  blocks: BlockStat[];
  /** Every access in order (only when asked for). */
  events: CacheEvent[];
  /** Lines in each slice when the run stopped, least recently used first. */
  contents: number[][];
}

export const FLOAT_BYTES = 4;
export const lineBytes = (lineFloats: number) => lineFloats * FLOAT_BYTES;
export const imageDims = (cfg: Pick<CacheConfig, 'tilesX' | 'tilesY' | 'tile'>) => ({
  width: cfg.tilesX * cfg.tile,
  height: cfg.tilesY * cfg.tile,
});

/** A small seeded generator (mulberry32): the same seed always gives the same placement. */
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

/** The grid each kernel launches over the image. */
export function grid(cfg: CacheConfig): { gx: number; gy: number; out: number } {
  const { width, height } = imageDims(cfg);
  const out = (cfg.kernel ?? 'cached') === 'cached' ? cfg.tile : cfg.tile - 2 * cfg.r;
  if (out < 1) return { gx: 0, gy: 0, out };
  return { gx: Math.ceil(width / out), gy: Math.ceil(height / out), out };
}

/** The slice that serves each block, in dispatch order. */
export function placeBlocks(n: number, cfg: Pick<CacheConfig, 'slices' | 'placement' | 'seed' | 'resident'>): number[] {
  const k = Math.max(1, cfg.slices);
  const R = Math.max(1, cfg.resident);
  const g = rng(cfg.seed);
  return Array.from({ length: n }, (_, i) => {
    if (k === 1) return 0;
    if (cfg.placement === 'round-robin') return i % k;
    if (cfg.placement === 'random') return Math.floor(g() * k);
    const waveStart = i - (i % R);
    const size = Math.min(R, n - waveStart);
    return Math.floor(((i - waveStart) * k) / size);
  });
}

/**
 * The lines one block touches, split into the lines it loads into shared memory and the
 * lines only its halo reads touch (each line once per block).
 */
export function blockLines(cfg: CacheConfig, bx: number, by: number): { load: number[]; halo: number[] } {
  const { width, height } = imageDims(cfg);
  const { tile: T, r, lineFloats: L } = cfg;
  const kernel = cfg.kernel ?? 'cached';
  const load = new Set<number>();
  const halo = new Set<number>();
  // Add the lines covering columns [c0, c1) of one row, clipped to the image.
  const span = (into: Set<number>, row: number, c0: number, c1: number, skip?: Set<number>) => {
    if (row < 0 || row >= height) return;
    const a = Math.max(0, c0);
    const b = Math.min(width, c1);
    if (a >= b) return;
    for (let l = Math.floor((row * width + a) / L); l <= Math.floor((row * width + b - 1) / L); l++)
      if (!skip?.has(l)) into.add(l);
  };
  if (kernel === 'cached') {
    const r0 = by * T;
    const c0 = bx * T;
    for (let row = r0; row < r0 + T; row++) span(load, row, c0, c0 + T);
    for (let row = r0 - r; row < r0 + T + r; row++) span(halo, row, c0 - r, c0 + T + r, load);
  } else {
    const out = T - 2 * r;
    const r0 = by * out - r;
    const c0 = bx * out - r;
    for (let row = r0; row < r0 + T; row++) span(load, row, c0, c0 + T);
  }
  const sorted = (s: Set<number>) => [...s].sort((a, b) => a - b);
  return { load: sorted(load), halo: sorted(halo) };
}

/** A fully associative LRU cache of `cap` lines, on JavaScript's Map insertion order. */
// (No TypeScript parameter properties here: the browser checks import this file with Node's type stripping.)
class Lru {
  private m = new Map<number, true>();
  private cap: number;
  constructor(cap: number) {
    this.cap = cap;
  }
  access(line: number): boolean {
    if (this.m.has(line)) {
      this.m.delete(line);
      this.m.set(line, true);
      return true;
    }
    if (this.cap <= 0) return false;
    this.m.set(line, true);
    if (this.m.size > this.cap) this.m.delete(this.m.keys().next().value as number);
    return false;
  }
  contents(): number[] {
    return [...this.m.keys()];
  }
}

/** Run the model. `stopAfter` stops after that many accesses (for drawing a moment of the run). */
export function simulate(cfg: CacheConfig, opts: { trace?: boolean; stopAfter?: number } = {}): CacheRun {
  const { width, height } = imageDims(cfg);
  const { gx, gy } = grid(cfg);
  const n = gx * gy;
  const k = Math.max(1, cfg.slices);
  const per = cfg.capacity === Infinity ? Infinity : Math.floor(cfg.capacity / k);
  const caches = Array.from({ length: k }, () => new Lru(per));
  const slice = placeBlocks(n, cfg);
  const lines = Array.from({ length: n }, (_, i) => blockLines(cfg, i % gx, Math.floor(i / gx)));
  const blocks: BlockStat[] = lines.map((l, i) => ({
    block: i, bx: i % gx, by: Math.floor(i / gx), slice: slice[i],
    loadLines: l.load.length, loadHits: 0, haloLines: l.halo.length, haloHits: 0, haloHitFraction: null,
  }));
  const events: CacheEvent[] = [];
  const stop = opts.stopAfter ?? Infinity;
  let accesses = 0;
  let hits = 0;
  const R = Math.max(1, cfg.resident);
  outer: for (let w = 0; w < n; w += R) {
    const wave = Array.from({ length: Math.min(R, n - w) }, (_, j) => w + j);
    for (const phase of ['load', 'halo'] as const) {
      // The wave's blocks run at once: their accesses take turns, one line each.
      const longest = Math.max(...wave.map((b) => lines[b][phase].length));
      for (let at = 0; at < longest; at++)
        for (const b of wave) {
          const line = lines[b][phase][at];
          if (line === undefined) continue;
          if (accesses >= stop) break outer;
          const hit = caches[slice[b]].access(line);
          accesses++;
          if (hit) {
            hits++;
            if (phase === 'load') blocks[b].loadHits++;
            else blocks[b].haloHits++;
          }
          if (opts.trace) events.push({ block: b, slice: slice[b], phase, line, hit });
        }
    }
  }
  for (const b of blocks) b.haloHitFraction = b.haloLines ? b.haloHits / b.haloLines : null;
  return {
    cfg, width, height, gx, gy, outputs: width * height, accesses, hits, misses: accesses - hits, blocks, events,
    contents: caches.map((c) => c.contents()),
  };
}

/** Lines the image occupies: the ideal, each fetched once. */
export const idealLines = (cfg: CacheConfig) => {
  const { width, height } = imageDims(cfg);
  return Math.ceil((width * height) / cfg.lineFloats);
};

/** Every block fetches every line it touches: no reuse between blocks (capacity 0). */
export function noReuseLines(cfg: CacheConfig): number {
  const { gx, gy } = grid(cfg);
  let s = 0;
  for (let by = 0; by < gy; by++)
    for (let bx = 0; bx < gx; bx++) {
      const l = blockLines(cfg, bx, by);
      s += l.load.length + l.halo.length;
    }
  return s;
}

/** DRAM bytes per output for a number of line fetches: the lines plus one 4-byte store per output. */
export function bytesPerOutput(cfg: CacheConfig, lineFetches: number): number {
  const { width, height } = imageDims(cfg);
  const outputs = width * height;
  return (lineFetches * lineBytes(cfg.lineFloats) + outputs * FLOAT_BYTES) / outputs;
}

/** FLOP per DRAM byte: 2m^2 FLOP per output (book p. 167) over the bytes per output. */
export const effectiveIntensity = (cfg: CacheConfig, bpo: number) => (2 * (2 * cfg.r + 1) ** 2) / bpo;

export interface CacheSummary {
  run: CacheRun;
  /** Bytes per output: this run, the ideal, Fig. 7.15 with no reuse between blocks, Fig. 7.12 with none, Fig. 7.12 through the same cache. */
  bytes: { run: number; ideal: number; allMiss: number; fig712: number; fig712Cached: number };
  intensity: { run: number; ideal: number; allMiss: number; fig712: number; fig712Cached: number };
  lines: { run: number; ideal: number; allMiss: number; fig712: number; fig712Cached: number };
  /** Halo lines that hit, over all halo lines of all blocks. */
  haloHitFraction: number | null;
}

/** Everything the explorer shows for one setting. */
export function summarise(cfg: CacheConfig): CacheSummary {
  const run = simulate({ ...cfg, kernel: 'cached' });
  const halo = { ...cfg, kernel: 'halo' as const };
  const canHalo = grid(halo).gx > 0;
  const lines = {
    run: run.misses,
    ideal: idealLines(cfg),
    allMiss: noReuseLines({ ...cfg, kernel: 'cached' }),
    fig712: canHalo ? noReuseLines(halo) : NaN,
    fig712Cached: canHalo ? simulate(halo).misses : NaN,
  };
  const b = (l: number) => bytesPerOutput(cfg, l);
  const bytes = { run: b(lines.run), ideal: b(lines.ideal), allMiss: b(lines.allMiss), fig712: b(lines.fig712), fig712Cached: b(lines.fig712Cached) };
  const i = (x: number) => effectiveIntensity(cfg, x);
  const intensity = { run: i(bytes.run), ideal: i(bytes.ideal), allMiss: i(bytes.allMiss), fig712: i(bytes.fig712), fig712Cached: i(bytes.fig712Cached) };
  const hl = run.blocks.reduce((s, x) => s + x.haloLines, 0);
  const hh = run.blocks.reduce((s, x) => s + x.haloHits, 0);
  return { run, bytes, intensity, lines, haloHitFraction: hl ? hh / hl : null };
}

/** Lines one row of tiles occupies: a yardstick for the cache's capacity. */
export const linesPerTileRow = (cfg: CacheConfig) => {
  const { width } = imageDims(cfg);
  return Math.ceil((width * cfg.tile) / cfg.lineFloats);
};

// ---------------------------------------------------------------- the scene's two runs
//
// "Where the halo comes from" (section 7.6) draws these two runs and nothing else. A small
// image (8 x 4 tiles of 8 x 8, radius 2) with a cache line exactly one tile row long, so
// a line is easy to see; a 48-line cache, less than one row of tiles' worth; one block at
// a time. Block A is the corner tile, B its right-hand neighbour, C the first block of the
// next row. The split run deals the blocks to two slices at random (seed 1).

export const SCENE_ONE_L2: CacheConfig = {
  tilesX: 8, tilesY: 4, tile: 8, r: 2, lineFloats: 8, capacity: 48, resident: 1, slices: 1, placement: 'in-order', seed: 1,
};
export const SCENE_SPLIT_L2: CacheConfig = { ...SCENE_ONE_L2, slices: 2, placement: 'random', seed: 1 };
export const SCENE_A = 0;
export const SCENE_B = 1;
/** The first block of the next row: a whole row of blocks after A. */
export const SCENE_C = SCENE_ONE_L2.tilesX;

/** Indices of the events of one block (optionally one phase) in a traced run. */
export const eventsOf = (run: CacheRun, block: number, phase?: 'load' | 'halo') =>
  run.events.flatMap((e, i) => (e.block === block && (!phase || e.phase === phase) ? [i] : []));

/** Line of image element (row, col). */
export const lineAt = (cfg: CacheConfig, row: number, col: number) =>
  Math.floor((row * imageDims(cfg).width + col) / cfg.lineFloats);

export interface SceneFacts {
  one: CacheRun;
  split: CacheRun;
  /** A's interior lines: B's left halo reads exactly these (the line is one tile row). */
  aLines: number[];
  /** B's loads that hit, of B's loads (one shared L2). */
  bLoadHits: number;
  bLoads: number;
  /** B's halo reads of A's lines that hit, of those reads (one shared L2). */
  bLeftHits: number;
  bLeft: number;
  /** A's bottom two rows, which C's top halo reads: how many of C's reads of them hit. */
  aBottom: number[];
  cOnABottomHits: number;
  cOnABottom: number;
  /** Lines the cache dropped between B's last access and C's first. */
  evictedBeforeC: number;
  /** Split run: the slices A and B landed on, and B's reads of A's lines that hit. */
  sliceA: number;
  sliceB: number;
  splitBLeftHits: number;
  splitBLeft: number;
  /** Bytes per output: all hits (ideal), the two runs, and no reuse between blocks. */
  bytes: { ideal: number; one: number; split: number; noReuse: number };
}

export function sceneFacts(): SceneFacts {
  const one = simulate(SCENE_ONE_L2, { trace: true });
  const split = simulate(SCENE_SPLIT_L2, { trace: true });
  const cfg = SCENE_ONE_L2;
  const T = cfg.tile;
  const aLines = blockLines(cfg, 0, 0).load;
  const aSet = new Set(aLines);
  const aBottom = [lineAt(cfg, T - 2, 0), lineAt(cfg, T - 1, 0)];
  const bSet = new Set(aBottom);
  const pick = (run: CacheRun, block: number, phase: 'load' | 'halo', set?: Set<number>) =>
    eventsOf(run, block, phase).map((i) => run.events[i]).filter((e) => !set || set.has(e.line));
  const bLoad = pick(one, SCENE_B, 'load');
  const bLeft = pick(one, SCENE_B, 'halo', aSet);
  const cOn = pick(one, SCENE_C, 'halo', bSet);
  const sLeft = pick(split, SCENE_B, 'halo', aSet);
  const lastB = Math.max(...eventsOf(one, SCENE_B));
  const firstC = Math.min(...eventsOf(one, SCENE_C));
  const before = new Set(simulate(cfg, { stopAfter: lastB + 1 }).contents[0]);
  const after = new Set(simulate(cfg, { stopAfter: firstC }).contents[0]);
  return {
    one, split, aLines, aBottom,
    bLoadHits: bLoad.filter((e) => e.hit).length, bLoads: bLoad.length,
    bLeftHits: bLeft.filter((e) => e.hit).length, bLeft: bLeft.length,
    cOnABottomHits: cOn.filter((e) => e.hit).length, cOnABottom: cOn.length,
    evictedBeforeC: [...before].filter((l) => !after.has(l)).length,
    sliceA: split.blocks[SCENE_A].slice, sliceB: split.blocks[SCENE_B].slice,
    splitBLeftHits: sLeft.filter((e) => e.hit).length, splitBLeft: sLeft.length,
    bytes: {
      ideal: bytesPerOutput(cfg, idealLines(cfg)),
      one: bytesPerOutput(cfg, one.misses),
      split: bytesPerOutput(cfg, split.misses),
      noReuse: bytesPerOutput(cfg, noReuseLines(cfg)),
    },
  };
}
