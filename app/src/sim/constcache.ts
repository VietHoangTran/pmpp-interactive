// One warp reading a table in constant memory (book §7.4, p. 171).
//
// What it models is the one rule the vendors document:
// - NVIDIA (notes/today/ch07-facts.md A3, Best Practices Guide §10.2.6): accesses to
//   different addresses by the threads of a warp are serialised, and the cost grows
//   linearly with the number of distinct addresses; when every thread reads the same
//   address the read can be as fast as a register. Here: one constant-cache read per
//   distinct address.
// - AMD (B1): `__constant__` data goes through the scalar L1 cache only when the read is
//   uniform across the wavefront; otherwise the accesses are serialised. Here: a yes or
//   no, "can the scalar cache serve this read", and nothing more. AMD gives no size.
// - Apple (C2): no constant cache is documented, so nothing is modelled for Apple GPUs.
//
// What it leaves out: every read is assumed to hit (no misses against NVIDIA's 8 KB
// working set per SM, no trips to device memory), there is no latency or clock, and
// the order in which distinct addresses are served is drawn in lane order only so that
// a picture can be made; no vendor documents the order.

export type Pattern = 'uniform' | 'lane' | 'laneGroup' | 'codebook';

export interface ReadOptions {
  /** Lanes in the warp: 32 on NVIDIA, 64 or 32 on AMD. */
  warpSize: number;
  /** Entries in a codebook (only the 'codebook' pattern uses it). */
  tableSize?: number;
  /** Lanes that share an entry in the 'laneGroup' pattern, T[lane / groupSize]. */
  groupSize?: number;
  /** Seed for the codebook's per-lane codes. */
  seed?: number;
  /** The entry every lane reads in the 'uniform' pattern (a filter tap). */
  entry?: number;
}

/** A small, fast, deterministic generator (mulberry32), so a seed always gives the same codes. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * The seed the section 7.4 explorer starts on and the "One address, one warp" scene draws:
 * 32 lanes, 16 entries, 14 distinct (pinned in the tests; the narration says 14).
 */
export const DEMO_SEED = 3;

/** Uniformly random codes in [0, k), one per lane: the weight codes a warp decodes at once. */
export function codes(seed: number, lanes: number, k: number): number[] {
  const r = rng(seed);
  return Array.from({ length: lanes }, () => Math.floor(r() * k));
}

/** The table entry each lane of one warp reads, indexed by lane. */
export function warpAddresses(pattern: Pattern, o: ReadOptions): number[] {
  const lanes = Array.from({ length: o.warpSize }, (_, i) => i);
  switch (pattern) {
    case 'uniform':
      return lanes.map(() => o.entry ?? 0);
    case 'lane':
      return lanes;
    case 'laneGroup':
      return lanes.map((l) => Math.floor(l / (o.groupSize ?? 8)));
    case 'codebook':
      return codes(o.seed ?? 1, o.warpSize, o.tableSize ?? 16);
  }
}

/** Distinct entries the warp asks for. */
export const distinctCount = (addresses: readonly number[]) => new Set(addresses).size;

/** NVIDIA's documented cost: one constant-cache read per distinct address (A3). */
export const nvidiaCost = distinctCount;

/** AMD's scalar cache serves `__constant__` reads only when every lane reads the same address (B1). */
export const amdScalarEligible = (addresses: readonly number[]) =>
  addresses.length > 0 && addresses.every((a) => a === addresses[0]);

/** Expected number of distinct entries when `lanes` lanes pick uniformly among k: k(1 - (1 - 1/k)^lanes). */
export const expectedDistinct = (k: number, lanes: number) => k * (1 - (1 - 1 / k) ** lanes);

export interface Round {
  /** The entry answered in this round. */
  address: number;
  /** Every lane that asked for it. */
  lanes: number[];
}

/** The serialised answers: one round per distinct address, in the order lanes first ask (drawing order only). */
export function replaySchedule(addresses: readonly number[]): Round[] {
  const rounds: Round[] = [];
  const at = new Map<number, Round>();
  addresses.forEach((a, lane) => {
    let r = at.get(a);
    if (!r) {
      r = { address: a, lanes: [] };
      at.set(a, r);
      rounds.push(r);
    }
    r.lanes.push(lane);
  });
  return rounds;
}

export interface Tap {
  fRow: number;
  fCol: number;
  /** Flat index into F, fRow*(2r+1)+fCol. */
  address: number;
}

/**
 * The filter reads of one warp for radius r: every lane reads the same tap, in the
 * same order, so each tap is one uniform read and the whole filter costs (2r+1)²
 * constant-cache reads per warp whatever the warp width.
 */
export function filterTrace(r: number): Tap[] {
  const m = 2 * r + 1;
  const out: Tap[] = [];
  for (let fRow = 0; fRow < m; fRow++) for (let fCol = 0; fCol < m; fCol++) out.push({ fRow, fCol, address: fRow * m + fCol });
  return out;
}

/** Largest odd filter width whose elements fit in `capacity` bytes (m^dims · bytes ≤ capacity). */
export function largestOddWidth(capacity: number, bytes = 4, dims: 1 | 2 = 2): number {
  let m = Math.floor((capacity / bytes) ** (1 / dims) + 1e-9);
  if (m % 2 === 0) m -= 1;
  while (m > 1 && m ** dims * bytes > capacity) m -= 2;
  return Math.max(m, 0);
}
