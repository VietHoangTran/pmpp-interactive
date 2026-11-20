// Chapter 17, exercise 3 (p. 422) and the bridge on page 17.5: converting COO to CSR with
// a histogram, an exclusive scan and a scatter, and the same four passes as they appear in
// mixture-of-experts routing. It runs the narrated animation "Routing tokens to experts",
// the start page's figure and scan question, and the routing questions of page 17.5.
//
// Sources:
//  - the book, §17.2 and §17.3, pp. 404–408: COO's rowIdx, colIdx and value; CSR's
//    rowPtrs, with rowPtrs[numRows] as the end marker (Figs. 17.3, 17.7); p. 410 and
//    exercise 3: the conversion "using parallel primitives including histogram and prefix
//    sum"; the notes' answer sketch (atomically incremented per-row cursor, within-row
//    order then non-deterministic);
//  - chapter 9 (histogram) and chapter 11 §11.1 (exclusive scan, pp. 252–254);
//  - p. 417: the JDS row array puts results back in the original order;
//  - ch17-facts D5 to D7 for what llama.cpp and mlx-lm do (ranks from a ballot and a
//    popcount, ids padded for unconditional access, Metal slices padded by one entry,
//    argsort and its inverse once there are 64 or more entries). The mapping of their code
//    onto the book's formats is ours.
//  - Appendix A §A.6, pp. 593–594: the order of floating-point additions changes the sum.
//
// What it leaves out: time, parallel execution itself (every pass is computed in one go),
// the router's weights and the expert GEMMs' arithmetic.

// ---------------------------------------------------------------- the three passes
/** Histogram: how many entries each row (expert) has. */
export function countByRow(rowIdx: readonly number[], numRows: number): number[] {
  const out = Array(numRows).fill(0);
  for (const r of rowIdx) out[r]++;
  return out;
}

/** Exclusive scan of the counts with the total appended: CSR's rowPtrs (Fig. 17.7). */
export function exclusiveScan(counts: readonly number[]): number[] {
  const out = [0];
  for (const c of counts) out.push(out[out.length - 1] + c);
  return out;
}

export interface Scatter {
  /** Where input entry i lands. */
  pos: number[];
  /** Which input entry sits at output position p: the inverse of `pos`. */
  order: number[];
  /** Entry i's rank inside its row. */
  rank: number[];
}

/**
 * The stable scatter: an entry's rank is the number of earlier entries in its row, which is
 * what a ballot over the lanes below plus a popcount computes, chunk after chunk. Input
 * order is kept inside every row, whatever order the threads happen to run in.
 */
export function stableScatter(rowIdx: readonly number[], rowPtrs: readonly number[]): Scatter {
  const seen = Array(rowPtrs.length).fill(0);
  const rank = rowIdx.map((r) => seen[r]++);
  const pos = rowIdx.map((r, i) => rowPtrs[r] + rank[i]);
  return { pos, order: inversePermutation(pos), rank };
}

/**
 * The atomic-cursor scatter: entries take `cursor[row]++` in the order they arrive. The
 * order of atomics on one address is undefined (ch09-facts A3), so `arrival` is an input:
 * a different arrival order gives a different order inside a row.
 */
export function cursorScatter(rowIdx: readonly number[], rowPtrs: readonly number[], arrival: readonly number[]): Scatter {
  const cursor = [...rowPtrs];
  const pos = Array(rowIdx.length).fill(-1);
  for (const i of arrival) pos[i] = cursor[rowIdx[i]]++;
  const rank = pos.map((p, i) => p - rowPtrs[rowIdx[i]]);
  return { pos, order: inversePermutation(pos), rank };
}

/** inv[perm[i]] = i. */
export function inversePermutation(perm: readonly number[]): number[] {
  const inv = Array(perm.length).fill(-1);
  perm.forEach((p, i) => (inv[p] = i));
  return inv;
}

/** out[p] = xs[order[p]]: gather in the order given. */
export const gather = <T>(xs: readonly T[], order: readonly number[]): T[] => order.map((i) => xs[i]);

/** A stable argsort, as mlx-lm's mx.argsort(indices) is used: equal keys keep their order. */
export const argsort = (keys: readonly number[]): number[] => keys.map((_, i) => i).sort((a, b) => keys[a] - keys[b] || a - b);

/** COO to CSR in three passes (exercise 3). Within a row, entries keep their input order. */
export function cooToCsr(rowIdx: readonly number[], colIdx: readonly number[], value: readonly number[], numRows: number) {
  const rowPtrs = exclusiveScan(countByRow(rowIdx, numRows));
  const { order } = stableScatter(rowIdx, rowPtrs);
  return { rowPtrs, colIdx: gather(colIdx, order), value: gather(value, order) };
}

export interface Slices {
  /** Where each row's padded slice starts, with the padded total appended. */
  starts: number[];
  /** Each slice's padded length: the count rounded up to a multiple of the tile. */
  padded: number[];
  total: number;
  pads: number;
}
/** Pads every non-empty slice to a multiple of `tile`, so every tile is full and every store unconditional. */
export function padSlices(counts: readonly number[], tile: number): Slices {
  const padded = counts.map((c) => Math.ceil(c / tile) * tile);
  const starts = exclusiveScan(padded);
  const total = starts[starts.length - 1];
  return { starts, padded, total, pads: total - counts.reduce((a, b) => a + b, 0) };
}

// ---------------------------------------------------------------- the book's 4 × 4 example
/** Fig. 17.1 in COO, sorted by row then column (Fig. 17.3). */
export const FIG171 = {
  rows: 4,
  cols: 4,
  rowIdx: [0, 0, 1, 1, 1, 2, 2, 3],
  colIdx: [0, 1, 0, 2, 3, 1, 2, 3],
  value: [1, 7, 5, 3, 9, 2, 8, 6],
} as const;

/** Fig. 17.1 as a dense 4 × 4 array, zeros where there is no entry. */
export function fig171Dense(): number[][] {
  const a = Array.from({ length: FIG171.rows }, () => Array(FIG171.cols).fill(0));
  FIG171.rowIdx.forEach((r, i) => (a[r][FIG171.colIdx[i]] = FIG171.value[i]));
  return a;
}

// ---------------------------------------------------------------- summation order
/**
 * Every result that adding `values` into 0 in float32 can give, over all orders of the
 * additions (what atomic adds into one y element may do). Sorted, distinct.
 */
export function float32Sums(values: readonly number[]): number[] {
  const f = Math.fround;
  const out = new Set<number>();
  const go = (acc: number, left: number[]) => {
    if (!left.length) {
      out.add(acc === 0 ? 0 : acc); // fold -0 into 0
      return;
    }
    left.forEach((v, k) => go(f(acc + f(v)), left.filter((_, j) => j !== k)));
  };
  go(0, [...values]);
  return [...out].sort((a, b) => a - b);
}

// ---------------------------------------------------------------- the routing example
export interface Pair {
  /** Flat slot: token · topk + choice. */
  slot: number;
  token: number;
  /** 0 for the router's first choice, 1 for its second. */
  choice: number;
  expert: number;
}

export interface Route {
  tokens: number;
  experts: number;
  topk: number;
  tile: number;
  /** The router's picks in token order: a COO list with rows = experts, columns = tokens. */
  pairs: Pair[];
  counts: number[];
  /** CSR's rowPtrs over experts. */
  offsets: number[];
  /**
   * Stable scatter of the pairs into expert order. `scatter.order` (expert-order position to
   * slot) is mlx-lm's argsort and plays JDS's row array; `scatter.pos` (slot to position) is
   * its inverse, the argsort of the argsort.
   */
  scatter: Scatter;
  slices: Slices;
  /** Padded position of each slot in the tiled layout. */
  paddedPos: number[];
  /** Experts with no tokens: their GEMM is skipped. */
  empty: number[];
}

/** The scene's fixed example: 8 tokens, top-2, 4 experts, expert 2 chosen by no token. */
export const ROUTER_PICKS: readonly (readonly [number, number])[] = [
  [0, 3],
  [1, 0],
  [3, 1],
  [0, 1],
  [3, 0],
  [1, 3],
  [0, 3],
  [1, 0],
];

export function route(picks: readonly (readonly number[])[] = ROUTER_PICKS, experts = 4, tile = 4): Route {
  const topk = picks[0]?.length ?? 0;
  const pairs: Pair[] = picks.flatMap((ps, token) => ps.map((expert, choice) => ({ slot: token * topk + choice, token, choice, expert })));
  const rowIdx = pairs.map((p) => p.expert);
  const counts = countByRow(rowIdx, experts);
  const offsets = exclusiveScan(counts);
  const scatter = stableScatter(rowIdx, offsets);
  const slices = padSlices(counts, tile);
  const paddedPos = pairs.map((p, i) => slices.starts[p.expert] + scatter.rank[i]);
  return {
    tokens: picks.length,
    experts,
    topk,
    tile,
    pairs,
    counts,
    offsets,
    scatter,
    slices,
    paddedPos,
    empty: counts.flatMap((c, e) => (c === 0 ? [e] : [])),
  };
}

/** Counts after the first k pairs have been counted (beat 2 of the scene). */
export const countsAfter = (r: Route, k: number): number[] =>
  countByRow(
    r.pairs.slice(0, Math.max(0, Math.min(k, r.pairs.length))).map((p) => p.expert),
    r.experts,
  );
