// One step of the Floyd-Warshall kernel, block by block. Chapter 16, section 16.3
// (book §16.4, pp. 377–381, Figs. 16.3 to 16.5).
//
// What it models, exactly: the corrected kernel of labs/ch16/snippets/fw_bottomup.cu with
// one block per table row (the lesson's 8-vertex example fits one block). Each block's
// thread 0 reads dist[row][k] once into a shared scalar; if that is the sentinel FW_INF the
// whole block retires; otherwise each thread reads dist[k][col], skips a sentinel, and
// writes dist[row][k] + dist[k][col] if it is smaller than its own cell. A step runs in
// place (the book's one table) or with two buffers (read the old table, write a new one).
//
// What it leaves out: real blocks interleave arbitrarily, and their threads too. Here the
// blocks run one after another, in a chosen order, each to completion. That is one of the
// orders a GPU may produce, enough to show that a step whose own row and column change
// has a result that depends on the order (a race); it cannot show every outcome.
//
// The example table is Fig. 16.5's distance table as printed: b to f is 9 there, while
// the graph beside it labels that edge 6 (notes/ch16.md, "Unclear in the PDF").

/** The integer sentinel for "no path": INT_MAX / 2, so that FW_INF + FW_INF fits in an int. */
export const INF = Math.floor(2147483647 / 2);

export const VERTICES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] as const;

const EDGES: [string, string, number][] = [
  ['a', 'b', 4], ['a', 'g', 7], ['a', 'h', 4],
  ['b', 'c', 9], ['b', 'f', 9], ['b', 'g', 8], ['b', 'h', 1],
  ['c', 'e', 10],
  ['e', 'c', 8], ['e', 'd', 6], ['e', 'f', 5],
  ['f', 'e', 6],
  ['g', 'b', 4], ['g', 'f', 7],
  ['h', 'c', 3],
];

/** Fig. 16.5's table "after initialization": 0 on the diagonal, FW_INF where it is blank. */
export const FIG_16_5: readonly (readonly number[])[] = (() => {
  const n = VERTICES.length;
  const d = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 0 : INF)));
  const at = (v: string) => VERTICES.indexOf(v as (typeof VERTICES)[number]);
  for (const [u, v, w] of EDGES) d[at(u)][at(v)] = w;
  return d;
})();

export type Mode = 'in-place' | 'two-buffers';
export type Order = 'top-first' | 'bottom-first';

export interface Write {
  col: number;
  from: number;
  to: number;
}

/** What one block (one table row) did during the step. */
export interface BlockEvent {
  row: number;
  /** dist[row][k] as thread 0 read it into the shared scalar. */
  distKCol: number;
  /** The block retired because dist[row][k] is the sentinel. */
  retired: boolean;
  /** dist[k][col] as each thread read it (FW_INF where there is no path from k). */
  rowK: number[];
  writes: Write[];
}

export interface Cell {
  i: number;
  j: number;
  from: number;
  to: number;
}

export interface StepTrace {
  k: number;
  mode: Mode;
  order: Order;
  before: number[][];
  blocks: BlockEvent[];
  after: number[][];
  /** Every cell whose value differs between before and after. */
  changed: Cell[];
  /** Rows whose block retired, in table order. */
  retired: number[];
  /** Row k or column k differs between before and after. */
  rowKChanged: boolean;
  colKChanged: boolean;
}

const copy = (d: readonly (readonly number[])[]) => d.map((r) => [...r]);

export interface StepOptions {
  mode?: Mode;
  order?: Order;
  /** Overrides dist[k][k] before the step (the lesson sets it to -1 to break the assumption). */
  diag?: number;
}

/** Runs step k of the kernel on a copy of dist and records what every block did. */
export function stepTrace(dist: readonly (readonly number[])[], k: number, opts: StepOptions = {}): StepTrace {
  const mode = opts.mode ?? 'in-place';
  const order = opts.order ?? 'top-first';
  const n = dist.length;
  const before = copy(dist);
  if (opts.diag !== undefined) before[k][k] = opts.diag;
  const out = copy(before); // the table blocks write to
  const read = mode === 'in-place' ? out : before; // the table blocks read from
  const rows = Array.from({ length: n }, (_, i) => (order === 'top-first' ? i : n - 1 - i));
  const blocks: BlockEvent[] = [];
  for (const row of rows) {
    const distKCol = read[row][k]; // thread 0, into the shared scalar, then a barrier
    if (distKCol === INF) {
      blocks.push({ row, distKCol, retired: true, rowK: [], writes: [] });
      continue;
    }
    const rowK: number[] = [];
    const writes: Write[] = [];
    for (let col = 0; col < n; col++) {
      const distKRow = read[k][col];
      rowK.push(distKRow);
      if (distKRow === INF) continue;
      const cand = distKCol + distKRow;
      // Each thread compares with its own cell, which only it writes.
      if (cand < out[row][col]) {
        writes.push({ col, from: out[row][col], to: cand });
        out[row][col] = cand;
      }
    }
    blocks.push({ row, distKCol, retired: false, rowK, writes });
  }
  const changed: Cell[] = [];
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) if (out[i][j] !== before[i][j]) changed.push({ i, j, from: before[i][j], to: out[i][j] });
  return {
    k,
    mode,
    order,
    before,
    blocks,
    after: out,
    changed,
    retired: blocks.filter((b) => b.retired).map((b) => b.row).sort((a, b) => a - b),
    rowKChanged: changed.some((c) => c.i === k),
    colKChanged: changed.some((c) => c.j === k),
  };
}

/** The table after the first m blocks of a traced step have run. */
export function stateAfter(t: StepTrace, m: number): number[][] {
  const d = copy(t.before);
  for (const b of t.blocks.slice(0, Math.max(0, m))) for (const w of b.writes) d[b.row][w.col] = w.to;
  return d;
}

/** Steps 0 .. upto-1, each in place, top row first: the table at the start of step upto. */
export function runSteps(dist: readonly (readonly number[])[], upto: number): number[][] {
  let d = copy(dist);
  for (let k = 0; k < upto; k++) d = stepTrace(d, k).after;
  return d;
}

/** Sequential Floyd-Warshall on the CPU, the reference (Fig. 16.3 without the parallel loops). */
export function floydWarshallCPU(dist: readonly (readonly number[])[]): number[][] {
  const d = copy(dist);
  const n = d.length;
  for (let k = 0; k < n; k++)
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        if (d[i][k] !== INF && d[k][j] !== INF && d[i][k] + d[k][j] < d[i][j]) d[i][j] = d[i][k] + d[k][j];
  return d;
}

/** A random directed graph with non-negative weights, as a distance table. Deterministic per seed. */
export function randomGraph(n: number, seed: number, density = 0.3, maxWeight = 20): number[][] {
  let x = seed >>> 0 || 1;
  const rnd = () => {
    x = (Math.imul(x, 1664525) + 1013904223) >>> 0;
    return x / 4294967296;
  };
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 0 : rnd() < density ? Math.floor(rnd() * (maxWeight + 1)) : INF)),
  );
}

export const show = (v: number) => (v === INF ? '' : String(v));
export const vname = (i: number) => VERTICES[i] ?? String(i);
