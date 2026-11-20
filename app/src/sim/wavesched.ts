// Who waits for whom: tiles of a wavefront on a few SMs. Chapter 16, section 16.3
// (book §16.8, pp. 397–399, Fig. 16.21; the relaunch per tile anti-diagonal of p. 384).
//
// A T x T grid of tiles; tile (r, c) needs the tile above it, (r - 1, c), and the tile to
// its left, (r, c - 1). S slots run one tile each per time unit. Three rules:
//
//  - 'front':   a barrier after every tile anti-diagonal (a relaunch per front, or a grid
//               barrier): the tiles of one front are spread over the slots, and the next
//               front starts only when the whole front is done.
//  - 'counter': one flag per tile. A free slot claims the next tile in front order (an
//               atomic counter) and spins in place until the flags of both neighbours are
//               set; then it computes the tile and sets the tile's flag.
//  - 'rows':    the book's persistent blocks, block i owning tile row i. Blocks start in
//               index order or reversed; a started block keeps its slot until its whole row
//               is done, spinning while the tile above its next tile is not done.
//
// Time is in steps. A tile that runs in step t is done at the end of t: a tile that needs
// it can run from step t + 1, and the slot it held is free from step t + 1.
//
// What it leaves out: every tile takes one step, a launch or barrier costs nothing, a
// spinning block costs its slot and nothing else, and slots start blocks in exactly the
// order given. The 'rows' rule with reversed start is one schedule that the missing
// ordering guarantee allows, not a model of any GPU's scheduler.

export type Rule = 'front' | 'counter' | 'rows';
export type StartOrder = 'index' | 'reversed';
export type SlotState = 'run' | 'spin' | 'idle';

export interface SlotCell {
  state: SlotState;
  /** The tile in the slot (running, or claimed and spinning), as its front-order number. */
  tile: number | null;
  /** For 'rows': the block in the slot. */
  block?: number;
}

export interface Schedule {
  T: number;
  S: number;
  rule: Rule;
  start: StartOrder;
  /** timeline[t][s]: what slot s does in step t. */
  timeline: SlotCell[][];
  /** Steps until every tile is done, or null if the run is stuck. */
  makespan: number | null;
  /** Slot-steps not spent computing a tile (idle or spinning), up to the makespan or the stuck step. */
  idle: number;
  /** doneAt[tile] (front-order number): the step at whose end the tile is done, or -1. */
  doneAt: number[];
  stuck: boolean;
  /** The step at which nothing could run any more, if stuck. */
  stuckAt: number | null;
  /** For 'front': the last step of each front, where the barrier sits. */
  barriers: number[];
}

/** Tiles in front order: anti-diagonal by anti-diagonal, top row first within one (Fig. 16.21). */
export function frontOrder(T: number): [number, number][] {
  const out: [number, number][] = [];
  for (let d = 0; d <= 2 * T - 2; d++)
    for (let r = 0; r < T; r++) {
      const c = d - r;
      if (c >= 0 && c < T) out.push([r, c]);
    }
  return out;
}

/** numbering[r][c]: the front-order number of tile (r, c). */
export function numbering(T: number): number[][] {
  const n = Array.from({ length: T }, () => new Array<number>(T).fill(-1));
  frontOrder(T).forEach(([r, c], i) => (n[r][c] = i));
  return n;
}

/** Sizes of the tile anti-diagonals: 1, 2, ..., T, ..., 2, 1. */
export const frontSizes = (T: number) => Array.from({ length: 2 * T - 1 }, (_, d) => Math.min(d, 2 * T - 2 - d, T - 1) + 1);

/** A relaunch per front: the sum over fronts of ceil(size / S). */
export const frontMakespan = (T: number, S: number) => frontSizes(T).reduce((a, n) => a + Math.ceil(n / S), 0);

function check(T: number, S: number) {
  if (!Number.isInteger(T) || T < 1) throw new Error(`T must be a positive integer, got ${T}`);
  if (!Number.isInteger(S) || S < 1) throw new Error(`S must be a positive integer, got ${S}`);
}

export function schedule(T: number, S: number, rule: Rule, start: StartOrder = 'index'): Schedule {
  check(T, S);
  const order = frontOrder(T);
  const num = numbering(T);
  const N = T * T;
  const doneAt = new Array<number>(N).fill(-1); // by front-order number
  const done = (r: number, c: number, t: number) => r < 0 || c < 0 || (doneAt[num[r][c]] >= 0 && doneAt[num[r][c]] < t);
  const timeline: SlotCell[][] = [];
  const barriers: number[] = [];
  let stuckAt: number | null = null;

  if (rule === 'front') {
    let t = 0;
    let i = 0;
    for (const size of frontSizes(T)) {
      const tiles = order.slice(i, i + size).map((_, j) => i + j);
      i += size;
      for (let a = 0; a < tiles.length; a += S) {
        const row: SlotCell[] = [];
        for (let s = 0; s < S; s++) {
          const tile = tiles[a + s];
          if (tile === undefined) row.push({ state: 'idle', tile: null });
          else {
            row.push({ state: 'run', tile });
            doneAt[tile] = t;
          }
        }
        timeline.push(row);
        t++;
      }
      barriers.push(t - 1);
    }
  } else if (rule === 'counter') {
    const slot = new Array<number>(S).fill(-1); // the tile each slot holds
    let next = 0;
    for (let t = 0; ; t++) {
      for (let s = 0; s < S; s++) if (slot[s] < 0 && next < N) slot[s] = next++;
      if (slot.every((x) => x < 0)) break;
      const row: SlotCell[] = [];
      const ran: number[] = [];
      for (let s = 0; s < S; s++) {
        const tile = slot[s];
        if (tile < 0) {
          row.push({ state: 'idle', tile: null });
          continue;
        }
        const [r, c] = order[tile];
        if (done(r - 1, c, t) && done(r, c - 1, t)) {
          row.push({ state: 'run', tile });
          ran.push(s);
        } else row.push({ state: 'spin', tile });
      }
      for (const s of ran) {
        doneAt[slot[s]] = t;
        slot[s] = -1;
      }
      timeline.push(row);
      if (!ran.length) {
        stuckAt = t; // cannot happen: the earliest claimed unfinished tile can always run
        break;
      }
    }
  } else {
    const queue = Array.from({ length: T }, (_, i) => (start === 'index' ? i : T - 1 - i));
    const slot = new Array<number>(S).fill(-1); // the block each slot holds
    const col = new Array<number>(T).fill(0); // the next tile column of each block
    for (let t = 0; ; t++) {
      for (let s = 0; s < S; s++) if (slot[s] < 0 && queue.length) slot[s] = queue.shift()!;
      if (slot.every((x) => x < 0)) break;
      const row: SlotCell[] = [];
      const ran: number[] = [];
      for (let s = 0; s < S; s++) {
        const b = slot[s];
        if (b < 0) {
          row.push({ state: 'idle', tile: null });
          continue;
        }
        const tile = num[b][col[b]];
        if (done(b - 1, col[b], t)) {
          row.push({ state: 'run', tile, block: b });
          ran.push(s);
        } else row.push({ state: 'spin', tile, block: b });
      }
      for (const s of ran) {
        const b = slot[s];
        doneAt[num[b][col[b]]] = t;
        if (++col[b] === T) slot[s] = -1;
      }
      timeline.push(row);
      if (!ran.length) {
        // Nothing ran and nothing will change: every slot holds a block waiting for a row
        // that has no slot, and no block ever gives its slot up.
        stuckAt = t;
        break;
      }
    }
  }

  const stuck = stuckAt !== null;
  let idle = 0;
  for (const row of timeline) for (const cell of row) if (cell.state !== 'run') idle++;
  return { T, S, rule, start, timeline, makespan: stuck ? null : timeline.length, idle, doneAt, stuck, stuckAt, barriers };
}

/** What slot s does in step t; past the end of a stuck run, the stuck step repeats forever. */
export function slotAt(sch: Schedule, t: number, s: number): SlotCell | null {
  if (t < sch.timeline.length) return sch.timeline[t][s];
  if (sch.stuck) return sch.timeline[sch.timeline.length - 1][s];
  return null;
}

/** Tiles (front-order numbers) whose flag is set after the first `steps` steps. */
export function flagsAfter(sch: Schedule, steps: number): Set<number> {
  const out = new Set<number>();
  sch.doneAt.forEach((t, tile) => {
    if (t >= 0 && t < steps) out.add(tile);
  });
  return out;
}

/** The tiles that run in each step, in ascending order, as the book draws its schedules. */
export const rowsOfRunning = (sch: Schedule) =>
  sch.timeline.map((row) => row.filter((c) => c.state === 'run').map((c) => c.tile as number).sort((a, b) => a - b));
