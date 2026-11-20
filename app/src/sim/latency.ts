// Chapter 4, section 4.6: why an SM holds more warps than it can execute. One
// execution unit, R resident warps, and a long wait after every few instructions.
// It runs the explorer "How many warps hide a wait?", the prediction above it and
// the narrated animation "Hiding a wait".
//
// Sources for the rules (the lesson cites them where they are used):
//  - the book, section 4.6, pp. 82 to 83: an SM executes only a subset of its warps
//    at any instant; a warp waiting on a long-latency operation is not selected; among
//    the warps that are ready a priority logic picks one; keeping the execution units
//    busy this way is latency tolerance;
//  - the book, p. 84 and its sidebar: choosing another warp inserts no idle time,
//    because every resident warp's state stays in hardware;
//  - the book, pp. 84 to 85: the example device holds 2048 threads per SM for 128
//    streaming processors, 16 to 1;
//  - notes/today/ch04-facts.md A11: NVIDIA confirms the mechanism and does not
//    document how a ready warp is chosen; B8: AMD documents an oldest-first fetch
//    arbiter and a round-robin issue arbiter.
//
// The model. Every warp repeats the same loop: `work` instructions, one per cycle it
// is given an issue slot, the last of which starts a memory operation; then `latency`
// cycles during which it cannot issue anything. Each cycle the unit issues from at
// most `slots` ready warps, the one that has been ready longest first (ties go to the
// lower warp number).
//
// What this leaves out (the lesson says so beside the explorer):
//  - the wait is the same fixed number of cycles every time, and any number of warps
//    may wait at once: memory has unlimited bandwidth here;
//  - an instruction takes one cycle, so the pipelined arithmetic and branch
//    latencies the book also mentions (pp. 83 to 84) are not modelled;
//  - "oldest first" is this model's choice. NVIDIA does not document its policy;
//  - every warp runs the same loop, and none ever finishes;
//  - registers, and therefore how many warps fit at all: that is section 4.7.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

/** What one warp is doing in one cycle. */
export type Cell = 'exec' | 'wait' | 'ready';

export interface LatencyConfig {
  /** R: warps resident on the unit. */
  warps: number;
  /** c: instructions a warp issues between two memory operations (at least 1). */
  work: number;
  /** L: cycles from starting a memory operation until the warp can issue again. */
  latency: number;
  /** Instructions the unit can issue per cycle, each from a different warp. */
  slots: number;
}

export interface Timeline {
  config: LatencyConfig;
  cycles: number;
  /** rows[w][t]: what warp w does in cycle t. */
  rows: Cell[][];
  /** chosen[t]: the warps that issued in cycle t, in the order they were picked. */
  chosen: number[][];
}

const whole = (v: number, min: number): number => (Number.isFinite(v) ? Math.max(min, Math.floor(v)) : min);

/** The configuration with every field made a usable whole number. */
export const normalise = (c: LatencyConfig): LatencyConfig => ({
  warps: whole(c.warps, 1),
  work: whole(c.work, 1),
  latency: whole(c.latency, 0),
  slots: whole(c.slots, 1),
});

/** Cycles one warp needs for one trip round its loop when it never has to queue: c + L. */
export const period = (c: LatencyConfig): number => {
  const n = normalise(c);
  return n.work + n.latency;
};

/** Runs the model for `cycles` cycles from a cold start: every warp ready at cycle 0. */
export function simulate(config: LatencyConfig, cycles: number): Timeline {
  const cfg = normalise(config);
  const n = whole(cycles, 0);
  const rows: Cell[][] = Array.from({ length: cfg.warps }, () => new Array<Cell>(n));
  const chosen: number[][] = [];
  const left = new Array<number>(cfg.warps).fill(cfg.work); // instructions left before the next memory operation
  const readyAt = new Array<number>(cfg.warps).fill(0); // first cycle the warp can issue again
  for (let t = 0; t < n; t++) {
    const ready: number[] = [];
    for (let w = 0; w < cfg.warps; w++) if (readyAt[w] <= t) ready.push(w);
    // Oldest first: the warp that became ready earliest. readyAt does not change during
    // a burst, so a warp that has started its instructions keeps its place until it waits.
    ready.sort((a, b) => readyAt[a] - readyAt[b] || a - b);
    const pick = ready.slice(0, cfg.slots);
    const picked = new Set(pick);
    for (let w = 0; w < cfg.warps; w++) rows[w][t] = picked.has(w) ? 'exec' : readyAt[w] <= t ? 'ready' : 'wait';
    for (const w of pick) {
      left[w] -= 1;
      if (left[w] === 0) {
        left[w] = cfg.work;
        readyAt[w] = t + 1 + cfg.latency;
      }
    }
    chosen.push(pick);
  }
  return { config: cfg, cycles: n, rows, chosen };
}

/** Issue slots used in cycles [from, to) as a fraction of the slots there were. */
export function busyFraction(tl: Timeline, from = 0, to = tl.cycles): number {
  const a = Math.max(0, from);
  const b = Math.min(tl.cycles, to);
  if (b <= a) return 0;
  let used = 0;
  for (let t = a; t < b; t++) used += tl.chosen[t].length;
  return used / ((b - a) * tl.config.slots);
}

/** Cycles in [from, to) in which the unit issued nothing at all. */
export function idleCycles(tl: Timeline, from = 0, to = tl.cycles): number {
  let idle = 0;
  for (let t = Math.max(0, from); t < Math.min(tl.cycles, to); t++) if (tl.chosen[t].length === 0) idle += 1;
  return idle;
}

/** Start-up cycles left out of the measurement, and the cycles it then covers. */
export function measureWindow(config: LatencyConfig): { from: number; to: number } {
  const cfg = normalise(config);
  const p = cfg.work + cfg.latency;
  // Long enough for the cold start (every warp ready at once) to have passed: the first
  // round of bursts takes R * c / slots cycles, then three more trips for good measure.
  // After that, a whole number of trips.
  const from = Math.ceil((cfg.warps * cfg.work) / cfg.slots) + 3 * p;
  return { from, to: from + 4 * p };
}

/**
 * Utilisation of the execution unit once the start-up has passed: the fraction of
 * issue slots that carry an instruction, measured on the simulated timeline.
 */
export function utilisation(config: LatencyConfig): number {
  const w = measureWindow(config);
  return busyFraction(simulate(config, w.to), w.from, w.to);
}

/**
 * The estimate from counting alone. One warp can use a slot for c of every c + L
 * cycles, so R warps can fill R * c / (c + L) slots, and the unit has `slots`.
 * With one slot this is min(1, R * c / (c + L)).
 */
export function estimate(config: LatencyConfig): number {
  const cfg = normalise(config);
  return Math.min(1, (cfg.warps * cfg.work) / (cfg.slots * (cfg.work + cfg.latency)));
}

/** The fewest warps that keep every slot busy: slots * (c + L) / c, rounded up. */
export function warpsNeeded(work: number, latency: number, slots = 1): number {
  const cfg = normalise({ warps: 1, work, latency, slots });
  return Math.ceil((cfg.slots * (cfg.work + cfg.latency)) / cfg.work);
}

// ---------------------------------------------------------------- what a switch that costs something would do
/** What a single-slot unit does in one cycle when changing warps is not free. */
export interface UnitCell {
  kind: 'exec' | 'switch' | 'idle';
  /** The warp it issues from, or is switching to. */
  warp: number | null;
}

/**
 * The same model with one issue slot and a price on changing warps: before the unit
 * can issue from a warp other than the one it last issued from, it spends `cost`
 * cycles issuing nothing (one thread's state saved, another's loaded: the book's
 * sidebar on context switching, p. 84). With cost 0 this is `simulate`. The cost is
 * an illustrative number; the book gives none.
 */
export function simulateSwitching(config: LatencyConfig, cycles: number, cost: number): UnitCell[] {
  const cfg = normalise({ ...config, slots: 1 });
  const k = whole(cost, 0);
  const total = whole(cycles, 0);
  const left = new Array<number>(cfg.warps).fill(cfg.work);
  const readyAt = new Array<number>(cfg.warps).fill(0);
  const out: UnitCell[] = [];
  let loaded: number | null = null; // the warp whose state the unit holds
  let switching = 0; // cycles of the current switch still to pay
  for (let t = 0; t < total; t++) {
    if (switching > 0) {
      switching -= 1;
      out.push({ kind: 'switch', warp: loaded });
      continue;
    }
    let pick: number | null = loaded != null && readyAt[loaded] <= t ? loaded : null;
    if (pick == null)
      for (let w = 0; w < cfg.warps; w++)
        if (readyAt[w] <= t && (pick == null || readyAt[w] < readyAt[pick])) pick = w;
    if (pick == null) {
      out.push({ kind: 'idle', warp: null });
      continue;
    }
    if (pick !== loaded && loaded != null && k > 0) {
      loaded = pick;
      switching = k - 1;
      out.push({ kind: 'switch', warp: pick });
      continue;
    }
    loaded = pick;
    out.push({ kind: 'exec', warp: pick });
    left[pick] -= 1;
    if (left[pick] === 0) {
      left[pick] = cfg.work;
      readyAt[pick] = t + 1 + cfg.latency;
    }
  }
  return out;
}

/** The share of cycles in [from, to) of each kind. */
export function unitShares(unit: UnitCell[], from = 0, to = unit.length): Record<UnitCell['kind'], number> {
  const out = { exec: 0, switch: 0, idle: 0 };
  const a = Math.max(0, from);
  const b = Math.min(unit.length, to);
  for (let t = a; t < b; t++) out[unit[t].kind] += 1;
  const nCycles = Math.max(1, b - a);
  return { exec: out.exec / nCycles, switch: out.switch / nCycles, idle: out.idle / nCycles };
}

/** A warp's row as runs of the same state, for drawing. */
export interface Run {
  state: Cell;
  start: number;
  length: number;
}
export function runs(row: Cell[]): Run[] {
  const out: Run[] = [];
  for (let t = 0; t < row.length; t++) {
    const last = out[out.length - 1];
    if (last && last.state === row[t]) last.length += 1;
    else out.push({ state: row[t], start: t, length: 1 });
  }
  return out;
}

/** How many cycles of each kind a warp spent in [from, to). */
export function tally(row: Cell[], from = 0, to = row.length): Record<Cell, number> {
  const out: Record<Cell, number> = { exec: 0, wait: 0, ready: 0 };
  for (let t = Math.max(0, from); t < Math.min(row.length, to); t++) out[row[t]] += 1;
  return out;
}

/** The explorer's two presets and the prediction's case. The cycle counts are illustrative, not measurements. */
export const PRESETS = {
  /** Little arithmetic per load: a matrix-vector product, an embedding lookup. */
  loadHeavy: { work: 2, latency: 30 },
  /** A long stretch of arithmetic on registers between loads. */
  computeHeavy: { work: 30, latency: 30 },
} as const;

/** The book's example device, pp. 84 to 85: 2048 threads per SM for 128 streaming processors. */
export const BOOK_OVERSUBSCRIPTION = { threadsPerSm: 2048, coresPerSm: 128, ratio: 16 } as const;
