// Sorting networks for chapter 14: the bitonic network of section 14.9 (pp. 345-346),
// and odd-even transposition sort (section 14.2) seen as a network. Sections 14.1 and
// 14.9 run on it.
//
// What it models, exactly: a network is a fixed list of stages; a stage is a list of
// comparators on disjoint positions; a comparator looks at two positions and swaps
// them when they are out of the order it wants. Which comparators exist does not depend
// on the data, only whether each one swaps. The comparison is strict, so two equal
// items are never swapped by one comparator; they can still change their relative
// order, because comparators reach across the array and move other items past them.
// That is why the network is not stable, and why comparing (value, index) pairs makes
// the result deterministic.
//
// Which order: the caller passes `first(a, b)`, true when a must come before b. The
// lessons sort probabilities largest first; `byValue` builds that order, optionally
// breaking ties by token index.
//
// What it leaves out: powers of two only for the bitonic network (`networkSort` pads
// other lengths with sentinels that sort last); no timing, no memory, no threads.
// A stage is drawn as one parallel step; on a GPU it is one barrier interval when the
// values sit in shared memory, or shuffles when they sit in registers.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

/** Compare positions lo and hi; `up` true means lo must end first in the order, false means hi must. */
export interface Comparator {
  lo: number;
  hi: number;
  up: boolean;
}
export type Stage = Comparator[];

export const isPow2 = (n: number) => Number.isInteger(n) && n >= 1 && (n & (n - 1)) === 0;
export const log2 = (n: number) => Math.round(Math.log2(n));

/**
 * Batcher's bitonic sorting network for n a power of two: log2(n)·(log2(n)+1)/2 stages of
 * n/2 comparators. Built the usual way: for block sizes k = 2, 4, ..., n, and for
 * distances j = k/2, ..., 1, position i is compared with i XOR j; the direction flips
 * between neighbouring blocks of size k, so each finished block is a bitonic sequence
 * for the next merge.
 */
export function bitonicStages(n: number): Stage[] {
  if (!isPow2(n)) throw new Error('bitonic network needs a power of two, got ' + n);
  const stages: Stage[] = [];
  for (let k = 2; k <= n; k *= 2)
    for (let j = k / 2; j >= 1; j /= 2) {
      const stage: Stage = [];
      for (let i = 0; i < n; i++) {
        const l = i ^ j;
        if (l > i) stage.push({ lo: i, hi: l, up: (i & k) === 0 });
      }
      stages.push(stage);
    }
  return stages;
}

/**
 * Odd-even transposition sort as a network of n steps (Fig. 14.1 is this network on 8
 * values): even steps compare (0,1), (2,3), ...; odd steps (1,2), (3,4), .... Fixed: it
 * runs all n steps whatever the data, with no flag and no host loop.
 */
export function transpositionStages(n: number, steps = n): Stage[] {
  const stages: Stage[] = [];
  for (let s = 0; s < steps; s++) {
    const stage: Stage = [];
    for (let i = s % 2; i + 1 < n; i += 2) stage.push({ lo: i, hi: i + 1, up: true });
    stages.push(stage);
  }
  return stages;
}

export const comparatorCount = (stages: Stage[]) => stages.reduce((s, st) => s + st.length, 0);
/** n/2 comparators per stage, log2(n)(log2(n)+1)/2 stages: 24 for 8, 80 for 16. */
export const bitonicCount = (n: number) => (n / 2) * ((log2(n) * (log2(n) + 1)) / 2);
/** n steps of alternately floor(n/2) and floor((n-1)/2) comparators: 28 for 8, 120 for 16. */
export const transpositionCount = (n: number) => comparatorCount(transpositionStages(n));

/** An item with a sort key and an identity (a token id): enough to see ties move. */
export interface Item {
  key: number;
  id: number;
}

/**
 * The order the lessons use: larger key first; with `byIndex`, equal keys by increasing id
 * (the (value, index) comparison). Without it, equal keys are "equivalent": neither first.
 */
export function byValue(byIndex: boolean): (a: Item, b: Item) => boolean {
  return byIndex ? (a, b) => a.key > b.key || (a.key === b.key && a.id < b.id) : (a, b) => a.key > b.key;
}

export interface StageTrace<T> {
  /** The values after this stage. */
  after: T[];
  /** Per comparator of the stage, in order: did it swap? */
  swapped: boolean[];
}

export interface NetworkRun<T> {
  input: T[];
  stages: StageTrace<T>[];
  out: T[];
  comparisons: number;
  swaps: number;
}

/** Runs the network. A comparator swaps only when the pair is strictly out of order. */
export function applyNetwork<T>(values: T[], stages: Stage[], first: (a: T, b: T) => boolean): NetworkRun<T> {
  let cur = values.slice();
  const trace: StageTrace<T>[] = [];
  let comparisons = 0;
  let swaps = 0;
  for (const stage of stages) {
    const next = cur.slice();
    const swapped = stage.map((c) => {
      comparisons++;
      const a = next[c.lo];
      const b = next[c.hi];
      const out = c.up ? first(b, a) : first(a, b);
      if (out) {
        next[c.lo] = b;
        next[c.hi] = a;
        swaps++;
      }
      return out;
    });
    trace.push({ after: next, swapped });
    cur = next;
  }
  return { input: values.slice(), stages: trace, out: cur, comparisons, swaps };
}

/**
 * Sorts any number of values with a bitonic network: pads to the next power of two with
 * sentinels that come after everything, runs the network, and drops the sentinels.
 */
export function networkSort<T>(values: T[], first: (a: T, b: T) => boolean): T[] {
  let n = 1;
  while (n < values.length) n *= 2;
  const padded: (T | null)[] = [...values, ...Array(n - values.length).fill(null)];
  const f = (a: T | null, b: T | null) => (a === null ? false : b === null ? true : first(a, b));
  return applyNetwork(padded, bitonicStages(n), f).out.filter((v): v is T => v !== null);
}

/** True when no adjacent pair is strictly out of order. */
export function isSorted<T>(values: T[], first: (a: T, b: T) => boolean): boolean {
  for (let i = 0; i + 1 < values.length; i++) if (first(values[i + 1], values[i])) return false;
  return true;
}

/** One group of equal keys in an output: the ids in output order, and whether they increase. */
export interface TieGroup {
  key: number;
  ids: number[];
  increasing: boolean;
}

/** The groups of equal keys (two or more items) in output order of first appearance. */
export function tieGroups(items: Item[]): TieGroup[] {
  const by = new Map<number, number[]>();
  for (const it of items) {
    const g = by.get(it.key);
    if (g) g.push(it.id);
    else by.set(it.key, [it.id]);
  }
  const out: TieGroup[] = [];
  for (const [key, ids] of by)
    if (ids.length > 1) out.push({ key, ids, increasing: ids.every((v, k) => k === 0 || ids[k - 1] < v) });
  return out;
}

/** The section 14.9 stepper's ties preset: 16 candidates of (probability, token). */
export const TIES16: Item[] = [
  0.05, 0.12, 0.05, 0.08, 0.12, 0.02, 0.08, 0.05, 0.12, 0.03, 0.08, 0.05, 0.02, 0.12, 0.03, 0.05,
].map((key, id) => ({ key, id }));
/** The first eight of them, for N = 8. */
export const TIES8: Item[] = TIES16.slice(0, 8);
