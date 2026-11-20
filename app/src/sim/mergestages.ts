// Merge-sort stage arithmetic for chapter 14, section 14.3 (pp. 333-334, Fig. 14.3), and
// Exercise 4 (p. 347) in outline: sort fixed-length segments, then merge neighbours
// pairwise, stage by stage, until one segment remains, sharing a fixed grid of blocks
// among each stage's merges.
//
// What it models:
//  - Per stage: how many merges, how long each merge's output is, and how the grid's
//    blocks are shared: blocks / merges blocks per merge when there are at least as many
//    blocks as merges (the book's example: eight blocks give 2, 4, 8 per merge), or
//    merges / blocks merges per block when there are fewer.
//  - Block-level split searches: a merge shared by b blocks needs b - 1 co-rank searches
//    to find where each block's share of the output starts in the two inputs (chapter 13,
//    section 13.3); each search takes about log2 of an input's length steps (chapter 13, section 13.4).
//    Chapter 13's kernels search again inside each block, per thread or per tile; that
//    is not counted here.
//  - A real stable merge sort on a few keys, stage by stage, taking from the left run on
//    ties, so the counts stay tied to data.
//
// What it leaves out: time, memory traffic, the cost of a launch; every block is equally
// fast. Lengths are powers of two (the explorer offers only those); an unpaired trailing
// segment, which Exercise 4 would copy, never arises.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

export const pow2 = (k: number) => 2 ** k;
export const log2i = (n: number) => Math.round(Math.log2(n));

export interface PlanInput {
  /** Keys to sort. */
  n: number;
  /** Length of the segments sorted before any merge. */
  segLen: number;
  /** Blocks in the grid. */
  blocks: number;
}

export interface StageRow {
  /** 0 is "sort each segment"; 1, 2, ... are merge stages. */
  stage: number;
  /** Sorted segments when the stage has finished. */
  segments: number;
  /** Merges in this stage (0 for stage 0). */
  merges: number;
  /** Length of one merge's output (stage 0: one segment). */
  elementsPerMerge: number;
  /** Blocks working on one merge (or one segment), when there are at least as many blocks as merges; else 0. */
  blocksPerMerge: number;
  /** Merges (or segments) one block handles, when there are fewer blocks than merges; else 0. */
  mergesPerBlock: number;
  /** Co-rank searches to split the merges among their blocks: merges × (blocks per merge − 1). */
  splitSearches: number;
  /** About this many steps per search, ceil(log2(input length + 1)); chapter 13's model stays within one more. */
  stepsPerSearch: number;
}

export interface Plan extends PlanInput {
  segments: number;
  /** Merge stages: log2(segments). */
  stages: number;
  rows: StageRow[];
}

/** The whole merge tree for n keys, segments of segLen, and a grid of `blocks` blocks. */
export function mergePlan({ n, segLen, blocks }: PlanInput): Plan {
  if (segLen > n) throw new Error('segment longer than the list');
  const segments = n / segLen;
  const stages = log2i(segments);
  const share = (units: number) =>
    blocks >= units ? { blocksPerMerge: blocks / units, mergesPerBlock: 0 } : { blocksPerMerge: 0, mergesPerBlock: units / blocks };
  const rows: StageRow[] = [
    { stage: 0, segments, merges: 0, elementsPerMerge: segLen, ...share(segments), splitSearches: 0, stepsPerSearch: 0 },
  ];
  for (let s = 1; s <= stages; s++) {
    const merges = segments / pow2(s);
    const elementsPerMerge = segLen * pow2(s);
    const sh = share(merges);
    rows.push({
      stage: s,
      segments: merges,
      merges,
      elementsPerMerge,
      ...sh,
      splitSearches: sh.blocksPerMerge > 1 ? merges * (sh.blocksPerMerge - 1) : 0,
      stepsPerSearch: Math.ceil(Math.log2(elementsPerMerge / 2 + 1)),
    });
  }
  return { n, segLen, blocks, segments, stages, rows };
}

/** Which blocks work on merge m of a stage (one block may serve several merges). */
export function blocksOf(row: StageRow, m: number): number[] {
  if (row.blocksPerMerge > 0) return Array.from({ length: row.blocksPerMerge }, (_, k) => m * row.blocksPerMerge + k);
  return [Math.floor(m / row.mergesPerBlock)];
}

// ---------------------------------------------------------------- a real merge sort
export interface Key {
  key: number;
  /** Position in the input: shows whether equal keys kept their order. */
  id: number;
}

/** On equal keys, which run gives up its element first. 'left' is chapter 13's A[i] <= B[j]: stable. */
export type TieRule = 'left' | 'right';

/** Merge two sorted runs; with the default rule, ties come from the left run first (stable). */
export function mergeRuns(a: Key[], b: Key[], tie: TieRule = 'left'): Key[] {
  const out: Key[] = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) out.push((tie === 'left' ? a[i].key <= b[j].key : a[i].key < b[j].key) ? a[i++] : b[j++]);
  return out.concat(a.slice(i), b.slice(j));
}

export interface Snapshot {
  stage: number;
  /** Length of each sorted segment after this stage. */
  segLen: number;
  items: Key[];
}

/** Stage 0 sorts each segment (stably); each later stage merges neighbouring segments. */
export function mergeSortStages(keys: number[], segLen: number, tie: TieRule = 'left'): Snapshot[] {
  let items: Key[] = keys.map((key, id) => ({ key, id }));
  const runs: Key[][] = [];
  for (let s = 0; s < items.length; s += segLen)
    runs.push(items.slice(s, s + segLen).sort((x, y) => x.key - y.key || x.id - y.id));
  items = runs.flat();
  const out: Snapshot[] = [{ stage: 0, segLen, items }];
  let len = segLen;
  let stage = 0;
  while (len < keys.length) {
    const next: Key[] = [];
    for (let s = 0; s < items.length; s += 2 * len) next.push(...mergeRuns(items.slice(s, s + len), items.slice(s + len, s + 2 * len), tie));
    items = next;
    len *= 2;
    stage++;
    out.push({ stage, segLen: len, items });
  }
  return out;
}

/** The sixteen keys of the section 14.3 panel: several repeats, so stability shows. */
export const PANEL_KEYS = [5, 2, 7, 2, 9, 4, 5, 1, 8, 3, 5, 7, 2, 6, 4, 9];
/** Segments of two: eight segments, as in Fig. 14.3's eight blocks. */
export const PANEL_SEG = 2;

/** The explorer's choices. */
export const N_CHOICES = [10, 12, 14, 16, 18, 20, 22, 24].map(pow2);
export const BLOCK_CHOICES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
