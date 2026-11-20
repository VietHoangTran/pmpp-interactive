// Chapter 4, section 4.7: how many blocks of a kernel one SM can hold, which of its
// budgets runs out first, and what occupancy that gives. It runs the occupancy
// calculator and the cliff explorer of section 4.7, the exercise presets there, and
// the occupancy readout of section 4.8's launch explorer.
//
// Sources for the rules (the lessons cite them where they are used):
//  - the book, section 4.7, p. 85: occupancy is the warps assigned to an SM over the
//    most it supports; block slots, thread slots, registers and shared memory are
//    partitioned dynamically among blocks; the example device allows 32 blocks,
//    64 warps (2048 threads) and 1024 threads per block;
//  - pp. 85 to 86: the seven worked block sizes; 65,536 registers per SM, so 32 per
//    thread at full occupancy; 31 and 33 registers at 512 threads per block;
//  - section 4.4, p. 75: a block whose size is not a multiple of the warp width has
//    its last warp padded with inactive threads;
//  - the summary, p. 90, states occupancy with threads instead of warps. Both are
//    computed here because they differ when warps are padded;
//  - the book, chapter 5, p. 116: up to 228 KB of shared memory per SM on the example;
//  - exercises 6, 7 and 8, pp. 91 to 92;
//  - notes/today/ch04-facts.md A12 (limits differ across current devices), A13 and
//    A14 (NVIDIA's definition; registers are partitioned among warps and the
//    granularity is not published), B9 and B10 (on HIP maxBlocksPerMultiProcessor is
//    thread slots / wavefront width, not an independent limit; a field can read 0).
//
// What this leaves out (the lessons say so beside the interactives):
//  - registers are counted per thread, threads x registers per thread, which is the
//    book's arithmetic. NVIDIA allocates them per warp and AMD per wave in granules,
//    and neither publishes enough to reproduce the count. For a real kernel the
//    vendor's occupancy function is the authority;
//  - padded lanes are charged a thread slot and no registers;
//  - shared memory the driver reserves, and the sizes it is handed out in;
//  - AMD's second register pool (scalar registers), which no device field reports;
//  - whether the compiler spills registers to stay under a limit.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

export type LimitKey = 'threads' | 'blocks' | 'regs' | 'shared';
export const LIMIT_KEYS: readonly LimitKey[] = ['threads', 'blocks', 'regs', 'shared'];

/** What one SM offers. null means "not reported": that budget is then left out, never guessed. */
export interface Limits {
  /** Resident threads per SM. */
  threadSlots: number | null;
  /** Resident blocks per SM. */
  blockSlots: number | null;
  /**
   * True when the block-slot figure was computed from the thread slots (HIP, facts B9).
   * It is then not a limit of its own and is not counted.
   */
  blockSlotsDerived: boolean;
  /** Registers per SM. */
  regs: number | null;
  /** Shared memory per SM, bytes. */
  shared: number | null;
  /** Threads in a warp. */
  width: number | null;
  /** Threads per block the device accepts at launch. */
  maxThreadsPerBlock: number | null;
}

/** What one block of a kernel asks for. */
export interface Kernel {
  /** Threads per block. */
  threads: number;
  /** Registers per thread; 0 leaves registers out. */
  regs: number;
  /** Shared memory per block, bytes; 0 leaves shared memory out. */
  shared: number;
}

export interface BudgetRow {
  key: LimitKey;
  /** What one SM has, or null when not reported. */
  cap: number | null;
  /** What one block takes. */
  need: number;
  /** Whole blocks this budget allows, or null when it is not counted. */
  fit: number | null;
  /** Why it is not counted. */
  why?: 'not-reported' | 'derived' | 'unused';
}

export interface Occupancy {
  rows: BudgetRow[];
  /** False when the thread slots are unknown or the block has no threads: no occupancy can be stated. */
  usable: boolean;
  /** The warp width used. When the device does not report one, 1 is used and `widthAssumed` is true. */
  width: number;
  widthAssumed: boolean;
  /** Warps one block is cut into, the last one padded if need be. */
  warpsPerBlock: number;
  /** Thread slots one block holds: whole warps. */
  slotsPerBlock: number;
  /** Blocks resident on one SM: the smallest fit among the counted budgets. */
  blocks: number;
  /** The budgets that allow exactly that many blocks. */
  binding: LimitKey[];
  residentThreads: number;
  residentWarps: number;
  /** Warp slots of one SM, when the thread slots are known. */
  warpSlots: number | null;
  /** Occupancy as the book defines it on p. 85 and NVIDIA does: resident warps over warp slots. 0 to 1. */
  byWarps: number;
  /** Occupancy as the summary on p. 90 states it: resident threads over thread slots. 0 to 1. */
  byThreads: number;
  /** Thread slots that no warp holds. */
  idleSlots: number;
  /** Thread slots held by padding lanes that run no thread. */
  paddedLanes: number;
  /** The block is larger than the device accepts at launch. */
  tooLarge: boolean;
}

const pos = (v: number | null | undefined): number | null => (v != null && Number.isFinite(v) && v > 0 ? Math.floor(v) : null);
const nonNeg = (v: number): number => (Number.isFinite(v) && v > 0 ? Math.floor(v) : 0);

/** Blocks each budget allows, the binding budget, and occupancy both ways. */
export function occupancy(limits: Limits, kernel: Kernel): Occupancy {
  const threads = nonNeg(kernel.threads);
  const regs = nonNeg(kernel.regs);
  const shared = nonNeg(kernel.shared);
  const threadSlots = pos(limits.threadSlots);
  const widthAssumed = pos(limits.width) == null;
  const width = pos(limits.width) ?? 1;
  const warpsPerBlock = Math.ceil(threads / width);
  const slotsPerBlock = warpsPerBlock * width;

  const row = (key: LimitKey, cap: number | null, need: number, why?: BudgetRow['why']): BudgetRow => ({
    key,
    cap,
    need,
    fit: why || cap == null || need <= 0 ? null : Math.floor(cap / need),
    why: why ?? (cap == null ? 'not-reported' : need <= 0 ? 'unused' : undefined),
  });
  const rows: BudgetRow[] = [
    row('threads', threadSlots, slotsPerBlock),
    row('blocks', pos(limits.blockSlots), 1, pos(limits.blockSlots) != null && limits.blockSlotsDerived ? 'derived' : undefined),
    row('regs', pos(limits.regs), threads * regs),
    row('shared', pos(limits.shared), shared),
  ];

  const usable = threadSlots != null && threads > 0;
  const counted = rows.filter((r) => r.fit != null);
  const blocks = usable && counted.length ? Math.min(...counted.map((r) => r.fit as number)) : 0;
  const binding = usable ? counted.filter((r) => r.fit === blocks).map((r) => r.key) : [];
  const warpSlots = threadSlots != null ? Math.floor(threadSlots / width) : null;
  const residentWarps = blocks * warpsPerBlock;
  const residentThreads = blocks * threads;
  const maxPerBlock = pos(limits.maxThreadsPerBlock);
  return {
    rows,
    usable,
    width,
    widthAssumed,
    warpsPerBlock,
    slotsPerBlock,
    blocks,
    binding,
    residentThreads,
    residentWarps,
    warpSlots,
    byWarps: usable && warpSlots ? residentWarps / warpSlots : 0,
    byThreads: usable ? residentThreads / (threadSlots as number) : 0,
    idleSlots: usable ? (threadSlots as number) - blocks * slotsPerBlock : 0,
    paddedLanes: blocks * (slotsPerBlock - threads),
    tooLarge: maxPerBlock != null && threads > maxPerBlock,
  };
}

/** A fraction as a percentage with at most one decimal: 0.875 -> 87.5, 2/3 -> 66.7. */
export const pct = (fraction: number): number => Math.round(fraction * 1000) / 10;

/** The most registers per thread that still allow every thread slot to be filled. */
export const regsAtFullOccupancy = (l: Limits): number | null =>
  pos(l.regs) != null && pos(l.threadSlots) != null ? Math.floor((l.regs as number) / (l.threadSlots as number)) : null;

/** The smallest block that can fill the thread slots before the block slots run out. */
export const smallestFillingBlock = (l: Limits): number | null =>
  pos(l.threadSlots) != null && pos(l.blockSlots) != null && !l.blockSlotsDerived
    ? Math.ceil((l.threadSlots as number) / (l.blockSlots as number))
    : null;

/**
 * Exercise 7's question: can an SM hold `blocks` blocks of `threads` threads, and how
 * full is it then? Only block slots and thread slots are asked about.
 */
export function assignment(limits: Limits, blocks: number, threads: number): { possible: boolean; occupancy: number } {
  const o = occupancy({ ...limits, regs: null, shared: null }, { threads, regs: 0, shared: 0 });
  const possible = o.usable && blocks >= 0 && blocks <= o.blocks;
  return { possible, occupancy: possible && limits.threadSlots ? (blocks * threads) / limits.threadSlots : 0 };
}

// ---------------------------------------------------------------- the staircase
export type Axis = 'regs' | 'threads';
export interface ScanPoint {
  x: number;
  blocks: number;
  /** By warps, 0 to 1. */
  occupancy: number;
  byThreads: number;
  binding: LimitKey[];
}

/** Occupancy as one input of the kernel takes each of `values` and the others stay put. */
export function scan(limits: Limits, kernel: Kernel, axis: Axis, values: number[]): ScanPoint[] {
  return values.map((x) => {
    const o = occupancy(limits, axis === 'regs' ? { ...kernel, regs: x } : { ...kernel, threads: x });
    return { x, blocks: o.blocks, occupancy: o.byWarps, byThreads: o.byThreads, binding: o.binding };
  });
}

/** Most registers a thread may use: the same on every compute capability NVIDIA lists (facts A12). */
export const MAX_REGS_PER_THREAD = 255;

/** Registers per thread from 1 up to the per-thread maximum. */
export const regValues = (): number[] => Array.from({ length: MAX_REGS_PER_THREAD }, (_, i) => i + 1);

/** Block sizes worth plotting: whole warps up to the largest block the device accepts. */
export function threadValues(l: Limits): number[] {
  const width = pos(l.width) ?? 1;
  const max = pos(l.maxThreadsPerBlock) ?? pos(l.threadSlots) ?? 1024;
  const out: number[] = [];
  for (let t = width; t <= max; t += width) out.push(t);
  return out;
}

export interface Step {
  /** The first value at which occupancy is `to`. */
  at: number;
  from: number;
  to: number;
}
/** Every place along a scan where occupancy changes between neighbouring values. */
export function steps(points: ScanPoint[]): Step[] {
  const out: Step[] = [];
  for (let i = 1; i < points.length; i++)
    if (points[i].occupancy !== points[i - 1].occupancy)
      out.push({ at: points[i].x, from: points[i - 1].occupancy, to: points[i].occupancy });
  return out;
}

// ---------------------------------------------------------------- devices
const KB = 1024;

/** The book's example device: section 4.7, pp. 85 to 86, and chapter 5, p. 116 for shared memory. */
export const BOOK: Limits = {
  threadSlots: 2048,
  blockSlots: 32,
  blockSlotsDerived: false,
  regs: 65536,
  shared: 228 * KB,
  width: 32,
  maxThreadsPerBlock: 1024,
};

/** The seven block sizes the book works through on its example device, in the book's order (pp. 85 to 86). */
export const BOOK_BLOCK_SIZES = [1024, 512, 256, 128, 64, 32, 768] as const;

/** Exercise 6 (p. 91): an SM that holds 1536 threads and 4 blocks. The warp width is the book's 32. */
export const EX6: Limits = {
  threadSlots: 1536, blockSlots: 4, blockSlotsDerived: false, regs: null, shared: null, width: 32, maxThreadsPerBlock: null,
};
export const EX6_OPTIONS = [128, 256, 512, 1024] as const;

/** Exercise 7 (p. 92): 64 blocks and 2048 threads per SM. */
export const EX7: Limits = {
  threadSlots: 2048, blockSlots: 64, blockSlotsDerived: false, regs: null, shared: null, width: 32, maxThreadsPerBlock: null,
};
/** Its five proposed assignments, (a) to (e): blocks on the SM, threads in each. */
export const EX7_ASSIGNMENTS = [
  { blocks: 8, threads: 128 },
  { blocks: 16, threads: 64 },
  { blocks: 32, threads: 32 },
  { blocks: 64, threads: 32 },
  { blocks: 32, threads: 64 },
] as const;

/** Exercise 8 (p. 92): 2048 threads, 32 blocks and 65,536 registers per SM. */
export const EX8: Limits = {
  threadSlots: 2048, blockSlots: 32, blockSlotsDerived: false, regs: 65536, shared: null, width: 32, maxThreadsPerBlock: null,
};
/** Its three kernels, (a) to (c). */
export const EX8_KERNELS: readonly Kernel[] = [
  { threads: 128, regs: 30, shared: 0 },
  { threads: 32, regs: 29, shared: 0 },
  { threads: 256, regs: 34, shared: 0 },
];

/** The fields of a device report this model reads. */
export const REPORT_FIELDS = {
  threadSlots: 'maxThreadsPerMultiProcessor',
  blockSlots: 'maxBlocksPerMultiProcessor',
  regs: 'regsPerMultiprocessor',
  shared: 'sharedMemPerMultiprocessor',
  width: 'warpSize',
  maxThreadsPerBlock: 'maxThreadsPerBlock',
} as const;

const reportedInt = (v: unknown): number | null => {
  const n = typeof v === 'number' ? v : typeof v === 'string' && v.trim() !== '' ? Number(v) : NaN;
  return Number.isInteger(n) && n > 0 ? n : null;
};

/** True for the CPU emulators' device reports, whose values describe no GPU. */
export const isSynthetic = (backend: string, device: Record<string, unknown>): boolean =>
  backend === 'emu' || backend === 'race' || (typeof device.note === 'string' && device.note.includes('synthetic'));

/**
 * Limits as one of this machine's device reports gives them (reports/runs/<backend>.json).
 * A missing field and a field of 0 both become "not reported": AMD's reference says some
 * of these fields can read 0 (facts B9). On HIP the block-slot figure is marked as derived.
 * Returns null for the emulators (their values are synthetic) and for a report with none
 * of the per-SM fields, which is what a Metal device gives (facts C1, C5).
 */
export function limitsFromReport(device: Record<string, unknown>, backend: string): Limits | null {
  if (isSynthetic(backend, device)) return null;
  const perSm = [REPORT_FIELDS.threadSlots, REPORT_FIELDS.blockSlots, REPORT_FIELDS.regs, REPORT_FIELDS.shared];
  if (!perSm.some((f) => f in device)) return null;
  return {
    threadSlots: reportedInt(device[REPORT_FIELDS.threadSlots]),
    blockSlots: reportedInt(device[REPORT_FIELDS.blockSlots]),
    blockSlotsDerived: backend === 'hip',
    regs: reportedInt(device[REPORT_FIELDS.regs]),
    shared: reportedInt(device[REPORT_FIELDS.shared]),
    width: reportedInt(device[REPORT_FIELDS.width]),
    maxThreadsPerBlock: reportedInt(device[REPORT_FIELDS.maxThreadsPerBlock]),
  };
}

/** The report fields that came back missing or zero, by the name the device uses. */
export function notReported(device: Record<string, unknown>): string[] {
  return Object.values(REPORT_FIELDS).filter((f) => reportedInt(device[f]) == null);
}
