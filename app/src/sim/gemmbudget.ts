// Chapter 15, section 15.7: the register and shared-memory budget of the
// register-tiled GEMM kernel. It runs the calculator "Register and shared-memory
// budget" and pins every number section 15.7 states (gemmbudget.test.ts).
//
// From the book (notes/ch15.md, section 15.7, p. 363):
//  - the thread's 8 x 8 output tile takes 64 registers and its two register strips
//    8 each: 64 + 8 + 8 = 80 registers before any temporaries;
//  - such kernels end up at the per-thread maximum, 255 registers, on "a modern GPU";
//  - a "typical SM" has 64 K registers, so at most 256 threads fit, one block of 256,
//    which is 12.5 % of the 2048 threads an SM can hold: the reason the block has 256
//    threads;
//  - the input tiles are small because the arithmetic intensity does not depend on k
//    (section 15.2): 128 x 8 floats = 4 KB each.
// From current documentation (notes/today/ch15-facts.md A3, B3, C3, E1; ch04-facts A12,
// A13, B10, C5):
//  - 255 registers per thread and 64 K per SM hold on every compute capability NVIDIA
//    lists; resident threads per SM are 2048, 1536 or 1024 depending on the capability;
//  - NVIDIA allocates registers "in fixed-size chunks" of undocumented size, and AMD per
//    wavefront in granules, so the per-thread product below is an upper bound. The exact
//    count comes from cudaOccupancyMaxActiveBlocksPerMultiprocessor (or the HIP function
//    of the same shape);
//  - AMD: up to 256 vector registers per thread; the register file per CU is documented
//    only as a range, so the reader's own regsPerMultiprocessor is the figure to use;
//  - Metal documents no register count, no register file and no occupancy API.
//
// What it leaves out (the lesson says so beside the calculator): allocation chunks and
// granules, registers the compiler reserves, warp rounding of the block, the block-slot
// limit, shared memory the driver reserves.
//
// Erasable TypeScript only, and no imports: the browser checks import this file
// directly in Node.

/** Bytes per element: the chapter's tiles are float. */
export const FLOAT_BYTES = 4;

export interface Shape {
  /** Thread-level output tile, rows and columns. */
  tM: number;
  tN: number;
  /** Registers for everything else: addresses, counters, the temporaries of the unrolled code. */
  extra: number;
  /** Threads per block. */
  threads: number;
  /** Block-level output tile and the depth of the input tiles. */
  bM: number;
  bN: number;
  bK: number;
  /** Padding columns on A_s (section 15.6 adds one). */
  pad: number;
  /** 1 for the kernel of sections 15.3 to 15.6, 2 for the double-buffered one of 15.8. */
  buffers: 1 | 2;
}

/** The kernel as the book sets it up (pp. 352, 355, 363): 128 x 128 x 8, 256 threads, 8 x 8 each. */
export const BOOK_SHAPE: Shape = { tM: 8, tN: 8, extra: 0, threads: 256, bM: 128, bN: 128, bK: 8, pad: 0, buffers: 1 };

export interface Limits {
  /** 32-bit registers per SM (or CU); 0 = not documented or not reported. */
  regsPerSm: number;
  /** Resident threads per SM; 0 = not known. */
  threadsPerSm: number;
  /** The most registers one thread may have; 0 = not documented. */
  maxRegsPerThread: number;
  /** Shared memory one block may use, bytes; 0 = not known. */
  sharedPerBlock: number;
  /** Shared memory per SM, bytes; 0 = not known. */
  sharedPerSm: number;
}

/**
 * The book's example SM: 65,536 registers, 255 per thread and 2048 threads (ch. 15,
 * p. 363), and 228 KB of shared memory (ch. 5, p. 116, the H100 the book uses
 * throughout). Used only as "the book's example".
 */
export const BOOK_LIMITS: Limits = { regsPerSm: 65536, threadsPerSm: 2048, maxRegsPerThread: 255, sharedPerBlock: 0, sharedPerSm: 228 * 1024 };

/** Per-thread register maxima the vendors document, per backend (ch04-facts A12; ch15-facts B3). */
export const MAX_REGS_PER_THREAD: Record<string, number> = { cuda: 255, hip: 256 };

/** Registers the thread's tiles need at least: the tM x tN outputs plus one A strip and one B strip. */
export const tileRegisters = (tM: number, tN: number) => tM * tN + tM + tN;

/** Bytes of the A tile in shared memory, bM rows of bK + pad floats. */
export const sharedA = (s: Pick<Shape, 'bM' | 'bK' | 'pad'>) => s.bM * (s.bK + s.pad) * FLOAT_BYTES;
/** Bytes of the B tile in shared memory, bK rows of bN floats. */
export const sharedB = (s: Pick<Shape, 'bK' | 'bN'>) => s.bK * s.bN * FLOAT_BYTES;
/** Shared memory one block declares: both tiles, times the number of buffers. */
export const sharedBytes = (s: Pick<Shape, 'bM' | 'bN' | 'bK' | 'pad' | 'buffers'>) => s.buffers * (sharedA(s) + sharedB(s));

/** The most registers per thread that still lets `blocks` blocks of `threads` threads share `regsPerSm`. */
export const regsForBlocks = (regsPerSm: number, threads: number, blocks: number) =>
  regsPerSm > 0 && threads > 0 && blocks > 0 ? Math.floor(regsPerSm / (threads * blocks)) : 0;

/** Occupancy, as a percentage, of `blocks` blocks of `threads` threads on an SM of `threadsPerSm`. */
export const occupancyPct = (blocks: number, threads: number, threadsPerSm: number) =>
  threadsPerSm > 0 ? (100 * blocks * threads) / threadsPerSm : 0;

export type LimitKey = 'registers' | 'threads' | 'shared';

export interface Budget {
  /** tM tN + tM + tN. */
  tileRegs: number;
  /** What the code asks for: tile registers plus the extra ones. */
  wanted: number;
  /** What a thread gets: `wanted`, capped at the per-thread maximum where one is known. */
  regs: number;
  /** More was wanted than one thread may have: the compiler keeps the rest in local memory. */
  spills: boolean;
  /** The per-thread bound: registers per SM divided by registers per thread. Null when registers are unknown. */
  threadBound: number | null;
  /** Whole blocks each limit allows, or null when that limit is unknown. */
  byLimit: Record<LimitKey, number | null>;
  /** Resident blocks: the smallest known limit. Null when neither registers nor thread slots are known. */
  blocks: number | null;
  /** The limits that equal `blocks`. */
  binding: LimitKey[];
  /** Resident threads and occupancy (percent), or null when the thread slots are unknown. */
  resident: number | null;
  occupancy: number | null;
  /** Shared memory of one block, and of each tile in one buffer. */
  shared: number;
  sharedA: number;
  sharedB: number;
  /** One block's tiles fit the per-block limit; null when that limit is unknown. */
  fitsBlock: boolean | null;
  /** threads x tM x tN equals bM x bN: the threads cover the output tile exactly once. */
  covers: boolean;
}

export function budget(s: Shape, l: Limits): Budget {
  const tileRegs = tileRegisters(s.tM, s.tN);
  const wanted = tileRegs + Math.max(0, s.extra);
  const spills = l.maxRegsPerThread > 0 && wanted > l.maxRegsPerThread;
  const regs = spills ? l.maxRegsPerThread : wanted;
  const shared = sharedBytes(s);
  const byLimit: Record<LimitKey, number | null> = {
    registers: l.regsPerSm > 0 && regs > 0 && s.threads > 0 ? Math.floor(l.regsPerSm / (regs * s.threads)) : null,
    threads: l.threadsPerSm > 0 && s.threads > 0 ? Math.floor(l.threadsPerSm / s.threads) : null,
    shared: l.sharedPerSm > 0 && shared > 0 ? Math.floor(l.sharedPerSm / shared) : null,
  };
  const known = (Object.keys(byLimit) as LimitKey[]).filter((k) => byLimit[k] != null);
  const blocks = byLimit.registers == null && byLimit.threads == null ? null : Math.min(...known.map((k) => byLimit[k] as number));
  const binding = blocks == null ? [] : known.filter((k) => byLimit[k] === blocks);
  const resident = blocks != null && l.threadsPerSm > 0 ? blocks * s.threads : null;
  const occupancy = resident != null ? occupancyPct(blocks as number, s.threads, l.threadsPerSm) : null;
  return {
    tileRegs,
    wanted,
    regs,
    spills,
    threadBound: l.regsPerSm > 0 && regs > 0 ? Math.floor(l.regsPerSm / regs) : null,
    byLimit,
    blocks,
    binding,
    resident,
    occupancy,
    shared,
    sharedA: sharedA(s),
    sharedB: sharedB(s),
    fitsBlock: l.sharedPerBlock > 0 ? shared <= l.sharedPerBlock : null,
    covers: s.threads * s.tM * s.tN === s.bM * s.bN,
  };
}
