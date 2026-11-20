// Chapter 6, section 6.3: vector loads and stores in vector addition. It runs the chunk
// planner of section 6.3 (who owns which elements, which accesses are vector and which
// scalar, how many instructions that takes, and whether the base pointer allows a
// vector access at all).
//
// Sources for the rules (the lesson cites them where they are used):
//  - the book, section 6.3, p. 138 and Fig. 6.12: launch a quarter of the threads; thread
//    i owns elements 4i to 4i+3 and moves them with float4 loads and a float4 store;
//  - the book, pp. 138 to 139 and Exercise 4: a partial last chunk falls back to scalar
//    accesses, and threads past the data do nothing;
//  - the book, p. 139: two vector loads replace eight scalar loads, 75 % fewer;
//  - notes/today/ch06-facts.md A11: a vector access needs its address aligned to the
//    vector's size (NVIDIA's developer blog); Nsight Compute's optimal sectors per request
//    for 32-, 64- and 128-bit accesses by a full warp are 4, 8 and 16;
//  - notes/today/ch06-facts.md D4: PyTorch's can_vectorize_up_to picks the widest of 8, 4,
//    2 or 1 elements whose byte size divides the pointer's address. That is PyTorch's
//    practice, not a vendor rule, and the lesson labels it so.
//
// What this leaves out: it counts instructions executed by threads; it says nothing about
// time, occupancy, caches or how a warp's accesses are grouped into transactions.
//
// Erasable TypeScript only: the browser checks import this file directly in Node.

export type Role = 'vector' | 'tail' | 'idle';

export interface ThreadPlan {
  /** Global thread index. */
  thread: number;
  block: number;
  role: Role;
  /** First element the thread owns (meaningless for an idle thread). */
  first: number;
  /** How many elements it handles: vecWidth for a vector chunk, fewer in the tail, 0 when idle. */
  count: number;
}

export interface Plan {
  n: number;
  blockSize: number;
  vecWidth: number;
  /** Chunks of vecWidth elements, the last possibly partial: ceil(n / vecWidth). */
  chunks: number;
  blocks: number;
  /** Threads launched: whole blocks. */
  threads: number;
  /** The thread with the partial last chunk, or null when n divides evenly. */
  tailThread: number | null;
  /** Elements the tail thread handles one at a time. */
  tailCount: number;
  /** Threads of the last block with nothing to do. */
  idle: number;
  roles: ThreadPlan[];
}

const whole = (v: number, min: number) => (Number.isFinite(v) ? Math.max(min, Math.floor(v)) : min);

/** One thread per chunk of vecWidth elements, launched in whole blocks of blockSize. */
export function plan(n: number, blockSize: number, vecWidth: number): Plan {
  const N = whole(n, 0);
  const B = whole(blockSize, 1);
  const w = whole(vecWidth, 1);
  const chunks = Math.ceil(N / w);
  const blocks = Math.ceil(chunks / B);
  const threads = blocks * B;
  const rem = N % w;
  const tailThread = w > 1 && rem !== 0 ? chunks - 1 : null;
  const roles: ThreadPlan[] = [];
  for (let t = 0; t < threads; t++) {
    const first = t * w;
    const role: Role = first >= N ? 'idle' : t === tailThread ? 'tail' : 'vector';
    roles.push({ thread: t, block: Math.floor(t / B), role, first, count: role === 'idle' ? 0 : Math.min(w, N - first) });
  }
  return { n: N, blockSize: B, vecWidth: w, chunks, blocks, threads, tailThread, tailCount: tailThread == null ? 0 : rem, idle: threads - chunks, roles };
}

export interface Instructions {
  /** Load and store instructions executed by all threads together, with vecWidth-wide vectors. */
  loads: number;
  stores: number;
  /** The same work done one float per thread. */
  scalarLoads: number;
  scalarStores: number;
  /** Percentage fewer load instructions than the scalar kernel. */
  fewerLoadsPct: number;
}

/**
 * Vector addition z = x + y on n elements: per full chunk two vector loads and one vector
 * store; per tail element two scalar loads and one scalar store.
 */
export function instructions(n: number, vecWidth: number): Instructions {
  const N = whole(n, 0);
  const w = whole(vecWidth, 1);
  const full = Math.floor(N / w);
  const rem = N - full * w;
  const loads = 2 * full + 2 * rem;
  const stores = full + rem;
  const scalarLoads = 2 * N;
  const scalarStores = N;
  return { loads, stores, scalarLoads, scalarStores, fewerLoadsPct: scalarLoads ? 100 * (1 - loads / scalarLoads) : 0 };
}

/** Whether a vector of vecWidth elements of elemBytes each may be read at this byte address. */
export const aligned = (byteAddress: number, vecWidth: number, elemBytes = 4) =>
  ((byteAddress % (vecWidth * elemBytes)) + vecWidth * elemBytes) % (vecWidth * elemBytes) === 0;

/**
 * PyTorch's run-time choice (MemoryAccess.cuh, can_vectorize_up_to; facts D4): the widest of
 * 8, 4, 2, 1 elements whose byte size divides the address. Here the address is a byte offset
 * from an allocation that is itself aligned far more strictly (cudaMalloc: at least 256 bytes,
 * facts A3), so the offset alone decides.
 */
export function widestVector(baseByteOffset: number, elemBytes: number, widths: readonly number[] = [8, 4, 2, 1]): number {
  for (const w of widths) if (aligned(baseByteOffset, w, elemBytes)) return w;
  return 1;
}

/** Chunks whose vector access would break the alignment rule, for a base pointer offset by this many bytes. */
export function misalignedChunks(p: Plan, baseByteOffset: number, elemBytes = 4): number[] {
  if (p.vecWidth === 1) return [];
  return p.roles.filter((r) => r.role === 'vector' && !aligned(baseByteOffset + r.first * elemBytes, p.vecWidth, elemBytes)).map((r) => r.thread);
}

/** Bytes one warp-wide load instruction moves: one access per lane. */
export const bytesPerWarpLoad = (warpWidth: number, vecWidth: number, elemBytes = 4) => warpWidth * vecWidth * elemBytes;

/**
 * Nsight Compute's optimal sectors per request for a warp of 32 active threads, by the
 * bytes each thread accesses (facts A11): 32-bit 4, 64-bit 8, 128-bit 16. NVIDIA's figures.
 */
export const SECTORS_PER_REQUEST: Readonly<Record<number, number>> = { 4: 4, 8: 8, 16: 16 };

/** Weights of `bits` each that one load of `bytes` carries (the quantised-weights bridge; ours). */
export const weightsPerLoad = (bytes: number, bits: number) => Math.floor((bytes * 8) / bits);
