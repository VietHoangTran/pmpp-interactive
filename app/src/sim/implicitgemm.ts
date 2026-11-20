// Fig. 19.11 (book p. 471), the tiled matrix multiplication with implicit
// unfolding, run block by block on small shapes.
//
// What it models: the kernel's values. Each block owns a T × T tile of
// Y = F · B (rows = output maps, columns = output pixels, grid z = sample). In
// every phase each thread loads one F element into Fds and gathers one B element
// straight from X with Eq. 19.5 into Bds; after the barrier each thread adds T
// products to its running sum; after the loop it stores the sum. Because both
// barriers are in place, the order of threads inside a phase cannot change any
// value, so a phase is computed as a whole.
//
// Variants, toggled independently:
//   - phase count: the printed (C·K·K) / T, which truncates, or the ceiling;
//   - bounds checks: none (as printed; the book leaves them out, p. 470), or
//     zero fill on the F load and the X gather and a guarded store;
//   - a planted bug: the input width W used in place of W_out = W − K + 1 as the
//     gather's divisor and modulus.
// Without bounds checks a load can fall outside its array. Such a read is
// counted, not executed: its value is undefined (NaN here), which is what
// AddressSanitizer stops on in the CPU emulator. A store from a thread outside Y's
// M × P range either lands on another output's slot (a stray write: on a GPU the
// final value then depends on which store lands last, so the slot is marked
// wrong whatever this run's order) or past the end of Y (counted, not executed).
//
// Also here: one warp's F-load, X-gather and Y-store addresses as runs of
// consecutive elements, and the shared-memory bank of each Fds/Bds access under
// NVIDIA's documented model (32 banks of 4 bytes, threads reading the same word
// do not conflict: notes/today/infra-facts.md A10, ch06-facts A12) for
// Exercise 4; and requested global-access totals for direct, explicit and
// implicit unfolding.
//
// Left out: timing, caches, coalescing hardware (runs are counted, not
// transactions), more than one sample in the erratum runs (grid z = 1), and any
// effect of an out-of-range access other than counting it.

import { warpBanks, type BankResult } from './banks';
import { BOOK_F, BOOK_SHAPE, BOOK_X, ckk, convDirect, makeData, outDims, pixels, preset, type ConvShape } from './unfold';

export interface KShape extends ConvShape {
  /** TILE_WIDTH. */
  T: number;
}

export interface Variant {
  /** ceil(C·K·K / T) phases instead of the printed truncation. */
  ceil: boolean;
  /** Zero fill on both loads and a guarded store. */
  bounds: boolean;
  /** Planted bug: W instead of W − K + 1 as the gather's divisor. */
  bug: boolean;
}
export const PRINTED: Variant = { ceil: false, bounds: false, bug: false };
export const CORRECTED: Variant = { ceil: true, bounds: true, bug: false };

export const phasesOf = (s: KShape, v: Pick<Variant, 'ceil'>) => (v.ceil ? Math.ceil(ckk(s) / s.T) : Math.floor(ckk(s) / s.T));
export const gridOf = (s: KShape) => ({ x: Math.ceil(pixels(s) / s.T), y: Math.ceil(s.M / s.T), z: s.N });

export interface Data {
  X: number[];
  F: number[];
}

/** The book's numbers for the book's shape; otherwise small positive integers, so any dropped term changes a sum. */
export function dataFor(s: ConvShape): Data {
  if (s.C === BOOK_SHAPE.C && s.H === BOOK_SHAPE.H && s.W === BOOK_SHAPE.W && s.K === BOOK_SHAPE.K && s.M === BOOK_SHAPE.M && s.N === 1)
    return { X: BOOK_X, F: BOOK_F };
  return { X: makeData(s.N * s.C * s.H * s.W, 1903 + s.C * 7 + s.H), F: makeData(s.M * ckk(s), 4711 + s.M) };
}

// ---------------------------------------------------------------- one thread's loads

export interface FLoad {
  /** Element of F read, or null when the bounds check zero-fills. */
  idx: number | null;
  oob: boolean;
  value: number;
}
export interface Gather {
  u: number;
  v: number;
  /** The X element (c, h, w) the gather computes, before any range test. */
  c: number;
  h: number;
  w: number;
  /** Flat element of X read, or null when the bounds check zero-fills. */
  idx: number | null;
  oob: boolean;
  value: number;
}

function fLoad(s: KShape, vr: Variant, F: readonly number[], Row: number, ph: number, tx: number): FLoad {
  const R = ckk(s);
  const col = ph * s.T + tx;
  if (vr.bounds && !(Row < s.M && col < R)) return { idx: null, oob: false, value: 0 };
  const idx = Row * R + col; // line 23
  const oob = idx < 0 || idx >= F.length;
  return { idx, oob, value: oob ? NaN : F[idx] };
}

function gather(s: KShape, vr: Variant, X: readonly number[], bz: number, ph: number, ty: number, Col: number): Gather {
  const { Wout } = outDims(s);
  const R = ckk(s);
  const P = pixels(s);
  const KK = s.K * s.K;
  const u = ph * s.T + ty; // lines 25–26
  const v = Col;
  const div = vr.bug ? s.W : Wout;
  const c = Math.floor(u / KK);
  const h = Math.floor((u % KK) / s.K) + Math.floor(v / div);
  const w = ((u % KK) % s.K) + (v % div);
  if (vr.bounds && !(u < R && v < P)) return { u, v, c, h, w, idx: null, oob: false, value: 0 };
  const idx = bz * s.C * s.H * s.W + c * s.H * s.W + h * s.W + w; // line 28
  const oob = idx < 0 || idx >= X.length;
  return { u, v, c, h, w, idx, oob, value: oob ? NaN : X[idx] };
}

// ---------------------------------------------------------------- one block

export interface PhaseRecord {
  ph: number;
  /** Per thread, index ty·T + tx. */
  f: FLoad[];
  b: Gather[];
  Fds: number[];
  Bds: number[];
  /** Running sums after this phase, per thread. */
  sums: number[];
}
export type StoreKind = 'own' | 'stray' | 'outside' | 'skipped';
export interface StoreRecord {
  Row: number;
  Col: number;
  idx: number | null;
  kind: StoreKind;
  value: number;
}
export interface BlockRecord {
  bx: number;
  by: number;
  bz: number;
  phases: PhaseRecord[];
  stores: StoreRecord[];
}

export function runBlock(s: KShape, vr: Variant, d: Data, bx: number, by: number, bz = 0): BlockRecord {
  const T = s.T;
  const P = pixels(s);
  const ySize = s.N * s.M * P;
  const n = phasesOf(s, vr);
  const sums = new Array(T * T).fill(0);
  const phases: PhaseRecord[] = [];
  for (let ph = 0; ph < n; ph++) {
    const f: FLoad[] = [];
    const b: Gather[] = [];
    for (let ty = 0; ty < T; ty++)
      for (let tx = 0; tx < T; tx++) {
        f.push(fLoad(s, vr, d.F, by * T + ty, ph, tx));
        b.push(gather(s, vr, d.X, bz, ph, ty, bx * T + tx));
      }
    const Fds = f.map((x) => x.value);
    const Bds = b.map((x) => x.value);
    for (let ty = 0; ty < T; ty++)
      for (let tx = 0; tx < T; tx++) {
        let acc = sums[ty * T + tx];
        for (let k = 0; k < T; k++) acc += Fds[ty * T + k] * Bds[k * T + tx]; // lines 32–33
        sums[ty * T + tx] = acc;
      }
    phases.push({ ph, f, b, Fds, Bds, sums: sums.slice() });
  }
  const stores: StoreRecord[] = [];
  for (let ty = 0; ty < T; ty++)
    for (let tx = 0; tx < T; tx++) {
      const Row = by * T + ty;
      const Col = bx * T + tx;
      const inside = Row < s.M && Col < P;
      const value = sums[ty * T + tx];
      if (vr.bounds && !inside) {
        stores.push({ Row, Col, idx: null, kind: 'skipped', value });
        continue;
      }
      const idx = bz * s.M * P + Row * P + Col; // line 37
      stores.push({ Row, Col, idx, kind: inside ? 'own' : idx < ySize ? 'stray' : 'outside', value });
    }
  return { bx, by, bz, phases, stores };
}

// ---------------------------------------------------------------- the whole grid

export type OutStatus = 'ok' | 'wrong' | 'clobbered';
export interface KernelResult {
  phases: number;
  /** Terms of the C·K·K reduction each output gets: min(phases·T, C·K·K). */
  termsUsed: number;
  terms: number;
  /** Extra products with u ≥ C·K·K: zero-filled with bounds checks, garbage without. */
  extraTerms: number;
  grid: { x: number; y: number; z: number };
  Y: number[];
  ref: number[];
  status: OutStatus[];
  wrong: number;
  /** Reads outside X and outside F, writes past the end of Y, and stores onto another output's slot. */
  oobX: number;
  oobF: number;
  outsideWrites: number;
  strayWrites: number;
  /** Gathers from X and loads of F actually issued (not zero-filled). */
  gathers: number;
  fLoads: number;
  correct: boolean;
}

export function runKernel(s: KShape, vr: Variant, d: Data = dataFor(s)): KernelResult {
  const R = ckk(s);
  const g = gridOf(s);
  const ref = convDirect(s, d.X, d.F);
  const Y = new Array(ref.length).fill(NaN);
  const clobbered = new Set<number>();
  let oobX = 0;
  let oobF = 0;
  let outsideWrites = 0;
  let strayWrites = 0;
  let gathers = 0;
  let fLoads = 0;
  for (let bz = 0; bz < g.z; bz++)
    for (let by = 0; by < g.y; by++)
      for (let bx = 0; bx < g.x; bx++) {
        const blk = runBlock(s, vr, d, bx, by, bz);
        for (const ph of blk.phases) {
          for (const f of ph.f) if (f.idx !== null) { fLoads++; if (f.oob) oobF++; }
          for (const b of ph.b) if (b.idx !== null) { gathers++; if (b.oob) oobX++; }
        }
        for (const st of blk.stores) {
          if (st.kind === 'own') Y[st.idx!] = st.value;
          else if (st.kind === 'stray') { strayWrites++; clobbered.add(st.idx!); }
          else if (st.kind === 'outside') outsideWrites++;
        }
      }
  const status: OutStatus[] = ref.map((r, i) => (clobbered.has(i) ? 'clobbered' : Y[i] === r ? 'ok' : 'wrong'));
  const n = phasesOf(s, vr);
  const wrong = status.filter((x) => x !== 'ok').length;
  return {
    phases: n,
    termsUsed: Math.min(n * s.T, R),
    terms: R,
    extraTerms: Math.max(0, n * s.T - R),
    grid: g,
    Y,
    ref,
    status,
    wrong,
    oobX,
    oobF,
    outsideWrites,
    strayWrites,
    gathers,
    fLoads,
    correct: wrong === 0,
  };
}

// ---------------------------------------------------------------- shapes for the erratum hunt and the warp explorer

export type HuntId = 'book' | 'c1' | 'c3' | 'all16' | 'p49';
export interface HuntShape {
  id: HuntId;
  label: string;
  shape: KShape;
}
export const HUNT_SHAPES: HuntShape[] = [
  { id: 'book', label: "the book's example, T = 2", shape: { ...BOOK_SHAPE, T: 2 } },
  { id: 'c1', label: 'LeNet C1, T = 16', shape: { ...preset('c1').shape, N: 1, T: 16 } },
  { id: 'c3', label: 'LeNet C3, T = 16', shape: { ...preset('c3').shape, N: 1, T: 16 } },
  { id: 'all16', label: 'C·K·K = 64, 8 × 8 output, M = 32, T = 16', shape: { C: 16, H: 9, W: 9, K: 2, M: 32, N: 1, T: 16 } },
  { id: 'p49', label: 'C·K·K = 64, 7 × 7 output, M = 16, T = 16', shape: { C: 16, H: 8, W: 8, K: 2, M: 16, N: 1, T: 16 } },
];
export const huntShape = (id: HuntId) => HUNT_SHAPES.find((h) => h.id === id)!;

/** Which of the three sizes are multiples of the tile: the printed kernel needs all three. */
export function multiples(s: KShape): { M: boolean; P: boolean; CKK: boolean } {
  return { M: s.M % s.T === 0, P: pixels(s) % s.T === 0, CKK: ckk(s) % s.T === 0 };
}

// ---------------------------------------------------------------- one warp (Exercise 4)

/** NVIDIA's documented warp width; Exercise 4 is asked in NVIDIA's model. */
export const NVIDIA_WARP = 32;
/** NVIDIA's documented shared-memory banks (infra-facts A10). */
export const NVIDIA_BANKS = 32;

export interface Lane {
  lane: number;
  tx: number;
  ty: number;
  Row: number;
  Col: number;
  u: number;
  v: number;
  /** Element addresses (in floats) of the three global accesses, null where the bounds check skips the access. */
  f: number | null;
  x: number | null;
  y: number | null;
}
export interface Run {
  start: number;
  length: number;
  firstLane: number;
  lastLane: number;
}
export interface WarpView {
  lanes: Lane[];
  fRuns: Run[];
  xRuns: Run[];
  yRuns: Run[];
  /** The gather's runs for each of the warp's rows of threads (one ty each). */
  xRunsByTy: { ty: number; runs: Run[] }[];
  /** Address jumps between consecutive gather runs of the same ty. */
  xJumps: number[];
  banks: { fdsWrite: BankResult; bdsWrite: BankResult; fdsRead: BankResult; bdsRead: BankResult };
}

/** Runs of consecutive addresses in lane order; lanes with no access are skipped. */
export function laneRuns(addrs: readonly (number | null)[]): Run[] {
  const runs: Run[] = [];
  addrs.forEach((a, lane) => {
    if (a === null) return;
    const last = runs[runs.length - 1];
    if (last && a === last.start + last.length && lane === last.lastLane + 1) {
      last.length++;
      last.lastLane = lane;
    } else runs.push({ start: a, length: 1, firstLane: lane, lastLane: lane });
  });
  return runs;
}

/**
 * The accesses of one warp of block (bx, by), sample 0, in phase ph of the
 * corrected kernel, with the shared reads taken at inner-loop step k.
 */
export function warpView(s: KShape, bx: number, by: number, ph: number, warp: number, k = 0): WarpView {
  const T = s.T;
  const P = pixels(s);
  const R = ckk(s);
  const d: Data = { X: new Array(s.N * s.C * s.H * s.W).fill(0), F: new Array(s.M * R).fill(0) };
  const lanes: Lane[] = [];
  for (let lane = 0; lane < NVIDIA_WARP; lane++) {
    const tid = warp * NVIDIA_WARP + lane;
    if (tid >= T * T) break;
    const ty = Math.floor(tid / T);
    const tx = tid % T;
    const Row = by * T + ty;
    const Col = bx * T + tx;
    const fl = fLoad(s, CORRECTED, d.F, Row, ph, tx);
    const g = gather(s, CORRECTED, d.X, 0, ph, ty, Col);
    lanes.push({ lane, tx, ty, Row, Col, u: g.u, v: g.v, f: fl.idx, x: g.idx, y: Row < s.M && Col < P ? Row * P + Col : null });
  }
  const tys = [...new Set(lanes.map((l) => l.ty))];
  const xRunsByTy = tys.map((ty) => ({ ty, runs: laneRuns(lanes.map((l) => (l.ty === ty ? l.x : null))) }));
  const xJumps = xRunsByTy.flatMap(({ runs }) => runs.slice(1).map((r, i) => r.start - (runs[i].start + runs[i].length - 1)));
  const bytes = (words: number[]) => words.map((w) => w * 4);
  return {
    lanes,
    fRuns: laneRuns(lanes.map((l) => l.f)),
    xRuns: laneRuns(lanes.map((l) => l.x)),
    yRuns: laneRuns(lanes.map((l) => l.y)),
    xRunsByTy,
    xJumps,
    banks: {
      fdsWrite: warpBanks(bytes(lanes.map((l) => l.ty * T + l.tx)), 4, NVIDIA_BANKS),
      bdsWrite: warpBanks(bytes(lanes.map((l) => l.ty * T + l.tx)), 4, NVIDIA_BANKS),
      fdsRead: warpBanks(bytes(lanes.map((l) => l.ty * T + k)), 4, NVIDIA_BANKS),
      bdsRead: warpBanks(bytes(lanes.map((l) => k * T + l.tx)), 4, NVIDIA_BANKS),
    },
  };
}

/** The worst shared-memory passes over every inner step k of one warp (1 = conflict-free). */
export function worstPasses(s: KShape, bx: number, by: number, ph: number, warp: number): number {
  let worst = 0;
  for (let k = 0; k < s.T; k++) {
    const b = warpView(s, bx, by, ph, warp, k).banks;
    worst = Math.max(worst, b.fdsWrite.passes, b.bdsWrite.passes, b.fdsRead.passes, b.bdsRead.passes);
  }
  return worst;
}

// ---------------------------------------------------------------- global accesses, three ways

export interface Traffic {
  /** Fig. 19.7: one X and one F load per term of every output. */
  direct: { x: number; f: number };
  /** Unfold into B in global memory, then a T-tiled GEMM reads it back. */
  explicit: { unfoldReadsX: number; writesB: number; gemmReadsB: number; f: number };
  /** Fig. 19.11 corrected: B tiles gathered from X, never stored. */
  implicit: { gathersX: number; f: number };
}

/** Requested global accesses (in elements, no caches) for one layer and tile width T. */
export function traffic(s: ConvShape, T: number): Traffic {
  const P = pixels(s);
  const R = ckk(s);
  const terms = s.N * s.M * P * R;
  const bEntries = s.N * R * P;
  const rowBlocks = Math.ceil(s.M / T);
  const colBlocks = Math.ceil(P / T);
  return {
    direct: { x: terms, f: terms },
    explicit: { unfoldReadsX: bEntries, writesB: bEntries, gemmReadsB: bEntries * rowBlocks, f: s.N * s.M * R * colBlocks },
    implicit: { gathersX: bEntries * rowBlocks, f: s.N * s.M * R * colBlocks },
  };
}

// ---------------------------------------------------------------- scene frames

/**
 * Frames of "Implicit unfolding, one tile at a time": the book's example with
 * T = 2, block (0, 0), which owns Y values 14, 20 (map 0) and 12, 24 (map 1).
 */
export const IMPLICIT_SCENE = (() => {
  const s: KShape = { ...BOOK_SHAPE, T: 2 };
  const d = dataFor(s);
  const block = runBlock(s, CORRECTED, d, 0, 0);
  const whole = runKernel(s, CORRECTED, d);
  return {
    shape: s,
    block,
    /** The block's final values, by thread (ty·T + tx). */
    finals: block.stores.map((x) => x.value),
    traffic: traffic(s, 2),
    gathers: whole.gathers,
  };
})();
