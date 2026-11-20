// Shared-memory bank conflicts of chapter 15's strip reads (book §15.6, pp. 361–363,
// Fig. 15.11; notes/ch15.md), and of the project's transposed W-tile store (ours).
//
// What it models: one warp's access to shared memory, counted with src/sim/banks.ts:
// bank = 32-bit word index mod the number of banks; distinct words in one bank are served
// one after another; lanes on the same word share it, a broadcast (ch15-facts A2,
// ch06-facts A12). 32 banks is NVIDIA's documented structure (A2). 64 banks is AMD's
// count on CDNA 4 and RDNA 2 to 4 (ch15-facts B2), used here with NVIDIA's rule: AMD
// documents 16 lanes per cycle but not how a 64-lane access is checked for conflicts,
// so every 64-bank figure is a model, not AMD's cost. Apple documents no banks at all
// (ch15-facts C2): there is no Metal model.
//
// The accesses (mappings from src/sim/gemmremap.ts, width 32 or 64):
//   A strip: a register of the A strip is A_s[row·ld + i] for each row the thread's tile
//     covers; ld is A_s's leading dimension (8 as printed, 9 with the padding column).
//       §15.3/§15.4 layout (Fig. 15.9): rows tRow + p, p = 0..7.
//       §15.5 layout: rows wRow + qr·32 + lRow + r, with p = 4·qr + r.
//   B strip: B_s[i·128 + col]; B_s keeps its 128-float rows (p. 363).
//       §15.3/§15.4 layout: columns tCol + p.   §15.5 layout: wCol + qc·QC + lCol + c, p = 4·qc + c.
//   Each register of a strip is one 32-bit load instruction, which is how the labs read
//   them. A compiler may merge loads into 64- or 128-bit ones; NVIDIA documents no bank
//   rule for those (ch15-facts A2), so only a profiler shows that case.
//   W-tile store (ours, the project's linear layer, lab 15.6): W is stored (out, in)
//   row-major, so a 128 x 8 tile of W (n rows, k columns) becomes the 8 x 128 B tile,
//   stored transposed as B_s[k·pitch + n]. Scalar loads use loadTile's mapping (Fig. 15.5
//   with height 128, width 8): thread t takes n = 32·s + t/8, k = t % 8 for s = 0..3.
//   float4 loads (the notes' Exercise 1 sketch): thread t takes n = t/2,
//   k = 4·(t % 2) + j, and writes the four values with four scalar stores, j = 0..3.
//
// Left out: time, AMD's 16-lane cycles, 64- and 128-bit shared accesses, alignment of
// the float4 global load (it needs K a multiple of 4), and anything about Metal.

import { warpBanks, type BankResult } from './banks';
import { BK, BN, THREADS, coarseThread, remapGeometry, remapThread, type Layout, type Width } from './gemmremap';

export type Banks = 32 | 64;
export type Strip = 'A' | 'B';
export type WMapping = 'scalar' | 'float4';

/** One lane's word, with where it sits in the shared array. */
export interface LaneWord {
  lane: number;
  t: number;
  /** Row and column of the shared array (A_s: row, i; B_s: i, column; W store: k, n). */
  row: number;
  col: number;
  word: number;
}

export interface Accessed extends BankResult {
  words: LaneWord[];
  /** The conflict degree: passes the busiest bank needs (1 = conflict-free). */
  degree: number;
}

const count = (words: LaneWord[], banks: Banks): Accessed => {
  const r = warpBanks(words.map((w) => w.word * 4), 4, banks);
  return { ...r, words, degree: r.passes };
};

export interface StripSpec {
  layout: Layout;
  strip: Strip;
  /** A_s leading dimension (ignored for B, whose rows stay 128 floats). */
  ld: number;
  banks: Banks;
  W: Width;
  warp: number;
  /** Step along the inner dimension, 0 to 7. */
  i: number;
  /** Which register of the strip, 0 to 7: a_r[p] or b_r[p] in the thread's logical tile. */
  p: number;
}

/** One strip-read instruction of one warp. */
export function stripRead(s: StripSpec): Accessed {
  const words: LaneWord[] = [];
  for (let lane = 0; lane < s.W; lane++) {
    const t = s.warp * s.W + lane;
    let row: number;
    let col: number;
    if (s.layout === 'coarse') {
      const ct = coarseThread(t);
      if (s.strip === 'A') [row, col] = [ct.tRow + s.p, s.i];
      else [row, col] = [s.i, ct.tCol + s.p];
    } else {
      const rt = remapThread(t, s.W);
      const g = remapGeometry(s.W);
      const hi = Math.floor(s.p / 4);
      const lo = s.p % 4;
      if (s.strip === 'A') [row, col] = [rt.wRow + hi * g.QR + rt.lRow + lo, s.i];
      else [row, col] = [s.i, rt.wCol + hi * g.QC + rt.lCol + lo];
    }
    const pitch = s.strip === 'A' ? s.ld : BN;
    words.push({ lane, t, row, col, word: row * pitch + col });
  }
  return count(words, s.banks);
}

export interface Worst {
  degree: number;
  at: { warp: number; i: number; p: number };
}

/** The worst conflict over every warp, every step i and every register p of the strip. */
export function stripWorst(s: Omit<StripSpec, 'warp' | 'i' | 'p'>): Worst {
  let best: Worst = { degree: 0, at: { warp: 0, i: 0, p: 0 } };
  for (let warp = 0; warp < THREADS / s.W; warp++)
    for (let i = 0; i < BK; i++)
      for (let p = 0; p < 8; p++) {
        const d = stripRead({ ...s, warp, i, p }).degree;
        if (d > best.degree) best = { degree: d, at: { warp, i, p } };
      }
  return best;
}

/** Leading dimensions in [lo, hi] for which the A strip read never conflicts. */
export function freeLds(layout: Layout, banks: Banks, W: Width, lo = 8, hi = 17): number[] {
  const out: number[] = [];
  for (let ld = lo; ld <= hi; ld++) if (stripWorst({ layout, strip: 'A', ld, banks, W }).degree === 1) out.push(ld);
  return out;
}

// ---------------------------------------------------------------- the project's W-tile store

export interface WStoreSpec {
  mapping: WMapping;
  /** Floats per row of B_s: 128 plus any padding. */
  pitch: number;
  banks: Banks;
  W: Width;
  warp: number;
  /** Scalar mapping: the sub-tile s, 0 to 3. float4 mapping: which of the four values, j, 0 to 3. */
  step: number;
}

/** One store instruction of the transposed W tile into B_s. */
export function wStore(s: WStoreSpec): Accessed {
  const words: LaneWord[] = [];
  for (let lane = 0; lane < s.W; lane++) {
    const t = s.warp * s.W + lane;
    const [n, k] = s.mapping === 'scalar' ? [32 * s.step + Math.floor(t / 8), t % 8] : [Math.floor(t / 2), 4 * (t % 2) + s.step];
    words.push({ lane, t, row: k, col: n, word: k * s.pitch + n });
  }
  return count(words, s.banks);
}

/** The worst over every warp and every step. `at.p` is the step. */
export function wStoreWorst(s: Omit<WStoreSpec, 'warp' | 'step'>): Worst {
  let best: Worst = { degree: 0, at: { warp: 0, i: 0, p: 0 } };
  for (let warp = 0; warp < THREADS / s.W; warp++)
    for (let step = 0; step < 4; step++) {
      const d = wStore({ ...s, warp, step }).degree;
      if (d > best.degree) best = { degree: d, at: { warp, i: 0, p: step } };
    }
  return best;
}

/** B_s pitches in [lo, hi] for which the W-tile store never conflicts. */
export function freePitches(mapping: WMapping, banks: Banks, W: Width, lo = 128, hi = 148): number[] {
  const out: number[] = [];
  for (let pitch = lo; pitch <= hi; pitch++) if (wStoreWorst({ mapping, pitch, banks, W }).degree === 1) out.push(pitch);
  return out;
}

// ---------------------------------------------------------------- the padded tile's sizes

/** Bytes of shared memory for A_s (bM rows of ld floats) and B_s (bK rows of 128 floats). */
export function sharedBytes(ld: number, bM = 128): { A: number; B: number; total: number } {
  const A = bM * ld * 4;
  const B = BK * BN * 4;
  return { A, B, total: A + B };
}

// ---------------------------------------------------------------- the line the book leaves out

export interface MisRead {
  /** The A strip element the thread means to read: row tRow + r, column i. */
  r: number;
  i: number;
  /** The word it reads with the unpadded offset tRow·bK and the padded leading dimension. */
  word: number;
  /** Where that word is in the padded tile. */
  row: number;
  col: number;
  /** Column bK of the padded tile: never written by loadTile. */
  padding: boolean;
  right: boolean;
}

/**
 * §15.3 layout, A_s padded to leading dimension bK + 1 and passed to mm, but the mm call's
 * start offset left as &A_s[tRow*bK] (notes, "Unclear in the PDF" item 4): what thread
 * row ty reads.
 */
export function wrongOffsetReads(ty: number, bK = BK): MisRead[] {
  const ld = bK + 1;
  const tRow = ty * 8;
  const out: MisRead[] = [];
  for (let r = 0; r < 8; r++)
    for (let i = 0; i < bK; i++) {
      const word = tRow * bK + r * ld + i;
      const row = Math.floor(word / ld);
      const col = word % ld;
      out.push({ r, i, word, row, col, padding: col === bK, right: word === (tRow + r) * ld + i });
    }
  return out;
}
