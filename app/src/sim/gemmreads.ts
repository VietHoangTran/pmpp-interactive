// One thread's mm() for one tile pair, in the two loop orders of section 15.4
// (book Fig. 15.6, p. 356, and Fig. 15.9, p. 359; notes/ch15.md).
//
//   'v1' (Fig. 15.6):  for row, for col, for i:  c[row][col] += a[row*lda + i] * b[i*ldb + col]
//                      every operand is read from shared memory, once per use.
//   'v2' (Fig. 15.9):  for i: a_r[row] = a[row*lda + i] (tM reads), b_r[col] = b[i*ldb + col]
//                      (tN reads), then for row, for col: c[row][col] += a_r[row] * b_r[col].
//
// What it models: the order of the multiply-adds, which shared-memory slots each one
// reads, and how often; the values, with small integers, so both orders can be checked
// to give the same tile. The counts per thread per tile pair are the notes' [inferred]
// figures: 2·tM·tN·bK shared reads against bK·(tM + tN), 1024 against 128 for 8 x 8 x 8.
//
// What it leaves out: time, latency, bank conflicts (section 15.6), and what a compiler
// does with the source: it counts reads as the source writes them. With integers the two
// orders agree exactly; with floats the additions into each c[row][col] happen in the same
// order (i = 0, 1, ...) in both loops, so the results agree bit for bit as well.

export interface Shape {
  tM: number;
  tN: number;
  bK: number;
}
export const BOOK_SHAPE: Shape = { tM: 8, tN: 8, bK: 8 };
export type Order = 'v1' | 'v2';

/** The thread-level input tiles: a is tM x bK (rows of A_s), b is bK x tN (columns of B_s), row-major. */
export interface Inputs {
  a: number[];
  b: number[];
}

/** Small integers, so a reader can check a sum by hand. Deterministic. */
export function smallInputs(s: Shape): Inputs {
  const a = Array.from({ length: s.tM * s.bK }, (_, k) => 1 + ((Math.floor(k / s.bK) + 2 * (k % s.bK)) % 3));
  const b = Array.from({ length: s.bK * s.tN }, (_, k) => (2 * Math.floor(k / s.tN) + (k % s.tN)) % 4);
  return { a, b };
}

export interface SharedRead {
  arr: 'A' | 'B';
  /** Position inside the thread-level input tile: A (row, i), B (i, col). */
  r: number;
  c: number;
}

export type Op =
  | { kind: 'fma'; row: number; col: number; i: number; reads: SharedRead[]; value: number }
  | { kind: 'loadA'; i: number; row: number; read: SharedRead; value: number }
  | { kind: 'loadB'; i: number; col: number; read: SharedRead; value: number };

export interface Trace {
  shape: Shape;
  order: Order;
  ops: Op[];
  /** The finished thread tile, tM x tN, row-major. */
  C: number[];
  sharedReads: number;
  /** Reads of the register strips a_r and b_r (zero in Fig. 15.6's order). */
  registerReads: number;
  fmas: number;
}

export function run(s: Shape, x: Inputs, order: Order): Trace {
  const C = new Array<number>(s.tM * s.tN).fill(0);
  const ops: Op[] = [];
  let sharedReads = 0;
  let registerReads = 0;
  if (order === 'v1') {
    for (let row = 0; row < s.tM; row++)
      for (let col = 0; col < s.tN; col++)
        for (let i = 0; i < s.bK; i++) {
          C[row * s.tN + col] += x.a[row * s.bK + i] * x.b[i * s.tN + col];
          sharedReads += 2;
          ops.push({ kind: 'fma', row, col, i, reads: [{ arr: 'A', r: row, c: i }, { arr: 'B', r: i, c: col }], value: C[row * s.tN + col] });
        }
  } else {
    for (let i = 0; i < s.bK; i++) {
      const aR: number[] = [];
      const bR: number[] = [];
      for (let row = 0; row < s.tM; row++) {
        aR[row] = x.a[row * s.bK + i];
        sharedReads++;
        ops.push({ kind: 'loadA', i, row, read: { arr: 'A', r: row, c: i }, value: aR[row] });
      }
      for (let col = 0; col < s.tN; col++) {
        bR[col] = x.b[i * s.tN + col];
        sharedReads++;
        ops.push({ kind: 'loadB', i, col, read: { arr: 'B', r: i, c: col }, value: bR[col] });
      }
      for (let row = 0; row < s.tM; row++)
        for (let col = 0; col < s.tN; col++) {
          C[row * s.tN + col] += aR[row] * bR[col];
          registerReads += 2;
          ops.push({ kind: 'fma', row, col, i, reads: [], value: C[row * s.tN + col] });
        }
    }
  }
  return { shape: s, order, ops, C, sharedReads, registerReads, fmas: s.tM * s.tN * s.bK };
}

// ---------------------------------------------------------------- closed forms
export const sharedReadsV1 = (s: Shape) => 2 * s.tM * s.tN * s.bK;
export const sharedReadsV2 = (s: Shape) => s.bK * (s.tM + s.tN);
export const fmas = (s: Shape) => s.tM * s.tN * s.bK;
export const readsPerFma = (s: Shape, order: Order) => (order === 'v1' ? sharedReadsV1(s) : sharedReadsV2(s)) / fmas(s);
/** Registers for the accumulators plus, in Fig. 15.9's order, the two strips (the book's 64 + 8 + 8 = 80, p. 363). */
export const registersFor = (s: Shape, order: Order) => s.tM * s.tN + (order === 'v2' ? s.tM + s.tN : 0);
/** How many FMAs use one strip value while it sits in a register: tN for an a_r value, tM for a b_r value. */
export const reuse = (s: Shape) => ({ a: s.tN, b: s.tM });

// ---------------------------------------------------------------- where a trace stands after k multiply-adds
export interface State {
  /** Multiply-adds done. */
  fmas: number;
  /** Ops executed (loads included): the trace prefix this state is the end of. */
  ops: number;
  C: number[];
  /** How many multiply-adds each element of C has received so far (0: not touched yet). */
  updates: number[];
  sharedReads: number;
  /** Reads so far of each slot of the thread's A rows (tM x bK) and B columns (bK x tN). */
  aReads: number[];
  bReads: number[];
  /** Fig. 15.9's strips as they stand (null before the first load). */
  aR: (number | null)[];
  bR: (number | null)[];
  /** The ops that happened in this step (the last FMA and, in v2, the strip loads just before it). */
  step: Op[];
}

/** How many ops of the trace, loads included, run up to and including its k-th multiply-add. */
export function opsForFmas(tr: Trace, k: number): number {
  if (k <= 0) return 0;
  let done = 0;
  for (let n = 0; n < tr.ops.length; n++) if (tr.ops[n].kind === 'fma' && ++done === k) return n + 1;
  return tr.ops.length;
}

/**
 * The trace after its first `n` ops. `step` holds the ops since the previous
 * multiply-add: the multiply-add just done and, in Fig. 15.9's order, the strip loads
 * that came just before it.
 */
export function stateAfterOps(tr: Trace, n: number): State {
  const s = tr.shape;
  const C = new Array<number>(s.tM * s.tN).fill(0);
  const updates = new Array<number>(s.tM * s.tN).fill(0);
  const aReads = new Array<number>(s.tM * s.bK).fill(0);
  const bReads = new Array<number>(s.bK * s.tN).fill(0);
  const aR: (number | null)[] = new Array(s.tM).fill(null);
  const bR: (number | null)[] = new Array(s.tN).fill(null);
  const end = Math.max(0, Math.min(n, tr.ops.length));
  let done = 0;
  let reads = 0;
  let stepStart = 0;
  const count = (r: SharedRead) => {
    reads++;
    if (r.arr === 'A') aReads[r.r * s.bK + r.c]++;
    else bReads[r.r * s.tN + r.c]++;
  };
  for (let j = 0; j < end; j++) {
    const op = tr.ops[j];
    if (op.kind === 'fma') {
      C[op.row * s.tN + op.col] = op.value;
      updates[op.row * s.tN + op.col]++;
      op.reads.forEach(count);
      done++;
      if (j < end - 1) stepStart = j + 1;
    } else {
      count(op.read);
      if (op.kind === 'loadA') aR[op.row] = op.value;
      else bR[op.col] = op.value;
    }
  }
  return { fmas: done, ops: end, C, updates, sharedReads: reads, aReads, bReads, aR, bR, step: tr.ops.slice(stepStart, end) };
}

/**
 * The trace after its first `k` multiply-adds. In Fig. 15.9's order a strip is loaded
 * just before the first multiply-add that uses it, so the loads are counted with that one.
 */
export const stateAt = (tr: Trace, k: number): State => stateAfterOps(tr, opsForFmas(tr, k));

/** The multiply-add count at the end of step i of Fig. 15.9's order (after i + 1 rank-1 updates). */
export const afterStrip = (s: Shape, i: number) => (i + 1) * s.tM * s.tN;
