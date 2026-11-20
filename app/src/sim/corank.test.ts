import { describe, expect, it } from 'vitest';
import {
  PRESETS,
  bitLength,
  coRank,
  coRankTrace,
  iRange,
  knownRange,
  mergeByThreads,
  mergeStable,
  parseSorted,
  partition,
  rng,
  sameMerge,
  sortedRandom,
  splitAt,
  type Tagged,
} from './corank';

const A = [1, 7, 8, 9, 10];
const B = [7, 10, 10, 12];
const state = (s: { i: number; j: number; iLow: number; jLow: number }) => [s.i, s.j, s.iLow, s.jLow];

/** The merge with ties sent to B: what the swapped tests are consistent with. */
function mergeTiesToB(a: number[], b: number[]): Tagged[] {
  const out: Tagged[] = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] < b[j]) out.push({ key: a[i], src: 'A', idx: i++ });
    else out.push({ key: b[j], src: 'B', idx: j++ });
  }
  for (; i < a.length; i++) out.push({ key: a[i], src: 'A', idx: i });
  for (; j < b.length; j++) out.push({ key: b[j], src: 'B', idx: j });
  return out;
}
const fromA = (c: Tagged[], k: number) => c.slice(0, k).filter((e) => e.src === 'A').length;

describe('the book\'s running example (Fig. 13.1, p. 304)', () => {
  it('merges to C = 1 7 7 8 9 10 10 10 12, A\'s 7 before B\'s 7 and A\'s 10 before B\'s 10s', () => {
    const C = mergeStable(A, B);
    expect(C.map((e) => e.key)).toEqual([1, 7, 7, 8, 9, 10, 10, 10, 12]);
    expect(C.map((e) => e.src + e.idx).join(' ')).toBe('A0 A1 B0 A2 A3 A4 B1 B2 B3');
  });
  it('co-ranks for k = 0..9 are 0 1 2 2 3 4 5 5 5 5 (ch13-facts E1, recomputed here)', () => {
    expect(Array.from({ length: 10 }, (_, k) => coRank(k, A, B))).toEqual([0, 1, 2, 2, 3, 4, 5, 5, 5, 5]);
  });
  it('C[4] has co-ranks (3, 1) and C[6] has (5, 1), not Fig. 13.3(b)\'s (4, 2) (p. 308)', () => {
    expect([coRank(4, A, B), 4 - coRank(4, A, B)]).toEqual([3, 1]);
    expect([coRank(6, A, B), 6 - coRank(6, A, B)]).toEqual([5, 1]);
    expect(splitAt(6, 4, A, B).valid).toBe(false);
    expect(splitAt(6, 4, A, B).sortedKeys).toBe(true); // the figure's split sorts the keys...
    expect(splitAt(6, 4, A, B).stable).toBe(false); // ...but puts B's 10 ahead of A's
  });
  it('k = 3 runs (3,0,0,0) -> (1,2,0,0) -> (2,1,1,0), as Figs. 13.6 to 13.8', () => {
    const t = coRankTrace(3, A, B);
    expect(t.steps.map(state)).toEqual([[3, 0, 0, 0], [1, 2, 0, 0], [2, 1, 1, 0]]);
    expect(t.steps.map((s) => s.test)).toEqual(['i-too-high', 'j-too-high', 'done']);
    expect(t.steps.map((s) => s.delta)).toEqual([2, 1, undefined]);
    expect([t.steps[0].t1.left, t.steps[0].t1.right]).toEqual([8, 7]); // A[2] = 8 > B[0] = 7
    expect([t.steps[1].t1.left, t.steps[1].t1.right, t.steps[1].t1.holds]).toEqual([1, 10, false]); // A[0] = 1 <= B[2] = 10
    expect([t.steps[1].t2!.left, t.steps[1].t2!.right]).toEqual([10, 7]); // B[1] = 10 >= A[1] = 7
    expect([t.steps[2].t1.left, t.steps[2].t1.right, t.steps[2].t2!.left, t.steps[2].t2!.right]).toEqual([7, 10, 7, 8]);
    expect(t.result).toBe(2);
    expect(t.moves).toBe(2);
    expect(t.candidates).toBe(4);
    expect(t.bound).toBe(2);
    expect(t.steps.map((s) => s.line)).toEqual([9, 14, 20]);
  });
  it('k = 3: the window of candidates goes from 0..3 to 0..2 to 2..2', () => {
    const t = coRankTrace(3, A, B);
    expect([knownRange(t, -1), knownRange(t, 0), knownRange(t, 1), knownRange(t, 2)]).toEqual([[0, 3], [0, 2], [2, 2], [2, 2]]);
  });
  it('exercise 2: k = 6 starts at (5, 1, 2, 1) and stops at once; k = 9 gives (5, 4)', () => {
    const t = coRankTrace(6, A, B);
    expect(state(t.init)).toEqual([5, 1, 2, 1]);
    expect(t.moves).toBe(0);
    expect(t.steps[0].t1.holds).toBe(false); // A[4] = 10 > B[1] = 10 is false
    expect(t.steps[0].t2!.inBounds).toBe(false); // i < m fails: no read
    expect(t.steps[0].reads).toEqual([{ arr: 'A', index: 4 }, { arr: 'B', index: 1 }]);
    const end = coRankTrace(9, A, B);
    expect([end.result, 9 - end.result, end.moves]).toEqual([5, 4, 0]);
    expect(end.steps[0].reads).toEqual([]);
  });
  it('exercise 1: C[8] has co-ranks (5, 3)', () => {
    expect([coRank(8, A, B), 8 - coRank(8, A, B)]).toEqual([5, 3]);
  });
  it('Fig. 13.4: at k = 4, i = 2 and i = 4 give an unsorted C, and the failing test points the right way', () => {
    const lo = splitAt(4, 2, A, B);
    expect([lo.valid, lo.failing, lo.sortedKeys]).toEqual([false, 'j-too-high', false]);
    expect(lo.prefix.map((e) => e.key)).toEqual([1, 7, 7, 10]); // B[1] = 10 in thread 0's share
    expect(lo.suffix[0].key).toBe(8); // while A[2] = 8 goes to thread 1
    const hi = splitAt(4, 4, A, B);
    expect([hi.valid, hi.failing, hi.sortedKeys]).toEqual([false, 'i-too-high', false]);
    expect([hi.t1.left, hi.t1.right]).toEqual([9, 7]); // A[3] = 9 > B[0] = 7
    expect(splitAt(4, 3, A, B).valid).toBe(true);
    expect(iRange(4, 5, 4)).toEqual([0, 4]);
  });
  it('three threads, three outputs each: co-ranks (0,0) (2,1) (5,1) (5,4), and thread 1 merges (&A[2], 3, &B[1], 0, &C[3]) (p. 314)', () => {
    const s = partition(A, B, 3, 3);
    expect(s.map((x) => [x.iCurr, x.jCurr])).toEqual([[0, 0], [2, 1], [5, 1]]);
    expect([s[2].iNext, s[2].jNext]).toEqual([5, 4]);
    const t1 = s[1];
    expect([t1.iCurr, t1.iNext - t1.iCurr, t1.jCurr, t1.jNext - t1.jCurr, t1.kCurr]).toEqual([2, 3, 1, 0, 3]);
    expect(sameMerge(mergeByThreads(A, B, s), mergeStable(A, B))).toBe(true);
  });
});

describe('the tie rule', () => {
  it('swapping > and >= gives co-ranks 0 1 1 2 3 4 4 4 5 5 on the running example', () => {
    expect(Array.from({ length: 10 }, (_, k) => coRank(k, A, B, 'swapped'))).toEqual([0, 1, 1, 2, 3, 4, 4, 4, 5, 5]);
  });
  it('with the swapped tests, three threads write sorted keys but put B\'s 10 ahead of A\'s 10', () => {
    const C = mergeByThreads(A, B, partition(A, B, 3, 3, 'swapped'));
    expect(C.map((e) => e!.key)).toEqual([1, 7, 7, 8, 9, 10, 10, 10, 12]);
    expect(sameMerge(C, mergeStable(A, B))).toBe(false);
    expect(C.map((e) => e!.src + e!.idx).join(' ')).toBe('A0 A1 B0 A2 A3 B1 A4 B2 B3');
  });
  it('all keys equal: the co-rank is min(k, m) (every tie goes to A)', () => {
    const a = [5, 5, 5, 5, 5];
    const b = [5, 5, 5, 5];
    for (let k = 0; k <= 9; k++) expect(coRank(k, a, b)).toBe(Math.min(k, 5));
  });
});

describe('against a stable merge, on many random inputs with heavy duplicates', () => {
  const r = rng(13);
  const cases: [number[], number[]][] = [];
  for (let c = 0; c < 3000; c++) {
    const m = Math.floor(r() * 13);
    const n = Math.floor(r() * 13);
    cases.push([sortedRandom(r, m, 0, 6), sortedRandom(r, n, 0, 6)]);
  }
  it('book rule: co_rank(k) = the number of A elements among the first k outputs of the stable merge', () => {
    let queries = 0;
    for (const [a, b] of cases) {
      const C = mergeStable(a, b);
      for (let k = 0; k <= a.length + b.length; k++) {
        expect(coRank(k, a, b)).toBe(fromA(C, k));
        queries++;
      }
    }
    expect(queries).toBeGreaterThan(30000);
  });
  it('swapped rule: the same, for the merge that sends ties to B', () => {
    for (const [a, b] of cases) {
      const C = mergeTiesToB(a, b);
      for (let k = 0; k <= a.length + b.length; k++) expect(coRank(k, a, b, 'swapped')).toBe(fromA(C, k));
    }
  });
  it('no read is ever outside an array, and i + j = k at every step', () => {
    for (const [a, b] of cases)
      for (let k = 0; k <= a.length + b.length; k++)
        for (const rule of ['book', 'swapped'] as const) {
          const t = coRankTrace(k, a, b, rule);
          for (const s of t.steps) {
            expect(s.i + s.j).toBe(k);
            for (const rd of s.reads) {
              expect(rd.index).toBeGreaterThanOrEqual(0);
              expect(rd.index).toBeLessThan(rd.arr === 'A' ? a.length : b.length);
            }
          }
        }
  });
  it('any number of threads writes the stable merge with the book rule', () => {
    for (const [a, b] of cases.slice(0, 500))
      for (const per of [1, 2, 3, 5]) {
        const threads = Math.ceil((a.length + b.length) / per) || 1;
        expect(sameMerge(mergeByThreads(a, b, partition(a, b, threads, per)), mergeStable(a, b))).toBe(true);
      }
  });
});

describe('cost: at most ceil(log2(w + 1)) moves, and the bound is attained', () => {
  it('bitLength(w) is ceil(log2(w + 1))', () => {
    expect([0, 1, 2, 3, 4, 7, 8, 1048576].map(bitLength)).toEqual([0, 1, 2, 2, 3, 3, 4, 21]);
  });
  it('exhaustively for m, n up to 5 with keys 0..2: never more, and for every w some input needs exactly that many', () => {
    const arrays: number[][] = [[]];
    const grow = (prefix: number[], len: number) => {
      if (prefix.length === len) return arrays.push(prefix);
      for (let v = prefix.length ? prefix[prefix.length - 1] : 0; v <= 2; v++) grow([...prefix, v], len);
    };
    for (let len = 1; len <= 5; len++) grow([], len);
    const worst = new Map<number, number>();
    for (const a of arrays)
      for (const b of arrays)
        for (let k = 0; k <= a.length + b.length; k++) {
          const t = coRankTrace(k, a, b);
          expect(t.moves).toBeLessThanOrEqual(t.bound);
          const w = t.candidates - 1;
          worst.set(w, Math.max(worst.get(w) ?? 0, t.moves));
        }
    for (const [w, mv] of worst) expect(mv).toBe(bitLength(w));
    expect(worst.get(5)).toBe(3);
  });
  it('random inputs up to 40 long never exceed the bound', () => {
    const r = rng(7);
    for (let c = 0; c < 400; c++) {
      const a = sortedRandom(r, Math.floor(r() * 40), 0, 9);
      const b = sortedRandom(r, Math.floor(r() * 40), 0, 9);
      for (let k = 0; k <= a.length + b.length; k++) {
        const t = coRankTrace(k, a, b);
        expect(t.moves).toBeLessThanOrEqual(t.bound);
      }
    }
  });
  it('m = n = k = 1,048,576 with all of B below A: 21 moves, the question\'s answer', () => {
    const N = 1 << 20;
    const a = new Array<number>(N).fill(1);
    const b = new Array<number>(N).fill(0);
    const t = coRankTrace(N, a, b);
    expect(t.candidates).toBe(N + 1);
    expect(t.bound).toBe(21);
    expect(t.moves).toBe(21);
    expect(t.result).toBe(0);
  });
});

describe('presets and input', () => {
  it('every preset is sorted and reproducible from its seed', () => {
    for (const p of PRESETS) {
      const x = p.make(42);
      const y = p.make(42);
      expect(x).toEqual(y);
      for (const arr of [x.A, x.B]) expect(arr.every((v, q) => q === 0 || arr[q - 1] <= v)).toBe(true);
    }
    const dup = PRESETS.find((p) => p.id === 'duplicates')!.make(1);
    expect([...dup.A, ...dup.B].every((v) => v >= 0 && v <= 3)).toBe(true);
  });
  it('parses sorted lists and rejects unsorted, oversized or non-integer input', () => {
    expect(parseSorted('1, 7, 8,9 10')).toEqual({ ok: true, values: [1, 7, 8, 9, 10] });
    expect(parseSorted('')).toEqual({ ok: true, values: [] });
    expect(parseSorted('3, 2').ok).toBe(false);
    expect(parseSorted('1, x').ok).toBe(false);
    expect(parseSorted('1.5').ok).toBe(false);
    expect(parseSorted(Array.from({ length: 17 }, (_, q) => q).join(',')).ok).toBe(false);
  });
});
