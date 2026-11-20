import { describe, expect, it } from 'vitest';
import {
  BOOK_KEYS,
  bin,
  firstDisorder,
  lsdSort,
  parallelPass,
  parseKeys,
  partitionPass,
  promptKey,
  randomKeys,
  reverseBits,
  shifts,
  sortedLow,
  writeRuns,
} from './radixsort';

const row = (s: string) => s.split(' ').map((b) => parseInt(b, 2));
const show = (ks: number[]) => ks.map((k) => bin(k, 4)).join(' ');
const nums = (s: string) => s.split(' ').map(Number);

// Rows as recorded in notes/ch14.md (Figs. 14.4, 14.6, 14.10).
const INPUT = '1100 0011 0110 1001 1111 1000 0101 1010 1001 0110 1011 1101 0100 1010 0111 0000';
const IT1 = '1100 0110 1000 1010 0110 0100 1010 0000 0011 1001 1111 0101 1001 1011 1101 0111';
const IT2 = '1100 1000 0100 0000 1001 0101 1001 1101 0110 1010 0110 1010 0011 1111 1011 0111';
const IT3 = '1000 0000 1001 1001 1010 1010 0011 1011 1100 0100 0101 1101 0110 0110 1111 0111';
const IT4 = '0000 0011 0100 0101 0110 0110 0111 1000 1001 1001 1010 1010 1011 1100 1101 1111';

describe("Fig. 14.4: the book's 1-bit LSD radix sort, four passes", () => {
  const run = lsdSort(BOOK_KEYS, 4, 1);
  it('the input is the shared sixteen keys', () => {
    expect(show(BOOK_KEYS)).toBe(INPUT);
  });
  it('every pass output matches the figure', () => {
    expect(run.passes.map((p) => show(p.output))).toEqual([IT1, IT2, IT3, IT4]);
    expect(run.passes.map((p) => p.shift)).toEqual([0, 1, 2, 3]);
  });
  it('passes 1 to 3 split 8/8 and pass 4 splits 7 zeros from 9 ones', () => {
    expect(run.passes.map((p) => p.counts)).toEqual([[8, 8], [8, 8], [8, 8], [7, 9]]);
  });
  it('after pass k the list is sorted by its low k bits, and no more', () => {
    expect(run.passes.map((p) => p.sortedBy)).toEqual([1, 2, 3, 4]);
    expect(run.sorted).toBe(true);
    expect(run.firstDisorder).toBe(-1);
  });
  it('question q-pass2: after pass 2 the first 0110 is at position 8', () => {
    expect(run.passes[1].output.indexOf(0b0110)).toBe(8);
    // the four keys before it end in 00, the four after them in 01
    expect(run.passes[1].output.slice(0, 8).map((k) => k & 3)).toEqual([0, 0, 0, 0, 1, 1, 1, 1]);
  });
  it('identities follow the keys: the scene pair 0101 (input 6) and 0100 (input 12)', () => {
    // input order 0101 before 0100; pass 1 puts 0100 first and every later pass keeps it there
    const pos = (id: number) => run.passes.map((p) => p.outIds.indexOf(id));
    expect(pos(6)).toEqual([11, 5, 10, 3]);
    expect(pos(12)).toEqual([5, 2, 9, 2]);
    for (const p of run.passes) for (let j = 0; j < 16; j++) expect(p.output[j]).toBe(BOOK_KEYS[p.outIds[j]]);
  });
});

describe('Fig. 14.10: the 2-bit radix', () => {
  const run = lsdSort(BOOK_KEYS, 4, 2);
  it('two passes, low two bits then high two bits, each as in the figure', () => {
    expect(run.passes.map((p) => p.shift)).toEqual([0, 2]);
    expect(run.passes.map((p) => show(p.output))).toEqual([IT2, IT4]);
  });
  it('pass 1 makes four buckets of four; pass 2 buckets of 2, 5, 6, 3', () => {
    expect(run.passes[0].counts).toEqual([4, 4, 4, 4]);
    expect(run.passes[1].counts).toEqual([2, 5, 6, 3]);
    expect(run.passes.map((p) => p.sortedBy)).toEqual([2, 4]);
  });
  it('half the passes of the 1-bit radix', () => {
    expect(run.passes.length * 2).toBe(lsdSort(BOOK_KEYS, 4, 1).passes.length);
  });
});

describe('stability and order of passes', () => {
  it('an unstable pass 4 leaves the top-bit-1 keys in reverse order of their low three bits', () => {
    const run = lsdSort(BOOK_KEYS, 4, 1, { unstablePass: 4 });
    expect(show(run.result)).toBe('0000 0011 0100 0101 0110 0110 0111 1111 1101 1100 1011 1010 1010 1001 1001 1000');
    expect(run.sorted).toBe(false);
    expect(run.firstDisorder).toBe(7); // 1111 before 1101
    const ones = run.result.slice(7);
    for (let k = 0; k + 1 < ones.length; k++) expect(ones[k] & 7).toBeGreaterThanOrEqual(ones[k + 1] & 7);
  });
  it('making pass 2, 3 or 4 unstable breaks the book example', () => {
    for (let p = 2; p <= 4; p++) expect(lsdSort(BOOK_KEYS, 4, 1, { unstablePass: p }).sorted).toBe(false);
  });
  it('an unstable pass 1 has no earlier order to lose: the keys come out sorted, but the two 1001s swap', () => {
    const run = lsdSort(BOOK_KEYS, 4, 1, { unstablePass: 1 });
    expect(run.sorted).toBe(true);
    expect(run.stable).toBe(false);
    expect(run.result[run.firstTieSwap]).toBe(0b1001);
    expect(run.resultIds.filter((id) => BOOK_KEYS[id] === 0b1001)).toEqual([8, 3]);
    expect(lsdSort(BOOK_KEYS, 4, 1).resultIds.filter((id) => BOOK_KEYS[id] === 0b1001)).toEqual([3, 8]);
    expect(lsdSort(BOOK_KEYS, 4, 1).stable).toBe(true);
  });
  it('an unstable pass loses what the earlier passes established', () => {
    const run = lsdSort(BOOK_KEYS, 4, 1, { unstablePass: 2 });
    expect(run.passes[0].sortedBy).toBe(1);
    expect(run.passes[1].sortedBy).toBe(0);
  });
  it('question q-msb: four stable passes from the top sort by the bit-reversed key', () => {
    const run = lsdSort(BOOK_KEYS, 4, 1, { order: 'msd' });
    expect(run.passes.map((p) => p.shift)).toEqual([3, 2, 1, 0]);
    expect(run.sorted).toBe(false);
    const rev = run.result.map((k) => reverseBits(k, 4));
    expect(firstDisorder(rev)).toBe(-1);
    // and the last pass's bit (bit 0) is the primary key: all even keys come first
    expect(run.result.slice(0, 8).every((k) => (k & 1) === 0)).toBe(true);
  });
  it('shifts in pass order', () => {
    expect(shifts(4, 1, 'lsd')).toEqual([0, 1, 2, 3]);
    expect(shifts(4, 1, 'msd')).toEqual([3, 2, 1, 0]);
    expect(shifts(8, 2, 'lsd')).toEqual([0, 2, 4, 6]);
  });
  it('random keys of every width sort correctly with stable passes', () => {
    for (const seed of [1, 2, 3, 99]) {
      const keys = randomKeys(40, 8, seed);
      const want = keys.slice().sort((a, b) => a - b);
      for (const r of [1, 2, 3, 4]) expect(lsdSort(keys, 8, r).result).toEqual(want);
    }
  });
});

describe('helpers', () => {
  it('sortedLow and firstDisorder', () => {
    expect(sortedLow(row(IT2), 4)).toBe(2);
    expect(sortedLow(row(IT4), 4)).toBe(4);
    expect(sortedLow(row(INPUT), 4)).toBe(0);
    expect(firstDisorder([1, 2, 2, 1])).toBe(2);
  });
  it('promptKey picks the middle key of the top bucket', () => {
    const p = partitionPass(BOOK_KEYS, 0, 1);
    const i = promptKey(p);
    expect(p.dst[i]).toBe(8 + 4); // ones start at 8; the fifth of eight
    expect(BOOK_KEYS[i]).toBe(0b1001);
    expect(i).toBe(8);
  });
  it('parseKeys reads binary or decimal and rejects bad input', () => {
    expect(parseKeys('0101 3 1111 0 1 2 3 4', 4, 8)).toEqual({ keys: [5, 3, 15, 0, 1, 2, 3, 4] });
    expect('error' in parseKeys('1 2 3', 4, 8)).toBe(true);
    expect('error' in parseKeys('16 1 1 1 1 1 1 1', 4, 8)).toBe(true);
    expect('error' in parseKeys('x 1 1 1 1 1 1 1', 4, 8)).toBe(true);
  });
});

describe('Figs. 14.5 and 14.6: one parallel pass, one thread per key', () => {
  const p = parallelPass(BOOK_KEYS, 0, 4);
  it('four blocks of four threads; i = blockIdx.x * blockDim.x + threadIdx.x', () => {
    expect(p.threads.map((t) => t.block * 4 + t.thread)).toEqual(p.threads.map((t) => t.i));
    expect(p.threads[13]).toMatchObject({ block: 3, thread: 1 });
  });
  it('bits, ones before and destinations are the figure\'s rows', () => {
    expect(p.bits).toEqual(nums('0 1 0 1 1 0 1 0 1 0 1 1 0 0 1 0'));
    expect(p.threads.map((t) => t.onesBefore)).toEqual(nums('0 0 1 1 2 3 3 4 4 5 5 6 7 7 7 8'));
    expect(p.ones).toBe(8);
    expect(p.scan[16]).toBe(8); // n + 1 entries: entry n is the total
    expect(p.threads.map((t) => t.dst)).toEqual(nums('0 8 1 9 10 2 11 3 12 4 13 14 5 6 15 7'));
  });
  it('the output is pass 1 of Fig. 14.4 and the pass is correct', () => {
    expect(show(p.out as number[])).toBe(IT1);
    expect(p.correct).toBe(true);
    expect(p.collisions).toEqual([]);
    expect(p.holes).toEqual([]);
  });
  it('the formula shows the thread\'s numbers', () => {
    expect(p.threads[1].formula).toBe('16 − 8 + 0 = 8');
    expect(p.threads[2].formula).toBe('2 − 1 = 1');
  });
  it('every bit of the book keys gives a correct pass', () => {
    for (let b = 0; b < 4; b++) expect(parallelPass(BOOK_KEYS, b, 4).correct).toBe(true);
  });
});

describe('the defects of the stepper', () => {
  it('ones use the zero formula: collisions and holes', () => {
    const p = parallelPass(BOOK_KEYS, 0, 4, 'zero-formula');
    expect(p.correct).toBe(false);
    expect(p.collisions.length).toBeGreaterThan(0);
    expect(p.holes.length).toBeGreaterThan(0);
    // thread 1 (a one with no ones before it) and thread 2 (a zero with one before it) both write slot 1
    expect(p.writers[1]).toEqual([1, 2]);
    expect(p.holes).toEqual([8, 9, 10, 11, 12, 13, 14, 15]);
  });
  it('a scan of n entries: every thread reads entry n out of range; the ones\' destinations are unknown', () => {
    const p = parallelPass(BOOK_KEYS, 0, 4, 'scan-n');
    expect(p.scan.length).toBe(16);
    expect(p.oobReads).toHaveLength(16);
    expect(p.unknown).toEqual(p.threads.filter((t) => t.bit === 1).map((t) => t.i));
    expect(p.correct).toBe(false);
  });
  it('the total from the last of n entries works for the book keys only because the last bit is 0 (erratum 4)', () => {
    expect(parallelPass(BOOK_KEYS, 0, 4, 'last-as-total').correct).toBe(true);
    const keys = BOOK_KEYS.slice(0, 15).concat([0b0001]);
    const p = parallelPass(keys, 0, 4, 'last-as-total');
    expect(p.correct).toBe(false);
    expect(p.oobWrites).toEqual([15]); // the last one lands on slot n
    expect(p.holes).toEqual([16 - p.ones]); // the first slot of the one bucket stays empty
  });
  it('an inclusive scan shifts every one by one slot: one write out of range, one hole', () => {
    const p = parallelPass(BOOK_KEYS, 0, 4, 'inclusive');
    expect(p.correct).toBe(false);
    expect(p.oobWrites).toEqual([14]); // the last one, key 0111 at i = 14
    expect(p.holes).toEqual([8]);
    expect(p.threads.filter((t) => t.bit === 0).every((t) => t.dst === parallelPass(BOOK_KEYS, 0, 4).threads[t.i].dst)).toBe(true);
  });
});

describe('questions of section 14.5', () => {
  it('q-dst-one and q-dst-zero: n = 20, 9 ones, index 13 with 6 ones before', () => {
    // a key list built to have those numbers
    const keys = Array(20).fill(0);
    [0, 2, 3, 5, 7, 9, 13, 16, 19].forEach((i) => (keys[i] = 1));
    const p = parallelPass(keys, 0, 4);
    expect(p.ones).toBe(9);
    expect(p.threads[13]).toMatchObject({ onesBefore: 6, dst: 17 });
    const z = keys.slice();
    z[13] = 0;
    z[14] = 1; // keep 9 ones, make index 13 a zero with 6 ones before it
    const q = parallelPass(z, 0, 4);
    expect(q.ones).toBe(9);
    expect(q.threads[13]).toMatchObject({ onesBefore: 6, dst: 7 });
  });
  it('q-runs: a warp\'s zeros land in one run and its ones in another', () => {
    for (const seed of [3, 5, 7, 11]) {
      const keys = randomKeys(64, 8, seed);
      const p = parallelPass(keys, 2, 32);
      for (let w = 0; w < 2; w++) {
        const ts = p.threads.slice(w * 32, w * 32 + 32);
        const zeros = ts.filter((t) => t.bit === 0).map((t) => t.dst!);
        const ones = ts.filter((t) => t.bit === 1).map((t) => t.dst!);
        if (zeros.length) expect(writeRuns(zeros)).toBe(1);
        if (ones.length) expect(writeRuns(ones)).toBe(1);
        if (zeros.length && ones.length) expect(writeRuns([...zeros, ...ones])).toBe(2);
      }
    }
  });
});
