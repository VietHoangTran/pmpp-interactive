import { describe, expect, it } from 'vitest';
import {
  FIG_14_1,
  FIG_14_1_ROWS,
  LARGEST_FIRST,
  PRESETS,
  TRAP,
  bookBound,
  isAscending,
  nLogN,
  overlapStep,
  pairedLaunches,
  parseValues,
  runToSorted,
  schedule,
  step,
  threadsFor,
  walk,
} from './oddeven';

describe('Fig. 14.1 (p. 332)', () => {
  it('reproduces the seven rows exactly, starting with an even step', () => {
    const rows = walk(FIG_14_1, 6);
    expect([rows[0].before, ...rows.map((r) => r.after)]).toEqual(FIG_14_1_ROWS);
  });
  it('even steps use four threads, odd steps three (the last odd pair is out of range)', () => {
    const rows = walk(FIG_14_1, 6);
    expect(rows.map((r) => r.comparisons)).toEqual([4, 3, 4, 3, 4, 3]);
    expect(rows[1].threads.map((t) => [t.t, t.i, t.active])).toEqual([[0, 1, true], [1, 3, true], [2, 5, true], [3, 7, false]]);
  });
  it('the sixth step (odd) makes no change, as the figure says', () => {
    expect(walk(FIG_14_1, 6)[5].swaps).toBe(0);
  });
  it('the book\'s literal rule stops after 6 steps here; ours launches a seventh, quiet, step', () => {
    const book = runToSorted(FIG_14_1, 'book');
    const ours = runToSorted(FIG_14_1, 'ours');
    expect([book.launched, book.sorted]).toEqual([6, true]);
    expect([ours.launched, ours.sorted, ours.sortedAfter]).toEqual([7, true, 5]);
  });
});

describe('the kernel\'s indexing (Fig. 14.2)', () => {
  it('q-threads: n = 10 launches 5 threads; an odd step uses 4 of them', () => {
    expect(threadsFor(10)).toBe(5);
    const r = step([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], 1);
    expect(r.threads.map((t) => t.i)).toEqual([1, 3, 5, 7, 9]);
    expect(r.comparisons).toBe(4);
  });
  it('odd lengths: floor(n/2) threads cover both parities', () => {
    expect(step([3, 2, 1, 0, 5, 4, 6], 0).threads.filter((t) => t.active).map((t) => t.i)).toEqual([0, 2, 4]);
    expect(step([3, 2, 1, 0, 5, 4, 6], 1).threads.filter((t) => t.active).map((t) => t.i)).toEqual([1, 3, 5]);
  });
});

describe('the worst case and the cost (p. 333)', () => {
  it('q-worst: [8, 1, ..., 7] is sorted after 7 steps; the 8 moves one place per step', () => {
    const rows = walk(LARGEST_FIRST, 7);
    expect(rows.map((r) => r.after.indexOf(8))).toEqual([1, 2, 3, 4, 5, 6, 7]);
    expect(runToSorted(LARGEST_FIRST, 'ours').sortedAfter).toBe(7);
  });
  it('with our rule the host then launches two more, quiet, steps: 9 in all', () => {
    const r = runToSorted(LARGEST_FIRST, 'ours');
    expect(r.launched).toBe(9);
    expect(r.rows.slice(7).map((x) => x.swaps)).toEqual([0, 0]);
    // The two quiet steps are an odd step (8) and the even step after it (9): either order counts.
    expect(r.rows.slice(7).map((x) => x.parity)).toEqual([1, 0]);
  });
  it('lab 14.1 reads the flag once per even-odd pair, so here it launches one step more: 10', () => {
    expect(pairedLaunches(LARGEST_FIRST)).toBe(10);
    expect(pairedLaunches(FIG_14_1)).toBe(8);
    expect(pairedLaunches([1, 3, 2, 4])).toBe(4);
    for (const p of PRESETS) {
      const ours = runToSorted(p.values, 'ours').launched;
      const paired = pairedLaunches(p.values);
      expect(paired % 2).toBe(0);
      expect(paired === ours || paired === ours + 1).toBe(true);
    }
  });
  it('reversed input of 8 needs 8 steps, and every comparison of the first swaps', () => {
    const r = runToSorted([9, 8, 7, 6, 5, 4, 3, 2], 'ours');
    expect(r.sortedAfter).toBe(8);
    expect(r.rows[0].swaps).toBe(4);
  });
  it('q-work: N steps of N/2 comparisons at N = 1,024 is 524,288; N log2 N is 10,240', () => {
    expect(bookBound(1024)).toBe(524288);
    expect(nLogN(1024)).toBe(10240);
  });
  it('random inputs of many lengths come out sorted under our rule within n steps of sorting', () => {
    let s = 3;
    const rnd = () => ((s = (s * 1103515245 + 12345) >>> 0) >>> 16) % 30;
    for (let n = 0; n <= 20; n++)
      for (let k = 0; k < 20; k++) {
        const v = Array.from({ length: n }, rnd);
        const r = runToSorted(v, 'ours');
        expect(r.out).toEqual(v.slice().sort((a, b) => a - b));
        expect(r.sortedAfter! <= Math.max(n, 0)).toBe(true);
      }
  });
});

describe('stopping (erratum 7: the host loop is not shown)', () => {
  it('q-stop: the trap [1, 3, 2, 4] makes no swap on its first even step', () => {
    expect(step([1, 3, 2, 4], 0).swaps).toBe(0);
    expect(runToSorted([1, 3, 2, 4], 'book')).toMatchObject({ launched: 1, sorted: false });
    expect(runToSorted([1, 3, 2, 4], 'ours')).toMatchObject({ sorted: true });
  });
  it('the distractors of q-stop all swap on their first even step', () => {
    for (const v of [[4, 3, 2, 1], [2, 1, 3, 4], [1, 2, 4, 3]]) expect(step(v, 0).swaps).toBeGreaterThan(0);
  });
  it('the eight-value trap stops unsorted under the literal rule', () => {
    expect(runToSorted(TRAP, 'book')).toMatchObject({ launched: 1, sorted: false, out: TRAP });
    expect(runToSorted(TRAP, 'ours').sorted).toBe(true);
  });
  it('already sorted input: one quiet step stops the literal rule, two stop ours', () => {
    expect(runToSorted([2, 3, 4, 5], 'book').launched).toBe(1);
    expect(runToSorted([2, 3, 4, 5], 'ours').launched).toBe(2);
  });
  it('fewer than two values: nothing to launch', () => {
    expect(runToSorted([], 'ours').launched).toBe(0);
    expect(runToSorted([5], 'ours').launched).toBe(0);
  });
  it('every preset is sorted at the end under our rule', () => {
    for (const p of PRESETS) expect(isAscending(runToSorted(p.values, 'ours').out)).toBe(true);
  });
});

describe('neighbouring pairs at once (the bug hunt)', () => {
  it('threads one after the other: one bubble-sort pass, still a permutation', () => {
    const r = overlapStep([9, 8, 7, 6, 5, 4, 3, 2], { kind: 'in-order' });
    expect(r.out).toEqual([8, 7, 6, 5, 4, 3, 2, 9]);
    expect(r.permutation).toBe(true);
    expect(overlapStep([9, 8, 7, 6, 5, 4, 3, 2], { kind: 'reversed' }).permutation).toBe(true);
  });
  it('two threads that both swap and share an element lose one value and duplicate another', () => {
    const r = overlapStep([3, 2, 1], { kind: 'race', at: 1 });
    expect(r.out).toEqual([2, 1, 2]);
    expect(r.lost).toEqual([3]);
    expect(r.duplicated).toEqual([2]);
    expect(r.permutation).toBe(false);
  });
  it('racing on element 4 of the reversed preset breaks the permutation', () => {
    const r = overlapStep([9, 8, 7, 6, 5, 4, 3, 2], { kind: 'race', at: 4 });
    expect(r.permutation).toBe(false);
  });
  it('a race where only one of the two threads swaps does no harm', () => {
    // Fig. 14.1's row, element 1: thread 0 has (4, 7), no swap; thread 1 has (7, 2), swap.
    expect(overlapStep(FIG_14_1, { kind: 'race', at: 1 }).permutation).toBe(true);
  });
  it('every thread runs its four operations once, loads before stores', () => {
    for (const il of [{ kind: 'in-order' as const }, { kind: 'reversed' as const }, { kind: 'race' as const, at: 3 }]) {
      const s = schedule(8, il);
      expect(s.length).toBe(7 * 4);
      for (let t = 0; t < 7; t++) expect(s.filter((x) => x.t === t).map((x) => x.op)).toEqual(['loadA', 'loadB', 'storeA', 'storeB']);
    }
  });
  it('a store with nothing to swap is skipped', () => {
    const r = overlapStep([1, 2, 3], { kind: 'in-order' });
    expect(r.events.filter((e) => e.op.startsWith('store')).every((e) => e.skipped)).toBe(true);
  });
});

describe('reading the reader\'s values', () => {
  it('accepts spaces or commas, 2 to 8 small integers', () => {
    expect(parseValues('4 7 2 3 8 5 9 6')).toEqual(FIG_14_1);
    expect(parseValues('4,7, 2')).toEqual([4, 7, 2]);
    expect(parseValues('4')).toBeNull();
    expect(parseValues('1 2 3 4 5 6 7 8 9')).toBeNull();
    expect(parseValues('1 x 3')).toBeNull();
    expect(parseValues('1 -2 3')).toBeNull();
  });
});
