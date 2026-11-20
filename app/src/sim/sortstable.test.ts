import { describe, expect, it } from 'vitest';
import {
  BATCH,
  BATCH_PER_ROW,
  BATCH_ROWS,
  BOOK_LIST,
  CANDIDATE_PRESETS,
  cascade,
  rowChecks,
  sortCandidates,
  stableSort,
  unstableSort,
  type PassKey,
} from './sortstable';
import { tieGroups } from './sortnet';

const preset = (id: string) => CANDIDATE_PRESETS.find((p) => p.id === id)!.items;

describe('the book\'s list (section 14.1, p. 330)', () => {
  it('by income, nonincreasing, stable: (32,80) stays before (29,80)', () => {
    expect(stableSort(BOOK_LIST, (p) => p.income, 'desc')).toEqual([
      { age: 30, income: 150 }, { age: 32, income: 80 }, { age: 29, income: 80 }, { age: 22, income: 45 },
    ]);
  });
  it('q-cascade: by age ascending, then stably by income nonincreasing, puts (29,80) before (32,80)', () => {
    const byAge = stableSort(BOOK_LIST, (p) => p.age, 'asc');
    expect(stableSort(byAge, (p) => p.income, 'desc')).toEqual([
      { age: 30, income: 150 }, { age: 29, income: 80 }, { age: 32, income: 80 }, { age: 22, income: 45 },
    ]);
  });
  it('[5, 2, 7, 1, 3, 2, 8] sorts to [1, 2, 2, 3, 5, 7, 8] either way', () => {
    const v = [5, 2, 7, 1, 3, 2, 8];
    expect(stableSort(v, (x) => x, 'asc')).toEqual([1, 2, 2, 3, 5, 7, 8]);
    expect(unstableSort(v, (x) => x, 'asc')).toEqual([1, 2, 2, 3, 5, 7, 8]);
  });
});

describe('ties in a candidate list (the first explorer)', () => {
  it('ties at two values: the network reorders the 0.15 group and leaves the 0.20 pair', () => {
    const g = tieGroups(sortCandidates(preset('two'), 'network', false));
    expect(g).toEqual([
      { key: 0.2, ids: [8, 40], increasing: true },
      { key: 0.15, ids: [15, 3, 52], increasing: false },
    ]);
  });
  it('the stable sort keeps every group in token order, with or without the index', () => {
    for (const p of CANDIDATE_PRESETS)
      for (const byIndex of [false, true]) expect(tieGroups(sortCandidates(p.items, 'stable', byIndex)).every((x) => x.increasing)).toBe(true);
  });
  it('adding the index makes the network agree with the stable sort on every preset', () => {
    for (const p of CANDIDATE_PRESETS) expect(sortCandidates(p.items, 'network', true)).toEqual(sortCandidates(p.items, 'stable', false));
  });
  it('all equal: no comparator fires, so the network keeps the input order by luck', () => {
    expect(sortCandidates(preset('equal'), 'network', false)).toEqual(preset('equal'));
  });
  it('no ties: nothing to check, and both sorts agree', () => {
    expect(tieGroups(sortCandidates(preset('none'), 'network', false))).toEqual([]);
    expect(sortCandidates(preset('none'), 'network', false)).toEqual(sortCandidates(preset('none'), 'stable', false));
  });
});

describe('two passes for a batch (the second explorer; PyTorch\'s few-row sort, ch14-facts D5)', () => {
  const run = (first: PassKey, secondStable: boolean) => {
    const second: PassKey = first === 'value' ? 'row' : 'value';
    const out = cascade(BATCH, [{ key: first, stable: true }, { key: second, stable: secondStable }]);
    return rowChecks(out[2], BATCH_ROWS, BATCH_PER_ROW);
  };
  it('the batch is 2 rows of 6, flattened row by row', () => {
    expect(BATCH.length).toBe(12);
    expect(rowChecks(BATCH, 2, 6).map((r) => r.contiguous)).toEqual([true, true]);
  });
  it('value first, then row with a stable sort: every row contiguous and sorted', () => {
    expect(run('value', true).map((r) => r.ok)).toEqual([true, true]);
  });
  it('value first, then row with the unstable sort: rows contiguous, values out of order (q-unstable-second)', () => {
    const r = run('value', false);
    expect(r.map((x) => x.contiguous)).toEqual([true, true]);
    expect(r.map((x) => x.sorted)).toEqual([false, false]);
  });
  it('row first, then value: rows interleaved, stable or not', () => {
    for (const s of [true, false]) {
      const r = run('row', s);
      expect(r.map((x) => x.contiguous)).toEqual([false, false]);
      expect(r.every((x) => !x.ok)).toBe(true);
    }
  });
  it('the cascade returns the input and each pass', () => {
    const c = cascade(BATCH, [{ key: 'value', stable: true }, { key: 'row', stable: true }]);
    expect(c.length).toBe(3);
    expect(c[0]).toEqual(BATCH);
    expect(c[1].map((t) => t.p)).toEqual(BATCH.map((t) => t.p).sort((a, b) => b - a));
  });
});
