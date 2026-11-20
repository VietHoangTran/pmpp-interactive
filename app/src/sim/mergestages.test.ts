import { describe, expect, it } from 'vitest';
import { coRankTrace } from './corank';
import { PANEL_KEYS, PANEL_SEG, blocksOf, mergePlan, mergeRuns, mergeSortStages } from './mergestages';

describe('the book\'s eight-block example (pp. 333-334, Fig. 14.3)', () => {
  it('eight segments on eight blocks: 2, 4, 8 blocks per merge over 3 stages', () => {
    const p = mergePlan({ n: 1024, segLen: 128, blocks: 8 });
    expect(p.stages).toBe(3);
    expect(p.rows.slice(1).map((r) => [r.merges, r.blocksPerMerge])).toEqual([[4, 2], [2, 4], [1, 8]]);
    expect(p.rows[0]).toMatchObject({ segments: 8, blocksPerMerge: 1, mergesPerBlock: 0 });
  });
  it('blocks per merge double each stage; elements per merge double too', () => {
    const p = mergePlan({ n: 2 ** 16, segLen: 256, blocks: 64 });
    const rows = p.rows.filter((r) => r.stage > 0 && r.blocksPerMerge > 0);
    for (let k = 1; k < rows.length; k++) {
      expect(rows[k].blocksPerMerge).toBe(2 * rows[k - 1].blocksPerMerge);
      expect(rows[k].elementsPerMerge).toBe(2 * rows[k - 1].elementsPerMerge);
    }
  });
  it('blocks 2k and 2k+1 share merge k in the first stage', () => {
    const r = mergePlan({ n: 1024, segLen: 128, blocks: 8 }).rows[1];
    expect([0, 1, 2, 3].map((m) => blocksOf(r, m))).toEqual([[0, 1], [2, 3], [4, 5], [6, 7]]);
  });
});

describe('the section\'s questions', () => {
  it('q-blocks: 64 blocks, 64 segments, merge stage 4: 16 blocks per merge', () => {
    const r = mergePlan({ n: 64 * 1024, segLen: 1024, blocks: 64 }).rows[4];
    expect([r.merges, r.blocksPerMerge]).toEqual([4, 16]);
  });
  it('q-stages: 2^20 keys in segments of 1,024: 10 merge stages', () => {
    const p = mergePlan({ n: 2 ** 20, segLen: 1024, blocks: 64 });
    expect(p.segments).toBe(1024);
    expect(p.stages).toBe(10);
  });
});

describe('fewer blocks than merges', () => {
  it('each block takes several whole merges and no split search is needed', () => {
    const p = mergePlan({ n: 2 ** 20, segLen: 1024, blocks: 64 });
    expect(p.rows[1]).toMatchObject({ merges: 512, blocksPerMerge: 0, mergesPerBlock: 8, splitSearches: 0 });
    expect(blocksOf(p.rows[1], 17)).toEqual([2]);
    // Merges fall below 64 at stage 5 (32 merges): 2 blocks per merge, one search each.
    expect(p.rows[5]).toMatchObject({ merges: 32, blocksPerMerge: 2, splitSearches: 32 });
    expect(p.rows[10]).toMatchObject({ merges: 1, blocksPerMerge: 64, splitSearches: 63, elementsPerMerge: 2 ** 20 });
  });
  it('stage 0 shares segments the same way', () => {
    expect(mergePlan({ n: 2 ** 20, segLen: 1024, blocks: 64 }).rows[0]).toMatchObject({ segments: 1024, mergesPerBlock: 16 });
  });
});

describe('steps per search', () => {
  it('is ceil(log2(input length + 1)); chapter 13\'s co-rank search stays within one more', () => {
    const p = mergePlan({ n: 16, segLen: 4, blocks: 4 });
    expect(p.rows[1].stepsPerSearch).toBe(3); // inputs of 4
    expect(p.rows[2].stepsPerSearch).toBe(4); // inputs of 8
    const A = [1, 3, 3, 5, 7, 8, 9, 12];
    const B = [2, 3, 4, 6, 7, 10, 11, 13];
    for (let k = 0; k <= 16; k++) expect(coRankTrace(k, A, B).steps.length).toBeLessThanOrEqual(p.rows[2].stepsPerSearch + 1);
  });
  it('holds for random sorted inputs with many repeats, inputs of 4 to 64', () => {
    let s = 9;
    const rnd = () => ((s = (s * 1103515245 + 12345) >>> 0) >>> 16) % 40;
    for (const m of [4, 8, 16, 32, 64]) {
      const bound = mergePlan({ n: 4 * m, segLen: m, blocks: 4 }).rows[1].stepsPerSearch;
      for (let t = 0; t < 50; t++) {
        const A = Array.from({ length: m }, rnd).sort((a, b) => a - b);
        const B = Array.from({ length: m }, rnd).sort((a, b) => a - b);
        for (let k = 0; k <= 2 * m; k++) expect(coRankTrace(k, A, B).steps.length).toBeLessThanOrEqual(bound + 1);
      }
    }
  });
});

describe('a real stable merge sort (the sixteen-key panel)', () => {
  const snaps = mergeSortStages(PANEL_KEYS, PANEL_SEG);
  it('has a sort stage and three merge stages for eight segments of two', () => {
    expect(snaps.map((s) => s.segLen)).toEqual([2, 4, 8, 16]);
    expect(snaps.length - 1).toBe(mergePlan({ n: 16, segLen: 2, blocks: 8 }).stages);
  });
  it('each stage leaves sorted runs of its segment length', () => {
    for (const s of snaps)
      for (let a = 0; a < 16; a += s.segLen) {
        const run = s.items.slice(a, a + s.segLen).map((x) => x.key);
        expect(run).toEqual(run.slice().sort((x, y) => x - y));
      }
  });
  it('the result is the stable sort: equal keys keep their input order', () => {
    const out = snaps[snaps.length - 1].items;
    expect(out.map((x) => x.key)).toEqual(PANEL_KEYS.slice().sort((a, b) => a - b));
    const stable = PANEL_KEYS.map((key, id) => ({ key, id })).sort((a, b) => a.key - b.key || a.id - b.id);
    expect(out).toEqual(stable);
  });
  it('q-stable: a merge that takes from the right run on ties loses stability', () => {
    const a = [{ key: 5, id: 0 }];
    const b = [{ key: 5, id: 1 }];
    expect(mergeRuns(a, b).map((x) => x.id)).toEqual([0, 1]);
    expect(mergeRuns(a, b, 'right').map((x) => x.id)).toEqual([1, 0]);
    // The question's runs: left [2a, 5a, 7], right [2b, 5b, 9] (ids: a before b).
    const left = [{ key: 2, id: 0 }, { key: 5, id: 1 }, { key: 7, id: 2 }];
    const right = [{ key: 2, id: 3 }, { key: 5, id: 4 }, { key: 9, id: 5 }];
    expect(mergeRuns(left, right, 'right').map((x) => `${x.key}:${x.id}`)).toEqual(['2:3', '2:0', '5:4', '5:1', '7:2', '9:5']);
    expect(mergeRuns(left, right, 'left').map((x) => `${x.key}:${x.id}`)).toEqual(['2:0', '2:3', '5:1', '5:4', '7:2', '9:5']);
  });
  it('q-morework: the last stage of 2^20 keys searches about 20 steps; 8 outputs per thread give 2,621,440 steps', () => {
    const last = mergePlan({ n: 2 ** 20, segLen: 1024, blocks: 64 }).rows.at(-1)!;
    expect(last.elementsPerMerge).toBe(2 ** 20);
    expect(last.stepsPerSearch).toBe(20);
    expect((2 ** 20 / 8) * last.stepsPerSearch).toBe(2621440);
  });
  it('with the right-first rule the panel still sorts its keys, but the 2s, 5s and 7s come out reordered', () => {
    const out = mergeSortStages(PANEL_KEYS, PANEL_SEG, 'right').at(-1)!.items;
    expect(out.map((x) => x.key)).toEqual(PANEL_KEYS.slice().sort((a, b) => a - b));
    const ids = (k: number) => out.filter((x) => x.key === k).map((x) => x.id);
    expect(ids(2)).toEqual([12, 3, 1]);
    expect(ids(5)).toEqual([10, 6, 0]);
    expect(ids(7)).toEqual([11, 2]);
    expect(ids(4)).toEqual([14, 5]);
    expect(ids(9)).toEqual([15, 4]);
  });
});
