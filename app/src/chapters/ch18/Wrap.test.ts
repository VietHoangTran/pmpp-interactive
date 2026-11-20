import { describe, expect, it } from 'vitest';
import {
  EMU_GRID, EX1C, EX1_CSR, EX1_ROWS, GRID_SUM, STRIDE_FRONTIER, STRIDE_MAX, STRIDE_THREE, TREE, TREE_SUM, questionsWrap,
} from './Wrap';

// The summary page computes Exercise 1's answers and every number its questions use from
// src/sim/bfs.ts and src/sim/gridbfs.ts. This test pins them against notes/ch18.md
// ("Exercises", our derivation) and against hand arithmetic, and checks the questions' shape.

describe('Exercise 1, as the page shows it', () => {
  it('1a: the rows of the adjacency matrix, 15 ones (notes: 0:{2,5}; 1:{0,4,7}; 2:{3}; ...)', () => {
    expect(EX1_ROWS).toEqual([[2, 5], [0, 4, 7], [3], [0, 6], [3], [1, 7], [4], [2, 4, 6]]);
    expect(EX1_ROWS.flat().length).toBe(15);
  });
  it('1b: CSR with sorted neighbour lists', () => {
    expect(EX1_CSR).toEqual({ srcPtrs: [0, 2, 5, 6, 8, 9, 11, 12, 15], dst: [2, 5, 0, 4, 7, 3, 0, 6, 3, 1, 7, 4, 2, 4, 6] });
  });
  it('1c: the table rows match the notes call by call', () => {
    const v = (k: string, what: string) => EX1C.find((r) => r.kernel === k && r.what.startsWith(what))!.values;
    expect(v('push', 'threads launched')).toEqual([8, 8, 8, 8]);
    expect(v('push', 'threads that loop')).toEqual([1, 2, 3, 2]);
    expect(v('pull', 'threads launched')).toEqual([8, 8, 8, 8]);
    expect(v('pull', 'threads that loop')).toEqual([7, 5, 2, 0]);
    expect(v('pull', 'threads that label')).toEqual([2, 3, 2, 0]);
    expect(v('edge', 'threads launched')).toEqual([15, 15, 15, 15]);
    expect(v('edge', 'threads whose source')).toEqual([2, 3, 8, 2]);
    expect(v('edge', 'threads that may label')).toEqual([2, 3, 4, 0]);
    expect(v('frontier', 'threads launched')).toEqual([1, 2, 3, 2]);
    expect(EX1C.length).toBe(9);
  });
});

describe('the numbers the mixed check uses', () => {
  it('a 16-ary tree 5 deep: 6 launches, a largest frontier of 16^5, about as many vertices as the 1000 x 1000 grid', () => {
    expect(TREE).toEqual({ b: 16, depth: 5 });
    expect(TREE_SUM.levels).toBe(5);
    expect(TREE_SUM.launches).toBe(6);
    expect(TREE_SUM.maxFrontier).toBe(1048576);
    expect(TREE_SUM.vertices).toBe(1118481);
    expect(GRID_SUM.vertices).toBe(1000000);
    expect(GRID_SUM.launches).toBe(1999);
    expect(GRID_SUM.maxFrontier).toBe(1000);
  });
  it('the emulator grid of 8 x 128 threads over a frontier of 2,500: 452 threads take 3, the rest 2', () => {
    expect(EMU_GRID.blocks * EMU_GRID.threads).toBe(1024);
    expect(STRIDE_FRONTIER).toBe(2500);
    expect(STRIDE_MAX).toBe(3);
    expect(STRIDE_THREE).toBe(2500 - 2048);
  });
  it('every question sits under this page\'s ids, numeric answers match the models, choices have one right option and a why for each wrong one', () => {
    expect(Object.keys(questionsWrap).sort()).toEqual([
      'ch18/wrap/q-exchange', 'ch18/wrap/q-sampler-order', 'ch18/wrap/q-stride-barrier', 'ch18/wrap/q-stride-count',
      'ch18/wrap/q-tree-launches',
    ]);
    const num = (id: string) => { const q = questionsWrap[id]; return q.kind === 'numeric' ? q.answer : NaN; };
    expect(num('ch18/wrap/q-tree-launches')).toBe(6);
    expect(num('ch18/wrap/q-stride-count')).toBe(452);
    for (const [id, q] of Object.entries(questionsWrap)) {
      expect(q.id).toBe(id);
      expect(q.source).toBeTruthy();
      if (q.kind !== 'choice') continue;
      expect(q.options.filter((o) => o.correct).length).toBe(1);
      for (const o of q.options) if (!o.correct) expect(o.why).toBeTruthy();
    }
  });
});
