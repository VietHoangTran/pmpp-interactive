import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  EX1,
  FIG410_PRINTED,
  FIG49,
  STRAIGHT_LINE,
  countTaken,
  discardedBySelect,
  ex1Line4,
  ex1Line7,
  ex1Trips,
  exercise1,
  fig410Trips,
  finished,
  ifElse,
  inputA,
  inputB,
  loop,
  loopBatch,
  loopRows,
  maskBelow,
  maskModulo,
  run,
  select,
  shuffledTrips,
  sortedTrips,
  startWarp,
  step,
  straightLine,
} from './divergence';
import { dims, members, warpsInGrid, warpsPerBlock } from './warps';

const W = FIG49.width;
const pct = (v: number) => Math.round(v * 1000) / 10;

describe('a warp with no branch (book pp. 76 to 77)', () => {
  it('every instruction is issued once and executed by every lane', () => {
    const t = run(straightLine(W));
    expect(t.spent).toBe(3);
    expect(t.issues.every((x) => x.count === W && x.efficiency === 1)).toBe(true);
    expect([t.useful, t.idle, t.uniform, t.efficiency, t.overhead]).toEqual([96, 0, 3, 1, 1]);
  });
});

describe('Fig. 4.9: if (threadIdx.x < 24) A else B, then C (p. 80)', () => {
  const taken = maskBelow(W, FIG49.threshold);
  const t = run(ifElse(W, taken, 1, 1));
  it('the condition is evaluated by all 32, A by 24 with 8 inactive, B by 8 with 24 inactive, C by all 32', () => {
    expect(t.issues.map((x) => [x.part, x.count])).toEqual([['op', 32], ['A', 24], ['B', 8], ['op', 32]]);
    expect(t.issues[1].active.indexOf(false)).toBe(24);
    expect(t.issues[2].active.indexOf(true)).toBe(24);
  });
  it('two passes over the branch, with efficiencies 75% and 25%', () => {
    expect(t.stmts[1]).toMatchObject({ kind: 'if', passes: 2, divergent: true, steps: 2 });
    expect([t.issues[1].efficiency, t.issues[2].efficiency]).toEqual([0.75, 0.25]);
  });
  it('a threshold of 0 or of the width leaves a single pass and no idle lane', () => {
    for (const T of [0, W]) {
      const u = run(ifElse(W, maskBelow(W, T), 1, 1));
      expect(u.stmts[1]).toMatchObject({ passes: 1, divergent: false });
      expect([u.spent, u.idle, u.overhead]).toEqual([3, 0, 1]);
    }
  });
  it('a threshold on a warp boundary costs the next warp nothing either', () => {
    // warp 1 holds threads 32 to 63: with T = 32 none of them takes A.
    const u = run(ifElse(W, maskBelow(W, 32, 32), 3, 3));
    expect(u.stmts[1].passes).toBe(1);
    expect(u.idle).toBe(0);
  });
  it('the cost follows the number of paths in the warp, not how many lanes take each', () => {
    const spent = [1, 8, 16, 24, 31].map((T) => run(ifElse(W, maskBelow(W, T), 3, 3)).spent);
    expect(new Set(spent)).toEqual(new Set([8]));
    expect(run(ifElse(W, maskBelow(W, 32), 3, 3)).spent).toBe(5);
    expect(run(ifElse(W, maskBelow(W, 24), 3, 3)).uniform).toBe(5);
  });
  it('with paths of 3 instructions the divergent warp spends 8 steps where uniform work needs 5', () => {
    const u = run(ifElse(W, taken, 3, 3));
    expect([u.spent, u.uniform, u.idle]).toEqual([8, 5, 96]);
    expect(u.overhead).toBe(1.6);
  });
  it('interleaved order issues the same instructions in another order, with identical totals', () => {
    const p = ifElse(W, taken, 3, 2);
    const s = run(p, 'serialised');
    const i = run(p, 'interleaved');
    expect(s.issues.filter((x) => x.stmt === 1).map((x) => x.part + x.k)).toEqual(['A0', 'A1', 'A2', 'B0', 'B1']);
    expect(i.issues.filter((x) => x.stmt === 1).map((x) => x.part + x.k)).toEqual(['A0', 'B0', 'A1', 'B1', 'A2']);
    for (const k of ['spent', 'useful', 'capacity', 'idle', 'uniform', 'efficiency', 'overhead'] as const) expect(i[k]).toBe(s[k]);
    expect(i.stmts).toEqual(s.stmts);
  });
  it('a path with no instructions is no pass, but the lanes that skip A still wait', () => {
    const u = run(ifElse(W, taken, 4, 0));
    expect(u.stmts[1]).toMatchObject({ passes: 1, divergent: true });
    expect(u.idle).toBe(8 * 4);
  });
  it('every other lane (i % 2 == 0) is as bad as any other split', () => {
    const u = run(ifElse(W, maskModulo(W, 2), 3, 3));
    expect(countTaken(maskModulo(W, 2))).toBe(16);
    expect(u.spent).toBe(8);
  });
});

describe('compute both, then select (ours; facts B7)', () => {
  it('no lane waits, and the steps are those of the divergent warp plus one', () => {
    const taken = maskModulo(W, 2);
    const s = run(select(W, 100, 100));
    const b = run(ifElse(W, taken, 100, 100));
    expect(s.idle).toBe(0);
    expect([b.spent, s.spent]).toEqual([202, 203]);
    expect(discardedBySelect(taken, 100, 100)).toBe(3200);
  });
  it('against a uniform warp a select always costs the other path as well', () => {
    expect(run(select(W, 2, 2)).spent - run(ifElse(W, maskBelow(W, W), 2, 2)).spent).toBe(3);
  });
  it('never cheaper than the branch, at any length of path (section 4.5 says so: the model gives a branch no cost of its own)', () => {
    for (const cost of [1, 2, 3, 8])
      for (const T of [0, 1, 16, 24, 31, 32]) {
        const s = run(select(W, cost, cost)).spent;
        const b = run(ifElse(W, maskBelow(W, T), cost, cost)).spent;
        expect(s).toBeGreaterThan(b);
        // A split warp pays both paths already, so the select adds exactly its own instruction.
        if (T > 0 && T < W) expect(s - b).toBe(1);
      }
    // The cases the lesson's list sends the reader to: paths of 1 and of 8 at T = 24.
    expect([1, 8].map((c) => [run(ifElse(W, maskBelow(W, 24), c, c)).spent, run(select(W, c, c)).spent])).toEqual([[4, 5], [18, 19]]);
  });
});

describe('Fig. 4.10: a loop with trip counts from data (p. 81)', () => {
  it('the eight printed columns: 8 iterations, 51 of 64 lane-steps', () => {
    const b = loopBatch(FIG410_PRINTED, 8);
    expect(b.warps).toHaveLength(1);
    expect([b.spent, b.useful, b.useful + b.idle]).toEqual([8, 51, 64]);
  });
  it('lanes drop out as drawn: all eight for four rows, then 7, 6, 4 and 2', () => {
    const rows = loopRows(FIG410_PRINTED);
    expect(rows.map((r) => r.filter(Boolean).length)).toEqual([8, 8, 8, 8, 7, 6, 4, 2]);
    expect(rows[4][3]).toBe(false); // the N = 4 column
    expect(rows[5][4]).toBe(false); // the N = 5 column
    expect(rows[7].map((v, i) => (v ? i : -1)).filter((i) => i >= 0)).toEqual([0, 6]); // the two N = 8 columns
  });
  it('a full warp of the printed values: 4 iterations without divergence, 4 with', () => {
    const t = run(loop(W, fig410Trips(W)));
    expect(t.stmts[1]).toMatchObject({ kind: 'loop', passes: 8, uniformIterations: 4, divergentIterations: 4, divergent: true });
    expect(t.issues.filter((x) => x.part === 'body').map((x) => x.count)).toEqual([32, 32, 32, 32, 28, 24, 16, 8]);
    expect([t.spent, t.useful]).toEqual([9, 32 + 204]);
  });
  it('the warp runs as long as its longest lane', () => {
    const t = run(loop(W, [...Array(31).fill(1), 9]));
    expect(t.stmts[1].passes).toBe(9);
    expect(t.idle).toBe(31 * 8);
  });
  it('equal trip counts do not diverge', () => {
    expect(run(loop(W, Array(W).fill(6))).stmts[1]).toMatchObject({ divergent: false, uniformIterations: 6, divergentIterations: 0 });
  });
});

describe('many warps of loops: shuffled against sorted (ours)', () => {
  const trips = shuffledTrips(8 * 32, 1, 64, 7);
  const shuffled = loopBatch(trips, 32);
  const sorted = loopBatch(sortedTrips(trips), 32);
  it('the work is the same in both orders', () => {
    expect(sorted.useful).toBe(shuffled.useful);
    expect(shuffled.useful).toBe(trips.reduce((a, b) => a + b, 0));
    expect(shuffledTrips(8 * 32, 1, 64, 7)).toEqual(trips);
  });
  it('sorted, each warp holds near-equal counts and runs about as long as it needs', () => {
    expect(sorted.warps.map((w) => w.longest)).toEqual([8, 16, 24, 32, 40, 48, 56, 64]);
    expect(sorted.spent).toBe(288);
    expect(sorted.uniform).toBe(260);
  });
  it('shuffled, every warp runs nearly as long as the longest sequence in the batch', () => {
    expect(shuffled.warps.every((w) => w.longest >= 56)).toBe(true);
    expect(shuffled.spent).toBeGreaterThan(1.6 * sorted.spent);
    expect(shuffled.idle).toBeGreaterThan(5 * sorted.idle);
  });
  it('a wider warp gains the same way', () => {
    expect(loopBatch(sortedTrips(trips), 64).spent).toBeLessThan(loopBatch(trips, 64).spent);
  });
});

describe('one fetch, many lanes (book pp. 76 to 79)', () => {
  it('each step fetches one instruction and writes one register in every lane', () => {
    let w = startWarp(0, 8);
    expect(w.regs.every((r) => r.r1 == null && r.r2 == null && r.r3 == null)).toBe(true);
    w = step(w);
    expect([w.pc, w.fetches, w.laneOps]).toEqual([1, 1, 8]);
    expect(w.regs.map((r) => r.r2)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
    expect(w.regs.every((r) => r.r1 == null && r.r3 == null)).toBe(true);
    w = step(w);
    expect(w.regs.map((r) => r.r3)).toEqual([10, 20, 30, 40, 50, 60, 70, 80]);
    w = step(w);
    expect(w.regs.map((r) => r.r1)).toEqual([11, 22, 33, 44, 55, 66, 77, 88]);
    expect(finished(w)).toBe(true);
    expect(step(w)).toBe(w);
  });
  it('three fetches serve every lane: 3 fetches, 3 x width lane-operations', () => {
    for (const width of [8, 32, 64]) {
      let w = startWarp(0, width);
      while (!finished(w)) w = step(w);
      expect([w.fetches, w.laneOps]).toEqual([3, 3 * width]);
      expect(w.fetches).toBe(run(straightLine(width)).spent);
      expect(w.laneOps).toBe(run(straightLine(width)).useful);
    }
  });
  it('a second warp has its own program counter and its own threads', () => {
    const a = step(step(startWarp(0, 8)));
    const b = step(startWarp(1, 8));
    expect([a.pc, b.pc]).toEqual([2, 1]);
    expect(b.first).toBe(8);
    expect(b.regs.map((r) => r.r2)).toEqual([9, 10, 11, 12, 13, 14, 15, 16]);
    expect(a.regs[0]).toEqual({ r1: null, r2: inputA(0), r3: inputB(0) });
  });
  it("the last instruction is the book's add r1, r2, r3", () => {
    expect(STRAIGHT_LINE.map((x) => x.text)).toEqual(['load r2, a[i]', 'load r3, b[i]', 'add r1, r2, r3']);
  });
});

describe('exercise 1 (book pp. 90 to 91; answers are ours, the book prints none)', () => {
  const e = exercise1(32);
  const w = (k: number) => e.detail.find((x) => x.block === 0 && x.warp === k)!;
  it('a and b: 4 warps per block, 32 in the grid', () => {
    expect([e.blocks, e.warpsPerBlock, e.warps]).toEqual([8, 4, 32]);
    expect(e.warpsPerBlock).toBe(warpsPerBlock(dims(EX1.blockDim), 32));
    expect(e.warps).toBe(warpsInGrid(8, dims(EX1.blockDim), 32));
  });
  it('c, line 4: 24 active warps, 16 divergent; efficiencies 100%, 25%, 75% for warps 0, 1 and 3 of block 0', () => {
    expect(e.line4).toEqual({ active: 24, divergent: 16 });
    expect([w(0), w(1), w(2), w(3)].map((x) => pct(x.line4.efficiency))).toEqual([100, 25, 0, 75]);
    expect([w(0), w(1), w(2), w(3)].map((x) => x.line4.count)).toEqual([32, 8, 0, 24]);
    expect(w(2).line4.active).toBe(false);
  });
  it('d, line 7: all 32 warps active and divergent; efficiency 50%', () => {
    expect(e.line7).toEqual({ active: 32, divergent: 32 });
    expect(pct(w(0).line7.efficiency)).toBe(50);
  });
  it('e, the loop: 3 iterations with no divergence and 2 with', () => {
    expect(e.loop).toEqual({ iterations: 5, uniform: 3, divergent: 2 });
    expect(e.detail.every((x) => x.uniformIterations === 3 && x.divergentIterations === 2)).toBe(true);
    expect(new Set(w(0).trips)).toEqual(new Set([5, 4, 3]));
  });
  it('the same statements as a program: line 4 in warp 1 of block 0 is one pass at 25%', () => {
    const t = run({ width: 32, stmts: [{ kind: 'if', label: 'line 3', taken: w(1).line4.mask, costA: 1, costB: 0 }] });
    expect(t.issues.map((x) => x.efficiency)).toEqual([0.25]);
    const l = run(loop(32, w(0).trips));
    expect(l.stmts[1]).toMatchObject({ uniformIterations: 3, divergentIterations: 2 });
  });
  it('lane l of warp w is thread w * width + l, as warps.ts cuts a 1D block', () => {
    const m = members(dims(EX1.blockDim), 32);
    for (const x of e.detail.filter((d) => d.block === 0)) expect(m[x.warp][0].x).toBe(x.first);
  });
  it('at width 64 (ours): 2 warps per block, both divergent at line 4', () => {
    const f = exercise1(64);
    expect([f.warpsPerBlock, f.warps]).toEqual([2, 16]);
    expect(f.line4).toEqual({ active: 16, divergent: 16 });
    expect(f.detail.slice(0, 2).map((x) => x.line4.count)).toEqual([40, 24]);
    expect(f.line7).toEqual({ active: 16, divergent: 16 });
    expect(f.loop).toEqual({ iterations: 5, uniform: 3, divergent: 2 });
  });
});

// The kernel of exercise 1 is shown in section 4.5 and kept as a real file so the
// CUDA front-end check parses it. These tests fail if the lesson's copy, the file
// and the model's three predicates drift apart.
const repo = path.resolve(__dirname, '../../..');
const snippet = fs.readFileSync(path.join(repo, 'labs/ch04/snippets/foo_kernel.cu'), 'utf8');
describe('exercise 1: the model, the lesson and labs/ch04/snippets/foo_kernel.cu agree', () => {
  const lines = snippet.split('\n');
  it('the statements sit on the lines the exercise names', () => {
    expect(lines[2]).toContain('if(threadIdx.x < 40 || threadIdx.x >= 104)');
    expect(lines[3]).toContain('b[i] = a[i] + 1;');
    expect(lines[5]).toContain('if(i%2 == 0)');
    expect(lines[6]).toContain('a[i] = b[i]*2;');
    expect(lines[8]).toContain('for(unsigned int j = 0; j < 5 - (i%3); ++j)');
    expect(snippet).toContain(`unsigned int N = ${EX1.n};`);
    expect(snippet).toContain(`foo_kernel<<<(N + ${EX1.blockDim} - 1)/${EX1.blockDim}, ${EX1.blockDim}>>>`);
  });
  it('the predicates are the file\'s conditions', () => {
    for (let t = 0; t < 128; t++) expect(ex1Line4(t)).toBe(t < 40 || t >= 104);
    for (let i = 0; i < 12; i++) {
      expect(ex1Line7(i)).toBe(i % 2 === 0);
      expect(ex1Trips(i)).toBe(5 - (i % 3));
    }
  });
  it('the lesson shows the file', () => {
    const src = fs.readFileSync(path.join(repo, 'app/src/chapters/ch04/S45.tsx'), 'utf8');
    const m = /const FOO = `([\s\S]*?)`;/.exec(src);
    expect(m).not.toBeNull();
    expect(m![1].trim()).toBe(snippet.trim());
  });
});
