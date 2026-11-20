import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  BOOK_LIST,
  FIG103A,
  FIG103B,
  FLT_MAX,
  HUGE_ROW,
  NEGATIVE_ROW,
  SHAPES,
  TIE_ROW,
  builderResult,
  builderTree,
  buildTree,
  compareOrders,
  disagreeingSet,
  evalFloat32,
  f32,
  f32bits,
  f32hex,
  f32short,
  f64short,
  foldSequential,
  generate,
  intTree,
  orderIndependent,
  pairCombine,
  pairs,
  randomRow,
  randomRowTest,
  reduceShape,
  rng,
  surveyOrders,
  tieRow,
  tournament,
  treeStats,
  ulpDistance,
  winners,
  type FoldOpId,
  type PairRule,
  type StartId,
} from './reducetree';

const levels = (values: number[], op: 'sum' | 'max', shape: 'loop' | 'neighbours' | 'halves') =>
  intTree(values, op, shape).levels.map((l) => l.map((nd) => Number(nd.value)));

describe('Fig. 10.3: the two trees the book draws (pp. 223, 227)', () => {
  it('(a) max over 3 1 7 0 4 1 6 3: 3 7 4 6, then 7 6, then 7', () => {
    expect(levels(FIG103A, 'max', 'neighbours')).toEqual([FIG103A, [3, 7, 4, 6], [7, 6], [7]]);
  });
  it('(b) sum over 4 7 2 3 8 5 9 6: 11 5 13 15, then 16 28, then 44', () => {
    expect(levels(FIG103B, 'sum', 'neighbours')).toEqual([FIG103B, [11, 5, 13, 15], [16, 28], [44]]);
  });
  it('the loop on (b) reads 4, 11, 13, 16, 24, 29, 38, 44 (the scene\'s first part)', () => {
    expect(levels(FIG103B, 'sum', 'loop').map((l) => l[0])).toEqual([4, 11, 13, 16, 24, 29, 38, 44]);
  });
  it('the two parenthesisations printed on p. 225', () => {
    expect(intTree(FIG103A, 'max', 'loop').expr).toBe('((((((3 max 1) max 7) max 0) max 4) max 1) max 6) max 3');
    expect(intTree(FIG103A, 'max', 'neighbours').expr).toBe('((3 max 1) max (7 max 0)) max ((4 max 1) max (6 max 3))');
  });
  it('4 adders, then 2, then 1: work 7, span 3; the loop does the same work in 7 steps', () => {
    const t = intTree(FIG103B, 'sum', 'neighbours');
    expect(t.opsPerStep).toEqual([4, 2, 1]);
    expect([t.work, t.span]).toEqual([7, 3]);
    const l = intTree(FIG103B, 'sum', 'loop');
    expect([l.work, l.span]).toEqual([7, 7]);
    expect(l.opsPerStep.every((k) => k === 1)).toBe(true);
  });
});

describe('work, span and parallelism (pp. 225-226)', () => {
  it('1024 inputs: 1023 operations in 10 steps, 512 at the peak, 102.3 on average', () => {
    const s = treeStats(1024);
    expect([s.work, s.span, s.peak]).toEqual([1023, 10, 512]);
    expect(s.opsPerStep[0]).toBe(512);
    expect(s.opsPerStep[9]).toBe(1);
    expect(s.average).toBeCloseTo(102.3, 1);
  });
  it('4096 inputs: average 341.25 against a peak of 2048 (question q-avg-4096)', () => {
    const s = treeStats(4096);
    expect(s.average).toBe(341.25);
    expect(s.peak).toBe(2048);
    expect(s.peak / s.average).toBeCloseTo(6, 0);
  });
  it('work is n - 1 for every shape and size', () => {
    for (const n of [2, 4, 8, 16])
      for (const shape of SHAPES) {
        const t = intTree(Array.from({ length: n }, (_, k) => k), 'sum', shape);
        expect(t.work).toBe(n - 1);
        expect(t.opsPerStep.reduce((a, b) => a + b, 0)).toBe(n - 1);
      }
    expect(treeStats(16).opsPerStep).toEqual([8, 4, 2, 1]);
  });
});

describe('what the operator must satisfy (pp. 224-225)', () => {
  const rand = rng(11);
  const ints = (n: number) => Array.from({ length: n }, () => Math.floor(rand() * 41) - 20);
  it('sum, max, min and product agree under every order on integers', () => {
    for (let k = 0; k < 200; k++) {
      const v = ints([4, 8, 16][k % 3]);
      for (const op of ['sum', 'max', 'min', 'product'] as const) {
        const r = SHAPES.map((s) => builderResult(v, op, s));
        expect(new Set(r).size).toBe(1);
      }
    }
  });
  it('subtraction: the trees disagree with the loop on Fig. 10.3(b)', () => {
    expect(builderResult(FIG103B, 'sub', 'loop')).toBe('-36');
    expect(builderResult(FIG103B, 'sub', 'neighbours')).toBe('-2');
  });
  it('subtraction cannot tell the two trees apart: each leaf is negated once per 1 bit of its index', () => {
    for (let k = 0; k < 200; k++) {
      const v = ints([4, 8, 16][k % 3]);
      expect(builderResult(v, 'sub', 'halves')).toBe(builderResult(v, 'sub', 'neighbours'));
      const signed = v.reduce((s, x, i) => s + (i.toString(2).split('1').length % 2 === 1 ? x : -x), 0);
      expect(builderResult(v, 'sub', 'neighbours')).toBe(String(signed));
    }
  });
  it('rewritten as a sum of negated operands, subtraction gives the loop\'s answer in any order', () => {
    const neg = FIG103B.map((x, i) => (i === 0 ? x : -x));
    for (const s of SHAPES) expect(builderResult(neg, 'sum', s)).toBe('-36');
  });
  it('join is associative but not commutative: the loop and neighbours read 4 7 2 3 8 5 9 6, halves does not', () => {
    expect(builderResult(FIG103B, 'join', 'loop')).toBe('47238596');
    expect(builderResult(FIG103B, 'join', 'neighbours')).toBe('47238596');
    expect(builderResult(FIG103B, 'join', 'halves')).toBe('48297536');
  });
  it('halves pairs x0 with x4, x1 with x5 ... with the lower position as the left operand', () => {
    expect(builderTree(FIG103B, 'join', 'halves').levels[1].map((n) => n.value)).toEqual(['48', '75', '29', '36']);
    expect(builderTree(FIG103B, 'join', 'halves').expr).toBe('((4 · 8) · (2 · 9)) · ((7 · 5) · (3 · 6))');
    expect(builderResult([12, 3, 45, 6], 'join', 'halves')).toBe('12,45,3,6');
  });
  it('builder trees share nodes between levels and kids, and show exact values', () => {
    const t = builderTree(FIG103B, 'sum', 'neighbours');
    expect(t.levels[1][0]).toBe(t.levels[2][0].kids![0]);
    expect(t.root.value).toBe('44');
    expect(builderTree(Array(16).fill(99), 'product', 'loop').root.value).toBe('8.51e31');
  });
  it('buildTree and the fast reduceShape agree for every shape', () => {
    for (let k = 0; k < 50; k++) {
      const v = generate(16, 'wide', k);
      for (const s of SHAPES) expect(f32bits(buildTree(v, (a, b) => f32(a + b), s).root.value)).toBe(f32bits(reduceShape(v, (a, b) => f32(a + b), s)));
    }
  });
});

describe('section 10.1: the sequential loop (pp. 221-223)', () => {
  it('the book\'s list sums to 34.6; in float32 the trace is 7, 9.1, 14.400001, 23.400002, 34.600002', () => {
    const t = foldSequential(BOOK_LIST, 'sum');
    expect(t.steps.map((s) => f32short(s.acc))).toEqual(['7', '9.1', '14.400001', '23.400002', '34.600002']);
    expect(f64short(t.truth)).toBe('34.6');
    expect(t.ok).toBe(true);
    expect(f64short(foldSequential(BOOK_LIST, 'sum', 'identity', 'f64').result)).toBe('34.6');
  });
  it('the float32 result is one float32 step above the float nearest 34.6', () => {
    const r = foldSequential(BOOK_LIST, 'sum').result;
    expect(f32hex(34.6)).toBe('0x420a6666');
    expect(f32hex(r)).toBe('0x420a6667');
    expect(ulpDistance(r, 34.6)).toBe(1);
  });
  it('N elements, N iterations from the identity; N - 1 from element 0 (footnote 1)', () => {
    expect(foldSequential(BOOK_LIST, 'sum', 'identity').steps.length).toBe(5);
    expect(foldSequential(BOOK_LIST, 'sum', 'first').steps.length).toBe(4);
  });
  it('0 as the max start is wrong on an all-negative row and right on a row with one huge logit', () => {
    expect(foldSequential(NEGATIVE_ROW, 'max', 'zero').ok).toBe(false);
    expect(foldSequential(NEGATIVE_ROW, 'max', 'zero').result).toBe(0);
    expect(foldSequential(HUGE_ROW, 'max', 'zero').ok).toBe(true);
    expect(foldSequential(NEGATIVE_ROW, 'argmax', 'zero').resultIndex).toBe(-1);
  });
  it('random rows rarely catch it: 0 as the max start fails on a few of 1000 rows, the identity on none', () => {
    const z = randomRowTest('max', 'zero');
    expect(z.wrong).toBeGreaterThan(0);
    expect(z.wrong).toBeLessThan(15);
    for (const op of ['sum', 'product', 'max', 'min', 'argmax'] as FoldOpId[]) {
      expect(randomRowTest(op, 'identity').wrong).toBe(0);
      expect(randomRowTest(op, 'first').wrong).toBe(0);
    }
  });
  it('-FLT_MAX is a safe start for max and argmax, and wrong for sum and min', () => {
    const wrong = (op: FoldOpId, st: StartId) => randomRowTest(op, st).wrong;
    expect(wrong('max', 'negmax')).toBe(0);
    expect(wrong('argmax', 'negmax')).toBe(0);
    expect(wrong('sum', 'negmax')).toBe(1000);
    expect(wrong('min', 'negmax')).toBe(1000);
    expect(f32(-FLT_MAX)).toBe(-FLT_MAX);
    expect(f32short(-FLT_MAX)).toBe('−3.4028235e+38');
  });
  it('argmax keeps the first of equal maxima; the truth is the first maximal index', () => {
    const t = foldSequential(TIE_ROW, 'argmax');
    expect([t.resultIndex, t.truthIndex, t.ok]).toEqual([1, 1, true]);
  });
  it('random rows are reproducible', () => {
    expect(randomRow(8, 3)).toEqual(randomRow(8, 3));
    expect(randomRow(8, 3)).not.toEqual(randomRow(8, 4));
  });
});

describe('float32 sums in three orders (p. 224, Appendix A section A.6)', () => {
  it('the explorer\'s starting data (1024 softmax terms, seed 1): the orders give more than one result', () => {
    const r = compareOrders(generate(1024, 'softmax', 1));
    expect(r.distinct).toBeGreaterThan(1);
    expect(SHAPES.every((s) => r.ulps[s] === ulpDistance(r.sums[s], f32(r.ref)))).toBe(true);
  });
  it('at 1024 values the tree is the closest more often than the loop, but not always', () => {
    for (const d of ['uniform', 'softmax', 'wide'] as const) {
      const s = surveyOrders(1024, d, 100);
      expect(s.disagree).toBeGreaterThan(90);
      expect(s.closest.neighbours).toBeGreaterThan(s.closest.loop);
      expect(s.alone.loop + s.closest.loop).toBeGreaterThan(0);
      expect(s.closest.neighbours).toBeLessThan(100);
    }
  });
  it('the scene\'s example: 8 values whose loop and tree sums differ in the last bits', () => {
    const d = disagreeingSet(8, 'uniform', 1);
    expect(f32hex(d.loop)).not.toBe(f32hex(d.tree));
    expect(ulpDistance(d.loop, d.tree)).toBeLessThan(8);
    expect(d.loop).toBe(evalFloat32(d.values, 'loop'));
  });
  it('small whole numbers add exactly in float32, so every order agrees', () => {
    const rand = rng(5);
    for (let k = 0; k < 100; k++) {
      const v = Array.from({ length: 16 }, () => Math.floor(rand() * 2000) - 1000);
      expect(new Set(SHAPES.map((s) => evalFloat32(v, s))).size).toBe(1);
    }
  });
  it('generators are seeded and in range', () => {
    expect(generate(64, 'uniform', 9)).toEqual(generate(64, 'uniform', 9));
    expect(generate(64, 'uniform', 9).every((x) => x >= -1 && x <= 1)).toBe(true);
    const sm = generate(256, 'softmax', 2);
    expect(Math.max(...sm)).toBe(1);
    expect(sm.every((x) => x > 0 && x <= 1)).toBe(true);
    expect(generate(256, 'wide', 2).some((x) => x < 0)).toBe(true);
  });
});

describe('who wins a tie (footnote 3, p. 225; ch10 facts A10, A13, B5, D2-D4)', () => {
  it('on the default row, keep-left depends on the order; lower-index and higher-index do not', () => {
    expect(winners(TIE_ROW, 'left')).toEqual({ loop: { v: 4, i: 1 }, neighbours: { v: 4, i: 1 }, halves: { v: 4, i: 4 } });
    expect(orderIndependent(TIE_ROW, 'left')).toBe(false);
    expect(orderIndependent(TIE_ROW, 'lower')).toBe(true);
    expect(winners(TIE_ROW, 'lower').halves.i).toBe(1);
    expect(winners(TIE_ROW, 'higher').halves.i).toBe(4);
  });
  it('commutative pair rules give the same winner under every order, on many rows with planted ties', () => {
    const rand = rng(3);
    for (let k = 0; k < 300; k++) {
      const n = k % 2 ? 16 : 8;
      const a = Math.floor(rand() * n);
      const b = (a + 1 + Math.floor(rand() * (n - 1))) % n;
      const row = tieRow(n, [a, b], null, k);
      for (const r of ['lower', 'higher', 'nanfirst'] as PairRule[]) expect(orderIndependent(row, r)).toBe(true);
      expect(winners(row, 'lower').loop.i).toBe(Math.min(a, b));
      expect(winners(row, 'higher').loop.i).toBe(Math.max(a, b));
    }
  });
  it('the lower-index rule is commutative on finite values; keep-left is not', () => {
    const p = pairs([2, 2]);
    const swap = (r: PairRule) => [pairCombine(r)(p[0], p[1]).i, pairCombine(r)(p[1], p[0]).i];
    expect(swap('lower')).toEqual([0, 0]);
    expect(swap('left')).toEqual([0, 1]);
  });
  it('with a NaN, the comparison max depends on where the NaN sits; fmaxf ignores it', () => {
    const row = tieRow(8, [1, 4], 6);
    expect(Number.isNaN(row[6])).toBe(true);
    expect(orderIndependent(row, 'cmp')).toBe(false);
    expect(winners(row, 'cmp').loop.v).toBe(row[7]);
    expect(orderIndependent(row, 'fmax')).toBe(true);
    expect(winners(row, 'fmax').loop.v).toBe(4);
  });
  it('NaN wins under the PyTorch and MLX policy, in every order', () => {
    const row = tieRow(8, [1, 4], 6);
    const w = winners(row, 'nanfirst');
    expect(SHAPES.every((s) => w[s].i === 6 && Number.isNaN(w[s].v))).toBe(true);
  });
  it('the comparison form of the lower-index rule is not commutative with a NaN operand (hipCUB, B5)', () => {
    const [a, b] = pairs([NaN, 3]);
    expect(pairCombine('lower')(a, b).i).toBe(0);
    expect(pairCombine('lower')(b, a).i).toBe(1);
  });
});

describe('the sidebar: knockout tournaments (pp. 226-227)', () => {
  it('8 teams: 3 rounds; with 4 venues 3 days, with 2 venues 4 days (question q-venues)', () => {
    expect(tournament(8, 4)).toMatchObject({ rounds: 3, days: 3, gamesPerRound: [4, 2, 1] });
    expect(tournament(8, 2)).toMatchObject({ rounds: 3, days: 4, daysPerRound: [2, 1, 1] });
  });
  it('1024 teams need 10 rounds of 512, 256, ... games', () => {
    const t = tournament(1024, 512);
    expect(t.rounds).toBe(10);
    expect(t.gamesPerRound.slice(0, 3)).toEqual([512, 256, 128]);
  });
});

// The C++ shown in section 10.1 (labs/ch10/snippets/fold.h), compiled and run, must
// give the model's floats bit for bit. Skipped when there is no C++ compiler.
const repo = path.resolve(__dirname, '../../..');
describe('labs/ch10/snippets/fold.h as section 10.1 shows it', () => {
  it('lines 8 to 10, which the section highlights, are the start, the loop and the operator', () => {
    const l = fs.readFileSync(path.join(repo, 'labs/ch10/snippets/fold.h'), 'utf8').split('\n');
    expect(l[7]).toContain('T acc = start;');
    expect(l[8]).toContain('for (std::size_t i = 0; i < n; ++i)');
    expect(l[9]).toContain('acc = op(acc, input[i]);');
    expect(fs.readFileSync(path.join(__dirname, '../chapters/ch10/S101.tsx'), 'utf8')).toContain('highlight={[8, 9, 10]}');
  });
});
const compiler = ['c++', 'clang++', 'g++'].find((c) => spawnSync(c, ['--version']).status === 0);
describe.skipIf(!compiler)('labs/ch10/snippets/fold.h, compiled and run', () => {
  const exe = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'ch10-fold-')), 'fold_test');
  const src = path.join(repo, 'labs/ch10/snippets/check/fold_test.cpp');
  const run = (...args: (string | number)[]) => {
    const r = spawnSync(exe, args.map(String), { encoding: 'utf8' });
    expect(r.status).toBe(0);
    return r.stdout.trim().split('\n');
  };
  const hex = (x: number) => f32hex(x).slice(2);
  it('builds without warnings', () => {
    const r = spawnSync(compiler!, ['-std=c++17', '-Wall', '-Wextra', '-O2', src, '-o', exe], { encoding: 'utf8' });
    expect(r.status).toBe(0);
    expect(r.stderr).toBe('');
  });
  it('the float loop over the book\'s list gives the model\'s trace, ending on 0x420a6667', () => {
    const want = foldSequential(BOOK_LIST, 'sum').steps.map((s) => hex(s.acc));
    expect(run('trace', ...BOOK_LIST)).toEqual([...want, want[want.length - 1]]);
    expect(want[want.length - 1]).toBe('420a6667');
  });
  it('max_logit finds -0.9 on the all-negative row; started at 0.0f the same loop returns 0', () => {
    expect(run('max', ...NEGATIVE_ROW)).toEqual([hex(foldSequential(NEGATIVE_ROW, 'max').result)]);
    expect(run('max0', ...NEGATIVE_ROW)).toEqual([hex(foldSequential(NEGATIVE_ROW, 'max', 'zero').result)]);
    expect(run('max0', ...NEGATIVE_ROW)).toEqual(['00000000']);
  });
  it('argmax_row keeps the lower index on the tie row, as the model does', () => {
    expect(run('argmax', ...TIE_ROW)).toEqual([String(foldSequential(TIE_ROW, 'argmax').resultIndex)]);
    expect(run('argmax', ...HUGE_ROW)).toEqual(['3']);
  });
});
