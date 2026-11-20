import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { BF16, FP16, FP32, TOY5, add as mfAdd, maxFinite, nextUp, round as mfRound, ulp as mfUlp } from './minifloat';
import {
  BOOK_FOUR,
  BUG_HUNT,
  BUG_HUNT_N,
  EX4_VALUES,
  OUR_FOUR,
  SCENE,
  TEASER_N,
  TOY_VALUES,
  absSum,
  adder,
  bugHunt,
  bugHuntValues,
  challengeHolds,
  chunkSweep,
  ex4Order,
  generate,
  kahanTrace,
  kahanVariant,
  measure,
  reference,
  rng,
  rounder,
  runningTrace,
  sceneValues,
  sortByMagnitude,
  sum,
  sumChunked,
  sumKahan,
  sumSequential,
  sumSorted,
  sumTree,
  teaser,
  toySum,
  treeTrace,
} from './fpsum';

// ---------------------------------------------------------------- the rounders against minifloat
describe('fast rounding agrees with minifloat', () => {
  const r = rng(11);
  const cases = (emin: number, emax: number, count: number) =>
    Array.from({ length: count }, () => (r() < 0.5 ? -1 : 1) * (1 + r()) * 2 ** Math.floor(emin - 12 + r() * (emax - emin + 14)));
  for (const [F, emin, emax] of [
    [FP32, -126, 127],
    [FP16, -14, 15],
    [BF16, -126, 127],
  ] as const) {
    it(`${F.name}: random values, midpoints between neighbours, and the edges of the range`, () => {
      const rd = rounder(F);
      const xs = cases(emin, emax, 20000);
      // exact midpoints between neighbouring values: the ties
      for (let k = 0; k < 4000; k++) {
        const a = mfRound(F, cases(emin, emax - 1, 1)[0]);
        const b = nextUp(F, a)!;
        if (Number.isFinite(b)) xs.push((a + b) / 2);
      }
      const max = maxFinite(F);
      xs.push(max, -max, max + mfUlp(F, max) / 2, max + mfUlp(F, max) / 2 - 2 ** (emin - 40), 0, -0, 2 ** (emin - F.m - 1), 3 * 2 ** (emin - F.m - 1));
      for (const x of xs) expect(Object.is(rd(x), mfRound(F, x)) || rd(x) === mfRound(F, x)).toBe(true);
    });
    it(`${F.name}: a + b of stored values is minifloat's correctly rounded sum`, () => {
      const add = adder(F);
      const rd = rounder(F);
      for (let k = 0; k < 20000; k++) {
        const a = rd(cases(emin, emax, 1)[0]);
        const b = rd(cases(emin, emax, 1)[0]);
        expect(add(a, b)).toBe(mfAdd(F, a, b));
      }
    });
  }
  it("the book's 5-bit format: every pair", () => {
    const add = adder(TOY5);
    for (const a of TOY_VALUES) for (const b of TOY_VALUES) expect(add(a, b)).toBe(mfAdd(TOY5, a, b));
  });
});

// ---------------------------------------------------------------- the reference
describe('reference(): Neumaier in binary64 against an exact sum', () => {
  const exact = (xs: number[]) => {
    // every value here is a float32: an integer multiple of 2^-149
    const big = xs.reduce((s, x) => s + BigInt(x * 2 ** 149), 0n);
    return Number(big) / 2 ** 149;
  };
  it('random float32 lists, with and without cancellation, match the exactly rounded sum', () => {
    const r = rng(3);
    for (let k = 0; k < 300; k++) {
      const n = 2 + Math.floor(r() * 200);
      const xs = Array.from({ length: n }, () => Math.fround((r() < 0.5 ? -1 : 1) * 2 ** (-30 + 60 * r())));
      const e = exact(xs);
      expect(Math.abs(reference(xs) - e)).toBeLessThanOrEqual(Math.abs(e) * 2 ** -52);
    }
  });
  it('a sum that cancels to the small values is still exact', () => {
    const xs = [2 ** 40, 1, -(2 ** 40), 2 ** -20, 3];
    expect(reference(xs)).toBe(exact(xs));
    expect(reference(xs)).toBe(4 + 2 ** -20);
  });
});

// ---------------------------------------------------------------- the four-value stepper
describe("the book's 5-bit example and the challenge (pp. 593–594)", () => {
  it('1 + 1 + 0.25 + 0.25: 2.0 in order, 2.5 as a tree', () => {
    const s = toySum(BOOK_FOUR, 'sequential');
    const t = toySum(BOOK_FOUR, 'tree');
    expect(s.adds.map((a) => a.rounded)).toEqual([2, 2, 2]);
    expect(s.adds.map((a) => a.lost)).toEqual([0, 0.25, 0.25]);
    expect(s.result).toBe(2);
    expect(t.adds.map((a) => a.rounded)).toEqual([2, 0.5, 2.5]);
    expect(t.result).toBe(2.5);
    expect(s.exact).toBe(2.5);
    expect(challengeHolds(BOOK_FOUR)).toBe(false);
  });
  it('ours, 0.25, 1, 0.25, 2: 3.5 in order (exact), 3.0 as a tree', () => {
    const s = toySum(OUR_FOUR, 'sequential');
    const t = toySum(OUR_FOUR, 'tree');
    expect(s.adds.map((a) => a.rounded)).toEqual([1.25, 1.5, 3.5]);
    expect(s.result).toBe(3.5);
    expect(s.error).toBe(0);
    expect(t.adds.map((a) => a.rounded)).toEqual([1.25, 2, 3]);
    expect(t.adds.map((a) => a.lost)).toEqual([0, 0.25, 0.25]);
    expect(t.result).toBe(3);
    expect(challengeHolds(OUR_FOUR)).toBe(true);
  });
  it('a brute-force search over positive values finds the tree-worse cases, ours among them', () => {
    const pos = TOY_VALUES.filter((v) => v > 0);
    const found: number[][] = [];
    for (const a of pos) for (const b of pos) for (const c of pos) for (const d of pos) if (challengeHolds([a, b, c, d])) found.push([a, b, c, d]);
    expect(found.length).toBeGreaterThan(0);
    expect(found.some((f) => f.join() === OUR_FOUR.join())).toBe(true);
  });
  it('a value outside the format, or a sum past 3.5, does not count', () => {
    expect(challengeHolds([0.3, 1, 0.25, 2])).toBe(false);
    expect(toySum([3.5, 3.5, 0.25, 0.25], 'sequential').saturated).toBe(true);
  });
});

// ---------------------------------------------------------------- long sums
describe('long sums', () => {
  it('FP32 sequential is exactly a Math.fround loop', () => {
    for (const kind of ['uniform', 'mixed', 'cancelling', 'softmax'] as const) {
      const x = generate(kind, 5000);
      let s = 0;
      for (const v of x) s = Math.fround(s + v);
      expect(sumSequential(x, 'fp32')).toBe(s);
    }
  });
  it('an FP16 loop over ones stops at 2048, a BF16 loop at 256; uniform values stop at the same places', () => {
    const ones = new Float64Array(10000).fill(1);
    expect(sumSequential(ones, 'fp16')).toBe(2048);
    expect(sumSequential(ones, 'bf16')).toBe(256);
    const u = generate('uniform', 65536);
    expect(sumSequential(u, 'fp16')).toBe(2048);
    expect(sumSequential(u, 'bf16')).toBe(256);
  });
  it('prediction p-half-loop: 65,536 uniform values, true sum about 32,768, an FP16 loop returns 2048', () => {
    const u = generate('uniform', 65536);
    expect(reference(u)).toBeGreaterThan(32000);
    expect(reference(u)).toBeLessThan(33500);
    expect(sumSequential(u, 'fp16')).toBe(2048);
    const run = runningTrace(u, 'fp16');
    expect(run[run.length - 1].acc).toBe(2048);
    expect(run[run.length - 1].exact).toBeCloseTo(reference(u), 6);
    // it reaches 2048 well before the end and stays there: "about 60,000" additions that add nothing
    const add = adder('fp16');
    let first = 0;
    for (let i = 0, acc = 0; i < u.length && !first; i++) {
      acc = add(acc, u[i]);
      if (acc === 2048) first = i + 1;
    }
    expect(run.find((p) => p.acc === 2048)!.i).toBeLessThan(10000);
    expect(65536 - first).toBeGreaterThan(58000);
    expect(65536 - first).toBeLessThan(62000);
  });
  it('question q-half-tree: 262,144 uniform values in FP16, the loop stalls at 2048 and the tree overflows', () => {
    for (const seed of [7, 1, 2, 3]) {
      const u = generate('uniform', 262144, seed);
      expect(reference(u)).toBeGreaterThan(130000);
      expect(sumSequential(u, 'fp16')).toBe(2048);
      expect(sumTree(u, 'fp16')).toBe(Infinity);
    }
    // FP16's largest value, for the explanation
    expect(maxFinite(FP16)).toBe(65504);
  });
  it('chunked with c = 1 is the tree, with c = n the loop; Kahan chunked with c = n is the Kahan loop', () => {
    const x = generate('mixed', 4096);
    for (const acc of ['fp32', 'fp16', 'bf16'] as const) {
      expect(sumChunked(x, acc, 1)).toBe(sumTree(x, acc));
      expect(sumChunked(x, acc, 4096)).toBe(sumSequential(x, acc));
      expect(sumChunked(x, acc, 4096, true)).toBe(sumKahan(x, acc));
    }
  });
  it('the chunked order is chapter 10 coarsening: thread t adds elements t, t + T, ...', () => {
    // 8 values, c = 4, so T = 2 threads: thread 0 adds x0, x2, x4, x6; thread 1 adds x1, x3, x5, x7.
    const x = [1, 2 ** -24, 2 ** -24, 2 ** -24, 2 ** -24, 2 ** -24, 2 ** -24, 2 ** -24];
    const t0 = [x[0], x[2], x[4], x[6]].reduce((s, v) => Math.fround(s + v), 0);
    const t1 = [x[1], x[3], x[5], x[7]].reduce((s, v) => Math.fround(s + v), 0);
    expect(sumChunked(x, 'fp32', 4)).toBe(Math.fround(t0 + t1));
  });
  it('the explorer\'s discoveries hold on its default size (2^16, FP32)', () => {
    const n = 65536;
    const e = (kind: Parameters<typeof generate>[0], method: Parameters<typeof sum>[1]['method'], acc: 'fp32' | 'fp16' | 'bf16' = 'fp32', chunk?: number) =>
      measure(generate(kind, n), { method, acc, chunk });
    // ascending beats descending for the loop
    expect(e('mixed-asc', 'sequential').ulps).toBeLessThan(e('mixed-desc', 'sequential').ulps / 5);
    // the tree and Kahan beat the long loop, on every data set with any error
    for (const kind of ['uniform', 'mixed', 'mixed-desc', 'cancelling', 'softmax', 'big-small'] as const) {
      const loop = e(kind, 'sequential').ulps;
      expect(e(kind, 'tree').ulps).toBeLessThan(loop);
      expect(e(kind, 'kahan').ulps).toBeLessThan(loop);
      // Kahan's bound is relative to the sum of |x|, so only data of one sign is within an ULP of the result
      if (kind !== 'cancelling') expect(e(kind, 'kahan').ulps).toBeLessThanOrEqual(1);
      else expect(e(kind, 'kahan').rel).toBeLessThan(1e-8);
    }
    // narrow accumulators stall
    expect(e('uniform', 'sequential', 'fp16').rel).toBeGreaterThan(0.9);
    expect(e('uniform', 'sequential', 'bf16').rel).toBeGreaterThan(0.99);
  });
  it('question q-sign: on data of both signs, sorting by magnitude beats sorting by value and the unsorted loop', () => {
    const x = generate('cancelling', 65536);
    const ref = reference(x);
    const byValue = Float64Array.from(x).sort();
    const err = (s: number) => Math.abs(s - ref);
    expect(err(sumSorted(x, 'fp32'))).toBeLessThan(err(sumSequential(byValue, 'fp32')) / 5);
    expect(err(sumSorted(x, 'fp32'))).toBeLessThan(err(sumSequential(x, 'fp32')));
    const s = sortByMagnitude([-3, 0.5, -0.25, 2, 0, -1]);
    expect([...s]).toEqual([0, -0.25, 0.5, -1, 2, -3]);
  });
  it('question q-coarse: one big, many small, 2^20 values in FP32: from c = 4 to c = 1024 the error grows about 100 times; per-thread Kahan stays within 2 ULPs', () => {
    const x = generate('big-small', 1 << 20);
    const sweep = chunkSweep(x, 'fp32', 1024);
    const at = (c: number) => sweep.find((p) => p.c === c)!;
    expect(at(1024).plain.ulps).toBeGreaterThan(50 * at(4).plain.ulps);
    // the explanation's "about 3 ULPs at c = 4 to about 260 at c = 1024"
    expect(at(4).plain.ulps).toBeGreaterThan(2);
    expect(at(4).plain.ulps).toBeLessThan(4);
    expect(at(1024).plain.ulps).toBeGreaterThan(240);
    expect(at(1024).plain.ulps).toBeLessThan(280);
    expect(at(1024).kahan.ulps).toBeLessThanOrEqual(2);
    // the error grows with c at every step
    for (let k = 3; k < sweep.length; k++) expect(sweep[k].plain.ulps).toBeGreaterThan(sweep[k - 1].plain.ulps);
  });
  it('on random FP32 data the same change moves the error far less (the explorer says so)', () => {
    for (const kind of ['uniform', 'softmax', 'mixed'] as const) {
      const sweep = chunkSweep(generate(kind, 65536), 'fp32', 1024);
      for (const p of sweep) expect(p.plain.ulps).toBeLessThan(3);
    }
  });
  it('question q-kahan16: on 65,536 uniform values a plain FP32 loop is more accurate than Kahan in FP16', () => {
    const u = generate('uniform', 65536);
    const k16 = measure(u, { method: 'kahan', acc: 'fp16' });
    const p32 = measure(u, { method: 'sequential', acc: 'fp32' });
    expect(p32.abs).toBeLessThan(k16.abs / 10);
    // Kahan does reach FP16's own resolution of the result: within one FP16 ULP
    expect(k16.ulps).toBeLessThanOrEqual(1);
    // the numbers the explanation quotes: about 32,656 against 32,664, and the FP32 loop about 0.08 off
    expect(k16.result).toBe(32656);
    expect(Math.round(k16.reference)).toBe(32664);
    expect(p32.abs).toBeGreaterThan(0.07);
    expect(p32.abs).toBeLessThan(0.09);
  });
  it('one big, many small: the loop keeps only the big one, Kahan is within 1 ULP', () => {
    const x = generate('big-small', 65536);
    expect(sumSequential(x, 'fp32')).toBe(1);
    expect(measure(x, { method: 'kahan', acc: 'fp32' }).ulps).toBeLessThanOrEqual(1);
    expect(reference(x)).toBeGreaterThan(1.001);
  });
});

// ---------------------------------------------------------------- the teaser
describe("the start page's teaser: one softmax denominator, four ways", () => {
  const t = teaser();
  const row = (id: string) => t.rows.find((r) => r.id === id)!.e;
  it('131,072 terms, all in (0, 1], the largest exactly 1', () => {
    const x = generate('softmax', TEASER_N);
    expect(x.reduce((m, v) => Math.max(m, v), 0)).toBe(1);
    expect(x.reduce((m, v) => Math.min(m, v), 1)).toBeGreaterThan(0);
    expect(absSum(x)).toBeCloseTo(t.reference, 6);
    expect(t.reference).toBeGreaterThan(2000);
  });
  it('the float loop and tree are both within 0.01%, the tree far closer', () => {
    expect(row('fp32-loop').rel).toBeLessThan(1e-4);
    expect(row('fp32-tree').rel).toBeLessThan(row('fp32-loop').rel / 10);
  });
  it('the half loop stalls far below the true sum (off by more than three quarters); the half tree is within 0.1%', () => {
    expect(row('fp16-loop').rel).toBeGreaterThan(0.75);
    expect(row('fp16-loop').result).toBeLessThan(1024);
    expect(row('fp16-loop').result).toBeGreaterThan(200); // "once its running sum is a few hundred"
    expect(row('fp16-tree').rel).toBeLessThan(1e-3);
    expect(row('fp16-tree').ulps).toBeLessThanOrEqual(1);
  });
});

// ---------------------------------------------------------------- Exercise 4
describe('thread to data, with rounding (Exercise 4, p. 600, with Figs. 10.6 and 10.8)', () => {
  it('the convergent tree pairs thread i with i + 8, 4, 2, 1; the neighbour tree pairs 2i with 2i + 1, 2, 4, 8', () => {
    const v = Array.from({ length: 16 }, (_, i) => i);
    const c = treeTrace(v, 'convergent', 'fp32');
    expect(c.levels.map((l) => l.stride)).toEqual([8, 4, 2, 1]);
    expect(c.levels.map((l) => l.active.length)).toEqual([8, 4, 2, 1]);
    expect(c.steps[2]).toMatchObject({ thread: 2, dst: 2, src: 10 }); // the start page's q-convergent
    const nb = treeTrace(v, 'neighbours', 'fp32');
    expect(nb.levels.map((l) => l.stride)).toEqual([1, 2, 4, 8]);
    expect(nb.levels.map((l) => l.active)).toEqual([[0, 1, 2, 3, 4, 5, 6, 7], [0, 2, 4, 6], [0, 4], [0]]);
    expect(nb.steps[8]).toMatchObject({ thread: 0, dst: 0, src: 2 });
    expect(c.result).toBe(120);
    expect(nb.result).toBe(120);
  });
  for (const acc of ['fp16', 'bf16'] as const) {
    it(`${acc}: on the ascending preset the neighbour tree is exact and the convergent tree loses the eight 1s`, () => {
      const v = ex4Order(EX4_VALUES[acc], 'ascending');
      const c = treeTrace(v, 'convergent', acc);
      const nb = treeTrace(v, 'neighbours', acc);
      expect(nb.error).toBe(0);
      expect(c.error).toBe(8);
      // first level: the last three 1s each meet a big value and vanish
      const lost = c.steps.filter((s) => s.level === 0).map((s) => s.lost);
      expect(lost).toEqual([0, 0, 0, 0, 0, 1, 1, 1]);
      // descending order is just as bad for the convergent tree, and still exact for neighbours
      const d = ex4Order(EX4_VALUES[acc], 'descending');
      expect(treeTrace(d, 'convergent', acc).error).toBe(8);
      expect(treeTrace(d, 'neighbours', acc).error).toBe(0);
    });
  }
  it('lost is exactly what the addition dropped', () => {
    const t = treeTrace([1, 2048, 0.5, 3, 7, 1e-3, 2, 5], 'convergent', 'fp16');
    for (const s of t.steps) expect(s.a + s.b - s.rounded).toBeCloseTo(s.lost, 12);
  });
});

// ---------------------------------------------------------------- Kahan
describe("Kahan's correction (the scene) and the bug hunt", () => {
  const steps = kahanTrace(sceneValues(), 'fp32');
  const u = (v: number) => v * 2 ** 25;
  it('the first small value: t rounds back to 1.0 and c holds minus 2^-25', () => {
    expect(steps[1]).toMatchObject({ y: SCENE.x, t: 1, sum: 1 });
    expect(u(steps[1].c)).toBe(-1);
  });
  it('the second: y is 2^-24, t is a tie and rounds back to 1.0, c is minus 2^-24', () => {
    expect(u(steps[2].y)).toBe(2);
    expect(steps[2].t).toBe(1);
    expect(u(steps[2].c)).toBe(-2);
  });
  it('the third: y is 3 x 2^-25, t moves up one ULP (2^-23) and c records the overshoot, plus 2^-25', () => {
    expect(u(steps[3].y)).toBe(3);
    expect(steps[3].t).toBe(1 + 2 ** -23);
    expect(u(steps[3].c)).toBe(1);
    expect(steps[3].plain).toBe(1);
  });
  it('after 4,096 values Kahan is exact, 1 + 2^-13; the plain sum is still 1.0, 1,024 ULPs short', () => {
    const last = steps[steps.length - 1];
    expect(last.sum).toBe(1 + 2 ** -13);
    expect(last.exact).toBe(1 + 2 ** -13);
    expect(last.plain).toBe(1);
    expect((last.exact - last.plain) / mfUlp(FP32, 1)).toBe(1024);
  });
  it('the bug hunt on its data: version 1 and 4 are the plain sum, 2 is Kahan, 3 is far worse', () => {
    const { plain, runs } = bugHunt(bugHuntValues(), 'fp32');
    expect(plain.result).toBe(1);
    const by = Object.fromEntries(runs.map((r) => [r.name, r]));
    expect(BUG_HUNT.map((b) => by[b.name].verdict)).toEqual(['plain', 'kahan', 'worse', 'plain']);
    expect(by.sign.result.result).toBe(1);
    expect(by['no-c'].result.result).toBe(1);
    // the exact sum, 1 + 2^-9 - 2^-25, is not a float; Kahan returns the nearest one
    expect(by.correct.result.result).toBe(Math.fround(reference(bugHuntValues())));
    expect(by.correct.result.ulps).toBe(0.25);
    expect(by['late-c'].result.result).toBeGreaterThan(1000);
    expect(bugHuntValues().length).toBe(BUG_HUNT_N);
  });
  it('on uniform data the sign-flipped loop is not Kahan either, and version 4 is still exactly the plain loop', () => {
    const x = generate('uniform', 65536);
    const { plain, runs } = bugHunt(x, 'fp32');
    const by = Object.fromEntries(runs.map((r) => [r.name, r]));
    expect(by.sign.result.ulps).toBeGreaterThan(10 * by.correct.result.ulps + 1);
    expect(by['no-c'].result.result).toBe(plain.result);
    expect(by.correct.verdict).toBe('kahan');
    expect(by['late-c'].verdict).toBe('worse');
  });
  it('kahanVariant("correct") is sumKahan', () => {
    const x = generate('softmax', 3000);
    expect(kahanVariant('correct', x, 'bf16')).toBe(sumKahan(x, 'bf16'));
  });
});

// ---------------------------------------------------------------- the C++ the lesson shows
// labs/appA/snippets/kahan.h, compiled with the host compiler and no fast math, must give the
// model's floats bit for bit. Skipped when there is no C++ compiler.
const repo = path.resolve(__dirname, '../../..');
describe('labs/appA/snippets/kahan.h as section A.6 shows it', () => {
  const src = fs.readFileSync(path.join(repo, 'labs/appA/snippets/kahan.h'), 'utf8');
  it('has the plain loop and four versions, in the order of BUG_HUNT', () => {
    for (const b of BUG_HUNT) expect(src).toContain(`// Version ${b.version}\ntemplate <typename T>\nT version${b.version}(`);
    expect(src).toContain('T plain_sum(');
  });
  it('version 4 keeps the comment that says what it stands for', () => {
    expect(src).toContain("c = T(0);  // fast math's (t - sum) - y");
  });
});
const compiler = ['c++', 'clang++', 'g++'].find((c) => spawnSync(c, ['--version']).status === 0);
describe.skipIf(!compiler)('labs/appA/snippets/kahan.h, compiled and run', () => {
  const exe = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'appA-kahan-')), 'kahan_check');
  const cpp = path.join(repo, 'labs/appA/snippets/check/kahan_check.cpp');
  const view = new DataView(new ArrayBuffer(4));
  const hex = (x: number) => {
    view.setFloat32(0, x);
    return view.getUint32(0).toString(16).padStart(8, '0');
  };
  const run = (...args: (string | number)[]) => {
    const r = spawnSync(exe, args.map(String), { encoding: 'utf8' });
    expect(r.status).toBe(0);
    return Object.fromEntries(r.stdout.trim().split('\n').map((l) => l.split(' ')));
  };
  const model = (x: ArrayLike<number>) => ({
    plain: hex(sumSequential(x, 'fp32')),
    ...Object.fromEntries(BUG_HUNT.map((b) => [`version${b.version}`, hex(kahanVariant(b.name, x, 'fp32'))])),
  });
  it('builds without warnings', () => {
    const r = spawnSync(compiler!, ['-std=c++17', '-Wall', '-Wextra', '-O2', cpp, '-o', exe], { encoding: 'utf8' });
    expect(r.stderr).toBe('');
    expect(r.status).toBe(0);
  });
  it('the bug-hunt data: every version gives the model\'s float', () => {
    expect(run('bughunt')).toEqual(model(bugHuntValues()));
  });
  it('uniform, cancelling and softmax data: the same, bit for bit', () => {
    for (const kind of ['uniform', 'cancelling', 'softmax', 'big-small'] as const) {
      const x = [...generate(kind, 1500)];
      expect(run('list', ...x.map(String))).toEqual(model(x));
    }
  });
});
