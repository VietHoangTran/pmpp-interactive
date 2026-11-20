import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  BOOK_PRINTED,
  CONVERGENT_SPEC,
  EXERCISE_6,
  FIG_VALUES,
  HUNT_FILE,
  INPLACE_FILE,
  INPLACE_KERNEL,
  INPLACE_LINES,
  MIRROR_SPEC,
  VARIANT,
  VARIANTS,
  arraysAfter,
  bookFormula,
  divergentAt,
  expectedOf,
  firstIssueStep,
  historyAt,
  iterations,
  requests,
  runInPlace,
  runShared,
  runSpec,
  stateAt,
  strides,
  type Outcome,
  type Schedule,
  type Variant,
} from './reduction';
import { functionSource } from './toolchain';

const repo = path.resolve(__dirname, '../../..');
const labFile = (rel: string) => fs.readFileSync(path.join(repo, 'labs', rel), 'utf8');
const SCHEDULES: Schedule[] = ['lockstep', 'in-order', 'reversed', 'shuffled'];
const ones = (n: number) => new Array<number>(n).fill(1);

describe('warps and efficiency, the 256-element example (pp. 230 to 234)', () => {
  const simple = iterations('simple', 256, 32);
  const conv = iterations('convergent', 256, 32);
  it('128 threads in 4 warps, 8 iterations each', () => {
    expect(simple.threads).toBe(128);
    expect(simple.warpCount).toBe(4);
    expect(simple.iterations.map((i) => i.stride)).toEqual([1, 2, 4, 8, 16, 32, 64, 128]);
    expect(conv.iterations.map((i) => i.stride)).toEqual([128, 64, 32, 16, 8, 4, 2, 1]);
  });
  it('the per-iteration table of the notes: same active threads, different active warps', () => {
    expect(simple.iterations.map((i) => i.active)).toEqual([128, 64, 32, 16, 8, 4, 2, 1]);
    expect(conv.iterations.map((i) => i.active)).toEqual([128, 64, 32, 16, 8, 4, 2, 1]);
    expect(simple.iterations.map((i) => i.activeWarps)).toEqual([4, 4, 4, 4, 4, 4, 2, 1]);
    expect(conv.iterations.map((i) => i.activeWarps)).toEqual([4, 2, 1, 1, 1, 1, 1, 1]);
  });
  it('simple: 864 slots, 255 results, efficiency 0.30 (p. 231)', () => {
    expect(simple.slots).toBe(864);
    expect(simple.results).toBe(255);
    expect(simple.efficiency.toFixed(2)).toBe('0.30');
  });
  it('convergent: 384 slots, 66% (p. 234)', () => {
    expect(conv.slots).toBe(384);
    expect(conv.results).toBe(255);
    expect(Math.floor(conv.efficiency * 100)).toBe(66);
  });
  it('iteration 7 of the simple kernel: warps 1 and 3 idle, warps 0 and 2 one thread each (p. 231)', () => {
    const it7 = simple.iterations[6];
    expect(it7.warps.map((w) => w.state)).toEqual(['divergent', 'idle', 'divergent', 'idle']);
    expect(it7.warps.map((w) => w.active)).toEqual([1, 0, 1, 0]);
  });
  it('divergent iterations: seven in the simple kernel, five in the convergent one (the book prints ten and five)', () => {
    expect(simple.divergentIterations).toBe(7);
    expect(conv.divergentIterations).toBe(5);
    expect(BOOK_PRINTED).toEqual({ simpleSlots: 736, divergentBefore: 10, divergentAfter: 5 });
    expect(simple.slots).not.toBe(BOOK_PRINTED.simpleSlots);
  });
  it('the convergent kernel: whole warps go idle; divergence only in the last five iterations', () => {
    expect(conv.iterations[1].warps.map((w) => w.state)).toEqual(['full', 'full', 'idle', 'idle']);
    expect(conv.iterations.map((i) => i.divergentWarps)).toEqual([0, 0, 0, 1, 1, 1, 1, 1]);
  });
});

describe('the book\'s formulas, written out, agree with the model for every N from 64 to 2048', () => {
  for (const N of [64, 128, 256, 512, 1024, 2048])
    it(`N = ${N}`, () => {
      expect(iterations('simple', N, 32).slots).toBe(bookFormula.simpleSlots(N));
      expect(iterations('convergent', N, 32).slots).toBe(bookFormula.convergentSlots(N));
      expect(requests('simple', N, 32).total).toBe(bookFormula.simpleRequests(N));
      expect(requests('shared', N, 32).total).toBe(bookFormula.sharedRequests(N));
      // The convergent kernel: three coalesced accesses per active warp per iteration (36 at 256, p. 237).
      expect(requests('convergent', N, 32).total).toBe(3 * bookFormula.convergentSlots(N) / 32);
      expect(iterations('simple', N, 32).results).toBe(N - 1);
    });
  it('p. 231 and p. 234 as printed: 864 and 384 at N = 256; 736 is not either', () => {
    expect(bookFormula.simpleSlots(256)).toBe(864);
    expect(bookFormula.convergentSlots(256)).toBe(384);
  });
});

describe('exercises 1 and 2 (p. 249) and the quiz', () => {
  it('exercise 1: simple kernel, 1024 elements, iteration 5: all 16 warps diverge', () => {
    expect(iterations('simple', 1024, 32).iterations[4].stride).toBe(16);
    expect(divergentAt('simple', 1024, 32, 5)).toBe(16);
    expect(iterations('simple', 1024, 32).iterations[4].warps.every((w) => w.active === 2)).toBe(true);
  });
  it('exercise 2: convergent kernel, iteration 5: none; divergence starts in iteration 6 with one warp', () => {
    expect(iterations('convergent', 1024, 32).iterations[4].stride).toBe(32);
    expect(divergentAt('convergent', 1024, 32, 5)).toBe(0);
    expect(divergentAt('convergent', 1024, 32, 6)).toBe(1);
  });
  it('convergent at 1024: 36 warp-iterations, 1152 slots, 1023 results, 89%', () => {
    const t = iterations('convergent', 1024, 32);
    expect(t.slots).toBe(1152);
    expect(t.results).toBe(1023);
    expect(Math.round(t.efficiency * 100)).toBe(89);
  });
  it('64-wide warps at 256: convergent 576 slots (44%), simple 960; one more divergent iteration', () => {
    const c = iterations('convergent', 256, 64);
    expect(c.iterations.map((i) => i.activeWarps)).toEqual([2, 1, 1, 1, 1, 1, 1, 1]);
    expect(c.slots).toBe(576);
    expect(Math.round(c.efficiency * 100)).toBe(44);
    expect(iterations('simple', 256, 64).slots).toBe(960);
    expect(c.divergentIterations).toBe(iterations('convergent', 256, 32).divergentIterations + 1);
  });
  it('a block smaller than a warp: one warp, padded lanes, and a full warp still costs its width', () => {
    const t = iterations('simple', 16, 32);
    expect(t.warpCount).toBe(1);
    expect(t.iterations[0].warps[0].real).toBe(8);
    expect(t.iterations[0].warps[0].state).toBe('full');
    expect(t.iterations[0].warps[0].lanes.filter((l) => l === null)).toHaveLength(24);
    expect(t.slots).toBe(4 * 32);
  });
  it('rejects sizes one block cannot take', () => {
    expect(() => iterations('simple', 4096, 32)).toThrow();
    expect(() => iterations('simple', 100, 32)).toThrow();
    expect(() => iterations('simple', 256, 0)).toThrow();
  });
});

describe('global-memory requests (pp. 235 to 237)', () => {
  it('141 / 36 / 9 at 256 and 1149 / 204 / 65 at 2048', () => {
    expect([requests('simple', 256, 32).total, requests('convergent', 256, 32).total, requests('shared', 256, 32).total]).toEqual([141, 36, 9]);
    expect([requests('simple', 2048, 32).total, requests('convergent', 2048, 32).total, requests('shared', 2048, 32).total]).toEqual([1149, 204, 65]);
  });
  it('simple kernel, first iteration: a warp\'s 32 own elements span two pieces of 32, so half the bytes are used', () => {
    const w0 = requests('simple', 256, 32).steps[0].warps[0];
    const own = w0.accesses.find((a) => a.name === 'own')!;
    expect(own.cells).toHaveLength(32);
    expect(own.segments).toEqual([0, 1]);
    expect(own.cells.length / (own.segments.length * 32)).toBe(0.5);
  });
  it('two active lanes 32 floats apart still cost two requests per access (iteration 5)', () => {
    const w0 = requests('simple', 256, 32).steps[4].warps[0];
    expect(w0.accesses.find((a) => a.name === 'own')!.cells).toEqual([0, 32]);
    expect(w0.accesses.map((a) => a.segments.length)).toEqual([2, 2, 2]);
  });
  it('the convergent kernel: one piece per access, and idle warps make no requests', () => {
    const r = requests('convergent', 256, 32);
    for (const s of r.steps) for (const w of s.warps) expect(w.accesses.map((a) => a.segments.length)).toEqual([1, 1, 1]);
    expect(r.steps.map((s) => s.warps.length)).toEqual([4, 2, 1, 1, 1, 1, 1, 1]);
  });
  it('the shared-memory kernel: N/32 coalesced loads, nothing in the loop, one store', () => {
    const r = requests('shared', 256, 32);
    expect(r.steps[0].requests).toBe(8);
    expect(r.steps.filter((s) => s.phase === 'iteration').every((s) => s.requests === 0)).toBe(true);
    expect(r.steps[r.steps.length - 1].requests).toBe(1);
  });
});

describe('exercise 6 (p. 250): the arrays after each iteration', () => {
  it('6a, simple kernel: input[4] ends at 17, output 36', () => {
    expect(arraysAfter('simple', EXERCISE_6)).toEqual([
      [8, 2, 11, 4, 13, 8, 4, 1],
      [19, 2, 11, 4, 17, 8, 4, 1],
      [36, 2, 11, 4, 17, 8, 4, 1],
    ]);
  });
  it('6b, convergent kernel: input[1] ends at 15', () => {
    expect(arraysAfter('convergent', EXERCISE_6)).toEqual([
      [11, 10, 10, 5, 5, 8, 3, 1],
      [21, 15, 10, 5, 5, 8, 3, 1],
      [36, 15, 10, 5, 5, 8, 3, 1],
    ]);
  });
});

describe('exercise 3 (p. 249): a kernel from parts', () => {
  it('Fig. 10.8\'s own parts give the convergent kernel\'s arrays', () => {
    const r = runSpec(CONVERGENT_SPEC, FIG_VALUES);
    expect(r.correct).toBe(true);
    expect(r.iterations.map((i) => i.after)).toEqual(arraysAfter('convergent', FIG_VALUES));
  });
  it('the mirrored answer lands the total in the last element, with no conflict and nothing out of bounds', () => {
    const r = runSpec(MIRROR_SPEC, FIG_VALUES);
    expect(r.correct).toBe(true);
    expect(r.totalInLast).toBe(true);
    expect(r.output).toBe(69);
    expect(r.iterations.map((i) => i.ops.map((o) => o.own))).toEqual([
      [8, 9, 10, 11, 12, 13, 14, 15], [12, 13, 14, 15], [14, 15], [15],
    ]);
  });
  it('the mirrored parts with the wrong test or the wrong writer fail', () => {
    expect(runSpec({ ...MIRROR_SPEC, test: 'lt' }, FIG_VALUES).correct).toBe(false);
    expect(runSpec({ ...MIRROR_SPEC, writer: 'first' }, FIG_VALUES).correct).toBe(false);
    const plus = runSpec({ ...MIRROR_SPEC, partner: 'plus' }, FIG_VALUES);
    expect(plus.oob).toBeGreaterThan(0);
  });
  it('a test that lets a thread read what another writes in the same iteration is a conflict', () => {
    const r = runSpec({ owned: 't', test: 'ge', partner: 'minus', writer: 'last' }, FIG_VALUES);
    expect(r.conflicts).toBeGreaterThan(0);
    expect(r.correct).toBe(false);
  });
});

describe('the in-place kernels, thread by thread (section 10.3)', () => {
  it('with the barrier, every order is correct and nothing is flagged', () => {
    for (const kind of ['simple', 'convergent'] as const)
      for (const s of SCHEDULES)
        for (const v of [FIG_VALUES, EXERCISE_6]) {
          const r = runInPlace(kind, v, { schedule: s });
          expect(r.outcome).toBe('correct');
          expect(r.issues).toEqual([]);
          expect(stateAt(r, r.steps.length).input).toEqual(arraysAfter(kind, v)[arraysAfter(kind, v).length - 1]);
        }
  });
  it('one release per iteration; every thread reaches every barrier', () => {
    const r = runInPlace('simple', FIG_VALUES, { schedule: 'in-order' });
    expect(r.steps.filter((s) => s.thread === -1)).toHaveLength(4);
    expect(r.program.every((ops) => ops.filter((o) => o.kind === 'barrier').length === 4)).toBe(true);
  });
  it('without it, thread 0 reads input[2] in iteration 2 before thread 1 has updated it (q-stale-read)', () => {
    const r = runInPlace('simple', FIG_VALUES, { schedule: 'in-order', barrier: false });
    expect(r.outcome).toBe('wrong');
    const first = r.issues[0];
    expect(first.kind).toBe('stale');
    expect(first.thread).toBe(0);
    const step = r.steps[firstIssueStep(r)! - 1];
    expect(step.op!.iter).toBe(2);
    expect(step.reads[1].index).toBe(2);
    expect(first.message).toContain("thread 1's partial from iteration 1");
  });
  it('without it, lockstep and highest-first still give the right total: the failure depends on the order', () => {
    expect(runInPlace('simple', FIG_VALUES, { schedule: 'lockstep', barrier: false }).outcome).toBe('correct');
    expect(runInPlace('simple', FIG_VALUES, { schedule: 'reversed', barrier: false }).outcome).toBe('correct');
    expect(runInPlace('simple', FIG_VALUES, { schedule: 'shuffled', barrier: false }).outcome).toBe('wrong');
  });
  it('exercise 6a thread by thread: input[4] = 17, output 36', () => {
    const r = runInPlace('simple', EXERCISE_6, { schedule: 'shuffled' });
    expect(r.output).toBe(36);
    expect(stateAt(r, r.steps.length).input[4]).toBe(17);
  });
  it('the history rows of Fig. 10.6: what each iteration wrote', () => {
    const r = runInPlace('simple', FIG_VALUES, { schedule: 'lockstep' });
    const rows = historyAt(r, r.steps.length);
    expect(rows.map((x) => x.written.filter(Boolean).length)).toEqual([0, 8, 4, 2, 1]);
    expect(rows[1].adds.map((a) => [a.own, a.partner])).toEqual([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]);
    expect(rows[4].values[0]).toBe(69);
    expect(rows.slice(1).map((x) => x.values)).toEqual(arraysAfter('simple', FIG_VALUES));
  });
  it('stateAt counts waiting threads, and the stepper\'s partial history follows the steps', () => {
    const r = runInPlace('simple', FIG_VALUES, { schedule: 'in-order' });
    const st = stateAt(r, 2);
    expect(st.status[0]).toBe('waiting');
    expect(st.status[1]).toBe('not started');
    expect(historyAt(r, 2)[1].written.filter(Boolean)).toHaveLength(1);
  });
});

describe('the barrier hunt (section 10.6), all-ones input: the sum must be 16', () => {
  const table = (v: Variant, values: number[]) => SCHEDULES.map((s) => runShared(v, values, s).outcome);
  const want: Record<Variant, Outcome[]> = {
    book: ['correct', 'correct', 'correct', 'correct'],
    bottom: ['correct', 'wrong', 'correct', 'wrong'],
    none: ['correct', 'wrong', 'correct', 'wrong'],
    'inside-if': ['deadlock', 'deadlock', 'deadlock', 'deadlock'],
    le: ['oob', 'oob', 'oob', 'oob'],
    two: ['correct', 'correct', 'correct', 'correct'],
    'two-race': ['correct', 'wrong', 'correct', 'wrong'],
  };
  for (const v of VARIANTS)
    it(`${v}: ${want[v].join(', ')}`, () => {
      expect(table(v, ones(16))).toEqual(want[v]);
    });
  it('the expected outputs: 16, and 0 for the sum of x - max', () => {
    expect(expectedOf('book', ones(16))).toBe(16);
    expect(expectedOf('two', ones(16))).toBe(0);
    expect(runShared('two', FIG_VALUES, 'shuffled').output).toBe(69 - 16 * 9);
  });
  it('barrier at the bottom: thread 0 reads input_s[4] before thread 4 has loaded it', () => {
    const r = runShared('bottom', ones(16), 'in-order');
    expect(r.issues[0].kind).toBe('stale');
    expect(r.issues[0].message).toContain('input_s[4]');
    expect(r.issues[0].message).toContain('nobody had written it yet');
    expect(Number.isNaN(r.output)).toBe(true);
  });
  it('barrier inside the if: threads 0 to 3 wait on line 7, threads 4 to 7 ended', () => {
    const r = runShared('inside-if', ones(16), 'lockstep');
    const d = r.issues.find((i) => i.kind === 'deadlock')!;
    expect(d.message).toContain('threads 0 to 3 wait at the barrier on line 7');
    expect(d.message).toContain('threads 4 to 7 ended without reaching it');
  });
  it('<=: thread 4 reads input_s[8], past the end of the 8-element array', () => {
    const r = runShared('le', ones(16), 'lockstep');
    const o = r.issues.find((i) => i.kind === 'oob')!;
    expect(o.thread).toBe(4);
    expect(o.message).toContain('input_s[8]');
  });
  it('two reductions without the barrier after the read: a slow thread finds the sum where the max was (write after read)', () => {
    const r = runShared('two-race', ones(16), 'in-order');
    const o = r.issues.find((i) => i.kind === 'overwritten')!;
    expect(o.thread).toBe(1);
    expect(o.message).toContain('needed the max in buf_s[0]');
    expect(r.output).toBe(14);
  });
});

describe('every highlighted line is the statement the model executes', () => {
  const inplace = labFile(INPLACE_FILE);
  const hunt = labFile(HUNT_FILE);
  const lineOf = (src: string, kernel: string, n: number) => {
    const f = functionSource(src, kernel);
    expect(f, kernel).not.toBeNull();
    expect(f!.text.split('\n')[0]).toContain(`void ${kernel}(`);
    return f!.text.split('\n')[n - 1];
  };
  const PATTERN: Record<string, RegExp> = {
    skip: /if \(/,
    add: /\+=/,
    max: /\? buf_s\[t\] : buf_s\[t \+ stride\]/,
    load: /input\[t\]/,
    'load-max': /input\[t\]|a > b \? a : b/,
    'load-diff': /\(a - m\) \+ \(b - m\)/,
    'read-max': /float m = buf_s\[0\];/,
    barrier: /__syncthreads\(\);/,
    store: /\*output = /,
  };
  it('s103_inplace.cu: the book\'s line numbers', () => {
    for (const kind of ['simple', 'convergent'] as const) {
      const k = INPLACE_KERNEL[kind];
      expect(lineOf(inplace, k, INPLACE_LINES.index)).toContain(kind === 'simple' ? 'i = 2*threadIdx.x;' : 'i = threadIdx.x;');
      expect(lineOf(inplace, k, INPLACE_LINES.loop)).toContain('for (unsigned int stride');
      expect(lineOf(inplace, k, INPLACE_LINES.test)).toContain(kind === 'simple' ? 'threadIdx.x % stride == 0' : 'threadIdx.x < stride');
      expect(lineOf(inplace, k, INPLACE_LINES.add)).toContain('input[i] += input[i + stride];');
      expect(lineOf(inplace, k, INPLACE_LINES.barrier)).toContain('__syncthreads();');
      expect(lineOf(inplace, k, INPLACE_LINES.storeTest)).toContain('threadIdx.x == 0');
      expect(lineOf(inplace, k, INPLACE_LINES.store)).toContain('*output = input[0];');
      for (const ops of runInPlace(kind, FIG_VALUES, { schedule: 'lockstep' }).program)
        for (const op of ops) for (const l of op.lines) expect(lineOf(inplace, k, l), `${kind} ${op.kind}`).toMatch(PATTERN[op.kind]);
    }
  });
  for (const v of VARIANTS)
    it(`s106_barrier_hunt.cu, ${VARIANT[v].kernel}`, () => {
      const r = runShared(v, FIG_VALUES, 'lockstep');
      for (const ops of r.program)
        for (const op of ops) for (const l of op.lines) expect(lineOf(hunt, VARIANT[v].kernel, l), `${v} ${op.kind} line ${l}`).toMatch(PATTERN[op.kind]);
      const barriers = hunt.split('\n').slice(functionSource(hunt, VARIANT[v].kernel)!.firstLine - 1, functionSource(hunt, VARIANT[v].kernel)!.lastLine)
        .filter((l) => l.includes('__syncthreads')).length;
      const sites = new Set(r.program.flat().filter((o) => o.kind === 'barrier').map((o) => o.lines[0]));
      expect(sites.size, 'every barrier in the source is modelled').toBe(barriers);
    });
  it('the 8-thread block of the hunt matches BLOCK_DIM in the file', () => {
    expect(hunt).toMatch(/#define BLOCK_DIM 8\b/);
    expect(strides('shared', 8)).toEqual([4, 2, 1]);
  });
});

// The three snippet files, run on the course's CPU emulator (labs/common/include/pmpp/emu)
// exactly as the lab runner builds the emu backend, must agree with the model. The
// emulator's "forward" order is the model's "each thread runs as far as it can, thread 0
// first"; "reverse" is the same from the last thread. Skipped when no C++ compiler works.
const compiler = ['c++', 'clang++', 'g++'].find((c) => spawnSync(c, ['--version']).status === 0);
describe.skipIf(!compiler)('the snippets on the emulator agree with the model', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch10-reduction-'));
  const exe = path.join(dir, 'reduction_check');
  const inc = path.join(repo, 'labs/common/include');
  const files = ['s103_inplace.cu', 's104_mirrored.cu', 's106_barrier_hunt.cu', 'check/reduction_check.cu'];
  const env = { ...process.env, ASAN_OPTIONS: 'detect_leaks=0' };
  const run = (kernel: string, values: number[]) => {
    const r = spawnSync(exe, [kernel, String(values.length / 2), ...values.map(String)], { encoding: 'utf8', env });
    return { status: r.status, out: r.stdout + r.stderr, lines: r.stdout.trim().split('\n').map((l) => l.split(' ')) };
  };
  it('builds', () => {
    const cu = files.map((f) => `#line 1 "${f}"\n` + fs.readFileSync(path.join(repo, 'labs/ch10/snippets', f), 'utf8')).join('\n');
    fs.writeFileSync(path.join(dir, 'all.cu'), cu);
    const t = spawnSync('python3', [path.join(repo, 'labs/tools/cu2emu.py'), path.join(dir, 'all.cu'), path.join(dir, 'all.cpp')], { encoding: 'utf8' });
    expect(t.status, t.stdout + t.stderr).toBe(0);
    const c = spawnSync(compiler!, ['-std=c++17', '-O1', '-g', '-fsanitize=address', '-fno-omit-frame-pointer', '-pthread', '-DPMPP_BACKEND_EMU',
      '-I', inc, '-I', path.join(inc, 'pmpp/emu/shim'), '-include', 'pmpp/emu/cuda_emu.h', path.join(dir, 'all.cpp'), '-o', exe], { encoding: 'utf8' });
    expect(c.status, c.stderr).toBe(0);
  }, 120_000);
  const mixed = Array.from({ length: 16 }, (_, k) => (k * 7) % 10);
  it('in-place and mirrored kernels: output and final array match in every order', () => {
    for (const [kernel, values] of [
      ['SimpleSumReductionKernel', EXERCISE_6], ['ConvergentSumReductionKernel', EXERCISE_6],
      ['SimpleSumReductionKernel', FIG_VALUES], ['ConvergentSumReductionKernel', FIG_VALUES],
    ] as const) {
      const kind = kernel === 'SimpleSumReductionKernel' ? 'simple' : 'convergent';
      const final = arraysAfter(kind, values)[arraysAfter(kind, values).length - 1];
      const r = run(kernel, [...values]);
      expect(r.status).toBe(0);
      expect(r.lines.map((l) => l[1])).toEqual(['forward', 'reverse', 'shuffle']);
      const total = values.reduce((a, b) => a + b, 0);
      for (const l of r.lines) expect(l.slice(2).map(Number)).toEqual([total, ...final]);
    }
    const m = runSpec(MIRROR_SPEC, FIG_VALUES);
    for (const l of run('MirroredSumReductionKernel', [...FIG_VALUES]).lines)
      expect(l.slice(2).map(Number)).toEqual([m.output, ...m.iterations[m.iterations.length - 1].after]);
  });
  for (const v of VARIANTS)
    it(`${VARIANT[v].kernel}: same outcome as the model in order and in reverse`, () => {
      for (const values of [ones(16), mixed]) {
        const r = run(VARIANT[v].kernel, values);
        if (v === 'le') {
          // AddressSanitizer stops the process on the read past input_s, on the add line.
          const f = functionSource(labFile(HUNT_FILE), VARIANT.le.kernel)!;
          expect(r.status).not.toBe(0);
          expect(r.out).toContain(`s106_barrier_hunt.cu:${f.firstLine + VARIANT.le.lines.add - 1}`);
          expect(runShared(v, values, 'in-order').outcome).toBe('oob');
          continue;
        }
        expect(r.status).toBe(0);
        for (const [line, schedule] of [[r.lines[0], 'in-order'], [r.lines[1], 'reversed']] as const) {
          const out = Number(line[2]);
          const model = runShared(v, values, schedule);
          if (v === 'inside-if') {
            // The emulator lets exited threads off, as NVIDIA's wording does; the model holds
            // the book's rule and reports the block stuck. The emulator's run passes.
            expect(model.outcome).toBe('deadlock');
            expect(out).toBe(expectedOf(v, values));
          } else if (model.outcome === 'correct') expect(out).toBe(model.expected);
          else if (Number.isNaN(model.output)) expect(out).not.toBe(model.expected);
          else expect(out).toBe(model.output);
        }
      }
    });
});
