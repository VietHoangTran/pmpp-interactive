import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  EXERCISE_1,
  EXERCISE_1_STEPS,
  FIG_11_3,
  FIG_11_5,
  FILE_VARIANTS,
  RACE,
  SCHEDULES,
  additionsAt,
  barriersPerScan,
  divergence,
  finalCount,
  firstFailure,
  kernelName,
  kernelSource,
  levels,
  naiveAdditions,
  program,
  rangeShort,
  rangeText,
  resultBuffer,
  run,
  sharedBytes,
  stateAt,
  strides,
  swaps,
  totalAdditions,
  treeAdders,
  treeOf,
  warpStates,
  type Config,
  type Schedule,
} from './ksblock';
import { functionSource } from './toolchain';

const cfg = (over: Partial<Config>, threads = 8): Config => ({ ...FIG_11_3, threads, ...over });
const SIXTEEN = Array.from({ length: 16 }, (_, k) => ((k * 5) % 7) + 1);

describe('Fig. 11.2 and exercise 1, level by level', () => {
  it('exercise 1: 4 6 7 1 2 8 5 2 after strides 1, 2 and 4 (p. 286)', () => {
    expect(levels(EXERCISE_1).slice(1).map((l) => l.values)).toEqual(EXERCISE_1_STEPS);
  });
  it('position 5 holds 18 after the stride-2 step (question q-ex1)', () => {
    expect(levels(EXERCISE_1)[2].values[5]).toBe(18);
  });
  it('the boxes of Fig. 11.2: what each position holds after each step', () => {
    const L = levels(Array.from({ length: 16 }, (_, i) => i + 1));
    const r = (lv: number, pos: number) => rangeText(L[lv].coef[pos]);
    expect([r(1, 1), r(1, 2), r(1, 15)]).toEqual(['x0..x1', 'x1..x2', 'x14..x15']);
    expect([r(2, 2), r(2, 3), r(2, 4), r(2, 15)]).toEqual(['x0..x2', 'x0..x3', 'x1..x4', 'x12..x15']);
    expect([r(3, 4), r(3, 7), r(3, 8), r(3, 15)]).toEqual(['x0..x4', 'x0..x7', 'x1..x8', 'x8..x15']);
    expect(L[4].coef.every((c, i) => rangeText(c) === (i === 0 ? 'x0' : `x0..x${i}`))).toBe(true);
  });
  it('after k steps the first 2^k positions are final: 8 of 16 after strides 1, 2, 4 (question q-final)', () => {
    expect(finalCount(16, 3)).toBe(8);
    const L = levels(SIXTEEN);
    const final = L[4].values;
    const settled = L[3].values.filter((v, i) => v === final[i] && i < 8).length;
    expect(settled).toBe(8);
  });
  it('49 adders for 16 elements: 15, 14, 12 and 8 per step (Fig. 11.2, p. 266)', () => {
    expect(strides(16).map((s) => additionsAt(16, s))).toEqual([15, 14, 12, 8]);
    expect(totalAdditions(16)).toBe(49);
    expect(totalAdditions(16)).toBe(16 * 4 - 15);
  });
  it('exercise 3: 20,481 additions as one 2048-element segment, 18,434 as two blocks of 1024', () => {
    expect(totalAdditions(2048)).toBe(20_481);
    expect(2 * totalAdditions(1024)).toBe(18_434);
  });
  it('the naive scan of p. 254: 523,776 additions for 1024 elements', () => {
    expect(naiveAdditions(1024)).toBe(523_776);
  });
  it('the reduction tree behind y15 is the heavy tree of Fig. 11.2: 8, 4, 2 boxes, then y15', () => {
    const t = treeOf(15, 16);
    const at = (lv: number) => t.nodes.filter((n) => n.level === lv).map((n) => n.pos).sort((a, b) => a - b);
    expect(at(1)).toEqual([1, 3, 5, 7, 9, 11, 13, 15]);
    expect(at(2)).toEqual([3, 7, 11, 15]);
    expect(at(3)).toEqual([7, 15]);
    expect(at(4)).toEqual([15]);
    expect(treeAdders(15, 16)).toBe(15);
  });
});

describe('exercise 2: which warps diverge', () => {
  it('only warp 0, and only while the stride is below the warp width: 5 iterations at 32, 6 at 64', () => {
    expect(divergence(256, 32)).toEqual({ iterations: 5, pairs: 5, strides: [1, 2, 4, 8, 16] });
    expect(divergence(256, 64)).toEqual({ iterations: 6, pairs: 6, strides: [1, 2, 4, 8, 16, 32] });
    expect(divergence(1024, 32).iterations).toBe(5);
  });
  it('at stride 4 warp 0 is divergent with 28 of 32 lanes active and the rest full; at stride 64 warps 0 and 1 are idle', () => {
    const w = warpStates(256, 32, 4);
    expect(w[0]).toMatchObject({ state: 'divergent', activeLanes: 28 });
    expect(w.slice(1).every((x) => x.state === 'full')).toBe(true);
    expect(warpStates(256, 32, 64).map((x) => x.state)).toEqual(['idle', 'idle', 'full', 'full', 'full', 'full', 'full', 'full']);
  });
});

describe('the thread-level run of Fig. 11.3', () => {
  it('in lockstep it passes through the levels of Fig. 11.2, barrier by barrier', () => {
    const tr = run(cfg({}), EXERCISE_1, 'lockstep');
    expect(tr.correct).toBe(true);
    const L = levels(EXERCISE_1);
    const opened = tr.steps.filter((s) => s.op === 'release' && s.barrier === 1);
    // The barrier-1 release at iteration k + 1 comes after every write of iteration k.
    for (let k = 0; k + 1 < opened.length; k++) {
      const st = stateAt(tr, opened[k + 1].index + 1);
      expect(st.buffers[0].map((s) => s.value)).toEqual(L[k + 1].values);
    }
    expect(tr.output).toEqual(L[L.length - 1].values);
    expect(tr.additions).toEqual([7, 6, 4]);
  });
  it('with both barriers it is correct in every order', () => {
    for (const s of SCHEDULES) {
      expect(run(cfg({}), EXERCISE_1, s).correct, s).toBe(true);
      expect(run(cfg({}, 16), SIXTEEN, s).correct, s).toBe(true);
    }
  });
  it('the book\'s race (pp. 257-258): without barrier 2, thread 4 stores before thread 6 reads, and x1 and x2 are counted twice', () => {
    const tr = run(cfg({ barrier2: false }), EXERCISE_1, 'book-race');
    expect(tr.wrong).toEqual([6]);
    expect(tr.violations).toHaveLength(1);
    expect(tr.violations[0]).toMatchObject({ kind: 'overwritten-before-read', thread: 6, slot: 4, iter: RACE.iter });
    expect(rangeText(tr.outputCoef[6])).toBe('x0 + 2x1 + 2x2 + x3 + x4 + x5 + x6');
    expect(rangeShort(tr.outputCoef[6])).toBe('x0..x6 +x1+x2');
    expect(rangeShort(tr.outputCoef[5])).toBe('x0..x5');
    expect(rangeShort(null)).toBe('?');
    // Exercise 1's values: 4 + 12 + 14 + 1 + 2 + 8 + 5 = 46 instead of 33.
    expect(tr.output[6]).toBe(46);
    expect(tr.reference[6]).toBe(33);
    // After the stride-2 step position 6 holds x1..x6.
    const after = tr.steps.find((s) => s.thread === 6 && s.op === 'write' && s.iter === 1)!;
    expect(rangeText(after.write!.coef)).toBe('x1..x6');
  });
  it('the same order with barrier 2 in place is correct', () => {
    expect(run(cfg({}), EXERCISE_1, 'book-race').correct).toBe(true);
  });
  it('the outcome table of section 11.2', () => {
    const row = (over: Partial<Config>) => Object.fromEntries(SCHEDULES.map((s) => [s, run(cfg(over), EXERCISE_1, s).correct]));
    expect(row({})).toEqual({ lockstep: true, 'in-order': true, reversed: true, shuffled: true, 'book-race': true });
    expect(row({ barrier2: false })).toEqual({ lockstep: true, 'in-order': false, reversed: true, shuffled: false, 'book-race': false });
    expect(row({ barrier1: false })).toEqual({ lockstep: true, 'in-order': true, reversed: false, shuffled: false, 'book-race': true });
    expect(row({ temp: false })).toEqual({ lockstep: true, 'in-order': false, reversed: true, shuffled: false, 'book-race': false });
  });
  it('without barrier 2, in order, the first failure is thread 2 reading a slot thread 1 already overwrote', () => {
    const tr = run(cfg({ barrier2: false }), EXERCISE_1, 'in-order');
    expect(tr.violations[0]).toMatchObject({ kind: 'overwritten-before-read', thread: 2, slot: 1, iter: 0 });
    expect(firstFailure(tr)).toBe(tr.violations[0].step + 1);
    expect(tr.output).toEqual([4, 10, 21, 28, 45, 66, 95, 119]);
  });
  it('without barrier 1, last thread first, threads read slots that have not been loaded yet', () => {
    const tr = run(cfg({ barrier1: false }), EXERCISE_1, 'reversed');
    expect(tr.violations[0]).toMatchObject({ kind: 'read-before-write', thread: 7, slot: 6, iter: 0 });
    expect(Number.isNaN(tr.output[7])).toBe(true);
  });
  it('a program puts the second barrier between read and write only when there is a temp', () => {
    expect(program(cfg({}), 3).map((i) => i.op).slice(0, 6)).toEqual(['load', 'barrier1', 'read', 'barrier2', 'write', 'barrier1']);
    expect(program(cfg({ temp: false }), 3).map((i) => i.op).slice(0, 6)).toEqual(['load', 'barrier1', 'read', 'write', 'barrier2', 'barrier1']);
    expect(program(cfg({}), 0).filter((i) => i.op === 'read')).toHaveLength(0);
  });
});

describe('Fig. 11.5: two buffers', () => {
  const dbl = (over: Partial<Config> = {}, threads = 8): Config => ({ ...FIG_11_5, threads, ...over });
  it('one barrier per iteration and correct in every order, including the book\'s race', () => {
    for (const s of SCHEDULES) {
      expect(run(dbl(), EXERCISE_1, s).correct, s).toBe(true);
      expect(run(dbl({}, 16), SIXTEEN, s).correct, s).toBe(true);
    }
    expect(program(dbl(), 5).filter((i) => i.op === 'barrier1')).toHaveLength(3);
    expect(barriersPerScan(1024, 'double')).toBe(10);
    expect(barriersPerScan(1024, 'single')).toBe(20);
  });
  it('without the copy it is wrong in every order, the same way each time: a missing copy, not a race', () => {
    const outs = SCHEDULES.map((s) => run(dbl({ copy: false }), EXERCISE_1, s));
    for (const tr of outs) {
      expect(tr.correct).toBe(false);
      expect(tr.wrong).toEqual([0, 2, 3, 5, 6]);
      expect(new Set(tr.violations.map((v) => v.kind))).toEqual(new Set(['missing-copy']));
    }
    // Position 3 ends with x2..x3 from iteration 0: a value from two iterations earlier.
    expect(rangeText(outs[0].outputCoef[3])).toBe('x2..x3');
    expect(Number.isNaN(outs[0].output[0])).toBe(true);
  });
  it('one buffer with barrier 1 only: the race comes back (single buffer, no barrier 2)', () => {
    expect(run(cfg({ barrier2: false }), EXERCISE_1, 'in-order').correct).toBe(false);
  });
  it('the answer sits in inBuffer_s: buffer2_s after an odd number of iterations, buffer1_s after an even one', () => {
    expect(resultBuffer(8)).toBe(2);
    expect(resultBuffer(256)).toBe(1);
    expect(resultBuffer(512)).toBe(2);
    expect(resultBuffer(1024)).toBe(1);
    expect(run(dbl(), EXERCISE_1, 'lockstep').resultBuf).toBe(1);
    expect(run(dbl({}, 16), SIXTEEN, 'lockstep').resultBuf).toBe(0);
    expect(swaps(256)).toHaveLength(8);
    expect(swaps(8).map((s) => [s.inBuf, s.outBuf])).toEqual([[0, 1], [1, 0], [0, 1]]);
  });
  it('shared memory: 8192 bytes for 1024 floats twice, against 4096', () => {
    expect(sharedBytes(1024, 'double')).toBe(8192);
    expect(sharedBytes(1024, 'single')).toBe(4096);
  });
  it('the copy branch copies 1, 2 and 4 values in the three iterations of an 8-thread block (Fig. 11.4)', () => {
    const tr = run(dbl(), EXERCISE_1, 'lockstep');
    expect([0, 1, 2].map((k) => tr.steps.filter((s) => s.op === 'copy' && s.iter === k).length)).toEqual([1, 2, 4]);
  });
});

// ---------------------------------------------------------------- the snippet files
const repo = path.resolve(__dirname, '../../..');
const snippet = (f: string) => fs.readFileSync(path.join(repo, 'labs/ch11/snippets', f), 'utf8');

describe('the source the steppers print is the snippet files\' text', () => {
  for (const v of FILE_VARIANTS)
    it(`${kernelName(v.cfg)} in ${v.file}`, () => {
      const f = functionSource(snippet(v.file), kernelName(v.cfg));
      expect(f, 'function found').not.toBeNull();
      expect(f!.text).toBe(kernelSource(v.cfg).code);
    });
  it('Fig. 11.3\'s lines are the book\'s: barriers on 10 and 14, read 13, write 16, store 19', () => {
    expect(kernelSource(FIG_11_3).line).toMatchObject({ barrier1: [10], read: [13], barrier2: [14], write: [16], store: [19] });
    expect(kernelSource(FIG_11_3).code.split('\n')).toHaveLength(21);
  });
  it('Fig. 11.5\'s lines are the book\'s: load 7, barrier 9, add 11-12, copy 14, swap 16-18, store 21', () => {
    expect(kernelSource(FIG_11_5).line).toMatchObject({ load: [7], barrier1: [9], add: [11, 12], copy: [14], swap: [16, 17, 18], store: [21] });
    expect(kernelSource(FIG_11_5).code.split('\n')).toHaveLength(23);
  });
  it('the printed typo "__ syncthreads" is fixed in the kernel', () => {
    expect(functionSource(snippet('s112_kogge_stone.cu'), 'kogge_stone_scan_kernel')!.text).not.toContain('__ syncthreads');
    expect(kernelSource(FIG_11_3).code).toContain('        __syncthreads();');
  });
});

// The snippet files, run on the course's CPU emulator (labs/common/include/pmpp/emu) as the
// lab runner builds the emu backend, must agree with the model. The emulator's "forward"
// order is the model's "each thread runs as far as it can, thread 0 first"; "reverse" is
// the same from the last thread. Its uninitialised shared memory holds the bytes 0xCD
// (a large negative float) where the model has NaN. Skipped when no C++ compiler works.
const compiler = ['c++', 'clang++', 'g++'].find((c) => spawnSync(c, ['--version']).status === 0);
describe.skipIf(!compiler)('the snippets on the emulator agree with the model', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch11-ks-'));
  const exe = path.join(dir, 's112_check');
  const inc = path.join(repo, 'labs/common/include');
  const env = { ...process.env, ASAN_OPTIONS: 'detect_leaks=0' };
  it('builds', () => {
    const files = ['s112_kogge_stone.cu', 's112_race_variants.cu', 'check/s112_check.cu'];
    const cu = files.map((f) => `#line 1 "${f}"\n` + snippet(f)).join('\n');
    fs.writeFileSync(path.join(dir, 'all.cu'), cu);
    const t = spawnSync('python3', [path.join(repo, 'labs/tools/cu2emu.py'), path.join(dir, 'all.cu'), path.join(dir, 'all.cpp')], { encoding: 'utf8' });
    expect(t.status, t.stdout + t.stderr).toBe(0);
    const c = spawnSync(compiler!, ['-std=c++17', '-O1', '-g', '-fsanitize=address', '-fno-omit-frame-pointer', '-pthread', '-DPMPP_BACKEND_EMU', '-DSEG_SIZE=16',
      '-I', inc, '-I', path.join(inc, 'pmpp/emu/shim'), '-include', 'pmpp/emu/cuda_emu.h', path.join(dir, 'all.cpp'), '-o', exe], { encoding: 'utf8' });
    expect(c.status, c.stderr).toBe(0);
  }, 180_000);
  const emu = (kernel: string, values: number[]) => {
    const r = spawnSync(exe, [kernel, ...values.map(String)], { encoding: 'utf8', env });
    expect(r.status, r.stderr).toBe(0);
    return Object.fromEntries(r.stdout.trim().split('\n').map((l) => {
      const [, order, ...vals] = l.split(' ');
      return [order, vals.map(Number)];
    })) as Record<'forward' | 'reverse' | 'shuffle', number[]>;
  };
  for (const v of FILE_VARIANTS)
    it(`${kernelName(v.cfg)}: forward and reverse match the model's two orders`, () => {
      for (const values of [EXERCISE_1, SIXTEEN]) {
        const lines = emu(kernelName(v.cfg), values);
        for (const [order, schedule] of [['forward', 'in-order'], ['reverse', 'reversed']] as [keyof typeof lines, Schedule][]) {
          const m = run({ ...v.cfg, threads: values.length }, values, schedule);
          lines[order].forEach((got, i) => {
            if (m.outputCoef[i]) expect(got, `${order} position ${i}`).toBe(m.output[i]);
            else expect(got, `${order} position ${i} read garbage`).toBeLessThan(-1e8);
          });
        }
        // A kernel the model finds correct in every order must be correct in the emulator's shuffled order too.
        const every = SCHEDULES.map((s) => run({ ...v.cfg, threads: values.length }, values, s));
        if (every.every((t) => t.correct)) expect(lines.shuffle).toEqual(every[0].reference);
      }
    });
});
