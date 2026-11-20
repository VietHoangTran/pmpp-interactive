import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  EX6,
  FIG116,
  FIG_11_20_PRINTED,
  additions,
  addersPerStep,
  bookKoggeStoneSteps,
  breakEven,
  brentKung,
  chargeOf,
  closedForm,
  finalAfter,
  finalAt,
  inclusiveScan,
  koggeStone,
  lg,
  loads,
  needs,
  needsChain,
  printedDiffers,
  rangeLabel,
  ranges,
  readWriteOverlap,
  reverseLevels,
  run,
  sequential,
  slotsTotal,
  span,
  stepsWithUnits,
  threadsOf,
  upSteps,
  warpView,
} from './prefixnet';

const repo = path.resolve(__dirname, '../../..');
const POW2 = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
const ints = (n: number, seed = 1) => Array.from({ length: n }, (_, k) => ((k * 7 + seed * 3) % 11) - 3);

describe('the networks are the book\'s figures', () => {
  it('Fig. 11.2: Kogge-Stone on 16 positions has 15, 14, 12 and 8 adders, 49 in all', () => {
    const ks = koggeStone(16);
    expect(addersPerStep(ks)).toEqual([15, 14, 12, 8]);
    expect(additions(ks)).toBe(49);
    expect(ks.steps.map((s) => s.stride)).toEqual([1, 2, 4, 8]);
    for (const s of ks.steps) for (const a of s.adders) expect(a.src).toBe(a.dst - s.stride);
  });
  it('Fig. 11.19: Brent-Kung on 16 positions has 8, 4, 2, 1, then 1, 3, 7 adders, 26 in 7 steps', () => {
    const bk = brentKung(16);
    expect(addersPerStep(bk)).toEqual([8, 4, 2, 1, 1, 3, 7]);
    expect([additions(bk), bk.steps.length, upSteps(bk)]).toEqual([26, 7, 4]);
    const rows = bk.steps.map((s) => s.adders.map((a) => [a.src, a.dst]));
    expect(rows[0]).toEqual([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]);
    expect(rows[1]).toEqual([[1, 3], [5, 7], [9, 11], [13, 15]]);
    expect(rows[2]).toEqual([[3, 7], [11, 15]]);
    expect(rows[3]).toEqual([[7, 15]]);
    expect(rows[4]).toEqual([[7, 11]]);
    expect(rows[5]).toEqual([[3, 5], [7, 9], [11, 13]]);
    expect(rows[6]).toEqual([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]]);
    expect(bk.steps.slice(4).map((s) => s.level)).toEqual([1, 2, 3]);
  });
  it('the reduction tree updates the positions of the form 2n - 1, 4n - 1, 8n - 1, 16n - 1 (p. 282)', () => {
    const bk = brentKung(16);
    bk.steps.slice(0, 4).forEach((s, k) => {
      for (const a of s.adders) expect((a.dst + 1) % 2 ** (k + 1)).toBe(0);
    });
  });
  it('exercise 1: Kogge-Stone on 4 6 7 1 2 8 5 2, step by step', () => {
    const st = run(koggeStone(8), EX6);
    expect(st.slice(1)).toEqual([
      [4, 10, 13, 8, 3, 10, 13, 7],
      [4, 10, 17, 18, 16, 18, 16, 17],
      [4, 10, 17, 18, 20, 28, 33, 35],
    ]);
  });
  it('exercise 6: Brent-Kung on 4 6 7 1 2 8 5 2; position 5 holds 28 after the first reverse level', () => {
    const st = run(brentKung(8), EX6);
    expect(st.slice(1)).toEqual([
      [4, 10, 7, 8, 2, 10, 5, 7],
      [4, 10, 7, 18, 2, 10, 5, 17],
      [4, 10, 7, 18, 2, 10, 5, 35],
      [4, 10, 7, 18, 2, 28, 5, 35],
      [4, 10, 17, 18, 20, 28, 33, 35],
    ]);
  });
  it('Fig. 11.6: both networks give the scan the book prints for its sixteen values', () => {
    const want = [2, 3, 6, 7, 7, 11, 12, 14, 14, 17, 18, 20, 25, 28, 29, 31];
    expect(inclusiveScan(FIG116)).toEqual(want);
    for (const net of [koggeStone(16), brentKung(16), sequential(16)]) expect(run(net, FIG116).at(-1)).toEqual(want);
  });
  it('every network scans any input, joins only adjacent ranges, and ends with x0..xi everywhere', () => {
    for (const n of POW2)
      for (const net of [koggeStone(n), brentKung(n), sequential(n)]) {
        const v = ints(n);
        expect(run(net, v).at(-1), `${net.id} ${n}`).toEqual(inclusiveScan(v));
        const last = ranges(net).at(-1)!;
        last.forEach((r, i) => expect([r.lo, r.hi]).toEqual([0, i]));
      }
  });
  it('after the reduction tree positions 0, 1, 3, 7 and 15 are final (p. 283)', () => {
    const bk = brentKung(16);
    expect(finalAfter(bk, 4)).toEqual([0, 1, 3, 7, 15]);
    expect(finalAt(bk)).toEqual([0, 1, 7, 2, 7, 6, 7, 3, 7, 6, 7, 5, 7, 6, 7, 4]);
    expect(finalAt(koggeStone(16))).toEqual([0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4]);
  });
});

describe('counts and closed forms', () => {
  it('additions match the closed forms, steps match the span', () => {
    for (const n of POW2) {
      expect(additions(koggeStone(n))).toBe(closedForm('kogge-stone', n));
      expect(additions(brentKung(n))).toBe(closedForm('brent-kung', n));
      expect(additions(sequential(n))).toBe(closedForm('sequential', n));
      expect(koggeStone(n).steps.length).toBe(span('kogge-stone', n));
      expect(brentKung(n).steps.length).toBe(span('brent-kung', n));
    }
    expect(closedForm('naive', 512)).toBe(512 * 511 / 2);
  });
  it('p. 266: 512 elements, 4097 additions, between 8 and 9 times the loop\'s 511; 512/9 = 56.9 times fewer steps', () => {
    expect(closedForm('kogge-stone', 512)).toBe(4097);
    const ratio = closedForm('kogge-stone', 512) / closedForm('sequential', 512);
    expect(ratio).toBeGreaterThan(8);
    expect(ratio).toBeLessThan(9);
    expect(ratio).toBeCloseTo(8.02, 2);
    expect(Number((span('sequential', 512) / span('kogge-stone', 512)).toFixed(1))).toBe(56.9);
  });
  it('p. 284: Brent-Kung on 1024 elements does 2036 additions in 19 steps (10 up, 9 down)', () => {
    const bk = brentKung(1024);
    expect([additions(bk), bk.steps.length, upSteps(bk)]).toEqual([2036, 19, 10]);
    expect(closedForm('brent-kung', 1024)).toBe(2036);
  });
  it('the reverse tree does N - 1 - log2N additions (p. 284)', () => {
    for (const n of POW2.slice(1)) {
      const bk = brentKung(n);
      expect(bk.steps.slice(upSteps(bk)).reduce((s, x) => s + x.adders.length, 0)).toBe(n - 1 - lg(n));
    }
  });
});

describe('loads, charges and steps with P units', () => {
  it('the analytic loads are the networks\' adders, done by a contiguous run of threads', () => {
    for (const n of POW2.slice(1))
      for (const id of ['kogge-stone', 'brent-kung'] as const) {
        const net = id === 'kogge-stone' ? koggeStone(n) : brentKung(n);
        const ls = loads(id, n);
        expect(ls.length).toBe(net.steps.length);
        net.steps.forEach((s, k) => {
          const th = s.adders.map((a) => a.thread).sort((a, b) => a - b);
          expect([ls[k].adds, ls[k].lo, ls[k].hi]).toEqual([th.length, th[0], th[th.length - 1]]);
          expect(th[th.length - 1] - th[0] + 1).toBe(th.length);
          expect(th[th.length - 1]).toBeLessThan(threadsOf(id, n));
        });
        expect(net.threads).toBe(threadsOf(id, n));
      }
  });
  it('p. 266: 1024 elements, 32 units: 320 steps, 3.2 times fewer than the loop\'s 1024', () => {
    expect(stepsWithUnits('kogge-stone', 1024, 32, 'threads')).toBe(320);
    expect(bookKoggeStoneSteps(1024, 32)).toBe(320);
    expect(stepsWithUnits('sequential', 1024, 32, 'threads')).toBe(1024);
    expect(1024 / stepsWithUnits('kogge-stone', 1024, 32, 'threads')).toBe(3.2);
  });
  it('the every-thread rule is the book\'s (N·log2N)/P whenever P divides N, and log2N once P >= N', () => {
    for (const n of POW2.slice(2))
      for (let p = 1; p <= n; p *= 2) expect(stepsWithUnits('kogge-stone', n, p, 'threads')).toBe(bookKoggeStoneSteps(n, p));
    expect(stepsWithUnits('kogge-stone', 512, 4096, 'threads')).toBe(9);
    expect(stepsWithUnits('brent-kung', 512, 4096, 'adds')).toBe(17);
  });
  it('2048 elements on 64 units: 352 steps, 5.8 times fewer', () => {
    expect(stepsWithUnits('kogge-stone', 2048, 64, 'threads')).toBe(352);
    expect(2048 / 352).toBeCloseTo(5.82, 2);
  });
  it('break-even: 13 units for 4096 elements (12 is not enough), and P > log2N in the book\'s formula', () => {
    expect(breakEven('kogge-stone', 4096, 'threads')).toBe(13);
    expect(stepsWithUnits('kogge-stone', 4096, 12, 'threads')).toBeGreaterThanOrEqual(4096);
    expect(bookKoggeStoneSteps(4096, 13)).toBeLessThan(4096);
    expect(bookKoggeStoneSteps(4096, 12)).toBe(4096);
    // The rounded-up rule agrees with the formula whenever log2N + 1 divides N closely enough;
    // it never needs fewer units than the formula says.
    for (let k = 3; k <= 20; k++) expect(breakEven('kogge-stone', 2 ** k, 'threads')!).toBeGreaterThanOrEqual(k + 1);
  });
  it('charging only the additions: 289 steps at 1024 and 32; Brent-Kung beats the loop from 2 units', () => {
    expect(stepsWithUnits('kogge-stone', 1024, 32, 'adds')).toBe(289);
    expect(stepsWithUnits('brent-kung', 1024, 32, 'adds')).toBe(71);
    expect(stepsWithUnits('brent-kung', 1024, 2, 'adds')).toBe(1023);
    expect(breakEven('brent-kung', 1024, 'adds')).toBe(2);
    expect(stepsWithUnits('brent-kung', 1024, 32, 'threads')).toBe(19 * 16);
  });
  it('whole warps: Kogge-Stone\'s idle threads are at the bottom of the block, so only warp 0 is part idle', () => {
    expect(stepsWithUnits('kogge-stone', 1024, 32, 'warps', 32)).toBe(289);
    expect(chargeOf({ adds: 1023, lo: 1, hi: 1023 }, 'warps', 1024, 32)).toBe(1024);
    expect(chargeOf({ adds: 992, lo: 32, hi: 1023 }, 'warps', 1024, 32)).toBe(992);
    expect(chargeOf({ adds: 3, lo: 0, hi: 2 }, 'warps', 512, 64)).toBe(64);
  });
  it('resource use: N·log2N lane slots when every thread is charged, the closed form when only additions are', () => {
    for (const n of POW2.slice(1)) {
      expect(slotsTotal('kogge-stone', n, 'threads')).toBe(n * lg(n));
      expect(slotsTotal('kogge-stone', n, 'adds')).toBe(closedForm('kogge-stone', n));
      expect(slotsTotal('brent-kung', n, 'adds')).toBe(closedForm('brent-kung', n));
      const w = slotsTotal('kogge-stone', n, 'warps', 32);
      expect(w).toBeGreaterThanOrEqual(closedForm('kogge-stone', n));
      expect(w).toBeLessThanOrEqual(n * lg(n) + 32 * lg(n));
    }
  });
  it('fewer units never means fewer steps', () => {
    for (const charge of ['adds', 'warps', 'threads'] as const)
      for (const id of ['kogge-stone', 'brent-kung'] as const) {
        let prev = Infinity;
        for (let p = 1; p <= 4096; p += p < 64 ? 1 : 37) {
          const s = stepsWithUnits(id, 4096, p, charge, 32);
          expect(s).toBeLessThanOrEqual(prev);
          prev = s;
        }
      }
  });
});

describe('one barrier per step: reads and writes of a step', () => {
  it('Brent-Kung never writes, in one step, a position another adder of that step reads', () => {
    for (const n of POW2) for (const s of brentKung(n).steps) expect(readWriteOverlap(s)).toEqual([]);
  });
  it('Kogge-Stone does, in every step but its last', () => {
    for (const n of POW2.slice(2)) {
      const ks = koggeStone(n);
      ks.steps.forEach((s, k) => {
        if (k < ks.steps.length - 1) expect(readWriteOverlap(s).length).toBeGreaterThan(0);
        else expect(readWriteOverlap(s)).toEqual([]);
      });
    }
    expect(readWriteOverlap(koggeStone(16).steps[1])).toEqual([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
  });
});

describe('inside one warp (pp. 284-285)', () => {
  it('32 lanes: Kogge-Stone 129 additions in 5 steps, 160 lane slots; Brent-Kung 57 in 9, 288', () => {
    const ks = warpView('kogge-stone', 32);
    const bk = warpView('brent-kung', 32);
    expect(ks.steps.map((s) => s.adds)).toEqual([31, 30, 28, 24, 16]);
    expect([ks.additions, ks.finishesAt, ks.slots, ks.idleSlots]).toEqual([129, 5, 160, 31]);
    expect(bk.steps.map((s) => s.adds)).toEqual([16, 8, 4, 2, 1, 1, 3, 7, 15]);
    expect([bk.additions, bk.finishesAt, bk.slots, bk.idleSlots]).toEqual([57, 9, 288, 231]);
  });
  it('64 lanes: 321 in 6 against 120 in 11', () => {
    const ks = warpView('kogge-stone', 64);
    const bk = warpView('brent-kung', 64);
    expect([ks.additions, ks.finishesAt, ks.slots]).toEqual([321, 6, 384]);
    expect([bk.additions, bk.finishesAt, bk.slots]).toEqual([120, 11, 704]);
  });
  it('the adding lanes are the destination positions', () => {
    expect(warpView('brent-kung', 8).steps.map((s) => s.lanes)).toEqual([[1, 3, 5, 7], [3, 7], [7], [5], [2, 4, 6]]);
    expect(warpView('kogge-stone', 8).steps[2].lanes).toEqual([4, 5, 6, 7]);
  });
});

describe('Fig. 11.20', () => {
  const rows = reverseLevels(16);
  it('rows Initial, Level 1 to 3 hold the ranges the book prints', () => {
    expect(rows.map((r) => r.label)).toEqual(['Initial', 'Level 1', 'Level 2', 'Level 3']);
    expect(rows[0].ranges.map(rangeLabel)).toEqual([
      'x0', 'x0..x1', 'x2', 'x0..x3', 'x4', 'x4..x5', 'x6', 'x0..x7',
      'x8', 'x8..x9', 'x10', 'x8..x11', 'x12', 'x12..x13', 'x14', 'x0..x15',
    ]);
    expect(rows[1].changed).toEqual([11]);
    expect(rangeLabel(rows[1].ranges[11])).toBe('x0..x11');
    expect(rows[2].changed).toEqual([5, 9, 13]);
    expect(rows[2].changed.map((i) => rangeLabel(rows[2].ranges[i]))).toEqual(['x0..x5', 'x0..x9', 'x0..x13']);
    expect(rows[3].changed).toEqual([2, 4, 6, 8, 10, 12, 14]);
    expect(rows[3].needs.every((v) => v === 0)).toBe(true);
  });
  it('position 14 needs 13 (x12..x13), 11 (x8..x11) and 7 (x0..x7), 1, 2 and 4 places apart; position 13 later takes 11', () => {
    expect(needsChain(rows[0].ranges, 14)).toEqual([13, 11, 7]);
    expect(rows[0].ranges[13]).toEqual({ lo: 12, hi: 13 });
    expect(needsChain(rows[1].ranges, 13)).toEqual([11]);
    expect(rangeLabel(rows[1].ranges[11])).toBe('x0..x11');
  });
  it('the printed shading agrees with the ranges except columns 9 and 11 of Initial and column 9 of Level 1', () => {
    expect(printedDiffers()).toEqual([
      { row: 0, col: 9, printed: 2, needs: 1 },
      { row: 0, col: 11, printed: 2, needs: 1 },
      { row: 1, col: 9, printed: 2, needs: 1 },
    ]);
    expect(FIG_11_20_PRINTED[0][14]).toBe(3);
  });
  it('no position ever needs more than log2N - 1 partial sums, from positions a power of two apart (p. 284)', () => {
    for (const n of POW2.slice(1)) {
      const st = ranges(brentKung(n))[lg(n)];
      expect(Math.max(...needs(st))).toBe(Math.max(0, lg(n) - 1));
      for (let i = 0; i < n; i++) {
        // Position 14: 13, 11, 7, gaps 1, 2, 4. Position 12: 11, 7, gaps 1, 4. Every gap is a
        // power of two and each is larger than the one before.
        const chain = [i, ...needsChain(st, i)];
        const gaps = chain.slice(1).map((c, k) => chain[k] - c);
        gaps.forEach((g, k) => {
          expect(Number.isInteger(Math.log2(g))).toBe(true);
          if (k > 0) expect(g).toBeGreaterThan(gaps[k - 1]);
        });
      }
    }
  });
  it('each reverse level completes exactly the positions it writes', () => {
    for (const r of rows.slice(1)) for (const i of r.changed) expect(r.needs[i]).toBe(0);
  });
});

// ---------------------------------------------------------------- the lesson's kernel on the emulator
// labs/ch11/snippets/s1110_brent_kung.cu, appended to check/s1110_check.cu, translated by
// labs/tools/cu2emu.py and built as the lab runner builds the emu backend (AddressSanitizer,
// three thread orders) and the race backend (ThreadSanitizer). A negative control is made
// from the snippet's own text by deleting the barrier at the top of both loops. Skipped
// when no C++ compiler or python3 is found.
const SNIPPET = path.join(repo, 'labs/ch11/snippets/s1110_brent_kung.cu');
const DRIVER = path.join(repo, 'labs/ch11/snippets/check/s1110_check.cu');
const snippet = fs.readFileSync(SNIPPET, 'utf8');

describe('the snippet is the model\'s network', () => {
  it('thread t performs the t-th adder at (t + 1)·2·stride - 1, in both trees, with SEG_SIZE/2 threads', () => {
    expect(snippet).toContain('unsigned int idx = (threadIdx.x + 1)*2*stride - 1;');
    expect(snippet).toContain('buffer_s[idx] += buffer_s[idx - stride];');
    expect(snippet).toContain('buffer_s[idx + stride] += buffer_s[idx];');
    expect(snippet).toContain('for(unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {');
    expect(snippet).toContain('for(unsigned int stride = SEG_SIZE/4; stride > 0; stride /= 2) {');
    expect(fs.readFileSync(DRIVER, 'utf8')).toContain('kernel<<<blocks, SEG_SIZE / 2>>>(input, output, n);');
    expect(snippet.split('\n').filter((l) => l.includes('__syncthreads();'))).toHaveLength(3);
  });
  it('the lesson loads it from the file, so there is no copy to drift', () => {
    const page = fs.readFileSync(path.join(repo, 'app/src/chapters/ch11/S1110.tsx'), 'utf8');
    expect(page).toContain("'ch11/snippets/s1110_brent_kung.cu'");
  });
});

const has = (cmd: string, args: string[]) => spawnSync(cmd, args).status === 0;
const compiler = ['c++', 'clang++', 'g++'].find((c) => has(c, ['--version']));
const python = has('python3', ['--version']);
describe.skipIf(!compiler || !python)('the snippet on the emulator agrees with the model', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch11-prefixnet-'));
  const inc = path.join(repo, 'labs/common/include');
  const env = {
    ...process.env,
    ASAN_OPTIONS: 'detect_leaks=0',
    TSAN_OPTIONS: 'abort_on_error=0:exitcode=66:halt_on_error=0:report_thread_leaks=0',
  };
  const variant = snippet
    .replace(/\n\s*__syncthreads\(\);\n(\s*unsigned int idx)/g, '\n$1')
    .replace('brent_kung_scan_kernel', 'brent_kung_no_barrier_kernel')
    .replace('#ifndef SEG_SIZE\n#define SEG_SIZE 2048\n#endif\n', '');
  const exe = (flavour: string, seg: number) => path.join(dir, `${flavour}-${seg}`);
  let raceBuilt = false;
  const lines = (r: { stdout: string }) => r.stdout.trim().split('\n').map((l) => l.split(' '));
  const runExe = (file: string, kernel: string, values: number[]) =>
    spawnSync(file, [kernel, ...values.map(String)], { encoding: 'utf8', env, timeout: 120_000 });
  /** Each block scans its own SEG_SIZE segment. */
  const blockScan = (values: number[], seg: number) =>
    values.map((_, i) => {
      const start = i - (i % seg);
      return inclusiveScan(values.slice(start, i + 1)).at(-1)!;
    });

  it('the negative control has no barrier left in its loops', () => {
    expect(variant.split('\n').filter((l) => l.includes('__syncthreads();'))).toHaveLength(1);
  });
  it('builds for SEG_SIZE 8, 16 and 64, and for the race backend', () => {
    const cu = `#line 1 "${SNIPPET}"\n${snippet}\n#line 1 "s1110_no_barrier.cu"\n${variant}\n#line 1 "${DRIVER}"\n` + fs.readFileSync(DRIVER, 'utf8');
    fs.writeFileSync(path.join(dir, 'all.cu'), cu);
    const t = spawnSync('python3', [path.join(repo, 'labs/tools/cu2emu.py'), path.join(dir, 'all.cu'), path.join(dir, 'all.cpp')], { encoding: 'utf8' });
    expect(t.status, t.stdout + t.stderr).toBe(0);
    const common = ['-std=c++17', '-O1', '-g', '-fno-omit-frame-pointer', '-pthread', '-DS1110_WITH_VARIANT',
      '-I', inc, '-I', path.join(inc, 'pmpp/emu/shim'), '-include', 'pmpp/emu/cuda_emu.h', path.join(dir, 'all.cpp')];
    for (const seg of [8, 16, 64]) {
      const c = spawnSync(compiler!, ['-fsanitize=address', '-DPMPP_BACKEND_EMU', `-DSEG_SIZE=${seg}`, ...common, '-o', exe('emu', seg)], { encoding: 'utf8' });
      expect(c.status, c.stderr).toBe(0);
    }
    const r = spawnSync(compiler!, ['-fsanitize=thread', '-DPMPP_BACKEND_RACE', '-DSEG_SIZE=16', ...common, '-o', exe('race', 16)], { encoding: 'utf8' });
    raceBuilt = r.status === 0; // ThreadSanitizer is not available everywhere
  }, 240_000);
  it('exercise 6 (SEG_SIZE 8) and Fig. 11.6 (SEG_SIZE 16): the network\'s final row in all three orders', () => {
    for (const [seg, values] of [[8, EX6], [16, FIG116]] as const) {
      const r = runExe(exe('emu', seg), 'brent_kung', [...values]);
      expect(r.status, r.stderr).toBe(0);
      const ls = lines(r);
      expect(ls.map((l) => l[1])).toEqual(['forward', 'reverse', 'shuffle']);
      const want = run(brentKung(seg), values).at(-1)!;
      for (const l of ls) expect(l.slice(2).map(Number)).toEqual(want);
    }
  });
  it('several blocks and a short last block (SEG_SIZE 64, 150 values): each segment scanned on its own', () => {
    const values = ints(150, 5);
    const r = runExe(exe('emu', 64), 'brent_kung', values);
    expect(r.status, r.stderr).toBe(0);
    for (const l of lines(r)) expect(l.slice(2).map(Number)).toEqual(blockScan(values, 64));
    // The first full segment is exactly the model's network on those 64 values.
    expect(blockScan(values, 64).slice(0, 64)).toEqual(run(brentKung(64), values.slice(0, 64)).at(-1));
  });
  it('without the barriers in its loops the same kernel goes wrong', () => {
    const r = runExe(exe('emu', 16), 'no_barrier', [...FIG116]);
    const want = inclusiveScan(FIG116);
    const wrong = lines(r).filter((l) => JSON.stringify(l.slice(2).map(Number)) !== JSON.stringify(want));
    expect(wrong.length).toBeGreaterThan(0);
  });
  it('the race backend reports no data race in the kernel, and reports the negative control', (ctx) => {
    if (!raceBuilt) return ctx.skip();
    const ok = runExe(exe('race', 16), 'brent_kung', [...FIG116]);
    expect(ok.status, ok.stderr.slice(0, 2000)).toBe(0);
    expect(ok.stderr).not.toContain('ThreadSanitizer: data race');
    for (const l of lines(ok)) expect(l.slice(2).map(Number)).toEqual(inclusiveScan(FIG116));
    const bad = runExe(exe('race', 16), 'no_barrier', [...FIG116]);
    expect(bad.stderr).toContain('ThreadSanitizer: data race');
  }, 120_000);
});

describe('the narration of "Up-sweep and down-sweep" agrees with the model', () => {
  const script = (n: number) => fs.readFileSync(path.join(repo, `content/ch11/narration/bk-sweeps-${n}.txt`), 'utf8').trim();
  const words = (s: string) => s.replace(/\[[^\]]*\]/g, '').split(/\s+/).filter(Boolean).length;
  const bk = brentKung(16);
  const ks = koggeStone(16);
  it('six scripts, 35 to 60 words each, no symbols read aloud', () => {
    for (let n = 1; n <= 6; n++) {
      expect(words(script(n)), `bk-sweeps-${n}`).toBeGreaterThanOrEqual(35);
      expect(words(script(n)), `bk-sweeps-${n}`).toBeLessThanOrEqual(60);
      expect(script(n), `bk-sweeps-${n}`).not.toMatch(/[=×*+/<>%#_(){}−]/);
    }
  });
  it('part 1: sixteen positions; the first step adds into the 8 odd positions from their left neighbours', () => {
    expect(bk.n).toBe(16);
    expect(bk.steps[0].adders.every((a) => a.dst % 2 === 1 && a.src === a.dst - 1)).toBe(true);
    expect(script(1)).toContain('16 positions');
    expect(script(1)).toContain(`${bk.steps[0].adders.length} odd positions`);
  });
  it('part 2: 15 additions in the reduction tree; positions 0, 1, 3, 7 and 15 final', () => {
    expect(bk.steps.slice(0, 4).reduce((s, x) => s + x.adders.length, 0)).toBe(15);
    expect(finalAfter(bk, 4)).toEqual([0, 1, 3, 7, 15]);
    expect(script(2)).toContain('15 additions');
    expect(script(2)).toMatch(/positions 0, 1, 3, 7 and 15/i);
  });
  it('part 3: level one adds position 7 into position 11, 4 places apart', () => {
    expect(bk.steps[4].adders).toEqual([{ src: 7, dst: 11, thread: 0 }]);
    expect(script(3)).toContain('position 11');
    expect(script(3)).toContain('position 7, 4 places to its left');
  });
  it('part 4: level two completes 5, 9 and 13 from 3, 7 and 11', () => {
    expect(bk.steps[5].adders.map((a) => [a.src, a.dst])).toEqual([[3, 5], [7, 9], [11, 13]]);
    expect(script(4)).toMatch(/positions 5, 9 and 13/i);
    expect(script(4)).toContain('positions 3, 7 and 11');
  });
  it('part 5: 26 additions in 7 steps, against Kogge-Stone\'s 49 in 4', () => {
    expect([additions(bk), bk.steps.length, additions(ks), ks.steps.length]).toEqual([26, 7, 49, 4]);
    expect(script(5)).toContain('26 additions in 7 steps');
    expect(script(5)).toContain('49 additions in 4');
  });
  it('part 6: on a 32-lane warp, Brent-Kung takes 9 steps and Kogge-Stone 5', () => {
    const b = warpView('brent-kung', 32);
    const k = warpView('kogge-stone', 32);
    expect([b.finishesAt, k.finishesAt]).toEqual([9, 5]);
    expect(script(6)).toContain('32 lanes');
    expect(script(6)).toContain(`${b.finishesAt} steps`);
    expect(script(6)).toContain(`${k.finishesAt} steps`);
  });
});
