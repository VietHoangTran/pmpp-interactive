import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  CORRECT,
  EX3,
  SCHEDULES,
  TOO_LARGE,
  UNVISITED,
  VARIANT,
  VARIANTS,
  coopLoop,
  finalMemory,
  gridFrontier,
  gridGraph,
  gridLaunches,
  gridLevels,
  gridSummary,
  hostLoop,
  hostStrip,
  kernelBody,
  residency,
  simulate,
  strideAssignment,
  treeGraph,
  treeSummary,
  variantOf,
  type Graph,
  type Schedule,
  type Shape,
  type Variant,
} from './gridbfs';
import { functionSource } from './toolchain';

const repo = path.resolve(__dirname, '../../..');
const SNIPPET = path.join(repo, 'labs/ch18/snippets/s186_coop_swap.cu');
const DRIVER = path.join(repo, 'labs/ch18/snippets/check/s186_coop_check.cu');
const source = fs.readFileSync(SNIPPET, 'utf8');
const kernelOf = (v: Variant) => functionSource(source, VARIANT[v].kernel)!;
const bodyOf = (v: Variant) => {
  const lines = kernelOf(v).text.split('\n');
  const open = lines.findIndex((l) => l.includes('{'));
  return lines.slice(open + 1, lines.length - 1).map((l) => l.replace(/^ {4}/, '')).join('\n');
};

// A BFS written here, independent of the model, to check the formulas.
function bfsSizes(n: number, out: (v: number) => number[]): number[] {
  const lv = new Array<number>(n).fill(-1);
  lv[0] = 0;
  const q = [0];
  for (let h = 0; h < q.length; h++)
    for (const u of out(q[h]))
      if (lv[u] < 0) {
        lv[u] = lv[q[h]] + 1;
        q.push(u);
      }
  const sizes: number[] = [];
  for (const l of lv) sizes[l] = (sizes[l] ?? 0) + 1;
  return sizes;
}
const gridOut = (W: number) => (v: number) => {
  const x = v % W, y = Math.floor(v / W), o: number[] = [];
  if (y > 0) o.push(v - W);
  if (x > 0) o.push(v - 1);
  if (x + 1 < W) o.push(v + 1);
  if (y + 1 < W) o.push(v + W);
  return o;
};

describe('frontier sizes from formulas', () => {
  it('the grid formula matches a BFS for W = 1 to 12', () => {
    for (let W = 1; W <= 12; W++) {
      const sizes = bfsSizes(W * W, gridOut(W));
      expect(sizes, `W ${W}`).toEqual(gridSummary(W).sizes);
      expect(sizes.length - 1).toBe(gridLevels(W));
    }
  });
  it('p-grid-levels: a 1000 x 1000 grid from a corner has 1,998 levels, needs 1,999 launches, and never holds more than 1,000', () => {
    const s = gridSummary(1000);
    expect(s.levels).toBe(1998);
    expect(gridLevels(1000)).toBe(1998);
    expect(s.launches).toBe(1999);
    expect(gridLaunches(1000)).toBe(1999);
    expect(s.maxFrontier).toBe(1000);
    expect(s.vertices).toBe(1_000_000);
    expect(gridFrontier(1000, 0)).toBe(1);
    expect(gridFrontier(1000, 999)).toBe(1000);
    expect(gridFrontier(1000, 1998)).toBe(1);
    expect(gridFrontier(1000, 1999)).toBe(0);
  });
  it('a b-ary tree has b^k vertices at level k, and few levels', () => {
    const g = treeGraph(4, 2);
    expect(bfsSizes(g.numVertices, (v) => g.dst.slice(g.srcPtrs[v], g.srcPtrs[v + 1]))).toEqual([1, 4, 16]);
    expect(treeSummary(8, 4)).toMatchObject({ sizes: [1, 8, 64, 512, 4096], levels: 4, launches: 5 });
  });
  it('the host loop: one launch and one read-back per non-empty frontier; one cooperative launch instead', () => {
    const h = hostLoop(gridSummary(5).sizes);
    expect(h.filter((e) => e.kind === 'launch')).toHaveLength(9);
    expect(h.filter((e) => e.kind === 'readback')).toHaveLength(9);
    expect(h[h.length - 1]).toEqual({ kind: 'readback', level: 9, next: 0 });
    expect(coopLoop(gridSummary(5).sizes)).toEqual({ launches: 1, readbacks: 0, gridBarriers: 27, levels: 9 });
    expect(coopLoop(gridSummary(5).sizes, 1).gridBarriers).toBe(9);
  });
  it('the scene\'s strips: with the same assumed round trip, it is about half of a thin grid\'s strip and about 1% of a fast-growing tree\'s', () => {
    const road = hostStrip(gridSummary(16).sizes, 10);
    expect(road.parts.filter((p) => p.kind === 'kernel')).toHaveLength(31);
    expect(road.total).toBe(256 + 310);
    expect(road.gapShare).toBeCloseTo(0.548, 3);
    const social = hostStrip(treeSummary(8, 4).sizes, 10);
    expect(social.total).toBe(4681 + 50);
    expect(social.gapShare).toBeCloseTo(0.0106, 4);
  });
  it('the stride assignment of Fig. 18.17: element i goes to thread i mod G', () => {
    expect(strideAssignment(10, 4)).toEqual([[0, 4, 8], [1, 5, 9], [2, 6], [3, 7]]);
    expect(strideAssignment(2, 4)).toEqual([[0], [1], [], []]);
  });
});

describe('residency (pp. 445-446)', () => {
  it('G <= R fits; G > R never completes under an ordinary launch and is refused under a cooperative one', () => {
    expect(residency(2, 3, 6)).toMatchObject({ slots: 6, fits: true, ordinary: 'completes', cooperative: 'launches', pending: 0 });
    expect(residency(2, 3, 7)).toMatchObject({ slots: 6, fits: false, resident: 6, pending: 1, ordinary: 'never completes', cooperative: 'refused' });
    expect(TOO_LARGE).toEqual({ cuda: 'cudaErrorCooperativeLaunchTooLarge', hip: 'hipErrorCooperativeLaunchTooLarge', metal: null });
  });
});

describe('the kernels are the snippet file\'s', () => {
  it('each variant\'s body is exactly the model\'s text', () => {
    for (const v of VARIANTS) expect(bodyOf(v), v).toBe(kernelBody(VARIANT[v].shape));
  });
  it('a removed barrier leaves a placeholder in the stepper\'s listing only', () => {
    const shown = kernelBody({ ...CORRECT, sync2: false }, true).split('\n');
    expect(shown).toContain('    // grid.sync() 2 removed');
    expect(shown).toHaveLength(kernelBody(CORRECT).split('\n').length);
  });
  it('variantOf finds the snippet kernel for each shape', () => {
    for (const v of VARIANTS) expect(variantOf(VARIANT[v].shape)).toBe(v);
    expect(variantOf({ ...CORRECT, sync1: false, sync2: false })).toBeNull();
  });
  it('the launch passes six arguments and the include is chosen by vendor', () => {
    expect(source).toMatch(/cudaLaunchCooperativeKernel\(\(void\*\)bfsCoop, numBlocks,\s*numThreadsPerBlock, kernelArgs, 0, 0\);/);
    expect(source).toContain('#if defined(__HIPCC__)\n#include <hip/hip_cooperative_groups.h>\n#else\n#include <cooperative_groups.h>\n#endif');
    expect(source).toContain('cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, bfsCoop,');
    expect(source).toContain('int numSMs = deviceProp.multiProcessorCount;');
  });
});

const GRID = gridGraph(5);
const TREE = treeGraph(4, 2);
const SMALL = gridGraph(4);
const run = (sh: Shape, s: Schedule, g: Graph = GRID, b = 2, t = 2) => simulate(sh, s, g, b, t);

describe('three barriers, three jobs', () => {
  it('every order is correct with all three barriers, on the grid and the tree, with 2 or 3 blocks', () => {
    for (const g of [GRID, TREE, SMALL])
      for (const b of [2, 3])
        for (const s of SCHEDULES) {
          const t = run(CORRECT, s, g, b);
          expect(t.outcome.kind, `${g.label} ${b} ${s}`).toBe('correct');
          expect(t.firstFailure).toBe(-1);
          expect(t.outcome.levelsCorrect).toBe(true);
        }
  });
  it('barrier 2 removed, in order (thread 0 first): thread 0 resets before the others read; they leave; a deadlock', () => {
    const t = run({ ...CORRECT, sync2: false }, 'forward');
    expect(t.steps[t.firstFailure].violation!.kind).toBe('read-after-reset');
    expect(t.outcome).toMatchObject({ kind: 'early-exit', label: 'early exit, deadlock', deadlock: true, levelsCorrect: false });
    expect(t.steps[t.steps.length - 1].op).toBe('deadlock');
    // Reversed, thread 0 reads last and resets after everyone has read.
    expect(run({ ...CORRECT, sync2: false }, 'reverse').outcome.kind).toBe('correct');
  });
  it('barrier 3 removed, reversed (thread 0 last): the others append before the reset, which throws their appends away', () => {
    const t = run({ ...CORRECT, sync3: false }, 'reverse');
    expect(t.steps[t.firstFailure].violation!.kind).toBe('late-reset');
    expect(t.outcome.kind).toBe('lost-appends');
    expect(t.outcome.label).toBe('appends lost to a late reset');
    expect(run({ ...CORRECT, sync3: false }, 'forward').outcome.kind).toBe('correct');
  });
  it('barrier 1 removed: a thread reads the size before others have appended', () => {
    const t = run({ ...CORRECT, sync1: false }, 'reverse');
    expect(t.steps[t.firstFailure].violation!.kind).toBe('partial-count');
    expect(t.outcome.kind).toBe('partial-count');
    for (const s of SCHEDULES) expect(run({ ...CORRECT, sync1: false }, s).firstFailure, s).toBeGreaterThanOrEqual(0);
  });
  it('each removed barrier has an order that fails and the failure is the stated kind', () => {
    const want: Record<string, string> = { sync1: 'partial-count', sync2: 'early-exit', sync3: 'lost-appends' };
    for (const k of ['sync1', 'sync2', 'sync3'] as const) {
      const kinds = SCHEDULES.map((s) => run({ ...CORRECT, [k]: false }, s).outcome.kind);
      expect(kinds, k).toContain(want[k]);
    }
  });
  it('Exercise 3: one barrier is enough in every order; without it, partial counts', () => {
    for (const g of [GRID, TREE, SMALL])
      for (const b of [2, 3])
        for (const s of SCHEDULES) expect(run(EX3, s, g, b).outcome.kind, `${g.label} ${s}`).toBe('correct');
    expect(run({ ...EX3, sync1: false }, 'forward').outcome.kind).toBe('partial-count');
  });
  it('the counter history of a correct level: appends, then every read, then one reset', () => {
    const t = run(CORRECT, 'forward');
    const lvl2 = t.history.filter((e) => e.level === 2);
    const kinds = lvl2.map((e) => e.kind);
    expect(kinds.lastIndexOf('append')).toBeLessThan(kinds.indexOf('read'));
    expect(kinds.filter((k) => k === 'read')).toHaveLength(4);
    expect(kinds[kinds.length - 1]).toBe('reset');
  });
  it('the stride: on the tree, level 2 has 16 elements and each of 4 threads takes 4', () => {
    const t = run(CORRECT, 'forward', TREE);
    const work2 = t.steps.filter((s) => s.op === 'work' && s.level === 3);
    expect(work2.map((s) => s.elements!.map((e) => e.i))).toEqual([[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]);
  });
  it('the gpu rule stops at the deadlock; the emu rule lets the survivors on and the levels come out wrong', () => {
    const emu = simulate({ ...CORRECT, sync2: false }, 'forward', GRID, 2, 2, 'emu');
    expect(emu.outcome.deadlock).toBe(false);
    expect(emu.outcome.kind).toBe('early-exit');
    expect(emu.outcome.label).toBe('early exit');
    expect(emu.outcome.levelsCorrect).toBe(false);
    expect(emu.steps.some((s) => s.op === 'release' && s.text.includes('the emulator does not wait'))).toBe(true);
  });
});

// The snippet file on the course's emulator, built as the lab runner builds the emu backend
// (labs/cmake/Backends.cmake): AddressSanitizer and the three thread orders. Skipped when no
// C++ compiler works.
const compiler = ['c++', 'clang++', 'g++'].find((c) => spawnSync(c, ['--version']).status === 0);
const clang = ['clang++'].find((c) => spawnSync(c, ['--version']).status === 0);
const inc = path.join(repo, 'labs/common/include');

describe.skipIf(!clang)('the CUDA front-end check (clang, stub headers, as labs/pmpp.py runs it)', () => {
  it('parses the snippet and the driver', () => {
    for (const f of [SNIPPET, DRIVER]) {
      const r = spawnSync(clang!, ['-x', 'cuda', '--cuda-host-only', '-nocudainc', '-nocudalib', '-fsyntax-only', '-std=c++17',
        '-I', inc, '-I', path.join(inc, 'pmpp/emu/shim'), '-include', 'pmpp/check/cuda_stub.h', f], { encoding: 'utf8' });
      if (r.status !== 0 && /cannot parse|unknown argument|error: language not recognized/.test(r.stderr)) return; // not a CUDA-capable clang
      // The driver is checked on its own only after the snippet: it uses the snippet's declarations.
      if (f === DRIVER) continue;
      expect(r.status, f + '\n' + r.stderr).toBe(0);
    }
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch18-coop-fe-'));
    const both = path.join(dir, 'both.cu');
    fs.writeFileSync(both, source + '\n' + fs.readFileSync(DRIVER, 'utf8'));
    const r = spawnSync(clang!, ['-x', 'cuda', '--cuda-host-only', '-nocudainc', '-nocudalib', '-fsyntax-only', '-std=c++17',
      '-I', inc, '-I', path.join(inc, 'pmpp/emu/shim'), '-include', 'pmpp/check/cuda_stub.h', both], { encoding: 'utf8' });
    expect(r.status, r.stderr).toBe(0);
  }, 60_000);
});

describe.skipIf(!compiler)('the snippet on the emulator agrees with the model', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch18-coop-'));
  const exe = path.join(dir, 'emu');
  const env = { ...process.env, ASAN_OPTIONS: 'detect_leaks=0' };
  it('builds', () => {
    const cu = '#line 1 "s186_coop_swap.cu"\n' + source + '\n#line 1 "check/s186_coop_check.cu"\n' + fs.readFileSync(DRIVER, 'utf8');
    fs.writeFileSync(path.join(dir, 'all.cu'), cu);
    const t = spawnSync('python3', [path.join(repo, 'labs/tools/cu2emu.py'), path.join(dir, 'all.cu'), path.join(dir, 'all.cpp')], { encoding: 'utf8' });
    expect(t.status, t.stdout + t.stderr).toBe(0);
    const c = spawnSync(compiler!, ['-fsanitize=address', '-DPMPP_BACKEND_EMU', '-std=c++17', '-O1', '-g', '-fno-omit-frame-pointer', '-pthread',
      '-I', inc, '-I', path.join(inc, 'pmpp/emu/shim'), '-include', 'pmpp/emu/cuda_emu.h', path.join(dir, 'all.cpp'), '-o', exe], { encoding: 'utf8' });
    expect(c.status, c.stderr).toBe(0);
  }, 240_000);

  const graphs: [string, string[], Graph][] = [
    ['grid 5', ['grid', '5'], GRID],
    ['grid 4', ['grid', '4'], SMALL],
    ['tree 4 2', ['tree', '4', '2'], TREE],
  ];
  for (const v of VARIANTS)
    for (const [name, args, g] of graphs)
      it(`${VARIANT[v].kernel}, ${name}: the emulator's three orders leave the memory the model's 'emu' rule leaves`, () => {
        for (const b of [2, 3])
          for (const s of SCHEDULES) {
            const r = spawnSync(exe, [VARIANT[v].kernel, s, ...args, String(b), '2'], { encoding: 'utf8', env });
            expect(r.status, `${s}\n${r.stderr.slice(0, 1500)}`).toBe(0);
            const parts = r.stdout.trim().split(' | ');
            const head = parts[0].split(' ');
            expect(head.slice(0, 3)).toEqual([VARIANT[v].kernel, s, '0']);
            const nums = (p: string) => p.split(' ').slice(1).map(Number);
            const m = finalMemory(simulate(VARIANT[v].shape, s, g, b, 2, 'emu'));
            expect(nums(parts[1]), `${s} ${b} levels`).toEqual(m.levels);
            expect(nums(parts[2]), `${s} ${b} A`).toEqual(m.A);
            expect(nums(parts[3]), `${s} ${b} B`).toEqual(m.B);
            expect(nums(parts[4]), `${s} ${b} counters`).toEqual(m.counters);
          }
      }, 120_000);

  it('the emulator refuses a cooperative grid one block larger than residentBlocks() computes', () => {
    const r = spawnSync(exe, ['toolarge', '64'], { encoding: 'utf8', env });
    expect(r.status).toBe(0);
    const [, resident, rc] = r.stdout.trim().split(' ');
    expect(Number(resident)).toBeGreaterThan(0);
    expect(Number(rc)).toBe(720); // cudaErrorCooperativeLaunchTooLarge (CUDA runtime types, ch18-facts A3)
    expect(r.stderr).toContain('cooperative launch too large');
  });
  it('a correct run leaves every vertex labelled with its BFS level', () => {
    const r = spawnSync(exe, ['bfsCoop', 'shuffle', 'grid', '5', '3', '2'], { encoding: 'utf8', env });
    const levels = r.stdout.trim().split(' | ')[1].split(' ').slice(1).map(Number);
    expect(levels.every((l) => l !== UNVISITED)).toBe(true);
    expect(levels).toEqual(Array.from({ length: 25 }, (_, v) => (v % 5) + Math.floor(v / 5)));
  });
});
