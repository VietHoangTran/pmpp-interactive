import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { BOOK, EX1, levels, runBFS, type Graph } from '../../sim/bfs';
import { FORMS, FORM_IDS, SCENARIOS, changedLines, region, runOrder, type FormId } from '../../sim/bfsvisit';

// Section 18.3 shows the book's five kernels, as printed and as corrected, and the eight
// visit forms of the stepper, all from labs/ch18/snippets/s183_variants.cu, loaded at run
// time. What can drift: the book's text and line numbers in the "as printed" regions, the
// lines the lesson says the corrections change, the stepper's listing (the model's text must
// be the file's), and what the models say the code does. The last group builds
// labs/ch18/snippets/check/s183_check.cu for the course's CPU emulator, as the lab runner
// builds the emu backend (AddressSanitizer on), runs it in the forward and reversed thread
// orders, and compares it with src/sim/bfs.ts and src/sim/bfsvisit.ts. Skipped when no C++
// compiler or python3 is found.
const repo = path.resolve(__dirname, '../../../..');
const FILE = 'labs/ch18/snippets/s183_variants.cu';
const DRIVER = 'labs/ch18/snippets/check/s183_check.cu';
const src = fs.readFileSync(path.join(repo, FILE), 'utf8');
const lesson = fs.readFileSync(path.join(__dirname, 'S183.tsx'), 'utf8').replace(/\s+/g, ' ');
const part = (name: string) => {
  const r = region(src, name);
  if (r == null) throw new Error('no region ' + name);
  return r;
};
const lines = (name: string) => part(name).split('\n');
const at = (name: string, n: number) => lines(name)[n - 1].trim();

describe('as printed: the book\'s text and line numbers (pp. 433-441)', () => {
  it('Fig. 18.6, push: 16 lines; the level check on 05, the CSR walk on 06-08, check-then-label on 09-11', () => {
    expect(lines('fig18_6_printed').length).toBe(16);
    expect(at('fig18_6_printed', 1)).toBe('__global__ void bfs_kernel(CSRGraph csrGraph, unsigned int* level,');
    expect(at('fig18_6_printed', 3)).toBe('unsigned int vertex = blockIdx.x*blockDim.x + threadIdx.x;');
    expect(at('fig18_6_printed', 5)).toBe('if(level[vertex] == currLevel - 1) {');
    expect(at('fig18_6_printed', 8)).toBe('unsigned int neighbor = csrGraph.dst[edge];');
    expect(at('fig18_6_printed', 9)).toBe('if(level[neighbor] == UNVISITED) {');
    expect(at('fig18_6_printed', 10)).toBe('level[neighbor] = currLevel;');
    expect(at('fig18_6_printed', 11)).toBe('*newVertexVisited = 1;');
  });
  it('Fig. 18.8, pull: 17 lines; the unvisited check on 05, the CSC walk, the label of its own vertex and the break on 10-12', () => {
    expect(lines('fig18_8_printed').length).toBe(17);
    expect(at('fig18_8_printed', 5)).toBe('if(level[vertex] == UNVISITED) {');
    expect(at('fig18_8_printed', 8)).toBe('unsigned int neighbor = cscGraph.src[edge];');
    expect(at('fig18_8_printed', 9)).toBe('if(level[neighbor] == currLevel - 1) {');
    expect(at('fig18_8_printed', 10)).toBe('level[vertex] = currLevel;');
    expect(at('fig18_8_printed', 12)).toBe('break;');
  });
  it('Fig. 18.10, edge-centric: 14 lines; the source check on 06, the label on 08-10', () => {
    expect(lines('fig18_10_printed').length).toBe(14);
    expect(at('fig18_10_printed', 3)).toBe('unsigned int edge = blockIdx.x*blockDim.x + threadIdx.x;');
    expect(at('fig18_10_printed', 5)).toBe('unsigned int vertex = cooGraph.src[edge];');
    expect(at('fig18_10_printed', 6)).toBe('if(level[vertex] == currLevel - 1) {');
    expect(at('fig18_10_printed', 8)).toBe('if(level[neighbor] == UNVISITED) {');
  });
  it('Fig. 18.12, frontiers: 20 lines; the bounds check on 06, the visit on 11, the append on 12-16', () => {
    expect(lines('fig18_12_printed').length).toBe(20);
    expect(at('fig18_12_printed', 6)).toBe('if(i < numPrevFrontier) {');
    expect(at('fig18_12_printed', 7)).toBe('unsigned int vertex = prevFrontier[i];');
    expect(at('fig18_12_printed', 11)).toBe('if(visitVertexAtomically(neighbor, level, currLevel)) {');
    expect(at('fig18_12_printed', 12)).toBe('cuda::atomic_ref<unsigned int, cuda::thread_scope_device>');
    expect(at('fig18_12_printed', 13)).toBe('numCurrFrontier_ref(*numCurrFrontier);');
    expect(at('fig18_12_printed', 15)).toBe('numCurrFrontier_ref.fetch_add(1, cuda::memory_order_relaxed);');
    expect(at('fig18_12_printed', 16)).toBe('currFrontier[currFrontierIdx] = neighbor;');
  });
  it('Fig. 18.14, the visit: 9 lines; the local expected value on 05, the strong compare-and-swap on 06-07', () => {
    expect(lines('fig18_14_printed').length).toBe(9);
    expect(at('fig18_14_printed', 5)).toBe('unsigned int unvisited = UNVISITED;');
    expect(at('fig18_14_printed', 6)).toBe('unsigned int visitSuccess = level_ref.compare_exchange_strong(unvisited,');
    expect(at('fig18_14_printed', 7)).toBe('currLevel, cuda::memory_order_relaxed, cuda::memory_order_relaxed);');
  });
  it('UNVISITED is UINT_MAX from <climits>, and the file says why it is not the page\'s numeric_limits', () => {
    expect(src).toContain('#include <climits>');
    expect(src).toContain('#define UNVISITED UINT_MAX');
    expect(src).toContain('cuda::std::numeric_limits<unsigned>, which names the');
  });
});

describe('corrected: only the lines the lesson names change', () => {
  const diff = (fig: string) => changedLines(part(fig + '_printed'), part(fig + '_fixed'));
  it('Fig. 18.6: lines 05, 09, 10, 11 become relaxed atomic loads and stores', () => {
    expect(diff('fig18_6')).toEqual([5, 9, 10, 11]);
    expect(at('fig18_6_fixed', 10)).toBe('storeRelaxed(level[neighbor], currLevel);');
  });
  it('Fig. 18.8: lines 05, 09, 10, 11', () => {
    expect(diff('fig18_8')).toEqual([5, 9, 10, 11]);
  });
  it('Fig. 18.10: lines 06, 08, 09, 10', () => {
    expect(diff('fig18_10')).toEqual([6, 8, 9, 10]);
  });
  it('Fig. 18.12: no change', () => {
    expect(diff('fig18_12')).toEqual([]);
  });
  it('Fig. 18.14: a HIP branch with the legacy atomicCAS; the CUDA lines are the book\'s', () => {
    const d = diff('fig18_14');
    const fixed = lines('fig18_14_fixed');
    expect(d.map((n) => fixed[n - 1].trim())).toEqual([
      '#if defined(__HIPCC__)',
      '// HIP: libhipcxx makes compare_exchange_strong weak at device scope; atomicCAS is strong.',
      'return atomicCAS(&level[vertex], UNVISITED, currLevel) == UNVISITED;',
      '#else',
      '#endif',
    ]);
  });
  it('the lesson computes its highlights with the same diff and describes these changes', () => {
    expect(lesson).toContain('changedLines(');
    expect(lesson).toContain("useLabFile('ch18/snippets/s183_variants.cu')");
    expect(lesson).toContain('relaxed atomic');
  });
});

describe('the stepper\'s listings are the file\'s', () => {
  for (const id of FORM_IDS)
    it(`${id}: the model's ${FORMS[id].code.length} lines are region ${FORMS[id].region}`, () => {
      expect(FORMS[id].code.join('\n')).toBe(part(FORMS[id].region));
    });
  it('the helpers the forms call: one append, a strong and a weak compare-and-swap, both relaxed', () => {
    const h = part('visit_helpers');
    expect(h).toContain('count.fetch_add(1, cuda::memory_order_relaxed)');
    expect(h).toContain('ref.compare_exchange_strong(e, desired, cuda::memory_order_relaxed, cuda::memory_order_relaxed)');
    expect(h).toContain('ref.compare_exchange_weak(e, desired, cuda::memory_order_relaxed, cuda::memory_order_relaxed)');
  });
});

// ---------------------------------------------------------------- on the emulator
const has = (cmd: string, args: string[]) => spawnSync(cmd, args).status === 0;
const cxx = ['c++', 'clang++', 'g++'].find((c) => has(c, ['--version']));
const python = ['python3'].find((c) => has(c, ['--version']));
const INC = path.join(repo, 'labs/common/include');
const FLAGS = ['-std=c++17', '-O1', '-g', '-fsanitize=address', '-fno-omit-frame-pointer', '-pthread', '-DPMPP_BACKEND_EMU',
  '-I', INC, '-I', path.join(INC, 'pmpp/emu/shim'), '-include', 'pmpp/emu/cuda_emu.h'];
const GRAPHS: Record<string, Graph> = { book: BOOK, ex1: EX1 };
const KERNEL_OF: Record<string, 'push' | 'pull' | 'edge' | 'frontier'> = {
  push_printed: 'push', push_fixed: 'push', pull_printed: 'pull', pull_fixed: 'pull',
  edge_printed: 'edge', edge_fixed: 'edge', frontier_printed: 'frontier', frontier_fixed: 'frontier',
};

describe.skipIf(!cxx || !python)('the kernels on the course emulator (labs/ch18/snippets/check/)', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch18-s183-'));
  const exe = path.join(dir, 'check');
  const env = { ...process.env, ASAN_OPTIONS: 'detect_leaks=0' };
  it('builds', () => {
    const cpp = path.join(dir, 'check.cpp');
    const t = spawnSync(python!, [path.join(repo, 'labs/tools/cu2emu.py'), path.join(repo, DRIVER), cpp], { encoding: 'utf8' });
    expect(t.status, t.stdout + t.stderr).toBe(0);
    const c = spawnSync(cxx!, [...FLAGS, '-I', path.dirname(path.join(repo, DRIVER)), cpp, '-o', exe], { encoding: 'utf8' });
    expect(c.status, c.stderr).toBe(0);
  }, 240_000);
  for (const order of ['forward', 'reverse'] as const)
    it(`${order}: every kernel labels the model's levels in the model's number of calls, and every visit form ends where the model's ${order} order ends`, () => {
      const r = spawnSync(exe, [order], { encoding: 'utf8', env, timeout: 60_000 });
      expect(r.status, r.stderr.slice(0, 2000)).toBe(0);
      const out = r.stdout.trim().split('\n');
      expect(out.at(-1)).toBe('DONE');
      const bfs = out.filter((l) => l.startsWith('bfs '));
      expect(bfs.length).toBe(3 * 8);
      for (const l of bfs) {
        const m = /^bfs (\w+) (\d+) (\w+) calls (\d+) sizes([\d ]*) levels ([\d ]+)$/.exec(l);
        expect(m, l).not.toBeNull();
        const [, g, root, kernel, calls, sizes, lv] = m!;
        const model = runBFS(GRAPHS[g], { variant: KERNEL_OF[kernel], root: Number(root), blockSize: 4, warpWidth: 4 });
        expect(lv.split(' ').map(Number), l).toEqual(levels(GRAPHS[g], Number(root)));
        expect(Number(calls), l).toBe(model.calls.length);
        if (KERNEL_OF[kernel] === 'frontier') expect(sizes.trim().split(' ').map(Number), l).toEqual(model.calls.map((c) => c.items));
      }
      const visits = out.filter((l) => l.startsWith('visit '));
      expect(visits.length).toBe(2 * 7);
      for (const l of visits) {
        const m = /^visit (\w+) ([\w-]+) level ([\d ]+) frontier([\d ]*) counter (\d+)$/.exec(l);
        expect(m, l).not.toBeNull();
        const [, sc, form, lv, fr, counter] = m!;
        const st = runOrder(FORMS[form as FormId], SCENARIOS[sc as 'book' | 'trap'], order).states.at(-1)!;
        expect(lv.split(' ').map(Number), l).toEqual(st.level);
        expect(fr.trim() ? fr.trim().split(' ').map(Number) : [], l).toEqual(st.frontier);
        expect(Number(counter), l).toBe(st.counter);
      }
    }, 120_000);
  it('while (!cas(e)) e = UNVISITED does not return on the emulator either', () => {
    expect(runOrder(FORMS['weak-reset'], SCENARIOS.trap, 'forward').outcome.kind).toBe('never-terminates');
    const r = spawnSync(exe, ['spin'], { encoding: 'utf8', env, timeout: 4000 });
    expect(r.status).toBeNull(); // stopped by the time limit
    expect(r.stdout).not.toContain('visit trap weak-reset');
  }, 30_000);
});
