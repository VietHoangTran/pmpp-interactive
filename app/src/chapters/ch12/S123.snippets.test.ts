import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { functionSource } from '../../sim/toolchain';
import { LINE_PAIRS, passFrom, vote } from '../../sim/warpvote';

// Section 12.3 shows Figs. 12.3 and 12.4 from labs/ch12/snippets/, loaded at run time, so
// there is no lesson copy to drift. What can drift: the line numbers the lesson and the
// "Which line is which?" pairs name, the differences from the page the lesson lists, and
// what the lesson says the code does. The last group builds both snippets for the course's
// CPU emulator (labs/tools/cu2emu.py, pmpp/emu/cuda_emu.h, AddressSanitizer on), runs
// labs/ch12/snippets/check/s123_check.cu, and compares what it prints with
// src/sim/warpvote.ts. It is skipped when no C++ compiler or python3 is found.
const repo = path.resolve(__dirname, '../../../..');
const read = (p: string) => fs.readFileSync(path.join(repo, p), 'utf8');
const FIG3 = 'labs/ch12/snippets/s123_fig12_3.cu';
const FIG4 = 'labs/ch12/snippets/s123_fig12_4_cg.cu';
const LAB = 'labs/ch12/lab1_unstable/solution/';
const lesson = fs.readFileSync(path.join(__dirname, 'S123.tsx'), 'utf8').replace(/\s+/g, ' ');
/** The kernel as the lesson shows it: the definition without the comment lines above it. */
const kernel = (file: string) => {
  const lines = functionSource(read(file), 'filter_kernel')!.text.split('\n');
  let k = 0;
  while (lines[k].trim().startsWith('//')) k++;
  return lines.slice(k);
};
const code = (s: string) => s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');

describe('Fig. 12.3 (labs/ch12/snippets/s123_fig12_3.cu)', () => {
  const line = kernel(FIG3);
  const at = (n: number) => line[n - 1].trim();
  it('keeps the book\'s 29 lines, so the lesson\'s line numbers are the book\'s', () => {
    expect(line.length).toBe(29);
    expect(at(1)).toBe('__global__ void filter_kernel(unsigned int* input, unsigned int* output,');
    expect(at(7)).toBe('unsigned int activeThreads = __activemask();');
    expect(at(10)).toBe('unsigned int leader = __ffs(activeThreads) - 1;');
    expect(at(11)).toBe('if(laneIdx() == leader) {');
    expect(at(13)).toBe('unsigned int numActive = __popc(activeThreads);');
    expect(at(17)).toBe('j = outputSize_ref.fetch_add(numActive, cuda::memory_order_relaxed);');
    expect(at(20)).toBe('j = __shfl_sync(activeThreads, j, leader);');
    expect(at(22)).toBe('unsigned int previousThreads = (1 << laneIdx()) - 1;');
    expect(at(23)).toBe('unsigned int previousActiveThreads = activeThreads & previousThreads;');
    expect(at(24)).toBe('unsigned int offset = __popc(previousActiveThreads);');
    expect(at(26)).toBe('output[j + offset] = val;');
  });
  it('differs from the page in line 08 only, which the lesson and the file header say', () => {
    expect(at(8)).toBe('unsigned int j = 0;');
    expect(read(FIG3)).toContain('line 08 gives j a value');
    expect(lesson).toContain('The snippet writes <code>unsigned int j = 0;</code>');
  });
  it('the lesson marks the four steps and the rank', () => {
    expect(lesson).toContain('highlight={[7, 10, 13, 17, 20, 22, 23, 24]}');
  });
  it('mentions the literal 32 only where the book assumes it, with a comment saying so', () => {
    const lines = read(FIG3).split('\n').filter((l) => /\b32\b/.test(code(l)));
    expect(lines).toEqual(['#define WARP_SIZE 32  // the book\'s assumption: every warp has 32 lanes']);
  });
  it('uses chapter 10\'s laneIdx() as printed there', () => {
    expect(read(FIG3)).toContain('__device__ unsigned int laneIdx() { return threadIdx.x%WARP_SIZE; }');
  });
});

describe('Fig. 12.4 (labs/ch12/snippets/s123_fig12_4_cg.cu)', () => {
  const line = kernel(FIG4);
  const at = (n: number) => line[n - 1].trim();
  it('keeps the book\'s 26 lines', () => {
    expect(line.length).toBe(26);
    expect(at(7)).toBe('cg::coalesced_group activeThreads = cg::coalesced_threads();');
    expect(at(8)).toBe('unsigned int j = 0;');
    expect(at(10)).toBe('if(activeThreads.thread_rank() == 0) {');
    expect(at(12)).toBe('unsigned int numActive = activeThreads.num_threads();');
    expect(at(16)).toBe('j = outputSize_ref.fetch_add(numActive, cuda::memory_order_relaxed);');
    expect(at(19)).toBe('j = activeThreads.shfl(j, 0);');
    expect(at(21)).toBe('unsigned int offset = activeThreads.thread_rank();');
    expect(at(23)).toBe('output[j + offset] = val;');
  });
  it('brings the header and namespace the page omits, with HIP\'s header path', () => {
    const src = read(FIG4);
    expect(src).toContain('#include <cooperative_groups.h>');
    expect(src).toContain('#include <hip/hip_cooperative_groups.h>');
    expect(src).toContain('namespace cg = cooperative_groups;');
  });
  it('the lesson marks the lines the pairs name', () => {
    expect(lesson).toContain('highlight={[7, 10, 12, 19, 21]}');
  });
});

describe('"Which line is which?" pairs', () => {
  const l3 = kernel(FIG3);
  const l4 = kernel(FIG4);
  for (const p of LINE_PAIRS)
    it(`${p.id}: Fig. 12.3 line ${p.fig3.join(', ')} and Fig. 12.4 line ${p.fig4} are the lines the pair names`, () => {
      expect(l3[p.fig3[0] - 1]).toContain(p.fig3Text);
      expect(l4[p.fig4 - 1]).toContain(p.fig4Text);
    });
  it('the book\'s own mapping (p. 294): 07 to 07, 10-11 to 10, 13 to 12, 22-24 to 21; the shuffle on 19', () => {
    expect(LINE_PAIRS.map((p) => [p.fig3, p.fig4])).toEqual([[[7], 7], [[10, 11], 10], [[13], 12], [[20], 19], [[22, 23, 24], 21]]);
  });
});

describe.skipIf(!fs.existsSync(path.join(repo, LAB, 'kernel.cu')))('lab 12.1, as the lesson describes it', () => {
  it('the ballot form: a typed mask, a vote by every lane, a full-mask shuffle, j starting at 0', () => {
    const cu = read(LAB + 'kernel.cu');
    expect(cu).toMatch(/\/\/ region: helpers[\s\S]*typedef unsigned long long lane_mask_t;[\s\S]*\/\/ endregion/);
    expect(cu).toContain('// region: warp');
    expect(cu).toContain('lane_mask_t vote = __ballot_sync(full, keep);');
    expect(cu).toContain('j = __shfl_sync(full, j, leader);');
    expect(cu).toContain('unsigned int j = 0;');
    expect(cu).toContain('return __popcll(m);');
  });
  it('the Metal form checks the count, which is why the visualiser\'s guard is on by default', () => {
    const metal = read(LAB + 'kernel.metal');
    expect(metal).toContain('// region: warp');
    expect(metal).toContain('simd_is_first() && count > 0');
    expect(metal).toContain('simd_prefix_exclusive_sum(flag)');
    expect(metal).toContain('j = simd_broadcast_first(j);');
    expect(lesson).toContain('const [guard, setGuard] = useState(true);');
  });
});

// ---------------------------------------------------------------- on the emulator
const has = (cmd: string, args: string[]) => spawnSync(cmd, args).status === 0;
const cxx = ['c++', 'clang++', 'g++'].find((c) => has(c, ['--version']));
const python = ['python3'].find((c) => has(c, ['--version']));
const INC = path.join(repo, 'labs/common/include');
const FLAGS = ['-std=c++17', '-O1', '-g', '-fsanitize=address', '-fno-omit-frame-pointer', '-pthread', '-DPMPP_BACKEND_EMU',
  '-I', INC, '-I', path.join(INC, 'pmpp/emu/shim'), '-include', 'pmpp/emu/cuda_emu.h'];
const DRIVER = 'labs/ch12/snippets/check/s123_check.cu';

function buildAndRun(defines: string[], name: string) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch12-s123-'));
  const cpp = path.join(dir, name + '.cpp');
  const t = spawnSync(python!, [path.join(repo, 'labs/tools/cu2emu.py'), path.join(repo, DRIVER), cpp], { encoding: 'utf8' });
  if (t.status !== 0) throw new Error('cu2emu failed:\n' + t.stdout + t.stderr);
  const obj = path.join(dir, name + '.o');
  const c = spawnSync(cxx!, [...FLAGS, ...defines, '-I', path.dirname(path.join(repo, DRIVER)), '-c', cpp, '-o', obj], { encoding: 'utf8' });
  if (c.status !== 0) throw new Error('compile failed:\n' + c.stderr);
  const exe = path.join(dir, name);
  const l = spawnSync(cxx!, ['-fsanitize=address', '-pthread', obj, '-o', exe], { encoding: 'utf8' });
  if (l.status !== 0) throw new Error('link failed:\n' + l.stderr);
  const r = spawnSync(exe, [], { encoding: 'utf8', env: { ...process.env, ASAN_OPTIONS: 'detect_leaks=0' }, timeout: 120000 });
  return { status: r.status, out: r.stdout, err: r.stderr };
}

const CASES: Record<string, number[]> = {
  book: [1, 2, 4, 5],
  rank: [2, 4, 5, 7],
  all: Array.from({ length: 32 }, (_, l) => l),
  lane31: [31],
};

describe.skipIf(!cxx || !python)('both listings on the course emulator (labs/ch12/snippets/check/)', () => {
  for (const [label, defines] of [['Fig. 12.3', []], ['Fig. 12.4', ['-DSNIPPET_CG']]] as const) {
    it(`${label}: every lane writes the slot the model gives, in all three thread orders; many blocks are right and side by side`, () => {
      const r = buildAndRun([...defines], label === 'Fig. 12.3' ? 'fig3' : 'fig4');
      expect(r.status, r.err).toBe(0);
      const warps = r.out.split('\n').filter((l) => l.startsWith('warp '));
      expect(warps.length).toBe(3 * Object.keys(CASES).length);
      for (const l of warps) {
        const [head, rest] = l.split(':');
        const name = head.split(' ')[1];
        const lanes = CASES[name];
        const model = vote({ width: 32, pass: passFrom(32, lanes), counter: 40, form: 'book' });
        const want = model.lanes.filter((L) => L.writes).map((L) => `${L.lane}>${L.slot}`).join(' ') + ` counter ${model.counterAfter}`;
        expect(rest.trim(), l).toBe(want);
      }
      const many = r.out.split('\n').filter((l) => l.startsWith('many '));
      expect(many.length).toBe(3);
      for (const l of many) expect(l).toMatch(/: right, side by side$/);
      expect(r.out).toContain('DONE');
    }, 180000);
  }
});
