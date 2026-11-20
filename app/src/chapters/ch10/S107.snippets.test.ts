import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { hybrid, smallValues, warpReduceTrace } from '../../sim/shuffle';
import { functionSource } from '../../sim/toolchain';

// Section 10.7 shows two snippets from labs/ch10/snippets/, loaded at run time, so there
// is no lesson copy to drift. What can drift: the lines the lesson names by number, the
// corrected kernel's agreement with lab 10.1, and the claims the lesson makes about how
// the code behaves. The last group builds both snippets for the course's CPU emulator
// (labs/tools/cu2emu.py, pmpp/emu/cuda_emu.h, AddressSanitizer on), runs the drivers in
// labs/ch10/snippets/check/, and compares what they print with src/sim/shuffle.ts. It is
// skipped when no C++ compiler or python3 is found.
const repo = path.resolve(__dirname, '../../../..');
const read = (p: string) => fs.readFileSync(path.join(repo, p), 'utf8');
const PRINTED = 'labs/ch10/snippets/s107_warp_reduce_as_printed.cu';
const HYBRID = 'labs/ch10/snippets/s107_hybrid.cu';
const LAB = 'labs/ch10/lab1_sum/solution/kernel.cu';
const printed = read(PRINTED);
const hybridSrc = read(HYBRID);
const lesson = fs.readFileSync(path.join(__dirname, 'S107.tsx'), 'utf8');
/** Source without comments, for claims about the code itself. */
const code = (s: string) => s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
/** A function's text without the comment lines above it. */
const body = (s: string, name: string) => functionSource(s, name)!.text.split('\n').filter((l, i, a) => !(l.trim().startsWith('//') && a.slice(0, i).every((x) => x.trim().startsWith('//')))).join('\n');

describe('Fig. 10.13 as printed (the "as printed" tab)', () => {
  it('has the book\'s constant, mask literal and loop', () => {
    expect(printed).toContain('#define WARP_SIZE 32');
    expect(printed).toContain('__device__ __inline__ float warp_reduce(float val) {');
    expect(printed).toContain('for(unsigned int stride = WARP_SIZE/2; stride > 0; stride /= 2) {');
    expect(printed).toContain('partialSum += __shfl_down_sync(0xffffffff, partialSum, stride);');
    expect(printed).toContain('__device__ unsigned int warpIdx() { return threadIdx.x/WARP_SIZE; }');
    expect(printed).toContain('__device__ unsigned int laneIdx() { return threadIdx.x%WARP_SIZE; }');
  });
  it('mentions the literal 32 only where the book assumes it, with a comment saying so', () => {
    const lines = printed.split('\n').filter((l) => /\b32\b/.test(code(l)));
    expect(lines).toEqual(['#define WARP_SIZE 32  // the book\'s assumption: every warp has 32 lanes']);
  });
});

describe('Fig. 10.14 corrected (the hybrid listing)', () => {
  const k = functionSource(hybridSrc, 'WarpLevelSumReductionKernel')!;
  const line = k.text.split('\n');
  it('keeps the book\'s 19 lines, so the lesson\'s line numbers are the book\'s', () => {
    expect(line.length).toBe(19);
    expect(line[0]).toBe('__global__ void WarpLevelSumReductionKernel(float* input, float* output) {');
    expect(line[3].trim()).toBe('input_s[t] = input[t] + input[t + BLOCK_DIM];');
    expect(line[4].trim()).toBe('for (unsigned int stride = blockDim.x / 2; stride >= (unsigned int)warpSize; stride /= 2) {');
    expect(line[10].trim()).toBe('__syncthreads();');
    expect(line[11].trim()).toBe('if (warpIdx() == 0) {');
    expect(line[12].trim()).toBe('float partialSum = input_s[t];');
    expect(line[13].trim()).toBe('partialSum = warp_reduce(partialSum);');
    expect(line.slice(14, 17).map((l) => l.trim())).toEqual(['if (threadIdx.x == 0) {', '*output = partialSum;', '}']);
  });
  it('the lesson marks lines 5 and 11 to 14 and the caption names them', () => {
    expect(lesson).toContain('highlight={[5, 11, 12, 13, 14]}');
    expect(lesson.replace(/\s+/g, ' ')).toContain('the loop that stops at the warp width (line 5), the barrier after it (line 11) and the hand-off (lines 12 to 14)');
  });
  it('has no literal 32 and no 32-bit mask literal in its code', () => {
    expect(code(hybridSrc)).not.toMatch(/\b32\b|0xffffffff/i);
  });
  it.skipIf(!fs.existsSync(path.join(repo, LAB)))('shares its mask type, warpIdx and warp_reduce with lab 10.1, character for character', () => {
    const lab = read(LAB);
    const typedef = (s: string) => /#if defined\(__HIPCC__\)[\s\S]*?#endif/.exec(s)?.[0];
    expect(typedef(hybridSrc)).toBeDefined();
    expect(typedef(hybridSrc)).toBe(typedef(lab));
    expect(body(hybridSrc, 'warpIdx')).toBe(body(lab, 'warpIdx'));
    expect(body(hybridSrc, 'warp_reduce')).toBe(body(lab, 'warp_reduce'));
  });
  it('the model\'s "as printed" hand-off is this loop: it runs while stride >= width', () => {
    expect(hybrid(smallValues(2048), 1024, 32).strides).toEqual([512, 256, 128, 64, 32]);
  });
});

// ---------------------------------------------------------------- on the emulator
const has = (cmd: string, args: string[]) => spawnSync(cmd, args).status === 0;
const cxx = ['c++', 'clang++', 'g++'].find((c) => has(c, ['--version']));
const python = ['python3'].find((c) => has(c, ['--version']));
const INC = path.join(repo, 'labs/common/include');
const FLAGS = ['-std=c++17', '-O1', '-g', '-fsanitize=address', '-fno-omit-frame-pointer', '-pthread', '-DPMPP_BACKEND_EMU',
  '-I', INC, '-I', path.join(INC, 'pmpp/emu/shim'), '-include', 'pmpp/emu/cuda_emu.h'];

function build(sources: string[], defines: string[], name: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch10-s107-'));
  const objs = sources.map((src) => {
    const stem = path.basename(src, '.cu');
    const cpp = path.join(dir, stem + '.cpp');
    const obj = path.join(dir, stem + '.o');
    const t = spawnSync(python!, [path.join(repo, 'labs/tools/cu2emu.py'), path.join(repo, src), cpp], { encoding: 'utf8' });
    if (t.status !== 0) throw new Error('cu2emu failed on ' + src + ':\n' + t.stdout + t.stderr);
    const c = spawnSync(cxx!, [...FLAGS, ...defines, '-I', path.dirname(path.join(repo, src)), '-c', cpp, '-o', obj], { encoding: 'utf8' });
    if (c.status !== 0) throw new Error('compile failed for ' + src + ':\n' + c.stderr);
    return obj;
  });
  const exe = path.join(dir, name);
  const l = spawnSync(cxx!, ['-fsanitize=address', '-pthread', ...objs, '-o', exe], { encoding: 'utf8' });
  if (l.status !== 0) throw new Error('link failed:\n' + l.stderr);
  return exe;
}
const run = (exe: string) => {
  const r = spawnSync(exe, [], { encoding: 'utf8', env: { ...process.env, ASAN_OPTIONS: 'detect_leaks=0' }, timeout: 120000 });
  return { status: r.status, out: r.stdout, err: r.stderr };
};
const lane32 = Array.from({ length: 32 }, (_, i) => i);

describe.skipIf(!cxx || !python)('the snippets on the course emulator (labs/ch10/snippets/check/)', () => {
  it('as printed: every lane returns what the model says, under three thread orders; early returns poison lane 0', () => {
    const r = run(build(['labs/ch10/snippets/check/s107_warp_reduce_test.cu'], [], 'warp_reduce_test'));
    expect(r.status).toBe(0);
    const full = r.out.split('\n').filter((l) => l.startsWith('full '));
    expect(full.length).toBe(3);
    const model = warpReduceTrace(lane32, { pattern: 'down' }).final.map(String);
    for (const l of full) expect(l.split(':')[1].trim().split(' ')).toEqual(model);
    const val = (key: string) => Number(new RegExp('^' + key + ': (\\S+)$', 'm').exec(r.out)![1]);
    // lanes 20 to 31 returned: the model says undefined, the emulator returned 0xCD bytes and said so
    expect(warpReduceTrace(lane32, { pattern: 'down', tail: { from: 20, mode: 'return' } }).lane0).toBeNull();
    expect(val('return 20')).not.toBe(190);
    expect(r.err).toContain('which is not taking part in the call');
    // staying and adding 0 gives the sum of lanes 0 to 19, as in the model
    const zero = warpReduceTrace(lane32, { pattern: 'down', tail: { from: 20, mode: 'zero' } }).lane0;
    expect(zero).toBe(190);
    expect(val('zero 20')).toBe(zero);
    expect(val('zero 1')).toBe(0);
  }, 120000);

  for (const B of [64, 1024]) {
    it(`Fig. 10.14 corrected, ${B} threads: right under three orders, with or without line 11; the early-hand-off control races without it`, () => {
      const r = run(build(['labs/ch10/snippets/s107_hybrid.cu', 'labs/ch10/snippets/check/s107_hybrid_test.cu'], [`-DBLOCK_DIM=${B}`], `hybrid_${B}`));
      expect(r.status).toBe(0);
      const lines = r.out.split('\n').filter((l) => /^(printed|noline11|early2) /.test(l));
      const parse = (l: string) => {
        const [name, rest] = l.split(':');
        const [got, want] = rest.trim().split(' ').map(Number);
        return { kernel: name.split(' ')[0], got, want };
      };
      const res = lines.map(parse);
      expect(res.length).toBe(9);
      const model = hybrid(smallValues(2 * B), B, 32);
      expect(res[0].want).toBe(model.expected);
      expect(model.correct).toBe(true);
      for (const x of res.filter((x) => x.kernel === 'printed' || x.kernel === 'noline11')) expect(x.got).toBe(x.want);
      // without line 11 the model finds no race for the printed hand-off, and one for early2
      expect(hybrid(smallValues(2 * B), B, 32, 'printed', { postLoopBarrier: false }).races).toEqual([]);
      expect(hybrid(smallValues(2 * B), B, 32, 'early2', { postLoopBarrier: false }).races.length).toBeGreaterThan(0);
      expect(res.filter((x) => x.kernel === 'early2').some((x) => x.got !== x.want)).toBe(true);
    }, 120000);
  }
});
