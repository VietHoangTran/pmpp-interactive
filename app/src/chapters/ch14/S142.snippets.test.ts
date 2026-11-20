import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { FIG_14_1, LARGEST_FIRST, TRAP, runToSorted, walk } from '../../sim/oddeven';
import { functionSource } from '../../sim/toolchain';

// Section 14.2 shows labs/ch14/snippets/s142_oddeven.cu, loaded at run time, so there is no
// lesson copy to drift. What can drift: the book's line numbers the lesson names, the two
// differences from the page it lists, and what it says the code does. The last two groups
// build the snippet for the course's CPU emulator (labs/tools/cu2emu.py,
// pmpp/emu/cuda_emu.h, AddressSanitizer on), run labs/ch14/snippets/check/s142_check.cu
// under the three thread orders and compare what it prints with src/sim/oddeven.ts; then
// build the race-detection flavour (ThreadSanitizer) and check that the plain store of
// Fig. 14.2 as printed is reported and the snippet's atomic store is not. Each group is
// skipped when no C++ compiler, python3 or working ThreadSanitizer is found.
const repo = path.resolve(__dirname, '../../../..');
const SNIPPET = 'labs/ch14/snippets/s142_oddeven.cu';
const DRIVER = 'labs/ch14/snippets/check/s142_check.cu';
const read = (p: string) => fs.readFileSync(path.join(repo, p), 'utf8');
const lesson = fs.readFileSync(path.join(__dirname, 'S142.tsx'), 'utf8').replace(/\s+/g, ' ');

describe('Fig. 14.2 in the snippet keeps the page\'s 13 lines', () => {
  const kernel = functionSource(read(SNIPPET), 'sort_kernel')!.text.split('\n');
  const at = (n: number) => kernel[n - 1].trim();
  it('line by line', () => {
    expect(kernel.length).toBe(13);
    expect(at(1)).toBe('__global__ void sort_kernel(unsigned int* data, unsigned int* hasChanged,');
    expect(at(3)).toBe('unsigned int i = 2*(blockIdx.x*blockDim.x + threadIdx.x)');
    expect(at(4)).toBe('+ (isOddStep ? 1 : 0);');
    expect(at(6)).toBe('if(data[i] > data[i + 1]) {');
    expect(at(7)).toBe('unsigned int tmp = data[i];');
  });
  it('differs from the page on lines 05 and 10 only, and the lesson says so', () => {
    expect(at(5)).toBe('if(i + 1 < N) {');
    expect(at(10)).toBe('atomicExch(hasChanged, 1u);');
    expect(lesson).toContain('highlight: [5, 10]');
    expect(lesson).toContain('<code>if(i &lt; N - 1)</code>');
    expect(lesson).toContain('<code>*hasChanged = 1;</code>');
    expect(read(SNIPPET)).toContain('line 05  i + 1 < N instead of i < N - 1');
  });
  it('the lesson loads the file itself, and the host loop from the same file', () => {
    expect(lesson).toContain("const SNIPPET = 'ch14/snippets/s142_oddeven.cu';");
    expect(lesson).toContain('useLabFile(SNIPPET)');
    expect(lesson).toContain("bare(snip, 'sort_kernel')");
    expect(lesson).toContain("functionSource(snip!, 'oddEvenSort')");
    expect(functionSource(read(SNIPPET), 'oddEvenSort')!.text).toContain('quiet = changed ? 0 : quiet + 1;');
  });
});

// ---------------------------------------------------------------- on the emulator
const has = (cmd: string, args: string[]) => spawnSync(cmd, args).status === 0;
const cxx = ['c++', 'clang++', 'g++'].find((c) => has(c, ['--version']));
const python = ['python3'].find((c) => has(c, ['--version']));
const INC = path.join(repo, 'labs/common/include');
const BASE = ['-std=c++17', '-O1', '-g', '-fno-omit-frame-pointer', '-pthread', '-I', INC, '-I', path.join(INC, 'pmpp/emu/shim'),
  '-include', 'pmpp/emu/cuda_emu.h'];
const TSAN_OPTIONS = 'abort_on_error=0:exitcode=66:halt_on_error=0:report_thread_leaks=0';

/** Translates the snippet and the driver side by side, so the driver's #include "../s142_oddeven.cu" finds the translation. */
function translate(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch14-s142-'));
  fs.mkdirSync(path.join(dir, 'check'));
  for (const [src, out] of [[SNIPPET, 's142_oddeven.cu'], [DRIVER, 'check/s142_check.cpp']]) {
    const t = spawnSync(python!, [path.join(repo, 'labs/tools/cu2emu.py'), path.join(repo, src), path.join(dir, out)], { encoding: 'utf8' });
    if (t.status !== 0) throw new Error('cu2emu failed:\n' + t.stdout + t.stderr);
  }
  return dir;
}

function build(dir: string, name: string, flags: string[], link: string[]) {
  const obj = path.join(dir, name + '.o');
  const c = spawnSync(cxx!, [...BASE, ...flags, '-c', path.join(dir, 'check/s142_check.cpp'), '-o', obj], { encoding: 'utf8' });
  if (c.status !== 0) throw new Error('compile failed:\n' + c.stderr);
  const exe = path.join(dir, name);
  const l = spawnSync(cxx!, [...link, '-pthread', obj, '-o', exe], { encoding: 'utf8' });
  if (l.status !== 0) throw new Error('link failed:\n' + l.stderr);
  return exe;
}

const list = (s: string) => s.trim().split(/\s+/).filter(Boolean).map(Number);

describe.skipIf(!cxx || !python)('the snippet on the course emulator (labs/ch14/snippets/check/)', () => {
  it('matches the model in all three thread orders: Fig. 14.1\'s rows, the steps the host launches, the sorted output', () => {
    const dir = translate();
    const exe = build(dir, 'emu', ['-fsanitize=address', '-DPMPP_BACKEND_EMU'], ['-fsanitize=address']);
    const r = spawnSync(exe, [], { encoding: 'utf8', env: { ...process.env, ASAN_OPTIONS: 'detect_leaks=0' }, timeout: 170000 });
    expect(r.status, r.stderr).toBe(0);
    const out = r.stdout.split('\n');
    expect(out.filter((l) => l.startsWith('order ')).length).toBe(3);

    // Single steps: Fig. 14.1, and the ten-element odd step of q-threads.
    const fig = walk(FIG_14_1, 6);
    const want = fig.map((s) => s.after.join(' ')).join(' | ') + ' | flags ' + fig.map((s) => (s.flag ? 1 : 0)).join(' ');
    for (const l of out.filter((x) => x.startsWith('rows fig:'))) expect(l.slice('rows fig: '.length)).toBe(want);
    const ten = walk([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], 2);
    const wantTen = ten.map((s) => s.after.join(' ')).join(' | ') + ' | flags 1 1';
    for (const l of out.filter((x) => x.startsWith('rows ten:'))) expect(l.slice('rows ten: '.length)).toBe(wantTen);
    expect(ten[1].after.slice(-1)).toEqual([1]); // i = 9 is out of range on the odd step: the last element is untouched

    // The host loop: steps launched under "two quiet steps", and the output.
    const cases: Record<string, number[]> = {
      fig: FIG_14_1, largest: LARGEST_FIRST, trap: TRAP, trap4: [1, 3, 2, 4], reversed: [9, 8, 7, 6, 5, 4, 3, 2],
      odd7: [3, 2, 1, 0, 5, 4, 6], one: [5], empty: [],
    };
    for (const [name, v] of Object.entries(cases)) {
      const lines = out.filter((x) => x.startsWith(`sort ${name}:`));
      expect(lines.length, name).toBe(3);
      const model = runToSorted(v, 'ours');
      for (const l of lines) {
        const m = /steps (\d+) out(.*)$/.exec(l)!;
        expect(Number(m[1]), l).toBe(model.launched);
        expect(list(m[2]), l).toEqual(model.out);
      }
    }
    expect(out.filter((l) => l === 'random 40: matches std::sort').length).toBe(3);
    expect(out.filter((l) => /^two blocks 520: matches std::sort, steps \d+$/.test(l)).length).toBe(3);
    expect(out.filter((l) => l === 'kernel N=0: untouched').length).toBe(3);
    expect(out).toContain('DONE');
  }, 180000);
});

// ---------------------------------------------------------------- the race-detection flavour
const tsanWorks = (() => {
  if (!cxx) return false;
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch14-tsan-'));
  const src = path.join(dir, 'p.cpp');
  fs.writeFileSync(src, '#include <thread>\nint main(){int x=0;std::thread t([&]{x=1;});t.join();return x==1?0:1;}\n');
  const c = spawnSync(cxx, ['-fsanitize=thread', '-pthread', src, '-o', path.join(dir, 'p')]);
  return c.status === 0 && spawnSync(path.join(dir, 'p')).status === 0;
})();

describe.skipIf(!cxx || !python || !tsanWorks)('the flag under the race-detection emulator (ThreadSanitizer)', () => {
  it('Fig. 14.2 as printed: the plain store to hasChanged is reported as a data race', () => {
    const dir = translate();
    const exe = build(dir, 'race_plain', ['-fsanitize=thread', '-DPMPP_BACKEND_RACE', '-DSNIPPET_AS_PRINTED'], ['-fsanitize=thread']);
    const r = spawnSync(exe, [], { encoding: 'utf8', env: { ...process.env, TSAN_OPTIONS }, timeout: 120000 });
    expect(r.stdout).toContain('race step as printed: flag 1, pairs ordered yes');
    expect(r.stderr).toContain('WARNING: ThreadSanitizer: data race');
    // The report names the store: the line of the driver's copy that holds "*hasChanged = 1;".
    const line = read(DRIVER).split('\n').findIndex((l) => l.includes('*hasChanged = 1;')) + 1;
    expect(r.stderr).toContain(`s142_check.cu:${line}`);
    expect(r.status).toBe(66);
  }, 120000);
  it('the snippet\'s atomicExch: the same step, no report', () => {
    const dir = translate();
    const exe = build(dir, 'race_atomic', ['-fsanitize=thread', '-DPMPP_BACKEND_RACE'], ['-fsanitize=thread']);
    const r = spawnSync(exe, [], { encoding: 'utf8', env: { ...process.env, TSAN_OPTIONS }, timeout: 120000 });
    expect(r.stdout).toContain('race step atomic: flag 1, pairs ordered yes');
    expect(r.stderr).not.toContain('ThreadSanitizer');
    expect(r.status).toBe(0);
  }, 120000);
});
