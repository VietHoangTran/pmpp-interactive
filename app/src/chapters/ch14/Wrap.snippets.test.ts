import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { functionSource } from '../../sim/toolchain';
import { SELECT_BINS, SELECT_K, passes, questionsWrap, rowBits, selectBin } from './Wrap';

// The wrap page shows labs/ch14/snippets/wrap_blocksort.cu, loaded at run time, for the
// question ch14/wrap/q-blocksort. What can drift: what the question says the kernel is, that
// the fix is one added line, and what it says the emulator shows. The last group builds the
// snippet with its driver (labs/ch14/snippets/check/wrap_check.cu, which holds the corrected
// copy) for the course's CPU emulator (labs/tools/cu2emu.py, pmpp/emu/cuda_emu.h,
// AddressSanitizer on) and runs it under the three thread orders. It is skipped when no C++
// compiler or python3 is found. The numeric questions' answers are pinned here too.
const repo = path.resolve(__dirname, '../../../..');
const SNIPPET = 'labs/ch14/snippets/wrap_blocksort.cu';
const DRIVER = 'labs/ch14/snippets/check/wrap_check.cu';
const read = (p: string) => fs.readFileSync(path.join(repo, p), 'utf8');
const page = fs.readFileSync(path.join(__dirname, 'Wrap.tsx'), 'utf8').replace(/\s+/g, ' ');

describe('the snippet and its corrected copy', () => {
  const shown = functionSource(read(SNIPPET), 'blockOddEvenSort')!.text.split('\n');
  // The driver's copy sits in namespace fixed; read from inside it, where it is at depth 0.
  const drv = read(DRIVER);
  const open = 'namespace fixed {';
  expect(drv).toContain(open);
  const fixed = functionSource(drv.slice(drv.indexOf(open) + open.length), 'blockOddEvenSort')!.text.split('\n');
  it('the page loads the file itself and shows only the kernel', () => {
    expect(page).toContain("export const BLOCKSORT_FILE = 'ch14/snippets/wrap_blocksort.cu';");
    expect(page).toContain('useLabFile(BLOCKSORT_FILE)');
    expect(page).toContain("functionSource(snip, 'blockOddEvenSort')");
  });
  it('has no barrier between steps, and the fix adds exactly one __syncthreads() as the last statement of the loop body', () => {
    const loop = shown.findIndex((l) => l.includes('for (unsigned int step = 0; step < n; ++step)'));
    expect(loop).toBeGreaterThan(0);
    expect(shown.filter((l) => l.includes('__syncthreads()')).length).toBe(2);
    expect(fixed.length).toBe(shown.length + 1);
    const added = fixed.findIndex((l, k) => l !== shown[k]);
    expect(fixed[added].trim()).toBe('__syncthreads();');
    expect([...fixed.slice(0, added), ...fixed.slice(added + 1)]).toEqual(shown);
    // The added line closes the loop body: the line after it is the loop's closing brace.
    expect(fixed[added + 1].trim()).toBe('}');
    expect(fixed.slice(loop, added).some((l) => l.includes('s[i + 1] = tmp;'))).toBe(true);
  });
  it('the launch the question names matches the driver\'s', () => {
    expect(page).toContain('<code>(n + 1) / 2</code>');
    expect(drv).toContain('<<<1, (n + 1) / 2>>>');
    expect(read(SNIPPET)).toContain('#define MAXN 1024');
    expect(page).toContain('up to 1,024 keys');
  });
});

describe('the numeric questions', () => {
  it('q-packed-rows: 64 rows need 6 bits, 38 bits in 8-bit digits is 5 passes', () => {
    expect(rowBits(64)).toBe(6);
    expect(passes(38, 8)).toBe(5);
    expect((questionsWrap['ch14/wrap/q-packed-rows'] as { answer: number }).answer).toBe(5);
  });
  it('q-select-bin: running totals 0, 3, 15, 55; the 50th is in the fourth bin; 15 above it; 35 still to pick', () => {
    expect(selectBin(SELECT_BINS, SELECT_K)).toEqual({ bin: 3, above: 15, still: 35 });
    expect((questionsWrap['ch14/wrap/q-select-bin'] as { answer: number }).answer).toBe(35);
  });
});

// ---------------------------------------------------------------- on the emulator
const has = (cmd: string, args: string[]) => spawnSync(cmd, args).status === 0;
const cxx = ['c++', 'clang++', 'g++'].find((c) => has(c, ['--version']));
const python = ['python3'].find((c) => has(c, ['--version']));
const INC = path.join(repo, 'labs/common/include');

describe.skipIf(!cxx || !python)('the snippet on the course emulator (labs/ch14/snippets/check/wrap_check.cu)', () => {
  it('as shown it leaves every case unsorted in all three thread orders; with the barrier it sorts them all', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch14-wrap-'));
    fs.mkdirSync(path.join(dir, 'check'));
    for (const [src, out] of [[SNIPPET, 'wrap_blocksort.cu'], [DRIVER, 'check/wrap_check.cpp']]) {
      const t = spawnSync(python!, [path.join(repo, 'labs/tools/cu2emu.py'), path.join(repo, src), path.join(dir, out)], { encoding: 'utf8' });
      expect(t.status, t.stdout + t.stderr).toBe(0);
    }
    const obj = path.join(dir, 'emu.o');
    const exe = path.join(dir, 'emu');
    const c = spawnSync(cxx!, ['-std=c++17', '-O1', '-g', '-fno-omit-frame-pointer', '-pthread', '-I', INC, '-I', path.join(INC, 'pmpp/emu/shim'),
      '-include', 'pmpp/emu/cuda_emu.h', '-fsanitize=address', '-DPMPP_BACKEND_EMU', '-c', path.join(dir, 'check/wrap_check.cpp'), '-o', obj], { encoding: 'utf8' });
    expect(c.status, c.stderr).toBe(0);
    const l = spawnSync(cxx!, ['-fsanitize=address', '-pthread', obj, '-o', exe], { encoding: 'utf8' });
    expect(l.status, l.stderr).toBe(0);
    const r = spawnSync(exe, [], { encoding: 'utf8', env: { ...process.env, ASAN_OPTIONS: 'detect_leaks=0' }, timeout: 170000 });
    expect(r.status, r.stderr).toBe(0);
    const out = r.stdout.split('\n');
    expect(out.filter((x) => x.startsWith('order ')).length).toBe(3);
    const cases = ['reversed 64', 'random 64', 'random 129', 'reversed 256'];
    for (const k of cases) {
      const shown = out.filter((x) => x.startsWith(`shown ${k}:`));
      const fixed = out.filter((x) => x.startsWith(`fixed ${k}:`));
      expect(shown.length, k).toBe(3);
      expect(fixed.length, k).toBe(3);
      // The emulator runs one thread at a time, so the values survive; the order does not.
      for (const x of shown) expect(x, k).toBe(`shown ${k}: sorted no, permutation yes`);
      for (const x of fixed) expect(x, k).toBe(`fixed ${k}: sorted yes, permutation yes`);
    }
    expect(out.filter((x) => x === 'fixed zero-one 8: 256 of 256 sorted').length).toBe(3);
    expect(out).toContain('DONE');
    // What the question says about it.
    expect(page).toContain('leaves 64, 129 and 256 keys unsorted in all three thread orders');
    expect(page).toContain('including all 256 lists of eight 0s and 1s');
  }, 180000);
});
