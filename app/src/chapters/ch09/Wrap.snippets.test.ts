import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { APPEND, EARLY, EARLY_FILE, MOE, appendMs, earlyExitLost, earlyImage, moeAtomics, questionsWrap } from './Wrap';
import { functionSource } from '../../sim/toolchain';

// The summary page shows one kernel of its own, loaded at run time from
// labs/ch09/snippets/wrap_early_exit.cu. This test ties that file to Fig. 9.10's shape and to
// the page's model (earlyExitLost), runs it on the course's CPU emulator through
// labs/ch09/snippets/check/wrap_early_exit_test.cu (labs/tools/cu2emu.py, pmpp/emu/cuda_emu.h,
// AddressSanitizer on), and pins the numbers the other questions state. The emulator part is
// skipped when no C++ compiler or python3 is found.
const repo = path.resolve(__dirname, '../../../..');
const read = (p: string) => fs.readFileSync(path.join(repo, p), 'utf8');
const SNIPPET = 'labs/' + EARLY_FILE;
const snippet = read(SNIPPET);
const code = (s: string) => s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');

describe('the kernel of the first question', () => {
  it('is Fig. 9.10\'s shape with an early return, and the API the chapter uses', () => {
    const k = functionSource(snippet, 'histogramEarlyExit');
    expect(k).not.toBeNull();
    const body = code(k!.text);
    expect(k!.text.startsWith('__global__ void histogramEarlyExit(const unsigned char* image, unsigned int* bins, unsigned int n) {')).toBe(true);
    expect(body).toMatch(/if\(i >= n\) return;/);
    expect(body.match(/__syncthreads\(\);/g)?.length).toBe(2);
    expect(body).toContain('cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref(bins_s[image[i]]);');
    expect(body).toContain('cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);');
    expect(body.match(/cuda::memory_order_relaxed/g)?.length).toBe(2);
    // Zeroing and commit hand out the bins by thread index, which is what the model assumes.
    expect(body.match(/for\(unsigned int b = threadIdx\.x; b < NUM_BINS; b \+= blockDim\.x\)/g)?.length).toBe(2);
  });
  it('the model: 232 threads stay in the last block; bins 232 to 255 each lose one count', () => {
    const { live, lost } = earlyExitLost(earlyImage(EARLY.n), EARLY.block);
    expect(live).toBe(232);
    expect([...lost.keys()].sort((a, b) => a - b)).toEqual(Array.from({ length: 24 }, (_, i) => 232 + i));
    expect([...lost.values()].every((v) => v === 1)).toBe(true);
    // A full last block loses nothing.
    expect(earlyExitLost(earlyImage(1024), 256).lost.size).toBe(0);
    const q = questionsWrap['ch09/wrap/q-early-exit'];
    expect(q.kind === 'choice' && q.options.findIndex((o) => o.correct)).toBe(1);
  });
});

describe('the other questions\' numbers', () => {
  it('mixture of experts: 32,768 updates, 128 blocks, at most 8,192 commits', () => {
    expect(MOE.tokens * MOE.topK).toBe(32768);
    expect(moeAtomics()).toEqual({ updates: 32768, blocks: 128, commits: 8192 });
  });
  it('one counter in the book\'s DRAM model: 100,000 atomics take 40 ms', () => {
    expect(APPEND.candidates).toBe(100000);
    expect(appendMs()).toBeCloseTo(40, 9);
    const q = questionsWrap['ch09/wrap/q-append'];
    expect(q.kind === 'numeric' && q.answer).toBe(40);
  });
  it('every question has exactly one correct option or a numeric answer, under this page\'s ids', () => {
    for (const [k, q] of Object.entries(questionsWrap)) {
      expect(q.id).toBe(k);
      expect(q.id.startsWith('ch09/wrap/')).toBe(true);
      if (q.kind === 'choice') expect(q.options.filter((o) => o.correct).length).toBe(1);
      else expect(Number.isFinite(q.answer)).toBe(true);
      expect(q.source).toBeTruthy();
    }
  });
});

// ---------------------------------------------------------------- on the emulator
const has = (cmd: string, args: string[]) => spawnSync(cmd, args).status === 0;
const cxx = ['c++', 'clang++', 'g++'].find((c) => has(c, ['--version']));
const python = ['python3'].find((c) => has(c, ['--version']));
const INC = path.join(repo, 'labs/common/include');
const FLAGS = ['-std=c++17', '-O1', '-g', '-fsanitize=address', '-fno-omit-frame-pointer', '-pthread', '-DPMPP_BACKEND_EMU',
  '-I', INC, '-I', path.join(INC, 'pmpp/emu/shim'), '-include', 'pmpp/emu/cuda_emu.h'];

function build(sources: string[], name: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch09-wrap-'));
  const objs = sources.map((src) => {
    const stem = path.basename(src, '.cu');
    const cpp = path.join(dir, stem + '.cpp');
    const obj = path.join(dir, stem + '.o');
    const t = spawnSync(python!, [path.join(repo, 'labs/tools/cu2emu.py'), path.join(repo, src), cpp], { encoding: 'utf8' });
    if (t.status !== 0) throw new Error('cu2emu failed on ' + src + ':\n' + t.stdout + t.stderr);
    const c = spawnSync(cxx!, [...FLAGS, '-I', path.dirname(path.join(repo, src)), '-c', cpp, '-o', obj], { encoding: 'utf8' });
    if (c.status !== 0) throw new Error('compile failed for ' + src + ':\n' + c.stderr);
    return obj;
  });
  const exe = path.join(dir, name);
  const l = spawnSync(cxx!, ['-fsanitize=address', '-pthread', ...objs, '-o', exe], { encoding: 'utf8' });
  if (l.status !== 0) throw new Error('link failed:\n' + l.stderr);
  return exe;
}

describe.skipIf(!cxx || !python)('the kernel on the course emulator (labs/ch09/snippets/check/wrap_early_exit_test.cu)', () => {
  it('loses exactly the counts the model says, at 256 and 128 threads, under three orders; guarding the update instead fixes it', () => {
    const exe = build([SNIPPET, 'labs/ch09/snippets/check/wrap_early_exit_test.cu'], 'early');
    const r = spawnSync(exe, [], { encoding: 'utf8', env: { ...process.env, ASAN_OPTIONS: 'detect_leaks=0' }, timeout: 120000 });
    expect(r.status).toBe(0);
    expect(r.stdout).toContain('DONE');
    const lines = r.stdout.split('\n').filter((l) => /^(shown|fixed) /.test(l));
    expect(lines.length).toBe(12);
    const image = earlyImage(EARLY.n);
    const want = new Array(256).fill(0);
    for (const v of image) want[v]++;
    for (const l of lines) {
      const m = /^(shown|fixed) (\d+) ([a-z ]+): wrong (\d+) of 256:(.*)$/.exec(l)!;
      expect(m).not.toBeNull();
      const [, kernel, block, , wrong, list] = m;
      if (kernel === 'fixed') {
        expect(Number(wrong)).toBe(0);
        continue;
      }
      const { lost } = earlyExitLost(image, Number(block));
      const expected = [...lost.entries()].sort((a, b) => a[0] - b[0]).map(([b, k]) => `${b}:${want[b] - k}:${want[b]}`);
      expect(list.trim().split(' ')).toEqual(expected);
      expect(Number(wrong)).toBe(lost.size);
    }
  }, 120000);
});
