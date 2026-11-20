import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { chapter } from './index';
import { CENTRE, CENTRE_FILE, ODD_BLOCK, questionsWrap, sharedTreeCovers, threadZeroColumns } from './Wrap';
import { strides } from '../../sim/reduction';
import { blockCosts } from '../../sim/shuffle';
import { functionSource } from '../../sim/toolchain';

// The summary page shows one kernel of its own, loaded at run time from
// labs/ch10/snippets/wrap_centre.cu. This test ties that file's helpers to lab 10.1, runs it
// on the course's CPU emulator (labs/tools/cu2emu.py, pmpp/emu/cuda_emu.h, AddressSanitizer
// on) through labs/ch10/snippets/check/wrap_centre_test.cu, and pins the numbers the other
// questions state. It also checks the chapter's wiring: every section the index lists, and
// the prediction ids the lab READMEs name. The emulator part is skipped when no C++
// compiler or python3 is found.
const repo = path.resolve(__dirname, '../../../..');
const read = (p: string) => fs.readFileSync(path.join(repo, p), 'utf8');
const SNIPPET = 'labs/' + CENTRE_FILE;
const LAB = 'labs/ch10/lab1_sum/solution/kernel.cu';
const snippet = read(SNIPPET);
/** Source without comments, for claims about the code itself. */
const code = (s: string) => s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');
/** A function's text without the comment lines above it. */
const body = (s: string, name: string) =>
  functionSource(s, name)!.text.split('\n').filter((l, i, a) => !(l.trim().startsWith('//') && a.slice(0, i).every((x) => x.trim().startsWith('//')))).join('\n');

describe('the kernel of the first question', () => {
  it('shares its mask type, defines, warpIdx, laneIdx and warp_reduce with lab 10.1, character for character', () => {
    const lab = read(LAB);
    const typedef = (s: string) => /#if defined\(__HIPCC__\)[\s\S]*?#endif/.exec(s)?.[0];
    expect(typedef(snippet)).toBeDefined();
    expect(typedef(snippet)).toBe(typedef(lab));
    for (const d of ['#define MAX_BLOCK_DIM 1024', '#define MIN_WARP_SIZE 32']) {
      const line = (s: string) => s.split('\n').find((l) => l.startsWith(d));
      expect(line(snippet)).toBe(line(lab));
    }
    for (const f of ['warpIdx', 'laneIdx', 'warp_reduce']) expect(body(snippet, f)).toBe(body(lab, f));
  });
  it('has the kernel the page names, and no 32-bit mask literal or literal width in its code', () => {
    const k = functionSource(snippet, 'centreRows');
    expect(k).not.toBeNull();
    expect(k!.text.startsWith('__global__ void centreRows(const float* x, float* y, int cols) {')).toBe(true);
    expect(code(snippet)).not.toMatch(/0xffffffff/i);
    expect(code(k!.text)).not.toMatch(/\b32\b/);
    // The defect is the one the question is about: the mean is taken from the s each thread holds.
    expect(k!.text).toContain('float mean = s / cols;');
    expect(k!.text).not.toMatch(/mean_s|__syncthreads\(\);\s*float mean/);
  });
  it('the right option names the columns thread 0 writes', () => {
    expect(threadZeroColumns(CENTRE.block, CENTRE.cols)).toEqual([0, 256, 512, 768]);
    expect(threadZeroColumns(64, 1000).length).toBe(16);
    const q = questionsWrap['ch10/wrap/q-centre'];
    expect(q.kind === 'choice' && q.options.findIndex((o) => o.correct)).toBe(1);
  });
});

describe('the other questions\' numbers', () => {
  it('Fig. 10.10 with a 96-thread block keeps 128 of 192 values; a power of two keeps all of them', () => {
    expect(ODD_BLOCK).toBe(96);
    expect(sharedTreeCovers(96)).toBe(128);
    for (const B of [32, 64, 128, 256, 512, 1024]) expect(sharedTreeCovers(B)).toBe(2 * B);
    const q = questionsWrap['ch10/wrap/q-oddblock'];
    expect(q.kind === 'numeric' && q.answer).toBe(128);
  });
  it('the two models of Fig. 10.10 agree: 10 barriers at 1,024 threads', () => {
    expect(strides('shared', 1024).length).toBe(10);
    expect(blockCosts('shared', 1024, 32).barriers).toBe(10);
  });
  it('every question has exactly one correct option or a numeric answer, under this page\'s ids', () => {
    for (const q of Object.values(questionsWrap)) {
      if (q.kind === 'choice') expect(q.options.filter((o) => o.correct).length).toBe(1);
      else expect(Number.isFinite(q.answer)).toBe(true);
      expect(q.id.startsWith('ch10/wrap/')).toBe(true);
    }
  });
});

describe('the chapter\'s wiring', () => {
  it('lists every section once, in the book\'s order, each with a file that exists', () => {
    expect(chapter.sections.map((s) => s.id)).toEqual(['start', '10.1', '10.2', '10.3', '10.4', '10.5', '10.6', '10.7', '10.8', '10.9', '10.10', 'labs', 'wrap']);
    const index = fs.readFileSync(path.join(__dirname, 'index.tsx'), 'utf8');
    expect(index).not.toMatch(/TEMPORARY/);
    for (const m of index.matchAll(/import\('\.\/(\w+)'\)/g)) expect(fs.existsSync(path.join(__dirname, m[1] + '.tsx'))).toBe(true);
  });
  it('the labs page asks for the prediction ids the lab READMEs name', () => {
    const labs = fs.readFileSync(path.join(__dirname, 'Labs.tsx'), 'utf8');
    for (const dir of ['lab1_sum', 'lab2_reducehunt', 'lab3_softmax_argmax']) {
      const ids = [...read(`labs/ch10/${dir}/README.md`).matchAll(/`(ch10\/[\w.]+\/p-[\w-]+)`/g)].map((m) => m[1]);
      expect(ids.length).toBeGreaterThan(0);
      for (const id of ids) expect(labs).toContain(`id="${id}"`);
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
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch10-wrap-'));
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

describe.skipIf(!cxx || !python)('the kernel on the course emulator (labs/ch10/snippets/check/wrap_centre_test.cu)', () => {
  it('only the columns thread 0 writes are right, in every row, at 256 and 64 threads, under three orders; the hand-back fixes it', () => {
    const exe = build([SNIPPET, 'labs/ch10/snippets/check/wrap_centre_test.cu'], 'centre');
    const r = spawnSync(exe, [], { encoding: 'utf8', env: { ...process.env, ASAN_OPTIONS: 'detect_leaks=0' }, timeout: 120000 });
    expect(r.status).toBe(0);
    const lines = r.stdout.split('\n').filter((l) => /^(shown|fixed) /.test(l));
    expect(lines.length).toBe(12);
    const rows = 3;
    for (const l of lines) {
      const m = /^(shown|fixed) (\d+) ([a-z ]+): right (\d+) of (\d+):(.*)$/.exec(l)!;
      expect(m).not.toBeNull();
      const [, kernel, block, , right, total, list] = m;
      if (kernel === 'fixed') {
        expect(Number(right)).toBe(Number(total));
        continue;
      }
      const want = Array.from({ length: rows }, (_, row) => threadZeroColumns(Number(block), CENTRE.cols).map((c) => `${row}:${c}`)).flat();
      expect(list.trim().split(' ')).toEqual(want);
      expect(Number(right)).toBe(want.length);
    }
  }, 120000);
});
