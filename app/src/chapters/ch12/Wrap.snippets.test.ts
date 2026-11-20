import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { chapter } from './index';
import { PAIRS_FILE, PAIR_FRACTION, onePassPairBytes, questionsWrap, threeKernelPairBytes } from './Wrap';
import { traffic } from '../../sim/stablescan';
import { functionSource } from '../../sim/toolchain';

// The summary page shows one kernel of its own, loaded at run time from
// labs/ch12/snippets/wrap_pairs.cu. This test pins the defect the first question is about,
// runs the kernel on the course's CPU emulator (labs/tools/cu2emu.py, pmpp/emu/cuda_emu.h,
// AddressSanitizer on) through labs/ch12/snippets/check/wrap_pairs_test.cu, and pins the
// numbers the other questions state. It also checks the chapter's wiring: every section
// the index lists, and the prediction ids the lab READMEs name. The emulator part is
// skipped when no C++ compiler or python3 is found.
const repo = path.resolve(__dirname, '../../../..');
const read = (p: string) => fs.readFileSync(path.join(repo, p), 'utf8');
const SNIPPET = 'labs/' + PAIRS_FILE;
const snippet = read(SNIPPET);
/** Source without comments, for claims about the code itself. */
const code = (s: string) => s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');

describe('the kernel of the first question', () => {
  it('has the kernel the page names, taking a pair\'s two slots from two counters', () => {
    const k = functionSource(snippet, 'candidatePairs');
    expect(k).not.toBeNull();
    expect(k!.text.startsWith('__global__ void candidatePairs(const float* logits, int cols, float threshold,')).toBe(true);
    const body = code(k!.text);
    expect(body.match(/fetch_add\(/g)?.length).toBe(2);
    expect(body).toContain('outIdx[idxRef.fetch_add(1, cuda::memory_order_relaxed)] = c;');
    expect(body).toContain('outVal[valRef.fetch_add(1, cuda::memory_order_relaxed)] = v;');
    // No barrier, no warp call: on emu nothing runs between one thread's two atomics.
    expect(body).not.toMatch(/__syncthreads|_sync\(|__activemask/);
  });
  it('the right option is the one about interleaving, and no option blames emu\'s orders', () => {
    const q = questionsWrap['ch12/wrap/q-pairs'];
    expect(q.kind).toBe('choice');
    if (q.kind === 'choice') expect(q.options.findIndex((o) => o.correct)).toBe(1);
  });
});

describe('the other questions\' numbers', () => {
  it('three kernels carrying pairs: 8 + 8 + 12 + 8f bytes per logit, 30 at a quarter kept; one pass 4 + 8f', () => {
    expect(PAIR_FRACTION).toBe(0.25);
    expect(threeKernelPairBytes(PAIR_FRACTION)).toBe(30);
    expect(onePassPairBytes(PAIR_FRACTION)).toBe(6);
    const q = questionsWrap['ch12/wrap/q-pairbytes'];
    expect(q.kind === 'numeric' && q.answer).toBe(30);
  });
  it('with 4-byte keys the same accounting is section 12.5\'s model: 30, 10 and 6 bytes at half kept', () => {
    // Keys of 4 bytes instead of 8-byte pairs: the model behind sections 12.5 and 12.6 and lab 12.3.
    expect(traffic('three-kernels', 0.5).total).toBe(30);
    const staged = traffic('staged', 0.5).total;   // 10 plus a few bytes per block for the counts
    expect(staged).toBeGreaterThan(10);
    expect(staged).toBeLessThan(10.1);
    expect(traffic('single-pass', 0.5).total).toBe(6);
  });
  it('every question has exactly one correct option or a numeric answer, under this page\'s ids', () => {
    for (const q of Object.values(questionsWrap)) {
      if (q.kind === 'choice') expect(q.options.filter((o) => o.correct).length).toBe(1);
      else expect(Number.isFinite(q.answer)).toBe(true);
      expect(q.id.startsWith('ch12/wrap/')).toBe(true);
    }
  });
});

describe('the chapter\'s wiring', () => {
  it('lists every section once, in the book\'s order, each with a file that exists', () => {
    expect(chapter.sections.map((s) => s.id)).toEqual(['start', '12.1', '12.2', '12.3', '12.4', '12.5', '12.6', '12.7', '12.8', 'labs', 'wrap']);
    const index = fs.readFileSync(path.join(__dirname, 'index.tsx'), 'utf8');
    expect(index).not.toMatch(/TEMPORARY/);
    for (const m of index.matchAll(/import\('\.\/(\w+)'\)/g)) expect(fs.existsSync(path.join(__dirname, m[1] + '.tsx'))).toBe(true);
  });
  it('the labs page asks for the prediction ids the lab READMEs name', () => {
    const labs = fs.readFileSync(path.join(__dirname, 'Labs.tsx'), 'utf8');
    for (const dir of ['lab1_unstable', 'lab2_filterhunt', 'lab3_stable', 'lab4_candidates', 'lab5_inplace']) {
      const ids = [...read(`labs/ch12/${dir}/README.md`).matchAll(/`(ch12\/[\w.]+\/p-[\w-]+)`/g)].map((m) => m[1]);
      expect(ids.length).toBeGreaterThan(0);
      for (const id of ids) expect(labs).toContain(`id="${id}"`);
    }
  });
  it('the labs page reads the lab directories that exist, with the files each lab has', () => {
    for (const dir of ['lab1_unstable', 'lab2_filterhunt', 'lab3_stable', 'lab4_candidates'])
      for (const f of ['kernel.cu', 'kernel.metal', 'launch.mm']) expect(fs.existsSync(path.join(repo, `labs/ch12/${dir}/solution/${f}`))).toBe(true);
    expect(fs.existsSync(path.join(repo, 'labs/ch12/lab5_inplace/solution/kernel.cu'))).toBe(true);
    expect(fs.existsSync(path.join(repo, 'labs/ch12/lab5_inplace/solution/kernel.metal'))).toBe(false);
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
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch12-wrap-'));
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

describe.skipIf(!cxx || !python)('the kernel on the course emulator (labs/ch12/snippets/check/wrap_pairs_test.cu)', () => {
  it('emu cannot see the defect: in all three thread orders both counts are right and every pair matches; so does the one-slot fix', () => {
    const exe = build([SNIPPET, 'labs/ch12/snippets/check/wrap_pairs_test.cu'], 'pairs');
    const r = spawnSync(exe, [], { encoding: 'utf8', env: { ...process.env, ASAN_OPTIONS: 'detect_leaks=0' }, timeout: 120000 });
    expect(r.status).toBe(0);
    expect(r.stdout).toContain('DONE');
    const lines = r.stdout.split('\n').filter((l) => /^(shown|fixed) /.test(l));
    expect(lines.length).toBe(6);
    const seen = new Set<string>();
    for (const l of lines) {
      const m = /^(shown|fixed) ([a-z ]+): count (\d+) (\d+) of (\d+), pairs (\d+) of (\d+), columns (yes|no)$/.exec(l);
      expect(m, l).not.toBeNull();
      const [, kernel, order, idx, val, want, matched, n, cols] = m!;
      seen.add(`${kernel} ${order}`);
      expect(Number(want)).toBeGreaterThan(0);
      expect(Number(idx)).toBe(Number(want));
      expect(Number(val)).toBe(Number(want));
      expect(Number(matched)).toBe(Number(n));
      expect(cols).toBe('yes');
    }
    for (const k of ['shown', 'fixed']) for (const o of ['in order', 'reversed', 'shuffled']) expect(seen.has(`${k} ${o}`)).toBe(true);
  }, 120000);
});
