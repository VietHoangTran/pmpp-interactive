import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  JDS_INDEX_LINE, JDS_LOOP_LINE, JDS_SNIPPET, JDS_STORE_LINE, PER_ITER, RAGGED, RAGGED_PCT, ROW_SUMS, SECTORS, UNPERMUTED,
  UNPERMUTED_RIGHT, questionsWrap, snippetBody,
} from './Wrap';
import { EXERCISE1, FIG17_14, toJDS } from '../../sim/sparse';
import { functionSource } from '../../sim/toolchain';

// The summary page shows exercise 5's JDS kernel, loaded at run time from
// labs/ch17/snippets/spmv_jds.cu (so there is no copy to drift). This test pins the lines the
// page highlights, pins the numbers its questions take from the models, runs the CUDA
// front-end check on the kernel and its driver, and runs the kernel on the course's CPU
// emulator through labs/ch17/snippets/check/spmv_jds_check.cu (AddressSanitizer, three
// thread orders), including the defect the first question describes. The compiler parts are
// skipped when no C++ compiler or python3 is found.
const repo = path.resolve(__dirname, '../../../..');
const read = (p: string) => fs.readFileSync(path.join(repo, p), 'utf8');
const FILE = 'labs/' + JDS_SNIPPET;
const DRIVER = 'labs/ch17/snippets/check/spmv_jds_check.cu';
const code = (s: string) => s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');

describe('the JDS kernel the summary page shows', () => {
  it('has the three highlighted lines, accumulates into the original row, and is ours by its header', () => {
    const s = read(FILE);
    const k = functionSource(s, 'spmv_jds_kernel');
    expect(k!.text.startsWith('__global__ void spmv_jds_kernel(JDSMatrix jdsMatrix, float* x, float* y)')).toBe(true);
    const lines = code(k!.text).split('\n').map((l) => l.trim());
    for (const l of [JDS_LOOP_LINE, JDS_INDEX_LINE, JDS_STORE_LINE]) expect(lines).toContain(l);
    expect(code(k!.text)).not.toMatch(/y\[[^\]]*\] = sum/);
    expect(s).toMatch(/exercise 5 \(p\. 422\) asks for one, and this answer is ours/);
  });
  it('the page shows the file without its header comment and keeps every kernel line', () => {
    const s = read(FILE);
    const shown = snippetBody(s);
    expect(shown.startsWith('//')).toBe(false);
    expect(s.endsWith(shown)).toBe(true);
    expect(shown).toContain('struct JDSMatrix');
    expect(shown).toContain('__global__ void spmv_jds_kernel');
  });
});

describe('the numbers the page and its questions use', () => {
  it('Fig. 17.14 in JDS, longest first: row and iterPtr as printed on p. 416; threads per iteration', () => {
    const j = toJDS(FIG17_14);
    expect(j.row).toEqual([1, 3, 5, 2, 4, 0]);
    expect(j.iterPtr).toEqual([0, 6, 11, 14, 15]);
    expect(PER_ITER).toEqual([6, 5, 3, 1]);
    expect(toJDS(EXERCISE1).row).toEqual([0, 2, 3, 1]);
    expect(toJDS(EXERCISE1).iterPtr).toEqual([0, 4, 7]);
  });
  it('the unpermuted store: row sums all different, only y[4] right', () => {
    expect(ROW_SUMS).toEqual([1, 14, 13, 27, 23, 42]);
    expect(UNPERMUTED).toEqual([14, 27, 42, 13, 23, 1]);
    expect(UNPERMUTED_RIGHT).toBe(1);
  });
  it('CSR over uniform rows of 16: 512 sector requests, 64 distinct sectors; ELL 4 per iteration, 64 in all', () => {
    expect(SECTORS).toEqual({ csrRequests: 512, csrDistinct: 64, ellPerIter: 4, ellDistinct: 64 });
  });
  it('the padded batch: 8,300 real tokens in 128,000 slots', () => {
    expect((RAGGED.seqs - 1) * RAGGED.short + RAGGED.long).toBe(8300);
    expect(RAGGED_PCT).toBeCloseTo(6.484375, 6);
    const q = questionsWrap['ch17/wrap/q-ragged'];
    expect(q.kind === 'numeric' && Math.abs(RAGGED_PCT - q.answer) <= q.answer * (q.tol ?? 0)).toBe(true);
  });
  it('every question sits under this page\'s ids, numeric answers match the models, choices have one right option and a why for each wrong one', () => {
    expect(Object.keys(questionsWrap).sort()).toEqual([
      'ch17/wrap/q-csr-sectors', 'ch17/wrap/q-jds-unpermuted', 'ch17/wrap/q-moe-combine', 'ch17/wrap/q-ragged', 'ch17/wrap/q-transpose',
    ]);
    const num = (id: string) => { const q = questionsWrap[id]; return q.kind === 'numeric' ? q.answer : NaN; };
    expect(num('ch17/wrap/q-jds-unpermuted')).toBe(1);
    expect(num('ch17/wrap/q-csr-sectors')).toBe(64);
    for (const [id, q] of Object.entries(questionsWrap)) {
      expect(q.id).toBe(id);
      if (q.kind !== 'choice') continue;
      expect(q.options.filter((o) => o.correct).length).toBe(1);
      for (const o of q.options) if (!o.correct) expect(o.why).toBeTruthy();
    }
  });
  it('the MoE question\'s premise: two adds into zero commute exactly, three terms in float32 can depend on order', () => {
    const f = Math.fround;
    const a = f(0.1), b = f(0.2);
    expect(f(f(0 + a) + b)).toBe(f(f(0 + b) + a));
    const r = f(1e8);
    const one = f(1), minus = f(-1e8);
    expect(f(f(r + one) + minus)).not.toBe(f(f(r + minus) + one));
  });
});

// ---------------------------------------------------------------- compilers
const has = (cmd: string, args: string[]) => spawnSync(cmd, args).status === 0;
const cxx = ['c++', 'clang++', 'g++'].find((c) => has(c, ['--version']));
const clang = ['clang++'].find((c) => has(c, ['--version']));
const python = ['python3'].find((c) => has(c, ['--version']));
const INC = path.join(repo, 'labs/common/include');

describe.skipIf(!clang)('the CUDA front-end check (clang, stub headers, as labs/pmpp.py runs it)', () => {
  it('parses the kernel and its driver', () => {
    for (const f of [FILE, DRIVER]) {
      const r = spawnSync(clang!, ['-x', 'cuda', '--cuda-host-only', '-nocudainc', '-nocudalib', '-fsyntax-only', '-std=c++17',
        '-I', INC, '-I', path.join(INC, 'pmpp/emu/shim'), '-include', 'pmpp/check/cuda_stub.h', path.join(repo, f)], { encoding: 'utf8' });
      if (r.status !== 0 && /cannot parse|unknown argument|error: language not recognized/.test(r.stderr)) return; // not a CUDA-capable clang
      expect(r.status, f + '\n' + r.stderr).toBe(0);
    }
  }, 60000);
});

describe.skipIf(!cxx || !python)('the kernel on the course emulator (labs/ch17/snippets/check/spmv_jds_check.cu)', () => {
  it('emu: the host-built arrays match Fig. 17.14; right on five matrices in three orders; the unpermuted store gets 5 of 6 wrong', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch17-wrap-'));
    const cpp = path.join(dir, 'jds.cpp');
    const t = spawnSync(python!, [path.join(repo, 'labs/tools/cu2emu.py'), path.join(repo, DRIVER), cpp], { encoding: 'utf8' });
    expect(t.status, t.stdout + t.stderr).toBe(0);
    const exe = path.join(dir, 'jds');
    const c = spawnSync(cxx!, ['-fsanitize=address', '-DPMPP_BACKEND_EMU', '-std=c++17', '-O1', '-g', '-fno-omit-frame-pointer', '-pthread',
      '-I', INC, '-I', path.join(INC, 'pmpp/emu/shim'), '-include', 'pmpp/emu/cuda_emu.h', '-I', path.join(repo, path.dirname(DRIVER)), cpp, '-o', exe],
    { encoding: 'utf8' });
    expect(c.status, c.stderr).toBe(0);
    const r = spawnSync(exe, [], { encoding: 'utf8', env: { ...process.env, ASAN_OPTIONS: 'detect_leaks=0' }, timeout: 120000 });
    expect(r.status, r.stderr).toBe(0);
    expect(r.stdout).toContain('arrays fig17_14: row 1,3,5,2,4,0 iterPtr 0,6,11,14,15');
    const checks = r.stdout.split('\n').filter((l) => l.startsWith('check '));
    expect(checks.length).toBe(5 * 3);
    for (const l of checks) expect(l, l).toMatch(/: ok$/);
    for (const m of ['fig17_1', 'fig17_14', 'exercise1', 'skewed', 'empty_rows'])
      for (const o of ['forward', 'reverse', 'shuffle']) expect(r.stdout).toContain(`check jds ${m} ${o}: ok`);
    expect(r.stdout).toContain(`unpermuted: wrong ${6 - UNPERMUTED_RIGHT} of 6`);
    expect(r.stdout).toContain('DONE');
  }, 180000);
});
