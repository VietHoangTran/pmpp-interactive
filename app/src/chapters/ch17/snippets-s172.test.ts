import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  COO_ATOMIC_LINES, CSC_LOAD_LINE, CSR_LOOP_LINE, ELL_ACCUMULATE_LINE, ELL_INDEX_LINE, ELL_PAD_TEST_LINE, SNIPPETS,
  body, questionsS172, scoreRows,
} from './S172';
import { FIG17_1, FIG17_13 } from '../../sim/sparse';
import { functionSource } from '../../sim/toolchain';

// Page 17.2 shows four kernels, loaded at run time from labs/ch17/snippets/ (so there is no
// copy to drift). This test pins the lines the page's prose and highlights name, runs the
// CUDA front-end check on every snippet, and runs them on the course's CPU emulator through
// labs/ch17/snippets/check/spmv_check.cu: emu (AddressSanitizer, three thread orders) for
// correctness and for the two demonstrations the questions rely on, and race
// (ThreadSanitizer) for the COO atomic. The compiler parts are skipped when no C++ compiler
// or python3 is found.
const repo = path.resolve(__dirname, '../../../..');
const read = (p: string) => fs.readFileSync(path.join(repo, p), 'utf8');
const file = (k: keyof typeof SNIPPETS) => 'labs/' + SNIPPETS[k];
const lines = (s: string) => s.split('\n').map((l) => l.trim());
/** Source without comments. */
const code = (s: string) => s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');

describe('the lines the page names are in the snippets', () => {
  it('COO: the book\'s atomic_ref with device scope and relaxed order, and the include the page leaves out', () => {
    const s = read(file('coo'));
    for (const l of COO_ATOMIC_LINES) expect(lines(s)).toContain(l);
    expect(s).toContain('#include <cuda/atomic>');
    const k = functionSource(s, 'spmv_coo_kernel');
    expect(k!.text.startsWith('__global__ void spmv_coo_kernel(COOMatrix cooMatrix, float* x, float* y)')).toBe(true);
  });
  it('CSR: the loop over rowPtrs[row] up to rowPtrs[row + 1], and an accumulate', () => {
    const s = read(file('csr'));
    expect(lines(s)).toContain(CSR_LOOP_LINE);
    expect(code(functionSource(s, 'spmv_csr_kernel')!.text)).toContain('y[row] += sum;');
  });
  it('ELL: the index line, the accumulate (not the page\'s assignment) and the pad test, in the two kernels', () => {
    const s = read(file('ell'));
    const book = code(functionSource(s, 'spmv_ell_kernel')!.text);
    const full = code(functionSource(s, 'spmv_ell_full_kernel')!.text);
    for (const k of [book, full]) {
      expect(k).toContain(ELL_INDEX_LINE);
      expect(k).toContain(ELL_ACCUMULATE_LINE);
      expect(k).not.toMatch(/y\[row\] = sum/);
    }
    expect(book).toContain('t < ellMatrix.nnzPerRow[row]');
    expect(full).toContain('t < ellMatrix.width');
    expect(full).toContain(ELL_PAD_TEST_LINE);
    // -1 needs a signed column index.
    expect(code(s)).toMatch(/int\* colIdx;/);
    expect(code(s)).not.toMatch(/unsigned int\* colIdx/);
    expect(code(s)).toMatch(/\bint col = ellMatrix\.colIdx\[i\];/);
  });
  it('CSC: x[col] loaded once before the loop, singular field names, an atomic add', () => {
    const s = read(file('csc'));
    const k = code(functionSource(s, 'spmv_csc_kernel')!.text);
    expect(k).toContain(CSC_LOAD_LINE);
    expect(k.indexOf(CSC_LOAD_LINE)).toBeLessThan(k.indexOf('for ('));
    expect(k).toContain('cscMatrix.rowIdx[i]');
    expect(k).toContain('cscMatrix.value[i]');
    expect(k).not.toMatch(/rowIdxs|values\[/);
    expect(k).toContain('y_ref.fetch_add(inValue*value, cuda::memory_order_relaxed);');
  });
  it('the page shows each file without its header comment and keeps every kernel line', () => {
    for (const k of Object.keys(SNIPPETS) as (keyof typeof SNIPPETS)[]) {
      const s = read(file(k));
      const shown = body(s);
      expect(shown.startsWith('//')).toBe(false);
      expect(s.endsWith(shown)).toBe(true);
      expect(shown).toMatch(/__global__ void spmv_/);
    }
  });
});

describe('the page\'s questions and the comparison table', () => {
  it('every question has one correct option or a numeric answer, under this page\'s ids', () => {
    expect(Object.keys(questionsS172).sort()).toEqual([
      'ch17/17.2/q-coo-y1', 'ch17/17.2/q-csc-use', 'ch17/17.2/q-ell-assign', 'ch17/17.2/q-ell-index',
      'ch17/17.2/q-ell-loop', 'ch17/17.2/q-ex1-pads', 'ch17/17.2/q-ex2-missing', 'ch17/17.2/q-pad-marker',
    ]);
    for (const [id, q] of Object.entries(questionsS172)) {
      expect(q.id).toBe(id);
      if (q.kind === 'choice') expect(q.options.filter((o) => o.correct).length).toBe(1);
    }
    const num = (id: string) => { const q = questionsS172[id]; return q.kind === 'numeric' ? q.answer : NaN; };
    expect([num('ch17/17.2/q-coo-y1'), num('ch17/17.2/q-ell-index'), num('ch17/17.2/q-ex1-pads')]).toEqual([3, 17, 1]);
  });
  it('the table on Fig. 17.1: COO and CSC collide on y, CSR and CSC read 4 runs, ELL 1; ELL\'s full width trades waiting for pads', () => {
    const r = Object.fromEntries(scoreRows(FIG17_1, 2).map((x) => [x.key, x]));
    expect([r.coo.summary.atomicUpdates, r.coo.summary.sharedY]).toEqual([8, 3]);
    expect([r.csc.summary.atomicUpdates, r.csc.summary.sharedY]).toEqual([8, 3]);
    expect(r.csr.summary.atomicUpdates).toBe(0);
    expect([r.coo.worst, r.csr.worst, r.ell.worst, r.jds.worst, r.csc.worst]).toEqual([1, 4, 1, 1, 4]);
    expect([r.ell.summary.idle, r.ell.summary.pad, r['ell-full'].summary.idle, r['ell-full'].summary.pad]).toEqual([4, 0, 0, 4]);
    expect(r.hyb.summary.atomicUpdates).toBe(1); // row 1's 9 spills at K = 2
  });
  it('the table on Fig. 17.13: ELL pads 22 slots, the hybrid at K = 2 pads 3', () => {
    const r = Object.fromEntries(scoreRows(FIG17_13, 2).map((x) => [x.key, x]));
    expect(r.ell.storage.pads).toBe(22);
    expect(r.hyb.storage.pads).toBe(3);
  });
});

// ---------------------------------------------------------------- compilers
const has = (cmd: string, args: string[]) => spawnSync(cmd, args).status === 0;
const cxx = ['c++', 'clang++', 'g++'].find((c) => has(c, ['--version']));
const clang = ['clang++'].find((c) => has(c, ['--version']));
const python = ['python3'].find((c) => has(c, ['--version']));
const INC = path.join(repo, 'labs/common/include');
const DRIVER = 'labs/ch17/snippets/check/spmv_check.cu';
const ALL = [...Object.keys(SNIPPETS).map((k) => file(k as keyof typeof SNIPPETS)), DRIVER];

describe.skipIf(!clang)('the CUDA front-end check (clang, stub headers, as labs/pmpp.py runs it)', () => {
  it('parses every snippet and the driver', () => {
    for (const f of ALL) {
      const r = spawnSync(clang!, ['-x', 'cuda', '--cuda-host-only', '-nocudainc', '-nocudalib', '-fsyntax-only', '-std=c++17',
        '-I', INC, '-I', path.join(INC, 'pmpp/emu/shim'), '-include', 'pmpp/check/cuda_stub.h', path.join(repo, f)], { encoding: 'utf8' });
      if (r.status !== 0 && /cannot parse|unknown argument|error: language not recognized/.test(r.stderr)) return; // not a CUDA-capable clang
      expect(r.status, f + '\n' + r.stderr).toBe(0);
    }
  }, 60000);
});

function build(flavour: 'emu' | 'race'): string | null {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch17-s172-'));
  const cpp = path.join(dir, 'all.cpp');
  const t = spawnSync(python!, [path.join(repo, 'labs/tools/cu2emu.py'), path.join(repo, DRIVER), cpp], { encoding: 'utf8' });
  if (t.status !== 0) throw new Error('cu2emu failed:\n' + t.stdout + t.stderr);
  const san = flavour === 'emu' ? ['-fsanitize=address', '-DPMPP_BACKEND_EMU'] : ['-fsanitize=thread', '-DPMPP_BACKEND_RACE'];
  const exe = path.join(dir, flavour);
  const c = spawnSync(cxx!, [...san, '-std=c++17', '-O1', '-g', '-fno-omit-frame-pointer', '-pthread', '-I', INC, '-I', path.join(INC, 'pmpp/emu/shim'),
    '-include', 'pmpp/emu/cuda_emu.h', '-I', path.join(repo, path.dirname(DRIVER)), cpp, '-o', exe], { encoding: 'utf8' });
  if (c.status !== 0) {
    if (flavour === 'race') return null; // ThreadSanitizer is not available everywhere
    throw new Error('compile failed:\n' + c.stderr);
  }
  return exe;
}

describe.skipIf(!cxx || !python)('the snippets on the course emulator (labs/ch17/snippets/check/spmv_check.cu)', () => {
  it('emu: every kernel right on Fig. 17.1 and a skewed matrix in all three orders; the page\'s y[row] = sum loses rows; column-0 pads give NaN', () => {
    const exe = build('emu')!;
    const r = spawnSync(exe, [], { encoding: 'utf8', env: { ...process.env, ASAN_OPTIONS: 'detect_leaks=0' }, timeout: 120000 });
    expect(r.status, r.stderr).toBe(0);
    expect(r.stdout).toContain('DONE');
    const checks = r.stdout.split('\n').filter((l) => l.startsWith('check '));
    expect(checks.length).toBe(7 * 2 * 3);
    for (const l of checks) expect(l, l).toMatch(/: ok$/);
    for (const k of ['coo', 'csr', 'ell', 'ell-full', 'ell-aligned', 'ell-aligned-full', 'csc'])
      for (const m of ['fig17_1', 'skewed']) for (const o of ['forward', 'reverse', 'shuffle']) expect(r.stdout).toContain(`check ${k} ${m} ${o}: ok`);
    // Question q-ell-assign: COO part first, then the page's line: every row wrong; with += none.
    expect(r.stdout).toContain('assign page: wrong 4 of 4');
    expect(r.stdout).toContain('assign snippet: wrong 0 of 4');
    // Question q-pad-marker: pads holding column 0 with x[0] = +Inf: the three padded rows are NaN; -1 pads: none.
    expect(r.stdout).toContain('padnan 0: nan 3');
    expect(r.stdout).toContain('padnan -1: nan 0');
  }, 180000);
  it('race: the snippet\'s COO atomic reports nothing; a copy with a plain += is reported on its update line', (ctx) => {
    const exe = build('race');
    if (!exe) return ctx.skip();
    const env = { ...process.env, TSAN_OPTIONS: 'abort_on_error=0:exitcode=66:halt_on_error=0:report_thread_leaks=0' };
    const atomic = spawnSync(exe, ['race-atomic'], { encoding: 'utf8', env, timeout: 120000 });
    expect(atomic.stdout).toContain('race atomic: launched');
    expect(atomic.stderr).not.toContain('WARNING: ThreadSanitizer: data race');
    const plain = spawnSync(exe, ['race-plain'], { encoding: 'utf8', env, timeout: 120000 });
    expect(plain.stdout).toContain('race plain: launched');
    expect(plain.stderr).toContain('WARNING: ThreadSanitizer: data race');
    const line = read(DRIVER).split('\n').findIndex((l) => l.includes('y[row] += x[cooMatrix.colIdx[i]]*cooMatrix.value[i];')) + 1;
    expect(line).toBeGreaterThan(0);
    expect(plain.stderr).toContain(`spmv_coo_plain_kernel(COOMatrix, float*, float*) spmv_check.cu:${line}`);
  }, 180000);
});
