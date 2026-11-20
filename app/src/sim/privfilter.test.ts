import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  BOOK_BLOCK_ORDER,
  BOOK_KEEP,
  CORRECT,
  GARBAGE,
  N,
  SCHEDULES,
  UNWRITTEN,
  VARIANT,
  VARIANTS,
  blockOrderOf,
  keyName,
  kernelBody,
  outcomeTable,
  publicAtomics,
  races,
  simulate,
  siteLine,
  statementLines,
  summary,
  type Schedule,
  type Shape,
  type Site,
  type Variant,
} from './privfilter';
import { functionSource } from './toolchain';

const repo = path.resolve(__dirname, '../../..');
const SNIPPET = path.join(repo, 'labs/ch12/snippets/s124_privatised_variants.cu');
const DRIVER = path.join(repo, 'labs/ch12/snippets/check/s124_privatised_check.cu');
const source = fs.readFileSync(SNIPPET, 'utf8');
const kernelOf = (v: Variant) => functionSource(source, VARIANT[v].kernel)!;
/** A kernel's body as the model writes it: the lines between the braces, one indent level removed. */
const bodyOf = (v: Variant) => {
  const lines = kernelOf(v).text.split('\n');
  const open = lines.findIndex((l) => l.includes('{'));
  return lines.slice(open + 1, lines.length - 1).map((l) => l.replace(/^ {4}/, '')).join('\n');
};
const ALL: boolean[] = new Array<boolean>(N).fill(true);
const NONE: boolean[] = new Array<boolean>(N).fill(false);
const SOME: boolean[] = [1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0].map((x) => x === 1);
const U = UNWRITTEN;

describe('the kernels are the snippet file\'s', () => {
  it('each variant\'s body is exactly the model\'s text', () => {
    for (const v of VARIANTS) expect(bodyOf(v), v).toBe(kernelBody(VARIANT[v].shape));
  });
  it('a removed barrier leaves a placeholder in the stepper\'s listing only', () => {
    const shown = kernelBody({ ...CORRECT, barrier2: false }, true).split('\n');
    expect(shown).toContain('// barrier 2 removed');
    expect(shown).toHaveLength(kernelBody(CORRECT).split('\n').length);
    expect(kernelBody({ ...CORRECT, atomicCounter: false })).toContain('unsigned int slot = outputSize_s++;');
  });
  it('the three barriers sit outside every conditional (p. 297)', () => {
    for (const l of kernelBody(CORRECT).split('\n')) if (l.includes('__syncthreads')) expect(l.startsWith('__syncthreads')).toBe(true);
  });
  it('no line of the stepper\'s listing is wider than its column holds (64 characters fit at 1440 px)', () => {
    for (const v of VARIANTS) for (const l of kernelBody(VARIANT[v].shape, true).split('\n')) expect(l.length, l).toBeLessThanOrEqual(64);
  });
  it('a statement over two lines is marked whole, one on a single line alone', () => {
    const lines = kernelBody(CORRECT, true).split('\n');
    const count = simulate(CORRECT, 'forward').steps.find((s) => s.op === 'count')!;
    expect(statementLines(CORRECT, true, count.line).map((n) => lines[n - 1].trim())).toEqual([
      'unsigned int slot = outputSize_s_ref.fetch_add(1,',
      'cuda::memory_order_relaxed);',
    ]);
    const zero = simulate(CORRECT, 'forward').steps.find((s) => s.op === 'zero')!;
    expect(statementLines(CORRECT, true, zero.line)).toEqual([zero.line]);
  });
});

describe('the correct kernel (Figs. 12.5, 12.6)', () => {
  it('every order gives the eight kept keys, each block\'s list contiguous', () => {
    for (const s of SCHEDULES) {
      const t = simulate(CORRECT, s);
      expect(t.outcome.kind, s).toBe('correct');
      expect(t.firstFailure).toBe(-1);
      expect(t.outcome.size).toBe(8);
    }
  });
  it('in order, in the emulator\'s block order: k1 k3 | k4 k6 | k8 k10 k11 | k14', () => {
    const t = simulate(CORRECT, 'forward');
    expect(t.outcome.output.slice(0, 8).map(keyName)).toEqual(['k1', 'k3', 'k4', 'k6', 'k8', 'k10', 'k11', 'k14']);
  });
  it('in Fig. 12.5\'s block order (green, orange, yellow, blue = blocks 1, 3, 2, 0) the chunks follow that order', () => {
    expect(BOOK_BLOCK_ORDER).toEqual([1, 3, 2, 0]);
    const t = simulate(CORRECT, 'forward', BOOK_KEEP, BOOK_BLOCK_ORDER);
    expect(t.outcome.output.slice(0, 8).map(keyName)).toEqual(['k4', 'k6', 'k14', 'k8', 'k10', 'k11', 'k1', 'k3']);
    const r = simulate(CORRECT, 'reverse', BOOK_KEEP, BOOK_BLOCK_ORDER);
    // Reversed threads fill each private list backwards; the book's figure shows k6 k4 | k14 | k10 k11 k8 | k3 k1.
    expect(r.outcome.output.slice(0, 8).map(keyName)).toEqual(['k6', 'k4', 'k14', 'k11', 'k10', 'k8', 'k3', 'k1']);
    expect(r.outcome.kind).toBe('correct');
  });
  it('one public atomic per block, issued even when the block keeps nothing', () => {
    const t = simulate(CORRECT, 'forward', NONE);
    expect(t.accesses.filter((a) => a.site === 'public-add')).toHaveLength(4);
    expect(t.outcome).toMatchObject({ kind: 'correct', size: 0 });
  });
  it('blocks run in a chosen order; random orders are seeded permutations', () => {
    expect(blockOrderOf('in-order')).toEqual([0, 1, 2, 3]);
    for (const seed of [1, 2, 3, 4, 5]) expect([...blockOrderOf('random', seed)].sort()).toEqual([0, 1, 2, 3]);
    expect(blockOrderOf('random', 1)).toEqual(blockOrderOf('random', 1));
  });
  it('shared memory starts as 0xCD bytes, the public output as 0xFF bytes', () => {
    const t = simulate(CORRECT, 'forward');
    expect(t.steps[0].after.blocks[1].outputSize_s).toBe(GARBAGE);
    expect(t.steps[0].after.output.every((v) => v === U)).toBe(true);
  });
});

describe('q-public: atomics on the public counter', () => {
  it('2^20 keys, blocks of 256, 1 percent kept: 4096 per block, 10,486 per key, about 9,000 per warp', () => {
    const a = publicAtomics(2 ** 20, 256, 0.01);
    expect(a.perBlock).toBe(4096);
    expect(a.perKey).toBe(10_486);
    expect(a.perWarp).toBeGreaterThan(8_900);
    expect(a.perWarp).toBeLessThan(9_100);
    // Per block does not depend on the fraction.
    expect(publicAtomics(2 ** 20, 256, 0.5).perBlock).toBe(4096);
  });
});

// The outcome table, by variant and emulator order. Every cell was also produced by the
// emulator itself (last describe block).
describe('the outcome table', () => {
  const table = outcomeTable();
  const cell = (v: Variant, s: Schedule) => table.find((r) => r.variant === v)!.cells[s];
  it('which order exposes which barrier', () => {
    const kinds = Object.fromEntries(table.map((r) => [r.variant, SCHEDULES.map((s) => r.cells[s].kind)]));
    expect(kinds).toEqual({
      correct: ['correct', 'correct', 'correct'],
      'no-barrier-1': ['correct', 'crash', 'crash'],
      'no-barrier-2': ['wrong', 'correct', 'wrong'],
      'no-barrier-3': ['correct', 'crash', 'crash'],
      'plain-counter': ['correct', 'correct', 'correct'],
    });
  });
  it('barrier 2 removed, in order: thread 0 reserves 0, 1, 1 and 0 slots; the output keeps 4 keys and reports 2', () => {
    const t = simulate(VARIANT['no-barrier-2'].shape, 'forward');
    expect(t.steps.filter((s) => s.op === 'reserve').map((s) => /reserves outputSize_s = (\d+)/.exec(s.text)![1])).toEqual(['0', '1', '1', '0']);
    expect(t.outcome.size).toBe(2);
    expect(t.outcome.output.slice(0, 5).map(keyName)).toEqual(['k4', 'k8', 'k14', 'k11', '']);
    expect(t.outcome.lost).toEqual([1, 3, 6, 10]);
    expect(summary(t.outcome)).toBe('size 2 (not 8), 4 keys lost');
    expect(t.steps[t.firstFailure].violation!.kind).toBe('short-reservation');
    expect(t.steps.some((s) => s.violation?.kind === 'overwrite')).toBe(true);
  });
  it('barrier 2 removed, shuffled: size 6 and one key lost', () => {
    expect(summary(cell('no-barrier-2', 'shuffle'))).toBe('size 6 (not 8), 1 key lost');
  });
  it('barrier 3 removed, reversed: thread 3 reads blockStart before thread 0 writes it and stores far out of range', () => {
    const t = simulate(VARIANT['no-barrier-3'].shape, 'reverse');
    const f = t.steps[t.firstFailure];
    expect(f.violation!.kind).toBe('crash');
    expect(f.violation!.message).toContain('blockStart has not been written yet');
    expect(f.block).toBe(0);
    expect(t.outcome.crash).toContain('output[');
  });
  it('barrier 1 removed, reversed: thread 3 counts from 0xCDCDCDCD', () => {
    const t = simulate(VARIANT['no-barrier-1'].shape, 'reverse');
    expect(t.steps[t.firstFailure].violation!.kind).toBe('garbage-count');
    expect(t.outcome.crash).toContain('output_s[3452816845]');
  });
  it('a failure in one order and a pass in another (q-barrier3: reversed exposes it, in order hides it)', () => {
    expect(cell('no-barrier-3', 'forward').kind).toBe('correct');
    expect(cell('no-barrier-3', 'reverse').kind).toBe('crash');
    expect(cell('no-barrier-2', 'forward').kind).toBe('wrong');
    expect(cell('no-barrier-2', 'reverse').kind).toBe('correct');
  });
  it('the race check: only the plain counter passes every emulator order and still races', () => {
    const r = Object.fromEntries(table.map((x) => [x.variant, x.races.map((y) => y.sites.join('|'))]));
    expect(r.correct).toEqual([]);
    expect(r['plain-counter']).toEqual(['count|count']);
    expect(r['no-barrier-1']).toContain('zero|count');
    expect(r['no-barrier-2']).toContain('count|reserve-read');
    expect(r['no-barrier-2']).toContain('commit-write|commit-write');
    expect(r['no-barrier-3']).toContain('reserve-write|commit-read-start');
    const passesEmu = table.filter((x) => SCHEDULES.every((s) => x.cells[s].kind === 'correct')).map((x) => x.variant);
    expect(passesEmu).toEqual(['correct', 'plain-counter']);
  });
  it('the plain counter gets the right slots one thread at a time, and the stepper says why it is still wrong', () => {
    const t = simulate(VARIANT['plain-counter'].shape, 'shuffle');
    expect(t.outcome.kind).toBe('correct');
    expect(t.steps.filter((s) => s.op === 'count').every((s) => s.note?.includes('race check'))).toBe(true);
  });
  it('more keep patterns', () => {
    for (const keep of [ALL, NONE, SOME])
      for (const s of SCHEDULES) expect(simulate(CORRECT, s, keep).outcome.kind).toBe('correct');
    expect(simulate(VARIANT['no-barrier-2'].shape, 'forward', ALL).outcome.duplicated.length + simulate(VARIANT['no-barrier-2'].shape, 'forward', ALL).outcome.lost.length).toBeGreaterThan(0);
  });
  it('every racing site can be found in its kernel', () => {
    for (const v of VARIANTS)
      for (const r of races(VARIANT[v].shape)) for (const s of r.sites) expect(siteLine(kernelOf(v).text, s), `${v} ${s}`).toBeGreaterThan(0);
  });
  it('the stepper\'s lines point at the statement each step runs', () => {
    const shapes: Shape[] = [CORRECT, { ...CORRECT, barrier1: false }, { ...CORRECT, atomicCounter: false, barrier3: false }];
    for (const sh of shapes) {
      const lines = kernelBody(sh, true).split('\n');
      for (const st of simulate(sh, 'forward').steps) {
        const l = lines[st.line - 1];
        const want: Record<string, RegExp> = {
          zero: /outputSize_s = 0/, arrive: /__syncthreads/, release: /__syncthreads/, load: /input\[i\]|cond\(val\)/,
          count: /fetch_add\(1|outputSize_s\+\+/, store: /output_s\[slot\]/, reserve: /blockStart = /, commit: /output\[blockStart/, skip: /threadIdx\.x < outputSize_s/,
        };
        expect(l, st.op).toMatch(want[st.op]);
      }
    }
  });
});

// The snippet file on the course's emulator, built as the lab runner builds the emu and race
// backends (labs/cmake/Backends.cmake): emu with AddressSanitizer and the three thread
// orders, race with ThreadSanitizer. One process per run, because the defects that write out
// of range stop the process. Skipped when no C++ compiler works.
const compiler = ['c++', 'clang++', 'g++'].find((c) => spawnSync(c, ['--version']).status === 0);
describe.skipIf(!compiler)('the snippets on the emulator agree with the model', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch12-privfilter-'));
  const inc = path.join(repo, 'labs/common/include');
  const exe = { emu: path.join(dir, 'emu'), race: path.join(dir, 'race') };
  const env = {
    ...process.env,
    ASAN_OPTIONS: 'detect_leaks=0',
    TSAN_OPTIONS: 'abort_on_error=0:exitcode=66:halt_on_error=0:report_thread_leaks=0',
  };
  let raceBuilt = false;
  it('builds', () => {
    const cu = '#line 1 "s124_privatised_variants.cu"\n' + source + '\n#line 1 "check/s124_privatised_check.cu"\n' + fs.readFileSync(DRIVER, 'utf8');
    fs.writeFileSync(path.join(dir, 'all.cu'), cu);
    const t = spawnSync('python3', [path.join(repo, 'labs/tools/cu2emu.py'), path.join(dir, 'all.cu'), path.join(dir, 'all.cpp')], { encoding: 'utf8' });
    expect(t.status, t.stdout + t.stderr).toBe(0);
    const common = ['-std=c++17', '-O1', '-g', '-fno-omit-frame-pointer', '-pthread', '-I', inc, '-I', path.join(inc, 'pmpp/emu/shim'), '-include', 'pmpp/emu/cuda_emu.h', path.join(dir, 'all.cpp')];
    const c = spawnSync(compiler!, ['-fsanitize=address', '-DPMPP_BACKEND_EMU', ...common, '-o', exe.emu], { encoding: 'utf8' });
    expect(c.status, c.stderr).toBe(0);
    const r = spawnSync(compiler!, ['-fsanitize=thread', '-DPMPP_BACKEND_RACE', ...common, '-o', exe.race], { encoding: 'utf8' });
    raceBuilt = r.status === 0; // ThreadSanitizer is not available everywhere
  }, 240_000);
  const flags = (keep: readonly boolean[]) => keep.map((k) => (k ? '1' : '0'));
  for (const [name, keep] of [['the book\'s keys', BOOK_KEEP], ['every key', ALL], ['another pattern', SOME]] as const)
    for (const v of VARIANTS)
      it(`${VARIANT[v].kernel}, ${name}: emu's three orders give the model's output, or stop where the model stops`, () => {
        for (const s of SCHEDULES) {
          const r = spawnSync(exe.emu, [VARIANT[v].kernel, s, ...flags(keep)], { encoding: 'utf8', env });
          const m = simulate(VARIANT[v].shape, s, keep).outcome;
          if (m.kind === 'crash') {
            expect(r.status, `${s}: expected AddressSanitizer to stop the run\n${r.stdout}`).not.toBe(0);
            expect(r.stderr, s).toMatch(/AddressSanitizer/);
          } else {
            expect(r.status, `${s}\n${r.stderr.slice(0, 1500)}`).toBe(0);
            const got = r.stdout.trim().split(' ');
            expect(got.slice(0, 2)).toEqual([VARIANT[v].kernel, s]);
            expect(got.slice(2).map(Number), s).toEqual([m.size, ...m.output]);
          }
        }
      }, 60_000);
  // ThreadSanitizer's reports, reduced to unordered pairs of source lines in the snippet file.
  const reported = (stderr: string) => {
    const pairs = new Set<string>();
    for (const rep of stderr.split('WARNING: ThreadSanitizer: data race').slice(1)) {
      const lines: number[] = [];
      for (const part of rep.split(/\n\s+(?=(?:Previous )?(?:[Aa]tomic )?(?:[Ww]rite|[Rr]ead) of size)/).slice(1)) {
        const m = /s124_privatised_variants\.cu:(\d+)/.exec(part);
        if (m) lines.push(Number(m[1]));
      }
      if (lines.length >= 2) pairs.add(lines.slice(0, 2).sort((a, b) => a - b).join('-'));
    }
    return pairs;
  };
  const possible = (v: Variant, keep: readonly boolean[]) => {
    const k = kernelOf(v);
    const at = (s: Site) => k.firstLine + siteLine(k.text, s);
    const sites = races(VARIANT[v].shape, keep).map((r) => r.sites);
    // With a real lost update two threads also share a slot of output_s.
    if (v === 'plain-counter') sites.push(['store', 'store']);
    return new Set(sites.map((p) => p.map(at).sort((a, b) => a - b).join('-')));
  };
  // Only the variants that cannot write out of range: the race flavour has no bounds checks.
  for (const v of ['correct', 'plain-counter', 'no-barrier-2'] as Variant[])
    it(`${VARIANT[v].kernel}: the race flavour reports nothing the model does not`, (ctx) => {
      if (!raceBuilt) return ctx.skip();
      for (const keep of [BOOK_KEEP, ALL]) {
        const want = possible(v, keep);
        for (let rep = 0; rep < 3; rep++) {
          const r = spawnSync(exe.race, [VARIANT[v].kernel, 'once', ...flags(keep)], { encoding: 'utf8', env });
          expect([0, 66]).toContain(r.status);
          for (const p of reported(r.stderr)) expect(want.has(p), `unexpected race ${p}\n${r.stderr.slice(0, 2000)}`).toBe(true);
          if (v === 'correct') expect(r.status).toBe(0);
        }
      }
      // Not required: that the race flavour catches the plain counter. It gives each block one
      // happens-before token for all its barriers, so a thread that leaves barrier 1 late
      // synchronises with threads already at barrier 2, which hides the race between two
      // increments in most runs (the same limitation histpriv.test.ts records for chapter 9).
    }, 120_000);
});

