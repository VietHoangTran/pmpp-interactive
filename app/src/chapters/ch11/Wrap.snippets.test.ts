import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { chapter } from './index';
import {
  LOGITS,
  LOGITS_BLOCK,
  MASS,
  MASS_FILE,
  TAIL,
  TAIL_P,
  TAIL_RESULT,
  koggeStoneBlock,
  questionsWrap,
  warpStarts,
} from './Wrap';
import { exclusiveIn, f32, inclusive, inclusiveIn } from '../../sim/scan';
import { functionSource } from '../../sim/toolchain';

// The summary page shows one kernel of its own, loaded at run time from
// labs/ch11/snippets/wrap_mass_before.cu. This test ties that file's helpers to lab 11.1,
// runs it on the course's CPU emulator (labs/tools/cu2emu.py, pmpp/emu/cuda_emu.h,
// AddressSanitizer on) through labs/ch11/snippets/check/wrap_mass_before_test.cu, and pins
// the numbers the other questions state. It also checks the chapter's wiring: every section
// the index lists, and the prediction ids the lab READMEs name. The emulator part is skipped
// when no C++ compiler or python3 is found.
const repo = path.resolve(__dirname, '../../../..');
const read = (p: string) => fs.readFileSync(path.join(repo, p), 'utf8');
const SNIPPET = 'labs/' + MASS_FILE;
const LAB = 'labs/ch11/lab1_blockscan/solution/kernel.cu';
const snippet = read(SNIPPET);
/** Source without comments, for claims about the code itself. */
const code = (s: string) => s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');

describe('the kernel of the first question', () => {
  it("shares its defines, mask type, laneIdx, warpIdx, warpScan and blockScan with lab 11.1, character for character", () => {
    const lab = read(LAB);
    const typedef = (s: string) => /#if defined\(__HIPCC__\)[\s\S]*?#endif/.exec(s)?.[0];
    expect(typedef(snippet)).toBeDefined();
    expect(typedef(snippet)).toBe(typedef(lab));
    for (const d of ['#define MAX_BLOCK_DIM 1024', '#define MIN_WARP_SIZE 32']) {
      const line = (s: string) => s.split('\n').find((l) => l.startsWith(d));
      expect(line(snippet)).toBeDefined();
      expect(line(snippet)).toBe(line(lab));
    }
    // A function's text without the comment lines above it (the lab file has a section rule there).
    const body = (s: string, name: string) =>
      functionSource(s, name)!.text.split('\n').filter((l, i, a) => !(l.trim().startsWith('//') && a.slice(0, i).every((x) => x.trim().startsWith('//')))).join('\n');
    for (const f of ['laneIdx', 'warpIdx', 'warpScan', 'blockScan']) {
      expect(functionSource(snippet, f), f).not.toBeNull();
      expect(body(snippet, f)).toBe(body(lab, f));
    }
  });
  it('has the kernel the page names, with the defect the question is about and no literal mask or width', () => {
    const k = functionSource(snippet, 'massBeforeRows');
    expect(k).not.toBeNull();
    expect(k!.text.startsWith('__global__ void massBeforeRows(const float* probs, float* massBefore, unsigned int cols) {')).toBe(true);
    expect(k!.text).toContain('__shfl_up_sync(full, inclusive, 1)');
    expect(k!.text).toContain('if (threadIdx.x == 0) {');
    expect(k!.text).not.toMatch(/laneIdx\(\) == 0/);
    expect(code(snippet)).not.toMatch(/0xffffffff/i);
    expect(code(k!.text)).not.toMatch(/\b32\b/);
  });
  it('the right option names the first lane of every warp but the first', () => {
    expect(warpStarts(MASS.block, MASS.cols, MASS.width)).toEqual([32, 64, 96, 128, 160, 192, 224]);
    expect(warpStarts(256, 200, 32)).toEqual([32, 64, 96, 128, 160, 192]);
    expect(warpStarts(64, 64, 32)).toEqual([32]);
    expect(warpStarts(256, 256, 64)).toEqual([64, 128, 192]);
    const q = questionsWrap['ch11/wrap/q-mass-before'];
    expect(q.kind === 'choice' && q.options.findIndex((o) => o.correct)).toBe(1);
  });
});

describe("the other questions' numbers", () => {
  it('a running max padded with 0 stores the right values, because every step reads to the left', () => {
    const cummax = LOGITS.map((_, i) => Math.max(...LOGITS.slice(0, i + 1)));
    const shown = koggeStoneBlock(LOGITS, LOGITS_BLOCK, Math.max, 0);
    expect(shown.slice(0, LOGITS.length)).toEqual(cummax);
    expect(shown.slice(LOGITS.length)).toEqual([0, 0]);
    expect(koggeStoneBlock(LOGITS, LOGITS_BLOCK, Math.max, -3.4e38).slice(0, LOGITS.length)).toEqual(cummax);
    // The same network with + is Fig. 11.3: it must agree with the plain scan.
    expect(koggeStoneBlock([3, 1, 7, 0, 4, 1, 6, 3], 8, (a, b) => a + b, 0)).toEqual(inclusive([3, 1, 7, 0, 4, 1, 6, 3]));
    const q = questionsWrap['ch11/wrap/q-maxpad'];
    expect(q.kind === 'choice' && q.options.findIndex((o) => o.correct)).toBe(1);
  });
  it('the float32 loop loses the whole tail; Kogge-Stone keeps it; the threshold splits them', () => {
    const row = [TAIL.head, ...new Array(TAIL.count).fill(TAIL.each)];
    const loop = inclusiveIn(row, 'loop', f32);
    const ks = inclusiveIn(row, 'kogge-stone', f32);
    expect(loop.at(-1)).toBe(f32(TAIL.head));
    expect(Number(loop.at(-1)!.toPrecision(4)).toString()).toBe(TAIL_RESULT.loop);
    expect(ks.at(-1)).toBe(1);
    expect(String(ks.at(-1))).toBe(TAIL_RESULT.koggeStone);
    // Top-p at TAIL_P keeps every token under the loop's mass, about half under Kogge-Stone's.
    const keptLoop = exclusiveIn(row, 'loop', f32).filter((m) => m < TAIL_P).length;
    const keptKs = exclusiveIn(row, 'kogge-stone', f32).filter((m) => m < TAIL_P).length;
    expect(keptLoop).toBe(row.length);
    expect(keptKs / row.length).toBeGreaterThan(0.45);
    expect(keptKs / row.length).toBeLessThan(0.55);
  }, 30000);
  it('a sequence\'s own scan is G[i] - G[s - 1] for +, on whole numbers', () => {
    const x = [5, 3, 0, 7, 2, 4, 9, 1, 6, 8];
    const starts = [0, 3, 4, 8];
    const G = inclusive(x);
    starts.forEach((s, k) => {
      const e = (starts[k + 1] ?? x.length) - 1;
      const own = inclusive(x.slice(s, e + 1));
      for (let i = s; i <= e; i++) expect(G[i] - (s > 0 ? G[s - 1] : 0)).toBe(own[i - s]);
    });
  });
  it("every question has exactly one correct option, under this page's ids", () => {
    for (const q of Object.values(questionsWrap)) {
      if (q.kind === 'choice') expect(q.options.filter((o) => o.correct).length).toBe(1);
      else expect(Number.isFinite(q.answer)).toBe(true);
      expect(q.id.startsWith('ch11/wrap/')).toBe(true);
    }
  });
});

describe("the chapter's wiring", () => {
  it("lists every section once, in the book's order, each with a file that exists", () => {
    expect(chapter.sections.map((s) => s.id)).toEqual(['start', '11.1', '11.2', '11.3', '11.4', '11.5', '11.6', '11.7', '11.8', '11.9', '11.10', 'labs', 'wrap']);
    const index = fs.readFileSync(path.join(__dirname, 'index.tsx'), 'utf8');
    expect(index).not.toMatch(/temporary/i);
    for (const m of index.matchAll(/import\('\.\/(\w+)'\)/g)) expect(fs.existsSync(path.join(__dirname, m[1] + '.tsx'))).toBe(true);
  });
  it('every prediction id a lab README names is asked on the labs page or in the section it names', () => {
    const labs = fs.readFileSync(path.join(__dirname, 'Labs.tsx'), 'utf8');
    const file = (sec: string) => path.join(__dirname, `S${sec.replace('.', '')}.tsx`);
    const dirs = ['lab1_blockscan', 'lab2_coarsened', 'lab3_scanhunt', 'lab4_global', 'lab5_singlepass', 'lab6_topp'];
    let n = 0;
    for (const dir of dirs) {
      const ids = [...read(`labs/ch11/${dir}/README.md`).matchAll(/`(ch11\/[\w.]+\/p-[\w-]+)`/g)].map((m) => m[1]);
      expect(ids.length, dir).toBeGreaterThan(0);
      for (const id of ids) {
        n++;
        const sec = id.split('/')[1];
        if (sec === 'labs') expect(labs, id).toContain(`id="${id}"`);
        else {
          expect(fs.readFileSync(file(sec), 'utf8'), id).toContain(`id="${id}"`);
          expect(labs, `labs page points at ${sec}`).toContain(`href('ch11', '${sec}')`);
        }
      }
    }
    expect(n).toBe(9);
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
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch11-wrap-'));
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

describe.skipIf(!cxx || !python)('the kernel on the course emulator (labs/ch11/snippets/check/wrap_mass_before_test.cu)', () => {
  it('only the first lane of every warp but the first is wrong, holding its inclusive mass, under three orders; the lane test fixes it', () => {
    const exe = build([SNIPPET, 'labs/ch11/snippets/check/wrap_mass_before_test.cu'], 'massbefore');
    const r = spawnSync(exe, [], { encoding: 'utf8', env: { ...process.env, ASAN_OPTIONS: 'detect_leaks=0' }, timeout: 120000 });
    expect(r.status).toBe(0);
    expect(r.stdout).toContain('DONE');
    const lines = r.stdout.split('\n').filter((l) => /^(shown|fixed) /.test(l));
    expect(lines.length).toBe(18);
    const rows = 3;
    for (const l of lines) {
      const m = /^(shown|fixed) (\d+) (\d+) ([a-z ]+): wrong (\d+) of (\d+):(.*) \| inclusive (\d+)$/.exec(l);
      expect(m, l).not.toBeNull();
      const [, kernel, block, cols, , wrong, total, list, incl] = m!;
      expect(Number(total)).toBe(rows * Number(cols));
      if (kernel === 'fixed') {
        expect(Number(wrong)).toBe(0);
        continue;
      }
      const want = Array.from({ length: rows }, (_, row) => warpStarts(Number(block), Number(cols), 32).map((c) => `${row}:${c}`)).flat();
      expect(list.trim().split(' ')).toEqual(want);
      expect(Number(wrong)).toBe(want.length);
      expect(Number(incl)).toBe(want.length);
    }
  }, 120000);
});
