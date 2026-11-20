import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { E1, NEG_QNAN, POS_QNAN, items, keyAsc, keyDesc, referenceDesc, stableBy } from '../../sim/floatkey';

// Section 14.4 shows the float-to-key transform from labs/ch14/snippets/s144_floatkey.cu.
// This test fails if the lesson's copy drifts from the file, and it builds the snippet for
// the course's CPU emulator (labs/tools/cu2emu.py, pmpp/emu/cuda_emu.h, AddressSanitizer
// on), runs labs/ch14/snippets/check/s144_check.cu under the emulator's three thread orders,
// and compares every key it prints with src/sim/floatkey.ts, bit for bit. The emulator part
// is skipped when no C++ compiler or python3 is found.
const repo = path.resolve(__dirname, '../../../..');
const SNIPPET = 'labs/ch14/snippets/s144_floatkey.cu';
const DRIVER = 'labs/ch14/snippets/check/s144_check.cu';
const read = (p: string) => fs.readFileSync(path.join(repo, p), 'utf8');
const lessonSrc = fs.readFileSync(path.join(__dirname, 'S144.tsx'), 'utf8');
const lesson = (constant: string) => {
  const m = new RegExp('const ' + constant + ' = `([\\s\\S]*?)`;').exec(lessonSrc);
  if (!m) throw new Error(`no ${constant} in S144.tsx`);
  return m[1];
};
const region = (src: string, name: string) => {
  const m = new RegExp('// region: ' + name + '\\n([\\s\\S]*?)\\n// endregion').exec(src);
  if (!m) throw new Error(`no region ${name}`);
  return m[1];
};

describe('section 14.4 shows the checked file', () => {
  it('the lesson\'s FLOATKEY is the region "floatkey" of the snippet, verbatim', () => {
    expect(lesson('FLOATKEY')).toBe(region(read(SNIPPET), 'floatkey'));
  });
  it('the snippet works on an unsigned view and never holds a float', () => {
    const code = region(read(SNIPPET), 'floatkey').replace(/\/\/.*$/gm, '');
    expect(code).not.toMatch(/\bfloat\b/);
    expect(code).not.toContain('__float_as_uint');
    expect(code).toContain('const unsigned int* logitBits');
  });
  it('the -0.0 step comes before the transform, and descending is the inverted key', () => {
    const code = region(read(SNIPPET), 'floatkey');
    expect(code.indexOf('if(u == 0x80000000u) u = 0u;')).toBeLessThan(code.indexOf('return u ^ mask;'));
    expect(code).toContain('return ~float_to_key(u);');
  });
});

// ---------------------------------------------------------------- on the emulator
const has = (cmd: string, args: string[]) => spawnSync(cmd, args).status === 0;
const cxx = ['c++', 'clang++', 'g++'].find((c) => has(c, ['--version']));
const python = ['python3'].find((c) => has(c, ['--version']));
const INC = path.join(repo, 'labs/common/include');
const FLAGS = ['-std=c++17', '-O1', '-g', '-fsanitize=address', '-fno-omit-frame-pointer', '-pthread', '-DPMPP_BACKEND_EMU',
  '-I', INC, '-I', path.join(INC, 'pmpp/emu/shim'), '-include', 'pmpp/emu/cuda_emu.h'];

/** Translate the snippet and its driver into a temporary tree with the same layout, then build and run. */
function buildAndRun(snippet: string, driver: string, name: string) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch14-' + name + '-'));
  fs.mkdirSync(path.join(dir, 'check'));
  const tr = (from: string, to: string) => {
    const t = spawnSync(python!, [path.join(repo, 'labs/tools/cu2emu.py'), path.join(repo, from), to], { encoding: 'utf8' });
    if (t.status !== 0) throw new Error('cu2emu failed:\n' + t.stdout + t.stderr);
  };
  tr(snippet, path.join(dir, path.basename(snippet)));
  const cpp = path.join(dir, 'check', name + '.cpp');
  tr(driver, cpp);
  const obj = path.join(dir, name + '.o');
  const c = spawnSync(cxx!, [...FLAGS, '-Wall', '-Wextra', '-c', cpp, '-o', obj], { encoding: 'utf8' });
  if (c.status !== 0) throw new Error('compile failed:\n' + c.stderr);
  const exe = path.join(dir, name);
  const l = spawnSync(cxx!, ['-fsanitize=address', '-pthread', obj, '-o', exe], { encoding: 'utf8' });
  if (l.status !== 0) throw new Error('link failed:\n' + l.stderr);
  const r = spawnSync(exe, [], { encoding: 'utf8', env: { ...process.env, ASAN_OPTIONS: 'detect_leaks=0' }, timeout: 120000 });
  fs.rmSync(dir, { recursive: true, force: true });
  return { status: r.status, out: r.stdout, err: r.stderr };
}

describe.skipIf(!cxx || !python)('s144_floatkey.cu on the course emulator (labs/ch14/snippets/check/)', () => {
  let out = '';
  it('builds and runs under AddressSanitizer in all three thread orders', () => {
    const r = buildAndRun(SNIPPET, DRIVER, 's144');
    expect(r.status, r.err).toBe(0);
    out = r.out;
    expect(out).toContain('DONE');
    expect(out.split('\n').filter((l) => l.startsWith('order '))).toEqual(['order in order', 'order reversed', 'order shuffled']);
  }, 180000);
  it('every key the kernel computes equals floatkey.ts, ascending and descending', () => {
    const keys = out.split('\n').filter((l) => l.startsWith('key '));
    expect(keys.length).toBe(3 * 612);
    for (const l of keys) {
      const [, inp, asc, desc] = l.split(' ');
      const u = parseInt(inp, 16) >>> 0;
      expect(parseInt(asc, 16) >>> 0, l).toBe(keyAsc(u));
      expect(parseInt(desc, 16) >>> 0, l).toBe(keyDesc(u));
    }
    // E1's table, as the kernel computed it
    const first = keys.slice(0, 12).map((l) => l.split(' ').slice(1).map((h) => parseInt(h, 16) >>> 0));
    E1.forEach((e, k) => expect(first[k]).toEqual([e.bits >>> 0, e.canon >>> 0, e.desc >>> 0]));
    expect(first[10]).toEqual([POS_QNAN, 0xffc00000, 0x003fffff]);
    expect(first[11]).toEqual([NEG_QNAN, 0x003fffff, 0xffc00000]);
  });
  it('sorted descending by its keys, the edge values come out in the model\'s reference order', () => {
    const want = referenceDesc(items([...E1.map((e) => e.bits), POS_QNAN, NEG_QNAN])).order.map((x) => x.token);
    expect(want).toEqual(stableBy(items([...E1.map((e) => e.bits), POS_QNAN, NEG_QNAN]), (x) => keyDesc(x.bits)).map((x) => x.token));
    const sorts = out.split('\n').filter((l) => l.startsWith('sort desc:'));
    expect(sorts.length).toBe(3);
    for (const l of sorts) expect(l.slice('sort desc:'.length).trim().split(' ').map(Number)).toEqual(want);
  });
});
