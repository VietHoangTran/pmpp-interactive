import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { BOOK_KEYS, lsdSort, parallelPass } from '../../sim/radixsort';
import { STEPS } from './S145';

// Section 14.5 steps through labs/ch14/snippets/s145_radix_iter.cu. This test fails if the
// lesson's copies (PASS, SCAN) drift from the file, if a line the stepper highlights stops
// being the line it names, or if the lesson's line mapping to the page goes stale. Then it
// builds the snippet for the course's CPU emulator (labs/tools/cu2emu.py,
// pmpp/emu/cuda_emu.h, AddressSanitizer on), runs labs/ch14/snippets/check/s145_check.cu
// under the emulator's three thread orders, and compares what it prints with
// src/sim/radixsort.ts. The emulator part is skipped when no C++ compiler or python3 is found.
const repo = path.resolve(__dirname, '../../../..');
const SNIPPET = 'labs/ch14/snippets/s145_radix_iter.cu';
const DRIVER = 'labs/ch14/snippets/check/s145_check.cu';
const read = (p: string) => fs.readFileSync(path.join(repo, p), 'utf8');
const lessonSrc = fs.readFileSync(path.join(__dirname, 'S145.tsx'), 'utf8');
const lessonFlat = lessonSrc.replace(/\s+/g, ' ');
const lesson = (constant: string) => {
  const m = new RegExp('const ' + constant + ' = `([\\s\\S]*?)`;').exec(lessonSrc);
  if (!m) throw new Error(`no ${constant} in S145.tsx`);
  return m[1];
};
const region = (src: string, name: string) => {
  const m = new RegExp('// region: ' + name + '\\n([\\s\\S]*?)\\n// endregion').exec(src);
  if (!m) throw new Error(`no region ${name}`);
  return m[1];
};
const PASS = lesson('PASS');
const line = (n: number) => PASS.split('\n')[n - 1].trim();

describe('section 14.5 shows the checked file', () => {
  it('PASS is the region "pass" and SCAN the region "scan", verbatim', () => {
    expect(PASS).toBe(region(read(SNIPPET), 'pass'));
    expect(lesson('SCAN')).toBe(region(read(SNIPPET), 'scan'));
  });
  it('the pass is 36 lines, none longer than 74 characters: two kernels and a host function that launches three times', () => {
    expect(PASS.split('\n')).toHaveLength(36);
    expect(Math.max(...PASS.split('\n').map((l) => l.length))).toBeLessThanOrEqual(74);
    expect(PASS.match(/<<</g)).toHaveLength(3);
  });
});

describe('the stepper highlights the lines it names', () => {
  const want: Record<string, string[]> = {
    i: ['radix_bits_kernel<<<numBlocks, blockSize>>>(input, bits, N, iter);', 'unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;'],
    load: ['unsigned int key = input[i];'],
    bit: ['unsigned int bit = (key >> iter) & 1;'],
    'store-bit': ['bits[i] = bit;'],
    scan: ['exclusive_scan_kernel<<<1, SCAN_THREADS>>>(bits, N);'],
    reload: [
      'radix_scatter_kernel<<<numBlocks, blockSize>>>(input, output, bits,',
      'N, iter);',
      'unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;',
      'unsigned int key = input[i];',
      'unsigned int bit = (key >> iter) & 1;',
    ],
    read: ['unsigned int numOnesBefore = bits[i];', 'unsigned int numOnesTotal = bits[N];'],
    dst: ['unsigned int dst = (bit == 0)', '? (i - numOnesBefore)', ': (N - numOnesTotal + numOnesBefore);'],
    store: ['output[dst] = key;'],
  };
  for (const s of STEPS)
    it(`${s.op}: lines ${s.lines.join(', ')}`, () => {
      expect(s.lines.map(line)).toEqual(want[s.op]);
    });
  it('the steps run in the kernel\'s order', () => {
    expect(STEPS.map((s) => s.op)).toEqual(['i', 'load', 'bit', 'store-bit', 'scan', 'reload', 'read', 'dst', 'store']);
  });
  it('the lesson\'s mapping to the page: line 03 is our lines 3 and 14, line 10 our line 33', () => {
    expect(lessonFlat).toContain("the page's line 03 is our lines 3 and 14, its line 10 is our line 33");
    expect(line(3)).toBe(line(14));
    expect(line(33)).toContain('exclusive_scan_kernel<<<');
  });
  it('the snippet comment and the lesson both say the bits array has N + 1 entries', () => {
    expect(line(27)).toBe('// bits has N + 1 entries.');
    expect(read(SNIPPET)).toContain('The bits array therefore has N + 1 entries');
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
  const r = spawnSync(exe, [], { encoding: 'utf8', env: { ...process.env, ASAN_OPTIONS: 'detect_leaks=0' }, timeout: 170000 });
  fs.rmSync(dir, { recursive: true, force: true });
  return { status: r.status, out: r.stdout, err: r.stderr };
}

const nums = (s: string) => s.trim().split(/\s+/).map(Number);
const field = (l: string, name: string) => {
  const part = l.split('|').map((x) => x.trim()).find((x) => x.startsWith(name + ' '));
  if (!part) throw new Error(`no ${name} in ${l}`);
  return nums(part.slice(name.length));
};

describe.skipIf(!cxx || !python)('s145_radix_iter.cu on the course emulator (labs/ch14/snippets/check/)', () => {
  let out = '';
  it('builds and runs under AddressSanitizer in all three thread orders', () => {
    const r = buildAndRun(SNIPPET, DRIVER, 's145');
    expect(r.status, r.err).toBe(0);
    out = r.out;
    expect(out).toContain('DONE');
    expect(out.split('\n').filter((l) => l.startsWith('order '))).toEqual(['order in order', 'order reversed', 'order shuffled']);
  }, 180000);
  it('Fig. 14.6 and the three other bits of the book keys: bits, the N + 1 scan entries, destinations and output match the model', () => {
    const lines = out.split('\n').filter((l) => l.startsWith('fig '));
    expect(lines).toHaveLength(12);
    for (const l of lines) {
      const bit = Number(l.slice(4, l.indexOf(':')));
      const pp = parallelPass(BOOK_KEYS, bit, 4);
      expect(field(l.slice(l.indexOf(':') + 1), 'bits'), l).toEqual(pp.bits);
      expect(field(l, 'scan'), l).toEqual(pp.scan);
      expect(field(l, 'dst'), l).toEqual(pp.threads.map((t) => t.dst));
      expect(field(l, 'out'), l).toEqual(pp.expected);
    }
    // Fig. 14.6's rows, as printed by the emulator run
    const f0 = lines[0];
    expect(field(f0, 'scan').slice(0, 16)).toEqual(nums('0 0 1 1 2 3 3 4 4 5 5 6 7 7 7 8'));
    expect(field(f0, 'scan')[16]).toBe(8);
    expect(field(f0, 'dst')).toEqual(nums('0 8 1 9 10 2 11 3 12 4 13 14 5 6 15 7'));
  }, 180000);
  it('four calls of radix_sort_iter give Fig. 14.4\'s four rows', () => {
    const lines = out.split('\n').filter((l) => l.startsWith('passes:'));
    expect(lines).toHaveLength(3);
    const run = lsdSort(BOOK_KEYS, 4, 1);
    for (const l of lines) expect(l.slice('passes:'.length).split('|').map(nums)).toEqual(run.passes.map((p) => p.output));
  }, 180000);
  it('random 8-bit keys: every bit\'s destinations match parallelPass', () => {
    const lines = out.split('\n').filter((l) => l.startsWith('keys '));
    expect(lines).toHaveLength(6);
    for (const l of lines) {
      const parts = l.slice(l.indexOf(':') + 1).split('|');
      const keys = nums(parts[0]);
      expect(keys).toHaveLength(16);
      for (const p of parts.slice(1)) {
        const [, b, ...dst] = p.trim().split(/\s+/);
        expect(dst.map(Number), l).toEqual(parallelPass(keys, Number(b), 4).threads.map((t) => t.dst));
      }
    }
  }, 180000);
  it('1,000 random 32-bit keys come out sorted after 32 passes, with a scan longer than its 256 threads', () => {
    expect(out.split('\n').filter((l) => l.startsWith('big:'))).toEqual(['big: sorted', 'big: sorted', 'big: sorted']);
  }, 180000);
});
