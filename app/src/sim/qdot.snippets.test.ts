import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  FRAG_N, FRAGMENTS, FRAGMENT_MODEL, QK, blockResult, dequant, innerSum, intRow, normals, parseFragments, q4Blocks, runFragment,
} from './qdot';

// Section X2.6's bug spot shows four fragments loaded at run time from
// labs/x2/snippets/accumulators.md. This test ties what the page shows to what runFragment
// models: the same four ids in the same order, each fragment's accumulator declaration and the
// operation that adds into it, and the type the model sums in.
const repo = path.resolve(__dirname, '../../..');
const md = fs.readFileSync(path.join(repo, 'labs/x2/snippets/accumulators.md'), 'utf8');
const frags = parseFragments(md);
const byId = Object.fromEntries(frags.map((f) => [f.id, f]));
/** Code lines without comments, trimmed. */
const lines = (code: string) => code.replace(/\/\/.*$/gm, '').split('\n').map((l) => l.trim()).filter(Boolean);

describe('labs/x2/snippets/accumulators.md against qdot.ts', () => {
  it('has the four fragments of the model, in its order, each with a title, a language and code', () => {
    expect(frags.map((f) => f.id)).toEqual(FRAGMENTS);
    for (const f of frags) {
      expect(f.title.length).toBeGreaterThan(5);
      expect(['MSL', 'CUDA']).toContain(f.lang);
      expect(f.code.split('\n').length).toBeGreaterThanOrEqual(4);
    }
  });

  it('declares each accumulator as the model says, and adds into it with the modelled operation', () => {
    for (const id of FRAGMENTS) {
      const { decl, op } = FRAGMENT_MODEL[id];
      const code = byId[id].code;
      expect(lines(code)[0].startsWith(decl + ' = '), id).toBe(true);
      expect(code, id).toContain(op);
    }
  });

  it('fragment 1: half4 operands into dot, so four products round in half before the float sum', () => {
    const c = byId['metal-half-dot'].code;
    expect(byId['metal-half-dot'].lang).toBe('MSL');
    expect(c).toMatch(/half4 w = half4\(d\) \* \(half4\(q\[i\/4\]\) - 8\.0h\);/);
    expect(c).toContain('sumf += dot(w, half4(x4[i/4]));');
    expect(c).toContain('i += 4');
    expect(FRAGMENT_MODEL['metal-half-dot'].inner).toBe('half-dot4');
  });

  it('fragment 2: one __half2 accumulator over the whole row, its halves added in float at the end', () => {
    const c = byId['cuda-hfma2'].code;
    expect(c).toContain('acc = __hfma2(w2[i], x2[i], acc);');
    expect(c).toContain('i < n / 2');
    expect(c).toContain('__low2float(acc) + __high2float(acc)');
    expect(FRAGMENT_MODEL['cuda-hfma2'].inner).toBe('half2-row');
  });

  it('fragments 3 and 4 differ only in the type of sumi, and the model sums them in int16 and int32', () => {
    const a = lines(byId['short-sum'].code);
    const b = lines(byId['int-then-float'].code);
    expect(a[0]).toBe('short sumi = 0;');
    expect(b[0]).toBe('int sumi = 0;');
    expect(a.slice(1)).toEqual(b.slice(1));
    // eight dp4a steps per block: a loop of 4 with two dot4 calls each, the step the model uses
    expect(byId['short-sum'].code.match(/dot4\(/g)!.length).toBe(2);
    expect(a.some((l) => l.startsWith('for (int i = 0; i < 4; ++i)'))).toBe(true);
    expect(a.at(-1)).toBe('sumf += d4 * (sumi * d8 - 8.0f * s8);');
    expect(FRAGMENT_MODEL['short-sum'].inner).toBe('int16');
    expect(FRAGMENT_MODEL['int-then-float'].inner).toBe('int32');
  });

  it('the model gives the verdicts the page grades against: only the int-then-float fragment is safe', () => {
    const runs = FRAGMENTS.map((id) => runFragment(id));
    expect(runs.map((r) => r.safe)).toEqual([false, false, false, true]);
    const short = runs[2];
    const int = runs[3];
    // same on realistic data, so a first test would pass; wraps on the worst case
    expect(short.realistic).toBe(int.realistic);
    expect(short.worst.event).toBe('wrap');
    expect(int.worst.event).toBe('none');
    expect(int.worst.result).toBe(int.worst.reference);
  });
});

// ---------------------------------------------------------------- on the course emulator
// Fragments 2, 3 and 4 are CUDA. labs/x2/snippets/check/accumulators_check.cu includes each one's
// fenced code, exactly as the page shows it, inside a one-thread kernel; it is translated by
// labs/tools/cu2emu.py, built against pmpp/emu/cuda_emu.h (which models __half2, __hfma2 and
// __dp4a) and run on the rows runFragment uses. Its results must be runFragment's, bit for bit on
// the worst-case rows and in the relative error on the realistic ones. -ffp-contract=off keeps
// each float operation rounded on its own, as the model computes them (nvcc contracts a * b + c
// by default, so a GPU's last bits may differ). Fragment 1 is MSL and is not run here.
const has = (cmd: string, args: string[]) => spawnSync(cmd, args).status === 0;
const cxx = ['c++', 'clang++', 'g++'].find((c) => has(c, ['--version']));
const python = ['python3'].find((c) => has(c, ['--version']));
const clang = process.env.PMPP_CLANG || ['/usr/bin/clang++', '/opt/homebrew/opt/llvm/bin/clang++', '/usr/local/bin/clang++'].find((p) => fs.existsSync(p));
const INC = path.join(repo, 'labs/common/include');
const DRIVER = path.join(repo, 'labs/x2/snippets/check/accumulators_check.cu');
const CUDA_FRAGMENTS = ['cuda-hfma2', 'short-sum', 'int-then-float'] as const;

const bitsOf = (f: number) => {
  const dv = new DataView(new ArrayBuffer(4));
  dv.setFloat32(0, f);
  return dv.getUint32(0);
};
const floatOf = (u: number) => {
  const dv = new DataView(new ArrayBuffer(4));
  dv.setUint32(0, u);
  return dv.getFloat32(0);
};
/** x as n * 2^e exactly (finite binary64). */
function dyadic(x: number): { n: bigint; e: number } {
  if (x === 0) return { n: 0n, e: 0 };
  const dv = new DataView(new ArrayBuffer(8));
  dv.setFloat64(0, x);
  const hi = dv.getUint32(0), lo = dv.getUint32(4);
  const ex = (hi >>> 20) & 0x7ff;
  let m = (BigInt(hi & 0xfffff) << 32n) | BigInt(lo);
  if (ex !== 0) m |= 1n << 52n;
  return { n: hi >>> 31 ? -m : m, e: ex === 0 ? -1074 : ex - 1075 };
}
/** sum of a[i] * b[i] computed exactly, then rounded once to binary64. */
function exactDot(a: number[], b: number[]): number {
  const terms = a.map((v, i) => { const p = dyadic(v), q = dyadic(b[i]); return { n: p.n * q.n, e: p.e + q.e }; });
  const e = Math.min(...terms.map((t) => t.e));
  let n = 0n;
  for (const t of terms) n += t.n << BigInt(t.e - e);
  if (n === 0n) return 0;
  const neg = n < 0n;
  let m = neg ? -n : n;
  let k = e;
  const bits = m.toString(2).length;
  if (bits > 64) {
    const drop = bits - 64;
    const sticky = m & ((1n << BigInt(drop)) - 1n) ? 1n : 0n;
    m = (m >> BigInt(drop)) | sticky;
    k += drop;
  }
  const v = Number(m) * 2 ** k;
  return neg ? -v : v;
}
const word = (b: number[]) => ((b[0] & 255) | ((b[1] & 255) << 8) | ((b[2] & 255) << 16) | ((b[3] & 255) << 24)) >>> 0;
type IntBlock = { qw: number[]; qx: number[]; dw: number; dx: number; s: number };
/** One block as the fragments read it: v = the 16 Q4_0 bytes (weight j low nibble, j + 16 high) as 4 words, u = the 32 codes as 8 words. */
function intLine(label: string, blocks: IntBlock[]): string {
  const parts = ['I', label, String(blocks.length)];
  for (const b of blocks) {
    const bytes = Array.from({ length: 16 }, (_, j) => b.qw[j] | (b.qw[j + 16] << 4));
    for (let i = 0; i < 4; i++) parts.push(String(word(bytes.slice(4 * i, 4 * i + 4))));
    for (let k = 0; k < 8; k++) parts.push(String(word(b.qx.slice(4 * k, 4 * k + 4))));
    for (const f of [b.dw, b.dx, b.s]) parts.push(String(bitsOf(f)));
  }
  return parts.join(' ');
}
const floatLine = (label: string, w: number[], x: number[]) => ['F', label, String(w.length), ...w.map(bitsOf), ...x.map(bitsOf)].join(' ');

function buildDriver(): { exe: string; dir: string } {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'pmpp-x2-frag-'));
  for (const id of CUDA_FRAGMENTS) fs.writeFileSync(path.join(dir, id + '.inc'), byId[id].code + '\n');
  const cu = path.join(dir, 'accumulators_check.cu');
  fs.copyFileSync(DRIVER, cu);
  const cpp = path.join(dir, 'accumulators_check.cpp');
  const t = spawnSync(python!, [path.join(repo, 'labs/tools/cu2emu.py'), cu, cpp], { encoding: 'utf8' });
  if (t.status !== 0) throw new Error('cu2emu failed:\n' + t.stdout + t.stderr);
  const exe = path.join(dir, 'frag');
  const c = spawnSync(cxx!, ['-std=c++17', '-O1', '-g', '-ffp-contract=off', '-fsanitize=address', '-fno-omit-frame-pointer', '-pthread',
    '-DPMPP_BACKEND_EMU', '-I', dir, '-I', INC, '-I', path.join(INC, 'pmpp/emu/shim'), '-include', 'pmpp/emu/cuda_emu.h', cpp, '-o', exe],
  { encoding: 'utf8' });
  if (c.status !== 0) throw new Error('compile failed:\n' + c.stderr);
  return { exe, dir };
}

describe.skipIf(!cxx || !python)('the CUDA fragments on the course emulator (labs/x2/snippets/check/accumulators_check.cu)', () => {
  it('compute what runFragment says: the same worst-case results, the same realistic errors', () => {
    const { exe, dir } = buildDriver();
    // realistic rows, as runFragment builds them (n = FRAG_N, seed 5)
    const seed = 5;
    const w = normals(FRAG_N, seed);
    const x = normals(FRAG_N, seed + 1);
    const wd = q4Blocks(w).flatMap(dequant);
    const row = intRow('q4q8', 'realistic', FRAG_N, seed);
    // worst-case rows: every Q4_0 code 15 and every activation code 127, d = 1 (the model's worstRow and worstFloat)
    const worstBlocks: IntBlock[] = Array.from({ length: FRAG_N / QK }, () => ({
      qw: new Array(QK).fill(15), qx: new Array(QK).fill(127), dw: 1, dx: 1, s: QK * 127 }));
    const input = [floatLine('realistic', wd, x), floatLine('worst', new Array(FRAG_N).fill(7), new Array(FRAG_N).fill(127)),
      intLine('realistic', row.blocks), intLine('worst', worstBlocks)].join('\n') + '\n';
    const r = spawnSync(exe, [], { input, encoding: 'utf8', env: { ...process.env, ASAN_OPTIONS: 'detect_leaks=0' }, timeout: 120000 });
    fs.rmSync(dir, { recursive: true, force: true });
    expect(r.status, r.stderr).toBe(0);
    expect(r.stdout).toContain('DONE');
    const got: Record<string, number> = {};
    for (const l of r.stdout.split('\n')) {
      const m = /^(\S+) (realistic|worst) (\d+)$/.exec(l);
      if (m) got[m[1] + '/' + m[2]] = floatOf(Number(m[3]));
    }
    expect(Object.keys(got).length).toBe(6);
    // the error measures runFragment uses
    let scF = 0;
    for (let j = 0; j < FRAG_N; j++) scF += Math.abs(wd[j] * x[j]);
    const exF = exactDot(wd, x);
    let refI = 0, scI = 0;
    for (const b of row.blocks) {
      const e = innerSum(b.qw, b.qx, 'int32', 4).exact[7];
      const re = b.dw * (e * b.dx - 8 * b.s);
      refI += re;
      scI += Math.abs(re);
    }
    for (const id of CUDA_FRAGMENTS) {
      const model = runFragment(id);
      const v = got[id + '/realistic'];
      const rel = id === 'cuda-hfma2' ? Math.abs(v - exF) / scF : Math.abs(v - refI) / scI;
      expect(rel, id + ' realistic').toBe(model.realistic);
      expect(got[id + '/worst'], id + ' worst').toBe(model.worst.result);
    }
    // and the verdicts the page grades against show in the emulator's numbers
    expect(got['cuda-hfma2/worst']).toBe(Infinity);
    expect(got['short-sum/worst']).not.toBe(got['int-then-float/worst']);
    expect(got['int-then-float/worst']).toBe((FRAG_N / QK) * QK * 7 * 127);
    expect(got['short-sum/realistic']).toBe(got['int-then-float/realistic']);
    // a sanity check on the block sum the model's blockResult gives for one worst block
    expect(blockResult('q4q8', worstBlocks[0], QK * 15 * 127)).toBe(QK * 7 * 127);
  }, 180000);

  it.skipIf(!clang)('pass the CUDA front-end check (clang, the course stub headers) with the fragments in place', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'pmpp-x2-fe-'));
    for (const id of CUDA_FRAGMENTS) fs.writeFileSync(path.join(dir, id + '.inc'), byId[id].code + '\n');
    const cu = path.join(dir, 'accumulators_check.cu');
    fs.copyFileSync(DRIVER, cu);
    const fe = (file: string) => spawnSync(clang as string, ['-x', 'cuda', '--cuda-host-only', '-nocudainc', '-nocudalib', '-fsyntax-only',
      '-std=c++17', '-I', INC, '-I', path.join(INC, 'pmpp', 'emu', 'shim'), '-include', 'pmpp/check/cuda_stub.h', file], { encoding: 'utf8' });
    const r = fe(cu);
    // a fragment that called a host-only function from the kernel would fail here
    fs.writeFileSync(path.join(dir, 'short-sum.inc'), byId['short-sum'].code.replace('dot4(', 'hostOnlyDot4(') + '\n');
    fs.writeFileSync(path.join(dir, 'host.h'), 'int hostOnlyDot4(int a, int b, int c);\n');
    const src = fs.readFileSync(cu, 'utf8').replace('#include <cuda_fp16.h>', '#include <cuda_fp16.h>\n#include "host.h"');
    fs.writeFileSync(cu, src);
    const bad = fe(cu);
    fs.rmSync(dir, { recursive: true, force: true });
    expect(r.status, r.stderr).toBe(0);
    expect(bad.status).not.toBe(0);
  });
});
