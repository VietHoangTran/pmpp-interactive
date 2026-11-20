import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { MP_TABLE, messagePassing, type LoadOrder, type Scope, type StoreOrder } from '../../sim/lookback';
import { functionSource } from '../../sim/toolchain';
import { FLAG_CU, FLAG_MSL, questionsS119 } from './S119';

// "Make the flag safe" (section 11.9) prints code from three checked files in
// labs/ch11/snippets. The CUDA listing is built by the model (src/sim/lookback.ts) for the
// combination the reader picks; this test ties each of the eight to the bodies of publish()
// and waitFor() in s119_flag_protocol.cu, and runs the CUDA front-end check (clang, the
// course's stub headers, the same flags as labs/pmpp.py) on a copy of the file for each
// combination. The two MSL tabs load their files as they are; this test compiles each at the
// language version its tab names, and checks the version below it rejects it (facts C3, C4).
// Nothing is run. The compiler parts are skipped when clang cannot parse CUDA or when the
// Metal compiler is missing.
const repo = path.resolve(__dirname, '../../../..');
const read = (p: string) => fs.readFileSync(path.join(repo, p), 'utf8');
const CU = 'labs/' + FLAG_CU;
const cu = read(CU);

/** The lines between a function's opening and closing brace. */
function body(source: string, name: string): string[] {
  const f = functionSource(source, name);
  if (!f) throw new Error(`no ${name}`);
  const lines = f.text.split('\n');
  const open = lines.findIndex((l) => l.trimEnd().endsWith('{'));
  return lines.slice(open + 1, -1);
}
/** The file's lines with the reader's orders and scope in place of the guaranteed ones. */
const variant = (text: string, store: StoreOrder, load: LoadOrder, scope: Scope) =>
  text
    .replaceAll('cuda::thread_scope_device', `cuda::thread_scope_${scope}`)
    .replaceAll('cuda::memory_order_release', `cuda::memory_order_${store}`)
    .replaceAll('cuda::memory_order_acquire', `cuda::memory_order_${load}`);

describe('the CUDA listing is the checked file', () => {
  it('the file holds the guaranteed combination: device scope, release store, acquire load', () => {
    const m = messagePassing('release', 'acquire', 'device');
    expect(body(cu, 'publish')).toEqual(m.writer.map((l) => '    ' + l));
    expect(body(cu, 'waitFor')).toEqual(m.reader.map((l) => '    ' + l));
  });
  it('every one of the eight combinations is the file with only the orders and the scope changed', () => {
    expect(MP_TABLE).toHaveLength(8);
    for (const { store, load, scope } of MP_TABLE) {
      const m = messagePassing(store, load, scope);
      const v = variant(cu, store, load, scope);
      expect(body(v, 'publish'), `${scope}/${store}/${load}`).toEqual(m.writer.map((l) => '    ' + l));
      expect(body(v, 'waitFor'), `${scope}/${store}/${load}`).toEqual(m.reader.map((l) => '    ' + l));
    }
  });
});

describe('the MSL tabs name checked files that say what the page says', () => {
  it('MSL 4.1: acquire and release only through the mem_flags overloads; MSL 3.2: relaxed atomics and device fences', () => {
    const [a, f] = FLAG_MSL.map((x) => read('labs/' + x.file));
    expect(a).toContain('atomic_store_explicit(&flags[bid], 1u, memory_order_release, mem_flags::mem_device);');
    expect(a).toContain('atomic_load_explicit(&flags[bid - 1], memory_order_acquire, mem_flags::mem_device)');
    expect(f).not.toMatch(/memory_order_(acquire|release)/);
    const fence = 'atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);';
    // a fence after the plain store and before the flag store; after the flag load and before the plain load
    expect(body(f, 'publish').map((l) => l.trim())).toEqual(['partialSums[bid] = prev + blockSum;', fence, 'atomic_store_explicit(&flags[bid], 1u, memory_order_relaxed);']);
    expect(body(f, 'waitFor').map((l) => l.trim())).toEqual([
      'while (atomic_load_explicit(&flags[bid - 1], memory_order_relaxed) == 0) {}',
      fence,
      'return partialSums[bid - 1];',
    ]);
    for (const s of [a, f]) expect(s).toContain('coherent(device) device float* partialSums');
  });
  it('the tabs use the fixed vocabulary: unverified, because the files are compiled and never run', () => {
    for (const x of FLAG_MSL) {
      expect(x.status.startsWith('unverified')).toBe(true);
      expect(x.status).not.toMatch(/\bverified on/);
    }
  });
  it("question q-scope claims only what NVIDIA marks: block scope on the store", () => {
    const q = questionsS119['ch11/11.9/q-scope'];
    expect(q.kind).toBe('choice');
    const right = q.kind === 'choice' ? q.options.find((o) => o.correct)!.label : '';
    expect(String(right)).toContain('block scope on the store');
    expect(String(right)).not.toMatch(/exact variant/);
  });
});

// ---------------------------------------------------------------- compilers
const SCRATCH = fs.mkdtempSync(path.join(os.tmpdir(), 'ch11-flag-'));
const INC = path.join(repo, 'labs/common/include');
const clang = process.env.PMPP_CLANG ?? 'clang++';
const FE = ['-x', 'cuda', '--cuda-host-only', '-nocudainc', '-nocudalib', '-fsyntax-only', '-std=c++17', '-I', INC, '-I', path.join(INC, 'pmpp/emu/shim'),
  '-include', 'pmpp/check/cuda_stub.h', '-DTILE_WIDTH=16'];
const probe = path.join(SCRATCH, 'probe.cu');
fs.writeFileSync(probe, '__global__ void k(float* a) { a[threadIdx.x] = 0; }\nvoid f(float* a) { k<<<1, 4>>>(a); }\n');
const canParseCuda = spawnSync(clang, [...FE, probe], { encoding: 'utf8' }).status === 0;
const hasMetal = spawnSync('xcrun', ['-sdk', 'macosx', 'metal', '--version'], { encoding: 'utf8' }).status === 0;

describe.skipIf(!canParseCuda)('the CUDA front-end check, on every combination the page can show', () => {
  it('all eight compile: a wrong order or scope is not a compile error, which is what makes it dangerous', () => {
    for (const { store, load, scope } of MP_TABLE) {
      const file = path.join(SCRATCH, `flag_${scope}_${store}_${load}.cu`);
      fs.writeFileSync(file, variant(cu, store, load, scope));
      const r = spawnSync(clang, [...FE, file], { encoding: 'utf8' });
      expect(r.status, `${scope}/${store}/${load}: ${r.stderr}`).toBe(0);
    }
  }, 60000);
  it('the check is not vacuous: a misspelt member inside the device function fails', () => {
    const file = path.join(SCRATCH, 'flag_broken.cu');
    fs.writeFileSync(file, cu.replace('f.store(1,', 'f.stor(1,'));
    const r = spawnSync(clang, [...FE, file], { encoding: 'utf8' });
    expect(r.status).not.toBe(0);
    expect(r.stderr).toContain("no member named 'stor'");
  }, 60000);
});

const metal = (file: string, std: string) =>
  spawnSync('xcrun', ['-sdk', 'macosx', 'metal', `-std=${std}`, '-c', path.join(repo, 'labs', file), '-o', path.join(SCRATCH, path.basename(file) + '.' + std + '.air')], { encoding: 'utf8' });

describe.skipIf(!hasMetal)('the MSL files compile at the versions their tabs name, and not below', () => {
  it('MSL 4.1: compiles at metal4.1; metal4.0 has no acquire or release', () => {
    const f = FLAG_MSL[0].file;
    const ok = metal(f, 'metal4.1');
    expect(ok.status, ok.stderr).toBe(0);
    const below = metal(f, 'metal4.0');
    expect(below.status).not.toBe(0);
    expect(below.stderr).toContain("undeclared identifier 'memory_order_release'");
  }, 60000);
  it('MSL 3.2, fences: compiles at metal3.2 and metal4.1; metal3.1 rejects coherent(device)', () => {
    const f = FLAG_MSL[1].file;
    for (const std of ['metal3.2', 'metal4.1']) {
      const ok = metal(f, std);
      expect(ok.status, `${std}: ${ok.stderr}`).toBe(0);
    }
    const below = metal(f, 'metal3.1');
    expect(below.status).not.toBe(0);
    expect(below.stderr).toMatch(/coherent\(device\)|expected expression/);
  }, 60000);
});
