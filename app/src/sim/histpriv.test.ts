import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  BROKEN,
  GARBAGE,
  HUNT_PIXELS,
  SCHEDULES,
  VARIANT,
  VARIANTS,
  VOCABS,
  caughtBy,
  diffLines,
  emuOrders,
  ex5a,
  ex5b,
  expectedBins,
  kib,
  makeValues,
  plan,
  races,
  region,
  sceneData,
  simulate,
  siteLine,
  type Schedule,
  type Site,
  type Variant,
} from './histpriv';
import { functionSource } from './toolchain';

const repo = path.resolve(__dirname, '../../..');
const SNIPPET = path.join(repo, 'labs/ch09/snippets/privatised_variants.cu');
const source = fs.readFileSync(SNIPPET, 'utf8');
const kernelOf = (v: Variant) => functionSource(source, VARIANT[v].kernel)!;

describe('the explorer: copies and contenders', () => {
  // The browser check (e2e/ch09-s94.mjs) asserts these same numbers on the page.
  const constant = makeValues('constant', 4096, 256);
  it('a constant image of 4,096 pixels, 256 threads per block: contention falls by the number of copies', () => {
    const rows = (['public', 'two', 'four', 'block'] as const).map((s) => plan(constant, 256, s, 1, 256, 4, 32 * 1024));
    expect(rows.map((r) => r.blocks)).toEqual([16, 16, 16, 16]);
    expect(rows.map((r) => r.copies)).toEqual([1, 2, 4, 16]);
    expect(rows.map((r) => r.updateContention)).toEqual([4096, 2048, 1024, 256]);
    expect(rows.map((r) => r.commitAtomics)).toEqual([0, 2, 4, 16]);
    expect(rows.map((r) => r.commitContention)).toEqual([0, 2, 4, 16]);
    expect(rows.map((r) => r.commitBound)).toEqual([0, 512, 1024, 4096]);
    expect(rows.map((r) => r.globalAtomics)).toEqual([4096, 4096 + 2, 4096 + 4, 16]);
    expect(rows[3].where).toBe('shared');
    expect(rows.map((r) => r.privateBytes)).toEqual([0, 2048, 4096, 16384]);
  });
  it('a uniform image: the commit, not the update, is what grows with copies', () => {
    const u = makeValues('uniform', 65_536, 256);
    const pub = plan(u, 256, 'public', 1, 256, 4, Infinity);
    const blk = plan(u, 256, 'block', 1, 256, 4, Infinity);
    expect(blk.updateContention).toBeLessThan(pub.updateContention);
    expect(blk.commitAtomics).toBeLessThanOrEqual(blk.commitBound);
    expect(blk.commitAtomics).toBeGreaterThan(blk.blocks * 100); // most bins are hit in a block of 256 random pixels
    expect(blk.commitContention).toBeLessThanOrEqual(blk.blocks);
  });
  it('256 bins fit anywhere; a vocabulary fits in no 32 KiB budget', () => {
    expect(plan(constant, 256, 'block', 1, 256, 4, 32 * 1024).fits).toBe(true);
    const q = VOCABS.find((v) => v.id === 'qwen3')!;
    const tokens = makeValues('bright', 4096, q.bins);
    const p = plan(tokens, 256, 'block', 1, q.bins, 4, 32 * 1024);
    expect(p.fits).toBe(false);
    expect(p.where).toBe('global');
    expect(p.copyBytes).toBe(607_744);
  });
  it('the vocabulary sizes in KiB (notes/today/ch09-facts.md, D7)', () => {
    expect(VOCABS.map((v) => v.bins)).toEqual([50_257, 151_936, 201_088]);
    expect(Number(kib(VOCABS[0].bins * 4).toFixed(1))).toBe(196.3);
    expect(kib(VOCABS[1].bins * 4)).toBe(593.5);
    expect(kib(VOCABS[2].bins * 4)).toBe(785.5);
  });
  it('exercise 5 (p. 220): 524,288, 65,536 and, coarsened by 4, 16,384', () => {
    expect(ex5a(524_288)).toBe(524_288);
    expect(ex5b(524_288, 128, 1024)).toBe(65_536);
    expect(ex5b(524_288, 128, 1024, 4)).toBe(16_384);
    // The model agrees when every bin of every block is hit (the "at most").
    const all = Array.from({ length: 524_288 }, (_, i) => i % 128);
    const p = plan(all, 1024, 'block', 1, 128, 4, Infinity);
    expect(p.commitAtomics).toBe(65_536);
    expect(plan(all, 1024, 'public', 1, 128, 4, Infinity).globalAtomics).toBe(524_288);
  });
  it('values stay inside the bins, and the generators are seeded', () => {
    for (const bins of [8, 128, 256, 50_257])
      for (const kind of ['uniform', 'bright', 'constant'] as const) {
        const v = makeValues(kind, 2000, bins);
        expect(v.every((x) => Number.isInteger(x) && x >= 0 && x < bins)).toBe(true);
        expect(makeValues(kind, 2000, bins)).toEqual(v);
      }
    // 'bright' tokens are skewed: the most frequent id is far above the uniform share.
    const t = makeValues('bright', 20_000, 50_257);
    expect(plan(t, 256, 'public', 1, 50_257, 4, Infinity).updateContention).toBeGreaterThan(1000);
  });
});

// The outcome matrix of the bug hunt. histPrivatised is the reference; the seven
// broken kernels follow. Every number here was also produced by the emulator itself
// (see the last describe block).
const MATRIX: Record<Variant, Record<Schedule, number[]>> = {
  correct: { forward: [0, 3, 1, 1, 0, 1, 2, 0], reverse: [0, 3, 1, 1, 0, 1, 2, 0], shuffle: [0, 3, 1, 1, 0, 1, 2, 0] },
  'no-zeroing': {
    forward: [0, 3, 1, 1, 0, 1, 2, 0].map((c) => (2 * GARBAGE + c) % 2 ** 32),
    reverse: [0, 3, 1, 1, 0, 1, 2, 0].map((c) => (2 * GARBAGE + c) % 2 ** 32),
    shuffle: [0, 3, 1, 1, 0, 1, 2, 0].map((c) => (2 * GARBAGE + c) % 2 ** 32),
  },
  'no-barrier-after-zeroing': { forward: [0, 1, 1, 1, 0, 1, 2, 0], reverse: [0, 3, 0, 1, 0, 1, 2, 0], shuffle: [0, 2, 1, 1, 0, 1, 2, 0] },
  'no-barrier-before-commit': { forward: [0, 3, 0, 1, 0, 1, 2, 0], reverse: [0, 1, 1, 1, 0, 1, 2, 0], shuffle: [0, 2, 0, 1, 0, 1, 2, 0] },
  'plain-shared-update': { forward: [0, 3, 1, 1, 0, 1, 2, 0], reverse: [0, 3, 1, 1, 0, 1, 2, 0], shuffle: [0, 3, 1, 1, 0, 1, 2, 0] },
  'plain-commit': { forward: [0, 3, 1, 1, 0, 1, 2, 0], reverse: [0, 3, 1, 1, 0, 1, 2, 0], shuffle: [0, 3, 1, 1, 0, 1, 2, 0] },
  'block-scope-commit': { forward: [0, 3, 1, 1, 0, 1, 2, 0], reverse: [0, 3, 1, 1, 0, 1, 2, 0], shuffle: [0, 3, 1, 1, 0, 1, 2, 0] },
  'commit-bound-blockDim': { forward: [0, 3, 1, 1, 0, 0, 0, 0], reverse: [0, 3, 1, 1, 0, 0, 0, 0], shuffle: [0, 3, 1, 1, 0, 0, 0, 0] },
};
const RACES: Record<Variant, string[]> = {
  correct: [],
  'no-zeroing': [],
  'no-barrier-after-zeroing': ['zero|update'],
  'no-barrier-before-commit': ['update|commit-read'],
  'plain-shared-update': ['update|update'],
  'plain-commit': ['commit-add|commit-add'],
  'block-scope-commit': [],
  'commit-bound-blockDim': [],
};

describe('the bug hunt: seven broken privatised kernels', () => {
  it('the hunt counts eight pixels into 0,3,1,1,0,1,2,0', () => {
    expect(expectedBins(HUNT_PIXELS)).toEqual([0, 3, 1, 1, 0, 1, 2, 0]);
  });
  for (const v of VARIANTS)
    it(`${v}: the emulator's three orders and the race check`, () => {
      for (const s of SCHEDULES) expect(simulate(v, s).bins, s).toEqual(MATRIX[v][s]);
      expect(races(v).map((r) => r.sites.join('|'))).toEqual(RACES[v]);
    });
  it('which tools see which defect: emu sees ordering, race sees missing atomics, nothing sees the scope', () => {
    const seen = Object.fromEntries(BROKEN.map((v) => [v, caughtBy(v)]));
    expect(seen).toEqual({
      'no-zeroing': { emu: true, race: false },
      'no-barrier-after-zeroing': { emu: true, race: true },
      'no-barrier-before-commit': { emu: true, race: true },
      'plain-shared-update': { emu: false, race: true },
      'plain-commit': { emu: false, race: true },
      'block-scope-commit': { emu: false, race: false },
      'commit-bound-blockDim': { emu: true, race: false },
    });
    expect(caughtBy('correct')).toEqual({ emu: false, race: false });
  });
  it('the counts that are wrong, per order', () => {
    const wrong = (v: Variant) => SCHEDULES.map((s) => simulate(v, s).wrong.length);
    expect(wrong('no-zeroing')).toEqual([8, 8, 8]);
    expect(wrong('no-barrier-after-zeroing')).toEqual([1, 1, 1]);
    expect(wrong('no-barrier-before-commit')).toEqual([1, 1, 2]);
    expect(wrong('commit-bound-blockDim')).toEqual([2, 2, 2]);
  });
  it('the emulator\'s orders: identity, reverse, and a shuffle that continues across blocks', () => {
    expect(emuOrders('forward', 2, 4)).toEqual([[0, 1, 2, 3], [0, 1, 2, 3]]);
    expect(emuOrders('reverse', 2, 4)).toEqual([[3, 2, 1, 0], [3, 2, 1, 0]]);
    const sh = emuOrders('shuffle', 2, 4);
    for (const o of sh) expect([...o].sort()).toEqual([0, 1, 2, 3]);
    expect(sh[0]).not.toEqual(sh[1]);
  });
});

describe('the snippet file holds every kernel the hunt shows', () => {
  it('one kernel per variant, each defined once', () => {
    for (const v of VARIANTS) {
      expect(kernelOf(v), VARIANT[v].kernel).not.toBeNull();
      expect(source.split(`void ${VARIANT[v].kernel}(`).length - 1).toBe(1);
    }
  });
  it('each broken kernel differs from the reference in exactly the way its shape says', () => {
    const ref = kernelOf('correct').text;
    const d = (v: Variant) => diffLines(ref, kernelOf(v).text);
    expect(d('correct')).toMatchObject({ removed: [], added: [] });
    expect(d('no-zeroing')).toMatchObject({ removed: ['for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x)', 'bins_s[b] = 0;'], added: [], removedAt: [3, 4] });
    expect(d('no-barrier-after-zeroing')).toMatchObject({ removed: ['__syncthreads();'], added: [], removedAt: [5] });
    expect(d('no-barrier-before-commit')).toMatchObject({ removed: ['__syncthreads();'], added: [], removedAt: [11] });
    expect(d('plain-shared-update').added).toEqual(['++bins_s[image[i]];']);
    expect(d('plain-shared-update').removed).toHaveLength(2);
    expect(d('plain-commit').added).toEqual(['bins[b] += count;']);
    expect(d('block-scope-commit').added).toEqual(['cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bin_ref(bins[b]);']);
    expect(d('commit-bound-blockDim').added).toEqual(['for (unsigned int b = threadIdx.x; b < blockDim.x; b += blockDim.x) {']);
    // Which barrier went: the first one sits before the index line, the second after the update.
    const lines = (v: Variant) => kernelOf(v).text.split('\n').map((l) => l.trim());
    expect(lines('no-barrier-after-zeroing').indexOf('__syncthreads();')).toBeGreaterThan(lines('no-barrier-after-zeroing').indexOf('bin_ref.fetch_add(1, cuda::memory_order_relaxed);'));
    expect(lines('no-barrier-before-commit').indexOf('__syncthreads();')).toBeLessThan(lines('no-barrier-before-commit').indexOf('bin_ref.fetch_add(1, cuda::memory_order_relaxed);'));
  });
  it('every racing site can be found in its kernel', () => {
    for (const v of VARIANTS)
      for (const r of races(v)) for (const s of r.sites) expect(siteLine(kernelOf(v).text, s), `${v} ${s}`).toBeGreaterThan(0);
  });
});

describe('region markers in lab files', () => {
  const file = ['#include <x>', '// region: pool', '__global__ void k() {', '  int a;', '}', '', '// endregion', 'int z;'].join('\n');
  it('finds a marked region and its lines', () => {
    expect(region(file, 'pool')).toEqual({ text: '__global__ void k() {\n  int a;\n}', firstLine: 3, lastLine: 5 });
    expect(region(file.replace('// region: pool', '// [pool]').replace('// endregion', '// [/pool]'), 'pool')?.firstLine).toBe(3);
    expect(region(file.replace('// region: pool', '// BEGIN shared').replace('// endregion', '// END shared'), 'shared')?.lastLine).toBe(5);
    expect(region(file, 'shared')).toBeNull();
    expect(region(file, 'poo')).toBeNull();
  });
});

describe('the narrated scene "Privatise, then commit"', () => {
  const D = sceneData();
  it('24 pixels, 3 blocks of 8 threads, 8 bins: the numbers the animation draws', () => {
    expect(D.blocks).toBe(3);
    expect(D.total).toEqual([1, 1, 1, 3, 0, 0, 4, 14]);
    expect(D.busiest).toBe(7);
    expect(D.publicContenders).toBe(14);
    expect(D.perBlockBusiest).toEqual([5, 4, 5]);
    expect(D.privateContenders).toBe(5);
    expect(D.commitContention).toBe(3);
    expect(D.globalBefore).toBe(24);
    expect(D.globalAfter).toBe(11);
    expect(D.commits).toHaveLength(11);
    expect(D.committers[7]).toBe(3);
  });
  const script = (n: number) => fs.readFileSync(path.join(repo, `content/ch09/narration/privatise-${n}.txt`), 'utf8');
  it('the scripts say the model\'s numbers', () => {
    expect(script(1)).toContain(`${D.pixels.length} pixels`);
    expect(script(1)).toContain(`bin ${D.busiest}`);
    expect(script(1)).toContain(`${D.publicContenders} threads from all ${D.blocks} blocks`);
    expect(script(3)).toContain(`${D.privateContenders} contenders`);
    expect(script(3)).toContain(`instead of ${D.publicContenders}`);
    expect(script(4)).toContain(`at most ${D.commitContention} contenders`);
    expect(script(5)).toContain(`issued ${D.globalBefore}`);
    expect(script(5)).toContain(`issued ${D.globalAfter}`);
  });
  it('each script is 35 to 60 words', () => {
    for (let n = 1; n <= 5; n++) {
      const words = script(n).replace(/\[[^\]]*\]/g, '').trim().split(/\s+/).length;
      expect(words, `privatise-${n}`).toBeGreaterThanOrEqual(35);
      expect(words, `privatise-${n}`).toBeLessThanOrEqual(60);
    }
  });
});

// The snippet file on the course's emulator, built as the lab runner builds the emu and
// race backends (labs/cmake/Backends.cmake): emu with AddressSanitizer and the three
// thread orders, race with ThreadSanitizer. Skipped when no C++ compiler works.
const compiler = ['c++', 'clang++', 'g++'].find((c) => spawnSync(c, ['--version']).status === 0);
describe.skipIf(!compiler)('the snippets on the emulator agree with the model', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch09-histpriv-'));
  const inc = path.join(repo, 'labs/common/include');
  const exe = { emu: path.join(dir, 'emu'), race: path.join(dir, 'race') };
  const env = {
    ...process.env,
    ASAN_OPTIONS: 'detect_leaks=0',
    TSAN_OPTIONS: 'abort_on_error=0:exitcode=66:halt_on_error=0:report_thread_leaks=0',
  };
  let raceBuilt = false;
  it('builds', () => {
    const cu =
      '#line 1 "privatised_variants.cu"\n' + source + '\n#line 1 "check/privatised_check.cu"\n' +
      fs.readFileSync(path.join(repo, 'labs/ch09/snippets/check/privatised_check.cu'), 'utf8');
    fs.writeFileSync(path.join(dir, 'all.cu'), cu);
    const t = spawnSync('python3', [path.join(repo, 'labs/tools/cu2emu.py'), path.join(dir, 'all.cu'), path.join(dir, 'all.cpp')], { encoding: 'utf8' });
    expect(t.status, t.stdout + t.stderr).toBe(0);
    const common = ['-std=c++17', '-O1', '-g', '-fno-omit-frame-pointer', '-pthread', '-I', inc, '-I', path.join(inc, 'pmpp/emu/shim'), '-include', 'pmpp/emu/cuda_emu.h', path.join(dir, 'all.cpp')];
    const c = spawnSync(compiler!, ['-fsanitize=address', '-DPMPP_BACKEND_EMU', ...common, '-o', exe.emu], { encoding: 'utf8' });
    expect(c.status, c.stderr).toBe(0);
    const r = spawnSync(compiler!, ['-fsanitize=thread', '-DPMPP_BACKEND_RACE', ...common, '-o', exe.race], { encoding: 'utf8' });
    raceBuilt = r.status === 0; // ThreadSanitizer is not available everywhere
  }, 180_000);
  for (const v of VARIANTS)
    it(`${VARIANT[v].kernel}: emu's three orders give the model's bins`, () => {
      const r = spawnSync(exe.emu, [VARIANT[v].kernel, ...HUNT_PIXELS.map(String)], { encoding: 'utf8', env });
      expect(r.status, r.stderr).toBe(0);
      const lines = r.stdout.trim().split('\n').map((l) => l.split(' '));
      expect(lines.map((l) => l[1])).toEqual(['forward', 'reverse', 'shuffle']);
      for (const [k, s] of SCHEDULES.entries()) expect(lines[k].slice(2).map(Number), s).toEqual(simulate(v, s).bins);
    });
  // ThreadSanitizer's reports, reduced to unordered pairs of source lines in the snippet file.
  const reported = (stderr: string) => {
    const pairs = new Set<string>();
    for (const rep of stderr.split('WARNING: ThreadSanitizer: data race').slice(1)) {
      const lines: number[] = [];
      for (const part of rep.split(/\n\s+(?=(?:Previous )?(?:[Aa]tomic )?(?:[Ww]rite|[Rr]ead) of size)/).slice(1)) {
        const m = /privatised_variants\.cu:(\d+)/.exec(part);
        if (m) lines.push(Number(m[1]));
      }
      if (lines.length >= 2) pairs.add(lines.slice(0, 2).sort((a, b) => a - b).join('-'));
    }
    return pairs;
  };
  const modelled = (v: Variant) => {
    const k = kernelOf(v);
    const at = (s: Site) => k.firstLine + siteLine(k.text, s);
    return new Set(races(v).map((r) => r.sites.map(at).sort((a, b) => a - b).join('-')));
  };
  for (const v of VARIANTS)
    it(`${VARIANT[v].kernel}: the race backend reports no pair the model does not${v === 'plain-shared-update' ? ' (on this 8-thread launch it may miss this kernel\'s race: see below)' : ', and misses none it should catch'}`, (ctx) => {
      if (!raceBuilt) return ctx.skip();
      const r = spawnSync(exe.race, [VARIANT[v].kernel, ...HUNT_PIXELS.map(String)], { encoding: 'utf8', env });
      expect([0, 66]).toContain(r.status);
      const got = reported(r.stderr);
      const want = modelled(v);
      for (const p of got) expect(want.has(p), `unexpected race ${p}\n${r.stderr.slice(0, 2000)}`).toBe(true);
      // The emulator gives each block one happens-before token for all its barriers, so a
      // thread that leaves barrier 1 late also synchronises with threads already at barrier
      // 2, which hides a race between two threads' updates in that interval in most runs.
      // Until that is fixed, plain-shared-update is only required not to report anything
      // the model does not predict.
      if (v !== 'plain-shared-update') expect([...got].sort()).toEqual([...want].sort());
    }, 60_000);
});
