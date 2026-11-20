import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  CORRECT,
  GARBAGE,
  PRESETS,
  SCHEDULES,
  VARIANT,
  VARIANTS,
  blockOrderOf,
  counterInterleavings,
  kernelBody,
  levelAtomics,
  listingLine,
  makeGraph,
  races,
  siteLine,
  simulate,
  summary,
  variantOf,
  type Options,
  type Preset,
  type Schedule,
  type Shape,
  type Site,
  type Variant,
} from './bfspriv';
import { functionSource } from './toolchain';

const repo = path.resolve(__dirname, '../../..');
const SNIPPET = path.join(repo, 'labs/ch18/snippets/s186_privatised_variants.cu');
const DRIVER = path.join(repo, 'labs/ch18/snippets/check/s186_privatised_check.cu');
const source = fs.readFileSync(SNIPPET, 'utf8');
const kernelOf = (v: Variant) => functionSource(source, VARIANT[v].kernel)!;
const bodyOf = (v: Variant) => {
  const lines = kernelOf(v).text.split('\n');
  const open = lines.findIndex((l) => l.includes('{'));
  return lines.slice(open + 1, lines.length - 1).map((l) => l.replace(/^ {4}/, '')).join('\n');
};
const opts = (o: Partial<Options> = {}): Options => ({ capacity: 2, blocks: 2, threads: 4, preset: 'fanout', ...o });
const run = (sh: Shape, s: Schedule, o: Partial<Options> = {}) => simulate(sh, s, opts(o));

describe('the kernels are the snippet file\'s', () => {
  it('each variant\'s body is exactly the model\'s text', () => {
    for (const v of VARIANTS) expect(bodyOf(v), v).toBe(kernelBody(VARIANT[v].shape));
  });
  it('the corrected line 29 is an atomic store through the same atomic_ref; the book\'s is plain', () => {
    expect(kernelBody(CORRECT)).toContain('numCurrFrontier_s_ref.store(PRIVATE_FRONTIER_CAPACITY,');
    expect(kernelBody({ ...CORRECT, reset: 'plain' })).toContain('                numCurrFrontier_s = PRIVATE_FRONTIER_CAPACITY;');
    expect(kernelBody({ ...CORRECT, reset: 'clamp' })).toContain('    if (numCurrFrontier_s > PRIVATE_FRONTIER_CAPACITY)');
  });
  it('a removed barrier leaves a placeholder in the stepper\'s listing only', () => {
    const shown = kernelBody({ ...CORRECT, barrier2: false }, true).split('\n');
    expect(shown).toContain('// barrier 2 removed');
    expect(shown).toHaveLength(kernelBody(CORRECT).split('\n').length);
  });
  it('the three barriers sit outside every conditional (p. 444)', () => {
    for (const l of kernelBody(CORRECT).split('\n')) if (l.includes('__syncthreads')) expect(l.startsWith('__syncthreads')).toBe(true);
  });
  it('the commit index is start + threadIdx.x + k·blockDim.x: consecutive threads, consecutive addresses (p. 444)', () => {
    const b = kernelBody(CORRECT);
    expect(b).toContain('for (unsigned int currFrontierIdx_s = threadIdx.x;');
    expect(b).toContain('currFrontierIdx_s += blockDim.x) {');
    expect(b).toContain('currFrontierStartIdx + currFrontierIdx_s;');
  });
  it('no line of the stepper\'s listing is wider than 73 characters', () => {
    for (const v of VARIANTS) for (const l of kernelBody(VARIANT[v].shape, true).split('\n')) expect(l.length, l).toBeLessThanOrEqual(73);
  });
  it('variantOf finds each snippet kernel', () => {
    for (const v of VARIANTS) expect(variantOf(VARIANT[v].shape)).toBe(v);
    expect(variantOf({ ...CORRECT, barrier1: false, barrier2: false })).toBeNull();
  });
});

describe('E1: the counter ends at min(N, capacity) in every interleaving (our check)', () => {
  it('for N up to 5 and capacity up to 4, with the atomic store, the plain store (same numbers) and the clamp', () => {
    for (const reset of ['atomic', 'plain', 'clamp'] as const)
      for (let n = 1; n <= 5; n++)
        for (let cap = 1; cap <= 4; cap++) {
          const r = counterInterleavings(n, cap, reset);
          expect(r.finals, `${reset} n=${n} cap=${cap}`).toEqual([Math.min(n, cap)]);
          expect(r.ok).toBe(true);
          expect(r.interleavings).toBeGreaterThan(0);
        }
  });
  it('without any reset the counter would end at N, which is why the reservation needs one', () => {
    // The clamp form clamps after the barrier; before it, the counter holds N.
    expect(counterInterleavings(5, 2, 'clamp').finals).toEqual([2]);
    expect(counterInterleavings(1, 2, 'atomic').interleavings).toBe(1);
    // Whoever adds first stores privately (2 steps); the other overflows (3 steps): 2 x C(4, 1).
    expect(counterInterleavings(2, 1, 'atomic').interleavings).toBe(8);
  });
});

describe('the corrected kernels', () => {
  const sweep: [number, number][] = [[1, 2], [2, 4], [3, 2], [4, 8], [2, 8], [1, 8]];
  it('every preset, order, size and capacity: the public frontier is a permutation of the new vertices, grouped by block', () => {
    for (const v of ['correct', 'clamp', 'plain-store'] as Variant[])
      for (const preset of PRESETS)
        for (const [blocks, threads] of sweep)
          for (let capacity = 1; capacity <= 8; capacity++)
            for (const s of SCHEDULES) {
              const t = run(VARIANT[v].shape, s, { preset, blocks, threads, capacity });
              const tag = `${v} ${preset} ${blocks}x${threads} cap ${capacity} ${s}`;
              expect(t.outcome.kind, tag).toBe('correct');
              expect(t.outcome.grouped, tag).toBe(true);
              expect(t.firstFailure, tag).toBe(-1);
              expect(t.tally.fig1815, tag).toBe(blocks + t.tally.overflows);
              expect(t.tally.fig1812, tag).toBe(makeGraph(preset, blocks, threads).expected.length);
            }
  });
  it('blocks reserve in the order they reach the counter: another block order, another arrangement, the same set', () => {
    const a = run(CORRECT, 'forward', { blocks: 4, threads: 2, capacity: 4 });
    const b = run(CORRECT, 'forward', { blocks: 4, threads: 2, capacity: 4, blockOrder: [2, 0, 3, 1] });
    expect(a.reservations.map((r) => r.block)).toEqual([0, 1, 2, 3]);
    expect(b.reservations.map((r) => r.block)).toEqual([2, 0, 3, 1]);
    const set = (t: typeof a) => [...t.steps[t.steps.length - 1].after.pub.slice(0, t.outcome.size)].sort((x, y) => x - y);
    expect(set(a)).toEqual(set(b));
    expect(a.steps[a.steps.length - 1].after.pub).not.toEqual(b.steps[b.steps.length - 1].after.pub);
    for (const kind of ['in-order', 'reversed', 'random'] as const) expect([...blockOrderOf(kind, 4)].sort()).toEqual([0, 1, 2, 3]);
  });
  it('fan-out, capacity 2, 2 blocks of 4: each block finds 8, keeps 2, sends 6 straight out; 14 global atomics instead of 16', () => {
    const t = run(CORRECT, 'forward');
    expect(t.tally).toMatchObject({ newVertices: 16, fig1812: 16, overflows: 12, reservations: 2, fig1815: 14 });
    expect(t.reservations.map((r) => r.count)).toEqual([2, 2]);
  });
  it('fan-out, capacity 8: no overflow, one global atomic per block', () => {
    const t = run(CORRECT, 'forward', { capacity: 8 });
    expect(t.tally).toMatchObject({ overflows: 0, fig1815: 2, fig1812: 16 });
  });
  it('the commit: pass k of a block writes consecutive addresses', () => {
    const t = run(CORRECT, 'forward', { capacity: 8, blocks: 1, threads: 4 });
    expect(t.commitPasses[0]).toEqual([[0, 1, 2, 3], [4, 5, 6, 7]]);
  });
  it('the shared preset: the compare-and-swap decides which block appends a shared vertex; the set never changes', () => {
    for (const order of [[0, 1, 2], [2, 1, 0], [1, 2, 0]]) {
      const t = run(CORRECT, 'shuffle', { preset: 'shared', blocks: 3, threads: 2, capacity: 2, blockOrder: order });
      expect(t.outcome.kind).toBe('correct');
      expect(t.tally.newVertices).toBe(6);
      expect(t.tally.cas).toBe(12);
    }
  });
});

describe('questions', () => {
  it('q-atomics: 3,000 appends from 24 blocks, no overflow: 3,000 with Fig. 18.12, 24 with Fig. 18.15', () => {
    expect(levelAtomics(3000, 24)).toEqual({ fig1812: 3000, fig1815: 24 });
  });
  it('q-overflow: capacity 8, a block of 4 threads finds 13 (the hub preset): the counter holds 8 and 5 went straight out', () => {
    for (const reset of ['atomic', 'plain'] as const)
      for (const s of SCHEDULES) {
        const t = run({ ...CORRECT, reset }, s, { preset: 'hub', blocks: 1, threads: 4, capacity: 8 });
        expect(t.tally.newVertices).toBe(13);
        expect(t.tally.overflows).toBe(5);
        const b2 = t.steps.find((x) => x.op === 'release' && x.barrier === 2)!;
        expect(b2.after.blocks[0].counter).toBe(8);
      }
  });
  it('q-commit-if: capacity 4 and 2 threads per block lose entries; capacity 2 does not', () => {
    const bad = run(VARIANT['if-commit'].shape, 'forward', { capacity: 4, threads: 2 });
    expect(bad.outcome.kind).toBe('wrong');
    expect(bad.steps.some((s) => s.violation?.kind === 'uncopied')).toBe(true);
    expect(run(VARIANT['if-commit'].shape, 'forward', { capacity: 2, threads: 2 }).outcome.kind).toBe('correct');
    expect(run(VARIANT['if-commit'].shape, 'forward', { capacity: 4, threads: 4 }).outcome.kind).toBe('correct');
  });
});

describe('each defect fails in a stated order', () => {
  it('barrier 2 removed, in order: thread 0 reserves before the others have counted', () => {
    const t = run(VARIANT['no-barrier-2'].shape, 'forward', { capacity: 8 });
    expect(t.steps[t.firstFailure].violation!.kind).toBe('short-reservation');
    expect(t.outcome.kind).not.toBe('correct');
    expect(run(VARIANT['no-barrier-2'].shape, 'reverse', { capacity: 8 }).outcome.kind).toBe('correct');
  });
  it('barrier 3 removed, reversed: a thread reads currFrontierStartIdx before thread 0 writes it, and stores far out of range', () => {
    const t = run(VARIANT['no-barrier-3'].shape, 'reverse', { capacity: 8 });
    expect(t.outcome.kind).toBe('crash');
    expect(t.steps[t.firstFailure].violation!.message).toContain('currFrontierStartIdx has not been written yet');
    expect(run(VARIANT['no-barrier-3'].shape, 'forward', { capacity: 8 }).outcome.kind).toBe('correct');
  });
  it('barrier 1 removed: the emulator\'s 0xCD leftover is far above capacity, so early appends overflow and nothing is lost; a small leftover loses entries', () => {
    for (const s of SCHEDULES) expect(run(VARIANT['no-barrier-1'].shape, s, { capacity: 4 }).outcome.kind, s).toBe('correct');
    const t = run(VARIANT['no-barrier-1'].shape, 'reverse', { capacity: 4, leftover: 1 });
    expect(t.outcome.kind).toBe('wrong');
    expect(['late-zero', 'private-overwrite']).toContain(t.steps[t.firstFailure].violation!.kind);
    expect(run(VARIANT['no-barrier-1'].shape, 'forward', { capacity: 4, leftover: 1 }).outcome.kind).toBe('correct');
  });
  it('no overflow check: more discoveries than capacity write past the private queue', () => {
    const t = run(VARIANT['no-overflow-check'].shape, 'forward');
    expect(t.outcome.kind).toBe('crash');
    expect(t.outcome.crash).toContain('outside the 2-entry private queue');
    expect(run(VARIANT['no-overflow-check'].shape, 'forward', { capacity: 8 }).outcome.kind).toBe('correct');
  });
  it('summary lines', () => {
    expect(summary(run(CORRECT, 'forward').outcome)).toBe('correct');
    expect(summary(run(VARIANT['if-commit'].shape, 'forward', { capacity: 4, threads: 2 }).outcome)).toMatch(/lost/);
  });
});

describe('the race check', () => {
  const o = opts();
  it('the corrected kernels race nowhere; the book\'s plain store races with the fetch_add, and the emulator\'s orders all pass it', () => {
    expect(races(CORRECT, o)).toEqual([]);
    expect(races(VARIANT.clamp.shape, o)).toEqual([]);
    const r = races(VARIANT['plain-store'].shape, o).map((x) => x.sites.join('|'));
    expect(r).toContain('count|reset');
    for (const s of SCHEDULES) expect(run(VARIANT['plain-store'].shape, s).outcome.kind).toBe('correct');
  });
  it('the plain store does not race when nothing overflows', () => {
    expect(races(VARIANT['plain-store'].shape, opts({ capacity: 8 }))).toEqual([]);
  });
  it('removed barriers race on the lines they used to order', () => {
    expect(races(VARIANT['no-barrier-1'].shape, o).map((x) => x.sites.join('|'))).toContain('zero|count');
    expect(races(VARIANT['no-barrier-2'].shape, o).map((x) => x.sites.join('|'))).toContain('count|reserve-read');
    expect(races(VARIANT['no-barrier-3'].shape, opts({ capacity: 8 })).map((x) => x.sites.join('|'))).toContain('reserve-write|commit-read-start');
  });
  it('the race panel can name both lines of the plain store\'s race in the listing', () => {
    const sh = VARIANT['plain-store'].shape;
    const lines = kernelBody(sh, true).split('\n');
    expect(lines[listingLine(sh, 'reset') - 1].trim()).toBe('numCurrFrontier_s = PRIVATE_FRONTIER_CAPACITY;');
    expect(lines[listingLine(sh, 'count') - 1].trim()).toBe('unsigned int currFrontierIdx_s = numCurrFrontier_s_ref');
  });
  it('every racing site can be found in its kernel', () => {
    for (const v of VARIANTS)
      for (const r of races(VARIANT[v].shape, opts({ capacity: 4 }))) for (const s of r.sites) expect(siteLine(kernelOf(v).text, s), `${v} ${s}`).toBeGreaterThan(0);
  });
  it('the stepper\'s lines point at the statement each step runs', () => {
    const want: Record<string, RegExp> = {
      zero: /numCurrFrontier_s = 0/, load: /prevFrontier\[i\]/, visit: /visitVertexAtomically/, count: /currFrontierIdx_s = numCurrFrontier_s_ref/,
      store: /currFrontier_s\[currFrontierIdx_s\] = neighbor/, reset: /numCurrFrontier_s(_ref\.store\(| = PRIVATE)/, append: /numCurrFrontier_ref|currFrontier\[currFrontierIdx\] = neighbor|\.fetch_add/,
      reserve: /currFrontierStartIdx = |\.fetch_add\(numCurrFrontier_s|numCurrFrontier_s > PRIVATE|numCurrFrontier_s = PRIVATE/, commit: /currFrontier\[currFrontier(Idx|StartIdx)/,
      skip: /numCurrFrontier_s/, arrive: /__syncthreads/, release: /__syncthreads/,
    };
    for (const sh of [CORRECT, VARIANT.clamp.shape, VARIANT['plain-store'].shape, VARIANT['if-commit'].shape, { ...CORRECT, barrier1: false }]) {
      const lines = kernelBody(sh, true).split('\n');
      for (const st of run(sh, 'forward').steps) {
        expect(st.lines.length, st.op).toBeGreaterThan(0);
        expect(st.lines.map((n) => lines[n - 1]).join('\n'), `${st.op}: ${st.text}`).toMatch(want[st.op]);
      }
    }
  });
  it('shared memory starts as 0xCD bytes', () => {
    expect(run(CORRECT, 'forward').initial.blocks[0].counter).toBe(GARBAGE);
  });
});

// The snippet file on the course's emulator, built as the lab runner builds the emu and race
// backends (labs/cmake/Backends.cmake): emu with AddressSanitizer and the three thread orders,
// race with ThreadSanitizer. One process per run, because the defects that write out of range
// stop the process. Skipped when no C++ compiler works.
const compiler = ['c++', 'clang++', 'g++'].find((c) => spawnSync(c, ['--version']).status === 0);
const clang = ['clang++'].find((c) => spawnSync(c, ['--version']).status === 0);
const inc = path.join(repo, 'labs/common/include');

describe.skipIf(!clang)('the CUDA front-end check (clang, stub headers, as labs/pmpp.py runs it)', () => {
  it('parses the snippet, and the snippet with its driver', () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch18-priv-fe-'));
    const both = path.join(dir, 'both.cu');
    fs.writeFileSync(both, source + '\n' + fs.readFileSync(DRIVER, 'utf8'));
    for (const f of [SNIPPET, both]) {
      const r = spawnSync(clang!, ['-x', 'cuda', '--cuda-host-only', '-nocudainc', '-nocudalib', '-fsyntax-only', '-std=c++17',
        '-I', inc, '-I', path.join(inc, 'pmpp/emu/shim'), '-include', 'pmpp/check/cuda_stub.h', f], { encoding: 'utf8' });
      if (r.status !== 0 && /cannot parse|unknown argument|error: language not recognized/.test(r.stderr)) return; // not a CUDA-capable clang
      expect(r.status, f + '\n' + r.stderr).toBe(0);
    }
  }, 60_000);
});

describe.skipIf(!compiler)('the snippets on the emulator agree with the model', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ch18-priv-'));
  const exe = { emu: path.join(dir, 'emu'), race: path.join(dir, 'race') };
  const env = {
    ...process.env,
    ASAN_OPTIONS: 'detect_leaks=0',
    TSAN_OPTIONS: 'abort_on_error=0:exitcode=66:halt_on_error=0:report_thread_leaks=0',
  };
  let raceBuilt = false;
  it('builds', () => {
    const cu = '#line 1 "s186_privatised_variants.cu"\n' + source + '\n#line 1 "check/s186_privatised_check.cu"\n' + fs.readFileSync(DRIVER, 'utf8');
    fs.writeFileSync(path.join(dir, 'all.cu'), cu);
    const t = spawnSync('python3', [path.join(repo, 'labs/tools/cu2emu.py'), path.join(dir, 'all.cu'), path.join(dir, 'all.cpp')], { encoding: 'utf8' });
    expect(t.status, t.stdout + t.stderr).toBe(0);
    const common = ['-std=c++17', '-O1', '-g', '-fno-omit-frame-pointer', '-pthread', '-I', inc, '-I', path.join(inc, 'pmpp/emu/shim'), '-include', 'pmpp/emu/cuda_emu.h', path.join(dir, 'all.cpp')];
    const c = spawnSync(compiler!, ['-fsanitize=address', '-DPMPP_BACKEND_EMU', ...common, '-o', exe.emu], { encoding: 'utf8' });
    expect(c.status, c.stderr).toBe(0);
    const r = spawnSync(compiler!, ['-fsanitize=thread', '-DPMPP_BACKEND_RACE', ...common, '-o', exe.race], { encoding: 'utf8' });
    raceBuilt = r.status === 0; // ThreadSanitizer is not available everywhere
  }, 240_000);

  const configs: [Preset, number, number, number][] = [
    ['fanout', 2, 2, 4], ['fanout', 4, 2, 4], ['fanout', 8, 1, 4], ['fanout', 3, 3, 2],
    ['shared', 2, 3, 2], ['shared', 1, 2, 4],
    ['hub', 8, 1, 4], ['hub', 4, 2, 4], ['hub', 6, 2, 8],
  ];
  for (const v of VARIANTS)
    it(`${VARIANT[v].kernel}: emu's three orders give the model's public frontier, or stop where the model stops`, () => {
      for (const [preset, capacity, blocks, threads] of configs)
        for (const s of SCHEDULES) {
          const tag = `${preset} cap ${capacity} ${blocks}x${threads} ${s}`;
          const r = spawnSync(exe.emu, [VARIANT[v].kernel, s, String(capacity), preset, String(blocks), String(threads)], { encoding: 'utf8', env });
          const m = simulate(VARIANT[v].shape, s, { preset, capacity, blocks, threads });
          if (m.outcome.kind === 'crash') {
            expect(r.status, `${tag}: expected AddressSanitizer to stop the run\n${r.stdout}`).not.toBe(0);
            expect(r.stderr, tag).toMatch(/AddressSanitizer/);
          } else {
            expect(r.status, `${tag}\n${r.stderr.slice(0, 1500)}`).toBe(0);
            const got = r.stdout.trim().split(' ');
            expect(got.slice(0, 2)).toEqual([VARIANT[v].kernel, s]);
            const last = m.steps[m.steps.length - 1].after;
            expect(got.slice(2).map(Number), tag).toEqual([last.count, ...last.pub]);
          }
        }
    }, 120_000);

  // ThreadSanitizer's reports, reduced to unordered pairs of source lines in the snippet file.
  const reported = (stderr: string) => {
    const pairs = new Set<string>();
    for (const rep of stderr.split('WARNING: ThreadSanitizer: data race').slice(1)) {
      const lines: number[] = [];
      for (const part of rep.split(/\n\s+(?=(?:Previous )?(?:[Aa]tomic )?(?:[Ww]rite|[Rr]ead) of size)/).slice(1)) {
        const m = /s186_privatised_variants\.cu:(\d+)/.exec(part);
        if (m) lines.push(Number(m[1]));
      }
      if (lines.length >= 2) pairs.add(lines.slice(0, 2).sort((a, b) => a - b).join('-'));
    }
    return pairs;
  };
  // A statement broken over two lines may be reported on either: ThreadSanitizer names the
  // line of the access, which for `x = ref\n.fetch_add(...)` is the second.
  const possible = (v: Variant, o: Options) => {
    const k = kernelOf(v);
    const at = (s: Site) => k.firstLine + siteLine(k.text, s);
    const sites = races(VARIANT[v].shape, o).map((r) => r.sites as Site[]);
    // A short reservation makes block ranges overlap; which public slots collide depends on
    // the schedule that actually ran, which need not be one of the emulator's three.
    if (v === 'no-barrier-2')
      sites.push(['commit-write', 'commit-write'], ['overflow-store', 'commit-write'], ['overflow-store', 'overflow-store']);
    const pairs = new Set<string>();
    for (const [sa, sb] of sites) {
      const [a, b] = [at(sa), at(sb)];
      for (const x of [a, a + 1]) for (const y of [b, b + 1]) pairs.add([x, y].sort((p, q) => p - q).join('-'));
    }
    return pairs;
  };
  // Only the variants that cannot write out of range: the race flavour has no bounds checks.
  for (const v of ['correct', 'clamp', 'plain-store', 'no-barrier-2'] as Variant[])
    it(`${VARIANT[v].kernel}: the race flavour reports nothing the model does not`, (ctx) => {
      if (!raceBuilt) return ctx.skip();
      for (const o of [opts(), opts({ preset: 'hub', capacity: 4, blocks: 2, threads: 4 })]) {
        const want = possible(v, o);
        for (let rep = 0; rep < 3; rep++) {
          const r = spawnSync(exe.race, [VARIANT[v].kernel, 'once', String(o.capacity), o.preset, String(o.blocks), String(o.threads)], { encoding: 'utf8', env });
          expect([0, 66]).toContain(r.status);
          for (const p of reported(r.stderr)) expect(want.has(p), `unexpected race ${p}\n${r.stderr.slice(0, 2000)}`).toBe(true);
          if (v === 'correct' || v === 'clamp') expect(r.status, r.stderr.slice(0, 2000)).toBe(0);
        }
      }
      // Not required here: that the race flavour catches the plain store in a given run. It
      // sees only the schedules that occur (labs/README.md). See the next test.
    }, 180_000);
  it('the race flavour does catch the book\'s plain store: in at least one of five runs it names the fetch_add and the store', (ctx) => {
    if (!raceBuilt) return ctx.skip();
    const k = kernelOf('plain-store');
    const store = k.firstLine + siteLine(k.text, 'reset');
    const add = k.firstLine + siteLine(k.text, 'count') + 1; // the `.fetch_add(` line
    let caught = false;
    for (let rep = 0; rep < 5 && !caught; rep++) {
      const r = spawnSync(exe.race, ['bfsPlainStore', 'once', '2', 'fanout', '2', '4'], { encoding: 'utf8', env });
      caught = reported(r.stderr).has([add, store].sort((a, b) => a - b).join('-'));
    }
    expect(caught).toBe(true);
  }, 120_000);
});
