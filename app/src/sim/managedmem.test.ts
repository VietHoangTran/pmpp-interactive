import { describe, expect, it } from 'vitest';
import {
  DECODE_TOKENS,
  EXPLORE_POLICIES,
  FAULT_PAGE,
  PRESETS,
  cost,
  decodeSweep,
  explore,
  faultStory,
  figLines,
  moved,
  pageList,
  runFig,
  scriptOps,
  simulate,
  steppable,
  warmTraffic,
  type Policy,
  type ScriptOp,
} from './managedmem';

const preset = (id: string) => PRESETS.find((p) => p.id === id)!.script;
const totals = (id: string) =>
  Object.fromEntries(EXPLORE_POLICIES.map((p) => [p, explore(p, preset(id)).counters])) as Record<Policy, ReturnType<typeof explore>['counters']>;

describe('Fig. C.1 under each mode (section C.6)', () => {
  it('the CPU original moves nothing', () => {
    for (const m of ['bulk', 'migrate', 'one'] as Policy[]) expect(moved(runFig('cpu', m).run.counters)).toBe(0);
  });

  it('full mode, kernel touches every page: as many pages cross as with explicit copies (q-same-bytes)', () => {
    const managed = runFig('managed', 'migrate').run.counters;
    const explicit = runFig('explicit', 'migrate').run.counters;
    expect(managed.h2d).toBe(8);
    expect(managed.d2h).toBe(8);
    expect(managed.gpuFaults).toBe(8);
    expect(managed.cpuFaults).toBe(8);
    expect(explicit.h2d).toBe(8);
    expect(explicit.d2h).toBe(8);
    expect(explicit.gpuFaults + explicit.cpuFaults).toBe(0);
  });

  it('full mode moves only the pages the kernel touches; limited mode and explicit copies move all', () => {
    const half = { half: true };
    expect(runFig('managed', 'migrate', half).run.counters).toMatchObject({ h2d: 4, d2h: 4 });
    expect(runFig('managed', 'bulk', half).run.counters).toMatchObject({ h2d: 8, d2h: 8, gpuFaults: 0, cpuFaults: 8 });
    expect(runFig('explicit', 'migrate', half).run.counters).toMatchObject({ h2d: 8, d2h: 8 });
  });

  it('limited mode moves every page at the launch, with no GPU faults', () => {
    const r = runFig('managed', 'bulk');
    const launch = r.run.steps[r.stepOf[3]];
    expect(launch.op.op).toBe('launch');
    expect(launch.counters.h2d).toBe(8);
    expect(launch.events.filter((e) => e.e === 'move' && e.why === 'bulk')).toHaveLength(8);
    expect(r.run.counters.gpuFaults).toBe(0);
  });

  it('only the one-memory mode moves nothing, for the managed port; explicit copies still copy', () => {
    expect(moved(runFig('managed', 'one').run.counters)).toBe(0);
    const ex = runFig('explicit', 'one').run.counters;
    expect(moved(ex)).toBe(0);
    expect(ex.copiedInMemory).toBe(16);
  });

  it('with the synchronise there are no errors in any mode', () => {
    for (const m of ['bulk', 'migrate', 'one'] as Policy[]) {
      expect(runFig('managed', m).run.errors).toEqual([]);
      expect(runFig('managed', m, { half: true }).run.errors).toEqual([]);
    }
  });

  it('delete cudaDeviceSynchronize(): a segmentation fault when limited, a race when full, an ordering error on one memory (q-delete-sync)', () => {
    const lim = runFig('managed', 'bulk', { noSync: true });
    expect(lim.run.errors.map((e) => e.kind)).toEqual(['segfault']);
    expect(lim.run.crashed).toBe(true);
    const use = lim.lines.findIndex((l) => l.role === 'use');
    expect(lim.run.errors[0].step).toBe(lim.stepOf[use]);
    expect(lim.run.steps[lim.stepOf[use + 1]].skipped).toBe(true); // cudaFree never runs

    const full = runFig('managed', 'migrate', { noSync: true });
    expect(full.run.errors.map((e) => e.kind)).toEqual(['race']);
    expect(full.run.errors[0].pages).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(full.run.crashed).toBe(false);
    expect(runFig('managed', 'migrate', { noSync: true, half: true }).run.errors[0].pages).toEqual([0, 1, 2, 3]);

    const one = runFig('managed', 'one', { noSync: true });
    expect(one.run.errors.map((e) => e.kind)).toEqual(['ordering']);
  });

  it('the deleted line stays in the listing but is never stepped', () => {
    const ls = figLines('managed', { noSync: true });
    const i = ls.findIndex((l) => l.role === 'sync');
    expect(ls[i].deleted).toBe(true);
    expect(steppable(ls)).not.toContain(i);
    expect(steppable(figLines('cpu'))).toHaveLength(6); // the blank line is skipped
  });

  it('the four lines Fig. C.1 changes are marked, and only those', () => {
    expect(figLines('managed').filter((l) => l.changed).map((l) => l.text.trim())).toEqual([
      'cudaMallocManaged(&data, N);',
      'qsort_char<<<grid, block>>>(data, N, 1);',
      'cudaDeviceSynchronize();',
      'cudaFree(data);',
    ]);
  });
});

describe('the C.6 questions, from the model', () => {
  it('q-oversub-windows: a limited system refuses managed memory beyond the GPU', () => {
    const r = simulate('bulk', [{ op: 'alloc', buf: 'm', kind: 'managed', pages: 15 }], { capacity: 10 });
    expect(r.errors[0].kind).toBe('oversubscribed');
    const f = simulate('migrate', [
      { op: 'alloc', buf: 'm', kind: 'managed', pages: 15 },
      { op: 'cpu', buf: 'm', write: true, pages: [...Array(15).keys()] },
      { op: 'launch', buf: 'm', reads: [...Array(15).keys()], writes: [] },
    ], { capacity: 10 });
    expect(f.errors).toEqual([]);
    expect(f.counters.evictions).toBe(5);
  });
});

describe('coherence by invalidation (section C.8)', () => {
  it('q-pages-moved: 64 pages written by the CPU, a kernel reads 4: bulk moves 64, fault and migrate 4', () => {
    const script: ScriptOp[] = [{ kind: 'cpu-write', from: 0, to: 63 }, { kind: 'kernel-read', from: 0, to: 3 }];
    const bulk = explore('bulk', script, 64).counters.h2d;
    const mig = explore('migrate', script, 64).counters.h2d;
    expect([bulk, mig]).toEqual([64, 4]);
    expect(bulk - mig).toBe(60);
  });

  it('Fig. C.1 on sixteen pages', () => {
    const t = totals('figC1');
    expect(moved(t.bulk)).toBe(32);
    expect(moved(t.migrate)).toBe(32);
    expect(moved(t.map)).toBe(0);
    expect(t.map.remote).toBe(16);
    expect(moved(t.replicate)).toBe(32);
    expect(t.replicate.invalidations).toBe(16);
  });

  it('one page of sixteen: bulk moves all sixteen, the others one', () => {
    const t = totals('one-page');
    expect(t.bulk.h2d).toBe(16);
    expect(t.migrate.h2d).toBe(1);
    expect(t.map).toMatchObject({ h2d: 0, remote: 1 });
    expect(t.replicate.h2d).toBe(1);
  });

  it('ping-pong on one counter page: migration moves it every turn, mapping never (q-pingpong)', () => {
    const t = totals('ping-pong');
    expect(t.bulk).toMatchObject({ h2d: 19, d2h: 4 });
    expect(t.migrate).toMatchObject({ h2d: 4, d2h: 4, gpuFaults: 4, cpuFaults: 4 });
    expect(t.map).toMatchObject({ h2d: 0, d2h: 0, remote: 4 });
    expect(t.replicate).toMatchObject({ h2d: 4, d2h: 4, invalidations: 8 });
    // Map wins whenever a remote page access costs less than two migrations.
    const best = (m: number, r: number) => EXPLORE_POLICIES.reduce((a, p) => (cost(t[p], m, r) < cost(t[a], m, r) ? p : a));
    expect(best(3, 1)).toBe('map');
    expect(best(1, 1)).toBe('map');
  });

  it('decode loop: weights stay put after the first token unless mapped; read-mostly wins at the default costs', () => {
    const t = totals('decode');
    expect(DECODE_TOKENS).toBe(6);
    expect(t.migrate).toMatchObject({ h2d: 20, d2h: 6 });
    expect(t.bulk).toMatchObject({ h2d: 21, d2h: 6 });
    expect(t.map).toMatchObject({ h2d: 0, d2h: 0, remote: 90 });
    expect(t.replicate).toMatchObject({ h2d: 15, d2h: 6, invalidations: 6 });
    const c = (p: Policy) => cost(t[p], 3, 1);
    expect([c('bulk'), c('migrate'), c('map'), c('replicate')]).toEqual([81, 78, 90, 63]);
  });

  it('a CPU touch while a kernel is active: a segmentation fault when limited, a race otherwise', () => {
    const script: ScriptOp[] = [{ kind: 'cpu-write', from: 0, to: 3 }, { kind: 'kernel-write', from: 0, to: 0 }, { kind: 'cpu-read', from: 0, to: 0 }];
    expect(explore('bulk', script).errors.map((e) => e.kind)).toEqual(['segfault']);
    for (const p of ['migrate', 'map', 'replicate'] as Policy[]) expect(explore(p, script).errors.map((e) => e.kind)).toEqual(['race']);
    // A page the kernel does not use is no race (but still a crash when limited).
    const other: ScriptOp[] = [{ kind: 'cpu-write', from: 0, to: 3 }, { kind: 'kernel-write', from: 0, to: 0 }, { kind: 'cpu-read', from: 2, to: 2 }];
    expect(explore('migrate', other).errors).toEqual([]);
    expect(explore('bulk', other).errors.map((e) => e.kind)).toEqual(['segfault']);
  });

  it('step k of a run is script line k − 1', () => {
    expect(scriptOps(preset('figC1'))).toHaveLength(5);
    expect(explore('migrate', preset('figC1')).steps).toHaveLength(5);
  });
});

describe('when the model does not fit (section C.8)', () => {
  it('q-cliff: at 105% of device memory with LRU, every page migrates every token once warm', () => {
    const tr = decodeSweep(105, 100, 4, 'lru');
    expect(tr[0]).toMatchObject({ h2d: 105, d2h: 5 });
    for (const t of tr.slice(1)) expect(t).toMatchObject({ h2d: 105, d2h: 105, faults: 105, remote: 0 });
  });

  it('keeping a prefix resident turns it into about 5% over the link', () => {
    const w = warmTraffic(105, 100, 'prefix');
    expect(w).toMatchObject({ h2d: 0, d2h: 0, remote: 5 });
    expect(w.remote / 105).toBeCloseTo(0.0476, 3);
  });

  it('at or below 100% nothing moves once warm; just past it, LRU jumps to everything', () => {
    for (const w of [50, 99, 100]) {
      expect(warmTraffic(w, 100, 'lru').h2d).toBe(0);
      expect(warmTraffic(w, 100, 'prefix').remote).toBe(0);
    }
    expect(warmTraffic(101, 100, 'lru').h2d).toBe(101);
    expect(warmTraffic(200, 100, 'lru').h2d).toBe(200);
    expect(warmTraffic(200, 100, 'prefix').remote).toBe(100);
  });

  it('the simulator with a capacity agrees with decodeSweep', () => {
    const W = 12, C = 10;
    const ops = [
      { op: 'alloc' as const, buf: 'm', kind: 'managed' as const, pages: W },
      { op: 'cpu' as const, buf: 'm', write: true, pages: [...Array(W).keys()] },
      ...Array.from({ length: 3 }, () => [{ op: 'launch' as const, buf: 'm', reads: [...Array(W).keys()], writes: [] }, { op: 'sync' as const }]).flat(),
    ];
    const r = simulate('migrate', ops, { capacity: C });
    const perToken = [2, 4, 6].map((k, i) => r.steps[k].counters.h2d - (i ? r.steps[k - 2].counters.h2d : 0));
    expect(perToken).toEqual(decodeSweep(W, C, 3, 'lru').map((t) => t.h2d));
  });
});

describe('the page-fault scene', () => {
  it('one GPU fault, in order: fault, unmap from the CPU, move, map on the GPU, replay', () => {
    const r = faultStory();
    const first = r.steps[2].events.map((e) => e.e);
    expect(first).toEqual(['fault', 'unmap', 'move', 'map', 'replay']);
    expect(r.steps[2].events.every((e) => e.page === FAULT_PAGE)).toBe(true);
    expect(r.steps[2].events[1].proc).toBe('cpu');
    expect(r.steps[2].events[2].dir).toBe('h2d');
    expect(r.steps[3].events.map((e) => e.e)).toEqual(['local']);
    expect(r.steps[5].events.map((e) => e.e)).toEqual(['fault', 'unmap', 'move', 'map', 'replay']);
    expect(r.steps[5].events[1].proc).toBe('gpu');
    expect(r.steps[5].events[2].dir).toBe('d2h');
    expect(r.counters).toMatchObject({ h2d: 1, d2h: 1, gpuFaults: 1, cpuFaults: 1 });
  });

  it('frames record the mappings between the steps of a fault', () => {
    const r = faultStory();
    const f = r.frames.slice(r.steps[2].start, r.steps[2].end).map((x) => x.view.m.pages[FAULT_PAGE]);
    expect(f.map((p) => [p.loc, p.cpuMap, p.gpuMap])).toEqual([
      ['cpu', true, false], // fault: still the CPU's
      ['cpu', false, false], // unmapped from the CPU
      ['gpu', false, false], // arrived, not yet mapped
      ['gpu', false, true], // mapped on the GPU
      ['gpu', false, true], // replayed
    ]);
  });
});

describe('helpers', () => {
  it('page lists read naturally', () => {
    expect(pageList([3])).toBe('page 3');
    expect(pageList([0, 1, 2, 3])).toBe('pages 0–3');
    expect(pageList([0, 2, 5])).toBe('pages 0, 2 and 5');
  });
});
