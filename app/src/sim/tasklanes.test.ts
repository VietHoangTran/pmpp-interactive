import { describe, expect, it } from 'vitest';
import { COMPARED, FILLS, HIP_DEFAULT_QUEUES, KERNEL_COUNTS, LAUNCH_RATIOS, compare, issueOrder, simulate, type LanesInput } from './tasklanes';

const base = { k: 8, f: 0.25, d: 1, L: 0.2 };
const run = (o: Partial<LanesInput> & { mode: LanesInput['mode'] }) => simulate({ ...base, ...o });

describe('one stream', () => {
  it('totals k d plus the first launch when L <= d', () => {
    expect(run({ mode: 'stream' }).total).toBeCloseTo(8.2, 12);
    expect(run({ mode: 'stream', L: 1 }).total).toBeCloseTo(9, 12);
    expect(run({ mode: 'stream', k: 3, L: 0.05 }).total).toBeCloseTo(3.05, 12);
  });
  it('never runs two kernels at once, so utilisation is about f', () => {
    const r = run({ mode: 'stream' });
    expect(r.maxConcurrent).toBe(1);
    expect(r.utilisation).toBeCloseTo((8 * 0.25) / 8.2, 12);
    expect(r.idleWhileLaunching).toBeCloseTo(0.2, 12); // only the first launch
  });
  it('when L > d the device waits for the host: total k L + d', () => {
    const r = run({ mode: 'stream', L: 2 });
    expect(r.total).toBeCloseTo(17, 12);
    expect(r.idleWhileLaunching).toBeCloseTo(16 - 7, 12); // kL - (k-1)d
  });
});

describe('a stream each', () => {
  it('with f <= 1/k and enough queues, all overlap: about d plus the launches', () => {
    const r = run({ mode: 'streams', f: 0.1, L: 0.05 });
    expect(r.maxConcurrent).toBe(8);
    expect(r.total).toBeCloseTo(8 * 0.05 + 1, 12);
    expect(r.rows).toEqual(Array.from({ length: 8 }, (_, j) => `stream ${j}`));
  });
  it(`${HIP_DEFAULT_QUEUES} hardware queues cap the overlap at 4 (8 streams)`, () => {
    expect(HIP_DEFAULT_QUEUES).toBe(4);
    const r = run({ mode: 'streams', f: 0.1, L: 0.05, queues: 4 });
    expect(r.maxConcurrent).toBe(4);
    expect(r.total).toBeCloseTo(2.2, 12);
    expect(r.rows).toEqual(['queue 0', 'queue 1', 'queue 2', 'queue 3']);
    // round-robin: stream j on queue j mod 4
    expect(r.kernels.map((x) => x.row)).toEqual([0, 1, 2, 3, 0, 1, 2, 3]);
  });
  it('q-hip-queues: 8 kernels at 10%, no launch cost, ROCm defaults: 2d, against d with a queue per stream', () => {
    expect(run({ mode: 'streams', f: 0.1, L: 0, queues: HIP_DEFAULT_QUEUES }).total).toBe(2);
    expect(run({ mode: 'streams', f: 0.1, L: 0 }).total).toBe(1);
  });
  it('kernels that each fill the device gain nothing from streams', () => {
    const one = run({ mode: 'stream', f: 1 });
    const many = run({ mode: 'streams', f: 1 });
    expect(many.total).toBeCloseTo(one.total, 12);
    expect(many.maxConcurrent).toBe(1);
  });
  it('fills cap the overlap: f = 0.25 lets 4 run at once even with unlimited queues', () => {
    const r = run({ mode: 'streams' });
    expect(r.maxConcurrent).toBe(4);
    expect(r.total).toBeCloseTo(2.8, 12);
  });
  it('when L > d streams do not help: the host is the bottleneck', () => {
    const r = run({ mode: 'streams', f: 0.1, L: 2 });
    expect(r.total).toBeCloseTo(17, 12);
    expect(r.maxConcurrent).toBe(1);
  });
  it('with pairs, each pair shares a stream and the consumer waits for its producer', () => {
    const r = run({ mode: 'streams', pairs: true, f: 0.1, L: 0.05 });
    expect(r.rows.length).toBe(4);
    for (let j = 0; j < 4; j++) {
      expect(r.kernels[2 * j + 1].dependsOn).toBe(2 * j);
      expect(r.kernels[2 * j + 1].start).toBeGreaterThanOrEqual(r.kernels[2 * j].end - 1e-12);
      expect(r.kernels[2 * j + 1].stream).toBe(j);
    }
    expect(r.total).toBeCloseTo(0.05 * 7 + 2, 12); // last producer launched at 7L, then two kernels
  });
});

describe('Metal encoder', () => {
  it('nothing starts before the commit, after all k dispatches are encoded', () => {
    const r = run({ mode: 'encoder' });
    expect(r.commit).toBeCloseTo(1.6, 12);
    expect(Math.min(...r.kernels.map((x) => x.start))).toBeCloseTo(1.6, 12);
    expect(r.host.every((h) => h.kind === 'encode')).toBe(true);
  });
  it('concurrent: independent dispatches overlap up to the fill; serial: one after another', () => {
    const c = run({ mode: 'encoder' });
    expect(c.maxConcurrent).toBe(4);
    expect(c.total).toBeCloseTo(3.6, 12);
    expect(c.barriers).toEqual([]);
    const s = run({ mode: 'encoder', concurrent: false });
    expect(s.maxConcurrent).toBe(1);
    expect(s.total).toBeCloseTo(9.6, 12);
    expect(s.rows).toEqual(['serial encoder']);
  });
  it('pairs in issue order: a barrier before every consumer, each a full join', () => {
    const r = run({ mode: 'encoder', pairs: true });
    expect(r.barriers.length).toBe(4);
    expect(r.total).toBeCloseTo(1.6 + 5, 12); // {0} {1,2} {3,4} {5,6} {7}
    for (const b of r.barriers) for (const x of r.kernels) expect(x.end <= b + 1e-12 || x.start >= b - 1e-12).toBe(true);
  });
  it('pairs encoded producers first: one barrier, two rounds', () => {
    expect(issueOrder({ ...base, mode: 'encoder', pairs: true, grouped: true })).toEqual([0, 2, 4, 6, 1, 3, 5, 7]);
    const r = run({ mode: 'encoder', pairs: true, grouped: true });
    expect(r.barriers).toHaveLength(1);
    expect(r.total).toBeCloseTo(1.6 + 2, 12);
    for (let j = 0; j < 4; j++) expect(r.kernels[2 * j + 1].start).toBeGreaterThanOrEqual(r.kernels[2 * j].end - 1e-12);
  });
  it('rows: each dispatch takes the first row free when it starts', () => {
    const r = run({ mode: 'encoder' });
    expect(r.rows.length).toBe(4);
    expect(new Set(r.kernels.map((x) => x.row)).size).toBe(4);
  });
});

describe('graph', () => {
  it('one launch for the whole sequence, then the kernels back to back', () => {
    const r = run({ mode: 'graph' });
    expect(r.host).toEqual([{ kind: 'graph', i: null, start: 0, end: 0.2 }]);
    expect(r.total).toBeCloseTo(8.2, 12);
  });
  it('when L > d the graph beats every launch-per-kernel mode', () => {
    const all = compare({ ...base, L: 2 });
    const g = all.find((x) => x.key === 'graph')!;
    expect(g.total).toBeCloseTo(10, 12);
    for (const x of all) if (x.key !== 'graph') expect(g.total).toBeLessThan(x.total);
  });
  it('when L is small the graph is no better than one stream, and streams beat both for small f', () => {
    const all = compare({ ...base, f: 0.1, L: 0.05 });
    const t = Object.fromEntries(all.map((x) => [x.key, x.total]));
    expect(t.graph).toBeCloseTo(t.stream, 12);
    expect(t.streams).toBeLessThan(t.graph);
  });
});

describe('invariants', () => {
  const inputs: LanesInput[] = [];
  for (const mode of ['stream', 'streams', 'encoder', 'graph'] as const)
    for (const k of [1, 2, 5, 8, 16])
      for (const f of [0.05, 0.25, 0.3, 0.5, 1])
        for (const L of [0.05, 0.5, 1, 3])
          for (const pairs of [false, true])
            for (const extra of [{}, { queues: 4 }, { concurrent: false }, { grouped: true }])
              inputs.push({ k, f, d: 1, L, mode, pairs, ...extra });
  it('fills never exceed 1, utilisation is at most 1, every kernel runs for d after it was issued', () => {
    for (const inp of inputs) {
      const r = simulate(inp);
      expect(r.utilisation).toBeLessThanOrEqual(1 + 1e-9);
      for (const o of r.occupancy) expect(o.fill).toBeLessThanOrEqual(1 + 1e-9);
      expect(r.kernels).toHaveLength(inp.k);
      for (const x of r.kernels) {
        expect(x.end - x.start).toBeCloseTo(1, 12);
        expect(x.start).toBeGreaterThanOrEqual(Math.min(inp.L, r.hostEnd) - 1e-12);
        if (x.dependsOn !== null) expect(x.start).toBeGreaterThanOrEqual(r.kernels[x.dependsOn].end - 1e-12);
      }
      expect(r.total).toBeCloseTo(Math.max(r.hostEnd, ...r.kernels.map((x) => x.end)), 12);
      expect(r.maxConcurrent).toBeLessThanOrEqual(Math.floor(1 / inp.f + 1e-9));
      // two kernels in one row never overlap
      for (let row = 0; row < r.rows.length; row++) {
        const xs = r.kernels.filter((x) => x.row === row).sort((a, b) => a.start - b.start);
        for (let i = 1; i < xs.length; i++) expect(xs[i].start).toBeGreaterThanOrEqual(xs[i - 1].end - 1e-12);
      }
    }
  });
  it('nothing beats d for the device part, and launch-per-kernel modes cannot end before k L + d... unless graphed', () => {
    for (const inp of inputs) {
      const r = simulate(inp);
      expect(r.total).toBeGreaterThanOrEqual(1 - 1e-12);
      if (inp.mode === 'stream' || inp.mode === 'streams') expect(r.total).toBeGreaterThanOrEqual(inp.k * inp.L + 1 - 1e-12);
      if (inp.mode === 'graph') expect(r.total).toBeCloseTo(inp.L + inp.k, 12);
    }
  });
});

describe('controls', () => {
  it('slider stops cover the lesson cases', () => {
    expect(KERNEL_COUNTS[0]).toBe(1);
    expect(KERNEL_COUNTS[KERNEL_COUNTS.length - 1]).toBe(16);
    expect(FILLS).toContain(0.1);
    expect(FILLS).toContain(0.25);
    expect(FILLS[FILLS.length - 1]).toBe(1);
    for (const r of [0.2, 1, 2]) expect(LAUNCH_RATIOS).toContain(r);
  });
  it('the compared modes are the six the lesson names', () => {
    expect(COMPARED.map((c) => c.key)).toEqual(['stream', 'streams', 'streams4', 'encoder', 'serial', 'graph']);
  });
});
