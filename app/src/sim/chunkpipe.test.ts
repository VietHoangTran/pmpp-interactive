import { describe, expect, it } from 'vitest';
import {
  CUDA_QUEUES,
  DEFAULTS,
  FOUR_CHUNKS,
  HALO_BOOK,
  HIP_QUEUES,
  LINES,
  ONE_ENGINE,
  PRESETS,
  Q_BOTTLENECK,
  START,
  bestChunks,
  chunkShares,
  fromReports,
  haloLanes,
  parseLabRows,
  perChunk,
  pinnedScene,
  sharingGap,
  simulate,
  stagingPlan,
  stepAt,
  stepOp,
  sweep,
  type PipeConfig,
} from './chunkpipe';

const run = (o: Partial<PipeConfig>) => simulate({ ...DEFAULTS, ...o });
const preset = (id: string) => PRESETS.find((p) => p.id === id)!.cfg;

// A small deterministic generator, so the random cases are the same on every run.
function rng(seed: number) {
  let h = seed >>> 0;
  return () => {
    h = (h + 0x6d2b79f5) >>> 0;
    let t = Math.imul(h ^ (h >>> 15), h | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe('the numbers the lesson states', () => {
  it('p-pipeline: 8 chunks of 1, 2, 1, a stream each, two engines: 18 against 32 serial', () => {
    const s = simulate(preset('p-pipeline'));
    expect(s.serial).toBe(32);
    expect(s.total).toBe(18);
    expect(s.e1!.holds).toBe(true);
    expect(s.e1!.value).toBe(18);
    expect(s.bottleneck).toBe('compute');
  });
  it('q-bottleneck: totals 8, 12, 4 in 4 chunks: 15 against 24', () => {
    const s = simulate(Q_BOTTLENECK);
    expect(perChunk(Q_BOTTLENECK, 4)).toEqual({ a: 2, b: 3, c: 1 });
    expect(s.total).toBe(15);
    expect(s.serial).toBe(24);
  });
  it('p-oneengine (E2): 4 chunks of 1, 2, 1 on one shared engine: 16 chunk by chunk, 10 breadth first', () => {
    const depth = simulate(ONE_ENGINE);
    expect(depth.total).toBe(16);
    expect(depth.serial).toBe(16);
    expect(simulate({ ...ONE_ENGINE, order: 'breadth' }).total).toBe(10);
    // with two engines the order does not matter here, and both give E1's 10 (E1's example)
    expect(simulate({ ...ONE_ENGINE, engines: 2 }).total).toBe(10);
    expect(simulate({ ...ONE_ENGINE, engines: 2, order: 'breadth' }).total).toBe(10);
  });
  it("E2's second case: 8 chunks of 1, 1, 1 on one engine: 24 depth first, 16 breadth first", () => {
    const c = { ...ONE_ENGINE, a: 8, b: 8, c: 8, n: 8, streams: 8 };
    expect(simulate(c).total).toBe(24);
    expect(simulate({ ...c, order: 'breadth' }).total).toBe(16);
  });
  it('q-twostreams: 12 chunks of 1, 1, 1: 19 with 2 streams, 14 with 3', () => {
    const c = { ...DEFAULTS, a: 12, b: 12, c: 12, n: 12 };
    expect(simulate({ ...c, streams: 2 }).total).toBe(19);
    expect(simulate({ ...c, streams: 3 }).total).toBe(14);
    expect(simulate({ ...c, streams: 2 }).e1!.holds).toBe(false);
    expect(simulate({ ...c, streams: 3 }).e1!.holds).toBe(true);
    expect(simulate({ ...c, streams: 1 }).total).toBe(36);
  });
  it('the scene: four chunks of 1, 2, 1 take 16 (the serial time) pageable and 10 pinned', () => {
    const sc = pinnedScene();
    expect(sc.pageable.total).toBe(16);
    expect(sc.pageable.serial).toBe(16);
    expect(sc.pinned.total).toBe(10);
    expect(sc.frames.length).toBe(sc.pages);
    expect(new Set([...sc.frames, sc.moved.to]).size).toBe(sc.pages + 1); // distinct frames, the new one free
    expect(Math.max(...sc.frames, sc.moved.to)).toBeLessThan(sc.frameCount);
  });
  it("the stepper's starting case: 4 chunks of 3, 4, 2: 21 against 36, unlike the prediction's 18", () => {
    const s = simulate(START);
    expect(s.total).toBe(21);
    expect(s.serial).toBe(36);
  });
  it('every preset matches the closed form whose conditions it meets', () => {
    for (const p of PRESETS) {
      const s = simulate(p.cfg);
      expect(s.e1!.holds, p.id).toBe(true);
      expect(s.e1!.matches, p.id).toBe(true);
    }
    const bp = simulate(preset('best-practices'));
    expect(bp.nvidia!.holds).toBe(true);
    expect(bp.total).toBe(18); // tE + tT/nStreams = 16 + 8/4
    expect(bp.nvidia!.value).toBe(18);
    expect(bp.ops.some((o) => o.kind === 'out')).toBe(false);
  });
});

describe('one stream gives no overlap', () => {
  it('S = 1 equals the serial time, whatever the chunk count', () => {
    for (const n of [1, 2, 8, 64])
      for (const engines of [1, 2] as const) {
        const s = run({ streams: 1, n, engines });
        expect(s.total).toBeCloseTo(32, 6);
        expect(s.speedup).toBeCloseTo(1, 6);
      }
  });
  it('and is slightly worse with a per-operation cost: 3·n·o more', () => {
    const s = run({ streams: 1, n: 8, overhead: 0.25 });
    expect(s.total).toBeCloseTo(32 + 3 * 8 * 0.25, 6);
    expect(s.serial).toBeCloseTo(32 + 3 * 0.25, 6);
    expect(s.speedup).toBeLessThan(1);
  });
});

describe('E1: a + b + c + (n - 1) max(a, b, c) whenever its conditions hold', () => {
  it('over random cases', () => {
    const r = rng(2311);
    let held = 0;
    let breadthHeld = 0;
    let queuesHeld = 0;
    for (let i = 0; i < 4000; i++) {
      const cfg: PipeConfig = {
        ...DEFAULTS,
        a: Math.floor(r() * 20) + (r() < 0.1 ? 0 : 1),
        b: Math.floor(r() * 20) + 1,
        c: r() < 0.15 ? 0 : Math.floor(r() * 20) + 1,
        n: 1 + Math.floor(r() * 24),
        streams: 1 + Math.floor(r() * 16),
        engines: r() < 0.8 ? 2 : 1,
        order: r() < 0.6 ? 'depth' : 'breadth',
        overhead: r() < 0.5 ? 0 : Math.floor(r() * 8) / 4,
        queues: [Infinity, CUDA_QUEUES, HIP_QUEUES][Math.floor(r() * 3)],
        items: r() < 0.2 ? Math.floor(r() * 60) * 4 + 4 : undefined,
      };
      const s = simulate(cfg);
      if (!s.e1 || !s.e1.holds) continue;
      held++;
      if (cfg.order === 'breadth') breadthHeld++;
      if (isFinite(cfg.queues) && cfg.streams > cfg.queues) queuesHeld++;
      expect(s.total, JSON.stringify(cfg)).toBeCloseTo(s.e1.value, 6);
      // the bubble is the fill and drain: (a + b + c) - max per chunk (E1)
      const p = perChunk(cfg, s.shares.length);
      expect(s.bubble, JSON.stringify(cfg)).toBeCloseTo(p.a + p.b + p.c - Math.max(p.a, p.b, p.c), 6);
    }
    expect(held).toBeGreaterThan(500);
    expect(breadthHeld).toBeGreaterThan(50);
    expect(queuesHeld).toBeGreaterThan(20);
  });
  it('with fixed per-chunk times the bubble does not shrink with n, only its share of the total does (E1)', () => {
    const t = [4, 8, 16].map((n) => run({ a: 1 * n, b: 2 * n, c: 1 * n, n, streams: 16 }));
    expect(t.map((s) => s.bubble)).toEqual([2, 2, 2]);
    const share = t.map((s) => s.bubble / s.total);
    expect(share[0]).toBeGreaterThan(share[1]);
    expect(share[1]).toBeGreaterThan(share[2]);
  });
  it('with fixed totals, as in the stepper, more chunks shrink the bubble itself', () => {
    const t = [4, 8, 16].map((n) => run({ a: 12, b: 24, c: 12, n, streams: 16 }));
    expect(t.map((s) => s.bubble)).toEqual([6, 3, 1.5]);
  });
  it('the sharing gap: S streams, Q queues', () => {
    expect(sharingGap({ ...DEFAULTS, streams: 8, queues: Infinity }, 8)).toBe(Infinity);
    expect(sharingGap({ ...DEFAULTS, streams: 3, queues: Infinity }, 12)).toBe(3);
    expect(sharingGap({ ...DEFAULTS, streams: 8, queues: 4 }, 8)).toBe(4);
    expect(sharingGap({ ...DEFAULTS, streams: 6, queues: 4 }, 12)).toBe(2); // streams 0 and 4 share queue 0
  });
});

describe("NVIDIA's rough estimate (Best Practices 10.1.2)", () => {
  it('tE + tT/nStreams or tT + tE/nStreams, with c = 0 and a stream per chunk', () => {
    const r = rng(77);
    for (let i = 0; i < 500; i++) {
      const n = 1 + Math.floor(r() * 32);
      const a = 1 + Math.floor(r() * 40);
      const b = 1 + Math.floor(r() * 40);
      const s = run({ a, b, c: 0, n, streams: Math.min(16, n), queues: Infinity });
      if (n > 16 && !s.nvidia!.holds) continue;
      expect(s.nvidia!.holds).toBe(true);
      expect(s.total).toBeCloseTo(b >= a ? b + a / n : a + b / n, 6);
    }
  });
  it('does not claim to hold with a copy back or a per-operation cost', () => {
    expect(run({ c: 4 }).nvidia!.holds).toBe(false);
    expect(run({ c: 0, overhead: 0.5 }).nvidia!.holds).toBe(false);
  });
});

describe('copy engines and issue order (E2)', () => {
  it('the bottleneck formula a + b + c + (n - 1) max(a + c, b) is not what one engine gives depth first', () => {
    const s = simulate(ONE_ENGINE);
    expect(s.total).not.toBe(1 + 2 + 1 + 3 * Math.max(2, 2));
    expect(s.e1!.holds).toBe(false);
  });
  it('explains the one-engine wait: queued behind a copy-out that waits for a kernel', () => {
    const s = simulate(ONE_ENGINE);
    const in1 = s.ops.find((o) => o.kind === 'in' && o.chunk === 1)!;
    expect(in1.wait!.cause).toBe('engine');
    expect(in1.wait!.text).toContain("chunk 0's copy-out, queued ahead of it");
    expect(in1.wait!.text).toContain("it waits for chunk 0's kernel");
  });
});

describe('streams, queues and waits', () => {
  it("chunk 2's copy-in waits for its stream: the design's example", () => {
    const s = run({ a: 12, b: 12, c: 12, n: 12, streams: 2 });
    const in2 = s.ops.find((o) => o.kind === 'in' && o.chunk === 2)!;
    expect(in2.wait).toMatchObject({ at: 2, cause: 'stream' });
    expect(in2.wait!.text).toBe("Chunk 2's copy-in waits: stream 0 is still copying out chunk 0.");
    expect(in2.start).toBe(3);
  });
  it('a kernel waiting for the GPU is told so in plain words', () => {
    const s = simulate(START);
    const k3 = s.ops.find((o) => o.kind === 'kernel' && o.chunk === 3)!;
    expect(k3.wait!.text).toBe("Chunk 3's kernel waits: the GPU runs one kernel at a time, and it is still running chunk 2's kernel.");
  });
  it('more streams than hardware queues serialise unrelated streams when issued breadth first', () => {
    const c = { ...DEFAULTS, a: 8, b: 8, c: 8, n: 8, streams: 8, order: 'breadth' as const };
    expect(simulate(c).total).toBe(10);
    const q4 = simulate({ ...c, queues: HIP_QUEUES });
    expect(q4.total).toBe(13);
    expect(q4.e1!.holds).toBe(false);
    const out0 = q4.ops.find((o) => o.kind === 'out' && o.chunk === 0)!;
    expect(out0.wait!.cause).toBe('queue');
    expect(out0.wait!.text).toContain('hardware queue 0 also holds stream 4');
    // depth first, four queues are as good as four streams, which is enough here
    expect(simulate({ ...c, order: 'depth', queues: HIP_QUEUES }).total).toBe(10);
    // CUDA's 8 queues bind only past 8 streams
    const c16 = { ...c, n: 16, a: 16, b: 16, c: 16, streams: 16 };
    expect(simulate(c16).total).toBe(18);
    expect(simulate({ ...c16, queues: CUDA_QUEUES }).total).toBeGreaterThan(18);
    expect(simulate({ ...c, queues: CUDA_QUEUES }).total).toBe(10);
  });
  it('maps the j-th chunk to stream j mod S and to queue (j mod S) mod Q', () => {
    const s = run({ n: 8, streams: 3, queues: 2 });
    const ins = s.ops.filter((o) => o.kind === 'in');
    expect(ins.map((o) => o.stream)).toEqual([0, 1, 2, 0, 1, 2, 0, 1]);
    expect(ins.map((o) => o.queue)).toEqual([0, 1, 0, 0, 1, 0, 0, 1]);
  });
  it('records the snippet lines that issued each operation', () => {
    const d = run({ n: 2 });
    expect(d.ops.map((o) => o.lines)).toEqual([
      [LINES.depth.in, LINES.api.in], [LINES.depth.kernel, LINES.api.kernel], [LINES.depth.out, LINES.api.out],
      [LINES.depth.in, LINES.api.in], [LINES.depth.kernel, LINES.api.kernel], [LINES.depth.out, LINES.api.out],
    ]);
    const b = run({ n: 2, order: 'breadth' });
    expect(b.ops.map((o) => o.lines[0])).toEqual([56, 56, 57, 57, 58, 58]);
    expect(b.ops.map((o) => `${o.kind}${o.chunk}`)).toEqual(['in0', 'in1', 'kernel0', 'kernel1', 'out0', 'out1']);
  });
});

describe('per-operation cost (E3) gives an optimum chunk count', () => {
  it('more chunks help, then hurt', () => {
    const cfg = { ...DEFAULTS, a: 12, b: 12, c: 12, streams: 16, overhead: 0.25 };
    const best = bestChunks(cfg);
    expect(best.n).toBeGreaterThan(2);
    expect(best.n).toBeLessThan(64);
    const [one, sixteen, last] = sweep(cfg, [1, 16, 64]);
    expect(best.total).toBeLessThan(one.total);
    expect(best.total).toBeLessThan(last.total);
    expect(last.total).toBeGreaterThan(sixteen.total);
  });
  it('with one stream every chunk count ties, and the tie goes to n = 1', () => {
    expect(bestChunks({ ...DEFAULTS, a: 12, b: 12, c: 12, streams: 1 }).n).toBe(1);
  });
  it('without the cost, more chunks never hurt when streams are plenty', () => {
    const totals = sweep({ ...DEFAULTS, a: 12, b: 12, c: 12, streams: 16 }, [1, 2, 4, 8, 16, 32, 64]).map((x) => x.total);
    for (let i = 1; i < totals.length; i++) expect(totals[i]).toBeLessThanOrEqual(totals[i - 1] + 1e-9);
  });
});

describe('pageable host memory', () => {
  it('equals the serial time and fills the host lane, depth first', () => {
    const r = rng(5);
    for (let i = 0; i < 200; i++) {
      const cfg: PipeConfig = {
        ...DEFAULTS, host: 'pageable', a: 1 + Math.floor(r() * 10), b: 1 + Math.floor(r() * 10), c: 1 + Math.floor(r() * 10),
        n: 1 + Math.floor(r() * 16), streams: 1 + Math.floor(r() * 16), engines: r() < 0.5 ? 1 : 2,
      };
      const s = simulate(cfg);
      expect(s.total).toBeCloseTo(s.serial, 6);
      expect(s.busy.host).toBeCloseTo(s.total, 6);
      expect(s.e1!.holds).toBe(false);
    }
  });
  it('blocks the host first for the stream, then for the copy', () => {
    const s = simulate({ ...FOUR_CHUNKS, host: 'pageable' });
    const kinds = s.host.map((h) => h.kind);
    expect(kinds.slice(0, 3)).toEqual(['staging', 'blocked', 'staging']);
    const out0 = s.ops.find((o) => o.kind === 'out' && o.chunk === 0)!;
    expect(s.host[1]).toMatchObject({ from: 1, to: 3, op: out0.id });
    expect(s.host[1].text).toContain("stream 0 to finish chunk 0's kernel");
    // the next chunk's kernel is queued late: chunk 1's kernel is issued only after chunk 1's copy-in
    const k1 = s.ops.find((o) => o.kind === 'kernel' && o.chunk === 1)!;
    expect(k1.issue).toBe(5);
  });
});

describe('Metal: K buffers in flight', () => {
  it('K = 1 equals the serial time', () => {
    for (const n of [1, 3, 8, 64]) {
      const s = run({ view: 'metal', streams: 1, n });
      expect(s.total).toBeCloseTo(32, 6);
    }
  });
  it('K >= 2 is at least n max(fill + drain, kernel), and overlaps', () => {
    const r = rng(9);
    for (let i = 0; i < 400; i++) {
      const cfg: PipeConfig = {
        ...DEFAULTS, view: 'metal', a: Math.floor(r() * 10) + 1, b: Math.floor(r() * 10) + 1, c: Math.floor(r() * 10),
        n: 1 + Math.floor(r() * 32), streams: 2 + Math.floor(r() * 6), overhead: r() < 0.5 ? 0 : 0.25,
      };
      const s = simulate(cfg);
      expect(s.total + 1e-9).toBeGreaterThanOrEqual(s.lowerBound!);
      expect(s.total).toBeLessThanOrEqual(s.serial + cfg.n * 3 * cfg.overhead + 1e-9);
    }
    const s = run({ view: 'metal', a: 8, b: 16, c: 8, n: 8, streams: 2 });
    expect(s.total).toBeLessThan(s.serial);
    expect(s.lanes).toEqual(['cpu', 'gpu']);
  });
  it('drains a slot before refilling it, waiting in waitUntilCompleted', () => {
    const s = run({ view: 'metal', a: 4, b: 8, c: 4, n: 4, streams: 2 });
    const cpu = s.ops.filter((o) => o.lane === 'cpu').map((o) => `${o.kind}${o.chunk}`);
    expect(cpu).toEqual(['in0', 'in1', 'out0', 'in2', 'out1', 'in3', 'out2', 'out3']);
    expect(s.host.every((h) => h.kind === 'waiting' && h.text.includes('waitUntilCompleted'))).toBe(true);
    expect(s.ops.filter((o) => o.tail).map((o) => o.chunk)).toEqual([2, 3]);
    expect(s.ops.find((o) => o.kind === 'in')!.lines).toEqual([LINES.metal.fill]);
    expect(s.ops.find((o) => o.kind === 'kernel')!.lines).toEqual([LINES.metal.commit]);
  });
});

describe('chunks', () => {
  it('equal by default; the snippet and lab split with an element count, empty chunks skipped', () => {
    expect(chunkShares(4)).toEqual([0.25, 0.25, 0.25, 0.25]);
    expect(chunkShares(4, 10)).toEqual([0.2, 0.3, 0.2, 0.3]);
    expect(chunkShares(8, 5).length).toBe(5);
    expect(chunkShares(8, 5).reduce((s, x) => s + x, 0)).toBeCloseTo(1, 12);
    expect(chunkShares(3, 0)).toEqual([]);
    expect(run({ n: 8, items: 5 }).ops.filter((o) => o.kind === 'kernel').length).toBe(5);
    expect(run({ n: 4, items: 10 }).e1!.conditions.find((c) => c.id === 'equal')!.holds).toBe(false);
  });
});

describe('stepping', () => {
  it('groups events by time; the first step is the host queuing everything at 0', () => {
    const s = simulate(preset('p-pipeline'));
    expect(s.steps[0].t).toBe(0);
    expect(s.steps[0].events[0].text).toBe('The host thread queues all 24 operations at time 0; every call returns at once.');
    expect(s.steps[s.steps.length - 1].t).toBe(18);
    const times = s.steps.map((x) => x.t);
    expect([...times].sort((x, y) => x - y)).toEqual(times);
    expect(new Set(times).size).toBe(times.length);
    expect(stepAt(s, -1)).toBe(null);
    expect(stepAt(s, 1.5)).toBe(1);
    const at1 = s.steps[1];
    expect(at1.t).toBe(1);
    expect(at1.events.map((e) => e.kind)).toEqual(['end', 'start', 'start']); // ends before starts
    expect(s.ops[stepOp(at1)!]).toMatchObject({ kind: 'in', chunk: 1 });
  });
  it('a pageable run narrates the host being stuck', () => {
    const s = simulate({ ...FOUR_CHUNKS, host: 'pageable' });
    expect(s.steps[0].events[0].text).toContain('only 1 operation queued before it is stuck');
    expect(s.steps.some((st) => st.events.some((e) => e.kind === 'block'))).toBe(true);
    expect(s.steps.some((st) => st.events.some((e) => e.kind === 'issue' && e.text.startsWith('Only now')))).toBe(true);
  });
});

describe('the reader\'s device (lab 23.1 rows)', () => {
  it('scales the parts so the largest is 24 units', () => {
    const p = fromReports({ name: 'parts', h2d_s: 0.004, kernel_s: 0.006, d2h_s: 0.003 }, 'cuda')!;
    expect(p.b).toBeCloseTo(24, 12);
    expect(p.a).toBeCloseTo(16, 12);
    expect(p.c).toBeCloseTo(12, 12);
    expect(p.unit).toBeCloseTo(0.00025, 15);
    expect(fromReports({ name: 'parts', fill_s: 1, kernel_s: 2, drain_s: 1 }, 'metal')!.a).toBeCloseTo(12, 12);
    expect(fromReports({ name: 'parts', fill_s: 1, kernel_s: 2, drain_s: 1 }, 'cuda')).toBe(null);
  });
  it('takes the copy-engine count from a CUDA report only (ROCm reports a constant 8, B3)', () => {
    const row = { name: 'parts', h2d_s: 1, kernel_s: 1, d2h_s: 1, async_engines: 1 };
    expect(fromReports(row, 'cuda', 'cuda')!.engines).toBe(1);
    expect(fromReports(row, 'cuda', 'hip')!.engines).toBe(undefined);
    expect(fromReports({ ...row, async_engines: 8 }, 'cuda', 'hip')!.engines).toBe(undefined);
  });
  it('reads the staged rows in the same units, and nothing without a parts row', () => {
    const rows = [
      { name: 'parts', h2d_s: 0.002, kernel_s: 0.004, d2h_s: 0.002 },
      { name: 'staged', streams: 4, chunks: 8, order: 'depth', seconds: 0.005 },
      { name: 'staged', streams: 4, chunks: 8, order: 'breadth', host: 'pinned', seconds: 0.006 },
      { name: 'staged', streams: 4, chunks: 16, order: 'depth', host: 'pageable', seconds: 0.008 },
      { name: 'serial', host: 'pinned', seconds: 0.008 },
    ];
    const got = parseLabRows(rows, 'cuda');
    expect(got.parts!.b).toBeCloseTo(24, 12);
    expect(got.staged.map((r) => [r.order, r.host, Number(r.units.toFixed(9))])).toEqual([
      ['depth', 'pinned', 30], ['breadth', 'pinned', 36], ['depth', 'pageable', 48],
    ]);
    expect(parseLabRows(rows.slice(1), 'cuda')).toEqual({ parts: null, staged: [] });
    const metal = parseLabRows([
      { name: 'parts', fill_s: 0.001, kernel_s: 0.002, drain_s: 0.001 },
      { name: 'inflight', buffers: 3, chunks: 16, seconds: 0.003 },
    ], 'metal');
    expect(metal.staged).toEqual([{ streams: 3, chunks: 16, order: 'depth', host: 'pinned', seconds: 0.003, units: 36 }]);
  });
});

describe('a pageable copy staged through pinned memory (the scene)', () => {
  it('pieces follow one another and the call returns after the last one', () => {
    const p = stagingPlan(8, 2, { hostRate: 0.5, dmaRate: 1, buffers: 2 });
    expect(p.pieces.length).toBe(4);
    expect(p.pieces.map((x) => [x.hostFrom, x.hostTo, x.dmaFrom, x.dmaTo])).toEqual([
      [0, 1, 1, 3], [1, 2, 3, 5], [3, 4, 5, 7], [5, 6, 7, 9],
    ]);
    expect(p.returnsAt).toBe(9);
    expect(p.pinnedTime).toBe(8);
    expect(p.hostCopying).toBe(4);
    for (const x of p.pieces) expect(x.dmaFrom).toBeGreaterThanOrEqual(x.hostTo);
    expect(stagingPlan(5, 2).pieces.map((x) => x.size)).toEqual([2, 2, 1]);
    expect(stagingPlan(0, 2).returnsAt).toBe(0);
  });
});

describe('Fig. 23.13: the exchange is hidden while it ends before the internal rows', () => {
  it("the book's proportions: overlap takes about two thirds of the time", () => {
    const a = haloLanes(HALO_BOOK, false);
    const b = haloLanes(HALO_BOOK, true);
    expect(a.total).toBe(18);
    expect(b.total).toBe(12);
    expect(b.hidden).toBe(true);
    expect(b.total / a.total).toBeCloseTo(2 / 3, 12);
    // the reduction cannot be overlapped: it starts after everything else
    const red = b.bars.find((x) => x.label.startsWith('reduce'))!;
    expect(red.from).toBe(Math.max(...b.bars.filter((x) => x !== red).map((x) => x.to)));
  });
  it('a longer exchange shows through', () => {
    const b = haloLanes({ ...HALO_BOOK, exchange: 4 }, true);
    expect(b.hidden).toBe(false);
    expect(b.exposed).toBe(0.5);
    expect(b.total).toBe(12.5);
  });
});
