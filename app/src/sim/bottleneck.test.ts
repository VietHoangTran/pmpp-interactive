import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { questionsS69 } from '../chapters/ch06/S69';
import type { NumericQ } from '../components/Ask';
import {
  ABOVE,
  OPS,
  PRESETS,
  RESOURCES,
  applyOp,
  canApply,
  compare,
  evaluate,
  headroom,
  labNum,
  placeRow,
  presetById,
  roofFraction,
  run,
  stopPoints,
  stopVerdict,
  tiledGemmBytes,
  type OpId,
} from './bottleneck';
import { SCENARIOS } from './checklist';

const P = (id: string) => presetById(id).kernel;
const step = (id: string, op: OpId) => run(P(id), [op]).steps[0];

describe('the model: run time is the slowest resource', () => {
  it('total is the largest of the four resource times', () => {
    for (const p of PRESETS) {
      const e = evaluate(p.kernel);
      expect(e.total).toBe(Math.max(...RESOURCES.map((r) => e.times[r])));
      expect(e.times[e.bottleneck]).toBe(e.total);
    }
  });
  it('starting kernels start where their descriptions say', () => {
    const naive = evaluate(P('naive'));
    expect([naive.total, naive.bottleneck, naive.occupancy]).toEqual([80, 'global', 1]);
    const sl = evaluate(P('shared-limited'));
    expect([sl.total, sl.bottleneck, sl.blocksPerSM, sl.limiter, sl.memRate]).toEqual([40, 'global', 3, 'shared', 0.75]);
    const up = evaluate(P('unpadded'));
    expect([up.total, up.bottleneck]).toEqual([80, 'shared']);
    const va = evaluate(P('vecadd'));
    expect([va.total, va.bottleneck, va.waves]).toEqual([60, 'global', 4]);
    const ex = evaluate(P('exchange'));
    expect([ex.total, ex.bottleneck]).toEqual([50, 'barrier']);
  });
  it('every preset fills whole waves to start with', () => {
    for (const p of PRESETS) expect(evaluate(p.kernel).tail).toBe(1);
  });
});

describe('p. 153: relieving a non-bottleneck gains nothing', () => {
  it('vector loads on a bandwidth-bound kernel at full occupancy leave the time unchanged', () => {
    const s = step('vecadd', 'vector');
    expect(s.verdict).toBe('same');
    expect(s.after.total).toBe(s.before.total);
    expect(s.after.times.compute).toBeLessThan(s.before.times.compute);
    expect(s.relievedNonBottleneck).toBe(true);
  });
  it('unrolling a bandwidth-bound kernel leaves the time unchanged', () => {
    const s = step('naive', 'unroll');
    expect(s.verdict).toBe('same');
    expect(s.relievedNonBottleneck).toBe(true);
  });
  it('anything but padding leaves the bank-conflicted kernel where it is', () => {
    for (const op of ['vector', 'unroll', 'coarsen', 'double-buffer', 'tile'] as OpId[]) expect(step('unpadded', op).verdict).toBe('same');
    const pad = step('unpadded', 'pad');
    expect(pad.verdict).toBe('faster');
    expect([pad.before.total, pad.after.total]).toEqual([80, 31]);
    expect(pad.after.bottleneck).toBe('compute');
  });
  it('padding with no conflicts gains nothing', () => {
    const s = step('shared-limited', 'pad');
    expect(s.verdict).toBe('same');
    expect(s.after.blocksPerSM).toBe(3); // 1/32 more shared memory does not cost a block here
  });
});

describe('p. 153: spending the bottleneck resource slows the kernel', () => {
  it('tiling a shared-memory-limited kernel raises its time', () => {
    const s = step('shared-limited', 'tile');
    expect(s.verdict).toBe('slower');
    expect([s.before.total, s.after.total]).toEqual([40, 80]);
    expect([s.before.blocksPerSM, s.after.blocksPerSM]).toEqual([3, 1]);
    expect(s.after.limiter).toBe('shared');
    // it did halve the re-read traffic: the loss is all occupancy
    expect(applyOp(P('shared-limited'), 'tile').reuseBytes).toBe(10);
  });
  it('double buffering a shared-memory-limited kernel is slower still', () => {
    const s = step('shared-limited', 'double-buffer');
    expect(s.verdict).toBe('slower');
    expect(s.after.times.barrier).toBeLessThan(s.before.times.barrier);
    expect(s.after.total).toBe(120);
  });
  it('tiling a bandwidth-bound kernel with occupancy to spare is faster (the good case)', () => {
    const s = step('naive', 'tile');
    expect(s.verdict).toBe('faster');
    expect([s.before.total, s.after.total]).toEqual([80, 48]);
    expect(s.after.occupancy).toBe(1);
  });
});

describe('the bottleneck moves', () => {
  it('naive matmul: global, then compute after tiling, then global after unrolling', () => {
    const { steps } = run(P('naive'), ['tile', 'unroll', 'vector', 'tile']);
    expect(steps.map((s) => s.after.bottleneck)).toEqual(['compute', 'global', 'global', 'compute']);
    expect(steps.map((s) => s.verdict)).toEqual(['faster', 'faster', 'same', 'faster']);
  });
  it('double buffering the barrier-bound kernel moves the bottleneck to global memory', () => {
    const s = step('exchange', 'double-buffer');
    expect(s.verdict).toBe('faster');
    expect(s.before.bottleneck).toBe('barrier');
    expect(s.after.bottleneck).toBe('global');
    expect(s.after.total).toBeCloseTo(32.8125, 10);
  });
});

describe('vector loads and unrolling cover latency at low occupancy (pp. 139, 149)', () => {
  it('on the shared-limited kernel both are faster, by filling the bus', () => {
    for (const op of ['vector', 'unroll'] as OpId[]) {
      const s = step('shared-limited', op);
      expect(s.verdict).toBe('faster');
      expect(s.after.memRate).toBe(1);
      expect(s.after.total).toBe(30);
    }
  });
});

describe('coarsening pitfalls (pp. 142–143)', () => {
  it('unnecessary: no change while the grid still fills whole waves; too much: a partial wave', () => {
    const { steps } = run(P('vecadd'), ['coarsen', 'coarsen', 'coarsen']);
    expect(steps.map((s) => s.verdict)).toEqual(['same', 'same', 'slower']);
    expect(steps.map((s) => s.after.waves)).toEqual([2, 1, 16 / 24]);
    expect(steps[2].after.tail).toBe(1.5);
    expect(steps[2].after.total).toBe(90);
  });
  it('useful where blocks reload the same inputs', () => {
    expect(step('naive', 'coarsen').verdict).toBe('faster');
    expect(step('shared-limited', 'coarsen').verdict).toBe('faster');
  });
});

describe('optimisations', () => {
  it('each can be applied only up to its limit, and some need something to act on', () => {
    expect(canApply(P('vecadd'), 'pad')).toEqual({ ok: false, why: 'no shared memory to pad' });
    expect(canApply(P('naive'), 'double-buffer')).toEqual({ ok: false, why: 'no barriers to remove' });
    const once = applyOp(P('naive'), 'unroll');
    expect(canApply(once, 'unroll').ok).toBe(false);
    expect(applyOp(once, 'unroll')).toBe(once);
    const r = run(P('naive'), ['tile', 'tile', 'tile', 'tile']);
    expect(r.steps).toHaveLength(3);
  });
  it('every op says what it does and cites a page', () => {
    for (const o of OPS) {
      expect(o.effect.length).toBeGreaterThan(20);
      expect(o.page).toMatch(/^pp?\. \d/);
    }
  });
  it('compare() calls equal totals the same', () => {
    expect(compare(40, 40)).toBe('same');
    expect(compare(40, 30)).toBe('faster');
    expect(compare(40, 80)).toBe('slower');
  });
});

describe('the scenario cards of 6.8 agree with this model', () => {
  const expectFrom: Record<string, () => string> = {
    'tiling-bandwidth': () => step('naive', 'tile').verdict,
    'padding-no-conflict': () => step('shared-limited', 'pad').verdict,
    'coarsen-small-grid': () => run(P('vecadd'), ['coarsen', 'coarsen', 'coarsen']).steps.map((s) => s.verdict).includes('slower') ? 'slower' : 'same',
    'vector-full': () => step('vecadd', 'vector').verdict,
    'double-buffer-barrier': () => step('exchange', 'double-buffer').verdict,
  };
  for (const s of SCENARIOS)
    it(s.id, () => {
      expect(expectFrom[s.id]).toBeDefined();
      expect(expectFrom[s.id]()).toBe(s.expect);
    });
});

describe('roofline arithmetic (p. 154)', () => {
  it('q-stop: 12 GB in 5 ms against 3000 GB/s is 80% of the roof', () => {
    // vector-addition counts: 1 FLOP per 12 bytes, so 12 GB is 1 GFLOP
    const r = roofFraction(1e9, 12e9, 0.005, 1e6, 3000);
    expect(r.intensity).toBeCloseTo(1 / 12, 12);
    expect(r.regime).toBe('memory');
    expect(r.roof).toBeCloseTo(250, 9);
    expect(r.achieved).toBeCloseTo(200, 9);
    expect(100 * r.fraction).toBeCloseTo(80, 9);
    expect(headroom(r.fraction)).toBeCloseTo(1.25, 12);
    const q = questionsS69['ch06/6.9/q-stop'] as NumericQ;
    expect(q.answer).toBe(80);
  });
  it("the book's speed-of-light example: 12 GB in 4 ms on 3.35 TB/s is about 90% (ch. 5, p. 96)", () => {
    const r = roofFraction(1e9, 12e9, 0.004, 66900, 3350);
    expect(Math.round(100 * r.fraction)).toBe(90);
  });
  it('vector addition at 1/12 FLOP/B: moving bytes at the measured bandwidth is 100% of the roof', () => {
    const n = 1e8;
    const bw = 271;
    const seconds = (12 * n) / (bw * 1e9);
    const r = roofFraction(n, 12 * n, seconds, 1881, bw);
    expect(r.intensity).toBeCloseTo(1 / 12, 12);
    expect(r.fraction).toBeCloseTo(1, 12);
    expect(stopVerdict(r.fraction, 0.8)).toBe('stop');
  });
  it('compute-bound above the threshold', () => {
    const r = roofFraction(1e12, 1e10, 1, 1000, 100); // 100 FLOP/B on a 10 FLOP/B device
    expect(r.regime).toBe('compute');
    expect(r.roof).toBe(1000);
  });
  it('stop verdicts: room below the threshold, stop at or above it, above beyond 110%', () => {
    expect(stopVerdict(0.5, 0.8)).toBe('room');
    expect(stopVerdict(0.8, 0.8)).toBe('stop');
    expect(stopVerdict(1.05, 0.8)).toBe('stop');
    expect(stopVerdict(ABOVE + 0.01, 0.8)).toBe('above');
    expect(headroom(0.25)).toBe(4);
    expect(headroom(1.3)).toBe(1);
  });
});

describe('tiled GEMM traffic, the book’s count', () => {
  it('square W with T dividing W: intensity T / 4; T = 1 is the naive 0.25', () => {
    for (const T of [1, 2, 4, 8, 16, 32]) {
      const W = 1024;
      expect((2 * W ** 3) / tiledGemmBytes(W, W, W, T)).toBeCloseTo(T / 4, 12);
    }
  });
  it('a linear layer at batch 1 is about 0.47 FLOP/B at tile 16, near the 0.5 limit of 5.1', () => {
    const I = (2 * 1 * 1024 * 4096) / tiledGemmBytes(1, 1024, 4096, 16);
    expect(I).toBeCloseTo(0.4706, 4);
    expect(I).toBeLessThan(0.5);
  });
  it('batch 256 at tile 16 is the square kernel’s T / 4', () => {
    expect((2 * 256 * 1024 * 4096) / tiledGemmBytes(256, 1024, 4096, 16)).toBeCloseTo(4, 12);
  });
});

describe('placing lab rows', () => {
  // Real rows from this machine's chapter 5 runs, if present; fixed copies otherwise.
  const tiled16 = { name: 'tiled', tile: 16, width: 4096, seconds: 0.0892595, gflops: 1539.76835 };
  const lin1 = { name: 'linear_layer', batch: 1, k: 1024, l: 4096, tile: 16, seconds: 0.000494874897, gflops: 16.9509669 };
  it('labNum', () => {
    expect(labNum('ch06/lab2_vector')).toBe('6.2');
    expect(labNum('ch05/lab3_gemm')).toBe('5.3');
  });
  it('a square tiled row lands at T / 4 with its measured GFLOP/s', () => {
    const p = placeRow('ch05/lab2_tiled', tiled16)!;
    expect(p.kind).toBe('gemm');
    expect(p.flop / p.bytes).toBeCloseTo(4, 12);
    expect(p.flop / p.seconds / 1e9).toBeCloseTo(1539.77, 1);
  });
  it('a linear-layer row lands at about 0.47 FLOP/B', () => {
    const p = placeRow('ch05/lab3_gemm', lin1)!;
    expect(p.batch).toBe(1);
    expect(p.flop / p.bytes).toBeCloseTo(0.4706, 4);
  });
  it('a vector-addition row with only GB/s gets its time from 12 bytes per element', () => {
    const p = placeRow('ch06/lab2_vector', { name: 'vecadd', variant: 'float4', n: 1e8, gb_per_s: 240 })!;
    expect(p.kind).toBe('stream');
    expect(p.flop / p.bytes).toBeCloseTo(1 / 12, 12);
    expect(p.seconds).toBeCloseTo(1.2e9 / 240e9, 15);
    expect(p.label).toBe('vecadd (float4)');
  });
  it('labels use the row\'s own variant names, once, and not the build variant', () => {
    const lin = { name: 'linear', variant: 'tuned', kernel: 'tuned', tile: 16, batch: 256, in: 1024, out: 4096, seconds: 0.0016 };
    expect(placeRow('ch06/lab4_linear', lin)!.label).toBe('linear (tuned), batch 256, tile 16');
    expect(placeRow('ch05/lab2_tiled', { ...tiled16, variant: 'solution' })!.label).toBe('tiled, width 4,096, tile 16');
  });
  it('lab 6.4 keeps tile 16 at batch 1 and 256; labs 6.2 and 6.5 keep every row', () => {
    const lin = (batch: number, tile: number) => ({
      lab: 'ch06/lab4_linear', row: { name: 'linear', kernel: 'tuned', tile, batch, in: 1024, out: 4096, seconds: 0.001 },
    });
    const pts = stopPoints([
      lin(1, 16), lin(16, 16), lin(256, 16), lin(1, 32),
      { lab: 'ch06/lab2_vector', row: { name: 'vecadd', variant: 'vec4', n: 67108864, seconds: 0.003 } },
      { lab: 'ch06/lab5_doublebuf', row: { name: 'matmul', kernel: 'double_buffered', tile: 32, width: 4096, seconds: 0.088 } },
      { lab: 'ch06/lab1_coalescing', row: { name: 'copy', n: 33554432, seconds: 0.00095, gb_per_s: 281 } },
    ]);
    expect(pts.map((p) => [p.lab, p.batch ?? null, p.tile ?? null])).toEqual([
      ['6.4', 1, 16], ['6.4', 256, 16], ['6.2', null, null], ['6.5', null, 32],
    ]);
    expect(pts[2].flop / pts[2].bytes).toBeCloseTo(1 / 12, 12);
    expect(pts[3].flop / pts[3].bytes).toBeCloseTo(8, 12);
  });
  it('rows without the counts are not placed', () => {
    expect(placeRow('ch06/lab2_vector', { name: 'vecadd', gb_per_s: 240 })).toBeNull();
    expect(placeRow('ch05/lab2_tiled', { name: 'naive', width: 4096, seconds: 0.15 })).toBeNull();
    expect(placeRow('ch06/lab4_linear', { name: 'linear', batch: 1, k: 1024, l: 4096, tile: 16 })).toBeNull();
  });
  it('stopPoints keeps tile 16 from lab 5.2 and batches 1 and 256 from lab 5.3', () => {
    const rows = [
      { lab: 'ch05/lab2_tiled', row: { ...tiled16, tile: 8 } },
      { lab: 'ch05/lab2_tiled', row: tiled16 },
      { lab: 'ch05/lab3_gemm', row: lin1 },
      { lab: 'ch05/lab3_gemm', row: { ...lin1, batch: 64 } },
      { lab: 'ch05/lab3_gemm', row: { ...lin1, batch: 256 } },
      { lab: 'ch05/lab1_roofline', row: { name: 'naive_matmul', width: 4096, seconds: 0.15, tile: 1 } },
    ];
    const pts = stopPoints(rows);
    expect(pts.map((p) => [p.lab, p.tile, p.batch])).toEqual([
      ['5.2', 16, undefined],
      ['5.3', 16, 1],
      ['5.3', 16, 256],
    ]);
  });
  it('reads this machine’s real chapter 5 rows when they exist', () => {
    const file = path.resolve(__dirname, '../../../reports/runs/metal.json');
    if (!fs.existsSync(file)) return;
    const runOn = JSON.parse(fs.readFileSync(file, 'utf8')) as { results: { lab: string; variant: string; bench: Record<string, unknown>[] }[] };
    const rows = runOn.results
      .filter((r) => r.variant === 'solution')
      .flatMap((r) => r.bench.map((row) => ({ lab: r.lab, row })));
    for (const p of stopPoints(rows)) {
      expect(p.seconds).toBeGreaterThan(0);
      expect(p.flop / p.bytes).toBeGreaterThan(0);
    }
  });
});
