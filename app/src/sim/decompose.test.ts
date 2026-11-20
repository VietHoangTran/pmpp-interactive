import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { warpAccess, runs } from './coalesce';
import {
  CRITERION_LABEL,
  DECOMP_LABEL,
  NVIDIA_WARP,
  PRESETS,
  SCENE,
  SPLITK_TINY,
  TOUR,
  conv,
  decompose,
  decomposeState,
  describeSKStep,
  describeWorker,
  gradeCard,
  hist,
  histValues,
  matvec,
  ownerOfWeight,
  sceneModel,
  serialise,
  splitChoices,
  splitKOrder,
  splitKRun,
  splitKTraffic,
  spmv,
  warp0Load,
  warpOps,
  type Interleave,
  type SplitKConfig,
} from './decompose';

const one = { warpWidth: 1, S: 1 };
const SECTOR = 32; // NVIDIA's coalescing unit (ch06-facts A2), as coalesce.ts SEGMENT_MODELS.nvidia

describe('convolution', () => {
  it('a 5-wide filter decomposed input-centric: 5 writers on every output away from the edges, each an atomic', () => {
    const p = conv(12, 5);
    expect(p.edges.length).toBe(54); // 8 inner outputs × 5, plus 3 + 4 at each edge
    const d = decompose(p, 'input', one);
    expect(d.writers).toEqual([3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 4, 3]);
    expect(d.metrics).toMatchObject({ threads: 12, outputsNeedingAtomics: 12, atomicUpdates: 54, plainStores: 0, worstWriters: 5, registerAccumulations: 0 });
  });
  it('output-centric: one writer per output, a register sum, a plain store', () => {
    const d = decompose(conv(12, 5), 'output', one);
    expect(d.writers.every((w) => w === 1)).toBe(true);
    expect(d.metrics).toMatchObject({ threads: 12, outputsNeedingAtomics: 0, atomicUpdates: 0, plainStores: 12, worstWriters: 1, registerAccumulations: 42 });
    // Easy mapping: each thread inspects only the inputs it uses.
    expect(d.metrics.inspectedTotal).toBe(d.metrics.contributions);
  });
  it('any odd width w: w writers in the middle', () => {
    for (const w of [3, 5, 7]) expect(decompose(conv(16, w), 'input', one).writers[8]).toBe(w);
  });
  it('the valid form: n − w + 1 outputs, every one with w inputs', () => {
    const p = conv(8, 3, true);
    expect(p.nOut).toBe(6);
    expect(decompose(p, 'input', one).writers).toEqual([3, 3, 3, 3, 3, 3]);
  });
});

describe('histogram', () => {
  it('the explorer preset: 16 inputs, 4 bins, half skewed to bin 0', () => {
    const v = histValues(16, 4, 0.5);
    expect(v).toEqual([0, 0, 2, 0, 0, 1, 0, 3, 0, 0, 2, 0, 0, 1, 0, 3]);
    const out = decompose(hist(v, 4), 'output', one);
    const inp = decompose(hist(v, 4), 'input', one);
    expect(out.metrics).toMatchObject({ threads: 4, atomicUpdates: 0, inspectedMax: 16, inspectedTotal: 64 });
    expect(inp.metrics).toMatchObject({ threads: 16, outputsNeedingAtomics: 4, atomicUpdates: 16, worstWriters: 10, inspectedTotal: 16 });
    // Even with a free lookup, the bins get uneven work: 10 of 16 in bin 0.
    expect(out.metrics.workMax).toBe(10);
    expect(out.metrics.imbalance).toBe(2.5);
    expect(inp.metrics.imbalance).toBe(1);
  });
  it('no skew: the inputs go round the bins evenly', () => {
    expect(histValues(12, 4, 0)).toEqual([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
  });
  it('q-hist-inspect: 1,000,000 inputs into 256 bins, one thread per bin reads 256 times as much', () => {
    const p = hist(histValues(1_000_000, 256, 0), 256);
    const out = decompose(p, 'output', one).metrics;
    const inp = decompose(p, 'input', one).metrics;
    expect(out.threads).toBe(256);
    expect(out.inspectedTotal).toBe(256_000_000);
    expect(inp.inspectedTotal).toBe(1_000_000);
    expect(out.inspectedTotal / inp.inspectedTotal).toBe(256);
  });
});

describe('sparse matrix-vector product', () => {
  it('a row with half the non-zeros: CSR is three times out of balance, COO is balanced and pays an atomic per non-zero', () => {
    const p = spmv(6, 8);
    expect(p.nIn).toBe(20);
    expect(p.rowOf.filter((r) => r === 0).length).toBe(10);
    // No row repeats a column.
    for (let r = 0; r < 6; r++) {
      const cs = p.colOf.filter((_, k) => p.rowOf[k] === r);
      expect(new Set(cs).size).toBe(cs.length);
    }
    const csr = decompose(p, 'output', one).metrics;
    const coo = decompose(p, 'input', one).metrics;
    expect(csr).toMatchObject({ threads: 6, atomicUpdates: 0, workMax: 10 });
    expect(csr.imbalance).toBeCloseTo(3, 12);
    expect(coo).toMatchObject({ threads: 20, outputsNeedingAtomics: 6, atomicUpdates: 20, worstWriters: 10, imbalance: 1 });
  });
});

describe('decode matrix-vector product', () => {
  const opts = (S = 4, warpWidth = NVIDIA_WARP) => ({ warpWidth, S });
  it('the narrow preset (2 outputs, 256 inputs): only split-K exposes more than a couple of warps, at S atomics per output', () => {
    const p = matvec(2, 256);
    expect(decompose(p, 'output', opts()).metrics).toMatchObject({ threads: 2, atomicUpdates: 0 });
    expect(decompose(p, 'warp', opts()).metrics).toMatchObject({ threads: 64, atomicUpdates: 0, imbalance: 1 });
    expect(decompose(p, 'splitk', opts(4)).metrics).toMatchObject({ threads: 256, outputsNeedingAtomics: 2, atomicUpdates: 8, worstWriters: 4, imbalance: 1 });
    expect(decompose(p, 'splitk', opts(1)).metrics).toMatchObject({ threads: 64, atomicUpdates: 0 });
    expect(decompose(p, 'input', opts()).metrics).toMatchObject({ threads: 256, outputsNeedingAtomics: 2, atomicUpdates: 512, worstWriters: 256, registerAccumulations: 0 });
  });
  it('warp 0\'s first load of W in NVIDIA\'s 32-byte sectors: one contiguous run for a warp per output, one sector per lane for a thread per output', () => {
    const wide = matvec(32, 256);
    const sectors = (m: 'output' | 'warp' | 'splitk' | 'input', p = wide, S = 4) => warpAccess(warp0Load(p, m, opts(S)), 4, SECTOR);
    expect(sectors('output').transactions).toBe(32);
    expect(runs(warp0Load(wide, 'output', opts()), 4)).toBe(32);
    expect(sectors('warp').transactions).toBe(4);
    expect(runs(warp0Load(wide, 'warp', opts()), 4)).toBe(1);
    expect(sectors('splitk').transactions).toBe(4);
    expect(sectors('input').transactions).toBe(4);
    // With only 2 outputs, a thread per output has 2 live lanes: 2 sectors.
    expect(sectors('output', matvec(2, 256)).transactions).toBe(2);
    expect(sectors('splitk', matvec(2, 256)).transactions).toBe(4);
    // A 64-wide warp reads twice as many sectors in one contiguous run.
    expect(warpAccess(warp0Load(wide, 'warp', opts(4, 64)), 4, SECTOR).transactions).toBe(8);
  });
  it('S choices keep every lane busy', () => {
    expect(splitChoices(256, 32)).toEqual([1, 2, 4, 8]);
    expect(splitChoices(64, 64)).toEqual([1]);
  });
  it('clicking a weight finds its owner, consistent with the workers\' reads', () => {
    for (const m of ['input', 'output', 'warp', 'splitk'] as const) {
      const p = matvec(4, 128);
      const d = decompose(p, m, opts(4));
      for (const [r, c] of [[0, 0], [3, 127], [2, 40], [1, 97]]) {
        const w = d.workers[ownerOfWeight(p, m, opts(4), r, c)];
        expect(w.useful.includes(c) && (m === 'input' || w.writes.some((x) => x.o === r))).toBe(true);
      }
    }
  });
  it('the split-K mapping in words, with the worker\'s own numbers', () => {
    const d = decompose(matvec(2, 256), 'splitk', opts(4));
    expect(describeWorker(d, 5)).toContain('Warp 5 owns output 5 ÷ 4 = 1 and chunk 5 mod 4 = 1, which starts at 64.');
    expect(describeWorker(d, 5)).toContain('W[1][64 + l + i × 32] for i = 0 to 1');
    expect(describeWorker(d, 5)).toContain('atomic');
    expect(describeWorker(decompose(matvec(2, 256), 'warp', opts()), 0)).toContain('plain store is safe');
  });
});

describe('presets', () => {
  it('four presets, with the numbers the browser check asserts', () => {
    const m = Object.fromEntries(PRESETS.map((p) => [p.id, decomposeState(p.state).metrics]));
    expect(PRESETS.length).toBe(4);
    expect(m['conv-scatter']).toMatchObject({ outputsNeedingAtomics: 12, atomicUpdates: 54, worstWriters: 5 });
    expect(m['hist-bins']).toMatchObject({ threads: 4, inspectedMax: 16, atomicUpdates: 0 });
    expect(m['spmv-coo']).toMatchObject({ threads: 20, atomicUpdates: 20, imbalance: 1 });
    expect(m['gemv-narrow']).toMatchObject({ threads: 256, atomicUpdates: 8, worstWriters: 4 });
  });
});

describe('the gather-scatter scene', () => {
  const s = sceneModel();
  it('8 inputs, 6 outputs, each the sum of 3 neighbours; every input feeds up to 3 outputs', () => {
    expect([s.p.nIn, s.p.nOut, SCENE.w]).toEqual([8, 6, 3]);
    expect(s.fanOut).toEqual([1, 2, 3, 3, 3, 3, 2, 1]);
    expect(s.sums).toEqual([8, 6, 10, 15, 16, 17]);
  });
  it('gather: 6 threads, 6 writes to 6 different cells, all on the first tick', () => {
    expect(s.gather.metrics).toMatchObject({ threads: 6, atomicUpdates: 0, plainStores: 6 });
    expect(s.gatherTicks.ticks.length).toBe(1);
    expect(s.gatherTicks.ticks[0].served.length).toBe(6);
  });
  it('scatter: 8 threads, 3 writers on every output, 3 ticks to drain one update per tick', () => {
    expect(s.scatter.metrics).toMatchObject({ threads: 8, outputsNeedingAtomics: 6, atomicUpdates: 18, worstWriters: 3 });
    expect(s.scatterTicks.issued).toEqual([3, 3, 3, 3, 3, 3]);
    expect(s.scatterTicks.ticks.length).toBe(3);
    expect(s.scatterTicks.ticks.map((t) => t.waiting)).toEqual([[2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]]);
    // Each output's queue drains in worker order.
    expect(s.scatterTicks.ticks.map((t) => t.served.find((x) => x.o === 2)!.worker)).toEqual([2, 3, 4]);
  });
  it('gather reads: 18 reads of 8 inputs, the middle four read 3 times each', () => {
    expect(s.reads).toEqual([1, 2, 3, 3, 3, 3, 2, 1]);
    expect(s.reads.reduce((a, b) => a + b, 0)).toBe(18);
  });
  it('the histogram preview: a bin thread inspects all 8 inputs to find its own', () => {
    expect(s.hp.values).toEqual([0, 1, 1, 1, 2, 0, 2, 0]);
    expect(s.histGather.workers.map((w) => [w.inspects, w.useful.length])).toEqual([[8, 3], [8, 3], [8, 2]]);
  });
  it('serialise: one tick per writer of the busiest output', () => {
    const d = decompose(conv(12, 5), 'input', one);
    expect(serialise(d).ticks.length).toBe(5);
  });
});

describe('split-K, warp by warp', () => {
  const cfg = (S: number, atomics: boolean, extra: Partial<SplitKConfig> = {}): SplitKConfig => ({ ...SPLITK_TINY, S, atomics, ...extra });
  const ALL: Interleave[] = ['sequential', 'round-robin', 'reversed'];

  it('the tiny case: 3 outputs × S warps, each with its strips, 2 shuffles and a commit', () => {
    expect(warpOps(cfg(1, true), 0).map((o) => o.kind)).toEqual(['load', 'fma', 'load', 'fma', 'load', 'fma', 'load', 'fma', 'shfl', 'shfl', 'store']);
    expect(warpOps(cfg(2, true), 0).map((o) => o.kind)).toEqual(['load', 'fma', 'load', 'fma', 'shfl', 'shfl', 'atomic']);
    expect(warpOps(cfg(4, false), 0).map((o) => o.kind)).toEqual(['load', 'fma', 'shfl', 'shfl', 'read', 'write']);
    expect(splitKOrder(cfg(1, true), 'sequential').length).toBe(33);
    expect(splitKOrder(cfg(2, true), 'sequential').length).toBe(42);
    expect(splitKOrder(cfg(4, false), 'sequential').length).toBe(72);
  });
  it('lanes cover W exactly once, in contiguous strips', () => {
    for (const S of [1, 2, 4]) {
      const r = splitKRun(cfg(S, true), splitKOrder(cfg(S, true), 'sequential'));
      const seen = new Map<string, number>();
      for (const st of r.steps)
        if (st.op.kind === 'load') {
          const live = st.ks!.filter((k) => k >= 0);
          expect(live).toEqual(live.map((_, t) => live[0] + t));
          for (const k of live) seen.set(`${st.j},${k}`, (seen.get(`${st.j},${k}`) ?? 0) + 1);
        }
      expect(seen.size).toBe(3 * 16);
      expect([...seen.values()].every((v) => v === 1)).toBe(true);
    }
  });
  it('with atomics (or S = 1) every schedule gives the reference', () => {
    for (const S of [1, 2, 4])
      for (const how of ALL) {
        const r = splitKRun(cfg(S, true), splitKOrder(cfg(S, true), how));
        expect(r.correct, `S=${S} ${how}`).toBe(true);
      }
  });
  it('every interleaving of two warps on one output: atomics always right, plain += loses a contribution in some', () => {
    // One output, S = 2, 8 inputs, a 4-wide warp: one strip per warp.
    const small = (atomics: boolean): SplitKConfig => ({ batch: 2, K: 8, N: 1, warpWidth: 4, S: 2, atomics });
    const interleavings = (a: number, b: number): number[][] => {
      if (a === 0) return [new Array(b).fill(1)];
      if (b === 0) return [new Array(a).fill(0)];
      return [...interleavings(a - 1, b).map((x) => [0, ...x]), ...interleavings(a, b - 1).map((x) => [1, ...x])];
    };
    const atom = interleavings(5, 5);
    expect(atom.length).toBe(252);
    expect(atom.every((o) => splitKRun(small(true), o).correct)).toBe(true);
    const plain = interleavings(6, 6);
    expect(plain.length).toBe(924);
    const lost = plain.filter((o) => !splitKRun(small(false), o).correct);
    expect(lost.length).toBeGreaterThan(0);
    // A schedule loses exactly when one warp's read and write straddle the other's write.
    for (const o of plain) {
      const r = splitKRun(small(false), o);
      expect(r.correct).toBe(r.lostCount === 0);
    }
    // Sequential warps never lose; lockstep always does.
    expect(splitKRun(small(false), [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]).correct).toBe(true);
    expect(splitKRun(small(false), [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]).correct).toBe(false);
  });
  it('the stepper: plain += with all warps together loses updates; one warp at a time does not', () => {
    for (const S of [2, 4]) {
      const rr = splitKRun(cfg(S, false), splitKOrder(cfg(S, false), 'round-robin'));
      expect(rr.correct).toBe(false);
      // All S warps of an output read before any writes: only the last write survives.
      expect(rr.lostCount).toBe(3 * 2 * (S - 1));
      const first = rr.steps.findIndex((s) => s.lost);
      expect(first).toBeGreaterThan(0);
      expect(rr.steps[first].op.kind).toBe('write');
      expect(splitKRun(cfg(S, false), splitKOrder(cfg(S, false), 'sequential')).correct).toBe(true);
    }
    // S = 2, round robin: the first write that loses is warp 1's (output 0), and it destroys chunk 0.
    const rr2 = splitKRun(cfg(2, false), splitKOrder(cfg(2, false), 'round-robin'));
    const st = rr2.steps.find((s) => s.lost)!;
    expect([st.warp, st.j, st.lost]).toEqual([1, 0, [0]]);
    expect(describeSKStep(rr2, rr2.steps.indexOf(st), 0)).toContain('chunk 0\'s contribution is overwritten and lost');
  });
  it('the bytes read do not change with S; threads and atomics do', () => {
    const t = [1, 2, 4].map((S) => splitKTraffic(cfg(S, true)));
    expect(new Set(t.map((x) => x.wBytes)).size).toBe(1);
    expect(new Set(t.map((x) => x.xBytes)).size).toBe(1);
    expect(t.map((x) => x.threads)).toEqual([12, 24, 48]);
    expect(t.map((x) => x.atomics)).toEqual([0, 12, 24]);
  });
  it('the step text prints the chosen lane\'s index arithmetic', () => {
    const c = cfg(2, true);
    const r = splitKRun(c, splitKOrder(c, 'sequential'));
    // Warp 3: output 1, chunk 1, starting at 8. Its first step is a load.
    const n = r.steps.findIndex((s) => s.warp === 3);
    const txt = describeSKStep(r, n, 2);
    expect(txt).toContain('Warp 3 owns output j = 3 ÷ 2 = 1 and chunk c = 3 mod 2 = 1, which starts at c × 16 ÷ 2 = 8');
    expect(txt).toContain('Lane 2: W[1][8 + 2 + 0 × 4] = W[1][10]');
  });
});

describe('the book\'s tour (notes/ch22.md, pp. 533–536)', () => {
  const notes = fs.readFileSync(path.resolve(__dirname, '../../../notes/ch22.md'), 'utf8');
  const table = notes.slice(notes.indexOf('**The book\'s tour of patterns'), notes.indexOf('### §22.3'));
  const rowOf = (key: string) => {
    const line = table.split('\n').find((l) => l.startsWith('|') && l.includes(key));
    if (!line) throw new Error('no notes row for ' + key);
    return line.split('|').map((c) => c.trim()).slice(1, -1);
  };
  const book = TOUR.filter((r) => !r.ours);
  it('17 rows from the book, 4 of ours, ids unique', () => {
    expect(book.length).toBe(17);
    expect(TOUR.filter((r) => r.ours).length).toBe(4);
    expect(new Set(TOUR.map((r) => r.id)).size).toBe(TOUR.length);
  });
  it('every book row has the decomposition the notes give', () => {
    for (const r of book) {
      const [, decomp, why] = rowOf(r.notesKey!);
      if (decomp.startsWith('Depends')) {
        // Sorting: one notes row, three algorithms.
        if (r.id === 'oddeven') expect(why).toMatch(/Odd-even sort: the two views are equivalent/);
        if (r.id === 'mergesort') expect(why).toMatch(/Merge sort: each level is mainly a merge, hence output-centric/);
        if (r.id === 'radix') expect(why).toMatch(/Radix sort: each iteration is input-centric/);
        continue;
      }
      expect(decomp.toLowerCase().startsWith(r.decomp === 'output' ? 'output-centric' : 'input-centric'), r.id + ': ' + decomp).toBe(true);
    }
  });
  it('the traps: merge is load balance, BFS is idempotent, COO and CSC are input-centric, odd-even is either', () => {
    const t = Object.fromEntries(TOUR.map((r) => [r.id, r]));
    expect(t.merge.accept).toEqual(['balance']);
    expect(t.merge.inputCentricAtomics).toBe('one-writer');
    expect(rowOf(t.merge.notesKey!)[2]).toMatch(/load balance under thread coarsening/);
    expect(rowOf(t.merge.notesKey!)[2]).toMatch(/would not need atomics either/);
    for (const id of ['bfs-push', 'bfs-pull']) {
      expect(t[id].inputCentricAtomics).toBe('idempotent');
      expect(t[id].accept).not.toContain('atomics');
    }
    expect(rowOf(t['bfs-push'].notesKey!)[2]).toMatch(/idempotent and need no atomics/);
    expect([t.coo.decomp, t.csc.decomp]).toEqual(['input', 'input']);
    expect(t.coo.accept).toContain('balance');
    expect(t.coo.accept).not.toContain('atomics');
    expect(t.oddeven.decomp).toBe('either');
    expect(t.filter.why).toMatch(/scan is itself output-centric/);
    expect(t.scan.why).toMatch(/prints "Chapter 12"/);
    expect(rowOf(t.scan.notesKey!)[0]).toMatch(/book says "Chapter 12"/);
  });
  it('grading, and the labels the classification shows', () => {
    const merge = TOUR.find((r) => r.id === 'merge')!;
    expect(gradeCard(merge, 'output', 'atomics')).toEqual({ decomp: true, criterion: false });
    expect(gradeCard(merge, 'output', 'balance')).toEqual({ decomp: true, criterion: true });
    expect(Object.keys(DECOMP_LABEL)).toEqual(['output', 'input', 'either']);
    expect(Object.values(CRITERION_LABEL)).toEqual(['avoids atomics', 'exposes parallelism', 'inputs easy to find', 'load balance', 'atomics not needed either way']);
  });
});
