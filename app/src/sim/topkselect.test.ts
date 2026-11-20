import { describe, expect, it } from 'vitest';
import { bitsOf, keyAsc } from './floatkey';
import {
  BISECT_PASSES,
  BISECT_PER_KEY,
  HF_TIES,
  MAX_PASSES,
  SCENE_K,
  SCENE_ROW,
  SELECT_PER_KEY,
  SORT_PASSES,
  SORT_PER_KEY,
  STEP_PRESETS,
  WALK_EXAMPLE,
  bisectionTrace,
  costs,
  countStatus,
  keepTiesTopK,
  kthKey,
  maskAbove,
  measuredPairs,
  near3,
  prefixHex,
  referenceTopK,
  selectTrace,
  trafficRatio,
  walkBins,
  warpsOf,
} from './topkselect';

function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Rows that stress the select: ties, both zeros, shared top bits, negatives, infinities. */
function randomRow(rand: () => number): number[] {
  const n = 1 + Math.floor(rand() * 160);
  const kind = Math.floor(rand() * 6);
  return Array.from({ length: n }, () => {
    switch (kind) {
      case 0:
        return Math.fround(-10 + rand() * 20);
      case 1:
        return Math.round(rand() * 6) / 2 - 1; // heavy ties
      case 2:
        return [0, -0, 1, -1, 0.5][Math.floor(rand() * 5)]; // signed zeros and ties
      case 3:
        return near3(Math.floor(rand() * 256)); // top 24 bits shared
      case 4:
        return Math.fround(3 + Math.floor(rand() * 2 ** 15) * 2 ** -22); // top 16 bits shared
      default:
        return rand() < 0.05 ? (rand() < 0.5 ? Infinity : -Infinity) : Math.fround(-(rand() * 50)); // negatives, infinities
    }
  });
}

describe('radix select equals a stable sort and cut', () => {
  it('on 4,000 seeded rows, every block size and warp width', () => {
    const rand = mulberry(2026);
    for (let t = 0; t < 4000; t++) {
      const row = randomRow(rand);
      const k = 1 + Math.floor(rand() * row.length);
      const T = [16, 32, 64][t % 3];
      const W = [8, 32, 64][Math.floor(t / 3) % 3];
      const tr = selectTrace(row, k, T, W);
      const got = tr.out.map((o) => o.col);
      expect(got).toEqual(referenceTopK(row, k));
      expect(tr.out.length).toBe(k);
      // Slots are 0..k-1 in column order.
      const slots = tr.gather.flatMap((g) => g.lanes.filter((l) => l.keep).map((l) => l.slot));
      expect(slots).toEqual(Array.from({ length: k }, (_, i) => i));
      expect(tr.greater + tr.need).toBe(k);
      expect(tr.passes.length).toBeGreaterThanOrEqual(1);
      expect(tr.passes.length).toBeLessThanOrEqual(MAX_PASSES);
      if (tr.exact) expect(tr.prefix).toBe(kthKey(row, k));
    }
  });
  it('-0.0 and +0.0 are one key, and ties go to the lowest columns', () => {
    expect(keyAsc(bitsOf(-0))).toBe(keyAsc(bitsOf(0)));
    const tr = selectTrace([-0, 1, 0, -0, 0, -1], 3);
    expect(tr.out.map((o) => o.col)).toEqual([0, 1, 2]);
    expect(tr.exact).toBe(true);
    expect(tr.need).toBe(2);
  });
});

describe('pass counts', () => {
  const passesOf = (id: string) => {
    const p = STEP_PRESETS.find((x) => x.id === id)!;
    return selectTrace(p.logits, p.k).passes.length;
  };
  it('one pass when the crossing bin must be taken whole', () => {
    expect(passesOf('peaked')).toBe(1);
    expect(selectTrace([1, 2, 3, 4], 4).passes.length).toBe(1);
    expect(selectTrace([-5, 9.5, 2.5], 1).passes.length).toBe(1);
  });
  it('two and three passes on constructed rows', () => {
    // 5.0 and 2.5 share the top byte (0xC0) and differ in the second.
    expect(selectTrace([5, 2.5, -1], 1).passes.length).toBe(2);
    // Keys 0xC0400200 and 0xC0400100 share the top 16 bits and differ in the third byte.
    expect(selectTrace([near3(0x200), near3(0x100), near3(0x100), -1], 1).passes.length).toBe(3);
  });
  it('four passes when keys share their top 24 bits or tie at the k-th', () => {
    expect(passesOf('top24')).toBe(4);
    expect(passesOf('ties')).toBe(4);
    expect(passesOf('scene')).toBe(4);
    expect(passesOf('top8')).toBeGreaterThanOrEqual(2);
    expect(passesOf('top16')).toBeGreaterThanOrEqual(3);
  });
  it('shared top bits leave pass 1 (and 2, 3) with a single bin', () => {
    const p = STEP_PRESETS.find((x) => x.id === 'top24')!;
    const tr = selectTrace(p.logits, p.k);
    for (const ps of tr.passes.slice(0, 3)) {
      expect(ps.hist.filter((c) => c > 0).length).toBe(1);
      expect(ps.above).toBe(0);
    }
  });
});

describe("the scene's row", () => {
  const tr = selectTrace(SCENE_ROW, SCENE_K, 16, 32);
  it('is 64 logits, k = 12, four passes', () => {
    expect(SCENE_ROW.length).toBe(64);
    expect(tr.passes.length).toBe(4);
    expect(tr.exact).toBe(true);
  });
  it('places 6, 2, 1 and 1 keys above the crossing bins, then needs 2 of 3 ties', () => {
    expect(tr.passes.map((p) => p.above)).toEqual([6, 2, 1, 1]);
    expect(tr.passes.map((p) => p.bin)).toEqual([0xc0, 0x40, 0x00, 150]);
    expect(tr.passes.map((p) => p.inBin)).toEqual([13, 8, 7, 3]);
    expect(tr.need).toBe(2);
    expect(tr.equal).toBe(3);
    expect(tr.prefix).toBe(keyAsc(bitsOf(near3(150))));
    expect(prefixHex(tr.passes[0].prefixOut, tr.passes[0].maskOut)).toBe('0xC0······');
  });
  it('pass 1 never diverges; pass 2 does', () => {
    expect(tr.passes[0].divergedWarps).toBe(0);
    expect(tr.passes[0].rounds.every((r) => r.lanes.every((l) => l.match))).toBe(true);
    expect(tr.passes[1].divergedWarps).toBeGreaterThan(0);
    expect(tr.passes[1].candidates).toBe(13);
  });
  it('thread t reads columns t, t + 16, t + 32, t + 48', () => {
    expect(tr.passes[0].rounds.map((r) => r.lanes[5].col)).toEqual([5, 21, 37, 53]);
  });
  it('decided in, out and undecided after each pass: 6/45/13, 8/48/8, 9/48/7, 10/51/3', () => {
    expect([1, 2, 3, 4].map((n) => countStatus(tr, n))).toEqual([
      { in: 6, out: 45, open: 13 },
      { in: 8, out: 48, open: 8 },
      { in: 9, out: 48, open: 7 },
      { in: 10, out: 51, open: 3 },
    ]);
    expect(countStatus(tr, 0)).toEqual({ in: 0, out: 0, open: 64 });
  });
  it('the gather keeps the two lowest-column ties and cuts the third', () => {
    const ties = tr.gather.flatMap((g) => g.lanes.filter((l) => l.cls === 'tie'));
    expect(ties.map((l) => [l.col, l.keep])).toEqual([[0, true], [22, true], [23, false]]);
  });
  it('the histogram after the last round is the pass histogram', () => {
    for (const p of tr.passes) expect(p.rounds[p.rounds.length - 1].histAfter).toEqual(p.hist);
  });
});

describe('the questions', () => {
  it('q-passes: at most 4 passes with 8-bit digits over 32-bit keys', () => {
    expect(MAX_PASSES).toBe(4);
  });
  it('q-bisect: 32 counting passes, eight times the select', () => {
    expect(BISECT_PASSES).toBe(32);
    expect(BISECT_PASSES / MAX_PASSES).toBe(8);
    const rand = mulberry(5);
    for (let t = 0; t < 200; t++) {
      const row = randomRow(rand);
      const k = 1 + Math.floor(rand() * row.length);
      const b = bisectionTrace(row, k);
      expect(b.steps.length).toBe(32);
      expect(b.threshold).toBe(kthKey(row, k));
    }
  });
  it('q-bytes-ratio: 80 / 20 = 4', () => {
    expect(SORT_PASSES).toBe(4);
    expect(SORT_PER_KEY).toBe(80);
    expect(SELECT_PER_KEY).toBe(20);
    expect(SORT_PER_KEY / SELECT_PER_KEY).toBe(4);
    expect(BISECT_PER_KEY).toBe(132);
  });
  it('the bin walk: bins of 2, 5, 30, 100 at k = 20 leave 13 needed from the third', () => {
    const w = walkBins(WALK_EXAMPLE.hist, WALK_EXAMPLE.k);
    expect(w.steps.map((s) => s.running)).toEqual([2, 7, 37]);
    expect(w.bin).toBe(150);
    expect(w.above).toBe(7);
    expect(w.stillNeeded).toBe(13);
  });
  it("q-hf-ties: transformers keeps 3, M5's select exactly tokens 0 and 1", () => {
    expect(keepTiesTopK(HF_TIES.logits, HF_TIES.k)).toEqual([0, 1, 2]);
    expect(selectTrace(HF_TIES.logits, HF_TIES.k).out.map((o) => o.col)).toEqual([0, 1]);
  });
  it('q-k1: k = 1 is argmax with ties to the lowest column', () => {
    const rand = mulberry(9);
    for (let t = 0; t < 300; t++) {
      const row = randomRow(rand);
      const keys = row.map((v) => keyAsc(bitsOf(v)));
      const best = keys.indexOf(Math.max(...keys));
      expect(selectTrace(row, 1).out.map((o) => o.col)).toEqual([best]);
    }
  });
});

describe('cost', () => {
  it("the select's traffic barely depends on k; the sort's not at all", () => {
    const V = 151936;
    const a = costs(V, 1, 1);
    const b = costs(V, 4000, 1);
    expect(a[0].perRow).toBe(b[0].perRow);
    expect(b[1].perRow / a[1].perRow).toBeLessThan(1.011);
    expect(trafficRatio(V, 40)).toBeGreaterThan(3.99);
    expect(trafficRatio(V, 4000)).toBeGreaterThan(3.9);
    expect(trafficRatio(V, V)).toBeCloseTo(80 / 28, 6);
  });
  it('bisection moves more than either', () => {
    const [sort, sel, bis] = costs(50257, 50, 8);
    expect(bis.perRow).toBeGreaterThan(sort.perRow);
    expect(bis.perRow).toBeGreaterThan(sel.perRow);
    expect(sort.total).toBe(sort.perRow * 8);
    expect(sel.blocks).toBe(8);
  });
});

describe('helpers', () => {
  it('masks and warps', () => {
    expect(maskAbove(32)).toBe(0);
    expect(maskAbove(24)).toBe(0xff000000);
    expect(maskAbove(0)).toBe(0xffffffff);
    expect(warpsOf(16, 8).length).toBe(2);
    expect(warpsOf(16, 64)[0].threads.filter((t) => t != null).length).toBe(16);
  });
  it('pairs lab X1.1 benchmark rows by shape', () => {
    const rows = [
      { name: 'radix_select', rows: 64, cols: 151936, k: 40, seconds: 0.002 },
      { name: 'full_sort', rows: 64, cols: 151936, k: 40, seconds: 0.01 },
      { name: 'cpu_partial_sort', rows: 64, cols: 151936, k: 40, seconds: 0.5 },
      { name: 'radix_select', rows: 1, cols: 151936, k: 40, seconds: 0.001 },
    ];
    const m = measuredPairs(rows);
    expect(m.length).toBe(1);
    expect(m[0].speedup).toBeCloseTo(5, 9);
  });
});
