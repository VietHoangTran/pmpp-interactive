import { describe, expect, it } from 'vitest';
import { index } from './histcoarsen';
import {
  CONSTANT_VALUE,
  IMAGE_PIXELS,
  PRESET_ORDER,
  SKY_WIDTH,
  WORKED,
  WORKED_COARSE,
  blockRun,
  elementIndex,
  makeImage,
  summary,
  trace,
  warpReference,
  workedImage,
  type Partitioning,
  type WarpConfig,
} from './histruns';

const BOTH: Partitioning[] = ['interleaved', 'contiguous'];

describe('line 15 is line 12 of histcoarsen.ts', () => {
  it('for every partitioning, block, thread and iteration tried', () => {
    for (const p of BOTH)
      for (const [B, C] of [[8, 3], [256, 4], [96, 7]])
        for (const b of [0, 1, 5])
          for (let t = 0; t < B; t += 7)
            for (let c = 0; c < C; c++) expect(elementIndex(p, b, t, c, B, C)).toBe(index(p, b, t, c, B, C));
  });
});

describe('the worked warp (q-runs-count)', () => {
  for (const p of BOTH) {
    const B = 32;
    const img = workedImage(p, B);
    const cfg: WarpConfig = { p, n: img.length, blockDim: B, coarse: WORKED_COARSE, block: 0, firstThread: 0, lanes: 8 };
    const tr = trace(img, cfg);
    const end = tr[tr.length - 1];
    it(`${p}: every lane sees its own sequence`, () => {
      for (let l = 0; l < 8; l++)
        expect(tr.slice(1, WORKED_COARSE + 1).map((s) => s.lanes[l].value)).toEqual([...WORKED[l]]);
    });
    it(`${p}: 7, 7, 7, 3, 3, 7, 7, 9 costs 4 atomics, three in the loop and the final flush`, () => {
      expect(end.lanes[0].atomics).toBe(4);
      expect(tr.filter((s) => s.lanes[0].branch === 'flush').map((s) => s.lanes[0].amount)).toEqual([3, 2, 2]);
      expect(end.lanes[0].branch).toBe('final');
      expect(end.lanes[0].amount).toBe(1);
    });
    it(`${p}: atomics per lane are the runs in its sequence`, () => {
      expect(end.lanes.map((l) => l.atomics)).toEqual([4, 1, 8, 4, 2, 1, 8, 4]);
      expect(end.atomics).toBe(32);
      expect(end.elements).toBe(64);
    });
    it(`${p}: flushed amounts add up to what the lanes read, bin by bin`, () => {
      expect(end.counts).toEqual(warpReference(img, cfg));
    });
    it(`${p}: divergence is counted where lanes take both sides of the branch`, () => {
      // c = 1: lane 0 extends (7, 7), lane 2 flushes (5 -> 6). The first loads and the final flush are not loop steps.
      expect(tr[2].c).toBe(1);
      expect(tr[2].divergent).toBe(true);
      expect(tr[1].divergent).toBe(false);
      expect(end.divergent).toBe(false);
      expect(end.divergentSteps).toBe(7);
    });
  }
  it('lanes that agree do not diverge', () => {
    const img = new Uint8Array(8 * 4).fill(3);
    const tr = trace(img, { p: 'interleaved', n: 32, blockDim: 8, coarse: 4, block: 0, firstThread: 0, lanes: 8 });
    expect(tr.some((s) => s.divergent)).toBe(false);
    expect(tr[tr.length - 1].atomics).toBe(8);
  });
  it('a lane with no first element stays out, and a lane past the end in one iteration skips it', () => {
    // 8 threads, factor 4, n = 8·3 + 5: in iteration 3 only lanes 0 to 4 have an element.
    const img = new Uint8Array(29).fill(1);
    const tr = trace(img, { p: 'interleaved', n: 29, blockDim: 8, coarse: 4, block: 0, firstThread: 0, lanes: 8 });
    expect(tr[4].lanes.map((l) => l.branch)).toEqual(['extend', 'extend', 'extend', 'extend', 'extend', 'inactive', 'inactive', 'inactive']);
    expect(tr[tr.length - 1].counts[1]).toBe(29);
    const short = new Uint8Array(5).fill(2);
    const t2 = trace(short, { p: 'interleaved', n: 5, blockDim: 8, coarse: 4, block: 0, firstThread: 0, lanes: 8 });
    expect(t2[1].lanes.map((l) => l.branch)).toEqual(['first', 'first', 'first', 'first', 'first', 'inactive', 'inactive', 'inactive']);
    expect(t2[t2.length - 1].lanes[7].branch).toBe('inactive');
    expect(t2[t2.length - 1].atomics).toBe(5);
  });
});

describe('the presets', () => {
  it('are deterministic', () => {
    for (const id of PRESET_ORDER) expect(makeImage(id)).toEqual(makeImage(id));
  });
  it('sky rows are constant and neighbouring rows differ', () => {
    const img = makeImage('sky');
    for (let r = 0; r < IMAGE_PIXELS / SKY_WIDTH; r++) {
      const row = img.subarray(r * SKY_WIDTH, (r + 1) * SKY_WIDTH);
      expect(row.every((v) => v === row[0])).toBe(true);
      if (r) expect(row[0]).not.toBe(img[(r - 1) * SKY_WIDTH]);
    }
  });
  it('stripes are runs of exactly 8', () => {
    const img = makeImage('stripes');
    for (let i = 1; i < img.length; i++) expect(img[i] === img[i - 1]).toBe(i % 8 !== 0);
  });
  it('token ids: fewer than one neighbouring pair in twenty is equal (the lesson says so)', () => {
    const img = makeImage('tokens');
    let eq = 0;
    for (let i = 1; i < img.length; i++) if (img[i] === img[i - 1]) eq++;
    expect(eq / (img.length - 1)).toBeGreaterThan(0.03);
    expect(eq / (img.length - 1)).toBeLessThan(0.05);
  });
});

describe('atomics per element (the summary table)', () => {
  it('flushes always add up to a plain count, for every preset, partitioning and size, partial blocks included', () => {
    for (const id of PRESET_ORDER)
      for (const p of BOTH)
        for (const [B, C, W] of [[256, 16, 32], [64, 7, 64], [32, 1, 32], [128, 64, 32]]) {
          const s = summary(makeImage(id, 10_000), p, B, C, W);
          expect(s.match).toBe(true);
          expect(s.elements).toBe(10_000);
        }
  });
  it('constant image: one atomic per thread, 1 / COARSE_FACTOR per element, no divergence', () => {
    for (const C of [1, 4, 16, 64]) {
      const s = summary(makeImage('constant'), 'interleaved', 256, C, 32);
      expect(s.perElement).toBe(1 / C);
      expect(s.divergentSteps).toBe(0);
    }
    expect(makeImage('constant')[0]).toBe(CONSTANT_VALUE);
  });
  it('noise: about one atomic per element', () => {
    const s = summary(makeImage('noise'), 'interleaved', 256, 16, 32);
    expect(s.perElement).toBeGreaterThan(0.99);
    expect(s.perElement).toBeLessThanOrEqual(1);
  });
  it('noise still splits some warps: a 32-lane warp has a lane that extends in roughly 1 − (255/256)^32 of its steps', () => {
    const s = summary(makeImage('noise'), 'interleaved', 256, 16, 32);
    const expected = 1 - (255 / 256) ** 32;
    expect(s.divergentShare!).toBeGreaterThan(expected - 0.03);
    expect(s.divergentShare!).toBeLessThan(expected + 0.03);
  });
  it('sky, blockDim 256, factor 16 (p-runs-fraction): a block covers one row, so 1 atomic per 16 elements', () => {
    const s = summary(makeImage('sky'), 'interleaved', 256, 16, 32);
    expect(s.perElement).toBe(1 / 16);
    expect(s.atomics).toBe(IMAGE_PIXELS / 16);
  });
  it('sky, factor 32: a block covers two rows, so each thread flushes once per row', () => {
    expect(summary(makeImage('sky'), 'interleaved', 256, 32, 32).perElement).toBe(2 / 32);
  });
  it('stripes merge under contiguous partitioning but not under interleaved with blockDim 256', () => {
    const img = makeImage('stripes');
    const inter = summary(img, 'interleaved', 256, 16, 32);
    const cont = summary(img, 'contiguous', 256, 16, 32);
    expect(inter.perElement).toBeGreaterThan(0.99);
    expect(cont.perElement).toBe(2 / 16);
  });
  it('token ids barely merge either way, and split warps more than noise does (the mixed case)', () => {
    const img = makeImage('tokens');
    for (const p of BOTH) expect(summary(img, p, 256, 16, 32).perElement).toBeGreaterThan(0.94);
    const tok = summary(img, 'interleaved', 256, 16, 32).divergentShare!;
    expect(tok).toBeGreaterThan(0.5);
    expect(tok).toBeGreaterThan(3 * summary(makeImage('noise'), 'interleaved', 256, 16, 32).divergentShare!);
  });
  it('factor 1 has no loop, so no divergence to count', () => {
    const s = summary(makeImage('noise'), 'interleaved', 256, 1, 32);
    expect(s.divergentShare).toBeNull();
    expect(s.perElement).toBe(1);
  });
  it('one block on its own agrees with the whole-image sum', () => {
    const img = makeImage('stripes');
    let atomics = 0;
    for (let b = 0; b < IMAGE_PIXELS / (256 * 16); b++) atomics += blockRun(img, 'contiguous', 256, 16, b, 32).atomics;
    expect(atomics).toBe(summary(img, 'contiguous', 256, 16, 32).atomics);
  });
});
