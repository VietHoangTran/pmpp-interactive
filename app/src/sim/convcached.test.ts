import { describe, expect, it } from 'vitest';
import {
  blockHistogram,
  classifyTaps,
  footprint,
  guard,
  interiorSpec,
  intensity,
  makeProblem,
  readsPerOutput,
  reference,
  runCached,
  warpSplits,
  type Problem,
} from './convcached';

describe('an interior 32 x 32 block at radius 2 (the numbers section 7.6 states)', () => {
  const h = blockHistogram(interiorSpec(32, 2));
  it('has 25,600 taps: 23,716 read shared memory, 1,884 read global memory, none are ghosts', () => {
    expect(h.taps).toBe(25600);
    expect(h.shared).toBe(23716);
    expect(h.global).toBe(1884);
    expect(h.ghost).toBe(0);
    expect(h.global / h.taps).toBeCloseTo(0.0736, 4);
  });
  it('the in-tile taps per axis sum to 154 over the 32 threads, and 154 squared is the shared count', () => {
    let perAxis = 0;
    for (let t = 0; t < 32; t++) for (let f = 0; f < 5; f++) perAxis += guard(t, 2, f, 32).intended ? 1 : 0;
    expect(perAxis).toBe(154);
    expect(perAxis * perAxis).toBe(h.shared);
  });
  it('reads 272 different halo elements: the 36 x 36 footprint less the 32 x 32 tile', () => {
    expect(h.distinctGlobal).toBe(36 * 36 - 32 * 32);
  });
});

describe('reads of N per output and the intensity they allow (m = 5)', () => {
  const b = readsPerOutput(32, 2);
  it('every global tap from DRAM: (1024 + 1884) / 1024 = 2.84', () => {
    expect(b.allTaps).toBeCloseTo(2.84, 2);
    expect(b.allTaps).toBe((1024 + 1884) / 1024);
  });
  it('each halo element once per block: 1296 / 1024, the book formula at t = 32 (Fig. 7.14: 5.52 FLOP/B)', () => {
    expect(b.perBlock).toBe(1296 / 1024);
    expect(intensity(b.perBlock, 2)).toBeCloseTo(5.52, 2);
  });
  it('Fig. 7.12 with a 32-wide input tile: 1024 / 784 = 1.31 reads, 5.42 FLOP/B as the book prints (p. 177)', () => {
    expect(b.fig712).toBeCloseTo(1.31, 2);
    expect(intensity(b.fig712, 2)).toBeCloseTo(5.42, 2);
  });
  it('the ideal: one read per output, m^2/4 = 6.25 FLOP/B (p. 167)', () => {
    expect(b.ideal).toBe(1);
    expect(intensity(1, 2)).toBe(6.25);
  });
  it('every tap from DRAM gives 3.26 FLOP/B, below Fig. 7.12', () => {
    expect(intensity(b.allTaps, 2)).toBeCloseTo(3.26, 2);
  });
});

describe('the printed guard in unsigned arithmetic (facts ch07 A8)', () => {
  it('tx = 0, R = 2, fCol = 0 wraps to 4294967294: ">= 0" is true, "< TILE_DIM" is false', () => {
    const g = guard(0, 2, 0, 32);
    expect(g.u32).toBe(4294967294);
    expect(g.signed).toBe(-2);
    expect(g.geZero).toBe(true);
    expect(g.ltTile).toBe(false);
    expect(g.intended).toBe(false);
  });
  it('for every tile, radius, thread and tap, ">= 0" is always true and "< TILE_DIM" alone gives the intended test', () => {
    for (const T of [4, 8, 16, 32])
      for (let R = 1; R <= 3; R++)
        for (let t = 0; t < T; t++)
          for (let f = 0; f <= 2 * R; f++) {
            const g = guard(t, R, f, T);
            expect(g.geZero).toBe(true);
            expect(g.ltTile).toBe(g.intended);
          }
  });
});

describe('printed line 21: N_s[ty+fRow][tx+fCol]', () => {
  it('every shared tap of an interior thread reads the slot R rows and R columns down and right of the right one', () => {
    for (const R of [1, 2, 3]) {
      const t = classifyTaps({ ...interiorSpec(16, R), tx: 7, ty: 5 });
      const shared = t.taps.filter((p) => p.kind === 'shared');
      expect(shared.length).toBeGreaterThan(0);
      for (const p of shared) {
        expect(p.printed![0] - p.slot![0]).toBe(R);
        expect(p.printed![1] - p.slot![1]).toBe(R);
        expect(p.wrongElement).toBe(true);
      }
    }
  });
  it('the printed index reaches TILE_DIM + R - 1, past the tile, and some reads leave N_s altogether', () => {
    const T = 32;
    const R = 2;
    let max = 0;
    for (let ty = 0; ty < T; ty++)
      for (let tx = 0; tx < T; tx++)
        for (const p of classifyTaps({ ...interiorSpec(T, R), tx, ty }).taps)
          if (p.printed) max = Math.max(max, p.printed[0], p.printed[1]);
    expect(max).toBe(T + R - 1);
    const h = blockHistogram(interiorSpec(T, R));
    expect(h.printedWrong).toBe(h.shared);
    expect(h.printedPastTile).toBeGreaterThan(0);
    expect(h.printedOutside).toBeGreaterThan(0);
    expect(h.printedOutside).toBeLessThan(h.printedPastTile);
  });
  it('thread (0, 0) of an interior block, radius 2: 9 shared taps, 16 global, the first shared tap is fRow = fCol = 2', () => {
    const t = classifyTaps({ ...interiorSpec(32, 2), tx: 0, ty: 0 });
    expect(t.taps.filter((p) => p.kind === 'shared').length).toBe(9);
    expect(t.taps.filter((p) => p.kind === 'global').length).toBe(16);
    const first = t.taps.find((p) => p.kind === 'shared')!;
    expect([first.fRow, first.fCol]).toEqual([2, 2]);
    expect(first.slot).toEqual([0, 0]);
    expect(first.printed).toEqual([2, 2]);
  });
});

describe('warps and the in-tile test', () => {
  it('tile 32, radius 2, warps of 32: all 32 warps diverge; edge rows on 12 or 16 taps, the rest on 20', () => {
    const w = warpSplits(32, 2, 32);
    expect(w.warps.length).toBe(32);
    expect(w.diverging).toBe(32);
    expect(w.warps.map((x) => x.splitTaps)).toEqual([12, 16, ...Array(28).fill(20), 16, 12]);
  });
  it('warps of 64 (two rows each): all 16 diverge', () => {
    const w = warpSplits(32, 2, 64);
    expect(w.warps.length).toBe(16);
    expect(w.diverging).toBe(16);
  });
  it('a 4 x 4 block is one partial warp of 16 lanes', () => {
    const w = warpSplits(4, 1, 32);
    expect(w.warps).toHaveLength(1);
    expect(w.warps[0].lanes).toBe(16);
    expect(w.warps[0].diverges).toBe(true);
  });
});

describe('blocks at the edge of the image', () => {
  it('a corner block skips ghost taps; a tile that overhangs the image reads ghost zeros from shared memory', () => {
    const corner = blockHistogram({ tile: 8, r: 2, width: 29, height: 27, bx: 0, by: 0 });
    expect(corner.ghost).toBeGreaterThan(0);
    expect(corner.zeroInTile).toBe(0);
    const last = blockHistogram({ tile: 8, r: 2, width: 29, height: 27, bx: 3, by: 3 });
    expect(last.active).toBe(5 * 3);
    expect(last.zeroInTile).toBeGreaterThan(0);
  });
  it('the footprint marks the tile shared and the in-image ring global, and counts every tap once', () => {
    const spec = interiorSpec(8, 2);
    const fp = footprint(spec);
    expect(fp.size).toBe(12);
    expect(fp.kind[2 * 12 + 2]).toBe('shared');
    expect(fp.kind[0]).toBe('global');
    expect(fp.taps.reduce((a, b) => a + b, 0)).toBe(blockHistogram(spec).taps);
  });
});

describe('running the whole kernel', () => {
  it('reproduces the book\'s 2D example: P[2][2] = 321 and P[1][0] = 112 (pp. 162 to 164)', () => {
    const N = [
      1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8, 9, 4, 5, 6, 7, 8, 5, 6, 5, 6, 7, 8, 5, 6, 7, 6, 7, 8, 9, 0,
      1, 2, 7, 8, 9, 0, 1, 2, 3,
    ];
    const F = [1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4, 5, 4, 3, 2, 3, 4, 3, 2, 1, 2, 3, 2, 1];
    const prob: Problem = { width: 7, height: 7, r: 2, N, F, ref: reference(N, F, 7, 7, 2) };
    expect(prob.ref[2 * 7 + 2]).toBe(321);
    expect(prob.ref[1 * 7 + 0]).toBe(112);
    for (const tile of [4, 8]) {
      const run = runCached(prob, { tile });
      expect(run.correct).toBe(true);
      expect(run.P[2 * 7 + 2]).toBe(321);
      expect(run.P[7]).toBe(112);
    }
  });
  it('corrected, it matches the direct convolution for ragged sizes, every tile and radius', () => {
    for (const tile of [4, 8, 16])
      for (const r of [1, 2, 3])
        for (const [w, h] of [[1, 1], [7, 5], [3 * tile + 5, 3 * tile + 3], [2 * tile, tile + 1]]) {
          const run = runCached(makeProblem(w, h, r, w * 31 + h), { tile });
          expect(run.correct).toBe(true);
          expect(run.outsideReads).toBe(0);
        }
  });
  it('as printed, almost every output is wrong and some reads leave N_s', () => {
    const prob = makeProblem(3 * 32 + 5, 3 * 32 + 3, 2);
    const run = runCached(prob, { tile: 32, asPrinted: true });
    expect(run.correct).toBe(false);
    expect(run.wrong.length).toBeGreaterThan(0.95 * prob.width * prob.height);
    expect(run.outsideReads).toBeGreaterThan(0);
    expect(run.pastTileReads).toBeGreaterThan(0);
  });
});
