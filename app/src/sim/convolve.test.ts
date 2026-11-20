import { describe, expect, it } from 'vitest';
import {
  ASYMMETRIC,
  BOOK_1D,
  BOOK_2D,
  EX2,
  EX3_FILTERS,
  basicLineNumbers,
  basicTaps,
  conv1d,
  conv2d,
  firstKernel,
  ghostCounts1d,
  ghostCounts2d,
  ghostFormula1d,
  ghostFormula2d,
  laneActivity,
  mapThread,
  ownerOf,
  partOf,
  patch2d,
  pyramidFilter,
  radiusOf,
  skippedMap,
  sourceIndex,
  stripsOf,
  toyImage,
  valid1d,
  warpDivergence,
  warpInfo,
  warpsInCorner,
  window1d,
} from './convolve';

const { x, f } = BOOK_1D;

describe('1D convolution, the book\'s worked numbers (Figs. 7.1 to 7.3)', () => {
  it('y[2] = 52 with products 8, 6, 25, 12, 1', () => {
    const w = window1d(x, f, 2);
    expect(w.sum).toBe(52);
    expect(w.taps.map((t) => t.product)).toEqual([8, 6, 25, 12, 1]);
    expect(w.touchesGhost).toBe(false);
  });
  it('y[3] = 47 with products 2, 15, 20, 3, 7', () => {
    const w = window1d(x, f, 3);
    expect(w.sum).toBe(47);
    expect(w.taps.map((t) => t.product)).toEqual([2, 15, 20, 3, 7]);
  });
  it('y[1] = 53 with one zero ghost cell (products 0, 24, 10, 15, 4)', () => {
    const w = window1d(x, f, 1);
    expect(w.sum).toBe(53);
    expect(w.taps.map((t) => t.product)).toEqual([0, 24, 10, 15, 4]);
    expect(w.taps.filter((t) => t.ghost).map((t) => t.at)).toEqual([-1]);
  });
  it('Exercise 1: y[0] = 51', () => {
    expect(window1d(x, f, 0).sum).toBe(51);
  });
  it('the whole row with zero ghost cells', () => {
    // y[4] = 5 + 12 + 5 + 21 + 3; y[5] = 4 + 3 + 35 + 9 + 0; y[6] = 1 + 21 + 15 + 0 + 0
    expect(conv1d(x, f)).toEqual([51, 53, 52, 47, 46, 51, 37]);
  });
});

describe('boundary policies', () => {
  it('maps a missing index to a zero, the nearest edge, or the other end', () => {
    expect(sourceIndex(-1, 7, 'zero')).toBeNull();
    expect(sourceIndex(-2, 7, 'replicate')).toBe(0);
    expect(sourceIndex(8, 7, 'replicate')).toBe(6);
    expect(sourceIndex(-1, 7, 'wrap')).toBe(6);
    expect(sourceIndex(8, 7, 'wrap')).toBe(1);
    expect(sourceIndex(3, 7, 'wrap')).toBe(3);
  });
  it('only the first and last r outputs depend on the policy (q-policy: y[0], y[1], y[5], y[6])', () => {
    const zero = conv1d(x, f, { policy: 'zero' });
    for (const policy of ['replicate', 'wrap'] as const) {
      const other = conv1d(x, f, { policy });
      const changed = other.map((v, i) => (v !== zero[i] ? i : -1)).filter((i) => i >= 0);
      expect(changed).toEqual([0, 1, 5, 6]);
    }
    const touched = x.map((_, i) => (window1d(x, f, i).touchesGhost ? i : -1)).filter((i) => i >= 0);
    expect(touched).toEqual([0, 1, 5, 6]);
  });
  it('replicate and wrap give these values at the edges', () => {
    // replicate: y[0] = 1*8 + 3*8 + 5*8 + 3*2 + 1*5
    expect(window1d(x, f, 0, { policy: 'replicate' }).sum).toBe(8 + 24 + 40 + 6 + 5);
    // wrap: y[0] = 1*7 + 3*3 + 5*8 + 3*2 + 1*5
    expect(window1d(x, f, 0, { policy: 'wrap' }).sum).toBe(7 + 9 + 40 + 6 + 5);
  });
});

describe('Exercise 2 and the flip', () => {
  it('the book\'s form with zero ghost cells gives 8, 21, 13, 20, 7', () => {
    expect(conv1d(EX2.x, EX2.f)).toEqual([8, 21, 13, 20, 7]);
  });
  it('flipping the filter (textbook convolution) gives 6, 23, 11, 20, 11', () => {
    expect(conv1d(EX2.x, EX2.f, { flip: true })).toEqual([6, 23, 11, 20, 11]);
  });
  it('replicating the edge gives 16, 21, 13, 20, 19 (a distractor in q-ex2)', () => {
    expect(conv1d(EX2.x, EX2.f, { policy: 'replicate' })).toEqual([16, 21, 13, 20, 19]);
  });
  it('no ghost cells (PyTorch padding=0) gives the three inner outputs 21, 13, 20', () => {
    expect(valid1d(EX2.x, EX2.f)).toEqual([21, 13, 20]);
  });
  it('a flip changes nothing for a symmetric filter and changes outputs for an asymmetric one', () => {
    expect(conv1d(x, f, { flip: true })).toEqual(conv1d(x, f));
    expect(conv1d(x, ASYMMETRIC, { flip: true })).not.toEqual(conv1d(x, ASYMMETRIC));
  });
});

describe('Exercise 3', () => {
  const ramp = [10, 20, 30, 40, 50];
  const by = (k: string) => EX3_FILTERS.find((e) => e.key === k)!.f;
  it('(a) is the identity', () => {
    expect(conv1d(ramp, by('a'))).toEqual(ramp);
  });
  it('(b) [0,0,1] gives y[i] = x[i+1], a shift left', () => {
    expect(conv1d(ramp, by('b'))).toEqual([20, 30, 40, 50, 0]);
  });
  it('(c) [1,0,0] gives y[i] = x[i-1], a shift right', () => {
    expect(conv1d(ramp, by('c'))).toEqual([0, 10, 20, 30, 40]);
  });
  it('under a flip, (b) and (c) swap', () => {
    expect(conv1d(ramp, by('b'), { flip: true })).toEqual(conv1d(ramp, by('c')));
  });
  it('(d) is a central difference: 10 per step inside a ramp of slope 10', () => {
    expect(conv1d(ramp, by('d')).slice(1, 4)).toEqual([10, 10, 10]);
  });
  it('(e) is a 3-point moving average', () => {
    expect(conv1d(ramp, by('e'))[2]).toBeCloseTo(30, 12);
  });
});

describe('2D convolution, Figs. 7.4 and 7.5', () => {
  const { N, F } = BOOK_2D;
  it('P[2][2] = 321, with row sums 27, 56, 95, 84, 59', () => {
    const p = patch2d(N, F, 2, 2);
    expect(p.sum).toBe(321);
    expect(p.ghostTaps).toBe(0);
    const rows = [0, 1, 2, 3, 4].map((k) => p.taps.slice(k * 5, k * 5 + 5).reduce((s, t) => s + t.product, 0));
    expect(rows).toEqual([27, 56, 95, 84, 59]);
    expect(p.taps.slice(10, 15).map((t) => t.product)).toEqual([9, 16, 25, 24, 21]);
  });
  it('P[1][0] = 112, one ghost row and two ghost columns', () => {
    const p = patch2d(N, F, 1, 0);
    expect(p.sum).toBe(112);
    expect(p.ghostTaps).toBe(5 + 4 * 2);
    expect(p.taps.map((t) => t.product)).toEqual([0, 0, 0, 0, 0, 0, 0, 4, 6, 6, 0, 0, 10, 12, 12, 0, 0, 12, 12, 10, 0, 0, 12, 10, 6]);
  });
  it('conv2d agrees with patch2d everywhere', () => {
    const P = conv2d(N, F);
    expect(P.data[2 * 7 + 2]).toBe(321);
    expect(P.data[1 * 7 + 0]).toBe(112);
  });
  it('pyramidFilter(2) is the book\'s filter', () => {
    expect(pyramidFilter(2).data).toEqual(BOOK_2D.F.data);
    expect(pyramidFilter(1).data).toEqual([1, 2, 1, 2, 3, 2, 1, 2, 1]);
  });
});

describe('Exercises 4 to 6: counting ghost cells and multiplications', () => {
  it('1D with Fig. 7.1\'s sizes: 4 ghost cells, 35 taps, 29 real ones', () => {
    expect(ghostCounts1d(7, 5)).toEqual({ ghostCells: 4, allTaps: 35, realTaps: 29 });
  });
  it('1D closed forms equal brute force for n >= r', () => {
    for (let m = 1; m <= 15; m += 2)
      for (let n = radiusOf(m) || 1; n <= 40; n++) expect(ghostFormula1d(n, m)).toEqual(ghostCounts1d(n, m));
  });
  it('the 1D closed form for real taps fails below n = r, which is why the exercise assumes otherwise', () => {
    expect(ghostFormula1d(1, 5).realTaps).toBe(-1);
    expect(ghostCounts1d(1, 5).realTaps).toBe(1);
  });
  it('Exercise 5 for an 8 x 8 image and a 3 x 3 filter: 36, 576, 484', () => {
    expect(ghostCounts2d(8, 8, 3, 3)).toEqual({ ghostCells: 36, allTaps: 576, realTaps: 484 });
    expect(ghostFormula2d(8, 8, 3, 3)).toEqual({ ghostCells: 36, allTaps: 576, realTaps: 484 });
  });
  it('square closed forms: ghost cells 4r(N + r), real taps (NM - r(r+1))^2, equal brute force for N >= r', () => {
    for (let m = 1; m <= 9; m += 2)
      for (let n = Math.max(1, radiusOf(m)); n <= 14; n++) {
        const r = radiusOf(m);
        const b = ghostCounts2d(n, n, m, m);
        expect(ghostFormula2d(n, n, m, m)).toEqual(b);
        expect(b.ghostCells).toBe(4 * r * (n + r));
        expect(b.realTaps).toBe((n * m - r * (r + 1)) ** 2);
      }
  });
  it('Exercise 6 (rectangles) closed forms equal brute force', () => {
    for (const [n1, n2, m1, m2] of [[5, 9, 3, 5], [7, 4, 5, 1], [10, 12, 7, 3], [3, 3, 5, 7]])
      expect(ghostFormula2d(n1, n2, m1, m2)).toEqual(ghostCounts2d(n1, n2, m1, m2));
  });
  it('skippedMap sums to allTaps - realTaps and is zero away from the border', () => {
    const map = skippedMap(8, 8, 3, 3);
    expect(map.reduce((s, v) => s + v, 0)).toBe(576 - 484);
    expect(map[3 * 8 + 3]).toBe(0);
    expect(map[0]).toBe(5);
  });
});

describe('mapping threads to outputs (Fig. 7.6, p. 166)', () => {
  it('thread (1,1) of block (1,1) with 4 x 4 blocks computes P[5][5], patch corner (3,3)', () => {
    expect(mapThread({ bx: 1, by: 1, tx: 1, ty: 1, bdx: 4, bdy: 4 }, 2)).toEqual({ outRow: 5, outCol: 5, cornerRow: 3, cornerCol: 3 });
  });
  it('q-map: thread (tx 3, ty 0) of block (bx 2, by 1), r = 2: corner (2, 9), flat index 41', () => {
    const m = mapThread({ bx: 2, by: 1, tx: 3, ty: 0, bdx: 4, bdy: 4 }, 2);
    expect([m.outRow, m.outCol, m.cornerRow, m.cornerCol]).toEqual([4, 11, 2, 9]);
    expect(m.cornerRow * 16 + m.cornerCol).toBe(41);
  });
  it('ownerOf inverts mapThread', () => {
    const o = ownerOf(5, 11, 4, 4);
    expect(o).toEqual({ bx: 2, by: 1, tx: 3, ty: 1, bdx: 4, bdy: 4 });
    expect(mapThread(o, 1).outCol).toBe(11);
  });
});

describe('the basic kernel, tap by tap', () => {
  const N = toyImage(16, 16);
  const F = pyramidFilter(2);
  it('P[5][5] on the toy image: 25 taps, none skipped, Pvalue equals conv2d', () => {
    const run = basicTaps(16, 16, 2, 5, 5, N.data, F.data);
    expect(run.taps).toHaveLength(25);
    expect(run.skipped).toBe(0);
    expect(run.pvalue).toBe(conv2d(N, F).data[5 * 16 + 5]);
    expect(run.store).toBe(85);
    expect(run.taps[0]).toMatchObject({ inRow: 3, inCol: 3, nIndex: 51, fIndex: 0 });
  });
  it('every output of the toy image agrees with conv2d, for r = 1, 2, 3', () => {
    for (const r of [1, 2, 3]) {
      const Fr = pyramidFilter(r);
      const ref = conv2d(N, Fr);
      for (let row = 0; row < 16; row++)
        for (let col = 0; col < 16; col++) expect(basicTaps(16, 16, r, row, col, N.data, Fr.data).pvalue).toBe(ref.data[row * 16 + col]);
    }
  });
  it('q-skip: the thread for P[0][1] with r = 2 skips 13 of its 25 taps', () => {
    expect(basicTaps(16, 16, 2, 0, 1).skipped).toBe(13);
  });
  it('the thread for P[0][0] with r = 2 skips 16 of 25 (the scene\'s fourth part)', () => {
    expect(basicTaps(16, 16, 2, 0, 0).skipped).toBe(16);
  });
  it('a thread whose output is outside the image does no taps (the output guard)', () => {
    const run = basicTaps(100, 100, 2, 5, 100);
    expect(run.inImage).toBe(false);
    expect(run.taps).toHaveLength(0);
    expect(run.store).toBeNull();
  });
  it('the toy image holds small integers', () => {
    expect(Math.min(...N.data)).toBe(1);
    expect(Math.max(...N.data)).toBe(9);
  });
});

describe('which warps diverge', () => {
  it('1D: 4096 outputs, r = 3, warps of 32: 2 of 128 diverge, the first and the last', () => {
    const d = warpDivergence(4096, 1, 32, 1, 3, 32);
    expect(d.warps).toBe(128);
    expect(d.ghost).toBe(2);
    expect(d.guard).toBe(0);
    expect(d.marked.map((w) => w.bx)).toEqual([0, 127]);
  });
  it('1D with 256-thread blocks: still 2 of 128', () => {
    const d = warpDivergence(4096, 1, 256, 1, 3, 32);
    expect(d.warps).toBe(128);
    expect(d.divergent).toBe(2);
  });
  it('32 x 32 blocks, warps of 32: only warps at the left and right edges diverge', () => {
    const d = warpDivergence(256, 256, 32, 32, 2, 32);
    expect(d.warps).toBe(64 * 32);
    expect(d.ghost).toBe(2 * 256);
    for (const w of d.marked) {
      expect(w.cls).toBe('ghost');
      expect(w.bx === 0 || w.bx === 7).toBe(true);
    }
  });
  it('with 32 x 32 blocks the share is 2 of every width / 32 warps, whatever r up to 32', () => {
    for (const r of [1, 3, 7]) expect(warpDivergence(1024, 1024, 32, 32, r, 32).ghost / (1024 * 32)).toBeCloseTo(2 / 32, 12);
  });
  it('16 x 16 blocks, warps of 32: the top rows diverge too, in interior columns', () => {
    const d = warpDivergence(256, 256, 16, 16, 2, 32);
    const top = d.marked.filter((w) => w.by === 0 && w.bx > 0 && w.bx < 15);
    expect(top.length).toBeGreaterThan(0);
    // rows 0-1 of an interior top block: rows 0 and 1 skip different filter rows
    const w0 = warpInfo(256, 256, 16, 16, 2, 32, 5, 0, 0);
    expect(w0.ghost).toBe(true);
    expect(w0.segments).toEqual([{ row: 0, c0: 80, c1: 95 }, { row: 1, c0: 80, c1: 95 }]);
    // rows 2-3: interior rows for r = 2, so uniform
    expect(warpInfo(256, 256, 16, 16, 2, 32, 5, 0, 1).cls).toBe('uniform');
    // with r = 3 row 2 is not interior any more
    expect(warpInfo(256, 256, 16, 16, 3, 32, 5, 0, 1).cls).toBe('ghost');
  });
  it('with 16 x 16 blocks the share grows with r', () => {
    const share = (r: number) => warpDivergence(512, 512, 16, 16, r, 32).ghost;
    expect(share(1)).toBeLessThan(share(3));
    expect(share(3)).toBeLessThan(share(5));
  });
  it('the share shrinks as the image grows', () => {
    const frac = (n: number) => {
      const d = warpDivergence(n, n, 16, 16, 2, 32);
      return d.divergent / d.warps;
    };
    expect(frac(1024)).toBeLessThan(frac(256));
    expect(frac(256)).toBeLessThan(frac(64));
  });
  it('100 x 100 with 32 x 32 blocks: the right-edge blocks split at the guard, rows past the image are idle', () => {
    const d = warpDivergence(100, 100, 32, 32, 2, 32);
    expect(d.warps).toBe(16 * 32);
    // block column 3 covers columns 96..127: each of its 100 in-image rows has 4 lanes in, 28 out
    expect(d.guard).toBe(100);
    // rows 100..127: 28 rows x 4 block columns, no lane inside
    expect(d.idle).toBe(28 * 4);
    expect(d.ghost + 0).toBe(d.marked.filter((w) => w.ghost).length);
  });
  it('a 64-wide warp spans two rows of a 32 x 32 block, so the top rows diverge again', () => {
    expect(warpInfo(1024, 1024, 32, 32, 1, 64, 4, 0, 0).ghost).toBe(true);
    expect(warpInfo(1024, 1024, 32, 32, 1, 32, 4, 0, 0).ghost).toBe(false);
  });
  it('the counts add up', () => {
    for (const [w, h, bx, by, r, ws] of [[100, 37, 16, 16, 2, 32], [1000, 1000, 32, 32, 3, 64], [64, 64, 8, 8, 1, 32], [33, 1, 32, 1, 2, 64]]) {
      const d = warpDivergence(w, h, bx, by, r, ws);
      expect(d.uniform + d.marked.length).toBe(d.warps);
      expect(d.marked.filter((x) => x.cls === 'idle').length).toBe(d.idle);
    }
  });
  it('warpsInCorner returns every warp of the blocks in the close-up', () => {
    const c = warpsInCorner(256, 256, 16, 16, 2, 32, 32, 32);
    expect(c).toHaveLength(4 * 8);
  });
  it('a partial warp: 32 x 1 blocks with 64-wide warps hold 32 lanes', () => {
    expect(warpInfo(4096, 1, 32, 1, 3, 64, 0, 0, 0).lanes).toBe(32);
  });
});

describe('the scene\'s fifth part: one block of 16 lanes walking the taps together', () => {
  it('block (0,0) of the toy, r = 2: lanes idle exactly at their ghost taps', () => {
    const outs = Array.from({ length: 16 }, (_, t) => ({ row: Math.floor(t / 4), col: t % 4 }));
    const act = laneActivity(16, 16, 2, outs);
    expect(act).toHaveLength(16);
    expect(act[0].filter(Boolean).length).toBe(9);
    expect(act[15].filter(Boolean).length).toBe(25);
    // the first tap (fRow 0, fCol 0) is a ghost for every lane in rows 0-1 or columns 0-1
    expect(act.map((l) => l[0])).toEqual(outs.map((o) => o.row >= 2 && o.col >= 2));
    // some taps split the lanes: that is divergence
    const split = act[0].map((_, k) => act.some((l) => l[k]) && act.some((l) => !l[k]));
    expect(split.filter(Boolean).length).toBeGreaterThan(0);
  });
});

describe('reading the lab source', () => {
  const src = [
    '// header',
    '// @part basic',
    '// comment about the kernel',
    '__global__ void k(const float* N, const float* F, float* P, int r, int width, int height) {',
    '    int outCol = blockIdx.x * blockDim.x + threadIdx.x;',
    '    int outRow = blockIdx.y * blockDim.y + threadIdx.y;',
    '    if (outRow >= height || outCol >= width) return;',
    '    float Pvalue = 0.0f;',
    '    for (int fRow = 0; fRow < 2 * r + 1; fRow++) {',
    '        for (int fCol = 0; fCol < 2 * r + 1; fCol++) {',
    '            int inRow = outRow - r + fRow;',
    '            int inCol = outCol - r + fCol;',
    '            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {',
    '                Pvalue += F[fRow * (2 * r + 1) + fCol] * N[inRow * width + inCol];',
    '            }',
    '        }',
    '    }',
    '    P[outRow * width + outCol] = Pvalue;',
    '}',
    '',
    'void launch() {}',
    '',
    '// @part const',
    'other',
  ].join('\n');
  it('partOf takes the text between two markers', () => {
    const p = partOf(src, 'basic');
    expect(p.startsWith('// comment about the kernel')).toBe(true);
    expect(p.endsWith('void launch() {}')).toBe(true);
    expect(partOf(src, 'const')).toBe('other');
    expect(partOf(src, 'missing')).toBe(src);
  });
  it('firstKernel stops at the kernel\'s closing brace', () => {
    const k = firstKernel(partOf(src, 'basic'));
    expect(k.split('\n')).toHaveLength(16);
    expect(k.endsWith('}')).toBe(true);
  });
  it('basicLineNumbers finds every statement once', () => {
    const k = firstKernel(partOf(src, 'basic'));
    expect(basicLineNumbers(k)).toEqual({ outCol: 2, outRow: 3, guard: 4, init: 5, inRow: 8, inCol: 9, test: 10, accumulate: 11, store: 15 });
  });
});

describe('strips for drawing', () => {
  it('a 1024 x 1024 image with 32 x 32 blocks: two strips, one per side edge, full height', () => {
    const d = warpDivergence(1024, 1024, 32, 32, 2, 32);
    const s = stripsOf(d.marked);
    expect(s).toEqual([
      { cls: 'ghost', row0: 0, row1: 1023, c0: 0, c1: 31 },
      { cls: 'ghost', row0: 0, row1: 1023, c0: 992, c1: 1023 },
    ]);
    expect(d.ghost / d.warps).toBe(0.0625);
  });
  it('strips cover exactly the marked warps\' threads', () => {
    const d = warpDivergence(100, 37, 16, 16, 2, 32);
    const area = (xs: { row0: number; row1: number; c0: number; c1: number }[]) => xs.reduce((a, s) => a + (s.row1 - s.row0 + 1) * (s.c1 - s.c0 + 1), 0);
    expect(area(stripsOf(d.marked))).toBe(d.marked.reduce((a, w) => a + w.lanes, 0));
  });
});
