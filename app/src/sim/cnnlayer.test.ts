import { describe, expect, it } from 'vitest';
import {
  BOOK_F,
  BOOK_LAYER,
  BOOK_X,
  BOOK_Y_PRINTED,
  FIG_19_4_LOOPS,
  LENET,
  batchTraffic,
  changedByFlip,
  directLoadsPerOutput,
  directRatio,
  directRatioMaps,
  directTrace,
  forward,
  forwardBatched,
  gradeLoops,
  lenetLayer,
  lessonRegion,
  output,
  parallelism,
  perInputMap,
  readsAtStep,
  readsOfThread,
  shapes,
  symmetricUnderRotation,
  terms,
} from './cnnlayer';

const S = BOOK_LAYER;

describe("the book's example, Fig. 19.2 and Eq. 19.1", () => {
  it('Y[0,0,0] = 7 + 5 + 2 = 14', () => {
    expect(output(S, BOOK_X, BOOK_F, 0, 0, 0)).toBe(14);
    expect(perInputMap(S, BOOK_X, BOOK_F, 0, 0, 0)).toEqual([7, 5, 2]);
  });
  it('the whole layer reproduces the printed outputs 14 20 15 24 / 12 24 17 26', () => {
    expect(forward(S, BOOK_X, BOOK_F)).toEqual([...BOOK_Y_PRINTED]);
  });
  it('output map 0, pixel (0, 1) is 20 = 10 + 8 + 2 (the prediction question)', () => {
    expect(perInputMap(S, BOOK_X, BOOK_F, 0, 0, 1)).toEqual([10, 8, 2]);
  });
  it('the term trace walks c, p, q in order with a running sum ending at the output', () => {
    const t = terms(S, BOOK_X, BOOK_F, 0, 0, 0);
    expect(t.length).toBe(12);
    expect(t.map((x) => [x.c, x.p, x.q])).toEqual([
      [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
      [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1],
      [2, 0, 0], [2, 0, 1], [2, 1, 0], [2, 1, 1],
    ]);
    expect(t.map((x) => x.sum)).toEqual([1, 3, 5, 7, 7, 9, 9, 12, 12, 14, 14, 14]);
    // Patch of X0 for pixel (0,0): 1 2 / 1 1; filter F[0,0]: 1 1 / 2 2.
    expect(t.slice(0, 4).map((x) => x.x)).toEqual([1, 2, 1, 1]);
    expect(t.slice(0, 4).map((x) => x.f)).toEqual([1, 1, 2, 2]);
  });
  it('the batched loop nest gives each sample its own outputs and shares F', () => {
    const X2 = [...BOOK_X, ...BOOK_X.map((v) => 2 * v)];
    const Y = forwardBatched(S, 2, X2, BOOK_F);
    expect(Y.slice(0, 8)).toEqual([...BOOK_Y_PRINTED]);
    expect(Y.slice(8)).toEqual(BOOK_Y_PRINTED.map((v) => 2 * v));
  });
});

describe('rotating every filter by 180 degrees (ours: a true convolution)', () => {
  it('Y[0,0,0] becomes 15: 8 + 5 + 2', () => {
    expect(output(S, BOOK_X, BOOK_F, 0, 0, 0, true)).toBe(15);
    expect(perInputMap(S, BOOK_X, BOOK_F, 0, 0, 0, true)).toEqual([8, 5, 2]);
  });
  it('F[0,1] and F[0,2] are unchanged by the rotation, F[0,0] is not', () => {
    expect(symmetricUnderRotation(S, BOOK_F, 0, 0)).toBe(false);
    expect(symmetricUnderRotation(S, BOOK_F, 0, 1)).toBe(true);
    expect(symmetricUnderRotation(S, BOOK_F, 0, 2)).toBe(true);
  });
  it('the flipped layer and the outputs it changes', () => {
    const flipped = forward(S, BOOK_X, BOOK_F, true);
    const changed = changedByFlip(S, BOOK_X, BOOK_F);
    expect(changed.length).toBeGreaterThan(0);
    for (let i = 0; i < 8; i++) expect(changed.includes(i)).toBe(flipped[i] !== BOOK_Y_PRINTED[i]);
    expect(changed).toContain(0);
  });
});

describe('shapes and LeNet-5 (p. 458)', () => {
  it('the example has 6 filters; F is M x C x K x K and has no batch dimension', () => {
    const sh = shapes(S, 4);
    expect(sh.filters).toBe(6);
    expect(sh.X).toEqual([4, 3, 3, 3]);
    expect(sh.F).toEqual([2, 3, 2, 2]);
    expect(sh.Y).toEqual([4, 2, 2, 2]);
  });
  it('C1 maps 1 x 32 x 32 to 6 x 28 x 28', () => {
    const sh = shapes(lenetLayer('C1')!);
    expect(sh.X).toEqual([1, 32, 32]);
    expect(sh.Y).toEqual([6, 28, 28]);
    expect(sh.filters).toBe(6);
    expect(sh.weights).toBe(150);
    expect(sh.macsPerSample).toBe(117600);
  });
  it('C3: 96 filters, 2400 weights, 6 x 14 x 14 to 16 x 10 x 10', () => {
    const sh = shapes(lenetLayer('C3')!);
    expect(sh.filters).toBe(96);
    expect(sh.weights).toBe(2400);
    expect(sh.X).toEqual([6, 14, 14]);
    expect(sh.Y).toEqual([16, 10, 10]);
    expect(sh.macsPerSample).toBe(240000);
  });
  it('C5: 1920 filters make 120 one-pixel maps', () => {
    const sh = shapes(lenetLayer('C5')!);
    expect(sh.filters).toBe(1920);
    expect(sh.Y).toEqual([120, 1, 1]);
    expect(sh.weights).toBe(48000);
  });
  it('the strip has the eight stages of Fig. 19.1, three of them convolutions', () => {
    expect(LENET.map((s) => s.id)).toEqual(['INPUT', 'C1', 'S2', 'C3', 'S4', 'C5', 'F6', 'OUTPUT']);
    expect(LENET.filter((s) => s.layer).map((s) => s.id)).toEqual(['C1', 'C3', 'C5']);
  });
  it("early layers: few channels, large maps; late layers: the reverse", () => {
    const [c1, c5] = [lenetLayer('C1')!, lenetLayer('C5')!];
    expect(c1.C).toBeLessThan(c5.C);
    expect(c1.H * c1.W).toBeGreaterThan(c5.H * c5.W);
  });
  it("F's share of the ideal traffic falls as the batch grows, and its bytes per sample fall as 1/N", () => {
    const c5 = lenetLayer('C5')!;
    const t1 = batchTraffic(c5, 1);
    const t256 = batchTraffic(c5, 256);
    expect(t1.fBytes).toBe(192000);
    expect(t1.xBytes).toBe(1600);
    expect(t1.yBytes).toBe(480);
    expect(t1.fShare).toBeCloseTo(192000 / 194080, 10);
    expect(t256.fShare).toBeLessThan(t1.fShare);
    expect(t256.fBytesPerSample).toBe(t1.fBytesPerSample / 256);
    const c1 = lenetLayer('C1')!;
    expect(batchTraffic(c1, 1).fShare).toBeCloseTo(600 / (4096 + 600 + 18816), 10);
  });
});

describe('the loops of Fig. 19.4 (p. 460)', () => {
  it('n, m, h, w are independent; c, p, q accumulate into one output', () => {
    expect(FIG_19_4_LOOPS.map((l) => l.v)).toEqual(['n', 'm', 'h', 'w', 'c', 'p', 'q']);
    expect(FIG_19_4_LOOPS.filter((l) => l.role === 'independent').map((l) => l.v)).toEqual(['n', 'm', 'h', 'w']);
    expect(FIG_19_4_LOOPS.map((l) => l.line)).toEqual([4, 5, 6, 7, 9, 10, 11]);
  });
  it('grades a classification', () => {
    expect(gradeLoops({ n: 'independent', m: 'independent', h: 'independent', w: 'independent', c: 'accumulates', p: 'accumulates', q: 'accumulates' })).toEqual([]);
    expect(gradeLoops({ n: 'independent', m: 'independent', h: 'independent', w: 'independent', c: 'independent', p: 'accumulates', q: 'accumulates' })).toEqual(['c']);
    expect(gradeLoops({}).length).toBe(7);
  });
  it('parallelism is N * M * H_out * W_out', () => {
    expect(parallelism(lenetLayer('C1')!, 1)).toBe(6 * 28 * 28);
    expect(parallelism(BOOK_LAYER, 3)).toBe(3 * 2 * 2 * 2);
  });
});

describe('what the direct kernel reads (the "direct-reads" scene)', () => {
  const tr = directTrace(S, BOOK_X, BOOK_F, 2);
  it('two blocks of four threads: gridDim (2, 1, 1)', () => {
    expect(tr.grid).toEqual({ x: 2, y: 1, z: 1 });
    expect(tr.blocks.length).toBe(2);
    expect(new Set(tr.blocks[0].reads.map((r) => r.thread)).size).toBe(4);
  });
  it('thread (0, 0) of block 0 makes 12 X loads and 12 F loads and its sum climbs to 14', () => {
    const t0 = readsOfThread(tr.blocks[0], 0);
    expect(t0.length).toBe(12);
    expect(t0[t0.length - 1].sum).toBe(14);
    expect(t0.map((r) => r.sum)).toEqual(terms(S, BOOK_X, BOOK_F, 0, 0, 0).map((x) => x.sum));
  });
  it('48 X reads of 27 values per block; centre 4, edges 2, corners 1 in every map', () => {
    const b0 = tr.blocks[0];
    expect(b0.xReads).toBe(48);
    expect(b0.distinctX).toBe(27);
    for (let c = 0; c < 3; c++) expect(b0.xHeat.slice(9 * c, 9 * c + 9)).toEqual([1, 2, 1, 2, 4, 2, 1, 2, 1]);
  });
  it('every step, the four threads of a block read one shared F address', () => {
    for (const b of tr.blocks) {
      expect(b.fAddressesPerStep).toEqual(new Array(12).fill(1));
      expect(readsAtStep(b, 5).length).toBe(4);
      expect(b.distinctF).toBe(12);
    }
  });
  it('block 1 reads all 48 again: 96 X reads in total for 27 distinct values', () => {
    expect(tr.blocks[1].xReads).toBe(48);
    expect(tr.xReads).toBe(96);
    expect(tr.distinctX).toBe(27);
    expect(tr.xHeat.slice(0, 9)).toEqual([2, 4, 2, 4, 8, 4, 2, 4, 2]);
  });
  it('the kernel stores the right outputs and requests 0.25 FLOP per byte', () => {
    expect(tr.Y).toEqual([...BOOK_Y_PRINTED]);
    expect(tr.flops).toBe(192);
    expect(tr.loadBytes).toBe(768);
    expect(tr.ratio).toBe(0.25);
  });
  it('LeNet C3: 300 loads per output for 150 multiply-adds, 0.25 FLOP/B', () => {
    const c3 = lenetLayer('C3')!;
    expect(directLoadsPerOutput(c3)).toBe(300);
    expect(directRatio(c3)).toBe(0.25);
  });
  it('two maps per thread sharing each X load: 600 FLOP over 1800 bytes, one third', () => {
    const c3 = lenetLayer('C3')!;
    expect(directRatioMaps(c3, 1)).toBe(directRatio(c3));
    expect(directRatioMaps(c3, 2)).toBeCloseTo(1 / 3, 12);
    expect(directRatioMaps(lenetLayer('C1')!, 2)).toBeCloseTo(1 / 3, 12);
  });
});

describe('lesson regions of lab files', () => {
  const file = ['#include <x>', '// [lesson:a]', 'int a = 1;', '  // [lesson:inner]', 'int b = 2;', '  // [/lesson:inner]', '// [/lesson:a]', 'int c;', '// [lesson:a]', 'int d;', '// [/lesson:a]', ''].join('\n');
  it('slices between the markers, drops marker lines, joins repeated regions', () => {
    expect(lessonRegion(file, 'a')).toEqual({ code: 'int a = 1;\nint b = 2;\n\nint d;', found: true });
    expect(lessonRegion(file, 'inner')).toEqual({ code: 'int b = 2;', found: true });
  });
  it('falls back to the whole file when the markers are missing', () => {
    expect(lessonRegion(file, 'missing')).toEqual({ code: file.trimEnd(), found: false });
    expect(lessonRegion('// [lesson:x]\nint a;', 'x').found).toBe(false);
  });
});
