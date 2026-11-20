import { describe, expect, it } from 'vitest';
import {
  BOOK_DEVICE,
  BOOK_PRESET,
  WELL_OPTIMISED,
  bestRegisterTiled,
  blockSizes,
  ceiling,
  copyGBs,
  countedAccesses,
  designCeilings,
  designTraffic,
  intensity,
  intensityLimit,
  kernelRates,
  labRows,
  measuredBandwidth,
  rowDesign,
  rowRate,
  rssOverSsa,
  scanBytes,
  shareOfCopy,
  threshold,
  toppBytesPerElement,
  type Row,
} from './scantraffic';

const G = (x: number) => x / 1e9;

describe('one scan: intensity and ceiling (§11.8, p. 272)', () => {
  it('a float scan moves 8 bytes per element and does about 1/8 OP/B', () => {
    const b = scanBytes('float');
    expect([b.load, b.store, b.total]).toEqual([4, 4, 8]);
    expect(intensityLimit(b.total)).toBe(0.125);
    // N - 1 additions over 8N bytes: just under 1/8 for any finite N.
    expect(intensity(1024, 8)).toBeCloseTo(1023 / 8192, 12);
    expect(intensity(1 << 26, 8)).toBeLessThan(0.125);
    expect(intensity(1 << 26, 8)).toBeCloseTo(0.125, 6);
  });
  it('the book\'s device: threshold 20 FLOP/B with 66.9 or with NVIDIA\'s 67', () => {
    expect(threshold(BOOK_DEVICE.tflopsBook, BOOK_DEVICE.bandwidthGBs)).toBeCloseTo(19.97, 2);
    expect(threshold(BOOK_DEVICE.tflopsNvidia, BOOK_DEVICE.bandwidthGBs)).toBeCloseTo(20.0, 6);
    // scan sits 160 times below it
    expect(threshold(BOOK_DEVICE.tflopsNvidia, BOOK_DEVICE.bandwidthGBs) / intensityLimit(8)).toBeCloseTo(160, 6);
  });
  it('the ceiling: 3.35 TB/s over 8 bytes is 419e9 elements per second (418.75 exactly)', () => {
    expect(G(ceiling(BOOK_DEVICE.bandwidthGBs, 8))).toBeCloseTo(418.75, 9);
    expect(Math.round(G(ceiling(3350, 8)))).toBe(419);
  });
  it('question q-half: half in, half out, float accumulator: 0.25 OP/B', () => {
    const b = scanBytes('half');
    expect(b.total).toBe(4);
    expect(intensityLimit(b.total)).toBe(0.25);
  });
  it('question q-ceiling: 800 GB/s copy, floats, one pass: 100e9 elements per second', () => {
    expect(G(ceiling(800, scanBytes('float').total))).toBe(100);
  });
  it('question q-topp-bytes and the explorer\'s flag option: 9 bytes per element', () => {
    expect(toppBytesPerElement()).toBe(9);
    expect(scanBytes('float', 'scan+mask').total).toBe(9);
    expect(scanBytes('half', 'scan+mask').total).toBe(5);
  });
  it('question q-80: reading the input twice moves 12 bytes where 8 suffice, so 85% measured is about 57% useful', () => {
    const useful = 0.85 * (8 / 12);
    expect(useful).toBeCloseTo(0.567, 3);
    expect(useful).toBeLessThan(WELL_OPTIMISED);
  });
});

describe('global scan designs (§11.9, pp. 273-276)', () => {
  it('scan-scan-add: (4 + 2/S)·N elements, (16 + 8/S)·N bytes, kernel by kernel as the book counts', () => {
    for (const S of [256, 1024, 4096]) {
      const t = designTraffic('ssa', S, 4);
      expect(t.elementsPerElement).toBeCloseTo(4 + 2 / S, 12);
      expect(t.bytesPerElement).toBeCloseTo(16 + 8 / S, 12);
    }
    const t = designTraffic('ssa', 1024, 4);
    const per = (k: number) => t.kernels[k].accesses.filter((a) => !a.bookkeeping).map((a) => [a.kind, a.perElement]);
    expect(per(0)).toEqual([['load', 1], ['store', 1]]);
    expect(per(1)).toEqual([['load', 1 / 1024], ['store', 1 / 1024]]);
    expect(per(2)).toEqual([['load', 1], ['store', 1]]);
    expect(t.formula).toBe('(16 + 8/S)·N bytes');
  });
  it('question q-ssa: S = 4096 gives 16.002 bytes per element', () => {
    expect(designTraffic('ssa', 4096).bytesPerElement).toBeCloseTo(16.001953125, 12);
    expect(Number(designTraffic('ssa', 4096).bytesPerElement.toFixed(3))).toBe(16.002);
  });
  it('reduce-scan-scan: (3 + 3/S)·N elements, (12 + 12/S)·N bytes', () => {
    for (const S of [256, 2048]) {
      const t = designTraffic('rss', S, 4);
      expect(t.elementsPerElement).toBeCloseTo(3 + 3 / S, 12);
      expect(t.bytesPerElement).toBeCloseTo(12 + 12 / S, 12);
    }
    const t = designTraffic('rss', 2048, 4);
    expect(t.kernels[0].accesses.map((a) => [a.kind, a.perElement])).toEqual([['load', 1], ['store', 1 / 2048]]);
    expect(t.formula).toBe('(12 + 12/S)·N bytes');
  });
  it('single pass: the ideal 8 bytes; its bookkeeping adds 24 bytes per block', () => {
    expect(designTraffic('single', 2048).bytesPerElement).toBe(8);
    expect(designTraffic('single', 2048, 4, true).bytesPerElement).toBeCloseTo(8 + 24 / 2048, 12);
    expect(countedAccesses(designTraffic('single', 2048), false)).toHaveLength(2);
    expect(countedAccesses(designTraffic('single', 2048), true)).toHaveLength(8);
  });
  it('bookkeeping the book leaves out: 2/S elements for scan-scan-add, 1/S for reduce-scan-scan', () => {
    const S = 1024;
    expect(designTraffic('ssa', S, 4, true).elementsPerElement - designTraffic('ssa', S, 4).elementsPerElement).toBeCloseTo(2 / S, 12);
    expect(designTraffic('rss', S, 4, true).elementsPerElement - designTraffic('rss', S, 4).elementsPerElement).toBeCloseTo(1 / S, 12);
  });
  it('the preset reproduces the book: 419, 209 and 279 ×10⁹ elements per second', () => {
    const c = designCeilings(BOOK_PRESET.bandwidthGBs, BOOK_PRESET.S, 4, BOOK_PRESET.bookkeeping);
    expect(Math.round(G(c.single))).toBe(419);
    expect(Math.round(G(c.ssa))).toBe(209);
    expect(Math.round(G(c.rss))).toBe(279);
    // and the book's "half" and "two thirds" of ideal
    expect(c.ssa / c.single).toBeCloseTo(0.5, 2);
    expect(c.rss / c.single).toBeCloseTo(2 / 3, 2);
  });
  it('half-precision elements halve every byte count', () => {
    expect(designTraffic('ssa', 1024, 2).bytesPerElement).toBeCloseTo((16 + 8 / 1024) / 2, 12);
    expect(designTraffic('rss', 1024, 2).formula).toBe('(6 + 6/S)·N bytes');
  });
});

describe('reading lab rows', () => {
  const rows: Row[] = [
    { name: 'copy', n: 1 << 26, seconds: 0.002, gb_per_s: 268.435456 },
    { name: 'scan', kernel: 'regtile', coarse: 4, n: 1 << 26, seconds: 0.0025 },
    { name: 'scan', kernel: 'regtile', coarse: 8, n: 1 << 26, seconds: 0.0023 },
    { name: 'scan', kernel: 'coarse', coarse: 8, n: 1 << 26, seconds: 0.004 },
    { name: 'scan', kernel: 'kogge', gb_per_s: 100 },
  ];
  it('the copy row, and rates from n / seconds or from GB/s at 8 bytes', () => {
    expect(copyGBs(rows)).toBeCloseTo(268.435456, 9);
    expect(rowRate(rows[1])).toBeCloseTo((1 << 26) / 0.0025, 3);
    expect(rowRate(rows[4])).toBeCloseTo(100e9 / 8, 3);
  });
  it('share of copy: a kernel as fast as the copy, per byte, scores 1', () => {
    expect(shareOfCopy((1 << 26) / 0.002, copyGBs(rows)!)).toBeCloseTo(1, 9);
  });
  it('kernel rates and the best register-tiled factor', () => {
    const r = kernelRates(rows, copyGBs(rows)!);
    expect(r).toHaveLength(4);
    expect(r[0].label).toBe('regtile, factor 4');
    const best = bestRegisterTiled(r)!;
    expect(best.marked).toBe(true);
    expect(best.rate.factor).toBe(8);
    expect(best.rate.share).toBeCloseTo(0.002 / 0.0023, 9);
  });
  it('the chapter 11 labs\' convention: rows named "scan", the copy as kernel "copy", the lab\'s own share used', () => {
    const lab: Row[] = [
      { name: 'scan', kernel: 'copy', block: 256, n: 1 << 24, seconds: 0.001, gb_per_s: 134.217728, share_of_copy: 1 },
      { name: 'scan', kernel: 'copy', block: 1024, n: 1 << 24, seconds: 0.0008, gb_per_s: 167.77216, share_of_copy: 1 },
      { name: 'scan', kernel: 'regtiled', coarse: 8, block: 256, n: 1 << 24, seconds: 0.00125, gb_per_s: 107.374, share_of_copy: 0.8 },
    ];
    expect(copyGBs(lab)).toBeCloseTo(167.77216, 6);
    const r = kernelRates(lab, copyGBs(lab)!);
    expect(r).toHaveLength(1);
    expect(r[0].share).toBe(0.8);
    expect(r[0].label).toBe('regtiled, factor 8, 256 threads');
    expect(blockSizes(lab)).toEqual([256, 1024]);
    expect(rowDesign({ name: 'scan', kernel: 'reduce_scan_scan' })).toBe('rss');
    expect(rowDesign({ name: 'scan', kernel: 'scan_scan_add' })).toBe('ssa');
  });
  it('this machine\'s bandwidth: lab 11.2\'s copy first, then lab 11.1\'s, then lab 5.1\'s probe', () => {
    const roof = { lab: 'ch05/lab1_roofline', variant: 'solution', bench: [{ name: 'roofline', bandwidth_gb_per_s: 270, compute_gflops: 1890 }] };
    const l1 = { lab: 'ch11/lab1_blockscan', variant: 'solution', bench: [{ name: 'scan', kernel: 'copy', gb_per_s: 250 }] };
    const l2 = { lab: 'ch11/lab2_coarsened', variant: 'solution', bench: [{ name: 'scan', kernel: 'copy', gb_per_s: 260 }] };
    const l2work = { lab: 'ch11/lab2_coarsened', variant: 'work', bench: [{ name: 'scan', kernel: 'copy', gb_per_s: 999 }] };
    const run = (results: { lab: string; variant: string; bench: Row[] }[]) => ({ generated: '2026-11-20', device: {}, results });
    expect(measuredBandwidth(run([roof]))).toEqual({ gbs: 270, source: 'the bandwidth probe of lab 5.1', threshold: 7 });
    expect(measuredBandwidth(run([roof, l1]))!.gbs).toBe(250);
    expect(measuredBandwidth(run([roof, l1, l2, l2work]))).toEqual({ gbs: 260, source: 'the copy kernel of lab 11.2', threshold: 7 });
    expect(measuredBandwidth(run([l2work]))).toBeUndefined();
    expect(labRows(run([l1, l2]), 'ch11/lab2_')).toHaveLength(1);
  });
  it('lab 11.4: designs from row names, and the speed ratio at 2^26', () => {
    expect(rowDesign({ name: 'reduce_scan_scan' })).toBe('rss');
    expect(rowDesign({ name: 'hier', design: 'scan-scan-add' })).toBe('ssa');
    expect(rowDesign({ name: 'copy' })).toBeUndefined();
    const lab4: Row[] = [
      { name: 'scan_scan_add', n: 1 << 26, seconds: 0.004 },
      { name: 'reduce_scan_scan', n: 1 << 26, seconds: 0.003 },
      { name: 'scan_scan_add', n: 1 << 20, seconds: 0.0001 },
    ];
    const r = rssOverSsa(lab4)!;
    expect(r.n).toBe(1 << 26);
    expect(r.ratio).toBeCloseTo(4 / 3, 9);
    expect(rssOverSsa([lab4[0]])).toBeUndefined();
  });
});
