import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  BOOK,
  CONFIRMATIONS,
  CORRECTIONS,
  DEFAULTS,
  DEVICES,
  LIMITS,
  LIMIT_ROWS,
  VENDOR,
  auditPrinted,
  built,
  figures,
  fmtX,
  linearLayerRatio,
  marks,
  memoryBandwidth,
  nvlinkOverHostLink,
  outputTileRatio,
  pcieRate,
  peakFromCores,
  printedCells,
  ratio,
  regime,
  threshold,
  tileFor,
  tiledRatio,
  type RowId,
} from './hwtable';

describe('Fig. 24.1 as printed (p. 578)', () => {
  it('holds the cells verbatim, including the two the notes flag', () => {
    const cell = (id: RowId) => BOOK.rows.find((r) => r.id === id)!;
    expect(BOOK.columns).toEqual(['G80 (2007)', 'B200 (2025)', 'B200/G80 Ratio']);
    expect(cell('fp32').g80).toBe('0.345 TFLOPS (FP32)');
    expect(cell('tensor').b200).toBe('2200 TFLOPS (Tensor Core FP32)');
    expect(cell('tensor').ratio).toBe('638x');
    expect(cell('membw').b200).toBe('8,000 GB/s');
    expect(cell('memcap').g80).toBe('1.5GB');
    expect(cell('pcie').g80).toBe('8 GB/s (Gen2 x16, each way)');
    expect(cell('nvlink').b200).toBe('1,800 TB/s');
    expect(cell('nvlink').g80).toBe('N/A');
  });

  it('232, 93, 128 and 8 follow from their inputs; 638 does not', () => {
    const a = Object.fromEntries(auditPrinted().map((x) => [x.row, x]));
    expect(a.fp32.matches).toBe(true);
    expect(a.membw.matches).toBe(true);
    expect(a.memcap.matches).toBe(true);
    expect(a.pcie.matches).toBe(true);
    expect(a.tensor.matches).toBe(false);
    expect(Math.round(a.tensor.recomputed)).toBe(6377);
    expect(a.tensor.wouldMatch).toBe(220);
    expect(Object.keys(a)).toHaveLength(5);
  });

  it('prints the arithmetic the lesson shows', () => {
    expect(auditPrinted().map((x) => x.arithmetic)).toEqual([
      '80 / 0.345 = 232',
      '2200 / 0.345 = 6,377',
      '8000 / 86.4 = 93',
      '192 / 1.5 = 128',
      '64 / 8 = 8',
    ]);
  });

  it('the table contradicts itself in exactly two cells: the 638x ratio and the TB/s unit', () => {
    const cells = printedCells();
    expect(cells).toHaveLength(15);
    expect(cells.filter((c) => c.kind === 'wrong').map((c) => c.id)).toEqual(['tensor.ratio', 'nvlink.b200']);
    // Read as printed, the NVLink cell is 225 times the same device's memory bandwidth.
    expect(cells.find((c) => c.id === 'nvlink.b200')!.arithmetic).toBe('1,800,000 GB/s / 8,000 GB/s = 225');
  });

  it("five cells pass the table's arithmetic but are corrected by NVIDIA's documents, each naming its facts item", () => {
    const cells = printedCells();
    const differs = cells.filter((c) => c.kind === 'differs');
    // 80 TFLOPS (A8), the tensor-core label (A9), 86.4 GB/s (A3, A4), 192 GB (A11), Gen2 (A6).
    expect(differs.map((c) => `${c.id}:${c.fact}`)).toEqual(['fp32.b200:A8', 'tensor.b200:A9', 'membw.g80:A3, A4', 'memcap.b200:A11', 'pcie.g80:A6']);
    for (const c of differs) expect(c.verdict).toMatch(/NVIDIA/);
    // The 2200 figure stands and its ratio is the error, but its label is corrected.
    const tc = cells.find((c) => c.id === 'tensor.b200')!;
    expect(tc.verdict).toContain('the figure stands and the ratio is the error');
    expect(tc.verdict).toContain('TF32 on Tensor Cores, counted with sparsity');
    expect(cells.find((c) => c.id === 'pcie.g80')!.verdict).toContain('Generation 1, 4 GB/s each way');
    // Every ratio but 638x is consistent, and so are the four inputs NVIDIA bears out.
    expect(cells.filter((c) => c.kind === 'consistent').map((c) => c.id)).toEqual([
      'fp32.g80', 'fp32.ratio', 'membw.b200', 'membw.ratio', 'memcap.g80', 'memcap.ratio', 'pcie.b200', 'pcie.ratio',
    ]);
    // Every printed cell has exactly one verdict, and the corrections name only real cells.
    for (const id of Object.keys(CORRECTIONS)) expect(cells.some((c) => c.id === id)).toBe(true);
    for (const id of Object.keys(CONFIRMATIONS)) expect(cells.find((c) => c.id === id)!.kind).toBe('consistent');
  });

  it('every facts item the audit cites exists in the facts file', () => {
    const facts = fs.readFileSync(path.resolve(__dirname, '../../../notes/today/ch24-facts.md'), 'utf8');
    for (const c of printedCells()) for (const item of (c.fact ?? '').split(', ').filter(Boolean)) expect(facts).toContain(`### ${item}. `);
  });
});

describe('the arithmetic behind the cells (facts A2, A3, A5)', () => {
  it('128 multiply-adds per clock x 2 x 1.35 GHz = 345.6 GFLOPS', () => {
    expect(peakFromCores(128, 1.35)).toBeCloseTo(345.6, 9);
  });
  it('a 384-bit bus gives 86.4 GB/s at 900 MHz and 76.8 GB/s at 800 MHz', () => {
    expect(memoryBandwidth(384, 900)).toBeCloseTo(86.4, 9);
    expect(memoryBandwidth(384, 800)).toBeCloseTo(76.8, 9);
  });
  it('PCI Express x16, each way: 4.0, 8.0 and 63.0 GB/s for generations 1, 2 and 5', () => {
    expect(pcieRate(1, 16)).toBeCloseTo(4.0, 9);
    expect(pcieRate(2, 16)).toBeCloseTo(8.0, 9);
    expect(pcieRate(5, 16).toFixed(1)).toBe('63.0');
    expect(() => pcieRate(9, 16)).toThrow();
  });
  it('NVLink each way is about 14.3 times a Gen5 x16 link', () => {
    expect(nvlinkOverHostLink().toFixed(1)).toBe('14.3');
  });
});

describe('ratios under each reading', () => {
  const r = (row: RowId, set: 'book' | 'vendor', over = {}) => Math.round(ratio(row, set, { ...DEFAULTS[set], ...over })!);
  it('the book as printed: 232, 6,377, 93, 128, 8', () => {
    expect([r('fp32', 'book'), r('tensor', 'book'), r('membw', 'book'), r('memcap', 'book'), r('pcie', 'book')]).toEqual([232, 6377, 93, 128, 8]);
    expect(ratio('nvlink', 'book')).toBeNull();
  });
  it("NVIDIA's documents, one board each side: 217, 6,366 or 3,183, 104, 120, 16", () => {
    expect(r('fp32', 'vendor')).toBe(217);
    expect(r('tensor', 'vendor')).toBe(6366);
    expect(r('tensor', 'vendor', { tensor: 'dense' })).toBe(3183);
    expect(r('membw', 'vendor')).toBe(104);
    expect(r('memcap', 'vendor')).toBe(120);
    expect(r('pcie', 'vendor')).toBe(16);
    expect(fmtX(ratio('tensor', 'vendor')!)).toBe('6,366x');
  });
  it('the vendor figures are the ones the facts file gives', () => {
    expect(VENDOR.g80.fp32.value).toBeCloseTo(0.3456, 9);
    expect(VENDOR.g80.membw.value).toBeCloseTo(76.8, 9);
    expect(figures('memcap', 'vendor')).toEqual({ from: 1.5, to: 180, unit: 'GB' });
    expect(figures('fp32', 'vendor').to).toBe(75);
    expect(figures('nvlink', 'vendor')).toEqual({ from: null, to: 1800, unit: 'GB/s' });
    expect(figures('nvlink', 'vendor', { ...DEFAULTS.vendor, nvlink: 'eachWay' }).to).toBe(900);
  });
  it('the printed host-link figures with a Generation 1 board give 16x', () => {
    expect(r('pcie', 'book', { g80Pcie: 1 })).toBe(16);
    expect(r('pcie', 'vendor', { g80Pcie: 2 })).toBe(8);
  });
  it('the host link grew least under every combination of options', () => {
    for (const set of ['book', 'vendor'] as const)
      for (const tensor of ['sparse', 'dense'] as const)
        for (const g80Pcie of [1, 2] as const) {
          const o = { tensor, g80Pcie, nvlink: 'both' as const };
          const host = ratio('pcie', set, o)!;
          for (const row of ['fp32', 'tensor', 'membw', 'memcap'] as const) expect(ratio(row, set, o)!).toBeGreaterThan(host * 5);
        }
  });
  it('says how each figure is built', () => {
    expect(built('fp32', 'book')[0].text).toContain('128 multiply-adds per clock × 2 × 1.35 GHz = 345.6 GFLOPS');
    expect(built('membw', 'book')[0].text).toContain('384 bits × 2 transfers per clock × 900 MHz / 8 = 86.4 GB/s');
    expect(built('membw', 'vendor')[0].text).toContain('800 MHz / 8 = 76.8 GB/s');
    expect(built('pcie', 'vendor')[0].text).toContain('2.5 GT/s × 8/10 encoding × 16 lanes / 8 = 4.0 GB/s');
    expect(built('pcie', 'book')[0].text).toContain('5 GT/s × 8/10 encoding × 16 lanes / 8 = 8.0 GB/s');
    expect(built('pcie', 'vendor')[1].text).toContain('32 GT/s × 128/130 encoding × 16 lanes / 8 = 63.0 GB/s');
    expect(built('nvlink', 'vendor')[1].text).toContain('14.3 times a Gen5 x16 host link');
    for (const row of BOOK.rows) for (const b of built(row.id, 'vendor')) expect(b.fact).toMatch(/^A\d+$/);
  });
});

describe('the threshold of section 5.1 applied to the table', () => {
  const dev = (key: string) => DEVICES.find((d) => d.key === key)!;
  const thr = (key: string) => threshold(dev(key).peak, dev(key).bw);
  it("offers the book's two columns and NVIDIA's dense TF32 figure, nothing else", () => {
    expect(DEVICES.map((d) => d.key)).toEqual(['book2007', 'book2025', 'book2025tc', 'book2025dense']);
  });
  it('thresholds: 3.993, 10, 275, 137.5, and 4.5 on the C870', () => {
    expect(thr('book2007').toFixed(3)).toBe('3.993');
    expect(thr('book2025')).toBe(10);
    expect(thr('book2025tc')).toBe(275);
    expect(thr('book2025dense')).toBe(137.5);
    expect(threshold(VENDOR.g80.fp32.value * 1000, VENDOR.g80.membw.value)).toBeCloseTo(4.5, 9);
  });
  it('tile widths: 16, 40, 1100, 550, and 18 on the C870', () => {
    expect(tileFor(thr('book2007'))).toBe(16);
    expect(tileFor(thr('book2025'))).toBe(40);
    expect(tileFor(thr('book2025tc'))).toBe(1100);
    expect(tileFor(thr('book2025dense'))).toBe(550);
    expect(tileFor(threshold(VENDOR.g80.fp32.value * 1000, VENDOR.g80.membw.value))).toBe(18);
  });
  it('on the printed 2007 figures T = 15 is memory-bound and T = 16 compute-bound', () => {
    expect(regime(tiledRatio(15), thr('book2007'))).toBe('memory-bound');
    expect(regime(tiledRatio(16), thr('book2007'))).toBe('compute-bound');
  });
  it('the threshold rose 2.5x for FP32 and about 69x against the tensor-core figure', () => {
    expect(thr('book2025') / thr('book2007')).toBeCloseTo(2.5, 1);
    expect(Math.round(thr('book2025tc') / thr('book2007'))).toBe(69);
  });
  it("kernel marks carry the book's ratios", () => {
    const m = Object.fromEntries(marks(32).map((x) => [x.id, x.ratio]));
    expect(m.vecadd).toBeCloseTo(0.083, 3);
    expect(m.naive).toBe(0.25);
    expect(m.tiled).toBe(8);
    expect(m.out128).toBe(32);
    expect(m.lin1).toBeCloseTo(0.9988, 4);
    expect(m.lin512).toBeCloseTo(315.08, 2);
    expect(outputTileRatio(32, 32)).toBe(8);
    expect(linearLayerRatio(1, 1024, 4096, 4)).toBeCloseTo(0.4994, 4);
  });
  it('batch-1 generation is memory-bound on every device; neither a 32-wide nor a 128-wide tile reaches the tensor-core threshold', () => {
    const m = Object.fromEntries(marks(32).map((x) => [x.id, x.ratio]));
    for (const d of DEVICES) expect(regime(m.lin1, threshold(d.peak, d.bw))).toBe('memory-bound');
    expect(regime(m.tiled, thr('book2025tc'))).toBe('memory-bound');
    expect(regime(m.out128, thr('book2025tc'))).toBe('memory-bound');
    expect(regime(m.out128, thr('book2025'))).toBe('compute-bound');
  });
  it('a T x T pair of float tiles needs about 9.7 MB at T = 1100 and 2.4 MB at T = 550', () => {
    expect((8 * 1100 * 1100) / 1e6).toBeCloseTo(9.68, 2);
    expect((8 * 550 * 550) / 1e6).toBeCloseTo(2.42, 2);
  });
});

describe('what limits what', () => {
  it('each row of the table has exactly one card', () => {
    for (const row of LIMIT_ROWS) expect(LIMITS.filter((c) => c.row === row.id)).toHaveLength(1);
    expect(LIMITS).toHaveLength(5);
  });
  it('each card names the earlier chapters it draws on, as its source does', () => {
    for (const c of LIMITS) {
      const chapters = [...c.from.matchAll(/\d+/g)].map((m) => m[0]);
      expect(chapters.length).toBeGreaterThan(0);
      for (const n of chapters) expect(c.source).toContain(`ch. ${n} `);
    }
    // The all-reduce card pins the topology, so the host link is not a fair answer.
    expect(LIMITS.find((c) => c.row === 'gpulink')!.card).toContain('NVLink');
  });
  it('question ids are stable and unique', () => {
    expect(LIMITS.map((c) => c.qid)).toEqual([
      'ch24/start/q-limit-load',
      'ch24/start/q-limit-decode',
      'ch24/start/q-limit-prefill',
      'ch24/start/q-limit-fit',
      'ch24/start/q-limit-allreduce',
    ]);
  });
});
