import { describe, expect, it } from 'vitest';
import {
  BOOK_DEVICE, FORMAT_IDS, LAYERS, bitsPerWeight, byteRatio, bytesPerWeight, ceiling, gemv, intensityLimit, kneeBatch,
  layer, mbu, modelBytes, overheadShare, pipelineBytesPerWeight, pipelinePasses, pipelineRelative, passBytes, qgemvRows,
  measuredBandwidth, measuredPipelines, pickRuns, speedupOverF32, weightOnlyIntensity,
} from './qtraffic';

describe('bytes per weight (x2-facts E1)', () => {
  it('matches the E1 table, scales included', () => {
    const want: Record<string, number> = {
      f32: 4, f16: 2, q8_0: 1.0625, q4_0: 0.5625, q4_1: 0.625, q4_k: 0.5625, mxfp4: 0.53125,
      mlx4_g32: 0.625, mlx4_g64: 0.5625, mlx4_g128: 0.53125, mlx4_g64_f32: 0.625,
    };
    for (const [f, b] of Object.entries(want)) expect(bytesPerWeight(f as never)).toBe(b);
    expect(bytesPerWeight('mlx4_g32_f32')).toBe(0.75);
    expect(bytesPerWeight('mlx4_g128_f32')).toBe(0.5625);
  });
  it('Q8_0 is 8.5 bits per weight, not 8', () => {
    expect(bitsPerWeight('q8_0')).toBe(8.5);
    expect(bitsPerWeight('q4_0')).toBe(4.5);
    expect(overheadShare('q8_0')).toBeCloseTo(2 / 34, 12);
  });
  it('the speed-up ceilings are the byte ratios: 3.76 for Q8_0, 7.11 for Q4_0, 1.89 from Q8_0 to Q4_0', () => {
    expect(byteRatio('q8_0')).toBeCloseTo(3.7647, 4);
    expect(byteRatio('q4_0')).toBeCloseTo(7.1111, 4);
    expect(byteRatio('q4_0', 'q8_0')).toBeCloseTo(1.8889, 4);
    expect(weightOnlyIntensity('q8_0')).toBeCloseTo(1.882, 3);
    expect(weightOnlyIntensity('q4_0')).toBeCloseTo(3.556, 3);
    expect(weightOnlyIntensity('f32')).toBe(0.5);
  });
  it('every format has whole blocks and fewer overhead bytes than block bytes', () => {
    for (const f of FORMAT_IDS) expect(overheadShare(f)).toBeLessThan(0.5);
  });
});

describe('GEMV traffic and the roofline', () => {
  it("reproduces the book's 1024-to-4096 layer at about 1 FLOP/B in 16-bit at batch 1 (p. 505), and 315 at batch 512", () => {
    const L = layer('1024x4096');
    expect(gemv('f16', L, 1, 2).intensity).toBeCloseTo(0.9988, 4);
    expect(gemv('f16', L, 512, 2).intensity).toBeCloseTo(315.08, 2);
  });
  it('float32 at batch 1 is about 0.5 FLOP/B', () => {
    expect(gemv('f32', layer('1024x4096'), 1, 4).intensity).toBeCloseTo(0.4994, 4);
  });
  it('the knee batches on the book example: 6 for Q4_0 with 16-bit activations, 41 for float32 (4096 to 4096)', () => {
    const L = layer('4096x4096');
    const T = BOOK_DEVICE.threshold!;
    expect(kneeBatch('q4_0', L, 2, T)).toBe(6);
    expect(gemv('q4_0', L, 6, 2).intensity).toBeCloseTo(21.11, 2);
    expect(gemv('q4_0', L, 5, 2).intensity).toBeCloseTo(17.62, 2);
    expect(kneeBatch('f32', L, 4, T)).toBe(41);
    expect(kneeBatch('f16', L, 2, T)).toBe(21);
  });
  it('a threshold above the batch limit has no knee', () => {
    const L = layer('1024x4096');
    expect(intensityLimit(L, 4)).toBeCloseTo(409.6, 6);
    expect(kneeBatch('q4_0', L, 4, 500)).toBeNull();
  });
  it('at batch 1 every format is memory-bound on the book example, and the speed-up is just under the byte ratio', () => {
    for (const L of LAYERS)
      for (const f of FORMAT_IDS) expect(ceiling(gemv(f, L, 1, 2), BOOK_DEVICE).bound).toBe('memory');
    const s = speedupOverF32('q4_0', layer('4096x4096'), 1, 4, BOOK_DEVICE);
    expect(s).toBeLessThan(byteRatio('q4_0'));
    expect(s).toBeGreaterThan(7.08);
  });
  it('past the knee the compute roof caps the time', () => {
    const c = ceiling(gemv('q4_0', layer('4096x4096'), 64, 2), BOOK_DEVICE);
    expect(c.bound).toBe('compute');
    expect(c.gflops).toBeCloseTo(20 * 3350, 6);
  });
});

describe('fused or unfused', () => {
  it('unfused Q4_0 to float32 moves 8.5625 bytes per weight, about 2.14 times the float32 GEMV', () => {
    expect(pipelineBytesPerWeight('unfused-f32', 'q4_0')).toBe(8.5625);
    expect(pipelineBytesPerWeight('unfused-f16', 'q4_0')).toBe(4.5625);
    expect(pipelineBytesPerWeight('fused', 'q4_0')).toBe(0.5625);
    expect(pipelineBytesPerWeight('f32', 'q4_0')).toBe(4);
    expect(8.5625 / 4).toBeCloseTo(2.14, 2);
    expect(pipelineRelative('unfused-f32', 'q4_0', layer('4096x4096'), 1, 4)).toBeCloseTo(2.14, 2);
    expect(pipelineRelative('unfused-f16', 'q4_0', layer('4096x4096'), 1, 4)).toBeCloseTo(1.14, 2);
    expect(1 / pipelineRelative('fused', 'q4_0', layer('4096x4096'), 1, 4)).toBeCloseTo(7.09, 2);
  });
  it('the passes of each pipeline add up to the bytes per weight plus the activations', () => {
    const L = layer('4096x4096');
    const n = L.in * L.out;
    const io = 4 * (L.in + L.out);
    for (const p of ['f32', 'unfused-f32', 'unfused-f16', 'fused'] as const)
      expect(passBytes(pipelinePasses(p, 'q4_0', L, 1, 4))).toBeCloseTo(pipelineBytesPerWeight(p, 'q4_0') * n + io, 3);
    expect(pipelinePasses('unfused-f32', 'q4_0', L, 1, 4).map((x) => x.kind)).toEqual(['read-weights', 'write-copy', 'read-copy', 'read-x', 'write-y']);
  });
});

describe('model size and bandwidth utilisation', () => {
  it("7 billion weights: 14 GB in 16-bit (the book's p. 505), 3.94 GB at Q4_0, 7.44 at Q8_0, 28 at float32", () => {
    expect(modelBytes(7e9, 'f16') / 1e9).toBe(14);
    expect(modelBytes(7e9, 'q4_0') / 1e9).toBeCloseTo(3.9375, 6);
    expect(modelBytes(7e9, 'q8_0') / 1e9).toBeCloseTo(7.4375, 6);
    expect(modelBytes(7e9, 'f32') / 1e9).toBe(28);
  });
  it('the q-mbu example: Q4_0 7B at 30 tokens per second on a 400 GB/s device is 29.5 %', () => {
    expect(mbu(modelBytes(7e9, 'q4_0'), 1 / 30, 400)).toBeCloseTo(0.2953, 4);
  });
});

describe('reading the reports', () => {
  const reports = {
    metal: {
      generated: '2026-11-20T00:00:00', device: { name: 'test GPU' },
      results: [
        { lab: 'ch05/lab1_roofline', variant: 'solution', bench: [{ name: 'roofline', compute_gflops: 1000, bandwidth_gb_per_s: 200 }] },
        { lab: 'x2/lab1_q8gemv', variant: 'solution', bench: [{ name: 'qgemv', kernel: 'q8', in: 4096, out: 4096, batch: 1, seconds: 1e-4 }, { name: 'other', seconds: 1 }] },
        { lab: 'x2/lab2_q4gemv', variant: 'solution', bench: [{ name: 'qgemv', in: 4096, out: 4096, batch: 2, ms: 0.05, weight_bytes: 9437184 }] },
      ],
    },
    emu: { results: [{ lab: 'x2/lab1_q8gemv', variant: 'solution', bench: [{ name: 'qgemv', in: 64, out: 64, batch: 1, seconds: 1 }] }] },
  };
  it('reads qgemv rows of GPU backends only, with format and weight bytes defaulted from the lab', () => {
    const rows = qgemvRows(reports as never);
    expect(rows).toHaveLength(2);
    expect(rows[0].format).toBe('q8_0');
    expect(rows[0].weightBytes).toBe(4096 * 4096 * 1.0625);
    expect(rows[0].weightGBs).toBeCloseTo((4096 * 4096 * 1.0625) / 1e-4 / 1e9, 6);
    expect(rows[1].format).toBe('q4_0');
    expect(rows[1].seconds).toBeCloseTo(5e-5, 12);
    expect(rows[1].labNum).toBe('X2.2');
    expect(measuredBandwidth(reports as never, 'metal')).toBe(200);
    expect(measuredBandwidth(reports as never, 'cuda')).toBeNull();
  });
});

describe('your runs beside the model', () => {
  const row = (lab: string, kernel: string, format: string, seconds: number, batch = 1) => ({ name: 'qgemv', kernel, format, in: 4096, out: 4096, batch, seconds });
  const reports = {
    metal: {
      results: [
        { lab: 'x2/lab1_q8gemv', variant: 'solution', bench: [row('x2/lab1_q8gemv', 'gather', 'f32', 4e-4), row('x2/lab1_q8gemv', 'q8_0', 'q8_0', 1e-4)] },
        { lab: 'x2/lab2_q4gemv', variant: 'solution', bench: [row('x2/lab2_q4gemv', 'gather', 'f32', 5e-4), row('x2/lab2_q4gemv', 'q4_0', 'q4_0', 1e-4), row('x2/lab2_q4gemv', 'q4_0_unfused', 'q4_0', 1e-3), row('x2/lab2_q4gemv', 'q8_0', 'q8_0', 2e-4)] },
      ],
    },
  };
  const rows = qgemvRows(reports as never);
  it('picks one fused row of the format and one float32 baseline, from the lab that owns the format', () => {
    const q4 = pickRuns(rows, 'metal', { in: 4096, out: 4096 }, 1, 'q4_0');
    expect(q4.map((r) => [r.lab, r.kernel])).toEqual([['x2/lab2_q4gemv', 'gather'], ['x2/lab2_q4gemv', 'q4_0']]);
    const q8 = pickRuns(rows, 'metal', { in: 4096, out: 4096 }, 1, 'q8_0');
    expect(q8.map((r) => [r.lab, r.kernel])).toEqual([['x2/lab1_q8gemv', 'gather'], ['x2/lab1_q8gemv', 'q8_0']]);
    expect(pickRuns(rows, 'metal', { in: 4096, out: 4096 }, 4, 'q4_0')).toEqual([]);
  });
  it("reads lab X2.2's unfused and fused times relative to its float32 GEMV", () => {
    const m = measuredPipelines(rows, 'metal', { in: 4096, out: 4096 }, 1)!;
    expect(m.f32).toBe(1);
    expect(m['unfused-f32']).toBeCloseTo(2, 12);
    expect(m.fused).toBeCloseTo(0.2, 12);
    expect(measuredPipelines(rows, 'cuda', { in: 4096, out: 4096 }, 1)).toBeNull();
  });
});
