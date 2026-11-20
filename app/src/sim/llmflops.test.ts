import { describe, expect, it } from 'vitest';
import { BLOCKS, BOOK_SHAPE, MILESTONES, PRIMITIVES, attentionPairs, crossover, judge, layerFlops, linearMacsPerToken } from './llmflops';

const book = (dModel: number) => ({ ...BOOK_SHAPE, dModel });

describe('FLOPs per layer, from the book\'s shapes', () => {
  it('hand count at N = 4, d_model = 2 (prefill, no skipping)', () => {
    const f = layerFlops(4, book(2), 'prefill');
    expect(f.proj).toBe(2 * 4 * 4 * 2 * 2); // 2 FLOP x 4 tokens x 4 d^2
    expect(f.ffn).toBe(2 * 4 * 2 * 2 * 8); // 2 FLOP x 4 tokens x 2 matrices x d x 4d
    expect(f.qk).toBe(2 * 16 * 2);
    expect(f.pv).toBe(2 * 16 * 2);
    expect(f.total).toBe(f.proj + f.ffn + f.qk + f.pv);
  });
  it('linear layers are 24 d_model^2 FLOPs per token, the products 4 N d_model', () => {
    const d = 4096;
    expect(2 * linearMacsPerToken(book(d))).toBe(24 * d * d);
    const f = layerFlops(4096, book(d), 'prefill');
    expect((f.qk + f.pv) / 4096).toBe(4 * 4096 * d);
  });
  it('at N = 4096, d_model = 4096 the products are 1/6 of the linear layers (question q-flops-share)', () => {
    expect(layerFlops(4096, book(4096), 'prefill').attnOverLinear).toBeCloseTo(1 / 6, 12);
    expect(layerFlops(4096, book(4096), 'decode').attnOverLinear).toBeCloseTo(1 / 6, 12);
  });
  it('the crossover is N = 6 d_model for the book\'s shape, in prefill and in decode', () => {
    expect(crossover(book(4096), 'prefill')).toBe(6 * 4096);
    expect(crossover(book(4096), 'decode')).toBe(6 * 4096);
    expect(layerFlops(6 * 4096, book(4096), 'prefill').attnOverLinear).toBeCloseTo(1, 12);
  });
  it('skipping the masked half roughly doubles the crossover', () => {
    const s = { ...book(1024), causalSkip: true };
    const n = crossover(s, 'prefill');
    expect(n).toBe(12 * 1024 - 1);
    expect(layerFlops(n, s, 'prefill').attnOverLinear).toBeCloseTo(1, 12);
    expect(attentionPairs(4, 'prefill', true)).toBe(10);
    expect(attentionPairs(4, 'decode', true)).toBe(4);
  });
  it('a three-matrix SwiGLU feed-forward moves the crossover out', () => {
    const s = { ...book(4096), ffn: 'swiglu' as const, ffRatio: 3.5 };
    expect(crossover(s, 'prefill')).toBe(((4 + 3 * 3.5) * 4096) / 2);
  });
});

describe('the blocks of Fig. 20.1', () => {
  it('has thirteen blocks with valid answers and a reason each', () => {
    expect(BLOCKS).toHaveLength(13);
    for (const b of BLOCKS) {
      expect(b.primitive.length).toBeGreaterThan(0);
      expect(b.primitive.every((p) => PRIMITIVES.includes(p))).toBe(true);
      expect(b.milestone.every((m) => MILESTONES.includes(m))).toBe(true);
      expect(b.why.length).toBeGreaterThan(20);
    }
  });
  it('only the three attention steps are new: milestone M6, primitive fused attention', () => {
    const m6 = BLOCKS.filter((b) => b.milestone.includes('M6')).map((b) => b.id);
    expect(m6).toEqual(['qkt', 'softmax', 'pv']);
    expect(BLOCKS.filter((b) => b.primitive.includes('attention')).map((b) => b.id)).toEqual(['qkt', 'softmax', 'pv']);
  });
  it('LayerNorm is M4-shaped (question q-layernorm)', () => {
    const norm = BLOCKS.find((b) => b.id === 'norm')!;
    expect(judge(norm, 'rowreduce', 'M4')).toEqual({ primitive: true, milestone: true });
    expect(judge(norm, 'gemm', 'M1-M3')).toEqual({ primitive: false, milestone: false });
    expect(judge(norm, null, null)).toEqual({ primitive: false, milestone: false });
  });
});
