import { describe, expect, it } from 'vitest';
import {
  GiB,
  KiB,
  MiB,
  PRESETS,
  aiGQA,
  aiMHA,
  aiMLA,
  aiMLAPrinted,
  aiMQA,
  asVariant,
  batchToReach,
  batchedDecodeAttentionAI,
  bytesPerToken,
  bytesPerTokenPrinted,
  cacheTotal,
  decodeAI,
  eq207,
  eq208,
  eq210,
  eq212,
  eq214,
  gqaTerms,
  kvHeadOf,
  linearAI,
  maxBatch,
  presetByKey,
  stepFloorSeconds,
  toFlopPerByte,
  tokensThatFit,
  variantOf,
} from './kvmath';

const shape = (key: string) => presetByKey(key)!.shape;

describe('the book’s three examples (p. 506) and facts F1', () => {
  it('“PaLM 2” as printed: 8.0 GiB', () => {
    expect(eq208(1, 4096, 64, 32, 256, 2)).toBe(8 * GiB);
    expect(cacheTotal(shape('palm2-printed'), 1, 4096, 2)).toBe(8 * GiB);
  });
  it('PaLM 62B with its real multi-query attention: 256 MiB, 32 times smaller', () => {
    expect(eq210(1, 4096, 64, 256, 2)).toBe(256 * MiB);
    expect(cacheTotal(shape('palm-62b'), 1, 4096, 2)).toBe(256 * MiB);
    expect(cacheTotal(shape('palm2-printed'), 1, 4096, 2) / cacheTotal(shape('palm-62b'), 1, 4096, 2)).toBe(32);
  });
  it('GPT-3: 18 GiB at the book’s 4,096 tokens, 9 GiB at its real 2,048 (question q-gpt3)', () => {
    expect(eq208(1, 4096, 96, 96, 128, 2)).toBe(18 * GiB);
    expect(cacheTotal(shape('gpt3'), 1, 2048, 2)).toBe(9 * GiB);
    expect(eq208(1, 4096, 96, 96, 128, 2) / 1e9).toBeCloseTo(19.33, 2);
  });
  it('Llama 2 7B: 512 KiB per token (Eq. 20.7), 2 GiB at 4,096 tokens', () => {
    expect(eq207(32, 32, 128, 2)).toBe(512 * KiB);
    expect(cacheTotal(shape('llama2-7b'), 1, 4096, 2)).toBe(2 * GiB);
  });
  it('Mistral 7B (8 K/V heads): 128 KiB per token, 512 MiB at 4,096 tokens', () => {
    expect(bytesPerToken(shape('mistral-7b'), 2)).toBe(128 * KiB);
    expect(cacheTotal(shape('mistral-7b'), 1, 4096, 2)).toBe(512 * MiB);
    expect(eq212(1, 4096, 32, 32, 4, 128, 2)).toBe(512 * MiB);
  });
  it('DeepSeek-V2 MLA: 69,120 bytes per token; Eq. 20.14 with d = 128 gives 15,360 (question q-mla-bytes)', () => {
    expect(bytesPerToken(shape('deepseek-v2'), 2)).toBe(69_120);
    expect(eq214(1, 1, 60, 128, 2)).toBe(15_360);
    expect(bytesPerTokenPrinted(shape('deepseek-v2'), 2)).toBe(15_360);
    expect(69_120 / 15_360).toBe(4.5);
  });
  it('DeepSeek-V2’s MHA equivalent is 3.75 MiB per token, 56.9 times its MLA cache', () => {
    const mha = asVariant(shape('deepseek-v2'), 'MHA');
    expect(bytesPerToken(mha, 2)).toBe(3.75 * MiB);
    expect(bytesPerToken(mha, 2) / bytesPerToken(shape('deepseek-v2'), 2)).toBeCloseTo(56.9, 1);
  });
  it('question q-batch-fit: 8 GiB left holds four 4,096-token Llama 2 7B sequences', () => {
    expect(maxBatch(8 * GiB, 0, shape('llama2-7b'), 4096, 2)).toBe(4);
    expect(maxBatch(22 * GiB, 14 * GiB, shape('llama2-7b'), 4096, 2)).toBe(4);
    expect(maxBatch(10 * GiB, 14 * GiB, shape('llama2-7b'), 4096, 2)).toBe(0);
  });
  it('tokens that fit in 1 GiB of Llama 2 7B cache: 2,048', () => {
    expect(tokensThatFit(GiB, shape('llama2-7b'), 2)).toBe(2048);
  });
});

describe('decode intensity (Eqs. 20.9 to 20.15; facts E4, F1)', () => {
  it('MHA is about one multiply-add per element', () => {
    expect(aiMHA(4096)).toBeCloseTo(0.99976, 5);
  });
  it('GQA at N = 4096, h_q = 32, g_q = 1, 4, 8, 32: 1.000, 3.996, 7.984, 31.752 (question q-gqa-ai)', () => {
    expect(aiGQA(4096, 32, 1)).toBeCloseTo(1.0, 3);
    expect(aiGQA(4096, 32, 4)).toBeCloseTo(3.996, 3);
    expect(aiGQA(4096, 32, 8)).toBeCloseTo(7.984, 3);
    expect(aiGQA(4096, 32, 32)).toBeCloseTo(31.752, 3);
    expect(aiGQA(4096, 32, 32)).toBeCloseTo(aiMQA(4096, 32), 12);
  });
  it('MLA at N = 4096, h_q = 32: 63.02 per [13], 63.50 as printed', () => {
    expect(aiMLA(4096, 32)).toBeCloseTo(63.02, 2);
    expect(aiMLAPrinted(4096, 32)).toBeCloseTo(63.5, 2);
  });
  it('as FLOP per byte: MHA is 1 in 16-bit and 0.5 in fp32 (question q-fp32-ai)', () => {
    expect(toFlopPerByte(aiMHA(4096), 2)).toBeCloseTo(1, 3);
    expect(toFlopPerByte(aiMHA(4096), 4)).toBeCloseTo(0.5, 3);
  });
  it('decodeAI picks the right formula from the shape', () => {
    expect(decodeAI(shape('llama2-7b'), 4096)).toBe(aiMHA(4096));
    expect(decodeAI(shape('mistral-7b'), 4096)).toBe(aiGQA(4096, 32, 4));
    expect(decodeAI(shape('palm-62b'), 4096)).toBe(aiMQA(4096, 32));
    expect(decodeAI(shape('deepseek-v2'), 4096)).toBe(aiMLA(4096, 128));
  });
  it('variants are recognised', () => {
    expect(PRESETS.map((p) => variantOf(p.shape))).toEqual(['MHA', 'MHA', 'MHA', 'MQA', 'GQA', 'GQA', 'MLA']);
  });
});

describe('batching (p. 505, Fig. 20.17)', () => {
  it('the 1,024-to-4,096 linear layer in 16-bit: 0.9988 at batch 1, 315.08 at batch 512 (question q-linear-ai)', () => {
    expect(linearAI(1, 1024, 4096, 2)).toBeCloseTo(0.9988, 4);
    expect(linearAI(512, 1024, 4096, 2)).toBeCloseTo(315.08, 2);
  });
  it('rises with the batch; decode attention does not move', () => {
    for (const b of [1, 2, 8, 64, 512]) {
      expect(linearAI(b * 2, 1024, 4096, 2)).toBeGreaterThan(linearAI(b, 1024, 4096, 2));
      expect(batchedDecodeAttentionAI(b, 4096, 2)).toBe(batchedDecodeAttentionAI(1, 4096, 2));
    }
  });
  it('batch needed to reach a ridge', () => {
    const b = batchToReach(100, 1024, 4096, 2)!;
    expect(linearAI(b, 1024, 4096, 2)).toBeGreaterThanOrEqual(100);
    expect(linearAI(b - 1, 1024, 4096, 2)).toBeLessThan(100);
    // The limit as b grows is 2 * 1024 * 4096 / (2 * 5120) = 819.2: a ridge above it is never reached.
    expect(batchToReach(900, 1024, 4096, 2)).toBeNull();
  });
});

describe('heads: who reads which K/V head (facts A8, D1, E3)', () => {
  it('FlashAttention’s example: 6 query heads, 2 K/V heads; head 4 reads K/V head 1 (question q-head-map)', () => {
    expect([0, 1, 2, 3, 4, 5].map((h) => kvHeadOf(h, 6, 2))).toEqual([0, 0, 0, 1, 1, 1]);
  });
  it('Mistral 7B: g_q = 4 queries per K/V head, G = 8 groups (question q-gq)', () => {
    expect(gqaTerms(32, 8)).toEqual({ gq: 4, G: 8 });
  });
  it('MHA and MQA as the two ends', () => {
    expect([0, 5, 31].map((h) => kvHeadOf(h, 32, 32))).toEqual([0, 5, 31]);
    expect([0, 5, 31].map((h) => kvHeadOf(h, 32, 1))).toEqual([0, 0, 0]);
  });
});

describe('floor time', () => {
  it('reads every byte once at the bandwidth', () => {
    expect(stepFloorSeconds(14e9, 2e9, 100e9)).toBeCloseTo(0.16, 12);
  });
});
