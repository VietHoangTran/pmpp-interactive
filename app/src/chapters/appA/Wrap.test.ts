import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { sumSequential, sumTree } from '../../sim/fpsum';
import { BF16, E4M3, FP16, FP32, classify, maxFinite, minNormal, minSubnormal, round, ulp } from '../../sim/minifloat';

// Every number the summary page's mixed check states, computed with the same models the
// sections use. The page's text is read too, so a number changed there without changing
// the model fails here.
const page = fs.readFileSync(path.join(__dirname, 'Wrap.tsx'), 'utf8');
const flat = page.replace(/\s+/g, ' ');

describe('q-rms-half: a norm\'s sum of squares in FP16', () => {
  const squares = Array<number>(4096).fill(4 * 4);
  it('the loop stalls at 32,768, on a tie that goes to the even neighbour', () => {
    expect(sumSequential(squares, 'fp16')).toBe(32768);
    expect(ulp(FP16, 32768)).toBe(32); // spacing above 32,768
    expect(ulp(FP16, 32767)).toBe(16); // and below it: every multiple of 16 up to 32,768 is exact
    expect(round(FP16, 32768 + 16)).toBe(32768); // the tie, to the even code
  });
  it('the halving tree reaches 65,536, which overflows FP16', () => {
    expect(sumTree(squares, 'fp16')).toBe(Infinity);
    expect(maxFinite(FP16)).toBe(65504);
    expect(round(FP16, 65536)).toBe(Infinity);
  });
  it('in FP32 both are exact', () => {
    expect(sumSequential(squares, 'fp32')).toBe(65536);
    expect(sumTree(squares, 'fp32')).toBe(65536);
  });
  it('the loop\'s answer gives a root mean square of about 2.83 and outputs about 1.41 times too large', () => {
    const rms = Math.sqrt(32768 / 4096);
    expect(rms).toBeCloseTo(2.83, 2);
    expect(4 / rms).toBeCloseTo(1.41, 2);
    expect(flat).toContain('32,768');
    expect(flat).toContain('2.83');
    expect(flat).toContain('1.41');
  });
});

describe('q-bf16-argmax: logits cast to BF16', () => {
  it('the spacing near 10 is 0.0625, and 10.01 and 10.03 both become 10', () => {
    expect(ulp(BF16, 10)).toBe(0.0625);
    expect(round(BF16, 10.01)).toBe(10);
    expect(round(BF16, 10.03)).toBe(10);
    // in FP32 they stay apart
    expect(round(FP32, 10.01)).not.toBe(round(FP32, 10.03));
    expect(flat).toContain('answer: 0.0625');
  });
});

describe('q-logprob-ftz: a logit 90 below the maximum', () => {
  it('exp(-90) is an FP32 denormal, so a device that flushes gives 0', () => {
    const p = Math.exp(-90);
    expect(p).toBeLessThan(minNormal(FP32));
    expect(p).toBeGreaterThan(minSubnormal(FP32)!);
    expect(classify(FP32, p).cls).toBe('subnormal');
    expect(Math.log(0)).toBe(-Infinity);
    expect(flat).toContain('8.2 × 10');
    expect(p).toBeGreaterThan(8.1e-40);
    expect(p).toBeLessThan(8.3e-40);
  });
});

describe('q-e4m3-small: an unscaled activation in E4M3', () => {
  it('0.0006 is below half the smallest E4M3 value, 2^-9, and rounds to 0', () => {
    expect(minSubnormal(E4M3)).toBe(2 ** -9);
    expect(0.0006).toBeLessThan(2 ** -10);
    expect(round(E4M3, 0.0006)).toBe(0);
    // with a power-of-two scale of 2^10 first, it keeps a value within E4M3's precision
    expect(round(E4M3, 0.0006 * 1024) / 1024).toBeCloseTo(0.0006, 4);
    expect(flat).toContain('answer: 0,');
  });
});
