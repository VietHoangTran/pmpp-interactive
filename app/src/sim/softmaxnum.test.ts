import { describe, expect, it } from 'vitest';
import { BF16, FP16, FP32, maxFinite, round, type Format } from './minifloat';
import { SOFTMAX_PRESETS, overflowThreshold, softmaxTrace, sumTolerance } from './softmaxnum';

const preset = (id: string) => SOFTMAX_PRESETS.find((p) => p.id === id)!.logits;
const finite = (v: number) => Number.isFinite(v);

describe('overflowThreshold', () => {
  it('is about 88.72 for FP32 and BF16 and about 11.09 for FP16', () => {
    expect(overflowThreshold(FP32)).toBeCloseTo(88.7228, 4);
    expect(overflowThreshold(BF16)).toBeCloseTo(88.7209, 4);
    expect(overflowThreshold(FP16)).toBeCloseTo(11.0901, 4);
  });
  it('is exactly where exp starts to round to infinity', () => {
    for (const f of [FP32, BF16, FP16]) {
      const t = overflowThreshold(f);
      expect(round(f, Math.exp(t - 1e-9))).toBe(maxFinite(f));
      expect(round(f, Math.exp(t + 1e-9))).toBe(Infinity);
    }
  });
});

describe('q-softmax-fp16: exp stored in FP16, no max subtraction', () => {
  it('logit 11 gives a finite term (e^11 is about 59,874, stored as 59,872); 12 overflows (about 162,755 > 65,504)', () => {
    const t11 = softmaxTrace([11], 1, FP16, false).rows[0].t;
    const t12 = softmaxTrace([12], 1, FP16, false).rows[0].t;
    expect(t11).toBe(59872);
    expect(t12).toBe(Infinity);
    expect(Math.exp(12)).toBeGreaterThan(maxFinite(FP16));
  });
});

describe('a row that overflows FP16 but not FP32', () => {
  const row = preset('typical');
  it('the typical row at scale 2: the term for 7.1 (z = 14.2) is infinite in FP16', () => {
    const h = softmaxTrace(row, 2, FP16, false);
    expect(h.firstInf).toEqual({ kind: 'inf', index: 3, quantity: 't', cause: 'exp-overflow' });
    expect(h.firstNan).toEqual({ kind: 'nan', index: 3, quantity: 'p', cause: 'inf-over-inf' });
    expect(h.rows[3].z).toBe(14.203125); // 7.1 is stored as 7.1015625
    expect(h.rows.filter((r) => r.p === 0).length).toBe(7); // every other probability becomes 0
    expect(h.ok).toBe(false);
  });
  it('the same row in FP32 and BF16 is fine without the subtraction', () => {
    for (const f of [FP32, BF16]) {
      const s = softmaxTrace(row, 2, f, false);
      expect(s.firstInf).toBeNull();
      expect(s.firstNan).toBeNull();
      expect(s.ok).toBe(true);
    }
  });
  it('FP32 overflows at scale 16 (z = 113.6)', () => {
    expect(softmaxTrace(row, 16, FP32, false).firstInf).toMatchObject({ index: 3, quantity: 't' });
    expect(softmaxTrace(row, 16, FP32, true).firstInf).toBeNull();
  });
});

describe('the denominator can overflow when no term does', () => {
  it('eight logits of 10 in FP16: each term is 22,032, the third partial sum passes 65,504', () => {
    const s = softmaxTrace(preset('flat'), 1, FP16, false);
    expect(s.rows[0].t).toBe(22032);
    expect(s.rows.every((r) => finite(r.t))).toBe(true);
    expect(s.firstInf).toEqual({ kind: 'inf', index: 2, quantity: 's', cause: 'sum-overflow' });
    expect(s.firstNan).toBeNull(); // no NaN at all ...
    expect(s.rows.every((r) => r.p === 0)).toBe(true); // ... yet every probability is 0
    expect(s.sumP).toBe(0);
  });
});

describe('infinity minus infinity', () => {
  it('a logit whose scaled value overflows FP16 makes the maximum infinite, and z − max is NaN', () => {
    const s = softmaxTrace([60000, 1, 2, 3], 2, FP16, true);
    expect(s.firstInf).toEqual({ kind: 'inf', index: 0, quantity: 'z', cause: 'scaled-overflow' });
    expect(s.firstNan).toEqual({ kind: 'nan', index: 0, quantity: 'd', cause: 'inf-minus-inf' });
  });
});

describe('subtracting the maximum', () => {
  const r = (() => {
    let h = 12345;
    return () => {
      h = (h * 1103515245 + 12345) % 2147483648;
      return h / 2147483648;
    };
  })();
  it('never yields an infinity or NaN when the scaled logits are finite, in any format', () => {
    for (const f of [FP32, BF16, FP16] as Format[]) {
      const limit = f === FP16 ? 2000 : 1e6;
      for (let i = 0; i < 400; i++) {
        const row = Array.from({ length: 8 }, () => (r() * 2 - 1) * limit);
        const scale = 0.25 * 2 ** Math.floor(r() * 7);
        const s = softmaxTrace(row, scale, f, true);
        expect(s.rows.every((e) => finite(e.z))).toBe(true);
        expect(s.firstInf).toBeNull();
        expect(s.firstNan).toBeNull();
        expect(s.rows.every((e) => e.t <= 1 && finite(e.p))).toBe(true);
        expect(s.denom).toBeGreaterThanOrEqual(1);
      }
    }
  });
  it('and its probabilities sum to 1 within a few units of the format', () => {
    for (const f of [FP32, BF16, FP16] as Format[])
      for (const p of SOFTMAX_PRESETS)
        for (const scale of [0.25, 1, 4, 16]) {
          const s = softmaxTrace(p.logits, scale, f, true);
          if (s.rows.some((e) => !finite(e.z))) continue; // huge logits × 16 overflow FP16 before anything else
          expect(Math.abs(s.sumP - 1), `${f.name} ${p.id} ×${scale}`).toBeLessThanOrEqual(sumTolerance(f, 8));
          expect(s.ok).toBe(true);
        }
  });
  it('fixes the presets that overflow without it', () => {
    expect(softmaxTrace(preset('huge'), 1, FP32, false).firstInf).not.toBeNull();
    expect(softmaxTrace(preset('huge'), 1, FP32, true).ok).toBe(true);
    expect(softmaxTrace(preset('large'), 1, FP16, true).ok).toBe(true);
    expect(softmaxTrace(preset('flat'), 1, FP16, true).rows.every((e) => e.p === 0.125)).toBe(true);
  });
  it('very negative logits underflow to 0, harmlessly', () => {
    const s = softmaxTrace(preset('wide'), 1, FP16, true);
    expect(s.rows[2].t).toBe(0);
    expect(s.rows[2].underflow).toBe(true);
    expect(s.rows[5].underflow).toBe(true); // e^-26 is below FP16's smallest denormal too
    expect(s.ok).toBe(true);
    expect(softmaxTrace(preset('wide'), 1, FP32, true).rows[2].underflow).toBe(false);
  });
});
