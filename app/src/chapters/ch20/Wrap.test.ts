import { describe, expect, it } from 'vitest';
import { DOUBLE, Q8, SPEC, SPLIT, questionsWrap } from './Wrap';

// The summary page's questions take their numbers from the chapter's models. This pins
// those numbers to hand arithmetic, so that a change in a model shows up here, and checks
// that each choice question has exactly one correct option that is not the longest.

describe('summary page, chapter 20', () => {
  it('the split-KV merge: 3·e^(2−5) + 1.5', () => {
    expect(SPLIT.m).toBe(5);
    expect(SPLIT.D).toBeCloseTo(3 * Math.exp(-3) + 1.5, 12);
    expect(SPLIT.D).toBeCloseTo(1.6494, 4);
  });

  it('the first of 4 verifying queries sees 997 of 1000 keys, the last all 1000', () => {
    expect(SPEC.first).toBe(997);
    expect(SPEC.last).toBe(1000);
  });

  it("double-buffered K^T and V: 35,328 B at B_c = 16 (over 32 KiB), 17,664 B at B_c = 8", () => {
    expect(DOUBLE.bc16.single).toBe(18688);
    expect(DOUBLE.bc16.double).toBe(2 * (2112 + 2048) * 4 + 512 * 4);
    expect(DOUBLE.bc16.double).toBe(35328);
    expect(DOUBLE.bc16.double).toBeGreaterThan(DOUBLE.limit);
    expect(DOUBLE.bc8.double).toBe(17664);
    expect(DOUBLE.bc8.double).toBeLessThanOrEqual(DOUBLE.limit);
  });

  it('q8_0 cache: 65,536 values per token at 34/32 bytes is 68 KiB', () => {
    expect(Q8.bytes).toBe(2 * 32 * 8 * 128 * (34 / 32));
    expect(Q8.bytes).toBe(69632);
    expect(questionsWrap['ch20/wrap/q-q8-cache']).toMatchObject({ answer: 68 });
  });

  it('every choice question has one correct option, and it is not the longest label', () => {
    for (const q of Object.values(questionsWrap)) {
      if (q.kind !== 'choice') continue;
      expect(q.options.filter((o) => o.correct).length).toBe(1);
      const len = (l: unknown) => (typeof l === 'string' ? l.length : 0);
      const correct = len(q.options.find((o) => o.correct)!.label);
      expect(q.options.some((o) => !o.correct && len(o.label) >= correct)).toBe(true);
    }
  });
});
