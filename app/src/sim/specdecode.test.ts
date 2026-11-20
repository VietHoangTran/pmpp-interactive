import { describe, expect, it } from 'vitest';
import { FIG_PROMPT, FIG_STEPS, expectedPerPass, figCounters, meanPerPass, replayFigure, simulate } from './specdecode';

describe('Fig. 20.18 replayed', () => {
  const [r1, r2] = replayFigure();
  it('round 1: two ticks and a cross; "Efficient" is rejected for "Parallel"', () => {
    expect(r1.prefix).toEqual(FIG_PROMPT);
    expect(r1.draft).toEqual(['Programming', 'Massively', 'Efficient']);
    expect(r1.target).toEqual(['Programming', 'Massively', 'Parallel']);
    expect(r1.ok).toEqual([true, true, false]);
    expect(r1.accepted).toEqual(['Programming', 'Massively']);
  });
  it('round 2 starts after the last accepted draft token and accepts all three', () => {
    expect(r2.prefix.join(' ')).toBe('The title of this book is Programming Massively');
    expect(r2.ok).toEqual([true, true, true]);
    expect([...r2.prefix, ...r2.accepted].join(' ')).toBe('The title of this book is Programming Massively Parallel Processors EOS');
  });
  it('the replay counts two target passes for five tokens', () => {
    expect(FIG_STEPS.length).toBe(7);
    expect(figCounters(6)).toEqual({ targetPasses: 2, draftTokens: 6, tokens: ['Programming', 'Massively', 'Parallel', 'Processors', 'EOS'] });
    expect(figCounters(3)).toEqual({ targetPasses: 1, draftTokens: 3, tokens: ['Programming', 'Massively'] });
    expect(figCounters(0).targetPasses).toBe(0);
  });
});

describe('the simulator', () => {
  it('closed forms by hand', () => {
    expect(expectedPerPass(3, 0.5, false)).toBeCloseTo(0.5 + 0.25 + 0.125, 12);
    expect(expectedPerPass(3, 0.5, true)).toBeCloseTo(1 + 0.5 + 0.25, 12);
    expect(expectedPerPass(4, 1, false)).toBe(4);
    expect(expectedPerPass(1, 0.3, true)).toBe(1);
  });
  it('with acceptance 1 every round keeps k tokens', () => {
    const r = simulate({ k: 4, alpha: 1, seed: 1, tokens: 40, keepTarget: false });
    expect(r.targetPasses).toBe(10);
    expect(r.tokensPerPass).toBe(4);
    expect(r.cacheReadsPerToken).toBe(0.25);
    expect(r.rowsPerRead).toBe(4);
  });
  it('the same seed gives the same run', () => {
    const a = simulate({ k: 5, alpha: 0.7, seed: 9, tokens: 100, keepTarget: true });
    const b = simulate({ k: 5, alpha: 0.7, seed: 9, tokens: 100, keepTarget: true });
    expect(a).toEqual(b);
  });
  it('rounds respect the rule: the figure keeps the matched prefix, the variant one more (at most k)', () => {
    const f = simulate({ k: 3, alpha: 0.6, seed: 4, tokens: 200, keepTarget: false });
    expect(f.rounds.every((r) => r.kept === r.matched && r.matched <= 3)).toBe(true);
    const v = simulate({ k: 3, alpha: 0.6, seed: 4, tokens: 200, keepTarget: true });
    expect(v.rounds.every((r) => r.kept === Math.min(3, r.matched + 1) && r.kept >= 1)).toBe(true);
  });
  for (const [k, alpha] of [[1, 0.5], [3, 0.6], [5, 0.8], [8, 0.9], [8, 0.3]] as const)
    for (const keep of [false, true])
      it(`k = ${k}, alpha = ${alpha}, ${keep ? 'keeping the target token' : 'the figure’s rule'}: mean over 400 seeds within 2% of the closed form`, () => {
        const m = meanPerPass(k, alpha, keep, 400);
        const e = expectedPerPass(k, alpha, keep);
        expect(Math.abs(m - e) / e).toBeLessThan(0.02);
      });
  it('under the figure’s rule, speculation needs alpha above one half to beat plain decoding even at great depth', () => {
    expect(expectedPerPass(8, 0.45, false)).toBeLessThan(1);
    expect(expectedPerPass(8, 0.6, false)).toBeGreaterThan(1);
  });
});
