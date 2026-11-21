import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { SCENE_S, compare as compareTokens } from '../../sim/tokencompare';
import { SYNTHETIC, stepLaunches, totals } from '../../sim/llamastep';
import { ROPE_BASE, theta } from '../../sim/ropepair';

// The scenes "A test that fails late" (X3.9), "Two ways to pair a head" and "One decode step"
// (X3.7) speak numbers and trends their models draw. Once the audio exists it goes on saying
// them, so each spoken claim is tied here to the model it describes, as excerpts-g2.test.ts
// does for "One framework, two softmaxes".
const repo = path.resolve(__dirname, '../../../..');
const script = (clip: string) => fs.readFileSync(path.join(repo, `content/x3/narration/${clip}.txt`), 'utf8').trim();

describe('late-divergence narration agrees with compare(SCENE_S)', () => {
  const c = compareTokens(SCENE_S);
  const CLOSE = 19;
  it('beat 2: other margins at least four tenths, often more than one; rounding about one ten-thousandth', () => {
    const s = script('late-divergence-2');
    expect(s).toContain('at least four tenths, often more than one');
    expect(s).toContain('about one ten-thousandth');
    const clear = c.steps.filter((r) => !r.flagged);
    expect(clear.length).toBeGreaterThan(c.steps.length / 2); // "at most steps"
    expect(Math.min(...clear.filter((r) => r.margin > 0.1).map((r) => r.margin))).toBeGreaterThanOrEqual(0.4);
    expect(clear.filter((r) => r.margin > 1).length).toBeGreaterThan(clear.length / 3); // "often"
    expect(c.tol).toBeGreaterThan(0.5e-4);
    expect(c.tol).toBeLessThanOrEqual(1e-4);
    // "A difference that small cannot reorder the top two": no step with a clear margin flips.
    for (const r of clear) expect(r.tfTok).toBe(r.refTok);
  });
  it('beat 3: step nineteen, six millionths apart, far below the rounding difference; the runner-up wins', () => {
    const s = script('late-divergence-3');
    expect(s).toContain('Step nineteen');
    expect(s).toContain('only six millionths apart');
    const r = c.steps[CLOSE];
    expect(r.margin).toBeCloseTo(6e-6, 12);
    expect(r.margin).toBeLessThan(c.tol / 10); // "much less"
    const sorted = r.ref.map((v, i) => [v, i] as const).sort((a, b) => b[0] - a[0]);
    expect(r.tfTok).toBe(sorted[1][1]);
    expect(r.frTok).toBe(sorted[1][1]);
  });
  it('beats 4 and 5: free running diverges from step twenty on; teacher forced, only nineteen differs and is flagged', () => {
    expect(script('late-divergence-4')).toContain('From step twenty');
    expect(script('late-divergence-4')).toContain('failure at step nineteen, and at every step after it');
    expect(c.steps[CLOSE + 1].frIn).not.toBe(c.steps[CLOSE + 1].refIn);
    expect(c.frDiffers).toEqual(Array.from({ length: c.steps.length - CLOSE }, (_, k) => CLOSE + k));
    expect(script('late-divergence-5')).toContain('only step nineteen differs');
    expect(c.tfDiffers).toEqual([CLOSE]);
    expect(c.flagged).toEqual([CLOSE]);
  });
});

describe('rope-pairs narration agrees with the RoPE model', () => {
  it('beat 2: positions 0 to 3; the first pair turns a full radian per step, each next pair ten times slower', () => {
    const s = script('rope-pairs-2');
    expect(s).toContain('from 0 to 3');
    expect(s).toContain('a full radian per step');
    expect(s).toContain('ten times slower');
    const HEADN = 8; // RopePairsScene's head size
    expect(ROPE_BASE).toBe(10000);
    expect(theta(0, HEADN)).toBe(1);
    for (let j = 1; j < HEADN / 2; j++) expect(theta(j - 1, HEADN) / theta(j, HEADN)).toBeCloseTo(10, 9);
  });
});

describe('llama-step narration agrees with the step model', () => {
  it('beat 7: matrix products dominate near the start; attention’s share rises with pos; weights stay the same', () => {
    expect(script('llama-step-7')).toContain('Near the start of a sequence the matrix products dominate');
    const all = (t: ReturnType<typeof totals>) => Object.values(t.byKind).reduce((a, b) => a + b, 0);
    const t0 = totals(stepLaunches(SYNTHETIC, 0));
    expect(t0.byKind.matmul / all(t0)).toBeGreaterThan(0.5);
    let prev = -1;
    for (let pos = 0; pos < SYNTHETIC.seqLen; pos += 8) {
      const t = totals(stepLaunches(SYNTHETIC, pos));
      const share = t.byKind.attention / all(t);
      expect(share).toBeGreaterThan(prev);
      expect(t.weightBytes).toBe(t0.weightBytes);
      prev = share;
    }
  });
});
