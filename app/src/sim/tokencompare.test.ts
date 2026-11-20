import { describe, expect, it } from 'vitest';
import {
  CLOSE_STEPS,
  NMSE_LLAMACPP,
  SCENE_S,
  START_S,
  STEPS,
  argmax,
  compare,
  flipThreshold,
  maxAbsErr,
  modelLogits,
  nmse,
  noiseAt,
  topTwoMargin,
} from './tokencompare';

// Pins what section X3.9 and the scene "A test that fails late" state about the toy model.

describe('NMSE as test-backend-ops computes it (D7)', () => {
  it('(1, 2, 2) against (1, 2, 2.03) is 1e-4, a thousand times the default threshold', () => {
    const v = nmse([1, 2, 2], [1, 2, 2.03]);
    expect(v).toBeCloseTo(1e-4, 12);
    expect(v / NMSE_LLAMACPP).toBeCloseTo(1000, 6);
  });
  it('is zero for identical rows and normalised by the reference', () => {
    expect(nmse([3, 4], [3, 4])).toBe(0);
    expect(nmse([30, 40], [30, 40.03])).toBeCloseTo(nmse([3, 4], [3, 4.003]), 15);
  });
});

describe('the toy model', () => {
  it('greedy takes the first maximum', () => {
    expect(argmax([1, 3, 3, 2])).toBe(1);
  });
  it('builds the three close steps with the stated margins', () => {
    const c = compare(0);
    for (const [t, m] of Object.entries(CLOSE_STEPS)) expect(c.steps[+t].margin).toBeCloseTo(m, 12);
    const others = c.steps.filter((r) => !(r.t in CLOSE_STEPS));
    expect(Math.min(...others.map((r) => r.margin))).toBeGreaterThan(0.4);
  });
  it('the noise lies in [-1, 1]', () => {
    for (let t = 0; t < STEPS; t++) for (const u of noiseAt(t)) expect(Math.abs(u)).toBeLessThanOrEqual(1);
  });
});

describe('two tests', () => {
  it('zero perturbation gives identical streams', () => {
    const c = compare(0);
    expect(c.frDiffers).toEqual([]);
    expect(c.tfDiffers).toEqual([]);
    expect(c.firstDivergence).toBeNull();
    expect(c.maxTfNmse).toBe(0);
    expect(c.steps.every((r) => r.frTok === r.refTok && r.tfTok === r.refTok)).toBe(true);
  });
  it('a flip happens only at steps whose margin is below twice the maximum error', () => {
    for (let e = -7; e <= -2; e += 0.1) {
      const c = compare(10 ** e);
      for (const r of c.steps) if (r.tfTok !== r.refTok) expect(r.margin).toBeLessThan(2 * r.tfMaxErr);
      for (const t of c.tfDiffers) expect(c.flagged).toContain(t);
    }
  });
  it('the three close steps flip in turn as s grows, and nothing else flips below 1e-2', () => {
    const th = [19, 13, 6].map((t) => flipThreshold(t)!);
    expect(th[0]).toBeGreaterThan(1e-6);
    expect(th[0]).toBeLessThan(1e-5);
    expect(th[1]).toBeGreaterThan(1e-4);
    expect(th[1]).toBeLessThan(1e-3);
    expect(th[2]).toBeGreaterThan(1e-3);
    expect(th[2]).toBeLessThan(1e-2);
    for (let t = 0; t < STEPS; t++) if (!(t in CLOSE_STEPS)) expect((flipThreshold(t) ?? 1) >= 1e-2).toBe(true);
    // the thresholds are exact: just below, no flip; just above, a flip
    for (const [t, s] of [[19, th[0]], [13, th[1]], [6, th[2]]]) {
      expect(compare(s * 0.999).tfDiffers).not.toContain(t);
      expect(compare(s * 1.001).tfDiffers).toContain(t);
    }
  });
  it('free running cascades from the first flip; teacher forcing keeps it to the flipped steps', () => {
    const c = compare(SCENE_S);
    expect(c.tfDiffers).toEqual([19]);
    expect(c.firstDivergence).toBe(19);
    expect(c.frDiffers).toEqual([19, 20, 21, 22, 23]);
    expect(c.flagged).toEqual([19]);
    // NMSE stays far below llama.cpp's threshold while the token flips
    expect(c.maxTfNmse).toBeLessThan(NMSE_LLAMACPP / 100);
    // after the divergence, the free-running logits come from a different input
    expect(c.steps[20].frIn).not.toBe(c.steps[20].refIn);
    expect(c.steps[20].frNmse).toBeGreaterThan(0.1);
  });
  it('at 1e-3 two steps flip and eleven free-running tokens differ, with every NMSE under 1e-7', () => {
    const c = compare(1e-3);
    expect(c.tfDiffers).toEqual([13, 19]);
    expect(c.firstDivergence).toBe(13);
    expect(c.frDiffers.length).toBe(11);
    expect(c.maxTfNmse).toBeLessThan(NMSE_LLAMACPP);
  });
  it('a step can be flagged without flipping', () => {
    const c = compare(10 ** -3.2);
    expect(c.flagged).toEqual([13, 19]);
    expect(c.tfDiffers).toEqual([19]);
  });
  it('the explorer starts with nothing flipped and nothing flagged', () => {
    const c = compare(START_S);
    expect(c.frDiffers).toEqual([]);
    expect(c.flagged).toEqual([]);
  });
  it('helpers', () => {
    expect(topTwoMargin([1, 5, 4.5, 2])).toBeCloseTo(0.5, 12);
    expect(maxAbsErr([1, 2], [1.5, 1])).toBe(1);
    expect(modelLogits(1, 0).length).toBe(16);
  });
});
