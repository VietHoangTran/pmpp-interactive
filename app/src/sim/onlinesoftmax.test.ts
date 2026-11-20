import { describe, expect, it } from 'vitest';
import {
  DEMO_SCORES,
  SENTINEL,
  V12,
  exactOver,
  mergeD,
  mergeStates,
  partial,
  partialState,
  presetScores,
  runOnline,
  tileOrder,
  type Form,
  type InitMax,
  type OnlineConfig,
  type PresetId,
  type TileOrder,
} from './onlinesoftmax';

const base = (over: Partial<OnlineConfig> = {}): OnlineConfig => ({
  scores: DEMO_SCORES,
  V: V12,
  bc: 4,
  row: 11,
  order: 'ltr',
  init: 'neg-inf',
  form: 'deferred',
  ...over,
});
const WIDTHS = [1, 2, 3, 4, 6, 12];
const ORDERS: TileOrder[] = ['ltr', 'rtl', 'shuffled'];
const FORMS: Form[] = ['deferred', 'normalised'];
const INITS: InitMax[] = ['neg-inf', 'sentinel'];
const PRESETS: PresetId[] = ['demo', 'random', 'late', 'offset'];

describe('Eq. 20.4 on the hand example A = {1, 3}, B = {2, 5}', () => {
  const A = partial([1, 3]);
  const B = partial([2, 5]);
  it('after A, m = 3 and D = 1.135', () => {
    expect(A.m).toBe(3);
    expect(A.D).toBeCloseTo(1.1353, 4);
  });
  it('B multiplies the old D by e^(3 − 5) = 0.135 and gives D = 1.203', () => {
    const r = mergeD(A.m, A.D, B.m, B.D);
    expect(r.m).toBe(5);
    expect(r.aFactor).toBeCloseTo(0.1353, 4);
    expect(r.bFactor).toBe(1);
    expect(r.D).toBeCloseTo(1.2034, 4);
    // the same as summing every exponential against the final max
    const direct = Math.exp(1 - 5) + Math.exp(3 - 5) + Math.exp(2 - 5) + Math.exp(5 - 5);
    expect(r.D).toBeCloseTo(direct, 12);
  });
  it('the order of the merge does not matter', () => {
    const ab = mergeD(A.m, A.D, B.m, B.D);
    const ba = mergeD(B.m, B.D, A.m, A.D);
    expect(ab.D).toBeCloseTo(ba.D, 12);
  });
});

describe('the recurrence equals full softmax', () => {
  it('for every tile width, order, form, start value and preset (full row)', () => {
    for (const p of PRESETS)
      for (const bc of WIDTHS)
        for (const order of ORDERS)
          for (const form of FORMS)
            for (const init of INITS) {
              const t = runOnline(base({ scores: presetScores(p), bc, order, form, init }));
              expect(t.nan, `${p} ${bc} ${order} ${form} ${init}`).toBe(false);
              expect(t.diff, `${p} ${bc} ${order} ${form} ${init}`).toBeLessThan(5e-5);
            }
  });
  it('for every causal row, left to right, from either start', () => {
    for (let row = 0; row < 12; row++)
      for (const bc of WIDTHS)
        for (const init of INITS) {
          const t = runOnline(base({ row, bc, init }));
          expect(t.nan).toBe(false);
          expect(t.diff).toBeLessThan(1e-6);
          expect(t.exact).toEqual(exactOver(DEMO_SCORES, V12, Array.from({ length: row + 1 }, (_, j) => j)));
        }
  });
  it('after every tile the estimate equals exact attention over the columns seen so far', () => {
    for (const order of ORDERS) {
      const t = runOnline(base({ bc: 3, order }));
      for (const s of t.steps) expect(s.diff!).toBeLessThan(1e-6);
    }
  });
  it('both forms give the same denominator and output at every step', () => {
    for (const bc of WIDTHS) {
      const a = runOnline(base({ bc, form: 'deferred' }));
      const b = runOnline(base({ bc, form: 'normalised' }));
      a.steps.forEach((s, i) => {
        expect(s.D).toBeCloseTo(b.steps[i].D, 5);
        s.out.forEach((x, c) => expect(x).toBeCloseTo(b.steps[i].out[c], 5));
      });
    }
  });
});

describe('the A and B terms of Eq. 20.4 (the corrected p. 502 attribution)', () => {
  it('scene row, B_c = 4: tile 2 raises the max from 1.6 to 2.9; tile 3 does not, its factor is exactly 1', () => {
    const t = runOnline(base());
    const [t1, t2, t3] = t.steps;
    expect(t1.mB).toBeCloseTo(1.6, 6);
    expect(t2.rose).toBe(true);
    expect(t2.mOld).toBeCloseTo(1.6, 6);
    expect(t2.mNew).toBeCloseTo(2.9, 6);
    expect(t2.aFactor).toBeCloseTo(Math.exp(1.6 - 2.9), 6);
    expect(t2.bFactor).toBe(1);
    expect(t3.rose).toBe(false);
    expect(t3.aFactor).toBe(1);
    expect(t3.bFactor!).toBeCloseTo(Math.exp(2.1 - 2.9), 6);
  });
  it('the B term the code adds equals D_B · e^(m_B − m_new)', () => {
    for (const s of runOnline(base({ bc: 3, scores: presetScores('random') })).steps)
      expect(s.bTerm).toBeCloseTo(s.DB * s.bFactor!, 5);
  });
});

describe('the start value of the running max (Exercise 3; facts A5, A7)', () => {
  const early = { row: 2, bc: 4, order: 'rtl' as TileOrder };
  it('minus infinity with a fully masked first tile gives NaN in the code form', () => {
    const t = runOnline(base({ ...early, init: 'neg-inf' }));
    expect(t.steps[0].visible).toEqual([]);
    expect(Number.isNaN(t.steps[0].aFactor)).toBe(true);
    expect(t.steps[0].nan).toBe(true);
    expect(t.nan).toBe(true);
    // D itself stays finite: only the unconditional rescale of O meets (−inf) − (−inf).
    expect(Number.isNaN(t.steps[0].D)).toBe(false);
  });
  it('the finite sentinel -FLT_MAX/2 is safe in the same order', () => {
    const t = runOnline(base({ ...early, init: 'sentinel' }));
    expect(t.steps[0].aFactor).toBe(1);
    expect(t.nan).toBe(false);
    expect(t.diff).toBeLessThan(1e-6);
  });
  it('left to right never meets the hazard: tile 0 always holds a visible column', () => {
    for (let row = 0; row < 12; row++) expect(runOnline(base({ row, init: 'neg-inf' })).nan).toBe(false);
  });
  it('the normalised form skips an empty tile (Eq. 20.5 would be 0/0)', () => {
    const t = runOnline(base({ ...early, form: 'normalised' }));
    expect(t.steps[0].skipped).toBe(true);
    expect(t.nan).toBe(false);
  });
  it('the sentinel is -FLT_MAX/2 as a float32', () => {
    expect(SENTINEL).toBe(Math.fround(-3.4028234663852886e38 / 2));
    expect(Number.isFinite(SENTINEL)).toBe(true);
  });
});

describe('merging partial results of separate blocks (the scene’s last beat)', () => {
  it('columns 0–7 and 8–11 computed apart and merged give exact attention', () => {
    const A = partialState(DEMO_SCORES, V12, [0, 1, 2, 3, 4, 5, 6, 7]);
    const B = partialState(DEMO_SCORES, V12, [8, 9, 10, 11]);
    const M = mergeStates(A, B);
    const ex = exactOver(DEMO_SCORES, V12, Array.from({ length: 12 }, (_, j) => j))!;
    M.O.forEach((x, c) => expect(x / M.D).toBeCloseTo(ex[c], 12));
    expect(A.m).toBeCloseTo(2.9, 12);
    expect(B.m).toBeCloseTo(2.1, 12);
    expect(M.aFactor).toBe(1);
  });
});

describe('presets', () => {
  it('+100: e^(largest score) overflows float32, the merged result does not', () => {
    const t = runOnline(base({ scores: presetScores('offset') }));
    expect(t.naiveExpOfMax).toBe(Infinity);
    expect(t.nan).toBe(false);
    expect(t.diff).toBeLessThan(5e-5);
  });
  it('late: the max sits in the last column', () => {
    const s = presetScores('late');
    expect(s.indexOf(Math.max(...s))).toBe(11);
  });
  it('tile orders: shuffled is a permutation that differs from the book’s', () => {
    for (const n of [2, 3, 4, 6, 12]) {
      const o = tileOrder(n, 'shuffled');
      expect([...o].sort((a, b) => a - b)).toEqual(Array.from({ length: n }, (_, i) => i));
      expect(o).not.toEqual(Array.from({ length: n }, (_, i) => i));
    }
  });
});
