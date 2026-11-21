import { describe, expect, it } from 'vitest';
import { committedPrediction, displayOrder, type ChoiceQ } from './Ask';

const q = (id: string, n: number, fixedOrder?: boolean): ChoiceQ => ({
  kind: 'choice', id, prompt: '', fixedOrder,
  options: Array.from({ length: n }, (_, i) => ({ label: String(i), correct: i === 0 })),
});

describe('displayOrder', () => {
  it('is a permutation of the written indices', () => {
    for (const n of [2, 3, 4, 5]) expect([...displayOrder(q('ch05/5.1/q-x', n))].sort()).toEqual([...Array(n).keys()]);
  });
  it('is the same on every call for the same id', () => {
    expect(displayOrder(q('ch10/10.3/q-tree', 4))).toEqual(displayOrder(q('ch10/10.3/q-tree', 4)));
  });
  it('keeps the written order when asked to', () => {
    expect(displayOrder(q('ch05/5.1/q-x', 4, true))).toEqual([0, 1, 2, 3]);
  });
  it('spreads the first written option over all positions across many ids', () => {
    const at = [0, 0, 0, 0];
    for (let k = 0; k < 400; k++) at[displayOrder(q(`ch${k % 24}/s${k}/q-${k * 7}`, 4)).indexOf(0)]++;
    for (const c of at) expect(c).toBeGreaterThan(60); // about 100 each
  });
});

describe('committedPrediction', () => {
  it('takes a stored number as the numeric prediction', () => {
    expect(committedPrediction(2.5)).toBe(2.5);
    expect(committedPrediction(0)).toBe(0);
    expect(committedPrediction(-3)).toBe(-3);
  });
  it('treats nothing, text, arrays and non-finite values as not committed', () => {
    for (const v of [undefined, null, '2', [1, 2], { a: 1 }, NaN, Infinity]) expect(committedPrediction(v)).toBeNull();
  });
  it('takes a choice only as the written index of an existing option', () => {
    expect(committedPrediction(0, 4)).toBe(0);
    expect(committedPrediction(3, 4)).toBe(3);
    expect(committedPrediction(4, 4)).toBeNull(); // an option removed since
    expect(committedPrediction(-1, 4)).toBeNull();
    expect(committedPrediction(1.5, 4)).toBeNull();
    expect(committedPrediction('1', 4)).toBeNull();
  });
});
