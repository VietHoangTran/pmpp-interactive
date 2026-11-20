import { describe, expect, it } from 'vitest';
import { displayOrder, type ChoiceQ } from './Ask';

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
