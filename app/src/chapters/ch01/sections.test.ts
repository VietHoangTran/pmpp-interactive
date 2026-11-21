import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { numericOk, type ChoiceQ, type NumericQ } from '../../components/Ask';
import { fractionNeeded, speedup } from '../../sim/amdahl';
import { CHAPTERS, chaptersBefore } from '../../sim/bookmap';
import { design, race } from '../../sim/designs';
import { LONG_FEW, SHORT_MANY, questions10 } from './S10';
import { FIVE_HUNDRED, questions13 } from './S13';
import { questions17 } from './S17';
import { BUDGET } from './TwoDesignsScene';

// Pins the answers of the section questions that a review changed, so that what each
// question states stays tied to the model it is computed from.
const correct = (q: ChoiceQ) => q.options.findIndex((o) => o.correct);
const everyWrongSaysWhy = (q: ChoiceQ) => q.options.filter((o) => !o.correct).every((o) => !!o.why);
const src = (f: string) => fs.readFileSync(path.join(__dirname, f), 'utf8').replace(/\s+/g, ' ');

describe('start page: q-chainlen asks for the count of pieces, not the amount of arithmetic', () => {
  const fast = design(BUDGET, 2);
  const many = design(BUDGET, 0);
  it('job X has more operations than job Y but far fewer chains', () => {
    expect(LONG_FEW).toEqual({ chains: 8, length: 1000 });
    expect(SHORT_MANY).toEqual({ chains: 1024, length: 5 });
    expect(LONG_FEW.chains * LONG_FEW.length).toBeGreaterThan(SHORT_MANY.chains * SHORT_MANY.length);
  });
  it('X goes to the fast units, Y to the many units', () => {
    const x = race(fast, many, LONG_FEW.chains, LONG_FEW.length);
    const y = race(fast, many, SHORT_MANY.chains, SHORT_MANY.length);
    expect([x.winner, x.a.last, x.b.last, x.a.waves]).toEqual(['a', 500, 1000, 2]);
    expect([y.winner, y.a.last, y.b.last, y.a.waves, y.b.waves]).toEqual(['b', 320, 80, 256, 16]);
  });
  it('the question marks "X on A, Y on B" correct and explains every wrong option with the model numbers', () => {
    const q = questions10['ch01/start/q-chainlen'] as ChoiceQ;
    expect(correct(q)).toBe(2);
    expect(String(q.options[2].label)).toMatch(/^X on A, Y on B\b/);
    expect(everyWrongSaysWhy(q)).toBe(true);
    expect(String(q.options[0].why)).toContain('done at 320');
    expect(String(q.options[1].why)).toContain('done at 1,000');
  });
});

describe('1.3: q-fivehundred is the second half of the reason, not the explorer task again', () => {
  const q = questions13['ch01/1.3/q-fivehundred'] as NumericQ;
  it('replaces q-ceiling100, whose answer was the task\'s', () => {
    expect(questions13['ch01/1.3/q-ceiling100']).toBeUndefined();
    expect(Object.keys(questions13)[0]).toBe('ch01/1.3/q-fivehundred');
  });
  it('asks for about 500x from a part made 1,000x faster: 99.9%', () => {
    expect(FIVE_HUNDRED).toEqual({ s: 1000, target: 500 });
    expect(q.answer).toBe(99.9);
    expect(numericOk(q, 100 * fractionNeeded(500, 1000)!)).toBe(true);
    expect(speedup(q.answer / 100, FIVE_HUNDRED.s)).toBeCloseTo(500.25, 2);
  });
  it("rejects the task's 99% and the neighbouring stops", () => {
    for (const v of [99, 99.8, 99.95, 99.99]) expect(numericOk(q, v)).toBe(false);
  });
});

describe('1.7', () => {
  it('p-before20: seven chapters by the arrows; a line would give eighteen, and the page reacts to 18', () => {
    expect(chaptersBefore([20])).toHaveLength(7);
    expect(CHAPTERS.filter((c) => c.n > 1 && c.n < 20)).toHaveLength(18);
    const page = src('S17.tsx');
    expect(page).toContain('Eighteen would be right if the chart were a line: chapters 2 to 19.');
    expect(page).not.toContain('Nineteen would be right');
  });
  it('q-portable: the correct option is the tuning one, and every distractor needs a reason to rule out', () => {
    const q = questions17['ch01/1.7/q-portable'] as ChoiceQ;
    expect(q.options).toHaveLength(4);
    expect(correct(q)).toBe(2);
    expect(everyWrongSaysWhy(q)).toBe(true);
    expect(String(q.options[0].why)).toContain('the peaks are the same');
  });
  it('q-sortneeds: the paragraph before it no longer gives the answer away', () => {
    expect(src('S17.tsx')).not.toContain('five of them in Part II');
  });
});
