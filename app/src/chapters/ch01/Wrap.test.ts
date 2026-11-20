import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { numericOk, type ChoiceQ, type NumericQ } from '../../components/Ask';
import { BOOK_CASES, ceiling, remaining, speedup, untouchedShare } from '../../sim/amdahl';
import { chaptersBefore, goal, leadsTo, prerequisites } from '../../sim/bookmap';
import { design, finish } from '../../sim/designs';
import { BUDGET } from './TwoDesignsScene';
import { JOB, PROFILE, questionsWrap } from './Wrap';

// The summary page shows no kernel (chapter 1 has no code), so there is no snippet
// to tie it to. This test pins the numbers its questions state to the models the
// sections run on, and a few of its statements to the notes they come from.
const repo = path.resolve(__dirname, '../../../..');
const notes = fs.readFileSync(path.join(repo, 'notes/ch01.md'), 'utf8');
const page = fs.readFileSync(path.join(__dirname, 'Wrap.tsx'), 'utf8');
const correct = (q: ChoiceQ) => q.options.findIndex((o) => o.correct);

describe('summary page of chapter 1', () => {
  it('has five questions with ids under ch01/wrap, each with one way to be right', () => {
    const ids = Object.keys(questionsWrap);
    expect(ids).toEqual([
      'ch01/wrap/q-reprofile',
      'ch01/wrap/q-both',
      'ch01/wrap/q-morethreads',
      'ch01/wrap/q-skip9',
      'ch01/wrap/q-sharedmemory',
    ]);
    for (const id of ids) {
      const q = questionsWrap[id];
      expect(q.id).toBe(id);
      expect(q.source).toBeTruthy();
      if (q.kind === 'choice') {
        expect(q.options.filter((o) => o.correct)).toHaveLength(1);
        // every wrong option says why it is wrong
        expect(q.options.filter((o) => !o.correct).every((o) => !!o.why)).toBe(true);
      }
    }
  });

  it('q-reprofile: 90% in kernels made 9 times faster leaves the host part with half of the run', () => {
    expect(PROFILE).toEqual({ p: 0.9, s: 9 });
    const left = remaining(PROFILE.p, PROFILE.s);
    expect(left.untouched).toBeCloseTo(0.1, 12);
    expect(left.accelerated).toBeCloseTo(0.1, 12);
    expect(left.total).toBeCloseTo(0.2, 12);
    expect(speedup(PROFILE.p, PROFILE.s)).toBeCloseTo(5, 10);
    expect(ceiling(PROFILE.p)).toBeCloseTo(10, 10);
    // halfway to the ceiling, and the kernels can at most double the speed again
    expect(ceiling(PROFILE.p) / speedup(PROFILE.p, PROFILE.s)).toBeCloseTo(2, 10);
    const q = questionsWrap['ch01/wrap/q-reprofile'] as NumericQ;
    expect(q.answer).toBe(50);
    expect(numericOk(q, 100 * untouchedShare(PROFILE.p, PROFILE.s))).toBe(true);
    // the tempting wrong answers are not accepted: the old share, and the old share of the new time
    expect(numericOk(q, 10)).toBe(false);
    expect(numericOk(q, 90)).toBe(false);
  });

  it('q-both: the chain on the fast units and the crowd on the many units finishes first', () => {
    expect(BUDGET).toBe(64);
    const fast = design(BUDGET, 2);
    const many = design(BUDGET, 0);
    expect([fast.units, fast.speed, many.units, many.speed]).toEqual([4, 4, 64, 1]);
    expect(JOB).toEqual({ chain: 1200, crowd: 1024, length: 100 });
    const chain = { fast: finish(fast, 1, JOB.chain).last, many: finish(many, 1, JOB.chain).last };
    const crowd = { fast: finish(fast, JOB.crowd, JOB.length), many: finish(many, JOB.crowd, JOB.length) };
    expect(chain).toEqual({ fast: 300, many: 1200 });
    expect([crowd.fast.waves, crowd.fast.last, crowd.many.waves, crowd.many.last]).toEqual([256, 6400, 16, 1600]);
    const times = {
      fastAlone: chain.fast + crowd.fast.last,
      manyAlone: chain.many + crowd.many.last,
      split: chain.fast + crowd.many.last,
      wrongWay: chain.many + crowd.fast.last,
    };
    expect(times).toEqual({ fastAlone: 6700, manyAlone: 2800, split: 1900, wrongWay: 7600 });
    expect(Math.min(...Object.values(times))).toBe(times.split);
    const q = questionsWrap['ch01/wrap/q-both'] as ChoiceQ;
    expect(correct(q)).toBe(2);
    // the options are in the order the times above are listed
    expect(String(q.options[0].why)).toContain('6,700');
    expect(String(q.options[1].why)).toContain('2,800');
    expect(String(q.options[3].why)).toContain('7,600');
  });

  it('q-skip9: chapter 9 is in front of three of the four sampling chapters and of neither matmul nor attention', () => {
    expect(leadsTo([9])).toEqual([11, 12, 14, 17, 18, 21, 22]);
    const sampling = goal('sampling').chapters;
    expect(sampling).toEqual([10, 11, 12, 14]);
    expect(sampling.filter((c) => prerequisites([c]).includes(9))).toEqual([11, 12, 14]);
    // chapter 10 is reached from 9 only when the dashed arrow is followed
    expect(prerequisites([10]).includes(9)).toBe(false);
    expect(prerequisites([10], { dashed: true }).includes(9)).toBe(true);
    expect(prerequisites([15]).includes(9)).toBe(false);
    expect(prerequisites([20]).includes(9)).toBe(false);
    expect(correct(questionsWrap['ch01/wrap/q-skip9'] as ChoiceQ)).toBe(2);
  });

  it('the reading-path figures under "What to carry forward" are the model\'s', () => {
    expect(chaptersBefore([15])).toEqual([2, 3, 4, 5, 6]);
    expect(chaptersBefore([20])).toHaveLength(7);
    // sampling is the longest of the three roads
    const road = (targets: number[]) => new Set([...targets, ...chaptersBefore(targets)]).size;
    expect(road(goal('sampling').chapters)).toBeGreaterThan(road([20]));
    expect(road([20])).toBeGreaterThan(road([15]));
  });

  it('the two choice questions without a model have the intended answers', () => {
    expect(correct(questionsWrap['ch01/wrap/q-morethreads'] as ChoiceQ)).toBe(1);
    expect(correct(questionsWrap['ch01/wrap/q-sharedmemory'] as ChoiceQ)).toBe(1);
  });

  it("the table of worked numbers quotes the book's three Amdahl cases as the model records them", () => {
    expect(BOOK_CASES.map((c) => c.book)).toEqual(['about 1.42', 'no more than 1.43', 'about 50']);
    expect(speedup(0.3, 100)).toBeCloseTo(1.4225, 4);
    expect(ceiling(0.3)).toBeCloseTo(1.4286, 4);
    expect(speedup(0.99, 100)).toBeCloseTo(50.25, 2);
    for (const i of [0, 1, 2]) expect(page).toContain(`{BOOK_CASES[${i}].book}×`);
  });

  it('what the page says about the book is in the notes', () => {
    // no exercises; the six-step recipe is on p. 15; the two readings of radix sort; the four-parts slip
    expect(notes).toContain('Chapter 1 has no exercise section');
    expect(notes).toMatch(/Six-step thought process \(p\. 15\)/);
    expect(notes).toContain('relies on the scan pattern covered in Chapter 14');
    expect(notes).toContain('"into four major parts" (p. xxii)');
    expect(notes).toContain('"consists of 14 chapters"');
    // every typo the page lists is one the notes recorded
    for (const typo of ['coverts', 'filter patter', 'divide-and-concur', 'leverage techniques scatter-to-gather transformations'])
      expect(notes).toContain(typo);
    for (const typo of ['"coverts"', '"filter patter"', '"divide-and-concur"', '"leverage\n          techniques scatter-to-gather transformations"'])
      expect(page.replace(/\s+/g, ' ')).toContain(typo.replace(/\s+/g, ' '));
  });

  it('the page gives no device figure of its own', () => {
    // The only hardware numbers allowed in a lesson are the book's H100 trio, labelled as the book's.
    const text = page.replace(/\{[^}]*\}/g, ' ');
    expect(text).not.toMatch(/\b\d+(\.\d+)?\s?(GB|TB|GiB|MB)(\/s)?\b/);
    expect(text).not.toMatch(/\b\d+\s+cores\b/);
    expect(text).toContain("34, 67 and 1,979 TFLOPS on the book's example GPU");
  });
});
