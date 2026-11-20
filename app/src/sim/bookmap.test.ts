import fs from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { PLAN } from '../chapters/registry';
import {
  CHAPTERS,
  COURSE_LINKS,
  COURSE_STAGES,
  EDGES,
  EXTRAS,
  FOURTH_TO_FIFTH,
  GOALS,
  NEW_IN_FIFTH,
  chapter,
  chaptersBefore,
  courseDepartures,
  courseOrder,
  coursePath,
  coursePosition,
  fifthFor,
  firstRenumbered,
  goal,
  isAcyclic,
  leadsTo,
  optionalPrerequisites,
  parents,
  prerequisites,
  readingOrder,
  successors,
  type Item,
} from './bookmap';

const repo = path.resolve(__dirname, '../../..');
const without1 = (xs: number[]) => xs.filter((n) => n !== 1);
const key = (x: { from: number; to: number }) => `${x.from}>${x.to}`;

describe('the arrows of Fig. P.1 as recorded in notes/ch01.md', () => {
  it('is the recorded edge list, no more and no less', () => {
    const recorded =
      '1>2 2>3 2>4 3>5 4>5 5>6 6>7 6>9 6>10 6>13 6>15 7>8 7>19 9>11 10>11 11>12 12>14 13>14 9>17 9>21 8>16 8>23 19>20 17>18 16>22 18>22 21>22 9>10';
    expect(EDGES.map(key).sort()).toEqual(recorded.split(' ').sort());
  });
  it('flags the dashed arrow and the two arrows read from the ambiguous junction', () => {
    expect(EDGES.filter((x) => x.dashed).map(key)).toEqual(['9>10']);
    expect(EDGES.filter((x) => x.uncertain).map(key).sort()).toEqual(['9>17', '9>21']);
  });
  it('has no cycle', () => {
    expect(isAcyclic()).toBe(true);
  });
  it('reaches every chapter 2 to 23 from chapter 1, and chapter 24 has no arrows', () => {
    expect(leadsTo([1])).toEqual(Array.from({ length: 22 }, (_, i) => i + 2));
    expect(EDGES.some((x) => x.from === 24 || x.to === 24)).toBe(false);
  });
  it('chapters 15, 20 and 23 lead nowhere further, and neither do 14 and 22', () => {
    for (const n of [15, 20, 23]) expect(successors(n, { dashed: true })).toEqual([]);
    const sinks = CHAPTERS.map((c) => c.n).filter((n) => n !== 24 && successors(n, { dashed: true }).length === 0);
    expect(sinks).toEqual([14, 15, 20, 22, 23]);
  });
  it('every arrow goes from a lower chapter number to a higher one', () => {
    expect(EDGES.every((x) => x.from < x.to)).toBe(true);
  });
});

describe('prerequisites', () => {
  it('chapter 15 needs all of Part I and nothing from Part II', () => {
    expect(without1(prerequisites([15]))).toEqual([2, 3, 4, 5, 6]);
    expect(chaptersBefore([15])).toEqual([2, 3, 4, 5, 6]);
    expect(prerequisites([15]).filter((n) => chapter(n).part === 'II')).toEqual([]);
    expect(parents(15)).toEqual([6]);
  });
  it('chapter 20 needs seven chapters: Part I, convolution and CNNs', () => {
    expect(without1(prerequisites([20]))).toEqual([2, 3, 4, 5, 6, 7, 19]);
    expect(chaptersBefore([20]).length).toBe(7);
  });
  it('chapter 14 needs ten, five of them in Part II', () => {
    const p = without1(prerequisites([14]));
    expect(p.length).toBe(10);
    expect(p.filter((n) => chapter(n).part === 'II')).toEqual([9, 10, 11, 12, 13]);
  });
  it('chapter 22 needs twelve', () => {
    expect(without1(prerequisites([22]))).toEqual([2, 3, 4, 5, 6, 7, 8, 9, 16, 17, 18, 21]);
  });
  it('chapter 7 is not a prerequisite of chapter 15', () => {
    expect(prerequisites([15])).not.toContain(7);
    for (const n of [4, 5, 6]) expect(prerequisites([15])).toContain(n);
  });
  it('the dashed arrow is followed only on request', () => {
    expect(without1(prerequisites([10]))).toEqual([2, 3, 4, 5, 6]);
    expect(without1(prerequisites([10], { dashed: true }))).toEqual([2, 3, 4, 5, 6, 9]);
    expect(optionalPrerequisites([10])).toEqual([9]);
    expect(optionalPrerequisites([14])).toEqual([]); // 9 is already there through 11
    expect(optionalPrerequisites([15])).toEqual([]);
  });
});

describe('goals (ours)', () => {
  it('are the five the lesson offers', () => {
    expect(GOALS.map((g) => g.id)).toEqual(['matmul', 'attention', 'sampling', 'quantised', 'backends']);
    expect(goal('matmul').chapters).toEqual([15]);
    expect(goal('attention').chapters).toEqual([20]);
    expect(goal('sampling').chapters).toEqual([10, 11, 12, 14]);
    expect(goal('sampling').extras).toEqual(['x1']);
    expect(goal('quantised').chapters).toEqual([15]);
    expect(goal('quantised').extras).toEqual(['appA', 'x2']);
    expect(goal('backends').extras).toEqual(['x3']);
  });
  it('sampling is the longest road: eleven chapters with its targets, chapter 1 not counted', () => {
    const g = goal('sampling');
    const all = without1(readingOrder(g.chapters));
    expect(all).toEqual([2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]);
    const size = (id: 'matmul' | 'attention' | 'sampling' | 'quantised') => without1(readingOrder(goal(id).chapters)).length;
    expect([size('matmul'), size('attention'), size('sampling'), size('quantised')]).toEqual([6, 8, 11, 6]);
  });
});

describe('readingOrder', () => {
  it('never lists a chapter before one of its prerequisites', () => {
    const all = CHAPTERS.map((c) => c.n);
    for (const t of [...all.map((n) => [n]), ...GOALS.map((g) => g.chapters), all]) {
      const order = readingOrder(t, { dashed: true });
      for (const x of EDGES) {
        if (order.includes(x.to)) expect(order.indexOf(x.from)).toBeLessThan(order.indexOf(x.to));
      }
    }
  });
  it('breaks ties by chapter number', () => {
    expect(readingOrder([20])).toEqual([1, 2, 3, 4, 5, 6, 7, 19, 20]);
    expect(readingOrder([14])).toEqual([1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14]);
    expect(readingOrder([15])).toEqual([1, 2, 3, 4, 5, 6, 15]);
  });
});

describe('chapters and tiers agree with the registry', () => {
  it('24 chapters with the registry titles, pages and tiers', () => {
    expect(CHAPTERS.length).toBe(24);
    for (const c of CHAPTERS) {
      const row = PLAN.find((r) => r.id === c.id)!;
      expect(row, c.id).toBeTruthy();
      expect(row.num).toBe(String(c.n));
      expect(c.title).toBe(row.title);
      expect(c.pages).toBe(row.pages);
      expect(c.tier).toBe(row.tier);
    }
  });
  it('the extras are registry rows too', () => {
    for (const x of EXTRAS) {
      const row = PLAN.find((r) => r.id === x.id)!;
      expect(row.num).toBe(x.num);
      expect(row.title).toBe(x.title);
      expect(row.tier).toBe(x.tier);
      expect(!!row.ours).toBe(x.pages === null);
      if (x.pages) expect(row.pages).toBe(x.pages);
    }
  });
  it('parts are 2 to 6, 7 to 15 and 16 to 23', () => {
    const of = (p: string | null) => CHAPTERS.filter((c) => c.part === p).map((c) => c.n);
    expect(of('I')).toEqual([2, 3, 4, 5, 6]);
    expect(of('II')).toEqual([7, 8, 9, 10, 11, 12, 13, 14, 15]);
    expect(of('III')).toEqual([16, 17, 18, 19, 20, 21, 22, 23]);
    expect(of(null)).toEqual([1, 24]);
  });
});

describe("this course's order (PLAN.md section 9)", () => {
  // "2, 3, 4, 6 (foundations) → 15 (matmul) → ... → condensed chapters 1, 8, ..., 24."
  const plan = fs.readFileSync(path.join(repo, 'PLAN.md'), 'utf8');
  const sentence = /2, 3, 4, 6 \(foundations\)[\s\S]*?23, 24\./.exec(plan)![0];
  const token = (t: string): Item => (/^\d+$/.test(t) ? Number(t) : t === 'A' ? 'appA' : t === 'C' ? 'appC' : (t.toLowerCase() as Item));
  const planStages: Item[][] = sentence
    .split('→')
    .map((part) => part.replace(/\([^)]*\)/g, '').replace('condensed chapters', '').replace(/\.\s*$/, ''))
    .map((part) => part.split(',').map((t) => t.trim()).filter(Boolean).map(token));

  it('is PLAN.md section 9, stage for stage, with chapter 1 moved to the front and chapter 5 added to the foundations', () => {
    const ours = COURSE_STAGES.map((s) => s.items.filter((i) => !(s.placedHere ?? []).includes(i as number))).filter((s) => s.length);
    const theirs = planStages.map((s) => s.filter((i) => i !== 1));
    expect(ours).toEqual(theirs);
    expect(COURSE_STAGES.flatMap((s) => s.placedHere ?? [])).toEqual([1, 5]);
  });
  it('lists every chapter and every extra exactly once', () => {
    const order = courseOrder();
    expect(order.length).toBe(29);
    expect(new Set(order).size).toBe(29);
    for (const c of CHAPTERS) expect(order).toContain(c.n);
    for (const x of EXTRAS) expect(order).toContain(x.id);
  });
  it('flattens to this order', () => {
    expect(courseOrder()).toEqual([
      1, 2, 3, 4, 5, 6, 15, 9, 10, 11, 12, 14, 'x1', 20, 22, 'appA', 'x2', 7, 19, 'appC', 'x3', 8, 13, 16, 17, 18, 21, 23, 24,
    ]);
    expect(coursePosition(15)).toBe(6);
  });
  it('the last stage holds condensed chapters only', () => {
    for (const i of COURSE_STAGES[COURSE_STAGES.length - 1].items) expect(chapter(i as number).tier).toBe('Condensed');
  });
  it('runs ahead of the arrows in three places: merge after sort, CNNs after attention, three applications after chapter 22', () => {
    expect(courseDepartures().map(key)).toEqual(['13>14', '19>20', '16>22', '18>22', '21>22']);
  });
  it('respects every prerequisite it adds itself', () => {
    for (const l of COURSE_LINKS) expect(coursePosition(l.from)).toBeLessThan(coursePosition(l.to));
  });
});

describe('coursePath', () => {
  it('matmul: the same six as the arrows', () => {
    expect(coursePath([15])).toEqual({ order: [1, 2, 3, 4, 5, 6, 15], added: [], later: [] });
  });
  it('attention: reduction is added, convolution and CNNs come later', () => {
    expect(coursePath([20])).toEqual({ order: [1, 2, 3, 4, 5, 6, 10, 20], added: [10], later: [7, 19] });
  });
  it('sampling: merge comes later, module X1 at the end', () => {
    const g = goal('sampling');
    expect(coursePath(g.chapters, g.extras)).toEqual({ order: [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 'x1'], added: [], later: [13] });
  });
  it('quantised kernels: chapter 15, Appendix A, then X2', () => {
    const g = goal('quantised');
    expect(coursePath(g.chapters, g.extras)).toEqual({ order: [1, 2, 3, 4, 5, 6, 15, 'appA', 'x2'], added: [], later: [] });
  });
  it('reading real backends: placed by position, after everything the course builds before it', () => {
    const p = coursePath([], ['x3']);
    expect(p.order[p.order.length - 1]).toBe('x3');
    expect(p.order.length).toBe(21);
    for (const n of [15, 20, 10, 11, 12, 14]) expect(p.order).toContain(n);
    expect(p.later).toEqual([]);
  });
  it('chapter 22: six of its twelve prerequisites are read after it', () => {
    expect(coursePath([22])).toEqual({ order: [1, 2, 3, 4, 5, 6, 9, 22], added: [], later: [7, 8, 16, 17, 18, 21] });
  });
  it('never orders a kept chapter before a kept prerequisite', () => {
    for (const c of CHAPTERS) {
      const { order } = coursePath([c.n]);
      for (const x of EDGES)
        if (order.includes(x.from) && order.includes(x.to)) expect(order.indexOf(x.from)).toBeLessThan(order.indexOf(x.to));
    }
  });
});

describe('4th edition to 5th edition (notes/today/ch01-facts.md, A15)', () => {
  it('has an entry for every 4th-edition chapter 1 to 23', () => {
    expect(FOURTH_TO_FIFTH.map((c) => c.n)).toEqual(Array.from({ length: 23 }, (_, i) => i + 1));
    expect(() => fifthFor(24)).toThrow();
  });
  it('numbers 1 to 11 are unchanged and 12 is the first that moved', () => {
    for (let n = 1; n <= 11; n++) expect(fifthFor(n)).toBe(n);
    expect(firstRenumbered()).toBe(12);
    for (let n = 12; n <= 23; n++) expect(fifthFor(n)).not.toBe(n);
  });
  it('maps the moved chapters as the facts file pairs them', () => {
    const got = Object.fromEntries(FOURTH_TO_FIFTH.map((c) => [c.n, fifthFor(c.n)]));
    expect(got).toMatchObject({ 12: 13, 13: 14, 14: 17, 15: 18, 16: 19, 17: null, 18: 21, 19: 22, 20: 23, 21: null, 22: null, 23: 24 });
  });
  it('three chapters are gone: MRI (folded into 21), dynamic parallelism, advanced practices (history now Appendix C)', () => {
    const gone = FOURTH_TO_FIFTH.filter((c) => c.fate.kind === 'removed');
    expect(gone.map((c) => c.n)).toEqual([17, 21, 22]);
    expect(gone[0].fate).toMatchObject({ foldedInto: 21 });
    expect(gone[2].fate).toMatchObject({ appendix: 'C' });
  });
  it('no two 4th-edition chapters land on the same 5th-edition chapter', () => {
    const to = FOURTH_TO_FIFTH.map((c) => fifthFor(c.n)).filter((n): n is number => n !== null);
    expect(new Set(to).size).toBe(to.length);
    expect(to.length).toBe(20);
  });
  it('the four 5th-edition chapters nothing maps to are the new ones', () => {
    const hit = new Set(FOURTH_TO_FIFTH.map((c) => fifthFor(c.n)));
    expect(CHAPTERS.map((c) => c.n).filter((n) => !hit.has(n))).toEqual(NEW_IN_FIFTH);
    expect(NEW_IN_FIFTH).toEqual([12, 15, 16, 20]);
  });
  it('the facts file records the 4th-edition titles used here', () => {
    const facts = fs.readFileSync(path.join(repo, 'notes/today/ch01-facts.md'), 'utf8');
    for (const c of FOURTH_TO_FIFTH) expect(facts, c.title).toMatch(new RegExp(`${c.n}\\.? ${c.title.replace(/[()]/g, '\\$&')}`));
  });
  it("the lesson's question: a 4th-edition chapter 16 is chapter 19 now, and 16 now is wavefront parallelism", () => {
    expect(fifthFor(16)).toBe(19);
    expect(chapter(19).title).toBe('Convolutional neural networks');
    expect(chapter(16).title).toContain('wavefront');
  });
});
