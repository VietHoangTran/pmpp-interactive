// The model behind chapter 24's steps map: which of the book's three steps each
// chapter belongs to (section 24.1, pp. 577–578), which chapters feed each of the
// reader's goals (PLAN.md sections 3 and 5), and the reader's own record per chapter.
//
// What it simplifies: a chapter either feeds a goal or does not. PLAN.md gives one
// reason per chapter and that is all the map shows; it says nothing about how much
// of a chapter a goal needs. Labs are counted from the run reports by the chapter
// prefix of their id, so a chapter whose labs have never been run shows none.
import { labStatus, realBackends, type Reports } from '../lib/reports';
import type { PlanRow } from '../chapters/types';

export type Step = 1 | 2 | 3 | 'outside';

export const STEPS: { step: Step; name: string; chapters: string; what: string }[] = [
  { step: 1, name: 'Step one', chapters: 'chapters 2 to 6', what: 'Kernels, thread-to-data mapping, the execution and memory model, the optimisation checklist.' },
  { step: 2, name: 'Step two', chapters: 'chapters 7 to 15', what: 'Parallel patterns, each one a vehicle for the step-one optimisations.' },
  { step: 3, name: 'Step three', chapters: 'chapters 16 to 23', what: 'Advanced patterns and applications: alternative decompositions and their data structures.' },
  { step: 'outside', name: "Outside the book's steps", chapters: 'chapters 1 and 24, the appendices, our modules', what: 'Framing, reference material, and the three modules this course adds.' },
];

/** The step the book assigns a chapter to on pp. 577–578; 'outside' for everything it does not place. */
export function stepOf(chapterId: string): Step {
  const m = /^ch(\d\d)$/.exec(chapterId);
  if (!m) return 'outside';
  const n = Number(m[1]);
  if (n >= 2 && n <= 6) return 1;
  if (n >= 7 && n <= 15) return 2;
  if (n >= 16 && n <= 23) return 3;
  return 'outside';
}

export type GoalId = 'matmul' | 'attention' | 'sampling' | 'quantised' | 'backends' | 'reading';
export interface Goal {
  id: GoalId;
  label: string;
  /** Chapter ids that feed it, from PLAN.md sections 3 and 5. */
  chapters: string[];
  /** True where every lab contributes, whatever its chapter. */
  everyLab?: boolean;
}
export const GOALS: Goal[] = [
  { id: 'matmul', label: 'matmul', chapters: ['ch03', 'ch04', 'ch05', 'ch06', 'ch15', 'ch19'] },
  { id: 'attention', label: 'attention', chapters: ['ch10', 'ch20', 'ch22'] },
  { id: 'sampling', label: 'sampling', chapters: ['ch09', 'ch10', 'ch11', 'ch12', 'ch14', 'x1'] },
  { id: 'quantised', label: 'quantised ops', chapters: ['appA', 'ch15', 'x2'] },
  { id: 'backends', label: 'other backends', chapters: ['appC', 'x3'], everyLab: true },
  { id: 'reading', label: 'reading real code', chapters: ['ch19', 'x3'] },
];
export const goal = (id: GoalId) => GOALS.find((g) => g.id === id)!;

/** The plan's rows that have a page in the app, grouped by step, in plan order. */
export function lanes(plan: PlanRow[]): { step: Step; rows: PlanRow[] }[] {
  return STEPS.map((s) => ({ step: s.step, rows: plan.filter((r) => r.id && stepOf(r.id) === s.step) }));
}

/** The distinct steps a goal draws on, in order. */
export function stepsOf(id: GoalId): Step[] {
  const set = new Set(goal(id).chapters.map(stepOf));
  return STEPS.map((s) => s.step).filter((s) => set.has(s));
}

export interface ChapterRecord {
  /** Sections of this chapter the reader has marked done. */
  sectionsDone: number;
  /** Labs of this chapter that appear in any run report. */
  labs: number;
  /** Of those, labs whose edited work/ copy passes on at least one real backend. */
  labsPassing: number;
}

/**
 * The reader's record per chapter. Labs are matched by the "chNN/" prefix of the
 * lab id, never by name. The emulator does not count as a backend here, and an
 * untouched starter is "not started", not a pass or a failure.
 */
export function rollup(
  plan: PlanRow[],
  sections: Record<string, { done?: boolean }>,
  reports: Reports,
): Record<string, ChapterRecord> {
  const out: Record<string, ChapterRecord> = {};
  for (const r of plan) {
    if (!r.id) continue;
    const prefix = r.id + '/';
    const sectionsDone = Object.entries(sections).filter(([k, v]) => k.startsWith(prefix) && v.done).length;
    const seen = new Set<string>();
    const passing = new Set<string>();
    for (const run of Object.values(reports))
      for (const x of run.results) if (x.lab.startsWith(prefix)) seen.add(x.lab);
    for (const backend of realBackends(reports))
      for (const lab of seen) if (labStatus(reports[backend], lab, 'work') === 'pass') passing.add(lab);
    out[r.id] = { sectionsDone, labs: seen.size, labsPassing: passing.size };
  }
  return out;
}
