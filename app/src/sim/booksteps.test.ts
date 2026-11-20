import { describe, expect, it } from 'vitest';
import { PLAN } from '../chapters/registry';
import type { Reports, RunResult } from '../lib/reports';
import { GOALS, STEPS, goal, gpuBackends, lanes, rollup, stepOf, stepsOf } from './booksteps';

describe('the three steps of pp. 577–578', () => {
  it('chapters 2–6, 7–15 and 16–23', () => {
    expect(stepOf('ch02')).toBe(1);
    expect(stepOf('ch06')).toBe(1);
    expect(stepOf('ch07')).toBe(2);
    expect(stepOf('ch15')).toBe(2);
    expect(stepOf('ch16')).toBe(3);
    expect(stepOf('ch22')).toBe(3);
    expect(stepOf('ch23')).toBe(3);
  });
  it('everything else is outside them', () => {
    for (const id of ['ch01', 'ch24', 'appA', 'appC', 'x1', 'x2', 'x3']) expect(stepOf(id)).toBe('outside');
  });
  it('lanes hold 5, 9 and 8 chapters, and 7 pages sit outside', () => {
    const l = lanes(PLAN);
    expect(l.map((x) => x.rows.length)).toEqual([5, 9, 8, 7]);
    expect(l.map((x) => x.step)).toEqual(STEPS.map((s) => s.step));
    // Every plan row with a page lands in exactly one lane.
    expect(l.flatMap((x) => x.rows)).toHaveLength(PLAN.filter((r) => r.id).length);
  });
});

describe('goals', () => {
  it('every chapter id in GOALS exists in the plan', () => {
    const ids = new Set(PLAN.map((r) => r.id));
    for (const g of GOALS) for (const c of g.chapters) expect(ids.has(c), `${g.id}: ${c}`).toBe(true);
  });
  it('sampling is chapters 9, 10, 11, 12, 14 and module X1: all step two plus a module the book lacks', () => {
    expect(goal('sampling').chapters).toEqual(['ch09', 'ch10', 'ch11', 'ch12', 'ch14', 'x1']);
    expect(stepsOf('sampling')).toEqual([2, 'outside']);
  });
  it('matmul draws on all three steps', () => {
    expect(stepsOf('matmul')).toEqual([1, 2, 3]);
  });
  it('every goal draws on more than one step', () => {
    for (const g of GOALS) expect(stepsOf(g.id).length + (g.everyLab ? 1 : 0), g.id).toBeGreaterThan(1);
  });
  it('only "other backends" counts every lab', () => {
    expect(GOALS.filter((g) => g.everyLab).map((g) => g.id)).toEqual(['backends']);
  });
});

describe('rollup', () => {
  const result = (lab: string, variant: string, backend: string, pass: boolean): RunResult => ({
    lab, variant, backend, tile: 0, built: true, test_rc: pass ? 0 : 1, tests: [{ case: 'a', pass }], bench: [],
  });
  const run = (backend: string, results: RunResult[], work_state: Record<string, 'untouched' | 'edited'>) => ({
    generated: '2026-11-20T00:00:00', device: { name: backend }, results, work_state,
  });

  it('counts sections marked done by chapter prefix', () => {
    const r = rollup(PLAN, { 'ch05/5.1': { done: true }, 'ch05/5.2': { done: false }, 'ch05/labs': { done: true }, 'ch15/start': { done: true } }, {});
    expect(r.ch05).toEqual({ sectionsDone: 2, labs: 0, labsPassing: 0 });
    expect(r.ch15.sectionsDone).toBe(1);
    expect(r.ch24.sectionsDone).toBe(0);
  });
  it('counts nothing for an untouched starter', () => {
    const reports: Reports = {
      metal: run('metal', [result('ch05/lab1_roofline', 'solution', 'metal', true), result('ch05/lab1_roofline', 'work', 'metal', false)], { 'ch05/lab1_roofline': 'untouched' }),
    };
    expect(rollup(PLAN, {}, reports).ch05).toEqual({ sectionsDone: 0, labs: 1, labsPassing: 0 });
    // Even a passing run does not count while work/ is still the starter.
    reports.metal.results[1] = result('ch05/lab1_roofline', 'work', 'metal', true);
    expect(rollup(PLAN, {}, reports).ch05.labsPassing).toBe(0);
  });
  it('counts an edited work copy that passes on a real backend, once per lab', () => {
    const reports: Reports = {
      metal: run('metal', [result('ch05/lab1_roofline', 'work', 'metal', true), result('ch05/lab2_tiled', 'work', 'metal', false), result('x1/lab1_greedy', 'work', 'metal', true)], {
        'ch05/lab1_roofline': 'edited', 'ch05/lab2_tiled': 'edited', 'x1/lab1_greedy': 'edited',
      }),
      cuda: run('cuda', [result('ch05/lab1_roofline', 'work', 'cuda', true)], { 'ch05/lab1_roofline': 'edited' }),
    };
    const r = rollup(PLAN, {}, reports);
    expect(r.ch05).toEqual({ sectionsDone: 0, labs: 2, labsPassing: 1 });
    expect(r.x1).toEqual({ sectionsDone: 0, labs: 1, labsPassing: 1 });
    expect(r.ch15.labs).toBe(0);
  });
  it('ignores the emu backend', () => {
    const reports: Reports = {
      emu: run('emu', [result('ch05/lab1_roofline', 'work', 'emu', true)], { 'ch05/lab1_roofline': 'edited' }),
    };
    expect(rollup(PLAN, {}, reports).ch05).toEqual({ sectionsDone: 0, labs: 1, labsPassing: 0 });
  });
  it('ignores the race backend, which is the emulator under ThreadSanitizer', () => {
    const reports: Reports = {
      race: run('race', [result('ch05/lab4_racehunt', 'work', 'race', true)], { 'ch05/lab4_racehunt': 'edited' }),
      metal: run('metal', [], {}),
    };
    expect(gpuBackends(reports)).toEqual(['metal']);
    expect(rollup(PLAN, {}, reports).ch05).toEqual({ sectionsDone: 0, labs: 1, labsPassing: 0 });
  });
  it('does not confuse chapter prefixes', () => {
    const r = rollup(PLAN, { 'x1/start': { done: true }, 'x2/start': { done: true }, 'appA/start': { done: true } }, {});
    expect([r.x1.sectionsDone, r.x2.sectionsDone, r.x3.sectionsDone, r.appA.sectionsDone, r.appC.sectionsDone]).toEqual([1, 1, 0, 1, 0]);
  });
});
