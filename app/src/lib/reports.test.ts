import { describe, expect, it } from 'vitest';
import { labStatus, labSummary, pickBenchRow, type BackendRun, type RunResult } from './reports';
import { narrowQuery, NARROW_QUERY } from './narrow';

const result = (over: Partial<RunResult>): RunResult => ({
  lab: 'ch09/lab1_x', variant: 'solution', backend: 'race', tile: 16, built: true, test_rc: 0,
  tests: [{ case: 'a', pass: true }, { case: 'b', pass: true }], bench: [], ...over,
});
const run = (results: RunResult[], work_state?: BackendRun['work_state']): BackendRun => ({
  generated: '2026-11-21T00:00:00', device: { name: 'test' }, results, work_state,
});

describe('pickBenchRow', () => {
  const rows = [
    { name: 'gemv', format: 'q8_0', n: 4096, seconds: 1 },
    { name: 'gemv', format: 'q4_0', n: 4096, seconds: 2 },
    { name: 'gemv', format: 'q4_0', n: '8192', seconds: 3 },
  ];
  it('matches numbers against numeric fields, as before', () => {
    expect(pickBenchRow(rows, 'gemv', { n: 4096 })?.seconds).toBe(1);
    expect(pickBenchRow(rows, 'gemv', { n: 8192 })).toBeUndefined(); // "8192" is text, not a number
  });
  it('matches strings against text fields', () => {
    expect(pickBenchRow(rows, 'gemv', { format: 'q4_0' })?.seconds).toBe(2);
    expect(pickBenchRow(rows, 'gemv', { format: 'q4_0', n: '8192' })?.seconds).toBe(3);
    expect(pickBenchRow(rows, 'gemv', { format: 'f16' })).toBeUndefined();
  });
  it('treats a missing field as no match and no rows as nothing', () => {
    expect(pickBenchRow(rows, 'gemv', { dtype: 'x' })).toBeUndefined();
    expect(pickBenchRow(undefined, 'gemv')).toBeUndefined();
  });
});

describe('labSummary', () => {
  it('counts tests, ThreadSanitizer reports and an aborted run', () => {
    const r = run([result({ test_rc: 66, tests: [{ case: 'a', pass: true }, { case: 'b', pass: false }], tsan: [{}, {}] })]);
    const s = labSummary(r, 'ch09/lab1_x', 'solution');
    expect(s).toEqual({ status: 'fail', recorded: true, buildFailed: false, aborted: true, passed: 1, total: 2, tsanReports: 2 });
  });
  it('agrees with labStatus, including an untouched starter as none', () => {
    const r = run([result({ variant: 'work', test_rc: 1, tests: [{ case: 'a', pass: false }] })], { 'ch09/lab1_x': 'untouched' });
    expect(labSummary(r, 'ch09/lab1_x', 'work').status).toBe('none');
    expect(labSummary(r, 'ch09/lab1_x', 'work').recorded).toBe(true);
    expect(labStatus(r, 'ch09/lab1_x', 'work')).toBe('none');
    expect(labSummary(undefined, 'ch09/lab1_x', 'solution')).toMatchObject({ status: 'none', recorded: false, total: 0 });
  });
  it('notices a target that did not build', () => {
    expect(labSummary(run([result({ built: false, tests: [] })]), 'ch09/lab1_x', 'solution').buildFailed).toBe(true);
  });
});

describe('narrowQuery', () => {
  it('turns a width into a max-width query and keeps a query string', () => {
    expect(narrowQuery(600)).toBe('(max-width: 600px)');
    expect(narrowQuery('(max-width: 40rem)')).toBe('(max-width: 40rem)');
    expect(narrowQuery()).toBe(NARROW_QUERY);
  });
});
