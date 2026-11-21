import { describe, expect, it } from 'vitest';
import { benchRow, defineLabs, raceSolutionStatus, solutionStatus, solutionTabStatus, type BenchSet } from './LabKit';
import type { BackendRun, Reports, RunResult } from '../lib/reports';

const [lab] = defineLabs('ch09', [{ num: '9.1', dir: 'lab1_x', name: 'x' }]);
const res = (backend: string, pass: boolean): RunResult => ({
  lab: lab.id, variant: 'solution', backend, tile: 16, built: true, test_rc: pass ? 0 : 1,
  tests: [{ case: 'a', pass }], bench: [],
});
const run = (backend: string, pass: boolean): BackendRun => ({ generated: '2026-11-21T00:00:00', device: {}, results: [res(backend, pass)] });
const reports = (spec: Record<string, boolean>): Reports => Object.fromEntries(Object.entries(spec).map(([b, p]) => [b, run(b, p)]));

describe('solutionTabStatus', () => {
  it('keeps the usual wording when the emulator passed and CUDA and HIP have no run', () => {
    expect(solutionTabStatus(reports({ emu: true, metal: true }), lab, 'kernel.cu')).toBe('emulator-checked; unverified on CUDA and HIP');
    expect(solutionTabStatus(reports({ emu: true, metal: true }), lab, 'kernel.metal')).toBe('verified on Metal');
  });
  it('says a run failed when it failed, in the table\'s words', () => {
    expect(solutionTabStatus(reports({ emu: false }), lab, 'kernel.cu')).toBe('unverified on the emulator: failed on this machine; unverified on CUDA and HIP');
    expect(solutionTabStatus(reports({ emu: true, cuda: false }), lab, 'kernel.cu')).toBe('emulator-checked; unverified on CUDA: failed on this machine; unverified on HIP');
    expect(solutionTabStatus(reports({ metal: false }), lab, 'launch.mm')).toBe('unverified: failed on this machine');
    expect(solutionStatus(reports({ metal: false }), 'metal', lab).text).toBe('unverified: failed on this machine');
  });
  it('says there was no run only when nothing ran', () => {
    expect(solutionTabStatus({}, lab, 'kernel.cu')).toBe('unverified: no run on this machine');
    expect(solutionTabStatus({}, lab, 'kernel.metal')).toBe('unverified: no run on this machine');
    expect(solutionStatus({}, 'cuda', lab).text).toBe('no run on this machine');
  });
  it('lists every passing backend, CUDA first', () => {
    expect(solutionTabStatus(reports({ emu: true, cuda: true, hip: true }), lab, 'kernel.cu')).toBe('verified on CUDA, verified on HIP, emulator-checked');
  });
});

describe('raceSolutionStatus', () => {
  it('reads the race run, and knows a lab not built for it', () => {
    expect(raceSolutionStatus(reports({ race: true }), lab, [])?.text).toBe('emulator-checked');
    expect(raceSolutionStatus(reports({ race: false }), lab, [lab.id])?.text).toBe('unverified: failed on this machine');
    expect(raceSolutionStatus({}, lab, [lab.id])?.text).toBe('no run on this machine');
    expect(raceSolutionStatus({}, lab, ['ch09/lab2_y'])?.text).toBe('not built for race');
  });
});

describe('benchRow', () => {
  const set: BenchSet = { backend: 'metal', device: 'd', date: '2026-11-21', variant: 'solution', rows: [{ name: 'gemv', format: 'q4_0', tile: 16 }] };
  it('matches strings and numbers together', () => {
    expect(benchRow(set, 'gemv', { format: 'q4_0', tile: 16 })).toBe(set.rows[0]);
    expect(benchRow(set, 'gemv', { format: 'q8_0' })).toBeUndefined();
  });
});
