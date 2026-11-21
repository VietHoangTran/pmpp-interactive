import { useEffect, useState } from 'react';

// Results written by ./verify.sh (reports/runs/<backend>.json): what the reader's
// own devices reported and measured. The app never substitutes made-up values.
export interface BenchRow {
  name: string;
  [k: string]: string | number;
}
export interface RunResult {
  lab: string;
  variant: string;
  backend: string;
  tile: number;
  built: boolean;
  test_rc: number | null;
  tests: { case: string; pass: boolean; detail?: string }[];
  bench: BenchRow[];
  /** ThreadSanitizer reports of a run on the race backend (empty when none was found). */
  tsan?: unknown[];
}
export interface BackendRun {
  generated: string;
  device: Record<string, string | boolean>;
  results: RunResult[];
  /** Per lab: 'untouched' while work/ is still identical to starter/, else 'edited'. */
  work_state?: Record<string, 'untouched' | 'edited'>;
}
export type Reports = Record<string, BackendRun>;

export function useReports(): { reports: Reports; loading: boolean; reload: () => void } {
  const [reports, setReports] = useState<Reports>({});
  const [loading, setLoading] = useState(true);
  const [n, setN] = useState(0);
  useEffect(() => {
    setLoading(true);
    fetch('/api/reports')
      .then((r) => r.json())
      .then(setReports)
      .catch(() => setReports({}))
      .finally(() => setLoading(false));
  }, [n]);
  return { reports, loading, reload: () => setN((x) => x + 1) };
}

// Real GPU backends only (the emulator has no meaningful timings).
/** Backends that ran on a GPU: neither emulator flavour (emu, race) counts as a device. */
export const realBackends = (r: Reports) => Object.keys(r).filter((b) => b !== 'emu' && b !== 'race');

export function benchRows(run: BackendRun | undefined, lab: string, variant = 'solution'): BenchRow[] {
  if (!run) return [];
  return run.results.filter((x) => x.lab === lab && x.variant === variant).flatMap((x) => x.bench);
}

/**
 * The first row called `name` whose fields equal `where`. A number matches a numeric field
 * with the same value; a string matches a field whose value, as text, is that string
 * (so { dtype: 'f16' } and { tile: 16 } both work). Nothing is coerced the other way.
 */
export function pickBenchRow(
  rows: readonly BenchRow[] | undefined,
  name: string,
  where: Record<string, number | string> = {},
): BenchRow | undefined {
  return rows?.find(
    (r) =>
      r.name === name &&
      Object.entries(where).every(([k, v]) => (typeof v === 'number' ? num(r[k]) === v : r[k] != null && String(r[k]) === v)),
  );
}

export type LabState = 'pass' | 'fail' | 'none';

/** Everything the labs pages say about one lab, one variant, on one backend. */
export interface LabSummary {
  /** pass: built and every test passed; fail: a run that did not; none: no evidence (or the untouched starter). */
  status: LabState;
  /** There is a recorded result at all (an untouched starter still counts as recorded). */
  recorded: boolean;
  /** Some target of the lab did not build. */
  buildFailed: boolean;
  /** The run stopped early: an exit code other than 0 and 1 (a crash, or a sanitizer aborting it). */
  aborted: boolean;
  passed: number;
  total: number;
  /** ThreadSanitizer reports in the run (race backend only; 0 elsewhere). */
  tsanReports: number;
}

/** One lab's results on one backend, counted. labStatus() is its `status`. */
export function labSummary(run: BackendRun | undefined, lab: string, variant: string): LabSummary {
  const rs = run?.results.filter((x) => x.lab === lab && x.variant === variant) ?? [];
  const tests = rs.flatMap((x) => x.tests);
  return {
    status: labStatus(run, lab, variant),
    recorded: rs.length > 0,
    buildFailed: rs.some((x) => !x.built),
    aborted: rs.some((x) => x.test_rc != null && x.test_rc !== 0 && x.test_rc !== 1),
    passed: tests.filter((t) => t.pass).length,
    total: tests.length,
    tsanReports: rs.reduce((n, x) => n + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0),
  };
}

export function labStatus(run: BackendRun | undefined, lab: string, variant: string): LabState {
  // An untouched starter fails its tests by design; that is "not started", not a failure.
  if (variant === 'work' && run?.work_state?.[lab] === 'untouched') return 'none';
  const rs = run?.results.filter((x) => x.lab === lab && x.variant === variant && x.tests.length) ?? [];
  if (!rs.length) return 'none';
  return rs.every((x) => x.built && x.test_rc === 0 && x.tests.every((t) => t.pass)) ? 'pass' : 'fail';
}

export const num = (v: unknown): number | undefined => (typeof v === 'number' && isFinite(v) ? v : undefined);

export function fmt(v: number, digits = 3): string {
  if (!isFinite(v)) return 'n/a';
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a >= 1000) return v.toLocaleString('en-US', { maximumFractionDigits: 0 });
  return Number(v.toPrecision(digits)).toString();
}
