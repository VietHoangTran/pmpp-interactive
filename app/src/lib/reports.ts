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
}
export interface BackendRun {
  generated: string;
  device: Record<string, string | boolean>;
  results: RunResult[];
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
export const realBackends = (r: Reports) => Object.keys(r).filter((b) => b !== 'emu');

export function benchRows(run: BackendRun | undefined, lab: string, variant = 'solution'): BenchRow[] {
  if (!run) return [];
  return run.results.filter((x) => x.lab === lab && x.variant === variant).flatMap((x) => x.bench);
}

export function labStatus(run: BackendRun | undefined, lab: string, variant: string): 'pass' | 'fail' | 'none' {
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
