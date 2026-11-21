// LabKit: everything a chapter's labs page shares with every other chapter's.
// A labs page is its lab descriptors, its prose, and these components. Chapter 5
// (src/chapters/ch05/Labs.tsx) is the worked example.
//
// ---------------------------------------------------------------- the smallest complete page
//
//   import type { Question } from '../../components/Ask';
//   import { Source } from '../../components/notes';
//   import {
//     BackendsAndEmulator, LabFiles, LabResults, LabsIntro, LabsPage, ReferenceSolution,
//     RunningLabs, VerificationStatus, defineLabs, msCol, numCol,
//   } from '../../components/LabKit';
//
//   export const questionsLabs: Record<string, Question> = {};
//
//   const LABS = defineLabs('chNN', [{ num: 'N.1', dir: 'lab1_short', name: 'short name' }]);
//   const [l1] = LABS;
//
//   export function Labs() {
//     return (
//       <LabsPage labs={LABS}>
//         <LabsIntro>One lab, in <code>labs/chNN/</code>: what it is for.</LabsIntro>
//         <RunningLabs example="N.1" />
//         <BackendsAndEmulator />
//         <VerificationStatus />
//
//         <h2>Lab N.1: its title</h2>
//         <p>What the reader writes, and why.</p>
//         <Source section="N.2" pages="..." />
//         <LabFiles lab={l1} files={[['kernel.cu', <>what to write in it</>]]} />
//         <p>Run <code>python3 labs/pmpp.py run N.1</code>. What the tests check.</p>
//         <LabResults lab={l1} cols={[msCol, numCol('GFLOP/s', 'gflops')]} />
//         <ReferenceSolution lab={l1} />
//       </LabsPage>
//     );
//   }
//
// Register it in the chapter's index.tsx exactly as chapter 5 does, and give it a
// browser check (e2e/ch05-labs.mjs is the model; its data-* selectors all come from here).
//
// ---------------------------------------------------------------- the pieces
//
// defineLabs(chapter, [{ num, dir, name, sources? }])
//     The lab descriptors, in page order. `num` is what the reader types after
//     `pmpp.py run` ("5.2"); `dir` is the directory under labs/<chapter>/ ("lab2_tiled");
//     `name` is the short name in the status table ("tiled"). The id ("ch05/lab2_tiled",
//     the key in reports/runs and the path under labs/) is built for you. `sources` lists
//     which of kernel.cu, kernel.metal and launch.mm the lab has; leave it out when it has
//     all three. A lab with no Metal version is `sources: ['kernel.cu']`.
//
// <LabsPage labs tiers?>
//     Wraps the whole page. Reads reports/runs once and gives every component below the
//     same results and the same "Reload results". Renders nothing of its own. `tiers`
//     colours identifiers in the reference solutions by the memory they live in (chapter 5
//     passes MATMUL_TIERS); leave it out and nothing is coloured.
//
// Standard sections, the same in every chapter. Each renders its own <h2>.
// <LabsIntro>lead</LabsIntro>
//     Your lead sentences, then the standard account of starter/, work/, solution/, the
//     driver and the three files. For a chapter whose labs do not have all three files,
//     write these two paragraphs yourself.
// <RunningLabs example alsoShownIn?>
//     "Running them": the commands table, with `example` (one of your lab numbers) in the
//     per-lab commands, where the results come from, and the note that every number is a
//     measurement. `alsoShownIn` names another place that shows the same runs.
// <BackendsAndEmulator />
//     "The four backends": what a pass on each one tells you, and what the emulator is.
// <VerificationStatus />
//     "Verification status of the reference solutions": the vocabulary, the status table
//     for your labs (<SolutionStatusTable />) and the CUDA/HIP caveat (<CudaHipCaveat />).
//
// Per lab, in this order.
// <LabFiles lab files={[[file, what the reader does in it], ...]} />
//     The files the reader edits, as paths under labs/<id>/work/.
// <LabPrediction id prompt unit measure reading missing? />
//     A prediction the reader commits before running the benchmark, then this machine's
//     measurement and one sentence comparing the two. `id` is a progress key
//     ("ch05/labs/p-lab2-speedup"): never rename one once shipped. `measure` turns the
//     reports into values, one per backend; build it with measure() below. `reading` says
//     how to read the number without saying what it should be. `missing` replaces the
//     standard "what to run" text shown when there is no measurement.
//     (<Reveal committed measure unit reading missing? /> is the part after the commit, for
//     use inside your own <Predict>.)
// <LabResults lab cols filter? />   or   <LabResults lab noBench />
//     The reader's pass/fail per backend (<WorkStatus lab />), then the benchmark rows from
//     the reports: the reader's own kernel where it passed, and the reference solution.
//     `cols` are the columns after Backend and Kernel: numCol(head, key, digits?) formats a
//     number, rawCol(head, key) prints the value as reported, msCol is "Time, ms" from the
//     row's seconds, and { head, num?, cell: (row) => ... } is anything else. `filter`
//     picks the rows. For a lab with no benchmark pass `noBench` (true for the standard
//     sentence, or your own text) in place of `cols`.
// <ReferenceSolution lab tiers? />
//     The "Show the reference solution" toggle: the lab's solution files as tabs, each with
//     a status computed from the reports in the project's fixed vocabulary.
//
// Reading the reports for a prediction (plain functions; none invents a value).
// measure(reports, lab, (set) => ({ value, detail, whose? }) | null)
//     One measured value per real backend, taken from the reader's own benchmark rows where
//     their kernel passed and from the reference solution's otherwise. Return null when the
//     rows you need are not there, and that backend is left out.
// benchRow(set, name, { key: value }?)   the first row with that name and those fields (numbers or strings)
// preferredBench(reports, lab)           the sets measure() walks, for a comparison across two labs
// benchSets(reports, lab)                every set: the reader's and the reference's, per backend
// whose(set)                             "your kernel" or "the reference solution (...)"
// compare(predicted, measured)           the comparison sentence
// solutionStatus(reports, backend, lab)  one cell of the status table
// solutionTabStatus(reports, lab, file)  the status a reference-solution tab shows (same words as the table)
// useLabs()                              { labs, reports, loading, reload, tiers, raceLabs } inside <LabsPage>
//
// The race backend (ThreadSanitizer emulator) for labs built with RACE:
// <RaceStatus lab work? />
//     One sentence on the reference solution on the race backend, and one on the reader's own
//     kernel (leave it out with work={false}). The status table also gets a "race" column
//     whenever one of the page's labs is built for it or has a race run.
//
// What is missing is said plainly. A backend with no run reads "no run on this machine"; a
// lab without kernel.metal reads "no Metal files in this lab" and gets no Metal tabs; a lab
// with no benchmark rows says what to run. Nothing here substitutes a value.
import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import { Predict } from './Ask';
import { Code, useLabFile } from './Code';
import { Aside, Bench, type Tier } from './notes';
import {
  benchRows,
  fmt,
  frontendStatus,
  labStatus,
  labSummary,
  num,
  pickBenchRow,
  realBackends,
  useReports,
  type BenchRow,
  type LabState,
  type Reports,
} from '../lib/reports';

// ---------------------------------------------------------------- lab descriptors
/** The files a reader edits. One .cu serves CUDA, HIP and the emulator; the other two are Metal. */
export type LabSource = 'kernel.cu' | 'kernel.metal' | 'launch.mm';
const ALL_SOURCES: readonly LabSource[] = ['kernel.cu', 'kernel.metal', 'launch.mm'];

export interface LabSpec {
  /** What the reader types after `pmpp.py run`, e.g. "5.2". */
  num: string;
  /** Directory under labs/<chapter>/, e.g. "lab2_tiled". */
  dir: string;
  /** Short name for the status table, e.g. "tiled". */
  name: string;
  /** Which student files the lab has. Leave out when it has all three. */
  sources?: readonly LabSource[];
}
export interface Lab extends LabSpec {
  /** "ch05/lab2_tiled": the key in reports/runs and the lab's path under labs/. */
  id: string;
  sources: readonly LabSource[];
}

// An authoring mistake is reported on the console, where the browser checks fail on it.
const complain = (what: string) => console.error('LabKit: ' + what);

/** The chapter's labs, in page order. Destructure the result: `const [l1, l2] = LABS`. */
export function defineLabs<const T extends readonly LabSpec[]>(chapter: string, specs: T): { readonly [K in keyof T]: Lab } {
  const labs = specs.map((s): Lab => ({ ...s, id: `${chapter}/${s.dir}`, sources: s.sources ?? ALL_SOURCES }));
  for (const [i, l] of labs.entries()) {
    if (labs.findIndex((x) => x.num === l.num || x.dir === l.dir) !== i) complain(`lab ${l.num} (${l.id}) repeats an earlier lab's num or dir.`);
    // `pmpp.py run 5.2` looks for the one lab whose id starts with ch05/lab2_.
    const m = /^(\d+)\.(\d+)$/.exec(l.num);
    const want = m ? `ch${m[1].padStart(2, '0')}/lab${m[2]}_` : '';
    if (!l.id.startsWith(want)) complain(`"pmpp.py run ${l.num}" looks for ${want}*, which is not ${l.id}.`);
  }
  return labs as unknown as { readonly [K in keyof T]: Lab };
}

// ---------------------------------------------------------------- shared vocabulary
export const BACKENDS = ['metal', 'emu', 'cuda', 'hip'] as const;
export type Backend = (typeof BACKENDS)[number];
const NAME: Record<Backend, string> = { metal: 'Metal', emu: 'emulator', cuda: 'CUDA', hip: 'HIP' };
const KERNEL: Record<Backend, LabSource> = { metal: 'kernel.metal', emu: 'kernel.cu', cuda: 'kernel.cu', hip: 'kernel.cu' };
/** What to say where a lab has no source for a backend. */
const noSource = (l: Lab, b: Backend): string | null =>
  l.sources.includes(KERNEL[b]) ? null : b === 'metal' ? 'no Metal files in this lab' : 'no kernel.cu in this lab';

// ---------------------------------------------------------------- the page's shared state
interface Kit {
  labs: readonly Lab[];
  reports: Reports;
  loading: boolean;
  reload: () => void;
  tiers: Record<string, Tier>;
  /** Ids of the labs built for the race backend (RACE in their CMakeLists.txt); null while reading. */
  raceLabs: readonly string[] | null;
}
const KitContext = createContext<Kit | null>(null);
const NO_TIERS: Record<string, Tier> = {};

/** Wraps a labs page: one reading of reports/runs, shared by every LabKit component inside. */
export function LabsPage({
  labs,
  tiers = NO_TIERS,
  children,
}: {
  labs: readonly Lab[];
  /** Identifier colours for the reference solutions (see Code.tsx). None by default. */
  tiers?: Record<string, Tier>;
  children: ReactNode;
}) {
  const shared = useReports();
  const raceLabs = useRaceLabs();
  return <KitContext.Provider value={{ labs, tiers, raceLabs, ...shared }}>{children}</KitContext.Provider>;
}

/** Which labs are built for the race backend, from the server (it reads each lab's CMakeLists.txt). */
function useRaceLabs(): readonly string[] | null {
  const [ids, setIds] = useState<readonly string[] | null>(null);
  useEffect(() => {
    fetch('/api/race-labs')
      .then((r) => (r.ok ? r.json() : []))
      .then((x) => setIds(Array.isArray(x) ? x : []))
      .catch(() => setIds([]));
  }, []);
  return ids;
}

export function useLabs(): Kit {
  const kit = useContext(KitContext);
  if (!kit) throw new Error('LabKit components must be rendered inside <LabsPage labs={...}>.');
  return kit;
}

// ---------------------------------------------------------------- status of the reference solutions
/** Status of a reference solution on one backend, in the project's fixed vocabulary. */
export function solutionStatus(reports: Reports, b: Backend, l: Lab): { text: string; tone: 'ok' | 'fail' | 'none' } {
  const s = labStatus(reports[b], l.id, 'solution');
  if (s === 'pass') return { text: b === 'emu' ? 'emulator-checked' : `verified on ${NAME[b]}`, tone: 'ok' };
  if (s === 'fail') return { text: 'unverified: failed on this machine', tone: 'fail' };
  return { text: noSource(l, b) ?? NO_RUN, tone: 'none' };
}

// One wording per status, everywhere on a labs page (status table, solution tabs, race line).
const FAILED = 'failed on this machine';
const NO_RUN = 'no run on this machine';
const WHERE: Record<Backend | 'race', string> = { metal: 'Metal', emu: 'the emulator', cuda: 'CUDA', hip: 'HIP', race: 'the race emulator' };
const and = (xs: string[]) => (xs.length < 3 ? xs.join(' and ') : xs.slice(0, -1).join(', ') + ' and ' + xs.at(-1));

/**
 * The status a reference-solution tab shows, in the same words as the status table. The
 * Metal tabs speak for Metal: "verified on Metal", "unverified: failed on this machine" or
 * "unverified: no run on this machine". The kernel.cu tab speaks for the emulator, CUDA and
 * HIP at once: what passed, then what failed ("unverified on CUDA: failed on this machine"),
 * then the CUDA-family backends with no run ("unverified on CUDA and HIP"), e.g.
 * "emulator-checked, front-end-checked; unverified on CUDA and HIP". "front-end-checked"
 * comes from this machine's last CUDA front-end check of the solution's kernel.cu (only while
 * the file is unchanged since, and only until CUDA itself has passed). It says a run failed
 * whenever one did.
 */
export function solutionTabStatus(reports: Reports, l: Lab | string, file: LabSource): string {
  const id = typeof l === 'string' ? l : l.id;
  const st = (b: Backend) => labStatus(reports[b], id, 'solution');
  if (file !== 'kernel.cu') {
    const s = st('metal');
    return s === 'pass' ? 'verified on Metal' : `unverified: ${s === 'fail' ? FAILED : NO_RUN}`;
  }
  // In the order the status reads: CUDA, HIP, then the emulator.
  const fam: Backend[] = ['cuda', 'hip', 'emu'];
  const passed = fam.filter((b) => st(b) === 'pass').map((b) => (b === 'emu' ? 'emulator-checked' : `verified on ${NAME[b]}`));
  const failed = fam.filter((b) => st(b) === 'fail').map((b) => WHERE[b]);
  const none = (['cuda', 'hip'] as Backend[]).filter((b) => st(b) === 'none').map((b) => NAME[b]);
  // The CUDA front-end check (clang parse against stubs) adds to the record until CUDA itself has passed.
  const fe = st('cuda') === 'pass' ? 'none' : frontendStatus(reports, `labs/${id}/solution/kernel.cu`);
  if (fe === 'ok') passed.push('front-end-checked');
  if (!passed.length && !failed.length && fe === 'none') return `unverified: ${NO_RUN}`;
  return [
    passed.join(', '),
    failed.length ? `unverified on ${and(failed)}: ${FAILED}` : '',
    fe === 'fail' ? `failed the CUDA front-end check on this machine` : '',
    none.length ? `unverified on ${none.join(' and ')}` : '',
  ]
    .filter(Boolean)
    .join('; ');
}

/** Status of a reference solution on the race backend; "not built for race" when its CMakeLists.txt does not ask for it. */
export function raceSolutionStatus(
  reports: Reports,
  l: Lab,
  raceLabs: readonly string[] | null,
): { text: string; tone: 'ok' | 'fail' | 'none' } {
  const s = labStatus(reports.race, l.id, 'solution');
  if (s === 'pass') return { text: 'emulator-checked', tone: 'ok' };
  if (s === 'fail') return { text: `unverified: ${FAILED}`, tone: 'fail' };
  if (raceLabs && !raceLabs.includes(l.id)) return { text: 'not built for race', tone: 'none' };
  return { text: noSource(l, 'emu') ?? NO_RUN, tone: 'none' };
}

/** Every lab of the page against every backend, with the runs the statuses come from. */
export function SolutionStatusTable() {
  const { labs, reports, loading, raceLabs } = useLabs();
  const runs = BACKENDS.filter((b) => reports[b]);
  // The race column appears when a lab of this page is built for it or has a race run.
  const race = labs.some((l) => raceLabs?.includes(l.id) || reports.race?.results.some((x) => x.lab === l.id));
  const raceRun = race && reports.race;
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
      <table className="data" data-table="solution-status">
        <thead>
          <tr>
            <th>Reference solution</th>
            {BACKENDS.map((b) => (
              <th key={b}>{b === 'emu' ? 'emu' : NAME[b]}</th>
            ))}
            {race ? <th title="The emulator run with real threads under ThreadSanitizer">race</th> : null}
          </tr>
        </thead>
        <tbody>
          {labs.map((l) => (
            <tr key={l.id}>
              <td>
                Lab {l.num}, {l.name}
              </td>
              {BACKENDS.map((b) => {
                const s = solutionStatus(reports, b, l);
                return (
                  <td key={b} data-lab={l.num} data-backend={b} data-status={loading ? 'loading' : s.text}>
                    {loading ? (
                      <span className="muted">reading reports</span>
                    ) : s.tone === 'none' ? (
                      <span className="muted">{s.text}</span>
                    ) : (
                      <span className={'chip ' + s.tone}>{s.text}</span>
                    )}
                  </td>
                );
              })}
              {race
                ? (() => {
                    const s = raceSolutionStatus(reports, l, raceLabs);
                    // data-race-cell, not data-lab (the four backend columns keep their selector) nor data-race-lab (chapters' race lines use it).
                    return (
                      <td key="race" data-race-cell={l.num} data-status={loading ? 'loading' : s.text}>
                        {loading ? (
                          <span className="muted">reading reports</span>
                        ) : s.tone === 'none' ? (
                          <span className="muted">{s.text}</span>
                        ) : (
                          <span className={'chip ' + s.tone}>{s.text}</span>
                        )}
                      </td>
                    );
                  })()
                : null}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="status-line" style={{ margin: '-0.8rem 0 1.4rem' }}>
        {loading
          ? 'Reading reports/runs.'
          : runs.length
            ? 'Runs on this machine: ' +
              runs.map((b) => `${b} (${String(reports[b].device.name ?? 'unnamed device')}), ${reports[b].generated.replace('T', ' ')}`).join('; ') +
              (raceRun ? `; race, ${reports.race.generated.replace('T', ' ')}` : '') +
              '.'
            : 'No runs on this machine yet. Run ./verify.sh in the repository root, then reload this page.'}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- benchmark rows from the reports
export interface BenchSet {
  backend: string;
  device: string;
  /** Day of the run, yyyy-mm-dd. */
  date: string;
  variant: 'work' | 'solution';
  rows: BenchRow[];
}

/** Benchmark rows per real backend: the reader's own kernel first when its tests passed, then the reference. */
export function benchSets(reports: Reports, l: Lab): BenchSet[] {
  return realBackends(reports).flatMap((b) => {
    const run = reports[b];
    const base = { backend: b, device: String(run.device.name ?? b), date: run.generated.slice(0, 10) };
    const out: BenchSet[] = [];
    const mine = labStatus(run, l.id, 'work') === 'pass' ? benchRows(run, l.id, 'work') : [];
    if (mine.length) out.push({ ...base, variant: 'work', rows: mine });
    const ref = benchRows(run, l.id, 'solution');
    if (ref.length) out.push({ ...base, variant: 'solution', rows: ref });
    return out;
  });
}
/** One set per backend: the reader's own if there is one, else the reference solution's. */
export function preferredBench(reports: Reports, l: Lab): BenchSet[] {
  const seen = new Set<string>();
  return benchSets(reports, l).filter((s) => (seen.has(s.backend) ? false : (seen.add(s.backend), true)));
}
/** Whose kernel a set of rows measured, as the reveal words it. */
export const whose = (s: BenchSet) =>
  s.variant === 'work' ? 'your kernel' : 'the reference solution (no passing benchmark of your own kernel yet)';

/**
 * The first row called `name` whose fields equal `where`: numbers match numeric fields, strings
 * match text fields, e.g. benchRow(s, 'tiled', { tile: 16 }) or benchRow(s, 'gemv', { format: 'q4_0' }).
 */
export const benchRow = (set: BenchSet | undefined, name: string, where: Record<string, number | string> = {}) =>
  pickBenchRow(set?.rows, name, where);

export interface Measured {
  backend: string;
  device: string;
  date: string;
  whose: string;
  value: number;
  /** The figures the value was computed from, with their units. */
  detail: string;
}

/**
 * One measured value per real backend for a prediction. `pick` gets that backend's preferred
 * rows and returns the value with the figures behind it, or null when the rows it needs are
 * missing; a backend without a finite value is left out, never filled in.
 */
export function measure(
  reports: Reports,
  l: Lab,
  pick: (set: BenchSet) => { value: number; detail: string; whose?: string } | null | undefined,
): Measured[] {
  return preferredBench(reports, l).flatMap((s) => {
    const m = pick(s);
    if (!m || !isFinite(m.value)) return [];
    return [{ backend: s.backend, device: s.device, date: s.date, whose: m.whose ?? whose(s), value: m.value, detail: m.detail }];
  });
}

// ---------------------------------------------------------------- prediction against measurement
/** One plain sentence comparing a committed prediction with a measurement. */
export function compare(predicted: number, measured: number): string {
  if (!(predicted > 0)) return 'Your prediction was not a positive ratio, so there is no factor to compare.';
  if (!(measured > 0)) return 'The measurement is not a positive ratio, so there is no factor to compare.';
  const r = predicted / measured;
  if (Math.abs(r - 1) <= 0.1) return 'Your prediction was within 10% of the measurement.';
  return r < 1
    ? `The measurement is ${fmt(1 / r)} times your prediction.`
    : `Your prediction is ${fmt(r)} times the measurement.`;
}

interface RevealProps {
  /** Reports to measured values, one per backend. Build it with measure(). */
  measure: (reports: Reports) => Measured[];
  /** What the number is a multiple of, e.g. "× the naive kernel". */
  unit: string;
  /** How to read the number, without saying what it should be. */
  reading: ReactNode;
  /** What to run when there is no measurement, if the standard instruction does not fit. */
  missing?: ReactNode;
}

/** What follows a committed prediction: this machine's measurement, compared with it. */
export function Reveal({ committed, measure: read, unit, reading, missing }: RevealProps & { committed: number }) {
  const { reports, loading, reload } = useLabs();
  const measured = read(reports);
  return (
    <>
      {measured.length ? (
        measured.map((m) => (
          <p key={m.backend} data-measured={fmt(m.value)} data-backend={m.backend}>
            Measured on {m.device} ({m.backend}) with {m.whose}, in the run of {m.date}: <b>{fmt(m.value)}</b> {unit}{' '}
            ({m.detail}). {compare(committed, m.value)}
          </p>
        ))
      ) : loading ? (
        <p data-measured="loading">Reading reports.</p>
      ) : (
        <p data-measured="none">
          No measurement on this machine yet.{' '}
          {missing ?? (
            <>
              Run <code>./verify.sh</code> without <code>--quick</code> on a machine with a GPU. It benchmarks the
              reference solution, and your own kernel once that passes its tests.
            </>
          )}{' '}
          <button className="quiet" onClick={reload}>
            Reload results
          </button>
        </p>
      )}
      <p className="muted">{reading}</p>
    </>
  );
}

/** A prediction committed before the benchmark, then the measurement. Stored, not graded. */
export function LabPrediction({
  id,
  prompt,
  ...reveal
}: RevealProps & {
  /** Progress key, e.g. "ch05/labs/p-lab2-speedup". Never rename one once shipped. */
  id: string;
  prompt: ReactNode;
}) {
  return (
    <Predict id={id} prompt={prompt} unit={reveal.unit}>
      {(v) => <Reveal committed={v} {...reveal} />}
    </Predict>
  );
}

// ---------------------------------------------------------------- the reader's results
/** Pass/fail of the reader's own work on every backend. */
export function WorkStatus({ lab: l }: { lab: Lab }) {
  const { reports, loading } = useLabs();
  return (
    <div className="row" data-work={l.id} style={{ marginBottom: '0.9rem' }}>
      <b>Your work</b>
      {BACKENDS.map((b) => {
        const run = reports[b];
        const { status: s, recorded, buildFailed, aborted, passed, total } = labSummary(run, l.id, 'work');
        const absent = recorded ? null : noSource(l, b);
        let body: ReactNode;
        if (loading) body = <span className="muted">reading reports</span>;
        else if (absent) body = <span className="muted">{absent}</span>;
        else if (!run) body = <span className="muted">{NO_RUN}</span>;
        else if (buildFailed) body = <span className="chip fail">does not build</span>;
        else if (s === 'none')
          body = (
            <span className="muted" title="No recorded run of this lab on this machine">
              not started
            </span>
          );
        else if (s === 'pass') body = <span className="chip ok">passes, {passed} of {total} tests</span>;
        else {
          // `aborted`: an exit code other than 0 and 1, the run stopped early (a crash, or AddressSanitizer aborting it).
          body = (
            <span className="chip fail">
              not passing: {passed} of {total} tests{aborted ? ', then the run aborted' : ''}
            </span>
          );
        }
        return (
          <span key={b} data-backend={b}>
            {b}: {body}
          </span>
        );
      })}
    </div>
  );
}

/**
 * The race backend for one lab, which the four status columns do not cover: the reference
 * solution on it, then (unless work={false}) the reader's own kernel. The race backend is
 * the emulator with real threads under ThreadSanitizer, so a pass there is
 * "emulator-checked", and with the plain emulator passing too, under both flavours.
 * data-race-status and data-race-work carry pass, fail or none for browser checks.
 */
export function RaceStatus({ lab: l, work = true }: { lab: Lab; work?: boolean }) {
  const { reports, loading, raceLabs } = useLabs();
  const run = reports.race;
  const sol = labSummary(run, l.id, 'solution');
  const mine = labSummary(run, l.id, 'work');
  const built = !raceLabs || raceLabs.includes(l.id) || sol.recorded || mine.recorded;
  const tsan = (n: number) => (n ? `${n} ThreadSanitizer report${n === 1 ? '' : 's'}` : 'no race reported');
  const emuToo = labStatus(reports.emu, l.id, 'solution') === 'pass';
  const solText = (): string => {
    if (sol.status === 'none') return `On this machine the reference solution has no run on the race backend yet.`;
    if (sol.status === 'pass')
      return `On this machine the reference solution passed all ${sol.total} tests on the race backend with ${tsan(sol.tsanReports)}: emulator-checked ${emuToo ? 'under both flavours' : 'on this flavour'}.`;
    return `On the race backend the reference solution is unverified: ${FAILED} (${sol.passed} of ${sol.total} tests passed${sol.buildFailed ? ', a target did not build' : ''}${sol.aborted ? ', then the run aborted' : ''}, ${tsan(sol.tsanReports)}).`;
  };
  const workText = (): string => {
    if (mine.status === 'none') return 'Your own kernel has no recorded run there, or is still the untouched starter.';
    if (mine.status === 'pass')
      return `Your own kernel passed all ${mine.total} tests there with ${mine.tsanReports ? `${tsan(mine.tsanReports)}: read them` : 'no race reported'}.`;
    return `Your own kernel passed ${mine.passed} of ${mine.total} tests there${mine.aborted ? ' before the run aborted' : ''}, with ${tsan(mine.tsanReports)} in the run.`;
  };
  return (
    <p
      data-race-status={loading ? 'loading' : sol.status}
      data-race-work={loading ? 'loading' : (mine.status satisfies LabState)}
      data-race-lab={l.num}
    >
      {loading
        ? 'Reading reports.'
        : !built
          ? `Lab ${l.num} is not built for the race backend (its CMakeLists.txt does not pass RACE).`
          : !run
            ? 'No run on the race backend on this machine yet.'
            : solText() + (work ? ' ' + workText() : '')}
    </p>
  );
}

/** A column of a results table, after the Backend and Kernel columns every table has. */
export interface Col {
  head: string;
  /** Right-aligned, for figures. */
  num?: boolean;
  cell: (r: BenchRow) => ReactNode;
}
/** A measured number, formatted to `digits` significant figures. Blank when the row lacks it. */
export const numCol = (head: string, key: string, digits = 3): Col => ({
  head,
  num: true,
  cell: (r) => {
    const v = num(r[key]);
    return v == null ? '' : fmt(v, digits);
  },
});
/** A parameter of the run (a width, a batch size) exactly as the report has it. Blank when the row lacks it. */
export const rawCol = (head: string, key: string): Col => ({ head, num: true, cell: (r) => String(r[key] ?? '') });
/** The row's `seconds`, in milliseconds. */
export const msCol: Col = {
  head: 'Time, ms',
  num: true,
  cell: (r) => {
    const v = num(r.seconds);
    return v == null ? '' : fmt(v * 1000);
  },
};

const NO_BENCH = 'This lab has no benchmark. The result is the line above: every test passing on the emulator and on a device.';

type LabResultsProps = { lab: Lab } & (
  | {
      /** Columns after Backend and Kernel. */
      cols: Col[];
      /** Which benchmark rows belong in the table. All of them by default. */
      filter?: (r: BenchRow) => boolean;
      noBench?: undefined;
    }
  | {
      /** For a lab without a benchmark: true for the standard sentence, or what counts as the result. */
      noBench: true | NonNullable<Exclude<ReactNode, boolean>>;
      cols?: undefined;
      filter?: undefined;
    }
);

/** "Lab N on this machine": the reader's status, then the benchmark rows the reports hold. */
export function LabResults({ lab: l, cols, filter, noBench }: LabResultsProps) {
  const { reports, loading, reload } = useLabs();
  const sets = cols ? benchSets(reports, l) : [];
  const any = sets.some((s) => s.rows.some((r) => !filter || filter(r)));
  return (
    <Bench
      title={`Lab ${l.num} on this machine`}
      task={
        <button className="quiet" onClick={reload}>
          Reload results
        </button>
      }
      foot={
        noBench
          ? "From this machine's recorded lab runs."
          : "From this machine's recorded lab runs. Measurements, not expected values: another device, or another run, gives other numbers."
      }
    >
      <WorkStatus lab={l} />
      {noBench ? (
        <p className="muted" style={{ margin: 0 }}>
          {noBench === true ? NO_BENCH : noBench}
        </p>
      ) : any ? (
        <div className="panel" style={{ overflowX: 'auto' }}>
          <table className="data" style={{ margin: 0 }} data-results={l.num}>
            <thead>
              <tr>
                <th>Backend</th>
                <th>Kernel</th>
                {(cols ?? []).map((c) => (
                  <th key={c.head} className={c.num ? 'num' : undefined}>
                    {c.head}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sets.flatMap((s) =>
                s.rows
                  .filter((r) => !filter || filter(r))
                  .map((r, i) => (
                    <tr key={s.backend + s.variant + i}>
                      <td>
                        {s.backend} <span className="muted">({s.device})</span>
                      </td>
                      <td>{s.variant === 'work' ? 'yours' : 'reference solution'}</td>
                      {(cols ?? []).map((c) => (
                        <td key={c.head} className={c.num ? 'num' : undefined}>
                          {c.cell(r)}
                        </td>
                      ))}
                    </tr>
                  )),
              )}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="muted" style={{ margin: 0 }} data-results-empty={l.num}>
          {loading ? (
            'Reading reports.'
          ) : (
            <>
              No benchmark rows for this lab on this machine. Run <code>./verify.sh</code> without{' '}
              <code>--quick</code> on a machine with a GPU, then reload. The emulator is never benchmarked.
            </>
          )}
        </p>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- files and the reference solution
/** The files the reader edits in this lab, and what to do in each. */
export function LabFiles({ lab: l, files }: { lab: Lab; files: [LabSource, ReactNode][] }) {
  for (const [f] of files) if (!l.sources.includes(f)) complain(`lab ${l.num} lists ${f}, which is not in its sources.`);
  return (
    <ul>
      {files.map(([f, what]) => (
        <li key={f}>
          <code>
            labs/{l.id}/work/{f}
          </code>
          : {what}
        </li>
      ))}
    </ul>
  );
}

/** Loads lab files with one useLabFile per component, so a lab can have any of the three. */
function LabFileTexts({ paths, children }: { paths: readonly string[]; children: (texts: (string | null)[]) => ReactNode }) {
  const [first, ...rest] = paths;
  const text = useLabFile(first);
  return rest.length ? (
    <LabFileTexts paths={rest}>{(more) => children([text, ...more])}</LabFileTexts>
  ) : (
    <>{children([text])}</>
  );
}

const TAB: Record<LabSource, string> = { 'kernel.cu': 'CUDA', 'kernel.metal': 'Metal kernel', 'launch.mm': 'Metal host' };

/** The real reference files, with a status per tab computed from the reports. */
function SolutionCode({ lab: l, tiers }: { lab: Lab; tiers: Record<string, Tier> }) {
  const { reports } = useLabs();
  // kernel.cu is one source for CUDA, HIP and the emulator; its tab speaks for all three.
  const dir = `${l.id}/solution/`;
  // In the fixed order of the tabs, whatever order the descriptor lists them in.
  const sources = ALL_SOURCES.filter((f) => l.sources.includes(f));
  const absent = [
    sources.includes('kernel.cu') ? '' : 'no kernel.cu, so nothing for CUDA, HIP or the emulator',
    sources.includes('kernel.metal') && sources.includes('launch.mm')
      ? ''
      : 'no ' + (['kernel.metal', 'launch.mm'] as LabSource[]).filter((f) => !sources.includes(f)).join(' or '),
  ].filter(Boolean);
  return (
    <LabFileTexts paths={sources.map((f) => dir + f)}>
      {(texts) => (
        <Code
          tabs={sources.map((f, i) => ({
            label: TAB[f],
            code: texts[i] ?? `// loading labs/${dir}${f}`,
            status: solutionTabStatus(reports, l, f),
            tiers,
          }))}
          caption={
            <>
              Reference solution of lab {l.num}, shown from <code>labs/{dir}</code>. The status by the tabs is
              computed from this machine's reports.{absent.length ? ` This lab has ${absent.join(', and ')}.` : null}
            </>
          }
        />
      )}
    </LabFileTexts>
  );
}

/** The "Show the reference solution" toggle. Nothing is fetched until it is opened. */
export function ReferenceSolution({
  lab: l,
  tiers,
}: {
  lab: Lab;
  /** Identifier colours for this lab, if they differ from the page's. */
  tiers?: Record<string, Tier>;
}) {
  const kit = useLabs();
  const [open, setOpen] = useState(false);
  if (!l.sources.length)
    return (
      <p className="ui muted" data-solution={l.num}>
        This lab has no kernel.cu, kernel.metal or launch.mm, so there is no reference source to show here.
      </p>
    );
  return (
    <>
      <p className="ui" data-solution={l.num}>
        <button className="quiet" aria-expanded={open} onClick={() => setOpen(!open)}>
          {open ? 'Hide the reference solution' : 'Show the reference solution'}
        </button>{' '}
        <span className="muted">Worth more after you have a version of your own that passes.</span>
      </p>
      {open ? <SolutionCode lab={l} tiers={tiers ?? kit.tiers} /> : null}
    </>
  );
}

// ---------------------------------------------------------------- standard sections
/** The page's opening: the chapter's lead, then how every lab directory is laid out. */
export function LabsIntro({ children }: { children: ReactNode }) {
  return (
    <>
      <p>
        {children} Every lab holds the same three files three times over: <code>starter/</code> is the exercise as
        shipped, <code>work/</code> is yours and starts as a copy of the starter, and <code>solution/</code> is the
        reference. A driver, <code>main.cpp</code>, tests what is in <code>work/</code> against a CPU reference and,
        on a real device, times it. You do not edit the driver.
      </p>
      <p>
        The three files are <code>kernel.cu</code>, <code>kernel.metal</code> and <code>launch.mm</code>. The first is
        plain CUDA and is built three ways from the one source: for CUDA, for HIP through a header of name mappings,
        and for a CPU emulator. The other two are the same kernel and its launch in Metal's shading language and
        Metal's host API.
      </p>
    </>
  );
}

/** The commands a reader needs, with `example` (a lab number of this chapter) in the per-lab ones. */
export function CommandsTable({ example }: { example: string }) {
  const { labs } = useLabs();
  if (!labs.some((l) => l.num === example)) complain(`the example lab ${example} is not one of this page's labs.`);
  // Each command and sentence is one string, so it is laid out as one run of text.
  const run = `python3 labs/pmpp.py run ${example}`;
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
      <table className="data">
        <thead>
          <tr>
            <th>Command, from the repository root</th>
            <th>What it does</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <code>./verify.sh</code>
            </td>
            <td>
              Every lab, every available backend, your work and the reference solutions; writes{' '}
              <code>reports/verify-*.md</code> and refreshes what this page shows. <code>--quick</code> skips the
              benchmarks.
            </td>
          </tr>
          <tr>
            <td>
              <code>python3 labs/pmpp.py doctor</code>
            </td>
            <td>Which backends can be built here, why the others cannot, and what each device reports.</td>
          </tr>
          <tr>
            <td>
              <code>{run}</code>
            </td>
            <td>{`Tests your work for lab ${example} on every backend.`}</td>
          </tr>
          <tr>
            <td>
              <code>{`${run} --bench`}</code>
            </td>
            <td>The same, then the benchmark if the tests pass.</td>
          </tr>
          <tr>
            <td>
              <code>{`${run} --backend emu`}</code>
            </td>
            <td>
              One backend only: <code>emu</code>, <code>metal</code>, <code>cuda</code> or <code>hip</code>.
            </td>
          </tr>
          <tr>
            <td>
              <code>{`${run} --solution`}</code>
            </td>
            <td>Runs the reference solution in place of your work.</td>
          </tr>
          <tr>
            <td>
              <code>{`python3 labs/pmpp.py reset ${example} --yes`}</code>
            </td>
            <td>
              {`Overwrites your work for lab ${example} with the starter. It is the only command here that replaces files in`}{' '}
              <code>work/</code>.
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

/** "Running them": the commands, where this page's results come from, and what its numbers mean. */
export function RunningLabs({
  example,
  alsoShownIn,
}: {
  /** One of this chapter's lab numbers, used in the per-lab commands. */
  example: string;
  /** Another place that shows the same runs, e.g. <>the roofline of <a href=...>section 5.1</a></>. */
  alsoShownIn?: ReactNode;
}) {
  return (
    <>
      <h2>Running them</h2>
      <p>
        One command builds, tests and benchmarks everything on every backend the machine has. The others are for the
        loop you will actually be in: edit, run one lab, read the failures.
      </p>
      <CommandsTable example={example} />
      <p>
        This page reads <code>reports/runs/</code>. Both <code>./verify.sh</code> and <code>pmpp.py run</code> write
        there, so what you just ran shows up here{alsoShownIn ? <>, and in {alsoShownIn},</> : null} when you reload
        the page. Only <code>./verify.sh</code> also writes the report file you can keep or share.
      </p>
      <Aside>
        <p>
          <b>What the numbers here mean.</b> Every figure on this page was measured on the machine the app is running
          on, in the run whose date is shown. None is an expected value. A different device, driver or run gives
          different numbers, and that difference is what the labs are for.
        </p>
      </Aside>
    </>
  );
}

/** "The four backends": what a pass on each tells you, and what the emulator is and is not. */
export function BackendsAndEmulator() {
  return (
    <>
      <h2>The four backends</h2>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
        <table className="data">
          <thead>
            <tr>
              <th>Backend</th>
              <th>Builds on</th>
              <th>What a pass there tells you</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>
                <code>metal</code>
              </td>
              <td>macOS</td>
              <td>The Metal kernel compiled, ran on that Mac's GPU and matched the CPU reference.</td>
            </tr>
            <tr>
              <td>
                <code>cuda</code>
              </td>
              <td>Linux with the CUDA toolkit</td>
              <td>
                The <code>.cu</code> kernel compiled, ran on that NVIDIA GPU and matched.
              </td>
            </tr>
            <tr>
              <td>
                <code>hip</code>
              </td>
              <td>Linux with ROCm</td>
              <td>
                The same <code>.cu</code> source, compiled as HIP, ran on that AMD GPU and matched.
              </td>
            </tr>
            <tr>
              <td>
                <code>emu</code>
              </td>
              <td>anywhere</td>
              <td>
                The same <code>.cu</code> source has correct logic on a CPU emulator. Nothing about a GPU.
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        The emulator exists so that a <code>.cu</code> kernel can be run and tested on a machine with no CUDA or HIP
        device. It is not a GPU and not an implementation of CUDA. It reproduces the parts of the model that decide
        whether a kernel is <em>correct</em>: block and thread indices, a private copy of each automatic variable per
        thread, <code>__shared__</code> arrays visible to the whole block, <code>__syncthreads()</code> as a real
        barrier, and dynamically sized shared memory.
      </p>
      <p>
        Two properties make it a good first stop when something is wrong. It runs each test under three fixed thread
        orders (in order, reversed, shuffled), so a missing barrier gives a wrong answer every time, where a GPU
        might give one now and then. And it is built with AddressSanitizer, so a read or write outside an array stops
        the run with a report that names the source line.
      </p>
      <p>
        It also models warps as groups of 32 threads in index order, with the warp intrinsics (shuffles, votes,{' '}
        <code>__syncwarp</code>), and fills every block's shared memory with <code>0xCD</code> bytes, so reading a
        slot nobody wrote shows up every time, where a GPU promises nothing about its contents.
      </p>
      <p>
        What it cannot tell you: anything about time, bandwidth, caches, coalescing, bank conflicts, divergence cost
        or occupancy, and nothing about lockstep execution inside a warp. Only one emulated thread runs at a time,
        switching at barriers, so it is not real concurrency (the <code>race</code> flavour runs threads at once under
        ThreadSanitizer for that). Host and device memory are the same memory, so dereferencing a device pointer on
        the host goes unnoticed. It is never benchmarked.
      </p>
    </>
  );
}

/**
 * How far the course author has checked the CUDA and HIP paths, and how a reader can help.
 * This restates PROGRESS.md: change it here, once, when a reports/verify-*.md from a CUDA or
 * HIP machine changes that record.
 */
export function CudaHipCaveat() {
  return (
    <>
      <p>
        A standing caveat from the course author's own record (<code>PROGRESS.md</code>): the author has run these
        solutions on Metal and on the emulator only. The CUDA sources are <b>front-end-checked</b>, meaning clang's
        CUDA mode parsed them against stub declarations and nothing was executed, and they are <b>unverified</b> on
        any NVIDIA device. The HIP path, including its header of name mappings, has never been compiled:{' '}
        <b>unverified</b>. A "verified on CUDA" or "verified on HIP" in the table above can therefore only ever come
        from your own machine.
      </p>
      <p>
        You can help close that gap. On a machine with a CUDA or ROCm toolkit, run <code>./verify.sh</code> and bring
        back the report it writes, <code>reports/verify-&lt;time&gt;-&lt;os&gt;-&lt;backends&gt;.md</code>. If a
        reference solution fails there, the report's failure details are what is needed to fix it.
      </p>
    </>
  );
}

/** "Verification status of the reference solutions": the vocabulary, the table, the caveat. */
export function VerificationStatus() {
  return (
    <>
      <h2>Verification status of the reference solutions</h2>
      <p>
        A lab can only teach you something if its reference solution is right, so here is exactly how far each has
        been checked <em>on this machine</em>. Each phrase has one meaning. <b>verified on Metal</b> (or CUDA, or
        HIP) means compiled and run on that device here, with results matching a CPU reference.{' '}
        <b>emulator-checked</b> means the logic passed on the CPU emulator and nothing more.{' '}
        <b>no run on this machine</b> means there is no evidence here either way, and{' '}
        <b>unverified: failed on this machine</b> means a run here did not pass. When a lab on this page is also built
        for the race emulator (the emulator with real threads under ThreadSanitizer), a <b>race</b> column shows it.
      </p>
      <SolutionStatusTable />
      <CudaHipCaveat />
    </>
  );
}
