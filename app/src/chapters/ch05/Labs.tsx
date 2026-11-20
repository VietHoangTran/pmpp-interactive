import { useState, type ReactNode } from 'react';
import { Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Source } from '../../components/notes';
import { Code, MATMUL_TIERS, useLabFile } from '../../components/Code';
import { benchRows, fmt, labStatus, num, realBackends, type BenchRow, type Reports } from '../../lib/reports';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';

// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

// ---------------------------------------------------------------- shared vocabulary
const BACKENDS = ['metal', 'emu', 'cuda', 'hip'] as const;
type Backend = (typeof BACKENDS)[number];
const NAME: Record<Backend, string> = { metal: 'Metal', emu: 'emulator', cuda: 'CUDA', hip: 'HIP' };

const LABS = [
  { num: '5.1', id: 'ch05/lab1_roofline', dir: 'lab1_roofline', name: 'roofline' },
  { num: '5.2', id: 'ch05/lab2_tiled', dir: 'lab2_tiled', name: 'tiled' },
  { num: '5.3', id: 'ch05/lab3_gemm', dir: 'lab3_gemm', name: 'gemm' },
  { num: '5.4', id: 'ch05/lab4_racehunt', dir: 'lab4_racehunt', name: 'race hunt' },
  { num: '5.5', id: 'ch05/lab5_dynamic', dir: 'lab5_dynamic', name: 'run-time tile' },
] as const;
type Lab = (typeof LABS)[number];
const lab = (n: Lab['num']) => LABS.find((l) => l.num === n) as Lab;

interface Shared {
  reports: Reports;
  loading: boolean;
  reload: () => void;
}

/** Status of a reference solution on one backend, in the project's fixed vocabulary. */
function solutionStatus(reports: Reports, b: Backend, id: string): { text: string; tone: 'ok' | 'fail' | 'none' } {
  const s = labStatus(reports[b], id, 'solution');
  if (s === 'pass') return { text: b === 'emu' ? 'emulator-checked' : `verified on ${NAME[b]}`, tone: 'ok' };
  if (s === 'fail') return { text: 'unverified: failed on this machine', tone: 'fail' };
  return { text: 'no run on this machine', tone: 'none' };
}

function SolutionTable({ reports, loading }: Shared) {
  const runs = BACKENDS.filter((b) => reports[b]);
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
      <table className="data" data-table="solution-status">
        <thead>
          <tr>
            <th>Reference solution</th>
            {BACKENDS.map((b) => (
              <th key={b}>{b === 'emu' ? 'emu' : NAME[b]}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {LABS.map((l) => (
            <tr key={l.id}>
              <td>
                Lab {l.num}, {l.name}
              </td>
              {BACKENDS.map((b) => {
                const s = solutionStatus(reports, b, l.id);
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
              '.'
            : 'No runs on this machine yet. Run ./verify.sh in the repository root, then reload this page.'}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- results from the reports
interface BenchSet {
  backend: string;
  device: string;
  date: string;
  variant: 'work' | 'solution';
  rows: BenchRow[];
}

/** Benchmark rows per real backend: the reader's own kernel first when its tests passed, then the reference. */
function benchSets(reports: Reports, id: string): BenchSet[] {
  return realBackends(reports).flatMap((b) => {
    const run = reports[b];
    const base = { backend: b, device: String(run.device.name ?? b), date: run.generated.slice(0, 10) };
    const out: BenchSet[] = [];
    const mine = labStatus(run, id, 'work') === 'pass' ? benchRows(run, id, 'work') : [];
    if (mine.length) out.push({ ...base, variant: 'work', rows: mine });
    const ref = benchRows(run, id, 'solution');
    if (ref.length) out.push({ ...base, variant: 'solution', rows: ref });
    return out;
  });
}
/** One set per backend: the reader's own if there is one, else the reference solution's. */
function preferred(reports: Reports, id: string): BenchSet[] {
  const seen = new Set<string>();
  return benchSets(reports, id).filter((s) => (seen.has(s.backend) ? false : (seen.add(s.backend), true)));
}
const whose = (s: BenchSet) =>
  s.variant === 'work' ? 'your kernel' : 'the reference solution (no passing benchmark of your own kernel yet)';

interface Measured {
  backend: string;
  device: string;
  date: string;
  whose: string;
  value: number;
  detail: string;
}

function measuredLab1(reports: Reports): Measured[] {
  return preferred(reports, lab('5.1').id).flatMap((s) => {
    const r = s.rows.find((x) => x.name === 'roofline');
    const naive = num(r?.naive_gflops);
    const ceiling = num(r?.naive_model_ceiling_gflops);
    if (!naive || !ceiling) return [];
    return [
      {
        ...s,
        whose: whose(s),
        value: naive / ceiling,
        detail: `${fmt(naive)} GFLOP/s against a model ceiling of ${fmt(ceiling)} GFLOP/s`,
      },
    ];
  });
}
function measuredLab2(reports: Reports): Measured[] {
  return preferred(reports, lab('5.2').id).flatMap((s) => {
    const r = s.rows.find((x) => x.name === 'tiled' && num(x.tile) === 16);
    const v = num(r?.speedup_vs_naive);
    if (!r || !v) return [];
    return [{ ...s, whose: whose(s), value: v, detail: `${fmt(num(r.gflops) ?? NaN)} GFLOP/s tiled, width ${String(r.width)}` }];
  });
}
function measuredLab3(reports: Reports): Measured[] {
  return preferred(reports, lab('5.3').id).flatMap((s) => {
    const at = (j: number) => num(s.rows.find((x) => x.name === 'linear_layer' && num(x.batch) === j)?.gflops);
    const one = at(1);
    const many = at(256);
    if (!one || !many) return [];
    return [{ ...s, whose: whose(s), value: many / one, detail: `${fmt(many)} GFLOP/s at batch 256, ${fmt(one)} at batch 1` }];
  });
}
function measuredLab5(reports: Reports): Measured[] {
  const fixed = preferred(reports, lab('5.2').id);
  return preferred(reports, lab('5.5').id).flatMap((s) => {
    const f = fixed.find((x) => x.backend === s.backend);
    const dyn = s.rows.find((x) => x.name === 'tiled_dynamic' && num(x.tile) === 16);
    const fix = f?.rows.find((x) => x.name === 'tiled' && num(x.tile) === 16);
    const a = num(dyn?.gflops);
    const b = num(fix?.gflops);
    if (!f || !dyn || !fix || !a || !b) return [];
    return [
      {
        ...s,
        whose: `lab 5.5 from ${whose(s)}, lab 5.2 from ${whose(f)}`,
        value: a / b,
        detail: `${fmt(a)} GFLOP/s at width ${String(dyn.width)} against ${fmt(b)} GFLOP/s at width ${String(fix.width)}`,
      },
    ];
  });
}

/** One plain sentence comparing a committed prediction with a measurement. */
function compare(predicted: number, measured: number): string {
  if (!(predicted > 0)) return 'Your prediction was not a positive ratio, so there is no factor to compare.';
  const r = predicted / measured;
  if (Math.abs(r - 1) <= 0.1) return 'Your prediction was within 10% of the measurement.';
  return r < 1
    ? `The measurement is ${fmt(1 / r)} times your prediction.`
    : `Your prediction is ${fmt(r)} times the measurement.`;
}

function Reveal({
  committed,
  measured,
  unit,
  reading,
  missing,
  shared,
}: {
  committed: number;
  measured: Measured[];
  unit: string;
  /** How to read the number, without saying what it should be. */
  reading: ReactNode;
  /** What to run when there is no measurement. */
  missing?: ReactNode;
  shared: Shared;
}) {
  return (
    <>
      {measured.length ? (
        measured.map((m) => (
          <p key={m.backend} data-measured={fmt(m.value)} data-backend={m.backend}>
            Measured on {m.device} ({m.backend}) with {m.whose}, in the run of {m.date}: <b>{fmt(m.value)}</b> {unit}{' '}
            ({m.detail}). {compare(committed, m.value)}
          </p>
        ))
      ) : shared.loading ? (
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
          <button className="quiet" onClick={shared.reload}>
            Reload results
          </button>
        </p>
      )}
      <p className="muted">{reading}</p>
    </>
  );
}

/** Pass/fail of the reader's own work on every backend. */
function WorkStatus({ id, reports, loading }: { id: string } & Shared) {
  return (
    <div className="row" data-work={id} style={{ marginBottom: '0.9rem' }}>
      <b>Your work</b>
      {BACKENDS.map((b) => {
        const run = reports[b];
        const rs = run?.results.filter((x) => x.lab === id && x.variant === 'work') ?? [];
        const s = labStatus(run, id, 'work');
        const total = rs.reduce((n, x) => n + x.tests.length, 0);
        const passed = rs.reduce((n, x) => n + x.tests.filter((t) => t.pass).length, 0);
        let body: ReactNode;
        if (loading) body = <span className="muted">reading reports</span>;
        else if (!run) body = <span className="muted">no run on this machine</span>;
        else if (rs.some((x) => !x.built)) body = <span className="chip fail">does not build</span>;
        else if (s === 'none')
          body = (
            <span className="muted" title="No result for this lab in the last ./verify.sh run on this machine">
              not started
            </span>
          );
        else if (s === 'pass') body = <span className="chip ok">passes, {passed} of {total} tests</span>;
        else {
          // Exit codes other than 0 and 1 mean the run stopped early (a crash, or AddressSanitizer aborting it).
          const aborted = rs.some((x) => x.test_rc != null && x.test_rc !== 0 && x.test_rc !== 1);
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

interface Col {
  head: string;
  num?: boolean;
  cell: (r: BenchRow) => ReactNode;
}
const n = (key: string, digits = 3) => (r: BenchRow) => {
  const v = num(r[key]);
  return v == null ? '' : fmt(v, digits);
};
const ms = (r: BenchRow) => {
  const v = num(r.seconds);
  return v == null ? '' : fmt(v * 1000);
};

function LabResults({
  l,
  cols,
  filter,
  noBench,
  shared,
}: {
  l: Lab;
  cols?: Col[];
  filter?: (r: BenchRow) => boolean;
  /** For a lab without a benchmark: what counts as the result. */
  noBench?: ReactNode;
  shared: Shared;
}) {
  const sets = cols ? benchSets(shared.reports, l.id) : [];
  const any = sets.some((s) => s.rows.some((r) => !filter || filter(r)));
  return (
    <Bench
      title={`Lab ${l.num} on this machine`}
      task={
        <button className="quiet" onClick={shared.reload}>
          Reload results
        </button>
      }
      foot={
        noBench
          ? "From this machine's last ./verify.sh run."
          : "From this machine's last ./verify.sh run. Measurements, not expected values: another device, or another run, gives other numbers."
      }
    >
      <WorkStatus id={l.id} {...shared} />
      {noBench ? (
        <p className="muted" style={{ margin: 0 }}>
          {noBench}
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
          {shared.loading ? (
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

/** The real reference files, hidden until asked for. */
function SolutionCode({ l, reports }: { l: Lab; reports: Reports }) {
  const cu = useLabFile(`ch05/${l.dir}/solution/kernel.cu`);
  const metal = useLabFile(`ch05/${l.dir}/solution/kernel.metal`);
  const host = useLabFile(`ch05/${l.dir}/solution/launch.mm`);
  const st = (b: Backend) => labStatus(reports[b], l.id, 'solution');
  // kernel.cu is one source for CUDA, HIP and the emulator.
  const got = [
    st('cuda') === 'pass' ? 'verified on CUDA' : '',
    st('hip') === 'pass' ? 'verified on HIP' : '',
    st('emu') === 'pass' ? 'emulator-checked' : '',
  ].filter(Boolean);
  const lacking = (['cuda', 'hip'] as Backend[]).filter((b) => st(b) !== 'pass').map((b) => NAME[b]);
  const cuStatus = got.length
    ? got.join(', ') + (lacking.length ? `; unverified on ${lacking.join(' and ')}` : '')
    : 'unverified (no run on this machine)';
  const metalStatus =
    st('metal') === 'pass'
      ? 'verified on Metal'
      : st('metal') === 'fail'
        ? 'unverified (failed on this machine)'
        : 'unverified (no run on this machine)';
  const path = (f: string) => `labs/ch05/${l.dir}/solution/${f}`;
  return (
    <Code
      tabs={[
        { label: 'CUDA', code: cu ?? `// loading ${path('kernel.cu')}`, status: cuStatus, tiers: MATMUL_TIERS },
        { label: 'Metal kernel', code: metal ?? `// loading ${path('kernel.metal')}`, status: metalStatus, tiers: MATMUL_TIERS },
        { label: 'Metal host', code: host ?? `// loading ${path('launch.mm')}`, status: metalStatus, tiers: MATMUL_TIERS },
      ]}
      caption={
        <>
          Reference solution of lab {l.num}, shown from <code>labs/ch05/{l.dir}/solution/</code>. The status at the top
          right is computed from this machine's reports.
        </>
      }
    />
  );
}

function Solution({ l, reports }: { l: Lab; reports: Reports }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <p className="ui" data-solution={l.num}>
        <button className="quiet" aria-expanded={open} onClick={() => setOpen(!open)}>
          {open ? 'Hide the reference solution' : 'Show the reference solution'}
        </button>{' '}
        <span className="muted">Worth more after you have a version of your own that passes.</span>
      </p>
      {open ? <SolutionCode l={l} reports={reports} /> : null}
    </>
  );
}

function Files({ l, files }: { l: Lab; files: [string, ReactNode][] }) {
  return (
    <ul>
      {files.map(([f, what]) => (
        <li key={f}>
          <code>
            labs/ch05/{l.dir}/work/{f}
          </code>
          : {what}
        </li>
      ))}
    </ul>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  const shared: Shared = useReports();
  const { reports } = shared;
  const l1 = lab('5.1');
  const l2 = lab('5.2');
  const l3 = lab('5.3');
  const l4 = lab('5.4');
  const l5 = lab('5.5');
  return (
    <>
      <p>
        Five labs, each in its own directory under <code>labs/ch05/</code>. The first four follow the chapter: measure
        your device, tile the kernel, generalise it, debug one. The fifth makes the tile width a run-time choice, as
        section 5.6 ends. Every lab holds the same three files three times over: <code>starter/</code> is the exercise
        as shipped, <code>work/</code> is yours and starts as a copy of the starter, and <code>solution/</code> is the
        reference. A driver, <code>main.cpp</code>, tests what is in <code>work/</code> against a CPU reference and,
        on a real device, times it. You do not edit the driver.
      </p>
      <p>
        The three files are <code>kernel.cu</code>, <code>kernel.metal</code> and <code>launch.mm</code>. The first is
        plain CUDA and is built three ways from the one source: for CUDA, for HIP through a header of name mappings,
        and for a CPU emulator. The other two are the same kernel and its launch in Metal's shading language and
        Metal's host API.
      </p>

      <h2>Running them</h2>
      <p>
        One command builds, tests and benchmarks everything on every backend the machine has. The others are for the
        loop you will actually be in: edit, run one lab, read the failures.
      </p>
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
                <code>python3 labs/pmpp.py run 5.2</code>
              </td>
              <td>Tests your work for lab 5.2 on every backend.</td>
            </tr>
            <tr>
              <td>
                <code>python3 labs/pmpp.py run 5.2 --bench</code>
              </td>
              <td>The same, then the benchmark if the tests pass.</td>
            </tr>
            <tr>
              <td>
                <code>python3 labs/pmpp.py run 5.2 --backend emu</code>
              </td>
              <td>
                One backend only: <code>emu</code>, <code>metal</code>, <code>cuda</code> or <code>hip</code>.
              </td>
            </tr>
            <tr>
              <td>
                <code>python3 labs/pmpp.py run 5.2 --solution</code>
              </td>
              <td>Runs the reference solution in place of your work.</td>
            </tr>
            <tr>
              <td>
                <code>python3 labs/pmpp.py reset 5.2 --yes</code>
              </td>
              <td>
                Overwrites your work for lab 5.2 with the starter. It is the only command here that replaces files in{' '}
                <code>work/</code>.
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p>
        This page reads <code>reports/runs/</code>. Both <code>./verify.sh</code> and <code>pmpp.py run</code> write
        there, so what you just ran shows up here, and in the roofline of{' '}
        <a href={href('ch05', '5.1')}>section 5.1</a>, when you reload the page. Only <code>./verify.sh</code> also
        writes the report file you can keep or share.
      </p>
      <Aside>
        <p>
          <b>What the numbers here mean.</b> Every figure on this page was measured on the machine the app is running
          on, in the run whose date is shown. None is an expected value. A different device, driver or run gives
          different numbers, and that difference is what the labs are for.
        </p>
      </Aside>

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
        What it cannot tell you: anything about time, bandwidth, caches, occupancy, warps or coalescing. Only one
        emulated thread runs at a time, switching at barriers, so it is not real concurrency. Host and device memory
        are the same memory, so dereferencing a device pointer on the host goes unnoticed. And blocks run one after
        another, so shared memory still holds what the previous block left, where a GPU promises nothing about its
        contents. It is never benchmarked.
      </p>

      <h2>Verification status of the reference solutions</h2>
      <p>
        A lab can only teach you something if its reference solution is right, so here is exactly how far each has
        been checked <em>on this machine</em>. Each phrase has one meaning. <b>verified on Metal</b> (or CUDA, or
        HIP) means compiled and run on that device here, with results matching a CPU reference.{' '}
        <b>emulator-checked</b> means the logic passed on the CPU emulator and nothing more.{' '}
        <b>no run on this machine</b> means there is no evidence here either way.
      </p>
      <SolutionTable {...shared} />
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

      {/* ------------------------------------------------------------ 5.1 */}
      <h2>Lab 5.1: the roofline of your device</h2>
      <p>
        You write the naive kernel of chapter 3 and its launch, from memory if you can, in CUDA and in Metal. Two
        probe kernels are provided: a vector addition, whose speed is set almost entirely by memory bandwidth, and a
        chain of multiply-adds held in registers, which hardly touches memory. Together they measure the two numbers
        of your device's roofline, and the benchmark then places your kernel against the ceiling that the 0.25 FLOP/B
        model gives it. This is the measurement the whole chapter argues from, taken on your hardware and not the
        book's.
      </p>
      <Source section="5.1" pages="93–97" figs="the kernel is Fig. 3.11, p. 61" />
      <Files
        l={l1}
        files={[
          ['kernel.cu', <>the body of <code>matmulKernel</code> and the launch in <code>launchMatmul</code></>],
          ['kernel.metal', <>the body of <code>matmul</code></>],
          ['launch.mm', <>buffer bindings and the dispatch in <code>encodeMatmul</code></>],
        ]}
      />
      <p>
        Run <code>python3 labs/pmpp.py run 5.1</code>, and add <code>--bench</code> once it passes. The tests check
        the two probes first, then your kernel against a CPU reference, within a tolerance, at widths 1, 5, 16, 33,
        100, 257 and 512 on a device and a shorter list on the emulator. Widths that are not a multiple of any block
        size are there on purpose.
      </p>
      <Predict
        id="ch05/labs/p-lab1-ceiling"
        prompt={
          <>
            Before you run the benchmark. The model gives the naive kernel a ceiling of 0.25 FLOP/B times your
            device's measured bandwidth. Your kernel's measured GFLOP/s will be what multiple of that ceiling? (1
            means exactly at it.)
          </>
        }
        unit="× the model ceiling"
      >
        {(v) => (
          <Reveal
            committed={v}
            measured={measuredLab1(reports)}
            unit="× the model ceiling"
            shared={shared}
            reading={
              <>
                Above 1, the kernel beat a ceiling that charges every load to global memory, which can only mean some
                loads were served from somewhere faster; the book notes exactly this about caches (§5.1, p. 97). Below
                1, the kernel has room left under the roof.
              </>
            }
          />
        )}
      </Predict>
      <LabResults
        l={l1}
        shared={shared}
        filter={(r) => r.name === 'roofline'}
        cols={[
          { head: 'Bandwidth probe, GB/s', num: true, cell: n('bandwidth_gb_per_s') },
          { head: 'Arithmetic probe, GFLOP/s', num: true, cell: n('compute_gflops') },
          { head: 'Threshold, FLOP/B', num: true, cell: n('threshold_flop_per_byte') },
          { head: 'Naive matmul, GFLOP/s', num: true, cell: n('naive_gflops') },
          { head: 'at width', num: true, cell: (r) => String(r.naive_width ?? '') },
          { head: 'Model ceiling, GFLOP/s', num: true, cell: n('naive_model_ceiling_gflops') },
        ]}
      />
      <Solution l={l1} reports={reports} />

      {/* ------------------------------------------------------------ 5.2 */}
      <h2>Lab 5.2: the tiled kernel</h2>
      <p>
        You write the tiled kernel of section 5.4: two shared tiles, and for every phase a load, a barrier, a partial
        dot product and a second barrier. The contract is the book's, square matrices whose width is a multiple of
        the tile width. The build defines <code>TILE_WIDTH</code> and compiles your file for several values, so
        nothing in your code may assume 16. The benchmark then answers the question the chapter keeps deferring: how
        much of the model's factor of T your device turns into speed.
      </p>
      <Source section="5.4" pages="108–112" figs="Fig. 5.9" />
      <Files
        l={l2}
        files={[
          ['kernel.cu', <><code>tiledMatmulKernel</code> and <code>launchTiledMatmul</code></>],
          ['kernel.metal', <><code>tiled_matmul</code>, the same steps in Metal's vocabulary</>],
          ['launch.mm', <>the dispatch at the end of <code>encodeTiledMatmul</code></>],
        ]}
      />
      <p>
        Run <code>python3 labs/pmpp.py run 5.2</code>. The tests compare your kernel with a CPU reference at tile
        width 16 for widths from 16 to 528 on a device (Metal also at tile width 8), and at 16, 32 and 64 on the
        emulator under all three thread orders. With <code>--bench</code>, a provided naive kernel and your tiled one
        are timed on the same matrices: on Metal at T = 4, 8, 16 and 32, on CUDA and HIP at the tile widths the build
        compiled (16, and also 8 and 32).
      </p>
      <Predict
        id="ch05/labs/p-lab2-speedup"
        prompt={
          <>
            Before you run the benchmark. At T = 16 the traffic model says the tiled kernel asks global memory for
            one sixteenth of what the naive kernel asks. What speed-up over the naive kernel will you measure on
            your device at T = 16?
          </>
        }
        unit="× the naive kernel"
      >
        {(v) => (
          <Reveal
            committed={v}
            measured={measuredLab2(reports)}
            unit="× the naive kernel"
            shared={shared}
            reading={
              <>
                The factor of 16 is a statement about what the kernel asks of memory. The speed-up is what the device
                made of it, which depends on how much of the naive kernel's traffic its caches were already
                absorbing. Compare the other tile widths in the table before you explain the number.
              </>
            }
          />
        )}
      </Predict>
      <LabResults
        l={l2}
        shared={shared}
        filter={(r) => r.name === 'naive' || r.name === 'tiled'}
        cols={[
          { head: 'Variant', cell: (r) => (r.name === 'tiled' ? `tiled, T = ${String(r.tile)}` : 'naive (provided)') },
          { head: 'Width', num: true, cell: (r) => String(r.width ?? '') },
          { head: 'Time, ms', num: true, cell: ms },
          { head: 'GFLOP/s', num: true, cell: n('gflops') },
          { head: 'Speed-up over naive', num: true, cell: n('speedup_vs_naive') },
        ]}
      />
      <Solution l={l2} reports={reports} />

      {/* ------------------------------------------------------------ 5.3 */}
      <h2>Lab 5.3: a general GEMM</h2>
      <p>
        You turn your lab 5.2 kernel into <code>C[j×l] = A[j×k] × B[k×l]</code> for any sizes. Three things change.{' '}
        <code>Width</code> meant three different things, and each use becomes j, k or l. Sizes are no longer multiples
        of the tile, so every memory access needs its own bounds check and the number of phases rounds up. And the
        grid is no longer square.
      </p>
      <Source section="5.5" pages="112–115" figs="Fig. 5.13" />
      <p>
        This is the kernel behind every linear layer, with j the batch size, k the inputs and l the outputs. The book
        gives rectangular matrices one paragraph and leaves the kernel as an exercise (pp. 114–115). Here it is the
        point: it is the first milestone of the through-line project to be built, M2 in the plan, and the reason the
        benchmark times a layer and not a square matrix.
      </p>
      <Beyond>
        <p>
          The through-line project, <code>nanoinfer</code>, is ours: one small GPU inference path for a decoder-only
          transformer, grown a milestone at a time.
        </p>
      </Beyond>
      <Files
        l={l3}
        files={[
          ['kernel.cu', <><code>gemmKernel</code> and <code>launchGemm</code>, both empty in the starter</>],
          ['kernel.metal', <>the body of <code>gemm</code></>],
          ['launch.mm', <>the dispatch at the end of <code>encodeGemm</code></>],
        ]}
      />
      <p>
        Run <code>python3 labs/pmpp.py run 5.3</code>. The tests are shapes chosen to break a kernel that still
        thinks in squares: the book's 3×3 example with a wider tile, a single element, exactly one tile, a
        rectangular case where everything is a multiple of the tile, one where nothing is, a single row (one token
        through a linear layer), an outer product, and on a device two larger awkward ones. The output buffer is
        filled with a marker value first, so an element your kernel never writes is caught. With{' '}
        <code>--bench</code>, a layer of 1024 inputs and 4096 outputs is timed at batch sizes from 1 to 1024.
      </p>
      <Predict
        id="ch05/labs/p-lab3-batch"
        prompt={
          <>
            Before you run the benchmark. The same kernel, the same 1024 → 4096 layer: its GFLOP/s at batch 256
            divided by its GFLOP/s at batch 1 will be what?
          </>
        }
        unit="× the batch-1 throughput"
      >
        {(v) => (
          <Reveal
            committed={v}
            measured={measuredLab3(reports)}
            unit="× the batch-1 throughput"
            shared={shared}
            reading={
              <>
                Section 5.1 counted why throughput should rise with the batch: the weights are fetched once and serve
                every row. The measured ratio is how much of that this kernel and your device deliver. Look at where
                the rise stops in the table, and at rows per second, which is the figure a server cares about.
              </>
            }
          />
        )}
      </Predict>
      <LabResults
        l={l3}
        shared={shared}
        filter={(r) => r.name === 'linear_layer'}
        cols={[
          { head: 'Batch', num: true, cell: (r) => String(r.batch ?? '') },
          { head: 'Time, ms', num: true, cell: ms },
          { head: 'GFLOP/s', num: true, cell: n('gflops') },
          { head: 'Rows per second', num: true, cell: n('rows_per_s') },
        ]}
      />
      <Solution l={l3} reports={reports} />

      {/* ------------------------------------------------------------ 5.4 */}
      <h2>Lab 5.4: race hunt</h2>
      <p>
        You are handed a tiled, boundary-checked kernel for square matrices of any width, in CUDA and in Metal, with
        three independent defects planted in it. It compiles, and it can pass when the width is a multiple of the
        tile width. You find and fix all three. These are the mistakes this chapter's kernel invites, and finding
        them in code you did not write is the skill you will need for your own.
      </p>
      <Source section="5.5" pages="115" figs="after Fig. 5.13" />
      <Files
        l={l4}
        files={[
          ['kernel.cu', <><code>tiledMatmulKernel</code>, with the three defects</>],
          ['kernel.metal', <><code>tiled_matmul</code>, with the same three</>],
          ['launch.mm', <>provided and correct; nothing to change</>],
        ]}
      />
      <p>The method matters more than the fixes:</p>
      <ol>
        <li>
          Run on the emulator first: <code>python3 labs/pmpp.py run 5.4 --backend emu</code>. Its thread orders are
          fixed, so a defect that depends on timing fails the same way every time. On a GPU the same defect may pass
          on some runs.
        </li>
        <li>
          If the run aborts, read the AddressSanitizer report at the end of the output. Its first stack frame names
          the kernel function and the line of <code>kernel.cu</code> that touched memory it should not have.
        </li>
        <li>
          Fix one defect at a time and run again. Note which widths fail and under which thread orders before and
          after each fix: that pattern is your evidence for what the next defect is.
        </li>
        <li>
          Make the same three fixes in <code>kernel.metal</code> and run on the device.
        </li>
      </ol>
      <p>
        The tests use widths 16, 32, 3, 24, 40 and 50, and 200 and 333 as well on a device, each run twice on fresh
        data and compared with a CPU reference. There is no benchmark.
      </p>
      <LabResults
        l={l4}
        shared={shared}
        noBench="This lab has no benchmark. The result is the line above: every test passing on the emulator and on a device. As shipped, the kernel is meant to fail."
      />
      <Solution l={l4} reports={reports} />

      {/* ------------------------------------------------------------ 5.5 */}
      <h2>Lab 5.5: tile width chosen at run time</h2>
      <p>
        You write the tiled, boundary-checked square kernel once more, with the tile width as an argument and no{' '}
        <code>#define</code>. In CUDA that means one unsized <code>extern __shared__</code> array carved into two
        tiles, with its size in bytes as the third launch parameter. In Metal it means a{' '}
        <code>threadgroup float*</code> kernel parameter bound with <code>[[threadgroup(0)]]</code>, whose size the
        host sets with <code>setThreadgroupMemoryLength:atIndex:</code>. One binary then serves every tile width, and
        the driver picks the largest one your device reports room for.
      </p>
      <Source section="5.6" pages="116–118" figs="Fig. 5.14" />
      <p>
        This is the working counterpart of the book's Fig. 5.14, which does not compile as printed.{' '}
        <a href={href('ch05', '5.6')}>Section 5.6</a> has you find what is wrong with the page; here you write the
        version that is right.
      </p>
      <Files
        l={l5}
        files={[
          ['kernel.cu', <><code>tiledMatmulDynamicKernel</code> and the byte count in <code>launchTiledMatmulDynamic</code></>],
          ['kernel.metal', <>the body of <code>tiled_matmul_dynamic</code>, which receives the tiles as one pointer</>],
          ['launch.mm', <>the threadgroup memory length and the dispatch in <code>encodeTiledMatmulDynamic</code></>],
        ]}
      />
      <p>
        Run <code>python3 labs/pmpp.py run 5.5</code>. The driver first works out the largest power-of-two tile whose
        block fits two limits the device reports, threads per block and shared memory per block, and prints them. It
        then tests tile widths 4, 8 and 16, and that largest one if it is bigger, each at widths T, 2T, 3 and 2T + 5,
        and at 200 and 333 on a device. With <code>--bench</code> it times a width of 2048 at every power-of-two tile
        from 4 up to the largest.
      </p>
      <Predict
        id="ch05/labs/p-lab5-dynamic"
        prompt={
          <>
            Before you run the benchmark. At T = 16, this kernel's GFLOP/s divided by the GFLOP/s of lab 5.2's
            kernel, whose tile width is fixed at compile time, will be what? (1 means the same speed, below 1 means
            the run-time version is slower.)
          </>
        }
        unit="× the compile-time kernel"
      >
        {(v) => (
          <Reveal
            committed={v}
            measured={measuredLab5(reports)}
            unit="× the compile-time kernel"
            shared={shared}
            missing={
              <>
                This comparison needs benchmark rows for lab 5.5 and for lab 5.2 from the same backend. Run{' '}
                <code>./verify.sh</code> without <code>--quick</code> on a machine with a GPU.
              </>
            }
            reading={
              <>
                Same algorithm and tile width. What differs is that the tile width is a value the kernel reads, where
                before it was a constant the compiler saw. The two benchmarks may also use different matrix widths;
                both are shown with the measurement.
              </>
            }
          />
        )}
      </Predict>
      <LabResults
        l={l5}
        shared={shared}
        filter={(r) => r.name === 'tiled_dynamic'}
        cols={[
          { head: 'T', num: true, cell: (r) => String(r.tile ?? '') },
          { head: 'Width', num: true, cell: (r) => String(r.width ?? '') },
          { head: 'Shared bytes per block', num: true, cell: n('shared_bytes_per_block', 6) },
          { head: 'Time, ms', num: true, cell: ms },
          { head: 'GFLOP/s', num: true, cell: n('gflops') },
        ]}
      />
      <Solution l={l5} reports={reports} />
    </>
  );
}
