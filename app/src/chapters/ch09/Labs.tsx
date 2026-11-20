import type { ReactNode } from 'react';
import type { Question } from '../../components/Ask';
import { Predict } from '../../components/Ask';
import { Beyond, Source, Today, type Tier } from '../../components/notes';
import {
  BackendsAndEmulator,
  LabFiles,
  LabPrediction,
  LabResults,
  LabsIntro,
  LabsPage,
  ReferenceSolution,
  Reveal,
  RunningLabs,
  VerificationStatus,
  defineLabs,
  measure,
  msCol,
  numCol,
  useLabs,
  type Lab,
} from '../../components/LabKit';
import { fmt, labStatus, num, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { href } from '../../lib/router';

// The page is built from components/LabKit.tsx; what is here is chapter 9's own content.
// No question on this page: the labs are the exercise. (Predictions are stored, not graded.)
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch09', [
  { num: '9.1', dir: 'lab1_racehunt', name: 'race hunt (optional)' },
  { num: '9.2', dir: 'lab2_privatised', name: 'privatised, coarsened, run-merged' },
  { num: '9.3', dir: 'lab3_tokencounts', name: 'token counts (M5 piece)' },
]);
const [l1, l2, l3] = LABS;

// Identifier colours for the reference solutions. Only names that mean one thing in all
// three files of a lab.
const tiers = (global: string[], shared: string[] = [], reg: string[] = []): Record<string, Tier> => ({
  ...Object.fromEntries(global.map((n) => [n, 'global' as Tier])),
  ...Object.fromEntries(shared.map((n) => [n, 'shared' as Tier])),
  ...Object.fromEntries(reg.map((n) => [n, 'reg' as Tier])),
});
const HUNT_TIERS = tiers(['image', 'bins', 'binsPool', 'binsPriv']);
const SHARED_TIERS = tiers(['image', 'bins'], ['bins_s'], ['bin_r', 'bNext', 'segment']);
const TOKEN_TIERS = tiers(['tokens', 'counts', 'logits', 'newTokens', 'rowTokens', 'rowCounts']);

// ---------------------------------------------------------------- reading the reports
const gps = (v: number) => `${fmt(v)} Gpixel/s`;
const ms = (s: number) => `${fmt(s * 1000)} ms`;
const count = (v: number) => v.toLocaleString('en-US');
const g = (r: BenchRow | undefined) => num(r?.gpixels_per_s);

/** Lab 9.1: the basic kernel, uniform bytes over a constant image (section 9.3's prediction). */
const skewSlowdown = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const uni = g(s.rows.find((r) => r.name === 'basic' && r.data === 'uniform'));
    const con = g(s.rows.find((r) => r.name === 'basic' && r.data === 'constant'));
    if (!uni || !con) return null;
    return { value: uni / con, detail: `${gps(uni)} on uniform bytes, ${gps(con)} on the constant image` };
  });

/** Lab 9.1: on the constant image, the pool kernel over the basic one. */
const poolGap = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const basic = g(s.rows.find((r) => r.name === 'basic' && r.data === 'constant'));
    const pool = g(s.rows.find((r) => r.name === 'pool' && r.data === 'constant'));
    if (!basic || !pool) return null;
    const pu = g(s.rows.find((r) => r.name === 'pool' && r.data === 'uniform'));
    return {
      value: pool / basic,
      detail: `${gps(pool)} for the pool kernel and ${gps(basic)} for the basic kernel on the constant image` + (pu ? `; the pool kernel ran at ${gps(pu)} on uniform bytes` : ''),
    };
  });

const lab2 = (rows: BenchRow[], name: string, data: string, coarse: number) =>
  rows.find((r) => r.name === name && r.data === data && num(r.coarse) === coarse);

/** Lab 9.2: shared (factor 16) over basic, constant image (section 9.4's prediction). */
const sharedSpeedup = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const basic = g(lab2(s.rows, 'basic', 'constant', 1));
    const shared = g(lab2(s.rows, 'shared', 'constant', 16));
    if (!basic || !shared) return null;
    return { value: shared / basic, detail: `${gps(shared)} for the shared-memory kernel at factor 16, ${gps(basic)} for the basic kernel, constant image` };
  });

/** Lab 9.2: interleaved over contiguous at factor 16, uniform bytes (section 9.5's prediction). */
const interleavedGain = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const inter = num(lab2(s.rows, 'shared', 'uniform', 16)?.seconds);
    const contig = num(lab2(s.rows, 'contiguous', 'uniform', 16)?.seconds);
    if (!inter || !contig) return null;
    return { value: contig / inter, detail: `${ms(inter)} interleaved, ${ms(contig)} contiguous, factor 16, uniform bytes` };
  });

/** Lab 9.2: runs over shared at factor 16, on the sky; uniform bytes alongside. */
const runsOnSky = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const runs = g(lab2(s.rows, 'runs', 'sky', 16));
    const shared = g(lab2(s.rows, 'shared', 'sky', 16));
    if (!runs || !shared) return null;
    const ru = g(lab2(s.rows, 'runs', 'uniform', 16));
    const su = g(lab2(s.rows, 'shared', 'uniform', 16));
    return {
      value: runs / shared,
      detail: `${gps(runs)} with run merging and ${gps(shared)} without, factor 16, sky` + (ru && su ? `; on uniform bytes the ratio is ${fmt(ru / su)} (${gps(ru)} against ${gps(su)})` : ''),
    };
  });

const tokenRow = (rows: BenchRow[], name: string, data: string) => rows.find((r) => r.name === name && r.data === data);
const shape = (r: BenchRow) => `batch ${String(r.batch)}, ${count(num(r.len) ?? 0)} tokens per row, vocabulary ${count(num(r.vocab) ?? 0)}`;

/** Lab 9.3: the provided pool over direct atomics, in time. */
const poolOverCount = (reports: Reports) =>
  measure(reports, l3, (s) => {
    const c = tokenRow(s.rows, 'count', 'uniform');
    const p = tokenRow(s.rows, 'count_pool', 'uniform');
    const cs = num(c?.seconds);
    const ps = num(p?.seconds);
    if (!c || !cs || !ps) return null;
    const cz = num(tokenRow(s.rows, 'count', 'zipf')?.seconds);
    const pz = num(tokenRow(s.rows, 'count_pool', 'zipf')?.seconds);
    return {
      value: ps / cs,
      detail:
        `${ms(ps)} for count_pool, ${ms(cs)} for direct atomics, uniform ids, ${shape(c)}` +
        (num(p?.blocks_per_row) ? `, ${String(p!.blocks_per_row)} pool blocks per row` : '') +
        (cz && pz ? `; Zipf-like ids: ${fmt(pz / cz)}` : ''),
    };
  });

/** Lab 9.3: recounting the context over one decode step. */
const recountOverAppend = (reports: Reports) =>
  measure(reports, l3, (s) => {
    const c = tokenRow(s.rows, 'count', 'uniform');
    const cs = num(c?.seconds);
    const as = num(tokenRow(s.rows, 'append', 'uniform')?.seconds);
    if (!c || !cs || !as) return null;
    return { value: cs / as, detail: `${ms(cs)} to count the context, ${ms(as)} for one append step, uniform ids, ${shape(c)}` };
  });

// ---------------------------------------------------------------- the race flavour, which the status tables do not show
type FullResult = RunResult & { tsan?: unknown[] };

function RaceLine({ lab: l }: { lab: Lab }) {
  const { reports, loading } = useLabs();
  const run = reports.race;
  const rs = (run?.results.filter((x) => x.lab === l.id) ?? []) as FullResult[];
  const sol = rs.filter((x) => x.variant === 'solution');
  const work = rs.filter((x) => x.variant === 'work');
  const s = labStatus(run, l.id, 'solution');
  const n = sol.flatMap((x) => x.tests).length;
  const wTests = work.flatMap((x) => x.tests);
  const wRaces = work.reduce((k, x) => k + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0);
  const ws = labStatus(run, l.id, 'work');
  const races = wRaces ? `, ThreadSanitizer reported ${wRaces} race${wRaces === 1 ? '' : 's'}` : '';
  return (
    <p data-race-status={loading ? 'loading' : s} data-race-lab={l.num}>
      {loading ? (
        'Reading reports.'
      ) : !run || (s === 'none' && !work.length) ? (
        <>
          No run of lab {l.num} on the race backend on this machine yet:{' '}
          <code>python3 labs/pmpp.py run {l.num} --backend race</code>.
        </>
      ) : (
        <>
          On the race backend of this machine, the reference solution{' '}
          {s === 'pass'
            ? `passed all ${n} tests with no race reported: emulator-checked under both flavours.`
            : s === 'fail'
              ? 'failed: unverified there.'
              : 'has no run yet.'}{' '}
          Your work there:{' '}
          {!work.length
            ? 'no run yet.'
            : ws === 'none'
              ? `not started (${wTests.filter((t) => t.pass).length} of ${wTests.length} tests pass as shipped${races}).`
              : ws === 'pass'
                ? `passes, ${wTests.length} of ${wTests.length} tests, no race reported.`
                : `not passing: ${wTests.filter((t) => t.pass).length} of ${wTests.length} tests${races}.`}
        </>
      )}
    </p>
  );
}

// ---------------------------------------------------------------- milestone M5
function MilestoneStatus() {
  const { reports, loading } = useLabs();
  const backends = ['metal', 'cuda', 'hip', 'emu'].filter((b) => reports[b]);
  const done = backends.filter((b) => labStatus(reports[b], l3.id, 'work') === 'pass');
  const name = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  return (
    <p data-milestone={loading ? 'loading' : done.join(',') || 'none'}>
      {loading
        ? 'Reading reports.'
        : !backends.length
          ? 'No lab results are recorded on this machine yet.'
          : done.length
            ? `Your own lab 9.3 passes on ${done.map(name).join(' and ')} on this machine: the counting piece of M5 is in place there.`
            : 'Your own lab 9.3 does not pass on any backend of this machine yet.'}
    </p>
  );
}

// ---------------------------------------------------------------- results-table columns
const IMAGE: Record<string, string> = { uniform: 'uniform bytes', constant: 'constant', sky: 'sky' };
const image = (r: BenchRow) => IMAGE[String(r.data)] ?? String(r.data ?? '');
const pixels = { head: 'Pixels', num: true, cell: (r: BenchRow) => (num(r.pixels) != null ? count(num(r.pixels)!) : '') };
const LAB2_NAME: Record<string, string> = {
  basic: 'basic (Fig. 9.6, provided)',
  contiguous: 'contiguous (Fig. 9.12, provided)',
  shared: 'shared (Fig. 9.14)',
  runs: 'runs (Fig. 9.15)',
};
const LAB3_NAME: Record<string, string> = {
  count: 'count',
  count_pool: 'count_pool (provided)',
  append: 'append (one step)',
  penalties: 'penalties',
};
/** "5.62 G tokens/s" for a rate in items per second. */
const perSecond = (v: number, what: string) =>
  v >= 1e9 ? `${fmt(v / 1e9)} G ${what}/s` : v >= 1e6 ? `${fmt(v / 1e6)} M ${what}/s` : v >= 1e3 ? `${fmt(v / 1e3)} k ${what}/s` : `${fmt(v)} ${what}/s`;
const rate = (r: BenchRow) => {
  const t = num(r.tokens_per_s);
  const e = num(r.elements_per_s);
  const text = t != null ? perSecond(t, 'tokens') : e != null ? perSecond(e, 'logits') : '';
  return <span style={{ whiteSpace: 'nowrap' }}>{text}</span>;
};
const nowrap = (t: string) => <span style={{ whiteSpace: 'nowrap' }}>{t}</span>;

const BENCH_MISSING = (lab: string) => (
  <>
    Once lab {lab} passes, run <code>python3 labs/pmpp.py run {lab} --bench</code> on a machine with a GPU, then reload.
    To measure the reference solution first, add <code>--solution</code>.
  </>
);

/** A prediction shared with a section: the same progress id, committed on either page. */
function SharedPrediction({
  id,
  prompt,
  unit,
  read,
  lab,
  reading,
}: {
  id: string;
  prompt: ReactNode;
  unit: string;
  read: (r: Reports) => ReturnType<typeof measure>;
  lab: string;
  reading: ReactNode;
}) {
  return (
    <Predict id={id} prompt={prompt} unit={unit}>
      {(v) => <Reveal committed={v} measure={read} unit={unit} missing={BENCH_MISSING(lab)} reading={reading} />}
    </Predict>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS}>
      <LabsIntro>
        Two labs and an optional third, each in its own directory under <code>labs/ch09/</code>. Lab 9.2 writes the
        chapter's fast kernels, privatised in shared memory, coarsened and run-merged, and times them against the basic
        one on skewed images. Lab 9.3 counts token ids over a whole vocabulary for repetition penalties, the counting
        piece of milestone M5. Lab 9.1 is optional: a hunt for three races in two histogram kernels that compile and
        often give the right counts. Section 9.4's bug hunt covers the same ground in the browser; the lab adds the real
        tools and times the skew prediction of section 9.3.
      </LabsIntro>
      <RunningLabs
        example="9.1"
        alsoShownIn={
          <>
            sections <a href={href('ch09', '9.3')}>9.3</a>, <a href={href('ch09', '9.4')}>9.4</a> and{' '}
            <a href={href('ch09', '9.5')}>9.5</a>
          </>
        }
      />
      <BackendsAndEmulator />
      <p>
        The emulator has a second flavour, <code>race</code>, and all three labs here are built for it. It runs the
        emulated threads concurrently under ThreadSanitizer and reports two threads that touch the same memory with
        nothing ordering them, with both source lines, even in a run whose counts came out right. It is the only backend
        here that can see a missing atomic: <code>emu</code> never interrupts a thread between its read and its write,
        so a plain <code>++bins[b]</code> is always right there. Ask for it with <code>--backend race</code>;{' '}
        <code>python3 labs/pmpp.py doctor</code> says whether your compiler can build it. The tables on this page show the
        first flavour only; a line under each lab gives the race results.
      </p>
      <Today checked="2026-11-20 (notes/today/ch09-facts.md A10, B6, C5, E1)">
        <p>
          NVIDIA's <code>compute-sanitizer --tool racecheck</code> checks shared memory only, so it cannot see a race on
          global bins. AMD documents no race detector, and Metal's Shader Validation lists no race check. ThreadSanitizer
          examines only the schedules that happened.
        </p>
      </Today>
      <p>
        Neither emulator gives scopes or memory orders any effect (<code>labs/README.md</code>), and Metal has no scope on
        a read-modify-write at all. A commit to the public bins written with <code>cuda::thread_scope_block</code> passes
        every test on every backend a Mac can run, and is still a data race under CUDA's memory model, which makes it
        undefined behaviour (notes/today/ch09-facts.md A2, C1). Check every scope by reading it.
      </p>
      <p>
        On HIP, <code>&lt;cuda/atomic&gt;</code> comes from libhipcxx, part of ROCm from 10.1. The course's HIP build
        links only <code>hip::host</code> today (<code>labs/cmake/Backends.cmake</code>), so all three labs may fail to
        find the header there until the shared build file links libhipcxx. Nobody has tried: there is no ROCm machine
        behind this course, and the HIP column below can only be filled by yours.
      </p>
      <Today checked="2026-11-20 (notes/today/infra-facts.md B2; ch09-facts.md B1)">
        <p>
          libhipcxx keeps the CUDA spelling and the scope argument, without a cluster scope.
        </p>
      </Today>
      <VerificationStatus />

      {/* ------------------------------------------------------------ 9.1 */}
      <h2>Lab 9.1 (optional): race hunt, two histogram kernels</h2>
      <p>
        A colleague hands you two 256-bin histograms of an 8-bit image. <code>histogramBasic</code> is Fig. 9.6, one
        thread per pixel. <code>histogramPool</code> is Fig. 9.9: one private histogram per block in a pool in global
        memory, updated with block-scope atomics, then a commit of the non-zero private bins to the public bins. Between
        them they have three independent defects, the same three in the CUDA and the Metal version. Everything compiles.
        Find and fix all three.
      </p>
      <Source section="9.2" pages="203–213" figs="Figs. 9.6, 9.9; the defects are ours" />
      <LabFiles
        lab={l1}
        files={[
          ['kernel.cu', <><code>histogramBasic</code> and <code>histogramPool</code>; the two launchers below them are correct</>],
          ['kernel.metal', <><code>histogram_basic</code> and <code>histogram_pool</code>, with the same three defects</>],
          ['launch.mm', <>correct as shipped; there for completeness</>],
        ]}
      />
      <p>
        The caller zeroes the bins and the pool, which holds one 256-bin histogram per block. The book prints no host code
        and never says who zeroes them; the launchers and that rule are ours. Every case compares all 256 counts with the
        sequential loop of Fig. 9.2, exactly. Images from 1 × 1 to 64 × 64 run everywhere, and 1024 × 1024 and 3000 ×
        1001 on a GPU; each in four kinds (uniform random bytes, every pixel 200, 0 and 255 in long runs, a smooth
        gradient), with blocks of 64 and 128 threads, fewer than the bins, so the commit loop must stride. Each kernel's
        first case is the constant image in 64 blocks of 64 threads, the most contended input there is. On a GPU every
        configuration is launched 20 times on freshly zeroed buffers.
      </p>
      <p>Before you run anything, read the two kernels and write down each defect you can see, and which backend you expect to show it. Then hunt, in this order:</p>
      <ol>
        <li>
          Emulator first: <code>python3 labs/pmpp.py run 9.1 --backend emu</code>. A missing barrier is wrong there
          under at least one of the three thread orders. A missing atomic never is.
        </li>
        <li>
          <code>python3 labs/pmpp.py run 9.1 --backend race</code>. Read the <code>kernel.cu:</code> line numbers in each
          report. One access of a pair can sit inside the emulator's own atomic (<code>pmpp_emu::atomic_rmw</code>):
          that is one of your <code>atomic_ref</code> calls racing with a plain access on the other line. Each pair of
          lines is reported once per run, so read each kernel's first case.
        </li>
        <li>Fix one thing and run again. The failures change as you go.</li>
        <li>
          Then your GPU: <code>python3 labs/pmpp.py run 9.1</code>. Note which defects it showed you and which it hid.
        </li>
      </ol>
      <p>
        Write the atomic fixes as the book does, <code>cuda::atomic_ref&lt;unsigned int, scope&gt;</code> and{' '}
        <code>fetch_add(…, cuda::memory_order_relaxed)</code>, with the scope chosen from the threads that share the
        counter. On Metal the fix changes the parameter's type to <code>device atomic_uint*</code>. And a Metal barrier
        names the memory it orders: decide which flag the pool kernel's barrier needs before you look it up.
      </p>
      <Today checked="2026-11-20 (notes/today/ch09-facts.md A1, A3, C1; ch05-facts.md C2)">
        <p>
          Left out, the scope defaults to system and the order to <code>seq_cst</code>. A plain write from several lanes
          of a warp leaves "which thread performs the final write" undefined. Metal's atomic functions take atomic types
          only.
        </p>
      </Today>
      <RaceLine lab={l1} />
      <p>
        With <code>--bench</code> the driver times both kernels on 2<sup>26</sup> pixels (fewer if the device reports too
        little memory for the image and the pool), 256 threads per block, on uniform bytes and on the constant image, best
        of 5 runs. Commit two predictions first. The book's model allows up to 256 times for the first: atomics on one
        location cannot overlap, and uniform bytes spread them over 256 locations.
      </p>
      <SharedPrediction
        id="ch09/9.3/p-skew-slowdown"
        lab="9.1"
        prompt={
          <>
            The same prediction as in section 9.3: commit it on either page. On <em>your</em> GPU, how many times slower
            will the basic kernel run on a constant image than on uniform random bytes?
          </>
        }
        unit="× slower"
        read={skewSlowdown}
        reading={
          <>
            Two things the model leaves out can make the real figure far smaller than 256. Device-scope atomics run in the
            L2 cache on NVIDIA and AMD, not in DRAM; Apple documents nothing (ch09-facts A4, B2, C4). And a compiler may
            merge one warp's same-address atomics: NVIDIA says so only in a 2017 blog post, "in many cases"; AMD's compiler
            does it only for an address that is the same across the wavefront, which a bin read from the image is not (A9,
            B5).
          </>
        }
      />
      <LabPrediction
        id="ch09/labs/p-lab1-pool-gap"
        prompt={
          <>
            On the constant image, how many times as fast as the basic kernel will the pool kernel run on your GPU?
          </>
        }
        unit="× the basic kernel"
        measure={poolGap}
        missing={BENCH_MISSING('9.1')}
        reading={
          <>
            On the constant image every atomic of the basic kernel joins one queue. The pool splits that queue into one per
            block in flight, and adds a commit of at most 256 bins per block (<a href={href('ch09', '9.4')}>section 9.4</a>
            ). The ratio on uniform bytes, in the results table, says what the split costs when there is no queue to split.
          </>
        }
      />
      <LabResults
        lab={l1}
        cols={[
          { head: 'Histogram', cell: (r) => (r.name === 'basic' ? 'basic (Fig. 9.6)' : r.name === 'pool' ? 'pool (Fig. 9.9)' : r.name) },
          { head: 'Image', cell: image },
          pixels,
          msCol,
          numCol('Gpixel/s', 'gpixels_per_s'),
        ]}
      />
      <p>
        Whatever your GPU showed, a configuration that passed 20 times showed you 20 schedules on one device. A plain
        concurrent increment is a data race, so a lost count is the typical outcome, not a promised one, and a clean run
        proves nothing about the next device.
      </p>
      <ReferenceSolution lab={l1} tiers={HUNT_TIERS} />

      {/* ------------------------------------------------------------ 9.2 */}
      <h2>Lab 9.2: privatised in shared memory, coarsened and run-merged</h2>
      <p>
        You write the chapter's two fast kernels, in CUDA and in Metal. <code>histogramShared</code> is Fig. 9.14: zero
        a private copy of the bins in shared memory with a strided loop, barrier, count <code>coarseFactor</code> pixels
        per thread with interleaved indexing and block-scope atomics, barrier, then add each non-zero private bin to the
        public bins with a device-scope atomic. <code>histogramRuns</code> is Fig. 9.15: the same, plus a run counter
        per thread, so that consecutive equal pixels of one thread cost one atomic. The book's coarsening factor is a
        compile-time macro; here it is a run-time argument, so that one build can be measured at several factors.
      </p>
      <Source section="9.4" pages="211–219" figs="Figs. 9.10, 9.14, 9.15" />
      <LabFiles
        lab={l2}
        files={[
          ['kernel.cu', <><code>histogramShared</code> and <code>histogramRuns</code>; the launchers below them are written for you</>],
          ['kernel.metal', <><code>histogram_shared</code> and <code>histogram_runs</code></>],
          ['launch.mm', <>written for you: 256 threads per threadgroup, or fewer if the pipeline cannot run 256</>],
        ]}
      />
      <p>
        Provided and not edited, in <code>labs/ch09/lab2_privatised/provided/</code>: the benchmark's two comparators,{' '}
        <code>refHistogramBasic</code> (Fig. 9.6) and <code>refHistogramContiguous</code> (Fig. 9.12, shared-memory
        privatisation with contiguous indexing), in CUDA and Metal. The tests run them too. Section{' '}
        <a href={href('ch09', '9.5')}>9.5</a> shows the contiguous one beside yours.
      </p>
      <p>
        On Metal the private bins are a <code>threadgroup atomic_uint</code> array declared in the kernel body, zeroed
        with <code>atomic_store_explicit</code> and read in the commit with <code>atomic_load_explicit</code>. That
        declaration is the form shipping code uses; the specification shows no example of it. It compiled and passed on
        the Mac this lab was built on.
      </p>
      <Today checked="2026-11-20 (notes/today/ch09-facts.md C1, D2)">
        <p>
          If your Metal compiler rejects the array, use a <code>threadgroup atomic_uint* [[threadgroup(0)]]</code>{' '}
          parameter sized on the host with <code>setThreadgroupMemoryLength</code>, as PyTorch's MPS histogram does.
        </p>
      </Today>
      <p>
        Run <code>python3 labs/pmpp.py run 9.2</code>, starting with <code>--backend emu</code> and then{' '}
        <code>--backend race</code>. Both kernels must match the CPU's counts exactly on images from 1 × 1 to 100 × 100
        (on a GPU also 1024 × 1024 and 4096 × 1025), most leaving the last block short of pixels, at coarsening factors
        1, 3, 4 and 16, on uniform bytes, a constant image, a sky (each row one value, changing slowly from row to row)
        and stripes of 8 equal bytes. What each backend can show you:
      </p>
      <ul>
        <li>
          Shared memory nobody zeroed holds <code>0xCD</code> bytes on <code>emu</code> and <code>race</code>, so a
          missing zeroing loop is wrong there every time. A GPU may hand you zeros, and then the tests pass on it.
        </li>
        <li>
          A missing barrier is wrong on <code>emu</code> under at least one thread order, and <code>race</code> names
          the lines involved.
        </li>
        <li>
          <code>race</code> runs a block of 256 threads on its own, so it cannot see two blocks racing in the commit. A
          plain <code>+=</code> there may show on your GPU as lost counts.
        </li>
        <li>
          Two mistakes pass every test. A zeroing or commit loop bounded by <code>blockDim.x</code> gives the right
          counts, because the launchers use 256 threads for 256 bins; it is wrong for any other block size, which lab 9.1
          does test. And contiguous indexing gives the right counts too: which pixels a thread takes changes the speed,
          not the result.
        </li>
      </ul>
      <RaceLine lab={l2} />
      <p>
        With <code>--bench</code> the driver times, on 2<sup>26</sup> pixels (8192 × 8192, so that a sky row is long),
        uniform bytes, the constant image and the sky: the provided basic kernel, the provided contiguous kernel at factor
        16, your <code>shared</code> at factors 1, 4, 16 and 64, and your <code>runs</code> at 16, best of 5 runs each.
        Commit three predictions first.
      </p>
      <SharedPrediction
        id="ch09/9.4/p-shared-speedup"
        lab="9.2"
        prompt={
          <>
            The same prediction as in section 9.4: commit it on either page. With every pixel the same, how many times
            faster than the basic kernel will the shared-memory kernel run on your GPU, at a coarsening factor of 16?
          </>
        }
        unit="× faster"
        read={sharedSpeedup}
        reading={
          <>
            The basic kernel's atomics all queue on one bin in the last-level cache; the shared kernel's queue on one bin
            per block in shared memory, then commit once per block. The factor-1 to factor-64 rows of the table show how
            much of the gain is privatisation and how much is coarsening.
          </>
        }
      />
      <SharedPrediction
        id="ch09/9.5/p-interleaved-gain"
        lab="9.2"
        prompt={
          <>
            The same prediction as in section 9.5: commit it on either page. At a coarsening factor of 16 on uniform
            random bytes, how many times faster than the contiguous kernel will the interleaved one run on your GPU?
          </>
        }
        unit="× faster"
        read={interleavedGain}
        reading={
          <>
            The book's argument is coalescing: in one step the threads of a warp should read neighbouring bytes (p. 216).
            NVIDIA documents that rule as 32-byte transactions per warp, AMD as cache-line requests; Apple documents no
            coalescing rule, so on a Mac this measurement is the only evidence there is (ch09-facts A7, B3, C4). A number
            below 1 means contiguous won on this device.
          </>
        }
      />
      <LabPrediction
        id="ch09/labs/p-lab2-runs-sky"
        prompt={
          <>
            On the sky image at factor 16, how many times as fast as <code>shared</code> will <code>runs</code> be? A
            number below 1 means you expect run merging to cost more than it saves. Decide too what you expect on
            uniform bytes.
          </>
        }
        unit="× shared"
        measure={runsOnSky}
        missing={BENCH_MISSING('9.2')}
        reading={
          <>
            A sky row is 8,192 equal bytes, far longer than the 256 pixels between one thread's consecutive reads, so
            nearly every iteration extends a run. Uniform bytes almost never repeat at that spacing, so there{' '}
            <code>runs</code> pays for its comparison and saves nothing (<a href={href('ch09', '9.6')}>section 9.6</a>).
            The uniform ratio is in the detail above.
          </>
        }
      />
      <LabResults
        lab={l2}
        cols={[
          { head: 'Histogram', cell: (r) => LAB2_NAME[r.name] ?? r.name },
          { head: 'Image', cell: image },
          { head: 'Coarse', num: true, cell: (r) => String(r.coarse ?? '') },
          msCol,
          numCol('Gpixel/s', 'gpixels_per_s'),
        ]}
      />
      <ReferenceSolution lab={l2} tiers={SHARED_TIERS} />

      {/* ------------------------------------------------------------ 9.3 */}
      <h2>Lab 9.3: token counts for repetition penalties (M5)</h2>
      <p>
        A sampler with frequency and presence penalties needs to know, for every request in the batch, how often each
        vocabulary id already occurs in that request's context. That is a histogram whose bins are a whole vocabulary:
        50,257 ids for GPT-2, 151,936 for Qwen3-8B, 201,088 for gpt-oss-20b. You write three kernels over row-major
        int32 token ids and uint32 counts, batch first: <code>countTokens</code> counts each row's tokens into that row's
        counts, <code>appendTokens</code> adds one new token per row (a decode step), and <code>applyPenalties</code>{' '}
        subtracts <code>count × frequencyPenalty + (count &gt; 0) × presencePenalty</code> from every logit.
      </p>
      <Source section="9.2" pages="203–217" figs="Figs. 9.6, 9.9, 9.14" />
      <p>
        Before you open <code>work/</code>, answer the six design questions the lab's README opens with: which of the
        chapter's privatisations still applies when one private copy is hundreds of KiB? Section{' '}
        <a href={href('ch09', '9.4')}>9.4</a>'s fit table puts the three vocabularies against the on-chip budgets your
        devices report. The README gives our answers at its end.
      </p>
      <Beyond>
        <p>
          Counting token ids, the decode step and the penalties are not in the book. The formula is llama.cpp's, which
          counts on the CPU; vLLM keeps a <code>[requests × vocabulary]</code> count tensor on the GPU (ch09-facts D4, D6,
          D7).
        </p>
      </Beyond>
      <LabFiles
        lab={l3}
        files={[
          [
            'kernel.cu',
            <>
              <code>countTokens</code>, <code>appendTokens</code>, <code>applyPenalties</code> and{' '}
              <code>launchCountTokens</code>, <code>launchAppendTokens</code>, <code>launchApplyPenalties</code>
            </>,
          ],
          ['kernel.metal', <><code>count_tokens</code>, <code>append_tokens</code>, <code>apply_penalties</code></>],
          ['launch.mm', <><code>encodeCountTokens</code>, <code>encodeAppendTokens</code>, <code>encodeApplyPenalties</code>; the grid sizes are yours</>],
        ]}
      />
      <p>
        Provided and not edited: <code>provided/count_pool.cu</code>, <code>.metal</code> and <code>.mm</code>, Fig.
        9.9's design applied to token counts. Every block gets a private copy of the whole vocabulary in a global-memory
        pool, counts there with block-scope atomics, then commits every non-zero bin to its request's row. The launcher
        zeroes the pool, and that is timed with it. It is the benchmark's comparator, and the tests run it too.
      </p>
      <p>
        On Metal there is no atomic view of a plain array: <code>count_tokens</code> declares the counts{' '}
        <code>device atomic_uint*</code> and updates them with <code>atomic_fetch_add_explicit</code>, relaxed and with no
        scope. The host buffer still holds plain 32-bit counters.
      </p>
      <Today checked="2026-11-20 (notes/today/ch09-facts.md C1)">
        <p>
          PyTorch binds a plain <code>uint32</code> tensor as <code>device atomic_uint*</code> in the same way.
        </p>
      </Today>
      <p>
        Run <code>python3 labs/pmpp.py run 9.3</code>, starting with <code>--backend emu</code>. Shapes run from one
        token over 1,000 ids up to 8 rows of 2,048 tokens and vocabularies of up to 201,088 ids (the large ones on a GPU
        only), each
        with four seeded distributions: uniform ids, Zipf-like ids where a few take most of the row, one id repeated
        through the whole row, and rows holding ids 0 and <code>vocab − 1</code>, which catch a wrong row offset. Counts
        must match exactly; counting and then appending one token per row must equal counting all of them; penalties must
        match the CPU within a tolerance, because Metal does not default to strict IEEE arithmetic, and with both
        penalties 0 every logit must come back bit for bit. The driver zeroes the counts first: the course's runtime
        fills new device buffers with <code>0xA5</code> bytes so that a missing zeroing shows. A plain increment in{' '}
        <code>appendTokens</code> must not be reported by <code>race</code>: one thread owns each row.
      </p>
      <RaceLine lab={l3} />
      <p>
        With <code>--bench</code> the driver runs batch 8, 4,096 tokens per row, vocabulary 151,936, on uniform and
        Zipf-like ids: your count, the provided pool, one append step and the penalties, each the best of at least 10 runs.
        The provided pool uses one token per thread, so 16 blocks per row, each zeroing and committing its own copy of
        151,936 bins. Commit two predictions first.
      </p>
      <LabPrediction
        id="ch09/labs/p-lab3-pool"
        prompt={
          <>
            Uniform ids: how many times as long as your direct atomics into each row will the provided{' '}
            <code>count_pool</code> take? A number below 1 means you expect the pool to be faster. Reason from what one
            pool block does: how many updates, against how many bins it must zero and read back.
          </>
        }
        unit="× the time of direct atomics"
        measure={poolOverCount}
        missing={BENCH_MISSING('9.3')}
        reading={
          <>
            Privatisation pays when a block's updates outnumber the bins it must initialise and commit. Here each pool
            block counts 256 tokens and handles 151,936 bins, so its overhead does not shrink with the queue it removes.
            Section <a href={href('ch09', '9.5')}>9.5</a>'s cure is fewer, busier blocks; the README's "Go further" works
            out how far that goes.
          </>
        }
      />
      <LabPrediction
        id="ch09/labs/p-lab3-append"
        prompt={
          <>
            How many times as long as one <code>appendTokens</code> step will recounting the whole context with{' '}
            <code>countTokens</code> take? Write the ratio of the work first, then the ratio of time you expect.
          </>
        }
        unit="× one append step"
        measure={recountOverAppend}
        missing={BENCH_MISSING('9.3')}
        reading={
          <>
            The work differs by the context length, 4,096 tokens against one per row. Both launches are short, so a fixed
            cost per launch can dominate both times; on CUDA and HIP the driver times with the host clock, launch included,
            and on Metal with the GPU's start and end times of one command buffer.
          </>
        }
      />
      <LabResults
        lab={l3}
        cols={[
          { head: 'Operation', cell: (r) => LAB3_NAME[r.name] ?? r.name },
          { head: 'Ids', cell: (r) => (r.data === 'zipf' ? 'Zipf' : String(r.data ?? '')) },
          {
            head: 'Batch × tokens, vocabulary',
            num: true,
            cell: (r) => nowrap(`${String(r.batch ?? '')} × ${num(r.len) != null ? count(num(r.len)!) : ''}, ${num(r.vocab) != null ? count(num(r.vocab)!) : ''}`),
          },
          msCol,
          { head: 'Rate', num: true, cell: rate },
        ]}
      />
      <p>
        The rate for <code>append</code> counts one token per row, so it is small by construction; compare its time, not
        its rate. The penalties' rate counts logits.
      </p>
      <h3>Project milestone M5, first piece</h3>
      <p>
        Milestone M5 of the through-line project is sampling on the GPU: top-k and top-p, with repetition penalties.
        Lab 9.3 is its counting piece: per-request token counts over <code>[batch × vocab]</code>, kept up to date one
        decode step at a time, and the frequency and presence penalties applied to the logits. Chapters 11, 12 and 14
        add the cumulative sum, the compaction and the sort that top-p and top-k need, and module X1 puts them together
        with this lab, including the sign-aware repetition penalty, which is left out here until its exact form is
        checked against llama.cpp's source.
      </p>
      <Beyond>
        <p>
          The through-line project, <code>nanoinfer</code>, is ours: one small GPU inference path for a decoder-only
          transformer, grown a milestone at a time.
        </p>
      </Beyond>
      <MilestoneStatus />
      <ReferenceSolution lab={l3} tiers={TOKEN_TIERS} />
    </LabsPage>
  );
}
