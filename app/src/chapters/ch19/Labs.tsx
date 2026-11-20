import type { Question } from '../../components/Ask';
import { Aside, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import {
  BackendsAndEmulator,
  LabFiles,
  LabPrediction,
  LabResults,
  LabsIntro,
  LabsPage,
  ReferenceSolution,
  RunningLabs,
  VerificationStatus,
  defineLabs,
  measure,
  msCol,
  numCol,
  preferredBench,
  rawCol,
  useLabs,
  type BenchSet,
  type Col,
} from '../../components/LabKit';
import { fmt, labStatus, num, realBackends, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { href } from '../../lib/router';
import { BOOK_DEFAULTS, llamaCuda, type LlamaPath } from '../../sim/convlib';
import { expansion, gemmShape, type ConvShape } from '../../sim/unfold';

// The page is built from components/LabKit.tsx; what is here is chapter 19's own content.
// No question on this page: the lab is the exercise. The three predictions are stored, not
// graded, under ch19/labs/p-lab1-late, p-lab1-early and p-lab1-unroll-share.
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('ch19', [{ num: '19.1', dir: 'lab1_convgemm', name: 'direct and unroll + GEMM' }]);
const [l1] = LABS;
/** Lab 6.4, whose launchLinear (milestone M2) the unroll + GEMM path calls. */
const LAB64 = 'ch06/lab4_linear';

/** Identifier colours in the reference solution: the tensors live in global memory, indices and sums in registers. */
const TIERS: Record<string, Tier> = {
  X: 'global', F: 'global', Y: 'global', Bt: 'global', Xn: 'global',
  acc: 'reg', m: 'reg', h: 'reg', w: 'reg', n: 'reg', c: 'reg', p: 'reg', q: 'reg', u: 'reg', v: 'reg', r: 'reg', i: 'reg',
};

// ---------------------------------------------------------------- the benchmark's layers
/** The five layers main.cpp times, in its order (Labs.test.ts ties them to the file). Ours, not the book's. */
export const LAB_LAYERS: readonly { id: string; note: string; shape: ConvShape }[] = [
  { id: 'early', note: 'few channels, a large map', shape: { N: 8, C: 3, H: 224, W: 224, M: 64, K: 3 } },
  { id: 'middle', note: 'more channels, a smaller map', shape: { N: 8, C: 64, H: 56, W: 56, M: 64, K: 3 } },
  { id: 'late', note: 'many channels, a small map', shape: { N: 8, C: 256, H: 14, W: 14, M: 256, K: 3 } },
  { id: 'pointwise', note: 'the late layer with 1 × 1 filters', shape: { N: 8, C: 256, H: 14, W: 14, M: 256, K: 1 } },
  { id: 'small', note: 'LeNet C3 at batch 64', shape: { N: 64, C: 6, H: 14, W: 14, M: 16, K: 5 } },
];
const TILE = 16;

/** What the model says about one layer: computed, never measured. */
export function layerFacts(s: ConvShape) {
  const g = gemmShape(s, { tile: TILE });
  const tileRows = Math.ceil(g.rows / TILE);
  const tileCols = Math.ceil(g.cols / TILE);
  return {
    gemm: `${g.rows} × ${g.inner} × ${g.cols}`,
    expansion: expansion(s),
    tileRows,
    tileCols,
    phases: g.phases,
    /** Width of the last phase's slice of the reduction. */
    lastPhase: g.inner - (g.phases - 1) * TILE,
    launches: s.N,
  };
}

/** llama.cpp's CUDA and HIP choice for each lab layer (ch19-facts D1, shape conditions only). */
export const llamaFor = (s: ConvShape): LlamaPath => llamaCuda({ ...s, ...BOOK_DEFAULTS }).path;
const LLAMA_WORDS: Record<LlamaPath, string> = {
  cublas: 'a cuBLAS GEMM on the input',
  direct: 'its direct kernel',
  implicit: 'its implicit GEMM',
  'cublas-or-direct': 'cuBLAS or its direct kernel (the facts do not say which test comes first)',
};

// ---------------------------------------------------------------- reading the benchmark rows
/** The row for one path and one layer, from one backend's preferred set. */
export const convRow = (s: BenchSet | undefined, name: string, shape: string): BenchRow | undefined =>
  s?.rows.find((r) => r.name === name && r.shape === shape);
const secs = (r: BenchRow | undefined) => num(r?.seconds);
const ms = (v: number | undefined) => (v ? `${fmt(v * 1000)} ms` : '?');
const source = (r: BenchRow | undefined) => (r?.gemm_source ? `, GEMM: ${String(r.gemm_source)}` : '');

/** Predictions 1 and 2: direct time over unroll + GEMM time on one layer. Above 1, unroll + GEMM was faster. */
const speedup = (layer: string) => (reports: Reports) =>
  measure(reports, l1, (s) => {
    const d = convRow(s, 'direct', layer);
    const g = convRow(s, 'unroll_gemm', layer);
    const a = secs(d);
    const b = secs(g);
    if (!a || !b) return null;
    return { value: a / b, detail: `${ms(a)} direct against ${ms(b)} unroll + GEMM${source(g)}` };
  });

/** Prediction 3: the N unrolls alone, as a percentage of the whole unroll + GEMM layer, on the early layer. */
const unrollShare = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const g = convRow(s, 'unroll_gemm', 'early');
    const whole = secs(g);
    const unroll = num(g?.unroll_seconds);
    if (!whole || !unroll) return null;
    return {
      value: (100 * unroll) / whole,
      detail: `${ms(unroll)} for the 8 unrolls alone, ${ms(whole)} for the whole layer, ${ms(num(g?.gemm_seconds))} for the 8 GEMMs alone`,
    };
  });

const MISSING = (
  <>
    This needs the benchmark rows of lab 19.1 from a real device. Run <code>python3 labs/pmpp.py run 19.1 --bench</code>{' '}
    once your kernels pass (or with <code>--solution</code> to time the reference). The emulator is never benchmarked.
  </>
);

// ---------------------------------------------------------------- results table
const PATH: Record<string, string> = { direct: 'direct', unroll_gemm: 'unroll + GEMM', implicit: 'implicit GEMM (provided)' };
const COLS: Col[] = [
  { head: 'Layer', cell: (r) => String(r.shape ?? '') },
  { head: 'Path', cell: (r) => PATH[r.name] ?? r.name },
  msCol,
  numCol('GFLOP/s', 'gflops'),
  { head: 'Unrolls alone, ms', num: true, cell: (r) => (num(r.unroll_seconds) ? fmt(num(r.unroll_seconds)! * 1000) : '') },
  { head: 'GEMMs alone, ms', num: true, cell: (r) => (num(r.gemm_seconds) ? fmt(num(r.gemm_seconds)! * 1000) : '') },
  {
    head: 'Unroll share',
    num: true,
    cell: (r) => (num(r.unroll_seconds) && num(r.seconds) ? `${fmt((100 * num(r.unroll_seconds)!) / num(r.seconds)!)}%` : ''),
  },
  numCol('Expansion (computed)', 'expansion'),
  numCol('Scratch bytes per sample', 'scratch_bytes'),
  rawCol('GEMM', 'gemm_source'),
];

/** Each path against the direct kernel, per layer and backend, from the preferred rows. */
function AgainstDirect() {
  const { reports, loading } = useLabs();
  if (loading) return null;
  const sets = preferredBench(reports, l1);
  const rows = sets.flatMap((s) => LAB_LAYERS.filter((L) => convRow(s, 'direct', L.id)).map((L) => ({ s, L })));
  if (!rows.length)
    return (
      <p className="muted" data-against="none">
        There are no benchmark rows for lab 19.1 on this machine yet, so there is nothing to compare. Run{' '}
        <code>python3 labs/pmpp.py run 19.1 --bench</code> on a device, then reload.
      </p>
    );
  const ratio = (s: BenchSet, layer: string, name: string) => {
    const a = secs(convRow(s, 'direct', layer));
    const b = secs(convRow(s, name, layer));
    return a && b ? fmt(a / b) : '';
  };
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto', marginBottom: '1.2rem' }}>
      <table className="data" style={{ margin: 0 }} data-against>
        <thead>
          <tr>
            <th>Backend</th>
            <th>Whose kernels</th>
            <th>Layer</th>
            <th className="num">Direct, ms</th>
            <th className="num">Direct time / unroll + GEMM time</th>
            <th className="num">Direct time / implicit time</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(({ s, L }) => (
            <tr key={s.backend + L.id} data-backend={s.backend} data-layer={L.id}>
              <td>
                {s.backend} <span className="muted">({s.device})</span>
              </td>
              <td>{s.variant === 'work' ? 'yours' : 'reference solution'}</td>
              <td>{L.id}</td>
              <td className="num">{fmt(secs(convRow(s, 'direct', L.id))! * 1000)}</td>
              <td className="num" data-col="unroll_gemm">
                {ratio(s, L.id, 'unroll_gemm')}
              </td>
              <td className="num" data-col="implicit">
                {ratio(s, L.id, 'implicit')}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** One line on lab 19.1 on the race backend, which the status table does not show. */
function RaceStatus() {
  const { reports, loading } = useLabs();
  const run = reports.race;
  const line = (variant: 'solution' | 'work') => {
    const rs = (run?.results.filter((x) => x.lab === l1.id && x.variant === variant) ?? []) as (RunResult & { tsan?: unknown[] })[];
    const tests = rs.flatMap((x) => x.tests);
    const found = rs.reduce((n, x) => n + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0);
    return { s: labStatus(run, l1.id, variant), tests, passed: tests.filter((t) => t.pass).length, found };
  };
  const sol = line('solution');
  const mine = line('work');
  return (
    <p data-race-status={loading ? 'loading' : sol.s} data-race-work={loading ? 'loading' : mine.s}>
      {loading ? (
        'Reading reports.'
      ) : !run ? (
        'No run on the race backend on this machine yet.'
      ) : (
        <>
          {sol.s === 'none'
            ? 'On this machine the reference solution has no run on the race backend yet.'
            : sol.s === 'pass'
              ? `On this machine the reference solution passed all ${sol.tests.length} tests on the race backend with ${sol.found ? `${sol.found} ThreadSanitizer report(s)` : 'no race reported'}: emulator-checked on this flavour.`
              : `On this machine the reference solution failed on the race backend (${sol.passed} of ${sol.tests.length} tests): unverified there.`}{' '}
          {mine.s === 'none'
            ? 'Your own kernels have no recorded run there, or are still the untouched starter.'
            : mine.s === 'pass'
              ? `Your own kernels passed all ${mine.tests.length} tests there with ${mine.found ? `${mine.found} ThreadSanitizer report(s): read them` : 'no race reported'}.`
              : `Your own kernels passed ${mine.passed} of ${mine.tests.length} tests there, with ${mine.found} ThreadSanitizer report(s) in the run.`}
        </>
      )}
    </p>
  );
}

/** Which GEMM this machine's runs used, and whether lab 6.4's own kernel passes yet. */
function GemmSource() {
  const { reports, loading } = useLabs();
  if (loading) return <p data-gemm-source="loading">Reading reports.</p>;
  const backends = realBackends(reports);
  const sources = [
    ...new Set(
      backends.flatMap((b) =>
        (['work', 'solution'] as const).flatMap((v) =>
          reports[b].results
            .filter((x) => x.lab === l1.id && x.variant === v)
            .flatMap((x) => x.bench)
            .map((r) => r.gemm_source)
            .filter((g): g is string => typeof g === 'string'),
        ),
      ),
    ),
  ];
  const passing = [...backends, 'emu'].filter((b) => labStatus(reports[b], LAB64, 'work') === 'pass');
  const where = (b: string) => (b === 'emu' ? 'the emulator' : b === 'metal' ? 'Metal' : b.toUpperCase());
  return (
    <p data-gemm-source={sources.join(',') || 'none'} data-lab64={passing.join(',') || 'none'}>
      {sources.length
        ? `The benchmark rows on this machine were timed with ${sources.map((x) => `"${x}"`).join(' and ')}.`
        : 'There are no benchmark rows for this lab on this machine yet, so no GEMM source is recorded.'}{' '}
      {passing.length
        ? `Your own lab 6.4 kernel passes its tests on ${passing.map(where).join(' and ')}, so PMPP_CH19_GEMM=work can use it.`
        : 'Your own lab 6.4 kernel has not passed its tests on this machine yet; until it does, PMPP_CH19_GEMM=work makes check (c) fail.'}
    </p>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  const facts = Object.fromEntries(LAB_LAYERS.map((L) => [L.id, layerFacts(L.shape)]));
  const late = facts.late;
  const early = facts.early;
  const small = facts.small;
  return (
    <LabsPage labs={LABS} tiers={TIERS}>
      <LabsIntro>
        One lab, in <code>labs/ch19/lab1_convgemm/</code>. It builds the layer of sections{' '}
        <a href={href('ch19', '19.2')}>19.2</a> and <a href={href('ch19', '19.3')}>19.3</a> two ways, directly and as an
        explicit unfold followed by a GEMM, and times them against each other and against the book's implicit GEMM. It adds
        no milestone to the through-line project: it calls milestone M2, lab 6.4's <code>launchLinear</code>, unchanged.
      </LabsIntro>
      <RunningLabs example="19.1" />
      <BackendsAndEmulator />
      <VerificationStatus />
      <p>
        This lab also runs on the <code>race</code> flavour of the emulator, with the threads concurrent under
        ThreadSanitizer, because its GEMM and the implicit kernel share tiles across barriers:{' '}
        <code>python3 labs/pmpp.py run 19.1 --backend race</code>.
      </p>
      <RaceStatus />

      <h2>Lab 19.1: a convolutional layer two ways</h2>
      <p>
        One convolutional layer, the book's: stride 1, no padding, row-major <code>float</code> tensors in NCHW order. X
        is <code>N × C × H × W</code>, F is <code>M × C × K × K</code>, Y is <code>N × M × H_out × W_out</code>, and P ={' '}
        <code>H_out · W_out</code> is the number of pixels in one output map. You compute it two ways.
      </p>
      <Source section="19.2, 19.3" pages="460–472" figs="Figs. 19.5, 19.7, 19.11; Eqs. 19.4, 19.5" />
      <ol>
        <li>
          <b>Direct.</b> Fig. 19.7, corrected: one thread per output pixel, blocks of <code>TILE_WIDTH × TILE_WIDTH</code>{' '}
          threads, grid <code>(M, T, N)</code> with the tile index linearised row by row as in Fig. 19.5, but with ceiling
          division and a bounds check on every output. You linearise the page's <code>X[n, c, h + p, w + q]</code> by hand.
        </li>
        <li>
          <b>Unroll + GEMM.</b> A kernel that writes one sample's unrolled input, transposed, as <code>Bt</code>: P rows,
          one per output pixel, each holding that pixel's <code>C·K·K</code> patch values in the book's (c, p, q) order.
          Then, for each sample, one unroll into a scratch <code>Bt</code> and one call of the project's GEMM.
        </li>
      </ol>
      <p>
        Why both. Section 19.3 argues that the explicit unfold costs memory and bandwidth (p. 467) and that the GEMM form
        pays when the matrices are large (p. 472). Here you measure both claims on your own device, on layers chosen to
        make the matrices large in different directions and, once, small. The second reason is the project's: the GEMM
        call is lab 6.4's <code>launchLinear(X, W, Y, batch, in, out)</code>, which computes <code>Y = X·Wᵀ</code> with W
        stored (out, in). Choosing its six arguments so that each sample's Y comes out <code>M × P</code>, in NCHW, with F
        never copied, is the test of whether the project's GEMM signature composes. Section 19.3 asks it as a question
        first.
      </p>
      <LabFiles
        lab={l1}
        files={[
          [
            'kernel.cu',
            <>
              <code>convDirectKernel</code> and <code>launchConvDirect</code>; <code>unrollKernel</code> and{' '}
              <code>launchUnroll</code>; <code>launchConvGemm</code>. Plain CUDA: the same file builds for HIP and both
              emulators.
            </>,
          ],
          ['kernel.metal', <><code>conv_direct</code> and <code>unroll_input</code></>],
          [
            'launch.mm',
            <>
              <code>encodeConvDirect</code>, <code>encodeUnroll</code> and <code>encodeConvGemm</code>. Its last function,{' '}
              <code>convGemmScratchProblem</code>, is provided and complete.
            </>,
          ],
        ]}
      />
      <p>
        On Metal, <code>encodeConvGemm</code> loops over the samples inside one compute encoder and calls the provided{' '}
        <code>encodeLinearAt</code>, which binds lab 6.4's <code>linear</code> kernel with a byte offset into each buffer.
        One scratch serves every sample because a plain compute encoder runs its dispatches in order and the buffers are
        hazard-tracked. Before the driver allocates that scratch, <code>convGemmScratchProblem</code> checks one sample's{' '}
        <code>Bt</code> against the device's <code>maxBufferLength</code> and fails with a message if it does not fit.
      </p>
      <Today checked="2026-11-20 (notes/today/ch02-facts.md C9; ch19-facts.md C2)">
        <p>
          A default compute encoder runs dispatches in order, with hazard tracking. <code>maxBufferLength</code> is at least
          256 MB by Apple's documentation; the device reports its own.
        </p>
      </Today>

      <h3>Where the code differs from the page</h3>
      <p>
        The lab departs from the book's figures in four places. Three are the notes' "Unclear in the PDF" items for pp.
        461, 462 and 471: section 19.2's "From blockIdx to pixel" explorer shows the first two failing and section 19.3's erratum hunt the
        third. The fourth fills a gap the book leaves.
      </p>
      <ul>
        <li>
          Fig. 19.5 divides <code>W_out / TILE_WIDTH</code>, which leaves 12 rows and 12 columns of LeNet C1's 28 × 28
          output uncomputed. The lab rounds up, and the threads with no pixel do nothing.
        </li>
        <li>
          Fig. 19.7 has no bounds check, and its signature lacks H, W and M, which the linearised indices need. The page
          also names its <code>dim3</code> variables <code>blockDim</code> and <code>gridDim</code>; the lab uses{' '}
          <code>dimBlock</code> and <code>dimGrid</code>, as lab 6.4 does, so they do not hide the built-in variables of
          those names.
        </li>
        <li>
          The book gives no unroll kernel. It describes explicit unrolling and its cost (p. 467) and moves on to the
          implicit version. The unroll kernel and its thread mapping are ours.
        </li>
        <li>
          Fig. 19.11, provided in <code>provided/implicit_gemm.*</code>: its phase count rounds up (the page's{' '}
          <code>(C*K*K)/TILE_WIDTH</code> runs one phase for 25 terms and drops 9), the F load, the X gather and the store
          are bounds-checked with zero fill, and the launch the book does not show uses{' '}
          <code>(ceil(P/16), ceil(M/16), N)</code> blocks.
        </li>
      </ul>

      <h3>Provided, not edited</h3>
      <ul>
        <li>
          <code>provided/gemm.cmake</code> picks which lab 6.4 kernel your GEMM path calls (below).
        </li>
        <li>
          <code>provided/linear_at.mm</code>: <code>encodeLinearAt</code>, lab 6.4's Metal launch with byte offsets.
        </li>
        <li>
          <code>provided/implicit_gemm.cu</code>, <code>.metal</code> and <code>_launch.mm</code>: the corrected Fig.
          19.11, timed beside your two paths. Section <a href={href('ch19', '19.3')}>19.3</a> shows it.
        </li>
        <li>
          <code>main.cpp</code>: the driver. Its CPU reference, <code>cpuConvLayer</code>, is Fig. 19.4 with the indices
          linearised and the sum kept in double.
        </li>
      </ul>

      <h3>Which GEMM</h3>
      <p>
        By default the unroll + GEMM path calls lab 6.4's reference solution. To call your own lab 6.4 kernel, run with{' '}
        <code>PMPP_CH19_GEMM=work</code>; <code>PMPP_CH19_GEMM=m2</code>, or nothing, goes back. The choice is read when
        the labs are configured, which <code>pmpp.py</code> does before every run, and every test run and benchmark row
        says which GEMM it used, <code>"m2 reference"</code> or <code>"m2 yours"</code>. With <code>work</code>, a compile
        error in your lab 6.4 file breaks this lab's build too.
      </p>
      <GemmSource />
      <Aside>
        <p>
          A recorded result can outlive a switch: the reports' source check does not cover the GEMM file, so look at the
          GEMM column of the results table.
        </p>
      </Aside>

      <h3>What the tests check</h3>
      <p>
        Run <code>python3 labs/pmpp.py run 19.1 --backend emu</code> first. It runs under AddressSanitizer, and X, F and{' '}
        <code>Bt</code> hold exactly their floats, so a read or write past any of them stops the run with the source line.
        Then <code>--backend race</code>, then <code>python3 labs/pmpp.py run 19.1</code> on your GPU. Every shape gets four
        checks:
      </p>
      <ul>
        <li>
          <b>(a) The unroll alone.</b> Every element of every sample's <code>Bt</code> is compared exactly with a CPU
          unroll. An index mistake is reported as an unroll error, with the row, the column and the X element it should
          have been, and not as a wrong GEMM.
        </li>
        <li>
          <b>(b) The direct path</b> and <b>(c) the unroll + GEMM path</b>, against <code>cpuConvLayer</code> with a
          tolerance scaled by the sum of <code>|x·f|</code> for each output. If (a) passed and (c) fails, the message
          points at the arguments of your GEMM call.
        </li>
        <li>
          <b>(d) The provided implicit GEMM.</b> A failure there is the course's bug, and the message says so.
        </li>
      </ul>
      <p>
        The shapes, as (N, C, H, W, M, K): the book's example (1, 3, 3, 3, 2, 2) with Fig. 19.2's integers, whose Y must be
        exactly <code>14 20 15 24 / 12 24 17 26</code> with no tolerance, since small integers are exact in float; the same
        twice; LeNet C1 (2, 1, 32, 32, 6, 5), where C·K·K = 25, the 28 × 28 output and M = 6 are none of them multiples of
        16; LeNet C3 (2, 6, 14, 14, 16, 5); a 1 × 1 layer (2, 8, 7, 9, 5, 1); one output pixel (1, 4, 5, 5, 3, 5); a
        rectangle (1, 2, 9, 17, 3, 3); and C·K·K a multiple of 16 (1, 16, 6, 6, 16, 2). On a GPU three larger layers are
        added. Y starts as NaN and is followed by a guard band of sentinel floats that must survive, and on a GPU each
        check runs twice with Y reset in between. The race flavour runs three of the small shapes.
      </p>
      <p>The untouched starter builds and fails: its empty kernels leave <code>Bt</code> and Y as NaN.</p>

      <h3>Before you benchmark, predict</h3>
      <p>
        The benchmark runs on real devices only, on five layer shapes of our choosing, given below in the chapter's names
        (M output maps, K the filter's side). For each layer, three paths take
        turns and the best of several rounds counts: your direct kernel; your unroll + GEMM, timed as the whole layer,
        then its N unrolls alone, then its N GEMMs alone on a filled scratch; and the provided implicit GEMM. Every path is
        first checked on two awkward shapes, because a timing of a wrong kernel means nothing.
      </p>
      <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto', marginBottom: '1.2rem' }}>
        <table className="data" style={{ margin: 0 }} data-table="layers">
          <thead>
            <tr>
              <th>Layer</th>
              <th>(N, C, H, W, M, K)</th>
              <th>Per-sample GEMM, M × C·K·K × P</th>
              <th className="num">Expansion, Eq. 19.4</th>
              <th className="num">16 × 16 tiles per GEMM</th>
              <th className="num">Phases</th>
            </tr>
          </thead>
          <tbody>
            {LAB_LAYERS.map((L) => {
              const f = facts[L.id];
              const s = L.shape;
              return (
                <tr key={L.id} data-layer={L.id}>
                  <td>
                    {L.id} <span className="muted">({L.note})</span>
                  </td>
                  <td>
                    ({s.N}, {s.C}, {s.H}, {s.W}, {s.M}, {s.K})
                  </td>
                  <td>{f.gemm}</td>
                  <td className="num" data-col="expansion">
                    {f.expansion.toFixed(2)}
                  </td>
                  <td className="num" data-col="tiles">
                    {f.tileRows} × {f.tileCols}
                  </td>
                  <td className="num" data-col="phases">
                    {f.phases}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p>
        Every number in this table is computed from the shapes, none measured. Write down three numbers before you run{' '}
        <code>python3 labs/pmpp.py run 19.1 --bench</code>.
      </p>
      <LabPrediction
        id="ch19/labs/p-lab1-late"
        prompt={
          <>
            The late layer. Direct time divided by unroll + GEMM time on your device: above 1 means unroll + GEMM is
            faster, below 1 slower.
          </>
        }
        unit="(direct time / unroll + GEMM time)"
        measure={speedup('late')}
        missing={MISSING}
        reading={
          <>
            Per sample the GEMM is {late.gemm}: {late.tileRows} × {late.tileCols} tiles, each running {late.phases}{' '}
            phases, so every tile of F and of <code>Bt</code> it loads is reused 16 times. The direct kernel shares nothing
            between threads, and requests 0.25 FLOP/B, but its neighbouring threads read overlapping patches that caches
            may serve. Compare the implicit GEMM's row too: it does the same arithmetic without writing{' '}
            <code>Bt</code>.
          </>
        }
      />
      <LabPrediction
        id="ch19/labs/p-lab1-early"
        prompt={<>The same ratio on the early layer: direct time divided by unroll + GEMM time.</>}
        unit="(direct time / unroll + GEMM time)"
        measure={speedup('early')}
        missing={MISSING}
        reading={
          <>
            Per sample the GEMM is {early.gemm}: {early.tileRows} tile rows and {early.tileCols.toLocaleString('en-US')}{' '}
            tile columns, but only {early.phases} phases, the second {early.lastPhase} of 16 wide. The reduction is short,
            so each tile is reused little, and <code>Bt</code> is {early.expansion.toFixed(2)} times the size of the
            sample it came from. Set the two layers' ratios side by side before you explain either.
          </>
        }
      />
      <LabPrediction
        id="ch19/labs/p-lab1-unroll-share"
        prompt={
          <>
            On the early layer, what percentage of unroll + GEMM's time goes to the 8 unrolls alone? (The benchmark times
            the unrolls and the GEMMs separately, so the two need not add up to the whole.)
          </>
        }
        unit="% of the layer's time"
        measure={unrollShare}
        missing={MISSING}
        reading={
          <>
            The unroll moves bytes and does no arithmetic: it reads X, writes <code>Bt</code>, and the GEMM reads{' '}
            <code>Bt</code> back. It also does integer divides and remainders for every element it writes, where the
            direct kernel does one divide and one remainder per thread before its loops. If the share surprises you, run
            the benchmark again: one run of a short kernel can mislead.
          </>
        }
      />
      <p>
        Things to weigh, for all three. None of them gives the answer, and some pull in opposite directions. What the
        vendors say about integer division, and llama.cpp's rule for small layers, were checked on the date in the
        margin; the rest is the chapter's arithmetic and ours.
      </p>
      <Today checked="2026-11-20 (ch19-facts A8, B5, C6, D1)">
        <p>
          CUDA Best Practices Guide; HIP documentation; WWDC16 session 606. llama.cpp: <code>small_conv</code> in{' '}
          <code>ggml-cuda/conv2d.cu</code>.
        </p>
      </Today>
      <ul>
        <li>
          Requested traffic is not traffic. Neighbouring threads ask for overlapping patches, and caches can serve a
          repeated request without another trip to memory. The direct kernel here has the same 0.25 FLOP/B as chapter
          5&rsquo;s naive matmul: compare your own lab 5.1 naive result with that ceiling on the roofline in{' '}
          <a href={href('ch05', '5.1')}>section 5.1</a> (tick &ldquo;show my measurements&rdquo;).
        </li>
        <li>
          The small layer, LeNet C3 at batch 64, makes {small.launches} unroll launches and {small.launches} GEMM
          launches, each GEMM {small.tileRows} tile row by {small.tileCols} tile columns. The book names launch overhead
          as a cost of small multiplications (p. 472).
        </li>
        <li>
          Integer division is costly. NVIDIA documents it as compiling to up to 20 instructions, AMD as taking many more
          cycles than a multiplication, and Apple's only statement is a 2016 talk saying most GPUs have no integer divide
          instruction.
        </li>
        <li>
          The reference unroll gives each thread one element of <code>Bt</code> in <code>Bt</code>'s own order, so
          neighbouring threads write neighbouring floats and read X along a patch row. Numbering the threads along the
          pixels instead reads X contiguously and writes <code>Bt</code> with a stride of C·K·K. You cannot have both
          without staging through shared memory. Which did you pick?
        </li>
        <li>
          llama.cpp's CUDA and HIP backend sends layers with C·K·K below 64 and fewer than 512 output pixels to a direct
          kernel. Which of the five layers would that rule send there? It is a production choice for llama.cpp's
          kernels, not a prediction for yours.
        </li>
      </ul>
      <LabResults lab={l1} cols={COLS} />
      <p>
        "Unroll share" is the N unrolls alone over the whole unroll + GEMM layer. "Expansion" is Eq. 19.4 for the layer,
        computed by the driver, and the scratch is one sample's <code>Bt</code>. GFLOP/s counts <code>2·N·M·C·K²·P</code>{' '}
        for every path, so the three paths of one layer are compared on the same work.
      </p>
      <p>The same rows, each path against the direct kernel. Above 1, the path was faster than direct.</p>
      <AgainstDirect />
      <p data-llama>
        The answer to the last question in the list: by llama.cpp's shape rule as read on 2026-11-20,{' '}
        {LAB_LAYERS.map((L, i) => (
          <span key={L.id} data-llama-layer={L.id} data-llama-path={llamaFor(L.shape)}>
            {i ? (i === LAB_LAYERS.length - 1 ? ', and ' : ', ') : ''}
            {L.id} would go to {LLAMA_WORDS[llamaFor(L.shape)]}
          </span>
        ))}
        . {LAB_LAYERS.some((L) => llamaFor(L.shape) === 'direct') ? '' : 'None of the five goes to its direct kernel. '}
        The small layer's reduction, C·K·K = {LAB_LAYERS[4].shape.C * LAB_LAYERS[4].shape.K ** 2}, is already above the
        cut-off of 64.
      </p>
      <Simplification>
        <p>
          Shape conditions only, as read on 2026-11-20 (ch19-facts D1): data types and the split-K condition are not
          modelled. The five layers are ours, chosen to span early, middle, late, 1 × 1 and small, and are no particular
          network.
        </p>
      </Simplification>

      <h3>Go further</h3>
      <p>
        Once your kernels pass, five things to try. The first is the book's own suggestion; the constant K and the
        whole-batch GEMM come from how production code handles this layer.
      </p>
      <Beyond>
        <p>Ours, from llama.cpp's and PyTorch's code as read on 2026-11-20 (ch19-facts D1, D2, D4, E2).</p>
      </Beyond>
      <ul>
        <li>
          The book leaves shared-memory tiling of the direct kernel to you (p. 463). Chapter 7's halo tiles, section{' '}
          <a href={href('ch07', '7.5')}>7.5</a> and lab 7.2, are the technique. Tile one input map, measure, then ask what
          the tile cannot give you: reuse across output maps, which section 19.3 gets from the GEMM form.
        </li>
        <li>
          Flip your unroll's thread mapping and time the unrolls alone, on the early and the late layer.
        </li>
        <li>
          Make <code>K</code> a compile-time constant, a macro in CUDA or a function constant in Metal, so that{' '}
          <code>/K</code> and <code>%K</code> divide by a literal, and time the unrolls again. llama.cpp does this for 3 ×
          3 and 1 × 1 filters.
        </li>
        <li>
          Unroll the whole batch into one <code>C·K·K × N·P</code> matrix and make one GEMM call. The result comes out{' '}
          <code>M × N × P</code> and needs a permute to become NCHW: time the permute with it.
        </li>
        <li>
          Run with <code>PMPP_CH19_GEMM=work</code> once your lab 6.4 kernel passes: the same layer, your own GEMM.
        </li>
      </ul>
      <p>
        If you compare with <code>torch.nn.functional.conv2d</code> on an NVIDIA GPU, a float32 call may run in TF32
        unless you turn that off. Compare with a tolerance and print the setting. Section{' '}
        <a href={href('ch19', '19.4')}>19.4</a> asks what to check first when the two disagree.
      </p>
      <Today checked="2026-11-20 (ch19-facts A6)">
        <p>
          <code>cudnn.allow_tf32</code> defaults to True; PyTorch plans to replace it with{' '}
          <code>cudnn.conv.fp32_precision</code>, whose default it does not state.
        </p>
      </Today>
      <ReferenceSolution lab={l1} />

      <h3>The project</h3>
      <p>
        This chapter adds no milestone. The lab reuses M2 as it is: lab 6.4's <code>launchLinear</code>, called once per
        sample with F as the input matrix and <code>Bt</code> as the weight matrix. Nothing in the GEMM had to change,
        which is the point of the project's conventions: row-major <code>float</code>, raw device pointers and explicit
        sizes. The <a href={href('ch19', 'wrap')}>summary</a> returns to what that means for the kernels to come.
      </p>
    </LabsPage>
  );
}
