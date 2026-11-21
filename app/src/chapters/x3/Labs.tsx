import { useState, type ReactNode } from 'react';
import { Predict, type Question } from '../../components/Ask';
import { Beyond, Today, type Tier } from '../../components/notes';
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
  benchRow,
  benchSets,
  defineLabs,
  measure,
  msCol,
  numCol,
  rawCol,
  useLabs,
  type BenchSet,
  type Col,
  type Lab,
} from '../../components/LabKit';
import { fmt, labStatus, num, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { href } from '../../lib/router';
import { LAB_GPU } from '../../sim/ckptformat';
import { JOB_LABEL, launchCount, stepLaunches, type Config } from '../../sim/llamastep';

// The page is built from components/LabKit.tsx; what is here is module X3's own content.
// No question on this page: the labs are the exercise. Its five predictions are stored, not
// graded: x3/labs/p-lab1-port-emu, p-lab1-rmsnorm-bw, p-lab1-rmsnorm-bw-256, p-lab2-bwfraction,
// p-lab2-attn-share.
export const questionsLabs: Record<string, Question> = {};

const LABS = defineLabs('x3', [
  { num: 'X3.1', dir: 'lab1_llama_pieces', name: 'RMSNorm, RoPE, SwiGLU and the port' },
  { num: 'X3.2', dir: 'lab2_tinyllama', name: 'the M10 Llama step' },
]);
const [l1, l2] = LABS;

/** Identifier colours in lab X3.1's reference solution: where each value lives. */
const tiers = (global: string[], shared: string[], reg: string[]): Record<string, Tier> => ({
  ...Object.fromEntries(global.map((k) => [k, 'global' as Tier])),
  ...Object.fromEntries(shared.map((k) => [k, 'shared' as Tier])),
  ...Object.fromEntries(reg.map((k) => [k, 'reg' as Tier])),
});
const TIERS_1 = tiers(
  ['x', 'w', 'y', 'in', 'out', 'pos', 'cosT', 'sinT', 'gate', 'up', 'v', 'x4', 'w4', 'y4', 'xr', 'yr'],
  ['scratch', 's_sum'],
  ['s', 'scale', 'sum', 'val', 'x0', 'x1', 'g', 'lane', 'warp', 'tid'],
);

// ---------------------------------------------------------------- reading the reports
const missingBench = (labNum: string, extra?: ReactNode) => (
  <>
    This needs the benchmark rows of lab {labNum} from a real device. Run{' '}
    <code>python3 labs/pmpp.py run {labNum} --bench</code> once your kernels pass (or with <code>--solution</code> to time
    the reference). The emulator is never benchmarked.{extra ? <> {extra}</> : null}
  </>
);
const NEED_51 = (
  <>
    The fraction also needs lab 5.1's bandwidth: run <code>python3 labs/pmpp.py run 5.1 --bench</code> first.
  </>
);

/** Lab X3.1: launchRmsNorm's counted bandwidth over lab 5.1's, at d = 4096, as a percentage. */
const rmsFraction = (rows: number) => (reports: Reports) =>
  measure(reports, l1, (s: BenchSet) => {
    const r = benchRow(s, 'rmsnorm', { rows, d: 4096 });
    const bw = num(r?.gb_per_s);
    const ref = num(r?.lab51_gb_per_s);
    if (bw == null || !ref) return null;
    return { value: (bw / ref) * 100, detail: `${fmt(bw)} GB/s counted against lab 5.1's ${fmt(ref)} GB/s, rows ${rows}, d 4096` };
  });

/** Lab X3.2: tokens per second at batch 1, position 16, as a percentage of the weight-bandwidth ceiling. */
const ceilingFraction = (reports: Reports) =>
  measure(reports, l2, (s: BenchSet) => {
    const r = benchRow(s, 'ceiling');
    const f = num(r?.fraction);
    if (f == null) return null;
    return {
      value: f * 100,
      detail: `${fmt(num(r!.tokens_per_s) ?? NaN)} tokens/s at batch 1, position 16, against a ceiling of ${fmt(num(r!.ceiling_tokens_per_s) ?? NaN)}: lab 5.1's ${fmt(num(r!.lab51_gb_per_s) ?? NaN)} GB/s over ${fmt((num(r!.weight_bytes_per_token) ?? NaN) / 1e6)} MB of weights per token`,
    };
  });

/** A piece row of lab X3.2: one group's time at a batch and position. */
const piece = (s: BenchSet, batch: number, position: number, group: string) =>
  s.rows.find((r) => r.name === 'piece' && num(r.batch) === batch && num(r.position) === position && r.group === group);

/** Lab X3.2: attention's share of the summed pieces at position 512, batch 1; position 16 alongside. */
const attentionShare = (reports: Reports) =>
  measure(reports, l2, (s: BenchSet) => {
    const late = piece(s, 1, 512, 'attention');
    const v = num(late?.share);
    if (v == null) return null;
    const early = num(piece(s, 1, 16, 'attention')?.share);
    return {
      value: v * 100,
      detail: `batch 1, the pieces timed alone with the ${GEMM[String(late!.gemm)] ?? String(late!.gemm)} for the matrix products; at position 16 the share was ${early == null ? 'not recorded' : `${fmt(early * 100)} %`}`,
    };
  });

// ---------------------------------------------------------------- results tables
const KERNEL_1: Record<string, string> = { rmsnorm: 'launchRmsNorm', rope: 'launchRope, mode 0', swiglu: 'launchSwiGLU' };
const usCol: Col = {
  head: 'Time, µs',
  num: true,
  cell: (r) => {
    const v = num(r.seconds);
    return v == null ? '' : fmt(v * 1e6);
  },
};
const pctOf51: Col = {
  head: '% of lab 5.1',
  num: true,
  cell: (r) => {
    const bw = num(r.gb_per_s);
    const ref = num(r.lab51_gb_per_s);
    return bw == null || !ref ? '' : fmt((bw / ref) * 100);
  },
};
const headsCol: Col = {
  head: 'Heads × size',
  num: true,
  cell: (r) => (num(r.heads) != null ? `${r.heads} × ${r.head_dim}` : ''),
};

const GEMM: Record<string, string> = { gemv: 'lab 22.1 gather', m2: 'lab 6.4 (M2)', m3: 'lab 15.6 (M3)' };
const ATTN: Record<string, string> = { grouped: 'lab 20.5, grouped', decode: 'lab 20.4' };
const big = (v: unknown) => (num(v) != null ? num(v)!.toLocaleString('en-US') : '?');
const mb = (v: unknown) => (num(v) != null ? fmt(num(v)! / 1e6) : '');
const STEP_COLS: Col[] = [
  { head: 'Matrix products', cell: (r) => GEMM[String(r.gemm)] ?? String(r.gemm ?? '') },
  { head: 'Attention', cell: (r) => ATTN[String(r.attention)] ?? String(r.attention ?? '') },
  rawCol('Batch', 'batch'),
  rawCol('Position', 'position'),
  msCol,
  numCol('Tokens/s', 'tokens_per_s'),
  numCol('Weight GB/s', 'weight_gb_per_s'),
  { head: 'Weights per token, MB', num: true, cell: (r) => mb(r.weight_bytes_per_token) },
  { head: 'KV read, MB', num: true, cell: (r) => mb(r.kv_bytes) },
];

/** Lab X3.2: the per-piece split, one row per (backend, kernel, batch, position), groups as columns. */
const GROUPS: [string, string][] = [
  ['matmuls', 'matrix products'],
  ['attention', 'append + attention'],
  ['norms+rope+swiglu+residual', 'norms, RoPE, SwiGLU, residual'],
  ['embed+lm head+argmax+advance', 'embed, LM head, argmax, advance'],
];
function PieceTable() {
  const { reports, loading } = useLabs();
  const rows = benchSets(reports, l2).flatMap((s) => {
    const keys = [...new Set(s.rows.filter((r) => r.name === 'piece').map((r) => `${r.gemm}|${r.batch}|${r.position}`))];
    return keys.map((k) => {
      const [gemm, batch, position] = k.split('|');
      const at = (g: string) => piece(s, Number(batch), Number(position), g) as BenchRow | undefined;
      return { s, gemm, batch, position, at };
    });
  });
  if (loading) return <p className="muted">Reading reports.</p>;
  if (!rows.length)
    return (
      <p className="muted" data-pieces-empty="X3.2">
        No per-piece timings on this machine yet: they come with <code>python3 labs/pmpp.py run X3.2 --bench</code>.
      </p>
    );
  return (
    <div className="wide" style={{ maxWidth: '62rem', overflowX: 'auto' }}>
      <table className="data" data-table="pieces">
        <thead>
          <tr>
            <th>Backend</th>
            <th>Kernel</th>
            <th>Matrix products</th>
            <th className="num">Batch</th>
            <th className="num">Position</th>
            {GROUPS.map(([, h]) => (
              <th key={h} className="num">
                {h}, ms (share, %)
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(({ s, gemm, batch, position, at }, i) => (
            <tr key={i}>
              <td>{s.backend}</td>
              <td>{s.variant === 'work' ? 'yours' : 'reference solution'}</td>
              <td>{GEMM[gemm] ?? gemm}</td>
              <td className="num">{batch}</td>
              <td className="num">{position}</td>
              {GROUPS.map(([g]) => {
                const r = at(g);
                const t = num(r?.seconds);
                const sh = num(r?.share);
                return (
                  <td key={g} className="num">
                    {t == null ? '' : `${fmt(t * 1000)} (${sh == null ? '?' : fmt(sh * 100)})`}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Lab X3.2: the shape of the benchmark model, as the driver reported it. */
function ModelLine() {
  const { reports, loading } = useLabs();
  const found = benchSets(reports, l2).flatMap((s) => {
    const r = benchRow(s, 'model');
    return r ? [{ s, r }] : [];
  });
  return (
    <p data-model={loading ? 'loading' : found.length ? found.map((f) => f.s.backend + ':' + f.s.variant).join(',') : 'none'}>
      {loading
        ? 'Reading reports.'
        : !found.length
          ? 'No benchmark model is recorded on this machine yet; the driver prints its shape when it runs the benchmark.'
          : found.map((f, i) => (
              <span key={i}>
                {i ? ' ' : ''}On {f.s.backend} ({f.s.variant === 'work' ? 'your run' : 'the reference solution’s run'}) the benchmark
                model was d {big(f.r.d)} with {big(f.r.heads)} query heads over {big(f.r.kv_heads)} KV heads, hidden{' '}
                {big(f.r.hidden)}, {big(f.r.layers)} layers, vocabulary {big(f.r.vocab)} and cache length {big(f.r.seq_len)}:{' '}
                {mb(f.r.weight_bytes)} MB of float32 weights.
              </span>
            ))}
    </p>
  );
}

// ---------------------------------------------------------------- the race backend, lab X3.1
/** One line on lab X3.1's runs on the race backend, which the status table does not show. */
function RaceLine({ lab }: { lab: Lab }) {
  const { reports, loading } = useLabs();
  const run = reports.race;
  const line = (variant: 'solution' | 'work') => {
    const rs = (run?.results.filter((x) => x.lab === lab.id && x.variant === variant) ?? []) as (RunResult & { tsan?: unknown[] })[];
    const tests = rs.flatMap((x) => x.tests);
    const found = rs.reduce((k, x) => k + (Array.isArray(x.tsan) ? x.tsan.length : 0), 0);
    return { s: labStatus(run, lab.id, variant), tests, passed: tests.filter((t) => t.pass).length, found };
  };
  const sol = line('solution');
  const mine = line('work');
  return (
    <p data-race={lab.num} data-race-status={loading ? 'loading' : sol.s} data-race-work={loading ? 'loading' : mine.s}>
      {loading ? (
        'Reading reports.'
      ) : !run ? (
        'No run on the race backend on this machine yet.'
      ) : (
        <>
          {sol.s === 'none'
            ? `On this machine the reference solution of lab ${lab.num} has no run on the race backend yet.`
            : sol.s === 'pass'
              ? `On this machine the reference solution of lab ${lab.num} passed all ${sol.tests.length} tests on the race backend with ${sol.found ? `${sol.found} ThreadSanitizer report(s)` : 'no race reported'}: emulator-checked on this flavour.`
              : `On this machine the reference solution of lab ${lab.num} failed on the race backend (${sol.passed} of ${sol.tests.length} tests): unverified there.`}{' '}
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

/** Lab X3.1's port prediction: how many of the three defects emu reports in the code as shipped. */
function PortReveal({ committed }: { committed: number }): ReactNode {
  // The count and the reason come at once; which defect is which waits until the reader says
  // the review is over, so the reveal does not hand over the answers before the first run.
  const [fixed, setFixed] = useState(false);
  return (
    <div data-reveal="p-lab1-port-emu">
      <p>
        When this lab was built, <code>emu</code> reported <b>2</b> of the three in the port as shipped, in all three of
        its thread orders. {committed === 2 ? 'That is your number.' : `You predicted ${committed}.`}
      </p>
      <p>
        The one it did not report is a missing barrier. The emulator runs each thread up to its next synchronisation point
        (a barrier or a warp shuffle) before the next thread runs, so a write that comes after the writer's own next
        synchronisation point can never overtake a slower thread's read. It shows a missing barrier only when a thread
        writes the shared buffer before that point. The race backend runs the threads at once and names both lines.
      </p>
      {fixed ? (
        <p data-reveal-detail="p-lab1-port-emu">
          Defect by defect: (a), the second stage reading slots no warp wrote, turned the d 64 cases to NaN, because a block
          of 256 has fewer warps than the stage reads slots and the emulator fills shared memory with <code>0xCD</code>.
          (c), the <code>float4</code> path taken when d is not a multiple of 4, stopped the run with an AddressSanitizer
          report at the vector read. (b), no barrier before the next row overwrites the buffer, passed every{' '}
          <code>emu</code> case until (a) was fixed by zeroing the slots first: the zeroing is a write before the writer's
          next synchronisation point, and then <code>emu</code> fails the several-rows cases in two of its orders. The race
          backend named (b) in the shipped code, at the slot write and the scale read. On the Metal GPU this lab was written
          on, only (c) failed; (a) and (b) passed every case there.
        </p>
      ) : (
        <p>
          <button type="button" onClick={() => setFixed(true)}>
            I have fixed all three: show which backend saw which
          </button>
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- lab X3.2's launch order, from the tested model
const GPU_MODEL: Config = {
  d: LAB_GPU.dim,
  hidden: LAB_GPU.hidden_dim,
  layers: LAB_GPU.n_layers,
  heads: LAB_GPU.n_heads,
  kvHeads: LAB_GPU.n_kv_heads,
  vocab: LAB_GPU.vocab_size,
  head: LAB_GPU.dim / LAB_GPU.n_heads,
  seqLen: LAB_GPU.seq_len,
};
function LaunchTable() {
  const all = stepLaunches(GPU_MODEL, 0);
  const L = GPU_MODEL.layers;
  const head = all.filter((l) => l.layer === null || l.layer === 0);
  const firstTail = head.findIndex((l) => l.layer === null && l.n > 0);
  const row = (l: (typeof all)[number]) => (
    <tr key={l.n} data-launch={l.job}>
      <td className="num">{l.n}</td>
      <td>
        {l.layer === 0 ? 'layer 0: ' : ''}
        {JOB_LABEL[l.job]}
      </td>
      <td>
        <code>{l.fn}</code>
      </td>
      <td>{l.exists ? l.lab : <b>{l.lab}</b>}</td>
    </tr>
  );
  return (
    <div className="wide" style={{ maxWidth: '58rem', overflowX: 'auto' }}>
      <table className="data" data-table="launches">
        <thead>
          <tr>
            <th className="num">#</th>
            <th>Operation</th>
            <th>Launch</th>
            <th>From</th>
          </tr>
        </thead>
        <tbody>
          {head.slice(0, firstTail).map(row)}
          <tr data-launch="layers">
            <td className="num">
              17–{16 * L}
            </td>
            <td colSpan={3} className="muted">
              layers 1 to {L - 1}: the same sixteen launches, each with its own layer's weights and cache
            </td>
          </tr>
          {head.slice(firstTail).map(row)}
        </tbody>
      </table>
      <p className="muted" style={{ marginTop: '0.4rem' }}>
        One step of lab X3.2's GPU test model ({L} layers, {all.length} launches), from <code>app/src/sim/llamastep.ts</code>
        , the model section <a href={href('x3', 'X3.7')}>X3.7</a> steps through; its unit test checks the order against{' '}
        <code>llama.h</code>. In general a step is {launchCount(1) - 16} + 16 × layers launches. Bold: new in this module.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS} tiers={TIERS_1}>
      <LabsIntro>
        Two labs, each in its own directory under <code>labs/x3/</code>. The first writes the three kernels a Llama
        decode step needs that milestones M0 to M8 lack (RMSNorm, RoPE and SwiGLU) and reviews an RMSNorm ported in the
        structure of llama.cpp's kernels, with three defects planted in it. The second is milestone M10: it composes
        every milestone into one greedy decode step of a Llama-style model and checks each logit and each token against a
        CPU reference.
      </LabsIntro>
      <p>
        <code>pmpp.py</code> names this module's labs as it names the book's: <code>python3 labs/pmpp.py run X3.1</code>{' '}
        runs lab X3.1, and <code>X3.2</code> is milestone M10.
      </p>
      <Beyond title="All of this is ours">
        <p>
          Module X3 is not in the book. The contracts, tests and models on this page are ours. What the kernels compute
          comes from the projects read on 2026-11-20 at the commits section <a href={href('x3', 'start')}>Before you
          start</a> lists; the reduction is lab 10.3's, from the book's section 10.8.
        </p>
      </Beyond>
      <RunningLabs example="X3.1" />
      <BackendsAndEmulator />
      <p>
        Lab X3.1 is also built for the emulator's race flavour, because its port shares one buffer between two reductions
        and across rows. <code>python3 labs/pmpp.py run X3.1 --backend race</code> runs a block's threads at once under
        ThreadSanitizer, which reports two accesses to one location with nothing ordering them, with both source lines. A
        clean run is evidence, not proof: it sees only the schedules that happened. Lab X3.2 is not: your code there
        launches pieces and writes no shared memory, and each piece was race-tested in its own lab.
      </p>
      <VerificationStatus />
      <p>
        Lab X3.2 runs other labs' reference solutions, which you do not edit. <code>provided/pieces.cmake</code> copies them
        into the build directory at configure time: labs 2.3, 6.4, 10.3, 20.4, 22.2 and X3.1 are required, and 15.6, 20.5,
        22.1 and X1.3 are used when present. It renames what would clash and makes one-line device helpers{' '}
        <code>static</code>, as lab 22.2's does. A missing required piece, or one whose signature differs from the one{' '}
        <code>llama.h</code> declares, skips the lab with a note naming it. So M10 runs the reference pieces, not your own
        versions of them: a failure in its "pieces alone" check is the course's, not yours.
      </p>

      {/* ------------------------------------------------------------ X3.1 */}
      <h2>Lab X3.1: the Llama pieces the course has not built, and a port review</h2>
      <p>
        Part A is the three kernels in their smallest form, row-major float32 with the tokens first. Section{' '}
        <a href={href('x3', 'X3.3')}>X3.3</a> reads the projects' RMSNorm kernels and section{' '}
        <a href={href('x3', 'X3.7')}>X3.7</a> places all three in the step.
      </p>
      <ul>
        <li>
          <code>launchRmsNorm(x, w, y, rows, d, eps)</code>: each row scaled by 1 / sqrt(mean(x²) + eps), times the weight.
          One block of 256 threads per row with lab 10.3's two-stage sum: a typed lane mask, <code>warpSize</code>, the
          second stage guarded. <code>y</code> may be <code>x</code>.
        </li>
        <li>
          <code>launchRope(x, pos, cosT, sinT, rows, heads, headDim, mode)</code>: in place, row r at position{' '}
          <code>pos[r]</code>, the tables built on the host in double so the kernel does no trigonometry. Mode 0 rotates
          the pairs (2i, 2i+1), llama.cpp's NORM and llama2.c's; mode 1 rotates (i, i + headDim/2), the NeoX pairing of
          Hugging Face's <code>rotate_half</code>.
        </li>
        <li>
          <code>launchSwiGLU(gate, up, out, n)</code>: gate / (1 + exp(−gate)) × up, llama.cpp's SiLU times the up
          projection. <code>out</code> may be <code>gate</code>.
        </li>
      </ul>
      <p>
        On Metal: <code>rms_norm_rows</code> (one threadgroup per row, <code>simd_sum</code>, the SIMD-group width read from
        the pipeline), <code>rope_rows</code> and <code>swiglu</code>, encoded into the caller's encoder. The tests feed
        SwiGLU a gate of −100, where exp(100) overflows. CUDA's default IEEE division turns −100 / infinity into −0; under
        Metal's fast math the behaviour of infinity is undefined, so the Metal kernel must compute the same function
        without forming it.
      </p>
      <Today checked="2026-11-20 (notes/today/ch05-facts.md C7; ch10-facts.md C3; ch21-facts.md A10)">
        <p>
          MSL defaults to non-IEEE floating-point optimisations (Apple's sources differ on exactly which mode); under fast
          math the handling of infinity and NaN is undefined. CUDA's division and <code>sqrtf</code> are correctly rounded
          by default.
        </p>
      </Today>
      <LabFiles
        lab={l1}
        files={[
          ['kernel.cu', <><code>rmsNormRowsKernel</code>, <code>ropeKernel</code>, <code>swigluKernel</code> and their launchers; part B's <code>rmsNormPortKernel</code> and <code>launchRmsNormPort</code></>],
          ['kernel.metal', <><code>rms_norm_rows</code>, <code>rope_rows</code>, <code>swiglu</code>; part B's <code>rms_norm_port</code></>],
          ['launch.mm', <><code>encodeRmsNorm</code>, <code>encodeRope</code>, <code>encodeSwiGLU</code>; part B's <code>encodeRmsNormPort</code>, which picks the <code>float4</code> path on the host as llama.cpp does</>],
        ]}
      />
      <h3>Part B: the port</h3>
      <p>
        A colleague ported an RMSNorm in the structure of llama.cpp's kernels: 256 threads for rows under 1,024 elements and
        1,024 above, a <code>float4</code> path when the launch judges it safe, and a two-stage reduction through one
        shared buffer of 64 slots whose slot 0 hands the scale to the block. A fixed grid of 4 blocks walks the rows, so a
        block reuses its buffer row after row (llama.cpp itself runs one block per row). It compiles and passes some tests.
        It has three independent defects, the same three in both files. Each is something llama.cpp's kernels guard
        against.
      </p>
      <p>
        The README lists what is not a defect: the typed lane mask and <code>warpSize</code>, the book's{' '}
        <code>float*</code> to <code>float4*</code> cast, binding a Metal buffer a second time as{' '}
        <code>device float4*</code>, the 64 slots, the grid of 4, the block-size rule, <code>1 / sqrtf</code> and reading
        the scale from slot 0.
      </p>
      <Today checked="2026-11-20 (notes/today/infra-facts.md A6, C7; x3-facts.md B2; ch10-facts.md D1)">
        <p>
          No vendor documents the pointer cast; Metal documents declaring the buffer <code>device float4*</code>. llama.cpp
          guards its second stage and checks <code>ncols % 4</code>.
        </p>
      </Today>
      <ol>
        <li>
          Read <code>work/kernel.cu</code> first and write down the defects you see, and for each the backend you expect to
          report it: <code>emu</code>, <code>race</code>, your GPU, or none.
        </li>
        <li>
          Run <code>python3 labs/pmpp.py run X3.1 --backend emu</code>. Its thread orders are fixed, shared memory starts as
          the byte <code>0xCD</code>, and AddressSanitizer stops the run at the first read past an allocation and names the
          line. The cases that read past an allocation come last.
        </li>
        <li>
          Fix one thing and run again. Then run <code>python3 labs/pmpp.py run X3.1 --backend race</code>.
        </li>
        <li>
          Make the same fixes in <code>kernel.metal</code> and <code>launch.mm</code> and run on your GPU:{' '}
          <code>python3 labs/pmpp.py run X3.1</code>. Note which defects it showed you and which it hid.
        </li>
      </ol>
      <Predict
        id="x3/labs/p-lab1-port-emu"
        unit="defects"
        prompt={<>After reading the port and before running it: of its three defects, how many will <code>emu</code> report in the code as shipped?</>}
      >
        {(c) => <PortReveal committed={c} />}
      </Predict>
      <p>
        Whatever your GPU showed, read it against what the vendors document. Shared or threadgroup memory nobody wrote holds
        whatever it holds, and no facts file we checked says what a GPU puts there. A case that passed every run on your GPU
        showed you that device's schedules, not a correct kernel.
      </p>

      <h3>What the tests check</h3>
      <ul>
        <li>
          RMSNorm: rows 1, 3 and 17 by d 1, 7, 64, 128, 576 (SmolLM2-135M's width), 1,000 and 4,097, in place and out of
          place, rows of different scales in one batch, an all-zero row whose output must be 0, and eps 1e-5. Outputs
          start as NaN, guard values after <code>y</code> must be untouched, and the inputs must come back unchanged.
        </li>
        <li>
          RoPE: positions that include 0 and the last table row; 1, 2 and 4 heads; head sizes 8, 64 and 128; both modes
          against the reference. Position 0 must return the input exactly, the two modes must differ, and llama.cpp's row
          permutation must turn mode 1 on (W x) into mode 0 on (P W x), for Q and for K under grouped-query attention.
          Permuting K with the query head count must break it.
        </li>
        <li>SwiGLU: n 1, 31, 1,000 and 4,099, out of place and in place, with gates of −100 and +100 giving finite results.</li>
        <li>
          The port: rows 1 and 3 (one row per block) and more rows than blocks; d 64 and 4,096, and d 577, 1,001 and 4,097,
          which are not multiples of 4. On devices and on <code>race</code> the inputs are followed by a guard band of
          1000.0, so a read past their end changes the result.
        </li>
      </ul>
      <p>
        Every result is compared with a double-precision CPU reference within k × FLT_EPSILON × (1 + |ref|), with k = 32
        for RMSNorm, 8 for RoPE and 16 for SwiGLU. The README derives each k from the length of the reduction and the
        documented error of each operation; it is the same on every backend.
      </p>
      <RaceLine lab={l1} />

      <h3>Before you benchmark</h3>
      <p>
        One call at rows 1 and d 4,096 moves about 48 KiB: the row in, the row out and the weight. Decide what fraction of
        your lab 5.1 bandwidth that reaches, and then what fraction 256 rows reach. Weigh three forces: one row is one
        block, and one block runs on one of the GPU's cores while the rest wait; every launch costs a fixed time whatever
        it moves; and the benchmark repeats the call on the same buffers, so data that fits in the caches need not come
        from memory after the first call, where lab 5.1 streamed far more than any cache holds.
      </p>
      <LabPrediction
        id="x3/labs/p-lab1-rmsnorm-bw"
        prompt={<>At rows = 1, d = 4,096, what percentage of your lab 5.1 bandwidth will <code>launchRmsNorm</code> reach?</>}
        unit="% of lab 5.1's bandwidth"
        measure={rmsFraction(1)}
        missing={missingBench('X3.1', NEED_51)}
        reading={
          <>
            The GB/s are counted, not measured: the bytes the call must move, once, divided by its time. A small
            percentage says the call is too short to be about bandwidth.
          </>
        }
      />
      <LabPrediction
        id="x3/labs/p-lab1-rmsnorm-bw-256"
        prompt={<>And at rows = 256, d = 4,096?</>}
        unit="% of lab 5.1's bandwidth"
        measure={rmsFraction(256)}
        missing={missingBench('X3.1', NEED_51)}
        reading={
          <>
            Above 100 % means the counted bytes did not all come from memory: 256 rows of 4,096 floats, in and out, are
            small enough for caches to serve a repeated call. Lab 5.1's figure is for a working set no cache holds.
          </>
        }
      />
      <LabResults lab={l1} cols={[{ head: 'Launch', cell: (r) => KERNEL_1[r.name] ?? r.name }, rawCol('Rows', 'rows'), rawCol('d', 'd'), headsCol, usCol, numCol('Counted GB/s', 'gb_per_s'), pctOf51]} />
      <p>
        How the benchmark works: each kernel at rows (tokens) 1, 16 and 256 for d 576 and 4,096. RoPE runs over heads of
        128 where d is a multiple of 128, else heads of 64 (576 is 9 × 64, SmolLM2-135M's shape); SwiGLU over rows × 4d
        elements. The counted bytes are each call's unique reads and writes, once. Each time is the best of 5 runs after an
        untimed one, each run repeating the call until it lasts at least 2 ms, and every timed result is checked first.
      </p>
      <ReferenceSolution lab={l1} />

      {/* ------------------------------------------------------------ X3.2 */}
      <h2>Lab X3.2: milestone M10, a Llama-style model end to end on the course's kernels</h2>
      <p>
        Every kernel of a Llama decode step now exists in this course. This lab composes them into one greedy step and runs
        a model that a script in the repository writes from a fixed seed. You write no arithmetic kernel. You decide the
        order of the pieces, their arguments and where each result lives, which is most of what a framework's model file
        does, and one small kernel that keeps the generation loop on the device.
      </p>
      <ul>
        <li>
          <code>launchLlamaStep(m, s)</code> (<code>encodeLlamaStep</code> on Metal, every dispatch in one compute encoder
          and one command buffer per step): one step for the whole batch, one launch per operation, nothing that waits for
          the GPU or copies to the host.
        </li>
        <li>
          <code>launchAdvancePositions</code> (<code>advance_positions</code> on Metal): one thread per sequence, at the end
          of each step. It records the step's argmax, feeds the next prompt token while the sequence is still in its prompt,
          and moves the position on. Before you write it, say what the host would have to do between two steps without
          it.
        </li>
      </ul>
      <LaunchTable />
      <p>
        Two questions to settle before you run it: which length must the attention see so that it reads the row you just
        appended and nothing after it, and where must the next tokens go so that the next step starts on the device?{' '}
        <code>llama.h</code> declares every piece with its argument order and every buffer's shape.
      </p>
      <p>
        <code>provided/make_model.py</code> writes the model in llama2.c's version 0 <code>.bin</code> format, the one
        section <a href={href('x3', 'X3.8')}>X3.8</a> decodes: a 28-byte header, then float32 weights in llama2.c's order.
        It uses the Python standard library only, with its own generator and little-endian packing, so every machine writes
        the same bytes, and CMake runs it at configure time. Nothing is downloaded by CMake, the script, the driver or the
        tests. The prompt goes through the step one token at a time, as llama2.c processes it: M6's prefill attention is not
        composed, because its layout is not the cache layout M7 reads.
      </p>
      <Beyond>
        <p>
          Ours: the model is llama2.c's <code>forward</code> (RMSNorm with eps 1e-5, grouped-query attention with RoPE base
          10,000, a SiLU-gated feed-forward, the classifier tied to the embedding) with head size 128, the size M7's decode
          kernels are compiled for. Its weights are uniform values with scales we chose so that greedy steps have large
          top-two margins. They are not a trained model.
        </p>
      </Beyond>
      <LabFiles
        lab={l2}
        files={[
          ['kernel.cu', <><code>advancePositionsKernel</code>, <code>launchAdvancePositions</code> and <code>launchLlamaStep</code></>],
          ['kernel.metal', <><code>advance_positions</code></>],
          ['launch.mm', <><code>encodeAdvancePositions</code> and <code>encodeLlamaStep</code></>],
        ]}
      />

      <h3>What the tests check</h3>
      <p>
        Run <code>python3 labs/pmpp.py run X3.2 --backend emu</code> first: it runs every thread of every launch of every
        step in three orders, so expect minutes. The driver prints the shapes: on the emulators d 256 with 2 query heads
        over 1 KV head, hidden 352, 2 layers, vocabulary 300; on a GPU d 512 with 4 over 2, hidden 1,376, 4 layers,
        vocabulary 4,099 and 60 generated tokens. Every size differs from every other, so a size passed in the wrong
        argument fails.
      </p>
      <ul>
        <li>
          <b>Model files.</b> The file's size equals section X3.8's formula for its header, and a checksum of its first and
          last weights equals what the script printed, so a generator that wrote other bytes on another machine fails here.
        </li>
        <li>
          <b>Pieces alone.</b> Each provided piece once against a CPU reference before your code runs, then your advance
          alone.
        </li>
        <li>
          <b>Teacher-forced logits.</b> Both sides are fed the reference's token at every step, so every step's logits can be
          compared. The driver prints each step's NMSE, llama.cpp's metric, and its largest error. A logit passes within
          4e-5 × (1 + mag), where mag is the size of the terms of the LM head's dot product: one tolerance for every
          backend, the README's.
        </li>
        <li>
          <b>Free-running greedy.</b> The tokens must equal the reference's at every step whose top-two margin exceeds the
          two logits' tolerances added. A step below that makes the rest "inconclusive from step k", printed, never passed
          silently. Section <a href={href('x3', 'X3.9')}>X3.9</a> shows why a flip there changes every later token.
        </li>
        <li>
          <b>Batch invariance.</b> Each sequence's tokens in a batch equal its tokens decoded alone.
        </li>
        <li>
          <b>RoPE pairing.</b> The model written with <code>--permute-qk</code>, llama.cpp's row permutation, run in mode 0,
          gives what the original gives in mode 1.
        </li>
        <li>
          <b>The sampler.</b> On a GPU, when lab X1.3's solution is present: M5 at temperature 0 gives the greedy tokens, and
          two runs at a fixed seed and temperature 0.8 give identical tokens. Sampled tokens are not compared across
          backends here.
        </li>
        <li>
          <b>Poison and guards.</b> Activations, logits and every cache row start as NaN, so a value nobody wrote, or a
          cache row the attention should not read, turns the logits to NaN; guard bands follow every output.
        </li>
      </ul>
      <Today checked="2026-11-20 in the facts pass (notes/today/x3-facts.md D1, D7, D9)">
        <p>
          llama.cpp's <code>test-backend-ops</code> compares each backend with the CPU by NMSE, 1e-7 per op by default.
          llama2.c's greedy step keeps the first maximum, and so does M4.
        </p>
      </Today>

      <h3>Before you benchmark</h3>
      <p>
        The benchmark prints the weight bytes per token and lab 5.1's bandwidth side by side; their quotient is the most
        tokens per second the weights alone allow at batch 1. Decide what fraction of that ceiling one step reaches, and,
        once you know the attention's share of the step at position 16, what it becomes at position 512. Weigh four
        forces. At batch 1 every projection is a matrix-vector product, and lab 22.1's gather reads each weight once where
        lab 6.4's tiles were built for matrices (section <a href={href('x3', 'X3.4')}>X3.4</a> shows llama.cpp switching
        kernels at 8 tokens). A step is sixteen launches per layer plus five, each short at batch 1. The weights read per
        token are fixed, while the cache the attention reads grows with the position, about 30 times from position 16 to
        512. And a small model's weights may sit partly in the device's caches.
      </p>
      <LabPrediction
        id="x3/labs/p-lab2-bwfraction"
        prompt={<>At batch 1, position 16, what percentage of the weight-bandwidth ceiling will one decode step reach in tokens per second?</>}
        unit="% of the ceiling"
        measure={ceilingFraction}
        missing={missingBench('X3.2', NEED_51)}
        reading={
          <>
            The ceiling assumes every weight comes from memory once per step at lab 5.1's rate and nothing else costs time.
            The step rows below show which matrix-product kernel the figure is for and what the others reached.
          </>
        }
      />
      <LabPrediction
        id="x3/labs/p-lab2-attn-share"
        prompt={<>At batch 1, position 512, what percentage of the summed per-piece time will attention (the append and the attention kernel) take?</>}
        unit="% of the pieces' summed time"
        measure={attentionShare}
        missing={missingBench('X3.2')}
        reading={
          <>
            The share is of the four groups' summed times, each piece timed alone in its own launch or command buffer, so
            the sum can exceed a step. Compare it with the share at 16 in the detail: the cache grew about 30 times, the
            weights not at all.
          </>
        }
      />
      <LabResults lab={l2} cols={STEP_COLS} filter={(r) => r.name === 'step'} />
      <ModelLine />
      <p>Each piece timed alone, times how often a step runs it, summed into four groups:</p>
      <PieceTable />
      <p>
        How the benchmark works: the same script writes a larger model at bench time (d 2,048 with 16 query heads over 4
        KV heads, hidden 5,632, 8 layers, vocabulary 32,000: our shape), halving d, the heads and the hidden width until the
        weights take at most a quarter of the memory the device reports. Its values repeat one random block, so the file
        is written in seconds and its tokens mean nothing. Steps run at batch 1 and 4 and positions 16 and 512, the cache
        rows before the position filled as after a prompt. Metal times one command buffer per step, CUDA and HIP one pair
        of events; each figure is the best of several runs after an untimed one.
      </p>

      <h3>A real checkpoint you download yourself</h3>
      <p>
        The loader reads llama2.c's format, so the first real checkpoint to try is one of Karpathy's tinyllamas, and its
        size check reproduces both files' sizes from their headers. They will not run as they are: their head sizes are 8
        and 48, and M7's decode kernels are compiled for 128, so the attention must first take the head size as a
        parameter. A Hugging Face Llama such as SmolLM2-135M needs the four conversions of section{' '}
        <a href={href('x3', 'X3.8')}>X3.8</a> on top. None of this was run when the lab was written: the README describes
        the work, unverified.
      </p>
      <Today checked="2026-11-20 in the facts pass (notes/today/x3-facts.md D2, D4, E2)">
        <p>
          stories260K: head size 8, 8 query heads over 4. stories15M: 48. SmolLM2-135M: 64, bfloat16, the NeoX pairing.
          None has 128.
        </p>
      </Today>

      <h3>The project milestone</h3>
      <p>
        <code>launchLlamaStep</code> completes milestone M10: a Llama-style decoder runs end to end on the course's own
        kernels, the whole generation loop stays on the device, and every logit and every decisive token is checked
        against a CPU reference. It follows the project's conventions: row-major float32 with the batch first, raw device
        pointers and explicit sizes, a <code>launch</code> host function beside each kernel, and the CPU reference in{' '}
        <code>main.cpp</code>. The plan's M10 names a small real checkpoint; the lab runs a generated one, because no small
        real checkpoint has head size 128 and the build downloads nothing.
      </p>
      <p>
        M10 runs float32 weights without M9. The <code>LinearFn</code> indirection in <code>llama.h</code> is where a
        quantised linear layer would plug in, but lab X2.4's <code>launchQLinear</code> takes a quantised matrix descriptor,
        not <code>LinearFn</code>'s float pointer and sizes, so <code>pieces.cmake</code> adds no M9 row. Making the two
        meet needs an adapter that is not built yet.
      </p>
      <ReferenceSolution lab={l2} />
    </LabsPage>
  );
}
