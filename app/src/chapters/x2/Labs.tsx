import { useState, type ReactNode } from 'react';
import { Predict, type Question } from '../../components/Ask';
import { Beyond, Source, Today, type Tier } from '../../components/notes';
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
import { fmt, labStatus, num, realBackends, type BenchRow, type Reports, type RunResult } from '../../lib/reports';
import { href } from '../../lib/router';
import { byteRatio, measuredBandwidth } from '../../sim/qtraffic';

// Module X2's labs page, built from components/LabKit.tsx. No graded question here: the labs
// are the exercise. Its nine predictions are stored, not graded: x2/labs/p-lab1-speedup,
// p-lab1-batch8, p-lab2-fused, p-lab2-unfused, p-lab3-emu, p-lab3-path, p-lab4-speedup,
// p-lab4-head, p-lab4-agree. pmpp.py knows these labs as X2.1 to X2.4.
export const questionsLabs: Record<string, Question> = {};

const FACTS = 'notes/today/x2-facts.md';

const LABS = defineLabs('x2', [
  { num: 'X2.1', dir: 'lab1_q8gemv', name: 'int8-weight GEMV' },
  { num: 'X2.2', dir: 'lab2_q4gemv', name: 'GEMV on Q4_0 blocks' },
  { num: 'X2.3', dir: 'lab3_qhunt', name: 'integer-path bug hunt' },
  { num: 'X2.4', dir: 'lab4_m9', name: 'the M9 decode step' },
]);
const [l1, l2, l3, l4] = LABS;

/** Identifier colours in the reference solutions: where each value lives. */
const tiers = (global: string[], shared: string[], reg: string[]): Record<string, Tier> => ({
  ...Object.fromEntries(global.map((k) => [k, 'global' as Tier])),
  ...Object.fromEntries(shared.map((k) => [k, 'shared' as Tier])),
  ...Object.fromEntries(reg.map((k) => [k, 'reg' as Tier])),
});
const TIERS_1 = tiers(['X', 'Wq', 'Wd', 'Y', 'q', 'd', 'x'], [], ['sum', 'four', 'scale', 'q0', 'q1', 'q2', 'q3', 'total', 'lane', 'j']);
const TIERS_2 = tiers(['X', 'W', 'Y', 'row', 'qs16', 'x'], [], ['sum', 'four', 'lo', 'hi', 'qx', 'sx', 'total', 'lane', 'j', 's']);
const TIERS_3 = tiers(['W', 'Xq', 'Y', 'X', 'w', 'q16', 'xb', 'u'], ['partial'], ['sum', 'sumi', 'v', 'vi0', 'vi1', 'dw', 't', 'lane', 'warp']);
const TIERS_4 = tiers(['E', 'q8', 'q8d', 'q4', 'tokens', 'H', 'blk'], [], ['i', 'r', 'j', 'k', 't', 'code']);

// ---------------------------------------------------------------- reading the reports
const ms = (s: number | undefined) => (s ? `${fmt(s * 1000)} ms` : '?');
const secs = (r: BenchRow | undefined) => num(r?.seconds);
/** The first row of a set with this name whose fields equal `where`, strings and numbers alike. */
const row = benchRow;
const qgemv = (s: BenchSet, kernel: string, batch: number, inW = 4096, outW = 4096) => row(s, 'qgemv', { kernel, batch, in: inW, out: outW });
const qstep = (s: BenchSet, layers: string, head: string, batch: number) => row(s, 'qstep', { format_layers: layers, format_head: head, batch });
/** Lab 5.1's bandwidth on a backend, as a percentage of which a row's weight GB/s is read. */
const ofBandwidth = (reports: Reports, backend: string, gbs: number | undefined) => {
  const bw = measuredBandwidth(reports, backend);
  return bw && gbs ? (100 * gbs) / bw : undefined;
};
const BYTES_Q8 = byteRatio('q8_0');
const BYTES_Q4 = byteRatio('q4_0');
const BYTES_Q4_Q8 = byteRatio('q4_0', 'q8_0');

/** Lab X2.1: Q8_0 over F32 at batch 1, 4096 to 4096, with lab 5.1's bandwidth alongside. */
const q8Speedup = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const f = qgemv(s, 'gather', 1);
    const q = qgemv(s, 'q8_0', 1);
    if (!secs(f) || !secs(q)) return null;
    const pct = ofBandwidth(reports, s.backend, num(q!.weight_gb_per_s));
    return {
      value: secs(f)! / secs(q)!,
      detail:
        `F32 ${ms(secs(f))} against Q8_0 ${ms(secs(q))} per launch, 4096 to 4096; the byte ratio is ${fmt(BYTES_Q8)}` +
        (pct != null ? `; Q8_0 read its weights at ${fmt(pct)} % of lab 5.1's bandwidth` : '; no lab 5.1 run to compare its bandwidth with'),
    };
  });

/** Lab X2.1: the same speed-up at batch 8, with batch 1's beside it. */
const q8Batch8 = (reports: Reports) =>
  measure(reports, l1, (s) => {
    const at = (b: number) => {
      const f = secs(qgemv(s, 'gather', b));
      const q = secs(qgemv(s, 'q8_0', b));
      return f && q ? f / q : undefined;
    };
    const eight = at(8);
    if (eight == null) return null;
    const one = at(1);
    return { value: eight, detail: `4096 to 4096 at batch 8` + (one != null ? `; at batch 1 it was ${fmt(one)}` : '; no batch-1 row') };
  });

/** Lab X2.2: fused Q4_0 over F32 at batch 1, 4096 to 4096, and over Q8_0 where that row exists. */
const q4Fused = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const f = secs(qgemv(s, 'gather', 1));
    const q4 = secs(qgemv(s, 'q4_0', 1));
    if (!f || !q4) return null;
    const q8 = secs(qgemv(s, 'q8_0', 1));
    return {
      value: f / q4,
      detail:
        `F32 ${ms(f)} against fused Q4_0 ${ms(q4)}, 4096 to 4096; the byte ratio is ${fmt(BYTES_Q4)}` +
        (q8 ? `; over lab X2.1's Q8_0 it is ${fmt(q8 / q4)}, against a byte ratio of ${fmt(BYTES_Q4_Q8)}` : '; no Q8_0 row'),
    };
  });

/** Lab X2.2: the provided unfused path over F32 at batch 1, 4096 to 4096. */
const q4Unfused = (reports: Reports) =>
  measure(reports, l2, (s) => {
    const f = secs(qgemv(s, 'gather', 1));
    const u = secs(qgemv(s, 'q4_0_unfused', 1));
    if (!f || !u) return null;
    const q4 = secs(qgemv(s, 'q4_0', 1));
    return {
      value: f / u,
      detail: `F32 ${ms(f)} against dequantise-then-gather ${ms(u)}, 4096 to 4096` + (q4 ? `; the fused kernel took ${ms(q4)}` : ''),
    };
  });

/** Lab X2.3: the integer path's time over the float path's, quantise pass included. */
const pathRatio = (reports: Reports) =>
  measure(reports, l3, (s) => {
    const at = (b: number) => {
      const i = row(s, 'qhunt', { path: 'int8_q8_1', batch: b, in: 4096, out: 4096 });
      const f = row(s, 'qhunt', { path: 'float_q4_0', batch: b, in: 4096, out: 4096 });
      const ti = num(i?.seconds_total);
      const tf = num(f?.seconds_total);
      return ti && tf ? { ti, tf, quant: num(i?.seconds_quantise) } : null;
    };
    const one = at(1);
    if (!one) return null;
    const eight = at(8);
    return {
      value: one.ti / one.tf,
      detail:
        `integer path ${ms(one.ti)}` + (one.quant != null ? ` (of which the quantise pass ${ms(one.quant)})` : '') +
        ` against the float path ${ms(one.tf)}, 4096 to 4096, batch 1` +
        (eight ? `; at batch 8 the ratio is ${fmt(eight.ti / eight.tf)}` : '; no batch-8 rows'),
    };
  });

/** Lab X2.4: a format pair's step over the F32 step at batch 1. */
const stepSpeedup = (layers: string, head: string, model: string) => (reports: Reports) =>
  measure(reports, l4, (s) => {
    const f = secs(qstep(s, 'f32', 'f32', 1));
    const q = qstep(s, layers, head, 1);
    if (!f || !secs(q)) return null;
    const pct = ofBandwidth(reports, s.backend, num(q!.weight_gb_per_s));
    return {
      value: f / secs(q)!,
      detail:
        `F32 step ${ms(f)} against ${ms(secs(q))}, d ${String(q!.d ?? '?')}, vocab ${num(q!.vocab)?.toLocaleString('en-US') ?? '?'}, batch 1; the bytes model says ${model}` +
        (pct != null ? `; the step read its weights at ${fmt(pct)} % of lab 5.1's bandwidth` : ''),
    };
  });

/** Lab X2.4: greedy steps in agreement with F32, at Q4_0, with Q8_0's alongside. */
const agreement = (reports: Reports) =>
  measure(reports, l4, (s) => {
    const q4 = qstep(s, 'q4_0', 'q4_0', 1);
    const a4 = num(q4?.agree_steps);
    if (a4 == null) return null;
    const q8 = qstep(s, 'q8_0', 'q8_0', 1);
    const a8 = num(q8?.agree_steps);
    return {
      value: a4,
      detail:
        `of 32 steps at batch 1, Q4_0 everywhere; first-step logits off by ${fmt(100 * (num(q4!.logits_rel_rms) ?? NaN))} % relative RMS` +
        (a8 != null ? `; Q8_0 agreed for ${a8} steps, its logits off by ${fmt(100 * (num(q8!.logits_rel_rms) ?? NaN))} %` : ''),
    };
  });

const missingBench = (labNum: string) => (
  <>
    This needs the benchmark rows of lab {labNum} from a real device. Run{' '}
    <code>python3 labs/pmpp.py run {labNum} --bench</code> once your kernels pass (or with <code>--solution</code> to time
    the reference). The emulator is never benchmarked.
  </>
);

// ---------------------------------------------------------------- results tables
const KERNEL: Record<string, string> = {
  gather: 'F32 gather (lab 22.1)',
  q8_0: 'Q8_0 (lab X2.1)',
  q4_0_unfused: 'Q4_0, dequantise then gather (provided)',
  q4_0: 'Q4_0 fused (lab X2.2)',
};
const PATH: Record<string, string> = {
  int8_q8_1: 'quantise to Q8_1, then the integer GEMV',
  float_q4_0: 'fused float Q4_0 GEMV',
};
const FMT: Record<string, string> = { f32: 'F32', q8_0: 'Q8_0', q4_0: 'Q4_0' };
const kernelCol: Col = { head: 'What ran', cell: (r) => KERNEL[String(r.kernel)] ?? String(r.kernel ?? '') };
const pathCol: Col = {
  head: 'What ran',
  cell: (r) => (r.path === 'float_q4_0' ? `${PATH.float_q4_0}, ${String(r.source ?? '')}` : PATH[String(r.path)] ?? String(r.path ?? '')),
};
const pairCol: Col = { head: 'Layers / head', cell: (r) => `${FMT[String(r.format_layers)] ?? r.format_layers} / ${FMT[String(r.format_head)] ?? r.format_head}` };
const msOf = (head: string, key: string): Col => ({ head, num: true, cell: (r) => (num(r[key]) == null ? '' : fmt(num(r[key])! * 1000)) });
const mbCol: Col = { head: 'Weights, MB', num: true, cell: (r) => (num(r.weight_bytes) == null ? '' : fmt(num(r.weight_bytes)! / 1e6)) };
const pctCol = (head: string, key: string): Col => ({ head, num: true, cell: (r) => (num(r[key]) == null ? '' : fmt(num(r[key])! * 100)) });

/** "% of lab 5.1": a row's weight GB/s over lab 5.1's measured bandwidth on the row's backend. */
function useBandwidthCol(): Col {
  const { reports } = useLabs();
  return {
    head: '% of lab 5.1',
    num: true,
    cell: (r) => {
      const p = ofBandwidth(reports, String(r.backend), num(r.weight_gb_per_s));
      return p == null ? '' : fmt(p);
    },
  };
}

function GemvResults({ lab }: { lab: Lab }) {
  const bw = useBandwidthCol();
  return (
    <LabResults
      lab={lab}
      cols={[kernelCol, rawCol('Batch', 'batch'), rawCol('In', 'in'), rawCol('Out', 'out'), msCol, numCol('Weight GB/s', 'weight_gb_per_s'), bw, numCol('vs F32', 'speedup_vs_f32')]}
    />
  );
}
function StepResults() {
  const bw = useBandwidthCol();
  return (
    <LabResults
      lab={l4}
      cols={[pairCol, rawCol('Batch', 'batch'), msCol, numCol('Tokens/s', 'tokens_per_s'), mbCol, numCol('Weight GB/s', 'weight_gb_per_s'), bw, numCol('vs F32', 'speedup_vs_f32'), pctCol('Logits error, %', 'logits_rel_rms'), rawCol('Agree', 'agree_steps')]}
    />
  );
}

/** Lab 5.1's measured bandwidth on each device: the 100 % of the "% of lab 5.1" columns. */
function BandwidthLine() {
  const { reports, loading } = useLabs();
  const found = realBackends(reports).flatMap((b) => {
    const bw = measuredBandwidth(reports, b);
    return bw ? [{ b, bw }] : [];
  });
  return (
    <p data-measured={loading ? 'loading' : found.length ? found.map((f) => `${f.b}:${fmt(f.bw)}`).join(',') : 'none'}>
      {loading
        ? 'Reading reports.'
        : found.length
          ? `On this machine, lab 5.1 measured ${found.map((f) => `${fmt(f.bw)} GB/s on ${f.b}`).join(' and ')}. That is the 100 % of the "% of lab 5.1" columns below.`
          : 'No run of lab 5.1 on this machine yet, so the "% of lab 5.1" columns stay empty. Run python3 labs/pmpp.py run 5.1 --bench first.'}
    </p>
  );
}

// ---------------------------------------------------------------- the race backend, for lab X2.3
/** One line on lab X2.3's runs on the race backend, which the status table does not show. */
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

/** Lab X2.3's first prediction: how many of the four defects emu reports. Not measured from the reports. */
function HuntReveal({ committed }: { committed: number }): ReactNode {
  // The count and the reason come at once; which defect showed where waits until the reader
  // says the hunt is over, so the reveal does not hand over the answers before the first run.
  const [fixed, setFixed] = useState(false);
  return (
    <div data-reveal="p-lab3-emu">
      <p>
        When this lab was built, each defect was planted alone into the reference solution, and <code>emu</code> reported{' '}
        <b>4</b> of the four. {committed === 4 ? 'That is your number.' : `You predicted ${committed}.`}
      </p>
      <p>
        Two of them show only on particular inputs or thread orders, which is why the tests hold the cases they do. One
        is a defect in integer arithmetic that only one constructed block reaches; take that case away and it passes
        everywhere. Another is a synchronisation defect: <code>emu</code> fills shared memory with a byte pattern, so a
        read that comes too early returns an absurd value in some of its thread orders. The race backend named the two
        lines of that one, and the Metal device the lab was built on passed it on every case.
      </p>
      {fixed ? (
        <p data-reveal-detail="p-lab3-emu">
          Defect by defect, each alone on <code>emu</code>: defect 1 (activation words 2i and 2i + 1) failed every
          integer-path case except the worst-case block, whose weights are all equal, so any pairing gives the same sum.
          Defect 2 (the <code>short</code> block sum) failed only the worst-case block, whose sum of 60,960 is past a
          short's 32,767. Defect 3 (8 times d where Q8_1's s belongs) failed every integer-path case. Defect 4 (the missing
          barrier) failed under the in-order and shuffled thread orders; race reported the write of{' '}
          <code>partial[warp][b]</code> and the read of <code>partial[lane][b]</code>; Metal passed all 8 cases.
        </p>
      ) : (
        <p>
          <button type="button" onClick={() => setFixed(true)}>
            I have fixed all four: show what each backend saw
          </button>
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- the page
export function Labs() {
  return (
    <LabsPage labs={LABS}>
      <LabsIntro>
        Four labs, each in its own directory under <code>labs/x2/</code>. The first stores a decode GEMV's weights as
        int8 codes with a scale per block of 32 and measures it against lab 22.1's float32 gather. The second reads
        llama.cpp's own 18-byte Q4_0 blocks. The third hunts four defects in llama.cpp's integer path. The fourth is
        milestone M9: lab 22.2's decode step with quantised weights, its speed-up and its error measured on your device.
      </LabsIntro>
      <Beyond title="Ours, not the book's">
        <p>
          The book has no integer quantisation and no quantised kernel (appendix A's §A.5 is two paragraphs). The formats
          are llama.cpp's; the contracts, layouts of lab X2.1, tests and defects are ours. The parts are the book's: chapter
          5's intensity, chapter 10's warp reduction, chapter 22's gather.
        </p>
      </Beyond>
      <Source section="A.5" pages="592–593" />
      <p>
        This module is not a chapter of the book, but <code>pmpp.py</code> knows its labs by their numbers:{' '}
        <code>python3 labs/pmpp.py run X2.1</code> runs lab X2.1, and so on to X2.4.
      </p>
      <RunningLabs
        example="X2.1"
        alsoShownIn={
          <>
            the measured panels of <a href={href('x2', 'X2.1')}>X2.1</a> and <a href={href('x2', 'X2.7')}>X2.7</a>
          </>
        }
      />
      <BackendsAndEmulator />
      <p>
        Lab X2.3 is also built for the emulator's race flavour, because its kernel hands per-warp partial sums to one warp
        through shared memory. <code>python3 labs/pmpp.py run X2.3 --backend race</code> runs a block's threads at once
        under ThreadSanitizer, which reports two accesses to one location with nothing ordering them, with both source
        lines. A clean run is evidence, not proof: it sees only the schedules that happened. The other three labs have no
        shared updates: every thread or warp writes its own outputs.
      </p>
      <p>
        One defect no emulator can see: a 4-byte load from an address that is not a multiple of 4. The host CPU allows it
        and AddressSanitizer does not check alignment, so it passes <code>emu</code>, <code>race</code> and perhaps your
        GPU, and it is undefined on every backend. Labs X2.2 and X2.3 read llama.cpp's 18-byte blocks, which are only
        2-byte aligned. Check those kernels by reading: every load from a block is 1 or 2 bytes wide.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${FACTS} A5, C5)`}>
        <p>
          CUDA: <code>char4</code> has size 4 and alignment 4. llama.cpp reads Q4_0 codes with two 16-bit loads, "assume at
          least 2 byte alignment". On Metal, <code>half</code> is 2-byte aligned.
        </p>
      </Today>
      <VerificationStatus />
      <p>
        Labs X2.3 and X2.4 call the GEMVs of labs X2.1 and X2.2. Their builds copy those labs' reference solutions into
        the build directory (<code>labs/x2/shared-g12/qpieces.cmake</code>, <code>labs/x2/lab4_m9/provided/pieces.cmake</code>),
        with any one-line helper made <code>static</code>, because every lab's file is called <code>kernel.cu</code> and
        two copies of one helper in one emulator program would clash. If a solution is missing or has another signature,
        a stand-in from <code>labs/x2/shared-g12/</code> takes its place, and the first line of the run says which one ran.
        The CPU reference quantisers live in <code>labs/x2/shared/qformat.h</code> (labs X2.1 and X2.2) and{' '}
        <code>labs/x2/shared-g12/qformats.h</code> (labs X2.3 and X2.4): two copies of llama.cpp's <code>_ref</code> rules,
        each checked on hand-derived blocks before any kernel runs.
      </p>
      <Beyond>
        <p>The copies, the stand-ins and the hand-derived fixture blocks are ours.</p>
      </Beyond>
      <h2>Bandwidth utilisation</h2>
      <p>
        A decode GEMV's speed limit is the device's bandwidth, so the useful question about a measured time is how close
        it came. Every benchmark row below reports its weight bytes over its time, and the tables divide that by what lab
        5.1 measured on the same device: PyTorch calls this model bandwidth utilisation. It counts only the weights, so it
        leaves out the activations, the outputs and any scratch traffic, and it can come out a little above what lab 5.1's
        simple copy reached.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${FACTS} A1)`}>
        <p>
          PyTorch's MBU: model size times inferences per second, divided by peak bandwidth. The course divides by lab
          5.1's measured bandwidth, not a datasheet peak.
        </p>
      </Today>
      <BandwidthLine />

      {/* ------------------------------------------------------------ X2.1 */}
      <h2>Lab X2.1: an int8-weight GEMV for batch-1 decode</h2>
      <p>
        Write <code>launchGemvQ8_0(X, Wq, Wd, Y, batch, in, out)</code>: Y[b][j] is the sum over k of X[b][k] times
        d[j][k / 32] times q[j][k], for batches up to 8. The rule is llama.cpp's Q8_0: per block of 32 weights, d is the
        largest magnitude over 127, each code is <code>roundf(w / d)</code>, and d is stored as FP16. The layout is ours:
        codes in one array <code>Wq [out × in]</code>, the FP16 scales as <code>uint16_t</code> bits in a second array{' '}
        <code>Wd [out × in/32]</code>, as MLX keeps its scales apart. Split that way, every row of codes is 4-byte aligned,
        and a lane can read four codes as one 32-bit word.
      </p>
      <Source section="A.5, 5.1, 22.2" pages="593, 93–97, 532–536" />
      <p>
        The mapping is lab 22.1's gather: one warp per output, the lanes striding along the row a word at a time, one
        register sum per batch row, each word's four products scaled by their block's d once, a warp sum, lane 0 writes.
        The only 32 in the kernel is <code>QK8_0</code>, the block size, and it has a name: the stride is{' '}
        <code>warpSize</code>. Lab X2.4 links this function, so its signature is fixed.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${FACTS} D1, D5)`}>
        <p>
          llama.cpp: <code>d = amax / ((1 &lt;&lt; 7) - 1)</code>, codes by <code>roundf</code>, d stored with{' '}
          <code>GGML_FP32_TO_FP16</code>. MLX stores scales and biases apart from its packed codes.
        </p>
      </Today>
      <LabFiles
        lab={l1}
        files={[
          ['kernel.cu', <><code>codeOf</code>, <code>gemvQ8_0Kernel</code> and <code>launchGemvQ8_0</code>; <code>warpSum</code> with its typed lane mask, <code>deviceWarpSize</code> and <code>halfBitsToFloat</code> are provided</>],
          ['kernel.metal', <><code>gemv_q8_0</code>, with the codes as <code>device const char4*</code> and the scales as <code>device const half*</code></>],
          ['launch.mm', <>the dispatch in <code>encodeGemvQ8_0</code>, into the encoder it is given; the buffer binding is provided</>],
        ]}
      />
      <h3>What the tests check</h3>
      <p>
        The driver's own parts first: its FP16 conversion on hand-derived values and on all 65,536 patterns, and its
        quantiser on a hand-derived block whose codes include the ties 2.5 and −2.5. Then shapes from (1, 32, 1) to (5,
        4096, 3) on the emulator, and up to (8, 4096, 4096) and (4, 14336, 4096) on a GPU. Each block of a row is scaled
        by its own power of two from 2⁻¹² to 2¹², so a scale read from the neighbouring block is at least 32 times wrong; every
        block holds codes 127 and −127; one row is all zero; the smallest scales are FP16 subnormals. Y starts as NaN with
        a guard band after it, every case launches twice, and results are compared with a double-precision dot on the
        dequantised weights within 1e-4 × (1 + Σ|x w|), never bit for bit.
      </p>
      <h3>Before you benchmark</h3>
      <p>
        The bytes model's ceiling is the ratio of bytes stored: 4 bytes per weight for F32 against 1.0625 for Q8_0, {fmt(BYTES_Q8)}.
        Section <a href={href('x2', 'X2.1')}>X2.1</a> says why that is a ceiling and not a promise. Decide how much of it
        survives on your device, then whether the ratio grows or shrinks at batch 8. Weigh two things: the activation
        bytes grow with the batch and are the same for both formats, and the work of turning a code into a float is done
        once per weight and shared by all the rows of the batch.
      </p>
      <LabPrediction
        id="x2/labs/p-lab1-speedup"
        prompt={<>At batch 1 on the 4096 to 4096 layer, how many times faster than lab 22.1's F32 gather will your Q8_0 kernel be?</>}
        unit="× the F32 gather's speed"
        measure={q8Speedup}
        missing={missingBench('X2.1')}
        reading={
          <>
            A measurement above the byte ratio does not mean the bytes model is wrong about bytes. The two kernels differ in
            more than their weights: this one reads X four floats at a time and lab 22.1's reads one. Read the bandwidth
            share first: it says how much room was left.
          </>
        }
      />
      <LabPrediction
        id="x2/labs/p-lab1-batch8"
        prompt={<>And at batch 8, same layer: how many times faster than F32?</>}
        unit="× the F32 gather's speed"
        measure={q8Batch8}
        missing={missingBench('X2.1')}
        reading={
          <>
            Both kernels read their weights once per launch whatever the batch, so a falling ratio says something other
            than the weights started to cost time: the activations, the arithmetic, or both. The table has batch 4 and
            the two MLP shapes.
          </>
        }
      />
      <GemvResults lab={l1} />
      <p>
        How the benchmark works: layers 4096 to 4096, 4096 to 14336 and 14336 to 4096 (the last two are our MLP shapes),
        batches 1, 4 and 8. One timed submission launches the kernel once per copy of its weights, copies that together
        hold at least 256 MB per format, so no launch finds its weights where the previous one left them. The two kernels
        take turns: one untimed round, then five, and each keeps its best. Metal times are the command buffer's GPU time;
        CUDA and HIP time on the host clock.
      </p>
      <ReferenceSolution lab={l1} tiers={TIERS_1} />

      {/* ------------------------------------------------------------ X2.2 */}
      <h2>Lab X2.2: a 4-bit GEMV on llama.cpp's Q4_0 blocks</h2>
      <p>
        Write <code>launchGemvQ4_0(X, W, Y, batch, in, out)</code> on llama.cpp's blocks exactly as{' '}
        <code>ggml-common.h</code> defines them: <code>struct BlockQ4_0 {'{'} uint16_t d; uint8_t qs[16]; {'}'}</code>, 18
        bytes, byte j holding code j in its low nibble and code j + 16 in its high nibble, each value (q − 8) × d. This is
        the format you meet in a GGUF file. W is <code>[out × in/32]</code> blocks; X and Y are float32, batch first.
      </p>
      <Source section="A.5, 22.2, 10.7" pages="592–593, 532–536, 237–241" />
      <p>
        Read the codes as bytes or 16-bit words, never through <code>int</code> or <code>char4</code>: section{' '}
        <a href={href('x2', 'X2.3')}>X2.3</a> shows which blocks would be misaligned. Unpack with masks. Folding the −8 out
        of the inner loop through the block's sum of x (section <a href={href('x2', 'X2.5')}>X2.5</a>) and giving a block
        several lanes (section <a href={href('x2', 'X2.4')}>X2.4</a>'s stepper) are your choices. The unfused path,
        dequantise W into a float32 scratch matrix and then run lab 22.1's gather, is provided and correct, for the
        benchmark only.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${FACTS} D1, C2)`}>
        <p>
          llama.cpp's <code>block_q4_0</code>: <code>ggml_half d</code> and <code>uint8_t qs[QK4_0 / 2]</code>. MSL's new{' '}
          <code>int4b_format</code> has no documented nibble order, and neither llama.cpp nor MLX uses it.
        </p>
      </Today>
      <LabFiles
        lab={l2}
        files={[
          ['kernel.cu', <><code>gemvQ4_0Kernel</code> and <code>launchGemvQ4_0</code>; the unfused baseline is in <code>provided/unfused.cu</code></>],
          ['kernel.metal', <><code>gemv_q4_0</code>, on <code>struct block_q4_0 {'{'} half d; uchar qs[16]; {'}'}</code> as llama.cpp declares it on Metal</>],
          ['launch.mm', <><code>encodeGemvQ4_0</code></>],
        ]}
      />
      <h3>What the tests check</h3>
      <p>
        The CPU quantiser first, on three hand-derived blocks: one whose extreme is negative (d = +1/8, bits{' '}
        <code>0x3000</code>), one whose extreme is positive (d = −1/4, <code>0xB400</code>), and an all-zero block, whose d
        is −0 (<code>0x8000</code>) with every byte <code>0x88</code>. Then lab X2.1's shapes and one more, (1, 32 × 33, 3):
        33 blocks per row, so rows start at both even and odd blocks. Each block has a positive and a negative peak, so
        both signs of d occur and the other peak clips to code 15; one block per row has its first 16 weights positive
        and its last 16 negative, so reading a byte as two neighbours fails; some blocks are all zero. The provided
        unfused path is tested too, so the benchmark compares correct kernels. Tolerance, NaN start, guard band and two
        launches as lab X2.1.
      </p>
      <h3>Before you benchmark</h3>
      <p>
        Three byte ratios: fused Q4_0 against F32 is 4 / 0.5625 = {fmt(BYTES_Q4)}; against Q8_0, 1.0625 / 0.5625 ={' '}
        {fmt(BYTES_Q4_Q8)}. The unfused path reads 0.5625 bytes per weight, writes 4 and reads 4 again, against F32's 4:
        the bytes model says it is about half as fast as not quantising at all (section{' '}
        <a href={href('x2', 'X2.1')}>X2.1</a>'s pipelines). Where the fused kernel falls short of its ratio, ask what it
        pays per weight that F32 does not: the unpacking, the scale, the −8.
      </p>
      <LabPrediction
        id="x2/labs/p-lab2-fused"
        prompt={<>At batch 1 on 4096 to 4096, how many times faster than the F32 gather will your fused Q4_0 kernel be?</>}
        unit="× the F32 gather's speed"
        measure={q4Fused}
        missing={missingBench('X2.2')}
        reading={
          <>
            The detail compares Q4_0 with lab X2.1's Q8_0 too. If Q4_0 keeps less of its ratio than Q8_0 did, the extra
            work per byte (two codes per byte, a −8, more scales per byte read) is starting to show.
          </>
        }
      />
      <LabPrediction
        id="x2/labs/p-lab2-unfused"
        prompt={<>And the provided unfused path, dequantise then gather: its speed as a multiple of F32's? (Below 1 means slower.)</>}
        unit="× the F32 gather's speed"
        measure={q4Unfused}
        missing={missingBench('X2.2')}
        reading={
          <>
            Its scratch matrix is not copied like the weights, so a cache may help it; the table's weight GB/s leaves that
            traffic out. Either way it is the case PyTorch measured: quantised weights help decode only when the
            dequantise is fused into the product.
          </>
        }
      />
      <GemvResults lab={l2} />
      <p>
        How the benchmark works: as lab X2.1, with four kernels taking turns: lab 22.1's F32 gather, lab X2.1's Q8_0
        solution when it is present, the provided unfused path and yours. The README's "go further" builds the prefill
        path, a tile of W dequantised into shared memory before lab 15.6's GEMM, which is how llama.cpp's{' '}
        <code>mul_mm</code> handles many tokens.
      </p>
      <ReferenceSolution lab={l2} tiers={TIERS_2} />

      {/* ------------------------------------------------------------ X2.3 */}
      <h2>Lab X2.3: bug hunt on the integer path</h2>
      <p>
        You are handed a Q4_0 × Q8_1 GEMV in llama.cpp's CUDA shape, in CUDA and in Metal. The activations are first
        quantised to Q8_1 (int8 codes per 32, a scale d and the precomputed s = d × Σq); each block is then an integer dot
        product of unsigned nibbles against signed bytes, rescaled once as d_w (sumi × d_x − 8 × s_x). One block of 4 warps
        per output row; the warps split the row's blocks; each warp reduces with shuffles and hands its partial sums to
        warp 0 through shared memory. It compiles, and it has four independent defects. Find and fix all four.
      </p>
      <Source section="10.7, 5.4" pages="237–241, 108–112" />
      <Today checked={`2026-11-20 in the facts pass (${FACTS} D3, A2, B1, C1)`}>
        <p>
          llama.cpp's <code>mmvq</code> quantises the activations to Q8_1 for every token. <code>__dp4a</code> exists on CUDA
          from sm_61; HIP has no <code>__dp4a</code>; MSL has no integer dot product.
        </p>
      </Today>
      <LabFiles
        lab={l3}
        files={[
          ['kernel.cu', <><code>gemvQ4_0Q8_1Kernel</code>, with the four defects; the quantiser, <code>dot4_i8</code> (which calls <code>__dp4a</code> only on CUDA sm_61 and later) and <code>launchGemvQ4_0Q8_1</code> are correct</>],
          ['kernel.metal', <><code>gemv_q4_0_q8_1</code>, with the same four</>],
          ['launch.mm', <>provided and correct; nothing to change</>],
        ]}
      />
      <p>
        The README lists what is not a defect: the 16-bit loads of the weights and the 32-bit loads of the 36-byte Q8_1
        blocks, the unsigned nibbles fed to a signed byte product, the typed lane mask and <code>warpSize</code>. The
        method matters more than the fixes:
      </p>
      <ol>
        <li>
          Before running anything, read <code>work/kernel.cu</code> and write down the defects you can see, and for each the
          backend you expect to report it: <code>emu</code>, <code>race</code>, your GPU, or none, and on which case.
        </li>
        <li>
          Run the emulator: <code>python3 labs/pmpp.py run X2.3 --backend emu</code>. Its thread orders are fixed, so what it
          reports, it reports every time. Use what the failing cases have in common: which shape, which kind of data,
          which thread order.
        </li>
        <li>
          Fix one defect at a time and run again. Then <code>python3 labs/pmpp.py run X2.3 --backend race</code>, which names
          the two lines that collide, whether or not a value came out wrong.
        </li>
        <li>
          Make the same fixes in <code>kernel.metal</code> and run on the device: <code>python3 labs/pmpp.py run X2.3</code>.
          Note which defects it showed you and which it hid.
        </li>
      </ol>
      <Predict
        id="x2/labs/p-lab3-emu"
        unit="defects"
        prompt={<>After reading the kernel and before running it: of the four defects, each planted alone, how many will <code>emu</code> report?</>}
      >
        {(c) => <HuntReveal committed={c} />}
      </Predict>
      <h3>What the tests check</h3>
      <p>
        Shapes (1, 32, 1), (1, 256, 3), (2, 512, 5) and (8, 1024, 9) with real weights through llama.cpp's Q4_0 rule.
        Three constructed cases: the worst-case block, every weight code 15 against activations that all quantise to 127,
        so every block's integer sum is 15 × 127 × 32 = 60,960 (section <a href={href('x2', 'X2.6')}>X2.6</a>); blocks whose
        low nibbles are 15 and high nibbles 0, or the other way round, so pairing a nibble with the wrong activation cannot
        pass; and power-of-two block scales. The reference is the same integer path on the CPU, exact integer block sums
        then the float rescale in double, within 1e-5 × (1 + the block magnitudes). Each case also prints, as information,
        how far the integer path is from the float path: that is the price of rounding the activations to int8.
      </p>
      <RaceLine lab={l3} />
      <p>
        Whatever your GPU showed, read it against what the vendors document: warps make independent progress between
        barriers, and reading and writing one location without synchronisation is undefined. A case that passed ten times
        on your GPU showed you ten schedules of one device, not a correct kernel.
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md A9; infra-facts.md C5; ch10-facts.md A1)">
        <p>
          NVIDIA: warps run independently. Apple: SIMD-groups "make independent forward progress ... in the absence of
          threadgroup barrier operations". CUDA: unsynchronised access to one location is undefined behaviour.
        </p>
      </Today>
      <h3>Before you benchmark</h3>
      <p>
        With the defects fixed, which path is faster for one token of decode, the quantise pass included: (a) quantise the
        activations to Q8_1, then this integer GEMV; or (b) lab X2.2's fused float GEMV? Both read the same Q4_0 weight
        bytes. Weigh what each backend gives the integer path: CUDA has a byte dot instruction from sm_61, the HIP build
        takes the plain four-product helper, and Metal has no integer dot at all, which is why llama.cpp's own Metal decode
        stays in float (section <a href={href('x2', 'X2.5')}>X2.5</a>).
      </p>
      <LabPrediction
        id="x2/labs/p-lab3-path"
        prompt={
          <>
            At batch 1 on 4096 to 4096, quantise pass included, what is the integer path's time divided by the float path's?
            (Below 1 means the integer path wins.)
          </>
        }
        unit="× the float path's time"
        measure={pathRatio}
        missing={missingBench('X2.3')}
        reading={
          <>
            This lab's mapping gives each thread whole blocks and reloads a block's activations for every row of the batch,
            so read the batch-8 ratio as a statement about this kernel, not about integer dot products. On Metal the
            "integer dot" is four byte products in ordinary instructions.
          </>
        }
      />
      <LabResults
        lab={l3}
        cols={[pathCol, rawCol('Batch', 'batch'), rawCol('In', 'in'), rawCol('Out', 'out'), msOf('Quantise, ms', 'seconds_quantise'), msOf('GEMV, ms', 'seconds_gemv'), msOf('Total, ms', 'seconds_total')]}
      />
      <p>
        How the benchmark works: 4096 to 4096 and 14336 to 4096, batches 1 and 8, random Q4_0 blocks. Successive timed runs
        read different copies of the weights, about 512 MB in all or a quarter of the device's memory if that is less: our
        choice, since Apple documents no cache sizes. Each figure is the best of at least 10 timed runs after an untimed
        one.
      </p>
      <ReferenceSolution lab={l3} tiers={TIERS_3} />

      {/* ------------------------------------------------------------ X2.4 */}
      <h2>Lab X2.4: milestone M9, the quantised decode step</h2>
      <p>
        Start from lab 22.2's decode step (M8) and store its weights in Q8_0 or Q4_0. Write three host functions.{' '}
        <code>launchEmbedQ(E, tokens, H, batch)</code> is a dequantising gather: one thread per element of H, reading its
        code and its block's scale, bit for bit the CPU's dequantised row. <code>launchQLinear(X, W, Y, batch)</code>{' '}
        dispatches on <code>W.fmt</code> to lab 22.1's <code>launchGemvGather</code>, lab X2.1's{' '}
        <code>launchGemvQ8_0</code> or lab X2.2's <code>launchGemvQ4_0</code>, for batches up to 8.{' '}
        <code>launchQDecodeStep(m, s)</code> is lab 22.2's step with those two in place of its embedding and its linear
        layers: embed, then per layer a linear layer, bias and residual, then the LM head with E, softmax and argmax, one
        launch per operation for the whole batch and nothing waiting.
      </p>
      <Source section="22.5, 20.6" pages="539–540, 504–506" />
      <p>
        The weights stay quantised in device memory. Nothing in the step writes a dequantised copy of a weight matrix:
        that would move more bytes than the float32 step does. The driver quantises lab 22.2's seeded toy model at load
        with llama.cpp's <code>_ref</code> rules, and runs four pairs of formats (layers, then the tied embedding and LM
        head): (F32, F32), (Q8_0, Q8_0), (Q4_0, Q4_0) and (Q4_0, F32).
      </p>
      <Today checked={`2026-11-20 (${FACTS} A1; notes/today/ch15-facts.md D1)`}>
        <p>
          PyTorch measured an unfused int8 path slower than FP16. llama.cpp uses matrix-vector kernels up to 8 tokens.
        </p>
      </Today>
      <LabFiles
        lab={l4}
        files={[
          ['kernel.cu', <>the two embedding kernels, <code>launchEmbedQ</code>, <code>launchQLinear</code> and <code>launchQDecodeStep</code>; <code>qdecode.h</code> declares <code>QMatrix</code> and <code>QDecodeModel</code></>],
          ['kernel.metal', <><code>embed_rows_q8_0</code> and <code>embed_rows_q4_0</code>; <code>embed_rows_f32</code> is provided</>],
          ['launch.mm', <><code>encodeEmbedQ</code>, <code>encodeQLinear</code> and <code>encodeQDecodeStep</code>, every dispatch in the caller's encoder: one command buffer per step</>],
        ]}
      />
      <h3>What the tests check</h3>
      <p>
        A toy model on the emulator: d 64, vocabulary 96, 2 layers, batches 1, 3 and 8, every format pair. The embedding
        alone first, for E in each format, on a batch holding the first token, the last and one token twice: every row of
        H bit for bit the CPU's. After every step, H and the logits against a CPU step in double on the dequantised
        weights, within lab 22.2's tolerance; the next token must equal the reference's wherever the reference's two best
        logits are further apart than their tolerances together. Then batch invariance: each sequence of the largest batch
        decodes exactly as it does alone. The weights are unchanged after the run, every output starts as NaN and has a
        guard band. On a GPU also lab 22.2's default shape, d 4096, 2 layers, vocabulary 32,000. The emulator run takes
        minutes.
      </p>
      <h3>Before you benchmark</h3>
      <p>
        Section <a href={href('x2', 'X2.7')}>X2.7</a>'s explorer computes the byte ratio of the whole step. With every
        matrix in Q4_0 it is close to the GEMV's {fmt(BYTES_Q4)}; with the 32,000 × 4,096 head left in F32 it is 1.21,
        because the head is most of the step's bytes. For the error, the toy model is random and has no attention: its
        agreement with F32 says nothing about a trained model's.
      </p>
      <LabPrediction
        id="x2/labs/p-lab4-speedup"
        prompt={<>At batch 1, how many times faster than the (F32, F32) step will the (Q4_0, Q4_0) step be?</>}
        unit="× the F32 step's speed"
        measure={stepSpeedup('q4_0', 'q4_0', `about ${fmt(BYTES_Q4)}`)}
        missing={missingBench('X2.4')}
        reading={
          <>
            A step is several launches, and the small ones (bias, residual, softmax, argmax) do not shrink with the weights.
            How much of the GEMV's speed-up from lab X2.2 survives says how much of the step they are.
          </>
        }
      />
      <LabPrediction
        id="x2/labs/p-lab4-head"
        prompt={<>And the (Q4_0, F32) step, with only the layers quantised: how many times faster than F32?</>}
        unit="× the F32 step's speed"
        measure={stepSpeedup('q4_0', 'f32', '1.21')}
        missing={missingBench('X2.4')}
        reading={<>If the head dominates the step's bytes, it dominates its time too, unless something else limits the step.</>}
      />
      <LabPrediction
        id="x2/labs/p-lab4-agree"
        prompt={<>Of 32 greedy steps from fixed start tokens, for how many will the Q4_0 step choose the same tokens as F32?</>}
        unit="steps"
        measure={agreement}
        missing={missingBench('X2.4')}
        reading={
          <>
            The count stops at the first step where any sequence chose another token. Read it beside the logits error: a
            random model's logits can be far apart, so a large error may still leave every argmax in place. Section{' '}
            <a href={href('x2', 'X2.7')}>X2.7</a>'s toy explorer shows the opposite case.
          </>
        }
      />
      <StepResults />
      <p>
        How the benchmark works: lab 22.2's default shape and seeds, batches 1, 2, 4 and 8, all three formats of every
        matrix on the device at once. One step is one command buffer on Metal and one pair of events on CUDA and HIP, best
        of several timed runs after an untimed one. Weight bytes count what one step must read: E once for the head, every
        layer's W and the biases. The logits error is the first step's relative RMS against F32; "Agree" is the number of
        greedy steps, from the first, in which every sequence chose F32's token.
      </p>

      <h3>The project milestone</h3>
      <p>
        <code>launchQDecodeStep</code> completes milestone M9: the decode step with int8 and 4-bit weights, its speed-up
        measured on your device and its output error reported against float32, by logits and by greedy tokens. It follows
        the project's conventions so later milestones can call it: row-major float32 activations with the batch first, raw
        device pointers and explicit sizes, a <code>launch</code> host function beside each kernel, the CPU reference in{' '}
        <code>main.cpp</code>, and the GEMVs of labs X2.1 and X2.2 with their fixed signatures. Module X3's milestone M10
        calls the reference solution's <code>launchQLinear</code> through a small adapter,{' '}
        <code>labs/x3/lab2_tinyllama/provided/m9_linear</code>, which binds each float weight pointer of its Llama model to a{' '}
        <code>QMatrix</code>: the same Llama step then runs with Q8_0 and Q4_0 weights, and its tokens are checked against a
        CPU reference on the same quantised weights.
      </p>
      <ReferenceSolution lab={l4} tiers={TIERS_4} />
    </LabsPage>
  );
}
