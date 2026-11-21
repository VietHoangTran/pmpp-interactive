import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { fmt, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BOOK_DEVICE,
  M8_DEFAULT,
  PROJ_1024,
  PROJ_4096,
  WEIGHT_BYTES,
  batchRange,
  bStar,
  bStarSearch,
  bookThroughput,
  intensity,
  kneeOf,
  m8Shape,
  measuredDevices,
  precisionFor,
  projectionSeries,
  stepAt,
  stepBStar,
  stepCurve,
  stepSeries,
  thresholdOf,
  weightBytes,
  type Device,
  type Padding,
  type Point,
  type ReportsLike,
  type Series,
  type Shape,
  type StepResult,
  type WeightBytes,
} from '../../sim/batching';
import { MIXES, SHORT_MAX, makeRequests, serve, type Mix, type Policy, type ServeResult } from '../../sim/serving';
import { ContinuousBatchingScene } from './ContinuousBatchingScene';
import { DecodeStepScene } from './DecodeStepScene';

// Section 22.5, Batching: latency versus throughput (book pp. 539–540). The book's part is
// the two motivations, the LLM batching argument with its two throughput formulas, and the
// CPU-versus-GPU remark. The roofline arithmetic (notes/today/ch22-facts.md group E), the
// serving bridge (D1 to D4) and the library notes (A11, B3, C3, D6) are ours and labelled.

// ---------------------------------------------------------------- questions
export const questionsS225: Record<string, Question> = {
  'ch22/22.5/q-scaling': {
    kind: 'choice',
    id: 'ch22/22.5/q-scaling',
    prompt: <>Which of the section's two motivations is weak scaling, in the sense of Gustafson's law?</>,
    options: [
      { label: 'The second: more instances finished in the time available.', correct: true },
      { label: 'The first: one problem, the 10-hour analysis, finished sooner.', why: 'Fixed problem, more processors, less time: that is strong scaling, the setting of Amdahl\'s law.' },
      { label: 'Both: any speed-up that lets more work finish is weak scaling.', why: 'The two differ in what is held fixed: the problem size, or the time.' },
    ],
    explain: (
      <p>
        Strong scaling keeps the problem fixed and asks how much sooner it finishes (the 10-hour analysis that must take
        4). Weak scaling lets the problem grow with the machine and asks how much more fits in the same time (more runs of
        the analysis, more users' queries). The mapping onto the book's two motivations is ours.
      </p>
    ),
    source: 'ch. 22 §22.5, p. 539; CUDA Best Practices Guide §4.1.3 (ch22-facts A9)',
  },
  'ch22/22.5/q-plateau': {
    kind: 'choice',
    id: 'ch22/22.5/q-plateau',
    prompt: (
      <>
        Lab 22.2 run with M3 (lab 15.6's GEMM) shows a time per step that is flat from batch 1 to 128. Does that show the
        step is bandwidth-bound across that range?
      </>
    ),
    options: [
      {
        label: 'Not by itself. M3\'s block tile is 128 rows tall and computes 128 rows whatever B is, so below 128 neither its grid nor its arithmetic changes.',
        correct: true,
      },
      { label: 'Yes: a time per step that stays flat as rows are added is the signature of a bandwidth-bound step.', why: 'A flat line means something fixed sets the time. With M3 it can be the 128 rows of arithmetic the tile does anyway.' },
      { label: 'No: a flat line means the step is compute-bound, with the arithmetic units as the fixed cost.', why: 'It may be either. Flatness alone cannot tell which fixed cost sets the time.' },
    ],
    explain: (
      <p>
        Tile quantisation: when the batch does not fill the block tile, the padding rows still cost their arithmetic
        (ch22-facts E3, from NVIDIA's matrix multiplication guide). To tell the two causes apart, compare with M2, whose
        16-row tile makes steps of 16, and compute an effective weight bandwidth, weight bytes per step divided by the
        measured time, against lab 5.1's bandwidth. In the explorer above, set the padding to 128 rows and look at the
        regime the projections are in at batch 1.
      </p>
    ),
    source: 'ch22-facts E3; labs/ch15/lab6_linear (bM = 128)',
  },
  'ch22/22.5/q-half-bytes': {
    kind: 'choice',
    id: 'ch22/22.5/q-half-bytes',
    prompt: <>Weights and activations go from float32 to 16-bit. Where does the projection's knee B* move?</>,
    options: [
      { label: 'To about half: from about 2T rows to about T rows.', correct: true },
      { label: 'It doubles, to about 4T rows: each row now brings fewer bytes to share the weights with.', why: 'Halving the bytes doubles the operations per byte at every batch, so the threshold is reached sooner, not later.' },
      { label: 'It does not move: the arithmetic is the same.', why: 'The arithmetic is the same and the bytes are halved. Intensity is the ratio of the two.' },
    ],
    explain: (
      <p>
        While B is much smaller than K and N, I(B) is about B/2 FLOP/B in float32 and about B in 16-bit, so the threshold
        T is reached near 2T rows against near T rows. Below the knee the step is the weight stream, so its time also
        halves: the bandwidth argument for quantised weights that module X2 and milestone M9 take further.
      </p>
    ),
    source: 'ch22-facts E1 (our arithmetic)',
  },
  'ch22/22.5/q-attention': {
    kind: 'choice',
    id: 'ch22/22.5/q-attention',
    prompt: <>Which parts of a decode step do more operations per byte as the batch grows?</>,
    options: [
      { label: 'The projections that carry weights: Q, K, V, the attention output, the MLP and the LM head.', correct: true },
      { label: 'Every part of the step, since every kernel now processes B rows in one launch.', why: 'Bias, residual, softmax and argmax do a fixed amount of work per element of each row, however many rows there are (E2).' },
      { label: 'Attention most of all, which is why p. 539 names the attention heads as what batching makes more intense.', why: 'Attention proper reads each request\'s own keys and values: nothing in it is shared between requests, so nothing is reused.' },
    ],
    explain: (
      <p>
        Batching pays where one fetched byte can serve many rows, and only the weights are shared. Attention's reads of K
        and V grow with the batch, one cache per request, like the elementwise pieces. Page 539's "attention heads" is
        right only if "attention" includes the QKV and output projections, an extended definition Orca's authors name
        in a footnote. Chapter 20 (p. 505) says the attention layers themselves do not benefit.
      </p>
    ),
    source: 'ch. 22 p. 539 against ch. 20 p. 505; ch22-facts A6 (Orca, OSDI 2022)',
  },
  'ch22/22.5/q-who-pays': {
    kind: 'choice',
    id: 'ch22/22.5/q-who-pays',
    prompt: (
      <>
        Eight users are waiting for their next token. Option A: decode one user at a time, round-robin, so one round is
        eight batch-1 steps. Option B: one step at batch 8, which takes 3 times as long as a batch-1 step. Which gives each
        user less time between tokens?
      </>
    ),
    options: [
      { label: 'B: 3 units per token instead of 8.', correct: true },
      { label: 'A, because each of its steps is shorter.', why: 'Each user gets one token per round, and a round of A is eight steps long.' },
      { label: 'The same: the work is the same.', why: 'The arithmetic is the same; the weight traffic is not. A reads the weights eight times per round, B once.' },
    ],
    explain: (
      <p>
        The book's "longer latency" for each query is measured against a server with a single user, where a batch-1
        step is all anyone waits. Against serving the same users one at a time, batching wins for every user, and by
        more the further below the knee the batch is.
      </p>
    ),
    source: 'ch. 22 §22.5, p. 539; our arithmetic',
  },
  'ch22/22.5/q-cpu-gpu': {
    kind: 'choice',
    id: 'ch22/22.5/q-cpu-gpu',
    prompt: (
      <>
        A server has two decode paths. Path A runs one request per step, and a step takes t. Path B always runs a step of
        64 rows, padding empty ones, and a step takes 4t. Which design goal does each serve, and what does a user who
        arrives at an empty server get from each?
      </>
    ),
    options: [
      {
        label: 'A serves latency, B throughput. A lone user gets a token every t on A and every 4t on B; with 64 users, B gives 16 times A\'s tokens per second.',
        correct: true,
      },
      {
        label: 'B serves both. With 64 users it delivers 16 times A\'s tokens per second, so even a lone user on an empty server gets each token sooner on B.',
        why: 'The 16 times is tokens per second for the device, spread over 64 rows. A lone user fills one row of B and still waits 4t for each token, against t on A.',
      },
      {
        label: 'A serves throughput and B latency. A finishes 4 steps in the time B finishes one, so it produces more tokens per second.',
        why: 'A finishes more steps, but each yields one token. A full step of B yields 64 tokens in 4t, which is 16 per t against A\'s 1.',
      },
    ],
    explain: (
      <p>
        Path A is the CPU's design goal in the book's analogy: the shortest time for each element, here each user's token.
        Path B is the GPU's: the most work per second over many elements, paid for by each one taking longer. Which is
        better depends on the load, not on the path. NVIDIA's Best Practices Guide puts the hardware side the same way:
        CPU cores "minimize latency for a small number of threads", GPUs handle many lightweight threads "to maximize
        throughput".
      </p>
    ),
    source: 'ch. 22 §22.5, pp. 539–540; CUDA Best Practices Guide §3.1 (ch22-facts A8); our arithmetic',
  },
  'ch22/22.5/q-continuous': {
    kind: 'choice',
    id: 'ch22/22.5/q-continuous',
    prompt: <>Why does continuous (iteration-level) batching matter for this section's trade-off?</>,
    options: [
      {
        label: 'It admits and retires requests every step, so the batch stays near the slot limit; each user then pays that batch\'s step time per token.',
        correct: true,
      },
      { label: 'It makes attention batch: requests decoded together share one pass over the keys and values.', why: 'Attention proper still reads each request\'s own K and V. Orca, which introduced the technique, runs attention per request.' },
      { label: 'It removes batching\'s cost in time per token, because no request waits for the others in its batch to finish.', why: 'It removes the waiting for other requests to finish. Each step still takes the time of the batch it carries.' },
    ],
    explain: (
      <p>
        Static batching wastes slots on finished requests and makes new ones wait, so the batch the server actually runs
        is smaller than its limit. Continuous batching keeps it full, which is what puts a busy server where the model's
        curve says: tokens per second near their ceiling, and each user's time per token set by the step at that batch.
      </p>
    ),
    source: 'ch22-facts D1 to D4 (ours, not the book\'s)',
  },
  'ch22/22.5/q-batched-gemm': {
    kind: 'choice',
    id: 'ch22/22.5/q-batched-gemm',
    prompt: (
      <>
        <code>cublasGemmStridedBatchedEx</code> runs many GEMMs in one call. Is it how the book's batching of user queries
        is implemented?
      </>
    ),
    options: [
      {
        label: 'No. The book stacks the requests\' rows into one GEMM with M = B against one shared weight matrix; batched GEMM is for many independent pairs.',
        correct: true,
      },
      { label: 'Yes: one instance of the batched call per request, each multiplying that request\'s row by the weight matrix, all in one call.', why: 'Each instance would carry its own copy of the weight operand, a matrix-vector product per request: the redundant weight traffic batching exists to remove.' },
      { label: 'Yes on HIP: hipBLAS defines its batched GEMM as one product over the stacked rows, unlike cuBLAS.', why: 'hipBLAS has the same batched entry points as cuBLAS, and they mean the same thing there: C_i = A_i B_i for each instance i.' },
    ],
    explain: (
      <p>
        Batched GEMM fits products that do not share an operand, such as each sequence's attention scores against its own
        keys. llama.cpp's CUDA backend sends a 2-D weight times a batch of tokens to one plain GEMM and keeps the batched
        calls for 3-D and 4-D operands.
      </p>
    ),
    source: 'ch22-facts A11 (cuBLAS), B3 (hipBLAS), C3 (MPS), D6 (llama.cpp)',
  },
};

// ---------------------------------------------------------------- shared setup
type ShapeKey = 'm8' | 'p4096' | 'p1024';
interface Setup {
  devKey: string;
  custom: { peak: number; bw: number };
  shapeKey: ShapeKey;
  wBytes: WeightBytes;
  pad: Padding;
  batch: number;
}
interface DevOption {
  key: string;
  label: string;
  dev: Device;
  note: string;
  backend?: string;
}
const BOOK_OPTION: DevOption = {
  key: 'book',
  label: "The book's example (H100)",
  dev: BOOK_DEVICE,
  note: "The book's figures for its example GPU (ch. 5, p. 94): 66.9 TFLOPS and 3.35 TB/s, and the threshold it prints for them, 20.0 FLOP/B. The book's example only.",
};

const fmtTime = (s: number) => (!isFinite(s) ? 'n/a' : s < 1e-3 ? `${fmt(s * 1e6)} µs` : s < 1 ? `${fmt(s * 1e3)} ms` : `${fmt(s)} s`);
const fmtRate = (v: number) => fmt(v, 3);

function useSetup() {
  const { reports } = useReports();
  const rep = reports as unknown as ReportsLike;
  const [setup, setSetup] = useState<Setup>({ devKey: 'book', custom: { peak: 1000, bw: 100 }, shapeKey: 'm8', wBytes: 4, pad: 1, batch: 1 });
  const options: DevOption[] = useMemo(
    () => [
      BOOK_OPTION,
      ...measuredDevices(rep).map((m) => ({
        key: m.backend,
        backend: m.backend,
        label: `Yours: ${m.name} (${m.backend})`,
        dev: { peak: m.peak, bw: m.bw },
        note: `Measured by lab 5.1 on ${m.date}: ${fmt(m.peak)} GFLOP/s and ${fmt(m.bw)} GB/s from two probe kernels, threshold ${fmt(m.peak / m.bw)} FLOP/B.`,
      })),
    ],
    [rep],
  );
  const option: DevOption =
    setup.devKey === 'custom'
      ? { key: 'custom', label: 'Custom', dev: setup.custom, note: 'Numbers you typed.' }
      : (options.find((o) => o.key === setup.devKey) ?? BOOK_OPTION);
  const m8 = useMemo(() => m8Shape(rep, option.backend), [rep, option.backend]);
  const shape: Shape = useMemo(() => (setup.shapeKey === 'm8' ? m8 : setup.shapeKey === 'p4096' ? PROJ_4096 : PROJ_1024), [setup.shapeKey, m8]);
  const prec = useMemo(() => precisionFor(setup.wBytes), [setup.wBytes]);
  return { reports: rep, setup, set: (p: Partial<Setup>) => setSetup((s) => ({ ...s, ...p })), options, option, shape, prec, m8 };
}
type SetupState = ReturnType<typeof useSetup>;

function shapeLabel(shape: Shape, fromRows: boolean) {
  if (shape.kind === 'proj') return `one ${shape.k} → ${shape.n} projection`;
  return `lab 22.2's step: d = ${shape.d}, ${shape.layers} layers, vocabulary ${fmt(shape.vocab, 6)}${fromRows ? ' (from your rows)' : ' (the lab\'s default)'}`;
}

// ---------------------------------------------------------------- the roofline explorer's plots
const MARKS = ['diamond', 'square', 'triangle', 'ring'] as const;
function Marker({ kind, x, y, filled = true }: { kind: (typeof MARKS)[number]; x: number; y: number; filled?: boolean }) {
  const f = filled ? 'var(--ink)' : 'var(--page)';
  if (kind === 'diamond') return <rect x={x - 4.5} y={y - 4.5} width={9} height={9} transform={`rotate(45 ${x} ${y})`} fill={f} stroke="var(--ink)" />;
  if (kind === 'square') return <rect x={x - 4} y={y - 4} width={8} height={8} fill={f} stroke="var(--ink)" />;
  if (kind === 'triangle') return <path d={`M${x},${y - 5.5} L${x + 5},${y + 4} L${x - 5},${y + 4} Z`} fill={f} stroke="var(--ink)" />;
  return <circle cx={x} cy={y} r={4.5} fill="none" stroke="var(--ink)" strokeWidth={1.8} />;
}

const POW2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

function BatchPlot({ kind, curve, batch, bstar, series }: { kind: 'time' | 'tokens'; curve: StepResult[]; batch: number; bstar: number | null; series: Series[] }) {
  const W = 440;
  const H = 280;
  const pad = { l: 64, r: 12, t: 14, b: 42 };
  const val = (r: StepResult) => (kind === 'time' ? r.seconds : r.tokensPerSecond);
  const mval = (p: Point) => (kind === 'time' ? p.seconds : p.batch / p.seconds);
  const ys = [...curve.map(val), ...series.flatMap((s) => s.points.map(mval))].filter((v) => v > 0 && isFinite(v));
  const lo = Math.min(...ys) / 1.6;
  const hi = Math.max(...ys) * 1.6;
  const X = (b: number) => pad.l + (Math.log2(Math.min(Math.max(b, 1), 1024)) / 10) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (H - pad.t - pad.b);
  const ticks: number[] = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
    for (const m of Math.log10(hi / lo) < 2.2 ? [1, 2, 5] : [1]) {
      const v = m * 10 ** e;
      if (v >= lo && v <= hi) ticks.push(v);
    }
  const label = (v: number) => (kind === 'time' ? fmtTime(v) : fmt(v, 2));
  const at = curve[Math.min(batch, curve.length) - 1];
  const values = POW2.filter((b) => b <= curve.length).map((b) => [b, val(curve[b - 1])]);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={kind === 'time' ? 'Time per step against batch' : 'Tokens per second against batch'}
      style={{ width: '100%', maxWidth: W, display: 'block' }} data-plot={kind}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {bstar != null && bstar <= 1024 ? (
        <rect x={pad.l} y={pad.t} width={X(bstar) - pad.l} height={H - pad.t - pad.b} fill="var(--global-wash)" />
      ) : null}
      {POW2.map((b) => (
        <g key={b}>
          <line x1={X(b)} x2={X(b)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(b)} y={H - pad.b + 14} fontSize="10.5" textAnchor="middle">{b}</text>
        </g>
      ))}
      {ticks.map((v) => (
        <g key={v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 5} y={Y(v) + 3.5} fontSize="10" textAnchor="end">{label(v)}</text>
        </g>
      ))}
      <text x={(W + pad.l) / 2} y={H - 8} fontSize="11" textAnchor="middle">batch B (log scale)</text>
      <text x={13} y={(H - pad.b + pad.t) / 2} fontSize="11" textAnchor="middle" transform={`rotate(-90 13 ${(H - pad.b + pad.t) / 2})`}>
        {kind === 'time' ? 'time per step' : 'tokens per second'}
      </text>
      {bstar != null && bstar <= 1024 ? (
        <g data-bstar-line={bstar.toFixed(1)}>
          <line x1={X(bstar)} x2={X(bstar)} y1={pad.t} y2={H - pad.b} stroke="var(--ink)" strokeDasharray="4 3" />
          <text x={X(bstar) + 4} y={pad.t + 13} fontSize="10.5">B* {bstar.toFixed(1)}</text>
          <text x={pad.l + 5} y={H - pad.b - 6} fontSize="10" style={{ fill: 'var(--global)' }}>bandwidth-bound GEMMs</text>
        </g>
      ) : null}
      <polyline data-series="model" data-values={JSON.stringify(values)}
        points={curve.map((r) => `${X(r.batch).toFixed(1)},${Y(val(r)).toFixed(1)}`).join(' ')}
        fill="none" stroke="var(--ink)" strokeWidth={2} />
      <circle cx={X(at.batch)} cy={Y(val(at))} r={5} fill="var(--reg)" stroke="var(--page)" strokeWidth={1.5} data-current={at.batch} />
      {series.map((s, i) => (
        <g key={s.key} data-measured={s.key} data-points={s.points.length}>
          {s.points.map((p) => (
            <g key={p.batch}>
              <Marker kind={MARKS[i % MARKS.length]} x={X(p.batch)} y={Y(mval(p))} />
            </g>
          ))}
          {s.knee != null ? (
            <g data-measured-knee={s.knee}>
              <circle cx={X(s.knee)} cy={Y(mval(s.points.find((p) => p.batch === s.knee)!))} r={9} fill="none" stroke="var(--ink)" strokeDasharray="2 2" />
            </g>
          ) : null}
        </g>
      ))}
    </svg>
  );
}

// ---------------------------------------------------------------- explorer 1: the step on the roofline
function DecodeRoofline({ st }: { st: SetupState }) {
  const { setup, set, options, option, shape, prec, reports, m8 } = st;
  const dev = option.dev;
  const T = thresholdOf(dev);
  const curve = useMemo(() => stepCurve(shape, prec, setup.pad, dev, batchRange()), [shape, prec, setup.pad, dev]);
  const r = curve[setup.batch - 1];
  const bstar = stepBStar(shape, prec, dev);
  const modelKnee = kneeOf(curve.map((c) => ({ batch: c.batch, seconds: c.seconds })));
  // The reader's own measurements, when this device is one lab 5.1 ran on; they are float32.
  const backend = option.backend;
  const allSeries: Series[] = backend
    ? shape.kind === 'step'
      ? stepSeries(reports, backend)
      : projectionSeries(reports, backend, shape.k, shape.n)
    : [];
  const series = setup.wBytes === 4 ? allSeries : [];
  const fromRows = shape.kind === 'step' && m8 !== M8_DEFAULT;
  const step = (k: ShapeKey, label: string) => (
    <button key={k} aria-pressed={setup.shapeKey === k} onClick={() => set({ shapeKey: k })}>{label}</button>
  );
  return (
    <Bench
      title="A decode step on the roofline"
      task="Find where time per step stops being flat, then change the bytes per weight and the tile padding."
      foot={
        <>
          Each piece at the roofline bound of its own minimum traffic, pieces one after another. The shaded band is
          where the GEMMs are bandwidth-bound; the dashed line is the model's B*. Filled markers are your measurements
          (float32 only), dotted rings their knees, where time per step has grown by half. {option.note}
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Device
          <select aria-label="device" value={setup.devKey} onChange={(e) => set({ devKey: e.target.value })}>
            {options.map((o) => (
              <option key={o.key} value={o.key}>{o.label}</option>
            ))}
            <option value="custom">Custom numbers</option>
          </select>
        </label>
        {setup.devKey === 'custom' ? (
          <>
            <label className="field">
              Peak (GFLOP/s)
              <input type="number" min="1" aria-label="custom peak" value={setup.custom.peak}
                onChange={(e) => set({ custom: { ...setup.custom, peak: Math.max(1, +e.target.value || 1) } })} />
            </label>
            <label className="field">
              Bandwidth (GB/s)
              <input type="number" min="1" aria-label="custom bandwidth" value={setup.custom.bw}
                onChange={(e) => set({ custom: { ...setup.custom, bw: Math.max(1, +e.target.value || 1) } })} />
            </label>
          </>
        ) : null}
        <label className="field">
          Model
          <span className="seg" role="group" aria-label="model">
            {step('m8', 'lab 22.2 step')}
            {step('p4096', '4096 → 4096')}
            {step('p1024', '1024 → 4096')}
          </span>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Bytes per weight
          <span className="seg" role="group" aria-label="bytes per weight">
            {WEIGHT_BYTES.map((b) => (
              <button key={b} aria-pressed={setup.wBytes === b} onClick={() => set({ wBytes: b })}>
                {b === 4 ? '4 (float32)' : b === 2 ? '2 (16-bit)' : b === 1 ? '1 (8-bit)' : '0.5 (4-bit)'}
              </button>
            ))}
          </span>
        </label>
        <label className="field">
          GEMM tile padding
          <span className="seg" role="group" aria-label="tile padding">
            {([1, 16, 128] as Padding[]).map((p) => (
              <button key={p} aria-pressed={setup.pad === p} onClick={() => set({ pad: p })}>
                {p === 1 ? 'none' : p === 16 ? '16 rows (M2)' : '128 rows (M3)'}
              </button>
            ))}
          </span>
        </label>
        <label className="field">
          Batch B = {setup.batch}
          <input type="range" min="0" max="10" step="1" aria-label="batch" value={Math.log2(setup.batch)}
            onChange={(e) => set({ batch: 2 ** +e.target.value })} />
        </label>
      </div>
      <div className="cols plot">
        <div className="panel">
          <p style={{ margin: '0 0 0.5rem' }}>
            <b>{shapeLabel(shape, fromRows)}</b>, weights {prec.weight} B, activations {prec.act} B, batch {setup.batch}
          </p>
          <table className="data" style={{ margin: 0 }} data-pieces="">
            <thead>
              <tr>
                <th>Piece</th>
                <th className="num">Time</th>
                <th className="num">Share</th>
                <th>Limited by</th>
              </tr>
            </thead>
            <tbody>
              {r.pieces.map((p) => (
                <tr key={p.key} data-piece={p.key}>
                  <td>{p.label}</td>
                  <td className="num">{fmtTime(p.seconds)}</td>
                  <td className="num">{fmt((100 * p.seconds) / r.seconds, 2)}%</td>
                  <td>{p.bound === 'memory' ? <span className="chip global">bandwidth</span> : <span className="chip reg">arithmetic</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ margin: '0.7rem 0 0' }} data-step-seconds={r.seconds}>
            Time per step <b>{fmtTime(r.seconds)}</b>; {fmtRate(r.tokensPerSecond)} tokens per second; each user waits{' '}
            <b>{fmtTime(r.perToken)}</b> per token.
          </p>
          <p style={{ margin: '0.5rem 0 0' }}>
            The GEMMs do <b data-intensity={r.intensity}>{fmt(r.intensity)}</b> FLOP/B against a threshold of{' '}
            <b data-threshold={T}>{fmt(T)}</b>. Their B* = <b data-bstar={bstar == null ? '' : bstar.toFixed(1)}>{bstar == null ? 'never' : bstar.toFixed(1)}</b>
            {modelKnee != null ? <>; the model's time per step is 1.5 times its batch-1 value at B = {modelKnee}</> : null}.
          </p>
          {series.length ? (
            <p style={{ margin: '0.5rem 0 0' }} data-overlay-note="">
              Measured: {series.map((s, i) => (
                <span key={s.key}>
                  {i ? '; ' : ''}
                  {s.label} ({MARKS[i % MARKS.length]}), {s.knee == null ? 'no knee up to ' + s.points.at(-1)!.batch : `knee at ${s.knee}`}
                </span>
              ))}.
            </p>
          ) : backend && allSeries.length ? (
            <p className="muted" style={{ margin: '0.5rem 0 0' }}>Your measurements are float32: choose 4 bytes per weight to see them.</p>
          ) : backend ? (
            <p className="muted" style={{ margin: '0.5rem 0 0' }}>
              No measurements of this shape on {backend} yet.{' '}
              {shape.kind === 'step' ? <>Lab 22.2 times it (<a href={href('ch22', 'labs')}>labs</a>).</> : <>Labs 5.3 and 15.6 time the 1024 → 4096 layer; 15.6 also 4096 → 4096.</>}
            </p>
          ) : null}
        </div>
        <div>
          <BatchPlot kind="time" curve={curve} batch={setup.batch} bstar={bstar} series={series} />
          <BatchPlot kind="tokens" curve={curve} batch={setup.batch} bstar={bstar} series={series} />
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the Predict's curve
function IntensityCurve() {
  const prec = precisionFor(4);
  const W = 420;
  const H = 170;
  const pad = { l: 40, r: 10, t: 10, b: 30 };
  const max = 96;
  const X = (b: number) => pad.l + ((b - 1) / (max - 1)) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - (v / 50) * (H - pad.t - pad.b);
  const bs = Array.from({ length: max }, (_, i) => i + 1);
  const b = bStar(4096, 4096, 20, prec)!;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Intensity of a 4096 by 4096 projection against batch" style={{ width: '100%', maxWidth: W, display: 'block', marginTop: '0.5rem' }}
      data-intensity-curve={b.toFixed(1)}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {[0, 10, 20, 30, 40, 50].map((v) => (
        <text key={v} x={pad.l - 5} y={Y(v) + 3.5} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink)' }}>{v}</text>
      ))}
      {[1, 16, 32, 48, 64, 80, 96].map((v) => (
        <text key={v} x={X(v)} y={H - pad.b + 13} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink)' }}>{v}</text>
      ))}
      <text x={(W + pad.l) / 2} y={H - 3} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink)' }}>batch B</text>
      <line x1={pad.l} x2={W - pad.r} y1={Y(20)} y2={Y(20)} stroke="var(--ink)" strokeDasharray="4 3" />
      <text x={W - pad.r - 4} y={Y(20) - 5} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink)' }}>threshold 20 FLOP/B</text>
      <polyline points={bs.map((x) => `${X(x)},${Y(intensity(4096, 4096, x, prec))}`).join(' ')} fill="none" stroke="var(--reg)" strokeWidth={2} />
      <line x1={X(b)} x2={X(b)} y1={Y(20)} y2={H - pad.b} stroke="var(--ink-3)" />
      <text x={X(b) + 4} y={H - pad.b - 6} fontSize="10.5" style={{ fill: 'var(--ink)' }}>B* = {b.toFixed(1)}</text>
    </svg>
  );
}

// ---------------------------------------------------------------- explorer 2: batch slots on a server
const COUNT = 1000;
const WINDOW = 240;
const LOADS = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256];

/** The window of steps the chart shows: WINDOW steps from the middle of the run, where the batch has settled. */
function chartWindow(steps: number): [number, number] {
  const from = Math.max(0, Math.floor(steps / 2) - WINDOW / 2);
  return [from, Math.min(steps, from + WINDOW)];
}

function SlotChart({ run }: { run: ServeResult }) {
  const W = 680;
  const H = 190;
  const l = 52;
  const [from, to] = chartWindow(run.steps.length);
  const cw = (W - l - 6) / WINDOW;
  const rh = (H - 22) / run.slots;
  const x = (k: number) => l + (k - from) * cw;
  const gap = rh >= 5 ? 0.6 : 0;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Slots against decode steps" style={{ width: '100%', minWidth: 520, display: 'block' }}
      data-slot-chart={run.policy} data-slots={run.slots} data-from={from} data-to={to}>
      <rect x={l} y={0} width={WINDOW * cw} height={H - 22} fill="var(--page)" stroke="var(--rule)" />
      {run.served
        .filter((r) => r.admitStep < to && r.returnStep > from)
        .map((r) => {
          const start = Math.max(r.admitStep, from);
          const genEnd = Math.min(r.lastStep + 1, to);
          const heldStart = Math.max(r.lastStep + 1, from);
          const heldEnd = Math.min(r.returnStep, to);
          return (
            <g key={r.id}>
              {genEnd > start ? (
                <rect x={x(start)} y={r.slot * rh + gap} width={Math.max(0.5, (genEnd - start) * cw - gap)} height={Math.max(0.5, rh - 2 * gap)}
                  fill="var(--ink-2)" opacity={r.length <= SHORT_MAX ? 0.45 : 1} data-short={r.length <= SHORT_MAX ? '1' : '0'} />
              ) : null}
              {heldEnd > heldStart ? (
                <rect x={x(heldStart)} y={r.slot * rh + rh / 2 - Math.min(1.5, rh / 4)} width={(heldEnd - heldStart) * cw} height={Math.max(0.5, Math.min(3, rh / 2))}
                  fill="var(--ink-3)" data-held="" />
              ) : null}
            </g>
          );
        })}
      {[0, 60, 120, 180, 240].filter((k) => from + k <= to).map((k) => (
        <text key={k} x={x(from + k)} y={H - 8} fontSize="10" textAnchor={k === 0 ? 'start' : k === 240 ? 'end' : 'middle'}>{from + k}</text>
      ))}
      <text x={l - 6} y={10} fontSize="10" textAnchor="end">slot 1</text>
      <text x={l - 6} y={H - 26} fontSize="10" textAnchor="end">{run.slots}</text>
    </svg>
  );
}

function ServerSlots({ st }: { st: SetupState }) {
  const { option, shape, prec, setup } = st;
  const [policy, setPolicy] = useState<Policy>('continuous');
  const [slots, setSlots] = useState(16);
  const [loadIdx, setLoadIdx] = useState(LOADS.indexOf(64));
  const [mix, setMix] = useState<Mix>('long');
  const [seed, setSeed] = useState(1);
  const dev = option.dev;
  const stepTime = useMemo(() => {
    const table: number[] = [];
    return (a: number) => (table[a] ??= stepAt(shape, a, prec, setup.pad, dev).seconds);
  }, [shape, prec, setup.pad, dev]);
  const t1 = stepTime(1);
  const cap1 = 1 / (t1 * MIXES[mix].mean);
  const load = LOADS[loadIdx];
  const rate = load * cap1;
  const runs = useMemo(() => {
    const requests = makeRequests({ count: COUNT, rate, mix, seed });
    return {
      static: serve({ requests, slots, policy: 'static', stepTime }),
      continuous: serve({ requests, slots, policy: 'continuous', stepTime }),
    };
  }, [rate, mix, seed, slots, stepTime]);
  const shown = runs[policy];
  const ratio = runs.continuous.tokensPerSecond / runs.static.tokensPerSecond;
  const row = (p: Policy) => {
    const r = runs[p];
    const cell = (k: string, v: number | null, text: string) => (
      <td className="num" data-k={k} data-v={v ?? ''}>{text}</td>
    );
    return (
      <tr key={p} data-policy={p} style={p === policy ? { background: 'var(--rule-soft)' } : undefined}>
        <td>{p}</td>
        {cell('tps', r.tokensPerSecond, fmtRate(r.tokensPerSecond))}
        {cell('tpt', r.meanTimePerToken, fmtTime(r.meanTimePerToken))}
        {cell('worst', r.worstTimePerToken, fmtTime(r.worstTimePerToken))}
        {cell('short', r.meanFinishShort, r.meanFinishShort == null ? 'none' : fmtTime(r.meanFinishShort))}
        {cell('util', r.utilisation, `${fmt(100 * r.utilisation, 2)}%`)}
        {cell('batch', r.meanBatch, fmt(r.meanBatch, 3))}
      </tr>
    );
  };
  return (
    <Bench
      title="Batch slots on a server"
      task="At heavy load, compare the two policies at the same slot limit; then raise the limit past the knee."
      foot={
        <>
          {COUNT} requests from a seeded generator; step times from the explorer above ({option.label}, {shapeLabel(shape, false).replace(' (the lab\'s default)', '')},
          weights {prec.weight} B, padding {setup.pad === 1 ? 'none' : setup.pad + ' rows'}). The chart shows {WINDOW} steps of the
          policy you picked, from the middle of the run, where the batch has settled: solid bars are requests of more than {SHORT_MAX} tokens, fainter bars shorter ones, and a thin line is a
          finished request still holding its slot.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Arrivals: {fmt(rate, 3)} requests per second ({load}× what one slot at batch 1 serves)
          <input type="range" min="0" max={LOADS.length - 1} step="1" aria-label="arrival rate" value={loadIdx}
            onChange={(e) => setLoadIdx(+e.target.value)} />
        </label>
        <label className="field">
          Output lengths
          <span className="seg" role="group" aria-label="output lengths">
            {(['short', 'long'] as Mix[]).map((m) => (
              <button key={m} aria-pressed={mix === m} onClick={() => setMix(m)} title={MIXES[m].label}>
                {m === 'short' ? 'similar' : 'long-tailed'}
              </button>
            ))}
          </span>
        </label>
        <button className="quiet" onClick={() => setSeed(seed + 1)}>New arrivals</button>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Slot limit = {slots}
          <input type="range" min="0" max="8" step="1" aria-label="slot limit" value={Math.log2(slots)}
            onChange={(e) => setSlots(2 ** +e.target.value)} />
        </label>
        <label className="field">
          Chart shows
          <span className="seg" role="group" aria-label="policy">
            {(['static', 'continuous'] as Policy[]).map((p) => (
              <button key={p} aria-pressed={policy === p} onClick={() => setPolicy(p)}>{p}</button>
            ))}
          </span>
        </label>
      </div>
      <SlotChart run={shown} />
      <p className="muted" style={{ margin: '0.2rem 0 0.8rem', fontSize: '0.78rem' }}>
        decode steps across ({chartWindow(shown.steps.length).join(' to ')}, from the middle of a run of {shown.steps.length}), slots down
      </p>
      <table className="data" style={{ margin: 0 }} data-serving="">
        <thead>
          <tr>
            <th>Policy</th>
            <th className="num">Tokens per second</th>
            <th className="num">Time per token, mean</th>
            <th className="num">worst</th>
            <th className="num">Short requests done after</th>
            <th className="num">Slots busy</th>
            <th className="num">Mean batch</th>
          </tr>
        </thead>
        <tbody>
          {row('static')}
          {row('continuous')}
        </tbody>
      </table>
      <p style={{ margin: '0.7rem 0 0' }} data-verdict={ratio}>
        At {slots} {slots === 1 ? 'slot' : 'slots'},{' '}
        {Math.abs(ratio - 1) < 0.02 ? 'the two policies make the same tokens per second.' : `continuous batching makes ${fmt(ratio, 3)} times static's tokens per second.`}
        {runs.continuous.meanBatch < 0.5 * slots ? ' The slots are mostly empty: at this load the limit is not what sets the batch.' : ''}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- the section
export function S225() {
  const st = useSetup();
  const f32 = precisionFor(4);
  const b8 = bookThroughput(8, 3, 1);
  const bBook = bStar(4096, 4096, 20, f32)!;
  const bWhole = bStarSearch(4096, 4096, 20, f32)!;
  const wb = weightBytes(M8_DEFAULT, f32);
  // The per-row pieces' share of lab 22.2's default step at batch 256 (book's device, float32).
  const r256 = stepAt(M8_DEFAULT, 256, f32, 1, BOOK_DEVICE);
  const rowShare = (100 * r256.rowSeconds) / r256.seconds;
  return (
    <>
      <p>
        Every technique in this book so far has had one aim: take the same input and finish it sooner. The book calls
        this latency optimisation. Its examples come from chapter 11 (double buffering, a warp-level technique, and the
        look-back of the single-pass scan) and from chapter 20, where flash attention shortens a transformer's time on a
        given input by removing redundant reads and writes of global memory.
      </p>
      <Aside>
        <p>
          The book's list reads "double buffering, warp-level shuffling register tiling, and look-back". The middle item
          looks like two of chapter 11's techniques run together, warp-level shuffles (§11.4) and register tiling
          (§11.7). That reading is ours.
        </p>
      </Aside>
      <p>
        Finishing one input sooner is not the only reason to optimise. The book names two. The first is the one this book
        has served all along: solve a given problem in less time. Its example is an investment firm whose overnight risk
        analysis of all its portfolios takes 10 hours, when the portfolio managers want it done in 4. The second is less
        obvious. The same firm wants many runs of the analysis, with different parameters, and they do not all fit in the
        window. Optimising for many instances brings a trade-off the first goal never had: the latency of each instance
        against the latency of all of them together.
      </p>
      <Source section="22.5" pages="539" />
      <p>
        If you know the vocabulary of parallel scaling, these are its two kinds. Finishing a fixed problem sooner on more
        hardware is strong scaling, the setting of Amdahl's law (chapter 1 and section 22.3). Fitting more problems into
        the same time, with the problem growing as the machine does, is weak scaling, where Gustafson's law applies.
      </p>
      <Beyond>
        <p>
          NVIDIA's Best Practices Guide uses both names (§4.1.3). Mapping the book's two motivations onto them is ours
          (ch22-facts A9).
        </p>
      </Beyond>
      <Ask q={questionsS225['ch22/22.5/q-scaling']} />

      <h2>Batching queries turns many products into one</h2>
      <p>
        LLM inference is the book's case of the second goal. Answering one user's query is one instance of the problem;
        a server has many users. If the server stacks the current row of several users' queries into one matrix, each
        projection that was a vector-matrix product per user becomes one matrix product for all of them. The weights are
        the same for every user, so a weight fetched once now serves every row. That matrix product takes longer than one
        vector-matrix product, but much less than all of them one after another.
      </p>
      <Source section="22.5" pages="539" />
      <p>The book states the gain for the QKV projection as two throughputs:</p>
      <Aside>
        <p>
          The book's prose says "vector-matrix" and its formula says "matrix-vector". Same operation: one row of
          activations against a weight matrix.
        </p>
      </Aside>
      <table className="data">
        <tbody>
          <tr>
            <td>with batching</td>
            <td>throughput = number of queries ÷ latency of the matrix multiplication</td>
          </tr>
          <tr>
            <td>without</td>
            <td>throughput = number of queries ÷ total latency of all the vector-matrix multiplications</td>
          </tr>
        </tbody>
      </table>
      <p>
        The first is higher. The price, as the book puts it, is that each query sees a longer QKV projection latency,
        while the group as a whole finishes sooner.
      </p>
      <Predict
        id="ch22/22.5/p-batch8"
        prompt="A batch of 8 queries: the matrix multiplication takes 3 times as long as one vector-matrix multiplication. By what factor does throughput change, against eight separate vector-matrix multiplications?"
        unit="×"
      >
        {(v) => (
          <p>
            Batched: 8 queries in 3 units. Separate: 8 queries in 8 units. Throughput rises by 8 / 3 ={' '}
            <b data-gain={b8.gain}>{fmt(b8.gain)}×</b>, while each query's own projection went from 1 unit to{' '}
            {b8.latencyFactor}.{' '}
            {Math.abs(v - b8.gain) < 0.05
              ? 'That is what you had.'
              : Math.abs(v - 3) < 0.05 || Math.abs(v - 1 / 3) < 0.05
                ? 'Three is the latency factor; throughput is queries per unit time, and both sides served 8 queries.'
                : 'Use the two formulas above: the numerators are equal, so the ratio is the inverse ratio of the times.'}
          </p>
        )}
      </Predict>

      <h2>Watch one step</h2>
      <p>
        Why is the matrix product so cheap? Watch one decode step of lab 22.2's decoder as the batch grows. Time is drawn
        in passes over the weights ({fmt(wb / 1e6, 3)} MB of them in float32 for the lab's default model), so the picture
        holds for any device whose threshold is the book's example 20 FLOP/B.
      </p>
      <Simplification>
        <p>
          The same model as the roofline explorer below: each piece at the bound of its own minimum traffic, pieces in
          sequence, no launch cost, no cache, no attention. Only relative time is drawn, so no device timing appears.
        </p>
      </Simplification>
      <DecodeStepScene />

      <h2>Find the knee on the roofline</h2>
      <p>
        Chapter 5 gives the tool. A kernel is held back by bandwidth while it performs fewer operations per byte than the
        device's threshold T, peak throughput over peak bandwidth (<a href={href('ch05', '5.1')}>§5.1</a>). Count a K → N
        projection over a batch of B rows in float32, each weight, input and output moved once: 2BKN operations for
        4(KN + BK + BN) bytes, so <code>I(B) = BKN / (2(KN + BK + BN))</code>. While B is much smaller than K and N, that
        is about B/2.
      </p>
      <Beyond>
        <p>
          The count is ours (ch22-facts E1), by chapter 5's method. In 16-bit it gives chapter 20's 1 FLOP/B at batch 1
          and 315 at batch 512 for its 1024 → 4096 layer.
        </p>
      </Beyond>
      <Predict
        id="ch22/22.5/p-book-knee"
        prompt="On the book's example device (threshold 20 FLOP/B, ch. 5 p. 94), what is the smallest batch at which a 4096 → 4096 float32 projection reaches the threshold?"
        unit="rows"
      >
        {(v) => (
          <>
            <p>
              <b data-knee-answer={bWhole}>{bWhole}</b>. Solving I(B) = 20 gives B* = {bBook.toFixed(1)}: a little above
              2T, because the activations add bytes as B grows.{' '}
              {v === bWhole ? 'Exactly.' : Math.abs(v - 40) <= 1 ? 'Close: B/2 = 20 gives 40, and the input and output rows push it to 40.8.' : 'Use I(B) ≈ B/2 for a first estimate, then correct for the activations.'}
            </p>
            <IntensityCurve />
          </>
        )}
      </Predict>
      <p>
        Now the whole step. The explorer runs every piece of lab 22.2's step through that count: the embedding, each
        layer's projection, bias and residual, the LM head, softmax and argmax. Run it on the book's example device, on
        yours if you have run lab 5.1, or on numbers you type. Choose "4096 → 4096" for the projection alone.
      </p>
      <Today checked="2026-11-20 (ch22-facts A4, infra-facts C13)">
        <p>
          NVIDIA's performance guides (2023) make this argument with this formula. Apple publishes no peak figures: on a
          Mac, lab 5.1's measurement is the only threshold.
        </p>
      </Today>
      <DecodeRoofline st={st} />
      <p>
        What to look for. Time per step stays nearly flat until B*, about 2T rows in float32 and about T in 16-bit, while
        tokens per second rise in proportion to B. Past it, tokens per second flatten and every user's time per token
        grows. The per-row pieces (bias, residual, softmax over the vocabulary, argmax) grow from the start and
        are shared by nobody; in lab 22.2's step, with its vocabulary of {fmt(M8_DEFAULT.vocab, 6)}, they are still only
        about {fmt(rowShare, 2)}% of the step at batch 256 on the book's example device. With M3's padding the curve is flat to 128 for a reason
        unrelated to bandwidth. And if a measured point is faster than the model's line, the device moved fewer bytes
        than the model charges, which is caches at work as in §5.1 (or a bandwidth probe that undershot).
      </p>
      <Simplification>
        <p>
          Each piece runs at the roofline bound of its own minimum traffic; pieces do not overlap; no launch cost, no
          cache; attention is left out (it does not batch). NVIDIA itself calls this comparison "a simplified rule of
          thumb". With 8- and 4-bit weights the model keeps activations at 16 bits.
        </p>
      </Simplification>
      <Ask q={questionsS225['ch22/22.5/q-plateau']} />
      <Ask q={questionsS225['ch22/22.5/q-half-bytes']} />

      <h2>What batching does not batch</h2>
      <p>
        Page 539 says batching raises the arithmetic intensity "of the attention heads". Chapter 20 says the attention
        layers "do not benefit from batching" (p. 505). Both hold, depending on whether "attention" includes the QKV and
        output projections. Those carry weights, and they batch. Attention proper multiplies each request's query by that
        request's own keys and values: nothing is shared, so its reads grow with the batch, like the elementwise pieces.
        Chapter 20's version is the one to keep.
      </p>
      <Aside>
        <p>
          Orca (OSDI 2022) batches everything but attention, which has no "model parameters" to reuse, and a footnote
          names both definitions of attention (ch22-facts A6).
        </p>
      </Aside>
      <Ask q={questionsS225['ch22/22.5/q-attention']} />

      <h2>Who pays</h2>
      <p>
        The book says each query experiences a longer projection latency. Longer than what? Than the same query alone, at
        batch 1. That is the book's reasoning; no vendor document we checked states it, and lab 22.2 measures it. But a
        server with eight users does not get to serve each of them alone. Its real alternative is to serve them one after
        another, and then every user waits a whole round between tokens.
      </p>
      <Source section="22.5" pages="539" />
      <Ask q={questionsS225['ch22/22.5/q-who-pays']} />

      <h2>The same trade-off as CPU against GPU</h2>
      <p>
        The book closes the argument with a link to chapter 1. Batching trades the latency of each instance for the
        throughput of the group, which is the trade-off that separates a CPU's design from a GPU's: the CPU has the
        shorter latency for each data element, the GPU the higher throughput over many.
      </p>
      <Source section="22.5" pages="539–540" />
      <Ask q={questionsS225['ch22/22.5/q-cpu-gpu']} />

      <h2>Where you meet it: continuous batching</h2>
      <p>
        A server only lives at a large B if its batch stays full, and requests do not cooperate: they arrive at random and
        want different numbers of tokens. The simple policy, static batching, starts a batch and runs it until its longest
        request is done. The policy real servers use admits and retires requests at every step.
      </p>
      <Beyond>
        <p>
          The book does not discuss serving policies in this chapter; chapter 20 mentions in-flight batching (p. 506).
          This part, the animation and the explorer are ours.
        </p>
      </Beyond>
      <ContinuousBatchingScene />
      <p>
        The explorer runs a thousand requests through a server whose step time comes from the explorer above, so the
        device, model, bytes per weight and padding you chose there apply here.
      </p>
      <Simplification>
        <p>
          No prefill: a request starts decoding when admitted. Step time depends only on the number of requests in the
          step. No KV-cache memory limit, which is what caps real batches (ch. 20, p. 505). A finished request in a static
          batch costs nothing. Arrivals come from a seed.
        </p>
      </Simplification>
      <ServerSlots st={st} />
      <p>
        At heavy load and the same slot limit, continuous batching keeps the slots full and returns short requests early.
        Raising the limit raises tokens per second until the step crosses the knee; past it, each user's time per token
        grows for little more throughput. Under light load the limit hardly matters, because the slots are never full.
      </p>
      <p>
        The servers you are likely to deploy expose that limit. vLLM (CUDA and ROCm) calls it the maximum number of sequences
        per iteration. llama.cpp's server, on CUDA, HIP and Metal, calls it slots, with continuous batching on by
        default. On a Mac, mlx-lm's server batches continuously through a BatchGenerator, 32 requests by default. The
        option names are a snapshot of the projects' sources; the trade-off is this section's.
      </p>
      <Today checked="2026-11-20 (ch22-facts D1 to D4)">
        <p>
          Iteration-level scheduling, in-flight and continuous batching are one technique (TensorRT-LLM docs). The
          limit: vLLM's <code>max_num_seqs</code>; llama-server's <code>-np</code> slots; mlx-lm's{' '}
          <code>--decode-concurrency</code>.
        </p>
      </Today>
      <Ask q={questionsS225['ch22/22.5/q-continuous']} />

      <h2>Batched GEMM is something else</h2>
      <p>
        Libraries use the word "batched" for a different thing. <code>cublasGemmStridedBatchedEx</code> and{' '}
        <code>cublasGemmBatchedEx</code>, hipBLAS's <code>hipblasGemmStridedBatchedEx</code> and{' '}
        <code>hipblasGemmBatchedEx</code>, and Metal Performance Shaders' <code>MPSMatrixMultiplication</code> over a range
        set by <code>batchStart</code> and <code>batchSize</code> all run many independent GEMMs, each with its own
        operands. The book's batching is one GEMM against one shared weight matrix, with M equal to the number of
        requests. Batched GEMM fits products that share nothing, such as each sequence's attention against its own keys.
      </p>
      <p>
        You will see both in llama.cpp. Its CUDA and HIP backend sends a 2-D weight times a batch of tokens to one plain
        GEMM. Up to 8 tokens it normally uses its own matrix-vector kernels instead of a GEMM (
        <code>MMVF_MAX_BATCH_SIZE</code> is 8); for some quantised types on particular GPUs the limit is lower, because
        the matrix kernel wins sooner. Its Metal backend does the same with{' '}
        <code>mul_mv_ext</code> for 2 to 8 tokens, for most weight types. Eight is a code constant, not a device figure: the crossover is
        measured, which is lab 22.2's point.
      </p>
      <Today checked="2026-11-20 (ch22-facts A11, B3, C3, D6)">
        <p>
          The entry points are in cuBLAS 13.5, hipBLAS at ROCm 10.2 and the macOS 27 SDK. llama.cpp's thresholds were read
          from its default branch and change often.
        </p>
      </Today>
      <Ask q={questionsS225['ch22/22.5/q-batched-gemm']} />

      <h2>On your device</h2>
      <p>
        Lab 22.2, milestone M8, times this decode step with your own kernels (M2, M3 or lab 22.1's gather, M0's adds,
        M4's softmax and argmax) at batch sizes from 1 upward, and records time per step and tokens per second. Make its predictions on
        the <a href={href('ch22', 'labs')}>labs page</a> from the explorer above, with your device selected, before you
        run it. Its rows then appear in that explorer.
      </p>
      <p>
        The same measurement in a real engine is llama.cpp's <code>llama-batched-bench</code>. With <code>-tgs</code> it
        decodes the sequences one at a time; without it, each step holds one token from every sequence: the book's
        separate products against one product, on a real model.
      </p>
      <Today checked="2026-11-20 (ch22-facts D3)">
        <p>
          <code>-tgs</code> is in the tool's argument parser, not its README. The README's table is one machine's numbers.
        </p>
      </Today>
      <p>
        The book ends the section, and its own argument, on how it thinks people learn: concepts first in one
        programming model, for solid footing, then generalised. This course adds HIP and Metal on the same footing.
      </p>
      <Source section="22.5" pages="540" />
    </>
  );
}
