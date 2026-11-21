import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { fmt, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BOOK_DEVICE, FORMATS, LAYERS, PIPELINES, bitsPerWeight, byteRatio, bytesPerWeight, ceiling, gemv, kneeBatch, layer,
  mbu, measuredBandwidth, measuredDevices, measuredPipelines, modelBytes, pickRuns, pipelineBytesPerWeight, pipelineRelative, qgemvRows, speedupOverF32,
  thresholdOf, weightOnlyIntensity, type Device, type FormatId, type PipelineId, type QgemvRow,
} from '../../sim/qtraffic';
import { ThreeTrafficsScene } from './ThreeTrafficsScene';

// Section X2.1 (ours): bytes per weight as the roofline knob of decode. A roofline
// explorer by weight format, a Predict and a bar explorer for fused and unfused
// dequantisation, the narrated scene "Three ways to feed one GEMV", and a bandwidth
// utilisation calculator over the reader's own lab X2.1 and X2.2 rows. Every number comes
// from qtraffic.ts.

const FACTS = 'notes/today/x2-facts.md';
const us = (s: number) => fmt(s * 1e6) + ' µs';

export const questionsSX21: Record<string, Question> = {
  'x2/X2.1/q-q8-ceiling': {
    kind: 'choice',
    id: 'x2/X2.1/q-q8-ceiling',
    prompt: <>Q8_0 stores 8-bit codes, a quarter of float32's 32 bits. Why is its speed-up ceiling over float32 3.76 and not 4?</>,
    options: [
      { label: 'Each block of 32 codes carries a 2-byte scale: 34 bytes, not 32.', correct: true },
      { label: 'The int8 multiply is slower than a float multiply on GPUs.', why: 'The ceiling is a bytes ceiling. In a memory-bound GEMV the arithmetic is not what sets the time.' },
      { label: 'The activations are still float32, and they cost a quarter of the bytes.', why: 'The activations are about in + out values against in × out weights: a sliver, not a quarter.' },
      { label: 'Each code is stored in 9 bits, so that it can hold its sign.', why: 'Q8_0 codes are int8, from -127 to 127. The extra half bit per weight is the scale.' },
    ],
    explain: (
      <p>
        34 bytes per 32 weights is 1.0625 bytes per weight, and 4 / 1.0625 = 3.76. Bits per weight include the scales:
        Q8_0 is 8.5, Q4_0 4.5. The block scales are part of what every token reads,
        and they are why block size is a bandwidth decision as well as an accuracy one.
      </p>
    ),
    source: `${FACTS} D1, E1`,
  },
  'x2/X2.1/q-knee-batch': {
    kind: 'numeric',
    id: 'x2/X2.1/q-knee-batch',
    prompt: (
      <>
        On the book's example device (threshold 20 FLOP/B), a 4096-to-4096 layer with Q4_0 weights and 16-bit
        activations: what is the smallest batch at which it reaches 20 FLOP/B? Count weights, inputs and outputs once each.
      </>
    ),
    answer: 6,
    unit: 'rows',
    explain: (
      <p>
        Batch 6 gives {gemv('q4_0', layer('4096x4096'), 6, 2).intensity.toFixed(1)} FLOP/B; batch 5 gives{' '}
        {gemv('q4_0', layer('4096x4096'), 5, 2).intensity.toFixed(1)}. With float32 weights and activations the same layer
        needs batch {kneeBatch('f32', layer('4096x4096'), 4, 20)}. Fewer bytes per weight make a small batch compute-bound
        sooner, and from there on the dequantise work, which this model ignores, starts to count.
      </p>
    ),
    source: 'ch. 5 §5.1, p. 94 (the threshold); our arithmetic (x2-facts E1)',
  },
  'x2/X2.1/q-unfused': {
    kind: 'choice',
    id: 'x2/X2.1/q-unfused',
    prompt: <>Dequantise Q4_0 weights to float32 in one kernel, then run the float32 GEMV, against running the float32 GEMV on float32 weights. Which is faster?</>,
    options: [
      { label: 'The float32 GEMV alone: the other path moves 2.1 times the bytes.', correct: true },
      { label: 'The two-kernel path, by about 7 times: it reads Q4_0.', why: 'It reads Q4_0 once, then writes and reads a float32 copy: 0.5625 + 4 + 4 bytes per weight.' },
      { label: 'They are the same: both end in the same float32 GEMV.', why: 'The same GEMV, plus a whole extra pass that writes its input.' },
      { label: 'The two-kernel path, slightly, because dequantising is cheap.', why: 'Cheap in arithmetic, not in bytes: the copy is as large as the float32 weights.' },
    ],
    explain: (
      <p>
        8.5625 against 4 bytes per weight is {(pipelineBytesPerWeight('unfused-f32', 'q4_0') / 4).toFixed(2)} times, and
        time follows bytes because both kernels are bandwidth-bound: an elementwise conversion and a float32 GEMV at
        batch 1 do almost no arithmetic per byte. The dequantised copy is written to memory and read back, so the quantisation saves nothing on the way. Only a kernel
        that unpacks in registers keeps the byte ratio.
      </p>
    ),
    source: `our bytes model; PyTorch's blog reports the same effect (${FACTS} A1)`,
  },
  'x2/X2.1/q-mbu': {
    kind: 'numeric',
    id: 'x2/X2.1/q-mbu',
    prompt: (
      <>
        A hypothetical device with a peak of 400 GB/s decodes a 7-billion-weight model in Q4_0 at 30 tokens per second.
        What is its model bandwidth utilisation, in percent?
      </>
    ),
    answer: Number((100 * mbu(modelBytes(7e9, 'q4_0'), 1 / 30, 400)).toFixed(1)),
    tol: 0.02,
    unit: '%',
    explain: (
      <p>
        7 × 10⁹ × 0.5625 = 3.94 GB per token, × 30 = 118 GB/s, and 118 / 400 = 29.5 %. Counting the weights only is
        PyTorch's definition; it ignores the KV cache and the activations, which is fine at short context and flattering
        at long context.
      </p>
    ),
    source: `${FACTS} A1 (the definition)`,
  },
  'x2/X2.1/q-int-mm': {
    kind: 'choice',
    id: 'x2/X2.1/q-int-mm',
    prompt: <>Why can batch-1 decode in PyTorch on CUDA not use <code>torch._int_mm</code>, and what serves it instead?</>,
    options: [
      { label: 'On CUDA it refuses a first dimension of 16 or less; weight-only ops serve decode.', correct: true },
      { label: '_int_mm has no CUDA implementation, only CPU and MPS ones.', why: 'It dispatches CPU, CUDA, MPS and XPU. On CUDA it checks the sizes and refuses small M.' },
      { label: 'It would run, but int8 activations lose too much accuracy for decode.', why: 'It does not work: the size check stops it before any arithmetic.' },
      { label: 'It needs int4 weights, packed by _convert_weight_to_int4pack first.', why: 'It multiplies int8 by int8 into int32.' },
    ],
    explain: (
      <p>
        The quantised GEMV and the quantised GEMM are different kernels. A GEMM library is built for many rows; decode has
        one. Weight-only kernels keep the activations in float and only shrink what crosses the bus.
      </p>
    ),
    source: `${FACTS} A6, D6`,
  },
  'x2/X2.1/q-regime-q8': {
    kind: 'choice',
    id: 'x2/X2.1/q-regime-q8',
    prompt: (
      <>
        A Q8_0 GEMV at batch 1 does 2 / 1.0625 = 1.88 FLOP/B. Device A: 200 GFLOP/s and 100 GB/s. Device B: 300 GFLOP/s
        and 250 GB/s. What does the roofline say?
      </>
    ),
    options: [
      { label: 'Memory-bound on A; compute-bound on B, whose threshold is lower.', correct: true },
      { label: 'Memory-bound on both: decode is memory-bound on any device.', why: "B's threshold is 300 / 250 = 1.2 FLOP/B, below 1.88." },
      { label: 'Compute-bound on both, since Q8_0 nearly doubles FP16\'s intensity.', why: "A's threshold is 2.0 FLOP/B, above 1.88." },
      { label: 'Memory-bound on B only, since B has more bandwidth per operation.', why: 'B has more bandwidth per operation, so its threshold is lower.' },
    ],
    explain: (
      <p>
        "Decode is memory-bound" is a statement about the pair of kernel and device. On a device with little arithmetic
        per byte, a quantised GEMV can cross the threshold at batch 1, and then the integer work and the scales, which
        this page's model counts as free, are what you are waiting for.
      </p>
    ),
    source: 'adapted from ch. 5 exercise 9, p. 119',
  },
};

// ---------------------------------------------------------------- hardware
interface Hardware extends Device {
  key: string;
  label: string;
  note: string;
}
const BOOK: Hardware = {
  ...BOOK_DEVICE, key: 'book', label: "The book's example (H100)",
  note: "The book's example device, p. 94: 66.9 TFLOPS, 3.35 TB/s, threshold 20 FLOP/B.",
};

function useHardware() {
  const { reports } = useReports();
  const mine = useMemo(
    () => measuredDevices(reports).map((d): Hardware => ({
      ...d, key: d.backend, label: `Yours: ${d.name} (${d.backend})`,
      note: `Measured by lab 5.1 on ${d.date}: two probe kernels, not the vendor's figures.`,
    })),
    [reports],
  );
  return { reports, options: [BOOK, ...mine] };
}

// ---------------------------------------------------------------- the decode roofline
const EXPLORER_FORMATS: FormatId[] = ['f32', 'f16', 'q8_0', 'q4_0', 'q4_1', 'q4_k', 'mxfp4', 'mlx4_g32', 'mlx4_g64', 'mlx4_g128', 'mlx4_g32_f32', 'mlx4_g64_f32', 'mlx4_g128_f32'];
const fmtLabel = (f: FormatId) => (f.startsWith('mlx') ? FORMATS[f].label : `${FORMATS[f].label}${FORMATS[f].owner.startsWith('llama') || f === 'mxfp4' ? ` (${FORMATS[f].owner})` : ''}`);

interface Mark {
  label: string;
  x: number;
  y?: number;
  kind: 'model' | 'ref' | 'measured';
}

function Roofline({ hw, marks }: { hw: Device; marks: Mark[] }) {
  const W = 680;
  const H = 360;
  const pad = { l: 64, r: 16, t: 14, b: 44 };
  const T = thresholdOf(hw);
  const peak = T * hw.bw;
  const xMin = 0.1;
  const xMax = 1000;
  const yMax = peak * 3;
  const yMin = hw.bw * xMin * 0.5;
  const X = (v: number) => pad.l + ((Math.log10(v) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin))) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * (H - pad.t - pad.b);
  const roof = (x: number) => Math.min(peak, hw.bw * x);
  const yt: number[] = [];
  for (let e = Math.ceil(Math.log10(yMin)); e <= Math.floor(Math.log10(yMax)); e++) yt.push(10 ** e);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Roofline of the decode GEMV" style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      <rect x={pad.l} y={pad.t} width={X(T) - pad.l} height={H - pad.t - pad.b} fill="var(--global-wash)" />
      <text x={pad.l + 8} y={pad.t + 16} fontSize="11" style={{ fill: 'var(--global)' }}>memory-bound</text>
      <text x={X(T) + 8} y={pad.t + 16} fontSize="11" style={{ fill: 'var(--reg)' }}>compute-bound</text>
      {[0.1, 1, 10, 100, 1000].map((v) => (
        <g key={'x' + v}>
          <line x1={X(v)} x2={X(v)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(v)} y={H - pad.b + 15} fontSize="11" textAnchor="middle">{v}</text>
        </g>
      ))}
      {yt.map((v) => (
        <g key={'y' + v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 6} y={Y(v) + 4} fontSize="11" textAnchor="end">{v >= 1000 ? v / 1000 + 'k' : v}</text>
        </g>
      ))}
      <text x={(W + pad.l) / 2} y={H - 8} fontSize="11.5" textAnchor="middle">FLOP per byte of traffic</text>
      <text x={14} y={(H - pad.b + pad.t) / 2} fontSize="11.5" textAnchor="middle" transform={`rotate(-90 14 ${(H - pad.b + pad.t) / 2})`}>GFLOP/s</text>
      <polyline points={`${X(xMin)},${Y(hw.bw * xMin)} ${X(T)},${Y(peak)} ${X(xMax)},${Y(peak)}`} fill="none" stroke="var(--ink)" strokeWidth="2.2" />
      <line x1={X(T)} x2={X(T)} y1={Y(peak)} y2={H - pad.b} stroke="var(--ink)" strokeDasharray="3 3" />
      <text x={X(T)} y={H - pad.b - 6} fontSize="11" textAnchor="middle">threshold {fmt(T)}</text>
      {marks.map((m, i) => {
        const x = Math.min(Math.max(m.x, xMin), xMax);
        const cx = X(x);
        if (m.kind !== 'measured') {
          const cy = Y(roof(x));
          return (
            <g key={i} data-mark={m.kind} data-x={m.x.toFixed(3)}>
              <circle cx={cx} cy={cy} r="5" fill={m.kind === 'model' ? 'var(--ink)' : 'var(--page)'} stroke="var(--ink)" strokeWidth="1.8" />
              <text x={m.kind === 'model' ? cx + 8 : cx - 8} y={cy - 8} fontSize="11" textAnchor={m.kind === 'model' ? 'start' : 'end'}>{m.label}</text>
            </g>
          );
        }
        const above = (m.y ?? 0) > roof(x) * 1.02;
        return (
          <g key={i} data-mark="measured">
            <rect x={cx - 4.5} y={Y(m.y!) - 4.5} width="9" height="9" transform={`rotate(45 ${cx} ${Y(m.y!)})`} fill={above ? 'var(--fail)' : 'var(--ok)'} />
            <text x={cx + 9} y={Y(m.y!) + 14} fontSize="11" style={{ fill: above ? 'var(--fail)' : 'var(--ok)' }}>{m.label}</text>
          </g>
        );
      })}
    </svg>
  );
}

function DecodeRoofline() {
  const { reports, options } = useHardware();
  const [key, setKey] = useState('book');
  const [custom, setCustom] = useState({ peak: 10000, bw: 500 });
  const [f, setF] = useState<FormatId>('q4_0');
  const [batch, setBatch] = useState(1);
  const [lid, setLid] = useState('4096x4096');
  const [act, setAct] = useState(2);
  const hw: Hardware = key === 'custom' ? { key, label: 'Custom', ...custom, note: 'Numbers you typed.' } : (options.find((o) => o.key === key) ?? BOOK);
  const L = layer(lid);
  const t = gemv(f, L, batch, act);
  const c = ceiling(t, hw);
  const ref = gemv('f32', L, batch, act);
  const cRef = ceiling(ref, hw);
  const sp = speedupOverF32(f, L, batch, act, hw);
  const T = thresholdOf(hw);
  const knee = kneeBatch(f, L, act, T);
  const kneeF32 = kneeBatch('f32', L, 4, T);
  const rows = pickRuns(qgemvRows(reports), hw.key, L, batch, f);
  const runF = rows.find((r) => r.format === f);
  const runRef = rows.find((r) => r.format === 'f32');
  const marks: Mark[] = [
    { label: `${FORMATS[f].label}, batch ${batch}`, x: t.intensity, kind: 'model' },
    ...(f === 'f32' ? [] : [{ label: `float32 weights, batch ${batch}`, x: ref.intensity, kind: 'ref' as const }]),
    ...rows.map((r): Mark => ({ label: `yours, ${FORMATS[r.format].label} (lab ${r.labNum})`, x: r.intensity, y: r.gflops, kind: 'measured' })),
  ];
  return (
    <Bench
      title="Decode GEMV roofline by weight format"
      task="Pick a format and drag the batch. Find where the speed-up stops being the byte ratio."
      foot={
        <>
          Log scales. The filled circle is the format you picked, the open circle float32 weights with the same
          activations and batch. Filled diamonds are your own lab X2.1 or X2.2 runs of this layer, format and batch, with float32 beside them, when this device has them. {hw.note}
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Hardware
          <select value={key} onChange={(e) => setKey(e.target.value)} aria-label="Hardware">
            {options.map((o) => <option key={o.key} value={o.key}>{o.label}</option>)}
            <option value="custom">Custom numbers</option>
          </select>
        </label>
        {key === 'custom' ? (
          <>
            <label className="field">
              Peak (GFLOP/s)
              <input type="number" min="1" value={custom.peak} aria-label="Peak" onChange={(e) => setCustom({ ...custom, peak: Math.max(1, +e.target.value) })} />
            </label>
            <label className="field">
              Bandwidth (GB/s)
              <input type="number" min="1" value={custom.bw} aria-label="Bandwidth" onChange={(e) => setCustom({ ...custom, bw: Math.max(1, +e.target.value) })} />
            </label>
          </>
        ) : null}
        <label className="field">
          Weight format
          <select value={f} onChange={(e) => setF(e.target.value as FormatId)} aria-label="Weight format">
            {EXPLORER_FORMATS.map((x) => <option key={x} value={x}>{fmtLabel(x)}</option>)}
          </select>
        </label>
        <label className="field">
          Layer
          <select value={lid} onChange={(e) => setLid(e.target.value)} aria-label="Layer">
            {LAYERS.map((l) => <option key={l.id} value={l.id}>{l.label}</option>)}
          </select>
        </label>
        <label className="field">
          Activations
          <span className="seg">
            {[4, 2].map((a) => (
              <button key={a} aria-pressed={act === a} onClick={() => setAct(a)}>{a === 4 ? 'float32' : '16-bit'}</button>
            ))}
          </span>
        </label>
        <label className="field">
          Batch = {batch}
          <input type="range" min="1" max="64" value={batch} aria-label="Batch" onChange={(e) => setBatch(+e.target.value)} />
        </label>
      </div>
      <div className="cols plot">
        <Roofline hw={hw} marks={marks} />
        <div className="panel" data-panel="roofline">
          <p style={{ margin: 0 }}>
            {FORMATS[f].label}: {bytesPerWeight(f)} bytes, {bitsPerWeight(f)} bits per weight. Weights {fmt(t.weightBytes / 1e6)} MB,
            activations and outputs {fmt((t.inputBytes + t.outputBytes) / 1e3)} kB.
          </p>
          <p style={{ fontSize: '1.4rem', margin: '0.5rem 0 0.2rem', fontWeight: 600 }} data-intensity={t.intensity.toFixed(3)}>
            {fmt(t.intensity)} FLOP/B
          </p>
          <p style={{ margin: '0 0 0.6rem' }}>
            {c.bound === 'memory' ? <span className="chip global">memory-bound</span> : <span className="chip reg">compute-bound</span>}{' '}
            <span className="muted">threshold {fmt(T)} FLOP/B</span>
          </p>
          <table className="data" style={{ margin: 0 }}>
            <tbody>
              <tr><td>Ceiling time, this layer</td><td className="num nowrap" data-time>{us(c.seconds)}</td></tr>
              <tr><td>Float32 weights, same activations</td><td className="num nowrap">{us(cRef.seconds)}</td></tr>
              <tr><td>Predicted speed-up over float32</td><td className="num" data-speedup={sp.toFixed(3)}><b>{sp.toFixed(2)}×</b></td></tr>
              <tr><td>Byte ratio of the weights alone</td><td className="num" data-ratio={byteRatio(f).toFixed(3)}>{byteRatio(f).toFixed(2)}×</td></tr>
              <tr><td>Batch where this format turns compute-bound</td><td className="num" data-knee={knee ?? 'none'}>{knee ?? 'never'}</td></tr>
              {runF && runRef && f !== 'f32' ? (
                <tr>
                  <td>Your measured speed-up (lab {runF.labNum})</td>
                  <td className="num" data-measured-speedup={(runRef.seconds / runF.seconds).toFixed(3)}><b>{(runRef.seconds / runF.seconds).toFixed(2)}×</b></td>
                </tr>
              ) : null}
              <tr><td>The same for float32 weights and activations</td><td className="num" data-knee-f32={kneeF32 ?? 'none'}>{kneeF32 ?? 'never'}</td></tr>
            </tbody>
          </table>
          {rows.length === 0 && key !== 'book' && key !== 'custom' ? (
            <p className="muted" style={{ margin: '0.6rem 0 0' }} data-no-runs>
              No run of this layer, format and batch on this device. Labs X2.1 and X2.2 time float32, Q8_0 and Q4_0 at
              batches 1, 4 and 8: <code>python3 labs/pmpp.py run X2.2 --bench</code>.
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- fused or unfused
const PIPE_FORMATS: FormatId[] = ['q4_0', 'q8_0', 'mlx4_g64'];

function FusedOrUnfused() {
  const { reports } = useReports();
  const [f, setF] = useState<FormatId>('q4_0');
  const [on, setOn] = useState<Record<PipelineId, boolean>>({ f32: true, 'unfused-f32': true, 'unfused-f16': false, fused: false });
  const L = layer('4096x4096');
  const maxBpw = Math.max(...PIPELINES.map((p) => pipelineBytesPerWeight(p.id, f)));
  const rows = qgemvRows(reports);
  const measured = f === 'q4_0'
    ? [...new Set(rows.map((r) => r.backend))].flatMap((b) => { const m = measuredPipelines(rows, b, L, 1); return m ? [{ b, m }] : []; })
    : [];
  return (
    <Bench
      title="Fused or unfused"
      task="Switch the four pipelines on and off. Which ones keep the byte ratio?"
      foot={<>Bytes model: a 4096 → 4096 layer at batch 1 with float32 activations, every byte at the same bandwidth, no caches. Time is relative to the float32 GEMV. With Q4_0, your own lab X2.2 times for the same layer appear beside the model where they exist.</>}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Quantised format
          <select value={f} onChange={(e) => setF(e.target.value as FormatId)} aria-label="Quantised format">
            {PIPE_FORMATS.map((x) => <option key={x} value={x}>{fmtLabel(x)}</option>)}
          </select>
        </label>
        {PIPELINES.map((p) => (
          <label key={p.id} className="check">
            <input type="checkbox" checked={on[p.id]} onChange={(e) => setOn({ ...on, [p.id]: e.target.checked })} /> {p.label}
          </label>
        ))}
      </div>
      <div className="panel" data-panel="pipelines">
        {PIPELINES.filter((p) => on[p.id]).map((p) => {
          const bpw = pipelineBytesPerWeight(p.id, f);
          const rel = pipelineRelative(p.id, f, L, 1, 4);
          return (
            <div key={p.id} data-pipeline={p.id} data-bpw={bpw} data-rel={rel.toFixed(3)} style={{ margin: '0 0 0.8rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap' }}>
                <span>{p.label}{p.kernels > 1 ? <span className="muted"> (two kernels)</span> : null}</span>
                <span className="num">
                  {bpw} bytes per weight, <b>{rel.toFixed(2)}×</b> the float32 time
                  {measured.map(({ b, m }) => (m[p.id] != null ? (
                    <span key={b} data-measured={b} data-value={m[p.id]!.toFixed(3)}>; measured on your {b}: <b>{m[p.id]!.toFixed(2)}×</b></span>
                  ) : null))}
                </span>
              </div>
              <svg viewBox="0 0 600 16" preserveAspectRatio="none" style={{ width: '100%', height: 16, display: 'block' }} aria-hidden="true">
                <rect x="0" y="2" width={(600 * bpw) / maxBpw} height="12" fill={rel > 1.001 ? 'var(--fail)' : rel < 0.999 ? 'var(--ok)' : 'var(--ink-3)'} />
              </svg>
            </div>
          );
        })}
        {PIPELINES.every((p) => !on[p.id]) ? <p className="muted" style={{ margin: 0 }}>Switch a pipeline on.</p> : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- bandwidth utilisation
function UtilisationCalc() {
  const { reports } = useReports();
  const rows = qgemvRows(reports);
  const [sel, setSel] = useState(0);
  const r: QgemvRow | undefined = rows[Math.min(sel, rows.length - 1)];
  const bw = r ? measuredBandwidth(reports, r.backend) : null;
  const [hand, setHand] = useState({ params: 8, f: 'q8_0' as FormatId, tps: 20, bw: 300 });
  const handBytes = modelBytes(hand.params * 1e9, hand.f);
  const handMbu = mbu(handBytes, 1 / Math.max(hand.tps, 1e-9), Math.max(hand.bw, 1e-9));
  return (
    <Bench
      title="Your bandwidth utilisation"
      task="Pick one of your runs, or type a model, a format, a speed and a bandwidth."
      foot={<>Weight bytes over time, over the bandwidth lab 5.1 measured on the same device: PyTorch's definition of model bandwidth utilisation ({FACTS} A1). Never a vendor's figure.</>}
    >
      <div className="cols">
        <div className="panel" data-panel="mbu-runs">
          <p style={{ margin: '0 0 0.5rem' }}><b>From your labs</b></p>
          {rows.length ? (
            <>
              <label className="field">
                Run
                <select value={sel} onChange={(e) => setSel(+e.target.value)} aria-label="Run">
                  {rows.map((x, i) => (
                    <option key={i} value={i}>lab {x.labNum}, {x.backend}, {FORMATS[x.format].label} ({x.kernel}), {x.in} → {x.out}, batch {x.batch}</option>
                  ))}
                </select>
              </label>
              {r ? (
                <table className="data" style={{ margin: '0.5rem 0 0' }}>
                  <tbody>
                    <tr><td>Weight bytes</td><td className="num">{fmt(r.weightBytes / 1e6)} MB</td></tr>
                    <tr><td>Time</td><td className="num">{us(r.seconds)}</td></tr>
                    <tr><td>Weight bandwidth</td><td className="num">{fmt(r.weightGBs)} GB/s</td></tr>
                    <tr>
                      <td>Of your lab 5.1 bandwidth</td>
                      <td className="num" data-mbu-run>{bw ? `${(100 * mbu(r.weightBytes, r.seconds, bw)).toFixed(1)} %` : 'run lab 5.1 on this device'}</td>
                    </tr>
                  </tbody>
                </table>
              ) : null}
            </>
          ) : (
            <p style={{ margin: 0 }} data-empty="runs">
              No runs of labs X2.1 or X2.2 on this machine yet. <code>python3 labs/pmpp.py run X2.1 --bench</code> and{' '}
              <code>python3 labs/pmpp.py run X2.2 --bench</code> produce them, and <code>python3 labs/pmpp.py run 5.1 --bench</code>{' '}
              measures the bandwidth they are compared with (<a href={href('x2', 'labs')}>labs</a>).
            </p>
          )}
        </div>
        <div className="panel" data-panel="mbu-hand">
          <p style={{ margin: '0 0 0.5rem' }}><b>By hand</b> <span className="muted">(a hypothetical device)</span></p>
          <div className="row">
            <label className="field">
              Weights, billions
              <input type="number" min="0.1" step="0.1" value={hand.params} aria-label="Weights, billions" onChange={(e) => setHand({ ...hand, params: +e.target.value })} />
            </label>
            <label className="field">
              Format
              <select value={hand.f} aria-label="Format" onChange={(e) => setHand({ ...hand, f: e.target.value as FormatId })}>
                {(['f16', 'q8_0', 'q4_0', 'q4_k', 'mlx4_g64'] as FormatId[]).map((x) => <option key={x} value={x}>{FORMATS[x].label.replace('MLX affine 4-bit', 'MLX 4-bit')}</option>)}
              </select>
            </label>
            <label className="field">
              Tokens per second
              <input type="number" min="0.1" step="0.1" value={hand.tps} aria-label="Tokens per second" onChange={(e) => setHand({ ...hand, tps: +e.target.value })} />
            </label>
            <label className="field">
              Peak bandwidth, GB/s
              <input type="number" min="1" value={hand.bw} aria-label="Peak bandwidth" onChange={(e) => setHand({ ...hand, bw: +e.target.value })} />
            </label>
          </div>
          <p style={{ margin: '0.5rem 0 0' }}>
            {fmt(handBytes / 1e9)} GB per token × {fmt(hand.tps)} = {fmt((handBytes * hand.tps) / 1e9)} GB/s:{' '}
            <b data-mbu-hand={(100 * handMbu).toFixed(1)}>{(100 * handMbu).toFixed(1)} %</b> of the peak.
            {handMbu > 1 ? <span className="chip fail" style={{ marginLeft: '0.4rem' }}>above 100 %: something is cached or the numbers disagree</span> : null}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the section
const TABLE_FORMATS: FormatId[] = ['f32', 'f16', 'q8_0', 'q4_0', 'q4_1', 'q4_k', 'mxfp4', 'mlx4_g32', 'mlx4_g64', 'mlx4_g128', 'mlx4_g64_f32'];

export function SX21() {
  const q = questionsSX21;
  const book = gemv('f16', layer('1024x4096'), 1, 2);
  return (
    <>
      <p>
        A matrix-vector product does one multiply and one add for each weight and reads each weight once. Its intensity is
        2 divided by the bytes per weight, whatever the code looks like. For the book's layer, 1024 inputs to 4096 outputs in
        16-bit at batch 1, that is {book.intensity.toFixed(3)} FLOP/B: the book's "about 1". Decode at batch 1 is made of
        such products, and it sits far to the left of any GPU's threshold.
      </p>
      <Source section="5.1" pages="93–97" />
      <p>
        Chapter 5 gave two ways to the right on the roofline: reuse each byte for more operations, or fetch fewer bytes per
        operation. Chapters 20 and 22 took the first, batching. This page takes the second.
      </p>
      <Source section="20.6" pages="504–505" />

      <h2>Fewer bytes per operation</h2>
      <p>
        Appendix A gives the argument in one sentence: a smaller format fetches fewer bytes for each arithmetic operation, so
        the application's intensity rises. The book makes it about floating-point arithmetic.
      </p>
      <Source section="A.5" pages="593" />
      <p>
        Here it is applied to the weights alone, stored as small integers with a scale per block, while the arithmetic
        stays in float or int32. The activations keep their format; only what crosses the bus shrinks.
      </p>
      <Beyond title="Ours, not the book's">
        <p>
          Applying §A.5 to integer weight formats is ours. PyTorch's own blog makes the same argument in your terms: the
          parameter count and the bandwidth are fixed, "But, we can change how many bytes each parameter is stored in!" (
          {FACTS} A1).
        </p>
      </Beyond>

      <h2>Bytes per weight, scales included</h2>
      <p>
        A quantised format is never just its code width. Every block of codes carries what is needed to turn them back into
        numbers: a scale, sometimes a minimum or a bias. Those bytes cross the bus on every token too.
      </p>
      <table className="data wide" data-table="bpw" style={{ maxWidth: '50rem' }}>
        <thead>
          <tr>
            <th>Format</th>
            <th>Whose</th>
            <th className="num">Block</th>
            <th className="num">Bytes per block</th>
            <th className="num">Bytes per weight</th>
            <th className="num">GEMV FLOP/B</th>
          </tr>
        </thead>
        <tbody>
          {TABLE_FORMATS.map((f) => (
            <tr key={f} data-format={f}>
              <td>{FORMATS[f].label.replace('MLX affine 4-bit, ', 'affine 4-bit, ')}</td>
              <td>{FORMATS[f].owner}</td>
              <td className="num">{FORMATS[f].blockWeights}</td>
              <td className="num">{FORMATS[f].blockBytes}</td>
              <td className="num" data-bpw>{bytesPerWeight(f)}</td>
              <td className="num">{fmt(weightOnlyIntensity(f))}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p>
        Q4_K gets to 4.5 bits with a scale and a minimum for every 32 weights by storing those in 6 bits each under one
        16-bit pair per 256. MXFP4's scale is a single byte, a power of two. Section{' '}
        <a href={href('x2', 'X2.2')}>X2.2</a> builds these formats; for now they are just byte counts.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${FACTS}, D1, D2, D5, E1; appA-facts D2)`}>
        <p>
          llama.cpp: <code>block_q8_0</code> 34 bytes, <code>block_q4_0</code> 18, <code>block_q4_1</code> 20 per 32
          weights; Q4_K 144 per 256. MLX: affine groups of 32, 64 or 128, scales and biases of the input's type.
        </p>
      </Today>

      <h2>The roofline of decode, by format</h2>
      <p>
        Pick a format and leave the batch at 1. The point moves right as the bytes fall, but on any real device it stays on
        the slope, so time is bytes over bandwidth, and the predicted speed-up over float32 is the ratio of bytes. Then
        drag the batch.
      </p>
      <Simplification>
        <p>
          Weights, inputs and outputs are each counted once, at full bandwidth. No caches, no launch overhead, and the
          unpacking and scaling are free. That last one stops being true near the threshold.
        </p>
      </Simplification>
      <DecodeRoofline />
      <p>
        Three things to find. First, at batch 1 the speed-up is the byte ratio to within the activations' sliver: 3.76 for
        Q8_0, 7.11 for Q4_0. Second, the scales show: MLX's 4-bit with float32 scales in groups of 32 is a long way short
        of Q4_0. Third, the knee moves: on the book's example a 4096-to-4096 layer turns compute-bound at batch 41 with
        float32, and at a single-digit batch with Q4_0 and 16-bit activations. Past the knee the model says nothing
        useful, because the work it treats as free is then what you wait for.
      </p>
      <p>
        If you have run labs X2.1 and X2.2, pick your device and a batch of 1, 4 or 8. Your measured speed-up can land
        below the byte ratio, or above it. Above it means your float32 kernel sat further below the bandwidth slope than
        your quantised one: compare the two diamonds with the roof, not with each other.
      </p>
      <Ask q={q['x2/X2.1/q-q8-ceiling']} />
      <Ask q={q['x2/X2.1/q-knee-batch']} label="Work it out" />
      <Ask q={q['x2/X2.1/q-regime-q8']} label="From chapter 5" />

      <h2>Fused or unfused</h2>
      <p>
        The byte ratio is a ceiling, and it assumes the quantised weights are what the GEMV reads. The obvious first
        implementation does something else: one kernel turns the codes back into floats and writes them out, and the
        float GEMV you already have reads them.
      </p>
      <Predict
        id="x2/X2.1/p-unfused"
        unit="× the float32 time"
        prompt={
          <>
            Dequantise Q4_0 weights to float32 in one kernel, then run lab 22.1's float32 GEMV on the result. How long does
            that take, as a multiple of the float32 GEMV on float32 weights? (1 means the same; 0.14 would be the full
            byte ratio.)
          </>
        }
      >
        {(c) => {
          const rel = pipelineRelative('unfused-f32', 'q4_0', layer('4096x4096'), 1, 4);
          return (
            <p data-reveal="unfused">
              <b>{rel.toFixed(2)} times</b>, slower than not quantising at all
              {Math.abs(c - rel) < 0.15 ? ', as you said' : c < 1 ? '. If you predicted a speed-up, follow the bytes' : ''}. The
              first kernel reads 0.5625 bytes per weight and writes 4; the second reads those 4 back. That is{' '}
              {pipelineBytesPerWeight('unfused-f32', 'q4_0')} bytes per weight against float32's 4. The explorer below
              has the other pipelines.
            </p>
          );
        }}
      </Predict>
      <FusedOrUnfused />
      <p>
        Dequantising to FP16 instead halves the copy and still loses to float32. Only the fused kernel, which reads codes
        and scales and unpacks them in registers on the way to the multiply, moves fewer bytes than it started with.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${FACTS}, A1)`}>
        <p>
          PyTorch's blog found an unfused int8 path "much worse than even the fp16 performance", on their device.
        </p>
      </Today>
      <p>
        Watch the three streams drain through the same pipe. The byte ratio is a ceiling, not a promise: the same blog's
        fused kernels stayed well short of it.
      </p>
      <Simplification>
        <p>Bytes only, at one fixed bandwidth: no unpacking cost, no caches, no launch overhead.</p>
      </Simplification>
      <ThreeTrafficsScene />
      <Ask q={q['x2/X2.1/q-unfused']} />

      <h2>Reporting a measurement</h2>
      <p>
        A time on its own says little. Report what fraction of your bandwidth the weights used: weight bytes per pass,
        times passes per second, over your peak bandwidth. PyTorch calls it model bandwidth utilisation. Use the peak lab
        5.1 measured on the same device, never a datasheet's. A GEMV near 100 % has nothing left to gain from code; one at
        30 % does.
      </p>
      <Ask q={q['x2/X2.1/q-mbu']} label="Work it out" />
      <UtilisationCalc />
      <p>
        The <a href={href('x2', 'labs')}>labs page</a> computes the same number for labs X2.1, X2.2 and X2.4 from their
        benchmark rows.
      </p>

      <h2>The library GEMM does not serve decode</h2>
      <p>
        One more trap before writing kernels. An int8 matrix multiply exists in every stack, and it is not the kernel decode
        needs: at one token there is one row.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${FACTS}, A6, D6)`}>
        <p>
          <code>torch._int_mm</code> on CUDA: <code>TORCH_CHECK(self.size(0) &gt; 16, ...)</code>. cuBLAS int8 needs
          4-byte alignment and leading dimensions that are multiples of 4. Weight-only ops serve batch 1.
        </p>
      </Today>
      <Ask q={q['x2/X2.1/q-int-mm']} />
    </>
  );
}
