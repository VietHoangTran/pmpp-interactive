import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { fmt, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { gpuBackends, measuredDevices, thresholdOf, type Device, type ReportsLike, type StepShape } from '../../sim/batching';
import { MAT_FORMAT_LABEL, type MatFormat } from '../../sim/quant';
import {
  BOOK_DEVICE,
  M8_DEFAULT,
  STEP_FORMATS,
  STEP_FORMAT_LABEL,
  TOY,
  bandwidthUtilisation,
  greedyRun,
  matrixWeightBytes,
  measuredSpeedup,
  measuredSteps,
  predictedSpeedup,
  stepBytes,
  toyModel,
  toyWeightBytes,
  type StepFormat,
  type StepFormats,
  type StepBytes,
} from '../../sim/qstep';

const FACTS = 'notes/today/x2-facts.md';
const CHECKED = (items: string) => `2026-11-20 in the facts pass (${FACTS}, ${items})`;
const MB = (b: number) => `${(b / 1e6).toFixed(1)} MB`;
const fmtTime = (s: number) => (s < 1e-3 ? `${fmt(s * 1e6)} µs` : `${fmt(s * 1e3)} ms`);
const F32F: StepFormats = { layers: 'f32', embed: 'f32' };
const ALLQ4: StepFormats = { layers: 'q4_0', embed: 'q4_0' };
const LAYERSQ4: StepFormats = { layers: 'q4_0', embed: 'f32' };

// ---------------------------------------------------------------- questions
const ratio = (f: StepFormats) => matrixWeightBytes(M8_DEFAULT, F32F) / matrixWeightBytes(M8_DEFAULT, f);
export const questionsSX27: Record<string, Question> = {
  'x2/X2.7/q-step-ratio': {
    kind: 'numeric',
    id: 'x2/X2.7/q-step-ratio',
    prompt: (
      <>
        Lab 22.2's default step: 2 layers of 4096 × 4096 and an LM head of 32,000 × 4096, all float32. Quantise only the
        two layers to Q4_0 and keep the head in float32. By how many times do the weight bytes per step shrink? (Two
        decimals.)
      </>
    ),
    answer: Number(ratio(LAYERSQ4).toFixed(2)),
    tol: 0.01,
    unit: '×',
    explain: (
      <p>
        Float32: 4 × (2 × 4096² + 32,000 × 4096) = {MB(matrixWeightBytes(M8_DEFAULT, F32F))}. Layers in Q4_0 at 0.5625
        bytes per weight: 18.9 MB, plus the head's 524.3 MB = {MB(matrixWeightBytes(M8_DEFAULT, LAYERSQ4))}. Ratio{' '}
        {ratio(LAYERSQ4).toFixed(2)}. Quantise the head too and it is {ratio(ALLQ4).toFixed(2)}: the head is 131 million of
        the 165 million weights.
      </p>
    ),
    source: `ours: ${FACTS} E1 bytes per weight; lab 22.2's default shape (labs/ch22/lab2_batching)`,
  },
  'x2/X2.7/q-embed': {
    kind: 'choice',
    id: 'x2/X2.7/q-embed',
    prompt: <>Once the embedding table E is quantised, why does the step need a gather that dequantises?</>,
    options: [
      { label: 'E is tied to the LM head: quantising it for the head means the gather reads blocks too, and keeping a float copy would cost the bytes just saved.', correct: true },
      { label: 'Because the gather is the slowest piece of the step.', why: 'At batch 1 it reads one row of E: about 16 KB in float32, nothing next to the head.' },
      { label: 'Because embeddings must be float32 for the softmax.', why: 'The gather feeds the first layer, not the softmax; and its output is float32 either way.' },
    ],
    explain: (
      <p>
        In lab 22.2 the LM head multiplies by E itself. Quantise E for the head and the same bytes are what the gather must
        read: one row of blocks per sequence, dequantised into H. A second, float copy of E would be 524 MB at the
        default shape, which is most of what quantising saved.
      </p>
    ),
    source: 'ours; lab 22.2’s tied LM head (labs/ch22/lab2_batching/decode.h)',
  },
  'x2/X2.7/q-margin': {
    kind: 'choice',
    id: 'x2/X2.7/q-margin',
    prompt: <>Which greedy steps are at risk when the weights are quantised?</>,
    options: [
      { label: 'Those whose float32 top-2 logit margin is smaller than the change quantisation makes to the logits.', correct: true },
      { label: 'The first steps, because errors are largest there.', why: 'Each step here starts from a token, not from accumulated state; the risk is per step.' },
      { label: 'The steps whose token has the largest logit.', why: 'A large winning logit is fine if the runner-up is far behind.' },
      { label: 'All of them equally.', why: 'In the toy model a step whose margin is several times the logits error never flips.' },
    ],
    explain: (
      <p>
        Greedy decoding keeps the argmax. Quantisation moves every logit a little; if the winner's lead over some rival
        is smaller than how far the two move relative to each other, the rival wins. Steps with a wide margin are safe
        whatever the error. In the toy model, look at which bars are red.
      </p>
    ),
    source: 'ours (toy model in app/src/sim/qstep.ts)',
  },
  'x2/X2.7/q-weights-vs-output': {
    kind: 'choice',
    id: 'x2/X2.7/q-weights-vs-output',
    prompt: <>Q4_0 has about 9% relative RMS error on normal weights. Can you predict the error in the logits from that?</>,
    options: [
      { label: 'No: errors partly cancel in each dot product, pass through every layer and the residual, and must be measured at the output.', correct: true },
      { label: 'Yes: also about 9%, because the logits are linear in the weights.', why: 'They are not linear in the weights: each layer feeds the next, and E appears twice, at the gather and at the head.' },
      { label: 'Yes: 9% times the number of layers.', why: 'Errors do not add like that; some cancel, and the residual carries the unquantised part forward.' },
    ],
    explain: (
      <p>
        In the toy model the weights' error is about 9% and the logits' error per step comes out larger, and it differs
        from step to step. A real model's sensitivity differs again. That is why PyTorch's and llama.cpp's own claims are
        about outputs (accuracy, perplexity), and why lab X2.4 checks logits and tokens, not weights.
      </p>
    ),
    source: `ours; ${FACTS} E2 for the 9%`,
  },
  'x2/X2.7/q-batch8': {
    kind: 'choice',
    id: 'x2/X2.7/q-batch8',
    prompt: (
      <>
        At batch 8 every weight read serves 8 rows. In the bytes model, does Q4_0's predicted speed-up over float32 grow
        or shrink from batch 1 to batch 8, and why measure it anyway?
      </>
    ),
    options: [
      {
        label: 'It shrinks: activation bytes grow with the batch while weight bytes do not, and once the Q4_0 GEMV’s intensity passes the device threshold it stops being bandwidth-bound; the dequantise arithmetic, absent from the model, decides the rest.',
        correct: true,
      },
      { label: 'It grows, because the dequantise work is shared by 8 rows.', why: 'The model has no dequantise work at all; and sharing it would only cut a cost, not raise the byte ratio.' },
      { label: 'It stays at the byte ratio, 7.11.', why: 'Only the weight bytes shrink 7.11 times; the rest of the step does not shrink at all.' },
    ],
    explain: (
      <p>
        At batch 1 the step is nearly all weight bytes, so the ratio is near 7.1. At batch 8 the logits, softmax and
        argmax move 8 times as many bytes, and on the book's example device a Q4_0 GEMV at batch 8 does 2 × 8 / 0.5625 ≈
        28 operations per byte of weights, above its threshold of 20: the head becomes compute-bound in the model. Real
        kernels also spend arithmetic unpacking and scaling, which the model leaves out. Only a measurement says how much
        of the ratio survives.
      </p>
    ),
    source: `ours; ${FACTS} E1, A1; the book's example device from ch. 5 p. 94`,
  },
};

// ---------------------------------------------------------------- explorer: where the step's bytes go
interface DevOption { key: string; label: string; dev: Device; backend?: string }
const BOOK_OPTION: DevOption = { key: 'book', label: "The book's example (H100)", dev: BOOK_DEVICE };

function useDevices() {
  const { reports } = useReports();
  const rep = reports as unknown as ReportsLike;
  const options: DevOption[] = useMemo(
    () => [BOOK_OPTION, ...measuredDevices(rep).map((m) => ({ key: m.backend, backend: m.backend, label: `Yours: ${m.name} (${m.backend}, lab 5.1)`, dev: { peak: m.peak, bw: m.bw } }))],
    [rep],
  );
  return { rep, options };
}

function StackedBar({ s, max, label }: { s: StepBytes; max: number; label: string }) {
  const Wd = 600;
  let x = 0;
  return (
    <div style={{ marginBottom: '0.5rem' }}>
      <div className="muted" style={{ marginBottom: '0.15rem' }}>{label}: {MB(s.bytes)}, {MB(s.weightBytes)} of it weights</div>
      <svg viewBox={`0 0 ${Wd} 22`} role="img" aria-label={`Bytes per piece, ${label}`} style={{ width: '100%', display: 'block' }} data-bar={label} data-bytes={s.bytes}>
        {s.pieces.map((p) => {
          const w = (p.weightBytes / max) * Wd;
          const o = ((p.bytes - p.weightBytes) / max) * Wd;
          const g = (
            <g key={p.key} data-piece={p.key}>
              <rect x={x} y={2} width={w} height={18} fill="var(--global)" />
              <rect x={x + w} y={2} width={o} height={18} fill="var(--global-wash)" stroke={o > 1 ? 'var(--global)' : 'none'} strokeWidth={0.5} />
              {w + o > 50 ? <text x={x + 4} y={15} fontSize="10" style={{ fill: w > 40 ? 'var(--page)' : 'var(--ink)' }}>{p.label}</text> : null}
            </g>
          );
          x += w + o;
          return g;
        })}
      </svg>
    </div>
  );
}

function StepExplorer() {
  const { rep, options } = useDevices();
  const [devKey, setDevKey] = useState('book');
  const [custom, setCustom] = useState<Device>({ peak: 10000, bw: 400 });
  const [shapeKey, setShapeKey] = useState<'m8' | 'custom'>('m8');
  const [cshape, setCshape] = useState<StepShape>({ kind: 'step', d: 2048, layers: 8, vocab: 32000 });
  const [layers, setLayers] = useState<StepFormat>('q4_0');
  const [embed, setEmbed] = useState<StepFormat>('f32');
  const [B, setB] = useState(1);
  const opt = devKey === 'custom' ? { key: 'custom', label: 'Custom', dev: custom } : (options.find((o) => o.key === devKey) ?? BOOK_OPTION);
  const shape = shapeKey === 'm8' ? M8_DEFAULT : cshape;
  const fm: StepFormats = { layers, embed };
  const base = stepBytes(shape, F32F, B, opt.dev);
  const q = stepBytes(shape, fm, B, opt.dev);
  const speed = base.seconds / q.seconds;
  const wratio = matrixWeightBytes(shape, F32F) / matrixWeightBytes(shape, fm);
  const meas = opt.backend ? measuredSteps(rep, opt.backend) : [];
  const fmtName = `${layers}/${embed}`;
  const m = opt.backend ? measuredSpeedup(meas, fmtName, B) : null;
  const seg = (v: StepFormat, set: (f: StepFormat) => void, aria: string) => (
    <span className="seg" role="group" aria-label={aria}>
      {STEP_FORMATS.map((f) => <button key={f} aria-pressed={v === f} onClick={() => set(f)}>{STEP_FORMAT_LABEL[f]}</button>)}
    </span>
  );
  const numIn = (label: string, val: number, set: (v: number) => void) => (
    <label className="field">
      {label}
      <input type="number" aria-label={label} min="1" value={val} onChange={(e) => set(Math.max(1, Math.round(+e.target.value || 1)))} />
    </label>
  );
  return (
    <Bench
      title="Where the step's bytes go"
      task="Quantise only the layers, then only the head. Then raise the batch to 8."
      foot={
        <>
          Every piece of lab 22.2's step at its roofline bound, max(bytes / bandwidth, operations / peak), one after
          another. Solid: weight bytes; light: activations, bias, logits and token ids, all float32. Not modelled: the
          dequantise arithmetic, launches, caches. Embedding rows are read in E's format, whole blocks. Device:{' '}
          {opt.label}, threshold {fmt(thresholdOf(opt.dev))} FLOP/B.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Step shape
          <span className="seg" role="group" aria-label="shape">
            <button aria-pressed={shapeKey === 'm8'} onClick={() => setShapeKey('m8')}>lab 22.2 default</button>
            <button aria-pressed={shapeKey === 'custom'} onClick={() => setShapeKey('custom')}>custom</button>
          </span>
        </label>
        {shapeKey === 'custom' ? (
          <>
            {numIn('d', cshape.d, (v) => setCshape({ ...cshape, d: Math.max(32, Math.round(v / 32) * 32) }))}
            {numIn('layers', cshape.layers, (v) => setCshape({ ...cshape, layers: v }))}
            {numIn('vocabulary', cshape.vocab, (v) => setCshape({ ...cshape, vocab: v }))}
          </>
        ) : (
          <span className="muted">d {M8_DEFAULT.d}, {M8_DEFAULT.layers} layers, vocabulary {M8_DEFAULT.vocab.toLocaleString('en-US')}</span>
        )}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">Layers' weights{seg(layers, setLayers, 'layer format')}</label>
        <label className="field">Embedding and LM head{seg(embed, setEmbed, 'embedding format')}</label>
        <label className="field">
          Batch {B}
          <input type="range" aria-label="batch" min="1" max="8" value={B} onChange={(e) => setB(+e.target.value)} />
        </label>
        <label className="field">
          Device
          <select aria-label="device" value={devKey} onChange={(e) => setDevKey(e.target.value)}>
            {options.map((o) => <option key={o.key} value={o.key}>{o.label}</option>)}
            <option value="custom">Custom numbers</option>
          </select>
        </label>
        {devKey === 'custom' ? (
          <>
            {numIn('peak GFLOP/s', custom.peak, (v) => setCustom({ ...custom, peak: v }))}
            {numIn('bandwidth GB/s', custom.bw, (v) => setCustom({ ...custom, bw: v }))}
          </>
        ) : null}
      </div>
      <StackedBar s={base} max={base.bytes} label="F32" />
      <StackedBar s={q} max={base.bytes} label={`layers ${STEP_FORMAT_LABEL[layers]}, head ${STEP_FORMAT_LABEL[embed]}`} />
      <div className="narrate" role="status" data-part="step" data-speedup={speed} data-wratio={wratio} style={{ margin: '0.7rem 0' }}>
        Weight bytes shrink {wratio.toFixed(2)} times. The model's step: {fmtTime(base.seconds)} in F32, {fmtTime(q.seconds)} here:{' '}
        <b>{speed.toFixed(2)} times faster</b> at batch {B}.{' '}
        {m != null ? <>Lab X2.4 measured {m.toFixed(2)} times on this device.</> : opt.backend ? <>No lab X2.4 measurement of this format at batch {B} on this device yet.</> : null}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" data-part="pieces" style={{ margin: 0, maxWidth: '44rem' }}>
          <thead><tr><th>piece</th><th className="num">bytes, F32</th><th className="num">bytes, here</th><th className="num">model time, here</th><th>bound</th></tr></thead>
          <tbody>
            {q.pieces.map((p, i) => {
              const bw = p.bytes / (opt.dev.bw * 1e9);
              return (
                <tr key={p.key} data-piece={p.key}>
                  <td>{p.label}</td>
                  <td className="num">{base.pieces[i].bytes.toLocaleString('en-US')}</td>
                  <td className="num">{p.bytes.toLocaleString('en-US')}</td>
                  <td className="num">{fmtTime(p.seconds)}</td>
                  <td>{p.seconds > bw * 1.0001 ? 'arithmetic' : 'bytes'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

function StepReveal({ predicted }: { predicted: number }) {
  const { rep, options } = useDevices();
  const book = predictedSpeedup(M8_DEFAULT, ALLQ4, 1, BOOK_DEVICE);
  const mine = options.filter((o) => o.backend);
  const meas = gpuBackends(rep).flatMap((b) => {
    const s = measuredSpeedup(measuredSteps(rep, b), 'q4_0/q4_0', 1);
    return s != null ? [{ b, s, name: String(rep[b].device?.name ?? b) }] : [];
  });
  return (
    <>
      <p data-part="reveal" data-ceiling={book}>
        The bytes model's ceiling at lab 22.2's shape, everything in Q4_0, batch 1: <b>{book.toFixed(2)} times</b> on the
        book's example device{mine.length ? <>, and {mine.map((o) => `${predictedSpeedup(M8_DEFAULT, ALLQ4, 1, o.dev).toFixed(2)} on ${o.label.replace('Yours: ', '')}`).join('; ')}</> : null}.
        It is a ceiling: it counts bytes, not the unpacking.
      </p>
      {meas.length ? (
        meas.map((x) => (
          <p key={x.b}>
            Lab X2.4 on <b>{x.name}</b> ({x.b}) measured <b>{x.s.toFixed(2)} times</b>; you predicted {predicted}.
          </p>
        ))
      ) : (
        <p className="muted">
          No lab X2.4 measurement on this machine yet. Run it from the <a href={href('x2', 'labs')}>labs page</a>; this box
          will then put the measurement beside your prediction.
        </p>
      )}
    </>
  );
}

// ---------------------------------------------------------------- explorer: a toy model, quantised
const TOY_FORMATS: MatFormat[] = ['f32', 'q8_0', 'q4_0', 'affine3', 'affine2'];

function ToyExplorer() {
  const model = useMemo(() => toyModel(), []);
  const [wf, setWf] = useState<MatFormat[]>(['q4_0', 'q4_0']);
  const [ef, setEf] = useState<MatFormat>('q4_0');
  const f = { W: wf, E: ef };
  const run = useMemo(() => greedyRun(model, f), [model, wf, ef]); // eslint-disable-line react-hooks/exhaustive-deps
  const flips = run.steps.filter((s) => s.flipped);
  const Wd = 640;
  const H = 170;
  const maxM = Math.max(...run.steps.map((s) => s.margin), 1);
  const maxE = Math.max(...run.steps.map((s) => s.errRms));
  const top = Math.max(maxM, 2 * maxE) * 1.05;
  const Y = (v: number) => 140 - (v / top) * 125;
  const cw = Wd / TOY.steps;
  const sel = (label: string, v: MatFormat, set: (x: MatFormat) => void) => (
    <label className="field">
      {label}
      <select aria-label={label} value={v} onChange={(e) => set(e.target.value as MatFormat)}>
        {TOY_FORMATS.map((x) => <option key={x} value={x}>{MAT_FORMAT_LABEL[x]}</option>)}
      </select>
    </label>
  );
  return (
    <Bench
      title="A toy model, quantised"
      task="Start with Q8_0 everywhere, then Q4_0, then affine 3-bit. Where do the red bars fall?"
      foot={
        <>
          A seeded random decoder with lab 22.2's structure: d {TOY.d}, vocabulary {TOY.vocab}, {TOY.layers} layers, tied
          embedding, {TOY.steps} greedy steps from token {TOY.start}. Bars: the float32 top-2 margin per step; dots: twice
          the RMS change in the logits. Each step is scored from the float32 run's token (teacher-forced); the token rows
          below run freely. Weights {toyWeightBytes(model, f).toLocaleString('en-US')} bytes here against{' '}
          {toyWeightBytes(model, { W: ['f32', 'f32'], E: 'f32' }).toLocaleString('en-US')} in F32.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        {sel('W0 (layer 0)', wf[0], (x) => setWf([x, wf[1]]))}
        {sel('W1 (layer 1)', wf[1], (x) => setWf([wf[0], x]))}
        {sel('E (embedding and head)', ef, setEf)}
      </div>
      <svg viewBox={`0 0 ${Wd} ${H}`} role="img" aria-label="Top-2 margin and logits error per step" style={{ width: '100%', display: 'block' }}>
        <line x1={0} x2={Wd} y1={140} y2={140} stroke="var(--rule)" />
        {run.steps.map((s) => (
          <g key={s.step} data-step={s.step} data-flipped={s.flipped ? 'yes' : 'no'}>
            <rect x={s.step * cw + 3} y={Y(s.margin)} width={cw - 6} height={140 - Y(s.margin)} fill={s.flipped ? 'var(--fail)' : 'var(--ink-3)'} />
            <circle cx={s.step * cw + cw / 2} cy={Y(2 * s.errRms)} r={3} fill="var(--reg)" />
            {s.step % 4 === 0 ? <text x={s.step * cw + cw / 2} y={156} fontSize="10" textAnchor="middle">{s.step}</text> : null}
          </g>
        ))}
        <text x={4} y={12} fontSize="10.5">margin (bar), 2 × logits RMS change (dot); red: the quantised model picks a different token</text>
      </svg>
      <div className="narrate" role="status" data-part="toy" data-flips={flips.length} data-div={run.firstDivergence ?? ''} style={{ margin: '0.6rem 0' }}>
        Weights' error: E {(100 * run.weightRel.E).toFixed(1)}%, W0 {(100 * run.weightRel.W[0]).toFixed(1)}%, W1{' '}
        {(100 * run.weightRel.W[1]).toFixed(1)}%. Logits' error, mean over steps: {(100 * run.meanRel).toFixed(1)}%.{' '}
        {flips.length} of {TOY.steps} steps pick a different token
        {flips.length ? `, at margins ${flips.map((s) => s.margin.toFixed(1)).join(', ')}` : ''}.{' '}
        {run.firstDivergence == null ? 'The free-running sequences never diverge.' : `The free-running sequences part at position ${run.firstDivergence}.`}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" data-part="tokens" style={{ margin: 0, fontVariantNumeric: 'tabular-nums' }}>
          <tbody>
            {(['F32', 'quantised'] as const).map((who) => {
              const toks = who === 'F32' ? run.f32Tokens : run.qTokens;
              return (
                <tr key={who}>
                  <th style={{ whiteSpace: 'nowrap' }}>{who}</th>
                  {toks.map((t, i) => (
                    <td key={i} className={who === 'quantised' && t !== run.f32Tokens[i] ? 't-fail' : ''} style={{ paddingRight: '0.35rem' }}>{t}</td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the page
export function SX27() {
  const q = questionsSX27;
  const headShare = (M8_DEFAULT.vocab * M8_DEFAULT.d) / (M8_DEFAULT.layers * M8_DEFAULT.d ** 2 + M8_DEFAULT.vocab * M8_DEFAULT.d);
  return (
    <>
      <p>
        Everything so far was one matrix. Milestone M8, lab 22.2's decode step, is several: an embedding gather, a linear
        layer with bias and residual per block, an LM head tied to the embedding table, a softmax and an argmax, all in
        float32 with the batch first. Decode is vector-matrix work, bound by the bytes of the weights. This section puts
        quantised matrices into that step, predicts what the whole step gains, and judges the damage where it matters: in
        the logits and the tokens.
      </p>
      <Beyond title="Ours, not the book's">
        <p>
          That batching turns decode's vector-matrix products into matrix products is the book's (chapter 22, §22.5).
          Lab 22.2's step is the project's, built on it; quantising the step is ours.
        </p>
      </Beyond>
      <Source section="22.5" pages="539–540" />
      <p>
        Where are the step's bytes? At lab 22.2's default shape (d 4096, 2 layers, vocabulary 32,000) the LM head alone is
        131 million of the 165 million weights, {(100 * headShare).toFixed(0)}%. The two layers together are the rest. The
        non-weight bytes (activations, the bias, the logits, the probabilities) are well under a megabyte at batch 1.
      </p>
      <Source section="20.6" pages="504–506" />

      <h2>What changes in milestone M9</h2>
      <p>
        Lab X2.4 makes the swap at the kernel level: every linear reads quantised weights through lab X2.1's int8 kernel
        or lab X2.2's Q4_0 kernel, and because E is shared by the gather and the head, the gather must dequantise one row
        of blocks per sequence. The bias, the residual and the sampler stay float32, as the project's convention says.
        Batches stay at 8 or below: that is where llama.cpp itself uses matrix-vector kernels for quantised weights.
      </p>
      <Today checked={`${CHECKED('D5')}; ch15 D1 for the batch limit`}>
        <p>
          <code>mlx.nn.QuantizedLinear</code> is, in MLX's words, "the quantized equivalent of mlx.nn.Linear", with
          groups of 64 and 4 bits by default. llama.cpp's <code>MMVQ_MAX_BATCH_SIZE</code> is 8.
        </p>
      </Today>
      <Ask q={q['x2/X2.7/q-embed']} />

      <h2>Predict from the bytes of the whole step</h2>
      <p>
        Section X2.1 predicted one GEMV's speed-up as its byte ratio. For a step, use the step's bytes. Quantise only the
        layers and the step barely moves, because the head dominates; quantise everything and the weight bytes fall by
        7.11 times, but the non-weight pieces and the launches do not shrink at all. Try both in the explorer before you
        answer.
      </p>
      <Simplification>
        <p>
          Each piece is timed at its roofline bound with the lab's float32 activations. No dequantise arithmetic, no
          launch cost, no caches, no attention (lab 22.2's step has none). The book's device numbers are its example
          only; your own come from lab 5.1.
        </p>
      </Simplification>
      <StepExplorer />
      <Ask q={q['x2/X2.7/q-step-ratio']} label="Work it out" />
      <Ask q={q['x2/X2.7/q-batch8']} />
      <Predict
        id="x2/X2.7/p-step-speedup"
        prompt={
          <>
            Before you run lab X2.4: how many times faster than the float32 step do you expect the all-Q4_0 step to be at
            batch 1 on <em>your</em> device?
          </>
        }
        unit="× faster"
      >
        {(v) => <StepReveal predicted={v} />}
      </Predict>

      <h2>The error that matters</h2>
      <p>
        Section X2.2 measured error on the weights. A model is judged by its outputs. So measure two things against the
        float32 step: the relative RMS error of the logits, and whether greedy decoding picks the same token. The toy
        model below has lab X2.4's emulator structure at a size the browser can run. Start each matrix at Q8_0, then lower
        one at a time and watch which steps flip.
      </p>
      <Simplification>
        <p>
          The toy model is random, not trained. Its tokens agreeing or not says nothing about a real model's perplexity or
          accuracy; it shows the mechanism: a flip needs a small margin.
        </p>
      </Simplification>
      <ToyExplorer />
      <Ask q={q['x2/X2.7/q-margin']} />
      <Ask q={q['x2/X2.7/q-weights-vs-output']} />
      <p>
        For real models, the projects say it themselves. PyTorch's team reports no perceptible accuracy loss from int8
        weight-only quantisation and some from int4, for which one scale per 32 weights is a common choice. Those are
        their statements about their models, not something this page measured.
      </p>
      <Today checked={CHECKED('A1')}>
        <p>
          PyTorch, "Accelerating Generative AI with PyTorch II: GPT, Fast": "using int8 weight-only quantization has no
          perceptible accuracy degradation, using int4 weight-only quantization does"; for int4, "one scaling factor per 32
          elements ... is also a common choice among the community".
        </p>
      </Today>
      <p>M9 quantises the weights and nothing else.</p>
      <Beyond title="The KV cache">
        <p>
          M9 leaves the KV cache in float32. llama.cpp can store it as <code>q8_0</code> or <code>q4_0</code> too (34 and 18
          bytes per 32 values; notes/today/ch20-facts.md D3), which attacks the attention traffic that batching does not
          help.
        </p>
      </Beyond>

      <h2>Hand-off to lab X2.4</h2>
      <p>
        Lab X2.4 builds this step on your GPU and reports time per step for each format and batch. The labs page puts the
        measurement beside the prediction you committed here, and turns it into bandwidth utilisation: the weight bytes
        per step divided by the time per step, as a fraction of your peak bandwidth from lab 5.1. With made-up numbers:
        100 MB of weights in 1 ms against a measured 200 GB/s is {(100 * bandwidthUtilisation(100e6, 1e-3, 200)).toFixed(0)}%. Go to the{' '}
        <a href={href('x2', 'labs')}>labs page</a> when you are ready.
      </p>
    </>
  );
}
