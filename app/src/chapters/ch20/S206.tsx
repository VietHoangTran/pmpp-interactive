import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { batchToReach, batchedDecodeAttentionAI, linearAI } from '../../sim/kvmath';
import { FIG_PROMPT, FIG_STEPS, expectedPerPass, figCounters, replayFigure, simulate } from '../../sim/specdecode';
import { BatchShareScene } from './BatchShareScene';
import { KvExplorer } from './KvExplorer';

// ---------------------------------------------------------------- ridges from lab 5.1

interface Ridge {
  key: string;
  label: string;
  ridge: number;
  bw?: number; // GB/s
}
const BOOK_RIDGE: Ridge = { key: 'book', label: 'the book’s example', ridge: 66900 / 3350, bw: 3350 };
function ridges(reports: Reports): Ridge[] {
  return realBackends(reports).flatMap((b) => {
    const r = benchRows(reports[b], 'ch05/lab1_roofline').find((x) => x.name === 'roofline');
    const peak = num(r?.compute_gflops);
    const bw = num(r?.bandwidth_gb_per_s);
    return peak && bw ? [{ key: b, label: `yours, ${String(reports[b].device.name ?? b)} (${b})`, ridge: peak / bw, bw }] : [];
  });
}

// ---------------------------------------------------------------- batching explorer

const LAYERS = [
  { key: 'book', label: '1,024 → 4,096 (the book’s)', inF: 1024, outF: 4096 },
  { key: 'qkv', label: '4,096 → 12,288 (fused Q, K, V)', inF: 4096, outF: 12288 },
  { key: 'square', label: '4,096 → 4,096', inF: 4096, outF: 4096 },
];
const N_ATTN = 4096;

function BatchPlot({ inF, outF, p, b, lines }: { inF: number; outF: number; p: number; b: number; lines: Ridge[] }) {
  const W = 600;
  const H = 300;
  const pad = { l: 52, r: 14, t: 14, b: 38 };
  const yMin = 0.1;
  const yMax = 2000;
  const X = (bb: number) => pad.l + (Math.log2(bb) / 9) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * (H - pad.t - pad.b);
  const bs = Array.from({ length: 91 }, (_, i) => 2 ** (i / 10));
  const lin = bs.map((x) => `${X(x).toFixed(1)},${Y(linearAI(x, inF, outF, p)).toFixed(1)}`).join(' ');
  const att = batchedDecodeAttentionAI(1, N_ATTN, p);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Intensity against batch size" style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {[1, 2, 4, 8, 16, 32, 64, 128, 256, 512].map((x) => (
        <g key={x}>
          <line x1={X(x)} x2={X(x)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(x)} y={H - pad.b + 14} fontSize="10.5" textAnchor="middle">{x}</text>
        </g>
      ))}
      {[0.1, 1, 10, 100, 1000].map((v) => (
        <g key={v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 6} y={Y(v) + 4} fontSize="10.5" textAnchor="end">{v}</text>
        </g>
      ))}
      <text x={(W + pad.l) / 2} y={H - 5} fontSize="11" textAnchor="middle">batch size</text>
      <text x={12} y={(H - pad.b + pad.t) / 2} fontSize="11" textAnchor="middle" transform={`rotate(-90 12 ${(H - pad.b + pad.t) / 2})`}>FLOP per byte</text>
      {lines.map((r, i) => (
        <g key={r.key} data-ridge={r.ridge.toFixed(3)}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(r.ridge)} y2={Y(r.ridge)} stroke="var(--ink)" strokeDasharray={r.key === 'book' ? '2 4' : '6 3'} />
          <text x={W - pad.r - 4} y={Y(r.ridge) - 4 - i * 12} fontSize="10" textAnchor="end">ridge, {r.label}: {fmt(r.ridge)}</text>
        </g>
      ))}
      <polyline points={lin} fill="none" stroke="var(--global)" strokeWidth="2.4" />
      <line x1={pad.l} x2={W - pad.r} y1={Y(att)} y2={Y(att)} stroke="var(--reg)" strokeWidth="2.4" />
      <text x={pad.l + 6} y={Y(att) + 15} fontSize="10.5" style={{ fill: 'var(--reg)' }}>decode attention, each request its own cache</text>
      <g data-linear-key="">
        <line x1={pad.l + 8} x2={pad.l + 30} y1={pad.t + 14} y2={pad.t + 14} stroke="var(--global)" strokeWidth="2.4" />
        <text x={pad.l + 36} y={pad.t + 18} fontSize="10.5" style={{ fill: 'var(--global)' }}>linear layer, weights shared</text>
      </g>
      <circle cx={X(b)} cy={Y(linearAI(b, inF, outF, p))} r="5" fill="var(--global)" />
      <circle cx={X(b)} cy={Y(att)} r="5" fill="var(--reg)" />
    </svg>
  );
}

function BatchingBench() {
  const { reports } = useReports();
  const lines = useMemo(() => [...ridges(reports), BOOK_RIDGE], [reports]);
  const [logB, setLogB] = useState(0);
  const [layerKey, setLayerKey] = useState('book');
  const [p, setP] = useState(2);
  const b = 2 ** logB;
  const L = LAYERS.find((x) => x.key === layerKey)!;
  const lin = linearAI(b, L.inF, L.outF, p);
  const att = batchedDecodeAttentionAI(b, N_ATTN, p);
  return (
    <Bench
      title="Batching raises one intensity, not the other"
      task="Find the batch at which the linear layer crosses your ridge. Then try to make attention cross it."
      foot={
        <>
          The linear layer counts its weights once and every input and output row (that reproduces the book’s 1 and 315).
          Attention is one decode step over a {N_ATTN.toLocaleString('en-US')}-token cache per request. Ridges are peak
          over bandwidth from lab 5.1, or the book’s example (66.9 TFLOP/s over 3.35 TB/s, ch. 5 p. 94).
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Batch b = {b}
          <input type="range" min="0" max="9" step="1" value={logB} aria-label="Batch size" onChange={(e) => setLogB(Number(e.target.value))} />
        </label>
        <label className="field" style={{ maxWidth: '100%', minWidth: 0 }}>
          Linear layer
          <select value={layerKey} onChange={(e) => setLayerKey(e.target.value)} aria-label="Linear layer" style={{ maxWidth: '100%' }}>
            {LAYERS.map((x) => <option key={x.key} value={x.key}>{x.label}</option>)}
          </select>
        </label>
        <label className="field">
          Element
          <select value={p} onChange={(e) => setP(Number(e.target.value))} aria-label="Element size" style={{ maxWidth: '100%' }}>
            <option value={4}>fp32, 4 bytes</option>
            <option value={2}>16-bit, 2 bytes</option>
            <option value={1}>8-bit, 1 byte</option>
          </select>
        </label>
      </div>
      <div className="cols plot">
        <BatchPlot inF={L.inF} outF={L.outF} p={p} b={b} lines={lines} />
        <div>
          <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-batching="">
            <tbody>
              <tr><td>Linear layer at b = {b}</td><td className="num" data-linear={lin.toFixed(4)}><b>{fmt(lin, 4)}</b> FLOP/B</td></tr>
              <tr><td>Decode attention at b = {b}</td><td className="num" data-attn={att.toFixed(4)}><b>{fmt(att, 4)}</b> FLOP/B</td></tr>
            </tbody>
          </table>
          <table className="data" style={{ background: 'var(--page)', margin: '0.8rem 0 0' }}>
            <thead><tr><th>Ridge</th><th className="num">linear crosses at</th><th className="num">attention</th></tr></thead>
            <tbody>
              {lines.map((r) => {
                const at = batchToReach(r.ridge, L.inF, L.outF, p);
                return (
                  <tr key={r.key} data-cross={at ?? 'never'}>
                    <td>{r.label}, {fmt(r.ridge)}</td>
                    <td className="num">{at == null ? 'never' : `b = ${at}`}</td>
                    <td className="num">{att >= r.ridge ? 'already' : 'never'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- speculative decoding

const ROUNDS = replayFigure();

function Tok({ t, tone }: { t: string; tone?: 'prefix' | 'draft' | 'ok' | 'bad' | 'mute' }) {
  const style: CSSProperties = {
    display: 'inline-block',
    padding: '0.05rem 0.4rem',
    margin: '0.1rem 0.2rem 0.1rem 0',
    borderRadius: 3,
    fontFamily: 'var(--sans)',
    fontSize: '0.8rem',
    border: '1px solid var(--rule)',
    background: tone === 'prefix' ? 'var(--bench)' : tone === 'ok' ? 'var(--ok-wash)' : tone === 'bad' ? 'var(--fail-wash)' : 'var(--cell)',
    color: tone === 'ok' ? 'var(--ok)' : tone === 'bad' ? 'var(--fail)' : tone === 'mute' ? 'var(--ink-3)' : 'var(--ink)',
    borderColor: tone === 'ok' ? 'var(--ok)' : tone === 'bad' ? 'var(--fail)' : 'var(--rule)',
  };
  return <span style={style}>{t}</span>;
}

function FigureReplay() {
  const [s, setS] = useState(0);
  const c = figCounters(s);
  const shown = (round: number, phase: 'draft' | 'verify' | 'accept') => FIG_STEPS.findIndex((x) => x.round === round && x.phase === phase) <= s;
  const prefixText = (prefix: string[]) => (prefix.length > FIG_PROMPT.length ? '… ' + prefix.slice(FIG_PROMPT.length).join(' ') : prefix.slice(-2).join(' '));
  return (
    <div data-replay-step={s}>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <button onClick={() => setS(0)} disabled={s === 0}>Restart</button>
        <button onClick={() => setS(s - 1)} disabled={s === 0}>Back</button>
        <button className="primary" onClick={() => setS(s + 1)} disabled={s >= FIG_STEPS.length - 1}>Step</button>
        <span className="muted" aria-live="polite">{s === 0 ? 'the prompt' : `round ${FIG_STEPS[s].round + 1}: ${FIG_STEPS[s].phase}`}</span>
      </div>
      <p style={{ margin: '0 0 0.6rem' }}>{FIG_PROMPT.map((t, i) => <Tok key={i} t={t} tone="prefix" />)}</p>
      {ROUNDS.map((r, ri) =>
        shown(ri, 'draft') ? (
          <div key={ri} className="panel" style={{ marginBottom: '0.6rem' }} data-round={ri + 1}>
            <div className="mx-label">Round {ri + 1}</div>
            <div><span className="muted">Predict (draft model): </span>{r.draft.map((t, i) => <Tok key={i} t={t} tone="draft" />)}</div>
            {shown(ri, 'verify') ? (
              <div style={{ marginTop: '0.3rem' }}>
                <span className="muted">Verify (target model, one pass, {r.draft.length} rows):</span>
                {r.draft.map((t, i) => (
                  <div key={i} style={{ marginLeft: '1rem' }} data-verify={r.ok[i] ? 'ok' : 'bad'}>
                    <span className="muted">after “{prefixText([...r.prefix, ...r.draft.slice(0, i)])}” the target says </span>
                    <Tok t={r.target[i]} tone={r.ok[i] ? 'ok' : 'bad'} />
                    <span className={r.ok[i] ? 't-ok' : 't-fail'}>{r.ok[i] ? '✓ matches' : `✗ the draft said “${t}”`}</span>
                  </div>
                ))}
              </div>
            ) : null}
            {shown(ri, 'accept') ? (
              <div style={{ marginTop: '0.3rem' }} data-accepted={r.accepted.join(' ')}>
                <span className="muted">Accept: </span>
                {r.accepted.map((t, i) => <Tok key={i} t={t} tone="ok" />)}
              </div>
            ) : null}
          </div>
        ) : null,
      )}
      <p className="muted" style={{ margin: 0 }} data-counters={`${c.targetPasses}/${c.tokens.length}`}>
        Target passes so far: <b>{c.targetPasses}</b>. Tokens kept: <b>{c.tokens.length}</b>
        {c.tokens.length ? <> ({c.tokens.join(' ')})</> : null}. Plain decoding would have needed {c.tokens.length} target{' '}
        {c.tokens.length === 1 ? 'pass' : 'passes'} for them.
      </p>
    </div>
  );
}

function OwnSpec() {
  const [k, setK] = useState(4);
  const [alpha, setAlpha] = useState(0.7);
  const [seed, setSeed] = useState(1);
  const [keep, setKeep] = useState(false);
  const r = useMemo(() => simulate({ k, alpha, seed, tokens: 64, keepTarget: keep }), [k, alpha, seed, keep]);
  const e = expectedPerPass(k, alpha, keep);
  return (
    <div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Speculation depth k = {k}
          <input type="range" min="1" max="8" value={k} aria-label="Speculation depth" onChange={(ev) => setK(Number(ev.target.value))} />
        </label>
        <label className="field">
          Acceptance per token α = {alpha.toFixed(2)}
          <input type="range" min="0.05" max="1" step="0.05" value={alpha} aria-label="Acceptance probability" onChange={(ev) => setAlpha(Number(ev.target.value))} />
        </label>
        <button onClick={() => setSeed(seed + 1)}>Another run</button>
        <label className="check">
          <input type="checkbox" checked={keep} onChange={(ev) => setKeep(ev.target.checked)} /> also keep the target’s token at the first mismatch (ours)
        </label>
      </div>
      <div className="cols plot">
        <div aria-label="Rounds" data-rounds={r.rounds.length}>
          {r.rounds.slice(0, 18).map((rd, i) => (
            <div key={i} style={{ display: 'flex', gap: 3, alignItems: 'center', marginBottom: 3 }}>
              <span className="muted" style={{ fontSize: '0.72rem', width: '3.2rem' }}>pass {i + 1}</span>
              {Array.from({ length: k }, (_, j) => {
                const ok = j < rd.matched;
                const bad = j === rd.matched;
                return (
                  <span key={j} style={{
                    width: 18, height: 14, borderRadius: 2,
                    background: ok ? 'var(--ok)' : bad ? 'var(--fail-wash)' : 'transparent',
                    border: `1px ${ok || bad ? 'solid' : 'dashed'} ${ok ? 'var(--ok)' : bad ? 'var(--fail)' : 'var(--rule)'}`,
                  }} />
                );
              })}
              <span style={{ fontSize: '0.75rem', marginLeft: 6 }}>{rd.kept} kept</span>
            </div>
          ))}
          {r.rounds.length > 18 ? <p className="muted" style={{ margin: 0 }}>… {r.rounds.length - 18} more passes</p> : null}
        </div>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-spec="">
          <tbody>
            <tr><td>Target passes for {r.tokens} tokens</td><td className="num" data-passes={r.targetPasses}>{r.targetPasses}</td></tr>
            <tr><td>Tokens per target pass, this run</td><td className="num" data-per-pass={r.tokensPerPass.toFixed(3)}>{fmt(r.tokensPerPass)}</td></tr>
            <tr><td>Expected (closed form)</td><td className="num" data-expected={e.toFixed(4)}>{fmt(e)}</td></tr>
            <tr><td>Reads of the whole cache per token kept</td><td className="num">{fmt(r.cacheReadsPerToken)}</td></tr>
            <tr><td>Plain decoding</td><td className="num">1</td></tr>
            <tr><td>Query rows sharing each read</td><td className="num">{r.rowsPerRead}</td></tr>
          </tbody>
        </table>
      </div>
      <p className="muted" style={{ marginBottom: 0 }}>
        Expected tokens per target pass at α = {alpha.toFixed(2)} for depths 1 to 8:{' '}
        {Array.from({ length: 8 }, (_, i) => fmt(expectedPerPass(i + 1, alpha, keep))).join(', ')}.
      </p>
    </div>
  );
}

function SpecBench() {
  const [mode, setMode] = useState<'book' | 'own'>('book');
  return (
    <Bench
      title="Speculative decoding"
      task={mode === 'book' ? 'Step through the book’s figure: draft, verify, accept, twice.' : 'Find the acceptance below which deeper speculation stops paying.'}
      foot={
        mode === 'book'
          ? 'The rounds of Fig. 20.18, replayed from src/sim/specdecode.ts. Each verify is one pass of the target model over all the drafted rows.'
          : 'Green: a draft token the target agreed with. Red: the first mismatch. Dashed: drafted but never reached. Model: src/sim/specdecode.ts.'
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Mode">
          <button aria-pressed={mode === 'book'} onClick={() => setMode('book')}>The book’s figure</button>
          <button aria-pressed={mode === 'own'} onClick={() => setMode('own')}>Your own</button>
        </span>
      </div>
      {mode === 'book' ? <FigureReplay /> : <OwnSpec />}
    </Bench>
  );
}

// ---------------------------------------------------------------- lab 20.4 prediction

function DecodeMeasured({ predicted }: { predicted: number }) {
  const { reports } = useReports();
  const found = realBackends(reports).flatMap((b) => {
    const roof = benchRows(reports[b], 'ch05/lab1_roofline').find((x) => x.name === 'roofline');
    const bw = num(roof?.bandwidth_gb_per_s);
    const rows = benchRows(reports[b], 'ch20/lab4_decode').filter((x) => x.name === 'decode' && num(x.gb_per_s));
    return rows.length ? [{ b, device: String(reports[b].device.name ?? b), bw, rows }] : [];
  });
  if (!found.length)
    return (
      <p>
        No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 20.4 --solution --bench</code> and come back:
        this box will compare your prediction with lab 20.4’s GB/s and lab 5.1’s.
      </p>
    );
  return (
    <>
      {found.map((f) => {
        const at = f.rows.find((r) => r.batch === 1 && r.len === 8192) ?? f.rows[f.rows.length - 1];
        const pct = f.bw ? (100 * num(at.gb_per_s)!) / f.bw : null;
        return (
          <div key={f.b}>
            <p>
              On <b>{f.device}</b> ({f.b}) the reference solution read its fp32 cache at <b>{fmt(num(at.gb_per_s)!)} GB/s</b>{' '}
              (batch {String(at.batch)}, {Number(at.len).toLocaleString('en-US')} tokens)
              {pct != null ? <>: <b>{fmt(pct)}%</b> of the {fmt(f.bw!)} GB/s lab 5.1 measured. You predicted {predicted}%.</> : '.'}
            </p>
            <table className="data" style={{ margin: '0 0 0.6rem' }}>
              <thead><tr><th className="num">batch</th><th className="num">tokens</th><th className="num">GB/s</th>{f.bw ? <th className="num">of lab 5.1</th> : null}</tr></thead>
              <tbody>
                {f.rows.map((r, i) => (
                  <tr key={i}>
                    <td className="num">{String(r.batch)}</td>
                    <td className="num">{Number(r.len).toLocaleString('en-US')}</td>
                    <td className="num">{fmt(num(r.gb_per_s)!)}</td>
                    {f.bw ? <td className="num">{fmt((100 * num(r.gb_per_s)!) / f.bw)}%</td> : null}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
      })}
      <p className="muted">
        Measurements from your machine, not expectations. The Flash-Decoding authors note that one query per sequence can
        leave most of a GPU idle when the batch is small, which is why production kernels also split the cache across
        blocks (ch20-facts A8); compare your batch 1 and batch 8 rows.
      </p>
    </>
  );
}

// ---------------------------------------------------------------- questions

export const questionsS206: Record<string, Question> = {
  'ch20/20.6/q-linear-ai': {
    kind: 'numeric',
    id: 'ch20/20.6/q-linear-ai',
    prompt: <>A linear layer with 1,024 inputs and 4,096 outputs runs in 16-bit on a batch of 512 rows. Counting the weights once plus every input and output row, how many FLOPs does it do per byte?</>,
    answer: 315,
    tol: 0.0035,
    unit: 'FLOP/B',
    explain: (
      <p>
        FLOPs: 512 × 2 × 1,024 × 4,096 = 4,294,967,296. Bytes: the weights, 1,024 × 4,096 × 2 = 8,388,608, plus 512 × (1,024
        + 4,096) × 2 = 5,242,880 for the rows in and out. The ratio is 315.08, the book’s 315. At batch 1 the same count
        gives 0.9988, the book’s “about 1”.
      </p>
    ),
    source: 'ch. 20 §20.6, p. 505; notes/today/ch20-facts.md F1',
  },
  'ch20/20.6/q-attn-batch': {
    kind: 'choice',
    id: 'ch20/20.6/q-attn-batch',
    prompt: <>Why does batching 64 decode requests together not raise the intensity of their attention?</>,
    options: [
      { label: 'Each request reads its own cache, so bytes grow with the FLOPs.', correct: true },
      { label: 'The softmax cannot be batched across requests of different lengths.', why: 'The softmax is a small part of the work and batches fine; the issue is what is read.' },
      { label: 'The causal mask differs from request to request in the batch.', why: 'A decode step needs no mask at all.' },
      { label: 'It does raise it, 64 times, just as it does for the projections.', why: 'That is what happens to the projections, whose weights are shared. The caches are not.' },
    ],
    explain: <p>Intensity rises when one loaded byte feeds more arithmetic. Weights are common to every request; a cache belongs to one request, so 64 requests read 64 caches and no byte is shared between them.</p>,
    source: 'ch. 20 §20.6, p. 505, Fig. 20.17',
  },
  'ch20/20.6/q-gpt3': {
    kind: 'numeric',
    id: 'ch20/20.6/q-gpt3',
    prompt: <>GPT-3 has 96 layers and 96 heads of 128. In 16-bit, how big is the KV cache of one sequence at its real context window of 2,048 tokens?</>,
    answer: 9,
    tol: 0.02,
    unit: 'GiB',
    explain: (
      <p>
        Eq. 20.8: 1 × 2,048 × 2 × 96 × (96 × 128) × 2 bytes = 9,663,676,416 bytes = 9 GiB. The book’s 18 GB uses 4,096
        tokens, twice the window GPT-3 was built with.
      </p>
    ),
    source: 'ch. 20 §20.6, p. 506; notes/today/ch20-facts.md E2, F1',
  },
  'ch20/20.6/q-batch-fit': {
    kind: 'numeric',
    id: 'ch20/20.6/q-batch-fit',
    prompt: <>After the weights, 8 GiB of memory remain for the cache. How many 4,096-token Llama 2 7B sequences fit in 16-bit?</>,
    answer: 4,
    explain: <p>Each sequence’s cache is 2 GiB (the book’s example: 4,096 × 2 × 32 × 4,096 × 2 bytes), and 8 / 2 = 4. Halve the context and twice as many fit: batch times context is what memory caps.</p>,
    source: 'ch. 20 §20.6, pp. 505–506',
  },
  'ch20/20.6/q-palm': {
    kind: 'choice',
    id: 'ch20/20.6/q-palm',
    prompt: <>The book’s “PaLM 2” row (64 layers, hidden dimension 8,192) gives a cache of 8 GB for 4,096 tokens. What is wrong with it?</>,
    options: [
      { label: 'The shape is PaLM 62B’s, which caches one K/V head, not 32.', correct: true },
      { label: 'Nothing: the arithmetic of Eq. 20.8 is right for that shape.', why: 'The arithmetic is right; the model it describes does not exist in that form.' },
      { label: 'It should be 16 GB, because K and V are both cached per layer.', why: 'Eq. 20.8 already has the factor 2 for K and V.' },
      { label: 'The precision should be 4 bytes, as the model was trained in fp32.', why: 'The book states 16-bit, which is what such models are served in.' },
    ],
    explain: (
      <p>
        Google withheld PaLM 2’s architecture, so these are not its numbers. The first PaLM paper lists PaLM 62B with 64 layers, 32 heads and d_model
        8,192, and uses multi-query attention with a head size of 256. With one K and one V head of 256 per layer, 4,096
        tokens take 4,096 × 2 × 64 × 256 × 2 bytes = 256 MiB, 32 times less than the book&rsquo;s figure.
      </p>
    ),
    source: 'notes/today/ch20-facts.md E2, F1; book p. 506',
  },
  'ch20/20.6/q-spec-ai': {
    kind: 'choice',
    id: 'ch20/20.6/q-spec-ai',
    prompt: <>Speculative decoding verifies k drafted tokens in one pass of the target model. What happens to the cache bytes read per verified token?</>,
    options: [
      { label: 'They fall about k times: one read of the prefix serves all k rows.', correct: true },
      { label: 'Nothing: each drafted token still reads the cache on its own.', why: 'They need not: the k rows share one prefix, so one pass can load each cached K and V once for all k (p. 508). Reading per token is what batching separate requests forces.' },
      { label: 'They double, because two models run and each reads a cache.', why: 'The draft model is small and has its own small cache; the target’s cache is read once per pass.' },
      { label: 'They fall to zero, because the drafts were already computed.', why: 'The cache is still read, once per pass.' },
    ],
    explain: <p>The book’s point on p. 508: a speculative batch is special because all its entries share a common prefix, so the same cached K and V serve every output token instead of being loaded once per token. Attention&rsquo;s intensity rises about k times.</p>,
    source: 'ch. 20 §20.6, p. 508; notes/today/ch20-facts.md E4',
  },
  'ch20/20.6/q-inflight': {
    kind: 'choice',
    id: 'ch20/20.6/q-inflight',
    prompt: <>The book says in-flight batching slots a new request into the batch with the lowest workload (p. 506). What do the paper it cites and NVIDIA’s documentation actually describe?</>,
    options: [
      { label: 'After every iteration, finished requests leave and waiting ones join.', correct: true },
      { label: 'The book’s description: a new request joins whichever batch has the least work.', why: 'Neither source describes choosing a batch by workload.' },
      { label: 'Padding every request to one length, so that the batch stays rectangular.', why: 'That is what iteration-level scheduling avoids.' },
      { label: 'Running prefill and decode on separate GPUs, each with its own batch.', why: 'TensorRT-LLM says the opposite: context-phase and generation-phase sequences are processed together.' },
    ],
    explain: (
      <p>
        This is iteration-level scheduling. The vLLM paper credits it to Orca: “After each iteration, completed requests are removed from
        the batch, and new ones are added.” TensorRT-LLM: “in-flight batching of requests (also known as continuous batching
        or iteration-level batching)”.
      </p>
    ),
    source: 'notes/today/ch20-facts.md E6',
  },
};

// ---------------------------------------------------------------- the page

export function S206() {
  return (
    <>
      <p>
        Section 20.4 left a decode step as a string of vector-matrix products doing about one operation per byte. This
        section asks two questions about that. What can raise the ratio? And how much memory does the cache cost, once
        many requests are served at the same time?
      </p>
      <Source section="20.6" pages="504" />

      <h2>Batching the projections</h2>
      <p>
        Batching is how every neural network raises its intensity: stack several input rows into a matrix, so that each
        weight loaded from memory serves all of them.
      </p>
      <Source section="20.6" pages="504–505" />
      <p>
        The book&rsquo;s example is a linear layer with 1,024 inputs and 4,096 outputs in 16-bit: about 1 FLOP per byte
        with one row, 315 with 512 rows. Its footnote adds that this is the theoretical gain; what a kernel gets depends
        on the tile size it uses.
      </p>
      <Aside>
        <p>The book does not show its count. Weights once, plus every input and output row, gives 0.9988 and 315.08 exactly (facts F1). Chapter 22 builds the same count into a whole decode step.</p>
      </Aside>
      <BatchingBench />
      <Ask q={questionsS206['ch20/20.6/q-linear-ai']} label="Work it out" />

      <h2>Why attention does not batch</h2>
      <p>
        Fig. 20.17 draws a batch of four decode steps. In the projections, all four queries share the same W_Q, W_K and
        W_V, so batching raises the projections’ intensity by the batch size. In attention, each query needs the cache of
        its own conversation: four queries read four different caches, and the bytes grow as fast as the arithmetic.
        Batching is then limited by memory: the weights plus one cache per request.
      </p>
      <Source section="20.6" pages="505" figs="Fig. 20.17" />
      <p>Watch both streams as the batch grows: which one&rsquo;s intensity moves, and which one fills memory.</p>
      <Simplification>
        <p>
          One layer of a Llama 2 7B shape, 16-bit, every request with the same 1,024-token cache. Only the weight and cache
          reads are counted: the Q, K and V projection and attention, no output projection, no feed-forward.
        </p>
      </Simplification>
      <BatchShareScene />
      <Ask q={questionsS206['ch20/20.6/q-attn-batch']} />

      <h2>How big the cache is</h2>
      <p>
        Weights first: a model of several billion parameters is typical, and 7 billion in a 16-bit format take about 14 GB.
        Then the cache. With l layers, h_q query heads of width d and p bytes per element, every token adds a K row and a
        V row to every head of every layer:
      </p>
      <Source section="20.6" pages="505–506" />
      <p style={{ textAlign: 'center' }}>
        per token (Eq. 20.7): 2 × l × h_q × d × p &nbsp;&nbsp;·&nbsp;&nbsp; b sequences of N tokens (Eq. 20.8): b × N × 2 × l × h_q × d × p
      </p>
      <p>
        The book works three examples, each one 16-bit sequence of 4,096 tokens: Llama 2 7B needs 2 GB, GPT-3 18 GB and
        “PaLM 2” 8 GB. The arithmetic is right in all three. Two of the inputs are not.
      </p>
      <Today checked="2026-11-20 (ch20-facts E2, F1)">
        <p>Llama 2 7B: 2 GiB, confirmed. GPT-3’s context window is 2,048 tokens, so 9 GiB, not 18. PaLM 2’s shape was withheld; the row is PaLM 62B, which uses multi-query attention with one K/V head of 256: 256 MiB, not 8 GB. The book’s GB are binary (GiB).</p>
      </Today>
      <p>
        The book’s conclusion survives the corrections: batch many requests and the caches quickly outgrow one GPU’s
        memory, which is why LLM serving commonly spans several GPUs. Size it for yourself, against the memory your own
        device reported.
      </p>
      <p>
        The last row is ours, not the book&rsquo;s: a floor on the time of one decode step, if every weight and every
        cached byte had to be read once at the bandwidth lab 5.1 measured. It answers &ldquo;how fast could decode
        possibly go here?&rdquo; with one division.
      </p>
      <Simplification>
        <p>The floor counts reads only, assumes every byte streams once at lab 5.1’s bandwidth, and ignores caches, overlap, kernel launches and everything outside attention and the weights.</p>
      </Simplification>
      <KvExplorer mode="size" />
      <Ask q={questionsS206['ch20/20.6/q-gpt3']} label="Work it out" />
      <Ask q={questionsS206['ch20/20.6/q-batch-fit']} label="Work it out" />
      <Ask q={questionsS206['ch20/20.6/q-palm']} />

      <h2>Requests of different lengths</h2>
      <p>
        The analysis so far gave every request the same length. Real requests differ in both input and output length, and
        the book notes that this can unbalance the work and leave the GPU underused. It names the remedy serving systems
        use, in-flight batching, and describes it as slotting a new request into the batch with the lowest workload as soon
        as an earlier one is served.
      </p>
      <Source section="20.6" pages="506" />
      <p>
        The source the book cites, and NVIDIA’s documentation, describe something simpler. After every iteration, finished
        requests leave the running batch and waiting ones join it, so the batch is re-formed at every decode step rather
        than once per batch of requests. The same paper’s complaint about memory, that cache wasted by fragmentation
        limits the batch size, is what section 20.7’s paged cache answers.
      </p>
      <Today checked="2026-11-20 (ch20-facts E6)">
        <p>The vLLM paper credits this iteration-level scheduling to Orca. TensorRT-LLM calls it “in-flight batching … (also known as continuous batching or iteration-level batching)”. Neither slots requests by lowest workload.</p>
      </Today>
      <Ask q={questionsS206['ch20/20.6/q-inflight']} />
      <p>
        The chapter argues only in terms of intensity and memory. It says nothing about the latency each user pays for the
        throughput a bigger batch buys. <a href={href('ch22', '22.5')}>Section 22.5</a> and milestone M8 take that up on your
        own device.
      </p>

      <h2>The intensity of decode attention</h2>
      <p>
        Eq. 20.9 counts one layer of decode attention with the cache: it reads 2 × h_q × d × N cached elements, plus Q′ and
        O′, and does 2 × N × h_q × d operations for (Q′Kᵀ)V. That is N / (1 + N), about one operation per element, and
        batching cannot change it.
      </p>
      <Source section="20.6" pages="506–507" figs="Eq. 20.9" />
      <p>
        The book counts a multiply-add as one operation and an element as one byte (p = 1). In FLOP per byte, that is 1 in
        16-bit and 0.5 in the fp32 the labs use.
      </p>
      <Today checked="2026-11-20 (ch20-facts E4)">
        <p>The book’s source, [13], states it per byte: “Every loaded BF16 (2 bytes) element of the cached key matrix performs one MAC (2 FLOPs) … yielding a 1:1 FLOP-to-byte ratio.”</p>
      </Today>
      <p>
        So a decode kernel’s speed is set by how fast it can read the cache. Lab 20.4 asks you to write one, and its
        benchmark reports the GB/s it reached. Commit to a number first.
      </p>
      <Predict
        id="ch20/20.6/p-decode-bw"
        prompt={
          <>
            Lab 20.4’s decode kernel reads an fp32 cache of 32 heads of 128 for one sequence of 8,192 tokens. What
            percentage of the bandwidth lab 5.1 measured on your device do you expect it to reach?
          </>
        }
        unit="%"
      >
        {(v) => <DecodeMeasured predicted={v} />}
      </Predict>

      <h2>Speculative decoding</h2>
      <p>
        Speculative decoding attacks the same ratio from another side. A small draft model guesses the next k tokens of a
        large target model (k is the speculation depth, 3 in the book’s figure). The target, much slower per token, checks
        all k guesses in one pass, executed in parallel like a batch. Tokens up to the first disagreement are accepted, and
        the next round starts after them.
      </p>
      <Source section="20.6" pages="507–508" figs="Fig. 20.18" />
      <p>
        The book gives two reasons it helps. Like a batch, it gives every projection weight several rows to serve. And it
        is a special batch: all its rows share one prefix, so the cached K and V of that prefix are loaded once and used by
        every row. That raises attention’s intensity, which batching separate requests cannot do.
      </p>
      <Today checked="2026-11-20 (ch20-facts E4)">
        <p>[13], Fig. 3: with a query length of 2, as in speculative decoding, h_q = 128 already “climbs beyond the roof” on their example device. Their H100 figures are theirs, not yours.</p>
      </Today>
      <p>
        Step through the figure, then switch to your own runs and look for where deeper speculation stops paying.
      </p>
      <Simplification>
        <p>Each draft token is accepted independently with probability α, which real models do not do. The draft model costs nothing here. The default rule is the figure’s: a round keeps only the matching draft tokens. The checkbox keeps the target’s own token at the first mismatch too, which the verify pass has already computed; that variant is ours.</p>
      </Simplification>
      <SpecBench />
      <p>
        Under the figure’s rule a round can keep nothing at all, so speculation needs α above one half to beat plain
        decoding even at great depth, and higher still at small depths. Keeping the target’s token makes every pass worth
        at least one token. Either way, each target pass reads the cache once for k query rows.
      </p>
      <Ask q={questionsS206['ch20/20.6/q-spec-ai']} />
    </>
  );
}
