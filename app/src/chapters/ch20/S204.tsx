import { useMemo, useState, type KeyboardEvent } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Beyond, Bench, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  MAT_NAMES,
  aiPoints,
  baselineStep,
  cacheBytes,
  cachedStep,
  diffStep,
  generate,
  mustCompute,
  scoreMarks,
  type CellKind,
  type MatName,
} from '../../sim/kvcache';
import { DecodeStreamScene } from './DecodeStreamScene';
import { fmtBytes } from './KvExplorer';

// ---------------------------------------------------------------- formatting

const minus = (s: string) => s.replace(/^-/, '−');
function cellText(name: MatName, v: number): string {
  if (v === -Infinity) return '−∞';
  if (Math.abs(v) < 0.005) return '0';
  if (name === 'P') return v.toFixed(2).replace(/^0/, '');
  if (name === 'S' || name === 'O') return Math.abs(v) < 0.05 ? '0' : minus(v.toFixed(1));
  return minus(String(Number(v.toFixed(2))));
}
function fmtFlops(f: number): string {
  const units: [number, string][] = [[1e18, 'EFLOP'], [1e15, 'PFLOP'], [1e12, 'TFLOP'], [1e9, 'GFLOP'], [1e6, 'MFLOP'], [1e3, 'kFLOP']];
  for (const [s, u] of units) if (f >= s) return `${fmt(f / s)} ${u}`;
  return `${fmt(f)} FLOP`;
}

// ---------------------------------------------------------------- which cells change

const N0 = 6;
const D0 = 4;
const TITLE: Record<MatName, string> = {
  X: 'X (embeddings)',
  Q: 'Q = X W_Q',
  K: 'K = X W_K',
  V: 'V = X W_V',
  S: 'S = Q Kᵀ/√d + M',
  P: 'P = softmax(S)',
  O: 'O = P V',
};
const empty = (): Record<MatName, Set<number>> => ({ X: new Set(), Q: new Set(), K: new Set(), V: new Set(), S: new Set(), P: new Set(), O: new Set() });

function CellsBench() {
  const [seed, setSeed] = useState(1);
  const diff = useMemo(() => diffStep(N0, D0, seed), [seed]);
  const [marks, setMarks] = useState(empty);
  const [revealed, setRevealed] = useState(false);
  const score = useMemo(() => scoreMarks(diff, marks), [diff, marks]);
  const need = useMemo(() => mustCompute(diff), [diff]);
  const total = MAT_NAMES.reduce((s, m) => s + need[m].size, 0);

  const toggle = (name: MatName, i: number) => {
    if (revealed) return;
    setMarks((m) => {
      const next = { ...m, [name]: new Set(m[name]) };
      if (next[name].has(i)) next[name].delete(i);
      else next[name].add(i);
      return next;
    });
  };
  const onKey = (e: KeyboardEvent, name: MatName, i: number) => {
    if (e.key === ' ' || e.key === 'Enter') {
      e.preventDefault();
      toggle(name, i);
    }
  };
  const marked = MAT_NAMES.reduce((s, m) => s + marks[m].size, 0);

  const grid = (name: MatName) => {
    const a = diff.after[name];
    const b = diff.before[name];
    return (
      <div key={name} data-mat={name}>
        <div className="mx-label t-global">{TITLE[name]}</div>
        <div className="mx global" style={{ gridTemplateColumns: `repeat(${a.cols}, auto)` }}>
          {diff.kind[name].map((k: CellKind, i) => {
            const r = Math.floor(i / a.cols);
            const c = i % a.cols;
            const old = r < b.rows && c < b.cols;
            const isMarked = marks[name].has(i);
            let cls = '';
            let strike = false;
            let text = '';
            if (!revealed) {
              cls = isMarked ? 'on' : old ? '' : 'void';
              text = old ? cellText(name, b.v[r * b.cols + c]) : '';
            } else {
              text = cellText(name, a.v[i]);
              if (k === 'new' || k === 'changed') cls = isMarked ? 'good' : 'bad';
              else if (k === 'masked') cls = 'void';
              else if (isMarked) {
                cls = 'bad';
                strike = true;
              }
            }
            return (
              <span key={i} className={cls} role="checkbox" aria-checked={isMarked} tabIndex={revealed ? -1 : 0}
                aria-label={`${name} row ${r} column ${c}`} data-kind={revealed ? k : undefined} data-marked={isMarked ? '' : undefined}
                onClick={() => toggle(name, i)} onKeyDown={(e) => onKey(e, name, i)}
                style={{ minWidth: '2.2rem', height: '1.7rem', fontSize: '0.7rem', cursor: revealed ? 'default' : 'pointer', textDecoration: strike ? 'line-through' : undefined }}>
                {text}
              </span>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <Bench
      title="Which cells change?"
      task={`Token ${N0 + 1} arrives. Mark every cell that needs new arithmetic, then reveal.`}
      foot={
        <>
          Values are rounded for display; the reveal compares them unrounded. Dashed cells before the reveal are the slots
          the new token adds. After it: green, a cell that needs computing and you marked; red, one you missed; red and
          struck through, a cell you marked that did not change; dashed, a masked cell whose value is known without arithmetic.
        </>
      }
    >
      <div className="row" style={{ alignItems: 'flex-start', gap: '1rem', marginBottom: '0.9rem' }}>
        {(['X', 'Q', 'K', 'V'] as MatName[]).map(grid)}
      </div>
      <div className="row" style={{ alignItems: 'flex-start', gap: '1rem', marginBottom: '0.9rem' }}>
        {(['S', 'P', 'O'] as MatName[]).map(grid)}
      </div>
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <button className="primary" onClick={() => setRevealed(true)} disabled={revealed}>Reveal</button>
        <button onClick={() => { setMarks(empty()); setRevealed(false); }}>Clear</button>
        <button onClick={() => { setSeed((s) => s + 1); setMarks(empty()); setRevealed(false); }}>Another problem</button>
        <span className="muted" aria-live="polite">{marked} marked</span>
      </div>
      {revealed ? (
        <div className="narrate" role="status" data-score={`${score.hit}/${score.needed}/${score.missed}/${score.extraSame}/${score.extraMasked}`}>
          {score.missed === 0 && score.extraSame === 0 && score.extraMasked === 0 ? (
            <b className="t-ok">All {total} cells exactly. </b>
          ) : (
            <>You marked {score.hit} of the {total} cells that need new arithmetic{score.missed ? `, and missed ${score.missed}` : ''}. </>
          )}
          {score.extraSame ? <>You marked {score.extraSame} {score.extraSame === 1 ? 'cell' : 'cells'} that did not change at all: not even in the last bit. </> : null}
          {score.extraMasked ? (
            <>You marked {score.extraMasked} masked {score.extraMasked === 1 ? 'cell' : 'cells'} of the new column, as Fig. 20.5 darkens them: they exist in the bigger matrix, but each is −∞ in S and 0 in P, known without arithmetic. </>
          ) : null}
          The new work is one row of each matrix: {D0} values each for X, Q, K, V and O, and {N0 + 1} each for S and P.
        </div>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- recompute or cache

const WIDTHS = [512, 1024, 2048, 4096, 5120, 8192, 12288];

function PerTokenPlot({ D, layers, at }: { D: number; layers: number; at: number }) {
  const W = 560;
  const H = 230;
  const pad = { l: 64, r: 14, t: 12, b: 36 };
  const nMax = 32768;
  const yMin = cachedStep(1, D, layers).total / 2;
  const yMax = baselineStep(nMax, D, layers).total * 2;
  const X = (n: number) => pad.l + (Math.log10(n) / Math.log10(nMax)) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * (H - pad.t - pad.b);
  const ns = Array.from({ length: 61 }, (_, i) => Math.round(nMax ** (i / 60)));
  const line = (f: (n: number) => number) => ns.map((n) => `${X(n).toFixed(1)},${Y(f(n)).toFixed(1)}`).join(' ');
  const yt: number[] = [];
  for (let e = Math.ceil(Math.log10(yMin)); e <= Math.floor(Math.log10(yMax)); e += 2) yt.push(10 ** e);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="FLOPs per generated token against context length" style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {[1, 10, 100, 1000, 10000].map((n) => (
        <g key={n}>
          <line x1={X(n)} x2={X(n)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(n)} y={H - pad.b + 14} fontSize="10.5" textAnchor="middle">{n.toLocaleString('en-US')}</text>
        </g>
      ))}
      {yt.map((v) => (
        <g key={v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 6} y={Y(v) + 4} fontSize="10" textAnchor="end">10<tspan dy="-5" fontSize="7.5">{Math.round(Math.log10(v))}</tspan></text>
        </g>
      ))}
      <text x={(W + pad.l) / 2} y={H - 4} fontSize="11" textAnchor="middle">tokens in the context, n</text>
      <polyline points={line((n) => baselineStep(n, D, layers).total)} fill="none" stroke="var(--fail)" strokeWidth="2" />
      <polyline points={line((n) => cachedStep(n, D, layers).total)} fill="none" stroke="var(--ok)" strokeWidth="2" />
      <text x={X(3)} y={Y(baselineStep(3, D, layers).total) - 10} fontSize="11" style={{ fill: 'var(--fail)' }}>recompute everything</text>
      <text x={X(200)} y={Y(cachedStep(200, D, layers).total) - 8} fontSize="11" style={{ fill: 'var(--ok)' }}>with a KV cache</text>
      <line x1={X(at)} x2={X(at)} y1={pad.t} y2={H - pad.b} stroke="var(--ink)" strokeDasharray="3 3" />
    </svg>
  );
}

function RecomputeBench() {
  const [logN, setLogN] = useState(11);
  const [D, setD] = useState(4096);
  const [layers, setLayers] = useState(32);
  const [logT, setLogT] = useState(8);
  const [p, setP] = useState(2);
  const N = Math.round(2 ** logN);
  const T = Math.round(2 ** logT);
  const n = N + 1;
  const base = baselineStep(n, D, layers);
  const cach = cachedStep(n, D, layers);
  const gen = generate(N, T, D, layers);
  const held = cacheBytes(N + T, D, layers, p);
  const row = (label: string, a: number, b: number) => (
    <tr key={label}>
      <td>{label}</td>
      <td className="num">{fmtFlops(a)}</td>
      <td className="num">{fmtFlops(b)}</td>
    </tr>
  );
  return (
    <Bench
      title="Recompute or cache"
      task="Move the context length and watch which costs grow with it, and how fast."
      foot={
        <>
          FLOPs count a multiply-add as two, per generated token, summed over the layers. D is the width of a token’s Q, K
          and V rows over all heads (the book’s hidden dimension h_q × d, p. 505). Model: <code>src/sim/kvcache.ts</code>.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Tokens already in the context, N = {N.toLocaleString('en-US')}
          <input type="range" min="0" max="15" step="0.25" value={logN} aria-label="Context length" onChange={(e) => setLogN(Number(e.target.value))} />
        </label>
        <label className="field">
          Width D
          <select value={D} onChange={(e) => setD(Number(e.target.value))} aria-label="Width">
            {WIDTHS.map((w) => <option key={w} value={w}>{w.toLocaleString('en-US')}</option>)}
          </select>
        </label>
        <label className="field">
          Layers = {layers}
          <input type="range" min="1" max="96" value={layers} aria-label="Layers" onChange={(e) => setLayers(Number(e.target.value))} />
        </label>
        <label className="field">
          Generate T = {T.toLocaleString('en-US')} tokens
          <input type="range" min="0" max="13" step="1" value={logT} aria-label="Tokens to generate" onChange={(e) => setLogT(Number(e.target.value))} />
        </label>
        <label className="field">
          Cache element
          <select value={p} onChange={(e) => setP(Number(e.target.value))} aria-label="Cache element">
            <option value={4}>fp32, 4 bytes</option>
            <option value={2}>16-bit, 2 bytes</option>
          </select>
        </label>
      </div>
      <div className="cols plot">
        <PerTokenPlot D={D} layers={layers} at={n} />
        <div>
          <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-recompute="">
            <thead>
              <tr><th>The next token (n = {n.toLocaleString('en-US')})</th><th className="num">recompute</th><th className="num">cached</th></tr>
            </thead>
            <tbody>
              {row('X W_Q, X W_K, X W_V', base.projections, cach.projections)}
              {row('Q Kᵀ', base.scores, cach.scores)}
              {row('P V', base.values, cach.values)}
              <tr>
                <td><b>per token</b></td>
                <td className="num" data-base={base.total}><b>{fmtFlops(base.total)}</b></td>
                <td className="num" data-cached={cach.total}><b>{fmtFlops(cach.total)}</b></td>
              </tr>
              <tr>
                <td>all {T.toLocaleString('en-US')} tokens</td>
                <td className="num">{fmtFlops(gen.baseline)}</td>
                <td className="num">{fmtFlops(gen.cached)}</td>
              </tr>
            </tbody>
          </table>
          <p style={{ margin: '0.7rem 0 0' }} data-ratio={base.total / cach.total}>
            Recomputing costs <b>{fmt(base.total / cach.total)}×</b> the cached step for this token,{' '}
            <b>{fmt(gen.baseline / gen.cached)}×</b> over all {T.toLocaleString('en-US')}. The price: after them the caches
            hold <b data-held={held}>{fmtBytes(held)}</b> ({(N + T).toLocaleString('en-US')} tokens × {layers} layers × K and V × {D.toLocaleString('en-US')} × {p} bytes).
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- roofline

interface Hardware {
  key: string;
  label: string;
  peak: number; // GFLOP/s
  bw: number; // GB/s
  note: string;
}
const BOOK: Hardware = {
  key: 'book',
  label: 'The book’s example (66.9 TFLOP/s, 3.35 TB/s)',
  peak: 66900,
  bw: 3350,
  note: 'The figures the book quotes in chapter 5 (p. 94), shown as the book’s example only.',
};
function measured(reports: Reports): Hardware[] {
  return realBackends(reports).flatMap((b) => {
    const r = benchRows(reports[b], 'ch05/lab1_roofline').find((x) => x.name === 'roofline');
    const peak = num(r?.compute_gflops);
    const bw = num(r?.bandwidth_gb_per_s);
    if (!peak || !bw) return [];
    return [{ key: b, label: `Yours: ${String(reports[b].device.name ?? b)} (${b})`, peak, bw, note: `Lab 5.1’s two probe kernels on ${b}, measured on ${reports[b].generated.slice(0, 10)}.` }];
  });
}

function RooflinePlot({ hw, points }: { hw: Hardware; points: { label: string; ai: number; prefill: boolean }[] }) {
  const W = 620;
  const H = 330;
  const pad = { l: 60, r: 14, t: 14, b: 40 };
  const xMin = 0.1;
  const xMax = 1000;
  const yMax = hw.peak * 3;
  const yMin = hw.bw * xMin * 0.5;
  const X = (v: number) => pad.l + ((Math.log10(v) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin))) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(yMin)) / (Math.log10(yMax) - Math.log10(yMin))) * (H - pad.t - pad.b);
  const ridge = hw.peak / hw.bw;
  const roof = (x: number) => Math.min(hw.peak, hw.bw * x);
  const yt: number[] = [];
  for (let e = Math.ceil(Math.log10(yMin)); e <= Math.floor(Math.log10(yMax)); e++) yt.push(10 ** e);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Roofline with prefill and decode products" style={{ width: '100%', maxWidth: W, display: 'block' }} data-ridge={ridge.toFixed(3)}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      <rect x={pad.l} y={pad.t} width={Math.max(0, X(ridge) - pad.l)} height={H - pad.t - pad.b} fill="var(--global-wash)" />
      <text x={pad.l + 6} y={pad.t + 15} fontSize="10.5" style={{ fill: 'var(--global)' }}>bandwidth-bound</text>
      <text x={X(ridge) + 6} y={pad.t + 15} fontSize="10.5" style={{ fill: 'var(--reg)' }}>compute-bound</text>
      {[0.1, 1, 10, 100, 1000].map((v) => (
        <g key={v}>
          <line x1={X(v)} x2={X(v)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(v)} y={H - pad.b + 14} fontSize="10.5" textAnchor="middle">{v}</text>
        </g>
      ))}
      {yt.map((v) => (
        <g key={v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 6} y={Y(v) + 4} fontSize="10" textAnchor="end">{v >= 1000 ? `${v / 1000}k` : v}</text>
        </g>
      ))}
      <text x={(W + pad.l) / 2} y={H - 6} fontSize="11" textAnchor="middle">FLOP per byte</text>
      <text x={13} y={(H - pad.b + pad.t) / 2} fontSize="11" textAnchor="middle" transform={`rotate(-90 13 ${(H - pad.b + pad.t) / 2})`}>GFLOP/s ceiling</text>
      <polyline points={`${X(xMin)},${Y(hw.bw * xMin)} ${X(ridge)},${Y(hw.peak)} ${X(xMax)},${Y(hw.peak)}`} fill="none" stroke="var(--ink)" strokeWidth="2.2" />
      <line x1={X(ridge)} x2={X(ridge)} y1={Y(hw.peak)} y2={H - pad.b} stroke="var(--ink)" strokeDasharray="3 3" />
      <text x={X(ridge)} y={H - pad.b - 5} fontSize="10.5" textAnchor="middle">ridge {fmt(ridge)}</text>
      {points.map((pt, i) => {
        const x = X(Math.min(Math.max(pt.ai, xMin), xMax));
        const y = Y(roof(pt.ai));
        return (
          <g key={pt.label} data-point={pt.label}>
            <circle cx={x} cy={y} r="5" fill={pt.prefill ? 'var(--reg)' : 'var(--page)'} stroke={pt.prefill ? 'var(--reg)' : 'var(--global)'} strokeWidth="2" />
            <text x={x + 8} y={y + (i % 2 ? 15 : -7)} fontSize="10.5">{pt.label}</text>
          </g>
        );
      })}
    </svg>
  );
}

function RooflineBench() {
  const { reports } = useReports();
  const options = useMemo(() => [...measured(reports), BOOK], [reports]);
  const [key, setKey] = useState<string | null>(null);
  const hw = options.find((o) => o.key === key) ?? options[0];
  const [logN, setLogN] = useState(11);
  const [p, setP] = useState(2);
  const [d, setD] = useState(128);
  const N = Math.round(2 ** logN);
  const pts = aiPoints(N, d, p);
  const ridge = hw.peak / hw.bw;
  return (
    <Bench
      title="Prefill and decode on your roofline"
      task="Slide the prompt length: which points move, and which never leave the slope?"
      foot={<>Open circles are decode, filled circles prefill; each sits at its ceiling, min(peak, bandwidth × FLOP/B). {hw.note}</>}
    >
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Roof
          <select value={hw.key} onChange={(e) => setKey(e.target.value)} aria-label="Roof">
            {options.map((o) => <option key={o.key} value={o.key}>{o.label}</option>)}
          </select>
        </label>
        <label className="field">
          Prompt length N = {N.toLocaleString('en-US')}
          <input type="range" min="0" max="14" step="0.25" value={logN} aria-label="Prompt length" onChange={(e) => setLogN(Number(e.target.value))} />
        </label>
        <label className="field">
          Element
          <select value={p} onChange={(e) => setP(Number(e.target.value))} aria-label="Element size">
            <option value={4}>fp32, 4 bytes</option>
            <option value={2}>16-bit, 2 bytes</option>
          </select>
        </label>
        <label className="field">
          Head width d
          <select value={d} onChange={(e) => setD(Number(e.target.value))} aria-label="Head width">
            {[64, 128, 256].map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
      </div>
      <div className="cols plot">
        <RooflinePlot hw={hw} points={pts.map((x) => ({ label: x.label, ai: x.ai, prefill: x.key.startsWith('prefill') }))} />
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-roofline="">
          <thead><tr><th>Product</th><th className="num">FLOP/B</th><th>Bound</th></tr></thead>
          <tbody>
            {pts.map((x) => (
              <tr key={x.key} data-key={x.key} data-ai={x.ai.toFixed(4)} data-bound={x.ai < ridge ? 'bandwidth' : 'compute'}>
                <td>{x.label}</td>
                <td className="num">{fmt(x.ai)}</td>
                <td>{x.ai < ridge ? <span className="chip global">bandwidth</span> : <span className="chip reg">compute</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions

export const questionsS204: Record<string, Question> = {
  'ch20/20.4/q-unchanged': {
    kind: 'choice',
    id: 'ch20/20.4/q-unchanged',
    prompt: <>When token N joins, why do rows 0 to N − 1 of P stay exactly the same, down to the last bit?</>,
    options: [
      { label: 'Each of those rows gains one entry, and it is masked, so it adds 0 to every sum.', correct: true },
      { label: 'They change slightly; the KV cache is an approximation that ignores the change.', why: 'The reveal compared them unrounded: identical. The cache is exact.' },
      { label: 'They change, but O does not, because multiplying by V cancels the change.', why: 'P does not change in the first place, and O’s old rows do not either.' },
      { label: 'Subtracting the row maximum cancels the new entry before the sum is taken.', why: 'The maximum stays the same because the new entry is −∞; the reason is the mask, not the subtraction.' },
    ],
    explain: (
      <p>
        Row r of S + M gains column N, and N &gt; r puts it above the diagonal, where M holds −∞. The row’s maximum is
        unchanged and its sum of exponentials gains e<sup>−∞</sup> = 0, which adds nothing to the denominator and nothing to
        the numerators, so every p in the row keeps its value. The book’s
        words: the masked elements “do not dilute any of the probabilities”.
      </p>
    ),
    source: 'ch. 20 §20.4, pp. 489–490, Fig. 20.5',
  },
  'ch20/20.4/q-new-row-cost': {
    kind: 'numeric',
    id: 'ch20/20.4/q-new-row-cost',
    prompt: (
      <>
        With a KV cache, one head of one layer, N = 4,096 cached tokens and d = 128: how many FLOPs do q′Kᵀ and p′V take
        together? Count a multiply-add as 2.
      </>
    ),
    answer: 2097152,
    explain: (
      <p>
        q′Kᵀ is N dot products of length d: 2Nd FLOPs. p′V adds N rows of d values, each scaled: another 2Nd. Together
        4Nd = 4 × 4,096 × 128 = 2,097,152. The three projections of the new row are separate and do not depend on N.
      </p>
    ),
    source: 'ch. 20 §20.4, p. 490, Fig. 20.6',
  },
  'ch20/20.4/q-fp32-ai': {
    kind: 'numeric',
    id: 'ch20/20.4/q-fp32-ai',
    prompt: <>That decode step reads every row of K and of V once, in fp32 (4 bytes per element). How many FLOPs does it do per byte read?</>,
    answer: 0.5,
    tol: 0.02,
    unit: 'FLOP/B',
    explain: (
      <p>
        4Nd FLOPs over 2Nd elements × 4 bytes = 0.5 FLOP/B, whatever N. In 16-bit the same step does 1 FLOP/B: that is the
        book’s Eq. 20.9 (about one operation per element, counted with p = 1), and the cited source states it as a 1:1
        FLOP-to-byte ratio for BF16 (facts E4).
      </p>
    ),
    source: 'ch. 20 §20.6, pp. 506–507, Eq. 20.9; notes/today/ch20-facts.md E4',
  },
  'ch20/20.4/q-is-causal': {
    kind: 'choice',
    id: 'ch20/20.4/q-is-causal',
    prompt: (
      <>
        A decode step calls <code>F.scaled_dot_product_attention(q, k, v, is_causal=True)</code> with one query and N cached
        keys. Which keys does the query attend to?
      </>
    ),
    options: [
      { label: 'Only key 0: PyTorch aligns the causal mask to the upper left.', correct: true },
      { label: 'All N keys, as a decode step should, since nothing comes after it.', why: 'That is what you want, and what FlashAttention (2.1 and later) and MLX do, but not what PyTorch’s flag does for a non-square mask.' },
      { label: 'Only the last key, the one that was just appended to the cache.', why: 'Upper-left alignment keeps the first column of the first row, not the last.' },
      { label: 'None: PyTorch raises an error for a non-square causal mask.', why: 'It does not: it documents the upper-left alignment.' },
    ],
    explain: (
      <p>
        PyTorch documents <code>is_causal</code> as an upper-left causal bias when the query is shorter than the keys:
        row 0 keeps column 0 only. For one query, pass no mask at all (the Llama reference builds a mask only when more than one token is
        processed) or a lower-right causal bias.
      </p>
    ),
    source: 'notes/today/ch20-facts.md D5, D3, A8, D4',
  },
  'ch20/20.4/q-prefill-bound': {
    kind: 'choice',
    id: 'ch20/20.4/q-prefill-bound',
    prompt: <>Prefill of a 2,000-token prompt and one decode step both compute q Kᵀ. Why is the first usually compute-bound and the second bandwidth-bound?</>,
    options: [
      { label: 'Each loaded K row meets 2,000 queries in prefill and one in decode.', correct: true },
      { label: 'Prefill runs on the tensor cores and decode on the ordinary cores.', why: 'Hardware choice follows from the shape, not the other way round; a GEMV gives tensor cores nothing to reuse.' },
      { label: 'Decode runs a softmax every step; prefill runs it once at the end.', why: 'Both run one over their scores, and the book’s count ignores it in both.' },
      { label: 'Prefill’s working set fits in cache, and decode’s KV cache does not.', why: 'The argument needs no cache at all: it is about how many times each loaded byte is used.' },
    ],
    explain: (
      <p>
        Chapter 5’s lesson in a new place. A loaded key row costs the same bytes in both phases; what differs is how many
        multiply-adds it feeds. Prefill is a GEMM whose intensity grows with N towards about 2d/p FLOP/B; decode is a GEMV
        near 2/p, whatever N. The roofline explorer shows the prefill point sliding right with N while the decode point
        stays where it is.
      </p>
    ),
    source: 'ch. 20 §20.4, pp. 491–492; ch. 5 §5.1',
  },
  'ch20/20.4/q-per-layer': {
    kind: 'choice',
    id: 'ch20/20.4/q-per-layer',
    prompt: <>Fig. 20.7 hands K and V from each layer to the same layer of the next iteration. Why can’t one cache serve every layer?</>,
    options: [
      { label: 'Each layer computes K and V from its own input with its own weights.', correct: true },
      { label: 'It could; one cache per layer is only a bookkeeping choice in the code.', why: 'The values differ from layer to layer, so they must be stored separately.' },
      { label: 'The layers have different head widths, so the rows would not line up.', why: 'In the models here they do not; the reason is that the values differ.' },
      { label: 'The causal mask differs from layer to layer, so the keys would not match.', why: 'The causal mask is the same in every layer.' },
    ],
    explain: (
      <p>
        X for layer 1 is the embeddings; X for layer 2 is layer 1’s output, and so on: a token’s K in layer l depends on
        everything the layers before l did to it. So K = X W_K is a different matrix in every layer, and the cache grows with the number of layers as well as with N (Eq. 20.7 has the factor l).
      </p>
    ),
    source: 'ch. 20 §20.4, p. 491, Fig. 20.7',
  },
};

// ---------------------------------------------------------------- the page

export function S204() {
  return (
    <>
      <p>
        Section 20.2 ran each new token through every layer by recomputing everything: in every generation step, each layer
        multiplies out X W_Q, X W_K, X W_V, then Q Kᵀ, then P V, five matrix products over all N rows of X. N can reach tens
        of thousands of tokens. Yet from one step to the next, X gains exactly one row. Before reading the book’s answer,
        work out for yourself how much of that arithmetic produces something new.
      </p>
      <Source section="20.4" pages="488" />

      <h2>Predict which cells change</h2>
      <p>
        Below is one head with six tokens and d = 4, drawn like the book’s Fig. 20.2, with the 1/√d scale and the mask
        folded into S. A seventh token is about to arrive. Each grid is already the size it will have with seven tokens;
        the dashed slots are the cells the new token adds. Click every cell you expect to need new arithmetic, then reveal.
        The reveal runs the attention at six tokens and at seven and compares every value.
      </p>
      <Simplification>
        <p>One head, four columns, small round inputs so that you can read the values. The models of section 20.6 have heads 128 or 256 wide and dozens of them per layer; the pattern is the same.</p>
      </Simplification>
      <CellsBench />
      <p>
        X, Q, K and V each gain one bottom row: the new token’s embedding, and three vector-matrix products of it with
        W_Q, W_K and W_V. In S, every old cell (r, c) depends only on query row r and key row c, so none moves. The new row
        of S is the new query against all keys, q′Kᵀ, one more vector-matrix product. The new column is the old queries
        against the new key, and every cell of it except (N, N) lies above the diagonal, so the mask makes it −∞. The book
        calls these “all zero elements”; that is their value after the softmax, not before it.
      </p>
      <Source section="20.4" pages="489" figs="Fig. 20.5" />
      <p>
        The old rows of P and O do not change either: not approximately, but bit for bit, as the reveal checks. The
        question below asks why.
      </p>
      <Ask q={questionsS204['ch20/20.4/q-unchanged']} />

      <h2>Keep K and V</h2>
      <p>
        So a decode step takes one new row of X into an attention head and must produce one new row of O. It still needs
        all of K and V: every earlier key to score the new query against, and every earlier value to mix. Those were
        computed in earlier steps and do not change, so store them instead of recomputing them. That store is the{' '}
        <b>KV cache</b>. (The book&rsquo;s footnote 6 adds that &ldquo;KV&rdquo; here is not a key-value store, though
        the idea is close: the output is a sum of values, weighted by how near the query is to each value&rsquo;s key.)
      </p>
      <Source section="20.4" pages="490" figs="footnote 6" />

      <h2>One token’s work</h2>
      <p>With a cache, one step of one head is five small operations (Fig. 20.6):</p>
      <ol>
        <li>q′ = x′W_Q, k′ = x′W_K, v′ = x′W_V: three vector-matrix products of the new row, 1 × d each;</li>
        <li>append k′ to K and v′ to V, which now have N + 1 rows;</li>
        <li>s′ = q′Kᵀ, a row of N + 1 scores;</li>
        <li>p′ = softmax(s′);</li>
        <li>o′ = p′V, one row of output.</li>
      </ol>
      <p>
        There is no mask. The newest token may attend to every cached token, and no later token exists yet for it to be
        kept from.
      </p>
      <Source section="20.4" pages="490–491" figs="Fig. 20.6" />
      <p>
        Meta’s reference Llama code agrees: it builds a causal mask only when more than one token goes through at once,
        that is, in prefill. A decode kernel can skip the mask entirely, which is one comparison per score saved and one
        source of bugs gone.
      </p>
      <Today checked="2026-11-20 (ch20-facts D3)">
        <p>In <code>llama/model.py</code> the mask is <code>None</code> unless <code>seqlen &gt; 1</code>; then it is a <code>torch.triu</code> mask with zeros prepended for the cached positions.</p>
      </Today>

      <h2>Prefill fills every layer’s cache</h2>
      <p>
        The first pass over the prompt, which section 20.2 called summarization, is also called <b>prefill</b>: every
        layer runs the full matrix version on the whole prompt and leaves its K and V behind. Every layer has its own
        cache. In each generation step the last output token becomes one new row for layer 1; each layer computes q′, k′
        and v′, appends to its own cache, computes o′ and passes one row on to the next layer.
      </p>
      <Source section="20.4" pages="491" figs="Fig. 20.7" />
      <Ask q={questionsS204['ch20/20.4/q-per-layer']} />

      <h2>Recompute or cache</h2>
      <p>
        Now count. The explorer adds up the five products for the next token with and without the cache. Before you move
        anything, decide which of the three rows of the table will grow fastest as N grows.
      </p>
      <Simplification>
        <p>Only the book’s five products are counted: no output projection, no feed-forward layer, no softmax. Those cost the same with or without a cache, except the softmax, which also shrinks to one row.</p>
      </Simplification>
      <RecomputeBench />
      <p>
        Recomputing grows with N² in Q Kᵀ and P V and with N D² in the projections. The cached step grows with N D in
        attention and not at all in the projections. And the ratio between them is exactly the number of tokens n: the
        baseline redoes the same row of work for every one of them, 6nD² + 4n²D = n(6D² + 4nD). The cost has moved from
        arithmetic to memory: the cache holds 2 × n × D × layers × p bytes, and the next section sizes it.
      </p>
      <Ask q={questionsS204['ch20/20.4/q-new-row-cost']} label="Work it out" />

      <h2>Compute-bound prefill, bandwidth-bound decode</h2>
      <p>
        The book characterises the two phases in one sentence each. Prefill is one pass of large matrix-matrix
        multiplications, typically compute-bound, and can keep a GPU busy. Generation is one pass per token of mostly
        vector-matrix products, typically memory-bandwidth-bound, and leaves most of the arithmetic units idle.
      </p>
      <Source section="20.4" pages="491–492" />
      <p>
        Chapter 5 gives the reason. In a GEMM each loaded row of K, or of a weight matrix, is reused by every row of the
        other operand; in a GEMV each loaded byte feeds one multiply-add and is never used again. Place the four products
        of one head on your own roofline, from <a href={href('ch05', '5.1')}>lab 5.1</a>.
      </p>
      <Simplification>
        <p>Compulsory traffic only: each operand read once and the result written once, no caches, one head’s products. Prefill’s Q Kᵀ writes its N × N scores, as the unfused baseline of section 20.3 does.</p>
      </Simplification>
      <RooflineBench />
      <p>
        Prefill’s points slide right as the prompt grows and cross the ridge; its Q Kᵀ approaches 2d/p FLOP per byte. The
        decode points stay near 2/p however long the cache is: 1 FLOP per byte in 16-bit, 0.5 in fp32. Watch a whole
        decode step stream through memory.
      </p>
      <Simplification>
        <p>
          One layer of a Llama 2 7B shape (32 heads of 128) at 4,096 tokens, 16-bit. Each byte is counted once: no cache
          reuse, no overlap. Only the Q, K and V projections and the two attention products; no output projection, no
          feed-forward.
        </p>
      </Simplification>
      <DecodeStreamScene />
      <Ask q={questionsS204['ch20/20.4/q-fp32-ai']} label="Work it out" />
      <Ask q={questionsS204['ch20/20.4/q-prefill-bound']} />

      <h2>How the cache is laid out in code you will read</h2>
      <p>
        The book stops at the algebra: it shows no data structure for the cache, no append and no decode kernel. Three
        code bases you are likely to read store it three ways: Meta&rsquo;s preallocated array, llama.cpp&rsquo;s ring of
        token slots, and mlx-lm&rsquo;s cache that grows in steps.
      </p>
      <Today checked="2026-11-20 (ch20-facts D3)">
        <p>Meta’s Llama reference preallocates <code>cache_k</code> as [max_batch, max_seq, kv_heads, head_dim] and writes <code>cache_k[:bsz, start_pos : start_pos + seqlen]</code>. llama.cpp keeps one K and one V tensor per layer, a ring of token slots written by index with <code>ggml_set_rows</code>; it has no block table. mlx-lm grows its cache in steps of 256 tokens.</p>
      </Today>
      <p>
        Lab 20.4 uses the first layout, one cache per layer of shape [batch, max_seq, kv_heads, head_dim], and asks you to
        write the append and the single-token decode step. Tokens come before heads, so appending one token writes
        one contiguous run of kv_heads × head_dim values.
      </p>
      <Beyond>
        <p>Lab 20.4 is ours: the book has no cache or decode code. See the <a href={href('ch20', 'labs')}>labs page</a>.</p>
      </Beyond>

      <h2>Two traps when you call a library</h2>
      <p>
        A decode step has one query and N keys, so its score matrix is 1 × N, not square. Libraries disagree about where a
        causal mask goes in a non-square matrix, and one popular call gets the decode step wrong if you pass the causal
        flag out of habit.
      </p>
      <Today checked="2026-11-20 (ch20-facts D5, A8, D4)">
        <p>PyTorch’s <code>is_causal=True</code> aligns the mask to the upper left when the query is shorter than the keys. FlashAttention (since 2.1) and MLX align it to the lower right, where the last query meets the last key.</p>
      </Today>
      <Ask q={questionsS204['ch20/20.4/q-is-causal']} />
      <p>
        The second trap is what the cache holds. In the book’s transformer, position information is added once to the
        embeddings. In the open models you will load, a rotary embedding is applied to Q and K inside every attention
        layer, before K is written to the cache. So the cache holds rotated keys, and a kernel that fuses the append must
        rotate first.
      </p>
      <Today checked="2026-11-20 (ch20-facts E8, D3)">
        <p>Llama 2 uses rotary positional embeddings; the reference code calls <code>apply_rotary_emb</code> on <code>xq</code> and <code>xk</code> and then writes <code>xk</code> into <code>cache_k</code>.</p>
      </Today>
    </>
  );
}
