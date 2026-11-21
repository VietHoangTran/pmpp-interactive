import { useMemo, useState, type CSSProperties, type DragEvent } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, NoteGroup, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  CHAIN_ROWS,
  DEFAULT_SETTINGS,
  M5_OPTIONS,
  M5_ORDER,
  MODELLED,
  ORDERS,
  ORDER_IDS,
  P_ORDER,
  chainRow,
  finalSet,
  optionsOf,
  pOrderCounts,
  runChain,
  stage,
  type ChainOptions,
  type ChainResult,
  type ChainSettings,
  type OrderId,
  type StageId,
} from '../../sim/chain';
import { ChainRowScene } from './ChainRowScene';

const FACTS = 'notes/today/x1-facts.md';
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const isModelled = (s: StageId) => (MODELLED as StageId[]).includes(s);
const PARAM: Partial<Record<StageId, (s: ChainSettings) => string>> = {
  'top-k': (s) => `k = ${s.k}`,
  'top-p': (s) => `p = ${s.p.toFixed(2)}`,
  'min-p': (s) => `min-p = ${s.minP.toFixed(2)}`,
  temperature: (s) => `T = ${fmtT(s.T)}`,
};
function fmtT(T: number) {
  return T < 1 ? T.toFixed(2) : T.toFixed(1);
}
const tFromSlider = (v: number) => Number((0.1 * Math.pow(40, v / 1000)).toPrecision(2));
const sliderFromT = (T: number) => Math.round((Math.log(T / 0.1) / Math.log(40)) * 1000);
const idsText = (ids: number[]) => ids.join(' ');

// ---------------------------------------------------------------- questions
export const questionsSX11: Record<string, Question> = {
  'x1/X1.1/q-temp-first': {
    kind: 'choice',
    id: 'x1/X1.1/q-temp-first',
    prompt: <>Temperature 0.5 is moved from the end of the chain to the start. Which truncations can now keep a different set of tokens?</>,
    options: [
      { label: 'Top-p and min-p, but not top-k', correct: true },
      { label: 'All three', why: 'Dividing by a positive T keeps the logits in the same order, so the k largest are still the k largest.' },
      { label: 'None: temperature only rescales the logits', why: 'Rescaling logits is not rescaling probabilities. The ratio of two probabilities is exp(difference / T), and it changes with T.' },
      { label: 'Only top-k', why: 'Top-k is the one stage whose set cannot change: it depends only on the order of the logits.' },
    ],
    explain: (
      <p>
        Top-k depends only on the order of the logits, and dividing by T &gt; 0 keeps that order. Top-p sums
        probabilities, and min-p compares each probability with the top one: p<sub>i</sub> / p<sub>max</sub> = exp((l
        <sub>i</sub> − l<sub>max</sub>) / T). At T = 0.5 the distribution is sharper, so fewer tokens reach the same p
        and fewer stay above the same fraction of the top.
      </p>
    ),
    source: 'ours; the orders: notes/today/x1-facts.md D1, D4',
  },
  'x1/X1.1/q-renorm': {
    kind: 'numeric',
    id: 'x1/X1.1/q-renorm',
    prompt: (
      <>
        Sorted probabilities 0.5, 0.25, 0.125, 0.0625, 0.0625. Top-k with k = 3 runs first, then top-p with p = 0.8, as in
        llama.cpp and transformers. How many tokens survive?
      </>
    ),
    answer: 2,
    unit: 'tokens',
    explain: (
      <p>
        Top-p sees only the three survivors, renormalised: 4/7, 2/7, 1/7. The mass before each is 0, 0.571 and 0.857, so
        the third is dropped. Over the full row, top-p would have kept 3 (mass before 0, 0.5, 0.75). Running a truncation
        after another one measures probability among the survivors, which always makes top-p's set smaller or equal.
      </p>
    ),
    source: 'ours; llama.cpp takes a softmax over the current candidates, transformers masks with minus infinity (x1-facts D4, ch11-facts D2)',
  },
  'x1/X1.1/q-hf-more': {
    kind: 'choice',
    id: 'x1/X1.1/q-hf-more',
    prompt: (
      <>
        T = 1.5 and p = 0.9 on the same row. Which order leaves more tokens for the draw: transformers' (temperature before
        top-p) or llama.cpp's (temperature last)?
      </>
    ),
    options: [
      { label: "transformers'", correct: true },
      { label: "llama.cpp's", why: 'llama.cpp decides the set at T = 1. T = 1.5 then only reweights the survivors; it cannot add any.' },
      { label: 'The same number: temperature does not create or remove tokens', why: 'It does not remove tokens itself, but it changes the masses top-p adds up.' },
    ],
    explain: (
      <p>
        T &gt; 1 flattens the distribution, so it takes more tokens to reach 0.9 of the mass. transformers applies T
        first, so its top-p sees the flatter row and keeps more. llama.cpp's top-p sees the row at T = 1 and keeps the same
        set whatever T is; T only changes how the draw weighs the survivors. Below T = 1 the difference reverses.
      </p>
    ),
    source: 'ours; orders: notes/today/x1-facts.md D1, D4',
  },
  'x1/X1.1/q-greedy-switch': {
    kind: 'choice',
    id: 'x1/X1.1/q-greedy-switch',
    prompt: (
      <>
        In llama.cpp's common sampler, with <code>temp</code> exactly 0 and no probabilities requested, what is drawn?
      </>
    ),
    options: [
      { label: 'Nothing is drawn: greedy replaces dist, and the largest logit wins, lowest index on a tie.', correct: true },
      { label: 'dist runs on a distribution that is one-hot on the largest logit.', why: 'That would give the same token, but the common sampler replaces dist with greedy when the last sampler is temperature at 0.' },
      { label: 'An error: temperature must be strictly positive, so the common sampler rejects temp = 0 at start-up.', why: "That is transformers' rule; it asks for do_sample=False instead." },
      { label: 'A random choice among the tokens tied for the largest logit.', why: "The greedy sampler scans with a strict >, so the first maximum stays; there is no random choice." },
    ],
    explain: (
      <p>
        <code>common/sampling.cpp</code> ends the chain with <code>llama_sampler_init_greedy()</code> when the last sampler
        is temperature with <code>temp == 0.0f</code> (or top-k with k = 1), and with <code>llama_sampler_init_dist(seed)</code>{' '}
        otherwise. The greedy sampler keeps index i only if its logit is strictly larger, so the lowest index wins a tie.
        M5 does the same at T ≤ 0 with lab 10.3's argmax.
      </p>
    ),
    source: 'notes/today/x1-facts.md D1, D2; notes/today/ch10-facts.md D2',
  },
};

// ---------------------------------------------------------------- the explorer: drawing
const NB = 16;
function MiniBars({ probs, alive, testId }: { probs: number[]; alive: boolean[]; testId: string }) {
  const W = 200;
  const H = 60;
  const bw = W / NB;
  const max = Math.max(...probs, 1e-9);
  return (
    <svg viewBox={`0 0 ${W} ${H + 14}`} style={{ width: '100%', display: 'block' }} role="img" aria-label={`Probabilities after ${testId}`}>
      <line x1={0} y1={H} x2={W} y2={H} stroke="var(--rule)" />
      {probs.map((p, i) => {
        const h = alive[i] ? Math.max(1.5, (p / max) * (H - 6)) : 6;
        return (
          <g key={i}>
            <rect x={i * bw + 1.5} y={H - h} width={bw - 3} height={h} rx="1"
              fill={alive[i] ? 'var(--ink-3)' : 'none'} fillOpacity={alive[i] ? 0.6 : 0}
              stroke={alive[i] ? 'var(--ink-2)' : 'var(--ink-3)'} strokeDasharray={alive[i] ? undefined : '2 2'} strokeWidth="0.8" />
            <text x={i * bw + bw / 2} y={H + 12} fontSize="9" textAnchor="middle" style={{ fill: alive[i] ? 'var(--ink)' : 'var(--ink-3)' }}>{i}</text>
          </g>
        );
      })}
    </svg>
  );
}

function Strip({ res, label }: { res: ChainResult; label: string }) {
  const W = 620;
  const X0 = 10;
  const SW = W - 2 * X0;
  let at = 0;
  const d = res.draw;
  return (
    <svg viewBox={`0 0 ${W} 52`} style={{ width: '100%', maxWidth: W, display: 'block' }} role="img" aria-label={`The draw for ${label}`} data-part="strip">
      {d.order.map((id, j) => {
        const x = X0 + at * SW;
        const w = d.probs[j] * SW;
        at += d.probs[j];
        const hit = id === d.token;
        return (
          <g key={id} data-segment={id}>
            <rect x={x} y={14} width={Math.max(0, w)} height={20} fill={hit ? 'var(--ink)' : 'var(--ink-3)'} fillOpacity={hit ? 0.85 : 0.3} stroke="var(--ink-2)" strokeWidth="0.7" />
            {w > 16 ? <text x={x + w / 2} y={28} fontSize="10.5" textAnchor="middle" style={{ fill: hit ? 'var(--page)' : 'var(--ink)' }}>{id}</text> : null}
          </g>
        );
      })}
      <line x1={X0 + d.u * SW} y1={6} x2={X0 + d.u * SW} y2={40} stroke="var(--ink)" strokeWidth="2" strokeDasharray="3 2" />
      <text x={Math.min(W - 30, Math.max(30, X0 + d.u * SW))} y={50} fontSize="10" textAnchor="middle">u = {d.u.toFixed(3)}</text>
    </svg>
  );
}

function StagePanels({ res, settings, testId }: { res: ChainResult; settings: ChainSettings; testId: string }) {
  return (
    <div data-part={testId} style={{ display: 'grid', gap: '0.5rem', gridTemplateColumns: 'repeat(auto-fill, minmax(13rem, 1fr))' }}>
      {res.snapshots.map((s, i) => {
        const ids = s.alive.flatMap((a, j) => (a ? [j] : []));
        const label = s.stage === 'input' ? 'input' : `${i}. ${stage(s.stage).label}`;
        const param = s.stage === 'input' ? '' : PARAM[s.stage]?.(settings) ?? '';
        return (
          <div key={s.stage} className="panel" style={{ padding: '0.45rem 0.5rem' }} data-stage={s.stage} data-kept={s.kept} data-ids={idsText(ids)}>
            <div className="ui" style={{ fontSize: '0.8rem', display: 'flex', justifyContent: 'space-between', gap: '0.4rem' }}>
              <b>{label}</b>
              <span className="muted">{param}</span>
            </div>
            <MiniBars probs={s.probs} alive={s.alive} testId={label} />
            <div className="ui" style={{ fontSize: '0.78rem' }}>
              {s.stage === 'temperature' && res.greedy ? 'greedy: argmax only' : `${s.kept} of ${NB} left`}
              {s.kept < NB ? <span className="muted">: {idsText(ids)}</span> : null}
            </div>
            {s.stage === 'min-p' && s.threshold != null && isFinite(s.threshold) ? (
              <div className="ui muted" style={{ fontSize: '0.74rem' }} data-threshold={s.threshold.toFixed(4)}>
                keeps logits ≥ {s.threshold.toFixed(2)} (largest {Math.max(...s.logits.filter((_, j) => s.alive[j])).toFixed(2)})
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------- the explorer
/** A comparison badge: heavier when the two orders differ. Neutral ink, not a failure colour. */
function Badge({ on, children }: { on: boolean; children: string }) {
  return (
    <span className="chip" data-badge={on ? 'differ' : 'same'}
      style={{ border: on ? '2px solid var(--ink)' : '1px solid var(--rule)', color: on ? 'var(--ink)' : 'var(--ink-2)', fontWeight: on ? 700 : 500 }}>
      {children}
    </span>
  );
}

interface Pinned {
  name: string;
  order: StageId[];
  opt: ChainOptions;
}

function ChainChips({ order, onMove, onDrop }: { order: StageId[]; onMove: (i: number, dir: -1 | 1) => void; onDrop: (from: number, to: number) => void }) {
  const [drag, setDrag] = useState<number | null>(null);
  const modelledIdx = order.flatMap((s, i) => (isModelled(s) ? [i] : []));
  return (
    <ol data-part="chips" aria-label="Order of the stages" style={{ listStyle: 'none', padding: 0, margin: '0 0 0.8rem', display: 'flex', flexWrap: 'wrap', gap: '0.35rem', alignItems: 'center' }}>
      {order.map((s, i) => {
        const m = isModelled(s);
        const pos = modelledIdx.indexOf(i);
        return (
          <li key={s} data-chip={s} data-modelled={m ? '1' : '0'}
            draggable={m}
            onDragStart={(e: DragEvent) => { setDrag(i); e.dataTransfer.effectAllowed = 'move'; }}
            onDragOver={(e: DragEvent) => { if (m && drag != null) e.preventDefault(); }}
            onDrop={(e: DragEvent) => { e.preventDefault(); if (drag != null && m) onDrop(drag, i); setDrag(null); }}
            style={{
              display: 'inline-flex', alignItems: 'center', gap: '0.15rem', fontFamily: 'var(--sans)', fontSize: '0.84rem',
              border: `1px ${m ? 'solid' : 'dashed'} ${m ? 'var(--ink-2)' : 'var(--ink-3)'}`, borderRadius: 3,
              padding: '0.1rem 0.25rem', color: m ? 'var(--ink)' : 'var(--ink-3)', background: m ? 'var(--page)' : 'transparent',
              cursor: m ? 'grab' : 'default',
            }}>
            {m ? (
              <button className="quiet" aria-label={`move ${s} earlier`} disabled={pos <= 0} onClick={() => onMove(i, -1)} style={{ padding: '0 0.3rem' }}>‹</button>
            ) : null}
            <span style={{ padding: '0 0.2rem' }}>{stage(s).label}{m ? '' : ' (not modelled)'}</span>
            {m ? (
              <button className="quiet" aria-label={`move ${s} later`} disabled={pos === modelledIdx.length - 1} onClick={() => onMove(i, 1)} style={{ padding: '0 0.3rem' }}>›</button>
            ) : null}
          </li>
        );
      })}
      <li data-chip="draw" style={{ fontFamily: 'var(--sans)', fontSize: '0.84rem', padding: '0.1rem 0.4rem', border: '1px solid var(--ink)', borderRadius: 3 }}>
        draw
      </li>
    </ol>
  );
}

function Slider({ label, value, min, max, step, onChange, disabled, note }: {
  label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; disabled?: boolean; note?: string;
}) {
  return (
    <label className="field" style={{ flex: '1 1 9.5rem', opacity: disabled ? 0.5 : 1 }}>
      {label}
      {note ? <span className="muted" style={{ fontSize: '0.75rem' }}>{note}</span> : null}
      <input type="range" min={min} max={max} step={step} value={value} aria-label={label.split(' =')[0]} disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))} style={{ width: '100%' }} />
    </label>
  );
}

function ReorderExplorer() {
  const [orderId, setOrderId] = useState<OrderId | 'custom'>('llama');
  const [order, setOrder] = useState<StageId[]>(ORDERS.llama.order);
  const [rowId, setRowId] = useState('peaked');
  const [s, setS] = useState<ChainSettings>(DEFAULT_SETTINGS);
  const [pinned, setPinned] = useState<Pinned | null>(null);
  const opt: ChainOptions = useMemo(() => (orderId === 'custom' ? M5_OPTIONS : optionsOf(ORDERS[orderId])), [orderId]);
  const row = chainRow(rowId);
  const res = useMemo(() => runChain(row, s, order, opt), [row, s, order, opt]);
  const pin = useMemo(() => (pinned ? runChain(row, s, pinned.order, pinned.opt) : null), [row, s, pinned]);
  const name = orderId === 'custom' ? 'your order' : ORDERS[orderId].label;
  const set = (patch: Partial<ChainSettings>) => setS((x) => ({ ...x, ...patch }));

  const pick = (id: OrderId) => {
    setOrderId(id);
    setOrder(ORDERS[id].order);
  };
  const custom = (next: StageId[]) => {
    setOrderId('custom');
    setOrder(next.filter(isModelled));
  };
  const move = (i: number, dir: -1 | 1) => {
    const m = order.flatMap((x, j) => (isModelled(x) ? [j] : []));
    const pos = m.indexOf(i);
    const j = m[pos + dir];
    if (j == null) return;
    const next = [...order];
    [next[i], next[j]] = [next[j], next[i]];
    custom(next);
  };
  const dropAt = (from: number, to: number) => {
    if (from === to) return;
    const next = [...order];
    const [x] = next.splice(from, 1);
    next.splice(to, 0, x);
    custom(next);
  };
  const loadPredict = () => {
    setRowId(P_ORDER.rowId);
    setS({ ...P_ORDER.settings });
    pick('hf');
    setPinned({ name: ORDERS.llama.label, order: ORDERS.llama.order, opt: optionsOf(ORDERS.llama) });
  };

  const fin = finalSet(res);
  const pinFin = pin ? finalSet(pin) : null;
  const sameSet = pinFin ? pinFin.join(',') === fin.join(',') : true;
  const sameTok = pin ? pin.token === res.token : true;
  const noFreq = !opt.hasFreqPresence;

  return (
    <Bench
      title="Reorder the chain"
      task="Pick an order, or move the stages yourself. Then move T and watch which panels change."
      foot={
        <>
          Bars are probabilities among the tokens still in play after each stage, scaled to the tallest in the panel; a dashed
          stub is a token already removed. The strip at the bottom is the draw: the survivors end to end, and the token u
          lands in. A custom order uses M5's rules: ties to the lowest id.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end', flexWrap: 'wrap' }}>
        <span style={FIELD}>
          Order
          <span className="seg" role="group" aria-label="Order preset" style={{ flexWrap: 'wrap' }}>
            {ORDER_IDS.map((id) => (
              <button key={id} aria-pressed={orderId === id} onClick={() => pick(id)}>{ORDERS[id].label}</button>
            ))}
          </span>
        </span>
        <span style={FIELD}>
          Row
          <span className="seg" role="group" aria-label="Row">
            {CHAIN_ROWS.map((r) => (
              <button key={r.id} aria-pressed={rowId === r.id} onClick={() => setRowId(r.id)}>{r.label}</button>
            ))}
          </span>
        </span>
        <button className="quiet" onClick={loadPredict}>Load the prediction's case</button>
      </div>
      <ChainChips order={order} onMove={move} onDrop={dropAt} />
      <div className="row" style={{ marginBottom: '0.6rem', flexWrap: 'wrap', gap: '0.6rem 1rem' }}>
        <Slider label={`T = ${fmtT(s.T)}`} value={sliderFromT(s.T)} min={0} max={1000} step={1} onChange={(v) => set({ T: tFromSlider(v) })} />
        <Slider label={`k = ${s.k}${s.k >= NB ? ' (off)' : ''}`} value={s.k} min={1} max={NB} step={1} onChange={(v) => set({ k: v })} />
        <Slider label={`p = ${s.p.toFixed(2)}${s.p >= 1 ? ' (off)' : ''}`} value={Math.round(s.p * 100)} min={5} max={100} step={1} onChange={(v) => set({ p: v / 100 })} />
        <Slider label={`min-p = ${s.minP.toFixed(2)}${s.minP <= 0 ? ' (off)' : ''}`} value={Math.round(s.minP * 100)} min={0} max={50} step={1} onChange={(v) => set({ minP: v / 100 })} />
      </div>
      <div className="row" style={{ marginBottom: '0.6rem', flexWrap: 'wrap', gap: '0.6rem 1rem' }}>
        <Slider label={`repetition = ${s.repeat.toFixed(2)}`} value={Math.round(s.repeat * 100)} min={100} max={200} step={5} onChange={(v) => set({ repeat: v / 100 })} />
        <Slider label={`frequency = ${s.freq.toFixed(2)}`} value={Math.round(s.freq * 100)} min={0} max={100} step={5} onChange={(v) => set({ freq: v / 100 })} disabled={noFreq} />
        <Slider label={`presence = ${s.presence.toFixed(2)}`} value={Math.round(s.presence * 100)} min={0} max={100} step={5} onChange={(v) => set({ presence: v / 100 })} disabled={noFreq} />
        <Slider label={`u = ${s.u.toFixed(3)}`} value={Math.round(s.u * 1000)} min={0} max={999} step={1} onChange={(v) => set({ u: v / 1000 })} />
        <label className="check" style={{ alignSelf: 'flex-end' }}>
          <input type="checkbox" checked={s.float32} onChange={(e) => set({ float32: e.target.checked })} /> float32
        </label>
      </div>
      {noFreq ? (
        <p className="muted ui" style={{ margin: '0 0 0.6rem', fontSize: '0.8rem' }} data-readout="no-freq">
          transformers has a repetition penalty and no frequency or presence penalty, so those two are ignored in its order.
        </p>
      ) : null}
      <p className="ui" style={{ margin: '0 0 0.4rem', fontSize: '0.85rem' }}><b>{name}</b> <span className="muted">context: {idsText(row.context as number[])}</span></p>
      <StagePanels res={res} settings={s} testId="panels" />
      <Strip res={res} label={name} />
      <p style={{ margin: '0.4rem 0 0.2rem' }} data-readout="result" data-final={idsText(fin)} data-token={res.token}>
        {res.greedy ? (
          <>T ≤ 0: greedy. The token is <b>{res.token}</b>.</>
        ) : (
          <>
            {name} draws from <b>{fin.length}</b> token{fin.length === 1 ? '' : 's'} ({idsText(fin)}); u = {s.u.toFixed(3)} lands in token <b>{res.token}</b>.
          </>
        )}
      </p>
      <div className="row" style={{ marginTop: '0.6rem', flexWrap: 'wrap' }}>
        <button onClick={() => setPinned({ name, order: [...order], opt })}>Pin this order for comparison</button>
        {pinned ? <button className="quiet" onClick={() => setPinned(null)}>Unpin</button> : null}
      </div>
      {pin && pinned ? (
        <div style={{ marginTop: '0.8rem' }} data-part="pinned">
          <p className="ui" style={{ margin: '0 0 0.4rem', fontSize: '0.85rem' }}>
            <b>pinned: {pinned.name}</b>{' '}
            <span className="muted">{pinned.order.filter(isModelled).map((x) => stage(x).label).join(' → ')} → draw</span>
          </p>
          <StagePanels res={pin} settings={s} testId="pinned-panels" />
          <Strip res={pin} label={pinned.name} />
          <p style={{ margin: '0.4rem 0 0' }} data-readout="compare" data-same-set={sameSet ? '1' : '0'} data-same-token={sameTok ? '1' : '0'}>
            <Badge on={!sameSet}>{sameSet ? 'same set' : 'different sets'}</Badge>{' '}
            <Badge on={!sameTok}>{sameTok ? 'same token' : 'different tokens'}</Badge>{' '}
            {pinned.name} draws from {pinFin!.length} ({idsText(pinFin!)}) and picks {pin.token}; {name} draws from {fin.length} and picks {res.token}.
          </p>
        </div>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the M5 contract
const CONTRACT: [string, string][] = [
  ['Order', `${M5_ORDER.map((x) => stage(x).label).join(' → ')} → draw. llama.cpp's default restricted to the stages M5 builds.`],
  ['Greedy', 'T ≤ 0 means no draw: lab 10.3’s argmax over the survivors.'],
  ['Ties', 'The lowest token id wins every tie, at every stage: the top-k cut, the stable sort, top-p’s position rule, greedy.'],
  ['Inputs', 'logits [rows × vocab] float32, read and not modified; token counts [rows × vocab] uint32, from lab 9.3; per row a seed, a stream id and a step (X1.7 explains why per row); the scalars k, p, min-p, T, repetition, frequency and presence.'],
  ['Output', 'tokens [rows] int32.'],
];

// ---------------------------------------------------------------- the section
export function SX11() {
  const q = questionsSX11;
  const counts = pOrderCounts();
  return (
    <>
      <p>
        Between the logits and the token there are usually five or six small operations, and every project chains them a
        little differently. This section names each one, shows which pairs of them interact, and fixes the order and the
        tie rule your milestone M5 will implement.
      </p>
      <NoteGroup>
        <Beyond title="Ours, not the book's">
          <p>
            Nothing on this page is in the book. The logits it starts from are the output of the book's un-embedding layer.
          </p>
        </Beyond>
        <Source section="20.1" pages="481" />
      </NoteGroup>

      <h2>Two kinds of stage, then a draw</h2>
      <p>
        A sampler is a chain of row-wise operations. Some <b>reshape</b> the logits without changing which tokens are in
        play: the penalties (lowering tokens that already appeared) and temperature (dividing every logit by T). Others{' '}
        <b>truncate</b>: top-k keeps the k largest, top-p keeps the smallest set of most likely tokens whose probability
        reaches p, and min-p keeps every token whose probability is at least min-p times the top one. Last, one{' '}
        <b>draw</b> picks a token from what is left, in proportion to its probability.
      </p>
      <p>
        Every stage reads a row and writes a row, or a shorter list. None does much arithmetic per byte, so each one is
        memory-bound, and each is written as one block per row (<a href={href('ch10', '10.10')}>section 10.10</a> has the
        cost of that at batch 1).
      </p>

      <h2>The orders in the code you know</h2>
      <table className="data" data-part="orders" style={{ maxWidth: '44rem' }}>
        <thead>
          <tr><th>Project</th><th>Order</th><th>Defaults</th></tr>
        </thead>
        <tbody>
          <tr data-order="llama">
            <td>llama.cpp</td>
            <td>{ORDERS.llama.order.map((x) => stage(x).label).join(', ')}, then dist</td>
            <td>top_k 40, top_p 0.95, min_p 0.05, temp 0.8; penalties off</td>
          </tr>
          <tr data-order="hf">
            <td>transformers</td>
            <td>repetition penalty and other processors, temperature, top-k, top-p, min-p, a few more warpers, then <code>torch.multinomial</code></td>
            <td>when unset: temperature 1.0, top_k 50, top_p 1.0</td>
          </tr>
          <tr data-order="mlx">
            <td>mlx-lm</td>
            <td>top-p, min-p, XTC, top-k, then temperature inside <code>categorical</code></td>
            <td>every filter off; temp 0, which means argmax</td>
          </tr>
        </tbody>
      </table>
      <p>
        transformers is the odd one out: it applies temperature before every truncation. llama.cpp and mlx-lm apply it
        last, after the set is decided. The min-p paper itself applies min-p after temperature, which transformers follows
        and llama.cpp's default chain does not. mlx-lm also runs top-p before top-k, unlike both. llama.cpp's chain has
        four stages this course does not build (DRY, top-n-sigma, typical and XTC); the explorer shows them greyed.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${FACTS}, D1, D4, D8, D9)`}>
        <p>
          llama.cpp <code>common/common.h</code>: <code>top_k = 40</code>, <code>top_p = 0.95f</code>,{' '}
          <code>min_p = 0.05f</code>, <code>temp = 0.80f</code>, <code>penalty_repeat = 1.00f</code>. transformers{' '}
          <code>_get_logits_processor</code>: temperature, then top-k, top-p, min-p ("Applied after temperature
          scaling"). mlx-lm <code>make_sampler(temp=0.0, top_p=0.0, min_p=0.0, ..., top_k=0)</code>.
        </p>
      </Today>

      <h2>When the order matters</h2>
      <p>
        Three facts decide it, and you can reason them out before touching anything. First, dividing by a positive T keeps
        the logits in the same order, so <b>top-k's set never depends on where temperature sits</b>. Second, top-p and
        min-p look at probabilities, and the ratio of two probabilities is exp((l<sub>a</sub> − l<sub>b</sub>) / T), so{' '}
        <b>their sets do depend on T</b> when temperature runs before them. Third, a truncation that runs after another one
        measures probability among the survivors only. llama.cpp's top-p takes a softmax over the current candidates, and
        transformers' top-k leaves minus infinity in the removed slots, which softmax turns into 0. Either way{' '}
        <b>top-p after top-k sums mass over the k survivors</b>, renormalised.
      </p>
      <Today checked={`2026-11-20 in the facts pass (notes/today/ch11-facts.md D2; ${FACTS}, D4)`}>
        <p>
          llama.cpp host top-p: softmax, sort, <code>cum_sum += pdata[i].p</code>. transformers top-k:{' '}
          <code>scores &lt; torch.topk(scores, top_k)[0][..., -1, None]</code> set to minus infinity.
        </p>
      </Today>
      <Ask q={q['x1/X1.1/q-temp-first']} />
      <Ask q={q['x1/X1.1/q-renorm']} />

      <h2>Watch one row go through</h2>
      <p>
        Before you change anything, watch the order once. The animation runs one twelve-token row through M5's chain with
        the settings it names; every bar is computed by the same model the explorer below uses.
      </p>
      <Simplification>
        <p>
          One row, double precision. The u of the last part is one Philox word for seed 0, row 0, step 0 (X1.7 builds the
          generator). The stages M5 does not build are left out.
        </p>
      </Simplification>
      <ChainRowScene />

      <h2>Reorder the chain</h2>
      <Predict
        id="x1/X1.1/p-order"
        unit="tokens"
        prompt={
          <>
            In the explorer below, take the row "three favourites, long tail" with k = 8, p = 0.9, T = 2, and min-p and the
            penalties off. How many tokens can transformers' order still draw from?
          </>
        }
      >
        {(c) => (
          <div data-reveal="order">
            <p>
              transformers' order keeps <b>{counts.hf}</b>
              {c === counts.hf ? ', as you said' : ''}; llama.cpp's keeps <b>{counts.llama}</b>. Both cut to the same 8 at
              top-k. The difference is what top-p sees: transformers' top-p sees the row at T = 2, flattened, and needs more
              tokens to reach 0.9; llama.cpp's sees it at T = 1. Press "Load the prediction's case" in the explorer to see
              both side by side, then move T: llama.cpp's count does not move at all.
            </p>
          </div>
        )}
      </Predict>
      <p>
        The explorer runs one sixteen-token row, with a context that repeats tokens 3 and 7, through any order of the five
        stages M5 builds. Pick a project's order or move the stages yourself, set the parameters, and pin one order to
        compare it with another.
      </p>
      <Simplification>
        <p>
          One row of 16 tokens. Double precision unless float32 is on; in double, top-p compares probabilities rounded to
          six decimals. Every project's top-p is modelled with the rule they share (mass before below p); mlx-lm's extra
          rule for ties is not. transformers' min-p compares probabilities, the model compares logits (the same set up to
          rounding). DRY, top-n-sigma, typical and XTC are shown, not run.
        </p>
      </Simplification>
      <ReorderExplorer />
      <p>
        Four things to find. Move T under any order and watch the top-k panel: its set never changes. Do the same under
        transformers' order and the top-p and min-p panels change, with more tokens above T = 1 and fewer below. Compare
        llama.cpp's order with mlx-lm's, which runs top-p before top-k: top-p over the whole row keeps more than top-p over
        the renormalised survivors. And move u: it only changes which survivor is drawn, never the set.
      </p>
      <Ask q={q['x1/X1.1/q-hf-more']} />

      <h2>Greedy is a different ending</h2>
      <p>
        At T = 0 there is nothing to draw. llama.cpp's common sampler swaps the final <code>dist</code> step for a greedy
        one, transformers refuses temperature 0 and asks for <code>do_sample=False</code>, and mlx-lm returns the argmax.
        M5 does what llama.cpp does: at T ≤ 0 the chain ends with lab 10.3's argmax over the survivors.
      </p>
      <Today checked={`2026-11-20 in the facts pass (${FACTS}, D1, D4, D8)`}>
        <p>
          llama.cpp: <code>llama_sampler_init_greedy()</code> when the last sampler is temperature with{' '}
          <code>temp == 0.0f</code>. transformers: "If you're looking for greedy decoding strategies, set{' '}
          <code>do_sample=False</code>." mlx-lm: <code>if temp == 0: return greedy_sampler</code>.
        </p>
      </Today>
      <Ask q={q['x1/X1.1/q-greedy-switch']} />

      <h2>Milestone M5's contract</h2>
      <p>
        This is the chain lab X1.3 implements. The choices are ours, and each has a reason. llama.cpp's order is what its
        users get without asking, and its rules were checked in the facts pass. Temperature last makes every
        truncation independent of T, so the kept set can be tested without it. And one tie rule, applied everywhere, is
        the only way two backends can agree on a row with equal logits.
      </p>
      <table className="data" data-part="contract" style={{ maxWidth: '44rem' }}>
        <tbody>
          {CONTRACT.map(([k, v]) => (
            <tr key={k}><th style={{ whiteSpace: 'nowrap' }}>{k}</th><td>{v}</td></tr>
          ))}
        </tbody>
      </table>
      <p>
        Where each stage runs: one block per row everywhere, and the batch is the grid. Only the penalties and top-k read
        the whole vocabulary. After top-k, every later stage touches k values, not 151,936, which is why the
        selection gets its own section (<a href={href('x1', 'X1.4')}>X1.4</a>) and why top-p over a sorted row is cheap (
        <a href={href('x1', 'X1.5')}>X1.5</a>).
      </p>
      <Aside>
        <p>
          With llama.cpp's default k = 40 the softmax, the scan and the draw each work on 40 numbers per row: most threads of a block have
          nothing to do. That is fine; those stages are a small part of the step, and lab X1.3 measures each stage.
        </p>
      </Aside>
    </>
  );
}
