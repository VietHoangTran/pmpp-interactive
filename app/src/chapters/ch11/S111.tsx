import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  ALL_PRESETS,
  BOOK_ORDERS,
  FLIP,
  ORDER_LABEL,
  RULES,
  RULE_FACT,
  RULE_LABEL,
  RULE_WHO,
  SAUSAGE_INCHES,
  TOKEN_COUNTS,
  band,
  exclusive,
  exclusiveIn,
  f32,
  f32hex,
  f32short,
  inclusive,
  offsets,
  sequentialAdditions,
  toExclusive,
  toInclusive,
  topP,
  ulpSigned,
  type Arith,
  type Order,
} from '../../sim/scan';

/** A caption above a group of buttons (a <label> would forward clicks on the caption to the first button). */
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const WRAP: CSSProperties = { flexWrap: 'wrap' };

// ---------------------------------------------------------------- cut the sausage
const SIZE_PRESETS = [
  { id: 'book', label: "the book's eight orders", values: BOOK_ORDERS },
  { id: 'tokens', label: 'token counts of six sequences', values: TOKEN_COUNTS },
];
function parseSizes(s: string): number[] | null {
  const parts = s.split(/[\s,]+/).filter(Boolean);
  if (parts.length < 1 || parts.length > 12) return null;
  const v = parts.map(Number);
  return v.every((x) => Number.isInteger(x) && x >= 0 && x <= 99) ? v : null;
}

function SausageExplorer() {
  const [values, setValues] = useState<number[]>(BOOK_ORDERS);
  const [text, setText] = useState(BOOK_ORDERS.join(' '));
  const [mode, setMode] = useState<'inclusive' | 'exclusive'>('inclusive');
  const [convert, setConvert] = useState(false);
  const inc = useMemo(() => inclusive(values), [values]);
  const exc = useMemo(() => exclusive(values), [values]);
  const off = useMemo(() => offsets(values), [values]);
  const isBook = values.join(' ') === BOOK_ORDERS.join(' ');
  const isTokens = values.join(' ') === TOKEN_COUNTS.join(' ');
  const bad = parseSizes(text) === null;
  const N = values.length;
  const length = isBook ? SAUSAGE_INCHES : Math.max(1, off.total);
  const W = 640;
  const X0 = 20;
  const scale = (W - 2 * X0) / length;
  const pos = (v: number) => X0 + v * scale;
  const marks = mode === 'inclusive' ? inc : exc;
  // Equal cut points (an order of 0) share one mark, labelled with how many cuts it is.
  const times = marks.map((m) => marks.filter((x) => x === m).length);
  const firstOf = marks.map((m, i) => marks.indexOf(m) === i);
  const unit = isBook ? 'inches' : isTokens ? 'slots' : 'units';
  const derived = mode === 'inclusive' ? toExclusive(inc) : toInclusive(exc, values);

  return (
    <Bench
      title="Cut the sausage"
      task="Switch between inclusive and exclusive. Then change the orders and watch which cuts move."
      foot={
        <>
          Every cut point comes from the scan in one go, so all pieces can be cut at once, in any order. The scans are
          exact here: sizes are whole numbers.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Scan
          <span className="seg" role="group" aria-label="Scan">
            <button aria-pressed={mode === 'inclusive'} onClick={() => setMode('inclusive')}>inclusive: where pieces end</button>
            <button aria-pressed={mode === 'exclusive'} onClick={() => setMode('exclusive')}>exclusive: where pieces start</button>
          </span>
        </span>
        <label className="check">
          <input type="checkbox" checked={convert} onChange={(e) => setConvert(e.target.checked)} /> show the conversion
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ flex: '1 1 16rem' }}>
          Sizes, in order (1 to 12 whole numbers)
          <input
            type="text"
            value={text}
            aria-label="Sizes"
            aria-invalid={bad}
            style={{ width: '100%', fontVariantNumeric: 'tabular-nums', borderColor: bad ? 'var(--fail)' : undefined }}
            onChange={(e) => {
              setText(e.target.value);
              const v = parseSizes(e.target.value);
              if (v) setValues(v);
            }}
          />
        </label>
        <span className="seg" role="group" aria-label="Presets" style={WRAP}>
          {SIZE_PRESETS.map((p) => (
            <button key={p.id} onClick={() => { setValues(p.values); setText(p.values.join(' ')); }}>{p.label}</button>
          ))}
        </span>
      </div>
      {bad ? <p className="t-fail" style={{ marginTop: 0 }}>Type 1 to 12 whole numbers from 0 to 99. The explorer keeps the last list that parsed.</p> : null}

      <svg viewBox={`0 0 ${W} 112`} role="img" aria-label="The pieces laid end to end, with the cut points" data-part="sausage" data-mode={mode}
        style={{ width: '100%', maxWidth: W, minWidth: 520, display: 'block' }}>
        {isBook ? (
          <g data-part="left">
            <rect x={pos(off.total)} y={56} width={pos(SAUSAGE_INCHES) - pos(off.total)} height={24} rx={11} fill="none" stroke="var(--ink-3)" strokeDasharray="4 3" />
            <text x={(pos(off.total) + pos(SAUSAGE_INCHES)) / 2} y={72} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
              {SAUSAGE_INCHES - off.total} inches left
            </text>
          </g>
        ) : null}
        {values.map((v, i) =>
          v > 0 ? (
            <g key={i} data-part="piece" data-index={i} data-start={exc[i]} data-end={inc[i]}>
              <rect x={pos(exc[i])} y={56} width={v * scale} height={24} fill={i % 2 ? 'var(--rule-soft)' : 'var(--cell)'} stroke="var(--ink-2)" />
              {v * scale > 16 ? <text x={pos(exc[i]) + (v * scale) / 2} y={72} fontSize="11" textAnchor="middle">{i}</text> : null}
            </g>
          ) : null,
        )}
        {marks.map((m, i) => (
          <g key={i} data-part="cut" data-index={i} data-at={m}>
            <line x1={pos(m)} y1={48} x2={pos(m)} y2={86} stroke="var(--ink)" strokeWidth={1.6} />
            {firstOf[i] ? (
              <text x={pos(m)} y={42} fontSize="11" textAnchor="middle" fontWeight="600">{times[i] > 1 ? `${m} ×${times[i]}` : m}</text>
            ) : null}
          </g>
        ))}
        <text x={X0} y={104} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
          {mode === 'inclusive' ? 'Cut at the end of every piece: the inclusive scan.' : 'Mark the start of every piece: the exclusive scan.'}
        </text>
      </svg>

      <div className="panel" style={{ overflowX: 'auto', marginTop: '0.6rem' }}>
        <table className="data" style={{ margin: 0 }} data-part="scan-table">
          <tbody>
            <tr>
              <th>i</th>
              {values.map((_, i) => <td key={i} className="num muted">{i}</td>)}
            </tr>
            <tr data-row="input">
              <th>size</th>
              {values.map((v, i) => <td key={i} className="num">{v}</td>)}
            </tr>
            <tr data-row="inclusive" style={{ fontWeight: mode === 'inclusive' ? 650 : 400 }}>
              <th>inclusive</th>
              {inc.map((v, i) => <td key={i} className="num">{v}</td>)}
            </tr>
            <tr data-row="exclusive" style={{ fontWeight: mode === 'exclusive' ? 650 : 400 }}>
              <th>exclusive</th>
              {exc.map((v, i) => <td key={i} className="num">{v}</td>)}
            </tr>
          </tbody>
        </table>
      </div>

      {convert ? (
        <div className="panel" style={{ marginTop: '0.6rem' }} data-part="conversion" data-from={mode}>
          {mode === 'inclusive' ? (
            <p style={{ marginTop: 0 }}>
              From inclusive to exclusive: shift every value one place to the right and put the identity, 0, in slot 0.
              The last inclusive value, {inc[N - 1]}, falls off the end.
            </p>
          ) : (
            <p style={{ marginTop: 0 }}>
              From exclusive to inclusive: shift every value one place to the left, and fill the last slot with the last
              exclusive value plus the last size: {exc[N - 1]} + {values[N - 1]} = {exc[N - 1] + values[N - 1]}.
            </p>
          )}
          <svg viewBox={`0 0 ${40 + N * 44} 92`} role="img" aria-label="The conversion as a one-place shift" style={{ width: '100%', maxWidth: 40 + N * 44, display: 'block' }}>
            {(mode === 'inclusive' ? [inc, derived] : [exc, derived]).map((row, r) =>
              row.map((v, i) => (
                <g key={`${r}-${i}`} data-part={r ? 'derived' : 'source'} data-index={i} data-value={v}>
                  <rect x={30 + i * 44} y={4 + r * 56} width={38} height={26} rx={2} fill="var(--cell)"
                    stroke={(mode === 'inclusive' && r === 1 && i === 0) || (mode === 'exclusive' && r === 1 && i === N - 1) ? 'var(--ink)' : 'var(--rule)'}
                    strokeWidth={(mode === 'inclusive' && r === 1 && i === 0) || (mode === 'exclusive' && r === 1 && i === N - 1) ? 2 : 1} />
                  <text x={49 + i * 44} y={21 + r * 56} fontSize="12" textAnchor="middle">{v}</text>
                </g>
              )),
            )}
            {Array.from({ length: N - 1 }, (_, i) => {
              const from = mode === 'inclusive' ? i : i + 1;
              const to = mode === 'inclusive' ? i + 1 : i;
              return <line key={i} x1={49 + from * 44} y1={31} x2={49 + to * 44} y2={59} stroke="var(--ink-3)" />;
            })}
            <text x={4} y={21} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{mode === 'inclusive' ? 'inc' : 'exc'}</text>
            <text x={4} y={77} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{mode === 'inclusive' ? 'exc' : 'inc'}</text>
          </svg>
        </div>
      ) : null}

      <p style={{ marginBottom: 0 }} data-readout="sausage">
        {isTokens ? (
          <>
            Sequence <i>k</i> occupies slots exclusive[<i>k</i>] up to inclusive[<i>k</i>] − 1: sequence 4 starts at slot{' '}
            <b>{exc[4]}</b>. The packed buffer needs <b>{off.total}</b> slots, the last inclusive value. Sequence 2 has no
            tokens, so it starts and ends where sequence 3 starts.
          </>
        ) : (
          <>
            Piece <i>i</i> runs from exclusive[<i>i</i>] to inclusive[<i>i</i>]. In all, <b>{off.total}</b> {unit}
            {isBook ? <> of the {SAUSAGE_INCHES}-inch sausage are served and {SAUSAGE_INCHES - off.total} are left</> : null}.
            The loop that computes either row does {sequentialAdditions(N)} additions.
          </>
        )}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- where the nucleus ends
function NucleusExplorer() {
  const [presetId, setPresetId] = useState('peaked');
  const preset = ALL_PRESETS.find((p) => p.id === presetId)!;
  const [p, setP] = useState(preset.p);
  const [float, setFloat] = useState(false);
  const [order, setOrder] = useState<Order>('loop');
  const probs = preset.probs;
  const arith: Arith = float ? { kind: 'f32', order } : { kind: 'exact' };
  const results = useMemo(() => RULES.map((r) => topP(probs, p, r, arith)), [probs, p, float, order]); // eslint-disable-line react-hooks/exhaustive-deps
  const base = results[0];
  const n = probs.length;
  const crossing = base.before.findIndex((b, i) => b < (float ? f32(p) : p) && base.upto[i] >= (float ? f32(p) : p));
  const loopB = useMemo(() => exclusiveIn(probs, 'loop'), [probs]);
  const ksB = useMemo(() => exclusiveIn(probs, 'kogge-stone'), [probs]);
  const bd = useMemo(() => band(probs, p), [probs, p]);
  const pf = f32(p);
  const disagree = loopB.map((v, i) => v < pf !== ksB[i] < pf);
  const W = 640;
  const X0 = 16;
  const sx = (v: number) => X0 + v * (W - 2 * X0);
  const pick = (id: string) => {
    const pr = ALL_PRESETS.find((x) => x.id === id)!;
    setPresetId(id);
    setP(pr.p);
    if (id === 'flip') setFloat(true);
  };
  const fmt = (v: number) => (float ? f32short(v) : String(Number(v.toFixed(6))));
  /** Float steps from p, written out only when they are few enough to mean something. */
  const steps = (u: number) => (Math.abs(u) <= 999 ? String(u) : u < 0 ? 'far below' : 'far above');
  const total = n ? (float ? f32(base.before[n - 1] + f32(probs[n - 1])) : base.upto[n - 1]) : 0;

  return (
    <Bench
      title="Where the nucleus ends"
      task="Move p until the crossing token changes. Then compare the four rules, and switch to float32."
      foot={
        <>
          Each token's piece runs from the mass before it to the mass up to and including it. A token is kept by the
          projects' rule when its piece starts before p. Exact mode uses decimal arithmetic on the probabilities as
          written; float32 rounds every addition, in the order you choose.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Probabilities, sorted from the top token down
          <span className="seg" role="group" aria-label="Rows" style={WRAP}>
            {ALL_PRESETS.map((pr) => (
              <button key={pr.id} aria-pressed={presetId === pr.id} onClick={() => pick(pr.id)}>{pr.label}</button>
            ))}
          </span>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ flex: '1 1 14rem' }}>
          p = {p.toFixed(2)}
          <input type="range" min="1" max="99" step="1" value={Math.round(p * 100)} aria-label="p"
            onChange={(e) => setP(Number(e.target.value) / 100)} style={{ width: '100%' }} />
        </label>
        <span style={FIELD}>
          Arithmetic
          <span className="seg" role="group" aria-label="Arithmetic">
            <button aria-pressed={!float} onClick={() => setFloat(false)}>exact</button>
            <button aria-pressed={float} onClick={() => setFloat(true)}>float32</button>
          </span>
        </span>
        {float ? (
          <span style={FIELD}>
            Order of the additions
            <span className="seg" role="group" aria-label="Order">
              {(['loop', 'kogge-stone'] as Order[]).map((o) => (
                <button key={o} aria-pressed={order === o} onClick={() => setOrder(o)}>{ORDER_LABEL[o]}</button>
              ))}
            </span>
          </span>
        ) : null}
      </div>

      <svg viewBox={`0 0 ${W} 86`} role="img" aria-label="The tokens' pieces of the unit interval, and p" data-part="mass"
        style={{ width: '100%', maxWidth: W, minWidth: 520, display: 'block' }}>
        {probs.map((v, i) =>
          v > 0 ? (
            <g key={i} data-part="token" data-index={i}>
              <rect x={sx(base.before[i])} y={30} width={Math.max(0, sx(base.upto[i]) - sx(base.before[i]))} height={24}
                fill={base.keep[i] ? 'var(--ink-3)' : 'var(--cell)'} fillOpacity={base.keep[i] ? 0.35 : 1}
                stroke={i === crossing ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={i === crossing ? 2.2 : 1} />
              {sx(base.upto[i]) - sx(base.before[i]) > 18 ? (
                <text x={(sx(base.before[i]) + sx(base.upto[i])) / 2} y={46} fontSize="11" textAnchor="middle">{i}</text>
              ) : null}
            </g>
          ) : null,
        )}
        <line x1={sx(p)} y1={18} x2={sx(p)} y2={64} stroke="var(--ink)" strokeWidth={2} />
        <text x={sx(p)} y={13} fontSize="12" textAnchor="middle" fontWeight="650">p = {p.toFixed(2)}</text>
        <text x={X0} y={80} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>0</text>
        <text x={W - X0} y={80} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>1</text>
        <text x={W / 2} y={80} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
          {crossing >= 0 ? `token ${crossing} crosses p` : 'no token crosses p: p falls exactly on a cut'}
        </text>
      </svg>

      <div className="panel" style={{ overflowX: 'auto', marginTop: '0.6rem' }}>
        <table className="data" style={{ margin: 0 }} data-part="masses">
          <tbody>
            <tr><th>token</th>{probs.map((_, i) => <td key={i} className="num muted">{i}</td>)}</tr>
            <tr><th>probability</th>{probs.map((v, i) => <td key={i} className="num">{fmt(float ? f32(v) : v)}</td>)}</tr>
            <tr data-row="before"><th>mass before (exclusive)</th>{base.before.map((v, i) => <td key={i} className="num">{fmt(v)}</td>)}</tr>
            <tr data-row="upto"><th>mass up to (inclusive)</th>{base.upto.map((v, i) => <td key={i} className="num">{fmt(v)}</td>)}</tr>
          </tbody>
        </table>
      </div>

      <div style={{ overflowX: 'auto', marginTop: '0.8rem' }}>
      <table className="data" style={{ margin: 0, background: 'var(--page)', minWidth: '36rem' }} data-part="rules">
        <thead>
          <tr><th style={{ paddingLeft: '0.6rem' }}>Rule</th><th>Who computes it this way</th><th>Tokens kept</th><th className="num">Count</th></tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <tr key={r.rule} data-rule={r.rule} data-kept={r.kept}>
              <td style={{ paddingLeft: '0.6rem' }}>{RULE_LABEL[r.rule]}</td>
              <td className="muted">{RULE_WHO[r.rule]} <span className="nowrap">({RULE_FACT[r.rule]})</span></td>
              <td>
                <span style={{ display: 'inline-flex', gap: 3, flexWrap: 'wrap' }}>
                  {r.keep.map((k, i) => (
                    <span key={i} data-token={i} data-keep={k ? '1' : '0'} className="mono"
                      style={{ display: 'inline-block', minWidth: '1.5rem', textAlign: 'center', padding: '0 0.2rem', borderRadius: 3,
                        border: `1px ${k ? 'solid' : 'dashed'} var(--ink-3)`, background: k ? 'var(--ink)' : 'transparent', color: k ? 'var(--page)' : 'var(--ink-3)' }}>
                      {i}
                    </span>
                  ))}
                </span>
              </td>
              <td className="num"><b>{r.kept}</b></td>
            </tr>
          ))}
        </tbody>
      </table>
      </div>

      {float ? (
        <div className="panel" style={{ overflowX: 'auto', marginTop: '0.8rem' }}>
          <p style={{ marginTop: 0 }}>
            The mass before each token in float32, summed in two orders, against p = {f32short(pf)} ({f32hex(pf)}). The
            last column is what a test may demand of the projects' rule: inside the rounding band either answer is
            right.
          </p>
          <table className="data" style={{ margin: 0 }} data-part="float">
            <thead>
              <tr>
                <th>token</th><th>loop order</th><th className="num">steps from p</th><th>keeps?</th>
                <th>Kogge-Stone order</th><th className="num">steps from p</th><th>keeps?</th><th>a test accepts</th>
              </tr>
            </thead>
            <tbody>
              {probs.map((_, i) => (
                <tr key={i} data-token={i} data-disagree={disagree[i] ? '1' : '0'} style={disagree[i] ? { fontWeight: 650 } : undefined}>
                  <td className="num">{i}</td>
                  <td className="mono">{f32short(loopB[i])} <span className="muted">{f32hex(loopB[i])}</span></td>
                  <td className="num">{steps(ulpSigned(loopB[i], pf))}</td>
                  <td>{loopB[i] < pf ? 'keep' : 'drop'}</td>
                  <td className="mono">{f32short(ksB[i])} <span className="muted">{f32hex(ksB[i])}</span></td>
                  <td className="num">{steps(ulpSigned(ksB[i], pf))}</td>
                  <td>{ksB[i] < pf ? 'keep' : 'drop'}</td>
                  <td data-verdict={bd.verdict[i]}>{bd.verdict[i]}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p style={{ marginBottom: 0 }} data-readout="orders">
            {disagree.some(Boolean) ? (
              <>
                The two orders disagree on token {disagree.indexOf(true)}: its mass before is{' '}
                {ulpSigned(loopB[disagree.indexOf(true)], pf)} steps from p in loop order and{' '}
                {ulpSigned(ksB[disagree.indexOf(true)], pf)} in Kogge-Stone order, so one keeps it and the other drops it.
                Exactly, the mass before it is {topP(probs, p, 'exclusive').before[disagree.indexOf(true)]}: precisely p.
              </>
            ) : (
              <>The two orders agree on every token here. The rounding band is ±{bd.bound.toExponential(1)}.</>
            )}
          </p>
        </div>
      ) : null}
      <p style={{ marginBottom: 0, marginTop: '0.6rem' }} data-readout="nucleus">
        At p = {p.toFixed(2)} the projects' rule keeps <b>{results[0].kept}</b> of {n} tokens; the off-by-one keeps{' '}
        <b>{results[1].kept}</b>
        {results[1].kept === 0 ? ', none at all' : ''}.
        {results[2].kept > results[0].kept ? <> mlx-lm keeps <b>{results[2].kept}</b>, because it keeps every token tied with the boundary token.</> : null}
        {results[2].kept < results[0].kept ? <> mlx-lm keeps <b>{results[2].kept}</b>: it compares with p times the total it computed, {fmt(total)}, which puts its threshold just below p.</> : null}
        {results[3].kept !== results[0].kept ? <> The ascending form keeps <b>{results[3].kept}</b>: it compares a different sum, with 1 − p, and its rounding lands differently.</> : null}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS111: Record<string, Question> = {
  'ch11/11.1/q-rule': {
    kind: 'numeric',
    id: 'ch11/11.1/q-rule',
    prompt: (
      <>
        Sorted probabilities 0.4, 0.3, 0.2, 0.1 and p = 0.6. How many tokens does the rule "keep a token while the
        inclusive sum up to it is below p" keep?
      </>
    ),
    answer: 1,
    unit: 'tokens',
    explain: (
      <p>
        The inclusive sums are 0.4, 0.7, 0.9, 1.0, and only 0.4 is below 0.6. The projects keep 2: token 1's mass before
        it is 0.4, below p, so it is kept although its own probability carries the total past p. Keeping the token that
        crosses p is the whole point of the nucleus; comparing the inclusive sum drops it. That is the off-by-one
        between the two scans.
      </p>
    ),
    source: 'ch. 11 §11.1, pp. 252–253; projects: notes/today/ch11-facts.md D2–D4',
  },
  'ch11/11.1/q-total': {
    kind: 'choice',
    id: 'ch11/11.1/q-total',
    prompt: <>E is the exclusive scan of an array x of N elements. The sum of all N elements of x is:</>,
    options: [
      { label: 'E[N−1]', why: 'E[N−1] stops before x[N−1].' },
      { label: 'E[N−1] + x[N−1]', correct: true },
      { label: 'E[0] + x[N−1]', why: 'E[0] is the identity, 0.' },
      { label: 'x[N−1]', why: 'That is only the last element.' },
    ],
    explain: (
      <p>
        The exclusive scan stops one input short at every position, including the last. Adding the last input back gives
        the total, which is also how p. 253 fills the last slot when it converts an exclusive scan into an inclusive one.
        In allocation it is the size of the buffer you need.
      </p>
    ),
    source: 'ch. 11 §11.1, p. 253',
  },
  'ch11/11.1/q-offsets': {
    kind: 'numeric',
    id: 'ch11/11.1/q-offsets',
    prompt: (
      <>
        Six sequences of 5, 3, 0, 7, 2 and 4 tokens are packed into one buffer, in this order. At which slot does sequence 4
        (counting from 0) start?
      </>
    ),
    answer: 15,
    explain: (
      <p>
        The exclusive scan of the counts is 0, 5, 8, 8, 15, 17: sequence 4 starts at 15. The buffer needs 17 + 4 = 21
        slots. Start offsets are exclusive scans, which is why the book calls the exclusive form the one memory
        allocation needs.
      </p>
    ),
    source: 'ch. 11 §11.1, p. 253 (the use for token counts is ours)',
  },
  'ch11/11.1/q-float-boundary': {
    kind: 'choice',
    id: 'ch11/11.1/q-float-boundary',
    prompt: (
      <>
        A GPU kernel and a CPU loop compute the exclusive scan of the same float32 row in different orders. For one
        token, the CPU's mass before it is exactly p. What may the GPU's keep mask say about that token?
      </>
    ),
    options: [
      {
        label: 'It must drop it, like the CPU: both compute the same exact mass.',
        why: "Neither computes the exact mass: each rounds its own float sum, and the GPU's may land just below p, so it keeps the token.",
      },
      {
        label: 'It must keep it: a tree adds the small values first, so its sum lands lower.',
        why: 'It may as well round to p or above. The order of the additions does not fix the direction of the rounding.',
      },
      {
        label: 'Either: inside a band set by the rounding bound, a test must accept both.',
        correct: true,
      },
      { label: 'Neither: the kernel is wrong if the sums differ at all.', why: 'Float addition is not associative. Two orders differing in the last bits is expected, not a bug.' },
    ],
    explain: (
      <p>
        A sum of n floats can move by about n units in the last place, whatever the order. A test of a top-p kernel must
        therefore accept either decision for tokens whose exact mass before them lies within that distance of p, and
        must use rows of exactly representable values (0.5, 0.25, 0.125, ...) to test the boundary rule itself, where
        every order gives the same bits.
      </p>
    ),
    source: 'ours, from ch10-facts A9 (float addition is not associative) and ch11-facts D4',
  },
  'ch11/11.1/q-cpu-ref': {
    kind: 'choice',
    id: 'ch11/11.1/q-cpu-ref',
    prompt: (
      <>
        Why do the labs compute their CPU reference with an explicit left-to-right loop rather than with{' '}
        <code>std::inclusive_scan</code>?
      </>
    ),
    options: [
      { label: 'The loop is faster on the CPU for the row lengths the labs test.', why: 'Speed is not the point of a reference.' },
      {
        label: "The standard leaves std::inclusive_scan's order open, so float results may vary.",
        correct: true,
      },
      {
        label: 'std::inclusive_scan skips the element at each position: it is really the exclusive scan.',
        why: 'It computes the inclusive scan; std::exclusive_scan is the other one.',
      },
      {
        label: "std::inclusive_scan arrived only in C++20, newer than the labs' language standard.",
        why: 'Both are in <numeric> since C++17.',
      },
    ],
    explain: (
      <p>
        cppreference says of both functions: "The result is non-deterministic if the add_op is not associative (such as
        floating-point addition)." The loop of Fig. 11.1 has one fixed order, so it gives the same bits every time. The
        GPU result is compared with it under a tolerance for float data, and bit for bit only for data whose every
        partial sum is exact in any order, such as all ones or small integers.
      </p>
    ),
    source: 'ch. 11 §11.1, p. 254; notes/today/ch11-facts.md A8',
  },
};

// ---------------------------------------------------------------- the section
export function S111() {
  const seq = useLabFile('ch11/snippets/s111_sequential.h');
  const d = ALL_PRESETS.find((x) => x.id === 'dyadic')!;
  const crossing = topP(d.probs, d.p, 'exclusive');
  return (
    <>
      <p>
        An <b>inclusive scan</b> takes a binary operator ⊕ and an array x₀, x₁, …, x<sub>N−1</sub>, and returns x₀, x₀ ⊕
        x₁, …, x₀ ⊕ x₁ ⊕ … ⊕ x<sub>N−1</sub>: output <i>i</i> combines every input up to and including input <i>i</i>.
        With addition it is also called a <b>prefix sum</b>, the book's running example: 3 1 7 0 4 1 6 3 scans to 3 4 11
        11 15 16 22 25.
      </p>
      <Source section="11.1" pages="252" />
      <p>
        The book's picture is a 40-inch sausage and eight people who order 3, 1, 7, 0, 4, 1, 6 and 3 inches. Served one
        after another, each cut waits for the one before it. But the inclusive scan of the orders gives every cut point at
        once: 3, 4, 11, 11, 15, 16, 22, 25 inches from the end. With all of them known, the cuts can be made in any order,
        or all together. The same is true of anything handed out in contiguous pieces: bread, campground space, a block
        of memory.
      </p>
      <Source section="11.1" pages="252–253" />
      <p>
        An <b>exclusive scan</b> leaves out the input at each position: it returns the operator's <b>identity</b> (0 for
        addition, the value that changes nothing), then x₀, then x₀ ⊕ x₁, and so on up to x<sub>N−2</sub>. For the
        sausage that is 0 3 4 11 11 15 16 22: where each piece <em>starts</em>. That is what memory allocation needs,
        since a caller gets back a pointer to the start of its block.
      </p>
      <Source section="11.1" pages="253" />
      <SausageExplorer />
      <p>
        The two scans differ by one place. Shift an inclusive scan right by one and put the identity first, and you have
        the exclusive scan. Shift an exclusive scan left, and fill the last slot with the last exclusive value plus the
        last input, and you are back. Because converting is this cheap, the book implements only inclusive scans from
        here on.
      </p>
      <Source section="11.1" pages="253" />
      <Ask q={questionsS111['ch11/11.1/q-total']} />
      <Ask q={questionsS111['ch11/11.1/q-offsets']} label="Work it out" />

      <h2>The loop every lab checks against</h2>
      <p>
        Sequentially, an inclusive scan is one loop: copy the first input, then each output is the previous output plus
        the next input. N elements take N − 1 additions, the least any scan can do. This loop is also the CPU reference
        of every lab in the chapter, for a reason the next paragraph gives.
      </p>
      <Source section="11.1" pages="253–254" figs="Fig. 11.1" />
      <Code
        tabs={[{ label: 'C++', code: seq ?? '// loading labs/ch11/snippets/s111_sequential.h' }]}
        highlight={[11, 12, 13]}
        caption={
          <>
            Fig. 11.1, with <code>const</code> on the input and a guard for an empty array, plus the exclusive scan and
            the two conversions (labs/ch11/snippets/s111_sequential.h). This is host C++, so the course's GPU statuses do
            not apply: the unit tests compile it, run it on the CPU and compare every float with the explorers on this
            page, bit for bit.
          </>
        }
      />
      <p>
        Scan is common enough that libraries have it. C++17 has <code>std::inclusive_scan</code> and{' '}
        <code>std::exclusive_scan</code>; on NVIDIA GPUs, Thrust and CUB have scans tuned for the hardware. The book
        expects you to use them in practice, and treats scan as a vehicle for teaching optimisation.
      </p>
      <Source section="11.1" pages="254" />
      <p>
        Read their documentation for one word: deterministic. Every one of these libraries, and AMD's rocPRIM, says that a
        floating-point scan may give different bits from run to run, because it may add in a different order. CUB has a
        mode that fixes the order for float addition, and rocPRIM a separate deterministic function. So the labs' reference
        is the explicit loop, whose order never changes.
      </p>
      <Today checked="2026-11-20 (ch11-facts A8)">
        <p>
          cppreference: "The result is non-deterministic if the add_op is not associative (such as floating-point
          addition)." CUB <code>DeviceScan</code> offers <code>run_to_run</code> for float <code>plus</code>; rocPRIM has{' '}
          <code>deterministic_inclusive_scan</code>.
        </p>
      </Today>
      <Ask q={questionsS111['ch11/11.1/q-cpu-ref']} />

      <h2>The token that crosses p</h2>
      <p>
        Top-p sampling sorts the probabilities of the next token in descending order and keeps the smallest set at the
        top whose total reaches p. The inclusive scan at token <i>i</i> is the mass up to and including it; the exclusive
        scan is the mass ranked before it. llama.cpp (on the host and in its backend graph), vLLM, transformers and mlx-lm
        all keep token <i>i</i> exactly when the mass before it is below p. So the token whose own probability carries the
        total past p is kept. Write "keep while the inclusive sum is below p" and you drop it: the off-by-one between
        the two scans.
      </p>
      <Beyond>
        <p>
          The book does not connect scan to sampling. The rules here are read from the projects' source on 2026-11-20
          (ch11-facts D2–D4); that they all keep the same set is our derivation.
        </p>
      </Beyond>
      <p>
        They reach that set three different ways. llama.cpp scans the descending row inclusively and adds one token after
        the cut. transformers and vLLM's PyTorch path sort ascending and drop tokens while the ascending sum is at most 1 −
        p. mlx-lm scans in reverse, exclusively, compares with p times the computed total, and then keeps every token tied
        with the boundary token. Ties are where they part: llama.cpp cuts between tied tokens wherever the sort put them,
        mlx-lm keeps them all.
      </p>
      <Simplification>
        <p>
          The probabilities are given and already sorted; there is no sampling step. Float32 is imitated by rounding after
          every addition. mlx-lm's comparison in log space is done on the probabilities. transformers' 1 − p is formed in
          double and rounded to float32 once, which is our assumption.
        </p>
      </Simplification>
      <Predict
        id="ch11/11.1/p-crossing"
        prompt={
          <>
            Sorted probabilities 0.5, 0.25, 0.125, 0.125 and p = 0.75. How many tokens does llama.cpp keep? Commit a
            number, then pick that row in the explorer below.
          </>
        }
        unit="tokens"
      >
        {(v) => (
          <p data-readout="crossing">
            It keeps <b>{crossing.kept}</b>. The masses before the tokens are {crossing.before.join(', ')}. Token 1's
            inclusive mass is exactly 0.75 = p, and it is kept, because the mass before it, 0.5, is below p. Token 2's mass
            before it is 0.75, not below p, so it is dropped. Every value here is a power of two, so float32 gives the same
            answer in any order, and transformers, vLLM and mlx-lm agree.{' '}
            {v === crossing.kept ? 'You had it.' : v === 1 ? 'You applied the inclusive rule, the off-by-one.' : ''}
          </p>
        )}
      </Predict>
      <NucleusExplorer />
      <p>
        Three things should have come out of it. Every project keeps the crossing token and the off-by-one drops it; on
        the peaked row with a small p it can even keep nothing. Ties decide the boundary differently in mlx-lm. And in
        float32, on the row where the two orders disagree, the exact mass before token {FLIP.token} is precisely p ={' '}
        {FLIP.p}: the loop's sum lands one float below p and keeps it, the Kogge-Stone sum lands on p and drops it.
      </p>
      <Ask q={questionsS111['ch11/11.1/q-rule']} label="Work it out" />
      <p>
        The sausage picture is also sampling itself. The inclusive scan of the probabilities cuts the interval from 0 to 1
        into one piece per token, as wide as its probability. Draw a uniform random number, find the piece it lands in,
        and that is the sampled token. Module X1 builds that search in{' '}
        <a href={href('x1', 'X1.8')}>X1.8, The draw: inverse CDF over a scan</a>.
      </p>
      <Aside>
        <p>
          Module X1 (sampling kernels) is ours, not the book's.
        </p>
      </Aside>

      <h2>Floating point at the boundary</h2>
      <p>
        A parallel scan adds the same probabilities in a different order from the loop, and float addition is not
        associative, so the last bits of the cumulative mass differ. Most of the time that changes nothing. But a token
        whose mass before it sits within rounding distance of p can land on either side, as the explorer showed. A test
        of a top-p kernel therefore needs a band around p, inside which it accepts either decision, and a stated tie
        rule; to test the rule itself it uses rows of powers of two, which every order adds exactly. Lab 11.6 states
        both.
      </p>
      <Beyond>
        <p>
          The book does not discuss this. Non-associativity: ch10-facts A9. The band in the explorer is γ<sub>n−1</sub>
          times the row's sum, a standard bound on any order of n − 1 additions.
        </p>
      </Beyond>
      <Ask q={questionsS111['ch11/11.1/q-float-boundary']} />
      <p>
        Next: how to compute a scan with many threads at once, in section <a href={href('ch11', '11.2')}>11.2</a>.
      </p>
    </>
  );
}
