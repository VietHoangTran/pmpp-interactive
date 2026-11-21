import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { fmt } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  BEFORE_AFTER,
  LAB116_RTOL,
  MINP_ROW,
  THRESHOLD_Q,
  VOCAB_PRESETS,
  filterBeforeSort,
  minPTestsF32,
  minPThreshold,
  minPView,
  plantedRow,
  rowView,
  type Placement,
} from '../../sim/nucleus';

// Section X1.6: min-p as a filter. The explorers draw what src/sim/nucleus.ts computes
// (minPView, minPTestsF32, filterBeforeSort), pinned in nucleus.test.ts.

const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const n0 = (v: number) => v.toLocaleString('en-US');
const f3 = (v: number) => (isFinite(v) ? v.toFixed(3) : '−∞');
const MINPS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5];

// ---------------------------------------------------------------- before or after temperature
function TempExplorer() {
  const [minP, setMinP] = useState(THRESHOLD_Q.minP);
  const [T, setT] = useState(1);
  const [placement, setPlacement] = useState<Placement>('before');
  const [p, setP] = useState(0.95);
  const [plant, setPlant] = useState(false);
  const planted = useMemo(() => plantedRow(minP), [minP]);
  const row = plant ? planted.logits : MINP_ROW;
  const v = minPView(row, minP, T, p);
  const f32 = plant ? minPTestsF32(planted.logits, minP) : null;
  const thr = placement === 'before' ? v.thrBefore : v.thrAfter;
  const W = 600;
  const H = 210;
  const X0 = 44;
  const lo = -3;
  const hi = 5.5;
  const sy = (x: number) => H - 26 - ((Math.max(lo, Math.min(hi, x)) - lo) / (hi - lo)) * (H - 46);
  const bw = (W - X0 - 10) / row.length;
  return (
    <Bench
      title="Min-p before or after temperature"
      task="Leave min-p at 0.05 and move T both ways, under each placement. Then plant a logit at the threshold."
      foot={
        <>
          Bars are the row's logits, largest first; filled bars survive min-p under the placement you chose. Both
          thresholds are drawn in the units of these unscaled logits: before temperature l_max + ln p_base, after
          temperature l_max + T ln p_base. A dot marks the tokens top-p keeps at this temperature.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          min-p (p_base)
          <span className="seg" role="group" aria-label="min-p">
            {MINPS.map((m) => (
              <button key={m} aria-pressed={minP === m} onClick={() => setMinP(m)}>{m}</button>
            ))}
          </span>
        </span>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          temperature T = {T.toFixed(2)}
          <input type="range" min={10} max={200} step={5} value={Math.round(T * 100)} aria-label="temperature" onChange={(e) => setT(Number(e.target.value) / 100)} style={{ width: '100%' }} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Min-p runs
          <span className="seg" role="group" aria-label="Placement">
            <button aria-pressed={placement === 'before'} onClick={() => setPlacement('before')}>before temperature (llama.cpp, mlx-lm)</button>
            <button aria-pressed={placement === 'after'} onClick={() => setPlacement('after')}>after (the paper, transformers)</button>
          </span>
        </span>
        <label className="field" style={{ flex: '1 1 8rem' }}>
          top-p p = {p.toFixed(2)}
          <input type="range" min={50} max={100} step={1} value={Math.round(p * 100)} aria-label="top-p" onChange={(e) => setP(Number(e.target.value) / 100)} style={{ width: '100%' }} />
        </label>
        <label className="check">
          <input type="checkbox" checked={plant} onChange={(e) => setPlant(e.target.checked)} /> plant a logit at the float threshold
        </label>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="The row's logits and min-p's thresholds" data-part="bars"
          style={{ width: '100%', maxWidth: W, minWidth: 460, display: 'block', background: 'var(--page)', border: '1px solid var(--rule)', borderRadius: 4 }}>
          {[-2, 0, 2, 4].map((g) => (
            <g key={g}>
              <line x1={X0} y1={sy(g)} x2={W - 10} y2={sy(g)} stroke="var(--rule-soft)" />
              <text x={X0 - 6} y={sy(g) + 4} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{g}</text>
            </g>
          ))}
          {v.tokens.map((t) => {
            const keep = placement === 'before' ? t.before : t.after;
            const x = X0 + t.index * bw + 2;
            const isPlanted = plant && t.index === planted.planted;
            return (
              <g key={t.index} data-token={t.index} data-keep={keep ? '1' : '0'} data-topp={t.topP ? '1' : '0'}>
                <rect x={x} y={Math.min(sy(t.logit), sy(0))} width={bw - 4} height={Math.abs(sy(t.logit) - sy(0))}
                  fill={keep ? 'var(--ink-3)' : 'var(--cell)'} fillOpacity={keep ? 0.45 : 1}
                  stroke={isPlanted ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={isPlanted ? 2 : 1} strokeDasharray={keep ? undefined : '3 2'} />
                {t.topP ? <circle cx={x + (bw - 4) / 2} cy={20} r={3} fill="var(--ink)" /> : null}
                <text x={x + (bw - 4) / 2} y={H - 10} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{t.logit.toFixed(1)}</text>
              </g>
            );
          })}
          <line x1={X0} y1={sy(v.thrBefore)} x2={W - 10} y2={sy(v.thrBefore)} stroke="var(--ink)"
            strokeWidth={placement === 'before' ? 2 : 1} strokeDasharray={placement === 'before' ? undefined : '5 4'} opacity={placement === 'before' ? 1 : 0.5} />
          <line x1={X0} y1={sy(v.thrAfter)} x2={W - 10} y2={sy(v.thrAfter)} stroke="var(--ink)"
            strokeWidth={placement === 'after' ? 2 : 1} strokeDasharray={placement === 'after' ? undefined : '5 4'} opacity={placement === 'after' ? 1 : 0.5} />
          {/* The higher threshold is labelled above its line, the lower one below, so the labels never overlap. */}
          <text x={W - 12} y={v.thrBefore >= v.thrAfter ? sy(v.thrBefore) - 4 : sy(v.thrBefore) + 12} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink)' }}>before: {f3(v.thrBefore)}</text>
          <text x={W - 12} y={v.thrBefore >= v.thrAfter ? sy(v.thrAfter) + 12 : sy(v.thrAfter) - 4} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink)' }}>after: {f3(v.thrAfter)}</text>
          <text x={X0} y={12} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>● kept by top-p at T = {T.toFixed(2)}</text>
        </svg>
      </div>
      <table className="data" style={{ margin: '0.8rem 0 0', background: 'var(--page)' }} data-part="kept">
        <tbody>
          <tr data-row="before" data-kept={v.keptBefore}>
            <td style={{ paddingLeft: '0.6rem' }}>min-p before temperature (llama.cpp, mlx-lm)</td>
            <td className="num">threshold {f3(v.thrBefore)}</td>
            <td className="num"><b>{v.keptBefore}</b> kept</td>
          </tr>
          <tr data-row="after" data-kept={v.keptAfter}>
            <td style={{ paddingLeft: '0.6rem' }}>min-p after temperature (the paper, transformers)</td>
            <td className="num">threshold {f3(v.thrAfter)} ({f3(v.thrAfter / T)} on the scaled logits)</td>
            <td className="num"><b>{v.keptAfter}</b> kept</td>
          </tr>
          <tr data-row="topp" data-kept={v.keptTopP}>
            <td style={{ paddingLeft: '0.6rem' }}>top-p {p.toFixed(2)} at T = {T.toFixed(2)}, for comparison</td>
            <td />
            <td className="num"><b>{v.keptTopP}</b> kept</td>
          </tr>
        </tbody>
      </table>
      <p style={{ margin: '0.7rem 0 0' }} data-readout="placement" data-threshold={thr}>
        {T < 1 && v.keptAfter < v.keptBefore
          ? `At T = ${T.toFixed(2)} min-p after temperature is stricter: ${v.keptAfter} tokens against ${v.keptBefore}.`
          : T > 1 && v.keptAfter > v.keptBefore
            ? `At T = ${T.toFixed(2)} min-p after temperature is looser: ${v.keptAfter} tokens against ${v.keptBefore}.`
            : `At T = ${T.toFixed(2)} both placements keep ${v.keptBefore}.`}{' '}
        Before temperature, the threshold does not move with T at all.
      </p>
      {f32 ? (
        <p style={{ margin: '0.5rem 0 0' }} data-readout="planted" data-logit={f32.logit[planted.planted] ? '1' : '0'} data-prob={f32.prob[planted.planted] ? '1' : '0'}>
          Planted at T = 1: logit {planted.logits[planted.planted].toPrecision(9)}, exactly llama.cpp's float threshold{' '}
          <code>max + logf(p)</code>. The logit test {f32.logit[planted.planted] ? 'keeps' : 'drops'} it; the probability test
          computed in float32 {f32.prob[planted.planted] ? 'keeps' : 'drops'} it
          {f32.logit[planted.planted] !== f32.prob[planted.planted] ? <b>: the two tests disagree by one rounding</b> : ', so they agree here'}.
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- filter before you sort
function FilterExplorer() {
  const [V, setV] = useState(151936);
  const [s, setS] = useState(1);
  const [seed, setSeed] = useState(1);
  const [minP, setMinP] = useState(0.05);
  const [p, setP] = useState(0.95);
  const row = useMemo(() => rowView(V, s, seed), [V, s, seed]);
  const f = filterBeforeSort(row, p, minP);
  const ids = (x: Uint32Array) => {
    const head = Array.from(x.slice(0, 8)).join(', ');
    return x.length > 8 ? `${head}, … (${n0(x.length)})` : head;
  };
  const routes = [
    { id: 'A', label: 'A. top-p then min-p over the whole sorted row (llama.cpp\'s order)', sorted: V, kept: f.routeA },
    { id: 'B', label: 'B. min-p as a compaction, sort only the survivors, top-p with full-row probabilities', sorted: f.survivors, kept: f.routeB },
    { id: 'C', label: 'C. as B, but top-p renormalises over the survivors', sorted: f.survivors, kept: f.routeC },
  ];
  return (
    <Bench
      title="Filter before you sort"
      task="Check that A and B agree, then see how few values B sorts. Set min-p to 0 and to a flat row; compare C."
      foot={
        <>
          A and B keep tokens largest first, ties by token id. Probabilities in A and B are the full row's: the same row max
          and the same softmax denominator. C divides by the survivors' sum instead.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          Vocabulary
          <span className="seg" role="group" aria-label="Vocabulary">
            {VOCAB_PRESETS.map((x) => (
              <button key={x} aria-pressed={V === x} onClick={() => setV(x)}>{n0(x)}</button>
            ))}
          </span>
        </span>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          shape s = {s.toFixed(2)} ({s < 0.8 ? 'flat' : s < 1.4 ? 'in between' : 'peaked'})
          <input type="range" min={0} max={3} step={0.05} value={s} aria-label="shape" onChange={(e) => setS(Number(e.target.value))} style={{ width: '100%' }} />
        </label>
        <button onClick={() => setSeed(seed + 1)}>another row</button>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <span style={FIELD}>
          min-p
          <span className="seg" role="group" aria-label="min-p">
            {[0, ...MINPS.slice(0, 5)].map((m) => (
              <button key={m} aria-pressed={minP === m} onClick={() => setMinP(m)}>{m === 0 ? 'off' : m}</button>
            ))}
          </span>
        </span>
        <label className="field" style={{ flex: '1 1 12rem' }}>
          top-p p = {p.toFixed(2)}
          <input type="range" min={50} max={100} step={1} value={Math.round(p * 100)} aria-label="top-p" onChange={(e) => setP(Number(e.target.value) / 100)} style={{ width: '100%' }} />
        </label>
      </div>
      <p style={{ marginTop: 0 }} data-readout="survivors" data-survivors={f.survivors}>
        Min-p {minP === 0 ? 'off' : minP} leaves <b>{n0(f.survivors)}</b> of {n0(V)} tokens
        {minP > 0 ? <> (threshold {f3(f.threshold)}, the row max {f3(row.lmax)} plus ln {minP})</> : null}. Route B sorts{' '}
        {n0(f.survivors)} values instead of {n0(V)}
        {f.survivors < V ? `: ${fmt(V / f.survivors)} times fewer` : ': no saving at all'}.
      </p>
      <div style={{ height: 12, background: 'var(--cell)', border: '1px solid var(--rule)', borderRadius: 2, marginBottom: '0.8rem' }} aria-hidden="true">
        <div style={{ height: '100%', width: `${Math.max(0.4, (f.survivors / V) * 100)}%`, background: 'var(--global)' }} />
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ margin: 0, background: 'var(--page)', minWidth: '38rem' }} data-part="routes">
          <thead>
            <tr><th style={{ paddingLeft: '0.6rem' }}>Route</th><th className="num">Values sorted</th><th className="num">Kept</th><th>Token ids kept, largest first</th></tr>
          </thead>
          <tbody>
            {routes.map((r) => (
              <tr key={r.id} data-route={r.id} data-kept={r.kept.length} data-ids={Array.from(r.kept.slice(0, 50)).join(',')}>
                <td style={{ paddingLeft: '0.6rem', whiteSpace: 'normal' }}>{r.label}</td>
                <td className="num">{n0(r.sorted)}</td>
                <td className="num"><b>{n0(r.kept.length)}</b></td>
                <td className="mono" style={{ fontSize: '0.75rem' }}>{ids(r.kept)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ margin: '0.7rem 0 0' }} data-readout="verdict" data-identical={f.identical ? '1' : '0'} data-c-differs={f.cDiffers ? '1' : '0'}>
        {f.identical ? <span className="chip ok">A and B keep identical sets</span> : <span className="chip fail">A and B differ</span>}{' '}
        top-p's prefix a = {n0(f.a)}, min-p's prefix b = {n0(f.b)}, kept = min(a, b) = {n0(f.routeA.length)}.{' '}
        {f.cDiffers
          ? `C keeps ${n0(f.routeC.length)}: renormalising over the survivors inflates every probability, so top-p stops sooner.`
          : 'C agrees here.'}
      </p>
      <p className="muted" style={{ margin: '0.4rem 0 0' }} data-readout="band" data-band={f.band}>
        {n0(f.band)} {f.band === 1 ? 'token has' : 'tokens have'} a top-p decision inside lab 11.6's band (mass before within{' '}
        {LAB116_RTOL} of p, relative): a float32 kernel may decide {f.band === 1 ? 'it' : 'them'} either way.{' '}
        {f.band === 0
          ? ''
          : f.bandInMinP === 0
            ? 'None of them is inside min-p\'s prefix, so here they cannot change what is kept.'
            : `${n0(f.bandInMinP)} of them ${f.bandInMinP === 1 ? 'is' : 'are'} inside min-p's prefix, where the decision can change what is kept.`}
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const BA = minPView(BEFORE_AFTER.logits, BEFORE_AFTER.minP, BEFORE_AFTER.T, 1);

export const questionsSX16: Record<string, Question> = {
  'x1/X1.6/q-before-after': {
    kind: 'numeric',
    id: 'x1/X1.6/q-before-after',
    prompt: (
      <>
        Logits [3.0, 2.0, 1.0], min-p 0.1, T = 0.5. llama.cpp (min-p before temperature) keeps all {BA.keptBefore}. How many
        does transformers (min-p after temperature) keep?
      </>
    ),
    answer: BA.keptAfter,
    unit: 'tokens',
    explain: (
      <p>
        After temperature the logits are [6, 4, 2] and the threshold is 6 + ln 0.1 = {f3(BA.thrAfter / BEFORE_AFTER.T)}: 6 and
        4 survive, 2 does not. Before temperature the threshold is 3 + ln 0.1 = {f3(BA.thrBefore)} and all three survive. A
        low temperature makes after-temperature min-p stricter.
      </p>
    ),
    source: 'notes/today/x1-facts.md D1, D4, D9',
  },
  'x1/X1.6/q-prefix': {
    kind: 'choice',
    id: 'x1/X1.6/q-prefix',
    prompt: (
      <>
        On a row sorted largest first, top-p keeps a prefix of length a and min-p a prefix of length b. In llama.cpp's
        order (top-p, then min-p), how many tokens survive both?
      </>
    ),
    options: [
      { label: "min(a, b), and at least 1: top-p keeps the max, so min-p's threshold does not move", correct: true },
      { label: 'a + b, since each filter adds the tokens it keeps', why: 'Both are prefixes of the same row; surviving both means being in the shorter one.' },
      { label: "It depends on which filter runs first, because top-p renormalises the row and that moves min-p's threshold", why: "In llama.cpp's order min-p's threshold uses the max logit, which top-p never removes; no renormalisation enters its test." },
    ],
    explain: (
      <p>
        Every truncation of a sorted row is a prefix. Min-p's threshold depends only on the largest logit, which top-p
        keeps, so min-p's prefix is the same whether top-p ran first or not. Running min-p first and then top-p over
        renormalised probabilities would be a different rule (route C in the explorer).
      </p>
    ),
    source: 'ours; notes/today/x1-facts.md D1, ch12-facts D3',
  },
  'x1/X1.6/q-equivalent': {
    kind: 'choice',
    id: 'x1/X1.6/q-equivalent',
    prompt: (
      <>
        Top-k is off. You run min-p first as a stable compaction, sort only the survivors and apply top-p using full-row
        probabilities. Does that keep the same tokens as llama.cpp's top-p then min-p over the whole row?
      </>
    ),
    options: [
      { label: 'Yes, in exact arithmetic: every token ranked before a survivor also survives, so the mass before it is unchanged', correct: true },
      { label: 'No: top-p needs the whole sorted row, so a filtered row must give a different nucleus', why: 'Top-p needs the mass ranked before each kept token, and all of that mass belongs to survivors.' },
      { label: 'Only when p is larger than min-p, so that top-p is the looser of the two filters', why: 'The two numbers are not comparable; the argument holds for any p and min-p.' },
    ],
    explain: (
      <p>
        A token ranked before a survivor has a logit at least as large, so it clears min-p's threshold too. The mass before
        each survivor, computed with the full row's denominator, is then the same sum over the same tokens. The kept
        prefix is min(a, b) both ways. In floats, the two routes can still differ for tokens inside the rounding band.
      </p>
    ),
    source: 'ours',
  },
  'x1/X1.6/q-mask': {
    kind: 'choice',
    id: 'x1/X1.6/q-mask',
    prompt: <>Why do llama.cpp's experimental backend samplers and mlx-lm set losing tokens to minus infinity instead of compacting them into a list?</>,
    options: [
      { label: 'A compute graph needs static shapes, and the number of survivors depends on the data', correct: true },
      { label: 'Masking moves fewer bytes than compaction, so it is the faster choice on every backend', why: 'A mask still reads and writes the whole row; speed is not the stated reason.' },
      { label: 'Compaction cannot be written on Metal, which has no stable filter primitive', why: 'Lab 12.4 compacts on Metal; MPSGraph also has nonZeroIndices.' },
    ],
    explain: (
      <p>
        MLX documents limited support for operations whose output shape depends on the input data. llama.cpp's backend
        top-k does shrink the row, but by a fixed k known when the graph is built. A hand-written kernel can return a list
        and a count; sizing the next step from that count needs a host read, or a cap with a fallback.
      </p>
    ),
    source: 'notes/today/ch12-facts.md D3, D6',
  },
  'x1/X1.6/q-space': {
    kind: 'choice',
    id: 'x1/X1.6/q-space',
    prompt: <>Why is the logit-space test l ≥ l_max + ln p_base the one a kernel should use?</>,
    options: [
      { label: 'It needs only the row max and one compare per token, with no exp and no softmax', correct: true },
      { label: 'It gives a different, better-calibrated set of tokens than the test on probabilities', why: 'In exact arithmetic it is the same set; only at the threshold can rounding make them differ.' },
      { label: 'Probabilities underflow to zero for most of a 150,000-token row, so their test is undefined', why: 'Small probabilities compare correctly; underflow is not the reason.' },
    ],
    explain: (
      <p>
        Divide p ≥ p_base · p_max by the common softmax denominator and take logs: the denominator cancels. The two tests
        pick the same set in exact arithmetic but can differ by one rounding exactly at the threshold, so a lab must say
        which one it tests.
      </p>
    ),
    source: 'ours; notes/today/x1-facts.md D9, ch12-facts D3',
  },
};

// ---------------------------------------------------------------- the section
export function SX16() {
  const thr = minPThreshold(THRESHOLD_Q.lmax, THRESHOLD_Q.minP);
  return (
    <>
      <p>
        Min-p keeps a token when its probability is at least a fraction p_base of the top token's. It adapts to the shape
        of the row: a confident row keeps few tokens, a flat one many. This page turns it into a single comparison, runs it
        two ways with labs you have, and shows that it can shrink the sort that top-p needs.
      </p>
      <Beyond>
        <p>
          Min-p is not in the book. The stable filter is §12.5 (pp. 297–298) and the row max is chapter 10's reduction.
        </p>
      </Beyond>

      <h2>One comparison, no softmax</h2>
      <p>
        Nguyen et al. keep every token with probability at least p_base × p_max. Every probability in a row has the same
        softmax denominator, so divide it out and take logs: keep a token when its logit is at least l_max + ln p_base. That
        needs the row's max and one compare per token, with no exp and no sum. llama.cpp's host min-p is exactly this test,
        a sequential stable filter on logits. transformers tests probabilities after a softmax and keeps a token exactly at
        the threshold; mlx-lm tests log-probabilities.
      </p>
      <Today checked="2026-11-20 (notes/today/x1-facts.md D4, D8, D9; ch12-facts D3)">
        <p>
          llama.cpp: <code>const float min_logit = max_logit + logf(ctx-&gt;p);</code> then keeps{' '}
          <code>logit &gt;= min_logit</code> into a new list.
        </p>
      </Today>
      <Predict id="x1/X1.6/p-threshold" prompt={<>The largest logit is 5.0 and min-p is 0.05. What is the logit threshold?</>}>
        {(v) => (
          <p data-reveal="threshold">
            5 + ln 0.05 = <b>{thr.toFixed(3)}</b>
            {Math.abs(v - thr) < 0.01 ? ', as you said' : ''}. Every token with a logit of 2.004 or more survives; the explorer
            below opens on that row.
          </p>
        )}
      </Predict>
      <Ask q={questionsSX16['x1/X1.6/q-space']} />

      <h2>Before or after temperature</h2>
      <p>
        Where min-p sits in the chain changes what it keeps. The paper and transformers apply it after temperature, which in
        the unscaled logits is a threshold of l_max + T ln p_base. llama.cpp's default chain and mlx-lm apply it before
        temperature, at T = 1, so temperature never moves it.
      </p>
      <Today checked="2026-11-20 (notes/today/x1-facts.md D1, D4, D8, D9)">
        <p>
          The paper: "After applying temperature scaling, the scaled threshold pscaled is computed". transformers' comment
          says the same. llama.cpp's default chain ends with temperature; mlx-lm applies it last.
        </p>
      </Today>
      <p>
        The explorer opens on a 16-token row whose largest logit is 5.0, with min-p 0.05: the threshold you just
        computed. Both thresholds are drawn; the placement you pick decides which one cuts.
      </p>
      <Simplification>
        <p>
          A fixed 16-token row; arithmetic in double, except the planted-logit check, which emulates float32 with round to
          nearest and sums in order. Real kernels sum in other orders.
        </p>
      </Simplification>
      <TempExplorer />
      <p>
        After-temperature min-p is stricter for T below 1 and looser above it, because T multiplies ln p_base, a negative
        number. Before-temperature min-p does not move with T at all.
      </p>
      <Ask q={questionsSX16['x1/X1.6/q-before-after']} label="Work it out" />

      <h2>Two kernels for one rule</h2>
      <p>
        On an unsorted row, min-p is lab 10.3's row max, one threshold per row (ln p_base computed once on the host, in
        float, as llama.cpp does), and lab 12.4's <code>launchCompactCandidates</code>: llama.cpp's host loop as a GPU
        stable filter, which writes a list of (column, logit) pairs and a count per row.
      </p>
      <Source section="12.5" pages="297–298" figs="Figs. 12.7, 12.8" />
      <p>
        On a sorted row, which is what M5 has after lab 14.3, the survivors are a prefix, so min-p is a count: the number of
        sorted logits at least the first one plus ln p_base. Lab X1.2 writes it. Top-p is a prefix of the same row, so after
        both the survivors are the shorter prefix, and never fewer than one token.
      </p>
      <Ask q={questionsSX16['x1/X1.6/q-prefix']} />

      <h2>Filter before you sort</h2>
      <p>
        With top-k off, a top-p over the whole vocabulary sorts the whole vocabulary. But min-p usually keeps very few
        tokens, so run it first: compact the survivors with lab 12.4, sort only them, and compute top-p's mass with the
        full row's probabilities. Every token ranked before a survivor survives min-p too, so the mass before each survivor
        is the same as over the whole row. In exact arithmetic the kept set is exactly llama.cpp's top-p then min-p.
      </p>
      <Beyond>
        <p>Filtering before sorting is our design, not a project's; the explorer checks it on synthetic rows.</p>
      </Beyond>
      <p>
        The explorer runs the three routes side by side on a whole synthetic row and counts what each one sorts. Route C is
        the tempting shortcut: min-p first, then a top-p that renormalises over what is left.
      </p>
      <Simplification>
        <p>
          Synthetic rows (seeded Zipf-like logits, as in X1.5), not a model's. Arithmetic in double; the band count says
          where float32 could decide differently.
        </p>
      </Simplification>
      <FilterExplorer />
      <p>
        The catch is the count. How many tokens survive depends on the data, so the size of the next sort is not known
        when the work is planned. A graph with static shapes cannot express it, which is why llama.cpp's backend samplers
        and MLX mask losers with minus infinity instead of compacting. A kernel can return a list and a count, but sizing
        the sort that follows needs either a host read of the counts or a cap with a fallback. M5 sidesteps it by running
        top-k first.
      </p>
      <Today checked="2026-11-20 (notes/today/ch12-facts.md D3, D6)">
        <p>
          MLX: "MLX has limited support for operations for which output shapes are dependent on input data." llama.cpp's
          backend min-p and top-p: "Apply -INFINITY bias for masked-out tokens".
        </p>
      </Today>
      <Ask q={questionsSX16['x1/X1.6/q-equivalent']} />
      <Ask q={questionsSX16['x1/X1.6/q-mask']} />

      <h2>Floats at the threshold</h2>
      <p>
        The logit test and the probability test agree in exact arithmetic, but not always at the threshold itself, where
        one rounding decides. Tick "plant a logit at the float threshold" in the first explorer: a logit set exactly to
        llama.cpp's <code>max + logf(p)</code> passes the logit test, and a float32 softmax can put its probability one
        rounding below p_base × p_max. Lab X1.2 plants such a logit in its tests, which is why its contract has to say which
        of the two tests it means.
      </p>
      <p>
        Next, the survivors need one token drawn from them: <a href={href('x1', 'X1.7')}>random numbers every backend agrees
        on</a>, then the draw itself.
      </p>
    </>
  );
}
