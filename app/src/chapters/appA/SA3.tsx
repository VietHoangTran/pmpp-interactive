import { Fragment, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  CLASSIFIER_FORMATS,
  OUTCOMES,
  OUTCOME_LABEL,
  classifierPresets,
  figA8Agrees,
  figA8Row,
  patternCounts,
  specialTable,
  type Outcome,
} from '../../sim/fpadd';
import { BF16, FP16, FP32, anatomy, bitsOfValue, flipBit, maxFinite, minNormal, show, totalBits, type Cls, type Format } from '../../sim/minifloat';
import { SOFTMAX_FORMATS, SOFTMAX_PRESETS, overflowThreshold, softmaxTrace, type Event, type Quantity } from '../../sim/softmaxnum';

const FACTS = 'notes/today/appA-facts.md';
const CHECKED = (items: string) => `2026-11-20 (${FACTS}, ${items})`;

// ---------------------------------------------------------------- the classifier (Fig. A.8)
const CLASS_LABEL: Record<Cls, string> = {
  zero: 'zero',
  subnormal: 'denormal',
  normal: 'normal number',
  inf: 'infinity',
  qnan: 'NaN, quiet',
  snan: 'NaN, signalling',
  nan: 'NaN',
  reserved: 'reserved',
};
const FIG_ROWS: { key: 'nan' | 'inf' | 'denorm' | 'zero'; e: string; m: string; meaning: string }[] = [
  { key: 'nan', e: '11…1', m: '≠ 0', meaning: 'NaN' },
  { key: 'inf', e: '11…1', m: '= 0', meaning: '(−1)^S × ∞' },
  { key: 'denorm', e: '00…0', m: '≠ 0', meaning: 'denormalised' },
  { key: 'zero', e: '00…0', m: '= 0', meaning: '0' },
];
const hexOf = (f: Format, b: number) => '0x' + (b >>> 0).toString(16).padStart(Math.ceil(totalBits(f) / 4), '0');
const fmtCount = (n: number) => n.toLocaleString('en-US');

function Classifier() {
  const [fid, setFid] = useState('fp32');
  const f = CLASSIFIER_FORMATS.find((x) => x.id === fid)!;
  const [bits, setBitsRaw] = useState(() => bitsOfValue(FP32, 1));
  const [hexText, setHexText] = useState(hexOf(FP32, bitsOfValue(FP32, 1)));
  const n = totalBits(f);
  const setBits = (b: number, g: Format = f) => {
    setBitsRaw(b >>> 0);
    setHexText(hexOf(g, b));
  };
  const a = anatomy(f, bits);
  const row = figA8Row(f, bits);
  const agrees = figA8Agrees(f, bits);
  const counts = useMemo(() => patternCounts(f), [f]);
  const presets = classifierPresets(f);
  const nan = a.cls === 'qnan' || a.cls === 'snan' || a.cls === 'nan';

  const bitButton = (i: number) => {
    const on = Math.floor(bits / 2 ** i) % 2 === 1;
    return (
      <button key={i} aria-pressed={on} data-bit={i} onClick={() => setBits(flipBit(bits, i))} title={`bit ${i}`}
        style={{ padding: '0.15rem 0', width: '1.45rem', fontFamily: 'var(--sans)', fontVariantNumeric: 'tabular-nums' }}>
        {on ? 1 : 0}
      </button>
    );
  };
  const range = (hi: number, lo: number) => Array.from({ length: hi - lo + 1 }, (_, k) => hi - k);
  const field = (label: string, idx: number[]) => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
      <span className="muted" style={{ fontSize: '0.8rem' }}>{label}</span>
      <span style={{ display: 'flex', flexWrap: 'wrap', gap: '2px' }}>{idx.map(bitButton)}</span>
    </div>
  );

  return (
    <Bench
      title="Classify a bit pattern"
      task="Press each preset in FP32, then switch to FP8 E4M3 and press 'largest finite'."
      foot={<>Formats from the IEEE standard (FP32, FP16), NVIDIA's tables (BF16) and the OCP specifications (FP8, FP4). Only FP32 is in the book.</>}
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label="Format" style={{ flexWrap: 'wrap', rowGap: '4px' }}>
          {CLASSIFIER_FORMATS.map((x) => (
            <button key={x.id} aria-pressed={x.id === fid} onClick={() => { setFid(x.id); setBits(bitsOfValue(x, 1), x); }}>
              {x.name.replace('FP8 ', '').replace('FP4 ', '')}
            </button>
          ))}
        </span>
        <label className="field">
          bits in hex
          <input type="text" value={hexText} aria-label="bits in hex" style={{ width: '9rem' }}
            onChange={(e) => {
              const t = e.target.value.trim();
              setHexText(e.target.value);
              const m = /^(0x)?([0-9a-f]+)$/i.exec(t);
              if (m) {
                const v = Number.parseInt(m[2], 16);
                if (v < 2 ** n) setBitsRaw(v >>> 0);
              }
            }} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem', gap: '0.4rem 0.5rem' }}>
        {presets.map((p) => (
          <button key={p.id} className="quiet" data-preset={p.id} disabled={p.bits === null} title={p.reason ?? ''}
            onClick={() => p.bits !== null && setBits(p.bits)}>
            {p.label}
          </button>
        ))}
      </div>
      {presets.some((p) => p.bits === null) ? (
        <ul className="muted" style={{ margin: '0 0 0.8rem', paddingLeft: '1.2rem' }} data-missing>
          {presets.filter((p) => p.bits === null).map((p) => <li key={p.id}>No {p.label}: {p.reason}.</li>)}
        </ul>
      ) : null}
      <div className="row" style={{ alignItems: 'flex-start', gap: '0.8rem 1rem', marginBottom: '0.9rem' }}>
        {f.signed ? field('S', [n - 1]) : null}
        {field(`exponent (${f.e} bits, bias ${f.bias})`, range(f.e + f.m - 1, f.m))}
        {f.m > 0 ? field(`mantissa (${f.m} bits)`, range(f.m - 1, 0)) : null}
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
        <div className="panel" data-cls={a.cls} data-value={String(a.value)}>
          <p style={{ margin: '0 0 0.4rem' }}>
            <b data-class>{CLASS_LABEL[a.cls]}</b>
          </p>
          <p className="mono" style={{ margin: '0 0 0.4rem' }}>
            S = {a.s}, E = {a.E}{a.E === 2 ** f.e - 1 ? ' (all ones)' : a.E === 0 ? ' (all zeros)' : ''}, M = {a.M}
          </p>
          <p style={{ margin: 0 }} data-shown>
            {a.rule === 'special'
              ? nan
                ? a.cls === 'qnan'
                  ? 'Top mantissa bit set: quiet. Arithmetic passes it on without a fuss.'
                  : a.cls === 'snan'
                    ? 'Top mantissa bit clear: signalling, by the IEEE convention. On a CPU, using it can raise an exception; on a GPU it cannot (see below).'
                    : `${f.name} does not divide its NaN patterns into quiet and signalling.`
                : `value ${show(a.value)}`
              : a.rule === 'denormal'
                ? <>value 0.M × 2<sup>{1 - f.bias}</sup> = {show(a.value, 9)}</>
                : <>value 1.M × 2<sup>{a.E - f.bias}</sup> = {show(a.value, 9)}</>}
          </p>
          {!agrees ? (
            <p className="t-fail" style={{ margin: '0.5rem 0 0' }} data-disagree>
              Fig. A.8 would call this pattern {row === 'nan' ? 'NaN' : row === 'inf' ? 'infinity' : row}. In {f.name} it is {a.cls === 'normal' ? `the number ${show(a.value)}` : CLASS_LABEL[a.cls]}.
            </p>
          ) : null}
        </div>
        <table className="data" style={{ margin: 0 }} aria-label="Fig. A.8">
          <thead>
            <tr><th>exponent</th><th>mantissa</th><th>meaning (Fig. A.8)</th></tr>
          </thead>
          <tbody>
            {FIG_ROWS.map((r) => (
              <tr key={r.key} data-fig={r.key} style={r.key === row ? { background: 'var(--rule-soft)', fontWeight: 600 } : undefined}>
                <td className="mono">{r.e}</td>
                <td>{r.m}</td>
                <td>{r.meaning}</td>
              </tr>
            ))}
            <tr data-fig="other" style={row === null ? { background: 'var(--rule-soft)', fontWeight: 600 } : undefined}>
              <td colSpan={2}>anything else</td>
              <td>a normal number</td>
            </tr>
          </tbody>
        </table>
      </div>
      <table className="data" style={{ background: 'var(--page)', margin: '1rem 0 0', maxWidth: '26rem' }} aria-label="How the patterns are spent" data-counts={JSON.stringify(counts)}>
        <thead>
          <tr><th style={{ paddingLeft: '0.6rem' }}>{f.name}: {fmtCount(counts.total)} patterns</th><th className="num">how many</th></tr>
        </thead>
        <tbody>
          {([['zero', counts.zero], ['denormal', counts.subnormal], ['normal', counts.normal], ['infinity', counts.inf]] as [string, number][]).map(([k, v]) => (
            <tr key={k}><td style={{ paddingLeft: '0.6rem' }}>{k}</td><td className="num">{fmtCount(v)}</td></tr>
          ))}
          <tr>
            <td style={{ paddingLeft: '0.6rem' }}>NaN{counts.qnan + counts.snan ? <span className="muted"> ({fmtCount(counts.qnan)} quiet, {fmtCount(counts.snan)} signalling)</span> : null}</td>
            <td className="num" data-nan-count={counts.qnan + counts.snan + counts.nan}>{fmtCount(counts.qnan + counts.snan + counts.nan)}</td>
          </tr>
          <tr><td style={{ paddingLeft: '0.6rem' }}>largest finite value</td><td className="num">{show(maxFinite(f), 6)}</td></tr>
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- what comes out?
function WhatComesOut() {
  const rows = useMemo(() => specialTable(), []);
  const [picks, setPicks] = useState<Record<string, Outcome | ''>>({});
  const [shown, setShown] = useState(false);
  const right = rows.filter((r) => picks[r.id] === r.outcome).length;
  const picked = rows.filter((r) => picks[r.id]).length;
  return (
    <Bench
      title="What comes out?"
      task="Pick an answer for every row, then reveal. Each row is evaluated in FP32 unless it says otherwise."
      foot={<>Every result is computed by the course's float model, which follows IEEE 754, and fmaxf by CUDA's documented rule.</>}
    >
      <table className="data" style={{ background: 'var(--page)', margin: '0 0 0.8rem' }}>
        <thead>
          <tr><th style={{ paddingLeft: '0.6rem' }}>expression</th><th>your answer</th><th>{shown ? 'result' : ''}</th></tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const p = picks[r.id] ?? '';
            const ok = p === r.outcome;
            const joined = shown ? { borderBottom: 'none' } : undefined;
            return (
              <Fragment key={r.id}>
                <tr data-row={r.id} data-outcome={r.outcome}>
                  <td className="mono" style={{ paddingLeft: '0.6rem', ...joined }}>{r.expr}</td>
                  <td style={joined}>
                    <select value={p} aria-label={'your answer for ' + r.expr} style={{ maxWidth: '10rem' }}
                      onChange={(e) => setPicks({ ...picks, [r.id]: e.target.value as Outcome })}>
                      <option value="">choose</option>
                      {OUTCOMES.map((o) => <option key={o} value={o}>{OUTCOME_LABEL[o]}</option>)}
                    </select>
                  </td>
                  <td className="nowrap" style={joined} data-result>
                    {shown ? (
                      <span className={'chip ' + (p ? (ok ? 'ok' : 'fail') : '')} data-mark={p ? (ok ? 'right' : 'wrong') : 'none'}>
                        {typeof r.value === 'boolean' ? String(r.value) : show(r.value)}
                      </span>
                    ) : null}
                  </td>
                </tr>
                {shown ? (
                  <tr data-rule={r.id}>
                    <td colSpan={3} style={{ paddingLeft: '0.6rem', paddingTop: 0 }}>
                      {r.rule} <span className="muted">({r.source})</span>
                    </td>
                  </tr>
                ) : null}
              </Fragment>
            );
          })}
        </tbody>
      </table>
      <div className="row">
        <button className="primary" onClick={() => setShown(true)} disabled={picked === 0 || shown}>Reveal</button>
        {shown ? <span data-score={right}>You got {right} of {rows.length}.</span> : <span className="muted">{picked} of {rows.length} answered</span>}
        {shown ? <button onClick={() => { setPicks({}); setShown(false); }}>Start again</button> : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- softmax overflow
const COLS: { q: Quantity | 'x'; label: string }[] = [
  { q: 'x', label: 'logit x' },
  { q: 'z', label: 'z = x × scale' },
  { q: 'd', label: 'exp argument' },
  { q: 't', label: 'term exp(·)' },
  { q: 's', label: 'running sum' },
  { q: 'p', label: 'probability' },
];

function causeText(e: Event, f: Format, row: { d: number }): string {
  const max = show(maxFinite(f), 6);
  switch (e.cause) {
    case 'scaled-overflow':
      return `Logit ${e.index + 1} times the scale is past ${f.name}'s largest value, ${max}: it saturates to infinity.`;
    case 'inf-minus-inf':
      return `The maximum is infinite, and so is logit ${e.index + 1}: ∞ − ∞ is NaN.`;
    case 'exp-overflow':
      return `exp(${show(row.d, 6)}) is past ${max}: an exp argument above about ${overflowThreshold(f).toFixed(2)} overflows ${f.name}.`;
    case 'sum-overflow':
      return `Every term is finite, but the running sum passes ${max} at element ${e.index + 1}.`;
    case 'inf-over-inf':
      return `An infinite term divided by an infinite sum: ∞ / ∞ is NaN.`;
    case 'nan-propagated':
      return 'A NaN from an earlier step passes straight through.';
  }
}

const SCALE_MIN = -2;
const SCALE_MAX = 4;

function SoftmaxOverflow() {
  const [texts, setTexts] = useState<string[]>(SOFTMAX_PRESETS[0].logits.map(String));
  const [lg, setLg] = useState(0); // log2 of the scale
  const [fid, setFid] = useState('fp32');
  const [subtract, setSubtract] = useState(false);
  const f = SOFTMAX_FORMATS.find((x) => x.id === fid)!;
  const scale = 2 ** lg;
  const xs = texts.map((t) => Number(t.trim().replace('−', '-')));
  const valid = texts.every((t) => t.trim() !== '') && xs.every((v) => Number.isFinite(v));
  const tr = useMemo(() => (valid ? softmaxTrace(xs, scale, f, subtract) : null), [texts, scale, f, subtract, valid]); // eslint-disable-line react-hooks/exhaustive-deps
  const mark = (i: number, q: Quantity) =>
    tr?.firstInf && tr.firstInf.index === i && tr.firstInf.quantity === q ? 'inf' : tr?.firstNan && tr.firstNan.index === i && tr.firstNan.quantity === q ? 'nan' : '';
  const allZero = !!tr && tr.rows.every((r) => r.p === 0);
  return (
    <Bench
      title="Softmax overflow"
      task="Raise the scale in FP16 until something breaks. Then tick 'subtract the maximum'. Then try 'eight equal logits of 10' in FP16."
      foot={<>Each value is rounded to the chosen format as it is computed. The sum runs left to right, one addition at a time.</>}
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label="Format">
          {SOFTMAX_FORMATS.map((x) => <button key={x.id} aria-pressed={x.id === fid} onClick={() => setFid(x.id)}>{x.name}</button>)}
        </span>
        <label className="field" style={{ minWidth: '14rem' }}>
          <span data-scale={scale}>scale (1/temperature): {Number(scale.toPrecision(3))}</span>
          <input type="range" min={SCALE_MIN} max={SCALE_MAX} step={1 / 16} value={lg} aria-label="scale" onChange={(e) => setLg(Number(e.target.value))} />
        </label>
        <label className="check">
          <input type="checkbox" checked={subtract} onChange={(e) => setSubtract(e.target.checked)} /> subtract the maximum
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', gap: '0.4rem 0.5rem' }}>
        {SOFTMAX_PRESETS.map((p) => (
          <button key={p.id} className="quiet" data-preset={p.id} onClick={() => setTexts(p.logits.map(String))}>{p.label}</button>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.9rem', gap: '0.4rem' }}>
        {texts.map((t, i) => (
          <input key={i} type="text" value={t} aria-label={`logit ${i + 1}`} style={{ width: '4.6rem' }}
            onChange={(e) => setTexts(texts.map((v, k) => (k === i ? e.target.value : v)))} />
        ))}
      </div>
      {tr ? (
        <>
          <table className="data" style={{ background: 'var(--page)', margin: '0 0 0.7rem' }} data-first-inf={tr.firstInf ? `${tr.firstInf.index}:${tr.firstInf.quantity}` : ''}
            data-first-nan={tr.firstNan ? `${tr.firstNan.index}:${tr.firstNan.quantity}` : ''}>
            <thead>
              <tr><th style={{ paddingLeft: '0.6rem' }}>i</th>{COLS.map((c) => <th key={c.q} className="num">{c.q === 'd' && !subtract ? 'exp argument (= z)' : c.q === 'd' ? 'exp argument z − max' : c.label}</th>)}</tr>
            </thead>
            <tbody>
              {tr.rows.map((r, i) => (
                <tr key={i} data-i={i}>
                  <td style={{ paddingLeft: '0.6rem' }}>{i + 1}</td>
                  {COLS.map((c) => {
                    const v = r[c.q as keyof typeof r] as number;
                    const m = c.q === 'x' ? '' : mark(i, c.q as Quantity);
                    const bad = !Number.isFinite(v);
                    return (
                      <td key={c.q} className="num" data-q={c.q} data-v={String(v)} data-mark={m}
                        style={m ? { background: 'var(--fail-wash)', color: 'var(--fail)', fontWeight: 700 } : bad ? { color: 'var(--fail)' } : c.q === 't' && r.underflow ? { color: 'var(--ink-3)' } : undefined}>
                        {show(v, 6)}
                        {c.q === 't' && r.underflow ? <span className="muted"> (underflow)</span> : null}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          <div className={'narrate' + (tr.ok ? '' : ' bad')} role="status" data-ok={tr.ok ? 1 : 0}>
            {tr.ok ? (
              <>
                <span className="chip ok">fine</span> Every probability is finite and they sum to {show(tr.sumP, 8)}.
                {tr.rows.some((r) => r.underflow) ? ' The terms marked underflow were too small for the format and became 0: harmless here, they are tiny next to the largest term.' : ''}
              </>
            ) : (
              <>
                {tr.firstInf ? <div><b className="t-fail">First infinity:</b> {causeText(tr.firstInf, f, tr.rows[tr.firstInf.index])}</div> : null}
                {tr.firstNan ? <div><b className="t-fail">First NaN:</b> {causeText(tr.firstNan, f, tr.rows[tr.firstNan.index])} From there it spreads, and nothing raises an error.</div> : null}
                {!tr.firstNan && allZero ? <div><b className="t-fail">No NaN, and still wrong:</b> every probability is a finite term divided by an infinite sum, which is 0. They sum to 0, not 1.</div> : null}
              </>
            )}
          </div>
          <p className="muted" style={{ margin: '0.6rem 0 0' }}>
            In {f.name}, exp overflows for arguments above about {overflowThreshold(f).toFixed(2)}.{' '}
            {subtract ? 'With the maximum subtracted, every argument is at most 0, every term at most 1, and the sum at least 1.' : ''}
          </p>
        </>
      ) : (
        <p className="t-fail">Every logit must be a number.</p>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsSA3: Record<string, Question> = {
  'appA/A.3/q-intervals': {
    kind: 'numeric',
    id: 'appA/A.3/q-intervals',
    prompt: <>Double precision has 11 exponent bits where single has 8. How many times as many exponent codes, and so intervals of representable numbers, does double have?</>,
    answer: 8,
    unit: 'times',
    explain: (
      <p>
        2<sup>11</sup> / 2<sup>8</sup> = 2,048 / 256 = 8. Each extra exponent bit doubles the number of codes, so three extra
        bits multiply it by 2<sup>3</sup>. The book prints "2<sup>(2<sup>3</sup>)</sup> = 256 times", which is not what
        three extra bits give.
      </p>
    ),
    source: 'book §A.3, p. 590 (the printed 256 is an apparent erratum, notes/appA.md)',
  },
  'appA/A.3/q-nan-count': {
    kind: 'numeric',
    id: 'appA/A.3/q-nan-count',
    prompt: <>How many of the 2<sup>32</sup> FP32 bit patterns are NaN, counting both signs and both kinds?</>,
    answer: 16777214,
    unit: 'patterns',
    explain: (
      <>
        <p>
          A NaN has the exponent all ones and any mantissa except all zeros (that one is infinity): 2<sup>23</sup> − 1
          patterns per sign, 2 × (2<sup>23</sup> − 1) = 2<sup>24</sup> − 2 = 16,777,214 in all. NVIDIA writes the same as
          2<sup>(mantissa + 1)</sup> − 2.
        </p>
        <p>
          The book's 2<sup>22</sup> − 1 counts one kind for one sign: the signalling NaNs, whose top mantissa bit is 0 and
          whose other 22 bits must not all be 0. The quiet ones number 2<sup>22</sup> per sign. The classifier's table shows
          the split.
        </p>
      </>
    ),
    source: `book §A.3, p. 591; ${FACTS} A7 and E1`,
  },
  'appA/A.3/q-softmax-fp16': {
    kind: 'numeric',
    id: 'appA/A.3/q-softmax-fp16',
    prompt: <>A softmax computes each term exp(x) in FP16 and stores it in FP16, without subtracting the maximum. What is the smallest integer logit x whose term overflows?</>,
    answer: 12,
    explain: (
      <p>
        e<sup>11</sup> ≈ 59,874, which FP16 holds (as 59,872). e<sup>12</sup> ≈ 162,755, far past FP16's largest value,
        65,504: it becomes infinity. The boundary is ln(65,520) ≈ 11.09, where 65,520 is the largest value plus half a
        step. In FP32 and BF16 the boundary is about 88.72.
      </p>
    ),
    source: 'our arithmetic on the FP16 format (appA-facts A2); checked by app/src/sim/softmaxnum.test.ts',
  },
  'appA/A.3/q-nan-source': {
    kind: 'choice',
    id: 'appA/A.3/q-nan-source',
    prompt: <>A softmax kernel's output contains NaN, although every input logit was a finite number. What is the most likely cause?</>,
    options: [
      { label: 'An exp overflowed to infinity, and the division gave ∞ / ∞.', correct: true },
      { label: 'A denormal was flushed to zero.', why: 'Flushing turns a tiny value into 0. A 0 term gives a 0 probability, not a NaN.' },
      { label: 'One of the logits was negative.', why: 'exp of a negative number is a small positive number. Very negative ones underflow to 0, which is harmless.' },
      { label: 'Rounding errors in the sum.', why: 'Rounding errors make a sum slightly wrong; they never make it NaN.' },
    ],
    explain: (
      <p>
        Finite inputs produce a NaN only through one of the book's four undefined operations. In a softmax the candidates
        are ∞ / ∞ (an overflowed term over an overflowed sum) and ∞ − ∞ (an overflowed logit minus an overflowed maximum).
        Both start with an overflow, and nothing on a GPU reports it. Subtracting the row's maximum keeps every exp
        argument at or below 0, which removes the overflow.
      </p>
    ),
    source: 'book §A.3, pp. 590–591; the softmax reading is ours',
  },
  'appA/A.3/q-metal-nan': {
    kind: 'choice',
    id: 'appA/A.3/q-metal-nan',
    prompt: <>A Metal kernel compiled with the default options checks its result with <code>isnan</code> to catch a broken softmax. Can it rely on that check?</>,
    options: [
      {
        label: 'No. Under fast math the handling of NaN and INF is undefined, and Apple documents the default mode three different ways.',
        correct: true,
      },
      { label: 'Yes. isnan is a library function, so the compiler cannot change it.', why: 'Fast math lets the compiler assume there are no NaNs at all, and then isnan can be folded to false.' },
      { label: 'Yes, because Metal supports signalling NaNs.', why: '"Signaling NaNs are not supported" in Metal, and floating-point exceptions are disabled.' },
      { label: 'No, because Metal has no NaN at all.', why: 'Metal supports NaN for float, half and bfloat, with fast math disabled.' },
    ],
    explain: (
      <p>
        Apple's web documentation says the default on Apple silicon is relaxed (which honours INF and NaN), its compiler
        specification says fast, and the SDK header says it depends on the language version. Either select{' '}
        <code>safe</code> or <code>relaxed</code> explicitly, or keep the values finite so there is nothing to catch: lab
        10.3 starts its maximum at <code>-FLT_MAX</code> instead of <code>-INFINITY</code> for this reason.
      </p>
    ),
    source: `${FACTS} C2; ch05-facts C7; ch10-facts C3`,
  },
};

// ---------------------------------------------------------------- the page
export function SA3() {
  const formats = [FP32, FP16, BF16];
  return (
    <>
      <p>
        IEEE 754 fixes the field widths of its formats and sets aside some bit patterns for things that are not ordinary
        numbers. This section is about both: how much precision single and double precision carry, and which patterns
        mean zero, denormal, infinity and NaN. Knowing the second is how you read a kernel's output when it goes wrong.
      </p>

      <h2>Single and double</h2>
      <p>
        Single precision spends its 32 bits as 1 sign bit, 8 exponent bits and 23 mantissa bits. Double precision spends
        64 as 1, 11 and 52. The 29 extra mantissa bits are where double's precision comes from: each one halves the
        largest representation error, so double's is 2<sup>−29</sup> of single's. The 3 extra exponent bits are where its
        range comes from: double reaches about 10<sup>308</sup>, against about 10<sup>80</sup> atoms in the universe.
      </p>
      <Source section="A.3" pages="590" />
      <Aside>
        <p>
          <b>Erratum.</b> The book says the extra exponent bits multiply the number of intervals "by 2<sup>(2<sup>3</sup>)</sup> = 256
          times". Three extra bits give 2<sup>3</sup> = 8 times as many exponent codes.
        </p>
      </Aside>
      <Ask q={questionsSA3['appA/A.3/q-intervals']} label="Work it out" />
      <p>
        The formats you will use most for inference are narrower. FP16 and BF16 are both 16 bits, split differently: FP16
        keeps more mantissa, BF16 keeps FP32's exponent. That choice decides which of them overflows first, as the softmax
        below will show.
      </p>
      <Beyond>
        <p>FP16 and BF16 are not in the book. Their fields and limits are from NVIDIA's Table 51 and the OCP and Clang definitions (appA-facts A2).</p>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th>format</th><th className="num">sign</th><th className="num">exponent</th><th className="num">mantissa</th><th className="num">largest value</th><th className="num">smallest normal</th></tr>
        </thead>
        <tbody>
          {formats.map((g) => (
            <tr key={g.id}>
              <td>{g.name}{g.notInBook ? <span className="muted"> (not in the book)</span> : null}</td>
              <td className="num">1</td>
              <td className="num">{g.e}</td>
              <td className="num">{g.m}</td>
              <td className="num">{show(maxFinite(g), 5)}</td>
              <td className="num">2<sup>{Math.log2(minNormal(g))}</sup></td>
            </tr>
          ))}
          <tr>
            <td>FP64 (double)</td>
            <td className="num">1</td>
            <td className="num">11</td>
            <td className="num">52</td>
            <td className="num">about 1.8e308</td>
            <td className="num">2<sup>−1022</sup></td>
          </tr>
        </tbody>
      </table>

      <h2>Four special patterns</h2>
      <p>
        Two exponent codes are not used for ordinary numbers: all ones and all zeros. Together with whether the mantissa
        is zero, they make four groups (Fig. A.8):
      </p>
      <Source section="A.3" pages="590" figs="Fig. A.8" />
      <ul>
        <li>exponent all ones, mantissa 0: infinity, with the sign bit's sign;</li>
        <li>exponent all ones, mantissa not 0: NaN, "not a number";</li>
        <li>exponent all zeros, mantissa 0: zero (there are two, +0 and −0);</li>
        <li>exponent all zeros, mantissa not 0: a denormal, read as 0.M instead of 1.M (section <a href={href('appA', 'A.2')}>A.2</a>).</li>
      </ul>
      <p>
        Every other pattern is a normal number. Use the classifier to find each group, then try the narrow formats. They
        do not all follow Fig. A.8.
      </p>
      <Classifier />
      <p>
        FP8 E4M3 gives up infinity and keeps one NaN pattern per sign, S.1111.111. Its top exponent code then holds seven
        more numbers, up to 448. With Fig. A.8's rule, the top code would hold only infinity and NaN, and the largest E4M3
        value would be 1.875 × 2<sup>7</sup> = 240. E5M2 follows the IEEE layout. FP4 E2M1 has neither infinity nor NaN:
        all 16 of its patterns are numbers. The narrower the format, the less it can afford to spend on special values.
      </p>
      <Today checked={CHECKED('A3, A4')}>
        <p>
          OCP's FP8 specification: E4M3 "does not represent infinities and uses only two bit patterns for NaN ... in order
          to increase emax to 8 and thus to increase the dynamic range by one binade." In FP4 and FP6 "No encodings are
          reserved for Inf or NaN". The 240 is our arithmetic.
        </p>
      </Today>

      <h2>Infinity</h2>
      <p>
        A result too large for the format, such as a large number divided by a very small one, does not wrap around. It
        <b> saturates</b> at +∞ or −∞ and stays there through later arithmetic, except that any representable number
        divided by ±∞ is 0. Integer arithmetic does something else: a result out of range wraps around, or is undefined.
      </p>
      <Source section="A.3" pages="590" />
      <Today checked={CHECKED('A7')}>
        <p>
          NVIDIA: "operations that overflow the representable range result in +Infinity or -Infinity", and "Converting a
          floating-point value to an integer type for which the result is not representable is undefined behavior. This
          also includes NaN and infinity."
        </p>
      </Today>
      <p>
        That last sentence matters as soon as you quantise: a kernel that turns an activation into an int8 code with a cast
        has undefined behaviour if the activation has already become infinity.
      </p>

      <h2>NaN</h2>
      <p>
        A NaN is the result of an operation with no defined value. The book lists four: 0/0, 0 × ∞, ∞/∞ and ∞ − ∞. NaNs are
        also a convenient marker for data that was never initialised. There are two kinds, told apart by the top mantissa
        bit. With it clear, a <b>signalling</b> NaN: using one as an operand raises an exception, which is how you catch a
        read of uninitialised data. With it set, a <b>quiet</b> NaN: arithmetic on it returns another quiet NaN, silently,
        and it is printed as "NaN" for you to find afterwards.
      </p>
      <Source section="A.3" pages="590–591" />
      <WhatComesOut />

      <h2>GPUs do not signal</h2>
      <p>
        The book says that GPU hardware does not support signalling NaNs: raising exceptions precisely in the middle of
        thousands of threads is hard. It notes that the spare mantissa bits of a NaN could carry the cause of the problem,
        and counts 2<sup>22</sup> − 1 NaN patterns in single precision.
      </p>
      <Source section="A.3" pages="591" />
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th>Backend</th><th>Signalling NaN</th><th>Exceptions</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>CUDA</td>
            <td>"although signaling NaN SNaN encodings are supported, they are not signaling and are handled as quiet"</td>
            <td>"all operations behave as if IEEE-754 exceptions are always masked"</td>
          </tr>
          <tr>
            <td>HIP</td>
            <td>in IEEE mode, on GFX9 to GFX11, signalling NaN inputs are quieted and passed on (LLVM's AMDGPU documentation; HIP's own docs say nothing)</td>
            <td>Clang's default ignores floating-point exceptions</td>
          </tr>
          <tr>
            <td>Metal</td>
            <td>"Signaling NaNs are not supported."</td>
            <td>"Floating-point exceptions are disabled in Metal."</td>
          </tr>
        </tbody>
      </table>
      <p>
        So on every GPU you will program, a NaN travels silently from where it is made to the output. Nor can you count on
        its spare bits to say where it came from: NVIDIA warns that "Floating-point operations may alter the bit patterns
        of input NaN payloads."
      </p>
      <Today checked={CHECKED('A7, B6, C2, E1')}>
        <p>
          The book's 2<sup>22</sup> − 1 is the number of signalling NaNs of one sign. All FP32 NaNs together number
          2<sup>24</sup> − 2; NVIDIA writes 2<sup>(mantissa + 1)</sup> − 2.
        </p>
      </Today>
      <Ask q={questionsSA3['appA/A.3/q-nan-count']} label="Work it out" />

      <h2>Where infinities come from: softmax</h2>
      <p>
        In an inference kernel the classic source of infinity is the exponential in softmax. exp(x) passes FP16's largest
        value at x ≈ 11.09, and FP32's and BF16's at about 88.72. Nothing stops a logit from being that large, and a low
        temperature (a scale above 1) multiplies it further. One overflowed term makes the denominator infinite, the overflowed
        term's probability ∞/∞ = NaN, and every other probability 0. Nothing reports it.
      </p>
      <Beyond>
        <p>
          The book never mentions softmax. The pieces are its own (saturation, p. 590; ∞/∞ is NaN, p. 591; no signalling,
          p. 591); putting them together is ours.
        </p>
      </Beyond>
      <p>
        The fix is the one lab 10.3 already uses for milestone M4: subtract the row's maximum before the exponential. Then
        every argument is at most 0, every term at most 1, and the denominator at least 1, since the largest term is
        exp(0) = 1. Mathematically nothing changes, because the factor e<sup>−max</sup> cancels between numerator and
        denominator.
      </p>
      <Simplification>
        <p>
          exp is computed exactly and rounded once; real exp functions are a few ULPs off. The sum runs left to right.
          IEEE rounding to nearest, denormals kept. Under Metal's fast math the behaviour of INF and NaN is undefined, so
          there the outcome is not this table's.
        </p>
      </Simplification>
      <SoftmaxOverflow />
      <p>
        Two things to notice. Very negative logits underflow: their terms become 0, which is harmless next to a term of 1.
        And with eight logits of 10 in FP16, no term overflows (each is about 22,000), yet the sum does at the third one:
        every probability becomes 0 with no NaN at all.
      </p>
      <Ask q={questionsSA3['appA/A.3/q-softmax-fp16']} label="Work it out" />
      <Ask q={questionsSA3['appA/A.3/q-nan-source']} />
      <p>
        Lab 10.3's Metal kernel makes one more choice for the same reason: its maximum starts at <code>-FLT_MAX</code>,
        not <code>-INFINITY</code>. And a max reduction written with <code>fmaxf</code> drops a NaN logit instead of
        passing it on, so the NaN can disappear from the maximum and resurface in the sum (<a href={href('ch10', 'labs')}>chapter 10's labs</a>).
      </p>
      <Today checked="2026-11-20 (ch10-facts C3, A10; appA-facts C2)">
        <p>
          Metal: "If fast math is enabled the behavior of handling NaN or INF (as inputs or outputs) is undefined." CUDA's{' '}
          <code>fmaxf</code> "Treats NaN arguments as missing data"; Metal's <code>fmax</code> does the same.
        </p>
      </Today>
      <Ask q={questionsSA3['appA/A.3/q-metal-nan']} />
      <p>
        Section <a href={href('appA', 'A.4')}>A.4</a> turns from special values to ordinary ones: how far the result of
        an addition can be from the exact answer.
      </p>
    </>
  );
}
