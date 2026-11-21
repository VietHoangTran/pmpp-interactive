import { useEffect, useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Beyond, Bench, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import { useSolutionStatus } from './solutionStatus';
import {
  BOOK_KEYS,
  bin,
  lsdSort,
  parseKeys,
  promptKey,
  randomKeys,
  type SortPass,
  type SortRun,
} from '../../sim/radixsort';
import {
  PRESETS,
  TRANSFORM_LABEL,
  bitsOf,
  explore,
  hex,
  items,
  rawAscending,
  show,
  type DescBy,
  type Transform,
} from '../../sim/floatkey';
import { LsdPassesScene } from './LsdPassesScene';

// The key transform as the lesson shows it: the region "floatkey" of
// labs/ch14/snippets/s144_floatkey.cu, which is front-end-checked and run on the emulator
// against src/sim/floatkey.ts. S144.snippets.test.ts fails if this copy drifts.
const FLOATKEY = `__host__ __device__ inline unsigned int float_to_key(unsigned int u) {
    if(u == 0x80000000u) u = 0u;                    // -0.0 sorts as +0.0
    unsigned int mask = (u & 0x80000000u) ? 0xFFFFFFFFu  // negative: invert every bit
                                          : 0x80000000u; // otherwise: set the sign bit
    return u ^ mask;
}

__host__ __device__ inline unsigned int float_to_key_desc(unsigned int u) {
    return ~float_to_key(u);                        // descending: invert, sort ascending
}

__global__ void logits_to_keys(const unsigned int* logitBits, unsigned int* keys,
                               unsigned int n, bool descending) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) {
        unsigned int u = logitBits[i];              // the float's bits, never a float
        keys[i] = descending ? float_to_key_desc(u) : float_to_key(u);
    }
}`;
const SNIPPET_STATUS = 'front-end-checked; emulator-checked; unverified on CUDA and HIP hardware';
const LAB3_METAL = 'ch14/lab3_sortrows/solution/kernel.metal';
const loaded = (t: string | null) => (t && !t.startsWith('// could not load') ? t : null);

// ---------------------------------------------------------------- the LSD stepper
type KeysMode = 'book' | 'shuffle' | 'own';
const OWN_DEFAULT = '0101 0100 1110 0011 1010 0001 0110 1001';
const CW = 44;
const AX = 12;
const TOP = 30;
const BOT = 146;

const digitName = (p: SortPass) => (p.bits === 1 ? `bit ${p.shift}` : `bits ${p.shift + p.bits - 1} and ${p.shift}`);
const digitShort = (p: SortPass) => (p.bits === 1 ? `bit ${p.shift}` : `bits ${p.shift + p.bits - 1}–${p.shift}`);
const sortedText = (k: number) =>
  k === 4 ? 'sorted' : k === 0 ? 'not in order by its low bits' : `in order by its low ${['', 'bit', 'two bits', 'three bits'][k]}`;
/** A halo so that a label stays readable where arrows cross it. */
const HALO = { paintOrder: 'stroke', stroke: 'var(--bench)', strokeWidth: 4, strokeLinejoin: 'round' } as const;

/** What the passes so far order the list by, primary first (true only when all were stable). */
function orderedBy(run: SortRun, upto: number): string {
  const ds = run.passes.slice(0, upto).map(digitShort).reverse();
  if (ds.length === 1) return ds[0];
  return ds[0] + ', then ' + ds.slice(1).join(', then ') + ' among equals';
}

function describe(run: SortRun, k: number): string {
  const p = run.passes[k - 1];
  const n = p.input.length;
  const buckets = p.counts
    .map((c, d) => (c ? `${c} with ${bin(d, p.bits)} to positions ${p.starts[d]}–${p.starts[d] + c - 1}` : null))
    .filter(Boolean)
    .join('; ');
  let s = `Pass ${k} used ${digitName(p)}: ${buckets}.`;
  s += p.unstable ? ' Its top bucket was filled backwards.' : ' Each bucket kept the order its keys arrived in.';
  const anyUnstable = run.passes.slice(0, k).some((q) => q.unstable);
  if (!anyUnstable && run.order === 'msd')
    s += ` The last pass has the final word: the list is now ordered by ${orderedBy(run, k)}. It is ${sortedText(p.sortedBy)}.`;
  else s += ` The list of ${n} keys is now ${sortedText(p.sortedBy)}.`;
  return s;
}

function KeyBox({ x, y, k, mark, markBits = 1, shade = 0, hot = false, ring = false, bad = false, ghost = false, row, slot }: {
  x: number; y: number; k: number; mark?: number; markBits?: number; shade?: number; hot?: boolean; ring?: boolean; bad?: boolean; ghost?: boolean;
  row: string; slot: number;
}) {
  const s = bin(k, 4);
  if (ghost) return <rect x={x} y={y} width={CW - 6} height={26} rx="2" fill="none" stroke="var(--rule)" strokeDasharray="3 3" />;
  return (
    <g data-row={row} data-slot={slot} data-key={s} data-hot={hot ? '1' : undefined} data-ring={ring ? '1' : undefined} data-bad={bad ? '1' : undefined}>
      <rect x={x} y={y} width={CW - 6} height={26} rx="2"
        fill={bad ? 'var(--fail-wash)' : hot ? 'var(--ink)' : 'var(--cell)'} stroke={bad ? 'var(--fail)' : hot || ring ? 'var(--ink)' : 'var(--rule)'}
        strokeWidth={ring && !hot ? 2.4 : 1} />
      {shade > 0 && !hot ? <rect x={x + 3 + (4 - shade) * 8} y={y + 3} width={shade * 8} height={20} rx="1" fill="var(--rule)" /> : null}
      {s.split('').map((ch, c) => {
        const bitIx = 3 - c;
        const isMark = mark != null && bitIx >= mark && bitIx < mark + markBits;
        return (
          <g key={c}>
            <text x={x + 7 + c * 8} y={y + 17.5} fontSize="12.5" textAnchor="middle" fontWeight={isMark ? 700 : 400}
              style={{ fill: hot ? 'var(--page)' : bad ? 'var(--fail)' : 'var(--ink)' }}>{ch}</text>
            {isMark ? <rect x={x + 3.5 + c * 8} y={y + 21} width={7} height={1.8} style={{ fill: hot ? 'var(--page)' : 'var(--ink)' }} /> : null}
          </g>
        );
      })}
    </g>
  );
}

/** askId: the key the next question is about; lastId: the key the last question was about. */
function PassPicture({ run, at, askId, lastId }: { run: SortRun; at: number; askId: number | null; lastId: number | null }) {
  const n = run.keys.length;
  const w = AX * 2 + n * CW;
  const cx = (j: number) => AX + j * CW + (CW - 6) / 2;
  const p = at > 0 ? run.passes[at - 1] : null;
  const next = at < run.passes.length ? run.passes[at] : null;
  const topKeys = p ? p.input : run.keys;
  const topIds = p ? p.inIds : run.keys.map((_, i) => i);
  const topMark = p ?? next!;
  const done = at === run.passes.length;
  const badFrom = done && !run.sorted ? run.firstDisorder : -1;
  const topBucket = p ? p.counts.length - 1 - [...p.counts].reverse().findIndex((c) => c > 0) : -1;
  return (
    <svg viewBox={`0 0 ${w} 236`} role="img" aria-label="Keys before and after the current pass" style={{ width: '100%', maxWidth: w, minWidth: Math.min(w, 560), display: 'block' }}>
      <defs>
        <marker id="lsdst-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--ink-2)' }} />
        </marker>
        <marker id="lsdst-arrow-bad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--fail)' }} />
        </marker>
      </defs>
      <text x={AX} y={TOP - 9} fontSize="11.5" fontWeight="600" style={HALO}>
        {p ? (p.index === 1 ? 'input' : `after pass ${p.index - 1}`) + `, ${digitShort(p)} marked` : `input, ${digitShort(next!)} marked: pass 1 reads it`}
      </text>
      {topKeys.map((k, j) => (
        <KeyBox key={j} x={AX + j * CW} y={TOP} k={k} mark={topMark.shift} markBits={topMark.bits} row="top" slot={j}
          hot={!p && askId === topIds[j]} />
      ))}
      {p
        ? p.dst.map((d, i) => {
            const bad = p.unstable && ((p.input[i] >>> p.shift) & ((1 << p.bits) - 1)) === topBucket;
            const hot = lastId != null && p.inIds[i] === lastId;
            return (
              <line key={i} data-arrow={i} data-to={d} x1={cx(i)} y1={TOP + 28} x2={cx(d)} y2={BOT - 3}
                stroke={bad ? 'var(--fail)' : hot ? 'var(--ink)' : 'var(--ink-2)'} strokeWidth={hot ? 2.4 : bad ? 1.5 : 1}
                opacity={hot || bad ? 1 : 0.75} markerEnd={bad ? 'url(#lsdst-arrow-bad)' : 'url(#lsdst-arrow)'} />
            );
          })
        : null}
      <text x={AX} y={BOT - 9} fontSize="11.5" fontWeight="600" opacity={p ? 1 : 0.5} style={HALO}>
        {p ? `after pass ${p.index}${next ? `; next: ${digitShort(next)}` : ''}` : 'after pass 1'}
      </text>
      {(p ? p.output : topKeys).map((k, j) => (
        <KeyBox key={j} x={AX + j * CW} y={BOT} k={k} row="bottom" slot={j} ghost={!p}
          mark={next && p ? next.shift : undefined} markBits={next?.bits}
          shade={p && !next ? p.sortedBy : 0}
          hot={!!p && askId === p.outIds[j]} ring={!!p && lastId === p.outIds[j]} bad={badFrom >= 0 && (j === badFrom || j === badFrom + 1)} />
      ))}
      {p
        ? p.counts.map((c, d) =>
            c ? (
              <g key={d} data-bucket={d}>
                <line x1={AX + p.starts[d] * CW + 2} x2={AX + (p.starts[d] + c) * CW - 8} y1={BOT + 36} y2={BOT + 36} stroke="var(--ink-2)" />
                <text x={AX + (p.starts[d] * CW + (p.starts[d] + c) * CW) / 2 - 3} y={BOT + 50} fontSize="11" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
                  {p.bits === 1 ? `${digitShort(p)} = ${d}` : bin(d, p.bits)}
                </text>
              </g>
            ) : null,
          )
        : null}
      {p ? (
        <text x={w - AX} y={BOT + 72} fontSize="12" textAnchor="end" data-sorted-by={p.sortedBy} style={{ fill: 'var(--ink-2)' }}>
          {sortedText(p.sortedBy)}
        </text>
      ) : null}
    </svg>
  );
}

function LsdStepper() {
  const [mode, setMode] = useState<KeysMode>('book');
  const [seed, setSeed] = useState(7);
  const [ownText, setOwnText] = useState(OWN_DEFAULT);
  const [radix, setRadix] = useState(1);
  const [unstable, setUnstable] = useState(0);
  const [msd, setMsd] = useState(false);
  const [at, setAt] = useState(0);
  const [guess, setGuess] = useState('');
  const [last, setLast] = useState<{ pass: number; guess: number | null; id: number; key: number; landed: number } | null>(null);

  const parsed = useMemo(() => (mode === 'own' ? parseKeys(ownText, 4, 8) : null), [mode, ownText]);
  const keys = useMemo(
    () => (mode === 'book' ? BOOK_KEYS : mode === 'shuffle' ? randomKeys(16, 4, seed) : parsed && 'keys' in parsed ? parsed.keys : BOOK_KEYS.slice(0, 8)),
    [mode, seed, parsed],
  );
  const passes = 4 / radix;
  const unstablePass = unstable > passes ? 0 : unstable;
  const run = useMemo(() => lsdSort(keys, 4, radix, { order: msd ? 'msd' : 'lsd', unstablePass: unstablePass || null }), [keys, radix, msd, unstablePass]);
  useEffect(() => {
    setAt(0);
    setLast(null);
    setGuess('');
  }, [run]);

  const next = at < run.passes.length ? run.passes[at] : null;
  const promptPos = next ? promptKey(next) : null;
  const askId = next && promptPos != null ? next.inIds[promptPos] : null;

  const step = () => {
    if (!next || promptPos == null) return;
    const g = guess.trim() === '' ? null : Number(guess);
    setLast({ pass: next.index, guess: g != null && !Number.isNaN(g) ? g : null, id: next.inIds[promptPos], key: next.input[promptPos], landed: next.dst[promptPos] });
    setGuess('');
    setAt((x) => Math.min(run.passes.length, x + 1));
  };

  const done = at === run.passes.length;
  return (
    <Bench
      title="Radix sort, pass by pass"
      task="Before each pass, say where the highlighted key will land. Then break a pass and watch the result."
      foot={
        <>
          Keys are 4 bits. The underlined bits are the digit the pass reads; the shaded low bits of the final row are
          the ones the list is in order by. Model: <code>src/sim/radixsort.ts</code> (its tests reproduce every row of
          Figs. 14.4 and 14.10).
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Keys
          <select value={mode} onChange={(e) => setMode(e.target.value as KeysMode)}>
            <option value="book">the book's sixteen (Fig. 14.4)</option>
            <option value="shuffle">sixteen random keys</option>
            <option value="own">your own eight keys</option>
          </select>
        </label>
        {mode === 'shuffle' ? <button onClick={() => setSeed(seed + 1)}>Shuffle</button> : null}
        {mode === 'own' ? (
          <label className="field">
            Eight keys, binary or 0 to 15
            <input type="text" value={ownText} aria-label="your eight keys" onChange={(e) => setOwnText(e.target.value)} style={{ width: '19rem' }} />
          </label>
        ) : null}
        <label className="field">
          Radix width
          <span className="seg" role="group" aria-label="Radix width">
            {[1, 2].map((r) => (
              <button key={r} aria-pressed={radix === r} onClick={() => setRadix(r)}>{r} bit{r > 1 ? 's' : ''}</button>
            ))}
          </span>
        </label>
        <label className="field">
          Make a pass unstable
          <select value={unstablePass} onChange={(e) => setUnstable(Number(e.target.value))} aria-label="Make a pass unstable">
            <option value={0}>none</option>
            {Array.from({ length: passes }, (_, k) => (
              <option key={k} value={k + 1}>pass {k + 1}</option>
            ))}
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={msd} onChange={(e) => setMsd(e.target.checked)} /> most significant bit first
        </label>
      </div>
      {parsed && 'error' in parsed ? <p className="t-fail" role="alert">Using the book's first eight keys: {parsed.error}.</p> : null}

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setAt(0); setLast(null); }} disabled={at === 0}>Restart</button>
        <button onClick={() => { setAt((x) => Math.max(0, x - 1)); setLast(null); }} disabled={at === 0}>Back</button>
        <button className="primary" onClick={step} disabled={!next}>Run pass {Math.min(at + 1, run.passes.length)}</button>
        <button onClick={() => { setAt(run.passes.length); setLast(null); }} disabled={done}>Run all</button>
        <span className="muted" aria-live="polite">{at} of {run.passes.length} passes</span>
      </div>

      {next && promptPos != null ? (
        <div className="row" style={{ marginBottom: '0.8rem' }} data-part="prompt">
          <span>
            Pass {next.index} reads {digitName(next)}. The highlighted key <b>{bin(next.input[promptPos], 4)}</b> is at
            position {promptPos}. Where will it land?
          </span>
          <input type="number" min={0} max={run.keys.length - 1} value={guess} aria-label="your guess"
            onChange={(e) => setGuess(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && step()} />
        </div>
      ) : null}

      <div className={'narrate' + (done && !run.sorted ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {at === 0 ? (
          <>Nothing has run. {run.passes.length} passes of {radix} bit{radix > 1 ? 's' : ''} will cover the 4-bit keys, {msd ? 'most' : 'least'} significant digit first.</>
        ) : (
          <>
            {last && last.pass === at ? (
              <>
                The key you were asked about, {bin(last.key, 4)} (outlined), landed at position <b>{last.landed}</b>
                {last.guess != null ? (last.guess === last.landed ? <>; you said {last.guess}: <span className="t-ok">right</span>. </> : <>; you said {last.guess}. </>) : '. '}
              </>
            ) : null}
            {describe(run, at)}
          </>
        )}
      </div>

      <PassPicture run={run} at={at} askId={askId} lastId={last && last.pass === at ? last.id : null} />

      {done ? (
        <p style={{ margin: '0.6rem 0 0' }} data-part="outcome">
          {run.sorted ? <span className="chip ok">sorted</span> : <span className="chip fail">not sorted</span>}{' '}
          {!run.sorted ? (
            <>
              {bin(run.result[run.firstDisorder], 4)} at position {run.firstDisorder} comes before{' '}
              {bin(run.result[run.firstDisorder + 1], 4)}.
            </>
          ) : !run.stable ? (
            <>
              but two equal keys ({bin(run.result[run.firstTieSwap], 4)}, positions {run.firstTieSwap} and {run.firstTieSwap + 1}) are out of
              their input order. A list of bare keys cannot show it; a list of (logit, token id) pairs would.
            </>
          ) : (
            'Equal keys are in their input order too.'
          )}
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- the float explorer
function FloatExplorer() {
  const [presetId, setPresetId] = useState('ties');
  const [tf, setTf] = useState<Transform>('none');
  const [zero, setZero] = useState(false);
  const [descBy, setDescBy] = useState<DescBy>('invert');
  const pr = PRESETS.find((p) => p.id === presetId)!;
  const xs = useMemo(() => items(pr.bits), [pr]);
  const r = useMemo(() => explore(xs, { transform: tf, zero, descBy }), [xs, tf, zero, descBy]);
  const label = (token: number) => pr.labels?.[token] ?? show(pr.bits[token]);
  const ok = r.firstMismatch < 0;
  return (
    <Bench
      title="Logits to keys"
      task="Find the settings that make every preset match the reference, and say what each one fixes."
      foot={
        <>
          Sorted largest first by a stable sort of the keys. Reference: {r.rule === "CUB's bits rule" ? "CUB's rule (the full transform, the −0.0 step, inverted; NaN by its bits)" : 'std::stable_sort with > on the values'}.
          Model: <code>src/sim/floatkey.ts</code> (its tests check E1's table bit for bit).
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Row of logits
          <select value={presetId} onChange={(e) => setPresetId(e.target.value)}>
            {PRESETS.map((p) => (
              <option key={p.id} value={p.id}>{p.label}</option>
            ))}
          </select>
        </label>
        <label className="field">
          Transform
          <span className="seg" role="group" aria-label="Transform">
            {(['none', 'sign', 'full'] as Transform[]).map((t) => (
              <button key={t} aria-pressed={tf === t} onClick={() => setTf(t)} title={TRANSFORM_LABEL[t]}>
                {t === 'none' ? 'raw bits' : t === 'sign' ? 'flip sign bit' : 'full'}
              </button>
            ))}
          </span>
        </label>
        <label className="check">
          <input type="checkbox" checked={zero} onChange={(e) => setZero(e.target.checked)} /> −0.0 step
        </label>
        <label className="field">
          Descending by
          <span className="seg" role="group" aria-label="Descending by">
            {(['invert', 'reverse'] as DescBy[]).map((d) => (
              <button key={d} aria-pressed={descBy === d} onClick={() => setDescBy(d)}>
                {d === 'invert' ? 'inverting the keys' : 'reversing an ascending sort'}
              </button>
            ))}
          </span>
        </label>
      </div>
      <table className="threads" aria-label="Each logit and its key" style={{ marginBottom: '0.9rem' }}>
        <thead>
          <tr>
            <th>token</th><th>value</th><th>bits</th><th>after −0.0 step</th><th>key</th><th>sorted ascending by</th>
          </tr>
        </thead>
        <tbody>
          {r.rows.map((row) => (
            <tr key={row.token} data-token={row.token}>
              <td>{row.token}</td>
              <td>{label(row.token)}</td>
              <td className="mono">{hex(row.bits)}</td>
              <td className="mono" style={{ color: zero && row.stepped !== row.bits ? 'var(--ink)' : 'var(--ink-3)' }}>{hex(row.stepped)}</td>
              <td className="mono">{hex(row.key)}</td>
              <td className="mono" data-sortkey={hex(row.sortKey)}>{hex(row.sortKey)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mx-label">Sorted, largest first</div>
      <div className="mx" style={{ display: 'flex', flexWrap: 'wrap', marginBottom: '0.5rem' }} data-part="output">
        {r.order.map((t, k) => (
          <span key={k} className={r.match[k] ? 'good' : 'bad'} data-token={t} style={{ padding: '0 0.35rem', minWidth: '3.4rem' }}>
            {show(pr.bits[t])}
            <sub>{t}</sub>
          </span>
        ))}
      </div>
      <div className="mx-label">Reference</div>
      <div className="mx" style={{ display: 'flex', flexWrap: 'wrap', marginBottom: '0.8rem' }} data-part="reference">
        {r.reference.map((t, k) => (
          <span key={k} data-token={t} style={{ padding: '0 0.35rem', minWidth: '3.4rem' }}>
            {show(pr.bits[t])}
            <sub>{t}</sub>
          </span>
        ))}
      </div>
      <p style={{ margin: 0 }} data-part="verdict">
        {ok ? (
          <span className="chip ok">matches the reference</span>
        ) : (
          <>
            <span className="chip fail">differs from the reference</span> first at position {r.firstMismatch}: token{' '}
            {r.order[r.firstMismatch]} where the reference has token {r.reference[r.firstMismatch]}.
          </>
        )}{' '}
        <span className="muted">Small numbers are token ids.</span>
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS144: Record<string, Question> = {
  'ch14/14.4/q-pass2': {
    kind: 'numeric',
    id: 'ch14/14.4/q-pass2',
    prompt: (
      <>
        Take the book's sixteen keys, <code>1100 0011 0110 1001 1111 1000 0101 1010 1001 0110 1011 1101 0100 1010 0111
        0000</code>. After pass 2 (bits 0 and 1), at which position, counting from 0, is the first <code>0110</code>?
      </>
    ),
    answer: 8,
    explain: (
      <p>
        After pass 2 the list is in order by its low two bits. Four keys end in 00 and four in 01, so the keys ending
        in 10 start at position 8, in the order pass 1 left them; <code>0110</code> is the first of them there (pass 1
        output: 1100 <b>0110</b> 1000 1010 …). The stepper shows it.
      </p>
    ),
    source: 'ch. 14 §14.4, pp. 335–336, Fig. 14.4',
  },
  'ch14/14.4/q-msb': {
    kind: 'choice',
    id: 'ch14/14.4/q-msb',
    prompt: <>Run the same four stable 1-bit passes from the most significant bit down (bit 3, then 2, 1, 0). What is the result in order by?</>,
    options: [
      { label: 'By the key: every pass is stable, so the order of the passes makes no difference.', why: 'Each stable pass makes its digit the most important one so far. The last pass wins.' },
      { label: 'By nothing in particular: without the low bits first, the result is random.', why: 'It is fully determined, and sorted, just by the wrong key.' },
      {
        label: 'By the bit-reversed key: bit 0, read by the last pass, becomes the primary key.',
        correct: true,
      },
    ],
    explain: (
      <p>
        A stable pass keeps the earlier order only among keys with equal digits, so whatever the last pass reads
        dominates. Least significant first puts the most significant digit last, where it belongs. Tick "most
        significant bit first" in the stepper: every even key ends up before every odd key. (Section 14.9's MSD radix
        sort is different: it sorts each bucket separately instead of the whole list each pass.)
      </p>
    ),
    source: 'our analysis of ch. 14 §14.4, p. 335; §14.9, p. 346',
  },
  'ch14/14.4/q-negatives': {
    kind: 'choice',
    id: 'ch14/14.4/q-negatives',
    prompt: (
      <>
        −1.0 is <code>0xBF800000</code> and −2.0 is <code>0xC0000000</code>. Flip only the sign bit of every value and
        sort ascending as unsigned integers. What happens to these two?
      </>
    ),
    options: [
      { label: 'They come out in the right order: with the sign bit flipped, both read as keys ordered by magnitude.', why: 'Their keys become 0x3F800000 and 0x40000000: −2.0 has the larger key.' },
      {
        label: '−2.0 still lands above −1.0: a negative\'s magnitude bits must be inverted too.',
        correct: true,
      },
      { label: 'Both land above every positive value, because their exponent bits are larger.', why: 'Flipping the sign bit fixed that part: negatives now start with 0.' },
    ],
    explain: (
      <p>
        Flipping the sign bit puts every negative below every positive, but a negative's remaining bits still grow with
        its magnitude, so among negatives the order is backwards. Inverting all 32 bits of a negative fixes both at
        once: −2.0 becomes <code>0x3FFFFFFF</code> and −1.0 <code>0x407FFFFF</code>.
      </p>
    ),
    source: 'notes/today/ch14-facts.md A3, E1',
  },
  'ch14/14.4/q-ties-desc': {
    kind: 'choice',
    id: 'ch14/14.4/q-ties-desc',
    prompt: (
      <>
        Tokens 3 and 7 have exactly equal probability. The row is sorted descending by sorting ascending, stably, and
        then reversing the result. Which of the two comes first?
      </>
    ),
    options: [
      { label: 'Token 7: reversing the whole list reverses the order of the ties too.', correct: true },
      { label: 'Token 3: the stable ascending sort put token 3 first, and a reversal keeps the order of ties.', why: 'It did, until the reversal turned the whole list round, ties included.' },
      { label: 'Either: as with any order among equal keys, it depends on the run.', why: 'Both steps are deterministic.' },
    ],
    explain: (
      <p>
        Inverting the keys and sorting ascending gives descending order with ties still in input order, token 3 first.
        That is CUB's rule ("the keys are inverted ... and the resulting keys are sorted in ascending order") and what{' '}
        <code>std::stable_sort</code> with <code>&gt;</code> gives. Try both settings on the "logits with ties" row.
      </p>
    ),
    source: 'notes/today/ch14-facts.md A3, E1',
  },
  'ch14/14.4/q-zero': {
    kind: 'choice',
    id: 'ch14/14.4/q-zero',
    prompt: (
      <>
        Without the −0.0 step, a row holds −0.0 at token 2 and +0.0 at token 9. Sorted descending (keys inverted,
        stable ascending sort), which comes first?
      </>
    ),
    options: [
      { label: '−0.0 (token 2): equal values keep their input order in a stable sort.', why: 'They are equal as floats, but their keys differ without the step.' },
      { label: 'It depends on the GPU\'s float compare, which may or may not treat the two zeros as equal.', why: 'The transform is integer arithmetic; it gives the same keys everywhere.' },
      {
        label: '+0.0 (token 9): its key, 0x80000000, is above −0.0\'s 0x7FFFFFFF.',
        correct: true,
      },
    ],
    explain: (
      <p>
        With the step, −0.0 becomes +0.0 before the transform, both get the key <code>0x80000000</code>, and the stable
        sort keeps token 2 first, as <code>std::stable_sort</code> with <code>&gt;</code> does. CUB and rocPRIM take this
        step; llama.cpp's top-k keys and PyTorch's CUDA and MPS keys do not, by our reading of their source.
      </p>
    ),
    source: 'notes/today/ch14-facts.md A3, B2, D2, D5, D6, E1',
  },
};

const RAW = [-2.0, -1.0, 0.5, 1.0];

export function S144() {
  const q = questionsS144;
  const metal = loaded(useLabFile(LAB3_METAL));
  const lab3Status = useSolutionStatus('ch14/lab3_sortrows');
  const metalLines = metal
    ? metal.split('\n').flatMap((l, k) => (/as_type<\s*uint\s*>|0x80000000u|~key\b/.test(l) ? [k + 1] : []))
    : [];
  const raw = rawAscending(RAW);
  return (
    <>
      <p>
        Every sort so far in this chapter compares keys. Radix sort never does. It looks at one digit of every key,
        deals the keys into buckets by that digit, and repeats with the next digit until it has used them all. Not
        comparing is what lets it beat the O(N log N) bound of comparison sorts.
      </p>
      <Source section="14.4" pages="334–335" />
      <p>
        With binary keys the convenient radix is a power of two, so that a digit is a fixed slice of bits. The chapter
        starts with the smallest: a 1-bit radix, two buckets per pass.
      </p>
      <Source pages="335" />

      <h2>Four passes over four bits</h2>
      <p>
        The book's example sorts sixteen 4-bit keys from the least significant bit up. Pass 1 sends every key whose
        bit 0 is 0 to the front and every key whose bit 0 is 1 to the back, each group in the order it arrived. Pass 2
        does the same with bit 1, on pass 1's output; then bit 2, then bit 3. After pass k the list is in order by its
        low k bits, so after pass 4 it is sorted.
      </p>
      <Source section="14.4" pages="335–336" figs="Fig. 14.4" />
      <Ask q={q['ch14/14.4/q-pass2']} label="Work it out, then check it in the stepper" />
      <p>
        Before each pass the stepper highlights a key and asks where it will land. Count the keys that go to earlier
        buckets, then the keys of its own bucket that are ahead of it.
      </p>
      <Simplification>
        <p>
          The stepper moves whole keys pass by pass. It says nothing about threads or memory: sections 14.5 and 14.6
          do. An "unstable" pass here fills its top bucket backwards, one specific way of not being stable.
        </p>
      </Simplification>
      <LsdStepper />

      <h2>Why every pass must be stable</h2>
      <p>
        Pass 2 can only build on pass 1 if it keeps what pass 1 did. Two keys with the same bit 1 were put in order by
        bit 0 in pass 1; a pass that keeps each bucket in arrival order leaves them that way. That is the whole
        correctness argument: a stable pass carries forward the order of everything before it, and its own digit
        becomes the most significant one sorted so far.
      </p>
      <Source section="14.4" pages="335" />
      <LsdPassesScene />
      <p>
        In the stepper, make pass 4 unstable and run all the passes. Then make pass 1 unstable instead. The keys come
        out sorted, because pass 1 had no earlier order to lose; only the two <code>1001</code>s swap places, which a
        list of bare keys cannot show and a list of (logit, token id) pairs would. A sampler needs that stability for a
        second reason, ours rather than the book's: equal probabilities then come out in a fixed, reproducible
        order.
      </p>
      <p>
        The order of the passes matters for the same reason. The last pass has the final word, so it must read the most
        significant digit: least significant first. Tick "most significant bit first" and predict the result before you
        run it.
      </p>
      <Ask q={q['ch14/14.4/q-msb']} />
      <p>
        A wider digit means fewer passes. With 2 bits per pass, four buckets each, the same keys sort in two passes;
        switch the stepper to 2 bits and compare with Fig. 14.10. Section 14.7 weighs what the wider digit costs.
      </p>
      <Source section="14.7" pages="342" figs="Fig. 14.10" />

      <h2>A pass is a stable partition</h2>
      <p>
        Chapter 12's stable filter moves the keys that pass a test to the front of the output, in input order, and
        drops the rest. A radix pass keeps the rest too: keys whose bit is 0 go to the front in input order, keys whose
        bit is 1 go to the back in input order. The book calls this the stable partition pattern, and its
        parallelisation and optimisation follow the stable filter's closely. In particular, every key's destination
        comes from one exclusive scan of the bits: <a href={href('ch14', '14.5')}>section 14.5</a>.
      </p>
      <Source section="14.4" pages="336" />

      <h2>Sorting logits</h2>
      <p>
        Radix sort works on unsigned integers. A sampler sorts logits or probabilities, which are floats, and wants the
        largest first. Reading a float's 32 bits as an unsigned integer costs nothing. The question is whether those
        integers come out in the floats' order.
      </p>
      <Beyond>
        <p>
          From here to the end of the page is ours. The book only warns that non-comparison sorts "may not generalize
          to arbitrary types of keys" (p. 330).
        </p>
      </Beyond>
      <Predict
        id="ch14/14.4/p-raw-order"
        prompt={
          <>
            Sort −2.0, −1.0, 0.5 and 1.0 by their raw bits, read as unsigned integers, ascending. At which position
            (0 to 3) does −1.0 end up?
          </>
        }

      >
        {(v) => (
          <>
            <p>
              It ends up at position <b>{raw.indexOf(-1)}</b>{v === raw.indexOf(-1) ? ', as you said' : ''}. The sorted
              bits: {raw.map((x) => `${show(bitsOf(x))} = ${hex(bitsOf(x))}`).join(', ')}.
            </p>
            <p>
              Positive floats are fine among themselves: the exponent sits above the mantissa, so a larger magnitude has
              larger bits. But every negative float has its top bit set, so all of them come after all the positives,
              and among themselves they run backwards, because their magnitudes still grow with their bits.
            </p>
          </>
        )}
      </Predict>
      <p>
        The fix has two rules. Set the sign bit of a non-negative value, so that it lands above every negative. Invert
        every bit of a negative value, so that a larger magnitude gives a smaller key. Read as unsigned integers, the
        results are in the order of the floats, infinities and subnormals included.
      </p>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md, A3, B2, E1)">
        <p>
          CUB documents exactly this for float keys: "For positive floating point values, the sign bit is inverted. For
          negative floating point values, the full key is inverted." rocPRIM uses the same transform.
        </p>
      </Today>
      <p>A sampler needs two more steps, and one decision.</p>
      <ul>
        <li>
          <b>The −0.0 step.</b> −0.0 and +0.0 are equal as floats, but their bits differ, and the transform keeps them
          apart: −0.0 becomes <code>0x7FFFFFFF</code>, +0.0 becomes <code>0x80000000</code>. So +0.0 always sorts
          above −0.0, whatever their order in the row. Map −0.0 to +0.0 before extracting digits.
        </li>
        <li>
          <b>Descending by inversion.</b> Get the largest first by inverting the key and sorting ascending, not by
          reversing an ascending result. Reversal also reverses the order of ties.
        </li>
        <li>
          <b>NaN.</b> CUB applies no special rule: after the transform +NaN sorts above +inf and −NaN below −inf. MLX
          puts NaN at the end, and PyTorch gives NaN a fixed key of its own. This course's labs use CUB's rule, and the
          explorer takes it as the reference for rows with NaN.
        </li>
      </ul>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md, A3, B2, D2, D4 to D6)">
        <p>
          CUB and rocPRIM take the −0.0 step. llama.cpp's top-k keys and PyTorch's CUDA and MPS keys do not, by our
          reading of their source. MLX documents "NaN values are placed at the end"; PyTorch's key functions test for
          NaN and return a constant.
        </p>
      </Today>
      <FloatExplorer />
      <Ask q={q['ch14/14.4/q-negatives']} />
      <Ask q={q['ch14/14.4/q-ties-desc']} />
      <Ask q={q['ch14/14.4/q-zero']} />
      <p>
        Here is the transform as a device function, with a kernel that applies it to a row. It takes the float's bits
        as an <code>unsigned int</code> and never holds a float, for reasons the next table gives.
      </p>
      <Code
        tabs={[{ label: 'CUDA (and HIP)', code: FLOATKEY, status: SNIPPET_STATUS }]}
        caption={
          <>
            From <code>labs/ch14/snippets/s144_floatkey.cu</code>. Its unit test runs it on the course's CPU emulator over
            E1's values, two NaNs and 600 other bit patterns and compares every key with the model, bit for bit.
          </>
        }
      />

      <h2>On each backend</h2>
      <p>
        Getting at a float's bits is the only part that differs, and on Metal the float rules decide how to do it.
      </p>
      <table className="data wide" style={{ maxWidth: '58rem' }} data-part="backends">
        <thead>
          <tr><th>Backend</th><th>The float's bits</th><th>What to watch</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>CUDA</td>
            <td><code>__float_as_uint(x)</code>, or <code>cuda::std::bit_cast&lt;unsigned int&gt;(x)</code></td>
            <td>Both copy the bits unchanged (ch14-facts A8).</td>
          </tr>
          <tr>
            <td>HIP</td>
            <td><code>__float_as_uint(x)</code>, same spelling</td>
            <td>It is in HIP's device header but not in AMD's published math reference (B6).</td>
          </tr>
          <tr>
            <td>Metal</td>
            <td><code>as_type&lt;uint&gt;(x)</code>: "The bits in the operand are returned directly without modification"</td>
            <td>
              Fast and relaxed math both allow "no signed zeros", and fast math leaves NaN and INF undefined. Do the
              −0.0 test and the transform on the <code>uint</code>, or bind the buffer as <code>device const uint*</code>{' '}
              and never hold a float (C3).
            </td>
          </tr>
          <tr>
            <td>The labs</td>
            <td>read the float buffer through an unsigned view on every backend</td>
            <td>It needs no bit-cast intrinsic at all, and on Metal it keeps every test on the integer side.</td>
          </tr>
        </tbody>
      </table>
      {metal ? (
        <Code
          tabs={[{ label: 'Metal (lab 14.3)', code: metal, highlight: metalLines, status: lab3Status.metal }]}
          caption={
            <>
              Lab 14.3's Metal kernel, loaded from <code>labs/{LAB3_METAL}</code>. The marked lines are its key
              transform{metal.includes('device const uint* probs') ? <>: it binds the probabilities as <code>device const uint*</code>, so no float is ever held</> : metal.includes('as_type') ? <>: it reads each float's bits with <code>as_type&lt;uint&gt;</code></> : null}.
              Try the <a href={href('ch14', 'labs')}>lab</a> before you read it.
            </>
          }
        />
      ) : null}
      <p>
        This transform is the first line of a GPU sampler's sort. llama.cpp's CUDA top-k computes it in a function
        called <code>top_k_float_to_ordered</code>, and its Metal top-k does the same with <code>as_type&lt;uint&gt;</code>;
        neither takes the −0.0 step.
      </p>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md, D2)">
        <p>
          Read on llama.cpp's master branch: <code>ggml/src/ggml-cuda/top-k.cu</code> and{' '}
          <code>ggml/src/ggml-metal/kernels/argsort.metal</code>. Our reading: neither merges −0.0 with +0.0, so +0.0
          ranks above −0.0 there, unlike CUB.
        </p>
      </Today>
    </>
  );
}
