import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile, type CodeTab } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { href } from '../../lib/router';
import {
  PRESETS,
  STOP_LABEL,
  overlapStep,
  parseValues,
  runToSorted,
  threadsFor,
  type Interleaving,
  type OverlapEvent,
  type StepResult,
  type StopRule,
} from '../../sim/oddeven';
import { functionSource } from '../../sim/toolchain';
import { OddEvenWalkScene } from './OddEvenWalkScene';
import { useSolutionStatus } from './solutionStatus';

const SNIPPET = 'ch14/snippets/s142_oddeven.cu';
const LAB = 'ch14/lab1_twosorts/solution/';
const STATUS = {
  snippet: 'front-end-checked; emulator-checked; unverified on CUDA and HIP hardware',
};
/** Where each value of the kernels lives. The lab tabs also hold a host loop: there only the device pointers are coloured. */
const TIERS: Record<string, Tier> = {
  data: 'global', hasChanged: 'global', flag: 'global',
  i: 'reg', tmp: 'reg', a: 'reg', b: 'reg',
};
/** Host code: its locals live in host memory, so only the device pointers get a tier. */
const HOST_TIERS: Record<string, Tier> = { data_d: 'global', changed_d: 'global' };
const missing = (s: string | null) => !s || s.startsWith('// could not load');
/** Long code in a table cell may wrap instead of widening the page. */
const WRAP = { whiteSpace: 'normal', overflowWrap: 'anywhere' } as const;
/** A function of a source file without the comment lines above it, so line 1 is the book's line 01. */
function bare(src: string | null, name: string): string | null {
  if (missing(src)) return null;
  const f = functionSource(src!, name);
  if (!f) return null;
  const lines = f.text.split('\n');
  let k = 0;
  while (k < lines.length && lines[k].trim().startsWith('//')) k++;
  return lines.slice(k).join('\n');
}

// ---------------------------------------------------------------- one row of values with pair brackets
function PairRow({ values, s, mark, predicted, onToggle, name }: {
  values: number[];
  /** The step whose pairs are bracketed. */
  s?: StepResult;
  /** Show which pairs swapped. */
  mark?: boolean;
  predicted?: Set<number>;
  onToggle?: (t: number) => void;
  name: string;
}) {
  const n = values.length;
  const swappedAt = new Set(mark && s ? s.threads.filter((t) => t.swapped).flatMap((t) => [t.i, t.i + 1]) : []);
  const pairAt = new Set(s ? s.threads.filter((t) => t.active).flatMap((t) => [t.i, t.i + 1]) : []);
  return (
    <div style={{ display: 'grid', gridTemplateColumns: `repeat(${n}, minmax(1.7rem, 2.6rem))`, columnGap: 4, rowGap: 4 }} data-row={name} data-values={values.join(' ')}>
      {values.map((v, k) => (
        <div key={'v' + k} className="mx global" style={{ display: 'block' }}>
          <span className={swappedAt.has(k) ? 'hot' : pairAt.has(k) ? 'on' : ''} style={{ minWidth: 0 }}>{v}</span>
        </div>
      ))}
      {s
        ? s.threads.map((t) =>
            t.active ? (
              <div key={'b' + t.t} style={{ gridColumn: `${t.i + 1} / span 2`, gridRow: 2, textAlign: 'center' }} data-pair={`${t.i}-${t.i + 1}`}
                data-swapped={mark ? (t.swapped ? 'yes' : 'no') : ''}>
                <div style={{ borderLeft: '1.5px solid var(--ink)', borderRight: '1.5px solid var(--ink)', borderBottom: `${mark && t.swapped ? 3 : 1.5}px solid var(--ink)`, height: 7, margin: '0 0.7rem' }} />
                {onToggle ? (
                  <button aria-pressed={predicted?.has(t.t)} onClick={() => onToggle(t.t)} style={{ fontSize: '0.72rem', padding: '0.1rem 0.3rem', marginTop: 2 }}>
                    t{t.t} swaps?
                  </button>
                ) : (
                  <div className="ui muted" style={{ fontSize: '0.72rem' }}>t{t.t}, i={t.i}</div>
                )}
              </div>
            ) : null,
          )
        : null}
    </div>
  );
}

// ---------------------------------------------------------------- the stepper
function Stepper() {
  const [preset, setPreset] = useState(0);
  const [text, setText] = useState(PRESETS[0].values.join(' '));
  const [values, setValues] = useState(PRESETS[0].values);
  const [rule, setRule] = useState<StopRule>('ours');
  const [pos, setPos] = useState(0);
  const [predict, setPredict] = useState(false);
  const [marked, setMarked] = useState<Set<number>>(new Set());
  const [verdict, setVerdict] = useState<{ right: number; total: number; step: number } | null>(null);
  const run = useMemo(() => runToSorted(values, rule), [values, rule]);
  const at = Math.min(pos, run.launched);
  const cur = at === 0 ? values : run.rows[at - 1].after;
  const last = at > 0 ? run.rows[at - 1] : undefined;
  const next = at < run.launched ? run.rows[at] : undefined;
  const done = at >= run.launched;
  const comparisons = run.rows.slice(0, at).reduce((s, r) => s + r.comparisons, 0);
  const swaps = run.rows.slice(0, at).reduce((s, r) => s + r.swaps, 0);
  const bad = parseValues(text) === null;

  useEffect(() => {
    setPos(0);
    setMarked(new Set());
    setVerdict(null);
  }, [values, rule]);

  const doStep = () => {
    if (!next) return;
    if (predict) {
      const actual = new Set(next.threads.filter((t) => t.swapped).map((t) => t.t));
      const active = next.threads.filter((t) => t.active);
      const right = active.filter((t) => actual.has(t.t) === marked.has(t.t)).length;
      setVerdict({ right, total: active.length, step: at + 1 });
    } else setVerdict(null);
    setMarked(new Set());
    setPos(at + 1);
  };
  const toggle = (t: number) => {
    const m = new Set(marked);
    if (m.has(t)) m.delete(t);
    else m.add(t);
    setMarked(m);
  };

  return (
    <Bench
      title="Step odd-even sort"
      task="Run Fig. 14.1, then the trap under each stopping rule. Then switch on prediction and call each step's swaps before you see them."
      foot={
        <>
          The data array lives in global memory. Bracketed cells are the pairs of the step shown; filled cells swapped.
          Each step is one launch of <code>sort_kernel</code> with {'⌊'}n/2{'⌋'} threads.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          Values
          <select value={preset} onChange={(e) => { const k = +e.target.value; setPreset(k); setValues(PRESETS[k].values); setText(PRESETS[k].values.join(' ')); }}>
            {PRESETS.map((p, k) => <option key={p.id} value={k}>{p.label}</option>)}
          </select>
        </label>
        <label className="field">
          or type 2 to 8 values
          <input type="text" value={text} aria-label="values" style={{ width: '11rem' }}
            onChange={(e) => { setText(e.target.value); const v = parseValues(e.target.value); if (v) setValues(v); }} />
        </label>
        {bad ? <span className="chip fail">2 to 8 whole numbers below 100</span> : null}
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          Host stops
          <select value={rule} onChange={(e) => setRule(e.target.value as StopRule)}>
            <option value="book">{STOP_LABEL.book}</option>
            <option value="ours">{STOP_LABEL.ours}</option>
          </select>
        </label>
        <label className="check">
          <input type="checkbox" checked={predict} onChange={(e) => { setPredict(e.target.checked); setMarked(new Set()); setVerdict(null); }} /> predict
          the next row
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPos(0); setVerdict(null); setMarked(new Set()); }} disabled={at === 0}>Restart</button>
        <button onClick={() => { setPos(at - 1); setVerdict(null); setMarked(new Set()); }} disabled={at === 0}>Back</button>
        <button className="primary" onClick={doStep} disabled={done}>{predict ? 'Check and step' : 'Step'}</button>
        <button onClick={() => { setPos(run.launched); setVerdict(null); setMarked(new Set()); }} disabled={done}>Run</button>
        <span className="muted" aria-live="polite" data-pos={at}>step {at} of {run.launched}</span>
      </div>

      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }}>
        {verdict ? (
          <>
            <span className={'chip ' + (verdict.right === verdict.total ? 'ok' : 'fail')} data-verdict={`${verdict.right}/${verdict.total}`}>
              you called {verdict.right} of {verdict.total} pairs right
            </span>{' '}
          </>
        ) : null}
        {last ? (
          <>
            Step {last.n} ordered the {last.parity ? 'odd' : 'even'} pairs: {last.comparisons} threads compared, {last.swaps} swapped, so the
            flag is {last.flag ? 1 : 0}.{' '}
          </>
        ) : (
          <>Nothing has run yet. Step 1 orders the even pairs. </>
        )}
        {done ? (
          <span data-stop={run.sorted ? 'sorted' : 'unsorted'}>
            The host stops after step {run.launched}{run.launched ? '' : ' (fewer than two values: nothing to launch)'}.{' '}
            {run.sorted ? <span className="chip ok">sorted</span> : <span className="chip fail">not sorted</span>}
          </span>
        ) : next ? (
          <>Next: step {next.n}, {next.parity ? 'odd' : 'even'} pairs.</>
        ) : null}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 5fr)' }}>
        <div>
          {last ? (
            <>
              <div className="mx-label">Step {last.n} ({last.parity ? 'odd' : 'even'} pairs), before and after</div>
              <PairRow values={last.before} s={last} mark name="before" />
              <div style={{ height: 6 }} />
              <PairRow values={last.after} name="after" />
            </>
          ) : (
            <>
              <div className="mx-label">Input</div>
              <PairRow values={cur} name="after" />
            </>
          )}
          {predict && next ? (
            <div style={{ marginTop: '1rem' }}>
              <div className="mx-label">Step {next.n} ({next.parity ? 'odd' : 'even'} pairs): mark the pairs you expect to swap</div>
              <PairRow values={cur} s={next} predicted={marked} onToggle={toggle} name="predict" />
            </div>
          ) : null}
          <div className="mx-label" style={{ marginTop: '1rem' }}>Every row so far</div>
          <table className="threads" aria-label="Rows so far" data-history={at}>
            <tbody>
              <tr><td>input</td><td className="mono">{values.join(' ')}</td><td /></tr>
              {run.rows.slice(0, at).map((r) => (
                <tr key={r.n} className={r.n === at ? 'cur' : ''}>
                  <td>step {r.n}, {r.parity ? 'odd' : 'even'}</td>
                  <td className="mono">{r.after.join(' ')}</td>
                  <td>{r.flag ? `flag 1` : 'quiet'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <div className="mx-label">Threads of {last ? `step ${last.n}` : 'the next step'} (i = 2t + {(last ?? next)?.parity ?? 0})</div>
          <table className="threads" aria-label="Threads">
            <thead>
              <tr><th>t</th><th>i</th><th>pair</th><th>does</th></tr>
            </thead>
            <tbody>
              {Array.from({ length: threadsFor(values.length) }, (_, t) => {
                const s = last ?? next;
                const th = s?.threads[t];
                if (!th) return null;
                return (
                  <tr key={t} data-thread={t}>
                    <td>{t}</td>
                    <td>{th.i}</td>
                    <td>{th.active ? `(${th.i}, ${th.i + 1})` : 'out of range'}</td>
                    <td>{!th.active ? 'nothing: i + 1 < n fails' : !last ? 'waits' : th.swapped ? `${th.a} > ${th.b}: swaps, sets flag` : `${th.a} ≤ ${th.b}: no swap`}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <table className="threads" aria-label="Counters" style={{ marginTop: '0.8rem' }}>
            <tbody>
              <tr><td>steps launched</td><td data-counter="steps">{at}</td></tr>
              <tr><td>comparisons</td><td data-counter="comparisons">{comparisons}</td></tr>
              <tr><td>swaps</td><td data-counter="swaps">{swaps}</td></tr>
              <tr><td>sorted after step</td><td data-counter="sortedAfter">{run.sortedAfter === null ? 'not yet' : run.sortedAfter <= at ? run.sortedAfter : 'not yet'}</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the bug hunt
const OP_TEXT: Record<OverlapEvent['op'], string> = { loadA: 'load', loadB: 'load', storeA: 'store', storeB: 'store' };

function describeEvent(e: OverlapEvent): string {
  const r = e.regs[e.t];
  const who = `Thread ${e.t} (owns ${e.t} and ${e.t + 1})`;
  if (e.op === 'loadA') return `${who} loads data[${e.index}] = ${e.value} into a.`;
  if (e.op === 'loadB')
    return `${who} loads data[${e.index}] = ${e.value} into b. ${r.swap ? `${r.a} > ${r.b}, so it will swap.` : `${r.a} ≤ ${r.b}: nothing to do.`}`;
  if (e.skipped) return `${who} has nothing to store.`;
  return `${who} stores ${e.op === 'storeA' ? 'b' : 'a'} = ${e.value} into data[${e.index}].`;
}

function NeighbourHunt() {
  const [preset, setPreset] = useState(1);
  const [kind, setKind] = useState<Interleaving['kind']>('race');
  const [atEl, setAtEl] = useState(4);
  const [pos, setPos] = useState(0);
  const values = PRESETS[preset].values;
  const il: Interleaving = kind === 'race' ? { kind, at: Math.min(atEl, values.length - 2) } : { kind };
  const run = useMemo(() => overlapStep(values, il), [values, kind, atEl]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => setPos(0), [preset, kind, atEl]);
  const total = run.events.length;
  const at = Math.min(pos, total);
  const ev = at > 0 ? run.events[at - 1] : undefined;
  const data = ev ? ev.data : values;
  const regs = ev ? ev.regs : values.slice(1).map(() => ({ done: false }) as OverlapEvent['regs'][number]);
  const finished = at >= total;
  const dup = new Set(finished ? run.duplicated : []);
  return (
    <Bench
      title="Neighbouring pairs at once"
      task="Pick the race, step the loads and stores, and find the value that disappears."
      foot={
        <>
          Every thread t owns the pair (t, t + 1) in the same step, as a naive parallel bubble sort would: neighbouring
          threads share an element. A thread loads both values, compares, and if needed stores both back.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          Values
          <select value={preset} onChange={(e) => setPreset(+e.target.value)}>
            {PRESETS.map((p, k) => <option key={p.id} value={k}>{p.label}</option>)}
          </select>
        </label>
        <label className="field">
          Interleaving
          <select value={kind} onChange={(e) => setKind(e.target.value as Interleaving['kind'])}>
            <option value="in-order">threads one after another, 0 first</option>
            <option value="reversed">threads one after another, last first</option>
            <option value="race">two threads race on one element</option>
          </select>
        </label>
        {kind === 'race' ? (
          <label className="field">
            Shared element
            <select value={il.kind === 'race' ? il.at : 1} onChange={(e) => setAtEl(+e.target.value)}>
              {Array.from({ length: values.length - 2 }, (_, k) => k + 1).map((e) => (
                <option key={e} value={e}>data[{e}]: threads {e - 1} and {e}</option>
              ))}
            </select>
          </label>
        ) : null}
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setPos(0)} disabled={at === 0}>Restart</button>
        <button onClick={() => setPos(at - 1)} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => setPos(at + 1)} disabled={finished}>Step</button>
        <button onClick={() => setPos(total)} disabled={finished}>Run to the end</button>
        <span className="muted" aria-live="polite">operation {at} of {total}</span>
      </div>
      <div className={'narrate' + (finished && !run.permutation ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }}>
        {ev ? describeEvent(ev) : 'Nothing has run yet.'}
        {finished ? (
          <span data-result={run.permutation ? 'permutation' : 'broken'}>
            {' '}
            {run.permutation ? (
              <span className="chip ok">still a permutation of the input</span>
            ) : (
              <>
                <span className="chip fail">not a permutation</span> lost: {run.lost.join(', ')}; duplicated: {run.duplicated.join(', ')}.
              </>
            )}
          </span>
        ) : null}
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 5fr) minmax(0, 6fr)' }}>
        <div>
          <div className="mx-label t-global">data (global memory)</div>
          <div className="mx global" data-row="data" data-values={data.join(' ')} style={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
            {data.map((v, k) => (
              <span key={k} className={dup.has(v) ? 'bad' : ev && ev.index === k && !ev.skipped ? 'hot' : ''}>
                {v}
                <sub>{k}</sub>
              </span>
            ))}
          </div>
          <div className="mx-label" style={{ marginTop: '0.8rem' }}>Input</div>
          <div className="mono muted">{values.join(' ')}</div>
        </div>
        <div>
          <table className="threads" aria-label="Naive threads">
            <thead>
              <tr><th>t</th><th>owns</th><th>a</th><th>b</th><th>swap?</th><th>last did</th></tr>
            </thead>
            <tbody>
              {regs.map((r, t) => {
                const lastOp = run.events.slice(0, at).filter((e) => e.t === t).pop();
                return (
                  <tr key={t} className={ev?.t === t ? 'cur' : ''}>
                    <td>{t}</td>
                    <td>({t}, {t + 1})</td>
                    <td className="t-reg">{r.a ?? '·'}</td>
                    <td className="t-reg">{r.b ?? '·'}</td>
                    <td>{r.swap === undefined ? '' : r.swap ? 'yes' : 'no'}</td>
                    <td>{lastOp ? (lastOp.skipped ? 'skipped a store' : `${OP_TEXT[lastOp.op]} data[${lastOp.index}]`) : ''}</td>
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

// ---------------------------------------------------------------- questions
export const questionsS142: Record<string, Question> = {
  'ch14/14.2/q-threads': {
    kind: 'numeric',
    id: 'ch14/14.2/q-threads',
    prompt: (
      <>
        Fig. 14.2 is launched with n/2 = 5 threads for n = 10. On an odd step, how many of those threads compare a pair?
      </>
    ),
    answer: 4,
    explain: (
      <p>
        On an odd step thread t takes i = 2t + 1: 1, 3, 5, 7, 9. Thread 4's pair would be (9, 10), and element 10 does not
        exist, so the bounds test fails and it does nothing. Fig. 14.1 shows the same drop for n = 8: four threads on even
        steps, three on odd ones.
      </p>
    ),
    source: 'ch. 14 §14.2, Fig. 14.2, p. 332',
  },
  'ch14/14.2/q-worst': {
    kind: 'numeric',
    id: 'ch14/14.2/q-worst',
    prompt: (
      <>
        The input is [8, 1, 2, 3, 4, 5, 6, 7], and the first step is even, as in Fig. 14.1. After how many steps is the list
        sorted?
      </>
    ),
    answer: 7,
    explain: (
      <p>
        Each step compares the 8 with exactly one neighbour, so it moves one place per step: positions 1, 2, ..., 7 after
        steps 1 to 7, and the values it passes shift down by one behind it. Sorted after 7 steps. The host does not know
        that yet: it stops after two steps in a row, both quiet, and those are steps 8 (odd) and 9 (even), so it launches 9
        in all. Lab 14.1's loop reads the flag only after each even-odd pair, so it runs the pair of steps 9 and 10
        before it stops: 10 launches.
      </p>
    ),
    source: 'ch. 14 §14.2, p. 333; src/sim/oddeven.ts',
  },
  'ch14/14.2/q-flag': {
    kind: 'choice',
    id: 'ch14/14.2/q-flag',
    prompt: <>Every thread that swaps writes 1 to the same flag, in the same step. Which statement is right?</>,
    options: [
      { label: 'Writes of the same value cannot race.', why: 'A race is about unordered accesses to one location, at least one a write; the values do not enter into it.' },
      { label: 'Add __syncthreads() before the write.', why: 'A barrier orders threads of one block only, and the writers here come from every block of the grid.' },
      {
        label: 'It is a data race, undefined behaviour in the CUDA memory model even though every write stores the same value. Make the store atomic; relaxed order is enough, because the host reads the flag only after the launch has finished.',
        correct: true,
      },
      { label: 'It only matters on AMD hardware.', why: 'The rule is in NVIDIA\'s own memory model, and it is about the language, not one vendor\'s hardware.' },
    ],
    explain: (
      <p>
        The book calls this race benign in practice and suggests an atomic for conservative programmers. NVIDIA's current
        guide is blunter: any data race is undefined behaviour. <code>atomicExch(hasChanged, 1u)</code> costs little here,
        since it runs only in threads that swapped, and it makes the program defined. No ordering beyond atomicity is
        needed: the host reads the flag after the launch has completed.
      </p>
    ),
    source: 'ch. 14 §14.2, p. 333; notes/today/ch10-facts.md A1; notes/today/ch14-facts.md A9',
  },
  'ch14/14.2/q-stop': {
    kind: 'choice',
    id: 'ch14/14.2/q-stop',
    prompt: <>Which four-element input makes a host loop that stops at the first quiet step return an unsorted list?</>,
    options: [
      { label: '[4, 3, 2, 1]', why: 'The first even step swaps (4, 3) and (2, 1).' },
      { label: '[2, 1, 3, 4]', why: 'The first even step swaps (2, 1), and then the list is sorted.' },
      { label: '[1, 2, 4, 3]', why: 'The first even step swaps (4, 3).' },
      { label: '[1, 3, 2, 4]', correct: true },
    ],
    explain: (
      <p>
        The first even step compares (1, 3) and (2, 4): both in order, no swap, flag 0. The out-of-order pair (3, 2) is an
        odd pair, which that step never looks at. A quiet even step says only that the even pairs are in order. Stop after
        two steps in a row, one even and one odd in either order, are both quiet: then every adjacent pair has been
        checked without anything moving in between.
      </p>
    ),
    source: 'our reading of p. 331 (notes/ch14.md, "Unclear in the PDF" item 7)',
  },
  'ch14/14.2/q-work': {
    kind: 'numeric',
    id: 'ch14/14.2/q-work',
    prompt: <>With the book's bound of N steps of N/2 comparisons each, how many comparisons does odd-even sort make for N = 1,024?</>,
    answer: 524288,
    explain: (
      <p>
        1,024 × 512 = 524,288. A sort that makes on the order of N log2 N comparisons needs about 1,024 × 10 = 10,240: about
        fifty times fewer. Running each step fully in parallel makes the time O(N); it does not make the work any smaller.
      </p>
    ),
    source: 'ch. 14 §14.2, p. 333',
  },
};

// ---------------------------------------------------------------- the page
export function S142() {
  const snip = useLabFile(SNIPPET);
  const kernel = bare(snip, 'sort_kernel');
  const host = missing(snip) ? null : functionSource(snip!, 'oddEvenSort')?.text ?? null;
  const labCu = useLabFile(LAB + 'kernel.cu');
  const labMetal = useLabFile(LAB + 'kernel.metal');
  const labMm = useLabFile(LAB + 'launch.mm');
  const labStatus = useSolutionStatus('ch14/lab1_twosorts');
  // Only the odd-even part of each lab file: the kernel and its host loop.
  const part = (src: string | null, names: string[]) => {
    const fs = missing(src) ? [] : names.map((nm) => functionSource(src!, nm)?.text);
    return fs.length && fs.every(Boolean) ? fs.join('\n\n') : null;
  };
  const labCuPart = part(labCu, ['oddEvenStep', 'launchOddEvenSort']);
  const labMetalPart = part(labMetal, ['odd_even_step']);
  const labMmPart = part(labMm, ['runOddEvenSort']);
  const tabs: CodeTab[] = kernel ? [{ label: 'Fig. 14.2, corrected', code: kernel, tiers: TIERS, highlight: [5, 10], status: STATUS.snippet }] : [];
  if (kernel && labCuPart) tabs.push({ label: 'Lab 14.1: CUDA and HIP', code: labCuPart, tiers: TIERS, status: labStatus.cu });
  if (kernel && labMetalPart) tabs.push({ label: 'Lab 14.1: Metal kernel', code: labMetalPart, tiers: TIERS, status: labStatus.metal });
  if (kernel && labMetalPart && labMmPart) tabs.push({ label: 'Lab 14.1: Metal host', code: labMmPart, tiers: {}, status: labStatus.metal });
  return (
    <>
      <p>
        Bubble sort compares neighbours and swaps those out of order, pass after pass, until nothing changes. Within a pass
        the order in which the pairs are visited does not matter, which looks like an invitation to give every pair to its
        own thread.
      </p>
      <Source section="14.2" pages="331" />
      <p>
        It is not. If one thread owns (0, 1) and another owns (1, 2), and both decide to swap, both read and write element 1.
        Whichever writes last wins, and the result need not even contain the same values as the input. Try it.
      </p>
      <NeighbourHunt />
      <p>
        With the threads one after another, the step is an ordinary bubble-sort pass and the values survive. The race needs
        both threads to load before either stores, and both to swap: then one value is written twice and another is lost.
        The model runs each load and store as one indivisible operation; real threads interleave more finely, which only
        adds ways to go wrong.
      </p>

      <h2>Disjoint pairs, alternating</h2>
      <p>
        The fix is to work only on pairs that share nothing. On an <b>even step</b> every pair whose first index is even is
        ordered: (0, 1), (2, 3), and so on. On an <b>odd step</b>, every pair whose first index is odd: (1, 2), (3, 4), ...
        Alternate the two until nothing is swapped. Within a step no two threads touch the same element, so the order of the
        threads cannot matter. This is odd-even transposition sort; Fig. 14.1 runs it on eight values, the last of six steps
        changing nothing.
      </p>
      <Source section="14.2" pages="331–332" figs="Fig. 14.1" />
      <p>
        The kernel performs one step. It is launched once per step with half as many threads as elements. Thread t takes
        the pair starting at i = 2t on an even step and i = 2t + 1 on an odd one, checks that the pair is inside the list,
        swaps if needed, and sets a flag so the host knows something moved.
      </p>
      <Source section="14.2" pages="331–332" figs="Fig. 14.2" />
      {kernel ? (
        <Code
          tabs={tabs}
          caption={
            <>
              Fig. 14.2 with the page's line numbers, from <code>labs/{SNIPPET}</code>. Two lines differ from the page:
              line 05 reads <code>if(i + 1 &lt; N)</code> where the page has <code>if(i &lt; N - 1)</code>, and line 10 uses{' '}
              <code>atomicExch</code> where the page has <code>*hasChanged = 1;</code>. HIP has <code>atomicExch</code> under
              the same name, so the file is written to serve HIP through the course's header; it has not been built for
              HIP here. The other tabs are the odd-even part of lab 14.1's reference solution: try the lab before
              reading them.
            </>
          }
        />
      ) : null}
      <p>
        Why the two changes. <code>N</code> is unsigned, so for an empty list <code>N - 1</code> wraps to the largest
        unsigned value, every thread passes the test, and every thread reads past the end; <code>i + 1 &lt; N</code> asks
        the same question without the subtraction. The flag gets its own part below, after you have watched the sort run.
      </p>
      <Beyond>
        <p>
          Both changes are ours. The wrap is noted in notes/ch14.md (code listings, Fig. 14.2, "Care"); the course
          emulator runs the corrected kernel on an empty list and checks that nothing is touched.
        </p>
      </Beyond>
      <Ask q={questionsS142['ch14/14.2/q-threads']} label="Work it out" />

      <h2>Watch it, then drive it</h2>
      <p>
        The animation shows what the stepper below lets you test: why the pairs never collide, how far a value can travel
        in one step, what each step costs, and when it is safe to stop.
      </p>
      <OddEvenWalkScene />
      <p>
        Now drive it yourself. Run Fig. 14.1 first and compare each row with the book's figure; then load the trap and run
        it under each stopping rule.
      </p>
      <Simplification>
        <p>
          One thread's compare and swap is a single step of the model; real threads interleave at the instruction level and
          run in warps. Divergence is not modelled: when some threads of a warp swap and others do not, the swap branch
          has a cost the model ignores (on Metal that cost is not documented: notes/today/ch04-facts.md C4). No timing.
        </p>
      </Simplification>
      <Stepper />

      <h2>A race that writes the same value</h2>
      <p>
        Look again at line 10 as printed: every thread that swaps writes 1 to <code>hasChanged</code>, many of them in the
        same step, with nothing ordering the writes. That is a data race: one location, unordered accesses, at least one a
        write. The book calls it benign, since every write stores the same value and storing 1 twice is the same as storing
        it once (the operation is idempotent), and then adds the caveat that matters: in the C++ memory model a data race is
        undefined behaviour even when it is idempotent. The code may work; nothing guarantees it. Conservative programmers,
        it says, use an atomic.
      </p>
      <Source section="14.2" pages="333" />
      <p>
        NVIDIA's current guide states the rule for CUDA directly: any data race results in undefined behaviour. So the
        snippet makes the store atomic. Relaxed order is enough: the host reads the flag only after the launch has finished,
        and nothing else depends on when the write becomes visible.
      </p>
      <Today checked="2026-11-20 (notes/today/ch10-facts.md A1; ch14-facts.md A9, B7; infra-facts.md A1, A2, B1, B2, C1)">
        <p>
          CUDA: <code>atomicExch</code> is relaxed and device-scoped; <code>cuda::atomic_ref</code> with{' '}
          <code>store(1u, cuda::memory_order_relaxed)</code> is the form NVIDIA now recommends. HIP has{' '}
          <code>atomicExch</code> under the same name; its documentation states no rule for plain racy writes (we could not
          verify one). Metal needs the flag declared as an atomic type.
        </p>
      </Today>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th>Setting the flag</th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>The atomic store</td>
            <td><code>atomicExch(flag, 1u)</code>, or <code style={WRAP}>cuda::atomic_ref&lt;unsigned int, cuda::thread_scope_device&gt;(*flag).store(1u, cuda::memory_order_relaxed)</code></td>
            <td><code>atomicExch(flag, 1u)</code>; <code>cuda::atomic_ref</code> through libhipcxx, part of ROCm from 10.1</td>
            <td><code style={WRAP}>atomic_store_explicit(flag, 1u, memory_order_relaxed)</code>, with <code>flag</code> a <code>device atomic_uint*</code></td>
          </tr>
          <tr>
            <td>A plain racy write</td>
            <td>undefined behaviour</td>
            <td>no rule stated in the HIP documentation</td>
            <td>not defined for plain accesses</td>
          </tr>
          <tr>
            <td>Reading it on the host</td>
            <td><code>cudaMemcpy</code> back: it returns only when the copy is done, and the copy waits for the launch before it</td>
            <td>
              <code>hipMemcpy</code> back, through the same source. HIP documents that it runs after the launch in the same
              stream; that it returns only when the copy is done is implied, not stated (ch02-facts B3). A{' '}
              <code>hipDeviceSynchronize()</code> after the copy, before reading the flag, makes the wait explicit.
            </td>
            <td>after the command buffer has completed (for example <code>waitUntilCompleted</code>), read the shared buffer</td>
          </tr>
        </tbody>
      </table>
      <p>
        You can watch the rule being enforced. The course's race-detection emulator runs the threads of a launch truly
        concurrently under ThreadSanitizer. Built with Fig. 14.2 as printed, one step on reversed input reports a data race
        on the line <code>*hasChanged = 1;</code>, although the sort comes out right; built with the snippet's{' '}
        <code>atomicExch</code>, the same step reports nothing.
      </p>
      <Aside>
        <p>
          That check is in <code>app/src/chapters/ch14/S142.snippets.test.ts</code>, with its driver in{' '}
          <code>labs/ch14/snippets/check/</code>. It is emulator-checked: no GPU was involved.
        </p>
      </Aside>
      <Ask q={questionsS142['ch14/14.2/q-flag']} />

      <h2>When to stop</h2>
      <p>
        The book says to keep alternating until no elements are swapped, and does not show the host loop. Read literally,
        "stop after a step with no swap" is wrong: the trap in the stepper, or simply [1, 3, 2, 4], makes no swap on its
        first even step and is not sorted. A quiet even step proves only that the even pairs are in order. What does prove
        it: two steps in a row, one even and one odd in either order, both quiet. Then every adjacent pair has been
        checked with nothing moving in between.
      </p>
      <Source section="14.2" pages="331" />
      <Ask q={questionsS142['ch14/14.2/q-stop']} />
      <p>
        Here is the loop in the snippet. Reading the flag has a precondition the book leaves implicit: the step must have
        finished. On CUDA the copy back provides that, because <code>cudaMemcpy</code> does not return until it is done and
        it runs after the launch in the same stream. On Metal the flag lives in a shared buffer and may be read once the
        command buffer that ran the step has completed.
      </p>
      <Beyond>
        <p>
          The host loop is ours, as is the stopping rule. CUDA's copy behaviour: notes/today/ch02-facts.md A4 and A5.
          Metal: ch05-facts.md C6, infra-facts.md C12.
        </p>
      </Beyond>
      {host ? (
        <Code
          tabs={[{ label: 'host loop', code: host, tiers: HOST_TIERS, status: STATUS.snippet }]}
          caption={<>From the same file. The quiet count resets whenever a step swaps.</>}
        />
      ) : null}

      <h2>What it costs</h2>
      <p>
        In the worst case the largest value starts at the front. Each step compares it with one neighbour, so it moves one
        place per step and needs about N steps to reach the end. With enough threads to run a step entirely in parallel, the
        time is O(N). Each step makes about N/2 comparisons, so the work is O(N²): far more than the order of N log N a
        good sequential sort needs.
      </p>
      <Source section="14.2" pages="333" />
      <Ask q={questionsS142['ch14/14.2/q-worst']} label="Work it out" />
      <Ask q={questionsS142['ch14/14.2/q-work']} label="Work it out" />
      <p>
        On a GPU there is a cost the O(N) hides: each step is a launch, and the snippet follows each launch with a copy
        back to read the flag. For a long list that is thousands of round trips. Lab 14.1's reference halves the reads: it
        launches an even and an odd step back to back and reads the flag once for the pair. The price is up to one more
        step, because two quiet steps that straddle two pairs, an odd then an even, do not stop it: for [8, 1, ..., 7] it
        launches 10 steps where the snippet's loop launches 9. The lab runs it on your GPU (see the{' '}
        <a href={href('ch14', 'labs')}>labs page</a>); before you run it, decide whether a list of a few thousand values
        would be sorted faster by this kernel or by one thread on the CPU.
      </p>
      <p>
        Where odd-even sort does earn its place is inside a single thread. MLX's GPU sort begins with each thread sorting
        the few values in its own registers, and the loop it uses is exactly this one: alternate even and odd pairs, a fixed
        number of times, with no flag and no launches. Inside one thread there is no race to avoid and no host to ask,
        and for a handful of values N² stays small; section 14.9 calls such fixed patterns sorting networks.
      </p>
      <Today checked="2026-11-20 (notes/today/ch14-facts.md, D4)">
        <p>
          <code>mlx/backend/metal/kernels/sort.h</code>, <code>ThreadSort</code>: <code>for (i = 0; i &lt; N_PER_THREAD; ++i)</code>{' '}
          over <code>j = i &amp; 1, j + 2, ...</code>, swapping <code>vals[j]</code> and <code>vals[j + 1]</code> when out of
          order.
        </p>
      </Today>
    </>
  );
}
