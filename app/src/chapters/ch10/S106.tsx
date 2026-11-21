import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { href } from '../../lib/router';
import {
  FIG_VALUES,
  HUNT_FILE,
  OUTCOME_LABEL,
  VARIANT,
  VARIANTS,
  fmtValue,
  requests,
  runShared,
  stateAt,
  strides,
  type Outcome,
  type Schedule,
  type ThreadRun,
  type TStep,
  type Variant,
} from '../../sim/reduction';
import { SCHEDULE_LABEL } from '../../sim/tiled';
import { functionSource } from '../../sim/toolchain';

import { NoteGroup } from './NoteGroup';
import { STATUS, useLabStatus } from './status';
const LAB = 'ch10/lab1_sum';
const SCHEDULES: Schedule[] = ['lockstep', 'in-order', 'reversed', 'shuffled'];
const TIERS: Record<string, Tier> = {
  input: 'global', output: 'global', row: 'global', input_s: 'shared', buf_s: 'shared',
  t: 'reg', stride: 'reg', a: 'reg', b: 'reg', m: 'reg', bdim: 'reg',
};
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const missing = (s: string | null) => !s || s.startsWith('// could not load');
const ONES = new Array<number>(16).fill(1);
const SHORT: Record<Variant, string> = {
  book: 'as in the book',
  bottom: 'barrier at the bottom',
  none: 'no barrier',
  'inside-if': 'barrier inside the if',
  le: '<= instead of <',
  two: 'max then sum, with the barrier',
  'two-race': 'max then sum, without it',
};
const ORDER_SHORT: Record<Schedule, string> = { lockstep: 'lockstep', 'in-order': 'thread 0 first', reversed: 'last thread first', shuffled: 'random' };
type Guess = '' | 'correct' | 'some' | 'deadlock' | 'oob';
const GUESS_LABEL: Record<Exclude<Guess, ''>, string> = {
  correct: 'correct in every order',
  some: 'wrong in some orders',
  deadlock: 'stuck at a barrier',
  oob: 'out of bounds',
};
/** The class of a whole row of outcomes, to compare with a prediction. */
function rowClass(outs: Outcome[]): Exclude<Guess, ''> {
  if (outs.every((o) => o === 'correct')) return 'correct';
  if (outs.some((o) => o === 'deadlock')) return 'deadlock';
  if (outs.some((o) => o === 'oob')) return 'oob';
  return 'some';
}

/** The two kernels of the hunt that are right; the other five are wrong on purpose. */
const CORRECT = new Set<Variant>(['book', 'two']);

// ---------------------------------------------------------------- the bug hunt
const arr = (v: Variant) => (VARIANT[v].two ? 'buf_s' : 'input_s');

function describe(v: Variant, run: ThreadRun, s: TStep): string {
  const A = arr(v);
  if (s.thread === -1) return `The barrier on line ${s.site} opens: all ${run.threads} threads have arrived.`;
  if (s.thread === -2) return 'Nothing can move.';
  const op = s.op!;
  const t = s.thread;
  const r = s.reads;
  const w = s.write ? fmtValue(s.write.value) : '';
  const test = VARIANT[v].le ? `${t} <= ${op.stride}` : `${t} < ${op.stride}`;
  switch (op.kind) {
    case 'load':
      return `Thread ${t} loads input[${r[0].index}] (${fmtValue(r[0].value)}) and input[${r[1].index}] (${fmtValue(r[1].value)}) and stores their sum, ${w}, in ${A}[${t}].`;
    case 'load-max':
      return `Thread ${t} loads input[${r[0].index}] (${fmtValue(r[0].value)}) and input[${r[1].index}] (${fmtValue(r[1].value)}) into a and b, and stores the larger, ${w}, in ${A}[${t}].`;
    case 'skip':
      return `Thread ${t}, ${op.phase === 0 && VARIANT[v].two ? 'max' : 'sum'} iteration ${op.iter} (stride ${op.stride}): ${test} is false, so it skips.`;
    case 'add':
      return `Thread ${t}, iteration ${op.iter} (stride ${op.stride}): ${test}, so ${A}[${t}] (${fmtValue(r[0].value)}) += ${A}[${r[1].index}] (${fmtValue(r[1].value)}): ${w}.`;
    case 'max':
      return `Thread ${t}, max iteration ${op.iter} (stride ${op.stride}): ${A}[${t}] = the larger of ${fmtValue(r[0].value)} and ${A}[${r[1].index}] (${fmtValue(r[1].value)}): ${w}.`;
    case 'read-max':
      return `Thread ${t} reads the max from ${A}[0]: m = ${fmtValue(r[0].value)}.`;
    case 'load-diff':
      return `Thread ${t} starts the sum: ${A}[${t}] = (a - m) + (b - m) = ${w}.`;
    case 'barrier':
      return `Thread ${t} reaches the barrier on line ${op.lines[0]} and waits.`;
    case 'store':
      return `Thread 0 copies ${A}[0] = ${fmtValue(r[0].value)} to *output.`;
    default:
      return '';
  }
}

function OutcomeChip({ o, detail }: { o: Outcome; detail?: string }) {
  return (
    <>
      <span className={'chip ' + (o === 'correct' ? 'ok' : 'fail')}>{OUTCOME_LABEL[o]}</span>
      {detail ? <div className="muted" style={{ fontSize: '0.75rem' }}>{detail}</div> : null}
    </>
  );
}

function BarrierHunt() {
  const src = useLabFile(HUNT_FILE);
  const [data, setData] = useState<'ones' | 'mixed'>('ones');
  const values = data === 'ones' ? ONES : [...FIG_VALUES];
  const [guess, setGuess] = useState<Record<Variant, Guess>>(() => Object.fromEntries(VARIANTS.map((v) => [v, ''])) as Record<Variant, Guess>);
  const [shown, setShown] = useState(false);
  const [variant, setVariant] = useState<Variant>('book');
  const [schedule, setSchedule] = useState<Schedule>('in-order');
  const [n, setN] = useState(0);

  const table = useMemo(
    () => Object.fromEntries(VARIANTS.map((v) => [v, SCHEDULES.map((s) => runShared(v, values, s))])) as Record<Variant, ThreadRun[]>,
    [data], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const run = table[variant][SCHEDULES.indexOf(schedule)];
  const total = run.steps.length;
  const at = Math.min(n, total);
  const cur = at > 0 ? run.steps[at - 1] : undefined;
  const st = stateAt(run, at);
  const firstBad = run.issues.length ? run.issues[0].step + 1 : null;
  const fn = !missing(src) ? functionSource(src!, VARIANT[variant].kernel) : null;
  const B = run.threads;
  const L = strides('shared', B).length;
  useEffect(() => setN(0), [variant, schedule, data]);

  const nextBarrier = () => {
    for (let i = at; i < total; i++) if (run.steps[i].thread < 0) return setN(i + 1);
    setN(total);
  };
  const highlight = cur?.op ? cur.op.lines : cur?.site ? [cur.site] : [];
  const allGuessed = VARIANTS.every((v) => guess[v] !== '');
  const regs = (t: number) => {
    // m as the thread last read it, from the steps so far.
    let m: number | null = null;
    for (let i = 0; i < at; i++) {
      const s = run.steps[i];
      if (s.thread === t && s.op?.kind === 'read-max') m = s.reads[0].value;
    }
    return m;
  };

  return (
    <Bench
      title="Where does the barrier go?"
      task="Predict each variant's outcome over all four thread orders, then run them and replay any cell."
      foot={
        <>
          The orders: all threads in lockstep; each thread runs as far as it can, thread 0 first, or the last thread first;
          a random interleaving. Every variant is a kernel in <code>labs/{HUNT_FILE}</code>, run here by a model of one block of {B} threads. A
          small label in a shared-memory slot says which value it holds: the reduction (max or sum) and the iteration that
          wrote it, 0 for the load. "Stuck at a barrier" stands for undefined behaviour: a GPU may hang, or carry on and
          compute something wrong.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span style={FIELD}>
          Input (16 values)
          <span className="seg" role="group" aria-label="Input">
            <button aria-pressed={data === 'ones'} onClick={() => setData('ones')}>all ones</button>
            <button aria-pressed={data === 'mixed'} onClick={() => setData('mixed')}>3, 1, 7, 0, ...</button>
          </span>
        </span>
        <span className="muted">
          The sum must be {fmtValue(runShared('book', values, 'lockstep').expected)}; for the two reductions, the sum of x − max must be{' '}
          {fmtValue(runShared('two', values, 'lockstep').expected)}.
        </span>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table className="data" style={{ background: 'var(--page)', margin: 0 }} data-part="hunt-table">
          <thead>
            <tr>
              <th style={{ paddingLeft: '0.6rem' }}>Variant</th>
              <th>Your prediction</th>
              {SCHEDULES.map((s) => <th key={s} title={SCHEDULE_LABEL[s]}>{ORDER_SHORT[s]}</th>)}
            </tr>
          </thead>
          <tbody>
            {VARIANTS.map((v) => {
              const outs = table[v].map((r) => r.outcome);
              const cls = rowClass(outs);
              return (
                <tr key={v} data-variant={v}>
                  <td style={{ paddingLeft: '0.6rem' }}>
                    <button className="quiet" aria-pressed={variant === v} onClick={() => setVariant(v)} title={VARIANT[v].label} style={{ whiteSpace: 'nowrap' }}>{SHORT[v]}</button>
                  </td>
                  <td>
                    <select value={guess[v]} onChange={(e) => setGuess({ ...guess, [v]: e.target.value as Guess })} aria-label={`prediction for ${SHORT[v]}`} style={{ maxWidth: '9rem' }}>
                      <option value="">choose</option>
                      {(Object.keys(GUESS_LABEL) as Exclude<Guess, ''>[]).map((g) => <option key={g} value={g}>{GUESS_LABEL[g]}</option>)}
                    </select>
                    {shown && guess[v] ? (
                      <div className={guess[v] === cls ? 't-ok' : 't-fail'} style={{ fontSize: '0.75rem' }} data-verdict={guess[v] === cls ? 'right' : 'wrong'}>
                        {guess[v] === cls ? 'as you predicted' : `actually: ${GUESS_LABEL[cls]}`}
                      </div>
                    ) : null}
                  </td>
                  {table[v].map((r, i) => (
                    <td key={i} data-outcome={shown ? r.outcome : 'hidden'}>
                      {shown ? (
                        <button className="quiet" style={{ padding: '0.15rem 0.3rem', textAlign: 'left' }}
                          onClick={() => { setVariant(v); setSchedule(SCHEDULES[i]); }} aria-label={`replay ${SHORT[v]}, ${SCHEDULES[i]}`}>
                          <OutcomeChip o={r.outcome} detail={r.outcome === 'wrong' ? (Number.isNaN(r.output) ? 'got garbage' : `got ${fmtValue(r.output)}`) : r.outcome === 'correct' && r.issues.length ? 'a read raced' : undefined} />
                        </button>
                      ) : (
                        <span className="muted">?</span>
                      )}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="row" style={{ margin: '0.8rem 0 1.1rem' }}>
        <button className="primary" onClick={() => setShown(true)} disabled={shown}>Run all four orders</button>
        {!allGuessed && !shown ? <span className="muted">Choose a prediction for every row first, or run anyway.</span> : null}
        {shown ? (
          <span data-readout="score">
            {VARIANTS.filter((v) => guess[v] && guess[v] === rowClass(table[v].map((r) => r.outcome))).length} of {VARIANTS.length} predictions right.
            Click a cell to replay it below.
          </span>
        ) : null}
      </div>

      <div className="panel" data-part="replay">
        <div className="row" style={{ marginBottom: '0.7rem' }}>
          <label className="field">
            Variant
            <select value={variant} onChange={(e) => setVariant(e.target.value as Variant)} aria-label="Variant">
              {VARIANTS.map((v) => <option key={v} value={v}>{VARIANT[v].label}</option>)}
            </select>
          </label>
          <label className="field">
            Thread order
            <select value={schedule} onChange={(e) => setSchedule(e.target.value as Schedule)} aria-label="Thread order">
              {SCHEDULES.map((s) => <option key={s} value={s}>{SCHEDULE_LABEL[s]}</option>)}
            </select>
          </label>
        </div>
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <button onClick={() => setN(0)} disabled={at === 0}>Restart</button>
          <button onClick={() => setN(at - 1)} disabled={at === 0}>Back</button>
          <button className="primary" onClick={() => setN(at + 1)} disabled={at >= total}>Step</button>
          <button onClick={nextBarrier} disabled={at >= total}>Run to the next barrier</button>
          <button onClick={() => setN(total)} disabled={at >= total}>Run to the end</button>
          {firstBad ? <button onClick={() => setN(firstBad)}>Go to the first failure</button> : null}
          <span className="muted" aria-live="polite" data-readout="hunt-step">step {at} of {total}</span>
        </div>
        <div className={'narrate' + (cur?.issue ? ' bad' : '')} role="status" data-readout="hunt-narrate" style={{ marginBottom: '0.9rem' }}>
          {cur ? (
            <>
              {describe(variant, run, cur)}
              {cur.issue ? <><br /><b className="t-fail">Failure: {cur.issue.message}.</b></> : null}
            </>
          ) : (
            <>Nothing has run yet. {B} threads, {B * 2} values, shared memory not yet written ("?"). Press Step.</>
          )}
        </div>
        <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 7fr) minmax(0, 5fr)' }}>
          {fn ? (
            <Code
              tabs={[{ label: 'the kernel', code: fn.text, tiers: TIERS, status: CORRECT.has(variant) ? STATUS.snippet : STATUS.wrong }]}
              highlight={highlight}
              caption={<><code>{VARIANT[variant].kernel}</code>, from <code>labs/{HUNT_FILE}</code>.</>}
            />
          ) : (
            <p className="muted">Loading <code>labs/{HUNT_FILE}</code>.</p>
          )}
          <div style={{ minWidth: 0 }}>
            <div className="mx-label t-global">input (global memory)</div>
            <div className="mx global" style={{ gridTemplateColumns: 'repeat(8, auto)', marginBottom: '0.8rem' }} data-part="input">
              {values.map((x, i) => (
                <span key={i} className={cur?.reads.some((r) => r.arr === 'input' && r.index === i) ? 'hot' : ''}>{x}</span>
              ))}
            </div>
            <div className="mx-label t-shared">{arr(variant)} (shared memory, this block)</div>
            <div className="mx shared" style={{ gridTemplateColumns: `repeat(${B}, auto)` }} data-part="shared">
              {st.shared.map((x, i) => {
                const w = (() => {
                  for (let k = at - 1; k >= 0; k--) {
                    const s = run.steps[k];
                    if (s.write?.arr === 'shared' && s.write.index === i) return s.write.version;
                  }
                  return null;
                })();
                const bad = cur?.reads.some((r) => r.arr === 'shared' && r.index === i && r.issue);
                const hot = cur?.write?.arr === 'shared' && cur.write.index === i;
                const read = cur?.reads.some((r) => r.arr === 'shared' && r.index === i);
                return (
                  <span key={i} className={bad ? 'bad' : hot ? 'hot' : read ? 'on' : ''} data-slot={i} data-value={fmtValue(x)}>
                    {fmtValue(x)}
                    {w ? <sub>{VARIANT[variant].two ? (w.phase === 0 ? 'max' : 'sum') : ''}{w.level}</sub> : null}
                  </span>
                );
              })}
            </div>
            {cur?.reads.some((r) => r.issue === 'oob') ? (
              <p className="t-fail" style={{ margin: '0.4rem 0 0' }}>The read of {arr(variant)}[{cur.reads.find((r) => r.issue === 'oob')!.index}] is past the end of the array.</p>
            ) : null}
            <p style={{ margin: '0.8rem 0 0.5rem' }} data-readout="hunt-output">
              <span className="mono">*output</span> = {Number.isNaN(st.output) ? (at >= total ? 'garbage or never written' : 'not written yet') : fmtValue(st.output)}
              {at >= total ? <> <OutcomeChip o={run.outcome} /></> : null}
            </p>
            <table className="threads" aria-label="Threads of the block">
              <thead>
                <tr><th>thread</th><th>status</th>{VARIANT[variant].two ? <th className="t-reg">m</th> : null}</tr>
              </thead>
              <tbody>
                {Array.from({ length: B }, (_, t) => (
                  <tr key={t} className={cur?.thread === t ? 'cur' : ''}>
                    <td>{t}</td>
                    <td>{st.status[t] === 'waiting' ? `waiting on line ${run.steps[st.last[t]].op!.lines[0]}` : st.status[t]}</td>
                    {VARIANT[variant].two ? <td className="t-reg">{regs(t) == null ? '·' : fmtValue(regs(t)!)}</td> : null}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="muted" style={{ margin: '0.5rem 0 0', fontSize: '0.78rem' }}>{L} iterations of the tree; stride {strides('shared', B).join(', ')}.</p>
          </div>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- global traffic
const TRAFFIC_N = [64, 128, 256, 512, 1024, 2048];

function Traffic() {
  const [k, setK] = useState(2);
  const N = TRAFFIC_N[k];
  const rows = (['simple', 'convergent', 'shared'] as const).map((kind) => ({ kind, total: requests(kind, N, 32).total }));
  const max = Math.max(...rows.map((r) => r.total));
  const label = { simple: 'simple (10.3)', convergent: 'convergent (10.4)', shared: 'shared memory (10.6)' };
  return (
    <Bench title="Global traffic, three kernels" task="Slide N: how do the three counts grow?"
      foot={<>32-wide warps, and the request rule of section 10.5 (ours: one request per aligned piece of 32 floats a warp touches in one access). The book's formulas count whole warps, so N starts at 64.</>}>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field" style={{ flexDirection: 'row', alignItems: 'center', gap: '0.6rem' }}>
          N
          <input type="range" min={0} max={TRAFFIC_N.length - 1} value={k} onChange={(e) => setK(Number(e.target.value))} aria-label="N" />
          <b data-readout="traffic-n">{N}</b> values
        </label>
      </div>
      <div data-part="traffic">
        {rows.map((r) => (
          <div key={r.kind} className="row" style={{ gap: '0.8rem', marginBottom: '0.35rem', flexWrap: 'nowrap' }} data-kind={r.kind} data-total={r.total}>
            <span style={{ width: '10.5rem', flex: 'none' }}>{label[r.kind]}</span>
            <span style={{ flex: 1, minWidth: 0 }}>
              <span style={{ display: 'block', height: '0.9rem', width: `${Math.max(0.5, (r.total / max) * 100)}%`, background: 'var(--global)', borderRadius: 2 }} />
            </span>
            <span style={{ width: '4.5rem', textAlign: 'right', flex: 'none', fontVariantNumeric: 'tabular-nums' }}>{r.total}</span>
          </div>
        ))}
      </div>
      <p style={{ margin: '0.8rem 0 0' }} data-readout="traffic-formula">
        Shared memory: N/32 + 1 = {N}/32 + 1 = {N / 32} coalesced loads (two per warp, {N / 64} warps) + 1 store = <b>{N / 32 + 1}</b>.
        The simple kernel needs {(rows[0].total / rows[2].total).toFixed(1)} times as many, the convergent one{' '}
        {(rows[1].total / rows[2].total).toFixed(1)} times.
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS106: Record<string, Question> = {
  'ch10/10.6/q-barrier-top': {
    kind: 'choice',
    id: 'ch10/10.6/q-barrier-top',
    prompt: <>Why does Fig. 10.10 put <code>__syncthreads()</code> at the top of the loop body?</>,
    options: [
      { label: 'To make the loop faster: at the top, threads reach their addition sooner after the barrier.', why: 'A barrier only ever adds waiting; the question is what it must order.' },
      {
        label: 'So the load\'s shared-memory writes, and each iteration\'s, are done before the next iteration reads them.',
        correct: true,
      },
      { label: 'No reason: the bottom of the loop would order the same writes and reads just as well.', why: 'Then nothing separates the first iteration from the load. Run "barrier at the bottom" in the hunt.' },
      { label: 'Because a barrier at the bottom would deadlock in the last iteration, when only thread 0 is active.', why: 'It does not deadlock; every thread still reaches it. It leaves the load unprotected.' },
    ],
    explain: (
      <p>
        At the top, the first barrier separates the load from the first iteration, and each later one separates an
        iteration from the one before. No barrier is needed after the loop: the last iteration's only writer is thread 0,
        and thread 0 is the one that reads <code>input_s[0]</code>. The book states the first point (p. 236); the second
        is our reading of the code.
      </p>
    ),
    source: 'ch. 10 §10.6, p. 236; notes/ch10.md (no barrier after the loop: inferred)',
  },
  'ch10/10.6/q-blockdim': {
    kind: 'choice',
    id: 'ch10/10.6/q-blockdim',
    prompt: (
      <>
        <code>BLOCK_DIM</code> is compiled as 1024, but the kernel of Fig. 10.10 is launched with 512 threads on a row of
        1024 values. What goes wrong?
      </>
    ),
    options: [
      { label: 'Nothing: the loop starts at blockDim.x/2, so the kernel adapts to the launched size.', why: 'The loop does. The load does not: it uses BLOCK_DIM.' },
      { label: 'Thread t loads input[t + 1024], past the row, and values 512 to 1023 are never loaded.', correct: true },
      { label: 'The launch fails, because the shared array is larger than the block.', why: 'A shared array may be larger than needed; nothing checks it against the block.' },
    ],
    explain: (
      <p>
        The load pairs element t with element <code>t + BLOCK_DIM</code>, a compile-time constant, while the tree uses{' '}
        <code>blockDim.x</code>, the launched size. They must agree, and the block must be a power of two for the tree to
        halve cleanly. The book mixes the two without saying so; this rule is our reading. Lab 10.1's version uses{' '}
        <code>blockDim.x</code> in both places and sizes the shared array for the largest block.
      </p>
    ),
    source: 'ch. 10 §10.6, Fig. 10.10, p. 237; notes/ch10.md "Unclear in the PDF" (inferred)',
  },
  'ch10/10.6/q-req-shared': {
    kind: 'numeric',
    id: 'ch10/10.6/q-req-shared',
    prompt: <>The shared-memory kernel on 2048 values, 32-wide warps. How many global-memory requests does it make?</>,
    answer: 65,
    explain: (
      <p>
        1024 threads in 32 warps, each warp loading two coalesced pieces of 32 floats (<code>input[t]</code> and{' '}
        <code>input[t + 1024]</code>): 64 requests. Then one store of the total. N/32 + 1 = 65, against 204 for the
        convergent kernel and 1149 for the simple one.
      </p>
    ),
    source: 'ch. 10 §10.6, pp. 236–237 (the formula); the 2048 case is ours',
  },
  'ch10/10.6/q-two-reductions': {
    kind: 'choice',
    id: 'ch10/10.6/q-two-reductions',
    prompt: (
      <>
        A softmax kernel reduces the max into <code>scratch[0]</code>, every thread reads <code>scratch[0]</code>, and then
        the sum reduction starts writing <code>scratch[t]</code>. What is needed between the read and the writes, and what
        kind of dependence does it protect?
      </>
    ),
    options: [
      { label: 'Nothing: each thread reads the max before it writes, in program order.', why: 'Program order holds within a thread. Thread 0 can write scratch[0] before thread 5 has read it.' },
      { label: 'A barrier, for a read-after-write (true) dependence on the max.', why: 'That one is protected by the barrier before the read. The question is about the gap after it.' },
      {
        label: 'A barrier, for a write-after-read (false) dependence; a second array would remove it.',
        correct: true,
      },
    ],
    explain: (
      <p>
        The sum's first writes reuse the slot every thread is still reading the max from. The writer needs nothing from
        the readers; they just must not lose the value. That is chapter 5's false dependence (§5.4): give the sum its own
        scratch array and the barrier goes away, at the cost of more shared memory. PyTorch and llama.cpp both keep one
        buffer and the barrier, with a comment saying why.
      </p>
    ),
    source: 'ch. 5 §5.4, pp. 110–111; notes/today/ch10-facts.md D1, D4',
  },
};

// ---------------------------------------------------------------- the page
export function S106() {
  const status = useLabStatus(LAB);
  const cu = useLabFile(LAB + '/solution/kernel.cu');
  const metal = useLabFile(LAB + '/solution/kernel.metal');
  // The lab files separate their kernels with "// ---- (a)" rulers; leave those out of the listing.
  const strip = (t: string) => t.split('\n').filter((l, i, all) => !(i < all.length - 1 && /^\/\/ -{8,}/.test(l))).join('\n');
  const kf = !missing(cu) ? functionSource(cu!, 'SharedMemorySumReductionKernel') : null;
  const mf = !missing(metal) ? functionSource(metal!, 'shared_memory_sum') : null;
  const k = kf ? { ...kf, text: strip(kf.text) } : null;
  const m = mf ? { ...mf, text: strip(mf.text) } : null;
  const r256 = { c: requests('convergent', 256, 32).total, s: requests('shared', 256, 32).total };
  return (
    <>
      <p>
        The convergent kernel still writes every partial sum to global memory and reads it back in the next iteration. Some
        of that traffic may be served by the last-level cache, but even there latency and bandwidth are, in the book's
        words, significant limitations. Shared memory is much faster than both. So keep the tree there.
      </p>
      <Source section="10.6" pages="236" />

      <h2>The tree in shared memory</h2>
      <p>
        Each thread loads its two original values from global memory, adds them, and stores the sum in a{' '}
        <code>__shared__</code> array. That addition is the first level of the tree, so the loop starts at{' '}
        <code>blockDim.x/2</code>. Every later level reads and writes shared memory only. The barrier moves to the top of
        the loop body, where it also separates the load from the first iteration. At the end thread 0 writes the total to
        global memory. Global accesses drop to N reads and one write, and the input is no longer overwritten.
      </p>
      <NoteGroup>
        <Source section="10.6" pages="236–237" figs="Figs. 10.9, 10.10" />
        <Aside>
          <p>
            Fig. 10.10's header has only <code>float* input</code>, but its body writes <code>*output</code>. On p. 236 the
            text cites "line 04" for the loop (it is line 05) and calls the last write "the final write to input[0]" (the code
            writes <code>*output</code>); on p. 237, "the shared memory kernel in Fig. 10.9" means Fig. 10.10.
          </p>
        </Aside>
      </NoteGroup>
      {k ? (
        <Code
          tabs={[
            { label: 'CUDA (lab 10.1)', code: k.text, tiers: TIERS, status: status.cu },
            ...(m ? [{ label: 'Metal (lab 10.1)', code: m.text, tiers: TIERS, status: status.metal }] : []),
          ]}
          caption={
            <>
              Lab 10.1's reference solution, from <code>labs/{LAB}/solution/</code>. Write it yourself in the{' '}
              <a href={href('ch10', 'labs')}>lab</a> before you study this one. Three differences from the page: the output
              parameter is restored, <code>blockDim.x</code> replaces the book's compile-time <code>BLOCK_DIM</code> (the
              shared array is sized for the largest block), and block b works on row b, as in sections 10.3 and 10.4.
            </>
          }
        />
      ) : (
        <p className="muted">The kernel is loaded from <code>labs/{LAB}/solution/kernel.cu</code>.</p>
      )}
      <p>
        Three things the code relies on that the book leaves unsaid. No barrier is needed after the loop: in the last
        iteration only thread 0 writes, and thread 0 is the one that reads the result. <code>BLOCK_DIM</code> must equal
        the launched <code>blockDim.x</code>. And the block must be a power of two, or the halving tree drops values.
      </p>
      <Beyond title="Our reading">
        <p>These three are inferred from the code (notes/ch10.md marks them so), not stated in the book.</p>
      </Beyond>

      <h2>Where does the barrier go?</h2>
      <p>
        Fig. 10.10 and four ways to get it wrong: the barrier moved to the bottom of the loop, dropped, or put inside the
        if, and the active test loosened to <code>&lt;=</code>. Then two kernels that run two reductions through one scratch
        array, as a softmax does (a max, then a sum that every thread computes with the max), with and without a barrier
        between them. For each, decide what happens over the four thread orders before you run them: correct in every
        order, wrong in some, stuck at a barrier, or a read out of bounds. Start with all-ones input, where the sum must be
        16.
      </p>
      <Simplification>
        <p>
          The same model as section 10.3's stepper: one source-level operation per step, "lockstep" with all reads of a
          round before its writes, unwritten shared memory read as "?". It holds the book's rule that every thread of a
          block reaches the same barrier, and reports a block that cannot leave one as stuck.
        </p>
      </Simplification>
      <BarrierHunt />
      <p>
        Two results deserve a second look. "Barrier at the bottom" and "no barrier" fail only in some orders: in lockstep,
        and with the highest thread first, every value happens to be in place before it is read. And "barrier inside the
        if" is stuck in every order. Under the book's rule, and Metal's, a barrier must be reached by every thread of the
        block, and threads 4 to 7 never enter the if. On a GPU it may even finish, since NVIDIA, and Apple for Apple
        silicon, let a thread that has exited off a barrier; it is still wrong.
      </p>
      <Today checked="2026-11-20 (ch04-facts A6, B3, C3; ch05-facts C2)">
        <p>
          NVIDIA allows a barrier in a conditional only if the condition is the same for the whole block; Metal requires
          every thread to enter it; AMD states no rule.
        </p>
      </Today>
      <p>
        The course's CPU emulator follows NVIDIA's wording: exited threads do not hold a barrier. Run on it, the
        barrier-inside-the-if kernel passes in all three of the emulator's thread orders. A test that passes does not make
        a kernel correct; it says this order, on this machine, went well. The emulator does catch the{' '}
        <code>&lt;=</code> variant: AddressSanitizer reports the read past the end of <code>input_s</code> on the line of
        the addition.
      </p>
      <Beyond title="Run, not assumed">
        <p>
          Both emulator results come from running the snippet file on the emulator in <code>app/src/sim/reduction.test.ts</code>,
          which also checks that the emulator and this model agree in the orders they share.
        </p>
      </Beyond>
      <Ask q={questionsS106['ch10/10.6/q-barrier-top']} />
      <Ask q={questionsS106['ch10/10.6/q-blockdim']} />

      <h2>What it saves</h2>
      <p>
        With the tree in shared memory, global memory sees N reads and one write, N + 1 accesses in all. Both reads of each
        thread are coalesced, so they take N/32 requests, plus one for the write. For 256 values: from {r256.c} requests in
        the convergent kernel to 8 + 1 = {r256.s}, four times fewer. And the input array is left as it was.
      </p>
      <NoteGroup>
        <Source section="10.6" pages="236–237" />
        <Simplification>
          <p>
            The request rule of section 10.5's explorer, ours: no caches, no timing, no transaction sizes.
          </p>
        </Simplification>
      </NoteGroup>
      <Traffic />
      <Ask q={questionsS106['ch10/10.6/q-req-shared']} label="Work it out" />

      <h2>Two reductions through one scratch array</h2>
      <p>
        A softmax over a row reduces twice: a max, which every thread then needs, and a sum of exp(x − max). In the bug
        hunt, "max then sum, without it" runs both through one shared scratch array with no barrier between them, and shows
        what goes wrong: a fast thread starts writing the sum into the slot a slow thread has not read the max from yet.
        That is a write-after-read hazard, the false dependence of chapter 5 (§5.4).
      </p>
      <Beyond>
        <p>The book does not discuss softmax.</p>
      </Beyond>
      <p>
        Production kernels keep the one array and the barrier, and say why in a comment. PyTorch's{' '}
        <code>BlockReduceSum</code> starts with a <code>__syncthreads()</code> so that it can be called twice in a row, and
        llama.cpp's softmax puts one between its max and its sum.
      </p>
      <Today checked="2026-11-20 (ch10-facts D1, D4)">
        <p>
          PyTorch: "prevent races when BlockReduces are called in a row". llama.cpp: "sync is needed as we reuse buf_iw
          across block_reduce invocations".
        </p>
      </Today>
      <Ask q={questionsS106['ch10/10.6/q-two-reductions']} />

      <h2>The same kernel on other backends</h2>
      <p>
        Switch tabs in the listing above. This kernel uses only what HIP spells as CUDA does (threads, shared memory, a
        barrier), so the lab's file is written to compile for HIP as it stands; that is unverified, since nothing here has
        been built with ROCm. In Metal the shared array is declared{' '}
        <code>threadgroup</code> in the kernel body, and this time the barrier's flag is{' '}
        <code>mem_flags::mem_threadgroup</code>: the partial sums now live in threadgroup memory, which is what that flag
        orders. Compare section 10.3, where they lived in device memory and needed <code>mem_device</code>.
      </p>
      <Beyond>
        <p>
          The book uses CUDA only. notes/today/ch05-facts.md B4, C1, C2; ch04-facts.md A6, B3, C3; ch10-facts.md, "Where the
          backends really differ".
        </p>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th>Concept</th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>Block-shared array</td>
            <td><code>__shared__ float input_s[..]</code></td>
            <td>same spelling</td>
            <td><code>threadgroup float input_s[..]</code> in the kernel body</td>
          </tr>
          <tr>
            <td>Barrier over shared memory</td>
            <td><code>__syncthreads()</code></td>
            <td>same spelling; "implicitly include a threadfence"</td>
            <td><code>threadgroup_barrier(mem_flags::mem_threadgroup)</code></td>
          </tr>
          <tr>
            <td>Barrier in a conditional</td>
            <td>only if the condition is the same for the whole block</td>
            <td>no rule stated in AMD's pages; follow CUDA's</td>
            <td>every thread must enter the conditional</td>
          </tr>
        </tbody>
      </table>
    </>
  );
}
