import { useEffect, useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code, useLabFile } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { href } from '../../lib/router';
import { STATUS, useLabStatus } from './status';
import {
  EXERCISE_6,
  FIG_VALUES,
  INPLACE_FILE,
  INPLACE_KERNEL,
  firstIssueStep,
  fmtValue,
  historyAt,
  isActive,
  isPow2,
  ownedCell,
  runInPlace,
  stateAt,
  strides,
  type HistoryRow,
  type InPlaceKind,
  type Schedule,
  type ThreadRun,
  type TStep,
} from '../../sim/reduction';
import { SCHEDULE_LABEL } from '../../sim/tiled';
import { functionSource } from '../../sim/toolchain';

import { NoteGroup } from './NoteGroup';

// The lab whose runs give the status of the listings taken from its files (and from labs/ch10/shared/, which it runs).
const LAB10_1 = 'ch10/lab1_sum';
const SCHEDULES: Schedule[] = ['lockstep', 'in-order', 'reversed', 'shuffled'];
/** Identifiers coloured by the memory they live in. */
const TIERS: Record<string, Tier> = { input: 'global', output: 'global', row: 'global', i: 'reg', stride: 'reg', t: 'reg', b: 'reg', bdim: 'reg' };
const missing = (s: string | null) => !s || s.startsWith('// could not load');

/** The Metal kernel of a file, by name, or null. */
const metalFn = (src: string | null, name: string) => (missing(src) ? null : functionSource(src!, name));

// ---------------------------------------------------------------- the stepper
const DATA = [
  { key: 'fig', label: '16 values, 8 threads (Fig. 10.6)' },
  { key: 'ex6', label: 'exercise 6: 6, 2, 7, 4, 5, 8, 3, 1' },
  { key: 'own', label: 'your own values' },
] as const;
type DataKey = (typeof DATA)[number]['key'];

/** Parses 2, 4, 8 or 16 numbers. */
function parseValues(s: string): number[] | null {
  const parts = s.split(/[\s,]+/).filter(Boolean);
  const v = parts.map((p) => Number(p.replace('−', '-')));
  if (!v.length || v.some((x) => !Number.isFinite(x))) return null;
  return isPow2(v.length) && v.length >= 2 && v.length <= 16 ? v : null;
}

/** The iteration a barrier opening closes: the iteration of the last operation before it. */
function iterBefore(run: ThreadRun, index: number): number {
  for (let i = index - 1; i >= 0; i--) if (run.steps[i].op) return run.steps[i].op!.iter;
  return 0;
}

/** One sentence for a step, with the arithmetic spelled out. */
function describeInPlace(kind: InPlaceKind, run: ThreadRun, s: TStep): string {
  if (s.thread === -1) {
    const k = iterBefore(run, s.index);
    return `The barrier opens: all ${run.threads} threads have finished iteration ${k}. Every partial sum of that iteration is in place before any thread starts the next.`;
  }
  const op = s.op!;
  const t = s.thread;
  const test = kind === 'simple' ? `${t} % ${op.stride}` : `${t} < ${op.stride}`;
  switch (op.kind) {
    case 'skip':
      return kind === 'simple'
        ? `Thread ${t}, iteration ${op.iter} (stride ${op.stride}): ${test} is ${t % op.stride}, not 0, so it skips the addition.`
        : `Thread ${t}, iteration ${op.iter} (stride ${op.stride}): ${test} is false, so it skips the addition.`;
    case 'add': {
      const [own, partner] = s.reads;
      return (
        `Thread ${t}, iteration ${op.iter} (stride ${op.stride}): ${kind === 'simple' ? `${test} == 0` : `${test}`}, so it adds ` +
        `input[${partner.index}] (${fmtValue(partner.value)}) into input[${own.index}] (${fmtValue(own.value)}): input[${own.index}] = ${fmtValue(s.write!.value)}.`
      );
    }
    case 'barrier':
      return `Thread ${t} reaches __syncthreads() at the end of iteration ${op.iter} and waits for the rest of the block.`;
    case 'store':
      return `Thread 0 copies input[0] = ${fmtValue(s.write!.value)} to *output.`;
    default:
      return '';
  }
}

/** Fig. 10.6's layout: the input, then one row per iteration with the elements it wrote and the two each one came from. */
function HistoryGrid({ rows, threads, kind, cur }: { rows: HistoryRow[]; threads: number; kind: InPlaceKind; cur?: TStep }) {
  const N = rows[0].values.length;
  const CW = N > 8 ? 34 : 46;
  const CH = 24;
  const GAP = 24;
  const LW = 74;
  const TOP = 18;
  const y = (r: number) => TOP + r * (CH + GAP);
  const x = (c: number) => LW + c * CW;
  const W = LW + N * CW + 4;
  const H = y(rows.length - 1) + CH + 4;
  const curIter = cur?.op && cur.op.iter > 0 ? cur.op.iter : null;
  const curOwn = cur?.op?.kind === 'add' ? cur.reads[0].index : null;
  const curPartner = cur?.op?.kind === 'add' ? cur.reads[1].index : null;
  const activeNow = (t: number) => curIter != null && isActive(kind, t, strides(kind, threads)[curIter - 1]);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="The input and the array after each iteration" data-part="history"
      style={{ width: W, maxWidth: '100%', minWidth: Math.min(W, 460), display: 'block' }}>
      <defs>
        <marker id="s103-arrow" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
          <path d="M0,0 L6,3 L0,6 z" fill="var(--ink-3)" />
        </marker>
        <marker id="s103-arrow-hot" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
          <path d="M0,0 L6,3 L0,6 z" fill="var(--ink)" />
        </marker>
      </defs>
      {Array.from({ length: threads }, (_, t) => (
        <text key={t} x={x(ownedCell(kind, t)) + CW / 2} y={11} fontSize="9.5" textAnchor="middle" data-thread-label={t}
          style={{ fill: activeNow(t) ? 'var(--ink)' : 'var(--ink-3)', fontWeight: activeNow(t) ? 700 : 400 }}>
          t{t}
        </text>
      ))}
      {rows.map((row, r) => (
        <g key={r} data-row={r}>
          <text x={LW - 8} y={y(r) + CH / 2 + 4} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
            {r === 0 ? 'input' : `stride ${row.stride}`}
          </text>
          {r > 0
            ? row.adds.map((a) => {
                const hot = curIter === r && curOwn === a.own && cur?.thread === a.thread;
                const stroke = hot ? 'var(--ink)' : 'var(--ink-3)';
                const marker = `url(#${hot ? 's103-arrow-hot' : 's103-arrow'})`;
                const tx = x(a.own) + CW / 2;
                return (
                  <g key={a.own} opacity={hot ? 1 : 0.75}>
                    <line x1={tx} y1={y(r - 1) + CH} x2={tx} y2={y(r) - 1} stroke={stroke} strokeWidth={hot ? 1.6 : 1} markerEnd={marker} />
                    <line x1={x(a.partner) + CW / 2} y1={y(r - 1) + CH} x2={tx + (a.partner > a.own ? 3 : -3)} y2={y(r) - 1}
                      stroke={stroke} strokeWidth={hot ? 1.6 : 1} markerEnd={marker} />
                  </g>
                );
              })
            : null}
          {row.values.map((v, c) => {
            const written = r === 0 || row.written[c];
            const bad = row.bad[c];
            const hot = (curIter === r && curOwn === c) || (curIter === r + 1 && (curPartner === c || curOwn === c) && cur?.op?.kind === 'add');
            return (
              <g key={c} data-cell={c} data-value={fmtValue(v)} data-written={written ? '1' : '0'} data-bad={bad ? '1' : '0'}>
                <rect x={x(c) + 1} y={y(r)} width={CW - 2} height={CH} rx="2"
                  fill={bad ? 'var(--fail-wash)' : written ? 'var(--global-wash)' : 'transparent'}
                  stroke={bad ? 'var(--fail)' : hot ? 'var(--ink)' : written ? 'var(--global)' : 'var(--rule)'}
                  strokeWidth={hot ? 2 : 1} strokeDasharray={written ? undefined : '3 2'} />
                <text x={x(c) + CW / 2} y={y(r) + CH / 2 + 4} fontSize="11" textAnchor="middle"
                  style={{ fill: bad ? 'var(--fail)' : written ? 'var(--ink)' : 'var(--ink-3)', fontWeight: written && r > 0 ? 650 : 400 }}>
                  {fmtValue(v)}
                </text>
              </g>
            );
          })}
        </g>
      ))}
    </svg>
  );
}

function Stepper() {
  const src = useLabFile(INPLACE_FILE);
  const [data, setData] = useState<DataKey>('fig');
  const [own, setOwn] = useState('5, 1, 4, 2, 8, 3, 6, 7');
  const [schedule, setSchedule] = useState<Schedule>('in-order');
  const [barrier, setBarrier] = useState(true);
  const [n, setN] = useState(0);
  const [playing, setPlaying] = useState(false);

  const parsed = parseValues(own);
  const values = data === 'fig' ? [...FIG_VALUES] : data === 'ex6' ? [...EXERCISE_6] : parsed;
  const key = values ? values.join(',') : '';
  const run = useMemo(() => (values ? runInPlace('simple', values, { barrier, schedule }) : null), [key, barrier, schedule]); // eslint-disable-line react-hooks/exhaustive-deps
  const total = run ? run.steps.length : 0;
  const at = Math.min(n, total);
  const cur = run && at > 0 ? run.steps[at - 1] : undefined;
  const st = run ? stateAt(run, at) : null;
  const rows = run ? historyAt(run, at) : null;
  const firstBad = run ? firstIssueStep(run) : null;
  const fn = !missing(src) ? functionSource(src!, INPLACE_KERNEL.simple) : null;

  useEffect(() => {
    setN(0);
    setPlaying(false);
  }, [key, barrier, schedule]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setN((v) => v + 1), 380);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);

  const toBarrier = () => {
    if (!run) return;
    for (let i = at; i < total; i++) if (run.steps[i].thread === -1) return setN(i + 1);
    setN(total);
  };
  const highlight = cur?.op ? cur.op.lines : cur?.thread === -1 ? [7] : [];
  const done = run != null && at >= total;
  const T = run?.threads ?? 0;

  return (
    <Bench
      title="Step the simple kernel"
      task="Follow thread 0 to the first barrier. Then untick the barrier and find the first read that goes wrong."
      foot={
        <>
          Each row below the input shows what one iteration wrote (solid) over what the iteration before left (dashed),
          with arrows from the two elements an addition read. A red element was written from a value that was not final
          yet. The labels above the input are the threads, over the element each one owns; the bold ones are active in
          the current iteration.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Data
          <select value={data} onChange={(e) => setData(e.target.value as DataKey)} aria-label="Data">
            {DATA.map((d) => <option key={d.key} value={d.key}>{d.label}</option>)}
          </select>
        </label>
        {data === 'own' ? (
          <label className="field" style={{ minWidth: 0 }}>
            2, 4, 8 or 16 numbers
            <input type="text" value={own} onChange={(e) => setOwn(e.target.value)} aria-label="your values" style={{ width: '17rem', maxWidth: '100%' }} />
          </label>
        ) : null}
        <label className="field">
          Thread order
          <select value={schedule} onChange={(e) => setSchedule(e.target.value as Schedule)} aria-label="Thread order">
            {SCHEDULES.map((s) => <option key={s} value={s}>{SCHEDULE_LABEL[s]}</option>)}
          </select>
        </label>
        <label className="check"><input type="checkbox" checked={barrier} onChange={(e) => setBarrier(e.target.checked)} /> __syncthreads()</label>
      </div>
      {!run ? (
        <p><span className="chip fail">Give 2, 4, 8 or 16 numbers, separated by commas or spaces.</span></p>
      ) : (
        <>
          <div className="row" style={{ marginBottom: '0.8rem' }}>
            <button onClick={() => { setPlaying(false); setN(0); }} disabled={at === 0}>Restart</button>
            <button onClick={() => { setPlaying(false); setN(at - 1); }} disabled={at === 0}>Back</button>
            <button className="primary" onClick={() => { setPlaying(false); setN(at + 1); }} disabled={at >= total}>Step</button>
            <button onClick={() => { setPlaying(false); toBarrier(); }} disabled={at >= total}>{barrier ? 'Run to the barrier' : 'Run to the end'}</button>
            <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? 'Pause' : 'Play'}</button>
            {firstBad ? <button onClick={() => { setPlaying(false); setN(firstBad); }}>Go to the first failure</button> : null}
            <span className="muted" aria-live="polite" data-readout="step">step {at} of {total}</span>
          </div>
          <div className={'narrate' + (cur?.issue ? ' bad' : '')} role="status" data-readout="narrate" style={{ marginBottom: '0.9rem' }}>
            {cur ? (
              <>
                {describeInPlace('simple', run, cur)}
                {cur.issue ? <><br /><b className="t-fail">Failure: {cur.issue.message}.</b></> : null}
              </>
            ) : (
              <>
                Nothing has run yet. {T} threads, {T * 2} values; thread t owns input[2t]. Press Step, or Run to the barrier to
                finish an iteration.
              </>
            )}
          </div>
          <div style={{ minWidth: 0, marginBottom: '0.6rem' }}>
            <div style={{ overflowX: 'auto' }}>
              <HistoryGrid rows={rows!} threads={T} kind="simple" cur={cur} />
            </div>
            <p style={{ margin: '0.7rem 0 0' }} data-readout="output">
              <span className="mono">*output</span> = {Number.isNaN(st!.output) ? 'not written yet' : fmtValue(st!.output)}
              {done ? (
                <>
                  {' '}
                  {run.outcome === 'correct' ? (
                    <span className="chip ok" data-outcome="correct">the sum, {fmtValue(run.expected)}</span>
                  ) : (
                    <span className="chip fail" data-outcome="wrong">wrong: the sum is {fmtValue(run.expected)}</span>
                  )}
                  {run.issues.length ? <span className="muted"> {run.issues.length} read{run.issues.length > 1 ? 's' : ''} of a value that was not final.</span> : null}
                </>
              ) : null}
            </p>
          </div>
          <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 7fr) minmax(0, 4fr)' }}>
            {fn ? (
              <Code
                tabs={[{ label: 'Fig. 10.5, one block', code: fn.text, tiers: TIERS, status: STATUS.snippet }]}
                highlight={highlight}
              />
            ) : (
              <p className="muted">Loading the kernel from <code>labs/{INPLACE_FILE}</code>.</p>
            )}
            <table className="threads" aria-label="Threads of the block" style={{ marginTop: '1.1rem' }}>
              <thead>
                <tr><th>thread</th><th>owns</th><th>status</th></tr>
              </thead>
              <tbody>
                {Array.from({ length: T }, (_, t) => (
                  <tr key={t} className={cur?.thread === t ? 'cur' : ''} data-thread={t}>
                    <td>{t}</td>
                    <td className="t-global">input[{2 * t}]</td>
                    <td>{st!.status[t] === 'waiting' ? 'waiting at the barrier' : st!.status[t]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS103: Record<string, Question> = {
  'ch10/10.3/q-ex6a': {
    kind: 'numeric',
    id: 'ch10/10.3/q-ex6a',
    prompt: (
      <>
        Book exercise 6a: run the simple kernel on the eight values {'{'}6, 2, 7, 4, 5, 8, 3, 1{'}'}. What does{' '}
        <code>input[4]</code> hold when the kernel ends?
      </>
    ),
    answer: 17,
    explain: (
      <p>
        Four threads, strides 1, 2 and 4. After stride 1 the array is {'{'}8, 2, 11, 4, 13, 8, 4, 1{'}'}; after stride 2,{' '}
        {'{'}19, 2, 11, 4, 17, 8, 4, 1{'}'}; after stride 4 only <code>input[0]</code> changes, to 36. Element 4 keeps the
        partial sum of the second half, 5 + 8 + 3 + 1 = 17: the input is gone, and the leftovers are partial sums. The book
        prints no answer; this one is ours, and the stepper above reproduces it.
      </p>
    ),
    source: 'ch. 10 §10.12, exercise 6a, p. 250; answer ours',
  },
  'ch10/10.3/q-stale-read': {
    kind: 'choice',
    id: 'ch10/10.3/q-stale-read',
    prompt: <>Remove the barrier and run the kernel on 16 values with 8 threads. Which read can see a value that is not final yet?</>,
    options: [
      { label: 'At stride 1, thread 0 reads input[1], which thread 1 is updating in the same iteration.', why: 'Odd elements belong to no thread. Nobody ever writes input[1], so there is nothing to wait for.' },
      { label: 'At stride 2, thread 0 reads input[2], which thread 1 writes in the iteration before.', correct: true },
      { label: 'At stride 2, thread 1 reads input[4], which thread 2 writes in the iteration before.', why: 'Thread 1 is not active at stride 2: 1 % 2 is not 0.' },
      { label: 'None: each thread reads only elements it owns or elements nobody writes.', why: 'The partner element input[i + stride] belongs to another thread from the second iteration on.' },
    ],
    explain: (
      <p>
        From the second iteration on, the partner element is another thread's partial sum: input[2] is thread 1's, input[4]
        thread 2's, and so on. Without the barrier nothing stops thread 0 from reaching its second addition before thread 1
        has done its first. In the stepper, untick the barrier with the order "each thread runs as far as it can, thread 0
        first" and press <i>Go to the first failure</i>.
      </p>
    ),
    source: 'ch. 10 §10.3, pp. 229–230, Fig. 10.6',
  },
  'ch10/10.3/q-barrier-outside': {
    kind: 'choice',
    id: 'ch10/10.3/q-barrier-outside',
    prompt: (
      <>
        Why is <code>__syncthreads()</code> outside the <code>if</code>, so that threads that skip the addition execute it
        too?
      </>
    ),
    options: [
      { label: 'Every thread of the block must reach the barrier; inside the if, inactive threads never would.', correct: true },
      { label: 'So that the inactive threads can pick up additions the active ones have not finished.', why: 'They do no addition; they only wait.' },
      { label: 'It is a style choice: inside the if, it would wait for the active threads, which is all it needs.', why: 'A barrier that only some threads reach breaks the rule of chapter 4.' },
      { label: 'To make the loop run faster, since all threads then leave the barrier together.', why: 'A barrier only ever adds waiting.' },
    ],
    explain: (
      <p>
        Chapter 4's rule: a barrier is executed by every thread of the block or by none. Put it inside the{' '}
        <code>if</code> and from the second iteration on the inactive threads skip it, so the active ones wait for threads
        that never come. NVIDIA calls a barrier under a condition that is not the same for the whole block undefined
        behaviour: it may hang, or carry on and compute something wrong. Section 10.6's bug hunt runs exactly this mistake.
      </p>
    ),
    source: 'ch. 4 §4.3; ch. 10 §10.3, p. 230; notes/today/ch04-facts.md A6',
  },
  'ch10/10.3/q-three-blocks': {
    kind: 'choice',
    id: 'ch10/10.3/q-three-blocks',
    prompt: (
      <>
        You need the sum of 5000 values and launch the book's kernel, unchanged, with 3 blocks of 1024 threads. What
        happens?
      </>
    ),
    options: [
      { label: 'Each block reduces its own third of the input, and the last block to finish writes *output.', why: 'Nothing in the kernel uses blockIdx. Every block computes i = 2*threadIdx.x.' },
      { label: 'The answer is right, only slower than one block, because the blocks repeat the same work.', why: 'The three blocks write the same elements at the same time.' },
      {
        label: 'All three reduce the first 2048 values in place at once, racing on them and on *output.',
        correct: true,
      },
      { label: 'The launch fails: a kernel that calls __syncthreads() may only be launched with one block.', why: 'The launch is legal. The barrier simply has no effect between blocks.' },
    ],
    explain: (
      <p>
        The index depends only on <code>threadIdx.x</code>, so every block works on elements 0 to 2047, each overwriting
        partial sums the others are still reading, and no barrier spans blocks to order them. All three write{' '}
        <code>*output</code>. Elements 2048 to 4999 are never touched. Section 10.9 removes the one-block limit by giving
        each block its own segment and combining the block sums; lab 10.1's version of this kernel already offsets{' '}
        <code>input</code> by the block, so that many independent rows can be timed at once.
      </p>
    ),
    source: 'ch. 10 §10.3, pp. 227–228; §10.9, p. 244',
  },
};

// ---------------------------------------------------------------- the page
export function S103() {
  const cu = useLabFile('ch10/shared/inplace_sum.cu');
  const metal = useLabFile('ch10/shared/inplace_sum.metal');
  const status = useLabStatus(LAB10_1);
  const k = !missing(cu) ? functionSource(cu!, INPLACE_KERNEL.simple) : null;
  const m = metalFn(metal, 'simple_sum');
  const barrierLine = m ? m.text.split('\n').findIndex((l) => l.includes('threadgroup_barrier')) + 1 : 0;
  const offsetLines = k ? k.text.split('\n').flatMap((l, i) => (/blockIdx|output\[/.test(l) ? [i + 1] : [])) : [];
  return (
    <>
      <p>
        The tree of the last section needs threads that hand partial sums to each other at every level: a thread may only
        add its partner's value once the partner has finished the level before. Threads can only wait for each other inside
        a block, with the barrier from chapter 4. So the first kernels use a single block. For N values you launch one
        block of N/2 threads; with at most 1024 threads in a block, that is at most 2048 values. Section 10.9 lifts the
        limit.
      </p>
      <NoteGroup>
        <Source section="10.3" pages="227–228" />
        <Aside>
          <p>
            The book says on p. 228 that the 2048-value limit goes "in Section 10.10". It goes in section 10.9, on inputs of
            any length; 10.10 is thread coarsening.
          </p>
        </Aside>
      </NoteGroup>

      <h2>Owner computes</h2>
      <p>
        The input sits in global memory and the kernel overwrites it. Thread t <b>owns</b> element 2t: it is the only
        thread that ever writes there. In the first iteration every thread adds its right-hand neighbour into its own
        element. In the second, the partial sums sit two apart, so the stride doubles to 2, and only every second thread
        has work: thread 0 adds element 2 (which holds the old 2 + 3) into element 0. The stride keeps doubling, and the
        threads that work in an iteration with stride s are those whose index is a multiple of s. That is the test{' '}
        <code>threadIdx.x % stride == 0</code>. At the end, element 0 holds the total, and thread 0 copies it to{' '}
        <code>*output</code>.
      </p>
      <NoteGroup>
        <Source section="10.3" pages="228–230" figs="Figs. 10.5, 10.6" />
        <Aside>
          <p>
            On p. 229 the book says the stride grows until it is greater than <code>blockIdx.x</code>, "the total number of
            threads in the block". The code, rightly, tests <code>blockDim.x</code>.
          </p>
        </Aside>
      </NoteGroup>
      {k ? (
        <Code
          tabs={[
            { label: 'CUDA (lab 10.1)', code: k.text, tiers: TIERS, status: status.cu, highlight: offsetLines },
            ...(m ? [{ label: 'Metal (lab 10.1)', code: m.text, tiers: TIERS, status: status.metal, highlight: barrierLine ? [barrierLine] : [] }] : []),
          ]}
          caption={
            <>
              <code>SimpleSumReductionKernel</code> as lab 10.1 provides it, from <code>labs/ch10/shared/inplace_sum.cu</code>{' '}
              (and <code>simple_sum</code> from <code>inplace_sum.metal</code>). One change from the page, on the marked
              lines: block b works on its own row of 2·blockDim.x values (<code>input</code> moves to the start of row b,
              the result goes to <code>output[blockIdx.x]</code>), so the lab can time many rows in one launch. With one
              block it is the book's kernel. HIP spells everything in it as CUDA does, so the file is written to compile
              for HIP as it stands (unverified: nothing here has been built with ROCm).
            </>
          }
        />
      ) : (
        <p className="muted">The kernel is loaded from <code>labs/ch10/shared/inplace_sum.cu</code>.</p>
      )}

      <h2>Step it</h2>
      <p>
        The stepper runs the book's one-block kernel for every thread of the block. Start with the order "each thread runs
        as far as it can, thread 0 first": thread 0 does its first addition and then stops at the barrier, because the
        others have not caught up. Run to the barrier a few times and watch element 0 collect the total while the other
        even elements keep the partial sums they made on the way.
      </p>
      <Simplification>
        <p>
          A thread advances one source-level operation at a time here, and an addition reads and writes in one step.
          "Lockstep" runs every thread one operation per round, all reads before all writes. No warps, no timing. The
          model is faithful about one thing: which value a thread finds when it reads.
        </p>
      </Simplification>
      <Stepper />
      <Ask q={questionsS103['ch10/10.3/q-ex6a']} label="Work it out, then check it in the stepper" />

      <h2>What the barrier guarantees</h2>
      <p>
        The <code>__syncthreads()</code> inside the loop makes every thread finish one iteration before any thread starts
        the next. That is exactly what the tree needs: a level's partial sums are all written before the next level reads
        them. The book adds a second point. A value one thread writes to memory should not be read by another without a
        memory fence; because writer and reader are in the same block here, the barrier serves as that fence too.
      </p>
      <NoteGroup>
        <Source section="10.3" pages="230" />
        <Today checked="2026-11-20 (ch10-facts A1)">
          <p>
            The call "strongly happens before" any participating thread leaves it. The book's "memory consistency model" is
            now section 5.8 of NVIDIA's guide, where a data race is undefined behaviour.
          </p>
        </Today>
      </NoteGroup>
      <p>
        So a dropped barrier is not merely a chance of reading an old value: by NVIDIA's rules the program then has no
        defined meaning. Untick the barrier in the stepper, keep the same order, and press <i>Go to the first failure</i>. Then try the
        other orders. Some give the right total anyway, which is the worst kind of bug: a test that passes proves nothing
        about the next device.
      </p>
      <Ask q={questionsS103['ch10/10.3/q-stale-read']} />
      <Ask q={questionsS103['ch10/10.3/q-barrier-outside']} />

      <h2>What this kernel costs</h2>
      <p>
        It works, and it is a fair picture of the tree, but carry three costs into the next sections. It destroys its
        input: the array ends full of partial sums. Its active threads are spread through every warp, so most warps
        diverge in most iterations; section 10.4 counts that. And neighbouring threads touch elements that are 2, then 4,
        then 8 apart, so their memory accesses do not combine; section 10.5 counts that. All three come from the same
        choice, that thread t owns element 2t.
      </p>
      <Source section="10.3–10.5" pages="228, 230, 234" />
      <Ask q={questionsS103['ch10/10.3/q-three-blocks']} />

      <h2>The same kernel on other backends</h2>
      <p>
        HIP takes the CUDA file as written (unverified: nothing here has been built with ROCm), with the same{' '}
        <code>__syncthreads()</code>, which AMD documents as a barrier that implicitly includes a thread fence. Metal is where this kernel needs care. Its barrier,{' '}
        <code>threadgroup_barrier(flags)</code>, orders only the kinds of memory named in its flags. The usual flag,{' '}
        <code>mem_flags::mem_threadgroup</code>, covers threadgroup memory, and this kernel keeps its partial sums in device
        memory. So lab 10.1's Metal version passes <code>mem_flags::mem_device</code> (the marked line in the Metal tab).
        With the threadgroup flag it would compile, run, and have no documented reason to be right.
      </p>
      <Beyond>
        <p>
          The book uses CUDA only. HIP: notes/today/ch10-facts.md, "Where the backends really differ" (block barrier as
          memory fence). Metal: notes/today/ch05-facts.md C2, the table of <code>mem_flags</code>.
        </p>
      </Beyond>
      <p>
        You run and time this kernel in <a href={href('ch10', 'labs')}>lab 10.1</a>, beside the convergent kernel of
        section 10.4 and the faster kernels of sections 10.6 to 10.10.
      </p>
    </>
  );
}
