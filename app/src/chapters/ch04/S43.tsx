import { useEffect, useId, useMemo, useRef, useState, type ReactNode, type RefObject } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { fmt, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  EXERCISE_4,
  KERNELS,
  finding,
  run,
  timeline,
  type Finding,
  type KernelId,
  type Order,
  type Rule,
  type Step,
} from '../../sim/barrier';
import {
  FIG45,
  TAIL_EXAMPLE,
  capacity,
  deviceFromReport,
  equalDurations,
  isGpuBackend,
  profile,
  schedule,
  tail,
  unevenDurations,
  usage,
  waveSizes,
  waves,
} from '../../sim/blocksched';
import { WavesScene } from './WavesScene';

// The kernels of this page, exactly as stored in labs/ch04/snippets/<name>.cu, where
// the CUDA front-end check parses them. app/src/sim/barrier.test.ts fails if a copy
// here drifts from its file.
const SNIPPETS: Record<KernelId, string> = {
  incorrect_barrier: `__global__ void incorrect_barrier_example(int n) {
    // ...
    if (threadIdx.x % 2 == 0) {
        // ... work of the even threads
        __syncthreads();
    }
    else {
        // ... work of the odd threads
        __syncthreads();
    }
}`,
  barrier_uniform: `__global__ void barrier_uniform(float* data) {
    __shared__ float s[4];                         // the block has 4 threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = data[i];                      // phase 0: write my slot
    if (blockIdx.x > 0) {
        __syncthreads();
    }
    data[i] = s[(threadIdx.x + 1) % blockDim.x];   // phase 1: read a neighbour's slot
}`,
  barrier_per_thread: `__global__ void barrier_per_thread(float* data) {
    __shared__ float s[4];                         // the block has 4 threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = data[i];                      // phase 0: write my slot
    if (threadIdx.x > 0) {
        __syncthreads();
    }
    data[i] = s[(threadIdx.x + 1) % blockDim.x];   // phase 1: read a neighbour's slot
}`,
  barrier_early_return: `__global__ void barrier_early_return(float* data, int n) {
    __shared__ float s[4];                         // the block has 4 threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;                            // threads past the end of the data leave here
    s[threadIdx.x] = data[i];                      // phase 0: write my slot
    __syncthreads();
    data[i] = s[0];                                // phase 1: read thread 0's slot
}`,
  barrier_loop: `__global__ void barrier_loop(float* data) {
    __shared__ float s[4];                         // the block has 4 threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // odd threads go round the loop twice, even threads once
    for (unsigned k = 0; k <= threadIdx.x % 2; ++k) {
        s[threadIdx.x] = data[i] + k;              // one round: write my slot
        __syncthreads();
    }
    data[i] = s[(threadIdx.x + 1) % blockDim.x];   // after the loop: read a neighbour's slot
}`,
  barrier_guarded: `__global__ void barrier_guarded(float* data, int n) {
    __shared__ float s[4];                         // the block has 4 threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool mine = i < n;                             // threads past the end stay, and do nothing
    if (mine) s[threadIdx.x] = data[i];            // phase 0: write my slot
    __syncthreads();
    if (mine) data[i] = s[0];                      // phase 1: read thread 0's slot
}`,
};
const TIERS: Record<string, Tier> = { s: 'shared', data: 'global', i: 'reg', k: 'reg', mine: 'reg' };
// The files parse with the CUDA front end; none of them is built or run. Two of them have a
// legal barrier (labs/ch04/snippets/README.md), so only the other four are "wrong on purpose".
const SNIPPET_STATUS = 'front-end-checked; wrong on purpose, never run';
const SNIPPET_STATUS_LEGAL = 'front-end-checked; never run';
const statusOf = (id: KernelId) => (id === 'barrier_uniform' || id === 'barrier_guarded' ? SNIPPET_STATUS_LEGAL : SNIPPET_STATUS);

const f1 = (v: number) => v.toFixed(1);
const round1 = (v: number) => Math.round(v * 10) / 10;
const list = (a: number[]) => (a.length <= 1 ? a.join('') : a.slice(0, -1).join(', ') + ' and ' + a[a.length - 1]);
const plural = (a: number[], one: string) => `${one}${a.length === 1 ? '' : 's'} ${list(a)}`;
const wait = (a: number[]) => (a.length === 1 ? 'waits' : 'wait');

/**
 * Width of an element in CSS pixels, so a drawing can use it as its viewBox and keep
 * its text at reading size on a phone. (Chapter 1 has the same helper; a shared one
 * would serve both.)
 */
function useWidth(initial: number, min = 280, max = 640): [RefObject<HTMLDivElement | null>, number] {
  const ref = useRef<HTMLDivElement>(null);
  const [w, setW] = useState(initial);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const read = () => setW(Math.max(min, Math.min(max, Math.round(el.clientWidth))));
    read();
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, [min, max]);
  return [ref, w];
}

// ---------------------------------------------------------------- barrier timeline
const AXIS = 5.5; // microseconds shown

/** Fig. 4.3 redrawn from the model: work, then waiting, then everyone continues together. */
function TimelineView({ durations, onChange }: { durations: readonly number[]; onChange?: (i: number, v: number) => void }) {
  const tl = timeline(durations);
  const hatch = useId();
  const svg = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState<number | null>(null);
  const [box, W] = useWidth(600, 290, 600);
  const narrow = W < 440;
  const L = narrow ? 54 : 60;
  const R = narrow ? 84 : 92;
  const row = 24;
  const H = durations.length * row + 30;
  const x = (t: number) => L + (t / AXIS) * (W - L - R);
  const timeAt = (clientX: number) => {
    const r = svg.current!.getBoundingClientRect();
    const px = (clientX - r.left) * (W / r.width);
    return Math.max(0.1, Math.min(5, round1(((px - L) / (W - L - R)) * AXIS)));
  };
  return (
    <div ref={box}>
    <svg ref={svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Threads reaching a barrier at different times"
      style={{ width: '100%', maxWidth: W, display: 'block', touchAction: onChange ? 'none' : undefined }}
      onPointerMove={(e) => { if (drag != null) onChange?.(drag, timeAt(e.clientX)); }}
      onPointerUp={() => setDrag(null)} onPointerLeave={() => setDrag(null)}>
      <defs>
        <pattern id={hatch} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="6" stroke="var(--ink-3)" strokeWidth="2" />
        </pattern>
      </defs>
      {[0, 1, 2, 3, 4, 5].map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={0} y2={durations.length * row} stroke="var(--rule)" />
          <text x={x(t)} y={durations.length * row + 13} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{t}</text>
        </g>
      ))}
      <text x={x(AXIS)} y={durations.length * row + 26} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>time, µs</text>
      {durations.map((d, i) => {
        const y = i * row + 4;
        return (
          <g key={i} data-thread={i}>
            <text x={L - 8} y={y + 11.5} fontSize="11" textAnchor="end">thread {i}</text>
            <rect x={x(0)} y={y} width={x(d) - x(0)} height={row - 9} fill="var(--ink-2)" />
            {tl.waits[i] > 0 ? (
              <rect x={x(d)} y={y} width={x(tl.release) - x(d)} height={row - 9} fill={`url(#${hatch})`} stroke="var(--ink-3)" strokeWidth="0.6" />
            ) : null}
            <rect x={x(tl.release) + 2} y={y + 3} width={x(AXIS) - x(tl.release) - 2} height={row - 15} fill="var(--ink-2)" opacity="0.28" />
            <text x={W - R + 8} y={y + 11.5} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
              {tl.waits[i] > 0 ? `waits ${f1(tl.waits[i])}` : 'last to arrive'}
            </text>
            {onChange ? (
              <circle cx={x(d)} cy={y + (row - 9) / 2} r="6.5" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.6" style={{ cursor: 'ew-resize' }}
                onPointerDown={(e) => { e.preventDefault(); setDrag(i); }} />
            ) : null}
          </g>
        );
      })}
      <line x1={x(tl.release)} x2={x(tl.release)} y1={0} y2={durations.length * row} stroke="var(--ink)" strokeWidth="2" />
    </svg>
    </div>
  );
}

const TIMELINE_PRESETS: { label: string; d: number[] }[] = [
  { label: 'exercise 4', d: [...EXERCISE_4] },
  { label: 'one slow thread', d: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0] },
  { label: 'balanced', d: [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5] },
];

function BarrierTimeline() {
  const [d, setD] = useState<number[]>([1.4, 2.2, 1.8, 0.9, 2.6, 1.2, 3.4, 2.0]);
  const tl = timeline(d);
  const set = (i: number, v: number) => setD((old) => old.map((o, k) => (k === i ? v : o)));
  return (
    <Bench
      title="Barrier timeline"
      task="Drag the end of a bar, or type a time. Which thread decides how long the others wait?"
      foot="Solid: working towards the barrier. Hatched: waiting at it. The line is the moment the last thread arrives and all are released. Pale: the next phase, which all start together."
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="muted">Set the eight times to:</span>
        {TIMELINE_PRESETS.map((p) => (
          <button key={p.label} onClick={() => setD(p.d)}>{p.label}</button>
        ))}
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 1fr)' }}>
        <div><TimelineView durations={d} onChange={set} /></div>
        <div>
          <div className="mx-label">Time each thread needs, µs</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '0.3rem 0.6rem' }}>
            {d.map((v, i) => (
              <label key={i} className="field" style={{ margin: 0 }}>
                thread {i}
                <input type="number" min={0.1} max={5} step={0.1} value={v} aria-label={`time of thread ${i}`} style={{ width: '4.6rem' }}
                  onChange={(e) => { const n = Number(e.target.value); if (e.target.value !== '' && isFinite(n)) set(i, Math.max(0.1, Math.min(5, round1(n)))); }} />
              </label>
            ))}
          </div>
        </div>
      </div>
      <p className="summary" role="status" style={{ margin: '0.9rem 0 0' }}>
        All released at <b data-k="release">{f1(tl.release)}</b> µs, when {plural(tl.slowest, 'thread')} arrive{tl.slowest.length === 1 ? 's' : ''}.
        Waiting: <b data-k="waiting">{f1(tl.waiting)}</b> of <b data-k="total">{f1(tl.total)}</b> µs of thread time, which is{' '}
        <b data-k="share">{f1(tl.share * 100)}%</b>.
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- bug hunt
type Mark = 'legal' | 'undefined' | 'depends';
const MARKS: { id: Mark; label: string }[] = [
  { id: 'legal', label: 'Legal' },
  { id: 'undefined', label: 'Undefined' },
  { id: 'depends', label: 'Depends on whose rule you read' },
];
interface Case {
  id: KernelId;
  tab: string;
  /** null: shown for comparison, nothing to mark. */
  verdict: Mark | null;
  why: ReactNode;
}
const CASES: Case[] = [
  {
    id: 'incorrect_barrier',
    tab: '1. Fig. 4.4',
    verdict: 'undefined',
    why: (
      <>
        Even threads call the barrier on line 5, odd threads the one on line 9. Two calls are two barriers, and neither
        is reached by the whole block. The book calls the result undefined; NVIDIA's guide says it "may hang or produce
        unintended side effects".
      </>
    ),
  },
  {
    id: 'barrier_uniform',
    tab: '2. if (blockIdx.x > 0)',
    verdict: 'legal',
    why: (
      <>
        <code>blockIdx.x</code> is the same for every thread of a block, so in any one block all threads call the
        barrier or none does. NVIDIA's guide uses this very condition as its correct example. Legal is not the same as
        right: run block 0 in the stepper and see what separates its two phases.
      </>
    ),
  },
  {
    id: 'barrier_per_thread',
    tab: '3. if (threadIdx.x > 0)',
    verdict: 'undefined',
    why: (
      <>
        The condition is false for thread 0 and true for the rest of the block. NVIDIA's guide uses this very condition
        as its wrong example. Thread 0 goes on to the second phase without waiting for anyone.
      </>
    ),
  },
  {
    id: 'barrier_early_return',
    tab: '4. early return',
    verdict: 'depends',
    why: (
      <>
        By the book's rule (executed by all threads of the block, or by none) it is a violation: the thread that
        returned never executes the call. NVIDIA's guide waits for "all non-exited threads", and Apple's specification
        says a thread that has ended does not block a barrier on Apple silicon, so there it is allowed. AMD's pages do
        not say. Code that must run everywhere does not rely on it: see the rewrite in the last tab.
      </>
    ),
  },
  {
    id: 'barrier_loop',
    tab: '5. barrier in a loop',
    verdict: 'undefined',
    why: (
      <>
        A loop condition is a condition, and this one differs inside the block. Apple's specification says it outright
        for loops: in each iteration, if any thread executes the barrier, all of them must. In the second round only
        the odd threads call it, and the even threads are already reading slots the odd ones are about to overwrite.
      </>
    ),
  },
  {
    id: 'barrier_guarded',
    tab: '4, rewritten',
    verdict: null,
    why: (
      <>
        The range test now guards the work and not the barrier. Every thread of the block reaches line 6, so the kernel
        is legal under every wording, at the cost of out-of-range threads staying until the end.
      </>
    ),
  },
];
const FINDING: Record<Finding, { cls: string; text: string }> = {
  'stuck-sites': { cls: 'fail', text: 'stuck: threads wait at two different calls' },
  'stuck-ended': { cls: 'fail', text: 'stuck: waiting for a thread that has ended' },
  'ran-ahead': { cls: 'fail', text: 'finished, but a thread started a phase too early' },
  partial: { cls: '', text: 'finished: ended threads were not waited for' },
  clean: { cls: 'ok', text: 'finished: every thread met at every barrier' },
};
const RULES: { id: Rule; label: string }[] = [
  { id: 'all-arrive', label: 'every thread must arrive (the book)' },
  { id: 'exited-excluded', label: 'ended threads do not block (NVIDIA, Apple)' },
];

/** One sentence for a step of the barrier model. */
function tell(s: Step, rule: Rule): string {
  if (s.kind === 'advance') {
    const me = s.states[s.thread!];
    const ran = s.lines!.length ? `runs line${s.lines!.length > 1 ? 's' : ''} ${list(s.lines!)}` : 'has nothing left to run';
    const end = me.status === 'waiting' ? ` and waits at the call on line ${me.site}` : ' and ends';
    const ahead = s.ahead
      ? ` It started phase ${s.ahead.phase} while ${plural(s.ahead.behind, 'thread')} still had earlier work to do: nothing separated the phases.`
      : '';
    return `Thread ${s.thread} ${ran}${end}.${ahead}`;
  }
  if (s.kind === 'release') {
    if (!s.absent!.length) return `Every thread of the block is at the call on line ${s.site}. The barrier opens and all of them continue.`;
    return `${plural(s.released!, 'Thread')} ${wait(s.released!)} at the call on line ${s.site}. ${plural(s.absent!, 'Thread')} ended without reaching it. Under this rule a thread that has ended does not count, so the barrier opens.`;
  }
  if (s.kind === 'stuck') {
    if (s.reason === 'different-sites')
      return `Nothing can move. ${s.waiting.map((g, i) => `${plural(g.threads, i ? 'thread' : 'Thread')} ${wait(g.threads)} at line ${g.site}`).join('; ')}. Two calls are two barriers, and neither will ever have the whole block.`;
    return `Nothing can move. ${plural(s.waiting[0].threads, 'Thread')} ${wait(s.waiting[0].threads)} at line ${s.waiting[0].site} for ${plural(s.absent!, 'thread')}, which ended without reaching the call${rule === 'all-arrive' ? ' and never will' : ''}.`;
  }
  return 'Every thread has ended.';
}

function BugHunt() {
  const [sel, setSel] = useState(0);
  const [marks, setMarks] = useState<Partial<Record<KernelId, Mark>>>({});
  const [draft, setDraft] = useState<Mark | null>(null);
  const [rule, setRule] = useState<Rule>('all-arrive');
  const [order, setOrder] = useState<Order>('ascending');
  const [block, setBlock] = useState(1);
  const [n, setN] = useState(0);

  const c = CASES[sel];
  const marked = c.verdict == null || marks[c.id] != null;
  const result = useMemo(() => run(KERNELS[c.id].build(4, block), rule, order), [c.id, block, rule, order]);
  const total = result.steps.length;
  const at = Math.min(n, total);
  const step = at > 0 ? result.steps[at - 1] : undefined;
  const states = step?.states;
  useEffect(() => setN(0), [sel, rule, order, block]);
  useEffect(() => setDraft(null), [sel]);

  const hl = !step ? [] : step.kind === 'advance' ? step.lines! : step.kind === 'release' ? [step.site!] : step.waiting.map((g) => g.site);
  const done = at >= total;
  const found = FINDING[finding(result)];
  const score = CASES.filter((k) => k.verdict != null && marks[k.id] === k.verdict).length;
  const answered = CASES.filter((k) => k.verdict != null && marks[k.id] != null).length;

  return (
    <Bench
      title="Is this barrier legal?"
      task="Mark each kernel first. Then step it, under both readings of the rule."
      foot="Each kernel runs in a block of 4 threads. In kernels 4 and '4, rewritten' the last thread of the block is past the end of the data."
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="Kernel" style={{ flexWrap: 'wrap', rowGap: '0.3rem' }}>
          {CASES.map((k, i) => (
            <button key={k.id} aria-pressed={i === sel} onClick={() => setSel(i)}>
              {k.tab}
              {k.verdict != null && marks[k.id] != null ? (marks[k.id] === k.verdict ? ' ✓' : ' ✗') : ''}
            </button>
          ))}
        </span>
        <span className="muted score" aria-live="polite">{answered} of 5 marked, {score} right</span>
      </div>
      {/* The listing gets the full width: its comments say which phase each line belongs to. */}
      <Code tabs={[{ label: c.id + '.cu', code: SNIPPETS[c.id], tiers: TIERS, status: statusOf(c.id) }]} highlight={hl} />
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 3fr) minmax(0, 2fr)' }}>
        <div>
          {c.verdict != null && !marks[c.id] ? (
            <div className="panel mark">
              <p style={{ margin: '0 0 0.5rem' }}><b>Your verdict on this barrier?</b></p>
              <div className="q" style={{ margin: 0, padding: 0, border: 0 }}>
                <div className="opts" role="radiogroup">
                  {MARKS.map((m) => (
                    <label key={m.id}>
                      <input type="radio" name={'mark-' + c.id} checked={draft === m.id} onChange={() => setDraft(m.id)} />
                      <span>{m.label}</span>
                    </label>
                  ))}
                </div>
              </div>
              <button className="primary" disabled={!draft} onClick={() => draft && setMarks({ ...marks, [c.id]: draft })}>Mark it</button>
            </div>
          ) : (
            <div className="panel verdict-box" data-verdict={c.verdict ?? 'none'}>
              {c.verdict != null ? (
                <p style={{ margin: '0 0 0.4rem' }}>
                  <span className={'chip ' + (marks[c.id] === c.verdict ? 'ok' : 'fail')}>
                    {marks[c.id] === c.verdict ? 'Right' : 'Not quite'}
                  </span>{' '}
                  <b>{MARKS.find((m) => m.id === c.verdict)!.label}.</b>
                </p>
              ) : (
                <p style={{ margin: '0 0 0.4rem' }}><b>For comparison.</b></p>
              )}
              <p style={{ margin: 0 }}>{c.why}</p>
            </div>
          )}
        </div>
        <div>
          {marked ? (
            <table className="threads" aria-label="Threads of the block" style={{ width: '100%' }}>
              <thead>
                <tr><th>thread</th><th>state</th></tr>
              </thead>
              <tbody>
                {[0, 1, 2, 3].map((t) => {
                  const st = states?.[t];
                  const txt = !st || (st.status === 'ready' && st.line == null) ? 'not started'
                    : st.status === 'waiting' ? `waiting at line ${st.site}`
                    : st.status === 'ended' ? 'ended'
                    : 'released, ready to run';
                  return (
                    <tr key={t} data-thread={t} data-status={st?.status ?? 'ready'} className={step?.kind === 'advance' && step.thread === t ? 'cur' : ''}>
                      <td>thread {t}</td>
                      <td>{txt}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          ) : null}
        </div>
      </div>
      {marked ? (
        <>
          <div className="row" style={{ margin: '0.9rem 0 0.7rem' }}>
            <span className="seg" role="group" aria-label="Rule">
              {RULES.map((r) => <button key={r.id} aria-pressed={r.id === rule} onClick={() => setRule(r.id)}>{r.label}</button>)}
            </span>
            <span className="seg" role="group" aria-label="Thread order">
              <button aria-pressed={order === 'ascending'} onClick={() => setOrder('ascending')}>thread 0 runs first</button>
              <button aria-pressed={order === 'descending'} onClick={() => setOrder('descending')}>thread 3 runs first</button>
            </span>
            {c.id === 'barrier_uniform' ? (
              <label className="field">
                Block
                <select value={block} onChange={(e) => setBlock(+e.target.value)} aria-label="Block">
                  <option value={1}>blockIdx.x = 1</option>
                  <option value={0}>blockIdx.x = 0</option>
                </select>
              </label>
            ) : null}
          </div>
          <div className="row" style={{ marginBottom: '0.7rem' }}>
            <button onClick={() => setN(0)} disabled={at === 0}>Restart</button>
            <button onClick={() => setN(at - 1)} disabled={at === 0}>Back</button>
            <button className="primary" onClick={() => setN(at + 1)} disabled={done}>Step</button>
            <button onClick={() => setN(total)} disabled={done}>Run to the end</button>
            <span className="muted" aria-live="polite">step {at} of {total}</span>
            {done ? <span className={'chip outcome ' + found.cls} data-finding={finding(result)}>{found.text}</span> : null}
          </div>
          <div className={'narrate' + (step && (step.kind === 'stuck' || step.ahead) ? ' bad' : '')} role="status">
            {step ? tell(step, rule) : 'Nothing has run yet. Each step lets one thread run until it reaches a barrier call or ends; when no thread can run, the waiting threads are released or found stuck.'}
          </div>
        </>
      ) : (
        <p className="muted" style={{ margin: '0.9rem 0 0' }}>Mark it, then the stepper for this kernel opens.</p>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- waves explorer
const MAX_WAVE_BLOCKS = 20000;
const intIn = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, Math.round(Number.isFinite(v) ? v : lo)));

function WavesExplorer() {
  const { reports } = useReports();
  const [blocks, setBlocks] = useState<number>(TAIL_EXAMPLE.blocks);
  const [units, setUnits] = useState<number>(TAIL_EXAMPLE.units);
  const [slots, setSlots] = useState<number>(TAIL_EXAMPLE.slotsPerUnit);
  const [uneven, setUneven] = useState(false);

  const cap = capacity(units, slots);
  const s = useMemo(
    () => schedule({ blocks, units, slotsPerUnit: slots, durations: uneven ? unevenDurations(blocks) : equalDurations(blocks) }),
    [blocks, units, slots, uneven],
  );
  const prof = useMemo(() => profile(s), [s]);
  const u = usage(s);
  const tl = tail(blocks, cap);
  const sizes = waveSizes(blocks, cap);
  const w = waves(blocks, cap);
  const mine = Object.keys(reports).filter(isGpuBackend).map((b) => deviceFromReport(b, reports[b].device)).filter((d) => d.units != null);
  const preset = (b: number, un: number, sl: number) => { setBlocks(b); setUnits(un); setSlots(sl); };

  const [box, W] = useWidth(600, 290, 600);
  const H = 190;
  const L = 44;
  const top = 16;
  const span = Math.max(1, s.makespan);
  const x = (t: number) => L + (t / span) * (W - L - 10);
  const y = (r: number) => top + (1 - r / Math.max(1, cap)) * (H - top - 26);
  const hatch = useId();
  const sizeText = sizes.length <= 6 ? sizes.join(', ') : `${sizes.slice(0, 3).join(', ')}, … , ${sizes[sizes.length - 1]}`;

  return (
    <Bench
      title="Waves and the tail"
      task="Change the block count until the last wave is full. Then make the block times uneven."
      foot="Time is in block-times: an equal block takes 1, an uneven one 1, 2, 3 or 4 (seeded). Hatched: slots with no block to run."
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="muted">Start from:</span>
        <button onClick={() => preset(FIG45.blocks, FIG45.small.units, FIG45.small.slotsPerUnit)}>Fig. 4.5, small device</button>
        <button onClick={() => preset(FIG45.blocks, FIG45.large.units, FIG45.large.slotsPerUnit)}>Fig. 4.5, large device</button>
        <button onClick={() => preset(TAIL_EXAMPLE.blocks, TAIL_EXAMPLE.units, TAIL_EXAMPLE.slotsPerUnit)}>the book's 660 on 264</button>
        {mine.map((d) => (
          <button key={d.name} onClick={() => setUnits(intIn(d.units!, 1, 4096))}>your {d.name}: {fmt(d.units!, 6)} from multiProcessorCount</button>
        ))}
      </div>
      {!mine.length ? (
        <p className="muted own-device" style={{ margin: '0 0 0.7rem' }}>
          No device report on this machine carries an SM count, so there is no preset for your device. A Metal device
          has none to report.
        </p>
      ) : null}
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <label className="field">
          Blocks in the grid
          <input type="number" min={1} max={MAX_WAVE_BLOCKS} step={1} value={blocks} aria-label="Blocks in the grid" onChange={(e) => setBlocks(intIn(+e.target.value, 1, MAX_WAVE_BLOCKS))} />
        </label>
        <label className="field">
          SMs
          <input type="number" min={1} max={4096} step={1} value={units} aria-label="SMs" onChange={(e) => setUnits(intIn(+e.target.value, 1, 4096))} />
        </label>
        <label className="field">
          Blocks per SM, for this kernel
          <input type="number" min={1} max={64} step={1} value={slots} aria-label="Blocks per SM" onChange={(e) => setSlots(intIn(+e.target.value, 1, 64))} />
        </label>
        <button onClick={() => setBlocks(intIn(Math.ceil(blocks / cap) * cap, 1, MAX_WAVE_BLOCKS))} disabled={blocks % cap === 0}>Round the grid up to full waves</button>
        <span className="seg" role="group" aria-label="Block times">
          <button aria-pressed={!uneven} onClick={() => setUneven(false)}>equal block times</button>
          <button aria-pressed={uneven} onClick={() => setUneven(true)}>uneven block times</button>
        </span>
      </div>
      <div className="cols plot">
        <div ref={box}>
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Blocks in progress over time" style={{ width: '100%', maxWidth: W, display: 'block' }}>
            <defs>
              <pattern id={hatch} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <line x1="0" y1="0" x2="0" y2="6" stroke="var(--ink-3)" strokeWidth="1.6" />
              </pattern>
            </defs>
            <rect x={x(0)} y={y(cap)} width={x(span) - x(0)} height={y(0) - y(cap)} fill={`url(#${hatch})`} />
            {prof.map((g) => (
              <rect key={g.from} x={x(g.from)} y={y(g.resident)} width={Math.max(0.4, x(g.to) - x(g.from) - (prof.length <= 40 ? 1.5 : 0))} height={y(0) - y(g.resident)} fill="var(--ink-2)" />
            ))}
            <line x1={x(0)} x2={x(span)} y1={y(cap)} y2={y(cap)} stroke="var(--ink)" strokeDasharray="5 3" />
            <line x1={x(0)} x2={x(span)} y1={y(0)} y2={y(0)} stroke="var(--ink-3)" />
            <text x={L - 6} y={y(cap) + 4} fontSize="10.5" textAnchor="end">{fmt(cap, 6)}</text>
            <text x={L - 6} y={y(0) + 4} fontSize="10.5" textAnchor="end">0</text>
            <text x={x(0)} y={H - 6} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>launch</text>
            <text x={x(span)} y={H - 6} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>last block ends, time {fmt(s.makespan, 6)}</text>
            <text x={x(0) + 4} y={top - 4} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>blocks in progress</text>
          </svg>
        </div>
        <div className="panel stats" role="status">
          <p style={{ margin: '0 0 0.4rem' }}>
            Runs at once: {fmt(units, 6)} × {slots} = <b data-k="capacity">{fmt(cap, 6)}</b> blocks
          </p>
          <p style={{ margin: '0 0 0.4rem' }}>
            Waves, by the book's count: {fmt(blocks, 6)} / {fmt(cap, 6)} = <b data-k="waves">{fmt(w, 4)}</b>
          </p>
          {uneven ? (
            <p style={{ margin: '0 0 0.4rem' }}>
              With uneven times the blocks no longer start together, so there are no clean waves to count.
            </p>
          ) : (
            <>
              <p style={{ margin: '0 0 0.4rem' }}>Blocks per wave: <span data-k="sizes">{sizeText}</span></p>
              <p style={{ margin: '0 0 0.4rem' }}>
                Last wave: <b data-k="idle">{fmt(tl.idle, 6)}</b> of {fmt(cap, 6)} slots idle, <b data-k="idle-share">{fmt(tl.idleShare * 100, 3)}%</b>{' '}
                {tl.idle === 0 ? <span className="chip ok">no tail</span> : null}
              </p>
            </>
          )}
          <p style={{ margin: 0 }}>
            From launch to the last block's end the device used <b data-k="utilisation">{fmt(u.utilisation * 100, 3)}%</b> of its block capacity.
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const TAIL_Q = tail(1000, 264);
export const questionsS43: Record<string, Question> = {
  'ch04/4.3/q-fig44': {
    kind: 'choice',
    id: 'ch04/4.3/q-fig44',
    prompt: <>The kernel of Fig. 4.4, above, runs in a block of 4 threads. Who waits where?</>,
    options: [
      { label: 'All four wait at the call on line 5, and later all four at the call on line 9.', why: 'A thread takes one branch. No thread reaches both calls.' },
      { label: 'Nobody waits for long: each pair is complete at its own barrier, so both pairs are released at once.', why: 'A barrier counts the whole block, not the threads that happened to take its branch.' },
      {
        label: 'Threads 0 and 2 wait at the call on line 5, 1 and 3 at line 9; neither call has all four, so it may hang.',
        correct: true,
      },
      { label: 'The compiler merges the two calls into one barrier after the if, so all four meet there.', why: 'It does not. Two calls in the source are two barriers.' },
    ],
    explain: (
      <p>
        The two calls look symmetrical, but each is its own barrier, and each needs every thread of the block. The even
        threads will never see an odd thread arrive at line 5, and the other way round. The book calls the result
        undefined behaviour and warns of deadlock; NVIDIA's guide speaks of threads reaching "the same" call.
      </p>
    ),
    source: 'ch. 4 §4.3, p. 72 (Fig. 4.4); notes/today/ch04-facts.md A6',
  },
  'ch04/4.3/q-uniform': {
    kind: 'choice',
    id: 'ch04/4.3/q-uniform',
    prompt: (
      <>
        <code>i</code> is the thread's global index, <code>blockIdx.x * blockDim.x + threadIdx.x</code>, and{' '}
        <code>n</code> is not a multiple of the block size. Which of these conditional barriers is legal?
      </>
    ),
    options: [
      { label: <code>if (threadIdx.x &lt; blockDim.x / 2) __syncthreads();</code>, why: 'True for half of each block and false for the other half.' },
      { label: <code>if (i &lt; n) __syncthreads();</code>, why: 'In the last block some threads pass the test and some do not.' },
      { label: <code>if (blockIdx.x % 2 == 0) __syncthreads();</code>, correct: true },
      { label: <code>if (data[i] &gt; 0) __syncthreads();</code>, why: 'A value read per thread differs between threads of a block.' },
    ],
    explain: (
      <p>
        The test to apply: can two threads of the same block get different answers? <code>blockIdx.x % 2</code> cannot
        differ inside a block, so each block either meets at the barrier as a whole or skips it as a whole. Whether the
        odd blocks then compute the right thing without a barrier is a separate question, and the answer is usually no.
      </p>
    ),
    source: 'ch. 4 §4.3, p. 72; NVIDIA\'s wording in notes/today/ch04-facts.md A6',
  },
  'ch04/4.3/q-tail': {
    kind: 'numeric',
    id: 'ch04/4.3/q-tail',
    prompt: (
      <>
        A grid has 1000 blocks that all take the same time. The device runs 264 of them at once. What percentage of its
        block capacity is idle during the last wave?
      </>
    ),
    answer: Math.round(TAIL_Q.idleShare * 100),
    tol: 0.02,
    unit: '%',
    explain: (
      <p>
        1000 / 264 is about 3.79 waves: three full waves take 792 blocks and leave {TAIL_Q.last} for the fourth. That
        wave leaves 264 − {TAIL_Q.last} = {TAIL_Q.idle} slots idle, and {TAIL_Q.idle} / 264 is about{' '}
        {Math.round(TAIL_Q.idleShare * 100)}%. A grid of 1056 blocks, four full waves, would have none.
      </p>
    ),
    source: 'ch. 4 §4.3, pp. 73–74 (the method; these numbers are ours)',
  },
  'ch04/4.3/q-gridbarrier': {
    kind: 'choice',
    id: 'ch04/4.3/q-gridbarrier',
    prompt: <>A grid-wide barrier limits how large the grid may be. A block-wide barrier puts no limit on the grid. Why the difference?</>,
    options: [
      { label: 'A grid-wide barrier is implemented in software, and its cost grows too fast for large grids.', why: 'It is heavier, but the limit is not about speed.' },
      {
        label: 'Every thread in a barrier\'s scope must be resident to reach it: a block always is, a large grid is not.',
        correct: true,
      },
      { label: 'A grid-wide barrier needs every block to be in the same cluster, and a cluster has a size limit.', why: 'Clusters have their own barrier. The grid-wide one does not go through them.' },
      { label: 'There is no difference: __syncthreads() also fails once the grid exceeds what runs at once.', why: 'A block barrier only needs its own block, which is always resident as a whole.' },
    ],
    explain: (
      <p>
        A thread that has not started cannot arrive. With the usual launch, most blocks of a large grid are still on
        the runtime's list while the first ones run, so a barrier across all of them would never open. That is why the
        grid-wide barrier comes with a special launch call and a grid no larger than what the device holds at once, and
        why the block-wide one is free of both: the runtime starts a block only when all its threads have a place.
      </p>
    ),
    source: 'ch. 4 §4.3, pp. 70–73; notes/today/ch04-facts.md A7',
  },
  'ch04/4.3/q-scal': {
    kind: 'choice',
    id: 'ch04/4.3/q-scal',
    prompt: <>Which habit breaks transparent scalability?</>,
    options: [
      { label: 'Asking the device for its SM count and choosing the number of blocks from it.', why: 'That tunes a launch to a device. The kernel is still correct on any other.' },
      { label: 'Calling __syncthreads() several times in one kernel, so that blocks stop at many barriers.', why: 'Barriers stay inside a block, which is the condition for scalability, not a threat to it.' },
      {
        label: 'Code that assumes how many blocks run at once, or in what order, such as reading another block\'s output.',
        correct: true,
      },
      { label: 'Launching more blocks than the device can run at once, so that some wait for a free SM.', why: 'That is the normal case. The runtime\'s list absorbs it.' },
    ],
    explain: (
      <p>
        The same code runs unchanged on any device because a block needs nothing from any other block. Any assumption
        about the others, which have finished, how many are resident, which SM they are on, is an assumption about one
        device's size and one run's schedule. It may hold on the machine it was tested on and fail on a larger or
        smaller one.
      </p>
    ),
    source: 'ch. 4 §4.3, pp. 73–74',
  },
};

/** Blocks in a sampling kernel's grid: one vocabulary, 256 threads per block, `perThread` elements each (ours). */
const SAMPLE_THREADS = 256;
const sampleBlocks = (vocabulary: number, perThread: number) => Math.ceil(vocabulary / (SAMPLE_THREADS * perThread));

export function S43() {
  const q = questionsS43;
  const ex = timeline(EXERCISE_4);
  const exShare = round1(ex.share * 100);
  const cap = capacity(TAIL_EXAMPLE.units, TAIL_EXAMPLE.slotsPerUnit);
  const bookTail = tail(TAIL_EXAMPLE.blocks, cap);
  return (
    <>
      <p>
        The threads of a block run at their own pace. When an algorithm has phases, where every thread must finish one
        step before any thread starts the next, something has to hold the fast ones back. In CUDA that is{' '}
        <code>__syncthreads()</code>. A thread that calls it is held at the call until every thread of its block has
        arrived there, and then all are released together. This is a <b>barrier</b>. The two leading underscores mark
        an intrinsic: something that looks like a function call and that the compiler turns straight into an
        instruction.
      </p>
      <Source section="4.3" pages="70" />
      <p>
        You already know an algorithm with phases. Softmax over a row of logits finds the maximum, then sums the
        exponentials, then divides. When one block handles the row, each phase needs every thread's result from the
        phase before, and a barrier is what stands between them. Chapter 10 builds that kernel.
      </p>
      <Beyond>
        <p>
          The softmax example is ours. The book's own picture is four friends who shop separately and must all be back
          at the car before anyone drives off (p. 71).
        </p>
      </Beyond>

      <h2>What waiting costs</h2>
      <p>
        Threads reach a barrier at different times. Those that arrive early do nothing until the last one is there.
        That idle time is the price of a barrier, and one thread sets it for the whole block.
      </p>
      <Source section="4.3" pages="71–72" figs="Fig. 4.3, redrawn below; exercise 4, p. 91" />
      <Predict
        id="ch04/4.3/p-waiting"
        prompt={
          <>
            The book's exercise 4: eight threads need {EXERCISE_4.map((v) => f1(v)).join(', ')} microseconds to reach a
            barrier. What percentage of the threads' total time, from start to release, is spent waiting?
          </>
        }
        unit="%"
      >
        {(v) => (
          <>
            <p>
              <b>{f1(exShare)}%</b>. All are released at {f1(ex.release)} µs, when the slowest thread arrives. Eight
              threads times {f1(ex.release)} µs is {f1(ex.total)} µs of thread time, of which {f1(ex.waiting)} µs is
              waiting. {Math.abs(v - exShare) <= 0.5 ? 'You had it.' : 'Each hatched bar below is one thread\'s share of that.'}
            </p>
            <TimelineView durations={EXERCISE_4} />
          </>
        )}
      </Predict>
      <p>
        The timeline below takes any eight times: drag the end of a bar or type a time, and it draws when each thread
        arrives and how long it waits.
      </p>
      <Simplification>
        <p>
          The timeline knows only the times you give it. It says nothing about what the barrier itself costs, and real
          threads arrive a warp at a time, not one by one.
        </p>
      </Simplification>
      <BarrierTimeline />
      <p>
        Try "one slow thread": seven threads spend three quarters of their time waiting for the eighth. Then "balanced":
        nobody waits. A barrier is cheap when the phase before it gives every thread about the same amount of work, and
        that is something you decide when you divide the work among threads.
      </p>

      <h2>The rules</h2>
      <p>
        A barrier can only open if every thread it waits for actually gets there. That puts conditions on where you
        may write one:
      </p>
      <ul>
        <li>a <code>__syncthreads()</code> must be executed by all threads of the block, or by none;</li>
        <li>inside an <code>if</code>, that means the condition has to come out the same for every thread of the block;</li>
        <li>two calls at two places in the source are two different barriers, even when they look alike.</li>
      </ul>
      <Source section="4.3" pages="72" figs="Fig. 4.4" />
      <p>
        The book's counter-example is below. The branch is chosen by the thread's own index, and each branch ends in
        a barrier.
      </p>
      <Aside>
        <p>
          <b>Apparent errata.</b> The printed listing has no <code>__global__</code>, and the text refers to its lines
          as 04, 06 and 10 where the listing numbers them 3, 5 and 9. The copy here adds <code>__global__</code> and
          keeps the listing's numbering.
        </p>
      </Aside>
      <Code
        tabs={[{ label: 'CUDA', code: SNIPPETS.incorrect_barrier, tiers: {}, status: SNIPPET_STATUS }]}
        highlight={[5, 9]}
        caption={
          <>
            The book's Fig. 4.4, stored as <code>labs/ch04/snippets/incorrect_barrier.cu</code>. It breaks the third
            rule, by way of the second.
          </>
        }
      />
      <Ask q={q['ch04/4.3/q-fig44']} label="Before you step it" />
      <p>
        The book calls what happens next undefined behaviour, and warns that a misplaced barrier can give wrong results
        or a deadlock: threads waiting for each other for ever. Keeping the rules is your job; nothing checks them for
        you.
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md, A6)">
        <p>
          NVIDIA's guide states the same rule, and two details more. Threads must reach "the same" call. And it waits
          only for "non-exited" threads: one that has returned does not hold the rest. A barrier under a condition that
          is not uniform "may hang or produce unintended side effects".
        </p>
      </Today>

      <h2>Is this barrier legal?</h2>
      <p>
        Five small kernels. Decide for each whether its barrier is legal before you look, then step through a block of
        four threads. The stepper can apply either reading of the rule: the book's, where every thread of the block
        must arrive, or the narrower one in NVIDIA's and Apple's documents, where a thread that has ended is not waited
        for.
      </p>
      <Simplification>
        <p>
          "Stuck" stands for what the vendors call undefined: a real device may hang or may run on and compute
          nonsense. One thread moves at a time, as far as it can, in one of two fixed orders. A run with nothing flagged
          shows that this order went well, not that the kernel is correct.
        </p>
      </Simplification>
      <BugHunt />
      <p>
        Kernel 3 under the second rule is worth a second look with "thread 3 runs first": everything finishes and
        nothing is flagged. That is how a misplaced barrier passes a test. The condition is still not uniform, and
        with the other order thread 0 reads a slot before its neighbour has written it.
      </p>
      <Ask q={q['ch04/4.3/q-uniform']} />

      <h2>What the system promises in return</h2>
      <p>
        The rules ask something of the runtime too. If some threads of a block could start while others had no place
        to run, the early ones might sit at a barrier indefinitely. CUDA rules that out by assigning resources to a
        block as a unit: a block starts only once the runtime has secured what all of its threads need to finish. So
        the threads of a block run close together in time, and a wait at a correctly placed barrier is always bounded.
        This is the other half of section 4.2's whole-block assignment.
      </p>
      <Source section="4.3" pages="72–73" />

      <h2>Wider scopes</h2>
      <p>
        The set of threads a barrier waits for is its <b>scope</b>. <code>__syncthreads()</code> has the block as its
        scope. Two wider ones exist, both through the Cooperative Groups API: a cluster-wide barrier for a thread block
        cluster, and a grid-wide barrier.
      </p>
      <Source section="4.3" pages="70–71" />
      <p>
        The book calls the grid-wide one heavier and says it comes with restrictions that make sure all threads of the
        grid really are running at once. Its worked example is in chapter 18.
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md, A7)">
        <p>
          The restrictions today: launch with <code>cudaLaunchCooperativeKernel</code> instead of{' '}
          <code>&lt;&lt;&lt;&gt;&gt;&gt;</code>, and keep the grid no larger than what is resident at once. The
          multi-device form of the cooperative launch was removed in CUDA 13.
        </p>
      </Today>
      <Ask q={q['ch04/4.3/q-gridbarrier']} />

      <h2>Why blocks may run in any order</h2>
      <p>
        Turn the block-wide scope around. If no barrier reaches across the edge of a block, then no block ever waits
        for another, and nothing in the model orders them. The runtime may run them in any order, a few at a time or
        many, on whatever the device has. The book calls the result <b>transparent scalability</b>: the same code runs
        unchanged on hardware with different amounts of resources, slowly on a small device and faster on a large one.
      </p>
      <Source section="4.3" pages="73–74" figs="Fig. 4.5, in the animation below" />
      <WavesScene />

      <h2>Waves and the tail effect</h2>
      <p>
        A group of blocks that runs at the same time is a <b>wave</b>, and the book counts waves as the blocks in the
        grid divided by the blocks that run at once. Eight blocks make four waves on a device that runs two at a time,
        and two waves on one that runs four. The count need not be whole. The book's example: {TAIL_EXAMPLE.blocks}{' '}
        blocks on a device that runs {cap} at once is {waves(TAIL_EXAMPLE.blocks, cap)} waves, two full ones and a last
        wave of {bookTail.last} that leaves {Math.round(bookTail.idleShare * 100)}% of the device with nothing to do.
        That is the <b>tail effect</b>. The remedy is a block count that is a multiple of what runs at once, which
        depends on the device.
      </p>
      <Source section="4.3" pages="73–74" />
      <p>
        How many waves to aim for depends on the blocks. When they all take about the same time, few waves are fine
        and a full last wave is what matters; chapter 6 uses that. When their loads differ, many waves are better: each
        slot that frees up early takes the next block, and the load evens out across SMs.
      </p>
      <Aside>
        <p>
          "Wave", "tail effect" and "transparent scalability" are the book's terms. NVIDIA's guide does not use them;
          AMD's HIP documentation does say a smaller GPU schedules blocks "in more waves" (notes/today/ch04-facts.md,
          A5, B2).
        </p>
      </Aside>
      <p>
        The explorer runs the scheduler of section 4.2 on grids too large to step through, and draws how many blocks
        are in progress from the launch to the end of the last block.
      </p>
      <Simplification>
        <p>
          Blocks are atoms with given durations, and "blocks per SM" is a number you type. On a real device it comes
          from section 4.7's budgets, and the time a block takes depends on what else is resident.
        </p>
      </Simplification>
      <WavesExplorer />
      <Ask q={q['ch04/4.3/q-tail']} label="Work it out" />
      <p>
        One case from LLM inference can sit at the other end. Sampling works on a single vector the size of the
        vocabulary, so its grid has vocabulary ÷ (threads per block × elements per thread) blocks, rounded up. A
        vocabulary of 32,000 at 256 threads per block and one element per thread is {sampleBlocks(32000, 1)} blocks:{' '}
        {fmt(waves(sampleBlocks(32000, 1), cap), 2)} of a wave on the book's {cap}. Set the explorer to{' '}
        {sampleBlocks(32000, 1)} blocks and look at the idle slots. No choice of block count fills such a device, and the
        tail is not the question. It is not always so small: a vocabulary of 128,000 at one element per thread is{' '}
        {sampleBlocks(128000, 1)} blocks, {fmt(waves(sampleBlocks(128000, 1), cap), 2)} waves, and the tail arithmetic
        applies again, until each thread takes four elements and the grid is back to {sampleBlocks(128000, 4)} blocks.
      </p>
      <Beyond>
        <p>
          Ours. The book does not discuss sampling; the counts are its wave arithmetic applied to vocabularies we chose
          as round numbers.
        </p>
      </Beyond>
      <Ask q={q['ch04/4.3/q-scal']} />

      <h2>On other backends</h2>
      <p>
        The barrier exists on all three backends. What differs is how much of the rule each vendor has written down,
        and whether anything wider than a block exists.
      </p>
      <Beyond>
        <p>
          The table is ours, checked against vendor documentation on 2026-11-20 (notes/today/ch04-facts.md, A6, A7,
          B3, B11, C3; ch05-facts.md C2). Nothing in this table has been run on CUDA or HIP hardware by this course.
        </p>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }} aria-label="Barriers across backends">
        <thead>
          <tr><th style={{ width: '18%' }}>What</th><th>CUDA</th><th>HIP on AMD</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>Block-wide barrier</td>
            <td><code>__syncthreads()</code></td>
            <td>same spelling</td>
            <td>
              <code>threadgroup_barrier(mem_flags::…)</code>. The flag names the memory to order:{' '}
              <code>mem_threadgroup</code> for threadgroup memory, <code>mem_device</code> for device buffers
            </td>
          </tr>
          <tr>
            <td>Rule for a barrier under a condition</td>
            <td>documented: only if the condition is uniform across the block</td>
            <td>not stated in AMD's pages. Follow NVIDIA's rule as the portable one; that choice is taken from CUDA's documentation, not AMD's</td>
            <td>documented, for conditionals and for loops</td>
          </tr>
          <tr>
            <td>A thread that has ended</td>
            <td>does not block the others ("non-exited threads")</td>
            <td>not stated</td>
            <td>does not block the others, documented for Apple silicon</td>
          </tr>
          <tr>
            <td>Cluster-wide barrier</td>
            <td><code>cluster.sync()</code>, compute capability 9.0 and up</td>
            <td>a <code>cluster_group</code> is documented; which devices support it is unverified</td>
            <td>none</td>
          </tr>
          <tr>
            <td>Grid-wide barrier</td>
            <td><code>this_grid().sync()</code>, launched with <code>cudaLaunchCooperativeKernel</code></td>
            <td><code>this_grid().sync()</code>, launched with <code>hipLaunchCooperativeKernel</code></td>
            <td>none documented. A phase that needs the whole grid to finish is a separate dispatch</td>
          </tr>
        </tbody>
      </table>
      <p>
        Lab 4.3 on the <a href={href('ch04', 'labs')}>labs page</a> is a barrier hunt on kernels you compile and
        run.
      </p>
    </>
  );
}
