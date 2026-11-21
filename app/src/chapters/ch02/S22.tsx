import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today, type Tier } from '../../components/notes';
import { Code, useLabFile } from '../../components/Code';
import { href } from '../../lib/router';
import { elements } from '../../sim/depend';
import {
  BACKENDS,
  BACKEND_LABEL,
  BOOK_PICTURE,
  WAIT_CALL,
  gridStateAt,
  overlap,
  program,
  readOutcomes,
  report,
  run,
  stmtText,
  waitCovers,
  type Backend,
  type Options,
  type ReadOutcome,
  type Stmt,
  type Timeline,
} from '../../sim/timeline';

const num = (v: number) => v.toLocaleString('en-US');
const ticks = (n: number) => `${n} tick${n === 1 ? '' : 's'}`;

// ---------------------------------------------------------------- interactive: the two-lane timeline
/** One sentence, built from the model, about the statement that has just returned. */
function say(tl: Timeline, done: number, backend: Backend): { text: string; bad: boolean } {
  if (done === 0)
    return { text: 'Nothing has run yet. Execution starts as one host thread, at statement 1. Press Step.', bad: false };
  const h = tl.host[done - 1];
  const s = tl.program[h.stmt];
  const len = h.to - h.from;
  if (s.kind === 'host') {
    const launched = tl.grids.filter((g) => g.stmt < h.stmt);
    const during = launched
      .map((g) => ({ grid: g.grid, ran: Math.max(0, Math.min(h.to, g.end) - Math.max(h.from, g.start)) }))
      .filter((x) => x.ran > 0);
    const meanwhile = !launched.length
      ? 'Nothing has been launched yet, so the device was idle.'
      : during.length
        ? `In that time the device ran ${during.map((x) => `grid ${x.grid} for ${ticks(x.ran)}`).join(' and ')}. Neither side waited for the other.`
        : 'The device was idle: every grid launched so far had already finished.';
    return { text: `The host thread did its own work for ${ticks(len)}. ${meanwhile}`, bad: false };
  }
  if (s.kind === 'launch') {
    const ahead = tl.grids.find((g) => g.stmt < h.stmt && g.end > h.to);
    const what = backend === 'metal' ? 'The commit' : 'The launch statement';
    return {
      text:
        `${what} returned at once (drawn as ${ticks(len)}). Grid ${s.grid} is queued: ` +
        (ahead
          ? `it will start when grid ${ahead.grid} has finished. The host waited for neither of them.`
          : 'none of its threads has run yet, and the host is already at its next statement.'),
      bad: false,
    };
  }
  if (s.kind === 'wait') {
    const waited = tl.grids.filter((g) => g.stmt < h.stmt && g.end > h.from).map((g) => 'grid ' + g.grid);
    return {
      text: waited.length
        ? `The host thread was blocked in ${WAIT_CALL[backend]} for ${ticks(len)}, until ${waited.join(' and ')} had finished. This is the only kind of statement in which the host waits.`
        : `${WAIT_CALL[backend]} had nothing left to wait for and returned at once.`,
      bad: false,
    };
  }
  const outs = readOutcomes(tl, h.stmt);
  if (outs.every((o) => o.outcome === 'guaranteed'))
    return {
      text: 'The host read the results. A blocking call had returned after every launch, so it read finished values, and it would on every run.',
      bad: false,
    };
  const parts = outs.map((o) =>
    o.outcome === 'not finished'
      ? `Grid ${o.grid} was still ${gridStateAt(tl, o.grid, h.from)}: the host read whatever was in memory at that moment.`
      : o.outcome === 'finished by timing only'
        ? `Grid ${o.grid} had finished, but only because the host work happened to take longer than the grid. Nothing in the program makes that so.`
        : `Grid ${o.grid} was safe to read.`,
  );
  return { text: 'The host read the results with no blocking call before it. ' + parts.join(' '), bad: true };
}

const OUTCOME_TEXT: Record<ReadOutcome, string> = {
  guaranteed: 'finished on every run',
  'finished by timing only': 'finished, by timing only',
  'not finished': 'not finished',
};

const WRAP = { whiteSpace: 'normal' } as const;

/**
 * At phone widths the timeline is wider than its panel and scrolls inside it.
 * Keep x (in viewBox units) in view, so "host is here" never steps off the side.
 */
function useKeepInView(x: number, viewBoxWidth: number) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    const svg = el?.querySelector('svg');
    if (!el || !svg || el.scrollWidth <= el.clientWidth + 1) return;
    const px = (x / viewBoxWidth) * svg.getBoundingClientRect().width;
    el.scrollLeft = Math.max(0, px - el.clientWidth / 2);
  }, [x, viewBoxWidth]);
  return ref;
}

// Timeline geometry (SVG units).
const TW = 920;
const TL = 104; // left edge of time 0
const TR = 16;
const HOST_Y = 46;
const HOST_H = 26;
const DEV_Y = 118;
const DEV_H = 52;
const AXIS_Y = 190;

/** A block of threads, drawn the way the book draws one: a box with a few wavy lines in it. */
function ThreadBlock({ x, y, h, faint }: { x: number; y: number; h: number; faint: boolean }) {
  const waves = Math.max(1, Math.floor((h - 8) / 8));
  const wave = (cx: number) => {
    let d = `M ${cx} ${y + 4}`;
    for (let i = 0; i < waves; i++) d += ` q ${i % 2 ? -3 : 3} 4 0 8`;
    return d;
  };
  return (
    <g opacity={faint ? 0.4 : 0.9}>
      <rect x={x} y={y} width={16} height={h} rx="1.5" fill="none" stroke="var(--ink-2)" strokeWidth="1" />
      {[4, 8, 12].map((dx) => (
        <path key={dx} d={wave(x + dx)} fill="none" stroke="var(--ink-2)" strokeWidth="1" />
      ))}
    </g>
  );
}

function TimelineBench() {
  const [opts, setOpts] = useState<Options>(BOOK_PICTURE);
  const [backend, setBackend] = useState<Backend>('cuda');
  const [step, setStep] = useState(0);
  const prog = useMemo(() => program(opts), [opts]);
  const tl = useMemo(() => run(prog), [prog]);
  const n = prog.length;
  const done = Math.min(step, n);
  const rep = report(tl, done);
  const now = rep.time;
  const said = say(tl, done, backend);
  const reads = readOutcomes(tl, done);
  const finished = done === n;

  // Changing where the host waits changes the program, so start it again. The sliders keep the place.
  const setSwitch = (o: Partial<Options>) => {
    setOpts((cur) => ({ ...cur, ...o }));
    setStep(0);
  };

  const span = Math.max(tl.end, 24);
  const X = (t: number) => TL + (t * (TW - TL - TR)) / span;
  const every = span > 40 ? 10 : 5;
  const text = (s: Stmt, i: number) => stmtText(s, backend, waitCovers(prog, i));
  const scroller = useKeepInView(X(now), TW);

  const hostLane: ReactNode[] = [];
  const arrows: ReactNode[] = [];
  tl.host.slice(0, done).forEach((h) => {
    const x = X(h.from);
    const w = Math.max(2, X(h.to) - x - 1);
    hostLane.push(
      <g key={h.stmt} data-host={h.stmt} data-kind={h.kind} data-state={h.state} data-from={h.from} data-to={h.to}>
        <rect x={x} y={HOST_Y} width={w} height={HOST_H} rx="2"
          fill={h.state === 'blocked' ? 'url(#tl-hatch)' : h.kind === 'host' ? 'var(--ink-2)' : 'var(--ink)'}
          stroke={h.state === 'blocked' ? 'var(--ink-3)' : 'none'} strokeDasharray={h.state === 'blocked' ? '4 3' : undefined} />
        {h.state === 'blocked' && w > 54 ? (
          <text x={x + w / 2} y={HOST_Y + HOST_H / 2 + 4} fontSize="11" textAnchor="middle" fontWeight="600">blocked</text>
        ) : null}
        <text x={x + w / 2} y={HOST_Y - 6} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{h.stmt + 1}</text>
      </g>,
    );
    const g = tl.grids.find((q) => q.stmt === h.stmt);
    if (g)
      arrows.push(
        <line key={'launch-' + g.grid} x1={X(g.enqueued)} y1={HOST_Y + HOST_H + 1} x2={X(g.start)} y2={DEV_Y - 2}
          stroke="var(--ink)" strokeWidth="1.3" markerEnd="url(#tl-arrow)" />,
      );
    if (h.kind === 'wait') {
      const last = tl.grids.filter((q) => q.stmt < h.stmt).pop();
      if (last && last.end === h.to)
        arrows.push(
          <line key={'back-' + h.stmt} x1={X(h.to)} y1={DEV_Y - 2} x2={X(h.to)} y2={HOST_Y + HOST_H + 3}
            stroke="var(--ink)" strokeWidth="1.3" markerEnd="url(#tl-arrow)" />,
        );
    }
  });

  const launched = tl.grids.filter((g) => g.stmt < done);
  const deviceLane = launched.map((g) => {
    const state = gridStateAt(tl, g.grid, now);
    const x = X(g.start);
    const w = X(g.end) - x;
    const ran = Math.max(0, Math.min(now, g.end) - g.start);
    const blocks = Math.max(0, Math.min(4, Math.floor((w - 6) / 22)));
    const pad = blocks ? (w - blocks * 16) / (blocks + 1) : 0;
    return (
      <g key={g.grid} data-grid={g.grid} data-state={state} data-start={g.start} data-end={g.end}>
        {ran > 0 ? <rect x={x} y={DEV_Y} width={X(g.start + ran) - x} height={DEV_H} rx="2" fill="var(--rule)" /> : null}
        <rect x={x} y={DEV_Y} width={w} height={DEV_H} rx="2" fill="none" stroke="var(--ink)"
          strokeWidth="1.4" strokeDasharray={state === 'finished' ? undefined : '5 3'} />
        {Array.from({ length: blocks }, (_, b) => (
          <ThreadBlock key={b} x={x + pad + b * (16 + pad)} y={DEV_Y + 7} h={DEV_H - 14}
            faint={g.start + ((b + 0.5) * (g.end - g.start)) / blocks > now && state !== 'finished'} />
        ))}
        <text x={x + w / 2} y={DEV_Y + DEV_H + 13} fontSize="10.5" textAnchor="middle" fontWeight="600">
          grid {g.grid}
          {w > 70 ? `, ${state}` : ''}
        </text>
      </g>
    );
  });

  return (
    <Bench
      title="One host thread, grids of device threads"
      task="Step through the program. Before each step, say what state each grid will be in when the statement returns."
      foot="Solid bar: the host thread running. Hatched bar: the host thread blocked inside a call. Dashed box: a grid that has been launched and has not finished. The number above a bar is the statement it belongs to."
    >
      <div data-timeline data-done={done} data-n={n}>
        <div className="row" style={{ marginBottom: '0.7rem' }}>
          <span className="row" style={{ gap: '0.5rem' }}>
            <span className="muted">After each launch:</span>
            <span className="seg" role="group" aria-label="What follows each launch">
              <button data-follow="wait" aria-pressed={opts.waitAfterLaunch} onClick={() => setSwitch({ waitAfterLaunch: true })}>
                a blocking call
              </button>
              <button data-follow="work" aria-pressed={!opts.waitAfterLaunch} onClick={() => setSwitch({ waitAfterLaunch: false })}>
                more host work
              </button>
            </span>
          </span>
          <label className="check">
            <input type="checkbox" data-b2b checked={opts.backToBack} onChange={(e) => setSwitch({ backToBack: e.target.checked })} />
            launch B straight after A
          </label>
          <span className="row" style={{ gap: '0.5rem' }}>
            <span className="muted">Written for</span>
            <span className="seg" role="group" aria-label="Backend">
              {BACKENDS.map((b) => (
                <button key={b} data-backend={b} aria-pressed={backend === b} onClick={() => setBackend(b)}>
                  {BACKEND_LABEL[b]}
                </button>
              ))}
            </span>
          </span>
        </div>
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <label className="field" style={{ flexDirection: 'row', alignItems: 'center', gap: '0.5rem' }}>
            <span>
              A grid runs for <b data-out="grid-ticks">{opts.gridTicks}</b>
            </span>
            <input type="range" min="1" max="12" step="1" value={opts.gridTicks} aria-label="Ticks each grid runs for"
              onChange={(e) => setOpts((cur) => ({ ...cur, gridTicks: Number(e.target.value) }))} />
          </label>
          <label className="field" style={{ flexDirection: 'row', alignItems: 'center', gap: '0.5rem' }}>
            <span>
              Host work takes <b data-out="host-ticks">{opts.hostTicks}</b>
            </span>
            <input type="range" min="1" max="12" step="1" value={opts.hostTicks} aria-label="Ticks each piece of host work takes"
              onChange={(e) => setOpts((cur) => ({ ...cur, hostTicks: Number(e.target.value) }))} />
          </label>
          <span className="muted">ticks</span>
        </div>

        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <button onClick={() => setStep(0)} disabled={done === 0}>Restart</button>
          <button onClick={() => setStep(done - 1)} disabled={done === 0}>Back</button>
          <button className="primary" onClick={() => setStep(done + 1)} disabled={finished}>Step</button>
          <button onClick={() => setStep(n)} disabled={finished}>Run to the end</button>
          <span className="muted" aria-live="polite" data-out="step">
            {done} of {n} statements have returned
          </span>
        </div>

        <div className="panel" style={{ overflowX: 'auto', marginBottom: '0.8rem' }} ref={scroller}>
          <svg viewBox={`0 0 ${TW} 208`} role="img"
            aria-label={`Timeline after ${done} of ${n} statements: a host lane above a device lane`}
            style={{ width: '100%', minWidth: 620, maxWidth: TW, display: 'block' }}>
            <defs>
              <pattern id="tl-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                <line x1="0" y1="0" x2="0" y2="6" stroke="var(--ink-3)" strokeWidth="1.6" />
              </pattern>
              <marker id="tl-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">
                <path d="M 0 0 L 8 4 L 0 8 z" fill="var(--ink)" />
              </marker>
            </defs>
            <text x={TL - 12} y={HOST_Y + 12} fontSize="12" fontWeight="600" textAnchor="end">Host</text>
            <text x={TL - 12} y={HOST_Y + 25} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>one thread</text>
            <text x={TL - 12} y={DEV_Y + 24} fontSize="12" fontWeight="600" textAnchor="end">Device</text>
            <text x={TL - 12} y={DEV_Y + 37} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>grids of threads</text>
            <line x1={TL} y1={HOST_Y + HOST_H / 2} x2={TW - TR} y2={HOST_Y + HOST_H / 2} stroke="var(--rule)" strokeDasharray="1 4" />
            <rect x={TL} y={DEV_Y} width={TW - TL - TR} height={DEV_H} fill="none" stroke="var(--rule-soft)" />
            {/* time axis */}
            <line x1={TL} y1={AXIS_Y} x2={TW - TR} y2={AXIS_Y} stroke="var(--rule)" />
            {Array.from({ length: Math.floor(span / every) + 1 }, (_, i) => i * every).map((t) => (
              <g key={t}>
                <line x1={X(t)} y1={AXIS_Y} x2={X(t)} y2={AXIS_Y + 4} stroke="var(--ink-3)" />
                <text x={X(t)} y={AXIS_Y + 15} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{t}</text>
              </g>
            ))}
            <text x={TL - 12} y={AXIS_Y + 15} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>ticks</text>
            {deviceLane}
            {hostLane}
            {arrows}
            {/* now */}
            <g data-now={now}>
              <line x1={X(now)} y1={20} x2={X(now)} y2={AXIS_Y} stroke="var(--ink)" strokeWidth="1" strokeDasharray="2 3" />
              <text x={Math.min(X(now), TW - TR - 40)} y={14} fontSize="10.5" fontWeight="600" textAnchor={X(now) > TW - 110 ? 'end' : 'start'}>
                host is here, t = {now}
              </text>
            </g>
          </svg>
        </div>

        <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 5fr)' }}>
          <div className="panel">
            <table className="threads" aria-label="The program" style={{ whiteSpace: 'normal' }}>
              <thead>
                <tr>
                  <th>#</th>
                  <th>statement</th>
                  <th>host</th>
                  <th>returned</th>
                </tr>
              </thead>
              <tbody>
                {prog.map((s, i) => {
                  const h = tl.host[i];
                  const ran = i < done;
                  return (
                    <tr key={i} className={i === done - 1 ? 'cur' : ''} data-stmt={i} data-kind={s.kind} data-ran={ran ? 'true' : 'false'}>
                      <td>{i + 1}</td>
                      <td style={{ whiteSpace: 'normal' }}>
                        <code data-out="stmt-text">{text(s, i)}</code>
                      </td>
                      <td>{ran ? h.state : i === done ? <span className="muted">next</span> : ''}</td>
                      <td>{ran ? `t = ${h.to}` : ''}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div>
            <div className={'narrate' + (said.bad ? ' bad' : '')} data-out="said" role="status" style={{ marginBottom: '0.8rem' }}>
              {said.text}
            </div>
            <div className="panel">
              <table className="threads" aria-label="State at this moment">
                <tbody>
                  <tr>
                    <td style={WRAP}>Host time</td>
                    <td>
                      <output data-out="time">t = {now}</output>
                    </td>
                  </tr>
                  {rep.grids.map((g) => (
                    <tr key={g.grid}>
                      <td style={WRAP}>Grid {g.grid}</td>
                      <td>
                        <output data-out={'grid-' + g.grid} data-state={g.state}>{g.state}</output>
                      </td>
                    </tr>
                  ))}
                  {rep.grids.map((g) => {
                    const r = reads.find((x) => x.grid === g.grid);
                    return (
                      <tr key={'read-' + g.grid}>
                        <td style={WRAP}>If the host read {g.grid}'s results now</td>
                        <td>
                          <output data-out={'read-' + g.grid} data-outcome={r ? r.outcome : 'not launched'}>
                            {r ? (
                              <span className={'chip ' + (r.outcome === 'guaranteed' ? 'ok' : 'fail')}>{OUTCOME_TEXT[r.outcome]}</span>
                            ) : (
                              <span className="muted">not launched</span>
                            )}
                          </output>
                        </td>
                      </tr>
                    );
                  })}
                  {finished ? (
                    <>
                      <tr>
                        <td style={WRAP}>The device finishes at</td>
                        <td>
                          <output data-out="device-end">t = {tl.grids[tl.grids.length - 1].end}</output>
                        </td>
                      </tr>
                      <tr>
                        <td style={WRAP}>Host work done while a grid ran</td>
                        <td>
                          <output data-out="overlap">{ticks(overlap(tl))}</output>
                        </td>
                      </tr>
                    </>
                  ) : null}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- the program on three backends
const LAB = 'ch02/lab1_vecadd/solution/';
const KERNEL_TIERS: Record<string, Tier> = { A: 'global', B: 'global', C: 'global', i: 'reg' };

/**
 * The 1-based lines of a .cu file that are device code: each `__global__` or
 * `__device__` function, from the comment lines directly above it to its
 * closing brace in column 0. Everything else in the file is host code.
 */
export function deviceLines(src: string): number[] {
  const lines = src.split('\n');
  const out: number[] = [];
  for (let at = 0; at < lines.length; at++) {
    if (!/^\s*__(global|device)__\b/.test(lines[at])) continue;
    let first = at;
    while (first > 0 && /^\s*\/\//.test(lines[first - 1])) first--;
    let last = at;
    while (last < lines.length - 1 && !/^}/.test(lines[last])) last++;
    for (let i = first; i <= last; i++) out.push(i + 1);
    at = last;
  }
  return out;
}

// ---------------------------------------------------------------- questions
// Numbers the page states, from the models.
const THREADS = elements(8, 2048, 4096);
const LOOP_PROGRAM: Stmt[] = [
  { kind: 'launch', grid: 'A', ticks: 3 },
  ...Array.from({ length: 10 }, (_, i): Stmt => ({ kind: 'launch', grid: 'B' + i, ticks: 3 })),
  { kind: 'wait' },
];
const LOOP_GRIDS = run(LOOP_PROGRAM).grids.length;

export const questionsS22: Record<string, Question> = {
  'ch02/2.2/q-returned': {
    kind: 'choice',
    id: 'ch02/2.2/q-returned',
    prompt: (
      <>
        <code>kernelA&lt;&lt;&lt;blocks, 256&gt;&gt;&gt;(x_d, n); auto t1 = now();</code> What is guaranteed to have
        happened by the time <code>t1</code> is taken?
      </>
    ),
    options: [
      {
        label: 'Every thread of the grid has finished.',
        why: 'That would be a launch that waits for its grid. It does not.',
      },
      {
        label: 'The grid has started running, though it may not have finished.',
        why: 'Not even that. The documentation says the host does not wait for the kernel to complete "or even start".',
      },
      {
        label: 'The launch has been handed to the runtime. No thread of the grid need have run.',
        correct: true,
      },
      {
        label: 'Nothing. The statement only declares the launch, and the grid runs at the next blocking call.',
        why: 'The launch is real: the kernel is set up for execution and queued. A blocking call does not start it. It only makes the host wait for it.',
      },
    ],
    explain: (
      <p>
        A launch is asynchronous. A clock read on the next line measures how long the launch statement took, not how
        long the kernel takes, which is the mistake <a href={href('ch02', 'labs')}>lab 2.2</a> is built around. To
        time the kernel, put a call that waits, such as <code>cudaDeviceSynchronize()</code>, before the second clock
        reading.
      </p>
    ),
    source: 'current CUDA documentation (the Today note; notes/today/ch02-facts.md, A5). The book does not say (§2.2, pp. 24–25)',
  },
  'ch02/2.2/q-inorder': {
    kind: 'choice',
    id: 'ch02/2.2/q-inorder',
    prompt: (
      <>
        <code>kA&lt;&lt;&lt;g, b&gt;&gt;&gt;(x_d); kB&lt;&lt;&lt;g, b&gt;&gt;&gt;(x_d); cudaMemcpy(x_h, x_d, bytes,
        cudaMemcpyDeviceToHost);</code>{' '}
        There is no explicit wait anywhere. Does <code>kB</code> see what <code>kA</code> wrote, and does the copy
        bring back the results of both kernels?
      </>
    ),
    options: [
      { label: 'Yes to both: one stream runs them in issue order, and the copy blocks.', correct: true },
      {
        label: 'No: kB may run before kA or alongside it, so it can miss kA’s writes.',
        why: 'Launches that name no stream go into one default stream, and a stream runs its operations in the order they were issued.',
      },
      {
        label: 'kB sees kA’s writes, but the copy may come back before kB has finished.',
        why: 'The copy is in the same in-order stream, behind both kernels, and cudaMemcpy does not return until the copy is complete.',
      },
      {
        label: 'Neither, unless a cudaDeviceSynchronize() follows each launch.',
        why: 'An explicit wait does no harm here, and it is not needed.',
      },
    ],
    explain: (
      <p>
        Three documented facts combine. Launches with no stream are queued into one default stream. A stream executes
        its operations in the order they were enqueued. <code>cudaMemcpy</code> blocks until the transfer is complete.
        So the copy cannot overtake the kernels and cannot return early. The last step is an inference: the facts
        check found no single sentence in the current guide that says a <code>cudaMemcpy</code> waits for earlier
        kernels. It is also why the book's finished program, which launches and then copies back with no wait between,
        returns a complete result. In the stepper this is "launch B straight after A" with a blocking call.
      </p>
    ),
    source: 'current CUDA documentation (notes/today/ch02-facts.md, A5); the program shape is the book’s Fig. 2.13, p. 38',
  },
  'ch02/2.2/q-grid': {
    kind: 'choice',
    id: 'ch02/2.2/q-grid',
    prompt: (
      <>
        A program launches <code>kernelA</code> once and then launches <code>kernelB</code> inside a host loop of 10
        iterations. Over the whole run, how many grids are there, how many host threads, and can two grids of this
        program run different code?
      </>
    ),
    options: [
      {
        label: '2 grids, 1 host thread. Yes, they can.',
        why: 'A grid belongs to a launch, not to a kernel. Ten launches of kernelB are ten grids.',
      },
      {
        label: `${LOOP_GRIDS} grids, 1 host thread. Yes, they can.`,
        correct: true,
      },
      {
        label: `${LOOP_GRIDS} grids, ${LOOP_GRIDS} host threads. Yes, they can.`,
        why: 'A launch creates device threads. The host is still the one thread that execution started with.',
      },
      {
        label: `${LOOP_GRIDS} grids, 1 host thread. No: every grid of a program runs the same kernel.`,
        why: 'The same-kernel rule holds inside one grid. Each launch names its own kernel.',
      },
    ],
    explain: (
      <p>
        Every kernel call launches a new grid: 1 + 10 = {LOOP_GRIDS}. All the threads of one grid run the kernel that
        was called, so the first grid runs <code>kernelA</code> and the other ten run <code>kernelB</code>. The host
        loop itself is ordinary sequential code in the single host thread.
      </p>
    ),
    source: '§2.2, p. 24 (Fig. 2.3); one kernel per grid: §2.5, pp. 32–33',
  },
};

export function S22() {
  const cu = useLabFile(LAB + 'kernel.cu');
  const metal = useLabFile(LAB + 'kernel.metal');
  const mm = useLabFile(LAB + 'launch.mm');
  return (
    <>
      <p>
        CUDA C++ is C++ with a small amount of new syntax and a library of functions, for machines that have both CPUs
        and GPUs. The book writes its examples in a plain C style and brings in C++ features where they help. Its
        claim for the language is that what you learn with it carries over easily to other languages and platforms.
        This course puts that claim to the test on HIP and Metal in every chapter.
      </p>
      <Source section="2.2" pages="24" />
      <p>
        A CUDA program is written for two kinds of processor. The CPU is the <b>host</b>. A GPU is a <b>device</b>,
        and a machine may have more than one. Host code and device code can sit in the same source file. Device code
        is marked with CUDA keywords and contains the <b>kernels</b>: functions that are executed in a data-parallel
        way. Everything that is not marked is host code, so every C++ program you have written is already a CUDA
        program. It is one with no device code in it.
      </p>
      <Source section="2.2" pages="24" />

      <h2>One host thread, grids of device threads</h2>
      <p>
        Execution starts in host code, as a single thread. When the host calls a kernel, a large number of threads is
        launched on the device. All the threads launched by one call are a <b>grid</b>. A grid is made of{' '}
        <b>blocks</b> of threads, every block the same size, and every thread of the grid runs the kernel that was
        called. When all of its threads have finished, the grid terminates. The next kernel call launches a new grid.
      </p>
      <Source section="2.2" pages="24–25" figs="Fig. 2.3; same code and equal blocks: §2.5, pp. 32–33" />
      <p>
        What does the host thread do while a grid runs? The book's figure draws the host thread, an arrow across to a
        grid and an arrow back. Its text says execution continues on the host when the grid has terminated, and adds
        that applications can overlap CPU and GPU execution. This chapter of the book never says whether the launch
        statement itself waits, and the figure can be read either way. The vendors' documentation is explicit, and
        the answer changes how you read every program from here on.
      </p>
      <Today checked="2026-11-20 (ch02-facts A5, B3, C4)">
        <p>
          A launch is asynchronous: the host does not wait for the kernel to complete, or even to start. A blocking
          call does the waiting: <code>cudaDeviceSynchronize()</code> holds the host thread until all work issued
          before it has completed. HIP documents the same model, with <code>hipDeviceSynchronize()</code>. On Metal,{' '}
          <code>commit</code> returns at once and <code>waitUntilCompleted</code> blocks.
        </p>
      </Today>
      <p>
        The stepper runs a program with the shape of the book's figure: host work, a launch of <code>KernelA</code>,
        host work, a launch of <code>KernelB</code>, and at the end the host reads the results. It opens with a
        blocking call after each launch, which gives the strict alternation the book's text describes. Step through
        that first. Then switch to "more host work" and step through again, and then launch B straight after A.
      </p>
      <Simplification>
        <p>
          The device runs one grid at a time, in launch order, as the default stream does. Streams, which let work
          overlap on the device, are left out. On Metal, one command buffer finishing before the next starts is the
          model's assumption: Apple documents only the order of the queue. Durations are ticks you set and nothing is
          measured. The one tick a launch takes and the one tick before a grid starts are drawing conventions.
        </p>
      </Simplification>
      <TimelineBench />
      <p>
        Four things to take from it. A launch statement returns with its grid queued: no device thread has run yet.
        The host waits only where a blocking call says so. Two launches with no wait between them still run one after
        the other on the device, in launch order, while the host runs ahead. And reading results is safe only after a
        blocking call that follows the launch. Without one, a correct result is an accident of timing: drag the grid
        duration above the host work and the same program reads unfinished values.
      </p>
      <p>
        For CUDA and HIP the order on the device is documented and not a convenience of the model. Both put every
        launch that names no stream into one default stream, and a stream runs its operations in the order they were
        issued. Apple documents less. A command queue keeps an ordered list of command buffers, and the picture
        assumes both are committed to one queue. That one command buffer's work finishes before the next begins is
        the model's assumption for Metal: Apple documents the order of the list, not that the buffers run one after
        the other. What Apple does document covers the case that matters for correctness. When a later command uses a
        buffer that an earlier one writes, Metal holds the later one back until the write has finished. That holds
        for buffers created from the device, which is how this chapter's labs create them. With that caveat, the
        backend switch in the stepper changes the spelling of the statements and nothing in the picture.
      </p>
      <Beyond>
        <p>
          From the vendors' documentation, not the book (ch02-facts A5, B3, C9; infra-facts C12). Read on
          2026-11-20.
        </p>
      </Beyond>
      <p>
        The program of this chapter cannot get the read wrong by accident, because the host fetches its results with{' '}
        <code>cudaMemcpy</code>, which is itself a blocking call. Two places where you can get it wrong are worth
        knowing now. A timer stopped on the line after a launch measures the launch and not the kernel. And on Apple
        silicon a Metal buffer in shared storage is memory the host can read at any moment, so there nothing but the
        wait stands between the host and values the kernel has not written yet.
      </p>
      <Beyond>
        <p>
          Also from current documentation (ch02-facts A5; ch05-facts C6). You meet the timer in lab 2.2 and the
          shared buffer in <a href={href('ch02', '2.4')}>section 2.4</a>.
        </p>
      </Beyond>
      <Ask q={questionsS22['ch02/2.2/q-returned']} />
      <Ask q={questionsS22['ch02/2.2/q-inorder']} />

      <h2>What a thread is here</h2>
      <p>
        The word thread comes with baggage, so the book defines it. A thread is a simplified view of how a processor
        executes a sequential program: the code, the point in the code that is being executed, and the values of its
        variables. A device thread is sequential in exactly this sense, so you can read a kernel from top to bottom
        as the story of one thread. The parallelism is in the number of them. With one thread for each output pixel,
        a large image means millions of threads in a single grid. A thread may also compute several outputs, which
        the book takes up later.
      </p>
      <Source section="2.2" pages="24–25" />
      <Predict
        id="ch02/2.2/p-threads"
        prompt={
          <>
            A residual add, <code>y[i] = x[i] + r[i]</code>, over a batch of 8 sequences of 2,048 tokens with hidden
            size 4,096, written with one thread per element. How many device threads does that one launch create?
          </>
        }
        unit="threads"
      >
        {(v) => (
          <p>
            8 × 2,048 × 4,096 = <b>{num(THREADS)}</b>, one thread for each element of the tensor.{' '}
            {v === THREADS
              ? 'That is what you had.'
              : v > 0 && Math.abs(Math.log2(v / THREADS)) < 1
                ? 'You were within a factor of two.'
                : 'If you were far off, check that all three dimensions went into the product: batch, tokens and hidden size.'}{' '}
            The book's sidebar reaches millions of threads with one thread per pixel of a large image. A tensor of
            activations reaches tens of millions in the same way, in one launch.
          </p>
        )}
      </Predict>
      <p>
        That count would be absurd for the threads you know. With <code>std::thread</code> or a thread pool, threads
        are expensive and few, and you hand each one a large piece of work. Do not carry that habit over. A device
        thread is closer to one iteration of a loop that has its own registers than to a worker you hand tasks to.
      </p>
      <Aside>
        <p>
          The comparison with <code>std::thread</code> and thread pools is ours. The book says "traditional CPU
          threads".
        </p>
      </Aside>
      <p>
        The book's reason is cost: device threads take very few clock cycles to create and schedule, because the
        hardware supports it, where a traditional CPU thread typically takes thousands of cycles.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 guide (notes/today/ch02-facts.md, A19)">
        <p>
          Could not be confirmed. The pages read give no cycle counts for creating a thread, so take this as the
          book's claim.
        </p>
      </Today>
      <Ask q={questionsS22['ch02/2.2/q-grid']} />

      <h2>Where the two halves of the program live</h2>
      <p>
        Below is the whole program you will write in <a href={href('ch02', 'labs')}>lab 2.1</a>, on three backends.
        Do not read the details yet. Look only at where the host code and the device code are. In CUDA both are in
        one file: the marked lines are device code and the rest is host code. The lab build is set up to compile
        that same file for HIP through a small header it puts in front of it. The header includes{' '}
        <code>&lt;hip/hip_runtime.h&gt;</code>, which AMD's compiler does not bring in by default, and renames the{' '}
        <code>cuda</code> calls of the host code to their <code>hip</code> counterparts. The kernel needs no change.
        In Metal the two halves are two files in two languages: the kernel is written in the Metal Shading Language,
        and the host side is Objective-C++ that calls the Metal framework.
      </p>
      <Beyond>
        <p>
          The book covers CUDA only. The HIP and Metal facts were read in AMD's and Apple's documentation on
          2026-11-20 (ch02-facts B1, B4, C1). The course's HIP build has not been compiled on AMD hardware:
          unverified.
        </p>
      </Beyond>
      <Code
        tabs={[
          {
            label: 'CUDA and HIP: one file',
            code: cu ?? '// loading labs/' + LAB + 'kernel.cu',
            tiers: KERNEL_TIERS,
            highlight: cu ? deviceLines(cu) : [],
          },
          { label: 'Metal: the kernel', code: metal ?? '// loading labs/' + LAB + 'kernel.metal', tiers: KERNEL_TIERS },
          { label: 'Metal: the host', code: mm ?? '// loading labs/' + LAB + 'launch.mm', tiers: {} },
        ]}
        caption={
          <>
            The reference solution of lab 2.1, from <code>labs/{LAB}</code>. In the first tab the marked lines are the
            device code. The other two tabs are separate files: <code>kernel.metal</code> is all device code and{' '}
            <code>launch.mm</code> is all host code. Whether each file has been run, and where, is on the{' '}
            <a href={href('ch02', 'labs')}>labs page</a>.
          </>
        }
      />
      <p>
        The next section starts on the host half: the function that surrounds a kernel call, and what its three parts
        are for (<a href={href('ch02', '2.3')}>section 2.3</a>).
      </p>
    </>
  );
}
