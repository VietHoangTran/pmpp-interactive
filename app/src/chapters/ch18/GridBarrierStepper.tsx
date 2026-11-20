import { useEffect, useMemo, useState } from 'react';
import { Code } from '../../components/Code';
import { Bench, type Tier } from '../../components/notes';
import { useReports } from '../../lib/reports';
import {
  CORRECT,
  EX3,
  PRESET_GRAPH,
  SCHEDULES,
  SCHEDULE_LABEL,
  TOO_LARGE,
  UNVISITED,
  UNWRITTEN,
  VARIANT,
  kernelBody,
  residency,
  simulate,
  variantOf,
  type Preset,
  type Rule,
  type Schedule,
  type Shape,
  type Step,
} from '../../sim/gridbfs';

// "Three grid barriers" and "Does it fit?" (section 18.6, book §18.7, Fig. 18.17 lines 65-74,
// pp. 445-448, and Exercise 3). Everything shown is computed by src/sim/gridbfs.ts, whose
// kernels are pinned to labs/ch18/snippets/s186_coop_swap.cu and run on the emulator by
// gridbfs.test.ts.

const TIERS: Record<string, Tier> = {
  prevFrontier: 'global', currFrontier: 'global', numCurrFrontier: 'global', numFrontier: 'global', level: 'global',
  csrGraph: 'global', numCurrFrontier_ref: 'global', numFrontier_ref: 'global',
  numPrevFrontier: 'reg', currLevel: 'reg', i: 'reg', vertex: 'reg', edge: 'reg', neighbor: 'reg', currFrontierIdx: 'reg', tmp: 'reg',
};

const show = (v: number) => (v === UNWRITTEN ? '' : String(v));

function Barriers() {
  const [shape, setShape] = useState<Shape>(CORRECT);
  const [preset, setPreset] = useState<Preset>('grid');
  const [blocks, setBlocks] = useState(2);
  const [schedule, setSchedule] = useState<Schedule>('forward');
  const [rule, setRule] = useState<Rule>('gpu');
  const [n, setN] = useState(0);
  const graph = useMemo(() => PRESET_GRAPH[preset](), [preset]);
  const trace = useMemo(() => simulate(shape, schedule, graph, blocks, 2, rule), [shape, schedule, graph, blocks, rule]);
  useEffect(() => setN(0), [trace]);
  const total = trace.steps.length;
  const at = Math.min(n, total);
  const cur: Step | undefined = at > 0 ? trace.steps[at - 1] : undefined;
  const snap = cur ? cur.after : trace.initial;
  const ex3 = shape.kernel === 'ex3';
  const variant = variantOf(shape);
  const events = trace.history.filter((e) => e.step < at).slice(-12);
  const nextBarrier = () => {
    for (let k = at; k < total; k++) if (trace.steps[k].op === 'release' || trace.steps[k].op === 'deadlock') return setN(k + 1);
    setN(total);
  };
  const sync = (k: 'sync1' | 'sync2' | 'sync3', label: string) => (
    <label className="check">
      <input type="checkbox" checked={shape[k]} aria-label={label} onChange={(e) => setShape({ ...shape, [k]: e.target.checked })} /> {label}
    </label>
  );
  const W = graph.kind === 'grid' ? graph.W! : Math.min(graph.numVertices, 11);
  // Frontier buffers: draw the widest level of the search, and any slot written beyond it.
  const widest = Math.max(...trace.reference.reduce<number[]>((c, l) => (l === UNVISITED ? c : ((c[l] = (c[l] ?? 0) + 1), c)), []));
  const lastWritten = (b: number[]) => b.reduce((m, x, i) => (x === UNWRITTEN ? m : i + 1), 0);
  const slots = Math.max(widest, lastWritten(snap.A), lastWritten(snap.B));
  const counterLabel = ex3 ? 'numFrontier[L]' : '*numCurrFrontier';

  return (
    <>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          Kernel
          <span className="seg" role="group" aria-label="Kernel">
            <button aria-pressed={!ex3} onClick={() => setShape(CORRECT)}>Fig. 18.17's swap</button>
            <button aria-pressed={ex3} onClick={() => setShape(EX3)}>Exercise 3: a counter per level</button>
          </span>
        </label>
        {ex3 ? sync('sync1', 'the one grid.sync()') : <>{sync('sync1', 'grid.sync() 1')}{sync('sync2', 'grid.sync() 2')}{sync('sync3', 'grid.sync() 3')}</>}
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          Graph
          <select value={preset} aria-label="Graph" onChange={(e) => setPreset(e.target.value as Preset)}>
            <option value="grid">a 5 × 5 grid from a corner</option>
            <option value="tree">a 4-ary tree of depth 2</option>
          </select>
        </label>
        <label className="field">
          Resident blocks of 2 threads
          <span className="seg" role="group" aria-label="Blocks">
            {[2, 3].map((b) => <button key={b} aria-pressed={blocks === b} onClick={() => setBlocks(b)}>{b}</button>)}
          </span>
        </label>
        <label className="field">
          Order
          <select value={schedule} aria-label="Order" onChange={(e) => setSchedule(e.target.value as Schedule)}>
            {SCHEDULES.map((s) => <option key={s} value={s}>{SCHEDULE_LABEL[s]}</option>)}
          </select>
        </label>
        <label className="field">
          A barrier some threads have left
          <span className="seg" role="group" aria-label="Barrier rule">
            <button aria-pressed={rule === 'gpu'} onClick={() => setRule('gpu')}>never opens (the contract)</button>
            <button aria-pressed={rule === 'emu'} onClick={() => setRule('emu')}>opens (the course's emulator)</button>
          </span>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setN(0)} disabled={at === 0}>Restart</button>
        <button onClick={() => setN(at - 1)} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => setN(at + 1)} disabled={at >= total}>Step</button>
        <button onClick={nextBarrier} disabled={at >= total}>Run to next barrier</button>
        <button onClick={() => setN(total)} disabled={at >= total}>Run to the end</button>
        {trace.firstFailure >= 0 ? <button onClick={() => setN(trace.firstFailure + 1)}>Go to the first failure</button> : null}
        <span className="muted" aria-live="polite">step {at} of {total}</span>
      </div>
      <div className={'narrate' + (cur?.violation ? ' bad' : '')} role="status" style={{ marginBottom: '0.9rem' }} data-narrate>
        {cur ? (
          <>
            {cur.text}
            {cur.violation ? <><br /><b className="t-fail">Failure: {cur.violation.message}.</b></> : null}
            {cur.note ? <><br /><span className="muted">{cur.note}</span></> : null}
          </>
        ) : (
          <>
            Nothing has run. One launch of {blocks} blocks of 2 threads, every block resident; vertex 0 is the root, and
            the frontier in buffer A holds it. Kernel: {variant ? VARIANT[variant].label : 'your own combination'}. A step
            runs one thread from one grid barrier to the next. Press Step.
          </>
        )}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 9fr) minmax(0, 5fr)' }}>
        <div style={{ minWidth: 0 }}>
          <Code tabs={[{ label: 'kernel body, Fig. 18.17 without its private queue', code: kernelBody(shape, true), tiers: TIERS }]} highlight={cur ? cur.lines : []} />
        </div>
        <div style={{ minWidth: 0 }}>
          <table className="threads" aria-label="Threads">
            <thead>
              <tr><th>thread</th><th>level</th><th title="its own numPrevFrontier">numPrev</th><th title="its prevFrontier and currFrontier">prev/curr</th><th>now</th></tr>
            </thead>
            <tbody>
              {snap.threads.map((t) => (
                <tr key={t.rank} className={cur?.rank === t.rank ? 'cur' : ''} data-thread={t.rank}>
                  <td>{t.rank} (b{t.block})</td>
                  <td>{t.currLevel}</td>
                  <td className="t-reg">{t.numPrev}</td>
                  <td>{t.prev}/{t.curr}</td>
                  <td>{t.status === 'waiting' ? (ex3 ? 'at the sync' : `at sync ${t.at}`) : t.status === 'exited' ? 'has left' : 'running'}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div className="mx-label t-global" style={{ marginTop: '0.7rem' }}>level[] (global memory); · is UNVISITED</div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${W}, auto)` }} data-levels={snap.level.map((l) => (l === UNVISITED ? '.' : l)).join(',')}>
            {snap.level.map((l, v) => (
              <span key={v} className={l === UNVISITED ? 'void' : l !== trace.reference[v] ? 'bad' : 'on'}>
                {l === UNVISITED ? '·' : l}<sub>{v}</sub>
              </span>
            ))}
          </div>
          {(['A', 'B'] as const).map((buf) => (
            <div key={buf}>
              <div className="mx-label t-global" style={{ marginTop: '0.5rem' }}>
                buffer {buf}{snap.threads[0].prev === buf ? ' (prevFrontier of thread 0)' : ' (currFrontier of thread 0)'}
              </div>
              <div className="mx global" style={{ gridTemplateColumns: `repeat(${Math.min(8, slots)}, auto)` }}>
                {snap[buf].slice(0, slots).map((x, i) => (
                  <span key={i} className={x === UNWRITTEN ? 'void' : 'on'}>{show(x)}<sub>{i}</sub></span>
                ))}
              </div>
            </div>
          ))}
          <div className="panel" style={{ marginTop: '0.7rem' }} data-counter={snap.counters.join(',')}>
            <b>{counterLabel}</b>{' '}
            {ex3 ? snap.counters.map((c, l) => (c ? `[${l}] ${c}` : null)).filter(Boolean).join(', ') || 'all 0' : snap.counters[0]}
            <div className="muted" style={{ fontSize: '0.8rem', marginTop: '0.3rem' }}>History so far (last 12):</div>
            <div style={{ fontSize: '0.8rem' }} data-history>
              {events.length ? events.map((e, k) => (
                <span key={k} className={'chip ' + (e.kind === 'reset' ? 'fail' : e.kind === 'read' ? 'reg' : 'global')} style={{ margin: '0 0.25rem 0.25rem 0' }}>
                  {e.kind === 'append' ? `t${e.rank} +1 → ${e.after}` : e.kind === 'read' ? `t${e.rank} reads ${e.value}` : `t${e.rank} resets ${e.before} → 0`}
                </span>
              )) : <span className="muted">nothing yet</span>}
            </div>
          </div>
          {at >= total ? (
            <p style={{ margin: '0.9rem 0 0' }} data-outcome={trace.outcome.label}>
              <span className={'chip ' + (trace.outcome.kind === 'correct' ? 'ok' : 'fail')}>{trace.outcome.label}</span>
              {trace.outcome.consequence ? <> {trace.outcome.consequence}.</> : null}
              {trace.outcome.kind === 'correct' ? <> Every vertex has its BFS level.</> : null}
            </p>
          ) : null}
        </div>
      </div>
    </>
  );
}

function Fit() {
  const { reports } = useReports();
  const [sms, setSms] = useState(2);
  const [perSM, setPerSM] = useState(2);
  const [grid, setGrid] = useState(6);
  const r = residency(sms, perSM, grid);
  // A device's SM count, if a CUDA or HIP run reported one; Metal reports none.
  const mine = (['cuda', 'hip'] as const)
    .map((b) => ({ b, v: Number(reports[b]?.device?.multiProcessorCount) }))
    .find((x) => x.v > 0);
  const num = (label: string, value: number, min: number, max: number, set: (v: number) => void, name: string) => (
    <label className="field">
      {label}
      <input type="number" min={min} max={max} value={value} aria-label={label} data-input={name} style={{ width: '5rem' }}
        onChange={(e) => set(Math.max(min, Math.min(max, Math.round(Number(e.target.value)) || min)))} />
    </label>
  );
  const drawSMs = Math.min(sms, 8);
  return (
    <>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        {num('SMs', sms, 1, 512, setSms, 'sms')}
        {num('blocks per SM (the occupancy query)', perSM, 1, 32, setPerSM, 'per-sm')}
        {num('blocks in the grid, G', grid, 1, 16384, setGrid, 'grid')}
        <button onClick={() => setGrid(r.slots)}>Size it: G = SMs × blocks per SM</button>
        {mine ? <button className="quiet" onClick={() => setSms(mine.v)}>Your {mine.b.toUpperCase()} device: {mine.v} SMs</button> : null}
      </div>
      <p className="muted" style={{ marginTop: 0 }}>
        Resident slots R = {sms} × {perSM} = <b data-slots>{r.slots}</b>. {mine ? '' : 'No CUDA or HIP run has reported an SM count on this machine; Metal reports none.'}
      </p>
      <div className="row" style={{ gap: '0.5rem', alignItems: 'flex-start', marginBottom: '0.8rem' }} aria-label="Resident slots">
        {Array.from({ length: drawSMs }, (_, s) => (
          <div key={s} className="mx global" style={{ gridTemplateColumns: `repeat(${Math.min(perSM, 4)}, auto)` }} title={`SM ${s}`}>
            {Array.from({ length: Math.min(perSM, 8) }, (_, k) => {
              const b = s * perSM + k;
              return <span key={k} className={b < r.resident ? 'on' : 'void'}>{b < r.resident ? `b${b}` : ''}</span>;
            })}
          </div>
        ))}
        {sms > drawSMs ? <span className="muted">and {sms - drawSMs} more SMs</span> : null}
      </div>
      {r.pending ? (
        <div style={{ marginBottom: '0.8rem' }}>
          <div className="mx-label">waiting for a slot that no resident block frees</div>
          <div className="mx" style={{ gridTemplateColumns: `repeat(${Math.min(r.pending, 8)}, auto)` }} data-pending={r.pending}>
            {Array.from({ length: Math.min(r.pending, 16) }, (_, k) => <span key={k} className="void">b{r.resident + k}</span>)}
          </div>
          {r.pending > 16 ? <span className="muted">and {r.pending - 16} more</span> : null}
        </div>
      ) : null}
      <table className="data" style={{ margin: 0 }}>
        <thead><tr><th>Launch</th><th>What happens</th></tr></thead>
        <tbody>
          <tr data-ordinary={r.ordinary}>
            <td>ordinary, <code>&lt;&lt;&lt; &gt;&gt;&gt;</code></td>
            <td>
              {r.fits
                ? 'Every block could be resident, but grid.sync() is documented only under the cooperative launch, and nothing checks the size for you. grid.is_valid() reports whether the grid can synchronise.'
                : `Never completes, by the book's argument (p. 445): ${r.resident} resident blocks wait at grid.sync() for ${r.pending} that cannot start until a resident block leaves. What grid.sync() actually does under this launch is not documented; grid.is_valid() reports whether the grid can synchronise.`}
            </td>
          </tr>
          <tr data-cooperative={r.cooperative}>
            <td>cooperative</td>
            <td>
              {r.fits ? (
                <span className="chip ok">launches</span>
              ) : (
                <>
                  <span className="chip fail">refused</span> at launch. CUDA: <code>{TOO_LARGE.cuda}</code>. HIP:{' '}
                  <code>{TOO_LARGE.hip}</code>. Metal: no counterpart, because it has no cooperative launch and no grid
                  barrier.
                </>
              )}
            </td>
          </tr>
        </tbody>
      </table>
    </>
  );
}

export function GridBarrierStepper() {
  const [tab, setTab] = useState<'barriers' | 'fit'>('barriers');
  return (
    <Bench
      title="Three grid barriers"
      task={tab === 'barriers'
        ? 'Untick one grid.sync() at a time and find the order that breaks it. Then try Exercise 3.'
        : 'Make the grid one block too large, and see where the failure is caught.'}
      foot={
        tab === 'barriers' ? (
          <>
            The whole search in one cooperative launch, with each new vertex appended straight to the public counter
            (Fig. 18.12's append) so that a thread runs from one grid barrier to the next without stopping. Threads take
            turns: in each pass every running thread runs to its next grid.sync() or out of the kernel, in the chosen
            order; then the barrier opens. "In order" and "reversed" are the emulator's first two orders.
          </>
        ) : (
          <>
            Counts only. The resident slots come from the formula of pp. 445–446; whether a real kernel gets that many
            depends on its registers and shared memory, which is what the occupancy query works out.
          </>
        )
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="tablist" aria-label="View">
          <button role="tab" aria-selected={tab === 'barriers'} aria-pressed={tab === 'barriers'} onClick={() => setTab('barriers')}>Three grid barriers</button>
          <button role="tab" aria-selected={tab === 'fit'} aria-pressed={tab === 'fit'} onClick={() => setTab('fit')}>Does it fit?</button>
        </span>
      </div>
      {tab === 'barriers' ? <Barriers /> : <Fit />}
    </Bench>
  );
}
