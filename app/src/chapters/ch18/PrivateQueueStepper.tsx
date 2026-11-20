import { useEffect, useMemo, useState } from 'react';
import { Code } from '../../components/Code';
import { Bench, type Tier } from '../../components/notes';
import {
  BLOCK_ORDER_LABEL,
  CORRECT,
  GARBAGE,
  MAX_CAPACITY,
  PRESETS,
  PRESET_LABEL,
  SCHEDULES,
  SCHEDULE_LABEL,
  SITE_TEXT,
  UNWRITTEN,
  blockOrderOf,
  kernelBody,
  listingLine,
  races,
  simulate,
  summary,
  variantOf,
  VARIANT,
  type BlockOrder,
  type Preset,
  type Reset,
  type Schedule,
  type Shape,
  type Step,
} from '../../sim/bfspriv';

// "One block's private frontier" (section 18.6, book §18.6, Fig. 18.15, pp. 442-444). Every
// number comes from src/sim/bfspriv.ts, whose kernels are pinned to
// labs/ch18/snippets/s186_privatised_variants.cu and run on the emulator and its race flavour
// by bfspriv.test.ts.

export const PRIV_TIERS: Record<string, Tier> = {
  currFrontier_s: 'shared', numCurrFrontier_s: 'shared', numCurrFrontier_s_ref: 'shared', currFrontierStartIdx: 'shared',
  currFrontier: 'global', numCurrFrontier: 'global', numCurrFrontier_ref: 'global', prevFrontier: 'global', level: 'global',
  csrGraph: 'global',
  i: 'reg', vertex: 'reg', edge: 'reg', neighbor: 'reg', currFrontierIdx_s: 'reg', currFrontierIdx: 'reg', currLevel: 'reg',
};

/** A shared-memory word: CD for the emulator's fill, else its value. */
const cell = (v: number) => (v === GARBAGE ? 'CD' : String(v));

/** The verdict is a sentence, so it wraps inside its panel (a .chip alone is nowrap). */
const WRAP = { whiteSpace: 'normal', maxWidth: '100%', boxSizing: 'border-box' } as const;

export function PrivateQueueStepper() {
  const [capacity, setCapacity] = useState(2);
  const [threads, setThreads] = useState(4);
  const [blocks, setBlocks] = useState(2);
  const [preset, setPreset] = useState<Preset>('fanout');
  const [shape, setShape] = useState<Shape>(CORRECT);
  const [schedule, setSchedule] = useState<Schedule>('forward');
  const [order, setOrder] = useState<BlockOrder>('in-order');
  const [seed, setSeed] = useState(1);
  const [small, setSmall] = useState(false);
  const [n, setN] = useState(0);

  const blockOrder = useMemo(() => blockOrderOf(order, blocks, seed), [order, blocks, seed]);
  const opts = useMemo(
    () => ({ capacity, blocks, threads, preset, blockOrder, leftover: small ? 1 : undefined }),
    [capacity, blocks, threads, preset, blockOrder, small],
  );
  const trace = useMemo(() => simulate(shape, schedule, opts), [shape, schedule, opts]);
  const raced = useMemo(() => races(shape, opts), [shape, opts]);
  useEffect(() => setN(0), [trace]);

  const total = trace.steps.length;
  const at = Math.min(n, total);
  const cur: Step | undefined = at > 0 ? trace.steps[at - 1] : undefined;
  const snap = cur ? cur.after : trace.initial;
  const prev = at > 1 ? trace.steps[at - 2].after : trace.initial;
  const block = cur ? cur.block : blockOrder[0];
  const listing = kernelBody(shape, true);
  const done = trace.steps.slice(0, at);
  const newSoFar = done.filter((s) => s.op === 'count').length;
  const reserved = trace.reservations.filter((r) => r.step < at);
  const variant = variantOf(shape);
  const expected = trace.graph.expected.length;
  // Public slots worth drawing: the expected frontier, and anything written past it.
  let shownSlots = Math.max(expected, snap.count);
  snap.how.forEach((h, j) => { if (h) shownSlots = Math.max(shownSlots, j + 1); });
  shownSlots = Math.min(shownSlots, snap.pub.length);

  const nextBarrier = () => {
    for (let k = at; k < total; k++) if (trace.steps[k].op === 'release') return setN(k + 1);
    setN(total);
  };
  const flag = (key: 'barrier1' | 'barrier2' | 'barrier3' | 'overflowCheck', label: string) => (
    <label className="check">
      <input type="checkbox" checked={shape[key]} onChange={(e) => setShape({ ...shape, [key]: e.target.checked })} /> {label}
    </label>
  );
  const range = (label: string, value: number, min: number, max: number, set: (v: number) => void, name: string) => (
    <label className="field">
      {label} <b data-value={name}>{value}</b>
      <input type="range" min={min} max={max} value={value} aria-label={label} onChange={(e) => set(Number(e.target.value))} />
    </label>
  );

  return (
    <Bench
      title="One block's private frontier"
      task="Find when overflow starts, then break the kernel one barrier at a time and find an order that shows it."
      foot={
        <>
          One level of a BFS: thread i of the launch holds previous-frontier vertex i. Blocks run one after another, in
          the order chosen; inside a block a thread runs until it reaches a barrier, then the next takes over, as on the
          course's emulator. Shared memory starts as 0xCD bytes (shown as CD) unless you choose the small leftover. The
          race check runs the emulator's three thread orders and reports two accesses to one location when one writes,
          they are not both atomic, and no barrier or program order puts one first.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        {range('PRIVATE_FRONTIER_CAPACITY', capacity, 1, MAX_CAPACITY, setCapacity, 'capacity')}
        {range('threads per block', threads, 2, 8, setThreads, 'threads')}
        {range('blocks', blocks, 1, 4, setBlocks, 'blocks')}
        <label className="field">
          Level
          <select value={preset} aria-label="Level preset" onChange={(e) => setPreset(e.target.value as Preset)}>
            {PRESETS.map((p) => <option key={p} value={p}>{PRESET_LABEL[p]}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          Counter reset on overflow
          <span className="seg" role="group" aria-label="Counter reset">
            {(['plain', 'atomic', 'clamp'] as Reset[]).map((r) => (
              <button key={r} aria-pressed={shape.reset === r} data-reset={r} onClick={() => setShape({ ...shape, reset: r })}>
                {r === 'plain' ? 'plain store' : r === 'atomic' ? 'atomic store' : 'clamp after barrier 2'}
              </button>
            ))}
          </span>
        </label>
        <label className="field">
          Commit
          <span className="seg" role="group" aria-label="Commit">
            <button aria-pressed={shape.commit === 'loop'} onClick={() => setShape({ ...shape, commit: 'loop' })}>strided loop</button>
            <button aria-pressed={shape.commit === 'if'} onClick={() => setShape({ ...shape, commit: 'if' })}>an if</button>
          </span>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        {flag('barrier1', 'barrier 1')}
        {flag('barrier2', 'barrier 2')}
        {flag('barrier3', 'barrier 3')}
        {flag('overflowCheck', 'overflow check')}
        <label className="field">
          Threads in a block run
          <select value={schedule} aria-label="Thread order" onChange={(e) => setSchedule(e.target.value as Schedule)}>
            {SCHEDULES.map((s) => <option key={s} value={s}>{SCHEDULE_LABEL[s]}</option>)}
          </select>
        </label>
        <label className="field">
          Blocks run and reserve
          <select value={order} aria-label="Block order" onChange={(e) => setOrder(e.target.value as BlockOrder)}>
            {(['in-order', 'reversed', 'random'] as BlockOrder[]).map((o) => <option key={o} value={o}>{BLOCK_ORDER_LABEL[o]}</option>)}
          </select>
        </label>
        {order === 'random' ? <button className="quiet" onClick={() => setSeed((s) => s + 1)}>Another order</button> : null}
        <label className="field">
          Shared memory starts as
          <select value={small ? 'small' : 'cd'} aria-label="Shared memory starts as" onChange={(e) => setSmall(e.target.value === 'small')}>
            <option value="cd">0xCD bytes (the emulator)</option>
            <option value="small">a leftover 1 (as a GPU may leave)</option>
          </select>
        </label>
        <button className="quiet" onClick={() => { setShape(CORRECT); setSmall(false); }}>Corrected kernel</button>
      </div>

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setN(0)} disabled={at === 0}>Restart</button>
        <button onClick={() => setN(at - 1)} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => setN(at + 1)} disabled={at >= total}>Step</button>
        <button onClick={nextBarrier} disabled={at >= total}>Run to next barrier</button>
        <button onClick={() => setN(total)} disabled={at >= total}>Run to the end</button>
        {trace.firstFailure >= 0 ? <button onClick={() => setN(trace.firstFailure + 1)}>Go to the first failure</button> : null}
        <span className="muted" aria-live="polite" data-step={at}>step {at} of {total}</span>
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
            Nothing has run. {blocks} block{blocks === 1 ? '' : 's'} of {threads} threads will find {expected} new vertices.
            Blocks run in the order {blockOrder.join(', ')}. Kernel: {variant ? VARIANT[variant].label : 'your own combination of changes'}.
            Press Step.
          </>
        )}
      </div>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 9fr) minmax(0, 5fr)' }}>
        <div style={{ minWidth: 0 }}>
          <Code tabs={[{ label: `kernel body: ${variant ? VARIANT[variant].label : 'your combination of changes'}`, code: listing, tiers: PRIV_TIERS }]} highlight={cur ? cur.lines : []} />
          <table className="threads" aria-label="Threads of the current block">
            <thead>
              <tr><th>block {block}</th><th>vertex</th><th>doing</th></tr>
            </thead>
            <tbody>
              {Array.from({ length: threads }, (_, t) => (
                <tr key={t} className={cur?.block === block && cur?.thread === t ? 'cur' : ''}>
                  <td>thread {t}</td>
                  <td>{block * threads + t}</td>
                  <td>{snap.blocks[block].status[t]}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ minWidth: 0 }}>
          <div className="mx-label t-shared">each block's shared memory: currFrontier_s and its counter</div>
          <div className="row" style={{ gap: '0.7rem', alignItems: 'flex-start', marginBottom: '0.9rem' }}>
            {snap.blocks.map((sb, b) => (
              <div key={b} data-block={b} style={{ opacity: cur && cur.block !== b ? 0.6 : 1 }}>
                <div className="muted" style={{ fontSize: '0.72rem', fontWeight: block === b ? 650 : 400 }}>block {b}</div>
                <div className="mx shared" style={{ gridTemplateColumns: `repeat(${Math.min(capacity, 4)}, auto)` }} aria-label={`currFrontier_s of block ${b}`}>
                  {sb.private_.map((v, k) => {
                    const hot = cur && cur.block === b && cur.op === 'store' && cur.text.includes(`currFrontier_s[${k}]`);
                    return (
                      <span key={k} data-slot={k} className={!sb.written.private_[k] ? 'void' : hot ? 'hot' : 'on'}>
                        {cell(v)}<sub>{k}</sub>
                      </span>
                    );
                  })}
                </div>
                <div style={{ fontSize: '0.72rem', marginTop: '0.25rem', fontFamily: 'var(--sans)' }}>
                  <div data-counter={b}>numCurrFrontier_s <b className={sb.written.counter ? 't-shared' : 'muted'}>{cell(sb.counter)}</b></div>
                  <div data-start={b}>currFrontierStartIdx <b className={sb.written.start ? 't-shared' : 'muted'}>{cell(sb.start)}</b></div>
                </div>
              </div>
            ))}
          </div>

          <div className="mx-label t-global">
            public frontier currFrontier (global memory), counter <span data-public-count>{snap.count}</span>
          </div>
          <div className="mx global" style={{ gridTemplateColumns: `repeat(${Math.min(8, Math.max(1, shownSlots))}, auto)` }} aria-label="public frontier">
            {snap.pub.slice(0, shownSlots).map((v, j) => {
              const h = snap.how[j];
              const fresh = !!cur && (v !== prev.pub[j] || h?.block !== prev.how[j]?.block);
              return (
                <span key={j} data-public={j} data-via={h?.via ?? ''} className={!h ? 'void' : fresh ? 'hot' : j >= snap.count ? 'bad' : 'on'}
                  title={h ? `written by block ${h.block}, ${h.via === 'overflow' ? 'straight from the overflow path' : 'in the commit'}` : 'not written'}>
                  {v === UNWRITTEN ? '' : v}
                  {h ? <sub>{h.via === 'overflow' ? `o${h.block}` : `b${h.block}`}</sub> : null}
                </span>
              );
            })}
          </div>
          <p className="muted" style={{ margin: '0.3rem 0 0.8rem', fontSize: '0.8rem' }}>
            Small b<i>k</i>: committed by block <i>k</i>. Small o<i>k</i>: sent straight out by block <i>k</i>'s overflow path.
          </p>

          <div className="panel" style={{ marginBottom: '0.8rem' }} data-reservations={reserved.map((r) => `${r.block}:${r.start}+${r.count}`).join(' ')}>
            <b>Reservations, in the order blocks reached the public counter</b>
            {reserved.length ? (
              <ol style={{ margin: '0.3rem 0 0', paddingLeft: '1.2rem' }}>
                {reserved.map((r) => (
                  <li key={r.block}>block {r.block}: {r.count === GARBAGE ? 'CD' : r.count} slot{r.count === 1 ? '' : 's'} from {r.start}</li>
                ))}
              </ol>
            ) : <div className="muted">none yet</div>}
            {Object.entries(trace.commitPasses).length && at >= total ? (
              <div style={{ marginTop: '0.4rem' }} data-commit-passes>
                Commit addresses, pass by pass:{' '}
                {Object.entries(trace.commitPasses).map(([b, passes]) => (
                  <span key={b} style={{ marginRight: '0.6rem' }}>block {b}: {passes.map((p) => `[${p.join(', ')}]`).join(' ')}</span>
                ))}
                <div className="muted">In each pass consecutive threads write consecutive addresses, so on a GPU a warp's stores coalesce.</div>
              </div>
            ) : null}
          </div>

          <div className="panel" style={{ marginBottom: '0.8rem' }} data-tally>
            <b>Atomics on the public counter</b>
            <div>
              so far: <b data-global-atomics>{snap.globalAtomics}</b> ({snap.overflows} overflow append{snap.overflows === 1 ? '' : 's'} and{' '}
              {reserved.length} reservation{reserved.length === 1 ? '' : 's'}); Fig. 18.12 would have made {newSoFar}, one per new vertex.
            </div>
            {at >= total ? (
              <div data-final-tally={`${trace.tally.fig1815}/${trace.tally.fig1812}/${trace.tally.overflows}`}>
                whole level: <b>{trace.tally.fig1815}</b> against <b>{trace.tally.fig1812}</b>: one per block ({trace.tally.reservations}) plus one per
                overflow ({trace.tally.overflows}).
              </div>
            ) : null}
          </div>

          <div className="panel" data-races={raced.map((r) => r.sites.join('|')).join(' ')}>
            <b>Race check, all three thread orders</b>
            {raced.length === 0 ? (
              <div><span className="chip ok">no race</span></div>
            ) : (
              <ul style={{ margin: '0.3rem 0 0', paddingLeft: '1.2rem' }}>
                {raced.map((r) => (
                  <li key={r.sites.join('|')} data-race={r.sites.join('|')}>
                    <span className="chip fail">race</span>{' '}
                    {r.sites[0] === r.sites[1]
                      ? <>line {listingLine(shape, r.sites[0])}, {SITE_TEXT[r.sites[0]]}, in two {r.example.a.block !== r.example.b.block ? 'blocks' : 'threads'}</>
                      : <>line {listingLine(shape, r.sites[0])}, {SITE_TEXT[r.sites[0]]}, against line {listingLine(shape, r.sites[1])}, {SITE_TEXT[r.sites[1]]}</>}
                    {' '}(on {r.example.addr})
                  </li>
                ))}
              </ul>
            )}
          </div>

          {at >= total ? (
            <p style={{ margin: '0.9rem 0 0' }} data-outcome={trace.outcome.kind}>
              {trace.outcome.kind === 'correct' ? (
                <span className="chip ok" style={WRAP} data-outcome-chip>
                  Correct: {trace.outcome.size} vertices{trace.outcome.grouped ? ', each block\'s commit in one contiguous range' : ''}.
                </span>
              ) : trace.outcome.kind === 'crash' ? (
                <span className="chip fail" style={WRAP} data-outcome-chip>Stopped: {trace.outcome.crash}.</span>
              ) : (
                <span className="chip fail" style={WRAP} data-outcome-chip>Wrong: {summary(trace.outcome)}.</span>
              )}
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

