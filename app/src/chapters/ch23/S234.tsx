import { useMemo, useState, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  DEFAULTS, DEFECTS, VERSIONS, changedEdges, defectsFor, edges, simulate,
  type Defect, type Durations, type EdgeId, type Issue, type Mechanism, type Priority, type Version,
} from '../../sim/haloloop';
import { LoopScene, Timeline, VIEW_W, r2, timelineHeight } from './LoopScene';

// ---------------------------------------------------------------- code shown
// Copies of labs/ch23/snippets/overlap_loop.cu (the marked regions, dedented) and
// labs/ch23/snippets/nccl_exchange.cu; S234.snippets.test.ts fails if they drift and runs
// the CUDA front-end check on both files.
const SETUP = `int low, high;                                    // least, then greatest priority
cudaDeviceGetStreamPriorityRange(&low, &high);    // a lower number is a higher priority
cudaStream_t topStream, bottomStream, internalStream;
cudaStreamCreateWithPriority(&internalStream, cudaStreamDefault, low);
cudaStreamCreateWithPriority(&topStream, cudaStreamDefault, high);
cudaStreamCreateWithPriority(&bottomStream, cudaStreamDefault, high);

cudaEvent_t resetL2, computeTop, computeBottom;   // only ever waited on: no timing
cudaEventCreateWithFlags(&resetL2, cudaEventDisableTiming);
cudaEventCreateWithFlags(&computeTop, cudaEventDisableTiming);
cudaEventCreateWithFlags(&computeBottom, cudaEventDisableTiming);

float* l2norm_h;                                  // pinned, so line 21 stays asynchronous
cudaMallocHost(&l2norm_h, sizeof(float));
float l2norm = 1.0f;                              // anything above tol`;

const LOOP = `for(unsigned int iter = 0; l2norm > tol && iter < iter_max; ++iter) {

    // zero the norm, then make both boundary streams wait for that
    cudaMemsetAsync(l2norm_d, 0, sizeof(float), internalStream);
    cudaEventRecord(resetL2, internalStream);
    cudaStreamWaitEvent(topStream, resetL2, 0);
    cudaStreamWaitEvent(bottomStream, resetL2, 0);

    // one kernel, three launches: row 1, row ny - 2, rows 2 to ny - 3
    launch_jacobi_kernel(output, input, l2norm_d, 3, nx, topStream);
    cudaEventRecord(computeTop, topStream);
    launch_jacobi_kernel(output + (ny - 3)*nx, input + (ny - 3)*nx, l2norm_d, 3,
                         nx, bottomStream);
    cudaEventRecord(computeBottom, bottomStream);
    launch_jacobi_kernel(output + nx, input + nx, l2norm_d, ny - 2, nx,
                         internalStream);

    // the copy of the norm waits for all three kernels
    cudaStreamWaitEvent(internalStream, computeTop, 0);
    cudaStreamWaitEvent(internalStream, computeBottom, 0);
    cudaMemcpyAsync(l2norm_h, l2norm_d, sizeof(float), cudaMemcpyDeviceToHost,
                    internalStream);

    // halo exchange: the host waits for each boundary row, then sends it
    const int top = rank > 0 ? rank - 1 : (numRanks - 1);
    const int bottom = (rank + 1) % numRanks;
    cudaStreamSynchronize(topStream);
    MPI_Sendrecv(output + nx, nx, MPI_FLOAT, top, 0,
                 output + (ny - 1)*nx, nx, MPI_FLOAT, bottom, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    cudaStreamSynchronize(bottomStream);
    MPI_Sendrecv(output + (ny - 2) * nx, nx, MPI_FLOAT, bottom, 0,
                 output, nx, MPI_FLOAT, top, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // the host needs the norm: wait for the copy, then reduce across ranks
    cudaStreamSynchronize(internalStream);
    MPI_Allreduce(l2norm_h, &l2norm, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    l2norm = std::sqrt(l2norm);

    std::swap(output, input);

}`;

const NCCL_EXCHANGE = `ncclGroupStart();
ncclSend(output + nx,          nx, ncclFloat, top,    nccl_comm, topStream);
ncclRecv(output + (ny - 1)*nx, nx, ncclFloat, bottom, nccl_comm, topStream);
ncclGroupEnd();
cudaEventRecord(exchangeTop, topStream);
ncclGroupStart();
ncclSend(output + (ny - 2)*nx, nx, ncclFloat, bottom, nccl_comm, bottomStream);
ncclRecv(output,               nx, ncclFloat, top,    nccl_comm, bottomStream);
ncclGroupEnd();
cudaEventRecord(exchangeBottom, bottomStream);

// the host's one wait: it needs the norm
cudaStreamSynchronize(internalStream);
MPI_Allreduce(l2norm_h, &l2norm, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
l2norm = std::sqrt(l2norm);

cudaStreamWaitEvent(internalStream, exchangeTop, 0);
cudaStreamWaitEvent(internalStream, exchangeBottom, 0);`;

const CHECKED_LOOP = 'front-end-checked (labs/ch23/snippets/overlap_loop.cu)';
const CHECKED_NCCL = 'front-end-checked (labs/ch23/snippets/nccl_exchange.cu)';

/** "05, 06" or "28–34" to line numbers. */
function linesOf(spec: string): number[] {
  const out: number[] = [];
  for (const part of spec.split(',')) {
    const m = /(\d+)(?:–(\d+))?/.exec(part.trim());
    if (!m) continue;
    const a = Number(m[1]);
    const b = m[2] ? Number(m[2]) : a;
    for (let k = a; k <= b; k++) out.push(k);
  }
  return out;
}

// ---------------------------------------------------------------- the dependency puzzle
const MECH: Record<Mechanism, string> = {
  stream: 'same stream',
  event: 'event + stream wait',
  host: 'host waits for a stream',
  blocking: 'a blocking call returns',
  barrier: 'a barrier on the stream',
  kernel: 'inside the kernel',
};
const PICKS: Mechanism[] = ['stream', 'event', 'host', 'blocking'];

// The two operations of each ordering in Fig. 23.14, for the feedback.
interface Ends {
  a: string;
  b: string;
  same: boolean;
  /** The second operation is a host call, not stream work. */
  hostSide: boolean;
  done: ReactNode;
  /** Where to look when the pick was a host wait but the code uses an event. */
  look?: string;
}
const ENDS: Record<EdgeId, Ends> = {
  'reset-top': {
    a: 'the memset (line 04, internalStream)', b: 'the top kernel (line 10, topStream)', same: false, hostSide: false, look: '05 and 06',
    done: <>Line 05 records <code>resetL2</code> in <code>internalStream</code> right after the memset, and line 06 makes <code>topStream</code> wait for it. No host thread waits.</>,
  },
  'reset-bottom': {
    a: 'the memset (line 04, internalStream)', b: 'the bottom kernel (line 12, bottomStream)', same: false, hostSide: false, look: '05 and 07',
    done: <>The same event, <code>resetL2</code>, waited for by <code>bottomStream</code> at line 07.</>,
  },
  'reset-internal': {
    a: 'the memset (line 04)', b: 'the internal kernel (line 15)', same: true, hostSide: false,
    done: <>Both are in <code>internalStream</code>, the memset first (line 04, then line 15): stream order is enough.</>,
  },
  'top-copy': {
    a: 'the top kernel (line 10, topStream)', b: 'the copy (line 21, internalStream)', same: false, hostSide: false, look: '11 and 19',
    done: <><code>computeTop</code> is recorded after the top kernel (line 11); <code>internalStream</code> waits for it at line 19, before the copy.</>,
  },
  'bottom-copy': {
    a: 'the bottom kernel (line 12, bottomStream)', b: 'the copy (line 21, internalStream)', same: false, hostSide: false, look: '14 and 20',
    done: <><code>computeBottom</code>, recorded at line 14 and waited for at line 20.</>,
  },
  'internal-copy': {
    a: 'the internal kernel (line 15)', b: 'the copy (line 21)', same: true, hostSide: false,
    done: <>Same stream: the internal kernel is ahead of the copy in <code>internalStream</code>.</>,
  },
  'top-exchange': {
    a: 'the top kernel (line 10, topStream)', b: 'MPI_Sendrecv (line 28)', same: false, hostSide: true,
    done: <><code>MPI_Sendrecv</code> is a host call, so only the host can hold it back: line 27 waits for <code>topStream</code>.</>,
  },
  'bottom-exchange': {
    a: 'the bottom kernel (line 12, bottomStream)', b: 'MPI_Sendrecv (line 32)', same: false, hostSide: true,
    done: <>Line 31 does the same for <code>bottomStream</code>, once the top exchange has returned.</>,
  },
  'copy-allreduce': {
    a: 'the copy (line 21, internalStream)', b: 'MPI_Allreduce (line 38)', same: false, hostSide: true,
    done: <><code>MPI_Allreduce</code> reads <code>l2norm_h</code> on the host, so the host waits for <code>internalStream</code> at line 37.</>,
  },
  'exchange-next': {
    a: 'the two MPI_Sendrecv calls (lines 28 and 32)', b: "the next iteration's launches", same: false, hostSide: true,
    done: <><code>MPI_Sendrecv</code> returns only when its exchange is complete (lines 28–34), so by the time the host queues the next iteration the halos are in (p. 573).</>,
  },
};

function hint(id: EdgeId, pick: Mechanism, right: Mechanism): ReactNode {
  const e = ENDS[id];
  if (pick === 'stream')
    return e.hostSide && id !== 'exchange-next'
      ? <>{e.b.replace(/ \(.*/, '')} is not in any stream: it is a host call. Stream order cannot hold it back.</>
      : id === 'exchange-next'
        ? <>Neither side is stream work: the exchanges and the launches are both host calls.</>
        : <>{e.a} and {e.b} are in different streams. Stream order only orders work inside one stream.</>;
  if (pick === 'event')
    return e.same
      ? <>Both are already in <code>internalStream</code>, in this order. An event would add nothing.</>
      : e.hostSide
        ? <>An event can make a stream wait, but {e.b.replace(/ \(.*/, '')} is a host call, not stream work.</>
        : <>Right idea for two streams, so check the other answer: is it really an event? It is not here.</>;
  if (pick === 'host')
    return e.same
      ? <>Nothing on the host waits here, and nothing needs to: both are in <code>internalStream</code>.</>
      : right === 'event'
        ? <>A <code>cudaStreamSynchronize</code> would also work, but it would stop the host. Fig. 23.14 does something cheaper: look at lines {e.look}.</>
        : <>There is no synchronisation call between line 34 and the next iteration&rsquo;s launches. What guarantees it is already in lines 28–34.</>;
  // blocking
  if (id === 'top-exchange' || id === 'bottom-exchange')
    return <><code>MPI_Sendrecv</code> does block, but only after it has started sending whatever is in the row. Something must stop the host before it calls it.</>;
  if (id === 'copy-allreduce')
    return <><code>MPI_Allreduce</code> blocks until the reduction is done, but it would read <code>l2norm_h</code> before the copy lands. The host has to wait first.</>;
  return <>Which call would block? Kernel launches, <code>cudaMemsetAsync</code>, <code>cudaMemcpyAsync</code> and <code>cudaEventRecord</code> all return at once.</>;
}

interface Row {
  pick: Mechanism | '';
  checked: Mechanism | null;
}
const blank = (): Record<EdgeId, Row> =>
  Object.fromEntries(edges('streams').map((e) => [e.id, { pick: '', checked: null }])) as Record<EdgeId, Row>;

type Compare = 'none' | 'nccl' | 'nvshmem';

function DependencyPuzzle() {
  const book = edges('streams');
  const [rows, setRows] = useState<Record<EdgeId, Row>>(blank);
  const [focus, setFocus] = useState<EdgeId | null>(null);
  const [compare, setCompare] = useState<Compare>('none');
  const solved = book.filter((e) => rows[e.id].checked === e.mechanism);
  const all = solved.length === book.length;
  const other = compare === 'none' ? null : edges(compare);
  const changed = compare === 'none' ? [] : changedEdges('streams', compare);
  const hl = focus ? linesOf(book.find((e) => e.id === focus)!.lines) : [];
  return (
    <Bench
      title="What enforces each ordering?"
      task="Ten orderings one iteration needs. For each, pick what makes it hold in Fig. 23.14."
      foot="The orderings are the ones the book names on pp. 559–563 and 573. The line numbers are the book's."
    >
      <div data-puzzle>
        <Code
          tabs={[
            { label: 'Fig. 23.14, corrected', code: LOOP, status: CHECKED_LOOP, highlight: hl },
            { label: 'setup (pp. 558–562)', code: SETUP, status: CHECKED_LOOP },
          ]}
          caption={
            <>
              The book&rsquo;s loop with its own line numbers. Line 19 is printed with an extra closing parenthesis on the
              page; this copy has the one Fig. 23.20 prints. The comments are ours. The setup tab gathers what the text
              shows in pieces, with all three events created without timing.
            </>
          }
        />
        <div className="panel">
          <div className="row" style={{ justifyContent: 'space-between', marginBottom: '0.2rem' }}>
            <b>Ten orderings</b>
            <span className="muted" data-puzzle-score>
              {solved.length} of {book.length} placed
            </span>
          </div>
          {book.map((e, n) => {
            const s = rows[e.id];
            const done = s.checked === e.mechanism;
            const o = other?.find((x) => x.id === e.id);
            const isChanged = changed.includes(e.id);
            return (
              <div key={e.id} data-edge={e.id} data-solved={done ? 'true' : 'false'}
                style={{ padding: '0.6rem 0', borderTop: n ? '1px solid var(--rule-soft)' : undefined }}>
                <div className="row" style={{ alignItems: 'flex-end' }}>
                  <p style={{ margin: 0, flex: '1 1 16rem', minWidth: '12rem' }}>
                    {e.first} <span className="muted">before</span> {e.then}
                  </p>
                  <label className="field">
                    Enforced by
                    <select aria-label={`Mechanism for ${e.id}`} value={s.pick} disabled={done}
                      onChange={(ev) => setRows((r) => ({ ...r, [e.id]: { pick: ev.target.value as Mechanism | '', checked: null } }))}>
                      <option value="">choose</option>
                      {PICKS.map((m) => (
                        <option key={m} value={m}>{MECH[m]}</option>
                      ))}
                    </select>
                  </label>
                  <div style={{ minWidth: '4.6rem' }}>
                    {done ? (
                      <span className="chip ok">placed</span>
                    ) : (
                      <button className="primary" aria-label={`Check ${e.id}`} disabled={s.pick === ''}
                        onClick={() => {
                          setRows((r) => ({ ...r, [e.id]: { ...r[e.id], checked: s.pick as Mechanism } }));
                          setFocus(e.id);
                        }}>
                        Check
                      </button>
                    )}
                  </div>
                </div>
                {s.checked ? (
                  <p data-feedback={e.id} role="status" style={{ margin: '0.4rem 0 0' }}>
                    {done ? (
                      <><span className="chip ok">{linesOf(e.lines).length > 1 ? 'lines' : 'line'} {e.lines}</span> {ENDS[e.id].done}</>
                    ) : (
                      <><span className="chip fail">not that</span> {hint(e.id, s.checked, e.mechanism)}</>
                    )}
                  </p>
                ) : null}
                {o ? (
                  <p data-compare-row={e.id} data-mechanism={o.mechanism} data-changed={isChanged ? 'true' : 'false'} style={{ margin: '0.4rem 0 0' }}>
                    <span className="muted">{VERSIONS[compare as Version].fig}:</span> {MECH[o.mechanism]}{' '}
                    <span className="muted">({/^\d/.test(o.lines) ? 'lines ' : ''}{o.lines})</span>{' '}
                    {isChanged ? <span className="chip ok">changed</span> : null}
                  </p>
                ) : null}
              </div>
            );
          })}
        </div>
        <div className="row" style={{ marginTop: '0.8rem', justifyContent: 'space-between' }}>
          <div className="row" style={{ gap: '0.5rem' }}>
            <span className="ui muted">Compare with</span>
            <span className="seg" role="group" aria-label="Compare with">
              {(['none', 'nccl', 'nvshmem'] as Compare[]).map((c) => (
                <button key={c} aria-pressed={compare === c} disabled={!all && c !== 'none'} onClick={() => setCompare(c)}>
                  {c === 'none' ? 'Fig. 23.14 only' : `${VERSIONS[c].label} (${VERSIONS[c].fig})`}
                </button>
              ))}
            </span>
          </div>
          <button className="quiet" onClick={() => { setRows(blank()); setFocus(null); setCompare('none'); }}>
            Start again
          </button>
        </div>
        {!all ? (
          <p className="ui muted" style={{ margin: '0.6rem 0 0' }}>Place all ten to compare them with the NCCL and NVSHMEM loops.</p>
        ) : compare === 'none' ? (
          <p className="panel" style={{ margin: '0.8rem 0 0' }} data-puzzle-done>
            <b>All ten placed.</b> Count the mechanisms: four kinds, and the last one is the host waiting in disguise. Now
            switch the comparison on and watch which answers move.
          </p>
        ) : (
          <p className="panel" style={{ margin: '0.8rem 0 0' }} data-compare-summary>
            {compare === 'nccl' ? (
              <>
                <b>{changed.length} answers changed.</b> The exchanges moved into the boundary streams, so their kernels
                now precede them by stream order, and the edge to the next iteration became an event wait (lines 43–44).
                Every host wait left except the one before <code>MPI_Allreduce</code>.
              </>
            ) : (
              <>
                <b>{changed.length} answers changed.</b> One kernel in one stream makes most orderings stream order. The puts
                leave from inside the kernel, after the values are computed, and a barrier on the stream (line 18)
                replaces the blocking exchange.
              </>
            )}
          </p>
        )}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- who waits
const ALL_VERSIONS: Version[] = ['mpi', 'streams', 'nccl', 'nvshmem'];
const SLIDERS: { key: keyof Durations; label: string; min: number; max: number; step: number }[] = [
  { key: 'internal', label: 'internal kernel', min: 2, max: 12, step: 0.5 },
  { key: 'boundary', label: 'boundary kernels', min: 0.5, max: 4, step: 0.25 },
  { key: 'exTop', label: 'top exchange', min: 0.5, max: 12, step: 0.5 },
  { key: 'exBottom', label: 'bottom exchange', min: 0.5, max: 12, step: 0.5 },
];

// Where to look when a removed ordering did not fire: the model's own findings.
const QUIET: Record<Exclude<Defect, 'none'>, Partial<Record<Version, ReactNode>>> = {
  'reset-waits': {
    streams: <>In this loop the reset always wins the race in the model: line 37 has just emptied <code>internalStream</code>, and the memset is the first thing in it. The bug is still there. Try the NCCL loop with one exchange much longer than the other.</>,
    nccl: <>Make one exchange much longer than the other, and longer than the internal kernel.</>,
  },
  'compute-waits': {
    streams: <>Make a boundary kernel outlast the internal one, or issue the internal kernel first with equal priority.</>,
    nccl: <>Make a boundary kernel outlast the internal one, or issue the internal kernel first with equal priority.</>,
  },
  'sync-top': {},
  'exchange-waits': { nccl: <>Make an exchange longer than the internal kernel.</> },
  barrier: { nvshmem: <>Make an exchange longer than the kernel.</> },
  pageable: {},
};

function Seg<T extends string>({ label, value, options, onChange, disabled }: {
  label: string; value: T; options: [T, string][]; onChange: (v: T) => void; disabled?: boolean;
}) {
  return (
    <div className="field" style={{ display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' }}>
      <span>{label}</span>
      <span className="seg" role="group" aria-label={label}>
        {options.map(([v, text]) => (
          <button key={v} aria-pressed={value === v} disabled={disabled} onClick={() => onChange(v)}>{text}</button>
        ))}
      </span>
    </div>
  );
}

function WhoWaits() {
  const [version, setVersion] = useState<Version>('streams');
  const [d, setD] = useState<Durations>(DEFAULTS);
  const [priority, setPriority] = useState<Priority>('higher');
  const [issue, setIssue] = useState<Issue>('boundary-first');
  const [defect, setDefect] = useState<Defect>('none');
  const one = version === 'mpi' || version === 'nvshmem';
  const r = useMemo(() => simulate({ ...d, version, priority, issue, defect }), [d, version, priority, issue, defect]);
  const others = useMemo(() => ALL_VERSIONS.map((v) => simulate({ ...d, version: v, priority, issue })), [d, priority, issue]);
  const t1 = Math.max(4, Math.ceil(r.end + 0.2));
  const svgH = 18 + timelineHeight(r.lanes.length, true) + 4;
  const pick = (v: Version) => {
    setVersion(v);
    if (!defectsFor(v).includes(defect)) setDefect('none');
  };
  const quiet = defect !== 'none' && r.violations.length === 0 ? QUIET[defect][version] : null;
  return (
    <Bench
      title="Who waits"
      task="Pick a version and drag the durations. Then remove one ordering and find when it breaks."
      foot="Times are unitless and come from this page's model (src/sim/haloloop.ts), not from any device. Two iterations are drawn."
    >
      <div data-who-waits data-version={version}>
        <div className="row" style={{ marginBottom: '0.7rem' }}>
          <Seg label="Version" value={version} onChange={pick}
            options={ALL_VERSIONS.map((v) => [v, `${VERSIONS[v].label} (${VERSIONS[v].fig})`] as [Version, string])} />
        </div>
        <div className="row" style={{ marginBottom: '0.7rem' }}>
          {SLIDERS.map((s) => (
            <label key={s.key} className="field" style={{ minWidth: '9.5rem' }}>
              <span>
                {s.label}: <b data-dur={s.key}>{d[s.key]}</b>
              </span>
              <input type="range" min={s.min} max={s.max} step={s.step} value={d[s.key]} aria-label={s.label}
                onChange={(e) => setD((x) => ({ ...x, [s.key]: Number(e.target.value) }))} />
            </label>
          ))}
        </div>
        <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
          <Seg label="Boundary streams' priority" value={priority} onChange={setPriority} disabled={one}
            options={[['equal', 'equal'], ['higher', 'higher (the book)']]} />
          <Seg label="Launch order" value={issue} onChange={setIssue} disabled={one}
            options={[['boundary-first', 'boundary kernels first (the book)'], ['internal-first', 'internal kernel first']]} />
          <label className="field" style={{ maxWidth: '100%' }}>
            Remove one ordering
            <select aria-label="Remove one ordering" value={defect} onChange={(e) => setDefect(e.target.value as Defect)} style={{ maxWidth: '100%' }}>
              {defectsFor(version).map((x) => (
                <option key={x} value={x}>
                  {x === 'none' ? (version === 'mpi' ? 'none (nothing to remove here)' : "none: the book's code") : `${DEFECTS[x].label} (${DEFECTS[x].where})`}
                </option>
              ))}
            </select>
          </label>
          <button className="quiet" onClick={() => { setD(DEFAULTS); setPriority('higher'); setIssue('boundary-first'); setDefect('none'); }}>
            Back to the defaults
          </button>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <svg viewBox={`0 0 ${VIEW_W} ${svgH}`} role="img" style={{ width: '100%', minWidth: 700, display: 'block' }}
            aria-label={`Timeline of two iterations of ${VERSIONS[version].label}`} data-axis={t1}>
            <Timeline r={r} y={18} t1={t1} slots gap violations />
          </svg>
        </div>
        <div className="legend ui" style={{ margin: '0.5rem 0 0.8rem', fontSize: '0.8rem' }}>
          <span><i style={{ background: 'var(--ink-2)' }} />kernel</span>
          <span><i style={{ border: '1px dashed var(--ink-3)' }} />queued, waiting for block slots</span>
          <span><i style={{ border: '1px dotted var(--ink-3)' }} />stream held by a wait or a barrier</span>
          <span><i style={{ background: 'repeating-linear-gradient(45deg, var(--ink-3) 0 1.5px, var(--page) 1.5px 4px)', border: '1px solid var(--ink-2)' }} />host blocked in a call</span>
          <span><i style={{ background: 'var(--global-wash)', border: '1px solid var(--global)' }} />data on the move: copy or exchange</span>
          <span><i style={{ background: 'var(--fail)' }} />an ordering broken</span>
        </div>
        <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(13rem, 1fr))' }}>
          <div className="panel" data-readouts>
            <table className="data" style={{ margin: 0 }}>
              <tbody>
                <tr>
                  <td>time per iteration</td>
                  <td className="num" data-readout="iter" data-value={r2(r.iterTime)}><b>{r2(r.iterTime)}</b></td>
                </tr>
                <tr>
                  <td>host blocked per iteration</td>
                  <td className="num" data-readout="blocked" data-value={r2(r.hostBlocked)}>{r2(r.hostBlocked)}</td>
                </tr>
                <tr>
                  <td>bottom rows ready, not yet sent</td>
                  <td className="num" data-readout="gap" data-value={r2(r.bottomGap)}>{r2(r.bottomGap)}</td>
                </tr>
                <tr>
                  <td>exchanges hidden</td>
                  <td className="num" data-readout="hidden" data-value={r.hidden ? 'yes' : 'no'}>
                    {r.hidden ? <span className="chip ok">yes</span> : <span className="chip fail">no</span>}
                  </td>
                </tr>
              </tbody>
            </table>
            <p className="ui muted" style={{ margin: '0.6rem 0 0', fontSize: '0.8rem' }} data-compare-times>
              Time per iteration at these durations:{' '}
              {others.map((o, k) => (
                <span key={o.input.version} data-version-time={o.input.version} data-value={r2(o.iterTime)}
                  style={o.input.version === version ? { color: 'var(--ink)', fontWeight: 600 } : undefined}>
                  {k ? ' · ' : ''}{VERSIONS[o.input.version].label} {r2(o.iterTime)}
                </span>
              ))}
              . Iteration 2 is the one measured.
            </p>
          </div>
          <div className="panel" data-problems={r.violations.length}
            style={r.violations.length ? { borderColor: 'var(--fail)', background: 'var(--fail-wash)' } : undefined} aria-live="polite">
            {r.violations.length ? (
              <>
                <p style={{ margin: '0 0 0.4rem' }}>
                  <b className="t-fail">
                    {r.violations.some((v) => v.kind !== 'stall') ? `${r.violations.filter((v) => v.kind !== 'stall').length} broken ordering${r.violations.filter((v) => v.kind !== 'stall').length > 1 ? 's' : ''}` : 'No wrong result, but the host stalls'}
                  </b>
                </p>
                <ul style={{ margin: 0, paddingLeft: '1.1rem' }}>
                  {r.violations.map((v, k) => (
                    <li key={k} data-violation={v.kind}>{v.text}</li>
                  ))}
                </ul>
              </>
            ) : (
              <>
                <p style={{ margin: 0 }} data-no-problems>
                  <b>No ordering is broken at these durations.</b>
                </p>
                {quiet ? <p style={{ margin: '0.4rem 0 0' }} className="muted" data-quiet>{defect !== 'none' ? <>The defect is there; it did not fire. </> : null}{quiet}</p> : null}
              </>
            )}
          </div>
        </div>
      </div>
    </Bench>
  );
}

// The longest exchange NCCL still hides at the defaults: from the model, for the prediction.
const NCCL_DEF = simulate({ ...DEFAULTS, exTop: 0.5, exBottom: 0.5, version: 'nccl' }).iters[0];
const WINDOW = NCCL_DEF.internal.end - Math.max(NCCL_DEF.top.end, NCCL_DEF.bottom.end);
const STREAMS_DEF = simulate({ ...DEFAULTS, version: 'streams' });

// ---------------------------------------------------------------- questions
export const questionsS234: Record<string, Question> = {
  'ch23/23.4/q-threerows': {
    kind: 'numeric',
    id: 'ch23/23.4/q-threerows',
    prompt: (
      <>
        <code>ny = 10</code>. The bottom launch of Fig. 23.14 receives <code>output + (ny − 3)*nx</code> and{' '}
        <code>input + (ny − 3)*nx</code> and a row count of 3. Counting the top halo as row 0, which row of the local
        array does it write?
      </>
    ),
    answer: 8,
    explain: (
      <p>
        Fig. 23.1&rsquo;s kernel numbers its rows from 1 and stops before row <code>ny − 1</code> of what it was given.
        Given 3 rows, that admits only row 1 of the three: the middle one. The launch&rsquo;s row 0 is the array&rsquo;s
        row 7 (<code>ny − 3</code>), so it writes row 8, which is <code>ny − 2</code>, the bottom boundary row. The same
        reasoning gives row 1 for the top launch and rows 2 to 7 for the internal one: every interior row once.
      </p>
    ),
    source: 'Fig. 23.14 lines 12–13 (p. 557) with the guard of Fig. 23.1 (p. 542); the derivation is ours (notes/ch23.md)',
  },
  'ch23/23.4/q-prio': {
    kind: 'choice',
    id: 'ch23/23.4/q-prio',
    prompt: (
      <>
        The internal kernel was launched first and its blocks fill the GPU. The two boundary kernels are then launched
        into streams of higher priority. When do they start?
      </>
    ),
    options: [
      {
        label: 'As blocks of the internal kernel finish: a priority preempts nothing, it gives pending kernels first claim on slots that free up.',
        correct: true,
      },
      {
        label: 'At once: the scheduler pauses some of the internal kernel’s running blocks to make room for the higher-priority kernels.',
        why: 'NVIDIA: priorities “do not preempt already-running work” (A2).',
      },
      {
        label: 'Only once the internal kernel has finished entirely, because a priority only orders kernels launched at the same moment.',
        why: 'That is what equal priority gives in the explorer. With a higher priority they take the first slots that free up.',
      },
      {
        label: 'At once, on part of the GPU set aside for high-priority streams that the internal kernel’s blocks cannot occupy.',
        why: 'No such reserved hardware is documented. A priority decides who gets a slot that frees, nothing more.',
      },
    ],
    explain: (
      <p>
        A stream priority is a hint about pending work. Once the internal kernel&rsquo;s blocks occupy every slot,
        nothing is taken from them. What the higher priority buys is the next slots: as each internal block finishes, a
        waiting boundary block takes its place instead of another internal block. In the explorer, with the internal
        kernel issued first, that is the difference between starting at the first block retirement and starting after
        the internal kernel&rsquo;s last block has been dispatched.
      </p>
    ),
    source: 'p. 558; CUDA 13.5 runtime reference and programming guide, checked 2026-11-20 (notes/today/ch23-facts.md, A2); the slot picture is this page’s model',
  },
  'ch23/23.4/q-hip-prio': {
    kind: 'choice',
    id: 'ch23/23.4/q-hip-prio',
    prompt: (
      <>
        On ROCm 10.2 your code reads <code>streamPrioritiesSupported</code> (1) and a priority range of −1 to 1, then
        creates the two boundary streams at −1. Are they preferred over the internal stream?
      </>
    ),
    options: [
      {
        label: 'No: by default the runtime treats every stream as normal priority (DEBUG_HIP_IGNORE_STREAM_PRIORITY is true), whatever it reports.',
        correct: true,
      },
      {
        label: 'Yes: −1 is the greatest priority in that range, and the device has just said that it supports stream priorities.',
        why: 'That is what the reports say. The runtime switch overrides them (B2).',
      },
      {
        label: 'Yes, but only their copies: AMD applies stream priorities to memory transfers rather than to kernel launches.',
        why: 'Nothing in AMD’s sources says so, and the switch makes every stream normal priority.',
      },
      {
        label: 'No: HIP’s convention is the reverse of CUDA’s, so −1 is the least priority in the range it reports.',
        why: 'HIP keeps CUDA’s convention: “Lower numbers represent higher priorities.” (B2)',
      },
    ],
    explain: (
      <p>
        AMD&rsquo;s header keeps CUDA&rsquo;s names and convention, and also warns that stream priority is disabled. The
        runtime flag <code>DEBUG_HIP_IGNORE_STREAM_PRIORITY</code> defaults to true and forces every queue to normal
        priority, while <code>hipGetDeviceProperties</code> still reports <code>streamPrioritiesSupported = 1</code> and
        the range comes back as −1 to 1. On AMD, do not build an overlap that depends on priorities, and do not try to
        measure one: issue the boundary kernels first.
      </p>
    ),
    source: 'not in the book; ROCm 10.2 sources, checked 2026-11-20 (notes/today/ch23-facts.md, B2)',
  },
  'ch23/23.4/q-memset': {
    kind: 'choice',
    id: 'ch23/23.4/q-memset',
    prompt: (
      <>
        The book replaces <code>cudaMemset</code> with <code>cudaMemsetAsync</code> in <code>internalStream</code>,
        saying <code>cudaMemset</code> is blocking. What would a plain <code>cudaMemset(l2norm_d, 0, sizeof(float))</code>{' '}
        actually do at line 04 of this loop?
      </>
    ),
    options: [
      {
        label: 'Return without waiting, but queue the reset in the legacy default stream, which waits for every blocking stream and holds them all.',
        correct: true,
      },
      {
        label: 'Block the host until the reset has finished on the device, which is exactly the reason the book gives for replacing it.',
        why: 'NVIDIA documents cudaMemset on device memory as asynchronous with respect to the host (A3).',
      },
      {
        label: 'Run in internalStream anyway, since the runtime puts a memset into the stream that last used the same pointer.',
        why: 'There is no such rule: the synchronous forms are issued through the default stream (A3).',
      },
      {
        label: 'Fail with an error, because l2norm_d lives in pinned host memory, which only the Async form is able to write.',
        why: 'l2norm_d is device memory. The pinned buffer is l2norm_h.',
      },
    ],
    explain: (
      <p>
        The book&rsquo;s fix stands, for another reason. A plain <code>cudaMemset</code> does not make the host wait,
        but it goes into the legacy default stream, and streams created with <code>cudaStreamDefault</code> are blocking
        streams: the legacy stream first waits for all of them and then makes all of them wait for it. Every iteration
        would begin with a device-wide join. <code>cudaMemsetAsync</code> in <code>internalStream</code> orders the reset
        with that one stream, and the <code>resetL2</code> event carries the order to the other two.
      </p>
    ),
    source: 'p. 559; CUDA 13.5 runtime reference, checked 2026-11-20 (notes/today/ch23-facts.md, A3, A9)',
  },
  'ch23/23.4/q-mechanism': {
    kind: 'choice',
    id: 'ch23/23.4/q-mechanism',
    prompt: (
      <>
        In Fig. 23.14 the copy of the norm, queued in <code>internalStream</code>, must wait for the top kernel in{' '}
        <code>topStream</code>. What enforces it, and why is the order of the host&rsquo;s calls not enough?
      </>
    ),
    options: [
      {
        label: 'An event recorded in topStream after the kernel, and a wait for it queued in internalStream: call order only orders work within one stream.',
        correct: true,
      },
      {
        label: 'Nothing extra: the host launches the top kernel before it queues the copy, and the runtime keeps that call order across all streams.',
        why: 'The runtime keeps call order within a stream. Across streams, work may run in any order (p. 556).',
      },
      {
        label: 'cudaMemcpyAsync itself waits for all earlier work on the device, in every stream, before it starts copying the norm.',
        why: 'That is the legacy default stream’s behaviour (A3), not that of a copy queued in a named stream.',
      },
      {
        label: 'The boundary streams have the higher priority, so the scheduler finishes their kernels before it starts the copy.',
        why: 'A priority is a hint for pending kernels. It orders nothing, and it does not apply to copies at all (A2).',
      },
    ],
    explain: (
      <p>
        Stream order is the only order the runtime keeps by itself, and it stops at the stream&rsquo;s edge. To make{' '}
        <code>internalStream</code> wait for a point in <code>topStream</code>, Fig. 23.14 records{' '}
        <code>computeTop</code> after the top kernel (line 11) and queues a wait for it in <code>internalStream</code>{' '}
        (line 19). A <code>cudaStreamSynchronize(topStream)</code> on the host would also work, but the host would sit
        idle until the top kernel finished.
      </p>
    ),
    source: 'pp. 559–560; CUDA 13.5 runtime reference (notes/today/ch23-facts.md, A5)',
  },
  'ch23/23.4/q-metal-events': {
    kind: 'choice',
    id: 'ch23/23.4/q-metal-events',
    prompt: (
      <>
        You port Fig. 23.14 to Metal. Lines 19 and 20 make the copy wait for two producers, one per boundary pass. Can
        one <code>MTLEvent</code> that both boundary passes signal do that job?
      </>
    ),
    options: [
      {
        label: 'No: an event’s value only increases to a given number, so two producers cannot combine signals; use one event each, or two values in a known order.',
        correct: true,
      },
      {
        label: 'Yes: Metal counts the signals an event receives, like a semaphore, so the copy can wait until the count reaches two.',
        why: 'Apple says the opposite: an event’s signal is “unlike a semaphore” (C2).',
      },
      {
        label: 'Yes, but only an MTLSharedEvent, which adds the signal counter that a plain MTLEvent does not have.',
        why: 'MTLSharedEvent adds waits from the CPU, other devices and other processes, not counting (C2).',
      },
      {
        label: 'No: work in one Metal command queue has no way at all to wait for work that runs in another queue.',
        why: 'It has: a command buffer can wait for an event value that a buffer in another queue signals (C2).',
      },
    ],
    explain: (
      <p>
        A Metal event is a 64-bit value that only grows, and a wait passes once the value reaches the number it waits
        for. If two passes signal the same event, the waiter cannot tell whether one or both have done so. Apple&rsquo;s
        advice is one event per producing workload. A CUDA event is different again, a snapshot of one stream, so Fig.
        23.14 also needs two of them, for a different reason: the producers are in two streams.
      </p>
    ),
    source: 'not in the book; Apple Metal documentation, checked 2026-11-20 (notes/today/ch23-facts.md, C2, A5)',
  },
  'ch23/23.4/q-drop-reset': {
    kind: 'choice',
    id: 'ch23/23.4/q-drop-reset',
    prompt: (
      <>
        Someone deletes the two <code>cudaStreamWaitEvent</code> calls on <code>resetL2</code> (lines 06–07). What can go
        wrong, and why does a test on the course&rsquo;s emulator still pass?
      </>
    ),
    options: [
      {
        label: 'A boundary kernel may add to the norm before the memset zeroes it, losing its part; the emulator runs streams synchronously, in call order.',
        correct: true,
      },
      {
        label: 'The program deadlocks, because topStream now waits for an event that is never recorded; the emulator does not model waits.',
        why: 'Removing a wait cannot create one. The emulator does have stand-ins for events (labs/README.md).',
      },
      {
        label: 'Nothing: the memset and both boundary kernels are in the same stream, so they stay in order without the waits.',
        why: 'The memset is in internalStream (line 04); the boundary kernels are in topStream and bottomStream.',
      },
      {
        label: 'The halos are sent twice per iteration; the emulator does not run MPI, so it cannot see the second exchange.',
        why: 'The exchanges are host calls at lines 28 to 34. The reset waits have nothing to do with them.',
      },
    ],
    explain: (
      <p>
        Without the waits nothing orders the reset in <code>internalStream</code> against the kernels in the other two
        streams. If a boundary block finishes and does its <code>atomicAdd</code> before the memset runs, the memset wipes
        that contribution, the norm comes out too small, and the loop can stop before it has converged. Whether it
        happens is a matter of timing. In the explorer it never fires in Fig. 23.14, where the host has just emptied{' '}
        <code>internalStream</code>, and it does fire in the NCCL loop, where that stream is still waiting on the
        exchanges. The emulator runs streams one call at a time, so the race cannot occur there at all.
      </p>
    ),
    source: 'pp. 559–560; labs/README.md (emulator streams are synchronous); this page’s model',
  },
  'ch23/23.4/q-nccl-waits': {
    kind: 'choice',
    id: 'ch23/23.4/q-nccl-waits',
    prompt: (
      <>
        In the NCCL loop (Fig. 23.20) the two waits on <code>exchangeTop</code> and <code>exchangeBottom</code> (lines
        43–44) are removed. What can happen?
      </>
    ),
    options: [
      {
        label: 'ncclGroupEnd only queued the exchanges, so the next iteration’s kernels may start before the halos arrive and read stale values.',
        correct: true,
      },
      {
        label: 'Nothing: ncclGroupEnd returns only when both the send and the receive of its group have completed on the GPU.',
        why: 'NCCL: the primitives “have been enqueued to the provided streams, but are not necessarily complete” (A12).',
      },
      {
        label: 'The next memset waits forever, because without lines 43–44 the events exchangeTop and exchangeBottom are never recorded.',
        why: 'Lines 31 and 36 record the events. Lines 43–44 only wait for them.',
      },
      {
        label: 'Each exchange runs twice, because exchangeTop and exchangeBottom are recorded again at the end of the next iteration.',
        why: 'Re-recording an event replaces its snapshot; it repeats no work (A5).',
      },
    ],
    explain: (
      <p>
        The group call does not block: after <code>ncclGroupEnd</code> the only guarantee is that the send and the
        receive are in the stream. Lines 43–44 queue waits in <code>internalStream</code>; the next memset and{' '}
        <code>resetL2</code> come after them, and both boundary kernels wait on <code>resetL2</code>, so nothing of the
        next iteration starts until both exchanges are complete. Without them, the next top kernel waits only for its
        own stream, which holds the top exchange, while the top halo it reads arrives with the bottom exchange.
      </p>
    ),
    source: 'pp. 567, 573; NCCL 2.32 API reference (notes/today/ch23-facts.md, A12); the chain through resetL2 is our reading of the code (notes/ch23.md)',
  },
  'ch23/23.4/q-nvshmem-barrier': {
    kind: 'choice',
    id: 'ch23/23.4/q-nvshmem-barrier',
    prompt: (
      <>
        In the NVSHMEM loop (Fig. 23.25) the host already synchronises the stream at line 14, before{' '}
        <code>MPI_Allreduce</code>. Why is <code>nvshmemx_barrier_all_on_stream</code> still needed at line 18?
      </>
    ),
    options: [
      {
        label: 'The sync proves only that this kernel finished computing and issuing puts; puts may land later, and the neighbours’ puts may be in flight.',
        correct: true,
      },
      {
        label: 'To reset the norm for the next iteration: the barrier zeroes the symmetric buffers on every processing element.',
        why: 'The memset at line 04 resets the norm. A barrier moves no data.',
      },
      {
        label: 'To release the symmetric heap, which nvshmem_malloc locks for as long as a kernel that writes to it runs.',
        why: 'Nothing is locked. The heap is freed once, with nvshmem_free, after the loop.',
      },
      {
        label: 'It is not needed: synchronising the stream also waits until every remote put the kernel made has been delivered.',
        why: 'The book says the opposite (p. 573), and so does NVSHMEM: a put “may return before the data is delivered” (A14).',
      },
    ],
    explain: (
      <p>
        <code>nvshmem_float_p</code> starts a remote store and may return before the data reaches the other PE. When the
        kernel ends, its puts can still be travelling, and so can the puts the neighbours aimed at this GPU&rsquo;s
        halos. The barrier on the stream makes every PE wait until all memory operations from the kernels before it in
        the stream have completed, and the next memset and kernel are queued behind it. Drop it, and once an exchange
        outlasts the rest of the iteration, the next kernel reads halos that have not arrived.
      </p>
    ),
    source: 'pp. 573–574; NVSHMEM documentation, checked 2026-11-20 (notes/today/ch23-facts.md, A14)',
  },
  'ch23/23.4/q-tp': {
    kind: 'choice',
    id: 'ch23/23.4/q-tp',
    prompt: (
      <>
        An inference server splits each layer&rsquo;s weights across 4 GPUs (tensor parallelism) instead of giving each
        GPU a quarter of the layers (pipeline parallelism). Which one puts a collective on every layer&rsquo;s critical
        path, and what does that ask of the hardware?
      </>
    ),
    options: [
      {
        label: 'Tensor parallelism: every layer ends in an all-reduce that all GPUs wait for, so it wants the fastest link and hides poorly.',
        correct: true,
      },
      {
        label: 'Pipeline parallelism: at every layer each stage has to wait for all the other stages, so it needs the fastest link.',
        why: 'A stage sends activations once, at its boundary, to the next stage. It waits for no all-reduce.',
      },
      {
        label: 'Neither: with NCCL both kinds overlap the communication fully with computation, so the link speed hardly matters.',
        why: 'The all-reduce’s result is the next layer’s input. Nothing independent is left to overlap it with.',
      },
      {
        label: 'Tensor parallelism, but only during prefill: in decoding the all-reduce is skipped because one token is too small.',
        why: 'Every layer’s partial outputs must be summed for every token, decoding included.',
      },
    ],
    explain: (
      <p>
        In tensor parallelism each GPU computes part of every layer, and the parts must be summed before the next layer
        can start. That is this chapter&rsquo;s halo exchange without the internal rows to hide behind. Pipeline
        parallelism sends activations once per stage boundary and can hide that transfer behind other micro-batches, at
        the cost of a bubble while the pipeline fills and drains.
      </p>
    ),
    source: 'not in the book; ours, on llama.cpp and vLLM sources and docs read 2026-11-20 (notes/today/ch23-facts.md, D2, D5, D7)',
  },
};

// ---------------------------------------------------------------- the page
export function S234() {
  const it = STREAMS_DEF.iters[0];
  return (
    <>
      <p>
        Section <a href={href('ch23', '23.3')}>23.3</a> gave you the parts: streams, asynchronous copies, pinned memory.
        This page assembles them into the book&rsquo;s loop and asks the question every program with more than one
        stream has to answer: when one operation must wait for another, who does the waiting? The host, a stream, or
        nobody. The answer is also what separates the chapter&rsquo;s three libraries.
      </p>
      <Source section="23.3" pages="556–563" figs="Figs. 23.14, 23.15" />

      <h2>One kernel, three launches</h2>
      <p>
        Fig. 23.14 keeps Fig. 23.1&rsquo;s kernel and launches it three times per iteration, each time on a slice of the
        rows and into its own stream: the top boundary row into <code>topStream</code>, the bottom one into{' '}
        <code>bottomStream</code>, the internal rows into <code>internalStream</code>. Each launch gets a pointer into the
        same arrays and a row count, and the kernel&rsquo;s guard does the rest. You met the arithmetic in the explorer
        on the <a href={href('ch23', 'start')}>first page</a>; here it is as a check.
      </p>
      <Source section="23.3" pages="557–558" figs="Fig. 23.14, lines 10–16" />
      <Ask q={questionsS234['ch23/23.4/q-threerows']} />
      <p>
        So the three launches write row 1, row <code>ny − 2</code> and rows 2 to <code>ny − 3</code>: every interior row
        exactly once. The book says which rows each launch needs as input; which rows each one writes is our derivation
        from the guard.
      </p>

      <h2>Priorities: a hint, not an order</h2>
      <p>
        Kernels in different streams may run in any order. The book expects the boundary kernels to run first, since
        they are launched first, and notes that nothing guarantees it. If the internal kernel starts first and its blocks
        occupy the whole GPU, the boundary kernels wait, the exchanges wait for them, and the overlap is gone. The
        book&rsquo;s remedy is to give the two boundary streams a higher priority. The setup code beside the puzzle below
        creates the streams that way.
      </p>
      <Source section="23.3" pages="558" />
      <p>
        Two things the book leaves open. The first is the direction. On CUDA a lower number means a higher priority, and
        the default is 0. <code>cudaDeviceGetStreamPriorityRange(&amp;low, &amp;high)</code> returns the least priority
        first and the greatest second, so the book&rsquo;s <code>low, high</code> is right as printed. The range itself
        belongs to the device; read it, do not assume it.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 runtime reference (notes/today/ch23-facts.md, A1)">
        <p>
          &ldquo;Stream priorities follow a convention where lower numbers imply greater priorities.&rdquo; Both outputs
          are 0 on a device without priorities. NVIDIA&rsquo;s own version of this code does what the book does.
        </p>
      </Today>
      <p>
        The second is what a priority does. NVIDIA calls it a hint: it applies to kernels still waiting to start, it
        does not preempt work that is already running, it guarantees no order, and it has no effect on copies. In the
        hazard the book describes, where the internal kernel already fills the GPU, a higher priority cannot make room.
        It only gives the boundary kernels first claim on the slots that free up as the internal kernel&rsquo;s blocks
        finish. So keep both of the book&rsquo;s measures, the launch order and the priority, and then measure: neither
        is a guarantee.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 runtime reference and programming guide (notes/today/ch23-facts.md, A2)">
        <p>
          Priorities &ldquo;do not preempt already-running work or provide any other functional guarantee on execution
          order&rdquo;, and &ldquo;have no effect on host-to-device and device-to-host memory operations&rdquo;.
        </p>
      </Today>
      <Ask q={questionsS234['ch23/23.4/q-prio']} />
      <p>
        On AMD the calls keep their names (<code>hipDeviceGetStreamPriorityRange</code>,{' '}
        <code>hipStreamCreateWithPriority</code>) and the convention, but ROCm 10.2 ignores priorities by default: a
        runtime switch treats every stream as normal priority, while the device still reports that it supports them.
        Metal has no priority for compute queues at all; the only priority in its headers belongs to the queue that
        loads file data into resources.
      </p>
      <Today checked="2026-11-20 against ROCm 10.2 sources and the macOS 27 SDK headers (notes/today/ch23-facts.md, B2, C3)">
        <p>
          HIP: <code>DEBUG_HIP_IGNORE_STREAM_PRIORITY</code> defaults to true. Metal: <code>MTLIOCommandQueue</code> has
          a priority; <code>MTLCommandQueue</code> has none.
        </p>
      </Today>
      <Ask q={questionsS234['ch23/23.4/q-hip-prio']} />

      <h2>One norm, three streams: events</h2>
      <p>
        The three kernels share one thing: the norm. Each adds its blocks&rsquo; squared residues to{' '}
        <code>l2norm_d</code> with an atomic. That creates two orderings that cross streams. The reset must happen before
        any of the three kernels adds to the norm, and all three must have finished before the norm is copied to the
        host. But the reset and the copy can each sit in only one stream.
      </p>
      <Source section="23.3" pages="559" />
      <p>
        Stream order cannot express that, and a host synchronisation could only at the price of stopping the host.
        CUDA&rsquo;s answer is the event: a marker you record in one stream that another stream can be told to wait for.{' '}
        <code>cudaEventRecord(e, s)</code> places it; <code>cudaStreamWaitEvent(s2, e, 0)</code> makes everything queued
        in <code>s2</code> afterwards wait until <code>s</code> reaches the marker. No host thread waits. Creating the
        event with <code>cudaEventDisableTiming</code> says it will never time anything, which makes it cheaper for
        exactly this use.
      </p>
      <Source section="23.3" pages="559–560" />
      <p>
        One detail matters inside a loop. An event is not a counter. Recording it captures what its stream holds at
        that moment, and recording it again replaces the capture. A wait uses whatever the event held when the wait was
        queued, and later records do not change it. That is why Fig. 23.14 can record the same three events in every
        iteration: each iteration&rsquo;s waits refer to that iteration&rsquo;s records.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 runtime reference (notes/today/ch23-facts.md, A5)">
        <p>
          Waits &ldquo;use the most recent captured state at the time of the API call, and are not affected by later
          calls to cudaEventRecord()&rdquo;. The book calls <code>cudaEventDisableTiming</code> a compile-time constant;
          it is a flag value. HIP has the same names and flags.
        </p>
      </Today>
      <p>
        The book also turns the reset from <code>cudaMemset</code> into <code>cudaMemsetAsync</code> in{' '}
        <code>internalStream</code>, saying the first is blocking. That reason is wrong: NVIDIA documents{' '}
        <code>cudaMemset</code> on device memory as asynchronous with respect to the host. The change is right for
        another reason. A plain <code>cudaMemset</code> goes into the legacy default stream, which waits for every
        blocking stream and makes them all wait for it, and the book&rsquo;s streams are blocking streams. The reset
        would join the whole device once per iteration.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 runtime reference (notes/today/ch23-facts.md, A3, B4)">
        <p>
          &ldquo;The cudaMemset functions are asynchronous with respect to the host except when the target memory is
          pinned host memory.&rdquo; AMD does not document whether plain <code>hipMemset</code> makes the host wait.
        </p>
      </Today>
      <Ask q={questionsS234['ch23/23.4/q-memset']} />
      <p>
        Now take the loop apart. One iteration needs the ten orderings below. For each, pick what enforces it in Fig.
        23.14: the two operations being in the same stream, an event and a stream wait, the host waiting for a stream,
        or a call that returns only when its work is done. The feedback cites the line. Once all ten are placed, compare
        them with the NCCL and NVSHMEM loops.
      </p>
      <DependencyPuzzle />
      <p>
        Look at what the answers have in common. In Fig. 23.14 work is ordered in one of three places: inside one
        stream, between streams through an event, or on the host, and a blocking call is the host waiting by another
        name. The NCCL and NVSHMEM loops move orderings off the host: into stream order, into event waits, into a
        barrier queued in a stream, or into the kernel&rsquo;s own order of operations. That is what frees the host and
        removes the idle time you are about to see.
      </p>
      <Ask q={questionsS234['ch23/23.4/q-mechanism']} />
      <p>
        Metal&rsquo;s events work differently, and the difference shows in exactly this pattern. An{' '}
        <code>MTLEvent</code> is a 64-bit value that only increases. A command buffer signals a value and another waits
        for a value, between passes, never inside one. Several producers cannot combine their signals into one event,
        so lines 19 and 20 become two events, or two values signalled in a known order. Metal&rsquo;s waits can also
        time out, which CUDA&rsquo;s cannot, and <code>MTLSharedEvent</code> adds waits from the CPU.
      </p>
      <Today checked="2026-11-20 against Apple's Metal documentation and the macOS 27 SDK headers (notes/today/ch23-facts.md, C2)">
        <p>
          &ldquo;Multiple producing workloads can&rsquo;t combine their signals with one event to unblock any dependent
          workloads.&rdquo;
        </p>
      </Today>
      <Ask q={questionsS234['ch23/23.4/q-metal-events']} />

      <h2>Copy early, wait once</h2>
      <p>
        Line 21 queues the copy of the norm to the host straight after the waits, before the halo exchange. Because{' '}
        <code>cudaMemcpyAsync</code> returns at once, the copy can run while the exchanges are in flight, as soon as the
        internal kernel is done. The host must not read <code>l2norm_h</code> before the copy has landed, and that is
        line 37&rsquo;s job: one wait on <code>internalStream</code>, just before <code>MPI_Allreduce</code>.
      </p>
      <Source section="23.3" pages="560–561" />
      <p>
        It only works if <code>l2norm_h</code> is pinned. Section <a href={href('ch23', '23.3')}>23.3</a> showed why: from
        pageable memory the runtime stages the copy through a pinned buffer of its own, with the host taking part, and
        the call stops being asynchronous. In this loop that would stop the host at line 21 until the internal kernel
        had finished, and both exchanges would start only after it. The explorer below lets you try that.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 programming guide (notes/today/ch23-facts.md, A6)">
        <p>
          Not an error: <code>cudaMemcpyAsync</code> &ldquo;will function correctly&rdquo; with pageable memory, &ldquo;but
          it will revert to a synchronous behavior which will not overlap with other work.&rdquo;
        </p>
      </Today>

      <h2>One iteration in time</h2>
      <p>
        Fig. 23.15 draws, for one iteration, when the host inserts each operation and when a stream runs it. Both are
        about time, so here they are as an animation. Every bar comes from the same model as the explorer after it, at
        the durations the explorer starts from: an internal kernel of {DEFAULTS.internal} units, boundary kernels of{' '}
        {DEFAULTS.boundary}, exchanges of {DEFAULTS.exTop}.
      </p>
      <Source section="23.3" pages="561–562" figs="Fig. 23.15" />
      <LoopScene />
      <p>
        The bottom rows were ready at {r2(it.bottom.end)} and left at {r2(it.exBottom.start)}. Nothing on the GPU caused
        that: the host was inside the top <code>MPI_Sendrecv</code>, and a blocking call cannot be interrupted to start
        another. The book names reducing this overhead as a design goal of NCCL and NVSHMEM.
      </p>
      <Source section="23.3" pages="562" />

      <h2>Who waits</h2>
      <p>
        Now drive the same model yourself. Before you touch it, commit to a number.
      </p>
      <Predict
        id="ch23/23.4/p-window"
        prompt={
          <>
            In the NCCL loop, with an internal kernel of {DEFAULTS.internal} and boundary kernels of {DEFAULTS.boundary},
            how long can each exchange be and still hide behind the internal kernel?
          </>
        }
        unit="units"
      >
        {(c) => (
          <p style={{ margin: 0 }} data-window={r2(WINDOW)}>
            The model says <b>{r2(WINDOW)}</b>{Math.abs(c - WINDOW) < 0.5 ? ', close to your number' : ''}. Each exchange starts
            when its boundary kernel ends, at about {r2(NCCL_DEF.bottom.end)}, and the internal kernel, which shared the
            GPU with the boundary kernels at first, ends at {r2(NCCL_DEF.internal.end)}. Check it: set both exchanges to{' '}
            {r2(WINDOW)} and then a little more, and watch &ldquo;exchanges hidden&rdquo;. In the MPI loop the two exchanges
            run one after the other, so each can be only about half as long.
          </p>
        )}
      </Predict>
      <p>
        Then try three things. Switch between MPI with streams and NCCL and watch the bottom exchange&rsquo;s lane.
        Lengthen both exchanges past the internal kernel and see which versions still hide them. Choose &ldquo;internal
        kernel first&rdquo; and switch the priority between equal and higher: when do the boundary kernels start? Finally,
        remove one ordering at a time and find durations at which it breaks something.
      </p>
      <Simplification>
        <p>
          One rank whose neighbours behave exactly like it. A GPU of 8 block slots; a kernel is equal-length blocks,
          nothing is preempted, and a one-kernel version runs its boundary rows&rsquo; blocks first. Each exchange has
          its own link. MPI calls hold the host until their exchange ends;
          NCCL&rsquo;s own kernels take no slots. A pageable copy is fully synchronous. Every API call costs the same small
          time. Times are unitless.
        </p>
      </Simplification>
      <WhoWaits />
      <p>
        Three things to take from it. In the MPI loop the bottom exchange waits for the blocking top exchange even when
        the bottom kernel finished long before, which is the overhead of p. 562; in the NCCL loop it does not, and the
        two exchanges run side by side. An exchange longer than the internal kernel cannot be hidden by any of the
        versions, which is the condition the book sets on p. 554: enough internal rows. And with the internal kernel
        issued first, a higher priority only gets the boundary kernels the first slots that free up, so the exchanges
        start late; with equal priority the boundary kernels wait for the internal kernel&rsquo;s last blocks, and the
        exchanges no longer overlap it at all.
      </p>
      <Source section="23.3" pages="554, 558, 562" />
      <p>
        The bug hunt has a second lesson. Most removed orderings break nothing at the default durations, and some break
        nothing in one loop and something in another: they are races between streams, decided by timing, not certain
        failures. The course&rsquo;s emulators run every stream synchronously, in call order, so none of these races can
        happen there. A test that passes on the emulator says nothing about them; only reasoning about the orderings
        does.
      </p>
      <Beyond>
        <p>
          The model and its findings are ours. The emulators&rsquo; stream stand-ins are synchronous (labs/README.md,
          &ldquo;What is and is not modelled&rdquo;).
        </p>
      </Beyond>
      <Ask q={questionsS234['ch23/23.4/q-drop-reset']} />
      <Ask q={questionsS234['ch23/23.4/q-nccl-waits']} />
      <Ask q={questionsS234['ch23/23.4/q-nvshmem-barrier']} />

      <h2>Three libraries, three places to wait</h2>
      <p>
        The book uses this loop to introduce three programming models. Each gets a paragraph here: what it is for, where
        to read it in the book, and what has changed since.
      </p>
      <p>
        <b>MPI</b> is for processes with separate memories that exchange messages. You start the same program several
        times with <code>mpiexec</code> (the standard&rsquo;s name; <code>mpirun</code> is a common synonym), and each
        process learns its rank and the number of ranks from a communicator, <code>MPI_COMM_WORLD</code> for all of them.
        The halo exchange uses <code>MPI_Sendrecv</code>, which sends to one neighbour, receives from the other and
        returns only when the exchange is complete; the norm uses <code>MPI_Allreduce</code>, which sums one float from
        every rank and gives each rank the result. Passing device pointers to MPI, as these loops do, needs a CUDA-aware
        MPI, and that is a property of how your MPI was built.
      </p>
      <Source section="23.2" pages="545–553" figs="Figs. 23.4–23.11" />
      <p>
        MPICH turns GPU support on by itself if it finds CUDA or HIP when it is configured; Open MPI has to be configured
        with CUDA and use a UCX that was built with it; MVAPICH&rsquo;s GPU line is now called MVAPICH-Plus. Check your
        build; do not assume it.
      </p>
      <Today checked="2026-11-20 against the MPI Forum, MPICH, Open MPI and MVAPICH documentation (notes/today/ch23-facts.md, A10, A11, B7)">
        <p>
          MPI 5.0 (June 2025) is the current standard. On AMD, ROCm-aware Open MPI goes through UCX.
        </p>
      </Today>
      <p>
        <b>NCCL</b> provides communication primitives that run on the GPU and are queued in streams, so the host stops
        being the go-between. A send and a receive run as one fused operation when you put them between{' '}
        <code>ncclGroupStart</code> and <code>ncclGroupEnd</code>. Note what <code>ncclGroupEnd</code> promises: the
        operations are queued, not complete. That is why lines 31 and 36 below record events, and lines 43–44 make the
        internal stream wait for them before the next iteration.
      </p>
      <Source section="23.4" pages="563–568" figs="Figs. 23.16–23.21" />
      <Code
        tabs={[{ label: 'Fig. 23.20, lines 27–44', code: NCCL_EXCHANGE, status: CHECKED_NCCL }]}
        firstLine={27}
        caption={
          <>
            The NCCL loop&rsquo;s exchange and the end of its body, with the book&rsquo;s line numbers; lines 1–26 are
            those of Fig. 23.14. The comment on line 38 is ours. The setup it assumes is in the file&rsquo;s header:
            each rank calls <code>cudaSetDevice</code> before <code>ncclCommInitRank</code>, which Fig. 23.17 leaves out.
          </>
        }
      />
      <p>
        Two cautions from NCCL&rsquo;s own documentation. <code>ncclCommInitRank</code> needs each rank&rsquo;s device to
        be set first, which Fig. 23.17 does not show. And using NCCL and a CUDA-aware MPI to move data between the same
        GPUs at the same time can deadlock. On AMD, RCCL keeps NCCL&rsquo;s function names; its header is{' '}
        <code>&lt;rccl/rccl.h&gt;</code>.
      </p>
      <Today checked="2026-11-20 against the NCCL 2.32 documentation and ROCm 10.2 sources (notes/today/ch23-facts.md, A11, A12, B5)">
        <p>
          <code>ncclGroupEnd</code>: the primitives &ldquo;have been enqueued to the provided streams, but are not
          necessarily complete.&rdquo;
        </p>
      </Today>
      <p>
        <b>NVSHMEM</b> is one-sided: a GPU thread can put a value into another process&rsquo;s GPU memory, or get one,
        without that process doing anything. Addresses are named through a symmetric heap: every process allocates the
        same objects at the same offsets, so a thread passes the address of its own copy and the library finds the
        remote one. In Fig. 23.23 a thread that computes a top boundary value puts it straight into the bottom halo of
        the top neighbour. That gives the overlap without splitting the kernel, and it lets steps that need
        communication between them be fused into one kernel. The price is the barrier on the stream: a put may return
        before its data is delivered.
      </p>
      <Source section="23.5" pages="568–574" figs="Figs. 23.22–23.25" />
      <p>
        The book adds that one put per grid point, as in Fig. 23.23, suits GPUs connected by NVLink, and that over slower
        links fewer, larger puts are better, for example <code>nvshmemx_float_put_block</code>, where a whole block
        cooperates on one put. That advice is the book&rsquo;s; no NVSHMEM page we found states it, so treat it as
        unverified.
      </p>
      <Today checked="2026-11-20 against the NVSHMEM documentation and GitHub, ROCm 10.2 and PyTorch 2.15 (notes/today/ch23-facts.md, A14, A15, B6, D5)">
        <p>
          NVSHMEM is open source (Apache-2.0) since August 2025. AMD&rsquo;s counterpart is rocSHMEM, with{' '}
          <code>rocshmem_</code> names and <code>_wg</code>/<code>_wave</code> variants. PyTorch&rsquo;s symmetric memory
          runs over NVSHMEM, rocSHMEM or NCCL windows.
        </p>
      </Today>
      <p>
        The book&rsquo;s summary draws the line between host-initiated and device-initiated communication.
        Host-initiated, as with NCCL here, suits data that is coarse-grained and can be determined independently of the
        computation it overlaps, like a boundary row computed by its own kernel. Device-initiated, as with NVSHMEM, suits
        fine-grained data determined as part of that computation. The book adds that the APIs are converging. They have
        converged further than it says: besides its two-sided calls, NCCL now registers symmetric windows, has a device
        API, and has one-sided <code>ncclPutSignal</code> and <code>ncclWaitSignal</code>, so &ldquo;may add one-sided
        communication soon&rdquo; is out of date.
      </p>
      <Source section="23.6" pages="574–575" />
      <Today checked="2026-11-20 against the NCCL 2.32 documentation (notes/today/ch23-facts.md, A13)">
        <p>
          Window registration since 2.27, the device API since 2.28, <code>ncclPutSignal</code> since 2.29.
        </p>
      </Today>
      <p>
        Metal has none of the three. On a Mac, work across several devices is a framework feature that spans machines,
        not a Metal API: MLX&rsquo;s distributed layer, for one, runs over MPI, a TCP ring, or RDMA over Thunderbolt.
      </p>
      <Today checked="2026-11-20 (notes/today/ch23-facts.md, C6; ch24-facts C4)">
        <p>No collective or one-sided API is documented in Metal.</p>
      </Today>

      <h2>Where this sits in a model server</h2>
      <p>
        Tensor parallelism puts an all-reduce at the end of every layer: each GPU holds a slice of every weight matrix,
        computes a partial output, and the partials must be summed before the next layer can start. That is the role
        the halo exchange plays in this chapter, without the internal rows to hide behind: nothing independent is left
        to compute while the sum travels. llama.cpp&rsquo;s <code>--split-mode tensor</code> uses the book&rsquo;s group
        pattern for it, <code>ncclAllReduce</code> between <code>ncclGroupStart</code> and <code>ncclGroupEnd</code> on
        each device&rsquo;s stream, through NCCL or, built for HIP, RCCL. It is also why vLLM keeps tensor parallelism
        inside a node, where the links are fastest, and spans nodes with pipeline parallelism, whose transfers can hide
        behind other work like the pipeline on page <a href={href('ch23', '23.3')}>23.3</a>.
      </p>
      <Beyond>
        <p>
          The mapping is ours. The facts under it: llama.cpp <code>master</code> and the vLLM docs, read 2026-11-20
          (notes/today/ch23-facts.md, D2, D7).
        </p>
      </Beyond>
      <Ask q={questionsS234['ch23/23.4/q-tp']} />

      <h2>Reading real backends</h2>
      <p>
        You will meet this page&rsquo;s mechanisms on a single GPU too. llama.cpp&rsquo;s CUDA backend creates its
        streams non-blocking and its events with <code>cudaEventDisableTiming</code>. After a copy between two devices
        it records an event in the source stream and makes the destination stream wait for it: Fig. 23.14&rsquo;s
        pattern, across devices. With <code>GGML_CUDA_GRAPH_OPT=1</code>, and CUDA graphs in use, it can run the Q, K
        and V projections on separate streams, forking with one event and joining with others. PyTorch&rsquo;s{' '}
        <code>stream.wait_stream(other)</code> is record-then-wait in one call. MLX on Metal gives each stream its own
        command queue and orders streams with <code>MTLSharedEvent</code> values.
      </p>
      <Today checked="2026-11-20 against llama.cpp master, PyTorch main and MLX main (notes/today/ch23-facts.md, D3, D4, D6)">
        <p>
          llama.cpp: <code>ggml-cuda.cu</code> and <code>common.cuh</code>. PyTorch: <code>torch/cuda/streams.py</code>.
          MLX: <code>mlx/backend/metal/device.cpp</code> and <code>event.cpp</code>. Line numbers drift.
        </p>
      </Today>
      <p>
        The <a href={href('ch23', 'labs')}>lab</a> takes the single-GPU half of this to your own device: hide copies
        behind a kernel with streams and pinned memory, and measure how much of the copy time disappears.
      </p>
    </>
  );
}
