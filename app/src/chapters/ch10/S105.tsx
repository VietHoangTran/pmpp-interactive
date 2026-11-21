import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  ACCESS_LABEL,
  KIND_LABEL,
  bookFormula,
  requests,
  type AccessName,
  type Kind,
  type RequestStep,
} from '../../sim/reduction';
import { widthChoices } from '../../sim/warps';

import { NoteGroup } from './NoteGroup';
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const SIZES = [16, 64, 256, 1024, 2048];
const KINDS: Kind[] = ['simple', 'convergent', 'shared'];
const SEGMENT = 32;
const ACCESSES: Record<Kind, AccessName[]> = {
  simple: ['own', 'partner', 'write'],
  convergent: ['own', 'partner', 'write'],
  shared: ['load-lo', 'load-hi', 'store'],
};
/** How the two read styles are drawn: the first access of a pair filled, the second outlined. */
const FIRST: AccessName[] = ['own', 'load-lo'];

/** The aligned pieces of 32 floats, one row each, with the elements one warp touches in one step marked. */
function SegmentRows({ step, warp, N, kind }: { step: RequestStep; warp: number; N: number; kind: Kind }) {
  const w = step.warps.find((x) => x.warp === warp);
  const marks = new Map<number, Set<AccessName>>();
  for (const a of w?.accesses ?? []) if (a.array === 'input') for (const c of a.cells) marks.set(c, new Set([...(marks.get(c) ?? []), a.name]));
  const total = Math.ceil(N / SEGMENT);
  const touched = [...new Set((w?.accesses ?? []).filter((a) => a.array === 'input').flatMap((a) => a.segments))].sort((a, b) => a - b);
  const rows = total <= 8 ? Array.from({ length: total }, (_, i) => i) : touched;
  const cw = 12;
  const rh = 18;
  const left = 92;
  const W = left + SEGMENT * cw + 4;
  const H = Math.max(1, rows.length) * (rh + 4) + 2;
  if (!rows.length)
    return <p className="muted" data-part="segments" data-rows="0">This warp touches no element of the input in this step.</p>;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Pieces of 32 floats of the input, with the elements this warp touches marked"
      data-part="segments" data-rows={rows.length} style={{ width: W, maxWidth: '100%', minWidth: Math.min(W, 420), display: 'block' }}>
      {rows.map((seg, r) => {
        const hit = touched.includes(seg);
        return (
          <g key={seg} data-segment={seg} data-hit={hit ? '1' : '0'}>
            <text x={left - 6} y={r * (rh + 4) + rh - 5} fontSize="10" textAnchor="end" style={{ fill: hit ? 'var(--ink)' : 'var(--ink-3)', fontWeight: hit ? 650 : 400 }}>
              {seg * SEGMENT}–{seg * SEGMENT + SEGMENT - 1}
            </text>
            <rect x={left - 1} y={r * (rh + 4)} width={SEGMENT * cw + 1} height={rh} fill="none" stroke={hit ? 'var(--global)' : 'var(--rule)'} strokeWidth={hit ? 1.5 : 1} rx="2" />
            {Array.from({ length: SEGMENT }, (_, i) => {
              const c = seg * SEGMENT + i;
              if (c >= N) return null;
              const m = marks.get(c);
              const first = m && [...m].some((x) => FIRST.includes(x));
              const second = m && [...m].some((x) => !FIRST.includes(x) && x !== 'write');
              return (
                <rect key={i} x={left + i * cw + 1} y={r * (rh + 4) + 3} width={cw - 2} height={rh - 6} rx="1" data-cell={c}
                  data-mark={m ? [...m].join(' ') : ''}
                  fill={first ? 'var(--global)' : second ? 'var(--global-wash)' : 'var(--cell)'}
                  stroke={second ? 'var(--global)' : 'var(--rule)'} strokeWidth={second ? 1.4 : 0.5} />
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

function SegmentExplorer() {
  const { reports } = useReports();
  const choices = useMemo(() => widthChoices(reports), [reports]);
  const [kind, setKind] = useState<Kind>('simple');
  const [N, setN] = useState(256);
  const [widthKey, setWidthKey] = useState('book');
  const [stepIx, setStepIx] = useState(0);
  const [warp, setWarp] = useState(0);
  const width = (choices.find((c) => c.key === widthKey) ?? choices[0]).width;
  const trace = useMemo(() => requests(kind, N, width), [kind, N, width]);
  const k = Math.min(stepIx, trace.steps.length - 1);
  const step = trace.steps[k];
  const warpCount = Math.ceil(N / 2 / width);
  const wsel = Math.min(warp, warpCount - 1);
  const mine = step.warps.find((x) => x.warp === wsel);
  const names = ACCESSES[kind];
  const book = width === 32 && N >= 64 ? (kind === 'simple' ? bookFormula.simpleRequests(N) : kind === 'shared' ? bookFormula.sharedRequests(N) : null) : null;
  const reset = () => { setStepIx(0); setWarp(0); };

  return (
    <Bench
      title="Which pieces does a warp touch?"
      task="Pick a warp and step through the loop: count the pieces of 32 floats it touches in each access."
      foot={
        <>
          Each row is an aligned piece of 32 floats (128 bytes) of the input. Filled cells are the elements the chosen
          warp's active lanes read first ({kind === 'shared' ? 'input[t]' : 'their own element, which they also write'}); outlined cells
          are the second read ({kind === 'shared' ? 'input[t + blockDim.x]' : 'the partner'}). A request is one piece touched in one access. With
          more than 8 pieces in the input, only the ones this warp touches are drawn.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span style={FIELD}>
          Kernel
          <span className="seg" role="group" aria-label="Kernel">
            {KINDS.map((x) => <button key={x} aria-pressed={kind === x} onClick={() => { setKind(x); reset(); }}>{x === 'shared' ? 'shared memory' : x}</button>)}
          </span>
        </span>
        <span style={FIELD}>
          N (values)
          <span className="seg" role="group" aria-label="N">
            {SIZES.map((x) => <button key={x} aria-pressed={N === x} onClick={() => { setN(x); reset(); }}>{x}</button>)}
          </span>
        </span>
        <label className="field" style={{ maxWidth: '100%', minWidth: 0 }}>
          Warp width
          <select value={widthKey} onChange={(e) => { setWidthKey(e.target.value); reset(); }} aria-label="Warp width">
            {choices.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <button onClick={() => setStepIx(Math.max(0, k - 1))} disabled={k === 0}>Back</button>
        <button className="primary" onClick={() => setStepIx(Math.min(trace.steps.length - 1, k + 1))} disabled={k >= trace.steps.length - 1}>Step</button>
        <label className="field" style={{ flexDirection: 'row', alignItems: 'center', gap: '0.5rem' }}>
          step
          <input type="range" min={0} max={trace.steps.length - 1} value={k} onChange={(e) => setStepIx(Number(e.target.value))} aria-label="Step" />
        </label>
        <label className="field">
          Warp
          <select value={wsel} onChange={(e) => setWarp(Number(e.target.value))} aria-label="Warp">
            {Array.from({ length: warpCount }, (_, i) => <option key={i} value={i}>warp {i} (threads {i * width} to {Math.min(N / 2, (i + 1) * width) - 1})</option>)}
          </select>
        </label>
      </div>
      <div className="narrate" role="status" data-readout="warp" data-requests={mine?.requests ?? 0} style={{ marginBottom: '0.9rem' }}>
        <b>{KIND_LABEL[kind]}, {step.label}.</b>{' '}
        {step.phase === 'iteration' && kind === 'shared' ? (
          <>This iteration reads and writes shared memory only: no warp makes a global-memory request.</>
        ) : !mine ? (
          <>Warp {wsel} has no active lane in this step, so it makes no request.</>
        ) : (
          <>
            Warp {wsel}:{' '}
            {mine.accesses.map((a, i) => (
              <span key={a.name}>
                {i ? '; ' : ''}
                {ACCESS_LABEL[a.name]}, {a.cells.length} element{a.cells.length === 1 ? '' : 's'} in {a.segments.length} piece{a.segments.length === 1 ? '' : 's'}
              </span>
            ))}
            . <b>{mine.requests} requests.</b>
          </>
        )}
      </div>
      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 6fr) minmax(0, 5fr)' }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ overflowX: 'auto' }}>
            <SegmentRows step={step} warp={wsel} N={N} kind={kind} />
          </div>
          {mine?.accesses.some((a) => a.array === 'output') ? <p className="muted" style={{ margin: '0.5rem 0 0' }}>Thread 0 writes <span className="mono">*output</span>: one more request, to the output array.</p> : null}
          {Math.ceil(N / SEGMENT) > 8 && mine ? (
            <p className="muted" style={{ margin: '0.5rem 0 0' }}>The input has {Math.ceil(N / SEGMENT)} pieces; the others are not touched by this warp in this step.</p>
          ) : null}
        </div>
        <div style={{ minWidth: 0, overflowX: 'auto' }}>
          <table className="threads" aria-label="Requests per step" data-part="request-table">
            <thead>
              <tr>
                <th>step</th><th>warps</th>
                {names.map((a) => <th key={a}>{a === 'own' ? 'own' : a === 'partner' ? 'partner' : a === 'write' ? 'write' : a === 'load-lo' ? 'in[t]' : a === 'load-hi' ? 'in[t+B]' : 'store'}</th>)}
                <th>total</th>
              </tr>
            </thead>
            <tbody>
              {trace.steps.map((s, i) => (
                <tr key={i} className={i === k ? 'cur' : ''}>
                  <td>{s.phase === 'iteration' ? `${s.n} (stride ${s.stride})` : s.phase === 'load' ? 'load' : 'store'}</td>
                  <td>{s.warps.length}</td>
                  {names.map((a) => <td key={a}>{s.byAccess[a] ?? 0}</td>)}
                  <td>{s.requests}</td>
                </tr>
              ))}
              <tr>
                <td><b>Total</b></td><td /><td colSpan={names.length} /><td data-readout="total"><b>{trace.total}</b></td>
              </tr>
            </tbody>
          </table>
          {book != null ? (
            <p className="muted" style={{ margin: '0.5rem 0 0' }} data-readout="book">
              The book's formula ({kind === 'shared' ? 'N/32 + 1, pp. 236–237' : 'p. 235'}) gives {book}.
            </p>
          ) : null}
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS105: Record<string, Question> = {
  'ch10/10.5/q-half-used': {
    kind: 'choice',
    id: 'ch10/10.5/q-half-used',
    prompt: <>In the simple kernel's first iteration, what fraction of the bytes delivered for a warp's own-element reads does the warp use?</>,
    options: [
      { label: 'All of them.', why: 'The 32 elements the warp reads are 2 apart, so they span 64 floats.' },
      { label: 'Half of them.', correct: true },
      { label: 'A quarter of them.', why: 'That is the second iteration, when the active lanes\' elements are 4 apart.' },
      { label: 'One in 32.', why: 'That comes later, once one lane per warp is active.' },
    ],
    explain: (
      <p>
        Thread t of the warp reads element 2t, so the warp's 32 reads cover elements 0, 2, ..., 62: two pieces of 32
        floats, of which every second float is used. Twice the transactions of a coalesced read, half the data used. In
        the second iteration the active lanes' elements are 4 apart, and it gets worse until one lane is left.
      </p>
    ),
    source: 'ch. 10 §10.5, pp. 234–235',
  },
  'ch10/10.5/q-req-2048': {
    kind: 'numeric',
    id: 'ch10/10.5/q-req-2048',
    prompt: <>The simple kernel on 2048 values with 32-wide warps. How many global-memory requests does it make, by the book's formula?</>,
    answer: 1149,
    explain: (
      <p>
        N/64 = 32 warps. Five iterations in which every warp makes two requests per access: 32 × 5 × 2. Then one request
        per access for each warp that still has an active lane: 32 + 16 + 8 + 4 + 2 + 1. Three accesses each:
        (320 + 63) × 3 = 1149. The convergent kernel needs 204, five and a half times fewer, and the shared-memory kernel of
        section 10.6 needs 65.
      </p>
    ),
    source: 'ch. 10 §10.5, p. 235 (the formula); the 2048 case is ours',
  },
  'ch10/10.5/q-softmax-loop': {
    kind: 'choice',
    id: 'ch10/10.5/q-softmax-loop',
    prompt: (
      <>
        A block of 256 threads walks a row of 32,000 logits to find its max. Thread t reads either (A){' '}
        <code>x[t + k*256]</code> for k = 0, 1, 2, ..., or (B) <code>x[t*125 + k]</code> for k = 0 to 124. Which coalesces?
      </>
    ),
    options: [
      { label: 'A', correct: true },
      { label: 'B', why: 'In each step k, a warp\'s 32 reads are 125 floats apart: 32 separate pieces for 32 floats.' },
      { label: 'Both: each thread reads 125 consecutive values in B, which a cache will catch.', why: 'Coalescing is about what the 32 lanes of a warp read in the same step, not what one thread reads over time.' },
    ],
    explain: (
      <p>
        The rule is the convergent kernel's: in each step, consecutive threads must read consecutive elements. In A, step k
        has the warp read 32 neighbouring logits, one piece. In B, thread t starts at 125t, so in any one step the warp's
        lanes are 125 floats apart. A is the interleaved partitioning of chapter 9 (§9.5), and the stride-by-block loop
        that section 10.10 uses to coarsen.
      </p>
    ),
    source: 'ch. 6 §6.1; ch. 9 §9.5; ch. 10 §10.10, Fig. 10.20',
  },
};

// ---------------------------------------------------------------- the page
export function S105() {
  const c256 = requests('convergent', 256, 32);
  const warpIters = c256.steps.reduce((a, s) => a + s.warps.length, 0);
  return (
    <>
      <p>
        The simple kernel has a second problem, and it comes from the same choice of who owns what. In every iteration each
        active thread makes two reads of global memory (its own element, and the partner one stride away) and one write
        (its own element). Chapter 6 said what makes those cheap: the lanes of a warp reading neighbouring elements in
        the same instruction, so that the hardware combines them (<a href={href('ch06', '6.1')}>section 6.1</a>). Here
        neighbouring threads own elements 2 apart.
      </p>
      <NoteGroup>
        <Source section="10.5" pages="234–235" />
        <Aside>
          <p>On p. 234 the book credits coalescing to chapter 5; this edition teaches it in chapter 6.</p>
        </Aside>
      </NoteGroup>
      <p>
        So in the first iteration a warp's reads span twice as many pieces of memory as a coalesced read would, and half the
        data that arrives is not used. In the second iteration the active lanes' elements are 4 apart: four times the
        transactions, a quarter used. It keeps getting worse as the stride grows, until each warp has one active lane and
        makes one request per access.
      </p>
      <Source section="10.5" pages="235" />
      <Ask q={questionsS105['ch10/10.5/q-half-used']} />

      <h2>Counting requests</h2>
      <p>
        The book counts the simple kernel's global-memory requests for 256 values as (N/64 × 5 × 2 + N/64 × 1 + N/64 ×
        1/2 + ... + 1) × 3. The first term covers five iterations in which every warp has at least two active threads and
        its scattered accesses need two requests; after that each warp with one active thread needs one, and their number
        halves every iteration; × 3 for the two reads and the write. For N = 256 that is (40 + 4 + 2 + 1) × 3 = 141.
      </p>
      <Source section="10.5" pages="235" />
      <p>
        The book does not say what a request is. To draw it, we use one rule: in one access, a warp makes one request per
        aligned piece of 32 floats (128 bytes) that its active lanes touch. That rule reproduces all three of the book's
        counts, 141 here, 36 for the convergent kernel and 9 for section 10.6's.
      </p>
      <Beyond title="Ours, not the book's">
        <p>
          Real memory systems differ in transaction sizes and caches (chapter 6). As in the book, the in-place kernels'
          last read of <code>input[0]</code> and write of <code>*output</code> are left out, and the shared-memory
          kernel's one write is counted.
        </p>
      </Beyond>
      <Predict
        id="ch10/10.5/p-requests"
        prompt={<>Before you select the convergent kernel below: how many global-memory requests does it make for 256 values?</>}
        unit="requests"
      >
        {(v) => (
          <p data-readout="p-requests">
            The model counts {c256.total}: {warpIters} warp-iterations, each with three accesses that touch one piece apiece.{' '}
            {v === c256.total ? 'Exactly what you predicted.' : v > c256.total ? 'Fewer than you predicted.' : 'More than you predicted.'} The book
            gives the same 36 on p. 237, without the arithmetic.
          </p>
        )}
      </Predict>
      <Simplification>
        <p>
          The request rule is ours (see the note above). The model knows nothing about caches, timing or transaction sizes,
          and neither do the course's CPU emulators, which do not model coalescing at all; lab 10.1 measures the kernels on
          your device.
        </p>
      </Simplification>
      <SegmentExplorer />

      <h2>The convergent kernel coalesces</h2>
      <p>
        In the convergent kernel neighbouring threads always touch neighbouring elements: own elements t, partners t +
        stride. Every access of an active warp falls in one piece, and a warp with no active thread makes no request at
        all. The only waste left is when fewer active lanes than a whole piece's worth ask for data, in the last
        iterations. The book gives 36 requests for 256 values against the simple kernel's 141.
      </p>
      <Source section="10.5" pages="235, 237" />
      <p>
        So the three changed lines of section 10.4 win twice: fewer execution slots, from less divergence, and fewer DRAM
        requests, from coalesced accesses. DRAM bandwidth is limited, so the book expects the convergent kernel to be
        significantly better on this count too.
      </p>
      <NoteGroup>
        <Source section="10.5" pages="235" />
        <Aside>
          <p>
            On p. 235, "the kernel in Fig. 10.7" (twice) means the kernel of Fig. 10.8; Fig. 10.7 is its diagram.
          </p>
        </Aside>
        <Today checked="2026-11-20 (notes/today/infra-facts.md C13)">
          <p>
            Apple documents no cache size, cache-line size or memory bandwidth figure for its GPUs, only that buffer reads
            go through a cache of limited size. On a Mac, the difference between these kernels is something to measure.
          </p>
        </Today>
      </NoteGroup>
      <Ask q={questionsS105['ch10/10.5/q-req-2048']} label="Work it out" />

      <h2>The same rule in a softmax</h2>
      <p>
        The rule you just counted decides more than reductions. A softmax over a row of logits starts with a max over the
        row: each thread of a block walks part of the row, and what matters for memory is what a warp's lanes read in the
        same step. Consecutive threads must read consecutive logits. That is the interleaved partitioning of{' '}
        <a href={href('ch09', '9.5')}>section 9.5</a>, and the loop section 10.10 uses to coarsen.
      </p>
      <Source section="10.10" pages="247" figs="Fig. 10.20" />
      <Ask q={questionsS105['ch10/10.5/q-softmax-loop']} />
    </>
  );
}
