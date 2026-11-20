import { useEffect, useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  FIG42,
  capacity,
  completionOrder,
  equalDurations,
  schedule,
  stateAt,
  unevenDurations,
  waitingAtLaunch,
  type SchedEvent,
  type Schedule,
} from '../../sim/blocksched';

// The grid of chapter 3's 62 x 76 picture: 4 rows of 5 blocks of 16 x 16 threads (ch. 3, p. 49).
const PICTURE_BLOCKS = 20;
const MAX_BLOCKS = 24;
const clampInt = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, Math.round(Number.isFinite(v) ? v : lo)));

/** One sentence for the event that has just happened. */
function describe(s: Schedule, n: number): string {
  const e: SchedEvent = s.events[n - 1];
  const prev = n >= 2 ? s.events[n - 2] : undefined;
  if (e.kind === 'complete') return `Time ${e.time}: block ${e.block} finishes on SM ${e.unit} and gives up its slot.`;
  if (prev && prev.kind === 'complete' && prev.unit === e.unit && prev.slot === e.slot)
    return `Time ${e.time}: block ${e.block}, next on the list, takes the slot on SM ${e.unit} that block ${prev.block} has just left. All of its threads go there together.`;
  return `Time ${e.time}: block ${e.block} is assigned to SM ${e.unit}, slot ${e.slot}. All of its threads go there together.`;
}

/** A strip per block: when it ran, and on which SM. */
function Strips({ s, n, now }: { s: Schedule; n: number; now: number }) {
  const blocks = s.spec.blocks;
  const started = new Set(s.events.slice(0, n).filter((e) => e.kind === 'assign').map((e) => e.block));
  const W = 560;
  const L = 34;
  const row = 13;
  const H = blocks * row + 22;
  const span = Math.max(1, s.makespan);
  const x = (t: number) => L + (t / span) * (W - L - 8);
  const ticks = Array.from({ length: Math.floor(span) + 1 }, (_, i) => i).filter((t) => span <= 12 || t % Math.ceil(span / 12) === 0);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="When each block runs" style={{ width: '100%', maxWidth: W, display: 'block' }}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={0} y2={blocks * row} stroke="var(--rule)" />
          <text x={x(t)} y={blocks * row + 12} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{t}</text>
        </g>
      ))}
      <text x={W - 8} y={blocks * row + 21} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>time</text>
      {s.placements.map((p) => {
        const y = p.block * row;
        const on = started.has(p.block);
        const upto = Math.min(p.end, Math.max(p.start, now));
        return (
          <g key={p.block} data-strip={p.block} data-on={on ? '1' : '0'}>
            <text x={L - 5} y={y + 9.5} fontSize="9.5" textAnchor="end" style={{ fill: on ? 'var(--ink)' : 'var(--ink-3)' }}>{p.block}</text>
            {on ? (
              <>
                <rect x={x(p.start)} y={y + 1.5} width={x(p.end) - x(p.start) - 1} height={row - 3} rx="2" fill="var(--cell)" stroke="var(--ink-3)" />
                <rect x={x(p.start)} y={y + 1.5} width={Math.max(0, x(upto) - x(p.start) - 1)} height={row - 3} rx="2" fill="var(--ink-2)" />
                <text x={x(p.start) + 4} y={y + 9.5} fontSize="8.5" style={{ fill: upto - p.start >= 0.5 ? 'var(--page)' : 'var(--ink-2)' }}>SM {p.unit}</text>
              </>
            ) : null}
          </g>
        );
      })}
      <line x1={x(now)} x2={x(now)} y1={0} y2={blocks * row} stroke="var(--ink)" strokeWidth="1.4" />
    </svg>
  );
}

function BlockStepper() {
  const [units, setUnits] = useState<number>(FIG42.units);
  const [slots, setSlots] = useState<number>(FIG42.slotsPerUnit);
  const [blocks, setBlocks] = useState(PICTURE_BLOCKS);
  const [uneven, setUneven] = useState(false);
  const [n, setN] = useState(0);
  const [playing, setPlaying] = useState(false);

  const s = useMemo(
    () => schedule({ blocks, units, slotsPerUnit: slots, durations: uneven ? unevenDurations(blocks) : equalDurations(blocks) }),
    [blocks, units, slots, uneven],
  );
  const total = s.events.length;
  const at = Math.min(n, total);
  const st = useMemo(() => stateAt(s, at), [s, at]);
  const filled = Math.min(blocks, s.capacity); // events that fill the device at launch
  const order = completionOrder(s);
  const inLaunchOrder = order.every((b, i) => b === i);

  useEffect(() => { setN(0); setPlaying(false); }, [blocks, units, slots, uneven]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) { setPlaying(false); return; }
    const id = window.setTimeout(() => setN((x) => x + 1), 380);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);
  const go = (k: number) => { setPlaying(false); setN(Math.max(0, Math.min(total, k))); };

  return (
    <Bench
      title="Blocks onto SMs"
      task="Step through the launch. Can you ever get more blocks running than SMs × slots, or one block on two SMs?"
      foot="Block times are in arbitrary units. With uneven times each block takes 1, 2, 3 or 4 units, fixed by a seed so that the run repeats."
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          SMs
          <select value={units} onChange={(e) => setUnits(+e.target.value)} aria-label="SMs">
            {[1, 2, 3, 4].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className="field">
          Blocks an SM holds at once
          <select value={slots} onChange={(e) => setSlots(+e.target.value)} aria-label="Blocks an SM holds at once">
            {[1, 2, 3, 4].map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className="field">
          Blocks in the grid
          <input type="number" min={1} max={MAX_BLOCKS} step={1} value={blocks} aria-label="Blocks in the grid"
            onChange={(e) => setBlocks(clampInt(+e.target.value, 1, MAX_BLOCKS))} />
        </label>
        <span className="seg" role="group" aria-label="Block times">
          <button aria-pressed={!uneven} onClick={() => setUneven(false)}>equal block times</button>
          <button aria-pressed={uneven} onClick={() => setUneven(true)}>uneven block times</button>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => go(0)} disabled={at === 0}>Restart</button>
        <button onClick={() => go(at - 1)} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => go(at + 1)} disabled={at >= total}>Step</button>
        <button onClick={() => go(filled)} disabled={at >= filled}>Fill the device</button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? 'Pause' : 'Play'}</button>
        <button onClick={() => go(total)} disabled={at >= total}>Run to the end</button>
        <span className="muted" aria-live="polite">event {at} of {total}</span>
      </div>
      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }}>
        {at === 0 ? (
          <>
            Nothing is assigned yet. The grid has {blocks} blocks. The device has {units} SM{units > 1 ? 's' : ''} with
            room for {slots} of these blocks each, so {s.capacity} can be in progress at once. Press Step.
          </>
        ) : (
          describe(s, at)
        )}
        {at >= total ? (
          <>
            {' '}Every block has run. They finished {inLaunchOrder ? 'in the order they were launched' : 'in a different order from the one they were launched in'}.
          </>
        ) : null}
      </div>
      <p className="counts" style={{ margin: '0 0 0.9rem' }}>
        <span className="chip" data-k="waiting">{st.pending.length} waiting</span>{' '}
        <span className="chip" data-k="resident">{st.resident} of {s.capacity} slots in use</span>{' '}
        <span className="chip" data-k="done">{st.done.length} finished</span>
      </p>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 5fr) minmax(0, 6fr)' }}>
        <div>
          <div className="mx-label">Not started (the runtime's list, next first)</div>
          <div className="row pending" style={{ gap: '0.3rem', minHeight: '1.9rem', marginBottom: '0.9rem' }}>
            {st.pending.length ? st.pending.map((b) => <span key={b} className="chip">{b}</span>) : <span className="muted">empty</span>}
          </div>
          <div className="row" style={{ alignItems: 'stretch', gap: '0.7rem', marginBottom: '0.9rem' }}>
            {st.slots.map((unit, u) => (
              <div key={u} className="panel sm" data-sm={u} style={{ padding: '0.5rem 0.6rem', minWidth: '6.4rem' }}>
                <div className="mx-label" style={{ marginBottom: '0.35rem' }}>SM {u}</div>
                {unit.map((b, k) => (
                  <div key={k} data-slot={k} data-block={b ?? ''}
                    style={{
                      border: b == null ? '1px dashed var(--ink-3)' : '1.5px solid var(--ink)',
                      borderRadius: 3, padding: '0.2rem 0.4rem', marginBottom: '0.3rem', minHeight: '2.5rem',
                      background: b == null ? 'transparent' : 'var(--cell)', color: b == null ? 'var(--ink-3)' : 'var(--ink)',
                    }}>
                    {b == null ? 'free' : (
                      <>
                        <b>block {b}</b>
                        <div className="muted" style={{ fontSize: '0.7rem', letterSpacing: '0.04em' }}>t0 t1 t2 … all here</div>
                      </>
                    )}
                  </div>
                ))}
              </div>
            ))}
          </div>
          <div className="mx-label">Finished, in the order they finished</div>
          <div className="row finished" style={{ gap: '0.3rem', minHeight: '1.9rem' }}>
            {st.done.length ? st.done.map((b) => <span key={b} className="chip ok">{b}</span>) : <span className="muted">none yet</span>}
          </div>
        </div>
        <div>
          <div className="mx-label">One strip per block: when it runs, and where</div>
          <Strips s={s} n={at} now={st.time} />
        </div>
      </div>
    </Bench>
  );
}

export const questionsS42: Record<string, Question> = {
  'ch04/4.2/q-crossblock': {
    kind: 'choice',
    id: 'ch04/4.2/q-crossblock',
    prompt: (
      <>
        Each block of a kernel writes the sum of its own slice to <code>out[blockIdx.x]</code>. Then, in the same
        kernel, block <code>b</code> reads <code>out[b - 1]</code> to turn the sums into a running total. The test
        passes on the author's GPU. Verdict?
      </>
    ),
    options: [
      { label: 'Correct: blocks are handed out in index order, so block b − 1 has written before block b reads.', why: 'The list may be handed out in index order and block b can still reach its read first, or run at the same moment on another SM.' },
      {
        label: 'Wrong: nothing orders blocks, so block b may read out[b − 1] before block b − 1 has written it. The running total has to be a separate step.',
        correct: true,
      },
      { label: 'Wrong only on a device with several SMs; on a single SM the blocks run one after another.', why: 'One SM holds several blocks at once, and no vendor promises an order even in series.' },
      { label: 'Correct once a __syncthreads() is placed before the read.', why: 'That barrier stops at the edge of the block. It cannot make one block wait for another.' },
    ],
    explain: (
      <p>
        A block is assigned whenever some SM has room, and it runs beside whatever else is resident. So there is no
        moment at which "block b − 1 is done" is true for block b. The test passes because of how one device happened to
        place the blocks that time. The fix is a second step that starts after every sum is written, for example a
        second kernel launched afterwards, or one of the scan algorithms of chapter 11, which exist for exactly this.
      </p>
    ),
    source: 'ch. 4 §4.2, p. 69 and §4.3, p. 73; NVIDIA\'s wording in notes/today/ch04-facts.md A5',
  },
  'ch04/4.2/q-waiting': {
    kind: 'numeric',
    id: 'ch04/4.2/q-waiting',
    prompt: (
      <>
        A device has 40 SMs. For your kernel and block size, each SM has room for 6 blocks. You launch a grid of 1000
        blocks. How many blocks are still waiting on the runtime's list just after launch?
      </>
    ),
    answer: waitingAtLaunch(1000, capacity(40, 6)),
    unit: 'blocks',
    explain: (
      <p>
        {capacity(40, 6)} blocks can be in progress at once (40 × 6). The other 1000 − {capacity(40, 6)} ={' '}
        {waitingAtLaunch(1000, capacity(40, 6))} wait, and each of them starts when some earlier block ends. The number
        6 is not a property of the device alone: it depends on what your kernel asks of an SM, which is section 4.7.
      </p>
    ),
    source: 'ch. 4 §4.2, p. 69 (the device and grid are ours)',
  },
  'ch04/4.2/q-whole': {
    kind: 'choice',
    id: 'ch04/4.2/q-whole',
    prompt: <>Why does the runtime never split one block across two SMs?</>,
    options: [
      { label: 'A block is one warp, and a warp cannot be divided.', why: 'A block is usually many warps. Section 4.4 cuts it up.' },
      {
        label: 'Its threads may meet at a barrier and share on-chip memory. Both live inside one SM, and the runtime secures room for all the threads before any of them starts.',
        correct: true,
      },
      { label: 'It could, but moving threads between SMs is slow, so it avoids it when it can.', why: 'It is a guarantee of the model, not a preference of the scheduler.' },
      { label: 'A block has at most 1024 threads and an SM holds more than that, so the question never comes up.', why: 'A smaller device, or a kernel that needs more per thread, would make it come up at once.' },
    ],
    explain: (
      <p>
        Whole-block assignment is what the two same-block services rest on. Shared memory is on-chip memory of one SM,
        so the threads that share it must be there. A barrier needs every thread of the block to be able to arrive, so
        all of them must have their resources before the first one runs. Split a block and neither can be offered.
      </p>
    ),
    source: 'ch. 4 §4.2, p. 69 and §4.3, pp. 72–73',
  },
};

export function S42() {
  const q = questionsS42;
  const cap = capacity(FIG42.units, FIG42.slotsPerUnit);
  const waiting = waitingAtLaunch(PICTURE_BLOCKS, cap);
  return (
    <>
      <p>
        A launch can ask for far more than any device has. Chapter 3's 1500 × 2000 picture needed 11,750 blocks of 256
        threads, about three million threads from one statement. No device runs that many at once. So the runtime does
        not start a grid; it starts as much of it as fits, and hands out the rest as hardware comes free.
      </p>
      <Source section="4.2" pages="69" />
      <p>
        The unit it hands out is the <b>block</b>. All threads of a block are assigned to the same SM at the same time.
        An SM can take several blocks at once, as many as its resources allow for this kernel; the book's picture shows
        three on each of two SMs.
      </p>
      <Source section="4.2" pages="69" figs="Fig. 4.2, redrawn in the stepper below" />
      <p>
        So a device has a ceiling on blocks in progress: the number of SMs times the blocks each SM holds. A grid
        usually has many more. The runtime keeps a list of the blocks that have not started and assigns the next one
        whenever a block that was running finishes.
      </p>
      <Aside>
        <p>
          <b>Apparent erratum.</b> The book says the limits on blocks per SM are discussed in "Section 4.6". They are
          in <a href={href('ch04', '4.7')}>section 4.7</a>.
        </p>
      </Aside>
      <Predict
        id="ch04/4.2/p-pending"
        prompt={
          <>
            Chapter 3's 62 × 76 picture was covered by a grid of {PICTURE_BLOCKS} blocks. Launch it on the toy device of
            the book's Fig. 4.2: {FIG42.units} SMs, each with room for {FIG42.slotsPerUnit} blocks. How many blocks are
            waiting right after the launch?
          </>
        }
        unit="blocks"
      >
        {(v) => (
          <p>
            <b>{waiting}</b>. The device runs {FIG42.units} × {FIG42.slotsPerUnit} = {cap} blocks at once, so{' '}
            {PICTURE_BLOCKS} − {cap} = {waiting} wait.{' '}
            {v === waiting ? 'You had it.' : v === cap ? `${cap} is the number that start; the question was how many do not.` : 'Step through it below.'}{' '}
            The stepper starts on exactly this launch.
          </p>
        )}
      </Predict>
      <Simplification>
        <p>
          A block here is an atom that takes one slot. "Slots per SM" stands for whichever real limit binds
          (section 4.7). Which free slot a block gets is our choice: no vendor documents the policy. Durations are
          made up; nothing here predicts time.
        </p>
      </Simplification>
      <BlockStepper />
      <p>
        Three things to take from it. The count of blocks in progress never passes SMs × slots, however long the list.
        A block is in one slot of one SM from its first step to its last. And once blocks take different times, the
        order in which they finish is no longer the order in which they started, so nothing later in your program may
        depend on either.
      </p>
      <Ask q={q['ch04/4.2/q-waiting']} label="Work it out" />

      <h2>What a block gets for being assigned whole</h2>
      <p>
        Because the threads of a block are on one SM at one time, they can do two things that threads of different
        blocks cannot. They can wait for each other at a <b>barrier</b>, which is{' '}
        <a href={href('ch04', '4.3')}>the next section</a>. And they can exchange data through <b>shared memory</b>, a
        fast memory inside the SM, which is chapter 5. Neither exists across blocks, for the reason the stepper shows:
        two blocks of one grid may be on different SMs, or not running at the same time at all.
      </p>
      <Source section="4.2" pages="69" />
      <Ask q={q['ch04/4.2/q-whole']} />
      <Today checked="2026-11-20 (notes/today/ch04-facts.md, A5)">
        <p>
          NVIDIA's guide says a block runs to completion on its SM "in most cases": with dynamic parallelism a block can
          be suspended to memory. It also says that which block runs on which SM can be neither controlled nor queried,
          and that no order is promised.
        </p>
      </Today>
      <Ask q={q['ch04/4.2/q-crossblock']} />

      <h2>Thread block clusters</h2>
      <p>
        With the Hopper generation CUDA gained an optional level between block and grid: the <b>cluster</b>, a group of
        blocks. What an SM is to a block, a GPC is to a cluster: the blocks of a cluster are scheduled together on one
        GPC. In return their threads can meet at a cluster-wide barrier and can reach each other's shared memory, which
        the book calls distributed shared memory.
      </p>
      <Source section="4.2" pages="69–70" />
      <p>
        The book describes clusters in prose only here and gives no code; chapter 5 comes back to them. For this
        chapter it is enough to know that the level exists and that it is optional.
      </p>
      <Beyond>
        <p>
          On CUDA, clusters need compute capability 9.0 or higher. HIP 7.17 documents a <code>cluster_group</code> and a
          per-device <code>clusterLaunch</code> flag; which AMD devices set it is unverified. Metal has no such level
          (notes/today/ch04-facts.md, A7, B11, C3).
        </p>
      </Beyond>

      <h2>On other backends</h2>
      <p>
        AMD documents the same guarantee and the same absence of one for HIP: all warps of a block execute on the same
        compute unit, and neither the mapping of blocks to compute units nor their order is defined. Apple documents
        threadgroups, their barrier and their memory, but no unit that a threadgroup is placed on, and we found no
        statement about the order in which threadgroups run. Treat order as unspecified there too.
      </p>
      <Beyond>
        <p>
          Not in the book. AMD's wording is in notes/today/ch04-facts.md, B2. The Metal sentence reports an absence in
          the documents we read (C1, C2): unverified.
        </p>
      </Beyond>
    </>
  );
}
