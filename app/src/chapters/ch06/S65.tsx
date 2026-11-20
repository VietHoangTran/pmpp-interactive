import { useMemo, useState } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { fmt, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { warpAccess } from '../../sim/coalesce';
import {
  BOOK_CAPACITY,
  PITFALLS,
  SECTOR_BYTES,
  VERDICTS,
  calculate,
  deviceSlots,
  elementsOf,
  residentUpperBound,
  warpInstruction,
  warpVectorInstruction,
  type Mapping,
  type Verdict,
} from '../../sim/coarsen';

// labs/ch06/snippets/s65_coarsen_toy.cu, which ./verify.sh front-end-checks.
// snippets-s65.test.ts fails if this copy drifts from the file.
const TOY = `// Section 6.5, pp. 141-142: the book's thread-coarsening toy, made compilable.
// foo stands for one unit of work (the book leaves it abstract); here it doubles
// one element. The bounds checks are ours: the book's toy has none.
__device__ void foo(float* data, unsigned int i) { data[i] = 2.0f*data[i]; }

// Before: one unit of work per thread. Launch one thread per element.
__global__ void fine(float* data, unsigned int n) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) foo(data, i);
}

// After: coarsening factor 4. Launch a quarter as many threads.
#define COARSEN 4
__global__ void coarse(float* data, unsigned int n) {
    unsigned int iStart = COARSEN*(blockIdx.x*blockDim.x + threadIdx.x);
    for (unsigned int c = 0; c < COARSEN; ++c) {   // the coarsening loop
        unsigned int i = iStart + c;
        if (i < n) foo(data, i);
    }
}

// Not in the book: the same factor with a strided mapping. Thread t handles
// t, t + T, t + 2T and t + 3T, where T is the number of threads in the grid.
__global__ void coarseStrided(float* data, unsigned int n) {
    unsigned int T = gridDim.x*blockDim.x;
    unsigned int t = blockIdx.x*blockDim.x + threadIdx.x;
    for (unsigned int c = 0; c < COARSEN; ++c) {
        unsigned int i = t + c*T;
        if (i < n) foo(data, i);
    }
}`;

const pct = (x: number) => `${fmt(100 * x)}%`;
const count = (n: number) => n.toLocaleString('en-US');

// ---------------------------------------------------------------- wave calculator
const TPB_OPTIONS = [64, 128, 256, 512, 1024];
type DeviceChoice = { key: string; label: string; capacity: number | null; note: string };

function WaveCalculator() {
  const { reports } = useReports();
  const [tpb, setTpb] = useState(256);
  const [units, setUnits] = useState(1056 * 256);
  const [factor, setFactor] = useState(1);
  const [dev, setDev] = useState('book');
  const [custom, setCustom] = useState(100);

  const devices: DeviceChoice[] = useMemo(() => {
    const list: DeviceChoice[] = [
      {
        key: 'book',
        label: "The book's hypothetical device (264)",
        capacity: BOOK_CAPACITY,
        note: "The book's example on p. 143: a device that runs 264 blocks at the same time. It names no device.",
      },
    ];
    for (const b of realBackends(reports).filter((x) => x === 'cuda' || x === 'hip' || x === 'metal')) {
      const d = reports[b].device;
      const slots = deviceSlots(d);
      const name = String(d.name ?? b);
      if (slots && (b === 'cuda' || b === 'hip'))
        list.push({
          key: b,
          label: `Yours: ${name} (${b})`,
          capacity: residentUpperBound(slots, tpb),
          note:
            `${slots.units} × min(${slots.blocksPerUnit} block slots, ${slots.threadsPerUnit || 'unreported'} thread slots ÷ ${tpb} threads) from this device's report. ` +
            'An upper bound: registers and shared memory per block can only lower it, and the occupancy API (chapter 4) gives the real figure for a real kernel.' +
            (b === 'hip' ? ' On HIP the block-slot field is the number of wave slots, not a limit of its own (ch04-facts B9).' : ''),
        });
      else
        list.push({
          key: b,
          label: `Yours: ${name} (${b})`,
          capacity: null,
          note: b === 'metal' ? 'Metal reports no number of blocks a device runs at once, so there is nothing to put here. Use a custom value.' : 'This report has no per-SM fields.',
        });
    }
    list.push({ key: 'custom', label: 'Custom', capacity: Math.max(1, Math.floor(custom)), note: 'A number you choose.' });
    return list;
  }, [reports, tpb, custom]);

  const device = devices.find((d) => d.key === dev) ?? devices[0];
  const capacity = device.capacity;
  const r = capacity ? calculate(units, tpb, factor, capacity) : null;
  const preset = (blocks: number) => {
    setTpb(256);
    setUnits(blocks * 256);
    setFactor(1);
    setDev('book');
  };

  const MAXROWS = 8;
  const rows = r ? r.fill.perWave : [];
  const shown = rows.length > MAXROWS ? [...rows.slice(0, MAXROWS - 1).map((n, i) => ({ n, w: i })), { n: rows[rows.length - 1], w: rows.length - 1 }] : rows.map((n, i) => ({ n, w: i }));
  const W = 560;
  const rowH = 30;

  return (
    <Bench
      title="Wave calculator"
      task="Reproduce the book's two cases, then find a factor that leaves only whole waves."
      foot="Each row is one wave: the blocks the device runs at the same time. Filled: a block is running in that slot. Dashed: the slot is idle."
    >
      <div className="row" style={{ marginBottom: '0.6rem' }}>
        <button onClick={() => preset(1056)}>The book's 1056 blocks</button>
        <button onClick={() => preset(1584)}>The book's 1584 blocks</button>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Work units
          <input type="number" min={1} value={units} aria-label="work units" style={{ width: '9rem' }}
            onChange={(e) => setUnits(Math.max(1, Math.floor(Number(e.target.value) || 1)))} />
        </label>
        <label className="field">
          Threads per block
          <select value={tpb} onChange={(e) => setTpb(+e.target.value)}>
            {TPB_OPTIONS.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </label>
        <label className="field">
          Coarsening factor
          <input type="range" min={1} max={64} value={factor} aria-label="coarsening factor slider" onChange={(e) => setFactor(+e.target.value)} />
          <input type="number" min={1} max={64} value={factor} aria-label="coarsening factor" style={{ width: '4.5rem' }}
            onChange={(e) => setFactor(Math.min(64, Math.max(1, Math.floor(Number(e.target.value) || 1))))} />
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Blocks the device runs at once
          <select value={device.key} onChange={(e) => setDev(e.target.value)} aria-label="device">
            {devices.map((d) => (
              <option key={d.key} value={d.key} disabled={d.capacity == null}>
                {d.label}{d.capacity == null ? ': no figure' : d.key !== 'book' && d.key !== 'custom' ? `: at most ${count(d.capacity)}` : ''}
              </option>
            ))}
          </select>
        </label>
        {device.key === 'custom' ? (
          <label className="field">
            Blocks at once
            <input type="number" min={1} value={custom} aria-label="custom capacity" style={{ width: '6rem' }}
              onChange={(e) => setCustom(Math.max(1, Math.floor(Number(e.target.value) || 1)))} />
          </label>
        ) : null}
      </div>
      <p className="muted ui" style={{ margin: '0 0 0.8rem' }}>{device.note}</p>
      {devices.filter((d) => d.capacity == null).map((d) => (
        <p key={d.key} className="muted ui" style={{ margin: '0 0 0.8rem' }} data-k="no-figure">
          {d.label}: {d.note}
        </p>
      ))}

      {r && capacity ? (
        <>
          <table className="data" style={{ background: 'var(--page)', margin: '0 0 0.8rem' }} aria-label="wave arithmetic">
            <tbody>
              <tr><td>threads</td><td data-k="threads">{count(r.threads)}</td><td className="muted">⌈{count(units)} ÷ {factor}⌉</td></tr>
              <tr><td>blocks</td><td data-k="blocks">{count(r.blocks)}</td><td className="muted">⌈threads ÷ {tpb}⌉</td></tr>
              <tr><td>waves</td><td data-k="waves">{fmt(r.waves)}</td><td className="muted">blocks ÷ {count(capacity)}</td></tr>
              <tr><td>slots idle</td><td data-k="idle">{pct(r.fill.idle)}</td><td className="muted">over all {r.fill.waveCount} wave{r.fill.waveCount === 1 ? '' : 's'}</td></tr>
              <tr>
                <td>wave time</td>
                <td data-k="time">{r.time}</td>
                <td className="muted">{r.fill.waveCount} wave{r.fill.waveCount === 1 ? '' : 's'} × {factor}, against {r.baseTime} uncoarsened, if coarsening saved nothing</td>
              </tr>
            </tbody>
          </table>
          <p style={{ margin: '0 0 0.8rem' }} data-k="verdict">
            {r.fill.whole ? (
              <span className="chip ok">only whole waves</span>
            ) : (
              <span className="chip fail">
                {r.waves < 1 ? 'less than one wave' : 'partial last wave'}: {count(r.fill.tail)} of {count(capacity)} slots busy
              </span>
            )}
          </p>
          <svg viewBox={`0 0 ${W + 160} ${shown.length * rowH + 30}`} role="img" aria-label="waves of blocks" style={{ width: '100%', maxWidth: W + 160, display: 'block' }}>
            {shown.map(({ n, w }, k) => {
              const y = k * rowH + 4;
              const fw = (W * n) / capacity;
              const skipped = rows.length > MAXROWS && k === MAXROWS - 1;
              const cells = capacity <= 48;
              return (
                <g key={w}>
                  {skipped ? (
                    <text x={0} y={y - 8} fontSize="11" style={{ fill: 'var(--ink-2)' }}>… {rows.length - MAXROWS} more full waves</text>
                  ) : null}
                  <text x={0} y={y + 16} fontSize="12" style={{ fill: 'var(--ink-2)' }} data-wave={w + 1} data-fill={n}>wave {w + 1}</text>
                  {cells ? (
                    Array.from({ length: capacity }, (_, s) => {
                      const cw = W / capacity;
                      return (
                        <rect key={s} x={60 + s * cw + 1} y={y + 2} width={Math.max(1, cw - 2)} height={rowH - 10} rx="2"
                          fill={s < n ? 'var(--ink-2)' : 'none'} stroke="var(--ink-3)" strokeDasharray={s < n ? undefined : '3 3'} />
                      );
                    })
                  ) : (
                    <>
                      <rect x={60} y={y + 2} width={W} height={rowH - 10} rx="2" fill="none" stroke="var(--ink-3)" strokeDasharray="4 3" />
                      <rect x={60} y={y + 2} width={fw} height={rowH - 10} rx="2" fill="var(--ink-2)" />
                    </>
                  )}
                  <text x={60 + W + 8} y={y + 16} fontSize="12" style={{ fill: 'var(--ink)' }}>{count(n)}/{count(capacity)}</text>
                </g>
              );
            })}
          </svg>
        </>
      ) : (
        <p>Choose a device with a number of blocks it runs at once.</p>
      )}
    </Bench>
  );
}

// ---------------------------------------------------------------- who handles what
const STEP_THREADS = 64; // two warps of 32
const WARP = 32;
const PER_ROW = 64; // eight sectors of eight floats

function MappingStepper() {
  const [C, setC] = useState(4);
  const [mapping, setMapping] = useState<Mapping>('contiguous');
  const [c, setCc] = useState(0);
  const cc = Math.min(c, C - 1);
  const units = STEP_THREADS * C;
  const addrs = warpInstruction(cc, C, mapping, WARP, STEP_THREADS);
  const acc = warpAccess(addrs, 4, SECTOR_BYTES);
  const touched = new Set(addrs.map((a) => a / 4));
  const sectors = new Set(acc.segments);
  const mine = new Set<number>();
  for (let lane = 0; lane < WARP; lane++) for (const e of elementsOf(lane, C, mapping, STEP_THREADS)) mine.add(e);
  const perInstr = Array.from({ length: C }, (_, k) => warpAccess(warpInstruction(k, C, mapping, WARP, STEP_THREADS), 4, SECTOR_BYTES));
  const loopSectors = perInstr.reduce((s, a) => s + a.transactions, 0);
  const vec = warpAccess(warpVectorInstruction(WARP), 16, SECTOR_BYTES);
  const rowsN = units / PER_ROW;
  const cell = 13;

  return (
    <Bench
      title="Who handles what"
      task="At C = 4, step through c with each mapping and count the sectors one load instruction of warp 0 touches."
      foot={<>One load instruction of warp 0 (threads 0 to 31) in loop iteration c. Sectors are NVIDIA's documented 32-byte unit (ch06-facts A2): eight floats. Outlined sectors are the ones this instruction requests. No cache: the whole-loop row counts each instruction's sectors afresh.</>}
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <span className="seg" role="group" aria-label="coarsening factor C">
          {[1, 2, 4, 8].map((k) => (
            <button key={k} aria-pressed={k === C} onClick={() => { setC(k); setCc(0); }}>C = {k}</button>
          ))}
        </span>
        <span className="seg" role="group" aria-label="mapping">
          <button aria-pressed={mapping === 'contiguous'} onClick={() => setMapping('contiguous')}>contiguous (the book's)</button>
          <button aria-pressed={mapping === 'strided'} onClick={() => setMapping('strided')}>strided</button>
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setCc(Math.max(0, cc - 1))} disabled={cc === 0}>Back</button>
        <button className="primary" onClick={() => setCc(Math.min(C - 1, cc + 1))} disabled={cc >= C - 1}>Next c</button>
        <span className="muted" aria-live="polite" data-k="iteration">loop iteration c = {cc} of {C - 1}</span>
      </div>

      <div className="narrate" role="status" style={{ marginBottom: '0.9rem' }} data-k="sectors" data-sectors={acc.transactions}>
        {mapping === 'contiguous' ? (
          <>Lane k loads element {C}k + {cc}: lanes are {4 * C} bytes apart. </>
        ) : (
          <>Lane k loads element k + {cc}·{STEP_THREADS}: lanes are 4 bytes apart. </>
        )}
        This instruction touches <b>{acc.transactions} sectors</b> and uses {pct(acc.efficiency)} of the bytes fetched.
      </div>

      <table className="threads" aria-label="elements per thread" style={{ marginBottom: '0.9rem' }}>
        <thead><tr><th>thread</th><th>elements it handles (c = 0 … {C - 1})</th></tr></thead>
        <tbody>
          {[0, 1, 2, 31].map((t) => (
            <tr key={t}>
              <td>{t}</td>
              <td>{elementsOf(t, C, mapping, STEP_THREADS).map((e, k) => (k === cc ? <b key={k}>{e} </b> : <span key={k}>{e} </span>))}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <svg viewBox={`0 0 ${PER_ROW * cell + 70} ${rowsN * (cell + 8) + 8}`} role="img" aria-label="the array, eight floats per sector"
        style={{ width: '100%', maxWidth: PER_ROW * cell + 70, display: 'block' }}>
        {Array.from({ length: rowsN }, (_, row) => (
          <g key={row}>
            <text x={0} y={row * (cell + 8) + cell + 2} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{row * PER_ROW}</text>
            {Array.from({ length: PER_ROW / 8 }, (_, s) => {
              const sec = (row * PER_ROW) / 8 + s;
              const x = 40 + s * 8 * cell;
              const y = row * (cell + 8) + 4;
              return (
                <g key={s}>
                  {Array.from({ length: 8 }, (_, k) => {
                    const e = sec * 8 + k;
                    const on = touched.has(e);
                    return (
                      <rect key={k} x={x + k * cell} y={y} width={cell - 1} height={cell} rx="1"
                        fill={on ? 'var(--global)' : mine.has(e) ? 'var(--global-wash)' : 'var(--cell)'} stroke="var(--rule)" strokeWidth="0.5" />
                    );
                  })}
                  {sectors.has(sec) ? (
                    <rect data-sector={sec} x={x - 1} y={y - 2} width={8 * cell + 1} height={cell + 4} fill="none" stroke="var(--ink)" strokeWidth="1.4" rx="2" />
                  ) : null}
                </g>
              );
            })}
          </g>
        ))}
      </svg>
      <p className="muted ui" style={{ margin: '0.4rem 0 0.9rem' }}>
        Solid: elements this instruction loads. Washed: elements warp 0 handles in other iterations. Row labels are element indices.
      </p>

      <table className="data" style={{ background: 'var(--page)', margin: 0 }} aria-label="the whole coarsening loop for warp 0">
        <thead><tr><th>warp 0, whole loop</th><th>load instructions</th><th>sectors requested, counted per instruction</th><th>bytes used</th></tr></thead>
        <tbody>
          <tr data-k="loop">
            <td>{mapping === 'contiguous' ? 'contiguous, scalar loads' : 'strided, scalar loads'}</td>
            <td>{C}</td>
            <td>{loopSectors}</td>
            <td>{pct((C * WARP * 4) / (loopSectors * SECTOR_BYTES))}</td>
          </tr>
          {C === 4 && mapping === 'contiguous' ? (
            <tr data-k="vector">
              <td>contiguous, one 16-byte load per thread (Fig. 6.12)</td>
              <td>1</td>
              <td>{vec.transactions}</td>
              <td>{pct(vec.efficiency)}</td>
            </tr>
          ) : null}
        </tbody>
      </table>
    </Bench>
  );
}

// ---------------------------------------------------------------- pitfall cards
function PitfallCards() {
  const [pick, setPick] = useState<Record<string, Verdict>>({});
  return (
    <Bench title="Helps or hurts?" task="Predict each card before you open it." foot="Answers come from the book's three pitfalls (pp. 142–143), its two examples of coarsening (p. 142) and what it says vector loads buy (p. 139).">
      <div className="cols" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(15rem, 1fr))' }}>
        {PITFALLS.map((p) => {
          const mine = pick[p.id];
          return (
            <div key={p.id} className="panel" data-card={p.id} style={{ background: 'var(--page)', padding: '0.8rem', borderRadius: 4 }}>
              <p style={{ marginTop: 0 }}>{p.scenario}</p>
              <div className="row" role="group" aria-label="your prediction">
                {VERDICTS.map((v) => (
                  <button key={v} aria-pressed={mine === v} onClick={() => setPick({ ...pick, [p.id]: v })}>{v}</button>
                ))}
              </div>
              {mine ? (
                <p style={{ marginBottom: 0 }} data-k="reveal">
                  <span className={'chip ' + (mine === p.answer ? 'ok' : 'fail')}>{mine === p.answer ? 'right' : 'the answer'}: {p.answer}</span>{' '}
                  {p.why} <span className="muted">({/[–,]/.test(p.pages) ? 'pp.' : 'p.'} {p.pages})</span>
                </p>
              ) : null}
            </div>
          );
        })}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS65: Record<string, Question> = {
  'ch06/6.5/q-factor': {
    kind: 'numeric',
    id: 'ch06/6.5/q-factor',
    prompt: (
      <>
        102,400 work units, 256 threads per block, a device that runs 100 blocks at once. What is the largest coarsening
        factor that still fills every slot in every wave?
      </>
    ),
    answer: 4,
    explain: (
      <p>
        At factor 1 there are 102,400 threads in 400 blocks: 4 full waves. Factor 2 gives 200 blocks (2 waves), factor 4
        gives 100 (one full wave). Factor 8 gives 50 blocks, half a wave: half the device sits idle while each block does
        eight units of work. Factors in between, such as 3, leave a partial wave.
      </p>
    ),
    source: 'ch. 6 §6.5, p. 143 (the same arithmetic as the book’s 1056-block case)',
  },
  'ch06/6.5/q-independent': {
    kind: 'choice',
    id: 'ch06/6.5/q-independent',
    prompt: (
      <>
        A ReLU kernel: one element per thread, no shared data, coalesced loads. You coarsen it by 4 and keep scalar loads.
        What do you expect?
      </>
    ),
    options: [
      { label: 'Little or no gain.', correct: true },
      { label: 'About 4× faster: a quarter of the threads do the same work.', why: 'Fewer threads doing more work each is the same work. The gain has to come from an overhead that goes away.' },
      { label: 'Much slower, because there are four times fewer threads.', why: 'With a large array there are still plenty of threads to fill the device; that pitfall needs a small grid.' },
    ],
    explain: (
      <p>
        Coarsening pays only when parallelising had a cost: redundant loads, redundant work, synchronisation, instruction
        overhead. Independent threads that share nothing have none for it to remove (pitfall 1). The exception is when the
        change also turns four scalar loads into one vector load, which is what Fig. 6.12 does: then an instruction
        overhead does go away.
      </p>
    ),
    source: 'ch. 6 §6.5, pp. 142–143',
  },
  'ch06/6.5/q-mapping': {
    kind: 'choice',
    id: 'ch06/6.5/q-mapping',
    prompt: (
      <>
        Coarsening by 4, thread i handles elements 4i to 4i + 3 with scalar loads in a loop over c. What does one load
        instruction of a warp look like?
      </>
    ),
    options: [
      { label: 'Lanes 16 bytes apart: the instruction touches 16 sectors and uses a quarter of each.', correct: true },
      { label: 'Perfectly coalesced, because each thread reads consecutive addresses.', why: 'Coalescing is about the lanes of one instruction, not about one thread over time. In one instruction, neighbouring lanes are four floats apart.' },
      { label: 'Uncoalesced only on AMD.', why: 'The addresses are the same on every device. NVIDIA documents the 32-byte sector this counts with (A2); AMD groups a wavefront\'s addresses into cache lines (B1).' },
    ],
    explain: (
      <p>
        In iteration c, lane k loads element 4k + c. Thirty-two lanes span 128 floats, sixteen 32-byte sectors, and use 4 of
        every 16 bytes fetched. Two fixes: make the four loads one 16-byte load per thread (Fig. 6.12), or map strided,
        i + c·T for T threads in the grid, so that every instruction reads 32 neighbouring floats again.
      </p>
    ),
    source: 'our analysis with the coalescing rule of §6.1; NVIDIA sector size from notes/today/ch06-facts.md A2',
  },
  'ch06/6.5/q-occupancy': {
    kind: 'numeric',
    id: 'ch06/6.5/q-occupancy',
    prompt: (
      <>
        Take the book's example SM from chapter 4: 65,536 registers and 2048 thread slots. Coarsening raises registers per
        thread from 32 to 64. What occupancy is now possible at most?
      </>
    ),
    answer: 50,
    unit: '%',
    explain: (
      <p>
        65,536 ÷ 64 = 1024 threads, half of 2048. Before, 65,536 ÷ 32 = 2048 filled every slot. The coarsened kernel now
        has half the warps to switch between while it waits on memory, so its gain has to exceed what those warps were
        hiding (pitfall 3).
      </p>
    ),
    source: 'ch. 6 §6.5, p. 143; the example limits from ch. 4 §4.7, pp. 85–86',
  },
  'ch06/6.5/q-scalability': {
    kind: 'choice',
    id: 'ch06/6.5/q-scalability',
    prompt: <>You tuned a coarsening factor on one GPU. What happens on another GPU, or with another input size?</>,
    options: [
      { label: 'It may need retuning, even recompiling: the best factor depends on the device and the data.', correct: true },
      { label: 'It carries over, because blocks are scheduled transparently.', why: 'Transparent scheduling is what one thread per element buys. A fixed factor bakes in an assumption about how many blocks the device runs at once.' },
      { label: 'The compiler picks a new factor for each device.', why: 'The factor is in your source (a constant like COARSEN); nothing chooses it for you.' },
    ],
    explain: (
      <p>
        The right factor depends on how many blocks the device runs at once and on how many units of work there are. The
        book's conclusion: coarsening makes scalability less transparent than one thread per element.
      </p>
    ),
    source: 'ch. 6 §6.5, p. 143',
  },
};

export function S65() {
  return (
    <>
      <p>
        Every kernel so far has used the finest grain there is: one thread per output element in vector addition, per
        pixel in grayscale and blur, per output entry in matrix multiplication. That choice buys{' '}
        <a href={href('ch04', '4.3')}>transparent scalability</a>: on a device large enough, everything runs at once; on
        a smaller one, the hardware runs the blocks in waves, and the code does not change.
      </p>
      <Source section="6.5" pages="141" />

      <h2>What parallelism costs</h2>
      <p>
        Splitting work across threads is not free. Different blocks may load the same data, threads may repeat work one
        thread could have done once, they have to synchronise, and every thread executes its own index arithmetic and
        bounds checks. When the threads truly run at the same time, that price buys speed. When the hardware runs them one
        wave after another anyway, it buys nothing.
      </p>
      <p>
        In that case, serialise part of the work yourself: give each thread several units. That is{' '}
        <b>thread coarsening</b>. The number of units per thread is the <b>coarsening factor</b>, and the loop a thread
        runs over its units is the <b>coarsening loop</b>.
      </p>
      <Source section="6.5" pages="141–142" />
      <Code
        tabs={[{ label: 's65_coarsen_toy.cu', code: TOY, status: 'front-end-checked; unverified on CUDA and HIP hardware' }]}
        highlight={[15, 16, 17, 18, 19]}
        caption={
          <>
            The book's toy before and after coarsening by 4, from <code>labs/ch06/snippets/</code>. Thread 0 handles
            elements 0 to 3, thread 1 elements 4 to 7. The bounds checks and the third kernel are ours.
          </>
        }
      />

      <h2>You have already seen it</h2>
      <p>
        The vector-load kernel of Fig. 6.12 is coarsening. Each thread adds four pairs of elements instead of one, and the
        overhead removed is instruction overhead: four load instructions per input become one 16-byte load. That only works
        because the thread's four elements sit next to each other.
      </p>
      <Source section="6.5" pages="142" figs="Fig. 6.12" />
      <p>
        Which raises a question the book does not ask: with scalar loads, is the book's mapping (thread i takes 4i to
        4i + 3) still coalesced? Coalescing is decided per instruction, across the lanes of a warp.
      </p>
      <Beyond>
        <p>
          This comparison is ours: the coalescing rule of §6.1 applied to the toy, counted in NVIDIA's 32-byte sectors.
          The strided mapping is also ours.
        </p>
      </Beyond>
      <p>
        Step through the loop below with both mappings, one load instruction of warp 0 at a time, and count the sectors
        each instruction requests.
      </p>
      <Simplification>
        <p>
          One instruction at a time, in NVIDIA&rsquo;s 32-byte sectors, with no cache: sectors that later iterations
          fetch again are counted again, although a cache might serve them.
        </p>
      </Simplification>
      <MappingStepper />
      <Ask q={questionsS65['ch06/6.5/q-mapping']} />
      <p>
        So the book's contiguous mapping pairs naturally with vector loads, and the strided one with scalar loads. Either
        way the array is read once; what differs is how many sectors each instruction drags in.
      </p>

      <h2>Coarsening the tiled matrix multiplication</h2>
      <p>
        Chapter 5's tiled kernel is the other example the book gives. Each block loads tiles of M and N, and neighbouring
        blocks load many of the same tiles. Fewer blocks, each computing a larger tile of the output, load each input tile
        fewer times: the overhead removed is redundant loading.
      </p>
      <Source section="6.5" pages="142" />
      <p>
        Chapter 15 does this properly, and its coarsened, register-tiled kernel is this course's third GEMM milestone.
      </p>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md, A15)">
        <p>
          NVIDIA's Best Practices Guide recommends the same idea for shared-memory kernels: one thread processing several
          elements, so that work common to them is done once. There is no API for it; it is a way of writing the kernel.
        </p>
      </Today>

      <h2>Three ways to get it wrong</h2>
      <p>
        The book names three pitfalls. Try the cards first; the rest of the section takes each one apart.
      </p>
      <PitfallCards />

      <h3>Coarsening when there is nothing to remove</h3>
      <p>
        If the threads are independent, share no data and need no coordination, there is no parallelisation overhead, and
        coarsening should change little.
      </p>
      <Source section="6.5" pages="142–143" />
      <Ask q={questionsS65['ch06/6.5/q-independent']} />

      <h3>Coarsening too much</h3>
      <p>
        Coarsening hands the hardware less parallelism. The book's device runs 264 blocks at once.
      </p>
      <Source section="6.5" pages="143" />
      <p>
        A grid of 1056 blocks is four full waves; coarsened by 8 it is 132 blocks, half a wave, and half the device idles.
        A grid of 1584 blocks is six waves; coarsened by 4 it is 396 blocks, one and a half waves, and the last half wave
        leaves the device half empty: the tail effect from <a href={href('ch04', '4.3')}>chapter 4</a>.
      </p>
      <Simplification>
        <p>
          Every block takes the same time and the device holds a fixed number at once. The factor does not change that
          number here, as it would if coarsening raised registers or shared memory per block; lower it by hand to see that
          pitfall. "Wave time" counts only the parallelism lost, none of the overhead saved.
        </p>
      </Simplification>
      <WaveCalculator />
      <p>
        Fewer waves is not the goal. Coarsened by 8, the 1056-block grid runs in one wave instead of four, but each block
        now does eight units of work: if coarsening saved nothing, it takes as long as eight of the old waves. Whole waves
        keep the device busy; the gain itself has to come from the overhead removed.
      </p>
      <Ask q={questionsS65['ch06/6.5/q-factor']} label="Work it out" />

      <h3>Losing occupancy</h3>
      <p>
        A coarsened thread usually keeps more state: several running results, several inputs. If that raises registers per
        thread or shared memory per block enough to lower occupancy, the kernel has fewer warps to switch between while it
        waits, and the coarsening gain has to make up for that.
      </p>
      <Source section="6.5" pages="143" />
      <Ask q={questionsS65['ch06/6.5/q-occupancy']} label="Work it out" />

      <h2>No factor travels well</h2>
      <p>
        The best factor depends on the device and on the data, and it usually has to be found by measuring. Moving to
        another GPU, or another problem size, can mean retuning it and recompiling. That is the price: a coarsened kernel
        scales less transparently than one with a thread per element.
      </p>
      <Source section="6.5" pages="143" />
      <Ask q={questionsS65['ch06/6.5/q-scalability']} />
    </>
  );
}
