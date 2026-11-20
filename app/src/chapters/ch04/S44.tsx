import { useEffect, useId, useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, num, useReports, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { STRAIGHT_LINE, finished, startWarp, step, type WarpRun } from '../../sim/divergence';
import {
  coordOf,
  cut,
  dims,
  laneOf,
  linearWorking,
  place,
  threadsIn,
  warpOf,
  widthChoices,
  type Dims,
} from '../../sim/warps';

// ---------------------------------------------------------------- drawing a block and its warps
// Warps are told apart by four shades of ink, never by hue: colour in this course
// means a memory tier or a failure.
const SHADES = [9, 30, 54, 80];
const shade = (warp: number) => `color-mix(in srgb, var(--ink) ${SHADES[warp % 4]}%, var(--cell))`;
const inkOn = (warp: number) => (SHADES[warp % 4] > 45 ? 'var(--page)' : 'var(--ink)');
const MAX_SLICES = 16;

/** The block as the kernel indexes it: x across, y down, one panel per z. */
function BlockView({
  d, width, pick, upTo, onPick,
}: {
  d: Dims;
  width: number;
  /** Linear index of the marked thread. */
  pick: number | null;
  /** When set, only threads with a linear index below it are painted. */
  upTo?: number;
  onPick?: (index: number) => void;
}) {
  const c = Math.max(6, Math.min(26, Math.floor(Math.min(352 / d.x, 352 / d.y))));
  const slices = Math.min(d.z, MAX_SLICES);
  const pad = 14;
  return (
    <div className="row" style={{ alignItems: 'flex-start', gap: '0.9rem 1.1rem' }} data-view="block">
      {Array.from({ length: slices }, (_, z) => (
        <svg key={z} viewBox={`0 0 ${d.x * c + pad + 2} ${d.y * c + pad + 2}`} role="img" aria-label={`Threads of the block with z = ${z}, shaded by warp`}
          style={{ width: d.x * c + pad + 2, maxWidth: '100%', flex: 'none' }}>
          <text x={pad} y={10} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{d.z > 1 ? `z = ${z}:  x →` : 'x →'}</text>
          <text x={0} y={pad + 10} fontSize="10" style={{ fill: 'var(--ink-2)' }}>y</text>
          {Array.from({ length: d.x * d.y }, (_, k) => {
            const x = k % d.x;
            const y = Math.floor(k / d.x);
            const p = place(x, y, z, d, width);
            const painted = upTo == null || p.index < upTo;
            const marked = pick === p.index;
            return (
              <g key={k}>
                <rect x={pad + x * c} y={pad + y * c} width={c - 1} height={c - 1} rx="1.5"
                  data-thread={`${x},${y},${z}`} data-index={p.index} data-warp={p.warp} data-lane={p.lane} data-painted={painted ? '' : undefined}
                  fill={marked ? 'var(--page)' : painted ? shade(p.warp) : 'var(--cell)'}
                  stroke={marked ? 'var(--ink)' : 'var(--rule)'} strokeWidth={marked ? 2.5 : 0.6}
                  onMouseEnter={onPick ? () => onPick(p.index) : undefined} onClick={onPick ? () => onPick(p.index) : undefined}
                  style={onPick ? { cursor: 'pointer' } : undefined} />
                {c >= 18 && (painted || marked) ? (
                  <text x={pad + x * c + c / 2 - 0.5} y={pad + y * c + c / 2 + 3.5} fontSize="10" textAnchor="middle" pointerEvents="none"
                    style={{ fill: marked ? 'var(--ink)' : inkOn(p.warp) }}>{p.warp}</text>
                ) : null}
              </g>
            );
          })}
        </svg>
      ))}
      {d.z > slices ? <span className="muted">and {d.z - slices} more values of z, cut the same way</span> : null}
    </div>
  );
}

/** The same threads as the hardware groups them: one row per warp, one cell per lane. */
function WarpRows({ d, width, pick, upTo, onPick }: { d: Dims; width: number; pick: number | null; upTo?: number; onPick?: (index: number) => void }) {
  const hatch = useId();
  const k = cut(d, width);
  const c = width <= 32 ? 13 : 9;
  const left = 56;
  const top = 14;
  const rh = 15;
  const W = left + width * c + 4;
  return (
    <svg viewBox={`0 0 ${W} ${top + k.warps * rh + 4}`} role="img" aria-label="The warps of the block, one row each, one cell per lane" data-view="warps"
      style={{ width: W, maxWidth: '100%', display: 'block' }}>
      <defs>
        <pattern id={hatch} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="1.2" />
        </pattern>
      </defs>
      <text x={left} y={10} fontSize="10" style={{ fill: 'var(--ink-2)' }}>lane 0</text>
      <text x={left + width * c - 2} y={10} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>lane {width - 1}</text>
      {Array.from({ length: k.warps }, (_, w) => (
        <g key={w}>
          <text x={left - 6} y={top + w * rh + 10.5} fontSize="10.5" textAnchor="end">warp {w}</text>
          {Array.from({ length: width }, (_, lane) => {
            const index = w * width + lane;
            const real = index < k.threads;
            const painted = real && (upTo == null || index < upTo);
            const marked = pick === index;
            return (
              <g key={lane}>
                <rect x={left + lane * c} y={top + w * rh} width={c - 1} height={rh - 2} rx="1.5"
                  data-index={real ? index : undefined} data-pad={real ? undefined : ''}
                  fill={marked ? 'var(--page)' : painted ? shade(w) : 'var(--cell)'} stroke={marked ? 'var(--ink)' : 'var(--rule)'} strokeWidth={marked ? 2.5 : 0.6}
                  onMouseEnter={real && onPick ? () => onPick(index) : undefined} onClick={real && onPick ? () => onPick(index) : undefined}
                  style={real && onPick ? { cursor: 'pointer' } : undefined} />
                {real ? null : <rect x={left + lane * c} y={top + w * rh} width={c - 1} height={rh - 2} rx="1.5" fill={`url(#${hatch})`} pointerEvents="none" />}
              </g>
            );
          })}
        </g>
      ))}
    </svg>
  );
}

/** One sentence of arithmetic for a thread of the block. */
function describeThread(index: number, d: Dims, width: number): string {
  const t = coordOf(index, d);
  const where = d.z > 1 ? `(x ${t.x}, y ${t.y}, z ${t.z})` : d.y > 1 ? `(x ${t.x}, y ${t.y})` : `x ${t.x}`;
  const work = linearWorking(t.x, t.y, t.z, d);
  return `Thread ${where}: linear index ${work}. Warp ${index} / ${width} rounded down = ${warpOf(index, width)}; lane ${index} mod ${width} = ${laneOf(index, width)}.`;
}

// ---------------------------------------------------------------- lab 4.1's Metal probe
interface ProbeRow {
  /** The threadgroup shape the probe kernel was dispatched with. */
  tx: number;
  ty: number;
  /** [[threads_per_simdgroup]] as the kernel saw it. */
  width: number;
  groups: number | undefined;
  /** Whether every SIMD-group was a consecutive run of the linear thread index. */
  consecutive: boolean | null;
}
interface Probe {
  device: string;
  whose: string;
  rows: ProbeRow[];
  /** The one width every row reported, or null when they differ. */
  width: number | null;
  consecutive: boolean | null;
}
const LAB1 = 'ch04/lab1_launchconfig';
/** The "probe" rows lab 4.1's driver writes after running the probe kernel on a Metal device, if it has run. */
function metalProbe(reports: Reports): Probe | null {
  const run = reports['metal'];
  if (!run) return null;
  for (const variant of ['work', 'solution']) {
    const rows = benchRows(run, LAB1, variant)
      .filter((r) => r.name === 'probe')
      .flatMap((r): ProbeRow[] => {
        const width = num(r.width_in_kernel);
        const c: unknown = r.consecutive_runs;
        return width
          ? [{ tx: num(r.threadgroup_x) ?? 0, ty: num(r.threadgroup_y) ?? 0, width, groups: num(r.simdgroups_per_threadgroup), consecutive: typeof c === 'boolean' ? c : null }]
          : [];
      });
    if (rows.length)
      return {
        device: String(run.device.name ?? 'your Metal device'),
        whose: variant === 'work' ? 'your probe kernel' : "the reference solution's probe kernel",
        rows,
        width: rows.every((r) => r.width === rows[0].width) ? rows[0].width : null,
        consecutive: rows.every((r) => r.consecutive === true) ? true : rows.some((r) => r.consecutive === false) ? false : null,
      };
  }
  return null;
}

const PRESETS: { label: string; d: Dims }[] = [
  { label: '4 × 4', d: dims(4, 4) },
  { label: '8 × 8', d: dims(8, 8) },
  { label: '16 × 16', d: dims(16, 16) },
  { label: '4 × 8 × 2', d: dims(4, 8, 2) },
  { label: '48 × 1', d: dims(48) },
];
const AXIS = ['x', 'y', 'z'] as const;
const count = (n: number, one: string, many = one + 's') => `${n} ${n === 1 ? one : many}`;

function WarpLab() {
  const { reports } = useReports();
  const choices = useMemo(() => widthChoices(reports), [reports]);
  const probe = useMemo(() => metalProbe(reports), [reports]);
  const [d, setD] = useState<Dims>(dims(8, 8));
  const [key, setKey] = useState('book');
  const [pick, setPick] = useState<number | null>(null);
  const [n, setN] = useState(0);
  const [playing, setPlaying] = useState(false);

  const metal = key === 'metal';
  const chosen = choices.find((c) => c.key === key) ?? choices[0];
  const width = metal ? (probe && probe.consecutive === true ? probe.width : null) : chosen.width;
  const cap = metal ? 1024 : chosen.maxThreads;
  const threads = threadsIn(d);
  const valid = AXIS.every((a) => Number.isInteger(d[a]) && d[a] >= 1);
  const over = valid && threads > cap;
  const ok = valid && !over && width != null;
  const k = ok ? cut(d, width) : null;
  const at = Math.min(n, threads);

  useEffect(() => {
    setPick(null);
    setN(0);
    setPlaying(false);
  }, [d, key]);
  useEffect(() => {
    if (!playing) return;
    if (!ok || at >= threads) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setN((v) => v + 1), threads > 128 ? 40 : 160);
    return () => window.clearTimeout(id);
  }, [playing, at, threads, ok]);

  const setAxis = (a: (typeof AXIS)[number], v: string) => setD({ ...d, [a]: v.trim() === '' ? 0 : Math.floor(Number(v)) });
  const cur = at > 0 ? at - 1 : null;
  const curT = cur != null ? coordOf(cur, d) : null;
  const prevT = cur != null && cur > 0 ? coordOf(cur - 1, d) : null;

  return (
    <>
      <Bench
        title="Warp painter"
        task="Set a block shape and a width. Which threads end up in the same warp?"
        foot={
          <>
            Each shade is one warp; a number in a cell is its warp. Hatched cells are lanes with no thread. Point at a thread, in either
            drawing, to see its arithmetic. {metal ? '' : chosen.note}
          </>
        }
      >
        <div className="row" style={{ marginBottom: '0.7rem' }}>
          {AXIS.map((a) => (
            <label className="field" key={a}>
              blockDim.{a}
              <input type="number" min={1} step={1} value={d[a] || ''} aria-label={`blockDim.${a}`} style={{ width: '5.2rem' }} onChange={(e) => setAxis(a, e.target.value)} />
            </label>
          ))}
          <label className="field">
            Warp width
            <select value={key} aria-label="Warp width" onChange={(e) => setKey(e.target.value)}>
              {choices.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
              <option value="metal">Metal: what lab 4.1's probe recorded</option>
            </select>
          </label>
        </div>
        <div className="row" style={{ marginBottom: '0.9rem' }}>
          <span className="muted">Presets</span>
          <span className="seg" role="group" aria-label="Presets">
            {PRESETS.map((p) => (
              <button key={p.label} aria-pressed={p.d.x === d.x && p.d.y === d.y && p.d.z === d.z} onClick={() => setD(p.d)}>{p.label}</button>
            ))}
          </span>
        </div>

        {metal ? (
          <div className="narrate" data-metal={probe ? 'probed' : 'none'} style={{ marginBottom: '0.9rem' }}>
            {probe ? (
              <>
                <p style={{ margin: '0 0 0.5em' }}>
                  On <b>{probe.device}</b>, {probe.whose} in lab 4.1 recorded this, one line per threadgroup shape it was dispatched with:
                </p>
                <table className="threads" style={{ width: 'auto', marginBottom: '0.5em' }}>
                  <thead>
                    <tr><th>threadgroup</th><th>threads in a SIMD-group</th><th>SIMD-groups</th><th>consecutive runs of the linear index</th></tr>
                  </thead>
                  <tbody>
                    {probe.rows.map((r, i) => (
                      <tr key={i} data-probe-row="">
                        <td>{r.tx} × {r.ty}</td>
                        <td>{r.width}</td>
                        <td>{r.groups ?? 'not recorded'}</td>
                        <td>{r.consecutive == null ? 'not recorded' : r.consecutive ? 'yes' : 'no'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {width != null
                  ? `The drawing below applies consecutive runs of ${width} to your block${probe.rows.some((r) => r.tx === d.x && r.ty === d.y && d.z === 1) ? ', which is one of the shapes the probe ran' : ', a shape the probe did not run'}. It is what one pipeline did on one device in these dispatches, not a rule Apple documents.`
                  : 'The groups were not all consecutive runs of one width, so no rule here can draw them. The table is the result.'}
              </>
            ) : (
              <>
                Nothing to draw. Metal says a threadgroup is cut into SIMD-groups "in an implementation-defined fashion" and documents no
                width, so a drawing made from a rule would be invented. Run lab 4.1 with its benchmark (
                <code>python3 labs/pmpp.py run 4.1 --bench</code>, then <code>./verify.sh</code>) and this choice will show what the probe
                kernel recorded on your device.
              </>
            )}
          </div>
        ) : null}

        {!valid ? (
          <p><span className="chip fail">Each dimension of a block is a whole number, 1 or more.</span></p>
        ) : over ? (
          <p data-over="">
            <span className="chip fail">{threads} threads in a block</span>{' '}
            is more than the {cap} this choice allows, so the launch would be refused. Nothing is drawn.
          </p>
        ) : k && width != null ? (
          <>
            <div className="row" style={{ marginBottom: '0.8rem' }} data-readout="cut">
              <span><b data-out="threads">{k.threads}</b> threads</span>
              <span><b data-out="warps">{k.warps}</b> {k.warps === 1 ? 'warp' : 'warps'} in the block</span>
              <span><b data-out="padded">{k.padded}</b> idle {k.padded === 1 ? 'lane' : 'lanes'}</span>
              <span data-out="rows">
                {d.y * d.z === 1
                  ? 'one row: a warp is a run of consecutive x'
                  : k.rowsPerWarp == null
                    ? 'warp boundaries fall inside rows'
                    : k.rowsPerWarp >= 1
                      ? `one warp covers ${count(k.rowsPerWarp, 'row')} of the block`
                      : `one row of the block is ${count(1 / k.rowsPerWarp, 'warp')}`}
              </span>
            </div>
            <div className="narrate" role="status" data-readout="thread" style={{ marginBottom: '0.9rem', minHeight: '2.6em' }}>
              {pick != null && pick < k.threads ? describeThread(pick, d, width) : 'Point at a thread.'}
            </div>
            <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
              <div>
                <div className="mx-label">The block as the kernel indexes it</div>
                {d.x <= 64 ? (
                  <BlockView d={d} width={width} pick={pick} onPick={setPick} />
                ) : (
                  <p className="muted">A row of {d.x} threads is too long to draw here. With one row, the linear order is just threadIdx.x; look at the warps.</p>
                )}
              </div>
              <div>
                <div className="mx-label">The same threads in linear order, cut into warps</div>
                <WarpRows d={d} width={width} pick={pick} onPick={setPick} />
              </div>
            </div>
          </>
        ) : null}
      </Bench>

      <p>
        The painter shows the result. The stepper below shows the walk that produces it, for the block and width you set
        above: it visits the threads in linear order, one index at a time. Watch what happens to <code>y</code> each time{' '}
        <code>x</code> runs out, and how far the index has moved when <code>y</code> or <code>z</code> goes up by one.
      </p>
      <Bench
        title="From index to lane"
        task="Step the linear index. Before each warp fills, say which thread will be its last."
        foot="The block and the width are the ones set in the painter. The index goes from 0 to one less than the number of threads in the block."
      >
        {ok && k && width != null ? (
          <>
            <div className="row" style={{ marginBottom: '0.8rem' }}>
              <button onClick={() => { setPlaying(false); setN(0); }} disabled={at === 0}>Restart</button>
              <button onClick={() => { setPlaying(false); setN(at - 1); }} disabled={at === 0}>Back</button>
              <button className="primary" onClick={() => { setPlaying(false); setN(at + 1); }} disabled={at >= threads}>Step</button>
              <button onClick={() => { setPlaying(false); setN(Math.min(threads, (Math.floor(at / width) + 1) * width)); }} disabled={at >= threads}>Fill this warp</button>
              <button onClick={() => setPlaying(!playing)} disabled={at >= threads}>{playing ? 'Pause' : 'Play'}</button>
              <span className="muted" aria-live="polite">{at} of {threads} threads placed</span>
            </div>
            <div className="narrate" role="status" data-readout="step" style={{ marginBottom: '0.9rem' }}>
              {cur == null || !curT ? (
                <>Nothing placed yet. Index 0 is thread (x 0, y 0, z 0); it goes to lane 0 of warp 0. Press Step.</>
              ) : (
                <>
                  Index {cur}: thread (x {curT.x}, y {curT.y}, z {curT.z}), because {linearWorking(curT.x, curT.y, curT.z, d)}. It goes to warp{' '}
                  {warpOf(cur, width)}, lane {laneOf(cur, width)}.{' '}
                  {prevT && curT.z !== prevT.z
                    ? `x and y both ran out, so z went up by one: one step of z is worth ${d.x * d.y} positions (blockDim.x times blockDim.y).`
                    : prevT && curT.y !== prevT.y
                      ? `x ran out, so y went up by one: one step of y is worth ${d.x} positions (blockDim.x).`
                      : ''}{' '}
                  {laneOf(cur, width) === width - 1 ? `That fills warp ${warpOf(cur, width)}.` : ''}
                  {at >= threads && k.padded > 0 ? ` The block is used up and the last warp has ${k.padded} lanes with no thread.` : ''}
                </>
              )}
            </div>
            <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)' }}>
              <div>
                {d.x <= 64 ? <BlockView d={d} width={width} pick={cur} upTo={at} /> : <p className="muted">One long row: the walk is just x = 0, 1, 2, ...</p>}
              </div>
              <div>
                <WarpRows d={d} width={width} pick={cur} upTo={at} />
              </div>
            </div>
          </>
        ) : (
          <p className="muted">Set a block the painter can draw, and the walk appears here.</p>
        )}
      </Bench>
    </>
  );
}

// ---------------------------------------------------------------- one fetch, many lanes
const LANE_CHOICES = [
  { v: 8, label: "8 lanes (the toy size of the book's Fig. 4.8)" },
  { v: 32, label: "32 lanes (the book's warp)" },
  { v: 64, label: '64 lanes (HIP on gfx9)' },
];
const REG_ROWS = [
  { key: 'r2', label: 'r2' },
  { key: 'r3', label: 'r3' },
  { key: 'r1', label: 'r1' },
] as const;

function FetchStepper() {
  const [lanes, setLanes] = useState(8);
  const [warps, setWarps] = useState<WarpRun[]>(() => [startWarp(0, 8), startWarp(1, 8)]);
  const [last, setLast] = useState<number | null>(null);
  const reset = (w: number) => {
    setLanes(w);
    setWarps([startWarp(0, w), startWarp(1, w)]);
    setLast(null);
  };
  const advance = (which: number[]) => {
    setWarps((ws) => ws.map((w, i) => (which.includes(i) ? step(w) : w)));
    setLast(which[which.length - 1]);
  };
  const said = () => {
    if (last == null) return 'Nothing has been fetched. Each processing block holds one warp, and each warp has its own program counter.';
    const w = warps[last];
    const ins = STRAIGHT_LINE[w.pc - 1];
    const a = w.regs[0];
    const z = w.regs[lanes - 1];
    const show = (r: typeof a, i: number) =>
      ins.dst === 'r1' ? `${r.r2} + ${r.r3} = ${r.r1}` : `${ins.dst} = ${r[ins.dst]}, for its own i = ${w.first + i}`;
    return `Processing block ${last} fetched "${ins.text}" once, and all ${lanes} lanes of warp ${last} executed it. Lane 0: ${show(a, 0)}. Lane ${lanes - 1}: ${show(z, lanes - 1)}.`;
  };
  return (
    <Bench
      title="One fetch, many lanes"
      task="Step each warp. Count the fetches, then count the additions."
      foot={
        <>
          The program is three instructions with no branch; the last is the book's own example. The inputs are small made-up numbers,
          different for every thread, so that you can check a lane by hand.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          Width of a warp
          <select value={lanes} aria-label="Width of a warp" onChange={(e) => reset(Number(e.target.value))}>
            {LANE_CHOICES.map((c) => <option key={c.v} value={c.v}>{c.label}</option>)}
          </select>
        </label>
        <button className="primary" onClick={() => advance([0])} disabled={finished(warps[0])}>Step warp 0</button>
        <button onClick={() => advance([1])} disabled={finished(warps[1])}>Step warp 1</button>
        <button onClick={() => advance([0, 1])} disabled={finished(warps[0]) && finished(warps[1])}>Step both</button>
        <button onClick={() => reset(lanes)} disabled={warps.every((w) => w.fetches === 0)}>Restart</button>
      </div>
      <div className="narrate" role="status" data-readout="fetch" style={{ marginBottom: '0.9rem' }}>{said()}</div>
      <div className="cols" style={lanes > 8 ? { gridTemplateColumns: 'minmax(0, 1fr)' } : undefined}>
        {warps.map((w, i) => (
          <div className="panel" key={i} data-warp={i} style={{ overflowX: 'auto' }}>
            <div className="mx-label">Processing block {i}: warp {i}, threads {w.first} to {w.first + lanes - 1}</div>
            <div className="row" style={{ alignItems: 'flex-start', gap: '0.6rem 1.4rem', marginBottom: '0.6rem' }}>
              <div>
                <div className="muted">instruction fetch and dispatch (one for the block)</div>
                <ol style={{ margin: '0.2rem 0 0', paddingLeft: '1.4rem' }} className="mono">
                  {STRAIGHT_LINE.map((ins, k) => (
                    <li key={k} data-pc={k === w.pc ? '' : undefined} style={{ fontWeight: k === w.pc - 1 ? 700 : 400, color: k < w.pc ? 'var(--ink)' : 'var(--ink-2)' }}>
                      {ins.text} <span className="muted" style={{ fontWeight: 400 }}>({ins.meaning}){k === w.pc - 1 ? ' : just executed' : k === w.pc ? ' : next' : ''}</span>
                    </li>
                  ))}
                </ol>
              </div>
              <div data-out="counts">
                <div><b data-out="fetches">{w.fetches}</b> {w.fetches === 1 ? 'fetch' : 'fetches'}</div>
                <div><b data-out="laneops">{w.laneOps}</b> lane-operations</div>
              </div>
            </div>
            <table className="threads" aria-label={`Registers of warp ${i}, one column per lane`} style={{ width: 'auto' }}>
              <thead>
                <tr>
                  <th>lane</th>
                  {w.regs.map((_, l) => <th key={l} style={{ textAlign: 'right', paddingLeft: '0.35rem' }}>{l}</th>)}
                </tr>
              </thead>
              <tbody>
                {REG_ROWS.map((r) => (
                  <tr key={r.key} className={w.pc > 0 && STRAIGHT_LINE[w.pc - 1].dst === r.key ? 'cur' : ''} data-reg={r.key}>
                    <td className="t-reg">{r.label}</td>
                    {w.regs.map((reg, l) => (
                      <td key={l} className="t-reg" style={{ textAlign: 'right', paddingLeft: '0.35rem' }}>{reg[r.key] == null ? '·' : reg[r.key]}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const P_ID = 'ch04/4.4/p-warp-of-thread';
const P_BLOCK = dims(8, 8);
const P_AT = place(3, 5, 0, P_BLOCK, 32);

export const questionsS44: Record<string, Question> = {
  'ch04/4.4/q-3d': {
    kind: 'numeric',
    id: 'ch04/4.4/q-3d',
    prompt: (
      <>
        A block has <code>blockDim</code> x 4, y 8, z 2 and the warp width is 32. Which warp (counting from 0) holds the
        thread with <code>threadIdx</code> x 1, y 2, z 1?
      </>
    ),
    answer: 1,
    explain: (
      <p>
        Linear index = x + y·Dx + z·Dx·Dy = 1 + 2·4 + 1·32 = 41. Dividing by 32 and rounding down gives warp 1; the
        remainder, 9, is its lane. A shortcut for this shape: one z slice is 4·8 = 32 threads, exactly one warp, so the
        warp number is the z index. That is the book's own example of a three-dimensional block.
      </p>
    ),
    source: 'ch. 4 §4.4, pp. 75–76',
  },
  'ch04/4.4/q-ex1ab': {
    kind: 'numeric',
    id: 'ch04/4.4/q-ex1ab',
    prompt: (
      <>
        The book's exercise 1, parts a and b. A kernel is launched over <code>N = 1024</code> elements with{' '}
        <code>(N + 128 - 1)/128</code> blocks of 128 threads, on a device whose warps are 32 wide. How many warps are in
        the grid?
      </>
    ),
    answer: 32,
    unit: 'warps',
    explain: (
      <p>
        A 128-thread block is 128 / 32 = 4 warps (part a). The launch has (1024 + 127) / 128 = 8 blocks in integer
        arithmetic, so the grid holds 8 · 4 = 32 warps (part b). On a 64-wide device the same launch is 16 warps. The
        book prints no answers; this one is ours.
      </p>
    ),
    source: 'ch. 4 exercise 1, pp. 90–91',
  },
  'ch04/4.4/q-pad': {
    kind: 'choice',
    id: 'ch04/4.4/q-pad',
    prompt: (
      <>
        You launch 48-thread blocks on a device with 32-wide warps and on one with 64-wide warps. How many warps does
        one block take on each, and how many lanes sit idle?
      </>
    ),
    options: [
      { label: '2 warps with 16 idle lanes on the first; 1 warp with 16 idle lanes on the second.', correct: true },
      { label: '2 warps with 16 idle lanes on both.', why: 'At 64 wide, 48 threads fit in one warp.' },
      { label: '1.5 warps on the first and 0.75 of a warp on the second, with nothing idle.', why: 'Warps are whole. A warp that is not full still occupies all of its lanes.' },
      { label: '2 warps on the first and 1 on the second, with nothing idle.', why: 'The lanes are there whether or not a thread fills them: 64 − 48 on the first device, and 64 − 48 on the second.' },
    ],
    explain: (
      <p>
        Warps in a block = threads divided by the width, rounded up; the last warp is padded. At 32: ⌈48 / 32⌉ = 2, with
        2·32 − 48 = 16 lanes unused. At 64: ⌈48 / 64⌉ = 1, with 64 − 48 = 16 unused. The idle lanes happen to be equal;
        the number of warps the scheduler has to work with is not. The 32-wide case is the book's; the 64-wide one is
        the same arithmetic with AMD's documented width.
      </p>
    ),
    source: 'ch. 4 §4.4, p. 75; notes/today/ch04-facts.md B6',
  },
  'ch04/4.4/q-rows': {
    kind: 'choice',
    id: 'ch04/4.4/q-rows',
    prompt: <>A block is 16 × 16. How many rows of the block does one warp cover when warps are 32 wide, and when they are 64 wide?</>,
    options: [
      { label: 'Two rows, and four rows.', correct: true },
      { label: 'One row, and two rows.', why: 'A row of this block is 16 threads, half of a 32-wide warp.' },
      { label: 'Two rows at both widths.', why: 'A wider warp takes a longer run of the same linear order.' },
      { label: 'Eight rows, and four rows.', why: 'Those are the numbers of warps in the block, not of rows in a warp.' },
    ],
    explain: (
      <p>
        x fills first, so a warp is a run of consecutive rows whenever the row length divides the width. 32 / 16 = 2
        rows; 64 / 16 = 4 rows. The block is 8 warps on one device and 4 on the other, and a row boundary that fell
        between two warps at 32 can fall inside one at 64. Section 4.5 counts what that does to a boundary check.
      </p>
    ),
    source: 'ch. 4 §4.5, p. 82; notes/today/ch04-facts.md B6',
  },
  'ch04/4.4/q-hardcode': {
    kind: 'choice',
    id: 'ch04/4.4/q-hardcode',
    prompt: (
      <>
        A kernel computes its lane as <code>threadIdx.x % 32</code> and builds lane masks in a 32-bit integer. It is
        compiled, unchanged, for a device whose warps are 64 wide. What happens?
      </>
    ),
    options: [
      {
        label: 'It compiles and runs, but it treats each 64-wide warp as two groups of 32: logic built on that lane number is wrong or uses half of the hardware, and a 32-bit mask cannot name lanes 32 to 63.',
        correct: true,
      },
      { label: 'Nothing changes: lanes are numbered 0 to 31 on every device.', why: 'On that device the lanes run from 0 to 63.' },
      { label: 'It fails to compile.', why: 'It is ordinary integer arithmetic. It compiles everywhere, which is why the mistake survives.' },
      { label: 'The device switches to 32-wide warps for this kernel.', why: 'The width belongs to the hardware. A kernel cannot set it.' },
    ],
    explain: (
      <>
        <p>
          AMD's HIP documentation says it in so many words: code that assumes 32 can run on a 64-wide device, and then
          uses half of the compute resources; and <code>threadIdx.x % warpSize</code> can exceed 31 there, so{' '}
          <code>1 &lt;&lt; lane</code> has to become a 64-bit shift. The fix is to read <code>warpSize</code> in device
          code and the device property on the host.
        </p>
        <p>
          You will meet exactly this in llama.cpp: its <code>argmax.cu</code> is written for{' '}
          <code>WARP_SIZE 32</code> throughout, while other kernels in the same directory ask for the physical width.
        </p>
      </>
    ),
    source: 'notes/today/ch05-facts.md B5; notes/today/ch04-facts.md D1',
  },
};

export function S44() {
  return (
    <>
      <p>
        Section 4.3 gave you the rule for correctness inside a block: assume the threads run in any order whatever, and
        when one phase must finish before the next starts, say so with a barrier. That rule stands. This section is
        about something the rule lets you ignore and performance does not: how the hardware in fact groups the threads
        of a block when it runs them.
      </p>
      <Source section="4.4" pages="74" />

      <h2>A block is cut into warps</h2>
      <p>
        An SM does not schedule the threads of a block one at a time. It cuts the block into groups of a fixed size,
        called <b>warps</b>, and the warp is the unit it schedules: when a warp is chosen, all of its threads execute
        the next instruction together. The book gives the size as 32 threads “in most implementations to date”, and in
        the same paragraph calls it an implementation detail that may change.
      </p>
      <Source section="4.4" pages="74, 77" figs="Fig. 4.6" />
      <p>
        Take that caution literally, because across the three backends of this course the number is already not one
        number. A kernel reads the width as <code>warpSize</code> on CUDA and HIP; a Metal kernel receives it as a
        parameter. No kernel you write should contain the literal 32 as a width.
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md A8, B6, C2; ch05-facts.md B5)">
        <p>
          CUDA: the documentation lists a warp size of 32 for every compute capability and describes{' '}
          <code>warpSize</code> as “commonly 32”. HIP: 64 on gfx9 devices (AMD Instinct), 32 on gfx10 and above. Metal:
          the unit is a SIMD-group; its width is a property of a compiled pipeline and has no documented value.
        </p>
      </Today>
      <p>
        If you have written AVX or NEON code, you already have the right picture: a warp is a vector's worth of threads,
        and each thread is one <b>lane</b> of it. The rest of this section is about which thread gets which lane.
      </p>

      <h2>Counting warps</h2>
      <p>
        A block of 256 threads is 256 / 32 = 8 warps. If an SM holds three such blocks, it holds 24 warps. When the
        block size is not a multiple of the width, the last warp is filled up with lanes that hold no thread: a
        48-thread block is 2 warps, and the second has 16 threads and 16 idle lanes. Those lanes are hardware that is
        reserved and does nothing for as long as the block lives.
      </p>
      <Source section="4.4" pages="74–75" figs="Fig. 4.6" />
      <p>
        In general a block of <i>T</i> threads on a device of width <i>W</i> is ⌈<i>T</i> / <i>W</i>⌉ warps with{' '}
        ⌈<i>T</i> / <i>W</i>⌉·<i>W</i> − <i>T</i> idle lanes. NVIDIA's guide says the same of its own devices: any
        number of threads is legal, and a total that is not a multiple of 32 leaves lanes of the last warp unused.
      </p>
      <Ask q={questionsS44['ch04/4.4/q-ex1ab']} label="Work it out" />
      <Ask q={questionsS44['ch04/4.4/q-pad']} />

      <h2>Which threads share a warp</h2>
      <p>
        For a one-dimensional block the cut follows <code>threadIdx.x</code>: warp 0 holds threads 0 to 31, warp 1
        holds 32 to 63, and warp <i>n</i> holds 32<i>n</i> to 32(<i>n</i> + 1) − 1. For two or three dimensions the
        threads are first put in one line, in row-major order: all of <code>y = 0</code> in increasing{' '}
        <code>x</code>, then <code>y = 1</code>, and for a three-dimensional block all of <code>z = 0</code> before any
        of <code>z = 1</code>. The line is then cut into consecutive runs of the width.
      </p>
      <Source section="4.4" pages="75–76" figs="Fig. 4.7" />
      <p>
        Written out, with <code>blockDim</code> = (<i>Dx</i>, <i>Dy</i>, <i>Dz</i>) and width <i>W</i>:
      </p>
      <ul>
        <li>linear index = <code>x + y·Dx + z·Dx·Dy</code></li>
        <li>warp = linear index / <i>W</i>, rounded down</li>
        <li>lane = linear index mod <i>W</i></li>
      </ul>
      <Aside>
        <p>
          The formula is ours. The book states the rule in words; NVIDIA documents it as strides (“threadIdx.y has a
          stride of blockDim.x, and threadIdx.z has a stride of blockDim.x * blockDim.y”) and says each warp holds
          consecutive, increasing thread IDs starting with thread 0. Neither prints the three lines above.
          (notes/today/ch04-facts.md A8)
        </p>
      </Aside>
      <p>
        It is the offset of element (<i>z</i>, <i>y</i>, <i>x</i>) in a row-major tensor of shape (<i>Dz</i>,{' '}
        <i>Dy</i>, <i>Dx</i>): flatten the block the way <code>reshape(-1)</code> would, and a warp is <i>W</i>{' '}
        consecutive elements. The book works three cases. A 4 × 4 block is 16 threads, half a warp, padded with 16 more
        positions. An 8 × 8 block is two warps: rows 0 to 3, then rows 4 to 7. A block with x 4, y 8, z 2 (the book
        writes it z first, as 2 × 8 × 4) is also two warps: the whole of <code>z = 0</code>, then the whole of{' '}
        <code>z = 1</code>.
      </p>
      <Predict
        id={P_ID}
        prompt={
          <>
            An 8 × 8 block, warps 32 wide. Which warp (counting from 0) holds the thread with <code>threadIdx.x = 3</code>
            , <code>threadIdx.y = 5</code>? Commit before you open the painter.
          </>
        }
      >
        {(v) => (
          <>
            <p data-reveal={P_ID}>
              {v === P_AT.warp ? 'Yes: ' : `It is warp ${P_AT.warp}. `}
              the linear index is {linearWorking(3, 5, 0, P_BLOCK)}, and {P_AT.index} / 32 rounded down is {P_AT.warp}, with
              lane {P_AT.index} mod 32 = {P_AT.lane}. Rows 0 to 3 fill warp 0, so everything from row 4 on is in warp 1.
              The marked cell is the thread.
            </p>
            <BlockView d={P_BLOCK} width={32} pick={P_AT.index} />
          </>
        )}
      </Predict>
      <p>
        Now change the shape and the width yourself. Try the presets first, then switch the width to 64 with the 8 × 8
        and the 16 × 16 block still selected, and watch which threads change company.
      </p>
      <Simplification>
        <p>
          The painter applies the cut that NVIDIA documents, with whatever width you choose. AMD documents the linear
          order and computes lanes as <code>threadIdx.x % warpSize</code> in its examples, but has no single sentence
          saying warps are consecutive runs (facts B6). For Metal no rule exists to apply, so that choice draws only
          what a kernel recorded on your device.
        </p>
      </Simplification>
      <WarpLab />
      <Ask q={questionsS44['ch04/4.4/q-3d']} label="Work it out" />

      <h2>What the shape of a block changes</h2>
      <p>
        The number of threads fixes the number of warps. The shape fixes who is in them. A 16 × 16 block and a 256 × 1
        block are both 8 warps at width 32, but in the first a warp is two rows of 16 neighbours in <code>x</code>, and
        in the second it is 32 neighbours in <code>x</code>. Change the width to 64 and the 16 × 16 block is 4 warps of
        four rows each.
      </p>
      <Source section="4.5" pages="82" />
      <p>
        NVIDIA's guide says multi-dimensional blocks are “for convenience only” and do not affect performance. That is
        true of the launch itself. It is not the whole story for a kernel, because everything the hardware does per
        warp depends on who is in the warp: the control divergence of <a href={href('ch04', '4.5')}>section 4.5</a>,
        and in chapter 6 both the merging of a warp's memory accesses and the conflicts between them in shared memory.
      </p>
      <Aside>
        <p>
          NVIDIA's sentence: “The use of multi-dimensional thread blocks and grids is for convenience only and does not
          affect performance.” (notes/today/ch04-facts.md A8). The forward references are to the book's §6.1 and §6.4.
        </p>
      </Aside>
      <Ask q={questionsS44['ch04/4.4/q-rows']} />

      <h2>One instruction, every lane</h2>
      <p>
        An SM executes a warp in SIMD fashion: single instruction, multiple data. At any instant one instruction is
        fetched, and it is executed for every thread of the warp, each on its own data.
      </p>
      <Source section="4.4" pages="76–77" figs="Fig. 4.8" />
      <p>
        To make that cheap an SM is built out of <b>processing blocks</b>: a group of streaming processors behind one
        instruction fetch and dispatch unit. The SM's warps are spread over its processing blocks, and all threads of a
        warp run on the same one. The book's figure draws a toy SM with two processing blocks of eight; of its example
        device it says an SM's 128 streaming processors form four processing blocks of 32.
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md A3)">
        <p>
          “Four processing blocks of 32” is the book's statement about its example and is unverified here: NVIDIA's
          text on that device gives 128 cores per SM and does not mention processing blocks. The term is NVIDIA's, from a 2017 whitepaper that describes older chips with two
          blocks of 32 and four of 16, each block with its own warp scheduler.
        </p>
      </Today>
      <p>
        The book's sidebar explains it with the processor of an architecture textbook: a control unit, holding the
        program counter and the instruction register, and a processing unit, holding the arithmetic logic and the
        registers. A processing block is one control unit wired to many processing units, one per lane. One instruction
        register drives them all, so the lanes cannot differ in what they do, only in what their registers contain:
        every lane executes <code>add r1, r2, r3</code>, and <code>r2</code> and <code>r3</code> hold different values
        in each.
      </p>
      <Source section="4.4" pages="78–79" figs="sidebar “Warps and SIMD Hardware”" />
      <p>
        Step it. Two processing blocks, as in the book's toy figure, each with its own warp.
      </p>
      <Simplification>
        <p>
          An instruction takes one step for a whole warp here, and a warp moves when you press its button. A real SM
          has pipelines, memory latency and a scheduler that picks the warp (section 4.6). The model is faithful about
          one thing: one fetch per instruction per warp, and one private set of registers per lane.
        </p>
      </Simplification>
      <FetchStepper />

      <h2>You write one thread</h2>
      <p>
        Nothing in a kernel's source mentions any of this. You write scalar code for one thread; the grouping into warps
        and the use of the SIMD hardware happen underneath. Because the scheduler in effect holds all threads of a warp
        to the same instruction, the book calls the model <b>SIMT</b>, single instruction, multiple thread, and
        contrasts it with CPU SIMD, where you or the compiler must spell the vector width out with intrinsics.
      </p>
      <Source section="4.4" pages="77" />
      <p>
        NVIDIA draws the same contrast and adds how much to care, with a comparison you will recognise: a warp is to a
        kernel what a cache line is to CPU code. You can ignore it when you ask whether the code is correct. You have to
        design for it when you ask how fast the code is. That is the stance of the rest of this chapter. The book's
        footnote says as much: SIMT is easier to write than SIMD, and threads of a warp that go different ways still
        cost you, which is the next section.
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md A9)">
        <p>
          NVIDIA's words: “analogous to the role of cache lines: the cache line size can be safely ignored when
          designing for correctness but must be considered in the code structure when designing for peak performance.”
        </p>
      </Today>

      <h2>Why build it this way</h2>
      <p>
        Control hardware is expensive: fetching and decoding instructions needs logic and access to the instruction
        cache. Sharing one control unit among many execution units means a smaller share of the chip goes to control
        and a larger share to arithmetic, which is what a throughput machine wants. The book expects designs of this
        kind to remain the norm for that reason.
      </p>
      <Source section="4.4" pages="77, 79" />
      <p>
        The same economy on AMD hardware has other proportions. AMD's nearest unit to a processing block is the SIMD: a
        compute unit of its CDNA family has four of them, each 16 lanes wide, and one instruction for a 64-wide warp
        takes four cycles, 16 threads per cycle. A processing block as the book describes it runs a whole warp in one
        step, so treat “processing block = SIMD” as a loose match.
      </p>
      <Beyond>
        <p>
          The figures are from AMD's HIP documentation (notes/today/ch04-facts.md B1). The pairing of the two units is
          ours; AMD does not draw it.
        </p>
      </Beyond>

      <h2>Warp-level primitives, by name only</h2>
      <p>
        The book ends the section by naming a family of API functions that let the threads of one warp exchange values
        and synchronise among themselves, without shared memory. Code written with them treats the warp as a level of
        its own between the block and the thread. They can pay off substantially, and chapters 10, 11 and 12 are where
        the book uses them; nothing in this chapter needs them.
      </p>
      <Source section="4.4" pages="77–78" />
      <p>
        One difference is worth carrying until then, because it is where “one source file for CUDA and HIP” stops being
        true. These functions name lanes with a bit mask. On CUDA the mask is a 32-bit <code>unsigned</code>. On HIP it
        is 64 bits on every device, 32-wide ones included, and the header rejects a 32-bit mask at compile time. Metal's
        SIMD-group functions hand back a <code>simd_vote</code> object instead of a raw integer.
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md A20, B12, C7)">
        <p>
          CUDA: all masks are <code>unsigned</code>. HIP: “All mask values returned or accepted by these built-ins are
          64-bit unsigned integer values, even when compiled for a device with 32 threads per warp.” Metal:{' '}
          <code>simd_vote</code> wraps an unspecified integer type.
        </p>
      </Today>

      <h2>On Metal, ask; do not compute</h2>
      <p>
        Everything above about the cut is CUDA's documented rule. Metal gives you the same linear thread index and then
        says the division into SIMD-groups is “implementation-defined”. A Metal kernel must therefore not work out its
        group as index / width. It asks, with three kernel parameters:{' '}
        <code>[[simdgroup_index_in_threadgroup]]</code>, <code>[[thread_index_in_simdgroup]]</code> and{' '}
        <code>[[threads_per_simdgroup]]</code>. The host reads the width from the compiled pipeline as{' '}
        <code>threadExecutionWidth</code>.
      </p>
      <Beyond>
        <p>
          The book uses CUDA only. Metal's wording and the three attributes are from the Metal Shading Language
          specification (notes/today/ch04-facts.md C2). Apple does not use the word warp; the comparison is ours.
        </p>
      </Beyond>
      <p>
        Lab 4.1 has you write a probe kernel that records those three values for every thread of a threadgroup. Once
        you have run it, choose “Metal” in the painter above: it shows what your device did, and nothing if you have
        not asked it yet. The labs are on the <a href={href('ch04', 'labs')}>labs page</a>.
      </p>

      <h2>Where the width appears in code you will read</h2>
      <p>
        The three projects this course prepares you to read all carry the width in their source, and all three show
        both habits: a hard-coded number in one file, a query in the next.
      </p>
      <Beyond>
        <p>
          Read from each project's default branch on 2026-11-20 (notes/today/ch04-facts.md D1 to D4). Files move;
          search for the identifier.
        </p>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th>Project</th><th>Hard-coded</th><th>Asked for</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>PyTorch, CUDA and ROCm</td>
            <td><code>#define C10_WARP_SIZE 32</code> on CUDA builds</td>
            <td>
              on ROCm builds <code>C10_WARP_SIZE</code> is a function (64 for gfx9 device code, 32 otherwise), and host
              code must call <code>at::cuda::warp_size()</code>
            </td>
          </tr>
          <tr>
            <td>llama.cpp, CUDA and HIP</td>
            <td><code>#define WARP_SIZE 32</code>, the width most kernels are written for</td>
            <td><code>ggml_cuda_get_physical_warp_size()</code>: 64 under HIP on gfx9, else 32</td>
          </tr>
          <tr>
            <td>llama.cpp, Metal</td>
            <td><code>#define N_SIMDWIDTH 32</code>, commented “assuming SIMD group size is 32”</td>
            <td>group and lane from <code>[[simdgroup_index_in_threadgroup]]</code> and <code>[[thread_index_in_simdgroup]]</code></td>
          </tr>
          <tr>
            <td>MLX, Metal</td>
            <td><code>constexpr int SIMD_SIZE = 32;</code> in its softmax kernel</td>
            <td><code>uint simd_size [[threads_per_simdgroup]]</code> in its arg-reduce kernel</td>
          </tr>
        </tbody>
      </table>
      <p>
        A kernel that asks is correct at any width. A kernel that hard-codes 32 is a kernel somebody has to audit before
        it meets a 64-wide device, and that audit is a realistic first contribution to any of these projects.
      </p>
      <Ask q={questionsS44['ch04/4.4/q-hardcode']} />
      <p>
        Next: what happens when the threads of one warp do not agree on the next instruction.
      </p>
    </>
  );
}
