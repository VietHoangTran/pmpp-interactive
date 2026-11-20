import { useEffect, useId, useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { useProgress } from '../../lib/progress';
import { benchRows, fmt, num, realBackends, useReports, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import {
  EX1,
  FIG49,
  countTaken,
  discardedBySelect,
  exercise1,
  fig410Trips,
  ifElse,
  loop,
  loopBatch,
  loopRows,
  maskBelow,
  maskModulo,
  run,
  select,
  shuffledTrips,
  sortedTrips,
  type Issue,
  type LoopBatch,
  type Order,
} from '../../sim/divergence';
import {
  BOOK_CASES,
  BOOK_WIDTH,
  boundary1D,
  boundary2D,
  causal,
  classifyGrid,
  dims,
  gridFor,
  regions,
  splitWarp1D,
  warpsInRightEdgeBlocks,
  widthChoices,
  type Picture,
  type WarpState,
  type WidthChoice,
} from '../../sim/warps';
import { WarpBranchScene } from './WarpBranchScene';

// The kernel of the book's exercise 1 (pp. 90 to 91), exactly as stored in
// labs/ch04/snippets/foo_kernel.cu, where the CUDA front-end check parses it.
// app/src/sim/divergence.test.ts fails if this copy drifts from the file.
const FOO = `__global__ void foo_kernel(int* a, int* b) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(threadIdx.x < 40 || threadIdx.x >= 104) {
        b[i] = a[i] + 1;
    }
    if(i%2 == 0) {
        a[i] = b[i]*2;
    }
    for(unsigned int j = 0; j < 5 - (i%3); ++j) {
        b[i] += j;
    }
}
void foo(int* a_d, int* b_d) {
    unsigned int N = 1024;
    foo_kernel<<<(N + 128 - 1)/128, 128>>>(a_d, b_d);
}`;

// ---------------------------------------------------------------- small helpers
const pct = (v: number) => `${Math.round(v * 1000) / 10}%`;
/** A count of warp-steps that may be fractional (work divided by the width). */
const stepsText = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(2));
const plural = (n: number, one: string, many = one + 's') => `${n} ${n === 1 ? one : many}`;
const big = (v: number) => v.toLocaleString('en-US');
/** The most lanes any control on this page stores a value for. */
const LANES_MAX = 128;
const cellFor = (width: number) => (width <= 32 ? 14 : width <= 64 ? 9 : 5);
/** A label above a group of buttons, laid out like label.field. */
const FIELD = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' } as const;
/** Lets a label.field holding a long select shrink to the width of a narrow screen. */
const NARROW_OK = { maxWidth: '100%', minWidth: 0 } as const;
/** A table that scrolls sideways on a narrow screen instead of running out of its box. */
const Wide = ({ children, style }: { children: ReactNode; style?: CSSProperties }) => (
  <div style={{ overflowX: 'auto', maxWidth: '100%', ...style }}>{children}</div>
);
/** Where the chosen width comes from, in one sentence. */
const widthSays = (c: WidthChoice): string =>
  c.source === 'book'
    ? 'The width is the 32 the book reports for CUDA devices to date (pp. 74, 77).'
    : c.source === 'doc'
      ? "The width is the 64 that AMD's HIP documentation gives for gfx9 devices."
      : `The width is the warpSize your device reported (${c.width}).`;
const LEFT = 150; // room for a row's label, left of lane 0
const RIGHT = 150; // room for a row's note, right of the last lane

// ---------------------------------------------------------------- lanes, one row per instruction
// Lanes that execute are solid ink (path B a lighter ink); lanes that wait are
// hatched. No hue: colour in this course means a memory tier or a failure.
interface LaneRow {
  label: string;
  active: boolean[];
  tone?: 'a' | 'b';
  note?: string;
  cur?: boolean;
  part?: string;
}

function LaneRows({
  name, width, rows, head, headLabel, onHead, mark,
}: {
  /** What the drawing is, for a screen reader. */
  name: string;
  width: number;
  rows: LaneRow[];
  /** One entry per lane, written above the rows: what each thread wants. */
  head?: (string | number)[];
  headLabel?: string;
  /** Makes the head cells buttons. */
  onHead?: (lane: number) => void;
  /** Draws a dashed line before this lane. */
  mark?: number;
}) {
  const hatch = useId();
  const c = cellFor(width);
  const top = head ? 48 : 26;
  const rh = 17;
  const W = LEFT + width * c + RIGHT;
  const H = top + Math.max(1, rows.length) * rh + 4;
  const ticks = [...new Set([...Array.from({ length: Math.ceil(width / 8) }, (_, i) => i * 8), width - 1])];
  const laneX = (l: number) => LEFT + l * c;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={name} data-view="lanes" style={{ width: W, maxWidth: '100%', display: 'block' }}>
      <defs>
        <pattern id={hatch} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="1.2" />
        </pattern>
      </defs>
      <text x={LEFT - 8} y={12} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>lane</text>
      {ticks.map((i) => (
        <text key={i} x={laneX(i) + c / 2 - 0.5} y={12} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{i}</text>
      ))}
      {head ? (
        <g data-head="">
          <text x={LEFT - 8} y={33} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{headLabel}</text>
          {head.map((h, l) => (
            <g key={l}>
              {onHead ? (
                <rect x={laneX(l)} y={21} width={c - 1} height={17} rx="1.5" data-head-lane={l} fill="var(--page)" stroke="var(--ink-3)" strokeWidth="0.8"
                  onClick={() => onHead(l)} style={{ cursor: 'pointer' }} />
              ) : null}
              <text x={laneX(l) + c / 2 - 0.5} y={33} fontSize={c >= 14 ? 10 : 7.5} textAnchor="middle" fontWeight="600" pointerEvents="none" style={{ fill: 'var(--ink)' }}>{h}</text>
            </g>
          ))}
        </g>
      ) : null}
      {rows.map((r, i) => {
        const y = top + i * rh;
        const count = r.active.filter(Boolean).length;
        return (
          <g key={i} data-row={i} data-count={count} data-part={r.part}>
            <text x={LEFT - 8} y={y + 11} fontSize="11" textAnchor="end" fontWeight={r.cur ? 650 : 400} style={{ fill: r.cur ? 'var(--ink)' : 'var(--ink-2)' }}>
              {r.cur ? '▸ ' : ''}{r.label}
            </text>
            {r.active.map((on, l) => (
              <g key={l}>
                <rect x={laneX(l)} y={y} width={c - 1} height={rh - 4} rx="1.5" data-lane={l} data-on={on ? '' : undefined}
                  fill={on ? (r.tone === 'b' ? 'var(--ink-2)' : 'var(--ink)') : 'var(--cell)'} stroke={on ? 'none' : 'var(--rule)'} strokeWidth="0.6" />
                {on ? null : <rect x={laneX(l)} y={y} width={c - 1} height={rh - 4} rx="1.5" fill={`url(#${hatch})`} pointerEvents="none" />}
              </g>
            ))}
            {r.note ? <text x={laneX(width) + 8} y={y + 11} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>{r.note}</text> : null}
          </g>
        );
      })}
      {mark != null && mark > 0 && mark < width ? (
        <line x1={laneX(mark) - 0.5} x2={laneX(mark) - 0.5} y1={16} y2={H - 2} stroke="var(--ink)" strokeWidth="1.5" strokeDasharray="4 3" pointerEvents="none" />
      ) : null}
    </svg>
  );
}

const MAX_TRIPS = 10;
/** One column per lane; click a column at a height to set that lane's trip count. */
function TripEditor({ width, trips, onSet }: { width: number; trips: number[]; onSet: (lane: number, trips: number) => void }) {
  const c = cellFor(width);
  const ch = 9;
  const W = LEFT + width * c + RIGHT;
  return (
    <svg viewBox={`0 0 ${W} ${MAX_TRIPS * ch + 6}`} role="group" aria-label="Trip count of every lane: click a column at a height to set it" data-view="trips"
      style={{ width: W, maxWidth: '100%', display: 'block', marginBottom: '0.3rem' }}>
      <text x={LEFT - 8} y={11} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{MAX_TRIPS} iterations</text>
      <text x={LEFT - 8} y={MAX_TRIPS * ch} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>1 iteration</text>
      <text x={LEFT + width * c + 8} y={MAX_TRIPS * ch} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>click to set a lane</text>
      {Array.from({ length: width }, (_, l) =>
        Array.from({ length: MAX_TRIPS }, (_, r) => {
          const t = MAX_TRIPS - r;
          const on = trips[l] >= t;
          return (
            <rect key={`${l}-${t}`} x={LEFT + l * c} y={2 + r * ch} width={c - 1} height={ch - 1} rx="1" data-lane={l} data-trip={t} data-on={on ? '' : undefined}
              fill={on ? 'color-mix(in srgb, var(--ink) 55%, var(--cell))' : 'var(--cell)'} stroke="var(--rule)" strokeWidth="0.5"
              onClick={() => onSet(l, trips[l] === t ? t - 1 : t)} style={{ cursor: 'pointer' }} />
          );
        }),
      )}
    </svg>
  );
}

// ---------------------------------------------------------------- the divergence simulator
type Kind = 'below' | 'modulo' | 'paint' | 'loop';
const KINDS: { v: Kind; label: string }[] = [
  { v: 'below', label: 'if (threadIdx.x < T) A else B' },
  { v: 'modulo', label: 'if (i % m == 0) A else B' },
  { v: 'paint', label: 'if (mask[i]) A else B, with a mask you paint' },
  { v: 'loop', label: 'for (k = 0; k < N[i]; ++k) A, with a trip count per lane' },
];
const ORDERS: { v: Order; label: string }[] = [
  { v: 'serialised', label: 'one path, then the other' },
  { v: 'interleaved', label: 'interleaved' },
];

function DivergenceSim() {
  const { reports } = useReports();
  const choices = useMemo(() => widthChoices(reports), [reports]);
  const [key, setKey] = useState('book');
  const chosen = choices.find((c) => c.key === key) ?? choices[0];
  const width = chosen.width;
  const [kind, setKind] = useState<Kind>('below');
  const [T, setT] = useState<number>(FIG49.threshold);
  const [m, setM] = useState(2);
  const [warp, setWarp] = useState(0);
  const [paint, setPaint] = useState<boolean[]>(() => maskBelow(LANES_MAX, 16));
  const [trips, setTrips] = useState<number[]>(() => fig410Trips(LANES_MAX));
  const [costA, setCostA] = useState(3);
  const [costB, setCostB] = useState(3);
  const [body, setBody] = useState(1);
  const [order, setOrder] = useState<Order>('serialised');
  /** Instructions issued so far; null means all of them. */
  const [n, setN] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);

  // The warp belongs to a block the size of exercise 1's (128 threads at the usual widths).
  const warpsInBlock = Math.max(1, Math.round(EX1.blockDim / width));
  const blockThreads = warpsInBlock * width;
  const w = Math.min(warp, warpsInBlock - 1);
  const first = w * width;
  const isLoop = kind === 'loop';
  const maskOf = (k: number): boolean[] =>
    kind === 'below' ? maskBelow(width, T, k * width) : kind === 'modulo' ? maskModulo(width, m, k * width) : Array.from({ length: width }, (_, i) => !!paint[i]);
  const taken = maskOf(w);
  const lanesTrips = Array.from({ length: width }, (_, i) => trips[i] ?? 0);
  const cond = kind === 'below' ? `threadIdx.x < ${T}` : kind === 'modulo' ? `i % ${m} == 0` : 'mask[i]';
  const program = isLoop ? loop(width, lanesTrips, body) : ifElse(width, taken, costA, costB, cond);
  const trace = run(program, order);
  const shown = n == null ? trace.spent : Math.min(n, trace.spent);
  const sel = isLoop ? null : run(select(width, costA, costB, cond));
  const st = trace.stmts[1];

  useEffect(() => {
    setN(null);
    setPlaying(false);
  }, [key, kind, T, m, warp, paint, trips, costA, costB, body, order]);
  useEffect(() => {
    if (!playing) return;
    if (shown >= trace.spent) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setN(shown + 1), 450);
    return () => window.clearTimeout(id);
  }, [playing, shown, trace.spent]);

  const label = (x: Issue): string =>
    x.part === 'A' ? `A, instruction ${x.k + 1}`
      : x.part === 'B' ? `B, instruction ${x.k + 1}`
        : x.part === 'body' ? `iteration ${(x.iter ?? 0) + 1}${body > 1 ? `, instr. ${x.k + 1}` : ''}`
          : x.stmt === 0 ? (isLoop ? x.label : `test: ${x.label}`)
            : x.label;
  const rows: LaneRow[] = trace.issues.slice(0, shown).map((x, i) => ({
    label: label(x),
    active: x.active,
    tone: x.part === 'B' ? 'b' : 'a',
    part: x.part,
    cur: i === shown - 1,
    note: x.count === width ? `all ${width}: 100%` : `${x.count} run, ${width - x.count} wait: ${pct(x.efficiency)}`,
  }));
  const said = (): string => {
    if (shown === 0)
      return isLoop
        ? 'Nothing issued yet. The first instruction reads each lane’s trip count; every lane executes it.'
        : 'Nothing issued yet. The first instruction is the test; every lane executes it, each on its own index.';
    const x = trace.issues[shown - 1];
    const lanes = x.count === width ? `All ${width} lanes execute it.` : `${x.count} of ${width} lanes execute it and ${width - x.count} wait.`;
    const end = shown === trace.spent
      ? ` That is the whole program: ${plural(trace.spent, 'warp-step')}, where the same work with no lane waiting would take ${stepsText(trace.uniform)}.`
      : '';
    return `Step ${shown} of ${trace.spent}: ${label(x)}. ${lanes}${end}`;
  };
  const setLaneTrips = (lane: number, t: number) => setTrips((old) => old.map((v, i) => (i === lane ? t : v)));
  const numberField = (name: string, value: number, set: (v: number) => void, min: number, max: number) => (
    <label className="field">
      {name}
      <input type="number" min={min} max={max} step={1} value={value} aria-label={name} style={{ width: '4.6rem' }}
        onChange={(e) => {
          const v = Math.floor(Number(e.target.value));
          if (e.target.value.trim() !== '' && Number.isFinite(v)) set(Math.max(min, Math.min(max, v)));
        }} />
    </label>
  );

  return (
    <Bench
      title="Divergence simulator"
      task="Choose a statement and predict the passes before you look at them."
      foot={
        <>
          One row is one instruction issued to the warp. Solid lanes execute it; hatched lanes wait. A lighter solid is path B. The
          warp is one of the {warpsInBlock} warps of a {blockThreads}-thread block. {widthSays(chosen)}
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field" style={NARROW_OK}>
          Statement
          <select value={kind} aria-label="Statement" onChange={(e) => setKind(e.target.value as Kind)}>
            {KINDS.map((k) => <option key={k.v} value={k.v}>{k.label}</option>)}
          </select>
        </label>
        <label className="field" style={NARROW_OK}>
          Warp width
          <select value={key} aria-label="Warp width" onChange={(e) => setKey(e.target.value)}>
            {choices.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        {kind === 'below' ? (
          <label className="field">
            <span>T = <b data-out="T">{T}</b></span>
            <input type="range" min={0} max={blockThreads} step={1} value={Math.min(T, blockThreads)} aria-label="Threshold T" style={{ width: '16rem' }}
              onChange={(e) => setT(Number(e.target.value))} />
          </label>
        ) : null}
        {kind === 'modulo' ? numberField('m', m, setM, 2, 16) : null}
        {isLoop ? numberField('Instructions in the loop body', body, setBody, 1, 3) : numberField('Instructions in A', costA, setCostA, 1, 8)}
        {isLoop ? null : numberField('Instructions in B', costB, setCostB, 0, 8)}
        {isLoop ? null : (
          <div style={FIELD}>
            Order of the passes
            <span className="seg" role="group" aria-label="Order of the passes">
              {ORDERS.map((o) => <button key={o.v} aria-pressed={order === o.v} onClick={() => setOrder(o.v)}>{o.label}</button>)}
            </span>
          </div>
        )}
      </div>
      {kind === 'below' || kind === 'modulo' ? (
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <span className="muted">Warp of the block</span>
          <span className="seg" role="group" aria-label="Warp of the block">
            {Array.from({ length: warpsInBlock }, (_, k) => {
              const a = countTaken(maskOf(k));
              return (
                <button key={k} aria-pressed={k === w} data-warp-summary={k} data-a={a} onClick={() => setWarp(k)}>
                  warp {k}: {a === width ? 'all A' : a === 0 ? 'all B' : `${a} A, ${width - a} B`}
                </button>
              );
            })}
          </span>
        </div>
      ) : null}
      {kind === 'paint' ? (
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <span className="muted">Click a lane's letter to flip it, or start from</span>
          <span className="seg" role="group" aria-label="Mask presets">
            <button onClick={() => setPaint(maskBelow(LANES_MAX, width / 2))}>first half A</button>
            <button onClick={() => setPaint(maskModulo(LANES_MAX, 2))}>every other lane</button>
            <button onClick={() => setPaint(maskBelow(LANES_MAX, 1))}>one lane A</button>
            <button onClick={() => setPaint(maskBelow(LANES_MAX, LANES_MAX))}>all A</button>
          </span>
        </div>
      ) : null}
      {isLoop ? (
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <span className="muted">Trip counts</span>
          <span className="seg" role="group" aria-label="Trip count presets">
            <button onClick={() => setTrips(fig410Trips(LANES_MAX))}>Fig. 4.10's printed values, repeated</button>
            <button onClick={() => setTrips([...sortedTrips(lanesTrips), ...Array<number>(LANES_MAX - width).fill(0)])}>the same counts, sorted</button>
            <button onClick={() => setTrips(Array<number>(LANES_MAX).fill(6))}>all 6</button>
            <button onClick={() => setTrips(Array.from({ length: LANES_MAX }, (_, i) => (i === 0 ? MAX_TRIPS : 2)))}>one lane of {MAX_TRIPS}, the rest 2</button>
          </span>
        </div>
      ) : null}

      <div className="row" style={{ marginBottom: '0.8rem' }} data-readout="trace">
        {isLoop ? (
          <span>
            <b data-out="passes">{st.passes}</b> iterations: <b data-out="uniform-iterations">{st.uniformIterations}</b> with every lane,{' '}
            <b data-out="divergent-iterations">{st.divergentIterations}</b> with lanes waiting
          </span>
        ) : (
          <span><b data-out="passes">{st.passes}</b> {st.passes === 1 ? 'pass' : 'passes'} over the branch</span>
        )}
        <span><b data-out="spent">{trace.spent}</b> warp-steps spent</span>
        <span><b data-out="uniform">{stepsText(trace.uniform)}</b> if no lane ever waited</span>
        <span><b data-out="overhead">{fmt(trace.overhead)}</b>× that cost</span>
        <span><b data-out="idle">{trace.idle}</b> idle lane-steps</span>
        <span><b data-out="efficiency">{pct(trace.efficiency)}</b> of lane-steps did work</span>
      </div>
      {sel ? (
        <p className="muted" data-readout="select" style={{ margin: '0 0 0.8rem' }}>
          Compute both and select, for comparison: <b data-out="select-spent">{sel.spent}</b> warp-steps, no lane waits,{' '}
          <b data-out="select-discarded">{discardedBySelect(taken, costA, costB)}</b> lane-steps of arithmetic computed and thrown away.
        </p>
      ) : null}

      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => { setPlaying(false); setN(0); }} disabled={shown === 0}>Restart</button>
        <button onClick={() => { setPlaying(false); setN(shown - 1); }} disabled={shown === 0}>Back</button>
        <button className="primary" onClick={() => { setPlaying(false); setN(shown + 1); }} disabled={shown >= trace.spent}>Step</button>
        <button onClick={() => { if (shown >= trace.spent) setN(0); setPlaying(!playing); }}>{playing ? 'Pause' : 'Play'}</button>
        <button onClick={() => { setPlaying(false); setN(null); }} disabled={shown >= trace.spent}>Show all</button>
        <span className="muted" aria-live="polite">{shown} of {trace.spent} instructions issued</span>
      </div>
      <div className="narrate" role="status" data-readout="step" style={{ marginBottom: '0.9rem' }}>{said()}</div>
      {isLoop ? <TripEditor width={width} trips={lanesTrips} onSet={setLaneTrips} /> : null}
      <LaneRows
        name="The instructions issued to the warp, one row each, one cell per lane"
        width={width}
        rows={rows}
        head={isLoop ? lanesTrips : taken.map((a) => (a ? 'A' : 'B'))}
        headLabel={isLoop ? 'trip count N' : 'wants'}
        onHead={kind === 'paint' ? (lane) => setPaint((old) => old.map((v, i) => (i === lane ? !v : v))) : undefined}
        mark={kind === 'below' ? T - first : undefined}
      />
    </Bench>
  );
}

// ---------------------------------------------------------------- many warps of loops
const TRIP_LO = 1;
const TRIP_HI = 64;

function BatchView({ batch, name }: { batch: LoopBatch; name: string }) {
  const hatch = useId();
  const VW = 760;
  const H = 84;
  const top = 4;
  const gap = 6;
  const count = batch.warps.length;
  const ww = (VW - gap * (count - 1)) / count;
  const lw = ww / batch.width;
  const s = H / TRIP_HI;
  return (
    <svg viewBox={`0 0 ${VW} ${H + top + 16}`} role="img" aria-label={name} style={{ width: VW, maxWidth: '100%', display: 'block' }}>
      <defs>
        <pattern id={hatch} width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="1.2" />
        </pattern>
      </defs>
      {batch.warps.map((wp, k) => {
        const x = k * (ww + gap);
        return (
          <g key={k} data-batch-warp={k} data-longest={wp.longest}>
            <rect x={x} y={top + H - wp.longest * s} width={ww} height={wp.longest * s} fill="var(--cell)" stroke="var(--rule)" strokeWidth="0.6" />
            <rect x={x} y={top + H - wp.longest * s} width={ww} height={wp.longest * s} fill={`url(#${hatch})`} />
            {wp.trips.map((t, l) => (
              <rect key={l} x={x + l * lw} y={top + H - t * s} width={Math.max(0.5, lw - (lw > 3 ? 0.6 : 0))} height={t * s} fill="var(--ink)" />
            ))}
            <text x={x + ww / 2} y={top + H + 12} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
              {count <= 8 ? `warp ${k}: ` : ''}{wp.longest}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function LengthsBatch() {
  const { reports } = useReports();
  const choices = useMemo(() => widthChoices(reports), [reports]);
  const [key, setKey] = useState('book');
  const chosen = choices.find((c) => c.key === key) ?? choices[0];
  const width = chosen.width;
  const [warps, setWarps] = useState(8);
  const [seed, setSeed] = useState(7);
  const trips = useMemo(() => shuffledTrips(warps * width, TRIP_LO, TRIP_HI, seed), [warps, width, seed]);
  const shuffled = loopBatch(trips, width);
  const sorted = loopBatch(sortedTrips(trips), width);
  const row = (id: string, name: string, b: LoopBatch) => (
    <tr data-batch={id}>
      <td>{name}</td>
      <td data-out="spent">{b.spent}</td>
      <td data-out="useful">{big(b.useful)}</td>
      <td data-out="idle">{big(b.idle)}</td>
      <td data-out="efficiency">{pct(b.efficiency)}</td>
    </tr>
  );
  return (
    <Bench
      title="One loop per thread, lengths from the data"
      task="The same lengths, in two orders. Which batch finishes first, and by how much?"
      foot={
        <>
          Each group of bars is one warp; a bar is one thread's trip count; the hatched area above the bars is lane time spent waiting for
          the warp's longest thread. The number under a warp is how many iterations it runs. Lengths are spread evenly from {TRIP_LO} to{' '}
          {TRIP_HI}. {widthSays(chosen)}
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        <div style={FIELD}>
          Warps
          <span className="seg" role="group" aria-label="Number of warps">
            {[4, 8, 16].map((v) => <button key={v} aria-pressed={warps === v} onClick={() => setWarps(v)}>{v}</button>)}
          </span>
        </div>
        <label className="field" style={NARROW_OK}>
          Warp width
          <select value={key} aria-label="Warp width of the batch" onChange={(e) => setKey(e.target.value)}>
            {choices.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
          </select>
        </label>
        <button onClick={() => setSeed(seed + 1)}>Shuffle again</button>
        <span className="muted">{warps * width} threads</span>
      </div>
      <div className="mx-label">Shuffled: lengths in the order they arrived</div>
      <BatchView batch={shuffled} name="Trip counts of every thread, in shuffled order, grouped by warp" />
      <div className="mx-label" style={{ marginTop: '0.8rem' }}>Sorted by length</div>
      <BatchView batch={sorted} name="The same trip counts sorted by length, grouped by warp" />
      <Wide style={{ marginTop: '0.9rem' }}><table className="threads" style={{ width: 'auto' }} aria-label="What each order costs">
        <thead>
          <tr><th>Order</th><th>warp-iterations spent</th><th>lane-iterations of work</th><th>lane-iterations waiting</th><th>lane time that is work</th></tr>
        </thead>
        <tbody>
          {row('shuffled', 'shuffled', shuffled)}
          {row('sorted', 'sorted', sorted)}
        </tbody>
      </table></Wide>
      <p style={{ margin: '0.7rem 0 0' }} data-readout="batch">
        The work is the same, <b>{big(sorted.useful)}</b> lane-iterations. Shuffled, the warps run for{' '}
        <b data-out="ratio">{fmt(shuffled.spent / sorted.spent)}</b> times as many iterations as sorted. With no waiting at all it would take{' '}
        <b data-out="uniform">{stepsText(sorted.uniform)}</b>.
      </p>
    </Bench>
  );
}

// ---------------------------------------------------------------- exercise 1, predicted first
const E1 = exercise1(BOOK_WIDTH);
const E1_WIDE = exercise1(64);
const block0 = (k: number) => E1.detail.find((x) => x.block === 0 && x.warp === k)!;
const EX1_ID = 'ch04/4.5/p-ex1';
const EX1_GROUPS: { title: string; fields: { key: string; label: string; unit: string; model: number }[] }[] = [
  {
    title: 'Line 4, guarded by the test on line 3',
    fields: [
      { key: 'l4-active', label: 'active warps in the grid', unit: 'warps', model: E1.line4.active },
      { key: 'l4-divergent', label: 'divergent warps in the grid', unit: 'warps', model: E1.line4.divergent },
      { key: 'l4-eff0', label: 'SIMD efficiency of warp 0 of block 0', unit: '%', model: 100 * block0(0).line4.efficiency },
      { key: 'l4-eff1', label: 'SIMD efficiency of warp 1 of block 0', unit: '%', model: 100 * block0(1).line4.efficiency },
      { key: 'l4-eff3', label: 'SIMD efficiency of warp 3 of block 0', unit: '%', model: 100 * block0(3).line4.efficiency },
    ],
  },
  {
    title: 'Line 7, guarded by the test on line 6',
    fields: [
      { key: 'l7-active', label: 'active warps in the grid', unit: 'warps', model: E1.line7.active },
      { key: 'l7-divergent', label: 'divergent warps in the grid', unit: 'warps', model: E1.line7.divergent },
      { key: 'l7-eff0', label: 'SIMD efficiency of warp 0 of block 0', unit: '%', model: 100 * block0(0).line7.efficiency },
    ],
  },
  {
    title: 'The loop on line 9',
    fields: [
      { key: 'loop-uniform', label: 'iterations with no divergence', unit: 'iterations', model: E1.loop.uniform },
      { key: 'loop-divergent', label: 'iterations with divergence', unit: 'iterations', model: E1.loop.divergent },
    ],
  },
];
const EX1_FIELDS = EX1_GROUPS.flatMap((g) => g.fields);
const near = (a: number, b: number) => Math.abs(a - b) < 0.5;

function Ex1Reveal({ mine }: { mine: number[] }) {
  const right = EX1_FIELDS.filter((f, i) => near(mine[i], f.model)).length;
  const b0 = [0, 1, 2, 3].map(block0);
  const statementRows = (pick: 'line4' | 'line7'): LaneRow[] =>
    b0.map((x) => {
      const s = x[pick];
      return {
        label: `warp ${x.warp}: threads ${x.first} to ${x.first + x.real - 1}`,
        active: s.mask,
        note: `${s.count} of ${E1.width}: ${pct(s.efficiency)}${s.divergent ? ', divergent' : s.active ? '' : ', not active'}`,
      };
    });
  const w0 = block0(0);
  return (
    <div data-reveal={EX1_ID}>
      <p>
        {right === EX1_FIELDS.length ? 'All ten match the model.' : `${right} of ${EX1_FIELDS.length} match the model.`} The book prints no
        answers; these are the counts of the model this page runs on, with the definitions given above.
      </p>
      <Wide style={{ marginBottom: '0.9rem' }}><table className="threads" style={{ width: 'auto' }} aria-label="Your entries beside the model's counts">
        <thead>
          <tr><th>Statement</th><th>Quantity</th><th>You</th><th>Model</th><th></th></tr>
        </thead>
        <tbody>
          {EX1_GROUPS.flatMap((g) =>
            g.fields.map((f) => {
              const i = EX1_FIELDS.indexOf(f);
              const ok = near(mine[i], f.model);
              return (
                <tr key={f.key} data-field={f.key} data-ok={ok ? 'yes' : 'no'}>
                  <td>{g.title.split(',')[0]}</td>
                  <td>{f.label}</td>
                  <td>{mine[i]}{f.unit === '%' ? '%' : ''}</td>
                  <td data-out="model">{f.model}{f.unit === '%' ? '%' : ''}</td>
                  <td className={ok ? 't-ok' : 't-fail'}>{ok ? 'matches' : 'differs'}</td>
                </tr>
              );
            }),
          )}
        </tbody>
      </table></Wide>
      <div className="mx-label">Line 4 in block 0: which lanes execute <code>b[i] = a[i] + 1</code></div>
      <LaneRows name="Line 4 in the four warps of block 0" width={E1.width} rows={statementRows('line4')} />
      <p>
        The test passes for threads 0 to 39 and 104 to 127 of every block. Warp 0 lies wholly inside the first range, warp 2 wholly outside
        both, and warps 1 and 3 are cut. So three warps of each block are active, 3 · 8 = {E1.line4.active}, and two of each are divergent, 2
        · 8 = {E1.line4.divergent}. Every block is the same because the test uses <code>threadIdx.x</code>, not <code>i</code>.
      </p>
      <div className="mx-label">Line 7 in block 0: which lanes execute <code>a[i] = b[i]*2</code></div>
      <LaneRows name="Line 7 in the four warps of block 0" width={E1.width} rows={statementRows('line7')} />
      <p>
        Even and odd <code>i</code> alternate lane by lane, so every one of the {E1.warps} warps has both paths in it and runs at{' '}
        {pct(w0.line7.efficiency)}. Half the lanes take the branch, and no arrangement of the launch can make a warp uniform: this is the worst
        pattern there is, and it costs exactly what a 31 to 1 split costs.
      </p>
      <div className="mx-label">The loop in warp 0 of block 0: one row per iteration</div>
      <LaneRows
        name="The loop of line 9 in warp 0 of block 0, one row per iteration"
        width={E1.width}
        head={w0.trips}
        headLabel="trip count"
        rows={loopRows(w0.trips).map((active, j) => {
          const c = countTaken(active);
          return { label: `j = ${j}`, active, note: c === E1.width ? `all ${E1.width}` : `${c} run, ${E1.width - c} wait` };
        })}
      />
      <p>
        Trip counts are 5, 4 and 3 as <code>i % 3</code> is 0, 1 and 2, and every warp holds all three. Iterations <code>j</code> = 0, 1 and 2
        are executed by every lane; in <code>j</code> = 3 a third of the lanes have finished, and in <code>j</code> = 4 two thirds have. That
        is {E1.loop.uniform} iterations without divergence and {E1.loop.divergent} with.
      </p>
      <p className="muted">
        The exercise assumes 32-wide warps. At 64 the same launch has {E1_WIDE.warps} warps, and line 4 splits all {E1_WIDE.line4.divergent} of
        them: {E1_WIDE.detail[0].line4.count} of 64 lanes execute it in the first warp of a block and {E1_WIDE.detail[1].line4.count} of 64 in
        the second.
      </p>
    </div>
  );
}

function Ex1Predict() {
  const { progress, answer } = useProgress();
  const prior = progress.answers[EX1_ID]?.value;
  const committed =
    Array.isArray(prior) && prior.length === EX1_FIELDS.length && prior.every((v) => typeof v === 'number') ? (prior as number[]) : null;
  const [text, setText] = useState<Record<string, string>>({});
  const values = EX1_FIELDS.map((f) => (text[f.key]?.trim() ? Number(text[f.key]) : NaN));
  const ready = values.every((v) => Number.isFinite(v));
  return (
    <div className="q" data-predict={EX1_ID}>
      <div className="kind">Predict first</div>
      <div className="prompt">
        The book's exercise 1, parts c, d and e, for the kernel and launch above on a device with 32-wide warps. Fill in all ten before you
        look: the reveal draws block 0's four warps for each statement.
      </div>
      {committed == null ? (
        <>
          {EX1_GROUPS.map((g) => (
            <div key={g.title} style={{ marginBottom: '0.8rem' }}>
              <div className="mx-label">{g.title}</div>
              {g.fields.map((f) => (
                <div className="row" key={f.key} style={{ marginBottom: '0.3rem', gap: '0.2rem 0.6rem' }}>
                  <span style={{ minWidth: '17rem' }}>{f.label}</span>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.5rem' }}>
                    <input type="number" step="any" value={text[f.key] ?? ''} aria-label={`${g.title}: ${f.label}`} data-field={f.key} style={{ width: '5.5rem' }}
                      onChange={(e) => setText({ ...text, [f.key]: e.target.value })} />
                    <span className="ui muted">{f.unit === '%' ? '%' : f.unit}</span>
                  </span>
                </div>
              ))}
            </div>
          ))}
          <button className="primary" disabled={!ready} onClick={() => ready && answer(EX1_ID, values)}>Commit prediction</button>
        </>
      ) : (
        <div className="verdict info">
          <p><b>You committed ten numbers.</b></p>
          <Ex1Reveal mine={committed} />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- the boundary-check counter
const STATE_FILL: Record<WarpState, string> = {
  full: 'color-mix(in srgb, var(--ink) 24%, var(--cell))',
  divergent: 'var(--ink)',
  idle: 'var(--cell)',
};
const STATE_WORDS: Record<WarpState, string> = {
  full: 'every thread passes the test',
  divergent: 'split: some threads pass and some do not',
  idle: 'no thread passes: launched, and idle',
};
const MAX_DRAWN_BLOCKS = 400;
const MAX_DRAWN_WARPS_1D = 640;

/** Every warp of a launch, drawn in place: bands inside each block (2D), or cells in a line (1D). */
function LaunchMap({ p, states, oneD }: { p: Picture; states: WarpState[][][]; oneD: boolean }) {
  const hatch = useId();
  const gridY = states.length;
  const gridX = states[0]?.length ?? 0;
  const perBlock = states[0]?.[0]?.length ?? 0;
  const defs = (
    <defs>
      <pattern id={hatch} width="4" height="4" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <line x1="0" y1="0" x2="0" y2="4" stroke="var(--ink-3)" strokeWidth="1" />
      </pattern>
    </defs>
  );
  const cell = (key: string, x: number, y: number, w: number, h: number, s: WarpState, attrs: Record<string, number>) => (
    <g key={key}>
      <rect x={x} y={y} width={w} height={h} data-state={s} {...Object.fromEntries(Object.entries(attrs).map(([k, v]) => [`data-${k}`, v]))}
        fill={STATE_FILL[s]} stroke="var(--rule)" strokeWidth="0.5" />
      {s === 'idle' ? <rect x={x} y={y} width={w} height={h} fill={`url(#${hatch})`} pointerEvents="none" /> : null}
    </g>
  );
  if (oneD) {
    const roomy = gridX * perBlock <= 64;
    const c = roomy ? 18 : 11;
    const h = roomy ? 22 : 16;
    const gap = 5;
    const perLine = Math.max(1, Math.floor(760 / (perBlock * c + gap)));
    const lines = Math.ceil(gridX / perLine);
    const W = Math.min(gridX, perLine) * (perBlock * c + gap);
    return (
      <svg viewBox={`0 0 ${W} ${lines * (h + 6) + 2}`} role="img" aria-label="Every warp of the launch in order, block by block" data-view="launch"
        style={{ width: W, maxWidth: '100%', display: 'block' }}>
        {defs}
        {states[0].map((block, bx) => {
          const x0 = (bx % perLine) * (perBlock * c + gap);
          const y0 = Math.floor(bx / perLine) * (h + 6) + 1;
          return block.map((s, k) => cell(`${bx}-${k}`, x0 + k * c, y0, c - 1, h, s, { block: bx, warp: k }));
        })}
      </svg>
    );
  }
  const S = Math.max(12, Math.min(84, Math.floor(700 / gridX)));
  const band = (S - 2) / perBlock;
  const ex = (p.width / p.blockX) * S;
  const ey = (p.height / p.blockY) * S;
  const W = gridX * S + 2;
  const H = gridY * S + 2;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Every block of the launch, each drawn as its warps from top to bottom, with the edge of the picture" data-view="launch"
      style={{ width: W, maxWidth: '100%', display: 'block' }}>
      {defs}
      {states.map((line, by) =>
        line.map((block, bx) => (
          <g key={`${bx}-${by}`}>
            {block.map((s, k) => cell(`${k}`, 1 + bx * S + 0.5, 1 + by * S + 0.5 + k * band, S - 2, band, s, { bx, by, warp: k }))}
            <rect x={1 + bx * S} y={1 + by * S} width={S - 1} height={S - 1} fill="none" stroke="var(--ink-3)" strokeWidth="0.8" />
          </g>
        )),
      )}
      <path d={`M ${1 + ex} 0 V ${Math.min(H, 1 + ey)} M 0 ${1 + ey} H ${Math.min(W, 1 + ex)}`} fill="none" stroke="var(--page)" strokeWidth="3.5" />
      <path d={`M ${1 + ex} 0 V ${Math.min(H, 1 + ey)} M 0 ${1 + ey} H ${Math.min(W, 1 + ex)}`} fill="none" stroke="var(--ink)" strokeWidth="1.5" strokeDasharray="5 3" />
    </svg>
  );
}

const WIDE = BOOK_CASES.find((c) => c.id === 'p200x150')!.picture;
const TALL = BOOK_CASES.find((c) => c.id === 'p150x200')!.picture;
const WIDE_R = regions(WIDE, BOOK_WIDTH);
const HUNT_PRINTED = 80;
const HUNT: { label: string; count: number }[] = [
  { label: 'Warps the test really splits, with 200 as the width and 150 as the height.', count: boundary2D(WIDE, BOOK_WIDTH).divergent },
  { label: 'Warps the test really splits, with 150 as the width and 200 as the height.', count: boundary2D(TALL, BOOK_WIDTH).divergent },
  { label: 'Every warp of every block the right edge crosses, split or not, with 200 as the width.', count: warpsInRightEdgeBlocks(WIDE, BOOK_WIDTH) },
  { label: 'Every warp of every block the right edge crosses, split or not, with 150 as the width.', count: warpsInRightEdgeBlocks(TALL, BOOK_WIDTH) },
  { label: 'Warps the test really splits on a device with 64-wide warps, with 200 as the width.', count: boundary2D(WIDE, 64).divergent },
];
const isInt = (v: number) => Number.isInteger(v) && v >= 1;

function BoundaryCounter() {
  const { reports } = useReports();
  const choices = useMemo(() => widthChoices(reports), [reports]);
  const [key, setKey] = useState('book');
  const chosen = choices.find((c) => c.key === key) ?? choices[0];
  const width = chosen.width;
  const [tab, setTab] = useState<'1d' | '2d'>('1d');
  const [v1, setV1] = useState({ n: 1003, block: 64 });
  const [v2, setV2] = useState<Picture>({ width: 76, height: 62, blockX: 16, blockY: 16 });
  const [huntPick, setHuntPick] = useState<number | null>(null);
  const [huntChecked, setHuntChecked] = useState(false);

  const oneD = tab === '1d';
  const p: Picture = oneD ? { width: v1.n, height: 1, blockX: v1.block, blockY: 1 } : v2;
  const valid = [p.width, p.height, p.blockX, p.blockY].every(isInt);
  const over = valid && p.blockX * p.blockY > chosen.maxThreads;
  const b = valid && !over ? boundary2D(p, width) : null;
  const drawable = b != null && (oneD ? b.warps <= MAX_DRAWN_WARPS_1D && b.warpsPerBlock <= 60 : b.blocks <= MAX_DRAWN_BLOCKS && b.warpsPerBlock <= 32);
  const states = useMemo(() => {
    if (!b || !drawable) return null;
    const g = gridFor(p);
    return classifyGrid(g.gridX, g.gridY, dims(p.blockX, p.blockY), width, (col, row) => col < p.width && row < p.height, true).states;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [p.width, p.height, p.blockX, p.blockY, width, drawable]);
  const split = oneD && b ? splitWarp1D(v1.n, v1.block, width) : null;

  const field = (name: string, value: number, set: (v: number) => void) => (
    <label className="field">
      {name}
      <input type="number" min={1} step={1} value={value || ''} aria-label={name} style={{ width: '6.4rem' }}
        onChange={(e) => set(e.target.value.trim() === '' ? 0 : Math.floor(Number(e.target.value)))} />
    </label>
  );
  const load = (c: Picture) => {
    if (c.height === 1 && c.blockY === 1) {
      setTab('1d');
      setV1({ n: c.width, block: c.blockX });
    } else {
      setTab('2d');
      setV2(c);
    }
  };

  return (
    <Bench
      title="Boundary-check counter"
      task="Set a size and a block. How many warps does the test split, and which ones?"
      foot={
        <>
          One thread per element, and the usual guard: <code>i &lt; n</code> in one dimension, <code>col &lt; width &amp;&amp; row &lt; height</code>{' '}
          in two. A warp is split when some of its threads pass the guard and some do not. {chosen.note}
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span className="seg" role="group" aria-label="Dimensions">
          <button aria-pressed={oneD} onClick={() => setTab('1d')}>1D: a vector</button>
          <button aria-pressed={!oneD} onClick={() => setTab('2d')}>2D: a picture or a matrix</button>
        </span>
        {oneD ? (
          <>
            {field('Elements n', v1.n, (v) => setV1({ ...v1, n: v }))}
            {field('Threads in a block', v1.block, (v) => setV1({ ...v1, block: v }))}
          </>
        ) : (
          <>
            {field('Width (columns)', v2.width, (v) => setV2({ ...v2, width: v }))}
            {field('Height (rows)', v2.height, (v) => setV2({ ...v2, height: v }))}
            {field('blockDim.x', v2.blockX, (v) => setV2({ ...v2, blockX: v }))}
            {field('blockDim.y', v2.blockY, (v) => setV2({ ...v2, blockY: v }))}
          </>
        )}
        <label className="field" style={NARROW_OK}>
          Warp width
          <select value={key} aria-label="Warp width of the launch" onChange={(e) => setKey(e.target.value)}>
            {choices.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
          </select>
        </label>
      </div>

      {!valid ? (
        <p><span className="chip fail">Every size is a whole number, 1 or more.</span></p>
      ) : over ? (
        <p data-over="">
          <span className="chip fail">{p.blockX * p.blockY} threads in a block</span> is more than the {chosen.maxThreads} this choice allows, so the
          launch would be refused.
        </p>
      ) : b ? (
        <>
          <div className="row" style={{ marginBottom: '0.7rem' }} data-readout="boundary">
            <span><b data-out="blocks">{big(b.blocks)}</b> {b.blocks === 1 ? 'block' : 'blocks'}{oneD ? '' : ` (${b.gridX} across, ${b.gridY} down)`}</span>
            <span><b data-out="threads">{big(b.threads)}</b> threads launched</span>
            <span><b data-out="warps">{big(b.warps)}</b> warps</span>
            <span><b data-out="divergent">{big(b.divergent)}</b> split</span>
            <span><b data-out="idle">{big(b.idle)}</b> idle</span>
            <span><b data-out="full">{big(b.full)}</b> full</span>
            <span><b data-out="share">{pct(b.share)}</b> of warps split</span>
          </div>
          {oneD ? (
            <div className="narrate" role="status" data-readout="split" style={{ marginBottom: '0.8rem', minHeight: '2.6em' }}>
              {split
                ? `The one warp that splits is warp ${split.warp} of block ${split.block}: ${plural(split.valid, 'thread passes', 'threads pass')} the test and ${split.invalid} ${split.invalid === 1 ? 'does' : 'do'} not. ${big(b.threads - v1.n)} of the ${big(b.threads)} threads launched have no element.`
                : `The last element is the last lane of a warp, so nothing splits. ${b.threads - v1.n > 0 ? `The ${big(b.threads - v1.n)} threads with no element fill whole warps, which stay idle.` : 'Every thread has an element.'}`}
            </div>
          ) : (
            <Wide style={{ marginBottom: '0.8rem' }}><table className="threads" style={{ width: 'auto' }} aria-label="The book's four regions">
              <thead>
                <tr><th>Region</th><th>blocks</th><th>valid columns × rows in a block</th><th>warps</th><th>split</th><th>idle</th></tr>
              </thead>
              <tbody>
                {b.regions.map((r) => (
                  <tr key={r.id} data-region={r.id}>
                    <td>{r.id}: {r.where}</td>
                    <td data-out="blocks">{r.blocks}</td>
                    <td>{r.blocks ? `${r.validX} × ${r.validY}` : 'none'}</td>
                    <td data-out="warps">{r.warps}</td>
                    <td data-out="divergent">{r.divergent}</td>
                    <td data-out="idle">{r.idle}</td>
                  </tr>
                ))}
              </tbody>
            </table></Wide>
          )}
          {states ? (
            <>
              <LaunchMap p={p} states={states} oneD={oneD} />
              <div className="row muted" style={{ margin: '0.5rem 0 0', gap: '0.3rem 1.1rem' }}>
                {(['full', 'divergent', 'idle'] as WarpState[]).map((s) => (
                  <span key={s} style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem' }}>
                    <i style={{ display: 'inline-block', width: '0.9em', height: '0.9em', background: STATE_FILL[s], border: '1px solid var(--ink-3)' }} />
                    {STATE_WORDS[s]}
                  </span>
                ))}
                {oneD ? <span>one cell per warp, grouped by block</span> : <span>each block is drawn as its warps, top to bottom; the dashed line is the edge of the data</span>}
              </div>
            </>
          ) : (
            <p className="muted" data-undrawn="">Too many warps to draw one by one. The counts above are exact.</p>
          )}
        </>
      ) : null}

      <div className="mx-label" style={{ marginTop: '1.1rem' }}>The book's examples, at the width chosen above</div>
      <Wide><table className="threads" style={{ width: 'auto' }} aria-label="The book's examples: the printed figure beside the model's">
        <thead>
          <tr><th>Case</th><th>Book p.</th><th>The book prints</th><th>The model counts</th><th></th></tr>
        </thead>
        <tbody>
          {BOOK_CASES.map((c) => {
            const m = boundary2D(c.picture, width);
            const model = `${m.divergent} of ${big(m.warps)}`;
            const differs = c.printed != null && width === BOOK_WIDTH && c.printed !== `${m.divergent} of ${m.warps}`;
            return (
              <tr key={c.id} data-case={c.id} data-differs={differs ? '' : undefined}>
                <td>{c.label}</td>
                <td>{c.page}</td>
                <td>{c.printed ?? 'no count (an exercise)'}</td>
                <td data-out="model">{model}{differs ? ' (the page differs)' : ''}</td>
                <td><button className="quiet" onClick={() => load(c.picture)}>Load</button></td>
              </tr>
            );
          })}
        </tbody>
      </table></Wide>

      <div className="q" style={{ marginTop: '1.1rem' }} data-hunt="">
        <div className="kind">Erratum hunt</div>
        <div className="prompt">
          The book says that for a “200 × 150” picture only 80 of the 1040 warps diverge. Load both readings above: neither count is 80. Which
          way of counting does give the printed {HUNT_PRINTED}?
        </div>
        <div className="opts" role="radiogroup">
          {HUNT.map((h, i) => (
            <label key={i} className={huntChecked && h.count === HUNT_PRINTED ? 'right' : huntChecked && huntPick === i ? 'wrong' : ''}>
              <input type="radio" name="ch04-hunt-80" checked={huntPick === i} onChange={() => { setHuntPick(i); setHuntChecked(false); }} />
              <span>
                {h.label}
                {huntChecked ? <span className="muted" data-hunt-count={h.count}> That count is {h.count}.</span> : null}
              </span>
            </label>
          ))}
        </div>
        <button className="primary" disabled={huntPick == null} onClick={() => setHuntChecked(true)}>Check</button>
        {huntChecked && huntPick != null ? (
          <div className={'verdict ' + (HUNT[huntPick].count === HUNT_PRINTED ? 'right' : 'wrong')}>
            <p><b>{HUNT[huntPick].count === HUNT_PRINTED ? 'That gives 80.' : `That gives ${HUNT[huntPick].count}, not 80.`}</b></p>
            <p>
              The printed region sizes (864, 72, 96 and 8 warps) are those of a picture 200 wide and 150 high. In that picture the right edge
              crosses {WIDE_R[1].blocks} full-height blocks, all of whose {WIDE_R[1].divergent} warps are split, and the corner block. The corner
              has {WIDE_R[3].validY} valid rows: {WIDE_R[3].divergent} of its warps are split and {WIDE_R[3].idle} lie below the picture, idle.{' '}
              {WIDE_R[1].divergent} + {WIDE_R[3].divergent} = {HUNT[0].count}. Counting all {WIDE_R[3].warps} warps of the corner gives{' '}
              {WIDE_R[1].divergent} + {WIDE_R[3].warps} = {warpsInRightEdgeBlocks(WIDE, BOOK_WIDTH)}. We cannot know what the authors counted; this is the simple count that reproduces their number. Either way the
              book's conclusion holds for this reading: {HUNT[0].count} or {HUNT_PRINTED} of 1040 is under 8%.
            </p>
          </div>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- lab 4.2's measurement
const LAB2 = 'ch04/lab2_divergence';
const DIVCOST_ID = 'ch04/4.5/p-divcost';
interface Lab2Result {
  backend: string;
  device: string;
  whose: string;
  width: number | undefined;
  /** branchKernel on "half of each warp's lanes", over the same mix sorted. */
  mixed: number;
  patterns: { pattern: string; ratio: number }[];
  /** selectKernel over branchKernel on the mixed pattern. */
  selectOverBranch: number | undefined;
  /** loopKernel with shuffled trip counts over the same counts sorted. */
  loopShuffled: number | undefined;
}
/** The rows lab 4.2's benchmark wrote on each real device, the reader's own kernels first. */
function lab2Results(reports: Reports): Lab2Result[] {
  return realBackends(reports)
    .filter((b) => b !== 'race')
    .flatMap((backend): Lab2Result[] => {
      const run = reports[backend];
      for (const variant of ['work', 'solution']) {
        const rows = benchRows(run, LAB2, variant);
        const isHalf = (r: BenchRow) => String(r.pattern ?? '').startsWith('half of each');
        const mixed = num(rows.find((r) => r.name === 'branch' && isHalf(r))?.ratio_to_sorted);
        if (mixed == null) continue;
        return [{
          backend,
          device: String(run.device.name ?? backend),
          whose: variant === 'work' ? 'your kernels' : 'the reference solution',
          width: num(rows.find((r) => r.name === 'branch')?.width),
          mixed,
          patterns: rows.filter((r) => r.name === 'branch').flatMap((r) => {
            const ratio = num(r.ratio_to_sorted);
            return ratio == null ? [] : [{ pattern: String(r.pattern), ratio }];
          }),
          selectOverBranch: num(rows.find((r) => r.name === 'select' && isHalf(r))?.ratio_to_branch),
          loopShuffled: num(rows.find((r) => r.name === 'loop' && r.order === 'shuffled')?.ratio_to_sorted),
        }];
      }
      return [];
    });
}
// What the pass count of this page says for long, equal paths mixed half and half.
const MODEL_MIXED = run(ifElse(BOOK_WIDTH, maskBelow(BOOK_WIDTH, BOOK_WIDTH / 2), 1000, 1000)).overhead;

function MeasuredDivergence({ predicted }: { predicted: number }) {
  const { reports, loading, reload } = useReports();
  const found = lab2Results(reports);
  if (!found.length)
    return loading ? (
      <p data-measured="loading">Reading reports.</p>
    ) : (
      <p data-measured="none">
        No measurement on this machine yet. Do lab 4.2 on the <a href={href('ch04', 'labs')}>labs page</a>. Once your kernels pass, run{' '}
        <code>python3 labs/pmpp.py run 4.2 --bench</code>: it prints the table and records it where this page reads it. Then come back:
        this box will set your prediction beside the result.{' '}
        <button className="quiet" onClick={reload}>Reload results</button>
      </p>
    );
  return (
    <>
      {found.map((f) => (
        <div key={f.backend} data-measured={f.backend} data-mixed={fmt(f.mixed)}>
          <p>
            On <b>{f.device}</b> ({f.backend}), with {f.whose}{f.width ? ` and a width of ${f.width}` : ''}: the pattern with both paths in every
            warp took <b>{fmt(f.mixed)}</b> times as long as the same mix sorted.{' '}
            {predicted > f.mixed * 1.25 ? 'You expected more.' : predicted < f.mixed / 1.25 ? 'You expected less.' : 'That is close to what you predicted.'}
          </p>
          <Wide style={{ marginBottom: '0.6rem' }}><table className="threads" style={{ width: 'auto' }} aria-label={`Lab 4.2 on ${f.backend}: every pattern`}>
            <thead>
              <tr><th>Pattern</th><th>time ÷ time of the same mix, sorted</th></tr>
            </thead>
            <tbody>
              {f.patterns.map((r) => <tr key={r.pattern} data-pattern={r.pattern}><td>{r.pattern}</td><td>{fmt(r.ratio)}</td></tr>)}
            </tbody>
          </table></Wide>
          <p>
            {f.selectOverBranch != null ? <>Computing both paths and selecting took <b data-out="select">{fmt(f.selectOverBranch)}</b> times as long as the branch on that mixed pattern. </> : null}
            {f.loopShuffled != null ? <>The loop kernel with shuffled trip counts took <b data-out="loop">{fmt(f.loopShuffled)}</b> times as long as with the same counts sorted.</> : null}
          </p>
          {f.backend === 'metal' ? (
            <p className="muted">
              On Metal the patterns were laid out from what a probe kernel reported about SIMD-groups, and Apple documents no model of
              divergence to compare this with.
            </p>
          ) : null}
        </div>
      ))}
      <p className="muted">
        These are measurements from your machine, not expected values. The pass count of this page says {fmt(MODEL_MIXED, 2)} for long, equal
        paths: two passes where a uniform warp makes one. A device can come out lower or higher than its pass count.
      </p>
    </>
  );
}

// ---------------------------------------------------------------- questions
const EX3 = boundary1D(2000, 512, BOOK_WIDTH);
const EX3_SPLIT = splitWarp1D(2000, 512, BOOK_WIDTH)!;
const TALL_B = boundary2D(TALL, BOOK_WIDTH);
const WIDE_B = boundary2D(WIDE, BOOK_WIDTH);
const CAUSAL = classifyGrid(1024 / 32, 1024 / 8, dims(32, 8), BOOK_WIDTH, causal);
const Q_TRIPS = shuffledTrips(8 * BOOK_WIDTH, TRIP_LO, TRIP_HI, 7);
const Q_SHUFFLED = loopBatch(Q_TRIPS, BOOK_WIDTH);
const Q_SORTED = loopBatch(sortedTrips(Q_TRIPS), BOOK_WIDTH);
const Q_BRANCH = run(ifElse(BOOK_WIDTH, maskModulo(BOOK_WIDTH, 2), 100, 100));
const Q_SELECT = run(select(BOOK_WIDTH, 100, 100));
const Q_UNIFORM = run(ifElse(BOOK_WIDTH, maskBelow(BOOK_WIDTH, BOOK_WIDTH), 100, 100));
const AMD64 = [0, 1].map((k) => countTaken(maskBelow(64, 32, k * 64)));

export const questionsS45: Record<string, Question> = {
  'ch04/4.5/q-ex2': {
    kind: 'numeric',
    id: 'ch04/4.5/q-ex2',
    prompt: <>The book's exercise 2. For vector addition, the vector has 2000 elements and each block has 512 threads. How many threads are in the grid?</>,
    answer: EX3.threads,
    unit: 'threads',
    explain: (
      <p>
        A grid is whole blocks: ⌈2000 / 512⌉ = {EX3.blocks} blocks of 512, so {EX3.threads} threads. {EX3.threads - 2000} of them have no element
        and are stopped by the boundary check. The book prints no answers; this one is ours.
      </p>
    ),
    source: 'ch. 4 exercise 2, p. 91',
  },
  'ch04/4.5/q-ex3': {
    kind: 'choice',
    id: 'ch04/4.5/q-ex3',
    prompt: <>The book's exercise 3. In that launch (2000 elements, 512-thread blocks, warps 32 wide), how many warps diverge at the boundary check?</>,
    options: [
      { label: '2: the last two warps both contain threads with no element to work on.', why: 'The last warp contains only such threads. All of them fail the test, so they all take the same path.' },
      { label: `${EX3.divergent}: the warp where the data ends, ${EX3_SPLIT.valid} threads in range and ${EX3_SPLIT.invalid} out.`, correct: true },
      { label: '0: every thread runs the same check, so every warp follows one path.', why: 'The statement is the same; its outcome is not. One warp has threads on both sides of 2000.' },
      { label: '48: each thread with no element takes the other path on its own.', why: 'Divergence is counted in warps: a warp either has more than one path in it or it does not.' },
    ],
    explain: (
      <p>
        2000 = 62 · 32 + 16. Warps 0 to 61 of the grid hold threads 0 to 1983, all in range. Warp 62 holds threads 1984 to 2015: 16 pass the
        test and 16 fail, so it diverges. Warp 63 holds 2016 to 2047, every one out of range: one path, no divergence, and {EX3.idle} warp of
        hardware that does nothing. A one-dimensional boundary can split at most one warp, whatever the size. The answer is ours.
      </p>
    ),
    source: 'ch. 4 exercise 3, p. 91; notes/today/ch04-facts.md E1',
  },
  'ch04/4.5/q-200x150': {
    kind: 'numeric',
    id: 'ch04/4.5/q-200x150',
    prompt: (
      <>
        A picture is 150 pixels wide and 200 high. It is processed with 16 × 16 blocks, one thread per pixel, on a device with 32-wide warps.
        How many warps diverge at the boundary check?
      </>
    ),
    answer: TALL_B.divergent,
    unit: 'warps',
    explain: (
      <p>
        150 = 9 · 16 + 6 and 200 = 12 · 16 + 8, so the grid is 10 blocks across and 13 down: {TALL_B.blocks} blocks, {big(TALL_B.warps)} warps. A
        warp is two rows of a block. The right edge leaves 6 valid columns of 16, so it splits every warp of the 12 full-height blocks it
        crosses: 12 · 8 = {TALL_B.regions[1].divergent}. The bottom edge leaves 8 valid rows, which is 4 whole warps in range and 4 out of range:
        none split. The corner block has 4 split warps. {TALL_B.regions[1].divergent} + {TALL_B.regions[3].divergent} = {TALL_B.divergent} of{' '}
        {big(TALL_B.warps)}, or {pct(TALL_B.share)}. That is why the book's “under 8%” needs the other reading: 200 wide and 150 high gives{' '}
        {WIDE_B.divergent}, or {pct(WIDE_B.share)}.
      </p>
    ),
    source: 'ch. 4 §4.5, p. 82; notes/today/ch04-facts.md E1',
  },
  'ch04/4.5/q-amd32': {
    kind: 'choice',
    id: 'ch04/4.5/q-amd32',
    prompt: (
      <>
        A kernel contains <code>if (threadIdx.x &lt; 32) {'{ ... }'} else {'{ ... }'}</code> and runs in 128-thread blocks on a device whose warps
        are 64 wide. Which warps diverge at it?
      </>
    ),
    options: [
      { label: 'None: 32 is a warp boundary, so each warp falls on one side of the test.', why: 'It is a warp boundary where warps are 32 wide. Here the first boundary is at 64.' },
      { label: 'Both warps of every block: the threshold sits inside the block, so both see it.', why: 'Warp 1 holds threads 64 to 127. None is below 32, so they all take the else path together.' },
      { label: `Warp 0 of every block, ${AMD64[0]} lanes each way; warp 1 fails the test whole.`, correct: true },
      { label: 'Warp 1 of every block: it holds the threads that come after the threshold.', why: 'Threads 0 to 31 and 32 to 63 are the ones that disagree, and they share warp 0.' },
    ],
    explain: (
      <p>
        At 64 wide, a 128-thread block is two warps: threads 0 to 63 and 64 to 127. The threshold 32 falls in the middle of the first. This
        statement is AMD's own example of a branch with no divergence, and it is divergence-free only on devices with 32-wide warps. A
        threshold written as a multiple of <code>warpSize</code> would hold on both.
      </p>
    ),
    source: 'notes/today/ch04-facts.md B7',
  },
  'ch04/4.5/q-buckets': {
    kind: 'choice',
    id: 'ch04/4.5/q-buckets',
    prompt: (
      <>
        A kernel gives each sequence of a batch one thread, and each thread loops up to its own sequence's length. You can launch it with the
        batch in arrival order or sorted by length. The total number of loop iterations is the same. Which is faster, and why?
      </>
    ),
    options: [
      { label: 'Neither: the total number of iterations is the same, so the work and the time are the same.', why: 'The same useful iterations, yes. What changes is how long finished lanes sit in a warp that is still running.' },
      { label: 'Arrival order: mixing long and short threads in every warp balances the load across warps.', why: 'Balance between warps does not help a lane that has finished. It waits for its own warp.' },
      { label: 'Sorted: grouping equal lengths together lets the launch use fewer warps for the batch.', why: 'The number of warps is threads divided by the width, in either order.' },
      { label: 'Sorted: a warp runs as long as its longest thread, and sorting makes a warp\'s lengths nearly equal.', correct: true },
    ],
    explain: (
      <p>
        This is the loop of the book's Fig. 4.10 at the scale of a batch. In the batch drawn on this page ({Q_TRIPS.length} threads, lengths{' '}
        {TRIP_LO} to {TRIP_HI}, 32-wide warps) both orders do {big(Q_SORTED.useful)} lane-iterations of work; sorted, the warps run for{' '}
        {Q_SORTED.spent} iterations in total, and shuffled for {Q_SHUFFLED.spent}. Bucketing by length is something you already do to cut
        padding; here it also makes warps uniform. The link to batching is ours, and so is the count; lab 4.2 measures the same thing on a
        device.
      </p>
    ),
    source: 'ch. 4 §4.5, p. 81 (Fig. 4.10); the batching example is ours',
  },
  'ch04/4.5/q-causal': {
    kind: 'choice',
    id: 'ch04/4.5/q-causal',
    prompt: (
      <>
        A kernel fills a T × T matrix of attention scores, one thread per score, with T = 1024. Each thread tests <code>col &lt;= row</code>{' '}
        and takes one path for kept scores and another for masked ones. Blocks are 32 wide and 8 high and warps are 32 wide. Roughly what
        share of the warps diverge at the test?
      </>
    ),
    options: [
      { label: 'About 50%: half of the scores are masked, so half of the warps take the other path.', why: 'Half of the threads take each path. But most warps lie wholly on one side of the diagonal.' },
      { label: `About ${Math.round(CAUSAL.share * 100)}%: in each row, only the warp that straddles the diagonal.`, correct: true },
      { label: 'All of them: the test depends on each thread\'s row and column, so it varies in every warp.', why: 'Depending on the index makes divergence possible. It happens only in a warp whose threads disagree.' },
      { label: 'None: blocks are 32 wide, so the diagonal falls on block edges and cuts no warp.', why: 'It crosses each row once, inside some warp, except where it ends exactly on a warp boundary.' },
    ],
    explain: (
      <p>
        With blocks 32 wide, a warp is 32 consecutive columns of one row. In row <i>r</i> the test changes from true to false after column{' '}
        <i>r</i>, so exactly one warp of the row can be split: the one holding column <i>r</i>. It is split unless <i>r</i> is that warp's last
        column, which happens in 1 row of 32. That leaves 1024 − 32 = {CAUSAL.divergent} split warps out of {big(CAUSAL.warps)}, or{' '}
        {pct(CAUSAL.share)}. The method is the book's count for the right edge of a picture, applied to a diagonal; the example is ours.
      </p>
    ),
    source: 'method: ch. 4 §4.5, p. 82; the example is ours',
  },
  'ch04/4.5/q-ex5': {
    kind: 'choice',
    id: 'ch04/4.5/q-ex5',
    prompt: (
      <>
        The book's exercise 5. A programmer says: “If I launch with only 32 threads in each block, I can leave out the{' '}
        <code>__syncthreads()</code> calls.” Is that a good idea?
      </>
    ),
    options: [
      { label: 'Yes: a 32-thread block is one warp, and a warp issues one instruction for all its threads, so they cannot drift apart.', why: 'That was a description of NVIDIA hardware before Volta, at one particular width. It is not something the programming model promises.' },
      { label: 'Yes: with one warp in a block a block-wide barrier has nothing to wait for, so it compiles to nothing anyway.', why: 'Nothing documents that, and the argument would still depend on the width being 32.' },
      {
        label: 'No: it assumes that a block is exactly one warp and that a warp stays in step, and neither holds on every device.',
        correct: true,
      },
      { label: 'No: a block of 32 threads is spread over two half-warps, so the barrier is still needed between them.', why: 'At a width of 32 it is one warp. The trouble is relying on that.' },
    ],
    explain: (
      <>
        <p>
          Three things are wrong with it. Section 4.4 began with the rule that inside a block you assume any order and state every
          dependence with a barrier. The width is an implementation detail the book itself expects to change, and it is already 64 on
          some AMD devices. And since Volta, NVIDIA's guide says code that assumes lockstep within a warp is invalid and should be
          synchronised explicitly.
        </p>
        <p>
          There is a cost on top: on the book's example device, 32-thread blocks cap occupancy at 50%, as{' '}
          <a href={href('ch04', '4.7')}>section 4.7</a> works out. The answer is ours; the book prints none.
        </p>
      </>
    ),
    source: 'ch. 4 exercise 5, p. 91; §4.4 p. 74, §4.5 p. 81, §4.7 pp. 85–86; notes/today/ch04-facts.md A10',
  },
  'ch04/4.5/q-select': {
    kind: 'choice',
    id: 'ch04/4.5/q-select',
    prompt: (
      <>
        Both paths of a branch are loops of about 100 instructions, and in every warp half of the lanes take each path. You replace the
        branch with “compute both results, then select one”. What do you gain?
      </>
    ),
    options: [
      { label: 'Nothing to speak of: every lane now runs both paths, about what the split warp already spent.', correct: true },
      { label: 'A factor of two: the second pass disappears, so the warp makes one pass instead of two.', why: 'The second pass disappears and comes back as the second half of a pass twice as long.' },
      { label: 'A large factor: a divergent branch is expensive on a GPU, and a select costs one instruction.', why: 'The select is cheap. Computing both of its inputs is what costs.' },
      { label: 'It gets twice as slow: every lane now does the work of both paths instead of one.', why: 'Twice as slow as a uniform warp, yes. Compared with the divergent warp it replaces, about the same.' },
    ],
    explain: (
      <p>
        Count warp-steps in this page's model: the divergent warp spends {Q_BRANCH.spent} (the test, 100 for A, 100 for B, then C); the
        select version spends {Q_SELECT.spent} and no lane waits, but half of every lane's arithmetic is thrown away. A uniform warp spends{' '}
        {Q_UNIFORM.spent}. In this model a select never costs less than the split branch it replaces, at any length of path: it adds the
        select. Whether it wins for paths of a few instructions depends on what a branch costs beyond its passes, which the model leaves out
        and no vendor documents; that would be our claim, and it is untested. For long paths the remedy is to stop mixing them inside a warp.
        AMD's guide lists both remedies, and lab 4.2 times the select against the branch on your device.
      </p>
    ),
    source: 'notes/today/ch04-facts.md A9, B7; the comparison is ours',
  },
};

// ---------------------------------------------------------------- the page
const V1003 = boundary1D(1003, 64, BOOK_WIDTH);
const V1003_SPLIT = splitWarp1D(1003, 64, BOOK_WIDTH)!;
const V10000 = boundary1D(10000, 64, BOOK_WIDTH);
const P76: Picture = { width: 76, height: 62, blockX: 16, blockY: 16 };
const P76_32 = boundary2D(P76, BOOK_WIDTH);
const P76_64 = boundary2D(P76, 64);
const P76_R = regions(P76, BOOK_WIDTH);

export function S45() {
  return (
    <>
      <p>
        A warp has one instruction stream. That is the whole economy of <a href={href('ch04', '4.4')}>section 4.4</a>: one fetch, every
        lane. It works for as long as every thread of the warp wants the same next instruction. This section is about what happens when
        they do not, how to count it, and when the count is large enough to care about.
      </p>
      <Source section="4.5" pages="79" />

      <h2>One stream, several paths</h2>
      <p>
        Put an <code>if</code> in a kernel and some threads of a warp may take it while others take the <code>else</code>. The hardware
        still issues one instruction at a time to the warp, so it visits the paths one after another. It makes a pass over the first path
        in which only the threads that chose it take effect, then a pass over the other path for the rest. Threads of one warp following
        different paths is what the book calls <b>control divergence</b>.
      </p>
      <Source section="4.5" pages="79–80" />
      <p>
        Two things follow, and they pull in opposite directions. Every thread still does exactly what the source says for it: the passes
        are how SIMD hardware keeps the promise that you write one scalar thread. And the warp pays for it twice: it makes more passes
        than a uniform warp would, and in each pass the lanes of the threads on the other path are occupied and idle.
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md A9)">
        <p>
          NVIDIA's guide calls it warp divergence or branch divergence and describes the same mechanism: the warp “executes each branch
          path taken, disabling threads that are not on that path”. It gives no figure for what a pass costs.
        </p>
      </Today>
      <p>
        If you have tuned CPU code, turn one habit around. A CPU branch is cheap when its outcome is regular over time, so that the
        predictor guesses it. A GPU branch is cheap when it goes the same way in every lane of the warp at the moment it is reached. What
        matters is agreement across neighbours, not regularity over time.
      </p>
      <Aside>
        <p>
          AMD's HIP guide draws this contrast as temporal against spatial uniformity (notes/today/ch04-facts.md B7). The wording here is
          ours.
        </p>
      </Aside>
      <p>
        Divergence is a sequence of events, so watch it happen once before you count anything. The animation follows one warp of the
        book's 32 lanes through a branch, a loop, and the question of what order the passes come in.
      </p>
      <Simplification>
        <p>
          Every row of the animation is one instruction issued to the whole warp, and it counts one step whether 32 lanes execute it or
          one. That is the book's picture of the mechanism. It is not a measured cost on any device.
        </p>
      </Simplification>
      <WarpBranchScene />

      <h2>An if-else, pass by pass</h2>
      <p>
        The book's example is a warp of threads 0 to 31 reaching <code>if (threadIdx.x &lt; 24)</code>. In one pass threads 0 to 23 execute
        A while 24 to 31 are inactive; in another, threads 24 to 31 execute B while 0 to 23 are inactive. After that the threads may come
        back together and execute C.
      </p>
      <Source section="4.5" pages="80" figs="Fig. 4.9" />
      <p>
        The simulator below opens on that case. It counts in <b>warp-steps</b>: one instruction issued to the warp is one step, whatever the
        number of lanes that execute it. Beside what the warp spent it shows what the same work would take if no lane ever waited, which is
        the work in lane-steps divided by the width.
      </p>
      <Simplification>
        <p>
          Every instruction costs one warp-step, however few lanes run it: the book's picture, not a measured cost (facts A9, B7, C4). A
          branch's own work, and a loop's test and jump, cost nothing here.
        </p>
      </Simplification>
      <DivergenceSim />
      <p>Four things to find before you read on. Predict each, then try it.</p>
      <ol>
        <li>
          Slide T from 24 to 16, then to 1. The split goes from 24 against 8 to 1 against 31. What happens to the number of passes and to
          the warp-steps spent?
        </li>
        <li>
          Slide T to 32, then look at the other warps of the block. Then set the width to 64 and look again.
        </li>
        <li>Switch the order of the passes to interleaved. Which readouts change?</li>
        <li>
          Set A and B to 1 instruction each, then to 8 each, and compare the branch with “compute both and select” at both sizes: first
          at T = 24, then at T = 32.
        </li>
      </ol>
      <p>
        What you should have found: the cost follows the number of different paths inside one warp, not how many lanes take each. One lane
        out of step costs the warp a whole extra pass. A threshold that falls on a warp boundary costs nothing, because each warp is then
        uniform; and where the boundaries are depends on the width. Selecting never comes out ahead in this model: in a split warp it costs
        one step more than the branch, the select itself, at any length of path, and in a uniform warp it also pays for the path the branch
        skips.
      </p>

      <h2>Loops: lanes drop out</h2>
      <p>
        A loop diverges when its trip count differs between the threads of a warp. In the book's figure each thread reads its own count
        from data, <code>N = a[threadIdx.x]</code>, and the counts range from four to eight. For the first four iterations every thread is
        active. From the fifth on, threads that have finished go inactive while the rest continue, until only the longest are left.
      </p>
      <Source section="4.5" pages="81" figs="Fig. 4.10" />
      <p>
        Choose the loop in the simulator's statement list. Its trip counts start as the figure's eight printed values, repeated across the
        warp; click a column to change a lane. The warp runs for as many iterations as its longest lane, so a single long thread sets the
        cost for all 32. Try the preset with one lane of 10 and the rest 2, and read the share of lane-steps that did work.
      </p>
      <Aside>
        <p>
          The figure and its text disagree. The columns are labelled 0, 1, 2, 4, 5, 6, 7 (there is no 3), and the text says that in the
          eighth iteration only threads 1 and 7 run, where the printed labels put the two counts of 8 under 0 and 7. The simulator uses
          the printed counts, 8, 6, 7, 4, 5, 6, 8, 7, and numbers the lanes itself.
        </p>
      </Aside>

      <h2>In what order the passes run</h2>
      <p>
        So far the passes came one after the other. The book dates that: on NVIDIA's Pascal architecture and earlier, one path runs to
        completion and then the other. From Volta on, the passes may be interleaved, a feature called independent thread scheduling. The
        simulator's order switch shows the consequence for counting, which is none: the same instructions are issued in a different order.
      </p>
      <Source section="4.5" pages="80–81" />
      <p>
        The consequence for correctness is real. You cannot assume that the threads of a warp have come back together after the two paths.
        If every thread of the warp must finish a phase before any goes on, say so with a warp-level barrier,{' '}
        <code>__syncwarp()</code>. Fig. 4.9 draws C as one pass with all lanes active; the text says only that the threads “may” reconverge, so
        read the drawing as something that can happen, not as a guarantee.
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md A10, B5, C2)">
        <p>
          NVIDIA: the passes may interleave from compute capability 7.0, and its current guide starts at 7.5. AMD documents lockstep,
          both branches serialised. Apple does not say.
        </p>
      </Today>
      <p>
        The barrier is also a place where one source file stops serving two backends. <code>__syncwarp()</code> with no argument compiles
        on CUDA and on HIP. CUDA's form with a mask takes a 32-bit <code>unsigned</code>; HIP's takes only a 64-bit integer and rejects{' '}
        <code>__syncwarp(0xFFFFFFFF)</code> at compile time. Metal's counterpart is <code>simdgroup_barrier</code>, which takes memory flags
        and no mask.
      </p>
      <Beyond>
        <p>
          The book names <code>__syncwarp()</code> and uses CUDA only. The HIP header and the Metal function are from
          notes/today/ch04-facts.md B4 and C3. HIP's <code>__syncwarp</code> is in its header and not on its language-extensions page.
        </p>
      </Beyond>
      <Ask q={questionsS45['ch04/4.5/q-ex5']} />

      <h2>Spotting divergence in source</h2>
      <p>
        Look at the condition. If its value can differ between the threads of a warp, the statement can diverge. The book's rule names
        conditions that depend on <code>threadIdx</code>: <code>if (threadIdx.x &gt; 2)</code> splits the first warp of every block into
        threads 0, 1 and 2 against the rest, and a loop bounded by a thread index can diverge the same way. Fig. 4.10 shows that per-thread
        data does it too.
      </p>
      <Source section="4.5" pages="81" />
      <p>
        The book's first exercise asks you to count this for a short kernel, using three terms the chapter never defines. This page uses
        them as follows. A warp is <b>active</b> at a statement when at least one of its threads executes it. It is <b>divergent</b> there
        when some of its threads execute it and some do not. Its <b>SIMD efficiency</b> at the statement is the number of threads that
        execute it divided by the width of the warp.
      </p>
      <Beyond title="Our definitions">
        <p>
          “SIMD efficiency” and “active” appear only in exercise 1 (p. 91). AMD's guide uses the first term once without defining it and
          defines a different metric, branch efficiency (facts B7). The definitions here are ours.
        </p>
      </Beyond>
      <p>
        Here is the exercise's kernel and its launch: <code>N = 1024</code> elements in blocks of 128 threads, so {E1.blocks} blocks of{' '}
        {E1.warpsPerBlock} warps at a width of 32, as you counted in section 4.4. Line 4, line 7 and the loop on line 9 are three different
        kinds of condition.
      </p>
      <Code
        tabs={[{ label: 'foo_kernel.cu', code: FOO, status: 'front-end-checked; never run' }]}
        highlight={[3, 4, 6, 7, 9, 10]}
        caption={
          <>
            The book's exercise 1 (pp. 90–91), stored as <code>labs/ch04/snippets/foo_kernel.cu</code>. It has no boundary check because N is
            a multiple of the block size.
          </>
        }
      />
      <Ex1Predict />

      <h2>Boundary checks, the common case</h2>
      <p>
        Most divergence in real kernels comes from one source. A grid is a whole number of blocks, the data is whatever size it is, and
        so the kernel tests whether its thread has an element at all: the <code>if (i &lt; n)</code> of vector addition. The warp that
        contains the end of the data has threads on both sides of the test.
      </p>
      <Source section="4.5" pages="81" />
      <p>
        Take the book's numbers: a vector of 1003 elements in blocks of 64 threads. That is {V1003.blocks} blocks and {big(V1003.threads)}{' '}
        threads, of which the last {V1003_SPLIT.invalid} must do nothing. The {V1003.blocks} blocks are {V1003.warps} warps, and only one of
        them, the second warp of the last block, has threads on both sides. The other {V1003.warps - 1} are uniform.
      </p>
      <Source section="4.5" pages="81–82" />
      <p>
        So the damage is one warp, and what it amounts to depends on how many warps there are. With 100 elements one warp of four
        diverges, which can matter. With 1000 it is one of 32, about 3%: even if that warp took twice as long, the whole launch would take
        about 3% longer. With 10,000 it is one of some three hundred. Boundary divergence is a fixed cost that large data dilutes.
      </p>
      <Source section="4.5" pages="82" />
      <p>
        In two dimensions the edge is longer. The book returns to chapter 3's picture, 76 pixels wide and 62 high, covered by 16 × 16
        blocks: {P76_32.blocks} blocks, each of 8 warps, each warp two rows of its block. It sorts the blocks into four regions. The{' '}
        {P76_R[0].blocks} blocks wholly inside the picture do not diverge. The {P76_R[1].blocks} blocks the right edge cuts have 12 valid
        and 4 invalid columns in every row, so all {P76_R[1].divergent} of their warps diverge. The {P76_R[2].blocks} blocks the bottom edge
        cuts do not diverge at all: the edge falls between two warps, and the warp below it is wholly outside. The corner block has seven
        divergent warps and one outside. That is {P76_32.divergent} of {P76_32.warps}.
      </p>
      <Source section="4.5" pages="82" />
      <p>
        The bottom edge got lucky: 62 is even, and a warp is two rows. The book notes that an odd height would split those warps too.
        Count it yourself below, then widen the picture and watch the share fall.
      </p>
      <Simplification>
        <p>
          The counter applies the cut NVIDIA documents, with the width you choose. AMD documents the linear order and the lane arithmetic
          but not the cut in one sentence, and Metal's cut is implementation-defined (facts B6, C2). It counts split warps. It says nothing
          about what a split warp costs, which depends on what the kernel does after the test.
        </p>
      </Simplification>
      <BoundaryCounter />
      <p>What the counter shows, in the order you are likely to meet it:</p>
      <ul>
        <li>In one dimension at most one warp is split, at any size, and none when the data ends on a warp boundary.</li>
        <li>
          In two dimensions with warps that are whole rows, a right edge splits every warp it crosses, while a bottom edge splits a warp
          only if it falls inside one. The split warps are roughly one block-column out of all the columns, so the share falls as the
          picture gets wider. The book's rule of thumb is under 2% for pictures more than 1000 pixels wide.
        </li>
        <li>
          Two of the book's printed figures do not survive a recount. Its “one of 313” warps for 10,000 elements is one of{' '}
          {V10000.warps} launched: 313 warps hold an element and one more is launched and idle. And its 80 of 1040 for the “200 × 150”
          picture is {WIDE_B.divergent} if 200 is the width and {TALL_B.divergent} if 150 is. Neither changes the book's conclusion for
          large data.
        </li>
      </ul>
      <Aside>
        <p>
          These recounts are ours (notes/ch04.md, “Unclear in the PDF” item 4; notes/today/ch04-facts.md E1). The book writes sizes as
          vertical × horizontal, which would make “200 × 150” 150 wide; its printed region sizes match 200 wide.
        </p>
      </Aside>
      <Ask q={questionsS45['ch04/4.5/q-ex2']} label="Work it out" />
      <Ask q={questionsS45['ch04/4.5/q-ex3']} />
      <Ask q={questionsS45['ch04/4.5/q-200x150']} label="Work it out" />

      <h2>The width changes the two-dimensional counts</h2>
      <p>
        Every count above assumed 32-wide warps. Switch the counter to 64 and the one-dimensional cases keep their single split warp, but
        the picture changes: a 16 × 16 block is now four warps of four rows each, the bottom edge at row 62 falls inside a warp, and the
        same picture has {P76_64.divergent} split warps of {P76_64.warps} where it had {P76_32.divergent} of {P76_32.warps}. A boundary
        that was free at one width is not at the other.
      </p>
      <Beyond>
        <p>
          The book counts at 32 only. The 64-wide counts are this page's model applied with the width HIP documents for gfx9 devices
          (notes/today/ch04-facts.md E1, B6).
        </p>
      </Beyond>
      <p>
        The same trap is in vendor documentation. AMD's performance guide shows <code>if (threadIdx.x &lt; 32)</code> as a branch with no
        divergence. Where a warp is 32 threads, that holds: the test falls on a warp boundary. AMD's own Instinct devices have 64-wide
        warps. Set the simulator to 64 lanes and T = 32 and look at warp 0.
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md B7)">
        <p>
          AMD's example is marked “Good: no divergence”, with the comment “All threads in first half-warp execute”. That it diverges on a
          64-wide device is our reading, by the arithmetic of the cut.
        </p>
      </Today>
      <Ask q={questionsS45['ch04/4.5/q-amd32']} />

      <h2>On Metal there may be no boundary to check</h2>
      <p>
        Metal offers two ways to launch. <code>dispatchThreadgroups</code> launches whole threadgroups, like a CUDA grid, and the kernel
        needs the same boundary check. <code>dispatchThreads</code> takes the exact size of the data and launches no thread outside it:
        the threadgroups on the edge are simply smaller. With that call there is no boundary check, so there is nothing for an edge to
        diverge on.
      </p>
      <Beyond>
        <p>
          Apple: “You don't need to check a thread's position in a grid if you use the dispatchThreads(_:threadsPerThreadgroup:)
          technique.” (notes/today/ch05-facts.md C5; ch04-facts.md C4). PyTorch's MPS backend launches this way (facts D4).
        </p>
      </Beyond>
      <p>
        That removes the commonest cause, not the mechanism. A Metal kernel with a data-dependent branch still has threads of one
        SIMD-group wanting different instructions, and here the documentation stops: Apple's specification says a thread can be inactive
        “due to flow control” and describes neither passes nor their cost. On Metal the picture of this section is a hypothesis, and lab
        4.2 is how you test it on your own device.
      </p>
      <Today checked="2026-11-20 (notes/today/ch04-facts.md C4)">
        <p>
          The multi-pass model is documented by NVIDIA and by AMD. No Apple source read for this course describes how divergent paths in a
          SIMD-group execute.
        </p>
      </Today>

      <h2>Where you have met both shapes</h2>
      <p>
        Both figures of this section are already in the models you run. A kernel that gives each sequence of a batch one thread and loops
        to that sequence's length is Fig. 4.10: trip counts from data. A causal mask, <code>col &lt;= row</code>, is Fig. 4.9: an{' '}
        <code>if</code> on the thread's position. The second is cheap for the reason boundary checks are cheap, which the question below
        has you count. The first is not cheap by default, and the fix is one you already apply for another reason.
      </p>
      <Beyond>
        <p>The book does not discuss batches or attention. This bridge, the batch below and both questions are ours.</p>
      </Beyond>
      <p>
        Below, every thread runs a loop whose length is drawn from 1 to 64, as lengths in a batch might be. The top drawing puts the
        threads in warps in arrival order, the bottom one sorted by length.
      </p>
      <Simplification>
        <p>
          A warp-iteration costs the same here whether one lane or all of them are still looping, and nothing else differs between the two
          orders. On a device, sorting also changes which memory each warp touches, which this model ignores.
        </p>
      </Simplification>
      <LengthsBatch />
      <p>
        Sorted, each warp holds near-equal lengths and stops about when its work is done. Shuffled, nearly every warp contains one of the
        long sequences and runs to it. Bucketing a batch by length, which you do to cut padding, makes warps uniform as a side effect.
      </p>
      <Ask q={questionsS45['ch04/4.5/q-buckets']} />
      <Ask q={questionsS45['ch04/4.5/q-causal']} />

      <h2>What you can do about it</h2>
      <p>
        First decide whether to do anything. A boundary check on large data is noise. A branch whose outcome is the same across a warp
        costs nothing however often it is taken. What is worth your time is a branch or a loop bound that differs between neighbouring
        threads over much of the grid, with real work on each path.
      </p>
      <p>
        Then there are two remedies. The first is to arrange the work so that each warp is uniform: sort or bucket the data, or split the
        cases into separate launches. The second is to remove the branch by computing both paths in every thread and selecting a result.
        The simulator's comparison line shows what the second costs: every lane now pays for both paths, which is what a split warp was
        already paying, plus the select. So in this model it never beats the branch it replaces. The model gives the branch itself no cost;
        whatever that cost is on a device is where a select could win, if anywhere, for paths a few instructions long. No vendor documents
        it, and a compiler may make the swap without asking you.
      </p>
      <Beyond>
        <p>
          The book stops at counting. AMD's guide lists both remedies, separate launches and “predication (branchless programming)”
          (facts B7); the compiler's swap is facts A9. When each pays is ours: lab 4.2 times the select on long paths only, and short paths
          are untested here.
        </p>
      </Beyond>
      <Ask q={questionsS45['ch04/4.5/q-select']} />

      <h2>Measure it</h2>
      <p>
        Everything on this page counted passes. Lab 4.2 times them. It runs one kernel with two equally long paths over the same data in
        several arrangements: all one path, the two paths sorted into separate warps, and the two paths mixed inside every warp. It also
        times the select version, and a loop with sorted and shuffled trip counts. The work is identical in each comparison; only who
        shares a warp changes.
      </p>
      <Predict
        id={DIVCOST_ID}
        prompt={
          <>
            The same work, with the two paths mixed half and half inside every warp instead of sorted so that each warp is uniform. How many
            times as long will it take on <em>your</em> GPU? Commit a number before you run lab 4.2.
          </>
        }
        unit="times as long"
      >
        {(v) => <MeasuredDivergence predicted={v} />}
      </Predict>
      <p>
        The lab is on the <a href={href('ch04', 'labs')}>labs page</a>. Next: why an SM is given far more warps than it can execute at
        once, and what the ones that are not executing are waiting for.
      </p>
    </>
  );
}
