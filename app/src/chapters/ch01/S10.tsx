import { useMemo, useState, type CSSProperties } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  design,
  race,
  regions,
  sweep,
  timeline,
  waves,
  type Design,
  type Finish,
  type Region,
  type SweepPoint,
} from '../../sim/designs';
import { BUDGET, Plan, TwoDesignsScene } from './TwoDesignsScene';
import { useWidth } from './useWidth';

// ---------------------------------------------------------------- helpers
const nf = (v: number) => v.toLocaleString('en-US', { maximumFractionDigits: 3 });
const plural = (n: number, one: string) => `${nf(n)} ${one}${n === 1 ? '' : 's'}`;
const MAX_CHAINS = 4096;
const MAX_LENGTH = 1000;
const HALVINGS = [0, 1, 2, 3];
/** A caption above a group of buttons (a <label> would forward clicks on the caption to the first button). */
const FIELD: CSSProperties = { display: 'inline-flex', flexDirection: 'column', gap: '0.2rem', color: 'var(--ink-2)' };
const describe = (d: Design) => `${plural(d.units, 'unit')}, speed ${d.speed}`;

// The pair the prediction and the default explorer use: the animation's two designs.
const FAST = design(BUDGET, 2);
const MANY = design(BUDGET, 0);

// q-chainlen: two jobs where the one with more arithmetic has fewer independent pieces.
// Both fit the explorer's ranges, so the reader can check them there.
export const LONG_FEW = { chains: 8, length: 1000 };
export const SHORT_MANY = { chains: 1024, length: 5 };
const longFew = race(FAST, MANY, LONG_FEW.chains, LONG_FEW.length);
const shortMany = race(FAST, MANY, SHORT_MANY.chains, SHORT_MANY.length);
const ops = (j: { chains: number; length: number }) => nf(j.chains * j.length);

// ---------------------------------------------------------------- Fig. 1.1, redrawn
/**
 * Our own drawing of the idea in the book's Fig. 1.1: the same chip area divided
 * two ways. Schematic only. The unit counts are the two designs used on this page.
 */
function AreaFigure() {
  const W = 640;
  const chipW = 286;
  const H = 262;
  // Below this width the two chips are drawn one above the other, each at its own size,
  // so that the labels stay legible on a phone.
  const [ref, measured] = useWidth(W);
  const stacked = measured < 600;
  const top = 34;
  const chip = (x: number, title: string, bands: [string, number][], d: Design, channels: number, channelNote: string) => {
    let y = top;
    const rows = d.units > 8 ? 4 : 1;
    const cols = d.units / rows;
    const out = bands.map(([label, h], i) => {
      const y0 = y;
      y += h;
      const arithmetic = i === bands.length - 1;
      return (
        <g key={label}>
          <rect x={x} y={y0} width={chipW} height={h} fill={arithmetic ? 'var(--bench)' : 'var(--cell)'} stroke="var(--ink)" strokeWidth="1" />
          {arithmetic ? (
            Array.from({ length: d.units }, (_, k) => (
              <rect key={k} x={x + 5 + (k % cols) * ((chipW - 10) / cols) + 1.5} y={y0 + 5 + Math.floor(k / cols) * ((h - 10) / rows) + 1.5}
                width={(chipW - 10) / cols - 3} height={(h - 10) / rows - 3} rx="1.5" fill="var(--page)" stroke="var(--ink)" strokeWidth={rows > 1 ? 0.9 : 1.5} />
            ))
          ) : (
            <text x={x + chipW / 2} y={y0 + h / 2 + 4} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--ink)' }}>{label}</text>
          )}
        </g>
      );
    });
    const bottom = y;
    return (
      <g>
        <text x={x} y={14} fontSize="13" fontWeight="600" style={{ fill: 'var(--ink)' }}>{title}</text>
        <text x={x} y={28} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{bands[bands.length - 1][0]}</text>
        {out}
        {Array.from({ length: channels }, (_, k) => {
          const cx = x + ((k + 0.5) * chipW) / channels;
          return <line key={k} x1={cx} x2={cx} y1={bottom} y2={bottom + 22} stroke="var(--ink)" strokeWidth={channels > 4 ? 2 : 5} />;
        })}
        <rect x={x} y={bottom + 22} width={chipW} height={22} fill="var(--cell)" stroke="var(--ink)" />
        <text x={x + chipW / 2} y={bottom + 37} fontSize="11.5" textAnchor="middle" style={{ fill: 'var(--ink)' }}>memory</text>
        <text x={x} y={bottom + 60} fontSize="11" style={{ fill: 'var(--ink-2)' }}>{channelNote}</text>
      </g>
    );
  };
  const cpu = (x: number) =>
    chip(x, 'Latency-oriented (a CPU)',
      [['control logic and branch prediction', 48], ['large caches', 58], [`arithmetic: ${describe(FAST).replace(', speed', ' of speed')}`, 54]], FAST, 2, 'few channels to memory');
  const gpu = (x: number) =>
    chip(x, 'Throughput-oriented (a GPU)',
      [['control', 18], ['small caches', 18], [`arithmetic: ${describe(MANY).replace(', speed', ' of speed')}`, 124]], MANY, 8, 'many channels to memory, each allowed to be slow');
  const svgStyle = { width: '100%', display: 'block', fontFamily: 'var(--sans)' } as const;
  return (
    <figure style={{ margin: '0.2rem 0 1.4rem' }} data-stacked={stacked ? '1' : '0'}>
      <div ref={ref} style={{ width: '100%' }}>
        {stacked ? (
          <div style={{ display: 'grid', gap: '1rem' }}>
            <svg viewBox={`-2 0 ${chipW + 4} ${H}`} role="img" aria-label="Chip area of a latency-oriented design" style={{ ...svgStyle, maxWidth: 360 }}>
              {cpu(0)}
            </svg>
            <svg viewBox={`-2 0 ${chipW + 4} ${H}`} role="img" aria-label="Chip area of a throughput-oriented design" style={{ ...svgStyle, maxWidth: 360 }}>
              {gpu(0)}
            </svg>
          </div>
        ) : (
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="The same chip area divided two ways" style={{ ...svgStyle, maxWidth: W }}>
            {cpu(0)}
            {gpu(W - chipW)}
          </svg>
        )}
      </div>
      <figcaption className="ui muted" style={{ fontSize: '0.8rem', marginTop: '0.4rem' }}>
        Where the area goes, redrawn from the idea of the book's Fig. 1.1. A schematic: the proportions are not
        measurements, and the unit counts are the two designs of the animation below.
      </figcaption>
    </figure>
  );
}

// ---------------------------------------------------------------- explorer drawings
/** Which unit runs which chain: one row per unit, time across, on a scale shared by both designs. */
function Strip({ d, chains, length, tMax, f }: { d: Design; chains: number; length: number; tMax: number; f: Finish }) {
  const W = 300;
  const top = 8;
  const hgt = 100;
  const x0 = 2;
  const w = W - 4;
  const X = (t: number) => x0 + (t / tMax) * w;
  const row = hgt / d.units;
  const shade = (wave: number) => (wave % 2 ? 'var(--ink-3)' : 'var(--ink-2)');
  const ws = waves(d, chains, length);
  let marks;
  if (chains <= 512) {
    // few enough to draw every chain
    marks = timeline(d, chains, length).map((s) => (
      <rect key={s.chain} x={X(s.start)} y={top + s.unit * row} width={Math.max(0.8, X(s.end) - X(s.start) - (ws.length > 1 ? 0.8 : 0))}
        height={Math.max(0.8, row - (row > 3 ? 1 : 0.4))} fill={shade(s.wave)} />
    ));
  } else if (ws.length <= 400) {
    // one block per wave
    marks = ws.map((v) => (
      <rect key={v.wave} x={X(v.start)} y={top} width={Math.max(0.6, X(v.end) - X(v.start) - 0.5)} height={(v.units / d.units) * hgt} fill={shade(v.wave)} />
    ));
  } else {
    // too many waves to tell apart: the full ones as one block, then the last
    const lastWave = ws[ws.length - 1];
    marks = [
      <rect key="full" x={X(0)} y={top} width={X(lastWave.start) - X(0)} height={hgt} fill="var(--ink-2)" />,
      <rect key="last" x={X(lastWave.start)} y={top} width={Math.max(0.6, X(lastWave.end) - X(lastWave.start))} height={(lastWave.units / d.units) * hgt} fill="var(--ink-3)" />,
    ];
  }
  const endRight = X(f.last) > W - 130;
  return (
    <svg viewBox={`0 0 ${W} 132`} role="img" aria-label={`Which of the ${d.units} units runs which chain, over time`} style={{ width: '100%', maxWidth: 440, display: 'block' }}>
      <rect x={x0} y={top} width={w} height={hgt} fill="var(--cell)" stroke="var(--rule)" />
      {marks}
      <line x1={X(f.first)} x2={X(f.first)} y1={top - 5} y2={top + hgt + 5} stroke="var(--ink)" strokeWidth="1" strokeDasharray="3 3" />
      <line x1={X(f.last)} x2={X(f.last)} y1={top - 5} y2={top + hgt + 5} stroke="var(--ink)" strokeWidth="1.8" />
      <text x={x0} y={top + hgt + 20} fontSize="11" style={{ fill: 'var(--ink-2)' }}>0</text>
      <text x={endRight ? X(f.last) : X(f.last) + 5} y={top + hgt + 20} fontSize="11" textAnchor={endRight ? 'end' : 'start'}>
        last result at {nf(f.last)}
      </text>
    </svg>
  );
}

/** Finishing time of both designs against the number of chains, log scales, with the current point marked. */
function CrossPlot({ pts, regs, chains, a, b }: { pts: SweepPoint[]; regs: Region[]; chains: number; a: Design; b: Design }) {
  // Drawn at its measured width, so that its text keeps its size on a phone.
  const [ref, W] = useWidth(560);
  const narrow = W < 440;
  // On a wide plot, room on the right for the label of the last point and the legend inside the frame.
  // On a narrow one the label turns left and the legend moves under the axis, where no line runs.
  const H = Math.round(Math.min(330, Math.max(250, W * 0.59))) + (narrow ? 40 : 0);
  const pad = { l: 58, r: narrow ? 26 : 70, t: 14, b: narrow ? 84 : 44 };
  const lo = Math.min(pts[0].a, pts[0].b) / 1.7;
  const hi = Math.max(pts[pts.length - 1].a, pts[pts.length - 1].b) * 1.7;
  const lx = (v: number) => Math.log2(v) / Math.log2(MAX_CHAINS);
  const X = (v: number) => pad.l + lx(v) * (W - pad.l - pad.r);
  const Y = (v: number) => H - pad.b - ((Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * (H - pad.t - pad.b);
  // A staircase has long flat runs: keep only the points where a value changes.
  const line = (key: 'a' | 'b') =>
    pts
      .filter((q, i) => i === 0 || i === pts.length - 1 || pts[i - 1][key] !== q[key] || pts[i + 1][key] !== q[key])
      .map((q) => `${X(q.chains).toFixed(1)},${Y(q[key]).toFixed(1)}`)
      .join(' ');
  const xt = [1, 4, 16, 64, 256, 1024, 4096];
  const yt: number[] = [];
  for (let e = Math.ceil(Math.log10(lo)); e <= Math.floor(Math.log10(hi)); e++) yt.push(10 ** e);
  const cur = pts[chains - 1];
  const same = cur.a === cur.b;
  const aAbove = cur.a > cur.b;
  // The slower design's label goes above its line and to the left, the faster one's below and to the right,
  // where neither staircase runs.
  const cx = X(chains);
  const upper = cx < pad.l + 84 ? { dx: 9, anchor: 'start' as const } : { dx: -9, anchor: 'end' as const };
  const lower = cx > W - 90 ? { dx: -9, dy: 18, anchor: 'end' as const } : { dx: 9, dy: 18, anchor: 'start' as const };
  const lowest = Math.max(Y(cur.a), Y(cur.b));
  const lg = narrow ? { x: 20, y: H - pad.b + 56 } : { x: W - pad.r - 176, y: H - pad.b - 34 };
  return (
    <div ref={ref} style={{ width: '100%', overflow: 'hidden' }}>
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Time to the last result against the number of independent chains" style={{ display: 'block' }}>
      <rect x={pad.l} y={pad.t} width={W - pad.l - pad.r} height={H - pad.t - pad.b} fill="var(--page)" stroke="var(--rule)" />
      {regs.filter((g) => g.winner === 'tie' && regs.length > 1).map((g) => (
        <rect key={g.from} x={X(g.from) - 1} y={pad.t} width={X(g.to) - X(g.from) + 2} height={H - pad.t - pad.b} fill="var(--rule-soft)" stroke="var(--rule)" />
      ))}
      {xt.map((v) => (
        <g key={'x' + v}>
          <line x1={X(v)} x2={X(v)} y1={pad.t} y2={H - pad.b} stroke="var(--rule-soft)" />
          <text x={X(v)} y={H - pad.b + 15} fontSize="11" textAnchor="middle">{nf(v)}</text>
        </g>
      ))}
      {yt.map((v) => (
        <g key={'y' + v}>
          <line x1={pad.l} x2={W - pad.r} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={pad.l - 6} y={Y(v) + 4} fontSize="11" textAnchor="end">{nf(v)}</text>
        </g>
      ))}
      <text x={(W - pad.r + pad.l) / 2} y={H - pad.b + 36} fontSize="11.5" textAnchor="middle">independent chains</text>
      <text x={14} y={(H - pad.b + pad.t) / 2} fontSize="11.5" textAnchor="middle" transform={`rotate(-90 14 ${(H - pad.b + pad.t) / 2})`}>time to the last result</text>
      <polyline points={line('a')} fill="none" stroke="var(--ink)" strokeWidth="2.2" strokeLinejoin="round" />
      <polyline points={line('b')} fill="none" stroke="var(--ink)" strokeWidth="2.2" strokeDasharray="6 4" strokeLinejoin="round" />
      {/* legend */}
      <line x1={lg.x} x2={lg.x + 28} y1={lg.y} y2={lg.y} stroke="var(--ink)" strokeWidth="2.2" />
      <text x={lg.x + 34} y={lg.y + 4} fontSize="11.5">A: {describe(a)}</text>
      <line x1={lg.x} x2={lg.x + 28} y1={lg.y + 18} y2={lg.y + 18} stroke="var(--ink)" strokeWidth="2.2" strokeDasharray="6 4" />
      <text x={lg.x + 34} y={lg.y + 22} fontSize="11.5">B: {describe(b)}</text>
      {/* the current point */}
      <line x1={cx} x2={cx} y1={pad.t} y2={lowest} stroke="var(--ink)" strokeDasharray="3 3" />
      <circle cx={cx} cy={Y(cur.a)} r="4.5" fill="var(--ink)" />
      <circle cx={cx} cy={Y(cur.b)} r="4.5" fill="var(--page)" stroke="var(--ink)" strokeWidth="1.8" />
      {same ? (
        <text x={cx + upper.dx} y={Y(cur.a) - 9} fontSize="11.5" fontWeight="600" textAnchor={upper.anchor}>A and B: {nf(cur.a)}</text>
      ) : (
        <>
          <text x={cx + (aAbove ? upper.dx : lower.dx)} y={Y(cur.a) + (aAbove ? -9 : lower.dy)} fontSize="11.5" fontWeight="600"
            textAnchor={aAbove ? upper.anchor : lower.anchor}>A: {nf(cur.a)}</text>
          <text x={cx + (aAbove ? lower.dx : upper.dx)} y={Y(cur.b) + (aAbove ? lower.dy : -9)} fontSize="11.5" fontWeight="600"
            textAnchor={aAbove ? lower.anchor : upper.anchor}>B: {nf(cur.b)}</text>
        </>
      )}
    </svg>
    </div>
  );
}

const leads = (regs: Region[]) =>
  regs
    .map((g) => {
      const span = g.from === g.to ? `at ${nf(g.from)}` : `from ${nf(g.from)} to ${nf(g.to)}`;
      return g.winner === 'tie' ? `a tie ${span}` : `${g.winner === 'a' ? 'A' : 'B'} leads ${span}`;
    })
    .join(', ');

function BudgetBench() {
  const [ha, setHa] = useState(2);
  const [hb, setHb] = useState(0);
  const [chains, setChains] = useState(1);
  const [length, setLength] = useState(100);
  const a = design(BUDGET, ha);
  const b = design(BUDGET, hb);
  const r = race(a, b, chains, length);
  const pts = useMemo(() => sweep(design(BUDGET, ha), design(BUDGET, hb), MAX_CHAINS, length), [ha, hb, length]);
  const regs = useMemo(() => regions(pts), [pts]);
  const tMax = Math.max(r.a.last, r.b.last);
  const clampChains = (v: number) => Math.max(1, Math.min(MAX_CHAINS, Math.round(v) || 1));

  const pick = (label: string, h: number, set: (h: number) => void) => (
    <span style={FIELD}>
      {label}: halve the latency of a unit {h === 1 ? 'once' : `${h} times`}
      <span className="seg" role="group" aria-label={label}>
        {HALVINGS.map((k) => (
          <button key={k} aria-pressed={h === k} onClick={() => set(k)}>
            {describe(design(BUDGET, k))}
          </button>
        ))}
      </span>
    </span>
  );
  const panel = (key: 'a' | 'b', d: Design, f: Finish) => (
    <div className="panel" data-design={key}>
      <p style={{ margin: '0 0 0.5rem' }}>
        <b>Design {key.toUpperCase()}</b>: {describe(d)}. {d.unitPower === 1 ? 'Each unit draws 1 of' : `Each unit draws ${d.unitPower} of`} the{' '}
        {BUDGET}.
      </p>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.6rem 0.9rem', alignItems: 'flex-start' }}>
        <svg viewBox="0 0 120 120" role="img" aria-label={`Floor plan: ${describe(d)}`} style={{ width: 120, flex: '0 0 auto', display: 'block' }}>
          <Plan x={0} y={0} size={120} d={d} busy={new Set(Array.from({ length: d.units - f.idle }, (_, i) => i))} />
        </svg>
        <div style={{ flex: '1 1 220px', minWidth: 0 }}>
          <Strip d={d} chains={chains} length={length} tMax={tMax} f={f} />
        </div>
      </div>
      <p className="muted" style={{ margin: '0.4rem 0 0' }} data-out={key + '-used'}>
        {f.idle === 0
          ? `Every unit gets a chain. ${plural(f.waves, 'wave')}.`
          : `${plural(d.units - f.idle, 'unit')} working, ${nf(f.idle)} never used. ${plural(f.waves, 'wave')}.`}
      </p>
    </div>
  );
  const stat = (label: string, key: string, va: number | string, vb: number | string) => (
    <tr>
      <td>{label}</td>
      <td className="num" data-stat={'a-' + key}>{typeof va === 'number' ? nf(va) : va}</td>
      <td className="num" data-stat={'b-' + key}>{typeof vb === 'number' ? nf(vb) : vb}</td>
    </tr>
  );
  return (
    <Bench
      title="Spend a power budget"
      task="Find the number of chains at which the lead changes hands. Then change the chain length and see what moves."
      foot={
        <>
          Time is counted in operations of a plain unit (speed 1). In the strips, one row is one unit, alternate waves
          alternate in shade, the dashed line is the first result and the solid line the last. Both strips share one
          time scale. The budget of {BUDGET} is an abstract number, not a device.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        {pick('Design A', ha, setHa)}
        {pick('Design B', hb, setHb)}
      </div>
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        <label className="field">
          Independent chains = {nf(chains)}
          <input type="range" min="0" max="120" value={Math.round(10 * Math.log2(chains))} aria-label="Independent chains (log scale)"
            onChange={(e) => setChains(clampChains(2 ** (+e.target.value / 10)))} />
        </label>
        <label className="field">
          exact number
          <input type="number" min="1" max={MAX_CHAINS} step="1" value={chains} aria-label="Independent chains (exact)"
            onChange={(e) => setChains(clampChains(+e.target.value))} />
        </label>
        <label className="field">
          Operations in a chain = {nf(length)}
          <input type="range" min="1" max={MAX_LENGTH} value={length} aria-label="Operations in a chain" onChange={(e) => setLength(+e.target.value)} />
        </label>
      </div>
      <div className="cols" style={{ marginBottom: '1.2rem' }}>
        {panel('a', a, r.a)}
        {panel('b', b, r.b)}
      </div>
      <div className="cols plot">
        <CrossPlot pts={pts} regs={regs} chains={chains} a={a} b={b} />
        <div className="panel">
          <p style={{ fontSize: '1.15rem', fontWeight: 600, margin: '0 0 0.2rem' }} data-out="verdict">
            {r.winner === 'tie'
              ? `A tie: both finish at ${nf(r.a.last)}.`
              : `Design ${r.winner.toUpperCase()} finishes first, ${nf(r.factor)} times sooner.`}
          </p>
          <p className="muted" style={{ margin: '0 0 0.7rem' }}>
            {plural(chains, 'chain')} of {plural(length, 'operation')}.
          </p>
          <table className="data" style={{ margin: 0 }}>
            <thead>
              <tr>
                <th />
                <th className="num">A</th>
                <th className="num">B</th>
              </tr>
            </thead>
            <tbody>
              {stat('Units × speed of each', 'shape', `${a.units} × ${a.speed}`, `${b.units} × ${b.speed}`)}
              {stat('Peak rate, every unit busy', 'peak', a.throughput, b.throughput)}
              {stat('First result at', 'first', r.a.first, r.b.first)}
              {stat('Last result at', 'last', r.a.last, r.b.last)}
              {stat('Rate delivered on this work', 'delivered', r.a.delivered, r.b.delivered)}
            </tbody>
          </table>
          <p className="muted" style={{ margin: '0.5rem 0 0' }}>Rates are operations per time unit.</p>
          <p style={{ margin: '0.7rem 0 0' }} data-out="regions">
            {regs.length === 1
              ? 'These two designs are the same, so every number of chains is a tie.'
              : `For these two designs, at any chain length: ${leads(regs)} chains, the most this explorer runs.`}
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questions10: Record<string, Question> = {
  'ch01/start/q-batch': {
    kind: 'choice',
    id: 'ch01/start/q-batch',
    prompt: (
      <>
        A GPU server runs a model on one request at a time (batch size 1). You switch it to batches of 32, and there are
        always enough requests waiting to fill a batch. What happens to the time one request takes once it starts
        running, and to the number of requests finished per second?
      </>
    ),
    options: [
      { label: 'Each request takes longer; requests per second go up.', correct: true },
      {
        label: 'Each request takes less time; requests per second go up.',
        why: 'A pass over 32 requests does more work than a pass over one. No single request gets its answer sooner than it would alone.',
      },
      {
        label: 'Each request takes the same time; requests per second go up 32 times.',
        why: 'The batched pass is not free. It takes longer than one request alone, though far less than 32 of them in a row.',
      },
      {
        label: 'Each request takes longer; requests per second stay the same.',
        why: 'Then nobody would batch. The batched pass takes much less time than its 32 requests run one after another.',
      },
    ],
    explain: (
      <p>
        One pass over 32 requests takes longer than a pass over one, and much less than 32 passes. So each request
        waits longer and the group is done sooner. The book puts it this way in its section on batching, and then says
        it is the same trade as the one between the two kinds of processor: one design gives each piece of work a
        shorter latency, the other gives many pieces a higher throughput. This page is about that trade, built into
        hardware.
      </p>
    ),
    source: 'ch. 22 §22.5, pp. 539–540',
  },
  'ch01/start/q-sixteenbit': {
    kind: 'choice',
    id: 'ch01/start/q-sixteenbit',
    prompt: (
      <>
        The book lists 34, 67 and 1,979 TFLOPS for 64-, 32- and 16-bit arithmetic on its example GPU. Going from 64 to
        32 bits doubles the rate. What explains the step of about 30 times from 32 to 16 bits?
      </>
    ),
    options: [
      {
        label: 'Halving the precision again: 16-bit values are that much cheaper to compute with.',
        why: 'The step from 64 to 32 bits shows what narrower values buy on the same units: a factor of 2. The same step again would give about 134.',
      },
      {
        label: 'A different kind of unit and an assumption: the third figure is for matrix-multiply units (tensor cores), and it assumes sparsity.',
        correct: true,
      },
      {
        label: 'Memory: 16-bit values need half the bandwidth, so more of them arrive per second.',
        why: 'These are peak rates of the arithmetic units. Memory does not enter into them.',
      },
      {
        label: 'A misprint in the book.',
        why: 'The vendor\'s own table prints the same three numbers.',
      },
    ],
    explain: (
      <p>
        Halving the width of the values doubled the rate once, so a further halving cannot account for a factor of 30.
        The third number has to come from a different row of the vendor's table, and it does: the row is "FP16 Tensor
        Core", and it carries the footnote "With sparsity". Tensor cores are units built for matrix multiplication.
        For LLM inference the comparison that matters is therefore not 16 bits against 32. It is matrix units against
        ordinary ones, and what a kernel has to do to use them, which is chapter 15.
      </p>
    ),
    source: 'book p. 3; NVIDIA H100 specification table, checked 2026-11-20 (notes/today/ch01-facts.md, A1)',
  },
  'ch01/start/q-chainlen': {
    kind: 'choice',
    id: 'ch01/start/q-chainlen',
    prompt: (
      <>
        Two jobs for the animation's designs, A ({describe(FAST)}) and B ({describe(MANY)}). Job X is{' '}
        {nf(LONG_FEW.chains)} independent chains of {nf(LONG_FEW.length)} operations: {ops(LONG_FEW)} operations in
        all. Job Y is {nf(SHORT_MANY.chains)} independent chains of {nf(SHORT_MANY.length)} operations:{' '}
        {ops(SHORT_MANY)} in all. In the explorer's model, which design finishes each job first?
      </>
    ),
    options: [
      {
        label: `Both on A: its units are ${FAST.speed} times faster.`,
        why: `For Y, A needs ${nf(shortMany.a.waves)} waves of ${nf(SHORT_MANY.length / FAST.speed)}, done at ${nf(shortMany.a.last)}. B needs ${nf(shortMany.b.waves)} waves of ${nf(SHORT_MANY.length)}, done at ${nf(shortMany.b.last)}.`,
      },
      {
        label: `Both on B: it has ${MANY.throughput / FAST.throughput} times the peak rate.`,
        why: `A peak rate needs every unit busy. For X, B can use only ${nf(LONG_FEW.chains)} of its ${MANY.units} units: one wave of ${nf(LONG_FEW.length)}, done at ${nf(longFew.b.last)}. A runs ${nf(longFew.a.waves)} waves of ${nf(LONG_FEW.length / FAST.speed)} and is done at ${nf(longFew.a.last)}.`,
      },
      { label: 'X on A, Y on B.', correct: true },
      {
        label: 'X on B, Y on A: the bigger job goes to the design with the higher peak rate.',
        why: `That is each job on the design that suits it least: X takes ${nf(longFew.b.last)} on B against ${nf(longFew.a.last)} on A, and Y takes ${nf(shortMany.a.last)} on A against ${nf(shortMany.b.last)} on B.`,
      },
    ],
    explain: (
      <p>
        Count the independent pieces before the operations. X has {nf(LONG_FEW.chains)}: more than A's {FAST.units}{' '}
        units and far fewer than B's {MANY.units}. A finishes it in {nf(longFew.a.waves)} waves, at{' '}
        {nf(longFew.a.last)}; B in one wave, at {nf(longFew.b.last)}. Y has {nf(SHORT_MANY.chains)}: B finishes in{' '}
        {nf(shortMany.b.waves)} waves, at {nf(shortMany.b.last)}, while A needs {nf(shortMany.a.waves)}, until{' '}
        {nf(shortMany.a.last)}. So the job with more arithmetic goes to the latency design. On one job the length of
        a chain scales both designs' times alike: it sets how long the job takes, not which design should take it.
        Both jobs fit the explorer above, if you want to check.
      </p>
    ),
    source: 'our model of the book\'s argument, ch. 1 §1.1, pp. 4–5',
  },
  'ch01/start/q-tokens': {
    kind: 'choice',
    id: 'ch01/start/q-tokens',
    prompt: (
      <>
        One request generates text token by token, and computing token t+1 needs token t. Where is the independence
        that a throughput-oriented processor needs?
      </>
    ),
    options: [
      {
        label: 'Across tokens: the GPU works on many future tokens of the request at once.',
        why: 'Token t+1 needs token t. Along the sequence, the work is one dependent chain.',
      },
      {
        label: 'Inside each step: every matrix product of the forward pass is thousands of dot products that do not depend on each other.',
        correct: true,
      },
      {
        label: 'Nowhere. Generation is sequential, so a GPU cannot help a single request.',
        why: 'The loop over tokens is sequential. What happens inside one turn of the loop is not.',
      },
    ],
    explain: (
      <p>
        The loop over tokens is the chain. The matrix products inside one step are the crowd: each output element is a
        dot product that needs no other output element, which is the picture in the last part of the animation.
        Batching adds a second source of independence, across requests. So a single stream is limited by the chain
        (one step after another) and helped by the crowd inside each step. This mapping is ours.
      </p>
    ),
    source: 'our analysis, using ch. 1 §1.1, pp. 4–6; the book treats generation in ch. 20',
  },
  'ch01/start/q-oneunitmore': {
    kind: 'numeric',
    id: 'ch01/start/q-oneunitmore',
    prompt: (
      <>
        Design B has 64 units of speed 1. The work is 80 independent chains of 100 operations each. In the explorer's
        model, how many time units pass before the last chain finishes? Work it out, then set the explorer to it.
      </>
    ),
    answer: 200,
    unit: 'time units',
    explain: (
      <p>
        64 chains start at once and finish at 100. The other 16 start then and finish at 200: two waves. A 65th unit
        would change nothing, because 80 chains would still need two waves. Taking 16 chains away would halve the
        time. Throughput arrives in steps the size of the hardware. You meet the same rounding up in chapter 2, where
        it sets how many blocks of threads a kernel launch needs.
      </p>
    ),
    source: 'our model of the book\'s argument, ch. 1 §1.1, pp. 4–5',
  },
};
/** The same record under the name the chapter index convention expects. */
export const questionsS10 = questions10;

// ---------------------------------------------------------------- page
export function S10() {
  const crowd = race(FAST, MANY, 1024, 100);
  return (
    <>
      <p>
        The book's first chapter has no code and no exercises. It argues that parallel programming is worth learning
        and lays out the rest of the book. This course keeps four things from it: why the two kinds of processor are
        built differently (this page); how to put a ceiling on a speed-up, and the four difficulties the book is
        organised around (<a href={href('ch01', '1.3')}>section 1.3</a>); and a map of the book for your goals (
        <a href={href('ch01', '1.7')}>section 1.7</a>).
      </p>
      <Source pages="1–20" />
      <p>
        Every later chapter opens with a check on what the chapters before it taught. Nothing comes before this one,
        so there is no such check here. One question from the framework side stands in for it. Answer it from what
        you know about serving models. The rest of the page explains why the answer is also a statement about
        hardware.
      </p>
      <Ask q={questions10['ch01/start/q-batch']} label="From what you already know" />

      <h2>Why there are two kinds of processor</h2>
      <p>
        Until about 2003 a program got faster with each new processor without being touched, because each new
        processor ran a single sequence of instructions faster than the last. Energy use and heat ended that: they
        capped both the clock rate and the work that fits in one clock period. Since then, more speed has come from
        running several instruction sequences (threads) at the same time, and a program benefits only if its work is
        split that way. Two lines of design followed. <b>Multi-core</b> processors keep each thread fast and add
        cores. <b>Many-thread</b> processors, the GPUs, give up speed per thread to maximise the total work done by a
        very large number of them.
      </p>
      <Source pages="1–3" />
      <p>
        The book's example of the second kind is NVIDIA's Hopper H100, with hundreds of thousands of threads on many
        simple pipelines. It quotes peak rates, as of 2022, of 34, 67 and
        1,979 TFLOPS for 64-, 32- and 16-bit arithmetic (one TFLOPS is 10<sup>12</sup> floating-point operations per
        second), against "a few TFLOPS" for a server CPU of the same year. It adds its own caveat: these are peak
        rates of the hardware, not the speed of any application.
      </p>
      <Source section="1.1" pages="3" />
      <Ask q={questions10['ch01/start/q-sixteenbit']} />
      <p>
        Read every figure of this kind with both points in mind. A peak rate is what the arithmetic units could do if
        they never had to wait for anything, and the largest figures on a specification sheet belong to the matrix
        units. How close a kernel gets to the first is the subject of chapter 5. What it takes to use the second is
        in chapter 15.
      </p>
      <Today checked="2026-11-20">
        <p>
          NVIDIA's H100 page still prints all three figures, for the SXM variant. 34 and 67 are its plain FP64 and
          FP32 rows. 1,979 is the row "FP16 Tensor Core", footnoted "With sparsity".
        </p>
      </Today>

      <h2>Where the chip area goes</h2>
      <p>
        The gap comes from what each design is asked to be good at. A CPU is a <b>latency-oriented</b> design: it is
        built so that each operation of a single thread finishes as soon as possible. That takes arithmetic units
        designed for short latency, large caches that turn slow memory accesses into fast ones, and control logic
        with branch prediction to keep one stream of instructions moving. All of it uses chip area and power that
        could have gone to more arithmetic units and more channels to memory.
      </p>
      <Source section="1.1" pages="4–5" figs="Fig. 1.1" />
      <AreaFigure />
      <p>
        A GPU is a <b>throughput-oriented</b> design. Its shape comes from video games, which need a very large number
        of floating-point calculations and memory accesses for every frame. So the area and the power go to many
        simple arithmetic units and many memory channels, with little control logic and small caches, and each single
        operation is allowed to take a long time.
      </p>
      <Today checked="2026-11-20">
        <p>
          NVIDIA's current CUDA programming guide states the same split: a GPU trades "lower single-thread
          performance" for "much greater total throughput", and GPUs "devote more transistors to data processing
          units".
        </p>
      </Today>
      <p>
        Why not build one chip that is good at both? The book's answer is a cost rule. To double the throughput,
        double the number of units: twice the area and twice the power. To halve the latency of a unit, you may need
        twice the current, more than twice the area and four times the power. Latency is far more expensive to buy
        than throughput, so a design has to choose. Treat this rule as the book's argument: no vendor document states
        it, and it is unverified. The animation and the explorer below take that one rule and follow where it leads.
      </p>
      <Source section="1.1" pages="5" />

      <h2>One budget, two designs</h2>
      <p>
        Two terms first. <b>Latency</b> is how long one piece of work takes from start to finish.{' '}
        <b>Throughput</b> is how much work is finished per unit of time, over all the pieces. The animation gives two
        designs the same power budget and the same work, and runs them against one clock. Watch it once before you
        use the controls further down.
      </p>
      <TwoDesignsScene />
      <Predict
        id="ch01/start/p-crowd"
        prompt={
          <>
            Design A has {FAST.units} units, each {FAST.speed} times faster than a plain one. Design B has {MANY.units}{' '}
            plain units. By the book's rule both draw the same power. The work is 1,024 independent chains of equal
            length. How many times sooner does B finish?
          </>
        }
        unit="times sooner"
      >
        {(v) => (
          <p>
            <b>{nf(crowd.factor)} times.</b> B runs {MANY.units} chains at a time, so 1,024 chains take{' '}
            {nf(crowd.b.waves)} waves. A runs {FAST.units} at a time and needs {nf(crowd.a.waves)} waves, each{' '}
            {FAST.speed} times shorter, which is the time of {nf(crowd.a.waves / FAST.speed)} of B's waves.{' '}
            {nf(crowd.a.waves / FAST.speed)} against {nf(crowd.b.waves)} is {nf(crowd.factor)}.{' '}
            {v === crowd.factor
              ? 'That is what you had.'
              : v === MANY.units / FAST.units
                ? `${nf(v)} is the ratio of the unit counts, or of the wave counts. It forgets that each of A's units is ${FAST.speed} times faster.`
                : v > 0 && v < 1
                  ? 'A number below 1 backs the faster units. They win only while there are too few chains to keep B\'s units busy.'
                  : v === MANY.units
                    ? `${nf(v)} counts B's units and leaves A out of the comparison.`
                    : 'Count the waves each design needs, then how long one wave takes.'}{' '}
            Set the explorer below to 1,024 chains and check it. Then bring the chains back down to 1 and watch the
            factor turn round.
          </p>
        )}
      </Predict>
      <BudgetBench />
      <p>
        Three things to take from it. On one budget the two designs differ by the same factor in opposite directions:
        the fast units win a single chain by exactly the factor by which the many units win a crowd. The length of a
        chain cancels out, so it never changes the winner or the factor. What decides the winner is how many pieces
        of the work are independent of each other. And latency bought on a fixed budget is paid for in peak rate:
        each halving takes the rate with every unit busy from {HALVINGS.map((h) => design(BUDGET, h).throughput).join(' to ')}.
      </p>
      <Simplification>
        <p>
          One step, half the latency for four times the power, is the book's rule (p. 5). Repeating it is our
          extrapolation; real circuits do not follow one exponent.
        </p>
        <p>
          Left out: the area cost, caches, memory channels, pipelining. A unit runs one chain at a time; a real GPU
          keeps many more threads resident than units (chapter 4). The budget of {BUDGET} is an abstract number, not
          a device.
        </p>
      </Simplification>
      <p>
        Look at the table once more before you leave it. The peak rate of a design never moves, whatever work you
        give it. The rate delivered does, and with a single chain it is a small fraction of the peak. That is the
        book's caveat about the H100 figures, in a model small enough to check by hand.
      </p>
      <Ask q={questions10['ch01/start/q-chainlen']} />
      <Ask q={questions10['ch01/start/q-oneunitmore']} />

      <h2>What the design asks of your code</h2>
      <p>
        In the explorer a unit sits idle whenever there are fewer chains than units. On a real GPU there is a second
        reason for a unit to have nothing to do. The design lets memory accesses and arithmetic take a long time, and
        a thread that is waiting for one of them has nothing to run. The book draws the conclusion for software: GPU
        code is expected to supply a very large number of threads, so that the hardware always has some that are
        ready while others wait.
      </p>
      <Source section="1.1" pages="5" />
      <p>
        Chapter 4 shows the mechanism (§4.6, pp. 82–85): the hardware keeps many more threads resident than it can
        execute at one instant, and at each step it picks one that is not waiting. The hundreds of thousands of
        threads the book gives the H100 are threads of this kind: resident, with far fewer executing at any moment.
      </p>
      <Today checked="2026-11-20 (notes/today/ch01-facts.md, A2)">
        <p>
          Two of NVIDIA's own H100 figures, multiplied together, give a count of that size for threads held resident.
          The multiplication is ours.
        </p>
      </Today>
      <p>
        Neither design replaces the other. The sequential parts of a program stay on the CPU and the numerically
        heavy parts go to the GPU, which is why the programming model you meet in chapter 2 has a host and a device.
      </p>
      <Source section="1.1 and 1.3" pages="5–6, 10" />

      <h2>The trade you already make</h2>
      <p>
        You make this trade every time you choose a batch size. At batch size 1 a request waits for nothing except
        its own arithmetic, which is the shortest wait one request can have. With a large batch every request takes
        longer, and the GPU finishes more tokens per second in total. Serving at batch size 1 asks a
        throughput-oriented processor to behave like a latency-oriented one. The book draws the same parallel
        itself: its section on batching, near the end, points back to the two designs of this chapter.
      </p>
      <Source section="22.5" pages="539–540" />
      <Ask q={questions10['ch01/start/q-tokens']} />

      <h2>Where the other backends fit</h2>
      <p>
        The book makes this argument with NVIDIA hardware, and nowhere in the chapter mentions HIP or Metal. By our
        reading of the vendors' own documents, the split between the two designs is not NVIDIA's alone. It carries
        over to the other two backends of this course: AMD GPUs, programmed through HIP, and Apple GPUs, programmed
        through Metal.
      </p>
      <Beyond>
        <p>
          HIP is CUDA's programming model on AMD GPUs. AMD documents it for AMD GPUs and says it is "not intended to
          be a drop-in replacement" for CUDA. Metal is Apple's graphics and compute API, with kernels written in the
          Metal Shading Language (MSL): a compute function is "also known as a compute kernel" and runs on "a grid of
          threads".
        </p>
        <p>
          One difference to hold on to. Fig. 1.1 draws the GPU with its own DRAM. That fits CUDA and HIP devices. It
          does not fit Apple silicon, where CPU and GPU share system memory. Chapter 2 takes it up.
        </p>
        <p className="muted">Checked 2026-11-20.</p>
      </Beyond>
    </>
  );
}
