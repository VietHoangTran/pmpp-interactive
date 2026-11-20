import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import {
  FIG49,
  STRAIGHT_LINE,
  fig410Trips,
  ifElse,
  loop,
  maskBelow,
  run,
  straightLine,
  type Issue,
  type Trace,
} from '../../sim/divergence';

// "A warp at a branch": one warp of the book's 32 lanes, one row per instruction
// issued, time running down the page. Every row is an Issue from src/sim/divergence.ts,
// the model the divergence simulator runs on. Seven beats, one narration clip each
// (content/ch04/narration/warp-branch-N.txt).
const W = FIG49.width;
const T = FIG49.threshold;
const COST = 3; // instructions in A and in B

const CELL = 14;
const X0 = 150; // left edge of lane 0
const Y0 = 56; // top of the first row
const ROW = 22;
const VIEW_W = 720;
const VIEW_H = 348;

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
/** How many of n things have appeared when the reveal runs from a to b. */
const shown = (p: number, a: number, b: number, n: number) => Math.min(n, Math.floor(ramp(p, a, b) * n + 1e-9));

const STRAIGHT = run(straightLine(W));
const SPLIT = maskBelow(W, T);
const BRANCH = run(ifElse(W, SPLIT, COST, COST, 'threadIdx.x < 24'));
const MIXED = run(ifElse(W, SPLIT, COST, COST, 'threadIdx.x < 24'), 'interleaved');
const TRIPS = fig410Trips(W);
const LOOP = run(loop(W, TRIPS));

const rowLabel = (x: Issue): string =>
  x.part === 'A' ? `A, instruction ${x.k + 1}` : x.part === 'B' ? `B, instruction ${x.k + 1}` : x.part === 'body' ? `iteration ${x.iter! + 1}` : x.label;

interface Frame {
  /** Rows drawn, oldest first. */
  rows: Issue[];
  /** Labels to use instead of rowLabel, by row. */
  labels?: string[];
  /** Letters or numbers written above each lane: what each thread wants. */
  wants?: (string | number)[];
  wantsLabel?: string;
  /** Draw a mark between lanes T-1 and T. */
  threshold?: number;
  /** A warp barrier drawn before the row with this index. */
  barrierBefore?: number;
  headline?: ReactNode;
  /** Lines of plain text on the right of the footer. */
  side?: string[];
  counters?: { label: string; value: string }[];
}

function Stage(f: Frame) {
  const laneX = (i: number) => X0 + i * CELL;
  let y = Y0;
  const rowY: number[] = [];
  f.rows.forEach((_, i) => {
    if (f.barrierBefore === i) y += 18;
    rowY.push(y);
    y += ROW;
  });
  const last = f.rows.length - 1;
  return (
    <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} role="img" aria-label="Animation of one warp executing instructions, one row per instruction" style={{ width: '100%', maxWidth: VIEW_W, display: 'block' }}>
      <defs>
        <pattern id="wb-wait" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="1.2" />
        </pattern>
      </defs>
      <text x={12} y={18} fontSize="12" fontWeight="600">one instruction stream</text>
      <text x={X0} y={18} fontSize="12" fontWeight="600">one warp: {W} lanes, each a thread with its own registers</text>
      {[0, 8, 16, 24, 31].map((i) => (
        <text key={i} x={laneX(i) + CELL / 2 - 1} y={34} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{i}</text>
      ))}
      {f.wants ? (
        <g>
          <text x={X0 - 8} y={48} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{f.wantsLabel}</text>
          {f.wants.map((w, i) => (
            <text key={i} x={laneX(i) + CELL / 2 - 1} y={48} fontSize="10" textAnchor="middle" fontWeight="600">{w}</text>
          ))}
        </g>
      ) : null}
      {f.threshold != null && f.threshold > 0 && f.threshold < W ? (
        <line x1={laneX(f.threshold) - 1} x2={laneX(f.threshold) - 1} y1={38} y2={Math.max(Y0 + ROW, y) - 4} stroke="var(--ink)" strokeWidth="1.5" strokeDasharray="4 3" />
      ) : null}
      {f.rows.map((r, i) => (
        <g key={i} data-row={i}>
          <text x={X0 - 8} y={rowY[i] + 12} fontSize="11.5" textAnchor="end" fontWeight={i === last ? 650 : 400}
            style={{ fill: i === last ? 'var(--ink)' : 'var(--ink-2)' }}>
            {i === last ? '▸ ' : ''}{f.labels?.[i] ?? rowLabel(r)}
          </text>
          {r.active.map((on, l) => (
            <g key={l}>
              <rect x={laneX(l)} y={rowY[i]} width={CELL - 2} height={ROW - 6} rx="1.5"
                fill={on ? (r.part === 'B' ? 'var(--ink-2)' : 'var(--ink)') : 'var(--cell)'} stroke={on ? 'none' : 'var(--rule)'} />
              {on ? null : <rect x={laneX(l)} y={rowY[i]} width={CELL - 2} height={ROW - 6} rx="1.5" fill="url(#wb-wait)" />}
            </g>
          ))}
          <text x={laneX(W) + 8} y={rowY[i] + 12} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
            {r.count === W ? `all ${W}` : `${r.count} run, ${W - r.count} wait`}
          </text>
        </g>
      ))}
      {f.barrierBefore != null && f.barrierBefore <= last ? (
        <g>
          <rect x={X0} y={rowY[f.barrierBefore] - 17} width={W * CELL - 2} height={13} rx="2" fill="var(--ink)" />
          <text x={X0 + (W * CELL) / 2} y={rowY[f.barrierBefore] - 7} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--page)' }}>
            warp barrier: every lane arrives before any goes on
          </text>
        </g>
      ) : null}
      {f.counters?.map((c, i) => (
        <g key={c.label}>
          <text x={12} y={282 + i * 18} fontSize="12" style={{ fill: 'var(--ink-2)' }}>{c.label}</text>
          <text x={X0 + 60} y={282 + i * 18} fontSize="12.5" fontWeight="650">{c.value}</text>
        </g>
      ))}
      {f.side?.map((s, i) => (
        <text key={i} x={X0 + 190} y={282 + i * 18} fontSize="12">{s}</text>
      ))}
      {f.headline ? <text x={12} y={338} fontSize="13" fontWeight="600">{f.headline}</text> : null}
    </svg>
  );
}

/** The usual counters of a trace cut after its first n rows. */
const counters = (rows: Issue[], uniform?: number) => {
  const useful = rows.reduce((a, r) => a + r.count, 0);
  return [
    { label: 'warp-steps spent', value: String(rows.length) },
    ...(uniform != null ? [{ label: 'a uniform warp needs', value: String(uniform) }] : []),
    { label: 'lanes left waiting', value: `${rows.length * W - useful} lane-steps` },
  ];
};

const WANTS = SPLIT.map((a) => (a ? 'A' : 'B'));
const condRow = BRANCH.issues.slice(0, 1);
const aRows = BRANCH.issues.filter((x) => x.part === 'A');
const bRows = BRANCH.issues.filter((x) => x.part === 'B');
const cRow = BRANCH.issues.slice(-1);

const beats: Beat[] = [
  {
    clip: 'warp-branch-1',
    label: 'In step',
    draw: (p) => {
      const n = Math.max(1, shown(p, 0.08, 0.8, STRAIGHT.issues.length));
      const rows = STRAIGHT.issues.slice(0, n);
      return (
        <Stage rows={rows} labels={STRAIGHT_LINE.map((x) => x.text)}
          counters={[{ label: 'instructions fetched', value: String(n) }, { label: 'lane-operations done', value: String(rows.reduce((a, r) => a + r.count, 0)) }]}
          headline={`${n} ${n === 1 ? 'fetch' : 'fetches'}, each executed by all ${W} lanes`} />
      );
    },
  },
  {
    clip: 'warp-branch-2',
    label: 'The split',
    draw: (p) => (
      <Stage rows={condRow} labels={['test: threadIdx.x < 24']} wants={p > 0.34 ? WANTS : undefined} wantsLabel="wants"
        threshold={p > 0.34 ? T : undefined}
        headline={p > 0.62 ? `${T} lanes want A, ${W - T} want B, and there is one instruction stream` : 'every lane evaluates the condition on its own index'} />
    ),
  },
  {
    clip: 'warp-branch-3',
    label: 'Pass A',
    draw: (p) => {
      const rows = [...condRow, ...aRows.slice(0, Math.max(1, shown(p, 0.12, 0.62, aRows.length)))];
      return (
        <Stage rows={rows} labels={['test: threadIdx.x < 24']} wants={WANTS} wantsLabel="wants" threshold={T} counters={counters(rows)}
          headline={`pass A: ${aRows[0].count} lanes execute, ${W - aRows[0].count} are masked and wait`} />
      );
    },
  },
  {
    clip: 'warp-branch-4',
    label: 'Pass B',
    draw: (p) => {
      const nb = Math.max(1, shown(p, 0.08, 0.5, bRows.length));
      const rows = [...condRow, ...aRows, ...bRows.slice(0, nb), ...(p > 0.6 ? cRow : [])];
      return (
        <Stage rows={rows} labels={['test: threadIdx.x < 24']} wants={WANTS} wantsLabel="wants" threshold={T} counters={counters(rows)}
          headline={p > 0.6 ? `then C, with all ${W} lanes again` : `pass B: ${bRows[0].count} lanes execute, ${W - bRows[0].count} wait`} />
      );
    },
  },
  {
    clip: 'warp-branch-5',
    label: 'The count',
    draw: (p) => {
      const t = T + Math.round(ramp(p, 0.36, 0.62) * (W - T));
      const tr: Trace = t === T ? BRANCH : run(ifElse(W, maskBelow(W, t), COST, COST, `threadIdx.x < ${t}`));
      return (
        <Stage rows={tr.issues} labels={[`test: threadIdx.x < ${t}`]} threshold={t}
          wants={maskBelow(W, t).map((a) => (a ? 'A' : 'B'))} wantsLabel="wants"
          counters={[
            { label: 'warp-steps spent', value: String(tr.spent) },
            { label: 'a uniform warp needs', value: String(tr.uniform) },
            { label: 'lanes left waiting', value: `${tr.idle} lane-steps` },
          ]}
          headline={t === W ? 'the split is on the edge of the warp: one path, no second pass' : `threshold ${t}: two paths inside one warp, two passes`} />
      );
    },
  },
  {
    clip: 'warp-branch-6',
    label: 'A loop',
    draw: (p) => {
      const body = LOOP.issues.filter((x) => x.part === 'body');
      const n = Math.max(1, shown(p, 0.2, 0.86, body.length));
      const rows = body.slice(0, n);
      const cur = rows[rows.length - 1];
      return (
        <Stage rows={rows} wants={TRIPS} wantsLabel="trip count"
          counters={[
            { label: 'iterations run', value: `${n} of ${LOOP.stmts[1].passes}` },
            { label: 'lanes left waiting', value: `${rows.length * W - rows.reduce((a, r) => a + r.count, 0)} lane-steps` },
          ]}
          headline={cur.count === W ? `iteration ${n}: every lane still has work` : `iteration ${n}: ${cur.count} lanes work, ${W - cur.count} have finished and wait`} />
      );
    },
  },
  {
    clip: 'warp-branch-7',
    label: 'No promised order',
    draw: (p) => {
      const mixed = MIXED.issues.slice(1, -1);
      const n = Math.max(1, shown(p, 0.1, 0.52, mixed.length));
      const done = p > 0.72;
      const rows = [...mixed.slice(0, n), ...(done ? MIXED.issues.slice(-1) : [])];
      return (
        <Stage rows={rows} wants={WANTS} wantsLabel="wants" threshold={T} barrierBefore={done ? mixed.length : undefined}
          counters={[{ label: 'warp-steps for A and B', value: `${mixed.length}, in any order` }]}
          side={['NVIDIA: the two paths may be mixed', 'AMD: lockstep is documented', 'Apple: not documented']}
          headline={done ? 'a step that needs every lane comes after a warp barrier' : 'the same instructions, with the two paths taking turns'} />
      );
    },
  },
];

export function WarpBranchScene() {
  return <Scene chapter="ch04" title="A warp at a branch" beats={beats} />;
}
