import { Scene, type Beat } from '../../components/Scene';
import { EXERCISE_1, FIG_11_3, levels, rangeText, run, totalAdditions } from '../../sim/ksblock';
import { sequentialAdditions } from '../../sim/scan';
import { KsGrid, fromLevels, fromTrace, type GridMark } from './KsGrid';

// "Doubling strides" (section 11.2): exercise 1's eight values scanned by Kogge-Stone in
// lockstep, the count of additions, and the book's race replayed. Every cell, arrow and
// number is computed by src/sim/ksblock.ts: levels() for the lockstep steps, run() with
// the book's race order for the last beat. Six beats, one narration clip each
// (content/ch11/narration/ks-doubling-N.txt).

const LV = levels(EXERCISE_1);
const B = EXERCISE_1.length;
const KS = totalAdditions(B);
const LOOP = sequentialAdditions(B);
export const RACE_TRACE = run({ ...FIG_11_3, threads: B, barrier2: false }, EXERCISE_1, 'book-race');
const after = (pred: (s: (typeof RACE_TRACE.steps)[number]) => boolean) => RACE_TRACE.steps.findIndex(pred) + 1;
/** The race's three moments: thread 4 stores early, thread 6 reads it, the end. */
export const RACE_AT = {
  early: after((s) => s.thread === 4 && s.op === 'write' && s.iter === 1),
  read: after((s) => s.thread === 6 && s.op === 'read' && s.iter === 1),
  end: RACE_TRACE.steps.length,
};

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

function Frame({ upto, reveal, hot, marks = [], headline, sub, bars }: {
  upto: number;
  /** 0..1: how much of the newest row is visible (arrows first, then values). */
  reveal: number;
  hot?: number;
  marks?: GridMark[];
  headline: string;
  sub?: string;
  bars?: number;
}) {
  const g = fromLevels(LV, upto, { hot, carries: true });
  const rows = g.rows.map((r, i) => (i === upto && reveal < 0.45 && upto > 0 ? r.map(() => null) : r));
  const arrows = g.arrows.filter((a) => a.to[0] < upto || reveal >= 0.12);
  return (
    <KsGrid
      rows={rows}
      labels={g.labels}
      arrows={arrows}
      marks={marks}
      cellW={66}
      label="Kogge-Stone on exercise 1"
      extraHeight={bars != null ? 132 : 64}
      extra={({ y, height }) => (
        <g>
          {bars != null ? (
            <g data-part="bars" data-ks={KS} data-loop={LOOP}>
              <text x={4} y={y(3) + 70} fontSize="11.5">Kogge-Stone: {LV.slice(1).map((l) => l.added.filter(Boolean).length).join(' + ')} = {KS} additions</text>
              <rect x={4} y={y(3) + 76} width={KS * 22 * bars} height={12} fill="var(--shared)" />
              <text x={4} y={y(3) + 104} fontSize="11.5">the loop of Fig. 11.1: {LOOP} additions</text>
              <rect x={4} y={y(3) + 110} width={LOOP * 22 * bars} height={12} fill="var(--ink-3)" />
            </g>
          ) : null}
          <text x={4} y={height - 26} fontSize="13.5" fontWeight="600" data-part="headline">{headline}</text>
          {sub ? <text x={4} y={height - 8} fontSize="11.5" style={{ fill: 'var(--ink-2)' }} data-part="sub">{sub}</text> : null}
        </g>
      )}
    />
  );
}

function RaceFrame({ p }: { p: number }) {
  const n = p < 0.3 ? RACE_AT.early : p < 0.62 ? RACE_AT.read : RACE_AT.end;
  const g = fromTrace(RACE_TRACE, n);
  const six = RACE_TRACE.outputCoef[6];
  const headline =
    n === RACE_AT.early
      ? 'stride 2 again: thread 4 stores x1..x4 before thread 6 has read'
      : n === RACE_AT.read
        ? 'thread 6 reads the new value and stores x1..x6'
        : `stride 4 adds x0..x2: ${rangeText(six)} = ${RACE_TRACE.output[6]}, not ${RACE_TRACE.reference[6]}`;
  return (
    <KsGrid
      rows={g.rows}
      labels={g.labels}
      arrows={g.arrows}
      marks={n === RACE_AT.end ? [] : g.marks}
      cellW={66}
      label="The race of pp. 257 to 258 on exercise 1"
      extraHeight={64}
      extra={({ height }) => (
        <g>
          <text x={4} y={height - 26} fontSize="13.5" fontWeight="600" data-part="headline">{headline}</text>
          <text x={4} y={height - 8} fontSize="11.5" style={{ fill: 'var(--ink-2)' }} data-part="sub">
            {p < 0.85 ? 'no barrier between the reads and the stores' : 'the fix: add into a private temp, wait at barrier 2, then store'}
          </text>
        </g>
      )}
    />
  );
}

const beats: Beat[] = [
  {
    clip: 'ks-doubling-1',
    label: 'Eight values',
    draw: (p) => <Frame upto={0} reveal={1} headline="each position holds only itself" sub={p > 0.5 ? 'goal: position i holds x0..xi, the total up to and including it' : undefined} />,
  },
  {
    clip: 'ks-doubling-2',
    label: 'Stride 1',
    draw: (p) => (
      <Frame upto={1} hot={1} reveal={ramp(p, 0, 0.6) + (p > 0.6 ? 1 : 0)}
        headline={p < 0.45 ? 'stride 1: every position but 0 adds its left neighbour' : 'windows of 2; position 1 is final'}
        sub={`${LV[1].added.filter(Boolean).length} additions in one step`} />
    ),
  },
  {
    clip: 'ks-doubling-3',
    label: 'Stride 2',
    draw: (p) => (
      <Frame upto={2} hot={2} reveal={ramp(p, 0, 0.5) + (p > 0.5 ? 1 : 0)}
        marks={p > 0.55 ? [{ row: 1, pos: 4, tone: 'read' }, { row: 1, pos: 6, tone: 'read' }, { row: 2, pos: 6, tone: 'write' }] : []}
        headline={p < 0.55 ? 'stride 2: each position adds the pair two to its left' : `position 6 = ${rangeText(LV[1].coef[4])} + ${rangeText(LV[1].coef[6])}, built in the step before`}
        sub="windows of 4; positions 2 and 3 are final" />
    ),
  },
  {
    clip: 'ks-doubling-4',
    label: 'Stride 4',
    draw: (p) => (
      <Frame upto={3} hot={3} reveal={ramp(p, 0, 0.5) + (p > 0.5 ? 1 : 0)}
        headline={`stride 4: windows of 8; every position is final after ${LV.length - 1} steps`}
        sub={p > 0.5 ? `log2 of ${B} is ${LV.length - 1}; for 1024 values it would be 10` : undefined} />
    ),
  },
  {
    clip: 'ks-doubling-5',
    label: 'The count',
    draw: (p) => <Frame upto={3} reveal={1} bars={ramp(p, 0.05, 0.5)} headline={`${KS} additions against ${LOOP}: fewer steps, more work`} />,
  },
  { clip: 'ks-doubling-6', label: 'The race', draw: (p) => <RaceFrame p={p} /> },
];

export function KsDoublingScene() {
  return <Scene chapter="ch11" title="Doubling strides" beats={beats} />;
}
