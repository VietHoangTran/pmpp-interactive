import { useState, type KeyboardEvent, type ReactNode } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { href } from '../../lib/router';
import {
  CHAPTERS,
  EDGES,
  EXTRAS,
  FOURTH_TO_FIFTH,
  GOALS,
  NEW_IN_FIFTH,
  chapter,
  chaptersBefore,
  courseDepartures,
  courseOrder,
  coursePath,
  extra,
  fifthFor,
  firstRenumbered,
  leadsTo,
  optionalPrerequisites,
  readingOrder,
  stageOf,
  type Edge,
  type ExtraId,
  type GoalId,
  type Item,
  type Tier,
} from '../../sim/bookmap';

// ---------------------------------------------------------------- small helpers
/** "2, 3 and 4" */
const list = (xs: (number | string)[]) =>
  xs.length <= 1 ? xs.join('') : xs.slice(0, -1).join(', ') + ' and ' + xs[xs.length - 1];
const edgeKey = (x: { from: number; to: number }) => `${x.from}>${x.to}`;
const isChapter = (i: Item): i is number => typeof i === 'number';

/** Break a short label into at most two lines for a small box. */
function twoLines(label: string): string[] {
  if (label.length <= 14) return [label];
  const words = label.split(' ');
  let best = 1;
  for (let i = 1; i < words.length; i++) {
    const a = words.slice(0, i).join(' ').length;
    const b = words.slice(0, best).join(' ').length;
    if (Math.abs(a - label.length / 2) < Math.abs(b - label.length / 2)) best = i;
  }
  return [words.slice(0, best).join(' '), words.slice(best).join(' ')];
}

// ---------------------------------------------------------------- the chart, in our own layout
// Positions are presentation only; which boxes and arrows exist comes from sim/bookmap.ts.
// Part I runs left to right, Parts II and III top to bottom, so that no arrow has to pass
// under a box it does not belong to.
const BW = 100;
const BH = 34;
const HW = BW / 2;
const HH = BH / 2;
const col = (c: number) => 86 + c * 110;
const AT: Record<number, [number, number]> = {
  1: [col(0), 26],
  // Part I has four columns of its own, spaced wider than the grid below it.
  2: [col(0), 116],
  3: [col(0) + 140, 92],
  4: [col(0) + 140, 140],
  5: [col(0) + 280, 116],
  6: [col(0) + 420, 116],
  7: [col(1), 240],
  15: [col(2), 240],
  9: [col(3), 240],
  10: [col(4), 240],
  13: [col(5), 240],
  8: [col(1), 292],
  11: [col(4), 292],
  12: [col(4), 344],
  14: [col(5), 396],
  19: [col(0), 490],
  23: [col(1), 490],
  16: [col(2), 490],
  21: [col(3), 490],
  17: [col(4), 490],
  20: [col(0), 542],
  18: [col(4), 542],
  22: [col(3), 594],
  24: [col(5), 650],
};
const EXTRA_AT: Record<ExtraId, [number, number]> = {
  appA: [col(0), 732],
  appC: [col(1), 732],
  x1: [col(2), 732],
  x2: [col(3), 732],
  x3: [col(4), 732],
};
const BANDS = [
  { part: 'Part I', top: 64, bottom: 168 },
  { part: 'Part II', top: 200, bottom: 424 },
  { part: 'Part III', top: 446, bottom: 622 },
];
const VIEW_W = 694;
const VIEW_H = 762;

/** From the bottom of one box to the top of another. */
const down = (a: number, b: number, dxa = 0, dxb = 0) => {
  const [xa, ya] = AT[a];
  const [xb, yb] = AT[b];
  const x1 = xa + dxa;
  const y1 = ya + HH;
  const x2 = xb + dxb;
  const y2 = yb - HH;
  const ym = (y1 + y2) / 2;
  return x1 === x2 ? `M${x1} ${y1}V${y2}` : `M${x1} ${y1}C${x1} ${ym} ${x2} ${ym} ${x2} ${y2}`;
};
/** From the right edge of one box to the left edge of another. */
const across = (a: number, b: number) => {
  const [xa, ya] = AT[a];
  const [xb, yb] = AT[b];
  const x1 = xa + HW;
  const x2 = xb - HW;
  const xm = (x1 + x2) / 2;
  return ya === yb ? `M${x1} ${ya}H${x2}` : `M${x1} ${ya}C${xm} ${ya} ${xm} ${yb} ${x2} ${yb}`;
};
const X = (n: number) => AT[n][0];
const Y = (n: number) => AT[n][1];
const ROUTE: Record<string, string> = {
  '1>2': down(1, 2),
  '2>3': across(2, 3),
  '2>4': across(2, 4),
  '3>5': across(3, 5),
  '4>5': across(4, 5),
  '5>6': across(5, 6),
  '6>7': down(6, 7, -36),
  '6>15': down(6, 15, -18),
  '6>9': down(6, 9),
  '6>10': down(6, 10, 18),
  '6>13': down(6, 13, 36),
  // The dashed arrow arches over the gap between the two boxes.
  '9>10': `M${X(9) + 34} ${Y(9) - HH}C${X(9) + 34} ${Y(9) - 42} ${X(10) - 34} ${Y(10) - 42} ${X(10) - 34} ${Y(10) - HH}`,
  '7>8': down(7, 8),
  '7>19': `M${X(7) - HW} ${Y(7)}C${X(19)} ${Y(7)} ${X(19)} ${Y(7) + 10} ${X(19)} ${Y(7) + 50}V${Y(19) - HH}`,
  '9>11': `M${X(9) + 26} ${Y(9) + HH}C${X(9) + 26} ${Y(11)} ${X(11) - 80} ${Y(11)} ${X(11) - HW} ${Y(11)}`,
  '10>11': down(10, 11),
  '11>12': down(11, 12),
  '12>14': `M${X(12) + HW} ${Y(12)}C${X(14) - 18} ${Y(12)} ${X(14) - 18} ${Y(12) + 10} ${X(14) - 18} ${Y(14) - HH}`,
  '13>14': `M${X(13) + 18} ${Y(13) + HH}V${Y(14) - HH}`,
  // Both arrows out of chapter 9 share a trunk, then fork in the gap between the parts.
  '9>21': `M${X(9)} ${Y(9) + HH}V${Y(21) - HH}`,
  '9>17': `M${X(9)} ${Y(9) + HH}V428C${X(9)} 456 ${X(17)} 440 ${X(17)} ${Y(17) - HH}`,
  '8>23': down(8, 23),
  '8>16': `M${X(8) + HW} ${Y(8)}C${X(16)} ${Y(8)} ${X(16)} ${Y(8) + 10} ${X(16)} ${Y(8) + 50}V${Y(16) - HH}`,
  '19>20': down(19, 20),
  '17>18': down(17, 18),
  '21>22': down(21, 22),
  '16>22': `M${X(16)} ${Y(16) + HH}V${Y(22) - 40}C${X(16)} ${Y(22)} ${X(22) - 90} ${Y(22)} ${X(22) - HW} ${Y(22)}`,
  '18>22': `M${X(18)} ${Y(18) + HH}C${X(18)} ${Y(22)} ${X(22) + 90} ${Y(22)} ${X(22) + HW} ${Y(22)}`,
};

type BoxState = 'target' | 'pre' | 'added' | 'later' | 'dim';

function Box({
  x,
  y,
  num,
  label,
  tier,
  state,
  shade,
  ordinal,
  attrs,
  onPick,
  aria,
}: {
  x: number;
  y: number;
  num: string;
  label: string;
  tier: Tier;
  state: BoxState;
  shade: boolean;
  ordinal?: number;
  attrs: Record<string, string>;
  onPick?: () => void;
  aria: string;
}) {
  const x0 = x - HW;
  const y0 = y - HH;
  const tab = 24;
  const filled = shade && tier === 'Deep';
  const tabFilled = shade && tier !== 'Condensed';
  const lines = twoLines(label);
  const onKey = (ev: KeyboardEvent) => {
    if (ev.key === 'Enter' || ev.key === ' ') {
      ev.preventDefault();
      onPick?.();
    }
  };
  return (
    <g
      {...attrs}
      data-state={state}
      data-tier={tier}
      role={onPick ? 'button' : undefined}
      tabIndex={onPick ? 0 : undefined}
      aria-pressed={onPick ? state === 'target' : undefined}
      aria-label={aria}
      onClick={onPick}
      onKeyDown={onPick ? onKey : undefined}
      style={{ cursor: onPick ? 'pointer' : 'default', opacity: state === 'dim' ? 0.42 : 1 }}
    >
      {state === 'added' ? (
        <rect x={x0 - 3.5} y={y0 - 3.5} width={BW + 7} height={BH + 7} rx="5" fill="none" stroke="var(--ink)" strokeWidth="1" />
      ) : null}
      <rect
        x={x0}
        y={y0}
        width={BW}
        height={BH}
        rx="3"
        fill={filled ? 'var(--ink)' : 'var(--cell)'}
        stroke="var(--ink)"
        strokeWidth={state === 'target' ? 3 : 1.2}
        strokeDasharray={state === 'later' ? '5 3' : undefined}
      />
      {tabFilled && !filled ? (
        <path d={`M${x0 + 3} ${y0}H${x0 + tab}V${y0 + BH}H${x0 + 3}a3 3 0 0 1 -3 -3V${y0 + 3}a3 3 0 0 1 3 -3z`} fill="var(--ink)" />
      ) : (
        <line x1={x0 + tab} x2={x0 + tab} y1={y0} y2={y0 + BH} stroke={filled ? 'var(--page)' : 'var(--ink)'} strokeOpacity="0.3" />
      )}
      <text x={x0 + tab / 2} y={y + 4.5} fontSize="12.5" fontWeight="700" textAnchor="middle" style={tabFilled ? { fill: 'var(--page)' } : undefined}>
        {num}
      </text>
      {lines.map((l, i) => (
        <text
          key={i}
          x={x0 + tab + 5}
          y={lines.length === 1 ? y + 3.5 : y - 2 + i * 11}
          fontSize="9.3"
          style={filled ? { fill: 'var(--page)' } : undefined}
        >
          {l}
        </text>
      ))}
      {ordinal != null ? (
        <g data-ordinal={ordinal}>
          <circle cx={x + HW - 1} cy={y0 + 1} r="8.5" fill="var(--page)" stroke="var(--ink)" strokeWidth="1" />
          <text x={x + HW - 1} y={y0 + 4.2} fontSize="8.6" fontWeight="650" textAnchor="middle">
            {ordinal}
          </text>
        </g>
      ) : null}
    </g>
  );
}

type Selection = { kind: 'goal'; id: GoalId } | { kind: 'chapter'; n: number };
type Mode = 'book' | 'course';

/** Why this course reads a chapter after something the chart puts it before. Ours. */
function whyLater(n: number): string {
  if (n === 13) return 'condensed here: the radix sort this course needs builds on scan and filter';
  if (n === 7 || n === 19) return 'read after attention: chapter 20 leans on chapters 5, 10 and 15';
  return chapter(n).tier === 'Condensed'
    ? 'condensed here, in the last stage'
    : `in a later stage (${stageOf(n).toLowerCase()})`;
}

function Chart({
  on,
  targets,
  added,
  later,
  extrasOn,
  extrasGoal,
  mode,
  shade,
  onPick,
}: {
  on: Set<number>;
  targets: number[];
  added: number[];
  later: number[];
  extrasOn: Set<ExtraId>;
  extrasGoal: ExtraId[];
  mode: Mode;
  shade: boolean;
  onPick: (n: number) => void;
}) {
  const departures = new Set(mode === 'course' ? courseDepartures().map(edgeKey) : []);
  const order = courseOrder();
  const ordinal = (i: Item) => (mode === 'course' ? order.indexOf(i) + 1 : undefined);
  const stateOf = (n: number): BoxState =>
    targets.includes(n) ? 'target' : later.includes(n) ? 'later' : added.includes(n) ? 'added' : on.has(n) ? 'pre' : 'dim';
  const edgeOn = (x: Edge) => on.has(x.from) && on.has(x.to);
  return (
    <svg
      viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
      role="group"
      aria-label="The book's chapters and their prerequisite arrows"
      style={{ width: '100%', minWidth: 600, maxWidth: VIEW_W, display: 'block' }}
    >
      <defs>
        <marker id="s17-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" orient="auto">
          <path d="M0 0.6L8 4L0 7.4z" style={{ fill: 'var(--ink)' }} />
        </marker>
      </defs>
      {BANDS.map((b) => (
        <g key={b.part}>
          <rect x="28" y={b.top} width={VIEW_W - 30} height={b.bottom - b.top} rx="4" fill="var(--page)" stroke="var(--rule)" />
          <text
            x="17"
            y={(b.top + b.bottom) / 2}
            fontSize="11"
            fontWeight="620"
            textAnchor="middle"
            transform={`rotate(-90 17 ${(b.top + b.bottom) / 2})`}
            style={{ fill: 'var(--ink-2)' }}
          >
            {b.part}
          </text>
        </g>
      ))}
      <rect x="28" y="686" width={VIEW_W - 30} height="72" rx="4" fill="none" stroke="var(--rule)" strokeDasharray="3 3" />
      <text x="36" y="702" fontSize="10" style={{ fill: 'var(--ink-2)' }}>
        Not boxes of the book's chart. A and C are the book's appendices; X1 to X3 are this course's modules.
      </text>
      {EDGES.map((x) => {
        const k = edgeKey(x);
        const active = edgeOn(x);
        const ahead = departures.has(k);
        return (
          <path
            key={k}
            data-edge={k}
            data-on={active ? '1' : '0'}
            data-kind={x.dashed ? 'dashed' : x.uncertain ? 'uncertain' : 'solid'}
            data-ahead={ahead ? '1' : undefined}
            d={ROUTE[k] ?? down(x.from, x.to)}
            fill="none"
            stroke="var(--ink)"
            strokeWidth={ahead ? 3 : 1.3}
            strokeDasharray={x.dashed ? '6 4' : x.uncertain ? '1.5 3.5' : undefined}
            strokeLinecap={x.uncertain ? 'round' : undefined}
            markerEnd="url(#s17-arrow)"
            opacity={active ? 1 : ahead ? 0.55 : 0.25}
          />
        );
      })}
      {CHAPTERS.map((c) => (
        <Box
          key={c.n}
          x={AT[c.n][0]}
          y={AT[c.n][1]}
          num={String(c.n)}
          label={c.short}
          tier={c.tier}
          state={stateOf(c.n)}
          shade={shade}
          ordinal={ordinal(c.n)}
          attrs={{ 'data-ch': String(c.n) }}
          onPick={() => onPick(c.n)}
          aria={`Chapter ${c.n}: ${c.title}`}
        />
      ))}
      {EXTRAS.map((x) => (
        <Box
          key={x.id}
          x={EXTRA_AT[x.id][0]}
          y={EXTRA_AT[x.id][1]}
          num={x.num}
          label={x.short}
          tier={x.tier}
          state={extrasGoal.includes(x.id) ? 'target' : extrasOn.has(x.id) ? 'pre' : 'dim'}
          shade={shade}
          ordinal={ordinal(x.id)}
          attrs={{ 'data-extra': x.id }}
          aria={`${x.pages ? 'Appendix' : 'Module'} ${x.num}: ${x.title}`}
        />
      ))}
    </svg>
  );
}

/** A short line sample for the legend. */
function LineSample({ dash, width = 1.3, round }: { dash?: string; width?: number; round?: boolean }) {
  return (
    <svg width="34" height="10" viewBox="0 0 34 10" aria-hidden="true" style={{ verticalAlign: 'middle', marginRight: '0.35em' }}>
      <path d="M1 5H27" fill="none" stroke="var(--ink)" strokeWidth={width} strokeDasharray={dash} strokeLinecap={round ? 'round' : undefined} />
      <path d="M26 1.6L33 5L26 8.4z" style={{ fill: 'var(--ink)' }} />
    </svg>
  );
}
function BoxSample({ kind }: { kind: 'Deep' | 'Standard' | 'Condensed' | 'target' | 'later' | 'added' }) {
  return (
    <svg width="26" height="14" viewBox="0 0 26 14" aria-hidden="true" style={{ verticalAlign: 'middle', marginRight: '0.35em' }}>
      {kind === 'added' ? <rect x="0.5" y="0.5" width="25" height="13" rx="3" fill="none" stroke="var(--ink)" strokeWidth="0.8" /> : null}
      <rect
        x={kind === 'added' ? 3 : 1.5}
        y={kind === 'added' ? 3 : 1.5}
        width={kind === 'added' ? 20 : 23}
        height={kind === 'added' ? 8 : 11}
        rx="2"
        fill={kind === 'Deep' ? 'var(--ink)' : 'var(--cell)'}
        stroke="var(--ink)"
        strokeWidth={kind === 'target' ? 2.6 : 1.1}
        strokeDasharray={kind === 'later' ? '4 2.5' : undefined}
      />
      {kind === 'Standard' ? <rect x="1.5" y="1.5" width="8" height="11" fill="var(--ink)" /> : null}
    </svg>
  );
}

function ReadingPath() {
  const [sel, setSel] = useState<Selection>({ kind: 'goal', id: 'matmul' });
  const [mode, setMode] = useState<Mode>('book');
  const [shade, setShade] = useState(false);

  const g = sel.kind === 'goal' ? GOALS.find((x) => x.id === sel.id)! : null;
  const targets = g ? g.chapters : [(sel as { n: number }).n];
  const extrasGoal: ExtraId[] = g ? g.extras : [];
  const single = targets.length === 1 ? targets[0] : null;

  // Everything below comes from the model.
  const path = coursePath(targets, extrasGoal);
  const order: Item[] = mode === 'book' ? [...readingOrder(targets), ...extrasGoal] : path.order;
  const later = mode === 'course' ? path.later : [];
  const added = mode === 'course' ? path.added : [];
  const chaptersInOrder = order.filter(isChapter);
  const on = new Set<number>([...chaptersInOrder, ...later]);
  const extrasOn = new Set<ExtraId>(order.filter((i): i is ExtraId => !isChapter(i)));
  const shown = order.filter((i) => i !== 1);
  const before = shown.filter((i) => isChapter(i) && !targets.includes(i)).length;
  const total = shown.filter(isChapter).length;
  const optional = optionalPrerequisites([...chaptersInOrder, ...later]).filter((n) => !on.has(n));
  const builtOn = single != null ? leadsTo([single]) : [];
  const offChart = mode === 'book' && targets.length === 0;

  const label = g
    ? `${g.label}: ${[
        g.chapters.length ? (g.chapters.length === 1 ? 'chapter ' : 'chapters ') + list(g.chapters) : '',
        ...g.extras.map((x) => (extra(x).pages ? 'Appendix ' : 'module ') + extra(x).num),
      ]
        .filter(Boolean)
        .join(', ')}`
    : `Chapter ${single}: ${chapter(single!).title}`;

  const row = (i: Item) => {
    if (isChapter(i)) {
      const c = chapter(i);
      const isTarget = targets.includes(i);
      return (
        <li key={i} data-item={i} style={{ marginBottom: '0.3em' }}>
          <span style={{ fontWeight: isTarget ? 700 : 500 }}>
            {c.n}. <a href={href(c.id)}>{c.title}</a>
          </span>{' '}
          <span className="muted">
            pp. {c.pages}, {c.tier.toLowerCase()}
            {isTarget ? ', your selection' : ''}
            {added.includes(i) ? ', added by this course' : ''}
          </span>
        </li>
      );
    }
    const x = extra(i);
    return (
      <li key={i} data-item={i} style={{ marginBottom: '0.3em' }}>
        <span style={{ fontWeight: extrasGoal.includes(i) ? 700 : 500 }}>
          {x.num}. <a href={href(x.id)}>{x.title}</a>
        </span>{' '}
        <span className="muted">{x.pages ? `appendix, pp. ${x.pages}` : "this course's module, not in the book"}</span>
      </li>
    );
  };

  return (
    <Bench
      title="Reading path"
      task="Pick a goal or click a chapter. Then switch to this course's order and see what moves."
      foot={
        <>
          Arrows as read from an image of the book's Fig. P.1 (p. xxiii), redrawn in our own layout. The two dotted
          arrows come from a junction that is ambiguous in the PDF. Goals, depth tiers and this course's order are
          ours (PLAN.md).
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="muted">Goal</span>
        <span className="seg" data-role="goals" style={{ flexWrap: 'wrap' }}>
          {GOALS.map((x) => (
            <button
              key={x.id}
              aria-pressed={sel.kind === 'goal' && sel.id === x.id}
              onClick={() => {
                setSel({ kind: 'goal', id: x.id });
                // Module X3 is not on the chart: only this course's order can place it.
                if (!x.chapters.length) setMode('course');
              }}
            >
              {x.label}
            </button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem' }}>
        <span className="muted">Order</span>
        <span className="seg" data-role="mode">
          <button aria-pressed={mode === 'book'} onClick={() => setMode('book')}>
            The book's arrows
          </button>
          <button aria-pressed={mode === 'course'} onClick={() => setMode('course')}>
            This course's order
          </button>
        </span>
        <label className="check">
          <input type="checkbox" checked={shade} onChange={(ev) => setShade(ev.target.checked)} />
          shade boxes by this course's depth
        </label>
      </div>
      <div className="row" style={{ alignItems: 'flex-start', gap: '1rem' }}>
        <div style={{ flex: '1 1 38rem', minWidth: 0, overflowX: 'auto' }}>
          <Chart
            on={on}
            targets={targets}
            added={added}
            later={later}
            extrasOn={extrasOn}
            extrasGoal={extrasGoal}
            mode={mode}
            shade={shade}
            onPick={(n) => setSel({ kind: 'chapter', n })}
          />
        </div>
        <div className="panel" data-role="path" style={{ flex: '1 1 13.5rem', minWidth: 0 }}>
          <p style={{ margin: '0 0 0.4rem' }}>
            <b data-role="selection">{label}</b>
          </p>
          {offChart ? (
            <p data-role="off-chart" style={{ margin: 0 }}>
              Module X3 is not on the book's chart, so the arrows say nothing about it. Switch to this course's order.
            </p>
          ) : (
            <>
              <p style={{ margin: '0 0 0.6rem' }}>
                Before the selection: <b data-stat="before">{before}</b> {before === 1 ? 'chapter' : 'chapters'}. With
                it: <b data-stat="total">{total}</b>.{' '}
                <span className="muted">
                  {mode === 'book' ? "By the book's arrows." : "In this course's order."} Chapter 1 is not counted: you
                  are in it.
                </span>
              </p>
              <ol data-role="order" style={{ margin: 0, paddingLeft: '1.2em', listStyle: 'none' }}>
                {shown.map(row)}
              </ol>
              {later.length ? (
                <>
                  <p style={{ margin: '0.7rem 0 0.3rem' }}>
                    The book's arrows also put {later.length === 1 ? 'this' : 'these'} first. This course reads{' '}
                    {later.length === 1 ? 'it' : 'them'} afterwards:
                  </p>
                  <ul data-role="later" style={{ margin: 0, paddingLeft: '1.2em' }}>
                    {later.map((n) => (
                      <li key={n} data-item={n} style={{ marginBottom: '0.3em' }}>
                        {n}. <a href={href(chapter(n).id)}>{chapter(n).title}</a>{' '}
                        <span className="muted">({whyLater(n)})</span>
                      </li>
                    ))}
                  </ul>
                </>
              ) : null}
              {optional.length ? (
                <p data-role="optional" className="muted" style={{ margin: '0.7rem 0 0' }}>
                  Optional by the chart's dashed arrow: chapter {list(optional)} before chapter 10.
                </p>
              ) : null}
              {single != null ? (
                <p data-role="leads" className="muted" style={{ margin: '0.7rem 0 0' }}>
                  {builtOn.length
                    ? `Chapters that build on chapter ${single}: ${list(builtOn)}.`
                    : single === 24
                      ? 'Chapter 24 has no arrows in the chart.'
                      : `Nothing in the chart builds on chapter ${single}. It leads nowhere further.`}
                </p>
              ) : null}
            </>
          )}
        </div>
      </div>
      <div className="legend" data-role="legend" style={{ marginTop: '0.8rem' }}>
        <span>
          <BoxSample kind="target" />
          your selection
        </span>
        <span>
          <LineSample />
          prerequisite
        </span>
        <span>
          <LineSample dash="6 4" />
          drawn dashed in the book (9 to 10)
        </span>
        <span>
          <LineSample dash="1.5 3.5" round />
          read from the ambiguous junction (9 to 17, 9 to 21)
        </span>
        {mode === 'course' ? (
          <>
            <span data-legend="ahead">
              <LineSample width={3} />
              an arrow this course runs ahead of
            </span>
            <span>
              <BoxSample kind="later" />
              read after your selection
            </span>
            <span>
              <BoxSample kind="added" />
              added by this course
            </span>
            <span>numbers in circles: position in this course's order</span>
          </>
        ) : null}
        {shade ? (
          <>
            <span data-legend="tier">
              <BoxSample kind="Deep" />
              deep
            </span>
            <span>
              <BoxSample kind="Standard" />
              standard
            </span>
            <span>
              <BoxSample kind="Condensed" />
              condensed
            </span>
          </>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- edition lookup
function EditionLookup() {
  const [n, setN] = useState(14);
  const [text, setText] = useState('14');
  const pick = (v: number) => {
    setN(v);
    setText(String(v));
  };
  const c4 = FOURTH_TO_FIFTH[n - 1];
  const to = fifthFor(n);
  const nowAtOldNumber = chapter(n);
  const unchanged = FOURTH_TO_FIFTH.filter((c) => c.fate.kind === 'same').length;
  const movedCount = FOURTH_TO_FIFTH.filter((c) => c.fate.kind === 'moved').length;
  const removed = FOURTH_TO_FIFTH.filter((c) => c.fate.kind === 'removed').map((c) => c.n);
  const cell = {
    height: '1.9rem',
    minWidth: '1.75rem',
    display: 'grid',
    placeItems: 'center',
    padding: 0,
    borderRadius: 2,
    fontVariantNumeric: 'tabular-nums',
  } as const;
  const grid = { display: 'grid', gridTemplateColumns: '8.5rem repeat(24, minmax(1.75rem, 1fr))', gap: 2, alignItems: 'center', minWidth: '52rem' } as const;
  return (
    <Bench
      title="Which edition is that page about?"
      task="Type or click the chapter number a page written before 2026 cites."
      foot={
        <>
          The 4th-edition contents are from the publisher's page. Pairing the two editions by title is our inference
          (notes/today/ch01-facts.md, A15). The 5th-edition side is the printed book.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <label className="field">
          4th-edition chapter (1 to {FOURTH_TO_FIFTH.length})
          <input
            type="number"
            min="1"
            max={FOURTH_TO_FIFTH.length}
            step="1"
            value={text}
            aria-label="4th-edition chapter number"
            onChange={(ev) => {
              setText(ev.target.value);
              const v = Number(ev.target.value);
              if (Number.isInteger(v) && v >= 1 && v <= FOURTH_TO_FIFTH.length) setN(v);
            }}
          />
        </label>
      </div>
      <div style={{ overflowX: 'auto', marginBottom: '0.9rem' }}>
        <div style={grid} data-role="fourth">
          <span className="muted">4th edition (2022)</span>
          {FOURTH_TO_FIFTH.map((c) => (
            <button
              key={c.n}
              className="quiet"
              data-n={c.n}
              data-fate={c.fate.kind}
              aria-pressed={c.n === n}
              aria-label={`4th edition, chapter ${c.n}`}
              onClick={() => pick(c.n)}
              style={{ ...cell, textDecoration: c.fate.kind === 'removed' ? 'line-through' : undefined }}
            >
              {c.n}
            </button>
          ))}
          <span />
        </div>
        <div style={{ ...grid, marginTop: 4 }} data-role="fifth">
          <span className="muted">5th edition</span>
          {CHAPTERS.map((c) => {
            const hit = to === c.n;
            const isNew = NEW_IN_FIFTH.includes(c.n);
            return (
              <span
                key={c.n}
                data-n={c.n}
                data-hit={hit ? '1' : undefined}
                data-new={isNew ? '1' : undefined}
                title={c.title}
                style={{
                  ...cell,
                  border: `1px ${isNew ? 'dashed' : 'solid'} ${hit || isNew ? 'var(--ink)' : 'var(--rule)'}`,
                  background: hit ? 'var(--ink)' : 'var(--cell)',
                  color: hit ? 'var(--page)' : 'var(--ink)',
                  fontWeight: hit ? 700 : 450,
                }}
              >
                {c.n}
              </span>
            );
          })}
        </div>
      </div>
      <div className="cols">
        <div className="panel" data-role="edition-result">
          <p style={{ margin: '0 0 0.4rem' }}>
            <b>4th edition, chapter {c4.n}:</b> {c4.title}
          </p>
          {c4.fate.kind === 'same' ? (
            <p style={{ margin: 0 }}>
              <b>Same number.</b> In the 5th edition it is chapter {c4.n}, {nowAtOldNumber.title}:{' '}
              <a href={href(nowAtOldNumber.id)}>this course's chapter {c4.n}</a>. The number held; the content may not
              have (see the table above).
            </p>
          ) : c4.fate.kind === 'moved' ? (
            <p style={{ margin: 0 }}>
              <b>New number: chapter {c4.fate.to}</b>, {chapter(c4.fate.to).title}:{' '}
              <a href={href(chapter(c4.fate.to).id)}>this course's chapter {c4.fate.to}</a>. In the 5th edition,
              chapter {c4.n} is {nowAtOldNumber.title}
              {NEW_IN_FIFTH.includes(c4.n) ? ', a new chapter' : ''}.
            </p>
          ) : (
            <p style={{ margin: 0 }}>
              <b>Removed.</b> {c4.fate.note}{' '}
              {c4.fate.foldedInto ? (
                <>
                  See <a href={href(chapter(c4.fate.foldedInto).id)}>this course's chapter {c4.fate.foldedInto}</a>.{' '}
                </>
              ) : null}
              {c4.fate.appendix ? (
                <>
                  See <a href={href('app' + c4.fate.appendix)}>Appendix {c4.fate.appendix}</a>.{' '}
                </>
              ) : null}
              In the 5th edition, chapter {c4.n} is {nowAtOldNumber.title}.
            </p>
          )}
        </div>
        <div className="panel" data-role="edition-summary">
          <p style={{ margin: '0 0 0.4rem' }}>
            <b>Across the whole list</b>
          </p>
          <p style={{ margin: 0 }}>
            Same number: chapters 1 to {firstRenumbered() - 1} ({unchanged}). New number: {movedCount}, every surviving
            chapter from {firstRenumbered()} on. Removed: {removed.length} (struck through: {list(removed)}). New in the
            5th edition, with no 4th-edition counterpart: {list(NEW_IN_FIFTH)} (dashed).
          </p>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
const BEFORE_20 = chaptersBefore([20]);
const PART2_BEFORE_14 = chaptersBefore([14]).filter((n) => chapter(n).part === 'II');

export const questions17: Record<string, Question> = {
  'ch01/1.7/q-postpone': {
    kind: 'choice',
    id: 'ch01/1.7/q-postpone',
    prompt: (
      <>
        You want a fast GEMM as soon as possible. By the book's arrows, which of these can wait until afterwards without
        breaking a prerequisite of chapter 15?
      </>
    ),
    options: [
      { label: 'Chapter 4, compute architecture and scheduling.', why: 'Chapter 4 feeds chapter 5, and 5 feeds 6.' },
      { label: 'Chapter 5, memory architecture and data locality.', why: 'Chapter 6 has one arrow into it, and it comes from 5.' },
      { label: 'Chapter 6, performance considerations.', why: 'The only arrow into chapter 15 comes from chapter 6.' },
      { label: 'Chapter 7, convolution.', correct: true },
    ],
    explain: (
      <p>
        Chapter 15 hangs directly off chapter 6, and 6 needs all of Part I: 5 before it, 3 and 4 before 5, 2 before
        those. Chapter 7 opens a different branch (8, 19 and what follows them). Nothing on the way to 15 passes
        through it.
      </p>
    ),
    source: 'Fig. P.1, p. xxiii',
  },
  'ch01/1.7/q-edition16': {
    kind: 'numeric',
    id: 'ch01/1.7/q-edition16',
    prompt: (
      <>
        A 2023 post says "PMPP chapter 16 shows how convolution layers are lowered to matrix multiplication". Which
        chapter of the 5th edition do you open?
      </>
    ),
    answer: fifthFor(16)!,
    explain: (
      <p>
        A post from 2023 can only mean the 4th edition (2022), whose chapter 16 is Deep learning. That material is
        chapter {fifthFor(16)} now, Convolutional neural networks. Chapter 16 of the 5th edition is wavefront
        parallelism, which has nothing to do with the post.
      </p>
    ),
    source: 'Preface pp. xviii–xxii; notes/today/ch01-facts.md, A15',
  },
  'ch01/1.7/q-sortneeds': {
    kind: 'numeric',
    id: 'ch01/1.7/q-sortneeds',
    prompt: (
      <>
        Top-p sampling wants sorted candidates. By the book's arrows, how many Part II chapters come before chapter 14?
      </>
    ),
    answer: PART2_BEFORE_14.length,
    explain: (
      <p>
        {list(PART2_BEFORE_14)}: histogram, reduction, scan, filter and merge. This course goes deep on 10, 11 and 12,
        because a softmax normaliser is a reduction, cumulative probability is a scan and keeping the top candidates is
        a filter. It condenses 13: merge is on the path for merge sort, and the sort this course needs is radix sort,
        which the Preface describes as built on the stable filter of chapter 12.
      </p>
    ),
    source: 'Fig. P.1, p. xxiii; Preface p. xx',
  },
  'ch01/1.7/q-portable': {
    kind: 'choice',
    id: 'ch01/1.7/q-portable',
    prompt: (
      <>
        A kernel passes its tests unchanged on a second vendor's GPU and runs at a third of the speed you expected.
        Which statement does this illustrate?
      </>
    ),
    options: [
      {
        label: 'A program that is correct on every supporting device usually needs modification to perform well on a new one.',
        correct: true,
      },
      {
        label: "Amdahl's law.",
        why: 'Amdahl is about the share of the run you did not speed up. Nothing here says anything about that share.',
      },
      {
        label: 'The port must contain a bug.',
        why: 'The tests pass. A slow kernel with correct output is a tuning problem until shown otherwise.',
      },
    ],
    explain: (
      <p>
        The book makes the remark about OpenCL (p. 13). AMD says the same about bringing CUDA code to HIP: expect manual
        porting and performance tuning. Correctness travels with the source. Performance has to be earned again on each
        device, by measuring there.
      </p>
    ),
    source: 'book p. 13; notes/today/ch01-facts.md, B2',
  },
};
/** The same record under the name the chapter-building rules use. */
export const questionsS17 = questions17;

// ---------------------------------------------------------------- what the 5th edition changed
const CHANGES: { kind: string; what: ReactNode; change: ReactNode; you: ReactNode }[] = [
  {
    kind: 'New',
    what: '12 Filtering',
    change: 'Unstable and stable filter; warp voting; coalesced atomic operations.',
    you: 'Keeping the top-k or top-p candidates is a filter.',
  },
  {
    kind: 'New',
    what: '15 Advanced optimizations for matrix multiplication',
    change: 'Coarsening, register tiling, vector loads and stores, bank conflicts, software pipelining; a survey of tensor cores and asynchronous copies.',
    you: 'Your first goal. The Preface calls matmul the most important computation GPUs are used for today.',
  },
  {
    kind: 'New',
    what: '16 Wavefront parallelism',
    change: 'Data parallelism out of dynamic programming, with sequence alignment as the example.',
    you: 'Condensed in this course.',
  },
  {
    kind: 'New',
    what: '20 Large language models',
    change: 'Attention over tokens, KV caching, flash attention, batching.',
    you: 'Your first goal again: the only attention kernels in the book.',
  },
  {
    kind: 'Rewritten',
    what: '6 Performance considerations',
    change: 'A longer checklist: loop unrolling, vector loads and stores, bank conflicts, double buffering, warp-level primitives.',
    you: 'The checklist the later chapters apply, 15 and 20 included.',
  },
  {
    kind: 'Rewritten',
    what: '11 Scan',
    change: 'More on work efficiency and on optimising one scan design; new variants; a corrected lookback using C++ atomics.',
    you: 'Cumulative probability for top-p is a scan.',
  },
  {
    kind: 'Rewritten',
    what: '23 Multi-GPU programming',
    change: 'A new running example and three models: CUDA-aware MPI, NCCL, NVSHMEM.',
    you: 'Condensed in this course, which keeps streams, asynchronous copies and pinned memory.',
  },
  {
    kind: 'Updated',
    what: '4 and 5',
    change: 'Examples on the H100; thread block clusters; distributed shared memory; a fuller treatment of arithmetic intensity.',
    you: "The execution and memory model everything else rests on. The device figures are the book's example, never yours.",
  },
  {
    kind: 'Updated',
    what: '9 Histogram',
    change: 'The CUDA C++ atomics API, with scope and memory ordering; code in other chapters moved to it too.',
    you: 'The spelling of atomics you will meet in every later listing.',
  },
  {
    kind: 'Updated',
    what: '10 Reduction',
    change: 'Warp-level primitives, warp shuffle above all.',
    you: 'Softmax normaliser, max and argmax; the shuffles the attention kernel uses.',
  },
  {
    kind: 'Updated',
    what: '14, 17, 18, 19',
    change: 'Odd-even sort added, radix sort tied to the stable filter; a column-wise sparse format; one kernel for all levels of a graph search; the ML background of 19 moved to a new Appendix B.',
    you: '14 gives sorted candidates. 19 is where convolution is lowered to GEMM.',
  },
  {
    kind: 'Updated',
    what: 'Appendix A',
    change: 'Low-precision floating-point formats, motivated by quantisation of LLM parameters.',
    you: "The anchor for this course's quantised-kernel module.",
  },
  {
    kind: 'Removed',
    what: 'Three chapters',
    change: 'Iterative MRI reconstruction (parts folded into 21); dynamic parallelism; other advanced practices (its history of the CUDA memory model is now Appendix C).',
    you: 'Older material that cites them has no chapter to open here.',
  },
];

// ---------------------------------------------------------------- the section
export function S17() {
  return (
    <>
      <p>
        This page is about using the book, not about GPUs. Two things save time later: knowing which chapters your
        goals depend on, and knowing which edition a page you found online is talking about.
      </p>

      <h2 style={{ marginTop: '1.6rem' }}>The shape of the book</h2>
      <p>
        Three parts between two bookends. <b>Part I</b>, chapters 2 to 6, is the fundamentals: how a kernel is written
        and launched, how threads map to data, how the hardware schedules them, where data lives, and a checklist of
        optimisations that the rest of the book reuses. <b>Part II</b>, chapters 7 to 15, takes one parallel pattern per
        chapter. Each is there for a technique it forces on you: histogram for atomic operations, reduction for the
        cost of barriers, scan for work efficiency. It ends with matrix multiplication in depth. <b>Part III</b>,
        chapters 16 to 23, is advanced patterns and applications, where the subject is choosing between ways to
        decompose a problem. Chapter 1 sits above the parts and chapter 24 closes the book. (The Preface says "four
        major parts" once, on p. xxii, and then describes three. The chart and §1.7 both say three.)
      </p>
      <Source section="1.7" pages="15–18" figs="Preface pp. xxii–xxiv" />

      <h2>What your goals depend on</h2>
      <p>
        The Preface draws the chapters as a chart of prerequisite arrows, so that an instructor can skip or reorder
        chapters without leaving a gap. You can use it the same way. Before you look at it, commit to a number.
      </p>
      <Aside>
        <p>
          <b>Book</b> Fig. P.1, p. xxiii. The arrows are the authors' advice to instructors (p. xxv), not hard
          dependencies. One junction of the chart is ambiguous in the PDF: the line from chapter 9 to chapters 17 and
          21 passes directly under the box of chapter 14. The arrow from 9 to 10 is drawn dashed.
        </p>
      </Aside>
      <Predict
        id="ch01/1.7/p-before20"
        prompt={
          <>
            Chapter 20 is the one on attention. By the book's prerequisite arrows, how many chapters must you have read
            before it, not counting chapter 1 or chapter 20 itself?
          </>
        }
        unit="chapters"
      >
        {(v) => (
          <p>
            <b>{BEFORE_20.length}</b>: chapters {list(BEFORE_20)}.{' '}
            {v === BEFORE_20.length
              ? 'That is what you had.'
              : v === 19
                ? 'Nineteen would be right if the chart were a line. It is not: most of Part II and Part III sits on branches that chapter 20 never touches.'
                : v === chaptersBefore([15]).length
                  ? 'Five is the answer for chapter 15, which needs Part I and nothing else. Chapter 20 adds convolution and CNNs.'
                  : v > BEFORE_20.length
                    ? 'Fewer than you thought: the chart branches after chapter 6, and chapter 20 sits on one short branch.'
                    : 'All of Part I is on the way, and then two more: convolution and CNNs.'}{' '}
            Check it in the explorer below.
          </p>
        )}
      </Predict>
      <p>
        The explorer is the same chart, redrawn. Pick a goal, or click any chapter, and it keeps what the arrows put
        before your choice and dims the rest.
      </p>
      <Simplification>
        <p>
          A chapter is one box. In the book a chapter often needs only a section or two of an earlier one, which the
          chart cannot show.
        </p>
        <p>
          The arrows were read from an image and one junction is ambiguous (see the note above). Which chapters a goal
          ends at, the depth tiers and "this course's order" are ours, from PLAN.md, not the book's.
        </p>
      </Simplification>
      <ReadingPath />
      <p>
        Four things to find. Chapter 15, the fast matmul, needs five chapters and nothing from Part II: you can go from
        chapter 6 straight to it. Chapter 20 needs seven. Sorting, chapter 14, pulls in ten, five of them in Part II,
        which is why sampling is the longest of your roads. And five chapters lead nowhere further (14, 15, 20, 22 and
        23), so nothing downstream waits on them.
      </p>
      <Ask q={questions17['ch01/1.7/q-postpone']} />

      <h3>Where this course does not follow the arrows</h3>
      <p>
        Switch the explorer to this course's order. It takes the book by what your goals need first: foundations,
        matmul, the sampling chain, attention and batching, quantisation, then the rest. The arrows are advice, and in
        three places this course does not take it. It also adds one prerequisite the chart does not have.
      </p>
      <Beyond title="Ours, not the book's">
        <p>
          The order is PLAN.md's. Inside a stage the explorer arranges chapters by the book's arrows, and every arrow
          the order runs ahead of is drawn heavy, so you can overrule it. The reasons in the list are ours.
        </p>
      </Beyond>
      <ul>
        <li>
          <b>Added: reduction before attention.</b> A softmax is a reduction over a row, and in the attention kernel of
          chapter 20 the threads of a warp pass values to each other with warp shuffles (pp. 497–501), which chapter 10
          introduces (§10.7). So chapter 10 comes before 20 here.
        </li>
        <li>
          <b>Attention before convolution and CNNs.</b> The chart routes chapter 20 through 7 and 19. The
          cross-references we recorded in chapter 20 point back to chapters 5, 10, 15 and 16 and to Appendix B, the
          deep-learning background you already have. None points to 7 or 19, so this course reads those two afterwards.
        </li>
        <li>
          <b>Merge after sort, and condensed.</b> The chart puts merge (13) before sort (14) because merge sort is
          built on it. The sort you need for sampling is radix sort, which the Preface describes as built on the stable
          filter of chapter 12 (p. xx).
        </li>
        <li>
          <b>Chapter 22 before three chapters it draws on.</b> The chart puts 16, 18 and 21 before 22. This course
          reads 22 straight after attention, for its section on batching as latency against throughput (§22.5), and
          keeps 16, 18 and 21 as condensed pages.
        </li>
      </ul>
      <Ask q={questions17['ch01/1.7/q-sortneeds']} />

      <h2>What the 5th edition changed</h2>
      <p>
        You are reading the 5th edition. The 4th is dated 2022, and most slides, blog posts and forum answers you will
        find were written against it or an earlier one. The Preface gives the reasons for a new edition: three more GPU
        generations (Ada Lovelace, Hopper, Blackwell), CUDA 11 through 13, and very large GPU clusters now serving LLM
        queries.
      </p>
      <Today checked="2026-11-20 against NVIDIA's pages (notes/today/ch01-facts.md, A5)">
        <p>
          A generation after Blackwell is shipping (Vera Rubin, per NVIDIA's results release of November 17, 2026) and
          the current toolkit is CUDA 13.5. NVIDIA's compute capability table did not list the new generation yet.
        </p>
      </Today>
      <p>Here is what the Preface says changed, with the part that concerns you.</p>
      <Source pages="xviii–xxii" figs="Preface" />
      <table className="data wide" data-role="changes">
        <thead>
          <tr>
            <th style={{ width: '6rem' }}>Kind</th>
            <th style={{ width: '13rem' }}>Chapter</th>
            <th>What changed</th>
            <th>Why it matters to you</th>
          </tr>
        </thead>
        <tbody>
          {CHANGES.map((c, i) => (
            <tr key={i}>
              <td>{i === 0 || CHANGES[i - 1].kind !== c.kind ? <b>{c.kind}</b> : null}</td>
              <td>{c.what}</td>
              <td>{c.change}</td>
              <td>{c.you}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p>
        The consequence for anything written against an earlier edition: it has no chapter on attention and no chapter
        on GEMM in depth, and from chapter 12 onward its chapter numbers mean something else.
      </p>
      <p>
        Of the removals, dynamic parallelism is the one you are most likely to meet in the wild: code running on the
        GPU that launches more GPU work. The Preface says the feature saw little adoption, partly because of its
        overhead, and points to the CUDA programming guide and to older editions for it.
      </p>
      <Today checked="2026-11-20 against the CUDA 13.5 guide and AMD's HIP docs (notes/today/ch01-facts.md, A14, C4; ch05-facts.md, B4)">
        <p>
          It left the book, not CUDA: the guide still documents it (the version called CDP2 is the default from CUDA
          12.0), with no deprecation notice. HIP does not support it. For Metal, none documented that we found; we did
          not search for it specifically.
        </p>
      </Today>

      <h3>Which edition is that page about?</h3>
      <p>
        A chapter number with no edition attached is ambiguous from 12 upward. The lookup takes a 4th-edition chapter
        number and shows where that material is now. Try 14, 16 and 21. The publisher's web page will not settle
        the question for you; the note says why.
      </p>
      <Today checked="2026-11-20 against the publisher's pages (notes/today/ch01-facts.md, A15)">
        <p>
          The publisher's pages for the 5th edition showed a pre-publication contents list: 25 chapters in four parts,
          no LLM chapter. The printed book has 24 in three parts. Trust the book.
        </p>
      </Today>
      <EditionLookup />
      <Ask q={questions17['ch01/1.7/q-edition16']} />

      <h2>Where CUDA sits among other interfaces</h2>
      <p>
        The book places CUDA against three others. <b>OpenMP</b>: you annotate loops with directives and a compiler and
        runtime produce the parallel code; it was born on CPUs and extended to GPUs, and it buys convenience at the
        price of control. <b>MPI</b>: for clusters whose nodes share no memory, so every exchange is an explicit
        message; that is the level above one GPU, which this course condenses in chapter 23. <b>OpenCL</b>: close to
        CUDA in concepts, built more on API calls and less on language extensions; an OpenCL program runs unchanged on
        any device that supports it, and usually needs modification to run well on a new one.
      </p>
      <Source section="1.5" pages="12–13" />
      <p>
        Hold on to that last remark. It is about OpenCL, and it describes every port you will do: correctness travels
        with the source, performance does not. The book never mentions HIP or Metal, the two interfaces this course
        adds, but AMD says of HIP what the book says of OpenCL: a CUDA programmer will find the API straightforward,
        and HIP is "not intended to be a drop-in replacement", so expect manual porting and performance tuning. As for
        OpenCL itself, Apple deprecated it in macOS 10.14 and names Metal as the replacement.
      </p>
      <Beyond>
        <p>HIP and Metal are this course's additions. The statements about them are the vendors' own.</p>
        <p className="muted">Checked 2026-11-20 against AMD's and Apple's documentation (notes/today/ch01-facts.md, B1, B2, A11).</p>
      </Beyond>
      <Ask q={questions17['ch01/1.7/q-portable']} />
      <p>
        For your third goal, here is how the three code bases you want to read arrange their GPU backends. Read each
        row for one thing: whether the AMD and Metal columns hold new code or the CUDA code again. The short version
        is three backends, two code bases.
      </p>
      <Today checked="2026-11-20 against each project's repository and docs (notes/today/ch01-facts.md, D1, D3, D4; ch05-facts.md, B9)">
        <p>
          Layouts move. <a href={href('x3')}>Module X3</a> checks them again when it opens the files.
        </p>
      </Today>
      <table className="data wide" data-role="backends">
        <thead>
          <tr>
            <th style={{ width: '7rem' }}>Project</th>
            <th>CUDA</th>
            <th>HIP (AMD)</th>
            <th>Metal</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>
              <b>PyTorch</b>
            </td>
            <td>
              Device name <code>'cuda'</code>.
            </td>
            <td>
              Device name <code>'cuda'</code> as well. The kernels are the CUDA sources, translated at build time.
            </td>
            <td>
              A separate <code>'mps'</code> device with its own hand-written <code>.metal</code> kernels.
            </td>
          </tr>
          <tr>
            <td>
              <b>llama.cpp</b>
            </td>
            <td>
              <code>ggml-cuda</code>
            </td>
            <td>
              <code>ggml-hip</code> holds build files only and compiles the <code>ggml-cuda</code> sources.
            </td>
            <td>
              <code>ggml-metal</code>, separate code, on by default on macOS.
            </td>
          </tr>
          <tr>
            <td>
              <b>MLX</b>
            </td>
            <td>A CUDA backend, on Linux.</td>
            <td>None.</td>
            <td>The primary backend. MLX describes itself as a framework for Apple silicon.</td>
          </tr>
        </tbody>
      </table>
      <p>
        Where an AMD backend exists it is the CUDA code under other names, and the Metal backend is a second code base
        written by hand. So CUDA learned well covers two of the three, and Metal is where a port means a rewrite:{' '}
        <a href={href('x3')}>module X3</a> reads these directories.
      </p>

      <h2>Three goals, three habits</h2>
      <p>The book states three goals for you. This course attaches a habit to each.</p>
      <Source section="1.6" pages="14" />
      <ul>
        <li>
          <b>Performance.</b> Program for high performance from a conceptual understanding of the hardware, without
          having to be a hardware expert. The habit: measure on your own device. Every lab benchmarks, and no lesson
          promises you a number.
        </li>
        <li>
          <b>Correct functionality and reliability.</b> The habit: run your CUDA kernels through the CPU emulator,
          which replays them under fixed thread orders, so a race is a schedule that fails every time you run it and
          not a wrong result once a week.
        </li>
        <li>
          <b>Scalability across hardware generations.</b> The book's key to it is regular, local memory access. The
          habit that goes with it: query the device, never assume it.
        </li>
      </ul>
      <p>
        The book's preview of chapter 2 names a six-step recipe for moving a computation to a GPU (§1.7, p. 15): pick
        the part to parallelise, allocate device memory, copy the data over, write the kernel, launch it, copy the
        results back. Chapter 2 works through it, so it is only named here. Apple's host-side sequence for a Metal
        compute kernel is the same steps under other names.
      </p>
      <Today checked="2026-11-20 against Apple's Metal documentation (notes/today/ch01-facts.md, C1)">
        <p>
          Apple's wording: find a GPU, create a pipeline for the function, create data objects the GPU can access, then
          create a command buffer, write commands into it and commit it to a command queue. Matching that to the book's
          six steps is our reading.
        </p>
      </Today>
    </>
  );
}
