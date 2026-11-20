import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { flagsAfter, numbering, schedule, slotAt, type Schedule, type SlotCell } from '../../sim/wavesched';

// "Waiting for everyone, or only for your neighbours" (section 16.3): Fig. 16.21's 4 x 4
// tiles on 2 SMs, then on 4, then the book's persistent row owners started in reverse.
// Five beats, one narration clip each (content/ch16/narration/neighbours-N.txt). Every
// lane, flag and count drawn comes from src/sim/wavesched.ts.
const T = 4;
export const SCENE = {
  front2: schedule(T, 2, 'front'),
  counter2: schedule(T, 2, 'counter'),
  front4: schedule(T, 4, 'front'),
  counter4: schedule(T, 4, 'counter'),
  rowsReversed: schedule(T, 2, 'rows', 'reversed'),
};
const NUM = numbering(T);

const W = 720;
const H = 340;
const GX = 22; // tile grid
const GY = 62;
const GC = 44;
const LX = 236; // lane labels
const CX = 290; // first step column
const CW = 40;
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

function Defs() {
  return (
    <defs>
      <marker id="nb-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill="var(--ink)" />
      </marker>
    </defs>
  );
}

/** The tile grid: front-order numbers, a flag square per tile, optional marks. */
function Grid({ flags, front, arrowsTo, spin }: {
  flags: Set<number>;
  front?: number[];
  arrowsTo?: [number, number];
  spin?: number[];
}) {
  const cx = (c: number) => GX + c * GC + GC / 2;
  const cy = (r: number) => GY + r * GC + GC / 2;
  return (
    <g data-grid>
      <text x={GX} y={GY - 12} fontSize="12" fontWeight="600">tiles, in front order</text>
      {NUM.flatMap((row, r) =>
        row.map((n, c) => {
          const done = flags.has(n);
          const inFront = front?.includes(n);
          const spinning = spin?.includes(n);
          return (
            <g key={n} data-tile={n} data-done={done ? 1 : 0}>
              <rect x={GX + c * GC + 1} y={GY + r * GC + 1} width={GC - 3} height={GC - 3} rx="2"
                fill={done ? 'var(--global-wash)' : 'var(--cell)'}
                stroke={spinning ? 'var(--fail)' : inFront ? 'var(--ink)' : 'var(--rule)'}
                strokeWidth={spinning || inFront ? 2.4 : 1}
                strokeDasharray={spinning ? '4 3' : undefined} />
              <text x={cx(c) - 1} y={cy(r) + 2} fontSize="14" textAnchor="middle">{n}</text>
              <rect data-flag={n} x={GX + c * GC + GC - 13} y={GY + r * GC + GC - 13} width="8" height="8" rx="1"
                fill={done ? 'var(--global)' : 'none'} stroke={done ? 'var(--global)' : 'var(--rule)'} />
            </g>
          );
        }),
      )}
      {arrowsTo ? (() => {
        const [r, c] = arrowsTo;
        return (
          <g data-arrows>
            {r > 0 ? <line x1={cx(c)} y1={cy(r - 1) + 10} x2={cx(c)} y2={cy(r) - 12} stroke="var(--ink)" strokeWidth="1.8" markerEnd="url(#nb-arrow)" /> : null}
            {c > 0 ? <line x1={cx(c - 1) + 10} y1={cy(r)} x2={cx(c) - 12} y2={cy(r)} stroke="var(--ink)" strokeWidth="1.8" markerEnd="url(#nb-arrow)" /> : null}
          </g>
        );
      })() : null}
      <text x={GX} y={GY + T * GC + 20} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
        <tspan x={GX} dy="0">small square: the tile's flag,</tspan>
        <tspan x={GX} dy="14">set when the tile is done</tspan>
      </text>
    </g>
  );
}

function SlotBox({ cell, t, s, x, y, h, mark }: { cell: SlotCell; t: number; s: number; x: number; y: number; h: number; mark?: boolean }) {
  const w = CW - 4;
  const common = { 'data-slot': `${t},${s}`, 'data-state': cell.state, 'data-tile': cell.tile ?? '' };
  if (cell.state === 'run')
    return (
      <g {...common}>
        <rect x={x} y={y} width={w} height={h} rx="2" fill="var(--ink)" fillOpacity="0.1" stroke={mark ? 'var(--ink)' : 'var(--ink-2)'} strokeWidth={mark ? 2.6 : 1} />
        <text x={x + w / 2} y={y + h / 2 + 4} fontSize="12.5" textAnchor="middle">{cell.tile}</text>
      </g>
    );
  return (
    <g {...common}>
      <rect x={x} y={y} width={w} height={h} rx="2" fill={cell.state === 'spin' ? 'var(--fail-wash)' : 'none'} stroke="var(--fail)" strokeDasharray="4 3" />
      <text x={x + w / 2} y={y + h / 2 + 4} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--fail)' }}>
        {cell.state === 'spin' ? `wait ${cell.tile}` : 'empty'}
      </text>
    </g>
  );
}

/** One schedule as lanes by steps, showing its first `shown` steps (more than its length repeats a stuck step). */
function Lanes({ sch, shown, y, label, rh = 32, total, mark }: {
  sch: Schedule; shown: number; y: number; label: string; rh?: number; total: number; mark?: [number, number];
}) {
  const boxes: ReactNode[] = [];
  for (let t = 0; t < shown; t++)
    for (let s = 0; s < sch.S; s++) {
      const cell = slotAt(sch, t, s);
      if (cell) boxes.push(<SlotBox key={`${t},${s}`} cell={cell} t={t} s={s} x={CX + t * CW + 2} y={y + s * rh + 2} h={rh - 4} mark={mark && mark[0] === t && mark[1] === s} />);
    }
  const done = Math.min(shown, sch.timeline.length);
  const finished = !sch.stuck && shown >= sch.timeline.length;
  return (
    <g data-lanes={sch.S} data-rule={sch.rule} data-shown={done} data-finished={finished ? 1 : 0}>
      <text x={LX} y={y - 26} fontSize="12" fontWeight="600">{label}</text>
      {Array.from({ length: total }, (_, t) => (
        <text key={t} x={CX + t * CW + CW / 2} y={y - 8} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{t + 1}</text>
      ))}
      {Array.from({ length: sch.S }, (_, s) => (
        <g key={s}>
          <text x={LX} y={y + s * rh + rh / 2 + 4} fontSize="11" style={{ fill: 'var(--ink-2)' }}>SM {s}</text>
          <rect x={CX} y={y + s * rh} width={total * CW} height={rh} fill="none" stroke="var(--rule-soft)" />
        </g>
      ))}
      {boxes}
      {sch.rule === 'front'
        ? sch.barriers.filter((b) => b < shown - 1 || (b === shown - 1 && finished)).map((b) => (
            <line key={b} data-barrier={b} x1={CX + (b + 1) * CW} y1={y - 4} x2={CX + (b + 1) * CW} y2={y + sch.S * rh + 4} stroke="var(--ink-2)" strokeWidth="1.2" strokeDasharray="2 3" />
          ))
        : null}
    </g>
  );
}

function Stage({ beat, headline, children, counter }: { beat: number; headline: string; children: ReactNode; counter?: ReactNode }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Tiles of a wavefront scheduled on a few SMs" style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <Defs />
      <g data-beat={beat}>
        <text data-headline x={GX} y={22} fontSize="13.5" fontWeight="600">{headline}</text>
        {children}
        {counter}
      </g>
    </svg>
  );
}

const Counter = ({ x, y, text, value }: { x: number; y: number; text: string; value: number | string }) => (
  <text data-counter={value} x={x} y={y} fontSize="13" fontWeight="600">{text}</text>
);

/** Steps shown at position p of a beat that plays a schedule of n steps. */
const play = (p: number, n: number, a = 0.06, b = 0.82) => Math.round(ramp(p, a, b) * n);

const beats: Beat[] = [
  {
    clip: 'neighbours-1',
    label: 'Sixteen tiles',
    draw: (p) => {
      const front = p >= 0.55 ? [3, 4, 5] : undefined;
      return (
        <Stage beat={1} headline={p < 0.55 ? 'each tile needs the tile above and the tile to its left' : 'one front: tiles 3, 4 and 5 need nothing from each other'}>
          <Grid flags={new Set()} arrowsTo={p >= 0.25 ? [1, 1] : undefined} front={front} />
          <Lanes sch={SCENE.front2} shown={0} y={78} label="2 SMs, as in the book's example" total={10} />
        </Stage>
      );
    },
  },
  {
    clip: 'neighbours-2',
    label: 'A relaunch per front',
    draw: (p) => {
      const sch = SCENE.front2;
      const shown = play(p, sch.makespan!);
      const idle = sch.timeline.slice(0, shown).flat().filter((c) => c.state !== 'run').length;
      return (
        <Stage beat={2} headline="a relaunch after every front: wait for the whole front" counter={
          <>
            <Counter x={CX} y={196} text={shown >= sch.makespan! ? `${sch.makespan} steps` : `step ${shown}`} value={shown} />
            <text data-idle={idle} x={CX + 120} y={196} fontSize="12" style={{ fill: 'var(--fail)' }}>{idle} empty lane-steps</text>
          </>
        }>
          <Grid flags={flagsAfter(sch, shown)} />
          <Lanes sch={sch} shown={shown} y={78} label="relaunch per front, 2 SMs" total={10} />
        </Stage>
      );
    },
  },
  {
    clip: 'neighbours-3',
    label: 'Flags',
    draw: (p) => {
      const sch = SCENE.counter2;
      const shown = play(p, sch.makespan!);
      const six = sch.timeline.findIndex((row) => row.some((c) => c.state === 'run' && c.tile === 6));
      const sixSlot = sch.timeline[six].findIndex((c) => c.state === 'run' && c.tile === 6);
      const markSix = shown > six && p >= 0.4;
      return (
        <Stage beat={3} headline="flags: a tile waits only for its own neighbours" counter={
          <>
            <Counter x={CX} y={196} text={shown >= sch.makespan! ? `${sch.makespan} steps` : `step ${shown}`} value={shown} />
            <text x={CX + 120} y={196} fontSize="12" style={{ fill: 'var(--ink-2)' }}>a relaunch per front took {SCENE.front2.makespan}</text>
            {markSix ? <text data-mark="six" x={CX + six * CW - 30} y={166} fontSize="11.5">tile 6 fills the empty lane of step {six + 1}</text> : null}
          </>
        }>
          <Grid flags={flagsAfter(sch, shown)} />
          <Lanes sch={sch} shown={shown} y={78} label="tiles from a counter, 2 SMs" total={10} mark={markSix ? [six, sixSlot] : undefined} />
        </Stage>
      );
    },
  },
  {
    clip: 'neighbours-4',
    label: 'Four SMs',
    draw: (p) => {
      const a = SCENE.front4;
      const b = SCENE.counter4;
      const shown = play(p, a.makespan!);
      return (
        <Stage beat={4} headline="as many SMs as the widest front: nothing to gain" counter={
          <Counter x={CX + 7 * CW + 16} y={190} text={shown >= a.makespan! ? `both ${a.makespan} steps` : `step ${shown}`} value={shown} />
        }>
          <Grid flags={flagsAfter(b, shown)} />
          <Lanes sch={a} shown={shown} y={72} rh={28} label="relaunch per front, 4 SMs" total={7} />
          <Lanes sch={b} shown={shown} y={222} rh={28} label="tiles from a counter, 4 SMs" total={7} />
        </Stage>
      );
    },
  },
  {
    clip: 'neighbours-5',
    label: 'Fixed rows',
    draw: (p) => {
      if (p < 0.5) {
        const sch = SCENE.rowsReversed;
        const shown = 1 + Math.floor(ramp(p, 0.08, 0.42) * 3);
        const waiting = sch.timeline[0].map((c) => c.tile as number);
        return (
          <Stage beat={5} headline="row owners started in reverse: every lane waits, nothing moves" counter={
            <Counter x={CX} y={196} text="0 tiles done" value="stuck" />
          }>
            <Grid flags={new Set()} spin={waiting} />
            <Lanes sch={sch} shown={shown} y={78} label="one block per tile row, rows 3 and 2 first" total={10} />
            <g data-notes>
              {['row 3 waits for row 2', 'row 2 waits for row 1', 'row 1 has no lane, and never will'].map((t, i) => (
                <text key={t} x={CX} y={228 + i * 18} fontSize="12" style={{ fill: i === 2 ? 'var(--fail)' : 'var(--ink-2)' }}>{t}</text>
              ))}
            </g>
          </Stage>
        );
      }
      const sch = SCENE.counter2;
      const shown = play(p, sch.makespan!, 0.55, 0.92);
      return (
        <Stage beat={5} headline="tiles from a counter in front order: waits only on started work" counter={
          <Counter x={CX} y={196} text={shown >= sch.makespan! ? `${sch.makespan} steps` : `step ${shown}`} value={shown} />
        }>
          <Grid flags={flagsAfter(sch, shown)} />
          <Lanes sch={sch} shown={shown} y={78} label="tiles from a counter, 2 SMs" total={10} />
        </Stage>
      );
    },
  },
];

export function NeighboursScene() {
  return <Scene chapter="ch16" title="Waiting for everyone, or only for your neighbours" beats={beats} />;
}
