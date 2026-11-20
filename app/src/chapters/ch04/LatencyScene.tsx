import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import {
  PRESETS, busyFraction, runs, simulate, simulateSwitching, unitShares, utilisation, warpsNeeded, type Cell, type Timeline,
  type UnitCell,
} from '../../sim/latency';
import { BOOK, occupancy } from '../../sim/occupancy';

// "Hiding a wait": section 4.6 in time. Six beats, one narration clip each
// (content/ch04/narration/latency-N.txt). Every cell, count and percentage drawn
// comes from sim/latency.ts, and the register file of beat 5 from sim/occupancy.ts.
const W = 720;
const H = 348;
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const pct = (v: number) => Math.round(v * 1000) / 10;

// The scene's own small case: 2 cycles of work, then a 6-cycle wait. Illustrative
// numbers, chosen so that the warps needed (4) fit on the screen.
export const SCENE = { work: 2, latency: 6, slots: 1 } as const;
export const CYCLES = 24;
export const NEEDED = warpsNeeded(SCENE.work, SCENE.latency);
const line = (warps: number) => simulate({ warps, ...SCENE }, CYCLES);
const ONE = line(1);
const THREE = line(3);
const ENOUGH = line(NEEDED);
/** The price of a switch on the CPU side of beat 4, in cycles. Illustrative: the book gives no figure. */
export const SWITCH_COST = 2;
const FREE = simulateSwitching({ warps: NEEDED, ...SCENE }, CYCLES, 0);
const COSTLY = simulateSwitching({ warps: NEEDED, ...SCENE }, CYCLES, SWITCH_COST);

// Beat 5: the book's example register file, blocks of two warps, at two register counts.
const REG_BLOCK = 2 * (BOOK.width as number);
export const REGS_A = 32;
export const REGS_B = 64;
const FITS_A = occupancy(BOOK, { threads: REG_BLOCK, regs: REGS_A, shared: 0 });
const FITS_B = occupancy(BOOK, { threads: REG_BLOCK, regs: REGS_B, shared: 0 });
const UNITS = FITS_A.warpSlots as number; // the register file drawn as this many equal cells
const REGS_PER_UNIT = (BOOK.regs as number) / UNITS;

// Beat 6: the explorer's two presets, at these numbers of resident warps.
export const LADDER = [1, 2, 4, 8, 16];
const SIDE_CYCLES = 64;
const side = (preset: { work: number; latency: number }) =>
  LADDER.map((warps) => ({
    warps,
    tl: simulate({ warps, ...preset, slots: 1 }, SIDE_CYCLES),
    use: utilisation({ warps, ...preset, slots: 1 }),
  }));
const HEAVY = side(PRESETS.computeHeavy);
const LOADY = side(PRESETS.loadHeavy);

const FILL: Record<Cell, string> = { exec: 'var(--ink)', wait: 'var(--global-wash)', ready: 'var(--rule)' };
const STROKE: Record<Cell, string> = { exec: 'var(--ink)', wait: 'var(--global)', ready: 'var(--ink-3)' };

function Frame({ children, headline, sub }: { children: ReactNode; headline?: ReactNode; sub?: ReactNode }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Animation of warps taking turns on one execution unit" style={{ width: '100%', maxWidth: W, display: 'block' }}>
      {children}
      {headline ? <text x={24} y={H - 30} fontSize="13.5" fontWeight="600">{headline}</text> : null}
      {sub ? <text x={24} y={H - 12} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>{sub}</text> : null}
    </svg>
  );
}

function Legend({ x, y, ready = true }: { x: number; y: number; ready?: boolean }) {
  const items: [Cell, string][] = [['exec', 'executing'], ['wait', 'waiting on memory']];
  if (ready) items.push(['ready', 'ready, not chosen']);
  return (
    <g>
      {items.map(([c, label], i) => (
        <g key={c}>
          <rect x={x + i * 150} y={y - 9} width={14} height={10} rx="1.5" fill={FILL[c]} stroke={STROKE[c]} strokeWidth="0.8" />
          <text x={x + i * 150 + 20} y={y} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>{label}</text>
        </g>
      ))}
    </g>
  );
}

/** One row per warp and the execution unit below them, drawn up to cycle `upto`. */
function Lanes({ x, y, tl, upto, cw, rh, labels = true, unit = true }: {
  x: number; y: number; tl: Timeline; upto: number; cw: number; rh: number; labels?: boolean; unit?: boolean;
}) {
  const shown = Math.max(0, Math.min(tl.cycles, Math.floor(upto)));
  const R = tl.rows.length;
  const uy = y + R * rh + 12;
  return (
    <g>
      {tl.rows.map((row, w) => (
        <g key={w}>
          {labels ? <text x={x - 8} y={y + w * rh + rh / 2 + 3} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>warp {w}</text> : null}
          <rect x={x} y={y + w * rh} width={tl.cycles * cw} height={rh - 3} fill="none" stroke="var(--rule-soft)" />
          {runs(row).filter((r) => r.start < shown).map((r) => {
            const len = Math.min(r.length, shown - r.start);
            return (
              <rect key={r.start} x={x + r.start * cw + 0.5} y={y + w * rh} width={Math.max(0.5, len * cw - 1)} height={rh - 3}
                rx={Math.min(2, rh / 4)} fill={FILL[r.state]} stroke={STROKE[r.state]} strokeWidth={rh > 12 ? 0.9 : 0.4} />
            );
          })}
        </g>
      ))}
      {unit ? (
        <g>
          {labels ? <text x={x - 8} y={uy + rh / 2 + 2} fontSize="10.5" textAnchor="end" fontWeight="600">unit</text> : null}
          {tl.chosen.slice(0, shown).map((pick, t) => (
            <rect key={t} x={x + t * cw + 0.5} y={uy} width={Math.max(0.5, cw - 1)} height={rh - 3} rx={Math.min(2, rh / 4)}
              fill={pick.length ? 'var(--ink)' : 'none'} stroke={pick.length ? 'var(--ink)' : 'var(--ink-3)'}
              strokeDasharray={pick.length ? undefined : '2 2'} strokeWidth="0.9" />
          ))}
          {rh > 14 && cw > 14
            ? tl.chosen.slice(0, shown).map((pick, t) =>
                pick.length ? (
                  <text key={'n' + t} x={x + t * cw + cw / 2} y={uy + (rh - 3) / 2 + 3.5} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--page)' }}>{pick[0]}</text>
                ) : null,
              )
            : null}
        </g>
      ) : null}
      <line x1={x} x2={x + tl.cycles * cw} y1={uy + (unit ? rh : -12) + 4} y2={uy + (unit ? rh : -12) + 4} stroke="var(--ink-3)" />
      <text x={x + tl.cycles * cw} y={uy + (unit ? rh : -12) + 16} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>cycles</text>
    </g>
  );
}

const busySoFar = (tl: Timeline, upto: number) => {
  const shown = Math.max(1, Math.min(tl.cycles, Math.floor(upto)));
  return { busy: Math.round(busyFraction(tl, 0, shown) * shown), shown };
};
/** The first wait of a warp: where it starts and where its data is back. */
const firstWait = (tl: Timeline, w: number) => runs(tl.rows[w]).find((r) => r.state === 'wait')!;

const LX = 96; // left edge of the wide timelines
const CW = 24;

/** A warp's resident state: its program counter and its registers, drawn in the register colour. */
function WarpState({ x, y, n, on }: { x: number; y: number; n: number; on: boolean }) {
  return (
    <g>
      <rect x={x} y={y} width={132} height={30} rx="3" fill={on ? 'var(--reg)' : 'var(--reg-wash)'} stroke="var(--reg)" />
      <text x={x + 8} y={y + 19} fontSize="10.5" style={{ fill: on ? 'var(--page)' : 'var(--ink)' }}>warp {n}: PC, registers</text>
    </g>
  );
}

function UnitStrip({ x, y, unit, upto, cw }: { x: number; y: number; unit: UnitCell[]; upto: number; cw: number }) {
  const shown = Math.max(0, Math.min(unit.length, Math.floor(upto)));
  return (
    <g>
      {unit.slice(0, shown).map((u, t) => (
        <g key={t}>
          <rect x={x + t * cw + 0.5} y={y} width={cw - 1} height={20} rx="2"
            fill={u.kind === 'exec' ? 'var(--ink)' : u.kind === 'switch' ? 'var(--rule)' : 'none'}
            stroke={u.kind === 'exec' ? 'var(--ink)' : 'var(--ink-3)'} strokeDasharray={u.kind === 'idle' ? '2 2' : undefined} strokeWidth="0.9" />
          {u.kind === 'exec' ? (
            <text x={x + t * cw + cw / 2} y={y + 14} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--page)' }}>{u.warp}</text>
          ) : null}
        </g>
      ))}
    </g>
  );
}

/** Two kernels of beat 6: one small timeline and its utilisation. */
function Side({ x, title, sub, rung }: { x: number; title: string; sub: string; rung: { warps: number; tl: Timeline; use: number } }) {
  const cw = 4.8;
  const rh = 9;
  return (
    <g>
      <text x={x} y={26} fontSize="12.5" fontWeight="600">{title}</text>
      <text x={x} y={42} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>{sub}</text>
      <Lanes x={x} y={56} tl={rung.tl} upto={SIDE_CYCLES} cw={cw} rh={rh} labels={false} />
      <text x={x} y={262} fontSize="12" fontWeight="600">
        {rung.warps} {rung.warps === 1 ? 'warp' : 'warps'}: unit busy {pct(rung.use)}%
      </text>
      <rect x={x} y={270} width={SIDE_CYCLES * cw} height={9} fill="none" stroke="var(--ink-3)" />
      <rect x={x} y={270} width={SIDE_CYCLES * cw * rung.use} height={9} fill="var(--ink)" />
    </g>
  );
}

const beats: Beat[] = [
  {
    clip: 'latency-1',
    label: 'One warp',
    draw: (p) => {
      const upto = ramp(p, 0.08, 0.9) * CYCLES;
      const { busy, shown } = busySoFar(ONE, upto);
      const wait = firstWait(ONE, 0);
      return (
        <Frame
          headline={`execution unit busy for ${busy} of ${shown} cycles: ${pct(busy / shown)}%`}
          sub={`${SCENE.work} cycles of work, then a ${SCENE.latency}-cycle wait on global memory`}
        >
          <Legend x={LX} y={34} ready={false} />
          <Lanes x={LX} y={84} tl={ONE} upto={upto} cw={CW} rh={30} />
          {upto > wait.start ? <text x={LX + wait.start * CW} y={76} fontSize="10.5" style={{ fill: 'var(--global)' }}>load sent</text> : null}
          {upto > wait.start + wait.length ? (
            <text x={LX + (wait.start + wait.length) * CW} y={76} fontSize="10.5" style={{ fill: 'var(--global)' }}>data back</text>
          ) : null}
        </Frame>
      );
    },
  },
  {
    clip: 'latency-2',
    label: 'Three warps',
    draw: (p) => {
      const upto = ramp(p, 0.08, 0.9) * CYCLES;
      const { busy, shown } = busySoFar(THREE, upto);
      return (
        <Frame headline={`execution unit busy for ${busy} of ${shown} cycles: ${pct(busy / shown)}%`} sub="when one warp waits, the scheduler issues from another that is ready">
          <Legend x={LX} y={34} />
          <Lanes x={LX} y={62} tl={THREE} upto={upto} cw={CW} rh={30} />
        </Frame>
      );
    },
  },
  {
    clip: 'latency-3',
    label: 'Enough warps',
    draw: (p) => {
      const upto = ramp(p, 0.06, 0.8) * CYCLES;
      const { busy, shown } = busySoFar(ENOUGH, upto);
      const last = firstWait(ENOUGH, NEEDED - 1);
      const first = firstWait(ENOUGH, 0);
      const meet = first.start + first.length; // the cycle warp 0's data is back
      return (
        <Frame
          headline={`${NEEDED} warps: busy for ${busy} of ${shown} cycles, ${pct(busy / shown)}%`}
          sub={p > 0.55 ? 'latency tolerance: no warp got faster; the wait is hidden behind other warps' : `as warp ${NEEDED - 1} sends its request, warp 0's data arrives`}
        >
          <Legend x={LX} y={34} />
          <Lanes x={LX} y={56} tl={ENOUGH} upto={upto} cw={CW} rh={30} />
          {upto >= meet && last.start === meet ? (
            <g>
              <line x1={LX + meet * CW} x2={LX + meet * CW} y1={50} y2={56 + NEEDED * 30 + 42} stroke="var(--ink)" strokeDasharray="4 3" />
              <text x={LX + meet * CW + 5} y={50} fontSize="10.5">cycle {meet}</text>
            </g>
          ) : null}
        </Frame>
      );
    },
  },
  {
    clip: 'latency-4',
    label: 'A free switch',
    draw: (p) => {
      const upto = ramp(p, 0.1, 0.92) * CYCLES;
      const t = Math.max(0, Math.min(CYCLES - 1, Math.floor(upto)));
      const gpu = FREE[t];
      const cpu = COSTLY[t];
      const shares = unitShares(COSTLY, 0, CYCLES);
      const cw = 12.6;
      // A switch of SWITCH_COST cycles: the first half saves the old state, the second loads the new one.
      let into = 0; // how many cycles of the current switch have been spent, this one included
      while (t - into >= 0 && COSTLY[t - into].kind === 'switch') into += 1;
      const loading = into > SWITCH_COST / 2;
      return (
        <Frame
          headline={`the SM changed warps for free; the other design spent ${pct(shares.switch)}% of its cycles saving and loading`}
          sub={`the same ${NEEDED} warps and the same waits on both sides; a ${SWITCH_COST}-cycle switch is an illustrative price`}
        >
          <text x={30} y={26} fontSize="12.5" fontWeight="600">an SM: every warp's state stays on the chip</text>
          {Array.from({ length: NEEDED }, (_, w) => <WarpState key={w} x={30} y={38 + w * 36} n={w} on={gpu.kind === 'exec' && gpu.warp === w} />)}
          <text x={176} y={38 + (gpu.warp ?? 0) * 36 + 19} fontSize="11">◀ issuing from here</text>
          <text x={30} y={204} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>execution unit</text>
          <UnitStrip x={30} y={210} unit={FREE} upto={upto} cw={cw} />

          <text x={384} y={26} fontSize="12.5" fontWeight="600">a CPU core: one thread's state at a time</text>
          <rect x={384} y={38} width={132} height={30} rx="3" fill={cpu.kind === 'exec' ? 'var(--reg)' : 'var(--reg-wash)'} stroke="var(--reg)" />
          <text x={392} y={57} fontSize="10.5" style={{ fill: cpu.kind === 'exec' ? 'var(--page)' : 'var(--ink)' }}>
            {cpu.kind === 'switch' ? 'PC, registers: in transit' : `thread ${cpu.warp ?? 0}: PC, registers`}
          </text>
          <rect x={384} y={96} width={300} height={84} rx="3" fill="var(--global-wash)" stroke="var(--global)" />
          <text x={392} y={112} fontSize="10.5" style={{ fill: 'var(--global)' }}>memory: the state of the threads that are not running</text>
          {Array.from({ length: NEEDED }, (_, w) => (
            <g key={w} opacity={cpu.warp === w && cpu.kind === 'exec' ? 0.25 : 1}>
              <rect x={392 + w * 72} y={124} width={64} height={44} rx="2" fill="var(--cell)" stroke="var(--ink-3)" />
              <text x={392 + w * 72 + 32} y={143} fontSize="10" textAnchor="middle">thread {w}</text>
              <text x={392 + w * 72 + 32} y={158} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>saved state</text>
            </g>
          ))}
          {cpu.kind === 'switch' ? (
            <text x={530} y={62} fontSize="11" fontWeight="600">{loading ? 'load ↑ the next thread' : 'save ↓ the old thread'}</text>
          ) : null}
          <text x={384} y={204} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>execution unit (grey: saving and loading)</text>
          <UnitStrip x={384} y={210} unit={COSTLY} upto={upto} cw={cw} />
        </Frame>
      );
    },
  },
  {
    clip: 'latency-5',
    label: 'The price',
    draw: (p) => {
      const second = p >= 0.55;
      const fits = second ? FITS_B : FITS_A;
      const regs = second ? REGS_B : REGS_A;
      const grow = second ? ramp(p, 0.6, 0.85) : ramp(p, 0.08, 0.45);
      const warps = Math.round(fits.residentWarps * grow);
      const perWarp = regs * fits.width; // registers one warp holds
      const span = perWarp / REGS_PER_UNIT; // cells of the drawing one warp takes
      const cols = 16;
      const cell = 38;
      const X = 56;
      const Y = 70;
      const cells: ReactNode[] = [];
      for (let u = 0; u < UNITS; u++)
        cells.push(<rect key={'u' + u} x={X + (u % cols) * cell} y={Y + Math.floor(u / cols) * cell} width={cell - 3} height={cell - 3} rx="2" fill="var(--cell)" stroke="var(--rule)" />);
      for (let w = 0; w < warps; w++) {
        const u = w * span;
        cells.push(
          <g key={'w' + w}>
            <rect x={X + (u % cols) * cell} y={Y + Math.floor(u / cols) * cell} width={span * cell - 3} height={cell - 3} rx="2" fill="var(--reg-wash)" stroke="var(--reg)" />
            <text x={X + (u % cols) * cell + (span * cell - 3) / 2} y={Y + Math.floor(u / cols) * cell + cell / 2 + 2} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--reg)' }}>{w}</text>
          </g>,
        );
      }
      return (
        <Frame
          headline={`${regs} registers per thread: ${warps} of ${fits.warpSlots} warp slots filled, ${(warps * perWarp).toLocaleString('en-US')} of ${(BOOK.regs as number).toLocaleString('en-US')} registers`}
          sub={second ? `twice the registers per thread: ${FITS_B.residentWarps} warps fit where ${FITS_A.residentWarps} did` : "the book's example SM: the register file is shared out among the resident warps"}
        >
          <text x={X} y={40} fontSize="12.5" fontWeight="600">the register file of one SM (the book's example)</text>
          <text x={X} y={56} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>one cell = {REGS_PER_UNIT.toLocaleString('en-US')} registers; a numbered box is one resident warp's share</text>
          {cells}
        </Frame>
      );
    },
  },
  {
    clip: 'latency-6',
    label: 'Two kernels',
    draw: (p) => {
      const i = Math.min(LADDER.length - 1, Math.floor(ramp(p, 0.1, 0.8) * LADDER.length));
      return (
        <Frame
          headline={`busy with ${warpsNeeded(PRESETS.computeHeavy.work, PRESETS.computeHeavy.latency)} warps on the left; ${warpsNeeded(PRESETS.loadHeavy.work, PRESETS.loadHeavy.latency)} needed on the right`}
          sub="token-by-token decoding looks like the right-hand kernel: little arithmetic per load"
        >
          <Side x={30} title="long compute between loads" sub={`${PRESETS.computeHeavy.work} cycles of work, then a ${PRESETS.computeHeavy.latency}-cycle wait`} rung={HEAVY[i]} />
          <Side x={384} title="a load every other instruction" sub={`${PRESETS.loadHeavy.work} cycles of work, then a ${PRESETS.loadHeavy.latency}-cycle wait`} rung={LOADY[i]} />
        </Frame>
      );
    },
  },
];

export function LatencyScene() {
  return <Scene chapter="ch04" title="Hiding a wait" beats={beats} />;
}
