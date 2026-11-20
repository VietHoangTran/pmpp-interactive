import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import {
  CORRECT,
  SWAP_ORDER,
  TOO_LARGE,
  coopLoop,
  gridGraph,
  gridSummary,
  hostStrip,
  residency,
  simulate,
  strideAssignment,
  treeSummary,
} from '../../sim/gridbfs';

// "One launch for every level" (section 18.6, book §18.7 pp. 445-448). Every drawing is
// computed from src/sim/gridbfs.ts, the model behind the section's "Three grid barriers"
// stepper and "Does it fit?" tab: frontier sizes from its formulas, the host loop as a strip
// in units of vertex work, the stride assignment, the residency model and the swap's event
// order from a simulated run. Time is drawn in levels and launches, never in seconds. Six
// beats, one narration clip each (content/ch18/narration/one-launch-N.txt).

/** The round trip, drawn as the work of this many vertices: an assumption for the picture, the same in every strip. */
export const SCENE_GAP = 10;
const ROAD_W = 16;
const ROAD = gridSummary(ROAD_W);
const SOCIAL = treeSummary(8, 4);
const ROAD_STRIP = hostStrip(ROAD.sizes, SCENE_GAP);
const SOCIAL_STRIP = hostStrip(SOCIAL.sizes, SCENE_GAP);
const COOP = coopLoop(ROAD.sizes, 3);
const BLOCKS = 3;
const THREADS = 2;
const G = BLOCKS * THREADS;
/** A level of the road grid with more elements than threads, so that some threads take two or three. */
export const STRIDE_LEVEL = 13;
const STRIDE = strideAssignment(ROAD.sizes[STRIDE_LEVEL], G);
const FIT = residency(2, 2, 6);
/** One correct level of a small search, for the swap's order of events. */
const RUN = simulate(CORRECT, 'forward', gridGraph(4), 2, 2);
export const SWAP_LEVEL = 3;
const LEVEL_EVENTS = RUN.history.filter((e) => e.level === SWAP_LEVEL);

export const SCENE_DATA = {
  road: ROAD, social: SOCIAL, roadStrip: ROAD_STRIP, socialStrip: SOCIAL_STRIP, coop: COOP,
  stride: STRIDE, fit: FIT, levelEvents: LEVEL_EVENTS,
};

const W = 720;
const H = 340;
const X0 = 24;
const X1 = 696;
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

function Svg({ label, children }: { label: string; children: ReactNode }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={label} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      {children}
    </svg>
  );
}
const T = ({ x, y, children, size = 12, anchor = 'start', weight, muted }: {
  x: number; y: number; children: ReactNode; size?: number; anchor?: 'start' | 'middle' | 'end'; weight?: number; muted?: boolean;
}) => (
  <text x={x} y={y} fontSize={size} textAnchor={anchor} fontWeight={weight} style={{ fill: muted ? 'var(--ink-2)' : 'var(--ink)' }}>{children}</text>
);

/** A host-loop strip: kernel bars and dashed round trips, revealed up to `upto` (0..1 of its length). */
function Strip({ y, strip, upto, label, h = 22 }: { y: number; strip: ReturnType<typeof hostStrip>; upto: number; label: string; h?: number }) {
  const scale = (X1 - X0) / strip.total;
  const end = upto * strip.total;
  return (
    <g>
      <T x={X0} y={y - 7} size={12} weight={600}>{label}</T>
      <rect x={X0} y={y} width={X1 - X0} height={h} fill="none" stroke="var(--rule-soft)" />
      {strip.parts.filter((p) => p.from < end).map((p, i) => {
        const w = Math.max(p.kind === 'kernel' ? 0.8 : 0.5, Math.min(p.len, end - p.from) * scale);
        return p.kind === 'kernel'
          ? <rect key={i} data-part="kernel" x={X0 + p.from * scale} y={y} width={w} height={h} fill="var(--ink-2)" />
          : <rect key={i} data-part="gap" x={X0 + p.from * scale} y={y + 3} width={w} height={h - 6} fill="none" stroke="var(--ink-3)" strokeDasharray="2 2" />;
      })}
    </g>
  );
}

/** The 16 x 16 road grid, coloured up to `level`, with the frontier at `level` marked. */
function RoadGrid({ x, y, level }: { x: number; y: number; level: number }) {
  const c = 8.5;
  const cells: ReactNode[] = [];
  for (let r = 0; r < ROAD_W; r++)
    for (let q = 0; q < ROAD_W; q++) {
      const l = r + q;
      cells.push(
        <rect key={r * ROAD_W + q} x={x + q * c} y={y + r * c} width={c - 1} height={c - 1}
          fill={l === level ? 'var(--ink)' : l < level ? 'var(--ink-3)' : 'var(--cell)'} stroke="var(--rule-soft)" strokeWidth="0.5" />,
      );
    }
  return <g>{cells}</g>;
}

function hostBeat(p: number): ReactNode {
  const upto = ramp(p, 0.05, 0.95);
  const kernelsShown = ROAD_STRIP.parts.filter((q) => q.kind === 'kernel' && q.from <= upto * ROAD_STRIP.total);
  const level = Math.max(0, kernelsShown.length - 1);
  const size = ROAD.sizes[level];
  return (
    <Svg label="The per-level host loop on a road-like grid">
      <RoadGrid x={X0} y={28} level={level} />
      <T x={190} y={44} size={13} weight={600}>Host loop, one launch per level</T>
      <T x={190} y={68}>{`launch ${kernelsShown.length} of ${ROAD.launches}: a frontier of ${size} ${size === 1 ? 'vertex' : 'vertices'}`}</T>
      <T x={190} y={88} muted>{`launches ${kernelsShown.length}, read-backs of the size ${kernelsShown.length}`}</T>
      <T x={190} y={112} muted>{`a ${ROAD_W} × ${ROAD_W} grid from a corner: ${ROAD.levels} levels, never more than ${ROAD.maxFrontier} vertices in one`}</T>
      <Strip y={196} strip={ROAD_STRIP} upto={upto} label="each bar a launch, as long as its frontier; each dashed box a round trip" />
      <T x={X0} y={246} muted>{`round trips: ${Math.round(ROAD_STRIP.gapShare * 100)}% of the strip`}</T>
      <T x={X0} y={298} size={11} muted>{`A round trip is drawn as the work of ${SCENE_GAP} vertices. That is an assumption for the picture;`}</T>
      <T x={X0} y={314} size={11} muted>the lab's floor row measures what one costs on your device.</T>
    </Svg>
  );
}

function socialBeat(p: number): ReactNode {
  const upto = ramp(p, 0.08, 0.85);
  return (
    <Svg label="The same host loop for a graph whose levels grow fast">
      <Strip y={52} strip={ROAD_STRIP} upto={1} label={`a road-like grid: ${ROAD.launches} launches`} h={18} />
      <T x={X0} y={92} muted>{`round trips: ${Math.round(ROAD_STRIP.gapShare * 100)}% of the strip`}</T>
      <Strip y={152} strip={SOCIAL_STRIP} upto={upto} label={`a graph that fans out 8 ways per hop: ${SOCIAL.launches} launches, frontiers up to ${SOCIAL.maxFrontier.toLocaleString('en-US')}`} />
      <T x={X0} y={200} muted>{upto > 0.98 ? `round trips: ${(SOCIAL_STRIP.gapShare * 100).toFixed(1)}% of the strip` : ''}</T>
      <T x={X0} y={252} size={13} weight={600}>{p > 0.6 ? 'Big frontiers: launch overhead is negligible. Small ones: it may dominate (p. 445).' : ''}</T>
      <T x={X0} y={298} size={11} muted>{`Both strips are drawn to the same rule: a bar per launch as long as its frontier, a round trip as ${SCENE_GAP} vertices of work.`}</T>
    </Svg>
  );
}

function strideBeat(p: number): ReactNode {
  const n = ROAD.sizes[STRIDE_LEVEL];
  const shown = Math.ceil(ramp(p, 0.05, 0.5) * n);
  const cw = 44;
  return (
    <Svg label="One cooperative launch, the frontier dealt out by stride">
      <T x={X0} y={26} size={13} weight={600}>{`One launch: ${BLOCKS} resident blocks of ${THREADS} threads, ${G} threads for every level`}</T>
      <T x={X0} y={46} muted>{`level ${STRIDE_LEVEL} of the grid has ${n} elements: thread r takes r, r + ${G}, r + ${2 * G}, ...`}</T>
      {STRIDE.map((els, r) => (
        <g key={r}>
          <T x={X0} y={78 + r * 22} size={11} muted>{`thread ${r} (block ${Math.floor(r / THREADS)})`}</T>
          {els.map((i, k) => (
            <g key={i} opacity={i < shown ? 1 : 0.12}>
              <rect data-element={i} data-owner={r} x={150 + k * (cw + 4)} y={64 + r * 22} width={cw} height={18} rx={2} fill="var(--cell)" stroke="var(--ink-2)" />
              <T x={150 + k * (cw + 4) + cw / 2} y={77 + r * 22} size={11} anchor="middle">{i}</T>
            </g>
          ))}
        </g>
      ))}
      <Strip y={232} strip={ROAD_STRIP} upto={1} label={`host loop: ${ROAD.launches} launches, ${ROAD.launches} read-backs`} h={16} />
      <g opacity={ramp(p, 0.5, 0.7)}>
        <T x={X0} y={284} size={12} weight={600}>{`one cooperative launch: no read-backs, ${COOP.gridBarriers} grid barriers (3 per level)`}</T>
        <rect x={X0} y={292} width={X1 - X0} height={16} fill="none" stroke="var(--rule-soft)" />
        {(() => {
          const total = ROAD.sizes.reduce((a, b) => a + b, 0);
          const scale = (X1 - X0) / total;
          let x = X0;
          return ROAD.sizes.map((s, k) => {
            const r = <g key={k}><rect x={x} y={292} width={Math.max(0.8, s * scale - 1)} height={16} fill="var(--ink-2)" /><line x1={x + s * scale - 0.5} x2={x + s * scale - 0.5} y1={288} y2={312} stroke="var(--ink)" strokeWidth="1" /></g>;
            x += s * scale;
            return r;
          });
        })()}
      </g>
      <T x={X0} y={330} size={11} muted>{p > 0.75 ? 'The CPU is free, and data can stay in shared memory or registers from one level to the next (p. 448).' : ''}</T>
    </Svg>
  );
}

function fitBeat(p: number): ReactNode {
  const arrive = ramp(p, 0.08, 0.35);
  const frozen = p > 0.4;
  const refused = ramp(p, 0.62, 0.75);
  const slotW = 70;
  return (
    <Svg label="A grid with more blocks than resident slots">
      <T x={X0} y={26} size={13} weight={600}>{`${FIT.grid} blocks, ${FIT.slots} resident slots (2 SMs × 2 blocks each)`}</T>
      {[0, 1].map((sm) => (
        <g key={sm}>
          <rect x={X0 + sm * 180} y={44} width={168} height={92} rx={4} fill="none" stroke="var(--ink-2)" />
          <T x={X0 + 8 + sm * 180} y={60} size={11} muted>{`SM ${sm}`}</T>
          {[0, 1].map((s) => {
            const b = sm * 2 + s;
            return (
              <g key={s} opacity={b < FIT.resident ? 1 : 0}>
                <rect data-resident={b} x={X0 + 10 + sm * 180 + s * (slotW + 8)} y={68 + (1 - arrive) * 0} width={slotW} height={28} rx={3} fill="var(--cell)" stroke="var(--ink)" />
                <T x={X0 + 10 + sm * 180 + s * (slotW + 8) + slotW / 2} y={86} size={11} anchor="middle">{`block ${b}`}</T>
              </g>
            );
          })}
          <line x1={X0 + 6 + sm * 180} x2={X0 + 162 + sm * 180} y1={112} y2={112} stroke="var(--ink)" strokeWidth="2" opacity={arrive} />
        </g>
      ))}
      <T x={X0} y={154} muted>{arrive > 0.99 ? 'grid barrier: blocks 0 to 3 have arrived and wait for 4 and 5' : 'blocks 0 to 3 run to the grid barrier'}</T>
      {Array.from({ length: FIT.pending }, (_, k) => FIT.resident + k).map((b, k) => (
        <g key={b}>
          <rect data-pending={b} x={420 + k * 84} y={68} width={slotW} height={28} rx={3} fill="none" stroke="var(--ink-3)" strokeDasharray="3 3" />
          <T x={420 + k * 84 + slotW / 2} y={86} size={11} anchor="middle" muted>{`block ${b}`}</T>
        </g>
      ))}
      <T x={420} y={118} size={11} muted>waiting for a slot, which no block frees</T>
      {frozen ? <T x={420} y={140} size={12} weight={600}>{refused < 0.5 ? 'never completes' : ''}</T> : null}
      <g opacity={refused}>
        <rect x={X0} y={176} width={X1 - X0} height={112} rx={4} fill="var(--fail-wash)" stroke="var(--fail)" />
        <T x={X0 + 12} y={198} size={12} weight={600}>Under a cooperative launch the same grid is refused at launch:</T>
        <T x={X0 + 12} y={220} size={12}>{`CUDA  cudaLaunchCooperativeKernel returns ${TOO_LARGE.cuda}`}</T>
        <T x={X0 + 12} y={240} size={12}>{`HIP   hipLaunchCooperativeKernel returns ${TOO_LARGE.hip}`}</T>
        <T x={X0 + 12} y={260} size={12}>Metal has no cooperative launch and no grid barrier</T>
        <T x={X0 + 12} y={280} size={11} muted>{`Size it from the occupancy query: SMs × blocks per SM = ${FIT.slots} blocks.`}</T>
      </g>
      <T x={X0} y={318} size={11} muted>Ordinary launch: the book's argument (p. 445). What grid.sync() does there is not documented.</T>
    </Svg>
  );
}

function swapBeat(p: number): ReactNode {
  const rows = SWAP_ORDER.length;
  const at = Math.min(rows - 1, Math.floor(ramp(p, 0.02, 0.7) * rows));
  const ex3 = ramp(p, 0.74, 0.86);
  const appends = LEVEL_EVENTS.filter((e) => e.kind === 'append').length;
  const reads = LEVEL_EVENTS.filter((e) => e.kind === 'read').length;
  return (
    <Svg label="The three grid barriers of the frontier swap">
      <T x={X0} y={24} size={13} weight={600}>The end of each level (Fig. 18.17, lines 65 to 74)</T>
      {SWAP_ORDER.map((r, k) => {
        const y = 40 + k * 32;
        const sync = r.guard !== null;
        return (
          <g key={k} opacity={k <= at ? 1 : 0.25} data-swap-row={k}>
            <rect x={X0} y={y} width={300} height={26} rx={3} fill={sync ? 'var(--ink)' : 'var(--cell)'} stroke="var(--ink-2)" />
            <text x={X0 + 10} y={y + 17} fontSize="12" style={{ fill: sync ? 'var(--page)' : 'var(--ink)' }}>{r.what}</text>
            {sync ? <T x={X0 + 312} y={y + 17} size={12} muted={k !== at}>{r.guard}</T> : null}
          </g>
        );
      })}
      <T x={X0} y={284} size={11} muted>In a run of the model (4 × 4 grid, 4 threads),</T>
      <T x={X0} y={300} size={11} muted>{`level ${SWAP_LEVEL}: ${appends} appends, then ${reads} reads, then the reset.`}</T>
      <g opacity={ex3}>
        <rect x={430} y={226} width={266} height={96} rx={4} fill="var(--page)" stroke="var(--ink-2)" />
        <T x={442} y={246} size={12} weight={600}>Exercise 3: a counter per level</T>
        <T x={442} y={266} size={11}>append to numFrontier[L]</T>
        <T x={442} y={284} size={11}>grid.sync(): the only barrier</T>
        <T x={442} y={302} size={11}>read numFrontier[L], swap. Nothing is reset.</T>
      </g>
    </Svg>
  );
}

function backendsBeat(p: number): ReactNode {
  const show = (a: number) => ramp(p, a, a + 0.12);
  const col = (x: number, title: string, lines: string[], o: number) => (
    <g opacity={o}>
      <rect x={x} y={34} width={214} height={150} rx={4} fill="var(--page)" stroke="var(--ink-2)" />
      <T x={x + 10} y={54} size={13} weight={600}>{title}</T>
      {lines.map((l, k) => <T key={k} x={x + 10} y={78 + k * 20} size={11}>{l}</T>)}
    </g>
  );
  const upto = ramp(p, 0.55, 0.95);
  const total = ROAD.sizes.reduce((a, b) => a + b, 0) + ROAD.sizes.length * 2;
  const scale = (X1 - X0) / total;
  let x = X0;
  return (
    <Svg label="The same idea on HIP and Metal">
      {col(X0, 'CUDA', ['<cooperative_groups.h>', 'cudaLaunchCooperativeKernel', '  4 arguments compile', 'grid.sync()', 'ranks are 64-bit'], show(0))}
      {col(253, 'HIP', ['<hip/hip_cooperative_groups.h>', 'hipLaunchCooperativeKernel', '  all 6 arguments', 'grid.sync(), same name', 'ranks are 32-bit'], show(0.12))}
      {col(482, 'Metal', ['no grid barrier', 'no cooperative launch', 'one dispatch per level', 'serial encoder: in order', 'size from a GPU buffer'], show(0.26))}
      <g opacity={ramp(p, 0.5, 0.6)}>
        <T x={X0} y={214} size={12} weight={600}>Metal: a dispatch per level, each sized by an indirect dispatch from the previous level's count</T>
        {ROAD.sizes.map((s, k) => {
          const x0 = x;
          x += s * scale + 2 * scale;
          return x0 < X0 + upto * (X1 - X0)
            ? <rect key={k} data-dispatch={k} x={x0} y={226} width={Math.max(0.8, s * scale)} height={18} fill="var(--ink-2)" />
            : null;
        })}
        <T x={X0} y={266} size={11} muted>No CPU read between dispatches. A threadgroup count of 0 is not documented: write at least 1 and return early.</T>
      </g>
      <T x={X0} y={312} size={12} weight={600}>{p > 0.85 ? 'Everything survives except the single launch itself.' : ''}</T>
    </Svg>
  );
}

const beats: Beat[] = [
  { clip: 'one-launch-1', label: 'The host loop', draw: hostBeat },
  { clip: 'one-launch-2', label: 'Big frontiers', draw: socialBeat },
  { clip: 'one-launch-3', label: 'One grid, every level', draw: strideBeat },
  { clip: 'one-launch-4', label: 'Why it must all fit', draw: fitBeat },
  { clip: 'one-launch-5', label: 'Three barriers, three jobs', draw: swapBeat },
  { clip: 'one-launch-6', label: 'On HIP and Metal', draw: backendsBeat },
];

export function OneLaunchScene() {
  return <Scene chapter="ch18" title="One launch for every level" beats={beats} />;
}
