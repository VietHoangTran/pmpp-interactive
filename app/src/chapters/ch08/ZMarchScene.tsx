import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { fmt } from '../../lib/reports';
import { BOOK_BLOCK_LIMIT, cubeAI, sharedBytes, slabAI, threadsCube, threadsSlab } from '../../sim/stencilslab';
import { CORRECT, localZ, makeProblem, planeTimeline, type PlaneFrame, type ZConfig } from '../../sim/zmarch';

// "Marching through z": the book's picture (Fig. 8.9), a 6-wide input tile whose block
// marches through 4 output planes, drawn from the same model the stepper runs. The
// block is the interior block of a 10^3 grid, so no load or output touches the edge.
// Every plane number, active thread and counter below comes from zmarch.planeTimeline;
// the comparison figures come from stencilslab. Six beats, one narration clip each
// (content/ch08/narration/z-march-N.txt).
const T = 6;
const base = { N: 10, inTile: T, ...CORRECT };
const prob = makeProblem(base.N);
const PLANES: ZConfig = { ...base, kernel: 'planes' };
const REGS: ZConfig = { ...base, kernel: 'registers' };
const TL = planeTimeline(PLANES, prob);
const TLR = planeTimeline(REGS, prob);
const iStart = TL.iStart;
const L = (z: number | null) => (z === null ? null : localZ(iStart, z));
const F = TL.frames;
const last = F[F.length - 1];
const activeCount = F[2].active.filter(Boolean).length;
// Where each block thread's output lands, in the order the threads compute.
const activeList = F[2].active.flatMap((on, t) => (on ? [t] : []));

// ---------------------------------------------------------------- geometry
const C = 11; // cell of an input or buffer plane
const SX = (n: number) => 120 + n * 84; // left edge of input plane n (0..5)
const IY = 40; // input planes
const BY = 154; // buffers
const OY = 270; // output planes
const RX = 618; // right column
const BAR_SCALE = 300; // beat 1: drawing units for 1,024 threads, the dashed limit line
const BAR_CAP = 470; // beat 1: longest bar drawn; a longer one is broken

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

type Tone = 'shared' | 'reg';
interface Box {
  name: string;
  /** Input plane it holds (local numbering, may be fractional while sliding). */
  at: number;
  tone: Tone;
  /** 0..1: how many of its cells are filled. */
  fill?: number;
  stale?: boolean;
  /** Vertical offset from the buffer row (used when registers peel away). */
  dy?: number;
  opacity?: number;
  /** Cells to mark (ty*T + tx). */
  hot?: number[];
  /** No line to the input plane (a box that is leaving shared memory). */
  noLink?: boolean;
}
interface Frame {
  dots?: number; // beat 1: one thread per point, opacity
  block?: number; // opacity of the flattened block of threads
  active?: boolean[];
  hotThread?: number | null;
  inShared?: number[]; // input planes currently held on chip (outlined)
  boxes?: Box[];
  barrier?: string | null;
  outputs?: Record<number, number>; // local output plane -> fraction of its 16 cells filled
  counters?: [string, string][];
  bars?: number;
  compare?: number;
  headline?: ReactNode;
  arrows?: number;
  /** Beat 6: opacity of the per-thread registers drawn on the block. */
  regs?: number;
}

function Plane({ x, y, fill, stroke, wash, hot, dots, n = T, cell = C }: {
  x: number; y: number; fill: string; stroke: string; wash?: (r: number, c: number) => boolean; hot?: number[]; dots?: number; n?: number; cell?: number;
}) {
  const cells: ReactNode[] = [];
  for (let r = 0; r < n; r++)
    for (let c = 0; c < n; c++) {
      const isHot = hot?.includes(r * n + c);
      cells.push(
        <rect key={r * n + c} x={x + c * cell} y={y + r * cell} width={cell - 1.5} height={cell - 1.5} rx="1"
          fill={isHot ? stroke : wash?.(r, c) ? fill : 'var(--cell)'} stroke={stroke} strokeWidth="0.6" />,
      );
      if (dots)
        cells.push(<circle key={'d' + (r * n + c)} cx={x + c * cell + (cell - 1.5) / 2} cy={y + r * cell + (cell - 1.5) / 2} r="1.6" fill="var(--ink)" opacity={dots} />);
    }
  return <g>{cells}</g>;
}

function Stage(f: Frame) {
  const halo = (r: number, c: number) => r === 0 || c === 0 || r === T - 1 || c === T - 1;
  return (
    <svg viewBox="0 0 720 372" role="img" aria-label="Animation of a block of threads marching through z" style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      {/* input tile, plane by plane */}
      <text x={SX(0)} y={14} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}>input tile in global memory, one x-y plane per square (z increases to the right)</text>
      {Array.from({ length: T }, (_, n) => (
        <g key={n}>
          <text x={SX(n) + 33} y={IY - 6} fontSize="10.5" textAnchor="middle">plane {n}</text>
          <Plane x={SX(n)} y={IY} fill="var(--global-wash)" stroke="var(--global)" wash={(r, c) => !halo(r, c)} dots={f.dots} />
          {f.inShared?.includes(n) ? <rect x={SX(n) - 3} y={IY - 3} width={T * C + 4.5} height={T * C + 4.5} fill="none" stroke="var(--shared)" strokeWidth="2" /> : null}
        </g>
      ))}

      {/* beat 1: threads needed by one thread per point */}
      {f.bars != null ? (
        <g>
          {[
            ['this tile, one thread per point', threadsCube(T)],
            ['a 32-wide cube', threadsCube(32)],
          ].map(([label, v], n) => {
            // To scale, BAR_SCALE units per 1,024 threads. A bar that would run past BAR_CAP is
            // cut short and drawn with a break, and its label says it is not to scale.
            const full = (Number(v) / BOOK_BLOCK_LIMIT) * BAR_SCALE;
            const broken = full > BAR_CAP;
            const w = Math.min(BAR_CAP, full) * f.bars!;
            const y = BY + 6 + n * 46;
            const cut = SX(0) + BAR_CAP * 0.82;
            return (
              <g key={n} data-bar={Number(v)} data-broken={broken ? 1 : 0}>
                <text x={SX(0)} y={y} fontSize="11.5">
                  {label}: {Number(v).toLocaleString('en-US')} threads{broken ? ' (not to scale)' : ''}
                </text>
                <rect x={SX(0)} y={y + 7} width={w} height={14} fill={Number(v) > BOOK_BLOCK_LIMIT ? 'var(--fail)' : 'var(--ink-2)'} />
                {broken && SX(0) + w > cut + 10 ? (
                  <g data-mark="break">
                    <polygon points={`${cut},${y + 3} ${cut + 9},${y + 3} ${cut + 3},${y + 25} ${cut - 6},${y + 25}`} fill="var(--page)" />
                    <line x1={cut} y1={y + 3} x2={cut - 6} y2={y + 25} stroke="var(--ink)" strokeWidth="1.2" />
                    <line x1={cut + 9} y1={y + 3} x2={cut + 3} y2={y + 25} stroke="var(--ink)" strokeWidth="1.2" />
                  </g>
                ) : null}
                {Number(v) > BOOK_BLOCK_LIMIT && f.bars! > 0.9 ? (
                  <text x={SX(0) + w + 6} y={y + 19} fontSize="11" style={{ fill: 'var(--fail)' }}>{Number(v) / BOOK_BLOCK_LIMIT} times the limit</text>
                ) : null}
              </g>
            );
          })}
          <line x1={SX(0) + BAR_SCALE} x2={SX(0) + BAR_SCALE} y1={BY - 8} y2={BY + 82} stroke="var(--ink)" strokeDasharray="3 3" />
          <text x={SX(0) + BAR_SCALE + 4} y={BY - 10} fontSize="11">at most {BOOK_BLOCK_LIMIT.toLocaleString('en-US')} threads in a block</text>
        </g>
      ) : null}

      {/* the block: one plane of threads */}
      {f.block ? (
        <g opacity={f.block}>
          <text x={12} y={BY - 22} fontSize="11.5" fontWeight="600">the block</text>
          <text x={12} y={BY - 8} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>{threadsSlab(T)} threads</text>
          {Array.from({ length: T * T }, (_, t) => {
            const r = Math.floor(t / T), c = t % T;
            const on = f.active ? f.active[t] : !halo(r, c);
            const hot = f.hotThread === t;
            return (
              <g key={t} data-thread={t} data-active={on ? 1 : 0}>
                {f.regs ? (
                  <rect x={18 + c * 13 - 6} y={BY + 6 + r * 13 - 6} width={12} height={12} rx="2" fill="var(--reg-wash)" stroke="var(--reg)" strokeWidth="1.2" opacity={f.regs} />
                ) : null}
                <circle cx={18 + c * 13} cy={BY + 6 + r * 13} r={hot ? 5 : 4}
                  fill={hot ? 'var(--ok)' : on ? 'var(--ink)' : 'var(--page)'} stroke="var(--ink)" strokeWidth="1.1" />
              </g>
            );
          })}
          {f.regs && f.regs > 0.5 ? (
            <>
              <text x={12} y={BY + 92} fontSize="10" style={{ fill: 'var(--reg)' }}>each thread keeps</text>
              <text x={12} y={BY + 105} fontSize="10" style={{ fill: 'var(--reg)' }}>inPrev, inCurr, inNext</text>
              <text x={12} y={BY + 118} fontSize="10" style={{ fill: 'var(--reg)' }}>in registers</text>
            </>
          ) : (
            <>
              <text x={12} y={BY + 92} fontSize="10" style={{ fill: 'var(--ink-2)' }}>filled: computes</text>
              <text x={12} y={BY + 105} fontSize="10" style={{ fill: 'var(--ink-2)' }}>hollow: only loads</text>
            </>
          )}
          {f.barrier ? (
            <g>
              <rect x={8} y={BY - 2} width={92} height={T * 13 + 2} rx="3" fill="var(--ink)" opacity="0.18" />
              <rect x={8} y={BY + T * 13 + 2} width={92} height={4} fill="var(--ink)" />
            </g>
          ) : null}
        </g>
      ) : null}

      {/* the window: shared-memory planes (and registers) */}
      {(f.boxes ?? []).map((b) => {
        const x = SX(b.at);
        const y = BY + (b.dy ?? 0);
        const col = b.tone === 'reg' ? 'var(--reg)' : 'var(--shared)';
        const wash = b.tone === 'reg' ? 'var(--reg-wash)' : 'var(--shared-wash)';
        const filled = Math.round((b.fill ?? 1) * T * T);
        return (
          <g key={b.name} opacity={(b.opacity ?? 1) * (b.stale ? 0.55 : 1)} data-box={b.name} data-at={b.at} data-tone={b.tone} data-stale={b.stale ? 1 : 0}>
            {b.noLink ? null : <line x1={x + 33} x2={x + 33} y1={IY + T * C + 4} y2={y - 4} stroke={col} strokeWidth="1.2" strokeDasharray={b.stale ? '2 3' : undefined} />}
            <Plane x={x} y={y} fill={wash} stroke={col} wash={(r, c) => r * T + c < filled} hot={b.hot} />
            <text x={x + 33} y={y + T * C + 13} fontSize="10.5" textAnchor="middle" fontWeight="600" style={{ fill: col }}>{b.name}</text>
            {b.stale ? <text x={x + 33} y={y + T * C + 25} fontSize="9.5" textAnchor="middle">free for the next load</text> : null}
          </g>
        );
      })}
      {f.arrows ? (
        <text x={SX(0)} y={BY - 31} fontSize="11" opacity={f.arrows} style={{ paintOrder: 'stroke', stroke: 'var(--page)', strokeWidth: 5 }}>each thread copies its own point: current → previous, next → current</text>
      ) : null}
      {f.barrier ? (
        <g>
          <rect x={SX(0) - 8} y={BY - 26} width={SX(5) + 74 - SX(0)} height={4} fill="var(--ink)" />
          <text x={SX(0)} y={BY - 31} fontSize="11.5" fontWeight="600" style={{ paintOrder: 'stroke', stroke: 'var(--page)', strokeWidth: 5 }}>{f.barrier}</text>
        </g>
      ) : null}

      {/* outputs */}
      {f.outputs ? (
        <g>
          <text x={12} y={OY + 26} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--ok)' }}>output</text>
          {[1, 2, 3, 4].map((n) => {
            const frac = f.outputs![n] ?? 0;
            const done = Math.round(frac * activeCount);
            return (
              <g key={n} data-output={n} data-done={done}>
                <text x={SX(n) + 33} y={OY - 5} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>output plane {n}</text>
                {Array.from({ length: (T - 2) * (T - 2) }, (_, q) => {
                  const r = Math.floor(q / (T - 2)), c = q % (T - 2);
                  return (
                    <rect key={q} x={SX(n) + 11 + c * C} y={OY + r * C} width={C - 1.5} height={C - 1.5} rx="1"
                      fill={q < done ? 'var(--ok)' : 'var(--cell)'} stroke={q < done ? 'var(--ok)' : 'var(--rule)'} strokeWidth="0.6" />
                  );
                })}
              </g>
            );
          })}
        </g>
      ) : null}

      {/* counters */}
      {f.counters ? (
        <g>
          {f.counters.map(([k, v], n) => (
            <g key={k} data-counter={k} data-value={v}>
              <text x={RX} y={BY + n * 34} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>{k}</text>
              <text x={RX} y={BY + 15 + n * 34} fontSize="14" fontWeight="650">{v}</text>
            </g>
          ))}
        </g>
      ) : null}

      {/* beat 5: the same slab at full width */}
      {f.compare ? (
        <g opacity={f.compare}>
          <rect x={SX(0) - 6} y={BY - 4} width={322} height={74} rx="4" fill="var(--page)" stroke="var(--ink)" />
          <text x={SX(0) + 4} y={BY + 15} fontSize="12" fontWeight="650">march at width 32: {threadsSlab(32).toLocaleString('en-US')} threads, {fmt(slabAI(32, 30))} FLOP/B</text>
          <text x={SX(0) + 4} y={BY + 36} fontSize="12">8-wide cube: {threadsCube(8)} threads, {fmt(cubeAI(8), 2)} FLOP/B</text>
          <text x={SX(0) + 4} y={BY + 56} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>both by the book's count of loads and stores (p. 192)</text>
        </g>
      ) : null}

      {f.headline ? <text x={12} y={364} fontSize="13" fontWeight="600">{f.headline}</text> : null}
    </svg>
  );
}

// ---------------------------------------------------------------- frames from the model
const counters = (fr: PlaneFrame, partStores?: number): [string, string][] => [
  ['planes loaded', String(fr.planesLoaded)],
  ['elements loaded', String(fr.loads)],
  ['outputs stored', String(partStores ?? fr.stores)],
];
const outputsOf = (fr: PlaneFrame) => Object.fromEntries(fr.outputPlanes.map((z) => [L(z)!, 1]));
// Where each part of the window sits, by frame index. After a shift, inNext_s still holds
// a copy of the new current plane that nobody reads; it is drawn empty, under the plane
// the next iteration loads into it (the next frame's hold.next).
const nextAt = (k: number) => (F[k].label === 'shifted' ? (F[k + 1] ? L(F[k + 1].hold.next) : null) : L(F[k].hold.next));
const windowOf = (k: number, before = k, t = 1): Box[] => {
  const fr = F[k];
  const pos = (now: number | null, then: number | null) => lerp(then ?? now ?? 0, now ?? then ?? 0, t);
  const boxes: Box[] = [
    { name: 'inPrev_s', at: pos(L(fr.hold.prev), L(F[before].hold.prev)), tone: 'shared' },
    { name: 'inCurr_s', at: pos(L(fr.hold.curr), L(F[before].hold.curr)), tone: 'shared' },
  ];
  const nx = nextAt(k);
  if (nx !== null) {
    const stale = fr.label === 'shifted';
    boxes.push({ name: 'inNext_s', at: pos(nx, nextAt(before) ?? nx), tone: 'shared', stale, fill: stale ? 0 : 1 });
  }
  return boxes;
};
const onChip = (fr: PlaneFrame) => [...new Set([fr.hold.prev, fr.hold.curr, fr.label === 'shifted' ? null : fr.hold.next].filter((z) => z !== null).map((z) => L(z)!))];

// One step of the march at fraction t of its way through the frames (1, 2, 3 for iteration n).
function marchFrame(n: number, t: number): Frame {
  const kl = 1 + 3 * n; // loaded
  const kc = 2 + 3 * n; // computed
  const ks = 3 + 3 * n; // shifted
  const kb = 3 * n; // the frame before this iteration
  const computed = F[kc];
  if (t < 1 / 3) {
    const u = t * 3;
    return {
      block: 1, active: computed.active, inShared: onChip(F[kl]), boxes: windowOf(kl, kb, Math.min(1, u * 2)),
      outputs: outputsOf(F[kb]), counters: counters(u > 0.5 ? F[kl] : F[kb]), barrier: u > 0.6 ? 'barrier 1' : null,
    };
  }
  if (t < 2 / 3)
    return { block: 1, active: computed.active, inShared: onChip(F[kl]), boxes: windowOf(kc), outputs: outputsOf(computed), counters: counters(computed) };
  const u = (t - 2 / 3) * 3;
  return {
    block: 1, active: computed.active, inShared: onChip(u < 0.3 ? computed : F[ks]), boxes: u < 0.3 ? windowOf(kc) : windowOf(ks, kc, ramp(u, 0.3, 0.9)),
    outputs: outputsOf(computed), counters: counters(F[ks]), barrier: u < 0.3 ? 'barrier 2' : null,
  };
}

const crossOf = (t: number) => {
  const r = Math.floor(t / T), c = t % T;
  return [t, (r - 1) * T + c, (r + 1) * T + c, r * T + c - 1, r * T + c + 1];
};

const beats: Beat[] = [
  {
    clip: 'z-march-1',
    label: 'Too many threads',
    draw: (p) => (
      <Stage dots={ramp(p, 0.05, 0.3)} bars={ramp(p, 0.3, 0.75)}
        headline={p < 0.5 ? `one thread per point: ${threadsCube(T)} threads for this tile` : `a 32-wide cube would need ${threadsCube(32).toLocaleString('en-US')}`} />
    ),
  },
  {
    clip: 'z-march-2',
    label: 'One plane',
    draw: (p) => {
      const f0 = F[0];
      const prevFill = ramp(p, 0.35, 0.6);
      const currFill = ramp(p, 0.65, 0.9);
      const boxes: Box[] = [];
      if (p >= 0.35) boxes.push({ name: 'inPrev_s', at: L(f0.hold.prev)!, tone: 'shared', fill: prevFill });
      if (p >= 0.65) boxes.push({ name: 'inCurr_s', at: L(f0.hold.curr)!, tone: 'shared', fill: currFill });
      const loadedSoFar = Math.min(f0.planesLoaded, (prevFill >= 1 ? 1 : 0) + (currFill >= 1 ? 1 : 0));
      return (
        <Stage block={ramp(p, 0.02, 0.3)} boxes={boxes} inShared={boxes.filter((b) => (b.fill ?? 0) >= 1).map((b) => b.at)}
          counters={[['planes loaded', String(loadedSoFar)], ['threads', String(threadsSlab(T))], ['of which compute', String(activeCount)]]}
          headline={p < 0.35 ? `${threadsSlab(T)} threads: one x-y plane of the tile` : 'before the loop: two planes loaded'} />
      );
    },
  },
  {
    clip: 'z-march-3',
    label: 'Load and compute',
    draw: (p) => {
      const loaded = F[1];
      const computed = F[2];
      const nextFill = ramp(p, 0.04, 0.24);
      const boxes = windowOf(1).map((b) => (b.name === 'inNext_s' ? { ...b, fill: nextFill } : b));
      const progress = ramp(p, 0.42, 0.95);
      const k = Math.min(activeList.length - 1, Math.floor(progress * activeList.length));
      const computing = p >= 0.42 && p < 0.97;
      const thread = activeList[k];
      if (computing)
        for (const b of boxes)
          b.hot = b.name === 'inCurr_s' ? crossOf(thread) : [thread];
      const stores = computing ? Math.round(progress * computed.stores) : p >= 0.97 ? computed.stores : 0;
      return (
        <Stage block={1} active={computed.active} hotThread={computing ? thread : null} boxes={boxes}
          inShared={onChip(loaded)} barrier={p >= 0.26 && p < 0.42 ? 'barrier 1: wait until the whole plane is in' : null}
          outputs={{ [L(computed.outputPlanes[0])!]: stores / computed.stores }}
          counters={counters(nextFill >= 1 ? loaded : F[0], stores)}
          headline={p < 0.42 ? 'load one new plane, then wait' : `${activeCount} threads: five values from the current plane, one before, one after`} />
      );
    },
  },
  {
    clip: 'z-march-4',
    label: 'Roll',
    draw: (p) => {
      const computed = F[2];
      const shifted = F[3];
      const slide = ramp(p, 0.3, 0.7);
      return (
        <Stage block={1} active={computed.active} boxes={slide > 0 ? windowOf(3, 2, slide) : windowOf(2)} inShared={onChip(slide >= 1 ? shifted : computed)}
          barrier={p < 0.3 ? 'barrier 2: no overwriting until everyone has read' : null} arrows={ramp(p, 0.3, 0.4) * (1 - ramp(p, 0.75, 0.85))}
          outputs={outputsOf(computed)} counters={counters(shifted)}
          headline={p < 0.62 ? 'barrier 2, then shift the window by one plane' : 'barrier 1: true dependence. barrier 2: false dependence'} />
      );
    },
  },
  {
    clip: 'z-march-5',
    label: 'March on',
    draw: (p) => {
      const rest = (F.length - 4) / 3; // iterations after the first
      if (p < 0.72) {
        const x = (p / 0.72) * rest;
        const n = Math.min(rest - 1, Math.floor(x));
        return <Stage {...marchFrame(n + 1, x - n)} headline={`iteration ${n + 2}: output plane ${L(F[2 + 3 * (n + 1)].i)}`} />;
      }
      return (
        <Stage block={1} active={last.active} boxes={windowOf(F.length - 1)} inShared={onChip(last)} outputs={outputsOf(last)}
          counters={counters(last)} compare={ramp(p, 0.76, 0.86)}
          headline={`${last.planesLoaded} input planes loaded for ${last.outputPlanes.length} output planes`} />
      );
    },
  },
  {
    clip: 'z-march-6',
    label: 'Registers',
    draw: (p) => {
      const at = TLR.frames[5]; // registers kernel, iteration 1, after the compute
      const same = F[5]; // the planes kernel at the same point
      const peel = ramp(p, 0.3, 0.75);
      const away = (z: number | null): Box['at'] => lerp(L(z)!, -1.1, peel);
      const boxes: Box[] = [
        { name: 'inPrev_s', at: away(at.hold.prev), tone: 'shared', opacity: 1 - peel, noLink: peel > 0 },
        { name: 'inCurr_s', at: L(at.sharedCurr)!, tone: 'shared' },
        { name: 'inNext_s', at: away(at.hold.next), tone: 'shared', opacity: 1 - peel, noLink: peel > 0 },
      ];
      return (
        <Stage block={1} active={at.active} boxes={boxes} regs={ramp(p, 0.55, 0.8)}
          inShared={peel >= 1 ? [L(at.sharedCurr)!] : onChip(same)} outputs={outputsOf(at)}
          counters={[
            ['loaded, registers', String(at.loads)],
            ['loaded, 3 planes', String(same.loads)],
            ['shared planes', peel >= 1 ? '1' : '3'],
            ['bytes at t = 32', sharedBytes(peel >= 1 ? 'registers' : 'planes', 32).toLocaleString('en-US')],
          ]}
          headline={p < 0.3 ? 'previous and next are read by one thread only' : 'into registers: shared memory keeps only the current plane'} />
      );
    },
  },
];

export function ZMarchScene() {
  return <Scene chapter="ch08" title="Marching through z" beats={beats} />;
}
