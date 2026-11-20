import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import {
  basicTaps,
  laneActivity,
  mapThread,
  pyramidFilter,
  stripsOf,
  toyImage,
  warpDivergence,
  warpsInCorner,
  type WarpClass,
} from '../../sim/convolve';

// "One thread per output": section 7.2's basic kernel in time, on the 16 x 16 toy image
// of Fig. 7.6 with 4 x 4 blocks and r = 2. Six beats, one narration clip each
// (content/ch07/narration/one-thread-per-output-N.txt). Every cell, count and sum drawn
// comes from sim/convolve.ts: mapThread, basicTaps, laneActivity, warpDivergence.

export const TOY = 16;
export const BD = 4;
export const R = 2;
const M = 2 * R + 1;
const N = toyImage(TOY, TOY);
const F = pyramidFilter(R);
/** Beat 2 and 3's thread, the book's example (p. 166). */
export const PICK = { bx: 1, by: 1, tx: 1, ty: 1, bdx: BD, bdy: BD };
/** Beat 4 and 5's thread: the top-left corner. */
export const CORNER = { bx: 0, by: 0, tx: 0, ty: 0, bdx: BD, bdy: BD };
const pickMap = mapThread(PICK, R);
const pickRun = basicTaps(TOY, TOY, R, pickMap.outRow, pickMap.outCol, N.data, F.data);
const cornerMap = mapThread(CORNER, R);
const cornerRun = basicTaps(TOY, TOY, R, cornerMap.outRow, cornerMap.outCol, N.data, F.data);
/** Beat 5: the 16 threads of block (0, 0), in linear order (x fastest). */
const LANES = Array.from({ length: BD * BD }, (_, t) => mapThread({ ...CORNER, tx: t % BD, ty: Math.floor(t / BD) }, R));
const ACT = laneActivity(TOY, TOY, R, LANES.map((l) => ({ row: l.outRow, col: l.outCol })));
export const SPLIT_TAPS = ACT[0].filter((_, k) => ACT.some((l) => l[k]) && ACT.some((l) => !l[k])).length;
/** Beat 6: a large image, as the narration says. */
export const BIG = { width: 1024, height: 1024, block: 32, warp: 32 };
const big = warpDivergence(BIG.width, BIG.height, BIG.block, BIG.block, R, BIG.warp);
export const BIG_SHARE = big.ghost / big.warps;
const bigStrips = stripsOf(big.marked);
const ZOOM = 64;
const zoomWarps = warpsInCorner(BIG.width, BIG.height, BIG.block, BIG.block, R, BIG.warp, ZOOM, ZOOM);

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const big3 = (v: number) => v.toLocaleString('en-US');

// Geometry
const C = 14; // cell size of the two 16 x 16 grids
const IX = 42, IY = 64; // input grid origin (cell 0, 0)
const OX = 322, OY = 64; // output grid origin
const TX = 572; // text column

const CLS_FILL: Record<WarpClass, string> = {
  uniform: 'color-mix(in srgb, var(--ink) 14%, var(--cell))',
  ghost: 'var(--ink)',
  guard: 'color-mix(in srgb, var(--ink) 55%, var(--cell))',
  idle: 'var(--cell)',
};

interface GridsProps {
  /** Output blocks lit, in raster order (beat 1). */
  lit?: number;
  /** The selected output. */
  out?: { row: number; col: number };
  /** Patch corner and opacity. */
  patch?: { row: number; col: number; opacity: number };
  /** Taps already done (input cells washed), and the current one. */
  done?: { row: number; col: number; inside: boolean }[];
  cur?: { row: number; col: number; inside: boolean } | null;
}

function Grids({ lit = 16, out, patch, done = [], cur }: GridsProps) {
  const ghostCells: ReactNode[] = [];
  if (patch)
    for (let a = 0; a < M; a++)
      for (let b = 0; b < M; b++) {
        const row = patch.row + a;
        const col = patch.col + b;
        if (row >= 0 && row < TOY && col >= 0 && col < TOY) continue;
        const isCur = cur && cur.row === row && cur.col === col;
        ghostCells.push(
          <rect key={`g${row},${col}`} x={IX + col * C} y={IY + row * C} width={C - 1.5} height={C - 1.5} rx="1"
            fill="none" stroke={isCur ? 'var(--ink)' : 'var(--ink-3)'} strokeWidth={isCur ? 1.6 : 0.8} strokeDasharray="2.5 2" opacity={patch.opacity} />,
        );
      }
  const doneSet = new Set(done.filter((d) => d.inside).map((d) => d.row * TOY + d.col));
  return (
    <g>
      <text x={IX} y={IY - 12 - (patch && patch.row < 0 ? -patch.row * C : 0)} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}>N (input)</text>
      {N.data.map((_, k) => {
        const row = Math.floor(k / TOY);
        const col = k % TOY;
        const hot = cur && cur.inside && cur.row === row && cur.col === col;
        const inPatch = patch && row >= patch.row && row < patch.row + M && col >= patch.col && col < patch.col + M;
        return (
          <rect key={k} x={IX + col * C} y={IY + row * C} width={C - 1.5} height={C - 1.5} rx="1"
            fill={hot ? 'var(--global)' : doneSet.has(k) ? 'var(--global-wash)' : 'var(--cell)'}
            stroke={inPatch ? 'var(--global)' : 'var(--rule)'} strokeWidth={inPatch ? 1 : 0.6} strokeOpacity={inPatch ? patch!.opacity : 1} />
        );
      })}
      {ghostCells}
      {patch ? (
        <rect x={IX + patch.col * C - 1.5} y={IY + patch.row * C - 1.5} width={M * C + 1.5} height={M * C + 1.5} fill="none"
          stroke="var(--ink)" strokeWidth="1.6" opacity={patch.opacity} />
      ) : null}
      <text x={OX} y={OY - 12} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}>P (output)</text>
      {Array.from({ length: (TOY / BD) * (TOY / BD) }, (_, b) => {
        const bx = b % (TOY / BD);
        const by = Math.floor(b / (TOY / BD));
        return (
          <rect key={'b' + b} x={OX + bx * BD * C - 1} y={OY + by * BD * C - 1} width={BD * C} height={BD * C}
            fill={b < lit ? 'var(--global-wash)' : 'none'} stroke="var(--ink-2)" strokeWidth="1.4" />
        );
      })}
      {N.data.map((_, k) => {
        const row = Math.floor(k / TOY);
        const col = k % TOY;
        const sel = out && out.row === row && out.col === col;
        return (
          <rect key={'o' + k} x={OX + col * C + 0.5} y={OY + row * C + 0.5} width={C - 2.5} height={C - 2.5} rx="1"
            fill={sel ? 'var(--global)' : 'transparent'} stroke="var(--rule)" strokeWidth="0.5" />
        );
      })}
    </g>
  );
}

function FilterBox({ cur }: { cur?: number | null }) {
  const c = 15;
  return (
    <g>
      <text x={TX} y={206} fontSize="11.5" fontWeight="600">filter F, {M} × {M}</text>
      {F.data.map((v, k) => (
        <g key={k}>
          <rect x={TX + (k % M) * c} y={214 + Math.floor(k / M) * c} width={c - 2} height={c - 2} rx="1"
            fill={cur === k ? 'var(--global)' : 'var(--cell)'} stroke="var(--global)" strokeWidth="0.6" />
          <text x={TX + (k % M) * c + (c - 2) / 2} y={214 + Math.floor(k / M) * c + 10} fontSize="8.5" textAnchor="middle"
            style={{ fill: cur === k ? 'var(--page)' : 'var(--ink)' }}>{v}</text>
        </g>
      ))}
    </g>
  );
}

function Register({ value, y = 130 }: { value: number; y?: number }) {
  return (
    <g>
      <text x={TX} y={y - 6} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--reg)' }}>Pvalue (a register)</text>
      <rect x={TX} y={y} width={120} height={28} rx="3" fill="var(--reg-wash)" stroke="var(--reg)" />
      <text x={TX + 60} y={y + 19} fontSize="14" fontWeight="700" textAnchor="middle" data-pvalue={value}>{value}</text>
    </g>
  );
}

function Stage({ children, headline, data }: { children: ReactNode; headline?: ReactNode; data?: Record<string, string | number> }) {
  const attrs = Object.fromEntries(Object.entries(data ?? {}).map(([k, v]) => ['data-' + k.toLowerCase(), String(v)]));
  return (
    <svg viewBox="0 0 720 340" role="img" aria-label="Animation of the basic convolution kernel" style={{ width: '100%', maxWidth: 720, display: 'block' }} {...attrs}>
      <defs>
        <pattern id="ch07-idle" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="5" stroke="var(--ink-3)" strokeWidth="1.1" />
        </pattern>
      </defs>
      {children}
      {headline ? <text x={20} y={328} fontSize="13" fontWeight="600" data-headline="">{headline}</text> : null}
    </svg>
  );
}

const tapCell = (run: typeof pickRun, k: number) => {
  const t = run.taps[k];
  return { row: t.inRow, col: t.inCol, inside: t.inside };
};

function walk(run: typeof pickRun, map: typeof pickMap, p: number, store: boolean) {
  const k = Math.min(run.taps.length, Math.floor(ramp(p, 0.04, 0.9) * (run.taps.length + 0.999)));
  const done = run.taps.slice(0, k).map((_, j) => tapCell(run, j));
  const cur = k > 0 ? tapCell(run, k - 1) : null;
  const t = k > 0 ? run.taps[k - 1] : null;
  const pv = t ? t.pvalue! : 0;
  const skipped = run.taps.slice(0, k).filter((x) => !x.inside).length;
  const stored = store && k === run.taps.length && p > 0.93 ? run.pvalue : null;
  return { k, done, cur, t, pv, skipped, stored, map };
}

const beats: Beat[] = [
  {
    clip: 'one-thread-per-output-1',
    label: 'Sixteen blocks',
    draw: (p) => {
      const lit = Math.min(16, Math.floor(ramp(p, 0.08, 0.8) * 16.99));
      return (
        <Stage headline={`${lit} of 16 blocks: ${lit * BD * BD} threads, one output each`} data={{ lit }}>
          <Grids lit={lit} />
          <text x={TX} y={80} fontSize="12">{TOY} × {TOY} outputs</text>
          <text x={TX} y={98} fontSize="12">{BD} × {BD} threads per block</text>
          <text x={TX} y={116} fontSize="12">{(TOY / BD) * (TOY / BD)} blocks</text>
        </Stage>
      );
    },
  },
  {
    clip: 'one-thread-per-output-2',
    label: 'One thread',
    draw: (p) => {
      const m = pickMap;
      const show = ramp(p, 0.55, 0.75);
      return (
        <Stage headline={`thread (${PICK.tx}, ${PICK.ty}) of block (${PICK.bx}, ${PICK.by}) computes P[${m.outRow}][${m.outCol}]`}
          data={{ outRow: m.outRow, outCol: m.outCol, cornerRow: m.cornerRow, cornerCol: m.cornerCol }}>
          <Grids out={p > 0.12 ? { row: m.outRow, col: m.outCol } : undefined}
            patch={show > 0 ? { row: m.cornerRow, col: m.cornerCol, opacity: show } : undefined} />
          <text x={TX} y={80} fontSize="11.5" opacity={ramp(p, 0.12, 0.25)}>outCol = {PICK.bx}·{BD} + {PICK.tx} = {m.outCol}</text>
          <text x={TX} y={98} fontSize="11.5" opacity={ramp(p, 0.25, 0.38)}>outRow = {PICK.by}·{BD} + {PICK.ty} = {m.outRow}</text>
          <text x={TX} y={130} fontSize="11.5" opacity={show}>patch corner</text>
          <text x={TX} y={148} fontSize="11" opacity={show}>(outRow − {R}, outCol − {R})</text>
          <text x={TX} y={166} fontSize="11.5" fontWeight="600" opacity={show}>= ({m.cornerRow}, {m.cornerCol})</text>
        </Stage>
      );
    },
  },
  {
    clip: 'one-thread-per-output-3',
    label: '25 taps',
    draw: (p) => {
      const w = walk(pickRun, pickMap, p, true);
      return (
        <Stage
          headline={w.stored != null ? `one store: P[${pickRun.store}] = ${w.stored}` : w.t ? `tap ${w.k} of ${M * M}: ${w.t.weight} × ${w.t.value} = ${w.t.product}` : 'tap by tap'}
          data={{ tap: w.k, pvalue: w.pv, stored: w.stored ?? '' }}>
          <Grids out={{ row: pickMap.outRow, col: pickMap.outCol }} patch={{ row: pickMap.cornerRow, col: pickMap.cornerCol, opacity: 1 }}
            done={w.done} cur={w.cur} />
          <Register value={w.pv} />
          {w.stored != null ? (
            <text x={TX} y={184} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }} data-stored={w.stored}>stored: P[{w.map === pickMap ? pickRun.store : cornerRun.store}] = {w.stored}</text>
          ) : null}
          <FilterBox cur={w.t ? w.t.fIndex : null} />
          {w.t ? (
            <text x={TX} y={80} fontSize="11.5">N[{w.t.inRow}][{w.t.inCol}] × F[{w.t.fRow}][{w.t.fCol}]</text>
          ) : null}
        </Stage>
      );
    },
  },
  {
    clip: 'one-thread-per-output-4',
    label: 'A corner thread',
    draw: (p) => {
      const w = walk(cornerRun, cornerMap, p, true);
      return (
        <Stage
          headline={w.k < M * M ? `${w.skipped} of ${w.k} taps so far skipped as ghost cells` : `${cornerRun.skipped} of ${M * M} taps skipped as ghost cells, ${M * M - cornerRun.skipped} multiplied`}
          data={{ tap: w.k, skipped: w.skipped, total: cornerRun.skipped }}>
          <Grids out={{ row: cornerMap.outRow, col: cornerMap.outCol }} patch={{ row: cornerMap.cornerRow, col: cornerMap.cornerCol, opacity: 1 }}
            done={w.done} cur={w.cur} />
          <Register value={w.pv} />
          {w.stored != null ? (
            <text x={TX} y={184} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }} data-stored={w.stored}>stored: P[{w.map === pickMap ? pickRun.store : cornerRun.store}] = {w.stored}</text>
          ) : null}
          <FilterBox cur={w.t ? w.t.fIndex : null} />
          {w.t ? (
            <text x={TX} y={80} fontSize="11.5" fontWeight={w.t.inside ? 400 : 600}>
              {w.t.inside ? `N[${w.t.inRow}][${w.t.inCol}] × F[${w.t.fRow}][${w.t.fCol}]` : `(${w.t.inRow}, ${w.t.inCol}) is a ghost cell: skipped`}
            </text>
          ) : null}
        </Stage>
      );
    },
  },
  {
    clip: 'one-thread-per-output-5',
    label: 'Idle lanes',
    draw: (p) => {
      const taps = M * M;
      const k = Math.min(taps, Math.floor(ramp(p, 0.05, 0.92) * (taps + 0.999)));
      const cw = 17, ch = 13, X0 = 92, Y0 = 70;
      const active = k > 0 ? ACT.filter((l) => l[k - 1]).length : 0;
      const splitSoFar = ACT[0].slice(0, k).filter((_, j) => ACT.some((l) => l[j]) && ACT.some((l) => !l[j])).length;
      return (
        <Stage
          headline={k > 0 ? `tap ${k}: ${active} of ${ACT.length} lanes multiply, ${ACT.length - active} sit idle; ${splitSoFar} taps so far split the warp` : 'block (0, 0): 16 threads, one warp'}
          data={{ tap: k, active, split: splitSoFar, splitTotal: SPLIT_TAPS }}>
          <text x={X0} y={Y0 - 28} fontSize="12" fontWeight="600">taps, in loop order →</text>
          <text x={14} y={Y0 - 8} fontSize="11" style={{ fill: 'var(--ink-2)' }}>lane (output)</text>
          {ACT.map((lane, li) => (
            <g key={li}>
              <text x={X0 - 6} y={Y0 + li * ch + 10} fontSize="9.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
                {li} ({LANES[li].outRow},{LANES[li].outCol})
              </text>
              {lane.map((on, j) => {
                if (j >= k) return <rect key={j} x={X0 + j * cw} y={Y0 + li * ch} width={cw - 2} height={ch - 2} fill="none" stroke="var(--rule)" strokeWidth="0.5" />;
                return on ? (
                  <rect key={j} x={X0 + j * cw} y={Y0 + li * ch} width={cw - 2} height={ch - 2} fill="var(--ink)" />
                ) : (
                  <rect key={j} x={X0 + j * cw} y={Y0 + li * ch} width={cw - 2} height={ch - 2} fill="url(#ch07-idle)" stroke="var(--ink-3)" strokeWidth="0.5" />
                );
              })}
            </g>
          ))}
          {k > 0 ? (
            <rect x={X0 + (k - 1) * cw - 1.5} y={Y0 - 3} width={cw + 1} height={ACT.length * ch + 4} fill="none" stroke="var(--ink)" strokeWidth="1.5" />
          ) : null}
          <rect x={X0} y={Y0 + ACT.length * ch + 14} width={12} height={10} fill="var(--ink)" />
          <text x={X0 + 18} y={Y0 + ACT.length * ch + 23} fontSize="11">multiplies</text>
          <rect x={X0 + 100} y={Y0 + ACT.length * ch + 14} width={12} height={10} fill="url(#ch07-idle)" stroke="var(--ink-3)" strokeWidth="0.5" />
          <text x={X0 + 118} y={Y0 + ACT.length * ch + 23} fontSize="11">idle: its tap is a ghost cell</text>
        </Stage>
      );
    },
  },
  {
    clip: 'one-thread-per-output-6',
    label: 'Zoom out',
    draw: (p) => {
      const S = 230, X0 = 40, Y0 = 50;
      const s = S / BIG.width;
      const zs = S / ZOOM;
      const ZX = 380;
      const show = ramp(p, 0.15, 0.4);
      return (
        <Stage
          headline={`${big3(big.ghost)} of ${big3(big.warps)} warps diverge: ${Math.round(BIG_SHARE * 10000) / 100} %, all at the left and right edges`}
          data={{ warps: big.warps, ghost: big.ghost, strips: bigStrips.length }}>
          <text x={X0} y={Y0 - 10} fontSize="12" fontWeight="600">{BIG.width} × {BIG.height} image, {BIG.block} × {BIG.block} blocks</text>
          <rect x={X0} y={Y0} width={S} height={S} fill={CLS_FILL.uniform} stroke="var(--ink)" strokeWidth="1" />
          {bigStrips.map((st, k) => (
            <rect key={k} x={X0 + st.c0 * s} y={Y0 + st.row0 * s} width={Math.max(3, (st.c1 - st.c0 + 1) * s)} height={Math.max(3, (st.row1 - st.row0 + 1) * s)}
              fill={CLS_FILL[st.cls]} opacity={show} />
          ))}
          <rect x={X0 - 1} y={Y0 - 1} width={ZOOM * s + 2} height={ZOOM * s + 2} fill="none" stroke="var(--ink)" strokeWidth="1.2" strokeDasharray="3 2" opacity={ramp(p, 0.4, 0.5)} />
          <g opacity={ramp(p, 0.45, 0.6)}>
            <text x={ZX} y={Y0 - 10} fontSize="12" fontWeight="600">its top-left {ZOOM} × {ZOOM} threads, one row of a block per warp</text>
            {zoomWarps.flatMap((w, wi) =>
              w.segments.filter((sg) => sg.row < ZOOM && sg.c0 < ZOOM).map((sg, si) => (
                <rect key={`${wi}-${si}`} x={ZX + sg.c0 * zs} y={Y0 + sg.row * zs} width={(Math.min(sg.c1, ZOOM - 1) - sg.c0 + 1) * zs - 0.6} height={zs - 0.6}
                  fill={CLS_FILL[w.cls]} />
              )),
            )}
            <line x1={ZX + R * zs} x2={ZX + R * zs} y1={Y0} y2={Y0 + S} stroke="var(--page)" strokeWidth="1" strokeDasharray="2 2" />
            <line x1={ZX} x2={ZX + S} y1={Y0 + R * zs} y2={Y0 + R * zs} stroke="var(--ink-2)" strokeWidth="1" strokeDasharray="2 2" />
            <text x={ZX + S + 6} y={Y0 + R * zs + 4} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>row {R}</text>
          </g>
          <rect x={X0} y={Y0 + S + 16} width={12} height={10} fill={CLS_FILL.ghost} />
          <text x={X0 + 18} y={Y0 + S + 25} fontSize="11">lanes disagree at the ghost test</text>
          <rect x={X0 + 210} y={Y0 + S + 16} width={12} height={10} fill={CLS_FILL.uniform} stroke="var(--rule)" />
          <text x={X0 + 228} y={Y0 + S + 25} fontSize="11">every lane takes the same path</text>
        </Stage>
      );
    },
  },
];

export function OneThreadScene() {
  return <Scene chapter="ch07" title="One thread per output" beats={beats} />;
}
