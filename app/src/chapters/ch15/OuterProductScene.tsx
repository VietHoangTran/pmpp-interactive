import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { threadTile } from '../../sim/gemmmap';
import { BOOK_SHAPE, afterStrip, run, sharedReadsV1, sharedReadsV2, smallInputs, stateAfterOps, stateAt, type State } from '../../sim/gemmreads';

// "Filling a register tile with outer products" (section 15.4): one thread, one tile
// pair, 8 x 8 x 8, drawn from the traces in src/sim/gemmreads.ts. Six beats, one
// narration clip each (content/ch15/narration/outer-product-N.txt).
const S = BOOK_SHAPE;
const X = smallInputs(S);
export const SCENE = {
  shape: S,
  v1: run(S, X, 'v1'),
  v2: run(S, X, 'v2'),
  readsV1: sharedReadsV1(S),
  readsV2: sharedReadsV2(S),
};
const V1 = SCENE.v1;
const V2 = SCENE.v2;

const C = 17; // cell size
const BX = 192, BY = 14; // b: the thread's 8 columns of B_s
const SBY = 158; // b_r
const AX = 20, AY = 184; // a: the thread's 8 rows of A_s
const SAX = 164; // a_r
const CX = BX, CY = AY; // the thread's patch of C
const RX = 372; // counters

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

interface Frame {
  st: State;
  order: 'v1' | 'v2';
  /** Shared slots read in this step, and the C cell written. */
  hotA?: Set<number>;
  hotB?: Set<number>;
  hotC?: Set<number>;
  hotAR?: Set<number>;
  hotBR?: Set<number>;
  /** Show re-read counts on the slots. */
  counts?: boolean;
  /** Draw the strips at all. */
  strips?: boolean;
  /** Reads shown on the two counters. */
  r1: number;
  r2: number | null;
  same?: boolean;
  headline: ReactNode;
}

function Cell({ x, y, v, fill, stroke, ink = 'var(--ink)', badge, attrs }: { x: number; y: number; v: ReactNode; fill: string; stroke: string; ink?: string; badge?: string; attrs?: Record<string, string> }) {
  return (
    <g {...attrs}>
      <rect x={x} y={y} width={C - 1.5} height={C - 1.5} rx="1.5" fill={fill} stroke={stroke} strokeWidth="0.8" />
      <text x={x + C / 2 - 0.75} y={y + C / 2 + 3} fontSize="9" textAnchor="middle" style={{ fill: ink }}>{v}</text>
      {badge ? <text x={x + C - 2} y={y + 6} fontSize="6.5" textAnchor="end" style={{ fill: 'var(--fail)' }}>{badge}</text> : null}
    </g>
  );
}

function Stage(f: Frame) {
  const { st } = f;
  const cells: ReactNode[] = [];
  for (let r = 0; r < S.tM; r++)
    for (let k = 0; k < S.bK; k++) {
      const i = r * S.bK + k;
      const n = st.aReads[i];
      const hot = f.hotA?.has(i);
      cells.push(
        <Cell key={'a' + i} x={AX + k * C} y={AY + r * C} v={X.a[i]} attrs={{ 'data-a': String(i), 'data-reads': String(n) }}
          fill={hot ? 'var(--shared)' : n ? 'var(--shared-wash)' : 'var(--cell)'} stroke="var(--shared)" ink={hot ? 'var(--page)' : 'var(--ink)'}
          badge={f.counts && n > 1 ? `×${n}` : undefined} />,
      );
    }
  for (let k = 0; k < S.bK; k++)
    for (let c = 0; c < S.tN; c++) {
      const i = k * S.tN + c;
      const n = st.bReads[i];
      const hot = f.hotB?.has(i);
      cells.push(
        <Cell key={'b' + i} x={BX + c * C} y={BY + k * C} v={X.b[i]} attrs={{ 'data-b': String(i), 'data-reads': String(n) }}
          fill={hot ? 'var(--shared)' : n ? 'var(--shared-wash)' : 'var(--cell)'} stroke="var(--shared)" ink={hot ? 'var(--page)' : 'var(--ink)'}
          badge={f.counts && n > 1 ? `×${n}` : undefined} />,
      );
    }
  for (let r = 0; r < S.tM; r++)
    for (let c = 0; c < S.tN; c++) {
      const i = r * S.tN + c;
      const hot = f.hotC?.has(i);
      const v = st.C[i];
      const u = st.updates[i] > 0;
      cells.push(
        <Cell key={'c' + i} x={CX + c * C} y={CY + r * C} v={u ? v : ''} attrs={{ 'data-c': String(i) }}
          fill={hot ? 'var(--reg)' : u ? 'var(--reg-wash)' : 'var(--cell)'} stroke="var(--reg)" ink={hot ? 'var(--page)' : 'var(--ink)'} />,
      );
    }
  if (f.strips) {
    for (let r = 0; r < S.tM; r++) {
      const v = st.aR[r];
      const hot = f.hotAR?.has(r);
      cells.push(<Cell key={'ar' + r} x={SAX} y={AY + r * C} v={v ?? ''} fill={hot ? 'var(--reg)' : v != null ? 'var(--reg-wash)' : 'var(--cell)'} stroke="var(--reg)" ink={hot ? 'var(--page)' : 'var(--ink)'} />);
    }
    for (let c = 0; c < S.tN; c++) {
      const v = st.bR[c];
      const hot = f.hotBR?.has(c);
      cells.push(<Cell key={'br' + c} x={BX + c * C} y={SBY} v={v ?? ''} fill={hot ? 'var(--reg)' : v != null ? 'var(--reg-wash)' : 'var(--cell)'} stroke="var(--reg)" ink={hot ? 'var(--page)' : 'var(--ink)'} />);
    }
  }
  const bar = (n: number) => (300 * n) / SCENE.readsV1;
  return (
    <svg viewBox="0 0 720 360" role="img" aria-label="One thread's multiply-adds in two loop orders" style={{ width: '100%', maxWidth: 720, display: 'block' }} data-order={f.order}>
      <text x={BX} y={BY - 4} fontSize="10.5" fontWeight="600" style={{ fill: 'var(--shared)' }}>B_s: the thread's 8 columns</text>
      <text x={AX} y={AY - 6} fontSize="10.5" fontWeight="600" style={{ fill: 'var(--shared)' }}>A_s: its 8 rows</text>
      {f.strips ? (
        <>
          <text x={SAX + C / 2} y={AY - 6} fontSize="10" textAnchor="middle" style={{ fill: 'var(--reg)' }}>a_r</text>
          <text x={BX + S.tN * C + 4} y={SBY + 12} fontSize="10" style={{ fill: 'var(--reg)' }}>b_r</text>
        </>
      ) : null}
      <text x={CX} y={CY + S.tM * C + 14} fontSize="10.5" fontWeight="600" style={{ fill: 'var(--reg)' }}>C_r: its 8 × 8 patch (registers)</text>
      {cells}
      {/* counters */}
      <text x={RX} y={34} fontSize="11.5" fontWeight="600">shared-memory reads, one thread, one tile pair</text>
      <text x={RX} y={58} fontSize="11">first order (Fig. 15.6)</text>
      <rect x={RX} y={64} width={300} height={14} fill="var(--cell)" stroke="var(--rule)" />
      <rect x={RX} y={64} width={bar(f.r1)} height={14} fill="var(--shared)" data-bar="v1" />
      <text x={RX + 306} y={75} fontSize="11" data-count="v1">{f.r1}</text>
      {f.r2 != null ? (
        <>
          <text x={RX} y={104} fontSize="11">loops swapped (Fig. 15.9)</text>
          <rect x={RX} y={110} width={300} height={14} fill="var(--cell)" stroke="var(--rule)" />
          <rect x={RX} y={110} width={bar(f.r2)} height={14} fill="var(--shared)" data-bar="v2" />
          <text x={RX + 306} y={121} fontSize="11" data-count="v2">{f.r2}</text>
        </>
      ) : null}
      <text x={RX} y={156} fontSize="11" style={{ fill: 'var(--ink-2)' }}>multiply-adds so far: {st.fmas} of {V1.fmas}</text>
      {f.same ? (
        <text x={RX} y={180} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--ok)' }} data-same>✓ same 64 values as the first order</text>
      ) : null}
      <text x={RX} y={330} fontSize="12.5" fontWeight="600">{f.headline}</text>
    </svg>
  );
}

const rowOf = (r: number, n: number) => new Set(Array.from({ length: n }, (_, k) => r * n + k));
const colOf = (c: number, rows: number, n: number) => new Set(Array.from({ length: rows }, (_, k) => k * n + c));

/** Highlights for the step that ended state `st` in order `order`. */
function hot(st: State, order: 'v1' | 'v2'): Pick<Frame, 'hotA' | 'hotB' | 'hotC' | 'hotAR' | 'hotBR'> {
  const last = st.step[st.step.length - 1];
  if (!last) return {};
  if (order === 'v1' && last.kind === 'fma')
    return { hotA: new Set([last.row * S.bK + last.i]), hotB: new Set([last.i * S.tN + last.col]), hotC: new Set([last.row * S.tN + last.col]) };
  const loads = st.step.filter((o) => o.kind !== 'fma');
  const out: Pick<Frame, 'hotA' | 'hotB' | 'hotC' | 'hotAR' | 'hotBR'> = {
    hotA: new Set(loads.flatMap((o) => (o.kind === 'loadA' ? [o.row * S.bK + o.i] : []))),
    hotB: new Set(loads.flatMap((o) => (o.kind === 'loadB' ? [o.i * S.tN + o.col] : []))),
    hotAR: new Set(loads.flatMap((o) => (o.kind === 'loadA' ? [o.row] : []))),
    hotBR: new Set(loads.flatMap((o) => (o.kind === 'loadB' ? [o.col] : []))),
  };
  if (last.kind === 'fma') {
    out.hotC = new Set([last.row * S.tN + last.col]);
    out.hotAR!.add(last.row);
    out.hotBR!.add(last.col);
  }
  return out;
}

/** A whole rank-1 update: the strips of step i and all 64 cells. */
function rank1(i: number): Pick<Frame, 'hotA' | 'hotB' | 'hotC' | 'hotAR' | 'hotBR'> {
  return {
    hotA: colOf(i, S.tM, S.bK),
    hotB: rowOf(i, S.tN),
    hotC: new Set(Array.from({ length: S.tM * S.tN }, (_, k) => k)),
    hotAR: new Set(Array.from({ length: S.tM }, (_, k) => k)),
    hotBR: new Set(Array.from({ length: S.tN }, (_, k) => k)),
  };
}

/** The block-level picture: tiles in shared memory, one thread's strips in registers. */
function Zoom({ p }: { p: number }) {
  const s = 1.5; // px per element
  const ax = 30, ay = 70, bx = ax + 8 * s + 12, by = 40;
  const tile = 128 * s;
  const { tRow, tCol } = threadTile(101); // an example thread: patch row 6, column 5
  const i = 3;
  const show = ramp(p, 0.35, 0.6);
  return (
    <svg viewBox="0 0 720 360" role="img" aria-label="Tiling at two levels" style={{ width: '100%', maxWidth: 720, display: 'block' }} data-zoom>
      <text x={ax} y={22} fontSize="11.5" fontWeight="600">one block</text>
      <rect x={bx} y={by} width={tile} height={8 * s} fill="var(--shared-wash)" stroke="var(--shared)" />
      <rect x={bx + tCol * s} y={by} width={8 * s} height={8 * s} fill="var(--shared)" opacity={0.55} />
      <rect x={bx + tCol * s} y={by + i * s} width={8 * s} height={s} fill="var(--shared)" />
      <text x={bx + tile + 6} y={by + 10} fontSize="10" style={{ fill: 'var(--shared)' }}>B_s (shared)</text>
      <rect x={ax} y={ay} width={8 * s} height={tile} fill="var(--shared-wash)" stroke="var(--shared)" />
      <rect x={ax} y={ay + tRow * s} width={8 * s} height={8 * s} fill="var(--shared)" opacity={0.55} />
      <rect x={ax + i * s} y={ay + tRow * s} width={s} height={8 * s} fill="var(--shared)" />
      <text x={ax - 4} y={ay - 6} fontSize="10" style={{ fill: 'var(--shared)' }}>A_s (shared)</text>
      <rect x={bx} y={ay} width={tile} height={tile} fill="var(--cell)" stroke="var(--ink)" />
      {Array.from({ length: 15 }, (_, k) => (
        <g key={k} stroke="var(--rule-soft)">
          <line x1={bx + (k + 1) * 8 * s} y1={ay} x2={bx + (k + 1) * 8 * s} y2={ay + tile} />
          <line x1={bx} y1={ay + (k + 1) * 8 * s} x2={bx + tile} y2={ay + (k + 1) * 8 * s} />
        </g>
      ))}
      <rect x={bx + tCol * s} y={ay + tRow * s} width={8 * s} height={8 * s} fill="var(--reg)" />
      <text x={bx} y={ay + tile + 16} fontSize="10" style={{ fill: 'var(--ink-2)' }}>C tile, 128 × 128, in the registers of 256 threads</text>
      <g opacity={show}>
        <line x1={bx + tCol * s + 12} y1={ay + tRow * s} x2={430} y2={110} stroke="var(--reg)" strokeDasharray="3 2" />
        <line x1={bx + tCol * s + 12} y1={ay + tRow * s + 12} x2={430} y2={250} stroke="var(--reg)" strokeDasharray="3 2" />
        <text x={430} y={84} fontSize="11.5" fontWeight="600">one thread</text>
        {Array.from({ length: 8 }, (_, r) => <rect key={'a' + r} x={430} y={124 + r * 15} width={13} height={13} fill="var(--reg-wash)" stroke="var(--reg)" />)}
        {Array.from({ length: 8 }, (_, c) => <rect key={'b' + c} x={450 + c * 15} y={104} width={13} height={13} fill="var(--reg-wash)" stroke="var(--reg)" />)}
        {Array.from({ length: 64 }, (_, k) => <rect key={'c' + k} x={450 + (k % 8) * 15} y={124 + Math.floor(k / 8) * 15} width={13} height={13} fill="var(--reg)" opacity={0.85} />)}
        <text x={575} y={114} fontSize="10" style={{ fill: 'var(--reg)' }}>b_r</text>
        <text x={430} y={258} fontSize="10" style={{ fill: 'var(--reg)' }}>a_r</text>
        <text x={430} y={280} fontSize="11" style={{ fill: 'var(--reg)' }}>C_r += a_r ⊗ b_r, once per step i</text>
      </g>
      <text x={ax} y={330} fontSize="12.5" fontWeight="600" opacity={ramp(p, 0.6, 0.8)}>
        global → shared, once per block; shared → registers, once per thread: an accumulated outer product
      </text>
    </svg>
  );
}

const finalV1 = stateAt(V1, V1.fmas);
const beats: Beat[] = [
  {
    clip: 'outer-product-1',
    label: 'One output',
    draw: (p) => {
      const k = Math.min(S.bK, Math.floor(ramp(p, 0.3, 0.95) * (S.bK + 0.99)));
      const st = stateAt(V1, k);
      return <Stage st={st} order="v1" {...hot(st, 'v1')} r1={st.sharedReads} r2={null} headline={`output (0, 0): ${st.sharedReads} shared reads so far`} />;
    },
  },
  {
    clip: 'outer-product-2',
    label: 'Read again',
    draw: (p) => {
      const k = p < 0.45 ? S.bK + Math.floor(ramp(p, 0.02, 0.4) * S.bK) : Math.round(2 * S.bK + ramp(p, 0.45, 0.92) * (V1.fmas - 2 * S.bK));
      const st = stateAt(V1, k);
      return (
        <Stage st={st} order="v1" {...hot(st, 'v1')} counts r1={st.sharedReads} r2={null}
          headline={k >= V1.fmas ? `${st.sharedReads} shared reads for ${V1.fmas} multiply-adds` : 'the same values of A, read again for every column'} />
      );
    },
  },
  {
    clip: 'outer-product-3',
    label: 'Swap the loops',
    draw: (p) => {
      const n = Math.round(ramp(p, 0.25, 0.85) * (S.tM + S.tN));
      const st = stateAfterOps(V2, n);
      return <Stage st={st} order="v2" {...hot(st, 'v2')} strips r1={SCENE.readsV1} r2={st.sharedReads} headline="step 0: column 0 of A and row 0 of B into registers" />;
    },
  },
  {
    clip: 'outer-product-4',
    label: 'Outer product',
    draw: (p) => {
      const k = Math.round(ramp(p, 0.1, 0.6) * S.tM * S.tN);
      // Before the first multiply-add: both strips loaded, nothing computed yet.
      const st = k ? stateAt(V2, k) : stateAfterOps(V2, S.tM + S.tN);
      const h = k >= S.tM * S.tN ? rank1(0) : k ? hot(st, 'v2') : {};
      return <Stage st={st} order="v2" {...h} strips r1={SCENE.readsV1} r2={st.sharedReads}
        headline={`${st.sharedReads} shared reads feed ${S.tM * S.tN} multiply-adds`} />;
    },
  },
  {
    clip: 'outer-product-5',
    label: 'Steps 1 to 7',
    draw: (p) => {
      const k = Math.round(afterStrip(S, 0) + ramp(p, 0.05, 0.75) * (V2.fmas - afterStrip(S, 0)));
      const i = Math.min(S.bK - 1, Math.floor((k - 1) / (S.tM * S.tN)));
      const st = stateAt(V2, k);
      const done = k >= V2.fmas;
      return (
        <Stage st={st} order="v2" {...rank1(i)} strips r1={SCENE.readsV1} r2={st.sharedReads} same={done && p > 0.8 && st.C.every((v, j) => v === finalV1.C[j])}
          headline={done ? `${st.sharedReads} reads against ${SCENE.readsV1}: ${SCENE.readsV1 / SCENE.readsV2} times fewer` : `step ${i}: ${st.sharedReads} reads so far`} />
      );
    },
  },
  { clip: 'outer-product-6', label: 'Two levels', draw: (p) => <Zoom p={p} /> },
];

export function OuterProductScene() {
  return <Scene chapter="ch15" title="Filling a register tile with outer products" beats={beats} />;
}
