import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import {
  SCENE_K,
  SCENE_ROW,
  SELECT_PER_KEY,
  SORT_PER_KEY,
  hex,
  selectTrace,
  statusAfter,
  type Decided,
} from '../../sim/topkselect';

// "Narrowing in on the k-th key" (section X1.4): the radix select of src/sim/topkselect.ts on
// its scene row (64 logits, k = 12, one block of 16 threads, one 32-lane warp). Six beats, one
// narration clip each (content/x1/narration/radix-select-N.txt). Every key, bin, count,
// prefix and slot drawn comes from selectTrace, pinned in topkselect.test.ts.
const TR = selectTrace(SCENE_ROW, SCENE_K, 16, 32);
const P = TR.passes;
const T = TR.blockThreads;
const R = Math.ceil(TR.n / T);

const W = 720;
const H = 372;
const MONO = { fontVariationSettings: "'MONO' 1" } as const;
const GX = 14; // grid origin
const GY = 44;
const LW = 44; // thread label column
const CW = 80; // cell width
const RH = 18; // row height
const RX = 398; // right panel
const HY = 46; // right panel top
const BH = 15; // histogram row height
const HW = 110; // longest histogram bar

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const short = (v: number) => (Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
const bytesOf = (key: number) => hex(key).slice(2).match(/../g) as string[];
const binHex = (b: number) => b.toString(16).toUpperCase().padStart(2, '0');

type CellMode = 'value' | 'key' | 'gather';
interface Cell {
  state: Decided | 'active' | 'idle' | 'kept' | 'dropped';
  slot?: number;
}
interface Frame {
  mode: CellMode;
  /** Byte of the key in bold (0 = top byte), or null. */
  byte: number | null;
  /** Round (column of the grid) being processed, or null. */
  round: number | null;
  cells: Cell[];
  hist?: number[];
  histLabel?: string;
  /** Crossing bin to mark in the histogram. */
  mark?: number;
  walk?: { bin: number; count: number; running: number; cross: boolean }[];
  lines?: ReactNode[];
  slots?: (number | null)[];
  bars?: number;
  headline: string;
}

const decided = (passes: number): Cell[] => Array.from({ length: TR.n }, (_, c) => ({ state: statusAfter(TR, passes, c) }));

function Grid({ f }: { f: Frame }) {
  return (
    <g>
      <text x={GX} y={GY - 22} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}>
        the row, as one block of 16 threads reads it
      </text>
      <text x={GX} y={GY - 7} fontSize="10" style={{ fill: 'var(--ink-2)' }}>thread</text>
      {Array.from({ length: R }, (_, r) => (
        <text key={r} x={GX + LW + r * CW + CW / 2} y={GY - 7} fontSize="10" textAnchor="middle"
          fontWeight={f.round === r ? 700 : 400} style={{ fill: f.round === r ? 'var(--ink)' : 'var(--ink-2)' }}>
          {r === 0 ? 'column t' : `t + ${r * T}`}
        </text>
      ))}
      {f.round != null ? (
        <rect x={GX + LW + f.round * CW - 2} y={GY - 3} width={CW} height={T * RH + 4} rx="3" fill="var(--rule-soft)" />
      ) : null}
      {Array.from({ length: T }, (_, t) => (
        <text key={t} x={GX + LW - 10} y={GY + t * RH + 12.5} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{t}</text>
      ))}
      {Array.from({ length: TR.n }, (_, c) => {
        const r = Math.floor(c / T);
        const t = c % T;
        const cell = f.cells[c];
        const x = GX + LW + r * CW;
        const y = GY + t * RH;
        const st = cell.state;
        const fill = st === 'active' ? 'var(--ink)' : st === 'in' || st === 'kept' ? 'var(--ok-wash)' : st === 'idle' ? 'var(--rule)' : 'var(--cell)';
        const stroke = st === 'in' || st === 'kept' ? 'var(--ok)' : st === 'active' ? 'var(--ink)' : 'var(--rule)';
        const ink = st === 'active' ? 'var(--page)' : 'var(--ink)';
        const dim = st === 'out' ? 0.32 : st === 'dropped' ? 0.6 : st === 'idle' ? 0.55 : 1;
        return (
          <g key={c} data-col={c} data-state={st} data-slot={cell.slot ?? ''} opacity={dim}>
            <rect x={x} y={y} width={CW - 4} height={RH - 3} rx="2" fill={fill} stroke={stroke} />
            {f.mode === 'value' ? (
              <text x={x + (CW - 4) / 2} y={y + 11.5} fontSize="10.5" textAnchor="middle" style={{ ...MONO, fill: ink }}>{short(TR.logits[c])}</text>
            ) : f.mode === 'gather' ? (
              <text x={x + 5} y={y + 11.5} fontSize="10" style={{ ...MONO, fill: ink }}>
                c{c}
                {cell.slot != null ? <tspan fontWeight="700" style={{ fill: 'var(--ok)' }} dx="5">→ {cell.slot}</tspan> : null}
                {st === 'dropped' ? <tspan dx="5">tie cut</tspan> : null}
              </text>
            ) : (
              <text x={x + 4} y={y + 11.5} fontSize="10" style={{ ...MONO, fill: ink }}>
                {bytesOf(TR.keys[c]).map((b, i) => (
                  <tspan key={i} fontWeight={f.byte === i ? 800 : 400} opacity={f.byte != null && i > f.byte ? 0.45 : 1}>
                    {b}{i < 3 ? ' ' : ''}
                  </tspan>
                ))}
              </text>
            )}

          </g>
        );
      })}
    </g>
  );
}

/** The nonempty bins of the 256, highest first, as bars; with the walk's running count beside them. */
function Hist({ f }: { f: Frame }) {
  if (!f.hist) return null;
  const rows = f.hist.map((c, b) => [b, c]).filter(([, c]) => c > 0).reverse();
  const max = Math.max(1, ...f.hist);
  const crossAt = f.walk?.findIndex((w) => w.cross) ?? -1;
  const walkDone = crossAt >= 0;
  return (
    <g>
      <text x={RX} y={HY - 24} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--shared)' }}>{f.histLabel ?? 'histogram, 256 bins in shared memory'}</text>
      <text x={RX} y={HY - 10} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>nonempty bins only, highest first</text>
      {rows.length === 0 ? <text x={RX} y={HY + 10} fontSize="10.5" style={{ fill: 'var(--ink-3)' }}>all 256 bins at 0</text> : null}
      {rows.map(([b, c], i) => {
        const y = HY + i * BH;
        const w = f.walk?.find((x) => x.bin === b);
        const below = walkDone && !w;
        return (
          <g key={b} data-bin={b} data-count={c} opacity={below ? 0.4 : 1}>
            <text x={RX} y={y + 10} fontSize="10" style={MONO}>bin {binHex(b)}</text>
            <rect x={RX + 52} y={y + 2} width={(c / max) * HW} height={BH - 5} fill={f.mark === b ? 'var(--ink)' : 'var(--shared)'} />
            <text x={RX + 56 + (c / max) * HW} y={y + 10} fontSize="10">{c}</text>
            {w ? (
              <text x={RX + 82 + HW} y={y + 10} fontSize="10" fontWeight={w.cross ? 700 : 400} data-walk={`${w.bin}:${w.count}:${w.running}`}>
                running {w.running}{w.cross ? ' ← the k-th key is here' : ': all in'}
              </text>
            ) : below ? (
              <text x={RX + 82 + HW} y={y + 10} fontSize="10">all out</text>
            ) : null}
          </g>
        );
      })}
    </g>
  );
}

function Right({ f }: { f: Frame }) {
  const nRows = f.hist ? f.hist.filter((c) => c > 0).length : 0;
  const y0 = HY + Math.max(nRows, 1) * BH + 16;
  return (
    <g>
      <Hist f={f} />
      {f.lines ? (
        <g>
          {f.lines.map((l, i) => (
            <text key={i} x={RX} y={y0 + i * 17} fontSize="11" data-line={i}>{l}</text>
          ))}
        </g>
      ) : null}
      {f.slots ? (
        <g>
          <text x={RX} y={HY - 24} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}>output: {SCENE_K} (column, logit) pairs, column order</text>
          {f.slots.map((c, i) => (
            <g key={i} data-out={i} data-out-col={c ?? ''}>
              <rect x={RX + i * 26} y={HY - 8} width={23} height={20} rx="2" fill={c == null ? 'none' : 'var(--ok-wash)'}
                stroke={c == null ? 'var(--rule)' : 'var(--ok)'} strokeDasharray={c == null ? '3 3' : undefined} />
              {c != null ? <text x={RX + i * 26 + 11.5} y={HY + 6} fontSize="10" textAnchor="middle">{c}</text> : null}
            </g>
          ))}
        </g>
      ) : null}
      {f.bars != null ? (
        <g data-bars="">
          <text x={RX} y={HY + 46} fontSize="11">radix select: {SELECT_PER_KEY} bytes per key (5 reads of the key)</text>
          <rect x={RX} y={HY + 52} width={(SELECT_PER_KEY / SORT_PER_KEY) * 300 * f.bars} height={12} fill="var(--global)" />
          <text x={RX} y={HY + 84} fontSize="11">LSD radix sort: {SORT_PER_KEY} bytes per key (4 passes)</text>
          <rect x={RX} y={HY + 90} width={300 * f.bars} height={12} fill="var(--global)" />
        </g>
      ) : null}
    </g>
  );
}

function Stage({ f, beat }: { f: Frame; beat: number }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Radix select narrowing in on the k-th key"
      data-beat={beat} style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <Grid f={f} />
      <Right f={f} />
      <text x={GX} y={H - 8} fontSize="12.5" fontWeight="650" data-headline="">{f.headline}</text>
    </svg>
  );
}

/** A counting pass, round by round: the current round's lanes are active or idle. */
function counting(pass: number, p: number): Frame {
  const ps = P[pass];
  const r = Math.min(R - 1, Math.floor(ramp(p, 0, 0.88) * R));
  const done = p >= 0.88;
  const cells = decided(pass).map((c, col) => {
    const round = Math.floor(col / T);
    if (!done && round === r) return { state: ps.rounds[r].lanes[col % T].match ? ('active' as const) : ('idle' as const) };
    return c;
  });
  const hist = done ? ps.hist : r > 0 ? ps.rounds[r - 1].histAfter : new Array(256).fill(0);
  const active = ps.rounds[r].lanes.filter((l) => l.match).length;
  return {
    mode: 'key', byte: pass, round: done ? null : r, cells, hist,
    histLabel: `pass ${pass + 1}: histogram of byte ${pass + 1}, 256 bins in shared memory`,
    headline: done
      ? `pass ${pass + 1} done: ${ps.candidates} keys counted in one read of the row`
      : `round ${r + 1}: thread t reads column t + ${r * T}; ${active} of ${T} lanes do the atomic`,
  };
}

function walked(pass: number, p: number): Frame {
  const ps = P[pass];
  const shown = ps.walk.steps.filter((_, i) => p > 0.1 + i * 0.22);
  const show = p > 0.72;
  return {
    mode: 'key', byte: pass, round: null, cells: decided(show ? pass + 1 : pass), hist: ps.hist,
    histLabel: `pass ${pass + 1}: histogram of byte ${pass + 1}`, mark: shown.length === ps.walk.steps.length ? ps.bin : undefined,
    walk: shown.map((w) => ({ bin: w.bin, count: w.count, running: w.running, cross: w.bin === ps.bin })),
    lines: show ? [`in: ${ps.above}, out: ${ps.below}, undecided: ${ps.inBin}`, `still needed from bin ${binHex(ps.bin)}: ${ps.needOut}`] : undefined,
    headline: show
      ? `${ps.above} in, ${ps.below} out, ${ps.inBin} undecided in bin ${binHex(ps.bin)}: ${ps.needOut} still needed`
      : 'walk the bins from the top until the running count reaches k',
  };
}

function gathered(p: number): Frame {
  const r = Math.min(R - 1, Math.floor(ramp(p, 0, 0.6) * R));
  const done = p >= 0.6;
  const upto = done ? R - 1 : r;
  const cells: Cell[] = decided(P.length).map((c) => c);
  const slots: (number | null)[] = new Array(SCENE_K).fill(null);
  for (const g of TR.gather.slice(0, upto + 1))
    for (const l of g.lanes) {
      if (l.col == null) continue;
      if (l.keep) {
        cells[l.col] = { state: 'kept', slot: l.slot as number };
        slots[l.slot as number] = l.col;
      } else if (l.cls === 'tie') cells[l.col] = { state: 'dropped' };
    }
  return {
    mode: 'gather', byte: null, round: done ? null : r, cells, slots, bars: done ? ramp(p, 0.62, 0.9) : undefined,
    headline: done
      ? `${SCENE_K} pairs in column order; the select read ${SELECT_PER_KEY} bytes per key, a sort moves ${SORT_PER_KEY}`
      : `gather, round ${r + 1}: keys above K* kept, ties kept lowest column first`,
  };
}

const beats: Beat[] = [
  {
    clip: 'radix-select-1',
    label: 'Keys',
    draw: (p) => {
      const keys = p >= 0.45;
      return (
        <Stage beat={1} f={{
          mode: keys ? 'key' : 'value', byte: keys ? 0 : null, round: null, cells: decided(0),
          hist: keys ? new Array(256).fill(0) : undefined,
          headline: keys ? 'each logit becomes a 32-bit key; the top byte is in bold' : `${TR.n} logits; we want the ${SCENE_K} largest`,
        }} />
      );
    },
  },
  { clip: 'radix-select-2', label: 'Top-byte histogram', draw: (p) => <Stage beat={2} f={counting(0, p)} /> },
  { clip: 'radix-select-3', label: 'Walk the bins', draw: (p) => <Stage beat={3} f={walked(0, p)} /> },
  {
    clip: 'radix-select-4',
    label: 'Only the prefix works',
    draw: (p) => {
      const f = p < 0.8 ? counting(1, p / 0.8 * 0.95) : walked(1, 1);
      if (p < 0.8)
        f.headline = p / 0.8 * 0.95 >= 0.88
          ? `pass 2: only ${P[1].candidates} keys match prefix ${binHex(P[0].bin)}; every round split the warp`
          : f.headline + '; grey lanes skip the branch';
      return <Stage beat={4} f={f} />;
    },
  },
  {
    clip: 'radix-select-5',
    label: 'The k-th key',
    draw: (p) => {
      const n = p < 0.5 ? 3 : 4;
      const last = P[n - 1];
      const lines = P.slice(0, n).map((ps) => `pass ${ps.index + 1}: bin ${binHex(ps.bin)}, ${ps.above} more in, ${ps.needOut} still needed`);
      return (
        <Stage beat={5} f={{
          mode: 'key', byte: n - 1, round: null, cells: decided(n), hist: last.hist, mark: last.bin,
          histLabel: `pass ${n}: histogram of byte ${n}`, lines,
          headline: n === 4
            ? `K* = ${hex(TR.prefix)}: ${TR.greater} keys above it, ${TR.need} of its ${TR.equal} ties needed`
            : `pass 3: ${last.inBin} keys left in bin ${binHex(last.bin)}, ${last.needOut} still needed`,
        }} />
      );
    },
  },
  { clip: 'radix-select-6', label: 'The gather', draw: (p) => <Stage beat={6} f={gathered(p)} /> },
];

export function RadixSelectScene() {
  return <Scene chapter="x1" title="Narrowing in on the k-th key" beats={beats} />;
}
