import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { DEMO_SEED, amdScalarEligible, distinctCount, filterTrace, replaySchedule, warpAddresses } from '../../sim/constcache';

// "One address, one warp" (section 7.4): six beats, one narration clip each
// (content/ch07/narration/constant-broadcast-N.txt). Every drawing is computed from
// sim/constcache.ts: the filter taps from filterTrace, the codebook's addresses and
// rounds from warpAddresses and replaySchedule with the fixed DEMO_SEED, so the
// count the narration says is the count drawn (pinned in constcache.test.ts).

// ---------------------------------------------------------------- the shared drawing
export type LaneState = 'idle' | 'asking' | 'served' | 'waiting';
export type EntryState = 'idle' | 'asked' | 'hot' | 'done';

/**
 * One warp above one table: a box per lane, a box per entry, a line from each lane to
 * the entry it reads. Used by the scene and by the section 7.4 explorer.
 */
export function WarpTable({
  addresses, tableSize, x = 20, y = 28, width = 680, gap = 120, lane = () => 'asking', entry = () => 'idle',
  lines = 1, hotLines, laneLabel, tableLabel, entryLabel, rounds,
}: {
  addresses: readonly number[];
  tableSize: number;
  x?: number;
  y?: number;
  width?: number;
  gap?: number;
  lane?: (i: number) => LaneState;
  entry?: (e: number) => EntryState;
  /** Opacity of the lane-to-entry lines. */
  lines?: number;
  /** Entries whose lines are drawn heavy (the one being answered). */
  hotLines?: Set<number>;
  laneLabel?: ReactNode;
  tableLabel?: ReactNode;
  entryLabel?: (e: number) => string | null;
  /** Round number shown over each asked entry. */
  rounds?: Map<number, number>;
}) {
  const lw = width / addresses.length;
  const ew = width / tableSize;
  const laneH = 18;
  const ty = y + laneH + gap;
  const laneFill: Record<LaneState, string> = { idle: 'var(--cell)', asking: 'var(--cell)', served: 'var(--const)', waiting: 'var(--cell)' };
  const entryFill: Record<EntryState, string> = { idle: 'var(--cell)', asked: 'var(--cell)', hot: 'var(--const)', done: 'var(--const-wash)' };
  return (
    <g>
      {laneLabel ? <text x={x} y={y - 7} fontSize="12" fontWeight="600">{laneLabel}</text> : null}
      {tableLabel ? <text x={x} y={ty + 24 + 16} fontSize="12" fontWeight="600" style={{ fill: 'var(--const)' }}>{tableLabel}</text> : null}
      {lines > 0
        ? addresses.map((a, i) => {
            const heavy = hotLines?.has(a);
            return (
              <line key={'l' + i} x1={x + i * lw + lw / 2} y1={y + laneH} x2={x + a * ew + ew / 2} y2={ty}
                stroke={heavy ? 'var(--const)' : 'var(--ink-3)'} strokeWidth={heavy ? 1.6 : 0.7}
                opacity={lines * (heavy || !hotLines ? 1 : 0.35)} />
            );
          })
        : null}
      {addresses.map((_, i) => {
        const s = lane(i);
        return (
          <rect key={'n' + i} data-lane={i} data-state={s} x={x + i * lw + Math.min(1, lw * 0.1)} y={y} width={Math.max(1, lw - Math.min(2, lw * 0.2))} height={laneH} rx="1.5"
            fill={laneFill[s]} stroke={s === 'idle' ? 'var(--rule)' : s === 'served' ? 'var(--const)' : 'var(--ink-2)'} strokeWidth={s === 'asking' ? 1.3 : 0.8}
            strokeDasharray={s === 'waiting' ? '2 2' : undefined} opacity={s === 'waiting' ? 0.55 : 1} />
        );
      })}
      {Array.from({ length: tableSize }, (_, e) => {
        const s = entry(e);
        const label = entryLabel?.(e);
        return (
          <g key={'e' + e} data-entry={e} data-state={s}>
            <rect x={x + e * ew + Math.min(0.5, ew * 0.1)} y={ty} width={Math.max(0.8, ew - Math.min(1, ew * 0.2))} height={24} rx="1.5"
              fill={entryFill[s]} stroke={s === 'idle' ? 'var(--rule)' : 'var(--const)'} strokeWidth={s === 'asked' ? 1.4 : 0.8} />
            {label && ew >= 16 ? (
              <text x={x + e * ew + ew / 2} y={ty + 16} fontSize={ew >= 30 ? 11 : 9.5} textAnchor="middle"
                style={{ fill: s === 'hot' ? 'var(--page)' : 'var(--ink-2)' }}>{label}</text>
            ) : null}
            {rounds?.has(e) && ew >= 12 ? (
              <text x={x + e * ew + ew / 2} y={ty - 4} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{rounds.get(e)}</text>
            ) : null}
          </g>
        );
      })}
    </g>
  );
}

// ---------------------------------------------------------------- the scene's data, all from the model
const WARP = 32; // the book's warp; AMD's 64-wide wavefronts are in the explorer
const R = 2; // the 5x5 filter of the book's examples
const TAPS = filterTrace(R);
const M = 2 * R + 1;
const CODEBOOK = 16;
const CODES = warpAddresses('codebook', { warpSize: WARP, tableSize: CODEBOOK, seed: DEMO_SEED });
const ROUNDS = replaySchedule(CODES);
const DISTINCT = distinctCount(CODES);
const UNIFORM = warpAddresses('uniform', { warpSize: WARP, entry: 0 });

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const tapName = (t: number) => `F[${TAPS[t].fRow}][${TAPS[t].fCol}]`;

function Stage({ children, data, headline }: { children: ReactNode; data: Record<string, string | number>; headline?: ReactNode }) {
  const attrs = Object.fromEntries(Object.entries(data).map(([k, v]) => ['data-' + k, String(v)]));
  return (
    <svg viewBox="0 0 720 380" role="img" aria-label="One warp reading a table in constant memory" style={{ width: '100%', maxWidth: 900, minWidth: 520, display: 'block' }} {...attrs}>
      {children}
      {headline ? <text x={20} y={368} fontSize="13" fontWeight="600" data-headline>{headline}</text> : null}
    </svg>
  );
}

function Counter({ x, y, label, value, of }: { x: number; y: number; label: string; value: number; of?: number }) {
  return (
    <g data-counter={label}>
      <text x={x} y={y} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>{label}</text>
      <text x={x} y={y + 24} fontSize="20" fontWeight="650" data-value>{value}</text>
      {of != null ? <text x={x + 14 + String(value).length * 11} y={y + 24} fontSize="12" style={{ fill: 'var(--ink-2)' }}>of {of}</text> : null}
    </g>
  );
}

/** The rounds as a row of boxes, filled as they are answered. */
function RoundStrip({ x, y, total, done, width = 420 }: { x: number; y: number; total: number; done: number; width?: number }) {
  const w = Math.min(28, width / total);
  return (
    <g data-strip={total}>
      {Array.from({ length: total }, (_, i) => (
        <rect key={i} x={x + i * w} y={y} width={w - 3} height={16} rx="2"
          fill={i < done ? 'var(--const)' : 'var(--cell)'} stroke="var(--const)" strokeWidth="0.8" />
      ))}
    </g>
  );
}

// A small warp-and-table picture for the vendor panels.
function Mini({ x, y, addresses, tableSize, title, verdict, ok }: {
  x: number; y: number; addresses: readonly number[]; tableSize: number; title: string; verdict: string; ok: boolean | null;
}) {
  return (
    <g>
      <text x={x} y={y} fontSize="12" fontWeight="600">{title}</text>
      <WarpTable addresses={addresses} tableSize={tableSize} x={x} y={y + 22} width={300} gap={70}
        lane={() => 'asking'} entry={(e) => (addresses.includes(e) ? 'asked' : 'idle')} lines={0.9} />
      <text x={x} y={y + 160} fontSize="12" data-verdict={ok == null ? 'none' : ok ? 'yes' : 'no'}
        style={{ fill: ok == null ? 'var(--ink-2)' : ok ? 'var(--ok)' : 'var(--fail)' }}>{verdict}</text>
    </g>
  );
}

const beats: Beat[] = [
  {
    clip: 'constant-broadcast-1',
    label: 'One address',
    draw: (p) => {
      const lit = p >= 0.3;
      const served = p >= 0.55;
      return (
        <Stage data={{ beat: 1, distinct: distinctCount(UNIFORM), answers: served ? 1 : 0 }}
          headline={served ? 'one address, one answer, all 32 lanes served' : 'every lane asks for F[0][0]'}>
          <WarpTable addresses={UNIFORM} tableSize={M * M} lines={ramp(p, 0.05, 0.25)} hotLines={served ? new Set([0]) : undefined}
            lane={() => (served ? 'served' : 'asking')} entry={(e) => (e === 0 && lit ? 'hot' : 'idle')}
            laneLabel="one warp: 32 lanes" tableLabel="F: the 5 × 5 filter in constant memory, row by row" entryLabel={(e) => String(e)} />
          <Counter x={20} y={250} label="answers fetched" value={served ? 1 : 0} />
          <Counter x={200} y={250} label="distinct addresses" value={lit ? 1 : 0} />
        </Stage>
      );
    },
  },
  {
    clip: 'constant-broadcast-2',
    label: 'Every tap',
    draw: (p) => {
      const k = Math.min(TAPS.length - 1, Math.floor(ramp(p, 0.04, 0.9) * TAPS.length));
      const addr = UNIFORM.map(() => TAPS[k].address);
      return (
        <Stage data={{ beat: 2, tap: k, distinct: distinctCount(addr), answers: k + 1 }}
          headline={k < TAPS.length - 1 ? `tap ${k + 1} of ${TAPS.length}: ${tapName(k)}, still one address` : `${TAPS.length} taps, ${TAPS.length} answers, whatever the warp's width`}>
          <WarpTable addresses={addr} tableSize={M * M} hotLines={new Set([TAPS[k].address])}
            lane={() => 'served'} entry={(e) => (e === TAPS[k].address ? 'hot' : e < TAPS[k].address ? 'done' : 'idle')}
            laneLabel="one warp: 32 lanes" tableLabel="F: the 5 × 5 filter in constant memory, row by row" entryLabel={(e) => String(e)} />
          <Counter x={20} y={250} label="answers fetched" value={k + 1} of={TAPS.length} />
          <Counter x={200} y={250} label="distinct addresses this tap" value={1} />
          <RoundStrip x={20} y={310} total={TAPS.length} done={k + 1} width={640} />
        </Stage>
      );
    },
  },
  {
    clip: 'constant-broadcast-3',
    label: 'A codebook',
    draw: (p) => {
      const asked = true; // the codes are there from the start; the rounds are what unfolds
      const done = Math.floor(ramp(p, 0.25, 0.95) * ROUNDS.length + 1e-9);
      const current = done < ROUNDS.length && p >= 0.25 ? ROUNDS[done] : null;
      const servedLanes = new Set(ROUNDS.slice(0, done).flatMap((r) => r.lanes));
      const doneEntries = new Set(ROUNDS.slice(0, done).map((r) => r.address));
      const order = new Map(ROUNDS.map((r, i) => [r.address, i + 1]));
      return (
        <Stage data={{ beat: 3, distinct: DISTINCT, rounds: done }}
          headline={
            done >= ROUNDS.length
              ? `${DISTINCT} different entries: ${DISTINCT} rounds, one after another`
              : current
                ? `round ${done + 1}: entry ${current.address} for lane${current.lanes.length > 1 ? 's' : ''} ${current.lanes.join(', ')}`
                : 'each lane asks for the entry its own code names'
          }>
          <WarpTable addresses={CODES} tableSize={CODEBOOK} lines={1} hotLines={current ? new Set([current.address]) : undefined}
            lane={(i) => (servedLanes.has(i) ? 'served' : current?.lanes.includes(i) ? 'asking' : p >= 0.25 ? 'waiting' : 'asking')}
            entry={(e) => (current?.address === e ? 'hot' : doneEntries.has(e) ? 'done' : order.has(e) && asked ? 'asked' : 'idle')}
            rounds={asked ? order : undefined}
            laneLabel="one warp: 32 lanes, each with its own 4-bit weight code" tableLabel="a 16-entry codebook in constant memory" entryLabel={(e) => String(e)} />
          <Counter x={20} y={250} label="rounds answered" value={done} of={DISTINCT} />
          <Counter x={200} y={250} label="distinct addresses" value={asked ? DISTINCT : 0} />
          <RoundStrip x={20} y={310} total={DISTINCT} done={done} />
        </Stage>
      );
    },
  },
  {
    clip: 'constant-broadcast-4',
    label: 'AMD',
    draw: (p) => (
      <Stage data={{ beat: 4, 'amd-filter': amdScalarEligible(UNIFORM) ? 'yes' : 'no', 'amd-codebook': amdScalarEligible(CODES) ? 'yes' : 'no' }}
        headline="AMD: the scalar cache serves uniform reads only, and AMD gives no size">
        <g>
          <Mini x={20} y={30} addresses={UNIFORM} tableSize={M * M} title="filter tap: 1 address"
            verdict={amdScalarEligible(UNIFORM) ? 'uniform: served by the scalar cache (sL1D)' : 'not uniform'} ok={amdScalarEligible(UNIFORM)} />
        </g>
        <g opacity={ramp(p, 0.5, 0.62)}>
          <Mini x={380} y={30} addresses={CODES} tableSize={CODEBOOK} title={`codebook lookup: ${DISTINCT} addresses`}
            verdict={amdScalarEligible(CODES) ? 'uniform' : 'not uniform: the accesses are serialised'} ok={amdScalarEligible(CODES)} />
        </g>
        <g opacity={ramp(p, 0.8, 0.9)}>
          <text x={20} y={250} fontSize="12">Size of __constant__ memory on AMD: not documented.</text>
          <text x={20} y={270} fontSize="12" style={{ fill: 'var(--ink-2)' }}>totalConstMem reports the largest allocation instead (CLR source).</text>
        </g>
      </Stage>
    ),
  },
  {
    clip: 'constant-broadcast-5',
    label: 'Apple',
    draw: (p) => (
      <Stage data={{ beat: 5 }} headline="Apple: no documented rule to model, so the lab measures">
        <g>
          <rect x={20} y={24} width={680} height={196} rx="4" fill="var(--page)" stroke="var(--rule)" />
          <text x={36} y={50} fontSize="12.5" fontWeight="600">What Apple documents about the constant address space</text>
          <text x={36} y={78} fontSize="12">MSL specification: read-only buffer memory from the device memory pool.</text>
          <text x={36} y={102} fontSize="12" opacity={ramp(p, 0.15, 0.3)}>WWDC 2016 talk: &ldquo;optimized for cases with a lot of data reuse&rdquo;;</text>
          <text x={36} y={120} fontSize="12" opacity={ramp(p, 0.15, 0.3)}>statically bounded data passed by reference can be preloaded into registers.</text>
          <text x={36} y={150} fontSize="12" opacity={ramp(p, 0.4, 0.55)} style={{ fill: 'var(--ink-2)' }}>Not documented: a constant cache, its size, a rule about lanes reading one address.</text>
          <text x={36} y={190} fontSize="12" opacity={ramp(p, 0.7, 0.85)}>Lab 7.1 times constant float* against a setBytes struct on your Mac.</text>
        </g>
      </Stage>
    ),
  },
  {
    clip: 'constant-broadcast-6',
    label: 'Side by side',
    draw: (p) => {
      const g = ramp(p, 0.1, 0.5);
      return (
        <Stage data={{ beat: 6, 'filter-rounds': distinctCount(UNIFORM), 'codebook-rounds': DISTINCT }}
          headline="best case and near-worst case for the same cache">
          <Mini x={20} y={30} addresses={UNIFORM} tableSize={M * M} title="filter tap (every lane, one address)" verdict="1 round per read" ok={null} />
          <Mini x={380} y={30} addresses={CODES} tableSize={CODEBOOK} title="codebook lookup (one code per lane)" verdict={`${DISTINCT} rounds per read`} ok={null} />
          <text x={20} y={232} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>serialised reads, NVIDIA's documented rule</text>
          <rect x={20} y={242} width={Math.max(2, 20 * g)} height={16} fill="var(--const)" />
          <text x={28 + 20 * g} y={255} fontSize="11.5">filter: 1</text>
          <rect x={20} y={266} width={Math.max(2, 20 * DISTINCT * g)} height={16} fill="var(--const)" />
          <text x={28 + 20 * DISTINCT * g} y={279} fontSize="11.5">codebook: {DISTINCT}</text>
          <text x={20} y={320} fontSize="12" opacity={ramp(p, 0.6, 0.8)}>llama.cpp on CUDA and HIP: quantisation tables as static const __device__ arrays.</text>
        </Stage>
      );
    },
  },
];

export function ConstantBroadcastScene() {
  return <Scene chapter="ch07" title="One address, one warp" beats={beats} />;
}
