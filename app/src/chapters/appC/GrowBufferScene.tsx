import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { apply, grow, run, type Block, type GrowStep } from '../../sim/vmmgrow';

// "Growing a buffer without copying it" (§C.9). One token sequence, 32 tokens with 4
// to a chunk, run through two strategies of src/sim/vmmgrow.ts: allocate twice as
// much and copy, and reserve once and map chunks. Every number and box drawn here
// comes from that model. Five beats, one narration clip each
// (content/appC/narration/grow-buffer-N.txt).
const TOKENS = 32;
const PER_CHUNK = 4;
const DOUBLE = grow('double', { tokens: TOKENS, bytesPerToken: 1, chunk: PER_CHUNK });
const VMM = grow('vmm', { tokens: TOKENS, bytesPerToken: 1, chunk: PER_CHUNK, reserve: TOKENS });
// The access story of beat 4, from the stepper model: map without a grant, read, grant, read.
const MAPPED = run(PER_CHUNK, [
  { op: 'reserve', size: TOKENS },
  { op: 'create', device: 0, size: PER_CHUNK },
  { op: 'map', chunk: 1, offset: 0 },
]).state;
const READ_BEFORE = apply(MAPPED, { op: 'read', device: 0, offset: 0, size: PER_CHUNK }).outcome;
const GRANTED = apply(MAPPED, { op: 'setAccess', offset: 0, size: PER_CHUNK, device: 0 }).state;
const READ_AFTER = apply(GRANTED, { op: 'read', device: 0, offset: 0, size: PER_CHUNK }).outcome;

const X0 = 40;
const TOP = DOUBLE.steps.reduce((m, s) => Math.max(m, s.to.base + s.to.size), 0); // addresses the doubling run touches
const U = (720 - X0 - 20) / TOP; // pixels per token slot
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

type Phase = 'settled' | 'both' | 'copy';

/** Maps p in [0, 1] onto tokens first..last, giving tokens in `slow` `weight` times the time. */
function pace(p: number, first: number, last: number, slow: (t: number) => boolean, weight: number): { t: number; frac: number } {
  const ts = Array.from({ length: last - first + 1 }, (_, k) => first + k);
  const w = ts.map((t) => (slow(t) ? weight : 1));
  const total = w.reduce((a, b) => a + b, 0);
  let acc = 0;
  const target = clamp(p) * total;
  for (let k = 0; k < ts.length; k++) {
    if (target < acc + w[k] || k === ts.length - 1) return { t: ts[k], frac: clamp((target - acc) / w[k]) };
    acc += w[k];
  }
  return { t: last, frac: 1 };
}
const phaseOf = (s: GrowStep, frac: number): Phase => (!s.moved ? 'settled' : frac < 0.35 ? 'both' : frac < 0.7 ? 'copy' : 'settled');

interface DState {
  t: number;
  phase: Phase;
  blocks: { b: Block; state: 'live' | 'old' | 'freed'; held: number }[];
  copied: number;
  moves: number;
  allocated: number;
  arrow?: { from: Block; to: Block };
}

/** The doubling run after token t, at a phase of t's step. */
function doubling(t: number, phase: Phase): DState {
  const s = DOUBLE.steps[t - 1];
  const freed = DOUBLE.steps.slice(0, t - 1).filter((x) => x.moved).map((x) => x.from as Block);
  const blocks: DState['blocks'] = freed.map((b) => ({ b, state: 'freed', held: 0 }));
  if (s.moved && phase !== 'settled') {
    const old = s.from as Block;
    blocks.push({ b: old, state: 'old', held: t - 1 });
    blocks.push({ b: s.to, state: 'live', held: phase === 'copy' ? t - 1 : 0 });
    return {
      t,
      phase,
      blocks,
      copied: phase === 'copy' ? s.copiedTotal : s.copiedTotal - s.copied,
      moves: phase === 'copy' ? s.pointerChanges : s.pointerChanges - 1,
      allocated: s.during,
      arrow: phase === 'copy' ? { from: old, to: s.to } : undefined,
    };
  }
  if (s.moved) blocks.push({ b: s.from as Block, state: 'freed', held: 0 });
  blocks.push({ b: s.to, state: 'live', held: t });
  return { t, phase, blocks, copied: s.copiedTotal, moves: s.pointerChanges, allocated: s.allocated };
}

function Doubling({ d, dim = false }: { d: DState; dim?: boolean }) {
  const y = 50;
  return (
    <g opacity={dim ? 0.45 : 1} data-part="double" data-copied={d.copied} data-moves={d.moves} data-allocated={d.allocated}>
      <text x={X0} y={16} fontSize="12" fontWeight="600">
        Allocate twice as much and copy
      </text>
      <line x1={X0} x2={X0 + TOP * U} y1={y + 36} y2={y + 36} stroke="var(--rule)" />
      <text x={X0 + TOP * U} y={y + 48} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>
        addresses →
      </text>
      {d.blocks.map(({ b, state, held }) => (
        <g key={b.base + ':' + state} data-part="block" data-base={b.base} data-size={b.size} data-state={state} data-held={held}>
          <rect
            x={X0 + b.base * U}
            y={y}
            width={b.size * U - 2}
            height={30}
            rx="2"
            fill={state === 'freed' ? 'none' : 'var(--cell)'}
            stroke={state === 'freed' ? 'var(--ink-3)' : 'var(--global)'}
            strokeWidth={state === 'live' ? 2 : 1.2}
            strokeDasharray={state === 'freed' ? '3 3' : undefined}
          />
          {Array.from({ length: held }, (_, k) => (
            <rect key={k} x={X0 + (b.base + k) * U + 1.5} y={y + 3} width={U - 5} height={24} fill="var(--global)" opacity={state === 'old' ? 0.5 : 1} />
          ))}
        </g>
      ))}
      {d.arrow ? (
        <path
          d={`M ${X0 + (d.arrow.from.base + d.arrow.from.size / 2) * U} ${y - 2} C ${X0 + (d.arrow.from.base + d.arrow.from.size / 2) * U} ${y - 22}, ${X0 + (d.arrow.to.base + d.arrow.to.size / 4) * U} ${y - 22}, ${X0 + (d.arrow.to.base + d.arrow.to.size / 4) * U} ${y - 4}`}
          fill="none"
          stroke="var(--global)"
          strokeWidth="2"
          markerEnd="url(#gb-arrow)"
          data-part="arrow"
        />
      ) : null}
      <text x={X0} y={y + 64} fontSize="12" data-part="double-counters">
        copied so far: {d.copied} token{d.copied === 1 ? '' : 's'} · base pointer moved: {d.moves} time{d.moves === 1 ? '' : 's'} · allocated now: {d.allocated} slots for {d.phase === 'settled' ? d.t : d.t - 1} token{(d.phase === 'settled' ? d.t : d.t - 1) === 1 ? '' : 's'}
      </text>
    </g>
  );
}

interface VState {
  reserved: number;
  mapped: number;
  held: number;
  granted: number;
  copied: number;
  moves: number;
  read?: 'crash' | 'ok';
  pool?: boolean;
}

function Mapping({ v, show = 1 }: { v: VState; show?: number }) {
  const y = 176;
  return (
    <g opacity={show} data-part="vmm" data-reserved={v.reserved} data-mapped={v.mapped} data-copied={v.copied} data-moves={v.moves} data-granted={v.granted}>
      <text x={X0} y={y - 16} fontSize="12" fontWeight="600">
        Reserve once, map chunks
      </text>
      {v.reserved ? (
        <rect x={X0} y={y} width={v.reserved * U - 2} height={30} rx="2" fill="none" stroke="var(--global)" strokeDasharray="5 3" data-part="reserved" />
      ) : null}
      {Array.from({ length: v.mapped }, (_, c) => (
        <g key={c} data-part="chunk" data-chunk={c}>
          <rect x={X0 + c * PER_CHUNK * U} y={y} width={PER_CHUNK * U - 2} height={30} rx="2" fill="var(--cell)" stroke="var(--global)" strokeWidth="2" />
          <text x={X0 + (c + 0.5) * PER_CHUNK * U} y={y + 46} fontSize="13" textAnchor="middle" style={{ fill: c < v.granted ? 'var(--ok)' : 'var(--fail)' }}>
            {c < v.granted ? '✓' : '✗'}
          </text>
        </g>
      ))}
      {Array.from({ length: v.held }, (_, k) => (
        <rect key={'t' + k} x={X0 + k * U + 1.5} y={y + 3} width={U - 5} height={24} fill="var(--global)" />
      ))}
      {v.mapped ? (
        <text x={X0 + v.reserved * U + 8} y={y + 46} fontSize="11" style={{ fill: 'var(--ink-2)' }}>
          ✓ access granted · ✗ not yet
        </text>
      ) : null}
      {v.pool ? (
        <g data-part="pool">
          <rect x={X0 + (v.reserved + 3) * U} y={y} width={PER_CHUNK * U - 2} height={30} rx="2" fill="var(--global-wash)" stroke="var(--global)" />
          <text x={X0 + (v.reserved + 3 + PER_CHUNK / 2) * U} y={y + 19} fontSize="11" textAnchor="middle">
            new chunk
          </text>
          <text x={X0 + (v.reserved + 3 + PER_CHUNK / 2) * U} y={y + 46} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
            memory, no address
          </text>
        </g>
      ) : null}
      {v.read ? (
        <text x={X0 + 2} y={y + 68} fontSize="12" fontWeight="600" data-part="read" data-status={v.read} style={{ fill: v.read === 'ok' ? 'var(--ok)' : 'var(--fail)' }}>
          {v.read === 'ok' ? 'kernel reads the first chunk: ok' : 'kernel reads the first chunk: crash, no access granted'}
        </text>
      ) : null}
      <text x={X0} y={y + 94} fontSize="12" data-part="vmm-counters">
        copied so far: {v.copied} · base pointer moved: {v.moves} times · memory behind the range: {v.mapped * PER_CHUNK} slots
      </text>
    </g>
  );
}

function Stage({ top, bottom, headline }: { top?: ReactNode; bottom?: ReactNode; headline: string }) {
  return (
    <svg viewBox="0 0 720 320" role="img" aria-label="Animation: growing a buffer by copying, and by mapping chunks into a reserved range" style={{ width: '100%', maxWidth: 720, display: 'block' }}>
      <defs>
        <marker id="gb-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--global)" />
        </marker>
      </defs>
      {top}
      <line x1={X0} x2={700} y1={134} y2={134} stroke="var(--rule)" />
      {bottom}
      <text x={X0} y={308} fontSize="13" fontWeight="600" data-part="headline">
        {headline}
      </text>
    </svg>
  );
}

const vmmAt = (t: number, granted: number, held: number): VState => {
  const s = VMM.steps[t - 1];
  return { reserved: TOKENS, mapped: s.mapped ?? 0, held, granted, copied: 0, moves: 0 };
};
const FINAL = doubling(TOKENS, 'settled');

const beats: Beat[] = [
  {
    clip: 'grow-buffer-1',
    label: 'The usual way',
    draw: (p) => {
      const firstMove = DOUBLE.steps.find((s) => s.moved) as GrowStep;
      const { t, frac } = pace(p, 1, firstMove.token, (x) => x === firstMove.token, 6);
      const ph = phaseOf(DOUBLE.steps[t - 1], frac);
      const d = doubling(t, ph);
      const headline =
        ph === 'both'
          ? `full: a new buffer of ${firstMove.to.size} slots elsewhere; both exist, ${d.allocated} slots for ${t - 1} tokens`
          : ph === 'copy'
            ? `copy all ${t - 1} tokens across`
            : DOUBLE.steps[t - 1].moved
              ? 'free the old buffer: the base pointer has moved'
              : `a buffer of ${DOUBLE.steps[0].to.size} slots, holding ${t} token${t === 1 ? '' : 's'}`;
      return <Stage top={<Doubling d={d} />} headline={headline} />;
    },
  },
  {
    clip: 'grow-buffer-2',
    label: 'Copies add up',
    draw: (p) => {
      const first = (DOUBLE.steps.find((s) => s.moved) as GrowStep).token + 1;
      const { t, frac } = pace(p, first, TOKENS, (x) => DOUBLE.steps[x - 1].moved, 6);
      const s = DOUBLE.steps[t - 1];
      const ph = phaseOf(s, frac);
      const d = doubling(t, ph);
      const headline = ph === 'copy' ? `token ${t}: copy ${s.copied} tokens into ${s.to.size} new slots` : `token ${t}: ${d.copied} tokens copied so far, pointer moved ${d.moves} times`;
      return <Stage top={<Doubling d={d} />} headline={headline} />;
    },
  },
  {
    clip: 'grow-buffer-3',
    label: 'Reserve',
    draw: (p) => (
      <Stage
        top={<Doubling d={FINAL} dim />}
        bottom={<Mapping v={{ reserved: TOKENS, mapped: 0, held: 0, granted: 0, copied: 0, moves: 0 }} show={ramp(p, 0.05, 0.35)} />}
        headline={`reserve ${TOKENS / PER_CHUNK} chunks of addresses: nothing behind them yet`}
      />
    ),
  },
  {
    clip: 'grow-buffer-4',
    label: 'Map and grant',
    draw: (p) => {
      const step = p < 0.3 ? 0 : p < 0.5 ? 1 : p < 0.75 ? 2 : 3;
      const v: VState =
        step === 0
          ? { reserved: TOKENS, mapped: 0, held: 0, granted: 0, copied: 0, moves: 0, pool: true }
          : step === 1
            ? { reserved: TOKENS, mapped: 1, held: 0, granted: 0, copied: 0, moves: 0 }
            : step === 2
              ? { reserved: TOKENS, mapped: 1, held: 0, granted: 0, copied: 0, moves: 0, read: READ_BEFORE.kind === 'crash' ? 'crash' : 'ok' }
              : { reserved: TOKENS, mapped: 1, held: 0, granted: 1, copied: 0, moves: 0, read: READ_AFTER.kind === 'ok' ? 'ok' : 'crash' };
      const headline = ['create a chunk: memory with no address yet', 'map it at the start of the range', 'mapped is not accessible: the read crashes', 'grant access: now a kernel can read it'][step];
      return <Stage top={<Doubling d={FINAL} dim />} bottom={<Mapping v={v} />} headline={headline} />;
    },
  },
  {
    clip: 'grow-buffer-5',
    label: 'Grow in place',
    draw: (p) => {
      const { t, frac } = pace(p, 1, TOKENS, (x) => !!VMM.steps[x - 1].newChunks, 2);
      const s = VMM.steps[t - 1];
      const fresh = !!s.newChunks && frac < 0.5;
      const v = vmmAt(t, (s.mapped ?? 0) - (fresh && t > 1 ? 1 : 0), fresh ? t - 1 : t);
      const headline =
        t === TOKENS && frac > 0.5
          ? `${TOKENS} tokens: copied ${DOUBLE.copiedTotal} against ${VMM.copiedTotal}; the pointer moved ${DOUBLE.pointerChanges} times against ${VMM.pointerChanges}`
          : fresh
            ? `token ${t}: map chunk ${s.mapped} right after the last, and grant access`
            : `token ${t}: ${s.mapped} chunk${s.mapped === 1 ? '' : 's'} mapped, nothing copied, same base`;
      return <Stage top={<Doubling d={FINAL} dim />} bottom={<Mapping v={v} />} headline={headline} />;
    },
  },
];

export function GrowBufferScene() {
  return <Scene chapter="appC" title="Growing a buffer without copying it" beats={beats} />;
}
