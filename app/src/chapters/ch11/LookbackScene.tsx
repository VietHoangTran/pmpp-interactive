import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import {
  SCENE_G,
  SCENE_R,
  flagsAt,
  phaseAt,
  rows,
  sceneRuns,
  simulate,
  type Flag,
  type LookbackRun,
  type Phase,
} from '../../sim/lookback';

// "Waiting for the block before you" (section 11.9): six blocks on a device that holds three,
// doing an inter-block scan. Six beats, one narration clip each
// (content/ch11/narration/lookback-N.txt). Every timeline, flag and count drawn comes from
// sim/lookback.ts runs with fixed seeds (sceneRuns).
export const SCENE = sceneRuns();
/** The same claimed-index single lookback with 32 blocks, all resident: the chain the narration names. */
export const SCENE_WIDE = simulate({ G: 32, R: 32, order: 'index', seed: 6, index: 'counter', scheme: 'single', progress: 'started' });

const W = 720;
const H = 340;
const LX = 8; // row labels
const TX = 150; // timeline start
const TW = 400; // timeline width
const FX = 580; // flags column
const TOP = 46;
const RH = 28;
const T_MAX = 24;
const X = (t: number) => TX + (Math.min(t, T_MAX) / T_MAX) * TW;
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));

const FILL: Record<Phase, string> = {
  local: 'var(--ink-3)',
  look: 'var(--ink)',
  spin: 'url(#lb-spin)',
  barrier: 'url(#lb-spin)',
  scan: 'var(--ink)',
  finish: 'var(--ink-2)',
  starved: 'none',
};

function Defs() {
  return (
    <defs>
      <pattern id="lb-spin" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="6" height="6" fill="var(--global-wash)" />
        <line x1="0" y1="0" x2="0" y2="6" stroke="var(--global)" strokeWidth="2" />
      </pattern>
    </defs>
  );
}

function Legend({ y }: { y: number }) {
  const items: [string, ReactNode][] = [
    ['scanning its segment', <rect key="a" width="14" height="10" fill="var(--ink-3)" opacity="0.6" />],
    ['spinning on a flag', <rect key="b" width="14" height="10" fill="url(#lb-spin)" />],
    ['looking back, adding', <rect key="c" width="14" height="10" fill="var(--ink)" />],
    ['adding and storing', <rect key="d" width="14" height="10" fill="var(--ink-2)" />],
    ['resident, not run', <rect key="e" width="14" height="10" fill="none" stroke="var(--fail)" strokeDasharray="3 2" />],
  ];
  // Five items in one row from the left edge: about 5.3 px a character at this size.
  let x = LX;
  return (
    <g>
      {items.map(([label, sw]) => {
        const g = (
          <g key={label} transform={`translate(${x} ${y})`}>
            {sw}
            <text x={18} y={9} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{label}</text>
          </g>
        );
        x += 18 + label.length * 5.3 + 22;
        return g;
      })}
    </g>
  );
}

interface StageProps {
  run: LookbackRun;
  /** Time shown, in rounds. */
  t: number;
  headline?: ReactNode;
  lines?: ReactNode[];
  /** Draw the chain of scanned sums (dependency links) up to time t. */
  chain?: boolean;
  /** Highlight the message passing between block 0 and block 1 (part 2). */
  message?: number;
  banner?: { text: string; tone: 'fail' | 'ok' } | null;
}

function Stage({ run, t, headline, lines = [], chain, message = 0, banner }: StageProps) {
  const rs = rows(run);
  const flags = flagsAt(run, Math.floor(t));
  const counter = run.config.index === 'counter';
  const resident = run.blocks.filter((b) => b.start != null && b.start <= t && (b.end == null || b.end > t)).length;
  const rowY = (hw: number) => TOP + rs.findIndex((b) => b.hw === hw) * RH;
  const rowOfIdx = (idx: number) => rs.find((b) => b.idx === idx);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Blocks of a single-pass scan waiting for each other" style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <Defs />
      <text x={LX} y={16} fontSize="12" fontWeight="600">
        {run.config.G} blocks, {run.config.R} slots; index {counter ? 'claimed from a counter' : 'from blockIdx.x'}; {run.config.scheme === 'decoupled' ? 'decoupled' : 'single'} lookback
      </text>
      <text x={TX} y={34} fontSize="10.5" style={{ fill: 'var(--ink-2)' }} data-resident={resident}>
        round {Math.floor(t)} · resident {resident} of {run.config.R}
      </text>
      <text x={FX} y={34} fontSize="10.5" fontWeight="600" style={{ fill: 'var(--global)' }}>flags (global memory)</text>
      {[0, 6, 12, 18, 24].map((r) => (
        <g key={r}>
          <line x1={X(r)} x2={X(r)} y1={TOP - 4} y2={TOP + rs.length * RH - 4} stroke="var(--rule-soft)" />
          <text x={X(r)} y={TOP + rs.length * RH + 8} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{r}</text>
        </g>
      ))}
      {rs.map((b, k) => {
        const y = TOP + k * RH;
        const now = phaseAt(b, Math.floor(t));
        return (
          <g key={b.hw} data-row={b.hw} data-idx={b.idx ?? ''} data-phase={now}>
            <text x={LX} y={y + 15} fontSize="11">
              {!counter ? `block ${b.hw}${b.start == null || b.start > t ? ' (queued)' : ''}` : b.start != null && b.start <= t ? `index ${b.idx} · blockIdx ${b.hw}` : `blockIdx ${b.hw} (queued)`}
            </text>
            {b.start == null || b.start > t ? (
              <rect x={TX} y={y + 4} width={TW} height={RH - 10} rx="2" fill="none" stroke="var(--rule)" strokeDasharray="3 3" />
            ) : null}
            {b.spans.filter((s) => s.from < t).map((s, i) => {
              const x0 = X(s.from);
              const x1 = X(Math.min(s.to, t));
              return s.phase === 'starved' ? (
                <rect key={i} x={x0} y={y + 4} width={Math.max(0.5, x1 - x0)} height={RH - 10} fill="none" stroke="var(--fail)" strokeDasharray="3 2" />
              ) : (
                <rect key={i} x={x0} y={y + 4} width={Math.max(0.5, x1 - x0)} height={RH - 10}
                  fill={FILL[s.phase]} opacity={s.phase === 'local' ? 0.6 : 1} />
              );
            })}
            {b.end != null && b.end <= t ? (
              <text x={X(b.end) + 4} y={y + 15} fontSize="10.5" style={{ fill: 'var(--ok)' }}>done</text>
            ) : null}
          </g>
        );
      })}
      {/* the flags, one per index */}
      {flags.map((f: Flag, i) => {
        const y = TOP + i * RH;
        return (
          <g key={i} data-flag={i} data-state={f}>
            <rect x={FX} y={y + 4} width={18} height={RH - 10} rx="2" fill={f === 'inclusive' ? 'var(--global)' : f === 'own' ? 'var(--global-wash)' : 'var(--cell)'} stroke="var(--global)" />
            <text x={FX + 24} y={y + 15} fontSize="10" style={{ fill: 'var(--ink-2)' }}>
              {i}: {f === 'inclusive' ? 'prefix ready' : f === 'own' ? 'own total' : '—'}
            </text>
          </g>
        );
      })}
      {/* chain links: from the block whose scanned sum was used to the block that used it */}
      {chain
        ? rs.flatMap((b) => {
            if (b.dep == null || b.idx == null) return [];
            const pub = run.events.find((e) => e.kind === 'inclusive' && e.idx === b.dep);
            const mine = run.events.find((e) => e.kind === 'inclusive' && e.idx === b.idx);
            const from = rowOfIdx(b.dep);
            if (!pub || !mine || !from || mine.t > t) return [];
            return [
              <line key={'c' + b.hw} data-link={`${b.dep}-${b.idx}`} x1={X(pub.t)} y1={rowY(from.hw) + RH - 6} x2={X(mine.t - 0.5)} y2={rowY(b.hw) + 4}
                stroke="var(--global)" strokeWidth="1.8" markerEnd="none" />,
            ];
          })
        : null}
      {message > 0 ? (
        <g opacity={message}>
          {(() => {
            const b0 = rowOfIdx(0)!;
            const b1 = rowOfIdx(1)!;
            const pub = run.events.find((e) => e.kind === 'inclusive' && e.idx === 0)!;
            return (
              <g data-message="0-1">
                <line x1={X(pub.t)} y1={rowY(b0.hw) + RH / 2} x2={FX - 4} y2={TOP + RH / 2} stroke="var(--global)" strokeDasharray="3 2" />
                <line x1={FX - 4} y1={TOP + RH / 2} x2={X(pub.t + 0.5)} y2={rowY(b1.hw) + RH / 2} stroke="var(--global)" strokeWidth="1.8" />
                <text x={X(pub.t) + 6} y={rowY(b0.hw) - 1} fontSize="10">store sum, then release the flag</text>
                <text x={X(pub.t + 1) + 6} y={rowY(b1.hw) + RH + 6} fontSize="10">acquire the flag, then read the sum</text>
              </g>
            );
          })()}
        </g>
      ) : null}
      <line x1={X(t)} x2={X(t)} y1={TOP - 8} y2={TOP + rs.length * RH - 2} stroke="var(--ink)" strokeWidth="1.2" />
      <Legend y={TOP + rs.length * RH + 16} />
      {banner ? (
        <g data-banner={banner.tone}>
          <rect x={TX} y={TOP + rs.length * RH + 34} width={TW} height={24} rx="3" fill={banner.tone === 'fail' ? 'var(--fail)' : 'var(--ok)'} />
          <text x={TX + TW / 2} y={TOP + rs.length * RH + 50} fontSize="12" textAnchor="middle" fontWeight="600" style={{ fill: 'var(--page)' }}>{banner.text}</text>
        </g>
      ) : null}
      {headline ? <text x={LX} y={H - 40} fontSize="13" fontWeight="600" data-headline>{headline}</text> : null}
      {lines.map((l, i) => (
        <text key={i} x={LX} y={H - 22 + i * 15} fontSize="11" style={{ fill: 'var(--ink-2)' }} data-line={i}>{l}</text>
      ))}
    </svg>
  );
}

const S = SCENE.single;
const firstPublish = S.events.find((e) => e.kind === 'inclusive' && e.idx === 0)!.t;
const b1 = S.blocks.find((b) => b.idx === 1)!;
const b1Read = b1.spans.find((s) => s.phase === 'look')!.from;

const beats: Beat[] = [
  {
    clip: 'lookback-1',
    label: 'Six blocks, three slots',
    draw: (p) => (
      <Stage run={S} t={ramp(p, 0.05, 0.95) * (firstPublish - 2)}
        headline={`${SCENE_G} blocks, ${SCENE_R} slots: the first ${SCENE_R} start, the rest queue`}
        lines={['Each block scans its segment; then it needs the total of every segment before it.']} />
    ),
  },
  {
    clip: 'lookback-2',
    label: 'One flag',
    draw: (p) => {
      const t = firstPublish - 2 + ramp(p, 0.05, 0.85) * (b1Read + 2 - (firstPublish - 2));
      return (
        <Stage run={S} t={t} message={t >= firstPublish ? 1 : 0}
          headline="indices 0, 1, 2 claimed in start order; index 1 spins on index 0's flag"
          lines={[`index 1 spins ${b1.wait} round${b1.wait === 1 ? '' : 's'}; it reads the sum only after the flag is set`]} />
      );
    },
  },
  {
    clip: 'lookback-3',
    label: 'The chain',
    draw: (p) => {
      const t = b1Read + 2 + ramp(p, 0.02, 0.7) * (S.time - b1Read - 2);
      return (
        <Stage run={S} t={t} chain
          headline={`a chain of ${S.criticalPath} links for ${SCENE_G} blocks`}
          lines={[p > 0.72 ? `with ${SCENE_WIDE.config.G} blocks all resident: ${SCENE_WIDE.criticalPath} links, finished after ${SCENE_WIDE.time} rounds` : `finished after ${S.time} rounds; ${S.waitTotal} rounds spent spinning in all`]} />
      );
    },
  },
  {
    clip: 'lookback-4',
    label: 'Without the counter',
    draw: (p) => {
      const D = SCENE.deadlock;
      return (
        <Stage run={D} t={ramp(p, 0.05, 0.75) * (D.time + 1)}
          banner={p > 0.78 ? { text: 'deadlock: every resident block spins, no slot can free', tone: 'fail' } : null}
          headline={`the scheduler starts blocks ${D.dispatch.slice(0, SCENE_R).join(', ')} first`}
          lines={[D.stuck.map((s) => `${s.idx} waits on ${s.on}${s.onStarted ? '' : ' (not started)'}`).join(' · ')]} />
      );
    },
  },
  {
    clip: 'lookback-5',
    label: 'Decoupled',
    draw: (p) => {
      const D = SCENE.decoupled;
      return (
        <Stage run={D} t={ramp(p, 0.03, 0.75) * D.time} chain
          headline={`spinning: ${S.waitTotal} rounds single, ${D.waitTotal} decoupled; ${D.repeated} extra additions`}
          lines={[`chain ${S.criticalPath} links single, ${D.criticalPath} decoupled; finished after ${S.time} rounds single, ${D.time} decoupled`]} />
      );
    },
  },
  {
    clip: 'lookback-6',
    label: 'Who promises progress',
    draw: (p) => {
      const M = SCENE.metal;
      return (
        <Stage run={M} t={ramp(p, 0.05, 0.6) * (M.time + 1)}
          banner={p > 0.62 ? { text: 'no progress guarantee: the waiter runs, the block it waits for does not', tone: 'fail' } : null}
          headline="CUDA documents progress for a started block; AMD and Metal do not"
          lines={['CUDA: claimed index is safe · HIP: unverified · Metal: unsafe, use separate dispatches']} />
      );
    },
  },
];

export function LookbackScene() {
  return <Scene chapter="ch11" title="Waiting for the block before you" beats={beats} />;
}
