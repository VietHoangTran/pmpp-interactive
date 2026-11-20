import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { SCENE, sceneRuns, type ChainFrame, type ChainPhase, type ChainRun } from '../../sim/inplace';

// "The scan chain orders the blocks" (section 12.7). Twelve keys of Fig. 12.10's pattern,
// three blocks of four, two resident slots, the hardware starting blockIdx 2, 1, 0. Six
// beats, one narration clip each (content/ch12/narration/scan-chain-N.txt). Every arrow,
// number, phase and outcome drawn comes from sim/inplace.ts (sceneRuns), pinned in
// inplace.test.ts ("chainTimeline: the scene's runs").
const RUNS = sceneRuns();
const OV = RUNS.overlap;
const N = SCENE.keep.length; // 12
const K = SCENE.perBlock; // 4
const NB = OV.blocks; // 3

const W = 720;
const H = 372;
const AX = 40; // array origin
const CW = 34; // array cell width
const ROW_BEFORE = 46;
const ROW_AFTER = 136;
const GX = 150; // Gantt origin
const GY = 196;
const GC = 30; // Gantt column width
const GR = 34; // Gantt row height

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const cx = (slot: number) => AX + slot * CW + CW / 2;

const PHASE_SHORT: Record<ChainPhase, string> = {
  queued: '',
  index: 'take',
  read: 'read',
  scan: 'count',
  wait: 'wait',
  publish: 'flag',
  write: 'write',
  done: '',
};

/** A row of the array: labels in each slot, with block ranges above. */
function ArrayRow({ y, values, label, keepMarks, read, brackets, numbered, labelBelow, dimFrom }: {
  y: number; values: (number | null)[]; label: string; keepMarks?: boolean; read?: boolean[]; brackets?: boolean; numbered?: boolean;
  labelBelow?: boolean; dimFrom?: number;
}) {
  return (
    <g>
      <text x={AX} y={labelBelow ? y + 44 : y - (brackets ? 24 : 6)} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}>{label}</text>
      {brackets
        ? Array.from({ length: NB }, (_, b) => (
            <g key={b}>
              <line x1={AX + b * K * CW + 2} x2={AX + (b + 1) * K * CW - 4} y1={y - 6} y2={y - 6} stroke="var(--ink-2)" />
              <text x={AX + b * K * CW + (K * CW) / 2} y={y - 10} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
                {numbered ? `number ${b}'s range` : `block ${b}`}
              </text>
            </g>
          ))
        : null}
      {values.map((v, s) => {
        const kept = keepMarks && v != null && SCENE.keep[v];
        return (
          <g key={s} data-slot={s} data-value={v == null ? '' : v} opacity={dimFrom != null && s >= dimFrom ? 0.4 : 1}>
            <rect x={AX + s * CW} y={y} width={CW - 3} height={26} rx="2"
              fill={kept ? 'var(--global-wash)' : 'var(--cell)'} stroke={kept ? 'var(--global)' : 'var(--rule)'} />
            <text x={cx(s) - 1} y={y + 17} fontSize="11" textAnchor="middle" style={{ fill: v == null ? 'var(--ink-3)' : 'var(--ink)' }}>
              {v == null ? '·' : `k${v}`}
            </text>
            {read?.[s] ? <circle cx={cx(s) - 1} cy={y + 32} r="2.6" style={{ fill: 'var(--ink-2)' }} /> : null}
          </g>
        );
      })}
    </g>
  );
}

/** Beat 1: every write of the in-place filter, red where it lands in an earlier block's range. */
function Arrows({ show }: { show: number }) {
  return (
    <g>
      <defs>
        <marker id="sc-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--ink-2)' }} />
        </marker>
        <marker id="sc-arrow-bad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--fail)' }} />
        </marker>
      </defs>
      {OV.writes.map((w, k) => {
        const on = clamp(show * OV.writes.length - k);
        if (on <= 0) return null;
        const bad = w.kind === 'earlier';
        return (
          <line key={w.key} data-arrow={w.key} data-kind={w.kind} opacity={on}
            x1={cx(w.from) - 1} y1={ROW_BEFORE + 28} x2={cx(w.to) - 1} y2={ROW_AFTER - 4}
            stroke={bad ? 'var(--fail)' : 'var(--ink-2)'} strokeWidth={bad ? 2.2 : 1.4}
            markerEnd={bad ? 'url(#sc-arrow-bad)' : 'url(#sc-arrow)'} />
        );
      })}
    </g>
  );
}

/** The blocks over time: one row per blockIdx, one column per step, up to frame `upto`. */
function Gantt({ run, upto, x0 = GX, y0 = GY, col = GC, title, showFlags = true, stuckNote = true, labels = true, compact = false }: {
  run: ChainRun; upto: number; x0?: number; y0?: number; col?: number; title: string; showFlags?: boolean; stuckNote?: boolean; labels?: boolean; compact?: boolean;
}) {
  const frames = run.frames.slice(1, upto + 1);
  const cur: ChainFrame = run.frames[upto];
  const last = run.frames[run.frames.length - 1];
  const stuck = run.outcome === 'starved' && upto === run.frames.length - 1;
  return (
    <g data-gantt={title}>
      <text x={x0 - 110} y={y0 - 12} fontSize="11.5" fontWeight="600">{title}</text>
      {Array.from({ length: NB }, (_, hw) => {
        const b = cur.blocks[hw];
        const y = y0 + hw * GR;
        return (
          <g key={hw} data-hw={hw} data-num={b.num ?? ''} data-phase={b.phase}>
            <text x={x0 - 110} y={y + 17} fontSize="11">blockIdx {hw}</text>
            <text x={x0 - 44} y={y + 17} fontSize="11" fontWeight="600" style={{ fill: b.num == null ? 'var(--ink-3)' : 'var(--reg)' }}>
              {b.num == null ? '·' : `#${b.num}`}
            </text>
            {frames.map((f, k) => {
              const ph = f.blocks[hw].phase;
              if (ph === 'queued' || ph === 'done') return null;
              const isWait = ph === 'wait';
              const isWrite = ph === 'write';
              return (
                <g key={k}>
                  <rect x={x0 + k * col} y={y + 2} width={col - 3} height={GR - 8} rx="2"
                    fill={isWait ? 'var(--fail-wash)' : isWrite ? 'var(--global-wash)' : 'var(--reg-wash)'}
                    stroke={isWait ? 'var(--fail)' : isWrite ? 'var(--global)' : 'var(--reg)'} strokeWidth="0.8" />
                  {labels ? <text x={x0 + k * col + (col - 3) / 2} y={y + 18} fontSize="9.5" textAnchor="middle">{PHASE_SHORT[ph]}</text> : null}
                </g>
              );
            })}
            {stuck && stuckNote && b.phase === 'wait' ? (
              <text x={x0 + frames.length * col + 4} y={y + 18} fontSize="10.5" style={{ fill: 'var(--fail)' }}>spins on #{b.waitsOn}</text>
            ) : null}
            {stuck && stuckNote && b.phase === 'queued' ? (
              <text x={x0 + frames.length * col + 4} y={y + 18} fontSize="10.5" style={{ fill: 'var(--fail)' }}>no slot: never starts</text>
            ) : null}
          </g>
        );
      })}
      {showFlags
        ? cur.flagAt.map((t, num) => {
            if (t == null || num + 1 >= NB) return null;
            const t2 = cur.flagAt[num + 1];
            const from = last.blocks.findIndex((b) => b.num === num);
            const to = last.blocks.findIndex((b) => b.num === num + 1);
            if (t2 == null || from < 0 || to < 0) return null;
            return (
              <line key={num} x1={x0 + (t - 1) * col + col / 2} y1={y0 + from * GR + (from < to ? GR - 6 : 2)}
                x2={x0 + (t2 - 1) * col + col / 2 - 2} y2={y0 + to * GR + (from < to ? 4 : GR - 6)}
                stroke="var(--ink)" strokeWidth="1.6" markerEnd="url(#sc-flag)" />
            );
          })
        : null}
      <text x={x0} y={y0 + NB * GR + 12} fontSize="10" style={{ fill: 'var(--ink-2)' }}>
        step {cur.step}
        {run === RUNS.dynamic ? ` · counter ${cur.counter}` : ''}
        {upto === run.frames.length - 1 ? (run.outcome === 'finished' ? ' · finished' : compact ? ' · starved' : ' · starved: nothing can change any more') : ''}
      </text>
    </g>
  );
}

function Slots({ frame, x = 520, y = 196 }: { frame: ChainFrame; x?: number; y?: number }) {
  return (
    <g data-part="slots">
      <text x={x} y={y - 12} fontSize="11.5" fontWeight="600">resident slots ({SCENE.slots})</text>
      {Array.from({ length: SCENE.slots }, (_, s) => {
        const b = frame.blocks.find((x) => x.slot === s);
        return (
          <g key={s} data-slot-holder={b ? b.hw : ''}>
            <rect x={x} y={y + s * 34} width={170} height={26} rx="3" fill="var(--cell)" stroke="var(--ink-2)" />
            <text x={x + 8} y={y + s * 34 + 17} fontSize="11">
              {b ? `blockIdx ${b.hw} as #${b.num}` : 'free'}
            </text>
          </g>
        );
      })}
    </g>
  );
}

function Stage({ children, headline }: { children: ReactNode; headline?: ReactNode }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Animation of three blocks filtering one array in place" style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <defs>
        <marker id="sc-flag" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" style={{ fill: 'var(--ink)' }} />
        </marker>
      </defs>
      {children}
      {headline ? <text x={AX} y={H - 8} fontSize="13" fontWeight="600" data-headline="">{headline}</text> : null}
    </svg>
  );
}

const frameAt = (run: ChainRun, a: number, b: number, p: number) => Math.min(b, a + Math.floor(p * (b - a + 1)));
const readBy = (f: ChainFrame) => {
  const out = Array(N).fill(false);
  for (const b of f.blocks)
    if (b.num != null && b.phase !== 'index' && b.phase !== 'queued')
      for (let s = b.num * K; s < (b.num + 1) * K; s++) out[s] = true;
  return out;
};

function Running({ run, a, b, p, headline }: { run: ChainRun; a: number; b: number; p: number; headline: (f: ChainFrame, k: number) => ReactNode }) {
  const k = frameAt(run, a, b, p);
  const f = run.frames[k];
  return (
    <Stage headline={headline(f, k)}>
      <ArrayRow y={ROW_BEFORE} values={f.array} label="the array, as it is now (a dot: its key has been read)" keepMarks brackets numbered read={readBy(f)}
        dimFrom={k === run.frames.length - 1 && run.outcome === 'finished' ? OV.count : undefined} />
      {k === run.frames.length - 1 && run.outcome === 'finished' ? (
        <text x={AX + OV.count * CW + 2} y={ROW_BEFORE + 46} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>left over: past the output</text>
      ) : null}
      <Gantt run={run} upto={k} title={run === RUNS.dynamic ? 'numbers from a counter' : 'numbers from blockIdx'} />
      <Slots frame={f} />
      <text x={AX} y={118} fontSize="11" data-events="" style={{ fill: 'var(--ink-2)' }}>
        {f.events.join('; ')}
      </text>
    </Stage>
  );
}

const D = RUNS.dynamic;
const B = RUNS.blockIdx;
const lastD = D.frames.length - 1;
const lastB = B.frames.length - 1;
const badWrites = OV.writes.filter((w) => w.kind === 'earlier').length;

const beats: Beat[] = [
  {
    clip: 'scan-chain-1',
    label: 'Who overwrites whom',
    draw: (p) => (
      <Stage headline={p > 0.75 ? `${badWrites} of ${OV.writes.length} writes land where an earlier block still has to read` : 'one array, three blocks, filtered in place'}>
        <ArrayRow y={ROW_BEFORE} values={OV.before} label="the array before" keepMarks brackets />
        <ArrayRow y={ROW_AFTER} values={OV.after.map((v, s) => (p > 0.6 && s < OV.count ? v : null))} label="the same array after" labelBelow />
        <Arrows show={ramp(p, 0.08, 0.6)} />
        <g opacity={ramp(p, 0.7, 0.9)} data-part="red-list">
          {OV.writes.filter((w) => w.kind === 'earlier').map((w, k) => (
            <text key={w.key} x={AX} y={212 + k * 20} fontSize="11.5">
              k{w.key} (block {w.writer}) → slot {w.to}, over k{w.victim}: block {w.reader} must read k{w.victim} first
            </text>
          ))}
        </g>
        <g opacity={ramp(p, 0.62, 0.8)}>
          <text x={AX + N * CW + 14} y={ROW_BEFORE + 60} fontSize="11.5" style={{ fill: 'var(--fail)' }} data-bad={badWrites}>red: lands in an</text>
          <text x={AX + N * CW + 14} y={ROW_BEFORE + 75} fontSize="11.5" style={{ fill: 'var(--fail)' }}>earlier block's range</text>
        </g>
      </Stage>
    ),
  },
  {
    clip: 'scan-chain-2',
    label: 'Numbers from a counter',
    draw: (p) => (
      <Running run={D} a={0} b={1} p={p}
        headline={(f) => (f.step === 0 ? 'nothing has started' : 'the first block to start gets number 0, whatever its blockIdx')} />
    ),
  },
  {
    clip: 'scan-chain-3',
    label: 'Publish and wait',
    draw: (p) => (
      <Running run={D} a={2} b={4} p={p}
        headline={(f) => (f.step < 4 ? 'both read their keys and count the kept ones' : '#0 publishes with a release; #1 waits with an acquire')} />
    ),
  },
  {
    clip: 'scan-chain-4',
    label: 'Write after the flag',
    draw: (p) => (
      <Running run={D} a={5} b={lastD} p={p}
        headline={(f, k) => (k < lastD ? 'each block writes only after the flag of the block before it' : D.correct && D.hazards === 0 ? 'finished: the stable filter, and no key overwritten before it was read' : 'finished')} />
    ),
  },
  {
    clip: 'scan-chain-5',
    label: 'Numbers from blockIdx',
    draw: (p) => (
      <Running run={B} a={0} b={lastB} p={p}
        headline={(f, k) => (k < lastB ? 'same start order, numbers from blockIdx' : 'starved: every resident block waits on one that cannot start')} />
    ),
  },
  {
    clip: 'scan-chain-6',
    label: 'Each backend',
    draw: (p) => (
      <Stage headline="the wait is safe only where a started block is promised to finish">
        <Gantt run={D} upto={lastD} x0={150} y0={40} col={20} title="numbers from a counter" labels={false} compact />
        <Gantt run={B} upto={lastB} x0={500} y0={40} col={20} title="numbers from blockIdx" labels={false} showFlags={false} compact />
        <g opacity={ramp(p, 0.1, 0.3)} data-backend="cuda">
          <text x={AX} y={196} fontSize="12" fontWeight="600">CUDA</text>
          <text x={AX + 70} y={196} fontSize="11.5">the counter makes the wait legal: started blocks are promised progress</text>
        </g>
        <g opacity={ramp(p, 0.35, 0.55)} data-backend="hip">
          <text x={AX} y={236} fontSize="12" fontWeight="600">HIP</text>
          <text x={AX + 70} y={236} fontSize="11.5">the same code; unverified, because AMD documents no such promise</text>
        </g>
        <g opacity={ramp(p, 0.6, 0.8)} data-backend="metal">
          <text x={AX} y={276} fontSize="12" fontWeight="600">Metal</text>
          <text x={AX + 70} y={276} fontSize="11.5">one threadgroup walks its range in order, with threadgroup barriers only</text>
        </g>
      </Stage>
    ),
  },
];

export function ScanChainScene() {
  return <Scene chapter="ch12" title="The scan chain orders the blocks" beats={beats} />;
}
