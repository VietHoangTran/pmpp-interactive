import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { M8_DEFAULT, precisionFor, relativeStep, stepBStar, type Padding } from '../../sim/batching';

// "One decode step as the batch grows" (section 22.5). Every number drawn comes from
// src/sim/batching.ts: lab 22.2's decode step (M8_DEFAULT shape), float32, at the book's
// example threshold of 20 FLOP/B (ch. 5 p. 94). Time is in relative units, "passes over
// the weights": one unit is the time to stream every layer's weight and the LM head's once
// at the device's bandwidth, so no device timing appears. Six beats, one narration clip
// each (content/ch22/narration/decode-step-N.txt).

export const SCENE_T = 20; // the book's example threshold, FLOP/B
const SHAPE = M8_DEFAULT;
const PREC = precisionFor(4);
const rel = (B: number, pad: Padding = 1) => relativeStep(SHAPE, B, PREC, pad, SCENE_T);
/** The step's B* at the book's threshold: where the GEMMs' arithmetic overtakes their traffic. */
export const SCENE_BSTAR = stepBStar(SHAPE, PREC, { peak: SCENE_T, bw: 1 })!;

const W = 720;
const H = 340;
const X0 = 168; // left edge of the timeline
const X1 = 700;
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const r3 = (v: number) => Math.round(v * 1000) / 1000;
const f2 = (v: number) => (v >= 10 ? v.toFixed(1) : v.toFixed(2));

function Frame({ children, batch, headline }: { children: ReactNode; batch?: number; headline: string }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="One decode step drawn against time" style={{ width: '100%', maxWidth: W, display: 'block' }}
      data-batch={batch ?? ''}>
      <text x={12} y={24} fontSize="15" fontWeight="600" data-headline="">{headline}</text>
      <text x={12} y={42} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
        time in passes over the weights, at the book's example threshold of {SCENE_T} FLOP/B
      </text>
      {children}
    </svg>
  );
}

/**
 * One step on a timeline: the weight stream (global memory), the arithmetic lane (busy a
 * fraction of the GEMM time), the per-row band after the GEMMs, the step's end, a time
 * cursor at `cursor` (in units; Infinity for none), and the tokens out once it has passed.
 */
function StepTimeline({ B, maxUnits, cursor, showRow = true }: { B: number; maxUnits: number; cursor: number; showRow?: boolean }) {
  const s = rel(B);
  const px = (X1 - X0) / maxUnits;
  const x = (u: number) => X0 + Math.min(u, maxUnits) * px;
  const at = Math.min(cursor, s.step);
  const busy = Math.min(1, s.compute / s.gemm); // fraction of the GEMM time the arithmetic units work
  const cells = 36;
  const cw = (s.gemm * px) / cells;
  const done = cursor >= s.step;
  const shown = Math.min(B, 64);
  const ticks: number[] = [];
  for (let u = 0; u <= maxUnits + 1e-9; u += maxUnits > 4 ? 1 : 0.5) ticks.push(u);
  return (
    <g data-mem={r3(s.memory)} data-comp={r3(s.compute)} data-row={r3(s.row)} data-step={r3(s.step)} data-gemm={r3(s.gemm)}>
      {ticks.map((u) => (
        <g key={u}>
          <line x1={x(u)} x2={x(u)} y1={56} y2={196} stroke="var(--rule-soft)" />
          <text x={x(u)} y={210} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{u}</text>
        </g>
      ))}
      {/* weights streaming in */}
      <text x={12} y={78} fontSize="11.5">weights from memory</text>
      <rect x={X0} y={64} width={s.memory * px} height={20} fill="var(--global-wash)" stroke="var(--global)" />
      <rect x={X0} y={64} width={Math.min(at, s.memory) * px} height={20} fill="var(--global)" />
      {/* the arithmetic lane: busy for a fraction `busy` of each slice of the GEMM time */}
      <text x={12} y={118} fontSize="11.5">arithmetic, {B} {B === 1 ? 'row' : 'rows'}</text>
      <rect x={X0} y={104} width={s.gemm * px} height={20} fill="none" stroke="var(--reg)" />
      {Array.from({ length: cells }, (_, i) => {
        const start = (i / cells) * s.gemm;
        if (start > at) return null;
        return <rect key={i} x={X0 + i * cw} y={104} width={Math.max(0.6, cw * busy)} height={20} fill="var(--reg)" />;
      })}
      <text x={X0} y={138} fontSize="10.5" style={{ fill: 'var(--reg)' }} data-busy={r3(busy)}>
        busy {busy >= 0.995 ? 'all the time' : `${Math.round(busy * 100)}% of the time`}
      </text>
      {/* the per-row band after the GEMMs */}
      {showRow ? (
        <g>
          <text x={12} y={158} fontSize="11.5">per-row work</text>
          <text x={12} y={171} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>bias, residual, softmax, argmax</text>
          <rect x={x(s.gemm)} y={144} width={Math.max(1, s.row * px)} height={20} fill="var(--ink-3)" opacity={at > s.gemm ? 1 : 0.35} />
          <text x={x(s.gemm) + s.row * px + 6} y={158} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>{f2(s.row)}</text>
        </g>
      ) : null}
      {/* end of the step and the time cursor */}
      <line x1={x(s.step)} x2={x(s.step)} y1={52} y2={196} stroke="var(--ink)" strokeDasharray="4 3" />
      <text x={Math.min(x(s.step) + 4, X1 - 4)} y={192} fontSize="11" textAnchor={x(s.step) > X1 - 90 ? 'end' : 'start'} data-steplen={r3(s.step)}>
        step: {f2(s.step)}
      </text>
      {isFinite(cursor) && !done ? <line x1={x(at)} x2={x(at)} y1={56} y2={172} stroke="var(--ink)" strokeWidth="1.6" /> : null}
      {/* tokens out */}
      <text x={12} y={244} fontSize="11.5">tokens out</text>
      <g data-tokens={done ? B : 0}>
        {Array.from({ length: shown }, (_, i) => (
          <rect key={i} x={X0 + (i % 32) * 16} y={232 + Math.floor(i / 32) * 16} width={12} height={12} rx={2}
            fill={done ? 'var(--ink)' : 'none'} stroke="var(--ink-3)" />
        ))}
        {B > shown ? <text x={X0 + 32 * 16 + 4} y={258} fontSize="11">+ {B - shown}</text> : null}
      </g>
      <text x={12} y={300} fontSize="12" data-rates="">
        <tspan>tokens per pass: {f2(s.tokensPerUnit)}</tspan>
        <tspan dx={24}>each user waits {f2(s.step)} {s.step < 1.995 ? 'pass' : 'passes'} per token</tspan>
      </text>
    </g>
  );
}

// ---------------------------------------------------------------- log-log curves
const BATCHES = Array.from({ length: 1024 }, (_, i) => i + 1);
interface Curve {
  key: string;
  label: string;
  values: number[]; // one per batch in BATCHES
  dash?: string;
  width?: number;
  tone?: string;
}

function LogLog({ x, y, w, h, title, curves, upTo, knee, yLabel }: {
  x: number; y: number; w: number; h: number; title: string; curves: Curve[]; upTo: number; knee?: number; yLabel: string;
}) {
  const all = curves.flatMap((c) => c.values);
  const lo = Math.min(...all) / 1.3;
  const hi = Math.max(...all) * 1.3;
  const X = (b: number) => x + (Math.log2(b) / 10) * w;
  const Y = (v: number) => y + h - ((Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo))) * h;
  const n = Math.max(1, Math.min(1024, Math.round(upTo)));
  const decades: number[] = [];
  for (let e = Math.ceil(Math.log10(lo)); e <= Math.floor(Math.log10(hi)); e++) decades.push(10 ** e);
  return (
    <g data-plot={title}>
      <text x={x} y={y - 10} fontSize="12" fontWeight="600">{title}</text>
      <rect x={x} y={y} width={w} height={h} fill="none" stroke="var(--rule)" />
      {[1, 4, 16, 64, 256, 1024].map((b) => (
        <g key={b}>
          <line x1={X(b)} x2={X(b)} y1={y} y2={y + h} stroke="var(--rule-soft)" />
          <text x={X(b)} y={y + h + 14} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{b}</text>
        </g>
      ))}
      {decades.map((v) => (
        <g key={v}>
          <line x1={x} x2={x + w} y1={Y(v)} y2={Y(v)} stroke="var(--rule-soft)" />
          <text x={x - 5} y={Y(v) + 3} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-3)' }}>{v >= 1000 ? `${v / 1000}k` : v}</text>
        </g>
      ))}
      <text x={x + w / 2} y={y + h + 28} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>batch (log scale)</text>
      <text x={x - 30} y={y + h / 2} fontSize="10.5" textAnchor="middle" transform={`rotate(-90 ${x - 30} ${y + h / 2})`} style={{ fill: 'var(--ink-2)' }}>{yLabel}</text>
      {knee ? (
        <g data-knee={knee.toFixed(1)}>
          <line x1={X(knee)} x2={X(knee)} y1={y} y2={y + h} stroke="var(--ink)" strokeDasharray="3 3" />
          <text x={X(knee) + 4} y={y + h - 8} fontSize="10.5">knee, B* = {knee.toFixed(1)}</text>
        </g>
      ) : null}
      {curves.map((c) => {
        const pts = BATCHES.slice(0, n).map((b) => `${X(b).toFixed(1)},${Y(c.values[b - 1]).toFixed(1)}`).join(' ');
        const last = c.values[n - 1];
        return (
          <g key={c.key} data-curve={c.key} data-upto={n} data-first={r3(c.values[0])} data-v128={r3(c.values[127])} data-last={r3(last)}>
            <polyline points={pts} fill="none" stroke={c.tone ?? 'var(--ink)'} strokeWidth={c.width ?? 2} strokeDasharray={c.dash} />
          </g>
        );
      })}
    </g>
  );
}

const STEP = BATCHES.map((b) => rel(b).step);
const TOKENS = BATCHES.map((b) => rel(b).tokensPerUnit);
const STEP16 = BATCHES.map((b) => rel(b, 16).step);
const STEP128 = BATCHES.map((b) => rel(b, 128).step);
const PADDED: Curve[] = [
  { key: 'pad1', label: 'no padding', values: STEP, tone: 'var(--ink)' },
  { key: 'pad16', label: '16-row tile (M2)', values: STEP16, dash: '6 3', tone: 'var(--ink-2)' },
  { key: 'pad128', label: '128-row tile (M3)', values: STEP128, dash: '2 3', width: 2.6, tone: 'var(--ink-2)' },
];
/** Batches beat 3 and beat 4 step through. */
export const DOUBLING = [1, 2, 4, 8, 16, 32];
export const PAST = [32, 64, 128, 256];
const pick = <T,>(xs: T[], p: number) => xs[Math.min(xs.length - 1, Math.floor(p * xs.length))];

const beats: Beat[] = [
  {
    clip: 'decode-step-1',
    label: 'Batch 1',
    draw: (p) => (
      <Frame batch={1} headline="batch 1: every weight read once, one token out">
        <StepTimeline B={1} maxUnits={1.6} cursor={ramp(p, 0.08, 0.85) * rel(1).step * 1.0001} showRow={false} />
      </Frame>
    ),
  },
  {
    clip: 'decode-step-2',
    label: 'Batch 8',
    draw: (p) => (
      <Frame batch={8} headline="batch 8: the same weight stream, 8 rows ride on it">
        <StepTimeline B={8} maxUnits={1.6} cursor={ramp(p, 0.08, 0.8) * rel(8).step * 1.0001} showRow={false} />
      </Frame>
    ),
  },
  {
    clip: 'decode-step-3',
    label: 'Per-row work',
    draw: (p) => {
      const B = pick(DOUBLING, ramp(p, 0.05, 0.95));
      return (
        <Frame batch={B} headline={`batch ${B}: the per-row band grows from the start`}>
          <StepTimeline B={B} maxUnits={1.6} cursor={Infinity} />
        </Frame>
      );
    },
  },
  {
    clip: 'decode-step-4',
    label: 'The knee',
    draw: (p) => {
      const B = pick(PAST, ramp(p, 0.05, 0.9));
      const past = B > SCENE_BSTAR;
      return (
        <Frame batch={B} headline={past ? `batch ${B}: past the knee (B* = ${SCENE_BSTAR.toFixed(1)}), arithmetic outlasts the stream` : `batch ${B}: still under the knee`}>
          <StepTimeline B={B} maxUnits={Math.max(1.6, rel(256).step * 1.08)} cursor={Infinity} />
        </Frame>
      );
    },
  },
  {
    clip: 'decode-step-5',
    label: 'Two curves',
    draw: (p) => {
      const upTo = 2 ** (10 * ramp(p, 0.05, 0.75));
      return (
        <Frame headline="the same model, two curves">
          <LogLog x={70} y={84} w={260} h={180} title="tokens per pass" yLabel="tokens per pass" upTo={upTo} knee={SCENE_BSTAR}
            curves={[{ key: 'tokens', label: 'tokens per pass', values: TOKENS }]} />
          <LogLog x={430} y={84} w={260} h={180} title="each user's time per token" yLabel="passes" upTo={upTo} knee={SCENE_BSTAR}
            curves={[{ key: 'perToken', label: 'passes per token', values: STEP }]} />
          <text x={12} y={326} fontSize="12" opacity={ramp(p, 0.7, 0.85)}>throughput for the group, against latency for each user</text>
        </Frame>
      );
    },
  },
  {
    clip: 'decode-step-6',
    label: 'Your device',
    draw: (p) => {
      const upTo = 2 ** (10 * ramp(p, 0.05, 0.7));
      return (
        <Frame headline="tile padding makes flat stretches of its own">
          <LogLog x={90} y={84} w={400} h={200} title="time per step" yLabel="passes over the weights" upTo={upTo}
            curves={PADDED} />
          <g data-legend="">
            {PADDED.map((c, i) => (
              <g key={c.key}>
                <line x1={520} x2={556} y1={92 + i * 20} y2={92 + i * 20} stroke={c.tone} strokeWidth={c.width ?? 2} strokeDasharray={c.dash} />
                <text x={564} y={96 + i * 20} fontSize="11">{c.label}</text>
              </g>
            ))}
          </g>
          <g opacity={ramp(p, 0.55, 0.7)}>
            <text x={520} y={172} fontSize="11.5" fontWeight="600">a 128-row tile computes</text>
            <text x={520} y={188} fontSize="11">128 rows whatever the batch,</text>
            <text x={520} y={203} fontSize="11">so its line is flat to 128 for</text>
            <text x={520} y={218} fontSize="11">a reason unrelated to bandwidth</text>
            <text x={520} y={246} fontSize="11.5" fontWeight="600">measure yours: lab 22.2</text>
          </g>
        </Frame>
      );
    },
  },
];

export function DecodeStepScene() {
  return <Scene chapter="ch22" title="One decode step as the batch grows" beats={beats} />;
}
