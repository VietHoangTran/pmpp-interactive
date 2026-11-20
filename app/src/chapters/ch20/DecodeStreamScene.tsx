import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { decodePhases, phaseTotals, type Phase } from '../../sim/kvcache';

// "One decode step streams the whole cache" (section 20.4). One layer of a Llama 2
// 7B-shaped model (32 heads of 128, so a 4,096-wide row; book p. 506), 16-bit, with 4,096
// tokens in the cache. Every counter comes from src/sim/kvcache.ts (decodePhases,
// phaseTotals). Six beats, one narration clip each (content/ch20/narration/decode-stream-N.txt).

export const SCENE = { n: 4096, h: 32, d: 128, p: 2 };
const D = SCENE.h * SCENE.d;
const PHASES = decodePhases(SCENE.n, SCENE.h, SCENE.d, SCENE.p);
const TOTAL = phaseTotals(PHASES);
/** Bytes per token at a few context lengths, for the last beat. */
export const GROWTH = [1024, 2048, 4096, 8192, 16384, 32768].map((n) => {
  const ph = decodePhases(n, SCENE.h, SCENE.d, SCENE.p);
  const t = phaseTotals(ph);
  return { n, bytes: t.bytes, weights: ph[0].bytesRead, ai: t.ai };
});

const W = 720;
const H = 340;
const STRIPS = 16; // each drawn strip of a cache stands for n / 16 tokens
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const mib = (b: number) => (b / 2 ** 20).toLocaleString('en-US', { maximumFractionDigits: 1, minimumFractionDigits: 1 });
const mflop = (f: number) => (f / 1e6).toLocaleString('en-US', { maximumFractionDigits: 1, minimumFractionDigits: 1 });

/** Totals after `done` whole phases plus a fraction `frac` of the next one. */
function partial(done: number, frac: number) {
  const before = phaseTotals(PHASES, done);
  const ph: Phase | undefined = PHASES[done];
  const bytes = before.bytes + (ph ? (ph.bytesRead + ph.bytesWritten) * frac : 0);
  const flops = before.flops + (ph ? ph.flops * frac : 0);
  return { bytes, flops, ai: bytes > 0 ? flops / bytes : 0 };
}

interface FrameProps {
  headline: string;
  weights?: number; // 0..1 how far the weight stream has got
  rows?: number; // 0..1 q', k', v' rows shown
  append?: number; // 0..1 the new cache row sliding in
  kStream?: number; // 0..1 how far down K
  vStream?: number; // 0..1 how far down V
  scores?: number; // 0..1 scores vector filled
  softmax?: boolean;
  out?: boolean;
  counters?: { bytes: number; flops: number; ai: number };
  children?: ReactNode;
}

function Cache({ x, label, stream, appended }: { x: number; label: string; stream?: number; appended: number }) {
  const y0 = 74;
  const hS = 12;
  const w = 86;
  const cur = stream == null ? -1 : Math.min(STRIPS - 1, Math.floor(stream * STRIPS));
  return (
    <g>
      <text x={x} y={y0 - 8} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}>{label}</text>
      {Array.from({ length: STRIPS }, (_, i) => {
        const done = stream != null && (i < cur || stream >= 1);
        const now = stream != null && stream < 1 && i === cur;
        return (
          <rect key={i} x={x} y={y0 + i * hS} width={w} height={hS - 1.5}
            fill={now ? 'var(--global)' : done ? 'var(--global-wash)' : 'var(--cell)'} stroke="var(--global)" strokeWidth="0.8" />
        );
      })}
      {/* the newest row, appended this step */}
      {appended > 0 ? (
        <rect x={x + (1 - appended) * 60} y={y0 + STRIPS * hS + 1} width={w} height={5} fill="var(--reg)" opacity={appended} data-appended="" />
      ) : null}
      <text x={x} y={y0 + STRIPS * hS + 22} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
        {SCENE.n.toLocaleString('en-US')} rows of {D.toLocaleString('en-US')}
      </text>
    </g>
  );
}

const CX = 490; // left edge of the counters panel
function Counter({ y, label, value, frac, unit }: { y: number; label: string; value: string; frac: number; unit: string }) {
  return (
    <g>
      <text x={CX} y={y} fontSize="11.5">{label}</text>
      <rect x={CX} y={y + 6} width={130} height={12} fill="var(--cell)" stroke="var(--rule)" />
      <rect x={CX} y={y + 6} width={130 * clamp(frac)} height={12} fill="var(--ink-2)" />
      <text x={CX + 138} y={y + 16} fontSize="11.5" fontWeight="600">{value} {unit}</text>
    </g>
  );
}

function Stage(f: FrameProps) {
  const c = f.counters ?? { bytes: 0, flops: 0, ai: 0 };
  const wx = 18;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="One decode step of one layer, streaming the weights and the cache"
      style={{ width: '100%', maxWidth: W, display: 'block' }}>
      <text x={12} y={24} fontSize="14.5" fontWeight="600" data-headline="">{f.headline}</text>
      <text x={W - 12} y={44} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>
        one layer, 32 heads of 128, 16-bit, {SCENE.n.toLocaleString('en-US')} cached tokens
      </text>
      {f.children ?? (
        <g data-bytes={Math.round(c.bytes)} data-flops={Math.round(c.flops)}>
          {/* the token's row and the three weight matrices */}
          <text x={wx} y={66} fontSize="12" fontWeight="600" style={{ fill: 'var(--reg)' }}>x′ (one row)</text>
          <rect x={wx} y={72} width={140} height={7} fill="var(--reg)" />
          {['W_Q', 'W_K', 'W_V'].map((name, i) => {
            const y = 96 + i * 62;
            const done = clamp(((f.weights ?? 0) - i / 3) * 3);
            return (
              <g key={name}>
                <rect x={wx} y={y} width={56} height={52} fill="var(--cell)" stroke="var(--global)" />
                <rect x={wx} y={y} width={56} height={52 * done} fill={done > 0 && done < 1 ? 'var(--global)' : 'var(--global-wash)'} opacity={done > 0 ? 1 : 0} />
                <text x={wx + 64} y={y + 20} fontSize="11.5" style={{ fill: 'var(--global)' }}>{name}</text>
                <text x={wx + 64} y={y + 35} fontSize="10" style={{ fill: 'var(--ink-2)' }}>4,096 × 4,096</text>
              </g>
            );
          })}
          {(['q′', 'k′', 'v′'] as const).map((name, i) => (
            <g key={name} opacity={clamp(((f.rows ?? 0) - i / 3) * 3)}>
              <text x={wx + 150} y={110 + i * 62} fontSize="12" fontWeight="600" style={{ fill: 'var(--reg)' }}>{name}</text>
              <rect x={wx + 150} y={116 + i * 62} width={60} height={6} fill="var(--reg)" />
            </g>
          ))}
          <Cache x={250} label="K cache" stream={f.kStream} appended={f.append ?? 0} />
          <Cache x={370} label="V cache" stream={f.vStream} appended={f.append ?? 0} />
          {/* scores and output */}
          {f.scores ? (
            <g>
              <text x={250} y={312} fontSize="11.5" style={{ fill: 'var(--reg)' }}>{f.softmax ? 'p′ = softmax(s′)' : 's′ = q′ Kᵀ'}</text>
              <rect x={250} y={318} width={206 * f.scores} height={6} fill={f.softmax ? 'var(--shared)' : 'var(--reg)'} />
              <text x={462} y={324} fontSize="10" style={{ fill: 'var(--ink-2)' }}>32 × 4,096 scores</text>
            </g>
          ) : null}
          {f.out ? (
            <g>
              <text x={wx + 150} y={292} fontSize="12" fontWeight="600" style={{ fill: 'var(--reg)' }}>o′</text>
              <rect x={wx + 150} y={298} width={60} height={6} fill="var(--reg)" />
            </g>
          ) : null}
          <Counter y={92} label="bytes read and written" value={mib(c.bytes)} unit="MiB" frac={c.bytes / TOTAL.bytes} />
          <Counter y={140} label="floating-point operations" value={mflop(c.flops)} unit="M" frac={c.flops / TOTAL.flops} />
          <text x={CX} y={200} fontSize="11.5">FLOP per byte so far</text>
          <text x={CX + 138} y={222} fontSize="15" fontWeight="650" data-ai={c.ai.toFixed(6)}>{c.bytes > 0 ? c.ai.toFixed(2) : '–'}</text>
          <text x={CX} y={252} fontSize="10" style={{ fill: 'var(--ink-2)' }}>the book’s example GPU can do</text>
          <text x={CX} y={265} fontSize="10" style={{ fill: 'var(--ink-2)' }}>about 20 FLOPs per byte it fetches</text>
          <text x={CX} y={278} fontSize="10" style={{ fill: 'var(--ink-2)' }}>(66.9 TFLOP/s, 3.35 TB/s; ch. 5 p. 94)</text>
        </g>
      )}
    </svg>
  );
}

/** The last beat: bytes per token against context length, every bar from decodePhases. */
function Growth({ p }: { p: number }) {
  const max = GROWTH[GROWTH.length - 1].bytes;
  const x0 = 60;
  const bw = 80;
  const base = 290;
  const hMax = 200;
  const shown = Math.max(1, Math.ceil(ramp(p, 0.05, 0.75) * GROWTH.length));
  return (
    <g>
      <text x={x0 - 44} y={74} fontSize="11.5">bytes one decode step of this layer moves, by cached tokens N</text>
      {GROWTH.slice(0, shown).map((g, i) => {
        const x = x0 + i * (bw + 20);
        const hw = (g.weights / max) * hMax;
        const hc = ((g.bytes - g.weights) / max) * hMax;
        return (
          <g key={g.n} data-n={g.n} data-bytes={g.bytes} data-ai={g.ai.toFixed(3)}>
            <rect x={x} y={base - hw} width={bw} height={hw} fill="var(--global-wash)" stroke="var(--global)" />
            <rect x={x} y={base - hw - hc} width={bw} height={hc} fill="var(--global)" />
            <text x={x + bw / 2} y={base - hw - hc - 20} fontSize="11" textAnchor="middle" fontWeight="600">{mib(g.bytes)} MiB</text>
            <text x={x + bw / 2} y={base - hw - hc - 6} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{g.ai.toFixed(3)} FLOP/B</text>
            <text x={x + bw / 2} y={base + 16} fontSize="11" textAnchor="middle">N = {g.n.toLocaleString('en-US')}</text>
          </g>
        );
      })}
      <rect x={x0 - 44} y={316} width={12} height={10} fill="var(--global)" />
      <text x={x0 - 28} y={325} fontSize="10.5">K and V caches</text>
      <rect x={x0 + 70} y={316} width={12} height={10} fill="var(--global-wash)" stroke="var(--global)" />
      <text x={x0 + 86} y={325} fontSize="10.5">projection weights (the same at every N)</text>
    </g>
  );
}

const beats: Beat[] = [
  {
    clip: 'decode-stream-1',
    label: 'Projections',
    draw: (p) => {
      const f = ramp(p, 0.08, 0.85);
      return <Stage headline="Three GEMVs: every weight read once to make one row" weights={f} rows={ramp(p, 0.3, 0.95)} counters={partial(0, f)} />;
    },
  },
  {
    clip: 'decode-stream-2',
    label: 'Append',
    draw: (p) => (
      <Stage headline="k′ and v′ become row N of this layer’s caches" weights={1} rows={1} append={ramp(p, 0.1, 0.6)} counters={partial(1, ramp(p, 0.1, 0.6))} />
    ),
  },
  {
    clip: 'decode-stream-3',
    label: 'q′ Kᵀ',
    draw: (p) => {
      const f = ramp(p, 0.05, 0.92);
      return <Stage headline="q′ meets every cached key: each row read once, used once" weights={1} rows={1} append={1} kStream={f} scores={f} counters={partial(2, f)} />;
    },
  },
  {
    clip: 'decode-stream-4',
    label: 'Softmax',
    draw: () => <Stage headline="A softmax over one short row of scores per head" weights={1} rows={1} append={1} kStream={1} scores={1} softmax counters={partial(4, 0)} />,
  },
  {
    clip: 'decode-stream-5',
    label: 'p′ V',
    draw: (p) => {
      const f = ramp(p, 0.05, 0.9);
      return (
        <Stage headline="The weighted sum streams every value row once more" weights={1} rows={1} append={1} kStream={1} vStream={f} scores={1} softmax
          out={p > 0.9} counters={partial(4, f)} />
      );
    },
  },
  {
    clip: 'decode-stream-6',
    label: 'Next token',
    draw: (p) => (
      <Stage headline="Next token: all of it again, one row longer">
        <Growth p={p} />
      </Stage>
    ),
  },
];

export function DecodeStreamScene() {
  return <Scene chapter="ch20" title="One decode step streams the whole cache" beats={beats} />;
}
