import type { ReactNode } from 'react';
import { Scene, type Beat } from '../../components/Scene';
import { JOB_LABEL, SYNTHETIC, stepLaunches, totals, type Launch } from '../../sim/llamastep';

// "One token through the step" (section X3.7): the launches of one decode step of the module's
// synthetic model, as lab X3.2 runs them, with the supplying lab lit at each launch. Seven beats,
// one clip each (content/x3/narration/llama-step-N.txt). Every box, shape and byte count comes
// from stepLaunches and totals in src/sim/llamastep.ts.
const CFG = SYNTHETIC;
const POS = 5;
const STEP = stepLaunches(CFG, POS);
const LAYER0 = STEP.filter((l) => l.layer === 0);
const HEAD_ = STEP.filter((l) => l.layer === null && l.n === 0);
const TAIL = STEP.filter((l) => l.layer === null && l.n > 0);
const LATE = CFG.seqLen - 1;

const W = 720;
const H = 330;
const clamp = (v: number) => Math.max(0, Math.min(1, v));
const ramp = (p: number, a: number, b: number) => clamp((p - a) / (b - a));
const MB = (b: number) => (b / 2 ** 20).toFixed(2) + ' MiB';
const KB = (b: number) => (b / 1024).toFixed(1) + ' KiB';

function Frame({ beat, headline, children }: { beat: number; headline: string; children: ReactNode }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label={headline} data-beat={beat} style={{ display: 'block' }}>
      <text x={14} y={22} fontSize="13" fontWeight="600" data-headline="" style={{ fill: 'var(--ink)' }}>{headline}</text>
      {children}
    </svg>
  );
}

/** Where each launch's box sits: embed on the left, layer 0's 16 launches in two rows, the tail on the right. */
function place(l: Launch): { x: number; y: number } {
  if (l.layer === null && l.n === 0) return { x: 14, y: 70 };
  if (l.layer === 0) {
    const k = LAYER0.indexOf(l);
    return { x: 104 + (k % 8) * 66, y: k < 8 ? 70 : 150 };
  }
  const k = TAIL.indexOf(l);
  return { x: 104 + k * 132, y: 236 };
}

function Box({ l, lit, done }: { l: Launch; lit: boolean; done: boolean }) {
  const { x, y } = place(l);
  const wide = l.layer === null && l.n > 0;
  const w = wide ? 124 : 60;
  return (
    <g data-launch={l.n} data-job={l.job} data-lit={lit ? 1 : 0}>
      <rect x={x} y={y} width={w} height={54} rx="4"
        fill={lit ? 'var(--rule)' : 'var(--cell)'} stroke={lit ? 'var(--ink)' : 'var(--rule)'} strokeWidth={lit ? 2 : 1}
        opacity={done || lit ? 1 : 0.55} />
      <text x={x + w / 2} y={y + 18} fontSize="10.5" textAnchor="middle" fontWeight={lit ? 700 : 400} style={{ fill: 'var(--ink)' }}>{wide ? JOB_LABEL[l.job] : JOB_LABEL[l.job].replace('residual add', 'residual')}</text>
      <text x={x + w / 2} y={y + 33} fontSize="9" textAnchor="middle" fontWeight={l.exists ? 400 : 700} style={{ fill: 'var(--ink-2)' }}>{l.lab.replace(' (or 6.4, 15.6)', '').replace(' (yours)', '')}</text>
      <text x={x + w / 2} y={y + 46} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{l.milestone}</text>
    </g>
  );
}

function Pipeline({ lit, upto }: { lit: string[]; upto: number }) {
  const all = [...HEAD_, ...LAYER0, ...TAIL];
  return (
    <g>
      <text x={104} y={62} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>layer 0 (then the same for layers 1 to {CFG.layers - 1})</text>
      <text x={104} y={228} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>after the last layer</text>
      {all.map((l) => <Box key={l.n} l={l} lit={lit.includes(String(l.n))} done={l.n < upto} />)}
      <text x={14} y={318} fontSize="10" style={{ fill: 'var(--ink-2)' }}>
        labs in bold are new in this module (X3.1, X3.2); d {CFG.d}, {CFG.heads} query heads over {CFG.kvHeads} KV heads of 128, hidden {CFG.hidden}, vocabulary {CFG.vocab}
      </text>
    </g>
  );
}

const nOf = (layer: number | null, jobs: string[]) => STEP.filter((l) => l.layer === layer && jobs.includes(l.job)).map((l) => String(l.n));
const shapeLine = (l: Launch) => `${l.fn}: ${l.inputs} → ${l.outputs}`;

const beats: Beat[] = [
  {
    clip: 'llama-step-1',
    label: 'Embed',
    draw: (p) => (
      <Frame beat={1} headline={`token in: the embedding row is copied into x (${CFG.d} floats, ${KB(STEP[0].weightBytes)} of weights)`}>
        <Pipeline lit={ramp(p, 0.1, 0.2) > 0 ? ['0'] : []} upto={0} />
        <text x={14} y={300} fontSize="10.5" opacity={ramp(p, 0.3, 0.5)} style={{ fill: 'var(--ink)' }} data-shape="embed">{shapeLine(STEP[0])}</text>
      </Frame>
    ),
  },
  {
    clip: 'llama-step-2',
    label: 'Norm and q, k, v',
    draw: (p) => {
      const lit = [...nOf(0, ['rmsnorm']).slice(0, 1), ...(p > 0.35 ? nOf(0, ['q', 'k', 'v']) : [])];
      const q = STEP.find((l) => l.layer === 0 && l.job === 'q')!;
      return (
        <Frame beat={2} headline="RMSNorm (lab X3.1), then three GEMVs (lab 22.1's gather at batch 1)">
          <Pipeline lit={lit} upto={1} />
          <text x={14} y={300} fontSize="10.5" opacity={ramp(p, 0.4, 0.6)} style={{ fill: 'var(--ink)' }} data-shape="q">{shapeLine(q)}, {MB(q.weightBytes)}</text>
        </Frame>
      );
    },
  },
  {
    clip: 'llama-step-3',
    label: 'RoPE and append',
    draw: (p) => {
      const ap = STEP.find((l) => l.layer === 0 && l.job === 'append')!;
      return (
        <Frame beat={3} headline={`RoPE on q and k at position ${POS}; k and v go to cache row ${POS}`}>
          <Pipeline lit={p < 0.5 ? nOf(0, ['rope-q', 'rope-k']) : nOf(0, ['append'])} upto={4} />
          <text x={14} y={300} fontSize="10.5" opacity={ramp(p, 0.5, 0.7)} style={{ fill: 'var(--ink)' }} data-shape="append">{shapeLine(ap)}</text>
        </Frame>
      );
    },
  },
  {
    clip: 'llama-step-4',
    label: 'Attention',
    draw: (p) => {
      const at = STEP.find((l) => l.layer === 0 && l.job === 'attention')!;
      return (
        <Frame beat={4} headline={`decode attention over ${POS + 1} cached rows per KV head (lab 20.5, grouped), then Wo and a residual add`}>
          <Pipeline lit={p < 0.55 ? nOf(0, ['attention']) : nOf(0, ['wo', 'residual']).slice(0, 2)} upto={7} />
          <text x={14} y={300} fontSize="10.5" opacity={ramp(p, 0.2, 0.4)} style={{ fill: 'var(--ink)' }} data-shape="attention">
            {at.fn}: reads {KB(at.cacheBytes)} of cache at pos {POS}
          </text>
        </Frame>
      );
    },
  },
  {
    clip: 'llama-step-5',
    label: 'Feed-forward',
    draw: (p) => {
      const jobs = ['rmsnorm', 'w1', 'w3', 'swiglu', 'w2', 'residual'];
      const ffn = LAYER0.slice(10);
      const k = Math.min(ffn.length - 1, Math.floor(p * ffn.length * 1.2));
      return (
        <Frame beat={5} headline={`the feed-forward: RMSNorm, W1 and W3, SwiGLU, W2, residual; then ${CFG.layers - 1} more layers`}>
          <Pipeline lit={[String(ffn[k].n)]} upto={ffn[k].n} />
          <text x={14} y={300} fontSize="10.5" style={{ fill: 'var(--ink)' }} data-ffn={jobs.join(',')}>{shapeLine(ffn[k])}</text>
        </Frame>
      );
    },
  },
  {
    clip: 'llama-step-6',
    label: 'Logits and argmax',
    draw: (p) => {
      const lm = TAIL.find((l) => l.job === 'lm-head')!;
      return (
        <Frame beat={6} headline={`final RMSNorm, the tied LM head (${MB(lm.weightBytes)}), lab 10.3's argmax`}>
          <Pipeline lit={p < 0.3 ? nOf(null, ['final-norm']) : p < 0.65 ? nOf(null, ['lm-head']) : nOf(null, ['argmax', 'advance'])} upto={STEP.length - 4} />
          <text x={14} y={300} fontSize="10.5" opacity={ramp(p, 0.3, 0.5)} style={{ fill: 'var(--ink)' }} data-shape="lm-head">{shapeLine(lm)}</text>
        </Frame>
      );
    },
  },
  {
    clip: 'llama-step-7',
    label: 'Totals',
    draw: (p) => {
      const pos = Math.round(ramp(p, 0.2, 0.9) * LATE);
      const t = totals(stepLaunches(CFG, pos));
      const all = t.weightBytes + t.actBytes + t.cacheBytes + t.tableBytes;
      const parts: [string, number][] = [
        ['matrix products', t.byKind.matmul],
        ['attention (cache reads)', t.byKind.attention],
        ['everything else', all - t.byKind.matmul - t.byKind.attention],
      ];
      return (
        <Frame beat={7} headline={`${t.launches} launches per token; ${MB(t.weightBytes)} of weights read per token`}>
          <text x={14} y={64} fontSize="11.5" style={{ fill: 'var(--ink)' }} data-pos={pos}>bytes moved per token at pos {pos}: {MB(all)}</text>
          {parts.map(([name, b], i) => (
            <g key={name} data-part={name} data-bytes={b}>
              <text x={14} y={110 + i * 56} fontSize="11" style={{ fill: 'var(--ink)' }}>{name}</text>
              <rect x={200} y={96 + i * 56} width={480} height={20} fill="none" stroke="var(--rule)" />
              <rect x={200} y={96 + i * 56} width={(480 * b) / all} height={20} fill="var(--ink-2)" />
              <text x={200} y={132 + i * 56} fontSize="10" style={{ fill: 'var(--ink-2)' }}>{((100 * b) / all).toFixed(1)} % ({MB(b)})</text>
            </g>
          ))}
          <text x={14} y={300} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
            the weights are the same at every pos; the cache that attention reads grows by one row per layer per step
          </text>
        </Frame>
      );
    },
  },
];

export function LlamaStepScene() {
  return <Scene chapter="x3" title="One token through the step" beats={beats} />;
}
