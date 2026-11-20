import { useEffect, useMemo, useState, type MouseEvent, type ReactNode } from 'react';
import { Ask, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, TierLegend, Today, type Tier } from '../../components/notes';
import { benchRows, num, realBackends, useReports } from '../../lib/reports';
import { href } from '../../lib/router';
import { SEGMENT_MODELS, runs as byteRuns, warpAccess } from '../../sim/coalesce';
import {
  CRITERION_LABEL,
  DECOMP_LABEL,
  INTERLEAVE_LABEL,
  MAPPING_LABEL,
  NVIDIA_WARP,
  PRESETS,
  SPLITK_TINY,
  TOUR,
  decomposeState,
  describeSKStep,
  describeWorker,
  gradeCard,
  mappingsOf,
  ownerOfWeight,
  splitChoices,
  splitKOrder,
  splitKRun,
  splitKTraffic,
  warp0Load,
  warpOps,
  warpPlace,
  type Criterion,
  type Decomp,
  type Decomposition,
  type ExplorerState,
  type Interleave,
  type Kind,
  type MatvecProblem,
  type SplitKConfig,
} from '../../sim/decompose';
import { GatherScatterScene } from './GatherScatterScene';

const SECTOR = SEGMENT_MODELS.nvidia.segmentBytes!; // NVIDIA's 32-byte unit (ch06-facts A2)
const fmt = (v: number) => v.toLocaleString('en-US');
const fmt2 = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(2));

// ---------------------------------------------------------------- the lesson's kernels
// Copies of labs/ch22/snippets/scatter_conv_plain.cu and gather_conv.cu, which are
// front-end-checked and run on both emulators by S222.snippets.test.ts. That test also
// fails if these copies drift from the files.
const SCATTER = `// Input-centric 1-D convolution (scatter): thread i owns in[i]
// and adds its contribution to every output that uses it.
// Wrong on purpose: several threads add into the same out[o],
// and the += is a plain read, add and write, not an atomic.
// out[o] = sum over t of filter[t] * in[o + t - r]
__global__ void convScatterPlain(const float* in, const float* filter,
                                 float* out, int n, int r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    for (int t = 0; t <= 2 * r; ++t) {
        int o = i - t + r;            // out[o] reads in[i] with tap t
        if (o >= 0 && o < n) out[o] += filter[t] * v;
    }
}

void launchConvScatterPlain(const float* in, const float* filter,
                            float* out, int n, int r) {
    cudaMemset(out, 0, n * sizeof(float));  // the threads add into it
    convScatterPlain<<<(n + 255) / 256, 256>>>(in, filter, out, n, r);
}`;

const GATHER = `// Output-centric 1-D convolution (gather): thread o owns out[o],
// adds up the inputs it needs in a register, and writes once.
// No other thread writes out[o].
// out[o] = sum over t of filter[t] * in[o + t - r]
__global__ void convGather(const float* in, const float* filter,
                           float* out, int n, int r) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n) return;
    float acc = 0.0f;
    for (int t = 0; t <= 2 * r; ++t) {
        int i = o + t - r;            // the input tap t reads
        if (i >= 0 && i < n) acc += filter[t] * in[i];
    }
    out[o] = acc;
}

void launchConvGather(const float* in, const float* filter,
                      float* out, int n, int r) {
    convGather<<<(n + 255) / 256, 256>>>(in, filter, out, n, r);
}`;

const CONV_TIERS: Record<string, Tier> = {
  in: 'global', filter: 'global', out: 'global',
  i: 'reg', o: 'reg', t: 'reg', v: 'reg', acc: 'reg', n: 'reg', r: 'reg',
};

/** Code in a table cell: may wrap at its spaces, never inside an identifier. */
const C = ({ children }: { children: ReactNode }) => (
  <code style={{ whiteSpace: 'normal', overflowWrap: 'normal', wordBreak: 'normal' }}>{children}</code>
);

// ---------------------------------------------------------------- explorer: warp widths
const LAB61 = 'ch06/lab1_coalescing';
interface WidthChoice {
  key: string;
  width: number;
  label: string;
}
/** The documented widths, plus any the reader's devices reported (lab 6.1's simd_width row, else warpSize). */
function useWidthChoices(): WidthChoice[] {
  const { reports } = useReports();
  const fixed: WidthChoice[] = [
    { key: 'nv', width: NVIDIA_WARP, label: `${NVIDIA_WARP} (NVIDIA; AMD gfx10 and later)` },
    { key: 'amd', width: 64, label: '64 (AMD gfx9, Instinct)' },
  ];
  const devices = realBackends(reports).flatMap((b) => {
    const rows = benchRows(reports[b], LAB61, 'solution');
    const w = num(rows.find((r) => r.name === 'simd_width')?.width) ?? num(Number(reports[b].device?.warpSize));
    return w ? [{ key: 'dev:' + b, width: w, label: `${w} (your ${String(reports[b].device.name ?? b)}, ${b})` }] : [];
  });
  return [...fixed, ...devices];
}

// ---------------------------------------------------------------- explorer: drawings
const DW = 720;

/** Inputs on top, workers in the middle, outputs below; arrows from the decomposition. */
function Bipartite({ d, sel, onSel }: { d: Decomposition; sel: number | null; onSel: (w: number) => void }) {
  const p = d.problem;
  const L = 74;
  const span = DW - L - 14;
  const pitch = (n: number) => Math.min(46, span / n);
  const xs = (n: number) => (k: number) => L + (span - n * pitch(n)) / 2 + k * pitch(n) + pitch(n) / 2;
  const inX = xs(p.nIn);
  const outX = xs(p.nOut);
  const output = d.mapping === 'output';
  const wX = (w: number) => (output ? outX(w) : inX(w));
  const cIn = Math.min(30, pitch(p.nIn) - 4);
  const cOut = Math.min(30, pitch(p.nOut) - 4);
  const rW = Math.max(4, Math.min(11, Math.min(output ? pitch(p.nOut) : pitch(p.nIn)) / 2 - 2));
  const IY = 26, WY = 132, OY = 222, H = 286;
  const label = (i: number): string => {
    if (p.kind === 'hist') return String(p.values[i]);
    if (p.kind === 'spmv') return String(p.rowOf[i]);
    return String(i);
  };
  const dim = (w: number) => (sel != null && sel !== w ? 0.18 : 1);
  const selW = sel != null ? d.workers[sel] : null;
  return (
    <svg viewBox={`0 0 ${DW} ${H}`} role="img" aria-label="Threads, the inputs they read and the outputs they write" style={{ width: '100%', minWidth: 560, display: 'block' }}>
      <text x={8} y={IY + cIn / 2 + 4} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}>in</text>
      <text x={8} y={IY + cIn / 2 + 18} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>
        {p.kind === 'hist' ? 'value (bin)' : p.kind === 'spmv' ? 'non-zero, row' : 'index'}
      </text>
      <text x={8} y={WY + 4} fontSize="11.5" style={{ fill: 'var(--ink-2)' }}>threads</text>
      <text x={8} y={OY + cOut / 2 + 4} fontSize="11.5" fontWeight="600" style={{ fill: 'var(--global)' }}>out</text>
      {/* reads */}
      {d.workers.map((w) => (
        <g key={'r' + w.id} opacity={dim(w.id)}>
          {w.useful.map((i) => (
            <line key={i} x1={inX(i)} y1={IY + cIn} x2={wX(w.id)} y2={WY - rW} stroke="var(--ink-3)" strokeWidth={sel === w.id ? 1.8 : 1} />
          ))}
        </g>
      ))}
      {selW?.inspectsAll
        ? Array.from({ length: p.nIn }, (_, i) =>
            selW.useful.includes(i) ? null : (
              <line key={'x' + i} x1={inX(i)} y1={IY + cIn} x2={wX(selW.id)} y2={WY - rW} stroke="var(--ink-3)" strokeDasharray="2 3" strokeWidth="1" />
            ),
          )
        : null}
      {/* writes */}
      {d.workers.map((w) => (
        <g key={'w' + w.id} opacity={dim(w.id)}>
          {w.writes.map((x) => {
            const bad = d.writers[x.o] > 1;
            return (
              <line key={x.o} x1={wX(w.id)} y1={WY + rW} x2={outX(x.o)} y2={OY - 2} stroke={bad ? 'var(--fail)' : 'var(--ink)'}
                strokeWidth={sel === w.id ? 2.2 : bad ? 1.3 : 1.5} />
            );
          })}
        </g>
      ))}
      <g data-row="in">
        {Array.from({ length: p.nIn }, (_, i) => {
          const used = selW?.useful.includes(i);
          return (
            <g key={i}>
              <rect x={inX(i) - cIn / 2} y={IY} width={cIn} height={cIn} rx="2" fill={used ? 'var(--global)' : 'var(--global-wash)'} stroke="var(--global)" />
              {cIn >= 14 ? (
                <text x={inX(i)} y={IY + cIn / 2 + 4} fontSize="11" textAnchor="middle" style={{ fill: used ? 'var(--page)' : 'var(--ink)' }}>{label(i)}</text>
              ) : null}
            </g>
          );
        })}
      </g>
      <g data-row="workers">
        {d.workers.map((w) => (
          <g key={w.id} data-worker={w.id} role="button" aria-label={`thread ${w.id}`} style={{ cursor: 'pointer' }} onClick={() => onSel(w.id)}>
            <circle cx={wX(w.id)} cy={WY} r={rW + 3} fill="transparent" />
            <circle cx={wX(w.id)} cy={WY} r={rW} fill={sel === w.id ? 'var(--ink)' : 'var(--page)'} stroke="var(--ink)" strokeWidth="1.3" />
          </g>
        ))}
      </g>
      <g data-row="out" data-writers={d.writers.join(' ')}>
        {d.writers.map((n, o) => {
          const bad = n > 1;
          return (
            <g key={o}>
              <rect x={outX(o) - cOut / 2} y={OY} width={cOut} height={cOut} rx="2" fill={bad ? 'var(--fail-wash)' : 'var(--cell)'} stroke={bad ? 'var(--fail)' : 'var(--rule)'} strokeWidth="1.3" />
              {cOut >= 14 ? <text x={outX(o)} y={OY + cOut / 2 + 4} fontSize="11" textAnchor="middle">{o}</text> : null}
              <text x={outX(o)} y={OY + cOut + 14} fontSize="10.5" fontWeight="600" textAnchor="middle" style={{ fill: bad ? 'var(--fail)' : 'var(--ink-2)' }}>
                {bad ? `${n}×` : '1'}
              </text>
            </g>
          );
        })}
      </g>
      <text x={8} y={OY + cOut + 14} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>writers</text>
    </svg>
  );
}

/** W as N rows of K weights, x above, y at the right. */
function GridDrawing({ d, sel, onSel }: { d: Decomposition; sel: number | null; onSel: (w: number) => void }) {
  const p = d.problem as MatvecProblem;
  const { N, K } = p;
  const L = 34, GX = L, GW = 500, XY = 22, XH = 12, GY = 52;
  const rowH = Math.max(5, Math.min(32, 250 / N));
  const GH = rowH * N;
  const colW = GW / K;
  const YX = GX + GW + 92;
  const H = GY + GH + 40;
  const S = d.opts.S;
  const loaded = warp0Load(p, d.mapping, d.opts).map((a) => a / 4);
  // Contiguous runs of warp 0's elements, [row, first column, last column], drawn as one bar each.
  const loadRuns: [number, number, number][] = [];
  for (const e of [...loaded].sort((a, b) => a - b)) {
    const row = Math.floor(e / K);
    const col = e % K;
    const last = loadRuns[loadRuns.length - 1];
    if (last && last[0] === row && last[2] === col - 1) last[2] = col;
    else loadRuns.push([row, col, col]);
  }
  const region = (w: number): { x: number; y: number; w: number; h: number } => {
    const wk = d.workers[w];
    if (d.mapping === 'input') return { x: GX + w * colW, y: GY, w: Math.max(2, colW), h: GH };
    const row = wk.writes[0].o;
    const first = wk.useful[0] ?? 0;
    return { x: GX + first * colW, y: GY + row * rowH, w: wk.useful.length * colW, h: rowH };
  };
  const arrows = d.workers.reduce((s, w) => s + w.writes.length, 0) <= 64;
  const click = (e: MouseEvent<SVGRectElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const col = Math.min(K - 1, Math.max(0, Math.floor(((e.clientX - box.left) / box.width) * K)));
    const row = Math.min(N - 1, Math.max(0, Math.floor(((e.clientY - box.top) / box.height) * N)));
    onSel(ownerOfWeight(p, d.mapping, d.opts, row, col));
  };
  const yC = (j: number) => GY + j * rowH + rowH / 2;
  const writeLine = (w: number, o: number, strong: boolean) => {
    const r = region(w);
    const x1 = d.mapping === 'input' ? r.x + r.w / 2 : r.x + r.w;
    // S warps of one row leave at different heights so that their arrows visibly converge on y.
    const y1 = d.mapping === 'input' ? GY - 2 : d.mapping === 'splitk' ? r.y + (r.h * ((w % S) + 1)) / (S + 1) : r.y + r.h / 2;
    const bad = d.writers[o] > 1;
    return (
      <line key={`${w}-${o}`} x1={x1} y1={y1} x2={YX - 2} y2={yC(o)} stroke={bad ? 'var(--fail)' : 'var(--ink)'}
        strokeWidth={strong ? 2 : 1} opacity={sel != null && !strong ? 0.25 : 0.9} />
    );
  };
  return (
    <svg viewBox={`0 0 ${DW} ${H}`} role="img" aria-label="The weight matrix, the vector x and the outputs y" style={{ width: '100%', minWidth: 560, display: 'block' }}>
      <text x={4} y={XY + XH - 2} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}>x</text>
      <rect x={GX} y={XY} width={GW} height={XH} fill="var(--global-wash)" stroke="var(--global)" />
      <text x={GX + GW + 6} y={XY + XH - 2} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>K = {K}</text>
      <text x={4} y={GY + 12} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}>W</text>
      <rect x={GX} y={GY} width={GW} height={GH} fill="var(--global-wash)" stroke="var(--global)" />
      {/* row lines and chunk dividers */}
      {N <= 32 ? Array.from({ length: N - 1 }, (_, j) => (
        <line key={'r' + j} x1={GX} x2={GX + GW} y1={GY + (j + 1) * rowH} y2={GY + (j + 1) * rowH} stroke="var(--global)" strokeWidth="0.5" opacity="0.6" />
      )) : null}
      {d.mapping === 'splitk'
        ? Array.from({ length: S - 1 }, (_, c) => (
            <line key={'c' + c} x1={GX + d.workers[c + 1].useful[0] * colW} x2={GX + d.workers[c + 1].useful[0] * colW} y1={GY} y2={GY + GH} stroke="var(--ink)" strokeWidth="1.2" />
          ))
        : null}
      {sel != null && d.workers[sel] ? (() => {
        const r = region(sel);
        return <rect data-selected-region x={r.x} y={r.y} width={r.w} height={r.h} fill="var(--global)" opacity="0.55" />;
      })() : null}
      {/* warp 0's first load */}
      <g data-loaded={loaded.length}>
        {loadRuns.map(([row, c0, c1]) => (
          <rect key={`${row}-${c0}`} x={GX + c0 * colW} y={GY + row * rowH} width={Math.max(1.5, (c1 - c0 + 1) * colW)} height={rowH} fill="var(--ink)" />
        ))}
      </g>
      <rect x={GX} y={GY} width={GW} height={GH} fill="transparent" style={{ cursor: 'pointer' }} onClick={click} data-grid />
      {/* writes */}
      {arrows
        ? d.workers.flatMap((w) => w.writes.map((x) => writeLine(w.id, x.o, sel === w.id)))
        : sel != null && d.workers[sel] ? d.workers[sel].writes.map((x) => writeLine(sel, x.o, true)) : null}
      <g data-row="out" data-writers={d.writers.join(' ')}>
        {d.writers.map((n, j) => {
          const bad = n > 1;
          return (
            <g key={j}>
              <rect x={YX} y={GY + j * rowH + (rowH > 8 ? 1 : 0)} width={24} height={rowH > 8 ? rowH - 2 : rowH} rx="1.5"
                fill={bad ? 'var(--fail-wash)' : 'var(--cell)'} stroke={bad ? 'var(--fail)' : 'var(--rule)'} />
              {rowH >= 10 || N <= 16 ? (
                <text x={YX + 30} y={yC(j) + 4} fontSize="10" fontWeight="600" style={{ fill: bad ? 'var(--fail)' : 'var(--ink-2)' }}>{bad ? `${fmt(n)}×` : '1'}</text>
              ) : null}
            </g>
          );
        })}
      </g>
      <text x={YX} y={GY - 6} fontSize="11" fontWeight="600" style={{ fill: 'var(--global)' }}>y</text>
      <text x={YX + 30} y={GY - 6} fontSize="9.5" style={{ fill: 'var(--ink-2)' }}>writers</text>
      <rect x={GX} y={GY + GH + 14} width={10} height={10} fill="var(--ink)" />
      <text x={GX + 16} y={GY + GH + 23} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>warp 0's first load of W</text>
      <rect x={GX + 190} y={GY + GH + 14} width={10} height={10} fill="var(--global)" opacity="0.55" />
      <text x={GX + 206} y={GY + GH + 23} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>what the selected {d.mapping === 'warp' || d.mapping === 'splitk' ? 'warp' : 'thread'} reads (click W to choose)</text>
    </svg>
  );
}

// ---------------------------------------------------------------- explorer
const KIND_LABEL: Record<Kind, string> = {
  conv: '1-D convolution',
  matvec: 'decode GEMV, y = W x',
  hist: 'histogram',
  spmv: 'sparse matrix × vector',
};

function Slider({ label, value, min, max, step = 1, on, show }: {
  label: string; value: number; min: number; max: number; step?: number; on: (v: number) => void; show?: string;
}) {
  return (
    <label className="field">
      {label}: {show ?? value}
      <input type="range" min={min} max={max} step={step} value={value} aria-label={label} onChange={(e) => on(Number(e.target.value))} />
    </label>
  );
}

function Explorer() {
  const [s, setS] = useState<ExplorerState>(PRESETS[0].state);
  const [preset, setPreset] = useState<string | null>(PRESETS[0].id);
  const [sel, setSel] = useState<number | null>(null);
  const [widthKey, setWidthKey] = useState('nv');
  const widths = useWidthChoices();
  const d = useMemo(() => decomposeState(s), [s]);
  const m = d.metrics;
  const set = (next: ExplorerState) => {
    setS(next);
    setPreset(null);
    setSel(null);
  };
  const pick = (id: string) => {
    const p = PRESETS.find((x) => x.id === id)!;
    setS(p.state);
    setPreset(id);
    setSel(null);
    setWidthKey('nv');
  };
  const mv = s.matvec;
  const choices = splitChoices(mv.K, mv.warpWidth);
  const access = s.kind === 'matvec' ? warpAccess(warp0Load(d.problem as MatvecProblem, d.mapping, d.opts), 4, SECTOR) : null;
  const accessRuns = s.kind === 'matvec' ? byteRuns(warp0Load(d.problem as MatvecProblem, d.mapping, d.opts), 4) : null;
  const unit = d.mapping === 'warp' || d.mapping === 'splitk' ? 'warp' : 'thread';
  const selected = sel != null && sel < d.workers.length ? sel : null;
  const row = (crit: string, what: ReactNode, key: string, value: ReactNode, bad = false) => (
    <tr>
      <td className="muted">{crit}</td>
      <td>{what}</td>
      <td className="num" data-metric={key} style={bad ? { color: 'var(--fail)', fontWeight: 600 } : undefined}>{value}</td>
    </tr>
  );
  return (
    <Bench
      title="Who writes, who conflicts"
      task="Pick a computation and a cut. Which criteria does each cut win?"
      foot={
        <>
          Red: an output written by more than one {unit}, which needs an atomic for every update. Arrows in are reads, arrows out
          are writes. Load balance counts useful work per thread (a lane, for the warp cuts). <TierLegend tiers={['reg', 'global']} />
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="muted">Presets</span>
        <span className="seg" role="group" aria-label="Presets" style={{ flexWrap: 'wrap' }}>
          {PRESETS.map((p) => (
            <button key={p.id} aria-pressed={preset === p.id} onClick={() => pick(p.id)}>{p.label}</button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label="Computation" style={{ flexWrap: 'wrap' }}>
          {(Object.keys(KIND_LABEL) as Kind[]).map((k) => (
            <button key={k} aria-pressed={s.kind === k} onClick={() => set({ ...s, kind: k, mapping: mappingsOf(k).includes(s.mapping) ? s.mapping : 'output' })}>{KIND_LABEL[k]}</button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="seg" role="group" aria-label="Decomposition" style={{ flexWrap: 'wrap' }}>
          {mappingsOf(s.kind).map((mp) => (
            <button key={mp} aria-pressed={d.mapping === mp} onClick={() => set({ ...s, mapping: mp })}>{MAPPING_LABEL[s.kind][mp]}</button>
          ))}
        </span>
      </div>
      <div className="row" style={{ marginBottom: '0.9rem', alignItems: 'flex-end' }}>
        {s.kind === 'conv' ? (
          <>
            <Slider label="Inputs n" value={s.conv.n} min={6} max={16} on={(v) => set({ ...s, conv: { ...s.conv, n: v } })} />
            <label className="field">
              Filter width
              <select value={s.conv.w} onChange={(e) => set({ ...s, conv: { ...s.conv, w: Number(e.target.value) } })}>
                {[3, 5, 7].map((w) => <option key={w} value={w}>{w}</option>)}
              </select>
            </label>
          </>
        ) : null}
        {s.kind === 'hist' ? (
          <>
            <Slider label="Inputs n" value={s.hist.n} min={8} max={24} step={4} on={(v) => set({ ...s, hist: { ...s.hist, n: v } })} />
            <Slider label="Bins" value={s.hist.bins} min={2} max={8} on={(v) => set({ ...s, hist: { ...s.hist, bins: v } })} />
            <Slider label="Skew to bin 0" value={s.hist.skew} min={0} max={1} step={0.25} show={`${Math.round(s.hist.skew * 100)}%`} on={(v) => set({ ...s, hist: { ...s.hist, skew: v } })} />
          </>
        ) : null}
        {s.kind === 'spmv' ? (
          <>
            <Slider label="Rows" value={s.spmv.rows} min={4} max={8} on={(v) => set({ ...s, spmv: { ...s.spmv, rows: v } })} />
            <Slider label="Extra non-zeros in row 0" value={s.spmv.heavy} min={0} max={12} step={2} on={(v) => set({ ...s, spmv: { ...s.spmv, heavy: v } })} />
          </>
        ) : null}
        {s.kind === 'matvec' ? (
          <>
            <label className="field">
              Outputs N
              <select value={mv.N} onChange={(e) => set({ ...s, matvec: { ...mv, N: Number(e.target.value) } })}>
                {[1, 2, 4, 8, 16, 32].map((v) => <option key={v} value={v}>{v}</option>)}
              </select>
            </label>
            <label className="field">
              Inputs K
              <select value={mv.K} onChange={(e) => {
                const K = Number(e.target.value);
                const ok = splitChoices(K, mv.warpWidth);
                set({ ...s, matvec: { ...mv, K, S: ok.includes(mv.S) ? mv.S : ok[ok.length - 1] } });
              }}>
                {[64, 128, 256, 512].map((v) => <option key={v} value={v}>{v}</option>)}
              </select>
            </label>
            <label className="field">
              Warp width
              <select value={widthKey} onChange={(e) => {
                const c = widths.find((x) => x.key === e.target.value)!;
                setWidthKey(c.key);
                const ok = splitChoices(mv.K, c.width);
                set({ ...s, matvec: { ...mv, warpWidth: c.width, S: ok.includes(mv.S) ? mv.S : ok[ok.length - 1] } });
              }}>
                {widths.map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
              </select>
            </label>
            {d.mapping === 'splitk' ? (
              <label className="field">
                Chunks S
                <select value={mv.S} onChange={(e) => set({ ...s, matvec: { ...mv, S: Number(e.target.value) } })}>
                  {choices.map((v) => <option key={v} value={v}>{v}</option>)}
                </select>
              </label>
            ) : null}
          </>
        ) : null}
        <label className="field">
          Show {unit}
          <input type="number" min={0} max={d.workers.length - 1} value={selected ?? ''} placeholder="none" aria-label={`Show ${unit}`}
            onChange={(e) => setSel(e.target.value === '' ? null : Math.max(0, Math.min(d.workers.length - 1, Number(e.target.value))))} style={{ width: '5.5rem' }} />
        </label>
      </div>

      <div style={{ maxWidth: 880, margin: '0 auto 0.8rem', overflowX: 'auto' }}>
        {s.kind === 'matvec' ? <GridDrawing d={d} sel={selected} onSel={setSel} /> : <Bipartite d={d} sel={selected} onSel={setSel} />}
      </div>
      <div className="cols">
        <div className="narrate" role="status" data-mapping-text>
          {selected != null
            ? describeWorker(d, selected)
            : `Click a ${unit}${s.kind === 'matvec' ? ' (a cell of W)' : ''} to see its index mapping. ${fmt(m.workers)} ${unit}s in all.`}
        </div>
        <div>
          <table className="data" style={{ background: 'var(--page)', margin: 0 }} aria-label="The book's criteria for this cut">
            <thead>
              <tr><th style={{ paddingLeft: '0.5rem' }}>Criterion (p. 533)</th><th>Measure</th><th className="num">Value</th></tr>
            </thead>
            <tbody>
              {row('Parallelism', 'threads launched', 'threads', fmt(m.threads))}
              {row('Gather or scatter', 'outputs that need atomics', 'outputs-atomic', `${fmt(m.outputsNeedingAtomics)} of ${fmt(d.problem.nOut)}`, m.outputsNeedingAtomics > 0)}
              {row('', 'atomic updates', 'atomics', fmt(m.atomicUpdates), m.atomicUpdates > 0)}
              {row('', 'most writers on one output', 'worst', fmt(m.worstWriters), m.worstWriters > 1)}
              {row('', <>additions in <span className="t-reg">registers</span></>, 'registers', fmt(m.registerAccumulations))}
              {row('Ease of mapping', `inputs the busiest ${unit} inspects`, 'inspected-max', fmt(m.inspectedMax))}
              {row('', `inputs inspected in all, for ${fmt(m.contributions)} contributions`, 'inspected', fmt(m.inspectedTotal), m.inspectedTotal > m.contributions)}
              {row('Load balance', 'useful work per thread, max ÷ mean', 'imbalance', fmt2(m.imbalance), m.imbalance > 1.5)}
              {access
                ? row('Coalescing (ch. 6)', <>sectors in warp 0's first load of <span className="t-global">W</span> ({SECTOR}-byte unit)</>, 'sectors',
                    `${access.transactions}${accessRuns === 1 ? ', one run' : ''}`, access.transactions > access.bytesNeeded / SECTOR + 1)
                : null}
            </tbody>
          </table>
        </div>
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- classification
type Answer = { decomp?: Decomp; crit?: Criterion; checked?: boolean };

function SortTheTour() {
  const [k, setK] = useState(0);
  const [ans, setAns] = useState<Record<string, Answer>>({});
  const row = TOUR[k];
  const a = ans[row.id] ?? {};
  const grade = a.checked && a.decomp && a.crit ? gradeCard(row, a.decomp, a.crit) : null;
  const put = (x: Answer) => setAns({ ...ans, [row.id]: { ...a, ...x, checked: false } });
  const verdictOf = (id: string): 'right' | 'partly' | 'wrong' | '' => {
    const r = TOUR.find((t) => t.id === id)!;
    const x = ans[id];
    if (!x?.checked || !x.decomp || !x.crit) return '';
    const g = gradeCard(r, x.decomp, x.crit);
    return g.decomp && g.criterion ? 'right' : g.decomp || g.criterion ? 'partly' : 'wrong';
  };
  const done = TOUR.filter((t) => verdictOf(t.id));
  const right = done.filter((t) => verdictOf(t.id) === 'right').length;
  const v = verdictOf(row.id);
  return (
    <Bench
      title="Sort the book's tour"
      task="For each card: which decomposition, and what decided it?"
      foot={<>Seventeen cards from pp. 533–536 and four of ours, marked. Several criteria can be right for one card; the feedback says which. Your answers here are not recorded.</>}
    >
      <div className="row" style={{ marginBottom: '0.8rem', gap: '0.3rem' }} aria-label="Cards">
        {TOUR.map((t, i) => {
          const vv = verdictOf(t.id);
          return (
            <button key={t.id} aria-pressed={i === k} onClick={() => setK(i)} title={t.name} data-dot={t.id} data-verdict={vv}
              style={{ padding: '0.1rem 0.45rem', minWidth: '2rem', borderColor: vv === 'right' ? 'var(--ok)' : vv ? 'var(--fail)' : undefined, color: vv === 'right' ? 'var(--ok)' : vv ? 'var(--fail)' : undefined }}>
              {i + 1}
            </button>
          );
        })}
      </div>
      <div className="panel" data-card={row.id} data-verdict={v}>
        <p style={{ margin: '0 0 0.6rem', fontSize: '1rem' }}>
          <b>{row.name}</b>{' '}
          <span className="muted">{row.ours ? '' : `chapter ${row.chapter}`}</span>{' '}
          {row.ours ? <span className="chip global">ours, not in the book</span> : null}
        </p>
        <div className="row" style={{ marginBottom: '0.6rem' }}>
          <span className="muted" style={{ minWidth: '7.5rem' }}>Decomposition</span>
          <span className="seg" role="group" aria-label="Decomposition" style={{ flexWrap: 'wrap' }}>
            {(Object.keys(DECOMP_LABEL) as Decomp[]).map((x) => (
              <button key={x} aria-pressed={a.decomp === x} onClick={() => put({ decomp: x })}>{DECOMP_LABEL[x]}</button>
            ))}
          </span>
        </div>
        <div className="row" style={{ marginBottom: '0.8rem' }}>
          <span className="muted" style={{ minWidth: '7.5rem' }}>What decided it</span>
          <span className="seg" role="group" aria-label="What decided it" style={{ flexWrap: 'wrap' }}>
            {(Object.keys(CRITERION_LABEL) as Criterion[]).map((x) => (
              <button key={x} aria-pressed={a.crit === x} onClick={() => put({ crit: x })}>{CRITERION_LABEL[x]}</button>
            ))}
          </span>
        </div>
        <div className="row">
          <button className="primary" disabled={!a.decomp || !a.crit} onClick={() => setAns({ ...ans, [row.id]: { ...a, checked: true } })}>Check</button>
          <button onClick={() => setK((k + TOUR.length - 1) % TOUR.length)}>Previous card</button>
          <button onClick={() => setK((k + 1) % TOUR.length)}>Next card</button>
          <span className="muted" aria-live="polite" data-score={right}>{done.length} of {TOUR.length} checked, {right} fully right</span>
        </div>
        {grade ? (
          <div className="narrate" style={{ marginTop: '0.8rem' }} data-feedback>
            <p style={{ margin: '0 0 0.4rem' }}>
              <span className={'chip ' + (grade.decomp ? 'ok' : 'fail')}>{grade.decomp ? 'decomposition right' : `it is ${DECOMP_LABEL[row.decomp]}`}</span>{' '}
              <span className={'chip ' + (grade.criterion ? 'ok' : 'fail')}>
                {grade.criterion ? 'criterion right' : `decided by: ${row.accept.map((c) => CRITERION_LABEL[c]).join(', or ')}`}
              </span>
            </p>
            <p style={{ margin: 0 }}>
              {row.why} <span className="muted">{row.ours ? `By the book's criteria, p. ${row.pages}.` : `Book pp. ${row.pages}.`}</span>
            </p>
          </div>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- split-K stepper
const SK_S = [1, 2, 4];
const SK_ORDERS: Interleave[] = ['sequential', 'round-robin', 'reversed'];

function SplitKStepper() {
  const [S, setS] = useState(2);
  const [atomics, setAtomics] = useState(true);
  const [order, setOrder] = useState<Interleave>('sequential');
  const [lane, setLane] = useState(1);
  const [n, setN] = useState(0);
  const [playing, setPlaying] = useState(false);
  const cfg: SplitKConfig = useMemo(() => ({ ...SPLITK_TINY, S, atomics }), [S, atomics]);
  const run = useMemo(() => splitKRun(cfg, splitKOrder(cfg, order)), [cfg, order]);
  const total = run.steps.length;
  const at = Math.min(n, total);
  const snap = run.snaps[at];
  const cur = at > 0 ? run.steps[at - 1] : undefined;
  const firstLost = run.steps.findIndex((s) => s.lost);
  const traffic = splitKTraffic(cfg);
  const { batch, K, N, warpWidth: ww } = cfg;

  useEffect(() => setN(0), [S, atomics, order]);
  useEffect(() => {
    if (!playing) return;
    if (at >= total) return setPlaying(false);
    const id = window.setTimeout(() => setN((x) => x + 1), 380);
    return () => window.clearTimeout(id);
  }, [playing, at, total]);
  const nextCommit = () => {
    for (let i = at; i < total; i++) if (['store', 'atomic', 'read', 'write'].includes(run.steps[i].op.kind)) return setN(i + 1);
    setN(total);
  };

  // Drawing.
  const CWD = 30, X0 = 70, XY = 30, WY = XY + batch * CWD + 26, YX = X0 + K * CWD + 40;
  const cw = cur ? cur.warp : -1;
  const place = cw >= 0 ? warpPlace(cfg, cw) : null;
  const loadK = cur?.ks ?? [];
  const YY = XY + 20; // first row of Y
  const YP = 54; // pitch of Y's rows
  const H = Math.max(WY + N * CWD + 20 + batch * 26, YY + batch * YP + 8 + Math.ceil((N * S) / 4) * 22) + 6;
  const wv = (j: number, k: number) => run.data.W[j * K + k];
  const xv = (b: number, k: number) => run.data.X[b * K + k];
  const VW = YX + N * (CWD + 16) + 20;
  return (
    <Bench
      title="Split-K, warp by warp"
      task="Step with S = 1, 2 and 4. Then commit with a plain += and run all warps together."
      foot={
        <>
          Batch {batch}, {K} inputs, {N} outputs, a toy warp of {ww} lanes. Values are small integers so every sum is exact. <TierLegend tiers={['reg', 'global']} />
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <label className="field">
          Chunks S
          <span className="seg" role="group" aria-label="Chunks S" style={{ flexWrap: 'wrap' }}>
            {SK_S.map((v) => <button key={v} aria-pressed={S === v} onClick={() => setS(v)}>{v}</button>)}
          </span>
        </label>
        <label className="field">
          Lane 0 commits with
          <span className="seg" role="group" aria-label="Commit" style={{ flexWrap: 'wrap' }}>
            <button aria-pressed={atomics} onClick={() => setAtomics(true)}>an atomic add</button>
            <button aria-pressed={!atomics} onClick={() => setAtomics(false)}>a plain +=</button>
          </span>
        </label>
        <label className="field">
          Warp order
          <select value={order} onChange={(e) => setOrder(e.target.value as Interleave)} aria-label="Warp order" style={{ maxWidth: '100%' }}>
            {SK_ORDERS.map((o) => <option key={o} value={o}>{INTERLEAVE_LABEL[o]}</option>)}
          </select>
        </label>
        <label className="field">
          Lane to follow
          <select value={lane} onChange={(e) => setLane(Number(e.target.value))} aria-label="Lane to follow">
            {Array.from({ length: ww }, (_, l) => <option key={l} value={l}>{l}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <button onClick={() => { setPlaying(false); setN(0); }} disabled={at === 0}>Restart</button>
        <button onClick={() => { setPlaying(false); setN(at - 1); }} disabled={at === 0}>Back</button>
        <button className="primary" onClick={() => { setPlaying(false); setN(at + 1); }} disabled={at >= total}>Step</button>
        <button onClick={() => { setPlaying(false); nextCommit(); }} disabled={at >= total}>Run to the next commit</button>
        <button onClick={() => setPlaying(!playing)} disabled={at >= total}>{playing ? 'Pause' : 'Play'}</button>
        {firstLost >= 0 ? <button onClick={() => { setPlaying(false); setN(firstLost + 1); }}>Go to the first lost update</button> : null}
        <span className="muted" aria-live="polite" data-step={at}>step {at} of {total}</span>
      </div>
      <div className={'narrate' + (cur?.lost ? ' bad' : '')} role="status" style={{ marginBottom: '0.8rem' }} data-sk-text>
        {cur ? describeSKStep(run, at - 1, lane) : (
          <>
            Nothing has run yet. {N * S} warps:{' '}
            {S === 1 ? <>warp g owns output g and reads all {K} inputs of row g.</> : <>warp g owns output g ÷ {S} and chunk g mod {S} of the {K} inputs, as in lab 22.1's <code>gemvSplitKKernel</code>.</>}{' '}
            Press Step.
          </>
        )}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${VW} ${H}`} role="img" aria-label="x, W, the current warp's lanes and Y" style={{ width: '100%', minWidth: 560, display: 'block' }}>
          {/* x rows */}
          {Array.from({ length: batch }, (_, b) => (
            <g key={'x' + b}>
              <text x={X0 - 8} y={XY + b * CWD + 19} fontSize="12" textAnchor="end" style={{ fill: 'var(--global)' }}>x[{b}]</text>
              {Array.from({ length: K }, (_, k) => {
                const hot = cur && (cur.op.kind === 'load' || cur.op.kind === 'fma') && loadK.includes(k);
                return (
                  <g key={k}>
                    <rect x={X0 + k * CWD} y={XY + b * CWD} width={CWD - 2} height={CWD - 2} rx="2" fill={hot ? 'var(--global)' : 'var(--global-wash)'} stroke="var(--global)" strokeWidth="0.8" />
                    <text x={X0 + k * CWD + CWD / 2 - 1} y={XY + b * CWD + 19} fontSize="11" textAnchor="middle" style={{ fill: hot ? 'var(--page)' : 'var(--ink)' }}>{xv(b, k)}</text>
                  </g>
                );
              })}
            </g>
          ))}
          {Array.from({ length: K }, (_, k) => (
            <text key={'k' + k} x={X0 + k * CWD + CWD / 2 - 1} y={XY - 6} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>{k}</text>
          ))}
          {/* W */}
          {Array.from({ length: N }, (_, j) => (
            <g key={'w' + j}>
              <text x={X0 - 8} y={WY + j * CWD + 19} fontSize="12" textAnchor="end" style={{ fill: 'var(--global)' }}>W[{j}]</text>
              {Array.from({ length: K }, (_, k) => {
                const mine = place && place.j === j && k >= place.start && k < place.start + place.len;
                const hot = cur && (cur.op.kind === 'load' || cur.op.kind === 'fma') && place?.j === j && loadK.includes(k);
                const isLane = hot && loadK[lane] === k;
                return (
                  <g key={k}>
                    <rect x={X0 + k * CWD} y={WY + j * CWD} width={CWD - 2} height={CWD - 2} rx="2"
                      fill={hot ? 'var(--global)' : mine ? 'var(--global-wash)' : 'var(--cell)'} stroke={isLane ? 'var(--ink)' : 'var(--global)'} strokeWidth={isLane ? 2.2 : 0.8} />
                    <text x={X0 + k * CWD + CWD / 2 - 1} y={WY + j * CWD + 19} fontSize="11" textAnchor="middle" style={{ fill: hot ? 'var(--page)' : 'var(--ink)' }}>{wv(j, k)}</text>
                  </g>
                );
              })}
            </g>
          ))}
          {/* chunk dividers */}
          {Array.from({ length: S - 1 }, (_, c) => {
            const x = X0 + ((c + 1) * K / S) * CWD - 1;
            return <line key={'d' + c} x1={x} x2={x} y1={WY - 6} y2={WY + N * CWD + 4} stroke="var(--ink)" strokeWidth="2" strokeDasharray="4 3" />;
          })}
          {/* the current warp's lanes */}
          <text x={X0 - 8} y={WY + N * CWD + 30} fontSize="11.5" textAnchor="end" style={{ fill: 'var(--reg)' }}>{cw >= 0 ? `warp ${cw}` : 'lanes'}</text>
          {Array.from({ length: batch }, (_, b) => (
            <g key={'acc' + b} data-acc={b}>
              <text x={X0 + ww * 44 + 10} y={WY + N * CWD + 34 + b * 26} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>running sum for x[{b}]</text>
              {Array.from({ length: ww }, (_, l) => (
                <g key={l}>
                  <rect x={X0 + l * 44} y={WY + N * CWD + 16 + b * 26} width={40} height={22} rx="2" fill="var(--reg-wash)" stroke={l === lane ? 'var(--ink)' : 'var(--reg)'} strokeWidth={l === lane ? 1.8 : 1} />
                  <text x={X0 + l * 44 + 20} y={WY + N * CWD + 31 + b * 26} fontSize="11" textAnchor="middle" style={{ fill: 'var(--reg)' }}>{cw >= 0 ? snap.acc[cw][b][l] : 0}</text>
                </g>
              ))}
            </g>
          ))}
          {Array.from({ length: ww }, (_, l) => (
            <text key={'ln' + l} x={X0 + l * 44 + 20} y={WY + N * CWD + 12} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>lane {l}</text>
          ))}
          {/* Y */}
          <text x={YX} y={XY - 6} fontSize="12" fontWeight="600" style={{ fill: 'var(--global)' }}>Y (global memory)</text>
          {Array.from({ length: N }, (_, j) => (
            <text key={'yj' + j} x={YX + j * (CWD + 16) + (CWD + 10) / 2} y={XY + 14} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>j {j}</text>
          ))}
          {Array.from({ length: batch }, (_, b) => (
            <g key={'y' + b}>
              {Array.from({ length: N }, (_, j) => {
                const cell = b * N + j;
                const touched = cur?.touched?.includes(cell);
                const complete = snap.parts[cell].length === S && snap.Y[cell] === run.data.ref[cell];
                const bad = at >= total && snap.Y[cell] !== run.data.ref[cell];
                const x = YX + j * (CWD + 16);
                const y = YY + b * YP;
                return (
                  <g key={j} data-y={`${b},${j}`} data-value={snap.Y[cell]} data-parts={snap.parts[cell].join(',')}>
                    <rect x={x} y={y} width={CWD + 10} height={CWD} rx="2" fill={bad ? 'var(--fail-wash)' : complete ? 'var(--ok-wash)' : 'var(--cell)'}
                      stroke={bad ? 'var(--fail)' : touched ? 'var(--ink)' : complete ? 'var(--ok)' : 'var(--global)'} strokeWidth={touched ? 2.2 : 1} />
                    <text x={x + (CWD + 10) / 2} y={y + 20} fontSize="12" textAnchor="middle">{snap.Y[cell]}</text>
                    <text x={x + (CWD + 10) / 2} y={y + CWD + 12} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
                      {snap.parts[cell].length ? 'c ' + snap.parts[cell].join(',') : '·'}
                    </text>
                  </g>
                );
              })}
              <text x={YX - 6} y={YY + b * YP + 20} fontSize="10.5" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>b {b}</text>
            </g>
          ))}
          {/* warp progress */}
          <g>
            {Array.from({ length: N * S }, (_, w) => {
              const total = warpOps(cfg, w).length;
              const done = snap.done[w];
              const x = YX + (w % 4) * 34;
              const y = YY + batch * YP + 8 + Math.floor(w / 4) * 22;
              return (
                <g key={w}>
                  <rect x={x} y={y} width={30} height={16} rx="2" fill={done === total ? 'var(--rule-soft)' : 'var(--page)'} stroke={w === cw ? 'var(--ink)' : 'var(--rule)'} strokeWidth={w === cw ? 2 : 1} />
                  <text x={x + 15} y={y + 12} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>w{w}</text>
                </g>
              );
            })}
          </g>
        </svg>
      </div>
      <div className="row" style={{ marginTop: '0.6rem' }}>
        <span data-sk-threads={traffic.threads}>{traffic.threads} threads</span>
        <span data-sk-atomics={traffic.atomics}>
          {S === 1
            ? 'no shared outputs: plain stores'
            : atomics
              ? `${traffic.atomics} atomic updates`
              : <b className="t-fail">{traffic.atomics} unprotected read-add-writes on shared outputs</b>}
        </span>
        <span className="muted">W read: {traffic.wBytes} bytes; x read: {traffic.xBytes} bytes, whatever S is</span>
        {at >= total ? (
          run.correct
            ? <span className="chip ok" data-result="right">Y matches the reference</span>
            : <span className="chip fail" data-result="wrong">Y is wrong: {run.lostCount} contributions lost</span>
        ) : null}
      </div>
    </Bench>
  );
}

// ---------------------------------------------------------------- questions
export const questionsS222: Record<string, Question> = {
  'ch22/22.2/q-conv-writers': {
    kind: 'numeric',
    id: 'ch22/22.2/q-conv-writers',
    prompt: (
      <>
        A 1-D convolution with a 5-wide filter is decomposed input-centric: each thread owns one input and adds its
        contributions into the outputs. Away from the edges, how many threads write each output?
      </>
    ),
    answer: 5,
    explain: (
      <p>
        Each output is the sum of 5 inputs, so the 5 threads that own them all add into it, and each of those updates needs
        an atomic. Output-centric, the output has one writer that adds the 5 products in a register and stores once. In the
        explorer's convolution preset (12 inputs) that is 54 atomic updates against none.
      </p>
    ),
    source: 'ch. 22 §22.2, pp. 533–534',
  },
  'ch22/22.2/q-hist-inspect': {
    kind: 'numeric',
    id: 'ch22/22.2/q-hist-inspect',
    prompt: (
      <>
        A histogram of 1,000,000 inputs into 256 bins is decomposed output-centric: one thread per bin. How many times as
        many input reads does it make as the input-centric version?
      </>
    ),
    answer: 256,
    unit: '×',
    explain: (
      <p>
        A bin's thread cannot know which inputs fall in its bin without reading them, so each of the 256 threads reads all
        1,000,000: 256 million reads against 1 million. Parallelism falls from a million threads to 256. And the book's
        third objection survives even a free lookup: the bins receive different counts, so the threads are out of balance.
      </p>
    ),
    source: 'ch. 22 §22.2, p. 534',
  },
  'ch22/22.2/q-merge': {
    kind: 'choice',
    id: 'ch22/22.2/q-merge',
    prompt: <>The book's merge is output-centric, and atomics are not the reason. What is?</>,
    options: [
      {
        label: 'Every output receives exactly one input, so neither version needs atomics; cutting the output gives every thread the same number of elements, while cutting one input would leave threads merging very different amounts.',
        correct: true,
      },
      { label: 'Output-centric avoids atomics on the output array.', why: 'An input-centric merge writes each output once too: every element has exactly one destination.' },
      { label: 'An input-centric thread could not find its inputs without searching.', why: 'Both versions search; the book\'s point is how much work each thread ends up with.' },
    ],
    explain: (
      <p>
        Load balance under thread coarsening. A thread that owns a chunk of the output does the same amount of work as
        every other. A thread that owns a chunk of one input finds a counterpart range in the other input that can be
        short or long, so threads merge very different numbers of elements and diverge.
      </p>
    ),
    source: 'ch. 22 §22.2, p. 535; ch. 13',
  },
  'ch22/22.2/q-bfs': {
    kind: 'choice',
    id: 'ch22/22.2/q-bfs',
    prompt: (
      <>
        In vertex-centric push BFS, several threads may write the same neighbour's level at the same time, with no
        atomic. Why is that allowed?
      </>
    ),
    options: [
      { label: 'Every writer writes the same value, the next level, so the update is idempotent: whichever write lands last, the result is the same.', correct: true },
      { label: 'CUDA serialises plain stores to one address.', why: 'Which thread\'s plain store survives is undefined (ch09-facts A3). It does not matter here only because they all store the same value.' },
      { label: 'A level is written only once per kernel launch.', why: 'Several threads in one launch can reach the same unvisited neighbour and write it.' },
    ],
    explain: (
      <p>
        With atomics out of the question, the book says what decides push against pull is the parallelism each extracts
        and the load balance it gives, and both depend on the graph.
      </p>
    ),
    source: 'ch. 22 §22.2, p. 536; ch. 18',
  },
  'ch22/22.2/q-coo': {
    kind: 'choice',
    id: 'ch22/22.2/q-coo',
    prompt: (
      <>
        A sparse matrix has one row that holds half its non-zeros. Why might COO's input-centric kernel, with an atomic per
        non-zero, beat CSR's thread per row?
      </>
    ),
    options: [
      { label: 'With CSR one thread gets half the work while the others finish early; COO spreads the non-zeros evenly over threads, and the atomics are the price of that balance.', correct: true },
      { label: 'COO uses less memory than CSR.', why: 'Memory is not the trade-off the book describes; it is balance and parallelism against atomics.' },
      { label: 'Atomics are faster than register sums.', why: 'Nothing documents that, and the book says the opposite (p. 533).' },
    ],
    explain: (
      <p>
        In the explorer's skewed preset, CSR's busiest thread does 3 times the mean work; COO's threads do one non-zero
        each and pay 20 atomic updates for it.
      </p>
    ),
    source: 'ch. 22 §22.2, pp. 535–536; ch. 17',
  },
  'ch22/22.2/q-plain-scatter': {
    kind: 'choice',
    id: 'ch22/22.2/q-plain-scatter',
    prompt: (
      <>
        The input-centric kernel above passes every test on the <code>emu</code> backend, in all three thread orders. Is it
        correct?
      </>
    ),
    options: [
      {
        label: 'No. Two threads can read the same out[o], each add, and each write back, losing one update. emu never interrupts a thread between synchronisation points, so it cannot show this; the race backend reports the += line. Fix it with atomicAdd, or use the gather form.',
        correct: true,
      },
      { label: 'Yes: emu runs three thread orders, so it would have caught a race.', why: 'Each emulated thread runs to its next synchronisation point before the next one starts, so a += is never interrupted (labs/README.md).' },
      { label: 'Yes: the outputs are floats, and float additions cannot conflict.', why: 'The type does not matter: a read, add and write from two threads to one address can lose an update whatever is stored there.' },
    ],
    explain: (
      <p>
        This page's unit test builds this exact file on both emulators. On <code>emu</code> it gives the right answer in
        every order. On <code>race</code>, ThreadSanitizer reports the <code>+=</code> line racing with itself, and outputs
        can come out wrong. The <code>atomicAdd</code> version and the gather version are right on both, with no report.
      </p>
    ),
    source: 'ch. 22 §22.2, p. 533; ch. 9 §9.2; labs/README.md',
  },
  'ch22/22.2/q-hip-atomics': {
    kind: 'choice',
    id: 'ch22/22.2/q-hip-atomics',
    prompt: (
      <>
        A split-K matrix-vector kernel adds its partial sums with <code>atomicAdd</code> on <code>float</code>. It runs well
        on an NVIDIA GPU and slowly on an AMD one. Which documented difference is a candidate?
      </>
    ),
    options: [
      { label: 'HIP compiles float atomicAdd safely by default, possibly as a compare-and-swap loop, unless the code is built with -munsafe-fp-atomics.', correct: true },
      { label: 'HIP has no float atomicAdd, so the call falls back to a lock.', why: 'HIP has it, with the same spelling (infra-facts B1).' },
      { label: 'AMD GPUs serialise all atomics device-wide.', why: 'Nothing documents that. Atomics on one address serialise on every backend; AMD documents its global atomics in the L2 (ch09-facts B2).' },
    ],
    explain: (
      <p>
        AMD's documentation: a float <code>atomicAdd</code> "may generate a hardware instruction that is faster than
        emulating the atomic operation using an atomic compare-and-swap (CAS) loop", and "by default the compiler does not
        generate unsafe instructions". It is a candidate, not a diagnosis: measure with and without the flag. The course's
        HIP build does not pass it.
      </p>
    ),
    source: 'notes/today/infra-facts.md B1, checked 2026-11-20',
  },
  'ch22/22.2/q-metal-max': {
    kind: 'choice',
    id: 'ch22/22.2/q-metal-max',
    prompt: (
      <>
        You are porting to Metal a scatter that keeps a running maximum with <code>atomicMax</code> on floats. It compiles
        for HIP. What do you do?
      </>
    ),
    options: [
      {
        label: 'Restructure it: a gather with one owner per output (a reduction), or map the floats to order-preserving unsigned keys and use atomic max on atomic_uint. MSL documents only add and sub for atomic_float, and CUDA rejects atomicMax on float too, so the HIP version was never portable.',
        correct: true,
      },
      { label: 'Call atomic_fetch_max_explicit on an atomic_float.', why: 'MSL lists max and min for atomic_int and atomic_uint; for atomic_float it documents add and sub only (infra-facts C1).' },
      { label: 'Metal has no atomics, so a scatter cannot be ported.', why: 'Metal has atomic_int, atomic_uint and atomic_float, and fetch-and-add works on all three (infra-facts C1).' },
    ],
    explain: (
      <p>
        HIP accepts <code>atomicMax(float*, float)</code>; CUDA's legacy <code>atomicMax</code> takes integer types only
        and its guide's recipe for a float is a compare-and-swap loop (infra-facts A1, B1). The order-preserving key is the
        transform section 14.4 uses to sort logits; applying it to an atomic max is ours.
      </p>
    ),
    source: 'notes/today/infra-facts.md A1, B1, C1 and its backend table, checked 2026-11-20',
  },
  'ch22/22.2/q-determinism': {
    kind: 'choice',
    id: 'ch22/22.2/q-determinism',
    prompt: (
      <>
        Your split-K kernel's logits differ in the last bits from run to run, and now and then greedy decoding picks a
        different token. Which of the book's criteria for choosing a decomposition (p. 533) does this fall under?
      </>
    ),
    options: [
      { label: 'Others that depend on the application.', correct: true },
      { label: 'The parallelism exposed.', why: 'S changes the parallelism, not which bits come out.' },
      { label: 'How easily a thread finds its inputs.', why: 'Every warp finds its chunk directly.' },
      { label: 'Load balance.', why: 'Equal chunks are balanced. The variation comes from the order in which the atomic adds land.' },
    ],
    explain: (
      <p>
        Atomics add in whatever order they arrive, and floating-point addition is not associative (appendix A, §A.6). A
        warp per output adds in the same order every run and gives the same bits. Whether that matters depends on the
        application; for reproducible decoding it does.
      </p>
    ),
    source: 'ch. 22 §22.2, p. 533; appendix A §A.6, pp. 593–594; the scenario is ours',
  },
};

// ---------------------------------------------------------------- the section
export function S222() {
  return (
    <>
      <p>
        You have chosen an algorithm. One decision remains before you write the kernel, and it shapes everything after it:
        what does one thread own? The book frames this as cutting the problem into sub-problems that can safely run at the
        same time. Finding the parallelism in a large problem is usually simple in principle and hard in practice; the
        hard part is choosing the unit of work each thread, or group of threads, takes on.
      </p>
      <Source section="22.2" pages="532" />

      <h2>Two ways to cut</h2>
      <p>
        There are two natural answers. In an <b>output-centric</b> decomposition each thread owns one or more outputs and
        computes them from the inputs they need: all of the inputs for direct Coulomb summation in chapter 21, a subset for
        matrix multiplication or the binned cutoff method. In an <b>input-centric</b> decomposition each thread owns one or
        more inputs and applies their contributions to the outputs: to one output for a histogram, to a subset for a
        hypothetical atom-centric cutoff method, to every output for atom-centric direct summation.
      </p>
      <Source section="22.2" pages="532" figs="Fig. 22.1" />
      <p>
        Sometimes the distinction disappears. In odd-even sort each thread reads a pair and writes the same pair; inputs and
        outputs correspond so directly that the label does not matter. In every case the two cuts compute the same result,
        and they can run very differently on the same hardware. The explorer further down draws both for four computations.
      </p>
      <Source pages="533" />

      <h2>Gather and scatter</h2>
      <p>
        An output-centric thread usually <b>gathers</b>: it pulls the effect of many inputs into one output. The book
        gives two reasons that suits a GPU. The running total lives in a private <span className="t-reg">register</span>{' '}
        until the end. And neighbouring threads need overlapping inputs, which they can share through the caches or{' '}
        <span className="t-shared">shared memory</span> instead of each fetching them from global memory.
      </p>
      <p>
        An input-centric thread usually <b>scatters</b>: it pushes one input's effect out to several outputs. Now several
        threads may update one output at the same time. The outputs must live in memory all of them can write, and every
        update to a shared output must be an atomic read-modify-write, or updates are lost. The book's verdict is that
        those atomics are significantly slower than the register additions they replace.
      </p>
      <Source pages="533" />
      <Today checked="2026-11-20 (notes/today/ch22-facts.md A1)">
        <p>
          The mechanism is documented: NVIDIA says a register access "consumes zero extra clock cycles per instruction"
          (Best Practices Guide 10.2.7), atomics on one address are serialised, and device-scope atomics run in the L2
          (ch09-facts A4). How much slower an atomic is than a register add is documented nowhere, so this course gives no
          ratio. Chapter 9's labs and lab 22.1 measure it on your device.
        </p>
      </Today>
      <p>
        A still picture of a scatter shows the arrows converging. What it cannot show is time: the queue that forms when
        several updates arrive at one address. The animation draws that queue.
      </p>
      <Simplification>
        <p>
          One tick is one update to one address. Every update is issued at once, and different addresses are served in
          parallel. Register additions, caches and the real length of a tick are not modelled; no backend documents that
          length.
        </p>
      </Simplification>
      <GatherScatterScene />

      <h2>Who writes, who conflicts</h2>
      <p>
        Gather against scatter is not the whole decision. The book lists more criteria: how much parallelism the
        decomposition exposes, how easily a thread can identify the inputs that contribute to its outputs, the load balance
        it induces, and others that depend on the application. Chapter 21 is the one place the book implements both cuts
        and compares them; everywhere else the choice was made silently.
      </p>
      <Source pages="533" />
      <p>
        The explorer makes the choice explicit. Pick a computation and a cut, then click a thread to see its index mapping.
        Try each preset, then change one thing at a time:
      </p>
      <ul>
        <li>Convolution: switch between the two cuts. Is there any criterion on which input-centric wins?</li>
        <li>Histogram, a thread per bin: how many inputs does each thread inspect? Then raise the skew.</li>
        <li>Skewed sparse matrix: compare CSR's balance with COO's, and what COO pays for its balance.</li>
        <li>
          Decode GEMV: with 2 outputs, which cut exposes more than a couple of warps, and at what price? Then set 32
          outputs and compare the sectors warp 0 touches with a thread per output and with a warp per output.
        </li>
      </ul>
      <Simplification>
        <p>
          A thread (or warp) adds all its contributions to one output in a register and writes once. Nothing is timed: the
          atomic counts say how many updates serialise, not how long they take. No caches, so an input read by two threads
          counts twice. The sector count is warp 0's first load of W, one instruction, in NVIDIA's 32-byte unit
          (ch06-facts A2); AMD's lines are 64 or 128 bytes and Apple documents none.
        </p>
      </Simplification>
      <Explorer />
      <Ask q={questionsS222['ch22/22.2/q-conv-writers']} label="Work it out" />
      <Ask q={questionsS222['ch22/22.2/q-hist-inspect']} label="Work it out" />

      <h2>The book's tour</h2>
      <p>
        The book then walks back through every pattern it has taught and names the decomposition each one used. Most are
        output-centric for the obvious reason. The instructive cards are the ones where atomics decide nothing at all. For
        each card, choose the decomposition and then what decided it.
      </p>
      <Source pages="533–536" />
      <SortTheTour />
      <p>
        Two things to keep from the tour. Real computations mix strategies: the stable filter is input-centric, yet the
        scan inside it is output-centric, and the hybrid ELL-COO format combines an output-centric part (ELL) with an
        input-centric one (COO). And for sparse data the book concludes that the best decomposition depends on the input
        dataset itself.
      </p>
      <Source pages="534–536" />
      <Aside>
        <p>
          Two slips in the printed tour. Page 534 cites "scan (Chapter 12)"; scan is chapter 11, as the same chapter says
          on p. 530. Page 536 says the electrostatic map favours output-centric "because of the benefit over the gather
          pattern of the scatter pattern"; the rest of the paragraph shows it means the benefit of gather over scatter.
        </p>
      </Aside>
      <Ask q={questionsS222['ch22/22.2/q-merge']} />
      <Ask q={questionsS222['ch22/22.2/q-bfs']} />
      <Ask q={questionsS222['ch22/22.2/q-coo']} />

      <h2>Split-K: the atomic arrives with the second warp</h2>
      <p>
        Now a decomposition you will write. During decoding, each new token multiplies its activation vector by every
        weight matrix: y = W x, a matrix-vector product (section 22.5 says why decoding has this shape). The output-centric
        cut gives a warp to each output. Its lanes stride along row j of W together, so each step reads one contiguous run,
        and lane 0 stores the warp's total. No atomics. But a matrix with few outputs gives few warps.
      </p>
      <p>
        Split-K cuts the input dimension as well. S warps share an output, each taking one chunk of the row, which
        multiplies the warps by S. It also brings the scatter back: S warps now add into each element of Y.
      </p>
      <Beyond>
        <p>
          The decode GEMV and split-K are ours; chapter 22 prints no code. The mapping is lab 22.1's{' '}
          <code>gemvSplitKKernel</code>: warp g owns output g ÷ splits and chunk g mod splits.
        </p>
      </Beyond>
      <p>
        The stepper runs a tiny case with a toy warp of 4 lanes so you can see every one: a batch of 2 rows, 16 inputs, 3
        outputs. Step through it with S = 1, then 2, then 4, and watch where the commit changes. Then switch the commit to
        a plain <code>+=</code> and run all warps together.
      </p>
      <Simplification>
        <p>
          The lanes of a warp advance together, one operation per step, and warps interleave in the order you choose. No
          timing, no caches. A real warp is wider: read <code>warpSize</code>, or the pipeline's execution width on Metal.
          Two racing plain <code>+=</code> typically lose an update as shown, but formally it is a data race with undefined
          behaviour (ch09-facts A3).
        </p>
      </Simplification>
      <SplitKStepper />
      <p>
        Whatever S is, the warps between them read every weight once and every element of x once per output; the byte
        counts under the drawing do not move. What S changes is how many threads share the work and how many atomic updates
        the outputs take, which is exactly the trade lab 22.1 measures on your device. See the{' '}
        <a href={href('ch22', 'labs')}>labs page</a>.
      </p>

      <h2>This scatter passes on the emulator</h2>
      <p>
        Here is the convolution both ways, as kernels. Read the scatter carefully, then answer the question below it.
      </p>
      <div className="wide" style={{ display: 'grid', gap: '0 1rem', gridTemplateColumns: 'repeat(auto-fit, minmax(min(100%, 30rem), 1fr))' }}>
        <Code tabs={[{ label: 'scatter: a thread per input', code: SCATTER, tiers: CONV_TIERS, status: 'emulator-checked and front-end-checked; wrong on purpose' }]} />
        <Code tabs={[{ label: 'gather: a thread per output', code: GATHER, tiers: CONV_TIERS, status: 'emulator-checked and front-end-checked; unverified on GPUs' }]} />
      </div>
      <Ask q={questionsS222['ch22/22.2/q-plain-scatter']} />
      <p>
        Lab 22.1 hides the same mistake in its split-K kernel. Run it on the <code>race</code> backend before you go looking
        for it.
      </p>

      <h2>In PyTorch terms</h2>
      <p>
        You have used both directions already. <code>torch.gather</code> is output-indexed:{' '}
        <code>out[i] = src[index[i]]</code>, so each output knows the one input it takes. <code>Tensor.scatter_add_</code>{' '}
        is input-indexed: <code>out[index[i]] += src[i]</code>, so each input knows where it goes, and two values of i with
        the same index meet at one output. That is this section's conflict, from the definitions alone: any parallel{' '}
        <code>scatter_add_</code> in which two i share an index has to make that <code>+=</code> safe somehow.
      </p>
      <Beyond>
        <p>
          The bridge is ours, from the two operations' definitions. It says nothing about how PyTorch implements either on
          any backend.
        </p>
      </Beyond>

      <h2>On HIP and Metal</h2>
      <p>
        A gather needs none of the following. A scatter's cost, and even its spelling, change by backend. The table is
        for the two scatters this section has met: a float sum and a float maximum into shared outputs.
      </p>
      <Beyond>
        <p>
          Not in the book. From notes/today/infra-facts.md (A1, B1, C1 and its backend table) and ch22-facts.md, checked
          2026-11-20. Nothing here was run on AMD hardware.
        </p>
      </Beyond>
      <table className="data wide" style={{ maxWidth: '58rem' }}>
        <thead>
          <tr><th>Scatter</th><th>CUDA</th><th>HIP</th><th>Metal</th></tr>
        </thead>
        <tbody>
          <tr>
            <td>Float add into a shared output</td>
            <td><C>atomicAdd(float*, float)</C></td>
            <td>Same call; compiled safely, possibly as a compare-and-swap loop, unless built with <C>-munsafe-fp-atomics</C></td>
            <td><C>atomic_fetch_add_explicit</C> on a buffer declared <C>device atomic_float*</C>; add and sub only; threadgroup memory from Metal 4.1</td>
          </tr>
          <tr>
            <td>Float max into a shared output</td>
            <td>No float overload; a compare-and-swap loop</td>
            <td><C>atomicMax(float*, float)</C> exists (CUDA rejects it, so not portable back)</td>
            <td>None documented</td>
          </tr>
          <tr>
            <td>Where device atomics run</td>
            <td>Serialised in the L2 (ch09-facts A4)</td>
            <td>L2 for global memory (ch09-facts B2)</td>
            <td>Not documented (ch09-facts C4)</td>
          </tr>
        </tbody>
      </table>
      <Ask q={questionsS222['ch22/22.2/q-hip-atomics']} />
      <Ask q={questionsS222['ch22/22.2/q-metal-max']} />

      <h2>The order of a scatter</h2>
      <p>
        One more consequence of scattering, which the book's chapter 22 does not raise. Float atomics add in whatever order
        they arrive, and floating-point addition is not associative: appendix A works an example in which two orders of
        the same four numbers give 2.0 and 2.5. So a scatter's result can change in its last bits from run to run, while a
        gather that adds in a fixed order gives the same bits every time. In the book's terms this belongs to the fourth
        criterion, "others that depend on the application".
      </p>
      <Beyond>
        <p>
          Ours, built on appendix A §A.6 (pp. 593–594). PyTorch's own documentation lists{' '}
          <code>Tensor.scatter_add_()</code> on a CUDA tensor among its normally nondeterministic operations
          (notes/today/ch21-facts.md D1, checked 2026-11-20).
        </p>
      </Beyond>
      <Ask q={questionsS222['ch22/22.2/q-determinism']} />
    </>
  );
}
