import { useMemo, useState } from 'react';
import { Ask, Predict, type Question } from '../../components/Ask';
import { Code } from '../../components/Code';
import { Aside, Bench, Beyond, Simplification, Source, Today } from '../../components/notes';
import { benchRows, fmt, num, realBackends, useReports, type BenchRow, type Reports } from '../../lib/reports';
import { href } from '../../lib/router';
import { strideCase, tileAccess } from '../../sim/banks';
import {
  EXPRESSIONS,
  SEGMENT_MODELS,
  exercise1,
  exprAccess,
  strideEfficiency,
  tileLoad,
  warpAccess,
  type AccessParams,
  type Arrangement,
  type Ex1Verdict,
  type ExprId,
  type Layout,
  type SegmentModelId,
} from '../../sim/coalesce';
import { CoalesceScene } from './CoalesceScene';

// The kernel of the book's exercise 1 (pp. 154–155), one source line per numbered
// line. A copy of labs/ch06/snippets/s61_foo_kernel.cu, which ./verify.sh
// front-end-checks; snippets-s61.test.ts fails if the two drift apart.
const FOO = `__global__ void foo_kernel(float* a, float* b, float* c, float* d, float* e) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ float a_s[256];
    __shared__ float bc_s[4*256];
    a_s[threadIdx.x] = a[i];
    for(unsigned int j = 0; j < 4; ++j) {
        bc_s[j*256 + threadIdx.x] = b[j*blockDim.x*gridDim.x + i] + c[i*4 + j];
    }
    __syncthreads();
    d[i + 8] = a_s[threadIdx.x];
    e[i*8] = bc_s[threadIdx.x*4];
}`;
const FOO_TIERS = {
  a: 'global', b: 'global', c: 'global', d: 'global', e: 'global', a_s: 'shared', bc_s: 'shared', i: 'reg', j: 'reg',
} as const;

const LAB61 = 'ch06/lab1_coalescing';
const pct = (x: number) => `${fmt(x * 100, 3)} %`;
const bytes = (n: number) => n.toLocaleString('en-US');

// ---------------------------------------------------------------- reports

/** Backends that are real GPUs: realBackends() drops the emulator but not its race-detection flavour. */
export const gpuBackends = (reports: Reports) =>
  realBackends(reports).filter((b) => b !== 'race' && !/emulator/i.test(String(reports[b].device?.name ?? '')));

/** Bench rows of a lab on one backend: the reference solution's if present, else the reader's own. */
function labRows(reports: Reports, backend: string, lab: string): BenchRow[] {
  const s = benchRows(reports[backend], lab, 'solution');
  return s.length ? s : benchRows(reports[backend], lab, 'work');
}

/** Warp widths the reader's devices reported: lab 6.1's simd_width row, else the device's warpSize. */
function useDeviceWidths(): { backend: string; device: string; width: number }[] {
  const { reports } = useReports();
  return gpuBackends(reports).flatMap((b) => {
    const row = labRows(reports, b, LAB61).find((r) => r.name === 'simd_width');
    const w = num(row?.width) ?? num(Number(reports[b].device.warpSize));
    return w ? [{ backend: b, device: String(reports[b].device.name ?? b), width: w }] : [];
  });
}

// ---------------------------------------------------------------- drawing

/** Below this width the explorer's drawings scroll sideways inside the bench instead of shrinking past legibility. */
const NARROW_MIN = 600;

/** One warp's lanes, arrows to their bytes, and the aligned segments those bytes fall in. */
function SegmentStrip({
  lanes,
  warpWidth,
  elemBytes,
  seg,
  unit,
}: {
  lanes: { lane: number; byte: number }[];
  warpWidth: number;
  elemBytes: number;
  seg: number;
  unit: string;
}) {
  const a = warpAccess(lanes.map((l) => l.byte), elemBytes, seg);
  const W = 700;
  const lw = Math.min(20, (W - 40) / warpWidth);
  const lx = (l: number) => 20 + l * lw;
  const n = a.segments.length;
  const gaps = a.segments.filter((s, i) => i > 0 && s !== a.segments[i - 1] + 1).length;
  const bw = Math.min(64, (W - 40 - gaps * 10) / Math.max(1, n));
  const xs: number[] = [];
  let x = 0;
  a.segments.forEach((s, i) => {
    if (i > 0 && s !== a.segments[i - 1] + 1) x += 10;
    xs.push(x);
    x += bw;
  });
  const off = (W - x) / 2;
  const boxX = (i: number) => off + xs[i];
  const posOf = (byte: number) => {
    const i = a.segments.indexOf(Math.floor(byte / seg));
    return boxX(i) + ((byte % seg) / seg) * bw;
  };
  const active = new Set(lanes.map((l) => l.lane));
  const SY = 104;
  const servedBy = (s: number) => lanes.filter((l) => Math.floor(l.byte / seg) <= s && Math.floor((l.byte + elemBytes - 1) / seg) >= s).map((l) => l.lane);
  return (
    <div style={{ overflowX: 'auto' }}>
    <svg viewBox={`0 0 ${W} 166`} role="img" aria-label="Lanes, their addresses and the memory segments they touch" style={{ width: '100%', minWidth: NARROW_MIN, display: 'block' }} data-segments={n}>
      <text x={20} y={12} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>lanes of one warp</text>
      {Array.from({ length: warpWidth }, (_, l) => (
        <g key={l} opacity={active.has(l) ? 1 : 0.25}>
          <rect x={lx(l)} y={18} width={lw - 2} height={16} rx="2" fill="var(--cell)" stroke="var(--ink-3)" />
          {lw >= 14 ? <text x={lx(l) + (lw - 2) / 2} y={29.5} fontSize="8.5" textAnchor="middle">{l}</text> : null}
        </g>
      ))}
      {lanes.map((l) => (
        <line key={l.lane} x1={lx(l.lane) + (lw - 2) / 2} y1={35} x2={posOf(l.byte) + 0.5} y2={SY - 1} stroke="var(--global)" strokeWidth="0.8" opacity="0.75" />
      ))}
      {a.segments.map((s, i) => {
        const lo = s * seg;
        return (
          <g key={s} data-segment={s}>
            <rect x={boxX(i)} y={SY} width={bw} height={20} fill="var(--global-wash)" stroke="var(--global)" strokeWidth="1" />
            {lanes
              .filter((l) => l.byte < lo + seg && l.byte + elemBytes > lo)
              .map((l) => {
                const a0 = Math.max(l.byte, lo);
                const a1 = Math.min(l.byte + elemBytes, lo + seg);
                return <rect key={l.lane} x={boxX(i) + ((a0 - lo) / seg) * bw} y={SY} width={((a1 - a0) / seg) * bw} height={20} fill="var(--global)" />;
              })}
            {i > 0 && s !== a.segments[i - 1] + 1 ? (
              <text x={boxX(i) - 5} y={SY + 14} fontSize="9" textAnchor="middle" style={{ fill: 'var(--ink-3)' }}>⋯</text>
            ) : null}
            {n <= 8 ? (
              <text x={boxX(i) + bw / 2} y={SY + 34} fontSize="9.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>
                {(() => {
                  const ls = servedBy(s);
                  return ls.length === 1 ? `lane ${ls[0]}` : ls.length === warpWidth || (ls.length > 1 && ls.every((v, j) => j === 0 || v === ls[j - 1] + 1)) ? `lanes ${ls[0]}–${ls[ls.length - 1]}` : `${ls.length} lanes`;
                })()}
              </text>
            ) : null}
          </g>
        );
      })}
      <text x={20} y={SY + 52} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
        {n} aligned {seg}-byte {unit}{n === 1 ? '' : 's'}, bytes {bytes(a.segments[0] * seg)} to {bytes((a.segments[n - 1] + 1) * seg - 1)}
        {gaps ? ' (gaps not to scale)' : ''}; solid: bytes the warp asked for
      </text>
    </svg>
    </div>
  );
}

/** A window onto the logical matrix around the cells the warp reads. */
function MatrixWindow({ name, Width, cells, colMajor }: { name: string; Width: number; cells: { r: number; c: number }[]; colMajor: boolean }) {
  const rs = cells.map((x) => x.r);
  const cs = cells.map((x) => x.c);
  const nR = Math.min(Width, 6);
  const nC = Math.min(Width, Math.max(...cs) - Math.min(...cs) + 1 + 4, 68);
  const r0 = Math.max(0, Math.min(Math.min(...rs) - 2, Width - nR));
  const c0 = Math.max(0, Math.min(Math.min(...cs) - 2, Width - nC));
  const cell = Math.min(16, 640 / nC);
  const hot = new Set(cells.map((x) => `${x.r},${x.c}`));
  return (
    <div style={{ overflowX: 'auto' }}>
    <svg viewBox={`0 0 700 ${nR * cell + 46}`} role="img" aria-label={`Logical view of ${name}`} style={{ width: '100%', minWidth: NARROW_MIN, display: 'block' }}>
      <text x={20} y={12} fontSize="10.5" style={{ fill: 'var(--ink-2)' }}>
        {name}, rows {r0}–{r0 + nR - 1}, columns {c0}–{c0 + nC - 1} of {Width}; stored {colMajor ? 'column by column (addresses run down)' : 'row by row (addresses run across)'}
      </text>
      {Array.from({ length: nR * nC }, (_, i) => {
        const r = r0 + Math.floor(i / nC);
        const c = c0 + (i % nC);
        const on = hot.has(`${r},${c}`);
        return (
          <rect key={i} x={20 + (c - c0) * cell} y={20 + (r - r0) * cell} width={cell - 1} height={cell - 1}
            fill={on ? 'var(--global)' : 'var(--cell)'} stroke={on ? 'var(--global)' : 'var(--rule)'} strokeWidth="0.6" />
        );
      })}
      <text x={20} y={nR * cell + 36} fontSize="10" style={{ fill: 'var(--ink-3)' }}>solid: the elements this warp reads at this step</text>
    </svg>
    </div>
  );
}

// ---------------------------------------------------------------- the explorer

const EXPR_ORDER: { id: ExprId; label: string }[] = [
  { id: 'n-row', label: 'N[k*Width + col], N row-major (Fig. 6.2)' },
  { id: 'n-col', label: 'N[col*Width + k], N column-major (Fig. 6.3)' },
  { id: 'm-row', label: 'M[row*Width + k] (the M read of both figures)' },
  { id: 'stride', label: 'in[i*stride]' },
  { id: 'offset', label: 'a[i + offset]' },
  { id: 'ex1-a', label: 'Exercise 1(a): a[i]' },
  { id: 'ex1-c', label: 'Exercise 1(c): b[j*blockDim.x*gridDim.x + i]' },
  { id: 'ex1-d', label: 'Exercise 1(d): c[i*4 + j]' },
  { id: 'ex1-g', label: 'Exercise 1(g): d[i + 8]' },
  { id: 'ex1-i', label: 'Exercise 1(i): e[i*8]' },
];

interface ExplorerState {
  expr: ExprId;
  Width: number;
  k: number;
  bdx: number; // 0 = at least as wide as the warp
  stride: number;
  offset: number;
  j: number;
  warp: number;
  elem: number;
  model: SegmentModelId;
  custom: number; // segment size chosen for Metal
}
const START: ExplorerState = { expr: 'n-row', Width: 1024, k: 0, bdx: 0, stride: 1, offset: 0, j: 0, warp: 32, elem: 4, model: 'nvidia', custom: 64 };
const PRESETS: { label: string; s: Partial<ExplorerState> }[] = [
  { label: 'Fig. 6.2: row-major N', s: { expr: 'n-row' } },
  { label: 'Fig. 6.3: column-major N', s: { expr: 'n-col' } },
  { label: 'The M read', s: { expr: 'm-row' } },
  { label: 'One float off', s: { expr: 'offset', offset: 1 } },
  { label: 'Stride 2', s: { expr: 'stride', stride: 2 } },
];

function CoalescingExplorer() {
  const [s, setS] = useState<ExplorerState>(START);
  const set = (o: Partial<ExplorerState>) => setS((x) => ({ ...x, ...o }));
  const widths = useDeviceWidths();
  const info = EXPRESSIONS[s.expr];
  const model = SEGMENT_MODELS[s.model];
  const seg = model.segmentBytes ?? s.custom;
  const elem = info.matrix ? 4 : s.elem;
  const params = (k: number): AccessParams => ({
    Width: s.Width, k, row: 0, stride: s.stride, offset: s.offset, warpWidth: s.warp, lane0: 0,
    blockDimX: s.bdx || undefined, j: s.j, gridThreads: 1024,
  });
  const a = exprAccess(s.expr, params(s.k), seg, elem);
  const lines = model.id === 'nvidia' ? warpAccess(a.lanes.map((l) => l.byte), elem, 128).transactions : null;
  const total = useMemo(() => {
    if (!info.matrix) return null;
    let f = 0;
    let n = 0;
    for (let k = 0; k <= s.k; k++) {
      const x = exprAccess(s.expr, params(k), seg, elem);
      f += x.bytesFetched;
      n += x.bytesNeeded;
    }
    return { fetched: f, needed: n };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s, seg, elem]);
  const cells = info.matrix ? a.lanes.map((l) => ({ r: l.r!, c: l.c! })) : [];

  return (
    <Bench
      title="Coalescing explorer"
      task="Predict the count for each preset, then check it. Change one thing at a time."
      foot={
        <>
          Counts one load instruction of one warp: the distinct aligned segments its lanes touch, in the unit the
          chosen vendor documents ({model.facts}). Arrays start on a 256-byte boundary. No caches, no timing.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem' }}>
        <span className="muted">Presets:</span>
        {PRESETS.map((p) => (
          <button key={p.label} className="quiet" onClick={() => setS({ ...START, warp: s.warp, model: s.model, custom: s.custom, ...p.s })}>
            {p.label}
          </button>
        ))}
      </div>
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <label className="field" style={{ maxWidth: '100%' }}>
          Access
          <select value={s.expr} onChange={(e) => set({ expr: e.target.value as ExprId, k: 0 })} style={{ maxWidth: '100%' }}>
            {EXPR_ORDER.map((x) => <option key={x.id} value={x.id}>{x.label}</option>)}
          </select>
        </label>
        <label className="field" style={{ maxWidth: '100%' }}>
          Warp width
          <select value={s.warp} onChange={(e) => set({ warp: Number(e.target.value) })} style={{ maxWidth: '100%' }}>
            <option value={32}>32 lanes (NVIDIA; AMD RDNA)</option>
            <option value={64}>64 lanes (AMD CDNA)</option>
            {widths.filter((w) => w.width !== 32 && w.width !== 64).map((w) => (
              <option key={w.backend} value={w.width}>{w.width} lanes (your device: {w.device})</option>
            ))}
          </select>
        </label>
        <label className="field" style={{ maxWidth: '100%' }}>
          Segment model
          <select value={s.model} onChange={(e) => set({ model: e.target.value as SegmentModelId })} style={{ maxWidth: '100%' }}>
            {Object.values(SEGMENT_MODELS).map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select>
        </label>
        {model.segmentBytes == null ? (
          <label className="field">
            Your choice (not a fact)
            <select value={s.custom} onChange={(e) => set({ custom: Number(e.target.value) })}>
              {[32, 64, 128].map((v) => <option key={v} value={v}>{v} bytes</option>)}
            </select>
          </label>
        ) : null}
      </div>
      <p className="muted" style={{ margin: '0 0 0.7rem', fontSize: '0.84rem' }} data-device-widths={widths.map((w) => w.width).join(',')}>
        {widths.length
          ? <>Your device{widths.length > 1 ? 's' : ''} reported: {widths.map((w) => `${w.device} (${w.backend}), ${w.width} lanes`).join('; ')}.</>
          : <>No warp width reported by your devices yet; lab 6.1 records it.</>}
      </p>
      <div className="row" style={{ marginBottom: '0.8rem', alignItems: 'flex-end' }}>
        {info.matrix ? (
          <>
            <label className="field">
              Width = {s.Width}
              <input type="range" min={4} max={2048} value={s.Width} aria-label="Width"
                onChange={(e) => { const W = Number(e.target.value); set({ Width: W, k: Math.min(s.k, W - 1) }); }} />
            </label>
            <label className="field">
              Block width (blockDim.x)
              <select value={s.bdx} onChange={(e) => set({ bdx: Number(e.target.value) })}>
                <option value={0}>as wide as the warp, or wider</option>
                <option value={16}>16</option>
                <option value={8}>8</option>
              </select>
            </label>
            <span className="row" style={{ gap: '0.4rem' }}>
              <button onClick={() => set({ k: 0 })} disabled={s.k === 0}>k = 0</button>
              <button onClick={() => set({ k: s.k - 1 })} disabled={s.k === 0}>k − 1</button>
              <button className="primary" onClick={() => set({ k: s.k + 1 })} disabled={s.k >= s.Width - 1}>k + 1</button>
              <span className="muted" aria-live="polite">k = {s.k}</span>
            </span>
          </>
        ) : null}
        {s.expr === 'stride' ? (
          <label className="field">
            stride = {s.stride} element{s.stride === 1 ? '' : 's'}
            <input type="range" min={1} max={64} value={s.stride} aria-label="stride" onChange={(e) => set({ stride: Number(e.target.value) })} />
          </label>
        ) : null}
        {s.expr === 'offset' ? (
          <label className="field">
            offset = {s.offset} element{s.offset === 1 ? '' : 's'}
            <input type="range" min={0} max={32} value={s.offset} aria-label="offset" onChange={(e) => set({ offset: Number(e.target.value) })} />
          </label>
        ) : null}
        {s.expr === 'ex1-c' || s.expr === 'ex1-d' ? (
          <label className="field">
            j
            <select value={s.j} onChange={(e) => set({ j: Number(e.target.value) })}>
              {[0, 1, 2, 3].map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          </label>
        ) : null}
        {!info.matrix ? (
          <label className="field">
            Element size
            <select value={s.elem} onChange={(e) => set({ elem: Number(e.target.value) })}>
              <option value={4}>4 bytes (float)</option>
              <option value={16}>16 bytes (float4)</option>
            </select>
          </label>
        ) : null}
      </div>

      <p className="narrate" style={{ margin: '0 0 0.8rem' }}>
        <code>{info.code}</code>
        {info.matrix
          ? ` with Width = ${s.Width}, k = ${s.k}: lane 0 reads element ${a.lanes[0]?.element ?? '–'}, lane 1 reads ${a.lanes[1]?.element ?? '–'}${a.lanes.length > 1 ? `, ${a.lanes[1].element - a.lanes[0].element} element${Math.abs(a.lanes[1].element - a.lanes[0].element) === 1 ? '' : 's'} further on` : ''}.`
          : `: lane 0 reads element ${a.lanes[0].element}, lane 1 reads ${a.lanes[1].element}, lane ${a.lanes.length - 1} reads ${a.lanes[a.lanes.length - 1].element}.`}
        {a.lanes.length < s.warp ? ` Only ${a.lanes.length} lanes are active: the kernel's guard idles the lanes past column ${s.Width - 1}.` : ''}
      </p>

      {info.matrix ? <MatrixWindow name={s.expr === 'm-row' ? 'M' : 'N'} Width={s.Width} cells={cells} colMajor={s.expr === 'n-col'} /> : null}
      <SegmentStrip lanes={a.lanes} warpWidth={s.warp} elemBytes={elem} seg={seg} unit={model.unit} />

      <div className="row" style={{ marginTop: '0.6rem' }} data-readout
        data-transactions={a.transactions} data-fetched={a.bytesFetched} data-needed={a.bytesNeeded} data-efficiency={a.efficiency} data-runs={a.runs}>
        <span><b>{a.transactions}</b> {model.unit}{a.transactions === 1 ? '' : 's'} of {seg} bytes{model.segmentBytes == null ? ' (your choice)' : ''}</span>
        {lines != null ? <span>in <b>{lines}</b> 128-byte line{lines === 1 ? '' : 's'}</span> : null}
        <span>fetched <b>{bytes(a.bytesFetched)}</b> bytes</span>
        <span>needed <b>{bytes(a.bytesNeeded)}</b></span>
        <span className={'chip ' + (a.efficiency === 1 ? 'ok' : a.efficiency <= 0.25 ? 'fail' : 'global')}>{pct(a.efficiency)} of fetched bytes used</span>
        <span className="muted">{a.runs === 1 ? 'one run of consecutive bytes' : `${a.runs} separate runs`}</span>
      </div>
      {total && s.k > 0 ? (
        <p className="muted" style={{ margin: '0.5rem 0 0' }} data-total-fetched={total.fetched} data-total-needed={total.needed}>
          Over k = 0 to {s.k}, counting each step afresh: fetched {bytes(total.fetched)} bytes for {bytes(total.needed)} needed
          {total.fetched !== total.needed ? ` (${fmt(total.fetched / total.needed)} times as much)` : ''}.
        </p>
      ) : null}
      {model.id === 'metal' ? (
        <p className="muted" style={{ margin: '0.5rem 0 0' }}>
          Apple documents no coalescing rule, line size or sector size (facts C1). The count above uses the size you picked; lab 6.1 is how to find out what a Mac does.
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- corner-turning stepper

const TILES = [4, 8, 16, 32];

function CornerTurnStepper() {
  const [layout, setLayout] = useState<Layout>('col');
  const [arr, setArr] = useState<Arrangement>('ch5');
  const [T, setT] = useState(4);
  const [warpW, setWarpW] = useState(4);
  const [Width, setWidth] = useState(16);
  const [bx, setBx] = useState(1);
  const [ph, setPh] = useState(2);
  const [w, setW] = useState(0);
  const [tx, setTx] = useState(3);
  const [ty, setTy] = useState(1);

  const widthsOk = [16, 64, 1024].filter((v) => v >= 2 * T);
  const W = widthsOk.includes(Width) ? Width : widthsOk[0];
  const nb = Math.min(W / T, 8);
  const BX = Math.min(bx, nb - 1);
  const PH = Math.min(ph, nb - 1);
  const ww = T === 4 ? warpW : Math.max(32, warpW);
  const load = tileLoad({ layout, arrangement: arr, T, bx: BX, ph: PH, Width: W, warpWidth: ww });
  const nw = load.warps.length;
  const cw = load.warps[Math.min(w, nw - 1)];
  const TX = Math.min(tx, T - 1);
  const TY = Math.min(ty, T - 1);
  const sel = load.threads.find((t) => t.tx === TX && t.ty === TY)!;
  const nb1 = load.threads.find((t) => t.tx === Math.max(0, TX - 1) && t.ty === TY)!;
  const inWarp = new Set(cw.threads.map((t) => `${t.sharedRow},${t.sharedCol}`));
  const cell = Math.min(26, 360 / T);
  const store =
    ww >= 32
      ? tileAccess({ T, pitch: T, orientation: arr === 'turned' ? 'x-first' : 'y-first', ty0: cw.threads[0].ty, tx0: cw.threads[0].tx, warpWidth: cw.threads.length })
      : null;
  const turned = arr === 'turned';
  const rowExpr = turned ? 'ph·T + tx' : 'ph·T + ty';
  const colExpr = turned ? 'bx·T + ty' : 'bx·T + tx';

  return (
    <Bench
      title="Corner-turning stepper"
      task="Step through the warps of one block's N-tile load in both arrangements. Which one gives each warp one run of addresses?"
      foot={
        <>
          The N tile of one block in one phase of the tiled kernel. Thread (tx, ty) is lane (ty·T + tx) mod the warp width.
          Sectors are NVIDIA's 32 bytes. The four-thread warp exists only in the book's figure; real warps are 32 or 64 lanes.
        </>
      }
    >
      <div className="row" style={{ marginBottom: '0.7rem', alignItems: 'flex-end' }}>
        <span className="seg" role="group" aria-label="Arrangement">
          <button aria-pressed={!turned} onClick={() => setArr('ch5')}>(a) chapter 5's mapping</button>
          <button aria-pressed={turned} onClick={() => setArr('turned')}>(b) corner turned</button>
        </span>
        <label className="field">
          N is stored
          <select value={layout} onChange={(e) => setLayout(e.target.value as Layout)}>
            <option value="col">column by column</option>
            <option value="row">row by row</option>
          </select>
        </label>
        <label className="field">
          Tile width T
          <select value={T} onChange={(e) => { const v = Number(e.target.value); setT(v); setW(0); if (v !== 4 && warpW < 32) setWarpW(32); }}>
            {TILES.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className="field">
          Warp width
          <select value={ww} onChange={(e) => { setWarpW(Number(e.target.value)); setW(0); }}>
            {T === 4 ? <option value={4}>4 (the figure's toy)</option> : null}
            <option value={32}>32</option>
            <option value={64}>64</option>
          </select>
        </label>
        <label className="field">
          Width
          <select value={W} onChange={(e) => setWidth(Number(e.target.value))}>
            {widthsOk.map((v) => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label className="field">
          bx
          <select value={BX} onChange={(e) => setBx(Number(e.target.value))}>
            {Array.from({ length: nb }, (_, i) => <option key={i} value={i}>{i}</option>)}
          </select>
        </label>
        <label className="field">
          phase
          <select value={PH} onChange={(e) => setPh(Number(e.target.value))}>
            {Array.from({ length: nb }, (_, i) => <option key={i} value={i}>{i}</option>)}
          </select>
        </label>
      </div>
      <div className="row" style={{ marginBottom: '0.8rem' }}>
        <button onClick={() => setW(Math.max(0, cw.warp - 1))} disabled={cw.warp === 0}>Previous warp</button>
        <button className="primary" onClick={() => setW(Math.min(nw - 1, cw.warp + 1))} disabled={cw.warp >= nw - 1}>Next warp</button>
        <span className="muted" aria-live="polite">warp {cw.warp + 1} of {nw}</span>
        <label className="field">
          Thread tx
          <select value={TX} onChange={(e) => setTx(Number(e.target.value))}>
            {Array.from({ length: T }, (_, i) => <option key={i} value={i}>{i}</option>)}
          </select>
        </label>
        <label className="field">
          ty
          <select value={TY} onChange={(e) => setTy(Number(e.target.value))}>
            {Array.from({ length: T }, (_, i) => <option key={i} value={i}>{i}</option>)}
          </select>
        </label>
      </div>

      <p className="narrate" style={{ margin: '0 0 0.8rem' }} data-thread-element={sel.element}>
        Thread (tx {TX}, ty {TY}) loads N({rowExpr}, {colExpr}) = N({PH}·{T} + {turned ? TX : TY}, {BX}·{T} + {turned ? TY : TX}) = N({sel.r}, {sel.c}).{' '}
        {layout === 'col'
          ? <>Column by column, that is column·Width + row = {sel.c}·{W} + {sel.r} = <b>{sel.element}</b>.</>
          : <>Row by row, that is row·Width + column = {sel.r}·{W} + {sel.c} = <b>{sel.element}</b>.</>}
        {TX > 0 ? <> Its neighbour tx {TX - 1} loads {nb1.element}, {Math.abs(sel.element - nb1.element)} element{Math.abs(sel.element - nb1.element) === 1 ? '' : 's'} away.</> : null}{' '}
        It stores the value at Nds[{sel.sharedRow}][{sel.sharedCol}], which is Nds[{turned ? 'tx' : 'ty'}][{turned ? 'ty' : 'tx'}].
      </p>

      <div className="cols" style={{ gridTemplateColumns: 'minmax(0, 2fr) minmax(0, 3fr)' }}>
        <div>
          <div className="mx-label t-global">the N tile (rows {PH * T}–{PH * T + T - 1}, columns {BX * T}–{BX * T + T - 1})</div>
          <svg viewBox={`0 0 ${T * cell + 2} ${T * cell + 2}`} style={{ width: '100%', maxWidth: T * cell + 2, display: 'block' }} role="img" aria-label="The tile, with the current warp's elements marked">
            {[...load.threads.filter((t) => t !== sel), sel].map((t) => {
              const me = t === sel;
              const on = inWarp.has(`${t.sharedRow},${t.sharedCol}`);
              return (
                <g key={t.t}>
                  <rect x={1 + t.sharedCol * cell} y={1 + t.sharedRow * cell} width={cell - 1.5} height={cell - 1.5}
                    fill={on ? 'var(--global)' : 'var(--global-wash)'} stroke={me ? 'var(--ink)' : 'var(--rule)'} strokeWidth={me ? 2.4 : 0.5} />
                  {T === 4 ? (
                    <text x={1 + t.sharedCol * cell + cell / 2} y={1 + t.sharedRow * cell + cell / 2 + 3} fontSize="8" textAnchor="middle"
                      fontWeight={me ? 700 : 400} style={{ fill: on ? 'var(--page)' : 'var(--ink)' }}>{t.element}</text>
                  ) : null}
                </g>
              );
            })}
          </svg>
          <p className="muted" style={{ fontSize: '0.8rem', margin: '0.3rem 0 0' }}>
            Shaded: elements this warp loads. Outlined: the selected thread's element. {T === 4 ? 'Numbers are flat indices into N.' : ''}
          </p>
        </div>
        <div>
          <div className="row" data-readout data-transactions={cw.access.transactions} data-efficiency={cw.access.efficiency} data-runs={cw.runs}>
            <span><b>{cw.access.transactions}</b> sector{cw.access.transactions === 1 ? '' : 's'} for this warp</span>
            <span className={'chip ' + (cw.access.efficiency === 1 ? 'ok' : cw.access.efficiency <= 0.25 ? 'fail' : 'global')}>{pct(cw.access.efficiency)} used</span>
            <span className="muted">{cw.runs === 1 ? 'one run of consecutive addresses' : `${cw.runs} separate runs`}</span>
          </div>
          {store ? (
            <p className="muted" style={{ margin: '0.6rem 0 0' }} data-store-passes={store.passes}>
              The store into the shared tile (<code>{turned ? 'Nds[tx][ty]' : 'Nds[ty][tx]'}</code>, {T} words per row) touches{' '}
              {store.banksUsed} bank{store.banksUsed === 1 ? '' : 's'}:{' '}
              {store.passes === 1 ? 'one pass, no conflict' : <b className="t-fail">{store.passes} passes under NVIDIA's bank rule</b>}.{' '}
              {turned && store.passes > 1 ? <>Section <a href={href('ch06', '6.4')}>6.4</a> is about this.</> : null}
            </p>
          ) : (
            <p className="muted" style={{ margin: '0.6rem 0 0' }}>
              With a four-thread warp there is no bank count to show; pick a warp width of 32 to see the shared-memory store.
            </p>
          )}
        </div>
      </div>
      <SegmentStrip lanes={cw.threads.map((t) => ({ lane: t.lane, byte: t.byte }))} warpWidth={ww} elemBytes={4} seg={32} unit="sector" />
    </Bench>
  );
}

// ---------------------------------------------------------------- Exercise 1 drill

const VERDICTS: Ex1Verdict[] = ['coalesced', 'uncoalesced', 'not applicable'];

function Exercise1Drill() {
  const ex = exercise1();
  const [pick, setPick] = useState<Record<string, Ex1Verdict | undefined>>({});
  const [checked, setChecked] = useState(false);
  const right = ex.filter((x) => pick[x.id] === x.verdict).length;
  const h = strideCase(4);
  return (
    <Bench
      title="Exercise 1 drill"
      task="Classify all nine accesses, then check."
      foot="Sector counts are for a 32-lane warp whose lane 0 has i = 0, in NVIDIA's 32-byte unit, with blockDim.x·gridDim.x = 1024. Your answers here are not recorded."
    >
      <table className="data" style={{ background: 'var(--page)', margin: 0 }}>
        <thead>
          <tr><th style={{ paddingLeft: '0.6rem' }}>Access</th><th>Line</th><th>Your call</th>{checked ? <th>Model</th> : null}</tr>
        </thead>
        <tbody>
          {ex.map((x) => (
            <tr key={x.id} data-access={x.id} data-result={checked ? (pick[x.id] === x.verdict ? 'right' : 'wrong') : ''}>
              <td style={{ paddingLeft: '0.6rem' }}>({x.id}) <code>{x.code}</code></td>
              <td>{String(x.line).padStart(2, '0')}</td>
              <td>
                <span className="seg" role="group" aria-label={`Access (${x.id})`}>
                  {VERDICTS.map((v) => (
                    <button key={v} aria-pressed={pick[x.id] === v} onClick={() => { setPick({ ...pick, [x.id]: v }); setChecked(false); }}>{v}</button>
                  ))}
                </span>
              </td>
              {checked ? (
                <td>
                  <span className={'chip ' + (pick[x.id] === x.verdict ? 'ok' : 'fail')}>{x.verdict}</span>{' '}
                  <span className="muted">
                    {x.space === 'shared'
                      ? x.sharedStride === 4
                        ? `shared memory needs no coalescing; section 6.4 will count this one as ${h.passes}-way on ${h.banksUsed} banks`
                        : 'shared memory needs no coalescing'
                      : `${x.transactions} sectors, ${pct(x.efficiency!)} used${x.runs === 1 ? '' : `, ${x.runs} runs`}`}
                  </span>
                </td>
              ) : null}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="row" style={{ marginTop: '0.8rem' }}>
        <button className="primary" onClick={() => setChecked(true)} disabled={Object.keys(pick).length < ex.length}>Check my answers</button>
        {checked ? <span aria-live="polite" data-score={right}>{right} of {ex.length} right.</span> : <span className="muted">{Object.keys(pick).length} of {ex.length} answered</span>}
      </div>
      {checked ? (
        <p style={{ margin: '0.7rem 0 0' }}>
          The rule behind every global line: is the index the same expression plus <code>i</code> for every lane, with a
          coefficient of 1 on <code>i</code>? A term that does not depend on the lane (<code>j·blockDim.x·gridDim.x</code>,{' '}
          <code>+ 8</code>) shifts the whole run and costs at most one extra segment when it breaks alignment. A coefficient
          on <code>i</code> (<code>i*4</code>, <code>i*8</code>) spreads the lanes apart.
        </p>
      ) : null}
    </Bench>
  );
}

// ---------------------------------------------------------------- your device

const STRIDES = [1, 2, 4, 8, 16, 32, 64];
const SEGS = [32, 64, 128];

function YourDevice61() {
  const { reports, loading } = useReports();
  const found = gpuBackends(reports)
    .map((b) => ({ b, device: String(reports[b].device.name ?? b), rows: labRows(reports, b, LAB61) }))
    .filter((x) => x.rows.length);
  const W = 680, H = 230, L = 50, R = 170, TOP = 14, BOT = 34;
  const xOf = (s: number) => L + (Math.log2(s) / 6) * (W - L - R);
  const yOf = (v: number) => TOP + (1 - Math.min(1.1, v) / 1.1) * (H - TOP - BOT);
  const measured = found.map((f) => {
    const st = f.rows.filter((r) => r.name === 'strided' && num(r.stride) != null).sort((p, q) => Number(p.stride) - Number(q.stride));
    const base = num(st.find((r) => r.stride === 1)?.gb_per_s);
    return {
      ...f,
      points: st.map((r) => ({ s: Number(r.stride), v: num(r.relative_to_stride1) ?? (base ? (num(r.gb_per_s) ?? 0) / base : NaN) })).filter((p) => isFinite(p.v)),
      copy: num(f.rows.find((r) => r.name === 'copy')?.gb_per_s),
      readRows: f.rows.find((r) => r.name === 'transpose' && (r.variant === 'read_rows' || r.kernel === 'read_rows')),
      writeRows: f.rows.find((r) => r.name === 'transpose' && (r.variant === 'write_rows' || r.kernel === 'write_rows')),
    };
  });
  const dash: Record<number, string> = { 32: '6 3', 64: '2 3', 128: '10 3 2 3' };
  return (
    <Bench
      title="Your device: coalescing"
      task="Where do your measurements sit against the three segment models?"
      foot="Model curves: fraction of fetched bytes a warp of 32 floats uses at each stride, from the same model as the explorer. Measured: lab 6.1's useful bandwidth relative to stride 1, which also includes the (coalesced) write and whatever the caches did. They are not the same quantity; compare shapes, not values."
    >
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Efficiency against stride: models and your measurements" style={{ width: '100%', maxWidth: W, display: 'block' }}>
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={yOf(v)} y2={yOf(v)} stroke="var(--rule-soft)" />
            <text x={L - 6} y={yOf(v) + 3} fontSize="10" textAnchor="end" style={{ fill: 'var(--ink-2)' }}>{v}</text>
          </g>
        ))}
        {STRIDES.map((s) => (
          <text key={s} x={xOf(s)} y={H - BOT + 16} fontSize="10" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>{s}</text>
        ))}
        <text x={(L + W - R) / 2} y={H - 4} fontSize="10.5" textAnchor="middle" style={{ fill: 'var(--ink-2)' }}>stride (floats)</text>
        {SEGS.map((g) => {
          const pts = STRIDES.map((s) => `${xOf(s)},${yOf(strideEfficiency(s, g))}`).join(' ');
          return (
            <g key={g} data-model={g}>
              <polyline points={pts} fill="none" stroke="var(--ink-3)" strokeWidth="1.4" strokeDasharray={dash[g]} />
              <line x1={W - R + 8} x2={W - R + 34} y1={TOP + 10 + SEGS.indexOf(g) * 16} y2={TOP + 10 + SEGS.indexOf(g) * 16} stroke="var(--ink-3)" strokeWidth="1.4" strokeDasharray={dash[g]} />
              <text x={W - R + 38} y={TOP + 14 + SEGS.indexOf(g) * 16} fontSize="10" style={{ fill: 'var(--ink-2)' }}>model, {g} B</text>
            </g>
          );
        })}
        {measured.map((m) => (
          <g key={m.b} data-measured={m.b}>
            <polyline points={m.points.map((p) => `${xOf(p.s)},${yOf(p.v)}`).join(' ')} fill="none" stroke="var(--global)" strokeWidth="2" />
            {m.points.map((p) => <circle key={p.s} cx={xOf(p.s)} cy={yOf(p.v)} r="3.2" fill="var(--global)" />)}
            <line x1={W - R + 8} x2={W - R + 34} y1={TOP + 62 + measured.indexOf(m) * 16} y2={TOP + 62 + measured.indexOf(m) * 16} stroke="var(--global)" strokeWidth="2" />
            <text x={W - R + 38} y={TOP + 66 + measured.indexOf(m) * 16} fontSize="10" style={{ fill: 'var(--global)' }}>{m.device}</text>
          </g>
        ))}
      </svg>
      {loading ? null : !measured.length ? (
        <p data-no-measurement>
          No measurement yet. Run <code>python3 labs/pmpp.py run 6.1 --solution --bench</code> on a machine with a GPU
          (or <code>run 6.1 --bench</code> once your own kernels pass) and reload: your stride sweep joins the model
          curves, and the two naive transposes appear below.
        </p>
      ) : (
        <table className="data" style={{ background: 'var(--page)', margin: '0.6rem 0 0' }}>
          <thead>
            <tr><th style={{ paddingLeft: '0.6rem' }}>Device</th><th className="num">copy GB/s</th><th className="num">transpose, strided stores</th><th className="num">transpose, strided loads</th><th className="num">time, strided stores ÷ strided loads</th></tr>
          </thead>
          <tbody>
            {measured.map((m) => {
              const tr = num(m.readRows?.seconds);
              const tw = num(m.writeRows?.seconds);
              return (
                <tr key={m.b} data-device={m.b}>
                  <td style={{ paddingLeft: '0.6rem' }}>{m.device} ({m.b})</td>
                  <td className="num">{m.copy != null ? fmt(m.copy) : '–'}</td>
                  <td className="num">{num(m.readRows?.gb_per_s) != null ? `${fmt(num(m.readRows?.gb_per_s)!)} GB/s` : '–'}</td>
                  <td className="num">{num(m.writeRows?.gb_per_s) != null ? `${fmt(num(m.writeRows?.gb_per_s)!)} GB/s` : '–'}</td>
                  <td className="num">{tr && tw ? `${fmt(tr / tw)}×` : '–'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </Bench>
  );
}

function TransposeReveal({ predicted }: { predicted: number }) {
  const { reports, loading } = useReports();
  const found = gpuBackends(reports).flatMap((b) => {
    const rows = labRows(reports, b, LAB61);
    const tr = num(rows.find((r) => r.name === 'transpose' && (r.variant === 'read_rows' || r.kernel === 'read_rows'))?.seconds);
    const tw = num(rows.find((r) => r.name === 'transpose' && (r.variant === 'write_rows' || r.kernel === 'write_rows'))?.seconds);
    return tr && tw ? [{ b, device: String(reports[b].device.name ?? b), ratio: tr / tw }] : [];
  });
  if (loading) return null;
  if (!found.length)
    return (
      <p data-no-measurement>
        No measurement on this machine yet. Run <code>python3 labs/pmpp.py run 6.1 --solution --bench</code> and reload:
        this box will compare your prediction with the two transposes' times.
      </p>
    );
  return (
    <>
      {found.map((f) => (
        <p key={f.b} data-ratio={f.ratio}>
          On <b>{f.device}</b> ({f.b}) the transpose with strided stores took <b>{fmt(f.ratio)}×</b> the time of the one with
          strided loads. {Math.abs(Math.log(predicted / f.ratio)) < Math.log(1.25) ? 'That is close to what you predicted.' : predicted > f.ratio ? 'You expected strided stores to cost more than they did here.' : 'You expected strided stores to cost less than they did here.'}
        </p>
      ))}
      <p className="muted">A measurement from your machine, not an expected value. Caches, write combining and the memory type all play a part, and another device can differ in either direction.</p>
    </>
  );
}

// ---------------------------------------------------------------- questions

export const questionsS61: Record<string, Question> = {
  'ch06/6.1/q-align': {
    kind: 'numeric',
    id: 'ch06/6.1/q-align',
    prompt: (
      <>
        A 32-lane warp reads 32 consecutive floats, but the first one starts one float past a 32-byte boundary. How
        many 32-byte sectors does the load touch?
      </>
    ),
    answer: 5,
    explain: (
      <p>
        The warp needs bytes 4 to 131 of the aligned region. Sectors cover 0–31, 32–63, 64–95, 96–127 and 128–159: the
        last four bytes spill into a fifth. NVIDIA's guide gives the same count for a sequential but misaligned warp.
        Memory from <code>cudaMalloc</code> starts on a 256-byte boundary, so the misalignment comes from the index, not
        the allocation.
      </p>
    ),
    source: 'book §6.1 p. 126; notes/today/ch06-facts.md A3',
  },
  'ch06/6.1/q-stride2': {
    kind: 'numeric',
    id: 'ch06/6.1/q-stride2',
    prompt: (
      <>
        Lane <code>i</code> of a 32-lane warp reads <code>in[2*i]</code> (floats, aligned start). What percentage of the
        bytes fetched does the warp use, in NVIDIA's 32-byte sectors?
      </>
    ),
    answer: 50,
    unit: '%',
    explain: (
      <p>
        The 32 lanes span 256 bytes, eight sectors, and use every other float in each: 128 of 256 bytes. NVIDIA's guide
        states 50 % for stride 2. Push the stride to 8 floats or more and each lane has a sector to itself: 4 bytes used
        of 32, 12.5 %, the worst case.
      </p>
    ),
    source: 'notes/today/ch06-facts.md A2',
  },
  'ch06/6.1/q-mread': {
    kind: 'choice',
    id: 'ch06/6.1/q-mread',
    prompt: (
      <>
        In the basic matmul kernel with blocks at least 32 threads wide, every lane of a warp reads the same{' '}
        <code>M[row*Width + k]</code>. Is that a coalescing problem?
      </>
    ),
    options: [
      { label: 'Yes: 32 lanes issue 32 requests, one per lane, even at a single address.', why: 'The lanes share one address. The hardware counts the segments touched, not the lanes.' },
      { label: 'No: one sector serves the whole warp, which needs only 4 bytes.', correct: true },
      { label: 'Yes: the addresses are not consecutive, so the hardware cannot merge them.', why: 'They are identical, which is the cheapest case of all: one segment.' },
    ],
    explain: (
      <p>
        The book does not analyse the M read in this section, because it is not a problem. With blocks 16 threads wide a
        warp covers two rows of threads, so it reads two different elements of M: two addresses, two sectors. Set the
        explorer's block width to 16 to see it.
      </p>
    ),
    source: 'book §6.1 pp. 127–128, Figs. 6.2–6.3; our analysis with NVIDIA\'s rule (facts A2)',
  },
  'ch06/6.1/q-linear': {
    kind: 'choice',
    id: 'ch06/6.1/q-linear',
    prompt: (
      <>
        <code>W</code> is stored as <code>nn.Linear</code> stores it, shape (out, in), row-major. Each thread owns one
        output feature <code>o</code> of one batch row and, in its loop over <code>k</code>, reads{' '}
        <code>W[o*in + k]</code>. Neighbouring lanes have neighbouring <code>o</code>. Is the read coalesced?
      </>
    ),
    options: [
      { label: 'Yes: each thread reads consecutive addresses as k advances.', why: 'Coalescing is about the lanes of one instruction, not about one thread over time. At a fixed k the lanes are far apart.' },
      { label: 'Only when in is a multiple of 32, so that every row of W starts on a sector boundary.', why: 'A multiple of 32 keeps rows aligned, but the lanes are still in elements apart.' },
      { label: 'No: at every k the lanes are in elements apart. This is Fig. 6.3.', correct: true },
    ],
    explain: (
      <p>
        One thread walking along its row is what a CPU programmer looks at, and it is the wrong view here. Two fixes: store
        the transposed weight once when the model loads (a layout change), or corner-turn the tile of <code>W</code> as
        it is loaded, which is lab 6.4.
      </p>
    ),
    source: 'book §6.1 pp. 128–129; notes/today/ch03-facts.md D6 for nn.Linear\'s layout',
  },
  'ch06/6.1/q-cornerturn': {
    kind: 'numeric',
    id: 'ch06/6.1/q-cornerturn',
    prompt: (
      <>
        The toy of Fig. 6.4: N is 16 × 16 and stored column by column (element (r, c) at <code>c*16 + r</code>),{' '}
        <code>T = 4</code>, block <code>bx = 1</code>, phase <code>ph = 2</code>. With corner turning, which flat index
        does thread <code>tx = 3, ty = 1</code> load?
      </>
    ),
    answer: 91,
    explain: (
      <p>
        Turned, the thread takes row ph·T + tx = 2·4 + 3 = 11 and column bx·T + ty = 1·4 + 1 = 5: index 5·16 + 11 = 91.
        Its neighbour tx = 2 loads 90, one element away. Under chapter 5's mapping the same two threads load rows
        ph·T + ty = 9 and columns 7 and 6: indices 121 and 105, sixteen apart.
      </p>
    ),
    source: 'book §6.1 pp. 129–130, Fig. 6.4',
  },
  'ch06/6.1/q-ex3': {
    kind: 'choice',
    id: 'ch06/6.1/q-ex3',
    prompt: (
      <>
        Exercise 3: in chapter 5's tiled matmul, which square tile widths avoid uncoalesced accesses to global memory?
      </>
    ),
    options: [
      { label: 'Any width: every tile row is read consecutively.', why: 'A tile 4 floats wide fills half of each sector it touches. Set T = 4 with N row by row in the stepper.' },
      {
        label: 'It depends on the reading. If each warp\'s load must be one consecutive run, only 32. If what counts is fetched bytes wasted in NVIDIA\'s 32-byte sectors, any multiple of 8 floats with aligned rows wastes nothing, 8 and 16 included.',
        correct: true,
      },
      { label: 'Only 16: half a warp per tile row, so each warp reads two whole rows.', why: 'Two rows are two runs, so the strict reading wants 32; under the sector reading 8 wastes no more bytes than 16. Nothing singles out 16.' },
    ],
    explain: (
      <>
        <p>
          The book's own remark that real corner-turning tiles are as wide as a warp (p. 130) supports the first
          reading; with the 1024-thread limit, 32 is also the largest square tile. NVIDIA's documented unit supports the
          second (facts A2): a 16-float row is two whole sectors. Extending that to 8 is our arithmetic from the 32-byte
          rule.
        </p>
        <p>In the stepper, store N row by row, keep chapter 5's mapping and try T = 8, 16 and 32.</p>
      </>
    ),
    source: 'book p. 155 (exercise 3), p. 130; notes/today/ch06-facts.md A2',
  },
  'ch06/6.1/q-store': {
    kind: 'choice',
    id: 'ch06/6.1/q-store',
    prompt: (
      <>
        Your naive transpose reads rows of the input and writes columns of the output, and it runs far below your copy
        kernel&rsquo;s bandwidth. Before you rewrite it, what tells you whether the strided stores are what costs the
        most?
      </>
    ),
    options: [
      {
        label: 'Time the mirror image, which reads columns and writes rows: the same bytes with the stride moved to the loads.',
        correct: true,
      },
      {
        label: 'Nothing: the book says uncoalesced stores hurt more than loads, so fix the stores first.',
        why: 'The book says they can hurt more (p. 130). NVIDIA ties the extra cost to ECC on GDDR memory and says HBM2 pays much less, so the answer depends on the device.',
      },
      {
        label: 'Count store instructions against load instructions: the side with more of them costs more.',
        why: 'Both kernels issue one load and one store per element. What differs is how many sectors each warp-wide store touches, not how many instructions run.',
      },
      {
        label: 'Run it on the emulator and read which strided stores its report flags.',
        why: 'The emulators check correctness. They model no time, caches or coalescing, so cost needs a real device.',
      },
    ],
    explain: (
      <p>
        Lab 6.1 runs exactly that pair. What the vendors document is uneven. For GDDR memory with ECC on, NVIDIA says:
        &ldquo;Scattered accesses increase ECC memory transfer overhead, especially when writing data to global
        memory.&rdquo; It says HBM2&rsquo;s ECC costs much less, and Nsight Compute flags a store pattern
        &ldquo;significantly exceeding expected sector count&rdquo;. The read-modify-write the book names is described
        in none of the pages we read. Whichever side costs more on your device, corner turning through shared memory
        lets both the loads and the stores run along rows.
      </p>
    ),
    source: 'book §6.1 p. 130; notes/today/ch06-facts.md A5; labs/README.md for what the emulators model',
  },
  'ch06/6.1/q-ex1c': {
    kind: 'choice',
    id: 'ch06/6.1/q-ex1c',
    prompt: <>Exercise 1(c): <code>b[j*blockDim.x*gridDim.x + i]</code> on line 07, for a fixed <code>j</code>. Coalesced?</>,
    options: [
      { label: 'Uncoalesced, because of the multiplication by blockDim.x·gridDim.x.', why: 'The product does not involve the lane. Only a coefficient on i spreads the lanes apart.' },
      { label: 'Coalesced: the j term is the same for every lane.', correct: true },
      { label: 'Coalescing does not apply.', why: 'b is a pointer into global memory.' },
    ],
    explain: <p>For each j the warp reads 32 consecutive floats starting at j·blockDim.x·gridDim.x. In NVIDIA's unit: four sectors, all used.</p>,
    source: 'book p. 155 (exercise 1); notes/ch06.md',
  },
  'ch06/6.1/q-ex1d': {
    kind: 'choice',
    id: 'ch06/6.1/q-ex1d',
    prompt: <>Exercise 1(d): <code>c[i*4 + j]</code> on line 07. Coalesced?</>,
    options: [
      { label: 'Uncoalesced: neighbouring lanes are 4 floats apart.', correct: true },
      { label: 'Coalesced, because j is the same for every lane and the loop reads every byte.', why: 'j is, but i is multiplied by 4. Over the loop every byte is read, but coalescing is judged one instruction at a time.' },
      { label: 'Coalescing does not apply to an index that contains a loop variable.', why: 'c is in global memory; what decides is how the index changes from lane to lane.' },
    ],
    explain: (
      <p>
        One instruction: 16 sectors for 128 useful bytes. Over the four values of j the loop does read every byte it
        fetched, which is why a cache might rescue it; the model counts each instruction afresh. The full drill:
        a coalesced, b not applicable, c coalesced, d uncoalesced, e, f and h not applicable, g coalesced, i uncoalesced.
      </p>
    ),
    source: 'book p. 155 (exercise 1); notes/ch06.md',
  },
  'ch06/6.1/q-ex1g': {
    kind: 'choice',
    id: 'ch06/6.1/q-ex1g',
    prompt: <>Exercise 1(g): <code>d[i + 8]</code> on line 10. Coalesced?</>,
    options: [
      { label: 'Uncoalesced, because the offset of 8 misaligns every lane\'s address.', why: 'Every lane gets the same offset, so the run stays consecutive; at worst it straddles one more segment.' },
      { label: 'Coalescing does not apply: line 10 is a store, not a load.', why: 'd is in global memory, and stores coalesce by the same rule (p. 130).' },
      { label: 'Coalesced; the offset can shift the run across one more segment.', correct: true },
    ],
    explain: (
      <p>
        Eight floats are exactly one 32-byte sector, so in NVIDIA's unit nothing extra is fetched: four sectors. In
        128-byte lines the same run straddles two lines instead of one. Try it with the explorer's segment model.
      </p>
    ),
    source: 'book p. 155 (exercise 1), p. 126 on alignment; facts A2, A3',
  },
};

// ---------------------------------------------------------------- the section

export function S61() {
  const nCol = exprAccess('n-col', { Width: 1024, k: 0, row: 0, stride: 1, offset: 0, warpWidth: 32, lane0: 0 }, 32);
  const nRow = exprAccess('n-row', { Width: 1024, k: 0, row: 0, stride: 1, offset: 0, warpWidth: 32, lane0: 0 }, 32);
  const amd = exprAccess('n-row', { Width: 1024, k: 0, row: 0, stride: 1, offset: 0, warpWidth: 64, lane0: 0 }, 128);
  return (
    <>
      <p>
        Chapter 5 cut how much data a kernel fetches from global memory. This section is about how efficiently each
        fetch happens. The two work together: tiling decides how many bytes cross the memory bus, coalescing decides
        how many trips they take.
      </p>
      <Source section="6.1" pages="124" />

      <h2>Why DRAM wants neighbours</h2>
      <p>
        Global memory is DRAM. Each bit is a tiny capacitor. Reading it means letting that small charge pull a long wire,
        the bit line, shared by about a thousand other cells, far enough for a sense amplifier to decide whether the
        charge was there.
      </p>
      <Source section="6.1" pages="124–125" figs="sidebar “Why are DRAMs so slow?”" />
      <p>
        The book puts this at tens of nanoseconds, against clock cycles well under one nanosecond. Bigger capacitors
        would be faster, but every generation makes them smaller to fit more bits on a chip, so, the book argues, the
        latency has not come down.
      </p>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md, A1)">
        <p>
          No vendor page we read gives a DRAM latency in nanoseconds or a trend. NVIDIA's guide puts uncached global
          accesses at “hundreds of clock cycles”. The trend is the book's claim.
        </p>
      </Today>
      <p>
        So DRAM is built for parallelism instead. Reading one location senses a whole run of its neighbours at once, many
        amplifiers working side by side, and the run then goes out to the processor quickly. That run is a{' '}
        <b>burst</b>. A program that uses whole bursts gets far more out of the memory than one that takes a few bytes from
        each of many scattered places.
      </p>
      <Source pages="125" />

      <h2>Caches move whole lines</h2>
      <p>
        GPUs also have L1 and L2 caches, built from SRAM. The book has kept them in the background because they are not
        a prominent part of the CUDA programming model, but they shape what you measure.
      </p>
      <Source section="6.1" pages="125–126" />
      <p>
        Data moves between them in fixed-size <b>cache lines</b>, so even after data is cached, a warp that uses every
        byte of the lines it touches moves fewer lines than one that does not.
      </p>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md, A4, B1, C1)">
        <p>
          NVIDIA: L1 and L2 lines are 128 bytes, made of four 32-byte sectors, and on compute capability 6.0 and later
          the unit of access is the 32-byte sector whether or not a load is cached in L1. AMD: 64- and/or 128-byte lines. Apple documents no line or sector
          size.
        </p>
      </Today>

      <h2>The rule</h2>
      <p>
        All lanes of a warp execute the same load at the same time, each with its own address. The hardware looks at the
        addresses together. If they fall on consecutive locations within the same burst or cache line, it merges them
        into one consolidated access, and the load is <b>coalesced</b>. Lane 0 reading location X, lane 1 reading X + 1,
        lane 2 reading X + 2 is the best case. Alignment is a second condition: when the run starts at the beginning of a
        burst or line, it needs the minimum number of transactions; when it does not, the same bytes straddle one more.
      </p>
      <Source section="6.1" pages="126" />
      <p>
        The book speaks of bursts and lines without giving a size. The explorer below counts in each vendor's documented
        unit and says which one it is using.
      </p>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md, A2, A3)">
        <p>
          NVIDIA states the rule in 32-byte transactions. 32 consecutive floats: 4. The same run off a 32-byte boundary:
          5. Stride 2: half of each fetched byte used. Lanes 32 bytes or more apart: one transaction each, 12.5 %.
          Consecutive lane order is “merely the typical way”: any permutation within the same sectors costs the same.
        </p>
      </Today>

      <h2>Count the transactions</h2>
      <p>
        Back to the basic matrix multiplication of chapter 3. In the k loop each thread reads{' '}
        <code>M[row*Width + k]</code> and <code>N[k*Width + col]</code>. Within a warp, <code>row</code>,{' '}
        <code>k</code> and <code>Width</code> are the same for every lane and <code>col</code> goes up by one from lane
        to lane, so the N addresses are consecutive: at each k the warp reads a piece of one row of N. The book adds that
        every kernel so far has coalesced this way without trying.
      </p>
      <Source section="6.1" pages="127–128" figs="Fig. 6.2" />
      <p>
        Now suppose N is stored column by column, so the read becomes <code>N[col*Width + k]</code>. Neighbouring lanes
        are Width elements apart, hundreds or thousands for real matrices.
      </p>
      <Source pages="128–129" figs="Fig. 6.3" />
      <Predict
        id="ch06/6.1/p-sectors"
        prompt={
          <>
            Width = 1024, N stored column by column, a 32-lane warp executes <code>N[col*Width + k]</code> for one k. How
            many 32-byte sectors does it touch?
          </>
        }
        unit="sectors"
      >
        {(v) => (
          <p>
            The model says <b>{nCol.transactions}</b>: the lanes are {bytes(nCol.lanes[1].byte - nCol.lanes[0].byte)} bytes
            apart, so each needs a sector of its own, and 4 bytes of its 32 are used. The row-major read touches{' '}
            {nRow.transactions}. {v === nCol.transactions ? 'You had it.' : 'Check it with the column-major preset below.'}
          </p>
        )}
      </Predict>
      <p>
        Use the presets first, then change one control at a time. Step k and watch the totals under the readout: the
        model counts every instruction afresh.
      </p>
      <Simplification>
        <p>
          One warp, one instruction: the distinct aligned segments its lanes touch. No caches between instructions, no
          timing. Segment sizes are each vendor's documented unit; for Metal, where none is documented, the size is a
          choice you make.
        </p>
      </Simplification>
      <CoalescingExplorer />
      <Ask q={questionsS61['ch06/6.1/q-align']} label="Work it out" />
      <Ask q={questionsS61['ch06/6.1/q-stride2']} label="Work it out" />
      <Ask q={questionsS61['ch06/6.1/q-mread']} />

      <h2>Watch one instruction</h2>
      <p>
        The same warp at the same k, in both layouts, and then over the k loop. What a still picture flattens is that this
        happens to one instruction, again and again.
      </p>
      <CoalesceScene />

      <h2>Where column-major reads come from</h2>
      <p>
        In C code, a column-major read usually comes from a transpose. You need a matrix and its transpose, storing both
        wastes memory, so you store one and read the other by swapping the row and column indices, which is the same as
        reading the stored one as column-major.
      </p>
      <Source pages="128" />
      <p>
        You meet this every day. <code>nn.Linear</code> stores its weight as (out_features, in_features) and computes{' '}
        <code>y = x·Wᵀ + b</code>. Write a kernel that gives each thread one output feature <code>o</code> of one batch
        row, and its k loop reads <code>W[o*in + k]</code>: along a row of W, which looks fine for one thread. But the
        lanes of a warp have neighbouring <code>o</code>, so at every k they are <code>in</code> elements apart.
      </p>
      <Beyond>
        <p>
          The layout of <code>nn.Linear</code>'s weight comes from PyTorch's documentation and source
          (notes/today/ch03-facts.md D6). The one-thread-per-feature kernel is ours, for illustration.
        </p>
      </Beyond>
      <Ask q={questionsS61['ch06/6.1/q-linear']} />

      <h2>Three ways out</h2>
      <p>
        When the natural access does not coalesce, the book offers three remedies. Change which thread handles which
        data. Change the layout of the data, for example by storing the transposed copy once. Or bring the data into
        shared memory in a coalesced order and do the awkward access there, where order costs little. Section 6.8 returns
        to the first two; the third is corner turning.
      </p>
      <Source pages="129" />

      <h2>Corner turning</h2>
      <p>
        Take the tiled kernel of chapter 5 with M row-major and N column-major. The M tile loads as before. For the N
        tile, chapter 5's mapping has a row of threads read a row of the tile, and column-major storage scatters a row.
        Swap the roles of <code>threadIdx.x</code> and <code>threadIdx.y</code> in the N load and consecutive threads
        read down the tile's left edge instead: a column, which column-major storage keeps contiguous.
      </p>
      <Source section="6.1" pages="129–130" figs="Fig. 6.4" />
      <p>
        The figure uses four threads and 4 × 4 tiles to stay small. In real kernels, the book says, the tile is as wide
        as a warp, so the first 32 threads of a block read 32 consecutive elements down the left edge.
      </p>
      <Source pages="130" />
      <p>
        Once the tile is in shared memory it can be read in any order with little penalty: shared memory is SRAM and has
        no coalescing rule. It has a rule of its own, which the stepper hints at. In arrangement (b) each thread writes
        its own row of the shared tile, so the warp writes a column, and{' '}
        <a href={href('ch06', '6.4')}>section 6.4</a> is about what that costs.
      </p>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md, A6)">
        <p>
          NVIDIA's Best Practices Guide teaches the same two steps in its C = AAᵀ example: a warp reads a row of A into a
          column of a shared tile, then the tile is padded. It also says that, apart from bank conflicts, shared memory
          has no penalty for non-sequential or unaligned access.
        </p>
      </Today>
      <CornerTurnStepper />
      <Ask q={questionsS61['ch06/6.1/q-cornerturn']} label="Work it out" />
      <Ask q={questionsS61['ch06/6.1/q-ex3']} />

      <h2>Stores count too</h2>
      <p>
        Coalescing applies to stores as well, and the book says uncoalesced stores can hurt more than uncoalesced loads:
        they become partial writes that need a read-modify-write, which takes more bandwidth on GPUs with
        error-correcting memory (ECC) enabled.
      </p>
      <Source pages="130" />
      <Today checked="2026-11-20 (notes/today/ch06-facts.md, A5)">
        <p>
          NVIDIA: with ECC on GDDR memory, scattered accesses, writes especially, add overhead; HBM2&rsquo;s ECC costs
          far less. No source we read describes the read-modify-write, so treat it as unverified.
        </p>
      </Today>
      <p>
        Lab 6.1 has two naive transposes that differ only in which side is strided: one reads rows and writes columns,
        the other reads columns and writes rows. Commit to a number before you run it.
      </p>
      <Predict
        id="ch06/6.1/p-transpose"
        prompt={
          <>
            On your device, the time of the transpose with strided stores divided by the time of the one with strided
            loads. Above 1 means strided stores cost more.
          </>
        }
        unit="×"
      >
        {(v) => <TransposeReveal predicted={v} />}
      </Predict>
      <Ask q={questionsS61['ch06/6.1/q-store']} />

      <h2>A warp is a good carpool</h2>
      <p>
        The benefit is less traffic: several accesses merged into one request, which needs them at the same time and at
        neighbouring addresses.
      </p>
      <Source pages="130–132" figs="Figs. 6.5–6.6" />
      <p>
        The book's picture is carpooling. Requests are vehicles and data are passengers; when the request rate passes
        what the DRAM system can carry, the road jams and the arithmetic units wait. Carpooling needs people on the same
        schedule, and the lanes of a warp are ideal carpoolers: they issue the load at the same instant.
      </p>
      <Aside>
        <p>
          CPU caches are shaped by DRAM bursts too: a CPU cache line maps to part of a burst, one, or several, and code
          that uses every byte of the lines it touches runs much faster there as well. The techniques of this chapter
          carry over. (Book p. 132.)
        </p>
      </Aside>

      <h2>Drill: exercise 1</h2>
      <p>
        The book's first exercise is a kernel with nine memory accesses. Classify each one before you check. Line
        numbers are the book's.
      </p>
      <Source section="6.11" pages="154–155" />
      <Code
        tabs={[{ label: 'foo_kernel', code: FOO, tiers: FOO_TIERS, status: 'front-end-checked; not run' }]}
        caption={<>Exercise 1's kernel as printed, from <code>labs/ch06/snippets/s61_foo_kernel.cu</code>. It is a drill, not a useful kernel.</>}
      />
      <Exercise1Drill />
      <Ask q={questionsS61['ch06/6.1/q-ex1c']} />
      <Ask q={questionsS61['ch06/6.1/q-ex1d']} />
      <Ask q={questionsS61['ch06/6.1/q-ex1g']} />

      <h2>On other backends</h2>
      <p>
        AMD documents the same burst-based mechanism. Its coalescing hardware groups a wavefront's addresses into the
        fewest cache-line requests, with 64- or 128-byte lines. Wavefronts are 64 lanes on CDNA (Instinct) and 32 on RDNA
        (Radeon), so on CDNA 64 consecutive floats are {amd.bytesNeeded} bytes: {amd.transactions} lines of 128 bytes, not
        one. Set the explorer's warp width to 64 to see it.
      </p>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md B1; ch05-facts.md B5)">
        <p>
          AMD: “the hardware can combine all 64 thread requests into as few as 4-8 cache line requests (depending on data
          size and alignment)”, and widely separated addresses can cut effective bandwidth “by up to 16x or more”.
        </p>
      </Today>
      <p>
        Apple documents no coalescing rule, no line size and no sector size. Its advice is qualitative: arrange the data
        to fit the access pattern, and shape threadgroups to match the layout. On a Mac, lab 6.1 is the only way to
        know. Whatever the backend, do not write the warp width into your code: read <code>warpSize</code> on CUDA and
        HIP, and the pipeline's <code>threadExecutionWidth</code> on Metal.
      </p>
      <Today checked="2026-11-20 (notes/today/ch06-facts.md C1)">
        <p>
          Apple's WWDC22 talk on compute: requests from a SIMD-group are packed into cache lines, and most of a line can
          go unused; “Re-arrange the data to fit the access pattern better”. No numbers are given.
        </p>
      </Today>

      <h2>Your device</h2>
      <p>
        Lab 6.1 sweeps the stride of a read from 1 to 64 floats and times the two transposes against a plain copy. The
        lab page has the details: <a href={href('ch06', 'labs')}>labs</a>.
      </p>
      <YourDevice61 />
    </>
  );
}
